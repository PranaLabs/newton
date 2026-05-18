# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cholesky + sparse-inverse linear solver for SolverFBA.

Setup phase (CPU):
    1. assemble scalar N*N PD Hessian ``A`` (caller-supplied).
    2. factor with ``scipy.sparse.linalg.splu`` (COLAMD ordering).
    3. compute ``S = L⁻¹`` keeping elimination-tree sparsity (ported from
       RealSim ``LDLT_computeLowerInverse``, SparseLDLT.cpp:225-347).
    4. upload to device as Warp BSR matrices.

Runtime (GPU):
    ``A⁻¹ b = Sᵀ · D⁻¹ · (S · b)`` — two ``bsr_mv`` calls plus diagonal scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
import warp.sparse as wps

if TYPE_CHECKING:
    import scipy.sparse as sp

from ...sim import Model
from .kernels import (
    accumulate_schur_W_kernel,
    apply_permutation_scalar_kernel,
    extract_component_kernel,
    insert_component_kernel,
    scale_by_diag_kernel,
)


def build_pd_system(
    model: Model,
    dt: float,
    pin_stiffness: float,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Assemble the constant scalar PD Hessian ``A`` for cloth.

    Args:
        model: The Newton model containing cloth triangles and (optionally)
            bending edges. Pinned particles are detected via
            ``particle_inv_mass == 0``.
        dt: Time step [s]. Enters as ``m_i / dt^2`` on the diagonal.
        pin_stiffness: PD soft-pin weight ``w_pin``. Larger means harder pin.

    Returns:
        A tuple ``(A, meta)`` where ``A`` is an ``N x N`` symmetric PSD CSR
        matrix (``N = model.particle_count``) and ``meta`` contains derived
        per-element data needed at the runtime stage:

        - ``meta["tri_indices"]`` : ``np.ndarray[int32]`` shape (T, 3)
        - ``meta["tri_rest_inv"]`` : ``np.ndarray[float64]`` shape (T, 2, 2)
        - ``meta["tri_area"]`` : ``np.ndarray[float64]`` shape (T,)
        - ``meta["tri_weight"]`` : ``np.ndarray[float64]`` shape (T,)  (= ke * area)
        - ``meta["edge_indices"]`` : ``np.ndarray[int32]`` shape (E, 4)
        - ``meta["edge_quad_q"]`` : ``np.ndarray[float64]`` shape (E, 4)  (k-coefs)
        - ``meta["edge_quad_scale"]`` : ``np.ndarray[float64]`` shape (E,)  (3/(A0+A1))
        - ``meta["edge_weight"]`` : ``np.ndarray[float64]`` shape (E,)
        - ``meta["pin_indices"]`` : ``np.ndarray[int32]`` (packed pinned indices)
        - ``meta["pin_weight"]`` : float
    """
    import scipy.sparse as _sp

    N = model.particle_count
    mass = model.particle_mass.numpy().astype(np.float64)
    inv_mass = model.particle_inv_mass.numpy().astype(np.float64)

    # COO triplet buffers — collected as a list of (rows, cols, vals) NumPy
    # arrays produced by each contribution (mass, pins, tris, bending), then
    # concatenated once at the end. This is dramatically faster than appending
    # Python scalars to lists for >1K particles.
    coo_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # ---- Mass diagonal ---- (only where inv_mass > 0; pinned masses go below)
    inv_dt2 = 1.0 / (dt * dt)
    free_mask = inv_mass > 0.0
    free_idx = np.where(free_mask)[0].astype(np.int32)
    if free_idx.size > 0:
        coo_parts.append((free_idx, free_idx, mass[free_idx] * inv_dt2))

    # ---- Pin diagonal ----
    pin_indices = np.where(inv_mass == 0.0)[0].astype(np.int32)
    if pin_indices.size > 0:
        coo_parts.append(
            (
                pin_indices,
                pin_indices,
                np.full(pin_indices.size, pin_stiffness, dtype=np.float64),
            )
        )

    # ---- Triangle stretching contribution ----
    tri_indices = np.zeros((0, 3), dtype=np.int32)
    tri_rest_inv = np.zeros((0, 2, 2), dtype=np.float64)
    tri_area = np.zeros((0,), dtype=np.float64)
    tri_weight = np.zeros((0,), dtype=np.float64)
    if model.tri_count > 0:
        tri_indices_flat = model.tri_indices.numpy().astype(np.int32)
        tri_indices = tri_indices_flat.reshape(-1, 3)

        # tri_poses stores the 2x2 rest-pose inverse per triangle.
        if model.tri_poses is not None:
            tri_rest_inv = model.tri_poses.numpy().astype(np.float64)
            # tri_poses is stored as wp.mat22; .numpy() returns shape (T, 2, 2).
        else:
            tri_rest_inv = _compute_rest_inv_from_positions(model)

        tri_area = model.tri_areas.numpy().astype(np.float64)
        # tri_materials column 0 is ke (area-weighted stretching stiffness).
        tri_materials = model.tri_materials.numpy().astype(np.float64)
        ke = tri_materials[:, 0]
        tri_weight = ke * tri_area

        # Vectorized 3x3 stencil scatter: G_t = ST @ Dm_inv_t, K_t = G_t @ G_t.T.
        # Then for each triangle we contribute w_t * K_t[a,b] to A[v_a, v_b]
        # for all (a, b) in 0..2. We assemble (T*9,) row/col/val arrays in one
        # broadcast and append a single COO part.
        # ST is the 3x2 reference-space gradient selector for the 3-vertex stencil.
        ST = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        # G_all: (T, 3, 2)
        G_all = np.einsum("ij,tjk->tik", ST, tri_rest_inv)
        # K_all: (T, 3, 3)
        K_all = np.einsum("tij,tkj->tik", G_all, G_all)
        # Scaled by per-triangle weight.
        vals_tri = (tri_weight[:, None, None] * K_all).reshape(-1)  # (T*9,)
        rows_tri = np.repeat(tri_indices, 3, axis=1).reshape(-1)  # (T*9,)
        cols_tri = np.tile(tri_indices, (1, 3)).reshape(-1)  # (T*9,)
        coo_parts.append((rows_tri.astype(np.int32), cols_tri.astype(np.int32), vals_tri))

    # ---- Bending edge contribution (4-vertex isometric stencil) ----
    edge_indices = np.zeros((0, 4), dtype=np.int32)
    edge_quad_q = np.zeros((0, 4), dtype=np.float64)
    edge_quad_scale = np.zeros((0,), dtype=np.float64)
    edge_weight = np.zeros((0,), dtype=np.float64)
    if model.edge_indices is not None:
        # Newton layout: [o0, o1, v1, v2] per row (o = opposite, v = shared edge).
        # o0 or o1 may be -1 for boundary edges (only one adjacent triangle).
        # Reorder interior edges to (v1, v2, o0, o1) = (v0, v1, v2, v3) for stencil.
        ei_flat = model.edge_indices.numpy().astype(np.int32)
        ei = ei_flat.reshape(-1, 4)
        # Keep only interior edges: both opposite vertices must be valid (>= 0).
        interior_mask = (ei[:, 0] >= 0) & (ei[:, 1] >= 0)
        ei_interior = ei[interior_mask]
        edge_indices = ei_interior[:, [2, 3, 0, 1]]

        # Compute cotangent k-coefficients from initial geometry.
        positions = model.particle_q.numpy().astype(np.float64)
        edge_quad_q, edge_quad_scale = _compute_isometric_bending_q(edge_indices, positions)

        if model.edge_bending_properties is not None:
            bend_props = model.edge_bending_properties.numpy().astype(np.float64)
            edge_weight = bend_props[interior_mask, 0]
        else:
            edge_weight = np.zeros(edge_indices.shape[0], dtype=np.float64)

        # Vectorized 4x4 edge scatter: A[v_a, v_b] += w_e * q_e[a] * q_e[b].
        # Build (E, 4, 4) outer-product block in one shot, then flatten.
        if edge_indices.shape[0] > 0:
            w_eff = edge_weight * edge_quad_scale  # (E,)
            # Outer product per edge: (E, 4, 4)
            qq = edge_quad_q[:, :, None] * edge_quad_q[:, None, :]
            block = w_eff[:, None, None] * qq
            vals_e = block.reshape(-1)  # (E*16,)
            rows_e = np.repeat(edge_indices, 4, axis=1).reshape(-1)  # (E*16,)
            cols_e = np.tile(edge_indices, (1, 4)).reshape(-1)  # (E*16,)
            # NB: zero-weight edges still contribute zeros; harmless to leave in
            # since CSR builder duplicates-sum, and they don't change A.
            coo_parts.append((rows_e.astype(np.int32), cols_e.astype(np.int32), vals_e))

    # ---- Tetrahedral stretching contribution ----
    tet_indices = np.zeros((0, 4), dtype=np.int32)
    tet_rest_inv = np.zeros((0, 3, 3), dtype=np.float64)
    tet_volume = np.zeros((0,), dtype=np.float64)
    tet_weight = np.zeros((0,), dtype=np.float64)
    mu_tet = np.zeros((0,), dtype=np.float64)
    if hasattr(model, "tet_count") and model.tet_count > 0:
        tet_indices_flat = model.tet_indices.numpy().astype(np.int32)
        tet_indices = tet_indices_flat.reshape(-1, 4)
        tet_rest_inv = model.tet_poses.numpy().astype(np.float64)
        # Volume from Dm_inv: det(Dm_inv) = 1/(6*V) => V = 1/(6*|det(Dm_inv)|)
        det_dm_inv = np.linalg.det(tet_rest_inv)
        tet_volume = 1.0 / (6.0 * np.abs(det_dm_inv) + 1.0e-30)
        tet_materials = model.tet_materials.numpy().astype(np.float64)
        mu_tet = tet_materials[:, 0]
        # PD weight: w = 2*mu * volume  (ARAP: singular values project to 1)
        tet_weight = 2.0 * mu_tet * tet_volume

        # Stencil selector ST is 4x3 (row = which edge coefficient for each tet vertex).
        ST_tet = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )  # (4, 3)
        # G_all[T, 4, 3] = ST @ Dm_inv per tet
        G_all = np.einsum("ab,tbc->tac", ST_tet, tet_rest_inv)  # (T, 4, 3)
        # K_all[T, 4, 4] = G @ G^T per tet (symmetric PSD)
        K_all = np.einsum("tac,tbc->tab", G_all, G_all)  # (T, 4, 4)
        wK_all = tet_weight[:, None, None] * K_all  # (T, 4, 4)

        # Vectorized scatter: for each (a,b) in 4x4, contribute wK[t,a,b] to A[tet[t,a], tet[t,b]]
        a_idx, b_idx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        a_flat = a_idx.flatten()  # 16
        b_flat = b_idx.flatten()  # 16
        # Gather all 16 (row, col, val) arrays, then concatenate once.
        tet_rows_parts = []
        tet_cols_parts = []
        tet_vals_parts = []
        for a, b in zip(a_flat, b_flat, strict=True):
            tet_rows_parts.append(tet_indices[:, a])
            tet_cols_parts.append(tet_indices[:, b])
            tet_vals_parts.append(wK_all[:, a, b])
        if len(tet_rows_parts) > 0:
            coo_parts.append(
                (
                    np.concatenate(tet_rows_parts).astype(np.int32),
                    np.concatenate(tet_cols_parts).astype(np.int32),
                    np.concatenate(tet_vals_parts),
                )
            )

    if coo_parts:
        rows = np.concatenate([p[0] for p in coo_parts])
        cols = np.concatenate([p[1] for p in coo_parts])
        vals = np.concatenate([p[2] for p in coo_parts])
    else:
        rows = np.zeros(0, dtype=np.int32)
        cols = np.zeros(0, dtype=np.int32)
        vals = np.zeros(0, dtype=np.float64)

    A = _sp.csr_matrix(
        (
            vals,
            (rows, cols),
        ),
        shape=(N, N),
    )

    meta: dict[str, Any] = {
        "tri_indices": tri_indices,
        "tri_rest_inv": tri_rest_inv,
        "tri_area": tri_area,
        "tri_weight": tri_weight,
        "edge_indices": edge_indices,
        "edge_quad_q": edge_quad_q,
        "edge_quad_scale": edge_quad_scale,
        "edge_weight": edge_weight,
        "pin_indices": pin_indices,
        "pin_weight": pin_stiffness,
        "tet_indices": tet_indices,
        "tet_rest_inv": tet_rest_inv,
        "tet_volume": tet_volume,
        "tet_weight": tet_weight,
        "tet_mu": mu_tet,
    }
    return A, meta


def _compute_rest_inv_from_positions(model: Model) -> np.ndarray:
    """Recompute the 2x2 rest-pose inverse per triangle from particle_q.

    Fallback used when ``model.tri_poses`` is not populated.
    """
    pos = model.particle_q.numpy().astype(np.float64)
    tri = model.tri_indices.numpy().astype(np.int32).reshape(-1, 3)
    out = np.zeros((tri.shape[0], 2, 2), dtype=np.float64)
    for t in range(tri.shape[0]):
        a, b, c = tri[t]
        e12 = pos[b] - pos[a]
        e13 = pos[c] - pos[a]
        n1 = e12 / max(np.linalg.norm(e12), 1e-20)
        e13_perp = e13 - np.dot(e13, n1) * n1
        n2 = e13_perp / max(np.linalg.norm(e13_perp), 1e-20)
        basis = np.stack([n1, n2], axis=1)  # 3x2
        edges = np.stack([e12, e13], axis=1)  # 3x2
        Dm = basis.T @ edges  # 2x2
        out[t] = np.linalg.inv(Dm)
    return out


def _splu_extract_factors(
    A,
) -> tuple:
    """Wrap scipy.sparse.linalg.splu and return (L_csc, U_csc, Dinv, perm_r, perm_c).

    For SPD A, splu effectively produces LU = L · U with U ≈ Dᵀ Lᵀ; we extract
    the lower factor L (with unit diagonal), the diagonal D = diag(U), and the
    permutation arrays. The factor satisfies P_r · A · P_cᵀ = L · D · Lᵀ
    (modulo numerical asymmetry of SuperLU's pivoting for non-symmetric input,
    which is unobservable on SPD input).

    Args:
        A: Sparse SPD matrix (CSC or CSR).

    Returns:
        Tuple (L_csc, U_csc, Dinv, perm_r, perm_c).
    """
    import scipy.sparse.linalg as spla

    lu = spla.splu(A.tocsc(), permc_spec="COLAMD")
    L = lu.L.tocsc()  # lower-tri with unit diagonal
    L.sort_indices()  # SuperLU may return unsorted column indices; sort for alignment
    U = lu.U.tocsc()
    Dinv = 1.0 / U.diagonal()
    perm_r = lu.perm_r.astype(np.int32)
    perm_c = lu.perm_c.astype(np.int32)
    return L, U, Dinv, perm_r, perm_c


def _elimination_tree(L) -> np.ndarray:
    """Build elimination tree parent[] from L's sparsity structure.

    parent[j] = smallest row index > j with L[parent[j], j] != 0, else -1.

    Args:
        L: Lower-triangular CSC sparse matrix.

    Returns:
        Integer array of length n with parent pointers.
    """
    n = L.shape[0]
    parent = -np.ones(n, dtype=np.int32)
    indptr = L.indptr
    indices = L.indices
    for j in range(n):
        for ptr in range(indptr[j], indptr[j + 1]):
            i = indices[ptr]
            if i > j:
                parent[j] = i
                break
    return parent


@wp.kernel
def _lower_inverse_column_kernel(
    S_outerPtr: wp.array[wp.int32],
    S_innerInd: wp.array[wp.int32],
    perm: wp.array[wp.int32],
    aL: wp.array[wp.float64],
    S_values: wp.array[wp.float64],
):
    """One thread per CSC column of S.

    Computes column ``index`` of ``S = L^{-1}`` using the sparse-inverse
    recurrence ported from RealSim ``LDLT_computeLowerInverse``
    (SparseLDLT.cpp:225-347). The column-level data flow is:

      - The thread writes only to ``S_values[outerPtr[index] : outerPtr[index+1]]``.
      - Inside the column it reads ``S_values[i-1]`` (within the same slice)
        and ``aL[k]`` (precomputed, read-only).

    There is NO cross-column ``S_values`` read, so columns are fully
    independent and can be processed in parallel without level scheduling.
    """
    index = wp.tid()
    start = S_outerPtr[index]
    end = S_outerPtr[index + 1]
    # Diagonal entry of S^{-1} (which is what S stores) at the column head.
    S_values[start] = wp.float64(1.0)

    for i in range(start + 1, end):
        row = S_innerInd[i - 1]
        col = perm[row]
        k_start = S_outerPtr[col] + 1
        k_end = S_outerPtr[col + 1]
        prev = S_values[i - 1]
        j = i
        k = k_start
        while j < end and k < k_end:
            S_values[j] = S_values[j] - aL[k] * prev
            j = j + 1
            k = k + 1


def _aligned_lower_vectorized(
    L_indptr: np.ndarray,
    L_indices: np.ndarray,
    L_data: np.ndarray,
    S_outerPtr: np.ndarray,
    S_innerInd: np.ndarray,
    invperm: np.ndarray,
    n: int,
) -> np.ndarray:
    """Vectorized port of step 3 (LDLT_computeAlignedLower).

    For each row ``r`` we scan L's column ``invperm[r]`` and place each
    nonzero into the matching slot in S's column ``r`` (matched by row
    index). The vectorization uses ``np.searchsorted`` per column —
    correct because both ``S_innerInd[slice]`` and ``L_indices[slice]``
    enumerate the elimination-tree path of ``invperm[r]`` (the former in
    elim-tree order, the latter in sorted row-index order, both restricted
    to the same set when L is the unit-lower Cholesky factor).
    """
    aL = np.zeros(S_outerPtr[n], dtype=np.float64)
    # Sort S_innerInd within each column once so np.searchsorted is valid.
    # The pattern construction above stores them in elim-tree-path order
    # (monotonically increasing in original row index, since parent[j] > j).
    # That happens to be the same as sorted ascending — so we can search
    # directly without per-row sorting.
    for r in range(n):
        invr = invperm[r]
        l_start = L_indptr[invr]
        l_end = L_indptr[invr + 1]
        if l_end == l_start:
            continue
        s_start = S_outerPtr[r]
        s_end = S_outerPtr[r + 1]
        # Subset of L's column-`invr` indices that lie in S's column-`r` pattern.
        l_rows = L_indices[l_start:l_end]
        s_rows = S_innerInd[s_start:s_end]
        # Place each L_data[ptrL] into aL at the slot matching L_indices[ptrL].
        # np.searchsorted is O((l + s) log s) per column; total O(S_nnz log).
        pos = np.searchsorted(s_rows, l_rows)
        # Only keep matches (S's pattern may be a subset of L's column).
        # Note: the elim-tree property guarantees s_rows ⊆ l_rows, but in
        # general L may have entries below the elim-tree path; mask them.
        valid = (pos < s_rows.size) & (s_rows[np.clip(pos, 0, s_rows.size - 1)] == l_rows)
        aL[s_start + pos[valid]] = L_data[l_start:l_end][valid]
    return aL


def compute_lower_inverse(
    L,
    parent: np.ndarray | None = None,
    invperm: np.ndarray | None = None,
    device: Any = None,
):
    """Compute S = L⁻¹ as a sparse CSC matrix.

    Ports RealSim ``LDLT_computeLowerInverse`` (SparseLDLT.cpp:225-347).
    Sparsity pattern of S is derived from L's elimination tree; this gives
    the exact sparse inverse (no thresholding, no fill-in).

    Args:
        L: Lower-triangular CSC matrix with unit diagonal.
        parent: Elimination tree parent array; computed from L if None.
        invperm: Inverse permutation (identity if None).
        device: Warp device to run the per-column kernel on. If ``None``,
            defaults to the current CUDA device when available, otherwise CPU.

    Returns:
        S as CSC; S · L ≈ I in the permuted basis.

    Performance note:
        Setup steps 1-3 (pattern + aligned-L) run on the host with vectorized
        NumPy. Step 4 (the column-by-column elimination, which dominates
        cost for large N) runs as a Warp kernel with one thread per column.
        Column independence is guaranteed by the algorithm: each thread
        writes only to its own column of ``S_values`` and reads only
        precomputed read-only data (``aL``, the CSC pattern, ``perm``)
        plus its own previously-written entries.

        Measured setup wall-time (build_pd_system + factorize_and_sparse_inverse,
        cloth grid, CUDA RTX 5090):

        =====  =====================  =====================
        N      pure Python (before)   Warp kernel (after)
        =====  =====================  =====================
        1089   8.8 s                  0.21 s
        4225   165 s                  0.26 s
        10201  ~16 min (extrapolated) 1.0 s
        =====  =====================  =====================
    """
    import scipy.sparse as _sp

    n = L.shape[0]
    if parent is None:
        parent = _elimination_tree(L)
    if invperm is None:
        invperm = np.arange(n, dtype=np.int32)
    perm = np.argsort(invperm).astype(np.int32)

    # --- 1 + 2. Build S's CSC pattern in one vectorized pass. ---
    # We walk each elim-tree path on the CPU but skip per-element Python
    # list growth: paths are written directly into S_innerInd, and column
    # offsets accumulate from per-column path lengths.
    S_outerPtr, S_innerInd = _build_inverse_pattern(parent, invperm, n)

    # --- 3. Compute aligned L values (vectorized over rows). ---
    aL = _aligned_lower_vectorized(
        L.indptr.astype(np.int64),
        L.indices.astype(np.int32),
        L.data.astype(np.float64),
        S_outerPtr,
        S_innerInd,
        invperm,
        n,
    )

    # --- 4. Solve for S column-by-column (Warp kernel). ---
    # Decide where to run. For tiny problems (n < 256) the CPU fallback is
    # often faster than CUDA dispatch + transfer overhead.
    if device is None:
        try:
            device = wp.get_cuda_device()
        except Exception:  # pragma: no cover - CPU-only environments
            device = wp.get_device("cpu")
    wp_device = wp.get_device(device) if isinstance(device, str) else device

    S_values = _compute_lower_inverse_column_warp(S_outerPtr, S_innerInd, perm, aL, wp_device)

    return _sp.csc_matrix((S_values, S_innerInd, S_outerPtr), shape=(n, n))


def _build_inverse_pattern(parent: np.ndarray, invperm: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (S_outerPtr, S_innerInd) for S = L⁻¹ given elim-tree parent.

    Vectorized over all columns simultaneously: at depth ``d``, the
    "frontier" array stores the depth-d ancestor of ``invperm[i]`` for
    each column ``i``. We advance the frontier by one parent step each
    iteration, scattering live entries into the flat ``S_innerInd``
    output via the column's running write-position. The loop terminates
    when every entry reaches the root (-1).

    Runtime: O(max_depth * n) vector ops, fully NumPy-vectorized.
    Memory: 2 dense (n,) arrays plus the output. No per-column Python.
    """
    parent_np = np.asarray(parent, dtype=np.int32)
    invperm_np = np.asarray(invperm, dtype=np.int32)

    # First, compute path lengths via a vectorized depth walk.
    # frontier[i] starts at invperm[i]; at each step we move to its parent
    # (where it isn't already -1). Path length = number of non-root nodes
    # visited (i.e., iterations where the entry hadn't yet hit -1).
    frontier = invperm_np.copy()
    path_len = np.zeros(n, dtype=np.int32)
    # Iterate until all frontier entries become -1. Each iteration is O(n).
    while True:
        alive = frontier != -1
        if not alive.any():
            break
        path_len[alive] += 1
        # Advance: frontier[i] -> parent[frontier[i]], guarded by alive.
        frontier = np.where(alive, parent_np[np.where(alive, frontier, 0)], -1)

    S_outerPtr = np.empty(n + 1, dtype=np.int32)
    S_outerPtr[0] = 0
    np.cumsum(path_len, out=S_outerPtr[1:])
    S_nnz = int(S_outerPtr[n])
    S_innerInd = np.empty(S_nnz, dtype=np.int32)

    # Second pass: replay the depth walk, scattering ancestors into the
    # right slots. write_pos[i] = where to write the next ancestor for col i.
    frontier = invperm_np.copy()
    write_pos = S_outerPtr[:n].copy()  # one per column, starts at column start
    while True:
        alive = frontier != -1
        if not alive.any():
            break
        # Write current frontier value at write_pos[i] for alive entries.
        cols_alive = np.where(alive)[0]
        S_innerInd[write_pos[cols_alive]] = frontier[cols_alive]
        write_pos[cols_alive] += 1
        # Advance frontier.
        frontier = np.where(alive, parent_np[np.where(alive, frontier, 0)], -1)

    return S_outerPtr, S_innerInd


def _compute_lower_inverse_column_warp(
    S_outerPtr: np.ndarray,
    S_innerInd: np.ndarray,
    perm: np.ndarray,
    aL: np.ndarray,
    wp_device,
) -> np.ndarray:
    """Run :func:`_lower_inverse_column_kernel` on a Warp device.

    Lays out all inputs as Warp arrays, dispatches one thread per S column,
    and returns ``S_values`` as a host NumPy array (consumed by SciPy CSC).
    """
    n = int(S_outerPtr.size - 1)
    S_nnz = int(S_outerPtr[n])

    # Avoid extra astype copies — callers already provide correct dtypes.
    S_outerPtr_d = wp.array(np.ascontiguousarray(S_outerPtr, dtype=np.int32), dtype=wp.int32, device=wp_device)
    S_innerInd_d = wp.array(np.ascontiguousarray(S_innerInd, dtype=np.int32), dtype=wp.int32, device=wp_device)
    perm_d = wp.array(np.ascontiguousarray(perm, dtype=np.int32), dtype=wp.int32, device=wp_device)
    aL_d = wp.array(np.ascontiguousarray(aL, dtype=np.float64), dtype=wp.float64, device=wp_device)
    S_values_d = wp.zeros(S_nnz, dtype=wp.float64, device=wp_device)

    wp.launch(
        _lower_inverse_column_kernel,
        dim=n,
        inputs=[S_outerPtr_d, S_innerInd_d, perm_d, aL_d, S_values_d],
        device=wp_device,
    )

    return S_values_d.numpy()


def build_particle_element_csr(
    element_indices: np.ndarray,
    n_particles: int,
    n_verts_per_element: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CSR adjacency from elements (tri / tet / edge stencil) to particles.

    For each particle ``p``, the entries in ``element_idx[offsets[p]:offsets[p+1]]``
    name the elements incident on ``p``, with the corresponding entry in
    ``local_vertex_idx`` giving which of the element's local vertices is ``p``.
    Used by SolverFBA's particle-centered PD reduction (atomic-free, deterministic).

    Args:
        element_indices: ``(num_elements, n_verts_per_element)`` particle indices.
        n_particles: Total particle count in the model.
        n_verts_per_element: 3 for triangles, 4 for tetrahedra or 4-vertex
            bending stencils.

    Returns:
        ``(offsets, element_idx, local_vertex_idx)``:

        - ``offsets``: ``(n_particles + 1,) int32`` such that
          ``offsets[p+1] - offsets[p]`` is the number of elements incident to ``p``.
        - ``element_idx``: ``(total_incidence,) int32`` element index per entry,
          where ``total_incidence == num_elements * n_verts_per_element``.
        - ``local_vertex_idx``: ``(total_incidence,) int32`` local-vertex index in
          ``[0, n_verts_per_element)`` for each entry.
    """
    num_elements = int(element_indices.shape[0])
    if num_elements == 0:
        return (
            np.zeros(n_particles + 1, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
    particles = element_indices.reshape(-1).astype(np.int64)
    elements = np.repeat(np.arange(num_elements, dtype=np.int64), n_verts_per_element)
    locals_ = np.tile(np.arange(n_verts_per_element, dtype=np.int64), num_elements)
    order = np.argsort(particles, kind="stable")
    particles = particles[order]
    elements = elements[order]
    locals_ = locals_[order]
    counts = np.bincount(particles, minlength=n_particles).astype(np.int32)
    offsets = np.zeros(n_particles + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    return offsets, elements.astype(np.int32), locals_.astype(np.int32)


def _compute_isometric_bending_q(edge_indices: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge length-4 vector q and per-edge scale ``3 / (A0 + A1)``.

    Bergou et al. 2006 isometric bending: for stencil (v0, v1, v2, v3) where
    (v0, v1) is the shared edge and (v2, v3) the opposite vertices of the two
    adjacent triangles, the per-edge bending Hessian contribution is
    ``(3 / (A0 + A1)) · q·qᵀ`` with q given by cotangents at all four
    edge-endpoint angles. See PDIsometricBendingEnergy.cpp::init().

    Implementation: fully vectorized over all edges with NumPy array math —
    no per-edge Python loop. Heron's formula and cotangent identities are
    evaluated as elementwise array ops.

    Args:
        edge_indices: Shape (E, 4), columns are (v0, v1, v2, v3).
        positions: Shape (N, 3) particle rest positions.

    Returns:
        A tuple ``(q, scale)`` where ``q`` has shape (E, 4) and ``scale`` has
        shape (E,) with per-edge values ``3 / (A0 + A1)``.
    """
    E = edge_indices.shape[0]
    if E == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    x0 = positions[edge_indices[:, 0]]
    x1 = positions[edge_indices[:, 1]]
    x2 = positions[edge_indices[:, 2]]
    x3 = positions[edge_indices[:, 3]]

    l01 = np.linalg.norm(x1 - x0, axis=1)
    l02 = np.linalg.norm(x2 - x0, axis=1)
    l12 = np.linalg.norm(x2 - x1, axis=1)
    l03 = np.linalg.norm(x3 - x0, axis=1)
    l13 = np.linalg.norm(x3 - x1, axis=1)

    # Heron's formula for triangle areas (clip negatives from FP roundoff).
    r0 = 0.5 * (l01 + l02 + l12)
    A0 = np.sqrt(np.maximum(r0 * (r0 - l01) * (r0 - l02) * (r0 - l12), 0.0))
    r1 = 0.5 * (l01 + l03 + l13)
    A1 = np.sqrt(np.maximum(r1 * (r1 - l01) * (r1 - l03) * (r1 - l13), 0.0))

    safe_A0 = np.maximum(A0, 1e-20)
    safe_A1 = np.maximum(A1, 1e-20)
    l01_sq = l01 * l01
    l02_sq = l02 * l02
    l12_sq = l12 * l12
    l03_sq = l03 * l03
    l13_sq = l13 * l13

    cot02 = (l01_sq - l02_sq + l12_sq) / (4.0 * safe_A0)
    cot12 = (l01_sq + l02_sq - l12_sq) / (4.0 * safe_A0)
    cot03 = (l01_sq - l03_sq + l13_sq) / (4.0 * safe_A1)
    cot13 = (l01_sq + l03_sq - l13_sq) / (4.0 * safe_A1)

    q = np.stack(
        [
            cot02 + cot03,
            cot12 + cot13,
            -(cot02 + cot12),
            -(cot03 + cot13),
        ],
        axis=1,
    )  # (E, 4)
    scale = 3.0 / np.maximum(A0 + A1, 1e-20)
    return q, scale


@dataclass
class FactorizedSystem:
    """Output of factorize_and_sparse_inverse."""

    S: sp.csc_matrix  # n x n, lower-tri inverse with elimination-tree sparsity
    ST: sp.csc_matrix  # transpose, runtime-uploaded separately
    Dinv: np.ndarray  # length n
    perm_r: np.ndarray  # length n, int32
    invperm_r: np.ndarray  # length n, int32


def factorize_and_sparse_inverse(A) -> FactorizedSystem:
    """Run COLAMD ordering + LU factor + sparse inverse on an SPD matrix.

    Args:
        A: scalar N x N PD Hessian (csr or csc, will be converted internally).

    Returns:
        FactorizedSystem with all data needed to construct an FBALinearSolver.
    """
    L, _U, Dinv, perm_r, _perm_c = _splu_extract_factors(A)
    parent = _elimination_tree(L)
    invperm_r = np.argsort(perm_r).astype(np.int32)
    S = compute_lower_inverse(L, parent=parent)
    ST = S.T.tocsc()
    return FactorizedSystem(S=S, ST=ST, Dinv=Dinv, perm_r=perm_r, invperm_r=invperm_r)


def _csc_to_bsr_1x1(M, device):
    """Build a Warp 1x1 BSR matrix (float64) from a SciPy CSC/CSR matrix.

    Args:
        M: SciPy sparse matrix (any format, converted to CSR internally).
        device: Warp device to place the BSR matrix on.

    Returns:
        A :class:`warp.sparse.BsrMatrix` with ``block_type=wp.float64``.
    """
    M_csr = M.tocsr()
    rows, cols = M_csr.nonzero()
    vals = M_csr.data.astype(np.float64)
    rows_wp = wp.array(rows.astype(np.int32), dtype=wp.int32, device=device)
    cols_wp = wp.array(cols.astype(np.int32), dtype=wp.int32, device=device)
    vals_wp = wp.array(vals, dtype=wp.float64, device=device)
    return wps.bsr_from_triplets(M.shape[0], M.shape[1], rows_wp, cols_wp, vals_wp)


class FBALinearSolver:
    """Runtime sparse-inverse solver ``x = A⁻¹ b`` via two BSR SpMVs per component.

    Implements ``A⁻¹ b = P_cᵀ · Sᵀ · D⁻¹ · S · P_r · b`` where ``S`` is the
    sparse lower-triangular inverse and ``P_r`` is the row permutation from
    :func:`factorize_and_sparse_inverse`.

    Args:
        factor: Output of :func:`factorize_and_sparse_inverse`.
        device: Warp device string or device object for GPU execution.
    """

    def __init__(self, factor: FactorizedSystem, device) -> None:

        self.device = wp.get_device(device) if isinstance(device, str) else device
        n = factor.S.shape[0]
        self.n = n

        # Build 1x1 BSR matrices from SciPy sparse factors.
        self._S_bsr = _csc_to_bsr_1x1(factor.S, self.device)
        self._ST_bsr = _csc_to_bsr_1x1(factor.ST, self.device)
        self._Dinv = wp.array(factor.Dinv.astype(np.float64), dtype=wp.float64, device=self.device)
        self._perm = wp.array(factor.perm_r.astype(np.int32), dtype=wp.int32, device=self.device)
        self._invperm = wp.array(factor.invperm_r.astype(np.int32), dtype=wp.int32, device=self.device)

        # Scalar scratch buffers — reused across components to avoid allocation.
        self._b_scalar = wp.empty(n, dtype=wp.float64, device=self.device)
        self._b_perm = wp.empty(n, dtype=wp.float64, device=self.device)
        self._Sb = wp.empty(n, dtype=wp.float64, device=self.device)
        self._DSb = wp.empty(n, dtype=wp.float64, device=self.device)
        self._SDSb = wp.empty(n, dtype=wp.float64, device=self.device)
        self._x_scalar = wp.empty(n, dtype=wp.float64, device=self.device)

    def solve(self, b: wp.array[wp.vec3], x: wp.array[wp.vec3]) -> None:
        """Compute ``x[i] = A⁻¹ · b[i]`` component-wise.

        Args:
            b: Input right-hand side, shape ``[n]``, dtype ``wp.vec3``.
            x: Output solution, shape ``[n]``, dtype ``wp.vec3``.
        """
        n = self.n
        dev = self.device
        for c in range(3):
            # 1. Extract scalar component c from b.
            wp.launch(extract_component_kernel, dim=n, inputs=[b, c], outputs=[self._b_scalar], device=dev)
            # 2. Apply inverse row permutation: b_perm[i] = b_scalar[invperm[i]].
            #    The formula is A⁻¹ b = P_r⁻¹ · Sᵀ · D⁻¹ · S · P_r · b (see docstring).
            #    The first permutation gathers b by invperm (un-applies the row reordering).
            wp.launch(
                apply_permutation_scalar_kernel,
                dim=n,
                inputs=[self._b_scalar, self._invperm],
                outputs=[self._b_perm],
                device=dev,
            )
            # 3. Multiply by S (lower-triangular inverse): Sb = S * b_perm.
            wps.bsr_mv(self._S_bsr, self._b_perm, self._Sb)
            # 4. Scale by D⁻¹: DSb[i] = Dinv[i] * Sb[i].
            wp.launch(scale_by_diag_kernel, dim=n, inputs=[self._Sb, self._Dinv], outputs=[self._DSb], device=dev)
            # 5. Multiply by Sᵀ: SDSb = Sᵀ * DSb.
            wps.bsr_mv(self._ST_bsr, self._DSb, self._SDSb)
            # 6. Apply row permutation: x_scalar[i] = SDSb[perm[i]].
            #    The second permutation gathers by perm_r (re-applies the row ordering).
            wp.launch(
                apply_permutation_scalar_kernel,
                dim=n,
                inputs=[self._SDSb, self._perm],
                outputs=[self._x_scalar],
                device=dev,
            )
            # 7. Insert result into component c of output x.
            wp.launch(insert_component_kernel, dim=n, inputs=[self._x_scalar, c], outputs=[x], device=dev)

    def _ensure_multi_rhs_buffers(self, R: int) -> None:
        """Allocate or grow multi-RHS scratch buffers to hold R right-hand sides."""
        n = self.n
        dev = self.device
        if hasattr(self, "_multi_rhs_cap") and self._multi_rhs_cap >= R:
            return
        # Grow by at least 1.5x to amortize repeated allocation.
        new_cap = max(R, int(getattr(self, "_multi_rhs_cap", 0) * 1.5) + 1)
        self._multi_rhs_cap = new_cap
        self._b_perm_multi = wp.empty(shape=(new_cap, n), dtype=wp.float64, device=dev)
        self._Sb_multi = wp.empty(shape=(new_cap, n), dtype=wp.float64, device=dev)
        self._DSb_multi = wp.empty(shape=(new_cap, n), dtype=wp.float64, device=dev)
        self._SDSb_multi = wp.empty(shape=(new_cap, n), dtype=wp.float64, device=dev)

    def solve_multi_rhs_scalar(
        self,
        b_multi: wp.array2d,
        x_multi: wp.array2d,
        R: int,
    ) -> None:
        """Apply A^{-1} to R scalar RHS rows in a single batched pass.

        Each row of ``b_multi`` is one scalar RHS (length N). Each row of
        ``x_multi`` receives the corresponding scalar solution A^{-1} b[r].

        Pipeline (same as :meth:`solve` but batched over all R rows):

        1. ``b_perm[r, i] = b_multi[r, invperm[i]]``  (multi-RHS invperm gather)
        2. ``Sb[r, i]     = sum_j S[i, j] * b_perm[r, j]``  (multi-RHS SpMV with S)
        3. ``DSb[r, i]    = Dinv[i] * Sb[r, i]``  (multi-RHS diag scale)
        4. ``SDSb[r, i]   = sum_j S^T[i, j] * DSb[r, j]``  (multi-RHS SpMV with S^T)
        5. ``x_multi[r, i] = SDSb[r, perm[i]]``  (multi-RHS perm gather)

        Args:
            b_multi: Input RHS rows, shape ``(≥R, N)`` float64.
            x_multi: Output solution rows, shape ``(≥R, N)`` float64; written in-place.
            R: Number of active RHS rows to process (must be ≤ b_multi.shape[0]).
        """
        from .kernels import (  # noqa: PLC0415
            apply_permutation_multi_rhs_kernel,
            bsr_mv_multi_rhs_scalar_kernel,
            scale_by_diag_multi_rhs_kernel,
        )

        self._ensure_multi_rhs_buffers(R)
        n = self.n
        dev = self.device

        # View active sub-slices (pointer into pre-allocated buffer, no copy).
        b_perm_v = wp.array(
            ptr=self._b_perm_multi.ptr,
            dtype=wp.float64,
            shape=(R, n),
            device=dev,
        )
        Sb_v = wp.array(ptr=self._Sb_multi.ptr, dtype=wp.float64, shape=(R, n), device=dev)
        DSb_v = wp.array(ptr=self._DSb_multi.ptr, dtype=wp.float64, shape=(R, n), device=dev)
        SDSb_v = wp.array(ptr=self._SDSb_multi.ptr, dtype=wp.float64, shape=(R, n), device=dev)

        # Step 1: b_perm[r, i] = b_multi[r, invperm[i]]
        wp.launch(
            apply_permutation_multi_rhs_kernel,
            dim=(R, n),
            inputs=[b_multi, self._invperm],
            outputs=[b_perm_v],
            device=dev,
        )

        # Step 2: Sb[r, i] = sum_j S[i,j] * b_perm[r, j]
        wp.launch(
            bsr_mv_multi_rhs_scalar_kernel,
            dim=(R, n),
            inputs=[
                self._S_bsr.offsets,
                self._S_bsr.columns,
                self._S_bsr.values,
                b_perm_v,
            ],
            outputs=[Sb_v],
            device=dev,
        )

        # Step 3: DSb[r, i] = Dinv[i] * Sb[r, i]
        wp.launch(
            scale_by_diag_multi_rhs_kernel,
            dim=(R, n),
            inputs=[Sb_v, self._Dinv],
            outputs=[DSb_v],
            device=dev,
        )

        # Step 4: SDSb[r, i] = sum_j S^T[i,j] * DSb[r, j]
        wp.launch(
            bsr_mv_multi_rhs_scalar_kernel,
            dim=(R, n),
            inputs=[
                self._ST_bsr.offsets,
                self._ST_bsr.columns,
                self._ST_bsr.values,
                DSb_v,
            ],
            outputs=[SDSb_v],
            device=dev,
        )

        # Step 5: x_multi[r, i] = SDSb[r, perm[i]]
        wp.launch(
            apply_permutation_multi_rhs_kernel,
            dim=(R, n),
            inputs=[SDSb_v, self._perm],
            outputs=[x_multi],
            device=dev,
        )

    def build_schur_complement(
        self,
        num_contacts: int,
        j_indices: wp.array,
        j_normals: wp.array,
        j_alpha: wp.array,
        j_tangent1: wp.array | None = None,
        j_tangent2: wp.array | None = None,
        use_isodof: bool = True,
        download: bool = True,
    ) -> np.ndarray | None:
        """Build ``W = J · A⁻¹ · Jᵀ`` as a dense NumPy array (batched GPU implementation).

        Stage A (``j_tangent1`` and ``j_tangent2`` are ``None``): emits an ``(M, M)``
        matrix with one row per contact (normal direction only).

        Stage B (both tangent arrays provided): emits a ``(3M, 3M)`` matrix with
        three rows per contact ordered ``[n_c, t1_c, t2_c]`` for c = 0..M-1.
        Off-diagonal coupling between contacts is fully included.

        Two implementations are supported:

        - ``use_isodof=True`` (default, Task P): computes only the selected
            ``A^{-1}[i, j]`` entries for ``i, j`` in the unique-particle set of
            ``j_indices`` (the "isolated DOFs" / isodofs).  ``W`` is then assembled
            from ``Wi``, the per-particle direction, and the contact alpha entirely
            on device.  This avoids the ``M`` multi-RHS Cholesky solves entirely.
            The ``_A_inv_Jt_d`` buffer is *not* populated on this path; callers that
            need the lambda correction must use the isodof-aware
            :meth:`apply_lambda_correction_isodof` helper.

        - ``use_isodof=False`` (regression / fallback): the previous Approach B
            multi-RHS solve. Computes ``A^{-1} J^T`` for all ``3 * M_rows`` columns
            and assembles ``W`` from those columns.  The ``_A_inv_Jt_d`` device
            buffer of shape ``(M_rows, N)`` is left populated and reused by
            :meth:`apply_lambda_correction_combined`.

        Args:
            num_contacts: Number of active contacts ``M``.
            j_indices: Particle index per contact, shape ``[M]``, dtype int32.
            j_normals: World-frame contact normal per contact, shape ``[M]``, dtype vec3.
            j_alpha: Jacobian coefficient per contact, shape ``[M]``, dtype float32.
            j_tangent1: First tangent direction per contact, shape ``[M]``, dtype vec3.
                When ``None``, Stage A behavior (M x M W).
            j_tangent2: Second tangent direction per contact, shape ``[M]``, dtype vec3.
                Must be provided together with ``j_tangent1``.
            use_isodof: If ``True`` (default) use the isodof-restricted build
                that skips the multi-RHS solve.  If ``False``, use the legacy
                multi-RHS path (kept for regression testing).
            download: If ``True`` (default) pull the result back as a NumPy
                array.  Set ``False`` from callers that consume ``W`` directly
                on device via :meth:`W_device_view` — this skips a multi-MB
                device-to-host copy that dominates the Schur build cost on
                large contact sets (see Demo 5).  When ``False`` the method
                returns ``None``; the device-side ``W`` remains accessible
                through :meth:`W_device_view`.

        Returns:
            Dense float64 NumPy array of shape ``(M, M)`` (Stage A) or
            ``(3M, 3M)`` (Stage B) when ``download=True``; otherwise ``None``.
        """
        from .kernels import (  # noqa: PLC0415
            compose_W_from_wi_kernel,
            compute_wi_tiled_kernel,
            densify_S_iso_scaled_kernel,
            pack_jacobian_3axis_kernel,
            unpack_to_A_inv_Jt_3axis_kernel,
        )

        has_friction = j_tangent1 is not None and j_tangent2 is not None
        rows_per_contact = 3 if has_friction else 1
        M = num_contacts
        total_rows = M * rows_per_contact
        rhs_total = 3 * total_rows
        n = self.n
        dev = self.device

        # --- Build unified per-row direction and particle arrays on device ---
        # For Stage A: total_rows == M, each row uses normal direction.
        # For Stage B: total_rows == 3M, rows interleaved [n_c, t1_c, t2_c].
        # ``j_*`` arrays may be over-allocated (capacity > M); slice to active range.
        idx_np = j_indices.numpy()[:M]  # (M,) int32 — small, needed for row dir array
        n_np = j_normals.numpy()[:M]  # (M, 3) float32
        a_np = j_alpha.numpy()[:M]  # (M,) float32
        if has_friction:
            t1_np = j_tangent1.numpy()[:M]  # (M, 3) float32
            t2_np = j_tangent2.numpy()[:M]  # (M, 3) float32

        # Build (total_rows,) arrays for particle index, direction, alpha.
        row_particle = np.empty(total_rows, dtype=np.int32)
        row_dir = np.empty((total_rows, 3), dtype=np.float32)
        row_alpha = np.empty(total_rows, dtype=np.float32)
        if has_friction:
            # Stage B: 3 rows per contact, interleaved (n, t1, t2).
            row_particle[0::3] = idx_np
            row_particle[1::3] = idx_np
            row_particle[2::3] = idx_np
            row_dir[0::3] = n_np
            row_dir[1::3] = t1_np
            row_dir[2::3] = t2_np
            row_alpha[0::3] = a_np
            row_alpha[1::3] = a_np
            row_alpha[2::3] = a_np
        else:
            row_particle[:] = idx_np
            row_dir[:] = n_np
            row_alpha[:] = a_np

        # Upload direction arrays to device.
        row_particle_d = wp.array(row_particle, dtype=wp.int32, device=dev)
        row_dir_d = wp.array(row_dir, dtype=wp.vec3, device=dev)
        row_alpha_d = wp.array(row_alpha, dtype=wp.float32, device=dev)

        # --- Allocate/reuse W device buffer (used by both paths). ---
        # Grow with a 1.5x headroom factor so a slow contact-count climb does
        # not realloc every frame.  ``M`` can spike by 10s of contacts between
        # frames in Demo 5 (squeezing), and a (cap, cap) dense float64 buffer
        # is the dominant device alloc on the FBA hot path.
        if (
            not hasattr(self, "_W_device_d")
            or self._W_device_d.shape[0] < total_rows
            or self._W_device_d.shape[1] < total_rows
        ):
            prev = getattr(self, "_W_device_d", None)
            prev_cap = prev.shape[0] if prev is not None else 0
            cap = max(total_rows, int(prev_cap * 1.5) + 1)
            self._W_device_d = wp.empty(shape=(cap, cap), dtype=wp.float64, device=dev)
        # Active row count for the populated sub-block — consumed by the NSN
        # GPU drivers to build a (M, M) wp.array view of ``_W_device_d`` without
        # touching the host.
        self._W_device_active_rows = total_rows

        # Stash per-row contact metadata for the isodof lambda-correction path.
        # (Used by :meth:`apply_lambda_correction_isodof`; harmless on the
        # legacy path which has its own ``_A_inv_Jt_d`` cache.)
        self._row_particle_d = row_particle_d
        self._row_dir_d = row_dir_d
        self._row_alpha_d = row_alpha_d
        self._row_total = total_rows

        # Inverse mapping (particle → incident contact rows) for the
        # deterministic ``J^T·lambda`` gather in
        # :meth:`apply_lambda_correction_isodof`. Each contact row maps to
        # exactly one particle (Stage A: 1 row/contact; Stage B: 3 rows/contact
        # all sharing one particle), so a stable-sort by particle yields a
        # CSR adjacency. Built once per contact-set update — host-side cost is
        # ``O(total_rows log total_rows)`` and ``total_rows`` is small (~1k).
        order = np.argsort(row_particle, kind="stable").astype(np.int32)
        sorted_particles = row_particle[order]
        counts = np.bincount(sorted_particles, minlength=n).astype(np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        self._isodof_row_offsets_d = wp.array(offsets, dtype=wp.int32, device=dev)
        self._isodof_row_indices_d = wp.array(order, dtype=wp.int32, device=dev)

        # =====================================================================
        # Isodof-restricted path (Task P): build ``Wi`` of size (k, k) for the
        # unique contacted particles, then assemble W via a 2D kernel reading
        # ``Wi[isodof_rank(p_cp), isodof_rank(p_c)]``. Skips the multi-RHS solve
        # entirely. ``_A_inv_Jt_d`` is intentionally NOT populated here — the
        # solver_fba lambda correction uses the isodof helper instead.
        # =====================================================================
        if use_isodof:
            return self._build_schur_isodof(
                row_particle,
                row_particle_d,
                row_dir_d,
                row_alpha_d,
                total_rows,
                compute_wi_tiled_kernel,
                densify_S_iso_scaled_kernel,
                compose_W_from_wi_kernel,
                download=download,
            )

        # =====================================================================
        # Legacy multi-RHS path: compute ``A^{-1} J^T`` for all ``3*total_rows``
        # columns, then assemble W via ``accumulate_schur_W_kernel``. Kept for
        # regression testing and as a fallback when the isodof path is disabled.
        # =====================================================================

        # --- Allocate/reuse A_inv_Jt buffer: shape (total_rows, N) on device ---
        if not hasattr(self, "_A_inv_Jt_d") or self._A_inv_Jt_d.shape[0] < total_rows or self._A_inv_Jt_d.shape[1] != n:
            self._A_inv_Jt_d = wp.empty(shape=(total_rows, n), dtype=wp.vec3, device=dev)

        # --- Allocate/reuse 2D float64 buffers for batched 3-axis multi-RHS pack/solve ---
        # Buffers must hold 3 * total_rows RHS rows (one block per spatial axis,
        # concatenated row-major).
        if not hasattr(self, "_b_multi_d") or self._b_multi_d.shape[0] < rhs_total or self._b_multi_d.shape[1] != n:
            cap = max(rhs_total, int(getattr(self, "_b_multi_cap", 0) * 1.5) + 1)
            self._b_multi_cap = cap
            self._b_multi_d = wp.zeros(shape=(cap, n), dtype=wp.float64, device=dev)
            self._y_multi_d = wp.zeros(shape=(cap, n), dtype=wp.float64, device=dev)

        # Active sub-views (pointer into pre-allocated buffer, no copy).
        b_multi_v = wp.array(
            ptr=self._b_multi_d.ptr,
            dtype=wp.float64,
            shape=(rhs_total, n),
            device=dev,
        )
        y_multi_v = wp.array(
            ptr=self._y_multi_d.ptr,
            dtype=wp.float64,
            shape=(rhs_total, n),
            device=dev,
        )

        # --- Batched 3-axis pass: single multi-RHS solve for all 3 spatial axes ---
        # Pack J^T for all 3 axes into one contiguous (3*total_rows, N) buffer, run
        # a single ``solve_multi_rhs_scalar`` over R = 3*total_rows RHS, then unpack
        # all 3 axes back into ``A_inv_Jt`` with one kernel.
        # Total kernel launches: 1 pack + 5 solve + 1 unpack = 7 launches,
        # vs 3 * 7 = 21 launches for the prior per-axis loop.
        self._ensure_multi_rhs_buffers(rhs_total)
        # Zero b_multi for all 3 axis blocks.
        b_multi_v.zero_()
        # Pack J^T axes: b_multi[axis*total_rows + r, particle[r]] = alpha[r] * dir[r][axis].
        wp.launch(
            pack_jacobian_3axis_kernel,
            dim=rhs_total,
            inputs=[total_rows, row_particle_d, row_dir_d, row_alpha_d],
            outputs=[b_multi_v],
            device=dev,
        )
        # Batched solve: y_multi = A^{-1} * b_multi for all 3*total_rows RHS at once.
        self.solve_multi_rhs_scalar(b_multi_v, y_multi_v, rhs_total)
        # Unpack: A_inv_Jt[r, i] = (y_multi[r, i], y_multi[total_rows+r, i], y_multi[2*total_rows+r, i]).
        wp.launch(
            unpack_to_A_inv_Jt_3axis_kernel,
            dim=(total_rows, n),
            inputs=[total_rows, y_multi_v, self._A_inv_Jt_d],
            device=dev,
        )

        # ``accumulate_schur_W_kernel`` writes ``W[c', c] = ...`` (pure assign,
        # not accumulate), so a pre-launch zero is unnecessary. Skipping the
        # host -> device zero blast saves ``total_rows^2 * 8`` bytes of upload
        # per build.

        # Launch 2D kernel: one thread per (c_prime, c) entry.
        wp.launch(
            accumulate_schur_W_kernel,
            dim=(total_rows, total_rows),
            inputs=[row_particle_d, row_dir_d, row_alpha_d, self._A_inv_Jt_d, self._W_device_d],
            device=dev,
        )

        # Single host pull for the (total_rows x total_rows) W matrix.
        # Kept for callers that still want a numpy view (CPU NSN reference,
        # regression tests).  The GPU NSN drivers consume the device-resident
        # ``_W_device_d`` directly via :meth:`W_device_view`, avoiding the
        # per-PD-iter host -> device round trip.
        if not download:
            return None
        W = self._W_device_d.numpy()[:total_rows, :total_rows].copy()

        return W

    def _build_schur_isodof(
        self,
        row_particle: np.ndarray,
        row_particle_d: wp.array,
        row_dir_d: wp.array,
        row_alpha_d: wp.array,
        total_rows: int,
        compute_wi_tiled_kernel,
        densify_S_iso_scaled_kernel,
        compose_W_from_wi_kernel,
        download: bool = True,
    ) -> np.ndarray | None:
        """Isodof-restricted Schur build.

        Mirrors RealSim ``addHAinvHT_gpu`` (CUDASparseInverseSolver.cpp:215-339):
        we collect the unique contacted DOFs ``isodofs``, compute the dense
        ``k x k`` matrix ``Wi[a, b] = A^{-1}[isodofs[a], isodofs[b]]`` via a
        column-intersection kernel over the sparse-inverse factor ``S``, and
        then assemble ``W[c', c]`` from per-contact ``(particle, dir, alpha)``
        triples and an ``isodof_rank`` lookup.

        The legacy multi-RHS solve over ``A`` is skipped entirely — for sparse
        meshes where ``k^2 * nnz_col(S) << M * nnz(S)``, this is the dominant
        speedup ("Task P", ~RealSim parity).

        ``Wi`` is computed via blocked dense ``X^T X`` GEMM (``wp.tile_matmul``):
        densify the selected columns of ``S`` into a padded ``(N_pad, k_pad)``
        buffer with ``sqrt(D_inv)`` baked into the values, then run a tiled
        SYRK-style matmul.  This replaces the prior two-pointer
        ``compute_wi_kernel`` which was O(k² · nnz_col) and dominated the Schur
        build on Demo 5 (Squeezing Ball) where ``nnz_col(S) ≈ 1700``.
        """
        N = self.n
        dev = self.device

        # --- Build isodofs (unique sorted particle indices) on host ---
        # For Stage A: row_particle has one entry per contact.
        # For Stage B: each contact contributes 3 identical particle entries
        # (n/t1/t2 share a particle), so dedup gives the same set.
        # M is small (~1000), so np.unique is cheap.
        isodofs_h = np.unique(row_particle.astype(np.int64)).astype(np.int32)
        k = int(isodofs_h.size)

        # --- Build per-particle rank table (-1 for non-isodof particles) ---
        # We use a dense ``[N]`` int32 array so the composition kernel can do
        # ``rank = isodof_rank[particle]`` in O(1). Memory: 4*N bytes (e.g.
        # 21 KB for N = 5325). Allocated on host then uploaded.
        isodof_rank_h = np.full(N, -1, dtype=np.int32)
        isodof_rank_h[isodofs_h] = np.arange(k, dtype=np.int32)

        # --- Build perm[isodofs[a]] for the Wi kernel ---
        # The runtime solve computes ``A^{-1} = P · S^T · D^{-1} · S · P^T``
        # (see :meth:`solve`), where ``P[i, k] = δ_{k, perm[i]}``. Therefore
        # ``A^{-1}[i, j] = M[perm[i], perm[j]]`` with ``M = S^T D^{-1} S``,
        # which gives
        #     Wi[a, b] = sum_k S[k, perm[isodofs[a]]] * Dinv[k]
        #                      * S[k, perm[isodofs[b]]].
        # We precompute ``perm[isodofs[a]]`` per isodof. Using ``invperm`` here
        # silently builds the wrong matrix (different particles map under the
        # two permutations) and produces NaN downstream once contacts engage.
        perm_h = self._perm.numpy()
        isodof_perm_h = perm_h[isodofs_h].astype(np.int32)

        # --- Upload to device (allocate / reuse buffers) ---
        # ``Wi`` and ``S_iso_scaled`` are padded to multiples of the GEMM tile
        # dimensions so the inner-loop ``wp.tile_matmul`` can step in fixed
        # ``_WI_TILE_*`` strides without per-iteration bound checks.
        from .kernels import _WI_TILE_K, _WI_TILE_M, _WI_TILE_N, _WI_TILE_THREADS  # noqa: PLC0415

        tile_m = int(_WI_TILE_M)
        tile_n = int(_WI_TILE_N)
        tile_k = int(_WI_TILE_K)
        k_pad = ((k + tile_m - 1) // tile_m) * tile_m
        if k_pad == 0:
            # No contacts touch any particle — nothing to compute.  Caller
            # treats an empty ``W`` view as a zero matrix.
            return None if not download else np.zeros((total_rows, total_rows), dtype=np.float64)
        N_pad = ((N + tile_k - 1) // tile_k) * tile_k

        # Wi buffer: padded ``(k_pad, k_pad)`` so the tiled GEMM writes a
        # contiguous block.  The cap-based reuse grows by 1.5x to amortize
        # reallocation when ``k`` climbs over a step.
        if not hasattr(self, "_isodofs_cap") or self._isodofs_cap < k_pad:
            cap_iso = max(k_pad, int(getattr(self, "_isodofs_cap", 0) * 1.5) + 1)
            # Round cap up to a multiple of the tile size so ``Wi_v`` slicing
            # below stays tile-aligned across resizes.
            cap_iso = ((cap_iso + tile_m - 1) // tile_m) * tile_m
            self._isodofs_cap = cap_iso
            self._isodof_perm_d = wp.empty(cap_iso, dtype=wp.int32, device=dev)
            self._Wi_device_d = wp.empty(shape=(cap_iso, cap_iso), dtype=wp.float64, device=dev)
        # Densified ``S_iso_scaled`` buffer: padded ``(N_pad, k_pad)``.
        # Width grows with a 1.5x headroom factor so a contact-count climb does
        # not realloc every frame; height is fixed at ``N_pad`` (set by the
        # immutable mesh DOF count).
        if (
            not hasattr(self, "_S_iso_scaled_d")
            or self._S_iso_scaled_d.shape[0] < N_pad
            or self._S_iso_scaled_d.shape[1] < k_pad
        ):
            cur_k = self._S_iso_scaled_d.shape[1] if hasattr(self, "_S_iso_scaled_d") else 0
            cap_k = max(k_pad, int(cur_k * 1.5) + 1)
            cap_k = ((cap_k + tile_m - 1) // tile_m) * tile_m
            self._S_iso_scaled_d = wp.empty(shape=(N_pad, cap_k), dtype=wp.float64, device=dev)
        if not hasattr(self, "_isodof_rank_d") or self._isodof_rank_d.shape[0] != N:
            self._isodof_rank_d = wp.empty(N, dtype=wp.int32, device=dev)
        # Upload host-prepared isodof arrays.
        self._isodof_perm_d.assign(isodof_perm_h)
        self._isodof_rank_d.assign(isodof_rank_h)

        # Active sub-views of the Wi / S_iso_scaled buffers, sized to the
        # padded extents the tiled GEMM expects.  ``Wi`` is materialized as
        # a contiguous ``(k_pad, k_pad)`` block because the buffer was
        # allocated at the same padded size (no stride trickery needed).
        Wi_cap = int(self._Wi_device_d.shape[0])
        if Wi_cap == k_pad:
            Wi_v = self._Wi_device_d
        else:
            # Cap is larger than k_pad — slice a top-left ``(k_pad, k_pad)``
            # view with explicit strides so tile_load reads the correct
            # row stride (== Wi_cap, not k_pad).
            Wi_v = wp.array(
                ptr=self._Wi_device_d.ptr,
                dtype=wp.float64,
                shape=(k_pad, k_pad),
                strides=(Wi_cap * 8, 8),
                device=dev,
            )
        S_iso_cap_n = int(self._S_iso_scaled_d.shape[0])
        S_iso_cap_k = int(self._S_iso_scaled_d.shape[1])
        if S_iso_cap_n == N_pad and S_iso_cap_k == k_pad:
            S_iso_v = self._S_iso_scaled_d
        else:
            S_iso_v = wp.array(
                ptr=self._S_iso_scaled_d.ptr,
                dtype=wp.float64,
                shape=(N_pad, k_pad),
                strides=(S_iso_cap_k * 8, 8),
                device=dev,
            )

        # --- Compute Wi = X^T · X where X = sqrt(D_inv) * S_iso ---
        # Step 1: densify selected columns of S into ``S_iso_scaled`` with the
        # sqrt(D_inv) factor baked in.  The buffer is zeroed first so padding
        # rows/cols and untouched (zero) entries of S contribute zero to the
        # subsequent dot products.
        S_iso_v.zero_()
        wp.launch(
            densify_S_iso_scaled_kernel,
            dim=k,
            inputs=[
                self._ST_bsr.offsets,
                self._ST_bsr.columns,
                self._ST_bsr.values,
                self._Dinv,
                self._isodof_perm_d,
            ],
            outputs=[S_iso_v],
            device=dev,
        )

        # Step 2: tiled dense GEMM ``Wi = S_iso_scaled^T · S_iso_scaled``.
        wp.launch_tiled(
            compute_wi_tiled_kernel,
            dim=(k_pad // tile_m, k_pad // tile_n),
            inputs=[S_iso_v, Wi_v, N_pad],
            block_dim=_WI_TILE_THREADS,
            device=dev,
        )

        # --- Compose W[c', c] = alpha[c']*alpha[c] * dot(dir[c'], dir[c]) * Wi[r', r] ---
        # The (total_rows x total_rows) W is written densely; one thread per
        # (c', c) pair. ``isodof_rank[N]`` provides the rank lookup per particle.
        wp.launch(
            compose_W_from_wi_kernel,
            dim=(total_rows, total_rows),
            inputs=[
                row_particle_d,
                row_dir_d,
                row_alpha_d,
                self._isodof_rank_d,
                Wi_v,
            ],
            outputs=[self._W_device_d],
            device=dev,
        )

        # Single host pull for the (total_rows x total_rows) W matrix.
        # Skipped when callers (e.g. the GPU NSN driver) consume ``W`` via
        # :meth:`W_device_view` — a multi-MB copy of a padded dense buffer
        # otherwise dominates the Schur build on Demo 5.
        if not download:
            return None
        W = self._W_device_d.numpy()[:total_rows, :total_rows].copy()
        return W

    def W_device_view(self) -> wp.array2d[wp.float64]:
        """Return a ``(M, M)`` device view of the most recently built Schur W.

        The view aliases :attr:`_W_device_d` for the active row block populated
        by the most recent :meth:`build_schur_complement` call.  Because the
        underlying buffer has capacity ``(cap, cap)`` (with ``cap >= M`` and
        ``cap`` grown by the 1.5x headroom factor), the view is non-contiguous
        in its outer dimension and explicit strides matching the parent buffer
        are passed; without them the default contiguous strides would mis-index
        every row beyond the first.

        Returns:
            ``wp.array2d[wp.float64]`` of shape ``(M, M)`` sharing storage with
            the persistent device buffer, or ``None`` if no W exists.
        """
        if not hasattr(self, "_W_device_d") or not hasattr(self, "_W_device_active_rows"):
            return None
        m = int(self._W_device_active_rows)
        if m == 0:
            return None
        cap = int(self._W_device_d.shape[0])
        dtype_size = 8  # wp.float64
        return wp.array(
            ptr=self._W_device_d.ptr,
            dtype=wp.float64,
            shape=(m, m),
            strides=(cap * dtype_size, dtype_size),
            device=self.device,
        )

    def apply_lambda_correction_isodof(
        self,
        lam: np.ndarray,
        out: wp.array,
    ) -> None:
        """Compute ``correction = A^{-1} J^T lambda`` via a single Cholesky solve.

        For the isodof path, ``_A_inv_Jt_d`` is *not* populated — instead we
        rebuild ``J^T lambda`` (a vec3 array of length ``N``, summed over
        contact rows by a particle-centered gather over the inverse-mapping
        CSR built in :meth:`build_schur_complement`) and call :meth:`solve`
        once. This is much cheaper than the multi-RHS dot accumulation when
        the contact set is small relative to the total DOF count, and the
        gather avoids ``wp.atomic_add`` so the assembly is bit-deterministic.

        Args:
            lam: Contact impulse vector, shape ``(total_rows,)``, float64.
                ``total_rows == M`` for Stage A, ``3 * M`` for Stage B.
            out: Output correction, shape ``[N]``, vec3. Overwritten.
        """
        from .kernels import gather_jt_lambda_kernel  # noqa: PLC0415

        if not hasattr(self, "_row_particle_d") or not hasattr(self, "_isodof_row_offsets_d"):
            raise RuntimeError(
                "apply_lambda_correction_isodof requires a prior isodof-mode "
                "build_schur_complement call (which sets _row_*_d / _isodof_row_*_d arrays)."
            )

        n = self.n
        dev = self.device
        n_rows = self._row_total

        if not hasattr(self, "_jt_lambda_d") or self._jt_lambda_d.shape[0] != n:
            self._jt_lambda_d = wp.empty(n, dtype=wp.vec3, device=dev)
        if not hasattr(self, "_lam_d") or self._lam_d.shape[0] < n_rows:
            cap = max(n_rows, int(getattr(self, "_lam_cap", 0) * 1.5) + 1)
            self._lam_cap = cap
            self._lam_d = wp.empty(cap, dtype=wp.float32, device=dev)

        self._lam_d.assign(lam.astype(np.float32))

        # Particle-centered gather: out[p] = sum over its incident contact rows.
        # Overwrites _jt_lambda_d (no zero-init needed; kernel writes every p).
        wp.launch(
            gather_jt_lambda_kernel,
            dim=n,
            inputs=[
                self._isodof_row_offsets_d,
                self._isodof_row_indices_d,
                self._row_dir_d,
                self._row_alpha_d,
                self._lam_d,
            ],
            outputs=[self._jt_lambda_d],
            device=dev,
        )

        # Single Cholesky solve: out = A^{-1} * (J^T lambda).
        self.solve(self._jt_lambda_d, out)

    def apply_lambda_correction_combined(
        self,
        lam: np.ndarray | wp.array,
        rhs_inout: wp.array[wp.vec3],
        x_out: wp.array[wp.vec3],
    ) -> None:
        """Compute ``x_out = A⁻¹ · (rhs_inout + Jᵀ · lam)`` with in-place rhs update.

        Implements RealSim's combined-form constraint correction from
        ``NonSmoothNewton.cpp:151-167`` (``applyConstraintCorrection``):

            b += dt² · Jᵀ · (ω · λ)
            _systemlinearsolver->solve(x, b)

        Used by :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA.step` to
        replace the prior split-form ``x_cur += A⁻¹ · Jᵀ · lam_apply``. The
        combined form is algebraically equivalent but bit-distinct under
        float32 rounding, matching RealSim's single ``A⁻¹`` solve over the
        already-incremented ``b`` vector.

        The implementation:

        1. Gathers ``Jᵀ · lam`` into the cached ``_jt_lambda_d`` buffer (same
           particle-centered gather as :meth:`apply_lambda_correction_isodof`).
        2. Accumulates that into ``rhs_inout`` in-place (``rhs_inout[p] +=
           (Jᵀ·lam)[p]``).
        3. Runs a single Cholesky solve ``x_out = A⁻¹ · rhs_inout``.

        Both the isodof and legacy non-isodof Schur build paths populate the
        per-row metadata (``_row_*_d``, ``_isodof_row_*_d``) needed for the
        gather, so this helper works under both modes.

        Args:
            lam: Contact impulse vector, shape ``(total_rows,)``. Accepts
                either a host ``numpy`` array (legacy CPU path) or a
                device-resident ``wp.array[wp.float64]`` produced by the
                NSN inner driver. ``total_rows == M`` for Stage A,
                ``3 * M`` for Stage B.
            rhs_inout: Right-hand side vec3 array of length ``N`` —
                MUTATED in place to ``rhs_inout + Jᵀ·lam``.
            x_out: Output position correction, shape ``[N]``, vec3.
                Overwritten with ``A⁻¹ · (rhs_inout + Jᵀ·lam)``.
        """
        from .kernels import accumulate_vec3_kernel, cast_fp64_to_fp32_kernel, gather_jt_lambda_kernel  # noqa: PLC0415

        if not hasattr(self, "_row_particle_d") or not hasattr(self, "_isodof_row_offsets_d"):
            raise RuntimeError(
                "apply_lambda_correction_combined requires a prior "
                "build_schur_complement call (which sets _row_*_d / "
                "_isodof_row_*_d arrays)."
            )

        n = self.n
        dev = self.device
        n_rows = self._row_total

        if not hasattr(self, "_jt_lambda_d") or self._jt_lambda_d.shape[0] != n:
            self._jt_lambda_d = wp.empty(n, dtype=wp.vec3, device=dev)
        if not hasattr(self, "_lam_d") or self._lam_d.shape[0] < n_rows:
            cap = max(n_rows, int(getattr(self, "_lam_cap", 0) * 1.5) + 1)
            self._lam_cap = cap
            self._lam_d = wp.empty(cap, dtype=wp.float32, device=dev)

        # Accept either a host numpy array (legacy path used by tests and
        # the CPU NSN driver) or a device-resident fp64 array (the hot
        # solver path emits ``_nsn_lam_apply_d`` directly to avoid the
        # per-PD-iter device->host->device round-trip).
        if isinstance(lam, wp.array):
            wp.launch(
                cast_fp64_to_fp32_kernel,
                dim=n_rows,
                inputs=[lam],
                outputs=[self._lam_d],
                device=dev,
            )
        else:
            self._lam_d.assign(lam.astype(np.float32))

        # Step 1: particle-centered gather Jᵀ·lam into _jt_lambda_d.
        wp.launch(
            gather_jt_lambda_kernel,
            dim=n,
            inputs=[
                self._isodof_row_offsets_d,
                self._isodof_row_indices_d,
                self._row_dir_d,
                self._row_alpha_d,
                self._lam_d,
            ],
            outputs=[self._jt_lambda_d],
            device=dev,
        )

        # Step 2: rhs_inout += Jᵀ·lam  (RealSim's b += dt²·J·(ω·λ); the dt²·ω
        # weighting is folded into ``lam`` by the caller, see
        # ``_solve_nsn_*`` which returns ``lam_apply = dt²·ω·λ``).
        wp.launch(
            accumulate_vec3_kernel,
            dim=n,
            inputs=[self._jt_lambda_d],
            outputs=[rhs_inout],
            device=dev,
        )

        # Step 3: single Cholesky solve x_out = A⁻¹ · rhs_inout.
        self.solve(rhs_inout, x_out)
