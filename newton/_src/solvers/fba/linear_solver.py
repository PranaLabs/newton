# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LDLᵀ + sparse-inverse linear solver for SolverFBA.

Setup phase (CPU):
    1. assemble scalar N*N PD Hessian ``A`` (caller-supplied).
    2. factor ``P A Pᵀ = L D Lᵀ`` with :func:`~.sparse_ldlt.factor_ldlt` (nested dissection, multifrontal).
    3. compute ``S = L⁻¹`` keeping elimination-tree sparsity.
    4. upload to device as Warp BSR matrices.

Runtime (on the device):
    ``A⁻¹ b = Sᵀ · D⁻¹ · (S · b)``, two ``bsr_mv`` calls plus diagonal scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
import warp.sparse as wps

from ...geometry import ParticleFlags
from ...sim import Model
from .kernels import (
    apply_permutation_scalar_kernel,
    extract_component_kernel,
    insert_component_kernel,
    scale_by_diag_kernel,
)
from .sparse_ldlt import CscMatrix, CsrMatrix, factor_ldlt


def particle_masses(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Masses [kg] and inverse masses [1/kg] of the particles, both zero for those the solver holds: the pinned
    ones, which have zero inverse mass, and the inactive ones."""
    mass = model.particle_mass.numpy().astype(np.float64)
    inv_mass = model.particle_inv_mass.numpy().astype(np.float64)
    held = (inv_mass == 0.0) | ((model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) == 0)
    mass[held] = 0.0
    inv_mass[held] = 0.0
    return mass, inv_mass


def build_pd_system(
    model: Model,
    dt: float,
    pin_stiffness: float,
    tri_weight_scale: float | np.ndarray = 1.0,
    tet_weight_scale: float | np.ndarray = 1.0,
) -> tuple[CsrMatrix, dict[str, Any]]:
    """Assemble the constant scalar PD Hessian ``A`` for cloth.

    Args:
        model: The Newton model containing cloth triangles and (optionally)
            bending edges. Pinned particles are detected via
            ``particle_inv_mass == 0``.
        dt: Time step [s]. Enters as ``m_i / dt^2`` on the diagonal.
        pin_stiffness: PD soft-pin weight ``w_pin``. Larger means harder pin.
        tri_weight_scale: Factor on each triangle's stretching weight: a scalar or one per triangle.
        tet_weight_scale: Factor on each tetrahedron's stretching weight: a scalar or one per tetrahedron.

    Returns:
        A tuple ``(A, meta)`` where ``A`` is an ``N x N`` symmetric PSD CSR
        matrix (``N = model.particle_count``) and ``meta`` contains derived
        per-element data needed at the runtime stage:

        - ``meta["tri_indices"]`` : ``np.ndarray[int32]`` shape (T, 3)
        - ``meta["tri_rest_inv"]`` : ``np.ndarray[float64]`` shape (T, 2, 2)
        - ``meta["tri_weight"]`` : ``np.ndarray[float64]`` shape (T,)  (= 2 * k_mu * area)
        - ``meta["edge_indices"]`` : ``np.ndarray[int32]`` shape (E, 4)
        - ``meta["edge_quad_q"]`` : ``np.ndarray[float64]`` shape (E, 4)  (k-coefs)
        - ``meta["edge_quad_scale"]`` : ``np.ndarray[float64]`` shape (E,)  (3/(A0+A1))
        - ``meta["edge_weight"]`` : ``np.ndarray[float64]`` shape (E,)
        - ``meta["pin_indices"]`` : ``np.ndarray[int32]`` (packed pinned indices)
    """
    N = model.particle_count
    mass, inv_mass = particle_masses(model)

    # COO triplet buffers, collected as a list of (rows, cols, vals) NumPy
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

        # The 2x2 rest-pose inverse of each triangle, shape (T, 2, 2).
        tri_rest_inv = model.tri_poses.numpy().astype(np.float64)

        tri_area = model.tri_areas.numpy().astype(np.float64)
        # tri_materials column 0 is k_mu, the second Lamé parameter [Pa]: PD weight 2 * k_mu * area.
        tri_materials = model.tri_materials.numpy().astype(np.float64)
        tri_weight = 2.0 * tri_materials[:, 0] * tri_area * tri_weight_scale

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
        tet_weight = 2.0 * mu_tet * tet_volume * tet_weight_scale

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

    A = CsrMatrix.from_triplets((N, N), rows, cols, vals)

    meta: dict[str, Any] = {
        "tri_indices": tri_indices,
        "tri_rest_inv": tri_rest_inv,
        "tri_weight": tri_weight,
        "edge_indices": edge_indices,
        "edge_quad_q": edge_quad_q,
        "edge_quad_scale": edge_quad_scale,
        "edge_weight": edge_weight,
        "pin_indices": pin_indices,
        "tet_indices": tet_indices,
        "tet_rest_inv": tet_rest_inv,
        "tet_volume": tet_volume,
        "tet_weight": tet_weight,
        "tet_mu": mu_tet,
    }
    return A, meta


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

    Computes column ``index`` of ``S = L^{-1}`` with the sparse-inverse
    recurrence of :func:`compute_lower_inverse`. The column-level data flow is:

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
    """Step 3 of :func:`compute_lower_inverse`, vectorized: the values of ``L`` aligned with the pattern of ``S``.

    For each row ``r`` we scan L's column ``invperm[r]`` and place each
    nonzero into the matching slot in S's column ``r`` (matched by row
    index). The vectorization uses ``np.searchsorted`` per column,
    correct because both ``S_innerInd[slice]`` and ``L_indices[slice]``
    enumerate the elimination-tree path of ``invperm[r]`` (the former in
    elim-tree order, the latter in sorted row-index order, both restricted
    to the same set when L is the unit lower-triangular factor of LDLᵀ).
    """
    aL = np.zeros(S_outerPtr[n], dtype=np.float64)
    # Sort S_innerInd within each column once so np.searchsorted is valid.
    # The pattern construction above stores them in elim-tree-path order
    # (monotonically increasing in original row index, since parent[j] > j).
    # That happens to be the same as sorted ascending, so we can search
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

        =====  ==========
        N      Setup time
        =====  ==========
        1089   0.02 s
        4225   0.3 s
        10201  1.0 s
        =====  ==========
    """
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
        device = wp.get_cuda_device() if wp.is_cuda_available() else wp.get_device("cpu")
    wp_device = wp.get_device(device) if isinstance(device, str) else device

    S_values = _compute_lower_inverse_column_warp(S_outerPtr, S_innerInd, perm, aL, wp_device)

    return CscMatrix((n, n), S_outerPtr, S_innerInd, S_values)


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
    and returns ``S_values`` as a host NumPy array.
    """
    n = int(S_outerPtr.size - 1)
    S_nnz = int(S_outerPtr[n])

    # Avoid extra astype copies: callers already provide correct dtypes.
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
    edge-endpoint angles.

    Implementation: fully vectorized over all edges with NumPy array math,
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

    S: CscMatrix  # n x n, lower-tri inverse with elimination-tree sparsity
    Dinv: np.ndarray  # length n
    perm_r: np.ndarray  # length n, int32
    invperm_r: np.ndarray  # length n, int32


def factorize_and_sparse_inverse(A, positions: np.ndarray | None = None) -> FactorizedSystem:
    """Factor an SPD matrix as ``P A Pᵀ = L D Lᵀ`` and compute ``S = L⁻¹``.

    Args:
        A: Scalar N x N PD matrix: a :class:`~.sparse_ldlt.CsrMatrix`, a dense array, or a matrix with ``tocoo()``.
        positions: Point of each unknown [m], shape ``[N, 3]``, for the nested dissection ordering; ``None``
            factors densely in the given order.

    Returns:
        FactorizedSystem with all data needed to construct an FBALinearSolver.
    """
    order, L, d = factor_ldlt(CsrMatrix.from_matrix(A), positions)
    S = compute_lower_inverse(L, parent=_elimination_tree(L))
    perm_r = np.argsort(order).astype(np.int32)
    return FactorizedSystem(S=S, Dinv=1.0 / d, perm_r=perm_r, invperm_r=order.astype(np.int32))


def _csc_to_bsr_1x1(M: CscMatrix, device, transpose: bool = False):
    """Build a Warp 1x1 BSR matrix (float64) from a CSC matrix or its transpose.

    Args:
        M: The matrix.
        device: Warp device to place the BSR matrix on.
        transpose: Build ``Mᵀ`` instead of ``M``.

    Returns:
        A :class:`warp.sparse.BsrMatrix` with ``block_type=wp.float64``.
    """
    rows, cols = M.indices, np.repeat(np.arange(M.shape[1]), np.diff(M.indptr))
    if transpose:
        rows, cols = cols, rows
    vals = M.data.astype(np.float64)
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

        # Build 1x1 BSR matrices from the sparse factors.
        self._S_bsr = _csc_to_bsr_1x1(factor.S, self.device)
        self._ST_bsr = _csc_to_bsr_1x1(factor.S, self.device, transpose=True)
        self._Dinv = wp.array(factor.Dinv.astype(np.float64), dtype=wp.float64, device=self.device)
        self._perm = wp.array(factor.perm_r.astype(np.int32), dtype=wp.int32, device=self.device)
        self._invperm = wp.array(factor.invperm_r.astype(np.int32), dtype=wp.int32, device=self.device)

        # Scalar scratch buffers, reused across components to avoid allocation.
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
            wp.launch(extract_component_kernel, dim=n, inputs=[b, c], outputs=[self._b_scalar], device=dev)
            self.solve_scalar(self._b_scalar, self._x_scalar)
            wp.launch(insert_component_kernel, dim=n, inputs=[self._x_scalar, c], outputs=[x], device=dev)

    def solve_scalar(self, b: wp.array[wp.float64], x: wp.array[wp.float64]) -> None:
        """Compute ``x = A⁻¹ · b`` for one scalar right-hand side.

        Args:
            b: Input right-hand side, shape ``[n]``. Not modified.
            x: Output solution, shape ``[n]``. Must not alias ``b``.
        """
        n = self.n
        dev = self.device
        # A⁻¹ b = P_r⁻¹ · Sᵀ · D⁻¹ · S · P_r · b: gather by invperm, two SpMVs
        # around the diagonal scale, then gather by perm.
        wp.launch(apply_permutation_scalar_kernel, dim=n, inputs=[b, self._invperm], outputs=[self._b_perm], device=dev)
        # Row i of S lists the descendants of i in the elimination tree, so the rows near the root span the
        # system and one thread per row serializes them; Warp picks tiles by the mean row length only, and
        # uses them only on CUDA.
        tile = 64 if dev.is_cuda else 0
        wps.bsr_mv(self._S_bsr, self._b_perm, self._Sb, tile_size=tile)
        wp.launch(scale_by_diag_kernel, dim=n, inputs=[self._Sb, self._Dinv], outputs=[self._DSb], device=dev)
        wps.bsr_mv(self._ST_bsr, self._DSb, self._SDSb, tile_size=tile)
        wp.launch(apply_permutation_scalar_kernel, dim=n, inputs=[self._SDSb, self._perm], outputs=[x], device=dev)
