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
    host_head,
    insert_component_kernel,
    lite_row_dirs_kernel,
    lite_row_slots_kernel,
    scale_by_diag_kernel,
)
from .sparse_ldlt import CscMatrix, CsrMatrix, factor_ldlt


def dense_schur_capacity(required: int, current: int) -> int:
    """Rows of a square dense Schur buffer that holds ``required`` rows and replaces one of ``current`` rows.

    Below 1024 rows the buffer grows by half again, so that a slowly climbing contact count does not reallocate it at
    every step. Larger buffers grow by 5%, in multiples of 128 rows: half again per side would more than double the
    memory of the largest allocation of the step, at the moment both buffers are held.
    """
    if max(required, current) < 1024:
        return max(required, int(current * 1.5) + 1)
    capacity = max(required, (105 * current + 99) // 100)
    return ((capacity + 127) // 128) * 128


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
    rayleigh_alpha: np.ndarray | None = None,
    rayleigh_beta: np.ndarray | None = None,
) -> tuple[CsrMatrix, dict[str, Any]]:
    """Assemble the constant scalar PD Hessian ``A`` for cloth.

    With Rayleigh damping, ``A`` also holds ``C / dt`` for the damping matrix ``C = alpha M + beta K``, where ``K``
    is the stretching and bending part of ``A``: implicit Euler applies the damping force ``-C v`` with
    ``v = (x - x_prev) / dt`` as ``C / dt`` in the matrix and ``C x_prev / dt`` in the right-hand side of each step.

    Args:
        model: The Newton model containing cloth triangles and (optionally)
            bending edges. Pinned particles are detected via
            ``particle_inv_mass == 0``.
        dt: Time step [s]. Enters as ``m_i / dt^2`` on the diagonal.
        pin_stiffness: PD soft-pin weight ``w_pin``. Larger means harder pin.
        tri_weight_scale: Factor on each triangle's stretching weight: a scalar or one per triangle.
        tet_weight_scale: Factor on each tetrahedron's stretching weight: a scalar or one per tetrahedron.
        rayleigh_alpha: Mass-proportional damping coefficient of each particle [1/s], or ``None`` for none.
        rayleigh_beta: Stiffness-proportional damping coefficient of each particle [s], or ``None`` for none. An
            element takes the mean of its particles' coefficients.

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
        - ``meta["damping"]`` : :class:`CsrMatrix` of ``C``, or ``None`` without damping
    """
    N = model.particle_count
    mass, inv_mass = particle_masses(model)

    # COO triplet buffers, collected as a list of (rows, cols, vals) NumPy
    # arrays produced by each contribution (mass, pins, tris, bending), then
    # concatenated once at the end. This is dramatically faster than appending
    # Python scalars to lists for >1K particles.
    coo_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    # The damping matrix C in the same form.
    damping_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def element_beta(indices: np.ndarray) -> np.ndarray:
        return rayleigh_beta[indices].mean(axis=1)

    # ---- Mass diagonal ---- (only where inv_mass > 0; pinned masses go below)
    inv_dt2 = 1.0 / (dt * dt)
    free_mask = inv_mass > 0.0
    free_idx = np.where(free_mask)[0].astype(np.int32)
    if free_idx.size > 0:
        coo_parts.append((free_idx, free_idx, mass[free_idx] * inv_dt2))
        if rayleigh_alpha is not None:
            damping_parts.append((free_idx, free_idx, rayleigh_alpha[free_idx] * mass[free_idx]))

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
        if rayleigh_beta is not None:
            beta_vals = (element_beta(tri_indices)[:, None, None] * tri_weight[:, None, None] * K_all).reshape(-1)
            damping_parts.append((rows_tri.astype(np.int32), cols_tri.astype(np.int32), beta_vals))

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
            if rayleigh_beta is not None:
                beta_vals = (element_beta(edge_indices)[:, None, None] * block).reshape(-1)
                damping_parts.append((rows_e.astype(np.int32), cols_e.astype(np.int32), beta_vals))

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
            if rayleigh_beta is not None:
                beta_tet = element_beta(tet_indices)
                damping_parts.append(
                    (
                        coo_parts[-1][0],
                        coo_parts[-1][1],
                        np.concatenate([beta_tet * wK_all[:, a, b] for a, b in zip(a_flat, b_flat, strict=True)]),
                    )
                )

    # ---- Implicit Rayleigh damping: C / dt ----
    damping = None
    if damping_parts:
        rows_d, cols_d, vals_d = (np.concatenate(a) for a in zip(*damping_parts, strict=True))
        damping = CsrMatrix.from_triplets((N, N), rows_d, cols_d, vals_d)
        coo_parts.append((rows_d, cols_d, vals_d / dt))

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
        "damping": damping,
    }
    return A, meta


def build_rigid_system(model: Model, bodies: np.ndarray, dt: float) -> CsrMatrix:
    """Assemble the constant rigid-body block of the PD system matrix.

    Body ``bodies[k]`` owns DOFs ``6k .. 6k+5``: the translation ``chi`` and the
    rotation vector ``rho``, both expressed in the body's pinned frame. Its block
    is ``diag(m·I₃, I) / dt²`` with ``I`` the inertia about the COM in the body
    frame, so the matrix stays constant while the body rotates and is factored
    once like the deformable block.

    Args:
        model: The Newton model providing ``body_mass`` and ``body_inertia``.
        bodies: Model body indices that own DOFs, shape ``[m]``.
        dt: Time step [s].

    Returns:
        The ``6m x 6m`` symmetric positive definite CSR matrix.

    Raises:
        ValueError: If a body has non-positive mass or an inertia tensor that is
            not positive definite.
    """
    bodies = np.asarray(bodies, dtype=np.int32)
    m = bodies.size
    if m == 0:
        return CsrMatrix.from_triplets((0, 0), [], [], [])
    mass = model.body_mass.numpy()[bodies].astype(np.float64)
    inertia = model.body_inertia.numpy()[bodies].astype(np.float64)
    bad = (mass <= 0.0) | (np.linalg.eigvalsh(inertia)[:, 0] <= 0.0)
    if np.any(bad):
        raise ValueError(f"Bodies {bodies[bad].tolist()} need positive mass and a positive definite inertia tensor.")

    blocks = np.zeros((m, 6, 6), dtype=np.float64)
    blocks[:, :3, :3] = mass[:, None, None] * np.eye(3)
    blocks[:, 3:, 3:] = inertia
    blocks /= dt * dt
    body, row, col = np.nonzero(blocks)
    return CsrMatrix.from_triplets((6 * m, 6 * m), 6 * body + row, 6 * body + col, blocks[body, row, col])


def rigid_block_inverses(A_rigid: CsrMatrix) -> np.ndarray:
    """Inverse of each 6x6 diagonal block of the rigid system matrix, shape [m, 6, 6].

    Raises:
        ValueError: If ``A_rigid`` is not block diagonal in 6x6 blocks.
    """
    A_rigid = CsrMatrix.from_matrix(A_rigid)
    rows, cols = A_rigid.row_of_entries(), A_rigid.indices
    if np.any(rows // 6 != cols // 6):
        raise ValueError("The rigid system matrix must be block diagonal")
    blocks = np.zeros((A_rigid.shape[0] // 6, 6, 6))
    blocks[rows // 6, rows % 6, cols % 6] = A_rigid.data
    return np.linalg.inv(blocks)


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
        # Slots of contact rows on several particles: (particles, coefficients, CSR offsets, CSR entries).
        self._row_slots = None

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

    def build_schur_complement(
        self,
        num_contacts: int,
        j_indices: wp.array,
        j_normals: wp.array,
        j_alpha: wp.array,
        j_tangent1: wp.array | None = None,
        j_tangent2: wp.array | None = None,
        download: bool = True,
        j_extra_indices: wp.array | None = None,
        j_extra_alpha: wp.array | None = None,
    ) -> np.ndarray | None:
        """Build ``W = J · A⁻¹ · Jᵀ`` as a dense NumPy array (batched GPU implementation).

        Frictionless (``j_tangent1`` and ``j_tangent2`` are ``None``): emits an ``(M, M)``
        matrix with one row per contact (normal direction only).

        Frictional (both tangent arrays provided): emits a ``(3M, 3M)`` matrix with
        three rows per contact ordered ``[n_c, t1_c, t2_c]`` for c = 0..M-1.
        Off-diagonal coupling between contacts is fully included.

        It computes only the entries ``A^{-1}[i, j]`` for ``i, j`` in the set of
        unique particles of the rows (the "isolated DOFs" / isodofs), read from the
        sparse inverse, and assembles ``W`` from them, the per-row directions and
        the Jacobian coefficients entirely on device, without solving with ``A``.

        Args:
            num_contacts: Number of active contacts ``M``.
            j_indices: Particle index per contact, shape ``[M]``, dtype int32.
            j_normals: World-frame contact normal per contact, shape ``[M]``, dtype vec3.
            j_alpha: Jacobian coefficient per contact, shape ``[M]``, dtype float32.
            j_tangent1: First tangent direction per contact, shape ``[M]``, dtype vec3.
                When ``None``, frictionless behavior (M x M W).
            j_tangent2: Second tangent direction per contact, shape ``[M]``, dtype vec3.
                Must be provided together with ``j_tangent1``.
            download: If ``True`` (default) pull the result back as a NumPy
                array.  Set ``False`` from callers that consume ``W`` directly
                on device via :meth:`W_device_view`. This skips a multi-MB
                device-to-host copy that dominates the Schur build cost on
                large contact sets.  When ``False`` the method
                returns ``None``; the device-side ``W`` remains accessible
                through :meth:`W_device_view`.
            j_extra_indices: Optional further particles of each contact,
                shape ``[M, K - 1]``, dtype int32, ``-1`` when unused. A
                contact then acts on ``alpha_k`` times its direction on each
                of its ``K`` particles.
            j_extra_alpha: Their Jacobian coefficients, shape ``[M, K - 1]``, dtype float32.

        Returns:
            Dense float64 NumPy array of shape ``(M, M)`` (frictionless) or
            ``(3M, 3M)`` (frictional) when ``download=True``; otherwise ``None``.
        """
        from .kernels import (  # noqa: PLC0415
            compute_wi_tiled_kernel,
            densify_S_iso_scaled_kernel,
        )

        has_friction = j_tangent1 is not None and j_tangent2 is not None
        rows_per_contact = 3 if has_friction else 1
        M = num_contacts
        total_rows = M * rows_per_contact
        n = self.n
        dev = self.device

        # --- Build unified per-row direction and particle arrays on device ---
        # Frictionless: total_rows == M, each row uses normal direction.
        # Frictional: total_rows == 3M, rows interleaved [n_c, t1_c, t2_c].
        # ``j_*`` arrays may be over-allocated (capacity > M); slice to active range.
        idx_np = host_head(j_indices, M)  # (M,) int32, small, needed for row dir array
        n_np = host_head(j_normals, M)  # (M, 3) float32
        a_np = host_head(j_alpha, M)  # (M,) float32
        if has_friction:
            t1_np = host_head(j_tangent1, M)  # (M, 3) float32
            t2_np = host_head(j_tangent2, M)  # (M, 3) float32

        # Build (total_rows,) arrays for particle index, direction, alpha.
        row_particle = np.empty(total_rows, dtype=np.int32)
        row_dir = np.empty((total_rows, 3), dtype=np.float32)
        row_alpha = np.empty(total_rows, dtype=np.float32)
        if has_friction:
            # Frictional: 3 rows per contact, interleaved (n, t1, t2).
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

        extra_p = extra_a = None
        extra_h = None if j_extra_indices is None else host_head(j_extra_indices, M)
        if extra_h is not None and (extra_h >= 0).any():
            extra_p = np.repeat(extra_h, rows_per_contact, axis=0)
            extra_a = np.repeat(host_head(j_extra_alpha, M), rows_per_contact, axis=0)

        isodof_particles = self._set_row_slots(row_particle, row_alpha, extra_p, extra_a)

        # Upload direction arrays to device.
        row_particle_d = wp.array(row_particle, dtype=wp.int32, device=dev)
        row_dir_d = wp.array(row_dir, dtype=wp.vec3, device=dev)
        row_alpha_d = wp.array(row_alpha, dtype=wp.float32, device=dev)

        # --- Allocate/reuse the W device buffer, grown by :func:`dense_schur_capacity`. ---
        # ``M`` can spike by 10s of contacts between frames in a squeezed soft body, and a (cap, cap) dense float64
        # buffer is the dominant device alloc on the FBA hot path.
        if (
            not hasattr(self, "_W_device_d")
            or self._W_device_d.shape[0] < total_rows
            or self._W_device_d.shape[1] < total_rows
        ):
            prev = getattr(self, "_W_device_d", None)
            prev_cap = prev.shape[0] if prev is not None else 0
            cap = dense_schur_capacity(total_rows, prev_cap)
            self._W_device_d = wp.empty(shape=(cap, cap), dtype=wp.float64, device=dev)
        # Active row count for the populated sub-block, consumed by the NSN
        # GPU drivers to build a (M, M) wp.array view of ``_W_device_d`` without
        # touching the host.
        self._W_device_active_rows = total_rows

        # Stash per-row contact metadata for the lambda correction
        # (:meth:`apply_lambda_correction_combined`).
        self._row_particle_d = row_particle_d
        self._row_dir_d = row_dir_d
        self._row_alpha_d = row_alpha_d
        self._row_total = total_rows

        # Inverse mapping (particle → incident contact rows) for the
        # deterministic ``J^T·lambda`` gather in
        # :meth:`apply_lambda_correction_combined`. Each contact row maps to
        # exactly one particle (frictionless: 1 row/contact; frictional: 3 rows/contact
        # all sharing one particle), so a stable-sort by particle yields a
        # CSR adjacency. Built once per contact-set update. Host-side cost is
        # ``O(total_rows log total_rows)`` and ``total_rows`` is small (~1k).
        order = np.argsort(row_particle, kind="stable").astype(np.int32)
        sorted_particles = row_particle[order]
        counts = np.bincount(sorted_particles, minlength=n).astype(np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        self._isodof_row_offsets_d = wp.array(offsets, dtype=wp.int32, device=dev)
        self._isodof_row_indices_d = wp.array(order, dtype=wp.int32, device=dev)

        # Build ``Wi`` of size (k, k) for the unique contacted particles, then
        # assemble W with a 2D kernel reading
        # ``Wi[isodof_rank(p_cp), isodof_rank(p_c)]``.
        return self._build_schur_isodof(
            isodof_particles,
            row_particle_d,
            row_dir_d,
            row_alpha_d,
            total_rows,
            compute_wi_tiled_kernel,
            densify_S_iso_scaled_kernel,
            download=download,
        )

    def _build_schur_isodof(
        self,
        row_particle: np.ndarray,
        row_particle_d: wp.array,
        row_dir_d: wp.array,
        row_alpha_d: wp.array,
        total_rows: int,
        compute_wi_tiled_kernel,
        densify_S_iso_scaled_kernel,
        download: bool = True,
    ) -> np.ndarray | None:
        """Isodof-restricted Schur build.

        Collects the unique contacted DOFs ``isodofs``, computes the dense
        ``k x k`` matrix ``Wi[a, b] = A^{-1}[isodofs[a], isodofs[b]]`` via a
        column-intersection kernel over the sparse-inverse factor ``S``, and
        then assembles ``W[c', c]`` from per-contact ``(particle, dir, alpha)``
        triples and an ``isodof_rank`` lookup.

        This skips the solves with ``A`` for the ``M`` columns of ``Jᵀ``, which
        dominate the cost on sparse meshes, where
        ``k^2 * nnz_col(S) << M * nnz(S)``.

        ``Wi`` is computed via blocked dense ``X^T X`` GEMM (``wp.tile_matmul``):
        densify the selected columns of ``S`` into a padded ``(N_pad, k_pad)``
        buffer with ``sqrt(D_inv)`` baked into the values, then run a tiled
        SYRK-style matmul. A sparse column intersection would cost
        O(k² · nnz_col), which dominates the Schur build of the squeezing ball
        example, where ``nnz_col(S) ≈ 1700``.
        """
        N = self.n
        dev = self.device

        # --- Build isodofs (unique sorted particle indices) on host ---
        # Frictionless: row_particle has one entry per contact.
        # Frictional: each contact contributes 3 identical particle entries
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
            # No contacts touch any particle: nothing to compute.  Caller
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
            # Cap is larger than k_pad: slice a top-left ``(k_pad, k_pad)``
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

        self._Wi_v = Wi_v
        self._compose_schur()

        # Single host pull for the (total_rows x total_rows) W matrix.
        # Skipped when callers (e.g. the GPU NSN driver) consume ``W`` via
        # :meth:`W_device_view`. A multi-MB copy of a padded dense buffer
        # otherwise dominates the Schur build on large contact sets.
        if not download:
            return None
        W = self._W_device_d.numpy()[:total_rows, :total_rows].copy()
        return W

    def _compose_schur(self) -> None:
        """Compose ``W`` from ``Wi`` and the row particles, directions and coefficients."""
        from .kernels import add_slot_terms_W_kernel, compose_W_from_wi_kernel  # noqa: PLC0415

        n = self._row_total
        # W[c', c] = alpha[c'] * alpha[c] * dot(dir[c'], dir[c]) * Wi[r', r], written densely; one thread per
        # (c', c) pair. ``isodof_rank[N]`` provides the rank lookup per particle.
        wp.launch(
            compose_W_from_wi_kernel,
            dim=(n, n),
            inputs=[self._row_particle_d, self._row_dir_d, self._row_alpha_d, self._isodof_rank_d, self._Wi_v],
            outputs=[self._W_device_d],
            device=self.device,
        )
        if self._row_slots is not None:
            wp.launch(
                add_slot_terms_W_kernel,
                dim=(n, n),
                inputs=[*self._row_slots[:2], self._row_dir_d, self._isodof_rank_d, self._Wi_v],
                outputs=[self._W_device_d],
                device=self.device,
            )

    def setup_lite_mass(self, inv_mass_d: wp.array, dt: float) -> None:
        """Cache the LiteNSN mass operator ``dt² · inv_mass`` of each particle.

        The lite NSN takes it in place of ``A⁻¹``. Pinned particles
        (``inv_mass == 0``) give zero, so contacts that target pinned vertices
        produce a zero entry in ``W_lite``, consistent with the full NSN, where
        the pin stiffness on the diagonal of ``A`` makes the pinned entries of
        ``A⁻¹`` nearly zero.

        The ``dt²`` factor follows the scaling of ``A``, whose mass diagonal
        ``M / dt²`` inverts to ``dt² · inv_mass``.  Without this factor,
        ``W_lite`` would be ``1/dt²`` times the full NSN's ``J A⁻¹ Jᵀ`` in
        scale, and FB-Newton's linearization would not match it.

        The sparse lite Schur complement of :meth:`build_schur_lite_triplets`
        reads the scalars ``dt² · inv_mass`` too.
        """
        n = inv_mass_d.shape[0]
        if n != self.n:
            raise ValueError(f"inv_mass length {n} != solver n {self.n}")
        inv_mass_h = inv_mass_d.numpy().astype(np.float64)
        # The lite kernels and ``build_schur_lite_dense`` read ``inv_mass`` and ``dt²`` directly; they are rebuilt
        # when dt or the pin layout changes.
        self._inv_mass_d = wp.array(inv_mass_h, dtype=wp.float64, device=self.device)
        self._lite_dt2 = float(dt) * float(dt)
        self._lite_s_d = wp.array(self._lite_dt2 * inv_mass_h, dtype=wp.float64, device=self.device)

    def _set_row_slots(self, row_particle, row_alpha, extra_p, extra_a) -> np.ndarray:
        """Keep the extra slots of rows on several particles, with a CSR from particle to ``row * K + slot``.

        Returns every particle of the rows, the first slots first.
        """
        self._row_slots = None
        if extra_p is None or not (extra_p >= 0).any():
            return row_particle
        dev, n = self.device, self.n
        rows, slots = np.nonzero(extra_p >= 0)
        width = extra_p.shape[1] + 1
        order = np.argsort(extra_p[rows, slots], kind="stable")
        offsets = np.zeros(n + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(extra_p[rows, slots], minlength=n))
        self._row_slots = (
            wp.array(np.column_stack([row_particle, extra_p]), dtype=wp.int32, device=dev),
            wp.array(np.column_stack([row_alpha, extra_a]), dtype=wp.float32, device=dev),
            wp.array(offsets, dtype=wp.int32, device=dev),
            wp.array((rows * width + slots + 1)[order].astype(np.int32), dtype=wp.int32, device=dev),
        )
        return np.concatenate([row_particle, extra_p[rows, slots]])

    def _setup_lite_rows_on_device(self, M, rows_per_contact, directions, j_indices, j_alpha, extra) -> int:
        """:meth:`setup_lite_row_meta` on the device for rows on several particles, without the list of particles
        that the rows touch, which only the per-particle lite kernel reads: the CSRs keep the order of a stable
        sort of the rows, and of the further slots row by row."""
        dev, n, width = self.device, self.n, extra[0].shape[1]
        R = M * rows_per_contact
        self._row_dir_d = wp.empty(R, dtype=wp.vec3, device=dev)
        wp.launch(lite_row_dirs_kernel, R, [*directions, rows_per_contact], [self._row_dir_d], device=dev)
        self._row_particle_d = wp.empty(R, dtype=wp.int32, device=dev)
        self._row_alpha_d = wp.empty(R, dtype=wp.float32, device=dev)
        slots = [wp.empty((R, width + 1), dtype=t, device=dev) for t in (wp.int32, wp.float32)]
        row_kv = [wp.empty(2 * R, dtype=wp.int32, device=dev) for _ in range(2)]
        slot_kv = [wp.empty(2 * R * width, dtype=wp.int32, device=dev) for _ in range(2)]
        counts = [wp.zeros(n, dtype=wp.int32, device=dev) for _ in range(2)]
        inputs = [j_indices, j_alpha, *extra, rows_per_contact]
        outputs = [self._row_particle_d, self._row_alpha_d, *slots, *row_kv, counts[0], *slot_kv, counts[1]]
        wp.launch(lite_row_slots_kernel, R, inputs, outputs, device=dev)
        offsets = [wp.zeros(n + 1, dtype=wp.int32, device=dev) for _ in range(2)]
        for kv, size, count, offset in zip((row_kv, slot_kv), (R, R * width), counts, offsets, strict=True):
            wp.utils.radix_sort_pairs(*kv, size)
            wp.utils.array_scan(count, offset[1:], inclusive=True)
        self._row_slots = (*slots, offsets[1], slot_kv[1])
        self._isodof_row_offsets_d, self._isodof_row_indices_d = offsets[0], row_kv[1]
        self._isodof_unique_d, self._n_isodofs, self._isodof_max_rows = None, 0, 0
        self._row_total = R
        return R

    def setup_lite_row_meta(
        self,
        num_contacts: int,
        j_indices: wp.array,
        j_normals: wp.array,
        j_alpha: wp.array,
        j_tangent1: wp.array | None = None,
        j_tangent2: wp.array | None = None,
        j_extra_indices: wp.array | None = None,
        j_extra_alpha: wp.array | None = None,
        host: tuple | None = None,
    ) -> int:
        """Populate per-row metadata for LiteNSN paths without building ``W``.

        The fused-block-diagonal lite kernel
        (:func:`kernels.fb_newton_lite_coulomb_kernel`) doesn't need an
        explicit ``W`` buffer at all, but
        :meth:`apply_lambda_correction_combined` still depends on
        ``_row_particle_d`` / ``_row_dir_d`` / ``_row_alpha_d`` /
        ``_isodof_row_*_d`` to gather ``Jᵀ·λ``.  This helper sets those up so
        the lite paths can call ``apply_lambda_correction_combined`` without
        going through :meth:`build_schur_lite_dense` (which would allocate the
        dense ``W`` buffer, defeating the lite memory win at large ``M``).

        ``host`` holds host copies of ``j_indices`` and ``j_alpha``, or
        ``None`` for them, so that they need not be read from the device.
        With ``j_extra_indices`` and ``j_extra_alpha``, the further particles
        of the contacts, shape ``[M, K - 1]``, as in
        :meth:`build_schur_complement`, the rows are set up on the device.

        Returns ``total_rows`` (``M`` frictionless, ``3M`` frictional).
        """
        has_friction = j_tangent1 is not None and j_tangent2 is not None
        rows_per_contact = 3 if has_friction else 1
        M = int(num_contacts)
        total_rows = M * rows_per_contact
        dev = self.device
        n = self.n
        if j_extra_indices is not None:
            normals = (j_normals, j_tangent1, j_tangent2) if has_friction else (j_normals, j_normals, j_normals)
            extra = (j_extra_indices, j_extra_alpha)
            return self._setup_lite_rows_on_device(M, rows_per_contact, normals, j_indices, j_alpha, extra)

        if host is None:
            host = (j_indices.numpy(), j_alpha.numpy())
        idx_np, a_np = host[0][:M].astype(np.int32), host[1][:M]
        row_particle = np.repeat(idx_np, rows_per_contact)
        row_alpha = np.repeat(a_np, rows_per_contact)
        tangents = (j_tangent1, j_tangent2) if has_friction else (j_normals, j_normals)
        self._row_dir_d = wp.empty(total_rows, dtype=wp.vec3, device=dev)
        inputs = [j_normals, *tangents, rows_per_contact]
        wp.launch(lite_row_dirs_kernel, total_rows, inputs, [self._row_dir_d], device=dev)
        self._row_slots = None

        self._row_particle_d = wp.array(row_particle, dtype=wp.int32, device=dev)
        self._row_alpha_d = wp.array(row_alpha, dtype=wp.float32, device=dev)
        self._row_total = total_rows

        order = np.argsort(row_particle, kind="stable").astype(np.int32)
        sorted_particles = row_particle[order]
        counts = np.bincount(sorted_particles, minlength=n).astype(np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        self._isodof_row_offsets_d = wp.array(offsets, dtype=wp.int32, device=dev)
        self._isodof_row_indices_d = wp.array(order, dtype=wp.int32, device=dev)

        # The per-particle fused kernel launches one thread
        # per *unique* contacted particle, not per all-particles.  Stash the
        # compact list of unique isodofs and the per-isodof contact count so
        # the kernel can index its rows directly.
        unique_isodofs = np.unique(row_particle).astype(np.int32)
        # Per-isodof local count (k_p) and the row range within
        # _isodof_row_indices_d.  The offsets in unique_isodofs index into the
        # all-particles CSR _isodof_row_offsets_d above.
        self._isodof_unique_d = wp.array(unique_isodofs, dtype=wp.int32, device=dev)
        self._n_isodofs = int(unique_isodofs.size)
        self._isodof_max_rows = int(counts.max()) if unique_isodofs.size else 0

        return total_rows

    def build_schur_lite_triplets(self, total_rows: int, cols_d: wp.array, vals_d: wp.array) -> wps.BsrMatrix:
        """``W = H (dt^2 M^-1) H^T`` from the triplets of ``H``, rows of 1 x 3 blocks over the particles in four
        slots per row, so that slot ``t`` lies in row ``t // 4``; slots with a negative column are skipped.

        Two rows couple where they share a particle. The slots are sorted by particle, and each row of ``W`` merges
        the rows on its particles in row order, summing ``dt^2 / m_p`` times the products of their blocks over the
        particles they share, without a general sparse product. ``W`` is a scalar CSR matrix in BSR form, with each
        row's columns in order.
        """
        from .kernels import lite_triplet_key_kernel, lite_w_row_fill_kernel, lite_w_row_nnz_kernel  # noqa: PLC0415

        dev, n, R = self.device, self.n, int(total_rows)
        slots = 4 * R
        # The radix sort takes keys and values twice the count it sorts.
        if getattr(self, "_lw_slot_cap", 0) < slots:
            self._lw_slot_cap = cap = max(slots, int(1.5 * getattr(self, "_lw_slot_cap", 0)))
            self._lw_keys = wp.empty(2 * cap, dtype=wp.int32, device=dev)
            self._lw_index = wp.empty(2 * cap, dtype=wp.int32, device=dev)
        if getattr(self, "_lw_counts", None) is None:
            self._lw_counts = wp.zeros(n, dtype=wp.int32, device=dev)
            self._lw_starts = wp.zeros(n, dtype=wp.int32, device=dev)
        if getattr(self, "_lw_row_cap", 0) < R + 1:
            self._lw_row_cap = cap = max(R + 1, int(1.5 * getattr(self, "_lw_row_cap", 0)))
            self._lw_row_nnz = wp.zeros(cap, dtype=wp.int32, device=dev)
            self._lw_offsets = wp.zeros(cap, dtype=wp.int32, device=dev)
        self._lw_counts.zero_()
        outputs = [self._lw_keys, self._lw_index, self._lw_counts]
        wp.launch(lite_triplet_key_kernel, slots, [cols_d, n], outputs, device=dev)
        wp.utils.radix_sort_pairs(self._lw_keys, self._lw_index, slots)
        wp.utils.array_scan(self._lw_counts, self._lw_starts, inclusive=False)
        row_nnz, offsets = self._lw_row_nnz[: R + 1], self._lw_offsets[: R + 1]
        row_nnz.zero_()
        inputs = [cols_d, self._lw_index, self._lw_starts, self._lw_counts]
        wp.launch(lite_w_row_nnz_kernel, R, inputs, [row_nnz], device=dev)
        wp.utils.array_scan(row_nnz, offsets, inclusive=False)
        nnz = int(offsets[R : R + 1].numpy()[0])
        if getattr(self, "_lw_nnz_cap", 0) < nnz:
            self._lw_nnz_cap = cap = max(nnz, int(1.5 * getattr(self, "_lw_nnz_cap", 0)))
            self._lw_columns = wp.empty(cap, dtype=wp.int32, device=dev)
            self._lw_values = wp.empty(cap, dtype=wp.float64, device=dev)
        inputs = [cols_d, vals_d, self._lw_index, self._lw_starts, self._lw_counts, self._lite_s_d, offsets]
        wp.launch(lite_w_row_fill_kernel, R, inputs, [self._lw_columns, self._lw_values], device=dev)
        W = getattr(self, "_lw_W", None)
        if W is None:
            W = self._lw_W = wps.bsr_zeros(rows_of_blocks=R, cols_of_blocks=R, block_type=wp.float64, device=dev)
        W.nrow, W.ncol, W.nnz = R, R, nnz
        W.offsets, W.columns, W.values = offsets, self._lw_columns, self._lw_values
        return W

    def reserve_a_schur_lite_bsr(self, W_bsr: wps.BsrMatrix) -> wps.BsrMatrix:
        """Storage for ``A_schur`` with the sparsity of ``W_bsr``, which it shares; its values are not set."""
        nnz = int(W_bsr.nnz)
        n_rows = int(W_bsr.nrow)

        if (
            not hasattr(self, "_A_schur_lite_bsr")
            or self._A_schur_lite_bsr is None
            or int(self._A_schur_lite_bsr.values.shape[0]) < nnz
        ):
            # Allocate a placeholder via bsr_zeros (scalar block_type = fp64).
            self._A_schur_lite_bsr = wps.bsr_zeros(
                rows_of_blocks=n_rows,
                cols_of_blocks=n_rows,
                block_type=wp.float64,
                device=self.device,
            )
            self._A_schur_lite_values_cap = max(nnz, 32)
            self._A_schur_lite_bsr.values = wp.empty(
                self._A_schur_lite_values_cap,
                dtype=wp.float64,
                device=self.device,
            )

        # Share W's pattern arrays (offsets + columns), no copy.
        self._A_schur_lite_bsr.nrow = n_rows
        self._A_schur_lite_bsr.ncol = n_rows
        self._A_schur_lite_bsr.nnz = nnz
        self._A_schur_lite_bsr.offsets = W_bsr.offsets
        self._A_schur_lite_bsr.columns = W_bsr.columns

        return self._A_schur_lite_bsr

    def build_a_schur_lite_bsr(
        self,
        W_bsr: wps.BsrMatrix,
        omega_d: wp.array,  # (total_rows,) fp64
        compliance_d: wp.array,  # (total_rows,) fp64
    ) -> wps.BsrMatrix:
        """Construct ``A_schur = ωωᵀ ⊙ W + diag(c)`` in BSR (same sparsity as W).

        ``warp.sparse`` collapses 1x1-block BSR storage to scalar fp64 values,
        so ``W_bsr.values.dtype == wp.float64`` (not ``mat1x1``).  We allocate
        the sibling A_schur matrix with the same convention.
        """
        from .kernels import build_a_schur_lite_bsr_kernel  # noqa: PLC0415

        A = self.reserve_a_schur_lite_bsr(W_bsr)
        wp.launch(
            build_a_schur_lite_bsr_kernel,
            dim=int(W_bsr.nrow),
            inputs=[
                W_bsr.offsets,
                W_bsr.columns,
                W_bsr.values,
                omega_d,
                compliance_d,
            ],
            outputs=[A.values],
            device=self.device,
        )
        return A

    def build_schur_lite_dense(
        self,
        num_contacts: int,
        j_indices: wp.array,
        j_normals: wp.array,
        j_alpha: wp.array,
        j_tangent1: wp.array | None = None,
        j_tangent2: wp.array | None = None,
    ) -> None:
        """Write the LiteNSN Schur ``W_lite`` densely into ``_W_device_d``.

        Uses :func:`compose_W_lite_kernel` to compute the dense projection of
        the sparse Schur ``H · (dt²·M⁻¹) · Hᵀ``.  Off-particle entries are
        explicitly zeroed; only entries sharing a particle take the lite
        approximation.

        The dense ``W_lite`` slots into the dense-PCR NSN inner solve of
        ``SolverFBA(nsn_schur_mode="lite")`` for frictionless contacts.

        Side effects: populates ``_W_device_d[:total_rows, :total_rows]`` and
        ``_W_device_active_rows`` so subsequent :meth:`W_device_view` calls
        return the lite Schur.  Also stashes per-row metadata
        (``_row_particle_d`` / ``_row_dir_d`` / ``_row_alpha_d``) and the
        isodof inverse-mapping CSR used by
        :meth:`apply_lambda_correction_combined`.
        """
        from .kernels import compose_W_lite_kernel  # noqa: PLC0415

        if not hasattr(self, "_inv_mass_d") or self._inv_mass_d is None:
            raise RuntimeError(
                "build_schur_lite_dense called before setup_lite_mass; SolverFBA must"
                " invoke setup_lite_mass on every _setup_pd_system pass."
            )
        if not hasattr(self, "_lite_dt2") or self._lite_dt2 is None:
            raise RuntimeError("setup_lite_mass did not cache dt²; check call order.")

        has_friction = j_tangent1 is not None and j_tangent2 is not None
        rows_per_contact = 3 if has_friction else 1
        M = int(num_contacts)
        total_rows = M * rows_per_contact
        dev = self.device
        n = self.n

        # --- Build unified per-row (particle, dir, alpha) on device.  Same
        # layout the dense full-NSN path uses; assembled fresh per call. ---
        idx_np = j_indices.numpy()[:M].astype(np.int32)
        n_np = j_normals.numpy()[:M]
        a_np = j_alpha.numpy()[:M]
        row_particle = np.empty(total_rows, dtype=np.int32)
        row_dir = np.empty((total_rows, 3), dtype=np.float32)
        row_alpha = np.empty(total_rows, dtype=np.float32)
        if has_friction:
            t1_np = j_tangent1.numpy()[:M]
            t2_np = j_tangent2.numpy()[:M]
            for k, direction in enumerate((n_np, t1_np, t2_np)):
                row_particle[k::3] = idx_np
                row_dir[k::3] = direction
                row_alpha[k::3] = a_np
        else:
            row_particle[:] = idx_np
            row_dir[:] = n_np
            row_alpha[:] = a_np

        row_particle_d = wp.array(row_particle, dtype=wp.int32, device=dev)
        row_dir_d = wp.array(row_dir, dtype=wp.vec3, device=dev)
        row_alpha_d = wp.array(row_alpha, dtype=wp.float32, device=dev)
        self._row_particle_d = row_particle_d
        self._row_dir_d = row_dir_d
        self._row_alpha_d = row_alpha_d
        self._row_total = total_rows

        # --- Allocate / grow the W_device buffer to (total_rows, total_rows). ---
        # Mirrors the full-NSN path's growth policy so callers using
        # :meth:`W_device_view` see the same buffer semantics.
        if (
            not hasattr(self, "_W_device_d")
            or self._W_device_d.shape[0] < total_rows
            or self._W_device_d.shape[1] < total_rows
        ):
            prev = getattr(self, "_W_device_d", None)
            prev_cap = prev.shape[0] if prev is not None else 0
            cap = dense_schur_capacity(total_rows, prev_cap)
            self._W_device_d = wp.empty(shape=(cap, cap), dtype=wp.float64, device=dev)
        self._W_device_active_rows = total_rows

        # Isodof inverse-mapping CSR (apply_lambda_correction_combined needs this).
        order = np.argsort(row_particle, kind="stable").astype(np.int32)
        sorted_particles = row_particle[order]
        counts = np.bincount(sorted_particles, minlength=n).astype(np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        self._isodof_row_offsets_d = wp.array(offsets, dtype=wp.int32, device=dev)
        self._isodof_row_indices_d = wp.array(order, dtype=wp.int32, device=dev)

        # --- Launch the lite Schur composition kernel.  Writes a contiguous
        # (total_rows, total_rows) slice of the cap-sized buffer. ---
        # The kernel reads ``_W_device_d`` as wp.array2d; the cap-sized buffer
        # has stride ``cap * 8``, so when the active block is < cap we
        # explicitly construct a strided view so the kernel writes to the
        # right elements.
        cap = int(self._W_device_d.shape[0])
        if cap == total_rows:
            W_v = self._W_device_d
        else:
            W_v = wp.array(
                ptr=self._W_device_d.ptr,
                dtype=wp.float64,
                shape=(total_rows, total_rows),
                strides=(cap * 8, 8),
                device=dev,
            )
        wp.launch(
            compose_W_lite_kernel,
            dim=(total_rows, total_rows),
            inputs=[
                row_particle_d,
                row_dir_d,
                row_alpha_d,
                self._inv_mass_d,
                wp.float64(self._lite_dt2),
            ],
            outputs=[W_v],
            device=dev,
        )

    def W_device_view(self) -> wp.array2d[wp.float64]:
        """Return a ``(M, M)`` device view of the most recently built Schur W.

        The view aliases :attr:`_W_device_d` for the active row block populated
        by the most recent :meth:`build_schur_complement` call.  Because the
        underlying buffer has capacity ``(cap, cap)`` (with ``cap >= M`` and
        ``cap`` grown by the 1.5x headroom factor), the view is non-contiguous
        in its outer dimension and explicit strides matching the parent buffer
        are passed; without them the default contiguous strides would misplace
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

    def _gather_slot_terms(self) -> None:
        """Add the extra slots of rows on several particles to ``_jt_lambda_d``."""
        if self._row_slots is None:
            return
        from .kernels import gather_jt_lambda_slots_kernel  # noqa: PLC0415

        _, slot_alpha, offsets, entries = self._row_slots
        wp.launch(
            gather_jt_lambda_slots_kernel,
            dim=self.n,
            inputs=[offsets, entries, self._row_dir_d, slot_alpha, self._lam_d],
            outputs=[self._jt_lambda_d],
            device=self.device,
        )

    def reserve_lambda_correction(self) -> None:
        """Allocate the buffers of :meth:`apply_lambda_correction_combined` for the rows of the last Schur build,
        so that a graph capture of it allocates nothing."""
        n_rows = self._row_total
        if not hasattr(self, "_jt_lambda_d") or self._jt_lambda_d.shape[0] != self.n:
            self._jt_lambda_d = wp.empty(self.n, dtype=wp.vec3, device=self.device)
        if not hasattr(self, "_lam_d") or self._lam_d.shape[0] < n_rows:
            cap = max(n_rows, int(getattr(self, "_lam_cap", 0) * 1.5) + 1)
            self._lam_cap = cap
            self._lam_d = wp.empty(cap, dtype=wp.float32, device=self.device)

    def apply_lambda_correction_combined(
        self,
        lam: np.ndarray | wp.array,
        rhs_inout: wp.array[wp.vec3],
        x_out: wp.array[wp.vec3],
    ) -> None:
        """Compute ``x_out = A⁻¹ · (rhs_inout + Jᵀ · lam)`` with in-place rhs update.

        Applies the constraint correction in combined form:

            b += dt² · Jᵀ · (ω · λ)
            x = A⁻¹ · b

        :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA.step` uses it
        instead of the split form ``x_cur += A⁻¹ · Jᵀ · lam_apply``, which is
        algebraically equivalent but rounds differently in float32: the
        combined form takes a single ``A⁻¹`` solve over the incremented ``b``.

        The implementation:

        1. Gathers ``Jᵀ · lam`` into the cached ``_jt_lambda_d`` buffer, a
           particle-centered gather over the inverse-mapping CSR, without atomics.
        2. Accumulates that into ``rhs_inout`` in-place (``rhs_inout[p] +=
           (Jᵀ·lam)[p]``).
        3. Runs a single solve ``x_out = A⁻¹ · rhs_inout`` with the factored ``A``.

        Every Schur build populates the per-row metadata (``_row_*_d``,
        ``_isodof_row_*_d``) that the gather reads.

        Args:
            lam: Contact impulse vector, shape ``(total_rows,)``. Accepts
                either a host ``numpy`` array, as the tests pass, or a
                device-resident ``wp.array[wp.float64]`` produced by the
                NSN inner driver. ``total_rows == M`` frictionless,
                ``3 * M`` frictional.
            rhs_inout: Right-hand side vec3 array of length ``N``,
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
        self.reserve_lambda_correction()

        # Accept either a host numpy array, as the tests pass, or a
        # device-resident fp64 array: the solver passes ``_nsn_lam_apply_d``
        # directly, without a round trip through the host in every PD
        # iteration.
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
        self._gather_slot_terms()

        # Step 2: rhs_inout += Jᵀ·lam, that is b += dt²·Jᵀ·(ω·λ): the dt²·ω
        # weighting is folded into ``lam`` by the caller, see
        # ``_solve_nsn_*`` which returns ``lam_apply = dt²·ω·λ``).
        wp.launch(
            accumulate_vec3_kernel,
            dim=n,
            inputs=[self._jt_lambda_d],
            outputs=[rhs_inout],
            device=dev,
        )

        # Step 3: single solve x_out = A⁻¹ · rhs_inout with the factored A.
        self.solve(rhs_inout, x_out)
