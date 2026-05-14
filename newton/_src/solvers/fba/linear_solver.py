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

if TYPE_CHECKING:
    import scipy.sparse as sp

from ...sim import Model


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

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # ---- Mass diagonal ----
    inv_dt2 = 1.0 / (dt * dt)
    for i in range(N):
        if inv_mass[i] > 0.0:
            rows.append(i)
            cols.append(i)
            vals.append(mass[i] * inv_dt2)

    # ---- Pin diagonal ----
    pin_indices = np.where(inv_mass == 0.0)[0].astype(np.int32)
    for i in pin_indices:
        rows.append(int(i))
        cols.append(int(i))
        vals.append(pin_stiffness)

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

        # ST is the 3x2 reference-space gradient selector for the 3-vertex stencil.
        ST = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        for t in range(tri_indices.shape[0]):
            G = ST @ tri_rest_inv[t]  # 3x2
            K = G @ G.T  # 3x3
            w = tri_weight[t]
            for a in range(3):
                ia = int(tri_indices[t, a])
                for b in range(3):
                    ib = int(tri_indices[t, b])
                    rows.append(ia)
                    cols.append(ib)
                    vals.append(w * K[a, b])

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

        for e in range(edge_indices.shape[0]):
            q = edge_quad_q[e]  # length 4
            w = edge_weight[e] * edge_quad_scale[e]  # combine user weight + isometric scale
            if w == 0.0:
                continue
            for a in range(4):
                ia = int(edge_indices[e, a])
                for b in range(4):
                    ib = int(edge_indices[e, b])
                    rows.append(ia)
                    cols.append(ib)
                    vals.append(w * q[a] * q[b])

    A = _sp.csr_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
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


def compute_lower_inverse(
    L,
    parent: np.ndarray | None = None,
    invperm: np.ndarray | None = None,
):
    """Compute S = L⁻¹ as a sparse CSC matrix.

    Ports RealSim ``LDLT_computeLowerInverse`` (SparseLDLT.cpp:225-347).
    Sparsity pattern of S is derived from L's elimination tree; this gives
    the exact sparse inverse (no thresholding, no fill-in).

    Args:
        L: Lower-triangular CSC matrix with unit diagonal.
        parent: Elimination tree parent array; computed from L if None.
        invperm: Inverse permutation (identity if None).

    Returns:
        S as CSC; S · L ≈ I in the permuted basis.

    Performance note:
        This is a pure-Python port for MVP correctness. Setup time scales
        roughly as O(S_nnz) Python operations. Measured: ~6s for N≈2500,
        ~40s for N≈10000. Future optimization (Numba / vectorized NumPy)
        is tracked as follow-up; for current cloth-scale MVP problems
        (N<5000), setup cost is acceptable but not interactive.
    """
    import scipy.sparse as _sp

    n = L.shape[0]
    if parent is None:
        parent = _elimination_tree(L)
    if invperm is None:
        invperm = np.arange(n, dtype=np.int32)
    perm = np.argsort(invperm).astype(np.int32)

    # --- 1. Count nnz of S (LDLT_computeLowerInverseNNZ) ---
    S_nnz = 0
    for i in range(n):
        index = int(invperm[i])
        innercount = 1
        while parent[index] != -1:
            innercount += 1
            index = int(parent[index])
        S_nnz += innercount

    # --- 2. Build S's CSC pattern (LDLT_computeLowerInversePattern) ---
    S_outerPtr = np.zeros(n + 1, dtype=np.int32)
    S_innerInd = np.zeros(S_nnz, dtype=np.int32)
    count = 0
    S_outerPtr[0] = 0
    for i in range(n):
        index = int(invperm[i])
        innercount = 1
        S_innerInd[count] = index
        while parent[index] != -1:
            S_innerInd[count + innercount] = int(parent[index])
            innercount += 1
            index = int(parent[index])
        count += innercount
        S_outerPtr[i + 1] = count

    # --- 3. Compute aligned L values (LDLT_computeAlignedLower) ---
    aL = np.zeros(S_nnz, dtype=np.float64)
    L_outerPtr = L.indptr
    L_innerInd = L.indices
    L_values = L.data
    for row in range(n):
        invr = int(invperm[row])
        ptrS = int(S_outerPtr[row])
        for ptrL in range(L_outerPtr[invr], L_outerPtr[invr + 1]):
            while ptrS < S_outerPtr[row + 1]:
                if S_innerInd[ptrS] == L_innerInd[ptrL]:
                    aL[ptrS] = L_values[ptrL]
                    break
                ptrS += 1

    # --- 4. Solve for S column-by-column (LDLT_computeLowerInverse_line) ---
    S_values = np.zeros(S_nnz, dtype=np.float64)
    for index in range(n):
        S_values[S_outerPtr[index]] = 1.0
        for i in range(S_outerPtr[index] + 1, S_outerPtr[index + 1]):
            col = int(perm[S_innerInd[i - 1]])
            j = i
            k = int(S_outerPtr[col]) + 1
            # Elimination tree guarantees S[col]'s column is the suffix-path of S[index]
            # starting at `col`, so j and k always exhaust simultaneously. `and` is
            # functionally equivalent to RealSim's `||` here (SparseLDLT.cpp:315) and
            # eliminates the need for an explicit overrun guard.
            while j < S_outerPtr[index + 1] and k < S_outerPtr[col + 1]:
                S_values[j] -= aL[k] * S_values[i - 1]
                j += 1
                k += 1

    return _sp.csc_matrix((S_values, S_innerInd, S_outerPtr), shape=(n, n))


def _compute_isometric_bending_q(edge_indices: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge length-4 vector q and per-edge scale ``3 / (A0 + A1)``.

    Bergou et al. 2006 isometric bending: for stencil (v0, v1, v2, v3) where
    (v0, v1) is the shared edge and (v2, v3) the opposite vertices of the two
    adjacent triangles, the per-edge bending Hessian contribution is
    ``(3 / (A0 + A1)) · q·qᵀ`` with q given by cotangents at all four
    edge-endpoint angles. See PDIsometricBendingEnergy.cpp::init().

    Args:
        edge_indices: Shape (E, 4), columns are (v0, v1, v2, v3).
        positions: Shape (N, 3) particle rest positions.

    Returns:
        A tuple ``(q, scale)`` where ``q`` has shape (E, 4) and ``scale`` has
        shape (E,) with per-edge values ``3 / (A0 + A1)``.
    """
    E = edge_indices.shape[0]
    q = np.zeros((E, 4), dtype=np.float64)
    scale = np.zeros((E,), dtype=np.float64)
    for e in range(E):
        v0, v1, v2, v3 = edge_indices[e]
        x0, x1, x2, x3 = positions[v0], positions[v1], positions[v2], positions[v3]
        l01 = np.linalg.norm(x1 - x0)
        l02 = np.linalg.norm(x2 - x0)
        l12 = np.linalg.norm(x2 - x1)
        l03 = np.linalg.norm(x3 - x0)
        l13 = np.linalg.norm(x3 - x1)
        # Heron's formula for triangle areas.
        r0 = 0.5 * (l01 + l02 + l12)
        A0 = np.sqrt(max(r0 * (r0 - l01) * (r0 - l02) * (r0 - l12), 0.0))
        r1 = 0.5 * (l01 + l03 + l13)
        A1 = np.sqrt(max(r1 * (r1 - l01) * (r1 - l03) * (r1 - l13), 0.0))
        # Four cotangents at the edge-endpoint angles.
        cot02 = (l01 * l01 - l02 * l02 + l12 * l12) / (4.0 * max(A0, 1e-20))
        cot12 = (l01 * l01 + l02 * l02 - l12 * l12) / (4.0 * max(A0, 1e-20))
        cot03 = (l01 * l01 - l03 * l03 + l13 * l13) / (4.0 * max(A1, 1e-20))
        cot13 = (l01 * l01 + l03 * l03 - l13 * l13) / (4.0 * max(A1, 1e-20))
        q[e] = np.array([cot02 + cot03, cot12 + cot13, -(cot02 + cot12), -(cot03 + cot13)])
        scale[e] = 3.0 / max(A0 + A1, 1e-20)
    return q, scale


@dataclass
class FactorizedSystem:
    """Output of factorize_and_sparse_inverse."""

    S: "sp.csc_matrix"          # n×n, lower-tri inverse with elimination-tree sparsity
    ST: "sp.csc_matrix"         # transpose, runtime-uploaded separately
    Dinv: np.ndarray            # length n
    perm_r: np.ndarray          # length n, int32
    invperm_r: np.ndarray       # length n, int32


def factorize_and_sparse_inverse(A) -> FactorizedSystem:
    """Run COLAMD ordering + LU factor + sparse inverse on an SPD matrix.

    Args:
        A: scalar N×N PD Hessian (csr or csc, will be converted internally).

    Returns:
        FactorizedSystem with all data needed to construct an FBALinearSolver.
    """
    L, _U, Dinv, perm_r, _perm_c = _splu_extract_factors(A)
    parent = _elimination_tree(L)
    invperm_r = np.argsort(perm_r).astype(np.int32)
    S = compute_lower_inverse(L, parent=parent)
    ST = S.T.tocsc()
    return FactorizedSystem(S=S, ST=ST, Dinv=Dinv, perm_r=perm_r, invperm_r=invperm_r)
