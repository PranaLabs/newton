# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 numerical gate: 4-particle Schur LCP matches NumPy reference.

Constructs a synthetic 4-particle row with a hand-computed Jacobian, calls
`FBALinearSolver.build_schur_lite_unified` with the 4-slot path, and compares
the resulting ``W = H · (dt² · M⁻¹) · Hᵀ`` against a NumPy dense reference.

Acceptance: max abs diff < 1e-9 (fp64 round-off).

Bypasses ``SolverFBA._apply_particle_contact_pass`` entirely — uses the
FBA linear solver directly with a 4-particle, 3-row block (n + 2 tangents).
"""

from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp
import warp as wp

from newton._src.solvers.fba.linear_solver import (
    FBALinearSolver,
    factorize_and_sparse_inverse,
)


def numpy_schur_unified(
    pa, pb, pc, pd, wa, wb, wc, wd, dirs, alphas, inv_mass, dt
):
    """Reference dense W = H · diag(dt² · inv_mass) ⊗ I₃ · Hᵀ."""
    n_rows = len(pa)
    n_particles = len(inv_mass)
    H = np.zeros((n_rows, 3 * n_particles), dtype=np.float64)
    for r in range(n_rows):
        d = dirs[r]
        alpha = alphas[r]
        slots = [(pa[r], wa[r]), (pb[r], wb[r]), (pc[r], wc[r]), (pd[r], wd[r])]
        for p, w in slots:
            if p < 0 or w == 0.0:
                continue
            H[r, 3 * p : 3 * p + 3] = w * alpha * d
    dt2 = dt * dt
    M_inv = np.zeros(3 * n_particles, dtype=np.float64)
    for p in range(n_particles):
        M_inv[3 * p : 3 * p + 3] = dt2 * inv_mass[p]
    W = H @ np.diag(M_inv) @ H.T
    return W


def main():
    wp.init()
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    print(f"device={device}")

    # ---- Setup: 4 particles, masses 1 / 2 / 3 / 4 kg, dt = 0.01 s ----
    n_particles = 4
    inv_mass = np.array([1.0, 0.5, 1.0 / 3.0, 0.25], dtype=np.float64)
    dt = 0.01

    # FBALinearSolver requires a factorized SPD matrix.  For this test it
    # doesn't matter what the matrix actually is (we only use `build_schur_lite_unified`
    # which depends on `setup_lite_mass`); use identity.
    A_dummy = sp.eye(n_particles, format="csc")
    factor = factorize_and_sparse_inverse(A_dummy)
    ls = FBALinearSolver(factor, device)
    inv_mass_d = wp.array(inv_mass.astype(np.float32), dtype=wp.float32, device=device)
    ls.setup_lite_mass(inv_mass_d, dt)

    # ---- Hand-crafted 4-particle rows ----
    # One v-t contact: particle 0 (vertex) collides with triangle (1, 2, 3).
    # Barycentric (0.5, 0.3, 0.2). 3 rows: n + t1 + t2.
    pa = np.array([0, 0, 0], dtype=np.int32)
    pb = np.array([1, 1, 1], dtype=np.int32)
    pc = np.array([2, 2, 2], dtype=np.int32)
    pd = np.array([3, 3, 3], dtype=np.int32)
    wa = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    wb = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    wc = np.array([-0.3, -0.3, -0.3], dtype=np.float32)
    wd = np.array([-0.2, -0.2, -0.2], dtype=np.float32)
    # Normal pointing +y; tangents along x and z.
    dirs = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    alphas = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    n_rows = 3

    # ---- Upload to device ----
    pa_d = wp.array(pa, dtype=wp.int32, device=device)
    pb_d = wp.array(pb, dtype=wp.int32, device=device)
    pc_d = wp.array(pc, dtype=wp.int32, device=device)
    pd_d = wp.array(pd, dtype=wp.int32, device=device)
    wa_d = wp.array(wa, dtype=wp.float32, device=device)
    wb_d = wp.array(wb, dtype=wp.float32, device=device)
    wc_d = wp.array(wc, dtype=wp.float32, device=device)
    wd_d = wp.array(wd, dtype=wp.float32, device=device)
    dir_d = wp.array(dirs, dtype=wp.vec3, device=device)
    alpha_d = wp.array(alphas, dtype=wp.float32, device=device)

    # ---- Build BSR Schur via the 4-slot path ----
    W_bsr = ls.build_schur_lite_unified(
        n_rows, pa_d, pb_d, dir_d, alpha_d,
        row_particle_c_d=pc_d, row_particle_d_d=pd_d,
        row_wa_d=wa_d, row_wb_d=wb_d, row_wc_d=wc_d, row_wd_d=wd_d,
    )

    # Extract W as dense fp64 via BSR CSR-like fields.
    def bsr_to_dense(W):
        """Manual BSR → dense for scalar-block (1×1, fp64) matrices."""
        nrow = int(W.nrow)
        ncol = int(W.ncol)
        offsets = W.offsets.numpy()
        cols = W.columns.numpy()
        vals = W.values.numpy()
        out = np.zeros((nrow, ncol), dtype=np.float64)
        for r in range(nrow):
            for k in range(int(offsets[r]), int(offsets[r + 1])):
                c = int(cols[k])
                v = vals[k]
                # ``v`` may be scalar fp64 or a 1×1 matrix depending on dtype.
                out[r, c] = float(v) if np.ndim(v) == 0 else float(v.flat[0])
        return out

    W_actual = bsr_to_dense(W_bsr)

    # ---- NumPy reference ----
    W_ref = numpy_schur_unified(
        pa, pb, pc, pd, wa, wb, wc, wd, dirs, alphas, inv_mass, dt,
    )

    print()
    print("W_actual:")
    print(W_actual)
    print()
    print("W_ref:")
    print(W_ref)
    print()

    diff = np.abs(W_actual - W_ref)
    max_diff = float(diff.max())
    rel_diff = max_diff / max(1e-30, float(np.abs(W_ref).max()))

    print(f"  max abs diff = {max_diff:.3e}")
    print(f"  max rel diff = {rel_diff:.3e}")
    print(f"  |W_ref|_max = {float(np.abs(W_ref).max()):.3e}")

    # ---- Also test a v-v fallback path (pc=pd=-1, wc=wd=0). ----
    pa_vv = np.array([0, 0, 0], dtype=np.int32)
    pb_vv = np.array([1, 1, 1], dtype=np.int32)
    pc_vv = np.array([-1, -1, -1], dtype=np.int32)
    pd_vv = np.array([-1, -1, -1], dtype=np.int32)
    wa_vv = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    wb_vv = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    wc_vv = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    wd_vv = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    pa_d2 = wp.array(pa_vv, dtype=wp.int32, device=device)
    pb_d2 = wp.array(pb_vv, dtype=wp.int32, device=device)
    pc_d2 = wp.array(pc_vv, dtype=wp.int32, device=device)
    pd_d2 = wp.array(pd_vv, dtype=wp.int32, device=device)
    wa_d2 = wp.array(wa_vv, dtype=wp.float32, device=device)
    wb_d2 = wp.array(wb_vv, dtype=wp.float32, device=device)
    wc_d2 = wp.array(wc_vv, dtype=wp.float32, device=device)
    wd_d2 = wp.array(wd_vv, dtype=wp.float32, device=device)

    W_bsr_vv4 = ls.build_schur_lite_unified(
        n_rows, pa_d2, pb_d2, dir_d, alpha_d,
        row_particle_c_d=pc_d2, row_particle_d_d=pd_d2,
        row_wa_d=wa_d2, row_wb_d=wb_d2, row_wc_d=wc_d2, row_wd_d=wd_d2,
    )
    W_vv4 = bsr_to_dense(W_bsr_vv4)

    # Reference: same shape, computed by numpy.
    W_vv_ref = numpy_schur_unified(
        pa_vv, pb_vv, pc_vv, pd_vv, wa_vv, wb_vv, wc_vv, wd_vv,
        dirs, alphas, inv_mass, dt,
    )
    diff_vv = float(np.abs(W_vv4 - W_vv_ref).max())
    print(f"  v-v-as-4slot vs ref: max abs diff = {diff_vv:.3e}")

    # ---- Also test legacy 2-slot path (no 4-slot kwargs) — must match v-v reference. ----
    W_bsr_vv2 = ls.build_schur_lite_unified(
        n_rows, pa_d2, pb_d2, dir_d, alpha_d,
    )
    W_vv2 = bsr_to_dense(W_bsr_vv2)
    diff_vv2 = float(np.abs(W_vv2 - W_vv_ref).max())
    print(f"  legacy 2-slot vs ref: max abs diff = {diff_vv2:.3e}")

    print()
    tol = 1.0e-9
    ok = (max_diff < tol) and (diff_vv < tol) and (diff_vv2 < tol)
    print(f"  Tier 2 (BSR vs NumPy, tol {tol:.0e}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
