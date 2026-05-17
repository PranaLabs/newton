# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NSN inner-loop GPU drivers (Steps 6-10 of the NSN GPU port).

Verifies that
:meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._solve_nsn_unilateral_gpu`
and :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._solve_nsn_coulomb_gpu`
reproduce the corresponding CPU/numpy drivers, plus a handful of edge cases
(lambda_cap, signed-cone Coulomb clamp, M=0).

Tolerance notes:
    The CPU reference solves ``A_schur · dlam = rhs`` with
    :func:`numpy.linalg.solve` (LU factorisation, ~fp64 noise). The GPU
    driver uses :class:`NSNPCRSolver` (Jacobi-preconditioned Conjugate
    Residual), which is theoretically exact in ``n`` steps but capped at
    ``min(max_iter, n)`` iterations -- see RealSim
    ``CUDADenseCRSolver.cpp:126``. At small ``n`` (~10) this caps relative
    accuracy at ~1e-4 (see ``test_fba_nsn_pcr.py::TestPCRReuse``). The
    parity tolerances below scale with ``n`` to reflect that.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import SolverFBA


def _make_unilateral_inputs(M: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a well-conditioned SPD ``W`` plus ``(r, pene0)`` for Stage A.

    Picks ``W = M_aux @ M_auxᵀ / M + I`` (matches the regime used by the
    NSN PCR tests) and standard-normal ``r`` / ``pene0`` scaled to typical
    contact magnitudes (~1e-3 m).
    """
    rng = np.random.default_rng(seed)
    aux = rng.standard_normal((M, M))
    W = (aux @ aux.T) / float(M) + np.eye(M)
    r = rng.standard_normal(M) * 1.0e-3
    pene0 = rng.standard_normal(M) * 1.0e-3
    return W.astype(np.float64), r.astype(np.float64), pene0.astype(np.float64)


def _make_coulomb_inputs(M: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a 3M-row Stage B test problem."""
    rng = np.random.default_rng(seed)
    aux = rng.standard_normal((3 * M, 3 * M))
    W = (aux @ aux.T) / float(3 * M) + np.eye(3 * M)
    r = rng.standard_normal(3 * M) * 1.0e-3
    pene0 = rng.standard_normal(3 * M) * 1.0e-3
    mu = 0.3 + 0.2 * rng.random(M)  # ~0.3..0.5
    return W.astype(np.float64), r.astype(np.float64), pene0.astype(np.float64), mu.astype(np.float64)


def _build_minimal_solver(friction: bool, lambda_cap: float | None = None) -> SolverFBA:
    """Construct a 2x2 cloth model so :class:`SolverFBA` initialises.

    The GPU NSN drivers operate purely on host-supplied ``W`` / ``r`` /
    ``pene0`` / ``mu`` arrays — the surrounding solver state (positions,
    pin energy, etc.) is irrelevant for these tests. We just need the
    constructor to succeed and have a Warp device wired up.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.5,
        cell_y=0.5,
        mass=1.0,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=1.0,
        edge_kd=0.0,
    )
    model = builder.finalize()
    return SolverFBA(model, iterations=1, friction=friction, lambda_cap=lambda_cap)


class TestNSNUnilateralGpuDriver(unittest.TestCase):
    """Verify the Stage A GPU driver matches the CPU/numpy reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA device required for NSN GPU driver tests")

    def test_unilateral_gpu_matches_cpu_single_contact(self):
        """One-contact problem: PCR exact in 1 step, full fp64 parity."""
        solver = _build_minimal_solver(friction=False)
        W, r, pene0 = _make_unilateral_inputs(M=1, seed=42)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solver._solve_nsn_unilateral(W, r, pene0, max_iters=3, dt=dt)
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_unilateral_gpu(W, r, pene0, max_iters=3, dt=dt)

        # n=1: PCR converges in 1 iter so the GPU path matches CPU to fp64 noise.
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_unilateral_gpu_matches_cpu_multi_contact(self):
        """Larger Schur block (M=50) where PCR converges well.

        At ``n=50`` PCR reaches ~1e-7 relative accuracy (see
        ``test_fba_nsn_pcr.TestPCRReuse``), so the GPU vs CPU drivers
        agree at fp64-noise level on a single FB-Newton iter.
        """
        solver = _build_minimal_solver(friction=False)
        W, r, pene0 = _make_unilateral_inputs(M=50, seed=7)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solver._solve_nsn_unilateral(W, r, pene0, max_iters=1, dt=dt)
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_unilateral_gpu(W, r, pene0, max_iters=1, dt=dt)

        # PCR relative accuracy ~1e-7 at n=50; absolute lambda magnitudes ~O(1),
        # so an atol of 1e-5 covers PCR convergence + cumulative FB-row error.
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-5)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_warm_start_persistence_gpu(self):
        """Two consecutive calls with warm start ω/λ from the prior iter
        must agree with the CPU reference (i.e. the GPU buffers do not
        corrupt or leak state across calls).

        Uses ``M=1`` so PCR is exact in one step — isolates the
        warm-start plumbing from PCR convergence noise.
        """
        solver = _build_minimal_solver(friction=False)
        W, r, pene0 = _make_unilateral_inputs(M=1, seed=123)
        dt = 0.01

        lam_cpu, omega_cpu, _ = solver._solve_nsn_unilateral(W, r, pene0, max_iters=1, dt=dt)
        lam_gpu, omega_gpu, _ = solver._solve_nsn_unilateral_gpu(W, r, pene0, max_iters=1, dt=dt)
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)

        # Now warm-start the next call with the just-computed (lam, omega).
        lam2_cpu, omega2_cpu, _ = solver._solve_nsn_unilateral(
            W, r, pene0, max_iters=1, lam_init=lam_cpu, omega_init=omega_cpu, dt=dt
        )
        lam2_gpu, omega2_gpu, _ = solver._solve_nsn_unilateral_gpu(
            W, r, pene0, max_iters=1, lam_init=lam_gpu, omega_init=omega_gpu, dt=dt
        )
        np.testing.assert_allclose(lam2_gpu, lam2_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(omega2_gpu, omega2_cpu, rtol=0, atol=1.0e-8)

    def test_lambda_cap_internal_dt2_gpu(self):
        """``lambda_cap`` (physical N) must be divided by ``dt²`` on GPU
        before the symmetric clip, matching the CPU implementation.

        Uses ``M=1`` so PCR is exact and the only source of CPU/GPU
        divergence would be the cap clip itself. The seed is chosen so
        the unconstrained lambda is large enough that the cap engages.
        """
        lambda_cap = 1.0e-4  # cap_internal = 1.0
        solver = _build_minimal_solver(friction=False, lambda_cap=lambda_cap)
        # Use a problem where r > 0 (penetration) so the FB-Newton step
        # produces a non-zero lambda. ``seed=11`` gives unclipped |lam| ~ 9.
        rng = np.random.default_rng(11)
        M = 1
        aux = rng.standard_normal((M, M))
        W = (aux @ aux.T) / float(M) + np.eye(M)
        r = np.abs(rng.standard_normal(M)) * 1.0e-3  # positive penetration
        pene0 = rng.standard_normal(M) * 1.0e-3
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solver._solve_nsn_unilateral(W, r, pene0, max_iters=2, dt=dt)
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_unilateral_gpu(W, r, pene0, max_iters=2, dt=dt)

        cap_internal = lambda_cap / (dt * dt)
        # Cap engaged: max(|lam|) saturates at cap_internal.
        self.assertLessEqual(np.max(np.abs(lam_gpu)), cap_internal + 1.0e-9)
        self.assertAlmostEqual(np.max(np.abs(lam_gpu)), cap_internal, delta=1.0e-9)
        # And matches CPU at fp64 noise (M=1: PCR exact in 1 iter).
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_gpu_path_unilateral_no_contacts(self):
        """``M=0`` early-return: GPU driver returns three empty fp64 arrays."""
        solver = _build_minimal_solver(friction=False)
        W = np.zeros((0, 0), dtype=np.float64)
        r = np.zeros(0, dtype=np.float64)
        pene0 = np.zeros(0, dtype=np.float64)
        lam, omega, lam_apply = solver._solve_nsn_unilateral_gpu(W, r, pene0, dt=0.01)
        self.assertEqual(lam.shape, (0,))
        self.assertEqual(omega.shape, (0,))
        self.assertEqual(lam_apply.shape, (0,))


class TestNSNCoulombGpuDriver(unittest.TestCase):
    """Verify the Stage B GPU driver matches the CPU/numpy reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA device required for NSN GPU driver tests")

    def test_coulomb_gpu_matches_cpu_three_contacts(self):
        """Three-contact friction problem (9-row Schur): the PCR n-cap is
        small here so we accept ~1e-6 relative drift, but the structural
        outputs (signs, magnitudes, omega) must still agree closely.
        """
        solver = _build_minimal_solver(friction=True)
        W, r, pene0, mu = _make_coulomb_inputs(M=3, seed=11)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solver._solve_nsn_coulomb(W, r, mu, pene0, max_iters=1, dt=dt)
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, max_iters=1, dt=dt)

        # n=9 PCR cap → ~1e-6 relative; lambdas of magnitude ~few → atol 1e-4.
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-4)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_coulomb_gpu_matches_cpu_large(self):
        """Larger Schur block where PCR converges well (n=60, M=20)."""
        solver = _build_minimal_solver(friction=True)
        W, r, pene0, mu = _make_coulomb_inputs(M=20, seed=11)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solver._solve_nsn_coulomb(W, r, mu, pene0, max_iters=1, dt=dt)
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, max_iters=1, dt=dt)

        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-5)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_box_clamp_signed_lam_n_gpu(self):
        """Verify the device signed-cone clamp invariant: ``|lam_t| <=
        |mu · lam_n|`` after clamp, matching the CPU loop's two-if-
        statement saturation."""
        solver = _build_minimal_solver(friction=True)
        # Use a seeded problem that exercises both positive and negative
        # ``lam_n`` during the inner iterates. Two iters give the FB Newton
        # step enough room to drive ``lam_n`` past zero on some rows.
        W, r, pene0, mu = _make_coulomb_inputs(M=4, seed=19)
        dt = 0.01

        lam_gpu, _omega_gpu, _lam_apply_gpu = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, max_iters=2, dt=dt)

        # After clamp, |lam_t| <= |mu * lam_n| per row.
        for c in range(4):
            lam_n = lam_gpu[3 * c]
            bound = abs(mu[c] * lam_n)
            self.assertLessEqual(abs(lam_gpu[3 * c + 1]), bound + 1.0e-9)
            self.assertLessEqual(abs(lam_gpu[3 * c + 2]), bound + 1.0e-9)

    def test_gpu_path_coulomb_no_contacts(self):
        """``M=0`` early-return for Stage B."""
        solver = _build_minimal_solver(friction=True)
        W = np.zeros((0, 0), dtype=np.float64)
        r = np.zeros(0, dtype=np.float64)
        pene0 = np.zeros(0, dtype=np.float64)
        mu = np.zeros(0, dtype=np.float64)
        lam, omega, lam_apply = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, dt=0.01)
        self.assertEqual(lam.shape, (0,))
        self.assertEqual(omega.shape, (0,))
        self.assertEqual(lam_apply.shape, (0,))


if __name__ == "__main__":
    unittest.main()
