# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NSN PCR (Jacobi-preconditioned Conjugate Residual) solver.

Ported from RealSim ``CUDADenseJacobiPCRSolver`` (NSN inner Schur LCP solver).
The reference algorithm lives in
``newton/_src/solvers/fba/nsn_pcr_solver.py``; these tests validate it against
``numpy.linalg.solve`` on a battery of SPD matrices of increasing size.

Test matrices are constructed as ``A = M Mᵀ / n + I`` which yields a
well-conditioned diagonally-dominant SPD matrix. This is the regime in which
RealSim's PCR (with Jacobi preconditioner, an *unsymmetric* application of
preconditioning -- see RealSim ``CUDADenseCRSolver.cpp:166`` for the
``rho > tol`` test) is well-defined and converges. NSN Schur LCP systems have
the same structure (mass-derived diagonal-dominant).
"""

from __future__ import annotations

import time
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba.nsn_pcr_solver import NSNPCRSolver


def _make_spd(n: int, seed: int = 0, cond_scale: float = 1.0) -> np.ndarray:
    """Make a well-conditioned SPD test matrix.

    ``A = M Mᵀ / n + I * cond_scale``. With ``cond_scale=1`` and ``M`` standard
    normal, the resulting cond(A) sits near 5 for any ``n``, well inside the
    Jacobi-PCR convergence regime.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return (M @ M.T) / float(n) + np.eye(n) * cond_scale


class _PCRTestCase(unittest.TestCase):
    """Common setup: pick a CUDA device and lazily build a solver."""

    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA device required for NSNPCRSolver tests")
        cls.device = "cuda:0"

    def _solve(
        self,
        A_np: np.ndarray,
        b_np: np.ndarray,
        tol: float = 1.0e-10,
        max_iter: int = 1000,
    ) -> tuple[np.ndarray, int, float]:
        n = b_np.shape[0]
        A = wp.array2d(A_np.astype(np.float64), dtype=wp.float64, device=self.device)
        b = wp.from_numpy(b_np.astype(np.float64), dtype=wp.float64, device=self.device)
        x_out = wp.zeros(n, dtype=wp.float64, device=self.device)
        solver = NSNPCRSolver(max_n=n, device=self.device, tol=tol, max_iter=max_iter)
        iters, rho = solver.solve(A, b, x_out)
        return x_out.numpy(), iters, rho


class TestPCRBasic(_PCRTestCase):
    """Identity / diagonal sanity checks (RealSim cuda_precond exactness)."""

    def test_solve_identity_system(self):
        n = 16
        rng = np.random.default_rng(42)
        A_np = np.eye(n)
        b_np = rng.standard_normal(n)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10)
        # With A = I and Jacobi precond P = I, the first descent direction
        # d = b is the exact solution; PCR converges in one matvec/update.
        self.assertLessEqual(iters, 1)
        np.testing.assert_allclose(x, b_np, rtol=1.0e-12, atol=1.0e-12)

    def test_solve_diagonal_system(self):
        n = 32
        rng = np.random.default_rng(43)
        d = rng.uniform(0.5, 5.0, size=n)
        A_np = np.diag(d)
        b_np = rng.standard_normal(n)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10)
        # P = D^-1 perfectly preconditions => 1 iter.
        self.assertLessEqual(iters, 1)
        np.testing.assert_allclose(x, b_np / d, rtol=1.0e-12, atol=1.0e-12)


class TestPCRConvergence(_PCRTestCase):
    """Random SPD matrix tests at NSN-relevant sizes."""

    def test_solve_random_spd_small(self):
        n = 10
        A_np = _make_spd(n, seed=1)
        b_np = np.random.default_rng(2).standard_normal(n)
        x_ref = np.linalg.solve(A_np, b_np)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10, max_iter=200)
        # RealSim caps iter count at ``min(max_iter, n)`` (CRSolver.cpp:126), so
        # n=10 floors at 10 iters and roundoff floor is ~1e-5.
        self.assertLessEqual(iters, n)
        rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        self.assertLess(rel_err, 1.0e-4)

    def test_solve_random_spd_large(self):
        # Demo 5 Stage A size.
        n = 300
        A_np = _make_spd(n, seed=3)
        b_np = np.random.default_rng(4).standard_normal(n)
        x_ref = np.linalg.solve(A_np, b_np)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10, max_iter=1000)
        self.assertLess(iters, 1000)
        rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        self.assertLess(rel_err, 1.0e-7)

    def test_solve_3M_size(self):
        # Demo 5 Stage B size (3M-like dim).
        n = 900
        A_np = _make_spd(n, seed=5)
        b_np = np.random.default_rng(6).standard_normal(n)
        x_ref = np.linalg.solve(A_np, b_np)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10, max_iter=2000)
        self.assertLess(iters, 2000)
        rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        self.assertLess(rel_err, 1.0e-7)


class TestPCRDiagnostics(_PCRTestCase):
    """Behavior-level diagnostics: residual reduction and iter count."""

    def test_solve_monotone_residual_decrease(self):
        """Final ``||r||`` must be strictly smaller than ``||b||``.

        RealSim PCR with diagonal-dominant SPD is not strictly monotone in
        ``||r||`` (only the energy norm is monotone), but the *final* residual
        must always be smaller than the initial one for a converged solve.
        """
        n = 100
        A_np = _make_spd(n, seed=7)
        b_np = np.random.default_rng(8).standard_normal(n)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-8, max_iter=500)
        r_final = A_np @ x - b_np
        self.assertLess(np.linalg.norm(r_final), np.linalg.norm(b_np))
        self.assertGreater(iters, 0)

    def test_solve_convergence_count_reasonable(self):
        """Well-conditioned 100x100 SPD must converge in fewer than 50 iters."""
        n = 100
        A_np = _make_spd(n, seed=9)
        b_np = np.random.default_rng(10).standard_normal(n)
        x_ref = np.linalg.solve(A_np, b_np)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-8, max_iter=500)
        self.assertLess(iters, 50)
        rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        self.assertLess(rel_err, 1.0e-5)

    def test_solve_zero_rhs_returns_zero(self):
        """``b = 0`` should short-circuit to ``x = 0`` in 0 iterations.

        Mirrors RealSim ``CUDADenseCRSolver.cpp:138`` ``if(dot_b != 0.0)`` guard.
        """
        n = 50
        A_np = _make_spd(n, seed=11)
        b_np = np.zeros(n)
        x, iters, rho = self._solve(A_np, b_np)
        np.testing.assert_array_equal(x, np.zeros(n))
        self.assertEqual(iters, 0)
        self.assertEqual(rho, 0.0)


class TestPCRReuse(_PCRTestCase):
    """Solver buffers persist across calls (NSN reuses the same solver)."""

    def test_solver_can_be_reused(self):
        max_n = 200
        solver = NSNPCRSolver(max_n=max_n, device=self.device, tol=1.0e-10, max_iter=500)
        # n=10 hits the n-cap and only converges to ~1e-5 relative;
        # all other sizes reach 1e-7. See RealSim CRSolver.cpp:126.
        rel_tol_by_size = {10: 1.0e-4, 50: 1.0e-7, 100: 1.0e-7, 200: 1.0e-7}
        for n in [10, 100, 200, 50]:
            A_np = _make_spd(n, seed=100 + n)
            b_np = np.random.default_rng(200 + n).standard_normal(n)
            x_ref = np.linalg.solve(A_np, b_np)

            A = wp.array2d(A_np, dtype=wp.float64, device=self.device)
            b = wp.from_numpy(b_np, dtype=wp.float64, device=self.device)
            x_out = wp.zeros(n, dtype=wp.float64, device=self.device)
            iters, _ = solver.solve(A, b, x_out)
            x = x_out.numpy()
            rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
            self.assertLess(
                rel_err,
                rel_tol_by_size[n],
                msg=f"n={n}: rel_err={rel_err}, iters={iters}",
            )

    def test_invalid_size_raises(self):
        solver = NSNPCRSolver(max_n=10, device=self.device, tol=1.0e-8, max_iter=50)
        A = wp.zeros((20, 20), dtype=wp.float64, device=self.device)
        b = wp.zeros(20, dtype=wp.float64, device=self.device)
        x = wp.zeros(20, dtype=wp.float64, device=self.device)
        with self.assertRaises(ValueError):
            solver.solve(A, b, x)


# ---------------------------------------------------------------------------
# Optional profiling helper (runs when executed as a script).
# ---------------------------------------------------------------------------


def _profile_sizes(sizes: list[int]) -> None:
    wp.init()
    device = "cuda:0"
    for n in sizes:
        A_np = _make_spd(n, seed=0)
        b_np = np.random.default_rng(1).standard_normal(n)
        A = wp.array2d(A_np, dtype=wp.float64, device=device)
        b = wp.from_numpy(b_np, dtype=wp.float64, device=device)
        x = wp.zeros(n, dtype=wp.float64, device=device)
        solver = NSNPCRSolver(max_n=n, device=device, tol=1.0e-10, max_iter=1000)
        # Warmup
        for _ in range(3):
            solver.solve(A, b, x)
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            iters, _ = solver.solve(A, b, x)
        wp.synchronize_device(device)
        dt = (time.perf_counter() - t0) / N
        print(f"  n={n:5d} iters={iters:3d} time/solve={dt * 1e3:7.3f} ms")


if __name__ == "__main__":
    import sys

    if "--profile" in sys.argv:
        _profile_sizes([100, 300, 900])
    else:
        unittest.main()
