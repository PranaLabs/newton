# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NSN PCR (Jacobi-preconditioned Conjugate Residual) solver.

The tests compare :class:`~newton._src.solvers.fba.nsn_pcr_solver.NSNPCRSolver`
with ``numpy.linalg.solve`` on SPD matrices of increasing size. Most are
``A = M Mᵀ / n + I`` with a standard normal ``M``, whose condition number is
near 5 for any ``n``; others spread the diagonal over seven decades, add
redundant rows, or add an ill-conditioned block.
"""

from __future__ import annotations

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
    """Identity and diagonal systems, which the Jacobi preconditioner solves exactly."""

    def test_solve_identity_system(self):
        """Solve an identity system exactly within the iteration cap."""
        n = 16
        rng = np.random.default_rng(42)
        A_np = np.eye(n)
        b_np = rng.standard_normal(n)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10)
        # With A = I and Jacobi precond P = I, the first descent direction
        # d = b is the exact solution; PCR converges in one matvec/update.
        # ``NSNPCRSolver`` checks convergence in batches of ``check_every``
        # (default 25) to amortise host syncs, so the iter counter rounds up
        # to the next batch boundary; algorithmically still one descent step.
        # The cap is ``min(max_iter, n) = 16``.
        self.assertLessEqual(iters, n)
        np.testing.assert_allclose(x, b_np, rtol=1.0e-12, atol=1.0e-12)

    def test_solve_diagonal_system(self):
        """Solve a diagonal system exactly, as the Jacobi preconditioner inverts it."""
        n = 32
        rng = np.random.default_rng(43)
        d = rng.uniform(0.5, 5.0, size=n)
        A_np = np.diag(d)
        b_np = rng.standard_normal(n)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10)
        # P = D^-1 perfectly preconditions => 1 iter algorithmically; iter
        # counter rounds up to ``check_every`` boundary (see identity test).
        self.assertLessEqual(iters, 25)
        np.testing.assert_allclose(x, b_np / d, rtol=1.0e-12, atol=1.0e-12)


class TestPCRConvergence(_PCRTestCase):
    """Random SPD matrix tests at NSN-relevant sizes."""

    def test_solve_random_spd_small(self):
        """Solve a random 10 x 10 SPD system to 1e-12 relative within its iteration cap of n."""
        n = 10
        A_np = _make_spd(n, seed=1)
        b_np = np.random.default_rng(2).standard_normal(n)
        x_ref = np.linalg.solve(A_np, b_np)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10, max_iter=200)
        # The solver caps its iteration count at ``min(max_iter, n)``; in exact
        # arithmetic, n iterations of conjugate residuals reach the solution.
        self.assertLessEqual(iters, n)
        rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        self.assertLess(rel_err, 1.0e-12)

    def test_solve_random_spd_large(self):
        """Solve a random 300 x 300 SPD system, the size of a frictionless solve with 300 contacts, to 1e-7 relative."""
        n = 300
        A_np = _make_spd(n, seed=3)
        b_np = np.random.default_rng(4).standard_normal(n)
        x_ref = np.linalg.solve(A_np, b_np)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-10, max_iter=1000)
        self.assertLess(iters, 1000)
        rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        self.assertLess(rel_err, 1.0e-7)

    def test_solve_3M_size(self):
        """Solve a random 900 x 900 SPD system, the size of a frictional solve with 300 contacts, to 1e-7 relative."""
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
        """End a converged solve with a residual norm strictly below ``||b||``.

        The preconditioned iteration need not decrease ``||r||`` at every step, but the final residual of a converged
        solve is smaller than the initial one.
        """
        n = 100
        A_np = _make_spd(n, seed=7)
        b_np = np.random.default_rng(8).standard_normal(n)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-8, max_iter=500)
        r_final = A_np @ x - b_np
        self.assertLess(np.linalg.norm(r_final), np.linalg.norm(b_np))
        self.assertGreater(iters, 0)

    def test_solve_convergence_count_reasonable(self):
        """Converge on a well-conditioned 100 x 100 SPD system in fewer than 50 iterations."""
        n = 100
        A_np = _make_spd(n, seed=9)
        b_np = np.random.default_rng(10).standard_normal(n)
        x_ref = np.linalg.solve(A_np, b_np)
        x, iters, _ = self._solve(A_np, b_np, tol=1.0e-8, max_iter=500)
        self.assertLess(iters, 50)
        rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        self.assertLess(rel_err, 1.0e-5)

    def test_solve_zero_rhs_returns_zero(self):
        """Return ``x = 0`` without iterating for ``b = 0``.

        The solver checks ``dot(b, b) == 0`` before it iterates.
        """
        n = 50
        A_np = _make_spd(n, seed=11)
        b_np = np.zeros(n)
        x, iters, rho = self._solve(A_np, b_np)
        np.testing.assert_array_equal(x, np.zeros(n))
        self.assertEqual(iters, 0)
        self.assertEqual(rho, 0.0)


class TestPCRScaledAndSingular(_PCRTestCase):
    """Systems with a widely spread diagonal, and with redundant rows."""

    def test_scaled_diagonal_converges(self):
        """Converge on a well-conditioned matrix rescaled by a diagonal that spans seven decades."""
        n = 120
        rng = np.random.default_rng(7)
        scale = np.sqrt(10.0 ** rng.uniform(-4.0, 3.0, n))
        A = scale[:, None] * _make_spd(n, seed=7) * scale[None, :]
        b = rng.standard_normal(n)
        x, _, _ = self._solve(A, b)
        self.assertLess(np.linalg.norm(b - A @ x) / np.linalg.norm(b), 1.0e-8)

    def test_sparse_scaled_diagonal_converges(self):
        """Solve the same system given as a BSR matrix, as the lite NSN builds it, with ``solve_sparse`` to the dense
        solve's result."""
        import warp.sparse as wps  # noqa: PLC0415

        n = 120
        rng = np.random.default_rng(7)
        scale = np.sqrt(10.0 ** rng.uniform(-4.0, 3.0, n))
        A = scale[:, None] * _make_spd(n, seed=7) * scale[None, :]
        b = rng.standard_normal(n)
        rows, cols = (wp.array(a, dtype=wp.int32, device=self.device) for a in np.nonzero(np.ones((n, n))))
        values = wp.array(A.ravel(), dtype=wp.float64, device=self.device)
        W = wps.bsr_from_triplets(n, n, rows, cols, values)
        x = wp.zeros(n, dtype=wp.float64, device=self.device)
        solver = NSNPCRSolver(max_n=n, device=self.device, tol=1.0e-10, max_iter=1000)
        solver.solve_sparse(W, wp.array(b, dtype=wp.float64, device=self.device), x)
        self.assertLess(np.linalg.norm(b - A @ x.numpy()) / np.linalg.norm(b), 1.0e-8)
        np.testing.assert_allclose(x.numpy(), self._solve(A, b)[0], rtol=1.0e-6, atol=1.0e-12)

    def test_near_null_directions_are_refused(self):
        """Keep the iterate bounded on redundant rows with an inconsistent right-hand side.

        Its gain and residual stay within three times those of the least-squares solution.
        """
        n, k = 120, 80
        rng = np.random.default_rng(11)
        J = rng.standard_normal((n, k))
        A = J @ J.T + 1.0e-11 * np.eye(n)
        b = A @ rng.standard_normal(n) + 1.0e-3 * rng.standard_normal(n)
        x, _, _ = self._solve(A, b)
        c = np.sqrt(np.diag(A))
        x_ls = np.linalg.lstsq(A, b, rcond=1.0e-9)[0]
        # Chart gain |C x| / |C^-1 b| of the bounded least-squares solution, and at most a few times it.
        gain = np.linalg.norm(c * x) / np.linalg.norm(b / c)
        self.assertLess(gain, 3.0 * np.linalg.norm(c * x_ls) / np.linalg.norm(b / c))
        self.assertLess(np.linalg.norm(b - A @ x), 3.0 * np.linalg.norm(b - A @ x_ls))


class TestPCRBlockPreconditioner(_PCRTestCase):
    def test_block_preconditioner_cuts_iterations(self):
        """Halve the iterations with an exact block preconditioner on an ill-conditioned trailing block, as for joint
        rows."""
        n, m = 80, 20
        rng = np.random.default_rng(13)
        A = _make_spd(n, seed=13)
        Q = np.linalg.qr(rng.standard_normal((m, m)))[0]
        A[n - m :, n - m :] += Q @ np.diag(10.0 ** np.linspace(-2.0, 3.0, m)) @ Q.T
        b = rng.standard_normal(n)
        block_inv = np.linalg.inv(A[n - m :, n - m :])

        def block_precond(v, out):
            values = out.numpy()
            values[n - m :] = block_inv @ v.numpy()[n - m : n]
            out.assign(values)

        iterations = []
        for precond in (None, block_precond):
            solver = NSNPCRSolver(max_n=n, device=self.device, tol=1.0e-10, max_iter=1000, check_every=1)
            x = wp.zeros(n, dtype=wp.float64, device=self.device)
            A_d = wp.array2d(A, dtype=wp.float64, device=self.device)
            iters, _ = solver.solve(A_d, wp.array(b, dtype=wp.float64, device=self.device), x, block_precond=precond)
            self.assertLess(np.linalg.norm(b - A @ x.numpy()) / np.linalg.norm(b), 1.0e-9)
            iterations.append(iters)
        self.assertLess(iterations[1], 0.5 * iterations[0])


class TestPCRReuse(_PCRTestCase):
    """Solver buffers persist across calls (NSN reuses the same solver)."""

    def test_solver_can_be_reused(self):
        """Reuse one solver for systems of 10, 100, 200 and 50 rows, each solved to its accuracy."""
        max_n = 200
        solver = NSNPCRSolver(max_n=max_n, device=self.device, tol=1.0e-10, max_iter=500)
        # n=10 runs its cap of n iterations, which reach the solution up to
        # rounding; the larger systems stop at the residual tolerance.
        rel_tol_by_size = {10: 1.0e-12, 50: 1.0e-7, 100: 1.0e-7, 200: 1.0e-7}
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

    def test_reinterpreted_matrix_buffer(self):
        """Solve a matrix view with the same pointer and size as before but other strides without reusing the cached
        graph."""
        n, cap = 40, 64
        solver = NSNPCRSolver(max_n=n, device=self.device, tol=1.0e-12, max_iter=500)
        buf = wp.zeros((cap, cap), dtype=wp.float64, device=self.device)
        b_np = np.random.default_rng(3).standard_normal(n)
        b = wp.from_numpy(b_np, dtype=wp.float64, device=self.device)
        x_out = wp.zeros(n, dtype=wp.float64, device=self.device)
        for k, strides in enumerate(((cap * 8, 8), (n * 8, 8))):
            A_np = _make_spd(n, seed=300 + k)
            flat = buf.numpy().ravel()
            rows = np.arange(n)[:, None] * (strides[0] // 8) + np.arange(n)[None, :]
            flat[rows] = A_np
            buf.assign(flat.reshape(cap, cap))
            A = wp.array(ptr=buf.ptr, dtype=wp.float64, shape=(n, n), strides=strides, device=self.device)
            solver.solve(A, b, x_out)
            np.testing.assert_allclose(x_out.numpy(), np.linalg.solve(A_np, b_np), rtol=1.0e-6)

    def test_invalid_size_raises(self):
        """Refuse a system larger than the solver's maximum size."""
        solver = NSNPCRSolver(max_n=10, device=self.device, tol=1.0e-8, max_iter=50)
        A = wp.zeros((20, 20), dtype=wp.float64, device=self.device)
        b = wp.zeros(20, dtype=wp.float64, device=self.device)
        x = wp.zeros(20, dtype=wp.float64, device=self.device)
        with self.assertRaises(ValueError):
            solver.solve(A, b, x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
