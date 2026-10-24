# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NSN inner-loop GPU drivers.

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
    ``min(max_iter, n)`` iterations. At small ``n`` (~10) this caps relative
    accuracy at ~1e-4 (see ``test_nsn_pcr.py::TestPCRReuse``). The
    parity tolerances below scale with ``n`` to reflect that.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import SolverFBA
from newton.tests.fba._reference import fb_frictional_row, fb_unilateral_row, solve_nsn_coulomb, solve_nsn_unilateral


def _make_unilateral_inputs(M: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a well-conditioned SPD ``W`` plus ``(r, pene0)`` for a unilateral problem.

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
    """Build a 3M-row frictional (Coulomb) test problem."""
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
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.5,
        cell_y=0.5,
        mass=1.0,
        tri_ke=500.0,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=1.0,
        edge_kd=0.0,
    )
    model = builder.finalize()
    return SolverFBA(model, iterations=1, friction=friction, lambda_cap=lambda_cap)


class TestNSNUnilateralGpuDriver(unittest.TestCase):
    """Verify the unilateral GPU driver matches the CPU/numpy reference."""

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

        lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_unilateral(
            W, r, pene0, max_iters=3, dt=dt, lambda_cap=solver.lambda_cap
        )
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_unilateral_gpu(W, r, pene0, max_iters=3, dt=dt)

        # n=1: PCR converges in 1 iter so the GPU path matches CPU to fp64 noise.
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_unilateral_gpu_matches_cpu_multi_contact(self):
        """Preserve CPU parity across device/host W switches and capacity growth."""
        solver = _build_minimal_solver(friction=False)
        dt = 0.01

        for M, expected_capacity in ((50, 50), (80, 80)):
            W, r, pene0 = _make_unilateral_inputs(M=M, seed=7)
            W_device = wp.array(W, dtype=wp.float64, device=solver._device)
            lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_unilateral(
                W, r, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap
            )
            for path in ("device", "host", "device"):
                with self.subTest(M=M, path=path):
                    host_w_before = getattr(solver, "_nsn_W_d", None)
                    lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_unilateral_gpu(
                        W if path == "host" else None,
                        r,
                        pene0,
                        max_iters=1,
                        dt=dt,
                        W_device=W_device if path == "device" else None,
                    )

                    # These sizes converge well; preserve the existing PCR tolerances.
                    np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-5)
                    np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
                    np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)
                    if path == "device":
                        # Device W must not create or grow an unused dense host-upload copy.
                        self.assertIs(getattr(solver, "_nsn_W_d", None), host_w_before)
                        self.assertEqual(solver._nsn_inner_n, expected_capacity)
                    else:
                        self.assertGreaterEqual(solver._nsn_W_d.shape[0], M)
                        self.assertGreaterEqual(solver._nsn_W_d.shape[1], M)

    def test_large_diagonal_growth_and_device_host_switch(self):
        """Bound large dense buffers while preserving strided-W solves across growth."""
        solver = _build_minimal_solver(friction=False)
        dt = 0.01

        for M, expected_capacity in ((1000, 1000), (1025, 1152), (1153, 1280)):
            diagonal = np.linspace(1.0, 3.0, M)
            W = np.diag(diagonal)
            r = np.linspace(-0.002, 0.003, M)
            pene0 = 0.0003 * np.sin(np.arange(M))
            omega, compliance, h = np.asarray(
                [fb_unilateral_row(-ri, 0.0, dt * dt * d, dt, p) for ri, d, p in zip(r, diagonal, pene0, strict=True)]
            ).T
            # A diagonal Schur system has an elementwise exact solve, with no O(M^3) reference work.
            lam = (h - omega * (pene0 - r)) / (dt * dt * (omega * omega * diagonal + compliance))
            lam_apply = dt * dt * omega * lam
            W_storage = wp.array(np.pad(W, ((0, 0), (0, 7))), dtype=wp.float64, device=solver._device)
            W_device = W_storage[:, :M]

            for path in ("device", "host", "device"):
                with self.subTest(M=M, path=path):
                    host_w_before = getattr(solver, "_nsn_W_d", None)
                    lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_unilateral_gpu(
                        W if path == "host" else None,
                        r,
                        pene0,
                        dt=dt,
                        W_device=W_device if path == "device" else None,
                    )
                    np.testing.assert_allclose(lam_gpu, lam, rtol=0, atol=1.0e-8)
                    np.testing.assert_allclose(omega_gpu, omega, rtol=0, atol=1.0e-12)
                    np.testing.assert_allclose(lam_apply_gpu, lam_apply, rtol=0, atol=1.0e-12)
                    if path == "device":
                        self.assertIs(getattr(solver, "_nsn_W_d", None), host_w_before)
                    else:
                        self.assertEqual(solver._nsn_W_d.shape, (expected_capacity, expected_capacity))
                    self.assertEqual(solver._nsn_inner_n, expected_capacity)
                    self.assertEqual(solver._nsn_a_schur_d.shape, (expected_capacity, expected_capacity))

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

        lam_cpu, omega_cpu, _ = solve_nsn_unilateral(W, r, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap)
        lam_gpu, omega_gpu, _ = solver._solve_nsn_unilateral_gpu(W, r, pene0, max_iters=1, dt=dt)
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)

        # Now warm-start the next call with the just-computed (lam, omega).
        lam2_cpu, omega2_cpu, _ = solve_nsn_unilateral(
            W,
            r,
            pene0,
            max_iters=1,
            lam_init=lam_cpu,
            omega_init=omega_cpu,
            dt=dt,
            lambda_cap=solver.lambda_cap,
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

        lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_unilateral(
            W, r, pene0, max_iters=2, dt=dt, lambda_cap=solver.lambda_cap
        )
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
    """Verify the Coulomb GPU driver matches the CPU/numpy reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA device required for NSN GPU driver tests")

    def test_sparse_full_storage_matches_dense_and_is_instance_local(self):
        """Sparse storage retains coupled contacts/joints and notices external W updates."""
        dense = _build_minimal_solver(friction=True, lambda_cap=1.0e-5)
        original_method = type(dense)._solve_nsn_coulomb_gpu
        sparse = SolverFBA(dense.model, iterations=1, friction=True, lambda_cap=1.0e-5, nsn_sparse_storage=True)
        self.assertIs(type(dense)._solve_nsn_coulomb_gpu, original_method)
        self.assertNotIn("_solve_nsn_coulomb_gpu", dense.__dict__)
        rng = np.random.default_rng(37)
        m, bilateral = 4, 3
        n = 3 * m + bilateral
        w = np.zeros((n, n))
        for begin, end in ((0, 6), (6, n)):
            a = rng.normal(size=(end - begin, end - begin))
            w[begin:end, begin:end] = a @ a.T / n + np.eye(end - begin)
        r = rng.normal(size=n) * 1.0e-3
        pene = np.zeros(n)
        mu = np.full(m, 0.5)
        device_w = wp.array(w, dtype=wp.float64, device=dense._device)
        for scale in (1.0, 1.3):
            device_w.assign(w * scale)
            kwargs = {
                "dt": 0.01,
                "max_iters": 2,
                "n_bilateral": bilateral,
                "contact_regularization": 0.1,
                "W_device": device_w,
            }
            expected = dense._solve_nsn_coulomb_gpu(None, r, mu, pene, **kwargs)
            actual = sparse._solve_nsn_coulomb_gpu(None, r, mu, pene, **kwargs)
            for got, want in zip(actual, expected, strict=True):
                np.testing.assert_allclose(got, want, rtol=1.0e-6, atol=1.0e-8)
            self.assertFalse(sparse._full_sparse_last_stats["conversion_cached"])
        self.assertIsNone(getattr(sparse, "_nsn_a_schur_d", None))
        self.assertLess(sparse._full_sparse_last_stats["nnz"], n * n)

    def test_coulomb_gpu_matches_cpu_three_contacts(self):
        """Three-contact friction problem (9-row Schur): the PCR n-cap is
        small here so we accept ~1e-6 relative drift, but the structural
        outputs (signs, magnitudes, omega) must still agree closely.
        """
        solver = _build_minimal_solver(friction=True)
        W, r, pene0, mu = _make_coulomb_inputs(M=3, seed=11)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_coulomb(
            W, r, mu, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap
        )
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, max_iters=1, dt=dt)

        # n=9 PCR cap → ~1e-6 relative; lambdas of magnitude ~few → atol 1e-4.
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-4)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_coulomb_gpu_matches_cpu_large(self):
        """Preserve Coulomb parity across device/host W switches and capacity growth."""
        solver = _build_minimal_solver(friction=True)
        dt = 0.01

        for M, expected_capacity in ((20, 60), (30, 91)):
            W, r, pene0, mu = _make_coulomb_inputs(M=M, seed=11)
            W_device = wp.array(W, dtype=wp.float64, device=solver._device)
            lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_coulomb(
                W, r, mu, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap
            )
            for path in ("device", "host", "device"):
                with self.subTest(M=M, path=path):
                    host_w_before = getattr(solver, "_nsn_W_d", None)
                    lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_coulomb_gpu(
                        W if path == "host" else None,
                        r,
                        mu,
                        pene0,
                        max_iters=1,
                        dt=dt,
                        W_device=W_device if path == "device" else None,
                    )

                    np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-5)
                    np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
                    np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)
                    if path == "device":
                        self.assertIs(getattr(solver, "_nsn_W_d", None), host_w_before)
                        self.assertEqual(solver._nsn_inner_n, expected_capacity)
                    else:
                        self.assertGreaterEqual(solver._nsn_W_d.shape[0], 3 * M)
                        self.assertGreaterEqual(solver._nsn_W_d.shape[1], 3 * M)

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

    def test_bilateral_rows_take_newton_step(self):
        """Bilateral rows after the contact rows follow the identity law in the same Schur solve."""
        M, R, dt = 20, 6, 0.01
        n = 3 * M + R
        rng = np.random.default_rng(23)
        aux = rng.standard_normal((n, n))
        W = aux @ aux.T / n + np.eye(n)
        r, pene0, mu = rng.standard_normal(n) * 1.0e-3, np.zeros(n), 0.3 + 0.2 * rng.random(M)
        solver = _build_minimal_solver(friction=True, lambda_cap=1.0e-7)
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, dt=dt, n_bilateral=R)

        # One FB-Newton step from lam = 0, with the identity law on the last R rows.
        precond = dt * np.diag(W) * np.where(np.arange(n) % 3 == 0, dt, 1.0)
        omega, compliance, h = np.ones(n), np.zeros(n), pene0.copy()
        for c in range(M):
            k = 3 * c
            omega[k], compliance[k], h[k] = fb_unilateral_row(-r[k], 0.0, precond[k], dt, 0.0)
            for t in (k + 1, k + 2):
                omega[t], compliance[t], h[t] = fb_frictional_row(-r[t], 0.0, 0.0, mu[c], precond[t], dt, 0.0)
        A_schur = np.outer(omega, omega) * W + np.diag(compliance)
        lam = np.linalg.solve(A_schur, (h - omega * (pene0 - r)) / (dt * dt))

        np.testing.assert_allclose(omega_gpu, omega, rtol=0, atol=1.0e-12)
        np.testing.assert_allclose(lam_gpu[3 * M :], lam[3 * M :], rtol=1.0e-6)
        # The step zeroes the bilateral violation; lambda_cap only clips contact rows.
        np.testing.assert_allclose((-r + W @ (dt * dt * omega * lam))[3 * M :], 0.0, atol=1.0e-12)
        self.assertGreater(np.abs(lam_gpu[3 * M :]).max(), 1.0e-7 / (dt * dt))
        self.assertLessEqual(np.abs(lam_gpu[: 3 * M]).max(), 1.0e-7 / (dt * dt) * (1.0 + 1.0e-12))
        np.testing.assert_allclose(lam_apply_gpu, dt * dt * omega_gpu * lam_gpu, rtol=1.0e-12)

    def test_gpu_path_coulomb_no_contacts(self):
        """Return empty arrays from the Coulomb driver for ``M=0``."""
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
