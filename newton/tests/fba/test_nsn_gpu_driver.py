# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NSN inner-loop GPU drivers.

Verifies that
:meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._solve_nsn_unilateral_gpu`
and :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._solve_nsn_coulomb_gpu`
reproduce the corresponding CPU/numpy drivers, plus a handful of edge cases
(lambda_cap, Coulomb box clamp, M=0).

Tolerance notes:
    The host reference solves ``A_schur · dlam = rhs`` with
    :func:`numpy.linalg.solve`, and the device driver with
    :class:`NSNPCRSolver` in fp64. On these systems, whose condition number
    is near 5, the PCR iterations the driver runs reach the solution up to
    rounding, so the device and host multipliers agree to 1e-10.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import SolverFBA
from newton.tests.fba._reference import fb_unilateral_row, solve_nsn_coulomb, solve_nsn_unilateral


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
    ``pene0`` / ``mu`` arrays. The surrounding solver state (positions,
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
        """Match the host unilateral NSN step on one pushing contact, where PCR is exact in one step."""
        solver = _build_minimal_solver(friction=False)
        W, r, pene0 = _make_unilateral_inputs(M=1, seed=1)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_unilateral(
            W, r, pene0, max_iters=3, dt=dt, lambda_cap=solver.lambda_cap
        )
        self.assertGreater(lam_cpu[0], 1.0)  # the contact pushes
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_unilateral_gpu(W, r, pene0, max_iters=3, dt=dt)

        # n=1: PCR converges in 1 iter so the GPU path matches CPU to fp64 noise.
        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_unilateral_gpu_matches_cpu_multi_contact(self):
        """Match the host unilateral NSN step on a 50-contact Schur block to 1e-10."""
        solver = _build_minimal_solver(friction=False)
        W, r, pene0 = _make_unilateral_inputs(M=50, seed=7)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_unilateral(
            W, r, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap
        )
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_unilateral_gpu(W, r, pene0, max_iters=1, dt=dt)

        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_device_w_allocates_no_upload_buffer(self):
        """Take ``W`` from the device without allocating the buffer that uploads a host ``W``, which a host ``W``
        allocates at the scratch capacity; both match the host reference."""
        solver = _build_minimal_solver(friction=False)
        W, r, pene0 = _make_unilateral_inputs(M=50, seed=7)
        dt = 0.01
        expected = solve_nsn_unilateral(W, r, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap)
        W_device = wp.array(W, dtype=wp.float64, device=solver._device)
        upload = None
        for path in ("device", "host", "device"):
            with self.subTest(path=path):
                got = solver._solve_nsn_unilateral_gpu(
                    W if path == "host" else None,
                    r,
                    pene0,
                    max_iters=1,
                    dt=dt,
                    W_device=W_device if path == "device" else None,
                )
                for g, e, atol in zip(got, expected, (1.0e-10, 1.0e-8, 1.0e-8), strict=True):
                    np.testing.assert_allclose(g, e, rtol=0, atol=atol)
                if path == "host":
                    upload = solver._nsn_W_d
                    self.assertEqual(upload.shape, (solver._nsn_inner_n, solver._nsn_inner_n))
                self.assertIs(getattr(solver, "_nsn_W_d", None), upload)

    def test_large_scratch_grows_by_five_percent(self):
        """Grow the dense scratch past 1024 rows by 5% in multiples of 128 rows, and solve a diagonal ``W`` given in
        a wider device array, or on the host, to its elementwise solution."""
        solver = _build_minimal_solver(friction=False)
        dt = 0.01
        for M, capacity in ((1000, 1000), (1025, 1152), (1153, 1280)):
            diagonal = np.linspace(1.0, 3.0, M)
            r = np.linspace(-0.002, 0.003, M)
            pene0 = 0.0003 * np.sin(np.arange(M))
            rows = [
                fb_unilateral_row(-ri, 0.0, dt * dt * d, dt, p) for ri, d, p in zip(r, diagonal, pene0, strict=True)
            ]
            omega, compliance, h = np.asarray(rows).T
            lam = (h - omega * (pene0 - r)) / (dt * dt * (omega * omega * diagonal + compliance))
            W_wide = wp.array(np.pad(np.diag(diagonal), ((0, 0), (0, 7))), dtype=wp.float64, device=solver._device)
            for path in ("device", "host"):
                with self.subTest(M=M, path=path):
                    got = solver._solve_nsn_unilateral_gpu(
                        np.diag(diagonal) if path == "host" else None,
                        r,
                        pene0,
                        dt=dt,
                        W_device=W_wide[:, :M] if path == "device" else None,
                    )
                    for g, e in zip(got, (lam, omega, dt * dt * omega * lam), strict=True):
                        np.testing.assert_allclose(g, e, rtol=0, atol=1.0e-8)
                    self.assertEqual(solver._nsn_inner_n, capacity)
                    self.assertEqual(solver._nsn_a_schur_d.shape, (capacity, capacity))

    def test_warm_start_persistence_gpu(self):
        """Match the host reference over two calls on one pushing contact, the second warm-started from the first.

        The device buffers neither corrupt nor leak state across calls. One contact makes PCR exact in one step, which
        isolates the warm start.
        """
        solver = _build_minimal_solver(friction=False)
        W, r, pene0 = _make_unilateral_inputs(M=1, seed=6)
        dt = 0.01

        lam_cpu, omega_cpu, _ = solve_nsn_unilateral(W, r, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap)
        self.assertGreater(lam_cpu[0], 1.0)  # the contact pushes
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
        """Divide ``lambda_cap`` [N] by ``dt²`` before the symmetric clip on the device, as the host reference does.

        One contact makes PCR exact, so only the clip could make the device and the host differ. The seed makes the
        unconstrained multiplier large enough for the cap to engage.
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
        """Return three empty fp64 arrays from the unilateral driver for ``M=0``."""
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

    def test_coulomb_gpu_matches_cpu_three_contacts(self):
        """Match the host Coulomb NSN step on three contacts, a 9-row Schur block, to 1e-10."""
        solver = _build_minimal_solver(friction=True)
        W, r, pene0, mu = _make_coulomb_inputs(M=3, seed=11)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_coulomb(
            W, r, mu, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap
        )
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, max_iters=1, dt=dt)

        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_coulomb_gpu_matches_cpu_large(self):
        """Match the host Coulomb NSN step on 20 contacts, a 60-row Schur block, to 1e-10."""
        solver = _build_minimal_solver(friction=True)
        W, r, pene0, mu = _make_coulomb_inputs(M=20, seed=11)
        dt = 0.01

        lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_coulomb(
            W, r, mu, pene0, max_iters=1, dt=dt, lambda_cap=solver.lambda_cap
        )
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, max_iters=1, dt=dt)

        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)

    def test_box_clamp_gpu(self):
        """Verify the device box clamp invariant: ``|lam_t| <= mu · lam_n``
        after the clamp, and no friction where ``lam_n`` is not positive."""
        solver = _build_minimal_solver(friction=True)
        # Use a seeded problem that exercises both positive and negative
        # ``lam_n`` during the inner iterates. Two iters give the FB Newton
        # step enough room to drive ``lam_n`` past zero on some rows.
        W, r, pene0, mu = _make_coulomb_inputs(M=4, seed=19)
        dt = 0.01

        lam_gpu, _omega_gpu, _lam_apply_gpu = solver._solve_nsn_coulomb_gpu(W, r, mu, pene0, max_iters=2, dt=dt)

        # After the clamp, |lam_t| <= mu * max(lam_n, 0) per contact.
        for c in range(4):
            bound = mu[c] * max(lam_gpu[3 * c], 0.0)
            self.assertLessEqual(abs(lam_gpu[3 * c + 1]), bound + 1.0e-9)
            self.assertLessEqual(abs(lam_gpu[3 * c + 2]), bound + 1.0e-9)

    def test_lambda_cap_keeps_the_friction_cone_gpu(self):
        """Clamp the tangent multipliers into the cone of the capped normal multiplier, as the host reference does.

        The warm start makes the first two contacts slide, on the boundaries of their cones, with normal multipliers
        near 4.6 that the cap of 2 (in internal units) halves.
        """
        W, r, pene0, mu = _make_coulomb_inputs(M=3, seed=11)
        dt = 0.01
        lam_init = np.zeros(9)
        lam_init[0::3] = 5.0
        solver = _build_minimal_solver(friction=True, lambda_cap=2.0 * dt * dt)

        lam_cpu, omega_cpu, lam_apply_cpu = solve_nsn_coulomb(
            W, r, mu, pene0, max_iters=1, lam_init=lam_init, dt=dt, lambda_cap=solver.lambda_cap
        )
        lam_gpu, omega_gpu, lam_apply_gpu = solver._solve_nsn_coulomb_gpu(
            W, r, mu, pene0, max_iters=1, lam_init=lam_init, dt=dt
        )

        np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(omega_gpu, omega_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_apply_gpu, lam_apply_cpu, rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(lam_gpu[[0, 3]], 2.0, rtol=0, atol=1.0e-9)
        for c in range(3):
            bound = mu[c] * max(lam_gpu[3 * c], 0.0)
            self.assertLessEqual(abs(lam_gpu[3 * c + 1]), bound + 1.0e-9)
            self.assertLessEqual(abs(lam_gpu[3 * c + 2]), bound + 1.0e-9)

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
