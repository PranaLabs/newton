# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GPU FB-row ``@wp.func`` ports and the Schur build kernel.

Verifies
:func:`~newton._src.solvers.fba.kernels.fb_unilateral_row_wp`,
:func:`~newton._src.solvers.fba.kernels.fb_frictional_row_wp`,
:func:`~newton._src.solvers.fba.kernels.build_a_schur_kernel` and
:func:`~newton._src.solvers.fba.kernels.coulomb_box_clamp_kernel` against
their CPU numpy/Python references, and
:func:`~newton._src.solvers.fba.kernels.fb_newton_lite_coulomb_kernel`
against the host NSN step with the lite mass operator.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba import kernels as K
from newton.tests.fba._reference import coulomb_box_clamp, fb_frictional_row, fb_unilateral_row, solve_nsn_coulomb


@wp.kernel
def _launch_fb_unilateral_row(
    penetration: wp.float64,
    lam: wp.float64,
    precond: wp.float64,
    dt: wp.float64,
    pene0: wp.float64,
    out: wp.array[wp.vec3d],
):
    out[0] = K.fb_unilateral_row_wp(penetration, lam, precond, dt, pene0)


@wp.kernel
def _launch_fb_frictional_row(
    penetration: wp.float64,
    lam_t: wp.float64,
    lam_n: wp.float64,
    mu: wp.float64,
    precond: wp.float64,
    dt: wp.float64,
    pene0: wp.float64,
    out: wp.array[wp.vec3d],
):
    out[0] = K.fb_frictional_row_wp(penetration, lam_t, lam_n, mu, precond, dt, pene0)


def _run_unilateral_wp(
    penetration: float,
    lam: float,
    precond: float,
    dt: float,
    pene0: float,
) -> tuple[float, float, float]:
    """Launch one thread of ``fb_unilateral_row_wp`` and return the result."""
    out = wp.zeros(1, dtype=wp.vec3d)
    wp.launch(
        _launch_fb_unilateral_row,
        dim=1,
        inputs=[
            wp.float64(penetration),
            wp.float64(lam),
            wp.float64(precond),
            wp.float64(dt),
            wp.float64(pene0),
        ],
        outputs=[out],
    )
    arr = out.numpy()[0]
    return float(arr[0]), float(arr[1]), float(arr[2])


def _run_frictional_wp(
    penetration: float,
    lam_t: float,
    lam_n: float,
    mu: float,
    precond: float,
    dt: float,
    pene0: float,
) -> tuple[float, float, float]:
    """Launch one thread of ``fb_frictional_row_wp`` and return the result."""
    out = wp.zeros(1, dtype=wp.vec3d)
    wp.launch(
        _launch_fb_frictional_row,
        dim=1,
        inputs=[
            wp.float64(penetration),
            wp.float64(lam_t),
            wp.float64(lam_n),
            wp.float64(mu),
            wp.float64(precond),
            wp.float64(dt),
            wp.float64(pene0),
        ],
        outputs=[out],
    )
    arr = out.numpy()[0]
    return float(arr[0]), float(arr[1]), float(arr[2])


class TestFbUnilateralRowWp(unittest.TestCase):
    """Verify GPU ``fb_unilateral_row_wp`` matches the CPU reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _assert_close(self, got: tuple[float, float, float], expected: tuple[float, float, float]) -> None:
        for g, e in zip(got, expected, strict=True):
            self.assertAlmostEqual(g, e, delta=1e-12)

    def test_fb_unilateral_row_wp_matches_python(self):
        """Match the host unilateral Fischer-Burmeister row across its branches and input signs."""
        cases = [
            # (penetration, lam, precond, dt, pene0)
            (0.01, 0.5, 2.0, 0.01, 0.005),  # generic positive
            (-0.002, 1.2, 1.5, 0.01, 0.001),  # negative penetration (omega > 1)
            (0.0, 0.0, 2.0, 0.01, 0.0),  # degenerate root < 1e-30 branch
            (0.0, 0.0, 1.0, 0.01, 0.0),  # explicit pene=lam=0 -> degenerate again
            (0.0, 0.3, 4.0, 0.005, 0.0),  # pene=0 with non-zero lam (root = |plam|)
            (0.05, 0.0, 1.0, 0.02, 0.01),  # lam=0 (root = |pene|)
            (-1.0e-20, 1.0e-20, 1.0, 0.01, 0.0),  # tiny inputs, still > 1e-30
        ]
        for pen, lam, precond, dt, pene0 in cases:
            with self.subTest(pen=pen, lam=lam, precond=precond, dt=dt, pene0=pene0):
                expected = fb_unilateral_row(pen, lam, precond, dt, pene0)
                got = _run_unilateral_wp(pen, lam, precond, dt, pene0)
                self._assert_close(got, expected)

    def test_newton_rhs_ignores_the_anchor_offset(self):
        """Keep the unilateral row's Newton right-hand side ``h - omega * (penetration + pene0)`` at ``-phi`` for any
        anchor offset ``pene0``.

        Moving the contact plane off the origin must not change it, including at the degenerate root, where a float32
        residual lands exactly on the plane.
        """
        cases = [(0.0, 0.0, 2.0, 0.01), (0.01, 0.5, 2.0, 0.01), (-0.002, 1.2, 1.5, 0.01), (0.05, 0.0, 1.0, 0.02)]
        for pen, lam, precond, dt in cases:
            phi = pen + precond * lam - np.hypot(pen, precond * lam)
            for pene0 in (0.0, 3.0, -7.25):
                for omega, _, h in (
                    fb_unilateral_row(pen, lam, precond, dt, pene0),
                    _run_unilateral_wp(pen, lam, precond, dt, pene0),
                ):
                    with self.subTest(pen=pen, lam=lam, pene0=pene0):
                        self.assertAlmostEqual(h - omega * (pen + pene0), -phi, delta=1e-12)


class TestFbFrictionalRowWp(unittest.TestCase):
    """Verify GPU ``fb_frictional_row_wp`` matches the CPU reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _assert_close(self, got: tuple[float, float, float], expected: tuple[float, float, float]) -> None:
        for g, e in zip(got, expected, strict=True):
            self.assertAlmostEqual(g, e, delta=1e-12)

    def test_fb_frictional_row_wp_matches_python(self):
        """Match the host frictional Fischer-Burmeister row across its active and inactive branches."""
        cases = [
            # (penetration, lam_t, lam_n, mu, precond, dt, pene0)
            (0.001, 0.1, 1.0, 0.5, 2.0, 0.01, 0.0),  # active, stick-ish
            (0.05, 0.4, 1.0, 0.5, 2.0, 0.01, 0.0),  # active, mu*lam_n - |lam_t| > 0
            (0.001, -0.2, 0.8, 0.4, 1.5, 0.01, 0.002),  # negative lam_t, non-zero pene0
            (0.0, 0.0, 1.0, 0.5, 2.0, 0.01, 0.0),  # active branch, abspenevel = 0
            (0.001, 0.1, 0.0, 0.5, 2.0, 0.01, 0.0),  # inactive branch (lam_n == 0)
            (0.001, 0.3, -0.1, 0.5, 2.0, 0.01, 0.0),  # inactive branch (lam_n < 0)
            # denom < 1e-30 edge: |lam_t| >> mu*lam_n with abspenevel = 0. Then root == |tmp|,
            # so abspenevel + mu*precond*lam_n - root = mu*precond*lam_n - |tmp|.
            # Choose mu*lam_n = |lam_t| exactly => tmp = 0 => root = 0 => denom = mu*precond*lam_n,
            # which is finite. To force denom ≈ 0 we pick lam_n = 0 (covered by inactive
            # branch). Instead pick a case where root - tmp is finite but denom is small:
            (0.0, 0.5, 1.0, 0.5, 1.0, 0.01, 0.0),  # mu*lam_n == |lam_t|, abspenevel = 0
        ]
        for pen, lam_t, lam_n, mu, precond, dt, pene0 in cases:
            with self.subTest(pen=pen, lam_t=lam_t, lam_n=lam_n, mu=mu, precond=precond, dt=dt, pene0=pene0):
                expected = fb_frictional_row(pen, lam_t, lam_n, mu, precond, dt, pene0)
                got = _run_frictional_wp(pen, lam_t, lam_n, mu, precond, dt, pene0)
                self._assert_close(got, expected)


class TestBuildASchurKernel(unittest.TestCase):
    """Verify GPU Schur build matches ``np.outer(omega, omega) * W + diag(c)``."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _check(self, M: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((M, M))
        W_np = (A @ A.T).astype(np.float64)  # SPD
        omega_np = rng.standard_normal(M).astype(np.float64)
        compliance_np = (rng.standard_normal(M) ** 2 + 0.1).astype(np.float64)

        W = wp.array(W_np, dtype=wp.float64)
        omega = wp.array(omega_np, dtype=wp.float64)
        compliance = wp.array(compliance_np, dtype=wp.float64)
        a_schur = wp.zeros((M, M), dtype=wp.float64)

        wp.launch(
            K.build_a_schur_kernel,
            dim=(M, M),
            inputs=[W, omega, compliance],
            outputs=[a_schur],
        )

        got = a_schur.numpy()
        expected = np.outer(omega_np, omega_np) * W_np + np.diag(compliance_np)
        max_err = np.abs(got - expected).max()
        self.assertLess(max_err, 1e-10)

    def test_build_a_schur_kernel_matches_numpy_small(self):
        """Build the Newton matrix ``outer(omega, omega) * W + diag(c)`` of a 5-row system as NumPy does."""
        self._check(M=5, seed=42)

    def test_build_a_schur_kernel_matches_numpy_medium(self):
        """Build the Newton matrix ``outer(omega, omega) * W + diag(c)`` of a 30-row system as NumPy does."""
        self._check(M=30, seed=7)


class TestCoulombBoxClampKernel(unittest.TestCase):
    """Verify the GPU box clamp of the tangent multipliers against the host reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_box_clamp_matches_reference(self):
        """Clamp tangents into ``±mu lambda_n``, and zero them where ``lambda_n`` is zero or negative."""
        lam_np = np.array(
            [
                [2.0, 0.5, -0.3],  # inside the box: unchanged
                [2.0, 3.0, -4.0],  # outside: clamped to ±mu lambda_n
                [0.0, 0.2, -0.1],  # no normal multiplier: no friction
                [-1.5, 0.4, 2.5],  # negative normal multiplier: no friction, lambda_n kept
            ],
            dtype=np.float64,
        ).reshape(-1)
        mu_np = np.array([0.5, 0.5, 0.5, 0.8], dtype=np.float64)
        lam = wp.array(lam_np, dtype=wp.float64)
        wp.launch(K.coulomb_box_clamp_kernel, dim=4, inputs=[wp.array(mu_np, dtype=wp.float64)], outputs=[lam])

        expected = lam_np.copy()
        coulomb_box_clamp(expected, mu_np)
        np.testing.assert_array_equal(lam.numpy(), expected)
        np.testing.assert_array_equal(
            expected.reshape(4, 3), [[2.0, 0.5, -0.3], [2.0, 1.0, -1.0], [0.0, 0.0, 0.0], [-1.5, 0.0, 0.0]]
        )


class TestLiteCoulombKernel(unittest.TestCase):
    """The per-contact lite FB-Newton kernel against the host NSN step."""

    def test_lite_kernel_keeps_friction_in_the_cone(self):
        """Take one warm-started lite Newton step that matches the host NSN step and leaves every tangent multiplier
        within the friction cone of its normal multiplier.

        The warm start is what the solver's local-global iterations pass on; from it the Newton step pushes the
        tangent multipliers of nearly separating contacts far outside the cone, which the clamp brings back.
        """
        rng = np.random.default_rng(0)
        M, dt = 16, 0.01
        normal = rng.normal(size=(M, 3))
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        helper = np.where(np.abs(normal[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        t1 = np.cross(normal, helper)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(normal, t1)
        inv_mass = rng.uniform(0.5, 2.0, M)
        mu = rng.uniform(0.1, 0.8, M)
        r = rng.normal(scale=1.0e-3, size=3 * M)
        pene0 = rng.normal(scale=1.0e-3, size=3 * M)
        lam0 = rng.normal(scale=10.0, size=3 * M)
        lam0[0::3] = np.abs(lam0[0::3])
        omega0 = rng.uniform(0.2, 1.0, 3 * M)
        # Each contact's block of the lite operator is dt^2 / m times the identity of its orthonormal frame.
        W = np.kron(np.diag(dt * dt * inv_mass), np.eye(3))
        lam_ref, omega_ref, lam_apply_ref = solve_nsn_coulomb(
            W, r, mu, pene0, max_iters=1, lam_init=lam0, omega_init=omega0, dt=dt
        )

        def vec3(a):
            return wp.array(a.astype(np.float32), dtype=wp.vec3)

        lam = wp.array(lam0, dtype=wp.float64)
        omega = wp.array(omega0, dtype=wp.float64)
        lam_apply = wp.zeros(3 * M, dtype=wp.float64)
        wp.launch(
            K.fb_newton_lite_coulomb_kernel,
            dim=M,
            inputs=[
                wp.empty(0, dtype=wp.int32),
                wp.array(np.arange(M, dtype=np.int32), dtype=wp.int32),
                vec3(normal),
                vec3(t1),
                vec3(t2),
                wp.array(np.ones(M, dtype=np.float32), dtype=wp.float32),
                wp.array(mu, dtype=wp.float64),
                wp.array(r, dtype=wp.float64),
                wp.array(pene0, dtype=wp.float64),
                wp.array(inv_mass, dtype=wp.float64),
                wp.float64(dt),
                wp.float64(0.0),
                wp.int32(0),
            ],
            outputs=[lam, omega, lam_apply],
        )
        lam = lam.numpy()
        cone = mu * np.abs(lam[0::3]) * (1.0 + 1.0e-12)
        self.assertTrue((np.abs(lam[1::3]) <= cone).all() and (np.abs(lam[2::3]) <= cone).all())
        # The kernel takes float32 directions; the multipliers still agree to rounding.
        np.testing.assert_allclose(lam, lam_ref, rtol=0.0, atol=1.0e-6 * np.abs(lam_ref).max())
        np.testing.assert_allclose(omega.numpy(), omega_ref, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            lam_apply.numpy(), lam_apply_ref, rtol=0.0, atol=1.0e-6 * np.abs(lam_apply_ref).max()
        )

    def test_per_particle_kernel_matches_the_host_step(self):
        """Take one warm-started lite Newton step on particles with one, two and four contacts, as the host NSN step
        does on the block-diagonal lite operator, with no force cap and with one that halves the largest normal
        multiplier.

        The warm start lies within the friction cones, as the local-global iterations pass it on, and makes two
        contacts slide on each of the particles with two and four contacts, so the cap shrinks cones that bound their
        tangent multipliers.
        """
        rng = np.random.default_rng(0)
        dt = 0.01
        contact_particle = np.array([0, 1, 1, 2, 2, 2, 2])
        M, N = contact_particle.size, 3
        normal = rng.normal(size=(M, 3))
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        helper = np.where(np.abs(normal[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        t1 = np.cross(normal, helper)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(normal, t1)
        # Rows interleave [n, t1, t2] per contact; the kernel reads float32 directions.
        row_dir = np.stack([normal, t1, t2], axis=1).reshape(-1, 3).astype(np.float32)
        row_particle = np.repeat(contact_particle, 3).astype(np.int32)
        inv_mass = rng.uniform(0.5, 2.0, N)
        mu = rng.uniform(0.1, 0.8, M)
        r = rng.normal(scale=1.0e-3, size=3 * M)
        pene0 = rng.normal(scale=1.0e-3, size=3 * M)
        lam0 = rng.normal(scale=1.0e5, size=3 * M)
        lam0[0::3] = np.abs(lam0[0::3])
        coulomb_box_clamp(lam0, mu)
        omega0 = rng.uniform(0.2, 1.0, 3 * M)
        # The lite operator couples the rows of a particle by the products of their directions.
        d = row_dir.astype(np.float64)
        same = row_particle[:, None] == row_particle[None, :]
        W = np.where(same, (d @ d.T) * (dt * dt * inv_mass[row_particle])[:, None], 0.0)
        offsets = np.concatenate([[0], np.cumsum(np.bincount(row_particle, minlength=N))]).astype(np.int32)
        lam_free = solve_nsn_coulomb(W, r, mu, pene0, max_iters=1, lam_init=lam0, omega_init=omega0, dt=dt)[0]
        for lambda_cap in (None, 0.5 * lam_free[0::3].max() * dt * dt):
            with self.subTest(lambda_cap=lambda_cap):
                lam_ref, omega_ref, lam_apply_ref = solve_nsn_coulomb(
                    W, r, mu, pene0, max_iters=1, lam_init=lam0, omega_init=omega0, dt=dt, lambda_cap=lambda_cap
                )
                lam = wp.array(lam0, dtype=wp.float64)
                omega = wp.array(omega0, dtype=wp.float64)
                lam_apply = wp.zeros(3 * M, dtype=wp.float64)
                wp.launch(
                    K.fb_newton_lite_coulomb_per_particle_kernel,
                    dim=N,
                    inputs=[
                        wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32),
                        wp.array(offsets, dtype=wp.int32),
                        wp.array(np.argsort(row_particle, kind="stable").astype(np.int32), dtype=wp.int32),
                        wp.array(row_particle, dtype=wp.int32),
                        wp.array(row_dir, dtype=wp.vec3),
                        wp.array(np.ones(3 * M, dtype=np.float32), dtype=wp.float32),
                        wp.array(mu, dtype=wp.float64),
                        wp.array(r, dtype=wp.float64),
                        wp.array(pene0, dtype=wp.float64),
                        wp.array(inv_mass, dtype=wp.float64),
                        wp.float64(dt),
                        wp.float64(0.0 if lambda_cap is None else lambda_cap / (dt * dt)),
                        wp.int32(lambda_cap is not None),
                        wp.int32(3 * M),
                    ],
                    outputs=[lam, omega, lam_apply],
                )
                lam = lam.numpy()
                cone = mu * np.maximum(lam[0::3], 0.0) * (1.0 + 1.0e-12)
                self.assertTrue((np.abs(lam[1::3]) <= cone).all() and (np.abs(lam[2::3]) <= cone).all())
                # Float32 directions again: the multipliers agree to rounding.
                np.testing.assert_allclose(lam, lam_ref, rtol=0.0, atol=1.0e-6 * np.abs(lam_ref).max())
                np.testing.assert_allclose(omega.numpy(), omega_ref, rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(
                    lam_apply.numpy(), lam_apply_ref, rtol=0.0, atol=1.0e-6 * np.abs(lam_apply_ref).max()
                )


if __name__ == "__main__":
    unittest.main()
