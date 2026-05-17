# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GPU FB-row ``@wp.func`` ports and the Schur build kernel.

Covers Step 2-4 of the NSN GPU port (see
``docs/superpowers/plans/2026-05-17-fba-nsn-gpu-port.md``): verifies
:func:`~newton._src.solvers.fba.kernels.fb_unilateral_row_wp`,
:func:`~newton._src.solvers.fba.kernels.fb_frictional_row_wp` and
:func:`~newton._src.solvers.fba.kernels.build_a_schur_kernel` against
their CPU numpy/Python references.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba import kernels as K
from newton._src.solvers.fba.solver_fba import fb_frictional_row, fb_unilateral_row


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
        # Five cases covering the main branches and signs of inputs.
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


class TestFbFrictionalRowWp(unittest.TestCase):
    """Verify GPU ``fb_frictional_row_wp`` matches the CPU reference."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _assert_close(self, got: tuple[float, float, float], expected: tuple[float, float, float]) -> None:
        for g, e in zip(got, expected, strict=True):
            self.assertAlmostEqual(g, e, delta=1e-12)

    def test_fb_frictional_row_wp_matches_python(self):
        cases = [
            # (penetration, lam_t, lam_n, mu, precond, dt, pene0)
            (0.001, 0.1, 1.0, 0.5, 2.0, 0.01, 0.0),  # active, stick-ish
            (0.05, 0.4, 1.0, 0.5, 2.0, 0.01, 0.0),  # active, mu*lam_n - |lam_t| > 0
            (0.001, -0.2, 0.8, 0.4, 1.5, 0.01, 0.002),  # negative lam_t, non-zero pene0
            (0.0, 0.0, 1.0, 0.5, 2.0, 0.01, 0.0),  # active branch, abspenevel = 0
            (0.001, 0.1, 0.0, 0.5, 2.0, 0.01, 0.0),  # inactive branch (lam_n == 0)
            (0.001, 0.3, -0.1, 0.5, 2.0, 0.01, 0.0),  # inactive branch (lam_n < 0)
            # denom < 1e-30 edge: |lam_t| >> mu*lam_n with abspenevel = 0 — root == |tmp|,
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
        self._check(M=5, seed=42)

    def test_build_a_schur_kernel_matches_numpy_medium(self):
        self._check(M=30, seed=7)


if __name__ == "__main__":
    unittest.main()
