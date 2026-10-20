# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fp64 SVD of the tetrahedral ARAP, corotational and Neo-Hookean projections.

The tetrahedral projection kernels promote ``F`` to ``wp.mat33d`` before
``wp.svd3``, project the singular values and rebuild ``P = U diag(sigma) V^T``
in ``wp.float64``, and store the per-vertex contributions in fp32. The tests
compare the contributions with NumPy fp64 references at rest and for
deformation gradients whose singular values span eight decades, and the
singular values of the fp64 SVD with NumPy's to 1e-10.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba import kernels as K


def _np_arap_proj(F: np.ndarray) -> np.ndarray:
    """fp64 ARAP projection R = U V^T with reflection fix."""
    U, _s, Vt = np.linalg.svd(F.astype(np.float64), full_matrices=True)
    V = Vt.T
    if np.linalg.det(U) * np.linalg.det(V) < 0.0:
        U[:, 2] = -U[:, 2]
    return U @ V.T


def _np_corot_sigma(sigma: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """fp64 closed-form 3D corotational sigma projection, matches kernel formula."""
    k = 2.0 * mu
    alpha = 4.0 * mu
    beta = lam
    n = 3.0
    b = 2.0 * mu + 3.0 * lam + k * sigma
    scale = beta * b.sum() / (alpha * (alpha + n * beta))
    return b / alpha - scale


def _np_nh_sigma(sigma: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """fp64 minimizer of the NH sigma projection ``Psi(s) + (k/2)|s - sigma|^2``, ``k = 2 mu``.

    Damped Newton steps that keep ``s`` positive and the objective decreasing, until the gradient vanishes.
    """
    k = 2.0 * mu
    s0 = sigma.astype(np.float64)

    def objective(s):
        log_J = np.log(np.prod(s))
        return 0.5 * mu * (np.dot(s, s) - 3.0) - mu * log_J + 0.5 * lam * log_J**2 + 0.5 * k * np.dot(s - s0, s - s0)

    s = s0.copy()
    for _ in range(100):
        log_J = np.log(np.prod(s))
        inv = 1.0 / s
        g = mu * (s - inv) + lam * log_J * inv + k * (s - s0)
        if np.linalg.norm(g) < 1.0e-12 * max(1.0, mu):
            break
        H = np.diag(mu + (mu + lam - lam * log_J) * inv * inv + k) + lam * np.outer(inv, inv) * (1.0 - np.eye(3))
        dx = np.linalg.solve(H, g)
        step = 1.0
        while np.any(s - step * dx <= 0.0) or objective(s - step * dx) > objective(s):
            step *= 0.5
        s = s - step * dx
    return s


def _reconstruct_proj(
    Ds: np.ndarray, Dm_inv: np.ndarray, P: np.ndarray, w: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numpy reference for the per-tet scatter contribution from a 3x3 P.

    Mirrors ``proj = w * Dm_inv * P^T``; rows scatter to t1/t2/t3 and -sum to
    t0. Returns the four vec3 contributions in tet local order.
    """
    proj = w * (Dm_inv @ P.T)
    row0 = proj[0]
    row1 = proj[1]
    row2 = proj[2]
    return -(row0 + row1 + row2), row0, row1, row2


def _build_F_from_tet(positions: np.ndarray, Dm_inv: np.ndarray) -> np.ndarray:
    e1 = positions[1] - positions[0]
    e2 = positions[2] - positions[0]
    e3 = positions[3] - positions[0]
    Ds = np.column_stack([e1, e2, e3])
    return Ds, Ds @ Dm_inv


def _launch_compute(kernel, positions, Dm_inv, weight, mu=None, lam=None):
    pos = wp.array(positions.astype(np.float32), dtype=wp.vec3)
    tet_idx = wp.array([0, 1, 2, 3], dtype=wp.int32)
    rest_inv = wp.array(
        [wp.mat33(*Dm_inv.astype(np.float32).flatten().tolist())],
        dtype=wp.mat33,
    )
    w = wp.array([float(weight)], dtype=wp.float32)
    contrib = wp.zeros((1, 4), dtype=wp.vec3)
    if mu is None:
        wp.launch(kernel, dim=1, inputs=[pos, tet_idx, rest_inv, w], outputs=[contrib])
    else:
        wp.launch(
            kernel,
            dim=1,
            inputs=[pos, tet_idx, rest_inv, w, wp.array([[mu, lam]], dtype=wp.vec2)],
            outputs=[contrib],
        )
    return contrib.numpy()  # shape (1, 4, 3)


def _make_ill_conditioned_tet(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a tet whose deformation gradient F has a large condition number.

    Builds Dm_inv via a regular unit-edge tet and stretches the deformed tet
    to give ``F = diag(stretch)`` with the smallest singular value at 1e-4.
    Mirrors a thin tip stretched far along one direction and squashed along
    another.
    """
    rest = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Dm = np.column_stack([rest[1] - rest[0], rest[2] - rest[0], rest[3] - rest[0]])
    Dm_inv = np.linalg.inv(Dm)

    rng = np.random.default_rng(seed)
    # Singular values spanning 8 orders of magnitude.
    sigma = np.array([1.0e4, 1.0, 1.0e-4])
    # Random orthogonal U, V.
    U, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(U) < 0:
        U[:, 0] = -U[:, 0]
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    F = U @ np.diag(sigma) @ V.T
    Ds = F @ Dm
    deformed = np.zeros((4, 3))
    deformed[0] = rng.standard_normal(3) * 0.1
    deformed[1] = deformed[0] + Ds[:, 0]
    deformed[2] = deformed[0] + Ds[:, 1]
    deformed[3] = deformed[0] + Ds[:, 2]
    return deformed, Dm_inv, F


class TestTetArapFp64(unittest.TestCase):
    """The tetrahedral ARAP projection against NumPy fp64."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_arap_matches_numpy_reference_well_conditioned(self):
        """Match the NumPy ARAP projection on an undeformed tetrahedron, whose projection is the identity."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Dm_inv = np.eye(3, dtype=np.float64)
        Ds, F = _build_F_from_tet(positions, Dm_inv)
        R = _np_arap_proj(F)
        w = 1.0
        exp = _reconstruct_proj(Ds, Dm_inv, R, w)

        got = _launch_compute(K.project_stretching_arap_tet_kernel, positions, Dm_inv, w)
        for i in range(4):
            np.testing.assert_allclose(got[0, i], exp[i], atol=1e-6)

    def test_arap_high_condition_F(self):
        """Match the NumPy fp64 ARAP projection of an ill-conditioned deformation gradient to 2e-5 relative."""
        positions, Dm_inv, F = _make_ill_conditioned_tet(seed=1)
        Ds = F @ np.linalg.inv(Dm_inv)
        R = _np_arap_proj(F)
        w = 2.5
        exp = _reconstruct_proj(Ds, Dm_inv, R, w)

        got = _launch_compute(K.project_stretching_arap_tet_kernel, positions, Dm_inv, w)
        # The kernel reads fp32 positions and stores fp32 contributions; they
        # agree with the fp64 reference to 2e-5 relative.
        for i in range(4):
            mag = max(1.0, np.linalg.norm(exp[i]))
            np.testing.assert_allclose(got[0, i], exp[i], rtol=2e-5, atol=2e-5 * mag)


class TestTetCorotFp64(unittest.TestCase):
    """The tetrahedral corotational projection against NumPy fp64."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_corot_identity_tet(self):
        """Match the NumPy corotational projection on an undeformed tetrahedron."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Dm_inv = np.eye(3, dtype=np.float64)
        Ds, F = _build_F_from_tet(positions, Dm_inv)
        mu, lam = 1000.0, 5000.0
        U, sigma, Vt = np.linalg.svd(F)
        sigma_proj = _np_corot_sigma(sigma, mu, lam)
        P = U @ np.diag(sigma_proj) @ Vt
        w = 1.0
        exp = _reconstruct_proj(Ds, Dm_inv, P, w)

        got = _launch_compute(
            K.project_stretching_corotational_tet_kernel,
            positions,
            Dm_inv,
            w,
            mu=mu,
            lam=lam,
        )
        for i in range(4):
            np.testing.assert_allclose(got[0, i], exp[i], atol=1e-3, rtol=1e-5)

    def test_corot_high_condition_F(self):
        """Match the NumPy corotational projection of an ill-conditioned deformation gradient to about 1e-3 relative."""
        positions, Dm_inv, F = _make_ill_conditioned_tet(seed=2)
        Ds = F @ np.linalg.inv(Dm_inv)
        mu, lam = 1000.0, 5000.0
        U, sigma, Vt = np.linalg.svd(F)
        sigma_proj = _np_corot_sigma(sigma, mu, lam)
        P = U @ np.diag(sigma_proj) @ Vt
        w = 0.5
        exp = _reconstruct_proj(Ds, Dm_inv, P, w)

        got = _launch_compute(
            K.project_stretching_corotational_tet_kernel,
            positions,
            Dm_inv,
            w,
            mu=mu,
            lam=lam,
        )
        # Corot projects sigma to a (small) constant offset around (1, 1, 1)
        # for very large sigma, so the contributions are dominated by the
        # Dm_inv * U V^T-shaped part. Allow ~1e-3 rel + 1e-3 abs (fp32
        # storage of values up to O(1e3)).
        for i in range(4):
            mag = max(1.0, np.linalg.norm(exp[i]))
            np.testing.assert_allclose(got[0, i], exp[i], rtol=1e-3, atol=1e-3 * mag)


class TestTetNeohookeanFp64(unittest.TestCase):
    """fp64 NH tet projection by L-BFGS against the converged numpy minimizer."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_nh_identity_tet(self):
        """Project an undeformed tet to the identity, where the NH gradient vanishes."""
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Dm_inv = np.eye(3, dtype=np.float64)
        Ds, F = _build_F_from_tet(positions, Dm_inv)
        mu, lam = 1000.0, 5000.0
        U, sigma, Vt = np.linalg.svd(F)
        sigma_proj = _np_nh_sigma(sigma, mu, lam)
        P = U @ np.diag(sigma_proj) @ Vt
        w = 1.0
        exp = _reconstruct_proj(Ds, Dm_inv, P, w)

        got = _launch_compute(
            K.project_stretching_neohookean_tet_kernel,
            positions,
            Dm_inv,
            w,
            mu=mu,
            lam=lam,
        )
        for i in range(4):
            np.testing.assert_allclose(got[0, i], exp[i], atol=1e-3, rtol=1e-5)

    def test_nh_mild_stretch(self):
        """Match the converged numpy minimizer on a moderate stretch (cond ~10)."""
        rng = np.random.default_rng(7)
        rest = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Dm = np.column_stack([rest[1] - rest[0], rest[2] - rest[0], rest[3] - rest[0]])
        Dm_inv = np.linalg.inv(Dm)
        sigma = np.array([5.0, 1.0, 0.5])
        U, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(U) < 0:
            U[:, 0] = -U[:, 0]
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        F = U @ np.diag(sigma) @ V.T
        Ds = F @ Dm
        positions = np.zeros((4, 3))
        positions[1] = Ds[:, 0]
        positions[2] = Ds[:, 1]
        positions[3] = Ds[:, 2]

        mu, lam = 1000.0, 5000.0
        Uref, sref, Vtref = np.linalg.svd(F)
        sigma_proj = _np_nh_sigma(sref, mu, lam)
        P = Uref @ np.diag(sigma_proj) @ Vtref
        w = 1.5
        exp = _reconstruct_proj(Ds, Dm_inv, P, w)

        got = _launch_compute(
            K.project_stretching_neohookean_tet_kernel,
            positions,
            Dm_inv,
            w,
            mu=mu,
            lam=lam,
        )
        for i in range(4):
            mag = max(1.0, np.linalg.norm(exp[i]))
            np.testing.assert_allclose(got[0, i], exp[i], rtol=2e-4, atol=2e-4 * mag)

    def test_nh_converges_on_second_stretch(self):
        """Match the converged numpy minimizer on a second stretch, sigma (2, 1, 0.7), to ~1e-3 relative."""
        rng = np.random.default_rng(11)
        rest = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Dm = np.column_stack([rest[1] - rest[0], rest[2] - rest[0], rest[3] - rest[0]])
        Dm_inv = np.linalg.inv(Dm)
        # Sigma chosen well clear of degeneracy.
        sigma = np.array([2.0, 1.0, 0.7])
        U, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(U) < 0:
            U[:, 0] = -U[:, 0]
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        F = U @ np.diag(sigma) @ V.T
        Ds = F @ Dm
        positions = np.zeros((4, 3))
        positions[1] = Ds[:, 0]
        positions[2] = Ds[:, 1]
        positions[3] = Ds[:, 2]

        mu, lam = 1000.0, 5000.0
        w = 0.75
        Uref, sref, Vtref = np.linalg.svd(F)
        P = Uref @ np.diag(_np_nh_sigma(sref, mu, lam)) @ Vtref
        exp = _reconstruct_proj(Ds, Dm_inv, P, w)
        got = _launch_compute(
            K.project_stretching_neohookean_tet_kernel,
            positions,
            Dm_inv,
            w,
            mu=mu,
            lam=lam,
        )
        for i in range(4):
            mag = max(1.0, np.linalg.norm(exp[i]))
            np.testing.assert_allclose(got[0, i], exp[i], rtol=1e-3, atol=1e-3 * mag)


class TestTetFp64SvdFunc(unittest.TestCase):
    """Spot-check ``wp.svd3`` on ``wp.mat33d`` agrees with numpy fp64 SVD."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_svd_d_singular_values_match_numpy(self):
        """Match NumPy's singular values of matrices spanning eight decades with the fp64 3 x 3 SVD."""

        @wp.kernel
        def k(F: wp.array[wp.mat33], out: wp.array[wp.vec3d]):
            f = F[0]
            fd = wp.mat33d(
                wp.float64(f[0, 0]),
                wp.float64(f[0, 1]),
                wp.float64(f[0, 2]),
                wp.float64(f[1, 0]),
                wp.float64(f[1, 1]),
                wp.float64(f[1, 2]),
                wp.float64(f[2, 0]),
                wp.float64(f[2, 1]),
                wp.float64(f[2, 2]),
            )
            _U, s, _V = wp.svd3(fd)
            out[0] = s

        rng = np.random.default_rng(13)
        for _ in range(5):
            sigma = np.array([rng.uniform(1.0, 1.0e4), rng.uniform(0.1, 1.0), rng.uniform(1.0e-4, 1.0e-1)])
            U, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            M = U @ np.diag(sigma) @ V.T
            a = wp.array([wp.mat33(*M.astype(np.float32).flatten().tolist())], dtype=wp.mat33)
            out = wp.zeros(1, dtype=wp.vec3d)
            wp.launch(k, dim=1, inputs=[a], outputs=[out])
            s_warp = np.array(out.numpy()[0])
            _, s_np, _ = np.linalg.svd(M.astype(np.float32).astype(np.float64))
            np.testing.assert_allclose(np.sort(s_warp)[::-1], np.sort(s_np)[::-1], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
