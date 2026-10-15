# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SolverFBA rigid-body kernels (pinned frame, RHS, retraction)."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba import rigid

DT = 0.01
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32).astype(np.float64)
MASS = 2.0
INERTIA = np.array([[0.1, 0.02, 0.0], [0.02, 0.2, 0.01], [0.0, 0.01, 0.3]])


def _f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


class TestFBARigidKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scipy.spatial.transform import Rotation

        wp.init()
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"
        cls.Rotation = Rotation

    def _body(self, w0=(0.5, -1.0, 2.0)):
        return {
            "q0": _f32(self.Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()),
            "p0": _f32([0.1, 0.2, 0.3]),
            "com": _f32([0.05, -0.02, 0.1]),
            "v0": _f32([1.0, -0.5, 0.2]),
            "w0": _f32(w0),
            "f": _f32([0.3, 0.1, -0.2]),
            "tau": _f32([0.01, -0.02, 0.03]),
        }

    def _prepare(self, body):
        dev = self.device
        arr = lambda v, t: wp.array([v], dtype=t, device=dev)  # noqa: E731
        bodies = wp.array([0], dtype=wp.int32, device=dev)
        body_q = arr(wp.transform(wp.vec3(*body["p0"]), wp.quat(*body["q0"])), wp.transform)
        q_pin = wp.empty(1, dtype=wp.quatd, device=dev)
        x_prev = wp.empty(1, dtype=wp.vec3d, device=dev)
        b = wp.empty(6, dtype=wp.float64, device=dev)
        wp.launch(
            rigid.prepare_rigid_step_kernel,
            dim=1,
            inputs=[
                bodies,
                body_q,
                arr(wp.spatial_vector(*body["v0"], *body["w0"]), wp.spatial_vector),
                arr(wp.spatial_vector(*body["f"], *body["tau"]), wp.spatial_vector),
                arr(wp.vec3(*body["com"]), wp.vec3),
                wp.array([MASS], dtype=wp.float32, device=dev),
                arr(wp.mat33(INERTIA), wp.mat33),
                wp.array([0], dtype=wp.int32, device=dev),
                arr(wp.vec3(*GRAVITY), wp.vec3),
                wp.float64(DT),
            ],
            outputs=[q_pin, x_prev, b],
            device=dev,
        )
        return bodies, body_q, q_pin, x_prev, b

    def _rigid_matrix(self):
        A = np.zeros((6, 6))
        A[:3, :3] = MASS * np.eye(3)
        A[3:, 3:] = _f32(INERTIA)
        return A / DT**2

    def test_rhs_is_free_flight_step(self):
        """b = A_rigid · u_free with u_free = (R_pinᵀ x_free, dt R_pinᵀ ω_free)."""
        body = self._body()
        _, _, q_pin, x_prev, b = self._prepare(body)

        R0 = self.Rotation.from_quat(body["q0"])
        R_pin = self.Rotation.from_rotvec(body["w0"] * DT) * R0
        np.testing.assert_allclose(self.Rotation.from_quat(q_pin.numpy()[0]).as_matrix(), R_pin.as_matrix(), atol=1e-12)
        x0 = body["p0"] + R0.apply(body["com"])
        np.testing.assert_allclose(x_prev.numpy()[0], x0, rtol=0.0, atol=1e-12)

        Rp = R_pin.as_matrix()
        inertia = _f32(INERTIA)
        w = Rp.T @ body["w0"]
        chi = Rp.T @ (x0 + DT * body["v0"] + DT**2 * (GRAVITY + body["f"] / MASS))
        rho = DT * w + DT**2 * np.linalg.solve(inertia, -np.cross(w, inertia @ w) + Rp.T @ body["tau"])
        np.testing.assert_allclose(b.numpy(), self._rigid_matrix() @ np.concatenate([chi, rho]), rtol=1e-10)

    def test_pinned_rotation_is_clamped(self):
        body = self._body(w0=(0.0, 0.0, 100.0))  # |ω| dt = 1 rad > 0.5
        _, _, q_pin, _, _ = self._prepare(body)
        expected = self.Rotation.from_rotvec([0.0, 0.0, 0.5]) * self.Rotation.from_quat(body["q0"])
        np.testing.assert_allclose(
            self.Rotation.from_quat(q_pin.numpy()[0]).as_matrix(), expected.as_matrix(), atol=1e-12
        )

    def test_rotation_noise_keeps_frame(self):
        body = self._body(w0=(1.0e-13, 0.0, 0.0))
        _, _, q_pin, _, _ = self._prepare(body)
        np.testing.assert_allclose(q_pin.numpy()[0], body["q0"] / np.linalg.norm(body["q0"]), rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
