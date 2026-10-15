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

    def test_free_step_pose_and_velocity(self):
        body = self._body()
        bodies, body_q, q_pin, x_prev, b = self._prepare(body)
        u = wp.array(np.linalg.solve(self._rigid_matrix(), b.numpy()), dtype=wp.float64, device=self.device)
        q_out = wp.empty(1, dtype=wp.transform, device=self.device)
        qd_out = wp.empty(1, dtype=wp.spatial_vector, device=self.device)
        wp.launch(
            rigid.retract_rigid_kernel,
            dim=1,
            inputs=[bodies, u, q_pin, x_prev, body_q, wp.array([wp.vec3(*body["com"])], dtype=wp.vec3), wp.float64(DT)],
            outputs=[q_out, qd_out],
            device=self.device,
        )

        # Reference: semi-implicit step with explicit gyroscopic torque in the
        # pinned frame, retracted from the step-start orientation.
        R0 = self.Rotation.from_quat(body["q0"])
        Rp = (self.Rotation.from_rotvec(body["w0"] * DT) * R0).as_matrix()
        inertia = _f32(INERTIA)
        w = Rp.T @ body["w0"]
        x0 = body["p0"] + R0.apply(body["com"])
        v = body["v0"] + DT * (GRAVITY + body["f"] / MASS)
        omega = Rp @ (w + DT * np.linalg.solve(inertia, -np.cross(w, inertia @ w) + Rp.T @ body["tau"]))
        R_new = self.Rotation.from_rotvec(omega * DT) * R0
        origin = x0 + DT * v - R_new.apply(body["com"])

        tf = q_out.numpy()[0]
        np.testing.assert_allclose(tf[:3], origin, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(self.Rotation.from_quat(tf[3:]).as_matrix(), R_new.as_matrix(), atol=1e-6)
        np.testing.assert_allclose(qd_out.numpy()[0], np.concatenate([v, omega]), rtol=1e-5, atol=1e-6)

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


class TestSolverFBARigidStep(unittest.TestCase):
    """SolverFBA integrates dynamic bodies and passes kinematic ones through."""

    @classmethod
    def setUpClass(cls):
        from scipy.spatial.transform import Rotation

        wp.init()
        cls.Rotation = Rotation

    def _model(self, inertia, gravity=-9.81, cloth=True):
        import newton  # noqa: PLC0415

        builder = newton.ModelBuilder(gravity=gravity)
        if cloth:  # far from the body, so it never interacts
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 5.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[
                    wp.vec3(0.0, 0.0, 0.0),
                    wp.vec3(1.0, 0.0, 0.0),
                    wp.vec3(0.0, 1.0, 0.0),
                    wp.vec3(1.0, 1.0, 0.0),
                ],
                indices=[0, 1, 2, 1, 3, 2],
                density=1.0,
                tri_ke=1.0e2,
                tri_ka=0.0,
                tri_kd=0.0,
                edge_ke=1.0,
                edge_kd=0.0,
            )
        builder.add_body(mass=MASS, com=wp.vec3(0.1, 0.0, 0.0), inertia=wp.mat33(inertia), lock_inertia=True)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
        return builder.finalize()

    def _run(self, model, qd0, steps, **kwargs):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        solver = SolverFBA(model, iterations=1, **kwargs)
        s0, s1 = model.state(), model.state()
        s0.body_qd.assign([wp.spatial_vector(*qd0), wp.spatial_vector(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)])
        for _ in range(steps):
            solver.step(s0, s1, None, None, DT)
            s0, s1 = s1, s0
        return s0

    def test_projectile_with_principal_spin(self):
        """Semi-implicit Euler for the COM; constant spin about a principal axis."""
        model = self._model(np.diag([0.1, 0.2, 0.3]))
        v0, w0, n = np.array([1.0, 0.0, 2.0]), np.array([0.0, 0.0, 3.0]), 10
        state = self._run(model, (*v0, *w0), n)

        g = model.gravity.numpy()[0].astype(np.float64)
        rot = self.Rotation.from_rotvec(w0 * n * DT)
        com = np.array([0.1, 0.0, 0.0])
        tf = state.body_q.numpy()[0]
        x = tf[:3] + self.Rotation.from_quat(tf[3:]).apply(com)
        np.testing.assert_allclose(x, com + n * DT * v0 + DT**2 * g * n * (n + 1) / 2, atol=1e-5)
        np.testing.assert_allclose(self.Rotation.from_quat(tf[3:]).as_matrix(), rot.as_matrix(), atol=1e-5)
        np.testing.assert_allclose(state.body_qd.numpy()[0], np.concatenate([v0 + n * DT * g, w0]), atol=1e-4)

    def test_rigid_only_model(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._model(np.diag([0.1, 0.2, 0.3]), cloth=False)
        v0, n = np.array([1.0, 0.0, 2.0]), 10
        state = self._run(model, (*v0, 0.0, 0.0, 0.0), n)
        g = model.gravity.numpy()[0].astype(np.float64)
        np.testing.assert_allclose(state.body_qd.numpy()[0, :3], v0 + n * DT * g, atol=1e-4)

        import newton  # noqa: PLC0415

        with self.assertRaises(ValueError):
            SolverFBA(newton.ModelBuilder().finalize())

    def test_torque_free_tumbling_conserves_momentum(self):
        inertia = np.diag([0.1, 0.2, 0.3])
        w0 = np.array([0.3, 5.0, 0.2])  # near the unstable intermediate axis
        state = self._run(self._model(inertia, gravity=0.0), (0.0, 0.0, 0.0, *w0), 100)

        R = self.Rotation.from_quat(state.body_q.numpy()[0, 3:]).as_matrix()
        w = state.body_qd.numpy()[0, 3:].astype(np.float64)
        L0, L = inertia @ w0, R @ inertia @ R.T @ w
        self.assertLess(np.linalg.norm(L - L0) / np.linalg.norm(L0), 5e-3)
        self.assertLess(abs(w @ R @ inertia @ R.T @ w / (w0 @ inertia @ w0) - 1.0), 5e-3)

    def test_kinematic_and_external_bodies_untouched(self):
        model = self._model(np.diag([0.1, 0.2, 0.3]))
        state = self._run(model, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0), 1)
        np.testing.assert_array_equal(state.body_q.numpy()[1], model.body_q.numpy()[1])

        from newton.solvers import SolverFBA  # noqa: PLC0415

        solver = SolverFBA(model, iterations=1, integrate_with_external_rigid_solver=True)
        s0, s1 = model.state(), model.state()
        s1.body_q.fill_(wp.transform(wp.vec3(7.0, 7.0, 7.0), wp.quat_identity()))
        solver.step(s0, s1, None, None, DT)
        np.testing.assert_array_equal(s1.body_q.numpy()[:, :3], 7.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
