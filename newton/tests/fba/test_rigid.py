# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SolverFBA rigid-body kernels (pinned frame, RHS, retraction)."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba import rigid
from newton._src.solvers.fba.linear_solver import rigid_block_inverses

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
        """Build the rigid right-hand side ``b = A_rigid · u_free`` of the free-flight step,
        ``u_free = (R_pinᵀ x_free, dt R_pinᵀ ω_free)``."""
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
        """Retract a free rigid step to the pose and velocity of a semi-implicit step with explicit gyroscopic torque in
        the pinned frame."""
        body = self._body()
        bodies, body_q, q_pin, x_prev, b = self._prepare(body)
        u = wp.array(np.linalg.solve(self._rigid_matrix(), b.numpy()), dtype=wp.float64, device=self.device)
        q_out = wp.empty(1, dtype=wp.transform, device=self.device)
        qd_out = wp.empty(1, dtype=wp.spatial_vector, device=self.device)
        wp.launch(
            rigid.retract_rigid_kernel,
            dim=1,
            inputs=[bodies, u, q_pin, x_prev, body_q, wp.array([wp.vec3(*body["com"])], dtype=wp.vec3), wp.float64(DT)],
            outputs=[q_out, qd_out, wp.empty(1, dtype=wp.vec3d), wp.empty(1, dtype=wp.quatd)],
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
        """Clamp the pinned frame's rotation to 0.5 rad per step."""
        body = self._body(w0=(0.0, 0.0, 100.0))  # |ω| dt = 1 rad > 0.5
        _, _, q_pin, _, _ = self._prepare(body)
        expected = self.Rotation.from_rotvec([0.0, 0.0, 0.5]) * self.Rotation.from_quat(body["q0"])
        np.testing.assert_allclose(
            self.Rotation.from_quat(q_pin.numpy()[0]).as_matrix(), expected.as_matrix(), atol=1e-12
        )

    def test_rotation_noise_keeps_frame(self):
        """Keep the step-start orientation as the pinned frame when the rotation is numerical noise."""
        body = self._body(w0=(1.0e-13, 0.0, 0.0))
        _, _, q_pin, _, _ = self._prepare(body)
        np.testing.assert_allclose(q_pin.numpy()[0], body["q0"] / np.linalg.norm(body["q0"]), rtol=0.0, atol=1e-15)


def _dense_jacobian(row_slot, row_jac, slot_count):
    J = np.zeros((row_slot.shape[0], 6 * slot_count))
    for r, k in np.argwhere(row_slot >= 0):
        J[r, 6 * row_slot[r, k] : 6 * row_slot[r, k] + 6] += row_jac[r, k]
    return J


class TestFBARigidRows(unittest.TestCase):
    """Delassus assembly and force gather of constraint rows on body slots."""

    def test_block_inverses(self):
        """Invert the 6 x 6 blocks of a block-diagonal rigid matrix, and refuse one that couples two bodies."""
        import scipy.sparse as sp

        rng = np.random.default_rng(0)
        blocks = rng.standard_normal((3, 6, 6))
        blocks = blocks @ blocks.transpose(0, 2, 1) + 6.0 * np.eye(6)
        A = sp.block_diag(blocks, format="csr")
        np.testing.assert_allclose(rigid_block_inverses(A), np.linalg.inv(blocks), rtol=1e-10)
        A = A.tolil()
        A[0, 6] = A[6, 0] = 1.0  # couples two bodies
        with self.assertRaises(ValueError):
            rigid_block_inverses(A.tocsr())

    def test_delassus_and_gather_match_dense_algebra(self):
        """Match the rigid Delassus matrix ``J M⁻¹ Jᵀ`` plus the soft block, and the row force gather ``b + Jᵀ f``, with
        dense algebra."""
        from scipy.linalg import block_diag

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        rng = np.random.default_rng(1)
        m, n_soft = 3, 3
        row_slot = np.array([[-1, -1], [-1, 0], [1, 2], [2, -1], [0, 1], [-1, -1], [2, 0], [1, -1]], dtype=np.int32)
        n = row_slot.shape[0]
        row_jac = rng.standard_normal((n, 2, 6))
        inv_mass = rng.standard_normal((m, 6, 6))
        inv_mass = inv_mass @ inv_mass.transpose(0, 2, 1) + np.eye(6)
        W_soft = rng.standard_normal((n_soft, n_soft))
        W_soft = W_soft @ W_soft.T
        J = _dense_jacobian(row_slot, row_jac, m)
        expected = J @ block_diag(*inv_mass) @ J.T
        expected[:n_soft, :n_soft] += W_soft

        row_slot_d = wp.array(row_slot, dtype=wp.int32, device=device)
        row_jac_d = wp.array(row_jac, dtype=wp.spatial_vectord, device=device)
        W = wp.empty((n, n), dtype=wp.float64, device=device)
        wp.launch(
            rigid.rigid_delassus_kernel,
            dim=(n, n),
            inputs=[
                wp.array(W_soft, dtype=wp.float64, device=device),
                n_soft,
                n,
                row_slot_d,
                row_jac_d,
                wp.array(inv_mass, dtype=wp.spatial_matrixd, device=device),
            ],
            outputs=[W],
            device=device,
        )
        np.testing.assert_allclose(W.numpy(), expected, rtol=1e-12, atol=1e-12)

        force, b_const = rng.standard_normal(n), rng.standard_normal(6 * m)
        incidence = [wp.array(a, dtype=wp.int32, device=device) for a in rigid.row_incidence(row_slot, m)]
        b = wp.empty(6 * m, dtype=wp.float64, device=device)
        wp.launch(
            rigid.row_force_rhs_kernel,
            dim=m,
            inputs=[
                *incidence,
                row_jac_d,
                wp.array(force, dtype=wp.float64, device=device),
                wp.array(b_const, dtype=wp.float64, device=device),
            ],
            outputs=[b],
            device=device,
        )
        np.testing.assert_allclose(b.numpy(), b_const + J.T @ force, rtol=1e-12)

    def test_row_residual_shifts_to_free_dofs(self):
        """Shift the rigid row residuals to the free-flight DOFs, ``r = -value + J (u - u_free)``."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        rng = np.random.default_rng(4)
        row_slot = np.array([[-1, 0], [1, 2], [2, -1], [-1, -1]], dtype=np.int32)
        row_jac, value = rng.standard_normal((4, 2, 6)), rng.standard_normal(4)
        u, u_free = rng.standard_normal(18), rng.standard_normal(18)
        r = wp.empty(4, dtype=wp.float64, device=device)
        wp.launch(
            rigid.rigid_row_residual_kernel,
            dim=4,
            inputs=[
                wp.array(value, dtype=wp.float64, device=device),
                wp.array(row_slot, dtype=wp.int32, device=device),
                wp.array(row_jac, dtype=wp.spatial_vectord, device=device),
                wp.array(u, dtype=wp.float64, device=device),
                wp.array(u_free, dtype=wp.float64, device=device),
            ],
            outputs=[r],
            device=device,
        )
        np.testing.assert_allclose(r.numpy(), -value + _dense_jacobian(row_slot, row_jac, 3) @ (u - u_free), rtol=1e-12)


class TestSolverFBARigidStep(unittest.TestCase):
    """SolverFBA integrates dynamic bodies and passes kinematic ones through."""

    @classmethod
    def setUpClass(cls):
        from scipy.spatial.transform import Rotation

        wp.init()
        cls.Rotation = Rotation

    def _model(self, inertia, gravity=-9.81, cloth=True):
        import newton  # noqa: PLC0415

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
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
                tri_ke=50.0,
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
        """Move a body spinning about a principal axis as semi-implicit Euler moves its center of mass, with constant
        spin."""
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
        """Step a model with a rigid body and no particles under gravity, and refuse a model with neither."""
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
        """Conserve the angular momentum and energy of a torque-free body tumbling near its unstable intermediate axis
        to 0.5%."""
        inertia = np.diag([0.1, 0.2, 0.3])
        w0 = np.array([0.3, 5.0, 0.2])  # near the unstable intermediate axis
        state = self._run(self._model(inertia, gravity=0.0), (0.0, 0.0, 0.0, *w0), 100)

        R = self.Rotation.from_quat(state.body_q.numpy()[0, 3:]).as_matrix()
        w = state.body_qd.numpy()[0, 3:].astype(np.float64)
        L0, L = inertia @ w0, R @ inertia @ R.T @ w
        self.assertLess(np.linalg.norm(L - L0) / np.linalg.norm(L0), 5e-3)
        self.assertLess(abs(w @ R @ inertia @ R.T @ w / (w0 @ inertia @ w0) - 1.0), 5e-3)

    def test_mass_change_refactors_the_rigid_system(self):
        """Fall freely after the body's mass and inertia double, once ``notify_model_changed`` reports them."""
        import newton  # noqa: PLC0415
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._model(np.diag([0.1, 0.2, 0.3]))
        solver = SolverFBA(model, iterations=1)
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, DT)
        s0, s1 = s1, s0
        for name, scale in (
            ("body_mass", 2.0),
            ("body_inertia", 2.0),
            ("body_inv_mass", 0.5),
            ("body_inv_inertia", 0.5),
        ):
            getattr(model, name).assign(scale * getattr(model, name).numpy())
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        v = s0.body_qd.numpy()[0, :3].astype(np.float64)
        solver.step(s0, s1, None, None, DT)
        g = model.gravity.numpy()[0].astype(np.float64)
        np.testing.assert_allclose(s1.body_qd.numpy()[0, :3], v + DT * g, atol=1e-5)

    def test_kinematic_and_external_bodies_untouched(self):
        """Leave kinematic bodies in place, and the output poses of bodies another solver integrates untouched."""
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
