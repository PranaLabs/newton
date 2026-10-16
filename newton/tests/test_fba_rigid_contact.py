# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA rigid contacts."""

import types
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import rigid_contact as RC
from newton._src.solvers.fba.rigid import retract_rigid_kernel
from newton._src.solvers.fba.rigid_contact import gather_rigid_contacts

_FIELDS = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal")


def _slots(model):
    dynamic = (model.body_flags.numpy() & int(newton.BodyFlags.DYNAMIC)) != 0
    slot = np.full(model.body_count, -1, dtype=np.int32)
    slot[dynamic] = np.arange(np.count_nonzero(dynamic))
    return slot


def _box_scene(device):
    """A dynamic box resting on the ground, its COM off the body origin, and a kinematic box beside it."""
    cfg = newton.ModelBuilder.ShapeConfig
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=cfg(mu=0.2))
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(box, xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()), cfg=cfg(mu=0.8))
    kinematic = builder.add_body(xform=wp.transform((3.0, 0.0, 0.5), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(kinematic)
    model = builder.finalize(device=device)
    state = model.state()
    return model, state, model.collide(state)


class TestGatherRigidContacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_box_on_ground(self):
        model, state, contacts = _box_scene(self.device)
        rc = gather_rigid_contacts(model, contacts, _slots(model), self.device)
        self.assertGreaterEqual(rc.count, 4)
        # Only the dynamic box: the kinematic box on the ground has nothing to integrate.
        body = rc.body.numpy()
        np.testing.assert_array_equal(np.sort(body, axis=1), np.tile([-1, 0], (rc.count, 1)))
        np.testing.assert_array_equal(rc.slot, np.where(body >= 0, 0, -1))
        np.testing.assert_allclose(rc.mu.numpy(), 0.5, rtol=1e-6)

        # The world anchors of both sides coincide on the box bottom.
        tf, com = state.body_q.numpy()[0], model.body_com.numpy()[0]
        world = rc.anchor.numpy().astype(np.float64)
        on_box = body >= 0
        world[on_box] = [
            wp.transform_point(wp.transform(*np.split(tf, [3])), wp.vec3(*(a + com))) for a in world[on_box]
        ]
        np.testing.assert_allclose(world[:, 0], world[:, 1], atol=1e-5)
        np.testing.assert_allclose(world[:, 0, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(np.abs(rc.normal.numpy()), np.tile([0.0, 0.0, 1.0], (rc.count, 1)), atol=1e-6)

    def test_order_does_not_depend_on_emission(self):
        model, _, contacts = _box_scene(self.device)
        n = int(contacts.rigid_contact_count.numpy()[0])
        perm = np.random.default_rng(0).permutation(n)
        shuffled = types.SimpleNamespace(rigid_contact_count=contacts.rigid_contact_count)
        for f in _FIELDS:
            a = getattr(contacts, f"rigid_contact_{f}")
            setattr(shuffled, f"rigid_contact_{f}", wp.array(a.numpy()[:n][perm], dtype=a.dtype, device=self.device))
        slot = _slots(model)
        ref, out = (gather_rigid_contacts(model, c, slot, self.device) for c in (contacts, shuffled))
        np.testing.assert_array_equal(out.slot, ref.slot)
        for f in ("body", "anchor", "normal", "mu"):
            np.testing.assert_array_equal(getattr(out, f).numpy(), getattr(ref, f).numpy())

    def test_no_integrated_body(self):
        model, _, contacts = _box_scene(self.device)
        self.assertIsNone(gather_rigid_contacts(model, contacts, np.full(model.body_count, -1), self.device))


class TestRigidContactRows(unittest.TestCase):
    """Rows of a static-dynamic, a dynamic-dynamic and a dynamic-kinematic contact."""

    def setUp(self):
        scope = wp.ScopedDevice("cpu")
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        builder = newton.ModelBuilder()
        poses = [
            ((0.2, 0.1, 0.5), (0.3, -0.2, 0.4)),
            ((0.3, -0.2, 1.4), (-0.6, 0.1, 0.9)),
            ((1.5, 0.0, 0.5), (0.0, 0.0, 1.0)),
        ]
        for k, (p, rpy) in enumerate(poses):
            builder.add_body(
                xform=wp.transform(p, wp.quat_rpy(*rpy)),
                com=wp.vec3(0.05, -0.02, 0.0),
                mass=1.0,
                inertia=wp.mat33(np.eye(3)),
                lock_inertia=True,
                is_kinematic=k == 2,
            )
        self.model = builder.finalize(device="cpu")
        self.state = self.model.state()
        self.v_kin = np.array([0.4, -0.3, 0.2, 0.5, 0.1, -0.7])
        self.state.body_qd.assign([[0.0] * 6, [0.0] * 6, self.v_kin])
        rng = np.random.default_rng(2)
        normal = rng.standard_normal((3, 3))
        self.rc = types.SimpleNamespace(
            body=wp.array([[-1, 0], [0, 1], [1, 2]], dtype=wp.int32, device="cpu"),
            anchor=wp.array(rng.uniform(-0.5, 0.5, (3, 2, 3)), dtype=wp.vec3, device="cpu"),
            normal=wp.array(normal / np.linalg.norm(normal, axis=1, keepdims=True), dtype=wp.vec3, device="cpu"),
        )
        self.slot = wp.array([0, 1, -1], dtype=wp.int32, device="cpu")
        tf = self.state.body_q.numpy().astype(np.float64)
        self.q0 = tf[:, 3:] / np.linalg.norm(tf[:, 3:], axis=1, keepdims=True)
        com = self.model.body_com.numpy()
        self.x0 = tf[:, :3] + [wp.quat_rotate(wp.quatd(*q), wp.vec3d(*c)) for q, c in zip(self.q0, com, strict=True)]

    def _world(self, body, r):
        """World anchor of a side at the step start."""
        if body < 0:
            return np.asarray(r, dtype=np.float64)
        return self.x0[body] + wp.quat_rotate(wp.quatd(*self.q0[body]), wp.vec3d(*r))

    def _gap(self, u, dt):
        x0, q0 = wp.array(self.x0[:2], dtype=wp.vec3d), wp.array(self.q0[:2], dtype=wp.quatd)
        x, q = wp.empty(2, dtype=wp.vec3d), wp.empty(2, dtype=wp.quatd)
        scratch = [wp.empty(2, dtype=wp.transform), wp.empty(2, dtype=wp.spatial_vector)]
        wp.launch(
            retract_rigid_kernel,
            dim=2,
            inputs=[
                wp.array([0, 1], dtype=wp.int32),
                wp.array(u, dtype=wp.float64),
                q0,
                x0,
                self.state.body_q,
                self.model.body_com,
                wp.float64(1.0),
            ],
            outputs=[*scratch, x, q],
            device="cpu",
        )
        gap, rc, state = wp.empty(9, dtype=wp.float64), self.rc, self.state
        wp.launch(
            RC.rigid_contact_gap_kernel,
            dim=3,
            inputs=[
                *(rc.body, rc.anchor, rc.normal, self.slot, x, q, x0),
                *(state.body_q, state.body_qd, self.model.body_com, wp.float64(dt), 0),
            ],
            outputs=[gap],
            device="cpu",
        )
        return gap.numpy()

    def _u0(self):
        """DOFs of the step-start pose, with the pinned frame there."""
        u0 = np.zeros(12)
        for k in range(2):
            u0[6 * k : 6 * k + 3] = wp.quat_rotate_inv(wp.quatd(*self.q0[k]), wp.vec3d(*self.x0[k]))
        return u0

    def test_gap_at_step_start(self):
        """Gaps are n · (a1 - a0); only the kinematic side slips, by dt times its anchor velocity."""
        dt = 0.01
        gap = self._gap(self._u0(), dt).reshape(3, 3)
        body, anchor, normal = self.rc.body.numpy(), self.rc.anchor.numpy(), self.rc.normal.numpy()
        world = np.array([[self._world(b, r) for b, r in zip(body[c], anchor[c], strict=True)] for c in range(3)])
        np.testing.assert_allclose(gap[:, 0], np.einsum("ij,ij->i", normal, world[:, 1] - world[:, 0]), atol=1e-6)
        np.testing.assert_allclose(gap[:2, 1:], 0.0, atol=1e-12)
        slip = dt * (self.v_kin[:3] + np.cross(self.v_kin[3:], world[2, 1] - self.x0[2]))
        tangential = slip - np.dot(slip, normal[2]) * normal[2]
        self.assertAlmostEqual(np.linalg.norm(gap[2, 1:]), np.linalg.norm(tangential), delta=1e-9)

    def test_jacobian_matches_finite_differences(self):
        row_slot, row_jac = wp.empty((9, 2), dtype=wp.int32), wp.empty((9, 2), dtype=wp.spatial_vectord)
        q_pin = wp.array(self.q0[:2], dtype=wp.quatd)
        wp.launch(
            RC.rigid_contact_jacobian_kernel,
            dim=(3, 3),
            inputs=[self.rc.body, self.rc.anchor, self.rc.normal, self.slot, q_pin, 0],
            outputs=[row_slot, row_jac],
            device="cpu",
        )
        np.testing.assert_array_equal(row_slot.numpy(), [[-1, 0]] * 3 + [[0, 1]] * 3 + [[1, -1]] * 3)
        jac = np.zeros((9, 12))
        for r, k in np.argwhere(row_slot.numpy() >= 0):
            s = row_slot.numpy()[r, k]
            jac[r, 6 * s : 6 * s + 6] += row_jac.numpy()[r, k]
        u0, eps = self._u0(), 1e-6
        fd = np.stack([(self._gap(u0 + eps * e, 0.01) - self._gap(u0 - eps * e, 0.01)) / (2 * eps) for e in np.eye(12)])
        np.testing.assert_allclose(jac, fd.T, atol=1e-8)


class TestParticleContactBodyRows(unittest.TestCase):
    """Body side of particle contacts on two integrated bodies and on the static ground."""

    def setUp(self):
        scope = wp.ScopedDevice("cpu")
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        builder = newton.ModelBuilder()
        shapes = [builder.add_ground_plane()]
        for p, rpy in (((0.2, 0.1, 0.5), (0.3, -0.2, 0.4)), ((0.3, -0.2, 1.4), (-0.6, 0.1, 0.9))):
            body = builder.add_body(xform=wp.transform(p, wp.quat_rpy(*rpy)), com=wp.vec3(0.05, -0.02, 0.0))
            shapes.append(builder.add_shape_box(body))
        self.model = model = builder.finalize(device="cpu")
        rng = np.random.default_rng(3)
        frame = np.linalg.qr(rng.standard_normal((3, 3, 3)))[0]  # per contact: normal, t1, t2
        body_pos = rng.uniform(-0.5, 0.5, (3, 3))
        tf = model.body_q.numpy().astype(np.float64)
        world = [wp.transform_point(wp.transform(*np.split(tf[k], [3])), wp.vec3(*body_pos[k])) for k in range(2)]
        self.contact = [
            wp.array(shapes[1:] + shapes[:1], dtype=wp.int32),
            wp.array(body_pos, dtype=wp.vec3),
            *(wp.array(frame[:, :, k], dtype=wp.vec3) for k in range(3)),
        ]
        self.world_anchor = wp.array([*world, body_pos[2]], dtype=wp.vec3)
        self.v_anchor = wp.array(rng.standard_normal((3, 3)), dtype=wp.vec3d)
        self.slot = wp.array([0, 1], dtype=wp.int32)
        self.bodies = [model.shape_body, self.slot, model.body_com]
        self.q0 = tf[:, 3:] / np.linalg.norm(tf[:, 3:], axis=1, keepdims=True)
        com = model.body_com.numpy()
        self.x0 = tf[:, :3] + [wp.quat_rotate(wp.quatd(*q), wp.vec3d(*c)) for q, c in zip(self.q0, com, strict=True)]

    def _residual(self, u, u_free, row_jac, dt=0.01):
        x, q = wp.empty(2, dtype=wp.vec3d), wp.empty(2, dtype=wp.quatd)
        scratch = [wp.empty(2, dtype=wp.transform), wp.empty(2, dtype=wp.spatial_vector)]
        x0, q0 = wp.array(self.x0, dtype=wp.vec3d), wp.array(self.q0, dtype=wp.quatd)
        wp.launch(
            retract_rigid_kernel,
            dim=2,
            inputs=[self.slot, wp.array(u, dtype=wp.float64), q0, x0, self.model.body_q, self.model.body_com, 1.0],
            outputs=[*scratch, x, q],
        )
        r = wp.zeros(9, dtype=wp.float64)
        wp.launch(
            RC.particle_contact_body_residual_kernel,
            dim=(3, 3),
            inputs=[
                *(*self.contact, self.world_anchor, self.v_anchor, *self.bodies, x, q, row_jac),
                *(wp.array(u, dtype=wp.float64), wp.array(u_free, dtype=wp.float64), wp.float64(dt)),
            ],
            outputs=[r],
        )
        return r.numpy()

    def test_anchor_follows_body(self):
        row_slot, row_jac = wp.empty((9, 2), dtype=wp.int32), wp.empty((9, 2), dtype=wp.spatial_vectord)
        q_pin = wp.array(self.q0, dtype=wp.quatd)
        wp.launch(
            RC.particle_contact_body_jacobian_kernel,
            dim=(3, 3),
            inputs=[*self.contact, *self.bodies, q_pin],
            outputs=[row_slot, row_jac],
        )
        np.testing.assert_array_equal(row_slot.numpy(), [[0, -1]] * 3 + [[1, -1]] * 3 + [[-1, -1]] * 3)
        jac = np.zeros((9, 12))
        for r, s in enumerate(row_slot.numpy()[:6, 0]):
            jac[r, 6 * s : 6 * s + 6] = row_jac.numpy()[r, 0]
        u0 = np.zeros(12)
        for k in range(2):
            u0[6 * k : 6 * k + 3] = wp.quat_rotate_inv(wp.quatd(*self.q0[k]), wp.vec3d(*self.x0[k]))
        no_jac = wp.zeros((9, 2), dtype=wp.spatial_vectord)

        # At the step start only the friction rows on bodies drop the dt v shift (float32 anchors).
        dt, v = 0.01, self.v_anchor.numpy()
        frame = np.stack([a.numpy() for a in self.contact[2:]], axis=1)  # [contact, row, xyz]
        expected = np.where(np.arange(9) % 3 > 0, -dt * np.einsum("crx,cx->cr", frame, v).ravel(), 0.0)
        expected[6:] = 0.0
        np.testing.assert_allclose(self._residual(u0, u0, no_jac, dt), expected, atol=2e-7)

        # The anchor moves with the body: d(residual)/du = -J_body.
        eps = 1e-6
        fd = [
            (self._residual(u0 + eps * e, u0, no_jac) - self._residual(u0 - eps * e, u0, no_jac)) / (2 * eps)
            for e in np.eye(12)
        ]
        np.testing.assert_allclose(np.stack(fd, axis=1), -jac, atol=1e-8)

        # The shift adds J (u - u_free).
        rng = np.random.default_rng(5)
        u, u_free = u0 + 1e-3 * rng.standard_normal(12), u0 + 1e-3 * rng.standard_normal(12)
        shift = self._residual(u, u_free, row_jac) - self._residual(u, u_free, no_jac)
        np.testing.assert_allclose(shift, jac @ (u - u_free), atol=1e-14)


class TestSolverFBARigidContacts(unittest.TestCase):
    """A 1 m box on the ground; tilting gravity turns the ground into an incline."""

    MU, H = 0.5, 0.01

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _run(self, gravity, v0, steps):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        cfg = newton.ModelBuilder.ShapeConfig(mu=self.MU)
        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=cfg)
        builder.add_shape_box(builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity())), cfg=cfg)
        model = builder.finalize()
        model.gravity.assign([wp.vec3(*gravity)])
        solver = SolverFBA(model, iterations=10)
        s0, s1 = model.state(), model.state()
        s0.body_qd.assign([[*v0, 0.0, 0.0, 0.0]])
        contacts, traj = model.contacts(), []
        for _ in range(steps):
            model.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, self.H)
            s0, s1 = s1, s0
            traj.append(np.concatenate([s0.body_q.numpy()[0], s0.body_qd.numpy()[0]]))
        return solver, model, np.array(traj)

    def test_box_rests_on_ground(self):
        solver, model, traj = self._run((0.0, 0.0, -9.81), (0.0, 0.0, 0.0), 100)
        np.testing.assert_allclose(traj[:, :3], np.tile([0.0, 0.0, 0.5], (100, 1)), atol=1e-6)
        np.testing.assert_allclose(traj[:, 7:], 0.0, atol=1e-6)
        rows = 3 * solver._rigid_contacts.count
        weight = model.body_mass.numpy()[0] * 9.81
        self.assertAlmostEqual(solver._nsn_lam_apply_d.numpy()[:rows:3].sum() / weight, 1.0, delta=1e-6)

    def test_sliding_box_decelerates_at_mu_g(self):
        g, v0, n = 9.81, 2.0, 60
        _, _, traj = self._run((0.0, 0.0, -g), (v0, 0.0, 0.0), n)
        t = self.H * np.arange(1, n + 1)
        sliding = t < v0 / (self.MU * g) - 2 * self.H
        np.testing.assert_allclose(traj[sliding, 7], v0 - self.MU * g * t[sliding], atol=5e-3)
        np.testing.assert_allclose(traj[-10:, 7:10], 0.0, atol=1e-6)  # static friction holds it
        self.assertAlmostEqual(traj[-1, 0], v0 * v0 / (2.0 * self.MU * g), delta=v0 * self.H)

    def test_incline_sticks_below_friction_angle(self):
        n = 50
        for tan_theta in (0.3, 0.8):
            theta = np.arctan(tan_theta)
            _, _, traj = self._run(9.81 * np.array([np.sin(theta), 0.0, -np.cos(theta)]), (0.0, 0.0, 0.0), n)
            if tan_theta < self.MU:
                self.assertLess(abs(traj[-1, 0]), 1e-4)
            else:
                a = 9.81 * (np.sin(theta) - self.MU * np.cos(theta))
                # Breaking away from rest is the slowest step for the FB-Newton iteration.
                self.assertAlmostEqual(traj[0, 7] / (a * self.H), 1.0, delta=0.05)
                np.testing.assert_allclose(np.diff(traj[:, 7]), a * self.H, rtol=5e-3)


class TestSolverFBAArticulationContacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def test_hinged_plank_rests_on_ground(self):
        """A frictionless plank hinged at one end rests its far end on the ground.

        Its revolute rows join the contact NSN, and the normal force balances
        the weight's moment about the hinge. A pendulum away from any contact
        keeps the Cholesky joint solve.
        """
        from scipy.optimize import brentq

        from newton.solvers import SolverFBA  # noqa: PLC0415

        hx, hz, pivot = 0.5, 0.02, 0.3
        alpha = brentq(lambda a: pivot - 2 * hx * np.sin(a) - hz * np.cos(a), 0.0, 1.0)  # far corner on the ground
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(alpha))
        center = wp.vec3(0.0, 0.0, pivot) + wp.quat_rotate(q, wp.vec3(hx, 0.0, 0.0))
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.0)
        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=cfg)
        plank = builder.add_link(xform=wp.transform(center, q))
        builder.add_shape_box(plank, hx=hx, hy=0.1, hz=hz, cfg=cfg)
        bob = builder.add_link(
            xform=wp.transform((3.0, 0.0, 2.0)), mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True
        )
        for link, p in ((plank, (0.0, 0.0, pivot)), (bob, (2.0, 0.0, 2.0))):
            x_cj = wp.transform((-hx if link == plank else -1.0, 0.0, 0.0))
            joint = builder.add_joint_revolute(
                -1, link, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform(p), child_xform=x_cj
            )
            builder.add_articulation([joint])
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s0, s1 = model.state(), model.state()
        contacts = model.contacts()
        for _ in range(50):
            model.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 0.01)
            s0, s1 = s1, s0

        tf = s0.body_q.numpy().astype(np.float64)
        np.testing.assert_allclose(tf[0], [*center, *q], atol=1e-6)
        np.testing.assert_allclose(s0.body_qd.numpy()[0], 0.0, atol=1e-5)
        self.assertEqual(solver._rc_joint_rows_d.shape[0], 5)  # the plank's revolute only
        for k, (anchor, pivot_world) in enumerate(
            (((-hx, 0.0, 0.0), (0.0, 0.0, pivot)), ((-1.0, 0.0, 0.0), (2.0, 0.0, 2.0)))
        ):
            hinge = tf[k, :3] + wp.quat_rotate(wp.quatd(*tf[k, 3:]), wp.vec3d(*anchor))
            np.testing.assert_allclose(hinge, pivot_world, atol=1e-6)
        self.assertLess(tf[1, 2], 1.9)  # the pendulum swings

        normal = solver._nsn_lam_apply_d.numpy()[: 3 * solver._rigid_contacts.count : 3].sum()
        x_contact = 2 * hx * np.cos(alpha) - hz * np.sin(alpha)
        weight = model.body_mass.numpy()[0] * 9.81
        self.assertAlmostEqual(normal / (weight * hx * np.cos(alpha) / x_contact), 1.0, delta=1e-4)

    def test_heavy_chain_rests_on_ground(self):
        """Eight links lying on the ground, each three times heavier than the last.

        The joint rows join the contact NSN, and the 2000:1 mass ratio spreads
        their block's spectrum; the joint block preconditioner keeps it solvable.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        length, half_height, parent, joints = 0.4, 0.05, -1, []
        for k in range(8):
            link = builder.add_link(xform=wp.transform((k * length + length / 2, 0.0, half_height)))
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0 * 3.0**k)
            builder.add_shape_box(link, hx=length / 2, hy=0.05, hz=half_height, cfg=cfg)
            x_pj = wp.transform((0.0, 0.0, half_height) if parent < 0 else (length / 2, 0.0, 0.0))
            x_cj = wp.transform((-length / 2, 0.0, 0.0))
            joints.append(
                builder.add_joint_revolute(parent, link, axis=(0.0, 1.0, 0.0), parent_xform=x_pj, child_xform=x_cj)
            )
            parent = link
        builder.add_articulation(joints)
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s0, s1 = model.state(), model.state()
        contacts = model.contacts()
        for _ in range(20):
            model.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 0.005)
            s0, s1 = s1, s0
        np.testing.assert_allclose(s0.body_q.numpy()[:, 2], half_height, atol=1e-6)

    def test_plank_leaning_on_box_stays_at_rest(self):
        """A hinged plank leans on the top edge of a box that rests on the ground.

        The hinge, the edge contacts and the ground contacts close a loop, which
        makes the contact NSN's Newton matrix nearly singular. With its contact
        rows regularized the plank and the box settle instead of being thrown.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=cfg)
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.25)))
        builder.add_shape_box(box, hx=0.4, hy=0.4, hz=0.25, cfg=cfg)
        hinge, half = wp.vec3(1.1, 0.0, 0.05), wp.vec3(0.04, 0.3, 0.6)
        lean = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.95)  # just above the box's edge
        plank = builder.add_link(xform=wp.transform(hinge + wp.quat_rotate(lean, wp.vec3(0.0, 0.0, half[2])), lean))
        builder.add_shape_box(plank, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
        joint = builder.add_joint_revolute(
            -1,
            plank,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(hinge),
            child_xform=wp.transform((0, 0, -half[2])),
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s0, s1 = model.state(), model.state()
        start = s0.body_q.numpy()[:, :3]
        contacts = model.contacts()
        speed = []
        for _ in range(60):
            model.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / 120.0)
            s0, s1 = s1, s0
            speed.append(np.abs(s0.body_qd.numpy()).max())

        self.assertEqual(solver._rc_joint_rows_d.shape[0], 5)
        self.assertLess(max(speed[20:]), 1e-3)
        np.testing.assert_allclose(s0.body_q.numpy()[:, :3], start, atol=2e-3)


class TestSolverFBATwoWayContacts(unittest.TestCase):
    """A cloth patch over a free box: particle contacts push on both."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _run(self, gravity, cloth_z, v_cloth, box_density, particle_mass, steps, ground=False):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(gravity=gravity)
        if ground:
            builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(box, cfg=newton.ModelBuilder.ShapeConfig(density=box_density))
        grid = {"dim_x": 4, "dim_y": 4, "cell_x": 0.1, "cell_y": 0.1, "tri_ke": 1.0e3, "tri_ka": 1.0e3, "edge_ke": 1.0}
        pos, vel = wp.vec3(-0.2, -0.2, cloth_z), wp.vec3(0.0, 0.0, v_cloth)
        builder.add_cloth_grid(pos=pos, rot=wp.quat_identity(), vel=vel, mass=particle_mass, **grid)
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s0, s1 = model.state(), model.state()
        contacts, momentum = model.contacts(), []
        m_p, m_b = model.particle_mass.numpy(), model.body_mass.numpy()[0]
        for _ in range(steps):
            model.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 0.005)
            s0, s1 = s1, s0
            momentum.append(m_p @ s0.particle_qd.numpy() + m_b * s0.body_qd.numpy()[0, :3])
        return solver, s0, np.array(momentum), m_p.sum(), m_b

    def test_collision_conserves_momentum(self):
        """Without gravity a 0.5 kg cloth hits a 1 kg box at 1 m/s and both move on at 1/3 m/s."""
        solver, state, momentum, m_cloth, m_box = self._run(0.0, 1.05, -1.0, 1.0, 0.02, 60)
        self.assertTrue(solver._two_way)
        # Each step rounds the float32 particle positions, by up to m_cloth ulp(1) / dt of momentum.
        np.testing.assert_allclose(momentum[:20], np.tile([0.0, 0.0, -m_cloth], (20, 1)), atol=5e-5)
        self.assertAlmostEqual(state.body_qd.numpy()[0, 2], -m_cloth / (m_cloth + m_box), delta=1e-3)

    def test_box_carries_cloth_weight(self):
        """A cloth resting on a box on the ground: the ground holds both weights."""
        solver, state, _, m_cloth, m_box = self._run(-9.81, 1.005, 0.0, 100.0, 0.2, 200, ground=True)
        m_s, m_r = solver._rc_counts
        rows = slice(3 * m_s, 3 * (m_s + m_r), 3)
        ground = solver._nsn_lam_apply_d.numpy()[rows].sum()
        self.assertAlmostEqual(ground / ((m_box + m_cloth) * 9.81), 1.0, delta=1e-4)
        np.testing.assert_allclose(state.body_q.numpy()[0], [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(state.particle_q.numpy()[:, 2], 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
