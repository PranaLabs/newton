# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA joint constraint rows."""

import itertools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import joints as J
from newton._src.solvers.fba.joints import BilateralSystem, build_joint_rows, locked_directions
from newton._src.solvers.fba.linear_solver import build_rigid_system, rigid_block_inverses
from newton._src.solvers.fba.rigid import retract_rigid_kernel

Dof = newton.ModelBuilder.JointDofConfig


def _link(builder, **kwargs):
    return builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True, **kwargs)


def _rows(model, device="cpu"):
    slot = np.where(model.body_flags.numpy() & int(newton.BodyFlags.DYNAMIC), 0, -1)
    slot[slot == 0] = np.arange(np.count_nonzero(slot == 0))
    return build_joint_rows(model, slot, device)


def _row_args(rows, slot):
    return [
        *(rows.row_joint, rows.row_angular, rows.row_kind, rows.row_dir, rows.row_axis2),
        *(rows.parent, rows.child, rows.X_pj, rows.X_cj, slot),
    ]


def _add_satisfied_chain(builder):
    """Prismatic (world, slid 0.3) -> revolute (turned 40 deg) -> ball, all satisfied."""
    q1 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7)
    q2 = wp.quat_rpy(0.3, -0.5, 1.1)
    x1 = wp.vec3(0.3, 0.5, 0.0) - wp.quat_rotate(q1, wp.vec3(0.0, -0.4, 0.0))
    x2 = x1 + wp.quat_rotate(q1, wp.vec3(0.2, 0.3, 0.0)) - wp.quat_rotate(q2, wp.vec3(0.0, 0.1, 0.2))
    l0 = _link(builder, xform=wp.transform((0.3, 0.0, 0.0)))
    l1 = _link(builder, xform=wp.transform(x1, q1), com=wp.vec3(0.05, 0.0, 0.0))
    l2 = _link(builder, xform=wp.transform(x2, q2))
    x_pj, x_cj = wp.transform((0.0, 0.5, 0.0)), wp.transform((0.0, -0.4, 0.0))
    joints = [
        builder.add_joint_prismatic(-1, l0, axis=(1.0, 0.0, 0.0)),
        builder.add_joint_revolute(l0, l1, axis=(0.0, 0.0, 1.0), parent_xform=x_pj, child_xform=x_cj),
        builder.add_joint_ball(
            l1, l2, parent_xform=wp.transform((0.2, 0.3, 0.0)), child_xform=wp.transform((0.0, 0.1, 0.2))
        ),
    ]
    builder.add_articulation(joints)


def _com_poses(model):
    """Float64 COM positions and unit orientations of all bodies."""
    tf = model.body_q.numpy().astype(np.float64)
    q = tf[:, 3:] / np.linalg.norm(tf[:, 3:], axis=1, keepdims=True)
    com = model.body_com.numpy()
    return tf[:, :3] + np.array([wp.quat_rotate(wp.quatd(*a), wp.vec3d(*c)) for a, c in zip(q, com, strict=True)]), q


def _jacobian(model, rows, slot, x, q, device):
    """Row slots and coefficients at the given poses, and the dense Jacobian over all slots."""
    R = rows.row_joint.shape[0]
    row_slot = wp.empty((R, 2), dtype=wp.int32, device=device)
    row_jac = wp.empty((R, 2), dtype=wp.spatial_vectord, device=device)
    poses = [wp.array(x, dtype=wp.vec3d, device=device), wp.array(q, dtype=wp.quatd, device=device)]
    wp.launch(
        J.joint_jacobian_kernel,
        dim=R,
        inputs=[*_row_args(rows, slot), *poses, model.body_q, model.body_com],
        outputs=[row_slot, row_jac],
        device=device,
    )
    dense = np.zeros((R, 6 * len(x)))
    for r, (slots, coeffs) in enumerate(zip(row_slot.numpy(), row_jac.numpy(), strict=True)):
        for k, c in zip(slots, coeffs, strict=True):
            if k >= 0:
                dense[r, 6 * k : 6 * k + 6] += c
    return row_slot, row_jac, dense


class TestFBAJointRows(unittest.TestCase):
    def test_locked_directions(self):
        """Return an orthonormal basis of the directions orthogonal to zero to three free axes."""
        for axes, rank in (([], 0), ([[0.0, 0.0, 2.0]], 1), ([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], 2), (np.eye(3), 3)):
            d = locked_directions(np.array(axes))
            self.assertEqual(d.shape, (3 - rank, 3))
            np.testing.assert_allclose(d @ d.T, np.eye(3 - rank), atol=1e-12)
            if rank:
                np.testing.assert_allclose(d @ np.array(axes).T, 0.0, atol=1e-12)

    def test_rows_per_joint_type(self):
        """Give ball, revolute and D6 joints their linear and angular rows along the locked directions, anchored
        relative to the child's center of mass."""
        builder = newton.ModelBuilder()
        l0, l1, l2 = _link(builder), _link(builder, com=wp.vec3(0.0, 0.5, 0.0)), _link(builder)
        joints = [
            builder.add_joint_ball(-1, l0),
            builder.add_joint_revolute(l0, l1, axis=(0.0, 0.0, 1.0), child_xform=wp.transform((0.0, 1.0, 0.0))),
            builder.add_joint_d6(
                l1,
                l2,
                linear_axes=[Dof(axis=(1.0, 0.0, 0.0)), Dof(axis=(0.0, 0.0, 1.0))],
                angular_axes=[Dof(axis=(0.0, 1.0, 0.0))],
            ),
        ]
        builder.add_articulation(joints)
        rows = _rows(builder.finalize())

        # ball: 3 linear; revolute: 3 linear + 2 angular; d6: 1 linear + 2 angular.
        joint, angular = rows.row_joint.numpy(), rows.row_angular.numpy()
        self.assertEqual(
            [np.count_nonzero((joint == k) & (angular == a)) for k in range(3) for a in (0, 1)], [3, 0, 3, 2, 1, 2]
        )
        d = rows.row_dir.numpy()
        np.testing.assert_allclose(d[(joint == 1) & (angular == 1)] @ [0.0, 0.0, 1.0], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(d[(joint == 2) & (angular == 0)]), [[0.0, 1.0, 0.0]], atol=1e-12)
        np.testing.assert_array_equal(rows.block_start, [0, 11])
        # Anchors are stored relative to the child COM.
        np.testing.assert_allclose(rows.X_cj.numpy()[1, :3], [0.0, 0.5, 0.0])

    def test_blocks_and_skipped_joints(self):
        """Group the rows of each articulation with integrated bodies into one block, skipping joints without an
        integrated body and free bodies."""
        builder = newton.ModelBuilder()
        chain = [_link(builder) for _ in range(4)]
        l4, kin = _link(builder), _link(builder, is_kinematic=True)
        joints = [builder.add_joint_fixed(-1, chain[0])]
        joints += [builder.add_joint_revolute(a, b) for a, b in itertools.pairwise(chain)]
        builder.add_articulation(joints)
        builder.add_articulation([builder.add_joint_prismatic(-1, l4, axis=(1.0, 0.0, 0.0))])
        builder.add_articulation([builder.add_joint_revolute(-1, kin)])  # no integrated body
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))  # free joint
        rows = _rows(builder.finalize())
        np.testing.assert_array_equal(rows.block_start, [0, 21, 26])
        np.testing.assert_array_equal(rows.row_joint.numpy(), np.repeat(np.arange(5), [6, 5, 5, 5, 5]))

    def test_rejects_distance_joint(self):
        """Refuse a distance joint on an integrated body."""
        builder = newton.ModelBuilder()
        l0 = _link(builder)
        builder.add_articulation([builder.add_joint_distance(-1, l0, min_distance=0.5, max_distance=1.0)])
        with self.assertRaises(NotImplementedError):
            _rows(builder.finalize())

    def test_rejects_kinematic_loops(self):
        """Refuse joints that lock more directions of a set of connected bodies than the bodies have: a four-bar
        linkage of revolute joints."""
        builder = newton.ModelBuilder()
        links = [_link(builder) for _ in range(3)]
        joints = [builder.add_joint_revolute(-1, links[0])]
        joints += [builder.add_joint_revolute(a, b) for a, b in itertools.pairwise(links)]
        builder.add_articulation(joints)
        builder.add_joint_revolute(-1, links[2])  # closes the loop
        with self.assertRaises(NotImplementedError):
            _rows(builder.finalize())

    def test_free_d6_joint_has_no_rows(self):
        """Give a D6 joint with every axis free no rows."""
        builder = newton.ModelBuilder()
        l0 = _link(builder)
        axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        linear, angular = [Dof(axis=a) for a in axes], [Dof(axis=a) for a in axes]
        builder.add_articulation([builder.add_joint_d6(-1, l0, linear_axes=linear, angular_axes=angular)])
        self.assertIsNone(_rows(builder.finalize()))


class TestFBAJointKernels(unittest.TestCase):
    def _residual(self, model, rows):
        """Row residuals at the model poses (no integrated slots: frames come from body_q)."""
        R = rows.row_joint.shape[0]
        none = wp.full(model.body_count, -1, dtype=wp.int32, device="cpu")
        res = wp.empty(R, dtype=wp.float64, device="cpu")
        args = _row_args(rows, none)
        empty = [wp.empty(0, dtype=wp.vec3d, device="cpu"), wp.empty(0, dtype=wp.quatd, device="cpu")]
        wp.launch(
            J.joint_residual_kernel,
            dim=R,
            inputs=[
                *(*args, *empty, model.body_q, model.body_com),
                *(wp.zeros(R, dtype=wp.float64, device="cpu"), wp.zeros(R, dtype=wp.quatd, device="cpu")),
                wp.zeros(R, dtype=wp.float64, device="cpu"),
            ],
            outputs=[res],
            device="cpu",
        )
        return res.numpy()

    def test_residual_measures_violation(self):
        """Measure the violation of a revolute joint about z, turned by phi about x and offset by delta, as delta and 2
        sin(phi/2) x."""
        phi, delta = 0.1, np.array([0.01, -0.02, 0.03])
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), phi)
        builder = newton.ModelBuilder()
        link = _link(builder, xform=wp.transform(wp.vec3(*delta) - wp.quat_rotate(q, wp.vec3(0.0, -0.5, 0.0)), q))
        builder.add_articulation([builder.add_joint_revolute(-1, link, child_xform=wp.transform((0.0, -0.5, 0.0)))])
        model = builder.finalize(device="cpu")
        rows = _rows(model)
        d, angular = rows.row_dir.numpy(), rows.row_angular.numpy() == 1
        expected = np.where(angular, d @ [2.0 * np.sin(phi / 2.0), 0.0, 0.0], d @ delta)
        np.testing.assert_allclose(self._residual(model, rows), expected, atol=1e-7)

    def test_jacobian_matches_finite_differences(self):
        """Match the joint row Jacobian with finite differences of the residuals under the rigid retraction."""
        builder = newton.ModelBuilder()
        _add_satisfied_chain(builder)
        self._assert_jacobian_matches_finite_differences(builder.finalize(device="cpu"))

    def test_jacobian_matches_finite_differences_off_the_anchors(self):
        """Match the Jacobian of a prismatic joint slid 0.3 m along a turned parent, whose locked rows turn with the
        parent away from the child's anchor, and of a D6 joint with two angular axes turned to (0.6, -0.4), whose
        Hooke row holds the angle between the parent's first axis and the child's second."""
        builder = newton.ModelBuilder()
        links = [_link(builder) for _ in range(3)]
        joints = [
            builder.add_joint_revolute(-1, links[0], axis=(0.0, 0.0, 1.0), child_xform=wp.transform((0.2, 0.0, 0.0))),
            builder.add_joint_prismatic(
                links[0], links[1], axis=(1.0, 0.0, 0.0), child_xform=wp.transform((0.0, 0.1, 0.0))
            ),
            builder.add_joint_d6(
                links[1],
                links[2],
                angular_axes=[Dof(axis=(1.0, 0.0, 0.0)), Dof(axis=(0.0, 1.0, 0.0))],
                parent_xform=wp.transform((0.0, 0.0, 0.3)),
                child_xform=wp.transform((0.1, 0.0, -0.2)),
            ),
        ]
        builder.add_articulation(joints)
        model = builder.finalize(device="cpu")
        model.joint_q.assign(np.array([0.7, 0.3, 0.6, -0.4], dtype=np.float32))
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        model.body_q.assign(state.body_q)
        kinds = _rows(model).row_kind.numpy()
        self.assertEqual(int(np.count_nonzero(kinds == int(J.ROW_HOOKE))), 1)
        self._assert_jacobian_matches_finite_differences(model)

    def _assert_jacobian_matches_finite_differences(self, model):
        rows, slot = _rows(model), wp.array(np.arange(3), dtype=wp.int32, device="cpu")
        R = rows.row_joint.shape[0]
        x0, q0 = _com_poses(model)
        x0_d, q0_d = wp.array(x0, dtype=wp.vec3d, device="cpu"), wp.array(q0, dtype=wp.quatd, device="cpu")
        row_args = _row_args(rows, slot)
        jac = _jacobian(model, rows, slot, x0, q0, "cpu")[2]

        def residual(u):
            x_live, q_live = wp.empty(3, dtype=wp.vec3d, device="cpu"), wp.empty(3, dtype=wp.quatd, device="cpu")
            scratch = [
                wp.empty(3, dtype=wp.transform, device="cpu"),
                wp.empty(3, dtype=wp.spatial_vector, device="cpu"),
            ]
            wp.launch(
                retract_rigid_kernel,
                dim=3,
                inputs=[
                    slot,
                    wp.array(u, dtype=wp.float64, device="cpu"),
                    q0_d,
                    x0_d,
                    model.body_q,
                    model.body_com,
                    wp.float64(1.0),
                ],
                outputs=[*scratch, x_live, q_live, wp.empty(3, dtype=wp.vec3d, device="cpu")],
                device="cpu",
            )
            res = wp.empty(R, dtype=wp.float64, device="cpu")
            wp.launch(
                J.joint_residual_kernel,
                dim=R,
                inputs=[
                    *row_args,
                    x_live,
                    q_live,
                    model.body_q,
                    model.body_com,
                    wp.zeros(R, dtype=wp.float64, device="cpu"),
                    wp.zeros(R, dtype=wp.quatd, device="cpu"),
                    wp.zeros(R, dtype=wp.float64, device="cpu"),
                ],
                outputs=[res],
                device="cpu",
            )
            return res.numpy()

        u0 = np.zeros(18)
        for k in range(3):  # chi = R_pinᵀ x with the pinned frame at the current pose
            u0[6 * k : 6 * k + 3] = wp.quat_rotate_inv(wp.quatd(*q0[k]), wp.vec3d(*x0[k]))
        np.testing.assert_allclose(residual(u0), 0.0, atol=1e-6)
        eps = 1e-6
        fd = np.stack([(residual(u0 + eps * e) - residual(u0 - eps * e)) / (2 * eps) for e in np.eye(18)], axis=1)
        np.testing.assert_allclose(jac, fd, atol=1e-7)


class TestBilateralSystem(unittest.TestCase):
    def test_matches_dense_algebra(self):
        """Match the per-block ``W = J A⁻¹ Jᵀ``, its Newton step and ``Jᵀ f`` with dense algebra."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        _add_satisfied_chain(builder)
        pend = _link(builder, xform=wp.transform((2.0, 0.0, 0.0)))
        x_pj, x_cj = wp.transform((2.0, 1.0, 0.0)), wp.transform((0.0, 1.0, 0.0))
        builder.add_articulation([builder.add_joint_revolute(-1, pend, parent_xform=x_pj, child_xform=x_cj)])
        model = builder.finalize(device=device)
        slot = np.arange(4)
        rows = build_joint_rows(model, slot, device)
        np.testing.assert_array_equal(rows.block_start, [0, 13, 18])
        A = build_rigid_system(model, slot, 0.01)
        row_slot, row_jac, jac = _jacobian(
            model, rows, wp.array(slot, dtype=wp.int32, device=device), *_com_poses(model), device
        )
        inv_mass = wp.array(rigid_block_inverses(A), dtype=wp.spatial_matrixd, device=device)
        system = BilateralSystem(rows, slot, inv_mass, device)
        system.factor(row_slot, row_jac)

        W = jac @ np.linalg.solve(A.toarray(), jac.T)
        np.testing.assert_allclose(W[:13, 13:], 0.0, atol=1e-14)  # independent blocks
        for k, (s, e) in enumerate(itertools.pairwise(rows.block_start)):
            np.testing.assert_allclose(system.W.numpy()[k, : e - s, : e - s], W[s:e, s:e], rtol=1e-10, atol=1e-14)

        rng = np.random.default_rng(0)
        residual, force = rng.standard_normal(18), wp.zeros(18, dtype=wp.float64, device=device)
        system.update(wp.array(residual, dtype=wp.float64, device=device), force)
        np.testing.assert_allclose(force.numpy(), -np.linalg.solve(W, residual), rtol=1e-8)

        b_const, b = rng.standard_normal(24), wp.empty(24, dtype=wp.float64, device=device)
        system.apply_force(row_jac, force, wp.array(b_const, dtype=wp.float64, device=device), b)
        np.testing.assert_allclose(b.numpy(), b_const + jac.T @ force.numpy(), rtol=1e-12)

        # Block solve on the second articulation's rows, placed at an offset.
        v, out = rng.standard_normal(8), wp.zeros(8, dtype=wp.float64, device=device)
        rows = wp.array(np.arange(13, 18), dtype=wp.int32, device=device)
        system.solve_rows(rows, 3, wp.array(v, dtype=wp.float64, device=device), out)
        np.testing.assert_allclose(out.numpy()[3:], np.linalg.solve(W[13:18, 13:18], v[3:]), rtol=1e-8)


class TestSolverFBAJoints(unittest.TestCase):
    """SolverFBA enforces joints in the rigid step."""

    def _run(self, model, h, steps):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        solver = SolverFBA(model, iterations=5)
        s0, s1 = model.state(), model.state()
        for _ in range(steps):
            solver.step(s0, s1, None, None, h)
            s0, s1 = s1, s0
        return s0.body_q.numpy()[0].astype(np.float64)

    def test_prismatic_matches_implicit_euler(self):
        """Move a body on a prismatic joint as implicit Euler does under the axis component of gravity alone,
        ``x_n = h² a n (n + 1) / 2``."""
        builder = newton.ModelBuilder()
        link = _link(builder)
        builder.add_articulation([builder.add_joint_prismatic(-1, link, axis=(1.0, 0.0, 0.0))])
        model = builder.finalize()
        model.gravity.assign([wp.vec3(1.0, -9.81, 0.5)])
        h, n = 0.01, 20
        np.testing.assert_allclose(self._run(model, h, n), [h * h * n * (n + 1) / 2, 0, 0, 0, 0, 0, 1], atol=1e-6)

    def test_two_axis_d6_keeps_its_pose(self):
        """Hold a body on a D6 joint with two angular axes at rest where the joint's coordinates (0.8, 0.8) put it,
        as Newton composes the rotation ``q(a0, q0) q(a1, q1)``, without gravity."""
        builder = newton.ModelBuilder()
        link = _link(builder)
        axes = [Dof(axis=(1.0, 0.0, 0.0)), Dof(axis=(0.0, 1.0, 0.0))]
        joint = builder.add_joint_d6(-1, link, angular_axes=axes, child_xform=wp.transform((0.0, 0.0, 0.5)))
        builder.add_articulation([joint])
        model = builder.finalize()
        model.gravity.assign([wp.vec3(0.0, 0.0, 0.0)])
        model.joint_q.assign(np.array([0.8, 0.8], dtype=np.float32))
        start = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, start)
        model.body_q.assign(start.body_q)
        tf = self._run(model, 0.01, 10)
        np.testing.assert_allclose(tf, start.body_q.numpy()[0], atol=1e-5)

    def test_prismatic_child_conserves_angular_momentum(self):
        """Keep the angular momentum of a free body and a child on a prismatic joint spinning without gravity, while
        the child slides out along the axis."""
        inertia_p, inertia_c = 0.1, 0.01
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia_p), lock_inertia=True)
        child = builder.add_link(
            mass=1.0, inertia=wp.mat33(np.eye(3) * inertia_c), lock_inertia=True, xform=wp.transform((0.5, 0.0, 0.0))
        )
        builder.add_articulation([builder.add_joint_free(parent), builder.add_joint_prismatic(parent, child)])
        model = builder.finalize()
        omega = np.array([0.0, 2.0, 0.0])
        x = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        com = x.mean(axis=0)
        model.body_qd.assign(np.array([[*np.cross(omega, p - com), *omega] for p in x], dtype=np.float32))

        def momentum(q, qd):
            q, qd = q.astype(np.float64), qd.astype(np.float64)
            center = q[:, :3].mean(axis=0)
            return sum(
                np.cross(q[k, :3] - center, qd[k, :3]) + i * qd[k, 3:] for k, i in enumerate((inertia_p, inertia_c))
            )

        from newton.solvers import SolverFBA  # noqa: PLC0415

        solver = SolverFBA(model, iterations=10)
        s0, s1 = model.state(), model.state()
        start = momentum(s0.body_q.numpy(), s0.body_qd.numpy())
        for _ in range(100):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        q = s0.body_q.numpy().astype(np.float64)
        offset = wp.quat_rotate_inv(wp.quatd(*q[0, 3:]), wp.vec3d(*(q[1, :3] - q[0, :3])))
        self.assertGreater(offset[0], 0.6)  # the child slid out
        np.testing.assert_allclose(momentum(q, s0.body_qd.numpy()), start, atol=2e-3 * np.linalg.norm(start))

    def test_rejects_articulations_beyond_one_tile(self):
        """Refuse a chain of thirteen revolute joints, whose 65 rows exceed what one tile Cholesky factors."""
        builder = newton.ModelBuilder()
        links = [_link(builder, xform=wp.transform((float(k), 0.0, 0.0))) for k in range(13)]
        x_cj = wp.transform((-0.5, 0.0, 0.0))
        parents = [-1, *links[:-1]]
        builder.add_articulation(
            [
                builder.add_joint_revolute(p, c, axis=(0.0, 0.0, 1.0), child_xform=x_cj)
                for p, c in zip(parents, links, strict=True)
            ]
        )
        with self.assertRaises(NotImplementedError):
            self._run(builder.finalize(), 0.01, 1)

    def test_pendulum_converges_to_ode(self):
        """Converge a compound pendulum released horizontally to the ODE solution at first order in h."""
        from scipy.integrate import solve_ivp

        T, I_pivot = 0.25, 2.0  # unit mass and inertia, COM 1 m from the pivot
        ode = solve_ivp(lambda t, y: [y[1], -9.81 * np.sin(y[0]) / I_pivot], (0.0, T), [np.pi / 2, 0.0], rtol=1e-12)
        errors = []
        for h in (0.005, 0.0025):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
            link = _link(builder, xform=wp.transform((1.0, 0.0, 0.0)))
            builder.add_articulation(
                [builder.add_joint_revolute(-1, link, axis=(0.0, 0.0, 1.0), child_xform=wp.transform((-1.0, 0.0, 0.0)))]
            )
            tf = self._run(builder.finalize(), h, round(T / h))
            anchor = tf[:3] + wp.quat_rotate(wp.quatd(*tf[3:]), wp.vec3d(-1.0, 0.0, 0.0))
            np.testing.assert_allclose(anchor, 0.0, atol=1e-6)
            errors.append(abs(np.arctan2(tf[0], -tf[1]) - ode.y[0, -1]))
        self.assertLess(errors[0], 2e-2)
        self.assertAlmostEqual(errors[0] / errors[1], 2.0, delta=0.2)


class TestSolverFBAJointLimits(unittest.TestCase):
    """SolverFBA enforces the limits and the passive damping of revolute and prismatic joints."""

    def _run(self, model, h, steps, **solver_args):
        """Each step's body poses and velocities, ``[step, body, 7 or 6]``, and its count of limit rows."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        solver = SolverFBA(model, iterations=5, **solver_args)
        s0, s1 = model.state(), model.state()
        q, qd, limit_rows = [], [], []
        for _ in range(steps):
            solver.step(s0, s1, None, None, h)
            s0, s1 = s1, s0
            q.append(s0.body_q.numpy().astype(np.float64))
            qd.append(s0.body_qd.numpy().astype(np.float64))
            limit_rows.append(solver._rc_limits)
        return np.array(q), np.array(qd), limit_rows

    @staticmethod
    def _pendulum(builder, lower, upper, damping=0.0):
        """A unit body on a revolute joint about z at the origin, its COM 1 m out along +x."""
        link = _link(builder, xform=wp.transform((1.0, 0.0, 0.0)))
        joint = builder.add_joint_revolute(
            -1,
            link,
            axis=(0.0, 0.0, 1.0),
            child_xform=wp.transform((-1.0, 0.0, 0.0)),
            limit_lower=lower,
            limit_upper=upper,
            damping=damping,
        )
        builder.add_articulation([joint])

    def test_revolute_limit_stops_the_pendulum(self):
        """Stop a pendulum released horizontally at its lower limit and hold it there, with its two limit rows
        present only near the limits."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        self._pendulum(builder, -0.5, 0.5)
        q, qd, limit_rows = self._run(builder.finalize(), 0.005, 200)
        angle = np.arctan2(q[:, 0, 1], q[:, 0, 0])
        self.assertGreater(angle.min(), -0.5 - 1e-3)
        self.assertAlmostEqual(angle[-1], -0.5, delta=1e-4)
        self.assertLess(abs(qd[-1, 0, 5]), 1e-3)
        # Released at 0.5 rad from either limit, it starts without limit rows; it ends with the lower one.
        self.assertEqual((limit_rows[0], limit_rows[-1]), (0, 1))

    def test_friction_holds_the_pendulum_at_its_limit(self):
        """Hold a pendulum that gravity pushes into its lower limit there, with joint friction on the axis of the
        limit's row, below and above gravity's torque; and with friction above gravity's torque alone, hold it where
        it is released."""
        for friction, lower, start in ((2.0, -0.5, -0.5), (20.0, -0.5, -0.5), (20.0, -1.0e10, -0.3)):
            with self.subTest(friction=friction, lower=lower):
                builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
                self._pendulum(builder, lower, 0.5 if lower > -1.0e10 else 1.0e10)
                builder.joint_friction[0] = friction
                builder.joint_q[0] = start
                model = builder.finalize()
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                model.body_q.assign(state.body_q)
                q, qd, _ = self._run(model, 0.005, 200)
                angle = np.arctan2(q[:, 0, 1], q[:, 0, 0])
                self.assertTrue(np.isfinite(q).all())
                np.testing.assert_allclose(angle, start, atol=1e-4)
                self.assertLess(np.abs(qd[:, 0, 5]).max(), 1e-3)

    def test_prismatic_limit_holds_the_slider(self):
        """Hold a slider that gravity pulls along its axis at its lower limit."""
        builder = newton.ModelBuilder()
        link = _link(builder)
        joint = builder.add_joint_prismatic(-1, link, axis=(1.0, 0.0, 0.0), limit_lower=-0.2, limit_upper=0.5)
        builder.add_articulation([joint])
        model = builder.finalize()
        model.gravity.assign([wp.vec3(-9.81, 0.0, 0.0)])
        q, qd, _ = self._run(model, 0.01, 100)
        self.assertGreater(q[:, 0, 0].min(), -0.2 - 1e-4)
        np.testing.assert_allclose(q[-1, 0, :3], [-0.2, 0.0, 0.0], atol=1e-5)
        self.assertLess(abs(qd[-1, 0, 0]), 1e-4)

    def test_damping_matches_implicit_euler(self):
        """Slow a free spin and a free slide of unit inertia with damping ``c`` by ``1 / (1 + c dt)`` per step,
        alone in the joint system and, near a limit, with the limit's row in the contact solve."""
        dt, n = 0.01, 40
        for kind, c, lower in (("revolute", 0.5, -1.0e10), ("revolute", 5.0, -1.0e10), ("prismatic", 2.0, -1.0e10),
                               ("revolute", 2.0, -0.01), ("prismatic", 2.0, -0.001)):  # fmt: skip
            with self.subTest(kind=kind, damping=c, lower=lower):
                builder = newton.ModelBuilder()
                link = _link(builder)
                add = builder.add_joint_revolute if kind == "revolute" else builder.add_joint_prismatic
                upper = 2.0 if lower > -1.0 else 1.0e10
                joint = add(-1, link, axis=(0.0, 0.0, 1.0), damping=c, limit_lower=lower, limit_upper=upper)
                builder.add_articulation([joint])
                model = builder.finalize()
                model.gravity.assign([wp.vec3(0.0, 0.0, 0.0)])
                lane = 5 if kind == "revolute" else 2
                qd0 = model.body_qd.numpy()
                qd0[0, lane] = 1.0
                model.body_qd.assign(qd0)
                _q, qd, limit_rows = self._run(model, dt, n)
                expected = (1.0 / (1.0 + c * dt)) ** np.arange(1, n + 1)
                np.testing.assert_allclose(qd[:, 0, lane], expected, rtol=1e-5)
                # Starting next to its lower limit, the axis takes its row for its first steps, which moves the joint
                # rows into the contact solve, the damper among them; moving away, it drops the row.
                near = lower > -1.0
                self.assertEqual((limit_rows[0], limit_rows[-1]), (int(near), 0))

    def test_ball_and_d6_damping_match_implicit_euler(self):
        """Damp each free axis of a ball and a D6 joint by its own coefficient: a free body of unit inertia on a ball
        joint at its COM slows its spin by ``1 / (1 + c dt)`` per step about every axis, and one on a D6 joint free
        along x and y and about z, damped along x and about z, slows its slide along x and its spin alone."""
        dt, n = 0.01, 40
        cases = (
            ("ball", [0.8, 0.8, 0.8], [0.0, 0.0, 0.0, 1.0, 0.5, -0.3]),
            ("d6", [3.0, 0.0, 0.0, 0.0, 0.0, 4.0], [1.0, 0.5, 0.0, 0.0, 0.0, 0.4]),
        )
        for kind, damping, qd0 in cases:
            with self.subTest(kind=kind):
                builder = newton.ModelBuilder()
                link = _link(builder)
                if kind == "ball":
                    joint = builder.add_joint_ball(-1, link, damping=damping[0])
                    lanes, coefficient = [3, 4, 5], damping  # body_qd lanes about x, y and z
                else:
                    joint = builder.add_joint_d6(
                        -1,
                        link,
                        linear_axes=[Dof(axis=(1.0, 0.0, 0.0), damping=damping[0]), Dof(axis=(0.0, 1.0, 0.0))],
                        angular_axes=[Dof(axis=(0.0, 0.0, 1.0), damping=damping[5])],
                    )
                    lanes, coefficient = [0, 1, 5], [damping[0], 0.0, damping[5]]  # along x and y, about z
                builder.add_articulation([joint])
                model = builder.finalize()
                model.gravity.assign([wp.vec3(0.0, 0.0, 0.0)])
                model.body_qd.assign(np.array([qd0], dtype=np.float32))
                _q, qd, _ = self._run(model, dt, n)
                for lane, c in zip(lanes, coefficient, strict=True):
                    expected = qd0[lane] * (1.0 / (1.0 + c * dt)) ** np.arange(1, n + 1)
                    np.testing.assert_allclose(qd[:, 0, lane], expected, rtol=1e-5, atol=1e-7, err_msg=f"lane {lane}")

    def test_second_axis_of_a_two_axis_d6_damps_about_the_child_axis(self):
        """Damp the spin of a body of unit inertia on a D6 joint with two angular axes, turned by 0.8 rad about the
        first, about the second axis as the child carries it: damping on the second axis slows it by
        ``1 / (1 + c dt)`` per step, and damping on the first leaves it."""
        dt, n, c = 0.01, 30, 5.0
        axis = np.array(wp.quat_rotate(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.8), wp.vec3(0.0, 1.0, 0.0)))
        for damped in (0, 1):
            with self.subTest(damped=damped):
                builder = newton.ModelBuilder()
                link = _link(builder)
                damping = [c if k == damped else 0.0 for k in (0, 1)]
                axes = [Dof(axis=(1.0, 0.0, 0.0), damping=damping[0]), Dof(axis=(0.0, 1.0, 0.0), damping=damping[1])]
                builder.add_articulation([builder.add_joint_d6(-1, link, angular_axes=axes)])
                model = builder.finalize()
                model.gravity.assign([wp.vec3(0.0, 0.0, 0.0)])
                model.joint_q.assign(np.array([0.8, 0.0], dtype=np.float32))
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                model.body_q.assign(state.body_q)
                model.body_qd.assign(np.array([[0.0, 0.0, 0.0, *(2.0 * axis)]], dtype=np.float32))
                _q, qd, _ = self._run(model, dt, n)
                rate = 2.0 * (1.0 / (1.0 + c * dt)) ** np.arange(1, n + 1) if damped == 1 else np.full(n, 2.0)
                np.testing.assert_allclose(qd[:, 0, 3:], rate[:, None] * axis, rtol=1e-4, atol=1e-5)

    def test_limits_in_several_worlds(self):
        """Stop the pendulums of two worlds at their own lower limits."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        for lower in (-0.5, -0.3):
            world = newton.ModelBuilder(up_axis=newton.Axis.Y)
            self._pendulum(world, lower, 0.5)
            builder.add_world(world)
        q, _qd, _ = self._run(builder.finalize(), 0.005, 200)
        angle = np.arctan2(q[-1, :, 1], q[-1, :, 0])
        np.testing.assert_allclose(angle, [-0.5, -0.3], atol=1e-4)

    def test_default_joints_take_no_limit_or_damper_rows(self):
        """Give joints at the default infinite limits and zero damping no limit candidates and no damper rows."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        _add_satisfied_chain(builder)
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        solver.step(model.state(), model.state(), None, None, 0.01)
        self.assertIsNone(solver._limits)
        self.assertFalse(solver._joint_dampers)
        self.assertEqual(solver._joint_system.compliance.shape[0], 0)

    def test_refuses_revolute_limits_known_up_to_turns(self):
        """Refuse a revolute axis with one finite limit."""
        for lower, upper in ((-0.5, 1.0e10), (-1.0e10, 0.5)):
            with self.subTest(lower=lower, upper=upper):
                builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
                self._pendulum(builder, lower, upper)
                with self.assertRaises(NotImplementedError):
                    self._run(builder.finalize(), 0.01, 1)


class TestSolverFBAJointState(unittest.TestCase):
    """SolverFBA's joints with kinematic parents, model edits and resets."""

    def test_damper_on_a_kinematic_parent_damps_the_relative_motion(self):
        """Spin a disc on a damped revolute joint up with the kinematic turntable it sits on, which turns at its
        prescribed velocity: implicit Euler of ``I dw/dt = c (w_table - w)`` from rest."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        rate, c, dt, n = 2.0, 1.0, 0.01, 200
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        table = builder.add_link(mass=10.0, inertia=wp.mat33(np.eye(3) * 10.0), lock_inertia=True, is_kinematic=True)
        disc = _link(builder, xform=wp.transform((0.0, 0.0, 0.5)))
        free = builder.add_joint_free(table)
        hinge = builder.add_joint_revolute(
            table, disc, axis=(0.0, 0.0, 1.0), parent_xform=wp.transform((0.0, 0.0, 0.5)), damping=c
        )
        builder.add_articulation([free, hinge])
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s0, s1 = model.state(), model.state()
        for k in range(n):
            q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
            q[0, 3:] = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rate * k * dt)
            qd[0] = (0.0, 0.0, 0.0, 0.0, 0.0, rate)
            s0.body_q.assign(q)
            s0.body_qd.assign(qd)
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
        expected = rate * (1.0 - (1.0 / (1.0 + c * dt)) ** n)
        self.assertAlmostEqual(float(s0.body_qd.numpy()[1, 5]), expected, delta=1e-3 * rate)

    def test_reset_clears_the_limit_warm_start(self):
        """Step a pendulum resting on its lower limit with the contact warm start, reset it, and step it as a fresh
        solver does."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        angle = -0.5
        link = _link(
            builder,
            xform=wp.transform((np.cos(angle), np.sin(angle), 0.0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), angle)),
        )
        joint = builder.add_joint_revolute(
            -1,
            link,
            axis=(0.0, 0.0, 1.0),
            child_xform=wp.transform((-1.0, 0.0, 0.0)),
            limit_lower=-0.5,
            limit_upper=0.5,
        )
        builder.add_articulation([joint])
        model = builder.finalize()

        def first_step(solver):
            s0, s1 = model.state(), model.state()
            solver.step(s0, s1, None, None, 0.01)
            return s1.body_qd.numpy()

        fresh = first_step(SolverFBA(model, iterations=1, contact_warmstart=True))
        solver = SolverFBA(model, iterations=1, contact_warmstart=True)
        s0, s1 = model.state(), model.state()
        for _ in range(100):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        solver.reset(s0)
        np.testing.assert_array_equal(first_step(solver), fresh)

    def test_joint_dof_flags_take_damping_and_limits(self):
        """Apply damping and limits edited after the first step once notify_model_changed names them with the joint
        DOF flags; without a notification the solver keeps the old ones."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        for flag in (None, newton.ModelFlags.JOINT_DOF_PROPERTIES, newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES):
            with self.subTest(flag=flag):
                builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
                link = _link(builder, xform=wp.transform((1.0, 0.0, 0.0)))
                joint = builder.add_joint_revolute(
                    -1, link, axis=(0.0, 0.0, 1.0), child_xform=wp.transform((-1.0, 0.0, 0.0))
                )
                builder.add_articulation([joint])
                model = builder.finalize()
                solver = SolverFBA(model, iterations=5)
                s0, s1 = model.state(), model.state()
                solver.step(s0, s1, None, None, 0.005)
                s0, s1 = s1, s0
                model.joint_limit_lower.assign([-0.3])
                model.joint_limit_upper.assign([0.3])
                if flag is not None:
                    solver.notify_model_changed(flag)
                for _ in range(200):
                    solver.step(s0, s1, None, None, 0.005)
                    s0, s1 = s1, s0
                q = s0.body_q.numpy()[0]
                angle = float(np.arctan2(q[1], q[0]))
                if flag is None:
                    self.assertLess(angle, -0.5)
                else:
                    self.assertAlmostEqual(angle, -0.3, delta=1e-3)

    def test_body_properties_keep_the_warm_start(self):
        """Step a pile of two boxes alike whether or not each step notifies BODY_PROPERTIES, which leaves the body
        flags alone and so rebuilds nothing."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        finals = []
        for notify in (False, True):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            for z in (0.05, 0.151):
                body = builder.add_body(xform=wp.transform((0.0, 0.0, z)))
                builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            model = builder.finalize()
            solver = SolverFBA(model, iterations=5, contact_warmstart=True)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            s0, s1 = model.state(), model.state()
            for _ in range(30):
                pipeline.collide(s0, contacts)
                if notify:
                    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
                solver.step(s0, s1, None, contacts, 0.01)
                s0, s1 = s1, s0
            finals.append(s0.body_q.numpy())
        np.testing.assert_array_equal(finals[1], finals[0])

    def test_failed_setup_raises_again(self):
        """Raise for a revolute joint with a single finite limit again at the next step rather than stepping without
        the bodies."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        link = _link(builder, xform=wp.transform((1.0, 0.0, 0.0)))
        joint = builder.add_joint_revolute(
            -1, link, axis=(0.0, 0.0, 1.0), child_xform=wp.transform((-1.0, 0.0, 0.0)), limit_lower=-0.5
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s0, s1 = model.state(), model.state()
        for _ in range(2):
            with self.assertRaises(NotImplementedError):
                solver.step(s0, s1, None, None, 0.01)

    def test_full_turn_limits_are_left_out(self):
        """Warn about revolute limits whose range spans a full turn, as many arm descriptions give, and swing the
        pendulum through them as without limits."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        finals = []
        for limits in ((-2.0 * np.pi, 2.0 * np.pi), None):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
            link = _link(builder, xform=wp.transform((1.0, 0.0, 0.0)))
            extra = {} if limits is None else {"limit_lower": limits[0], "limit_upper": limits[1]}
            joint = builder.add_joint_revolute(
                -1, link, axis=(0.0, 0.0, 1.0), child_xform=wp.transform((-1.0, 0.0, 0.0)), **extra
            )
            builder.add_articulation([joint])
            model = builder.finalize()
            solver = SolverFBA(model, iterations=5)
            s0, s1 = model.state(), model.state()
            if limits is None:
                solver.step(s0, s1, None, None, 0.01)
            else:
                with self.assertWarns(newton.exceptions.NewtonWarning):
                    solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            for _ in range(50):
                solver.step(s0, s1, None, None, 0.01)
                s0, s1 = s1, s0
            finals.append(s0.body_q.numpy())
        np.testing.assert_array_equal(finals[0], finals[1])


def _implicit_euler(n, dt, inertia, ke=0.0, kd=0.0, c=0.0, armature=0.0, q_target=0.0, qd_target=0.0, force=0.0):
    """Coordinates and rates of an axis of ``inertia`` from rest at 0 over ``n`` backward Euler steps of its drive
    ``ke (q* - q) + kd (q_dot* - q_dot)``, damping ``-c q_dot``, armature and constant generalized ``force``."""
    q = v = 0.0
    m = inertia + armature
    out = []
    for _ in range(n):
        v = (m * v / dt + ke * (q_target - q) + kd * qd_target + force) / (m / dt + ke * dt + kd + c)
        q += dt * v
        out.append((q, v))
    return np.array(out)


def _box_euler(n, dt, inertia, ke=0.0, kd=0.0, c=0.0, armature=0.0, q_target=0.0, qd_target=0.0, force=0.0, **box):
    """Coordinates and rates of an axis as :func:`_implicit_euler`, its drive's force bounded by ``box["effort"]`` and
    a dry friction of ``box["friction"]`` on it: each step's end rate ``w`` holds ``h(w) + f = 0`` for the axis's
    other forces ``h`` less its inertial force and a friction force ``f`` within ``[-F, F]`` that opposes ``w``."""
    effort, friction = box.get("effort", np.inf), box.get("friction", 0.0)
    q = v = 0.0
    m = inertia + armature
    out = []
    for _ in range(n):

        def h(w, q=q, v=v):
            drive = ke * (q_target - q - dt * w) + kd * (qd_target - w)
            return np.clip(drive, -effort, effort) - c * w + force - m * (w - v) / dt

        w = 0.0
        if abs(h(0.0)) > friction:
            # h decreases with w: bisect h(w) = s F on the side of the motion.
            s = np.sign(h(0.0))
            lo, hi = (0.0, 1.0e3) if s > 0 else (-1.0e3, 0.0)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if (h(mid) - s * friction > 0.0) == (s > 0):
                    lo, hi = (mid, hi) if s > 0 else (lo, mid)
                else:
                    lo, hi = (lo, mid) if s > 0 else (mid, hi)
            w = 0.5 * (lo + hi)
        v = w
        q += dt * v
        out.append((q, v))
    return np.array(out)


class TestSolverFBAJointDrives(unittest.TestCase):
    """SolverFBA's joint drives, armature and generalized joint forces (``Control.joint_f``)."""

    @staticmethod
    def _axis_model(kind, lower=-1.0e10, upper=1.0e10, **dof):
        """A unit body on a revolute joint about z through its COM or a prismatic joint along z, without gravity, its
        DOF's ``target_ke``, ``target_kd``, ``damping``, ``armature``, ``friction`` and ``effort_limit`` set from
        ``dof``."""
        builder = newton.ModelBuilder()
        link = _link(builder)
        add = builder.add_joint_revolute if kind == "revolute" else builder.add_joint_prismatic
        builder.add_articulation([add(-1, link, axis=(0.0, 0.0, 1.0), limit_lower=lower, limit_upper=upper)])
        model = builder.finalize()
        model.gravity.assign([wp.vec3(0.0, 0.0, 0.0)])
        for name in ("target_ke", "target_kd", "damping", "armature", "friction"):
            getattr(model, f"joint_{name}").assign([dof.get(name, 0.0)])
        model.joint_effort_limit.assign([dof.get("effort_limit", 1.0e6)])
        return model

    @staticmethod
    def _run(model, dt, n, targets=None, joint_f=None):
        """Each step's body poses and velocities, ``[step, 7 or 6]`` of body 0, with ``targets`` as
        ``{step: (q*, q_dot*)}`` set on the control from that step on, and a constant ``joint_f``."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        solver = SolverFBA(model, iterations=5)
        control = model.control()
        if joint_f is not None:
            control.joint_f.assign(np.asarray(joint_f, dtype=np.float32))
        s0, s1 = model.state(), model.state()
        q, qd = [], []
        for k in range(n):
            if targets and k in targets:
                control.joint_target_q.assign(np.atleast_1d(np.float32(targets[k][0])))
                control.joint_target_qd.assign(np.atleast_1d(np.float32(targets[k][1])))
            solver.step(s0, s1, control, None, dt)
            s0, s1 = s1, s0
            q.append(s0.body_q.numpy()[0].astype(np.float64))
            qd.append(s0.body_qd.numpy()[0].astype(np.float64))
        return np.array(q), np.array(qd)

    @staticmethod
    def _coordinate(kind, q, qd):
        """The axis coordinate and rate of :meth:`_axis_model` from its body's poses and velocities."""
        if kind == "revolute":
            return 2.0 * np.arctan2(q[:, 5], q[:, 6]), qd[:, 5]
        return q[:, 2], qd[:, 2]

    def test_drives_match_implicit_euler(self):
        """Drive an axis of unit inertia by position and velocity targets, with damping and armature, step for step
        as the backward Euler step of its forces does, alone in the joint system and, next to a limit, with the
        joint rows in the contact solve."""
        dt, n = 0.01, 100
        cases = (
            ("revolute", {}, {"target_ke": 50.0}, (0.5, 0.0)),
            ("revolute", {}, {"target_ke": 50.0, "target_kd": 5.0}, (0.5, 0.0)),
            ("revolute", {}, {"target_kd": 4.0}, (0.0, 2.0)),
            ("revolute", {}, {"target_ke": 50.0, "target_kd": 5.0, "damping": 1.0, "armature": 0.5}, (-0.4, 0.0)),
            ("revolute", {"lower": -0.01, "upper": 2.0}, {"target_ke": 50.0, "target_kd": 5.0}, (0.5, 0.0)),
            ("prismatic", {}, {"target_ke": 100.0, "target_kd": 10.0, "armature": 0.3}, (0.2, 0.0)),
            ("prismatic", {}, {"target_kd": 3.0}, (0.0, -0.5)),
        )
        for kind, limits, dof, target in cases:
            with self.subTest(kind=kind, limits=limits, **dof):
                model = self._axis_model(kind, **limits, **dof)
                q, qd = self._run(model, dt, n, targets={0: target})
                coordinate, rate = self._coordinate(kind, q, qd)
                gains = {"ke": "target_ke", "kd": "target_kd", "c": "damping", "armature": "armature"}
                expected = _implicit_euler(
                    n,
                    dt,
                    1.0,
                    **{k: dof.get(v, 0.0) for k, v in gains.items()},
                    q_target=target[0],
                    qd_target=target[1],
                )
                np.testing.assert_allclose(coordinate, expected[:, 0], rtol=1e-4, atol=1e-6)
                np.testing.assert_allclose(rate, expected[:, 1], rtol=1e-4, atol=1e-6)

    def test_effort_limits_bound_the_drives(self):
        """Bound the drive force of an axis of unit inertia by its effort limit, its damping and armature unbounded,
        step for step as the backward Euler step with the drive's force clamped: a position drive that saturates
        on its way to a far target and on its way back from the overshoot, a velocity drive that saturates until
        it nears its target rate, alone in the joint system and, next to a limit, with the joint rows in the
        contact solve; the default effort limit leaves the drive unbounded."""
        dt, n = 0.01, 300
        cases = (
            ("revolute", {}, {"target_ke": 50.0, "target_kd": 5.0, "effort_limit": 2.0}, (1.0, 0.0)),
            ("revolute", {}, {"target_ke": 50.0, "target_kd": 5.0, "damping": 1.0, "armature": 0.5,
                              "effort_limit": 2.0}, (1.0, 0.0)),
            ("prismatic", {}, {"target_kd": 10.0, "effort_limit": 1.0}, (0.0, 2.0)),
            ("revolute", {"lower": -0.01, "upper": 2.0}, {"target_ke": 50.0, "target_kd": 5.0, "effort_limit": 2.0},
             (1.0, 0.0)),
            ("revolute", {}, {"target_ke": 50.0, "target_kd": 5.0, "effort_limit": 1.0e6}, (1.0, 0.0)),
        )  # fmt: skip
        for kind, limits, dof, target in cases:
            with self.subTest(kind=kind, limits=limits, **dof):
                model = self._axis_model(kind, **limits, **dof)
                q, qd = self._run(model, dt, n, targets={0: target})
                coordinate, rate = self._coordinate(kind, q, qd)
                gains = {"ke": "target_ke", "kd": "target_kd", "c": "damping", "armature": "armature"}
                expected = _box_euler(
                    n,
                    dt,
                    1.0,
                    **{k: dof.get(v, 0.0) for k, v in gains.items()},
                    q_target=target[0],
                    qd_target=target[1],
                    effort=dof["effort_limit"],
                )
                np.testing.assert_allclose(coordinate, expected[:, 0], rtol=1e-4, atol=1e-6)
                np.testing.assert_allclose(rate, expected[:, 1], rtol=1e-4, atol=1e-6)

    def test_joint_friction_holds_and_slides(self):
        """Hold an axis of unit inertia still against a generalized force below its joint friction and slide it
        under one above, then hold a driven axis short of its target where the drive's force falls below the
        friction, each step for step as the backward Euler step with dry friction, alone in the joint system and,
        next to a limit, with the joint rows in the contact solve."""
        dt, n = 0.01, 200
        cases = (
            ("revolute", {}, {"friction": 1.0}, None, 0.5),
            ("revolute", {}, {"friction": 0.2}, None, 0.5),
            ("prismatic", {}, {"friction": 0.3, "damping": 2.0}, None, -1.0),
            ("revolute", {}, {"friction": 1.0, "target_ke": 20.0, "target_kd": 4.0}, (0.6, 0.0), 0.0),
            ("revolute", {"lower": -0.01, "upper": 2.0}, {"friction": 0.2}, None, 0.5),
            ("revolute", {"lower": -0.01, "upper": 2.0}, {"friction": 1.0, "target_ke": 20.0, "target_kd": 4.0},
             (0.6, 0.0), 0.0),
        )  # fmt: skip
        for kind, limits, dof, target, force in cases:
            with self.subTest(kind=kind, limits=limits, force=force, **dof):
                model = self._axis_model(kind, **limits, **dof)
                q, qd = self._run(model, dt, n, targets=None if target is None else {0: target}, joint_f=[force])
                coordinate, rate = self._coordinate(kind, q, qd)
                gains = {"ke": "target_ke", "kd": "target_kd", "c": "damping", "armature": "armature"}
                expected = _box_euler(
                    n,
                    dt,
                    1.0,
                    **{k: dof.get(v, 0.0) for k, v in gains.items()},
                    q_target=0.0 if target is None else target[0],
                    force=force,
                    friction=dof["friction"],
                )
                np.testing.assert_allclose(coordinate, expected[:, 0], rtol=1e-4, atol=1e-6)
                np.testing.assert_allclose(rate, expected[:, 1], rtol=1e-4, atol=1e-6)

    def test_joint_forces_accelerate_the_axes(self):
        """Accelerate an axis of unit inertia with armature 0.5 by a constant generalized force, as the backward
        Euler step does, and a body on a FREE joint by the joint's world wrench."""
        dt, n = 0.01, 50
        for kind in ("revolute", "prismatic"):
            with self.subTest(kind=kind):
                model = self._axis_model(kind, armature=0.5)
                q, qd = self._run(model, dt, n, joint_f=[2.0])
                coordinate, rate = self._coordinate(kind, q, qd)
                expected = _implicit_euler(n, dt, 1.0, armature=0.5, force=2.0)
                np.testing.assert_allclose(coordinate, expected[:, 0], rtol=1e-4, atol=1e-6)
                np.testing.assert_allclose(rate, expected[:, 1], rtol=1e-4, atol=1e-6)
        with self.subTest(kind="free"):
            builder = newton.ModelBuilder()
            link = _link(builder)
            builder.add_articulation([builder.add_joint_free(link)])
            model = builder.finalize()
            model.gravity.assign([wp.vec3(0.0, 0.0, 0.0)])
            wrench = [1.0, -2.0, 0.5, 0.3, 0.0, -0.6]
            _q, qd = self._run(model, dt, n, joint_f=wrench)
            np.testing.assert_allclose(qd[-1], n * dt * np.array(wrench), rtol=1e-5, atol=1e-9)

    def test_position_drive_holds_a_pendulum_against_gravity(self):
        """Settle a pendulum of 1 kg at 1 m whose drive holds it at 0 rad against gravity where the drive's torque
        balances gravity's, ``ke (0 - q) = m g l cos(q)``."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        TestSolverFBAJointLimits._pendulum(builder, -1.0e10, 1.0e10)
        model = builder.finalize()
        ke, kd = 2000.0, 60.0
        model.joint_target_ke.assign([ke])
        model.joint_target_kd.assign([kd])
        q, _qd = self._run(model, 0.005, 400)
        angle = np.arctan2(q[-1, 1], q[-1, 0])
        expected = 0.0
        for _ in range(20):
            expected = -9.81 * np.cos(expected) / ke
        self.assertAlmostEqual(angle, expected, delta=1e-5)

    def test_drives_follow_the_control_targets(self):
        """Move a driven axis to the control's position target and then to a new one set during the run."""
        model = self._axis_model("revolute", target_ke=200.0, target_kd=30.0)
        q, qd = self._run(model, 0.01, 300, targets={0: (0.3, 0.0), 150: (-0.2, 0.0)})
        coordinate, _rate = self._coordinate("revolute", q, qd)
        self.assertAlmostEqual(coordinate[149], 0.3, delta=1e-3)
        self.assertAlmostEqual(coordinate[-1], -0.2, delta=1e-3)

    def test_unapplied_dof_properties_warn(self):
        """Warn at the first step about velocity limits, which FBA does not apply, and not about the defaults, effort
        limits or joint friction."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        cases = (
            ({}, None),
            ({"effort_limit": 50.0}, None),
            ({"effort_limit": 50.0, "target_ke": 10.0}, None),
            ({"friction": 0.2}, None),
            ({"velocity_limit": 3.0}, "joint_velocity_limit"),
        )
        for dof, name in cases:
            with self.subTest(**dof):
                builder = newton.ModelBuilder()
                link = _link(builder)
                builder.add_articulation([builder.add_joint_revolute(-1, link, damping=1.0, **dof)])
                model = builder.finalize()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    SolverFBA(model).step(model.state(), model.state(), None, None, 0.01)
                messages = [str(w.message) for w in caught if "does not apply" in str(w.message)]
                if name is None:
                    self.assertEqual(messages, [])
                else:
                    self.assertEqual(len(messages), 1)
                    self.assertIn(name, messages[0])

    def test_position_drives_of_several_angular_axes_raise(self):
        """Refuse position drives on the angular axes of ball joints and of D6 joints with two angular axes, which
        have no single angle, at the first step; accept velocity drives on them."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        def step(model):
            SolverFBA(model).step(model.state(), model.state(), None, None, 0.01)

        for kind in ("ball", "d6"):
            for gain in ("target_ke", "target_kd"):
                with self.subTest(kind=kind, gain=gain):
                    builder = newton.ModelBuilder()
                    link = _link(builder)
                    if kind == "ball":
                        joint = builder.add_joint_ball(-1, link)
                    else:
                        axes = [Dof(axis=(1.0, 0.0, 0.0)), Dof(axis=(0.0, 1.0, 0.0))]
                        joint = builder.add_joint_d6(-1, link, angular_axes=axes)
                    builder.add_articulation([joint])
                    model = builder.finalize()
                    gains = getattr(model, f"joint_{gain}").numpy()
                    gains[0] = 10.0
                    getattr(model, f"joint_{gain}").assign(gains)
                    if gain == "target_ke":
                        with self.assertRaisesRegex(NotImplementedError, "one free angular axis"):
                            step(model)
                    else:
                        step(model)


class TestSolverFBAMimicJoints(unittest.TestCase):
    """SolverFBA holds mimic joints (``Model.joint_mimic_joint``) at ``c0 + c1`` times their leaders' coordinates."""

    @staticmethod
    def _hand(kind, c0, c1, free_base=False, grasp=False):
        """A base of 2 kg with two fingers of 0.1 kg on prismatic joints along y or revolute joints about x, the
        first driven to 0.03 m or 0.4 rad and the second its mimic; with ``grasp``, box fingers close on a box of
        50 g between them. Without gravity. The model and the two joints."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        base = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
        fingers = []
        for side in (1.0, -1.0):
            xform = wp.transform((0.0, 0.05 * side, 0.0))
            finger = builder.add_link(mass=0.1, inertia=wp.mat33(np.eye(3) * 1e-4), lock_inertia=True, xform=xform)
            if grasp:
                builder.add_shape_box(finger, hx=0.01, hy=0.01, hz=0.02)
            fingers.append(finger)
        joints = [builder.add_joint_free(base) if free_base else builder.add_joint_fixed(-1, base)]
        target = 0.03 if kind == "prismatic" else 0.4
        for k, side in enumerate((1.0, -1.0)):
            add = builder.add_joint_prismatic if kind == "prismatic" else builder.add_joint_revolute
            axis = (0.0, -side, 0.0) if kind == "prismatic" else (1.0, 0.0, 0.0)
            drive = {"target_ke": 2000.0, "target_kd": 50.0, "target_pos": target} if k == 0 else {}
            parent_xform = wp.transform((0.0, 0.05 * side, 0.0))
            joints.append(add(base, fingers[k], axis=axis, parent_xform=parent_xform, **drive))
        builder.add_articulation(joints)
        if grasp:
            box = builder.add_body(mass=0.05, inertia=wp.mat33(np.eye(3) * 1e-5), lock_inertia=True)
            builder.add_shape_box(box, hx=0.015, hy=0.02, hz=0.015)
        model = builder.finalize()
        leaders = model.joint_mimic_joint.numpy()
        leaders[joints[2]] = joints[1]
        model.joint_mimic_joint.assign(leaders)
        coeffs = model.joint_mimic_coeffs.numpy()
        coeffs[joints[2]] = (c0, c1)
        model.joint_mimic_coeffs.assign(coeffs)
        return model, joints[1], joints[2]

    @staticmethod
    def _run(model, leader, follower, steps, dt, contacts=False):
        """The two joints' coordinates and the bodies' total linear momentum after each step, and the solver."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        solver = SolverFBA(model, iterations=5)
        pipeline = newton.CollisionPipeline(model) if contacts else None
        found = pipeline.contacts() if contacts else None
        s0, s1, joint_state = model.state(), model.state(), model.state()
        start = model.joint_q_start.numpy()
        out, joined = [], False
        for _ in range(steps):
            if pipeline is not None:
                pipeline.collide(s0, found)
            solver.step(s0, s1, None, found, dt)
            s0, s1 = s1, s0
            joined |= solver._rc_mimic is not None
            newton.eval_ik(model, s0, joint_state.joint_q, joint_state.joint_qd)
            q = joint_state.joint_q.numpy()
            momentum = (s0.body_qd.numpy()[:, :3] * model.body_mass.numpy()[:, None]).sum(axis=0)
            out.append((q[start[leader]], q[start[follower]], np.abs(momentum).max()))
        return np.array(out), joined

    def test_followers_hold_the_leaders_coordinates(self):
        """Hold a prismatic and a revolute follower at ``c0 + c1`` times their leader's coordinate at every step while
        the leader's drive moves it to its target."""
        for kind in ("prismatic", "revolute"):
            for c0, c1 in ((0.0, 1.0), (0.01, 0.5), (-0.02, -2.0)):
                with self.subTest(kind=kind, c0=c0, c1=c1):
                    model, leader, follower = self._hand(kind, c0, c1)
                    out, _ = self._run(model, leader, follower, 100, 0.01)
                    np.testing.assert_allclose(out[:, 1], c0 + c1 * out[:, 0], rtol=0.0, atol=1e-6)
                    self.assertAlmostEqual(out[-1, 0], 0.03 if kind == "prismatic" else 0.4, delta=1e-4)

    def test_mimic_forces_conserve_momentum(self):
        """Keep the total momentum of a free hand whose fingers move asymmetrically, ``c1 = 0.5``, at zero: the pair
        applies equal and opposite forces to its three bodies."""
        for kind in ("prismatic", "revolute"):
            with self.subTest(kind=kind):
                model, leader, follower = self._hand(kind, 0.01, 0.5, free_base=True)
                out, _ = self._run(model, leader, follower, 100, 0.01)
                np.testing.assert_allclose(out[:, 1], 0.01 + 0.5 * out[:, 0], rtol=0.0, atol=1e-6)
                self.assertLess(out[:, 2].max(), 1e-6)

    def test_mimic_fingers_grasp_through_the_contact_solve(self):
        """Close mimic fingers on a box, which brings their joint rows into the contact solve: the fingers stop at the
        box, the follower at ``c1`` times the leader, the box between them."""
        for c1 in (1.0, 0.5):
            with self.subTest(c1=c1):
                model, leader, follower = self._hand("prismatic", 0.0, c1, grasp=True)
                out, joined = self._run(model, leader, follower, 150, 0.005, contacts=True)
                self.assertTrue(joined)
                np.testing.assert_allclose(out[:, 1], c1 * out[:, 0], rtol=0.0, atol=1e-6)
                # The fingers' inner faces 4 cm apart, the box's width: 0.07 m between the finger centers.
                np.testing.assert_allclose(0.1 - out[-1, 0] - out[-1, 1], 0.06, atol=2e-4)

    def test_unsupported_mimics(self):
        """Warn about a mimic between ball joints, which Newton's solvers leave out, and refuse one between D6 joints
        with two angular axes, whose angles FBA does not measure."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        for kind in ("ball", "d6"):
            with self.subTest(kind=kind):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                links = [_link(builder) for _ in range(2)]
                if kind == "ball":
                    joints = [builder.add_joint_ball(-1, link) for link in links]
                else:
                    axes = [Dof(axis=(1.0, 0.0, 0.0)), Dof(axis=(0.0, 1.0, 0.0))]
                    joints = [builder.add_joint_d6(-1, link, angular_axes=axes) for link in links]
                for joint in joints:
                    builder.add_articulation([joint])
                model = builder.finalize()
                leaders = model.joint_mimic_joint.numpy()
                leaders[joints[1]] = joints[0]
                model.joint_mimic_joint.assign(leaders)
                solver = SolverFBA(model)
                if kind == "ball":
                    with self.assertWarnsRegex(newton.exceptions.NewtonWarning, "mimic"):
                        solver.step(model.state(), model.state(), None, None, 0.01)
                else:
                    with self.assertRaisesRegex(NotImplementedError, "at most one angular axis"):
                        solver.step(model.state(), model.state(), None, None, 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
