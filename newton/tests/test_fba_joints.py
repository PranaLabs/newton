# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA joint constraint rows."""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import joints as J
from newton._src.solvers.fba.joints import build_joint_rows, locked_directions
from newton._src.solvers.fba.rigid import retract_rigid_kernel

Dof = newton.ModelBuilder.JointDofConfig


def _link(builder, **kwargs):
    return builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True, **kwargs)


def _rows(model):
    slot = np.where(model.body_flags.numpy() & int(newton.BodyFlags.DYNAMIC), 0, -1)
    slot[slot == 0] = np.arange(np.count_nonzero(slot == 0))
    return build_joint_rows(model, slot, "cpu")


class TestFBAJointRows(unittest.TestCase):
    def test_locked_directions(self):
        for axes, rank in (([], 0), ([[0.0, 0.0, 2.0]], 1), ([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], 2), (np.eye(3), 3)):
            d = locked_directions(np.array(axes))
            self.assertEqual(d.shape, (3 - rank, 3))
            np.testing.assert_allclose(d @ d.T, np.eye(3 - rank), atol=1e-12)
            if rank:
                np.testing.assert_allclose(d @ np.array(axes).T, 0.0, atol=1e-12)

    def test_rows_per_joint_type(self):
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
        builder = newton.ModelBuilder()
        l0 = _link(builder)
        builder.add_articulation([builder.add_joint_distance(-1, l0, min_distance=0.5, max_distance=1.0)])
        with self.assertRaises(NotImplementedError):
            _rows(builder.finalize())


class TestFBAJointKernels(unittest.TestCase):
    def _residual(self, model, rows):
        """Row residuals at the model poses (no integrated slots: frames come from body_q)."""
        R = rows.row_joint.shape[0]
        none = wp.full(model.body_count, -1, dtype=wp.int32, device="cpu")
        res = wp.empty(R, dtype=wp.float64, device="cpu")
        args = [rows.row_joint, rows.row_angular, rows.row_dir, rows.parent, rows.child, rows.X_pj, rows.X_cj, none]
        empty = [wp.empty(0, dtype=wp.vec3d, device="cpu"), wp.empty(0, dtype=wp.quatd, device="cpu")]
        wp.launch(
            J.joint_residual_kernel,
            dim=R,
            inputs=[*args, *empty, model.body_q, model.body_com],
            outputs=[res],
            device="cpu",
        )
        return res.numpy()

    def test_residual_measures_violation(self):
        """A revolute about z, turned by phi about x and offset by delta, reads delta and 2 sin(phi/2) x."""
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

    def _satisfied_chain(self):
        """Prismatic (world, slid 0.3) -> revolute (turned 40 deg) -> ball, all satisfied."""
        builder = newton.ModelBuilder()
        q1 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7)
        q2 = wp.quat_rpy(0.3, -0.5, 1.1)
        x1 = wp.vec3(0.3, 0.5, 0.0) - wp.quat_rotate(q1, wp.vec3(0.0, -0.4, 0.0))
        x2 = x1 + wp.quat_rotate(q1, wp.vec3(0.2, 0.3, 0.0)) - wp.quat_rotate(q2, wp.vec3(0.0, 0.1, 0.2))
        l0 = _link(builder, xform=wp.transform((0.3, 0.0, 0.0)))
        l1 = _link(builder, xform=wp.transform(x1, q1), com=wp.vec3(0.05, 0.0, 0.0))
        l2 = _link(builder, xform=wp.transform(x2, q2))
        joints = [
            builder.add_joint_prismatic(-1, l0, axis=(1.0, 0.0, 0.0)),
            builder.add_joint_revolute(
                l0,
                l1,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform((0.0, 0.5, 0.0)),
                child_xform=wp.transform((0.0, -0.4, 0.0)),
            ),
            builder.add_joint_ball(
                l1, l2, parent_xform=wp.transform((0.2, 0.3, 0.0)), child_xform=wp.transform((0.0, 0.1, 0.2))
            ),
        ]
        builder.add_articulation(joints)
        return builder.finalize(device="cpu")

    def test_jacobian_matches_finite_differences(self):
        model = self._satisfied_chain()
        rows, slot = _rows(model), wp.array(np.arange(3), dtype=wp.int32, device="cpu")
        R = rows.row_joint.shape[0]
        tf = model.body_q.numpy().astype(np.float64)
        q0 = tf[:, 3:] / np.linalg.norm(tf[:, 3:], axis=1, keepdims=True)
        x0 = np.array(
            [wp.quat_rotate(wp.quatd(*q), wp.vec3d(*c)) for q, c in zip(q0, model.body_com.numpy(), strict=True)]
        )
        x0 += tf[:, :3]
        x0_d, q0_d = wp.array(x0, dtype=wp.vec3d, device="cpu"), wp.array(q0, dtype=wp.quatd, device="cpu")
        row_args = [rows.row_joint, rows.row_angular, rows.row_dir, rows.parent, rows.child, rows.X_pj, rows.X_cj, slot]

        row_slot = wp.empty((R, 2), dtype=wp.int32, device="cpu")
        row_jac = wp.empty((R, 2), dtype=wp.spatial_vectord, device="cpu")
        wp.launch(
            J.joint_jacobian_kernel,
            dim=R,
            inputs=[*row_args, x0_d, q0_d, model.body_q, model.body_com],
            outputs=[row_slot, row_jac],
            device="cpu",
        )
        jac = np.zeros((R, 18))
        for r, (slots, coeffs) in enumerate(zip(row_slot.numpy(), row_jac.numpy(), strict=True)):
            for s, c in zip(slots, coeffs, strict=True):
                if s >= 0:
                    jac[r, 6 * s : 6 * s + 6] += c

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
                outputs=[*scratch, x_live, q_live],
                device="cpu",
            )
            res = wp.empty(R, dtype=wp.float64, device="cpu")
            wp.launch(
                J.joint_residual_kernel,
                dim=R,
                inputs=[*row_args, x_live, q_live, model.body_q, model.body_com],
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
