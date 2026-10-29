# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the joint drives of the fully coupled Franka FBA examples."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.cloth.franka_fba_utils import JointDrive, joint_residual
from newton.solvers import SolverFBA


def transform(p):
    return wp.transform(wp.vec3(*p), wp.quat_identity())


class TestJointDrive(unittest.TestCase):
    def test_wrenches_balance_between_bodies(self):
        """A revolute joint on the world, and a prismatic and a revolute joint between bodies: the drive wrenches of
        the joints between bodies sum to zero force and zero moment, so only the world joint's torque is left."""
        builder = newton.ModelBuilder()
        inertia = wp.mat33(np.eye(3) * 0.01)
        a, b, c = (
            builder.add_link(xform=transform(p), mass=1.0, inertia=inertia)
            for p in ((0.2, 0, 0), (0.4, 0.1, 0), (0.3, -0.2, 0.1))
        )
        joints = [
            builder.add_joint_revolute(-1, a, axis=(0, 0, 1), child_xform=transform((-0.2, 0, 0))),
            builder.add_joint_prismatic(a, b, axis=(1, 0, 0), parent_xform=transform((0.1, 0.1, 0))),
            builder.add_joint_revolute(a, c, axis=(0, 1, 0), parent_xform=transform((0.1, -0.2, 0.1))),
        ]
        builder.add_articulation(joints)
        model = builder.finalize(device="cpu")
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        drive = JointDrive(model, 30.0, effort_limit=np.full(3, 1.0e3))
        state.clear_forces()
        tau = drive.apply(state, np.array([0.3, 0.05, -0.2]), np.zeros(3), 0.01)
        f = state.body_f.numpy().astype(float)
        com = np.array([wp.transform_point(wp.transform(*X), wp.vec3(0.0)) for X in state.body_q.numpy()])
        np.testing.assert_allclose(f[:, :3].sum(axis=0), 0.0, atol=1.0e-4)
        moment = f[:, 3:].sum(axis=0) + np.cross(com, f[:, :3]).sum(axis=0)
        np.testing.assert_allclose(moment, (0.0, 0.0, tau[0]), atol=1.0e-4)
        self.assertTrue(np.all(np.abs(tau) > 1.0e-3))

    def test_contact_load_acts_back_on_driven_arm(self):
        """A driven arm holds a paddle level while SolverFBA integrates it; a 0.5 kg cloth patch resting on the
        paddle through frictional contacts raises the joint torque by the load's moment and lowers the paddle, which a
        paddle without the particle does not."""
        results = []
        for load in (False, True):
            builder = newton.ModelBuilder()
            arm = builder.add_link(xform=transform((0.15, 0, 0.5)), mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
            builder.add_shape_box(arm, hx=0.15, hy=0.05, hz=0.01)
            joint = builder.add_joint_revolute(
                -1, arm, axis=(0, 1, 0), parent_xform=transform((0, 0, 0.5)), child_xform=transform((-0.15, 0, 0))
            )
            builder.add_articulation([joint])
            # a 0.5 kg cloth patch on the paddle, 0.25 m out, or on the ground beside it
            z = 0.5 + 0.01 + 0.012 if load else 0.012
            builder.add_cloth_grid(
                pos=wp.vec3(0.21, -0.04, z),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=4,
                dim_y=4,
                cell_x=0.02,
                cell_y=0.02,
                mass=0.02,
                tri_ke=100.0,
                particle_radius=0.01,
            )
            builder.add_ground_plane()
            model = builder.finalize(device="cpu")
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            solver = SolverFBA(model, iterations=10, nsn_schur_mode="full")
            pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
            contacts = pipeline.contacts()
            drive = JointDrive(model, 10.0, effort_limit=np.full(1, 50.0))
            dt = 1.0 / 240.0
            for _ in range(480):
                state_0.clear_forces()
                drive.apply(state_0, np.zeros(1), np.zeros(1), dt)
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, None, contacts, dt)
                state_0, state_1 = state_1, state_0
            drive.measure(state_0)
            gap, _ = joint_residual(model, state_0.body_q.numpy())
            self.assertLess(gap, 1.0e-4)
            results.append((drive.q[0], drive.tau[0]))
        (q_free, tau_free), (q_load, tau_load) = results
        # the patch sits 0.25 m out: its weight turns the paddle down, about +y
        self.assertLess(abs(q_free), 2.0e-3)
        self.assertGreater(q_load - q_free, 5.0e-3)
        self.assertAlmostEqual(tau_free - tau_load, 0.5 * 9.81 * 0.25, delta=0.3)


if __name__ == "__main__":
    unittest.main()
