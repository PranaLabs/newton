# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for episode resets and joint drives in the Franka FBA examples."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.examples.cloth import example_cloth_dual_franka_fba as towel
from newton.examples.cloth.franka_fba_utils import JointDrive, joint_residual
from newton.solvers import SolverFBA
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.viewer import ViewerNull

if USD_AVAILABLE:
    from newton.examples.cloth import example_shirt_fold_fba as shirt_example


def transform(p):
    return wp.transform(wp.vec3(*p), wp.quat_identity())


class TestFrankaFBAEpisodes(unittest.TestCase):
    """Exercise the real examples with coarse cloth to keep episode checks short."""

    @staticmethod
    def example(shirt, worlds=2, device="cpu"):
        if shirt and not USD_AVAILABLE:
            raise unittest.SkipTest("requires usd-core for the shirt asset")

        args = SimpleNamespace(
            world_count=worlds, fully_coupled=False, action_noise=0.0001, world_spacing=2.0, cloth_edge_length=0.1
        )
        with wp.ScopedDevice(device), patch.object(towel, "CELL", 0.1):
            return (shirt_example if shirt else towel).Example(ViewerNull(), args)

    def test_cpu_gravity_survives_robot_step(self):
        """Preserve CPU gravity while Featherstone advances the arms with gravity temporarily disabled."""
        for shirt in (False, True):
            with self.subTest(shirt=shirt):
                ex = self.example(shirt)
                gravity = ex.model.gravity.numpy().copy()
                ex.step()
                np.testing.assert_array_equal(ex.model.gravity.numpy(), gravity)

    def test_partial_episode_reset(self):
        """Match a fresh episode after reset and an uninterrupted reference in the neighboring world."""
        for shirt in (False, True):
            with self.subTest(shirt=shirt):
                ex, control = self.example(shirt), self.example(shirt)
                initial_ik = ex.ik_q.numpy().copy()
                for _ in range(2):
                    ex.step()
                    control.step()
                neighbor = {
                    name: getattr(ex.state_0, name).numpy().copy()
                    for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
                }
                mask = wp.array([True, False, False], dtype=wp.bool, device="cpu")
                ex.reset(mask)
                np.testing.assert_array_equal(ex.episode_time, [0.0, control.sim_time])
                np.testing.assert_array_equal(ex.ik_q.numpy()[0], initial_ik[0])
                for name, before in neighbor.items():
                    stride = len(before) // 2
                    np.testing.assert_array_equal(getattr(ex.state_0, name).numpy()[stride:], before[stride:])
                fresh = self.example(shirt)
                for _ in range(3):
                    ex.step()
                    control.step()
                    fresh.step()
                for name in neighbor:
                    value = getattr(ex.state_0, name).numpy()
                    stride = len(value) // 2
                    np.testing.assert_allclose(value[:stride], getattr(fresh.state_0, name).numpy()[:stride], atol=2e-6)
                    np.testing.assert_allclose(
                        value[stride:], getattr(control.state_0, name).numpy()[stride:], atol=2e-6
                    )
                # An empty selection is a no-op; a full reset restores both buffers.
                ex.reset(wp.array([False, False, False], dtype=wp.bool, device="cpu"))
                ex.reset()
                for state in (ex.state_0, ex.state_1):
                    np.testing.assert_array_equal(state.particle_q.numpy(), ex.model.particle_q.numpy())
                    np.testing.assert_array_equal(state.joint_q.numpy(), ex.model.joint_q.numpy())
                np.testing.assert_array_equal(ex.episode_time, 0.0)
                ex.test_final()
                if not shirt:
                    # At the completed episode's boundary, an unfolded towel must fail the final check.
                    ex.plan_second_fold(range(ex.world_count))
                    end = ex.keys[0][-1][0]
                    ex.episode_time[:] = np.nextafter(end, -np.inf)
                    with self.assertRaises(AssertionError):
                        ex.test_final()

    def test_single_world_global_reset(self):
        """Reset an unreplicated scene through the global slot and reject malformed masks before mutation."""
        for shirt in (False, True):
            with self.subTest(shirt=shirt):
                ex = self.example(shirt, worlds=1)
                ex.step()
                before = ex.state_0.particle_q.numpy().copy()
                # An unreplicated model's entities are global, not world 0.
                ex.reset(wp.array([True, False], dtype=wp.bool, device="cpu"))
                np.testing.assert_array_equal(ex.state_0.particle_q.numpy(), before)
                with self.assertRaises(ValueError):
                    ex.reset(wp.array([True], dtype=wp.bool, device="cpu"))
                np.testing.assert_array_equal(ex.state_0.particle_q.numpy(), before)
                ex.reset(wp.array([False, True], dtype=wp.bool, device="cpu"))
                np.testing.assert_array_equal(ex.state_0.particle_q.numpy(), ex.model.particle_q.numpy())
                np.testing.assert_array_equal(ex.state_0.joint_q.numpy(), ex.model.joint_q.numpy())
                ex.step()
                self.assertTrue(np.isfinite(ex.state_0.particle_q.numpy()).all())

    def test_reset_preserves_neighbor_grasp_and_home_phase(self):
        """Retain the neighboring episode's grasp rows, return trajectory and self-contact margin."""
        for shirt in (False, True):
            with self.subTest(shirt=shirt):
                ex, control = self.example(shirt), self.example(shirt)
                for e in (ex, control):
                    if shirt:
                        e.key_pose(e.plan.home[0] - 0.02, noise=False)
                        e.episode_time[:] = e.plan.home[0] + 0.4
                        e.key_pose(e.episode_time, noise=False)
                        e.update_contact_margin()
                    else:
                        q = e.key_pose(2.2, iterations=100, noise=False)
                        for state in (e.state_0, e.state_1):
                            state.joint_q.assign(q)
                            state.joint_qd.zero_()
                            newton.eval_fk(e.model, state.joint_q, state.joint_qd, state)
                        e.episode_time[:] = 2.2
                        e.update_grasp()
                        self.assertIsNotNone(e.grasp_rows)
                        e.step()
                rows = ex.grasp_rows
                ex.reset(wp.array([True, False, False], dtype=wp.bool, device="cpu"))
                if shirt:
                    self.assertIsNone(ex.home_q[0])
                    np.testing.assert_array_equal(ex.home_q[1], control.home_q[1])
                    np.testing.assert_array_equal(
                        ex.contact_margin, [shirt_example.LAND_MARGIN, shirt_example.FOLD_MARGIN]
                    )
                else:
                    keep = rows[0][:, 0] >= ex.particles
                    self.assertTrue(keep.any())
                    for actual, before in zip(ex.grasp_rows[:4], rows[:4], strict=True):
                        np.testing.assert_array_equal(actual, before[keep])
                for _ in range(3):
                    if shirt:
                        ex.episode_time += ex.frame_dt
                        control.episode_time += control.frame_dt
                        actual = ex.key_pose(ex.episode_time).reshape(2, -1)
                        expected = control.key_pose(control.episode_time).reshape(2, -1)
                        np.testing.assert_array_equal(actual[1], expected[1])
                    else:
                        ex.step()
                        control.step()
                        for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
                            actual, expected = (getattr(e.state_0, name).numpy() for e in (ex, control))
                            np.testing.assert_allclose(
                                actual[len(actual) // 2 :], expected[len(actual) // 2 :], atol=2e-6
                            )


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
