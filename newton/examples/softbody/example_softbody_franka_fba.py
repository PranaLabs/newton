# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Body Franka FBA
#
# A Franka Panda picks a soft Neo-Hookean ball off a table between its
# fingertip pads, lifts it, sets it down and lets go. Only friction holds
# the ball: it is compressible, so the closed pads press into it and their
# contacts carry its weight. Newton's IK solver poses the robot and
# Featherstone integrates it as a kinematic chain; SolverFBA steps the ball
# against the moving links.
#
# Command: python -m newton.examples softbody_franka_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.examples.softbody.example_squeezing_ball_fba import _ball_mesh
from newton.solvers import SolverFBA, SolverFeatherstone

RADIUS = 0.03  # [m]
CENTER = wp.vec3(0.0, -0.5, 0.2 + RADIUS)  # resting on the table top at 0.2 m
OPEN, CLOSED = 0.04, 0.022  # finger joint positions [m]: the pads stand twice that apart
LIFT = 0.1  # [m]
THICKNESS = 0.001  # contact thickness of the ball surface [m]
# Key times [s] with the height of the middle of the pads over the ball center [m] and the finger positions [m].
KEY_T = [0.0, 1.5, 2.5, 4.0, 5.5, 7.0, 8.0, 9.0]
KEY_H = [0.12, 0.0, 0.0, LIFT, LIFT, 0.0, 0.0, 0.12]
KEY_Q = [OPEN, OPEN, CLOSED, CLOSED, CLOSED, CLOSED, OPEN, OPEN]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 1.0  # rubber pads on the fingers
        builder.add_urdf(
            str(newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            joint_damping_mode="passive",
        )
        builder.joint_q[:] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0, OPEN, OPEN]
        self.hand = builder.body_count - 3
        self.fingers = [builder.body_count - 2, builder.body_count - 1]
        table = newton.ModelBuilder.ShapeConfig(mu=0.5)
        builder.add_shape_box(
            -1, xform=wp.transform((0.0, -0.5, 0.1), wp.quat_identity()), hx=0.4, hy=0.4, hz=0.1, cfg=table
        )

        # A 113 g foam ball: young 50 kPa, poisson 0.4.
        young, poisson = 5.0e4, 0.4
        mu = young / (2.0 * (1.0 + poisson))
        lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        vertices, tets = _ball_mesh(13)
        builder.add_soft_mesh(
            pos=CENTER + wp.vec3(0.0, 0.0, THICKNESS),
            rot=wp.quat_identity(),
            scale=RADIUS,
            vel=wp.vec3(0.0),
            vertices=vertices.tolist(),
            indices=tets.ravel().tolist(),
            density=1000.0,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
            particle_radius=THICKNESS,
        )
        builder.add_ground_plane()
        self.model = builder.finalize()

        # IK places the middle of the fingertip pads, 0.21 m along the last link, pointing down; the joint limits
        # keep the arm in range. The robot starts in the first key pose.
        self.ik_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.pads = ik.IKObjectivePosition(self.hand, wp.vec3(0.0, 0.0, 0.21065), wp.array([CENTER], dtype=wp.vec3))
        down = ik.IKObjectiveRotation(self.hand, wp.quat_identity(), wp.array([(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4))
        limits = ik.IKObjectiveJointLimit(self.model.joint_limit_lower, self.model.joint_limit_upper, weight=10.0)
        objectives = [self.pads, down, limits]
        self.ik = ik.IKSolver(self.model, 1, objectives, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self.model.joint_q.assign(self.key_pose(0.0, iterations=100))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.joint_qd = wp.zeros_like(self.state_0.joint_qd)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.solver = SolverFBA(
            self.model,
            iterations=10,
            stretching_model="neohookean",
            integrate_with_external_rigid_solver=True,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.005)
        self.contacts = self.pipeline.contacts()
        self.gravity = self.model.gravity.numpy()

        # The collision boxes of the fingers: body, transform in the body and half extents.
        m = self.model
        boxes = np.isin(m.shape_body.numpy(), self.fingers) & (m.shape_type.numpy() == newton.GeoType.BOX)
        shapes = zip(
            m.shape_body.numpy()[boxes], m.shape_transform.numpy()[boxes], m.shape_scale.numpy()[boxes], strict=True
        )
        self.finger_boxes = [(body, wp.transform(*X), half) for body, X, half in shapes]
        self.lifted, self.held, self.before = 0.0, [], None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.24, -0.26, 0.333), pitch=-8.0, yaw=225.0)

    def key_pose(self, t, iterations=24):
        """Joint positions of the robot at time ``t`` of the key poses."""
        height, finger = np.interp(t, KEY_T, KEY_H), np.interp(t, KEY_T, KEY_Q)
        self.pads.set_target_position(0, CENTER + wp.vec3(0.0, 0.0, height))
        self.ik.step(self.ik_q, self.ik_q, iterations=iterations)
        q = self.ik_q.numpy()[0]
        q[-2:] = finger
        return q

    def simulate(self):
        target = self.key_pose(self.sim_time + self.frame_dt)
        self.joint_qd.assign((target - self.state_0.joint_q.numpy()) / self.frame_dt)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            # Featherstone moves only the robot along the IK velocities; the forces it puts on the particles
            # are its own model of the ball.
            particles, self.model.particle_count = self.model.particle_count, 0
            self.model.gravity.zero_()
            self.state_0.joint_qd.assign(self.joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particles
            self.model.gravity.assign(self.gravity)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        x, body_q = self.state_0.particle_q.numpy(), self.state_0.body_q.numpy()
        speed = np.abs(self.state_0.particle_qd.numpy()).max()
        assert speed < 0.5, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        for body, X_bs, half in self.finger_boxes:
            X = wp.transform(*body_q[body]) * X_bs
            d = np.abs((x - X.p) @ np.array(wp.quat_to_matrix(X.q)).reshape(3, 3)) - half
            gap = (np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)).min()
            assert gap > 0.25 * THICKNESS, f"a particle is {gap * 1e3:.2f} mm from a finger at t={self.sim_time:.2f} s"
        center = x.mean(axis=0)
        self.lifted = max(self.lifted, center[2] - CENTER[2])
        if KEY_T[3] + 0.25 <= self.sim_time <= KEY_T[4]:
            self.held.append(center - body_q[self.fingers, :3].mean(axis=0))
        if self.before is None and self.sim_time >= KEY_T[1]:
            self.before = x

    def test_final(self):
        assert self.lifted > 0.9 * LIFT, f"the ball rose by {self.lifted * 1e3:.1f} mm"
        slip = np.ptp(np.array(self.held), axis=0).max()
        assert slip < 0.5e-3, f"the ball slid by {slip * 1e3:.2f} mm in the hold"
        # Back on the table, the ball regains the shape it had before the grasp, up to a rigid motion.
        a, b = (q - q.mean(axis=0) for q in (self.state_0.particle_q.numpy(), self.before))
        u, _, vt = np.linalg.svd(b.T @ a)
        deviation = np.linalg.norm(b @ u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt - a, axis=1).max()
        assert deviation < 0.2e-3, f"the ball is {deviation * 1e3:.2f} mm off its shape after the release"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
