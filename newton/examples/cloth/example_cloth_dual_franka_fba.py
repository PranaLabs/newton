# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dual Franka FBA
#
# Two Franka Panda arms pinch the far corners of a towel and fold it in
# half, its layers resting on each other through cloth self-contact. IK
# poses the arms and Featherstone moves them kinematically; SolverFBA steps
# the towel with the lite NSN and holds the pinched particles.
#
# Command: uv run -m newton.examples cloth_dual_franka_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.solvers import SolverFBA, SolverFeatherstone

TABLE_TOP, CELL = 0.2, 0.01  # table top and towel grid spacing [m]
NEAR, FAR = -0.2, 0.16  # towel edges along x [m]; the fold line lies midway
CORNERS_Y = (-0.75, -0.25)  # the towel's side edges, in line with the arms' bases [m]
THICKNESS, SELF_RADIUS = 0.004, 0.01  # the towel's contact thickness and the distance its folded layers keep [m]
OPEN, CLOSED = 0.04, 0.012  # finger joint positions [m]: the pads stand twice that apart
# From the last link to the middle of the fingertip pads, and their half width and half length [m]; their axes in
# the last link's frame, as columns: closing, spanning the pads and out of the fingertips.
PAD, PAD_HALF_WIDTH, PAD_HALF_LENGTH = 0.21065, 0.00875, 0.00925
PAD_AXES = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)]]) / np.sqrt(2.0)
INSET = 0.005  # the pads straddle the 2 x 2 particles at each corner [m]
GRASP = TABLE_TOP + 2.0 * THICKNESS + 0.002 + PAD_HALF_LENGTH  # pad middles just over the towel [m]
# The pinched particles rise over 0.5 s to 2 mm under the pads' lower edges: into the gap, they would pull the cloth
# beside it 1.4 mm into the pads, whose contacts see a particle's mass alone.
GRASP_RISE, INTO_GAP = 0.5, -0.002
# Key times [s], pad middles' x and height [m] and finger positions [m]: down onto the far corners, pinch, a half circle
# about the fold line, set down 5 mm over the layer's self-contact radius and contact thickness, let go and rise.
_ARC, _MID, _R = np.radians([45.0, 90.0, 135.0, 180.0]), 0.5 * (FAR + NEAR), 0.5 * (FAR - NEAR) - INSET
KEY_T = [0.0, 1.5, 2.5, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0, 7.0]
KEY_X = [_MID + _R] * 3 + [*(_MID + _R * np.cos(_ARC)), *[_MID - _R] * 3]
KEY_Z = [GRASP + 0.1, GRASP, GRASP, *(GRASP + _R * np.sin(_ARC[:3])), *[GRASP + SELF_RADIUS + 0.005] * 3, GRASP + 0.1]
KEY_Q = [OPEN, OPEN, *[CLOSED] * 6, OPEN, OPEN]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt, self.sim_substeps = 1.0 / 60.0, 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 1.0
        urdf = str(newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf")
        self.hands, self.fingers = [], []
        options = {"floating": False, "enable_self_collisions": False, "collapse_fixed_joints": True}
        for y in CORNERS_Y:
            builder.add_urdf(urdf, xform=wp.transform((-0.55, y, 0.0), wp.quat_identity()), **options)
            builder.joint_q[-9:] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0, OPEN, OPEN]
            self.hands.append(builder.body_count - 3)
            self.fingers.append(builder.joint_coord_count - 2)
        builder.add_shape_box(
            -1, xform=wp.transform((0.0, -0.5, 0.1), wp.quat_identity()), hx=0.4, hy=0.4, hz=0.1, color=(0.5,) * 3
        )
        # A 1 kg/m^2 towel whose stretching, 50 N/m, and bending match the Franka shirt example's per unit mass.
        towel = {"tri_ke": 25.0, "tri_ka": 50.0, "edge_ke": 7.5e-6, "particle_radius": THICKNESS}
        cells = round((FAR - NEAR) / CELL), round((CORNERS_Y[1] - CORNERS_Y[0]) / CELL)
        corner = wp.vec3(NEAR, CORNERS_Y[0], TABLE_TOP + THICKNESS)
        builder.add_cloth_grid(
            pos=corner,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=cells[0],
            dim_y=cells[1],
            cell_x=CELL,
            cell_y=CELL,
            mass=CELL * CELL,
            **towel,
        )
        builder.add_ground_plane()
        self.model = builder.finalize()

        # IK places the middle of each hand's fingertip pads, pointing down; the arms start in the first key pose.
        self.ik_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.pads = [ik.IKObjectivePosition(h, wp.vec3(0.0, 0.0, PAD), wp.zeros(1, wp.vec3)) for h in self.hands]
        down = wp.array([(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4)
        objectives = [*self.pads, *(ik.IKObjectiveRotation(h, wp.quat_identity(), down) for h in self.hands)]
        objectives.append(ik.IKObjectiveJointLimit(self.model.joint_limit_lower, self.model.joint_limit_upper, 10.0))
        self.ik = ik.IKSolver(self.model, 1, objectives, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self.model.joint_q.assign(self.key_pose(0.0, iterations=100))

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.joint_qd = wp.zeros_like(self.state_0.joint_qd)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.solver = SolverFBA(
            self.model,
            iterations=5,
            stretching_model="arap",
            nsn_schur_mode="lite",
            nsn_linear_iterations=10,
            integrate_with_external_rigid_solver=True,
            soft_contact_thickness=THICKNESS,
            # Bounds a contact row at 1 N, five times what a landing particle needs.
            lambda_cap=1.0,
            particle_enable_self_contact=True,
            particle_self_contact_radius=SELF_RADIUS,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=SELF_RADIUS,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=THICKNESS)
        self.contacts = self.pipeline.contacts()
        self.gravity = self.model.gravity.numpy()
        self.held = [None, None]  # per hand: pinched particles, their pad frame points before and after rising, time
        # The fingers' collision boxes: body, transform in it and half extents.
        m, bodies = self.model, [h + k for h in self.hands for k in (1, 2)]
        boxes = np.isin(m.shape_body.numpy(), bodies) & (m.shape_type.numpy() == newton.GeoType.BOX)
        shapes = zip(*(a.numpy()[boxes] for a in (m.shape_body, m.shape_transform, m.shape_scale)), strict=True)
        self.finger_boxes = [(body, wp.transform(*X), half) for body, X, half in shapes]

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.25, -0.5, 0.95), pitch=-25.0, yaw=180.0)

    def key_pose(self, t, iterations=24):
        """Joint positions of both arms at time ``t`` of the key poses."""
        x, z, finger = (np.interp(t, KEY_T, k) for k in (KEY_X, KEY_Z, KEY_Q))
        for pads, y, inward in zip(self.pads, CORNERS_Y, (INSET, -INSET), strict=True):
            pads.set_target_position(0, wp.vec3(x, y + inward, z))
        self.ik.step(self.ik_q, self.ik_q, iterations=iterations)
        q = self.ik_q.numpy()[0]
        for f in self.fingers:
            q[f : f + 2] = finger
        return q

    def pad_frame(self, state, hand):
        """Middle of a hand's fingertip pads and their axes."""
        X = wp.transform(*state.body_q.numpy()[self.hands[hand]])
        R = np.array(wp.quat_to_matrix(X.q)).reshape(3, 3)
        return np.array(X.p) + R @ np.array([0.0, 0.0, PAD]), R @ PAD_AXES

    def update_grasp(self):
        """Pinch the particles under each closed gap, raise them and hold them at the pads' poses after the substep."""
        q, rows = self.state_1.joint_q.numpy(), []
        for hand, f in enumerate(self.fingers):
            origin, axes = self.pad_frame(self.state_1, hand)
            if q[f] > 0.5 * (OPEN + CLOSED):
                self.held[hand] = None
            elif self.held[hand] is None and q[f] < CLOSED + 1.0e-3:
                local = (self.state_0.particle_q.numpy() - origin) @ axes
                depth = local[:, 2] - PAD_HALF_LENGTH  # below the pads' lower edges
                gap = np.all(np.abs(local[:, :2]) <= [CLOSED - THICKNESS, PAD_HALF_WIDTH], axis=1)
                taken = np.flatnonzero(gap & (depth >= 0.0) & (depth <= 2.0 * THICKNESS + 0.002))[:10]
                rise = np.array([0.0, 0.0, depth[taken].max(initial=0.0) + INTO_GAP])
                self.held[hand] = (taken, local[taken], local[taken] - rise, self.sim_time)
            if self.held[hand] is not None:
                taken, start, end, t0 = self.held[hand]
                s = min((self.sim_time - t0) / GRASP_RISE, 1.0)
                rows.append((taken, origin + (start + s * (end - start)) @ axes.T))
        if not rows:
            self.solver.set_particle_constraints(None, None, None)
            return
        p, x = (np.concatenate(a) for a in zip(*rows, strict=True))
        eye = np.tile(np.eye(3), (len(p), 1))
        self.solver.set_particle_constraints(np.repeat(p, 3)[:, None], np.ones((3 * len(p), 1)), x.reshape(-1), eye)

    def step(self):
        target = self.key_pose(self.sim_time + self.frame_dt)
        self.joint_qd.assign((target - self.state_0.joint_q.numpy()) / self.frame_dt)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            # Featherstone moves only the arms along the IK velocities.
            particles, self.model.particle_count = self.model.particle_count, 0
            self.model.gravity.zero_()
            self.state_0.joint_qd.assign(self.joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0.particle_f.zero_()
            self.model.particle_count = particles
            self.model.gravity.assign(self.gravity)
            self.pipeline.collide(self.state_0, self.contacts)
            self.update_grasp()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def test_post_step(self):
        x, body_q = self.state_0.particle_q.numpy(), self.state_0.body_q.numpy()
        speed = np.abs(self.state_0.particle_qd.numpy()).max()
        assert speed < 2.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        free = np.ones(len(x), dtype=bool)
        for hand, held in enumerate(self.held):
            if held is not None:
                free[held[0]] = False
                origin, axes = self.pad_frame(self.state_0, hand)
                off = np.abs(x[held[0]] - origin - held[2] @ axes.T).max()
                assert self.sim_time - held[3] < GRASP_RISE or off < 1.0e-3, f"a pinched particle is {off:.4f} m off"
        for body, X_bs, half in self.finger_boxes:
            X = wp.transform(*body_q[body]) * X_bs
            inside = np.abs((x[free] - X.p) @ np.array(wp.quat_to_matrix(X.q)).reshape(3, 3)) - half
            assert inside.max(axis=1).min() > 0.0, f"a towel particle is inside a finger at t={self.sim_time:.2f} s"

    def test_final(self):
        # The far half lies folded over the near half: where it covers the flat near half, away from the fold's
        # curl and the edge that swings past, it keeps at least half the self-contact radius from it.
        x0, x = self.model.particle_q.numpy(), self.state_0.particle_q.numpy()
        far = x[x0[:, 0] > _MID]
        assert (far[:, 0] < _MID + SELF_RADIUS).all(), "part of the towel's far half is not folded over"
        over = far[(far[:, 0] > NEAR + CELL) & (far[:, 0] < _MID - 2.0 * SELF_RADIUS)]
        low = over[:, 2].min() - (TABLE_TOP + THICKNESS)
        assert low > 0.5 * SELF_RADIUS, f"the folded layer comes {low * 1e3:.1f} mm close to the one below"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=420)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
