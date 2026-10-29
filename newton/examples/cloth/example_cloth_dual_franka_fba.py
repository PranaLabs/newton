# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dual Franka FBA
#
# Two Franka Panda arms fold a square towel twice into a square a quarter
# its size: they pinch its far corners and lay them on its near ones, then
# pinch both layers at the corners of one side of the folded towel and
# lay them on the other side. Closing on a corner, the finger pads squeeze
# the towel between them up into a ridge; the hand holds the ridge while
# the towel around it turns over, and opening, lays it out flat where the
# fold puts it. The layers rest on each other through cloth self-contact.
# IK poses the arms and Featherstone moves them kinematically; SolverFBA
# steps the towel with the lite NSN and holds the pinched particles. Light
# Rayleigh mass damping keeps the towel from bouncing as the hands carry
# it.
#
# With --fully-coupled, an experiment: SolverFBA integrates the arms as
# well, with their joints as constraints and their contacts with the towel
# and the table in the same frictional solve, so the towel acts back on
# them. Stable PD drives with gravity compensation and force limits follow
# the IK targets, and the fingers pinch the towel by friction alone, with
# no held particles; the pinches are not reliable enough for the folds.
#
# With --world-count N, N copies of the arms, table and towel fold in
# parallel worlds, stepped by one Featherstone and one SolverFBA, each
# world with its own IK problem, fold plan and grasps; --action-noise
# biases each world's arm joint targets by a fixed amount, so that the
# worlds differ.
#
# Command: python -m newton.examples cloth_dual_franka_fba
# Parallel worlds: python -m newton.examples cloth_dual_franka_fba --world-count 4 --action-noise 0.0001
# Fully coupled: python -m newton.examples cloth_dual_franka_fba --fully-coupled
###########################################################################

from typing import NamedTuple

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.examples.cloth.franka_fba_utils import (
    ArmNoise,
    JointDrive,
    add_world_args,
    joint_residual,
    view_worlds,
    world_args,
)
from newton.solvers import SolverFBA, SolverFeatherstone

TABLE_TOP, CELL = 0.2, 0.01  # table top and towel grid spacing [m]
SIZE, FAR = 0.4, 0.16  # the towel's side and its far edge along x [m]
NEAR, SIDES = FAR - SIZE, (-0.5 - 0.5 * SIZE, -0.5 + 0.5 * SIZE)  # its near edge along x and side edges along y [m]
BASES_Y = (-0.75, -0.25)  # the arms' bases along y [m]
# The towel rests THICKNESS over the table, its particle radius plus the table's margin, and its layers keep
# SELF_MARGIN apart. The finger pads keep FINGER_MARGIN [m] from it: the lite operator lets towel that a pinch pulls
# against a pad settle into its contact distance.
THICKNESS, CLOTH_RADIUS, SELF_MARGIN, FINGER_MARGIN = 0.004, 0.001, 0.006, 0.003
OPEN, CLOSED = 0.04, 0.004  # finger joint positions, wide open and closed on a corner [m]; the pads stand twice apart
# From the last link to the middle of the fingertip pads, and their half width and half length [m]; their axes in
# the last link's frame, as columns: closing, spanning the pads and out of the fingertips.
PAD, PAD_HALF_WIDTH, PAD_HALF_LENGTH = 0.21065, 0.00875, 0.00925
PAD_THICKNESS = 0.0152  # a pad's extent along the closing axis [m]
PAD_AXES = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)]]) / np.sqrt(2.0)
# The pads come down on the towel with their lower edges at its contact distance, OVER [m] from their middle, and
# hold the towel that wraps around their lower edges at that distance under them, UNDER [m] from their middle.
OVER = CLOTH_RADIUS + FINGER_MARGIN + PAD_HALF_LENGTH
UNDER = PAD_HALF_LENGTH + CLOTH_RADIUS + FINGER_MARGIN
# While the fingers close on a corner, the towel under and between the pads and the row beside their inner ends, from
# each layer in turn, follows them as static friction would: under a pad it moves with the pad, between the pads it
# shortens with the gap and rises into a ridge that keeps its length. At most SQUEEZED particles per hand, nearest the
# pads' middle, take three rows each within the lite operator's 64 point rows.
SQUEEZED = 10
# A weak IK pull on each elbow, ELBOW_OUT [m] outward from its start, keeps the arms apart when both hands work on
# the same side of the towel.
ELBOW_WEIGHT, ELBOW_OUT = 0.05, 0.25
# Timing [s] of a fold: down onto the corners, pinch, a half circle over the fold line, hold, let the ridge out,
# rise; the start of the second fold, and the time the arms take to draw back after it.
DOWN, PINCH, ARC, HOLD, LET_OUT, RISE = 1.5, 1.0, 2.3, 1.2, 0.5, 0.5
FOLD2, SETTLE = 7.0, 1.0
GRASP = TABLE_TOP + THICKNESS + OVER  # pad middles over the flat towel [m]
# Past the top of their half circle the pads stay CLEAR [m] over where they lay the corners; the second fold lays its
# corners LAY_CLEAR [m] over the layers they come down on, as the folded towel lies less even than the flat one.
CLEAR, LAY_CLEAR = 0.007, 0.003


class Fold(NamedTuple):
    """How the hands take the corners of one fold."""

    yaw: float
    """Turn of both hands about the vertical [deg], which closes their fingers along the fold line, so that the
    ridge they hold turns over between the pads with the towel around it."""
    grip_open: float
    """Finger position open around a corner [m], which sets how much towel the squeeze gathers into the ridge."""
    gathered: float
    """How much the radius of the hands' half circle shrinks over its first 45 degrees, and grows back over its last
    [m]: the towel that the squeeze gathers from between the fold line and the hand."""


# With --fully-coupled: the rubber finger pads grip the towel with a friction coefficient over the table's, so that
# pads pressed on it PRESS [m] into their contact distance drag it over the table; the hands take each corner INSET
# [m] in from it along the fold line, the fingers' closing axis, and across it, so that both pads come down on the
# towel and the towel between them buckles up as they close. The fingers close to 0; stopped on the towel over STOP
# [m], they squeeze it with SQUEEZE [N] more as their command closes, within GRIP_FORCE [N]. The drives' natural
# frequencies [rad/s] keep stable at four substeps.
PAD_MU, PRESS, INSET_ALONG, INSET_ACROSS = 2.0, 0.002, 0.025, 0.01
STOP, SQUEEZE, GRIP_FORCE = 0.0015, 15.0, 20.0
ARM_FREQUENCY, FINGER_FREQUENCY = 30.0, 450.0

FOLDS = (Fold(-45.0, 0.018, 0.05), Fold(45.0, 0.018, 0.0))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fully_coupled = getattr(args, "fully_coupled", False)
        self.world_count, noise, spacing = world_args(args)
        if self.fully_coupled and self.world_count > 1:
            raise ValueError("--fully-coupled runs a single world")
        # The fully coupled drives push the arms with the forces of each step's start, which four substeps keep stable.
        self.frame_dt, self.sim_substeps = 1.0 / 60.0, 4 if self.fully_coupled else 2
        self.closed, self.press = (0.0, PRESS) if self.fully_coupled else (CLOSED, 0.0)
        self.inset = (INSET_ALONG, INSET_ACROSS) if self.fully_coupled else (0.0, 0.0)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        def new_builder():
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.mu = 1.0
            builder.default_shape_cfg.margin = THICKNESS - CLOTH_RADIUS
            return builder

        # One world's arms, table and towel; parallel worlds replicate it in place and share the ground.
        builder = template = new_builder()
        urdf = str(newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf")
        self.hands, self.fingers, self.elbows = [], [], []
        options = {
            "floating": False,
            "enable_self_collisions": False,
            "collapse_fixed_joints": True,
            "joint_damping_mode": "passive",
        }
        for y in BASES_Y:
            template.add_urdf(urdf, xform=wp.transform((-0.55, y, 0.0), wp.quat_identity()), **options)
            template.joint_q[-9:] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0, OPEN, OPEN]
            self.hands.append(template.body_count - 3)
            self.elbows.append(template.body_count - 6)  # link4
            self.fingers.append(template.joint_coord_count - 2)
        template.add_shape_box(
            -1, xform=wp.transform((0.0, -0.5, 0.1), wp.quat_identity()), hx=0.4, hy=0.4, hz=0.1, color=(0.5,) * 3
        )
        # A 1 kg/m^2 towel with a stretching weight of 50 N/m (2 * tri_ke) and soft bending.
        towel = {"tri_ke": 25.0, "tri_ka": 50.0, "edge_ke": 7.5e-6, "particle_radius": CLOTH_RADIUS}
        cells = round(SIZE / CELL)
        template.add_cloth_grid(
            pos=wp.vec3(NEAR, SIDES[0], TABLE_TOP + THICKNESS),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=cells,
            dim_y=cells,
            cell_x=CELL,
            cell_y=CELL,
            mass=CELL * CELL,
            **towel,
        )
        # strides of a world's bodies, joint coordinates and particles
        self.bodies, self.coords = template.body_count, template.joint_coord_count
        self.particles = template.particle_count
        if self.world_count > 1:
            builder = new_builder()
            builder.replicate(template, self.world_count)
        builder.add_ground_plane()
        self.model = builder.finalize()
        fingers = [h + k + w * self.bodies for w in range(self.world_count) for h in self.hands for k in (1, 2)]
        margin, body = self.model.shape_margin.numpy(), self.model.shape_body.numpy()
        margin[np.isin(body, fingers)] = FINGER_MARGIN
        self.model.shape_margin.assign(margin)
        if self.fully_coupled:
            mu = self.model.shape_material_mu.numpy()
            mu[np.isin(body, fingers)] = PAD_MU
            self.model.shape_material_mu.assign(mu)

        # IK places the middle of each hand's fingertip pads, pointing down and turned by the keys' yaw, and pulls the
        # elbows outward a little; the arms start in the first key pose. Each world is one problem on one world's
        # arms.
        n = self.world_count
        ik_model = self.model if n == 1 else template.finalize()
        home = ik_model.state()
        newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, home)
        self.ik_q = wp.array(np.tile(ik_model.joint_q.numpy(), (n, 1)), dtype=wp.float32)
        self.pads = [ik.IKObjectivePosition(h, wp.vec3(0.0, 0.0, PAD), wp.zeros(n, wp.vec3)) for h in self.hands]
        down = wp.array([(1.0, 0.0, 0.0, 0.0)] * n, dtype=wp.vec4)
        self.turns = [ik.IKObjectiveRotation(h, wp.quat_identity(), wp.clone(down)) for h in self.hands]
        objectives = [*self.pads, *self.turns]
        objectives.append(ik.IKObjectiveJointLimit(ik_model.joint_limit_lower, ik_model.joint_limit_upper, 10.0))
        body_q = home.body_q.numpy()
        for b, side in zip(self.elbows, (-1.0, 1.0), strict=True):
            target = wp.array([body_q[b][:3] + np.array((0.0, side * ELBOW_OUT, 0.0))] * n, dtype=wp.vec3)
            objectives.append(ik.IKObjectivePosition(b, wp.vec3(0.0), target, weight=ELBOW_WEIGHT))
        self.ik = ik.IKSolver(ik_model, n, objectives, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self.ik_graph = None  # (iterations, graph) of the IK step
        self.noise = ArmNoise(ik_model, self.fingers, n, noise)
        self.plan_first_fold()
        self.model.joint_q.assign(self.key_pose(0.0, iterations=100, noise=False))

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.joint_qd = wp.zeros_like(self.state_0.joint_qd)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        if self.fully_coupled:
            self.init_fully_coupled()
        else:
            self.init_kinematic_arms()
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=THICKNESS)
        self.contacts = self.pipeline.contacts()
        self.gravity = self.model.gravity.numpy()
        # per world and hand: the squeezed or held towel, see update_grasp
        self.held = [[None, None] for _ in range(self.world_count)]
        self.first_fold_checked = False
        # The fingers' collision boxes: body, transform in it and half extents.
        m, bodies = self.model, [h + k for h in self.hands for k in (1, 2)]
        boxes = np.isin(m.shape_body.numpy(), bodies) & (m.shape_type.numpy() == newton.GeoType.BOX)
        shapes = zip(*(a.numpy()[boxes] for a in (m.shape_body, m.shape_transform, m.shape_scale)), strict=True)
        self.finger_boxes = [(body, wp.transform(*X), half) for body, X, half in shapes]

        self.viewer.set_model(self.model)
        if self.world_count == 1:
            self.viewer.set_camera(pos=wp.vec3(0.45, -0.95, 0.75), pitch=-38.0, yaw=137.0)
        else:
            view_worlds(self.viewer, self.model, spacing, (-0.2, -0.5, 0.0), 0.8, pitch=-60.0, yaw=-90.0)

    def init_kinematic_arms(self):
        """IK poses the arms and Featherstone moves them kinematically; SolverFBA steps the towel with the lite NSN
        and holds the pinched particles."""
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.solver = SolverFBA(
            self.model,
            iterations=5,
            stretching_model="arap",
            nsn_schur_mode="lite",
            nsn_linear_iterations=10,
            integrate_with_external_rigid_solver=True,
            # Bounds a contact row at 1 N, five times what a landing particle needs.
            lambda_cap=1.0,
            particle_enable_self_contact=True,
            particle_self_contact_margin=SELF_MARGIN,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=SELF_MARGIN,
            # Light mass damping [1/s], as air drag gives: the edge between the hands no longer bounces as they carry
            # the towel over. Stiffness damping, which also damps turning elements, would hold the fold back.
            rayleigh_alpha=4.0,
        )

    def init_fully_coupled(self):
        """SolverFBA integrates the arms and the towel; the joint drives follow the IK targets, apart from the measured
        joints. Contacts on integrated bodies need the full operator."""
        self.target = self.model.joint_q.numpy().astype(float)
        finger = np.zeros(self.model.joint_dof_count, dtype=bool)
        finger[[f + k for f in self.fingers for k in (0, 1)]] = True
        limit = np.where(finger, GRIP_FORCE, self.model.joint_effort_limit.numpy())
        self.drive = JointDrive(self.model, np.where(finger, FINGER_FREQUENCY, ARM_FREQUENCY), effort_limit=limit)
        self.solver = SolverFBA(
            self.model,
            iterations=5,
            stretching_model="arap",
            nsn_schur_mode="full",
            nsn_linear_iterations=10,
            particle_enable_self_contact=True,
            particle_self_contact_margin=SELF_MARGIN,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=SELF_MARGIN,
            rayleigh_alpha=4.0,
        )

    # Plan

    def add_fold(self, world, t0, grips, sets, height, layers, axis):
        """Keys of world ``world``'s next fold from ``t0``: down onto the ``grips`` (x, y) of both hands with the pad middles at
        ``height``, pinch, a half circle along ``axis`` (0: x, 1: y) towards the ``sets``, laid ``layers`` layers
        higher, hold, let the ridge out, rise. Returns the fold's end time."""
        keys, arcs, dirs = self.keys[world], self.arcs[world], self.dirs[world]
        fold = FOLDS[len(arcs)]
        grips, sets = np.asarray(grips, dtype=float), np.asarray(sets, dtype=float)
        height = np.broadcast_to(np.asarray(height, dtype=float), (2,))
        lay = height + layers * SELF_MARGIN + (LAY_CLEAR if layers > 1 else 0.0)
        # the way the fold carries the corners, which sets the way the held towel turns over
        dirs.append(np.array([*np.sign(sets - grips)[0] * np.eye(2)[axis], 0.0]))

        def key(t, xy, z, q):
            keys.append((t, [np.array((*p, z_p)) for p, z_p in zip(xy, z, strict=True)], q, fold.yaw))

        t = t0 + DOWN
        key(t, grips, height, fold.grip_open)
        t += PINCH
        key(t, grips, height, self.closed)
        arcs.append((t, t + ARC))
        # A half circle about the fold line midway between grips and sets, whose radius shrinks by the gathered
        # towel over its first 45 degrees, as a half circle as wide as the towel would pull the fold line along, and
        # grows back over its last 45, so that the hands lay the corners on the sets.
        mid, radius = 0.5 * (grips + sets), 0.5 * (grips - sets)
        for theta in np.radians(np.arange(15.0, 181.0, 15.0)):
            ramp = min(theta / (0.25 * np.pi), 1.0, (np.pi - theta) / (0.25 * np.pi))
            r = radius - fold.gathered * np.sign(radius) * ramp
            xy = mid + r * np.cos(theta) * np.eye(2)[axis]
            xy[:, 1 - axis] = grips[:, 1 - axis]
            z = height + np.abs(r[:, axis]) * np.sin(theta)
            key(t + ARC * theta / np.pi, xy, np.maximum(z, lay + CLEAR) if theta > 0.5 * np.pi else z, self.closed)
        t += ARC
        sets = [p[:2] for p in keys[-1][1]]
        t += HOLD
        key(t, sets, lay, self.closed)
        t += LET_OUT
        key(t, sets, lay, fold.grip_open)
        t += RISE
        key(t, sets, lay + 0.1, OPEN)
        return t

    def plan_first_fold(self):
        """Fold the far half over the near half: the far corners onto the near ones."""
        along, across = self.inset
        grips = [(FAR - across, y + side * along) for y, side in zip(SIDES, (1.0, -1.0), strict=True)]
        sets = [(NEAR + across, y) for _, y in grips]
        # per world: the key poses (time, pad middles, finger position, yaw), the half circles' times and the folds'
        # directions; the fold under way is the same in all worlds
        n = self.world_count
        self.keys, self.arcs, self.dirs, self.fold = (
            [[] for _ in range(n)],
            [[] for _ in range(n)],
            [[] for _ in range(n)],
            0,
        )
        grasp = GRASP - self.press
        for w in range(n):
            self.keys[w].append((0.0, [np.array((*g, grasp + 0.1)) for g in grips], FOLDS[0].grip_open, FOLDS[0].yaw))
            self.add_fold(w, 0.0, grips, sets, grasp, 1, axis=0)

    def plan_second_fold(self):
        """Fold the side SIDES[1] of the folded towel over the side SIDES[0], from the corners where the first fold left
        them: hand 0 takes the corner at the near edge, hand 1 the corner at the fold. The arms then draw back over
        the far corners."""
        X0, X = self.model.particle_q.numpy(), self.state_0.particle_q.numpy()
        self.fold = 1
        for w in range(self.world_count):
            x0, x = (a[w * self.particles : (w + 1) * self.particles] for a in (X0, X))
            side = np.abs(x0[:, 1] - SIDES[1]) < 0.5 * CELL
            near, far = (np.abs(x0[:, 0] - e) < 0.5 * CELL for e in (NEAR, FAR))
            corner = 0.5 * (x[side & near][0] + x[side & far][0])
            fold = x[side][np.argmax(x[side][:, 0])]
            along, across = self.inset
            grips = [(corner[0] + along, corner[1] - across), (fold[0] - along, fold[1] - across)]
            # The pads come down over the folded towel's top layer around each grip.
            height = []
            for g in grips:
                under = np.linalg.norm(x[:, :2] - g, axis=1) < PAD_HALF_WIDTH + PAD_THICKNESS
                height.append(np.percentile(x[under, 2], 90) + OVER - self.press)
            height = np.array(height)
            sets = [(g[0], SIDES[0] + across) for g in grips]
            keys = self.keys[w]
            above = [np.array((*g, h + 0.1)) for g, h in zip(grips, height, strict=True)]
            keys.append((FOLD2 + DOWN, above, FOLDS[1].grip_open, FOLDS[1].yaw))
            end = self.add_fold(w, FOLD2 + DOWN, grips, sets, height, 2, axis=1) + SETTLE
            keys.append((end, [p + np.array((0.0, 0.0, 0.1)) for p in keys[0][1]], OPEN, FOLDS[0].yaw))

    def key_pose(self, t, iterations=24, noise=True):
        """Joint positions of both arms of every world at time ``t`` of their key poses, with the worlds' action
        noise unless ``noise`` is false, shape [world_count * joint_coord_count of a world]."""
        fingers = []
        for w, keys in enumerate(self.keys):
            times = [k[0] for k in keys]
            fingers.append(np.interp(t, times, [k[2] for k in keys]))
            yaw = np.radians(np.interp(t, times, [k[3] for k in keys]))
            down = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw)) * wp.quat(1.0, 0.0, 0.0, 0.0)
            for hand, (pads, turn) in enumerate(zip(self.pads, self.turns, strict=True)):
                p = [np.interp(t, times, [k[1][hand][i] for k in keys]) for i in range(3)]
                pads.set_target_position(w, wp.vec3(*p))
                turn.set_target_rotation(w, wp.vec4(*down))
        if self.ik_q.device.is_cuda:
            if self.ik_graph is None or self.ik_graph[0] != iterations:
                with wp.ScopedCapture(device=self.ik_q.device) as capture:
                    self.ik.step(self.ik_q, self.ik_q, iterations=iterations)
                self.ik_graph = (iterations, capture.graph)
            wp.capture_launch(self.ik_graph[1])
        else:
            self.ik.step(self.ik_q, self.ik_q, iterations=iterations)
        # a copy: the noise stays out of the IK's own state
        q = self.ik_q.numpy().copy()
        for f in self.fingers:
            q[:, f] = q[:, f + 1] = fingers
        return (self.noise(q) if noise else q).reshape(-1)

    def finger_command(self, world, t):
        """Commanded finger position of world ``world`` at time ``t``."""
        keys = self.keys[world]
        return np.interp(t, [k[0] for k in keys], [k[2] for k in keys])

    def fold_angle(self, world, fold, t):
        """How far fold ``fold`` has turned world ``world``'s towel over at time ``t`` [rad]."""
        a0, a1 = self.arcs[world][fold]
        return np.pi * float(np.clip((t - a0) / (a1 - a0), 0.0, 1.0))

    # Grasp

    def pad_frame(self, body_q, world, hand):
        """Middle of the fingertip pads of a hand of world ``world`` and their axes, from the body poses ``body_q``."""
        X = wp.transform(*body_q[self.hands[hand] + world * self.bodies])
        R = np.array(wp.quat_to_matrix(X.q)).reshape(3, 3)
        return np.array(X.p) + R @ np.array([0.0, 0.0, PAD]), R @ PAD_AXES

    def update_grasp(self):
        """Hold the towel each hand has taken at the targets of its phase: squeezed while the fingers close, held
        while they are closed, laid out flat while they open, and let go once they stand open."""
        q, q0, rows = self.state_1.joint_q.numpy(), self.state_0.joint_q.numpy(), []
        body_q, body_q0 = self.state_1.body_q.numpy(), self.state_0.body_q.numpy()
        t = self.sim_time + self.sim_dt
        grip_open = FOLDS[self.fold].grip_open
        for w in range(self.world_count):
            # The phase follows the fingers' commanded motion; the squeeze follows their actual gap.
            q_now, q_next = self.finger_command(w, self.sim_time), self.finger_command(w, t)
            closing, opening = q_next < q_now, q_next > q_now
            for hand, finger in enumerate(self.fingers):
                f = finger + w * self.coords
                origin, axes = self.pad_frame(body_q, w, hand)
                gap, held = q[f], self.held[w][hand]
                if held is not None and held["phase"] == "let_out" and gap >= grip_open - 5.0e-4:
                    held = None
                if held is not None and held["phase"] == "hold" and opening:
                    held = dict(held, phase="let_out", local=self.held_local(held, t))
                if held is None and closing and gap < grip_open - 5.0e-4:
                    # the towel the pads take is found where they stood before this substep
                    held = self.start_squeeze(w, *self.pad_frame(body_q0, w, hand), q0[f])
                if held is not None and held["phase"] == "squeeze" and not closing:
                    held = self.start_hold(held, gap, axes)
                self.held[w][hand] = held
                if held is None or not len(held["taken"]):
                    continue
                if held["phase"] == "squeeze":
                    local = self.squeezed(held, gap)
                elif held["phase"] == "hold":
                    local = self.held_local(held, t)
                else:
                    local = self.laid_out(held, gap)
                # the world's particles, by their index in the model
                rows.append((held["taken"] + w * self.particles, origin + local @ axes.T))
        if not rows:
            self.solver.set_particle_constraints(None, None, None)
            return
        p, x = (np.concatenate(a) for a in zip(*rows, strict=True))
        eye = np.tile(np.eye(3), (len(p), 1))
        self.solver.set_particle_constraints(
            np.repeat(p, 3)[:, None], np.ones((3 * len(p), 1)), x.reshape(-1), directions=eye
        )

    def start_squeeze(self, world, origin, axes, gap):
        """The towel of world ``world`` that a hand's pads squeeze, every layer down to the table, in the pads' frame:
        under and between the pads and the row beside their inner ends, nearest their middle first, from each layer in
        turn. Its particles are numbered within the world."""
        x = self.state_0.particle_q[world * self.particles : (world + 1) * self.particles].numpy()
        local = (x - origin) @ axes
        depth = local[:, 2] - PAD_HALF_LENGTH  # below the pads' lower edges
        strip = np.abs(local[:, 0]) <= gap + PAD_THICKNESS
        strip &= np.abs(local[:, 1]) <= PAD_HALF_WIDTH + CELL
        candidates = np.flatnonzero(strip & (depth >= -CLOTH_RADIUS) & (depth <= 2.5 * SELF_MARGIN))
        layer = np.full(len(x), -1)
        layer[candidates] = np.rint((depth[candidates] - depth[candidates].min(initial=0.0)) / SELF_MARGIN)
        order = candidates[np.argsort(np.linalg.norm(local[candidates, :2], axis=1))]
        queues = [list(order[layer[order] == k]) for k in range(layer.max(initial=-1) + 1)]
        taken = []
        while len(taken) < SQUEEZED and any(queues):
            for queue in queues:
                if queue and len(taken) < SQUEEZED:
                    taken.append(queue.pop(0))
        taken = np.array(taken, dtype=int)
        return {
            "phase": "squeeze",
            "world": world,
            "taken": taken,
            "start": local[taken],
            "gap": gap,
            "fold": self.fold,
            "layer": layer[taken],
        }

    @staticmethod
    def squeezed(held, gap):
        """Pad-frame points of a squeezed strip at the pads' gap: under a pad with it, between the pads pressed
        together and raised into a ridge whose two halves keep their length, as a tent's."""
        u, g0 = held["start"][:, 0], held["gap"]
        between = np.abs(u) < g0
        along = np.where(between, u * gap / g0, np.sign(u) * (np.abs(u) - g0 + gap))
        rise = np.sqrt(max(g0 * g0 - gap * gap, 0.0)) * np.cos(0.5 * np.pi * along / gap) ** 2
        local = held["start"].copy()
        local[:, 0] = along
        local[:, 2] -= np.where(between, rise, 0.0)
        return local

    def start_hold(self, squeeze, gap, axes):
        """What the closed pads hold of the squeezed towel: the towel between them, and beside each of its rows along
        the closing axis the next particle under a pad, which the towel wraps around the pad's lower edge, at its
        contact distance there. Each column of it, every layer that started at one place along the closing axis,
        turns about the middle of its part between the pads' ends: there the layers trade places as the fold turns
        them over, and the towel beyond the pads' inner ends turns around them, down onto the layers the fold lays
        it on."""
        local, layer, start = self.squeezed(squeeze, gap), squeeze["layer"], squeeze["start"]
        slot = np.abs(local[:, 0]) < gap
        wrap = np.zeros(len(local), dtype=bool)
        for k in np.flatnonzero(slot):
            row = ~slot & (layer == layer[k]) & (np.abs(local[:, 1] - local[k, 1]) < 0.5 * CELL)
            for side in (1.0, -1.0):
                beside = np.flatnonzero(row & (side * (local[:, 0] - local[k, 0]) > 0.0))
                if len(beside):
                    wrap[beside[np.argmin(np.abs(local[beside, 0] - local[k, 0]))]] = True
        local[wrap, 2] = np.maximum(local[wrap, 2], UNDER)
        keep = slot | wrap
        taken, local, layer, start = squeeze["taken"][keep], local[keep], layer[keep], start[keep]
        between = np.abs(local[:, 1]) <= PAD_HALF_WIDTH  # between the pads' ends
        middle, drop = local.copy(), np.zeros(len(taken))
        layers = layer.max(initial=0) + 1
        for i in range(len(taken)):
            column = (np.abs(start[:, 0] - start[i, 0]) < 0.5 * CELL) & between
            if column.any():
                middle[i] = local[column].mean(axis=0)
            if not between[i]:
                # turned half around its middle, it drops by this onto its layer's place, see laid_out
                drop[i] = UNDER + (layers - 1 - layer[i]) * SELF_MARGIN - (2.0 * middle[i, 2] - local[i, 2])
        # the towel turns about up x the fold's direction
        sign = float(np.sign(np.cross((0.0, 0.0, 1.0), self.dirs[squeeze["world"]][squeeze["fold"]]) @ axes[:, 0]))
        return {
            "phase": "hold",
            "world": squeeze["world"],
            "taken": taken,
            "local": local,
            "middle": middle,
            "drop": drop,
            "sign": sign,
            "fold": squeeze["fold"],
            "layer": layer,
            "gap": gap,
        }

    def held_local(self, held, t):
        """Pad-frame targets of the held towel at time ``t``: each particle turned about its column's middle and the
        closing axis by the fold's angle, and the towel beyond the pads' ends dropped by that share of its drop."""
        theta = self.fold_angle(held["world"], held["fold"], t) * held["sign"]
        c, s = np.cos(theta), np.sin(theta)
        rel = held["local"] - held["middle"]
        turned = rel.copy()
        turned[:, 1], turned[:, 2] = c * rel[:, 1] - s * rel[:, 2], s * rel[:, 1] + c * rel[:, 2]
        turned[:, 2] += abs(theta) / np.pi * held["drop"]
        return held["middle"] + turned

    @staticmethod
    def laid_out(held, gap):
        """Pad-frame targets of the held towel as the pads open: spread along the closing axis with them and lowered
        onto the layers they press, flat, each layer at its place in the turned stack, once they stand open."""
        g0, g1 = held["gap"], FOLDS[held["fold"]].grip_open
        s = float(np.clip((gap - g0) / (g1 - g0), 0.0, 1.0))
        local = held["local"].copy()
        u = local[:, 0]
        local[:, 0] = np.where(np.abs(u) < g0, u * gap / g0, np.sign(u) * (np.abs(u) - g0 + gap))
        layers = held["layer"].max(initial=0) + 1
        # the fold turned the pinch over: its first layer now lies lowest
        flat = UNDER + (layers - 1 - held["layer"]) * SELF_MARGIN
        local[:, 2] += s * (flat - local[:, 2])
        return local

    # Step

    def step(self):
        if self.fold == 0 and self.sim_time >= FOLD2 - 1.0e-9:
            self.plan_second_fold()
        if self.fully_coupled:
            self.step_fully_coupled()
            return
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

    def step_fully_coupled(self):
        """SolverFBA steps the arms and the towel together; the drives follow the joint targets, interpolated over the
        frame, and fingers stopped on the towel squeeze it."""
        start, target = self.target, self.key_pose(self.sim_time + self.frame_dt)
        velocity = (target - start) / self.frame_dt
        grip_open = FOLDS[self.fold].grip_open
        closing = np.clip((grip_open - self.finger_command(0, self.sim_time)) / grip_open, 0.0, 1.0)
        for k in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            q, _ = self.drive.measure(self.state_0)
            squeeze = np.zeros_like(q)
            for f in self.fingers:
                squeeze[f : f + 2] = np.where(q[f : f + 2] > STOP, -SQUEEZE * closing, 0.0)
            s = (k + 1) / self.sim_substeps
            self.drive.apply(self.state_0, start + s * (target - start), velocity, self.sim_dt, squeeze)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
        self.target = target

    def test_post_step(self):
        X, body_q = self.state_0.particle_q.numpy(), self.state_0.body_q.numpy()
        speed = np.abs(self.state_0.particle_qd.numpy()).max()
        assert speed < 2.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        if self.fully_coupled:
            gap, turn = joint_residual(self.model, body_q)
            assert gap < 1.0e-3 and turn < 1.0e-2, (
                f"joints {gap * 1e3:.2f} mm, {turn:.4f} rad off at t={self.sim_time:.2f} s"
            )
            fingers = self.drive.q[[f + k for f in self.fingers for k in (0, 1)]]
            assert fingers.min() > -2.0e-3, f"fingers closed to {fingers.min() * 1e3:.1f} mm at t={self.sim_time:.2f} s"
        check_fold = not self.first_fold_checked and self.sim_time >= FOLD2 - 0.5 * self.frame_dt
        for w in range(self.world_count):
            at = f"in world {w} at t={self.sim_time:.2f} s"
            x = X[w * self.particles : (w + 1) * self.particles]
            free = np.ones(len(x), dtype=bool)
            for hand, held in enumerate(self.held[w]):
                if held is not None:
                    free[held["taken"]] = False
                    if held["phase"] == "hold" and len(held["taken"]):
                        origin, axes = self.pad_frame(body_q, w, hand)
                        off = np.abs(x[held["taken"]] - origin - self.held_local(held, self.sim_time) @ axes.T).max()
                        assert off < 1.0e-3, f"a held particle is {off:.4f} m off {at}"
            for body, X_bs, half in self.finger_boxes:
                X_b = wp.transform(*body_q[body + w * self.bodies]) * X_bs
                inside = np.abs((x[free] - X_b.p) @ np.array(wp.quat_to_matrix(X_b.q)).reshape(3, 3)) - half
                assert inside.max(axis=1).min() > 0.0, f"a towel particle is inside a finger {at}"
            if check_fold:
                # Before the second fold, the far half lies turned over the middle onto the near half.
                x0 = self.model.particle_q.numpy()[w * self.particles : (w + 1) * self.particles]
                far = x0[:, 0] > 0.5 * (NEAR + FAR)
                off = np.abs(x[far, 0] - (NEAR + FAR - x0[far, 0])).max()
                assert off < 0.01, f"the first fold leaves a particle {off * 1e3:.1f} mm off its place in world {w}"
        self.first_fold_checked |= check_fold

    def test_final(self):
        # In every world, the towel lies within the square a quarter its size, in four layers in their order: the
        # near half's quarter under the far half's, which the first fold laid on it, then the other two, which the
        # second fold turned over; nothing lies over the top layer.
        X0, X = self.model.particle_q.numpy(), self.state_0.particle_q.numpy()
        for w in range(self.world_count):
            x0, x = (a[w * self.particles : (w + 1) * self.particles] for a in (X0, X))
            lo, hi = np.array((NEAR, SIDES[0])), np.array((0.5 * (NEAR + FAR), -0.5))
            out = np.maximum(lo - x[:, :2], x[:, :2] - hi).max()
            assert out < 0.02, f"the folded towel reaches {out * 1e3:.0f} mm past its square in world {w}"
            far, side = x0[:, 0] > 0.5 * (NEAR + FAR), x0[:, 1] > -0.5
            heights = [np.median(x[m, 2]) for m in (~far & ~side, far & ~side, far & side, ~far & side)]
            assert (np.diff(heights) > 0.5 * SELF_MARGIN).all(), (
                f"the layers lie at {np.round(heights, 4)} m in world {w}"
            )
            assert x[:, 2].max() < heights[-1] + SELF_MARGIN, f"part of the towel lies over its top layer in world {w}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=990)
        parser.add_argument(
            "--fully-coupled",
            action="store_true",
            help="Integrate the arms in SolverFBA too, driven by force-limited joint drives, with friction-only pinches.",
        )
        add_world_args(parser)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
