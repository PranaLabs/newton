# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Shirt Fold FBA
#
# Two Franka Panda arms on either side of a table fold a T-shirt that drops
# onto it: each side in toward the middle, the arm on that side holding the
# shoulder while the other reaches across for the hem corner, then the hem
# up toward the collar. Closing fingers squeeze the cloth between them into a
# bulge and hold it. The shirt lands with a 12 mm self-contact margin, which
# thins to 6 mm for the folds; edge-edge self contacts keep its creases
# apart, friction anchors keep the folds from creeping open, and Rayleigh
# damping of the stretch rate settles its ringing after the landing and
# each release. IK poses the arms and Featherstone moves them kinematically;
# SolverFBA steps the shirt with the lite contact operator, accumulating its
# contact forces over each step's iterations, and holds the grasped cloth.
#
# With --fully-coupled, an experiment: SolverFBA integrates the arms as
# well, with their joints as constraints and their contacts with the shirt
# in the same frictional solve with the full operator, so the shirt acts
# back on them. Stable PD drives with gravity compensation and force
# limits follow the IK targets, and the fingers pinch the shirt by friction
# alone, with no held particles. The arms' shapes still do not collide with
# the table or each other. The full operator keeps a dense block over the
# contacted particles: at the default 6480 particles it ran out of memory,
# dense and sparse, so the experiment runs on a coarser shirt, a resolution
# tradeoff whose fold is not known to succeed.
#
# With --world-count N, N copies of the arms, table and shirt fold in
# parallel worlds, stepped by one Featherstone and one SolverFBA, each
# world with its own IK problem, measured heights and grasps; --action-noise
# biases each world's arm joint targets by a fixed amount, so that the
# worlds differ.
#
# Command: python -m newton.examples shirt_fold_fba
# Parallel worlds: python -m newton.examples shirt_fold_fba --world-count 4 --action-noise 0.0001
# Coarser shirt: python -m newton.examples shirt_fold_fba --cloth-edge-length 0.0222
# Fully coupled: python -m newton.examples shirt_fold_fba --fully-coupled --cloth-edge-length 0.0222
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
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

TABLE_TOP = 0.2  # [m]
# The shirt's particle radius, its contact thickness against the finger pads, and the margin of the table and
# the arm bases, which keep it RADIUS apart [m].
CLOTH_RADIUS, SHAPE_MARGIN = 0.002, 0.006
RADIUS = CLOTH_RADIUS + SHAPE_MARGIN
# Self-contact margin while the shirt lands, which its falling layers cannot cross in a step, and once it lies
# still [m], reached over THIN_RAMP [s] from THIN_TIME [s] on: thinned at once, the stacked layers drop together.
LAND_MARGIN, FOLD_MARGIN, THIN_TIME, THIN_RAMP = 0.012, 0.006, 4.5, 1.0
OPEN, CLOSED = 0.04, 0.008  # finger joint positions [m]: the pads stand twice that apart
# From the last link to the middle of the fingertip pads, and their half width and half length [m]; their axes in
# the last link's frame, as columns: closing, spanning the pads and out of the fingertips.
PAD, PAD_HALF_WIDTH, PAD_HALF_LENGTH = 0.21065, 0.00875, 0.00925
PAD_AXES = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)]]) / np.sqrt(2.0)
# While the fingers close, the strip of the shirt's layers under and between the pads, at most SQUEEZED particles per
# hand, keeps its place along the closing axis relative to the pads, as static friction under the fingertips would:
# the cloth under a pad moves with it and the cloth between them shortens with the gap, so that it buckles up.
# Closed, the hand holds that bulge, at most MAX_PINCHED particles between the pads from BULGE_HEIGHT over their
# middle down to the cloth under them: three rows each, which with the strip's rows stay within the lite operator's
# 64 exact rows. PAD_THICKNESS is a pad's extent along the closing axis [m].
MAX_PINCHED, SQUEEZED, BULGE_HEIGHT, PAD_THICKNESS = 10, 32, 0.04, 0.0152
GRASP_PAUSE = 0.5  # the closed hands' pause before they turn the fold [s]
# Closed, the pads hold the cloth between them down to HOLD_DEPTH under their lower edges [m]: the bulge and, for each
# layer of the squeezed strip that has none in it, one particle, at most HOLD_LOWER of them, so that the lower layers
# turn with the fold instead of staying behind while the bulge, most of the pinch, stays between the pads.
HOLD_DEPTH, HOLD_LOWER = 0.004, 3
# While a hand grasps, the shirt within ANCHOR_FREE of its pads keeps no friction anchors, so that it can buckle and
# slide toward the pads [m].
ANCHOR_FREE = 0.05
HOVER = 0.08  # approach height over a grip [m]
BASES = ((0.48, -0.35), (-0.48, -0.35))  # the arms' bases beside the table, facing it [m]
# The arms' joints at rest, Franka's ready pose, and where it holds the pads: ahead of the base and high [m].
READY = [0.0, -0.25 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.5 * np.pi, 0.25 * np.pi]
REST_AHEAD, REST_HEIGHT = 0.307, 0.487
# The +x side fold: shoulder and hem-corner grips and the fold line at x = SIDE_LINE; the -x side mirrors it. The
# hem fold: grips at the hem corners of the folded body and the fold line at y = HEM_LINE [m].
SHOULDER, HEM_CORNER, SIDE_LINE = (0.24, -0.70), (0.21, -0.215), 0.12
HEM_GRIP, HEM_LINE = (0.06, -0.20), -0.48
# The hem fold's arc height as a share of its radius, and its duration [s].
HEM_LIFT, HEM_ARC = 1.0, 4.0
# With --fully-coupled: the rubber finger pads grip the shirt with a friction coefficient over the table's, and the
# arms press them PRESS [m] into their contact distance; the fingers close to 0 where the plan closes them, and
# stopped on the shirt over STOP [m], squeeze it with SQUEEZE [N] more as their command closes, within GRIP_FORCE [N].
# The drives' natural frequencies [rad/s] keep stable at four substeps.
PAD_MU, PRESS, STOP, SQUEEZE, GRIP_FORCE = 2.0, 0.002, 0.0015, 15.0, 20.0
ARM_FREQUENCY, FINGER_FREQUENCY = 30.0, 450.0


def rotation(axis, theta):
    """Rotation matrix by ``theta`` about ``axis``."""
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * K @ K


def pick_layers(candidates, depth, off_middle, limit=MAX_PINCHED):
    """Up to ``limit`` of the candidates, taken in turn from each layer, nearest the middle first, so that a pinch
    holds every layer."""
    if not len(candidates):
        return candidates
    layer = np.rint((depth - depth.min()) / FOLD_MARGIN).astype(int)
    queues = [list(candidates[layer == k][np.argsort(off_middle[layer == k])]) for k in np.unique(layer)]
    taken = []
    while len(taken) < limit and any(queues):
        for queue in queues:
            if queue and len(taken) < limit:
                taken.append(queue.pop(0))
    return np.array(taken, dtype=int)


class Plan:
    """Key poses of both hands, per hand a list of (t, x, y, ref, dz, finger, theta).

    The pad middle is at (x, y, z), with z = dz when ref is None and otherwise dz over a height measured on the shirt
    when first needed: for a (fold, hand) pair the grip height, and for ("land", fold, hand) the height that lays the
    turned pinch on the shirt under the hand's last point. Theta is the angle the pinch has turned about its fold line.
    """

    def __init__(self, rest):
        self.t = 0.0
        self.keys = [[(0.0, x, y, None, z, OPEN, 0.0)] for x, y, z in rest]
        self.folds = []

    def to(self, dt, *targets):
        """Move each hand to its target (x, y, ref, dz, finger, theta) in ``dt``; a None target holds the hand."""
        self.t += dt
        for keys, target in zip(self.keys, targets, strict=True):
            keys.append((self.t, *(keys[-1][1:] if target is None else target)))

    def above(self, hand, grip):
        """Target over ``grip`` for the next fold."""
        return (grip[0], grip[1], (len(self.folds), hand), HOVER, OPEN, 0.0)

    def fold(self, grips, axis, line, *, arc=3.0, approach=2.0, steps=12, lift=1.0):
        """Pinch both grips, turn them half a circle about the fold line (x = line for axis 0, y = line for axis 1),
        lay them down and let go."""
        i = len(self.folds)
        refs = [(i, h) for h in range(2)]
        axes = []
        for g in grips:
            d = np.zeros(3)
            d[axis] = np.sign(g[axis] - line)
            axes.append(np.cross(d, [0.0, 0.0, 1.0]))  # turns d toward +z
        info = {"grips": grips, "axes": axes, "t0": self.t, "laid": []}
        self.to(approach, *(self.above(h, g) for h, g in enumerate(grips)))
        self.to(1.0, *((g[0], g[1], r, 0.0, OPEN, 0.0) for g, r in zip(grips, refs, strict=True)))
        self.to(0.5, *((g[0], g[1], r, 0.0, CLOSED, 0.0) for g, r in zip(grips, refs, strict=True)))
        self.to(GRASP_PAUSE, None, None)
        for k in range(1, steps + 1):
            theta = np.pi * k / steps
            hands = []
            for h, (g, r) in enumerate(zip(grips, refs, strict=True)):
                radius = g[axis] - line
                p = list(g)
                p[axis] = line + radius * np.cos(theta)
                if k < steps:
                    hands.append((p[0], p[1], r, lift * abs(radius) * np.sin(theta), CLOSED, theta))
                else:
                    info["laid"].append((p[0], p[1]))
                    hands.append((p[0], p[1], ("land", i, h), 0.0, CLOSED, theta))
            self.to(arc / steps, *hands)
        self.to(0.5, None, None)
        self.to(0.5, *((*k[-1][1:5], OPEN, k[-1][6]) for k in self.keys))
        info["t1"] = self.t
        self.to(1.0, *((*k[-1][1:4], k[-1][4] + HOVER, OPEN, k[-1][6]) for k in self.keys))
        self.folds.append(info)

    def sample(self, hand, t):
        """The key poses around time ``t`` and the fraction of the way from the first to the second."""
        keys = self.keys[hand]
        times = [k[0] for k in keys]
        j = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(keys) - 2))
        s = float(np.clip((t - times[j]) / max(times[j + 1] - times[j], 1e-9), 0.0, 1.0))
        return keys[j], keys[j + 1], s

    def fold_at(self, t):
        """The fold whose approach began last by time ``t``, or None."""
        started = [i for i, f in enumerate(self.folds) if f["t0"] <= t]
        return started[-1] if started else None


def shop_fold():
    """Key poses of the shop fold: the sides in, then the hem up. Arm 0 stands at +x, arm 1 at -x."""
    rest = [(b[0] - REST_AHEAD * np.sign(b[0]), b[1], REST_HEIGHT) for b in BASES]
    plan = Plan(rest)
    plan.to(5.0, None, None)  # the shirt lands
    # +x side: arm 0 at its shoulder, arm 1 across at the hem corner
    plan.fold([SHOULDER, HEM_CORNER], 0, SIDE_LINE)
    # -x side: arm 1 to its shoulder first, then arm 0 across to the hem corner, so that the arms never cross
    grips = [(-HEM_CORNER[0], HEM_CORNER[1]), (-SHOULDER[0], SHOULDER[1])]
    plan.to(1.5, None, plan.above(1, grips[1]))
    plan.to(1.5, plan.above(0, grips[0]), None)
    plan.fold(grips, 0, -SIDE_LINE, approach=0.5)
    # the hem up, both hands at the hem corners of the folded body
    plan.fold([HEM_GRIP, (-HEM_GRIP[0], HEM_GRIP[1])], 1, HEM_LINE, arc=HEM_ARC, lift=HEM_LIFT)
    # back to the ready pose in joint space
    plan.home = (plan.t, plan.t + 2.0)
    plan.to(2.0, *((x, y, None, z, OPEN, 0.0) for x, y, z in rest))
    plan.to(1.0, None, None)
    return plan


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fully_coupled = getattr(args, "fully_coupled", False)
        edge_length = float(getattr(args, "cloth_edge_length", 0.0111))
        if not np.isfinite(edge_length) or edge_length <= 0.0:
            raise ValueError(f"cloth_edge_length must be a positive length [m], got {edge_length}")
        self.world_count, noise, spacing = world_args(args)
        if self.fully_coupled and self.world_count > 1:
            raise ValueError("--fully-coupled runs a single world")
        # Two substeps of 1/120 s: the lite operator takes a particle's mass alone for its response, which overstates
        # it more the longer the step; accumulating the contact forces over a step's iterations keeps the 6 mm layers
        # apart at this step.
        # The fully coupled drives push the arms with the forces of each step's start, which four substeps keep stable.
        self.frame_dt, self.sim_substeps = 1.0 / 60.0, 4 if self.fully_coupled else 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.plan = shop_fold()

        def new_builder():
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.mu = 1.5
            builder.default_shape_cfg.margin = SHAPE_MARGIN
            return builder

        # One world's arms, table and shirt; parallel worlds replicate it in place and share the ground.
        builder = template = new_builder()
        urdf = str(newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf")
        options = {
            "floating": False,
            "enable_self_collisions": False,
            "collapse_fixed_joints": True,
            "joint_damping_mode": "passive",
        }
        self.hands, self.fingers = [], []
        for x, y in BASES:
            face = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi if x > 0.0 else 0.0)
            template.add_urdf(urdf, xform=wp.transform((x, y, 0.0), face), **options)
            template.joint_q[-9:] = [*READY, OPEN, OPEN]
            self.hands.append(template.body_count - 3)
            self.fingers.append(template.joint_coord_count - 2)
        table = wp.transform((0.0, -0.5, 0.5 * TABLE_TOP), wp.quat_identity())
        template.add_shape_box(-1, xform=table, hx=0.4, hy=0.4, hz=0.5 * TABLE_TOP, color=(0.5,) * 3, label="table")
        # The shirt of example_cloth_franka in meters: the asset, in centimeters, remeshed to even edges of
        # edge_length, 1.1 cm by default, dropped from 22 cm
        # over the table. Its rest pose is folded, so its bending holds no rest angle.
        stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        mesh = newton.utils.remesh_mesh(
            newton.usd.get_mesh(stage.GetPrimAtPath("/root/shirt")), method="isotropic", edge_length=100.0 * edge_length
        )
        template.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.70, 0.42),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            scale=0.01,
            vel=wp.vec3(0.0),
            vertices=[wp.vec3(v) for v in mesh.vertices],
            indices=mesh.indices,
            density=200.0,
            tri_ke=5000.0,
            tri_ka=1.0e4,
            edge_ke=1.5e-3,
            particle_radius=CLOTH_RADIUS,
        )
        # strides of a world's bodies, joint coordinates and particles
        self.bodies, self.coords = template.body_count, template.joint_coord_count
        self.particles = template.particle_count
        if self.world_count > 1:
            builder = new_builder()
            builder.replicate(template, self.world_count)
        builder.add_ground_plane()
        # The arms follow the IK and only the shirt's contacts are read: the shapes do not collide with each other.
        for k in range(builder.shape_count):
            builder.shape_flags[k] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.model = builder.finalize()
        # The finger pads touch the shirt: no margin on the fingers' shapes.
        margin, body = self.model.shape_margin.numpy(), self.model.shape_body.numpy()
        for s, b in enumerate(body):
            if b >= 0 and "finger" in str(self.model.body_label[b]):
                margin[s] = 0.0
        self.model.shape_margin.assign(margin)
        if self.fully_coupled:
            mu = self.model.shape_material_mu.numpy()
            mu[margin == 0.0] = PAD_MU
            self.model.shape_material_mu.assign(mu)
        self.model.edge_rest_angle.zero_()

        # IK places the middle of each hand's fingertip pads, pointing down with the fingers closing along y, so that
        # the hands' long sides face each other when they pass; each hand keeps the yaw it has in the ready pose. Each
        # world is one problem on one world's arms.
        n = self.world_count
        ik_model = self.model if n == 1 else template.finalize()
        self.ik_q = wp.array(np.tile(ik_model.joint_q.numpy(), (n, 1)), dtype=wp.float32)
        self.pads = [ik.IKObjectivePosition(h, wp.vec3(0.0, 0.0, PAD), wp.zeros(n, wp.vec3)) for h in self.hands]
        objectives = list(self.pads)
        for hand, (x, _) in zip(self.hands, BASES, strict=True):
            yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), (np.pi if x > 0.0 else 0.0) - 0.25 * np.pi)
            down = wp.mul(yaw, wp.quat(1.0, 0.0, 0.0, 0.0))
            objectives.append(
                ik.IKObjectiveRotation(hand, wp.quat_identity(), wp.array([wp.vec4(*down)] * n, dtype=wp.vec4))
            )
        objectives.append(ik.IKObjectiveJointLimit(ik_model.joint_limit_lower, ik_model.joint_limit_upper, 10.0))
        self.ik = ik.IKSolver(ik_model, n, objectives, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self.noise = ArmNoise(ik_model, self.fingers, n, noise)
        # per world: the measured heights of the key poses and the landings of the pinches, see height
        self.heights, self.landing = [{} for _ in range(n)], [{} for _ in range(n)]
        self.ready_q, self.home_q = ik_model.joint_q.numpy().copy(), None
        self.ik_graph = None  # (iterations, graph) of the IK step
        self.model.joint_q.assign(self.key_pose(0.0, iterations=100, noise=False))

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.joint_qd = wp.zeros_like(self.state_0.joint_qd)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        if self.fully_coupled:
            self.init_fully_coupled()
        else:
            self.init_kinematic_arms()
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=RADIUS)  # finds the table 2 RADIUS out
        self.contacts = self.pipeline.contacts()
        self.gravity = self.model.gravity.numpy()
        # Per world and hand: the grasp, dragging the strip under the closing pads or holding the bulge between the
        # closed ones, with its particles, numbered within the world, and their targets in the pad frame, or None.
        self.held = [[None, None] for _ in range(n)]
        self.pinched = []  # particles each pinch took, in any world
        self.rows = None  # the pinched particles of all worlds and their targets in the last substep

        self.viewer.set_model(self.model)
        if n == 1:
            self.viewer.set_camera(pos=wp.vec3(0.0, 0.2, 1.38), pitch=-59.0, yaw=-90.0)
        else:
            view_worlds(self.viewer, self.model, spacing, (0.0, -0.4, 0.0), 0.8, pitch=-60.0, yaw=-90.0)

    def init_kinematic_arms(self):
        """IK poses the arms and Featherstone moves them kinematically; SolverFBA steps the shirt with the lite contact
        operator and holds the grasped cloth."""
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.solver = SolverFBA(
            self.model,
            iterations=5,
            stretching_model="arap",
            nsn_schur_mode="lite",
            nsn_linear_iterations=10,
            # The layers that press on each other take the forces whose motion through the system matrix holds them
            # apart, rather than those that the particles' masses alone predict would.
            nsn_lite_accumulate=True,
            integrate_with_external_rigid_solver=True,
            # Bounds a contact row at 200 N: two self contacts of the landing layers reach it for a step.
            lambda_cap=200.0,
            # Stiffness damping [s] settles the few-hertz ringing of the heavy, soft shirt after its landing and each
            # release; it leaves translations, and so the fall, undamped.
            rayleigh_beta=3.0e-3,
            # on the stretch rate alone, so that a released flap falls as fast as without damping
            rayleigh_corotated=True,
            friction_anchors=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=LAND_MARGIN,
            particle_self_contact_edge_edge=True,
            # edge pairs as near as their particles' nearest triangles too, which the stacked folds bring together
            particle_self_contact_edge_edge_ratio=1.0,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=LAND_MARGIN,
        )

    def init_fully_coupled(self):
        """SolverFBA integrates the arms and the shirt; the joint drives follow the IK targets, apart from the measured
        joints. Contacts on integrated bodies need the full operator, which takes the place of the lite operator's
        accumulation."""
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
            # the dense operator of the landing shirt's contacts takes several gigabytes
            nsn_sparse_storage=True,
            nsn_linear_iterations=10,
            lambda_cap=200.0,
            rayleigh_beta=3.0e-3,
            rayleigh_corotated=True,
            friction_anchors=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=LAND_MARGIN,
            particle_self_contact_edge_edge=True,
            particle_self_contact_edge_edge_ratio=1.0,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=LAND_MARGIN,
        )

    def world_particles(self, world):
        """Positions of world ``world``'s shirt particles [m]."""
        return self.state_0.particle_q[world * self.particles : (world + 1) * self.particles].numpy()

    def top_near(self, world, xy):
        """Height of world ``world``'s highest shirt particle within 1.5 cm of ``xy`` [m]."""
        q = self.world_particles(world)
        near = np.linalg.norm(q[:, :2] - np.asarray(xy), axis=1) < 0.015
        return q[near, 2].max() if near.any() else TABLE_TOP + RADIUS

    def height(self, world, ref):
        """Pad-middle height of a measured key pose in world ``world`` [m]."""
        landing = self.landing[world]
        if ref[0] == "land":
            _, i, hand = ref
            if (i, hand) not in landing:  # nothing pinched: set the empty pads down as on the grip
                landing[(i, hand)] = (self.top_near(world, self.plan.folds[i]["laid"][hand]), PAD_HALF_LENGTH + 0.002)
            top, drop = landing[(i, hand)]
            # the turned pinch's lowest particle a self-contact margin and 3 mm over the shirt there
            return top + FOLD_MARGIN + 0.003 + drop
        i, hand = ref
        # the pads' lower edges 1 mm over the shirt's contact thickness against them, or pressed into it when coupled
        lift = -PRESS if self.fully_coupled else 0.001
        return self.top_near(world, self.plan.folds[i]["grips"][hand]) + CLOTH_RADIUS + lift + PAD_HALF_LENGTH

    def key_z(self, world, key):
        ref, heights = key[3], self.heights[world]
        if ref is None:
            return key[4]
        if ref not in heights:
            heights[ref] = self.height(world, ref)
        return heights[ref] + key[4]

    def hand_targets(self, world, t):
        """Each hand's pad-middle target, finger position and pinch angle in world ``world`` at time ``t``."""
        out = []
        for hand in range(2):
            a, b, s = self.plan.sample(hand, t)
            za, zb = self.key_z(world, a), self.key_z(world, b)
            x, y, finger, theta = (a[k] + s * (b[k] - a[k]) for k in (1, 2, 5, 6))
            out.append(((x, y, za + s * (zb - za)), finger, theta))
        return out

    def key_pose(self, t, iterations=8, noise=True):
        """Joint positions of both arms of every world at time ``t``, with the worlds' action noise unless ``noise`` is
        false, shape [world_count * joint_coord_count of a world]."""
        targets = [self.hand_targets(w, t) for w in range(self.world_count)]
        if t > self.plan.home[0]:  # back to the ready pose in joint space, easing in and out
            if self.home_q is None:
                self.home_q = self.ik_q.numpy().copy()
            s = min((t - self.plan.home[0]) / (self.plan.home[1] - self.plan.home[0]), 1.0)
            q = self.home_q + (0.5 - 0.5 * np.cos(np.pi * s)) * (self.ready_q - self.home_q)
            self.ik_q.assign(q)
            return (self.noise(q) if noise else q).reshape(-1)
        for w, world_targets in enumerate(targets):
            for pads, (p, _, _) in zip(self.pads, world_targets, strict=True):
                pads.set_target_position(w, wp.vec3(*p))
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
        for w, world_targets in enumerate(targets):
            for f, (_, finger, _) in zip(self.fingers, world_targets, strict=True):
                # coupled fingers close to 0 where the plan closes them, and stop on the shirt
                q[w, f : f + 2] = OPEN * (finger - CLOSED) / (OPEN - CLOSED) if self.fully_coupled else finger
        return (self.noise(q) if noise else q).reshape(-1)

    def pad_frame(self, body_q, world, hand):
        """Middle of the fingertip pads of a hand of world ``world`` and their axes, from the body poses ``body_q``."""
        X = wp.transform(*body_q[self.hands[hand] + world * self.bodies])
        R = np.array(wp.quat_to_matrix(X.q)).reshape(3, 3)
        return np.array(X.p) + R @ np.array([0.0, 0.0, PAD]), R @ PAD_AXES

    def start_drag(self, world, origin, axes, half_gap):
        """The strip of world ``world``'s shirt layers under and between the pads, with its place along the closing
        axis."""
        x = self.world_particles(world)
        local = (x - origin) @ axes
        depth = local[:, 2] - PAD_HALF_LENGTH  # below the pads' lower edges
        strip = (np.abs(local[:, 0]) <= half_gap + PAD_THICKNESS) & (np.abs(local[:, 1]) <= PAD_HALF_WIDTH)
        candidates = np.flatnonzero(strip & (depth >= 0.0) & (x[:, 2] > TABLE_TOP))
        taken = pick_layers(candidates, depth[candidates], np.abs(local[candidates, 1]), limit=SQUEEZED)
        # the layers of the column: a new one wherever the depths, in order, part by more than half a margin
        layer = np.zeros(len(taken), dtype=int)
        if len(taken):
            order = np.argsort(depth[taken])
            layer[order] = np.concatenate([[0], np.cumsum(np.diff(depth[taken][order]) > 0.5 * FOLD_MARGIN)])
        return {
            "phase": "drag",
            "taken": taken,
            "along": local[taken, 0],
            "gap": half_gap,
            "axis": axes[:, 0].copy(),
            "layer": layer,
        }

    def start_hold(self, world, hand, i, origin, axes, half_gap, drag=None):
        """Hold the bulge of world ``world``'s shirt that the closed pads squeezed up between them as it stands, with a
        particle of each layer of the strip that it leaves out, turning it with its fold."""
        fold = self.plan.folds[i]
        x = self.world_particles(world)
        local = (x - origin) @ axes
        between = (np.abs(local[:, 0]) <= half_gap + 0.002) & (np.abs(local[:, 1]) <= PAD_HALF_WIDTH + 0.004)
        between &= (local[:, 2] >= -BULGE_HEIGHT) & (local[:, 2] <= PAD_HALF_LENGTH + HOLD_DEPTH)
        between &= x[:, 2] > TABLE_TOP
        candidates = np.flatnonzero(between)
        bulge = candidates[np.argsort(np.linalg.norm(local[candidates], axis=1))]
        lower = []
        if drag is not None and len(drag["taken"]):
            # the strip's layers that the nearest bulge particles leave out, each by its particle nearest the pads'
            # middle, in place of the farthest of those
            strip, layer = drag["taken"], drag["layer"]
            layer_of = dict(zip(strip.tolist(), layer.tolist(), strict=True))
            for _ in range(2):
                kept = bulge[: MAX_PINCHED - len(lower)]
                held_layers = {layer_of[p] for p in kept.tolist() + lower if p in layer_of}
                for k in np.unique(layer):
                    members = strip[layer == k]
                    if k not in held_layers and len(lower) < HOLD_LOWER:
                        lower.append(int(members[np.argmin(np.linalg.norm(local[members, :2], axis=1))]))
                        held_layers.add(k)
        lower = [p for p in lower if p not in set(bulge[: MAX_PINCHED - len(lower)].tolist())]
        taken = np.array(list(bulge[: MAX_PINCHED - len(lower)]) + lower, dtype=int)
        held = local[taken]
        middle = held.mean(axis=0) if len(taken) else np.zeros(3)
        # turned half a circle about its middle, the pinch's lowest particle lies this far under the pads
        drop = 2.0 * middle[2] - held[:, 2].min() if len(taken) else PAD_HALF_LENGTH + 0.002
        self.landing[world][(i, hand)] = (self.top_near(world, fold["laid"][hand]), drop)
        self.pinched.append(len(taken))
        return {"phase": "hold", "taken": taken, "local": held, "middle": middle, "axis": axes.T @ fold["axes"][hand]}

    def update_grasp(self):
        """While a hand's fingers close at a grip, its pads drag the layers under them in, so that the cloth between
        them buckles up; once they are closed, hold that bulge at the pads, turning it with the fold, until they
        open."""
        q, rows, drag_rows, free = self.state_1.joint_q.numpy(), [], [], []
        body_q, xy = self.state_1.body_q.numpy(), self.state_0.particle_q.numpy()[:, :2]
        i = self.plan.fold_at(self.sim_time)
        for w in range(self.world_count):
            targets = self.hand_targets(w, self.sim_time)
            first = w * self.particles  # of the world's particles in the model
            for hand, finger in enumerate(self.fingers):
                f = finger + w * self.coords
                origin, axes = self.pad_frame(body_q, w, hand)
                a, b, _ = self.plan.sample(hand, self.sim_time)
                closing = b[5] < a[5]
                held = self.held[w][hand]
                if held is not None and held["phase"] == "hold" and q[f] > 0.5 * (OPEN + CLOSED):
                    held = None
                if held is None and closing and i is not None and q[f] < OPEN - 5.0e-4:
                    held = self.start_drag(w, origin, axes, q[f])
                if held is not None and held["phase"] == "drag" and q[f] < CLOSED + 1.0e-3:
                    held = self.start_hold(w, hand, i, origin, axes, q[f], drag=held)
                self.held[w][hand] = held
                if held is None:
                    continue
                world_xy = xy[first : first + self.particles]
                free.append(first + np.flatnonzero(np.linalg.norm(world_xy - origin[:2], axis=1) < ANCHOR_FREE))
                if held["phase"] == "drag":
                    # under a pad: with it; between the pads: shortened with the gap
                    u, g0, g = held["along"], held["gap"], q[f]
                    along = np.where(np.abs(u) >= g0, np.sign(u) * (np.abs(u) - g0 + g), u * g / g0)
                    d = np.tile(held["axis"], (len(u), 1))
                    drag_rows.append((first + held["taken"], d, origin @ held["axis"] + along))
                else:
                    turn = rotation(held["axis"], targets[hand][2])
                    patch = held["middle"] + (held["local"] - held["middle"]) @ turn.T
                    rows.append((first + held["taken"], origin + patch @ axes.T))
        self.solver.set_friction_anchor_exclusion(np.concatenate(free) if free else None)
        rows = [r for r in rows if len(r[0])]
        drag_rows = [r for r in drag_rows if len(r[0])]
        if not rows and not drag_rows:
            self.rows = None
            self.solver.set_particle_constraints(None, None, None)
            return
        particles, directions, values = [], [], []
        if rows:
            p, x = (np.concatenate(a) for a in zip(*rows, strict=True))
            self.rows = (p, x)
            particles.append(np.repeat(p, 3))
            directions.append(np.tile(np.eye(3), (len(p), 1)))
            values.append(x.reshape(-1))
        else:
            self.rows = None
        for taken, d, along in drag_rows:
            particles.append(taken)
            directions.append(d)
            values.append(along)
        particles = np.concatenate(particles)
        # Every row takes the exact step, so that the dragged strip follows the pads too.
        directions = np.concatenate(directions)
        coefficients = np.ones((len(particles), 1))
        self.solver.set_particle_constraints(
            particles[:, None], coefficients, np.concatenate(values), directions=directions, exact=True
        )

    def step(self):
        thin = (self.sim_time - THIN_TIME) / THIN_RAMP
        if -0.5 * self.frame_dt <= thin * THIN_RAMP <= THIN_RAMP + 0.5 * self.frame_dt:
            margin = LAND_MARGIN + min(max(thin, 0.0), 1.0) * (FOLD_MARGIN - LAND_MARGIN)
            self.solver.set_particle_self_contact(margin=margin)
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
        """SolverFBA steps the arms and the shirt together; the drives follow the joint targets, interpolated over the
        frame, and fingers stopped on the shirt squeeze it."""
        start, target = self.target, self.key_pose(self.sim_time + self.frame_dt)
        velocity = (target - start) / self.frame_dt
        closing = [float(np.clip(1.0 - target[f] / OPEN, 0.0, 1.0)) for f in self.fingers]
        for k in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            q, _ = self.drive.measure(self.state_0)
            squeeze = np.zeros_like(q)
            for f, c in zip(self.fingers, closing, strict=True):
                squeeze[f : f + 2] = np.where(q[f : f + 2] > STOP, -SQUEEZE * c, 0.0)
            s = (k + 1) / self.sim_substeps
            self.drive.apply(self.state_0, start + s * (target - start), velocity, self.sim_dt, squeeze)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
        self.target = target

    def test_post_step(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), f"non-finite shirt positions at t={self.sim_time:.2f} s"
        speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max()
        assert speed < 5.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        # The table's contacts keep the particle radius between the shirt and the table top; laying the hem over the
        # stacked layers presses the lowest one up to 5.3 mm into it, and fingers that close on a stack, dragging
        # every layer, press the lowest one for a few frames up to its radius under the top.
        low = x[:, 2].min() - TABLE_TOP
        assert low > -CLOTH_RADIUS, f"a particle is {-low * 1e3:.1f} mm under the table top at t={self.sim_time:.2f} s"
        if self.fully_coupled:
            gap, turn = joint_residual(self.model, self.state_0.body_q.numpy())
            assert gap < 1.0e-3 and turn < 1.0e-2, (
                f"joints {gap * 1e3:.2f} mm, {turn:.4f} rad off at t={self.sim_time:.2f} s"
            )
            fingers = self.drive.q[[f + k for f in self.fingers for k in (0, 1)]]
            assert fingers.min() > -2.0e-3, f"fingers closed to {fingers.min() * 1e3:.1f} mm at t={self.sim_time:.2f} s"
        if self.rows is not None:
            off = np.linalg.norm(x[self.rows[0]] - self.rows[1], axis=1).max()
            assert off < 1.0e-3, f"a pinched particle is {off * 1e3:.2f} mm off at t={self.sim_time:.2f} s"

    def test_final(self):
        assert min(self.pinched, default=6) >= 6, f"the pinches took {self.pinched} particles"
        done = [self.sim_time >= f["t1"] + 1.0 for f in self.plan.folds]
        for w in range(self.world_count):
            x = self.world_particles(w)
            if done[0]:
                side = np.percentile(x[:, 0], 99)
                assert side < SIDE_LINE + 0.05, f"the +x side reaches x = {side:.3f} m after its fold in world {w}"
            if done[1]:
                side = np.percentile(x[:, 0], 1)
                assert side > -SIDE_LINE - 0.05, f"the -x side reaches x = {side:.3f} m after its fold in world {w}"
            if done[2]:
                hem = np.percentile(x[:, 1], 99)
                assert hem < HEM_LINE + 0.05, f"the hem half reaches y = {hem:.3f} m after its fold in world {w}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=2250)
        parser.add_argument(
            "--fully-coupled",
            action="store_true",
            help="Integrate the arms in SolverFBA too, driven by force-limited joint drives, with friction-only pinches.",
        )
        parser.add_argument(
            "--cloth-edge-length",
            type=float,
            default=0.0111,
            help="Edge length [m] the shirt is remeshed to, for both modes. The full contact operator of "
            "--fully-coupled runs out of memory at the default 6480 particles; 0.0222 gives a coarser shirt.",
        )
        add_world_args(parser)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
