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
# Command: python -m newton.examples shirt_fold_fba
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
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
        # Two substeps of 1/120 s: the lite operator takes a particle's mass alone for its response, which overstates
        # it more the longer the step; accumulating the contact forces over a step's iterations keeps the 6 mm layers
        # apart at this step.
        self.frame_dt, self.sim_substeps = 1.0 / 60.0, 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.plan = shop_fold()

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 1.5
        builder.default_shape_cfg.margin = SHAPE_MARGIN
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
            builder.add_urdf(urdf, xform=wp.transform((x, y, 0.0), face), **options)
            builder.joint_q[-9:] = [*READY, OPEN, OPEN]
            self.hands.append(builder.body_count - 3)
            self.fingers.append(builder.joint_coord_count - 2)
        table = wp.transform((0.0, -0.5, 0.5 * TABLE_TOP), wp.quat_identity())
        builder.add_shape_box(-1, xform=table, hx=0.4, hy=0.4, hz=0.5 * TABLE_TOP, color=(0.5,) * 3, label="table")
        # The shirt of example_cloth_franka in meters: the asset remeshed to even 1.1 cm edges, dropped from 22 cm
        # over the table. Its rest pose is folded, so its bending holds no rest angle.
        stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        mesh = newton.utils.remesh_mesh(
            newton.usd.get_mesh(stage.GetPrimAtPath("/root/shirt")), method="isotropic", edge_length=1.11
        )
        builder.add_cloth_mesh(
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
        self.model.edge_rest_angle.zero_()

        # IK places the middle of each hand's fingertip pads, pointing down with the fingers closing along y, so that
        # the hands' long sides face each other when they pass; each hand keeps the yaw it has in the ready pose.
        self.ik_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.pads = [ik.IKObjectivePosition(h, wp.vec3(0.0, 0.0, PAD), wp.zeros(1, wp.vec3)) for h in self.hands]
        objectives = list(self.pads)
        for hand, (x, _) in zip(self.hands, BASES, strict=True):
            yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), (np.pi if x > 0.0 else 0.0) - 0.25 * np.pi)
            down = wp.mul(yaw, wp.quat(1.0, 0.0, 0.0, 0.0))
            objectives.append(
                ik.IKObjectiveRotation(hand, wp.quat_identity(), wp.array([wp.vec4(*down)], dtype=wp.vec4))
            )
        objectives.append(ik.IKObjectiveJointLimit(self.model.joint_limit_lower, self.model.joint_limit_upper, 10.0))
        self.ik = ik.IKSolver(self.model, 1, objectives, lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
        self.heights, self.landing = {}, {}
        self.ready_q, self.home_q = self.model.joint_q.numpy().copy(), None
        self.ik_graph = None  # (iterations, graph) of the IK step
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
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=RADIUS)  # finds the table 2 RADIUS out
        self.contacts = self.pipeline.contacts()
        self.gravity = self.model.gravity.numpy()
        # Per hand: the grasp, dragging the strip under the closing pads or holding the bulge between the closed ones,
        # with its particles and their targets in the pad frame, or None.
        self.held = [None, None]
        self.pinched = []  # particles each pinch took
        self.rows = None  # the pinched particles and their targets in the last substep

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, 0.2, 1.38), pitch=-59.0, yaw=-90.0)

    def top_near(self, xy):
        """Height of the highest shirt particle within 1.5 cm of ``xy`` [m]."""
        q = self.state_0.particle_q.numpy()
        near = np.linalg.norm(q[:, :2] - np.asarray(xy), axis=1) < 0.015
        return q[near, 2].max() if near.any() else TABLE_TOP + RADIUS

    def height(self, ref):
        """Pad-middle height of a measured key pose [m]."""
        if ref[0] == "land":
            _, i, hand = ref
            if (i, hand) not in self.landing:  # nothing pinched: set the empty pads down as on the grip
                self.landing[(i, hand)] = (self.top_near(self.plan.folds[i]["laid"][hand]), PAD_HALF_LENGTH + 0.002)
            top, drop = self.landing[(i, hand)]
            # the turned pinch's lowest particle a self-contact margin and 3 mm over the shirt there
            return top + FOLD_MARGIN + 0.003 + drop
        i, hand = ref
        # the pads' lower edges 1 mm over the shirt's contact thickness against them
        return self.top_near(self.plan.folds[i]["grips"][hand]) + CLOTH_RADIUS + 0.001 + PAD_HALF_LENGTH

    def key_z(self, key):
        ref = key[3]
        if ref is None:
            return key[4]
        if ref not in self.heights:
            self.heights[ref] = self.height(ref)
        return self.heights[ref] + key[4]

    def hand_targets(self, t):
        """Each hand's pad-middle target, finger position and pinch angle at time ``t``."""
        out = []
        for hand in range(2):
            a, b, s = self.plan.sample(hand, t)
            za, zb = self.key_z(a), self.key_z(b)
            x, y, finger, theta = (a[k] + s * (b[k] - a[k]) for k in (1, 2, 5, 6))
            out.append(((x, y, za + s * (zb - za)), finger, theta))
        return out

    def key_pose(self, t, iterations=8):
        """Joint positions of both arms at time ``t``."""
        targets = self.hand_targets(t)
        if t > self.plan.home[0]:  # back to the ready pose in joint space, easing in and out
            if self.home_q is None:
                self.home_q = self.ik_q.numpy()[0].copy()
            s = min((t - self.plan.home[0]) / (self.plan.home[1] - self.plan.home[0]), 1.0)
            q = self.home_q + (0.5 - 0.5 * np.cos(np.pi * s)) * (self.ready_q - self.home_q)
            self.ik_q.assign(q[None, :])
            return q
        for pads, (p, _, _) in zip(self.pads, targets, strict=True):
            pads.set_target_position(0, wp.vec3(*p))
        if self.ik_q.device.is_cuda:
            if self.ik_graph is None or self.ik_graph[0] != iterations:
                with wp.ScopedCapture(device=self.ik_q.device) as capture:
                    self.ik.step(self.ik_q, self.ik_q, iterations=iterations)
                self.ik_graph = (iterations, capture.graph)
            wp.capture_launch(self.ik_graph[1])
        else:
            self.ik.step(self.ik_q, self.ik_q, iterations=iterations)
        q = self.ik_q.numpy()[0]
        for f, (_, finger, _) in zip(self.fingers, targets, strict=True):
            q[f : f + 2] = finger
        return q

    def pad_frame(self, state, hand):
        """Middle of a hand's fingertip pads and their axes."""
        X = wp.transform(*state.body_q.numpy()[self.hands[hand]])
        R = np.array(wp.quat_to_matrix(X.q)).reshape(3, 3)
        return np.array(X.p) + R @ np.array([0.0, 0.0, PAD]), R @ PAD_AXES

    def start_drag(self, origin, axes, half_gap):
        """The strip of the shirt's layers under and between the pads, with its place along the closing axis."""
        x = self.state_0.particle_q.numpy()
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

    def start_hold(self, hand, i, origin, axes, half_gap, drag=None):
        """Hold the bulge that the closed pads squeezed up between them as it stands, with a particle of each layer of
        the strip that it leaves out, turning it with its fold."""
        fold = self.plan.folds[i]
        x = self.state_0.particle_q.numpy()
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
        self.landing[(i, hand)] = (self.top_near(fold["laid"][hand]), drop)
        self.pinched.append(len(taken))
        return {"phase": "hold", "taken": taken, "local": held, "middle": middle, "axis": axes.T @ fold["axes"][hand]}

    def update_grasp(self):
        """While a hand's fingers close at a grip, its pads drag the layers under them in, so that the cloth between
        them buckles up; once they are closed, hold that bulge at the pads, turning it with the fold, until they
        open."""
        q, rows, drag_rows = self.state_1.joint_q.numpy(), [], []
        i = self.plan.fold_at(self.sim_time)
        targets = self.hand_targets(self.sim_time)
        for hand, f in enumerate(self.fingers):
            origin, axes = self.pad_frame(self.state_1, hand)
            a, b, _ = self.plan.sample(hand, self.sim_time)
            closing = b[5] < a[5]
            held = self.held[hand]
            if held is not None and held["phase"] == "hold" and q[f] > 0.5 * (OPEN + CLOSED):
                held = None
            if held is None and closing and i is not None and q[f] < OPEN - 5.0e-4:
                held = self.start_drag(origin, axes, q[f])
            if held is not None and held["phase"] == "drag" and q[f] < CLOSED + 1.0e-3:
                held = self.start_hold(hand, i, origin, axes, q[f], drag=held)
            self.held[hand] = held
            if held is None:
                continue
            if held["phase"] == "drag":
                # under a pad: with it; between the pads: shortened with the gap
                u, g0, g = held["along"], held["gap"], q[f]
                along = np.where(np.abs(u) >= g0, np.sign(u) * (np.abs(u) - g0 + g), u * g / g0)
                drag_rows.append((held["taken"], np.tile(held["axis"], (len(u), 1)), origin @ held["axis"] + along))
            else:
                turn = rotation(held["axis"], targets[hand][2])
                patch = held["middle"] + (held["local"] - held["middle"]) @ turn.T
                rows.append((held["taken"], origin + patch @ axes.T))
        free = []
        for hand, held in enumerate(self.held):
            if held is not None:
                origin, _ = self.pad_frame(self.state_1, hand)
                xy = self.state_0.particle_q.numpy()[:, :2]
                free.append(np.flatnonzero(np.linalg.norm(xy - origin[:2], axis=1) < ANCHOR_FREE))
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
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), f"non-finite shirt positions at t={self.sim_time:.2f} s"
        speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max()
        assert speed < 5.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        # The table's contacts keep the particle radius between the shirt and the table top; laying the hem over the
        # stacked layers presses the lowest one up to 5.3 mm into it, and fingers that close on a stack, dragging
        # every layer, press the lowest one for a few frames up to its radius under the top.
        low = x[:, 2].min() - TABLE_TOP
        assert low > -CLOTH_RADIUS, f"a particle is {-low * 1e3:.1f} mm under the table top at t={self.sim_time:.2f} s"
        if self.rows is not None:
            off = np.linalg.norm(x[self.rows[0]] - self.rows[1], axis=1).max()
            assert off < 1.0e-3, f"a pinched particle is {off * 1e3:.2f} mm off at t={self.sim_time:.2f} s"

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert min(self.pinched, default=6) >= 6, f"the pinches took {self.pinched} particles"
        done = [self.sim_time >= f["t1"] + 1.0 for f in self.plan.folds]
        if done[0]:
            side = np.percentile(x[:, 0], 99)
            assert side < SIDE_LINE + 0.05, f"the +x side reaches x = {side:.3f} m after its fold"
        if done[1]:
            side = np.percentile(x[:, 0], 1)
            assert side > -SIDE_LINE - 0.05, f"the -x side reaches x = {side:.3f} m after its fold"
        if done[2]:
            hem = np.percentile(x[:, 1], 99)
            assert hem < HEM_LINE + 0.05, f"the hem half reaches y = {hem:.3f} m after its fold"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=2250)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
