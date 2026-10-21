# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Shirt Fold FBA
#
# Two Franka Panda arms on either side of a table fold a T-shirt that drops
# onto it: each side in toward the middle, the arm on that side holding the
# shoulder while the other reaches across for the hem corner, then the hem
# up toward the collar. The shirt lands with a 12 mm self-contact margin and
# folds with a 6 mm one; edge-edge self contacts keep its creases apart and
# friction anchors keep the folds from creeping open. IK poses the arms and
# Featherstone moves them kinematically; SolverFBA steps the shirt with the
# lite contact operator and holds the pinched layers.
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
RADIUS = 0.008  # the shirt's contact thickness against shapes [m]
# Self-contact margin while the shirt lands, which its falling layers cannot cross in a step, and once it lies
# still [m], from THIN_TIME [s] on.
LAND_MARGIN, FOLD_MARGIN, THIN_TIME = 0.012, 0.006, 4.5
OPEN, CLOSED = 0.04, 0.016  # finger joint positions [m]: the pads stand twice that apart
# From the last link to the middle of the fingertip pads, and their half width and half length [m]; their axes in
# the last link's frame, as columns: closing, spanning the pads and out of the fingertips.
PAD, PAD_HALF_WIDTH, PAD_HALF_LENGTH = 0.21065, 0.00875, 0.00925
PAD_AXES = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)]]) / np.sqrt(2.0)
# A pinch raises the shirt's layers under the closed pads over 0.5 s until the lowest is 2 mm under the pads' lower
# edges, and holds at most 10 particles: three rows each, within the lite operator's 64 point rows.
GRASP_RISE, INTO_GAP, MAX_PINCHED = 0.5, -0.002, 10
HOVER = 0.08  # approach height over a grip [m]
BASES = ((0.48, -0.35), (-0.48, -0.35))  # the arms' bases beside the table, facing it [m]
# The arms' joints at rest, Franka's ready pose, and where it holds the pads: ahead of the base and high [m].
READY = [0.0, -0.25 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.5 * np.pi, 0.25 * np.pi]
REST_AHEAD, REST_HEIGHT = 0.307, 0.487
# The +x side fold: shoulder and hem-corner grips and the fold line at x = SIDE_LINE; the -x side mirrors it. The
# hem fold: grips at the hem corners of the folded body and the fold line at y = HEM_LINE [m].
SHOULDER, HEM_CORNER, SIDE_LINE = (0.24, -0.70), (0.21, -0.215), 0.12
HEM_GRIP, HEM_LINE = (0.10, -0.20), -0.48


def rotation(axis, theta):
    """Rotation matrix by ``theta`` about ``axis``."""
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * K @ K


def pick_layers(candidates, depth, off_middle):
    """Up to MAX_PINCHED of the candidates, taken in turn from each layer, nearest the middle of the gap first, so
    that a pinch holds every layer between the pads."""
    if not len(candidates):
        return candidates
    layer = np.rint((depth - depth.min()) / FOLD_MARGIN).astype(int)
    queues = [list(candidates[layer == k][np.argsort(off_middle[layer == k])]) for k in np.unique(layer)]
    taken = []
    while len(taken) < MAX_PINCHED and any(queues):
        for queue in queues:
            if queue and len(taken) < MAX_PINCHED:
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

    def fold(self, grips, axis, line, *, arc=3.0, approach=2.0, steps=12):
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
        self.to(GRASP_RISE, None, None)
        for k in range(1, steps + 1):
            theta = np.pi * k / steps
            hands = []
            for h, (g, r) in enumerate(zip(grips, refs, strict=True)):
                radius = g[axis] - line
                p = list(g)
                p[axis] = line + radius * np.cos(theta)
                if k < steps:
                    hands.append((p[0], p[1], r, abs(radius) * np.sin(theta), CLOSED, theta))
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
    plan.fold([HEM_GRIP, (-HEM_GRIP[0], HEM_GRIP[1])], 1, HEM_LINE, arc=4.0)
    # back to the ready pose in joint space
    plan.home = (plan.t, plan.t + 2.0)
    plan.to(2.0, *((x, y, None, z, OPEN, 0.0) for x, y, z in rest))
    plan.to(1.0, None, None)
    return plan


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        # Five substeps of 1/300 s: the lite operator takes a particle's mass alone for its response, which
        # overstates it more the longer the step, and the 6 mm layers need its contacts close within each step.
        self.frame_dt, self.sim_substeps = 1.0 / 60.0, 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.plan = shop_fold()

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 1.5
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
            particle_radius=RADIUS,
        )
        builder.color()
        builder.add_ground_plane()
        self.model = builder.finalize()
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
            # Bounds a contact row at 200 N: two self contacts of the landing layers reach it for a step.
            lambda_cap=200.0,
            friction_anchors=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=LAND_MARGIN,
            particle_self_contact_edge_edge=True,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=LAND_MARGIN,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=RADIUS)
        self.contacts = self.pipeline.contacts()
        self.gravity = self.model.gravity.numpy()
        # Per hand: pinched particles, their pad-frame points before and after rising, the pinch's middle, its fold
        # axis in the pad frame, and the time it closed.
        self.held = [None, None]
        self.pinched = []  # particles each pinch took
        self.rows = None  # the pinched particles and their targets in the last substep

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, 0.2, 1.38), pitch=-59.0, yaw=-90.0)

    def top_near(self, xy):
        """Height of the highest shirt particle within 1.5 cm of ``xy`` [m]."""
        q = self.state_0.particle_q.numpy() if hasattr(self, "state_0") else self.model.particle_q.numpy()
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
        # the pads' lower edges 2 mm over the shirt
        return self.top_near(self.plan.folds[i]["grips"][hand]) + RADIUS + 0.002 + PAD_HALF_LENGTH

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

    def key_pose(self, t, iterations=24):
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

    def update_grasp(self):
        """Pinch every layer of the shirt under each closed gap, raise the pinch and turn it with its fold, so that a
        laid flap lies flipped like the cloth around it, and hold it at the pads' pose after the substep."""
        q, rows = self.state_1.joint_q.numpy(), []
        i = self.plan.fold_at(self.sim_time)
        targets = self.hand_targets(self.sim_time)
        for hand, f in enumerate(self.fingers):
            origin, axes = self.pad_frame(self.state_1, hand)
            if q[f] > 0.5 * (OPEN + CLOSED):
                self.held[hand] = None
            elif self.held[hand] is None and q[f] < CLOSED + 1.0e-3 and i is not None:
                fold = self.plan.folds[i]
                x = self.state_0.particle_q.numpy()
                local = (x - origin) @ axes
                depth = local[:, 2] - PAD_HALF_LENGTH  # below the pads' lower edges
                gap = np.all(np.abs(local[:, :2]) <= [CLOSED - 0.002, PAD_HALF_WIDTH + 0.004], axis=1)
                candidates = np.flatnonzero(gap & (depth >= 0.0) & (x[:, 2] > TABLE_TOP))
                taken = pick_layers(candidates, depth[candidates], np.linalg.norm(local[candidates, :2], axis=1))
                rise = np.array([0.0, 0.0, depth[taken].max(initial=0.0) + INTO_GAP])
                end = local[taken] - rise
                middle = end.mean(axis=0) if len(taken) else np.zeros(3)
                # turned half a circle about its middle, the pinch's lowest particle lies this far under the pads
                drop = 2.0 * middle[2] - end[:, 2].min() if len(taken) else PAD_HALF_LENGTH + 0.002
                self.landing[(i, hand)] = (self.top_near(fold["laid"][hand]), drop)
                self.held[hand] = (taken, local[taken], end, middle, axes.T @ fold["axes"][hand], self.sim_time)
                self.pinched.append(len(taken))
            if self.held[hand] is not None:
                taken, start, end, middle, axis, t0 = self.held[hand]
                s = min((self.sim_time - t0) / GRASP_RISE, 1.0)
                patch = middle + (start + s * (end - start) - middle) @ rotation(axis, targets[hand][2]).T
                rows.append((taken, origin + patch @ axes.T))
        rows = [r for r in rows if len(r[0])]
        if not rows:
            self.rows = None
            self.solver.set_particle_constraints(None, None, None)
            return
        p, x = (np.concatenate(a) for a in zip(*rows, strict=True))
        self.rows = (p, x)
        eye = np.tile(np.eye(3), (len(p), 1))
        self.solver.set_particle_constraints(
            np.repeat(p, 3)[:, None], np.ones((3 * len(p), 1)), x.reshape(-1), directions=eye
        )

    def step(self):
        if abs(self.sim_time - THIN_TIME) < 0.5 * self.frame_dt:
            self.solver.set_particle_self_contact(margin=FOLD_MARGIN)
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
        # stacked layers presses the lowest one up to 5.3 mm into it.
        low = x[:, 2].min() - TABLE_TOP
        assert low > 0.0, f"a particle is {-low * 1e3:.1f} mm under the table top at t={self.sim_time:.2f} s"
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
