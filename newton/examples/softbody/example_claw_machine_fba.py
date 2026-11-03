# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Claw Machine FBA
#
# A claw machine full of soft toys. 25 tetrahedral toys, bears, elephants,
# bunnies and T2 toys of 88k particles and 305k tetrahedra, start in an
# ordered pile in the air over the cabinet's tray, fall onto it under a
# gravity that ramps up, and settle for six seconds. Then an articulated
# claw, whose X, Y and Z stages and the plate that opens and closes its
# three fingers the example drives with force-limited PD forces, centers over
# the green bear at the back, lowers, closes on it with a 14 N limit that
# rises to 20 N and 35 N as it lifts, carries it clear of the pile and holds
# it. Every toy stays dynamic: SolverFBA integrates the toys and the claw and
# solves the contacts between the toys, with the claw, the tray and the glass
# walls, with the claw's joints, in one semismooth Newton step per
# local-global iteration, on the Schur matrix in sparse form.
#
# The toys keep 41 mm from each other, about the distance between their
# coarse simulation meshes and their visual surfaces, which follow the
# tetrahedra, and their particle radius from the tray and the walls, 28 mm,
# or 8 mm for the bunnies, the T2 toys and the green bear, but only 2 mm
# from the claw's fingers, so that the fingers close on the bear's visual
# surface.
# A toy does not collide with itself, except the elephants: a falling
# elephant's body would crush the thin part it lands on.
#
# Command: python -m newton.examples claw_machine_fba
# Skip the fall: --start settled --num-frames 420
# Start with the claw open over the bear: --start grasp --num-frames 285
###########################################################################

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.softbody.claw_machine_fba_utils import (
    SceneVisuals,
    build_linkage,
    measure,
    plate_force_limit,
    plate_position_gain,
    target_at,
    tet_volumes,
    transform,
)
from newton.solvers import SolverFBA

# The example's drag on the toys, -MASS_DAMPING m v, explicit in each step, which calms the pile.
MASS_DAMPING = wp.constant(12.0)  # [1/s]
TARGET = 9  # the green bear at the back
SETTLE = 6.0  # [s] before the claw moves and its fingers collide


@wp.kernel
def mass_damping(velocity: wp.array[wp.vec3], mass: wp.array[wp.float32], force: wp.array[wp.vec3]):
    i = wp.tid()
    force[i] = -MASS_DAMPING * mass[i] * velocity[i]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.asset_root = Path(newton.examples.get_asset("claw_machine_urban/scene.json")).parent
        self.scene = json.loads((self.asset_root / "scene.json").read_text())
        self.finger_contact_radius = args.finger_contact_radius
        if self.finger_contact_radius is None:
            self.finger_contact_radius = self.scene["finger_contact_radius"]
        with np.load(self.asset_root / "simulation.npz", allow_pickle=False) as data:
            self.data = dict(data)
        self.origin = np.asarray(self.scene["origin"])
        self.centers = np.asarray(self.scene["body_centers"])
        self.offsets = self.data["particle_offsets"]
        self.tets = self.data["tet_indices"]
        self.rest = self.data["rest_particle_q"]
        self.rest_volumes = tet_volumes(self.rest, self.tets)
        self.frame_dt = 1 / self.scene["fps"]
        self.sim_dt = 1 / self.scene["sim_hz"]
        self.sim_substeps = self.scene["sim_hz"] // self.scene["fps"]
        self.settle_tick = round(SETTLE * self.scene["sim_hz"])
        self.tick = round(self.scene["checkpoint_times_s"][args.start] * self.scene["sim_hz"])
        self.sim_time = self.tick * self.sim_dt
        self.end_tick = round((SETTLE + float(self.scene["trajectory"][-1][0])) * self.scene["sim_hz"])
        self.build_model()
        if self.tick >= self.settle_tick:
            self.collide_fingers()
        if args.start != "initial":
            for field in ("particle_q", "particle_qd", "body_q", "body_qd"):
                getattr(self.state_0, field).assign(self.data[args.start + "_" + field])
        self.visuals = None if args.viewer == "null" else SceneVisuals(self.asset_root, self.model.device, self.offsets)
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.show_joints = False
        self.set_scene_camera()
        self.hold_minimum_lift = float("inf")
        self.hold_minimum_clearance = float("inf")
        self.hold_frames = 0
        self.carry_start_center = None
        target = slice(self.offsets[TARGET], self.offsets[TARGET + 1])
        self.target_initial_center = self.data["settled_particle_q"][target].mean(axis=0)

    def set_scene_camera(self):
        """Move the camera along the scene's path: from the wide view of the cabinet to the side view of the grasp,
        between the transition's start and end."""
        config = self.scene["camera"]
        start, end = config["transition_seconds"]
        u = float(np.clip((self.sim_time - start) / (end - start), 0.0, 1.0))
        w = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        initial_target = np.asarray(config["initial_target"])
        final_target = np.asarray(config["final_target_y_up"])
        initial = np.asarray(config["initial_position_exact"]) - initial_target
        final = np.asarray(config["final_position_y_up"]) - final_target
        angle = (1.0 - w) * np.arctan2(initial[2], initial[0]) + w * np.arctan2(final[2], final[0])
        radius = (1.0 - w) * np.linalg.norm(initial[[0, 2]]) + w * np.linalg.norm(final[[0, 2]])
        offset = np.array([radius * np.cos(angle), (1.0 - w) * initial[1] + w * final[1], radius * np.sin(angle)])
        target = (1.0 - w) * initial_target + w * final_target
        position = target + offset
        # The path is given Y-up; the scene is Z-up.
        position = position[[0, 2, 1]] * [1.0, -1.0, 1.0] - self.origin
        direction = -offset[[0, 2, 1]] * [1.0, -1.0, 1.0]
        pitch = np.degrees(np.arctan2(direction[2], np.linalg.norm(direction[:2])))
        yaw = np.degrees(np.arctan2(direction[1], direction[0]))
        self.viewer.set_camera(pos=wp.vec3(*position), pitch=float(pitch), yaw=float(yaw))
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = config["fov_fixed_degrees"]

    def build_model(self):
        """The claw, the toys, the tray, the glass walls and the fingers' collision capsules, which the toys pass
        through until :meth:`collide_fingers`."""
        builder = build_linkage(self.scene)
        tet_offsets = self.data["tet_offsets"]
        for i, desc in enumerate(self.scene["layout"]):
            p0, p1 = self.offsets[i : i + 2]
            t0, t1 = tet_offsets[i : i + 2]
            material = self.data["tet_materials"][t0]
            radius = float(desc.get("particle_radius", 0.008 if desc.get("target") else 0.028))
            builder.add_soft_mesh(
                pos=wp.vec3(),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=self.rest[p0:p1].tolist(),
                indices=(self.tets[t0:t1] - p0).ravel().tolist(),
                density=float(desc.get("mass", 0.5)) / self.rest_volumes[t0:t1].sum(),
                k_mu=float(material[0]),
                k_lambda=float(material[1]),
                k_damp=0.0,
                particle_radius=radius,
                add_surface_mesh_edges=False,
                label=desc.get("label", f"toy_{i}"),
            )
        builder.add_shape_box(
            -1,
            xform=transform(np.array([-0.0045, 0.006, 1.87]) - self.origin),
            hx=1.5705,
            hy=1.585,
            hz=0.04,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, density=0.0, is_visible=False),
            label="tray",
        )
        bounds = np.array([[-1.5786355, -1.582482, 1.91], [1.569963, 1.594493, 4.929962]])
        for axis in (0, 1):
            for side in (0, 1):
                center = bounds.mean(axis=0)
                half = (bounds[1] - bounds[0]) * 0.5
                center[axis] = bounds[side, axis] + (-0.01 if side == 0 else 0.01)
                half[axis] = 0.01
                builder.add_shape_box(
                    -1,
                    xform=transform(center - self.origin),
                    hx=half[0],
                    hy=half[1],
                    hz=half[2],
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.3, density=0.0, is_visible=False),
                    label=f"glass_wall_{axis}_{side}",
                )
        cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=1.0,
            margin=0.0,
            gap=0.003,
            is_visible=False,
            has_shape_collision=True,
            has_particle_collision=False,
        )
        for i, capsule in enumerate(self.scene["capsules"]):
            builder.add_shape_capsule(
                capsule["body"],
                xform=wp.transform(wp.vec3(*capsule["position"]), wp.quat(*capsule["quaternion"])),
                radius=capsule["radius"],
                half_height=capsule["half_height"],
                cfg=cfg,
                label=f"finger_capsule_{i}",
            )
        self.model = builder.finalize()
        self.model.particle_mu = 0.5
        self.model.tet_materials.assign(self.data["tet_materials"])
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.015)
        self.contacts = self.pipeline.contacts()
        # The tray and the four walls are shapes 0-4; the fingers' capsules follow.
        self.fingers = fingers = range(5, self.model.shape_count)
        self.solver = SolverFBA(
            self.model,
            iterations=self.scene["iterations"],
            stretching_model="neohookean",
            nsn_schur_mode="full",
            nsn_sparse_storage=True,
            nsn_linear_iterations=self.scene["linear_iterations"],
            shape_particle_radius_override=dict.fromkeys(fingers, self.finger_contact_radius),
            friction=True,
            face_contacts=True,
            face_contact_gap=0.012,
            contact_regularization=0.1,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.041,
            particle_self_contact_gap=0.003,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.045,
            # Each toy is a group, which collides with the others, and only an elephant with itself: its body would
            # otherwise crush the thin part it lands on and invert its tetrahedra.
            particle_self_contact_group_ids=np.repeat(np.arange(25, dtype=np.int32), np.diff(self.offsets)),
            particle_self_contact_group_margins=[
                0.041 if d["toy_name"] == "toy_elephant_05" else 0.0 for d in self.scene["layout"]
            ],
        )

    def collide_fingers(self):
        """Let the fingers' capsules collide with the toys from now on."""
        flags = self.model.shape_flags.numpy()
        flags[self.fingers.start : self.fingers.stop] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        self.model.shape_flags.assign(flags)

    def step(self):
        for _ in range(self.sim_substeps):
            if self.tick >= self.end_tick:
                break
            if self.tick == self.settle_tick:
                self.collide_fingers()
            t = self.tick * self.sim_dt
            gravity = 1.0
            q, qd = measure(
                self.state_0.body_q.numpy().astype(float), self.state_0.body_qd.numpy().astype(float), self.centers
            )
            kp = np.full(4, 600.0)
            limits = np.array([60.0, 60.0, 60.0, 30.0])
            if t < SETTLE:
                u = float(np.clip((t - 0.7) / 0.75, 0.0, 1.0))
                gravity = u * u * (3.0 - 2.0 * u)
                reference, velocity = np.zeros(4), np.zeros(4)
            else:
                reference, velocity, _ = target_at(t - SETTLE, self.scene["trajectory"])
                kp[3] = plate_position_gain(t - SETTLE, self.scene["plan"])
                limits[3] = plate_force_limit(t - SETTLE, self.scene["plan"])
            self.model.set_gravity((0.0, 0.0, -9.81 * gravity))
            # Gravity feedforward of the Z stage's load and the plate's.
            ff = np.array(
                [
                    0.0,
                    0.0,
                    9.81 * sum(self.scene["body_masses"][2:]),
                    np.interp(q[3], self.scene["gravity_grid"], self.scene["gravity_plate"]),
                ]
            )
            effort = np.clip(
                kp * (reference - q) + np.array([50.0, 45.0, 45.0, 20.0]) * (velocity - qd) + ff * gravity,
                -limits,
                limits,
            )
            if t >= SETTLE:
                effort[2] = max(effort[2], ff[2] - 8.0)
            # Each motor pushes its body along its axis and its parent back.
            wrench = np.zeros((10, 6), dtype=np.float32)
            axes = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 1.0]])
            for j, parent in enumerate([-1, 0, 1, 2]):
                force = effort[j] * axes[j]
                wrench[j, :3] += force
                if parent >= 0:
                    wrench[parent, :3] -= force
            self.state_0.clear_forces()
            self.state_0.body_f.assign(wrench)
            wp.launch(
                mass_damping,
                len(self.rest),
                inputs=[self.state_0.particle_qd, self.model.particle_mass],
                outputs=[self.state_0.particle_f],
                device=self.model.device,
            )
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.tick += 1
        self.sim_time = self.tick * self.sim_dt

    def render(self):
        if self.sim_time <= self.scene["camera"]["transition_seconds"][1] + self.frame_dt:
            self.set_scene_camera()
        self.viewer.begin_frame(self.sim_time)
        if self.visuals is not None:
            self.visuals.render(self.viewer, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        points = self.state_0.particle_q.numpy()
        assert np.isfinite(points).all() and np.isfinite(self.state_0.body_q.numpy()).all(), "Nonfinite state"
        ratio = tet_volumes(points, self.tets) / self.rest_volumes
        assert ratio.min() > 0.0, f"Inverted tet at {self.sim_time:.3f}s"
        target = points[self.offsets[TARGET] : self.offsets[TARGET + 1]]
        center = target.mean(axis=0)
        carry_start = SETTLE + self.scene["trajectory"][-3][0]
        hold_start = SETTLE + self.scene["trajectory"][-2][0]
        if self.carry_start_center is None and self.sim_time >= carry_start:
            self.carry_start_center = center.copy()
        if self.sim_time >= hold_start:
            other = np.concatenate([points[: self.offsets[TARGET]], points[self.offsets[TARGET + 1] :]])
            self.hold_minimum_lift = min(self.hold_minimum_lift, float(center[2] - self.target_initial_center[2]))
            self.hold_minimum_clearance = min(
                self.hold_minimum_clearance, float(target[:, 2].min() - other[:, 2].max())
            )
            self.hold_frames += 1

    def test_final(self):
        self.test_post_step()
        if self.tick >= self.end_tick:
            points = self.state_0.particle_q.numpy()
            target = points[self.offsets[TARGET] : self.offsets[TARGET + 1]]
            other = np.concatenate([points[: self.offsets[TARGET]], points[self.offsets[TARGET + 1] :]])
            assert target[:, 2].mean() - self.target_initial_center[2] > 0.7, "Target was not lifted"
            assert target[:, 2].min() > other[:, 2].max() + 0.05, "Target is not clear of the pile"
            assert self.hold_minimum_lift > 0.7 and self.hold_minimum_clearance > 0.05, "Grip slipped during hold"
            assert self.hold_frames >= 60, "Full two-second hold was not tested"
            carry = np.linalg.norm(target.mean(axis=0)[:2] - self.carry_start_center[:2])
            assert carry > 0.15, "Target was not carried horizontally"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)
        parser.add_argument(
            "--start",
            choices=("initial", "settled", "grasp"),
            default="initial",
            help="Start in the air, from the settled pile at 6 s, or with the claw open over the bear at 10.5 s.",
        )
        parser.add_argument(
            "--finger-contact-radius",
            type=float,
            default=None,
            help="Contact thickness of the toys against the fingers [m]; the scene's is 0.002.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
