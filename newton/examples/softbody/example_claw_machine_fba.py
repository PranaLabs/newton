# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Force-controlled Urban claw grasping a free green bear among 25 soft toys.

Run: uv run --no-sync -m newton.examples claw_machine_fba
Skip the six-second fall: --start settled --num-frames 420
Start at the saved open-gripper pose: --start grasp --num-frames 285

The default 20-second sequence runs at 240 Hz with 5 x 10 iterations.
Prepared checkpoints restore positions AND velocities only at initialization;
subsequent motion follows motor forces, joint constraints and frictional contact.
"""

import gc
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.claw_machine_fba_utils import (
    SceneVisuals,
    build_linkage,
    measure,
    planned_target,
    plate_force_limit,
    plate_position_gain,
    tet_volumes,
    transform,
)
from newton.solvers import SolverFBA


@wp.kernel
def mass_damping(velocity: wp.array[wp.vec3], mass: wp.array[wp.float32], force: wp.array[wp.vec3]):
    i = wp.tid()
    force[i] = -12.0 * mass[i] * velocity[i]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.finger_contact_radius = getattr(args, "finger_contact_radius", None)
        self.asset_root = Path(newton.examples.get_asset("claw_machine_urban/scene.json")).parent
        self.scene = json.loads((self.asset_root / "scene.json").read_text())
        if self.finger_contact_radius is None:
            self.finger_contact_radius = self.scene["finger_contact_radius"]
        self.solver_reset_times = self.scene["solver_reset_times_s"]
        if getattr(args, "grasp_plan", None) is not None:
            plan = json.loads(Path(args.grasp_plan).read_text())
            keys = plan["trajectory_keyframes"]
            times = np.array([row[0] for row in keys], dtype=float)
            commands = np.array([row[1] for row in keys], dtype=float)
            if (
                len(keys) < 3
                or times[0] != 0.0
                or np.any(np.diff(times) <= 0.0)
                or commands.shape != (len(keys), 4)
                or not np.isfinite(times).all()
                or not np.isfinite(commands).all()
                or plan.get("target_toy_index", 9) != 9
            ):
                raise ValueError("Expected finite ordered motor keyframes for green bear 9")
            self.scene["trajectory"] = keys
            self.scene["plan"] = plan.get("grasp_plan", {})
            self.solver_reset_times = plan.get("solver_reset_times_s", [])
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
        self.tick = round(self.scene["checkpoint_times_s"][args.start] * self.scene["sim_hz"])
        self.sim_time = self.tick * self.sim_dt
        self.end_tick = round((6.0 + float(self.scene["trajectory"][-1][0])) * self.scene["sim_hz"])
        self.solver_reset_ticks = {round((6.0 + float(t)) * self.scene["sim_hz"]) for t in self.solver_reset_times}
        self.build_model(fingers=self.tick >= 1440)
        if args.start != "initial":
            for field in ("particle_q", "particle_qd", "body_q", "body_qd"):
                getattr(self.state_0, field).assign(self.data[args.start + "_" + field])
        self.visuals = None if args.viewer == "null" else SceneVisuals(self.asset_root, self.model.device, self.offsets)
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.show_joints = False
        self.set_scene_camera()
        self.phase = "INITIAL / ORDERED IN AIR"
        self.checkpoint_path = args.save_checkpoint
        self.minimum_volume_ratio = 1.0
        self.test_mode = getattr(args, "test", False)
        self.hold_minimum_lift = float("inf")
        self.hold_minimum_clearance = float("inf")
        self.hold_frames = 0
        self.carry_start_center = None
        self.target_initial_center = self.data["settled_particle_q"][self.offsets[9] : self.offsets[10]].mean(axis=0)

    def set_scene_camera(self):
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
        # The saved render camera is Y-up; Newton uses the source Z-up coordinates.
        position = position[[0, 2, 1]] * [1.0, -1.0, 1.0] - self.origin
        direction = -offset[[0, 2, 1]] * [1.0, -1.0, 1.0]
        pitch = np.degrees(np.arctan2(direction[2], np.linalg.norm(direction[:2])))
        yaw = np.degrees(np.arctan2(direction[1], direction[0]))
        self.viewer.set_camera(pos=wp.vec3(*position), pitch=float(pitch), yaw=float(yaw))
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = config["fov_fixed_degrees"]

    def build_model(self, *, fingers):
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
        if fingers:
            cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=1.0,
                margin=0.0,
                gap=0.003,
                is_visible=False,
                has_shape_collision=True,
                has_particle_collision=True,
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
        self.make_solver()

    def make_solver(self):
        self.solver = SolverFBA(
            self.model,
            iterations=5,
            stretching_model="neohookean",
            nsn_schur_mode="full",
            nsn_sparse_storage=True,
            shape_particle_radius_override=dict.fromkeys(range(5, self.model.shape_count), self.finger_contact_radius)
            if self.finger_contact_radius is not None
            else None,
            friction=True,
            face_contacts=True,
            face_contact_margin=0.012,
            contact_regularization=0.1,
            nsn_linear_iterations=10,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.041,
            particle_self_contact_gap=0.003,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.045,
            particle_self_contact_group_ids=np.repeat(np.arange(25, dtype=np.int32), np.diff(self.offsets)),
            particle_self_contact_group_margins=np.array(
                [d.get("self_contact_margin", 0.041) for d in self.scene["layout"]], dtype=np.float32
            ),
        )

    def step(self):
        for _ in range(self.sim_substeps):
            if self.tick >= self.end_tick:
                break
            # Match the original segment boundaries, carrying physical state forward.
            # Contact/factorization caches restart, but no toy is pinned or reset.
            if self.tick == 1440 and self.model.shape_count == 5:
                saved = {
                    field: getattr(self.state_0, field).numpy()
                    for field in ("body_q", "body_qd", "particle_q", "particle_qd")
                }
                self.solver = None
                gc.collect()
                self.build_model(fingers=True)
                for field, value in saved.items():
                    getattr(self.state_0, field).assign(value)
                self.viewer.set_model(self.model)
                if self.visuals is not None:
                    self.visuals.static_logged = False
                self.viewer.show_particles = self.viewer.show_triangles = self.viewer.show_joints = False
            if self.tick in self.solver_reset_ticks:
                self.solver = None
                gc.collect()
                self.make_solver()
            t = self.tick * self.sim_dt
            gravity = 1.0
            q, qd = measure(
                self.state_0.body_q.numpy().astype(float), self.state_0.body_qd.numpy().astype(float), self.centers
            )
            kp = np.full(4, 600.0)
            limits = np.array([60.0, 60.0, 60.0, 30.0])
            if t < 6.0:
                u = float(np.clip((t - 0.7) / 0.75, 0.0, 1.0))
                gravity = u * u * (3.0 - 2.0 * u)
                reference, velocity = np.zeros(4), np.zeros(4)
                self.phase = "INITIAL / ORDERED IN AIR" if t < 0.7 else "GRAVITY / FREE PILE"
            else:
                reference, velocity, self.phase = planned_target(t - 6.0, self.scene["trajectory"], self.scene["plan"])
                kp[3] = plate_position_gain(t - 6.0, self.scene["plan"])
                limits[3] = plate_force_limit(t - 6.0, self.scene["plan"])
            self.model.set_gravity((0.0, 0.0, -9.81 * gravity))
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
            if t >= 6.0:
                effort[2] = max(effort[2], ff[2] - 8.0)
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
            if self.test_mode:
                points = self.state_0.particle_q.numpy()
                ratio = tet_volumes(points, self.tets) / self.rest_volumes
                self.minimum_volume_ratio = min(self.minimum_volume_ratio, float(ratio.min()))
                assert np.isfinite(points).all() and ratio.min() > 0.0, f"Invalid tet at tick {self.tick}"
        self.sim_time = self.tick * self.sim_dt

    def render(self):
        self.set_scene_camera()
        self.viewer.begin_frame(self.sim_time)
        if self.visuals is not None:
            self.visuals.render(self.viewer, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        points = self.state_0.particle_q.numpy()
        assert np.isfinite(points).all() and np.isfinite(self.state_0.body_q.numpy()).all(), "Nonfinite state"
        ratio = tet_volumes(points, self.tets) / self.rest_volumes
        self.minimum_volume_ratio = min(self.minimum_volume_ratio, float(ratio.min()))
        assert ratio.min() > 0.0, f"Inverted tet at {self.sim_time:.3f}s"
        target = points[self.offsets[9] : self.offsets[10]]
        center = target.mean(axis=0)
        carry_start = 6.0 + self.scene["trajectory"][-3][0]
        hold_start = 6.0 + self.scene["trajectory"][-2][0]
        if self.carry_start_center is None and self.sim_time >= carry_start:
            self.carry_start_center = center.copy()
        if self.sim_time >= hold_start:
            other = np.concatenate([points[: self.offsets[9]], points[self.offsets[10] :]])
            self.hold_minimum_lift = min(self.hold_minimum_lift, float(center[2] - self.target_initial_center[2]))
            self.hold_minimum_clearance = min(
                self.hold_minimum_clearance, float(target[:, 2].min() - other[:, 2].max())
            )
            self.hold_frames += 1
        if self.tick % self.scene["sim_hz"] == 0:
            print(f"t={self.sim_time:.1f}s minimum_J={self.minimum_volume_ratio:.6f}", flush=True)

    def test_final(self):
        self.test_post_step()
        assert np.all(self.model.particle_inv_mass.numpy() > 0.0), "All toys must remain dynamic"
        assert np.array_equal(self.model.tet_materials.numpy(), self.data["tet_materials"])
        if self.tick >= self.end_tick:
            points = self.state_0.particle_q.numpy()
            target = points[self.offsets[9] : self.offsets[10]]
            other = np.concatenate([points[: self.offsets[9]], points[self.offsets[10] :]])
            assert target[:, 2].mean() - self.target_initial_center[2] > 0.7, "Target was not lifted"
            assert target[:, 2].min() > other[:, 2].max() + 0.05, "Target is not clear of the pile"
            assert self.hold_minimum_lift > 0.7 and self.hold_minimum_clearance > 0.05, "Grip slipped during hold"
            assert self.hold_frames >= 60, "Full two-second hold was not tested"
            carry = np.linalg.norm(target.mean(axis=0)[:2] - self.carry_start_center[:2])
            assert carry > 0.15, "Target was not carried horizontally"
            print(
                f"PASS: lift={self.hold_minimum_lift:.3f}m clearance={self.hold_minimum_clearance:.3f}m carry={carry:.3f}m",
                flush=True,
            )
        if self.checkpoint_path:
            np.savez_compressed(
                self.checkpoint_path,
                time=self.sim_time,
                **{
                    field: getattr(self.state_0, field).numpy()
                    for field in ("particle_q", "particle_qd", "body_q", "body_qd")
                },
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    parser.add_argument(
        "--start",
        choices=("initial", "settled", "grasp"),
        default="initial",
        help="Initial layout or a prepared dynamic checkpoint (6 s / 10.5 s).",
    )
    parser.add_argument("--save-checkpoint", default=None, help="Write the final state when --test is enabled.")
    parser.add_argument(
        "--finger-contact-radius",
        type=float,
        default=None,
        help="Override the default 0.002 m gripper thickness; preserve tray, walls and self contact.",
    )
    parser.add_argument(
        "--grasp-plan",
        type=Path,
        default=None,
        help="Optional JSON motor trajectory and force schedules; does not modify the restored physical state.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
