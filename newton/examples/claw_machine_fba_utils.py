# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Prepared Urban asset loading, skinning and force-reference interpolation."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton


def transform(point):
    return wp.transform(wp.vec3(*point), wp.quat_identity())


def build_linkage(scene):
    centers = np.asarray(scene["body_centers"])
    fingers = np.asarray(scene["fingers"])
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    for name, center, mass, inertia in zip(
        scene["body_names"], centers, scene["body_masses"], scene["body_inertias"], strict=True
    ):
        builder.add_link(
            xform=transform(center), mass=mass, inertia=wp.mat33(np.eye(3) * inertia), lock_inertia=True, label=name
        )
    joints = []
    axes = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 1.0]])
    for i, parent in enumerate([-1, 0, 1, 2]):
        joints.append(
            builder.add_joint_prismatic(
                parent,
                i,
                axis=axes[i],
                parent_xform=transform(centers[3] if i == 3 else np.zeros(3)),
                child_xform=transform(np.zeros(3)),
                label=f"motor_{i}",
            )
        )
    for i, (upper, _lower, carrier, tangent, _) in enumerate(fingers):
        lo, up = 4 + 2 * i, 5 + 2 * i
        joints.append(
            builder.add_joint_revolute(
                3,
                lo,
                axis=tangent,
                parent_xform=transform(carrier - centers[3]),
                child_xform=transform(carrier - centers[lo]),
                label=f"C_hinge_{i}",
            )
        )
        joints.append(
            builder.add_joint_revolute(
                2,
                up,
                axis=tangent,
                parent_xform=transform(upper - centers[2]),
                child_xform=transform(upper - centers[up]),
                label=f"U_hinge_{i}",
            )
        )
    builder.add_articulation(joints, label="xyz_and_claw_tree")
    dof = newton.ModelBuilder.JointDofConfig
    for i, (_, lower, _, tangent, _) in enumerate(fingers):
        lo, up = 4 + 2 * i, 5 + 2 * i
        builder.add_joint_d6(
            lo,
            up,
            linear_axes=[dof(axis=tangent)],
            angular_axes=[dof(axis=x) for x in np.eye(3)],
            parent_xform=transform(lower - centers[lo]),
            child_xform=transform(lower - centers[up]),
            label=f"L_planar_closure_{i}",
        )
    return builder


def measure(poses, velocities, centers):
    q = np.array(
        [poses[0, 0], poses[1, 1] - poses[0, 1], poses[2, 2] - poses[1, 2], poses[3, 2] - poses[2, 2] - centers[3, 2]]
    )
    qd = np.array(
        [
            velocities[0, 0],
            velocities[1, 1] - velocities[0, 1],
            velocities[2, 2] - velocities[1, 2],
            velocities[3, 2] - velocities[2, 2],
        ]
    )
    return q, qd


def target_at(t, keys):
    for (t0, x0, _), (t1, x1, phase) in pairwise(keys):
        if t <= t1:
            s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
            delta = np.array(x1) - np.array(x0)
            return (
                np.array(x0) + s**3 * (10 + s * (-15 + 6 * s)) * delta,
                30 * s * s * (1 - s) * (1 - s) / (t1 - t0) * delta,
                phase,
            )
    return np.array(keys[-1][1]), np.zeros(4), keys[-1][2]


def tet_volumes(points, tets):
    p = points[tets]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6


def planned_target(t, keys, plan):
    """Evaluate a saved C2-continuous command splice when branching a safe checkpoint.

    Only the motor reference changes. Actual rigid/soft positions and velocities
    are restored from the checkpoint unchanged and then integrated normally.
    """
    splice = plan.get("command_splice")
    if splice is not None and splice["start_time_s"] <= t <= splice["end_time_s"]:
        duration = splice["end_time_s"] - splice["start_time_s"]
        u = (t - splice["start_time_s"]) / duration
        c = np.asarray(splice["coefficients"], float)
        q = sum(c[i] * u**i for i in range(6))
        qd = sum(i * c[i] * u ** (i - 1) for i in range(1, 6)) / duration
        return q, qd, splice.get("phase", "LOWER / FORCE-LIMITED APPROACH")
    return target_at(t, keys)


def plate_position_gain(t, plan, default=600.0):
    """Smoothly load a closed grip using the same capped physical motor.

    The target remains 70 mm; this changes stiffness, not the state or an
    attachment. The gain schedule is saved with the trajectory/checkpoint.
    """
    schedule = plan.get("plate_gain_schedule")
    for override in plan.get("plate_gain_schedule_overrides", []):
        if t >= float(override["start_time_s"]):
            schedule = override
    if schedule is None:
        return default
    start = float(schedule["start_time_s"])
    duration = float(schedule["ramp_duration_s"])
    initial = float(schedule.get("initial_kp_N_per_m", default))
    final = float(schedule["final_kp_N_per_m"])
    if duration <= 0 or not np.isfinite([start, duration, initial, final]).all() or initial <= 0 or final < initial:
        raise ValueError("Invalid plate gain schedule")
    u = float(np.clip((t - start) / duration, 0.0, 1.0))
    blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    return initial + (final - initial) * blend


def plate_force_limit(t, plan, default=30.0):
    """Evaluate the original cap and later continuous motor-only overrides."""
    schedule = plan.get("plate_force_limit_schedule")
    for override in plan.get("plate_force_limit_overrides", []):
        if t >= float(override["start_time_s"]):
            schedule = override
    if schedule is None:
        return default
    start = float(schedule["start_time_s"])
    duration = float(schedule["ramp_duration_s"])
    initial = float(schedule["initial_force_limit_N"])
    final = float(schedule["final_force_limit_N"])
    if duration <= 0 or not np.isfinite([start, duration, initial, final]).all() or initial <= 0 or final <= 0:
        raise ValueError("Invalid plate force-limit schedule")
    u = float(np.clip((t - start) / duration, 0.0, 1.0))
    blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    return initial + (final - initial) * blend


class SceneVisuals:
    """Render the authored USD surface using the prepared FEM skin bindings."""

    def __init__(self, asset_directory, device, offsets):
        self.root = Path(asset_directory)
        self.offsets = offsets
        self.scene = json.loads((self.root / "scene.json").read_text())
        with np.load(self.root / "skin_bindings.npz", allow_pickle=False) as data:
            self.skins = [
                {key[len(f"toy_{i:02d}_") :]: data[key] for key in data.files if key.startswith(f"toy_{i:02d}_")}
                for i in range(len(offsets) - 1)
            ]
        stage = Usd.Stage.Open(str(self.root / "visuals.usdc"))
        self.meshes = []
        for desc in json.loads((self.root / "visuals.json").read_text()):
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(desc["path"]))
            rest = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            self.meshes.append(
                dict(
                    desc,
                    rest=rest,
                    points=wp.array(rest, dtype=wp.vec3, device=device),
                    indices=wp.array(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), dtype=wp.int32, device=device),
                    color=tuple(mesh.GetDisplayColorAttr().Get()[0]),
                    opacity=float(mesh.GetDisplayOpacityAttr().Get()[0]),
                    uvs=wp.array(np.asarray(uv.Get()), dtype=wp.vec2, device=device) if uv and uv.HasValue() else None,
                )
            )
        self.static_logged = False

    def deform(self, particles):
        result = []
        for i, skin in enumerate(self.skins):
            points = particles[self.offsets[i] : self.offsets[i + 1]].astype(np.float64)
            if "surface_particle_indices" in skin:
                result.append(points[skin["surface_particle_indices"]])
                continue
            corners = points[skin["tets"]]
            current = np.stack([corners[:, j] - corners[:, 0] for j in (1, 2, 3)], axis=-1)
            u, _, vt = np.linalg.svd(current @ skin["inverse_rest"])
            u[:, :, -1] *= np.linalg.det(u @ vt)[:, None]
            rotation = u @ vt
            anchor = np.einsum("vki,vkij->vkj", skin["weights"], points[skin["anchor_indices"]])
            maps = anchor + np.einsum("vkij,vkj->vki", rotation[skin["visual_tet"]], skin["residual"])
            result.append(np.einsum("vk,vki->vi", skin["blend"], maps))
        return result

    def render(self, viewer, state):
        toys = self.deform(state.particle_q.numpy())
        poses = state.body_q.numpy()
        for mesh in self.meshes:
            body, toy = mesh["body"], mesh["toy"]
            if body < 0 and toy < 0 and self.static_logged:
                continue
            if toy >= 0:
                points = toys[toy][mesh["start"] : mesh["start"] + mesh["count"]]
                mesh["points"].assign(points.astype(np.float32))
            elif mesh["path"].startswith("/Scene/carriage_") or mesh["path"].startswith("/Scene/hoist_"):
                points = mesh["rest"] + np.asarray(self.scene["body_centers"])[body]
                offset = np.array([poses[1, 0], poses[1, 1], 0.0])
                if mesh["path"].startswith("/Scene/carriage_"):
                    offset[0] = 0.0
                mesh["points"].assign((points + offset).astype(np.float32))
            elif body >= 0:
                q = poses[body, 3:6]
                cross = np.cross(q, mesh["rest"])
                points = mesh["rest"] + 2 * (poses[body, 6] * cross + np.cross(q, cross)) + poses[body, :3]
                mesh["points"].assign(points.astype(np.float32))
            viewer.log_mesh(
                mesh["path"],
                mesh["points"],
                mesh["indices"],
                color=mesh["color"],
                opacity=mesh["opacity"],
                roughness=mesh["roughness"],
                metallic=mesh["metallic"],
                backface_culling=False,
                uvs=mesh["uvs"],
                texture=str(self.root / mesh["texture"]) if mesh["texture"] else None,
            )
        # Decorative suspension cable: actual XY carriage and Z support, no forces.
        top = np.array([poses[1, 0], poses[1, 1], 5.02 - self.scene["origin"][2]])
        bottom = np.array([poses[2, 0], poses[2, 1], 4.885 - self.scene["origin"][2] + poses[2, 2]])
        direction = bottom - top
        direction /= np.linalg.norm(direction)
        side = np.cross(direction, [0.0, 1.0, 0.0])
        side /= np.linalg.norm(side)
        across = np.cross(direction, side)
        angles = np.arange(12) * 2 * np.pi / 12
        ring = 0.006 * (np.cos(angles)[:, None] * side + np.sin(angles)[:, None] * across)
        points = np.vstack([top + ring, bottom + ring]).astype(np.float32)
        faces = np.array(
            [[i, (i + 1) % 12, 12 + i] for i in range(12)]
            + [[(i + 1) % 12, 12 + (i + 1) % 12, 12 + i] for i in range(12)],
            dtype=np.int32,
        )
        viewer.log_mesh(
            "/Scene/cable",
            wp.array(points, dtype=wp.vec3, device=state.body_q.device),
            wp.array(faces.ravel(), dtype=wp.int32, device=state.body_q.device),
            color=(0.13, 0.14, 0.15),
            roughness=0.35,
            metallic=0.65,
        )
        self.static_logged = True
