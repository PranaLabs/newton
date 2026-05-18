# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example SqueezingBall FBA
#
# A 7129-particle Neo-Hookean tet ball drops between two pairs of rolling
# cylinders (mu=0.5, |omega|=3 rad/s) onto a static plane (mu=0.5).  The
# rolling cylinders friction-drag the ball laterally as it passes through.
# Mirrors the RealSim CudaTests/SqueezingBall reference scene.
#
# Command: uv run -m newton.examples squeezing_ball_fba
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

# RealSim ball mesh; falls back gracefully when the asset is absent.
_MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/ball_volume_7129P.mesh")

# Simulation parameters (CudaTests/SqueezingBall/ball_7k.json).
_DT = 0.01
_PD_ITER = 10  # RealSim's offline binding overrides scene LocalGlobal_CUDA=5 to 10
_NSN_ITER = 1
_GRAVITY = -10.0  # Y-down

_YOUNG = 1.0e4
_POISSON = 0.4
_OBJ_MASS = 1000.0

_BALL_SCALE = 3.0
_BALL_TRANS = np.array([0.0, 2.6, 0.0])

_CYL_RADIUS = 1.0
_CYL_HALF_HEIGHT = 3.0
_FRICTION_MU = 0.5

# (base, axis_world, omega_rad_s) per cylinder, in shape-index order.
_CYLINDERS: list[tuple[np.ndarray, np.ndarray, float]] = [
    (np.array([0.0, -1.0, -1.5]), np.array([1.0, 0.0, 0.0]), -3.0),
    (np.array([0.0, -1.0, +1.5]), np.array([1.0, 0.0, 0.0]), +3.0),
    (np.array([-1.5, -3.5, 0.0]), np.array([0.0, 0.0, 1.0]), +3.0),
    (np.array([+1.5, -3.5, 0.0]), np.array([0.0, 0.0, 1.0]), -3.0),
]


def _load_medit_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Medit ``.mesh`` parser (text format, 1-indexed tets)."""
    vertices: list[tuple[float, float, float]] = []
    tets: list[tuple[int, int, int, int]] = []
    state = None
    n_verts = 0
    n_tets = 0
    vcount = 0
    tcount = 0
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Vertices"):
                state = "vcount"
                continue
            if line.startswith("Tetrahedra"):
                state = "tcount"
                continue
            if line in ("End", "End\n"):
                break
            if line.startswith(("MeshVersionFormatted", "Dimension", "Triangles", "Edges", "Normals")):
                state = "skip" if not line.startswith(("MeshVersionFormatted", "Dimension")) else "header"
                continue
            if state == "vcount":
                n_verts = int(line)
                state = "verts"
                continue
            if state == "tcount":
                n_tets = int(line)
                state = "tets"
                continue
            if state == "verts":
                if vcount < n_verts:
                    parts = line.split()
                    vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    vcount += 1
                    if vcount == n_verts:
                        state = None
                continue
            if state == "tets":
                if tcount < n_tets:
                    parts = line.split()
                    tets.append((int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1))
                    tcount += 1
                    if tcount == n_tets:
                        state = None
                continue
    return np.asarray(vertices, dtype=np.float64), np.asarray(tets, dtype=np.int32)


def _lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def _cyl_xform(base: np.ndarray, axis: np.ndarray) -> wp.transform:
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    v = np.cross(z, a)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, a))
    if s < 1e-9:
        q = wp.quat_identity() if c > 0 else wp.quat(1.0, 0.0, 0.0, 0.0)
    else:
        ang = math.atan2(s, c)
        axis_n = v / s
        half = ang * 0.5
        sh = math.sin(half)
        q = wp.quat(axis_n[0] * sh, axis_n[1] * sh, axis_n[2] * sh, math.cos(half))
    return wp.transform(wp.vec3(*base.tolist()), q)


_BALL_COLOR = (0.55, 0.42, 0.82)  # blue-purple
_CYL_COLOR = (0.55, 0.55, 0.58)  # gray
_MARKER_COLOR = (0.95, 0.30, 0.30)  # red stripe to show cylinder rotation
_MARKER_RADIUS = 0.07
_MARKER_OFFSET_Z = 0.0  # along cylinder axis (mid-length)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = int(round(1.0 / _DT))
        self.frame_dt = _DT
        self.sim_time = 0.0

        if not _MESH_PATH.exists():
            raise FileNotFoundError(
                f"Ball mesh not found at {_MESH_PATH}.  Update _MESH_PATH at the top of this example."
            )
        verts_local, tets = _load_medit_mesh(_MESH_PATH)
        verts_world = verts_local * _BALL_SCALE + _BALL_TRANS

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=_GRAVITY)

        total_vol = 0.0
        for t in tets:
            v0, v1, v2, v3 = (verts_world[i] for i in t)
            total_vol += abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
        density = _OBJ_MASS / max(total_vol, 1e-12)

        mu, lam = _lame_from_young_poisson(_YOUNG, _POISSON)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*v.tolist()) for v in verts_world],
            indices=tets.reshape(-1).tolist(),
            density=density,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )

        # Cylinders (gray) + a red marker sphere attached to each cylinder
        # surface to visualize rotation.  Markers are sites (non-colliding).
        self._cyl_shape_ids: list[int] = []
        self._cyl_init_xforms: list[wp.transform] = []
        self._marker_shape_ids: list[int] = []
        self._cyl_angles: list[float] = []

        for base, axis, _omega in _CYLINDERS:
            xform = _cyl_xform(base, axis)
            cyl_id = builder.add_shape_cylinder(
                body=-1,
                xform=xform,
                radius=_CYL_RADIUS,
                half_height=_CYL_HALF_HEIGHT,
                color=_CYL_COLOR,
            )
            self._cyl_shape_ids.append(cyl_id)
            self._cyl_init_xforms.append(xform)
            self._cyl_angles.append(0.0)

            # Initial marker at θ=0 on cylinder surface (local +X axis, mid-length).
            marker_xform_init = self._marker_xform(xform, 0.0)
            mk_id = builder.add_shape_sphere(
                body=-1,
                xform=marker_xform_init,
                radius=_MARKER_RADIUS,
                color=_MARKER_COLOR,
                as_site=True,
            )
            self._marker_shape_ids.append(mk_id)

        builder.add_shape_plane(
            plane=(0.0, 1.0, 0.0, 10.0),
            body=-1,
            width=0.0,
            length=0.0,
        )

        self.model = builder.finalize()

        # Uniform mass lumping (RealSim Mass.cpp:12-21 parity).
        uniform_mass = _OBJ_MASS / self.model.particle_count
        mass_arr = np.full(self.model.particle_count, uniform_mass, dtype=np.float32)
        inv_mass_arr = np.full(self.model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
        self.model.particle_mass.assign(mass_arr)
        self.model.particle_inv_mass.assign(inv_mass_arr)

        if hasattr(self.model, "shape_material_mu") and self.model.shape_material_mu is not None:
            mu_arr = self.model.shape_material_mu.numpy()
            mu_arr[:] = _FRICTION_MU
            self.model.shape_material_mu.assign(mu_arr.astype(np.float32))

        n_max_contacts = self.model.particle_count
        mu_override = np.full(n_max_contacts, _FRICTION_MU, dtype=np.float64)
        shape_omega = {i: _CYLINDERS[i][2] for i in range(4)}

        self.solver = SolverFBA(
            self.model,
            iterations=_PD_ITER,
            nsn_iterations=_NSN_ITER,
            friction=True,
            stretching_model="neohookean",
            mu=mu,
            lam=lam,
            mu_per_pair_override=mu_override,
            shape_angular_velocity=shape_omega,
            lambda_cap=1.0e12,
            pin_stiffness=1.0e10,
        )

        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.05)
        self.contacts = self.pipeline.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Suppress default tri mesh; we'll log the ball ourselves with a
        # custom blue-purple color via ``log_mesh`` in ``render``.
        self.viewer.show_triangles = False
        self.viewer.set_model(self.model)

        # Camera centered on the squeeze zone.  Cylinders sit at y in
        # [-1, -3.5]; the ball drops from y≈2.6 through the gap into the
        # plane at y=-10, so the centroid of activity is around y≈-3.
        # ``set_camera(pos, pitch, yaw)`` — pitch tilts down from horizon,
        # yaw rotates around the world up axis.
        self.viewer.set_camera(wp.vec3(6.0, 0.0, 6.0), pitch=-15.0, yaw=-45.0)

        # Cache the initial shape_transform array as a NumPy buffer; per-step
        # rotation updates write into a copy of this and assign in bulk.
        self._shape_transform_init_np = self.model.shape_transform.numpy().copy()

        # Cache ball-mesh data for the custom blue-purple render.
        self._ball_tri_indices = self.model.tri_indices.flatten()

    @staticmethod
    def _marker_xform(cyl_xform: wp.transform, angle: float) -> wp.transform:
        """Marker position: at θ on the cylinder surface (local +X rotated by
        angle about local +Z), expressed in world frame."""
        base = np.asarray(cyl_xform.p, dtype=np.float64)
        q = np.asarray(cyl_xform.q, dtype=np.float64)  # (qx, qy, qz, qw)
        # Radial point in cylinder-local frame, at half-length mid-Z.
        rx_local = _CYL_RADIUS * math.cos(angle)
        ry_local = _CYL_RADIUS * math.sin(angle)
        rz_local = _MARKER_OFFSET_Z
        local = np.array([rx_local, ry_local, rz_local], dtype=np.float64)
        # Rotate local → world using cyl_xform quaternion.
        qx, qy, qz, qw = q
        # Standard quaternion-vector rotation.
        t = 2.0 * np.cross([qx, qy, qz], local)
        world_rel = local + qw * t + np.cross([qx, qy, qz], t)
        world_pos = base + world_rel
        return wp.transform(wp.vec3(*world_pos.tolist()), wp.quat_identity())

    def _advance_cylinder_rotation(self) -> None:
        """Update cylinder shape_transforms (rotation about local +Z) and
        marker shape_transforms (position on cylinder surface)."""
        new_xforms = self._shape_transform_init_np.copy()
        for i, (_base, _axis, omega) in enumerate(_CYLINDERS):
            self._cyl_angles[i] += omega * self.frame_dt
            angle = self._cyl_angles[i]
            cyl_xform = self._cyl_init_xforms[i]

            # Rotate the cylinder's quaternion by ``angle`` about local +Z.
            # cos(θ/2) + sin(θ/2)·k.  Compose with the cylinder's init quat.
            half = angle * 0.5
            sh = math.sin(half)
            local_rot = np.array([0.0, 0.0, sh, math.cos(half)], dtype=np.float64)
            base_q = np.asarray(cyl_xform.q, dtype=np.float64)
            # Quaternion multiply: q_new = base_q ∘ local_rot.
            bx, by, bz, bw = base_q
            lx, ly, lz, lw = local_rot
            new_q = np.array(
                [
                    bw * lx + bx * lw + by * lz - bz * ly,
                    bw * ly - bx * lz + by * lw + bz * lx,
                    bw * lz + bx * ly - by * lx + bz * lw,
                    bw * lw - bx * lx - by * ly - bz * lz,
                ],
                dtype=np.float64,
            )
            cyl_id = self._cyl_shape_ids[i]
            new_xforms[cyl_id][:3] = np.asarray(cyl_xform.p, dtype=np.float64)
            new_xforms[cyl_id][3:] = new_q

            # Marker — position rotates with angle on cylinder surface.
            marker_xform = self._marker_xform(cyl_xform, angle)
            mk_id = self._marker_shape_ids[i]
            new_xforms[mk_id][:3] = np.asarray(marker_xform.p, dtype=np.float64)
            new_xforms[mk_id][3:] = np.asarray(marker_xform.q, dtype=np.float64)
        self.model.shape_transform.assign(new_xforms.astype(np.float32))

    def step(self):
        self._advance_cylinder_rotation()
        self.state_0.clear_forces()
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, None, self.contacts, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Custom-color ball mesh (overrides the suppressed default).
        self.viewer.log_mesh(
            "/ball",
            self.state_0.particle_q,
            self._ball_tri_indices,
            color=_BALL_COLOR,
            backface_culling=False,
        )
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(q)), "non-finite particle positions"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
