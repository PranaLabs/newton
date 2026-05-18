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
_CYL_HALF_HEIGHT = 3.6  # 1.2x of the CudaTests reference 3.0 (visual extent only)
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
_STRIPE_COLOR_A = (0.20, 0.20, 0.22)  # dark stripe
_STRIPE_COLOR_B = (0.85, 0.85, 0.88)  # light stripe
# Longitudinal stripe boxes attached to each cylinder surface.  We place
# ``_N_STRIPES`` evenly around the cylinder at +radius offset; their
# shape_transform is updated per step so they ride with the cylinder
# rotation, producing a visible grid/striped texture.
_N_STRIPES = 8
_STRIPE_THICKNESS = 0.04  # radial half-extent (sticks slightly proud of cylinder surface)
_STRIPE_HALF_WIDTH = 0.06  # tangential half-extent
# Transverse ring marker rings at +/- 0.6 h_half along the axis, to give a
# 2D-grid feel rather than just longitudinal lines.
_N_RING_MARKERS = 6
_RING_OFFSETS = (-0.66, 0.0, 0.66)  # multiples of half-height
_RING_MARKER_RADIUS = 0.07


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

        # Cylinders (gray) + a grid of small site-shapes attached to each
        # cylinder surface to visualize rotation.  Per cylinder we add:
        #   - ``_N_STRIPES`` longitudinal stripe-boxes alternating dark/light
        #     placed around the circumference at radius offset,
        #   - 3 transverse marker-sphere rings at ±0.66·half_height and mid.
        # All markers are sites (non-colliding) and their shape_transforms
        # are advanced by ``omega * dt`` each step in ``_advance_cylinder_rotation``.
        self._cyl_shape_ids: list[int] = []
        self._cyl_init_xforms: list[wp.transform] = []
        self._cyl_angles: list[float] = []
        # Per-cylinder lists of (shape_id, base_angle, axis_z_offset, kind),
        # kind in {"stripe", "marker"}.  ``base_angle`` is the local angle
        # at sim time 0; runtime angle = base_angle + cyl_angle[i].
        self._marker_data: list[list[tuple[int, float, float, str]]] = []

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

            per_cyl: list[tuple[int, float, float, str]] = []

            # Longitudinal stripes (alternating dark/light) around the circumference.
            for k in range(_N_STRIPES):
                theta = k * (2.0 * math.pi / _N_STRIPES)
                color = _STRIPE_COLOR_A if (k % 2 == 0) else _STRIPE_COLOR_B
                stripe_xf = self._stripe_xform(xform, theta, 0.0)
                # Long thin box: along cylinder axis = local Z = world axis-direction.
                # half_extents in box-local frame: (tangential, radial, axial)
                # We build the orientation so the box's long axis follows the cylinder's Z.
                sid = builder.add_shape_box(
                    body=-1,
                    xform=stripe_xf,
                    hx=_STRIPE_HALF_WIDTH,
                    hy=_STRIPE_THICKNESS,
                    hz=_CYL_HALF_HEIGHT * 0.98,
                    color=color,
                    as_site=True,
                )
                per_cyl.append((sid, theta, 0.0, "stripe"))

            # Transverse ring markers at three z offsets, ``_N_RING_MARKERS``
            # per ring, alternating colors for visibility.
            for z_mult in _RING_OFFSETS:
                z_off = z_mult * _CYL_HALF_HEIGHT
                for k in range(_N_RING_MARKERS):
                    theta = (k + 0.5) * (2.0 * math.pi / _N_RING_MARKERS)
                    color = _STRIPE_COLOR_B if (k % 2 == 0) else _STRIPE_COLOR_A
                    marker_xf = self._marker_xform(xform, theta, z_off)
                    mid = builder.add_shape_sphere(
                        body=-1,
                        xform=marker_xf,
                        radius=_RING_MARKER_RADIUS,
                        color=color,
                        as_site=True,
                    )
                    per_cyl.append((mid, theta, z_off, "marker"))

            self._marker_data.append(per_cyl)

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

        # Camera framed on the squeeze zone.  Cylinders sit at y in
        # [-1, -3.5]; the ball drops from y≈2.6 through the gap into the
        # plane at y=-10, so the centroid of activity is around y≈-3.
        # Use ``camera.look_at`` (sets pitch+yaw consistently for the
        # requested target) so the scene shows up centered on first frame.
        if hasattr(self.viewer, "camera"):
            cam_pos = wp.vec3(13.0, 3.0, 13.0)
            cam_target = wp.vec3(0.0, -3.0, 0.0)
            self.viewer.camera.pos = self.viewer.camera._as_vec3(cam_pos)
            self.viewer.camera.look_at(cam_target)

        # Cache the initial shape_transform array as a NumPy buffer; per-step
        # rotation updates write into a copy of this and assign in bulk.
        self._shape_transform_init_np = self.model.shape_transform.numpy().copy()

        # Cache ball-mesh data for the custom blue-purple render.
        self._ball_tri_indices = self.model.tri_indices.flatten()

    @staticmethod
    def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector ``v`` by quaternion ``q = (qx, qy, qz, qw)``."""
        qx, qy, qz, qw = q
        t = 2.0 * np.cross([qx, qy, qz], v)
        return v + qw * t + np.cross([qx, qy, qz], t)

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Hamilton product of two quats in (x, y, z, w) order: ``a ∘ b``."""
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _marker_xform(cyl_xform: wp.transform, angle: float, z_offset: float = 0.0) -> wp.transform:
        """Sphere marker on the cylinder surface at local angle θ and axial
        offset (along the cylinder's local +Z).  Returns world-space pose."""
        base = np.asarray(cyl_xform.p, dtype=np.float64)
        q = np.asarray(cyl_xform.q, dtype=np.float64)
        local = np.array(
            [_CYL_RADIUS * math.cos(angle), _CYL_RADIUS * math.sin(angle), z_offset],
            dtype=np.float64,
        )
        world_pos = base + Example._quat_rotate_vec(q, local)
        return wp.transform(wp.vec3(*world_pos.tolist()), wp.quat_identity())

    @staticmethod
    def _stripe_xform(cyl_xform: wp.transform, angle: float, z_offset: float) -> wp.transform:
        """Longitudinal stripe-box pose: positioned at cylinder surface at
        angle θ (radially outward), oriented so its long axis follows the
        cylinder's local +Z."""
        base = np.asarray(cyl_xform.p, dtype=np.float64)
        q_cyl = np.asarray(cyl_xform.q, dtype=np.float64)
        # Position: radius·(cosθ, sinθ, 0) in cylinder-local; rotate to world.
        local_pos = np.array(
            [
                (_CYL_RADIUS + _STRIPE_THICKNESS) * math.cos(angle),
                (_CYL_RADIUS + _STRIPE_THICKNESS) * math.sin(angle),
                z_offset,
            ],
            dtype=np.float64,
        )
        world_pos = base + Example._quat_rotate_vec(q_cyl, local_pos)
        # Orientation: the box's local axes need to be aligned so that:
        #   box_x = radial direction (cosθ, sinθ, 0 in cyl-local)
        #   box_y = perpendicular to radial in the cyl-local XY plane
        #   box_z = cylinder's local +Z (axial)
        # Compose: world_quat = q_cyl ∘ q_local_angle_about_Z
        half = angle * 0.5
        sh = math.sin(half)
        q_local = np.array([0.0, 0.0, sh, math.cos(half)], dtype=np.float64)
        q_world = Example._quat_mul(q_cyl, q_local)
        return wp.transform(wp.vec3(*world_pos.tolist()), wp.quat(*q_world.tolist()))

    def _advance_cylinder_rotation(self) -> None:
        """Update cylinder shape_transforms (rotation about local +Z) and
        marker / stripe shape_transforms (positions on cylinder surface).

        Sign convention: SolverFBA's friction kinematics use
        ``v_anchor = -omega * cross(axis_world, r_local)`` (see
        ``solver_fba.py``).  Substituting this into ``v_surface = w_world x r``
        gives ``w_world = -omega * axis_world``, i.e. positive ``omega`` in
        the ``_CYLINDERS`` table corresponds to a *negative* world-frame
        angular velocity about the cylinder axis.  The visual rotation must
        therefore decrement the angle by ``omega * dt`` (negating) so the
        cylinder surface spin matches the direction the ball experiences
        through friction.
        """
        new_xforms = self._shape_transform_init_np.copy()
        for i, (_base, _axis, omega) in enumerate(_CYLINDERS):
            self._cyl_angles[i] -= omega * self.frame_dt
            angle = self._cyl_angles[i]
            cyl_xform = self._cyl_init_xforms[i]

            # Cylinder body itself: rotate quat about local +Z by ``angle``.
            half = angle * 0.5
            sh = math.sin(half)
            q_local = np.array([0.0, 0.0, sh, math.cos(half)], dtype=np.float64)
            base_q = np.asarray(cyl_xform.q, dtype=np.float64)
            new_q = self._quat_mul(base_q, q_local)
            cyl_id = self._cyl_shape_ids[i]
            new_xforms[cyl_id][:3] = np.asarray(cyl_xform.p, dtype=np.float64)
            new_xforms[cyl_id][3:] = new_q

            # Stripes + ring markers move with the cylinder rotation.
            for sid, base_angle, z_off, kind in self._marker_data[i]:
                theta = base_angle + angle
                if kind == "stripe":
                    xf = self._stripe_xform(cyl_xform, theta, z_off)
                else:
                    xf = self._marker_xform(cyl_xform, theta, z_off)
                new_xforms[sid][:3] = np.asarray(xf.p, dtype=np.float64)
                new_xforms[sid][3:] = np.asarray(xf.q, dtype=np.float64)
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
