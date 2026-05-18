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
_CYL_HALF_HEIGHT = 3.6  # 1.2x of the CudaTests reference 3.0
# Multiplicative visual shrink applied only to the cylinder's render shape,
# so the gray body sits slightly inside the stripe/marker grid (radius 1.0)
# and the grid reads as ``on the surface''. Collision uses a separate full-size
# invisible cylinder, so this shrink does NOT change physics.
_CYL_VISUAL_SHRINK = 0.97
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
_CYL_COLOR = (0.55, 0.55, 0.58)  # gray (modulates the checker texture)
_CHECKER_DARK = (0.18, 0.18, 0.22)
_CHECKER_LIGHT = (0.85, 0.85, 0.88)
# Checker tiling along U (circumference) and V (axial); chosen so squares
# look roughly uniform at radius=1, half_height=3.6.
_CHECKER_U_TILES = 12.0
_CHECKER_V_TILES = 14.0
_CYL_MESH_U_SEGMENTS = 64  # azimuthal triangulation of the visual cylinder mesh


def _make_cylinder_mesh(
    radius: float, half_height: float, u_segments: int, u_tiles: float, v_tiles: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-aligned cylinder triangle mesh with UVs tiled (u_tiles x v_tiles).

    Endcap vertices are pinned to a single UV inside a dark cell so the caps
    render as solid dark, leaving the checker pattern only on the side wall.
    Side-wall triangulation duplicates the seam (i=0 and i=u_segments share a
    position but have distinct U coords) so GL_REPEAT wrapping is clean.
    """
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []
    # Side wall — two rings at z = -h and +h.
    for j, z in enumerate((-half_height, +half_height)):
        for i in range(u_segments + 1):
            theta = i * (2.0 * math.pi / u_segments)
            verts.append((radius * math.cos(theta), radius * math.sin(theta), z))
            uvs.append(((i / u_segments) * u_tiles, j * v_tiles))
    ring_size = u_segments + 1
    for i in range(u_segments):
        a, b = i, i + 1
        c, d = ring_size + i, ring_size + i + 1
        tris.append((a, b, d))
        tris.append((a, d, c))
    # Endcaps — each cap gets a center vertex + duplicated ring with cap UV.
    cap_uv = (0.25, 0.25)  # inside a dark cell of the 2x2 checker
    for sign, ccw_swap in ((-1.0, True), (+1.0, False)):
        center = len(verts)
        verts.append((0.0, 0.0, sign * half_height))
        uvs.append(cap_uv)
        ring_start = len(verts)
        for i in range(u_segments + 1):
            theta = i * (2.0 * math.pi / u_segments)
            verts.append((radius * math.cos(theta), radius * math.sin(theta), sign * half_height))
            uvs.append(cap_uv)
        for i in range(u_segments):
            a, b = ring_start + i, ring_start + i + 1
            tris.append((center, b, a) if ccw_swap else (center, a, b))
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(tris, dtype=np.int32).reshape(-1),
        np.asarray(uvs, dtype=np.float32),
    )


def _make_checker_texture(size: int = 64) -> np.ndarray:
    """2x2 checker RGB uint8 texture; GL_REPEAT tiles it across UVs."""
    img = np.empty((size, size, 3), dtype=np.uint8)
    half = size // 2
    dark = np.array([int(255 * c) for c in _CHECKER_DARK], dtype=np.uint8)
    light = np.array([int(255 * c) for c in _CHECKER_LIGHT], dtype=np.uint8)
    img[:half, :half] = dark
    img[:half, half:] = light
    img[half:, :half] = light
    img[half:, half:] = dark
    return img


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

        # Each cylinder is two shapes:
        #   - collision: full-size, invisible primitive (drives physics + NSN
        #     friction kinematics via ``shape_angular_velocity``),
        #   - visual: UV-mapped triangle mesh as a site (non-colliding),
        #     with a tiled 2x2 checker texture so the rotation is visible.
        # The visual mesh is shrunk by ``_CYL_VISUAL_SHRINK`` so it sits just
        # inside the collision surface — collision unaffected.
        self._cyl_shape_ids: list[int] = []  # visual mesh shape ids (rotate per step)
        self._cyl_collision_ids: list[int] = []  # collision-cylinder shape ids
        self._cyl_init_xforms: list[wp.transform] = []
        self._cyl_angles: list[float] = []

        cyl_visual_verts, cyl_visual_idx, cyl_visual_uvs = _make_cylinder_mesh(
            _CYL_RADIUS * _CYL_VISUAL_SHRINK,
            _CYL_HALF_HEIGHT * _CYL_VISUAL_SHRINK,
            _CYL_MESH_U_SEGMENTS,
            _CHECKER_U_TILES,
            _CHECKER_V_TILES,
        )
        cyl_visual_mesh = newton.Mesh(
            vertices=cyl_visual_verts,
            indices=cyl_visual_idx,
            uvs=cyl_visual_uvs,
            compute_inertia=False,
            color=_CYL_COLOR,
            texture=_make_checker_texture(),
        )

        for base, axis, _omega in _CYLINDERS:
            xform = _cyl_xform(base, axis)

            cfg_collision = builder.default_shape_cfg.copy()
            cfg_collision.is_visible = False
            coll_id = builder.add_shape_cylinder(
                body=-1,
                xform=xform,
                radius=_CYL_RADIUS,
                half_height=_CYL_HALF_HEIGHT,
                cfg=cfg_collision,
            )
            self._cyl_collision_ids.append(coll_id)

            cfg_visual = builder.default_site_cfg.copy()
            vis_id = builder.add_shape_mesh(
                body=-1,
                xform=xform,
                mesh=cyl_visual_mesh,
                cfg=cfg_visual,
                color=_CYL_COLOR,
            )
            self._cyl_shape_ids.append(vis_id)
            self._cyl_init_xforms.append(xform)
            self._cyl_angles.append(0.0)

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
        shape_omega = {
            self._cyl_collision_ids[i]: _CYLINDERS[i][2] for i in range(len(_CYLINDERS))
        }

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

    def _advance_cylinder_rotation(self) -> None:
        """Update each visual-cylinder mesh's shape_transform to spin about
        its local +Z axis.  Collision cylinders stay at their initial xform
        — they're rotationally symmetric so this is invariant to physics,
        and ``shape_angular_velocity`` carries the kinematic surface speed
        to NSN's friction step directly.

        Sign convention: SolverFBA friction uses
        ``v_anchor = -omega * cross(axis_world, r_local)``, which corresponds
        to a *negative* world-frame angular velocity along ``axis_world``
        when ``omega > 0``.  The visual rotation therefore decrements the
        angle by ``omega * dt``.
        """
        new_xforms = self._shape_transform_init_np.copy()
        for i, (_base, _axis, omega) in enumerate(_CYLINDERS):
            self._cyl_angles[i] -= omega * self.frame_dt
            angle = self._cyl_angles[i]
            cyl_xform = self._cyl_init_xforms[i]
            half = angle * 0.5
            q_local = np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)
            base_q = np.asarray(cyl_xform.q, dtype=np.float64)
            new_q = self._quat_mul(base_q, q_local)
            vis_id = self._cyl_shape_ids[i]
            new_xforms[vis_id][:3] = np.asarray(cyl_xform.p, dtype=np.float64)
            new_xforms[vis_id][3:] = new_q
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
