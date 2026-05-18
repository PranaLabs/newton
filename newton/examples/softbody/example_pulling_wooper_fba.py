# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PullingWooper FBA
#
# A 5325-particle Neo-Hookean tet "wooper" pinned at one arm and pulled
# downward at 2 m/s through two static cylinders (no friction).  Mirrors
# the RealSim CudaTests/PullingWooper reference scene.  Cylinders use the
# same checker-textured-mesh visual recipe as example_squeezing_ball_fba.
#
# Command: uv run -m newton.examples pulling_wooper_fba
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

_MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/wooper_volume_5325P.mesh")

_DT = 0.01
_PD_ITER = 10
_NSN_ITER = 1
_GRAVITY = 0.0  # PullingWooper scene.json sets gravity = 0

_YOUNG = 1.0e7
_POISSON = 0.3
_OBJ_MASS = 1000.0

# Wooper transform: rotation Z=-90 deg, translation (0, 4, 0).
_ROT_Z_DEG = -90.0
_TRANS = np.array([0.0, 4.0, 0.0])

# Pin AABB (WORLD coords) — matches RealSim wooper_5k.json semantics.
_PIN_LO = (-2.5, 2.0, -0.3)
_PIN_HI = (-1.5, 4.0, 0.3)
_PIN_DIR = np.array([0.0, -1.0, 0.0])  # pull -Y at 2 m/s, cap 10 m total
_PIN_VEL = 2.0
_PIN_MAXLENGTH = 10.0

_CYL_RADIUS = 1.0
_CYL_HALF_HEIGHT = 3.0
_CYL_VISUAL_SHRINK = 0.97
_CYL0_BASE = np.array([0.0, -1.0, -1.3])
_CYL1_BASE = np.array([0.0, -1.0, +1.3])
_CYL_AXIS = np.array([1.0, 0.0, 0.0])

_WOOPER_COLOR = (0.45, 0.75, 0.88)  # light blue
_CYL_COLOR = (0.55, 0.55, 0.58)
_CHECKER_DARK = (0.18, 0.18, 0.22)
_CHECKER_LIGHT = (0.85, 0.85, 0.88)
_CHECKER_U_TILES = 12.0
_CHECKER_V_TILES = 12.0
_CYL_MESH_U_SEGMENTS = 64


def _load_medit_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Medit ``.mesh`` parser (text format, 1-indexed tets)."""
    vertices: list[tuple[float, float, float]] = []
    tets: list[tuple[int, int, int, int]] = []
    state = None
    n_verts = n_tets = vcount = tcount = 0
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
                n_verts = int(line); state = "verts"; continue
            if state == "tcount":
                n_tets = int(line); state = "tets"; continue
            if state == "verts":
                if vcount < n_verts:
                    p = line.split()
                    vertices.append((float(p[0]), float(p[1]), float(p[2])))
                    vcount += 1
                    if vcount == n_verts:
                        state = None
                continue
            if state == "tets":
                if tcount < n_tets:
                    p = line.split()
                    tets.append((int(p[0]) - 1, int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1))
                    tcount += 1
                    if tcount == n_tets:
                        state = None
                continue
    return np.asarray(vertices, dtype=np.float64), np.asarray(tets, dtype=np.int32)


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def _cyl_xform(base: np.ndarray, axis: np.ndarray) -> wp.transform:
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    v = np.cross(z, a); s = float(np.linalg.norm(v)); c = float(np.dot(z, a))
    if s < 1e-9:
        q = wp.quat_identity() if c > 0 else wp.quat(1.0, 0.0, 0.0, 0.0)
    else:
        ang = math.atan2(s, c); axis_n = v / s
        half = ang * 0.5; sh = math.sin(half)
        q = wp.quat(axis_n[0] * sh, axis_n[1] * sh, axis_n[2] * sh, math.cos(half))
    return wp.transform(wp.vec3(*base.tolist()), q)


def _select_in_aabb(points: np.ndarray, lo: tuple, hi: tuple) -> np.ndarray:
    p = np.asarray(points)
    lo = np.asarray(lo, dtype=p.dtype); hi = np.asarray(hi, dtype=p.dtype)
    mask = np.all((p >= lo) & (p <= hi), axis=1)
    return np.where(mask)[0].astype(np.int32)


def _make_cylinder_mesh(
    radius: float, half_height: float, u_segments: int, u_tiles: float, v_tiles: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-aligned cylinder triangle mesh with UVs tiled (u_tiles x v_tiles).

    Endcap vertices map to a single UV inside a dark cell so caps render
    as solid dark; side-wall seam is duplicated so GL_REPEAT wraps cleanly.
    """
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []
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
    cap_uv = (0.25, 0.25)
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
                f"Wooper mesh not found at {_MESH_PATH}.  Update _MESH_PATH at the top of this example."
            )
        verts_local, tets = _load_medit_mesh(_MESH_PATH)
        verts_world = verts_local @ _rot_z(_ROT_Z_DEG).T + _TRANS

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

        for base in (_CYL0_BASE, _CYL1_BASE):
            xform = _cyl_xform(base, _CYL_AXIS)
            cfg_collision = builder.default_shape_cfg.copy()
            cfg_collision.is_visible = False
            builder.add_shape_cylinder(
                body=-1,
                xform=xform,
                radius=_CYL_RADIUS,
                half_height=_CYL_HALF_HEIGHT,
                cfg=cfg_collision,
            )
            cfg_visual = builder.default_site_cfg.copy()
            builder.add_shape_mesh(
                body=-1,
                xform=xform,
                mesh=cyl_visual_mesh,
                cfg=cfg_visual,
                color=_CYL_COLOR,
            )

        pin_idx = _select_in_aabb(verts_world, _PIN_LO, _PIN_HI)

        self.model = builder.finalize()

        uniform_mass = _OBJ_MASS / self.model.particle_count
        self.model.particle_mass.assign(
            np.full(self.model.particle_count, uniform_mass, dtype=np.float32)
        )
        inv = np.full(self.model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
        inv[pin_idx] = 0.0  # pin: inv_mass = 0
        self.model.particle_inv_mass.assign(inv)

        self.solver = SolverFBA(
            self.model,
            iterations=_PD_ITER,
            nsn_iterations=_NSN_ITER,
            friction=False,
            stretching_model="neohookean",
            mu=mu,
            lam=lam,
            lambda_cap=1.0e12,
            pin_stiffness=1.0e10,
        )

        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.05)
        self.contacts = self.pipeline.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self._pin_idx = pin_idx
        self._pin_init = verts_world[pin_idx].copy()
        self._targets = verts_world.copy().astype(np.float32)
        self._pulled = 0.0

        # The default soft-body triangle render is suppressed; we draw the
        # wooper ourselves below with a custom color.
        self.viewer.show_triangles = False
        self.viewer.set_model(self.model)
        self._wooper_tri_indices = self.model.tri_indices.flatten()

        # Camera setup must run AFTER set_model since set_model resets
        # camera framing to fit the new scene.
        if hasattr(self.viewer, "camera"):
            cam_pos = wp.vec3(10.0, 3.0, 12.0)
            cam_target = wp.vec3(0.0, -1.0, 0.0)
            self.viewer.camera.pos = self.viewer.camera._as_vec3(cam_pos)
            self.viewer.camera.look_at(cam_target)

    def step(self):
        delta = _PIN_VEL * self.frame_dt
        if self._pulled + delta > _PIN_MAXLENGTH:
            delta = max(0.0, _PIN_MAXLENGTH - self._pulled)
        self._pulled += delta
        self._targets[self._pin_idx] = self._pin_init + (_PIN_DIR * self._pulled).astype(np.float32)
        self.solver.set_pin_targets(self._targets)

        self.state_0.clear_forces()
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, None, self.contacts, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/wooper",
            self.state_0.particle_q,
            self._wooper_tri_indices,
            color=_WOOPER_COLOR,
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
