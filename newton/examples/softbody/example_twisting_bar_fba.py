# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example TwistingBar (NH) FBA
#
# An 11340-particle Neo-Hookean tet cube (E=1e9, nu=0.45) twisted by two
# ROLLING pin caps rotating about +/-Y at 10 deg/s, capped at +/-90 deg.
# 810 frames at dt=0.01.  Mirrors RealSim CudaTests/TwistingBarNH.
#
# Command: uv run -m newton.examples twisting_bar_fba
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

_MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/cube_volume_11340P.mesh")

_DT = 0.01
_PD_ITER = 10  # RealSim's offline binding overrides scene LocalGlobal_CUDA=5 to 10
_GRAVITY = 0.0
_TOTAL_MASS = 1000.0

_YOUNG = 1.0e9
_NU = 0.45

_PIN_AVEL_DEG_PER_S = 10.0
_PIN_AVEL = math.radians(_PIN_AVEL_DEG_PER_S)
_MAX_ANGLE = math.pi / 2.0  # 90 deg
_PIN_Y_TOP = 1.99
_PIN_Y_BOT = -1.99

_BAR_COLOR = (0.93, 0.55, 0.31)  # warm orange (modulates the checker)
_CHECKER_DARK = (0.18, 0.18, 0.22)
_CHECKER_LIGHT = (0.85, 0.85, 0.88)
# Cylindrical UV around Y: U = atan2(x, z) tiled _CHECKER_U_TILES times,
# V = (y + 2) / 4 tiled _CHECKER_V_TILES times.  Twisting deforms the
# surface around the Y axis, so the checker rotates with it.
_CHECKER_U_TILES = 8.0
_CHECKER_V_TILES = 8.0


def _make_checker_texture(size: int = 64) -> np.ndarray:
    img = np.empty((size, size, 3), dtype=np.uint8)
    half = size // 2
    dark = np.array([int(255 * c) for c in _CHECKER_DARK], dtype=np.uint8)
    light = np.array([int(255 * c) for c in _CHECKER_LIGHT], dtype=np.uint8)
    img[:half, :half] = dark; img[:half, half:] = light
    img[half:, :half] = light; img[half:, half:] = dark
    return img


def _load_medit_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Medit ``.mesh`` parser (text format, 1-indexed tets)."""
    vertices: list[tuple[float, float, float]] = []
    tets: list[tuple[int, int, int, int]] = []
    state = None
    n_verts = n_tets = vc = tc = 0
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Vertices"):
                state = "vc"; continue
            if line.startswith("Tetrahedra"):
                state = "tc"; continue
            if line in ("End", "End\n"):
                break
            if line.startswith(("MeshVersionFormatted", "Dimension", "Triangles", "Edges", "Normals")):
                state = "skip" if not line.startswith(("MeshVersionFormatted", "Dimension")) else "h"
                continue
            if state == "vc":
                n_verts = int(line); state = "v"; continue
            if state == "tc":
                n_tets = int(line); state = "t"; continue
            if state == "v":
                if vc < n_verts:
                    p = line.split()
                    vertices.append((float(p[0]), float(p[1]), float(p[2])))
                    vc += 1
                    if vc == n_verts:
                        state = None
                continue
            if state == "t":
                if tc < n_tets:
                    p = line.split()
                    tets.append((int(p[0]) - 1, int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1))
                    tc += 1
                    if tc == n_tets:
                        state = None
                continue
    return np.asarray(vertices, dtype=np.float64), np.asarray(tets, dtype=np.int32)


def _lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = int(round(1.0 / _DT))
        self.frame_dt = _DT
        self.sim_time = 0.0

        if not _MESH_PATH.exists():
            raise FileNotFoundError(
                f"Cube mesh not found at {_MESH_PATH}.  Update _MESH_PATH at the top of this example."
            )
        verts, tets = _load_medit_mesh(_MESH_PATH)
        n_verts = len(verts)

        # Pin caps along +/-Y edges of the bar.
        y = verts[:, 1]
        top_pins = np.where(y > _PIN_Y_TOP)[0].astype(np.int32)
        bot_pins = np.where(y < _PIN_Y_BOT)[0].astype(np.int32)

        mu, lam = _lame_from_young_poisson(_YOUNG, _NU)

        # Density is overridden by uniform mass lumping below; the value
        # only matters for builder bookkeeping.
        total_vol = 0.0
        for t in tets:
            v0, v1, v2, v3 = (verts[i] for i in t)
            total_vol += abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
        density = _TOTAL_MASS / max(total_vol, 1e-12)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=_GRAVITY)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*v.tolist()) for v in verts],
            indices=tets.reshape(-1).tolist(),
            density=density,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )

        self.model = builder.finalize()

        # Uniform mass lumping per RealSim Mass.cpp:12-21.
        uniform_mass = _TOTAL_MASS / self.model.particle_count
        self.model.particle_mass.assign(
            np.full(self.model.particle_count, uniform_mass, dtype=np.float32)
        )
        inv = np.full(self.model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
        inv[top_pins] = 0.0
        inv[bot_pins] = 0.0
        self.model.particle_inv_mass.assign(inv)

        self.solver = SolverFBA(
            self.model,
            iterations=_PD_ITER,
            pin_stiffness=1.0e12,
            stretching_model="neohookean",
            mu=mu,
            lam=lam,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self._top_pins = top_pins
        self._bot_pins = bot_pins
        self._verts_init = verts.astype(np.float64).copy()
        self._frame = 0

        self.viewer.show_triangles = False
        self.viewer.set_model(self.model)
        self._bar_tri_indices = self.model.tri_indices.flatten()

        # Cylindrical UV from rest positions: U wraps around the Y axis,
        # V runs along Y.  The -Z face hosts the wrap seam; it sits behind
        # the camera at the default framing so the seam stays out of view.
        uv_np = np.empty((self.model.particle_count, 2), dtype=np.float32)
        xs = verts[:, 0]
        ys = verts[:, 1]
        zs = verts[:, 2]
        uv_np[:, 0] = ((np.arctan2(xs, zs) / (2.0 * math.pi)) + 0.5) * _CHECKER_U_TILES
        uv_np[:, 1] = ((ys + 2.0) / 4.0) * _CHECKER_V_TILES
        self._bar_uvs = wp.array(uv_np, dtype=wp.vec2)
        self._bar_texture = _make_checker_texture()

        # Camera framing the +Y-up bar from the front-right.
        if hasattr(self.viewer, "camera"):
            cam_pos = wp.vec3(4.5, 1.5, 4.5)
            cam_target = wp.vec3(0.0, 0.0, 0.0)
            self.viewer.camera.pos = self.viewer.camera._as_vec3(cam_pos)
            self.viewer.camera.look_at(cam_target)

    def step(self):
        # Advance to "after frame f+1" (RealSim 1-indexed convention):
        # ROLLING pin angle = avel * (frame+1) * dt, clamped to MAX_ANGLE.
        f = self._frame + 1
        top_angle = min(_PIN_AVEL * f * self.frame_dt, _MAX_ANGLE)
        bot_angle = max(-_PIN_AVEL * f * self.frame_dt, -_MAX_ANGLE)

        R_top = _rot_y(top_angle)
        R_bot = _rot_y(bot_angle)

        x_ref = self.state_0.particle_q.numpy().copy().astype(np.float64)
        x_ref[self._top_pins] = self._verts_init[self._top_pins] @ R_top.T
        x_ref[self._bot_pins] = self._verts_init[self._bot_pins] @ R_bot.T
        self.solver.set_pin_targets(x_ref.astype(np.float32))

        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt
        self._frame = f

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/bar",
            self.state_0.particle_q,
            self._bar_tri_indices,
            uvs=self._bar_uvs,
            texture=self._bar_texture,
            color=_BAR_COLOR,
            backface_culling=False,
        )
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(q)), "non-finite particle positions"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
