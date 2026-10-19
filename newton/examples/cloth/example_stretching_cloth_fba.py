# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example StretchingCloth FBA
#
# A 20201-particle Neo-Hookean cloth pulled along +/-x by two pin sets on
# opposing edges, ~1 m total per side over 1200 steps at dt=0.01.  Mirrors
# the RealSim CudaTests/StretchingCloth reference scene.  The cloth is
# rendered with a tiled 2x2 checker texture so the stretching is visible.
#
# Command: uv run -m newton.examples stretching_cloth_fba
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

_DT = 0.01
_PD_ITER = 10
_GRAVITY = -1.0  # Y-up, RealSim scene.json gravity=[0,-1,0]

_YOUNG = 35483.87096774193  # RealSim's material modulus; SolverFBA's tri gauge scales it back to 1e5
_POISSON = 0.45
_OBJ_MASS = 1000.0

_ROT_X_DEG = 90.0
_TRANS = np.array([0.0, 0.0, 0.0])

# Pin AABBs (world coords after Rx(90)).
_PIN0_LO = (0.99, -0.1, -1.1)
_PIN0_HI = (1.1, 0.1, 1.1)
_PIN0_DIR = np.array([1.0, 0.0, 0.0])
_PIN1_LO = (-1.1, -0.1, -1.1)
_PIN1_HI = (-0.99, 0.1, 1.1)
_PIN1_DIR = np.array([-1.0, 0.0, 0.0])
_PIN_VEL = 0.1
_PIN_MAXLENGTH = 1.0

_CLOTH_COLOR = (0.92, 0.86, 0.72)  # light cream — modulates the checker
_CHECKER_DARK = (0.18, 0.18, 0.22)
_CHECKER_LIGHT = (0.85, 0.85, 0.88)
# Squares per side across the [-1, 1] x [-1, 1] cloth sheet.  Each "tile"
# in the UV is a 2x2 checker (so 8 tiles -> 16 checker squares per axis).
_CLOTH_CHECKER_TILES = 8.0


def _square_cloth(n: int) -> tuple[np.ndarray, np.ndarray]:
    """An ``n x n`` grid on ``[-1, 1]^2`` in the ``z = 0`` plane with a vertex at each cell center, four triangles per cell."""
    g = np.linspace(-1.0, 1.0, n + 1)
    c = 0.5 * (g[1:] + g[:-1])
    uv = np.vstack([np.stack(np.meshgrid(a, a, indexing="ij"), -1).reshape(-1, 2) for a in (g, c)])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    quad = np.stack([i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1], axis=1)
    center = (n + 1) ** 2 + i * n + j
    tris = np.concatenate([np.stack([quad[:, k], quad[:, (k + 1) % 4], center], axis=1) for k in range(4)])
    return np.column_stack([uv, np.zeros(len(uv))]).astype(np.float32), tris.astype(np.int32)


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def _select_in_aabb(points: np.ndarray, lo: tuple, hi: tuple) -> np.ndarray:
    p = np.asarray(points)
    lo = np.asarray(lo, dtype=p.dtype)
    hi = np.asarray(hi, dtype=p.dtype)
    return np.where(np.all((p >= lo) & (p <= hi), axis=1))[0].astype(np.int32)


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

        verts_local, tris = _square_cloth(100)  # 20201 particles
        verts_world = (verts_local @ _rot_x(_ROT_X_DEG).T + _TRANS).astype(np.float32)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, _GRAVITY, 0.0))

        e1 = verts_world[tris[:, 1]] - verts_world[tris[:, 0]]
        e2 = verts_world[tris[:, 2]] - verts_world[tris[:, 0]]
        tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        density = _OBJ_MASS / max(float(tri_area.sum()), 1e-12)

        mu, lam = _lame_from_young_poisson(_YOUNG, _POISSON)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*v.tolist()) for v in verts_world],
            indices=tris.reshape(-1).tolist(),
            density=density,
            tri_ke=mu,
            tri_ka=lam,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )

        pin0_idx = _select_in_aabb(verts_world, _PIN0_LO, _PIN0_HI)
        pin1_idx = _select_in_aabb(verts_world, _PIN1_LO, _PIN1_HI)

        self.model = builder.finalize()

        uniform_mass = _OBJ_MASS / self.model.particle_count
        self.model.particle_mass.assign(np.full(self.model.particle_count, uniform_mass, np.float32))
        inv = np.full(self.model.particle_count, 1.0 / uniform_mass, np.float32)
        inv[pin0_idx] = 0.0
        inv[pin1_idx] = 0.0
        self.model.particle_inv_mass.assign(inv)

        self.solver = SolverFBA(
            self.model,
            iterations=_PD_ITER,
            friction=False,
            stretching_model="neohookean",
            nh_solver="lbfgs",
            pin_stiffness=1.0e10,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self._pin0_idx = pin0_idx
        self._pin1_idx = pin1_idx
        self._pin0_init = verts_world[pin0_idx].copy()
        self._pin1_init = verts_world[pin1_idx].copy()
        self._targets = verts_world.copy()
        self._pulled = 0.0

        # Cloth render data — per-particle UVs derived from rest position
        # (x, z) mapped to [0, _CLOTH_CHECKER_TILES] so GL_REPEAT tiles a
        # 2x2 checker texture across the rest-state sheet.
        self._cloth_tri_indices = self.model.tri_indices.flatten()
        # Rest sheet spans roughly x ∈ [-1, 1] and z ∈ [-1, 1] (post Rx(90)).
        uv_np = np.empty((verts_world.shape[0], 2), dtype=np.float32)
        uv_np[:, 0] = (verts_world[:, 0] + 1.0) * 0.5 * _CLOTH_CHECKER_TILES
        uv_np[:, 1] = (verts_world[:, 2] + 1.0) * 0.5 * _CLOTH_CHECKER_TILES
        self._cloth_uvs = wp.array(uv_np, dtype=wp.vec2)
        self._cloth_texture = _make_checker_texture()

        self.viewer.show_triangles = False
        self.viewer.set_model(self.model)

        # Camera: side/overhead view from +x +y +z toward (0, -0.3, 0), so the stretching is visible.
        self.viewer.set_camera(pos=wp.vec3(1.5, 3.0, 3.0), pitch=-44.5, yaw=-116.6)

    def step(self):
        delta = _PIN_VEL * self.frame_dt
        if self._pulled + delta > _PIN_MAXLENGTH:
            delta = max(0.0, _PIN_MAXLENGTH - self._pulled)
        self._pulled += delta
        self._targets[self._pin0_idx] = self._pin0_init + (_PIN0_DIR * self._pulled).astype(np.float32)
        self._targets[self._pin1_idx] = self._pin1_init + (_PIN1_DIR * self._pulled).astype(np.float32)
        self.solver.set_pin_targets(self._targets)

        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/cloth",
            self.state_0.particle_q,
            self._cloth_tri_indices,
            uvs=self._cloth_uvs,
            texture=self._cloth_texture,
            color=_CLOTH_COLOR,
            backface_culling=False,
        )
        self.viewer.end_frame()

    def test_post_step(self):
        speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max()
        assert speed < 5.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"

    def test_final(self):
        # Pulled to a stretch s, the sheet narrows in its middle, but not below an incompressible strip in free
        # uniaxial stretch, 2 / sqrt(s): 1.83 m against 1.75 m at s = 1.3, 1.51 m against 1.41 m at s = 2. Gravity
        # 1 m/s^2 sags it by at most 14 cm, at 0.6 s, before the tension builds up.
        q, q0 = self.state_0.particle_q.numpy(), self.model.particle_q.numpy()
        assert np.all(np.isfinite(q)), "non-finite particle positions"
        pins = np.concatenate([self._pin0_idx, self._pin1_idx])
        assert np.abs(q[pins] - self._targets[pins]).max() < 1.0e-4, "pinned edges off their targets"
        middle = np.abs(q0[:, 0]) < 0.01
        width = q[middle, 2].max() - q[middle, 2].min()
        stretch = 1.0 + self._pulled
        assert 2.0 / math.sqrt(stretch) < width <= 2.0, f"the middle is {width:.3f} m wide at stretch {stretch:.2f}"
        assert q[:, 1].min() > -0.2, f"the sheet sags to y = {q[:, 1].min():.3f} m"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1200)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
