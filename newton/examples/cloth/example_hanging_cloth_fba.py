# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example HangingCloth FBA
#
# A 10225-particle Neo-Hookean cloth tilted 30 deg about X and pinned
# along its top edge, draping under gravity (-10 m/s^2 along Y) with
# isometric bending.  Mirrors RealSim Tests/ClothTest.
#
# Command: uv run -m newton.examples hanging_cloth_fba
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

_DT = 0.05  # ClothTest/scene.json timestep
_PD_ITER = 10  # scene.json LocalGlobal = 10
_GRAVITY = -10.0  # scene.json gravity = [0, -10, 0]

_YOUNG = 1.0e4
_POISSON = 0.4
_OBJ_MASS = 1000.0
_BENDING = 1.0e-3  # softer than ClothTest's 0.1 — Newton's edge_ke uses a
# different normalisation than RealSim's bending coefficient, so 0.1 here
# is far too stiff and washes out the drape wrinkles.

_ROT_X_DEG = 30.0  # object.json rotation=[30, 0, 0]
_TRANS = np.array([0.0, 0.0, 0.0])

# Pin AABB in world coordinates (post-transformation), per object.json.
_PIN_LO = (-1.1, 0.8, -1.1)
_PIN_HI = (+1.1, 1.1, +1.1)

_CLOTH_COLOR = (0.78, 0.32, 0.36)  # muted red — drape wrinkles read better
# on a solid colour than a tiled checker, which hides the fine ripples.


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


def build_hanging_cloth_builder() -> tuple[newton.ModelBuilder, np.ndarray]:
    """Construct a single-env hanging-cloth ModelBuilder (not finalized).

    Returns a `(builder, pin_idx)` pair so callers can finalize and then
    zero out `particle_mass` on the pinned indices.  The pin selection
    must happen pre-finalize because it indexes into the local cloth
    vertex array — after `replicate()` is layered on top, per-world
    indices are re-numbered and the AABB sweep would select the wrong particles.
    """
    verts_local, tris = _square_cloth(71)  # 10225 particles
    verts_world = (verts_local @ _rot_x(_ROT_X_DEG).T + _TRANS).astype(np.float32)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=_GRAVITY)

    e1 = verts_world[tris[:, 1]] - verts_world[tris[:, 0]]
    e2 = verts_world[tris[:, 2]] - verts_world[tris[:, 0]]
    tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    density = _OBJ_MASS / max(float(tri_area.sum()), 1e-12)

    mu, _lam = _lame_from_young_poisson(_YOUNG, _POISSON)

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*v.tolist()) for v in verts_world],
        indices=tris.reshape(-1).tolist(),
        density=density,
        tri_ke=2.0 * mu,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=_BENDING,
        edge_kd=0.0,
    )

    pin_idx = _select_in_aabb(verts_world, _PIN_LO, _PIN_HI)
    return builder, pin_idx


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = int(round(1.0 / _DT))
        self.frame_dt = _DT
        self.sim_time = 0.0

        builder, pin_idx = build_hanging_cloth_builder()
        self.model = builder.finalize()

        uniform_mass = _OBJ_MASS / self.model.particle_count
        mass_arr = np.full(self.model.particle_count, uniform_mass, np.float32)
        mass_arr[pin_idx] = 0.0  # static anchor; matches example_cloth_hanging_fba pattern
        self.model.particle_mass.assign(mass_arr)
        inv = np.full(self.model.particle_count, 1.0 / uniform_mass, np.float32)
        inv[pin_idx] = 0.0
        self.model.particle_inv_mass.assign(inv)

        self.solver = SolverFBA(self.model, iterations=_PD_ITER)
        self.pins = pin_idx

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self._cloth_tri_indices = self.model.tri_indices.flatten()

        self.viewer.show_triangles = False
        self.viewer.set_model(self.model)

        # Camera framing the tilted-and-draping sheet from front-right, toward (0, -0.4, 0).
        self.viewer.set_camera(pos=wp.vec3(2.8, 0.6, 3.2), pitch=-13.2, yaw=-131.2)

    def step(self):
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
            color=_CLOTH_COLOR,
            backface_culling=False,
        )
        self.viewer.end_frame()

    def test_post_step(self):
        speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max()
        assert speed < 10.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"

    def test_final(self):
        # After 10 s the 2 m sheet hangs 2.6 m down from its pinned top edge, stretched by its own weight,
        # and swings at 0.2 m/s.
        q, q0 = self.state_0.particle_q.numpy(), self.model.particle_q.numpy()
        assert np.all(np.isfinite(q)), "non-finite particle positions"
        assert np.abs(q[self.pins] - q0[self.pins]).max() < 1.0e-5, "pinned particles moved"
        top = q0[self.pins, 1].min()
        assert q[:, 1].mean() < top - 1.0, f"the sheet's centroid is at y = {q[:, 1].mean():.2f} m"
        assert q[:, 1].min() > -3.0, f"the sheet reaches down to y = {q[:, 1].min():.2f} m"
        speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max()
        assert speed < 1.0, f"the sheet still moves at {speed:.2f} m/s"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
