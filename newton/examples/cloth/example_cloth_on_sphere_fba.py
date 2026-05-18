# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ClothOnSphere FBA
#
# A 1013-particle TRI_ARAP cloth (Young=1e5, Poisson=0.4, isometric bending)
# dropped onto a static sphere (radius 3, mu=0.3) under gravity -10 Y.
# Mirrors RealSim CudaTests/ParallelEnvTest single-env geometry — this is
# the Phase 1 fixture for SolverFBA multi-env / LiteNSN validation.
#
# Command: uv run -m newton.examples cloth_on_sphere_fba
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

_MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/cloth/square_1013P.obj")

# ParallelEnvTest/scene.json + cloth0.json
_DT = 0.01
_PD_ITER = 5  # scene.json LocalGlobal = 5
_NSN_ITER = 1  # FB-Newton outer count (RealSim runs 1 per step; locked)
_GRAVITY = -10.0
_OBJ_MASS = 1000.0

_YOUNG = 1.0e5
_POISSON = 0.4
_BENDING = 20.0  # scene.json bending=20; revisit if drape explodes (see plan §1.1)

# Rx(90°) then scale 3.5 then trans (0, 3.5, 0).
_ROT_X_DEG = 90.0
_SCALE = 3.5
_TRANS = np.array([0.0, 3.5, 0.0])

# Static sphere collider (ParallelEnvTest/scene.json sphereCollisions[0]).
_SPHERE_RADIUS = 3.0
_SPHERE_CENTER = np.array([0.0, 0.0, 0.0])
_FRICTION_MU = 0.3
_PIN_STIFFNESS = 1.0e10  # carried forward from Demo 5 (no pins, only used by Stage B residual)

# Visual checker for the cloth (planar UVs from rest pose, projected XZ).
_CLOTH_COLOR = (0.78, 0.62, 0.32)  # warm tan
_CHECKER_DARK = (0.18, 0.18, 0.22)
_CHECKER_LIGHT = (0.85, 0.85, 0.88)
_CHECKER_TILES = 16.0  # uv-tile multiplier across the cloth bounding box

_SPHERE_COLOR = (0.55, 0.55, 0.65)


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    with open(path) as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "v":
                verts.append([float(t) for t in tokens[1:4]])
            elif tokens[0] == "f":
                idx = [int(t.split("/")[0]) - 1 for t in tokens[1:4]]
                tris.append(idx)
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


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


def build_cloth_on_sphere_builder() -> newton.ModelBuilder:
    """Construct a single-env cloth-on-sphere ``ModelBuilder`` (not finalized).

    Used both by the user-facing :class:`Example` and by the multi-env
    benchmark harness (``scripts/fba_multienv_cloth_bench.py``).
    """
    if not _MESH_PATH.exists():
        raise FileNotFoundError(
            f"Cloth mesh not found at {_MESH_PATH}.  Update _MESH_PATH at the top of this example."
        )
    verts_local, tris = _load_obj(_MESH_PATH)

    # Rx(90°) then scale then translate (matches RealSim transformation block).
    verts_world = ((verts_local * _SCALE) @ _rot_x(_ROT_X_DEG).T + _TRANS).astype(np.float32)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=_GRAVITY)

    # Density bookkeeping — actual particle mass is uniform-lumped post-finalize.
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

    # Static sphere collider — body=-1 = world-frame static shape.
    sphere_cfg = builder.default_shape_cfg.copy()
    sphere_cfg.mu = _FRICTION_MU
    builder.add_shape_sphere(
        body=-1,
        xform=wp.transform(wp.vec3(*_SPHERE_CENTER.tolist()), wp.quat_identity()),
        radius=_SPHERE_RADIUS,
        cfg=sphere_cfg,
        color=_SPHERE_COLOR,
    )

    return builder


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = int(round(1.0 / _DT))
        self.frame_dt = _DT
        self.sim_time = 0.0

        # Solver mode is a CLI option exposed via the example parser; see
        # ``__main__`` block below.  Defaults to "full" so the example
        # exercises the validated single-env path unless explicitly asked
        # for LiteNSN.
        self.schur_mode = getattr(args, "schur_mode", "full")

        builder = build_cloth_on_sphere_builder()
        self.model = builder.finalize()

        # Uniform mass lumping (RealSim Mass.cpp:12-21 parity).
        uniform_mass = _OBJ_MASS / self.model.particle_count
        self.model.particle_mass.assign(
            np.full(self.model.particle_count, uniform_mass, dtype=np.float32)
        )
        self.model.particle_inv_mass.assign(
            np.full(self.model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
        )

        n_max_contacts = self.model.particle_count
        mu_override = np.full(n_max_contacts, _FRICTION_MU, dtype=np.float64)

        mu, lam = _lame_from_young_poisson(_YOUNG, _POISSON)

        # Phase 2.3 adds `nsn_schur_mode` to SolverFBA.  Until then the kwarg
        # only goes through when supported, so Phase 1 (this file's initial
        # validation pass) keeps working against pre-Phase-2 commits.
        solver_kwargs = dict(
            iterations=_PD_ITER,
            nsn_iterations=_NSN_ITER,
            friction=True,
            stretching_model="arap",
            mu_per_pair_override=mu_override,
            pin_stiffness=_PIN_STIFFNESS,
        )
        try:
            self.solver = SolverFBA(self.model, **solver_kwargs, nsn_schur_mode=self.schur_mode)
        except TypeError:
            if self.schur_mode != "full":
                raise NotImplementedError(
                    "nsn_schur_mode='lite' requires SolverFBA Phase 2.3 changes"
                )
            self.solver = SolverFBA(self.model, **solver_kwargs)

        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.05)
        self.contacts = self.pipeline.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Per-frame trajectory buffer (only populated when args.dump_npz is set).
        self._dump_npz: Path | None = getattr(args, "dump_npz", None)
        self._traj: list[np.ndarray] = []
        self._n_contacts_per_frame: list[int] = []

        self.viewer.show_triangles = False
        self.viewer.set_model(self.model)

        self._cloth_tri_indices = self.model.tri_indices.flatten()

        # Planar UV from rest XZ position, tiled across the cloth footprint.
        verts_rest = self.state_0.particle_q.numpy()
        x = verts_rest[:, 0]
        z = verts_rest[:, 2]
        x_range = max(float(x.max() - x.min()), 1e-6)
        z_range = max(float(z.max() - z.min()), 1e-6)
        uv = np.empty((self.model.particle_count, 2), dtype=np.float32)
        uv[:, 0] = ((x - x.min()) / x_range) * _CHECKER_TILES
        uv[:, 1] = ((z - z.min()) / z_range) * _CHECKER_TILES
        self._cloth_uvs = wp.array(uv, dtype=wp.vec2)
        self._cloth_texture = _make_checker_texture()

        if hasattr(self.viewer, "camera"):
            cam_pos = wp.vec3(6.5, 4.0, 6.5)
            cam_target = wp.vec3(0.0, 1.0, 0.0)
            self.viewer.camera.pos = self.viewer.camera._as_vec3(cam_pos)
            self.viewer.camera.look_at(cam_target)

    def step(self):
        self.state_0.clear_forces()
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, None, self.contacts, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

        if self._dump_npz is not None:
            self._traj.append(self.state_0.particle_q.numpy().copy())
            n_c = int(self.contacts.soft_contact_count.numpy()[0]) if (
                hasattr(self.contacts, "soft_contact_count") and self.contacts.soft_contact_count is not None
            ) else 0
            self._n_contacts_per_frame.append(n_c)

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

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(q)), "non-finite particle positions"
        if self._dump_npz is not None and self._traj:
            self._dump_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                self._dump_npz,
                positions=np.stack(self._traj, axis=0).astype(np.float32),
                n_contacts=np.array(self._n_contacts_per_frame, dtype=np.int32),
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--schur-mode",
        type=str,
        choices=["full", "lite"],
        default="full",
        help="NSN Schur build mode (default: full).",
    )
    parser.add_argument(
        "--dump-npz",
        type=Path,
        default=None,
        help="If set, dump per-frame particle positions + contact counts to this .npz.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
