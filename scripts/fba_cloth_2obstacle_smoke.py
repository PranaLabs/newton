# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Phase 1.4 single-env smoke: cloth dropping onto sphere + box.

A 1013-particle cloth descends and lands on TWO static colliders positioned so
that an interior strip of cloth particles touches both simultaneously (k_p=2
in the lite block-diagonal kernel).

Validates the per-particle fused kernel
(``fb_newton_lite_coulomb_per_particle_kernel``) against the dense full-NSN
reference at p95 < 5 cm / max < 15 cm per the FBA verification tiers.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA

_MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/cloth/square_1013P.obj")
_DT = 0.01
_PD_ITER = 5
_NSN_ITER = 1
_GRAVITY = -10.0
_OBJ_MASS = 1000.0
_YOUNG = 1.0e5
_POISSON = 0.4
_BENDING = 20.0
_FRICTION_MU = 0.3
_PIN_STIFFNESS = 1.0e10


def _rot_x(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _load_obj(p):
    v, t = [], []
    with open(p) as f:
        for ln in f:
            t_ = ln.split()
            if not t_: continue
            if t_[0] == "v":
                v.append([float(x) for x in t_[1:4]])
            elif t_[0] == "f":
                t.append([int(x.split("/")[0]) - 1 for x in t_[1:4]])
    return np.array(v, np.float32), np.array(t, np.int32)


def _lame(y, n):
    mu = y / (2 * (1 + n))
    lam = y * n / ((1 + n) * (1 - 2 * n))
    return mu, lam


def build_scene():
    verts_local, tris = _load_obj(_MESH_PATH)
    # Rx(90°), scale 3.5, translate to (0, 3.5, 0) — same as cloth_on_sphere.
    verts_world = ((verts_local * 3.5) @ _rot_x(90.0).T + np.array([0, 3.5, 0])).astype(np.float32)
    b = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=_GRAVITY)
    e1 = verts_world[tris[:, 1]] - verts_world[tris[:, 0]]
    e2 = verts_world[tris[:, 2]] - verts_world[tris[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    density = _OBJ_MASS / max(float(area.sum()), 1e-12)
    mu, _lam = _lame(_YOUNG, _POISSON)
    b.add_cloth_mesh(
        pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0),
        vertices=[wp.vec3(*v.tolist()) for v in verts_world],
        indices=tris.reshape(-1).tolist(), density=density,
        tri_ke=2.0 * mu, tri_ka=0.0, tri_kd=0.0,
        edge_ke=_BENDING, edge_kd=0.0,
    )
    # Sphere at origin (same as cloth_on_sphere).
    cfg_sphere = b.default_shape_cfg.copy()
    cfg_sphere.mu = _FRICTION_MU
    b.add_shape_sphere(
        body=-1,
        xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
        radius=3.0, cfg=cfg_sphere,
    )
    # Box positioned so its top face is at sphere equator height (y=0), and
    # offset along +X so it overlaps the sphere's right half — cloth particles
    # at the right edge touch both sphere and box top.  Box halves (1.5, 0.5, 3).
    cfg_box = b.default_shape_cfg.copy()
    cfg_box.mu = _FRICTION_MU
    b.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(3.0, -0.5, 0.0), wp.quat_identity()),
        hx=1.5, hy=0.5, hz=3.0, cfg=cfg_box,
    )
    return b


def run(schur_mode: str, num_frames: int):
    b = build_scene()
    m = b.finalize()
    um = _OBJ_MASS / m.particle_count
    m.particle_mass.assign(np.full(m.particle_count, um, np.float32))
    m.particle_inv_mass.assign(np.full(m.particle_count, 1.0 / um, np.float32))
    mu_o = np.full(m.particle_count, _FRICTION_MU, np.float64)
    solver = SolverFBA(
        m, iterations=_PD_ITER, nsn_iterations=_NSN_ITER, friction=True,
        stretching_model="arap", mu_per_pair_override=mu_o,
        pin_stiffness=_PIN_STIFFNESS, nsn_schur_mode=schur_mode,
    )
    pipe = newton.CollisionPipeline(m, soft_contact_margin=0.05)
    c = pipe.contacts()
    s0 = m.state(); s1 = m.state()
    positions = np.empty((num_frames, m.particle_count, 3), np.float32)
    step_ms = np.empty(num_frames, np.float32)
    max_kp_seen = 0
    for f in range(num_frames):
        s0.clear_forces()
        pipe.collide(s0, c)
        wp.synchronize_device()
        t = time.perf_counter()
        solver.step(s0, s1, None, c, _DT)
        wp.synchronize_device()
        step_ms[f] = (time.perf_counter() - t) * 1000
        s0, s1 = s1, s0
        positions[f] = s0.particle_q.numpy()
        ls = solver._linear_solver
        if ls is not None and hasattr(ls, "_isodof_max_k"):
            max_kp_seen = max(max_kp_seen, ls._isodof_max_k)
        if f % 50 == 0:
            nc = int(c.soft_contact_count.numpy()[0]) if hasattr(c, "soft_contact_count") else 0
            print(f"  [{schur_mode}] frame {f}: y_min={positions[f,:,1].min():.3f} "
                  f"n_contacts={nc} max_kp_rows={max_kp_seen} step_ms={step_ms[f]:.2f}")
    return positions, step_ms, max_kp_seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-frames", type=int, default=300)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp"))
    args = p.parse_args()
    wp.init()

    print("=== Newton-full (reference) ===")
    pos_full, ms_full, max_k_full = run("full", args.num_frames)
    print(f"  max_kp_rows in any particle: {max_k_full} (rows; >3 ⇒ k_p>1 contacts)")

    print("=== Newton-lite (per-particle kernel) ===")
    pos_lite, ms_lite, max_k_lite = run("lite", args.num_frames)
    print(f"  max_kp_rows in any particle: {max_k_lite}")

    # Acceptance gate: p95 / max drift @ frame 200 (or last frame if shorter).
    f_check = min(200, args.num_frames - 1)
    err = np.linalg.norm(pos_full[f_check] - pos_lite[f_check], axis=1)
    print(f"\nDrift Newton-full vs Newton-lite @ frame {f_check}:")
    print(f"  p50 = {np.percentile(err,50)*100:.3f} cm")
    print(f"  p95 = {np.percentile(err,95)*100:.3f} cm")
    print(f"  max = {err.max()*100:.3f} cm")
    print(f"Gate: p95<5cm, max<15cm")
    p95 = float(np.percentile(err, 95))
    mx = float(err.max())
    ok = p95 < 0.05 and mx < 0.15
    print("PASS" if ok else "FAIL")

    np.savez(args.out_dir / "fba_2obstacle_full.npz", positions=pos_full, step_ms=ms_full)
    np.savez(args.out_dir / "fba_2obstacle_lite.npz", positions=pos_lite, step_ms=ms_lite)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
