# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Demo 5: SqueezingBall — CudaTests reproduction.

A 7129-particle Neo-Hookean tet ball drops between 2 pairs of rolling
cylinders (mu=0.5, |ω|=3 rad/s) onto a static plane (mu=0.5).  The rolling
cylinders friction-drag the ball laterally as it passes through.

Outputs PNGs + perf row to ``scripts/contact_demos_out/demo5/``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA
from scripts.fba_cudatest_bench.medit import load_medit_mesh
from scripts.fba_cudatest_bench.perf import record_row, stats_from_times_ms

MESH_PATH = Path(
    "/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/ball_volume_7129P.mesh"
)
DT = 0.01
TOTAL_FRAMES = 600
PD_ITERATIONS = 5
NSN_ITERATIONS = 1
GRAVITY = -10.0  # Y-down

YOUNG = 1.0e4
POISSON = 0.4
OBJ_MASS = 1000.0

BALL_SCALE = 3.0
BALL_TRANS = np.array([0.0, 2.6, 0.0])

CYL_RADIUS = 1.0
CYL_HALF_HEIGHT = 3.0
FRICTION_MU = 0.5

# (base, axis_world, omega_rad_s) per cylinder, in shape-index order.
CYLINDERS: list[tuple[np.ndarray, np.ndarray, float]] = [
    (np.array([0.0, -1.0, -1.5]), np.array([1.0, 0.0, 0.0]), -3.0),
    (np.array([0.0, -1.0, +1.5]), np.array([1.0, 0.0, 0.0]), +3.0),
    (np.array([-1.5, -3.5, 0.0]), np.array([0.0, 0.0, 1.0]), +3.0),
    (np.array([+1.5, -3.5, 0.0]), np.array([0.0, 0.0, 1.0]), -3.0),
]

OUT_DIR = Path(__file__).parent / "contact_demos_out" / "demo5"
SNAPSHOT_FRAMES = [0, 50, 150, 300, 450, 599]


def transform_verts(verts_local: np.ndarray) -> np.ndarray:
    return verts_local * BALL_SCALE + BALL_TRANS


def lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def cyl_xform(base: np.ndarray, axis: np.ndarray) -> wp.transform:
    """Build a Newton transform that orients a Z-aligned cylinder along ``axis``."""
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


def build_model() -> tuple:
    verts_local, tets = load_medit_mesh(MESH_PATH)
    verts_world = transform_verts(verts_local)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=GRAVITY)

    total_vol = 0.0
    for t in tets:
        v0, v1, v2, v3 = (verts_world[i] for i in t)
        total_vol += abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
    density = OBJ_MASS / max(total_vol, 1e-12)

    mu, lam = lame_from_young_poisson(YOUNG, POISSON)
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

    for base, axis, _omega in CYLINDERS:
        builder.add_shape_cylinder(
            body=-1,
            xform=cyl_xform(base, axis),
            radius=CYL_RADIUS,
            half_height=CYL_HALF_HEIGHT,
        )
    # Plane y = -10 (Y-up world): normal (0,1,0), d = 10.
    builder.add_shape_plane(
        plane=(0.0, 1.0, 0.0, 10.0),
        body=-1,
        width=0.0,
        length=0.0,
    )

    model = builder.finalize()

    if hasattr(model, "shape_material_mu") and model.shape_material_mu is not None:
        mu_arr = model.shape_material_mu.numpy()
        mu_arr[:] = FRICTION_MU
        model.shape_material_mu.assign(mu_arr.astype(np.float32))

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = pipeline.contacts()
    return model, pipeline, contacts, verts_world, mu, lam


def render_frame(ax, q: np.ndarray, frame: int) -> None:
    ax.clear()
    step = max(1, len(q) // 500)
    pts = q[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 1], cmap="viridis")
    ax.set_title(f"SqueezingBall — frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-11, 6)
    ax.set_zlim(-4, 4)


def run() -> dict:
    model, pipeline, contacts, verts_world, mu, lam = build_model()
    n_tets = len(model.tet_indices.numpy()) // 4
    print(f"  particles={model.particle_count}  tets={n_tets}  shapes={model.shape_count}")

    shape_omega = {i: CYLINDERS[i][2] for i in range(4)}
    n_max_contacts = model.particle_count
    mu_override = np.full(n_max_contacts, FRICTION_MU, dtype=np.float64)

    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=True,
        stretching_model="neohookean",
        mu=mu,
        lam=lam,
        mu_per_pair_override=mu_override,
        shape_angular_velocity=shape_omega,
    )
    s_in = model.state()
    s_out = model.state()

    snapshots: dict[int, np.ndarray] = {}
    step_times: list[float] = []

    for frame in range(TOTAL_FRAMES + 1):
        if frame in SNAPSHOT_FRAMES:
            snapshots[frame] = s_in.particle_q.numpy().copy()
        if frame == TOTAL_FRAMES:
            break

        wp.synchronize_device()
        t0 = time.perf_counter()
        s_in.clear_forces()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, DT)
        s_in, s_out = s_out, s_in
        wp.synchronize_device()
        step_times.append(1000.0 * (time.perf_counter() - t0))

    q_final = s_in.particle_q.numpy()
    finite_ok = bool(np.all(np.isfinite(q_final)))
    stats = stats_from_times_ms(step_times)
    stats["stable"] = finite_ok
    stats["min_y"] = float(q_final[:, 1].min())
    stats["x_drift"] = float(q_final[:, 0].mean())
    stats["z_drift"] = float(q_final[:, 2].mean())
    return {"snapshots": snapshots, "stats": stats}


def main() -> None:
    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Demo 5: SqueezingBall")
    print("=" * 60)

    result = run()
    stats = result["stats"]
    print(
        f"  mean_ms={stats['mean_ms']:.2f}  median={stats['median_ms']:.2f}  p95={stats['p95_ms']:.2f}"
    )
    print(
        f"  stable={stats['stable']}  min_y={stats['min_y']:.3f}  "
        f"x_drift={stats['x_drift']:.3f}  z_drift={stats['z_drift']:.3f}"
    )

    fig = plt.figure(figsize=(4 * len(result["snapshots"]), 4))
    fig.suptitle("Demo 5: SqueezingBall")
    for col, (frame, q) in enumerate(sorted(result["snapshots"].items())):
        ax = fig.add_subplot(1, len(result["snapshots"]), col + 1, projection="3d")
        render_frame(ax, q, frame)
    plt.tight_layout()
    out = OUT_DIR / "squeezing_ball_strip.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  snapshot strip: {out}")

    record_row("SqueezingBall", stats)
    print("  perf row saved to scripts/fba_cudatest_rows.json")


if __name__ == "__main__":
    main()
