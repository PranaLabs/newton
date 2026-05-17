# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Demo 4: PullingWooper — CudaTests reproduction.

Pulls a 5325-particle tet wooper through two static cylinders. Single
PULLING pin action driven externally via ``SolverFBA.set_pin_targets``.

Outputs PNGs + perf row to ``scripts/contact_demos_out/demo4/``.
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
from scripts.fba_cudatest_bench.pin import select_in_aabb

# ---------------------------------------------------------------------------
# Constants from RealSim CudaTests/PullingWooper
# ---------------------------------------------------------------------------
MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/wooper_volume_5325P.mesh")
DT = 0.01
TOTAL_FRAMES = 500
PD_ITERATIONS = 5
NSN_ITERATIONS = 1  # RealSim does 1 FB-Newton step per NSN call
GRAVITY = 0.0  # PullingWooper scene.json sets gravity = 0

YOUNG = 1.0e7
POISSON = 0.3
OBJ_MASS = 1000.0  # total object mass — density = OBJ_MASS / total_vol

# Transformation from wooper_5k.json: rotation Z=-90deg, trans=(0, 4, 0), scale=1.0
ROT_Z_DEG = -90.0
TRANS = np.array([0.0, 4.0, 0.0])

# Pin AABB in object-local frame.
PIN_LO = (-2.5, 2.0, -0.3)
PIN_HI = (-1.5, 4.0, 0.3)
# Single PULLING action: dir=-Y at 2 m/s, capped at 10 m total.
PIN_DIR = np.array([0.0, -1.0, 0.0])
PIN_VEL = 2.0
PIN_MAXLENGTH = 10.0

# Cylinders: both along world +X axis.
CYL_RADIUS = 1.0
CYL_HALF_HEIGHT = 3.0
CYL0_BASE = np.array([0.0, -1.0, -1.3])
CYL1_BASE = np.array([0.0, -1.0, 1.3])
CYL_AXIS = np.array([1.0, 0.0, 0.0])

OUT_DIR = Path(__file__).parent / "contact_demos_out" / "demo4"
SNAPSHOT_FRAMES = [0, 50, 150, 300, 450, 499]


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform_verts(verts_local: np.ndarray) -> np.ndarray:
    return verts_local @ rot_z(ROT_Z_DEG).T + TRANS


def lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def cyl_xform(base: np.ndarray, axis: np.ndarray) -> wp.transform:
    """Build a Newton transform that orients a Z-aligned cylinder along ``axis``.

    Newton's ``add_shape_cylinder`` extends along local +Z. We compute the
    rotation that maps +Z to ``axis``.
    """
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    v = np.cross(z, a)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, a))
    if s < 1e-9:
        if c > 0:
            q = wp.quat_identity()
        else:
            q = wp.quat(1.0, 0.0, 0.0, 0.0)  # 180 deg about X
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

    # Density from total mass / total volume (sum of tet volumes in world frame).
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
        add_surface_mesh_edges=False,  # pure tet body, no bending edges
    )

    # Cylinders (static obstacles).
    builder.add_shape_cylinder(
        body=-1,
        xform=cyl_xform(CYL0_BASE, CYL_AXIS),
        radius=CYL_RADIUS,
        half_height=CYL_HALF_HEIGHT,
    )
    builder.add_shape_cylinder(
        body=-1,
        xform=cyl_xform(CYL1_BASE, CYL_AXIS),
        radius=CYL_RADIUS,
        half_height=CYL_HALF_HEIGHT,
    )

    # PIN_LO/PIN_HI are in WORLD coordinates (matches RealSim wooper_5k.json semantics).
    pin_idx = select_in_aabb(verts_world, PIN_LO, PIN_HI)

    model = builder.finalize()

    # Match RealSim's uniform mass lumping (Mass.cpp:12-21):
    # obj_mass / particle_count per particle, replacing Newton's FE lumping.
    # Locked decision #3 of the FBA-RealSim parity plan; per-demo override since
    # FE lumping remains the correct default for Newton's other solvers.
    # total_mass = OBJ_MASS (1000.0) from CudaTests/PullingWooper/wooper_5k.json
    # mechanical_props.obj_mass (per scripts/realsim_baseline/decisions.json).
    uniform_mass = OBJ_MASS / model.particle_count
    mass_arr = np.full(model.particle_count, uniform_mass, dtype=np.float32)
    inv_mass_arr = np.full(model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
    model.particle_mass.assign(mass_arr)
    model.particle_inv_mass.assign(inv_mass_arr)

    # Mark pinned particles as inv_mass = 0 (applied AFTER uniform lumping so
    # pin semantics win over the uniform assign above).
    inv_m = model.particle_inv_mass.numpy()
    inv_m[pin_idx] = 0.0
    model.particle_inv_mass.assign(inv_m)

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = pipeline.contacts()
    return model, pipeline, contacts, verts_world, pin_idx, mu, lam


def render_frame(ax, q: np.ndarray, frame: int) -> None:
    ax.clear()
    step = max(1, len(q) // 500)
    pts = q[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 1], cmap="viridis")
    ax.set_title(f"PullingWooper — frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-10, 6)
    ax.set_zlim(-3, 3)


def run() -> dict:
    model, pipeline, contacts, verts_world, pin_idx, mu, lam = build_model()
    n_tets = len(model.tet_indices.numpy()) // 4
    print(f"  particles={model.particle_count}  tets={n_tets}  pinned={len(pin_idx)}")

    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=False,
        stretching_model="neohookean",
        mu=mu,
        lam=lam,
        # RealSim PullingWooper/scene.json::constraintsolver.maxforce = 1e12 (no
        # effective cap).  Wired here for Phase 2.4 / per-scene λ-cap alignment.
        lambda_cap=1.0e12,
    )
    s_in = model.state()
    s_out = model.state()

    pin_init = verts_world[pin_idx].copy()
    targets = verts_world.copy().astype(np.float32)

    snapshots: dict[int, np.ndarray] = {}
    step_times: list[float] = []
    pulled = 0.0

    for frame in range(TOTAL_FRAMES + 1):
        if frame in SNAPSHOT_FRAMES:
            snapshots[frame] = s_in.particle_q.numpy().copy()
        if frame == TOTAL_FRAMES:
            break

        delta = PIN_VEL * DT
        if pulled + delta > PIN_MAXLENGTH:
            delta = max(0.0, PIN_MAXLENGTH - pulled)
        pulled += delta
        targets[pin_idx] = pin_init + (PIN_DIR * pulled).astype(np.float32)
        solver.set_pin_targets(targets)

        wp.synchronize_device()
        t0 = time.perf_counter()
        s_in.clear_forces()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, DT)
        wp.synchronize_device()
        step_times.append(1000.0 * (time.perf_counter() - t0))
        s_in, s_out = s_out, s_in

    q_final = s_in.particle_q.numpy()
    finite_ok = bool(np.all(np.isfinite(q_final)))
    stats = stats_from_times_ms(step_times)
    stats["stable"] = finite_ok
    stats["min_y"] = float(q_final[:, 1].min())
    stats["pulled"] = pulled
    return {"snapshots": snapshots, "stats": stats}


def main() -> None:
    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Demo 4: PullingWooper")
    print("=" * 60)

    result = run()
    stats = result["stats"]
    print(f"  mean_ms={stats['mean_ms']:.2f}  median={stats['median_ms']:.2f}  p95={stats['p95_ms']:.2f}")
    print(f"  stable={stats['stable']}  min_y={stats['min_y']:.3f}  pulled={stats['pulled']:.2f}")

    fig = plt.figure(figsize=(4 * len(result["snapshots"]), 4))
    fig.suptitle("Demo 4: PullingWooper")
    for col, (frame, q) in enumerate(sorted(result["snapshots"].items())):
        ax = fig.add_subplot(1, len(result["snapshots"]), col + 1, projection="3d")
        render_frame(ax, q, frame)
    plt.tight_layout()
    out = OUT_DIR / "pulling_wooper_strip.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  snapshot strip: {out}")

    record_row("PullingWooper", stats)
    print("  perf row saved to scripts/fba_cudatest_rows.json")


if __name__ == "__main__":
    main()
