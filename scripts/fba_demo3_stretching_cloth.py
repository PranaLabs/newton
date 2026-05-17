# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Demo 3: StretchingCloth — CudaTests reproduction.

Pulls a 20201-particle Neo-Hookean cloth along ``±x`` via two PULLING pins
on opposing edges.  No contacts; pure FEM stretching, so the solver runs
with ``friction=False`` and no NSN iterations.

Scene reference: ``RealSim_py/realsim_py/simulation/config/CudaTests/StretchingCloth``
(``scene.json`` + ``object_NH_20k.json``).

Outputs trajectory + snapshot strip + perf row to
``scripts/contact_demos_out/demo3/`` and ``/tmp/fba_demo3_stretching_cloth.npz``.

Usage:
    uv run python scripts/fba_demo3_stretching_cloth.py
"""

from __future__ import annotations

import argparse
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
from scripts.fba_cudatest_bench.perf import record_row, stats_from_times_ms
from scripts.fba_cudatest_bench.pin import select_in_aabb

# ---------------------------------------------------------------------------
# Constants from RealSim CudaTests/StretchingCloth
# ---------------------------------------------------------------------------
MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/cloth/square_20201P.obj")
DT = 0.01
TOTAL_FRAMES = 1200
PD_ITERATIONS = 10  # RealSim's offline binding overrides scene LocalGlobal_CUDA=5 to 10
NSN_ITERATIONS = 1  # No contacts — NSN unused but keeps the API uniform.
GRAVITY = -1.0  # Y-up, scene.json gravity=[0, -1, 0]

YOUNG = 1.0e5
POISSON = 0.45
OBJ_MASS = 1000.0  # mechanical_props.obj_mass

# Object transformation from object_NH_20k.json:
#   rotation=[90, 0, 0] deg (about X), trans=(0,0,0), scale=1.
# Rx(90): (x,y,z) -> (x, -z, y), so the mesh -- authored as a Z=0 plane in
# Blender -- becomes a Y=0 sheet spanning x,z ∈ [-1, 1].
ROT_X_DEG = 90.0
TRANS = np.array([0.0, 0.0, 0.0])

# Pin AABBs (world coordinates after object transformation).  RealSim
# evaluates the pin filter on the transformed mesh.
PIN0_LO = (0.99, -0.1, -1.1)
PIN0_HI = (1.1, 0.1, 1.1)
PIN0_DIR = np.array([1.0, 0.0, 0.0])

PIN1_LO = (-1.1, -0.1, -1.1)
PIN1_HI = (-0.99, 0.1, 1.1)
PIN1_DIR = np.array([-1.0, 0.0, 0.0])

PIN_VEL = 0.1  # m/s
PIN_MAXLENGTH = 1.0  # m (per pin)

OUT_DIR = Path(__file__).parent / "contact_demos_out" / "demo3"
NPZ_PATH = Path("/tmp/fba_demo3_stretching_cloth.npz")
SNAPSHOT_FRAMES = [0, 300, 600, 900, 1200]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an OBJ file, returning ``(vertices (V,3) float32, faces (F,3) int32)``.

    Args:
        path: Path to the OBJ mesh.

    Returns:
        Tuple of ``(vertices, faces)`` with 0-based triangle indices.
    """
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


def rot_x(deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix for a rotation of ``deg`` degrees about +X."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def transform_verts(verts_local: np.ndarray) -> np.ndarray:
    """Apply RealSim's object transformation: ``Rx(90)`` then translate."""
    R = rot_x(ROT_X_DEG).astype(verts_local.dtype)
    return verts_local @ R.T + TRANS.astype(verts_local.dtype)


def lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    """Convert (Young's modulus, Poisson ratio) to first/second Lamé parameters."""
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model() -> tuple:
    """Construct the StretchingCloth model and identify the two pin sets.

    Returns:
        Tuple of ``(model, verts_world, pin0_idx, pin1_idx, mu, lam)``.
    """
    verts_local, tris = load_obj(MESH_PATH)
    verts_world = transform_verts(verts_local)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=GRAVITY)

    # Density from total triangle area to give RealSim's obj_mass when summed
    # by FE lumping.  We then overwrite with uniform mass below for full parity,
    # so density only affects intermediate builder state.
    e1 = verts_world[tris[:, 1]] - verts_world[tris[:, 0]]
    e2 = verts_world[tris[:, 2]] - verts_world[tris[:, 0]]
    tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total_area = float(tri_area.sum())
    density = OBJ_MASS / max(total_area, 1e-12)

    mu, lam = lame_from_young_poisson(YOUNG, POISSON)

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*v.tolist()) for v in verts_world],
        indices=tris.reshape(-1).tolist(),
        density=density,
        tri_ke=2.0 * mu,  # PD stiffness handed to add_cloth_mesh; SolverFBA
        tri_ka=0.0,  # uses (mu, lam) via stretching_model="neohookean".
        tri_kd=0.0,
        edge_ke=0.0,  # bending=0 in object_NH_20k.json
        edge_kd=0.0,
    )

    pin0_idx = select_in_aabb(verts_world, PIN0_LO, PIN0_HI)
    pin1_idx = select_in_aabb(verts_world, PIN1_LO, PIN1_HI)

    model = builder.finalize()

    # RealSim uniform mass lumping (Mass.cpp:12-21): obj_mass / N per particle.
    # Locked decision #3 of the FBA-RealSim parity plan.
    uniform_mass = OBJ_MASS / model.particle_count
    mass_arr = np.full(model.particle_count, uniform_mass, dtype=np.float32)
    inv_mass_arr = np.full(model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
    model.particle_mass.assign(mass_arr)
    model.particle_inv_mass.assign(inv_mass_arr)

    # Mark pinned particles inv_mass = 0 AFTER the uniform assign above.
    inv_m = model.particle_inv_mass.numpy()
    inv_m[pin0_idx] = 0.0
    inv_m[pin1_idx] = 0.0
    model.particle_inv_mass.assign(inv_m)

    return model, verts_world, pin0_idx, pin1_idx, mu, lam


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_frame(ax, q: np.ndarray, frame: int) -> None:
    """Scatter-render cloth particles for the snapshot strip."""
    ax.clear()
    step = max(1, len(q) // 1500)
    pts = q[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=pts[:, 0], cmap="coolwarm")
    ax.set_title(f"StretchingCloth — frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.2, 0.5)
    ax.set_zlim(-1.5, 1.5)


# ---------------------------------------------------------------------------
# Sim loop
# ---------------------------------------------------------------------------
def run(diag_frame: int | None = None, diag_out: str | None = None) -> dict:
    """Run the StretchingCloth simulation for ``TOTAL_FRAMES`` steps.

    Args:
        diag_frame: Optional 1-indexed step number at which to dump per-PD-outer-iter
            intermediate state for parity comparison with RealSim.
        diag_out: Output path for the diagnostic ``.npz`` dump. Required if
            ``diag_frame`` is set.

    Returns:
        Dict with keys ``"snapshots"``, ``"stats"``, ``"trajectory"``.
    """
    model, verts_world, pin0_idx, pin1_idx, mu_lame, lam = build_model()
    n_tris = int(model.tri_count)
    print(f"  particles={model.particle_count}  tris={n_tris}  pin0={len(pin0_idx)}  pin1={len(pin1_idx)}")

    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=False,
        stretching_model="neohookean",
        nh_solver="lbfgs",
        mu=mu_lame,
        lam=lam,
        # Pin stiffness: 1e10 aligns to RealSim's empirically observed
        # `w_pin_R` in the system matrix (per intermediate-variable diff at
        # frame 10: FBA RHS pin contribution was 100x stronger than RealSim's
        # under the A_FBA = A_R/dt^2 scaling). Was 1e12 default.
        pin_stiffness=1.0e10,
    )
    if diag_frame is not None:
        if diag_out is None:
            raise ValueError("--diag-out required when --diag-frame is set")
        solver.configure_diagnostic_dump(int(diag_frame), str(diag_out))
    s_in = model.state()
    s_out = model.state()

    pin0_init = verts_world[pin0_idx].copy().astype(np.float32)
    pin1_init = verts_world[pin1_idx].copy().astype(np.float32)
    targets = verts_world.copy().astype(np.float32)

    trajectory = np.empty((TOTAL_FRAMES + 1, model.particle_count, 3), dtype=np.float32)
    snapshots: dict[int, np.ndarray] = {}
    step_times: list[float] = []
    pulled = 0.0

    for frame in range(TOTAL_FRAMES + 1):
        q_now = s_in.particle_q.numpy()
        trajectory[frame] = q_now
        if frame in SNAPSHOT_FRAMES:
            snapshots[frame] = q_now.copy()
        if frame == TOTAL_FRAMES:
            break

        # PullingAction(vel*dt, maxlength, direction): advance each pin by
        # ``vel*dt`` per frame in its direction until ``maxlength`` is reached,
        # then hold the last position.  RealSim init.h:1005.
        delta = PIN_VEL * DT
        if pulled + delta > PIN_MAXLENGTH:
            delta = max(0.0, PIN_MAXLENGTH - pulled)
        pulled += delta
        targets[pin0_idx] = pin0_init + (PIN0_DIR * pulled).astype(np.float32)
        targets[pin1_idx] = pin1_init + (PIN1_DIR * pulled).astype(np.float32)
        solver.set_pin_targets(targets)

        wp.synchronize_device()
        t0 = time.perf_counter()
        s_in.clear_forces()
        # No collision pipeline: StretchingCloth has no contacts.
        solver.step(s_in, s_out, None, None, DT)
        wp.synchronize_device()
        step_times.append(1000.0 * (time.perf_counter() - t0))
        s_in, s_out = s_out, s_in

    q_final = trajectory[-1]
    finite_ok = bool(np.all(np.isfinite(q_final)))
    stats = stats_from_times_ms(step_times)
    stats["stable"] = finite_ok
    stats["pulled"] = float(pulled)
    stats["min_y"] = float(q_final[:, 1].min())
    stats["x_extent"] = float(q_final[:, 0].max() - q_final[:, 0].min())
    return {"snapshots": snapshots, "stats": stats, "trajectory": trajectory}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run Demo 3 and emit trajectory, snapshot strip, and perf row."""
    parser = argparse.ArgumentParser(description="FBA Demo 3: StretchingCloth")
    parser.add_argument(
        "--diag-frame",
        type=int,
        default=None,
        help="1-indexed step number at which to dump PD intermediate state for parity check",
    )
    parser.add_argument(
        "--diag-out",
        type=str,
        default=None,
        help="Output .npz path for the diagnostic dump",
    )
    args = parser.parse_args()

    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Demo 3: StretchingCloth")
    print("=" * 60)

    wall_t0 = time.perf_counter()
    result = run(diag_frame=args.diag_frame, diag_out=args.diag_out)
    wall_time_s = time.perf_counter() - wall_t0
    stats = result["stats"]
    stats["wall_time_s"] = round(wall_time_s, 3)

    print(
        f"  wall_time_s={stats['wall_time_s']:.2f}  "
        f"mean_step_ms={stats['mean_ms']:.2f}  "
        f"median={stats['median_ms']:.2f}  p95={stats['p95_ms']:.2f}"
    )
    print(
        f"  stable={stats['stable']}  pulled={stats['pulled']:.3f} m  "
        f"x_extent={stats['x_extent']:.3f} m  min_y={stats['min_y']:.3f}"
    )

    np.savez_compressed(NPZ_PATH, positions=result["trajectory"])
    print(f"  trajectory saved: {NPZ_PATH}  shape={result['trajectory'].shape}")

    for frame, q in sorted(result["snapshots"].items()):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        render_frame(ax, q, frame)
        out = OUT_DIR / f"frame_{frame:04d}.png"
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)

    n_snap = len(result["snapshots"])
    fig = plt.figure(figsize=(4 * n_snap, 4))
    fig.suptitle("Demo 3: StretchingCloth")
    for col, (frame, q) in enumerate(sorted(result["snapshots"].items())):
        ax = fig.add_subplot(1, n_snap, col + 1, projection="3d")
        render_frame(ax, q, frame)
    plt.tight_layout()
    strip = OUT_DIR / "stretching_cloth_strip.png"
    fig.savefig(strip, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  snapshot strip: {strip}")

    record_row("StretchingCloth", stats)
    print("  perf row saved to scripts/fba_cudatest_rows.json")


if __name__ == "__main__":
    main()
