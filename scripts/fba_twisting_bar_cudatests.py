# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Twisting bar CudaTests-params run for SolverFBA with Neo-Hookean energy.

Matches CudaTests/TwistingBarNH setup exactly:
  - Mesh: cube_volume_11340P.mesh (11340 verts, ~58956 tets)
  - E=1e9, nu=0.45
  - 810 frames, dt=0.01
  - PD iterations=5
  - ROLLING pins (same as RealSim)

Usage::

    uv run python scripts/fba_twisting_bar_cudatests.py --energy neohookean

Outputs (all in scripts/twisting_bar_out_cudatests/):
    {energy}/frame_*.png       — rendered snapshots
    {energy}/trajectory.npz
    perf_summary.txt           — performance comparison table
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib  # noqa: TID253 (headless Agg backend required)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np
import warp as wp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import newton
from newton.solvers import SolverFBA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "twisting_bar_out_cudatests"
MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/cube_volume_11340P.mesh")

# ---------------------------------------------------------------------------
# CudaTests simulation parameters (match CudaTests/TwistingBarNH exactly)
# ---------------------------------------------------------------------------
DT = 0.01
NUM_FRAMES = 810
PD_ITERATIONS = 5
TOTAL_MASS = 1000.0  # kg

# E=1e9, nu=0.45 -> Lamé parameters
YOUNG = 1.0e9
NU = 0.45
MU = YOUNG / (2.0 * (1.0 + NU))
LAM = YOUNG * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

# Rolling pin parameters
MAX_ANGLE = math.pi / 2.0  # 90 degrees (maxrotation=90)
PIN_AVEL = 10.0  # rad/s
PIN_Y_TOP = 1.99
PIN_Y_BOT = -1.99

# Snapshot frames (subset of 810)
SNAPSHOT_FRAMES = [0, 8, 16, 50, 100, 400, 800, 809]


# ---------------------------------------------------------------------------
# 1. Medit .mesh parser (copied from fba_twisting_bar.py)
# ---------------------------------------------------------------------------
def load_medit_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Medit .mesh file (text format, 1-based tet indices)."""
    vertices = []
    tets = []
    state = None
    n_verts = 0
    n_tets = 0
    vert_count = 0
    tet_count_read = 0

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
                if line.startswith(("Triangles", "Edges", "Normals")):
                    state = "skip_section"
                else:
                    state = "header"
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
                if vert_count < n_verts:
                    parts = line.split()
                    vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    vert_count += 1
                    if vert_count == n_verts:
                        state = None
                continue
            if state == "tets":
                if tet_count_read < n_tets:
                    parts = line.split()
                    tets.append((int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1))
                    tet_count_read += 1
                    if tet_count_read == n_tets:
                        state = None
                continue
            if state == "skip_section":
                try:
                    int(line)
                except ValueError:
                    pass
                continue

    verts_np = np.array(vertices, dtype=np.float64)
    tets_np = np.array(tets, dtype=np.int32)
    return verts_np, tets_np


# ---------------------------------------------------------------------------
# 2. Rodrigues rotation matrix around Y axis
# ---------------------------------------------------------------------------
def rot_y(angle: float) -> np.ndarray:
    """Return 3x3 rotation matrix around Y axis for given angle [rad]."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. Build Newton model
# ---------------------------------------------------------------------------
def build_model(
    verts: np.ndarray,
    tets: np.ndarray,
    mu: float,
    lam: float,
) -> tuple[newton.Model, np.ndarray, np.ndarray]:
    """Build Newton ModelBuilder for twisting bar."""
    N = len(verts)
    T = len(tets)
    mass_per_particle = TOTAL_MASS / N

    y = verts[:, 1]
    top_pins = np.where(y > PIN_Y_TOP)[0]
    bot_pins = np.where(y < PIN_Y_BOT)[0]

    print(f"  Vertices: {N}, Tets: {T}")
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    print(
        f"  BBox: x=[{bbox_min[0]:.3f},{bbox_max[0]:.3f}] "
        f"y=[{bbox_min[1]:.3f},{bbox_max[1]:.3f}] "
        f"z=[{bbox_min[2]:.3f},{bbox_max[2]:.3f}]"
    )
    print(f"  Top pins (y>{PIN_Y_TOP}): {len(top_pins)}  Bot pins (y<{PIN_Y_BOT}): {len(bot_pins)}")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)

    pos_list = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    vel_list = [wp.vec3(0.0, 0.0, 0.0)] * N
    mass_list = [mass_per_particle] * N
    builder.add_particles(pos_list, vel_list, mass_list)

    for idx in top_pins:
        builder.particle_mass[int(idx)] = 0.0
    for idx in bot_pins:
        builder.particle_mass[int(idx)] = 0.0

    print("  Adding tetrahedra...", flush=True)
    tets_flat = tets.flatten().tolist()
    for t in range(T):
        i0, i1, i2, i3 = tets_flat[t * 4], tets_flat[t * 4 + 1], tets_flat[t * 4 + 2], tets_flat[t * 4 + 3]
        builder.add_tetrahedron(i0, i1, i2, i3, k_mu=float(mu), k_lambda=float(lam), k_damp=0.0)

    model = builder.finalize()
    print(f"  Model: particle_count={model.particle_count}, tet_count={model.tet_count}")
    return model, top_pins, bot_pins


# ---------------------------------------------------------------------------
# 4. Render a frame
# ---------------------------------------------------------------------------
def render_frame(q: np.ndarray, frame_idx: int, out_path: Path, label: str) -> None:
    """Render a snapshot from particle positions (subsampled every 4th vertex)."""
    q_sub = q[::4]
    x, y, z = q_sub[:, 0], q_sub[:, 1], q_sub[:, 2]
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=x, cmap="coolwarm", s=1.5, alpha=0.7, vmin=-2.0, vmax=2.0)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.0, 2.0)
    ax.view_init(elev=15, azim=30)
    ax.set_title(f"{label}  f={frame_idx}", fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="X coord")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Run Newton simulation
# ---------------------------------------------------------------------------
def run_newton_energy(
    model: newton.Model,
    verts_init: np.ndarray,
    top_pins: np.ndarray,
    bot_pins: np.ndarray,
    label: str,
    stretching_model: str,
    out_subdir: Path,
) -> tuple[np.ndarray, dict]:
    """Run Newton SolverFBA for NUM_FRAMES steps with rolling pin update."""
    mu_arg = MU if stretching_model != "arap" else None
    lam_arg = LAM if stretching_model != "arap" else None
    kw = {"mu": mu_arg, "lam": lam_arg} if mu_arg is not None else {}

    t_setup_0 = time.perf_counter()
    solver = SolverFBA(model, iterations=PD_ITERATIONS, pin_stiffness=1e12, stretching_model=stretching_model, **kw)
    # Trigger setup (first step builds the linear system)
    state_in = model.state()
    state_out = model.state()
    state_in.clear_forces()
    solver.step(state_in, state_out, None, None, DT)
    if wp.is_cuda_available():
        wp.synchronize()
    t_setup_1 = time.perf_counter()
    setup_time = t_setup_1 - t_setup_0
    print(f"  [{label}] Setup+first-step time: {setup_time * 1000:.1f} ms", flush=True)

    # Reset state for actual run
    state_in = model.state()
    state_out = model.state()

    snapshot_set = set(SNAPSHOT_FRAMES)
    snaps: list[tuple[int, np.ndarray]] = []
    step_times: list[float] = []
    nan_frame: int | None = None

    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    print(f"  [{label}] Running {NUM_FRAMES} frames...", flush=True)
    for f in range(NUM_FRAMES):
        # Compute rolling angles
        top_angle = min(PIN_AVEL * (f + 1) * DT, MAX_ANGLE)
        bot_angle = max(-PIN_AVEL * (f + 1) * DT, -MAX_ANGLE)

        R_top = rot_y(top_angle)
        R_bot = rot_y(bot_angle)

        x_ref = model.particle_q.numpy().copy().astype(np.float64)

        v_top = verts_init[top_pins] - center
        x_ref[top_pins] = (v_top @ R_top.T) + center

        v_bot = verts_init[bot_pins] - center
        x_ref[bot_pins] = (v_bot @ R_bot.T) + center

        solver.set_pin_targets(x_ref.astype(np.float32))

        state_in.clear_forces()
        if wp.is_cuda_available():
            wp.synchronize()
        t0 = time.perf_counter()
        solver.step(state_in, state_out, None, None, DT)
        if wp.is_cuda_available():
            wp.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        state_in, state_out = state_out, state_in

        # Check for NaN
        if nan_frame is None:
            q_np = state_in.particle_q.numpy()
            if not np.all(np.isfinite(q_np)):
                nan_frame = f
                print(f"  [{label}] NaN detected at frame {f}! Stopping simulation.", flush=True)
                break

        # Save snapshot
        if f in snapshot_set:
            q_snap = state_in.particle_q.numpy().copy()
            snaps.append((f, q_snap))
            img_path = out_subdir / f"frame_{f:04d}.png"
            render_frame(q_snap, f, img_path, label)
            print(f"  [{label}] frame {f} rendered -> {img_path.name}", flush=True)

        if f % 100 == 0 and f > 0:
            recent = np.array(step_times[-100:]) * 1000.0
            print(
                f"  [{label}] frame {f}/{NUM_FRAMES}  mean_100={recent.mean():.1f} ms",
                flush=True,
            )

    times_arr = np.array(step_times) * 1000.0  # ms
    stats = {
        "setup_ms": setup_time * 1000.0,
        "mean_ms": float(times_arr.mean()),
        "median_ms": float(np.median(times_arr)),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "nan_frame": nan_frame,
        "n_steps": len(step_times),
    }
    print(
        f"  [{label}] Done. mean={stats['mean_ms']:.2f} ms  "
        f"median={stats['median_ms']:.2f} ms  p95={stats['p95_ms']:.2f} ms"
        + (f"  NaN@{nan_frame}" if nan_frame is not None else "  stable"),
        flush=True,
    )

    # Save trajectory snapshots
    if snaps:
        snap_frames = [s[0] for s in snaps]
        snap_arrays = np.stack([s[1] for s in snaps], axis=0)
        out_subdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_subdir / "trajectory.npz"),
            frames=np.array(snap_frames, dtype=np.int32),
            positions=snap_arrays,
        )

    return np.array([]), stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Newton FBA TwistingBar CudaTests params")
    parser.add_argument(
        "--energy",
        choices=["neohookean", "arap", "corotational"],
        default="neohookean",
        help="Stretching energy model to use (default: neohookean)",
    )
    args = parser.parse_args()

    wall_start = time.perf_counter()

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}")
    print(f"Warp version: {wp.__version__}")
    print(f"Newton version: {newton.__version__}")
    print(f"Energy: {args.energy}")
    print(f"E={YOUNG:.1e}, nu={NU}, mu={MU:.4e}, lam={LAM:.4e}")
    print(f"Frames={NUM_FRAMES}, dt={DT}, PD_iter={PD_ITERATIONS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load mesh
    print(f"\n=== Loading mesh: {MESH_PATH} ===")
    t0 = time.perf_counter()
    verts, tets = load_medit_mesh(MESH_PATH)
    print(f"  Load time: {(time.perf_counter() - t0) * 1000:.0f} ms")
    N = len(verts)
    T = len(tets)
    print(f"  Vertices: {N} (expect 11340)")
    print(f"  Tets:     {T} (expect ~58956)")

    # Build model
    print(f"\n=== Building model ===")
    t_build0 = time.perf_counter()
    model, top_pins, bot_pins = build_model(verts, tets, MU, LAM)
    t_build1 = time.perf_counter()
    print(f"  Model build: {(t_build1 - t_build0) * 1000:.0f} ms")

    # Run simulation
    label = args.energy
    stretching_model = args.energy  # "neohookean", "arap", "corotational"
    out_subdir = OUT_DIR / label

    print(f"\n=== Newton SolverFBA: {label.upper()} (CudaTests params) ===")
    _, stats = run_newton_energy(
        model=model,
        verts_init=verts,
        top_pins=top_pins,
        bot_pins=bot_pins,
        label=label,
        stretching_model=stretching_model,
        out_subdir=out_subdir,
    )

    # Performance summary
    print("\n=== Performance Summary (CudaTests E=1e9) ===")
    nan_f = stats.get("nan_frame")
    stable_str = f"NaN@{nan_f}" if nan_f is not None else "stable (all 810)"
    completed = stats.get("n_steps", 0)

    header = f"{'':30s} {'setup(ms)':>10} {'mean(ms)':>10} {'median(ms)':>10} {'p95(ms)':>10} {'frames':>8} {'stable':>16}"
    sep = "-" * len(header)
    row = (
        f"  Newton {label:22s}"
        f" {stats.get('setup_ms', 0):>10.1f}"
        f" {stats.get('mean_ms', 0):>10.2f}"
        f" {stats.get('median_ms', 0):>10.2f}"
        f" {stats.get('p95_ms', 0):>10.2f}"
        f" {completed:>8}"
        f" {stable_str:>16}"
    )
    print(header)
    print(sep)
    print(row)

    # Save perf summary
    perf_path = OUT_DIR / "perf_summary.txt"
    note = (
        "Newton FBA CudaTests params: E=1e9, nu=0.45, 810 frames, dt=0.01, PD_iter=5\n"
        f"Mesh: cube_volume_11340P.mesh (11340 verts, ~58956 tets)\n"
        f"Device: {device}\n\n"
    )
    with open(str(perf_path), "w") as fh:
        fh.write(note)
        fh.write(header + "\n")
        fh.write(sep + "\n")
        fh.write(row + "\n")
        fh.write("\n")
        fh.write(f"RealSim NH (CudaTests, E=1e9): see /tmp/realsim_twistbar_nh.log\n")
        fh.write(f"Comparison table (fill in after RealSim run):\n\n")
        fh.write(f"{'':30s} {'frames':>8} {'stable':>16} {'mean(ms)':>10} {'median(ms)':>10} {'p95(ms)':>10}\n")
        fh.write(f"{'-'*90}\n")
        fh.write(
            f"  Newton NH (CudaTests, E=1e9)  {completed:>8} {stable_str:>16} "
            f"{stats.get('mean_ms', 0):>10.2f} {stats.get('median_ms', 0):>10.2f} "
            f"{stats.get('p95_ms', 0):>10.2f}\n"
        )
        fh.write(f"  RealSim NH (CudaTests, E=1e9) {'?':>8} {'?':>16} {'?':>10} {'?':>10} {'?':>10}\n")

    print(f"\n  Perf summary: {perf_path}")
    wall_total = time.perf_counter() - wall_start
    print(f"=== Total wall time: {wall_total:.1f}s ({wall_total / 60:.1f} min) ===")


if __name__ == "__main__":
    main()
