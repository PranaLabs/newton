# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Twisting bar demo for SolverFBA with ARAP, Corotational, and Neo-Hookean energy models.

Reproduces RealSim's TwistingBar demo (1200 frames, dt=0.01, 10 PD iters,
bar x∈[-1,1] y∈[-2,2] z∈[-1,1], 11340 vertices ~59K tets, NH E=1e6 nu=0.45)
and side-by-side compares Newton FBA with the RealSim reference run.

Usage::

    uv run python scripts/fba_twisting_bar.py

Outputs (all in scripts/twisting_bar_out/):
    {arap,corot,neohookean}/frame_*.png    — rendered snapshots
    {arap,corot,neohookean}/trajectory.npz
    realsim/frame_*.png                    — if RealSim run succeeds
    summary.png                            — 4-column montage
    timings.npz                            — per-energy step times
    perf_summary.txt                       — performance comparison table
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
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
OUT_DIR = SCRIPT_DIR / "twisting_bar_out"
MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/cube_volume_11340P.mesh")
REALSIM_ABC = Path("/home/ziqiu/work/RealSim_py/realsim_py/simulation/output_abc/TwistingBar/output_obj_0.abc")
DUMP_ABC = Path("/tmp/dump_abc_traj")

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DT = 0.01
N_FRAMES = 1200
N_ITER = 10
TOTAL_MASS = 1000.0  # kg

# Lamé from E=1e6, nu=0.45:
#   mu  = E / (2*(1+nu))       = 1e6 / 2.9  ≈ 3.4483e5
#   lam = E*nu / ((1+nu)*(1-2*nu)) = 1e6*0.45 / (1.45*0.10) ≈ 3.1034e6
E_REF = 1.0e6
NU_REF = 0.45
MU = E_REF / (2.0 * (1.0 + NU_REF))
LAM = E_REF * NU_REF / ((1.0 + NU_REF) * (1.0 - 2.0 * NU_REF))

# Rolling pin parameters (degrees → radians via π/2 = 90°)
MAX_ANGLE = math.pi / 2.0  # 90 degrees
PIN_AVEL = 10.0  # rad/s
PIN_Y_TOP = 1.99
PIN_Y_BOT = -1.99

# Snapshot frames
SNAPSHOT_FRAMES = [0, 8, 16, 24, 50, 100, 200, 400, 800, 1199]


# ---------------------------------------------------------------------------
# 1. Medit .mesh parser
# ---------------------------------------------------------------------------
def load_medit_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Medit .mesh file (text format, 1-based tet indices).

    Args:
        path: Path to the .mesh file.

    Returns:
        Tuple ``(vertices, tets)`` where ``vertices`` has shape ``(N, 3)``
        and ``tets`` has shape ``(T, 4)`` with 0-based indices.
    """
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
            # Skip other section headers (Triangles, MeshVersionFormatted, Dimension, ...)
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
                    # 1-based -> 0-based
                    tets.append((int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1))
                    tet_count_read += 1
                    if tet_count_read == n_tets:
                        state = None
                continue
            if state == "skip_section":
                # first line is count, rest are data rows — we skip all
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
    """Build Newton ModelBuilder for twisting bar.

    Returns:
        Tuple ``(model, top_pins, bot_pins)`` where pin arrays are 0-based
        vertex indices with zero inv_mass (kinematic).
    """
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
        f"  BBox: x=[{bbox_min[0]:.3f},{bbox_max[0]:.3f}] y=[{bbox_min[1]:.3f},{bbox_max[1]:.3f}] z=[{bbox_min[2]:.3f},{bbox_max[2]:.3f}]"
    )
    print(f"  Top pins (y>{PIN_Y_TOP}): {len(top_pins)}  Bot pins (y<{PIN_Y_BOT}): {len(bot_pins)}")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)

    # Bulk add particles (avoids 11k Python-loop overhead via extend())
    pos_list = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    vel_list = [wp.vec3(0.0, 0.0, 0.0)] * N
    mass_list = [mass_per_particle] * N
    builder.add_particles(pos_list, vel_list, mass_list)

    # Pin particles: set mass=0 for kinematic
    for idx in top_pins:
        builder.particle_mass[int(idx)] = 0.0
    for idx in bot_pins:
        builder.particle_mass[int(idx)] = 0.0

    # Add tetrahedra (uses add_tetrahedron loop — ~59K iterations)
    print("  Adding tetrahedra...", flush=True)
    tets_flat = tets.flatten().tolist()
    for t in range(T):
        i0, i1, i2, i3 = tets_flat[t * 4], tets_flat[t * 4 + 1], tets_flat[t * 4 + 2], tets_flat[t * 4 + 3]
        builder.add_tetrahedron(i0, i1, i2, i3, k_mu=float(mu), k_lambda=float(lam), k_damp=0.0)

    model = builder.finalize()
    print(f"  Model: particle_count={model.particle_count}, tet_count={model.tet_count}")
    return model, top_pins, bot_pins


# ---------------------------------------------------------------------------
# 4. Render a frame with matplotlib 3D scatter
# ---------------------------------------------------------------------------
def render_frame(
    q: np.ndarray,
    frame_idx: int,
    out_path: Path,
    label: str,
) -> None:
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
# 5. Run Newton simulation for one energy model
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
    """Run Newton SolverFBA for N_FRAMES steps with rolling pin update.

    Returns:
        Tuple of ``(trajectory_array, timings_dict)`` where trajectory is
        shape ``(len(SNAPSHOT_FRAMES), N, 3)`` in float32.
    """
    # Solver
    mu_arg = MU if stretching_model != "arap" else None
    lam_arg = LAM if stretching_model != "arap" else None
    kw = {"mu": mu_arg, "lam": lam_arg} if mu_arg is not None else {}

    t_setup_0 = time.perf_counter()
    solver = SolverFBA(model, iterations=N_ITER, pin_stiffness=1e12, stretching_model=stretching_model, **kw)
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

    print(f"  [{label}] Running {N_FRAMES} frames...", flush=True)
    for f in range(N_FRAMES):
        # Compute rolling angles
        top_angle = min(PIN_AVEL * (f + 1) * DT, MAX_ANGLE)
        bot_angle = max(-PIN_AVEL * (f + 1) * DT, -MAX_ANGLE)

        R_top = rot_y(top_angle)
        R_bot = rot_y(bot_angle)

        # Build new x_ref: start from model initial positions
        x_ref = model.particle_q.numpy().copy().astype(np.float64)  # (N, 3) float32->float64

        # Update top pins
        v_top = verts_init[top_pins] - center
        x_ref[top_pins] = (v_top @ R_top.T) + center

        # Update bottom pins
        v_bot = verts_init[bot_pins] - center
        x_ref[bot_pins] = (v_bot @ R_bot.T) + center

        solver.set_pin_targets(x_ref.astype(np.float32))

        # Step
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
            # Render
            img_path = out_subdir / f"frame_{f:04d}.png"
            render_frame(q_snap, f, img_path, label)
            print(f"  [{label}] frame {f} rendered -> {img_path.name}", flush=True)

        if f % 100 == 0 and f > 0:
            recent = np.array(step_times[-100:]) * 1000.0
            print(
                f"  [{label}] frame {f}/{N_FRAMES}  mean_100={recent.mean():.1f} ms",
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
    snap_frames = [s[0] for s in snaps]
    snap_arrays = np.stack([s[1] for s in snaps], axis=0)  # (S, N, 3)
    out_subdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_subdir / "trajectory.npz"),
        frames=np.array(snap_frames, dtype=np.int32),
        positions=snap_arrays,
    )

    return snap_arrays, stats


# ---------------------------------------------------------------------------
# 6. Read RealSim trajectory via dump_abc_traj binary
# ---------------------------------------------------------------------------
def read_realsim_trajectory_abc(abc_path: Path, dump_bin: Path) -> np.ndarray | None:
    """Extract per-frame vertex positions from an Alembic file."""
    if not dump_bin.exists():
        print(f"  dump_abc_traj binary not found at {dump_bin}. Skipping RealSim render.")
        return None
    if not abc_path.exists():
        print(f"  RealSim abc not found at {abc_path}. Skipping RealSim render.")
        return None
    try:
        result = subprocess.run(
            [str(dump_bin), str(abc_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        lines = result.stdout.strip().split("\n")
        pos = 0
        n_frames = int(lines[pos])
        pos += 1
        frames = []
        for _f in range(n_frames):
            n_verts = int(lines[pos])
            pos += 1
            verts = []
            for _v in range(n_verts):
                parts = lines[pos].split()
                pos += 1
                verts.append([float(p) for p in parts])
            frames.append(verts)
        return np.array(frames, dtype=np.float32)
    except Exception as exc:
        print(f"  Failed to read RealSim trajectory: {exc}")
        return None


# ---------------------------------------------------------------------------
# 7. Run RealSim TwistingBar
# ---------------------------------------------------------------------------
def run_realsim() -> tuple[np.ndarray | None, dict | None]:
    """Run RealSim TwistingBar demo in offline mode, collect timings.

    Returns:
        Tuple ``(trajectory_array, timings)`` or ``(None, None)`` on failure.
    """
    realsim_dir = Path("/home/ziqiu/work/RealSim_py/realsim_py")
    scene_path = realsim_dir / "simulation/config/Demos/TwistingBar/scene.json"
    binary = realsim_dir / "build/bin/RealSim"

    if not binary.exists():
        print(f"  RealSim binary not found: {binary}")
        return None, None
    if not scene_path.exists():
        print(f"  Scene config not found: {scene_path}")
        return None, None

    # Clean old output
    abc_dir = realsim_dir / "simulation/output_abc/TwistingBar"
    if abc_dir.exists():
        shutil.rmtree(str(abc_dir))

    print("  Launching RealSim TwistingBar (timeout=1800s)...", flush=True)
    try:
        result = subprocess.run(
            [str(binary), "-m", str(scene_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
            cwd=str(realsim_dir),
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        print("  RealSim timed out after 30 minutes.")
        return None, None
    except Exception as exc:
        print(f"  RealSim failed: {exc}")
        return None, None

    # Save log
    log_path = OUT_DIR / "realsim_stdout.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(log_path), "w") as fh:
        fh.write(stdout)

    # Parse per-step timings from RealSim stdout.
    # RealSim output contains ANSI color codes; strip them before matching.
    # Line format (after stripping):
    #   "LocalGlobalSolver::update, time cost = 14.5 ms, local cost = ... ms, global cost = ... ms"
    # We parse the "global cost" (SpMV solve, directly comparable to Newton step time).
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    global_pattern = re.compile(r"LocalGlobalSolver::update.*?global cost\s*=\s*([0-9]+\.?[0-9]*)\s*ms")
    total_pattern = re.compile(r"LocalGlobalSolver::update.*?time cost\s*=\s*([0-9]+\.?[0-9]*)\s*ms")
    rs_times = []  # global (SpMV) times
    rs_total_times = []  # full step times
    for line in stdout.split("\n"):
        clean = ansi_escape.sub("", line)
        m = global_pattern.search(clean)
        if m:
            rs_times.append(float(m.group(1)))
        m = total_pattern.search(clean)
        if m:
            rs_total_times.append(float(m.group(1)))

    print(f"  RealSim: found {len(rs_times)} global timing lines in stdout", flush=True)
    rs_timings = None
    if rs_times:
        arr = np.array(rs_times)
        arr_total = np.array(rs_total_times)
        rs_timings = {
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "total_mean_ms": float(arr_total.mean()),
            "total_median_ms": float(np.median(arr_total)),
            "n_steps": len(rs_times),
        }
        print(
            f"  RealSim global(SpMV): mean={rs_timings['mean_ms']:.2f} ms "
            f"median={rs_timings['median_ms']:.2f} ms p95={rs_timings['p95_ms']:.2f} ms"
        )
        print(
            f"  RealSim total step:   mean={rs_timings['total_mean_ms']:.2f} ms "
            f"median={rs_timings['total_median_ms']:.2f} ms"
        )

    # Read trajectory
    traj = read_realsim_trajectory_abc(REALSIM_ABC, DUMP_ABC)
    return traj, rs_timings


# ---------------------------------------------------------------------------
# 8. Build summary montage
# ---------------------------------------------------------------------------
def build_montage(
    arap_dir: Path,
    corot_dir: Path,
    nh_dir: Path,
    realsim_dir: Path,
    out_path: Path,
) -> None:
    """Build a 4-col x N-row montage from saved frame PNGs."""
    frames_to_show = SNAPSHOT_FRAMES
    cols = ["Newton ARAP", "Newton Corot", "Newton NH", "RealSim NH"]
    dirs = [arap_dir, corot_dir, nh_dir, realsim_dir]

    n_rows = len(frames_to_show)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col_idx, (col_label, d) in enumerate(zip(cols, dirs, strict=True)):
        for row_idx, f in enumerate(frames_to_show):
            ax = axes[row_idx, col_idx]
            img_path = d / f"frame_{f:04d}.png"
            if img_path.exists():
                img = plt.imread(str(img_path))
                ax.imshow(img)
            else:
                ax.set_facecolor("#cccccc")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_label, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"f={f}", fontsize=8, rotation=0, labelpad=35)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  Montage saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    wall_start = time.perf_counter()

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}")
    print(f"Warp version: {wp.__version__}")
    print(f"Newton version: {newton.__version__}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load mesh ----
    print(f"\n=== Loading mesh: {MESH_PATH} ===")
    t0 = time.perf_counter()
    verts, tets = load_medit_mesh(MESH_PATH)
    print(f"  Load time: {(time.perf_counter() - t0) * 1000:.0f} ms")
    N = len(verts)
    T = len(tets)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    print(f"  Vertices: {N} (expect ~11340)")
    print(f"  Tets:     {T} (expect ~58956)")
    print(f"  BBox X: [{bbox_min[0]:.4f}, {bbox_max[0]:.4f}]  (expect [-1,1])")
    print(f"  BBox Y: [{bbox_min[1]:.4f}, {bbox_max[1]:.4f}]  (expect [-2,2])")
    print(f"  BBox Z: [{bbox_min[2]:.4f}, {bbox_max[2]:.4f}]  (expect [-1,1])")

    y = verts[:, 1]
    top_pins_count = int(np.sum(y > PIN_Y_TOP))
    bot_pins_count = int(np.sum(y < PIN_Y_BOT))
    print(f"  Top pins (y>{PIN_Y_TOP}): {top_pins_count}")
    print(f"  Bot pins (y<{PIN_Y_BOT}): {bot_pins_count}")

    print(f"\nLamé parameters: E={E_REF:.1e}, nu={NU_REF}")
    print(f"  mu  = {MU:.4e} Pa")
    print(f"  lam = {LAM:.4e} Pa")

    # ---- Step 2: Build models & run Newton (3 energies) ----
    all_newton_stats = {}
    energy_configs = [
        ("arap", "arap"),
        ("corot", "corotational"),
        ("neohookean", "neohookean"),
    ]

    for label, stretching_model in energy_configs:
        print(f"\n=== Newton SolverFBA: {label.upper()} ===")
        out_subdir = OUT_DIR / label

        t_build0 = time.perf_counter()
        model, top_pins, bot_pins = build_model(verts, tets, MU, LAM)
        t_build1 = time.perf_counter()
        print(f"  Model build: {(t_build1 - t_build0) * 1000:.0f} ms")

        _, stats = run_newton_energy(
            model=model,
            verts_init=verts,
            top_pins=top_pins,
            bot_pins=bot_pins,
            label=label,
            stretching_model=stretching_model,
            out_subdir=out_subdir,
        )
        all_newton_stats[label] = stats

    # ---- Step 3: Run RealSim ----
    print("\n=== RealSim TwistingBar reference run ===")
    elapsed_so_far = time.perf_counter() - wall_start
    TIME_BUDGET = 3600.0  # 60 minutes total
    REALSIM_BUDGET = 1800.0  # 30 minutes for RealSim
    realsim_out_dir = OUT_DIR / "realsim"
    realsim_out_dir.mkdir(parents=True, exist_ok=True)

    if elapsed_so_far > TIME_BUDGET - REALSIM_BUDGET:
        print(f"  Time budget exhausted ({elapsed_so_far:.0f}s elapsed). Skipping RealSim.")
        rs_traj = None
        rs_timings = None
    else:
        rs_traj, rs_timings = run_realsim()

    # Render RealSim snapshots if available
    if rs_traj is not None:
        n_rs_frames = rs_traj.shape[0]
        print(f"  RealSim trajectory: {rs_traj.shape}")
        for f in SNAPSHOT_FRAMES:
            # RealSim ABC starts at frame 0 (initial config), so frame f in trajectory
            # corresponds to f in 0-indexed (frame 0 = after step 1 in most configs)
            rs_f = min(f, n_rs_frames - 1)
            img_path = realsim_out_dir / f"frame_{f:04d}.png"
            render_frame(rs_traj[rs_f], f, img_path, "RealSim NH")
            print(f"  RealSim frame {f} -> {img_path.name}")
    else:
        print("  RealSim trajectory not available; skipping RealSim renders.")

    # ---- Step 4: Summary montage ----
    print("\n=== Building summary montage ===")
    summary_path = OUT_DIR / "summary.png"
    build_montage(
        arap_dir=OUT_DIR / "arap",
        corot_dir=OUT_DIR / "corot",
        nh_dir=OUT_DIR / "neohookean",
        realsim_dir=realsim_out_dir,
        out_path=summary_path,
    )

    # ---- Step 5: Save timings ----
    print("\n=== Saving timings ===")
    timings_data = {}
    for label, stats in all_newton_stats.items():
        for k, v in stats.items():
            timings_data[f"{label}_{k}"] = v
    if rs_timings is not None:
        for k, v in rs_timings.items():
            timings_data[f"realsim_{k}"] = v

    np.savez(str(OUT_DIR / "timings.npz"), **{k: np.array(v) for k, v in timings_data.items()})
    print("  timings.npz saved")

    # ---- Step 6: Performance summary table ----
    print("\n=== Performance Summary ===")
    header = f"{'':20s} {'setup(ms)':>10} {'mean(ms)':>10} {'median(ms)':>10} {'p95(ms)':>10} {'stable':>8}"
    print(header)
    print("-" * len(header))
    lines = [header, "-" * len(header)]

    for label in ["arap", "corot", "neohookean"]:
        st = all_newton_stats.get(label, {})
        nan_f = st.get("nan_frame")
        stable = f"NaN@{nan_f}" if nan_f is not None else "yes"
        row = (
            f"  Newton {label:12s}"
            f" {st.get('setup_ms', 0):>10.1f}"
            f" {st.get('mean_ms', 0):>10.2f}"
            f" {st.get('median_ms', 0):>10.2f}"
            f" {st.get('p95_ms', 0):>10.2f}"
            f" {stable:>8}"
        )
        print(row)
        lines.append(row)

    if rs_timings is not None:
        row = (
            f"  RealSim NH          "
            f" {'N/A':>10}"
            f" {rs_timings.get('mean_ms', 0):>10.2f}"
            f" {rs_timings.get('median_ms', 0):>10.2f}"
            f" {rs_timings.get('p95_ms', 0):>10.2f}"
            f" {'yes':>8}"
        )
        print(row)
        lines.append(row)
    else:
        row = "  RealSim NH           (not run)"
        print(row)
        lines.append(row)

    perf_path = OUT_DIR / "perf_summary.txt"
    with open(str(perf_path), "w") as fh:
        fh.write(f"Newton (device={device}) vs RealSim performance summary\n\n")
        fh.write("\n".join(lines) + "\n")
    print(f"  Saved: {perf_path}")

    wall_total = time.perf_counter() - wall_start
    print(f"\n=== Total wall time: {wall_total:.1f}s ({wall_total / 60:.1f} min) ===")
    print(f"Summary image: {summary_path}")
    print(f"Perf table:    {perf_path}")


if __name__ == "__main__":
    main()
