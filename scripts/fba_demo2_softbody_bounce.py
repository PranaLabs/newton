# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Demo 2: 4x4x4 soft cube (FBA tet ARAP) bouncing on a ground plane.

No friction (Stage A) — shows pure bounce/squish behavior under gravity.

Snapshots at frames [0, 20, 50, 80, 120, 200].
Saves PNGs to scripts/contact_demos_out/demo2/.

Usage:
    uv run python scripts/fba_demo2_softbody_bounce.py
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA

OUTPUT_DIR = Path(__file__).parent / "contact_demos_out" / "demo2"
SNAPSHOT_FRAMES = [0, 20, 50, 80, 120, 200]
TOTAL_FRAMES = 200
DT = 1.0 / 60.0

# Tet cube parameters
GRID_DIM = 4  # 4x4x4 tets
CELL_SIZE = 0.1  # 0.4 m cube side
DROP_HEIGHT = 1.0  # starting height above plane (center z)


def build_model() -> tuple:
    """Build a soft tet cube dropped above a ground plane (no friction)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

    # Ground plane at Z=0.
    builder.add_ground_plane()

    # Soft tet cube elevated by DROP_HEIGHT.
    # add_soft_grid places the centre at pos.
    center_z = DROP_HEIGHT + GRID_DIM * CELL_SIZE / 2.0
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, center_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=GRID_DIM,
        dim_y=GRID_DIM,
        dim_z=GRID_DIM,
        cell_x=CELL_SIZE,
        cell_y=CELL_SIZE,
        cell_z=CELL_SIZE,
        density=500.0,
        k_mu=5.0e3,
        k_lambda=5.0e3,
        k_damp=0.0,
    )
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.1)
    contacts = pipeline.contacts()
    return model, pipeline, contacts


def render_frame(ax, q: np.ndarray, frame: int, title: str) -> None:
    """Render tet-body particle positions as a 3D scatter."""
    ax.clear()
    step = max(1, len(q) // 300)
    pts = q[::step]
    c = pts[:, 2]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=c, cmap="plasma")
    ax.set_title(f"{title} — frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    z_max = DROP_HEIGHT + GRID_DIM * CELL_SIZE + 0.2
    ax.set_zlim(-0.2, z_max)


def run_demo() -> tuple:
    """Run the softbody bounce demo (no friction).

    Returns:
        Tuple of (snapshots dict, mean_step_ms, finite_ok, min_z_final).
    """
    model, pipeline, contacts = build_model()

    solver = SolverFBA(
        model,
        iterations=5,
        friction=False,
        stretching_model="arap",
    )
    s_in = model.state()
    s_out = model.state()

    snapshots: dict[int, np.ndarray] = {}
    step_times = []

    for frame in range(TOTAL_FRAMES + 1):
        if frame in SNAPSHOT_FRAMES:
            snapshots[frame] = s_in.particle_q.numpy().copy()

        if frame == TOTAL_FRAMES:
            break

        t0 = time.perf_counter()
        s_in.clear_forces()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, DT)
        s_in, s_out = s_out, s_in
        step_times.append(time.perf_counter() - t0)

    q_final = s_in.particle_q.numpy()
    finite_ok = bool(np.all(np.isfinite(q_final)))
    min_z = float(q_final[:, 2].min())
    mean_step_ms = 1000.0 * float(np.mean(step_times))

    print(f"  [softbody-bounce] finite={finite_ok}  min_z={min_z:.3f}  mean_step={mean_step_ms:.1f} ms")
    return snapshots, mean_step_ms, finite_ok, min_z


def save_snapshots(snapshots: dict) -> list[Path]:
    """Save each snapshot as a separate PNG."""
    paths = []
    for frame, q in sorted(snapshots.items()):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        render_frame(ax, q, frame, "Softbody bounce (ARAP, no friction)")
        out = OUTPUT_DIR / f"softbody_bounce_frame{frame:04d}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)
        paths.append(out)
    return paths


def save_snapshot_strip(snapshots: dict) -> Path:
    """Save a single-row strip of all snapshots."""
    frames = sorted(snapshots.keys())
    n = len(frames)
    fig = plt.figure(figsize=(4 * n, 4))
    fig.suptitle("Demo 2: Softbody Bounce (ARAP, no friction)", fontsize=12)
    for col, frame in enumerate(frames):
        ax = fig.add_subplot(1, n, col + 1, projection="3d")
        render_frame(ax, snapshots[frame], frame, "")
    plt.tight_layout()
    out = OUTPUT_DIR / "softbody_bounce_strip.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    wp.init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Demo 2: Softbody Bounce on Plane")
    print("=" * 50)

    t0 = time.perf_counter()
    snaps, ms, ok, min_z = run_demo()
    wall_time = time.perf_counter() - t0

    paths = save_snapshots(snaps)
    strip_path = save_snapshot_strip(snaps)

    print()
    print("Performance summary:")
    print(f"  mean_step={ms:.1f} ms  wall={wall_time:.1f}s  stable={ok}  min_z={min_z:.3f}")
    print(f"\nSnapshot strip: {strip_path}")
    for p in paths:
        print(f"  {p}")

    perf_path = OUTPUT_DIR / "perf.txt"
    perf_path.write_text(
        f"Demo 2: Softbody Bounce ({GRID_DIM}x{GRID_DIM}x{GRID_DIM} tet cube, "
        f"{TOTAL_FRAMES} frames, dt={DT})\n"
        f"ARAP no-friction: {ms:.2f} ms/step  stable={ok}  min_z={min_z:.4f}  "
        f"wall={wall_time:.1f}s\n"
    )
    print(f"Perf table saved: {perf_path}")


if __name__ == "__main__":
    main()
