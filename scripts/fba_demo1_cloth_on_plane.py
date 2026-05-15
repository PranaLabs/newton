# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Demo 1: 16x16 cloth falling onto a ground plane.

Runs two sub-demos:
  A) friction=False (Stage A — pure normal contact)
  B) friction=True  (Stage B — Coulomb friction, mu=0.3)

Snapshots at frames [0, 30, 60, 120, 240, 480] for each variant.
Saves PNGs to scripts/contact_demos_out/demo1/.

Usage:
    uv run python scripts/fba_demo1_cloth_on_plane.py
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

OUTPUT_DIR = Path(__file__).parent / "contact_demos_out" / "demo1"
SNAPSHOT_FRAMES = [0, 15, 30, 60, 120, 180]
TOTAL_FRAMES = 180
DT = 1.0 / 60.0
DIM = 8  # 8x8 cloth grid (M up to ~81 contacts is tractable for dense Schur)
CELL_SIZE = 0.1  # 0.8 m total cloth width
CLOTH_HEIGHT = 0.6  # starting height above plane


def build_model(friction_mu: float) -> tuple:
    """Build a 16x16 cloth above a Z=0 ground plane.

    Args:
        friction_mu: Per-shape material friction mu (set on ground plane shape).

    Returns:
        Tuple of (model, pipeline, contacts, solver_no_friction, solver_friction).
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

    # Ground plane at Z=0 (Z-up).
    builder.add_ground_plane()

    # Set friction mu on the shape (ground plane is shape index 0).
    # We'll set this after finalize on the model.

    # 16x16 cloth grid starting at origin, elevated by CLOTH_HEIGHT.
    cloth_offset = wp.vec3(
        -DIM * CELL_SIZE / 2.0,
        -DIM * CELL_SIZE / 2.0,
        CLOTH_HEIGHT,
    )
    builder.add_cloth_grid(
        pos=cloth_offset,
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=DIM,
        dim_y=DIM,
        cell_x=CELL_SIZE,
        cell_y=CELL_SIZE,
        mass=0.05,  # 0.05 kg per vertex
        tri_ke=1.0e4,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=1.0e-2,
        edge_kd=0.0,
    )
    model = builder.finalize()

    # Set friction mu on the ground plane shape.
    if hasattr(model, "shape_material_mu") and model.shape_material_mu is not None:
        mu_arr = model.shape_material_mu.numpy()
        mu_arr[0] = friction_mu
        model.shape_material_mu.assign(mu_arr.astype(np.float32))

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.1)
    contacts = pipeline.contacts()
    return model, pipeline, contacts


def render_frame(ax, q: np.ndarray, frame: int, title: str) -> None:
    """Render cloth particle positions as a 3D scatter."""
    ax.clear()
    # Subsample for speed
    step = max(1, len(q) // 200)
    pts = q[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
    ax.set_title(f"{title} — frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(-0.2, CLOTH_HEIGHT + 0.1)


def run_demo(use_friction: bool) -> dict:
    """Run the cloth-on-plane demo.

    Args:
        use_friction: If True, use Stage B Coulomb friction (mu=0.3).

    Returns:
        Dict mapping frame_index -> particle positions array.
    """
    friction_mu = 0.3 if use_friction else 0.0
    model, pipeline, contacts = build_model(friction_mu)

    # Per-pair override: sqrt(particle_mu * shape_mu). Use override array so
    # we control the combined mu precisely.
    particle_count = model.particle_count
    n_contacts_max = particle_count  # at most one contact per particle
    mu_override = np.full(n_contacts_max, 0.3, dtype=np.float64) if use_friction else None

    solver = SolverFBA(
        model,
        iterations=5,
        friction=use_friction,
        mu_per_pair_override=mu_override,
    )
    s_in = model.state()
    s_out = model.state()

    snapshots: dict[int, np.ndarray] = {}
    step_times = []

    label = "friction" if use_friction else "no-friction"

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

    print(
        f"  [{label}] finite={finite_ok}  min_z={min_z:.3f}  "
        f"mean_step={mean_step_ms:.1f} ms"
    )

    return snapshots, mean_step_ms, finite_ok, min_z


def save_snapshot_grid(
    snapshots_a: dict,
    snapshots_b: dict,
    label_a: str,
    label_b: str,
) -> Path:
    """Save a 2-row grid of snapshots (A=no-friction, B=friction)."""
    frames = [f for f in SNAPSHOT_FRAMES if f in snapshots_a]
    n_cols = len(frames)

    fig = plt.figure(figsize=(4 * n_cols, 8))
    fig.suptitle("Demo 1: Cloth on Plane", fontsize=14, y=1.01)

    for col, frame in enumerate(frames):
        # Row 0: no-friction
        ax = fig.add_subplot(2, n_cols, col + 1, projection="3d")
        render_frame(ax, snapshots_a[frame], frame, label_a)
        # Row 1: friction
        ax2 = fig.add_subplot(2, n_cols, n_cols + col + 1, projection="3d")
        render_frame(ax2, snapshots_b[frame], frame, label_b)

    plt.tight_layout()
    out = OUTPUT_DIR / "cloth_on_plane_comparison.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    wp.init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Demo 1: Cloth on Plane")
    print("=" * 50)

    print("Running sub-demo A: no friction ...")
    t0 = time.perf_counter()
    snaps_a, ms_a, ok_a, min_z_a = run_demo(use_friction=False)
    wall_a = time.perf_counter() - t0

    print("Running sub-demo B: friction mu=0.3 ...")
    t0 = time.perf_counter()
    snaps_b, ms_b, ok_b, min_z_b = run_demo(use_friction=True)
    wall_b = time.perf_counter() - t0

    # Save individual snapshots for each variant.
    for frame, q in snaps_a.items():
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        render_frame(ax, q, frame, "No friction")
        out = OUTPUT_DIR / f"no_friction_frame{frame:04d}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)

    for frame, q in snaps_b.items():
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        render_frame(ax, q, frame, "Friction mu=0.3")
        out = OUTPUT_DIR / f"friction_frame{frame:04d}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)

    # Comparison grid.
    grid_path = save_snapshot_grid(snaps_a, snaps_b, "No friction", "Friction mu=0.3")

    # Performance summary.
    print()
    print("Performance summary:")
    print(f"  no-friction  : {ms_a:.1f} ms/step, wall {wall_a:.1f}s, stable={ok_a}, min_z={min_z_a:.3f}")
    print(f"  friction mu=0.3: {ms_b:.1f} ms/step, wall {wall_b:.1f}s, stable={ok_b}, min_z={min_z_b:.3f}")
    print(f"\nComparison grid saved: {grid_path}")

    # Write perf table to file for README.
    perf_path = OUTPUT_DIR / "perf.txt"
    perf_path.write_text(
        f"Demo 1: Cloth on Plane ({DIM}x{DIM} grid, {TOTAL_FRAMES} frames, dt={DT})\n"
        f"no-friction : {ms_a:.2f} ms/step  stable={ok_a}  min_z={min_z_a:.4f}\n"
        f"friction mu=0.3: {ms_b:.2f} ms/step  stable={ok_b}  min_z={min_z_b:.4f}\n"
    )
    print(f"Perf table saved: {perf_path}")


if __name__ == "__main__":
    main()
