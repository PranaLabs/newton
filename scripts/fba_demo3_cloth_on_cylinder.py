# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Demo 3: 32x32 cloth dropped onto an angled cylinder.

Runs two sub-demos:
  A) friction=False — cloth slides off freely
  B) friction=True  (mu=0.4) — cloth grips the cylinder

Snapshots at frames [0, 30, 60, 120, 200, 360] for each variant.
Saves PNGs to scripts/contact_demos_out/demo3/.

Usage:
    uv run python scripts/fba_demo3_cloth_on_cylinder.py
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

OUTPUT_DIR = Path(__file__).parent / "contact_demos_out" / "demo3"
SNAPSHOT_FRAMES = [0, 20, 40, 80, 140, 200]
TOTAL_FRAMES = 200
DT = 1.0 / 60.0

# Cloth parameters (reduced for tractable Schur with full-drape contact set)
DIM = 12
CELL_SIZE = 0.08  # ~1.0 m x 1.0 m cloth
CLOTH_HEIGHT = 1.0  # starting height of cloth centre above cylinder top

# Cylinder parameters
CYLINDER_RADIUS = 0.3
CYLINDER_HALF_HEIGHT = 0.8
# Tilt the cylinder ~30 deg around X axis to make friction more visible.
CYLINDER_TILT_DEG = 30.0


def build_model(use_friction: bool) -> tuple:
    """Build a 32x32 cloth above a tilted cylinder.

    Args:
        use_friction: Whether to enable friction.

    Returns:
        Tuple of (model, pipeline, contacts).
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

    # Tilted cylinder (static, body=-1).
    # Rotate 30 deg around X axis to tilt the cylinder.
    angle_rad = np.deg2rad(CYLINDER_TILT_DEG)
    # Quaternion for rotation around X: (sin(a/2)*x, 0, 0, cos(a/2))
    qx = float(np.sin(angle_rad / 2.0))
    qw = float(np.cos(angle_rad / 2.0))
    tilt_quat = wp.quat(qx, 0.0, 0.0, qw)

    # Place cylinder so its top is at approximately z=0.5.
    cylinder_center_z = 0.5
    builder.add_shape_cylinder(
        body=-1,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, cylinder_center_z),
            tilt_quat,
        ),
        radius=CYLINDER_RADIUS,
        half_height=CYLINDER_HALF_HEIGHT,
    )

    # Cloth grid centered above the cylinder.
    cloth_start_x = -DIM * CELL_SIZE / 2.0
    cloth_start_y = -DIM * CELL_SIZE / 2.0
    cloth_start_z = cylinder_center_z + CYLINDER_HALF_HEIGHT + CLOTH_HEIGHT
    builder.add_cloth_grid(
        pos=wp.vec3(cloth_start_x, cloth_start_y, cloth_start_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=DIM,
        dim_y=DIM,
        cell_x=CELL_SIZE,
        cell_y=CELL_SIZE,
        mass=0.02,
        tri_ke=8.0e3,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=5.0e-3,
        edge_kd=0.0,
    )
    model = builder.finalize()

    # Set friction mu on the cylinder shape.
    if use_friction and hasattr(model, "shape_material_mu") and model.shape_material_mu is not None:
        mu_arr = model.shape_material_mu.numpy()
        mu_arr[0] = 0.4
        model.shape_material_mu.assign(mu_arr.astype(np.float32))

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.08)
    contacts = pipeline.contacts()
    return model, pipeline, contacts


def render_frame(ax, q: np.ndarray, frame: int, title: str) -> None:
    """Render cloth positions as a 3D scatter."""
    ax.clear()
    step = max(1, len(q) // 300)
    pts = q[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="coolwarm")
    ax.set_title(f"{title} — frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    z_top = 0.5 + CYLINDER_HALF_HEIGHT + CLOTH_HEIGHT + 0.2
    ax.set_zlim(-0.8, z_top)


def run_demo(use_friction: bool) -> tuple:
    """Run the cloth-on-cylinder demo.

    Args:
        use_friction: If True, use Stage B Coulomb friction (mu=0.4).

    Returns:
        Tuple of (snapshots dict, mean_step_ms, finite_ok, min_z_final).
    """
    model, pipeline, contacts = build_model(use_friction)

    particle_count = model.particle_count
    mu_override = np.full(particle_count, 0.4, dtype=np.float64) if use_friction else None

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

    label = "friction mu=0.4" if use_friction else "no-friction"

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

    print(f"  [{label}] finite={finite_ok}  min_z={min_z:.3f}  mean_step={mean_step_ms:.1f} ms")
    return snapshots, mean_step_ms, finite_ok, min_z


def save_snapshot_grid(
    snapshots_a: dict,
    snapshots_b: dict,
    label_a: str,
    label_b: str,
) -> Path:
    """Save a 2-row comparison grid."""
    frames = [f for f in SNAPSHOT_FRAMES if f in snapshots_a]
    n_cols = len(frames)

    fig = plt.figure(figsize=(4 * n_cols, 8))
    fig.suptitle("Demo 3: Cloth on Cylinder", fontsize=14, y=1.01)

    for col, frame in enumerate(frames):
        ax = fig.add_subplot(2, n_cols, col + 1, projection="3d")
        render_frame(ax, snapshots_a[frame], frame, label_a)
        ax2 = fig.add_subplot(2, n_cols, n_cols + col + 1, projection="3d")
        render_frame(ax2, snapshots_b[frame], frame, label_b)

    plt.tight_layout()
    out = OUTPUT_DIR / "cloth_on_cylinder_comparison.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    wp.init()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Demo 3: Cloth on Cylinder")
    print("=" * 50)

    print("Running sub-demo A: no friction ...")
    t0 = time.perf_counter()
    snaps_a, ms_a, ok_a, min_z_a = run_demo(use_friction=False)
    wall_a = time.perf_counter() - t0

    print("Running sub-demo B: friction mu=0.4 ...")
    t0 = time.perf_counter()
    snaps_b, ms_b, ok_b, min_z_b = run_demo(use_friction=True)
    wall_b = time.perf_counter() - t0

    # Save individual snapshots.
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
        render_frame(ax, q, frame, "Friction mu=0.4")
        out = OUTPUT_DIR / f"friction_frame{frame:04d}.png"
        fig.savefig(out, dpi=80)
        plt.close(fig)

    # Comparison grid.
    grid_path = save_snapshot_grid(snaps_a, snaps_b, "No friction", "Friction mu=0.4")

    print()
    print("Performance summary:")
    print(f"  no-friction   : {ms_a:.1f} ms/step, wall {wall_a:.1f}s, stable={ok_a}, min_z={min_z_a:.3f}")
    print(f"  friction mu=0.4: {ms_b:.1f} ms/step, wall {wall_b:.1f}s, stable={ok_b}, min_z={min_z_b:.3f}")
    print(f"\nComparison grid saved: {grid_path}")

    perf_path = OUTPUT_DIR / "perf.txt"
    perf_path.write_text(
        f"Demo 3: Cloth on Cylinder ({DIM}x{DIM} grid, {TOTAL_FRAMES} frames, dt={DT})\n"
        f"no-friction   : {ms_a:.2f} ms/step  stable={ok_a}  min_z={min_z_a:.4f}\n"
        f"friction mu=0.4: {ms_b:.2f} ms/step  stable={ok_b}  min_z={min_z_b:.4f}\n"
    )
    print(f"Perf table saved: {perf_path}")


if __name__ == "__main__":
    main()
