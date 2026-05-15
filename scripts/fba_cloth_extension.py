# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth extension demo — reproduces RealSim's ErrorAnalysis_Extension scene.

Runs for ARAP, Corotational, Neo-Hookean. Saves PNG snapshots and trajectories.

Usage:
    uv run python scripts/fba_cloth_extension.py
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA

REALSIM_MESH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/cloth/square_10225P.obj")
OUTPUT_ROOT = Path(__file__).parent / "cloth_extension_out"

# Match RealSim config
TIMESTEP = 0.01
NUM_FRAMES = 802
PD_ITERATIONS = 5
YOUNG = 1.0e8
POISSON = 0.45
MASS_TOTAL = 1000.0
PULL_VEL = 0.1  # m/s
PULL_MAX_DISP = 1.0  # m

# Lame parameters (mu, lam) from RealSim ElasticEnergy::setElasticParameter
MU = YOUNG / (2.0 * (1.0 + POISSON))  # ~3.448e7
LAM_NH = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))  # ~3.103e8

SNAPSHOT_FRAMES = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800]


def load_obj(path: Path):
    """Parse .obj returning (vertices Nx3 float32, indices Mx3 int32)."""
    verts = []
    tris = []
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


def transform_vertices(
    verts: np.ndarray,
    rotation_deg=(90.0, 0.0, 0.0),
    translation=(0.0, 0.0, -3.0),
) -> np.ndarray:
    """Apply RealSim's cloth.json transformation: rotation around X then translation.

    cloth.json: rotation=[90,0,0] around X-axis, trans=[0,0,-3].
    Rx(90): y -> z, z -> -y. So (x,y,z) -> (x, -z, y). Then add (0,0,-3).
    """
    rx_deg = rotation_deg[0]
    rx = np.deg2rad(rx_deg)
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ],
        dtype=np.float32,
    )
    out = (verts @ R.T) + np.array(translation, dtype=np.float32)
    return out


def detect_pin_indices(verts_world: np.ndarray):
    """Find vertices in the left/right pin boxes (world space after transformation).

    From cloth.json:
      pin 0 box: [0.99, -0.1, -4.1, 1.1, 0.1, -1.9]  -> right edge (x > 0.99)
      pin 1 box: [-1.1, -0.1, -4.1, -0.99, 0.1, -1.9] -> left  edge (x < -0.99)
    """
    x = verts_world[:, 0]
    y = verts_world[:, 1]
    z = verts_world[:, 2]
    in_y = (y > -0.1) & (y < 0.1)
    in_z = (z > -4.1) & (z < -1.9)
    right = np.where((x > 0.99) & (x < 1.1) & in_y & in_z)[0]
    left = np.where((x > -1.1) & (x < -0.99) & in_y & in_z)[0]
    return left.astype(np.int32), right.astype(np.int32)


def build_model(
    vertices: np.ndarray,
    indices: np.ndarray,
    left_pins: np.ndarray,
    right_pins: np.ndarray,
):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

    # Compute density from total area to match RealSim's obj_mass=1000
    e1 = vertices[indices[:, 1]] - vertices[indices[:, 0]]
    e2 = vertices[indices[:, 2]] - vertices[indices[:, 0]]
    tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total_area = float(tri_area.sum())
    density = MASS_TOTAL / total_area

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices],
        indices=indices.flatten().tolist(),
        density=density,
        tri_ke=2.0 * MU,  # PD stiffness = 2*mu for ARAP
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=0.0,  # no bending (cloth.json bending=0)
        edge_kd=0.0,
    )

    # Override mass to be exactly MASS_TOTAL / N (RealSim uniform mass convention)
    per_mass = MASS_TOTAL / len(vertices)
    for i in range(len(vertices)):
        builder.particle_mass[i] = per_mass

    # Pin the box-identified particles (mass=0 => pinned)
    for i in list(left_pins) + list(right_pins):
        builder.particle_mass[i] = 0.0

    return builder.finalize()


def render_frame(
    verts: np.ndarray,
    indices_flat: np.ndarray,
    frame_idx: int,
    model_name: str,
    out_dir: Path,
):
    """Render cloth mesh to PNG via matplotlib wireframe top-down view."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    tri = indices_flat.reshape(-1, 3)

    # Use plot_trisurf; for 10K vertex mesh use low linewidth to avoid clutter
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=tri,
        alpha=0.85,
        edgecolor="none",
        color="steelblue",
    )
    # Add wireframe overlay for a subset of edges (thin)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-4.5, -1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{model_name} — frame {frame_idx:04d}  (t={frame_idx * TIMESTEP:.2f}s)")
    ax.view_init(elev=20, azim=-55)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"frame_{frame_idx:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_one(
    model_name: str,
    stretching_model: str,
    mu: float,
    lam: float | None,
    vertices_init: np.ndarray,
    indices: np.ndarray,
    left_pins: np.ndarray,
    right_pins: np.ndarray,
):
    """Run extension simulation for one energy model, save snapshots."""
    print(f"\n=== {model_name} ===")
    t0 = time.time()

    model = build_model(vertices_init, indices, left_pins, right_pins)
    kwargs: dict = {"iterations": PD_ITERATIONS, "stretching_model": stretching_model}
    if stretching_model != "arap":
        kwargs["mu"] = mu
        kwargs["lam"] = lam
    solver = SolverFBA(model, **kwargs)

    s_in, s_out = model.state(), model.state()
    x_ref = vertices_init.copy()  # (N, 3) float32 reference positions

    out_dir = OUTPUT_ROOT / model_name.lower()

    # Snapshot frame 0 (initial state)
    if 0 in SNAPSHOT_FRAMES:
        render_frame(vertices_init, indices.flatten(), 0, model_name, out_dir)
        print(f"  frame 0: initial cloth x ∈ [{vertices_init[:,0].min():.3f}, {vertices_init[:,0].max():.3f}]")

    trajectory_snaps: dict = {}

    nan_detected = False
    for f in range(NUM_FRAMES):
        # Update pin targets: pull edges outward until max displacement
        displacement = min(PULL_VEL * TIMESTEP * (f + 1), PULL_MAX_DISP)
        x_ref[left_pins, 0] = vertices_init[left_pins, 0] - displacement
        x_ref[right_pins, 0] = vertices_init[right_pins, 0] + displacement
        solver.set_pin_targets(x_ref)

        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, TIMESTEP)
        s_in, s_out = s_out, s_in

        frame_num = f + 1
        if frame_num in SNAPSHOT_FRAMES:
            q = s_in.particle_q.numpy()
            if not np.all(np.isfinite(q)):
                print(f"  WARNING: NaN/Inf detected at frame {frame_num}! Stopping early.")
                nan_detected = True
                break
            trajectory_snaps[frame_num] = q.copy()
            render_frame(q, indices.flatten(), frame_num, model_name, out_dir)
            print(
                f"  frame {frame_num:3d}: pin_disp={displacement:.3f}m  "
                f"x ∈ [{q[:,0].min():.3f}, {q[:,0].max():.3f}]"
            )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(trajectory_snaps)} snapshots saved)")

    if trajectory_snaps:
        np.savez(
            out_dir / "trajectory.npz",
            **{f"frame_{f}": q for f, q in trajectory_snaps.items()},
        )

    return trajectory_snaps, nan_detected


def build_summary(results: dict, indices: np.ndarray):
    """Build a side-by-side summary PNG for all snapshot frames x all models."""
    models = list(results.keys())
    frames = SNAPSHOT_FRAMES[1:]  # skip frame 0 (initial)
    # Also add frame 0 from vertices_init if we have it
    # Filter to only frames that appear in all results
    all_frames = sorted({f for snaps, _nan in results.values() for f in snaps})
    if not all_frames:
        print("No frames to summarize.")
        return

    ncols = len(models)
    nrows = len(all_frames)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), subplot_kw={"projection": "3d"}
    )
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    tri = indices.reshape(-1, 3)

    for row_i, frame_num in enumerate(all_frames):
        for col_j, model_name in enumerate(models):
            ax = axes[row_i][col_j]
            snaps, nan_detected = results[model_name]
            if frame_num in snaps:
                q = snaps[frame_num]
                ax.plot_trisurf(
                    q[:, 0],
                    q[:, 1],
                    q[:, 2],
                    triangles=tri,
                    alpha=0.85,
                    edgecolor="none",
                    color="steelblue",
                )
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(-4.5, -1.5)
            ax.set_xlabel("X", fontsize=6)
            ax.set_ylabel("Y", fontsize=6)
            ax.set_zlabel("Z", fontsize=6)
            t = frame_num * TIMESTEP
            ax.set_title(f"{model_name}\nf={frame_num} t={t:.1f}s", fontsize=8)
            ax.view_init(elev=20, azim=-55)
            ax.tick_params(labelsize=5)

    plt.suptitle("SolverFBA Cloth Extension — ARAP / Corotational / Neo-Hookean", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = OUTPUT_ROOT / "summary.png"
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary saved to: {out_path}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    wp.init()

    print("Loading mesh...")
    verts_local, indices = load_obj(REALSIM_MESH)
    print(f"  Mesh: {len(verts_local)} vertices, {len(indices)} triangles")

    verts_world = transform_vertices(verts_local)
    left_pins, right_pins = detect_pin_indices(verts_world)
    print(f"  Left pins:  {len(left_pins)} vertices")
    print(f"  Right pins: {len(right_pins)} vertices")

    if len(left_pins) == 0 or len(right_pins) == 0:
        print("ERROR: No pin vertices detected! Check transformation / box coordinates.")
        return

    results: dict = {}

    # ARAP
    snaps, nan = run_one(
        "ARAP",
        "arap",
        MU,
        None,
        verts_world,
        indices,
        left_pins,
        right_pins,
    )
    results["ARAP"] = (snaps, nan)

    # Corotational (moderate lam for numerical stability)
    snaps, nan = run_one(
        "Corot",
        "corotational",
        MU,
        MU * 1.2,
        verts_world,
        indices,
        left_pins,
        right_pins,
    )
    results["Corot"] = (snaps, nan)

    # Neo-Hookean (true Lame pair)
    snaps, nan = run_one(
        "NeoHookean",
        "neohookean",
        MU,
        LAM_NH,
        verts_world,
        indices,
        left_pins,
        right_pins,
    )
    results["NeoHookean"] = (snaps, nan)

    print("\n=== Building summary montage ===")
    build_summary(results, indices.flatten())

    print("\nAll done.")
    print(f"Output directory: {OUTPUT_ROOT.resolve()}")
    print(f"Frame PNGs per model in subdirectories: {list(OUTPUT_ROOT.iterdir())}")


if __name__ == "__main__":
    main()
