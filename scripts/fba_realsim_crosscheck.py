# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Newton-side harness for the FBA × RealSim trajectory cross-check.

Loads the same scene used by RealSim's FBACrossCheck (square_113P.obj cloth,
two pinned corners, gravity [0,0,-10], dt=0.01, 5 PD iterations, 50 frames),
runs Newton SolverFBA, and saves both Newton and RealSim trajectories as .npy
files for bit-level comparison.

Usage::

    uv run python scripts/fba_realsim_crosscheck.py

Outputs:
    scripts/fba_newton_trajectory.npy   — shape (50, 113, 3), float32
    scripts/fba_realsim_trajectory.npy  — shape (50, 113, 3), float32
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
OBJ_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/cloth/square_113P.obj")
ABC_PATH = Path(
    "/home/ziqiu/work/RealSim_py/realsim_py/simulation/output_abc/FBACrossCheck/output_obj_0.abc"
)
DUMP_ABC = Path("/tmp/dump_abc_traj")
NEWTON_TRAJ_PATH = SCRIPT_DIR / "fba_newton_trajectory.npy"
REALSIM_TRAJ_PATH = SCRIPT_DIR / "fba_realsim_trajectory.npy"

# ---------------------------------------------------------------------------
# Simulation parameters (must match RealSim FBACrossCheck)
# ---------------------------------------------------------------------------
DT = 0.01
N_FRAMES = 50
N_ITER = 5
GRAVITY = np.array([0.0, 0.0, -10.0], dtype=np.float32)

# Young=1e4, Poisson=0.4 → μ = E / (2(1+ν)) = 1e4 / 2.8 ≈ 3571.43
# RealSim ARAP weight = 2μ per area; Newton tri_ke = 2μ = 7142.86
# But following the task spec: tri_ke = 3571 (= 2μ as PD weight, area-scaled later)
TRI_KE = 3571.0
EDGE_KE = 0.1

# Pin boxes from cloth.json (xmin, ymin, zmin, xmax, ymax, zmax)
PIN_BOXES = [
    (-1.1, 0.9, -0.1, -0.9, 1.1, 0.1),
    (0.9, 0.9, -0.1, 1.1, 1.1, 0.1),
]


# ---------------------------------------------------------------------------
# 1. Load OBJ mesh
# ---------------------------------------------------------------------------
def load_obj(path: Path) -> tuple[list[wp.vec3], list[int]]:
    """Parse .obj file, return (vertices, triangle_indices_0based)."""
    vertices: list[wp.vec3] = []
    indices: list[int] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append(wp.vec3(x, y, z))
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Each part may be "v", "v/vt", or "v/vt/vn" — take first int.
                face_indices = [int(p.split("/")[0]) - 1 for p in parts]
                # Triangulate fans if needed (all faces should already be triangles).
                for i in range(1, len(face_indices) - 1):
                    indices.extend([face_indices[0], face_indices[i], face_indices[i + 1]])
    return vertices, indices


# ---------------------------------------------------------------------------
# 2. Compute total area for density
# ---------------------------------------------------------------------------
def compute_total_area(vertices: list[wp.vec3], indices: list[int]) -> float:
    verts_np = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float64)
    idxs = np.array(indices, dtype=np.int32).reshape(-1, 3)
    total = 0.0
    for tri in idxs:
        e1 = verts_np[tri[1]] - verts_np[tri[0]]
        e2 = verts_np[tri[2]] - verts_np[tri[0]]
        cross = np.cross(e1, e2)
        total += 0.5 * np.linalg.norm(cross)
    return total


# ---------------------------------------------------------------------------
# 3. Find pin indices from bounding boxes
# ---------------------------------------------------------------------------
def find_pin_indices(vertices: list[wp.vec3], boxes: list[tuple]) -> list[int]:
    pinned: list[int] = []
    for i, v in enumerate(vertices):
        x, y, z = v[0], v[1], v[2]
        for xmin, ymin, zmin, xmax, ymax, zmax in boxes:
            if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                pinned.append(i)
                break
    return pinned


# ---------------------------------------------------------------------------
# 4. Build Newton model
# ---------------------------------------------------------------------------
def build_model(vertices: list[wp.vec3], indices: list[int]) -> newton.Model:
    total_area = compute_total_area(vertices, indices)
    # density such that sum of (density * area/3) over all triangle-vertex
    # contributions = 1.0 kg total mass for free particles
    density = 1.0 / total_area
    print(f"  total_area = {total_area:.6f} m², density = {density:.6f} kg/m²")

    # Z-up world (mesh is in XY plane, gravity is [0,0,-10])
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-10.0)

    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices,
        indices=indices,
        density=density,
        tri_ke=TRI_KE,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=EDGE_KE,
        edge_kd=0.0,
    )

    # Pin the two top corners.
    pin_indices = find_pin_indices(vertices, PIN_BOXES)
    print(f"  Pinned vertex indices: {pin_indices}")
    for idx in pin_indices:
        v = vertices[idx]
        print(f"    index {idx}: pos = ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")
        builder.particle_mass[idx] = 0.0

    model = builder.finalize()
    print(f"  particle_count = {model.particle_count}, tri_count = {model.tri_count}")

    # Verify/override gravity (builder sets it from up_axis + gravity scalar)
    # model.gravity is a wp.array[wp.vec3] of shape [world_count].
    # The builder sets it from up_axis=Z and gravity=-10.0 → [0,0,-10], so
    # no override needed. But we verify:
    g = model.gravity.numpy()
    print(f"  model.gravity = {g}")

    return model


# ---------------------------------------------------------------------------
# 5. Run Newton SolverFBA for N_FRAMES steps
# ---------------------------------------------------------------------------
def run_newton(model: newton.Model) -> np.ndarray:
    solver = SolverFBA(model, iterations=N_ITER)
    state_in = model.state()
    state_out = model.state()

    N = model.particle_count
    trajectory = np.zeros((N_FRAMES, N, 3), dtype=np.float32)

    for f in range(N_FRAMES):
        state_in.clear_forces()
        solver.step(state_in, state_out, None, None, DT)
        trajectory[f] = state_out.particle_q.numpy()
        state_in, state_out = state_out, state_in

    return trajectory


# ---------------------------------------------------------------------------
# 6. Read RealSim trajectory from Alembic via dump_abc_traj binary
# ---------------------------------------------------------------------------
def read_realsim_trajectory() -> np.ndarray:
    if not DUMP_ABC.exists():
        # Compile if missing (should not happen in normal CI)
        raise FileNotFoundError(
            f"dump_abc_traj binary not found at {DUMP_ABC}. "
            "Please compile it with the instructions in the spec."
        )

    result = subprocess.run(
        [str(DUMP_ABC), str(ABC_PATH)],
        capture_output=True,
        text=True,
        check=True,
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
    traj = np.array(frames, dtype=np.float32)  # (50, 113, 3)
    return traj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    wp.init()

    print("=== FBA × RealSim Cross-Check ===\n")

    # --- Load mesh ---
    print(f"Loading mesh from {OBJ_PATH} ...")
    vertices, indices = load_obj(OBJ_PATH)
    print(f"  {len(vertices)} vertices, {len(indices) // 3} triangles")

    # Sanity-check first few vertex positions (must match frame-0 RealSim abc)
    print("  First 4 vertices (0-based):")
    for i in range(4):
        v = vertices[i]
        print(f"    [{i}] = ({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})")

    # --- Build model ---
    print("\nBuilding Newton model ...")
    model = build_model(vertices, indices)

    # Capture initial positions (frame 0 = before any step)
    initial_q = model.particle_q.numpy().copy()
    print(f"\n  Initial positions of pin candidates:")
    for idx in find_pin_indices(vertices, PIN_BOXES):
        p = initial_q[idx]
        print(f"    index {idx}: ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})")

    # --- Run Newton ---
    print(f"\nRunning Newton SolverFBA for {N_FRAMES} frames (dt={DT}, iter={N_ITER}) ...")
    newton_traj = run_newton(model)
    print(f"  Newton trajectory shape: {newton_traj.shape}")
    if not np.all(np.isfinite(newton_traj)):
        print("  WARNING: non-finite values in Newton trajectory!")
    np.save(NEWTON_TRAJ_PATH, newton_traj)
    print(f"  Saved to {NEWTON_TRAJ_PATH}")

    # --- Read RealSim trajectory ---
    print(f"\nReading RealSim trajectory from {ABC_PATH} ...")
    realsim_traj = read_realsim_trajectory()
    print(f"  RealSim trajectory shape: {realsim_traj.shape}")
    np.save(REALSIM_TRAJ_PATH, realsim_traj)
    print(f"  Saved to {REALSIM_TRAJ_PATH}")

    # --- Summary ---
    print(f"\nNewton trajectory:    {NEWTON_TRAJ_PATH}   shape={newton_traj.shape}")
    print(f"RealSim trajectory:   {REALSIM_TRAJ_PATH}  shape={realsim_traj.shape}")

    print("\nPer-frame max L2 deltas (frames 0, 10, 20, 30, 40, 49):")
    for f in [0, 10, 20, 30, 40, 49]:
        delta = np.linalg.norm(newton_traj[f] - realsim_traj[f], axis=-1).max()
        print(f"  frame {f:2d}:  {delta:.3e}")

    print("\nPer-frame max L2 deltas (all frames):")
    for f in range(N_FRAMES):
        delta = np.linalg.norm(newton_traj[f] - realsim_traj[f], axis=-1).max()
        print(f"  frame {f:2d}:  {delta:.3e}")

    # Spot-check pinned vertices stability in Newton (must not move)
    pin_indices = find_pin_indices(vertices, PIN_BOXES)
    print(f"\nPinned vertex positions at frame 49 (Newton):")
    for idx in pin_indices:
        p = newton_traj[49, idx]
        print(f"  [{idx}] = ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})")


if __name__ == "__main__":
    main()
