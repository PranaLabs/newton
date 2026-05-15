# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: SolverFBA ARAP vs Corotational step time on a 32x32 cloth.

Usage:
    uv run scripts/fba_arap_vs_corot_bench.py

Reports mean/median per-step time (ms) for each model, plus Corot/ARAP ratio.
"""

import time

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA

WARMUP_STEPS = 10
BENCH_STEPS = 100
DIM = 32
# Lamé parameters matching RealSim default (E=1e4, nu=0.4).
MU = 3571.0
LAM = 4286.0


def build_cloth(dim: int = DIM) -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    builder.add_cloth_grid(
        pos=wp.vec3(0, 2, 0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0, 0, 0),
        dim_x=dim,
        dim_y=dim,
        cell_x=0.05,
        cell_y=0.05,
        mass=0.1,
        tri_ke=2.0 * MU,  # tri_ke = 2μ for fair comparison
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=0.1,
        edge_kd=0.0,
        fix_left=False,
    )
    top_left = dim * (dim + 1)
    top_right = dim * (dim + 1) + dim
    builder.particle_mass[top_left] = 0.0
    builder.particle_mass[top_right] = 0.0
    return builder.finalize()


def run_bench(model: newton.Model, solver: SolverFBA, label: str) -> np.ndarray:
    """Run warmup + bench steps; return per-step wall times in seconds."""
    s_in, s_out = model.state(), model.state()
    dt = 1.0 / 60.0

    print(f"  [{label}] Warming up ({WARMUP_STEPS} steps)...", flush=True)
    for _ in range(WARMUP_STEPS):
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, dt)
        s_in, s_out = s_out, s_in
    # Sync after warmup.
    if wp.is_cuda_available():
        wp.synchronize()

    print(f"  [{label}] Benchmarking ({BENCH_STEPS} steps)...", flush=True)
    times = []
    for _ in range(BENCH_STEPS):
        s_in.clear_forces()
        if wp.is_cuda_available():
            wp.synchronize()
        t0 = time.perf_counter()
        solver.step(s_in, s_out, None, None, dt)
        if wp.is_cuda_available():
            wp.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        s_in, s_out = s_out, s_in

    times_arr = np.array(times)
    q = s_in.particle_q.numpy()
    assert np.all(np.isfinite(q)), f"[{label}] NaN detected in output!"
    return times_arr


def main() -> None:
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}")
    print(f"Cloth: {DIM}x{DIM} grid ({(DIM + 1) * (DIM + 1)} particles, {2 * DIM * DIM} triangles)")
    print(f"Lame params: mu={MU}, lam={LAM}")
    print()

    # Build separate models so factorizations are independent.
    print("Building cloth models and factorizing PD systems...")
    model_arap = build_cloth()
    model_corot = build_cloth()

    solver_arap = SolverFBA(model_arap, iterations=10, stretching_model="arap")
    solver_corot = SolverFBA(model_corot, iterations=10, stretching_model="corotational", mu=MU, lam=LAM)

    # Trigger _setup_pd_system eagerly.
    print("Priming solvers (first step to trigger setup)...")
    dt = 1.0 / 60.0
    for solver, model in [(solver_arap, model_arap), (solver_corot, model_corot)]:
        s_in, s_out = model.state(), model.state()
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, dt)
    print()

    print("--- ARAP ---")
    times_arap = run_bench(model_arap, solver_arap, "ARAP")
    print()

    print("--- Corotational ---")
    times_corot = run_bench(model_corot, solver_corot, "Corot")
    print()

    # Convert to ms.
    ms_arap = times_arap * 1000.0
    ms_corot = times_corot * 1000.0

    def stats(arr: np.ndarray, label: str) -> None:
        print(
            f"  {label:12s}  mean={arr.mean():.3f} ms  median={np.median(arr):.3f} ms"
            f"  p95={np.percentile(arr, 95):.3f} ms  min={arr.min():.3f} ms  max={arr.max():.3f} ms"
        )

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    stats(ms_arap, "ARAP")
    stats(ms_corot, "Corotational")
    ratio = ms_corot.mean() / ms_arap.mean()
    overhead_pct = (ratio - 1.0) * 100.0
    print(f"\n  Corot / ARAP ratio: {ratio:.3f}x  ({overhead_pct:+.1f}% overhead)")
    print("=" * 70)


if __name__ == "__main__":
    main()
