# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""FBA vs VBD calibrated benchmark on a 32x32 hanging cloth.

See docs/superpowers/specs/2026-05-20-fba-vs-vbd-bench-design.md for the
design rationale. Run::

    uv run python scripts/fba_vs_vbd_bench.py --out-dir scripts/fba_vs_vbd_bench_out
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA, SolverVBD

# Anchor parameters (match the spec, do not change without updating the doc).
GRID_DIM = 32
CELL = 0.05
MASS = 0.1
GRAVITY = -9.81  # m/s²; scalar passed to ModelBuilder (up_axis=Y expands to (0, -9.81, 0))
FBA_MU = 1000.0
FBA_LAM = 1000.0
FBA_EDGE_KE = 0.1
FRAME_DT = 1.0 / 100.0
ANCHOR_ITERATIONS = 200

# Phase 1 grids.
ALPHA_GRID_1D = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0)
BETA_GRID_2D = (0.5, 0.75, 1.0, 1.5, 2.0)

# Phase 2 sweep grids.
FBA_SWEEP_ITERS = (5, 10, 20, 40)
VBD_SWEEP_CONFIGS = (
    (2, 10),
    (5, 10),
    (10, 10),
    (10, 20),
)

# --- Builder & model construction ---


def build_builder(tri_ke: float, tri_ka: float, edge_ke: float) -> newton.ModelBuilder:
    """Construct the 32x32 hanging cloth builder with corner-pin masses set.

    `tri_ke` and `tri_ka` map directly to the Stable Neo-Hookean (mu, lambda)
    that VBD reads from `tri_materials`; FBA reads `(mu, lam)` from its own
    constructor; the builder's `tri_ke`/`tri_ka` are consumed only by VBD.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=GRAVITY)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 2.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=GRID_DIM,
        dim_y=GRID_DIM,
        cell_x=CELL,
        cell_y=CELL,
        mass=MASS,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=0.0,
        edge_ke=edge_ke,
        edge_kd=0.0,
        fix_left=False,
    )
    top_left = GRID_DIM * (GRID_DIM + 1)
    top_right = top_left + GRID_DIM
    builder.particle_mass[top_left] = 0.0
    builder.particle_mass[top_right] = 0.0
    return builder


def build_model_fba(mu: float = FBA_MU, lam: float = FBA_LAM, edge_ke: float = FBA_EDGE_KE):
    """Finalize a model intended for SolverFBA. No coloring needed."""
    builder = build_builder(tri_ke=mu, tri_ka=lam, edge_ke=edge_ke)
    return builder.finalize()


def build_model_vbd(alpha: float = 1.0, beta: float = 1.0):
    """Finalize a model for SolverVBD with calibration scales applied.

    `alpha` scales `tri_ke` (the VBD Stable-NH `mu`); `beta` scales `edge_ke`.
    """
    builder = build_builder(
        tri_ke=alpha * FBA_MU,
        tri_ka=FBA_LAM,
        edge_ke=beta * FBA_EDGE_KE,
    )
    builder.color(include_bending=True)
    return builder.finalize()


# --- Timing & trajectory recording ---


@dataclass
class RunResult:
    """Per-frame wall-clock and full trajectory from one solver run."""

    trajectory: np.ndarray  # (n_frames, n_particles, 3), float32
    wall_clock_ms: np.ndarray  # (n_frames,), float64
    fba_timing_summary: dict | None = None


def _sync():
    if wp.is_cuda_available():
        wp.synchronize_device()


def run_solver(
    model,
    solver,
    n_frames: int,
    substeps: int = 1,
    n_warmup: int = 10,
    frame_dt: float = FRAME_DT,
) -> RunResult:
    """Step `solver` for `n_frames` rendered frames; record trajectory + wall-clock.

    For VBD-style multi-substep configs, pass `substeps > 1`; internally we
    call `solver.step(...)` `substeps` times per rendered frame with
    `dt = frame_dt / substeps`. Wall-clock per rendered frame includes all
    inner step calls.

    `n_warmup` frames are still recorded into the trajectory (we want them
    for completeness) but their wall-clock measurements are discarded by the
    caller — `run_solver` returns the full `wall_clock_ms` array; callers
    slice off `[n_warmup:]` when computing mean/stddev.
    """
    state_in = model.state()
    state_out = model.state()
    n_particles = model.particle_count
    trajectory = np.empty((n_frames, n_particles, 3), dtype=np.float32)
    wall = np.empty(n_frames, dtype=np.float64)
    sub_dt = frame_dt / substeps

    for f in range(n_frames):
        _sync()
        t0 = time.perf_counter()
        for _ in range(substeps):
            state_in.clear_forces()
            solver.step(state_in, state_out, None, None, sub_dt)
            state_in, state_out = state_out, state_in
        _sync()
        wall[f] = (time.perf_counter() - t0) * 1000.0
        trajectory[f] = state_in.particle_q.numpy()
        if not np.all(np.isfinite(trajectory[f])):
            raise RuntimeError(f"non-finite positions at frame {f}; solver diverged")

    fba_summary = None
    if hasattr(solver, "get_timing_summary"):
        try:
            fba_summary = solver.get_timing_summary()
        except Exception as exc:
            import warnings  # noqa: PLC0415
            warnings.warn(f"get_timing_summary raised {exc!r}; fba_timing_summary will be None", stacklevel=2)
            fba_summary = None

    return RunResult(trajectory=trajectory, wall_clock_ms=wall, fba_timing_summary=fba_summary)


# --- Phase 0: FBA anchor ---


def phase0_fba_anchor(out_dir: Path, n_frames: int = 800, n_warmup: int = 10) -> dict:
    """Run FBA at iter=200 and store the full trajectory as ground truth.

    Returns a metadata dict describing the run and the on-disk paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "x_FBA_ref.npy"

    model = build_model_fba(mu=FBA_MU, lam=FBA_LAM, edge_ke=FBA_EDGE_KE)
    solver = SolverFBA(
        model,
        iterations=ANCHOR_ITERATIONS,
        stretching_model="neohookean",
        mu=FBA_MU,
        lam=FBA_LAM,
        enable_perf_timing=True,
    )

    print(f"[phase 0] running FBA anchor iter={ANCHOR_ITERATIONS} for {n_frames} frames...")
    t0 = time.perf_counter()
    res = run_solver(model, solver, n_frames=n_frames, substeps=1, n_warmup=n_warmup)
    elapsed = time.perf_counter() - t0
    print(f"[phase 0] done in {elapsed:.1f} s; mean ms/frame (excl. warmup) "
          f"= {res.wall_clock_ms[n_warmup:].mean():.2f}")

    np.save(ref_path, res.trajectory)

    sag = float(res.trajectory[-1, :, 1].min())
    initial_y = float(res.trajectory[0, :, 1].min())
    meta = {
        "n_frames": n_frames,
        "n_warmup": n_warmup,
        "iterations": ANCHOR_ITERATIONS,
        "mu": FBA_MU,
        "lam": FBA_LAM,
        "edge_ke": FBA_EDGE_KE,
        "frame_dt": FRAME_DT,
        "elapsed_s": elapsed,
        "mean_ms_per_frame": float(res.wall_clock_ms[n_warmup:].mean()),
        "terminal_min_y": sag,
        "initial_min_y": initial_y,
        "max_sag": initial_y - sag,
        "fba_timing_summary": res.fba_timing_summary,
        "ref_path": str(ref_path),
    }
    return meta


# --- Smoke check (Task 1 only; replaced by real CLI in Task 8) ---


def _smoke():
    np.random.seed(0)
    m_fba = build_model_fba()
    m_vbd = build_model_vbd(alpha=1.0, beta=1.0)
    assert m_fba.particle_count == m_vbd.particle_count == (GRID_DIM + 1) ** 2
    assert m_fba.tri_count == m_vbd.tri_count == 2 * GRID_DIM * GRID_DIM
    assert len(m_vbd.particle_color_groups) > 0

    out = Path("/tmp/fba_vs_vbd_smoke_out")
    meta = phase0_fba_anchor(out, n_frames=30, n_warmup=5)
    ref = np.load(out / "x_FBA_ref.npy")
    assert ref.shape == (30, 1089, 3), f"ref shape {ref.shape}"
    assert meta["max_sag"] > 0.0, "cloth did not descend during anchor"
    print(f"OK: max_sag={meta['max_sag']:.4f} m, mean_ms/frame={meta['mean_ms_per_frame']:.2f}")


if __name__ == "__main__":
    _smoke()
