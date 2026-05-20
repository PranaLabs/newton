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


# --- Smoke check (Task 1 only; replaced by real CLI in Task 8) ---


def _smoke():
    np.random.seed(0)
    m_fba = build_model_fba()
    m_vbd = build_model_vbd(alpha=1.0, beta=1.0)
    assert m_fba.particle_count == m_vbd.particle_count == (GRID_DIM + 1) ** 2
    assert m_fba.tri_count == m_vbd.tri_count == 2 * GRID_DIM * GRID_DIM
    assert len(m_vbd.particle_color_groups) > 0

    solver_fba = SolverFBA(m_fba, iterations=5, stretching_model="neohookean", mu=FBA_MU, lam=FBA_LAM)
    res_fba = run_solver(m_fba, solver_fba, n_frames=10, substeps=1, n_warmup=2)
    assert res_fba.trajectory.shape == (10, 1089, 3)
    assert np.all(np.isfinite(res_fba.trajectory))
    assert res_fba.trajectory[-1, :, 1].min() < res_fba.trajectory[0, :, 1].min(), (
        "cloth did not descend; gravity/pin setup is wrong"
    )

    solver_vbd = SolverVBD(m_vbd, iterations=5, particle_enable_self_contact=False)
    res_vbd = run_solver(m_vbd, solver_vbd, n_frames=10, substeps=2, n_warmup=2)
    assert res_vbd.trajectory.shape == (10, 1089, 3)
    assert np.all(np.isfinite(res_vbd.trajectory))

    print(
        f"OK: fba mean ms/frame={res_fba.wall_clock_ms[2:].mean():.3f}, "
        f"vbd mean ms/frame={res_vbd.wall_clock_ms[2:].mean():.3f}"
    )


if __name__ == "__main__":
    _smoke()
