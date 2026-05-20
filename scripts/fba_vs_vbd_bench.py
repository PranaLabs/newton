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
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
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


# --- Smoke check (Task 1 only; replaced by real CLI in Task 8) ---


def _smoke():
    np.random.seed(0)
    m_fba = build_model_fba()
    m_vbd = build_model_vbd(alpha=1.0, beta=1.0)
    assert m_fba.particle_count == m_vbd.particle_count == (GRID_DIM + 1) ** 2, (
        f"particle count mismatch: fba={m_fba.particle_count}, vbd={m_vbd.particle_count}"
    )
    assert m_fba.tri_count == m_vbd.tri_count == 2 * GRID_DIM * GRID_DIM, (
        f"tri count mismatch: fba={m_fba.tri_count}, vbd={m_vbd.tri_count}"
    )
    # VBD must have coloring populated.
    assert len(m_vbd.particle_color_groups) > 0, "VBD model is missing particle_color_groups"
    assert len(m_fba.particle_color_groups) == 0, "FBA model should not be colored"
    print(
        f"OK: particle_count={m_fba.particle_count}, tri_count={m_fba.tri_count}, "
        f"vbd_colors={len(m_vbd.particle_color_groups)}"
    )


if __name__ == "__main__":
    _smoke()
