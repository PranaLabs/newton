# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 smoke test for SolverFBA multi-env support.

Drives two replicated copies of the hanging-cloth scene through SolverFBA
(no contacts, just PD + bending + gravity + pin), and verifies that the
two envs evolve identically modulo a fixed translation.  See
``docs/superpowers/plans/2026-05-18-fba-multienv-cloth.md`` Phase 0.

Usage::

    uv run python scripts/fba_multienv_hanging_cloth_smoke.py \
        --num-frames 100 --spacing 5.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import newton
from newton.examples.cloth.example_hanging_cloth_fba import (
    _DT,
    _OBJ_MASS,
    _PD_ITER,
    build_hanging_cloth_builder,
)
from newton.solvers import SolverFBA


def _run(world_count: int, spacing: tuple[float, float, float], num_frames: int):
    sub, pin_idx_local = build_hanging_cloth_builder()
    n_per_env = sub.particle_count

    if world_count == 1:
        main = sub
    else:
        main = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=sub.gravity)
        main.replicate(sub, world_count=world_count, spacing=spacing)

    model = main.finalize()

    uniform_mass = _OBJ_MASS / n_per_env
    mass_arr = np.full(model.particle_count, uniform_mass, dtype=np.float32)
    inv_arr = np.full(model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
    for w in range(world_count):
        base = w * n_per_env
        for li in pin_idx_local:
            mass_arr[base + li] = 0.0
            inv_arr[base + li] = 0.0
    model.particle_mass.assign(mass_arr)
    model.particle_inv_mass.assign(inv_arr)

    solver = SolverFBA(model, iterations=_PD_ITER)
    state_0 = model.state()
    state_1 = model.state()

    t0 = time.perf_counter()
    for _f in range(num_frames):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, _DT)
        state_0, state_1 = state_1, state_0
    elapsed = time.perf_counter() - t0

    q = state_0.particle_q.numpy()
    return q, elapsed, n_per_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--spacing", type=float, default=5.0,
                        help="X-axis spacing between worlds (m).")
    parser.add_argument("--also-single", action="store_true", default=True,
                        help="Also time a single-env reference run (default on).")
    parser.add_argument("--rel-tol", type=float, default=1e-5,
                        help="Tier 2 relative-norm threshold (env-to-env diff).")
    parser.add_argument("--out", type=Path,
                        default=Path("scripts/realsim_baseline/fba_2env_hanging_cloth_q.npz"))
    args = parser.parse_args()

    spacing = (args.spacing, 0.0, 0.0)

    print(f"[Phase 0] 2-env hanging-cloth smoke @ {args.num_frames} frames, spacing={spacing}")
    q2, t2, n_per_env = _run(world_count=2, spacing=spacing, num_frames=args.num_frames)
    print(f"  2-env: {t2:.2f}s ({args.num_frames / t2:.1f} steps/s)")

    t1 = None
    if args.also_single:
        _q1, t1, _n = _run(world_count=1, spacing=(0.0, 0.0, 0.0),
                           num_frames=args.num_frames)
        print(f"  1-env: {t1:.2f}s ({args.num_frames / t1:.1f} steps/s)")

    # Tier 1: finite + bounded.
    finite = np.all(np.isfinite(q2))
    max_abs = float(np.max(np.abs(q2)))
    print(f"\n[Tier 1] finite={finite}, max|q|={max_abs:.3f} m")
    if not finite or max_abs > 100.0:
        print("  FAIL")
        return 1
    print("  PASS")

    # Tier 2: env-to-env identity modulo spacing offset.
    q2 = q2.reshape(2, n_per_env, 3)
    offset = np.array(spacing, dtype=q2.dtype)
    diff = q2[1] - (q2[0] + offset)
    abs_diff = np.linalg.norm(diff)
    ref_norm = np.linalg.norm(q2[0])
    rel = abs_diff / max(ref_norm, 1e-12)
    print(f"\n[Tier 2] env1 - (env0 + offset): |diff|={abs_diff:.3e}, rel={rel:.3e}")
    print(f"  per-particle max|diff|={float(np.max(np.linalg.norm(diff, axis=1))):.3e} m")
    tier2_pass = rel < args.rel_tol
    print(f"  threshold={args.rel_tol:.0e} -> {'PASS' if tier2_pass else 'FAIL'}")

    # Tier 3: step-time scaling (informational).
    if t1 is not None:
        ratio = t2 / t1
        print(f"\n[Tier 3] step_time(2env) / step_time(1env) = {ratio:.2f} (expect 1.5-3.0)")

    # Save reference frame for downstream inspection.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, q2=q2, diff_per_particle=np.linalg.norm(diff, axis=1))
    print(f"\nSaved {args.out}")

    return 0 if tier2_pass else 2


if __name__ == "__main__":
    sys.exit(main())
