# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Multi-env cloth-on-sphere benchmark for SolverFBA (Phase 3.1).

Replicates the single-env cloth-on-sphere example to N worlds using
``builder.replicate(spacing=(10,10,10))`` and times per-step cost in both
``nsn_schur_mode={full, lite}``.  Optionally dumps per-frame positions for
the cross-env consistency gate (Phase 3.3).

Usage::

    # Single-env smoke / regression
    uv run python scripts/fba_multienv_cloth_bench.py --n-envs 1 --mode lite

    # Phase 3.2 sweep
    for N in 1 4 9 16 25; do
        for M in full lite; do
            uv run python scripts/fba_multienv_cloth_bench.py \
                --n-envs $N --mode $M --num-frames 300 \
                --out scripts/multienv_results/n${N}_${M}.npz
        done
    done
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.cloth.example_cloth_on_sphere_fba import (
    _DT,
    _FRICTION_MU,
    _NSN_ITER,
    _OBJ_MASS,
    _PD_ITER,
    _PIN_STIFFNESS,
    _YOUNG,
    _POISSON,
    _lame_from_young_poisson,
    build_cloth_on_sphere_builder,
)
from newton.solvers import SolverFBA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--mode", choices=["full", "lite"], required=True)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--spacing", type=float, default=10.0,
                        help="Spacing per axis in meters (cubic offset).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Dump per-frame positions to this .npz.")
    parser.add_argument("--first-last-only", action="store_true",
                        help="Only retain frame 0, 200, 299 to save memory at "
                             "large N (Phase 3.3 cross-env gate frames).")
    args = parser.parse_args()

    wp.init()

    sub = build_cloth_on_sphere_builder()
    n_particles_per_env = sub.particle_count
    print(f"[Phase 3] N={args.n_envs} mode={args.mode} frames={args.num_frames}")
    print(f"  particles/env={n_particles_per_env}")

    if args.n_envs == 1:
        main_b = sub
    else:
        main_b = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=sub.gravity)
        main_b.replicate(sub, world_count=args.n_envs,
                         spacing=(args.spacing, args.spacing, args.spacing))

    model = main_b.finalize()
    print(f"  model: particle_count={model.particle_count} "
          f"shape_count={model.shape_count} world_count={model.world_count}")

    uniform_mass = _OBJ_MASS / n_particles_per_env
    model.particle_mass.assign(
        np.full(model.particle_count, uniform_mass, dtype=np.float32)
    )
    model.particle_inv_mass.assign(
        np.full(model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
    )

    mu_override = np.full(model.particle_count, _FRICTION_MU, dtype=np.float64)
    _mu, _lam = _lame_from_young_poisson(_YOUNG, _POISSON)

    try:
        solver = SolverFBA(
            model,
            iterations=_PD_ITER,
            nsn_iterations=_NSN_ITER,
            friction=True,
            stretching_model="arap",
            mu_per_pair_override=mu_override,
            pin_stiffness=_PIN_STIFFNESS,
            nsn_schur_mode=args.mode,
        )
    except Exception as exc:
        print(f"  SolverFBA construction failed: {exc}")
        traceback.print_exc()
        return 2

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()

    if args.first_last_only:
        kept = {0, args.num_frames // 2, args.num_frames - 1, 200, 299}
        positions = {f: np.empty((model.particle_count, 3), dtype=np.float32)
                     for f in kept if f < args.num_frames}
    else:
        positions = np.empty((args.num_frames, model.particle_count, 3), dtype=np.float32)
    step_ms = np.empty(args.num_frames, dtype=np.float32)
    n_contacts_per_frame = np.empty(args.num_frames, dtype=np.int32)

    try:
        for f in range(args.num_frames):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            wp.synchronize_device()
            t0 = time.perf_counter()
            solver.step(state_0, state_1, None, contacts, _DT)
            wp.synchronize_device()
            step_ms[f] = (time.perf_counter() - t0) * 1000.0
            state_0, state_1 = state_1, state_0
            if args.first_last_only:
                if f in positions:
                    positions[f] = state_0.particle_q.numpy()
            else:
                positions[f] = state_0.particle_q.numpy()
            nc = 0
            if hasattr(contacts, "soft_contact_count") and contacts.soft_contact_count is not None:
                nc = int(contacts.soft_contact_count.numpy()[0])
            n_contacts_per_frame[f] = nc
            if f % 50 == 0 or f == args.num_frames - 1:
                print(f"  frame {f}: step_ms={step_ms[f]:.2f}  n_contacts={nc}")
    except Exception as exc:
        print(f"  step loop crashed at frame {f}: {exc}")
        traceback.print_exc()
        return 3

    if args.num_frames > 100:
        settled = step_ms[100:]
        print(f"  step_mean(all)={step_ms.mean():.2f} ms  "
              f"step_mean(settled)={settled.mean():.2f} ms  "
              f"p95={np.percentile(settled, 95):.2f} ms")
    else:
        print(f"  step_mean(all)={step_ms.mean():.2f} ms")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = dict(
            step_ms=step_ms,
            n_contacts=n_contacts_per_frame,
            n_envs=args.n_envs,
            mode=args.mode,
            particles_per_env=n_particles_per_env,
        )
        if args.first_last_only:
            # Stash per-kept-frame positions as separate keys.
            for f, p in positions.items():
                save_kwargs[f"positions_f{f}"] = p
        else:
            save_kwargs["positions"] = positions
        np.savez(args.out, **save_kwargs)
        print(f"  saved {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
