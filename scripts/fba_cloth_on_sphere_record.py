# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Drive ``example_cloth_on_sphere_fba`` in pure-physics mode (no viewer)
and dump per-frame particle positions + contact counts.

Phase 1.3 / Phase 2.4 trajectory capture harness.
"""

from __future__ import annotations

import argparse
import time
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
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--schur-mode", choices=["full", "lite"], default="full")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--perf-timing", action="store_true",
                        help="Enable SolverFBA per-phase timing.")
    args = parser.parse_args()

    wp.init()

    builder = build_cloth_on_sphere_builder()
    model = builder.finalize()

    uniform_mass = _OBJ_MASS / model.particle_count
    model.particle_mass.assign(
        np.full(model.particle_count, uniform_mass, dtype=np.float32)
    )
    model.particle_inv_mass.assign(
        np.full(model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
    )

    mu_override = np.full(model.particle_count, _FRICTION_MU, dtype=np.float64)
    _mu, _lam = _lame_from_young_poisson(_YOUNG, _POISSON)

    solver_kwargs = dict(
        iterations=_PD_ITER,
        nsn_iterations=_NSN_ITER,
        friction=True,
        stretching_model="arap",
        mu_per_pair_override=mu_override,
        pin_stiffness=_PIN_STIFFNESS,
        enable_perf_timing=args.perf_timing,
    )
    try:
        solver = SolverFBA(model, **solver_kwargs, nsn_schur_mode=args.schur_mode)
    except TypeError:
        if args.schur_mode != "full":
            raise NotImplementedError(
                "nsn_schur_mode='lite' requires SolverFBA Phase 2.3 changes"
            ) from None
        solver = SolverFBA(model, **solver_kwargs)

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()

    positions = np.empty((args.num_frames, model.particle_count, 3), dtype=np.float32)
    n_contacts = np.empty(args.num_frames, dtype=np.int32)
    step_ms = np.empty(args.num_frames, dtype=np.float32)

    print(f"[Phase 1/2] cloth_on_sphere mode={args.schur_mode} frames={args.num_frames}")
    for f in range(args.num_frames):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        wp.synchronize_device()
        t0 = time.perf_counter()
        solver.step(state_0, state_1, None, contacts, _DT)
        wp.synchronize_device()
        step_ms[f] = (time.perf_counter() - t0) * 1000.0
        state_0, state_1 = state_1, state_0
        positions[f] = state_0.particle_q.numpy()
        nc = 0
        if hasattr(contacts, "soft_contact_count") and contacts.soft_contact_count is not None:
            nc = int(contacts.soft_contact_count.numpy()[0])
        n_contacts[f] = nc
        if f % 50 == 0 or f == args.num_frames - 1:
            print(f"  frame {f}: y_min={positions[f][:,1].min():.3f} y_max={positions[f][:,1].max():.3f} "
                  f"n_contacts={nc} step_ms={step_ms[f]:.2f}")

    if not np.all(np.isfinite(positions)):
        print("ERROR: non-finite positions")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(positions=positions, n_contacts=n_contacts, step_ms=step_ms)
    if args.perf_timing and hasattr(solver, "get_timing_summary"):
        try:
            summary = solver.get_timing_summary()
            save_kwargs["timing_summary"] = np.asarray(repr(summary), dtype=object)
        except Exception as exc:  # noqa: BLE001
            print(f"  (perf timing summary not captured: {exc})")
    np.savez(args.out, **save_kwargs)
    print(f"Saved {args.out}  (step_mean={step_ms.mean():.2f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
