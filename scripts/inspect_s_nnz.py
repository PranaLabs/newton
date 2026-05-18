# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Inspect S factor sparsity for Demo 5."""

from __future__ import annotations

import numpy as np
import warp as wp


def main() -> int:
    from newton.solvers import SolverFBA  # noqa: PLC0415
    from scripts.fba_demo5_squeezing_ball import (  # noqa: PLC0415
        CYLINDERS,
        DT,
        FRICTION_MU,
        NSN_ITERATIONS,
        PD_ITERATIONS,
        build_model,
    )

    model, pipeline, contacts, _, mu, lam = build_model()
    shape_omega = {i: CYLINDERS[i][2] for i in range(4)}
    mu_override = np.full(model.particle_count, FRICTION_MU, dtype=np.float64)

    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=True,
        stretching_model="neohookean",
        mu=mu,
        lam=lam,
        mu_per_pair_override=mu_override,
        shape_angular_velocity=shape_omega,
        lambda_cap=1.0e12,
        pin_stiffness=1.0e10,
        enable_perf_timing=True,
    )
    s_in = model.state()
    s_out = model.state()
    s_in.clear_forces()
    pipeline.collide(s_in, contacts)
    solver.step(s_in, s_out, None, contacts, DT)
    wp.synchronize()
    ls = solver._linear_solver
    print(f"N = {ls.n}")
    st_offsets = ls._ST_bsr.offsets.numpy()
    nnz_per_col = np.diff(st_offsets)
    print(f"nnz(S) = {st_offsets[-1]}")
    print(f"nnz(S)/N = {st_offsets[-1] / ls.n:.2f}")
    print(
        f"S col nnz: min={nnz_per_col.min()}, max={nnz_per_col.max()}, mean={nnz_per_col.mean():.1f}, median={np.median(nnz_per_col):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
