# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark sub-steps of FBA build_schur_complement on Demo 5.

Patches ``FBALinearSolver._build_schur_isodof`` with per-sub-step timers
that synchronize the device around each chunk:

  1. host prep (np.unique + isodof_rank + perm gather)
  2. upload (assign isodof_perm_d, isodof_rank_d)
  3. compute_wi_kernel
  4. compose_W_from_wi_kernel
  5. inverse-mapping CSR build (host)

Runs SqueezingBall for a small number of frames and prints averaged sub-step
times.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import warp as wp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    from newton._src.solvers.fba.linear_solver import FBALinearSolver  # noqa: PLC0415

    # ---- monkey-patch _build_schur_isodof to time sub-steps -------------
    samples = defaultdict(list)
    extras = defaultdict(list)

    original_isodof = FBALinearSolver._build_schur_isodof  # noqa: F841 — kept for restore

    def instrumented(
        self,
        row_particle,
        row_particle_d,
        row_dir_d,
        row_alpha_d,
        total_rows,
        compute_wi_tiled_kernel,
        densify_S_iso_scaled_kernel,
        compose_W_from_wi_kernel,
        download=True,
    ):
        from newton._src.solvers.fba.kernels import (  # noqa: PLC0415
            _WI_TILE_K,
            _WI_TILE_M,
            _WI_TILE_N,
            _WI_TILE_THREADS,
        )

        tile_m = int(_WI_TILE_M)
        tile_n = int(_WI_TILE_N)
        tile_k = int(_WI_TILE_K)
        N = self.n
        dev = self.device

        wp.synchronize_device()
        t0 = time.perf_counter()
        isodofs_h = np.unique(row_particle.astype(np.int64)).astype(np.int32)
        k = int(isodofs_h.size)
        isodof_rank_h = np.full(N, -1, dtype=np.int32)
        isodof_rank_h[isodofs_h] = np.arange(k, dtype=np.int32)
        perm_h = self._perm.numpy()
        isodof_perm_h = perm_h[isodofs_h].astype(np.int32)
        wp.synchronize_device()
        samples["1_host_prep"].append((time.perf_counter() - t0) * 1000.0)
        extras["k_unique"].append(k)
        extras["total_rows"].append(total_rows)

        k_pad = ((k + tile_m - 1) // tile_m) * tile_m
        N_pad = ((N + tile_k - 1) // tile_k) * tile_k
        extras["k_pad"].append(k_pad)
        extras["N_pad"].append(N_pad)

        if not hasattr(self, "_isodofs_cap") or self._isodofs_cap < k_pad:
            cap = max(k_pad, int(getattr(self, "_isodofs_cap", 0) * 1.5) + 1)
            cap = ((cap + tile_m - 1) // tile_m) * tile_m
            self._isodofs_cap = cap
            self._isodof_perm_d = wp.empty(cap, dtype=wp.int32, device=dev)
            self._Wi_device_d = wp.empty(shape=(cap, cap), dtype=wp.float64, device=dev)
        if (
            not hasattr(self, "_S_iso_scaled_d")
            or self._S_iso_scaled_d.shape[0] < N_pad
            or self._S_iso_scaled_d.shape[1] < k_pad
        ):
            self._S_iso_scaled_d = wp.empty(shape=(N_pad, k_pad), dtype=wp.float64, device=dev)
        if not hasattr(self, "_isodof_rank_d") or self._isodof_rank_d.shape[0] != N:
            self._isodof_rank_d = wp.empty(N, dtype=wp.int32, device=dev)

        wp.synchronize_device()
        t0 = time.perf_counter()
        self._isodof_perm_d.assign(isodof_perm_h)
        self._isodof_rank_d.assign(isodof_rank_h)
        wp.synchronize_device()
        samples["2_upload"].append((time.perf_counter() - t0) * 1000.0)

        Wi_cap = int(self._Wi_device_d.shape[0])
        if Wi_cap == k_pad:
            Wi_v = self._Wi_device_d
        else:
            Wi_v = wp.array(
                ptr=self._Wi_device_d.ptr, dtype=wp.float64, shape=(k_pad, k_pad), strides=(Wi_cap * 8, 8), device=dev
            )
        S_n = int(self._S_iso_scaled_d.shape[0])
        S_k = int(self._S_iso_scaled_d.shape[1])
        if S_n == N_pad and S_k == k_pad:
            S_iso_v = self._S_iso_scaled_d
        else:
            S_iso_v = wp.array(
                ptr=self._S_iso_scaled_d.ptr,
                dtype=wp.float64,
                shape=(N_pad, k_pad),
                strides=(S_k * 8, 8),
                device=dev,
            )

        wp.synchronize_device()
        t0 = time.perf_counter()
        S_iso_v.zero_()
        wp.launch(
            densify_S_iso_scaled_kernel,
            dim=k,
            inputs=[
                self._ST_bsr.offsets,
                self._ST_bsr.columns,
                self._ST_bsr.values,
                self._Dinv,
                self._isodof_perm_d,
            ],
            outputs=[S_iso_v],
            device=dev,
        )
        wp.synchronize_device()
        samples["3a_densify"].append((time.perf_counter() - t0) * 1000.0)

        wp.synchronize_device()
        t0 = time.perf_counter()
        wp.launch_tiled(
            compute_wi_tiled_kernel,
            dim=(k_pad // tile_m, k_pad // tile_n),
            inputs=[S_iso_v, Wi_v, N_pad],
            block_dim=_WI_TILE_THREADS,
            device=dev,
        )
        wp.synchronize_device()
        samples["3b_matmul"].append((time.perf_counter() - t0) * 1000.0)

        wp.synchronize_device()
        t0 = time.perf_counter()
        wp.launch(
            compose_W_from_wi_kernel,
            dim=(total_rows, total_rows),
            inputs=[
                row_particle_d,
                row_dir_d,
                row_alpha_d,
                self._isodof_rank_d,
                Wi_v,
            ],
            outputs=[self._W_device_d],
            device=dev,
        )
        wp.synchronize_device()
        samples["4_compose_W"].append((time.perf_counter() - t0) * 1000.0)

        if not download:
            return None
        wp.synchronize_device()
        t0 = time.perf_counter()
        W = self._W_device_d.numpy()[:total_rows, :total_rows].copy()
        wp.synchronize_device()
        samples["5_W_dl"].append((time.perf_counter() - t0) * 1000.0)
        return W

    FBALinearSolver._build_schur_isodof = instrumented

    # ---- run demo 5 -----------------------------------------------------
    from newton.solvers import SolverFBA  # noqa: PLC0415
    from scripts.fba_demo5_squeezing_ball import (  # noqa: PLC0415
        CYLINDERS,
        DT,
        FRICTION_MU,
        NSN_ITERATIONS,
        PD_ITERATIONS,
        build_model,
    )

    model, pipeline, contacts, _verts_world, mu, lam = build_model()
    shape_omega = {i: CYLINDERS[i][2] for i in range(4)}
    n_max_contacts = model.particle_count
    mu_override = np.full(n_max_contacts, FRICTION_MU, dtype=np.float64)

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

    s_in = model.state()
    s_out = model.state()
    for f in range(args.frames):
        wp.synchronize()
        s_in.clear_forces()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, DT)
        wp.synchronize()
        s_in, s_out = s_out, s_in
        if f == args.warmup - 1:
            for k_ in list(samples.keys()):
                samples[k_].clear()
            for k_ in list(extras.keys()):
                extras[k_].clear()

    print("\n========== Schur isodof sub-step timing ==========")
    for k_ in sorted(samples.keys()):
        arr = np.asarray(samples[k_])
        if arr.size == 0:
            print(f"  {k_}: no samples")
            continue
        print(
            f"  {k_:18s}: mean={arr.mean():7.3f} ms  median={np.median(arr):7.3f} ms  "
            f"max={arr.max():7.3f} ms  n={arr.size}"
        )
    print(f"\n  k_unique (mean):   {np.mean(extras['k_unique']):.1f}")
    print(f"  total_rows (mean): {np.mean(extras['total_rows']):.1f}")
    overall = sum(np.mean(samples[k_]) if samples[k_] else 0.0 for k_ in samples)
    print(f"  sub-step total:    {overall:.3f} ms")
    summary = solver.get_timing_summary(last_n_steps=None)
    print(f"  solver.schur_ms:   {summary['schur_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
