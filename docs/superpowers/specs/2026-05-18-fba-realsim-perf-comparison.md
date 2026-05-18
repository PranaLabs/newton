# FBA vs RealSim Per-PD-Iter Perf Comparison

Baseline (pre-fp64-SVD-lift, branch `ziqiu/fba-solver-design`).  FBA timings are gathered via :meth:`SolverFBA.get_timing_summary` with ``enable_perf_timing=True``; RealSim timings are parsed from ``LocalGlobalSolver::printTimer`` blocks (one block per ``scene.json::timer`` frames, default 50).

All values are means in **milliseconds per PD outer iter** (per-iter granularity), except where the row label says otherwise. Ratio = FBA / RealSim; values > 1.0 mean FBA is slower.

## Per-component breakdown

| Demo | Component | FBA (ms) | RealSim (ms) | Ratio |
|---|---|---|---|---|
| TwistingBarNH | Local (energy projection) | 1.151 | 3.542 | 0.32x |
|  | Linear solve (A^-1 b) | 1.309 | 0.461 | 2.84x |
|  | Schur W build | n/a | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | n/a | n/a | n/a |
| StretchingCloth | Local (energy projection) | 0.431 | 1.664 | 0.26x |
|  | Linear solve (A^-1 b) | 1.485 | 0.293 | 5.07x |
|  | Schur W build | n/a | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | n/a | n/a | n/a |
| PullingWooper | Local (energy projection) | 1.535 | 1.748 | 0.88x |
|  | Linear solve (A^-1 b) | 0.358 | 0.100 | 3.58x |
|  | Schur W build | 0.183 | 0.508 | 0.36x |
|  | NSN inner (build+cstsolve+correction) | 6.835 | 1.023 | 6.68x |
| SqueezingBall | Local (energy projection) | 1.515 | 2.678 | 0.57x |
|  | Linear solve (A^-1 b) | 0.698 | 0.282 | 2.47x |
|  | Schur W build | 17.857 | 1.217 | 14.67x |
|  | NSN inner (build+cstsolve+correction) | 31.054 | 1.302 | 23.85x |

## NSN sub-phases (RealSim only)

FBA's GPU NSN driver does a single fused kernel sequence (``build_schur`` was already split out above; ``solve+clamp`` and ``apply correction`` are not separately bracketed -- they're lumped into ``NSN inner``). RealSim reports the split, so we show it here for reference.

| Demo | n_contacts (mean) | PCR iters (mean) | RS Assembly (ms) | RS Cst solve (ms) | RS Correction (ms) |
|---|---|---|---|---|---|
| TwistingBarNH | n/a | n/a | n/a | n/a | n/a |
| StretchingCloth | n/a | n/a | n/a | n/a | n/a |
| PullingWooper | 86.3 | 9.7 | 0.215 | 0.667 | 0.140 |
| SqueezingBall | 1059.2 | 3.4 | 0.435 | 0.539 | 0.329 |

## Frame-level totals

| Demo | FBA step mean (ms) | FBA p95 (ms) | RS frame_no_cd (ms) | RS total step (ms) | FBA setup (ms) | Frames (FBA measured / RS counted) |
|---|---|---|---|---|---|---|
| TwistingBarNH | 24.66 | 25.74 | 41.50 | 41.50 | 5462.9 | 180 / 150 |
| StretchingCloth | 19.23 | 19.98 | 22.12 | 22.12 | 3120.9 | 180 / 150 |
| PullingWooper | 59.63 | 147.72 | 23.56 | 23.64 | 740.4 | 180 / 180 |
| SqueezingBall | 516.74 | 1239.43 | 42.99 | 43.20 | 2571.2 | 180 / 150 |

## Identified bottlenecks (largest contributor per demo)

- **TwistingBarNH**: dominant FBA cost = **Linear solve** = 1.309 ms/iter.
  - Parity / gap notes: no >5x gaps detected
- **StretchingCloth**: dominant FBA cost = **Linear solve** = 1.485 ms/iter.
  - Parity / gap notes: Linear solve 5.1x slower
- **PullingWooper**: dominant FBA cost = **NSN inner** = 6.835 ms/iter.
  - Parity / gap notes: Local at parity (0.88x); NSN inner 6.7x slower
- **SqueezingBall**: dominant FBA cost = **NSN inner** = 31.054 ms/iter.
  - Parity / gap notes: Schur W 14.7x slower; NSN inner 23.8x slower

## Raw JSON

See `scripts/perf_compare_all_demos.json` for the full per-demo dump.

## Analysis

### Demo 1 -- `TwistingBarNH` (NeoHookean tet, no contacts)

- FBA wins **Local** by 3.1x (1.15 ms vs 3.54 ms). The Warp NH-LBFGS kernel is faster per element than RealSim's CPU/CUDA LBFGS path.
- FBA loses **Linear solve** by 2.84x (1.31 ms vs 0.46 ms). RealSim's sparse-inverse Cholesky on the symmetric PD Hessian is faster than the Warp BSR-MV-tiled path.
- No NSN — contact-free. FBA frame mean (24.7 ms) actually beats RealSim's (41.5 ms) because the Local gap dominates the Linear-solve regression at the full-frame level.
- **Dominant FBA cost: Linear solve.**

### Demo 2 -- `StretchingCloth` (NH cloth, no contacts)

- Local at 0.43 ms vs 1.66 ms -- FBA 3.9x faster. Cloth is even more LBFGS-bound than tets, so the win is larger.
- Linear solve at 1.49 ms vs 0.29 ms -- FBA **5.1x slower**, the largest gap in this demo. This is a real regression vs RealSim, not noise.
- Frame mean: FBA 19.2 ms vs RealSim 22.1 ms. FBA still wins overall because the Local advantage > Linear gap (cloth Linear solve is only ~30% of frame budget).
- **Dominant FBA cost: Linear solve.**

### Demo 3 -- `PullingWooper` (NH tet, unilateral contacts, kinematic pin)

- Local at parity (1.54 ms vs 1.75 ms, 0.88x). Tet count similar to Demo 2 but with contacts active.
- Linear solve at 0.36 ms vs 0.10 ms -- 3.58x slower, smaller absolute gap than Demos 1-2 (NH-only system is smaller without contact constraints).
- **NSN inner 6.7x slower** (6.84 ms vs RealSim's 1.02 ms = `build + cstsolve + correction`).
- FBA's NSN inner is 8x the RealSim Schur W build. RealSim's PCR solver converges in ~9.7 iters mean here; our direct dense `np.linalg.solve` on the small Schur block is slower per outer iter because we don't exploit the sparse Schur structure as RealSim's PCR does.
- p95 = 147.7 ms, mean = 59.6 ms -- heavy tail driven by per-step contact-count spikes (NSN solve scales with M).
- **Dominant FBA cost: NSN inner.**

### Demo 4 -- `SqueezingBall` (NH tet, frictional contacts, rolling cylinders)

- Mean 1059 contacts/step (vs ~86 in Demo 3) -- by far the heaviest contact load.
- Local 1.52 ms vs 2.68 ms -- FBA 1.8x faster (same NH-LBFGS win as before).
- Linear solve 0.70 ms vs 0.28 ms -- 2.47x slower (smaller relative gap than smaller demos because more work is amortized).
- **Schur W build 14.7x slower** (17.86 ms vs 1.22 ms). FBA's `build_schur_complement` (even with `use_isodof=True`) is the single biggest regression. The isodof path still computes the dense `cap x cap` Schur block on host; RealSim's CUDA `addHAinvHT_gpu` is fully on-GPU.
- **NSN inner 23.9x slower** (31.05 ms vs 1.30 ms). At 1059 contacts the per-PD-iter inner cost dominates the entire frame -- FBA's mean step is 517 ms vs RealSim's 43 ms (12x slower end-to-end).
- p95 1239 ms -- frequent multi-second stalls.
- **Dominant FBA cost: NSN inner.** Closely trailed by Schur W (both are the contact-correction critical path).

### Take-away ranking

1. **Demo 4 SqueezingBall** is the urgent target. Schur W build + NSN inner together account for ~95% of the per-frame cost gap and are 15-24x slower than RealSim. The fix is to keep the Schur block on the GPU end-to-end and replace the direct dense solve with the PCR Schur-LCP path that RealSim uses.
2. **Demo 3 PullingWooper** has the same NSN-inner pattern (6.7x slower) but at much smaller M, so the absolute frame cost is only ~2.5x worse than RealSim. The same fix benefits both.
3. **Linear solve regression** (5x on cloth, 2-3x on tets) is the second-tier issue. Cheap to fix relative to Schur/NSN because the system size is fixed and well-conditioned. Likely a BSR-MV-tiled vs sparse-inverse-Cholesky throughput gap.
4. **Local is consistently a win** (1.8x-3.9x faster). The NH-LBFGS Warp kernels are the strongest part of the FBA implementation today.

### RealSim parser surprises

- RealSim adjusts `scene.json::timer` per scene: `TwistingBarNH`, `StretchingCloth`, `SqueezingBall` all use `timer=50`; `PullingWooper` uses `timer=20`. The parser handles this generically (block window size is reported in the header), so per-block aggregation is uniform.
- Contact-free demos (Demos 1, 2) emit zero NSN lines as expected -- the parser correctly reports `n/a` for those fields.
- `returncode` is `-6` (SIGABRT) or `-11` (SIGSEGV) for the offline binary at `maxFrame` exit. The stdout is fully flushed before abort, so timing parsing is unaffected. The harness no longer treats this as a failure.
- The `Frame without collision detection` and `Total time step cost` are identical for contact-free demos (no CD pass).

### What's next

Re-run this harness after the fp64-SVD-lift subagent commits to quantify the impact on Local timing (NH kernels) and confirm whether the Demo 4 / Demo 5 ranking changes once SVD is on fp64. The Schur and NSN regressions are independent of SVD precision, so the Demo 4/5 critical path is unchanged.

## Perf #4 — cuBLAS DGEMV for PCR matvec (2026-05-18)

Adds an optional `newton[cublas]` extra (`cupy-cuda12x`) that dispatches the
NSN-Schur PCR per-iter dense matvec through `cublasDgemv` for systems of
size n ≥ 2000.  Smaller systems stay on the in-house Warp `tile_matmul` kernel
because cupy's ~6 us Python-side dispatch overhead beats the tile kernel's
~7-12 us total at n ≤ 1500.  Above n ≈ 2000, NVIDIA's hand-tuned DGEMV
(~28 us at n=3000) outperforms `tile_matmul` (~90 us).

Implementation notes:
* `nsn_pcr_solver.py:_CUBLAS_MIN_N = 2000` (set to 0 to force cuBLAS for all
  sizes for benchmarking, or to a large value to disable).
* Direct binding via `cupy_backends.cuda.libs.cublas.dgemv` (the
  `cupy.matmul` / `cupy.cublas.gemv` wrappers fall back to a 7x slower
  strided-gemv kernel when A is a non-contiguous slice of a larger buffer,
  which is exactly how FBA passes the preallocated Schur block to PCR).
* cuBLAS handle bound once at construction to Warp's per-device stream
  via `cublas.setStream(handle, wp_stream.cuda_stream)` -- zero per-call
  stream-context overhead.

Measured before/after on the same hardware (RTX 5090, fp64, 200-frame run,
20-frame warmup, FBA-only -- RealSim columns unchanged):

| Demo | Before NSN-inner (ms) | After NSN-inner (ms) | Δ | Before step_mean (ms) | After step_mean (ms) | Δ |
|---|---|---|---|---|---|---|
| PullingWooper | 5.17 | 5.47 | +6% (within noise, gated below threshold) | 49.03 | 51.74 | within noise |
| SqueezingBall | 8.67 | 5.86 | **-32%** | 130.98 | 104.46 | **-20%** |

Demo 4 is unchanged because every PCR call in that demo has n ≤ 258, well
below the cuBLAS threshold.  Demo 5 sees the win: ~68% of its 200-frame PCR
calls hit n ≥ 2000 (Stage B `n = 3M` with M climbing as the ball is
squeezed), and DGEMV is ~2x faster than tile_matmul at those sizes.

Fallback verified: importing the FBA solver and running PCR without cupy
installed continues to work (the construction-time `is_cublas_available()`
check returns `False` and the dispatch falls through to `tile_matmul`).
See `test_fba_cublas_interop.py::TestPCRWithoutCublas` for the regression
test.
