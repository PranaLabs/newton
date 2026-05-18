# FBA vs RealSim Per-PD-Iter Perf Comparison

Baseline (pre-fp64-SVD-lift, branch `ziqiu/fba-solver-design`).  FBA timings are gathered via :meth:`SolverFBA.get_timing_summary` with ``enable_perf_timing=True``; RealSim timings are parsed from ``LocalGlobalSolver::printTimer`` blocks (one block per ``scene.json::timer`` frames, default 50).

All values are means in **milliseconds per PD outer iter** (per-iter granularity), except where the row label says otherwise. Ratio = FBA / RealSim; values > 1.0 mean FBA is slower.

**NSN-Schur PCR matvec path:** Warp ``tile_matmul`` (block-per-row tile reduction), wrapped in a CUDA graph that replays one full PCR iter per ``cuGraphLaunch`` (Perf #5).  An earlier optional cuBLAS DGEMV path (Perf #4 via cupy) was removed in Perf #6 because cuBLAS trips ``cudaErrorStreamCaptureImplicit`` during graph capture, and the captured ``tile_matmul`` path wins by a large margin (Demo 5 step_mean ~43 ms graph + tile_matmul vs ~99 ms cuBLAS + eager).

## Per-component breakdown

| Demo | Component | FBA (ms) | RealSim (ms) | Ratio |
|---|---|---|---|---|
| TwistingBarNH | Local (energy projection) | 1.177 | n/a | n/a |
|  | Linear solve (A^-1 b) | 1.340 | n/a | n/a |
|  | Schur W build | n/a | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | n/a | n/a | n/a |
| StretchingCloth | Local (energy projection) | 0.484 | n/a | n/a |
|  | Linear solve (A^-1 b) | 1.524 | n/a | n/a |
|  | Schur W build | n/a | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | n/a | n/a | n/a |
| PullingWooper | Local (energy projection) | 1.363 | n/a | n/a |
|  | Linear solve (A^-1 b) | 0.396 | n/a | n/a |
|  | Schur W build | 2.044 | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | 0.774 | n/a | n/a |
| SqueezingBall | Local (energy projection) | 1.300 | n/a | n/a |
|  | Linear solve (A^-1 b) | 0.734 | n/a | n/a |
|  | Schur W build | 0.873 | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | 3.318 | n/a | n/a |

## NSN sub-phases (RealSim only)

FBA's GPU NSN driver does a single fused kernel sequence (``build_schur`` was already split out above; ``solve+clamp`` and ``apply correction`` are not separately bracketed -- they're lumped into ``NSN inner``). RealSim reports the split, so we show it here for reference.

| Demo | n_contacts (mean) | PCR iters (mean) | RS Assembly (ms) | RS Cst solve (ms) | RS Correction (ms) |
|---|---|---|---|---|---|
| TwistingBarNH | n/a | n/a | n/a | n/a | n/a |
| StretchingCloth | n/a | n/a | n/a | n/a | n/a |
| PullingWooper | n/a | n/a | n/a | n/a | n/a |
| SqueezingBall | n/a | n/a | n/a | n/a | n/a |

## Frame-level totals

| Demo | FBA step mean (ms) | FBA p95 (ms) | RS frame_no_cd (ms) | RS total step (ms) | FBA setup (ms) | Frames (FBA measured / RS counted) |
|---|---|---|---|---|---|---|
| TwistingBarNH | 25.36 | 26.39 | n/a | n/a | 5794.5 | 80 / ? |
| StretchingCloth | 20.26 | 21.75 | n/a | n/a | 3235.2 | 80 / ? |
| PullingWooper | 19.02 | 19.59 | n/a | n/a | 716.9 | 80 / ? |
| SqueezingBall | 63.79 | 89.28 | n/a | n/a | 2410.8 | 80 / ? |

## Identified bottlenecks (largest contributor per demo)

- **TwistingBarNH**: dominant FBA cost = **Linear solve** = 1.340 ms/iter.
  - Parity / gap notes: no >5x gaps detected
- **StretchingCloth**: dominant FBA cost = **Linear solve** = 1.524 ms/iter.
  - Parity / gap notes: no >5x gaps detected
- **PullingWooper**: dominant FBA cost = **Schur W** = 2.044 ms/iter.
  - Parity / gap notes: no >5x gaps detected
- **SqueezingBall**: dominant FBA cost = **NSN inner** = 3.318 ms/iter.
  - Parity / gap notes: no >5x gaps detected

## Raw JSON

See `scripts/perf_compare_all_demos.json` for the full per-demo dump.
