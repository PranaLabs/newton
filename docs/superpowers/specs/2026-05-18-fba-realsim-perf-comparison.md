# FBA vs RealSim Per-PD-Iter Perf Comparison

Baseline (pre-fp64-SVD-lift, branch `ziqiu/fba-solver-design`).  FBA timings are gathered via :meth:`SolverFBA.get_timing_summary` with ``enable_perf_timing=True``; RealSim timings are parsed from ``LocalGlobalSolver::printTimer`` blocks (one block per ``scene.json::timer`` frames, default 50).

All values are means in **milliseconds per PD outer iter** (per-iter granularity), except where the row label says otherwise. Ratio = FBA / RealSim; values > 1.0 mean FBA is slower.

## Per-component breakdown

| Demo | Component | FBA (ms) | RealSim (ms) | Ratio |
|---|---|---|---|---|
| TwistingBarNH | Local (energy projection) | 1.152 | n/a | n/a |
|  | Linear solve (A^-1 b) | 1.326 | n/a | n/a |
|  | Schur W build | n/a | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | n/a | n/a | n/a |
| StretchingCloth | Local (energy projection) | 0.441 | n/a | n/a |
|  | Linear solve (A^-1 b) | 1.488 | n/a | n/a |
|  | Schur W build | n/a | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | n/a | n/a | n/a |
| PullingWooper | Local (energy projection) | 1.335 | n/a | n/a |
|  | Linear solve (A^-1 b) | 0.367 | n/a | n/a |
|  | Schur W build | 2.094 | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | 6.285 | n/a | n/a |
| SqueezingBall | Local (energy projection) | 1.276 | n/a | n/a |
|  | Linear solve (A^-1 b) | 0.706 | n/a | n/a |
|  | Schur W build | 0.846 | n/a | n/a |
|  | NSN inner (build+cstsolve+correction) | 2.851 | n/a | n/a |

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
| TwistingBarNH | 24.94 | 25.92 | n/a | n/a | 5758.9 | 80 / ? |
| StretchingCloth | 19.41 | 20.63 | n/a | n/a | 3202.3 | 80 / ? |
| PullingWooper | 20.49 | 18.92 | n/a | n/a | 704.2 | 80 / ? |
| SqueezingBall | 58.03 | 71.99 | n/a | n/a | 2365.8 | 80 / ? |

## Identified bottlenecks (largest contributor per demo)

- **TwistingBarNH**: dominant FBA cost = **Linear solve** = 1.326 ms/iter.
  - Parity / gap notes: no >5x gaps detected
- **StretchingCloth**: dominant FBA cost = **Linear solve** = 1.488 ms/iter.
  - Parity / gap notes: no >5x gaps detected
- **PullingWooper**: dominant FBA cost = **NSN inner** = 6.285 ms/iter.
  - Parity / gap notes: no >5x gaps detected
- **SqueezingBall**: dominant FBA cost = **NSN inner** = 2.851 ms/iter.
  - Parity / gap notes: no >5x gaps detected

## Raw JSON

See `scripts/perf_compare_all_demos.json` for the full per-demo dump.
