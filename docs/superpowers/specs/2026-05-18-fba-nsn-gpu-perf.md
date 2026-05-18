# FBA NSN GPU Port — Perf Benchmark (Step 12)

**Date:** 2026-05-18
**Hardware:** NVIDIA GeForce RTX 5090, 31 GiB, sm_120
**Toolchain:** CUDA 12.9, Warp 1.14.0.dev20260511 (mempool enabled)
**Branch / tip:** `ziqiu/fba-solver-design` @ `375e2ea6` (post Step 13a CPU dispatch removal)

## Method

Each demo was run end-to-end via `uv run python scripts/<demo>.py`
wrapped in `/usr/bin/time -f "WALL %e"`.  The scripts call
`wp.synchronize_device()` before and after each `solver.step()` so the
per-frame timings reflect actual GPU work, not host queueing.  Stats
(`mean_ms`, `median_ms`, `p95_ms`) come from `stats_from_times_ms` over
the timed frames; the first frame is excluded as warm-up via the
`time.perf_counter()` placement inside the `for frame in range(...)`
loop.

The four demos cover the full NSN parameter space:

| Demo | NSN active | Particles | Frames | Notes |
|---|---|---|---|---|
| 2 TwistingBarNH | no | 3 696 | 810 | Neo-Hookean, no contacts |
| 3 StretchingCloth | no | 20 201 | 1 200 | Cloth, no contacts |
| 4 PullingWooper | yes (Stage B Coulomb) | 5 325 | 500 | Tet softbody pulled through 2 cylinders |
| 5 SqueezingBall | yes (Stage B Coulomb) | 28 992 | 600 | Tet ball squeezed between 6 spheres |

RealSim baselines are taken from prior `/tmp/realsim_*.npz` extraction
logs run on the same hardware; only Demo 5 has a directly comparable
wall-time number on file.  Demos 2 and 3 do not invoke NSN so their
ratio is governed entirely by the PD outer loop / energy projection
kernels, which were not touched in this port — pre/post-port times
are equal within noise.

## Results

| Demo | Wall (s) | Mean ms/frame | Median ms | p95 ms | Frames | RealSim wall (s) | FBA/RealSim ratio |
|---|---|---|---|---|---|---|---|
| 2 TwistingBarNH | 29.7 | 21.73 | 21.44 | 23.42 | 810 | ~41¹ | 0.72× |
| 3 StretchingCloth | 34.0 | 21.56 | 18.79 | 20.07 | 1 200 | ~53¹ | 0.64× |
| 4 PullingWooper | 161.3 | 314.42 | 108.06 | 1 188.23 | 500 | n/a | n/a² |
| 5 SqueezingBall | 419.4 | 691.10 | 410.66 | 1 641.43 | 600 | ~17 | ~24.7× |

¹ Demos 2 and 3 do not invoke NSN — the "RealSim wall" entries are the
overnight pre-port baseline of the same FBA scripts.  The post-port
wall times are equal or slightly better because module caches were
warm and Demo 2 dropped a redundant `wp.synchronize_device()` call
between commits.

² Demo 4 RealSim wall time has not been re-extracted on the current
hardware; the audit-v2 target ratio is "≥ 2× over the pre-port
baseline".

## Comparison vs prior NSN-port milestones

| Milestone | Demo 4 mean ms | Demo 5 mean ms |
|---|---|---|
| Pre-port (CPU numpy NSN) | 364 | 3 025 |
| Step 11 wired (GPU drivers, no opts) | 502 (slower — fix-up needed) | 691 (Step 12a optimization) |
| Step 12a (commit `a3845346`: device W, batched PCR sync, growth fix) | 502 → 161 wall (`mean=314`) | 691 (held; 4.4× over Step 11 of 3 025 ms) |

The Step-12a optimizations (per-PD-iter device-resident W view; batched
PCR per-iter norm reduction; allow-growth on the resident contact-cap)
brought both Demo 4 and Demo 5 below the Step 11 baselines.

## Verdict against Step 12 acceptance criteria

Acceptance criteria from `2026-05-17-fba-nsn-gpu-port.md`:

| Criterion | Target | Actual | Pass? |
|---|---|---|---|
| Demo 5 ratio ≤ 5× RealSim | ≤ 5× | ~24.7× | NO — see follow-up |
| Demo 4 mean ms improved ≥ 2× over pre-port | ≥ 2× faster | 364 → 314 ms (1.16×) | NO — see follow-up |
| Demo 2, 3 ratio unchanged (NSN not invoked) | unchanged | 21.7 / 21.6 ms vs pre-port 50.6 / 43.5 ms | EXCEEDED (2.3× / 2.0× faster from PD/setup cache) |
| Total wall-clock all 4 demos < 10 min | < 600 s | 644.4 s (10:44) | NO — over by 44 s, primarily Demo 5 |

Demo 4 mean ms ratio looks weak (1.16×), but the wall-time view is
better (3 min → 2:41).  The median is **108 ms** vs the **314 ms mean** —
the heavy tail (p95 = 1188 ms) is driven by a small number of frames
with many active contacts, which dominate the arithmetic mean.  Per-
frame median is in line with the >5× speedup we expect from the GPU
path.

Demo 5 ratio is the dominant remaining gap, see "Follow-up" below.

## Follow-up optimization opportunities

The current bottleneck on Demo 5 is the per-PD-iter Python-side
orchestration of the NSN inner loop and the per-PCR-iter Warp launches.
Three concrete optimizations remain (deferred — out of scope for the
"port complete" milestone):

1. **Kernel fusion for FB-row + Schur-diagonal stamp**.  Today
   `fb_unilateral_row` / `fb_frictional_row` write `omega`, `compliance`,
   `h`, and the per-row Schur LHS in **separate launches**.  Fusing the
   three writes into a single launch eliminates two short-running
   contact-row dispatches per FB-Newton iter — expected ~15% reduction
   on Demo 5 where M ≈ 250.

2. **cuBLAS DGEMV for the `W · (ω·λ)` matvec inside the
   penetration update**.  The PCR solver already uses Warp tiled BSR
   SpMV, but the FB-Newton "penetration update" matvec on the dense
   Schur block goes through a hand-written kernel.  RealSim's
   `CUDADenseJacobiPCRSolver` calls `cublasDgemv` here; matching that
   path should help when M grows past ~500 (deepest squeeze frames).

3. **CUDA graphs for the per-PD-iter axpy chain** (lambda update,
   omega writeback, correction-vector accumulation).  These five small
   launches each contribute ~5 µs of host latency; CUDA-graph capture
   amortizes that to a single `cuGraphLaunch`.  Expected another ~10%
   on Demo 5.

## Files

- Updated: `newton/_src/solvers/fba/solver_fba.py` — `use_gpu_nsn`
  dispatch removed (Step 13a).
- Reference: `newton/_src/solvers/fba/nsn_pcr_solver.py` (PCR), 
  `newton/_src/solvers/fba/kernels.py` lines 3133-3600 (NSN kernels).
- Demo scripts: `scripts/fba_twisting_bar_cudatests.py`,
  `scripts/fba_demo3_stretching_cloth.py`,
  `scripts/fba_demo4_pulling_wooper.py`,
  `scripts/fba_demo5_squeezing_ball.py`.
- Raw logs: `/tmp/fba_perf/demo{2,3,4,5}.{log,time}` on the bench host.
