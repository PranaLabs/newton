# SolverFBA Development Progress

Live tracker for the Newton port of RealSim's "Fast But Accurate" projective-dynamics cloth solver.

- **Branch**: `ziqiu/fba-solver-design` on `PranaLabs/newton`
- **Upstream target**: `newton-physics/newton` (eventual PR after feature parity)
- **Last updated**: 2026-05-15

## Status overview

| Phase | Status | Notes |
|---|---|---|
| MVP (cloth + Pin + gravity, ARAP + isometric bending) | ✅ Done | 13 tasks, 27 tests, example runs |
| Robustness tests (extreme configs / forces / reconfigure / stability) | ✅ Done | 14 new tests |
| Stability diagnosis (proved 30× gap was parameter, not bug) | ✅ Done | `docs/superpowers/specs/2026-05-15-fba-stability-diagnosis.md` |
| Pre-commit / lint cleanup | ✅ Done | All hooks green |
| **Numerical RealSim cross-check** | 🚧 In progress | This week |
| Contact / collision (plane / primitives + CCD/DCD) | ⬜ Not started | Largest remaining feature gap |
| Softbody (PDTetrahedronEnergy) | ⬜ Not started | Builder + new project_tet kernel |
| Material model extensions (Corotational / Neo-Hookean / StVK) | ⬜ Not started | Dispatch hook already in place |
| Differentiability via `wp.Tape()` | ⬜ Not started | Forward path probably already grad-friendly |
| Performance: `compute_lower_inverse` accel | ⬜ Not started | N=10K = 40 s pure Python; numba/Cython candidates |
| Upstream PR prep (split, squash, strip docs/superpowers/) | ⬜ Not started | After feature parity |

## Done — by commit

| SHA | Subject |
|---|---|
| `5f3d560f` | Add SolverFBA MVP design spec |
| `489d07d3` | Add SolverFBA MVP implementation plan |
| `76186d14` | Plan: use wp.svd2 instead of hand-rolled 2x2 eigen |
| `31522558` | Plan: document bsr_mv execution semantics |
| `23a831f4` | Task 1 — Package skeleton |
| `209004f3` | Task 2 — Add build_pd_system (initial) |
| `5d9688d8` | Task 2 fix — correct isometric bending cotangent formula |
| `84a58691` | Task 2 cleanup — remove dead branches |
| `6974935a` | Task 3 — Port LDLT_computeLowerInverse |
| `ba6c38f4` | Task 3 fix — loop bound and perf doc |
| `f3ce4837` | Task 4 — factorize_and_sparse_inverse pipeline |
| `f3f1cbca` | Task 5 — FBALinearSolver with Warp BSR |
| `95245edf` | Task 6 — trivial vec3 kernels |
| `74a750e7` | Task 7 — ARAP stretching kernel + 3x2 SVD |
| `a71560d2` | Task 7 fix — Dm_inv transpose bug |
| `2ca93d8f` | Task 8 — isometric bending projection kernel |
| `997b5518` | Task 9 — Pin projection kernel |
| `9a91da78` | Task 10 — SolverFBA __init__ + _setup_pd_system |
| `5faff4c8` | Task 11 — SolverFBA.step PD outer loop |
| `df34203b` | Task 11 fix — float64 precision + permutation order |
| `416ff0d7` | Task 12 — public API exports |
| `2d96ff5b` | Task 13 — integration tests T7/T8 + example |
| `ee0a0794` | Polish SolverFBA docs and fix pre-commit lint |
| `743f1df8` | Robustness A — extreme-config tests |
| `19c3ab5b` | Robustness B — external-force + state-buffer tests |
| `cd1d97ad` | Robustness C — reconfigure (dt change, notify_model_changed) tests |
| `8826d728` | Robustness D — stability boundary + regression test |
| `d14fdb75` | Use realistic tri_ke=1e4 in examples/tests |
| `a8405f87` | Clean pre-commit warnings + commit diagnosis doc |

## In progress

### Numerical cross-check with RealSim

**Goal**: confirm Newton port matches RealSim within tight tolerance on identical scene + identical step count, not just "formulas match by hand-check".

**Plan**:
1. Verify RealSim Python module (`realsim_py`) imports and runs a minimal scene from `/home/ziqiu/work/RealSim_py/realsim_py/`.
2. Pick a canonical test scene with deterministic initial conditions (likely a small hanging cloth, ≤256 particles, ≤100 steps).
3. Translate the same scene to Newton's `ModelBuilder` (positions, indices, mass, ke, edge stencil).
4. Run both, dump positions every 10 steps.
5. Compare: max L2 norm of position delta vs initial-position scale. Target: relative error < 1e-3 over 100 steps.
6. If gap exceeds target, identify the divergence source (per-step? cumulative? specific element?).

**Acceptance**: a unittest `TestFBARealSimAgreement` that loads pre-computed RealSim trajectories from a fixture file and asserts agreement. Doesn't require RealSim at test time; the fixture is captured once and committed.

## Backlog (ordered)

1. **Contact + collision** — `update_contacts` impl, plane / sphere / primitive CCD/DCD, Schur-complement `addHAinvHT` / `apply_constraint` in `FBALinearSolver`.
2. **Softbody (tet FEM)** — PDTetrahedronEnergy port + builder + project_tet_arap_kernel.
3. **Material models** — Corotational (almost free), Neo-Hookean (per-tri L-BFGS), StVK.
4. **Performance** — `compute_lower_inverse` numba/Cython acceleration. Decision criterion: when target mesh size routinely > 5K particles and setup time exceeds 10 s.
5. **Differentiability** — `wp.Tape()` round-trip, gradients on initial conditions / params.
6. **Cross-solver bench** — performance + accuracy table FBA vs Style3D vs VBD on common scenes.

## Known limitations (acknowledge, don't fix yet)

- `compute_lower_inverse`: ~6 s for N=2.5K, ~40 s for N=10K (pure Python). Documented.
- Position output is float32 (vec3 type); interior solve is float64. Precision floor ~1e-7 m. Documented.
- `stretching_model` must be `"arap"`; other values raise `NotImplementedError`. By design — extension hook ready.
- `update_contacts` raises. No contact yet.
- Free cloth (no pinned particles) works numerically but may produce COLAMD orderings with reduced numerical conditioning; flagged in code reviews, no observed failure on tested sizes.

## Open process notes

- `docs/superpowers/{specs,plans}/*` is internal methodology — strip before upstream PR.
- Commit history is fine-grained (29 commits since branch off `main`); consider interactive rebase + squash before upstream PR.
- AGENTS.md compliance audited at MVP wrap-up; re-audit before PR (CHANGELOG entry exists, SPDX present, naming OK, no new heavy deps).

## How to update this file

Whenever a substantial unit lands (a task / a fix-set / a phase boundary), append the SHA + subject to the "Done — by commit" table and adjust the status overview. Add a short paragraph under "In progress" when starting any new work; move it to "Done" once landed.
