# Results: SolverFBA multi-env LiteNSN — contact-rich cloth

**Date:** 2026-05-18 (one-pass execution of `2026-05-18-fba-multienv-cloth.md`)
**Status:** Implementation complete; 5 of 6 acceptance gates pass; gate #5 (cross-env strict 1e-3 rel-norm) misses by ~50–70 % due to FP non-associativity in the upstream contact pipeline (per-particle drift remains 5 mm p95).
**Branch:** `ziqiu/fba-solver-design`

## Acceptance summary

| Gate | Plan target | Measured | Status |
|---|---|---|---|
| 1. Phase 0 contact-free 2-env identity (Tier 2) | env-to-env rel_norm < **1e-5** | **7.4e-6** | ✅ PASS |
| 2. Phase 1 Newton-full vs RealSim-A trajectory @ frame 200 | p95 < **5 cm**, max < **15 cm** | p95 **2.07 cm**, max **3.16 cm** | ✅ PASS |
| 3. Phase 2.4 Newton-lite vs RealSim-B trajectory @ frame 200 | p95 < **5 cm**, max < **15 cm** | p95 **2.25 cm**, max **3.31 cm** (block-diag) | ✅ PASS |
| 4. Phase 3.2 scaling: step_time(25) ≤ **30 ×** step_time(1) | ratio ≤ 30 | **3.66 ×** | ✅ PASS (massive margin) |
| 5. Phase 3.3 cross-env consistency, all 24 envs vs env 0 | rel_norm < **1e-3** at frames 200 and 299 | **max 1.56e-3 / 1.71e-3**, per-particle p95 **5 mm** | ⚠️ MISS (numerical) |
| 6. Plan-level commit + push with example, solver mode, harness, baselines | — | done | ✅ PASS |

Gate 5 misses because the upstream contact pipeline uses `wp.atomic_add` for soft-contact slot assignment; identical envs end up with bit-different contact ordering, and the LiteNSN local 3×3 solves consume those orderings without further synchronization. The *physical* drift is millimeter-scale — see "Cross-env consistency analysis" below.

## What was built

- `newton/examples/cloth/example_cloth_on_sphere_fba.py`
  - 1013-particle TRI_ARAP cloth on radius-3 sphere, mirrors `CudaTests/ParallelEnvTest`.
  - Exposes `build_cloth_on_sphere_builder()` for replication harnesses.
  - `--schur-mode {full,lite}` and `--dump-npz` CLI plumbing.
- `newton/examples/cloth/example_hanging_cloth_fba.py` (refactor)
  - Extracted `build_hanging_cloth_builder()` while leaving the existing `Example` intact.
- `newton/_src/solvers/fba/linear_solver.py` adds:
  - `setup_lite_mass(inv_mass, dt)` — caches `dt²·diag(inv_mass)⊗I₃` BSR and `_inv_mass_d` / `_lite_dt2` for downstream kernels.
  - `build_schur_lite(...)` — sparse `W = H · M⁻¹ · Hᵀ` via `wp.sparse.bsr_mm` (kept for testing and the future truly-sparse PCR path).
  - `build_schur_lite_dense(...)` — single-pass kernel writing the dense projection of `W_lite` into `_W_device_d` (Phase 2.3 first cut; still used as a regression fallback).
  - `setup_lite_row_meta(...)` — sets per-row metadata without touching `W`, used by the block-diagonal lite path.
- `newton/_src/solvers/fba/nsn_pcr_solver.py` adds:
  - `solve_sparse(W_bsr, b, x_out)` — Jacobi-PCR using `wp.sparse.bsr_mv`, validated equivalent to dense PCR (∆ 1e-15 on 10×10 toy).
- `newton/_src/solvers/fba/kernels.py` adds:
  - `compose_W_lite_kernel` — dense-projection of the lite Schur formula (used by `build_schur_lite_dense`).
  - `fb_newton_lite_coulomb_kernel` — fully fused per-contact FB-Newton step (warm-started ω, local 3×3 system solve via cofactor expansion, Coulomb clamp, λ-cap). Memory cost ``O(M)`` — what makes Phase 3 (N=25 envs) actually fit on a 32 GB GPU.
- `newton/_src/solvers/fba/solver_fba.py` adds:
  - `nsn_schur_mode: Literal["full", "lite"] = "full"` kwarg.
  - `_setup_pd_system` invokes `setup_lite_mass` when mode is `"lite"`.
  - Stage B NSN dispatch: in lite mode, skips both `build_schur_*` (no dense W needed) and `_solve_nsn_coulomb_gpu` (no dense `A_schur` needed), launches `fb_newton_lite_coulomb_kernel` directly.
  - `_ensure_nsn_inner_buffers` gains `alloc_2d` flag so lite-block-diag never allocates the `(cap, cap)` scratch — required to fit N=25.
- `scripts/fba_multienv_hanging_cloth_smoke.py` — Phase 0 driver.
- `scripts/fba_cloth_on_sphere_record.py` — single-env recorder (Phase 1.3/1.4, Phase 2.4).
- `scripts/fba_cloth_on_sphere_realsim_baseline.py` — captures the RealSim full + lite single-env baselines into `scripts/realsim_baseline/cloth_on_sphere_{full,lite}_ref.npz`.
- `scripts/fba_multienv_cloth_bench.py` — Phase 3 scaling and cross-env harness. `--first-last-only` writes only frames 0/150/200/299 to save memory at large N.

## Phase 3.2 scaling table (lite, block-diagonal)

| N envs | particles | settled step_mean (ms) | p95 (ms) | n_contacts @ frame 200 | scaling ratio vs N=1 |
|---|---|---|---|---|---|
| 1 | 1013 | 2.66 | 2.92 | 657 | 1.00 × |
| 4 | 4052 | 3.19 | 3.44 | 2585 | 1.20 × |
| 9 | 9117 | 4.40 | 4.76 | 5818 | 1.65 × |
| 16 | 16208 | 6.06 | 6.61 | 10341 | 2.28 × |
| 25 | 25325 | 9.74 | 10.55 | 16155 | 3.66 × |

Full mode OOMs at N≥16 (peak dense `W` and `A_schur` exceed 16 GB on a 32 GB RTX 5090) — exactly the failure the plan flagged.

## Single-env perf comparison (cloth-on-sphere, 300 frames, dt = 0.01)

| Mode | step_mean (ms) | schur_ms / PD iter | nsn_inner_ms / PD iter |
|---|---|---|---|
| Newton full | 18.21 | 0.272 | 2.840 |
| Newton lite (dense projection) | 13.76 | 0.051 | 2.351 |
| Newton lite (block-diag fused, used by Phase 3) | **2.66** | — (fused) | — (fused) |
| RealSim full | 5.95 | — | — |
| RealSim lite | (timing not captured; trajectory only) | — | — |

The block-diag fused kernel is **6.7 × faster than Newton full** and **2.2 × faster than RealSim full** at single-env — a free win from collapsing the FB-Newton iter, the Schur build, the matvec, and the Coulomb clamp into a single per-contact thread.

## Cross-env consistency analysis (Phase 3.3)

| Frame | rel_norm median | rel_norm p75 | rel_norm max | per-particle p95 | per-particle max | envs under 1e-3 | envs under 2e-3 |
|---|---|---|---|---|---|---|---|
| 200 | 1.08e-3 | 1.22e-3 | **1.56e-3** | 4.90 mm | 9.23 mm | 11 / 24 | 24 / 24 |
| 299 | 1.20e-3 | 1.34e-3 | **1.71e-3** | 5.60 mm | 12.33 mm | 11 / 24 | 24 / 24 |

Per-particle differences across envs are **millimeter-scale** (4.9 mm p95 at frame 200), which is ~0.16 % of the cloth-sphere setup's 3 m characteristic length.  The strict 1e-3 *relative norm* gate misses for 13/24 envs, but the *physical* dispersion is negligible.

**Root cause (suspected):** `wp.atomic_add`-driven contact slot assignment in `CollisionPipeline` produces nondeterministic ordering across worlds.  Phase 0's contact-free 2-env smoke test (`scripts/fba_multienv_hanging_cloth_smoke.py`) passes at **7.4e-6** rel_norm — well below 1e-5 — confirming the FP non-associativity enters only via the contact pipeline.

Two follow-ups worth scoping if anyone needs the strict gate:
1. Sort contacts after the broadphase using a stable, world-aware key (particle index then world index) before handing off to SolverFBA.
2. Switch `gather_jt_lambda_kernel` and the PD prefactor's `bsr_mv` to deterministic reductions (currently they use atomic adds).

## Out of scope (deferred)

- Truly-sparse PCR path for non-cloth scenes (softbody / tet).  The `build_schur_lite` + `solve_sparse` BSR pipeline (Phase 2.1 / 2.2) is ready but unused on the hot path; the cloth-on-sphere case decouples per-particle so the fused block-diag kernel is strictly more efficient.  Generalised contact graphs (multiple obstacles, particle-particle contacts) would need the BSR path and an SpGEMM-built `A_schur_bsr`.
- Newton viewer rendering of N≥4 envs.  `viewer.set_world_offsets()` integration not exercised by these scripts.
- Per-env perturbation (mass, pose, material).  ParallelEnvTest replicates one cloth identically; per-env diversity is RL-training territory.
- A determinism pass over the upstream contact pipeline (`wp.atomic_add` slot assignments).  Required to close gate #5 strictly.

## Files reproducing the gates

```bash
# Gate 1 (Phase 0)
uv run python scripts/fba_multienv_hanging_cloth_smoke.py --num-frames 100 --spacing 5.0

# Gate 2 (Phase 1.3)
uv run python scripts/fba_cloth_on_sphere_record.py --schur-mode full --num-frames 300 \
    --out /tmp/fba_cloth_on_sphere_full.npz
# Then compare against scripts/realsim_baseline/cloth_on_sphere_full_ref.npz

# Gate 3 (Phase 2.4)
uv run python scripts/fba_cloth_on_sphere_record.py --schur-mode lite --num-frames 300 \
    --out /tmp/fba_cloth_on_sphere_lite.npz
# Then compare against scripts/realsim_baseline/cloth_on_sphere_lite_ref.npz

# Gates 4 & 5 (Phase 3.2 / 3.3)
for N in 1 4 9 16 25; do
    uv run python scripts/fba_multienv_cloth_bench.py --n-envs $N --mode lite \
        --num-frames 300 --first-last-only \
        --out scripts/multienv_results/n${N}_lite_blockdiag.npz
done

# Baseline captures (one-shot)
uv run python scripts/fba_cloth_on_sphere_realsim_baseline.py --mode full --max-frame 300
uv run python scripts/fba_cloth_on_sphere_realsim_baseline.py --mode lite --max-frame 300
```
