# FBA-RealSim Verification Criteria (3-tier)

**Date:** 2026-05-17
**Status:** Active. Supersedes plan's original `Resolved decisions #6` single-threshold "1e-3 m drift" criterion.
**Applies to:** Phase 3 per-demo trajectory verification. Each demo emits a verification report containing all three tiers.

## Tier 1 — Primary gate (BINARY, must pass)

Per-demo physical event criterion plus stability sentinels. Either the demo passes or it doesn't — no partial credit.

### Per-demo physical event

| Demo | Binary success criterion |
|---|---|
| 2 TwistingBarNH | Bar twists to RealSim's final rotation angle, stays column-shaped (no fragmentation), no particle drifts off the bar axis by > 0.5× bar radius |
| 3 StretchingCloth | Pin reaches PULLING `maxlength`, cloth stretches without tearing (no triangle inversion), bulk of cloth particles stay within the stretched-cloth envelope |
| 4 PullingWooper | All 10 pin actions complete (`pulled = 10`); body deforms past the cylinder gap; final-frame `min_y` matches RealSim within Tier-2 drift |
| 5 SqueezingBall | Ball center reaches RealSim's final y-position with ball still cohesive (ball-radius blob, not a particle cloud); no particle outside `\|q\| > 50 m` at any frame |

### Stability sentinels (apply to every demo)

- No NaN in any particle position across all frames
- No `Inf` in any particle position
- No particle escapes the scene bounding box × 10 (a generous "particle didn't fly off into space" check)

### Failure semantics

If Tier 1 fails, the demo is **NOT GREEN** regardless of Tier-2/3 numbers. Investigate immediately; do not ship.

## Tier 2 — Secondary gate (QUANTITATIVE, target — investigate if missed)

Per-frame max particle position drift vs RealSim's `output_obj_0.abc` reference trajectory.

### Per-demo drift threshold

| Demo | Threshold | Rationale |
|---|---|---|
| 2 TwistingBarNH (no contact) | **`max(drift_t) < 1e-3 m`** over all frames | fp32 SVD vs fp64 RealSim: per-particle ~1e-7 noise/frame; 810 frames cumulative ~8e-5 m; 1e-3 m gives 10× headroom. Plan's original target |
| 3 StretchingCloth (no contact) | **`max(drift_t) < 1e-3 m`** | Same as above; cloth has TRI NH energy but no contact, similar noise budget |
| 4 PullingWooper (moderate contact) | **`max(drift_t) < 1e-2 m`** | 2 cylinder contacts; lexsort ordering noise; Schur Cholesky impl differs from RealSim; NH non-linear amplification in compression basin |
| 5 SqueezingBall (heavy contact) | **`max(drift_t) < 5e-2 m`** (or `< 5%` of ball radius) | 4 cylinders + plane + Coulomb friction; NH at extreme compression; widely-divergent contact-detection ordering can produce sub-frame trajectory drift that gets amplified |

`drift_t = max_p ||fba_pos[t, p] - realsim_pos[t, p]||_2` (Euclidean per-particle, take the max across particles, repeat per frame).

### Why per-demo thresholds (not a single number)

A single fixed threshold (plan's original 1e-3 m) is too tight for contact-heavy demos (4, 5) because:
- fp32 vs fp64 Cholesky differ by ~1e-7 per solve; per-PD-iter compounds
- GPU non-associative reductions in the Schur build produce up to 1e-7 per W entry
- NH energy basin is narrow under compression; gradient noise gets amplified non-linearly
- Contact-detection ordering differences (FBA's lexsort vs RealSim's natural order) produce permuted-but-equivalent λ; gather operations expose ordering as ~1e-6 per particle per step

For contactless demos these effects are smaller — 1e-3 m is realistic.

### Failure semantics

If Tier 2 misses (drift exceeds threshold) but Tier 1 passes:
- **Do NOT auto-fail.** Investigate.
- Tier 3 diagnostics will indicate whether the gap is local (a few outlier particles) or global (everything is shifted).
- Outliers (e.g., one free corner particle bouncing): acceptable if rest of mesh agrees
- Global shift: indicates systematic divergence, queue for investigation

## Tier 3 — Tertiary diagnostics (always reported, never auto-fail)

Reported per demo regardless of Tier 1/2 outcome. Used to localize gaps when Tier 2 misses.

### Diagnostics (per demo, per frame)

| Diagnostic | What it captures | Expected discrepancy |
|---|---|---|
| **COM trajectory** (`(fba_com[t] − realsim_com[t])`) | Bulk motion alignment; robust to single-particle outliers | `< 1e-3 m` |
| **Kinetic energy curve** (`KE_t = sum_p 0.5·m_p·\|v_p\|²`) | Systematic energy injection/drain (e.g., from line-search divergence or wrong sign in scatter) | Relative `< 5%` per frame |
| **Contact count vs frame** | Contact detection alignment; ordering vs cushion vs primitive-shape SDF bugs | Exact match expected. Any difference is a bug |
| **Bounding-box diagonal** | "Did the geometry explode locally" vs "did it stay bounded" | Relative `< 5%` per frame |
| **Per-particle drift histogram** (per frame, all particles) | Distribution of drift (median, p90, p99, max) | Most particles `< Tier-2 threshold`; long tail acceptable |

### Diagnostic-driven action table

| Observation | Likely cause | Action |
|---|---|---|
| Tier 1 fails + Tier 3 KE diverges fast | Energy injection (sign flip in scatter or grad) | Re-audit NH/ARAP scatter formula |
| Tier 1 passes + Tier 2 misses + Tier 3 COM aligned | Local outlier particles; Tier-2 max is dominated by one corner | Acceptable; document |
| Tier 1 passes + Tier 2 misses + Tier 3 COM diverges | Systematic drift, possibly from λ-permutation or pene0 ordering | Investigate contact detection / lexsort |
| Tier 3 contact count differs | Collision detection cushion / SDF / mesh-query divergence | Re-audit per-shape (Y.1, Y.2 already queued; check others) |
| Tier 3 KE small but BBox diverges | Local explosion (NH compression basin) | Investigate LBFGS convergence on outlier elements |
| Tier 1 fails AND Tier 3 KE stable | Algorithm correct, demo behavior wrong (e.g., wrong pin trajectory) | Re-audit scene config: PIN_AVEL units, mesh file, etc. |

## Implementation

Phase 3 verification script per demo:

```python
def verify_demo(fba_traj: np.ndarray, realsim_traj: np.ndarray, demo_id: str) -> dict:
    """
    Returns dict with keys:
      tier1 (bool):    binary pass/fail
      tier1_failures:  list of failed checks
      tier2 (bool):    drift threshold pass/fail
      tier2_max:       observed max drift
      tier2_target:    expected threshold
      tier3:           dict of diagnostic series (COM drift, KE rel-diff, contact-count, bbox-diff, drift histogram)
      verdict:         "GREEN" | "YELLOW (Tier-2 miss, investigate)" | "RED (Tier-1 fail)"
    """
    ...
```

Verification outputs go to `scripts/contact_demos_out/demo<N>/verification_report.json` + a snapshot strip PNG.

## Cross-reference

- Plan: `docs/superpowers/plans/2026-05-17-cudatests-realsim-parity.md` — Resolved decision #6 (originally "1e-3 m initial; relax to 1e-2 if FP order accumulates") is superseded by this doc.
- Memory: `/home/ziqiu/.claude/projects/-home-ziqiu-work/memory/realsim-port-success-binary.md` — the "binary success" framing remains; this doc operationalizes it as Tier 1.
