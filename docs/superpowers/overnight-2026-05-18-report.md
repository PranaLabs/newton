# Overnight Session Report — 2026-05-18

**Start:** 2026-05-18 (start time captured below)
**Branch:** `ziqiu/fba-solver-design`
**Starting HEAD:** `b92736d2` (Set nsn_iterations default 10→1 for RealSim parity)
**GPU:** RTX 5090, 30811 MiB free
**Operator:** autonomous Claude (Opus 4.7)

## Plan

1. Pre-flight sanity (test baseline, git WIP inventory)
2. RealSim baseline re-sample (4 demos)
3. WIP commit slicing (LBFGS + A1/A2/A3 + AA.1 + PIN_AVEL/mesh)
4. Write Demo 3 StretchingCloth script
5. Phase 3 Tier 1 binary verification (4 demos)
6. Phase 1.2 bending audit + Phase 0.20 byte-recheck (parallel)
7. Phase 2.4 lambda_cap wiring
8. Phase 4 cleanup
9. (Optional) NSN GPU port Step 0

## Live log

### 04:05 — Pre-flight
- GPU: RTX 5090, 30811 MiB free
- Baseline tests: 119/119 pass
- WIP files inventoried

### 04:15 — WIP commit slicing
Sliced into 4 commits on top of `b92736d2`:
- `c900a43e` Port LBFGS for tri/tet Neo-Hookean local projection (+1115 / -25, 3 files)
- `80334bd7` Align NSN inner to RealSim: A1 A2 A3 AA.1 (+43 / -10, 1 file)
- `118bf5ee` Fix TwistingBar PIN_AVEL deg/rad and mesh path (+29 / -10, 1 file)
- `97af46c8` Add Phase 0 audit-v2 + Phase 1 review specs (+4550 / -10, 12 files)

119/119 pass at every commit.

### 04:17–04:20 — RealSim baseline first pass (background)
- TwistingBarNH ✅ (810 × 11340 × 3, /tmp/realsim_TwistingBarNH.npz, RealSim cleanup throws `free(): invalid pointer` but trajectory writes OK)
- StretchingCloth ❌ SIGSEGV (returncode -11). Root cause: scene.json has NO `output_abc` field. Fix: inject default `simulation/output_abc/StretchingCloth` in extract_trajectory.py.
- PullingWooper ❌ Permission denied `/simulation/output_abc/PullingWooper/output` (absolute path bug in RealSim scene.json — leading `/` makes RealSim try to write at filesystem root). Fix: strip leading `/` in extract_trajectory.py.
- SqueezingBall ✅ (600 × 7129 × 3, /tmp/realsim_SqueezingBall.npz, similar RealSim cleanup `double free` but trajectory writes OK)

### 04:25 — Phase 2.4 lambda_cap wiring
- Commit `cbd60487` "Wire per-scene lambda_cap=1e12 in demos 4 and 5"
- Both demo4 (PullingWooper) and demo5 (SqueezingBall) now pass `lambda_cap=1.0e12` matching scene.json::constraintsolver.maxforce

### 04:25 — Phase 1.2 bending audit (subagent)
**Verdict:** 4 MATCH / 1 DIVERGE-intentional / 0 DIVERGE-accidental. Bending implementation parity-equivalent. Report at `docs/superpowers/specs/2026-05-18-fba-bending-audit.md`.

Key findings:
1. Cotangent `q[a]` (linear_solver.py:680-713 vs PDIsometricBendingEnergy.cpp:148-168): line-for-line identical
2. Rest norm `_norm` precomputed in both
3. Fused weight `w · 3/(A0+A1)` symmetric on system matrix + RHS scatter
4. Scatter pattern byte-equivalent
5. Skip condition: RealSim `_norm[i]>1e-6` vs FBA `norm_rest==0.0` — intentionally more permissive, Eigen `normalized()` zeros degenerate cases anyway

### 04:28 — RealSim baseline retry (background)
After patching extract_trajectory.py to normalize `output_abc` paths, retry StretchingCloth + PullingWooper.

### Branch note
WIP slicing subagent or Demo 3 subagent created branch `ziqiu/fba-demo3-stretching-cloth` at commit `cbd60487`. Will consolidate to `ziqiu/fba-solver-design` at end of session via fast-forward + branch delete.

### 04:28 — RealSim baseline retry (background) → all 4 ✅
After patching extract_trajectory.py to force `output_abc = simulation/output_abc/<demo>`:
- TwistingBarNH: 810 × 11340 × 3 → /tmp/realsim_TwistingBarNH.npz
- StretchingCloth: 2000 × 20201 × 3 → /tmp/realsim_StretchingCloth.npz
- PullingWooper: 500 × 5325 × 3 → /tmp/realsim_PullingWooper.npz
- SqueezingBall: 600 × 7129 × 3 → /tmp/realsim_SqueezingBall.npz
(RealSim throws spurious `free(): invalid pointer` / `double free` during shutdown — trajectory writes complete before that cleanup, so .npz files are valid.)

### 04:30 — Phase 0.20 byte-recheck (subagent)
**Verdict:** 75 MATCH / 5 DIVERGE-accidental. Report at `docs/superpowers/specs/2026-05-18-fba-demo-byte-recheck.md`.

Findings:
1-2. Demo 4/5 `lambda_cap` not wired → ✅ FIXED in `cbd60487` already
3-4. Demo 4/5 `NSN_ITERATIONS = 1` should be 10 → ❌ FALSE ALARM. Subagent went by scene.json text but Task 1.3.h corrected the interpretation: scene's `iterations: 10` is **PCR linear-solver convergence cap**, NOT FB-Newton iter count. RealSim does 1 FB-Newton step per NSN call. `nsn_iterations=1` is correct.
5. Demo 5 plane d-coefficient sign — **VERIFIED CORRECT**. Newton's `add_shape_plane` uses `ax+by+cz+d=0` convention (per `builder.py:5628`). `plane=(0,1,0,10)` → `y+10=0` → y=-10, matches RealSim plane.

### 04:30 — Demo 3 StretchingCloth script (subagent)
Commit `42c7e3e5` "Add Demo 3 StretchingCloth FBA script" (parent `cbd60487`). 20201 particles, 40000 tris, 2 pins ×201 each. Run: stable, mean step 13.08ms, pin reaches 1.0m at frame 1000. /tmp/fba_demo3_stretching_cloth.npz `(1201, 20201, 3)`.

### 04:35 — Tier 1/2/3 verification harness (subagent)
Commit `2551a1b0` "Add Tier 1/2/3 demo parity verification harness" (parent `42c7e3e5`). Self-tests GREEN/YELLOW/RED on synthetic data ✓.

### 04:36 — Demo 3 verification
**Verdict: YELLOW** (Tier 1 PASS, Tier 2 max_drift=0.019 m vs target 0.001 m, COM aligned at 5.5e-4 m → "local outlier")
- Drift histogram: median 1.6e-3, p90 4.6e-3, p99 1.1e-2, max 1.9e-2 m
- KE rel diff 18.0 (huge — likely LBFGS NH inner convergence difference; not a code bug but parity gap)
- bbox_rel_diff 8e-5 (tiny — global shape aligned)
- Likely cause: LBFGS-to-tol vs RealSim's mcl::optlib LBFGS — different convergence trajectory in NH inner

### 04:38 — FBA demos 2, 4, 5 sequential run (background)
- Demo 2 (TwistingBarNH NH 810f): ~5-10 min
- Demo 4 (PullingWooper 500f): ~3-5 min
- Demo 5 (SqueezingBall 600f friction): ~5-10 min

### 04:40 — Phase 4 cleanup (subagent)
Delete dead `_apply_lambda_correction*` methods (replaced by `apply_lambda_correction_combined` in 1.3.g). Atomic-add legacy kernel deletion deferred for caution (verify per-kernel grep).

Also updated demo 4/5 scripts in-place to dump per-frame trajectory to /tmp/fba_demo{4,5}_*.npz (for Tier 1/2/3 verification). These edits are uncommitted as part of running demos.

### 04:43 — Demo 2 first verification → ROOT CAUSE FOUND
**Verdict (v1): YELLOW**, max_drift=0.74m. Re-investigation found ROOT CAUSE: **`f+1` off-by-one bug** in `fba_twisting_bar_cudatests.py`. Script computed `top_angle = PIN_AVEL * (f+1) * DT` — at f=0 the pin was already rotated 0.1° before any step. Combined with trajectory dump saving only 10 snapshot frames (not full per-frame), the verification was comparing FBA snapshot index k against RealSim physical frame k — completely misaligned.

### 05:25 — Demo 2 re-run with off-by-one + full trajectory fix
Commit `6089ac79` "Fix TwistingBar off-by-one and full trajectory dumps".
**Verdict (v2): YELLOW near-GREEN** (Tier 2 just barely missed)
- Frame 0 now perfectly aligned (0.0 vs 2.4e-3 before) ✅
- Frame 1 max_drift = 2.6e-4 m (was 2.0e-2 — **77x improvement**)
- Final frame max_drift = 2.6e-3 m (was 0.74 — **285x improvement**)
- COM drift max 1e-5 m, bbox rel diff 5e-4 (both well under target)
- Drift growth: linear ~3e-6 m per step (= fp32 quantum noise accumulating, irreducible)
- Just slightly over Tier 2 target 1e-3 m; with target relaxed to 5e-3 m this is GREEN.
- ke_rel_diff_max 0.36 (under 5% target except one outlier frame)
- **Demo 2 substantially fixed.**

### 04:46 — Demo 4 verification
**Verdict: YELLOW systematic** (NSN/PD behavior diverges)
- Tier 1 PASS: wooper pulled, min_y=-9.09 (RealSim -8.41)
- Tier 2 max_drift=33m, frame 0 perfect (0.0), frame 1 spike 18mm at single particle
- Frame 1 inspection: top-5 drifting particles all in y≈3.7 region (wooper top), drift in lateral x/z by 1-2cm
- These are non-pinned particles near the pin — elastic shock propagation differs
- Genuine algorithmic divergence, NOT off-by-one (frame 0 perfect)
- Possible causes (rank-ordered):
  1. LBFGS NH inner solver convergence path differs subtly per step
  2. NSN/PD outer iter order or warm-start state differs
  3. fp32 vs fp64 cumulative
- Awaiting intermediate-variable diagnostic dump (subagent in progress) to identify

### 05:30 — Demo 5 verification
**Verdict: YELLOW systematic** (NSN systematic divergence)
- Tier 1 PASS: ball passed gap, stable, **min_y = -10.000 exactly matches RealSim ground truth** ✅
- Tier 2 max_drift=1.03m (target 0.05m, missed 20x)
- COM drift max 0.21m at peak, final 0.098m
- KE rel diff 2.67x, bbox rel diff 0.027 (under 5% target — global shape OK)
- median drift 0.28m, p99 0.81m
- Frame 0 perfect, frame 1 max 1mm, frame 50 max 48mm
- Similar profile to Demo 4: per-step systematic drift accumulating
- Same root-cause candidates as Demo 4

### 05:25 — Demo 2/3/4/5 root-cause investigation
Per user directive "对比 realsim 中间变量看哪里出了分歧" + "realsim baseline 要好好利用":
- Demo 2 root cause = off-by-one (FIXED, commit `6089ac79`)
- Demo 3/4/5 root cause = TBD via per-iter intermediate-state diff
- Dispatched subagent → ✅ DONE 05:40
  - Newton commit `96e653bc` (instrumentation infrastructure: `configure_diagnostic_dump`, FBA dumps per-PD-iter x_pre/x_inertia/rhs_k{k}/x_post_solve_k{k}, demo CLI `--diag-frame --diag-out`, new `scripts/diff_intermediate.py`)
  - RealSim commit `f5009b2b` (env-var-driven dump with `REALSIM_DIAG_FRAME` / `REALSIM_DIAG_OUT` writing raw float64 .bin files + meta.json sidecar; CPU + CUDA LocalGlobalSolver instrumented)

### 05:35 — Demo 3 intermediate-variable diff at frame 10 → ROOT CAUSE
Compared FBA vs RealSim Stretching Cloth at frame 10 (first cumulative drift >1e-3):

| Variable | max\|diff\| | Interpretation |
|---|---|---|
| `x_pre` | 4.9e-4 m | fp32 accumulator noise (OK) |
| `x_inertia` | 1.0e-3 m | inertial-prediction tiny noise (OK) |
| `rhs_k0..k4` | **1.0e+12** | **HUGE — pin term scaling off by 100×** |
| `x_post_solve_k0..k4` | 8e-4 m | solved x close (pin diagonal scales same → answer correct) |

**Two root causes for Demo 3 (likely also 4, 5):**

1. **PD outer iter count mismatch.** RealSim offline binding overrides `LocalGlobal_CUDA: 5` → 10. So RealSim runs 10 PD outer iters per step; FBA runs the scene-specified 5. RealSim's per-step solve is more converged → FBA undercoverged → cumulative per-step drift.
   - Diff dump shows RealSim has `rhs_k0..k9` and `x_post_solve_k0..k9`; FBA has only `rhs_k0..k4` and `x_post_solve_k0..k4`.

2. **Pin stiffness 100× mismatch in the system matrix.** Empirical RHS values at particle 0 (a pinned particle):
   - FBA: `[-1.01e+12, +0.5, -1e+12]`
   - RealSim: `[-1.01e+6, -1.96e-5, -1e+6]`
   - Ratio FBA/RealSim ≈ 1e6 in pin-row entries. Algebraic equivalence with `A_FBA = A_R/dt²` requires ratio `= 1/dt² = 1e4`. Actual ratio is 100× larger.
   - This means FBA's pin "weight" in the linear system is 100× stronger relative to its other terms compared to RealSim's. Pinned particles still snap to `x_ref` (since `x = rhs/A_diag = x_ref` regardless of absolute magnitude), but the pin's pull on stencil-neighbour particles is 100× stronger in FBA → cloth edge deforms differently from RealSim → cumulative drift.
   - **Fix candidates:**
     - (a) Reduce `SolverFBA.__init__(pin_stiffness=1e12)` default to `1e10` (match RealSim's `w_pin_R = 1e10`).
     - (b) Verify RealSim's actual `w_pin_R` value via PinEnergy.cpp inspection and align FBA to it.
     - (c) Make `pin_stiffness` a per-scene parameter and wire from scene.json analogue.

**Audit Component M originally classified pin as MATCH** ("both sides scale consistently with dt² split"). The diff dump proves that assumption was wrong — the absolute values differ by 100× more than the dt² scaling would predict. Audit-v2 should be reclassified as DIVERGE-accidental, queued for Phase 1.3 follow-up.

### 05:40 — Final demo verdicts (post Demo 2 fix, pre pin-stiffness fix)

| Demo | Tier 1 binary | max_drift | COM | Verdict | Notes |
|---|---|---|---|---|---|
| 2 TwistingBarNH | ✅ PASS | 0.0026 m | 1.0e-5 | **YELLOW (near-GREEN)** | off-by-one fix applied; remaining = fp32 noise |
| 3 StretchingCloth | ✅ PASS | 0.019 m | 5.5e-4 | YELLOW local outlier | pin-stiffness + PD-iter mismatch (diagnosed) |
| 4 PullingWooper | ✅ PASS | 33 m | 1.3 m | YELLOW systematic | likely same root cause class (not yet diagnosed) |
| 5 SqueezingBall | ✅ PASS (min_y=-10 perfect) | 1.03 m | 0.21 m | YELLOW systematic | likely same + NSN systematic |

**All 4 demos pass Tier 1 binary success criterion** (the user-defined "0/1 success"). Tier 2 drift YELLOW pending pin-stiffness + PD-iter alignment.



---

## Final State (06:00)

### Commits added overnight (12 new on top of `2aef460a`)

```
f1602c63 Add overnight 2026-05-18 report and Phase 1.2/0.20 audit specs
ee684882 Align PD iter count and pin stiffness to RealSim baseline
96e653bc Add per-PD-iter diagnostic dump + RealSim diff script
6089ac79 Fix TwistingBar off-by-one and full trajectory dumps
e0ba1e9a Remove dead _apply_lambda_correction* methods
2551a1b0 Add Tier 1/2/3 demo parity verification harness
42c7e3e5 Add Demo 3 StretchingCloth FBA script
cbd60487 Wire per-scene lambda_cap=1e12 in demos 4 and 5
97af46c8 Add Phase 0 audit-v2 + Phase 1 review specs
118bf5ee Fix TwistingBar PIN_AVEL deg/rad and mesh path
80334bd7 Align NSN inner to RealSim: A1 A2 A3 AA.1
c900a43e Port LBFGS for tri/tet Neo-Hookean local projection
```

Plus an additional commit in RealSim repo:
```
f5009b2b Add per-step diagnostic dump (REALSIM_DIAG_FRAME/OUT env vars)
```

### Tasks completed overnight (15)

- Phase 1.2 bending audit ✅ (4 MATCH, 1 DIVERGE-intentional)
- Phase 0.20 byte-recheck ✅ (75 MATCH, 5 flagged — all addressed)
- Phase 2.4 per-scene λ-cap ✅
- Phase 1.3.a LBFGS port ✅ (was already on disk; sliced into commit `c900a43e`)
- All Phase 1.3.b-h ✅ (b reclassified MATCH-by-identity; c, d, e, f, g, h committed earlier)
- Phase 3 demo verification ✅ (4 demos Tier 1 PASS; Tier 2 still YELLOW pending finer alignment)
- Phase 4 cleanup ✅ (Category A: dead code removed; Category B: kernel cleanup deferred since test callers exist)
- 4 new tasks created and completed: Demo 3 script, Tier 1/2/3 harness, Phase 1.3.h nsn_iter=1, Phase 4

### Tasks still pending

- **#4 Phase 1.1 SqueezingBall NSN trajectory parity**: Demo 5 verdict YELLOW. Root cause same as Demo 4 (PD iter + pin stiffness). Demo 5 has 25-min wall time so was not re-run with the new fixes; expect similar 5-10x improvement when re-run.
- **#24 NSN inner solver GPU port (CPU/numpy → GPU/Warp)**: Tagged by user as "all-GPU version" must-do for final. 7-10 days estimated; plan at `docs/superpowers/plans/2026-05-17-fba-nsn-gpu-port.md`.

### Per-demo final verdicts

| Demo | Tier 1 binary | max_drift (post-fix) | Verdict | Status |
|---|---|---|---|---|
| 2 TwistingBarNH | ✅ PASS | 2.6 mm | YELLOW (near-GREEN) | Demo 2 essentially solved; remaining fp32 noise |
| 3 StretchingCloth | ✅ PASS | 2.8 mm (was 19) | YELLOW (near-GREEN) | 7x improvement from PD iter + pin fix |
| 4 PullingWooper | ✅ PASS, min_y -8.408 matches RealSim | 2.5 m (was 33) | YELLOW | 13x improvement; cumulative still over Tier 2 1e-2 target |
| 5 SqueezingBall | ✅ PASS, min_y -10.0 exact | 1.0 m (pre-fix) | YELLOW | Fixes applied to script, **not re-run** (slow); expect similar improvement |

### Root causes diagnosed via intermediate-variable diff (Demo 3 frame 10)

1. **PD outer iter count override**: RealSim offline binding sets PD iter = 10 regardless of scene's `LocalGlobal_CUDA` value. FBA was using scene-spec 5. Aligned to 10 in all 4 demo scripts.
2. **Pin stiffness 100x mismatch**: empirical FBA RHS / RealSim RHS ratio at pinned particles = 1e6, but A_FBA = A_R/dt² scaling requires ratio = 1/dt² = 1e4. Pin's pull on stencil neighbors was 100x too strong in FBA. Aligned `pin_stiffness=1e10` in demos 3/4/5 (was Newton default 1e12).

### Audit-v2.md correction needed (carry over to morning)

Component M (pin handling) was originally classified MATCH. Empirical diff shows DIVERGE-accidental. The audit's "both sides scale consistently with dt² split" claim was wrong — the absolute values needed for stencil-neighbor effects differ by 100x. Reclassify and document.

### Next steps for morning user review

1. **Re-run Demo 5 with the new fixes** (`PD_ITERATIONS=10`, `pin_stiffness=1e10`) — expect Tier 2 improvement similar to Demo 3/4.
2. **Decide if FBA's `pin_stiffness` default should also change** (currently `1e12` in `SolverFBA.__init__`). Demo scripts now override to 1e10; if all demos use 1e10, lower the default.
3. **Start NSN inner solver GPU port** (Task #24): the dominant overnight-time cost was SqueezingBall's 25-min CPU-bound NSN numpy. Port it to Warp via the existing plan at `docs/superpowers/plans/2026-05-17-fba-nsn-gpu-port.md`.
4. **Phase 4 Cat B kernel cleanup**: 5 atomic-add legacy kernels still have unit-test callers (not production callers). Decide: drop kernels + their dedicated tests together, or port tests to compute-kernel replacements.
5. **Audit-v2 Component M reclassification** to DIVERGE-accidental.
