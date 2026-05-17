# Full RealSim Computational Parity for SolverFBA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

---

## For a fresh Claude session: orientation

If you are a fresh session with no prior conversation context, read this section first.

### What this project is

Newton (NVIDIA's open-source physics engine) is getting a new solver called `SolverFBA` ("Fast But Accurate") — a Warp-native port of RealSim's projective-dynamics + non-smooth-Newton solver. The user is the author of RealSim. The port aims to reproduce every demo in RealSim's CudaTests suite with computationally-identical behavior, so that the same scene produces the same trajectory (within FP noise) in both engines.

### Repos and paths

- **Newton repo (this codebase)**: `/home/ziqiu/work/newton` on branch `ziqiu/fba-solver-design`
- **RealSim repo (reference)**: `/home/ziqiu/work/RealSim_py/realsim_py` — source of truth for every algorithm
- **RealSim binary**: `/home/ziqiu/work/RealSim_py/realsim_py/build/bin/RealSim`
- **CudaTests scene configs**: `/home/ziqiu/work/RealSim_py/realsim_py/simulation/config/CudaTests/{TwistingBar,TwistingBarNH,...,CableGrabRaptor}/scene.json`
- **RealSim ground-truth trajectories** (re-extractable from the .abc files): produced by running the RealSim binary in offline mode (see memory `realsim-test-mode`); driver at `scripts/realsim_baseline/run_one.py`

### Newton's repo conventions (CRITICAL — see `AGENTS.md`)

- `newton/_src/` is internal; only `newton/tests/` and `newton/_src/*` may import from it
- PEP 604 unions (`X | None`), Google-style docstrings, types in annotations
- `wp.array[X]` bracket syntax for Warp arrays
- **`unittest` not pytest** for tests
- Imperative commit subjects ~50 chars
- SPDX year = file creation year (don't update when modifying)
- `uvx pre-commit run -a` must be clean before every commit
- No `wp.synchronize_device()` before `.numpy()` (redundant)
- Always run `git status -uall`-free commands (i.e. plain `git status`); the user's tree has many untracked artifacts (plans, npz, png) that are intentional

### Critical memories to consult

These persist across sessions, in `/home/ziqiu/.claude/projects/-home-ziqiu-work/memory/`:

- `realsim-port-discipline.md` — **READ FIRST**: trust RealSim, suspect port bugs first. Past failures (PGS-as-NSN, simplified-FB-Newton, wrong bending scatter) all came from violating this.
- `realsim-test-mode.md` — how to invoke RealSim binary in offline mode (NOT a `--offline` flag; inject `"offline": true` + `"maxFrame": N` into the scene.json).

### Current state at plan kickoff (HEAD = `7948ffe5`)

Run `git log --oneline -25` for the full commit history. Highlights:

- **104/104 FBA tests pass** (`uv run --extra dev -m newton.tests -k test_solver_fba`)
- PullingWooper demo runs stably: `min_y ≈ -8.41, pulled = 10`, `mean_ms ≈ 363` (slow vs P1's 10ms because NSN is now full FB-Newton; perf is acceptable trade-off for correctness)
- SqueezingBall demo: deterministic but trajectory diverges from RealSim — `min_y = -5.68` vs RealSim's `-10.02`. This is the lead-edge investigation in Phase 1.
- Phase 0 (baselines), Phase 1 (PullingWooper isodof Schur), Phase 2 (kinematic friction), P2-D (deterministic PD), P2-E (FB-Newton NSN port) all merged. See `docs/superpowers/fba-dev-progress.md` for the journey.

### What "byte-identical trajectory parity" means here

The user wants every CudaTests demo's particle positions to match RealSim's `output_obj_0.abc` reference at every frame within ≤ 1e-3 m (per the Resolved decisions). This is a high bar — much tighter than the original "qualitative parity" spec from P0. To meet it, every algorithmic block in `SolverFBA` must replicate RealSim's exact formula, sign convention, data layout, and iteration scheme. Where RealSim has a known bug (Corotational Eigen `.trace()`, Coulomb rectangular cone), the Resolved decisions table specifies whether to reproduce or stay correct.

### Working style expected

- **Subagent-driven**: per the user's prior P1/P2 cadence, each task gets a fresh subagent (via `superpowers:subagent-driven-development`); review between tasks. Don't run the whole plan in one session's context.
- **Surface every algorithmic substitution**: if you find yourself thinking "I'll do X instead of RealSim's Y because Z", STOP, write the substitution into the task report explicitly, and let the user accept/reject before implementing.
- **Frame-by-frame validation**: don't ship a demo with only visual snapshot inspection. Always diff trajectory against RealSim's abc.
- **Pre-commit clean**: `uvx pre-commit run -a` before every commit. The repo has formatters / typo checks / Warp-syntax checks that catch sloppy commits.

---



**Goal:** Make every FBA-side CudaTests demo computationally identical to its RealSim binary counterpart, frame-by-frame within FP-epsilon (≤ 1e-3 m per particle position). Each algorithmic block in `SolverFBA` must reference and replicate the exact RealSim source-line semantics.

**Context for new readers:** Prior phases (P0–P2-E) shipped a working but partially-divergent port. Documented and undocumented substitutions accumulated:
- Stage A/B inner solver was implemented as projected-Gauss-Seidel under the name "NSN" (silent substitution; identified post-hoc, corrected in commit `7948ffe5` to true FB-Newton)
- Bending kernel used the wrong scatter formula for non-flat rest (corrected in `cfe3f11e`)
- Mass lumping is FE-standard in FBA vs uniform `total_mass/N` in RealSim (intentional, see decisions below)
- Corotational uses correct symmetric trace in FBA vs Eigen `.trace()` bug in RealSim (intentional)
- Coulomb cone shape: FBA is circular `|λ_t|₂ ≤ μλ_n`; RealSim is rectangular box `|λ_t1|, |λ_t2| ≤ μλ_n` (intentional)

**This plan** starts from HEAD `7948ffe5` and progresses toward computational parity. Every fix references a specific RealSim source file:line. No naming-mismatched substitutions allowed (e.g. method `_solve_nsn_*` MUST contain NSN, not PGS).

## Debugging discipline (mandatory)

**RealSim 的 CudaTests demo 已经被跑过成百上千次、经历过检验是稳定的**。同样的计算逻辑 1:1 搬到 Newton-FBA 必然能跑通（noise behavior 大致一致）。

**因此当一个 demo 在 FBA 上行为偏离 RealSim 时，怀疑顺序严格按以下：**

1. **第一怀疑**：我的移植里有 bug（sign 错、formula 抄错、algorithm 替换错、data 布局错）
2. **第二怀疑**：我的移植里有 bug
3. **第三怀疑**：我的移植里有 bug
4. **第四怀疑**（极少）：RealSim 真的有未发现的 bug

**禁用的推理路径**：
- ❌ "RealSim 的公式从物理直觉看反了，所以我用'正确'版本" — 这是把自己的物理偏见凌驾于 validated 实现之上
- ❌ "RealSim 用 PGS/MinimumMap/A，FBA 用 B 应该等价" — 算法等价性必须通过 frame-by-frame 数值验证证明，不能假设
- ❌ "把代码命名为 `_solve_nsn_*` 然后实现 PGS，因为 LCP solver 是 LCP solver" — naming 必须严格匹配实际算法
- ❌ "demo 视觉上看着像 RealSim、所以移植对了" — 这是 verification bias

**强制的验证路径**：
- 每个计算块都要逐行对照 RealSim 源码（file:line），不能"大致按 RealSim 的思路"
- 替换 RealSim 公式的地方必须在 spec/plan/commit message 显式说明 "我选了 X 不是 RealSim 的 Y，因为 Z"，让 reviewer 能拒绝
- demo 失败 → frame-by-frame 数值 diff 找第一帧分歧点 → 在那一帧 dump 每个 row 的 NSN 状态 → 与 RealSim 同帧 hand-trace 对比
- 历史上每次"以为 RealSim 错"事后看都是 FBA 实现错（参见 P2-E v1 的失败、SqueezingBall 的 anchor sign 失败）

**对已知 RealSim bug 的特殊处理**（在 Resolved decisions 表里逐项列出）：必须明确文档化、并量化预期的 trajectory delta。

## Resolved decisions (locked at plan kickoff, 2026-05-17)

User-confirmed alignment policy: **byte alignment with RealSim except where the divergence is a known-correctness bug in RealSim** (e.g. Eigen `.trace()` on 2D/3D Vec returns only x[0]). Per item:

| # | Item | Direction | Rationale |
|---|---|---|---|
| 1 | Corotational `.trace()` bug (tri + tet) | **Keep Newton's correct symmetric formula** | RealSim bug; Newton stays correct. Documented as deliberate divergence; affects only Corot demos. |
| 2 | Coulomb cone shape | **Switch to RealSim's rectangular box** `|λ_t1|, |λ_t2| ≤ μ·λ_n` per axis | RealSim's `boundConstraintForces` convention. Affects SqueezingBall etc. |
| 3 | Mass lumping | **Switch to RealSim's uniform** `mass/N` per particle | Affects all demos with non-uniform meshes (basically all). |
| 4 | NSN inner iter default | **Switch to RealSim's cap = 10** | Was 1 (P1's perf-driven choice); RealSim caps at 10. |
| 5 | λ-cap default | **Switch to per-scene values** (`maxforce` from scene.json: 1e12 default, 100 for ClothOnKnives/SharpCorner) | Mirror RealSim's per-config setup. |
| 6 | Drift tolerance | **1e-3 m initial; relax to 1e-2 if FP order accumulates** | Tight initial target, validate on Demo 1 first. |
| 7 | Plan scope | **All 10 CudaTests demos** | User-confirmed. Includes new infra: multi-env (P9), cable element + soft-soft (P10). |


**Newton code conventions** (per `AGENTS.md`, enforced for every commit):
- `newton/_src/` is internal; examples and tests use public modules only
- PEP 604 unions (`X | None`), Google-style docstrings, types in annotations
- `wp.array[X]` bracket syntax
- `unittest` not pytest
- Imperative commits ~50 chars
- SPDX year = file creation year (no updates on modify)
- No `wp.synchronize_device()` before `.numpy()` (redundant — the copy syncs)
- `uvx pre-commit run -a` clean before every commit

---

## Pre-flight checks

- [ ] Branch `ziqiu/fba-solver-design`, HEAD = `7948ffe5`.
- [ ] Working tree clean (`git status --short` excluding documented untracked artifacts).
- [ ] All 104 FBA tests pass: `uv run --extra dev -m newton.tests -k test_solver_fba`.
- [ ] PullingWooper demo stable: `uv run python scripts/fba_demo4_pulling_wooper.py | tail -3`. Expected `min_y ≈ -8.41, pulled=10`.

---

## Phase 0 — Audit refresh

### Task 0.1: Re-run the full FBA-vs-RealSim audit

Re-run Task Q's audit methodology (see `docs/superpowers/plans/2026-05-16-cudatests-p0p1-wooper.md` end-of-document) with the current HEAD. Produce an updated table for these components:

| ID | Component | RealSim file:line | FBA file:line | Verdict (MATCH / DIVERGE-intentional / DIVERGE-accidental / UNKNOWN) | Notes |

**Components to audit** (each gets one row):
- A: PD outer loop control flow
- B: Inertial prediction (`compute_inertial_kernel`)
- C: Tri ARAP local projection
- D: Tri Corotational
- E: Tri Neo-Hookean
- F: Tet ARAP
- G: Tet Corotational
- H: Tet Neo-Hookean
- I: Isometric bending (post H' fix)
- J: RHS assembly
- K: Linear solve (Cholesky + sparse inverse)
- L: Velocity update
- M: Pin handling
- N: Mass lumping
- O: Damping
- P: Gravity
- Q: Tangent basis
- R: Coulomb cone shape (rectangular box vs circular)
- S: Contact Jacobian construction
- T: λ-cap / maxforce
- U: λ warm-start within step (post T' fix)
- V: NSN solver — must verify FB-Newton port (post P2-E, commit `7948ffe5`) actually matches RealSim's `_omega`, `_nonsmooth_compliance`, `_h`, `_pene0` semantics
- W: Position-LCP coupling — RealSim's `build()` does an in-iter `_systemlinearsolver->solve(_x, b_modified)`; FBA uses the algebraic identity `J·x_corrected = (pene0 - r) + W·λ`. Verify equivalence.
- X: Lambda correction: RealSim applies `Δq = dt²·A⁻¹·J^T·(ω·λ)`, FBA applies `A⁻¹·J^T·lam_apply` where `lam_apply = dt²·ω·λ`. Verify per-PD-outer accumulation matches.
- Y: `_pene0` semantics: RealSim friction `_pene0 = H_f·q_prev` (previous-step tangential position); FBA `pene0 = dot(t, world_anchor + dt·v_anchor)`. Verify these compute the same value.
- Z: Multi-environment (`parallelEnv`) — not yet supported.

**Output:** one markdown file `docs/superpowers/specs/2026-05-17-fba-realsim-audit-v2.md`. Each component cited with concrete file:line refs from BOTH codebases.

**Acceptance:** verdict for every component is one of MATCH / DIVERGE-intentional / DIVERGE-accidental / UNKNOWN. Zero "UNKNOWN" allowed.

### Task 0.2: Apply locked decisions (no new ambiguity)

All decision points from "Resolved decisions" above are locked. This task verifies they're implementable and ready for Phase 2:

1. Read RealSim source for each decision:
   - Box cone clamp: `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/lagrange/solvers/NonSmoothNewton.cpp:400-403`
   - Uniform mass: `/home/ziqiu/work/RealSim_py/realsim_py/src/.../Mass.cpp:12-21` (or wherever `addObjectMass` lives)
   - NSN iter cap: `simulation/config/CudaTests/*/scene.json: constraintsolver.iterations: 10`
   - λ-cap: `scene.json: constraintsolver.maxforce`
2. Confirm each value is consistent across all 10 CudaTests scene.jsons. If a demo has a different value, scope it as per-demo config (matched on the FBA side).
3. **Output**: a small `decisions.json` at `/home/ziqiu/work/newton/scripts/realsim_baseline/decisions.json` recording the per-demo config values, used by Phase 2 implementations.

### Task 0.3: Per-demo configuration audit

For each CudaTests demo (10 total), document on a single markdown table:

| Demo | RealSim scene.json fields | FBA demo script | Verdict |

Verify byte-by-byte that the FBA demo script (or its loader for those not yet written) reproduces every scene.json field that affects physics. Specifically: gravity, dt, stop/maxFrame, LocalGlobal iter count, constraint solver iter count + maxforce + linear solver type, all collision shapes (cylinder/plane/box/mesh), every cylinder rolling velocity, all pin actions (PULLING/ROLLING), every mechanical-prop (Young, Poisson, mass, constitutive, bending, bending_type).

For demos not yet written (P3 onward), this becomes the scene-loader spec.

---

## Phase 1 — Rectify accidental divergences

Each accidental divergence from Phase 0 audit gets a numbered task here. Pre-known items:

### Task 1.1: NSN trajectory-parity gap (SqueezingBall)

P2-E v2 (`7948ffe5`) ported NSN's algorithmic structure but SqueezingBall still drifts (FBA min_y=-5.68 vs RealSim -10.02). Likely root causes (rank-ordered, from prior diagnose):

1. **`_pene0` semantics**: RealSim friction `_pene0` uses previous-step position `H_f·q_prev`, NOT the current frame's anchor + v_anchor·dt. Investigate whether FBA's `_contact_tangent1_offset_h` (current-frame anchor + v_shift) is numerically equivalent.
2. **Position-LCP coupling**: RealSim's `build()` does `_systemLS.solve(_x, b + dt²·J^T·(ω·λ))` then `rhs = (1/dt²)·(h − ω·J·_x)`. FBA uses identity `J·_x = (pene0 - r) + W·λ`. Verify exactness when `b` is not constant across NSN iters (FBA recomputes `r` only at start of step, not within NSN iter).
3. **Per-row precond**: RealSim `precond_n = dt²·W_ii, precond_t = dt·W_ii`. P2-E v2 has this. Re-verify.
4. **Box vs cone clamp**: see Phase 0 Task 0.2 decision.

**Step 1.1.a**: write a debug script that, on PullingWooper (small, deterministic), captures every per-row NSN state (`omega`, `compliance`, `h`, `lam`, `pene0`) and a corresponding RealSim diagnostic build. Run frame 1 only. Compare row-by-row.

Implementation: dispatch a `general-purpose` agent with the spec "run FBA + RealSim with diagnostic print enabled, dump per-row NSN state at frame 1, diff." Output a table with per-row diff for each NSN variable.

**Step 1.1.b**: based on diff, identify the specific row/variable that diverges. Fix root cause. Re-run.

**Step 1.1.c**: acceptance: per-row NSN state at frame 1 matches within 1e-6, AND SqueezingBall final-frame `min_y ≤ -8`, `|drift| < 1.5`.

### Task 1.2: Bending edge weight + offsets

Audit `_compute_isometric_bending_q` against RealSim's `PDIsometricBendingEnergy::init`. Confirm cotangent weights, edge stencil ordering, and `_norm[i]` computation (added in H' fix). Cite file:line on both sides.

If divergence found: fix per RealSim's formula, add unit test that materializes a known stencil and verifies bit-identical to a hand-computed reference.

### Task 1.3: Other accidental divergences from Phase 0

One task per audit row marked `DIVERGE-accidental`. Each follows the pattern:
- Identify root cause (file:line on both sides)
- Write failing unit test that captures the disagreement at fixed inputs
- Fix
- Verify all FBA tests pass
- Commit with reference to the divergent RealSim file:line

---

## Phase 2 — Apply locked alignment decisions

Each task implements one locked decision from the top "Resolved decisions" table. Each is a separate commit with reference to RealSim source.

### Task 2.1: Switch Coulomb cone clamp to RealSim's box

P2-E v2's `_solve_nsn_coulomb` already does per-axis box clamp at its end (`solver_fba.py:1581+`). Verify; remove the no-longer-used `project_coulomb_cone` helper (`solver_fba.py:90`) and its caller chain if any remains. Update any test that was checking circular-cone behavior.

Acceptance: `grep "project_coulomb_cone" newton/_src/solvers/fba/` returns no results except the removed function's site (which is now empty).

### Task 2.2: Bump NSN iter default to 10

Change `nsn_iterations: int = 1` to `nsn_iterations: int = 10` in `SolverFBA.__init__`. Update the test that asserted `default == 1` (added in Task N). Remove the 3 local `nsn_iterations=10` overrides from P2-E v2 tests (`test_single_plane_contact_pushes_particle_away`, `test_multiple_plane_contacts_no_interpenetration`, `test_friction_disabled_matches_stage_a`) — they should pass with the new default.

Re-measure PullingWooper and SqueezingBall: expect ~10× perf cost. If PullingWooper exceeds 100ms (vs RealSim 24ms), revisit perf optimization separately. The target is correctness first.

### Task 2.3: Switch mass lumping to uniform `total_mass/N`

Add a path in FBA model construction (probably via `add_soft_mesh` / `add_cloth_grid` extension or per-solver override) that overwrites the particle-mass array with `total_mass / num_particles` instead of FE lumping. The simplest implementation: in each demo script, after `builder.finalize()`, do `model.particle_mass.assign(np.full(N, total_mass/N))`. Avoid modifying Newton's builder API.

Reference: RealSim `Mass.cpp:12-21` `addObjectMass`.

### Task 2.4: Switch λ-cap default to RealSim per-scene values

Update each demo script to pass `lambda_cap` matching its source scene.json's `maxforce`:
- TwistingBar/TwistingBarNH/StretchingCloth/PullingWooper/CrossingGingerbreadman/SqueezingBall: 1e12 (≈no cap)
- ClothOnKnives/SharpCorner: 100
- ParallelEnvTest: 1e10 (from its scene.json)
- CableGrabRaptor: 1e12

These are per-demo; not a `SolverFBA.__init__` default change. Just set in each demo's solver construction.

---

## Phase 3 — Per-demo trajectory verification

For each of the 10 CudaTests demos, this is the acceptance gate:

### Task 3.X (per demo)

1. **Run FBA demo to completion**, capturing per-frame particle positions to `/tmp/fba_<demo>.npz` (shape `(frames, particles, 3)`).
2. **Run RealSim binary in offline mode**, ensure `output_abc` is written, extract to `/tmp/realsim_<demo>.npz` (same shape).
3. **Compute per-frame max-particle position drift**: `drift[t] = max(|fba[t] - realsim[t]|_2 across particles)`.
4. **Acceptance**: `max(drift) < 1e-3` over all frames. If not met, return to Phase 1 with the failing frame as the diagnostic input.
5. **Output**: snapshot strip showing both FBA and RealSim side-by-side at frames [0, 25%, 50%, 75%, 100%] of the demo. Save to `scripts/contact_demos_out/demo<N>/comparison_strip.png`.
6. **Commit**: per-demo update to `scripts/contact_demos_out/README.md` and `docs/superpowers/fba-dev-progress.md` with the measured drift.

### Demo order (10 total — full CudaTests coverage)

| # | Demo | New infra needed | Notes |
|---|---|---|---|
| 1 | TwistingBar | none — already shipped | TET ARAP, ROLLING pin. Verify byte-identical post P2-D refactor. |
| 2 | TwistingBarNH | none — already shipped | TET NH, ROLLING pin. Corot diverges intentionally; NH should match. |
| 3 | StretchingCloth | none — already shipped | TRI ARAP, PULLING pin. |
| 4 | PullingWooper | none — already shipped | TET NH + 2 static cyl + PULLING. Mass lumping switch will affect; re-verify. |
| 5 | SqueezingBall | (already in tree) | The currently-failing case. Box-cone switch + uniform mass + nsn=10 should close gap. |
| 6 | CrossingGingerbreadman | scene-from-json loader | TET NH + 13 static cyl + PULLING. Largely existing infra. |
| 7 | SharpCorner | `add_shape_box` validated through FBA Stage A path; needs unit test | TRI ARAP cloth_5k + static box. Validates 5k cloth + primitive shape contact. |
| 8 | ClothOnKnives | `GeoType.MESH` (triangulated mesh) contact validated through FBA Stage A/B | TRI NH cloth_5k + static knives. Newton's `create_soft_contacts` supports MESH; verify FBA's `update_contacts` consumes it correctly. |
| 9 | ParallelEnvTest | **Multi-environment scaffold + LiteSchur sparse path** | 25 × 1013 TRI cloth + 25 spheres. RealSim's `LiteNonSmoothNewton_CUDA` (block-diagonal Schur) is the perf path; without it, dense W at 3M=~3·M_total = ~3000+ contacts is infeasible. See Phase 5. |
| 10 | CableGrabRaptor | **Bergou 1D rod element + soft-soft contact + dynamic pin sequencer + torus mesh** | 2 fingers + raptor_10k + cable + torus. Largest new-infra demo. See Phase 5. |

Demos 1–4 should be byte-identical after Phase 2 with minimal new work. Demos 5–8 are the integration gate for Phase 3. Demos 9–10 require Phase 5 infrastructure first.

---

## Phase 5 — New infrastructure for Demos 9 & 10

### Task 5.1: SharpCorner (Demo 7) — primitive box contact through FBA Stage A path

**Files**: `scripts/fba_demo7_sharp_corner.py` (new), unit test in `test_solver_fba.py`.

Newton's `add_shape_box(half_extents=...)` produces a `GeoType.BOX` primitive. `create_soft_contacts` (`newton/_src/geometry/kernels.py`) handles BOX via SDF query. Verify the contact normal sign, anchor position, and that FBA's `update_contacts` ingests them correctly for Stage A.

Acceptance: a unit test with a stationary particle near a box face produces the expected contact normal and offset. Then Demo 7 runs the cloth_5k scene; trajectory diff vs RealSim's abc passes.

### Task 5.2: ClothOnKnives (Demo 8) — `GeoType.MESH` contact path

**Files**: `scripts/fba_demo8_cloth_on_knives.py`, unit test for MESH contact.

Newton's `add_shape_mesh` produces a `GeoType.MESH`. `create_soft_contacts` supports MESH via `wp.mesh_query_point_sign_normal` (`geometry/kernels.py:1094`). Need to verify the soft_contact_* output (normal, depth, body_pos) is correctly populated and consumed by FBA's `update_contacts`.

Unit test: a small cloth grid above a single triangulated tetrahedron; verify the contact set after `pipeline.collide()` is sensible. Then Demo 8.

### Task 5.3: ParallelEnvTest (Demo 9) — multi-env scaffold

**Files**: `newton/_src/solvers/fba/solver_fba.py` (multi-env support), `newton/_src/solvers/fba/linear_solver.py` (LiteSchur path), `scripts/fba_demo9_parallel_env.py`.

This is a major piece. Two sub-tasks:

#### Task 5.3.a: Multi-env model construction

Build 25 sub-models with identical topology but per-env independent state. The simplest path: 25 separate `ModelBuilder` runs concatenated into one global model via Newton's `replicate` or by hand-stacking; with each env's particle range tracked.

Each env has its own PD factor. From Phase 2 (P6 spec) we chose NOT to share factors — each env can have different Young/bending parameters. **Update**: re-confirm; ParallelEnvTest config has uniform parameters across envs, so factor sharing would work. **Decide** at start of Task 5.3.a.

#### Task 5.3.b: LiteSchur sparse path

Implement RealSim's `LiteNonSmoothNewton_CUDA` analogue: when contacts span 25 envs and W is block-diagonal across envs, build a sparse BSR W instead of dense. Solve per-block (small dense Cholesky per env's Schur block).

Acceptance: 25-env scene runs, wall ≤ 25× single-env wall. Trajectory matches RealSim per Phase 3.

### Task 5.4: CableGrabRaptor (Demo 10) — cable + soft-soft + torus + dynamic pin

**Files**: `newton/_src/solvers/fba/kernels.py` (new Bergou cable kernels), `newton/_src/solvers/fba/solver_fba.py` (cable energy + soft-soft contact + dynamic pin), `scripts/fba_demo10_cable_grab_raptor.py`.

The largest new-infra piece. Split into:

#### Task 5.4.a: Bergou 1D elastic rod element

Add cable energy kernel: per cable segment, bending + stretch terms with `cableInvStiff` regularization. Cable is a sequence of N nodes; per-node bending stencil uses 3 consecutive nodes for curvature. Unit test: a straight cable deforms under a transverse pull, returns to rest. Acceptance: numerical agreement with a hand-derived Hessian on a 4-node sample.

#### Task 5.4.b: Torus collider

`torus.obj` is a triangulated mesh. Reuse Task 5.2's `add_shape_mesh` path. No new code if 5.2 lands first.

#### Task 5.4.c: Dynamic pin sequencer

Gripper config has 7 sequential `PULLING`/`ROLLING` actions with `maxlength`/`maxrotation` auto-step. Extend `set_pin_targets()` to consume an action queue:
```python
solver.set_pin_actions([
    {"action": "PULLING", "dir": (0, 0, 0), "vel": 1.0, "maxlength": 2.0},
    {"action": "PULLING", "dir": (0, 1, 0), "vel": 2.0, "maxlength": 10.0},
    {"action": "ROLLING", "center": (0,0,0), "axis": (0,1,0), "avel": 45.0, "maxrotation": 135.0},
    ...
])
```

The sequencer advances current pin position per step and transitions to the next action when the trigger is reached.

#### Task 5.4.d: Soft-soft contact

Extend FBA's Stage A/B Jacobian assembly to handle two-sided contacts: `J·x = [J_a | −J_b]·[x_a; x_b]`. The Schur block dimension doubles per contact. Re-verify W symmetry and the box clamp signs for both sides.

Acceptance: a 2-particle soft-soft collision test (two tiny tet cubes pushing into each other) gives symmetric impulses. Then Demo 10.

---

## Phase 4 — Code-organization cleanup

After Phase 3, the FBA solver code has accumulated legacy kernels (atomic-add variants kept around as "reference") and some duplication. Final pass:

### Task 4.1: Remove legacy kernels

Delete unused atomic-add scatter kernels (`project_stretching_arap_tet_kernel` etc.) replaced by the compute+gather pattern in P2-D. Run `grep "wp.atomic_add" newton/_src/solvers/fba/kernels.py` — expect 0 matches after cleanup.

### Task 4.2: Cone projection deduplication

If Task 2.1 settled on box clamp, remove `project_coulomb_cone` (unused) and any leftover circular-cone code paths.

### Task 4.3: Documentation refresh

Update `docs/superpowers/fba-dev-progress.md` with the complete commit list from this plan. Mark prior plans (P0+P1, P2-A through P2-E) as superseded where appropriate.

---

## Acceptance checklist (whole plan)

- [ ] Phase 0 audit table has zero UNKNOWN verdicts
- [ ] Every `DIVERGE-accidental` from Phase 0 has a corresponding fix commit in Phase 1
- [ ] Every `DIVERGE-intentional` is either rectified per user decision (Task 0.2) or annotated in `audit-v2.md`
- [ ] Demos 1–8 each pass `max(drift) < 1e-3` against RealSim's abc trajectory
- [ ] All FBA unit tests pass (current 104 + any new from Phase 1 fixes)
- [ ] `uvx pre-commit run -a` clean on all commits
- [ ] No `wp.atomic_add` in active FBA solver path (legacy kernels removed in Phase 4)
- [ ] No method named `_solve_nsn_*` containing PGS or other non-NSN algorithm (current implementation is FB-Newton per RealSim spec — verify)

---

## Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Per-row diff (Task 1.1.a) reveals fundamental algorithmic gap beyond NSN | Med | High | Surface to user; if user wants RealSim parity at any cost, port full position-LCP coupling including `_systemLS.solve` per build() |
| User chooses RealSim cone-box clamp + this regresses physical correctness expectations | Low | Med | Document the choice with reference to RealSim source; let physics-aware reviewers raise flags upstream |
| 1e-3 drift tolerance is too tight (FP order changes in W·λ products give ~1e-4 noise; cumulative over 600 frames could exceed) | Med | Med | Relax to 1e-2 if needed; first verify Demo 1–3 (no contact, less FP cumulative) hold to 1e-4 |
| Demo 7 (SharpCorner) static box contact untested through FBA Stage A | Med | Med | Treat as new infrastructure; add per-shape test before demo run |
| Demo 8 (ClothOnKnives) GeoType.MESH contact path untested in FBA | Med | Med | Same as above |

---

## Newton code-convention checklist (per commit)

Each commit produced under this plan must:

- [ ] Run `uvx pre-commit run -a` clean
- [ ] Use PEP 604 unions everywhere new code is added
- [ ] Use bracket syntax for Warp arrays (`wp.array[wp.vec3]` not `wp.array(dtype=...)`)
- [ ] Use `unittest` (not pytest); test names follow `test_<noun>_<verb_phrase>` pattern
- [ ] Imperative commit subject ~50 chars
- [ ] No SPDX year update on existing files
- [ ] Tests reference public API (`from newton.solvers import SolverFBA`); only `test_solver_fba.py` may import `newton._src.solvers.fba.solver_fba`
- [ ] Docstrings Google-style, types in annotation (not docstring), SI units annotated on physical quantities
- [ ] No `wp.synchronize_device()` before `.numpy()`

---

## Open questions (all answered at plan kickoff)

See "Resolved decisions" table at top of plan. All 7 prior open questions are locked.

---

## Estimated phasing & rough effort

| Phase | Effort | Output |
|---|---|---|
| Phase 0 (audit) | 0.5 day | audit-v2.md, decisions.json |
| Phase 1 (rectify accidental divergences) | 2-3 days | NSN trajectory parity, bending re-verify, etc. |
| Phase 2 (apply locked decisions) | 1 day | box cone, nsn=10, uniform mass, per-scene λ-cap |
| Phase 3 (per-demo verification, demos 1-6) | 1-2 days | 6 demos byte-aligned with RealSim |
| Phase 5.1-5.2 (Demo 7-8 mesh contact) | 2-3 days | SharpCorner + ClothOnKnives |
| Phase 5.3 (Demo 9 multi-env + LiteSchur) | 4-5 days | ParallelEnvTest |
| Phase 5.4 (Demo 10 cable + soft-soft) | 5-7 days | CableGrabRaptor — largest piece |
| Phase 4 (cleanup) | 0.5 day | legacy kernel removal, docs |
| **Total** | **~3 weeks** | All 10 demos byte-aligned with RealSim |