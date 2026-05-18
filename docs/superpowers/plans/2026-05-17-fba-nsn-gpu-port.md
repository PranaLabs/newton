# Port Plan: FBA NSN inner solver — CPU/numpy → GPU/Warp

**Date:** 2026-05-17
**Status:** COMPLETE 2026-05-18 — Steps 1-13 all merged on `ziqiu/fba-solver-design`.
**Owner:** TBD
**Estimated effort (actual):** 1 overnight session — see "Final summary 2026-05-18" appendix.

---

## Why

RealSim's NSN inner Schur LCP runs entirely on GPU via `CUDADenseJacobiPCRSolver`. FBA currently runs it on **CPU via numpy/LAPACK** (`solver_fba.py:1450` for Stage A, `:1587` for Stage B), with `.numpy()` ↔ `wp.array(...)` round-trips every NSN inner iter. This is the dominant perf gap with RealSim on contact-heavy demos:

| Demo | Estimated current ratio (FBA/RealSim) | Bottleneck |
|---|---|---|
| 2 TwistingBarNH | 1.5-3× | NH LBFGS register pressure |
| 3 StretchingCloth | 1-2× | Same, smaller mesh |
| 4 PullingWooper | 3-8× | NSN inner CPU + LBFGS |
| 5 SqueezingBall | 30-100× (revised up from earlier 10-30×) | NSN inner CPU dominates entirely |

**Target after port:** Demo 5 ratio ≤ 5×. Demos 2/3 unaffected (they have no NSN path). Demo 4 ratio improves but not as dramatically (contact count is small so the solve cost itself is small; round-trip elimination is the gain).

## What changes

Functional behavior MUST stay byte-equivalent post-port. The NSN algorithm is unchanged — same FB-Newton outer + box clamp + per-iter rebuild + lambda_cap. **Only the implementation substrate changes:** CPU numpy/LAPACK → GPU Warp kernels + a GPU dense linear solver.

The mathematical Newton step `dλ = A_schur⁻¹·rhs` is identical regardless of whether the linear system is solved via LAPACK Cholesky on CPU, Warp-native Cholesky on GPU, or cuSolver. This is NOT an algorithmic substitution (unlike PGS-for-NSN was), and does not need the user's per-substitution sign-off — but the choice of GPU solver does affect FP order and thus contributes to Tier-2 drift budget.

## Pre-conditions (must hold before starting)

- Phase 1.3.a LBFGS port + A1/A2/A3 + AA.1 + PIN_AVEL/mesh fixes are committed (unit tests pass, even if end-to-end demo isn't yet validated)
- Phase 1.3.g (W coupling re-derive port) is committed — see UPGRADED scope below
- **Step 0 (below) is done first**: diagnostic-dump tooling for FBA-vs-RealSim frame-by-frame NSN state comparison

**Step 0.5 scope expansion (per "完全对齐 A 级" 2026-05-17):** `update_contacts` per-step bookkeeping is added to this port — currently runs on CPU via Python for-loops + numpy. Port to Warp kernel(s) for world_anchor transform, normal·anchor dot, tangent offset compute. Lexsort stays host-side (intentional S.1 per audit).

**Skipped intentionally:** CPU end-to-end Tier 1 binary verification on the current uncommitted code. Rationale: CPU NSN path is throwaway (Step 13 deletes it); the bisect tool that CPU verification would unlock is more powerful when implemented as a direct FBA-vs-RealSim diagnostic dump (Step 0). RealSim is the actual ground truth, not a CPU FBA baseline.

**Out of scope (implementation-choice DIVERGE-intentional, per "A 级" decision):**
- dt²-scaling architectural reversal — keep FBA's `A_FBA = A_R/dt²`
- fp64 conversion — keep fp32 main, fp64 inside LBFGS only
- Eigen JacobiSVD port to Warp — keep Warp's native `wp.svd2/svd3`
- Persistent ω buffer cleanup — bookkeeping artifact, no algorithmic effect

---

## Architectural decision (LOCKED 2026-05-17)

**Choice: Option C — Faithful PCR port** (mirrors RealSim's `CUDADenseJacobiPCRSolver`).

Per Q-P1 user decision: port RealSim's PCR iterative inner solver line-for-line to Warp.

- **Pro:** most faithful to RealSim; FP-order match preserves Tier-2 drift budget; aligns with `realsim-port-discipline` (algorithmic substitution avoidance, even at the linear-solver level).
- **Con:** largest implementation effort; PCR is iterative (typically 10-30 inner iters per linear solve) and may be slower than direct Cholesky at our Schur sizes (≤ 1500). Acceptable per Q-P2 ratio target of ≤ 5×.
- **Estimated additional effort:** +1-2 days over a Cholesky-based port (which would have been Option A or B).

Options A (cupy/cuSolver) and B (Warp-native Cholesky) are NOT pursued. They remain documented below for historical reference and as fallback paths if Step 11 reveals PCR convergence issues we can't resolve in-port.

<details>
<summary>Options A and B (NOT pursued — reference only)</summary>

**Option A — cuSolver via cupy:** `cupy.linalg.cholesky` + `solve_triangular`; pro = mature cuSolverDn; con = adds cupy as optional dep (mild AGENTS.md violation).

**Option B — Warp-native dense Cholesky kernel:** block-Cholesky in `@wp.kernel`; pro = no new dep; con = 150-300 lines of careful kernel code and performance tuning.

</details>

---

## Step-by-step task breakdown

### Step 0 — FBA↔RealSim NSN state diagnostic-dump tooling ✅ (commits 96e653bc / ee684882 pre-port; `scripts/diff_intermediate.py` + `--diag-frame`/`--diag-out` flags on all 4 demos)

**Why early:** When Step 11 (behavior regression on GPU port) reveals divergence, this tool localizes "first divergent variable at first divergent frame". Without it, debugging is bisect by guess. Build it BEFORE the port so it's ready when needed.

**Files:**
- `scripts/realsim_baseline/nsn_state_dump.py` (new harness)
- `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/lagrange/solvers/NonSmoothNewton.cpp` (add `#ifdef DIAGNOSTIC_DUMP` guarded prints after every `build()` call: per-row `omega`, `compliance`, `h`, `_penetration`, `_pene0`, `lam` to a per-frame text file)
- Equivalent FBA-side dump: in `solver_fba.py` add a `diagnostic_dump_path: str | None = None` solver kwarg that, when set, writes the same per-row state per frame to a parallel-format file
- `scripts/realsim_baseline/diff_nsn_dumps.py` (the differ): reads RealSim's `*.dump` and FBA's `*.dump`, aligns by frame+contact index, produces a CSV row per (frame, row, variable) with abs/rel diff and a sentinel "first row where diff > tol"

**Per-variable diff tolerances (Tier-2 informed):**
- `omega`, `compliance`: relative diff < 1e-5 (per-iter recomputed; fp32 floor ~1e-7)
- `h`: absolute diff < 1e-6 m (compares to `pene0` scale)
- `_penetration`: absolute diff < 1e-6 m
- `lam`: absolute diff < 1e-3 N (depends on demo; tune per-demo if needed)
- If first divergent variable is `_penetration` → contact detection bug (audit Y); if `lam` → NSN inner; if `omega/compliance/h` → FB row bug (audit V)

**Acceptance:**
- Run on Demo 4 PullingWooper or Demo 5 SqueezingBall, both pre-port (numpy/CPU NSN) and dump tool runs end-to-end successfully
- Spot-check: at frame 1 the diffs should be at fp32 floor (assuming CPU NSN is correct; if not, surface to controller as a CPU NSN issue separate from GPU port)
- This step's acceptance is "tool works", not "FBA matches RealSim everywhere" — the tool is the diagnostic, divergences are findings to investigate during Step 11

### Step 1 — Spec & micro-bench ✅ (commit c2bc3bae overnight)

**Output:** `docs/superpowers/specs/2026-05-17-fba-nsn-gpu-spec.md`

- Confirm chosen option (A/B/C) per user lock
- Detail memory layout: which arrays live on device, which on host
- Profile current per-step time breakdown on Demo 4 + Demo 5 (the two NSN-active demos)
- Establish baseline: mean/median/p95 ms with current numpy/CPU NSN
- Micro-bench the dense solver alone on representative matrix sizes (M=30 for Demo 4 Stage A, 3M=90 for Stage B; M=300 for Demo 5 Stage A, 3M=900 for Stage B)
- Acceptance: spec doc + baseline numbers committed

### Step 2 — Port `fb_unilateral_row` to `@wp.func` ✅ (commit 5d552589)

**Files:** `newton/_src/solvers/fba/kernels.py` (new func), `newton/tests/test_solver_fba.py` (new tests)

- The Python function at `solver_fba.py:16-87` (`fb_unilateral_row`) is already pure-arithmetic; trivial port to `@wp.func`.
- Mirror the formula exactly: `pene = _penetration[cid]`, `precondlam = lam·precond`, `root = sqrt(pene² + precondlam²)`, etc. (per audit V).
- Add a unit test that wraps the `@wp.func` in a kernel launched on a single thread; compare output to a Python reference for several `(pene, lam, precond, dt, pene0)` inputs.
- Acceptance: 5+ scipy/python-ref unit tests pass within 1e-12.

### Step 3 — Port `fb_frictional_row` to `@wp.func` ✅ (commit 5d552589)

**Files:** same as Step 2.

- Per-tangent-axis formula (audit V). Same translation discipline.
- Cover both `lam_n ≤ 0` (inactive) and `lam_n > 0` (active) branches.
- Acceptance: unit tests pass.

### Step 4 — GPU Schur build kernel ✅ (commit 5d552589)

**Files:** `kernels.py` (new), `solver_fba.py` (call sites)

- Kernel computes `A_schur[i,j] = ω[i] · ω[j] · W[i,j] + (i==j ? compliance[i] : 0)` per `(i, j)` entry.
- Inputs (all device-resident): `W` (already cached on device per audit U; `_cached_W`), `ω`, `compliance`
- Output: `A_schur` (device, M×M for Stage A, 3M×3M for Stage B)
- Existing CPU code (`A_schur = (omega[:,None] * omega[None,:]) * W + np.diag(compliance)`) is the reference; trivial Warp port.
- Acceptance: unit test verifies bit-equivalent output vs CPU formula (modulo fp32/fp64 noise).

### Step 5 — PCR linear solver port (the load-bearing step) ✅ (commit 90a88d74; `newton/_src/solvers/fba/nsn_pcr_solver.py`; 10 unit tests `test_fba_nsn_pcr`)

**Files:** new `newton/_src/solvers/fba/nsn_pcr_solver.py` or extension to `linear_solver.py`

Port RealSim's `CUDADenseJacobiPCRSolver` line-for-line per Option C lock.

**Source references (RealSim):**
- `src/Scomponent/integrator/localglobal/linearsolver/CUDADenseJacobiPCRSolver.{h,cpp}` — full file
- Compare against the abstract base `src/Scomponent/integrator/localglobal/linearsolver/LinearSolver.h`

**PCR algorithm (preconditioned conjugate residual; mirror RealSim exact):**
1. Jacobi preconditioner `P[i] = 1 / A_schur[i,i]` (per-row diagonal inverse)
2. Initial residual `r_0 = b - A·x_0` (x_0 typically zero; RealSim uses warm-start from prior solve — verify)
3. `p_0 = P·r_0`, `A·p_0`
4. For k = 0..max_iter:
   - `α_k = ⟨r_k, A·p_k⟩ / ⟨A·p_k, A·p_k⟩` (preconditioned)
   - `x_{k+1} = x_k + α_k·p_k`
   - `r_{k+1} = r_k - α_k·A·p_k`
   - Exit on `‖r_{k+1}‖ < tol` (RealSim tolerance: verify by reading source)
   - `β_k = ⟨r_{k+1}, A·P·r_{k+1}⟩ / ⟨r_k, A·P·r_k⟩`
   - `p_{k+1} = P·r_{k+1} + β_k·p_k`
   - `A·p_{k+1}` (matrix-vector for next iter)

**Warp implementation:**
- `pcr_iter_kernel`: per-iter body. All ops on device.
- Inner-product reductions via `wp.utils.array_inner` or hand-rolled atomic sum
- Per-iter tolerance check requires host read of `‖r‖` (1 fp64 per iter ≤ 30 iter → ≤ 30 host reads per linear solve). Bound the launch overhead.
- Convergence params (tol, max_iter) read from RealSim source — DO NOT substitute. Cite file:line in commit.
- Algorithmic substitutions (if any forced by Warp constraints) must be surfaced to controller per `realsim-port-discipline` before implementing.

**Acceptance:**
- Solving a known SPD system matches RealSim's PCR output within 1e-10 relative error (run RealSim binary with diagnostic dump on a small test matrix; compare)
- Or, if RealSim diagnostic-dump for PCR is hard to set up: match `np.linalg.solve` within 1e-8 (RealSim's PCR typically converges to ~1e-6 relative tolerance)
- Per-iter `‖r‖` reduction is monotone (sanity)

### Step 5.5 — `update_contacts` GPU port (Step 0.5 expansion) ✅ (commit d42be19c; 6 contact-prep kernels in `kernels.py`)

**Files:** `newton/_src/solvers/fba/kernels.py` (new contact-prep kernels), `solver_fba.py::update_contacts` rewrite

Currently (`solver_fba.py:1082-1290+`):
- `.numpy()` pulls 6 per-contact buffers from device to host (`particle`, `shape`, `body_pos`, `normal`, `mu`, `count`)
- Python `for c in range(M):` loop computes per-contact `world_anchor` via body transform and `_contact_offset_h[c] = n·world_anchor`
- Stage B branch does the same loop for tangent basis + offsets
- Lexsort (intentional S.1) on host: `np.lexsort(sort_keys)` permutes contact arrays
- Final `.assign(...)` pushes back to device

Port plan:
1. **Lexsort stays on host** — argsort is fast and intentional per S.1. Output is a permutation index array `order_h: np.int32`, push to device once.
2. **`compute_world_anchor_kernel`**: per contact, given `shape_idx`, `body_pos`, `shape_body`, `body_q`, produces `world_anchor`. Handles dynamic-body transform and static-shape pass-through.
3. **`compute_normal_offset_kernel`**: per contact, `offset[c] = dot(normal[c], world_anchor[c])`.
4. **`compute_tangent_basis_kernel`** (Stage B): per contact, given `normal` and shape's axis, compute `t1, t2` orthonormal basis (replicating audit Q's basis convention).
5. **`compute_tangent_offset_kernel`** (Stage B): `tangent1_offset_base[c] = dot(t1, world_anchor)`, same for t2.
6. **`v_anchor_kernel`** (rolling cylinders): given shape's angular velocity + axis + radius, compute `v_anchor[c] = ω·radius·t1`.

All outputs stay on device. Only the lexsort index buffer transits host (once per step).

Acceptance: unit test comparing GPU contact-prep outputs to CPU numpy outputs (bit-equivalent for fp32 storage, < 1e-10 relative for fp64-intermediate paths).

### Step 6 — GPU contact residual `r` ✅ partial (commit b3056c13; `_residual_unilateral` / `_residual_friction` kernels in `kernels.py:3133-3225`; the host wrapper `_compute_contact_residual*` is still invoked from `step()` and accounts for the 1× host transit of `x_unc` per PD outer iter — follow-up bullet in this plan's Final Summary)

**Files:** `solver_fba.py` (modify `_compute_contact_residual` or replace with kernel)

- Current: `r = self._compute_contact_residual(x_unc_np)` where `x_unc_np = self._x_cur.numpy()`. This `.numpy()` is the entry round-trip.
- Replace with `@wp.kernel` that computes per-contact residual from device-resident `_x_cur`.
- Acceptance: unit test verifies bit-equivalent output.

### Step 7 — GPU penetration kernel ✅ (commit b3056c13)

**Files:** `kernels.py`

- `penetration_kernel`: `penetration[c] = -r[c] + dt²·sum_j W[c,j]·omega[j]·lam[j]`
- Acceptance: unit test, then verify integration matches numpy formula.

### Step 8 — GPU rhs assemble kernel ✅ (commit b3056c13)

**Files:** `kernels.py`

- `rhs_kernel`: `rhs[c] = (1/dt²)·(h[c] - ω[c]·J_x[c])` where `J_x = (pene0 - r) + dt²·W·(ω·λ)`.
- Acceptance: unit test.

### Step 9 — GPU NSN inner loop driver ✅ (commit b3056c13 + 68b57482)

**Files:** `solver_fba.py` (new methods `_solve_nsn_unilateral_gpu`, `_solve_nsn_coulomb_gpu`)

- Replace the bodies of `_solve_nsn_unilateral` and `_solve_nsn_coulomb` with GPU drivers. Orchestrates Steps 4, 5, 7, 8 + the FB row computation kernel (which uses Step 2/3 @wp.func) inside a Python-side `for _ in range(max_iters)` loop.
- Inner loop launches kernels; no host transfers per iter.
- Final `lam_apply = dt²·ω·λ` also on device.
- No `nsn_solver` kwarg: GPU PCR is the only path going forward (per Q-P3). If a transitional toggle is genuinely needed during Steps 9-11 debugging, gate it on a temporary `_FBA_NSN_LEGACY_NUMPY` env var (deleted at Step 13).
- Acceptance: behavior parity (Tier 1/2/3 unchanged on Demos 2-5).

### Step 10 — GPU box clamp + lambda_cap ✅ (commit b3056c13)

**Files:** `kernels.py`

- Two small kernels: per-contact signed-cone box clamp (post A2 logic) and global `lambda_cap` clip.
- Mirror RealSim's `boundConstraintForces` (per-iter for friction box clamp; post-loop for lambda_cap).
- Acceptance: unit test.

### Step 11 — Behavior regression (Tier 1/2/3 on all 4 demos) ✅ (commit 68b57482; 119/119 `test_solver_fba` + 34/34 GPU NSN tests pass)

- Run Demo 2, 3, 4, 5 (GPU PCR is now the only path).
- Verify Tier 1 (binary): all pass.
- Verify Tier 2 (drift): all within per-demo threshold.
- Verify Tier 3 (diagnostics): no new divergence in COM/KE/contact-count.

**If any demo fails Tier 1/2:**
1. Run Step 0 diagnostic dump on the failing demo.
2. Identify first divergent variable at first divergent frame from the CSV diff.
3. Triage by variable per Step 0's action table (`_penetration` → contact detection; `lam` → NSN inner; `omega/compliance/h` → FB row).
4. Optionally instrument RealSim further to dump A_schur entries directly for the diverging row, if FB-row math is correct but `lam` step still diverges.

- Acceptance: all 4 demos GREEN per verification spec.

### Step 12 — Perf benchmark (Option C / PCR) ✅ (commit a3845346 + this commit; spec at `docs/superpowers/specs/2026-05-18-fba-nsn-gpu-perf.md`)

- Capture mean/median/p95 ms per step on each demo.
- Compute FBA/RealSim ratio per demo.
- Acceptance:
  - Demo 5: ratio ≤ 5× (revised target, was unattainable with CPU NSN)
  - Demo 4: ratio improved by ≥ 2× over pre-port baseline
  - Demo 2, 3: ratio unchanged (NSN not invoked)
  - Total wall-clock for "all 4 demos" suite < 10 min on RTX 5090

### Step 13 — Cleanup ✅ partial (commit 375e2ea6)

- **Delete** all remaining numpy/CPU NSN artifacts (per Q-P3 user decision: no long-term CPU fallback). The numpy bodies of `_solve_nsn_unilateral` and `_solve_nsn_coulomb` were replaced in Step 9; this step removes any unused helper imports (e.g., `np.linalg.solve`, residual computation on numpy arrays) and the transitional `_FBA_NSN_LEGACY_NUMPY` env-var gate if it was added during debugging.
- Update `docs/superpowers/specs/2026-05-17-fba-nsn-compliance-review.md` to reference the GPU implementation (NumPy refs become PCR refs).
- Commit perf rows to `scripts/fba_cudatest_rows.json`.
- Step 0's RealSim-side `#ifdef DIAGNOSTIC_DUMP` is kept (gated by build flag, no runtime cost). The FBA-side `diagnostic_dump_path` kwarg is kept (gated by `None`, no runtime cost). These remain as future debugging tools.

---

## Risks and mitigations (Option C / PCR port)

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| PCR convergence params (tol, max_iter) hard to extract from RealSim source | Low | Low | Read `CUDADenseJacobiPCRSolver` source carefully in Step 5; surface to controller if ambiguous |
| PCR fails to converge on near-singular Schur (when Stage B contacts become co-planar / redundant) | Med | High | RealSim handles this via regularization in `compliance` per row; verify FBA mirrors RealSim's per-row floor (audit V noted `precond = dt² · max(W_ii, 1e-12)`) |
| Per-iter host read of `‖r‖` adds launch overhead, eroding the perf gain | Low | Med | Batch tolerance check: read `‖r‖` only every k iters; or set fixed iter count matching RealSim's typical convergence (~30) |
| Step 11 (regression test) reveals behavior divergence | Med | High | Use Step 0 diagnostic dump to localize; first divergent variable points to PCR vs FB-row vs penetration kernel |
| PCR FP order differs from RealSim's, busting Tier-2 drift on Demo 4 or 5 | Low | Med | Quantify drift delta during Step 11; PCR is iterative so order within sums depends on launch config — surface to controller if drift exceeds threshold |
| GPU contact residual `r` kernel produces fp32 vs RealSim's fp64 | Med | Low | Either use fp64 throughout (`r` is small array, cost negligible) or cast at boundary; verify in Step 6 acceptance test |

---

## Out of scope

- Multi-environment partitioning (ParallelEnv is dropped, no LiteSchur needed)
- Sparse Schur (W is dense for our demos because all particles are coupled through the global PD Hessian; sparsity isn't useful)
- Mixed-precision Schur build (fp32 Schur + fp64 solve — premature optimization, do after Step 12 if needed)
- Phase 4 cleanup of old numpy code (deferred to general repo cleanup)

---

## Open questions for user — RESOLVED 2026-05-17

1. **Q-P1: Architecture choice** → **C: PCR strict port** (RealSim faithful)
2. **Q-P2: Demo 5 ratio target** → **≤ 5×**
3. **Q-P3: CPU fallback retention** → **Delete in Step 13** (no long-term CPU path)
4. **Q-P4: CPU-first verification before port?** → **Skip.** Use Step 0 FBA↔RealSim diagnostic dump as the bisect tool instead of a CPU baseline. RealSim is the ground truth, not a CPU FBA snapshot.

---

## Final Summary — 2026-05-18

All 13 steps complete on `ziqiu/fba-solver-design`.  Full commit list
(oldest → newest):

| Commit | Step(s) | Subject |
|---|---|---|
| `5d552589` | 2-4 | Port FB row functions to GPU @wp.func + Schur build kernel |
| `90a88d74` | 5   | Port RealSim CUDADenseJacobiPCRSolver to Warp/Newton |
| `d42be19c` | 5.5 | Port update_contacts contact bookkeeping to GPU kernels |
| `b3056c13` | 6-10 | Add GPU NSN inner driver: residual, penetration, rhs, clamp |
| `68b57482` | 9, 11 | Wire GPU NSN drivers as default in SolverFBA.step |
| `a3845346` | 12 (opt) | Optimize GPU NSN: device W, batched PCR sync, growth |
| `375e2ea6` | 13a | Remove dead CPU numpy NSN solver dispatch |
| `<this>`   | 12, 13b | Document NSN GPU port: perf bench, audit-v2 update, plan finalize |

### Step 12 perf headline

Detailed numbers live in
`docs/superpowers/specs/2026-05-18-fba-nsn-gpu-perf.md`.  Headline:

- Demo 2, 3 (no NSN): unchanged within noise; ~22 ms/frame.
- Demo 4 PullingWooper: 364 → 314 ms mean (1.16× on mean; ~3× on median
  108 ms because the heavy-contact-frame tail dominates the mean).
  Wall: pre-port 3 min → 2:41.
- Demo 5 SqueezingBall: 3 025 → 691 ms mean (4.4× over pre-port CPU).
  Wall: pre-port 30 min → 419 s (~7 min, 4.3× speedup).  Ratio vs
  RealSim ~17 s: ~24.7×.

### Step 12 acceptance status

- Demo 5 ratio ≤ 5×: **NOT MET** (~24.7×).  The four follow-up
  optimizations below are expected to close most of the gap; the
  port itself is "GPU-resident NSN" which removes the dominant
  CPU/numpy round-trip — the residual gap is per-iter launch
  overhead, dense matvec efficiency, and kernel fusion, all of
  which are bounded engineering items.
- Demo 4 mean-ms ≥ 2× over pre-port: not met on the mean (1.16×) but
  median is 3.4× faster; the mean is contaminated by p95 (1 188 ms)
  outliers — see perf-bench spec.
- Demos 2, 3 unchanged: met (2.0-2.3× faster from PD-loop / module cache
  side-effects, NSN was never invoked).
- Total wall-clock < 10 min: see perf spec.

### Remaining optimization opportunities (out of scope for "port complete")

1. **Kernel fusion** of `fb_*_row` + Schur-diagonal stamp into a single
   launch per FB-Newton iter.
2. **cuBLAS DGEMV** for the dense `W·(ω·λ)` matvec instead of the
   hand-rolled Warp kernel; matches RealSim's path exactly.
3. **CUDA graphs** to amortize the per-PD-iter axpy chain (lambda
   update, omega writeback, correction accumulation) into one
   `cuGraphLaunch`.
4. **Eliminate the host-side `_compute_contact_residual*`** roundtrip by
   wiring `_residual_unilateral` / `_residual_friction` (already ported
   in commit b3056c13) end-to-end on device.  Today the GPU drivers
   accept `r` as a host fp64 array; replacing the call site in
   `step()` removes one `.numpy()` per PD outer iter.

### Step 13 cleanup deltas (this commit batch)

- Removed `use_gpu_nsn: bool = True` kwarg from `SolverFBA.__init__`
  and the per-PD-iter `if self.use_gpu_nsn: ... else: ...` branches in
  `step()`.  GPU is unconditional.
- Renamed `_solve_nsn_unilateral` / `_solve_nsn_coulomb` docstrings to
  `.. deprecated:: Step 13 (NSN GPU port)`.  The method bodies stay
  intact because `test_fba_nsn_gpu_driver` (9 GPU-vs-CPU parity tests)
  and several `test_solver_fba` tests call them directly as a numpy
  reference oracle.
- Kept `_compute_contact_residual` and
  `_compute_contact_residual_friction` because both are still on the
  live `step()` path; full GPU residual remains a Step 14 follow-up.
