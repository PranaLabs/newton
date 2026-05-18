# Demo 4 / Demo 5 Tier 2 Drift — Root-Cause Investigation

**Date:** 2026-05-18
**Branch:** `ziqiu/fba-solver-design`
**Task:** #35 — investigate Tier 2 YELLOW (systematic drift) verdicts for PullingWooper
(Demo 4) and SqueezingBall (Demo 5) post NSN-GPU-port (commit `0d27fd11`).

**Verdict from prior run (commit `0d27fd11`):**
- Demo 4 PullingWooper: Tier 1 PASS (`min_y = -8.408`, matches RealSim `-8.41`),
  Tier 2 max-drift `2.4 m`, COM drift max `0.40 m` → **YELLOW systematic**.
- Demo 5 SqueezingBall: Tier 1 PASS (`min_y = -10.000`, exact), Tier 2 max-drift
  `1.01 m`, COM drift max `0.21 m` → **YELLOW systematic**.

**Constraint per task spec:** read-only investigation, no production code changes.

---

## Step 1 — first frame with significant divergence

Script `/tmp/find_first_diverge.py` (transient, not committed) computes
per-frame `max_t ||fba[t] − realsim[t]||_max` over each demo's saved
trajectory. Alignment is `FBA[t] vs RS[t]` (both indexed from initial state).

### Demo 4 PullingWooper

| frame | max_drift (m) |
|-------|---------------|
| 0     | 0.000e+00     |
| **1** | **1.944e-02** |
| 2     | 2.438e-02     |
| 5     | 6.426e-02     |
| 10    | 1.261e-01     |
| 20    | 2.590e-01     |
| 270   | 8.724e-01     |
| 467   | 2.654e+00 (max) |

**First frame > 1e-3 m: frame 1 (drift = 19.4 mm).** Monotone growth
from step 1; no abrupt jump within the first 100 frames. The largest
frame-to-frame jumps appear at frames 437/445/461/462 (~0.18 m each)
when the wooper tail enters the cylinder contact zone — secondary
amplification once contacts activate, but the systematic component is
already established by step 1.

### Demo 5 SqueezingBall

| frame | max_drift (m) |
|-------|---------------|
| 0     | 0.000e+00     |
| 1     | 1.672e-03     |
| 2     | 1.910e-03     |
| 3     | 2.613e-03     |
| **4** | **1.876e-02** (first big spike) |
| 5     | 2.396e-03     |
| 10    | 1.215e-03     |
| 70    | 2.375e-02     |
| 120   | 5.105e-02     |
| 220   | 7.905e-01     |
| 312   | 1.014e+00 (max) |

**First frame > 1e-3 m: frame 1 (drift = 1.67 mm).** A larger spike at
frame 4 (18.8 mm) then dissipates, but accumulation resumes once friction
kicks in (~frame 50+). Top jumps are at frames 398/428/501 (~5 cm each)
during sustained rolling/squeezing.

---

## Step 2 — diagnostic dumps captured

| demo | frame | FBA dump                            | RealSim dump prefix         |
|------|-------|-------------------------------------|-----------------------------|
| 4    | 1     | `/tmp/fba_diag_demo4_f1.npz`        | `/tmp/rs_diag_demo4_f1`     |
| 5    | 4     | `/tmp/fba_diag_demo5_f4.npz`        | `/tmp/rs_diag_demo5_f4`     |

RealSim binary used: `/home/ziqiu/work/RealSim_py/realsim_py/build/bin/RealSim`
(commit `f5009b2bb` — the env-driven PD intermediate diag instrumentation
referenced in the task prompt). RealSim's process crashes at exit
(`double free or corruption (out)`) during Alembic shutdown after the
diag step has already been written; the bin/meta files are valid.

Both dumps contain `x_pre`, `x_inertia`, `rhs_k{0..9}`, `x_post_solve_k{0..9}`
at `(N, 3)` float64.

---

## Step 3 — diff results

### Demo 4 frame 1 (no contacts active — wooper tail is 4+ m above cylinders)

```
variable             max_diff     mean_diff    median_diff   argmax_i
rhs_k{0..9}          3.96e+10     6.19e+08     2.25e+04      3498
x_post_solve_k0      9.48e-03     1.65e-05     3.46e-07      3193
x_post_solve_k1      1.36e-02     1.13e-05     8.85e-07      3193
x_post_solve_k2      1.59e-02     1.61e-05     1.85e-06      3193
...
x_post_solve_k9      1.91e-02     7.20e-05     3.06e-06      3193
x_inertia            2.38e-07     4.39e-08     2.43e-08      1304
x_pre                2.38e-07     4.39e-08     2.43e-08      1304
```

### Demo 5 frame 4 (friction + contacts active)

```
variable             max_diff     mean_diff    median_diff   argmax_i
rhs_k{0..9}          8.33e+03     2.37e+03     2.03e+03      979
x_post_solve_k0      1.38e-02     1.72e-05     6.39e-07      989
x_post_solve_k9      9.06e-03     1.67e-05     6.67e-07      1378
x_inertia            ~fp32-eps    -            -             -
x_pre                ~fp32-eps    -            -             -
```

---

## Step 3a — `rhs_k*` divergence is a formulation difference, NOT a bug

The Demo 4 `rhs_k*` diff shows a **clean 10000x ratio** across all
particles (min ratio 9.9992e+03, max 1.0001e+04, mean 1.0000e+04 over all
5325 particles; identical sign pattern). Ratio `10000 = 1/dt² = 1/0.01²`.

This is the *expected* difference between two algebraically-equivalent PD
formulations:

| | system matrix `A` | RHS |
|-|------------------|-----|
| **RealSim** (`LocalGlobalSolver.cpp:200-205`, `Mass.cpp:35-43,66-70`) | `M + dt²·Σ w_e·AᵀA` | `M·sₙ + dt²·Σ proj` |
| **FBA** (`linear_solver.py:82-86`, `kernels.py:170-181`) | `M/dt² + Σ w_e·AᵀA` | `(M/dt²)·x_inertia + Σ proj` |

Multiply FBA's system by `dt²` → identical to RealSim's. The two `x`
solutions should be bit-identical in infinite precision. The `rhs_k*`
dumps therefore differ by exactly `1/dt²`. **This is intentional and
the `diff_intermediate.py` comparison of `rhs_k*` is meaningless for
parity verification.** Only `x_pre`, `x_inertia`, `x_post_solve_k{k}`
are unit-compatible across the two implementations.

(Demo 5 frame 4 shows a different magnitude — ~8e3 — because the
input scale at that frame is different, but the same 1/dt² formulation
ratio applies.)

### First genuinely-divergent variable

For both demos the first frame-1 / frame-4 unit-compatible divergence
is at **`x_post_solve_k0` — the output of the first PD global solve,
BEFORE any NSN correction**. (Confirmed for Demo 4: zero active contacts
at frame 1, ergo NSN is a no-op.)

- Demo 4 frame 1: `x_post_solve_k0` diverges by **9.5 mm** at particle
  3193 (wooper tail tip at world `(1.35, 3.70, −2.72)`).
- Demo 5 frame 4: `x_post_solve_k0` diverges by **13.8 mm** at particle
  989 (ball surface at `(0.005, −0.051, −1.21)`, near cylinder 0
  contact zone).

`x_pre` and `x_inertia` agree to **2.4e-7 m** — exactly the fp32 epsilon
at scale ~1 m. So the divergence is introduced entirely **between**
`x_inertia` and `x_post_solve_k0`, i.e., during PD local projection +
global solve (+ in Demo 5's case, the in-iter NSN correction inside
`globalSolve`).

### Drift evolution within frame 1 of Demo 4 (no contacts)

```
k=0   max_drift=9.48e-03
k=1            1.36e-02
k=2            1.59e-02
k=3            1.68e-02
...
k=9            1.91e-02
```

Monotone growth across PD outer iterations. Same argmax particle (3193)
throughout — the wooper tail tip, where deformation gradient `F` is
largest. **No contacts** are involved in this growth → the source is
the local-projection / global-solve loop only.

---

## Step 4 — Hypothesis ranking

### Hypothesis 1 (top): fp32 precision in PD local projection (NH SVD + scatter)

**Evidence:**
- `x_inertia` already at fp32 epsilon (2.4e-7) — `state_in.particle_q` is
  `wp.array[wp.vec3]` = fp32 in Newton's model.
- `_rhs` is allocated as `wp.empty(N, dtype=wp.vec3, device=device)` =
  fp32 (`solver_fba.py:348`). All projection scatters (`add_inertia_to_rhs`,
  `project_pin_kernel`, `project_stretching_neohookean_tet_compute_kernel*`,
  `gather_per_particle_kernel`) write to fp32 `_rhs`.
- The NH local projection kernel runs **SVD in fp32**
  (`kernels.py:1554-1556`: `F = Ds * Dm_inv` with `Ds`, `Dm_inv` both
  `wp.mat33` fp32; `wp.svd3(F)` is fp32). LBFGS sigma minimization runs
  in fp64 but operates on **fp32 SVD inputs**, then casts the result
  back to fp32 for `P = U·diag(σ)·Vᵀ` and the `w·(Dm_inv·Pᵀ)` scatter.
- RealSim runs the full chain in fp64 (Eigen `MatX3R` with `real = double`,
  `signedEigenSVD` fp64, LBFGS fp64 throughout).
- Linear solve internals (`FBALinearSolver.solve`) ARE fp64 — the
  precision loss is upstream, in the RHS assembly.
- Drift is monotone in PD outer iter (9.5 → 19.1 mm in 10 iters) and the
  worst particle is the tail tip where `|F|` is largest. Both are
  characteristic of a precision-driven local projection error.

**Magnitude:** input drift 2e-7 m → output drift 1e-2 m across 1 frame
× 10 PD iters. That's ~50,000x amplification. For a stiff system
(`young=1e7`, `mu≈3.85e6`) with `(M/dt²) ≈ 1880`, the local projection
coefficient `w·K = 2μ·V·K ≈ 7.7e5` per tet at the diagonal — A is
heavily dominated by projection terms, not mass terms, in tetrahedral
regions. The condition number of `A` is therefore set by `2μ·V/dt²`
ratios, which can easily be 1e5+. **Amplification consistent with
fp32 input to a well-conditioned-but-stiff fp64 solve.**

**Fix candidate (NOT applied, citation only):**
- `newton/_src/solvers/fba/kernels.py:1542-1591` (
  `project_stretching_neohookean_tet_compute_kernel_lbfgs`) and the sibling
  Newton-5 kernel at `:1421-1505` — cast `Ds`, `Dm_inv`, `F` to fp64 before
  `wp.svd3`. Warp 1.14 supports fp64 SVD via `wp.svd3` on `mat33d`.
  Cast result back to fp32 only at the final `contributions[t, k] = ...`
  scatter.
- `newton/_src/solvers/fba/solver_fba.py:348` — promote `_rhs` to fp64
  (allocate as `wp.array[wp.vec3d]` and update all scatter kernels). This
  is the heavier fix (touches every projection kernel signature) but
  matches RealSim's data type 1:1.

**Effort estimate:** 1-2 days. The kernel rewrites are surgical but
touch ~6 projection kernels and the gather. **Cannot avoid the fp64
ripple through `_x_inertia`, `_x_cur`, `_rhs`, `_tet_contrib_d`,
`_tri_contrib_d`, `_edge_contrib_d`** to fully match RealSim, but a
partial fix (fp64 only inside the NH SVD/LBFGS sub-block, with fp32
scatter) is achievable in 0.5 day and likely captures most of the
precision drift.

### Hypothesis 2: SVD branch convention mismatch (svd3 vs signedEigenSVD)

**Evidence:**
- RealSim uses `tools::svd::signedEigenSVD(F, U, s, V)` — explicitly
  *signed* SVD that yields a deformation-preserving sign convention.
- Warp's `wp.svd3` returns standard SVD; sign of `det(U)` and `det(V)`
  is not guaranteed to match RealSim's convention.
- The NH energy is invariant under joint sign flips, but the **per-iter
  LBFGS starting point and step direction** depend on the sign
  convention. Different `σ_init` → different LBFGS path → different
  final `P` (within LBFGS tolerance `1e-6`).
- However: `x_pre`/`x_inertia` agreement to fp32 epsilon argues the
  initial `σ_init` matches at fp32. The discrepancy could be at fp32
  noise in σ_init triggering different LBFGS branches.

**Magnitude:** depends on tet count near the deformation peak. Probably
sub-mm per tet, but could compound. Lower priority than H1.

**Fix candidate:** N/A — Warp's `wp.svd3` does not expose sign control.
Workaround would be to post-process `U, V` to enforce
`det(U)·det(V) == det(F)/|det(F)|` matching RealSim's convention. Need
to consult `RealSim_py/realsim_py/src/Scomponent/tools/svd/signedEigenSVD.cpp`
(not yet read) to confirm the exact convention.

**Effort estimate:** 0.5 day to align convention, plus regression test.

### Hypothesis 3: PD `_rhs` accumulation in fp32

**Evidence:**
- Every PD-iter starts with `zero_vec3_kernel(_rhs)`, then accumulates
  mass + pins + tet projections + tri projections + bending into fp32 `_rhs`.
- `mass·x_inertia / dt²` for the wooper at `y ≈ 4.0`, `m ≈ 0.188`, `dt² = 1e-4`
  produces `~7500` per component. Tet projection contributions are
  `w·(Dm_inv · Pᵀ) ≈ 2μ·V` per tet vertex ≈ `1.4e3` per tet (with V
  ~ 1.8e-4 and `2μ=7.7e6`). Summed over `~30` adjacent tets per
  particle: order `~4e4`. So `_rhs` values are `~4e4`. fp32 rel-error
  ~1.2e-7 → abs-error in `_rhs` per particle is ~5e-3. **Same order as
  the observed Demo 4 k=0 drift (9.5e-3 m).**
- After the fp64 sparse solve, the `5e-3`-error-RHS still produces a
  `5e-3 / A_diag` displacement error. `A_diag` for a tet vertex with
  `~30` adjacent tets at `2μ·V` each is `~3e4` (in FBA's M/dt² scaled
  units), so the per-particle displacement error is `~5e-3/3e4 ≈ 1.7e-7`
  m — far smaller than observed.

**Reconciliation:** the displacement error scales with the **condition
number of A**, not just the diagonal. For a stiff system with anisotropic
deformation, the error can be 10⁴+ times the per-diagonal scale.

**Fix candidate:** promote `_rhs` to fp64 (Hypothesis 1's heavier fix).

**Effort estimate:** subsumed by H1.

### Hypothesis 4 (low): NSN PCR precision at small contact count

**Evidence:** Per `2026-05-17-fba-precision-params-variants-review.md`,
PCR was found to have 1e-4 precision at M=3. Demo 4 frame 1 has M=0
contacts → not applicable. Demo 5 frame 4 has many contacts (ball
squeezed between 2 cylinders + plane) → moderately exercising NSN, but
the divergence at `x_post_solve_k0` for Demo 5 already accounts for the
NSN correction (per RealSim's `globalSolve` flow which calls
`build/solve/applyConstraintCorrection` inside `globalSolve` — see
`LocalGlobalSolver.cpp:373-387`). FBA dumps `x_post_solve_k{k}` **before**
NSN correction is applied (per `solver_fba.py:946-953`), so the Demo 5
diff at `x_post_solve_k0` is a mixed comparison: FBA pre-NSN vs
RealSim post-NSN. **The diff dumps are not yet apples-to-apples for
contact frames.**

**Fix candidate:** add a `x_post_constraint_k{k}` capture in FBA (after
`apply_lambda_correction_combined`) to enable proper post-NSN comparison.
NOT a code-path fix, just diagnostic infrastructure.

**Effort estimate:** 1 hour for the diag plumbing change.

### Hypothesis 5 (ruled out): mass-lumping divergence

**Ruled out:** `x_inertia` (= `x_prev + dt·v_prev + dt²·g`) agrees with
RealSim to fp32 epsilon. `x_inertia` consumes mass only via `f_ext/m` —
since Demo 4 has no external forces beyond gravity (which is mass-
independent) and Demo 5's gravity is similarly uniform, `x_inertia`
agreement implies `inv_mass` matches RealSim's `1.0/m` to fp32. Locked
decision `mass_lumping = uniform_total_over_n` (`decisions.json`) holds
correctly.

---

## Step 5 — Decision points for the user

**Recommended next action (highest expected gain per dev-hour):**

1. **Partial precision lift inside NH/ARAP/Corot local projection
   kernels** (Hypothesis 1, surgical version): cast `F = Ds·Dm_inv` to
   `mat33d` before `wp.svd3`; run SVD + LBFGS / Newton-5 in fp64; cast
   the final `P` back to fp32 for the scatter. Touches ~3 kernels in
   `kernels.py`, no signature changes elsewhere. **Estimated 4-6 hours
   incl. regression test.** Expected effect: should reduce Demo 4
   `x_post_solve_k0` drift from 9.5 mm towards fp32-epsilon levels of
   the *output* (because the residual amplification in the global solve
   is set by `A`'s condition number on a now-accurate RHS).

2. **Add `x_post_constraint_k{k}` to FBA diag dump** (Hypothesis 4 fix
   candidate): 1 hour. **Strongly recommended before any further
   Demo 5 investigation** — current dumps are not apples-to-apples
   for contact frames, and any production fix would be measured against
   misaligned diagnostics.

3. **Full fp64 `_rhs` promotion** (Hypothesis 1, heavier version): defer
   unless step 1 yields insufficient improvement. **Estimated 1-2 days**
   touching ~12 kernels + array allocations.

**Items the user should explicitly approve before code work begins:**
- Is the fp64 partial lift in local projections (step 1 above) acceptable
  given the perf budget? `wp.svd3(mat33d)` is ~2x slower per tet; with
  ~5k tets per frame at 60+ Hz this should be negligible, but worth
  confirming.
- Is the diag-dump augmentation (step 2 above) low-risk enough to ship
  alongside the fix, or should it land as a separate commit first?

**If results from step 1 do NOT close the gap:** the residual drift is
likely sign-convention (Hypothesis 2), and requires a `signedEigenSVD`
port (~0.5 day) or acceptance of the YELLOW Tier 2 status as a known
fp32-vs-fp64 systematic.

---

## File references

- **FBA local projection kernel (NH LBFGS):**
  `newton/_src/solvers/fba/kernels.py:1508-1591`
- **FBA local projection kernel (NH Newton-5):**
  `newton/_src/solvers/fba/kernels.py:1420-1505`
- **FBA RHS allocation (fp32):**
  `newton/_src/solvers/fba/solver_fba.py:348`
- **FBA inertia-to-rhs kernel:**
  `newton/_src/solvers/fba/kernels.py:170-181`
- **FBA PD system build (`M/dt²` formulation):**
  `newton/_src/solvers/fba/linear_solver.py:82-86`
- **FBA linear solver internals (fp64):**
  `newton/_src/solvers/fba/linear_solver.py:765-837`
- **RealSim PD assembly (`M`/`dt²·K` formulation):**
  `RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp:198-226`
- **RealSim NH tet projection (fp64 LBFGS):**
  `RealSim_py/realsim_py/include/Scomponent/integrator/localglobal/energy/elastic/PDNeohookeanTetrahedronEnergyParallel.h:10-91`
- **RealSim `_b = f = M*sn` build:**
  `RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp:277,289-298`
- **RealSim Mass uniform lumping (= matches FBA):**
  `RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/energy/Mass.cpp:12-21`
- **RealSim diag env-var instrumentation:**
  `RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp:60-167`
- **Per-frame divergence scan script (transient):** `/tmp/find_first_diverge.py`
- **Diag npz / bin (transient):**
  `/tmp/fba_diag_demo4_f1.npz`, `/tmp/rs_diag_demo4_f1_*`,
  `/tmp/fba_diag_demo5_f4.npz`, `/tmp/rs_diag_demo5_f4_*`
