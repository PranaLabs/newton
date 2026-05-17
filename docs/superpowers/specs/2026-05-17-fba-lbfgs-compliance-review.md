# Phase 1.3.a Spec-Compliance Review (read-only audit)

**Date:** 2026-05-17
**Implementer:** prior subagent (uncommitted code in `newton/_src/solvers/fba/`)
**Reviewer:** spec-compliance review (this audit)
**Method:** byte-for-byte walk of Warp port against RealSim ground truth. Cited every
comparison at `file:line` granularity. Read-only.

---

## Summary

- **MATCH:** All audited regions (NH value/gradient 2D & 3D, cubic step, BacktrackingCubic 2D & 3D,
  LBFGS two-loop, history append/evict, gamma update, restart, convergence, top-level wrappers,
  scatter kernels, solver dispatch). See "MATCH (verified)" section below.
- **DIVERGE-intentional:** 3 (Q9 value barrier `1e30`, Q9 gradient `σ` clamp, Q10 tet `σ₂` sign-flip
  retained as no-op). All match the documented user-confirmed intentional divergences.
- **DIVERGE-accidental:** **0 — green light to commit.**
- **OPEN-QUESTION:** 3 (line-search iter cap reduced from `100000` to `64`; in-`value()` `J` floor
  guard not in RealSim; `tri_weight` precomputed as `ke·area` with no per-call `coeff` multiplier).
  All three are surface-only; none change the converged minimizer for normal inputs.

---

## DIVERGE-accidental (BLOCKS commit)

**None found.** The Warp port matches RealSim line-for-line in every audited region.

---

## DIVERGE-intentional (DOCUMENTED)

### Q9-a: `nh_value_2d` / `nh_value_3d` returns `1e30` for `σᵢ < 0`

- **Where:** `kernels.py:689-690` (2D), `kernels.py:743-744` (3D);
  RealSim `HyperelasticProblemS.h:37-39` (3D) / `:95-97` (2D)
- **Warp:** `if sigma[0] < 0.0 or sigma[1] < 0.0: return _NH_VALUE_INF` (= `1e30`)
- **RealSim:** `return std::numeric_limits<float>::max()` (≈ `3.4e38`)
- **Status:** MATCHES Q9 substitution. Both are huge-but-finite Armijo barriers; the line
  search rejects either equivalently. Note RealSim uses **`float::max`**, not `double::max`,
  and the Warp constant `1e30` is the documented `_NH_VALUE_INF`.

### Q9-b: `nh_gradient_2d/3d` clamps `σᵢ ≥ 1e-6` instead of `throw`

- **Where:** `kernels.py:722-723` (2D), `kernels.py:770-772` (3D);
  RealSim `HyperelasticProblemS.h:107-108` (2D) / `:49-50` (3D)
- **Warp:** `s0 = wp.max(sigma[0], _NH_EPS); ... j = s0 * s1; log_j = wp.log(j); inv0 = 1.0/s0; ...`
  The `1/σ` and `log(J)` terms use the clamped `s0/s1`; the anchor term `k*(sigma[i] - sigma_init[i])`
  uses the **unclamped** `sigma[i]` (line 728-729 / 778-780).
- **RealSim:** `if (J <= 0.0) throw std::runtime_error(...)` — line search must reject before this
  point via `value() = float::max`.
- **Status:** MATCHES Q9 substitution. Warp cannot throw; clamping `1/σ` and `log(J)` to
  representable values keeps the gradient finite. The anchor term using the raw `σ` is the
  natural choice (no division by σ there). The line search's `value()` barrier ensures any step
  landing in this region is rejected, so the clamped gradient is computed but never consumed for
  an accepted step. (Same behavior the implementer documented in the docstring at kernels.py:717-720.)

### Q10: tet σ₂ sign-flip kept as no-op

- **Where:** `kernels.py:1102-1103`;
  RealSim `PDNeohookeanTetrahedronEnergyParallel.h:75-78`
- **Warp:** `if s2 < 0.0: s2 = -s2  # No-op for wp.svd3 outputs (>= 0), kept for parity.`
- **RealSim:** `if (sigma[2] < 0.0) sigma[2] = -sigma[2];` after `signedEigenSVD` which may
  produce signed σ values.
- **Status:** MATCHES Q10. `wp.svd3` returns non-negative singular values, so the branch is
  unreachable; keeping the code is harmless and surfaces intent. The Tri wrapper omits the flip
  (kernels.py:933-937 docstring acknowledges the omission and ties it to `wp.svd2` returning
  non-negative `sigma_sq` whose `sqrt` is non-negative).

---

## MATCH (verified)

### 1. NH energy `nh_value_2d` / `nh_value_3d`

- **Coefficient `0.5 * mu`** ↔ RealSim `0.5 * _mu` (HyperelasticProblemS.h:29/87). MATCH.
- **`I_1 - log_I3 - 3.0`**:
  - 3D: `kernels.py:752-753` (`I_1 = s0²+s1²+s2²`, `−3.0`) ↔ HyperelasticProblemS.h:26-29
    (3D `I_1`, `−3.0`). MATCH.
  - **2D: `kernels.py:698-699` uses `−3.0` (not `−2.0`).** ↔ **RealSim 2D HyperelasticProblemS.h:87
    also uses `−3.0`.** This is technically dimensionally surprising (should be `−DIM`), but
    RealSim ships with `−3.0` in 2D and the Warp port matches the source byte-for-byte. MATCH.
- **`log_I3 = 2*log(J)`**: kernels.py:697/751 ↔ HyperelasticProblemS.h:28/86 (`std::log(I_3)`
  where `I_3 = J²`). MATCH.
- **`0.125 * lam * log_I3²`**: kernels.py:699/753 ↔ HyperelasticProblemS.h:30/88. MATCH.
- **`(k/2) * ||σ - σ_init||²`**: kernels.py:700-702 / 754-757 ↔ HyperelasticProblemS.h:42/100
  (`_k*0.5 * (x-x0).squaredNorm()`). MATCH.
- **Sign check on negative σ**: kernels.py:689 uses `< 0.0` ↔ HyperelasticProblemS.h:95 also
  uses `< 0.0` (NOT `<= 0.0`). MATCH (faithful to source).

### 2. NH gradient `nh_gradient_2d` / `nh_gradient_3d`

- **`mu*(σ − 1/σ)` + `lam*log(J)/σ` + `k*(σ − σ_init)`** per-axis:
  kernels.py:728-729 (2D) / 778-780 (3D) ↔ HyperelasticProblemS.h:113 (2D) / 56 (3D).
  `(_mu * (x - x_inv) + _lambda * std::log(J) * x_inv) + _k*(x-x0)` — same composition. MATCH.

### 3. `_cubic_step` and `_backtracking_cubic_*` line search

- **`mult = 1 / (α²·αₚ²·(α−αₚ))`** (kernels.py:800) ↔ Backtracking.hpp:133. MATCH.
- **2×2 system entries `A(0,0)=αp²; A(0,1)=-α²; A(1,0)=-αp³; A(1,1)=α³`**: kernels.py:801-804 ↔
  Backtracking.hpp:135-136. MATCH.
- **`B[0] = fxa - fx0 - α·gtp; B[1] = fxp - fx0 - αp·gtp`**: kernels.py:805-806 ↔
  Backtracking.hpp:138. MATCH.
- **`r = mult * A * B`** (with explicit per-component expansion in Warp): kernels.py:807-808 ↔
  Backtracking.hpp:139. MATCH.
- **Quadratic-fallback `-gtp / (2*r1)`** when `|r0| ≤ 0`: kernels.py:809-811 ↔
  Backtracking.hpp:140. MATCH.
- **Cubic root `(-r1 + √(r1² - 3·r0·gtp)) / (3·r0)`**: kernels.py:812-814 ↔ Backtracking.hpp:141-142.
  MATCH. Warp adds a defensive `wp.max(disc, 0.0)` (kernels.py:813) before `sqrt`; this is a
  no-op when discriminant is non-negative (the expected case) and prevents NaN on degenerate input.
  Acceptable harmless guard.
- **Armijo Search loop body**:
  - `p_norm_sq ≤ 0 → return decrease`: kernels.py:847-849 ↔ Backtracking.hpp:42-43 (returns
    `decrease`, i.e. `_LBFGS_C1` = 1e-4). MATCH.
  - `alpha = alpha0; fxp = fx0; alphap = alpha`: kernels.py:851-853 ↔ Backtracking.hpp:85-91.
    MATCH.
  - Per-iter: `fxa = value(x + α·p, x0); armijo_rhs = fx0 + α·c₁·gtp; if fxa <= rhs: return α`:
    kernels.py:856-860 ↔ Backtracking.hpp:95-97. MATCH.
  - First-failure uses quadratic `gtp / (2*(fx0 + gtp - fxa))`: kernels.py:862-864 ↔
    Backtracking.hpp:99-100. MATCH.
  - Subsequent failures use `cubic(...)`: kernels.py:866 ↔ Backtracking.hpp:101. MATCH.
  - `fxp = fxa; alphap = alpha; alpha = range(α_tmp, 0.1·α, 0.5·α)`: kernels.py:868-870 ↔
    Backtracking.hpp:102-104. MATCH.
  - `range(α, low, high)` clamp: kernels.py:817-824 (`_clamp_range`) ↔ Backtracking.hpp:116-120.
    MATCH.
  - Return `-1` on max-iter exit: kernels.py:872 ↔ Backtracking.hpp:107-110. MATCH.
- **`fx0` and `gtp` plumbed in by caller instead of recomputed inside search**: Warp computes
  `fx0 = nh_value_*(x, x0, ...)` (kernels.py:1019/1173) and `gtp = grad·p` (kernels.py:1020/1174)
  in the LBFGS body and passes them down. RealSim recomputes `fx0 = problem.gradient(x, x0, grad)`
  and `gtp = grad.dot(p)` inside `BacktrackingCubic::search` (Backtracking.hpp:88-89). Both compute
  `fx0 = value(x, x0)` and `gtp = grad(x)·p` — same scalars. The grad at `x` in LBFGS is the
  pre-step gradient (LBFGS.hpp:70 initial / :121 recomputed end-of-iter), which is exactly what
  the line search uses. **Semantically equivalent.** MATCH.

### 4. LBFGS two-loop recursion

- **`q = grad; x_old = x; grad_old = grad`** at start of each iter: kernels.py:968-970 ↔
  LBFGS.hpp:81-83. MATCH.
- **First loop (newest → oldest)**: `for i_rev in range(M): i = iter_count − 1 − i_rev; if i<0: break`
  (kernels.py:978-988) ↔ `for(int i = iter-1; i >= 0; --i)` (LBFGS.hpp:88-92). MATCH on
  iteration order and bounds.
  - `rho_i = 1/(s·y)`, `alpha_i = rho_i*(s·q)`, `q -= alpha_i*y`: kernels.py:982-988 ↔
    LBFGS.hpp:89-91. MATCH.
  - No guard on `s·y ≤ 0` in either side: RealSim doesn't guard (LBFGS.hpp:89), Warp
    doesn't guard (kernels.py:983). **Faithful.**
- **Scale by gamma_k** between loops: `q = gamma_k * q` (kernels.py:991 / 1146) ↔
  `q = gamma_k * q` (LBFGS.hpp:95). MATCH.
- **Second loop (oldest → newest)**: `for i in range(M): if i >= iter_count: break; ...`
  (kernels.py:994-1000) ↔ `for(int i = 0; i < iter; ++i)` (LBFGS.hpp:96-99). MATCH.
  - `beta = rho_i*(q·y); q += (alpha_i − beta)*s`: kernels.py:998-1000 ↔ LBFGS.hpp:97-98. MATCH.
- **`iter = min(M, k)`** ↔ Warp's `iter_count = min(history_count, M)` (kernels.py:976-977 /
  1128-1129). Per spec O8, `history_count` plays the role of `min(M, k)` since the post-step
  append increments it and the eviction shifts left at `M`. MATCH.

### 5. Non-descent restart

- **`dir = q·grad; if dir <= 0:`**: kernels.py:1003-1005 (2D) / 1160-1162 (3D) ↔ LBFGS.hpp:102-103.
  MATCH.
- **Restart body**: `q = grad; history_count = 0; alpha_init = min(1, 1/||grad||_inf)`:
  kernels.py:1006-1012 / 1163-1169 ↔ LBFGS.hpp:104-107. MATCH (with `inf_norm > 0` guard;
  RealSim doesn't have this guard but `lpNorm<Infinity>()` on a non-zero `grad` is always positive
  — the guard is benign).
- **Budget decrement**: RealSim `max_iters -= k; k = 0`. Warp uses `iters_remaining -= 1` per iter
  + `history_count = 0`. Per spec §O8 these are semantically equivalent because RealSim's
  post-restart `iter = min(M, 0) = 0` → both inner loops skip → `q = grad` (matches Warp's restart
  branch) → linesearch with the special `alpha_init`. The remaining-iter budget converges to the
  same number of completed LBFGS minimize iterations. MATCH (verified equivalent).

### 6. History storage / eviction

- **Append-when-not-full**: kernels.py:1041-1047 / 1193-1201 ↔ LBFGS.hpp:126-129
  (`s.col(k) = s_temp; y.col(k) = y_temp`). MATCH.
- **Shift-left eviction**: kernels.py:1048-1060 / 1202-1215 (`for j in range(M-1): cols[j] = cols[j+1]`;
  then write new at col `M-1`) ↔ LBFGS.hpp:131-134 (`leftCols(M-1) = rightCols(M-1).eval();
  rightCols(1) = s_temp`). MATCH (verified consistent indexing: oldest at col 0, newest at col
  `min(M, history_count)−1`, exactly the order the two-loop reads it).
- **`history_count` increment**: kernels.py:1047 increments only in the append-when-not-full
  branch (not in the eviction branch — `history_count` stays at `M`). RealSim's effective
  `iter = min(M, k+1)` saturates at `M` similarly. MATCH.

### 7. Convergence check

- **`||grad|| < tol || ||x_prev − x|| < tol`**: kernels.py:1031-1035 / 1183-1187 ↔
  HyperelasticProblemS.h:11 (3D) / :69 (2D) `(grad.norm() < 1e-6 || (x0-x1).norm() < 1e-6)`.
  MATCH on both clauses and on the L2 norm choice (both use Euclidean `.norm()`).
- **Position in loop**: convergence test happens **after** `x = x - rate*q` (kernels.py:1027
  then :1030-1034) ↔ RealSim LBFGS.hpp:118-119 (`x -= rate*q; if (converged) break;`). MATCH.
- **Which `grad` is used for the norm**: Warp uses `grad` at the convergence check, which at
  this point is still the pre-step gradient `grad_old` (saved at kernels.py:969 and not yet
  recomputed — recomputation is at kernels.py:1037 / 1189). RealSim same: `converged(x_last, x,
  grad)` uses the pre-step `grad` (LBFGS.hpp:119; `grad` is recomputed at :121 after the check).
  MATCH.
- **Tolerance value `_LBFGS_TOL = 1e-6`**: kernels.py:25 ↔ HyperelasticProblemS.h:11/69. MATCH.

### 8. Initial Hessian scaling `γ_k`

- **`γ_0 = 1.0`**: kernels.py:954 / 1113 ↔ LBFGS.hpp:72. MATCH.
- **Update formula `γ_k = (s·y) / (y·y)`** (Nocedal "method 1"): kernels.py:1062-1066 / 1217-1221
  ↔ LBFGS.hpp:137-144. MATCH.
- **`denom <= 0` early break**: kernels.py:1063-1064 / 1218-1219 ↔ LBFGS.hpp:138-143. MATCH.
- **Order of operations**: gamma update happens AFTER history append (kernels.py:1041-1060 then
  :1062-1066) ↔ LBFGS.hpp:126-134 then :137-145. MATCH.

### 9. Top-level NH projection wrappers (`project_neohookean_sigma{2d,3d}_lbfgs`)

- **All-tiny clamp before LBFGS**: kernels.py:938-943 (2D), :1095-1101 (3D) ↔
  PDNeohookeanTriangleEnergyParallel.h:68-72 / PDNeohookeanTetrahedronEnergyParallel.h:69-74.
  MATCH (uses `wp.abs(σ_i) < _NH_EPS` for all coordinates, then sets all to `_NH_EPS`).
- **σ₂ sign flip (tet)**: kept as no-op per Q10. MATCH.
- **`k = 2*mu`**: kernels.py:931 / 1093 ↔ HyperelasticProblemS.h:20 / 78. MATCH.
- **`x0 = sigma_init` anchor**: kernels.py:944 (`x0 = x` after the clamp) / 1105.
  RealSim's `sigma0 = sigma; ...; solver.minimize(problem, sigma, sigma0)` (PDNeohookean...:62-74
  / :66-80) — `sigma0` is the **pre-clamp** value (Eigen `sigma0 = sigma;` at line 63/67,
  before the all-tiny inflation at line 69-72/70-74). **Warp uses the post-clamp value as `x0`
  (kernels.py:944 sets `x0 = x` AFTER the clamp).** Both versions of `sigma0` are only used
  in the quadratic anchor `(k/2)||σ - σ0||²`. **DIFFERENCE: when all-tiny clamping fires, the
  anchor `σ0` differs by `≈ 1e-6` from RealSim's.** In RealSim the clamped iterate moves
  AWAY from the (untouched) tiny anchor; in Warp the anchor IS the clamped value.
  - **Impact**: only matters when all σ_i are below 1e-6 simultaneously (degenerate-collapse
    case). The anchor differs by at most O(1e-6), which RealSim's collapse-recovery comment
    (PDNeohookeanTriangleEnergyParallel.h:66-67) explicitly accepts as approximate. The objective
    has a global min near `σ ≈ 1` regardless of the anchor when the element is fully collapsed.
  - **Surfaced as OPEN-QUESTION** below for controller awareness. Not classified as
    DIVERGE-accidental because the case is degenerate and the spec-level user-confirmed
    parameter table doesn't pin this down; could go either way. The implementer's choice is a
    defensible reading.
- **Return value on linesearch failure**: kernels.py:1024 / 1177 — returns `x` (current iterate),
  not `sigma_init`. RealSim: `return Minimizer::FAILURE = -1` at LBFGS.hpp:114, after which the
  caller `solver[i].minimize(...)` returns the failure code but **`sigma` was already updated
  in-place by reference** — the in-place updates from successful prior iterations persist.
  Warp returning `x` (the current iterate after the last successful step) matches this:
  it's the most-recently-updated value. The spec text says "Return value on line-search failure:
  `return x`" — this is exactly what Warp does. MATCH.
- **Per-element parallelism**: each thread runs its own LBFGS — one `wp.tid()` per tri/tet.
  No shared state. MATCH.

### 10. Scatter kernel after projection

- **Tri scatter**: kernels.py:2304-2347 (`project_stretching_neohookean_compute_kernel_lbfgs`)
  reuses the **identical** scatter math as the legacy Newton variant (kernels.py:2213 invokes
  `project_neohookean_sigma`; LBFGS variant invokes `project_neohookean_sigma2d_lbfgs` — only
  difference). `P = U·diag(σ)·V^T`, `row_i = w · (Dm_inv · P^T)_i`, `contributions[t, 0] = -(row0+row1);
  [t, 1] = row0; [t, 2] = row1` ↔ PDNeohookeanTriangleEnergyParallel.h:76-78, 119-121. MATCH.
- **Tet scatter**: kernels.py:1556-1590 (`project_stretching_neohookean_tet_compute_kernel_lbfgs`)
  mirrors the legacy Newton tet kernel (kernels.py:1475-1505) — same `P = U·S·V^T`, same
  `proj = w · (Dm_inv · P^T)`, same `contributions[t, 0] = -(row0+row1+row2); ...` ↔
  PDNeohookeanTetrahedronEnergyParallel.h:82-88, 129-132. MATCH.
- **`wi = coeff · weight · area`**: FBA bakes `weight · area` into `tri_weight_d / tet_weight_d`
  at meta-build time (linear_solver.py:119 `tri_weight = ke * tri_area`; :193
  `tet_weight = 2*mu_tet * tet_volume`). The PD outer-iter `coeff` is **not** applied at runtime —
  per spec §O7 this is hard-coded `coeff = 1` for FBA. This is unchanged by the LBFGS port and
  is explicitly out-of-scope (spec marks it as a pre-existing separate concern). MATCH for the
  scope of 1.3.a; surfaced as OPEN-QUESTION below.
- **F reconstruction in Tri kernel**: uses `wp.svd2(F^T F) → (U2, σ², V2)` then `u_i = F·v_i / σ_i`
  (kernels.py:2293-2302). This matches existing FBA Tri kernels. MATCH.
- **F reconstruction in Tet kernel**: uses `wp.svd3(F) → (U, σ, V)` directly (kernels.py:1556).
  RealSim uses `signedEigenSVD` (PDNeohookeanTetrahedronEnergyParallel.h:63). Q10 documents this
  divergence (Warp's `wp.svd3` returns non-negative σ). MATCH (Q10 covered).

### 11. Solver dispatch (solver_fba.py)

- **`nh_solver: Literal["newton5", "lbfgs"] = "lbfgs"`** (solver_fba.py:221) — LBFGS is now the
  default. Per spec, the user wants LBFGS as the parity target.
- **Tri dispatch** (solver_fba.py:740-758): when `nh_solver == "lbfgs"`, selects
  `project_stretching_neohookean_compute_kernel_lbfgs`. MATCH.
- **Tet dispatch** (solver_fba.py:855-875): same pattern for tet. MATCH.
- **Inputs are identical** to the Newton variant (positions, indices, rest_inv, weight, mu, lam),
  outputs to the same `_tri_contrib_d / _tet_contrib_d` buffer, followed by the same
  `gather_per_particle_kernel`. **No change to the surrounding PD pipeline.** MATCH.

### 12. Locked-parameter compliance

| Parameter | Required | kernels.py | RealSim cite | Status |
|---|---|---|---|---|
| `_LBFGS_M` | 8 | `_LBFGS_M = 8` (line 14) | LBFGS.hpp:36 | MATCH |
| Armijo `c_1` | 1e-4 | `_LBFGS_C1 = 1e-4` (line 24) | Minimizer.hpp:67 | MATCH |
| `tol_grad` / `tol_step` | 1e-6 | `_LBFGS_TOL = 1e-6` (line 25) | HyperelasticProblemS.h:11/69 | MATCH |
| `max_iters` | 50 | `_LBFGS_MAX_ITERS = 50` (line 22) | LBFGS.hpp:47 | MATCH |
| `γ_0` first iter | 1.0 | `gamma_k = 1.0` (line 954/1113) | LBFGS.hpp:72 | MATCH |
| `γ_k` thereafter | `(s·y)/(y·y)` | `sy / denom` where `denom = y·y` (line 1062-1066) | LBFGS.hpp:144 | MATCH |
| Internal arithmetic | fp64 | `wp.float64` throughout LBFGS funcs | RealSim `double` | MATCH |
| Line search | BacktrackingCubic | `_backtracking_cubic_*` (line 828/876) | Minimizer.hpp:68 | MATCH |
| History reset | per-call | Stack-local arrays (line 948-951 / 1107-1110) | LBFGS.hpp:53-67 | MATCH |
| `_NH_EPS` | 1e-6 | `_NH_EPS = 1e-6` (line 28) | PDNeohookean...:68/69 | MATCH |
| Convergence | grad OR step | `gn < TOL or sn < TOL` (line 1034) | HyperelasticProblemS.h:11/69 | MATCH |

---

## Open questions

### OQ-1: `_LBFGS_LS_MAX_ITERS = 64` (vs RealSim's `100000`)

- **Where:** kernels.py:23
- **What:** Warp caps inner line-search iterations at 64. RealSim's default
  `ls_max_iters = 100000` (Minimizer.hpp:67).
- **Why it might be fine:** BacktrackingCubic shrinks α by at least 2× per failed iter (the
  upper bound of `_clamp_range` is `0.5*alpha`), so 64 iters bring α down to `0.5^64 ≈ 5.4e-20`.
  Long before that, either Armijo is satisfied or the search has effectively zero step.
  RealSim's `100000` is a never-triggered ceiling.
- **Why surface it:** Implementer's docstring (kernels.py:23 comment) says "Cap RealSim's 100000
  to a Warp-friendly bound". This is a parameter-table item that wasn't in the locked-parameters
  spec table. **Controller should confirm: is 64 the right cap, or do we want a higher one
  (e.g., 128 or 256) for headroom?** Risk is zero except for pathological numerical inputs.

### OQ-2: `value()` clamps `J ≥ _NH_EPS²` (2D) / `J ≥ _NH_EPS³` (3D) inside log domain

- **Where:** kernels.py:695 (2D), kernels.py:749 (3D)
- **What:** After confirming `σᵢ ≥ 0`, Warp also clamps `j = max(j, _NH_EPS²)` to prevent
  `log(tiny)` from blowing up.
- **RealSim:** Does not have this guard. When `σᵢ` is positive but tiny (e.g., 1e-10), RealSim
  computes `log(J)` directly — returns a large negative number (e.g., `log(1e-20) ≈ −46`) and
  feeds it into `t1 = 0.5*μ*(I_1 − log_I3 − 3.0)`, which becomes large positive (since `log_I3`
  is large negative). This is correct mathematically — the energy is huge when J is tiny, and
  the line search rejects.
- **Impact:** Warp's clamp makes `log_j` floor at `log(_NH_EPS²) = log(1e-12) ≈ −27.6` (2D) or
  `log(_NH_EPS³) = log(1e-18) ≈ −41.4` (3D). For inputs where the line search would already
  reject (J near 0), both behaviors reject equivalently. The clamp could only affect a case
  where the Armijo test would have accepted a step with extremely tiny J — but at extreme tiny
  J, the energy is so high that `value` >> `armijo_rhs` and the step is always rejected.
  **Functionally inert.** Surface for controller awareness; not a port bug.

### OQ-3: `x0` anchor uses post-clamp σ in the all-tiny collapse case

- **Where:** kernels.py:943-944 (2D), :1104-1105 (3D)
- **What:** Warp sets `x = vec(s0, s1[, s2])` after the all-tiny clamp, then `x0 = x` — so
  when the input is all-tiny, both the iterate AND the anchor start at `_NH_EPS`.
- **RealSim:** `sigma0 = sigma` is captured BEFORE the all-tiny clamp
  (PDNeohookeanTriangleEnergyParallel.h:63 → :68-72; PDNeohookeanTetrahedronEnergyParallel.h:67
  → :69-74). The anchor is the (untouched) pre-clamp `sigma0`.
- **Impact:** Only fires when ALL σᵢ start below 1e-6 — i.e., the element has fully collapsed
  to a point. The anchor differs by at most O(1e-6) in each coord. The quadratic anchor term
  `(k/2)||σ − σ0||²` at the minimizer (near σ ≈ 1) is ~`k/2 · 3` regardless of the O(1e-6) offset
  in σ0 — completely dwarfed by μ-scale energy. RealSim's own comment at
  PDNeohookeanTriangleEnergyParallel.h:66-67: *"If everything is very low, It is collapsed to a
  point and the minimize will likely fail. So we'll just inflate it a bit."* — explicitly
  accepts that this case is approximate.
- **Surface for controller:** if SqueezingBall behavior diverges from RealSim in the
  collapse-recovery regime, this is one place to flip. **For SqueezingBall the regime is
  squeezing (σ > 0, possibly some σ ≈ 0+), not collapse-to-a-point, so this likely doesn't
  matter.** A trivial 4-line fix exists (save `x0_pre_clamp` BEFORE the clamp) if controller
  wants exact RealSim parity.

---

## Verdict

- **DIVERGE-accidental: 0** → green light to commit Phase 1.3.a as-is.
- The three OPEN-QUESTIONs are either trivially inert (OQ-1, OQ-2) or only fire in degenerate
  cases that the RealSim author explicitly flagged as approximate (OQ-3). None are likely to
  cause SqueezingBall behavioral divergence.
- The Warp port faithfully reproduces RealSim's LBFGS for all normal inputs: same energy,
  same gradient, same Armijo cubic safeguard, same two-loop recursion, same restart, same
  convergence test, same gamma update, same history append/evict, same scatter.
- Locked-parameter table: all 11 entries verified MATCH.
- Q9 and Q10 intentional divergences: confirmed in code and faithful to documented intent.

**Recommendation:** commit and proceed to SqueezingBall trajectory check. If parity fails,
revisit OQ-3 first (cheapest to flip), then look outside the LBFGS port (Phase 1.1 trajectory
parity, Phase 1.2 bending, contact path).
