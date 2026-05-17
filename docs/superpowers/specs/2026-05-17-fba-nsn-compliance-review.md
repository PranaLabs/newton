# Phase 1.X NSN Spec-Compliance Review (read-only audit)

**Date:** 2026-05-17
**Audited:** FBA NSN port at HEAD = `2aef460a` (working tree clean except for `solver_fba.py` / `kernels.py` Phase 1.3.a LBFGS uncommitted edits; NSN code itself unchanged from Phase 2.2 `7948ffe5` + Phase 2.1 box-cone + Phase 2.2 iter-cap `92bb6a78`).
**RealSim ground truth:** `/home/ziqiu/work/RealSim/RealSim/.claude/worktrees/agent-ae0e87a6/` (three worktree copies identical by md5; canonical content). Files cited:
- `src/Scomponent/lagrange/solvers/NonSmoothNewton.cpp` (424 lines)
- `src/Scomponent/lagrange/solvers/CUDANonSmoothNewton.cpp` (522 lines)
- `src/Scomponent/tools/math/CUDAOperations.cu` (FB CUDA kernels)
- `src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp` (PD outer driver)
- `include/Scomponent/lagrange/LagrangeConstraint.h` (`pene0 = dir·point`)
- `src/Scomponent/collisiondetection/SphereCollision.cpp` (anchor cushion)

**Excluded:** `LiteNonSmoothNewton`, `CUDALiteNonSmoothNewton` (out of scope per ParallelEnv cut, 2026-05-17).

---

## Summary

- **MATCH:** 8 regions — Region 1 (FB unilateral row), Region 2 (FB frictional row), Region 3 (Schur W build / rhs derivation), Region 4 (apply-correction dt² scaling and gather), Region 5 (Coulomb box clamp on tangent axes), Region 6 (per-iter sequence within a single NSN call), Region 8 (`pene0 = n·anchor` formula and per-step set), Region 9 (λ/ω reset per step, precond `dt²·W_n`/`dt·W_t`).
- **DIVERGE-intentional:** 4 — Decision #2 (box cone implemented both sides), Decision #4 (iter-cap=10), S.1 (FBA lexsort), NSN-tolerance early-exit not ported.
- **DIVERGE-accidental:** **2** — see below. Both involve **`λ_n ≥ 0` hard clamp** that RealSim does **not** enforce.
- **OPEN-QUESTION:** 3 — listed at bottom.

If the goal of NSN parity is binary trajectory match, the two DIVERGE-accidental items below should be addressed before SqueezingBall is re-verified. They cannot be ruled out as contributors to the explosion: a positive-only λ_n cuts off the Newton step's ability to retract overshoot in transient deep-penetration iterates, and a `cone = μ·max(0,λ_n)` (rather than RealSim's `cone = μ·λ_n` which can be negative) prevents tangential forces from being clamped to the "physically meaningful" closed interval when normal lambda is transiently negative.

---

## DIVERGE-accidental (BLOCKS contact-demo verification)

### A1. Stage A NSN clamps `λ_n ≥ 0` (RealSim does not)

- **Where:** `newton/_src/solvers/fba/solver_fba.py:1441` (inside the inner Newton loop of `_solve_nsn_unilateral`).
- **FBA says:**
  ```python
  lam = lam + dlam
  np.maximum(lam, 0.0, out=lam)   # line 1441 — clamps λ ≥ 0 every Newton iter
  ```
- **RealSim says** (`src/Scomponent/lagrange/solvers/NonSmoothNewton.cpp:391-394` in `boundConstraintForces`):
  ```cpp
  } else if (_constraint.ctype[i] == RealSim::contact::ContactPair::unilateral) {
      if (_lambda[cid] > _maxforce) _lambda[cid] = _maxforce;
      if (_lambda[cid] < -_maxforce) _lambda[cid] = -_maxforce;
      //            if(_lambda[cid] < 0.0) _lambda[cid] = 0.0;     ← commented out
  }
  ```
  Same `if(_lambda < 0) _lambda = 0` is **commented out** for the friction branch too (line 398). RealSim's `maxforce` is a symmetric `[-_maxforce, +_maxforce]` clamp; the Signorini `λ_n ≥ 0` complementarity is enforced by the Fischer-Burmeister function itself, not by post-hoc clamping.
- **Why it matters:** FBA's Stage A path runs whenever `friction=False`. The Stage A 1-row Schur is symmetric; without the clamp, Newton iterates can briefly push λ_n negative before the FB row's omega correction pulls it back to a complementarity-satisfying point. The forced `np.maximum(lam, 0.0)` truncates each Newton step, breaking the Newton-Raphson update structure and converting Stage A from "FB Newton" back toward a PGS-style projected solver (the exact substitution flagged in `realsim-port-discipline.md` as the failure mode of P2-E v1). Observable: Stage A trajectory will diverge from a RealSim run with `_mu = 0` whenever a contact transient produces a negative dλ.
- **Suggested fix:** Remove line 1441. RealSim's `maxforce` clamp belongs after the inner loop and should be the symmetric `[-cap, +cap]` clip at line 1444 (already present, modulo the units bug in audit T.2). Phase 1.3.c (Stage A vs 3-row μ=0 equivalence test) cannot pass while this clamp differs.

### A2. Stage B NSN forces `λ_n_c = max(0, λ_n_c)` before per-axis box clamp (RealSim does not)

- **Where:** `newton/_src/solvers/fba/solver_fba.py:1574-1579` (inside the inner Newton loop of `_solve_nsn_coulomb`).
- **FBA says:**
  ```python
  # boundConstraintForces: clamp box per contact.
  for c in range(M):
      lam_n_c = max(0.0, lam[3 * c])           # line 1575 — forces λ_n ≥ 0
      lam[3 * c] = lam_n_c                     # line 1576 — writes back the clipped λ_n
      cone = mu[c] * lam_n_c                   # line 1577 — cone derived from clipped λ_n
      lam[3 * c + 1] = float(np.clip(lam[3 * c + 1], -cone, cone))
      lam[3 * c + 2] = float(np.clip(lam[3 * c + 2], -cone, cone))
  ```
- **RealSim says** (`NonSmoothNewton.cpp:395-403` `boundConstraintForces`, friction branch):
  ```cpp
  if (_lambda[cid] > _maxforce) _lambda[cid] = _maxforce;
  if (_lambda[cid] < -_maxforce) _lambda[cid] = -_maxforce;
  //            if(_lambda[cid] < 0.0) _lambda[cid] = 0.0;      ← commented out
  if (_lambda[cid + 1] > (mu * _lambda[cid])) _lambda[cid + 1] = mu * _lambda[cid];
  if (_lambda[cid + 1] < (-mu * _lambda[cid])) _lambda[cid + 1] = -mu * _lambda[cid];
  if (_lambda[cid + 2] > (mu * _lambda[cid])) _lambda[cid + 2] = mu * _lambda[cid];
  if (_lambda[cid + 2] < (-mu * _lambda[cid])) _lambda[cid + 2] = -mu * _lambda[cid];
  ```
  Same CUDA path in `src/Scomponent/tools/math/CUDAOperations.cu:537-547`. RealSim's `cone` is `mu * _lambda[cid]`, which is **signed** — negative when λ_n is transiently negative. The two sequential clamps (`> mu·λ_n` then `< -mu·λ_n`) still bound |λ_t| ≤ μ·|λ_n|, but the polarity is preserved and λ_n itself is never forced to zero.
- **Why it matters:** The audit doc `2026-05-17-fba-realsim-audit-v2.md:187` claims this "matches RealSim", but it only matches the |λ_t| ≤ μ·λ_n cone restriction; it diverges on the `λ_n ≥ 0` enforcement. Same reasoning as A1: the FB-Newton update should be a free dλ followed by symmetric `[-_maxforce, +_maxforce]`. Forcing λ_n ≥ 0 *inside* the inner Newton loop short-circuits the update and converts the Stage B Coulomb solve from FB-Newton toward Signorini-PGS. Concrete impact on SqueezingBall: when the ball hits the squeeze surfaces, transient Newton iters during the friction-active phase will produce dλ that briefly drives λ_n below zero before correcting; FBA clamps this away, while RealSim lets the FB function self-correct. The resulting λ_t cap (`cone = mu * max(0, λ_n)`) is then also clipped tighter than RealSim's `cone = mu * λ_n` (which is negative-allowed), so friction force magnitudes diverge whenever any contact's iterate touches λ_n < 0.
- **Suggested fix:** Replace lines 1574-1579 with the RealSim sequential structure:
  ```python
  for c in range(M):
      lam_n = lam[3 * c]
      cone_hi = mu[c] * lam_n
      cone_lo = -mu[c] * lam_n
      if lam[3 * c + 1] > cone_hi: lam[3 * c + 1] = cone_hi
      if lam[3 * c + 1] < cone_lo: lam[3 * c + 1] = cone_lo
      if lam[3 * c + 2] > cone_hi: lam[3 * c + 2] = cone_hi
      if lam[3 * c + 2] < cone_lo: lam[3 * c + 2] = cone_lo
  ```
  (i.e. leave λ_n untouched by `boundConstraintForces`; the symmetric `lambda_cap` clip below handles maxforce.) Phase 1.3.c equivalence (Stage A vs 3-row μ=0) cannot hold while this divergence is present either.

---

## DIVERGE-intentional (DOCUMENTED, matches user decisions)

### I1. Decision #2 — Rectangular box Coulomb cone (per-axis)

- FBA: per-axis `np.clip(lam_t1, -cone, cone)` / `np.clip(lam_t2, -cone, cone)` at `solver_fba.py:1578-1579`.
- RealSim: `NonSmoothNewton.cpp:400-403` (and CUDA `CUDAOperations.cu:543-546`) — same per-axis box.
- ✓ Both implementations are the rectangular box. The cone magnitude derivation differs only by the A2 `max(0, λ_n)` divergence above.

### I2. Decision #4 — NSN iter cap = 10

- FBA constructor default: `nsn_iterations: int = 10` (`solver_fba.py:217`, committed `92bb6a78`).
- RealSim CudaTests scenes set `constraintsolver.iterations: 10` consistently.
- ✓ Default matches.

### I3. S.1 — FBA contact lexsort

- `solver_fba.py:1118-1129` (`update_contacts`): lexsort by `(normal_z, normal_y, normal_x, shape, particle)` to make atomic-add accumulation deterministic across runs.
- RealSim collects contacts in collision-detection order without sorting.
- ✓ Per S.1 user decision, this is an FBA-only determinism aid. Note: it does **not** introduce a permutation bug — all four contact arrays (`particle_h`, `shape_h`, `body_pos_h`, `normal_h`) are permuted by the same `order` (lines 1126-1129), and Jacobian rows are subsequently built from these same sorted arrays via `_contact_particle_d`, `_contact_normal_d` (line 1163-1164). The lambda/omega/rhs ordering is consistent with J row ordering by construction.

### I4. NSN tolerance early-exit NOT ported

- RealSim's `_constraintlinearsolver->solve_vec(_dlam, _rhs)` returns iterations and has internal `tol` (e.g. `DENSE_CG`/`DENSE_PCR_JACOBI_CUDA` constructors at `NonSmoothNewton.cpp:42, 47, 53, 58, 63, 67` use `_tol`); the outer PD loop has no `‖rhs‖ < tol` early-exit, but the inner linear solver does converge-out.
- FBA's `_solve_nsn_*` runs fixed `max_iters` (no `‖rhs‖` check) and the inner `np.linalg.solve` is a direct Cholesky/LU (no iter cap).
- ✓ Per the 2026-05-17 user decision: keep fixed 10 iter; drift to be quantified in Phase 3.

---

## MATCH (verified)

### Region 1 — `fb_unilateral_row` (FB unilateral row builder)

- **RealSim** `NonSmoothNewton.cpp:332-341` (CPU) and `CUDAOperations.cu:319-326` (CUDA kernel):
  ```cpp
  real pene = _penetration[cid];
  real precondlam = _lambda[cid] * _precond[cid];
  real root = sqrt(pene * pene + precondlam * precondlam);
  _omega[cid] = 1 - pene / root;
  _nonsmooth_compliance[cid] = (1 - precondlam / root) * (_precond[cid] / (dt * dt));
  _h[cid] = -(pene + precondlam - root) + _omega[cid] * (_penetration[cid] + _pene0[cid]);
  ```
- **FBA** `solver_fba.py:43-50`:
  ```python
  pene = penetration
  plam = precond * lam
  root = math.sqrt(pene * pene + plam * plam)
  ...
  omega = 1.0 - pene / root
  compliance = (1.0 - plam / root) * (precond / (dt * dt))
  h = -(pene + plam - root) + omega * (penetration + pene0)
  ```
- ✓ Byte-for-byte match. Signs, dt² scaling, root-form, and the `omega * (penetration + pene0)` correction term all transcribe correctly. The only addition is a defensive `if root < 1e-30: return 0.0, precond/dt², pene0` guard (FBA line 46-47) — RealSim would divide-by-zero; the early-out keeps Stage A from NaN'ing when a contact is exactly touching with zero λ. Numerically inert when `root > 0`.

### Region 2 — `fb_frictional_row` (FB frictional row builder)

- **RealSim** `NonSmoothNewton.cpp:343-378` (CPU) and `CUDAOperations.cu:328-368` (CUDA):
  ```cpp
  // inactive branch (lam_n ≤ 0)
  _omega[cid+1] = 0.0;
  _nonsmooth_compliance[cid+1] = 1.0 / dt;
  _h[cid+1] = -dt * _lambda[cid+1];
  // active branch (lam_n > 0)
  real abspenevel1 = abs(_penetration[cid + 1] / dt);
  real tmp1 = _precond[cid + 1] * (mu * _lambda[cid] - abs(_lambda[cid + 1]));
  real root1 = sqrt(abspenevel1*abspenevel1 + tmp1*tmp1);
  _nonsmooth_compliance[cid+1] = ((root1 - tmp1) / (abspenevel1 + mu*_precond[cid+1]*_lambda[cid] - root1)) * (_precond[cid+1] / dt);
  _h[cid+1] = -dt*dt * _nonsmooth_compliance[cid+1] * _lambda[cid+1] + _pene0[cid+1];
  ```
- **FBA** `solver_fba.py:74-87`:
  ```python
  if lam_n <= 0.0:
      return 0.0, 1.0 / dt, -dt * lam_t
  abspenevel = abs(penetration / dt)
  tmp = precond * (mu * lam_n - abs(lam_t))
  root = math.sqrt(abspenevel * abspenevel + tmp * tmp)
  denom = abspenevel + mu * precond * lam_n - root
  ...
  compliance = ((root - tmp) / denom) * (precond / dt)
  h = -(dt * dt) * compliance * lam_t + pene0
  return 1.0, compliance, h
  ```
- ✓ Byte-for-byte match. Sign on `penetration/dt` then `abs()` matches RealSim's `abs(_penetration[cid + 1] / dt)`. The `(root - tmp) / (abspenevel + μ·precond·λ_n − root)` Newton-inverse-Jacobian denominator transcribes verbatim. `h = -dt²·c·λ_t + pene0` matches. The `abs(denom) < 1e-30` early-out in FBA is again defensive; RealSim divides-by-zero.

### Region 3 — Schur W system build

- **RealSim** `NonSmoothNewton.cpp:124` and CUDA `CUDAOperations.cu:399-411` (kernel `kernel_NonSmoothDelasus`):
  ```cpp
  out[i*n+j] = omega[i] * omega[j] * W[i*n+j];
  if (i == j) out[i*n+j] += compliance[i];
  ```
  i.e. `delasus = (ω ωᵀ) ⊙ W + diag(compliance)`.
- **FBA** `solver_fba.py:1431, 1562`:
  ```python
  A_schur = (omega[:, None] * omega[None, :]) * W + np.diag(compliance)
  ```
- ✓ Match — outer-product form, in-place `+diag(compliance)`.

- **RealSim** rhs derivation (`NonSmoothNewton.cpp:126-134`):
  ```cpp
  // tmp_c1 = ω · λ
  // tmp_dof = J · (ω · λ)
  // b_reshaped += dt² · J · (ω · λ)
  // x = A⁻¹ · b_reshaped
  // tmp_c1 = Jᵀ · x         (= J · x in CSR-by-DoF terms)
  // tmp_c2 = ω · (Jᵀ · x)
  // rhs = (1/dt²) · (h − ω · Jᵀ · x)
  ```
- **FBA** `solver_fba.py:1517, 1561-1565`:
  ```python
  penetration = -r + (dt * dt) * (W @ (omega * lam))
  ...
  A_schur = (ω·ωᵀ) ⊙ W + diag(compliance)
  J_x = (pene0 - r) + (dt * dt) * (W @ (omega * lam))
  rhs = (1.0 / (dt * dt)) * (h - omega * J_x)
  ```
  where `r = pene0 − α·J·x_unc` (from `_compute_contact_residual_friction`, line 1342) and `W = J·A⁻¹·Jᵀ`.
- ✓ Algebraic equivalence:
  - `pene0 − r = α·J·x_unc`, so `J_x = α·J·x_unc + dt²·W·(ω·λ) = J·x_corrected`. The dt² placement is identical.
  - `penetration = −r + dt²·W·(ω·λ) = J·x_corrected − pene0`. Matches RealSim's `_penetration = J·q − pene0` at line 117-118 because in RealSim the position `q` arriving at `build()` is the previously-corrected position from the last `applyConstraintCorrection`. FBA emulates this without explicitly applying the correction by injecting `dt²·W·(ω·λ)` into the penetration formula.
  - `rhs = (1/dt²)·(h − ω·J_x)` is identical to RealSim's `_rhs = (1/dt²)·(_h − ω·J_x)`.

- FBA stores `A_FBA = A_R / dt²` and `λ_FBA = λ_R / dt²`. Both factors cancel in observable quantities (the residual `r`, the correction `Δq = A⁻¹·Jᵀ·(dt²·ω·λ)`). Note: the FBA `precond[3c] = dt² · w_n` at line 1511 matches RealSim's `precond[cid] = dt²·diag[cid]` at `NonSmoothNewton.cpp:191, 195`. `precond[3c+1] = dt · w_t` matches RealSim line 196-197.

### Region 4 — λ correction application

- **RealSim** `applyConstraintCorrection` at `NonSmoothNewton.cpp:151-171` / `CUDANonSmoothNewton.cpp:448-482`:
  ```cpp
  _lambda += _dlam;
  tmp_c1 = ω · _lambda;
  tmp_dof = J · tmp_c1;       // CSR SpMV
  b += dt² · tmp_dof;
  x = A⁻¹ · b;                // full re-solve
  boundConstraintForces();
  ```
- **FBA** `solver_fba.py:920-940` (Stage B) and `:956-975` (Stage A):
  ```python
  lam, omega_last, lam_apply = self._solve_nsn_coulomb(...)
  ...
  correction = self._apply_lambda_correction_friction(lam_apply)
  wp.launch(accumulate_vec3_kernel, ..., inputs=[correction], outputs=[self._x_cur])
  ```
  where `lam_apply = dt² · ω_last · lam` (line 1583) and `_apply_lambda_correction_friction(lam_apply)` computes `A_FBA⁻¹ · Jᵀ · lam_apply` (`solver_fba.py:1612-1633`, dispatching to `linear_solver.py:1294-1352` `apply_lambda_correction_isodof`).
- ✓ Match. `Δq = A_FBA⁻¹·Jᵀ·(dt²·ω·λ) = dt²·A⁻¹·Jᵀ·(ω·λ)` (since `A_FBA = A_R / dt²` and `λ_FBA = λ_R / dt²` cancel exactly). The gather `Jᵀ·lam_apply` is done via `gather_jt_lambda_kernel` over particle-centered CSR (deterministic, no atomic adds — confirmed `linear_solver.py:1337-1349`). Ordering of `ω · λ` (apply ω BEFORE the gather) matches RealSim's `parallelDiagMV(tmp_c1, ω, λ)` then `parallelSpMV(tmp_dof, J, tmp_c1)` at `NonSmoothNewton.cpp:161-162`.

### Region 5 — Box-clamp per Coulomb axis

- ✓ See I1 above (intentional match on box shape). The `λ_n ≥ 0` flavoring is split out as DIVERGE-accidental A2.

### Region 6 — Iteration loop structure & convergence

- **RealSim outer driver** `LocalGlobalSolver.cpp:174-190`:
  ```cpp
  for (unsigned int nb_iter = 0; nb_iter < _maxIter; nb_iter++) {
      _b = f;
      localProjection(_b, _nextpos, _dt*_dt);
      globalSolve(_nextpos, _b);   // → build → solve → applyConstraintCorrection (one Newton step)
  }
  ```
- **RealSim build/solve/correction sequence** `LocalGlobalSolver.cpp:252-259`:
  ```cpp
  _constraintsolver->build(x, b);
  _constraintsolver->solve();
  _constraintsolver->applyConstraintCorrection(x, b);
  ```
  i.e. **one** NSN Newton step per PD outer iter. Across `_maxIter` PD outer iters, NSN accumulates `_maxIter` Newton steps.
- **FBA** `solver_fba.py:666-975` outer loop and `:1412/:1515` inner Newton loop:
  - Per PD outer iter (`for _k in range(self.iterations)`): runs ONE NSN call which internally does `nsn_iterations` Newton steps (`for _ in range(max_iters)` inside `_solve_nsn_coulomb`).
  - λ and ω persist across PD outer iters via `_lam_*_persistent` / `_omega_*_persistent` (lines 926-931, 961-966) — matching RealSim's per-step `cuda_lambda.setZero` once at `prepare_gpu` (`CUDANonSmoothNewton.cpp:365-366`).
- ✓ Per-iter sequence within a single NSN call matches RealSim: penetration recompute → FB row eval → A_schur build → rhs build → dλ solve → λ accumulate → boundConstraintForces (modulo A1/A2 above).
- ✗ Net Newton-step count differs: FBA does `iterations × nsn_iterations` Newton steps total; RealSim does `_maxIter`. Within FBA's inner loop, `r = pene0 − α·J·x_unc` is held constant (line 914), so the inner iters converge to a fixed point of the **frozen** local-projection. RealSim instead re-runs local-projection between Newton steps. This is the **Phase 1.3.g** known DIVERGE-accidental queued for measurement; not re-flagged here.

### Region 7 — Position-LCP coupling

- The audit `2026-05-17-fba-realsim-audit-v2.md` Component W marked this DIVERGE-accidental queued for Phase 1.3.g. Implementation matches the audit description: FBA's split-form `penetration = -r + dt²·W·(ω·λ)` is exact when local-projection output is linear in `x` (ARAP/Corot have piecewise-linear local proj; NH-LBFGS is fully nonlinear). The audit verdict stands.

### Region 8 — `_pene0` semantics

- **RealSim** `include/Scomponent/lagrange/LagrangeConstraint.h:54-64`:
  ```cpp
  pene0[cstId] = cinfo._dir.dot(cinfo._point);
  ```
  Per-shape `_point` includes the 0.01m sphere/cylinder cushion (`SphereCollision.cpp:121, 127-129`: `pair.addBasis(1, index, coeff, orient, point - 0.01 * orient)` for the normal row).
- **FBA** `solver_fba.py:1160` (`update_contacts`):
  ```python
  offset_h[c] = float(np.dot(n, world_anchor))
  ```
  and per-step at `:917-919` (Stage B build):
  ```python
  pene0_b[0::3] = self._contact_offset_h[:M]
  pene0_b[1::3] = self._contact_tangent1_offset_h[:M]
  pene0_b[2::3] = self._contact_tangent2_offset_h[:M]
  ```
- ✓ Formula matches (`pene0 = dir · anchor`). Per-step refresh from `update_contacts` matches RealSim's per-step `initPenetration` call inside `prepare_gpu` at `CUDANonSmoothNewton.cpp:347`.
- ✗ Audit Y known gaps stand (not re-flagged): no 0.01m sphere/cylinder normal-row cushion (Phase 1.3.e), plane tangent uses current-step projection (Phase 1.3.b).

### Region 9 — Pre-step buffer setup (caches, lexsort, init λ, init ω)

- **FBA per-step reset** `solver_fba.py:624-640`:
  ```python
  M = self._contact_count
  if M > 0:
      ...
      self._lam_unilateral_persistent.fill(0.0)
      self._lam_coulomb_persistent.fill(0.0)
      self._omega_unilateral_persistent.fill(0.0)
      self._omega_coulomb_persistent.fill(0.0)
  ```
  Also `_invalidate_schur_cache` resets λ/ω to None (line 1023-1027), called from `update_contacts` (line 1074) which `step()` invokes before the persistent-reset block. So λ is reset to zero every step (either via `fill(0.0)` or fresh allocation).
- **RealSim per-step reset** `CUDANonSmoothNewton.cpp:365-366` (inside `prepare_gpu` which runs once per step via `LocalGlobalSolver::prepare`):
  ```cpp
  cuda_lambda.setZero(_num_constraint);
  cuda_dlambda.setZero(_num_constraint);
  ```
- ✓ Match — λ resets per step (per frame), accumulates across PD outer iters within a step.
- Lexsort: see I3.
- Precond: see Region 3.

### Region 10 — Stage A (unilateral-only) path

- ✓ Algebraic correspondence with Stage B at μ=0 holds modulo A1/A2:
  - Stage A's `_solve_nsn_unilateral` solves `M × M` Schur with normal-only rows.
  - Stage B with `mu = 0` and all `lam_n ≤ 0` initially has tangent rows in the inactive branch (compliance = `1/dt`, h = `−dt·lam_t`, ω = 0), so the 3M × 3M Schur reduces:
    - For tangent rows: `A_schur[t,t] = compliance = 1/dt`, off-diagonals `A_schur[t,·] = 0·ω_other` (since ω_t = 0). So tangent rows decouple → `dλ_t = h_t / compliance_t = −dt²·lam_t`, which exactly cancels `λ_t` on the next iter (returns to zero).
    - Normal rows: `A_schur[n,n] = ω_n²·W_nn + compliance_n`, h_n same form. Stage A's `A_schur[n,n] = ω_n·ω_n·W_nn + compliance_n` matches exactly.
  - So at μ=0 the two paths yield identical normal λ trajectories — **but only after the A1/A2 fix**. Pre-fix, the `np.maximum` clamp may activate at different iters between paths.
- Phase 1.3.c test cannot pass until A1/A2 are fixed. Once they are fixed, the equivalence holds by construction.

---

## Open questions

### O1. Effective Newton-step count `iterations × nsn_iterations`

- FBA's defaults are `iterations=10` (PD outer) and `nsn_iterations=10` (NSN inner), so 100 Newton-build evaluations per step. RealSim does 10 per step (1 per PD outer iter, _maxIter=10).
- Within a single PD outer iter, the inner loop holds `r = pene0 − α·J·x_unc` constant (since x_unc is updated only once, at the end). So inner iters 2-10 converge a frozen subproblem.
- For purely linear local-projections (ARAP, Corot are piecewise-linear), the converged inner subproblem is the same as RealSim's outer fixed point. For Neo-Hookean, the local-projection is highly nonlinear, and the inner loop's converged answer may overshoot what RealSim's per-outer-iter Newton step would produce.
- **Question:** Should the inner loop be capped at 1 (matching RealSim's outer-driven Newton) until LBFGS NH lands and we can re-measure? Or accepted as a "more accurate per-outer-iter" deviation? Currently flagged Phase 1.3.g but the default 10 commit may have moved the trajectory further from RealSim.

### O2. `lambda_cap` per-iter vs post-loop

- FBA applies `np.clip(lam, -cap, +cap)` **once** after the inner loop (line 1444, 1582). RealSim applies its analog (`maxforce`) **every** PD outer iter via `boundConstraintForces`. Since FBA's inner loop is in-step (one NSN call per outer iter), the FBA `lambda_cap` does fire once per outer iter too. But it does NOT fire between inner Newton iters.
- **Question:** Should `lambda_cap` move INTO the inner loop (run per Newton iter, alongside box-cone clamping) to match RealSim exactly? Coupled with the audit T.2 units bug; resolve together in Phase 1.3.d.

### O3. ω_warm-start carryover after Schur cache invalidation

- `_invalidate_schur_cache` sets `_omega_*_persistent = None` (line 1026-1027). When `update_contacts` fires on every step (line 1074), this drops the previous step's ω. In RealSim, `_omega` is a member that survives across PD outer iters but is **rebuilt every `build()` call** before being used (line 121 then line 124), so any "carryover" ω from previous step is overwritten on first call of the new step. The only place initial ω matters is FBA line 1517 (`penetration` formula): on the very first Newton iter of the very first PD outer iter of a step, `ω = 0` ⟹ `penetration = −r` ⟹ no kinematic correction yet — which is consistent with starting from x_unc.
- **Question:** Is the per-step `_omega_*_persistent = None` deliberate, or should ω carry the LAST iter's ω from the previous step (so that across-step λ-warm-start would also need carryover)? Currently λ is forcibly zeroed per step too (line 626-636) so this is internally consistent; just worth confirming with the user that across-step warm-start was deliberately dropped. Phase 1.3.f tangentially relates.

---

## Verification methodology

- Ground-truth = RealSim source at `agent-ae0e87a6` worktree, all three worktree copies confirmed md5-identical for `NonSmoothNewton.cpp` and `CUDANonSmoothNewton.cpp`.
- Cross-checked CPU CPU↔GPU parity inside RealSim: `nonsmoothFischerBurmeisterUnilateralFunction` (CPU `NonSmoothNewton.cpp:332-341`) vs `kernel_NonSmoothFunctionFB` (CUDA `CUDAOperations.cu:299-369`) — confirmed identical formulas.
- All FBA citations are against `solver_fba.py` at the HEAD code state (`2aef460a` working-tree NSN code unchanged).
- No edits to source or tests; this is a strict read-only audit per the prior LBFGS review pattern.

---

## Re-review 2026-05-17 (post A1/A2 fix)

Read-only re-audit of `newton/_src/solvers/fba/solver_fba.py` after the controller applied the A1 and A2 fixes from the prior section. Ground truth unchanged: RealSim `agent-ae0e87a6` worktree, `NonSmoothNewton.cpp:380-410` (`boundConstraintForces`).

### A1 verification

**Status: fixed correctly.**

- `solver_fba.py:1440-1447` (`_solve_nsn_unilateral`, end of inner Newton loop):
  ```python
  lam = lam + dlam
  # No lam_n >= 0 clamp here. RealSim's boundConstraintForces
  # (NonSmoothNewton.cpp:391-394, unilateral branch) explicitly
  # comments out `if(_lambda[cid] < 0.0) _lambda[cid] = 0.0;` —
  # FB-Newton allows transient negative lambda inside the inner
  # solve and lets the FB residual converge. The previous
  # np.maximum(lam, 0.0) clamp converted Stage A toward
  # Signorini-PGS (see realsim-port-discipline P2-E v1 failure).
  ```
- The `np.maximum(lam, 0.0, out=lam)` line documented at the old `solver_fba.py:1441` is gone. Inner loop now ends at `lam = lam + dlam` with no further clamp.
- Post-loop `lambda_cap` clip survives unchanged at `solver_fba.py:1449-1450`:
  ```python
  if self.lambda_cap is not None:
      np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)
  ```
  Symmetric `[-cap, +cap]` — consistent with RealSim's `_maxforce` being symmetric (`NonSmoothNewton.cpp:392-393`).
- Comment cites `NonSmoothNewton.cpp:391-394` (correct line numbers — verified) and references the P2-E v1 failure pattern from `realsim-port-discipline.md` (correct match).

### A2 verification

**Status: fixed correctly.**

- `solver_fba.py:1579-1598` (`_solve_nsn_coulomb`, per-contact bound block) now reads:
  ```python
  # boundConstraintForces (NonSmoothNewton.cpp:395-403, friction
  # branch): signed cone clamp. lam_n is NOT clamped to >= 0 inside
  # the inner loop (the `if(_lambda[cid] < 0.0) = 0` line is
  # explicitly commented out in RealSim). Tangent clamp is two
  # independent if-statements applied in order; when lam_n < 0 this
  # saturates lam_t to -mu*lam_n (positive) rather than producing a
  # degenerate np.clip with low > high. Mirroring RealSim's exact
  # mutation order:
  for c in range(M):
      lam_n_c = lam[3 * c]  # SIGNED, per RealSim
      upper = mu[c] * lam_n_c
      lower = -mu[c] * lam_n_c
      if lam[3 * c + 1] > upper:
          lam[3 * c + 1] = upper
      if lam[3 * c + 1] < lower:
          lam[3 * c + 1] = lower
      if lam[3 * c + 2] > upper:
          lam[3 * c + 2] = upper
      if lam[3 * c + 2] < lower:
          lam[3 * c + 2] = lower
  ```
- Cross-check against RealSim `NonSmoothNewton.cpp:399-403`:
  ```cpp
  //            if(_lambda[cid] < 0.0) _lambda[cid] = 0.0;
  if (_lambda[cid + 1] > (mu * _lambda[cid])) _lambda[cid + 1] = mu * _lambda[cid];
  if (_lambda[cid + 1] < (-mu * _lambda[cid])) _lambda[cid + 1] = -mu * _lambda[cid];
  if (_lambda[cid + 2] > (mu * _lambda[cid])) _lambda[cid + 2] = mu * _lambda[cid];
  if (_lambda[cid + 2] < (-mu * _lambda[cid])) _lambda[cid + 2] = -mu * _lambda[cid];
  ```
- Verbatim correspondence:
  - `lam_n_c = lam[3*c]` ≡ `_lambda[cid]` (signed, untouched).
  - `upper = mu[c] * lam_n_c` ≡ `mu * _lambda[cid]`.
  - `lower = -mu[c] * lam_n_c` ≡ `-mu * _lambda[cid]`.
  - First two `if`s mutate `lam[3*c+1]` (≡ `_lambda[cid+1]`) in the SAME order as RealSim (`> upper` then `< lower`); the second `if` is evaluated against the (possibly mutated) value of `lam[3*c+1]`, so when `lam_n < 0` (upper < 0 < lower) the first `if` writes `upper` (negative), then the second `if` sees the mutated value still less than `lower`, and overwrites with `lower` (positive). Final state: `lam_t = -mu*|lam_n|` — RealSim's saturate-to-`-mu*lam_n` semantic preserved.
  - Same for `lam[3*c+2]` rows.
- Post-loop `lambda_cap` clip preserved at `solver_fba.py:1600-1601`.
- Comment cites `NonSmoothNewton.cpp:395-403` and explains the `lam_n < 0` saturation behavior. ✓

### Newly discovered DIVERGE-accidental

#### A3. Stage A correction guard is **signed**, not magnitude (post-A1 regression vector)

- **Where:** `solver_fba.py:967` (in `step()`, Stage A path):
  ```python
  if np.any(lam_apply > 1e-15):
      correction = self._apply_lambda_correction(lam_apply)
      wp.launch(accumulate_vec3_kernel, ...)
  ```
- **FBA says:** The guard `lam_apply > 1e-15` is a one-sided test. After the A1 fix, `lam_apply = dt² · ω · lam` may now contain negative entries (when transient λ_n is negative); the guard skips applying the correction if every entry is `≤ 1e-15`, including the case where all entries are negative.
- **Stage B counterpart at `solver_fba.py:932`** uses `np.any(np.abs(lam_apply) > 1e-15):` — magnitude-based. Asymmetric.
- **RealSim says** (`NonSmoothNewton.cpp:151-171`, `applyConstraintCorrection`): no gating at all — the correction is always applied. The `_lambda += _dlam` accumulates and `b += dt² · J · (ω · _lambda)` runs unconditionally each PD outer iter.
- **Why it matters:** Pre-A1, Stage A always had `λ ≥ 0` (`np.maximum` clamp), so `lam_apply = dt² · ω · lam ≥ 0` and the one-sided guard fired correctly whenever any λ was non-trivial. After A1 removed the clamp, λ_n can be transiently negative; if all M λ_n values happen to be negative at the end of an inner-loop block, the guard skips the correction entirely — different from RealSim (which always applies) and also different from FBA's own Stage B (which applies based on magnitude).
- **Observable impact:** Visible only when Stage A path is used (no friction, `friction=False`) AND the Stage A inner loop converges to all-negative λ. After A1 this can happen during a transient that RealSim would let self-correct over multiple PD outer iters. Could show up on a unilateral-only scene (e.g. a frictionless contact demo) as a missed retraction of an overshoot.
- **Fix:** Change line 967 to `if np.any(np.abs(lam_apply) > 1e-15):` to match Stage B and RealSim's "always apply" intent (with a magnitude floor for FP-noise short-circuit). Single-line change. Defer or apply now per the controller's discretion.

### Re-confirmed open questions (no change)

- **O1** (effective Newton-step count `iterations × nsn_iterations` = 100 in FBA vs 10 in RealSim) — still open, queued Phase 1.3.g.
- **O2** (`lambda_cap` post-inner-loop vs RealSim's per-PD-outer-iter `_maxforce`) — still open, queued Phase 1.3.d. Note: now that the box-cone clamp at A2 is **inside** the inner Newton loop, an analogous in-loop `lambda_cap` would be the strictly RealSim-matching choice. Defer per audit T.2.
- **O3** (`_invalidate_schur_cache` resets λ/ω) — still open, queued Phase 1.3.f. Per-step reset block at `solver_fba.py:624-640` is internally consistent with the `_invalidate_schur_cache` reset at line 1024-1027.

### Re-scan results (other regions — no new findings)

- **3a. Stage A body (`_solve_nsn_unilateral` lines 1366-1452):** ω/penetration recompute per inner iter (lines 1413-1416), `A_schur = (ω⊗ω) ⊙ W + diag(c)` (line 1431), `rhs = (1/dt²)·(h − ω·J_x)` (line 1434), `lam += dlam` (line 1440). All sub-blocks match the friction path at μ=0 sub-block (Region 10 reasoning). MATCH (modulo A3 above).
- **3b. λ-correction apply for Stage A (`_apply_lambda_correction` + `apply_lambda_correction_isodof`):** `solver_fba.py:1695-1729` + `linear_solver.py:1294-1352`. Sign-symmetric in `lam` (handles negative λ correctly via single Cholesky solve). MATCH.
- **3c. Per-step preparation (`solver_fba.py:619-640`):** λ_reset, ω_reset, penetration cleared via per-iter formula. The kinematic anchor shift at lines 609-617 applies once before the PD outer loop, matching RealSim's per-step `initPenetration`. MATCH.
- **3d. Schur W build (`linear_solver.py:952-1130`):** `W = J · A⁻¹ · Jᵀ` (docstring `solver_fba.py:962` and isodof assembly), single build per step at first PD outer iter, reused for all subsequent iters within the step (Task L cache, `solver_fba.py:901-912, 943-952`). Stage A and Stage B use independent layouts (M×M vs 3M×3M) but consistent J^T sign convention. MATCH.
- **3e. Penetration recompute per inner iter:** ✓ Both `_solve_nsn_unilateral:1415` and `_solve_nsn_coulomb:1523` recompute `penetration = -r + (dt²)·(W @ (ω·lam))` at the top of each inner iter using current λ,ω. RealSim equivalent: `build()` recomputes `_penetration = J·q − pene0` once per PD outer iter (only); FBA's per-inner-iter recompute is the "frozen-local-projection inner solve" pattern flagged in Region 6 / Phase 1.3.g. Not a new finding.
- **3f. ω application order:** rhs uses current iter's ω at `solver_fba.py:1434, 1571`. ω is overwritten in place per inner iter (lines 1416, 1525). MATCH.
- **3g. Frame-edge λ persistence:** `_lam_*_persistent` reset per step (lines 624-640) and after `_invalidate_schur_cache` (lines 1024-1027). Within a step, λ accumulates across PD outer iters. Matches RealSim's per-step `cuda_lambda.setZero` in `prepare_gpu`. No change from prior review.
- **3h. lambda_cap clip:** still post-inner-loop (`solver_fba.py:1449-1450, 1600-1601`). RealSim applies per PD outer iter inside `boundConstraintForces`. Same as O2 — re-confirmed.
- **3i. Sign/scaling checks:** `J_x = (pene0 − r) + dt²·W·(ω·λ)`, `rhs = (1/dt²)·(h − ω·J_x)`, `Δq = dt²·A⁻¹·Jᵀ·(ω·λ)` — all signs match RealSim transcription from Region 3/4. No new sign issues.

### Summary

- **A1 fix:** accepted.
- **A2 fix:** accepted.
- **New DIVERGE-accidental:** 1 (A3 — Stage A correction-apply guard is signed, exposes negative-λ states after the A1 fix).
- **Combined verdict:** **yellow light**. A1+A2 are correct. A3 is a side-effect of A1 — the prior `np.maximum(lam, 0.0)` clamp accidentally masked an asymmetric guard at the correction-apply site. The fix is trivial (`lam_apply > 1e-15` → `np.abs(lam_apply) > 1e-15` at `solver_fba.py:967`), and consistent with both Stage B's existing guard at line 932 and RealSim's "always apply" semantic. Recommend either:
  1. Apply A3 alongside A1+A2 in the same commit (preferred — keeps NSN parity changes atomic).
  2. Apply A3 in a follow-up commit before running SqueezingBall verification.
  Phase 1.3.c equivalence (Stage A vs 3-row μ=0) cannot be cleanly demonstrated with A3 outstanding because Stage A may silently no-op when Stage B (μ=0) would still apply the correction.

---

### Final review 2026-05-17 (post A3 fix)

Read-only re-audit of `newton/_src/solvers/fba/solver_fba.py` after the controller applied the A3 fix. Ground truth unchanged: RealSim `agent-ae0e87a6` worktree, `NonSmoothNewton.cpp` (`boundConstraintForces` lines 380-410, `applyConstraintCorrection` lines 151-171, `nonsmoothFischerBurmeisterFrictionalFunction` lines 343-378).

**A3 verification:** **fixed correctly.**

- `solver_fba.py:967-973` (Stage A correction-apply branch) now reads:
  ```python
  # Symmetric magnitude guard, matching Stage B above and
  # RealSim's NonSmoothNewton.cpp:151-171 (always apply
  # correction). Pre-A1-fix the one-sided guard was masked
  # by the np.maximum(lam, 0.0) clamp; removing that clamp
  # exposes transient all-negative lam_apply, which a
  # one-sided guard would silently skip.
  if np.any(np.abs(lam_apply) > 1e-15):
      correction = self._apply_lambda_correction(lam_apply)
      wp.launch(accumulate_vec3_kernel, ...)
  ```
- Guard form `np.any(np.abs(lam_apply) > 1e-15)` is identical to Stage B at `solver_fba.py:932`. ✓
- Comment cites the rationale: post-A1 transient negative `lam_apply` would have been silently dropped by the prior `lam_apply > 1e-15` one-sided test, and cross-references RealSim's `NonSmoothNewton.cpp:151-171` "always apply" semantic. ✓

**Ripple check:** **no further λ-sign issues found.**

Grep results across `solver_fba.py` (full file):

- `lam <= 0` / `lam > 0` patterns:
  - `solver_fba.py:75` — `if lam_n <= 0.0:` in `fb_frictional_row`. **NOT a `λ ≥ 0` assumption**; this is the verbatim port of RealSim's inactive-friction branch at `NonSmoothNewton.cpp:348` (`if (_lambda[cid] <= 0.0) {`). Same `<= 0.0` polarity, same branch structure: inactive returns `(0.0, 1.0/dt, -dt·lam_t)`, active returns the FB-root form. Behavior on transient negative λ_n matches RealSim by construction. ✓
  - `solver_fba.py:71` — comment `Active contact (lam_n > 0): smooth complementarity ...` is documentation of the FB inactive/active dichotomy. ✓
  - `solver_fba.py:289` — `if lam <= 0` is the **Lamé second parameter** in `SolverFBA.__init__` (constitutive material); unrelated to contact λ. ✓
- `np.clip` / `np.maximum` near λ:
  - `solver_fba.py:1456, 1607` — `np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)`: **symmetric**, sign-preserving. Matches RealSim's `if (_lambda > _maxforce) ... if (_lambda < -_maxforce) ...` at `NonSmoothNewton.cpp:392-393, 396-397`. ✓
  - `solver_fba.py:1416` — `np.where(np.abs(diag_W) > 1e-12, np.maximum(diag_W, 1e-12), 1.0)`: clamps **W diagonal** (geometric), not λ. Safe. ✓
- Magnitude sentinels:
  - `solver_fba.py:46, 82` — `root < 1e-30` / `abs(denom) < 1e-30`: FB-row numerics, magnitude-based (no λ-sign assumption). ✓
  - `solver_fba.py:932, 973` — both correction-apply guards now use `np.abs(lam_apply) > 1e-15`. ✓
  - `solver_fba.py:1678, 1753` — `abs(lam[row]) < 1e-15` in the fallback `_apply_lambda_correction_friction` and `_apply_lambda_correction` loops: magnitude-based, so a negative-but-significant λ still applies. ✓
- Schur W build (`linear_solver.py:952-1130`): **purely geometric** (uses J and A only; no λ reference). Confirmed by grep of `linear_solver.py` — the only `lam` references are in the `apply_lambda_correction_isodof` helper (which is sign-linear in λ via the gather→Cholesky-solve pipeline) and accompanying docstrings. ✓
- Compliance/omega computation in `fb_unilateral_row` and `fb_frictional_row`: byte-for-byte verbatim port of RealSim `NonSmoothNewton.cpp:332-378`, including the `lam_n <= 0` inactive branch and the `(root - tmp) / (abspenevel + μ·precond·λ_n − root)` active-branch Newton-inverse-Jacobian denominator. Handles transient `λ_n < 0` exactly as RealSim does. ✓
- `_apply_lambda_correction*` paths (`solver_fba.py:1611-1773`): both the cached `_A_inv_Jt_d` accumulation (kernel `accumulate_lambda_correction_kernel`) and the isodof `apply_lambda_correction_isodof` path compute `correction = A⁻¹·Jᵀ·λ` as a sign-linear operation. The fallback loop skip is `abs(lam[row]) < 1e-15` — magnitude-based. No λ-sign assumption. ✓
- Stage A vs Stage B parity post-fix:
  - Stage A inner loop (`_solve_nsn_unilateral:1418-1453`): A1-fix in place, no `np.maximum(lam, 0.0)` clamp inside loop.
  - Stage B inner loop (`_solve_nsn_coulomb:1527-1604`): A2-fix in place, signed `lam_n_c` with two-independent-if tangent clamp (no `max(0, lam_n)` and no `np.clip` with degenerate bounds).
  - Stage A correction guard (line 973): A3-fix in place, `np.abs(...) > 1e-15`.
  - Stage B correction guard (line 932): already `np.abs(...) > 1e-15`.
  - Post-loop `lambda_cap` symmetric clip: both stages (lines 1456, 1607).
  - All Stage A vs Stage B asymmetries documented in the prior re-review are now closed. ✓

**A1/A2 still in place:** **confirmed, no regression.**

- A1 (`_solve_nsn_unilateral`): `solver_fba.py:1446` ends inner loop with `lam = lam + dlam`; the previous `np.maximum(lam, 0.0, out=lam)` line is gone. Comment at lines 1447-1453 documents the removal and cites `NonSmoothNewton.cpp:391-394`. ✓
- A2 (`_solve_nsn_coulomb`): `solver_fba.py:1585-1604` uses signed `lam_n_c = lam[3*c]`, computes `upper = mu[c]*lam_n_c` and `lower = -mu[c]*lam_n_c`, and applies four independent `if` clamps in RealSim's exact mutation order (matching `NonSmoothNewton.cpp:399-403`). No `max(0, lam_n)`, no `np.clip` with degenerate bounds. ✓

**Summary verdict:** **GREEN (ready to commit).**

A1 + A2 + A3 + Phase 1.3.a LBFGS form a coherent, internally-consistent change set. The Stage A path's correction-apply asymmetry that was exposed by the A1 fix is now closed by A3; no further ripples from the combined fix were found. The FBA NSN path is now byte-for-byte equivalent to RealSim's `NonSmoothNewton.cpp` `build → solve → applyConstraintCorrection` sequence on contact-λ-sign handling (modulo the still-open Phase 1.3.{c,d,e,f,g} items, all of which are queued as separate work and do not interact with the A1/A2/A3 sign-handling concerns).

Recommended: commit the A1+A2+A3 NSN-sign-handling fix as one atomic patch with the LBFGS Phase 1.3.a port (alongside, but logically separable). Phase 1.3.c (Stage A vs 3-row μ=0 equivalence test) can now be run as a regression check; with A3 in place, the equivalence should hold by construction.
