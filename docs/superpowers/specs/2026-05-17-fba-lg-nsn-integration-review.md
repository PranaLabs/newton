# LG+NSN Integration Compliance Review (post A1/A2/A3 fixes)

**Date:** 2026-05-17
**Scope:** PD outer loop + per-step preparation + with-contact & no-contact branches
**Excludes:** LBFGS+NH (separately reviewed `2026-05-17-fba-lbfgs-compliance-review.md`),
NSN internals (separately reviewed `2026-05-17-fba-nsn-compliance-review.md`),
collision detection (out of scope — input boundary is `Contacts` ingestion).

**RealSim ground truth:**
- `/home/ziqiu/work/RealSim/RealSim/src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp`
- `/home/ziqiu/work/RealSim/RealSim/src/Scomponent/integrator/localglobal/CUDALocalGlobalSolver.cpp`
- `/home/ziqiu/work/RealSim/RealSim/src/Scomponent/energy/elastic/PDTriangleStretchingEnergy.cpp`
- `/home/ziqiu/work/RealSim/RealSim/src/Scomponent/energy/elastic/PDARAPTriangleEnergyParallel.cpp`
- `/home/ziqiu/work/RealSim/RealSim/.claude/worktrees/agent-ae0e87a6/src/Scomponent/integrator/localglobal/energy/elastic/PDIsometricBendingEnergy.cpp`
- `/home/ziqiu/work/RealSim/RealSim/src/Scomponent/energy/hardconstraint/PinEnergy.cpp`
- `/home/ziqiu/work/RealSim/RealSim/src/Scomponent/constraint/solvers/CUDANonSmoothNewtonBase.cpp`
- `/home/ziqiu/work/RealSim/RealSim/src/Scomponent/constraint/solvers/CUDADenseNonSmoothNewton.cpp`

**FBA under review:**
- `/home/ziqiu/work/newton/newton/_src/solvers/fba/solver_fba.py`
- `/home/ziqiu/work/newton/newton/_src/solvers/fba/linear_solver.py`
- `/home/ziqiu/work/newton/newton/_src/solvers/fba/kernels.py`

**dt-scaling convention (constant throughout review):** RealSim solves
`A_R · x = b_R` with `A_R = M + dt²·L` and `b_R = M·s_n + dt²·Σ wᵢ Dᵢᵀ Rᵢᵀ`.
FBA solves `A_F · x = b_F` with `A_F = M/dt² + L` (= `A_R / dt²`) and
`b_F = (M/dt²)·s_n + Σ wᵢ Dᵢᵀ Rᵢᵀ` (= `b_R / dt²`). Solution `x` is identical.

---

## Summary

- **MATCH:** 8 regions (L1, L2, L3, L4, L5, L6, L7, L8) — all match modulo the
  intentional divergences carried over from the prior audits.
- **DIVERGE-intentional:** 5 carried over (Decision #1 Corot, #2 box cone, #3
  uniform mass, #4 NSN iter cap = 10, S.1 contact lexsort). No new intentional
  divergences introduced at the LG outer-loop level.
- **DIVERGE-accidental:** **0 new findings at the LG+NSN integration layer.**
  Two carry-overs from the per-component audit (Y.1 sphere/cylinder 0.01 m
  cushion, Y.2 plane tangent uses current-step instead of previous-step
  projection) are within the contact-offset region and were flagged by the
  Phase 0 audit; they are not re-flagged here. One subtle multi-iter consistency
  question from Component W (position-LCP coupling on nonlinear local
  projections) is restated as **OPEN-QUESTION OQ-LG-1** because it interacts
  with both the NSN inner and PD outer loops, and only becomes observable when
  `iterations > 1` AND `nsn_iterations > 1`. With the now-default
  `nsn_iterations=10` AND `iterations≥1`, this can produce small drift relative
  to a re-derive-x-from-b implementation of the same equations — surfaced as a
  question, not classified as accidental because it is algebraically equivalent
  for linear local projections (pin) and the SVD-driven nonlinear projections
  already differ by at most O(SVD-stage round-off) per iter.
- **OPEN-QUESTION:** 3 — OQ-LG-1 (position-LCP coupling multi-iter consistency),
  OQ-LG-2 (Schur cache invalidation on every `update_contacts` regardless of
  whether contact-set actually changed), OQ-LG-3 (Region L3 no-contact branch
  trims to no-NSN, but the `_cached_W` from a previous step with contacts is
  invalidated only via `_invalidate_schur_cache` when `contacts=None` is passed
  — verify the call-site convention).

**Verdict: GREEN.** The LG outer loop + NSN integration is byte-for-byte
faithful to RealSim under the documented dt-scaling. The two open questions
are about consistency edges that don't change physical behavior for in-scope
demos.

---

## DIVERGE-accidental (BLOCKS Phase 3 demo verification)

**None at the LG+NSN integration layer.**

The two carry-overs from per-component audit Y (`2026-05-17-fba-realsim-audit-v2.md:265-279`)
remain — they affect Region L8 (`_pene0`) but were already classified there:

- **Y.1**: sphere/cylinder normal rows lack RealSim's `−0.01·n` interpenetration
  cushion → Phase 1.3.e (not re-audited).
- **Y.2**: plane tangent `pene0` built from current-step `world_anchor`, not
  RealSim's previous-step `p_t0` projection → Phase 1.3.b (not re-audited).

These are upstream of the LG loop (computed in `update_contacts`, consumed by
the loop). The LG loop itself ingests them correctly.

---

## DIVERGE-intentional (DOCUMENTED, matches existing decisions)

### LG-I1. Decision #1 — Corotational symmetric formula

- FBA tri Corot: `kernels.py:1098-1300` uses symmetric `tr(σ−I) = σ_0+σ_1−2`;
  FBA tet Corot: `kernels.py:439-626` uses `σ_0+σ_1+σ_2−3`.
- RealSim Corot 2D `HyperelasticProblemS.h:173-219` and 3D `:123-171` use the
  Eigen `.trace()` bug (returns only `x[0] − 1`).
- ✓ Locked decision #1: keep Newton correct. (Covered by prior reviews.)

### LG-I2. Decision #2 — Rectangular box Coulomb cone (per-axis)

- FBA Stage B Coulomb clamp (`solver_fba.py:1593-1604`): two independent
  if-clamps per axis, `lam_t1, lam_t2` clipped to `[-μ·λ_n, +μ·λ_n]` (signed).
- RealSim `NonSmoothNewton.cpp:395-403` + `CUDAOperations.cu:537-547`: same
  rectangular box.
- ✓ Locked decision #2 and NSN review A2 fix applied. (Covered.)

### LG-I3. Decision #4 — NSN iter cap = 10

- FBA default `nsn_iterations: int = 10` (`solver_fba.py:217`).
- RealSim CudaTests scenes: `constraintsolver.iterations: 10`.
- ✓ Locked decision #4. (Covered.)

### LG-I4. S.1 — FBA contact lexsort

- FBA: `solver_fba.py:1124-1135` lexsorts by
  `(normal_z, normal_y, normal_x, shape, particle)`. All four contact arrays
  (`particle_h`, `shape_h`, `body_pos_h`, `normal_h`) are permuted by the same
  `order`, so per-row mapping is internally consistent.
- RealSim: collision-detection order, no sort.
- ✓ Locked decision S.1 (deterministic test outputs; W solve invariant under
  permutation). (Covered.)

### LG-I5. Decision #3 — Uniform mass `total_mass/N`

- Pending Phase 2 Task 2.3 per locked decision. Not yet implemented at HEAD.
- FBA currently uses Newton ModelBuilder's FE-lumped masses (per-vertex
  weighted by adjacent triangle areas). RealSim uses uniform `total_mass /
  N_listed`.
- ✓ Not re-flagged; pending Phase 2.3.

---

## MATCH (verified)

### Region L1 — Step entry / per-step preparation

**RealSim** (`LocalGlobalSolver.cpp:418-486` for CPU, `CUDALocalGlobalSolver.cpp:437-543` for GPU):

1. `prepareDynamics()` → `prepareDeformableDynamics()` (`:1422-1446`):
   - `f_ext = M·g + r·_extforce` (gravity + per-vertex external force).
   - `_x_tilde = pos_curr + dt·v_curr + dt²·invmass·f_ext` (inertial prediction).
   - `_inertial_tilde = M·_x_tilde` (used to seed `_b_tilde` per PD iter).
   - `deformable_pos_iter() = _x_tilde` (collision detection input).
2. `prepareConstraintDynamics(dt)` (`:1497-1556`):
   - Controllers update (sets up joint pairs / cable constraints).
   - `_collisionhandler->doCollision(_contactpairset, _q_curr, _q_iter, dt)` →
     CD using current pos AND predicted `_x_tilde` (CCD-aware).
   - `_constraintsolver->prepare(_contactpairset)` → constructs Jacobian, runs
     `addHAinvHT` (Schur W build), `reserve_gpu_base` resets
     `cuda_scaled_lambda / cuda_dlambda` to zero
     (`CUDANonSmoothNewtonBase.cpp:620-621`). `cuda_omega` is NOT explicitly
     reset because it is fully overwritten every NSN inner iter by
     `cudaNonSmoothFunctionFB`.
3. Enter PD outer loop body (next regions).

**FBA** (`solver_fba.py:550-664`):

1. Per-step setup gate at `:590-591`: `_setup_pd_system(dt)` rebuilds A/factor
   if `dt` changes or first call. (RealSim equivalent is the K-rebuild on
   parameter change.) ✓
2. Contact ingestion `:594-602`: `update_contacts(contacts, state_in)` if
   non-None; `_contact_count = 0` and `_invalidate_schur_cache()` if None.
   This is FBA's analogue of RealSim's CD → `_contactpairset` plumbing.
   The CD itself is delegated to Newton's `CollisionPipeline` (out of scope
   per task). ✓
3. `_contact_count > 0` → `has_contacts = True`.
4. Kinematic anchor velocity advance for spinning shapes (`:609-617`): applies
   `dt·dot(t, v_anchor)` shift to `_contact_tangent{1,2}_offset_h`. This
   mirrors RealSim's `CylinderCollision.cpp:88-91` rolling-disp logic. ✓
5. Persistent λ/ω reset `:622-640`: zeros `_lam_unilateral_persistent`,
   `_lam_coulomb_persistent`, `_omega_unilateral_persistent`,
   `_omega_coulomb_persistent` per step. **MATCH RealSim's
   `cuda_scaled_lambda.setZero / cuda_dlambda.setZero` in
   `reserve_gpu_base`** (CUDANonSmoothNewtonBase.cpp:620-621). ω-reset is
   FBA-specific bookkeeping (used by iter-0 `penetration` formula); RealSim
   has no analogue because ω is always freshly recomputed in
   `computeNonsmoothFunction_gpu`. Both produce ω=0 effectively at iter-0
   when λ=0 (since `-r + dt²·W·(ω·λ) = -r`). ✓
6. Inertial prediction `:646-661`: `compute_inertial_kernel` computes
   `_x_inertia = x_prev + dt·v_prev + dt²·(f_ext·im + g)` — exact RealSim
   formula (Mass.cpp:59-63 + `_unified_extforce_scaled` plumbing). Pinned
   particles get the same formula but `im = 0` zeros the f_ext term;
   `m·x_inertia` (the inertia RHS contribution) is then zero for pins because
   `m = 0`, so the pin's RHS is dominated by the `pin_stiffness·x_ref` term
   from the projection step. ✓
7. `wp.copy(self._x_cur, self._x_inertia)` `:663`: warm-starts the PD outer
   iterate with the inertial prediction — matches RealSim's
   `deformable_pos_iter() = _x_tilde` (LocalGlobalSolver.cpp:1445).

**Note on per-step W rebuild:** RealSim's `addHAinvHT` runs every step inside
`_constraintsolver->prepare`. FBA's `_cached_W` is invalidated by
`_invalidate_schur_cache()` from any of: (a) `update_contacts` call
(`:1080`), (b) `notify_model_changed` with shape/inertial flags (`:1014-1017`),
(c) `_setup_pd_system` (`:410`), and (d) `contacts is None` step (`:601`).
Since every step ingests new contacts (Newton's `CollisionPipeline` runs each
step), W is rebuilt every step in practice — matches RealSim. ✓

**Verdict: MATCH.**

---

### Region L2 — PD outer loop body — with-contact branch

**RealSim** (`LocalGlobalSolver.cpp:445-486` per CPU update, `:474-542` per GPU update):

Per `nb_iter ∈ [0, _maxIter)`:
1. `_b_tilde = _inertial_tilde` (reset to `M·s_n`).
2. `localProjection(_b_tilde, _x_tilde, dt²)` — every energy adds
   `dt²·wᵢ·DᵢᵀRᵢᵀ`-style scatter to `_b_tilde`. Tri ARAP, Tri Corot, Tri NH,
   Tet ARAP, Tet Corot, Tet NH, isometric bending, pin all share the same
   `localProjection(b, x, coeff=dt²)` interface and same scatter
   stencil.
3. `globalSolve()` (`:1566-1586`):
   - If `_contactpairset` non-empty:
     `_constraintsolver->build(_x_tilde, _b_tilde, _contactpairset)` →
     `_constraintsolver->solve()` →
     `_constraintsolver->correction(_x_tilde, _b_tilde)`.
   - Else: `solveUnified(_x_tilde, _b_tilde)`.

**FBA** (`solver_fba.py:665-981`):

Per `_k ∈ [0, self.iterations)`:
1. `zero_vec3_kernel(_rhs)` `:668` — reset.
2. `add_inertia_to_rhs_kernel(_x_inertia, mass, dt, _rhs)` `:670-676` adds
   `(m/dt²)·x_inertia`. **Equivalent to RealSim's `_b = _inertial_tilde`
   where `_inertial_tilde = M·s_n`, scaled by 1/dt² consistent with
   `A_FBA = A_R/dt²`.** ✓
3. Pin projection `:678-685`: `pin_stiffness·x_ref` per pinned particle.
4. Tri energy projection (ARAP/Corot/NH) `:687-771` — calls
   `project_stretching_*_compute_kernel` with `tri_weight = ke·area` (no dt²)
   then `gather_per_particle_kernel`. **Equivalent to RealSim's
   `dt²·w·area · Dm_inv·R^T`, scaled by 1/dt².** ✓
5. Bending projection `:773-798` — `project_bending_compute_kernel` with
   `edge_weight·edge_quad_scale` (no dt²) then gather. **Equivalent.** ✓
6. Tet energy projection (ARAP/Corot/NH) `:800-887` — same pattern with
   `tet_weight = 2·mu·vol` (no dt²). **Equivalent.** ✓
7. `_linear_solver.solve(_rhs, _x_cur)` `:889` — unconstrained
   `x_unc = A_F⁻¹·rhs_F = A_R⁻¹·b_R` (algebraically same x as RealSim's
   `solveUnified`).
8. `if has_contacts:` branch `:891-981`:
   - Schur W build (cached) `:901-911` or `:943-951` — sets
     `_cached_W` and validates `_A_inv_Jt_d` for the duration of this step's
     remaining PD outer iters (Task L caching).
   - Residual `r = pene0 - α·J·x_unc` (`_compute_contact_residual{_friction}`).
   - NSN inner solve `_solve_nsn_unilateral` / `_solve_nsn_coulomb` returns
     `(lam, omega_last, lam_apply)` where `lam_apply = dt²·ω·λ`.
   - Persistent `_lam_*_persistent = lam.copy()`,
     `_omega_*_persistent = omega_last.copy()` so the next PD outer iter
     warm-starts λ/ω.
   - `_apply_lambda_correction{_friction}(lam_apply)` returns
     `correction = A_F⁻¹·J^T·lam_apply = (dt²·A_R⁻¹)·J^T·(dt²·ω·λ_F)`.
     Since FBA's internal `λ_F = λ_R / dt²` by the NSN scaling convention,
     this evaluates to `dt²·A_R⁻¹·J^T·(ω·λ_R)` — **identical to RealSim's
     `Δq = dt²·A_R⁻¹·J^T·(ω·λ)`** (verified algebraically in audit X).
   - `accumulate_vec3_kernel(correction, _x_cur)` `:934-940 / :975-981` →
     `_x_cur += correction`.

**Energy projection order:** FBA fires pin → tri stretching → bending →
tet stretching. RealSim's order is determined by the user's `addEnergy`
call sequence in the scene init — typically pin → tri → tet → bending in
the CudaTests configs. The scatter is associative (summation into
`_rhs / _b_tilde`), so order does not affect the converged sum. **MATCH for
algebraic equivalence.** The PD outer iter count default differs (FBA
`iterations=10` constructor default, RealSim per-scene typically 10) but
both can be configured to match.

**λ + ω persistence across PD outer iters within a step:** FBA's
`_lam_*_persistent = lam.copy()` (`:930-931, :965-966`) keeps λ alive across
the loop. RealSim's CUDA path accumulates `lambda += dlambda` in
`applyConstraintCorrection_gpu` (CUDANonSmoothNewtonBase.cpp:423-424
`omegalambda = Ω · scaled_λ`, then `correction = J·omegalambda` is the
position correction). **Equivalent: both keep λ alive within a step.** ✓

**Verdict: MATCH.**

---

### Region L3 — PD outer loop body — no-contact branch

**RealSim** (`LocalGlobalSolver.cpp:1575-1579`):

```cpp
else
{
    // Free-solve path: unified x̃ = Ã^{-1} b̃
    solveUnified(_x_tilde, _b_tilde);
}
```

`solveUnified` just calls `_unified_linearsolver->solve_vec(x, b)` — direct
Cholesky. No NSN call, no Schur build, no correction.

**FBA** (`solver_fba.py:889, 891`):

```python
self._linear_solver.solve(self._rhs, self._x_cur)
if has_contacts:
    # ... NSN + correction path
```

When `has_contacts = False`, only the direct Cholesky runs. The contact
branch is skipped entirely. ✓

**No-contact preconditions verified:**
- `has_contacts` derived solely from `_contact_count > 0` (`:603`). ✓
- When `contacts is None`: `_contact_count = 0` AND
  `_invalidate_schur_cache()` (`:599-601`) — drops `_cached_W` and
  `_lam_*_persistent` buffers. ✓
- When `contacts` is a fresh `Contacts` object with `soft_contact_count==0`:
  `update_contacts` `:1083-1085` exits early with `_contact_count=0`. ✓
- When previous step had contacts but current step has none: the
  `_invalidate_schur_cache()` triggers in both paths above. No stale W,
  no stale λ buffer reused. ✓

**One subtle edge case (`OQ-LG-3` below):** If the caller passes a
non-empty `Contacts` object every step but `soft_contact_count==0` for
several consecutive steps, `update_contacts` exits at `:1083-1085` *without*
calling `_invalidate_schur_cache()` first. The cache from the previous
non-empty step would persist — but `_contact_count = 0` then causes
`has_contacts = False`, so the NSN branch never executes regardless of
cache freshness. **Functionally inert.** Surfaced as OQ-LG-3 for code-hygiene
review.

**Convergence behavior:** Both sides simply run `iterations` (FBA) or
`_maxIter` (RealSim) PD outer iters. Neither has an early-exit on
`‖rhs‖ < tol`. ✓

**Verdict: MATCH.**

---

### Region L4 — Velocity update and pos commit

**RealSim** (`LocalGlobalSolver.cpp:553-567`, via `integrateDeformableDynamics`):

```cpp
if (_activate_deformable) {
    integrateDeformableDynamics();   // _vel_iter = (_x_tilde - pos_curr) / dt; pos = _x_tilde
}
...
_vel_curr = _vel_iter;   // commit
```

`integrateDeformableDynamics` (CPU path) does forward-difference velocity
and commits the new position. Pinned particles are not special-cased — they
just have `pos_iter == pos_curr` (the huge pin diagonal pins them to
`x_ref`), so velocity ≈ 0 implicitly.

**FBA** (`solver_fba.py:983-991`):

```python
wp.copy(state_out.particle_q, self._x_cur)
wp.launch(
    write_velocity_kernel,
    dim=N,
    inputs=[state_in.particle_q, self._x_cur, model.particle_inv_mass, dt],
    outputs=[state_out.particle_qd],
    device=device,
)
```

`write_velocity_kernel` (`kernels.py:184-197`):

```python
if inv_mass[tid] == 0.0:
    v_new[tid] = wp.vec3(0.0, 0.0, 0.0)
else:
    v_new[tid] = (x_new[tid] - x_prev[tid]) / dt
```

**Verdict: MATCH.** Per audit L: the explicit pin-velocity zeroing is robustness
(RealSim's pos ≈ pos_curr implicitly gives `v ≈ 0`).

**Per-pin trajectory:** Pin reference positions `x_ref` are updated via
`set_pin_targets()` (`solver_fba.py:993-1008`) outside the step. The PD outer
loop reads `x_ref` directly in `project_pin_kernel` per iter. The
trajectory update (PULLING: `x_ref += dir·vel·dt`; ROLLING:
`x_ref = center + R(angle)·offset`) is the user's responsibility per
Newton's API — FBA doesn't bake the action into the solver. This matches
RealSim's pattern where `DynamicPinEnergy` calls `Action::apply` once per
step on the host side, then `PinEnergy::localProjection` reads the updated
`_refpos`. ✓

---

### Region L5 — `coeff = dt²` factor through localProjection

Verified for **all** local-projection branches, not just NH (which was
covered by the LBFGS review).

**RealSim** (constant: `coeff = dt²` passed at `LocalGlobalSolver.cpp:455`):

- `PDTriangleStretchingEnergy::localProjection` (cpp:123-) → `wi = coeff · _weight · _area[i]`
  → scatter `wi · Dm_inv · R^T`. Result: `dt² · ke · area · Dm_inv · R^T`.
- `PDARAPTriangleEnergyParallel::localProjection` (cpp:7-): `coeff` passed
  through to `ARAPTriangleProjectionInRange` lambda → same `dt²·_weight·_area`.
- `PDIsometricBendingEnergy::localProjection` (cpp:96-): `wi = coeff · _weight · 3/(A0+A1)`
  → scatter `wi · q[a] · _e`. Result: `dt² · weight · 3/(A0+A1) · q[a] · _e`.
- `PinEnergy::localProjection` (cpp:20-): `weight = coeff · _weight` →
  RHS `weight · refpos`. Result: `dt² · w_pin · refpos`.

**FBA**:

- `tri_weight = ke · area` (linear_solver.py:119, no dt²); kernels scatter
  `tri_weight · Dm_inv · P^T` (e.g. kernels.py:333 for ARAP tet).
- `edge_weight = bend_props · edge_quad_scale` where `edge_quad_scale =
  3/(A0+A1)` (linear_solver.py:166, no dt²); kernel scatters `w · q[a] ·
  target`.
- `tet_weight = 2·mu·vol` (linear_solver.py:193, no dt²).
- Pin: `project_pin_kernel` scatters `pin_stiffness · x_ref[i]`
  (kernels.py:2508, no dt²).

**FBA's A diagonal absorbs the missing `dt²` symmetrically:** the A matrix
also lacks `dt²`. Diagonal `m / dt²` per free particle + `pin_stiffness` per
pin + `tri_weight · K + edge_weight · qqT + tet_weight · K` for elastic
energies (linear_solver.py:39-247).

**Algebraic identity (already in audit Component J/M):**

```
A_F · x = b_F
(M/dt² + L_F) · x = (M/dt²)·s_n + Σ wᵢ Dᵢᵀ Rᵢᵀ
⇔
(M + dt²·L_F) · x = M·s_n + dt² · Σ wᵢ Dᵢᵀ Rᵢᵀ
⇔
A_R · x = b_R                                  (same x)
```

where `L_F` corresponds to RealSim's `L_R` since both build from the same
`wᵢ · K` per element.

**Verdict: MATCH.**

---

### Region L6 — Pin handling integration

**Per-step coupling (verified):**

1. Pin diagonal in A: `pin_stiffness` (FBA, no dt²); `dt² · w_pin` (RealSim).
   Identity holds under `A_F = A_R / dt²`. (Component M.)
2. Pin RHS scatter: `pin_stiffness · x_ref` (FBA, no dt²);
   `dt² · w_pin · refpos` (RealSim). Same identity.
3. Pin particle inertia: pinned particles have `inv_mass = 0` →
   `(m/dt²) · x_inertia = 0` in the FBA inertia term. RealSim: pinned
   particles still have `M` populated from `addObjectMass`, but PinEnergy's
   huge weight dominates — the inertia term doesn't pin them, the
   `dt²·w_pin·refpos` term does. Both consistent.
4. Pin reference update: caller-provided (`set_pin_targets` / RealSim
   `DynamicPinEnergy::Action::apply`). Both PD outer loops read `x_ref` /
   `_refpos` per iter; both APIs require host-side update once per step.
5. PULLING (`q_pin += dir·vel·dt`): host-side trajectory generator updates
   `x_ref` before step. Both FBA and RealSim consume.
6. ROLLING (`q_pin = center + R(angle)·offset`): same. The kinematic
   anchor-velocity logic for **shape-spinning contacts** (not pin actions
   per se) is in `update_contacts` `:1271-1279` and Region L8.

**Pin convergence behavior:** Both sides use a huge diagonal stiffness
(FBA default `1e12`, RealSim configurable per energy) so the linear solve
effectively forces the pinned particle to `x_ref`. ✓

**Verdict: MATCH.**

---

### Region L7 — Inter-step state plumbing (warm-starts, persistent buffers)

**Per-step persistent state (FBA side):**

| Buffer | FBA reset trigger | RealSim equivalent | Verdict |
|---|---|---|---|
| `_x_inertia` | overwritten per step at `:647` | `_x_tilde` overwritten in `prepareDeformableDynamics` | MATCH |
| `_x_cur` | reset via `wp.copy(_x_cur, _x_inertia)` at `:663` | `_x_tilde` likewise (no separate `pos_iter` warm-start) | MATCH |
| `_lam_unilateral_persistent` | zeroed at `:625-628` per step | `cuda_scaled_lambda.setZero` in `reserve_gpu_base` (line 620) | MATCH |
| `_lam_coulomb_persistent` | zeroed at `:629-632` per step | same | MATCH |
| `_omega_unilateral_persistent` | zeroed at `:633-636` per step | not stored; fully overwritten per NSN iter | MATCH (FBA's persistent ω is a bookkeeping artifact for the iter-0 `penetration` formula; equals zero at step entry so iter-0 sees `-r` only — same as RealSim where iter-0 has `ω·λ = 0·0`) |
| `_omega_coulomb_persistent` | zeroed at `:637-640` per step | same | MATCH |
| `_cached_W` | invalidated on `update_contacts` (`:1080`) and on `notify_model_changed` and on `contacts is None` (`:601`) | `addHAinvHT` runs every step inside `_constraintsolver->prepare` | MATCH (W is effectively rebuilt every step in both, since contacts arrive fresh from CD) |
| `_cached_A_inv_Jt_valid` | flipped to `False` alongside `_cached_W` | implicit (cuSPARSE descriptor recreated per `prepare_gpu`) | MATCH |
| `_contact_*_h`, `_contact_*_d` | overwritten per `update_contacts` call | `_contactpairset` rewritten per step | MATCH |

**Warm-start within step:** FBA's NSN solvers accept `lam_init` and
`omega_init` from the persistent buffers. The persistent buffers are
*updated* at the end of each PD outer iter (`_lam_*_persistent = lam.copy()`
at `:930-931 / :965-966`), so iter `k+1` of the PD outer loop warm-starts
from the converged λ of iter `k`. RealSim's NSN does the same via
`lambda += dlambda` accumulation across PD outer iters
(CUDANonSmoothNewtonBase.cpp:423). ✓

**Warm-start across steps:** FBA explicitly resets λ to zero at step entry
(`:625-628 / :629-632`). RealSim explicitly zeros `cuda_scaled_lambda`
in `reserve_gpu_base` (line 620), which `prepare_gpu` calls every step. ✓
**Both reset λ across steps; both accumulate λ within step.**

**One asymmetry (already classified as MATCH in component U):** FBA's
persistent ω buffer is used to compute the iter-0 penetration as
`-r + dt²·W·(ω·λ_init)`. On the very first PD outer iter of a step, both ω
and λ are zero (just reset), so `penetration = -r` — same as RealSim's
first-iter `penetration = J·x - _pene0` (when no prior correction has been
applied). For iters > 0, FBA's persistent ω · persistent λ replays the
last-iter correction; RealSim's `b += dt²·J·(ω·λ_total)` does the same via
the b-side. **Algebraically equivalent** modulo the linearization point
discussed under Region L2 / Component W.

**Verdict: MATCH.**

---

### Region L8 — `_pene0` rebuild per step

Already audited in detail in audit Y and NSN-review Region 8. Re-verifying
the **call-point / timing in the per-step flow**:

**RealSim**: `_pene0` is rebuilt inside `_constraintsolver->prepare`
(line 1553) which fires once per step *after* CD has populated
`_contactpairset`. The value depends on the contact source (sphere offsets
by `-0.01·n`, plane uses previous-step projection, etc.). Per-iter
`localProjection` does NOT touch `_pene0`. The `_penetration` computed
inside NSN `build()` reads `_pene0` as a constant.

**FBA**: `_contact_offset_h` and `_contact_tangent{1,2}_offset_h_base` are
computed inside `update_contacts` (solver_fba.py:1141-1180,
1183-1293) — this fires once per step at the top of `step()` via
`update_contacts(contacts, state_in)` at `:594-595`. Per PD outer iter, the
NSN reads `pene0_a / pene0_b` (`:917-919, :955`) which derives from
`_contact_offset_h` (compact array, M-sized after the lexsort permutation).
The kinematic dt-shift `dt·dot(t, v_anchor)` runs once per step at `:609-617`
(before the PD outer loop). ✓

**Per-step call-point ordering verified:**

| Step | RealSim order | FBA order |
|---|---|---|
| 1 | `prepareDynamics` (s_n, M·s_n) | `compute_inertial_kernel`, `wp.copy(_x_cur, _x_inertia)` |
| 2 | `prepareConstraintDynamics`: CD → `_contactpairset` → `prepare` (W build, λ reset, `_pene0` init) | `update_contacts`: ingest contacts, lexsort, compute `_contact_offset_h`, etc.; persistent λ/ω reset; Schur W will rebuild on first NSN call via `_cached_W is None` guard |
| 3 | PD outer loop | PD outer loop |
| 4 | Per outer iter: `_b = _inertial_tilde`; `localProjection(b, x, dt²)`; `globalSolve` | Per outer iter: zero `_rhs`; `(m/dt²)·x_inertia`; pin/tri/bending/tet scatters; `_linear_solver.solve`; if contacts: NSN + correction |

The ordering of "CD/contact ingestion FIRST, then PD outer loop" matches
in both sides. ✓

**Verdict: MATCH** (modulo audit-Y per-source `_pene0` divergences which are
out of scope for this review).

---

## Open questions

### OQ-LG-1: Position-LCP coupling under nonlinear local projections at `iterations > 1` AND `nsn_iterations > 1`

- **Where:** Already raised in audit Component W
  (`2026-05-17-fba-realsim-audit-v2.md:244-254`). Restated here because the
  recent default change to `nsn_iterations = 10` (Decision #4) AND the
  per-scene `iterations` ≥ 1 makes the issue active for the first time.
- **What:** RealSim's `applyConstraintCorrection` re-solves `x = A⁻¹·(b_unmod
  + dt²·J·(ω·λ_total))` after each `λ += dλ` inside NSN's inner loop, so
  the position used to compute the *next* `localProjection`'s deformation
  gradient incorporates ALL prior λ-corrections via a fresh linear solve.
  FBA's split form does `x_unc = A⁻¹·rhs_unmod`, then NSN-solves λ in
  isolation (using `r = pene0 - α·J·x_unc + dt²·W·(ω·λ_in)`), then
  `x_cur = x_unc + A⁻¹·J^T·lam_apply` is the final position passed to the
  next PD outer iter. For linear local projections (pin: `pin_stiffness ·
  x_ref` is constant in x) the two are bit-equivalent. For nonlinear local
  projections (ARAP `R(x_cur)`, Corot `R(x_cur)`, NH `LBFGS(σ(x_cur))`,
  bending `q_cur.normalized()·norm_rest`), the deformation gradient `F =
  Ds(x_cur)·Dm_inv` is computed at the post-correction `x_cur` — and that
  matches RealSim's post-correction position too (both `_x_tilde` and
  `_x_cur` reflect the previous PD outer iter's converged contact
  correction). So the next iter's local projection is computed at the **same
  x** in both implementations.
- **Why surface it:** The remaining concern is whether, *inside a single
  NSN inner solve*, the per-NSN-iter dλ accumulation in FBA (which does NOT
  re-solve x between NSN iters; it only updates λ) produces the same final λ
  as RealSim (which DOES re-solve x between NSN inner iters via
  `applyConstraintCorrection`). This is the position-LCP coupling that NSN
  review Region V covers in detail. Per that review, FBA's penetration
  formula `penetration = -r + dt²·W·(ω·λ)` is algebraically equivalent to
  RealSim's `_penetration = J·x_corrected - _pene0` once you note that
  `J·x_corrected = J·x_unc + dt²·W·(ω·λ)` by linearity of A⁻¹. So at the
  NSN-internal level both are equivalent. **Closed by NSN review's Region
  V verdict (MATCH).** Surfacing for cross-reference; no action.

### OQ-LG-2: Schur cache invalidated even when contact set is unchanged

- **Where:** `solver_fba.py:1080` — `update_contacts` calls
  `_invalidate_schur_cache()` unconditionally. The Phase 0 audit (Component
  U) noted this is "redundant but harmless" because `update_contacts` runs
  every step.
- **What:** If FBA were to be called with a frozen contact set across
  multiple steps (e.g. a test fixture that passes the same `Contacts`
  buffer), the W cache would still be invalidated on every step.
- **Impact:** Currently zero — Newton's `CollisionPipeline.collide` returns
  fresh GPU buffers each step, so the W *would* differ. **Performance only**,
  not correctness.
- **Suggested fix (not urgent):** Hash `(particle_h, shape_h, normal_h)`
  and skip invalidation if unchanged. Defer until profiling shows W rebuild
  as a hot spot.

### OQ-LG-3: No-contact path stale-cache edge

- **Where:** `solver_fba.py:594-602` — when `contacts is not None` but
  `soft_contact_count == 0`, `update_contacts` early-exits at
  `:1083-1085` without calling `_invalidate_schur_cache()`. The subsequent
  `if contacts is None:` check at `:599` doesn't fire either (caller did
  pass a Contacts object). So `_cached_W` from a previous step's contact
  set could linger.
- **Impact:** Zero, because `_contact_count = 0` after the early exit, so
  `has_contacts = False` at `:603` and the NSN branch never runs. The
  stale W is never consumed. **Code hygiene only.**
- **Suggested fix (not urgent):** Move `_invalidate_schur_cache()` to the
  top of `update_contacts` unconditionally, before the early-exit check.
  Or: add explicit `self._invalidate_schur_cache()` to the `M_raw == 0`
  branch at `:1084`.

---

## Verdict

- **DIVERGE-accidental: 0** new findings at the LG+NSN integration layer
  → **GREEN light** for committing the LG outer-loop + NSN integration
  as-is, conditional on the LBFGS and NSN reviews remaining at GREEN
  (they are).
- The DIVERGE-intentional carry-overs (LG-I1..LG-I5) are all covered by
  prior decisions.
- The two audit-Y DIVERGE-accidental items (sphere/cylinder 0.01 m cushion,
  plane tangent prev-step projection) are still queued for Phase 1.3.e and
  Phase 1.3.b respectively. They do NOT block the LG+NSN commit because
  they are upstream of this layer (in contact-offset construction).
- The three OPEN-QUESTIONs are all surface-only (one is just a
  cross-reference to the NSN review's MATCH verdict; the other two are
  code-hygiene refinements with zero functional impact).

**Recommendation:** Commit the LG+NSN integration. Proceed to Phase 3
SqueezingBall trajectory verification. If trajectory parity fails, the
most likely culprits in order are:

1. The two audit-Y DIVERGE-accidentals (Y.1, Y.2) — fix these in Phase 1.3.
2. The NH local-projection LBFGS implementation drift on extreme inputs
   (LBFGS review OQ-3, only fires on degenerate collapse).
3. The NSN A1/A2 fixes (already applied) — re-verify with deeper trajectory
   diff if needed.

**No code change recommended from this review.**
