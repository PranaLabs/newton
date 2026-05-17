# FBA-vs-RealSim Audit v2 (Phase 0 Task 0.1)

**Date:** 2026-05-17
**Newton HEAD:** `b97f7715` (code-equivalent to `7948ffe5` — only doc commits since)
**RealSim ref:** `/home/ziqiu/work/RealSim_py/realsim_py` (working tree as of audit)

Locked decisions (per the FBA-RealSim parity plan) treated as
ground-truth when classifying intentional divergences:

1. Corotational `.trace()` bug (tri + tet) — keep Newton's correct symmetric formula
2. Coulomb cone shape: rectangular box (`|λ_t| ≤ μ·λ_n`) — already in FBA at `solver_fba.py:1567-1573`
3. Mass lumping: switch FBA to RealSim's uniform `mass/N`
4. NSN inner iter default → 10
   > **CORRECTION 2026-05-17 (Task 1.3.h)**: RealSim's `constraintsolver.iterations: 10` is the **PCR iterative-solver** convergence cap, NOT a FB-Newton iteration count. RealSim does exactly 1 FB-Newton step per NSN call. Decision #4's original "switch to RealSim's cap = 10" was based on a misread. The correct alignment is `nsn_iterations=1`. FBA's direct `np.linalg.solve` on the Schur block is equivalent to RealSim's PCR-to-convergence.
5. λ-cap default per-scene (from `scene.json::constraintsolver.maxforce`)
6. Drift tolerance 1e-3 m
7. Plan scope: all 10 CudaTests demos

## Summary

- **MATCH:** 13 (A, B, F, I, J, K, L, M, O, P, Q, V, X)
- **DIVERGE-intentional:** 7 (C, D, G, N, R, Z, S.1 lexsort — post-2026-05-17 user decision)
- **DIVERGE-accidental:** 7 (E, H, S.3, T, U, W, Y) — feed into Phase 1.  *Y.2 sub-item reclassified to MATCH-by-identity 2026-05-17 (see below); Y.1, Y.3, Y.4 remain.*
- **UNKNOWN:** 0

**User decisions resolving the audit's open questions (recorded 2026-05-17):**

1. **G — 3D Eigen `.trace()` bug:** Locked decision #1 ("keep Newton's correct symmetric formula") extends to 3D. Documentation update only; classification stays DIVERGE-intentional.
2. **E/H — Tri/Tet NH solver:** **Port LBFGS to Warp** (full port, not Newton+convergence-check substitute). Becomes Phase 1.3.a.
3. **Y.2 — Plane tangent `_pene0`:** **Port RealSim's previous-step projection.** Track `q_prev` and re-project onto plane in `update_contacts`. Becomes Phase 1.3.b.
4. **S.3 — Stage A unilateral-only path:** **Keep the path** but add an equivalence test proving `friction=False` ≡ `mu=0` numerically. Becomes Phase 1.3.c.
5. **T.2 — `lambda_cap` units:** Internal auto-convert: constructor accepts user-facing physical force, divides by `dt²` once at solver init for internal `lam` clamping. Aligns with RealSim's `_maxforce` semantics. Becomes Phase 1.3.d.
6. **S.1 — Contact lexsort:** **Keep lexsort.** Determinism for test reproducibility is valuable; lexsort only permutes `lam` indices and dense W solve is invariant. Reclassified DIVERGE-intentional with rationale; **no Phase 1 fix needed**.

(See "DIVERGE-accidental list" at the end for the Phase 1 task seed.)

A note on the dt-scaling: FBA stores the PD Hessian as `A_FBA = (M/dt²) + Σ
w_E K_E` whereas RealSim stores `A_R = M + dt²·Σ w_E K_E` (i.e. `A_FBA =
A_R/dt²`). The PD energy and NSN row formulas all use dt² weights
consistently, so `x_unc_FBA = x_R` and `correction_FBA = correction_R`
hold identically at every PD outer iter (verified algebraically below
under V/X). The internal `lam_FBA` represents `lam_R/dt²` — this only
matters for the user-supplied `lambda_cap`, which currently clips in
FBA-scaled units (see T).

## Component-by-component

### A — PD outer loop control flow

- **RealSim:** `src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp:132-197`. Per-step flow:
  1. `_mass->computeSn(_nextpos, pos, vel, f_ext, dt)` — `_nextpos = pos + dt·vel + dt²·M⁻¹·f` (inertial prediction).
  2. `_mass->computeMx(f, _nextpos, 1.0)` — `f = M·s_n`.
  3. `prepare(pos, _nextpos, dt)` — collision detection → `_contactpairset`; constraint solver `prepare` runs `addHAinvHT`, `setZero` on `cuda_lambda`, etc.
  4. PD outer loop (line 163): for `nb_iter` in `0..maxIter-1`:
     - `_b = f` (reset to `M·s_n`)
     - `localProjection(_b, _nextpos, dt²)` — every energy scatters `dt²·w·D^T·R^T`-style into `_b`
     - `globalSolve(_nextpos, _b)` — if contacts: `cs->build, ::solve, ::applyConstraintCorrection`; else `_linearsolver->solve(_nextpos, _b)`.
  5. `vel = (_nextpos - pos)/dt`; `pos = _nextpos`.
- **Newton FBA:** `newton/_src/solvers/fba/solver_fba.py:570-993`. Per-step flow:
  1. `compute_inertial_kernel` → `_x_inertia = x + dt·v + dt²·(f·im + g)` (kernels.py:110-141).
  2. `wp.copy(self._x_cur, self._x_inertia)` (line 681).
  3. Reset persistent λ, ω at line 642-658 once per step.
  4. PD outer loop (line 684): for `_k` in `0..iterations-1`:
     - `zero_vec3_kernel(_rhs)`; `add_inertia_to_rhs_kernel` adds `(m/dt²)·x_inertia`.
     - Pin projection: `pin_stiffness·x_ref` for each pinned particle.
     - Energy projections (tri ARAP/Corot/NH, tet ARAP/Corot/NH, isometric bending) scatter into `_rhs`.
     - `_linear_solver.solve(_rhs, _x_cur)` — overwrite `_x_cur` with `A_FBA⁻¹·rhs`.
     - If `has_contacts`: build Schur W (cached), residual, NSN solve, `accumulate_vec3_kernel` adds correction.
  5. Write velocity: `v = (x_new - x_prev)/dt` for free particles.
- **Verdict:** **MATCH**.
- **Notes:** Both apply the contact correction *additively* on top of the unconstrained solve, just with different in-memory representations (RealSim re-calls `solve(x, b_mod)`, FBA computes `A_FBA⁻¹·J^T·lam_apply` and adds to `_x_cur`). Algebraically equivalent (linearity of A⁻¹). λ resets are once per step in both.

### B — Inertial prediction (`compute_inertial_kernel`)

- **RealSim:** `src/Scomponent/integrator/localglobal/energy/Mass.cpp:59-63` — `s_n[i] = pos[i] + dt·vel[i] + dt²·invmass[i]·f[i]`. `f` is *only* external (gravity in `computeExternalForce` LocalGlobalSolver.cpp:199-211).
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:110-141` — `x_inertia = x_prev + dt·v_prev + dt²·(f_ext·im + g)`. `f_ext = state_in.particle_f` (user external forces), `g = gravity[particle_world]`.
- **Verdict:** **MATCH**.
- **Notes:** FBA adds `g` unconditionally even for pinned particles. The result for pinned particles is unused (mass-weighted inertia RHS uses `m`, which is 0 for pins). Per-environment gravity via `particle_world` lookup is a Newton extension irrelevant for single-env demos.

### C — Tri ARAP local projection

- **RealSim:** `include/Scomponent/integrator/localglobal/energy/elastic/PDARAPTriangleEnergy.h:23-31` — `JacobiSVD<Mat3x2>(F)`, `S = [[I,0],[0,0]]` 3x2 (clamps SVs to 1), `P = U·S·V^T`. Scatter via `PDTriangleStretchingEnergy.cpp:132-153` — `_proj = wi · _restMatrix · P^T`; `rhs[t0] += -row0-row1; rhs[t1] += row0; rhs[t2] += row1`.
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:174-211, 906-976` — `svd_3x2(F)`: reduces 3x2 SVD to `wp.svd2(F^T F)`, recovers `U`, sets `S = I` (clamp), reconstructs `P = U·V^T`. Then `w·Dm_inv·P^T` scattered with the same `-row0-row1, row0, row1` pattern.
- **Verdict:** **MATCH**.
- **Notes:** Both compute the SVD-clamped-to-1 projection. Numerical SVD path differs (Eigen JacobiSVD vs Warp's 2x2 eigen path), but on float64 these agree to <1e-12. FBA float32 storage may add up to ~1e-7 noise per particle; that's documented and is below the 1e-3 m drift tolerance.

### D — Tri Corotational

- **RealSim:** `include/Scomponent/integrator/localglobal/energy/elastic/PDCorotationalTriangleEnergyParallel.h:41-81` — `JacobiSVD<Mat3x2>(F)`, then `LBFGS<double,2>` minimizes `CorotProjectionProblem2D` (`HyperelasticProblemS.h:173-219`) to convergence (`grad.norm()<1e-6` or `(x0-x1).norm()<1e-6`). The energy density uses `(x - I).trace()` on an `Eigen::Vector2d`, which is a 2×1 matrix — Eigen's `.trace()` sums `[0..min(rows,cols)) = [0..1) = x[0]` only, **producing an anisotropic energy that penalises only the first singular value**.
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:1050-1095, 1098-1300` — `wp.svd2(F^T F)` → singular values; closed-form 2x2 Sherman-Morrison on the symmetric system `(4μ+λ·11^T)·s = b`, where the symmetric `(σ - I).trace() = σ_0 + σ_1 - 2` is used (standard corotational).
- **Verdict:** **DIVERGE-intentional**.
- **Notes:** Locked decision #1 keeps Newton's correct symmetric formula. Also: FBA uses closed-form (1 iteration of Sherman-Morrison) vs RealSim's LBFGS-to-convergence. After fixing #1 wouldn't be applicable here (FBA stays correct).

### E — Tri Neo-Hookean

- **RealSim:** `include/Scomponent/integrator/localglobal/energy/elastic/PDNeohookeanTriangleEnergyParallel.h` (mirrors the tet parallel form below) + `HyperelasticProblemS.h:73-120` — energy `0.5·μ·(I_1 - log_I3 - 3) + 0.125·λ·log_I3²` (where `log_I3 = log(J²) = 2·log(J)`), so equivalent to `(μ/2)·(I_1 - 2·log(J) - 3) + (λ/2)·log²(J)`. Gradient: `μ·(s - 1/s) + λ·log(J)/s + k·(s - s0)` with `k = 2μ`. **Minimized via LBFGS to convergence** (`grad.norm()<1e-6` or `(x0-x1).norm()<1e-6`).
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:1303-1355` — same energy density and gradient; closed-form 2x2 Hessian inverse; **fixed 5 Newton iterations** with sigma clamp `>1e-6`.
- **Verdict:** **DIVERGE-accidental**.
- **Notes:** Same energy, same gradient — but 5 Newton iters vs LBFGS-to-convergence. Newton on the NH PD projection converges quadratically near the optimum, so 5 iters is "usually enough" but not provably ≤1e-6 grad-norm in all cases. Trajectory parity at 1e-3 m may hold for typical motions; need verification. Not covered by locked decisions; queue for Phase 1.

### F — Tet ARAP

- **RealSim:** `include/Scomponent/integrator/localglobal/energy/elastic/PDARAPTetrahedronEnergy.h:23-33` — `signedEigenSVD(F)` → `U, sigma, V`; `R = U·V^T` (signed). Scatter via `PDTetrahedronEnergy.cpp:176-196` — `_proj = wi·DmInv·R^T`; `rhs[t0] += -row0-row1-row2; rhs[t1..3] += row0,row1,row2`.
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:214-248, 251-386` — `wp.svd3(F)` → `U, sigma, V`; detects reflection via `det(U)·det(V) < 0` and flips column 2 of U; `R = U·V^T`. Scatter pattern identical.
- **Verdict:** **MATCH**.
- **Notes:** Reflection handling differs in implementation (RealSim's `signedEigenSVD` bakes the sign into `U`; FBA detects via determinant) but produces the same final R. `_tet_weight = 2·mu·volume` matches RealSim's `_weight·_vol[i] = 2μ·vol` (since RealSim sets `_weight = 2μ` for ARAP via `setElasticParameter`).

### G — Tet Corotational

- **RealSim:** `include/Scomponent/integrator/localglobal/energy/elastic/PDCorotationalTetrahedronEnergyParallel.h:41-86` + `HyperelasticProblemS.h:123-171`. `signedEigenSVD(F)`; LBFGS<double,3> minimizes `CorotProjectionProblem3D`. The energy uses `(x - I).trace()` on `Eigen::Vector3d` — **same .trace() bug as tri Corot**, summing only `x[0] - 1`.
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:389-436, 439-626` — `wp.svd3(F)`; closed-form Sherman-Morrison on `(4μ + λ·11^T)·s = b` using the standard symmetric trace `tr(σ-I) = σ_0+σ_1+σ_2 - 3`.
- **Verdict:** **DIVERGE-intentional**.
- **Notes:** Locked decision #1 (covers both tri AND tet Corot per the spec at `docs/superpowers/specs/2026-05-15-fba-corot-realsim-discrepancy.md`).

### H — Tet Neo-Hookean

- **RealSim:** `include/Scomponent/integrator/localglobal/energy/elastic/PDNeohookeanTetrahedronEnergyParallel.h:41-91` + `HyperelasticProblemS.h:15-63` — `signedEigenSVD(F)`, flip `sigma[2]` if negative (so the SVD becomes unsigned non-negative), then `LBFGS<double,3>` solves NH3D problem to convergence. Gradient: `μ·(s - 1/s) + λ·log(J)/s + k·(s - s0)`.
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:629-718, 721-903` — `wp.svd3(F)` (which returns unsigned non-negative SVs), 5 fixed Newton iterations with closed-form 3x3 Hessian inverse via cofactors.
- **Verdict:** **DIVERGE-accidental**.
- **Notes:** Same problem as E — identical formulation, but RealSim runs LBFGS-to-grad-norm-1e-6 while FBA does fixed 5 Newton iters. The number of inner iterations is not in the locked decisions. Queue for Phase 1 (likely the same fix path as E).

### I — Isometric bending (post H' fix)

- **RealSim:** `src/Scomponent/integrator/localglobal/energy/elastic/PDIsometricBendingEnergy.cpp:101-128, 130-177`. Per-edge: cotangent weights `q[a]` from rest geometry; rest norm `_norm[i] = ‖Σ q[a]·restPos[a]‖`. `localProjection`: `e = Σ q[a]·pos[a]`; if `_norm[i] > 1e-6`: `_e = e.normalized() · _norm[i]`; scatter `rhs[indices[j]] += wi · q[j] · _e` with `wi = coeff·_weight·3/(A0+A1)` and `coeff = dt²`.
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:1564-1701`. Per-edge: same cotangent weights (`linear_solver.py:_compute_isometric_bending_q`), `edge_norm = ‖q·x_rest‖` precomputed at setup (`solver_fba.py:472-481`). `project_bending_compute_kernel`: `qTxcur = Σ q[a]·pos[a]`; if `qTxcur` norm > 1e-12: `target = qTxcur · (norm_rest/norm_cur)`; scatter `w · q[a] · target` per local vertex.
- **Verdict:** **MATCH** (post H' fix).
- **Notes:** FBA precomputes `edge_norm` at setup. Edge weight already absorbs `3/(A0+A1)`. The skip condition differs (RealSim `_norm[i]>1e-6` vs FBA `norm_cur<1e-12` + early return on zero `norm_rest`); for any non-degenerate edge both branches execute identically.

### J — RHS assembly

- **RealSim:** `LocalGlobalSolver.cpp:163-179, 232-237` — per PD outer iter: `_b = f` (where `f = M·s_n` from `Mass::computeMx`); `localProjection(_b, _nextpos, dt²)` runs every energy's `localProjection`. The `coeff = dt²` is passed through and multiplies the per-energy weight before scatter.
- **Newton FBA:** `newton/_src/solvers/fba/solver_fba.py:685-895`. Per PD outer iter: `zero_vec3_kernel(_rhs)`; `add_inertia_to_rhs_kernel` (inertia term `(m/dt²)·x_inertia`); pin → tri stretching → bending → tet stretching scatters.
- **Verdict:** **MATCH**.
- **Notes:** dt² is absorbed differently: RealSim has `dt²·w·D^T·R^T` and inertia `M·s_n`; FBA has `(M/dt²)·s_n + w·D^T·R^T`. Since the system matrix is also scaled by 1/dt² in FBA, `x_unc_FBA = A_FBA⁻¹·rhs_FBA = (dt²·A_R)⁻¹·(M·s_n + dt²·Σ w·D^T·R^T)/dt² · ... = A_R⁻¹·b_R = x_R`. Same position.

### K — Linear solve (Cholesky + sparse inverse)

- **RealSim:** `src/Scomponent/linearsolver/direct/CUDASparseInverseSolver.cpp` + `SparseInverseSolver.cpp:121-150` (`computeLowerInverse_fast`) — runs Eigen's `SimplicialLDLT` factor `A = P·L·D·L^T·P^T`, then `LDLT_computeLowerInverse` produces `S = L⁻¹` with elimination-tree sparsity. Runtime: `x = P^T·S^T·D⁻¹·S·P·b`.
- **Newton FBA:** `newton/_src/solvers/fba/linear_solver.py:291-318, 320-462, 765-836` — `scipy.sparse.linalg.splu` factor (COLAMD ordering), extract L/U/D; `compute_lower_inverse` ports `LDLT_computeLowerInverse` (linear_solver.py:355-462); upload S, S^T, D⁻¹, perm to Warp BSR. Runtime: same `x = P^T·S^T·D⁻¹·S·P·b`, two BSR SpMVs per spatial component.
- **Verdict:** **MATCH**.
- **Notes:** RealSim uses Eigen SimplicialLDLT permutation, FBA uses SuperLU/COLAMD — different permutations but same factorization structure on SPD input. Numerically equivalent (modulo perm-dependent FP round-off).

### L — Velocity update

- **RealSim:** `LocalGlobalSolver.cpp:188` — `vel = (_nextpos - pos)/dt` (unconditional, all particles).
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:158-171` — `v_new = (x_new - x_prev)/dt` for `inv_mass != 0`, else `v_new = 0` (pins).
- **Verdict:** **MATCH**.
- **Notes:** For pinned particles RealSim doesn't zero explicitly, but `pos` and `_nextpos` are equal for pins (zero RHS contribution from inertia + huge pin diagonal pins them to `x_ref`), so velocity is also ~0. The explicit FBA zero is robustness, not behavior change.

### M — Pin handling

- **RealSim:** `src/Scomponent/integrator/localglobal/energy/hardconstraint/PinEnergy.cpp:6-39`. Matrix: `A[3i,3i] += dt²·w_pin` (line 21 with `coeff=dt²`). RHS: `localProjection` adds `dt²·w_pin·_refpos`.
- **Newton FBA:** `newton/_src/solvers/fba/linear_solver.py:89-97` builds `A[i,i] += pin_stiffness` (no dt²). `newton/_src/solvers/fba/kernels.py:1704-1725` adds `pin_stiffness·x_ref` to RHS.
- **Verdict:** **MATCH**.
- **Notes:** Both sides scale consistently with the dt² split (RealSim's `A_R` uses `dt²·w_pin` on diag and RHS uses `dt²·w_pin·x_ref`; FBA's `A_FBA = A_R/dt²` so diag uses `w_pin` and RHS uses `w_pin·x_ref`). Pin detection: FBA flags `inv_mass==0` as pinned; RealSim requires explicit `addPinEnergy(weight, vertices, refpos)`. For typical demos all pinned vertices are explicitly listed, so this difference doesn't matter.

### N — Mass lumping

- **RealSim:** `src/Scomponent/integrator/localglobal/energy/Mass.cpp:12-21` — `addObjectMass(total_mass, vertices)` distributes `total_mass / vertices.size()` uniformly per listed vertex. The scene JSON typically passes `objectmass` per geometry.
- **Newton FBA:** `newton/_src/solvers/fba/linear_solver.py:71-86` — uses `model.particle_mass.numpy()` which is FE-lumped from per-triangle / per-tet density·area (or density·vol/4 for tets) by Newton's `ModelBuilder`. Result: non-uniform per-particle mass for triangles where some vertices are adjacent to more triangles than others.
- **Verdict:** **DIVERGE-intentional**.
- **Notes:** Locked decision #3 says switch FBA to RealSim uniform. Not yet implemented; will be done in Phase 2 Task 2.3.

### O — Damping

- **RealSim:** No explicit damping in the PD integrator. `LocalGlobalSolver` uses pure implicit Euler with `s_n = pos + dt·vel + dt²·M⁻¹·f`; numerical damping is intrinsic.
- **Newton FBA:** Likewise no explicit damping. `compute_inertial_kernel` is pure implicit Euler.
- **Verdict:** **MATCH**.
- **Notes:** Newton's `particle_radius`, `particle_ka`, etc. are unused by SolverFBA. Newton has its own `damping` field in some solvers (e.g. XPBD), but SolverFBA ignores it.

### P — Gravity

- **RealSim:** `LocalGlobalSolver.cpp:199-211` — `computeExternalForce(f)`: zero `f`, then per particle `acc[i] = _grav`, then `f = M·acc`. Single scene-wide gravity vector.
- **Newton FBA:** `newton/_src/solvers/fba/kernels.py:110-141` — `g = gravity[particle_world[tid]]` (per-particle indirection via `particle_world`); `x_inertia += dt²·g`. For single-env demos `particle_world == 0` for all particles and `gravity[0]` is the scene gravity.
- **Verdict:** **MATCH**.
- **Notes:** The per-env indirection is a Newton-only extension that is a no-op when `particle_world` is all zeros, which is the case for the in-scope CudaTests demos.

### Q — Tangent basis

- **RealSim:** `include/Scomponent/collisiondetection/BaseCollision.h:25-35` — `generateTangentDirections(out0, out1, in)`: `tmp = ex` (or `ey` if `in==ex` exactly); `out0 = in × tmp / ‖·‖`; `out1 = in × out0 / ‖·‖`.
- **Newton FBA:** `newton/_src/solvers/fba/solver_fba.py:124-147` — `compute_tangent_basis(n)`: `ref = ey` if `|n[0]|>0.9` else `ex`; `t1 = n × ref / ‖·‖`; `t2 = n × t1 / ‖·‖`. Cylinder-aligned override (`solver_fba.py:1232-1249`): if shape is spinning, `t1 = normalize(normal × axis)`, `t2 = normalize(normal × t1)`.
- **Verdict:** **MATCH**.
- **Notes:** RealSim's exact-equality fallback degenerates when `in == ex` strictly; FBA's `|n[0]|>0.9` threshold is more robust. For all CudaTests demos the floor normal is `+y`, sphere normals are radial and varied — both formulas produce identical bases (since `n[0]` is small for non-near-X normals). The cylinder-aligned override in FBA exactly mirrors RealSim's `CylinderCollision.cpp:77` (`tangent = (normal × axis).normalized()`).

### R — Coulomb cone shape (rectangular box vs circular)

- **RealSim:** `src/Scomponent/lagrange/solvers/NonSmoothNewton.cpp:395-404` — `boundConstraintForces`: rectangular box clamp per axis, `lam_t1, lam_t2 ∈ [-μ·lam_n, +μ·lam_n]`.
- **Newton FBA:** `newton/_src/solvers/fba/solver_fba.py:1567-1573` (inside `_solve_nsn_coulomb`): `lam_n_c = max(0, lam[3c])`, `cone = mu[c]·lam_n_c`, `lam[3c+1] = np.clip(., -cone, cone)`, `lam[3c+2] = np.clip(., -cone, cone)`. Rectangular box, matches RealSim.
- **Verdict:** **MATCH** (post-P2-E; locked decision #2 already implemented). Originally the project shipped with a circular projection helper `project_coulomb_cone` (`solver_fba.py:90-121`); that helper is no longer in the active code path but is still exported.
- **Notes:** The dead `project_coulomb_cone` function is a cleanup candidate (Phase 4 Task 4.3 "cone dedup"). It can produce confusion for readers; suggest deletion or clear docstring marking it inactive.

### S — Contact Jacobian construction

- **RealSim:** `include/Scomponent/lagrange/LagrangeConstraint.h:33-51` — per `LocalBasis`: `triplets[cstId, dof*3+k] = coeff·dir[k]` for k=0..2. For particle-shape contacts `coeff=1`, `dof=particle_index`, so each contact row has 3 nonzeros (one per spatial axis at the particle DOF).
- **Newton FBA:** `newton/_src/solvers/fba/solver_fba.py:1057-1309` (`update_contacts`). Contact data is computed host-side per step and uploaded:
  - `_contact_particle_d[c] = particle_index`
  - `_contact_normal_d[c] = world-frame contact normal`
  - `_contact_alpha_d[c] = 1.0` (always for particle-shape contacts)
  - `_contact_tangent1_d, _contact_tangent2_d` for friction.
  - Inside `build_schur_complement` (`linear_solver.py:952-1179`): the (3M, total_rows, 3M) packing kernels build J^T column-by-column on demand.
- **Verdict:** **DIVERGE-accidental** — contact ordering and contact reduction.
- **Notes:**
  1. **Ordering**: FBA lexicographically sorts contacts by `(normal_z, normal_y, normal_x, shape, particle)` (solver_fba.py:1126-1137) to defeat the GPU atomic-add nondeterminism of Newton's `create_soft_contacts`. RealSim's order is "as visited by collision detection" which itself is parallel-TBB-loop nondeterministic in CPU detectors. Different ordering → different lam permutation → identical solve only if W is exactly symmetric and dlam is computed by a direct solver. FBA uses dense `np.linalg.solve` (which is direct), so order shouldn't affect the final lam_apply numerically — but the per-contact lam mapping differs, and any later post-processing that reads lam at a specific index will see different values.
  2. **Contact filter**: FBA filters `particle == -1` sentinels (line 1104) and deduplicates implicitly via the lexsort. RealSim has no equivalent filter; it trusts the collision detector to emit only valid contacts.
  3. **Stage A vs Stage B**: When `friction=False` FBA runs an M-row Stage A NSN (unilateral only). RealSim has no such mode — `mu=0` contacts still go through `nonsmoothUnilateralFunction` (single-row), and `mu>0` go through `nonsmoothFrictionalFunction` (3-row). FBA's Stage A path is a small simplification; in practice most demos use friction.

  These are accidental in the sense that they aren't part of any locked decision and they CAN cause measurable but small trajectory differences (especially #1 if subsequent code reads lam by index).

### T — λ-cap / maxforce

- **RealSim:** `src/Scomponent/lagrange/solvers/NonSmoothNewton.cpp:380-408` (`boundConstraintForces`) — clamps `_lambda[cid] ∈ [-_maxforce, +_maxforce]` per row. `_maxforce` is a per-scene `constraintsolver.maxforce` setting from the scene JSON (read via `init.h`).
- **Newton FBA:** `newton/_src/solvers/fba/solver_fba.py:1448-1449, 1575-1576` — applies `np.clip(lam, -self.lambda_cap, self.lambda_cap)` at the end of `_solve_nsn_unilateral` and `_solve_nsn_coulomb`. The cap is a constructor argument with no default per-scene wiring.
- **Verdict:** **DIVERGE-accidental**.
- **Notes:** Two parts:
  1. No per-scene default — every demo currently must set `lambda_cap` manually in the demo script. Locked decision #5 says this should be wired from `maxforce` in `scene.json`. Phase 2 Task 2.4.
  2. **Scaling mismatch**: FBA's internal `lam` is `lam_R/dt²` (see overview note above). When the user passes `lambda_cap=X`, FBA clips `|lam_R/dt²| ≤ X`, equivalent to `|lam_R| ≤ X·dt²`. RealSim's `_maxforce` clips `|lam_R| ≤ X`. So at `dt=0.01` the FBA clip is `1e-4` times smaller in physical force units than expected. This is an active bug in the lambda_cap semantics that would surface the moment any scene actually triggers the clamp — currently masked because no in-scope demo activates `maxforce` clipping. Phase 1.3 input.

### U — λ warm-start within step (post T' fix)

- **RealSim:** `src/Scomponent/lagrange/solvers/CUDANonSmoothNewton.cpp:365-366` — `cuda_lambda.setZero(_num_constraint); cuda_dlambda.setZero(_num_constraint)` in `prepare_gpu`, which runs once per step. Inside `applyConstraintCorrection_gpu` (`:456`) `lambda += dlambda` per PD outer iter, so λ accumulates across outer iters within the step.
- **Newton FBA:** `newton/_src/solvers/fba/solver_fba.py:638-658` — at step entry resets `_lam_unilateral_persistent`, `_lam_coulomb_persistent`, `_omega_unilateral_persistent`, `_omega_coulomb_persistent` to zero, sized to current `M`. `_solve_nsn_unilateral`/`_solve_nsn_coulomb` accept `lam_init` and `omega_init` arguments and accumulate `lam += dlam` per inner NSN iter (line 1445 / 1565).
- **Verdict:** **DIVERGE-accidental**.
- **Notes:** The structure is right (warm-start within step, reset across step). The accidental gap: **FBA also resets the warm-start when the contact set changes** (`_invalidate_schur_cache` at line 1021-1035 sets the persistent buffers to None). This invalidation fires on every `update_contacts` call regardless of whether the set actually changed — `Contacts` arrays at each step have fresh GPU buffers from the collision pipeline. So in practice the warm-start is essentially reset every step anyway, except across PD outer iters within a single step where the buffers are kept alive. That part DOES work as intended — but it overlaps with the `setZero` at line 644-646, which already runs every step. Net: the cache-invalidation reset is redundant but harmless. The "ω warm-start across PD outer" path in FBA uses persistent ω; RealSim doesn't store ω across PD outer iters (it computes ω fresh from `lam` each `computeNonsmoothFunction_gpu`). FBA's persistent ω is used to compute iter-0 penetration `-r + dt²·W·(ω·λ)`; this is logically equivalent to RealSim's behavior because the `r` value already reflects the corrected position from the previous PD outer iter via x_unc (a fresh `A⁻¹·rhs` is taken before each NSN call, so position has the previous correction baked in implicitly through energy projection's deformation gradient on x_cur). The two paths are *algebraically* equivalent but not literally identical at the per-iter level. Queue for Phase 1.1 verification.

### V — NSN solver (FB-Newton port)

- **RealSim:** `src/Scomponent/lagrange/solvers/NonSmoothNewton.cpp:332-378` defines the FB row formulas; `build()` (line 102-137) computes:
  1. `_penetration = J·pos - _pene0` (line 117-118)
  2. `computeNonsmoothFunction` fills `_omega, _nonsmooth_compliance, _h` per row.
     - Unilateral FB (line 332-341): `pene = _penetration[cid]`, `precondlam = lam·precond`, `root = sqrt(pene² + precondlam²)`, `_omega = 1 - pene/root`, `_compliance = (1 - precondlam/root)·precond/dt²`, `_h = -(pene+precondlam-root) + omega·(pene + pene0)`.
     - Friction FB (line 343-378): if `lam[cid] ≤ 0`, `omega=0, compliance=1/dt, h=-dt·lam_t`. Else: per tangent axis, `abspenevel = |penetration/dt|`, `tmp = precond·(μ·lam_n - |lam_t|)`, `root = sqrt(abspenevel² + tmp²)`, `compliance = ((root-tmp)/(abspenevel + μ·precond·lam_n - root))·precond/dt`, `omega = 1`, `h = -dt²·compliance·lam_t + pene0`.
  3. `_nonsmooth_delasus = ω·W·ω + diag(compliance)` (line 124, `computeNonsmoothDelasus`).
  4. `b_reshaped += dt²·J·(ω·λ)` (line 126-128 — adds the lambda-projected correction).
  5. `_systemlinearsolver->solve_vec(_x, b_reshaped)` (line 130 — recover corrected x).
  6. `_rhs = (1/dt²)·(_h - ω·J^T·_x)` (line 132-134).
- **Newton FBA:** `newton/_src/solvers/fba/solver_fba.py:16-87` (FB row), `:1374-1451` (unilateral), `:1453-1578` (Coulomb):
  - Unilateral row (`fb_unilateral_row`): identical formula to RealSim's `nonsmoothFischerBurmeisterUnilateralFunction`. Verified line-by-line.
  - Frictional row (`fb_frictional_row`): identical formula to RealSim's `nonsmoothFischerBurmeisterFrictionalFunction` — same inactive-case branch, same active-case `compliance = (root-tmp)/denom · precond/dt`, same `h = -dt²·compliance·lam_t + pene0`.
  - Schur LHS: `A_schur = (ω·ω^T) * W + diag(compliance)` (`:1436, :1556`) — exactly RealSim's `ω·W·ω + diag(compliance)` once you note both sides build a rank-1 outer-product as `(ω·1)·diag(ω)·W·diag(ω)·(1·ω^T)`. Match.
  - Penetration: FBA `penetration = -r + dt²·W·(ω·λ)` with `r = pene0 - α·J·x_unc` so `-r = J·x_unc - pene0`; plus `dt²·W·(ω·λ)` accounts for the in-iter correction. RealSim's `_penetration = J·x_corrected - pene0 = J·x_unc + dt²·_delasus·(ω·λ) - pene0`. With `W_FBA = dt²·_delasus_R` (because FBA's A is 1/dt² of RealSim's) and FBA's `λ_FBA = λ_R/dt²` (because dt² cancels through h·ω structure), `dt²·W_FBA·(ω·λ_FBA) = dt²·dt²·_delasus_R·(ω·λ_R/dt²) = dt²·_delasus_R·(ω·λ_R)`. **Same value.** Match.
  - RHS: FBA `rhs = (1/dt²)·(h - ω·J_x)` where `J_x = (pene0 - r) + dt²·W·(ω·λ)`. Algebraically equivalent to RealSim's `(1/dt²)·(_h - ω·J^T·_x)` once the dt-scaling factors out.
- **Verdict:** **MATCH**.
- **Notes:** Iter-default differs (FBA 1, RealSim per-scene typically 10) — locked decision #4 covers this. λ accumulation across inner NSN iters: FBA `lam = lam + dlam` (line 1445, 1565); RealSim `lambda += dlam` (line 158 of `applyConstraintCorrection`). The per-axis Coulomb clamp inside FBA's inner loop (line 1568-1573) is run every inner iter while RealSim clamps only in `boundConstraintForces` after the full NSN solve. That's a subtle difference but only manifests when nsn_iter > 1.
- **Iter-count note 2026-05-17**: `NonSmoothNewton.cpp` body shows no FB-Newton outer loop — `build()` → `solve()` (PCR Schur LCP) → `applyConstraintCorrection()` runs exactly once per call. FBA's `nsn_iterations` parameter is a FBA-only multi-iter extension, defaulted to 1 in Task 1.3.h for RealSim parity. Values >1 add extra FB-Newton linearizations not present in RealSim.

### W — Position-LCP coupling

- **RealSim:** `NonSmoothNewton.cpp:126-130` and `:158-167` (`applyConstraintCorrection`):
  1. In `build`: `b += dt²·J·(ω·λ)` and re-solve `x = A⁻¹·b`. The `x` returned is `A⁻¹·(b_unmod + dt²·J·(ω·λ_prev))`.
  2. In `applyConstraintCorrection`: `λ += dλ`, then `b += dt²·J·(ω·λ)` and re-solve `x = A⁻¹·b`. The final `x` is `A⁻¹·(b_unmod + dt²·J·(ω·λ_total))`.
- **Newton FBA:** `solver_fba.py:921-983` — at each PD outer iter:
  1. `_x_cur` is freshly overwritten by `_linear_solver.solve(_rhs, _x_cur)` → `x_unc = A_FBA⁻¹·rhs` (no contact correction). This corresponds to RealSim's `b_unmod` solve.
  2. Residual `r` computed; NSN solve returns `lam_apply = dt²·ω·λ`.
  3. `correction = A_FBA⁻¹·J^T·lam_apply` (via `_apply_lambda_correction(_friction)`); `_x_cur += correction`.
- **Verdict:** **DIVERGE-accidental** (minor, masked) — equivalent linear algebra but RealSim re-derives `x_unc` implicitly through the second solve.
- **Notes:** By linearity, `A⁻¹·(b + dt²·J·(ω·λ)) = A⁻¹·b + dt²·A⁻¹·J·(ω·λ)`, so FBA's split-form matches RealSim's combined-form. The "minor divergence" is purely about how penetration is computed in the *next* PD outer iter. In RealSim, the next iter's `build` reads `pos` (which is `x_unc + dt²·A⁻¹·J^T·(ω·λ_prev)` from the previous iter) and computes `_penetration = J·pos - pene0`. In FBA, the next iter freshly recomputes `x_unc = A_FBA⁻¹·rhs_new` (using fresh-energy-projection on the corrected `_x_cur`) and then adds back `+ dt²·W·(ω·λ_prev)` inside `penetration = -r + dt²·W·(ω·λ_init)`. The two are equivalent when the linear solve is exact and energy projections are linear-in-x — true for Pin, but the SVD-based ARAP/Corot/NH energy projections are nonlinear-in-x. **So in practice, FBA computes `penetration` using ω·λ_prev from the previous PD outer iter applied to the CURRENT-iter `x_unc`, whereas RealSim implicitly has the previous-iter correction baked into the deformation gradients used for the next localProjection. There is a subtle multi-iter mismatch.** Importantly, for `iterations=1` PD with `nsn_iterations=10` (RealSim default), this doesn't matter — only one PD outer iter runs. For `iterations=10` PD with `nsn_iterations=1` (current FBA default, also locked to be raised) it can drift. Queue for Phase 1.3 along with NSN iter change.

### X — Lambda correction application

- **RealSim:** `applyConstraintCorrection_gpu` (`CUDANonSmoothNewton.cpp:448-482`): `λ_total += dλ` (cuBLAS axpy), compute `tmp = ω·λ_total` (diag·vec), then `b += dt²·J·tmp` (cuSPARSE SpMV), then `_systemlinearsolver->solve(x, b)` (full re-solve over A). Net effect: `Δq = dt²·A_R⁻¹·J^T·(ω·λ_total) - dt²·A_R⁻¹·J^T·(ω·λ_total_prev)` between iters, but the result `x` is always re-derived from `b_unmod + dt²·J·(ω·λ_total)`.
- **Newton FBA:** `_apply_lambda_correction[_friction]` (solver_fba.py:1580-1742) computes `correction = A_FBA⁻¹·J^T·lam_apply` (where `lam_apply = dt²·ω·λ` from `_solve_nsn_*`), then `_x_cur += correction` (`accumulate_vec3_kernel`). The `A_FBA⁻¹·J^T·lam_apply` is computed via `apply_lambda_correction_isodof` (`linear_solver.py:1294-1352`): gather `J^T·lam_apply` via particle-centered CSR, then one full `solve(jt_lambda, out)` call.
- **Verdict:** **MATCH**.
- **Notes:** Both compute `Δx = dt²·A⁻¹·J^T·(ω·λ_total)` per PD outer iter. The FBA-vs-RealSim dt² scaling cancels: `A_FBA⁻¹ = dt²·A_R⁻¹`, `lam_apply_FBA = dt²·ω·λ_FBA = dt²·ω·(λ_R/dt²) = ω·λ_R`, so `correction_FBA = dt²·A_R⁻¹·J^T·(ω·λ_R)` — same physical displacement as RealSim. Verified algebraically.

### Y — `_pene0` semantics

- **RealSim:** `include/Scomponent/lagrange/LagrangeConstraint.h:54-64` (`initPenetration`): `_pene0[cstId] = _dir·_point` where `_dir` and `_point` are per-`LocalBasis`. The values per row depend on the contact source:
  - **Sphere** (`SphereCollision.cpp:121, 127-129`): normal row uses `point - 0.01·n` (allow 0.01m interpenetration); tangent rows use `point` (the surface contact point on the sphere).
  - **Cylinder static** (`CylinderCollision.cpp:84-86`): normal row uses `point - 0.01·n`.
  - **Cylinder rolling** (`CylinderCollision.cpp:88-91`): normal row uses `point - 0.01·n`; tangent row "axis" uses `point`; tangent row (rolling direction) uses `point + disp·tangent` where `disp = radius·avel·dt`.
  - **Plane** (`PlaneCollision.cpp:104-131`): normal row uses `point_n = p_t1 - n·(p_t1·n - base·n) · n` (current step particle projected onto plane). Tangent rows use `point_f = p_t0 - n·(p_t0·n - base·n) · n` (PREVIOUS step particle projected onto plane).
- **Newton FBA:** `solver_fba.py:1148-1180, 1183-1293`. For every contact (regardless of source):
  - `world_anchor = transform_point(body_q[b], soft_contact_body_pos[c])` for dynamic bodies; `world_anchor = body_pos` for static shapes.
  - `_contact_offset_h[c] = n·world_anchor` (normal pene0)
  - `_contact_tangent1_offset_h_base[c] = t1·world_anchor`, `_contact_tangent2_offset_h_base[c] = t2·world_anchor`.
  - Then per step (`solver_fba.py:627-635`): `_contact_tangent1_offset_h = base + dt·dot(t1, v_anchor)`, same for t2. `v_anchor` is computed for spinning shapes as `-ω · (axis × r_local) = ω·radius·t1` (when t1 = normal×axis).
- **Verdict:** **DIVERGE-accidental** — multiple gaps:
  1. **No 0.01 m interpenetration offset.** RealSim sphere/cylinder normal rows offset `point` by `-0.01·n` to allow 1cm interpenetration as a stability cushion. FBA uses `world_anchor` directly which has no offset (Newton's `soft_contact_body_pos` is the literal surface anchor). Implication: contacts engage harder, may cause more bouncing on first frame of contact.
  2. **Plane tangent uses current-step projection, not previous-step.** RealSim's plane tangent `pene0 = t·point_f` where `point_f` is built from `p_t0` (PREVIOUS step particle position). FBA's tangent `pene0 = t·world_anchor` uses the body-frame anchor (which Newton's collision pipeline derives from the CURRENT step). For sphere/cylinder, RealSim uses `point` (current surface anchor) for tangent — there's no time-shift. So only the plane case diverges.
  3. **Rolling cylinder displacement implemented correctly.** RealSim's tangent (rolling) `pene0 = t·(point + disp·t) = t·point + disp` with `disp = radius·avel·dt`. FBA computes `tangent1_offset = t1·world_anchor + dt·(t1·v_anchor) = t1·world_anchor + dt·(ω·radius)` for the rolling axis (since `t1 = normal × axis_world` and `v_anchor = -ω·(axis × r_local) = ω·radius·t1`). So `disp_FBA = ω·radius·dt = disp_RS`. ✓
  4. **Tangent basis sign on `t2`:** FBA uses `t2 = normal × t1` (axis-aligned direction); RealSim's cylinder uses the second basis as `axis` directly (parallel to the cylinder axis). For static cylinder (no rolling): `t2 = normal × t1 = normal × (normal × axis)/|·| = (normal·axis)·normal - axis ≈ -axis` (when normal⊥axis on the radial face). So FBA's t2 ≈ -axis, RealSim's t2 = +axis. **Sign difference**, but both directions parametrise the same tangent plane, so the result of `μ·lam_n` clipping per axis is symmetric — no observable effect.

  Net effect: items (1) and (2) are real divergences. Queue for Phase 1.

### Z — Multi-environment (`parallelEnv`)

- **RealSim:** `LocalGlobalSolver.cpp:71-77` — `unsigned _num_env`, the energy list is sized as `energy_per_env * num_env`, and energies for env 0 are initialised normally while envs 1..N copy from env 0's geometry-only data. This drives `parallelEnv` support for multi-env training.
- **Newton FBA:** No analogue. Only `model.particle_world` indexing for per-env gravity exists. There is no per-env partitioning of energies, contacts, or the linear-solve sparsity.
- **Verdict:** **DIVERGE-intentional**.
- **Notes:** The plan explicitly defers multi-env to Phase 5 ("New infra for demos 7-10"). Out of scope for Phase 0-2.

## DIVERGE-accidental list (Phase 1 input)

These rows have no covering locked decision and need explicit rectification. Each maps to a numbered Phase 1 sub-task:

- **E (Tri Neo-Hookean) / H (Tet Neo-Hookean)** → **Phase 1.3.a**: RealSim uses LBFGS-to-convergence; FBA uses 5 fixed Newton iterations. Same energy, same gradient. **Resolution:** Port RealSim's `LBFGS<double,2>`/`<double,3>` + Wolfe line search to Warp kernels (user decision 2026-05-17).
- **Y.1 (sphere/cylinder normal 0.01 m offset)** → **Phase 1.3.e**: RealSim sphere/cylinder normal rows offset `point` by `−0.01·n` (1 cm interpenetration cushion). FBA uses `world_anchor` directly. Add offset on Newton side at `update_contacts` for sphere/cylinder shape rows.
- **Y.2 (plane tangent prev-step projection)** → **RECLASSIFIED 2026-05-17 as MATCH-by-identity** (was Phase 1.3.b). Math identity: for any point `P`, `proj_plane(P) = P − (n·(P−base))·n`, so `n·proj_plane(P) = n·base` (independent of P) and `t·proj_plane(P) = t·P` (since `t·n=0`). Thus FBA's plane normal `pene0_n = n·world_anchor = n·base` equals RealSim's `pene0_n = orient·point_n = orient·base`; FBA's plane tangent `pene0_t = t·world_anchor = t·p_t0` equals RealSim's `pene0_t = t·point_f = t·p_t0` (FBA's `world_anchor` is the projection of `state.particle_q` at `pipeline.collide` time, which IS `p_t0` since Newton's collision pipeline reads state.particle_q raw without inertial prediction). The audit's earlier framing was wrong about the source-point time having any observable effect on the projection-derived pene0 values. **Detection set may still diverge** (RealSim detects against `p_t1`, FBA against `p_t0`), but that is a detection-pipeline question, not a pene0 question — out of Phase 1 scope. Task #14 closed.
- **S.3 (Stage A unilateral-only path)** → **Phase 1.3.c**: FBA's `friction=False` `_solve_nsn_unilateral` has no RealSim equivalent. **Resolution:** Keep the path (perf-valuable) + add unit test proving numerical equivalence with the `friction=True, μ=0` path. User decision 2026-05-17.
- **T.2 (`lambda_cap` units)** → **Phase 1.3.d**: FBA internal `lam` is `lam_R/dt²`; user-supplied `lambda_cap` currently clamps in scaled units. **Resolution:** Constructor auto-converts: `self._lambda_cap_internal = lambda_cap / (dt*dt)`. User-facing semantic now matches RealSim's `_maxforce` (physical force). T.1 (per-scene wiring) stays as Phase 2.4.
- **U (λ warm-start)** → **Phase 1.3.f**: `_invalidate_schur_cache` redundantly resets persistent λ on every contact-set update. Decouple: invalidate Schur factor but preserve λ buffers across contact-set changes when shape matches. Verify warm-start benefit on multi-step demos.
- **W (Position-LCP coupling)** → **Phase 1.3.g** (UPGRADED 2026-05-17 to direct port): Per user "完全对齐 A 级" decision, port RealSim's `_systemlinearsolver->solve` in-iter re-derive directly. Delete FBA's split-form `correction = A⁻¹·J^T·lam_apply`; in NSN apply, after `λ += dλ`, do `b += dt²·J·(ω·λ)` then `linearsolver.solve(x, b)` for the next PD outer iter's local projection input. The earlier "quantify then decide" framing is superseded — this is no longer "algebraically equivalent", it's a strict algorithmic port.

**Documentation-only follow-up (no Phase 1 sub-task; carried into Phase 4):**

- **G (3D Tet Corot Eigen `.trace()` bug):** Extend `docs/superpowers/specs/2026-05-15-fba-corot-realsim-discrepancy.md` to explicitly cover 3D case as same bug pattern as 2D. Locked decision #1 already applies. User decision 2026-05-17.
- **S.1 (contact lexsort):** Reclassified DIVERGE-intentional. Lexsort produces deterministic test outputs; only permutes `lam` indices; dense W solve invariant under permutation. User decision 2026-05-17.

## Open questions surfaced during audit — RESOLVED 2026-05-17

All 6 open questions have been resolved by the user. See the "User decisions" block in the Summary section above. Original audit text retained below for reference.

### Original questions (now resolved):

1. **Eigen `.trace()` bug on 3D vectors (Component G/Tet Corot):** RealSim's `CorotProjectionProblem3D::energy_density` (HyperelasticProblemS.h:139-141) calls `(x - I).trace()` on an `Eigen::Vector3d`, which is a 3×1 matrix — Eigen's `.trace()` sums `[0..min(rows,cols)) = [0..1)`, returning just `x[0] - 1`. This is the same bug pattern as the 2D case (covered by locked decision #1). I'm treating this as also covered by #1, but the original spec writeup at `docs/superpowers/specs/2026-05-15-fba-corot-realsim-discrepancy.md` should be explicitly extended to mention 3D as well.

2. **LBFGS-vs-Newton (Components E, H — Tri/Tet Neo-Hookean):** The trajectory-parity impact of replacing FBA's 5-iteration Newton with LBFGS-to-grad-norm-1e-6 is unmeasured. RealSim's LBFGS may converge differently from Newton's quadratic basin on this convex sub-problem. Two ways to resolve: (a) port LBFGS to Warp (high engineering cost); or (b) crank FBA's Newton iter count and add a convergence check (cheap). I recommend (b) and listing it as a Phase 1 task. Need user direction.

3. **Plane tangent pene0 time-shift (Component Y, item 2):** RealSim uses `point_f` from `p_t0` (previous step) for plane friction tangent pene0. FBA uses `world_anchor` from `soft_contact_body_pos` (current step). Newton's collision pipeline (`create_soft_contacts`) computes body_pos from the CURRENT particle position at detection time. To match RealSim we'd need to either (a) record the previous-step particle position and re-project onto the plane in `update_contacts`, or (b) accept the divergence as a Newton-collision-pipeline limitation. The implementation impact is non-trivial; need user direction before queueing.

4. **Stage A unilateral-only path (Component S, item 3):** FBA has `friction=False` that takes a separate M-row Stage A code path (`_solve_nsn_unilateral` instead of `_solve_nsn_coulomb`). RealSim has no equivalent; it always runs the 3-row friction path even when `mu=0`. For exact parity FBA should either (a) drop the Stage A path and always use 3-row, or (b) keep both paths but verify they produce numerically identical results when `friction=False` matches `mu=0`. I lean toward (a) for cleanliness; need user direction.

5. **`lambda_cap` scaling (Component T, item 2):** Want to confirm the user wants FBA to internally convert `lambda_cap` (user-supplied in physical force units, matching RealSim's `_maxforce`) → `lambda_cap/dt²` (FBA internal scaled units) before clamping. This is the cleanest fix but changes the existing constructor semantic. Need confirmation it's acceptable.

6. **Contact ordering (Component S, item 1):** FBA's lexsort makes simulations deterministic across GPU thread schedulings. RealSim has no equivalent. Removing the lexsort makes FBA bit-equivalent in ordering to whatever Newton's collision pipeline produces (which is GPU-atomic-nondeterministic). The user accepts this trade-off (determinism vs RealSim-byte-exact)? My read of the discipline note is "RealSim is ground truth", so I'd recommend dropping the lexsort if RealSim doesn't have one — but this needs a user decision.
