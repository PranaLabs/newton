# SolverFBA Contact — Phase 4 Spec (Schur-Complement Hard Constraints)

- **Date:** 2026-05-15
- **Status:** Draft — implementation roadmap pending user review
- **Owner:** ziqiu-zeng
- **Tracking branch:** `ziqiu/fba-solver-design`
- **Supersedes:** the contact "Phase 1" sketch implicit in `2026-05-15-fba-solver-design.md` §"Non-goals" (soft-penalty path is **explicitly skipped**)

## 1. TL;DR

Add hard-constraint contacts to `SolverFBA` by porting RealSim's **NonSmoothNewton + Schur-complement** pipeline (`W = J A⁻¹ Jᵀ`, dense λ solve, particle-side correction `Δb = Jᵀ λ` applied through the existing FBA sparse-inverse solver). The unilateral, no-friction sub-case (Stage A) lights up cloth-on-plane; later sub-stages add Coulomb friction (B) and general particle-vs-shape contacts (C); cloth self-contact (D) is post-MVP.

## 2. Newton's Existing Contact Infrastructure

### 2.1 `Contacts` storage layout

`newton/_src/sim/contacts.py:43` defines `class Contacts`. The fields that matter for soft (particle ↔ shape) contacts are populated by the collision pipeline:

| Field | Type | Source | Notes |
|---|---|---|---|
| `soft_contact_count` | `wp.array[int32]`, length 1 | view into `contact_counters[1:2]` (`contacts.py:264`) | live count; atomic-incremented by the soft-contact kernel |
| `soft_contact_particle` | `wp.array[int]`, length `soft_contact_max` | filled in `kernels.py:1139` | Newton's particle index (single int — see §4.4 self-contact gap) |
| `soft_contact_shape` | `wp.array[int]`, length `soft_contact_max` | filled in `kernels.py:1136` | shape index participant in the contact |
| `soft_contact_body_pos` | `wp.array[wp.vec3]` (optional grad) | `kernels.py:1137` — see derivation `contacts.py:267-268` | contact point on the body, **in shape-local frame**, NOT world |
| `soft_contact_body_vel` | `wp.array[wp.vec3]` | `kernels.py:1138` | local-frame vel of the contact anchor |
| `soft_contact_normal` | `wp.array[wp.vec3]` | `kernels.py:1140` — `world_normal = wp.transform_vector(X_ws, n)` | **world frame**, outward (body → particle) |
| `soft_contact_tids` | `wp.array[int]` | counter source for `counter_increment` | thread-id provenance (debug) |

Notably absent from `Contacts`: tangent basis, friction μ per contact, signed-distance scalar, particle weights/coefficients (vert-face self-contact). Friction μ is reconstructed by VBD from `model.soft_contact_mu × model.shape_material_mu` (`solver_vbd.py:2088`).

### 2.2 The collision pipeline

`newton/_src/sim/collide.py:902` defines `CollisionPipeline.collide(state, contacts)`. End-to-end soft-contact path for "cloth on plane":

1. The pipeline first runs **rigid-rigid** broad + narrow phase (writes `rigid_contact_*`); irrelevant here.
2. At `collide.py:1196-1230` the pipeline launches `create_soft_contacts` (kernel `newton/_src/geometry/kernels.py:1001`).
3. Per-thread, indexed by `tid = particle_index * shape_count + shape_index`:
   - Skip non-active particles / non-collide shapes (`kernels.py:1031-1034`).
   - World-id filter (`kernels.py:1041`).
   - Transform `particle_q[particle_index]` into shape-local frame (`kernels.py:1059`).
   - Evaluate the per-shape SDF — supported: `SPHERE`, `BOX`, `CAPSULE`, `CYLINDER`, `CONE`, `ELLIPSOID`, `MESH`, `CONVEX_MESH`, `PLANE`, `HFIELD` (`kernels.py:1070-1124`).
   - If `d < margin + radius`, atomically allocate a slot and write `(shape, body_pos, body_vel, particle, world_normal)`.
4. The `soft_contact_margin` flows in from `CollisionPipeline.__init__` (default 0.01) or `collide(soft_contact_margin=...)` override (`collide.py:909`).

**Who calls `collide()`?** Not `SolverBase.step`. Examples explicitly run it each substep — e.g. `newton/examples/cloth/example_cloth_hanging.py:165`:

```python
self.model.collide(self.state_0, self.contacts)
self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
```

`Model.collide(...)` (in `Model`) is a convenience wrapper around an internal `CollisionPipeline`. For Phase 4 we **do not need** to change this contract: users keep calling `collide()` before `solver.step(...)`; `SolverFBA.step` will consume `contacts.soft_contact_*` directly.

### 2.3 Available particle-vs-shape primitives

`newton/_src/sim/builder.py` exposes all the shapes we need:

- Plane / ground plane: `builder.py:5611-5701` (`add_shape_plane`, `add_ground_plane`).
- Sphere: `builder.py:5703`.
- Cylinder: `builder.py:5945`.
- Box, capsule, ellipsoid, cone, mesh, convex hull, heightfield: 5842-6134.

Demo 1 (cloth on plane), Demo 2 (softbody on plane), Demo 3 (cloth on cylinder / sphere) all use shapes whose SDFs are already wired up by `create_soft_contacts` — so **no new narrow-phase code is needed** in Newton for any of the three target demos.

### 2.4 Reference solver: VBD's soft-contact ingestion

`SolverVBD` (`newton/_src/solvers/vbd/solver_vbd.py`) is the canonical reference for soft-penalty contact. The key bits:

- `solver_vbd.py:2148-2183` launches `accumulate_particle_body_contact_force_and_hessian` once per color group.
- The kernel signature is `newton/_src/solvers/vbd/particle_vbd_kernels.py:2900-2967`. Inputs include `contact_shape`, `contact_body_pos`, `contact_body_vel`, `contact_normal` — exactly the four soft-contact fields above. Friction μ enters via two arrays (`body_particle_contact_material_mu`, `shape_material_mu`).
- It delegates to `_eval_body_particle_contact` (`rigid_vbd_kernels.py:757`), which recomputes the penetration depth `penetration_depth = -(n·(particle_pos - bx) - radius)` (line 796) from `body_pos` and the world normal. **That formula is the one we will reuse to fill the Schur-complement constraint vector `c` (signed distance).**
- The actual force law `_compute_body_particle_contact_force` (`rigid_vbd_kernels.py:722`) is a **soft-penalty / Baumgarte-damped** model — `force = n · ke · d + …`. Phase 4 throws that away in favour of hard λ-projection.

### 2.5 What the FBA solver currently does

`SolverFBA.update_contacts(...)` in `newton/_src/solvers/fba/solver_fba.py:428` raises `NotImplementedError("Contact-aware FBA solver TBD")`. `SolverFBA.step` (`solver_fba.py:203`) ignores its `contacts` argument (docstring at line 220-221: "Unused in MVP"). The PD outer loop (`solver_fba.py:266-391`) runs:

```python
for _k in range(self.iterations):
    zero(rhs); add inertia; project pin/stretch/bending → rhs
    self._linear_solver.solve(self._rhs, self._x_cur)   # x_cur = A^-1 . rhs
```

`FBALinearSolver.solve` (`linear_solver.py:749`) applies `A⁻¹ b = P⁻ᵀ Sᵀ D⁻¹ S P b` via three vec3 components × {S, ST} BSR SpMVs.

## 3. RealSim's Phase 4 Algorithm

### 3.1 PD outer loop integration

`LocalGlobalSolver::update` (`/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp:132-196`) is the canonical reference. The contact-aware PD step is:

```cpp
//compute external forces; sn = pos + dt*vel + (dt²/M)*f_ext; b = M·sn
prepare(pos, _nextpos, _dt);                                 // line 156
for(unsigned int nb_iter = 0; nb_iter < _maxIter; nb_iter++ ){
    _b = f;
    localProjection(_b, _nextpos, _dt*_dt);                  // local step → rhs
    globalSolve(_nextpos, _b);                               // line 175
}
```

`prepare(...)` runs collision detection and **builds the constraint system once per step** (LCS.cpp:213-230):

```cpp
_contactpairset.clear();
if(_collisionhandler) _collisionhandler->doCollision(_contactpairset, pos0, pos1, dt);
if(_constraintsolver) _constraintsolver->prepare(_contactpairset);  // builds J, W, etc.
```

`globalSolve` (LCS.cpp:239-253) is the key branch:

```cpp
void LocalGlobalSolver::globalSolve(MatX3R &x, MatX3R &b){
    if(_constraintsolver != nullptr && !_contactpairset.empty()){
        _constraintsolver->build(x, b);                // build NSN system
        _constraintsolver->solve();                    // solve W·Δλ = r
        _constraintsolver->applyConstraintCorrection(x, b);
    } else {
        _linearsolver->solve(x, b);                    // no contacts: usual A⁻¹·b
    }
}
```

### 3.2 NSN `prepare()` — building J, pene0, W

`BaseConstraintSolver::prepare(contactPairSet)` (`/home/ziqiu/work/RealSim_py/realsim_py/include/Scomponent/lagrange/solvers/BaseConstraintSolver.h:72-104`) is called **once per step**:

```cpp
_num_constraint = _constraint.getConstraintsInfo(contactPairSet);
std::vector<Eigen::Triplet<real>> triplets;
_constraint.buildJacobian(triplets, contactPairSet);          // J: M×3N
_jacobian.setFromTriplets(triplets.begin(), triplets.end());
_csJ = CSMatrix(_jacobian); _csJT = _csJ.switchOrder();
_pene0.resize(_num_constraint);
_delasus.resize(_num_constraint, _num_constraint);
_penetration.resize(_num_constraint);
_lambda.resize(_num_constraint); _lambda.setZero();
_systemlinearsolver->addHAinvHT(_delasus, _csJ);              // W = J·A⁻¹·Jᵀ
_constraint.initPenetration(_pene0, contactPairSet);          // pene0 = dir·point
```

**Jacobian assembly** (`LagrangeConstraint.h:33-51`): each `LocalBasis` writes one row indexed by `cstId`, with column triplets at `dof[k]*3 + {0,1,2}` and values `coeff[k] * dir[{0,1,2}]`:

```cpp
for(int k=0; k<cinfo._n; k++){
    triplets.emplace_back(cstId, dof[k]*3,   coeff[k] * dir[0]);
    triplets.emplace_back(cstId, dof[k]*3+1, coeff[k] * dir[1]);
    triplets.emplace_back(cstId, dof[k]*3+2, coeff[k] * dir[2]);
}
```

So each constraint row is `J_c = Σ_k coeff[k] · (e_{dof[k]} ⊗ dir)`. For a single particle-plane contact, `_n=1`, `coeff[0]=1`, `dof[0]=particle_index`, `dir=world_normal` — i.e. `J_c · x` is the signed displacement of the particle along the normal.

**`addHAinvHT(W, J)`** (`/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/linearsolver/direct/CUDASparseInverseSolver.cpp:34-40, 215-339`) is the heart of Schur build. Algorithm:

1. `analysisContact(J)` (CUDASparseInverseSolver.cpp:219 → SparseInverseSolver.cpp:179-184) extracts the **isolated DOF list** `_Ibar` — the set of vertex DOFs that are touched by *any* contact's Jacobian row.
2. Materialise `ST_new` = `S^T` restricted to the new isolated DOF columns (CUDASparseInverseSolver.cpp:256-257).
3. Form `Wi = ST · D⁻¹ · S` projected onto isolated rows by `cudaSelectSpMM` (line 261-262); the result `Wi` is `n_isodof × n_isodof_new` — an *approximate dense* per-DOF subblock of `A⁻¹`.
4. Reuse-strategy across timesteps: prev-frame `Wi_prev` is copied for unchanged isodofs; only the **new** isodof columns get refreshed (lines 240-263). This is the chief reason the Schur build is cheap in steady state.
5. SpMM: `JhWi = JhT · Wi` (cusparseSpMM, line 313-317); then `W = JhT · JhWiᵀ` (cusparseSpMM, line 320-324). Result is `M × M` dense.

Mathematically: `W = Jh · (S^T · D⁻¹ · S) · Jh^T = J · A⁻¹ · J^T` (with `Jh` = J restricted to isolated DOF columns; this restriction is valid because `J` is zero outside isodofs).

### 3.3 NSN `build(x, b)` — assemble Δλ system per PD inner iter

`NonSmoothNewton.cpp:102-137`:

```cpp
void NonSmoothNewtonSolver::build(const MatX3R &pos, const MatX3R &b) {
    real dt = _mstate->_dt;
    // penetration = J·x - pene0  (signed gap along normal/tangent)
    VecXR pos_reshaped = pos.transpose().reshaped(pos.rows() * 3, 1);
    VecXR b_reshaped   = b.transpose().reshaped(b.rows() * 3, 1);
    parallelSpMV(_penetration, _csJT, pos_reshaped);     // _csJT used as (J)^T → confusing naming
    _penetration -= _pene0;

    computeNonsmoothFunction();                          // sets _omega, _h, _nonsmooth_compliance
    computeNonsmoothDelasus(_nonsmooth_delasus, _delasus, _omega, _nonsmooth_compliance);  // W_nsm = ω·W·ω + diag(compliance)

    parallelDiagMV(_tmp_c1, _omega, _lambda);
    parallelSpMV(_tmp_dof, _csJ, _tmp_c1);               // J^T·(ω·λ)  → dof space
    b_reshaped += dt * dt * _tmp_dof;                    // augmented RHS with current λ contribution

    _systemlinearsolver->solve_vec(_x, b_reshaped);      // x_unc = A⁻¹·b'

    parallelSpMV(_tmp_c1, _csJT, _x);                    // J·x_unc
    parallelDiagMV(_tmp_c2, _omega, _tmp_c1);
    _rhs = (1.0 / (dt * dt)) * (_h - _tmp_c2);           // RHS of W·Δλ = rhs
}
```

(Note RealSim's "_csJ" stores J^T and "_csJT" stores J — the naming is `switchOrder()`-relative; in the math below we keep J = M×3N.)

`solve()` (NonSmoothNewton.cpp:139-149):

```cpp
_constraintlinearsolver->set(&_nonsmooth_delasus);     // small dense W_nsm
_constraintlinearsolver->solve_vec(_dlam, _rhs);
```

`applyConstraintCorrection(x, b)` (NonSmoothNewton.cpp:151-171):

```cpp
_lambda += _dlam;                                       // accumulate λ
parallelDiagMV(_tmp_c1, _omega, _lambda);               // ω·λ (active-set mask)
parallelSpMV(_tmp_dof, _csJ, _tmp_c1);                  // J^T·(ω·λ)
VecXR db = dt*dt * _tmp_dof;
b += db.reshaped(3, b.rows()).transpose();              // augment RHS
_systemlinearsolver->solve(x, b);                       // x = A⁻¹·b_aug
boundConstraintForces();                                // box-bound λ for stability
```

### 3.4 The non-smooth (Coulomb-cone) projection step

`computeNonsmoothFunction()` (NonSmoothNewton.cpp:207-233) dispatches per pair type. For Stage A we only need:

- **`nonsmoothMinimumMapUnilateralFunction`** (line 279-291):
  ```cpp
  if (_penetration[cid] < (_precond[cid] * _lambda[cid])) {
      _omega[cid] = 1.0; _nonsmooth_compliance[cid] = 0.0; _h[cid] = _pene0[cid];   // active
  } else {
      _omega[cid] = 0.0; _nonsmooth_compliance[cid] = _precond[cid] / (dt*dt);
      _h[cid] = -_precond[cid] * _lambda[cid];                                       // inactive
  }
  ```
  The min-map function picks the active arm when `J·x - pene0 < diag(W)·λ` (penetration deeper than current λ implies normal force should increase).

For Stage B (Coulomb friction), `nonsmoothMinimumMapFrictionalFunction` (line 293-330) adds the box-projection of tangent λ components by `μλ_n`. The Fischer-Burmeister variants (line 332-378) are smoother alternatives we may pick later for convergence quality.

`boundConstraintForces()` (line 380-408) is the **explicit** cone-projection postcondition: clamp normal λ ≥ 0 and `|λ_t| ≤ μλ_n`.

### 3.5 Per-step cost outline (from RealSim profiling)

NonSmoothNewton timing keys collected in `LocalGlobalSolver.cpp:294-301`: `prepare`, `build`, `cstsolve`, `correction`, plus the dominant **`schur`** cost (the `addHAinvHT` Wi/W build, owned by the linear solver, line 297).

Empirically (from the timer block 350-376):

- Schur build (`schur`): typically the dominant constraint-side cost; bound by `O(|Ibar| · |Ibar_new|)` SpMM. With reuse, steady-state is `|Ibar_new|` small (only changed-isodof columns are re-materialised).
- `build` (per inner iter): one `A⁻¹·b'` solve (the existing system solve) + a couple of SpMVs.
- `cstsolve`: dense `(M×M)` Cholesky or CG.
- `correction`: one extra `A⁻¹·b_aug` system solve.

So **the constraint pipeline adds ~2 system solves + 1 Schur build + 1 small dense solve per PD inner iter** (one is the prepare-time amortised Schur).

## 4. Newton-Side Mapping & Gaps

### 4.1 RealSim `ContactPair` / `LocalBasis` → Newton `Contacts.soft_contact_*`

`ContactPair.h` (`/home/ziqiu/work/RealSim_py/realsim_py/include/Scomponent/contact/ContactPair.h:7-21, 23-61`):

```cpp
class LocalBasis {
    int _n;            // number of impacted DOFs
    VecXI _index;      // impacted dofs
    VecXR _coeff;      // coefficient (barycentric-like)
    Vec3R _dir;        // contact normal
    Vec3R _point;      // contact point
};
class ContactPair {
    enum TypePair { bilateral, unilateral, friction, none };
    real mu;                                 // friction coeff or inverse stiffness
    std::vector<LocalBasis> _localBasis;     // multiple bases per pair (e.g. normal + 2 tangents)
};
```

| RealSim LocalBasis field | Newton soft_contact equivalent | Notes |
|---|---|---|
| `_n` (# impacted DOFs) | implicit = 1 | Newton `soft_contact_particle` is single int |
| `_index[k]` (k=0..n-1) | `soft_contact_particle[c]` (k=0 only) | **GAP: no vert-face self-contact** (see §4.4) |
| `_coeff[k]` | implicit = 1.0 | barycentric weights not present |
| `_dir` | `soft_contact_normal[c]` (world frame) | sign convention matches (outward = body→particle) |
| `_point` | `soft_contact_body_pos[c]` transformed to world | RealSim stores point in world; Newton in shape-local. Conversion needed. |
| `pair.type()` (unilateral/friction/bilateral) | implicit unilateral; friction added in Stage B | We pick type per stage at the solver level. |
| `pair.mu` | per-shape `model.shape_material_mu` mixed with `model.soft_contact_mu` via `sqrt(μ_p · μ_s)` (VBD precedent) | Newton has no per-contact μ — must mix from shape arrays. |
| `pair._localBasis.size()` (constraints per pair) | 1 (unilateral) or 3 (friction = normal + 2 tangents) | Same as RealSim; we generate the tangent rows internally. |

### 4.2 Penetration `pene0` / `c` (signed distance)

RealSim sets `pene0[c] = dir · point` (`LagrangeConstraint.h:60`). The constraint reads `J·x - pene0 = dir·x_particle - dir·x_anchor = signed-distance(particle, plane through anchor)`.

For Newton:
- `world_anchor = wp.transform_point(body_q[shape_body[shape]] * shape_transform[shape], body_pos)` — body-frame contact point pushed to world.
- `pene0[c] = dot(normal, world_anchor) - radius[particle]` so that `J·x - pene0 = dot(normal, particle_q - world_anchor) - radius` is the standard signed-gap function. Compare VBD's `penetration_depth = -(n·(particle_pos - bx) - radius)` (`rigid_vbd_kernels.py:796`) — same quantity, opposite sign convention. We adopt the **gap ≥ 0 is non-penetrating** convention to align with NSN min-map.

### 4.3 Constraint Jacobian for particle-plane / particle-sphere / particle-cylinder

All three demos hit the `_n = 1, coeff = 1.0` case. So the Jacobian row is simply:

```
J[c, 3*p+i] = normal[i],  with p = soft_contact_particle[c],  i ∈ {0,1,2}
```

Per-step nnz(J) = `3 · soft_contact_count`. For Stage B friction we add two tangent rows per active normal contact, each pointing the same particle in two orthogonal tangent directions.

### 4.4 Gaps (call-outs)

1. **`soft_contact_particle` is a single int.** RealSim's `LocalBasis._index` is a `VecXI` so it can encode multi-vertex (e.g. vert-face for cloth self-contact: 4 indices with barycentric weights). **For Phase 4 Stages A/B/C this is fine** — we only do particle-vs-static-shape. Stage D (cloth self-contact) requires extending `Contacts` with a separate buffer (e.g. `soft_self_contact_indices: wp.array2d[int]` shape (max, 4), `soft_self_contact_coeffs: wp.array2d[float]` shape (max, 4)). **Out of scope for the immediate roadmap.**
2. **No per-contact μ.** Stage B will mix from `model.soft_contact_mu` (global) and `model.shape_material_mu[shape]` per VBD's convention (`solver_vbd.py:2087-2106` and `rigid_vbd_kernels.py:861`). If user wants per-pair μ, that requires an extension to `Contacts` (consistent with `EXTENDED_ATTRIBUTES` mechanism at `contacts.py:55`).
3. **No tangent basis stored.** We compute tangents on-the-fly from the normal (orthonormalise an arbitrary off-axis vector). This matches RealSim's implicit assumption — tangents aren't stored in `LocalBasis` either; they're consumed inside NSN.
4. **Schur reuse strategy.** RealSim's `Wi`/`Wi_prev` reuse depends on contact correspondence frame-to-frame. Newton has `contact_matching` (rigid-side, `CollisionPipeline.__init__` mode `"sticky"` / `"latest"`, see `collide.py:558`) but **no soft-contact matching** yet. Stage A can skip reuse (full rebuild each step); Stage B can add a simple isodof-set hash for partial reuse if profile demands.

## 5. Math Derivation — Schur Complement KKT

### 5.1 Setup

Per-step we solve (component-wise in 3D, isotropic):

```
A · x* = b̃         minimise ½xᵀAx − bᵀx subject to constraints
J · x* ≥ c           unilateral (Stage A)
                    + friction cone constraints on additional rows (Stage B)
```

with
- `A = M/dt² + Σ_e w_e · Aᵉ^T Aᵉ + w_pin · I_pin` (scalar N×N, already factored as `S^T D⁻¹ S P`).
- `b̃` from PD local step (RHS of the unconstrained PD global solve).
- `J` is `M × N` per-component-broadcast-to-3 (each row has up to `n_indices` nonzeros).
- `c` is the signed-distance threshold s.t. `J·x − c ≥ 0` means "no penetration".

### 5.2 KKT system (unilateral, no friction)

Lagrangian `L = ½xᵀAx − bᵀx − λᵀ(Jx − c)`. Stationarity:

```
A·x = b + Jᵀ·λ           (1)
J·x ≥ c, λ ≥ 0, λ ⊥ (J·x − c)   (complementarity)   (2)
```

Substitute (1) → x = A⁻¹(b + Jᵀλ). Plug into (2):

```
J·A⁻¹·Jᵀ · λ ≥ c − J·A⁻¹·b
W · λ ≥ r          with  W = J A⁻¹ Jᵀ,  r = c − J·x_unc,  x_unc = A⁻¹·b
λ ≥ 0,  λ ⊥ (W·λ − r)
```

This is a **standard LCP** of size M. NSN solves it iteratively by re-linearising the min-map (active-set heuristic with smoothing). Per inner iter:

```
W_nsm · Δλ = (1/dt²)·(h − ω·J·x_unc)         small dense solve, M×M
λ ← λ + Δλ
b ← b + dt²·Jᵀ·(ω·λ)
x ← A⁻¹·b                                     re-solve the full system
project λ onto feasible set                  (clamping, box bounds)
```

For Stage A's min-map (NonSmoothNewton.cpp:282-290) `ω` is the active-set indicator, `compliance` is 0 active / `precond/dt²` inactive, and the result converges to `λ_active > 0`, `λ_inactive = 0`.

### 5.3 Stage B — friction cone

Each active normal contact spawns two tangent rows in J:

```
J_t1[c, 3p+i] = t1[i],   J_t2[c, 3p+i] = t2[i]
```

with `(n, t1, t2)` an orthonormal triad. `pene0_t = dir_t · world_anchor`. The friction part of `W` couples normal/tangent rows through `A⁻¹`.

Coulomb cone projection (`boundConstraintForces`, NonSmoothNewton.cpp:380-408):

```
λ_n ← max(0, λ_n)
λ_t ← clamp(λ_t, -μ·λ_n, +μ·λ_n)           (box approximation of 2D disc)
```

The NSN min-map for friction (`nonsmoothMinimumMapFrictionalFunction`, line 293-330) adds a non-smooth term that smoothly transitions between sticking (no slide → λ_t inside cone) and sliding (slip → λ_t on boundary).

### 5.4 Building W in Newton

The natural Newton implementation hooks `FBALinearSolver` (which already owns S, S^T, D⁻¹, perm):

For each constraint row `J_c` (sparse, length-3N vector, vec3 per particle), produce `y_c = A⁻¹·J_c^T` and assemble `W[c, c'] = J_{c'} · y_c`.

Stage-A naive implementation: `M` calls to `FBALinearSolver.solve(...)` is one option, but each `solve` is component-vec3 (3 SpMV pairs × 2 BSR = ~6 SpMVs per solve). Equivalently we want a **scalar-RHS variant**: `solve_scalar(b_scalar, x_scalar)` that bypasses the vec3 extract/insert kernels. Since the row `J_c` for a single-particle contact has just one non-zero vec3 entry `J_c · x = n · x_p`, the scalar-component solves with RHS `e_p · n_i` give us back `(A⁻¹)_{:,p} · n_i`. Summing over i: `y_c = Σ_i n_i · (A⁻¹)_{:,p_i_th_col}`. **In practice we can pre-extract per-particle 3-vector columns of A⁻¹ for active particles and do the multiplication CPU-side or as small SpMM.**

A cleaner formulation borrowing RealSim's "isolated DOF" trick: let `P` be the set of distinct particles touched by any active contact (`|P| ≤ M`, and typically `|P| ≈ M` for non-coincident contacts). Form a "fat-RHS" matrix `E_P` of size 3N × 3|P| (one column per (particle p ∈ P, axis i)) — each column is unit basis. Then `Y = A⁻¹ E_P` is a 3N × 3|P| dense block. Build `W[c, c'] = Σ_i Σ_j J_c[3p+i] · Y[3p'+j, 3|P|-col-for-(p,j)] · J_{c'}[3p'+j]`.

For Stage A (small M, single-particle contacts), the simplest concrete recipe is:

1. For each active contact c with particle p_c and normal n_c, run **one scalar BSR-solve** `(A⁻¹ · e_{p_c})` → column vector `α_c[:,:]` of shape (N,1) — but this is still 3 component solves (one per axis) if we want a vec3 column. Total cost: O(M) component solves × 2 BSR SpMVs = O(M · K · nnz(S)) where K=3. For N=10K, nnz(S) ≈ 20·N, M=100: 100·3·2·200K ≈ 120M flops per Schur build. Achievable in milliseconds on RTX-class GPUs.

2. Reduce `W[c, c'] = Σ_{i,j} n_c[i] · α_{c'}[p_c, ??]` — needs more care; see §8 sub-task 3.

The alternative, **batched** version: stack the M contact normal-axis basis vectors into a single 3N × M matrix `B`, push it through `(S^T D⁻¹ S)` as a SpMM, then dot rows back into the J rows. Warp doesn't ship cusparseSpMM; we'd lean on `wps.bsr_mm` if available or do M separate SpMVs into a 3N × M dense allocation. **Need a perf prototype to choose.**

### 5.5 Correction step

After λ is updated, the full re-solve

```
b_aug = b + dt²·Jᵀ·(ω·λ)
x_cur = A⁻¹ · b_aug    (existing FBALinearSolver.solve)
```

is **one extra existing-style vec3 solve** — i.e. ~6 BSR SpMVs. Per PD inner iter the constraint pipeline adds (at most) one extra existing solve plus the schur build + small dense Δλ solve.

## 6. Implementation Phases

| Stage | Scope | Effort | Dependencies | Demo target |
|---|---|---|---|---|
| **A — Unilateral, no friction** | Particle-vs-plane only; M ≤ ~256; full Schur rebuild each step; min-map for λ ≥ 0. | ~2 wks (1 wk algo + 1 wk validation) | None | Demo 1 (cloth on plane) |
| **B — Coulomb friction** | Add 2 tangent rows per active normal contact; NSN min-map for friction (line 293-330); cone projection in `boundConstraintForces`. | ~1.5 wks | Stage A correctness | Demo 1 stays stable with μ > 0; cloth no slide-through |
| **C — General particle-vs-shape** | Reuse existing `create_soft_contacts` — sphere/cylinder/box/mesh all work with no new narrow-phase code. Verify constraint Jacobian generation handles all soft-contact shape sources unchanged. | ~0.5 wk (testing only) | A + B | Demos 2 (softbody cube on plane) + 3 (cloth on cylinder) |
| **D — Cloth self-contact (post-MVP)** | Extend `Contacts` to hold `(indices[4], coeffs[4])` for vert-face self-contacts. Add Newton-side triangle BVH + closest-feature search (VBD has this via `tri_mesh_collision.py`). Generalise J assembly to `_n > 1` LocalBasis. | ~3-4 wks; **out of scope for initial Phase 4 PR** | A + B + C, plus a `Contacts` API change | Cloth-on-cloth in future PR |

### 6.1 Dependency / risk graph

```
A  ── B ── C ── (demos done, ship)
│
└── D  (separate PR, extends Contacts)
```

Stage A is the high-risk path because it builds the **first** Schur pipeline. Once it works (any demo, even synthetic small case), B/C are mostly bookkeeping.

## 7. Demo Plan

### 7.1 Demo 1: Cloth-on-plane

- Setup: cloth grid (e.g. 32×32 = 1024 particles); ground plane at z=0; gravity -9.8 m/s²; cloth starts above plane.
- Expected: cloth falls under gravity, contacts the plane, settles. Particles should stop drifting through the plane within ~50 steps.
- Validation: max penetration depth across all particles after step k should be **bounded by margin** (typically < 1e-3 m) once contact-active phase begins. No drift over 500+ steps.
- RealSim cross-check: run the same scene in RealSim (cloth + plane, NSN, identical `dt`, identical PD iterations) and compare per-step max-penetration. Target: same order of magnitude (e.g. both < 1e-3 m).

### 7.2 Demo 2: Softbody cube bouncing on plane

- Setup: tet softbody (e.g. 8×8×8 element cube = ~700 particles, ~4000 tets); plane at z=0; cube dropped from h=2m.
- Expected: cube bounces (slightly compliant) and eventually settles. Velocity damping handled implicitly by PD.
- Validation: total kinetic energy decreases monotonically after first impact; final settled state has max penetration < 1e-3 m.
- RealSim cross-check: same scene in RealSim's tet-softbody demo. Compare bounce height + settling time within ±10%.

### 7.3 Demo 3: Cloth on cylinder/sphere

- Setup: cloth (32×32) draped over a horizontal cylinder (radius=0.3m, axis along x). Cloth starts above cylinder; let it drape.
- Expected: cloth wraps around the cylinder, contacts on both sides, hangs. No penetration. With Stage B friction enabled, cloth holds in place; without (Stage A), it slides to the lowest hanging point.
- Validation: visual inspection + per-step penetration < margin. With μ=0.5 (Stage B), cloth should partially stick — measure final centroid position vs frictionless.
- RealSim cross-check: same draping setup. Compare equilibrium centroid position within ±5cm.

### 7.4 Cross-cutting tests (unittest)

- **`test_fba_contact_plane_minimal`**: 1 particle, 1 plane, no other forces. After 1 contact-active step, λ should be exactly `m·g/dt²` (the reaction force to gravity). x should land exactly on the plane (within float32 precision).
- **`test_fba_contact_schur_matches_reference`**: 2 particles, 1 plane, compare full A⁻¹·b (no constraint) and Schur-corrected result against a hand-built reference (small enough to invert in dense).
- **`test_fba_contact_no_drift`**: 100 particles, all in contact with plane, run 1000 steps with gravity. Max penetration should not grow.
- **`test_fba_contact_friction_stick_slip`** (Stage B): single particle on horizontal plane with tangential gravity; verify stick condition (μ·λ_n > F_t) holds → no motion.

## 8. API Design

### 8.1 Public API on `SolverFBA`

Consistent with `SolverBase` (no new public method names beyond what's already required):

```python
class SolverFBA(SolverBase):
    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        pin_stiffness: float = 1e12,
        stretching_model: Literal["arap", "corotational", "neohookean"] = "arap",
        mu: float | None = None,
        lam: float | None = None,
        # NEW (Phase 4):
        contact_iterations: int = 5,
        contact_max: int = 4096,
        contact_friction: bool = False,           # Stage B toggle
        contact_solver: Literal["min_map", "fischer_burmeister"] = "min_map",
        contact_max_force: float = 1e8,
    ) -> None:
        ...
```

Public methods (overrides of `SolverBase`):

```python
def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
    """Cache the active particle-vs-shape contact set and rebuild the
    Schur structure (J, c, W) for the upcoming step().

    Reads ``contacts.soft_contact_count``, ``soft_contact_particle``,
    ``soft_contact_shape``, ``soft_contact_normal``, ``soft_contact_body_pos``,
    ``soft_contact_body_vel`` and (with ``state`` provided) the current
    ``state.particle_q`` for the initial penetration ``pene0``.

    Args:
        contacts: :class:`newton.Contacts` populated by a
            :meth:`newton.CollisionPipeline.collide` call.  Only soft contacts
            are consumed; rigid contacts are ignored.
        state: Optional :class:`newton.State`; if provided, ``state.particle_q``
            seeds the initial ``pene0`` and warm-starts ``lambda``.

    Notes:
        This method is called automatically by :meth:`step` when
        ``contacts`` is non-None; users do not normally call it directly.
        It is exposed so external pipelines (e.g. AVBD-style fixed-point
        loops) can refresh constraint state out-of-band.
    """

def step(
    self, state_in: State, state_out: State,
    control: Control | None, contacts: Contacts | None, dt: float
) -> None:
    """Advance one implicit-Euler PD step. When ``contacts`` is non-None
    and contains at least one active soft contact, run the Schur-complement
    NonSmoothNewton inner loop; otherwise fall back to the existing PD path.
    """
```

`set_pin_targets` (existing, line 403) is unchanged. `notify_model_changed` (line 420) is unchanged.

### 8.2 Internal API on `FBALinearSolver`

The Schur build needs a few new methods on `FBALinearSolver`:

```python
class FBALinearSolver:
    # existing
    def solve(self, b: wp.array[wp.vec3], x: wp.array[wp.vec3]) -> None: ...

    # NEW (Phase 4):
    def solve_scalar(
        self, b_scalar: wp.array[wp.float64], x_scalar: wp.array[wp.float64]
    ) -> None:
        """Scalar-RHS variant of :meth:`solve` for one cartesian component.

        Used by Phase 4 Schur build to avoid the vec3 extract/insert overhead
        when computing ``A⁻¹·J_c^T`` for a single contact row.

        Args:
            b_scalar: Input scalar RHS, shape [n], dtype float64.
            x_scalar: Output, shape [n], dtype float64.
        """

    def apply_sparse_inverse_column(
        self,
        particle_indices: wp.array[int],    # length M, particle index per contact
        contact_normals: wp.array[wp.vec3], # length M, contact normal per contact
        out: wp.array2d[wp.float64],        # shape (M, 3*n), dense output
    ) -> None:
        """Compute ``out[c, :] = (A⁻¹ · e_{p_c} ⊗ n_c)^T`` for c = 0..M-1.

        This is the inner loop of Schur-complement build: for each contact c,
        apply ``A⁻¹`` to the sparse RHS that places normal n_c at particle
        ``particle_indices[c]``.  Result is stored as a dense (M, 3·n) matrix
        so the subsequent W[c, c'] reduction is a plain dot product.

        For M ≪ N this is M independent BSR SpMVs (≈3 per contact).  For
        large M, callers may prefer the batched :meth:`apply_sparse_inverse_block`.
        """

    def build_schur_complement(
        self,
        J_particles: wp.array[int],          # length M, particle per row
        J_normals: wp.array[wp.vec3],        # length M, normal per row (Stage A unilateral)
        W_out: wp.array2d[wp.float64],       # shape (M, M), dense output
    ) -> None:
        """Materialise ``W = J · A⁻¹ · Jᵀ`` for the current active-contact set.

        Combines :meth:`apply_sparse_inverse_column` + a small M×M dot-product
        reduction.  Output is row-major dense ``(M, M)`` on the device.

        Notes:
            Stage A assumes single-particle rows (n=1, coeff=1).  Stage B
            extends with tangent rows by stacking three rows per active
            contact (normal + 2 tangents); the M dimension grows by 3×.
            Stage D (self-contact) requires generalising to multi-particle
            rows; a separate overload will accept (particle_indices: 2D,
            coeffs: 2D) when that lands.
        """
```

### 8.3 New internal state on `SolverFBA`

Per-step device buffers (allocated lazily once `contact_max` is bound):

```python
# Stage-A scoped state on SolverFBA:
self._contact_active_count_d: wp.array[int]          # length 1
self._contact_particle_d:     wp.array[int]          # length contact_max
self._contact_normal_d:       wp.array[wp.vec3]      # length contact_max
self._contact_pene0_d:        wp.array[wp.float64]   # length contact_max
self._contact_lambda_d:       wp.array[wp.float64]   # length contact_max  (persisted across substeps)
self._contact_omega_d:        wp.array[wp.float64]   # length contact_max  (active-set mask)
self._contact_W_d:            wp.array2d[wp.float64] # shape (contact_max, contact_max), dense
self._contact_rhs_d:          wp.array[wp.float64]   # length contact_max
self._contact_dlambda_d:      wp.array[wp.float64]   # length contact_max
self._contact_jtl_d:          wp.array[wp.vec3]      # length particle_count, J^T·(ω·λ) result
self._dense_solver:           # small CPU/GPU dense solver — Cholesky on (contact_max, contact_max)

# Stage B additions (μ + tangents):
self._contact_friction_mu_d:  wp.array[wp.float64]   # length contact_max
self._contact_tangent1_d:     wp.array[wp.vec3]
self._contact_tangent2_d:     wp.array[wp.vec3]
```

### 8.4 New Warp kernels (Phase 4 scope)

- `build_contact_pene0_kernel` — given (particle_q, soft_contact_*), compute `pene0[c] = dot(normal, world_anchor) - radius[p]`.
- `compute_contact_active_set_kernel` — implements the min-map function (NonSmoothNewton.cpp:282-290): writes `omega[c]` and `h[c]`.
- `apply_jt_omega_lambda_kernel` — scatter `J^T·(ω·λ)` from contact-space to per-particle vec3 forces.
- `assemble_w_dense_kernel` (Stage A first draft) — reads per-contact "column of A⁻¹·e_p" from the dense buffer produced by `apply_sparse_inverse_column` and writes the `M×M` dot products.
- `clamp_lambda_unilateral_kernel` (Stage A) / `clamp_lambda_cone_kernel` (Stage B) — implements `boundConstraintForces` (NonSmoothNewton.cpp:380-408).
- Stage B only: `compute_tangent_basis_kernel` — given world normal, emit a deterministic orthonormal `(t1, t2)`. Householder-style.

All kernel signatures must use `wp.array[T]` annotation (per `AGENTS.md`).

### 8.5 No breaking changes

- `Contacts` class **untouched** for Stages A-C. Stage D adds new optional fields (additive).
- `CollisionPipeline.collide` API unchanged. Users continue to call `self.model.collide(state, contacts)` before `solver.step(...)`.
- `SolverBase.step` signature unchanged.
- Phase 4 lights up by passing a non-None `contacts` to `solver.step(...)`; absent that, FBA's behaviour is exactly the current MVP.

## 9. Open Questions

1. **Q1 (mu provenance):** VBD mixes per-particle `model.soft_contact_mu` with per-shape `model.shape_material_mu` via `sqrt`. RealSim uses per-pair `ContactPair.mu`. For Phase 4 we adopt the VBD convention for compatibility with existing examples. **Decision needed:** is that acceptable, or do we want per-contact μ via an extended attribute? *(Mark for user.)*

2. **Q2 (Schur build strategy):** §5.4 outlines two approaches: M independent SpMVs vs batched SpMM. Warp ships `wps.bsr_mv` but `wps.bsr_mm` for BSR-dense is less polished. **Recommendation:** start with M independent solves (correct, simple); benchmark on cloth-on-cylinder (M~thousands) and revisit. *(No blocker for Stage A.)*

3. **Q3 (`update_contacts` semantics):** RealSim's `prepare()` runs once per step before the PD outer loop. The Schur W is built once and reused across all inner iters. Newton's user pattern is `pipeline.collide()` then `solver.step()` — we'll call `update_contacts` internally as the first thing inside `step()`. **But:** should `update_contacts` be **idempotent**? If a user calls it manually before `step()`, do we skip the re-fetch? Suggest: store `contacts.contact_generation` last seen and skip rebuild if unchanged.

4. **Q4 (warm-starting λ):** Persisting λ across substeps requires matching contacts across substeps. Without contact-matching for soft contacts (§4.4 gap 4), λ-warm-start is only safe within a single step. **Stage A: zero λ at start of each `step()`.** Stage B may need warm-start for friction convergence. *(Mark for user; depends on whether stick-slip transitions are critical for demos.)*

5. **Q5 (NSN convergence tolerance):** RealSim parameterises `_maxIter` and `_tol` (BaseConstraintSolver.h:115). Newton FBA should expose the same as `contact_iterations` (already in §8.1 API). **Default:** 5 inner iters with no early-exit (matches RealSim's default behaviour, no convergence check).

6. **Q6 (rigid-body coupling):** RealSim's NSN runs in particle-only land. Newton's `Contacts.soft_contact_body_vel` carries the body velocity at the contact anchor — useful for moving-shape contacts (Demo 3 with rotating cylinder). For Phase 4 we treat the shape as kinematic (its motion comes from outside FBA), and propagate the body velocity into a moving `c(t)` if needed. **Decision:** Stage A treats all shapes as static (`body_q_prev == body_q`); Stage B is the natural place to add `body_vel · dt` into the constraint RHS. *(Defer to Stage B kickoff.)*

7. **Q7 (Mass-only "rigid-equivalent" pipeline):** RealSim has both `addHAinvHT(W, J)` (PD path) and `addHMinvHT(W, J)` (rigid path, M^-1 only). PD path treats W as dense; rigid path treats W as sparse SpGEMM. FBA's case is unambiguously PD (full A inverse), so we only need the dense path. *(No question — just confirming.)*

8. **Q8 (early-exit when contacts.soft_contact_count==0):** Implementation: read `contacts.soft_contact_count[0]` device-to-host once per step, branch to existing PD path. This adds a small CPU sync; for graph capture we may need to graph two variants. **Decision:** Stage A reads device-to-host (simpler). Optimise later if graph capture matters.

## 10. Performance Estimates

### 10.1 Cost model per PD inner iter

Let:
- `N` = particle count
- `M` = active soft contact count
- `nnz(S)` = nnz of sparse-inverse factor S (typically `~20 N` for cloth at N=1000-10K, see `linear_solver.py:466`)
- `T_bsr` = time per BSR SpMV of dimension `(N, N)` — measured ~7 μs at N=10K on RTX 5090 (from existing FBA notes)

Per PD inner iter, contact-aware cost adds:

| Step | Cost | Notes |
|---|---|---|
| **Schur build** (once per `step()`, amortised across PD inner iters) | `M · 3 · 2 · T_bsr` for `A⁻¹·e_p ⊗ n` evaluations  +  `M² · O(1)` reduction | Amortised: `O(M²)` if M small, `O(M · N · log) ` for moderate M. RealSim's Wi reuse trick can reduce to `O(M_new² + M_new · M)` in steady state. |
| **build()**: SpMVs `J·x`, `J^T·(ω·λ)`, augmented A⁻¹·b | `2 · T_bsr × 3 (vec3)` for the system solve + 2 small SpMVs | ~6 BSR SpMVs ≈ 42 μs at N=10K |
| **cstsolve()**: dense `W·Δλ = r` | Stage A: dense Cholesky O(M³). M=100 → 1e6 flops ≈ 0.01 ms | Tiny |
| **applyConstraintCorrection()**: extra `A⁻¹·b_aug` | ~6 BSR SpMVs ≈ 42 μs | One extra system solve per inner iter |

### 10.2 End-to-end estimates per `step()`

Assuming **5 PD inner iters** (`iterations=5`) and **5 NSN inner iters** running inside each PD iter (`contact_iterations=5`):

| Scenario | M | Schur build | Per-iter contact overhead | Total contact-related time | Existing PD baseline (FBA MVP) | Total `step()` |
|---|---|---|---|---|---|---|
| Cloth-on-plane, sparse | 10 | 10·6·7 = 420 μs (build_w) | 5·5·(42+0.01+42) = 2.1 ms | ~2.5 ms | ~0.4 ms | ~2.9 ms |
| Cloth-on-plane, settled | 100 | 100·6·7 = 4.2 ms | 5·5·(42+0.01+42) = 2.1 ms | ~6.3 ms | ~0.4 ms | ~6.7 ms |
| Cloth-on-cylinder, draped | 1000 | 1000·6·7 = 42 ms (uh-oh) | 5·5·(42+0.5+42) = 2.5 ms | ~44.5 ms | ~0.4 ms | ~45 ms |
| Hypothetical extreme | 10000 | 10000·6·7 = 420 ms | 5·5·100 = 2.5 ms | ~422 ms | ~0.4 ms | ~422 ms |

**Read-out:** Stage A (M < 100) is fast enough for real-time. Demo 3 (cloth on cylinder, M~1000) is acceptable but Schur-build-dominated; this is where RealSim's `Wi` reuse pays off (refresh only `|M_new|` columns). **Plan: ship Stage A without reuse; add reuse in a follow-up if Demo 3 misses targets.** For the hypothetical M=10K case (extreme cloth-cloth self-contact), the dense `W` even at M=10K is 800 MB — well past device limits — so we either ship sparse-W path (RealSim has `addHMinvHT_gpu` for sparse-W at line 368, but only for rigid M^-1; the PD-A^-1 case has no sparse path in RealSim either) **or** revert to per-row PCG on `W` (no explicit `W` materialisation). **Mark for Stage D / post-MVP.**

### 10.3 Memory cost

- Dense `W` at `M × M` float64: M=100 → 80 KB; M=1000 → 8 MB; M=10000 → 800 MB. **Set `contact_max` hard cap at ~4096 for Phase 4** (32 MB W), revisit if Demo 3 needs more.
- `apply_sparse_inverse_column` produces a dense (M, 3·N) intermediate: at N=10K, M=100 → 24 MB; M=1000 → 240 MB. **This is the binding constraint** — for Demo 3 we may need to avoid full materialisation and stream column-by-column. *(Implementation detail; flag for Stage A profiling.)*

### 10.4 Comparison anchor with RealSim

Per the existing crosscheck spec (`2026-05-15-fba-realsim-crosscheck-results.md`), the unconstrained FBA PD path already matches RealSim to ~6 decimal places. Phase 4 success is binary: **with the same `dt`, same iterations, same μ, same scene**, the per-step max-penetration vs RealSim should be within ~10% (the LCS sub-iter count and FB vs MinMap choice are the only allowed degrees of freedom).

## 11. Out of Scope (this phase)

- Cloth-cloth and cloth-self contact (Stage D). Requires extending `Contacts` (see §4.4 gap 1) and adding a triangle BVH; reuse VBD's `tri_mesh_collision.py` if possible.
- Rigid-body coupling (FBA-side body integration). FBA assumes shapes are externally driven; rigid solver runs separately.
- CCD/DCD distinction. Newton's `create_soft_contacts` is a DCD predicate; we inherit its limitations (e.g. tunnelling at high velocities). Mitigation: smaller `dt` or larger `soft_contact_margin`.
- Hydroelastic contact (already in Newton for rigids; not relevant for soft-particle contacts).

## 12. Acceptance Criteria

Phase 4 ships when **all** of the following hold:

1. Demo 1 (cloth-on-plane) runs 500 steps with max-penetration < `soft_contact_margin` throughout.
2. Demo 2 (softbody cube on plane) bounces and settles without unbounded oscillation.
3. Demo 3 (cloth on cylinder) drapes and reaches a stable configuration.
4. `test_fba_contact_*` unittest suite (§7.4) passes on CUDA backend.
5. With μ=0 (Stage A) and no contacts active, `step()` is bit-identical to current MVP (regression guard).
6. Performance: at M=100, N=10K, contact-aware `step()` ≤ 10 ms on RTX 5090 (sets the upper bound for "ship-able").

## 13. References

- Newton: `newton/_src/sim/contacts.py:43-359`, `newton/_src/sim/collide.py:464-1231`, `newton/_src/geometry/kernels.py:1001-1141`, `newton/_src/solvers/vbd/solver_vbd.py:2040-2230`, `newton/_src/solvers/vbd/particle_vbd_kernels.py:2900-2967`, `newton/_src/solvers/vbd/rigid_vbd_kernels.py:722-829`, `newton/_src/solvers/fba/solver_fba.py:1-429`, `newton/_src/solvers/fba/linear_solver.py:680-787`, `newton/_src/solvers/solver.py:177-356`, `newton/_src/sim/builder.py:5611-5701`.
- RealSim: `realsim_py/include/Scomponent/contact/ContactPair.h:7-65`, `realsim_py/include/Scomponent/lagrange/LagrangeConstraint.h:14-73`, `realsim_py/include/Scomponent/lagrange/solvers/BaseConstraintSolver.h:72-188`, `realsim_py/src/Scomponent/lagrange/solvers/NonSmoothNewton.cpp:13-424`, `realsim_py/src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp:132-253`, `realsim_py/src/Scomponent/linearsolver/direct/CUDASparseInverseSolver.cpp:34-339`, `realsim_py/src/Scomponent/linearsolver/direct/SparseInverseSolver.cpp:179-184`.
- Prior FBA specs: `docs/superpowers/specs/2026-05-15-fba-solver-design.md`, `2026-05-15-fba-realsim-crosscheck-results.md`, `2026-05-15-fba-stability-diagnosis.md`.
