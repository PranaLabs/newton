# SolverFBA Design Spec (MVP)

- **Date:** 2026-05-15
- **Status:** Draft — pending user review
- **Owner:** ziqiu-zeng
- **Tracking branch:** `ziqiu/fba-solver-design`

## 1. Purpose and Scope

Port the cloth solver from the RealSim engine (`/home/ziqiu/work/RealSim_py/realsim_py`) into Newton as a new solver `SolverFBA` ("Fast But Accurate", the paper name) that sits parallel to `SolverVBD`, `SolverXPBD`, and `SolverStyle3D`. The port is a **Warp-native rewrite**: the new solver operates on Newton's `Model` / `State` objects, uses Warp kernels at runtime, and does not link to RealSim's existing C++/CUDA codebase.

MVP scope is intentionally narrow: a **cloth** simulator with **Pin** constraints and **gravity**, using **Projective Dynamics** (PD) local-global integration with a **Cholesky + sparse-inverse** linear solver that preserves RealSim's signature runtime kernel (two block-sparse SpMVs in place of GPU triangular solve).

### Non-goals (this MVP)

- Softbody (tetrahedral FEM) — handled in a future PR via the same solver class.
- Plane / sphere / box / cylinder / torus contact and CCD/DCD — future PR.
- Coupling with Newton's rigid bodies or other solvers — future PR.
- Differentiable simulation via `wp.Tape()` — design avoids pessimising future diff but does not implement it.
- Per-world heterogeneous topology — all parallel worlds share one PD matrix.
- RealSim's JSON scene loader — Newton's `ModelBuilder` is the sole input path.

## 2. Architecture

`SolverFBA(SolverBase)` is a Warp-native Projective Dynamics solver. Each `step()` performs one implicit-Euler backward step of cloth dynamics via PD local-global iteration:

1. **Predict (warm-start)**
   `x_inertia = x_prev + dt · v_prev + (dt² / m) · f_ext`. The PD inner loop starts from `x⁰ = x_inertia`.

2. **Local step (parallel over elements)**
   For each cloth triangle, compute the ARAP projection of its 3×2 deformation gradient onto the rest manifold (SVD followed by clamping singular values to 1). For each bending edge (4-vertex stencil), compute the isometric bending local target (Bergou et al. 2006). For each pinned particle, the projection is its reference position `x_ref`.

3. **Global step (one sparse linear solve)**
   Assemble the right-hand side
   `b = (M/dt²) · x_inertia + Σ_e Aᵀ_e · w_e · proj_e(F_e) + Σ_pin w_pin · x_ref`,
   then solve `A · x = b` where the PD Hessian
   `A = M/dt² + Σ_e w_e · Aᵀ_e · A_e + Σ_pin w_pin · I`
   is constant when topology, dt, stiffness and pinning are fixed.

4. **PD outer iteration** — repeat steps 2–3 K times (default `iterations=10`), yielding the final position `x_new`.

5. **Velocity write-back**
   `v_new = (x_new − x_prev) / dt`, write `state_out.particle_q` and `state_out.particle_qd`.

### 2.1 Linear solver — Cholesky + sparse inverse

The constant PD Hessian `A` is factorised once at solver construction. The runtime per-step linear solve uses RealSim's signature technique: precompute the sparse matrix `S = L⁻¹` (where `A = L D Lᵀ`) and execute the solve as two sparse mat-vecs plus a diagonal scaling:

`A⁻¹ b  =  Lᵀ⁻¹ D⁻¹ L⁻¹ b  =  Sᵀ · D⁻¹ · (S · b)`

This avoids the inherently sequential GPU triangular solve. Pipeline:

- **Setup (CPU, once)** via Python + SciPy:
  1. Assemble `A` as `scipy.sparse.csr_matrix` (scalar N×N, isotropic).
  2. `scipy.sparse.linalg.splu(A, permc_spec='COLAMD')` — provides ordering + LU factors. Because `A` is SPD, the LU factors are equivalent to LDLᵀ; extract `L`, `U`, the permutation arrays.
  3. Compute `S = L⁻¹` keeping the elimination-tree-derived sparsity pattern. This is a **pure-Python port** of RealSim's `LDLT_computeLowerInverse` (file `src/Scomponent/tools/math/SparseLDLT.cpp`).
  4. Upload `S`, `Sᵀ`, `D⁻¹`, `perm`, `invperm` to the device as Warp arrays / BSR matrices.
- **Per-step (GPU)**:
  1. Permute RHS scalar component by row order.
  2. `bsr_mv(S, b)` → intermediate vector.
  3. Multiply elementwise by `D⁻¹`.
  4. `bsr_mv(Sᵀ, ...)` → solution component.
  5. Un-permute. Repeat for x, y, z components (matrix is shared; `vec3` decomposed into three scalar solves — see §2.3).

### 2.2 METIS is dropped

RealSim's `LDLT_ordering` calls `METIS_NodeND` for fill-reducing nested-dissection ordering. We do **not** carry over METIS. SciPy's SuperLU uses COLAMD by default; on cloth-like (quasi-2D) meshes the fill-in penalty vs METIS is typically 10–30 %, negligible for MVP-scale problems. The paper's distinctive technical kernel — the elimination-tree-derived `S = L⁻¹` — is independent of ordering and is preserved verbatim.

### 2.3 Scalar N×N matrix layout

The PD Hessian for isotropic cloth has identical structure and values along x, y, z (each non-zero block is `α · I₃`). We store `A`, `S`, `Sᵀ` as scalar `N × N` matrices and execute the global solve three times (once per component). Rationale: a standard 3×3-block BSR with diagonal blocks would store nine values per non-zero entry (9× memory traffic) while only one is meaningful; storing scalar N×N is 3× cheaper in bandwidth. Three kernel launches add ≪1 ms of launch latency, negligible at 60 Hz. When (future) anisotropic terms couple x/y/z (friction, ADMM friction, anisotropic constitutive models), the design will revisit this.

### 2.4 Material model extension hook

PD's local-global structure makes constitutive-model swaps essentially free at the Hessian level. Only the per-element projection kernel changes between ARAP, Corotational, Neo-Hookean, and St. Venant–Kirchhoff. MVP exposes the dispatch point but implements only ARAP.

`SolverFBA.__init__` takes `stretching_model: Literal["arap"] = "arap"` and raises `NotImplementedError` for any other value. The `kernels.py` module follows a naming convention `project_stretching_<model>_kernel` with a uniform signature so future variants are drop-in additions. A `_tri_params: wp.array[wp.vec4]` slot is allocated (16 bytes / triangle, e.g. 160 KB for 10K triangles) to hold per-triangle `(μ, λ, …)` for future models; ARAP ignores it.

### 2.5 Constraint solver interface

Not pre-reserved in MVP, per Newton's "no premature abstraction" rule (`AGENTS.md`). Future contact / hard-constraint work will add Schur-complement helpers (`addHAinvHT`, `apply_constraint`, modelled after RealSim's `CUDASparseInverseSolver`) and the `update_contacts()` override in a focused PR. To keep that PR small, the `FBALinearSolver` exposes its constants (`S_bsr`, `ST_bsr`, `Dinv`, `perm`, `invperm`) as plain readable attributes so future Schur code can build `J · A⁻¹ · Jᵀ` without further refactor.

## 3. Components

```
newton/_src/solvers/fba/
├── __init__.py            # exports SolverFBA
├── solver_fba.py          # SolverFBA(SolverBase) — assembly, step, lifecycle
├── kernels.py             # Warp kernels for predict/project/scatter/velocity
└── linear_solver.py       # Cholesky+sparse-inverse setup + FBALinearSolver
```

Public surface added to `newton/_src/solvers/__init__.py` and re-exported from `newton/solvers.py` as `SolverFBA`.

### 3.1 `solver_fba.py` — `class SolverFBA(SolverBase)`

```python
class SolverFBA(SolverBase):
    """Fast But Accurate projective-dynamics cloth solver.

    PD local-global with prefactored constant Hessian; runtime solve uses a
    sparse inverse S = L⁻¹ for two GPU SpMVs in place of triangular solve.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        pin_stiffness: float = 1e12,
        stretching_model: Literal["arap"] = "arap",
    ): ...

    def step(self, state_in, state_out, control, contacts, dt) -> None: ...
    def notify_model_changed(self, flags: int) -> None: ...
    # update_contacts → raises NotImplementedError in MVP
```

Setup work in `__init__` (one-time):

1. Validate inputs (see §5).
2. Compute per-triangle rest data: `restMatrix = (basisᵀ · edges)⁻¹` (2×2), `area = det(...) / 2`. Stored on device as `wp.array[wp.mat22]`, `wp.array[float]`. Per-triangle weight `w_i = ke · area_i`.
3. Compute per-bending-edge isometric quadratic form (4×4 matrix per edge) and weight.
4. Identify pinned particles by `model.particle_inv_mass[i] == 0` (matching Newton's `add_cloth_grid(fix_left=…)` convention). Build `_pin_idx`, `_x_ref` from initial `model.particle_q`.
5. Assemble scalar N×N PD Hessian `A` on CPU (`scipy.sparse.csr_matrix`):
   `A_diag += m_i / dt²` for free particles;
   `A += Σ_tri w_i · K_i` (3×3 PSD block `K_i = G_i · G_iᵀ` with `G_i = ST · restMatrix_i`, broadcast onto particle indices in scalar layout);
   `A += Σ_edge w_e · Q_e` (4×4 PSD block);
   `A += Σ_pin w_pin · I` (diagonal).
6. `splu(A, permc_spec='COLAMD')` → extract `L`, diagonal `D`, `perm_r`, `perm_c`.
7. Port of `LDLT_computeLowerInverse` (Python+NumPy) → produces `S` as `scipy.sparse.csr_matrix`.
8. Upload to device: `S_bsr`, `ST_bsr` (1×1 BSR), `Dinv`, `perm`, `invperm` arrays.
9. Allocate per-step working buffers (§4).

Step work in `step()`:

```
compute_inertial_kernel        (per particle: x_inertia from x_prev, v_prev, f_ext, gravity, dt)
for k in range(iterations):
    zero_vec3_kernel           (clear rhs)
    add_inertia_to_rhs_kernel  (per particle: rhs += (m/dt²)·x_inertia)
    project_pin_kernel         (per pinned particle: atomic_add(rhs[i], w_pin·x_ref))
    project_stretching_arap_kernel  (per triangle: SVD → scatter w·Aᵀ·proj into RHS)
    project_bending_kernel     (per edge: 4-vertex stencil → scatter)
    linear_solver.solve(rhs → x_cur)  (component-wise: apply_perm → bsr_mv(S) → ×D⁻¹ → bsr_mv(Sᵀ) → apply_invperm)
write_velocity_kernel          (per particle: v = (x_new − x_prev) / dt)
write state_out.particle_q = x_cur, state_out.particle_qd = v
```

`notify_model_changed` rebuilds `A → S → BSR` from scratch on `SHAPE_PROPERTIES` / `BODY_INERTIAL_PROPERTIES`. Pure re-allocation; old buffers replaced atomically after success so a transient failure leaves the solver functional.

### 3.2 `kernels.py` — Warp kernels

Each kernel has a single clear purpose; signatures use bracket Warp array typing per `AGENTS.md`.

- `compute_inertial_kernel` — per particle: `x_inertia = x_prev + dt·v_prev + (dt²/m)·(f_ext + g_world)`.
- `zero_vec3_kernel` — clear RHS.
- `add_inertia_to_rhs_kernel` — per particle: `rhs += (m/dt²) · x_inertia`.
- `project_pin_kernel` — per pinned particle: `atomic_add(rhs[idx], w_pin · x_ref)`.
- `project_stretching_arap_kernel` — per triangle: 3×2 SVD of `F`, clamp singular values to 1, reconstruct projection `P`, scatter `w · restMatrixᵀ · Pᵀ` into 3 vertex rows of RHS via `atomic_add`.
- `project_bending_kernel` — per edge (4-vertex stencil): evaluate isometric quadratic, scatter.
- `apply_permutation_kernel` — gather `y[i] = x[perm[i]]`.
- `scale_by_diag_kernel` — `y[i] = D⁻¹[i] · x[i]`.
- `write_velocity_kernel` — `v = (x_new − x_prev) / dt`.

A `@wp.func svd_3x2(F: wp.mat32) -> (U, sigma, V)` helper is added in `kernels.py` (reused by future material models).

### 3.3 `linear_solver.py` — three responsibilities

1. **`build_pd_system(model, dt, pin_stiffness) -> (scipy.sparse.csr_matrix, dict)`** — CPU assembly of scalar N×N `A` plus auxiliary maps (pin indices, triangle weights, etc.) ready for upload.
2. **`factorize_and_sparse_inverse(A) -> (S_csr, ST_csr, Dinv, perm, invperm)`** — wraps `scipy.sparse.linalg.splu` for ordering + numeric factorization, then runs the Python port of `LDLT_computeLowerInverse`.
3. **`class FBALinearSolver`** — holds the device BSR matrices and exposes `solve(b: wp.array[wp.vec3], x: wp.array[wp.vec3])`. Internally: per-component apply-perm → bsr_mv(S) → scale by `D⁻¹` → bsr_mv(Sᵀ) → apply-invperm. Returns nothing; writes into `x`.

## 4. Data flow

### 4.1 Read from `Model` (setup, one-time)

| Field | Use |
|---|---|
| `model.particle_q` | initial positions → triangle `restMatrix`, `area`; pin `x_ref` |
| `model.particle_mass` | PD Hessian `M/dt²` diagonal |
| `model.particle_inv_mass` | pin detection (`inv_mass == 0`) |
| `model.tri_indices` | triangle topology |
| `model.tri_materials` / cloth stiffness fields | per-triangle `ke` |
| `model.edge_indices` | bending 4-vertex stencil |
| `model.edge_bending_properties` | per-edge bending stiffness |
| `model.particle_world` | per-world gravity indexing |
| `model.gravity` | per-world gravity vectors |

### 4.2 Per `step()`

| In | Out |
|---|---|
| `state_in.particle_q` (read) | `state_out.particle_q` (write) |
| `state_in.particle_qd` (read) | `state_out.particle_qd` (write) |
| `state_in.particle_f` (read) | — |
| `control`, `contacts` | ignored in MVP |

### 4.3 Solver-owned device buffers

```
self._x_ref           wp.array[wp.vec3]   pin reference positions
self._pin_idx         wp.array[int32]     packed pin indices
self._tri_rest_inv    wp.array[wp.mat22]
self._tri_area        wp.array[float]
self._tri_weight      wp.array[float]
self._tri_params      wp.array[wp.vec4]   reserved (μ,λ,...) for future models
self._edge_quad       wp.array[wp.mat44]  isometric bending quadratic form
self._edge_weight     wp.array[float]
self._x_inertia       wp.array[wp.vec3]   warm-start
self._x_cur           wp.array[wp.vec3]   current PD iterate
self._rhs             wp.array[wp.vec3]   RHS accumulator
# Linear-solver constants (re-built on notify_model_changed):
self._linear_solver   FBALinearSolver
  ._S_bsr             wp.sparse.BsrMatrix
  ._ST_bsr            wp.sparse.BsrMatrix
  ._Dinv              wp.array[float]
  ._perm              wp.array[int32]
  ._invperm           wp.array[int32]
  ._tmp_b_perm        wp.array[float]     scratch per-component
  ._tmp_Sb            wp.array[float]
  ._tmp_x_perm        wp.array[float]
```

### 4.4 Multi-world

`model.particle_world[i]` indexes `model.gravity[w]`, matching Style3D/VBD. MVP requires all worlds to share topology / dt / stiffness so a single `A` (and `S`) suffices. The solver does not branch on per-world topology; that's a future extension.

### 4.5 In-place safety

- `state_in` is read-only.
- `state_out` is written only at the end of `step()`.
- `_x_cur` is mutated inside the PD loop but is independent of `state_in/out`, so the standard ping-pong pattern (`state_in, state_out = state_out, state_in` between steps) works.
- No `wp.synchronize()` inside `step()`.

## 5. Error handling and lifecycle

### 5.1 `__init__` (hard errors)

| Condition | Action |
|---|---|
| `model.particle_count == 0` | `raise ValueError` |
| `model.tri_count == 0` | `raise ValueError` (MVP is cloth-only) |
| `stretching_model != "arap"` | `raise NotImplementedError` |
| `pin_stiffness <= 0` | `raise ValueError` |
| `iterations < 1` | `raise ValueError` |
| `splu` failure | propagate scipy's exception |
| `S` contains non-finite | `raise RuntimeError` with hint to check `pin_stiffness`, `ke`, `dt` |

No deep diagnostics on splu failure in MVP — tests catch most issues; if a user hits an obscure singular case, the bug is small and focused.

### 5.2 `step()` (hot path)

No exceptions, no NaN guards, no `wp.synchronize()`. Newton convention: trust kernels + cover with tests.

### 5.3 `notify_model_changed(flags)`

| Flag | Response |
|---|---|
| `SHAPE_PROPERTIES` | full re-setup: re-read positions, rebuild `A`, re-factorize, re-compute `S`, re-upload. |
| `BODY_INERTIAL_PROPERTIES` | full re-setup (mass enters `A`). |
| `MODEL_PROPERTIES` | refresh `gravity` array only. |
| `JOINT_*` | ignored. |

Re-setup is non-mutating until success: if any step fails, prior `_linear_solver` remains valid.

### 5.4 `update_contacts(contacts, state)`

Raises `NotImplementedError("Contact-aware FBA solver TBD; MVP supports gravity + pin only")`.

## 6. Testing

Tests live in `newton/tests/test_solver_fba.py` (Newton's convention: one file per solver, `unittest`, no pytest). Eight tests, each targeting a specific failure mode.

| # | Test | Targets |
|---|---|---|
| T1 | `test_pd_hessian_assembly` | geometry / weight / row indexing errors in `build_pd_system` |
| T2 | `test_sparse_inverse_correctness` | algorithmic correctness of `compute_lower_inverse` (compare to `np.linalg.inv(L.toarray())` on small random SPD) |
| T3 | `test_linear_solver_end_to_end` | full setup pipeline + Warp BSR runtime: random SPD `A`, random `b`, assert `‖A·x − b‖ / ‖b‖ < 1e-6` |
| T4 | `test_arap_projection_kernel` | per-triangle SVD-and-clamp logic in Warp kernel |
| T5 | `test_bending_projection_kernel` | 4-vertex stencil indexing: flat configuration → zero bending RHS contribution |
| T6 | `test_permutation_roundtrip` | `apply_perm` ∘ `apply_invperm` = identity (catches direction-swap bugs) |
| T7 | `test_smoke_cloth_hanging` | 50-step run of pinned hanging cloth: no NaN/Inf in final state |
| T8 | `test_steady_state_cloth_hanging` | 32×32 cloth, four corners pinned, 500 steps: final center vertex within 5 % of SolverVBD's result under identical setup |

Plus one example: `newton/examples/cloth/example_cloth_hanging_fba.py`, following Newton's `Example` class format; `test_final()` reuses T8's criterion. Verified by `uv run -m newton.examples cloth_hanging_fba --viewer null --test`.

Not separately tested (covered by T7/T8): `zero_vec3_kernel`, `add_inertia_to_rhs_kernel`, `scale_by_diag_kernel`, `write_velocity_kernel`. They are trivial — a regression there fails the smoke test immediately.

## 7. Conventions and dependencies

- **License:** Apache 2.0, with SPDX header on every new file (`Copyright (c) 2026 The Newton Developers`).
- **No new runtime dependencies** beyond Newton's existing stack: Warp, NumPy, SciPy (already used by Newton's broader test/example suite). No METIS, no scikit-sparse, no native code.
- **Naming:** `SolverFBA` (prefix-first per `AGENTS.md`).
- **Style:** PEP 604 unions, `wp.array[X]` bracket annotations, Google-style docstrings with SI-unit notes for physical fields.
- **Tests:** `unittest`, never call `wp.synchronize()` before `.numpy()`.
- **Branch:** `ziqiu/fba-solver-design` → implementation branches `ziqiu/fba-solver-mvp`, `ziqiu/fba-solver-tests` to keep PRs focused.

## 8. Implementation phases (preview)

The implementation plan (next step, separate doc) is expected to break MVP into:

1. **Linear-solver core** — `linear_solver.py` + T1/T2/T3. Headless, no Warp kernels yet beyond BSR plumbing.
2. **Element kernels** — `kernels.py` ARAP + bending + helpers, T4/T5/T6.
3. **Solver integration** — `solver_fba.py`, exports, T7/T8 + example.

Each phase is a separate PR.

## 9. References

- Bouaziz, S., Martin, S., Liu, T., Kavan, L., Pauly, M. *Projective Dynamics: Fusing Constraint Projections for Fast Simulation.* SIGGRAPH 2014.
- Liu, T., Bargteil, A. W., O'Brien, J. F., Kavan, L. *Fast Simulation of Mass-Spring Systems.* SIGGRAPH Asia 2013.
- Liu, T., Bouaziz, S., Kavan, L. *Quasi-Newton Methods for Real-Time Simulation of Hyperelastic Materials.* TOG 2017.
- Bergou, M., Wardetzky, M., Robinson, S., Audoly, B., Grinspun, E. *Discrete Elastic Rods / Isometric Bending.* SIGGRAPH 2006/2008.
- RealSim source: `/home/ziqiu/work/RealSim_py/realsim_py` (private). Key files referenced:
  `src/Scomponent/tools/math/SparseLDLT.cpp` (LDLᵀ + sparse inverse),
  `src/Scomponent/integrator/localglobal/energy/elastic/PDTriangleStretchingEnergy.cpp`,
  `include/Scomponent/integrator/localglobal/energy/elastic/PDARAPTriangleEnergy.h`,
  `src/Scomponent/integrator/localglobal/energy/hardconstraint/PinEnergy.cpp`.
- Style3D solver (`newton/_src/solvers/style3d/`) — Newton's existing PD-based cloth solver, used as a reference for kernel structure, BSR plumbing, and multi-world conventions.
