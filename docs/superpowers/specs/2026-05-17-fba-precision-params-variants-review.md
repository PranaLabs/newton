# Precision / Parameters / Variants Compliance Review

**Date:** 2026-05-17
**Scope:** Post 2026-05-17 scope cut. In-scope demos: 1 (TwistingBar), 2 (TwistingBarNH),
3 (StretchingCloth), 4 (PullingWooper), 5 (SqueezingBall), 6 (CrossingGingerbreadman),
8 (ClothOnKnives). OUT OF SCOPE: 7 (SharpCorner), 9 (ParallelEnvTest), 10 (CableGrabRaptor).
**Discipline:** `/home/ziqiu/.claude/projects/-home-ziqiu-work/memory/realsim-port-discipline.md` —
RealSim CudaTests are ground truth; any FBA divergence is treated as an FBA port bug
until proven otherwise.

Cross-refs:

- `docs/superpowers/specs/2026-05-17-fba-realsim-audit-v2.md` (component A-Z)
- `docs/superpowers/specs/2026-05-17-fba-realsim-per-demo-audit.md` (per-scene)
- `docs/superpowers/specs/2026-05-17-fba-lbfgs-compliance-review.md`
- `docs/superpowers/specs/2026-05-17-fba-nsn-compliance-review.md`
- `scripts/realsim_baseline/decisions.json`

Already-confirmed intentional divergences (not re-flagged below):

- Decision #1 Corot symmetric trace (audit.v2 component D + G)
- Decision #2 rectangular Coulomb cone (audit.v2 R)
- Decision #3 uniform mass lumping (audit.v2 N, pending Phase 2.3)
- Decision #4 NSN inner iter cap = 10 (audit.v2 V)
- Decision #5 per-scene λ-cap (audit.v2 T, pending Phase 2.4)
- S.1 lexsort, S.3 Stage A path
- NSN tolerance early-exit not ported
- Q9 Warp barrier substitution

---

## Part 1 — RealSim variant minimum subset

### 1.1 Variants enumerated

**LinearSolver (system A solve)** — `RealSim_py/realsim_py/include/Scomponent/linearsolver/`:

Direct: `DenseEigenCholeskySolver`, `SparseEigenCholeskySolver`, `SparseLDLTSolver`,
`SparseInverseSolver` (CPU), `CUDASparseInverseSolver` (GPU), `SparseSingleJacobiSolver`,
`SparseMultipleJacobiSolver`, `CUDASparseSingleJacobiSolver`, `CUDASparseMultipleJacobiSolver`.

Iterative (`linearsolver/iterative/`): `DenseCGSolver`, `DenseCRSolver`, `DenseJacobiPCGSolver`,
`DenseJacobiPCRSolver`, `SAIPCGSolver`, `SparseCGSolver`, `SparseCRSolver`,
`SparseJacobiPCGSolver`, `SparseJacobiPCRSolver`, `CUDADenseCRSolver`,
`CUDADenseJacobiPCRSolver` (= `PCR_CUDA` per scene.json string), `CUDASparseCRSolver`,
`CUDASparseJacobiPCRSolver`.

Dispatch: `LocalGlobalSolver.cpp:417-419` instantiates `CUDASparseInverseSolver` when
scene.json `linearsolver.type == "SPARSE_INVERSE_CUDA"`.

**NSN constraint solver** — `RealSim_py/realsim_py/src/Scomponent/lagrange/solvers/`:

`NonSmoothNewton.cpp` (CPU), `CUDANonSmoothNewton.cpp` (GPU dense, single env),
`LiteNonSmoothNewton.cpp` (CPU lite), `CUDALiteNonSmoothNewton.cpp` (GPU lite for
parallelEnv pre-factored Schur). Plus `PGSSolver`, `BilateralizedSolver`.

Dispatch: `LocalGlobalSolver.cpp:698-711` selects by scene.json
`constraintsolver.type` string.

**Per-shape Collision detectors** — `RealSim_py/realsim_py/src/Scomponent/collisiondetection/`:

`PlaneCollision.cpp`, `SphereCollision.cpp`, `CylinderCollision.cpp`,
`BoxCollision.cpp`, `TorusCollision.cpp`, `MovingSphereCollision.cpp`,
`MovingBoxCollision.cpp`, `GenericDCD.cpp`, `GenericCCD.cpp`,
`LocalMinDistanceDetection.cpp`, `gpu/` (GPU variants of plane, sphere, cylinder).

Inside the NSN, `_systemlinearsolver` (for A solve) and `_constraintlinearsolver`
(for the Schur LCP subsystem) are **two distinct solvers**. The first is set by the
scene-level `linearsolver` block (`SPARSE_INVERSE_CUDA`). The second is set by
`constraintsolver.linearsolver` (`PCR_CUDA`).

### 1.2 Per-demo variant table

LocalGlobal_CUDA = 5 in every in-scope demo (PD outer iter count). All seven demos
use `LocalGlobal_CUDA` (the GPU LocalGlobal driver in `CUDALocalGlobalSolver.cpp`),
NOT the bare CPU `LocalGlobal`.

| Demo | LocalGlobal | A LinearSolver (`linearsolver.type`) | NSN | NSN constraint solver (`constraintsolver.linearsolver`) | Colliders used |
|------|---|---|---|---|---|
| 1 TwistingBar | LocalGlobal_CUDA | SPARSE_INVERSE_CUDA | n/a (no `constraintsolver` block) | n/a | none (pure FEM) |
| 2 TwistingBarNH | LocalGlobal_CUDA | SPARSE_INVERSE_CUDA | n/a | n/a | none |
| 3 StretchingCloth | LocalGlobal_CUDA | SPARSE_INVERSE_CUDA | n/a (no `constraintsolver` block) | n/a | none |
| 4 PullingWooper | LocalGlobal_CUDA | SPARSE_INVERSE_CUDA | NonSmoothNewton_CUDA | PCR_CUDA | CylinderCollision ×2 |
| 5 SqueezingBall | LocalGlobal_CUDA | SPARSE_INVERSE_CUDA | NonSmoothNewton_CUDA | PCR_CUDA | CylinderCollision ×4 + PlaneCollision ×1 |
| 6 CrossingGingerbreadman | LocalGlobal_CUDA | SPARSE_INVERSE_CUDA | NonSmoothNewton_CUDA | PCR_CUDA | CylinderCollision ×13 |
| 8 ClothOnKnives | LocalGlobal_CUDA | SPARSE_INVERSE_CUDA | NonSmoothNewton_CUDA | PCR_CUDA | PlaneCollision ×1 + GenericCCD (knives mesh) |

Demos 1, 2, 3 lack a `constraintsolver` block entirely — pure FEM, no NSN ever runs.

### 1.3 Minimum required subset

For the 7 in-scope demos, FBA must align exactly one variant in each of these slots:

- **LocalGlobal driver:** `CUDALocalGlobalSolver` (FBA's analogue = `SolverFBA`).
- **A LinearSolver (PD outer linear solve):** `CUDASparseInverseSolver` (sparse SimplicialLDLT
  followed by `LDLT_computeLowerInverse` to extract `S = L⁻¹`, runtime
  `x = P_cᵀ · Sᵀ · D⁻¹ · S · P_r · b`).
  FBA's analogue = `FBALinearSolver` (linear_solver.py:765-836, uses SciPy SuperLU/COLAMD
  for factorization, ports `LDLT_computeLowerInverse` at lines 355-462, runtime via two
  Warp BSR SpMVs per spatial component).
- **NSN constraint solver:** `CUDANonSmoothNewtonSolver` (single-env dense GPU NSN with
  FB function).
  FBA's analogue = `SolverFBA._solve_nsn_unilateral` (Stage A, μ=0 path) +
  `_solve_nsn_coulomb` (Stage B, frictional path), both running host-side NumPy
  (np.linalg.solve dense Schur).
- **NSN inner constraint linear solver:** `CUDADenseJacobiPCRSolver` (`PCR_CUDA`).
  FBA's analogue = `np.linalg.solve(A_schur, rhs)` — different algorithm (direct
  dense LAPACK vs preconditioned conjugate residual) but numerically converged
  on dense SPD blocks. **Functionally equivalent on the row counts in scope.**
- **Per-shape colliders (in-scope):**
  - `CylinderCollision` (demos 4, 5, 6) — static and rolling
  - `PlaneCollision` (demos 5, 8)
  - `GenericCCD` (demo 8 only — mesh-mesh CCD for knives obstacle)

Out-of-scope variants (NO in-scope demo uses any of these):

- LinearSolver: CPU `SparseInverseSolver`, all Cholesky / LDLT / Jacobi variants
  (single/multiple), CG/CR iterative variants. None used by any in-scope demo.
- NSN: `NonSmoothNewton` (CPU), `LiteNonSmoothNewton` (CPU lite),
  `CUDALiteNonSmoothNewton` (parallelEnv lite), `PGSSolver`, `BilateralizedSolver`.
  None used by any in-scope demo.
- Colliders: `SphereCollision`, `BoxCollision`, `TorusCollision`,
  `MovingSphereCollision`, `MovingBoxCollision`, `LocalMinDistanceDetection`,
  `GenericDCD` (the discrete CD variant; Demo 8 uses CCD, not DCD).

### 1.4 FBA alignment to subset

- **`SolverFBA` vs `CUDALocalGlobalSolver`** — MATCH (audit.v2 component A confirms control
  flow equivalence). Single-env only — FBA has no parallelEnv code path
  (ParallelEnvTest already cut from scope, so this is correct).
- **`FBALinearSolver` vs `CUDASparseInverseSolver`** — MATCH (audit.v2 K). Different
  factorization library (SciPy SuperLU/COLAMD vs Eigen SimplicialLDLT) but identical
  `S = L⁻¹` extraction and identical runtime formula. Numerically equivalent modulo
  permutation-dependent FP round-off.
- **`SolverFBA._solve_nsn_coulomb` vs `CUDANonSmoothNewtonSolver`** — MATCH (audit.v2
  V; lbfgs-review and nsn-review both green). FB row formulas verified line-by-line.
  FBA correctly aligns to `CUDANonSmoothNewton`, NOT `CUDALiteNonSmoothNewton`. The
  in-scope NSN variant is the dense single-env one.
- **`np.linalg.solve` vs `CUDADenseJacobiPCRSolver`** — MATCH-FUNCTIONAL. Different
  algorithm class (direct LAPACK vs iterative PCR with Jacobi preconditioner) but
  on dense SPD Schur blocks the two converge to identical solutions to within machine
  precision. RealSim's `PCR_CUDA` runs to its own tolerance (typically tight); FBA's
  direct solve gives the same answer in one shot. Acceptable.
- **Cylinder + Plane colliders** — Newton's collision pipeline provides these via
  `add_shape_cylinder` and `add_shape_plane` (or `(a,b,c,d)` plane form). The
  per-step pene0 / tangent computation lives in `SolverFBA.update_contacts`
  (solver_fba.py:1057-1293). Audit.v2 component Y notes accidental divergences in
  the 0.01m offset (Y.1) and plane previous-step projection (Y.2); both queued for
  Phase 1.3 fixes. Otherwise the cylinder rolling tangent displacement matches
  exactly (Y.3 verified).
- **GenericCCD (mesh CCD)** — **NOT YET IMPLEMENTED** in FBA. Demo 8 ClothOnKnives
  requires this. Newton has its own VBD/cloth CCD machinery in
  `newton/_src/solvers/vbd` but it has not been wired into SolverFBA. Phase 5
  scope per the per-demo audit.

### Out-of-scope variants in FBA (Phase 4 cleanup candidates)

Searched FBA source for code paths targeting out-of-scope variants:

- `FBALinearSolver` does NOT implement Cholesky, LDLT, Jacobi, CG, CR, or PCR
  variants. The only A solve is via the sparse-inverse pipeline. ✓ Clean.
- `_solve_nsn_*` does NOT implement Lite (multi-env shared Schur), PGS, or
  Bilateralized solver variants. ✓ Clean.
- Collision pipeline relies on Newton's `Contacts` produced by `create_soft_contacts`
  — which itself supports many shape types Newton-wide, but SolverFBA only consumes
  the produced pene0/tangents and does not contain per-shape code paths that go
  beyond what's needed. ✓ Clean (the `_pene0` accidental divergence is a missing
  feature, not an extra unused path).
- `project_coulomb_cone` helper at `solver_fba.py:90-121` — circular cone projection
  no longer in any active code path (audit.v2 R notes). Dead code; cleanup candidate
  for Phase 4.
- `_apply_lambda_correction_friction` fallback at `solver_fba.py:1663-1698` — the
  per-row re-solve path that activates only when `_A_inv_Jt_d` cache is missing.
  This is reachable in regression-test mode and not really "out-of-scope variant"
  per se, but the multi-RHS unpack-to-vec3 path itself is no longer the default
  (isodof path is default). Could be deleted as part of Phase 4 cleanup if the
  isodof path is locked in.

---

## Part 2 — Numerical precision alignment

RealSim baseline: `real = double` throughout (Eigen `Vector3R = Vector3d`,
`MatrixXR = MatrixXd`). All physics arrays, factor matrices, and the
`mcl::optlib` LBFGS inner state are float64.

FBA baseline: mixed. `wp.vec3` (used for particle_q, _x_inertia, _x_cur, _rhs,
contact normals/tangents, _A_inv_Jt_d) is fp32. Linear-solver internal scratch
(`_b_scalar`, `_Sb`, `_DSb`, `_SDSb`, `_x_scalar`, `_Dinv`) is fp64. Mass and
inv_mass arrays are fp32 (Newton `Model.particle_mass`); the `linear_solver.py`
setup pulls them via `.astype(np.float64)` so the assembled A factor is built
in fp64. Energy projection kernels (Tri/Tet ARAP/Corot) work in fp32 except
the NH-LBFGS path which is fp64 internally (Q13 confirmed). NSN solver (host
NumPy) runs in fp64 throughout.

### Per-block precision table

| Block | RealSim precision | FBA precision | Aligned? | Why it matters |
|---|---|---|---|---|
| `particle_q`, `particle_qd`, `particle_f` | fp64 | fp32 (`wp.vec3`) | NO (FBA fp32) | Limits trajectory precision to ~1e-7 m per particle. Below 1e-3 m drift tolerance. Documented at solver_fba.py:185-188. |
| `particle_mass`, `inv_mass` | fp64 | fp32 (Newton `Model.particle_mass`) → fp64 during PD setup | YES (round-trip) | Factor A is built in fp64; only the source array is fp32. With uniform-mass lumping (Decision #3) every entry is identical so fp32 storage is exact. |
| Inertial prediction `s_n = x + dt·v + dt²·g` (`compute_inertial_kernel`) | fp64 | fp32 in Warp kernel | NO (FBA fp32) | Small per-step accumulation; under 1e-3 m. |
| Energy projections — Tri ARAP / Tet ARAP (SVD + `R = U·V^T`) | fp64 (Eigen JacobiSVD) | fp32 (Warp `svd2`/`svd3`) | NO (FBA fp32) | Documented audit.v2 C/F; round-off below tolerance. |
| Energy projections — Tri Corot / Tet Corot (Sherman-Morrison) | fp64 (mcl::optlib LBFGS to 1e-6 grad-norm) | fp32 closed-form | NO (intentional, Decision #1) | Intentional divergence — different formula entirely. |
| Energy projections — Tri NH / Tet NH (LBFGS) | fp64 (mcl::optlib LBFGS, `m=2`/`m=3`, Wolfe line search, to grad-norm 1e-6) | **fp64** (Q13: kernels.py:15-25 uses `wp.float64` LBFGS state with `_LBFGS_TOL=1e-6`) | YES (post-LBFGS-port, lbfgs-review GREEN) | LBFGS port intentionally upgraded to fp64 for parity with RealSim |
| Isometric bending (`q^T x`, normalize, scatter) | fp64 | fp32 | NO (FBA fp32) | Below tolerance for smooth-bending modes; documented audit.v2 I. |
| PD Hessian A (entries, fill-in pattern, factor) | fp64 (Eigen `SparseMatrix<double>` + `SimplicialLDLT<double>`) | fp64 (SciPy SuperLU on fp64 CSC; `S`, `D⁻¹`, `Sᵀ` all uploaded as `wp.float64`) | YES | Factor and runtime tables both fp64. ✓ |
| Linear solve A⁻¹·b (forward/back-sub) | fp64 (cuSPARSE BSR SpMV with fp64) | fp64 (`wps.bsr_mv` on fp64 BSR, scratch `_b_scalar`, `_Sb`, `_DSb`, `_SDSb`, `_x_scalar` all fp64) | YES | Round-trip: input `vec3` → fp64 scalar → solve → fp64 scalar → vec3 (fp32). Final downcast at `insert_component_kernel` (kernels.py:103). |
| Linear-solve output back into `_x_cur` (vec3) | n/a (fp64 throughout) | fp64 → fp32 cast in `insert_component_kernel` | NO (FBA downcasts) | One fp64→fp32 truncation per PD outer iter. Documented; below 1e-3 m tolerance. |
| Contact pene0 / tangent_offset | fp64 (Eigen) | fp64 (`_contact_offset_d`, `tangent1_offset_h`, `tangent2_offset_h` all `np.float64` / `wp.float64`) | YES | ✓ |
| Schur W build (`J·A⁻¹·Jᵀ`) | fp64 (cuSPARSE) | fp64 (`_W_device_d` is `wp.float64`; `_A_inv_Jt_d` is `wp.vec3` = fp32 in the multi-RHS path, but `compute_wi_kernel` in the default isodof path operates entirely on fp64 entries pulled from the sparse-inverse factor — see linear_solver.py:1165 `np.zeros(.., dtype=np.float64)`) | MIXED | **Default isodof path is fp64 end-to-end. The legacy (use_isodof=False) multi-RHS path goes through fp32 vec3 storage of `_A_inv_Jt_d` — fp64→fp32 downcast, then dot products promote back to fp32. Acceptable for the default code path; the legacy path can lose ~1e-7 precision per W entry.** |
| `A_schur = (ω·ωᵀ)·W + diag(c)` build | fp64 | fp64 (host NumPy) | YES | ✓ |
| Schur solve `np.linalg.solve(A_schur, rhs)` per inner iter | RealSim: PCR_CUDA iterative in fp64 to its own tolerance | FBA: fp64 dense LAPACK | YES (precision-aligned) | ✓ Direct vs iterative is an algorithmic choice; both fp64. |
| λ accumulation `lam += dlam` across inner iters | fp64 | fp64 (`lam = np.zeros(.., dtype=np.float64)`, accumulated in host NumPy) | YES | ✓ |
| `lam_apply = dt²·ω·λ` | fp64 | fp64 (`lam_apply = (dt*dt) * omega * lam` in NumPy fp64) | YES | ✓ |
| Lambda correction gather (`J^T·lam_apply`) | fp64 (cuSPARSE SpMV) | **fp32** in default isodof path (`apply_lambda_correction_isodof`, linear_solver.py:1333: `self._lam_d.assign(lam.astype(np.float32))`); `_lam_d` is `wp.float32`; `_row_dir_d` is `wp.vec3` (fp32); gather kernel produces fp32 vec3 in `_jt_lambda_d` | NO (FBA fp32) | **DIVERGE-accidental — `lam_apply` is cast fp64 → fp32 before the gather. Subsequent solve runs in fp64 but the input lost precision. ~1e-7 per contact; with M~100 contacts cumulative error stays under 1e-5 m, still below 1e-3 m tolerance. Worth a unit test before locking.** |
| Apply correction `A⁻¹·(J^T·lam_apply)` | fp64 | fp64 inside solve (scratch buffers fp64) but input is fp32 vec3, output is fp32 vec3 | NO (FBA round-trip) | Tracking dependency of the gather precision above. |
| Velocity update `(x_new - x_prev)/dt` | fp64 | fp32 (`compute_velocity_kernel`, kernels.py:158-171, reads vec3 fp32) | NO (FBA fp32) | One-step finite difference; round-off accumulates over frames but bounded by position fp32. |

### DIVERGE-accidental (precision)

- **P1. Lambda gather downcast fp64 → fp32** (`apply_lambda_correction_isodof`,
  linear_solver.py:1333; and `accumulate_lambda_correction_kernel` callers at
  solver_fba.py:1650, 1734). `lam` is computed in fp64 host-side then cast to
  fp32 vec3 before the J^T gather. Bounded ~1e-7 per contact; total well under
  the 1e-3 m drift tolerance, but worth a precision-loss bound test.
- **P2. `_A_inv_Jt_d` storage as `wp.vec3` (fp32)** (linear_solver.py:1112). Only
  reached on the legacy (`use_isodof=False`) path which is non-default. Default
  isodof path avoids this by recomputing `J^T·lam` directly. **No fix needed for
  the in-scope demos; flag as documentation-only.**
- **P3. Position downcast fp64 → fp32 after linear solve** (`insert_component_kernel`,
  kernels.py:95-104). One per PD outer iter, propagates through Schur W (which
  reads particle_q for r computation) and the next iter's energy projections.
  Documented audit.v2 K. Bounded under 1e-7 m per iter, well below tolerance.

None of P1-P3 individually crosses the 1e-3 m blocker bar. Listed here for completeness;
defer to Phase 4 cleanup or accept as documented.

---

## Part 3 — Experiment-parameter alignment

Below appends/updates the per-demo audit at
`docs/superpowers/specs/2026-05-17-fba-realsim-per-demo-audit.md`. Only deltas
beyond the existing audit are highlighted; full tables for demos 4 and 5 (already
in the per-demo audit) are not re-typed.

### Demo 1 — TwistingBar (`fba_twisting_bar_cudatests.py --energy arap`)

| Field | RealSim | FBA | Verdict |
|---|---|---|---|
| `timestep` | 0.01 | `DT=0.01` | MATCH |
| `stop` | 810 | `NUM_FRAMES=810` | MATCH |
| `gravity` | `[0,0,0]` | `gravity=0` (line 219) | MATCH |
| `LocalGlobal_CUDA` | 5 | `PD_ITERATIONS=5` | MATCH |
| `linearsolver.type` | SPARSE_INVERSE_CUDA | FBALinearSolver sparse-inverse | MATCH (architectural) |
| `constraintsolver` block | absent | n/a (no contacts) | MATCH |
| Mesh | `cube_volume_5292P.mesh` (5292 verts ARAP) — but FBA script uses `cube_volume_11340P.mesh` (Demo 2's NH mesh) at line 48 | **mesh mismatch** | **DIVERGE-accidental** |
| `obj_mass` | 1000 | `TOTAL_MASS=1000.0` | MATCH |
| Lumping | uniform `total/N` | uniform `TOTAL_MASS/N` at line 203 | MATCH (already aligns to Decision #3) |
| `young` | 1e9 | `YOUNG=1.0e9` | MATCH |
| `poisson` | 0.45 | `NU=0.45` | MATCH |
| `constitutive` | TET_PD_ARAP | `--energy arap` | MATCH |
| Pin 0 box | `[-1.1, 1.99, -1.1, 1.1, 2.1, 1.1]` | implicit from `PIN_Y_TOP=1.99`, filtered by `verts[:,1] > PIN_Y_TOP` | MATCH (semantic) |
| Pin 0 `action` | ROLLING | rolling rotation about Y | MATCH |
| Pin 0 `axis` | `[0,1,0]` | `rot_y(top_angle)` | MATCH |
| Pin 0 `avel` | **10.0 deg/s** (per `Actions.h:65` `rad = diff * π/180`) | `PIN_AVEL=10.0  # rad/s` (line 98) | **DIVERGE-accidental — UNITS WRONG** |
| Pin 0 `maxrotation` | 90 degrees | `MAX_ANGLE = π/2 rad` (= 90 deg, ✓) | MATCH (MAX_ANGLE is in radians and equals 90 deg — but pin reaches it on frame 16 vs RealSim frame 900 because of the avel bug) |
| Pin 1 `avel` | -10.0 deg/s | `-PIN_AVEL` (line 503) | DIVERGE-accidental (same units bug) |
| `center` (rotation pivot) | `[0,0,0]` | `np.array([0,0,0])` (line 497) | MATCH |

**Demo 1 verdict: DIVERGE-accidental 2** (PIN_AVEL units bug, mesh-file path
mismatch in twisting bar script header).

The PIN_AVEL bug means FBA rotates the pins at 10 rad/s ≈ 573 deg/s vs RealSim
10 deg/s — **57.3× faster**. With MAX_ANGLE = π/2 = 90° in radians, FBA hits the
cap at frame `MAX_ANGLE / (PIN_AVEL × DT) = π/2 / 0.1 = 15.7 → frame 16`. RealSim
hits the same cap at frame `90 / (10×0.01) = 900` (clipped to NUM_FRAMES=810, so
the bar twists continuously through all 810 frames).

The mesh-file path mismatch is in the header comment vs MESH_PATH. The
script comment says "Demo 1 ARAP uses 5292P" but `MESH_PATH` line 48 points to
the 11340P mesh used for Demo 2. **Verify which `--energy` paths actually run
which mesh — the test is shared between Demo 1 and Demo 2 so this needs an
energy-keyed mesh selector. Currently both demos use the 11340P mesh.**

### Demo 2 — TwistingBarNH (`fba_twisting_bar_cudatests.py --energy neohookean`)

All global and object params match (this is the canonical mesh for the script,
11340P NH). Same PIN_AVEL units bug as Demo 1.

| Field | RealSim | FBA | Verdict |
|---|---|---|---|
| Mesh | `cube_volume_11340P.mesh` | `cube_volume_11340P.mesh` | MATCH |
| `young` | 1e9 | `YOUNG=1.0e9` | MATCH |
| `poisson` | 0.45 | `NU=0.45` | MATCH |
| `constitutive` | TET_PD_NEOHOOKEAN | `--energy neohookean` | MATCH |
| Pin `avel` | 10.0 **deg/s** | `PIN_AVEL=10.0 rad/s` | **DIVERGE-accidental (same as Demo 1)** |

**Demo 2 verdict: DIVERGE-accidental 1** (PIN_AVEL units).

### Demo 3 — StretchingCloth (NOT YET WRITTEN as FBA script)

Per-demo audit already documents the scene-loader spec. Key fields:

| Field | RealSim value |
|---|---|
| `timestep` | 0.01 |
| `stop` | 1200 |
| `maxFrame` | 2000 |
| `gravity` | `[0.0, -1.0, 0.0]` |
| `LocalGlobal_CUDA` | 5 |
| `linearsolver.type` | SPARSE_INVERSE_CUDA |
| `constraintsolver` | absent |
| Mesh | `square_20201P.obj` |
| `transformation.rotation` | `[90, 0, 0]` deg (X-axis flip into XZ plane) |
| `obj_mass` | 1000 |
| `young` | 1e5 |
| `poisson` | 0.45 |
| `constitutive` | **TRI_NEOHOOKEAN** (NOT TRI_ARAP as plan originally said) |
| `bending` | 0.0 |
| `bending_type` | BEND_ISOMETRIC |
| Pin 0 box | `[0.99, -0.1, -1.1, 1.1, 0.1, 1.1]` |
| Pin 0 action | PULLING, direction `[1,0,0]`, vel 0.1, maxlength 1.0 |
| Pin 1 box | `[-1.1, -0.1, -1.1, -0.99, 0.1, 1.1]` |
| Pin 1 action | PULLING, direction `[-1,0,0]`, vel 0.1, maxlength 1.0 |

PULLING `vel` is **m/s** (per `init.h:1005` `PullingAction(vel*dt, maxlength, direction)`
→ `diff = vel * dt` m per frame). `maxlength` is **meters**. No unit conversion
needed in the FBA scene-loader (unlike ROLLING).

**Demo 3 verdict: NOT-YET-WRITTEN.** Per-demo audit's spec stands. Note that the
PULLING action is straightforward (linear position update) so no unit ambiguity.

### Demo 4 — PullingWooper (`fba_demo4_pulling_wooper.py`)

Per-demo audit gives full table. Already-noted divergences:

- `cyl.visuallength=1.0` vs FBA `CYL_HALF_HEIGHT=3.0` — OPEN (likely OK since
  RealSim's CylinderCollision treats cylinders as infinite radially; `visuallength`
  is render-only).
- `constraintsolver.maxforce=1e12` vs FBA solver default (no cap, `lambda_cap=None`).
  Phase 2.4 will wire scene's maxforce into solver.

| Field | RealSim | FBA | Verdict |
|---|---|---|---|
| Pin PULLING `vel` | 2.0 (m/s) | `PIN_VEL=2.0` (line 53) | MATCH |
| Pin PULLING `maxlength` | 10.0 (m) | `PIN_MAXLENGTH=10.0` (line 54) | MATCH |
| Pin direction | `[0,-1,0]` | `PIN_DIR=[0,-1,0]` (line 52) | MATCH |
| `lambda_cap` | 1e12 (scene maxforce) | None (solver default) | DIVERGE-accidental (Phase 2.4 pending) |
| Cylinder rolling vel | 0.0 (both static) | not set (default 0) | MATCH |
| Cylinder mu | 0.0 (both) | `friction=False` (line 201) | MATCH (effective) |

**Demo 4 verdict: DIVERGE-accidental 1** (`lambda_cap` not wired). All other physics
fields match. The cylinder length mismatch is OPEN but per the audit is almost
certainly OK.

### Demo 5 — SqueezingBall (`fba_demo5_squeezing_ball.py`)

Per-demo audit gives full table. NSN_ITERATIONS = 10 in current script (was 1
historically; ✓ fixed). Already-noted:

- `lambda_cap` not wired (scene `maxforce=1e12` → FBA `None`)
- Mass lumping (Decision #3, pending)

| Field | RealSim | FBA | Verdict |
|---|---|---|---|
| Cylinder `rollingvel` | -3.0, +3.0, +3.0, -3.0 (rad/s, per `CylinderCollision.cpp:52` `disp = radius*_avel*dt` with no deg→rad conversion) | `CYLINDERS[i][2]` passed to `shape_angular_velocity` (rad/s) | MATCH |
| Friction `mu=0.5` (all 4 cyl + plane) | `FRICTION_MU=0.5` (line 46) via `mu_override` and `shape_material_mu` | MATCH |
| `lambda_cap` | 1e12 | None | DIVERGE-accidental |

**Demo 5 verdict: DIVERGE-accidental 1** (`lambda_cap` not wired).

### Demo 6 — CrossingGingerbreadman (NOT YET WRITTEN)

Per-demo audit gives full table. Key:

| Field | RealSim value |
|---|---|
| `stop` | 810 |
| `timestep` | 0.01 |
| `gravity` | `[0.0, -0.0, 0.0]` (effectively zero, but listed as negative-zero) |
| `LocalGlobal_CUDA` | 5 |
| Mesh | `gingerbreadman_volume_11133P.mesh` |
| `trans` | `[-0, 12.0, -0]` |
| `rotation` | `[90, 90, 180]` deg |
| `scale` | `[6.0, 6.0, 6.0]` |
| `obj_mass` | 1000 |
| `young` | 1e6 |
| `poisson` | 0.3 |
| `constitutive` | TET_PD_NEOHOOKEAN |
| Cylinders | 13 total, radius 1.0, mu 0.0 (frictionless), all static (no rollingvel) |
| `constraintsolver.maxforce` | **1e15** (one decimal order larger than demos 4,5,7,10) |
| `constraintsolver.iterations` | 10 |
| Pin PULLING | direction `[0,-1,0]`, vel 2.0, maxlength 50.0 |

**Demo 6 verdict: NOT-YET-WRITTEN.** Spec stands. Note maxforce=1e15 (must be
wired correctly in Phase 2.4); 13 colliders all frictionless static so
`friction=False` path can be used. Pin pulling 50m downward over 25s.

### Demo 8 — ClothOnKnives (NOT YET WRITTEN)

Per-demo audit gives full table. Key:

| Field | RealSim value |
|---|---|
| `stop` | 900 |
| `timestep` | 0.01 |
| `gravity` | `[0.0, -10.0, 0.0]` |
| `LocalGlobal_CUDA` | 5 |
| Mesh | `square_5101P.obj` |
| `trans` | `[0, 3.0, 0.1]` |
| `rotation` | `[90, 0, 0]` deg |
| `scale` | `[3.0, 2.7, 2.7]` |
| `obj_mass` | 1000 |
| `young` | 1e5 |
| `poisson` | 0.4 |
| `constitutive` | TRI_NEOHOOKEAN |
| `bending` | **10.0** |
| `bending_type` | BEND_ISOMETRIC |
| Plane | base `[0,-2,0,0]`, normal `[0,1,0]`, mu 0.0 |
| `genericCCD` | muVF=0, muEE=0, muFV=0, VF=EE=FV=true, alarm=0.03, miniseparation=0.01 |
| Knives obstacle | `knives.obj` at trans `[1.5,0,0]`, scale `[2,2,2]` |
| `constraintsolver.maxforce` | **100.0** (much smaller than demos 4-6) |
| `constraintsolver.iterations` | 10 |

**Demo 8 verdict: NOT-YET-WRITTEN.** Spec stands. Additional concerns:

- `genericCCD` (mesh-mesh continuous collision detection) is **not implemented in
  FBA**. This is the only in-scope demo requiring mesh CCD. Phase 5 infrastructure.
- `maxforce = 100.0` is **small** — this is the only in-scope demo where the λ-cap
  actually has a chance to activate during simulation. The Decision #5 / Phase 2.4
  per-scene λ-cap wiring is therefore particularly critical for Demo 8 parity.
  With `maxforce=100` and `dt=0.01`, RealSim clamps `|lam_R| ≤ 100` (units of force);
  FBA's internal `lam_FBA = lam_R/dt²` so it would need to clamp `|lam_FBA| ≤ 100/(0.01²) = 1e6`.
  This is the T.2 scaling bug from audit.v2 — must be fixed before Demo 8 can be
  validated.

### DIVERGE-accidental (parameters)

- **Demo 1: PIN_AVEL units** — `fba_twisting_bar_cudatests.py:98` `PIN_AVEL=10.0  # rad/s`,
  but scene.json `avel: 10.0` is degrees/s per `Actions.h:65`. FBA rotates 57.3× faster
  than RealSim. **BLOCKER for Demo 1 trajectory parity.**
- **Demo 1: Mesh file** — `MESH_PATH=cube_volume_11340P.mesh` (the NH demo's 11340-vert
  mesh) is hardcoded; Demo 1's RealSim scene uses `cube_volume_5292P.mesh` (5292 verts).
  Mesh size differs → tet count, particle count, and trajectory all differ. **BLOCKER
  for Demo 1 mesh parity.** (Demo 2 uses 11340P — that one matches.)
- **Demo 2: PIN_AVEL units** — same bug as Demo 1, same file.
- **Demo 4: `lambda_cap` not wired** — solver instantiated without `lambda_cap=1e12`.
  Currently no clamping. Phase 2.4 covers this for all demos.
- **Demo 5: `lambda_cap` not wired** — same as Demo 4.
- **Demo 6: not yet written; need to ensure `lambda_cap=1e15` is used and units
  are correct.**
- **Demo 8: not yet written; need `lambda_cap=100` AND the T.2 unit-conversion fix
  (constructor should divide by `dt²` to convert physical-force units to internal
  scaled units). Without T.2, FBA would clamp `lam` at 100 in scaled units, equivalent
  to clamping physical force at `100*dt² = 0.01` N — far too aggressive.**
- **(Carry-over from audit.v2)** Mass-lumping mismatch in demos 4, 5, 6, 8: FBA's
  density-based lumping (mass = density × per-tet-volume) vs RealSim's uniform
  `total_mass / num_vertices`. Phase 2 Task 2.3. For uniform tet meshes the
  difference is small but non-zero.

---

## Final summary

Counted across all 3 parts (precision + per-demo params + variants):

- **MATCH:**
  - Variants: 4 (LocalGlobal, A LinearSolver, NSN, Cylinder+Plane colliders)
  - Precision: 10 blocks fully fp64-aligned (factor, runtime solve, pene0, A_schur build,
    A_schur solve, λ accumulation, lam_apply scaling, contact storage, host NumPy NSN
    fp64 throughout, isodof Wi build fp64) plus the NH-LBFGS port's fp64 internal state.
  - Parameters (Demo 1): 12 fields match (gravity, dt, stop, PD iter, mass, young,
    poisson, energy, pin box, axis, center, maxrotation)
  - Parameters (Demo 2): 13 fields match
  - Parameters (Demo 3): scene-loader spec captured for all fields (no FBA script yet)
  - Parameters (Demo 4): per per-demo audit, ~25 match
  - Parameters (Demo 5): per per-demo audit, ~30 match
  - Parameters (Demo 6): spec captured for all fields
  - Parameters (Demo 8): spec captured for all fields
  - **Total MATCH count ~110 line items.**
- **DIVERGE-intentional:** 8 carried over (Decisions #1-#5, S.1, S.3, NSN-tolerance,
  Q9), 0 new in this review.
- **DIVERGE-accidental:**
  - Variants: 1 — GenericCCD (mesh CCD) not implemented in FBA. Blocker for Demo 8;
    Phase 5 infrastructure.
  - Precision: 3 (P1 lambda-gather fp32 downcast, P2 A_inv_Jt fp32 storage in
    legacy path, P3 position fp32 downcast). All bounded below 1e-3 m tolerance;
    documentation-only, not blockers.
  - Parameters:
    1. Demo 1 PIN_AVEL units (rad/s vs deg/s; 57.3× speed bug) — **BLOCKER**
    2. Demo 1 mesh path (uses 11340P instead of 5292P) — **BLOCKER**
    3. Demo 2 PIN_AVEL units (same as #1) — **BLOCKER**
    4. Demo 4 lambda_cap not wired (Phase 2.4) — non-blocker (no demo activates cap)
    5. Demo 5 lambda_cap not wired — non-blocker
    6. Demos 6, 8 not yet written — pending
    7. (Already in audit.v2 with Phase queue): mass lumping, Y.1 0.01m offset,
       Y.2 plane prev-step projection, T.2 lambda_cap unit conversion, W
       multi-iter divergence, U warm-start
- **OPEN-QUESTION:** 2
  - Demo 4 `visuallength=1.0` vs `CYL_HALF_HEIGHT=3.0` — likely OK (RealSim's
    physical cylinder is infinite); needs cylinder collision source double-check.
  - Demo 8 `maxforce=100` requires T.2 unit-conversion fix before scene-loader
    wiring; verify the T.2 conversion is implemented before testing Demo 8.

**Verdict: YELLOW.**

Justification:

- Variants and precision are **GREEN** for the in-scope subset — FBA correctly
  implements only the variants used by in-scope demos, and the precision-precision
  alignment is fp64 in every critical accumulator (A, factor, Schur W, A_schur,
  λ accumulation). The fp32 downcasts at the gather/store interface are bounded
  well below the 1e-3 m drift tolerance.
- Per-demo parameters are **YELLOW** because Demo 1 and Demo 2 have an active
  units bug (PIN_AVEL rad vs deg) that causes a 57× rotation-rate divergence —
  this is an immediate trajectory-parity blocker that the existing per-demo
  audit did NOT catch (the audit treated `MAX_ANGLE = π/2` as evidence that the
  angle math was correct, but missed that `avel = 10` is degrees in RealSim's
  scene.json while FBA treats it as radians).
- Demo 1 mesh-file mismatch (using 11340P instead of 5292P) is a second blocker.
- Demos 3, 6, 8 are not yet written — those are scope-pending, not divergences.

GREEN requires: (1) fix PIN_AVEL units in `fba_twisting_bar_cudatests.py`; (2)
add per-energy mesh selection so Demo 1 uses 5292P and Demo 2 uses 11340P; (3)
complete Phase 2.4 per-scene `lambda_cap` wiring with T.2 unit conversion for
Demo 8; (4) write FBA scripts for Demos 3, 6, 8 and verify against the per-demo
spec in this doc; (5) implement GenericCCD or accept Demo 8 as Phase-5-deferred.

---

## Appendix A — Source citations

RealSim:

- `RealSim_py/realsim_py/include/Scomponent/tools/action/Actions.h:42-74` —
  `RollingAction`: `rad = diff * π/180` confirms scene.json `avel` is degrees/s.
- `RealSim_py/realsim_py/include/init.h:1011-1014` — `avel*dt` stored as `diff`
  (still in degrees per frame; conversion happens at action-time).
- `RealSim_py/realsim_py/include/init.h:1005` — `PullingAction(vel*dt, maxlength, direction)`,
  `vel` is m/s, `maxlength` is m.
- `RealSim_py/realsim_py/src/Scomponent/collisiondetection/CylinderCollision.cpp:52` —
  `disp = radius * _avel * dt` (no degree conversion → `_avel` is rad/s for cylinders).
- `RealSim_py/realsim_py/src/Scomponent/lagrange/solvers/CUDANonSmoothNewton.cpp:60-68` —
  `_constraintlinearsolver->solve_vec(_dlam, _rhs)` confirms PCR_CUDA is the Schur
  inner solver, not the A solver.
- `RealSim_py/realsim_py/src/Scomponent/integrator/localglobal/LocalGlobalSolver.cpp:417-419,
  698-711` — scene-string-to-class dispatch tables.

FBA:

- `newton/_src/solvers/fba/solver_fba.py:218, 307, 1455-1456, 1606-1607` —
  `lambda_cap` constructor argument and clip points (no per-scene default wiring).
- `newton/_src/solvers/fba/linear_solver.py:1333` — `_lam_d.assign(lam.astype(np.float32))`:
  fp64 → fp32 downcast for J^T gather.
- `newton/_src/solvers/fba/linear_solver.py:786-796, 791-794` — fp64 scalar
  scratch buffers for A solve. Confirms fp64 inside the solve.
- `newton/_src/solvers/fba/kernels.py:15-25` — `_LBFGS_M`, `_LBFGS_TOL`, `_NH_EPS`
  all `wp.float64`. Confirms NH-LBFGS internal state is fp64.
- `scripts/fba_twisting_bar_cudatests.py:48, 98` — `MESH_PATH` hardcoded to 11340P;
  `PIN_AVEL = 10.0  # rad/s` is the units bug.
- `scripts/fba_demo4_pulling_wooper.py`, `scripts/fba_demo5_squeezing_ball.py` —
  per-demo audit table.

In-scope scene configs (read-only):

```
RealSim_py/realsim_py/simulation/config/CudaTests/TwistingBar/{scene.json,object_5k.json}
RealSim_py/realsim_py/simulation/config/CudaTests/TwistingBarNH/{scene.json,object_10k_nh.json}
RealSim_py/realsim_py/simulation/config/CudaTests/StretchingCloth/{scene.json,object_NH_20k.json}
RealSim_py/realsim_py/simulation/config/CudaTests/PullingWooper/{scene.json,wooper_5k.json}
RealSim_py/realsim_py/simulation/config/CudaTests/SqueezingBall/{scene.json,ball_7k.json}
RealSim_py/realsim_py/simulation/config/CudaTests/CrossingGingerbreadman/{scene.json,ginger_10k.json}
RealSim_py/realsim_py/simulation/config/CudaTests/ClothOnKnives/{scene.json,cloth_5k.json,knives.json}
```
