# CudaTests Full Reproduction — Design Spec

**Date:** 2026-05-16
**Owner:** FBA solver team
**Status:** Draft, pending user approval

## 1. Goal & Acceptance Criteria

Reproduce all 10 RealSim CudaTests demos in Newton/`SolverFBA`.

- **Behavior:** qualitative consistency. Each demo's PNG / abc trajectory looks like RealSim's (same gross deformation, stable, no penetration, no energy blow-up). No requirement that per-particle positions match within ε.
- **Performance:** on the **same machine**, mean `ms/step` (5 PD outer iter, dt=0.01, matching iteration counts) must satisfy `FBA ≤ 1.0× RealSim_binary`. RealSim numbers come from the project's offline binary built with `output_abc` enabled.
- **Cadence:** demo-by-demo. Each demo gets a working PR + perf row before moving on.

### Out of Scope
- Self-collision (RealSim's CudaTests have it commented out at `init.h:549-553`)
- Frame-accurate trajectory match (separate work, only TwistingBar / StretchingCloth have it today)
- Editor / GUI parity

## 2. Demo Inventory

| Demo | Particles | Energy | Pin | Colliders | Multi-deformable | New infra |
|---|---|---|---|---|---|---|
| TwistingBar | ~125 TET | ARAP | ROLLING | – | – | ✅ shipped |
| TwistingBarNH | ~125 TET | NH | ROLLING | – | – | ✅ shipped |
| StretchingCloth | 1013 TRI | ARAP | PULLING | – | – | ✅ shipped |
| PullingWooper | 5325 TET | NH | PULLING | 2 static cyl (mu=0) | – | 5k regime validation |
| CrossingGingerbreadman | 11133 TET | NH | PULLING | 13 static cyl (mu=0), g=0 | – | many-cylinder broad-phase |
| SqueezingBall | 7129 TET | NH | – | 4 rolling cyl (mu=0.5, ω=±3) + plane | – | kinematic body motion in contact anchor |
| SharpCorner | 5101 TRI + bend 5.0 | ARAP | – | 1 static box | – | (covered by Phase 1) |
| ClothOnKnives | 5101 TRI + bend 10.0 | NH | – | static knives mesh + plane | – | `GeoType.MESH` static collider path |
| ParallelEnvTest | 25 × 1013 TRI + bend 20.0 | ARAP | – | 25 spheres (mu=0.3) | – | streamed multi-env + LiteSchur option |
| CableGrabRaptor | 2 fingers + raptor_10k | NH + cable | dynamic 7-step | torus + 2 planes | YES (finger×finger, finger×raptor) | Bergou cable, soft-soft contact, dynamic pin sequencer |

### 2.1 RealSim solver settings to mirror
- `LocalGlobal_CUDA: 5` — already the FBA default
- `constraintsolver.iterations: 10` — **FBA currently 20; lower to 10**
- `maxforce: 100` for ClothOnKnives/SharpCorner; `1E+12` elsewhere — **add per-step λ-cap to NSN**
- `function: FB` — current closed-form Coulomb projection is mathematically equivalent

### 2.2 Self-collision (rechecked)
`init.h:528-565` shows `genericCCD/DCD.detect_all=true` pairs (mesh_i × mesh_j) for i<j and (obstacle_i × mesh_j). The mesh_i × mesh_i block (549-553) is commented out. No CudaTests demo uses self-collision.

## 3. Current FBA Capability Baseline

- Energies: TRI{ARAP, Corot, NH} + TET{ARAP, Corot, NH} + isometric bending
- Pin: PD-energy embedded, `set_pin_targets()` supports PULLING/ROLLING
- Linear: Cholesky + sparse-inverse (`L⁻¹`) + BSR multi-RHS SpMV
- Contact:
  - Stage A (normal Schur), Stage B (Coulomb friction Schur), Stage C (validated against box + cylinder + plane primitives)
  - **Dense** Schur W (M² build, multi-RHS Approach B → 21 launches/build)
- Verified at: 81–169 particles, 10–43 ms/step
- Not yet verified at: 5k+ particles, MESH collider, kinematic colliders, multi-mesh contact, cable, multi-env

## 4. Architecture Decisions

### 4.1 Dense vs Lite Schur
**Default = dense.** Identical to RealSim's `NonSmoothNewton_CUDA`. Dense will not "blow up" at 5k particles if RealSim works on the same hardware — same math, same M, same machine. Performance regressions at 5k mean an implementation bug (launch overhead, missing CUDA graph, missed warm-start), not an algorithm change.

**Lite (sparse Schur) is a P6-only knob**, mirroring RealSim's `LiteNonSmoothNewton_CUDA` which is only invoked by the parallel-env config. It exploits block-diagonal Delassus when M = ∑M_env over independent envs.

### 4.2 ParallelEnv: streamed independent sub-models
**No factor sharing.** Each env can have its own (Young, bending, etc.) — factors must be independent. Path: build 25 independent `SolverFBA` instances or 25 sub-mesh ranges within one solver, dispatch on CUDA streams. The LiteSchur path additionally batches the dense per-env Schur blocks into one BSR matrix for the NSN.

### 4.3 Kinematic colliders (SqueezingBall, P7d for ROLLING fingers)
Colliders stay `body=-1` from FBA's POV. Their pose is updated externally per-frame; contact anchor velocity `v_anchor = ω × r_local` is read from a per-shape twist field and fed into the Stage B tangent residual as `r_t = dot(t, x_anchor + v_anchor·dt) − dot(t, x_unc)`.

### 4.4 MESH colliders (P5, P7b)
Static triangle mesh via `add_shape_mesh`. Newton's `create_soft_contacts` already supports `GeoType.MESH` via `wp.mesh_query_point_sign_normal` (`geometry/kernels.py:1094`). FBA only needs to verify its Stage A/B consumes the resulting `soft_contact_*` arrays correctly when shape_type == MESH (untested in our path so far).

### 4.5 Cable element (P7a)
1D Bergou-style elastic rod as a new energy term in `fba/kernels.py`:
- Stretch: per-segment ‖edge‖ = rest_len
- Bending: per-internal-node curvature → rank-1 Q matrix
- Plugs into the same Cholesky+sparse-inverse linear pipeline (extends `A` block)

### 4.6 Soft-soft contact (P7c)
Currently Stage A/B assume one side is `body=-1` static. Extend J to two-sided: J·x = [J_a | −J_b]·[x_a; x_b]. Schur block dimension doubles per contact; reuse dense W path. NSN restart per-PD-outer unchanged.

### 4.7 Dynamic pin sequencer (P7d)
RealSim's gripper config has 7 sequential `dynamic` entries (PULLING/ROLLING with `maxlength`/`maxrotation` triggering the next entry). Extend `set_pin_targets()` to consume an array of `(action, params, terminate_condition)` and auto-step.

## 5. Phased Plan

### Phase 0 — RealSim Baseline (3–5 days)
- Build `RealSim_py` binary with `output_abc` enabled (offline mode, auto-`stop`)
- Run all 10 demos, store `{demo, frames, mean_ms_per_step, vram_peak, abc_path}` to `scripts/cudatest_baselines.json`
- FBA-side: a `scripts/fba_cudatest_bench.py` driver that reproduces each scene from a parallel `cudatest_fba_configs.json`, runs N steps, dumps `(demo, ms_per_step, finite_ok, min_z, peak_mem)` matched to baseline rows

### Phase 1 — PullingWooper (1 week)
- First 5k-particle contact demo
- Validates Cholesky+sparse-inv path at 5k regime (setup ≤ 1 s acceptable, kept off the per-frame loop)
- Already-supported features: TET NH, PULLING, Stage A static cylinder
- **Perf debt fold-in (if dense Schur measures slow at 5k):** CUDA graph capture for the per-frame step; λ warm-start across PD outer iters; NSN iter 20→10; λ-cap
- Acceptance: visual match + `ms/step ≤ RealSim_binary` on same machine

### Phase 2 — SqueezingBall (1 week)
- New infra: kinematic cylinder rotation in the contact anchor (Section 4.3)
- Stage B Coulomb friction (mu=0.5) + plane mu=0.5
- Acceptance: ball gets squished between rolling cylinders, falls onto plane with friction-arrested motion

### Phase 3 — CrossingGingerbreadman (3–5 days)
- Multi-cylinder static, gravity=0, PULLING
- Stresses contact density: M can spike 5–10× the per-frame baseline
- Acceptance: gingerbreadman threads through corridor, no exploding contacts

### Phase 4 — SharpCorner (3–5 days)
- First 5k TRI cloth demo (P1 was TET). Brings up the cloth Cholesky+sparse-inv at 5k.
- Collider = `add_shape_box` (already validated by demo3 cylinder/plane infra)
- ARAP cloth_5k folds over box corner; visible two-flap symmetric drape
- No truly new infra; same perf debt items from P1 apply if cloth path is slower than TET path

### Phase 5 — ClothOnKnives (1 week)
- New infra: `GeoType.MESH` static collider validated end-to-end in FBA Stage A/B
- Verify Newton's `create_soft_contacts` MESH branch supplies correct normals/depths into our `soft_contact_*` arrays
- Acceptance: cloth_5k hangs on knives mesh with sharp ridges at knife edges

### Phase 6 — ParallelEnvTest (1–2 weeks)
- 25 independent sub-models (factors not shared — each env can have different Young/bending)
- Streamed CUDA-graph capture per env
- **LiteSchur option** introduced: pack 25 per-env Schur blocks into a block-diagonal BSR W; solve with sparse linear solver per NSN iter. Mirrors RealSim `LiteNonSmoothNewton_CUDA`.
- Acceptance: 25 cloths drape on 25 spheres simultaneously; wall ≤ 25× single-env wall (better with streams)

### Phase 7 — CableGrabRaptor (2–3 weeks)
The terminal demo. Split into 7a/b/c/d:

- **7a — Bergou cable element (1 week)**
  - Add stretch + bending in `fba/kernels.py`
  - **Unit test:** isolated single-mesh-plus-cable script. Drive cable endpoints, verify mesh deforms in the expected shape (no contact, no collider).
  - Cross-check against an analytical bent-rod static solution
- **7b — Torus collider (1–2 days)**
  - `torus.obj` is a triangulated surface mesh → reuse P5's `add_shape_mesh` path. Zero new code if P5 lands first.
- **7c — Soft-soft contact (1 week)**
  - Two-sided contact Jacobian J = [J_a | −J_b]
  - Schur block dimension 2× per contact pair; sparse-W treatment used only if M_pairs huge
  - Verify with a 2-tet-cube interpenetration micro-demo first
- **7d — Dynamic pin sequencer (2–3 days)**
  - Extend `set_pin_targets()` to consume an action queue with auto-step on `maxlength`/`maxrotation`
- **Acceptance:** two fingers (cable-driven) grasp 10k raptor, raptor deforms, torus catches it

## 6. Cross-Cutting Performance Backlog
Folded into the phases that trigger them:
- **CUDA Graph capture** of the inner per-frame solve → triggered if P1 5k measurement is launch-bound
- **λ warm-start across PD outer iters** → triggered with CUDA graph (since both edit the same restart story)
- **NSN iter 20 → 10** + **λ-cap** → done in P1
- **Stable factor caching** (skip rebuild when topology unchanged) → done in P1 if not already
- **LiteSchur (sparse-W, block-diagonal exploit)** → P6 only

## 7. Risk Register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| 5k cloth Cholesky unstable | Low | High | RealSim works at 5k on same hardware; if FBA cannot, it's a setup bug — debug, not algorithm switch |
| Bergou cable accuracy off | Med | High | 7a unit test gates merge |
| Soft-soft contact Schur ill-conditioned | Med | High | Block-diagonal preconditioner; fall back to LiteNSN sparse path locally if needed |
| ParallelEnv memory (25 factors) | Med | Med | LiteSchur block-diagonal storage already amortizes; can cap envs if needed |
| MESH collider normal sign on backfaces | Low | Med | P5 includes a unit test against synthetic knife mesh |

## 8. Effort Estimate

| Phase | Item | Effort |
|---|---|---|
| 0 | RealSim baseline | 3–5 days |
| 1 | PullingWooper + 5k validation + perf debt fold-in | 1 week |
| 2 | SqueezingBall | 1 week |
| 3 | CrossingGingerbreadman | 3–5 days |
| 4 | SharpCorner | 3–5 days |
| 5 | ClothOnKnives | 1 week |
| 6 | ParallelEnvTest + LiteSchur | 1–2 weeks |
| 7 | CableGrabRaptor (7a/b/c/d) | 2–3 weeks |
| **Total** | | **8–10 weeks** |

## 9. Deliverables

- `scripts/cudatest_baselines.json` — RealSim baselines
- `scripts/fba_cudatest_bench.py` — FBA driver, per-demo perf row
- Per-demo PR with: scene script `scripts/fba_demoN_<name>.py`, perf entry, snapshot strip PNG
- New infra commits as separate PRs:
  - `kinematic_collider_motion` (P2)
  - `mesh_collider_in_fba` (P5)
  - `cable_element` (P7a)
  - `soft_soft_contact` (P7c)
  - `dynamic_pin_sequencer` (P7d)
  - `lite_schur_parallel_env` (P6)
- Updated `docs/superpowers/fba-dev-progress.md` after each phase

## 10. Open Questions
- Where does P0 baseline live in the repo? Same `scripts/` or a sibling `scripts/realsim_baseline/`?
- For demos with cylinders, can we keep using Newton's `add_shape_cylinder` primitive everywhere, or do any RealSim demos rely on cylinder *visual length* truncation that the SDF doesn't match? (Currently no evidence required, defer until P3.)
- Per-shape friction: RealSim has per-cylinder mu; current FBA already supports `mu_per_pair_override` — re-verify in P2/P3 that the per-shape mu maps cleanly.
