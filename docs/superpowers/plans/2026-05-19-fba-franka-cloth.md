# Plan: Port `example_cloth_franka` from SolverVBD to SolverFBA

**Date:** 2026-05-19
**Status:** Drafted, not started.
**Owner:** TBD
**Estimated effort:** ~3 weeks (Phase 0 cm-scale smoke 0.5 d, Phase 1 multi-contact-per-particle 2-3 d, Phase 2 dynamic shape velocity 1-2 d, Phase 3 cloth self-contact 1.5-2 w, Phase 4 demo integration 3-5 d).
**Target scenario:** `newton/examples/cloth/example_cloth_franka.py` (T-shirt folding with a Franka Emika Panda gripper) — replace `SolverVBD` with `SolverFBA(nsn_schur_mode="lite")`.

---

## Why

The Franka cloth demo is Newton's reference for **dynamic-collider cloth manipulation**: a Featherstone-controlled robot grips, lifts, drags and folds a T-shirt sitting on a table.  It exercises four things SolverFBA does **not** support today:

1. **Multiple contacts per cloth particle** — a cloth vertex can be touched simultaneously by both Franka fingers and the table.  The `fb_newton_lite_coulomb_kernel` introduced in the multi-env plan assumes ≤ 1 contact per particle (true for cloth-on-sphere, false everywhere else).
2. **Dynamic rigid-body kinematic velocity** — the gripper translates and rotates every step.  Without the contact-anchor velocity tracking that motion, friction with the gripper produces near-zero shear and the cloth slides through the fingers.
3. **Cloth self-contact** — the fold-up sequence stacks shirt layers on top of each other.  No self-contact → the layers pass through each other.
4. **cm-scale physics** — VBD's demo runs in centimeters for numerical stability.  FBA has been tuned at meter scale only; the PD prefactor coefficients change by 10⁴ at cm scale.

The high-level architectural payoff: every cloth scene currently using VBD becomes addressable by FBA, opening parallel-env training (the multi-env plan from `2026-05-18-fba-multienv-cloth.md` already plumbed) for arbitrary cloth-rigid scenarios — not just cloth-on-static-sphere.

**User-locked invariant (from review of this plan):** LiteNSN **stays sparse** at every step.  Even with multiple contacts per particle (Phase 1) and with cloth self-contact (Phase 3), we never fall back to a dense ``W``.  The dense projection is only kept as the regression baseline `build_schur_lite_dense` for unit tests; production paths use either the per-particle fused kernel (block-diagonal sparsity) or the BSR + sparse-PCR pipeline (graph-induced sparsity).

## What changes

Four orthogonal pieces of work, joined by a final demo-integration phase:

1. **Generalised block-diagonal LiteNSN kernel** — replace the per-contact fused kernel with a per-*particle* fused kernel that handles a variable number of contacts ``k_p`` at each particle.  Sparsity stays block-diagonal-by-particle (the strict invariant: no self-contact); blocks grow from 3×3 to 3·k_p × 3·k_p.  Still sparse, still O(M) memory, still no dense W.

2. **Dynamic shape kinematic velocity** — read each contact's parent body 6-DoF velocity (`state.body_qd[shape_body[s]]`) and project to the tangent plane at the anchor point.  Replaces the current `shape_angular_velocity` hard-coded scalar-Z-rotation field for use cases that need full 6-DoF (Franka gripper) instead of just cylinder rolling (SqueezingBall).

3. **Cloth self-contact** — add particle-particle broadphase (BVH on cloth vertices), generate FB rows for particle-particle contacts, and consume them in the NSN inner.  Self-contact breaks the "block-diagonal-by-particle" sparsity (since a single row now touches two particles), so the constraint solve switches from the fused-per-particle kernel to the **BSR + sparse-PCR pipeline already implemented in `2026-05-18-fba-multienv-cloth.md` Phase 2.1/2.2 and currently unused on the hot path**.  This is the largest single piece of work.

4. **cm-scale physics tuning** — re-derive ARAP `tri_ke`, edge `edge_ke`, and `pin_stiffness` defaults at cm scale.  The PD prefactor's mass diagonal is `m/dt²` (scale-invariant) but the stretching contribution is `tri_ke · area`, where ``area`` grows 10⁴× when distances are in cm.

5. **Demo integration** — `example_cloth_franka.py` (or a sibling `example_cloth_franka_fba.py`) drops `SolverVBD` for `SolverFBA(nsn_schur_mode="lite", stretching_model="arap")`, plumbs the contact-pipeline + self-contact through FBA's `update_contacts`, and produces a rendered video matching the VBD reference at p95 < 5 cm trajectory drift in non-folding key-pose intervals.

The PD outer loop, A_FBA matrix assembly, pin handling, and rigid-cloth pipeline are **unchanged**.

## Pre-conditions

- `2026-05-18-fba-multienv-cloth.md` deliverables landed (commit `1843ca82`+):
  - `SolverFBA(nsn_schur_mode="lite")` exists with the per-contact fused block-diag kernel.
  - `FBALinearSolver.build_schur_lite` (BSR via `bsr_mm`) and `NSNPCRSolver.solve_sparse` exist but are not on the hot path — Phase 3 below is what finally puts them on the hot path.
- `pipeline.collide(state, contacts)` already supports articulated bodies — it reads `shape_body[s]` and `body_q[shape_body[s]]` to compute each shape's world transform.  No new contact-pipeline work is needed in this plan.

## Architectural decisions

### LOCKED: lite path stays sparse in all phases — never dense W

LiteNSN's memory and perf story only holds when the Schur ``W`` is stored sparsely.  Falling back to dense at multi-contact-per-particle or at self-contact would (a) defeat the multi-env perf gain, (b) OOM at large contact counts on the Franka demo, and (c) split the codebase into two contradictory "lite" semantics.  Multi-contact/particle keeps the strict block-diagonal sparsity (just with bigger blocks per particle); self-contact widens the sparsity to graph-induced but remains << dense.

### LOCKED: per-particle fused kernel for the strictly-block-diagonal case (Phase 1)

When the only contacts are particle-vs-rigid (no self-contact), every cloth particle's contacts decouple from every other particle's.  One CUDA thread per particle reads its ``k_p`` rows from the row CSR (already built by `setup_lite_row_meta`), builds the local ``3k_p × 3k_p`` block, runs the FB-Newton step, applies the Coulomb cone clamp, and writes ``λ`` / ``ω`` / ``lam_apply``.  No BSR matrix, no SpGEMM, no PCR, no dense buffer.  This is the strict generalisation of `fb_newton_lite_coulomb_kernel` from a fixed 3×3 block to a variable 3·k_p × 3·k_p block.

### LOCKED: BSR + sparse-PCR for the self-contact case (Phase 3)

Self-contact contributes Jacobian rows that touch **two** particles each.  ``W_lite[i, j] != 0`` iff rows ``i`` and ``j`` share at least one particle, which is the contact-graph adjacency — sparse, but no longer block-diagonal.  The fused-per-particle kernel cannot represent this without losing parallelism.  Phase 3 routes contacts through `build_schur_lite` (sparse SpGEMM via `bsr_mm`, already exists) and solves with `NSNPCRSolver.solve_sparse` (already exists, equivalent to dense PCR within fp64 round-off).  This is what the multi-env plan's Phase 2.1/2.2 was secretly preparing for — those code paths were validated end-to-end against dense but never had a consumer; the Franka demo is the consumer.

### LOCKED: cap ``k_p`` at 8 contacts per particle in the fused kernel

Cloth-rigid contact rarely produces more than ~4 contacts at one particle even in worst-case fold geometries.  Padding to a fixed maximum lets the kernel unroll register-resident arrays and avoids dynamic-stack allocations.  If a particle exceeds 8 contacts we fall back to the BSR + sparse-PCR path (the same one Phase 3 uses) for that step — automatic, no user knob.

### LOCKED: dynamic shape velocity is read from `body_qd`, not a per-shape ω

The existing `shape_angular_velocity` field is a per-shape scalar ω about local-Z.  It is kept for backwards compatibility with SqueezingBall (cylinder rolling) but **superseded** at runtime when `shape_body[s] >= 0` and that body has a non-zero `body_qd`.  Mixing both is well-defined: per-shape ω contributes only when the shape is parented to ``body=-1`` (static-frame kinematic rotation, e.g. a rotating sphere fixture); body-driven velocity contributes only when ``shape_body[s] >= 0``.

### LOCKED: cloth self-contact uses RealSim's spring-mass FB form, not LCP

RealSim's self-contact rows use the same Fischer-Burmeister unilateral + frictional form as particle-rigid contact, just with two particles per row.  We port that line-for-line.  Reference: `RealSim/src/Simulation/Contact/SelfCollisionResolver.cpp` (TBD line range, to be inspected during Phase 3.1).

### LOCKED: scale (cm vs m) is the example's choice, not FBA's

FBA stays unitless — the user sets `gravity`, `tri_ke`, `pin_stiffness` consistent with their chosen units.  Phase 4 just re-tunes the Franka demo's coefficients for cm scale; no change to the solver.

## Phase 0 — cm-scale single-env smoke (~0.5 day)

**Goal:** Confirm `SolverFBA` runs at cm scale without precision loss.  Isolates the units question from every other piece of new code.

### Step 0.1 — port `example_hanging_cloth_fba` to cm

Scale every coordinate, mass, and stiffness by the appropriate power of 100 (positions ×100, velocities ×100, gravity ×100, `tri_ke` ×10⁻², `edge_ke` ×10⁰).  Keep dt the same.  Run 100 frames headless and verify the drape geometry is identical to the m-scale reference modulo the uniform 100× scale factor.

### Step 0.2 — acceptance gate

The cm-scale run's particle positions, divided by 100, must match the m-scale run to within **1 mm p95** (the same FP-tolerance bar as Phase 0 of the multi-env plan).  If not, the issue is precision in the PD prefactor (look at `_csc_to_bsr_1x1` Cholesky in fp32→fp64 transition) or in the lite mass diagonal (`dt² · inv_mass`, which is in 1/m² · m² · cm⁴ = cm² when units shift — needs to stay invariant under unit choice, which it does as long as gravity scales consistently).

## Phase 1 — multi-contact-per-particle in lite (2-3 days)

**Goal:** `SolverFBA(nsn_schur_mode="lite")` correctly handles particles touched by 2-8 contacts, with **strict block-diagonal sparsity preserved** (still O(M) memory, no dense W).

### Step 1.1 — generalised per-particle fused kernel

Replace `fb_newton_lite_coulomb_kernel` (per-contact, 3×3 fixed) with `fb_newton_lite_coulomb_per_particle_kernel` (per-particle, 3·k_p × 3·k_p variable).  One thread per **isodof particle** (unique contacted particle).  Inputs:

```
isodofs               (K,)  int32      # unique particle indices touched
isodof_row_offsets    (K+1,) int32     # CSR: row range per isodof
isodof_row_indices    (M_total,) int32 # rows belonging to each isodof (in contact-row order)
contact_normal/t1/t2  (M,)  vec3       # unchanged
contact_alpha         (M,)  float32    # unchanged
contact_mu            (M,)  float64    # unchanged
contact_particle      (M,)  int32      # unchanged
r, pene0              (3M,) float64    # unchanged
inv_mass              (N,)  float64    # unchanged
dt, cap_internal, use_cap, n_fb_iters  # unchanged
```

The kernel:

1. Loads the particle's ``k_p`` rows via the CSR.
2. Builds the local ``3k_p × 3k_p`` ``W`` block:
   ```
   W[r1, r2] = α[r1] · α[r2] · dot(dir[r1], dir[r2]) · dt² · inv_m[p]
   ```
   for ``r1, r2`` in the particle's row set.  Storage: register-resident `wp.mat` of size up to 24×24 (k_p ≤ 8) — fits in register file with some spill to L1.
3. Loads λ_local, ω_local, r_local, pene0_local for the particle.
4. Runs the FB-Newton iteration: penetration, FB row evaluation, A_schur build, Cholesky solve (3k_p × 3k_p), λ update.
5. Applies Coulomb cone clamp **per contact** (each contact's 3 rows are still a normal+2 tangents triple).
6. Writes back λ, ω, lam_apply.

The CSR is **already built** by `FBALinearSolver.setup_lite_row_meta` (multi-env plan Phase 2.3) — Phase 1 just consumes it correctly instead of assuming ``k_p == 1``.

### Step 1.2 — Cholesky for variable small-block size

Write a `wp.func cholesky_solve_pd_small(L, b, n)` template that handles n ∈ {3, 6, 9, 12, 15, 18, 21, 24}.  For each n, generate an unrolled in-place Cholesky factorise + forward/back substitute (fp64).  ~30 lines per n via a small Python codegen step (or by hand — 8 specialisations is tractable).

### Step 1.3 — fallback when ``k_p > 8``

When `isodof_row_offsets[p+1] - isodof_row_offsets[p] > 8`, the kernel emits a sentinel (skips the update; warm-start λ unchanged) and the host code re-runs the affected step through the BSR + sparse-PCR path (the same one Phase 3 uses).  In practice ``k_p > 8`` never happens in the Franka demo; this is a safety net.

### Step 1.4 — validation against full NSN

Build a 2-obstacle cloth scene: cloth-on-sphere PLUS a static box positioned so some cloth particles touch both at frame 50.  Run with `nsn_schur_mode="lite"` and `"full"`.  Validate cloth-trajectory drift at frame 200 under the same tolerance as the multi-env plan (p95 < 5 cm, max < 15 cm; gate inherited from the FBA verification tiers memory).

The 2-obstacle scene is also a **regression** for the existing 1-contact-per-particle path (the single sphere case): the new per-particle kernel must reproduce the multi-env plan's cloth-on-sphere Phase 2.4 result bit-for-bit when ``k_p == 1`` everywhere.

## Phase 2 — dynamic shape kinematic velocity (1-2 days)

**Goal:** When a cloth contact's parent shape is attached to a moving rigid body (Franka gripper finger), the contact's tangent anchor velocity tracks the body's 6-DoF motion at the contact point.  This is what lets friction actually *grip*.

### Step 2.1 — body-velocity-driven anchor velocity

In `compute_v_anchor_kernel` (or its successor), for each contact ``c``:

```
parent = shape_body[contact_shape[c]]
if parent < 0:
    # Static-frame kinematic (existing path): use per-shape ω about local-Z.
    v_anchor[c] = -ω_shape · axis_local_z × (anchor_world - shape_centre_world)
else:
    # Body-driven (new path):
    body_xform = body_q[parent]
    v_body = wp.spatial_top(body_qd[parent])      # linear vel at body COM, world frame
    ω_body = wp.spatial_bottom(body_qd[parent])
    r_world = anchor_world - transform_get_translation(body_xform · body_com_local)
    v_anchor[c] = v_body + cross(ω_body, r_world)
```

`anchor_world` is already cached by the existing contact-resolution machinery (`contact_offset_d` provides the projection).

### Step 2.2 — `contact_shape` device array

Currently FBA caches `contact_particle_d` but not `contact_shape_d`.  Add the latter — `contacts.soft_contact_shape0` (the rigid-side shape) is populated by `CollisionPipeline`.  Wire it through `update_contacts`.

### Step 2.3 — validation against SqueezingBall and a new "rolling-box-on-cloth" smoke

The squeezing-ball example uses `shape_angular_velocity` exclusively (parent body = -1).  After Phase 2 it must still produce identical trajectories.  Then a new smoke scene with a **dynamically translating box** sliding across a clamped cloth: the cloth should follow the box laterally when friction is high.

## Phase 3 — cloth self-contact via BSR + sparse-PCR (1.5-2 weeks)

**Goal:** Cloth particles can touch other cloth particles.  This is what makes the Franka **folding** sequence (vs just grasping/moving) work.

### Step 3.1 — particle-particle broadphase

Port RealSim's `SelfCollisionResolver` (or use `wp.HashGrid` directly, which Newton already exposes).  Output: a list of candidate `(particle_a, particle_b, distance_lt_threshold)` pairs.  Threshold is `particle_self_contact_radius + particle_self_contact_margin` — same semantics as VBD's existing fields, so users get a drop-in upgrade.

### Step 3.2 — self-contact row generation

Convert each surviving pair to a Stage B contact row block:

```
row_normal      = (q_a - q_b).normalised()
row_tangent1    = orthogonal_to(row_normal, world_up)
row_tangent2    = row_normal × row_tangent1
row_alpha       = 1.0   # by convention, same as particle-rigid
row_particle_a  = a
row_particle_b  = b
```

The Jacobian for self-contact has **two** nonzero particle blocks per row (`+dir` at particle a, `-dir` at particle b).  This is what breaks the strict block-diagonal sparsity — and exactly what `build_schur_lite` already handles via `bsr_mm` (because `H` is built with arbitrary per-row sparsity).

### Step 3.3 — extend `setup_lite_row_meta` for two-particle rows

The current CSR (`_isodof_row_offsets_d` / `_isodof_row_indices_d`) maps each row to **one** particle.  For self-contact, each row maps to **two**.  Build a new "row → two particles" array `_row_particle2_d` (with sentinel ``-1`` for one-particle rows) so downstream kernels can handle both.

### Step 3.4 — route through `build_schur_lite` instead of the fused kernel

When the contact set contains any self-contact rows, switch dispatch from the per-particle fused kernel (Phase 1) to:

```
W_bsr = ls.build_schur_lite(M, j_indices, j_normals, j_alpha, j_tangent1, j_tangent2,
                             j_indices_b=...)   # NEW: optional second-particle list
A_schur_bsr = build_a_schur_lite_bsr(W_bsr, omega, compliance)   # NEW: in-place Hadamard + diag-add
dlam = pcr_solve_sparse(A_schur_bsr, rhs, ...)                    # existing
```

`build_a_schur_lite_bsr` is the only new kernel: it iterates over W's nonzero blocks, scales by ``ω[row]·ω[col]``, and adds ``c[row]`` on the diagonal.  Same sparsity as W (Hadamard preserves zeros).  ~40 lines.

The PCR sparse solve (`NSNPCRSolver.solve_sparse`) is **already implemented and validated** (multi-env plan Phase 2.2).  This is what was meant by "currently unused on the hot path": Phase 3 finally consumes it.

### Step 3.5 — `compute_nsn_rhs` sparse variant

The dense `compute_nsn_rhs_kernel` does an inline ``W·λ`` matvec.  The sparse equivalent: precompute ``W_lam = bsr_mv(W_bsr, lam_omega)`` once per FB-Newton iter, then a tiny per-row kernel does ``rhs[i] = (1/dt²) · (h[i] - ω[i] · (pene0[i] + W_lam[i]))``.

### Step 3.6 — validation against VBD self-contact

Run a "self-folding cloth" smoke (a single cloth pinned at two corners, allowed to fall and self-collide).  Compare Newton-FBA vs Newton-VBD trajectories at frame 200.  Tolerance: **p95 < 5 cm, max < 15 cm** at the cm scale used by the Franka demo (so 5 mm / 1.5 mm post-conversion to m).  This is **not** apples-to-apples (different self-contact formulations), but is a sanity check; the user-facing acceptance is Phase 4.

## Phase 4 — Franka demo integration (3-5 days)

### Step 4.1 — fork the example

Create `newton/examples/cloth/example_cloth_franka_fba.py` (don't mutate the VBD reference; both should coexist for regression).  Same Franka URDF, same T-shirt, same key-pose sequence, same scale (cm).

### Step 4.2 — replace solver construction

```python
self.cloth_solver = SolverFBA(
    self.model,
    iterations=self.iterations,
    nsn_iterations=1,
    friction=True,
    stretching_model="arap",
    mu_per_pair_override=...,
    pin_stiffness=...,                       # tuned for cm scale
    nsn_schur_mode="lite",
)
```

No `integrate_with_external_rigid_solver` knob — FBA reads `state.body_q` / `state.body_qd` directly from the Featherstone-driven state.

### Step 4.3 — cm-scale tuning

`tri_ke`, `edge_ke`, `pin_stiffness` re-derived from the m-scale single-env example.  Document the conversion in the example header.

### Step 4.4 — perf budget

VBD does the demo at ~30-60 fps.  FBA-lite single-env was 2.66 ms/step at m scale (cloth-on-sphere, 1013 particles).  T-shirt has ~3-5× more particles, multi-contact, and self-contact.  Realistic budget: **≤ 33 ms/step** (30 fps real-time) at single-env.  If we miss by < 2× the demo is still acceptable as a "physics-rich quality reference"; if we miss by > 3× then either (a) self-contact PCR is the bottleneck (Phase 3 tuning), or (b) we narrow the demo scope (skip the dual-fold-sequence).

### Step 4.5 — visual acceptance

Side-by-side render of VBD vs FBA at the same key-pose interval.  Subjective acceptance:

- Cloth grasps and lifts cleanly (no slipping out of fingers).
- During folds, layers stack rather than penetrate.
- No NaN, no runaway stretching.

Numerical acceptance is **not** a strict trajectory match (different solvers; the demo is a long open-loop sequence where FP drift compounds).  Instead:

- Particle bounding box stays within the test_final volume (`p_lower` / `p_upper` in the existing example).
- Particle velocity max stays under the existing 200 cm/s limit.

## Out of scope (deferred to follow-up plans)

- **Two-way coupling (cloth-on-robot reaction force)**: the Franka demo can be run with the robot's controller assuming the cloth is light enough to ignore (which is approximately true at the demo's parameters).  Full two-way coupling means feeding ``J^T λ_contact`` back into Featherstone's spatial vector accumulator — non-trivial, deferred.
- **VBD parity tests** (bit-level): different solvers, different timestep semantics.  Visual acceptance only at Phase 4.
- **Friction beyond Coulomb**: viscous tangent damping, stiction breakaway thresholds — VBD has stiction, FBA does not.  Out of scope; the demo plays at sliding-friction-only.
- **Cloth-cloth self-contact in the BSR path's CUDA-graph capture**: PCR sparse is currently eager-only (no graph capture) due to `bsr_mv` shape changes.  Re-enabling capture is a perf-tuning follow-up.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Phase 1's per-particle fused kernel produces register spills on RTX 5090 at k_p = 8 (block size 24×24 fp64 = 576 entries = 2.3 KB per thread) | Kernel becomes slower than the dense Cholesky | Step 1.2's per-n unrolled specialisations let the compiler keep small-k variants fully in registers; large-k variants accept some L1 spill.  Profile at k = 1, 3, 5, 8 separately. |
| Phase 3's `bsr_mm` reshape on every frame defeats CUDA graph capture, costing ~ms/step in launch overhead | Self-contact perf bad | Use `wps.bsr_mm(..., reuse_topology=True, max_new_nnz=...)` once the topology stabilises (settled cloth contact set is fairly stable across PD outer iters).  Multi-env plan flagged this same trade-off. |
| Cloth self-contact rows generate ill-conditioned A_schur at near-zero penetration (the FB function compliance term goes to zero → A_schur becomes nearly singular) | PCR fails to converge in 100 iters → solver returns inf | Match RealSim's compliance floor of `1e-12` and re-validate.  Worst case: cap PCR iters and let one frame's λ get stale; FB-Newton recovers next step. |
| cm-scale ``A_FBA`` factorisation has different fill-in than m-scale due to AMD reordering of cm-sized entries | PD prefactor build OOMs at cm scale on the T-shirt mesh | Phase 0 Step 0.2 is the early-warning signal; if it fires, suppress AMD and use natural ordering instead (one-line scipy override). |
| Multi-contact-per-particle hits >8 contacts and the BSR fallback path produces a different result than the fused kernel (different floating-point reduction order) | Trajectory drift between identical configs running on different envs | Same root cause as the multi-env plan's Phase 3.3 strict gate failure (FP non-associativity).  Document; not blocker. |
| Featherstone-driven `body_qd` at sub-cm/s precision (the slow EE motion in the key-pose sequence) underflows when projected to contact-anchor velocity | Friction "drag" intermittent on slow gripper motion | Use fp64 throughout the v_anchor computation (Newton's `body_qd` is already fp32; promote on read). |

## Done definition

This plan is done when:

1. ✅ Phase 0 cm-scale port of `example_hanging_cloth_fba` matches the m-scale baseline within 1 mm p95 (after the uniform 100× scale-factor undo).
2. ✅ Phase 1 multi-contact/particle 2-obstacle smoke matches the full-NSN reference at p95 < 5 cm / max < 15 cm @ frame 200, **without using dense W** (verified by allocator instrumentation: `_W_device_d` stays unallocated in lite mode).
3. ✅ Phase 2 squeezing-ball regression unchanged + rolling-box-on-cloth smoke shows the cloth following the box laterally (friction-driven motion).
4. ✅ Phase 3 self-folding-cloth smoke produces a visually correct fold with no layer penetration; particle bounding box and velocity remain finite for 300 frames.
5. ✅ Phase 4 `example_cloth_franka_fba.py` runs the full Franka key-pose sequence to completion, producing a rendered video.  Visual acceptance: clean grasp, clean lift, layered fold (no obvious cloth-through-cloth artefacts).
6. ✅ Plan-level commit with the example, the new kernels, the self-contact code path, a side-by-side VBD-vs-FBA render, and a follow-up spec for two-way coupling.
