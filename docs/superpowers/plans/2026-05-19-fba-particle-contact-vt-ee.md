# Plan: SolverFBA particle-contact — extend from v-v to v-t + e-e

**Date:** 2026-05-19
**Status:** Drafted, not started.
**Owner:** TBD
**Estimated effort:** ~2-3 weeks (Phase 0 BVH setup 2-3 d, Phase 1 v-t broadphase + 4-particle Jacobian 4-5 d, Phase 2 BSR extension 2-3 d, Phase 3 e-e 4-5 d, Phase 4 friction + Franka validation 4-5 d).
**Target failure mode:** Vertex passes between the vertices of a triangle on another cloth surface, evading the current `ParticleContactBroadphase` (which only emits vertex-vertex pairs).

---

## Why

The Phase 3 cloth-particle-contact landing in `2026-05-19-fba-franka-cloth.md` enumerates **vertex-vertex (v-v)** pairs only.  V-v alone systematically misses two of the three classical cloth-contact failure modes:

| Type | Geometry | Caught by v-v? |
|---|---|---|
| **v-v** | Two particles close to each other | ✅ Yes |
| **v-t** (vertex-triangle) | One particle close to a triangle's *interior* (not its vertices) | ❌ The particle is > threshold from any of the triangle's 3 vertices |
| **e-e** (edge-edge) | Two edges crossing in 3D, far from any vertices | ❌ All 4 endpoints may be far from each other |

Empirically observed in the cloth-on-bar self-contact run (commit `2bdfa527`): the two draped halves no longer pass through each other globally, but **small residual interpenetrations remain** wherever a vertex of one half slips between vertices of the other.  In the multi-layer cloth-on-sphere stacking experiment, this manifests as visible v-t leakage between draped layers.

VBD's reference implementation (`SolverVBD` with `particle_vertex_contact_buffer_size=16`, `particle_edge_contact_buffer_size=20`) caches both v-t and e-e separately per particle / per edge.  RealSim's NSN cloth code also distinguishes the three contact types.

**This plan ports v-t and e-e into the existing `ParticleContactBroadphase` + BSR pipeline.**  Once landed, cloth-on-cloth stacking, fold-up sequences, and gripper-cloth grasping should be visually penetration-free.

## What changes

Five orthogonal pieces of work, joined by a final validation phase:

1. **Per-cloth triangle BVH + edge BVH**, refit each step on the current particle positions.
2. **v-t broadphase + Jacobian**: per particle, find the closest triangle on any cloth (intra and inter); emit a Stage B contact row with a **4-particle Jacobian** (vertex + 3 triangle vertices, signed by barycentric weights).
3. **Extend `build_schur_lite_unified` to 4 nonzero H blocks per row** (currently up to 2).  Sparsity widens from contact-graph-by-vertex to contact-graph-by-vertex-and-triangle, still ≪ dense.
4. **e-e broadphase + Jacobian**: per edge, find the closest other edge; emit a Stage B row with a 4-particle Jacobian (2 endpoints of each edge, signed by parametric positions).
5. **4-particle `Jᵀ·λ` scatter + corresponding ``apply_dt2_inv_mass_correction``**: each row touches 4 particles with explicit ±α·dir·λ signs.

The PD outer loop, A_FBA assembly, rigid-contact NSN inner, and the existing v-v code path are **unchanged** — v-v becomes a special case of v-t (when the barycentric closest point lands exactly on a triangle vertex) and remains a fast path.

## Pre-conditions

- Phase 3 deliverables from `2026-05-19-fba-franka-cloth.md` are landed (commit `2bdfa527`):
  - `ParticleContactBroadphase` (v-v) exists and is validated on cloth-on-bar.
  - `build_schur_lite_unified` accepts 1- or 2-particle rows via `row_particle_b ∈ {-1, ≥0}`.
  - `_solve_nsn_coulomb_lite_bsr` works end-to-end.
  - `apply_dt2_inv_mass_correction_kernel` (LiteNSN-consistent diagonal correction) is the chosen correction path.
- `wp.BVH` exists in our pinned Warp build.  **Confirmed via grep**: `warp/_src/bvh.py` defines `BVH(lowers, uppers, ...).refit()`.

## Architectural decisions

### LOCKED: 4-particle rows are the new max; never expand to 5+

V-t has 4 particles (vertex + 3 triangle verts).  E-e has 4 (2 endpoints of each edge).  No physical cloth-cloth contact needs more.  The BSR H matrix gets a fixed cap of 4 nonzero blocks per row.

### LOCKED: separate BVHs per cloth, queried across all clouths each step

A single global BVH over all cloths' triangles would mix unrelated meshes and bloat the query.  Per-cloth BVHs let the topology filter (exclude triangles that share a vertex with the query particle) work cleanly and let the cross-mesh case be a "query particle in mesh A against triangle BVH of mesh B" naturally.  Each cloth has one tri BVH and one edge BVH; refit costs are O(n_tris) per cloth per step.

### LOCKED: v-v rows stay as a fast path under v-t

When the v-t broadphase returns a triangle whose closest point to the query vertex is at a triangle vertex (one of the three barycentric weights is 1, others 0), emit a v-v row (2-particle Jacobian) instead of a v-t row (4-particle).  This preserves the optimal H sparsity for the common "two cloth particles directly facing each other" case and means the Phase 3 v-v code path keeps running unchanged for that fraction of contacts.

### LOCKED: edge-edge tangent basis built from the segment-pair, not world axes

When two near-parallel edges approach, the segment-segment distance vector is well-defined and points along the shortest line connecting them.  When the edges are exactly parallel, the closest-points pair is degenerate (a family of solutions).  We treat |n| < 1e-9 as the degenerate case and skip emission — the v-t path picks up the contact instead.

### LOCKED: friction μ on v-t and e-e is the same `particle_contact_friction` as v-v

There is no physical reason to use different friction for different contact-row types within the same cloth-cloth interaction.  Single μ keeps the API + code path simple.

### LOCKED: single-env correctness before multi-env (carry-forward)

Same rule as the Franka plan: every new code path is gated on single-env Tier 2 validation before any multi-env run.  No phase rolls forward on faith.

## Phase 0 — Per-cloth BVH setup (~2-3 days)

**Goal:** Build and refit a triangle BVH and an edge BVH for each cloth in the model.  Independent of contact generation — Phase 0 is just plumbing.

### Step 0.1 — Enumerate cloth meshes from the model

The current model stores triangles as a flat `model.tri_indices` (T×3 int32) and edges as `model.edge_indices` (E×4 int32 — Newton's bending stencil).  To group by cloth, partition triangles by their constituent particles' `particle_world` (each `add_cloth_mesh` call gets its own world if we used replicate, otherwise a single contiguous block).

For the simpler case where each cloth is added via a separate `add_cloth_mesh` call in the *same* world (the multi-layer demo), we need a separate `cloth_id` array.  Add `model.tri_cloth_id` (T,) int32 — set by `add_cloth_mesh` to the call index.

### Step 0.2 — Build `wp.BVH` per cloth

Triangle BVH: per cloth, build `wp.BVH(lowers, uppers)` over the current triangle AABBs.  AABBs are computed from the 3 triangle vertices' positions.  Owned by a new `ParticleContactBVH` helper class.

Edge BVH: same structure over the cloth's interior edges (where `edge_indices[i, 2]` and `[i, 3]` are the two endpoints).

### Step 0.3 — Per-step refit

Each `_apply_particle_contact_pass` call:
1. Recompute AABBs from `state.particle_q`.
2. Call `bvh.refit()`.  (Refit is O(T log T); does *not* re-build the tree topology, just the bounding boxes.)

### Step 0.4 — Acceptance gate (Phase 0)

Build a 2-cloth model, refit BVH for 100 steps, confirm `bvh.lowers` / `bvh.uppers` track the deforming AABBs to within fp32 precision.  No physics correctness yet.

## Phase 1 — v-t broadphase + 4-particle Jacobian (~4-5 days)

**Goal:** Replace the current v-v-only path with v-t.  V-v emerges as a special case (barycentric weight at one vertex == 1).

### Step 1.1 — v-t broadphase kernel

Per particle (one thread):
1. Query the triangle BVH of *every other* cloth (and own cloth, with topology filter):
   ```
   wp.bvh_query_aabb(bvh.id, p_lower, p_upper) → triangle iterator
   ```
   where the AABB is `p ± (particle_radius + margin)`.
2. For each candidate triangle:
   a. Skip if the triangle shares a vertex with the query particle (topology filter — only applies for own-cloth queries).
   b. Compute closest point on the triangle to the particle.  Use barycentric coordinates `(b1, b2, b3)` with `b1+b2+b3=1`, `bi ≥ 0`.
   c. Compute `dist = ||p - closest||` and `n = (p - closest) / dist`.
   d. If `dist < threshold`, atomic-append a `(v, t1, t2, t3, b1, b2, b3, n, dist)` row to the output buffer.

### Step 1.2 — v-v special-case detection

Inside the broadphase kernel, after computing barycentric weights, if `max(b1,b2,b3) > 1 - 1e-6`, the closest point is a vertex — emit a v-v row (2-particle) instead of v-t (4-particle).  Keeps the H sparsity tight for this case.

### Step 1.3 — Extend per-row data to carry 4 particles

Currently `_pc_row_pa_d` and `_pc_row_pb_d` carry up to 2 particles per row.  Add:
- `_pc_row_pc_d`, `_pc_row_pd_d` — third and fourth particle indices (or `-1` for v-v / v-rigid rows)
- `_pc_row_wa_d, _pc_row_wb_d, _pc_row_wc_d, _pc_row_wd_d` — float32 signed Jacobian weights per particle

For v-v: `pa, pb` valid, `pc = pd = -1`, weights `(+1, -1, 0, 0)`.
For v-t: `pa = v`, `pb = t1`, `pc = t2`, `pd = t3`, weights `(+1, -b1, -b2, -b3)`.
For e-e (Phase 3): `pa = a1`, `pb = a2`, `pc = b1`, `pd = b2`, weights `(+(1-α), +α, -(1-β), -β)`.

### Step 1.4 — Acceptance gate (Phase 1, single-env)

Two-layer cloth stacking on the sphere (existing demo): with v-v only, layers visibly overlap.  With v-t enabled, the **upper layer should rest on top of the lower layer without interpenetration** — verified by:
- Tier 1: no NaN, particles stay in a sane bounding box.
- Tier 2: per-frame penetration count (pairs of layers' triangles overlapping when projected to the sphere surface normal) ≤ 5 at all frames from 200-600.
- Tier 3: rendered video shows two distinct draped layers.

## Phase 2 — Extend `build_schur_lite_unified` to 4 H blocks per row (~2-3 days)

**Goal:** The BSR sparse Schur path consumes H with 1, 2, or 4 nonzero blocks per row.

### Step 2.1 — Extend triplet emission

`build_schur_lite_unified` currently emits `total_rows + n_self` triplets.  Extend to emit up to `4 · total_rows` triplets, with one triplet per nonzero weight on each row.  Per row r, walk the 4 weight slots and emit a `(r, col, value)` triplet for each `weight != 0`.

### Step 2.2 — Extend scatter for 4-particle rows

`scatter_jt_lambda_two_particle_kernel` → `scatter_jt_lambda_n_particle_kernel`.  For each row, walk all 4 particle slots and atomic-add `weight · α · dir · λ` to the corresponding entry of `_pc_jt_lambda_d`.

### Step 2.3 — Acceptance gate (Phase 2)

Mathematical: build a hand-crafted 4-particle row test (4 cloth particles arranged in a tetrahedron-ish, one v-t contact).  Solve manually using full NumPy LCP and compare against the BSR pipeline output.  Match to fp64 round-off (1e-12 relative).

## Phase 3 — e-e broadphase + Jacobian (~4-5 days)

**Goal:** Edge-edge contacts caught and resolved.

### Step 3.1 — e-e broadphase kernel

Per cloth edge (one thread):
1. Query the edge BVH of every other cloth (and own with topology filter — exclude edges sharing a vertex with the query edge).
2. For each candidate edge B:
   a. Compute parametric closest points on the two segments.  Segment-segment closest-point uses Eberly's algorithm: 1 parametric solve in [0,1]², degenerate-case fallback when |a × b| < ε.
   b. `dist = ||p_A - p_B||`, `n = (p_A - p_B) / dist`.
   c. If `dist < threshold`, emit `(a1, a2, b1, b2, α, β, n, dist)` row.

### Step 3.2 — Degenerate edge-pair handling

When edges are exactly parallel and overlapping, the segment-segment minimum is a non-unique line (a range of α-β).  Detect via `|cross(dir_A, dir_B)| < 1e-9` and skip the e-e emission for that pair — the v-t path will catch the contact via one of the 4 vertices.

### Step 3.3 — Wire into Phase 2's 4-particle row pipeline

e-e rows fill the 4-particle slots: `pa = a1, pb = a2, pc = b1, pd = b2`, weights `(+(1-α), +α, -(1-β), -β)`.  No code-path changes downstream — BSR build / NSN inner / scatter already handle arbitrary 4-particle rows from Phase 2.

### Step 3.4 — Acceptance gate (Phase 3)

Specific edge-edge stress: two cloth strips pinned at opposite corners, configured to cross each other in mid-air (an "X" shape).  Without e-e, the strips pass through each other.  With e-e, they should constrain each other at the crossing.  Verified by:
- Tier 1: no NaN, finite throughout 500 frames.
- Tier 2: at the crossing point, the two strips remain separated by ≥ `particle_radius` (configured contact threshold).
- Tier 3: rendered video shows the X-crossing remains stable.

## Phase 4 — Friction tangent rows + integration validation (~4-5 days)

**Goal:** v-t and e-e contacts have Coulomb friction (Stage B), and the full pipeline validates on the Franka cloth-folding demo.

### Step 4.1 — Tangent basis per contact type

- v-t tangent basis: triangle's plane (already perpendicular to `n` since `n` is the triangle normal).  `t1` = any in-plane vector (e.g., triangle's first edge normalised), `t2 = n × t1`.
- e-e tangent basis: when edges are nearly perpendicular, `t1` can be along the projection of one edge onto the plane perpendicular to `n`; when nearly parallel, the v-t fallback applies anyway.

Each contact emits 3 rows (n + t1 + t2), all with the same 4-particle Jacobian (just different `dir`).

### Step 4.2 — Friction μ wiring

`particle_contact_friction` value (from SolverFBA kwarg) populates `_pc_contact_mu_d` for v-t and e-e rows just as it did for v-v.  No new knob.

### Step 4.3 — Acceptance gate (Phase 4, single-env Franka)

Run `example_cloth_franka_fba.py` (from the Franka plan) with v-t + e-e enabled, full key-pose sequence.  Visual acceptance:
- Cloth folds layer correctly without inter-layer penetration.
- Gripper finger pinches cloth without cloth slipping through the finger pads.
- No NaN, particle bbox within sane limits.

Numerical:
- Per-step v-t pair count: 0 to ~1000 (cloth-T-shirt scale).  Confirms broadphase is firing.
- Per-step e-e pair count: 0 to ~500.
- step_mean ≤ 4× Phase 3 v-v step_mean (10.7 ms × 4 = 43 ms — still real-time-ish at 30 fps).

## Out of scope (deferred to follow-up plans)

- **Continuous collision detection (CCD)**: this plan does discrete contact only.  For very fast cloth motion (gripper closing rapidly), particles can still tunnel between BVH refits.  CCD requires per-step temporal swept-volume BVH queries — significant additional work.
- **Per-edge / per-vertex BVH caching**: the per-step refit is the easy-but-not-optimal path.  Building a hierarchical incremental update would be faster but adds complexity; leave for later optimization.
- **multi-env v-t / e-e**: rigorously the same as Phase 4 of the Franka plan — `replicate()` should compose, but the BVH-per-cloth setup needs to enumerate per-world cloths.  Validate single-env first; multi-env is a separate phase.
- **GPU-side parallel BVH build** (vs refit): we use refit-only since topology is fixed.  Re-building from scratch would be needed only if cloth topology changes mid-simulation (not in our scope).

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `wp.BVH.refit()` is too slow when called every step on 1k-tri cloths | Phase 4 step_mean blows past the 43 ms budget | Profile early in Phase 0.  If too slow, batch refits across multiple cloths in a single launch, or skip refit every other step (acceptable for cloth's continuous deformation). |
| v-t and e-e produce massive over-counting of contacts (each pair generates rows in both v-t and e-e and v-v simultaneously) | W matrix gets dense by row, BSR perf collapses | Phase 1.2's v-v fallback already de-dupes v-t→v-v.  e-e emission excludes pairs where v-t catches the closest point.  If profiling still shows over-counting, add explicit pair dedup via a per-step hash set. |
| 4-particle rows produce ill-conditioned `A_schur` when barycentric weights are near-degenerate (e.g., closest point at a triangle vertex but we mis-detected and emitted a v-t row) | PCR fails to converge, lam blows up | Phase 1.2 detection is conservative (`b > 1 - 1e-6` → v-v); test with adversarial geometry.  Worst case PCR caps iters and the row's λ stays bounded by lambda-cap. |
| Edge-edge degenerate parallel case isn't caught and produces NaN normal | `Jᵀ·λ` accumulates NaN, scatter pollutes a particle, sim diverges | Step 3.2's `|cross| < ε` check is the guard.  If we still see NaN in testing, add a sanity check in `emit_*_kernel` for non-finite `n`. |
| Phase 4 step time exceeds 43 ms even with all optimizations | Real-time Franka demo not achievable; degrades visual story | Acceptable; document the perf cap.  The physics is the goal here, not real-time.  Future plan addresses perf. |

## Done definition

This plan is done when:

1. ✅ Phase 0 BVH refit tracks deforming AABBs to fp32 round-off, single-env, 2 clouths.
2. ✅ Phase 1 v-t two-layer-on-sphere shows ≤ 5 frame-overlap-events from frame 200-600 (single-env).
3. ✅ Phase 2 BSR pipeline reproduces a hand-crafted 4-particle LCP to fp64 round-off.
4. ✅ Phase 3 e-e X-crossing strips stay separated by ≥ `particle_radius` at the crossing (single-env).
5. ✅ Phase 4 Franka cloth-folding demo runs the full key-pose sequence with visible layered fold, no obvious penetration, no NaN.  step_mean ≤ 43 ms.
6. ✅ Plan-level commit with new BVHs, v-t kernels, e-e kernels, extended BSR pipeline, validation harnesses, side-by-side video (v-v vs v-t+e-e) of cloth-on-sphere multi-layer and Franka.
