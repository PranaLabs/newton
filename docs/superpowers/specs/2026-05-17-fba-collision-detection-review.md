# Collision Detection Compliance Review

**Date:** 2026-05-17
**Scope:** Per-shape collision-data alignment from Newton pipeline → FBA NSN inputs
**Shapes covered:** Sphere, Cylinder static, Cylinder rolling, Plane, Mesh
**Excluded:** Box (no in-scope demo uses)

**Reference sources:**
- RealSim collision impls: `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/collisiondetection/` (NB: actual path is `collisiondetection`, not `lagrange/collision` as the task prompt suggested).
- RealSim contact base: `include/Scomponent/collisiondetection/BaseCollision.h:25-35` (tangent basis).
- RealSim Jacobian/`pene0`: `include/Scomponent/lagrange/LagrangeConstraint.h:33-64`; `LocalBasis` struct at `include/Scomponent/contact/ContactPair.h:7-21`.
- Newton pipeline: `newton/_src/geometry/kernels.py:1001-1140` (`create_soft_contacts`).
- FBA consumer: `newton/_src/solvers/fba/solver_fba.py:1055-1307` (`update_contacts`) and `:605-617` (per-step kinematic update).

## Summary

- **MATCH:** 3 (Sphere, Cylinder static, Cylinder rolling — modulo queued Y.1)
- **DIVERGE-intentional:** 2 (Y.1 normal `−0.01·n` offset, Y.2 plane tangent prev-step — both Phase 1.3)
- **DIVERGE-accidental (NEW):** 2 — see below
  - **AA.1**: Friction μ — FBA `sqrt(particle_mu · shape_mu)`; RealSim uses shape `_mu` directly. Masked in all in-scope demos by `mu_per_pair_override`, but breaks for any demo that doesn't override.
  - **AA.2**: Mesh contact (Demo 8 ClothOnKnives) — Newton uses single-particle SDF query; RealSim's `GenericDCD` produces **barycentric-weighted multi-DOF** VF/EE constraints. **Architectural divergence**; BLOCKS Demo 8 verification.

## Per-shape

### Sphere (Demo 5 SqueezingBall — static center, no rotation)

**RealSim:** `src/Scomponent/collisiondetection/SphereCollision.cpp:48-83`
- Detection: `(p1 - center).squaredNorm() < r²` (current step).
- Normal `orient = (p0 - center)/‖p0 - center‖` (previous step direction).
- Anchor `point = center + radius · orient` (on sphere surface).
- Normal row: `dir = orient`, `_point = point − 0.01·orient` (1cm interpenetration cushion).
- Tangent rows: `dir = t0/t1` from `generateTangentDirections(orient)`; `_point = point` (no cushion).
- μ: per-shape `_mu`.

**Newton + FBA:**
- `create_soft_contacts` (`kernels.py:1070-1072`): `d = sdf_sphere(x_local, r)`, `n = sdf_sphere_grad = x_local/‖x_local‖` (outward, matches RealSim's `orient`).
- `body_pos = X_bs · (x_local − n·d) = X_bs · (center_local + r·orient_local) = center + r·orient_world` ⇒ surface point. Matches RealSim's `point`.
- `update_contacts` (`solver_fba.py:1146-1166`): `world_anchor = body_pos` (static shape), `offset_n = n · world_anchor`.
- Tangent basis: arbitrary (`compute_tangent_basis`) — same convention as RealSim's `generateTangentDirections` (RealSim uses `ex`/`ey` fallback exact-equality, FBA uses `|n[0]|>0.9` threshold; functionally identical for SqueezingBall normals).
- v_anchor = 0 (sphere is not spinning in any in-scope scene; `is_spinning_shape` only fires for cylinders).

**Divergences:**
- **Y.1 (queued):** FBA does NOT apply `−0.01·orient` cushion on normal anchor. `pene0_n_FBA = n · (center + r·orient)`; `pene0_n_RS = n · (center + r·orient − 0.01·orient) = pene0_n_FBA − 0.01`.
- Tangent anchor: matches (both use `point` = surface anchor in current step).
- μ: see AA.1.

**Verdict:** MATCH (modulo Y.1).

### Cylinder static (Demos 4, 5, 6 — non-spinning cylinder shapes)

**RealSim:** `src/Scomponent/collisiondetection/CylinderCollision.cpp:62-97` with `_avel = 0` ⇒ `disp = 0`.
- Detection: `radial_norm < radius` on current `p0`.
- Normal `n = radial/‖radial‖`; `point = cbase + r·n`.
- Normal row: `_point = point − 0.01·n`.
- Tangent rows when `_avel == 0`: same as rolling — `_point = point` for `axis` tangent, `_point = point + 0·tangent = point` for rolling tangent.
- μ: `_mu`.

**Newton + FBA:**
- SDF cylinder (Z-up local), `n = sdf_cylinder_grad` (outward radial for radial face).
- `body_pos = X_bs · (x_local − n·d)` ⇒ surface point on cylinder shell.
- `update_contacts`: `_shape_omega_h[s_idx] == 0` ⇒ `is_spinning_shape = False` ⇒ arbitrary tangent basis (no axis-aligned override), `v_anchor = 0` ⇒ no dt shift.

**Divergences:**
- **Y.1 (queued):** No `−0.01·n` cushion.
- Tangent basis orientation differs (FBA arbitrary, RealSim uses `axis` + `normal × axis`), but the rectangular Coulomb box is rotation-invariant in the tangent plane, so trajectory unaffected (audit Y.4).

**Verdict:** MATCH (modulo Y.1).

### Cylinder rolling (Demo 1 TwistingBar, Demo 5 SqueezingBall — `shape_angular_velocity` set)

**RealSim:** `CylinderCollision.cpp:88-91, 137-141`:
- Normal row: `dir = n`, `_point = point − 0.01·n` ⇒ `pene0_n = n·point − 0.01`.
- Tangent row 1 (axis): `dir = _axis`, `_point = point` ⇒ `pene0_axis = axis·point`.
- Tangent row 2 (rolling): `dir = (normal × axis).normalized()`, `_point = point + disp · tangent` where `disp = radius · _avel · dt`. ⇒ `pene0_roll = tangent·point + disp`.

**FBA** (`update_contacts:1213-1279, step:605-617`):
- `is_spinning_shape = True` ⇒ `axis_world = quat·(0,0,1)`, `t1 = normalize(n × axis_world)` (rolling direction, MATCHES RealSim's second tangent), `t2 = normalize(n × t1)` ≈ `−axis_world` (sign flipped vs RealSim, but rotation-invariant clamp absorbs sign).
- `v_anchor = −ω · (axis_world × r_local) = ω · radius · t1`.
- `tangent1_offset_base = t1 · world_anchor`; per-step shift `+ dt · t1·v_anchor = + dt·ω·radius = + disp`. ⇒ `pene0_t1_FBA = t1·world_anchor + disp`. **MATCHES RealSim's `pene0_roll`** (modulo `world_anchor==point` on surface).
- `tangent2_offset_base = t2 · world_anchor`; shift `+ dt · t2·v_anchor = + dt · t2·(ω·radius·t1) = 0` (since t2⊥t1). ⇒ `pene0_t2_FBA = t2·world_anchor`. **MATCHES RealSim's `pene0_axis = axis·point`** up to the t2-vs-axis sign flip (rotation-invariant).

Algebra reverified by inspection; the audit Y.3 conclusion holds.

**Divergences:**
- **Y.1 (queued):** No `−0.01·n` cushion.

**Verdict:** MATCH (modulo Y.1).

### Plane (Demo 5 SqueezingBall — finite static plane)

**RealSim:** `src/Scomponent/collisiondetection/PlaneCollision.cpp:100-135`:
- Detection: `norm1 = _orientation · (p1 − base) < 0` (particle on the negative side of plane = penetrating).
- Normal row: `dir = _orientation`, `_point = point_n = p1 − norm1·_orientation` (projection of CURRENT particle onto plane). ⇒ `pene0_n = _orientation · point_n`. **No 0.01 m cushion** (sphere/cylinder only).
- Tangent rows: `dir = _tangent0/_tangent1`, `_point = point_f = p0 − norm0·_orientation` (projection of **PREVIOUS** particle onto plane). ⇒ `pene0_t = tangent · point_f`.

**FBA:**
- Newton SDF: `n = (0,0,1)_local`; `body_pos = X_bs · (x_local − n·d) = X_bs · (x_local with z=0)` = projection of CURRENT particle onto plane. So `world_anchor = body_pos` is the current-step plane projection.
- Normal `pene0`: `n · world_anchor` = `_orientation · point_n`. **MATCHES RealSim.**
- Tangent `pene0`: `t · world_anchor` (current-step). **DIVERGES from RealSim** (Y.2 queued).
- v_anchor = 0 (plane has no `shape_angular_velocity`); no kinematic dt shift.

**Divergences:**
- **Y.2 (queued):** Tangent uses current-step instead of previous-step projection.

**Verdict:** MATCH (modulo Y.2).

### Mesh (Demo 8 ClothOnKnives — static knives mesh + dynamic cloth)

**RealSim** uses `GenericDCD::doVFDetection` + `doEEDetection` (`src/Scomponent/collisiondetection/GenericDCD.cpp:99-286`), NOT a static mesh SDF. ClothOnKnives scene config (`config/Demos/ClothOnKnives/scene.json:18-28`) declares:
```
"genericCCD": { "muVF": 0.0, "muEE": 0.0, "muFV": 0.0,
                "VF": true, "EE": true, "FV": false,
                "alarm": 0.03, "miniseparation": 0.01, "detect_all": true }
```
- `mesh_i` (knives) is static; `mesh_j` (cloth) is dynamic. With `first_static=true`:
  - **VF branch** (lines 130-148): tests static knife VERTEX `v` against dynamic cloth TRIANGLE `(f0, f1, f2)`. Constraint has **3 DOFs** = `(f0, f1, f2)` with barycentric coefficients `(1−α−β, α, β)`. Direction = `normal = −(AB × AC).normalized()` (triangle outward). Anchor = `p_t0.row(v) + minimum_separation·normal` (the static knife vertex position, shifted by 1cm in normal). Tangent rows use anchor = `p_t0.row(v)` (no separation shift).
  - **EE branch** (lines 224-244): tests static knife EDGE against dynamic cloth EDGE. 2 DOFs from cloth edge endpoints with weights `(1−β, β)`. Direction = `(eb0 + β·CD − ea0 − α·AB).normalized()` (CCD penetration direction). Anchor = the static knife-edge interpolated point + separation.
- μ = 0 throughout (no friction). All contacts go through the **unilateral** FB path on RealSim's side.

**Newton + FBA:**
- `create_soft_contacts` for `geo_type == MESH` (`kernels.py:1094-1116`): calls `wp.mesh_query_point_sign_normal(mesh, x_local/scale, ...)`. This is a **point-to-mesh SDF closest-point query** — it produces a single contact per (particle, shape) pair: `body_pos = X_bs · shape_p` (closest point on knife surface), `normal = (x_local − shape_p) · sign` (signed by inside/outside test).
- `update_contacts` packs this as a 1-DOF particle contact with `alpha = 1.0` and `world_anchor = body_pos`.

**Architectural divergence:**
1. **Direction of detection is swapped:** RealSim tests KNIFE vertices against CLOTH triangles (knives poke up through cloth). Newton tests CLOTH particles against KNIFE volume (cloth particles enter knife SDF).
2. **DOF count per constraint differs:** RealSim VF generates 3-DOF constraints with barycentric weights; Newton/FBA always generates 1-DOF constraints.
3. **EE contacts entirely absent in Newton's path** — `create_soft_contacts` has no edge-edge mode.
4. **Contact normal differs:** RealSim's VF normal is the cloth-triangle outward normal; Newton's normal is the knife-surface outward gradient. These are not the same direction (different surfaces).
5. **Anchor differs:** RealSim's VF anchor = static knife-vertex position; Newton's anchor = closest-knife-surface-point to the cloth particle.

The 1cm `minimum_separation` is analogous to (but distinct from) the 0.01 m cushion used for sphere/cylinder. RealSim's mesh path uses `point + separation·normal` (additive cushion in the normal direction), while sphere/cylinder use `point − 0.01·normal` (negative cushion). The sign matters: separation is added in the direction that allows 1cm gap before contact engages.

**Divergence verdict: DIVERGE-accidental (architectural). BLOCKS Demo 8 verification.**

It is **not possible** to make Newton's `create_soft_contacts` SDF path produce barycentric-weighted multi-DOF contacts. To match RealSim on ClothOnKnives, Newton/FBA must either:
- (a) Add a separate `GenericDCD`-style VF/EE detection path that emits multi-DOF contacts and extends FBA's contact buffer schema to support `n_dof > 1` per row with barycentric coeffs (matches RealSim's `LocalBasis::_n, _index, _coeff` array), OR
- (b) Accept that Demo 8 is out-of-scope for byte-exact verification with the current Newton pipeline and validate only the cloth dynamics + plane contact behavior.

## DIVERGE-accidental list (BLOCKS)

### AA.1 — Friction μ mixing rule

**RealSim:** `_mu` is taken directly from the per-shape collision constructor (`SphereCollision._mu`, `CylinderCollision._mu`, `PlaneCollision._mu`). The friction coefficient stored on `ContactPair.mu` is the obstacle's own μ, with no mixing against any per-particle friction.

**FBA** (`solver_fba.py:1293-1306`):
```python
mu_h[c] = float(np.sqrt(particle_mu * float(shape_mat_mu[s_idx])))
```
This is VBD-style geometric-mean mixing between `particle_mu` (per-cloth-particle) and `shape_material_mu[shape]`.

**Impact:**
- **Currently MASKED in all in-scope demos** because Demos 1/3/5 pass `mu_per_pair_override=np.full(M, fixed_mu)` and Demos 2/4 use `friction=False` (no μ path at all).
- A new demo that omits `mu_per_pair_override` would silently produce μ_effective = √(particle_mu · shape_mu) ≠ RealSim's μ_effective = shape_mu. With default `particle_mu = 0.5` and shape μ = 0.5, FBA gives 0.5 (coincidence), but for shape μ = 0.3, FBA gives √(0.5·0.3) ≈ 0.387 instead of 0.3.

**Resolution:** Either drop sqrt-mixing and use `shape_material_mu` directly (RealSim parity), or document the divergence and require `mu_per_pair_override` for all RealSim-parity demos.

### AA.2 — Mesh contact representation (Demo 8 architectural)

See "Mesh" section above. Newton's `wp.mesh_query_point_sign_normal` produces single-particle SDF contacts; RealSim's `GenericDCD` produces barycentric multi-DOF VF/EE constraints with reversed detection direction. The two pipelines cannot be made byte-equivalent without re-architecting either the Newton collision pipeline or FBA's contact-buffer schema.

**Resolution required before Demo 8 verification.**

## Other items verified (no divergence)

- **Contact-set update frequency:** Both RealSim (`prepare`) and FBA (`update_contacts` called once from `step()`) update the active set exactly once per step. ✓
- **Contact lexsort (S.1):** Reclassified DIVERGE-intentional per 2026-05-17 user decision; permutes `lam` indices only, dense W solve invariant. ✓
- **Normal direction convention:** Particle above plane (y=0) ⇒ both produce `n = (0, +1, 0)` (push direction from obstacle to particle). ✓ (RealSim plane orientation defaulted to `+y` if zero-norm; Newton plane local +z transformed to world +y by shape transform).
- **Tangent basis rotation invariance for cylinder rolling:** Verified by algebra (`t2_FBA = normal × t1 ≈ −axis`, RealSim's `t2 = +axis`; sign flip absorbed by symmetric box clamp).
- **v_anchor sign for spinning cylinder:** FBA's `v_anchor = −ω · (axis × r_local)`. Substituting `r_local = radius · normal_radial`: `axis × normal = −(normal × axis) = −t1`, so `v_anchor = −ω · (−radius · t1) = +ω·radius·t1`. Resulting `dt·(t1·v_anchor) = dt·ω·radius = disp`. Matches RealSim's `disp = radius·_avel·dt` on the rolling-direction tangent.
- **Static cylinder treated as non-spinning:** `is_spinning_shape` predicate requires `_shape_omega_h[s_idx] != 0`. Default is 0 unless `shape_angular_velocity` constructor arg sets it. ✓
- **Per-step contact-set culling/sorting alignment:** Lexsort permutes `body_pos_h, normal_h, particle_h, shape_h` consistently *before* the offset/tangent computation loops, so per-contact `_contact_offset_h[c]`, `_contact_normal_h[c]`, etc. stay row-aligned. ✓

## Open questions

1. **AA.1 resolution:** Should FBA drop the `sqrt(particle_mu · shape_mu)` mixing and use `shape_material_mu` directly (clean RealSim parity), or is the sqrt-mixing intentional for forward compatibility with VBD-style co-simulation? Decision needed before any non-overriding demo can run.
2. **AA.2 resolution:** Demo 8 verification requires either (a) a new `GenericDCD` path in Newton, or (b) explicit out-of-scope acknowledgment. The 2026-05-17 scope cut kept Demo 8 in scope; this finding may force a re-cut. Confirm.
3. **Plane normal cushion:** RealSim plane DOES NOT apply the `−0.01·n` cushion (only sphere/cylinder do). FBA also does not. ✓ Verify this asymmetry is the intended physics (plane is treated as harder constraint than sphere/cylinder).
