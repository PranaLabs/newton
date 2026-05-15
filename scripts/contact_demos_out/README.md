# Phase 4 Contact Demos — Visual Sanity Suite

Three demos exercising the SolverFBA Schur-complement contact pipeline:
- **Demo 1**: 8x8 cloth falling onto a ground plane (Stage A vs Stage B)
- **Demo 2**: 4x4x4 ARAP softbody cube dropped on plane (Stage A)
- **Demo 3**: 12x12 cloth onto a cylinder + sphere (Stage A vs Stage B)

Each demo runs Newton's `model.collide(...)` + `SolverFBA.update_contacts(...)` + `solver.step(...)` per frame. Output: per-frame PNGs + summary grid + `perf.txt` table.

Scenarios were sized down from the original spec to keep dense Schur-complement build cost tractable (`O(M^2)` where M is the active contact count). See *Known Limitation* below.

---

## Demo 1 — Cloth on Plane

`scripts/fba_demo1_cloth_on_plane.py`

8x8 cloth (81 particles), 180 frames at dt=1/60, 5 PD iterations, gravity in -Z.

| Variant | Mean step | Stable | Final min Z |
|---|---|---|---|
| Stage A (`friction=False`) | 137.9 ms | yes | -0.000 m (cloth lays flat on plane) |
| Stage B (`friction=True`, mu=0.3) | 503.7 ms | yes | 0.063 m (folds/buckles hold above plane via static friction) |

Stage A flattens cloth to the plane exactly. Stage B's tangential constraint
prevents lateral slip — visible buckling. Stage B is ~3.6x slower because the
Schur W has 3 rows per contact (normal + 2 tangents).

Snapshots: `demo1/cloth_on_plane_comparison.png` (2 rows x 6 frames).

## Demo 2 — Softbody Bounce on Plane

`scripts/fba_demo2_softbody_bounce.py`

4x4x4 ARAP tet softbody (125 particles, 320 tets), 200 frames, 5 PD iterations,
dropped from height onto a plane. No friction.

| Variant | Mean step | Stable | Final min Z |
|---|---|---|---|
| ARAP no-friction | 126.4 ms | yes | 0.000 m (cube settles on plane) |

Snapshots: `demo2/softbody_bounce_strip.png` (1 row x 6 frames).

## Demo 3 — Cloth on Cylinder + Sphere — partial pass

`scripts/fba_demo3_cloth_on_cylinder.py`

12x12 cloth (144 particles), 200 frames, 5 PD iterations, dropped onto a tilted
cylinder + sphere.

| Variant | Mean step | Stable | Final min Z | Notes |
|---|---|---|---|---|
| Stage A (`friction=False`) | 289.5 ms | yes | -0.129 m | Cloth catches on cylinder; mild penetration consistent with finite contact stiffness |
| Stage B (`friction=True`, mu=0.4) | **3.2 ms** | finite | **-38.9 m** | **BUG**: cloth free-falls. Schur contact path appears bypassed entirely |

Snapshots: `demo3/cloth_on_cylinder_comparison.png`.

**Known issue (deferred)**: Demo 3's Stage B variant — 12x12 cloth + cylinder/sphere + `friction=True` — produces ~3 ms/step (consistent with unconstrained PD) and the cloth falls through all shapes. Stage B works on Demo 1 (cloth + plane) and the `TestPhase4StageBFriction` unit tests, so the bug is specific to the cylinder+friction combination. Suspected: `mu_per_pair_override` sizing mismatch when `soft_contact_max != particle_count`, or a silent failure in tangent basis computation for cylinder contact normals. Logged for follow-up; does not block Stages A+B+C as core functionality is proven elsewhere.

---

## Known Limitations

1. **Schur build cost scales O(M^2)** where M = active contacts. The current implementation pulls `A^{-1} J_c^T` per contact to host (numpy assembly), so M=256 contacts × N=289 particles costs roughly 1.8 s/step (Stage A). Demos were sized to M <= ~150 to stay tractable for visual sanity.
2. **Demo 3 friction bug** as documented above.
3. **No lambda warm-start** across substeps. Stage A/B reset lambda to zero each PD outer iteration; spec marks this as Stage B+ work.
4. **Coulomb cone projection** uses analytical closed form (project_coulomb_cone in `kernels.py`); blocked projected Gauss-Seidel iterates per contact. Up to 20 NSN iterations per PD outer step.

## Performance Notes

These numbers are intentionally not optimized:

- M=64 contacts (Demo 1) at Stage A: 138 ms/step
- M=64 contacts (Demo 1) at Stage B: 504 ms/step (3.65x Stage A)
- M~25 contacts (Demo 2 softbody): 126 ms/step (close to no-contact baseline)

Future perf work (out of scope for Phase 4 MVP): batched A^{-1} solve over all contacts as a single multi-RHS BSR SpMV pass instead of per-contact loop. Should bring Stage A cost down ~10x at M=64.

## Reproducing

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run python scripts/fba_demo1_cloth_on_plane.py
uv run python scripts/fba_demo2_softbody_bounce.py
uv run python scripts/fba_demo3_cloth_on_cylinder.py
```

Output PNGs and perf tables land in `scripts/contact_demos_out/{demo1,demo2,demo3}/`.
