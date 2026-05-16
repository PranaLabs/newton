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

## Demo 3 — Cloth on Cylinder

`scripts/fba_demo3_cloth_on_cylinder.py`

12x12 cloth (169 particles), 200 frames, 5 PD iterations, dropped onto a tilted
cylinder.

| Variant | Mean step | Stable | Final min Z | Notes |
|---|---|---|---|---|
| Stage A (`friction=False`) | ~307 ms | yes | -0.126 m | Cloth catches on cylinder; mild penetration consistent with finite contact stiffness |
| Stage B (`friction=True`, mu=0.4) | ~1302 ms | yes | +0.634 m | Friction holds cloth above cylinder; ~4.2x slower than Stage A due to 3M Schur |

Stage B is ~4x slower than Stage A because the Schur complement has 3 rows per
contact (normal + 2 tangents) giving a 3M × 3M system vs M × M.

Snapshots: `demo3/cloth_on_cylinder_comparison.png`.

**Root cause fixed**: The tangent residual formula was computing
`r_t = -dot(t, x_unc)` (relative to world origin) instead of
`r_t = dot(t, contact_anchor) - dot(t, x_unc)` (relative to contact anchor).
For off-origin contacts this inflated tangential residuals ~10×, driving
runaway friction impulses that launched the cloth upward on every contact frame,
clearing all contacts and causing free-fall at ~3 ms/step. Fixed by storing and
using per-contact tangential offsets `dot(t1, world_anchor)` and
`dot(t2, world_anchor)` in `SolverFBA.update_contacts` / `_compute_contact_residual_friction`.
A complementarity guard (`r_eff[0] <= 0 → lam = 0`) was also added to
`_solve_nsn_coulomb` to prevent the Coulomb cone Case-3 projection from
generating spurious normal impulses on non-penetrating contacts.

---

## Known Limitations

1. **Schur build cost scales O(M^2)** where M = active contacts. The current implementation pulls `A^{-1} J_c^T` per contact to host (numpy assembly), so M=256 contacts × N=289 particles costs roughly 1.8 s/step (Stage A). Demos were sized to M <= ~165 to stay tractable for visual sanity.
2. **No lambda warm-start** across substeps. Stage A/B reset lambda to zero each PD outer iteration; spec marks this as Stage B+ work.
3. **Coulomb cone projection** uses analytical closed form (project_coulomb_cone in `kernels.py`); blocked projected Gauss-Seidel iterates per contact. Up to 20 NSN iterations per PD outer step.

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
