# Phase 4 Contact Demos — Visual Sanity Suite

Three demos exercising the SolverFBA Schur-complement contact pipeline:
- **Demo 1**: 8x8 cloth falling onto a ground plane (Stage A vs Stage B)
- **Demo 2**: 4x4x4 ARAP softbody cube dropped on plane (Stage A)
- **Demo 3**: 12x12 cloth onto a cylinder + sphere (Stage A vs Stage B)

Each demo runs Newton's `model.collide(...)` + `SolverFBA.update_contacts(...)` + `solver.step(...)` per frame. Output: per-frame PNGs + summary grid + `perf.txt` table.

---

## Approach B Performance (Current)

**Approach B** replaces the sequential per-row solve loop (~15 kernel launches per
contact row x total_rows rows) with 3 batched axis passes using a custom multi-RHS
BSR SpMV kernel. Each axis pass fires only 5 kernel launches regardless of
`total_rows`, reducing launch overhead dramatically.

### Demo 1 — Cloth on Plane (8x8, N=81)

| Variant | Approach A | Approach B | Speedup |
|---|---|---|---|
| Stage A (friction=False) | 137.9 ms | **10.9 ms** | **12.6x** |
| Stage B (friction=True, mu=0.3) | 503.7 ms | **43.3 ms** | **11.6x** |

### Demo 3 — Cloth on Cylinder (12x12, N=169)

| Variant | Approach A | Approach B | Speedup |
|---|---|---|---|
| Stage A (friction=False) | ~307 ms | **15.7 ms** | **~19.6x** |
| Stage B (friction=True, mu=0.4) | ~1302 ms | **41.9 ms** | **~31x** |

---

## Demo 1 — Cloth on Plane

`scripts/fba_demo1_cloth_on_plane.py`

8x8 cloth (81 particles), 180 frames at dt=1/60, 5 PD iterations, gravity in -Z.

| Variant | Mean step | Stable | Final min Z |
|---|---|---|---|
| Stage A (`friction=False`) | 10.9 ms | yes | -0.000 m (cloth lays flat on plane) |
| Stage B (`friction=True`, mu=0.3) | 43.3 ms | yes | -0.000 m (folds/buckles hold above plane via static friction) |

Stage A flattens cloth to the plane exactly. Stage B's tangential constraint
prevents lateral slip — visible buckling. Stage B is ~4x slower than Stage A because the
Schur W has 3 rows per contact (normal + 2 tangents), but both are now much faster than
Approach A due to the batched multi-RHS solver.

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
| Stage A (`friction=False`) | 15.7 ms | yes | -0.127 m | Cloth catches on cylinder; mild penetration consistent with finite contact stiffness |
| Stage B (`friction=True`, mu=0.4) | 41.9 ms | yes | +0.648 m | Friction holds cloth above cylinder; ~2.7x slower than Stage A due to 3M Schur |

Stage B is ~2.7x slower than Stage A (vs ~4x for Approach A) because the larger
`total_rows` used to dominate via launch overhead — now the multi-RHS kernel amortizes
that cost.

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

1. **Schur build cost scales O(M^2)** where M = active contacts. The W matrix kernel is 2D (one thread per (c', c) pair), so M=256 contacts is still tractable. The bottleneck is now the multi-RHS SpMV which is O(R * nnz(S)).
2. **No lambda warm-start** across substeps. Stage A/B reset lambda to zero each PD outer iteration; spec marks this as Stage B+ work.
3. **Coulomb cone projection** uses analytical closed form (project_coulomb_cone in `kernels.py`); blocked projected Gauss-Seidel iterates per contact. Up to 20 NSN iterations per PD outer step.

## Performance Notes (Approach B — current)

Approach B multi-RHS BSR SpMV:

- Demo 1 Stage A (M~64, N=81): **10.9 ms/step** (was 138 ms, 12.6x speedup)
- Demo 1 Stage B (3M~192, N=81): **43.3 ms/step** (was 504 ms, 11.6x speedup)
- Demo 3 Stage A (M~variable, N=169): **15.7 ms/step** (was ~307 ms, ~19.6x speedup)
- Demo 3 Stage B (3M~variable, N=169): **41.9 ms/step** (was ~1302 ms, ~31x speedup)
- Demo 2 softbody (no-contact-solve path): 126 ms/step (unchanged)

Approach B fires 3 batched axis passes x 7 launches/pass = 21 kernel launches per
`build_schur_complement` call, vs ~15 x total_rows launches for Approach A.
At N=81 cloth (total_rows=192 for Stage B), this is 21 vs ~2880 launches.

## Reproducing

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run python scripts/fba_demo1_cloth_on_plane.py
uv run python scripts/fba_demo2_softbody_bounce.py
uv run python scripts/fba_demo3_cloth_on_cylinder.py
```

Output PNGs and perf tables land in `scripts/contact_demos_out/{demo1,demo2,demo3}/`.

---

## Demo 4 — PullingWooper (CudaTests reproduction)

`scripts/fba_demo4_pulling_wooper.py`

5325-particle tet wooper (Neo-Hookean, E=1e7, ν=0.3, mass=1000),
500 frames at dt=0.01, single PULLING action (dir=-Y, vel=2 m/s,
maxlength=10 m), 2 static cylinders (r=1, axis=+X, mu=0), gravity=0,
5 PD outer iter × 1 NSN inner iter, isodof Schur build.

| Solver        | mean ms | median ms | p95 ms | stable |
|---------------|---:|---:|---:|:---:|
| RealSim NSN_CUDA (baseline) | 24.08 | – | – | ✓ |
| SolverFBA (isodof, nsn=1) | **10.54** | 7.58 | 24.21 | ✓ |
| SolverFBA (isodof, nsn=10) | 21.28 | 18.43 | 50.95 | ✓ |

Acceptance:
- Visual: wooper threads through both cylinders, no penetration, no fragmentation
- Stability: `stable=True`, `min_y=-8.410`, `pulled=10.00` at final frame
- Perf: ms/step ≤ RealSim baseline ✓ (0.44× at nsn=1)

### Performance journey

| Stage | mean ms | What changed |
|---|---:|---|
| Initial (Task G) | 505 | Naive Schur build |
| I' Batch 3-axis solves | 204 (sync) | One R=3M solve instead of 3 axis-loop solves |
| J' λ correction → Warp | 438 | Host loop eliminated (was 53 ms/step) |
| L Cache A⁻¹·Jᵀ + W | 100.5 | Build Schur once per step instead of 3.4× |
| N NSN default 10 → 1 | 93 | Single Newton step matches RealSim parity |
| H' Bending fix | 93 | Non-flat rest formula match (Wooper has no bending — no perf change) |
| T' λ warm-start across PD outer | 92 | RealSim parity for contact convergence |
| **P Isodof Schur** | **10.54** | Restrict A⁻¹ to (isodofs × isodofs) instead of dense R=3M |

Snapshots: `demo4/pulling_wooper_strip.png` (6 frames spaced across the pull).
