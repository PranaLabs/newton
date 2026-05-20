# SolverFBA branch

Working branch (`ziqiu/fba-solver-design`) for **`SolverFBA`** — a
projective-dynamics + non-smooth-Newton soft-body solver added to
Newton, ported from RealSim/CudaTests with full Warp + sparse-PCR
plumbing. This README only covers what's new on this branch; see
upstream `main` for general Newton documentation.

## Quickstart

```bash
uv sync --extra examples
uv run python -m newton.examples <example_name>
```

## What this branch adds

### New solver module: `newton/_src/solvers/fba/`

| File | Purpose |
|---|---|
| `solver_fba.py` | Public `SolverFBA` class. Outer loop, PD prefactor cache, NSN driver, particle-contact pipeline. |
| `linear_solver.py` | A_FBA assembly (ARAP / corotational / NeoHookean stretching, bending, pin), Cholesky factor cache, lite Schur W build (BSR pipeline, 4-particle Jacobian). |
| `nsn_pcr_solver.py` | Sparse + dense PCR inner driver for FB-Newton λ-system. |
| `kernels.py` | All Warp kernels: per-particle fused FB-Newton (k_p ≤ 4 inline Cholesky), residual / scatter / row emit (v-v / v-t / e-e), BSR build, mass-correction. |
| `particle_contact.py` | Legacy v-v particle-particle broadphase (`wp.HashGrid` + topology + rest-pose exclusion CSR). |

`SolverFBA` is exposed via `from newton.solvers import SolverFBA`. The
v-t / e-e self-contact paths additionally reuse `TriMeshCollisionDetector`
from `newton/_src/solvers/vbd/` (no modifications to that file).

### Key implementation choices

* **Multi-env via lite NSN**: `nsn_schur_mode='lite'` builds W as a
  BSR sparse Schur over per-particle blocks, scaling near-linearly
  across replicated worlds (6.7× single-env speedup, 3.66× scaling
  from N=1 → N=25).
* **Per-particle fused FB-Newton kernel**: 3·k_p × 3·k_p Cholesky
  inline in registers for k_p ≤ 4 contacts; falls back to BSR path
  beyond.
* **4-slot row data layout** for self-contact: every row carries up to
  4 particle indices + 4 signed Jacobian weights, with `-1` sentinels
  for unused slots. v-v (2 slots), v-t (4 slots), e-e (4 slots) all
  share the same downstream pipeline (BSR triplets, scatter, residual).
* **Penetration-only contact emit**: `emit_vt_rows_kernel` and
  `emit_ee_rows_kernel` only allocate LCP rows for pairs with
  `signed_gap < 0`. Pairs in the broadphase warm zone don't bloat W
  with rank-deficient rows. Brings W dimension from 15× DOF down to
  near 1× DOF on a 2500-vertex cloth.
* **Diagnostics**: `_pc_last_vt_hits`, `_pc_last_ee_hits` (broadphase
  candidates) and `_pc_last_n_contacts` (post-filter, = W dim / 3)
  for Tier 3 introspection.

## Demos

### CudaTests / RealSim ports

These were the original "ground-truth" demos used to validate the FBA
port line-by-line against RealSim.

| Demo | Command |
|---|---|
| Demo 2 — twisting bar | `uv run python -m newton.examples twisting_bar_fba` |
| Demo 3 — stretching cloth | `uv run python -m newton.examples stretching_cloth_fba` |
| Demo 4 — pulling wooper | `uv run python -m newton.examples pulling_wooper_fba` |
| Demo 5 — squeezing ball | `uv run python -m newton.examples squeezing_ball_fba` |
| RealSim cloth-on-sphere | `uv run python -m scripts.fba_cloth_on_sphere_realsim_baseline` |
| Hanging cloth (m / cm scale) | `uv run python -m newton.examples cloth_hanging_fba` |
| Soft-body hanging | `uv run python -m newton.examples softbody_hanging_fba` |

### Cloth scenes

| Scene | Script | Notes |
|---|---|---|
| Cloth on 2 obstacles | `scripts/fba_cloth_2obstacle_smoke.py` | Multi-contact validation. |
| Cloth on horizontal bar | `scripts/fba_cloth_on_bar_smoke.py` | Self-contact baseline (cloth wraps + drapes). |
| Cloth on sphere (single env) | `newton.examples cloth_on_sphere_fba` | Per-particle FB-Newton hot path. |
| Cloth on sphere (multi-env) | `scripts/fba_multienv_cloth_bench.py` | N=1, 4, 9, 16, 25 worlds via `replicate()` + per-world offsets. |
| Multi-layer cloth on sphere | `scripts/fba_cloth_layers_on_sphere.py` | 2-3 stacked cloths. `--no-ee` to skip e-e. `--self-contact` enables v-t. |
| Tablecloth lift (work-in-progress) | `scripts/fba_tablecloth_lift.py` | Center pin lifts cloth off ground plane. Working tree only — known NaN at strong fold transients, see "Known limitations" below. |
| Franka cloth folding (work-in-progress) | `newton.examples cloth_franka_fba` | Robot gripper cloth manipulation. v-t self-contact, e-e disabled for stability on T-shirt mesh. |

### Particle-contact validation (Phase 0/1/2/3/4 acceptance gates)

| Phase | Script | Gate |
|---|---|---|
| 0 — detector smoke | `scripts/fba_phase0_tri_mesh_detector_smoke.py` | AABB drift ≤ fp32 over 100 steps. |
| 1 — v-t emit | `scripts/fba_phase1_vt_layers_smoke.py` | 600 frames, 2-layer cloth on sphere, peak penetration ≤ 5. |
| 2 — BSR 4-block | `scripts/fba_phase2_4particle_lcp.py` | Hand-crafted 4-particle LCP vs NumPy ≤ 1e-9 rel. |
| 3 — e-e emit | `scripts/fba_phase3_ee_xcross_smoke.py` | X-crossed cloth strips, EE-ON keeps separation ≥ 5 mm. |
| Self-folding bar | `scripts/fba_self_contact_broadphase_smoke.py` | v-v broadphase + self-folding validation. |

### Render utilities

* `scripts/fba_cloth_on_sphere_record.py` — single-env render to MP4.
* `scripts/render_multienv_cloth_to_video.py` — multi-env grid render
  with auto camera framing.

## Code changes summary (relative to upstream `main`)

This is a high-level map of what's new; commit log has the granular
history (`git log --oneline ziqiu/fba-solver-design --not origin/main`).

1. **`newton/_src/solvers/fba/` (new module)** — `SolverFBA` core,
   including PD prefactor + FB-Newton + sparse-PCR + multi-env lite
   Schur + per-particle fused kernel + particle-contact pipeline.

2. **Self-contact subsystem** —
   * `ParticleContactBroadphase` (v-v, `wp.HashGrid`, topology + rest
     CSR exclusion).
   * v-t / e-e via `TriMeshCollisionDetector` reuse.
   * 4-slot row data, BSR Schur with 1 / 2 / 4 H blocks per row,
     `scatter_jt_lambda_n_particle_kernel`.
   * `apply_dt2_inv_mass_correction_kernel` for the post-NSN
     position update.
   * Friction tangents (3 rows per contact: normal + 2 tangent,
     shared 4-slot Jacobian).
   * **Penetration-only emit** in `emit_vt_rows_kernel` and
     `emit_ee_rows_kernel` — skip pairs with `signed_gap > 0` to
     keep W from becoming rank-deficient.

3. **FBA examples added under `newton/examples/`**: `cloth_hanging_fba`,
   `cloth_on_sphere_fba`, `cloth_franka_fba`, `stretching_cloth_fba`,
   `twisting_bar_fba`, `pulling_wooper_fba`, `softbody_hanging_fba`,
   `squeezing_ball_fba`, and the cm-scale `example_hanging_cloth_fba_cm.py`.

4. **Diagnostics**: `_pc_last_vt_hits` (v-t broadphase candidates),
   `_pc_last_ee_hits` (e-e broadphase candidates), and
   `_pc_last_n_contacts` (post-filter actual emitted contacts;
   `W dim = 3 × _pc_last_n_contacts`).

5. **Plans and supporting docs** under `docs/superpowers/plans/` —
   notably `2026-05-19-fba-franka-cloth.md` and
   `2026-05-19-fba-particle-contact-vt-ee.md`.

## Known limitations

* **Tablecloth-lift NaN at fold transients** — when the per-step
  contact count jumps from O(10) to O(10³) in 1-2 frames (e.g. a
  cloth folding back on itself), `apply_dt2_inv_mass_correction_kernel`
  applies a position update that the soft-cloth PD prefactor can't
  damp in one NSN iteration. Setting `lambda_cap` stabilises this
  but clips ground contact too. Proper fix is a PD-aware correction
  or per-pair λ warm-start across steps; not yet implemented.
* **e-e on dense meshes** — the Franka T-shirt mesh saturates e-e
  broadphase (~80k candidates per step) and OOMs the BSR mm scratch.
  Penetration-only emit knocks this down ~10×; production runs
  currently keep `particle_contact_ee=False` on this asset until
  per-pair dedup with v-t lands.
* **Single-env only for self-contact** — `replicate()` composition
  with v-t / e-e BVH refit not yet validated.

## Running regression

The Phase 0–3 acceptance smokes plus the SqueezingBall and
cloth-on-bar regressions cover the full self-contact + lite NSN
pipeline:

```bash
uv run python -m scripts.fba_phase0_tri_mesh_detector_smoke --frames 20
uv run python -m scripts.fba_phase1_vt_layers_smoke
uv run python -m scripts.fba_phase2_4particle_lcp
uv run python -m scripts.fba_phase3_ee_xcross_smoke
uv run python -m scripts.fba_cloth_on_bar_smoke --self-contact
uv run python -m newton.examples squeezing_ball_fba --viewer null --num-frames 200
```

All six pass green on `9314e244`.
