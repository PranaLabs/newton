# FBA vs VBD Benchmark — Design Spec

**Date:** 2026-05-20
**Branch:** `ziqiu/fba-solver-design`
**Author:** ziqiu
**Status:** Design approved by user; ready for implementation plan

---

## 1. Goal

Produce a fair, reproducible head-to-head comparison between `SolverFBA` and
`SolverVBD` on a single cloth scene, answering the question: **at matched
visual behavior, which solver is faster per simulated frame, and how does the
wall-clock vs accuracy trade-off look?**

Output is a single script with structured artifacts: numeric report,
Pareto/quality plots, side-by-side video, and final-frame error heatmaps.

Non-goals:
- No multi-scene sweep (no soft body, no contact)
- No grid-size sweep (single 32×32 grid)
- No GPU/CPU/precision sweep
- No claims about "absolute physical correctness" — only comparative behavior
  between the two solvers under a calibrated common reference

## 2. Why this is hard

Naively passing identical `tri_ke` to both solvers does **not** produce
matched physics:

- `SolverVBD` reads `tri_materials[:, 0:2]` as `(mu, lambda)` of a Stable
  Neo-Hookean (Smith et al. 2018) membrane energy.
- `SolverFBA` defaults to ARAP and supports `stretching_model ∈ {arap,
  corotational, neohookean}`; the NH path takes `(mu, lam)` from the
  constructor and additionally applies a per-triangle `tri_weight = ke·area`
  scale.

Even after aligning to NH on both sides, residual implementation differences
(bending term, damping interpretation, mass lumping) mean the converged
shapes can disagree. Without addressing this, an RMS-vs-reference comparison
is measuring constitutive mismatch, not solver efficiency.

## 3. Approach (3 phases)

### Phase 0 — FBA anchor (deterministic, one-shot)

Fix FBA as the reference physics. Run once at high iteration count and store
the full state trajectory; this is the ground truth all subsequent points
compare against.

| Setting | Value |
|---|---|
| Grid | 32×32 cloth, cell 0.05 m |
| Pinning | Two top corners (`particle_mass = 0`) |
| Gravity | `(0, -9.81, 0)`, Y-up |
| Stiffness | `tri_ke = 1000`, `tri_ka = 1000`, `tri_kd = 0`, `edge_ke = 0.1`, `edge_kd = 0` |
| Mass | `0.1` per particle |
| `dt` | 1/100 s |
| Frame count | 800 (enough to settle: ~8 s) |
| Solver | `SolverFBA(stretching_model="neohookean", mu=1000, lam=1000, iterations=200)` |

Outputs of Phase 0:
- `x_FBA_ref[t, v, 3]` — full trajectory, shape `(800, 1089, 3)`
- `x_FBA_star = x_FBA_ref[-1]` — terminal anchor shape
- Stored to disk as `.npy` so subsequent phases don't need to re-run it

### Phase 1 — VBD calibration to FBA behavior

User-chosen strategy: tune VBD parameters so its **steady-state behavior
visually matches FBA's**, then compare against this matched baseline.

**Search space (start at 1D, auto-upgrade to 2D if needed):**

1D grid: `α ∈ {0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0}` controls
`tri_ke_VBD = α · 1000`. All other params identical to FBA anchor
(`tri_ka = 1000`, `edge_ke = 0.1`).

**Per-candidate run:** `SolverVBD(iterations=10, particle_enable_self_contact=False)`
with `substeps=10`, 800 frames at `dt = 1/100`. Record terminal shape.

**Selection:** `α* = argmin_α ‖x_VBD_terminal(α) − x_FBA*‖_2`

**Floor measurement:** re-run VBD at `α*` with `(substeps=40, iter=40)` →
`x_VBD_hi`. Define `floor_RMS = ‖x_VBD_hi − x_FBA*‖_2`. This is the **lower
bound** on cross-solver agreement; all Pareto points are interpreted relative
to it.

**Auto-upgrade to 2D:**
- If `α*` is at the boundary of the grid (first or last element of the
  array): extend the grid by appending one additional outer point in the
  same direction (e.g. add `0.35` to the low end, or `3.0` to the high end).
  Maximum 2 such extensions.
- If `floor_RMS / max_sag_FBA` > 0.5% after best 1D fit
  (where `max_sag_FBA = ‖x_FBA*[:, 1] − initial_y‖_∞`), add a 2D grid over
  `(α, β)` with `β ∈ {0.5, 0.75, 1.0, 1.5, 2.0}` controlling
  `edge_ke_VBD = β · 0.1`. Re-run the floor verification at the 2D optimum.
- 2D grid is 8 × 5 = 40 candidates; still bounded

**Phase 1 outputs:**
- `calibration.json` — `{α*, floor_RMS, grid_losses, upgraded_to_2d, β*}`
- `calibration.png` — loss curve (1D) or heatmap (2D) with chosen point marked

### Phase 2 — Pareto sweep

With calibrated `α*` (and optionally `β*`), measure wall-clock vs RMS for
both solvers across a fixed budget grid.

**FBA configs (4):** `iter ∈ {5, 10, 20, 40}`, single substep, all other
params = anchor.

**VBD configs (4):** `(substeps, iter) ∈ {(2,10), (5,10), (10,10), (10,20)}`,
`tri_ke = α* · 1000`, `edge_ke = β* · 0.1` if 2D was triggered.

**Per-config measurement:** 800 frames at `dt = 1/100`. Skip first 10 frames
as warmup (JIT, kernel cache). Record:
- Per-frame `wall_clock_step[t]` — wrap `solver.step(...)` with
  `wp.synchronize() / time.perf_counter() / synchronize`
- Per-frame `rms[t] = sqrt(mean_v ‖x_config[t, v] − x_FBA_ref[t, v]‖_2^2)` —
  root-mean of per-vertex L2 displacement, in meters
- For FBA only, also record `solver.get_timing_summary()` (PD breakdown);
  stored as supplementary, not shown on Pareto plot

**Shared reference:** all Pareto points compare against `x_FBA_ref[t]`.
- FBA points measure self-convergence rate
- VBD points measure how well calibrated-VBD tracks FBA's trajectory
- The `floor_RMS` from Phase 1 is the asymptote for VBD points

### Phase 3 — Outputs

All under `scripts/fba_vs_vbd_bench_out/`:

| Artifact | Format | Content |
|---|---|---|
| `results.json` | JSON | all timings, RMS curves, config metadata, floor, α* (β*) |
| `calibration.png` | PNG | Phase 1 loss curve (or 2D heatmap) |
| `pareto.png` | PNG | scatter: x=mean wall-clock/step (ms), y=terminal RMS (m). FBA blue, VBD orange. Each point labeled with its config. Horizontal dashed line at floor_RMS. |
| `rms_over_time.png` | PNG | 8 curves (4 FBA + 4 VBD), `rms[t]` vs frame index. Identifies transient vs settled behavior. |
| `three_up.mp4` | MP4 | side-by-side video: Reference (FBA-200) \| FBA-best \| VBD-best. "best" = lowest terminal RMS per solver. Frame-aligned, identical camera. |
| `error_heatmap_fba.png` | PNG | terminal frame, vertices colored by `‖Δx‖` vs `x_FBA*`. FBA-best config. |
| `error_heatmap_vbd.png` | PNG | same for VBD-best config. Shared color scale. |
| `report.md` | Markdown | header with α*, floor; results table; embedded images |

## 4. Architecture

Single self-contained Python script: `scripts/fba_vs_vbd_bench.py`.

```
fba_vs_vbd_bench.py
├── build_model_fba(α=1.0, β=1.0)        -> Model            # neohookean, no coloring
├── build_model_vbd(α, β)                 -> Model            # neohookean params + coloring
├── run_solver(model, solver, n_frames,
│              n_warmup, record=True)     -> RunResult
│       RunResult = {trajectory[t,v,3],
│                    wall_clock_step[t],
│                    fba_timing_summary?}
├── phase0_fba_anchor(out_dir)            -> x_FBA_ref, x_FBA_star
├── phase1_calibrate_vbd(x_FBA_star,
│                       grid, out_dir)    -> α*, β*, floor_RMS
├── phase2_sweep(x_FBA_ref, α*, β*,
│               out_dir)                  -> list[ConfigResult]
├── render_three_up(x_FBA_ref, fba_best,
│                   vbd_best, out_dir)    -> mp4
├── plot_calibration / plot_pareto /
│   plot_rms_over_time / plot_heatmaps
└── write_report
```

### CLI

```
uv run python scripts/fba_vs_vbd_bench.py \
    [--out-dir scripts/fba_vs_vbd_bench_out] \
    [--frames 800] \
    [--warmup 10] \
    [--skip-video] \
    [--skip-phase0]      # reuse cached x_FBA_ref.npy
    [--skip-phase1]      # reuse cached calibration.json
```

`--skip-phase0` / `--skip-phase1` are essential during iteration: rendering and
calibration are the slow parts; sweep + plots iterate fast.

### Builder layer (the "same scene" guarantee)

`build_builder(tri_ke, tri_ka, edge_ke)` returns a fresh `ModelBuilder` with
the 32×32 cloth grid (parameters passed straight into `add_cloth_grid`) and
the two top-corner pin masses set. `α` and `β` are applied at this layer —
**no post-finalize mutation of model arrays**.

- FBA path: `build_builder(mu, lam, 0.1).finalize()` with `mu = lam = 1000`.
- VBD path: `b = build_builder(α·1000, 1000, β·0.1); b.color(include_bending=True); b.finalize()`.

`β` defaults to 1.0 in 1D search; only varies in the 2D upgrade branch.

### Wall-clock measurement

```python
def timed_step(solver, s_in, s_out, dt):
    wp.synchronize_device()
    t0 = time.perf_counter()
    solver.step(s_in, s_out, None, None, dt)
    wp.synchronize_device()
    return time.perf_counter() - t0
```

Warmup discards the first `n_warmup` step times before stats. The first step
of FBA triggers SPDM symbolic factorization; first step of VBD triggers BVH
allocation. Neither is representative.

### Video rendering

Reuse the established headless `ViewerGL` pattern from
`scripts/render_demo5_to_video.py` and `scripts/fba_cloth_layers_on_sphere.py`:

```python
viewer = newton.viewer.ViewerGL(width=W, height=H, headless=True)
viewer.set_model(model)
# per frame:
viewer.begin_frame(t); viewer.log_state(state); viewer.end_frame()
frame_buf = viewer.get_frame()  # HxWx3 uint8
```

Three viewers are instantiated for the three-up (Reference, FBA-best,
VBD-best). Each gets the same camera pose. Each step we capture all three
frames, `np.hstack` them, and append to an `imageio.get_writer(..., fps=100,
codec="libx264")` MP4. Identical seed (cloth has no RNG, but Warp BVH does
during construction — color groups are deterministic given the mesh).

## 5. Determinism

- Same random seed not strictly needed (no stochastic kernels on hot path)
- `builder.color()` is deterministic given the topology
- Floating-point reductions in VBD's Gauss-Seidel sweeps are deterministic
  per Warp kernel (no atomic adds in the membrane path); FBA's PD scatter
  uses deterministic compute+gather
- All artifacts must be reproducible byte-for-byte across reruns. CI does
  not run this; user re-runs locally.

## 6. Tolerance and acceptance

This is exploratory work, not a CI gate. The script does not assert pass/fail.
Output is interpreted by reading `report.md`. The script is "done" when:

1. All three phases run end-to-end without errors
2. `pareto.png` shows distinguishable FBA and VBD scatter clusters
3. `three_up.mp4` is visually coherent (no exploding cloth, all three settle)
4. `report.md` contains α*, floor_RMS, and the per-config table

If 1D calibration's `floor_RMS / max_sag_FBA` exceeds 0.5%, the auto-upgrade
to 2D triggers. If 2D still fails the threshold, the script proceeds anyway
and `report.md` flags the result as "calibration imperfect, see Phase 1 plot".

## 7. Failure modes

- **VBD diverges at high `α`**: not unusual at stiff settings. Detection:
  `np.any(np.isnan(x)) or np.any(np.abs(x) > 100)`. Action: mark candidate
  as failed in `calibration.json`, exclude from argmin, continue.
- **FBA anchor doesn't settle**: 800 frames at 100 Hz is 8 s, should suffice
  for 5 m of cloth under gravity. If terminal velocity > 1e-3 m/s, warn but
  proceed (cloth pendulum mode is OK).
- **Video render OOM**: `ViewerGL` headless at default resolution is
  ~1280×720; three windows + 8 sweep videos worth of frames is large but
  manageable. If OOM, drop to 640×360 and proceed.
- **First-step JIT timing pollution**: handled by `--warmup 10`.

## 8. Code locations touched

- New file: `scripts/fba_vs_vbd_bench.py`
- New dir: `scripts/fba_vs_vbd_bench_out/` (gitignored as `scripts/*_out/`
  pattern already shows in the tree)
- No public API changes
- No changes to `newton/` source
- No new tests under `newton/tests/` (this is a script, not a feature)

## 9. Estimated effort

- Phase 0 implementation: ~30 lines (calls existing FBA construction)
- Phase 1: ~80 lines (grid loop + loss + auto-upgrade)
- Phase 2: ~60 lines (sweep loop + record)
- Phase 3 rendering: ~120 lines (three viewers + imageio)
- Plotting + report: ~100 lines
- Total: ~400 lines, single file

End-to-end runtime on a single RTX 4090 (estimate):
- Phase 0: ~40 s (FBA iter=200 × 800 frames)
- Phase 1: ~3 min (8 candidates × 800 frames + 1 hi-fi verify)
- Phase 2: ~2 min (8 configs × 800 frames)
- Three-up video render: ~3 min (3 viewers × 800 frames + encode)
- Total: ~9 min cold; ~5 min with `--skip-phase0 --skip-phase1`

## 10. Decisions log

- **Scene:** hanging cloth only (no contact). Contact-active scenes have
  no shared fixed point between solvers; deferred.
- **Quality metric:** per-vertex RMS position error vs reference trajectory.
  Chosen over implicit Euler residual norm and energy curves.
- **Reference:** FBA at `iter=200` with explicit NH params. Chosen over
  scipy offline solve (extra dependency) and per-solver references (less
  cleanly comparable).
- **Calibration:** VBD `tri_ke` scaled by `α`; auto-upgrades to 2D
  `(α, β)` only if 1D fails. Chosen over fixed `α=1` (constitutive mismatch
  contaminates the comparison) and full 3D search (search budget explodes).
- **Sweep design:** preset 4×4 grid, Pareto scatter. Chosen over bisect-to-
  threshold (non-monotone risk) and single-config (doesn't answer "equal
  quality").
- **Stiffness anchor value:** `mu = lam = 1000`. Mid-range cloth — VBD
  example uses `tri_ke = 1e3` comfortably; FBA's docstring notes
  `tri_ke ~ 1e4` is realistic but 1e3 keeps the comparison numerically
  benign on both sides. Not investigating stiff regime in this pass.
- **Grid size:** 32×32 fixed. Scaling sweep deferred.
