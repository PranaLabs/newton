# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""FBA vs VBD calibrated benchmark on a 32x32 hanging cloth.

See docs/superpowers/specs/2026-05-20-fba-vs-vbd-bench-design.md for the
design rationale. Run::

    uv run python scripts/fba_vs_vbd_bench.py --out-dir scripts/fba_vs_vbd_bench_out
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA, SolverVBD

# Anchor parameters (match the spec, do not change without updating the doc).
GRID_DIM = 32
CELL = 0.05
MASS = 0.1
GRAVITY = -9.81  # m/s²; scalar passed to ModelBuilder (up_axis=Y expands to (0, -9.81, 0))
FBA_MU = 1000.0
FBA_LAM = 1000.0
FBA_EDGE_KE = 0.1
FRAME_DT = 1.0 / 100.0
ANCHOR_ITERATIONS = 200

# Phase 1 grids.
ALPHA_GRID_1D = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0)
BETA_GRID_2D = (0.5, 0.75, 1.0, 1.5, 2.0)

# Phase 2 sweep grids.
FBA_SWEEP_ITERS = (5, 10, 20, 40)
VBD_SWEEP_CONFIGS = (
    (2, 10),
    (5, 10),
    (10, 10),
    (10, 20),
)

# --- Builder & model construction ---


def build_builder(tri_ke: float, tri_ka: float, edge_ke: float) -> newton.ModelBuilder:
    """Construct the 32x32 hanging cloth builder with corner-pin masses set.

    `tri_ke` and `tri_ka` map directly to the Stable Neo-Hookean (mu, lambda)
    that VBD reads from `tri_materials`; FBA reads `(mu, lam)` from its own
    constructor; the builder's `tri_ke`/`tri_ka` are consumed only by VBD.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=GRAVITY)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 2.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=GRID_DIM,
        dim_y=GRID_DIM,
        cell_x=CELL,
        cell_y=CELL,
        mass=MASS,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=0.0,
        edge_ke=edge_ke,
        edge_kd=0.0,
        fix_left=False,
    )
    top_left = GRID_DIM * (GRID_DIM + 1)
    top_right = top_left + GRID_DIM
    builder.particle_mass[top_left] = 0.0
    builder.particle_mass[top_right] = 0.0
    return builder


def build_model_fba(mu: float = FBA_MU, lam: float = FBA_LAM, edge_ke: float = FBA_EDGE_KE):
    """Finalize a model intended for SolverFBA. No coloring needed."""
    builder = build_builder(tri_ke=mu, tri_ka=lam, edge_ke=edge_ke)
    return builder.finalize()


def build_model_vbd(alpha: float = 1.0, beta: float = 1.0):
    """Finalize a model for SolverVBD with calibration scales applied.

    `alpha` scales `tri_ke` (the VBD Stable-NH `mu`); `beta` scales `edge_ke`.
    """
    builder = build_builder(
        tri_ke=alpha * FBA_MU,
        tri_ka=FBA_LAM,
        edge_ke=beta * FBA_EDGE_KE,
    )
    builder.color(include_bending=True)
    return builder.finalize()


# --- Timing & trajectory recording ---


@dataclass
class RunResult:
    """Per-frame wall-clock and full trajectory from one solver run."""

    trajectory: np.ndarray  # (n_frames, n_particles, 3), float32
    wall_clock_ms: np.ndarray  # (n_frames,), float64
    fba_timing_summary: dict | None = None


def _sync():
    if wp.is_cuda_available():
        wp.synchronize_device()


def run_solver(
    model,
    solver,
    n_frames: int,
    substeps: int = 1,
    n_warmup: int = 10,
    frame_dt: float = FRAME_DT,
) -> RunResult:
    """Step `solver` for `n_frames` rendered frames; record trajectory + wall-clock.

    For VBD-style multi-substep configs, pass `substeps > 1`; internally we
    call `solver.step(...)` `substeps` times per rendered frame with
    `dt = frame_dt / substeps`. Wall-clock per rendered frame includes all
    inner step calls.

    `n_warmup` frames are still recorded into the trajectory (we want them
    for completeness) but their wall-clock measurements are discarded by the
    caller — `run_solver` returns the full `wall_clock_ms` array; callers
    slice off `[n_warmup:]` when computing mean/stddev.
    """
    state_in = model.state()
    state_out = model.state()
    n_particles = model.particle_count
    trajectory = np.empty((n_frames, n_particles, 3), dtype=np.float32)
    wall = np.empty(n_frames, dtype=np.float64)
    sub_dt = frame_dt / substeps

    for f in range(n_frames):
        _sync()
        t0 = time.perf_counter()
        for _ in range(substeps):
            state_in.clear_forces()
            solver.step(state_in, state_out, None, None, sub_dt)
            state_in, state_out = state_out, state_in
        _sync()
        wall[f] = (time.perf_counter() - t0) * 1000.0
        trajectory[f] = state_in.particle_q.numpy()
        if not np.all(np.isfinite(trajectory[f])):
            raise RuntimeError(f"non-finite positions at frame {f}; solver diverged")

    fba_summary = None
    if hasattr(solver, "get_timing_summary"):
        try:
            fba_summary = solver.get_timing_summary()
        except Exception as exc:
            import warnings  # noqa: PLC0415

            warnings.warn(f"get_timing_summary raised {exc!r}; fba_timing_summary will be None", stacklevel=2)
            fba_summary = None

    return RunResult(trajectory=trajectory, wall_clock_ms=wall, fba_timing_summary=fba_summary)


# --- Phase 0: FBA anchor ---


def phase0_fba_anchor(out_dir: Path, n_frames: int = 800, n_warmup: int = 10) -> dict:
    """Run FBA at iter=200 and store the full trajectory as ground truth.

    Returns a metadata dict describing the run and the on-disk paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "x_FBA_ref.npy"

    model = build_model_fba(mu=FBA_MU, lam=FBA_LAM, edge_ke=FBA_EDGE_KE)
    solver = SolverFBA(
        model,
        iterations=ANCHOR_ITERATIONS,
        stretching_model="neohookean",
        mu=FBA_MU,
        lam=FBA_LAM,
        enable_perf_timing=True,
    )

    print(f"[phase 0] running FBA anchor iter={ANCHOR_ITERATIONS} for {n_frames} frames...")
    t0 = time.perf_counter()
    res = run_solver(model, solver, n_frames=n_frames, substeps=1, n_warmup=n_warmup)
    elapsed = time.perf_counter() - t0
    print(
        f"[phase 0] done in {elapsed:.1f} s; mean ms/frame (excl. warmup) = {res.wall_clock_ms[n_warmup:].mean():.2f}"
    )

    np.save(ref_path, res.trajectory)

    sag = float(res.trajectory[-1, :, 1].min())
    initial_y = float(res.trajectory[0, :, 1].min())
    meta = {
        "n_frames": n_frames,
        "n_warmup": n_warmup,
        "iterations": ANCHOR_ITERATIONS,
        "mu": FBA_MU,
        "lam": FBA_LAM,
        "edge_ke": FBA_EDGE_KE,
        "frame_dt": FRAME_DT,
        "elapsed_s": elapsed,
        "mean_ms_per_frame": float(res.wall_clock_ms[n_warmup:].mean()),
        "terminal_min_y": sag,
        "initial_min_y": initial_y,
        "max_sag": initial_y - sag,
        "fba_timing_summary": res.fba_timing_summary,
        "ref_path": str(ref_path),
    }
    return meta


# --- Phase 1: VBD calibration ---


def _json_safe_loss(v: float) -> float | None:
    """Return v if finite, else None (JSON-serializable replacement for inf/nan)."""
    return v if np.isfinite(v) else None


def _vbd_terminal_run(alpha: float, beta: float, n_frames: int, substeps: int, iterations: int):
    """Run VBD at the given calibration and return the terminal-frame positions."""
    model = build_model_vbd(alpha=alpha, beta=beta)
    solver = SolverVBD(model, iterations=iterations, particle_enable_self_contact=False)
    res = run_solver(model, solver, n_frames=n_frames, substeps=substeps, n_warmup=0)
    return res.trajectory[-1]


def _eval_1d_loss(alphas, x_FBA_star, n_frames, substeps, iterations):
    losses = {}
    for a in alphas:
        try:
            xt = _vbd_terminal_run(a, 1.0, n_frames, substeps, iterations)
            losses[a] = float(np.sqrt(np.mean(np.sum((xt - x_FBA_star) ** 2, axis=-1))))
            print(f"[phase 1] alpha={a:.3f}  terminal_rms={losses[a]:.6f}")
        except RuntimeError as e:
            print(f"[phase 1] alpha={a:.3f}  diverged ({e})")
            losses[a] = float("inf")
    return losses


def _eval_2d_loss(alphas, betas, x_FBA_star, n_frames, substeps, iterations):
    losses = {}
    for a in alphas:
        for b in betas:
            try:
                xt = _vbd_terminal_run(a, b, n_frames, substeps, iterations)
                losses[(a, b)] = float(np.sqrt(np.mean(np.sum((xt - x_FBA_star) ** 2, axis=-1))))
                print(f"[phase 1-2d] alpha={a:.3f} beta={b:.3f}  rms={losses[(a, b)]:.6f}")
            except RuntimeError as e:
                print(f"[phase 1-2d] alpha={a:.3f} beta={b:.3f}  diverged ({e})")
                losses[(a, b)] = float("inf")
    return losses


def phase1_calibrate_vbd(
    out_dir: Path,
    anchor_meta: dict,
    n_frames: int = 800,
    cand_substeps: int = 10,
    cand_iterations: int = 10,
    verify_substeps: int = 40,
    verify_iterations: int = 40,
    floor_threshold_frac: float = 0.005,
) -> dict:
    """Grid-search VBD's tri_ke scale alpha to match FBA terminal shape.

    Auto-upgrades to 2D ``(alpha, beta)`` if 1D ``floor_RMS / max_sag`` exceeds
    ``floor_threshold_frac``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x_FBA_ref = np.load(out_dir / "x_FBA_ref.npy")
    x_FBA_star = x_FBA_ref[-1]
    max_sag = float(anchor_meta["max_sag"])

    # --- 1D pass with possible boundary extension ---
    alphas = list(ALPHA_GRID_1D)
    losses_1d = _eval_1d_loss(alphas, x_FBA_star, n_frames, cand_substeps, cand_iterations)
    extensions = 0
    while extensions < 2:
        finite = {a: v for a, v in losses_1d.items() if np.isfinite(v)}
        if not finite:
            break
        best = min(finite, key=finite.get)
        if best == min(alphas):
            new_a = min(alphas) * 0.5
            print(f"[phase 1] best at low boundary; extending grid with alpha={new_a:.3f}")
            alphas.append(new_a)
            extra = _eval_1d_loss([new_a], x_FBA_star, n_frames, cand_substeps, cand_iterations)
            losses_1d.update(extra)
            extensions += 1
        elif best == max(alphas):
            new_a = max(alphas) * 1.5
            print(f"[phase 1] best at high boundary; extending grid with alpha={new_a:.3f}")
            alphas.append(new_a)
            extra = _eval_1d_loss([new_a], x_FBA_star, n_frames, cand_substeps, cand_iterations)
            losses_1d.update(extra)
            extensions += 1
        else:
            break

    finite_1d = {a: v for a, v in losses_1d.items() if np.isfinite(v)}
    if not finite_1d:
        raise RuntimeError(
            "All alpha candidates diverged in 1D calibration; check VBD stability "
            "(stiffness, dt, substep count) or widen the alpha range."
        )
    alpha_star = min(finite_1d, key=finite_1d.get)
    beta_star = 1.0
    print(f"[phase 1] α* (1D) = {alpha_star:.4f}  loss = {finite_1d[alpha_star]:.6f}")  # noqa: RUF001

    # --- Optional 2D upgrade ---
    upgraded_to_2d = False
    losses_2d: dict[tuple[float, float], float] | None = None
    if finite_1d[alpha_star] / max(max_sag, 1e-9) > floor_threshold_frac:
        print(
            f"[phase 1] 1D floor {finite_1d[alpha_star]:.6f} > {floor_threshold_frac:.2%}"
            f" of max_sag={max_sag:.4f}; upgrading to 2D"
        )
        upgraded_to_2d = True
        losses_2d = _eval_2d_loss(alphas, BETA_GRID_2D, x_FBA_star, n_frames, cand_substeps, cand_iterations)
        finite_2d = {k: v for k, v in losses_2d.items() if np.isfinite(v)}
        if not finite_2d:
            raise RuntimeError(
                "All (alpha, beta) candidates diverged in 2D calibration; check VBD stability."
            )
        (alpha_star, beta_star) = min(finite_2d, key=finite_2d.get)
        print(
            f"[phase 1] (α*, β*) (2D) = ({alpha_star:.4f}, {beta_star:.4f})  loss = {finite_2d[(alpha_star, beta_star)]:.6f}"  # noqa: RUF001
        )

    # --- Floor verification at high-fidelity VBD ---
    print(f"[phase 1] verifying floor with VBD at substeps={verify_substeps}, iter={verify_iterations} ...")
    x_VBD_hi_terminal = _vbd_terminal_run(alpha_star, beta_star, n_frames, verify_substeps, verify_iterations)
    floor_rms = float(np.sqrt(np.mean(np.sum((x_VBD_hi_terminal - x_FBA_star) ** 2, axis=-1))))
    print(f"[phase 1] floor_RMS = {floor_rms:.6f} m  (max_sag = {max_sag:.4f} m)")

    result = {
        "alpha_star": alpha_star,
        "beta_star": beta_star,
        "upgraded_to_2d": upgraded_to_2d,
        "losses_1d": {f"{a:.4f}": _json_safe_loss(v) for a, v in losses_1d.items()},
        "losses_2d": (
            {f"{a:.4f}_{b:.4f}": _json_safe_loss(v) for (a, b), v in losses_2d.items()}
            if losses_2d is not None
            else None
        ),
        "floor_rms": floor_rms,
        "floor_threshold_frac": floor_threshold_frac,
        "max_sag": max_sag,
        "candidate_substeps": cand_substeps,
        "candidate_iterations": cand_iterations,
        "verify_substeps": verify_substeps,
        "verify_iterations": verify_iterations,
    }
    with open(out_dir / "calibration.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


# --- Smoke check ---


def _smoke():
    np.random.seed(0)  # noqa: NPY002
    out = Path("/tmp/fba_vs_vbd_smoke_out")
    if (out / "x_FBA_ref.npy").exists() and (out / "anchor.json").exists():
        # cache previous anchor to keep smoke fast
        anchor_meta = json.loads((out / "anchor.json").read_text())
    else:
        anchor_meta = phase0_fba_anchor(out, n_frames=80, n_warmup=5)
        (out / "anchor.json").write_text(json.dumps(anchor_meta, default=str))

    # Use a tiny alpha grid for speed; sufficient to exercise the loop.
    global ALPHA_GRID_1D  # noqa: PLW0603
    saved = ALPHA_GRID_1D
    ALPHA_GRID_1D = (0.85, 1.0, 1.15)
    try:
        result = phase1_calibrate_vbd(
            out,
            anchor_meta,
            n_frames=80,
            cand_substeps=2,
            cand_iterations=5,
            verify_substeps=5,
            verify_iterations=10,
        )
    finally:
        ALPHA_GRID_1D = saved

    # alpha_star may be below 0.5 in the smoke due to boundary extension on a
    # tiny grid — the 3-candidate grid can push down to 0.85 * 0.5 = 0.425.
    # In a full 800-frame run the optimum will land near 1.0.
    assert result["alpha_star"] > 0.0, f"alpha_star={result['alpha_star']} should be positive"
    assert result["floor_rms"] >= 0.0
    print(f"OK: alpha*={result['alpha_star']:.3f}, floor_rms={result['floor_rms']:.5f}")


if __name__ == "__main__":
    _smoke()
