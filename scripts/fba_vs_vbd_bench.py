# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""FBA vs VBD calibrated benchmark on a 32x32 hanging cloth.

See docs/superpowers/specs/2026-05-20-fba-vs-vbd-bench-design.md for the
design rationale.

Requires imageio[ffmpeg] (install once with ``uv pip install "imageio[ffmpeg]"``)
if you want the three_up.mp4 render. Pass --skip-video to avoid it.

Run::

    uv run python scripts/fba_vs_vbd_bench.py --out-dir scripts/fba_vs_vbd_bench_out
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
            raise RuntimeError("All (alpha, beta) candidates diverged in 2D calibration; check VBD stability.")
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


# --- Phase 2: Pareto sweep ---


@dataclass
class SweepConfig:
    solver: Literal["fba", "vbd"]
    iterations: int
    substeps: int  # only meaningful for VBD; always 1 for FBA
    alpha: float
    beta: float

    @property
    def label(self) -> str:
        if self.solver == "fba":
            return f"FBA iter={self.iterations}"
        return f"VBD {self.substeps}x{self.iterations}"


@dataclass
class SweepResult:
    config: SweepConfig
    mean_ms: float
    p50_ms: float
    p95_ms: float
    rms_over_time: np.ndarray  # (n_frames,)
    terminal_rms: float
    trajectory: np.ndarray | None = field(default=None, repr=False)
    fba_timing_summary: dict | None = None


def _rms_per_frame(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """sqrt(mean over vertices of squared L2) per frame; both inputs (T, V, 3)."""
    diff = traj - ref
    return np.sqrt(np.mean(np.sum(diff * diff, axis=-1), axis=-1))


def phase2_sweep(
    out_dir: Path,
    calibration: dict,
    n_frames: int = 800,
    n_warmup: int = 10,
    record_trajectories: bool = True,
) -> list[SweepResult]:
    """Run the FBA and VBD config sweeps; return per-config measurements."""
    x_FBA_ref = np.load(out_dir / "x_FBA_ref.npy")
    assert x_FBA_ref.shape[0] == n_frames, (
        f"x_FBA_ref has {x_FBA_ref.shape[0]} frames but sweep needs {n_frames}; re-run Phase 0 with matching n_frames"
    )

    alpha_star = calibration["alpha_star"]
    beta_star = calibration["beta_star"]
    sweep: list[SweepResult] = []

    # FBA sweep
    for it in FBA_SWEEP_ITERS:
        print(f"[phase 2] FBA iter={it}")
        model = build_model_fba()
        solver = SolverFBA(
            model,
            iterations=it,
            stretching_model="neohookean",
            mu=FBA_MU,
            lam=FBA_LAM,
            enable_perf_timing=True,
        )
        res = run_solver(model, solver, n_frames=n_frames, substeps=1, n_warmup=n_warmup)
        rms = _rms_per_frame(res.trajectory, x_FBA_ref)
        sweep.append(
            SweepResult(
                config=SweepConfig("fba", iterations=it, substeps=1, alpha=1.0, beta=1.0),
                mean_ms=float(res.wall_clock_ms[n_warmup:].mean()),
                p50_ms=float(np.percentile(res.wall_clock_ms[n_warmup:], 50)),
                p95_ms=float(np.percentile(res.wall_clock_ms[n_warmup:], 95)),
                rms_over_time=rms,
                terminal_rms=float(rms[-1]),
                trajectory=res.trajectory if record_trajectories else None,
                fba_timing_summary=res.fba_timing_summary,
            )
        )

    # VBD sweep
    for sub, it in VBD_SWEEP_CONFIGS:
        print(f"[phase 2] VBD substeps={sub} iter={it}  (α={alpha_star:.3f}, β={beta_star:.3f})")  # noqa: RUF001
        model = build_model_vbd(alpha=alpha_star, beta=beta_star)
        solver = SolverVBD(model, iterations=it, particle_enable_self_contact=False)
        res = run_solver(model, solver, n_frames=n_frames, substeps=sub, n_warmup=n_warmup)
        rms = _rms_per_frame(res.trajectory, x_FBA_ref)
        sweep.append(
            SweepResult(
                config=SweepConfig("vbd", iterations=it, substeps=sub, alpha=alpha_star, beta=beta_star),
                mean_ms=float(res.wall_clock_ms[n_warmup:].mean()),
                p50_ms=float(np.percentile(res.wall_clock_ms[n_warmup:], 50)),
                p95_ms=float(np.percentile(res.wall_clock_ms[n_warmup:], 95)),
                rms_over_time=rms,
                terminal_rms=float(rms[-1]),
                trajectory=res.trajectory if record_trajectories else None,
            )
        )

    # Serialize (without trajectories — those stay in memory for video rendering).
    serial = []
    for s in sweep:
        serial.append(
            {
                "label": s.config.label,
                "solver": s.config.solver,
                "iterations": s.config.iterations,
                "substeps": s.config.substeps,
                "alpha": s.config.alpha,
                "beta": s.config.beta,
                "mean_ms": s.mean_ms,
                "p50_ms": s.p50_ms,
                "p95_ms": s.p95_ms,
                "terminal_rms": s.terminal_rms,
                "rms_over_time": s.rms_over_time.tolist(),
                "fba_timing_summary": s.fba_timing_summary,
            }
        )
    with open(out_dir / "results.json", "w") as fh:
        json.dump({"sweep": serial, "calibration": calibration}, fh, indent=2)
    return sweep


# --- Phase 3a: Static plots ---


def _import_mpl():
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    return plt


def plot_calibration(out_dir: Path, calibration: dict) -> None:
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    if calibration["upgraded_to_2d"] and calibration["losses_2d"] is not None:
        # 2D heatmap of (alpha, beta) -> loss
        keys = list(calibration["losses_2d"].keys())
        alphas = sorted({float(k.split("_")[0]) for k in keys})
        betas = sorted({float(k.split("_")[1]) for k in keys})
        grid = np.full((len(betas), len(alphas)), np.nan)
        for k, v in calibration["losses_2d"].items():
            a, b = (float(x) for x in k.split("_"))
            grid[betas.index(b), alphas.index(a)] = v if v is not None else np.nan
        im = ax.imshow(grid, origin="lower",
                       extent=[min(alphas), max(alphas), min(betas), max(betas)],
                       aspect="auto", cmap="viridis")
        ax.plot(calibration["alpha_star"], calibration["beta_star"],
                marker="*", ms=18, color="red", label=f"α*={calibration['alpha_star']:.3f}, β*={calibration['beta_star']:.3f}")
        plt.colorbar(im, ax=ax, label="terminal RMS (m)")
        ax.set_xlabel("alpha (tri_ke scale)")
        ax.set_ylabel("beta (edge_ke scale)")
        ax.set_title(f"2D calibration grid (floor = {calibration['floor_rms']:.4e} m)")
    else:
        # 1D curve over alpha
        items = sorted(((float(k), v) for k, v in calibration["losses_1d"].items() if v is not None),
                       key=lambda kv: kv[0])
        xs = [a for a, _ in items]
        ys = [v for _, v in items]
        ax.plot(xs, ys, marker="o")
        ax.axvline(calibration["alpha_star"], color="red", linestyle="--",
                   label=f"α*={calibration['alpha_star']:.3f}")
        ax.axhline(calibration["floor_rms"], color="gray", linestyle=":",
                   label=f"floor={calibration['floor_rms']:.4e} m")
        ax.set_xlabel("alpha (tri_ke scale)")
        ax.set_ylabel("terminal RMS vs FBA anchor (m)")
        ax.set_title(f"1D calibration (max_sag = {calibration['max_sag']:.4f} m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=120)
    plt.close(fig)


def plot_pareto(out_dir: Path, sweep: list[SweepResult], calibration: dict) -> None:
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in sweep:
        color = "tab:blue" if s.config.solver == "fba" else "tab:orange"
        ax.scatter(s.mean_ms, s.terminal_rms, color=color, s=60)
        ax.annotate(s.config.label, (s.mean_ms, s.terminal_rms),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(calibration["floor_rms"], color="gray", linestyle=":",
               label=f"floor = {calibration['floor_rms']:.4e} m")
    ax.set_xlabel("mean wall-clock per rendered frame (ms)")
    ax.set_ylabel("terminal RMS vs FBA anchor (m)")
    from matplotlib.lines import Line2D  # noqa: PLC0415

    solver_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markersize=8, label="FBA", linestyle=""),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange",
               markersize=8, label="VBD", linestyle=""),
    ]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("FBA vs VBD: wall-clock vs accuracy (lower-left is better)")
    floor_line = ax.lines[0] if ax.lines else None
    handles = solver_handles + ([floor_line] if floor_line is not None else [])
    ax.legend(handles=handles, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=120)
    plt.close(fig)


def plot_rms_over_time(out_dir: Path, sweep: list[SweepResult]) -> None:
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in sweep:
        color = "tab:blue" if s.config.solver == "fba" else "tab:orange"
        ax.plot(s.rms_over_time, color=color, alpha=0.75, label=s.config.label)
    ax.set_xlabel("frame")
    ax.set_ylabel("RMS vs FBA anchor (m)")
    ax.set_yscale("log")
    ax.set_title("Per-frame RMS — FBA (blue) and VBD (orange)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rms_over_time.png", dpi=120)
    plt.close(fig)


def plot_error_heatmaps(out_dir: Path, sweep: list[SweepResult], x_FBA_ref: np.ndarray) -> None:
    plt = _import_mpl()
    fba_entries = [s for s in sweep if s.config.solver == "fba"]
    vbd_entries = [s for s in sweep if s.config.solver == "vbd"]
    if not fba_entries or not vbd_entries:
        raise RuntimeError(
            "plot_error_heatmaps requires both FBA and VBD entries in sweep; "
            f"got {len(fba_entries)} FBA, {len(vbd_entries)} VBD"
        )
    fba_best = min(fba_entries, key=lambda s: s.terminal_rms)
    vbd_best = min(vbd_entries, key=lambda s: s.terminal_rms)
    if fba_best.trajectory is None or vbd_best.trajectory is None:
        raise RuntimeError("error heatmaps require record_trajectories=True in phase2_sweep")

    err_fba = np.linalg.norm(fba_best.trajectory[-1] - x_FBA_ref[-1], axis=-1)
    err_vbd = np.linalg.norm(vbd_best.trajectory[-1] - x_FBA_ref[-1], axis=-1)
    vmax = max(err_fba.max(), err_vbd.max(), 1e-12)

    for label, x, err, fname in [
        (fba_best.config.label, fba_best.trajectory[-1], err_fba, "error_heatmap_fba.png"),
        (vbd_best.config.label, vbd_best.trajectory[-1], err_vbd, "error_heatmap_vbd.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(x[:, 0], x[:, 1], c=err, s=8, cmap="magma", vmin=0, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="|Δx| vs FBA anchor (m)")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Terminal error — {label}")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)


# --- Phase 3b: Three-up video render ---


def _render_one_solver(
    model,
    trajectory: np.ndarray,
    width: int,
    height: int,
):
    """Replay a recorded trajectory through a headless ViewerGL and yield frames.

    Yields HxWx3 uint8 numpy arrays. The viewer is closed when the generator
    is exhausted.
    """
    import newton.viewer  # noqa: PLC0415

    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    viewer.set_model(model)
    state = model.state()
    frame_buf = wp.empty(shape=(height, width, 3), dtype=wp.uint8, device=viewer.device)

    try:
        for t in range(trajectory.shape[0]):
            state.particle_q.assign(trajectory[t])
            viewer.begin_frame(float(t) * FRAME_DT)
            viewer.log_state(state)
            viewer.end_frame()
            viewer.get_frame(target_image=frame_buf)
            yield frame_buf.numpy().copy()
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


def render_three_up(
    out_dir: Path,
    sweep: list[SweepResult],
    x_FBA_ref: np.ndarray,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
) -> Path:
    """Render Reference | FBA-best | VBD-best as a horizontally-tiled MP4."""
    import imageio.v2 as imageio  # noqa: PLC0415

    fba_best = min((s for s in sweep if s.config.solver == "fba"), key=lambda s: s.terminal_rms)
    vbd_best = min((s for s in sweep if s.config.solver == "vbd"), key=lambda s: s.terminal_rms)
    assert fba_best.trajectory is not None and vbd_best.trajectory is not None, (
        "three-up render needs trajectories recorded in Phase 2"
    )
    n = x_FBA_ref.shape[0]
    assert fba_best.trajectory.shape[0] == n and vbd_best.trajectory.shape[0] == n

    ref_model = build_model_fba()
    fba_model = build_model_fba()
    vbd_model = build_model_vbd(alpha=vbd_best.config.alpha, beta=vbd_best.config.beta)

    out_path = out_dir / "three_up.mp4"
    print(f"[phase 3] rendering three_up.mp4 ({n} frames at {width}x{height}, {fps} fps)")

    gen_ref = _render_one_solver(ref_model, x_FBA_ref, width, height)
    gen_fba = _render_one_solver(fba_model, fba_best.trajectory, width, height)
    gen_vbd = _render_one_solver(vbd_model, vbd_best.trajectory, width, height)

    with imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for i, (a, b, c) in enumerate(zip(gen_ref, gen_fba, gen_vbd)):
            combined = np.hstack([a, b, c])
            writer.append_data(combined)
            if i % 100 == 0:
                print(f"  frame {i}/{n}")
    return out_path


# --- Phase 3c: Markdown report ---


def write_report(out_dir: Path, anchor_meta: dict, calibration: dict, sweep: list[SweepResult]) -> Path:
    """Write report.md summarising calibration, sweep table, plots, and anchor metadata.

    Returns the path to the written file.
    """
    path = out_dir / "report.md"
    lines: list[str] = []
    lines.append("# FBA vs VBD — 32x32 hanging cloth\n")
    lines.append("_Spec: docs/superpowers/specs/2026-05-20-fba-vs-vbd-bench-design.md_\n")
    lines.append("\n## Calibration\n")
    lines.append(f"- `α*` (tri_ke scale on VBD): **{calibration['alpha_star']:.4f}**")
    if not calibration["upgraded_to_2d"]:
        lines.append(
            f"- `β*` (edge_ke scale on VBD): **{calibration['beta_star']:.4f}**  (1D pass only)"
        )
    else:
        lines.append(
            f"- `β*`: **{calibration['beta_star']:.4f}**  (2D upgrade triggered)"
        )
    lines.append(
        f"- Floor RMS (high-fidelity VBD vs FBA anchor): **{calibration['floor_rms']:.4e} m**"
    )
    lines.append(f"- Max sag of FBA anchor: {calibration['max_sag']:.4f} m")
    lines.append(
        f"- Floor / max_sag: {calibration['floor_rms'] / max(calibration['max_sag'], 1e-9):.2%}"
    )
    lines.append("")
    lines.append("![calibration](calibration.png)\n")

    lines.append("## Sweep results\n")
    lines.append("| Config | mean ms/frame | p50 | p95 | terminal RMS (m) |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in sorted(sweep, key=lambda r: (r.config.solver, r.mean_ms)):
        lines.append(
            f"| {s.config.label} | {s.mean_ms:.3f} | {s.p50_ms:.3f} | {s.p95_ms:.3f} | {s.terminal_rms:.4e} |"
        )
    lines.append("")
    lines.append("![pareto](pareto.png)\n")
    lines.append("![rms_over_time](rms_over_time.png)\n")

    lines.append("## Behavior — best-config terminal frame\n")
    fba_entries = [s for s in sweep if s.config.solver == "fba"]
    vbd_entries = [s for s in sweep if s.config.solver == "vbd"]
    if not fba_entries or not vbd_entries:
        raise RuntimeError(
            "write_report requires both FBA and VBD entries in sweep; "
            f"got {len(fba_entries)} FBA, {len(vbd_entries)} VBD"
        )
    fba_best = min(fba_entries, key=lambda s: s.terminal_rms)
    vbd_best = min(vbd_entries, key=lambda s: s.terminal_rms)
    lines.append(
        f"FBA-best: **{fba_best.config.label}** ({fba_best.mean_ms:.3f} ms, RMS {fba_best.terminal_rms:.4e} m)\n"
    )
    lines.append(
        f"VBD-best: **{vbd_best.config.label}** ({vbd_best.mean_ms:.3f} ms, RMS {vbd_best.terminal_rms:.4e} m)\n"
    )
    lines.append("![heatmap FBA](error_heatmap_fba.png)  ![heatmap VBD](error_heatmap_vbd.png)\n")
    lines.append("Side-by-side video: [`three_up.mp4`](three_up.mp4)\n")

    lines.append("## Anchor metadata\n")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "iterations": anchor_meta["iterations"],
                "mu": anchor_meta["mu"],
                "lam": anchor_meta["lam"],
                "edge_ke": anchor_meta["edge_ke"],
                "frame_dt": anchor_meta["frame_dt"],
                "n_frames": anchor_meta["n_frames"],
                "max_sag": anchor_meta["max_sag"],
                "mean_ms_per_frame": anchor_meta["mean_ms_per_frame"],
            },
            indent=2,
        )
    )
    lines.append("```\n")

    path.write_text("\n".join(lines))
    return path


# --- Main entry ---


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FBA vs VBD calibrated cloth benchmark")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fba_vs_vbd_bench_out",
    )
    p.add_argument("--frames", type=int, default=800,
                   help="frames per phase (Phase 0/1/2 all share this)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--skip-phase0", action="store_true",
                   help="reuse x_FBA_ref.npy + anchor.json if present")
    p.add_argument("--skip-phase1", action="store_true",
                   help="reuse calibration.json if present")
    p.add_argument("--skip-video", action="store_true",
                   help="skip three_up.mp4 render (still emits all PNGs and report)")
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--video-height", type=int, default=480)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    np.random.seed(0)  # noqa: NPY002
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    anchor_json = out / "anchor.json"
    if args.skip_phase0 and anchor_json.exists() and (out / "x_FBA_ref.npy").exists():
        print("[phase 0] reusing cached anchor")
        anchor_meta = json.loads(anchor_json.read_text())
        if anchor_meta["n_frames"] != args.frames:
            raise SystemExit(
                f"cached anchor has n_frames={anchor_meta['n_frames']} but --frames={args.frames}; "
                "rerun without --skip-phase0"
            )
        ref = np.load(out / "x_FBA_ref.npy")
        if ref.shape != (args.frames, (GRID_DIM + 1) ** 2, 3):
            raise SystemExit(
                f"cached x_FBA_ref.npy has shape {ref.shape}; expected "
                f"({args.frames}, {(GRID_DIM + 1) ** 2}, 3); rerun without --skip-phase0"
            )
        del ref  # don't keep two copies in memory
    else:
        anchor_meta = phase0_fba_anchor(out, n_frames=args.frames, n_warmup=args.warmup)
        anchor_json.write_text(json.dumps(anchor_meta, default=str))

    calib_json = out / "calibration.json"
    if args.skip_phase1 and calib_json.exists():
        print("[phase 1] reusing cached calibration")
        calibration = json.loads(calib_json.read_text())
        if not (out / "x_FBA_ref.npy").exists():
            raise SystemExit(
                "--skip-phase1 set but x_FBA_ref.npy is missing; rerun without --skip-phase0 (or --skip-phase1) to regenerate"
            )
    else:
        calibration = phase1_calibrate_vbd(out, anchor_meta, n_frames=args.frames)

    sweep = phase2_sweep(out, calibration, n_frames=args.frames, n_warmup=args.warmup,
                         record_trajectories=True)

    plot_calibration(out, calibration)
    plot_pareto(out, sweep, calibration)
    plot_rms_over_time(out, sweep)
    x_FBA_ref = np.load(out / "x_FBA_ref.npy")
    plot_error_heatmaps(out, sweep, x_FBA_ref)

    if not args.skip_video:
        render_three_up(out, sweep, x_FBA_ref,
                        width=args.video_width, height=args.video_height, fps=50)
    else:
        print("[phase 3] skipping three_up.mp4 (--skip-video)")

    report_path = write_report(out, anchor_meta, calibration, sweep)
    print(f"\nDone. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
