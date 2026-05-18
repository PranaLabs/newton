# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Side-by-side per-PD-iter perf comparison: Newton FBA vs RealSim CudaTests.

For each of the four flagship CudaTests demos --

* ``TwistingBarNH``      (NeoHookean, no contacts)
* ``StretchingCloth``    (NeoHookean cloth, no contacts)
* ``PullingWooper``      (NeoHookean tet, unilateral contacts)
* ``SqueezingBall``      (NeoHookean tet, frictional contacts + rolling cyls)

-- this script runs both Newton FBA (with :class:`SolverFBA` instantiated with
``enable_perf_timing=True``) and RealSim offline, extracts matched-granularity
per-PD-iter mean timing, and writes a markdown table to::

    docs/superpowers/specs/2026-05-18-fba-realsim-perf-comparison.md

The FBA side reuses each demo's existing ``run()`` helper but instantiates a
local copy of the simulator loop so we can flip ``enable_perf_timing=True``
and query :meth:`SolverFBA.get_timing_summary` after the run finishes. To keep
the harness self-contained, we re-implement a minimal loop per demo rather
than refactoring the demo scripts.

Usage:
    uv run python scripts/perf_compare_all_demos.py            # full run
    uv run python scripts/perf_compare_all_demos.py --frames 200  # quick
    uv run python scripts/perf_compare_all_demos.py --skip-realsim  # FBA only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "docs/superpowers/specs/2026-05-18-fba-realsim-perf-comparison.md"
JSON_DUMP = REPO_ROOT / "scripts/perf_compare_all_demos.json"


@dataclass
class DemoResult:
    demo: str
    fba: dict = field(default_factory=dict)
    realsim: dict = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# FBA side. Each helper runs ``n_frames`` steps with ``enable_perf_timing=True``
# and returns the timing summary plus wall-clock metadata.
# ---------------------------------------------------------------------------


def _run_fba_twisting_bar_nh(n_frames: int, warmup: int) -> dict:
    """Replicate ``fba_twisting_bar_cudatests.py`` minimally with timing on."""
    from newton.solvers import SolverFBA  # noqa: PLC0415
    from scripts.fba_twisting_bar_cudatests import (  # noqa: PLC0415
        DT,
        LAM,
        MAX_ANGLE,
        MESH_PATH_PER_ENERGY,
        MU,
        PD_ITERATIONS,
        PIN_AVEL,
        build_model,
        load_medit_mesh,
        rot_y,
    )

    verts, tets = load_medit_mesh(MESH_PATH_PER_ENERGY["neohookean"])
    model, top_pins, bot_pins = build_model(verts, tets, MU, LAM)
    verts_init = verts.copy().astype(np.float64)
    center = np.zeros(3, dtype=np.float64)

    t_setup0 = time.perf_counter()
    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        pin_stiffness=1e12,
        stretching_model="neohookean",
        mu=MU,
        lam=LAM,
        enable_perf_timing=True,
    )
    s_in = model.state()
    s_out = model.state()
    s_in.clear_forces()
    solver.step(s_in, s_out, None, None, DT)
    if wp.is_cuda_available():
        wp.synchronize()
    setup_ms = (time.perf_counter() - t_setup0) * 1000.0

    s_in = model.state()
    s_out = model.state()
    step_times = []
    for f in range(n_frames):
        top_angle = min(PIN_AVEL * (f + 1) * DT, MAX_ANGLE)
        bot_angle = max(-PIN_AVEL * (f + 1) * DT, -MAX_ANGLE)
        R_top = rot_y(top_angle)
        R_bot = rot_y(bot_angle)
        x_ref = model.particle_q.numpy().copy().astype(np.float64)
        v_top = verts_init[top_pins] - center
        x_ref[top_pins] = (v_top @ R_top.T) + center
        v_bot = verts_init[bot_pins] - center
        x_ref[bot_pins] = (v_bot @ R_bot.T) + center
        solver.set_pin_targets(x_ref.astype(np.float32))
        s_in.clear_forces()
        if wp.is_cuda_available():
            wp.synchronize()
        if f == warmup:
            solver.reset_timing()
        t0 = time.perf_counter()
        solver.step(s_in, s_out, None, None, DT)
        if wp.is_cuda_available():
            wp.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        s_in, s_out = s_out, s_in

    summary = solver.get_timing_summary(last_n_steps=None)
    return _pack_fba(summary, step_times, setup_ms, warmup, n_frames)


def _run_fba_stretching_cloth(n_frames: int, warmup: int) -> dict:
    from newton.solvers import SolverFBA  # noqa: PLC0415
    from scripts.fba_demo3_stretching_cloth import (  # noqa: PLC0415
        DT,
        NSN_ITERATIONS,
        PD_ITERATIONS,
        PIN0_DIR,
        PIN1_DIR,
        PIN_MAXLENGTH,
        PIN_VEL,
        build_model,
    )

    model, verts_world, pin0_idx, pin1_idx, mu_lame, lam = build_model()

    t_setup0 = time.perf_counter()
    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=False,
        stretching_model="neohookean",
        nh_solver="lbfgs",
        mu=mu_lame,
        lam=lam,
        pin_stiffness=1.0e10,
        enable_perf_timing=True,
    )
    s_in = model.state()
    s_out = model.state()
    s_in.clear_forces()
    pin0_init = verts_world[pin0_idx].copy().astype(np.float32)
    pin1_init = verts_world[pin1_idx].copy().astype(np.float32)
    targets = verts_world.copy().astype(np.float32)
    solver.step(s_in, s_out, None, None, DT)
    if wp.is_cuda_available():
        wp.synchronize()
    setup_ms = (time.perf_counter() - t_setup0) * 1000.0

    s_in = model.state()
    s_out = model.state()
    step_times = []
    pulled = 0.0
    for f in range(n_frames):
        delta = PIN_VEL * DT
        if pulled + delta > PIN_MAXLENGTH:
            delta = max(0.0, PIN_MAXLENGTH - pulled)
        pulled += delta
        targets[pin0_idx] = pin0_init + (PIN0_DIR * pulled).astype(np.float32)
        targets[pin1_idx] = pin1_init + (PIN1_DIR * pulled).astype(np.float32)
        solver.set_pin_targets(targets)
        if wp.is_cuda_available():
            wp.synchronize()
        if f == warmup:
            solver.reset_timing()
        t0 = time.perf_counter()
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, DT)
        if wp.is_cuda_available():
            wp.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        s_in, s_out = s_out, s_in
    summary = solver.get_timing_summary(last_n_steps=None)
    return _pack_fba(summary, step_times, setup_ms, warmup, n_frames)


def _run_fba_pulling_wooper(n_frames: int, warmup: int) -> dict:
    from newton.solvers import SolverFBA  # noqa: PLC0415
    from scripts.fba_demo4_pulling_wooper import (  # noqa: PLC0415
        DT,
        NSN_ITERATIONS,
        PD_ITERATIONS,
        PIN_DIR,
        PIN_MAXLENGTH,
        PIN_VEL,
        build_model,
    )

    model, pipeline, contacts, verts_world, pin_idx, mu, lam = build_model()

    t_setup0 = time.perf_counter()
    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=False,
        stretching_model="neohookean",
        mu=mu,
        lam=lam,
        lambda_cap=1.0e12,
        pin_stiffness=1.0e10,
        enable_perf_timing=True,
    )
    s_in = model.state()
    s_out = model.state()
    s_in.clear_forces()
    pipeline.collide(s_in, contacts)
    solver.step(s_in, s_out, None, contacts, DT)
    if wp.is_cuda_available():
        wp.synchronize()
    setup_ms = (time.perf_counter() - t_setup0) * 1000.0

    s_in = model.state()
    s_out = model.state()
    pin_init = verts_world[pin_idx].copy()
    targets = verts_world.copy().astype(np.float32)
    pulled = 0.0
    step_times = []
    for f in range(n_frames):
        delta = PIN_VEL * DT
        if pulled + delta > PIN_MAXLENGTH:
            delta = max(0.0, PIN_MAXLENGTH - pulled)
        pulled += delta
        targets[pin_idx] = pin_init + (PIN_DIR * pulled).astype(np.float32)
        solver.set_pin_targets(targets)
        if wp.is_cuda_available():
            wp.synchronize()
        if f == warmup:
            solver.reset_timing()
        t0 = time.perf_counter()
        s_in.clear_forces()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, DT)
        if wp.is_cuda_available():
            wp.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        s_in, s_out = s_out, s_in
    summary = solver.get_timing_summary(last_n_steps=None)
    return _pack_fba(summary, step_times, setup_ms, warmup, n_frames)


def _run_fba_squeezing_ball(n_frames: int, warmup: int) -> dict:
    from newton.solvers import SolverFBA  # noqa: PLC0415
    from scripts.fba_demo5_squeezing_ball import (  # noqa: PLC0415
        CYLINDERS,
        DT,
        FRICTION_MU,
        NSN_ITERATIONS,
        PD_ITERATIONS,
        build_model,
    )

    model, pipeline, contacts, _verts_world, mu, lam = build_model()
    shape_omega = {i: CYLINDERS[i][2] for i in range(4)}
    n_max_contacts = model.particle_count
    mu_override = np.full(n_max_contacts, FRICTION_MU, dtype=np.float64)

    t_setup0 = time.perf_counter()
    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=True,
        stretching_model="neohookean",
        mu=mu,
        lam=lam,
        mu_per_pair_override=mu_override,
        shape_angular_velocity=shape_omega,
        lambda_cap=1.0e12,
        pin_stiffness=1.0e10,
        enable_perf_timing=True,
    )
    s_in = model.state()
    s_out = model.state()
    s_in.clear_forces()
    pipeline.collide(s_in, contacts)
    solver.step(s_in, s_out, None, contacts, DT)
    if wp.is_cuda_available():
        wp.synchronize()
    setup_ms = (time.perf_counter() - t_setup0) * 1000.0

    s_in = model.state()
    s_out = model.state()
    step_times = []
    for f in range(n_frames):
        if wp.is_cuda_available():
            wp.synchronize()
        if f == warmup:
            solver.reset_timing()
        t0 = time.perf_counter()
        s_in.clear_forces()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, DT)
        if wp.is_cuda_available():
            wp.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        s_in, s_out = s_out, s_in
    summary = solver.get_timing_summary(last_n_steps=None)
    return _pack_fba(summary, step_times, setup_ms, warmup, n_frames)


def _pack_fba(summary: dict, step_times_ms: list[float], setup_ms: float, warmup: int, n_frames: int) -> dict:
    measured = step_times_ms[warmup:]
    arr = np.asarray(measured)
    return {
        "local_ms": summary["local_ms"],
        "linear_solve_ms": summary["linear_solve_ms"],
        "schur_ms": summary["schur_ms"],
        "nsn_inner_ms": summary["nsn_inner_ms"],
        "n_pd_iter_samples": int(summary["n_samples"]),
        "setup_ms": float(setup_ms),
        "step_mean_ms": float(arr.mean()) if arr.size else None,
        "step_median_ms": float(np.median(arr)) if arr.size else None,
        "step_p95_ms": float(np.percentile(arr, 95)) if arr.size else None,
        "n_frames_total": int(n_frames),
        "n_frames_measured": int(arr.size),
    }


FBA_RUNNERS = {
    "TwistingBarNH": _run_fba_twisting_bar_nh,
    "StretchingCloth": _run_fba_stretching_cloth,
    "PullingWooper": _run_fba_pulling_wooper,
    "SqueezingBall": _run_fba_squeezing_ball,
}


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def _fmt(v: float | None, fmt: str = "{:.3f}") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return fmt.format(v)


def _ratio(f: float | None, r: float | None) -> str:
    if f is None or r is None or r == 0:
        return "n/a"
    return f"{f / r:.2f}x"


def format_markdown(results: list[DemoResult]) -> str:
    """Build the markdown report from the per-demo results."""
    lines: list[str] = []
    lines.append("# FBA vs RealSim Per-PD-Iter Perf Comparison")
    lines.append("")
    lines.append(
        "Baseline (pre-fp64-SVD-lift, branch `ziqiu/fba-solver-design`).  "
        "FBA timings are gathered via :meth:`SolverFBA.get_timing_summary` with "
        "``enable_perf_timing=True``; RealSim timings are parsed from "
        "``LocalGlobalSolver::printTimer`` blocks (one block per "
        "``scene.json::timer`` frames, default 50)."
    )
    lines.append("")
    lines.append(
        "All values are means in **milliseconds per PD outer iter** "
        "(per-iter granularity), except where the row label says "
        "otherwise. Ratio = FBA / RealSim; values > 1.0 mean FBA is slower."
    )
    lines.append("")
    lines.append(
        "**NSN-Schur PCR matvec path:** Warp ``tile_matmul`` (block-per-row "
        "tile reduction), wrapped in a CUDA graph that replays one full PCR "
        "iter per ``cuGraphLaunch`` (Perf #5).  An earlier optional cuBLAS "
        "DGEMV path (Perf #4 via cupy) was removed in Perf #6 because cuBLAS "
        "trips ``cudaErrorStreamCaptureImplicit`` during graph capture, and "
        "the captured ``tile_matmul`` path wins by a large margin "
        "(Demo 5 step_mean ~43 ms graph + tile_matmul vs ~99 ms cuBLAS + eager)."
    )
    lines.append("")
    lines.append("## Per-component breakdown")
    lines.append("")
    lines.append("| Demo | Component | FBA (ms) | RealSim (ms) | Ratio |")
    lines.append("|---|---|---|---|---|")
    for res in results:
        if res.error:
            lines.append(f"| {res.demo} | ERROR | | | {res.error} |")
            continue
        f, r = res.fba, res.realsim
        rows = [
            ("Local (energy projection)", f.get("local_ms"), r.get("local_ms")),
            ("Linear solve (A^-1 b)", f.get("linear_solve_ms"), r.get("linear_solve_ms")),
            ("Schur W build", f.get("schur_ms"), r.get("schur_ms")),
            ("NSN inner (build+cstsolve+correction)", f.get("nsn_inner_ms"), r.get("total_global_ms")),
        ]
        for i, (label, fv, rv) in enumerate(rows):
            demo_cell = res.demo if i == 0 else ""
            lines.append(f"| {demo_cell} | {label} | {_fmt(fv)} | {_fmt(rv)} | {_ratio(fv, rv)} |")
    lines.append("")
    lines.append("## NSN sub-phases (RealSim only)")
    lines.append("")
    lines.append(
        "FBA's GPU NSN driver does a single fused kernel sequence "
        "(``build_schur`` was already split out above; ``solve+clamp`` "
        "and ``apply correction`` are not separately bracketed -- they're "
        "lumped into ``NSN inner``). RealSim reports the split, so we "
        "show it here for reference."
    )
    lines.append("")
    lines.append(
        "| Demo | n_contacts (mean) | PCR iters (mean) | RS Assembly (ms) | RS Cst solve (ms) | RS Correction (ms) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for res in results:
        if res.error:
            continue
        r = res.realsim
        lines.append(
            f"| {res.demo} | {_fmt(r.get('n_constraints_mean'), '{:.1f}')} | "
            f"{_fmt(r.get('cst_solve_iter_mean'), '{:.1f}')} | "
            f"{_fmt(r.get('build_ms'))} | {_fmt(r.get('cst_solve_ms'))} | "
            f"{_fmt(r.get('correction_ms'))} |"
        )
    lines.append("")
    lines.append("## Frame-level totals")
    lines.append("")
    lines.append(
        "| Demo | FBA step mean (ms) | FBA p95 (ms) | RS frame_no_cd (ms) | "
        "RS total step (ms) | FBA setup (ms) | Frames (FBA measured / RS counted) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for res in results:
        if res.error:
            continue
        f, r = res.fba, res.realsim
        lines.append(
            f"| {res.demo} | {_fmt(f.get('step_mean_ms'), '{:.2f}')} | "
            f"{_fmt(f.get('step_p95_ms'), '{:.2f}')} | "
            f"{_fmt(r.get('frame_no_cd_ms'), '{:.2f}')} | "
            f"{_fmt(r.get('total_step_ms'), '{:.2f}')} | "
            f"{_fmt(f.get('setup_ms'), '{:.1f}')} | "
            f"{f.get('n_frames_measured', '?')} / {r.get('n_frames', '?')} |"
        )
    lines.append("")
    lines.append("## Identified bottlenecks (largest contributor per demo)")
    lines.append("")
    for res in results:
        if res.error:
            lines.append(f"- **{res.demo}**: failed -- {res.error}")
            continue
        bottleneck = _identify_bottleneck(res.fba)
        gap_notes = _classify_gaps(res.fba, res.realsim)
        lines.append(f"- **{res.demo}**: dominant FBA cost = {bottleneck}.")
        if gap_notes:
            lines.append(f"  - Parity / gap notes: {gap_notes}")
    lines.append("")
    lines.append("## Raw JSON")
    lines.append("")
    lines.append(f"See `{JSON_DUMP.relative_to(REPO_ROOT)}` for the full per-demo dump.")
    return "\n".join(lines) + "\n"


def _identify_bottleneck(fba: dict) -> str:
    entries = [
        ("Local", fba.get("local_ms")),
        ("Linear solve", fba.get("linear_solve_ms")),
        ("Schur W", fba.get("schur_ms")),
        ("NSN inner", fba.get("nsn_inner_ms")),
    ]
    finite = [(k, v) for k, v in entries if v is not None]
    if not finite:
        return "n/a"
    finite.sort(key=lambda kv: kv[1], reverse=True)
    k, v = finite[0]
    return f"**{k}** = {v:.3f} ms/iter"


def _classify_gaps(fba: dict, rs: dict) -> str:
    """Tag components with >5x gap and components at parity."""
    notes: list[str] = []
    pairs = [
        ("Local", fba.get("local_ms"), rs.get("local_ms")),
        ("Linear solve", fba.get("linear_solve_ms"), rs.get("linear_solve_ms")),
        ("Schur W", fba.get("schur_ms"), rs.get("schur_ms")),
        ("NSN inner", fba.get("nsn_inner_ms"), rs.get("total_global_ms")),
    ]
    for label, fv, rv in pairs:
        if fv is None or rv is None or rv == 0:
            continue
        ratio = fv / rv
        if ratio >= 5.0:
            notes.append(f"{label} {ratio:.1f}x slower")
        elif ratio <= 0.2:
            notes.append(f"{label} {1.0 / ratio:.1f}x faster")
        elif 0.8 <= ratio <= 1.25:
            notes.append(f"{label} at parity ({ratio:.2f}x)")
    return "; ".join(notes) if notes else "no >5x gaps detected"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=200, help="Per-demo step count (default 200)")
    ap.add_argument("--warmup", type=int, default=20, help="Warm-up frames excluded from timing (default 20)")
    ap.add_argument(
        "--demos", nargs="+", default=None, choices=sorted(FBA_RUNNERS), help="Subset of demos (default: all four)"
    )
    ap.add_argument(
        "--skip-realsim", action="store_true", help="Skip RealSim runs (e.g. when the binary is unavailable)"
    )
    ap.add_argument("--skip-fba", action="store_true", help="Skip FBA runs (e.g. for parser-only debugging)")
    ap.add_argument("--realsim-timer", type=int, default=None, help="Override scene.json::timer for RealSim runs")
    args = ap.parse_args()

    demos = args.demos or ["TwistingBarNH", "StretchingCloth", "PullingWooper", "SqueezingBall"]
    wp.init()

    results: list[DemoResult] = []
    for demo in demos:
        print(f"\n=== {demo} ({args.frames} frames, warmup={args.warmup}) ===", flush=True)
        res = DemoResult(demo=demo)
        if not args.skip_fba:
            try:
                t0 = time.perf_counter()
                res.fba = FBA_RUNNERS[demo](args.frames, args.warmup)
                print(
                    f"  [FBA] done in {time.perf_counter() - t0:.1f} s  "
                    f"step_mean={_fmt(res.fba.get('step_mean_ms'), '{:.2f}')} ms",
                    flush=True,
                )
            except Exception as exc:
                res.error = f"FBA: {type(exc).__name__}: {exc}"
                print(f"  [FBA] FAILED: {res.error}", flush=True)
        if not args.skip_realsim:
            try:
                from scripts.realsim_baseline.timing import run_realsim_with_timing  # noqa: PLC0415

                t0 = time.perf_counter()
                rs = run_realsim_with_timing(demo, max_frame=args.frames, timer_interval=args.realsim_timer)
                res.realsim = rs.to_dict()
                print(
                    f"  [RS]  done in {time.perf_counter() - t0:.1f} s  "
                    f"total_step={_fmt(rs.total_step_ms, '{:.2f}')} ms  "
                    f"blocks={rs.n_blocks}",
                    flush=True,
                )
            except Exception as exc:
                # Don't overwrite an FBA-side error.
                err = f"RealSim: {type(exc).__name__}: {exc}"
                res.error = res.error or err
                print(f"  [RS]  FAILED: {err}", flush=True)
        results.append(res)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(format_markdown(results))
    JSON_DUMP.parent.mkdir(parents=True, exist_ok=True)
    JSON_DUMP.write_text(json.dumps([r.__dict__ for r in results], indent=2))
    print(f"\nWrote report -> {REPORT_PATH}")
    print(f"Wrote JSON   -> {JSON_DUMP}")


if __name__ == "__main__":
    main()
