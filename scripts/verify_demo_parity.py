# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tier 1/2/3 FBA-RealSim demo parity verification harness.

Reads an FBA and a RealSim trajectory ``.npz`` (each with ``positions`` of
shape ``(frames, particles, 3)``) and emits a 3-tier verdict per
``docs/superpowers/specs/2026-05-17-fba-verification-criteria.md``.

Tier 1 (BINARY, must pass):
    - Per-demo physical event (e.g. wooper drops past gap)
    - Stability sentinels (no NaN/Inf, no particle outside 10x scene bbox)

Tier 2 (QUANTITATIVE, target -- investigate if missed):
    max-over-time, max-over-particles ||fba_pos - realsim_pos||_2
    target per demo:
        TwistingBarNH     1e-3 m
        StretchingCloth   1e-3 m
        PullingWooper     1e-2 m
        SqueezingBall     5e-2 m

Tier 3 (diagnostics, always reported, never auto-fail):
    COM drift, KE relative diff, bbox-diagonal relative diff,
    per-particle drift histogram (median/p90/p99/max).

Usage:
    uv run python scripts/verify_demo_parity.py \
        --demo SqueezingBall \
        --fba /tmp/fba_demo5_squeezing_ball.npz \
        --realsim /tmp/realsim_SqueezingBall.npz \
        [--out-json scripts/contact_demos_out/demo5/verdict.json] \
        [--out-strip scripts/contact_demos_out/demo5/comparison_strip.png]
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Demo configuration: Tier 2 thresholds + Tier 1 physical-event spec.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoSpec:
    name: str
    tier2_threshold_m: float
    physical_event: str


DEMO_SPECS: dict[str, DemoSpec] = {
    "TwistingBarNH": DemoSpec(
        name="TwistingBarNH",
        tier2_threshold_m=1e-3,
        physical_event="bar twists >= 80 deg without fragmentation",
    ),
    "StretchingCloth": DemoSpec(
        name="StretchingCloth",
        tier2_threshold_m=1e-3,
        physical_event="cloth x-bbox expands by >= 1.5 m without NaN",
    ),
    "PullingWooper": DemoSpec(
        name="PullingWooper",
        tier2_threshold_m=1e-2,
        physical_event="wooper min_y <= -7.0 m without NaN",
    ),
    "SqueezingBall": DemoSpec(
        name="SqueezingBall",
        tier2_threshold_m=5e-2,
        physical_event="ball mean_y < -8.0 m with no particle at |q| > 50 m",
    ),
}


# ---------------------------------------------------------------------------
# Loading + frame-rate alignment
# ---------------------------------------------------------------------------


def _load_positions(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"trajectory file not found: {path}")
    data = np.load(path)
    if "positions" not in data.files:
        raise KeyError(f"{path} missing key 'positions' (found keys: {list(data.files)})")
    pos = np.asarray(data["positions"])
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"{path} 'positions' must have shape (frames, particles, 3); got {pos.shape}")
    return pos.astype(np.float32, copy=False)


def _align_frames(fba: np.ndarray, realsim: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Reconcile frame counts. If they match, return as-is. If one is an
    integer multiple of the other, subsample the longer one. Otherwise truncate
    both to the shorter length.
    """
    nf_fba = fba.shape[0]
    nf_rs = realsim.shape[0]
    if nf_fba == nf_rs:
        return fba, realsim, "match"
    if nf_fba > nf_rs and nf_fba % nf_rs == 0:
        stride = nf_fba // nf_rs
        return fba[::stride][:nf_rs], realsim, f"fba subsampled by {stride}"
    if nf_rs > nf_fba and nf_rs % nf_fba == 0:
        stride = nf_rs // nf_fba
        return fba, realsim[::stride][:nf_fba], f"realsim subsampled by {stride}"
    n = min(nf_fba, nf_rs)
    return fba[:n], realsim[:n], f"truncated to {n} (no integer ratio)"


# ---------------------------------------------------------------------------
# Tier 1: stability sentinels + physical-event predicates
# ---------------------------------------------------------------------------


def _stability_sentinels(fba: np.ndarray, realsim: np.ndarray) -> dict:
    """No NaN/Inf in FBA. No particle outside 10x RealSim scene bbox.

    RealSim trajectory defines the "reference" scene bbox: we take the union of
    its first-frame and last-frame bboxes, compute a center + half-extent, and
    flag any FBA particle whose distance from the center exceeds 10x the
    half-extent magnitude.
    """
    fba_has_nan = bool(np.isnan(fba).any())
    fba_has_inf = bool(np.isinf(fba).any())
    # Scene radius derived from RealSim reference (first + last frame envelope).
    rs_lo = np.minimum(realsim[0].min(axis=0), realsim[-1].min(axis=0))
    rs_hi = np.maximum(realsim[0].max(axis=0), realsim[-1].max(axis=0))
    center = 0.5 * (rs_lo + rs_hi)
    half_extent = 0.5 * (rs_hi - rs_lo)
    scene_radius = float(np.linalg.norm(half_extent))
    scene_radius = max(scene_radius, 1.0)  # avoid degenerate zero-extent
    bbox_radius_10x = 10.0 * scene_radius
    # FBA escape check (skip if NaN, undefined distance).
    if fba_has_nan or fba_has_inf:
        fba_max_dist = float("inf")
        escape = True
    else:
        d = np.linalg.norm(fba - center[None, None, :], axis=-1)
        fba_max_dist = float(d.max())
        escape = fba_max_dist > bbox_radius_10x
    return {
        "fba_has_nan": fba_has_nan,
        "fba_has_inf": fba_has_inf,
        "fba_max_dist_from_scene_center_m": fba_max_dist,
        "scene_bbox_radius_m": scene_radius,
        "scene_bbox_radius_10x_m": bbox_radius_10x,
        "particle_escape": bool(escape),
        "stability_pass": not (fba_has_nan or fba_has_inf or escape),
    }


def _physical_event_twisting_bar(fba: np.ndarray) -> tuple[bool, dict]:
    """Top and bottom pinned slabs rotate +/- 10 deg/s; verify >= 80 deg
    rotation of pinned vertices within first ~900 frames.

    Heuristic: pick the topmost and bottommost slab of vertices (top 1% / bottom
    1% by initial y), project onto the xz-plane around the bar axis, and track
    angular displacement vs frame 0. Max angular sweep over time should reach
    80 deg.
    """
    p0 = fba[0]
    y = p0[:, 1]
    top_thresh = np.quantile(y, 0.99)
    bot_thresh = np.quantile(y, 0.01)
    top_idx = np.where(y >= top_thresh)[0]
    bot_idx = np.where(y <= bot_thresh)[0]
    # Centroid axis (xz of first frame).
    center_xz = p0[:, [0, 2]].mean(axis=0)

    def _max_sweep(idx: np.ndarray) -> float:
        xz0 = p0[idx][:, [0, 2]] - center_xz
        theta0 = np.arctan2(xz0[:, 1], xz0[:, 0])
        max_abs = 0.0
        for t in range(fba.shape[0]):
            xz = fba[t, idx][:, [0, 2]] - center_xz
            theta = np.arctan2(xz[:, 1], xz[:, 0])
            d = np.angle(np.exp(1j * (theta - theta0)))  # unwrap to (-pi, pi)
            max_abs = max(max_abs, float(np.max(np.abs(d))))
        return math.degrees(max_abs)

    top_sweep_deg = _max_sweep(top_idx) if top_idx.size else 0.0
    bot_sweep_deg = _max_sweep(bot_idx) if bot_idx.size else 0.0
    sweep_deg = max(top_sweep_deg, bot_sweep_deg)
    return sweep_deg >= 80.0, {
        "top_sweep_deg": top_sweep_deg,
        "bot_sweep_deg": bot_sweep_deg,
        "threshold_deg": 80.0,
    }


def _physical_event_stretching_cloth(fba: np.ndarray) -> tuple[bool, dict]:
    x0 = fba[0, :, 0]
    bbox_x0 = float(x0.max() - x0.min())
    xN = fba[-1, :, 0]
    bbox_xN = float(xN.max() - xN.min())
    growth = bbox_xN - bbox_x0
    has_nan = bool(np.isnan(fba).any())
    return (growth >= 1.5 and not has_nan), {
        "bbox_x_initial_m": bbox_x0,
        "bbox_x_final_m": bbox_xN,
        "bbox_x_growth_m": growth,
        "threshold_growth_m": 1.5,
        "fba_has_nan": has_nan,
    }


def _physical_event_pulling_wooper(fba: np.ndarray) -> tuple[bool, dict]:
    final_min_y = float(fba[-1, :, 1].min())
    has_nan = bool(np.isnan(fba).any())
    return (final_min_y <= -7.0 and not has_nan), {
        "final_min_y_m": final_min_y,
        "threshold_max_min_y_m": -7.0,
        "fba_has_nan": has_nan,
    }


def _physical_event_squeezing_ball(fba: np.ndarray) -> tuple[bool, dict]:
    final_mean_y = float(fba[-1, :, 1].mean())
    # No-scatter sentinel: no particle ever further than 50 m from origin.
    # Spec: "no particle outside |q| > 50 m at any frame" (verification-criteria.md).
    finite = fba[np.isfinite(fba).all(axis=-1)]
    if finite.size:
        overall_max_norm = float(np.linalg.norm(finite, axis=-1).max())
    else:
        overall_max_norm = float("inf")
    has_nan = bool(np.isnan(fba).any())
    return (final_mean_y < -8.0 and overall_max_norm <= 50.0 and not has_nan), {
        "final_mean_y_m": final_mean_y,
        "overall_max_norm_m": overall_max_norm,
        "threshold_max_final_mean_y_m": -8.0,
        "threshold_max_norm_m": 50.0,
        "fba_has_nan": has_nan,
    }


def _physical_event(demo: str, fba: np.ndarray) -> tuple[bool, dict]:
    if demo == "TwistingBarNH":
        return _physical_event_twisting_bar(fba)
    if demo == "StretchingCloth":
        return _physical_event_stretching_cloth(fba)
    if demo == "PullingWooper":
        return _physical_event_pulling_wooper(fba)
    if demo == "SqueezingBall":
        return _physical_event_squeezing_ball(fba)
    raise ValueError(f"unknown demo: {demo}")


# ---------------------------------------------------------------------------
# Tier 2: max-particle drift
# ---------------------------------------------------------------------------


def _tier2(fba: np.ndarray, realsim: np.ndarray, threshold_m: float) -> dict:
    if fba.shape != realsim.shape:
        return {
            "target_m": threshold_m,
            "max_drift_m": float("inf"),
            "pass": False,
            "argmax_frame": -1,
            "argmax_particle": -1,
            "note": f"shape mismatch fba={fba.shape} realsim={realsim.shape}",
        }
    diff = fba - realsim
    # ||.||_2 per particle per frame -> (frames, particles)
    drift = np.linalg.norm(diff, axis=-1)
    if np.isnan(drift).any():
        # NaN in FBA already caught by stability; record gracefully.
        max_drift = float("inf")
        argmax_flat = -1
    else:
        max_drift = float(drift.max())
        argmax_flat = int(np.argmax(drift))
    if argmax_flat >= 0:
        argmax_frame = argmax_flat // drift.shape[1]
        argmax_particle = argmax_flat % drift.shape[1]
    else:
        argmax_frame = -1
        argmax_particle = -1
    return {
        "target_m": threshold_m,
        "max_drift_m": max_drift,
        "pass": max_drift < threshold_m,
        "argmax_frame": int(argmax_frame),
        "argmax_particle": int(argmax_particle),
    }


# ---------------------------------------------------------------------------
# Tier 3: diagnostics
# ---------------------------------------------------------------------------


def _bbox_diag(positions: np.ndarray) -> np.ndarray:
    """Per-frame bbox diagonal length."""
    lo = positions.min(axis=1)
    hi = positions.max(axis=1)
    return np.linalg.norm(hi - lo, axis=-1)


def _ke_proxy(positions: np.ndarray) -> np.ndarray:
    """Per-frame kinetic-energy proxy = sum of squared per-particle velocity
    (finite-difference, unit mass). Acts as the comparison curve; absolute scale
    is irrelevant -- only the relative diff between FBA and RealSim matters.
    """
    if positions.shape[0] < 2:
        return np.zeros(positions.shape[0], dtype=np.float64)
    vel = np.diff(positions, axis=0)
    ke = 0.5 * np.sum(vel * vel, axis=(1, 2))
    # Pad with leading zero so output length == frames.
    return np.concatenate([[0.0], ke.astype(np.float64)])


def _tier3(fba: np.ndarray, realsim: np.ndarray, tier2_threshold_m: float) -> dict:
    out: dict = {}
    # COM drift series.
    fba_com = fba.mean(axis=1)
    rs_com = realsim.mean(axis=1)
    com_diff = np.linalg.norm(fba_com - rs_com, axis=-1)
    if np.isnan(com_diff).any():
        com_max = float("inf")
    else:
        com_max = float(com_diff.max())
    out["com_drift_m_max"] = com_max
    out["com_drift_m_final"] = float(com_diff[-1]) if not np.isnan(com_diff[-1]) else float("inf")
    out["com_target_m"] = 1e-3

    # KE proxy relative diff.
    ke_fba = _ke_proxy(fba)
    ke_rs = _ke_proxy(realsim)
    denom = np.maximum(np.abs(ke_rs), 1e-12)
    ke_rel = np.abs(ke_fba - ke_rs) / denom
    finite = np.isfinite(ke_rel)
    out["ke_rel_diff_max"] = float(ke_rel[finite].max()) if finite.any() else float("inf")
    out["ke_target_rel"] = 0.05

    # BBox diagonal relative diff.
    bb_fba = _bbox_diag(fba)
    bb_rs = _bbox_diag(realsim)
    bb_denom = np.maximum(np.abs(bb_rs), 1e-12)
    bb_rel = np.abs(bb_fba - bb_rs) / bb_denom
    finite_bb = np.isfinite(bb_rel)
    out["bbox_rel_diff_max"] = float(bb_rel[finite_bb].max()) if finite_bb.any() else float("inf")
    out["bbox_target_rel"] = 0.05

    # Per-particle drift histogram (max across frames per particle).
    if fba.shape == realsim.shape and not np.isnan(fba).any():
        per_frame_drift = np.linalg.norm(fba - realsim, axis=-1)  # (frames, particles)
        per_particle_max = per_frame_drift.max(axis=0)
        out["drift_histogram"] = {
            "median": float(np.median(per_particle_max)),
            "p90": float(np.quantile(per_particle_max, 0.90)),
            "p99": float(np.quantile(per_particle_max, 0.99)),
            "max": float(per_particle_max.max()),
            "tier2_threshold_m": tier2_threshold_m,
        }
    else:
        out["drift_histogram"] = {
            "median": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("inf"),
            "tier2_threshold_m": tier2_threshold_m,
            "note": "shape mismatch or NaN in FBA",
        }

    # Contact count placeholder (Tier 3 spec asks for it but trajectories don't
    # carry contact info; downstream tooling should fill this in if available).
    out["contact_count_note"] = (
        "contact count per frame not derivable from positions alone; supply via separate npz if needed"
    )

    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _verdict(tier1_pass: bool, tier2_pass: bool, com_aligned: bool) -> tuple[str, str]:
    if not tier1_pass:
        return "RED", "Tier 1 failed (physical event or stability)"
    if tier2_pass:
        return "GREEN", "Tier 1 passed, Tier 2 within threshold"
    if com_aligned:
        return "YELLOW", "Tier 2 missed but COM aligned -- local outlier"
    return "YELLOW", "Tier 2 missed and COM diverges -- systematic"


# ---------------------------------------------------------------------------
# Comparison strip PNG
# ---------------------------------------------------------------------------


def _save_strip(fba: np.ndarray, realsim: np.ndarray, out_path: Path, demo: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = fba.shape[0]
    idxs = [
        0,
        max(1, n_frames // 4),
        max(2, n_frames // 2),
        max(3, (3 * n_frames) // 4),
        n_frames - 1,
    ]
    idxs = sorted({min(n_frames - 1, i) for i in idxs})
    # Pad if dedup removed entries (very short trajectories).
    while len(idxs) < 5:
        idxs.append(idxs[-1])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"{demo} parity: FBA (top) vs RealSim (bottom)")
    # Shared scatter range from union of both for visual consistency.
    all_pts = np.concatenate([fba.reshape(-1, 3), realsim.reshape(-1, 3)], axis=0)
    finite_pts = all_pts[np.isfinite(all_pts).all(axis=-1)]
    if finite_pts.size:
        xlim = (float(finite_pts[:, 0].min()), float(finite_pts[:, 0].max()))
        ylim = (float(finite_pts[:, 1].min()), float(finite_pts[:, 1].max()))
    else:
        xlim = (-1.0, 1.0)
        ylim = (-1.0, 1.0)

    for col, fi in enumerate(idxs):
        for row, (label, traj) in enumerate([("FBA", fba), ("RealSim", realsim)]):
            ax = axes[row, col]
            pts = traj[fi]
            finite = np.isfinite(pts).all(axis=-1)
            ax.scatter(pts[finite, 0], pts[finite, 1], s=1, c="C0" if row == 0 else "C1")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.set_title(f"{label} frame {fi}")
            ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level verify()
# ---------------------------------------------------------------------------


def verify(
    demo: str,
    fba_npz: Path,
    realsim_npz: Path,
    out_strip: Path | None = None,
) -> dict:
    if demo not in DEMO_SPECS:
        raise ValueError(f"unknown demo {demo!r}; valid: {sorted(DEMO_SPECS)}")
    spec = DEMO_SPECS[demo]
    fba = _load_positions(fba_npz)
    realsim = _load_positions(realsim_npz)
    fba_aligned, realsim_aligned, align_note = _align_frames(fba, realsim)

    stab = _stability_sentinels(fba_aligned, realsim_aligned)
    event_pass, event_details = _physical_event(demo, fba_aligned)
    tier1_pass = bool(stab["stability_pass"] and event_pass)

    tier2 = _tier2(fba_aligned, realsim_aligned, spec.tier2_threshold_m)
    tier3 = _tier3(fba_aligned, realsim_aligned, spec.tier2_threshold_m)
    com_aligned = tier3["com_drift_m_max"] < tier3["com_target_m"]
    verdict, verdict_reason = _verdict(tier1_pass, tier2["pass"], com_aligned)

    report = {
        "demo": demo,
        "fba_npz": str(fba_npz),
        "realsim_npz": str(realsim_npz),
        "frames_fba": int(fba.shape[0]),
        "frames_realsim": int(realsim.shape[0]),
        "frames_aligned": int(fba_aligned.shape[0]),
        "frame_alignment_note": align_note,
        "particles_fba": int(fba.shape[1]),
        "particles_realsim": int(realsim.shape[1]),
        "tier1": {
            "binary_pass": tier1_pass,
            "physical_event": spec.physical_event,
            "physical_event_pass": bool(event_pass),
            "physical_event_details": event_details,
            "stability_pass": bool(stab["stability_pass"]),
            "stability_details": stab,
        },
        "tier2": tier2,
        "tier3": tier3,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }

    if out_strip is not None:
        try:
            _save_strip(fba_aligned, realsim_aligned, out_strip, demo)
            report["strip_png"] = str(out_strip)
        except Exception as exc:  # pragma: no cover
            report["strip_png_error"] = repr(exc)

    return report


# ---------------------------------------------------------------------------
# JSON serialization (numpy/scalar safety)
# ---------------------------------------------------------------------------


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON-serializable: {type(obj)}")


def _sanitize(obj):
    """Replace non-finite floats with sentinels recursively so json.dump can
    emit a valid JSON document (default json allows NaN, but downstream tools
    often choke on it).
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--demo",
        required=False,
        choices=sorted(DEMO_SPECS),
        help="Demo name (selects Tier 1 physical event + Tier 2 threshold)",
    )
    p.add_argument("--fba", required=False, type=Path, help="FBA trajectory .npz")
    p.add_argument("--realsim", required=False, type=Path, help="RealSim trajectory .npz")
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write the verdict JSON report",
    )
    p.add_argument(
        "--out-strip",
        type=Path,
        default=None,
        help="Optional path to write a 2x5 comparison-strip PNG",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in synthetic tests (GREEN/YELLOW/RED) and exit",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        return _run_self_tests()
    missing = [
        flag
        for flag, val in [
            ("--demo", args.demo),
            ("--fba", args.fba),
            ("--realsim", args.realsim),
        ]
        if val is None
    ]
    if missing:
        raise SystemExit(f"missing required args: {', '.join(missing)} (or use --self-test)")
    report = verify(
        demo=args.demo,
        fba_npz=args.fba,
        realsim_npz=args.realsim,
        out_strip=args.out_strip,
    )
    safe = _sanitize(report)
    text = json.dumps(safe, indent=2, default=_json_default, allow_nan=False)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")
    return 0 if report["verdict"] != "RED" else 1


# ---------------------------------------------------------------------------
# Self-tests: synthetic GREEN / YELLOW / RED parity scenarios
# ---------------------------------------------------------------------------


def _synthetic_squeezing_ball_trajectory(seed: int = 0) -> np.ndarray:
    """Make a (frames, particles, 3) trajectory whose ball-shaped point cloud
    drops from y=+1 to y=-9 over 600 frames, so the SqueezingBall Tier 1
    physical event passes (final mean_y < -8, no scatter)."""
    rng = np.random.default_rng(seed)
    n_frames = 600
    n_particles = 200
    # Sphere of radius 0.3 around origin.
    pts0 = rng.normal(size=(n_particles, 3)).astype(np.float32)
    pts0 /= np.linalg.norm(pts0, axis=1, keepdims=True)
    pts0 *= 0.3
    ys = np.linspace(1.0, -9.0, n_frames, dtype=np.float32)
    traj = np.broadcast_to(pts0[None, :, :], (n_frames, n_particles, 3)).copy()
    traj[:, :, 1] += ys[:, None]
    return traj


def _run_self_tests() -> int:
    rng = np.random.default_rng(42)
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Baseline trajectory.
        traj = _synthetic_squeezing_ball_trajectory()

        # ---- GREEN: identical trajectories + 1e-6 noise.
        noise = rng.normal(scale=1e-6, size=traj.shape).astype(np.float32)
        fba_path = td_path / "fba_green.npz"
        rs_path = td_path / "rs_green.npz"
        np.savez_compressed(fba_path, positions=traj + noise)
        np.savez_compressed(rs_path, positions=traj)
        report = verify("SqueezingBall", fba_path, rs_path)
        print(
            "[self-test GREEN]",
            "verdict=",
            report["verdict"],
            "tier2_max=",
            report["tier2"]["max_drift_m"],
            "com_max=",
            report["tier3"]["com_drift_m_max"],
        )
        if report["verdict"] != "GREEN":
            failures.append(f"GREEN scenario produced {report['verdict']}")

        # ---- YELLOW: 0.1 m shift in y (systematic, COM diverges).
        shifted = traj.copy()
        shifted[:, :, 1] += 0.1
        fba_path = td_path / "fba_yellow.npz"
        rs_path = td_path / "rs_yellow.npz"
        # Apply shift to FBA (so RealSim is the unshifted reference).
        np.savez_compressed(fba_path, positions=shifted)
        np.savez_compressed(rs_path, positions=traj)
        # Adjust shifted final mean_y so Tier 1 still passes (shifted by 0.1 m
        # still has mean_y ~ -8.9 in final frame).
        report = verify("SqueezingBall", fba_path, rs_path)
        print(
            "[self-test YELLOW]",
            "verdict=",
            report["verdict"],
            "tier2_max=",
            report["tier2"]["max_drift_m"],
            "com_max=",
            report["tier3"]["com_drift_m_max"],
        )
        if report["verdict"] != "YELLOW":
            failures.append(f"YELLOW scenario produced {report['verdict']}")

        # ---- RED: NaN injected.
        nan_traj = traj.copy()
        nan_traj[100, 0, 0] = np.nan
        fba_path = td_path / "fba_red.npz"
        rs_path = td_path / "rs_red.npz"
        np.savez_compressed(fba_path, positions=nan_traj)
        np.savez_compressed(rs_path, positions=traj)
        report = verify("SqueezingBall", fba_path, rs_path)
        print(
            "[self-test RED]",
            "verdict=",
            report["verdict"],
            "stability_pass=",
            report["tier1"]["stability_pass"],
            "fba_has_nan=",
            report["tier1"]["stability_details"]["fba_has_nan"],
        )
        if report["verdict"] != "RED":
            failures.append(f"RED scenario produced {report['verdict']}")

    if failures:
        print("SELF-TEST FAILED:", failures)
        return 1
    print("SELF-TEST OK: GREEN / YELLOW / RED all matched expected verdicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
