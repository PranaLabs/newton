# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Parse the per-PD-iter timing blocks emitted by ``LocalGlobalSolver::printTimer``.

RealSim prints a profile block every ``timer`` frames (default 50, set by
``scene.json::timer``). Each block aggregates the per-PD-iter mean cost for
each sub-phase over the trailing window:

* ``Schur cost``           - Schur ``W`` build mean per PD outer iter (NSN only).
* ``Local cost``           - Local energy projection mean per PD iter.
* ``Linear solve cost``    - ``A^{-1} b`` mean per PD iter.
* ``Assembly``             - NSN ``build`` per PD outer iter (NSN only).
* ``Cst solve``            - NSN Schur LCP / PCR solve (NSN only).
* ``Correction``           - NSN apply-correction (NSN only).
* ``Total Global cost``    - ``Assembly + Cst solve + Correction`` (NSN only).
* ``Frame without collision detection`` - frame wall minus collision detection.
* ``Total time step cost`` - whole frame wall.

This module aggregates *across* all blocks emitted by a single offline run,
then returns the per-field mean weighted by the count of windows. Per-block
window length is recoverable from ``Printing profile over N time steps``.

Source: ``LocalGlobalSolver.cpp:415-510``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.realsim_baseline import DEMOS, REALSIM_ROOT

REALSIM_BIN = REALSIM_ROOT / "build/bin/RealSim"

# RealSim peppers stdout with ANSI color escapes; strip before regex matching.
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# One block delimited by ``Printing profile over N time steps`` ... ``-----``.
BLOCK_HEADER_RE = re.compile(r"Printing profile over\s+(\d+)\s+time steps")

# Per-field regexes. All matches run on the *whole* ANSI-stripped stdout;
# multiple blocks may appear, in which case we average across blocks.
_FIELD_RE = {
    "n_constraints": re.compile(r"Number of constraints over\s+\d+\s+time steps,\s+count\s*=\s*([0-9.eE+-]+)"),
    "cst_solve_iter": re.compile(
        r"Number of constraint solver iteration over\s+\d+\s+times steps,\s+count\s*=\s*([0-9.eE+-]+)"
    ),
    "schur": re.compile(r"Schur cost\s*=\s*([0-9.eE+-]+)\s*ms"),
    "local": re.compile(r"Local cost\s*=\s*([0-9.eE+-]+)\s*ms"),
    "linear_solve": re.compile(r"Linear solve cost\s*=\s*([0-9.eE+-]+)\s*ms"),
    "build": re.compile(r"Assembly called by\s+\d+\s+times,\s+cost\s*=\s*([0-9.eE+-]+)\s*ms"),
    "cst_solve_ms": re.compile(r"Cst solve called by\s+\d+\s+times,\s+cost\s*=\s*([0-9.eE+-]+)\s*ms"),
    "correction": re.compile(r"Correction called by\s+\d+\s+times,\s+cost\s*=\s*([0-9.eE+-]+)\s*ms"),
    "total_global": re.compile(r"Total Global cost\s*=\s*([0-9.eE+-]+)\s*ms"),
    "frame_no_cd": re.compile(r"Frame without collision detection\s*=\s*([0-9.eE+-]+)\s*ms"),
    "total_step": re.compile(r"Total time step cost\s*=\s*([0-9.eE+-]+)\s*ms"),
}


@dataclass
class RealSimTiming:
    """Aggregated RealSim per-PD-iter timing for a single demo run.

    All ``*_ms`` fields are means in milliseconds per the granularity quoted
    by :mod:`scripts.realsim_baseline.timing` module docstring. Fields tied
    to NSN are ``None`` when no constraints are active (e.g. StretchingCloth).
    """

    demo: str
    n_frames: int
    timer_interval: int
    n_blocks: int
    n_constraints_mean: float | None
    cst_solve_iter_mean: float | None
    schur_ms: float | None
    local_ms: float
    linear_solve_ms: float
    build_ms: float | None
    cst_solve_ms: float | None
    correction_ms: float | None
    total_global_ms: float | None
    frame_no_cd_ms: float
    total_step_ms: float
    wall_s: float | None = None
    returncode: int | None = None

    def to_dict(self) -> dict:
        return {
            "demo": self.demo,
            "n_frames": self.n_frames,
            "timer_interval": self.timer_interval,
            "n_blocks": self.n_blocks,
            "n_constraints_mean": self.n_constraints_mean,
            "cst_solve_iter_mean": self.cst_solve_iter_mean,
            "schur_ms": self.schur_ms,
            "local_ms": self.local_ms,
            "linear_solve_ms": self.linear_solve_ms,
            "build_ms": self.build_ms,
            "cst_solve_ms": self.cst_solve_ms,
            "correction_ms": self.correction_ms,
            "total_global_ms": self.total_global_ms,
            "frame_no_cd_ms": self.frame_no_cd_ms,
            "total_step_ms": self.total_step_ms,
            "wall_s": self.wall_s,
            "returncode": self.returncode,
        }


def _mean_or_none(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def parse_realsim_stdout(stdout: str, demo: str = "?", timer_interval: int = 50) -> RealSimTiming:
    """Parse all ``Printing profile`` blocks in ``stdout`` and return aggregated means.

    Per-block window sizes can vary (last block is partial), so the
    *unweighted* mean is reported. RealSim itself uses the same scheme when
    cross-block comparisons are made.

    Args:
        stdout: RealSim stdout, ANSI-coded or not.
        demo: Demo name (carried through for downstream identification).
        timer_interval: ``scene.json["timer"]`` value; carried through.

    Returns:
        :class:`RealSimTiming` instance. If no blocks were emitted at all
        (e.g. the run died before reaching ``timer`` frames), local / total
        fields are 0.0 and ``n_blocks`` is 0.
    """
    stdout = ANSI.sub("", stdout)
    blocks = BLOCK_HEADER_RE.findall(stdout)
    n_blocks = len(blocks)
    frames_in_block = [int(b) for b in blocks]
    n_frames_total = sum(frames_in_block) if frames_in_block else 0

    fields: dict[str, list[float]] = {k: [] for k in _FIELD_RE}
    for key, rx in _FIELD_RE.items():
        for m in rx.finditer(stdout):
            try:
                fields[key].append(float(m.group(1)))
            except ValueError:
                pass

    local_ms = _mean_or_none(fields["local"]) or 0.0
    linear_solve_ms = _mean_or_none(fields["linear_solve"]) or 0.0
    frame_no_cd_ms = _mean_or_none(fields["frame_no_cd"]) or 0.0
    total_step_ms = _mean_or_none(fields["total_step"]) or 0.0

    return RealSimTiming(
        demo=demo,
        n_frames=n_frames_total,
        timer_interval=timer_interval,
        n_blocks=n_blocks,
        n_constraints_mean=_mean_or_none(fields["n_constraints"]),
        cst_solve_iter_mean=_mean_or_none(fields["cst_solve_iter"]),
        schur_ms=_mean_or_none(fields["schur"]),
        local_ms=local_ms,
        linear_solve_ms=linear_solve_ms,
        build_ms=_mean_or_none(fields["build"]),
        cst_solve_ms=_mean_or_none(fields["cst_solve_ms"]),
        correction_ms=_mean_or_none(fields["correction"]),
        total_global_ms=_mean_or_none(fields["total_global"]),
        frame_no_cd_ms=frame_no_cd_ms,
        total_step_ms=total_step_ms,
    )


def _make_offline_scene(scene_path: Path, max_frame: int | None, timer_interval: int | None) -> tuple[Path, int]:
    """Inject ``offline=true``, ``maxFrame``, and (optionally) override ``timer``.

    Returns the temp scene path and the effective ``timer`` interval.
    """
    cfg = json.loads(scene_path.read_text())
    cfg["offline"] = True
    if max_frame is None:
        max_frame = int(cfg.get("maxFrame") or cfg.get("stop") or 100)
    cfg["maxFrame"] = int(max_frame)
    if timer_interval is not None:
        cfg["timer"] = int(timer_interval)
    effective_timer = int(cfg.get("timer", 50))
    fd, tmp_path = tempfile.mkstemp(prefix=f"realsim_timing_{scene_path.parent.name}_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(cfg, indent=2))
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path), effective_timer


def run_realsim_with_timing(
    demo: str,
    max_frame: int | None = None,
    timer_interval: int | None = None,
    timeout_s: int = 3600,
) -> RealSimTiming:
    """Run RealSim offline once and return :class:`RealSimTiming`.

    Args:
        demo: Key into :data:`scripts.realsim_baseline.DEMOS`.
        max_frame: Optional override for ``scene.json::maxFrame``. Defaults
            to ``DEMOS[demo]["expected_frames"]`` then the scene's stored value.
        timer_interval: Optional override for ``scene.json::timer`` (the
            number of frames per profile block).
        timeout_s: Subprocess timeout in seconds.

    Returns:
        :class:`RealSimTiming` populated from stdout. ``wall_s`` and
        ``returncode`` are filled from the subprocess; if no profile blocks
        were emitted, the timing fields are 0.0 (sentinel) and ``n_blocks=0``.
    """
    if demo not in DEMOS:
        raise KeyError(f"unknown demo {demo!r}; pick from {sorted(DEMOS)}")
    cfg = DEMOS[demo]
    scene = REALSIM_ROOT / cfg["scene"]
    if not scene.exists():
        raise FileNotFoundError(f"scene not found: {scene}")
    if max_frame is None:
        max_frame = cfg.get("expected_frames")
    tmp_scene, effective_timer = _make_offline_scene(scene, max_frame, timer_interval)
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(REALSIM_BIN), "-m", str(tmp_scene)],
            cwd=str(REALSIM_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        wall = time.perf_counter() - t0
    finally:
        tmp_scene.unlink(missing_ok=True)

    timing = parse_realsim_stdout(proc.stdout, demo=demo, timer_interval=effective_timer)
    timing.wall_s = round(wall, 2)
    timing.returncode = int(proc.returncode)
    return timing


def main() -> None:  # pragma: no cover - thin CLI for manual checks
    ap = argparse.ArgumentParser(description="Parse RealSim per-PD-iter timing blocks")
    ap.add_argument("demo", choices=sorted(DEMOS.keys()))
    ap.add_argument("--max-frame", type=int, default=None)
    ap.add_argument("--timer-interval", type=int, default=None)
    ap.add_argument("--stdout-from-file", type=str, default=None, help="Skip running; parse an existing log")
    args = ap.parse_args()

    if args.stdout_from_file:
        timing = parse_realsim_stdout(
            Path(args.stdout_from_file).read_text(),
            demo=args.demo,
            timer_interval=args.timer_interval or 50,
        )
    else:
        timing = run_realsim_with_timing(args.demo, max_frame=args.max_frame, timer_interval=args.timer_interval)
    print(json.dumps(timing.to_dict(), indent=2))


if __name__ == "__main__":
    main()
