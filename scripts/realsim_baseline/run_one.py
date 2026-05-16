# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run one CudaTests demo via the RealSim binary in offline mode and return timing stats."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.realsim_baseline import DEMOS, REALSIM_ROOT

REALSIM_BIN = REALSIM_ROOT / "build/bin/RealSim"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TIMING_RE = re.compile(r"Total time step cost\s*=\s*([0-9]+\.?[0-9]*)\s*ms")


def _make_offline_scene(scene_path: Path, max_frame: int | None) -> Path:
    """Deep-copy a CudaTests scene.json with ``offline=true`` and ``maxFrame`` injected."""
    cfg = json.loads(scene_path.read_text())
    cfg["offline"] = True
    if max_frame is None:
        max_frame = int(cfg.get("maxFrame") or cfg.get("stop") or 100)
    cfg["maxFrame"] = int(max_frame)
    tmp = Path(tempfile.mkstemp(prefix=f"realsim_{scene_path.parent.name}_", suffix=".json")[1])
    tmp.write_text(json.dumps(cfg, indent=2))
    return tmp


def run(demo: str, max_frame: int | None = None) -> dict:
    cfg = DEMOS[demo]
    scene = REALSIM_ROOT / cfg["scene"]
    if not scene.exists():
        return {"demo": demo, "status": "missing_scene", "scene": str(scene)}

    if max_frame is None:
        max_frame = cfg.get("expected_frames")
    tmp_scene = _make_offline_scene(scene, max_frame)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(REALSIM_BIN), "-m", str(tmp_scene)],
        cwd=str(REALSIM_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    wall = time.perf_counter() - t0

    stdout_clean = ANSI.sub("", proc.stdout)
    times_ms = [float(m.group(1)) for m in TIMING_RE.finditer(stdout_clean)]
    if not times_ms:
        return {
            "demo": demo,
            "status": "no_timing",
            "wall_s": wall,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "stdout_tail": stdout_clean[-2000:],
        }

    sorted_t = sorted(times_ms)
    n = len(sorted_t)
    return {
        "demo": demo,
        "status": "ok",
        "frames_timed": n,
        "mean_ms": round(sum(sorted_t) / n, 3),
        "median_ms": round(sorted_t[n // 2], 3),
        "p95_ms": round(sorted_t[min(n - 1, int(0.95 * n))], 3),
        "wall_s": round(wall, 1),
        "scene": str(scene),
        "max_frame": max_frame,
        "returncode": proc.returncode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=sorted(DEMOS.keys()))
    ap.add_argument("--max-frame", type=int, default=None, help="Override maxFrame for offline run")
    args = ap.parse_args()
    result = run(args.demo, max_frame=args.max_frame)
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
