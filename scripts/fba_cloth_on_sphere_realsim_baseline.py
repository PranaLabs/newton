# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Capture RealSim single-env cloth-on-sphere baselines (Phase 1.2).

Drives ``CudaTests/ParallelEnvTest`` with ``parallelEnv.matrix=[1,1,1]`` and
constraint-solver type overridden per ``--mode``.  Output is written to
``scripts/realsim_baseline/cloth_on_sphere_<mode>_ref.npz``.

Usage::

    uv run python scripts/fba_cloth_on_sphere_realsim_baseline.py \
        --mode full --max-frame 300

    uv run python scripts/fba_cloth_on_sphere_realsim_baseline.py \
        --mode lite --max-frame 300
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REALSIM_ROOT = Path("/home/ziqiu/work/RealSim_py/realsim_py")
REALSIM_BIN = REALSIM_ROOT / "build/bin/RealSim"
DUMP_ABC_BIN = Path("/tmp/dump_abc_traj")
SCENE_PATH = REALSIM_ROOT / "simulation/config/CudaTests/ParallelEnvTest/scene.json"

ANSI = re.compile(r"\x1b\[[0-9;]*m")

_MODE_TO_SOLVER = {
    "full": "NonSmoothNewton_CUDA",
    "lite": "LiteNonSmoothNewton_CUDA",
}


def _make_scene(mode: str, max_frame: int, tag: str) -> tuple[Path, str]:
    cfg = json.loads(SCENE_PATH.read_text())
    cfg["offline"] = True
    cfg["maxFrame"] = int(max_frame)
    cfg["parallelEnv"] = {"matrix": [1, 1, 1], "space": [10, 10, 10]}
    cfg["constraintsolver"]["type"] = _MODE_TO_SOLVER[mode]
    out_subdir = f"ClothOnSphere_{tag}"
    cfg["output_abc"] = f"simulation/output_abc/{out_subdir}"
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"realsim_cloth_on_sphere_{mode}_", suffix=".json"
    )
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(cfg, indent=2))
    return Path(tmp_path), out_subdir


def _run_realsim(scene_path: Path) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(REALSIM_BIN), "-m", str(scene_path)],
        cwd=str(REALSIM_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    return proc.returncode, time.perf_counter() - t0, ANSI.sub("", proc.stdout)[-3000:]


def _extract_abc(out_subdir: str) -> np.ndarray:
    abc_path = REALSIM_ROOT / "simulation/output_abc" / out_subdir / "output_obj_0.abc"
    if not abc_path.exists():
        raise FileNotFoundError(f"ABC not produced at {abc_path}")
    proc = subprocess.Popen(
        [str(DUMP_ABC_BIN), str(abc_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    num_frames = int(proc.stdout.readline().strip())
    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        n_verts = int(proc.stdout.readline().strip())
        verts = np.empty((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            parts = proc.stdout.readline().split()
            verts[i, 0] = float(parts[0])
            verts[i, 1] = float(parts[1])
            verts[i, 2] = float(parts[2])
        frames.append(verts)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"dump_abc_traj failed: rc={proc.returncode}")
    return np.stack(frames, axis=0)


def _parse_step_ms(stdout_tail: str, max_frame: int) -> np.ndarray:
    """Pull per-step `Total time step cost = X ms` lines from RealSim stdout.

    Returns a 1D array of step times in ms.  Missing entries are filled with NaN.
    """
    pattern = re.compile(r"Total time step cost\s*=\s*([0-9]+\.?[0-9]*)\s*ms")
    matches = pattern.findall(stdout_tail)
    ms = np.array([float(x) for x in matches], dtype=np.float32)
    if ms.size == 0:
        return np.full(max_frame, np.nan, dtype=np.float32)
    if ms.size >= max_frame:
        return ms[-max_frame:]
    out = np.full(max_frame, np.nan, dtype=np.float32)
    out[-ms.size:] = ms
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["full", "lite"], required=True)
    parser.add_argument("--max-frame", type=int, default=300)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/realsim_baseline"),
    )
    args = parser.parse_args()

    if not REALSIM_BIN.exists():
        print(f"ERROR: RealSim binary not found at {REALSIM_BIN}", file=sys.stderr)
        return 1

    tag = "full" if args.mode == "full" else "lite"
    print(f"[Phase 1.2] capturing RealSim {args.mode} baseline @ frames={args.max_frame}")

    scene, out_subdir = _make_scene(args.mode, args.max_frame, tag)
    try:
        rc, wall, stdout_tail = _run_realsim(scene)
        if rc != 0 and rc != -6:  # RealSim cleanly exits with SIGABRT after offline run sometimes
            print(f"WARN: RealSim returncode={rc}, wall={wall:.1f}s")
            print(stdout_tail[-1500:])
            # Still try extraction — RealSim's offline pipeline often writes
            # the final frame just before this teardown crash.
    finally:
        scene.unlink(missing_ok=True)

    positions = _extract_abc(out_subdir)
    step_ms = _parse_step_ms(stdout_tail, max_frame=positions.shape[0])
    print(f"  positions: {positions.shape}, wall={wall:.1f}s, "
          f"step_mean={float(np.nanmean(step_ms)):.2f} ms")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"cloth_on_sphere_{tag}_ref.npz"
    np.savez(
        out_path,
        positions=positions,
        step_ms=step_ms,
        n_frames=positions.shape[0],
    )
    print(f"  saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
