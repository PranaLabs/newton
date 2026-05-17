# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run RealSim offline and extract trajectory to .npz for Tier 1/2/3 verification.

Usage:
    python -m scripts.realsim_baseline.extract_trajectory <DemoName> [--max-frame N]

Writes ``/tmp/realsim_<DemoName>.npz`` with key ``positions`` shape ``(frames, particles, 3)``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from scripts.realsim_baseline import DEMOS, REALSIM_ROOT

REALSIM_BIN = REALSIM_ROOT / "build/bin/RealSim"
DUMP_ABC_BIN = Path("/tmp/dump_abc_traj")
ABC_DIR = REALSIM_ROOT / "simulation/output_abc"


def _make_offline_scene(scene_path: Path, max_frame: int | None) -> Path:
    cfg = json.loads(scene_path.read_text())
    cfg["offline"] = True
    if max_frame is None:
        max_frame = int(cfg.get("maxFrame") or cfg.get("stop") or 100)
    cfg["maxFrame"] = int(max_frame)
    # Normalize output_abc path uniformly across all demos: force
    # ``simulation/output_abc/<DemoName>`` regardless of scene-json setting.
    # Fixes per-scene drift:
    #   - StretchingCloth has no ``output_abc`` field → inject default.
    #   - PullingWooper has absolute path ``/simulation/.../PullingWooper/output``
    #     (leading slash points at fs root + extra ``/output`` subdir, so
    #     RealSim either fails Permission Denied or writes one level too
    #     deep for extract_abc to find).
    demo_name = scene_path.parent.name
    cfg["output_abc"] = f"simulation/output_abc/{demo_name}"
    fd, tmp_path = tempfile.mkstemp(prefix=f"realsim_{scene_path.parent.name}_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(cfg, indent=2))
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path)


def run_realsim(demo: str, max_frame: int | None = None) -> dict:
    cfg = DEMOS[demo]
    scene = REALSIM_ROOT / cfg["scene"]
    if not scene.exists():
        return {"status": "missing_scene", "scene": str(scene)}
    if max_frame is None:
        max_frame = cfg.get("expected_frames")
    tmp_scene = _make_offline_scene(scene, max_frame)
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(REALSIM_BIN), "-m", str(tmp_scene)],
            cwd=str(REALSIM_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        wall = time.perf_counter() - t0
    finally:
        tmp_scene.unlink(missing_ok=True)
    return {
        "status": "ok" if proc.returncode == 0 else "nonzero_rc",
        "wall_s": round(wall, 1),
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-1000:],
    }


def extract_abc(demo: str) -> tuple[np.ndarray, dict]:
    abc_path = ABC_DIR / demo / "output_obj_0.abc"
    if not abc_path.exists():
        return np.array([]), {"status": "missing_abc", "abc_path": str(abc_path)}
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
        return np.array([]), {"status": "dump_failed", "returncode": proc.returncode}
    positions = np.stack(frames, axis=0)
    return positions, {"status": "ok", "frames": positions.shape[0], "particles": positions.shape[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=sorted(DEMOS.keys()))
    ap.add_argument("--max-frame", type=int, default=None)
    ap.add_argument("--out-npz", type=str, default=None)
    args = ap.parse_args()

    out_path = Path(args.out_npz) if args.out_npz else Path(f"/tmp/realsim_{args.demo}.npz")

    print(f"[{args.demo}] running RealSim offline...", flush=True)
    rs_info = run_realsim(args.demo, max_frame=args.max_frame)
    print(f"[{args.demo}] RealSim: {rs_info}", flush=True)

    print(f"[{args.demo}] extracting trajectory...", flush=True)
    positions, ex_info = extract_abc(args.demo)
    print(f"[{args.demo}] extract: {ex_info}", flush=True)

    if ex_info["status"] != "ok":
        sys.exit(1)

    np.savez_compressed(out_path, positions=positions)
    print(f"[{args.demo}] saved {out_path} ({positions.shape})", flush=True)


if __name__ == "__main__":
    main()
