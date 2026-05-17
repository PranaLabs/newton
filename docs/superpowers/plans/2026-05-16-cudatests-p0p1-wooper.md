# CudaTests Phase 0 + Phase 1 (PullingWooper) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the RealSim-baseline measurement pipeline (Phase 0) and ship the first 5k-particle contact demo (PullingWooper, Phase 1) with FBA `ms/step ≤ RealSim ms/step` on the user's machine.

**Architecture:** A user-space `scripts/realsim_baseline/` driver invokes the existing `/home/ziqiu/work/RealSim_py/realsim_py/build/bin/RealSim` binary in offline mode against each CudaTests `scene.json`, parses its timing output, and writes `scripts/cudatest_baselines.json`. The FBA side adds `scripts/fba_cudatest_bench.py` as a parallel driver that consumes the same demo registry and produces a comparable row. Phase 1 wires `PullingWooper` end-to-end by reading the existing wooper Medit `.mesh`, building a Newton model with a single PULLING action driven by `SolverFBA.set_pin_targets()`, and recording one perf row. Conservative changes to `SolverFBA` (NSN iter default 20→10, λ-cap parameter) land in Phase 1 unconditionally; CUDA Graph capture and λ warm-start are conditional on the perf gate.

**Tech Stack:** Python 3, `uv`, `warp`, `numpy`, Newton (`/home/ziqiu/work/newton`), Medit `.mesh` parser (already in `scripts/fba_twisting_bar_cudatests.py`), RealSim binary, `matplotlib` (Agg backend) for snapshot strips, `unittest` (NOT pytest — see AGENTS.md).

---

## Pre-flight Checks

These confirm assumptions in the plan. Do them once before Task 0.1.

- [ ] **Confirm we are on the FBA dev branch.**

```bash
cd /home/ziqiu/work/newton
git status
```
Expected: `On branch ziqiu/fba-solver-design`.

- [ ] **Confirm the RealSim binary exists.**

```bash
ls -la /home/ziqiu/work/RealSim_py/realsim_py/build/bin/RealSim
```
Expected: file exists, executable bit set. If missing, see Task 0.1.

- [ ] **Confirm the wooper mesh exists.**

```bash
ls -la /home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/wooper_volume_5325P.mesh
```
Expected: ~few MB text file beginning with `MeshVersionFormatted 2`.

- [ ] **Confirm `set_pin_targets` is the right pin API and PD energy already handles pinned vertices.**

Read: `newton/_src/solvers/fba/solver_fba.py:559-575`. Confirm `set_pin_targets(target_positions)` accepts `wp.array[wp.vec3]` or numpy `(N, 3)` float32. Confirm pinned particles are those with `inv_mass == 0` (matches `add_soft_mesh` / `add_cloth_grid` API).

---

## Phase 0: RealSim Baseline

The Phase 0 deliverable is `scripts/cudatest_baselines.json` with one row per demo, plus a re-runnable driver. Demos that fail to run on this machine still get a row with `status: "failed"` so we know not to gate on them.

### Task 0.1: Verify RealSim binary runs in offline mode

**Files:**
- (no files modified — sanity check only)

- [ ] **Step 1: Try the binary against `TwistingBar` with offline mode.**

```bash
cd /home/ziqiu/work/RealSim_py/realsim_py
./build/bin/RealSim simulation/config/CudaTests/TwistingBar/scene.json --offline 2>&1 | tail -20
```

Expected: prints per-frame timing lines and exits at `frame >= scene.json:"stop"` value (810 frames). If it complains about `--offline` not being a flag, see Step 2.

- [ ] **Step 2 (if --offline is rejected): Find the actual offline mode flag.**

```bash
./build/bin/RealSim --help 2>&1 | head -40
```
Use whatever flag the help output names (likely `--offline=true`, `--no-gui`, or via `"offline": true` in scene.json — `SqueezingBall/scene.json` already sets this).

- [ ] **Step 3: Locate the per-step timing in the output.**

```bash
./build/bin/RealSim simulation/config/CudaTests/TwistingBar/scene.json --offline 2>&1 | grep -iE "frame|step|ms" | head -20
```

Note the exact format of the timing line (e.g. `frame 42 ... step_time 5.21 ms`). This is what Task 0.2 will regex-match.

### Task 0.2: Baseline driver script

**Files:**
- Create: `scripts/realsim_baseline/run_one.py`
- Create: `scripts/realsim_baseline/run_all.py`
- Create: `scripts/realsim_baseline/__init__.py` (empty)

- [ ] **Step 1: Define the demo registry as a Python module.**

Create `scripts/realsim_baseline/__init__.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""RealSim CudaTests baseline drivers."""
```

- [ ] **Step 2: Define the demo registry.**

Add to `scripts/realsim_baseline/__init__.py`:
```python
from pathlib import Path

REALSIM_ROOT = Path("/home/ziqiu/work/RealSim_py/realsim_py")

DEMOS: dict[str, dict] = {
    "TwistingBar": {"scene": "simulation/config/CudaTests/TwistingBar/scene.json", "expected_frames": 810},
    "TwistingBarNH": {"scene": "simulation/config/CudaTests/TwistingBarNH/scene.json", "expected_frames": 810},
    "StretchingCloth": {"scene": "simulation/config/CudaTests/StretchingCloth/scene.json"},
    "PullingWooper": {"scene": "simulation/config/CudaTests/PullingWooper/scene.json", "expected_frames": 500},
    "CrossingGingerbreadman": {"scene": "simulation/config/CudaTests/CrossingGingerbreadman/scene.json", "expected_frames": 810},
    "SqueezingBall": {"scene": "simulation/config/CudaTests/SqueezingBall/scene.json", "expected_frames": 600},
    "SharpCorner": {"scene": "simulation/config/CudaTests/SharpCorner/scene.json", "expected_frames": 200},
    "ClothOnKnives": {"scene": "simulation/config/CudaTests/ClothOnKnives/scene.json", "expected_frames": 900},
    "ParallelEnvTest": {"scene": "simulation/config/CudaTests/ParallelEnvTest/scene.json", "expected_frames": 300},
    "CableGrabRaptor": {"scene": "simulation/config/CudaTests/CableGrabRaptor/scene.json", "expected_frames": 6000},
}
```

(Fill `StretchingCloth` `expected_frames` after running it once — its `scene.json` has no `stop`.)

- [ ] **Step 3: Single-demo runner with offline injection.**

The RealSim binary at `/home/ziqiu/work/RealSim_py/realsim_py/build/bin/RealSim` is built from `simulation/simulation.cpp` and dispatches on `"offline": <bool>` in the scene.json (see source line 461-471). There is no `--offline` CLI flag. CudaTests scene.json files generally do not set the field, so we inject it into a temp copy. Offline mode auto-exits at `step == simconfig["maxFrame"]` (the realtime `"stop"` field is unrelated). Every `"timer": N` steps (default 50) the binary prints a profile block ending with an ANSI-colored line `Total time step cost = X ms` — that is the per-block-average we record.

Create `scripts/realsim_baseline/run_one.py`:
```python
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
    """Deep-copy a CudaTests scene.json with ``"offline": true`` and ``"maxFrame"`` injected."""
    cfg = json.loads(scene_path.read_text())
    cfg["offline"] = True
    if max_frame is None:
        # Mirror "stop" into "maxFrame" if upstream didn't set it.
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
```

Notes:
- The binary may segfault in its destructor after writing alembic — `returncode != 0` is not a failure if `frames_timed > 0` and ms values look sensible. The driver reports `returncode` so the user can sanity-check.
- Profile blocks emit every `timer` steps (default 50). For `maxFrame=810`, that's ~16 samples — enough for a mean/median. If finer granularity is needed, also inject `"timer": 10` into the temp scene.

- [ ] **Step 4: Smoke-test the runner.**

```bash
cd /home/ziqiu/work/newton
uv run python -m scripts.realsim_baseline.run_one TwistingBar --max-frame 810
```

Expected: JSON with `"status": "ok"`, `mean_ms` around 8–10 ms (TwistingBar at ~125 particles is fast), and `frames_timed` around 16 (810/50). If `"status": "no_timing"`, inspect `stdout_tail` — possibly the binary segfaulted before reaching the first profile block. Re-run with `--max-frame 200` to get a quicker turnaround.

- [ ] **Step 5: Commit.**

```bash
cd /home/ziqiu/work/newton
git add scripts/realsim_baseline/__init__.py scripts/realsim_baseline/run_one.py
git commit -m "Add RealSim CudaTests baseline driver for one demo"
```

### Task 0.3: Run all 10 baselines, persist JSON

**Files:**
- Create: `scripts/realsim_baseline/run_all.py`
- Create: `scripts/cudatest_baselines.json` (output)

- [ ] **Step 1: All-demos driver.**

Create `scripts/realsim_baseline/run_all.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run every CudaTests demo via the RealSim binary; write scripts/cudatest_baselines.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.realsim_baseline import DEMOS
from scripts.realsim_baseline.run_one import run

OUT = Path(__file__).parent.parent / "cudatest_baselines.json"


def main() -> None:
    results: dict[str, dict] = {}
    for demo in DEMOS:
        print(f"[baseline] running {demo} ...", file=sys.stderr, flush=True)
        results[demo] = run(demo)
        print(f"  -> {results[demo].get('status')}  mean_ms={results[demo].get('mean_ms')}", file=sys.stderr)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it.**

```bash
cd /home/ziqiu/work/newton
uv run python -m scripts.realsim_baseline.run_all 2>&1 | tee scripts/realsim_baseline_log.txt
```

Expected: `cudatest_baselines.json` populated. Demos with `status != "ok"` get noted but we move on — Phase 1 only needs `PullingWooper` ok.

- [ ] **Step 3: Verify PullingWooper has a valid row.**

```bash
cd /home/ziqiu/work/newton
python -c "import json; d=json.load(open('scripts/cudatest_baselines.json')); print(d['PullingWooper'])"
```

Expected: `{"demo": "PullingWooper", "status": "ok", "mean_ms": <float>, ...}`. If not "ok", **STOP** — Phase 1 cannot gate without this.

- [ ] **Step 4: Commit.**

```bash
cd /home/ziqiu/work/newton
git add scripts/realsim_baseline/run_all.py scripts/cudatest_baselines.json
git commit -m "Run RealSim CudaTests baselines, persist JSON"
```

### Task 0.4: Shared FBA bench infrastructure

**Files:**
- Create: `scripts/fba_cudatest_bench/__init__.py`
- Create: `scripts/fba_cudatest_bench/medit.py`
- Create: `scripts/fba_cudatest_bench/perf.py`
- Test: `newton/tests/test_fba_cudatest_bench.py`

- [ ] **Step 1: Move the Medit `.mesh` parser into a shared module (so Task 1.3 reuses it).**

Create `scripts/fba_cudatest_bench/__init__.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared bench harness for SolverFBA on CudaTests demos."""
```

Create `scripts/fba_cudatest_bench/medit.py` (copy verbatim from `scripts/fba_twisting_bar_cudatests.py:107-172`, header replaced):
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Medit `.mesh` parser (text format, 1-based tet indices)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_medit_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Medit ``.mesh`` file.

    Args:
        path: Path to ``foo_volume_NP.mesh``.

    Returns:
        ``(verts, tets)`` where ``verts`` is ``(V, 3) float64`` and ``tets`` is
        ``(T, 4) int32`` with zero-based indices.
    """
    vertices: list[tuple[float, float, float]] = []
    tets: list[tuple[int, int, int, int]] = []
    state: str | None = None
    n_verts = 0
    n_tets = 0
    vert_count = 0
    tet_count_read = 0

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Vertices"):
                state = "vcount"
                continue
            if line.startswith("Tetrahedra"):
                state = "tcount"
                continue
            if line in ("End", "End\n"):
                break
            if line.startswith(("MeshVersionFormatted", "Dimension", "Triangles", "Edges", "Normals")):
                state = "skip_section" if line.startswith(("Triangles", "Edges", "Normals")) else "header"
                continue

            if state == "vcount":
                n_verts = int(line)
                state = "verts"
                continue
            if state == "tcount":
                n_tets = int(line)
                state = "tets"
                continue
            if state == "verts" and vert_count < n_verts:
                parts = line.split()
                vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                vert_count += 1
                if vert_count == n_verts:
                    state = None
                continue
            if state == "tets" and tet_count_read < n_tets:
                parts = line.split()
                tets.append((int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1))
                tet_count_read += 1
                if tet_count_read == n_tets:
                    state = None
                continue

    return np.array(vertices, dtype=np.float64), np.array(tets, dtype=np.int32)
```

- [ ] **Step 2: Perf row writer.**

Create `scripts/fba_cudatest_bench/perf.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""FBA per-demo perf row, comparable to scripts/cudatest_baselines.json."""

from __future__ import annotations

import json
from pathlib import Path

ROW_PATH = Path(__file__).parent.parent / "fba_cudatest_rows.json"


def record_row(demo: str, row: dict) -> None:
    """Insert/update a per-demo perf row in ``scripts/fba_cudatest_rows.json``."""
    data: dict[str, dict] = {}
    if ROW_PATH.exists():
        data = json.loads(ROW_PATH.read_text())
    data[demo] = row
    ROW_PATH.write_text(json.dumps(data, indent=2))


def stats_from_times_ms(times_ms: list[float]) -> dict:
    """Compute ``mean_ms``, ``median_ms``, ``p95_ms`` from a list of per-step times in ms."""
    n = len(times_ms)
    if n == 0:
        return {"frames_timed": 0}
    sorted_t = sorted(times_ms)
    return {
        "frames_timed": n,
        "mean_ms": round(sum(sorted_t) / n, 3),
        "median_ms": round(sorted_t[n // 2], 3),
        "p95_ms": round(sorted_t[min(n - 1, int(0.95 * n))], 3),
    }
```

- [ ] **Step 3: Test the Medit parser against a known mesh.**

Create `newton/tests/test_fba_cudatest_bench.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the FBA CudaTests bench harness utilities."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.fba_cudatest_bench.medit import load_medit_mesh
from scripts.fba_cudatest_bench.perf import stats_from_times_ms

WOOPER = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/wooper_volume_5325P.mesh")


class MeditParserTests(unittest.TestCase):
    def test_wooper_dims(self) -> None:
        if not WOOPER.exists():
            self.skipTest(f"wooper mesh not present at {WOOPER}")
        verts, tets = load_medit_mesh(WOOPER)
        self.assertEqual(verts.shape[1], 3)
        self.assertEqual(verts.shape[0], 5325)
        self.assertEqual(tets.shape[1], 4)
        # Zero-based: no index >= verts count, none < 0.
        self.assertGreaterEqual(int(tets.min()), 0)
        self.assertLess(int(tets.max()), verts.shape[0])


class PerfStatsTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(stats_from_times_ms([]), {"frames_timed": 0})

    def test_basic(self) -> None:
        s = stats_from_times_ms([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(s["frames_timed"], 5)
        self.assertEqual(s["mean_ms"], 3.0)
        self.assertEqual(s["median_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the tests.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_fba_cudatest_bench
```

Expected: 3 tests pass (or `test_wooper_dims` skipped if mesh missing).

- [ ] **Step 5: Commit.**

```bash
cd /home/ziqiu/work/newton
git add scripts/fba_cudatest_bench/__init__.py scripts/fba_cudatest_bench/medit.py scripts/fba_cudatest_bench/perf.py newton/tests/test_fba_cudatest_bench.py
git commit -m "Add shared FBA CudaTests bench utilities"
```

---

## Phase 1: PullingWooper

### Task 1.1: SolverFBA — NSN iter default 20→10, λ-cap parameter

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py:518,538,822-895` (NSN call sites + signatures)
- Modify: `newton/_src/solvers/fba/solver_fba.py` (constructor + `__init__` to accept `nsn_iterations` and `lambda_cap`)
- Modify: `newton/tests/test_solver_fba.py` (add coverage)

Rationale: RealSim's `constraintsolver.iterations: 10` and `maxforce: 100` (ClothOnKnives/SharpCorner) / `1E+12` (default) should be honored. We make `nsn_iterations=10` the default and add `lambda_cap=None` (no cap) as the default — so PullingWooper (`maxforce=1E+12`) remains uncapped.

- [ ] **Step 1: Write the failing test first.**

Append to `newton/tests/test_solver_fba.py`:
```python
class SolverFBAConstructorOptionsTests(unittest.TestCase):
    """Constructor exposes NSN iter count and lambda cap."""

    def _tiny_cloth_model(self):
        import newton
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.1, -0.1, 0.2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=3,
            dim_y=3,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.05,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
        )
        return builder.finalize()

    def test_default_nsn_iter_is_10(self) -> None:
        from newton.solvers import SolverFBA
        model = self._tiny_cloth_model()
        solver = SolverFBA(model)
        self.assertEqual(solver.nsn_iterations, 10)

    def test_nsn_iter_overridable(self) -> None:
        from newton.solvers import SolverFBA
        model = self._tiny_cloth_model()
        solver = SolverFBA(model, nsn_iterations=25)
        self.assertEqual(solver.nsn_iterations, 25)

    def test_lambda_cap_default_is_none(self) -> None:
        from newton.solvers import SolverFBA
        model = self._tiny_cloth_model()
        solver = SolverFBA(model)
        self.assertIsNone(solver.lambda_cap)

    def test_lambda_cap_settable(self) -> None:
        from newton.solvers import SolverFBA
        model = self._tiny_cloth_model()
        solver = SolverFBA(model, lambda_cap=100.0)
        self.assertEqual(solver.lambda_cap, 100.0)
```

- [ ] **Step 2: Run to confirm failure.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k SolverFBAConstructorOptions
```

Expected: 4 FAILs with `AttributeError: 'SolverFBA' object has no attribute 'nsn_iterations'`.

- [ ] **Step 3: Add constructor parameters.**

Find the `__init__` signature in `newton/_src/solvers/fba/solver_fba.py`. Add two parameters before the trailing kwargs (preserve existing order):

```python
        nsn_iterations: int = 10,
        lambda_cap: float | None = None,
```

In the `__init__` body, store them:
```python
        self.nsn_iterations = int(nsn_iterations)
        self.lambda_cap = lambda_cap
```

- [ ] **Step 4: Plumb `nsn_iterations` into the NSN call sites.**

Edit `newton/_src/solvers/fba/solver_fba.py:518`:
```python
                    lam = self._solve_nsn_coulomb(W, r, self._contact_mu_h[:M], max_iters=self.nsn_iterations)
```

Edit `newton/_src/solvers/fba/solver_fba.py:538`:
```python
                    lam = self._solve_nsn_unilateral(W, r, max_iters=self.nsn_iterations)
```

- [ ] **Step 5: Apply `lambda_cap` after each NSN solve.**

In both `_solve_nsn_coulomb` and `_solve_nsn_unilateral` (after the loop, before `return lam`), add:
```python
        if self.lambda_cap is not None:
            np.clip(lam, -self.lambda_cap, self.lambda_cap, out=lam)
```

- [ ] **Step 6: Re-run the failing test, confirm pass.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k SolverFBAConstructorOptions
```

Expected: 4 PASS.

- [ ] **Step 7: Re-run the full FBA test suite to confirm no regressions.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_solver_fba
```

Expected: all 73 prior tests + 4 new = 77 PASS. If any prior test was hardcoded to expect 20 NSN iters and silently relied on extra iters for convergence, lower the test's tolerance or override `nsn_iterations=20` in that test — do NOT raise the default back.

- [ ] **Step 8: Commit.**

```bash
cd /home/ziqiu/work/newton
git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "Make SolverFBA NSN iteration count and lambda cap configurable"
```

### Task 1.2: Pin-region selector helper

**Files:**
- Create: `scripts/fba_cudatest_bench/pin.py`
- Test: `newton/tests/test_fba_cudatest_bench.py` (extend)

RealSim pin configs use AABBs in object-local coordinates. Reuse pattern.

- [ ] **Step 1: Add the test.**

Extend `newton/tests/test_fba_cudatest_bench.py`:
```python
class PinSelectorTests(unittest.TestCase):
    def test_aabb_selects_inside_only(self) -> None:
        import numpy as np
        from scripts.fba_cudatest_bench.pin import select_in_aabb

        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        idx = select_in_aabb(verts, lo=(-0.1, -0.1, -0.1), hi=(0.6, 0.6, 0.6))
        np.testing.assert_array_equal(idx, np.array([0, 3], dtype=np.int32))
```

- [ ] **Step 2: Run to confirm failure.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k PinSelector
```

Expected: FAIL `ModuleNotFoundError: scripts.fba_cudatest_bench.pin`.

- [ ] **Step 3: Implement.**

Create `scripts/fba_cudatest_bench/pin.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pin selection helpers — AABB in object-local coords -> particle index list."""

from __future__ import annotations

import numpy as np


def select_in_aabb(
    verts: np.ndarray,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
) -> np.ndarray:
    """Return ``int32`` indices of vertices whose coordinates are within ``[lo, hi]``.

    Args:
        verts: ``(V, 3)`` vertex positions.
        lo: Lower AABB bound ``(x, y, z)``.
        hi: Upper AABB bound ``(x, y, z)``.

    Returns:
        ``(K,)`` array of selected vertex indices in ascending order.
    """
    lo_arr = np.asarray(lo, dtype=verts.dtype)
    hi_arr = np.asarray(hi, dtype=verts.dtype)
    mask = np.all((verts >= lo_arr) & (verts <= hi_arr), axis=1)
    return np.flatnonzero(mask).astype(np.int32)
```

- [ ] **Step 4: Verify pass.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k PinSelector
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
cd /home/ziqiu/work/newton
git add scripts/fba_cudatest_bench/pin.py newton/tests/test_fba_cudatest_bench.py
git commit -m "Add AABB pin-region selector for CudaTests-derived scenes"
```

### Task 1.3: PullingWooper demo script (first cut, no perf tuning)

**Files:**
- Create: `scripts/fba_demo4_pulling_wooper.py`

The wooper scene parameters from `wooper_5k.json` + `scene.json`:
- Mesh: `wooper_volume_5325P.mesh` (5325 particles, ~tet count from file)
- Transformation: `trans=(0, 4, 0)`, rot Z=-90deg, scale=1.0
- `young=1e7`, `poisson=0.3`, `obj_mass=1000` (RealSim density model — apply via `density` arg)
- Pin AABB in object-local: `box=[-2.5, 2, -0.3, -1.5, 4, 0.3]`
- Dynamic pull: `direction=(0, -1, 0)`, `vel=2.0`, `maxlength=10.0`
- Cylinders: cyl0 base=(0, -1, -1.3) axis=(1,0,0) r=1; cyl1 base=(0, -1, 1.3) axis=(1,0,0) r=1
- `timestep=0.01`, gravity=0, `stop=500` frames

Lamé params from Young/Poisson (Newton uses `k_mu`, `k_lambda` directly):
- `mu = E / (2 * (1+nu))` = 1e7 / 2.6 ≈ 3.846e6
- `lam = E*nu / ((1+nu)*(1-2*nu))` = 1e7 * 0.3 / (1.3 * 0.4) ≈ 5.769e6

- [ ] **Step 1: Create the script.**

Create `scripts/fba_demo4_pulling_wooper.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Demo 4: PullingWooper — CudaTests reproduction.

Pulls a 5325-particle tet wooper through two static cylinders. Single
PULLING pin action driven externally via ``SolverFBA.set_pin_targets``.

Outputs PNGs + perf row to ``scripts/contact_demos_out/demo4/``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA
from scripts.fba_cudatest_bench.medit import load_medit_mesh
from scripts.fba_cudatest_bench.perf import record_row, stats_from_times_ms
from scripts.fba_cudatest_bench.pin import select_in_aabb

# ---------------------------------------------------------------------------
# Constants from RealSim CudaTests/PullingWooper
# ---------------------------------------------------------------------------
MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/volume/wooper_volume_5325P.mesh")
DT = 0.01
TOTAL_FRAMES = 500
PD_ITERATIONS = 5
NSN_ITERATIONS = 10
GRAVITY = 0.0  # RealSim PullingWooper sets gravity = 0

YOUNG = 1.0e7
POISSON = 0.3
OBJ_MASS = 1000.0  # RealSim treats this as total object mass — convert to density below.

# Transformation from wooper_5k.json: rotation Z=-90deg, trans=(0, 4, 0), scale=1.0
ROT_Z_DEG = -90.0
TRANS = np.array([0.0, 4.0, 0.0])

# Pin AABB (object-local).
PIN_LO = (-2.5, 2.0, -0.3)
PIN_HI = (-1.5, 4.0, 0.3)
# Single PULLING action.
PIN_DIR = np.array([0.0, -1.0, 0.0])
PIN_VEL = 2.0
PIN_MAXLENGTH = 10.0

# Cylinders.
CYL_RADIUS = 1.0
CYL_LEN = 6.0  # cosmetic length; collision is along axis
CYL0_BASE = np.array([0.0, -1.0, -1.3])
CYL1_BASE = np.array([0.0, -1.0, 1.3])
CYL_AXIS = np.array([1.0, 0.0, 0.0])

OUT_DIR = Path(__file__).parent / "contact_demos_out" / "demo4"
SNAPSHOT_FRAMES = [0, 50, 150, 300, 450, 499]


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform_verts(verts_local: np.ndarray) -> np.ndarray:
    return verts_local @ rot_z(ROT_Z_DEG).T + TRANS


def lame_from_young_poisson(young: float, poisson: float) -> tuple[float, float]:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def cyl_xform(base: np.ndarray, axis: np.ndarray) -> wp.transform:
    """Build a Newton transform that orients a cylinder along ``axis``.

    Newton's ``add_shape_cylinder`` extends along local +Z (see
    ``newton/_src/sim/builder.py:5959``). We compute the rotation that maps
    +Z to ``axis``.
    """
    z = np.array([0.0, 0.0, 1.0])
    a = axis / np.linalg.norm(axis)
    v = np.cross(z, a)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, a))
    if s < 1e-9:
        if c > 0:
            q = wp.quat_identity()
        else:
            q = wp.quat(1.0, 0.0, 0.0, 0.0)  # 180 deg about X
    else:
        ang = math.atan2(s, c)
        axis_n = v / s
        half = ang * 0.5
        sh = math.sin(half)
        q = wp.quat(axis_n[0] * sh, axis_n[1] * sh, axis_n[2] * sh, math.cos(half))
    return wp.transform(wp.vec3(*base.tolist()), q)


def build_model() -> tuple:
    verts_local, tets = load_medit_mesh(MESH_PATH)
    verts_world = transform_verts(verts_local)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=GRAVITY)

    # Mass per particle from total mass (RealSim convention).
    # Use add_soft_mesh for direct vertex/tet control. Density = total_mass / total_volume.
    # add_soft_mesh accepts (verts, indices, density). It computes per-particle mass.
    total_vol = 0.0
    for t in tets:
        v0, v1, v2, v3 = (verts_world[i] for i in t)
        total_vol += abs(np.dot(np.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
    density = OBJ_MASS / max(total_vol, 1e-12)

    mu, lam = lame_from_young_poisson(YOUNG, POISSON)
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*v.tolist()) for v in verts_world],
        indices=tets.reshape(-1).tolist(),
        density=density,
        k_mu=mu,
        k_lambda=lam,
        k_damp=0.0,
        add_surface_mesh_edges=False,  # pure tet body, no bending edges from surface
    )

    # Cylinders (static obstacles).
    builder.add_shape_cylinder(body=-1, xform=cyl_xform(CYL0_BASE, CYL_AXIS), radius=CYL_RADIUS, half_height=CYL_LEN / 2.0)
    builder.add_shape_cylinder(body=-1, xform=cyl_xform(CYL1_BASE, CYL_AXIS), radius=CYL_RADIUS, half_height=CYL_LEN / 2.0)

    # Pin selection in object-local frame.
    pin_idx = select_in_aabb(verts_local, PIN_LO, PIN_HI)

    model = builder.finalize()

    # Mark pinned particles as inv_mass=0.
    inv_m = model.particle_inv_mass.numpy()
    inv_m[pin_idx] = 0.0
    model.particle_inv_mass.assign(inv_m)

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = pipeline.contacts()
    return model, pipeline, contacts, verts_world, pin_idx


def render_frame(ax, q: np.ndarray, frame: int) -> None:
    ax.clear()
    step = max(1, len(q) // 500)
    pts = q[::step]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 1], cmap="viridis")
    ax.set_title(f"PullingWooper — frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-10, 6)
    ax.set_zlim(-3, 3)


def run() -> dict:
    model, pipeline, contacts, verts_world, pin_idx = build_model()
    print(f"  particles={model.particle_count}  tets={len(model.tet_indices.numpy()) // 4}  pinned={len(pin_idx)}")

    solver = SolverFBA(
        model,
        iterations=PD_ITERATIONS,
        nsn_iterations=NSN_ITERATIONS,
        friction=False,
        stretching_model="neohookean",
    )
    s_in = model.state()
    s_out = model.state()

    pin_init = verts_world[pin_idx].copy()
    targets = verts_world.copy().astype(np.float32)

    snapshots: dict[int, np.ndarray] = {}
    step_times: list[float] = []
    pulled = 0.0

    for frame in range(TOTAL_FRAMES + 1):
        if frame in SNAPSHOT_FRAMES:
            snapshots[frame] = s_in.particle_q.numpy().copy()
        if frame == TOTAL_FRAMES:
            break

        delta = PIN_VEL * DT
        if pulled + delta > PIN_MAXLENGTH:
            delta = max(0.0, PIN_MAXLENGTH - pulled)
        pulled += delta
        targets[pin_idx] = pin_init + (PIN_DIR * pulled).astype(np.float32)
        solver.set_pin_targets(targets)

        t0 = time.perf_counter()
        s_in.clear_forces()
        pipeline.collide(s_in, contacts)
        solver.step(s_in, s_out, None, contacts, DT)
        s_in, s_out = s_out, s_in
        step_times.append(1000.0 * (time.perf_counter() - t0))

    q_final = s_in.particle_q.numpy()
    finite_ok = bool(np.all(np.isfinite(q_final)))
    stats = stats_from_times_ms(step_times)
    stats["stable"] = finite_ok
    stats["min_y"] = float(q_final[:, 1].min())
    stats["pulled"] = pulled
    return {"snapshots": snapshots, "stats": stats}


def main() -> None:
    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Demo 4: PullingWooper")
    print("=" * 60)

    result = run()
    stats = result["stats"]
    print(f"  mean_ms={stats['mean_ms']:.2f}  median={stats['median_ms']:.2f}  p95={stats['p95_ms']:.2f}")
    print(f"  stable={stats['stable']}  min_y={stats['min_y']:.3f}  pulled={stats['pulled']:.2f}")

    fig = plt.figure(figsize=(4 * len(result["snapshots"]), 4))
    fig.suptitle("Demo 4: PullingWooper")
    for col, (frame, q) in enumerate(sorted(result["snapshots"].items())):
        ax = fig.add_subplot(1, len(result["snapshots"]), col + 1, projection="3d")
        render_frame(ax, q, frame)
    plt.tight_layout()
    out = OUT_DIR / "pulling_wooper_strip.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  snapshot strip: {out}")

    record_row("PullingWooper", stats)
    print(f"  perf row saved to scripts/fba_cudatest_rows.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: First run.**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_demo4_pulling_wooper.py
```

Expected: prints `particles=5325`, prints a per-step mean, writes a PNG strip, writes a perf row. **If `add_soft_mesh` API mismatches**, consult `newton/_src/sim/model_builder.py` for the exact signature and adjust the `add_soft_mesh(...)` call. **If `add_shape_cylinder` orientation looks wrong in the snapshot** (cylinder ends visible from wrong side), fix the `cyl_xform` axis convention.

- [ ] **Step 3: Sanity-check the snapshot.**

Open `scripts/contact_demos_out/demo4/pulling_wooper_strip.png` and confirm:
- Frame 0: wooper is rotated and translated to start position
- Frame 50-150: pin region moves down through the cylinders
- Frame 300-499: wooper is fully pulled past the cylinders, deformed but stable
- No NaN/explosion (stable=True printed)

If visual is clearly wrong (mesh fragments, explosion, wooper drifting away from cylinders), STOP and debug before measurement gating.

- [ ] **Step 4: Commit the first-cut demo.**

```bash
cd /home/ziqiu/work/newton
git add scripts/fba_demo4_pulling_wooper.py
git commit -m "Add CudaTests PullingWooper demo for SolverFBA (first cut)"
```

### Task 1.4: Initial perf gate against RealSim baseline

**Files:**
- Read: `scripts/cudatest_baselines.json` (from Task 0.3)
- Read: `scripts/fba_cudatest_rows.json` (written by Task 1.3 run)

- [ ] **Step 1: Compare.**

```bash
cd /home/ziqiu/work/newton
python -c "
import json
b = json.load(open('scripts/cudatest_baselines.json'))['PullingWooper']
f = json.load(open('scripts/fba_cudatest_rows.json'))['PullingWooper']
print(f'RealSim mean_ms = {b[\"mean_ms\"]:.2f}')
print(f'FBA     mean_ms = {f[\"mean_ms\"]:.2f}')
print(f'ratio (FBA/RealSim) = {f[\"mean_ms\"]/b[\"mean_ms\"]:.2f}')
"
```

- [ ] **Step 2: Decide branch.**

| FBA/RealSim ratio | Branch |
|---|---|
| ≤ 1.0 | **Done.** Skip Tasks 1.5, 1.6. Jump to Task 1.7 (visual polish + commit perf row). |
| 1.0 < r ≤ 2.0 | Do **Task 1.5** (CUDA Graph capture). After it, re-measure; if still > 1.0×, do **Task 1.6** (λ warm-start). |
| > 2.0 | **STOP.** Something's wrong beyond launch overhead — check NSN iteration ratio (FBA may still be doing 20 iters; verify `nsn_iterations=10` made it through), or factor rebuild on every frame (check `_dt_setup` cache). Diagnose before tuning. |

### Task 1.5 (CONDITIONAL): CUDA Graph capture of the step body

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py` (`step` method)
- Test: `newton/tests/test_solver_fba.py` (extend)

**Only do this task if Task 1.4 returned ratio > 1.0.** Goal: amortize Warp kernel launch overhead via `wp.ScopedCapture` of the inner-iteration kernels.

- [ ] **Step 1: Add a failing test that asserts graph mode preserves identical outputs.**

Append to `newton/tests/test_solver_fba.py`:
```python
class SolverFBAGraphCaptureTests(unittest.TestCase):
    """CUDA graph capture must not change numerical output."""

    def test_graph_mode_matches_eager(self) -> None:
        import newton
        from newton.solvers import SolverFBA

        def build() -> tuple:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
            builder.add_cloth_grid(
                pos=wp.vec3(-0.1, -0.1, 0.2), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=4, dim_y=4, cell_x=0.05, cell_y=0.05, mass=0.05,
                tri_ke=1.0e4, tri_ka=0.0, tri_kd=0.0,
            )
            return builder.finalize()

        m_eager = build()
        m_graph = build()
        s_e_in, s_e_out = m_eager.state(), m_eager.state()
        s_g_in, s_g_out = m_graph.state(), m_graph.state()

        eager = SolverFBA(m_eager, iterations=5, use_cuda_graph=False)
        graph = SolverFBA(m_graph, iterations=5, use_cuda_graph=True)

        for _ in range(5):
            eager.step(s_e_in, s_e_out, None, None, 1.0 / 60.0)
            graph.step(s_g_in, s_g_out, None, None, 1.0 / 60.0)
            s_e_in, s_e_out = s_e_out, s_e_in
            s_g_in, s_g_out = s_g_out, s_g_in

        np.testing.assert_allclose(
            s_e_in.particle_q.numpy(),
            s_g_in.particle_q.numpy(),
            atol=1e-5,
            err_msg="graph mode diverged from eager",
        )
```

- [ ] **Step 2: Run to confirm failure.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k SolverFBAGraphCapture
```

Expected: FAIL with `unexpected keyword argument 'use_cuda_graph'`.

- [ ] **Step 3: Add `use_cuda_graph` to constructor.**

In `SolverFBA.__init__`, add `use_cuda_graph: bool = False`. Store as `self.use_cuda_graph`.

- [ ] **Step 4: Capture the inner Warp body of `step`.**

The capturable region is the per-PD-outer-iter loop (everything inside the `for it in range(self.iterations)` block — local projection + RHS assembly + linear solve). NSN is host-side numpy and cannot be captured; treat it as the seam.

Sketch (drop-in for the existing `step` method, preserving the contact branches):
```python
        if self.use_cuda_graph and self._contact_count == 0 and self._graph is None:
            with wp.ScopedCapture(device=self._device) as cap:
                self._step_inner(state_in, state_out, dt)
            self._graph = cap.graph
        if self._graph is not None and self._contact_count == 0:
            wp.capture_launch(self._graph)
        else:
            self._step_inner(state_in, state_out, dt)
```

(Refactor the existing inner body of `step` into `_step_inner(self, state_in, state_out, dt)`. Stage A/B contact paths bypass the cached graph because the contact set changes per frame.)

Add `self._graph = None` to `__init__`. Invalidate in `notify_model_changed`.

- [ ] **Step 5: Re-run tests.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k SolverFBAGraphCapture
```

Expected: PASS.

- [ ] **Step 6: Wire `use_cuda_graph=True` into the wooper demo.**

In `scripts/fba_demo4_pulling_wooper.py:run()`, change the `SolverFBA(...)` call to add `use_cuda_graph=True`. Re-run the demo. Compare new `mean_ms` to baseline.

- [ ] **Step 7: Commit.**

```bash
cd /home/ziqiu/work/newton
git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py scripts/fba_demo4_pulling_wooper.py
git commit -m "Add CUDA graph capture for SolverFBA non-contact step path"
```

### Task 1.6 (CONDITIONAL): λ warm-start across PD outer iters

**Only do this task if Task 1.4 ratio > 1.0 even after Task 1.5.**

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py` (`__init__`, `update_contacts`, `_solve_nsn_*`, `step`)
- Test: `newton/tests/test_solver_fba.py`

NSN currently restarts from `lam = 0` each PD outer iter. Reuse the previous outer iter's `lam` as the initial guess. Acceptance is **demo `mean_ms` drops** with `use_warm_start=True` vs `False`. (No standalone unit test — the equivalence is mathematical: warm-start with the converged `lam_prev` plus zero residual leaves the solution unchanged; this is exercised by Task 1.1's regression tests.)

- [ ] **Step 1: Add `use_warm_start` parameter.**

In `SolverFBA.__init__`, add `use_warm_start: bool = True`. Store as `self.use_warm_start = bool(use_warm_start)`. Initialise `self._lam_prev_unilateral: np.ndarray | None = None` and `self._lam_prev_coulomb: np.ndarray | None = None`.

- [ ] **Step 2: Invalidate the warm-start state when the contact set changes.**

In `update_contacts`, after the new `self._contact_count` is set, compare to the previous count. If different, set both `_lam_prev_*` to `None`. Same when `_ensure_contact_buffers` grows the buffers.

- [ ] **Step 3: Plumb `lam_init` into the NSN solvers.**

Add `lam_init: np.ndarray | None = None` to `_solve_nsn_unilateral` and `_solve_nsn_coulomb`.

In `_solve_nsn_unilateral` (around line 841), replace:
```python
        M = len(r)
        lam = np.zeros(M, dtype=np.float64)
```
with:
```python
        M = len(r)
        if lam_init is not None and lam_init.shape == (M,):
            lam = lam_init.astype(np.float64, copy=True)
        else:
            lam = np.zeros(M, dtype=np.float64)
```

In `_solve_nsn_coulomb` (around line 878), replace:
```python
        M = len(mu)
        lam = np.zeros(3 * M, dtype=np.float64)
```
with:
```python
        M = len(mu)
        if lam_init is not None and lam_init.shape == (3 * M,):
            lam = lam_init.astype(np.float64, copy=True)
        else:
            lam = np.zeros(3 * M, dtype=np.float64)
```

- [ ] **Step 4: Pass `lam_prev` at the call sites and save the result back.**

At line 518 (Coulomb call):
```python
                    lam = self._solve_nsn_coulomb(
                        W, r, self._contact_mu_h[:M],
                        max_iters=self.nsn_iterations,
                        lam_init=self._lam_prev_coulomb if self.use_warm_start else None,
                    )
                    self._lam_prev_coulomb = lam.copy()
```

At line 538 (unilateral call) — same pattern with `_lam_prev_unilateral`.

- [ ] **Step 5: Regression check — full FBA test suite.**

```bash
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_solver_fba
```

Expected: all tests still pass. If a contact test fails because warm-start magnified a pre-existing numerical artifact, set `use_warm_start=False` only in that specific test, do NOT change the default.

- [ ] **Step 6: Wire `use_warm_start=True` (default) into the wooper demo and re-measure.**

The wooper demo already gets `use_warm_start=True` for free since it's the default. Re-run:
```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_demo4_pulling_wooper.py
```

Compare ratio against the post-Task-1.5 number.

- [ ] **Step 7: Commit.**

```bash
cd /home/ziqiu/work/newton
git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "Warm-start NSN lambda across SolverFBA PD outer iters"
```

### Task 1.7: Final perf row + snapshot polish + README

**Files:**
- Modify: `scripts/contact_demos_out/README.md`
- Read: `scripts/fba_cudatest_rows.json`

- [ ] **Step 1: Append the PullingWooper row to the demos README.**

Add a new section to `scripts/contact_demos_out/README.md`:
```markdown
## Demo 4 — PullingWooper (CudaTests reproduction)

`scripts/fba_demo4_pulling_wooper.py`

5325-particle tet wooper (NH, E=1e7, ν=0.3), 500 frames at dt=0.01,
single PULLING action (dir=-Y, vel=2 m/s, maxlength=10 m), 2 static cylinders
(r=1, mu=0), 5 PD outer × 10 NSN inner, gravity=0.

| Solver        | mean ms/step | median | p95  | Notes                       |
|---------------|--------------|--------|------|-----------------------------|
| RealSim       | (see baseline) | …    | …    | NSN_CUDA + PCR_CUDA         |
| SolverFBA     | (see row)    | …      | …    | matches `nsn_iter=10`       |

Fill numbers from `scripts/cudatest_baselines.json["PullingWooper"]` and
`scripts/fba_cudatest_rows.json["PullingWooper"]`.

Acceptance:
- Visual: wooper pulls through both cylinders, deforms around them, no penetration
- Stability: `stable=True`, `pulled ≈ 5–10 m` at final frame
- Perf: ms/step ≤ RealSim baseline
```

- [ ] **Step 2: Update the progress doc.**

Edit `docs/superpowers/fba-dev-progress.md` — add an entry marking Phase 1 (PullingWooper) complete with the perf numbers.

- [ ] **Step 3: Final commit.**

```bash
cd /home/ziqiu/work/newton
git add scripts/contact_demos_out/README.md docs/superpowers/fba-dev-progress.md
git commit -m "Document PullingWooper demo and perf row"
```

- [ ] **Step 4: Run pre-commit checks.**

```bash
cd /home/ziqiu/work/newton
uvx pre-commit run -a
```

Expected: no errors. Fix anything reported (lint, formatting) and amend the relevant earlier commit if needed.

---

## Out of Scope (Deferred to Phase 2+)

These show up in PullingWooper's scene.json but do **not** need to be implemented for P1 to be accepted:

- `output_abc` cross-check trajectory (P0 baseline gives `ms/step`; we don't need per-frame trajectory match per the spec's qualitative acceptance criterion)
- Per-shape friction (cylinders have `mu=0` in PullingWooper)
- ROLLING pin (PullingWooper has only PULLING)
- λ warm-start across **substeps** within a single frame (no substeps for FBA's single 0.01s step)

## Acceptance Checklist

- [ ] `scripts/cudatest_baselines.json` exists, `PullingWooper.status == "ok"`
- [ ] `scripts/fba_cudatest_rows.json` exists, `PullingWooper.mean_ms` ≤ baseline `mean_ms`
- [ ] `scripts/contact_demos_out/demo4/pulling_wooper_strip.png` shows the wooper threading through both cylinders without penetration or explosion
- [ ] All FBA unit tests pass: `uv run --extra dev -m newton.tests -k test_solver_fba` and `-k test_fba_cudatest_bench`
- [ ] `uvx pre-commit run -a` clean

---

## Audit (Task Q, 2026-05-16) — Queued correctness fixes

Comprehensive read-only audit of FBA vs RealSim numerical components turned up 2 actionable correctness divergences. Both are queued; do them **before** Task P (isodof) so the perf rerun rides on the corrected math.

### Task H' — Fix isometric bending scatter formula

**File**: `newton/_src/solvers/fba/kernels.py:1049-1067` (bending kernel scatter).

**Divergence**:
- Newton currently scatters `P = q[a] · (q · x_ref)` — minimises `‖(q·x_cur) − (q·x_ref)‖²`. Wrong for non-flat rest: it tries to restore the rest curvature **vector** (direction + magnitude), so any deviation in direction (e.g. cloth folding the "wrong way" first) is penalised even though the formulation should only care about curvature magnitude.
- RealSim scatters `P = q[a] · ê · ‖q·x_ref‖` where `ê = (q·x_cur).normalized()` — minimises `‖(q·x_cur) − ê·‖q·x_ref‖‖²`. Direction follows current curvature, magnitude pulls back to rest.

**Why it matters now**: PullingWooper and the existing demos use flat-rest cloth so the formulas agree. **SharpCorner (bending=5), ClothOnKnives (bending=10), and ParallelEnvTest (bending=20)** also start flat — they will agree initially. But once any of these cloths develops nontrivial curvature during simulation, the per-step behaviour diverges. CableGrabRaptor with its draped/curved cable+raptor scene will fail to reproduce RealSim qualitatively unless this is fixed.

**Steps**:

- [ ] **Step 1: Read the current kernel.** `newton/_src/solvers/fba/kernels.py:1021-1067`. Note the loop structure and how `qTxref` (or equivalent) is computed.

- [ ] **Step 2: Read RealSim reference.** `/home/ziqiu/work/RealSim_py/realsim_py/src/Scomponent/integrator/projectivedynamics/PDIsometricBendingEnergy.cpp` lines 101-128 — specifically the `ê = (sum_a q[a]·pos[a]).normalized()` then `scatter_a += w · q[a] · ê · _norm[i]` pattern. `_norm[i]` is the rest curvature magnitude (cached at init).

- [ ] **Step 3: Modify the kernel to match RealSim.** Replace the scatter line so it computes `qTx_cur = Σ q[a]·x_cur[a]`, normalises `ê = qTx_cur / ‖qTx_cur‖` (guard zero norm with epsilon), then scatters `P_a = w · q[a] · ê · ‖q·x_ref‖`. The rest curvature magnitude `‖q·x_ref‖` must be precomputed per-edge at solver setup time (mirror RealSim's `_norm[i]`).

- [ ] **Step 4: Write a regression test.** Build a small curved-rest cloth (e.g. quad bent 30° in the rest pose) under the current Newton kernel; capture trajectory. Then with the fixed kernel; capture trajectory. Assert the fixed-kernel trajectory matches a hand-derived expected (or a 1-step analytical sanity). Don't gate on numerical match to the old kernel — the math is changing intentionally.

- [ ] **Step 5: Run the full FBA suite** and confirm flat-rest tests still pass (they should — formula is equivalent when `‖q·x_ref‖ ≈ 0`).
  ```bash
  uv run --extra dev -m newton.tests -k test_solver_fba
  ```

- [ ] **Step 6: Commit.**
  ```bash
  git add newton/_src/solvers/fba/kernels.py newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
  git commit -m "Match RealSim isometric bending scatter for non-flat rest"
  ```

### Task T' — Warm-start λ across PD outer iters within a step

**Files**: `newton/_src/solvers/fba/solver_fba.py:916, 955` (the `lam = np.zeros(...)` lines inside `_solve_nsn_unilateral` and `_solve_nsn_coulomb`).

**Divergence**:
- Newton resets `lam = np.zeros(...)` at the start of every NSN call, so every PD outer iter starts from `λ = 0`. For PD with `iterations=5` (CudaTests default), the contact impulse only has 5 outer iters to converge — and each iter resets the warm-start.
- RealSim sets `cuda_lambda.setZero` **once per frame** in `prepare_gpu`, then accumulates `λ += dλ` across PD outer iters (`NonSmoothNewton.cpp:158`). λ persists across outer iters within a step.

**Why it matters now**: With low PD iter counts (5 outer × 1 NSN inner today), Newton's contact response under-shoots in the first 2-3 PD outer iters relative to RealSim. PullingWooper still passes stability checks because the wooper's contact is light, but stiffer contacts (SqueezingBall, ClothOnKnives) will show more pronounced lag.

**Steps**:

- [ ] **Step 1: Add `_lam_persistent` buffers.** In `SolverFBA.__init__`, add `self._lam_unilateral_persistent: np.ndarray | None = None` and `self._lam_coulomb_persistent: np.ndarray | None = None`. Initialise to None.

- [ ] **Step 2: Reset on contact-set change.** In `update_contacts`, when the contact set changes (existing detection point), reset both `_lam_*_persistent` to None.

- [ ] **Step 3: Reset at step entry.** At the start of `step()` (after `update_contacts` and before the PD outer loop), reset both persistent λ buffers if they are None — i.e. allocate fresh zeros sized to the current contact count.

  Note: at step boundaries this **does** reset λ to zero. That matches RealSim's `prepare_gpu.setZero` per-frame behaviour. The warm-start is **only across PD outer iters within a step**, not across steps.

- [ ] **Step 4: Plumb `lam_init` into NSN solvers.** Add `lam_init: np.ndarray | None = None` param to `_solve_nsn_unilateral` and `_solve_nsn_coulomb`. Replace `lam = np.zeros(...)` with:
  ```python
  if lam_init is not None and lam_init.shape == (...,):
      lam = lam_init.astype(np.float64, copy=True)
  else:
      lam = np.zeros(..., dtype=np.float64)
  ```
  (Shapes: `(M,)` unilateral, `(3M,)` coulomb — same as Task 1.6 step 3 in this plan, never executed.)

- [ ] **Step 5: At each NSN call site**, pass `lam_init=self._lam_*_persistent` and save the result back. Sites are `solver_fba.py:573, 597` after the most recent edits.

- [ ] **Step 6: Verify all 77 FBA tests pass.**
  ```bash
  uv run --extra dev -m newton.tests -k test_solver_fba
  ```

- [ ] **Step 7: Re-measure PullingWooper.** Sync timing, 500 frames. Expect `mean_ms` modest drop (1-2 fewer PGS iters to converge in practice) AND a behaviour-validity confirmation that `pulled=10.00, stable=True, min_y` unchanged.

- [ ] **Step 8: Commit.**
  ```bash
  git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
  git commit -m "Warm-start contact lambda across PD outer iters in SolverFBA"
  ```

### Then continue with Task P (isodof Schur)

After H' and T' commit, return to **Task P — Isodof-restricted Schur build** (originally drafted in this plan's task list under task #59). Brief:

- Restrict `A⁻¹` compute to the (k × k) block where k = unique contacted DOFs, mirroring RealSim `SparseInverseSolver.cpp:215-339` (`addHAinvHT_gpu` + `cudaSelectSpMM`).
- Form `W = Jh · Wi · Jhᵀ` where `Jh` is J restricted to those columns.
- Cache Wi across PD outer iters (already structured by Task L's cache layer).
- Add `use_isodof: bool = True` constructor flag (already present in solver_fba.py since the user's intermediate edit at agent rejection).
- Cross-frame Wi reuse (RealSim's `_Wi_prev`) is **Phase 5 / optional** — defer unless mean_ms still exceeds RealSim after the within-step path lands.

Acceptance: PullingWooper `mean_ms ≤ 24.08 ms` (RealSim baseline) with all 77 FBA tests passing.

### Known divergences kept on purpose

The audit confirmed Newton intentionally diverges from RealSim on:
- Triangle + tet **Corotational** energy: Newton uses correct symmetric `μ‖σ−I‖² + (λ/2)tr(σ−I)²`; RealSim hits an Eigen `.trace()` bug on 2D/3D Vec (`HyperelasticProblemS.h:173-219` for cloth, `:123-171` for tet). Doc: `docs/superpowers/specs/2026-05-15-fba-corot-realsim-discrepancy.md`. Newton stays correct; spec extended to cover tet too.
- **Coulomb cone shape**: Newton uses circular cone (physically correct); RealSim uses rectangular box clamp (`NonSmoothNewton.cpp:380-408`). Newton stays correct.
- **Mass lumping**: Newton uses standard FE lumping (`density · area / 3` or `density · vol / 4`); RealSim uses uniform `total_mass / N`. Both correct for uniform meshes; Newton stays with FE convention.
