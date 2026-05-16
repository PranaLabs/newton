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

# Cap long demos to keep total batch under ~30 minutes wall time.
# Each demo's "expected_frames" in DEMOS is the upstream RealSim "stop";
# we override here only when the original count is impractical for a baseline run.
MAX_FRAME_OVERRIDES: dict[str, int] = {
    "CableGrabRaptor": 1000,  # upstream 6000 — too long for a baseline
    "ClothOnKnives": 600,  # upstream 900
}


def main() -> None:
    results: dict[str, dict] = {}
    for demo in DEMOS:
        max_frame = MAX_FRAME_OVERRIDES.get(demo)
        suffix = f" (capped to {max_frame})" if max_frame else ""
        print(f"[baseline] running {demo}{suffix} ...", file=sys.stderr, flush=True)
        results[demo] = run(demo, max_frame=max_frame)
        status = results[demo].get("status")
        mean = results[demo].get("mean_ms")
        print(f"  -> status={status}  mean_ms={mean}", file=sys.stderr, flush=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
