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
