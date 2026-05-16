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
