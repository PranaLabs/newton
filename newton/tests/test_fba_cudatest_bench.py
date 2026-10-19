# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the FBA CudaTests bench harness utilities."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.fba_cudatest_bench.perf import stats_from_times_ms
from scripts.fba_cudatest_bench.pin import select_in_aabb


class PerfStatsTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(stats_from_times_ms([]), {"frames_timed": 0})

    def test_basic(self) -> None:
        s = stats_from_times_ms([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(s["frames_timed"], 5)
        self.assertEqual(s["mean_ms"], 3.0)
        self.assertEqual(s["median_ms"], 3.0)


class PinSelectorTests(unittest.TestCase):
    def test_aabb_selects_inside_only(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
