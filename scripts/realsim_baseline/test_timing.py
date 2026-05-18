# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`scripts.realsim_baseline.timing` parser.

Run with:
    uv run --extra dev python -m unittest scripts.realsim_baseline.test_timing
"""

from __future__ import annotations

import unittest

from scripts.realsim_baseline.timing import parse_realsim_stdout

ANSI_BLOCK_NSN = """
\x1b[33m
Printing profile over \x1b[34m50\x1b[33m time steps\x1b[0m
\x1b[33m-------------------------------------------------------------------\x1b[0m
\x1b[32mNumber of constraints over \x1b[34m50\x1b[32m time steps, count = \x1b[34m120.3\x1b[0m
\x1b[32mNumber of constraint solver iteration over \x1b[34m50\x1b[32m times steps, count = \x1b[34m8.5\x1b[0m
\x1b[32mSchur cost = \x1b[34m1.25\x1b[32m ms\x1b[0m
\x1b[32mLocal cost = \x1b[34m0.5\x1b[32m ms\x1b[0m
\x1b[32mLinear solve cost = \x1b[34m2.0\x1b[32m ms\x1b[0m
\x1b[32mAssembly called by \x1b[34m50\x1b[32m times, cost = \x1b[34m0.1\x1b[32m ms\x1b[0m
\x1b[32mCst solve called by \x1b[34m50\x1b[32m times, cost = \x1b[34m0.2\x1b[32m ms\x1b[0m
\x1b[32mCorrection called by \x1b[34m50\x1b[32m times, cost = \x1b[34m0.3\x1b[32m ms\x1b[0m
\x1b[32mTotal Global cost = \x1b[34m0.6\x1b[32m ms\x1b[0m
\x1b[32mFrame without collision detection = \x1b[34m15.0\x1b[32m ms\x1b[0m
\x1b[32mTotal time step cost = \x1b[34m16.5\x1b[32m ms\x1b[0m
"""

PLAIN_BLOCK_NO_NSN = """
Printing profile over 50 time steps
-------------------------------------------------------------------
Local cost = 0.6 ms
Linear solve cost = 2.5 ms
Frame without collision detection = 16.0 ms
Total time step cost = 17.5 ms
"""


class TestParseRealSimStdout(unittest.TestCase):
    def test_nsn_block_parses_all_fields(self) -> None:
        t = parse_realsim_stdout(ANSI_BLOCK_NSN, demo="X", timer_interval=50)
        self.assertEqual(t.demo, "X")
        self.assertEqual(t.n_blocks, 1)
        self.assertEqual(t.n_frames, 50)
        self.assertAlmostEqual(t.n_constraints_mean, 120.3)
        self.assertAlmostEqual(t.cst_solve_iter_mean, 8.5)
        self.assertAlmostEqual(t.schur_ms, 1.25)
        self.assertAlmostEqual(t.local_ms, 0.5)
        self.assertAlmostEqual(t.linear_solve_ms, 2.0)
        self.assertAlmostEqual(t.build_ms, 0.1)
        self.assertAlmostEqual(t.cst_solve_ms, 0.2)
        self.assertAlmostEqual(t.correction_ms, 0.3)
        self.assertAlmostEqual(t.total_global_ms, 0.6)
        self.assertAlmostEqual(t.frame_no_cd_ms, 15.0)
        self.assertAlmostEqual(t.total_step_ms, 16.5)

    def test_no_nsn_block_leaves_constraint_fields_none(self) -> None:
        t = parse_realsim_stdout(PLAIN_BLOCK_NO_NSN, demo="Y", timer_interval=50)
        self.assertEqual(t.n_blocks, 1)
        # NSN-tied fields stay None.
        self.assertIsNone(t.schur_ms)
        self.assertIsNone(t.n_constraints_mean)
        self.assertIsNone(t.cst_solve_iter_mean)
        self.assertIsNone(t.build_ms)
        self.assertIsNone(t.cst_solve_ms)
        self.assertIsNone(t.correction_ms)
        self.assertIsNone(t.total_global_ms)
        # Always-present fields parse fine.
        self.assertAlmostEqual(t.local_ms, 0.6)
        self.assertAlmostEqual(t.linear_solve_ms, 2.5)
        self.assertAlmostEqual(t.frame_no_cd_ms, 16.0)
        self.assertAlmostEqual(t.total_step_ms, 17.5)

    def test_multiple_blocks_averaged(self) -> None:
        block_a = "Printing profile over 50 time steps\nLocal cost = 1.0 ms\nLinear solve cost = 2.0 ms\nFrame without collision detection = 10.0 ms\nTotal time step cost = 11.0 ms\n"
        block_b = "Printing profile over 50 time steps\nLocal cost = 3.0 ms\nLinear solve cost = 4.0 ms\nFrame without collision detection = 20.0 ms\nTotal time step cost = 21.0 ms\n"
        t = parse_realsim_stdout(block_a + block_b, demo="Z", timer_interval=50)
        self.assertEqual(t.n_blocks, 2)
        self.assertEqual(t.n_frames, 100)
        self.assertAlmostEqual(t.local_ms, 2.0)
        self.assertAlmostEqual(t.linear_solve_ms, 3.0)
        self.assertAlmostEqual(t.frame_no_cd_ms, 15.0)
        self.assertAlmostEqual(t.total_step_ms, 16.0)

    def test_empty_stdout_returns_zero_blocks(self) -> None:
        t = parse_realsim_stdout("", demo="Q", timer_interval=50)
        self.assertEqual(t.n_blocks, 0)
        self.assertEqual(t.n_frames, 0)
        # No samples -> sentinel zero rather than None for required fields
        # to match how :func:`run_realsim_with_timing` reports a degenerate run.
        self.assertEqual(t.local_ms, 0.0)
        self.assertEqual(t.total_step_ms, 0.0)
        self.assertIsNone(t.schur_ms)


if __name__ == "__main__":
    unittest.main()
