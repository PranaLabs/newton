# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the block partial sums of the fused PCR iterations (``NSNPCRSolver.solve_sparse_fused``).

Each block of 256 rows writes its partial sums of the iteration's dot products; every block then needs their
total. Up to 32 blocks each thread adds them in order, past that the block reduces them in a tile. Both must give
the iterations of ``solve_sparse``, and all blocks the same total.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba.nsn_pcr_solver import _FUSED_BLOCK, _SERIAL_BLOCKS, NSNPCRSolver


def _sparse_spd(n: int, per_row: int, seed: int):
    """A diagonally dominant sparse SPD matrix of ``n`` rows, about ``2 per_row`` entries off the diagonal per row,
    as CSR triplets."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), per_row)
    cols = rng.integers(0, n, size=n * per_row)
    vals = rng.standard_normal(n * per_row)
    keep = rows != cols
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    rows, cols, vals = np.concatenate([rows, cols]), np.concatenate([cols, rows]), np.concatenate([vals, vals])
    diag = np.bincount(rows, weights=np.abs(vals), minlength=n) + rng.uniform(0.5, 2.0, n)
    return (
        np.concatenate([rows, np.arange(n)]),
        np.concatenate([cols, np.arange(n)]),
        np.concatenate([vals, diag]),
    )


class TestPCRBlockPartials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The fused PCR iterations run on CUDA")
        cls.device = "cuda:0"

    def _both(self, n: int, seed: int) -> list[np.ndarray]:
        """The solutions of ``solve_sparse`` and ``solve_sparse_fused`` after ten iterations."""
        import warp.sparse as wps  # noqa: PLC0415

        rows, cols, vals = _sparse_spd(n, 4, seed)
        W = wps.bsr_from_triplets(
            n,
            n,
            wp.array(rows, dtype=wp.int32, device=self.device),
            wp.array(cols, dtype=wp.int32, device=self.device),
            wp.array(vals, dtype=wp.float64, device=self.device),
        )
        b = wp.array(np.random.default_rng(seed + 1).standard_normal(n), dtype=wp.float64, device=self.device)
        out = []
        for name in ("solve_sparse", "solve_sparse_fused"):
            solver = NSNPCRSolver(max_n=n, device=self.device, tol=1.0e-10, max_iter=10)
            x = wp.zeros(n, dtype=wp.float64, device=self.device)
            self.assertEqual(getattr(solver, name)(W, b, x), 10)
            out.append(x.numpy())
        return out

    def test_few_blocks_add_in_order(self):
        """Up to 32 blocks (here 3 and 32) the threads add the partial sums: the iterations of ``solve_sparse``."""
        for blocks in (3, int(_SERIAL_BLOCKS)):
            with self.subTest(blocks=blocks):
                x_ref, x = self._both(blocks * int(_FUSED_BLOCK) - 17, seed=blocks)
                np.testing.assert_allclose(x, x_ref, rtol=1.0e-11, atol=1.0e-13 * np.abs(x_ref).max())

    def test_many_blocks_reduce_in_a_tile(self):
        """Past 32 blocks (here 40) the blocks reduce the partial sums in a tile: the same iterations."""
        blocks = int(_SERIAL_BLOCKS) + 8
        x_ref, x = self._both(blocks * int(_FUSED_BLOCK) - 5, seed=7)
        np.testing.assert_allclose(x, x_ref, rtol=1.0e-11, atol=1.0e-13 * np.abs(x_ref).max())


if __name__ == "__main__":
    unittest.main(verbosity=2)
