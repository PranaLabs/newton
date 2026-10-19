# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of SolverFBA's NumPy sparse LDLᵀ factorization."""

import unittest

import numpy as np

from newton._src.solvers.fba.sparse_ldlt import CsrMatrix, nested_dissection


def grid_system(nx: int, ny: int, nz: int) -> tuple[CsrMatrix, np.ndarray]:
    """A mass-shifted graph Laplacian of a box grid, and the grid points."""
    index = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    pairs = [
        np.stack([a.ravel(), b.ravel()], axis=1)
        for a, b in ((index[1:], index[:-1]), (index[:, 1:], index[:, :-1]), (index[:, :, 1:], index[:, :, :-1]))
    ]
    edges = np.concatenate(pairs)
    n = index.size
    rows = np.concatenate([edges[:, 0], edges[:, 1], np.arange(n)])
    cols = np.concatenate([edges[:, 1], edges[:, 0], np.arange(n)])
    degree = np.bincount(edges.ravel(), minlength=n)
    values = np.concatenate([-np.ones(2 * len(edges)), degree + 0.1])
    points = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1)
    return CsrMatrix.from_triplets((n, n), rows, cols, values), points.reshape(-1, 3).astype(np.float64)


class TestCsrMatrix(unittest.TestCase):
    def test_from_triplets_sums_duplicates(self):
        """Repeated entries add up and the columns of each row come out sorted."""
        matrix = CsrMatrix.from_triplets((3, 4), [2, 0, 2, 0], [1, 3, 1, 0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(matrix.toarray(), [[4.0, 0.0, 0.0, 2.0], [0.0] * 4, [0.0, 4.0, 0.0, 0.0]])
        np.testing.assert_array_equal(matrix.indices, [0, 3, 1])


class TestNestedDissection(unittest.TestCase):
    def test_nodes_tile_the_order(self):
        """The order is a permutation, and the nodes cover it in sequence with children before their parent."""
        matrix, points = grid_system(20, 20, 5)
        order, nodes = nested_dissection(matrix, points, leaf_size=16)
        np.testing.assert_array_equal(np.sort(order), np.arange(matrix.shape[0]))
        end = 0
        for k, (first, last, children) in enumerate(nodes):
            self.assertEqual(first, end)
            self.assertTrue(all(c < k for c in children))
            end = last
        self.assertEqual(end, matrix.shape[0])

    def test_separators_split_the_graph(self):
        """Every coupling between two nodes joins a node and one of its ancestors, never two siblings' subtrees."""
        matrix, points = grid_system(12, 12, 4)
        order, nodes = nested_dissection(matrix, points, leaf_size=8)
        parent = np.full(len(nodes), -1)
        node_of = np.empty(matrix.shape[0], dtype=np.int64)
        for k, (first, last, children) in enumerate(nodes):
            parent[children] = k
            node_of[order[first:last]] = k

        def ancestors(k):
            while k >= 0:
                yield k
                k = parent[k]

        for i, j in zip(matrix.row_of_entries(), matrix.indices, strict=True):
            a, b = node_of[i], node_of[j]
            self.assertTrue(a in ancestors(b) or b in ancestors(a), (i, j))


if __name__ == "__main__":
    unittest.main()
