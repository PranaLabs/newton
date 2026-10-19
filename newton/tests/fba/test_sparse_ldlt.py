# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of SolverFBA's NumPy sparse LDLᵀ factorization."""

import unittest

import numpy as np

from newton._src.solvers.fba.sparse_ldlt import CsrMatrix, factor_ldlt, nested_dissection


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


class TestFactorLdlt(unittest.TestCase):
    def assert_factors(self, matrix, positions):
        order, L, d = factor_ldlt(matrix, positions, leaf_size=16)
        dense = matrix.toarray()[np.ix_(order, order)]
        Ld = L.toarray()
        np.testing.assert_allclose(np.diag(Ld), 1.0)
        np.testing.assert_allclose(np.triu(Ld, 1), 0.0)
        self.assertLess(np.abs(Ld * d @ Ld.T - dense).max(), 1e-12 * np.abs(dense).max())

    def test_mesh_systems(self):
        """Dissected sheet and block grids factor to round-off."""
        for shape in ((30, 30, 1), (10, 10, 10)):
            with self.subTest(shape=shape):
                self.assert_factors(*grid_system(*shape))

    def test_disconnected_meshes(self):
        """Two separate grids, whose dissection has an empty separator, factor to round-off."""
        a, pa = grid_system(8, 8, 2)
        b, pb = grid_system(6, 6, 3)
        n = a.shape[0] + b.shape[0]
        rows = np.concatenate([a.row_of_entries(), b.row_of_entries() + a.shape[0]])
        cols = np.concatenate([a.indices, b.indices + a.shape[0]])
        both = CsrMatrix.from_triplets((n, n), rows, cols, np.concatenate([a.data, b.data]))
        self.assert_factors(both, np.concatenate([pa, pb + 100.0]))

    def test_dense_without_positions(self):
        """Without points the factorization is dense, in the matrix's own order."""
        rng = np.random.default_rng(5)
        root = rng.standard_normal((40, 40))
        matrix = CsrMatrix.from_matrix(root @ root.T + 40.0 * np.eye(40))
        order, _, _ = factor_ldlt(matrix)
        np.testing.assert_array_equal(order, np.arange(40))
        self.assert_factors(matrix, None)

    def test_rejects_indefinite(self):
        """A matrix with a negative eigenvalue is not factored."""
        with self.assertRaises(np.linalg.LinAlgError):
            factor_ldlt(CsrMatrix.from_matrix(np.diag([1.0, -1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
