# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of SolverFBA's NumPy sparse LDLᵀ factorization."""

import unittest
from unittest import mock

import numpy as np

from newton._src.solvers.fba import linear_solver
from newton._src.solvers.fba.linear_solver import factorize_and_sparse_inverse
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
        """Sum repeated triplets and sort the columns of each row."""
        matrix = CsrMatrix.from_triplets((3, 4), [2, 0, 2, 0], [1, 3, 1, 0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(matrix.toarray(), [[4.0, 0.0, 0.0, 2.0], [0.0] * 4, [0.0, 4.0, 0.0, 0.0]])
        np.testing.assert_array_equal(matrix.indices, [0, 3, 1])


class TestNestedDissection(unittest.TestCase):
    def test_nodes_tile_the_order(self):
        """Return a permutation whose nodes cover it in sequence, children before their parent."""
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
        """Separate the graph so that every coupling between two nodes joins a node and one of its ancestors, never two
        siblings' subtrees."""
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
        """Factor dissected sheet and block grids to round-off."""
        for shape in ((30, 30, 1), (10, 10, 10)):
            with self.subTest(shape=shape):
                self.assert_factors(*grid_system(*shape))

    def test_disconnected_meshes(self):
        """Factor two separate grids, whose dissection has an empty separator, to round-off."""
        a, pa = grid_system(8, 8, 2)
        b, pb = grid_system(6, 6, 3)
        n = a.shape[0] + b.shape[0]
        rows = np.concatenate([a.row_of_entries(), b.row_of_entries() + a.shape[0]])
        cols = np.concatenate([a.indices, b.indices + a.shape[0]])
        both = CsrMatrix.from_triplets((n, n), rows, cols, np.concatenate([a.data, b.data]))
        self.assert_factors(both, np.concatenate([pa, pb + 100.0]))

    def test_dense_without_positions(self):
        """Factor densely in the matrix's own order when no points are given."""
        rng = np.random.default_rng(5)
        root = rng.standard_normal((40, 40))
        matrix = CsrMatrix.from_matrix(root @ root.T + 40.0 * np.eye(40))
        order, _, _ = factor_ldlt(matrix)
        np.testing.assert_array_equal(order, np.arange(40))
        self.assert_factors(matrix, None)

    def test_rejects_indefinite(self):
        """Refuse to factor a matrix with a negative eigenvalue."""
        with self.assertRaises(np.linalg.LinAlgError):
            factor_ldlt(CsrMatrix.from_matrix(np.diag([1.0, -1.0, 2.0])))


def _join(blocks, permutation):
    """The block diagonal matrix of ``blocks`` with its unknowns renumbered by ``permutation``."""
    rows, cols, values, first = [], [], [], 0
    for block in blocks:
        rows.append(block.row_of_entries() + first)
        cols.append(block.indices + first)
        values.append(block.data)
        first += block.shape[0]
    place = np.argsort(permutation)
    rows, cols = place[np.concatenate(rows)], place[np.concatenate(cols)]
    return CsrMatrix.from_triplets((first, first), rows, cols, np.concatenate(values))


class TestFactorizeWorlds(unittest.TestCase):
    def assert_system(self, matrix, positions, worlds=None):
        """``P A Pᵀ = L D Lᵀ`` with ``S = L⁻¹`` to round-off, and ``A⁻¹ b`` solves ``A x = b``; return the system."""
        if worlds is None:
            system = factorize_and_sparse_inverse(matrix, positions)  # the call without world grouping, as before
        else:
            system = factorize_and_sparse_inverse(matrix, positions, worlds=worlds)
        order = system.invperm_r
        np.testing.assert_array_equal(np.sort(order), np.arange(matrix.shape[0]))
        np.testing.assert_array_equal(system.perm_r[order], np.arange(matrix.shape[0]))
        A = matrix.toarray()
        S = system.S.toarray()
        L = np.linalg.inv(S)
        dense = A[np.ix_(order, order)]
        self.assertLess(np.abs(L / system.Dinv @ L.T - dense).max(), 1e-12 * np.abs(dense).max())
        b = np.random.default_rng(3).standard_normal(matrix.shape[0])
        x = np.empty_like(b)
        x[order] = S.T @ (system.Dinv * (S @ b[order]))
        self.assertLess(np.abs(A @ x - b).max(), 1e-10 * np.abs(b).max())
        return system

    def assert_whole(self, matrix, positions, worlds=None):
        """The system is exactly that of :func:`factor_ldlt` of the whole matrix."""
        system = self.assert_system(matrix, positions, worlds)
        order, L, d = factor_ldlt(matrix, positions)
        np.testing.assert_array_equal(system.invperm_r, order)
        np.testing.assert_array_equal(system.Dinv, 1.0 / d)
        S = linear_solver.compute_lower_inverse(L, parent=linear_solver._elimination_tree(L))
        np.testing.assert_array_equal(system.S.indptr, S.indptr)
        np.testing.assert_array_equal(system.S.indices, S.indices)
        np.testing.assert_array_equal(system.S.data, S.data)

    def assert_world_blocks(self, matrix, positions, worlds, system):
        """Each world takes consecutive places of the order, in ascending world order, with the factor that
        :func:`factor_ldlt` gives its own matrix, all of its components together."""
        A, S, first = matrix.toarray(), system.S.toarray(), 0
        for w in np.unique(worlds):
            dofs = np.flatnonzero(worlds == w)
            m = dofs.size
            block = CsrMatrix.from_matrix(A[np.ix_(dofs, dofs)])
            order, L, d = factor_ldlt(block, None if positions is None else positions[dofs])
            np.testing.assert_array_equal(system.invperm_r[first : first + m], dofs[order])
            np.testing.assert_array_equal(system.Dinv[first : first + m], 1.0 / d)
            Sw = linear_solver.compute_lower_inverse(L, parent=linear_solver._elimination_tree(L)).toarray()
            np.testing.assert_allclose(S[first : first + m, first : first + m], Sw, rtol=1e-14, atol=0.0)
            self.assertFalse(np.any(S[first : first + m, :first]) or np.any(S[first : first + m, first + m :]))
            first += m
        self.assertEqual(first, matrix.shape[0])

    def test_components_without_worlds_factor_whole(self):
        """Without world grouping, or with one world, a matrix of several components, such as isolated unknowns
        or interleaved bodies, factors as a whole, exactly as :func:`factor_ldlt` does."""
        blocks = [grid_system(4, 4, 2)[0], grid_system(3, 5, 1)[0], CsrMatrix.from_matrix(np.array([[2.5]]))]
        points = [grid_system(4, 4, 2)[1], grid_system(3, 5, 1)[1], np.zeros((1, 3))]
        n = sum(b.shape[0] for b in blocks)
        permutation = np.random.default_rng(7).permutation(n)
        matrix = _join(blocks, permutation)
        for positions in (np.concatenate(points)[permutation], None):
            for worlds in (None, np.zeros(n, dtype=np.int32), np.full(n, 3, dtype=np.int32)):
                with self.subTest(positions=positions is not None, worlds=None if worlds is None else worlds[0]):
                    self.assert_whole(matrix, positions, worlds)
        isolated = CsrMatrix.from_matrix(np.diag(np.linspace(1.0, 2.0, 64)))
        self.assert_whole(isolated, np.random.default_rng(1).standard_normal((64, 3)))

    def test_interleaved_heterogeneous_worlds(self):
        """Interleaved worlds of different sizes, one of two components, with an empty world number between them,
        factor one world at a time and keep each world's components together."""
        blocks = [
            grid_system(4, 4, 2)[0],
            grid_system(3, 5, 1)[0],
            CsrMatrix.from_matrix(np.array([[2.5]])),
            grid_system(2, 3, 2)[0],
        ]
        points = [grid_system(4, 4, 2)[1], grid_system(3, 5, 1)[1], np.zeros((1, 3)), grid_system(2, 3, 2)[1]]
        world_of_block = [3, 0, 3, 1]  # world 2 is empty; world 3 holds two components
        sizes = [b.shape[0] for b in blocks]
        n = sum(sizes)
        permutation = np.random.default_rng(11).permutation(n)
        matrix = _join(blocks, permutation)
        worlds = np.repeat(world_of_block, sizes)[permutation].astype(np.int32)
        for positions in (np.concatenate(points)[permutation], None):
            with self.subTest(positions=positions is not None):
                calls = []

                def factor(matrix, positions=None, leaf_size=64, calls=calls):
                    calls.append(matrix.shape[0])
                    return factor_ldlt(matrix, positions, leaf_size)

                with mock.patch.object(linear_solver, "factor_ldlt", side_effect=factor):
                    system = self.assert_system(matrix, positions, worlds)
                self.assertEqual(calls, [sizes[1], sizes[3], sizes[0] + sizes[2]])
                self.assert_world_blocks(matrix, positions, worlds, system)

    def test_colocated_worlds_factor_alone(self):
        """Co-located replica worlds reach the multifrontal factor one at a time, so no front spans two."""
        block, points = grid_system(5, 5, 2)
        replicas = 4
        n = replicas * block.shape[0]
        permutation = np.arange(n).reshape(replicas, -1).T.ravel()  # interleave the replicas' unknowns
        matrix = _join([block] * replicas, permutation)
        positions = np.tile(points, (replicas, 1))[permutation]
        worlds = (permutation // block.shape[0]).astype(np.int32)
        sizes = []

        def factor(matrix, positions=None, leaf_size=64):
            sizes.append(matrix.shape[0])
            return factor_ldlt(matrix, positions, leaf_size)

        with mock.patch.object(linear_solver, "factor_ldlt", side_effect=factor):
            system = self.assert_system(matrix, positions, worlds)
        self.assertEqual(sizes, [block.shape[0]] * replicas)
        self.assert_world_blocks(matrix, positions, worlds, system)
        reference = factor_ldlt(block, points)[1]
        self.assertEqual(system.S.data.size, replicas * linear_solver.compute_lower_inverse(reference).data.size)

    def test_coupled_worlds_factor_whole(self):
        """Worlds coupled to each other, or to a global unknown of world -1, factor as a whole, exactly as before."""
        block, points = grid_system(4, 4, 1)
        replicas, m = 3, block.shape[0]
        matrix = _join([block] * replicas, np.arange(replicas * m))
        rows = np.r_[matrix.row_of_entries(), np.arange(0, replicas * m, m), np.full(replicas, replicas * m)]
        cols = np.r_[matrix.indices, np.full(replicas, replicas * m), np.arange(0, replicas * m, m)]
        values = np.r_[matrix.data + 0.0, -0.1 * np.ones(2 * replicas)]
        coupled = CsrMatrix.from_triplets(
            (replicas * m + 1,) * 2, np.r_[rows, replicas * m], np.r_[cols, replicas * m], np.r_[values, 1.0]
        )
        points = np.r_[np.tile(points, (replicas, 1)), np.zeros((1, 3))]
        worlds = np.r_[np.repeat(np.arange(replicas), m), -1].astype(np.int32)
        for positions in (points, None):
            for labels in (None, worlds):
                with self.subTest(positions=positions is not None, worlds=labels is not None):
                    self.assert_whole(coupled, positions, labels)

    def test_diagonal_and_empty_matrices(self):
        """Factor a matrix of singletons, one world each, and the empty matrix."""
        diagonal = CsrMatrix.from_matrix(np.diag([2.0, 0.5, 4.0]))
        worlds = np.array([1, 0, 2], dtype=np.int32)
        system = self.assert_system(diagonal, np.zeros((3, 3)), worlds)
        np.testing.assert_allclose(system.Dinv[system.perm_r], [0.5, 2.0, 0.25])
        self.assert_world_blocks(diagonal, np.zeros((3, 3)), worlds, system)
        empty = factorize_and_sparse_inverse(CsrMatrix.from_matrix(np.zeros((0, 0))))
        self.assertEqual(empty.S.shape, (0, 0))
        self.assertEqual(empty.Dinv.size, 0)


if __name__ == "__main__":
    unittest.main()
