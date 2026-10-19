# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse LDLᵀ factorization of SolverFBA's symmetric positive definite systems with NumPy.

The projective-dynamics matrix of a mesh couples particles that are close in space. Geometric nested dissection
orders its unknowns by recursive bisection of their positions, eliminating each half before the separator between
them, which keeps the elimination tree shallow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CsrMatrix:
    """A compressed sparse row matrix of NumPy arrays, with sorted column indices in each row."""

    shape: tuple[int, int]
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray

    @staticmethod
    def from_triplets(shape: tuple[int, int], rows, cols, values) -> CsrMatrix:
        """Sum the entries ``(rows[k], cols[k]) += values[k]`` into a matrix of ``shape``."""
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        order = np.lexsort((cols, rows))
        rows, cols, values = rows[order], cols[order], values[order]
        first = np.ones(rows.size, dtype=bool)
        first[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        starts = np.flatnonzero(first)
        data = np.add.reduceat(values, starts) if starts.size else np.zeros(0)
        indptr = np.searchsorted(rows[starts], np.arange(shape[0] + 1))
        return CsrMatrix((int(shape[0]), int(shape[1])), indptr, cols[starts].astype(np.int32), data)

    @staticmethod
    def from_matrix(matrix) -> CsrMatrix:
        """A :class:`CsrMatrix` of ``matrix``: a :class:`CsrMatrix`, a dense array, or a matrix with ``tocoo()``."""
        if isinstance(matrix, CsrMatrix):
            return matrix
        if hasattr(matrix, "tocoo"):
            coo = matrix.tocoo()
            return CsrMatrix.from_triplets(coo.shape, coo.row, coo.col, coo.data)
        dense = np.asarray(matrix, dtype=np.float64)
        rows, cols = np.nonzero(dense)
        return CsrMatrix.from_triplets(dense.shape, rows, cols, dense[rows, cols])

    @property
    def nnz(self) -> int:
        """Number of stored entries."""
        return int(self.indices.size)

    def row_of_entries(self) -> np.ndarray:
        """Row index of every stored entry."""
        return np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))

    def toarray(self) -> np.ndarray:
        """The matrix as a dense array."""
        dense = np.zeros(self.shape)
        np.add.at(dense, (self.row_of_entries(), self.indices), self.data)
        return dense


def nested_dissection(matrix: CsrMatrix, positions: np.ndarray, leaf_size: int = 64) -> tuple[np.ndarray, list]:
    """Order the unknowns of a symmetric matrix whose row ``i`` belongs to a point at ``positions[i]``.

    The points are split at the median of their widest coordinate; the separator is the side of the cut, among
    the two, with fewer points that have a neighbor across it. Both halves are ordered recursively before the
    separator, down to blocks of at most ``leaf_size`` unknowns.

    Args:
        matrix: Symmetric matrix whose pattern is the coupling graph.
        positions: Point of each unknown [m], shape ``[n, 3]``.
        leaf_size: Largest block that is not split further.

    Returns:
        ``(order, nodes)``: ``order[k]`` is the unknown eliminated ``k``-th, and ``nodes`` the dissection tree in
        postorder as ``(first, end, children)``: each node eliminates ``order[first:end]`` after its children.
    """
    indptr, indices = matrix.indptr, matrix.indices
    positions = np.asarray(positions, dtype=np.float64)
    side = np.zeros(matrix.shape[0], dtype=np.int8)
    order, nodes = [], []

    def boundary(part: np.ndarray, other: int) -> np.ndarray:
        lengths = indptr[part + 1] - indptr[part]
        entries = np.repeat(indptr[part] - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())
        return np.unique(np.repeat(part, lengths)[side[indices[entries]] == other])

    def dissect(unknowns: np.ndarray, first: int) -> tuple[int, int]:
        """Order ``unknowns`` from position ``first``; return the end position and the index of their root node."""
        if unknowns.size <= leaf_size:
            order.append(unknowns)
            nodes.append((first, first + unknowns.size, []))
            return first + unknowns.size, len(nodes) - 1
        points = positions[unknowns]
        axis = int(np.argmax(np.ptp(points, axis=0)))
        lower = points[:, axis] < np.median(points[:, axis])
        if lower.all() or not lower.any():
            lower = np.arange(unknowns.size) < unknowns.size // 2
        halves = [unknowns[lower], unknowns[~lower]]
        side[halves[0]], side[halves[1]] = 1, 2
        cuts = [boundary(halves[0], 2), boundary(halves[1], 1)]
        k = 0 if cuts[0].size <= cuts[1].size else 1
        separator, halves[k] = cuts[k], np.setdiff1d(halves[k], cuts[k])
        side[unknowns] = 0
        children = []
        for half in halves:
            if half.size:
                first, child = dissect(half, first)
                children.append(child)
        order.append(separator)
        nodes.append((first, first + separator.size, children))
        return first + separator.size, len(nodes) - 1

    dissect(np.arange(matrix.shape[0]), 0)
    return np.concatenate(order).astype(np.int64), nodes
