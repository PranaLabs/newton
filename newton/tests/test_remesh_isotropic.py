# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from newton._src.geometry.remesh_isotropic import _collapse, _normals, _split, _topology


def graded_sheet():
    """A 10 x 4 rectangle in the xy plane, with columns 1.67 apart on its left half and 0.5 apart on its right."""
    xs = np.concatenate([np.linspace(0.0, 5.0, 4), np.linspace(5.5, 10.0, 10)])
    ys = np.linspace(0.0, 4.0, 5)
    v = np.array([[x, y, 0.0] for y in ys for x in xs])
    n = len(xs)
    f = [[j * n + i, j * n + i + 1, (j + 1) * n + i + 1] for j in range(len(ys) - 1) for i in range(n - 1)]
    f += [[j * n + i, (j + 1) * n + i + 1, (j + 1) * n + i] for j in range(len(ys) - 1) for i in range(n - 1)]
    return v, np.array(f)


class TestSplitCollapse(unittest.TestCase):
    def check_sheet(self, v, f):
        """The faces cover the rectangle, face up, and every edge has one or two faces; boundary vertices lie on the
        rectangle's sides."""
        normal = _normals(v, f)
        np.testing.assert_allclose(0.5 * normal[:, 2].sum(), 40.0, rtol=1e-12)
        self.assertTrue((normal[:, 2] > 0.0).all())
        _, edges, _, count = _topology(f)
        self.assertTrue((count <= 2).all())
        b = v[np.unique(edges[count == 1])]
        on_side = (
            np.isclose(b[:, 0], 0.0) | np.isclose(b[:, 0], 10.0) | np.isclose(b[:, 1], 0.0) | np.isclose(b[:, 1], 4.0)
        )
        self.assertTrue(on_side.all())
        return np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)

    def test_split_bounds_the_edges(self):
        v, f = _split(*graded_sheet(), 4.0 / 3.0)
        self.assertLessEqual(self.check_sheet(v, f).max(), 4.0 / 3.0)

    def test_collapse_lengthens_the_short_edges(self):
        """Collapses take most of the 74 edges shorter than 0.8 and keep every edge under 4/3, the faces up and the
        rectangle whole. The left 0.75 apart, and those of the 0.5 apart that touch a corner, would stretch an edge
        past 4/3 or cut the corner."""
        v, f = _split(*graded_sheet(), 4.0 / 3.0)
        v, f = _collapse(v, f, 0.8, 4.0 / 3.0)
        length = self.check_sheet(v, f)
        self.assertLessEqual(length.max(), 4.0 / 3.0)
        self.assertLessEqual((length < 0.8).sum(), 22)
        self.assertLessEqual((length < 0.6).sum(), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
