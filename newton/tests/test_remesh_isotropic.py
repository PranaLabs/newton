# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton._src.geometry.remesh_isotropic import _collapse, _normals, _split, _topology, remesh_isotropic


def graded_sheet():
    """A 10 x 4 rectangle in the xy plane, with columns 1.67 apart on its left half and 0.5 apart on its right."""
    xs = np.concatenate([np.linspace(0.0, 5.0, 4), np.linspace(5.5, 10.0, 10)])
    ys = np.linspace(0.0, 4.0, 5)
    v = np.array([[x, y, 0.0] for y in ys for x in xs])
    n = len(xs)
    f = [[j * n + i, j * n + i + 1, (j + 1) * n + i + 1] for j in range(len(ys) - 1) for i in range(n - 1)]
    f += [[j * n + i, (j + 1) * n + i + 1, (j + 1) * n + i] for j in range(len(ys) - 1) for i in range(n - 1)]
    return v, np.array(f)


def check_sheet(test, v, f):
    """Assert that the faces cover the rectangle of :func:`graded_sheet`, face up, that every edge has one or two
    faces and that boundary vertices lie on the rectangle's sides; return the edge lengths."""
    normal = _normals(v, f)
    np.testing.assert_allclose(0.5 * normal[:, 2].sum(), 40.0, rtol=1e-12)
    test.assertTrue((normal[:, 2] > 0.0).all())
    _, edges, _, count = _topology(f)
    test.assertTrue((count <= 2).all())
    b = v[np.unique(edges[count == 1])]
    on_side = np.isclose(b[:, 0], 0.0) | np.isclose(b[:, 0], 10.0) | np.isclose(b[:, 1], 0.0) | np.isclose(b[:, 1], 4.0)
    test.assertTrue(on_side.all())
    return np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)


class TestSplitCollapse(unittest.TestCase):
    def test_split_bounds_the_edges(self):
        """Split the edges of a graded sheet longer than 4/3 until none is."""
        v, f = _split(*graded_sheet(), 4.0 / 3.0)
        self.assertLessEqual(check_sheet(self, v, f).max(), 4.0 / 3.0)

    def test_collapse_lengthens_the_short_edges(self):
        """Collapse most of the 74 edges shorter than 0.8, keeping every edge under 4/3, the faces up and the
        rectangle whole.

        The short edges that remain, 0.75 long or 0.5 long at a corner, would stretch an edge past 4/3 or cut the
        corner if collapsed.
        """
        v, f = _split(*graded_sheet(), 4.0 / 3.0)
        v, f = _collapse(v, f, 0.8, 4.0 / 3.0)
        length = check_sheet(self, v, f)
        self.assertLessEqual(length.max(), 4.0 / 3.0)
        self.assertLessEqual((length < 0.8).sum(), 22)
        self.assertLessEqual((length < 0.6).sum(), 7)


def quality(v, f):
    """Ratio of each triangle's area to that of the equilateral triangle with the same mean squared edge."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    squares = ((b - a) ** 2 + (c - b) ** 2 + (a - c) ** 2).sum(axis=1)
    return 2.0 * np.sqrt(3.0) * np.linalg.norm(np.cross(b - a, c - a), axis=1) / squares


class TestRemeshIsotropic(unittest.TestCase):
    def test_remesh_evens_out_a_graded_sheet(self):
        """Remesh a sheet with edges of 0.5 to 1.9 to edges of at most 4/3, mostly over 0.65, on the same rectangle.

        Over a quarter of the input edges are 0.5 long, and the right triangles become nearly equilateral.
        """
        v, f = remesh_isotropic(*graded_sheet(), edge_length=1.0)
        length = check_sheet(self, v.astype(np.float64), f)
        self.assertLessEqual(length.max(), 4.0 / 3.0)
        self.assertGreater(np.percentile(length, 5), 0.65)
        self.assertGreater(np.percentile(quality(v, f), 5), 0.75)

    def test_remesh_keeps_a_sphere_closed_on_its_surface(self):
        """Remesh a UV sphere, dense at its poles, into a closed surface on the input triangles.

        The result keeps Euler characteristic 2, and its vertices lie between the input triangles' inner radius and
        the unit sphere.
        """
        theta, phi = np.meshgrid(np.linspace(0.0, np.pi, 17)[1:-1], np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False))
        v = np.c_[np.sin(theta.T).ravel() * np.cos(phi.T).ravel(), np.sin(theta.T).ravel() * np.sin(phi.T).ravel()]
        v = np.r_[[[0.0, 0.0, 1.0]], np.c_[v, np.cos(theta.T).ravel()], [[0.0, 0.0, -1.0]]]
        ring = lambda j, i: 1 + 32 * j + i % 32  # noqa: E731
        f = [[0, ring(0, i), ring(0, i + 1)] for i in range(32)] + [
            [ring(14, i + 1), ring(14, i), 449] for i in range(32)
        ]
        f += [[ring(j, i), ring(j + 1, i), ring(j + 1, i + 1)] for j in range(14) for i in range(32)]
        f += [[ring(j, i), ring(j + 1, i + 1), ring(j, i + 1)] for j in range(14) for i in range(32)]
        u, g = remesh_isotropic(v, np.array(f), edge_length=0.2)
        _, edges, _, count = _topology(g)
        self.assertTrue((count == 2).all())
        self.assertEqual(len(u) - len(edges) + len(g), 2)
        radius = np.linalg.norm(u, axis=1)
        self.assertGreater(radius.min(), np.cos(np.pi / 16))
        self.assertLess(radius.max(), 1.0 + 1e-6)
        self.assertGreater(np.percentile(quality(u, g), 1), 0.75)

    def test_remesh_mesh_takes_the_isotropic_method(self):
        """Remesh through ``newton.utils.remesh_mesh(method="isotropic")`` to the mesh ``remesh_isotropic`` gives."""
        v, f = graded_sheet()
        mesh = newton.utils.remesh_mesh(newton.Mesh(v, f.ravel()), method="isotropic", edge_length=1.0)
        u, g = remesh_isotropic(v, f, edge_length=1.0)
        np.testing.assert_allclose(mesh.vertices, u)
        np.testing.assert_array_equal(mesh.indices.reshape(-1, 3), g)


if __name__ == "__main__":
    unittest.main(verbosity=2)
