# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.self_contact import SelfContact, particle_rings


def _closest_on_triangle(p, a, b, c):
    """Closest point of a triangle to ``p`` by dense sampling of barycentric coordinates [m]."""
    u, v = np.mgrid[0.0:1.0:201j, 0.0:1.0:201j]
    keep = u + v <= 1.0
    points = np.outer(1.0 - u[keep] - v[keep], a) + np.outer(u[keep], b) + np.outer(v[keep], c)
    return points[np.argmin(np.linalg.norm(points - p, axis=1))]


class TestSelfContactDetection(unittest.TestCase):
    """Two sheets of 2 cm cells, the second 1 cm over the first and shifted 3 mm along x and y."""

    def _sheets(self):
        builder = newton.ModelBuilder()
        for z, shift in ((0.0, 0.0), (0.01, 0.003)):
            builder.add_cloth_grid(
                pos=wp.vec3(shift, shift, z),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=10,
                dim_y=10,
                cell_x=0.02,
                cell_y=0.02,
                mass=0.01,
            )
        return builder.finalize()

    def test_vertex_face_takes_the_nearest_triangle(self):
        """Each particle has one contact, with its nearest triangle of the other sheet, along the direction from
        the closest point to the particle."""
        model = self._sheets()
        q, tris = model.particle_q.numpy(), model.tri_indices.numpy()
        particles, alpha, normal = SelfContact(model, "vertex_face", 0.015).detect(model.particle_q)
        np.testing.assert_array_equal(particles[:, 0], np.arange(model.particle_count))
        half = model.particle_count // 2
        for k in range(0, len(particles), 17):
            v = particles[k, 0]
            other = tris[(tris >= half).all(axis=1) if v < half else (tris < half).all(axis=1)]
            closest = [_closest_on_triangle(q[v], *q[t]) for t in other]
            nearest = closest[np.argmin([np.linalg.norm(q[v] - c) for c in closest])]
            point = -(alpha[k, 1:] @ q[particles[k, 1:]])
            np.testing.assert_allclose(point, nearest, atol=2e-4)
            np.testing.assert_allclose(normal[k], (q[v] - point) / np.linalg.norm(q[v] - point), atol=1e-5)

    def test_vertex_vertex_pairs_once(self):
        """Every pair of particles within the margin is one contact; a sheet's own particles lie 2 cm apart, so
        all pairs join the two sheets."""
        model = self._sheets()
        q = model.particle_q.numpy()
        particles, alpha, normal = SelfContact(model, "vertex_vertex", 0.015).detect(model.particle_q)
        d = np.linalg.norm(q[:, None] - q[None], axis=2)
        expected = set(zip(*np.nonzero(np.triu(d < 0.015, 1)), strict=True))
        self.assertEqual({(a, b) for a, b in particles[:, :2]}, expected)
        np.testing.assert_array_equal(alpha[:, :2], np.tile([1.0, -1.0], (len(particles), 1)))
        expected_normal = (q[particles[:, 0]] - q[particles[:, 1]]) / d[tuple(particles[:, :2].T)][:, None]
        np.testing.assert_allclose(normal, expected_normal, atol=1e-6)

    def test_rings_exclude_near_particles(self):
        """On one sheet, a 5 cm margin finds only pairs more than two mesh edges apart."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=6,
            dim_y=6,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.01,
        )
        model = builder.finalize()
        tris = model.tri_indices.numpy()
        offsets, ring = particle_rings(tris, model.particle_count, 2)
        one = [set(tris[(tris == v).any(axis=1)].ravel()) | {v} for v in range(model.particle_count)]
        for v in range(model.particle_count):
            self.assertEqual(set(ring[offsets[v] : offsets[v + 1]]), set().union(*(one[u] for u in one[v])))
        particles, _, _ = SelfContact(model, "vertex_vertex", 0.05).detect(model.particle_q)
        self.assertGreater(len(particles), 0)
        for a, b in particles[:, :2]:
            self.assertNotIn(b, ring[offsets[a] : offsets[a + 1]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
