# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.types import GeoType
from newton._src.solvers.fba.face_contact import face_contact_kernel

HALF = np.array([0.5, 0.5, 0.5])


def _box_sdf(x, half=HALF):
    q = np.abs(x) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def _draped_triangle(scale=0.15):
    """A triangle over the box's top edge at ``x = z = 0.5``: its vertices are outside, the edge pokes through."""
    corner = np.array([0.5, 0.0, 0.5])
    return corner + scale * (np.array([[0.3, 0.0, 0.52], [0.7, -0.3, 0.3], [0.7, 0.3, 0.33]]) - corner)


class TestFaceContactKernel(unittest.TestCase):
    """Deepest points of triangles in a box with half extents 0.5 m."""

    def _query(self, points, X_wb=None, X_bs=None):
        """Bary, depth, body_pos and normal of the contacts between one triangle and the box."""
        X_bs, body = (wp.transform_identity(), -1) if X_wb is None else (X_bs, 0)
        inputs = [(points, wp.vec3), ([[0, 1, 2]], wp.int32), ([0], wp.int32), ([0], wp.int32)]
        inputs += [([X_bs if X_wb is None else X_wb], wp.transform), ([X_bs], wp.transform)]
        inputs += [([body], wp.int32), ([int(GeoType.BOX)], wp.int32), ([HALF], wp.vec3)]
        outputs = [wp.zeros(1, dtype=t) for t in (wp.int32, wp.int32, wp.vec3, wp.float32, wp.vec3, wp.vec3)]
        inputs = [wp.array(a, dtype=t) for a, t in inputs]
        wp.launch(face_contact_kernel, dim=1, inputs=[*inputs, 32, 0.02], outputs=outputs)
        n = int(outputs[0].numpy()[0])
        return [a.numpy()[:n] for a in outputs[2:]]

    def test_edge_through_triangle(self):
        """The contact is the brute-force deepest point, anchored on the box surface."""
        points = _draped_triangle()
        bary, depth, body_pos, normal = self._query(points)
        self.assertEqual(len(depth), 1)
        u, v = np.mgrid[0.0:1.0:401j, 0.0:1.0:401j]
        grid = np.stack([1.0 - u - v, u, v], axis=-1).reshape(-1, 3)
        deepest = _box_sdf(grid[grid[:, 0] >= 0.0] @ points).min()
        self.assertLess(deepest, -5e-3)
        self.assertGreater(_box_sdf(points).min(), 0.0)
        self.assertAlmostEqual(float(depth[0]), deepest, delta=2e-4)
        self.assertAlmostEqual(float(_box_sdf(bary[0] @ points)), float(depth[0]), delta=1e-6)
        np.testing.assert_allclose(body_pos[0], bary[0] @ points - depth[0] * normal[0], atol=1e-6)
        self.assertAlmostEqual(float(_box_sdf(body_pos[0])), 0.0, delta=1e-6)

    def test_triangle_on_face_has_no_face_contact(self):
        """A triangle 1 mm above the top face is deepest at a vertex, which particle contacts cover."""
        points = np.array([[-0.2, -0.2, 0.501], [0.2, -0.2, 0.501], [0.0, 0.2, 0.502]])
        self.assertEqual(len(self._query(points)[1]), 0)

    def test_moving_body(self):
        """Moving the box with a body moves the contact with it; the anchor stays in the body frame."""
        local = _draped_triangle()
        X_wb = wp.transform((1.0, -2.0, 0.5), wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.8, 0.5)), 0.7))
        X_bs = wp.transform((0.1, 0.0, -0.2), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4))
        X_ws = wp.transform_multiply(X_wb, X_bs)
        world = np.array([wp.transform_point(X_ws, wp.vec3(*p)) for p in local])
        bary, depth, body_pos, normal = self._query(local)
        moved = self._query(world, X_wb, X_bs)
        np.testing.assert_allclose(moved[0], bary, atol=1e-5)
        np.testing.assert_allclose(moved[1], depth, atol=1e-5)
        np.testing.assert_allclose(moved[2][0], wp.transform_point(X_bs, wp.vec3(*body_pos[0])), atol=1e-5)
        np.testing.assert_allclose(moved[3][0], wp.transform_vector(X_ws, wp.vec3(*normal[0])), atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
