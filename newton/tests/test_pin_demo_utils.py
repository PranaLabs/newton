# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the geometric acceptance tests used by the pin demos."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.pin_demo_utils import SurfaceIntersectionCheck


class TestSurfaceIntersectionCheck(unittest.TestCase):
    def count(self, vertices, scale=1.0):
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=scale,
            vel=wp.vec3(),
            vertices=np.asarray(vertices).tolist(),
            indices=[0, 1, 2, 3, 4, 5],
            density=1.0,
        )
        model = builder.finalize()
        return SurfaceIntersectionCheck(model).intersections(model.particle_q)

    def test_crossing_at_different_scales(self):
        vertices = [[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, -0.5, -1], [0, -0.5, 1], [0, 0.5, 0]]
        for scale in (1.0, 0.01, 0.001):
            with self.subTest(scale=scale):
                self.assertEqual(self.count(vertices, scale), 1)

    def test_separated_small_triangles_with_overlapping_bounds(self):
        # Actual shoulder faces: the unscaled Warp predicate reports a false hit.
        vertices = [
            [0.15063153, 1.2339364, -0.01706888],
            [0.1436708, 1.2412329, -0.01275041],
            [0.15231134, 1.2367985, -0.00767644],
            [0.1354052, 1.2482758, -0.00827578],
            [0.13311891, 1.2411968, 0.00220041],
            [0.14408201, 1.2416159, -0.00253444],
        ]
        self.assertEqual(self.count(vertices), 0)


if __name__ == "__main__":
    unittest.main()
