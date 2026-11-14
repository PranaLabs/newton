# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for moving SolverFBA tendons' paths from device arrays (``set_tendon_paths``)."""

import unittest

import numpy as np
import warp as wp

import newton

H = 0.05  # half size of the boxes [m]


def _atwood():
    """Boxes of 20 g and 30 g on one string over two fixed points 0.2 m apart."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    bodies = []
    for x, m in ((-0.1, 0.02), (0.1, 0.03)):
        body = builder.add_body(xform=wp.transform((x, 0.0, 0.6 - H), wp.quat_identity()))
        builder.add_shape_box(body, hx=H, hy=H, hz=H, cfg=newton.ModelBuilder.ShapeConfig(density=m / (2 * H) ** 3))
        bodies.append(body)
    return builder.finalize(), bodies


class TestTendonPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def test_device_paths_step_as_host_paths(self):
        """Paths moved each step from device arrays give the steps of the same paths given to set_tendons, bitwise."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        A, B = np.array([-0.1, 0.0, 1.0]), np.array([0.1, 0.0, 1.0])
        rest = 2 * 0.4 + 0.2

        def run(device_paths):
            model, (b1, b2) = _atwood()
            solver = SolverFBA(model, iterations=10, nsn_sparse_storage=True, nsn_linear_iterations=10)
            s0, s1 = model.state(), model.state()
            points = [[(0, 0, H), (0, 0, H)]]
            solver.set_tendons([[b1, b2]], points, [[A, B]], [0.2], [rest])
            for k in range(60):
                # the pulleys drift sideways a little each step, so the paths change
                shift = np.array([1e-4 * k, 0.0, 0.0])
                targets, middle = [[A + shift, B + shift]], [0.2 + 1e-5 * k]
                if device_paths:
                    solver.set_tendon_paths(
                        wp.array(np.asarray(targets, np.float32), dtype=wp.vec3, device=model.device),
                        wp.array(middle, dtype=wp.float64, device=model.device),
                    )
                else:
                    solver.set_tendons([[b1, b2]], points, targets, middle, [rest])
                solver.step(s0, s1, None, None, 0.002)
                s0, s1 = s1, s0
            return s0.body_q.numpy()

        q_host, q_device = run(False), run(True)
        np.testing.assert_array_equal(q_device, q_host)
        self.assertGreater(abs(q_host[0, 2] - (0.6 - H)), 1e-3)  # the boxes moved

    def test_paths_are_validated(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model, (b1, b2) = _atwood()
        solver = SolverFBA(model, iterations=1)
        d = model.device
        targets = wp.zeros((1, 2), dtype=wp.vec3, device=d)
        middle = wp.zeros(1, dtype=wp.float64, device=d)
        with self.assertRaises(ValueError):  # no tendons yet
            solver.set_tendon_paths(targets, middle)
        solver.set_tendons([[b1, b2]], [[(0, 0, H), (0, 0, H)]], [[(0, 0, 1), (0, 0, 1)]], [0.2], [1.0])
        solver.set_tendon_paths(targets, middle)
        for bad in (
            (wp.zeros((2, 2), dtype=wp.vec3, device=d), middle),
            (wp.zeros((1, 2), dtype=wp.vec3d, device=d), middle),
            (targets, wp.zeros(1, dtype=wp.float32, device=d)),
            (np.zeros((1, 2, 3)), middle),
            (wp.zeros((1, 2), dtype=wp.vec3, device="cpu"), middle),
        ):
            with self.assertRaises(ValueError):
                solver.set_tendon_paths(*bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
