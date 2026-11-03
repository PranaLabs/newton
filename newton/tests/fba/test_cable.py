# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.example_cable_gripper_fba import Cable
from newton.solvers import SolverFBA


class TestCable(unittest.TestCase):
    """An 8 x 2 x 2 cm finger of 5 mm cells clamped at x = 0, with a cable of seven 1 cm segments at height ``z``."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _finger(self, z=0.018):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        mu, lam = 1.0e6 / 2.6, 1.0e6 * 0.3 / (1.3 * 0.4)  # young 1 MPa, poisson 0.3
        builder.add_soft_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=16,
            dim_y=4,
            dim_z=4,
            cell_x=0.005,
            cell_y=0.005,
            cell_z=0.005,
            density=1000.0,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
        )
        model = builder.finalize()
        x = model.particle_q.numpy()
        mass = np.where(x[:, 0] < 1e-6, 0.0, model.particle_mass.numpy())
        model.particle_mass.assign(mass)
        model.particle_inv_mass.assign(np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0))
        points = np.array([[s, 0.01, z] for s in np.linspace(0.005, 0.075, 8)])
        cable = Cable(model, np.stack([points[:-1], points[1:]], axis=1))
        return model, cable, SolverFBA(model, iterations=10, stretching_model="neohookean")

    def _pull(self, ratio, steps=200):
        """Shorten the cable linearly to ``ratio`` of its rest length over 1 s, then hold it."""
        model, cable, solver = self._finger()
        s0, s1 = model.state(), model.state()
        for step in range(steps):
            cable.length = cable.rest_length * (1.0 - (1.0 - ratio) * min(1.0, step / 100))
            solver.set_particle_constraints(*cable.rows())
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        return model, cable, s0.particle_q.numpy()

    def test_pulled_cable_bends_the_finger(self):
        """Bend the finger toward a cable pulled 10% shorter, every segment keeping its share of the length.

        A cable at its rest length leaves the finger where it is.
        """
        model, cable, x = self._pull(0.9)
        x0 = model.particle_q.numpy()
        tip = np.isclose(x0[:, 0], x0[:, 0].max())
        self.assertGreater(float(x[tip, 2].mean() - x0[tip, 2].mean()), 0.01)
        p = np.einsum("pk,pkd->pd", cable.weights, x[cable.particles])
        segments = np.linalg.norm(p[1::2] - p[0::2], axis=1)
        np.testing.assert_allclose(segments, 0.9 * cable.rest_length * cable.share, atol=1e-5)
        np.testing.assert_allclose(self._pull(1.0, steps=50)[2], x0, atol=1e-6)

    def test_sudden_pull_near_the_axis(self):
        """Settle a finger whose cable, 5 mm over its axis, is pulled 10% shorter at once.

        The cable compresses the finger with up to 22 N, where 5 N at the tip buckles it. The length rows keep the taut
        cable's transverse stiffness, so the finger settles; rows held along their direction at the start of each step
        would let it flutter by 10 cm.
        """
        model, cable, solver = self._finger(z=0.015)
        x0 = model.particle_q.numpy()
        tip = np.isclose(x0[:, 0], x0[:, 0].max())
        cable.length = 0.9 * cable.rest_length
        solver.set_particle_constraints(*cable.rows())
        s0, s1, heights = model.state(), model.state(), []
        for _ in range(250):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            heights.append(float(s0.particle_q.numpy()[tip, 2].mean() - x0[tip, 2].mean()))
        self.assertGreater(heights[-1], 0.01)
        self.assertLess(max(heights[-100:]) - min(heights[-100:]), 1e-3)

    def test_cable_pulls_but_does_not_push(self):
        """Hold a cable shortened to half its rest length at 85% of it, and a lengthened one at its rest length."""
        _, cable, _ = self._finger()
        for ratio, held in ((0.5, 0.85), (1.5, 1.0)):
            cable.length = ratio * cable.rest_length
            self.assertAlmostEqual(float(cable.rows()[2].sum()), held * cable.rest_length, places=12)

    def test_points_outside_the_body(self):
        """Refuse a cable whose points lie outside the tetrahedral body."""
        model, _, _ = self._finger()
        with self.assertRaises(ValueError):
            Cable(model, [[0.01, 0.01, 0.01], [0.2, 0.01, 0.01]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
