# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton

CORNERS = [0, 4, 20, 24]


class TestParticleConstraints(unittest.TestCase):
    """Bilateral rows on the particles of a 0.4 m cloth, solved with its contacts on a sphere below."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _model(self):
        builder = newton.ModelBuilder()
        builder.add_shape_sphere(-1, xform=wp.transform((0.2, 0.2, 0.92), wp.quat_identity()), radius=0.05)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            edge_ke=0.1,
            particle_radius=0.005,
        )
        return builder.finalize()

    def _run(self, model, rows=None, steps=60):
        solver = newton.solvers.SolverFBA(model, iterations=10)
        if rows is not None:
            solver.set_particle_constraints(*rows)
        pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.02)
        contacts, s0, s1 = pipeline.contacts(), model.state(), model.state()
        for _ in range(steps):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
        return s0.particle_q.numpy()

    def test_rows_hold_with_contacts(self):
        """One row holds the mean height of the corners, one the corner-to-corner extent along the diagonal.
        The held cloth sags onto the sphere; without the rows it drapes over it and its corners drop."""
        model = self._model()
        x0 = model.particle_q.numpy()
        diagonal = (x0[24] - x0[0]) / np.linalg.norm(x0[24] - x0[0])
        rows = (
            [CORNERS, [24, 0, -1, -1]],
            [[0.25] * 4, [1.0, -1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], diagonal],
            [x0[CORNERS, 2].mean(), np.linalg.norm(x0[24] - x0[0])],
        )
        held = self._run(model, rows)
        self.assertAlmostEqual(float(held[CORNERS, 2].mean()), 1.0, delta=1e-5)
        self.assertAlmostEqual(float(np.dot(diagonal, held[24] - held[0])), float(rows[3][1]), delta=1e-5)
        self.assertLess(float(held[12, 2]), 0.98)  # the middle sags onto the sphere top at 0.97
        self.assertGreater(float(held[12, 2]), 0.968)
        self.assertLess(float(self._run(model)[CORNERS, 2].mean()), 0.95)

    def test_needs_the_full_coulomb_mode(self):
        solver = newton.solvers.SolverFBA(self._model(), friction=False)
        with self.assertRaises(ValueError):
            solver.set_particle_constraints([[0]], [[1.0]], [[0.0, 0.0, 1.0]], [1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
