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
            [x0[CORNERS, 2].mean(), np.linalg.norm(x0[24] - x0[0])],
            [[0.0, 0.0, 1.0], diagonal],
        )
        held = self._run(model, rows)
        self.assertAlmostEqual(float(held[CORNERS, 2].mean()), 1.0, delta=1e-5)
        self.assertAlmostEqual(float(np.dot(diagonal, held[24] - held[0])), float(rows[2][1]), delta=1e-5)
        self.assertLess(float(held[12, 2]), 0.98)  # the middle sags onto the sphere top at 0.97
        self.assertGreater(float(held[12, 2]), 0.968)
        self.assertLess(float(self._run(model)[CORNERS, 2].mean()), 0.95)

    def _cloth_on_box(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(box, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))
        grid = {"dim_x": 4, "dim_y": 4, "cell_x": 0.1, "cell_y": 0.1, "tri_ke": 1.0e3, "tri_ka": 1.0e3, "edge_ke": 1.0}
        builder.add_cloth_grid(
            pos=wp.vec3(-0.2, -0.2, 1.005), rot=wp.quat_identity(), vel=wp.vec3(0.0), mass=0.2, **grid
        )
        return builder.finalize()

    def _run_on_box(self, model, solver, steps):
        """The state after ``steps`` steps of 5 ms and the particle velocities before the last one."""
        contacts, s0, s1 = model.contacts(), model.state(), model.state()
        for _ in range(steps):
            model.collide(s0, contacts)
            v = s0.particle_qd.numpy()
            solver.step(s0, s1, None, contacts, 0.005)
            s0, s1 = s1, s0
        return s0, v

    def test_rows_with_a_free_body(self):
        """A cloth on a free box on the ground, its corners held 5 cm over the box top. Its contacts on the box
        join the rigid NSN and the rows with them: the ground and the rows carry both weights, less the rate of
        change of the cloth's momentum, to float32 rounding of the cloth positions (5 kg ulp(1) / dt^2 = 0.024 N)."""
        model = self._cloth_on_box()
        solver = newton.solvers.SolverFBA(model, iterations=10)
        solver.set_particle_constraints([CORNERS], [[0.25] * 4], [1.05], directions=[[0.0, 0.0, 1.0]])
        s0, v = self._run_on_box(model, solver, 200)
        self.assertTrue(solver._two_way)
        self.assertAlmostEqual(float(s0.particle_q.numpy()[CORNERS, 2].mean()), 1.05, delta=1e-5)
        m_s, m_r = solver._rc_counts
        lam = solver._nsn_lam_apply_d.numpy()
        ground, held = lam[3 * m_s : 3 * (m_s + m_r) : 3].sum(), lam[solver._rc_n - 1]
        m = model.particle_mass.numpy()
        weight = (model.body_mass.numpy()[0] + m.sum()) * 9.81
        self.assertGreater(held, 30.0)  # of the cloth's 49 N
        self.assertAlmostEqual(ground + held - weight, (m @ (s0.particle_qd.numpy() - v))[2] / 0.005, delta=0.05)
        np.testing.assert_allclose(s0.body_q.numpy()[0], [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_length_row_with_a_free_body(self):
        """A length row stretches the diagonal of the cloth on the free box by 2 cm."""
        model = self._cloth_on_box()
        x0 = model.particle_q.numpy()
        length = float(np.linalg.norm(x0[24] - x0[0])) + 0.02
        solver = newton.solvers.SolverFBA(model, iterations=10)
        solver.set_particle_constraints([[24, 0]], [[1.0, -1.0]], [length])
        x = self._run_on_box(model, solver, 100)[0].particle_q.numpy()
        self.assertTrue(solver._two_way)
        self.assertAlmostEqual(float(np.linalg.norm(x[24] - x[0])), length, delta=1e-5)

    def test_needs_the_full_coulomb_mode(self):
        solver = newton.solvers.SolverFBA(self._model(), friction=False)
        with self.assertRaises(ValueError):
            solver.set_particle_constraints([[0]], [[1.0]], [1.0])

    def test_lite_rows_hold_a_particle_at_a_point(self):
        """Three rows of the lite NSN hold the middle particle of a cloth on the ground, first 2 mm into the
        ground, then 5 cm over it while they drag it at 0.2 m/s, and its contacts carry no force."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        grid = {"dim_x": 4, "dim_y": 4, "cell_x": 0.1, "cell_y": 0.1, "tri_ke": 1.0, "tri_ka": 1.0, "edge_ke": 0.01}
        builder.add_cloth_grid(pos=wp.vec3(-0.2, -0.2, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), mass=1.0, **grid)
        model = builder.finalize()
        solver = newton.solvers.SolverFBA(model, iterations=10, nsn_schur_mode="lite")
        contacts, s0, s1 = model.contacts(), model.state(), model.state()
        x0 = model.particle_q.numpy()[12]
        for k in range(60):
            target = x0 + ([0.0, 0.0, -0.002] if k < 30 else [0.2 * (k - 29) / 60.0, 0.0, 0.05])
            solver.set_particle_constraints([[12]] * 3, [[1.0]] * 3, target, np.eye(3))
            model.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
            if k in (29, 59):
                np.testing.assert_allclose(s0.particle_q.numpy()[12], target, atol=1e-6)
            if k == 29:
                m = solver._contact_count
                rows = np.flatnonzero(solver._linear_solver._row_particle_d.numpy()[: 3 * m] == 12)
                self.assertEqual(len(rows), 3)
                np.testing.assert_array_equal(solver._nsn_lam_apply_d.numpy()[rows], 0.0)

    def test_lite_rows_hold_a_stiff_cloth_at_points(self):
        """Rows of the lite NSN lift a 2 x 2 corner of a stiff cloth on the ground 5 cm in 1/3 s. The particles'
        blocks of A^-1 are a few percent of dt^2 / m here, where the rows lagged by up to 17 mm with the
        particles' contacts; they solve their own Delassus matrix and hold the corner on its targets."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        grid = {"dim_x": 4, "dim_y": 4, "cell_x": 0.1, "cell_y": 0.1, "tri_ke": 1e3, "tri_ka": 1e3, "edge_ke": 0.1}
        builder.add_cloth_grid(
            pos=wp.vec3(-0.2, -0.2, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), mass=0.01, **grid
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFBA(model, iterations=10, nsn_schur_mode="lite")
        contacts, s0, s1 = model.contacts(), model.state(), model.state()
        held = [0, 1, 5, 6]
        x0 = model.particle_q.numpy()[held]
        for k in range(20):
            target = x0 + np.array([0.0, 0.0, 0.05 * (k + 1) / 20])
            rows = np.repeat(held, 3)[:, None], np.ones((12, 1)), target.reshape(-1), np.tile(np.eye(3), (4, 1))
            solver.set_particle_constraints(*rows)
            model.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
            np.testing.assert_allclose(s0.particle_q.numpy()[held], target, atol=1e-6)

    def test_lite_rows_take_a_direction_and_one_particle(self):
        solver = newton.solvers.SolverFBA(self._model(), nsn_schur_mode="lite")
        with self.assertRaises(ValueError):
            solver.set_particle_constraints([[0]], [[1.0]], [1.0])
        with self.assertRaises(ValueError):
            solver.set_particle_constraints([[0, 1]], [[1.0, -1.0]], [0.0], [[0.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
