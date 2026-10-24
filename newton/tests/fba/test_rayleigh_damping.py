# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton

_CLOTH = {
    "rot": wp.quat_identity(),
    "dim_x": 6,
    "dim_y": 6,
    "cell_x": 0.05,
    "cell_y": 0.05,
    "mass": 0.01,
    "tri_ke": 500.0,
    "tri_ka": 1.0e3,
    "edge_ke": 1.0e-2,
    "particle_radius": 0.0,
}
_V0 = (0.3, -0.2, 0.1)


def _beam(builder, y, cells=(8, 2, 2), length=0.8, width=0.1, young=2.0e5):
    """A soft cantilever along x, clamped at x = 0, with a Poisson ratio of 0.3, weighing 0.2 kg."""
    cell = (length / cells[0], width / cells[1], width / cells[2])
    start = builder.particle_count
    builder.add_soft_grid(
        pos=wp.vec3(0.0, y, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=cells[0],
        dim_y=cells[1],
        dim_z=cells[2],
        cell_x=cell[0],
        cell_y=cell[1],
        cell_z=cell[2],
        density=0.2 / (cells[0] * (cells[1] + 1) * (cells[2] + 1) * np.prod(cell)),
        k_mu=young / 2.6,
        k_lambda=young * 0.3 / (1.3 * 0.4),
        k_damp=0.0,
        fix_left=True,
    )
    x0 = np.array(builder.particle_q[start:])
    tip = start + int(np.argmin(np.linalg.norm(x0 - [length, y + 0.5 * width, 1.0 + 0.5 * width], axis=1)))
    return np.arange(start, builder.particle_count), tip


class TestRayleighDamping(unittest.TestCase):
    """Rayleigh damping of SolverFBA, the forces -(alpha M + beta K) v integrated implicitly."""

    def _run(self, model, steps, dt=0.01, state=None, **damping):
        solver = newton.solvers.SolverFBA(model, iterations=10, **damping)
        s0, s1 = state if state is not None else model.state(), model.state()
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
        return s0

    def _free_cloth(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_cloth_grid(pos=wp.vec3(0.0), vel=wp.vec3(*_V0), **_CLOTH)
        return builder.finalize()

    def _cantilevers(self, count=1):
        builder = newton.ModelBuilder()
        beams = [_beam(builder, 0.3 * k) for k in range(count)]
        return builder.finalize(), beams

    def test_zero_damping_changes_nothing(self):
        """Step a cantilever with both coefficients zero, as scalars and as arrays, exactly as without them."""
        model, _ = self._cantilevers()
        plain = self._run(model, 20).particle_q.numpy()
        zero = self._run(model, 20, rayleigh_alpha=0.0, rayleigh_beta=np.zeros(model.particle_count))
        np.testing.assert_array_equal(zero.particle_q.numpy(), plain)

    def test_mass_damping_slows_a_free_cloth_by_implicit_euler(self):
        """Slow a cloth that moves uniformly without forces by 1 / (1 + alpha dt) each step."""
        alpha, steps, dt = 10.0, 20, 0.01
        v = self._run(self._free_cloth(), steps, dt, rayleigh_alpha=alpha).particle_qd.numpy()
        expected = np.array(_V0) / (1.0 + alpha * dt) ** steps
        np.testing.assert_allclose(v, np.broadcast_to(expected, v.shape), atol=2.0e-5)

    def test_stiffness_damping_spares_translation(self):
        """Keep a uniformly moving cloth's velocity under stiffness damping: K holds no translation."""
        v = self._run(self._free_cloth(), 20, rayleigh_beta=0.1).particle_qd.numpy()
        np.testing.assert_allclose(v, np.broadcast_to(_V0, v.shape), atol=2.0e-5)

    def test_stiffness_damping_brings_a_cantilever_to_rest(self):
        """Bring a soft cantilever that gravity sets swinging to rest at its sag, while it swings on without damping.

        The tip speeds over the last 0.5 s of 3 s compare."""
        speeds = {}
        for beta in (0.0, 1.0e-4):
            model, ((_, tip),) = self._cantilevers()
            solver = newton.solvers.SolverFBA(model, iterations=10, rayleigh_beta=beta)
            s0, s1, peak = model.state(), model.state(), 0.0
            for k in range(600):
                solver.step(s0, s1, None, None, 0.005)
                s0, s1 = s1, s0
                if k >= 500:
                    peak = max(peak, float(np.linalg.norm(s0.particle_qd.numpy()[tip])))
            speeds[beta] = peak
            self.assertLess(s0.particle_q.numpy()[tip, 2], 1.05 - 0.01, "the cantilever does not sag")
        self.assertGreater(speeds[0.0], 0.05)
        self.assertLess(speeds[1.0e-4], 0.1 * speeds[0.0])

    def test_per_particle_coefficients_damp_one_body(self):
        """Damp only the second of two cantilevers when only its particles carry a coefficient: the first moves as
        without damping."""
        model, ((first, _), (second, _)) = self._cantilevers(count=2)
        beta = np.zeros(model.particle_count)
        beta[second] = 1.0e-4
        plain = self._run(model, 60, dt=0.005).particle_q.numpy()
        damped = self._run(model, 60, dt=0.005, rayleigh_beta=beta).particle_q.numpy()
        np.testing.assert_allclose(damped[first], plain[first], atol=1.0e-6)
        self.assertGreater(np.abs(damped[second] - plain[second]).max(), 1.0e-3)

    def test_time_step_change_refactors_the_damping(self):
        """Take a step at a new time step as a solver set up at that time step from the same state does: the damping
        enters the system matrix and the right-hand side together."""
        model, _ = self._cantilevers()
        damping = {"rayleigh_alpha": 2.0, "rayleigh_beta": 1.0e-4}
        solver = newton.solvers.SolverFBA(model, iterations=10, **damping)
        s0, s1 = model.state(), model.state()
        for _ in range(10):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        start = model.state()
        start.particle_q.assign(s0.particle_q)
        start.particle_qd.assign(s0.particle_qd)
        solver.step(s0, s1, None, None, 0.004)
        fresh = self._run(model, 1, dt=0.004, state=start, **damping)
        np.testing.assert_allclose(s1.particle_q.numpy(), fresh.particle_q.numpy(), atol=1.0e-6)

    def _spin(self, **damping):
        """Angular momentum per unit mass [m^2/s] of a free cloth that spins about its normal at 2 rad/s, after 0.5 s."""
        model = self._free_cloth()
        state = model.state()
        x = state.particle_q.numpy()
        state.particle_qd.assign(np.cross([0.0, 0.0, 2.0], x - x.mean(axis=0)).astype(np.float32))
        end = self._run(model, 50, 0.01, state=state, **damping)
        x, v = end.particle_q.numpy(), end.particle_qd.numpy()
        return float(np.cross(x - x.mean(axis=0), v)[:, 2].sum())

    def _breathe(self, **damping):
        """Positions of a free cloth stretched by 5 % at rest, after two steps of 10 ms."""
        model = self._free_cloth()
        state = model.state()
        x = state.particle_q.numpy()
        state.particle_q.assign((x.mean(axis=0) + 1.05 * (x - x.mean(axis=0))).astype(np.float32))
        state.particle_qd.zero_()
        return self._run(model, 2, 0.01, state=state, **damping).particle_q.numpy()

    def test_corotated_stiffness_damping_spares_rotation(self):
        """Keep a free cloth's spin under corotated stiffness damping as without damping, while the constant damping
        brakes it."""
        free = self._spin()
        constant = self._spin(rayleigh_beta=0.05)
        corotated = self._spin(rayleigh_beta=0.05, rayleigh_corotated=True)
        self.assertLess(constant, 0.9 * free)
        self.assertAlmostEqual(corotated / free, 1.0, delta=1.0e-3)

    def test_corotated_stiffness_damping_damps_stretch(self):
        """Damp the breathing of a stretched free cloth as the constant damping does: without rotation, the two
        agree."""
        free = self._breathe()
        constant = self._breathe(rayleigh_beta=0.05)
        corotated = self._breathe(rayleigh_beta=0.05, rayleigh_corotated=True)
        self.assertGreater(np.abs(constant - free).max(), 1.0e-3)
        np.testing.assert_allclose(corotated, constant, atol=1.0e-5)

    def test_corotated_damping_needs_arap(self):
        """Refuse corotated damping with the corotational and Neo-Hookean stretching models, and without
        stiffness-proportional damping to apply."""
        with self.assertRaises(ValueError):
            newton.solvers.SolverFBA(
                self._free_cloth(), stretching_model="corotational", rayleigh_beta=0.01, rayleigh_corotated=True
            )
        with self.assertRaises(ValueError):
            newton.solvers.SolverFBA(self._free_cloth(), rayleigh_corotated=True)

    def test_refuse_negative_or_misshaped_coefficients(self):
        """Refuse negative or non-finite coefficients and arrays without one value per particle."""
        model = self._free_cloth()
        for damping in (
            {"rayleigh_alpha": -1.0},
            {"rayleigh_beta": np.inf},
            {"rayleigh_beta": np.zeros(model.particle_count + 1)},
            {"rayleigh_alpha": np.zeros((model.particle_count, 2))},
        ):
            with self.subTest(**{k: np.shape(v) or v for k, v in damping.items()}), self.assertRaises(ValueError):
                newton.solvers.SolverFBA(model, **damping)


class TestRayleighDampingContacts(unittest.TestCase):
    """Mass damping with the lite contact operator, which takes the mass term of the system matrix, damping
    included, for a particle's response."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def test_lite_contacts_hold_their_distance_with_mass_damping(self):
        """Rest a cloth on a box at the contact distance with strong mass damping and the lite contact operator.

        With the undamped mass in its response, the operator would leave the cloth 1.2 mm inside the contact distance.
        """
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.05)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.15, -0.15, 0.02), vel=wp.vec3(0.0), **{**_CLOTH, "particle_radius": 0.005}
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFBA(model, iterations=10, nsn_schur_mode="lite", rayleigh_alpha=50.0)
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
        contacts, s0, s1 = pipeline.contacts(), model.state(), model.state()
        for _ in range(120):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
        np.testing.assert_allclose(s0.particle_q.numpy()[:, 2], 0.005, atol=1.0e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
