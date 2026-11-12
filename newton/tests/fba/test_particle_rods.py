# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the particle rods of SolverFBA: springs along a polyline and bending elements at its interior vertices."""

import unittest

import numpy as np

import newton
from newton._src.solvers.fba.linear_solver import SPRING_BENDING_FACTOR, SPRING_INERTIA_FACTOR, build_pd_system
from newton.solvers import SolverFBA


def _line(n, length=0.3, z=0.5):
    """``n + 1`` points along x from the origin, at height ``z``."""
    return np.c_[np.linspace(0.0, length, n + 1), np.zeros(n + 1), np.full(n + 1, z)]


def _discrete_tip_sag(segments, length, radius, young, density, gravity=9.81):
    """Small-deflection tip sag of a rod clamped by its first two particles under its own weight, from the rod's
    discrete bending energy: stencils ``[1/l_-, -(1/l_- + 1/l_+), 1/l_+]`` of weight ``EI / v`` and length-lumped
    loads."""
    n = segments + 1
    l = np.full(segments, length / segments)
    ei = young * np.pi * radius**4 / 4.0
    K = np.zeros((n, n))
    for i in range(1, n - 1):
        g = np.zeros(n)
        g[i - 1 : i + 2] = 1.0 / l[i - 1], -(1.0 / l[i - 1] + 1.0 / l[i]), 1.0 / l[i]
        K += ei / (0.5 * (l[i - 1] + l[i])) * np.outer(g, g)
    lumped = np.zeros(n)
    lumped[:-1] += 0.5 * l
    lumped[1:] += 0.5 * l
    load = density * np.pi * radius**2 * lumped * gravity
    free = np.arange(2, n)
    return np.linalg.solve(K[np.ix_(free, free)], load[free])[-1]


class TestAddParticleRod(unittest.TestCase):
    """Authoring particle rods with :meth:`SolverFBA.add_particle_rod`."""

    def test_rod_has_lumped_masses_axial_springs_and_bends(self):
        """Give each particle the line density times half its segments' length and the rod's radius, each segment
        a spring of stiffness ``E pi r^2 / l`` and each interior vertex a bend of stiffness ``E pi r^4 / 4``."""
        builder = newton.ModelBuilder()
        points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.2, 0.0], [0.1, 0.2, 0.3]])
        particles = SolverFBA.add_particle_rod(builder, points, radius=0.01, youngs_modulus=2.0e9, density=1500.0)
        model = builder.finalize(device="cpu")
        self.assertEqual(particles, [0, 1, 2, 3])
        area = np.pi * 0.01**2
        segment = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            model.particle_mass.numpy(), 1500.0 * area * 0.5 * np.array([0.1, 0.3, 0.5, 0.3]), rtol=1e-6
        )
        np.testing.assert_allclose(model.particle_radius.numpy(), 0.01)
        np.testing.assert_array_equal(model.spring_indices.numpy().reshape(-1, 2), [[0, 1], [1, 2], [2, 3]])
        np.testing.assert_allclose(model.spring_stiffness.numpy(), 2.0e9 * area / segment, rtol=1e-6)
        np.testing.assert_array_equal(model.fba.rod_bend_particles.numpy(), [[0, 1, 2], [1, 2, 3]])
        np.testing.assert_allclose(model.fba.rod_bend_stiffness.numpy(), 2.0e9 * np.pi * 0.01**4 / 4.0, rtol=1e-6)

    def test_total_mass_and_bending_stiffness_override(self):
        """Spread a given total mass evenly over the length, and take a given bending stiffness."""
        builder = newton.ModelBuilder()
        SolverFBA.add_particle_rod(
            builder, _line(4, length=0.4), radius=0.002, youngs_modulus=1.0e6, mass=0.2, bending_stiffness=1.0e-5
        )
        model = builder.finalize(device="cpu")
        np.testing.assert_allclose(model.particle_mass.numpy(), [0.025, 0.05, 0.05, 0.05, 0.025], rtol=1e-6)
        np.testing.assert_allclose(model.fba.rod_bend_stiffness.numpy(), 1.0e-5, rtol=1e-6)

    def test_invalid_rods_are_rejected(self):
        """Reject a rod of fewer than two points, repeated points, no or two mass inputs, or a bad radius."""
        builder = newton.ModelBuilder()
        for kwargs in (
            {"points": _line(0)[:1], "density": 1.0},
            {"points": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "density": 1.0},
            {"points": _line(2)},
            {"points": _line(2), "density": 1.0, "mass": 1.0},
            {"points": _line(2), "density": 1.0, "radius": 0.0},
        ):
            args = {"radius": 0.01, "youngs_modulus": 1.0e6, **kwargs}
            with self.assertRaises(ValueError):
                SolverFBA.add_particle_rod(builder, **args)

    def test_rod_bends_follow_their_rod_into_each_world(self):
        """Offset a rod's bending elements by the particles before it when its builder becomes a world."""
        rod = newton.ModelBuilder()
        SolverFBA.add_particle_rod(rod, _line(3), radius=0.005, youngs_modulus=1.0e8, density=1000.0)
        builder = newton.ModelBuilder()
        builder.add_world(rod)
        builder.add_world(rod)
        model = builder.finalize(device="cpu")
        np.testing.assert_array_equal(
            model.fba.rod_bend_particles.numpy(), [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]]
        )

    def test_rod_bends_across_worlds_are_rejected(self):
        """Reject a bending element whose particles belong to different worlds."""
        rod = newton.ModelBuilder()
        SolverFBA.add_particle_rod(rod, _line(1), radius=0.005, youngs_modulus=1.0e8, density=1000.0)
        builder = newton.ModelBuilder()
        builder.add_world(rod)
        builder.add_world(rod)
        SolverFBA.register_custom_attributes(builder)
        builder.add_custom_values(**{"fba:rod_bend_particles": (0, 1, 2), "fba:rod_bend_stiffness": 1.0})
        with self.assertRaisesRegex(ValueError, "different worlds"):
            SolverFBA(builder.finalize())


class TestRodSystem(unittest.TestCase):
    """The rods' terms of the constant PD system matrix."""

    def test_bend_weighs_its_stencil_by_ei_over_its_length(self):
        """Add ``EI / v g g^T`` for each bend, ``g = [1/l_-, -(1/l_- + 1/l_+), 1/l_+]``, whose energy at a small
        turn ``phi`` is ``EI / 2 kappa^2 v`` with ``kappa = phi / v``; damp it with ``beta`` times the same."""
        builder = newton.ModelBuilder()
        points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]
        SolverFBA.add_particle_rod(builder, points, radius=0.01, youngs_modulus=1.0e7, density=1000.0)
        model = builder.finalize(device="cpu")
        beta = 0.01
        _, meta = build_pd_system(model, dt=0.01, pin_stiffness=1.0e6, rayleigh_beta=np.full(3, beta))
        ei, v = 1.0e7 * np.pi * 0.01**4 / 4.0, 0.15
        g = np.array([10.0, -15.0, 5.0])
        np.testing.assert_allclose(meta["rod_bend_weight"], [ei / v], rtol=1e-6)
        k_spring = meta["spring_stiffness"]
        springs = np.array([[k_spring[0], -k_spring[0], 0.0], [-k_spring[0], k_spring.sum(), -k_spring[1]]])
        springs = np.vstack([springs, [0.0, -k_spring[1], k_spring[1]]])
        np.testing.assert_allclose(meta["damping"].toarray(), beta * (ei / v * np.outer(g, g) + springs), rtol=1e-6)
        # A small turn phi of the second segment about the middle vertex.
        phi = 1.0e-3
        x = np.array([[0.0, 0.0], [0.1, 0.0], [0.1 + 0.2 * np.cos(phi), 0.2 * np.sin(phi)]])
        energy = 0.5 * ei / v * np.sum((g @ x) ** 2)
        np.testing.assert_allclose(energy, 0.5 * ei * (phi / v) ** 2 * v, rtol=1e-6)

    def test_rod_springs_add_the_bending_term_to_their_penalty(self):
        """Raise the penalty of a rod's springs by ``0.1 EI / l^3`` over that of plain springs."""
        builder = newton.ModelBuilder()
        SolverFBA.add_particle_rod(builder, _line(3, length=0.3), radius=0.005, youngs_modulus=1.0e8, density=1000.0)
        model = builder.finalize(device="cpu")
        solver = SolverFBA(model)
        dt = 0.01
        _, meta = build_pd_system(
            model, dt=dt, pin_stiffness=1.0e6, spring_bending_stiffness=solver._spring_bending_stiffness()
        )
        mass = model.particle_mass.numpy().astype(np.float64)
        ei = 1.0e8 * np.pi * 0.005**4 / 4.0
        lighter = np.minimum(mass[:-1], mass[1:])
        expected = SPRING_INERTIA_FACTOR * lighter / dt**2 + SPRING_BENDING_FACTOR * ei / 0.1**3
        np.testing.assert_allclose(meta["spring_penalty"], expected, rtol=1e-5)


class TestSolverFBAParticleRods(unittest.TestCase):
    """Particle rods stepped by SolverFBA."""

    def _cantilever_sag(self, young, segments=20, length=0.3, radius=0.005, density=1000.0):
        builder = newton.ModelBuilder()
        particles = SolverFBA.add_particle_rod(
            builder, _line(segments, length), radius=radius, youngs_modulus=young, density=density
        )
        builder.particle_mass[particles[0]] = 0.0
        builder.particle_mass[particles[1]] = 0.0
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10, rayleigh_alpha=5.0)
        s0, s1 = model.state(), model.state()
        for _ in range(300):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        self.assertLess(np.abs(s0.particle_qd.numpy()).max(), 1.0e-3)
        return 0.5 - float(s0.particle_q.numpy()[-1, 2])

    def test_cantilever_sags_by_its_bending_stiffness(self):
        """Settle a rod clamped by its first two particles to the small-deflection sag of its discrete bending
        energy: 1.43 mm at E = 1e9 Pa and 14.3 mm at E = 1e8 Pa for 0.3 m of radius 5 mm."""
        for young, rtol in ((1.0e9, 1.0e-3), (1.0e8, 5.0e-3)):
            expected = _discrete_tip_sag(20, 0.3, 0.005, young, 1000.0)
            np.testing.assert_allclose(self._cantilever_sag(young), expected, rtol=rtol)

    def test_hanging_rod_stays_straight_and_holds_its_weight(self):
        """Hang a rod from its pinned top particle: it stays straight down, each segment stretched by the weight
        below it over the segment's stiffness."""
        builder = newton.ModelBuilder()
        points = np.c_[np.zeros(9), np.zeros(9), np.linspace(1.0, 0.6, 9)]
        particles = SolverFBA.add_particle_rod(builder, points, radius=0.004, youngs_modulus=1.0e6, density=1200.0)
        builder.particle_mass[particles[0]] = 0.0
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10, rayleigh_alpha=5.0)
        s0, s1 = model.state(), model.state()
        for _ in range(300):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        x = s0.particle_q.numpy().astype(np.float64)
        np.testing.assert_allclose(x[:, :2], 0.0, atol=1.0e-6)
        stretch = -np.diff(x[:, 2]) - 0.05
        mass = model.particle_mass.numpy().astype(np.float64)
        weight_below = np.cumsum(mass[::-1])[::-1][1:] * 9.81
        stiffness = model.spring_stiffness.numpy().astype(np.float64)
        np.testing.assert_allclose(stretch, weight_below / stiffness, rtol=1.0e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
