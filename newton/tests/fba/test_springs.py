# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the springs of SolverFBA, ADMM stretch elements of the PD system."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.linear_solver import SPRING_INERTIA_FACTOR, build_pd_system
from newton.solvers import SolverFBA

_GRAVITY = np.array([0.0, 0.0, -9.81])


def _chain(builder=None, n=6, seg=0.05, mass=0.01, ke=1.0e3, x=0.0):
    """A chain of ``n`` springs of rest length ``seg`` hanging from a particle pinned by a zero mass at z = 1."""
    builder = newton.ModelBuilder() if builder is None else builder
    start = builder.particle_count
    for i in range(n + 1):
        builder.add_particle(pos=(x, 0.0, 1.0 - seg * i), vel=(0.0, 0.0, 0.0), mass=0.0 if i == 0 else mass, radius=0.0)
    for i in range(n):
        builder.add_spring(start + i, start + i + 1, ke=ke, kd=0.0, control=0.0)
    return builder


def _implicit_euler(x0, v0, mass, springs, rest, ke, dt, pin_weight, pinned):
    """One implicit Euler step of particles joined by springs, solved by Newton's method:
    ``M (x - x~) / dt^2 + grad U(x) + w (x_p - x0_p) = 0`` with ``x~ = x0 + dt v0 + dt^2 g`` for the free particles,
    ``U`` the springs' energy ``k/2 (|x_b - x_a| - l)^2`` and ``w`` the pins' weight."""
    n = len(x0)
    free = mass > 0.0
    target = np.where(free[:, None], x0 + dt * v0 + dt * dt * _GRAVITY, x0)
    x = target.copy()
    for _ in range(50):
        F = (mass[:, None] / dt**2) * (x - target)
        F[pinned] += pin_weight * (x[pinned] - x0[pinned])
        K = np.zeros((3 * n, 3 * n))
        diag = mass / dt**2 + np.where(pinned, pin_weight, 0.0)
        K[np.arange(3 * n), np.arange(3 * n)] = np.repeat(diag, 3)
        for (a, b), l0, k in zip(springs, rest, ke, strict=True):
            d = x[b] - x[a]
            length = np.linalg.norm(d)
            u = d / length
            f = k * (length - l0) * u
            F[a] -= f
            F[b] += f
            Ke = k * ((1.0 - l0 / length) * np.eye(3) + (l0 / length) * np.outer(u, u))
            for i, j, sign in ((a, a, 1.0), (b, b, 1.0), (a, b, -1.0), (b, a, -1.0)):
                K[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += sign * Ke
        dx = np.linalg.solve(K, -F.reshape(-1)).reshape(n, 3)
        x += dx
        if np.abs(dx).max() < 1.0e-14:
            break
    return x


class TestSpringSystem(unittest.TestCase):
    """The springs' terms of the constant PD system matrix and of its Rayleigh damping matrix."""

    def test_springs_weigh_their_penalty_in_A_and_their_stiffness_in_C(self):
        """Weigh each spring's stencil ``x_b - x_a`` by its penalty ``60 m / dt^2`` in ``A``, ``m`` the mean mass of
        its particles, and by its stiffness in the damping matrix ``C``."""
        builder = newton.ModelBuilder()
        for i, m in enumerate((0.02, 0.04, 0.06)):
            builder.add_particle(pos=(0.1 * i, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=m, radius=0.0)
        builder.add_spring(0, 1, ke=500.0, kd=0.0, control=0.0)
        builder.add_spring(1, 2, ke=700.0, kd=0.0, control=0.0)
        model = builder.finalize(device="cpu")
        dt, beta = 0.01, 0.002
        A, meta = build_pd_system(model, dt=dt, pin_stiffness=1.0e6, rayleigh_beta=np.full(3, beta))
        A = A.toarray()
        rho = SPRING_INERTIA_FACTOR * np.array([0.03, 0.05]) / dt**2
        np.testing.assert_allclose(meta["spring_penalty"], rho)
        np.testing.assert_allclose(meta["spring_stiffness"], [500.0, 700.0])
        np.testing.assert_allclose(meta["spring_rest_length"], [0.1, 0.1], rtol=1e-6)
        C = meta["damping"].toarray()
        np.testing.assert_allclose(
            C, beta * np.array([[500.0, -500.0, 0.0], [-500.0, 1200.0, -700.0], [0.0, -700.0, 700.0]])
        )
        mass = np.array([0.02, 0.04, 0.06]) / dt**2
        stencil = np.array([[rho[0], -rho[0], 0.0], [-rho[0], rho[0] + rho[1], -rho[1]], [0.0, -rho[1], rho[1]]])
        np.testing.assert_allclose(A, np.diag(mass) + stencil + C / dt)

    def test_springs_between_held_particles_take_their_stiffness(self):
        """Weigh a spring whose particles are both pinned by its stiffness, and leave out springs without one."""
        builder = newton.ModelBuilder()
        for i in range(3):
            builder.add_particle(pos=(0.1 * i, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.0 if i < 2 else 0.01, radius=0.0)
        builder.add_spring(0, 1, ke=300.0, kd=0.0, control=0.0)
        builder.add_spring(1, 2, ke=0.0, kd=0.0, control=0.0)
        model = builder.finalize(device="cpu")
        _, meta = build_pd_system(model, dt=0.01, pin_stiffness=1.0e6)
        np.testing.assert_array_equal(meta["spring_indices"], [[0, 1]])
        np.testing.assert_allclose(meta["spring_penalty"], [300.0])


class TestSolverFBASprings(unittest.TestCase):
    """Springs stepped by SolverFBA: their ADMM local steps converge to the implicit Euler step of the springs."""

    def test_hanging_chain_holds_its_weight(self):
        """Settle a chain hanging from a pinned particle: each spring stretches by the weight below it over its
        stiffness, as the exact static balance has it."""
        n, seg, mass, ke = 6, 0.05, 0.01, 1.0e3
        model = _chain(n=n, seg=seg, mass=mass, ke=ke).finalize()
        solver = SolverFBA(model, iterations=20, rayleigh_alpha=5.0)
        s0, s1 = model.state(), model.state()
        for _ in range(600):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        x = s0.particle_q.numpy().astype(np.float64)
        stretch = np.linalg.norm(np.diff(x, axis=0), axis=1) - seg
        weight_below = (n - np.arange(n)) * mass * 9.81
        np.testing.assert_allclose(stretch, weight_below / ke, rtol=2.0e-3)
        self.assertLess(np.abs(s0.particle_qd.numpy()).max(), 1.0e-4)

    def test_step_converges_to_implicit_euler(self):
        """Step a chain thrown with random velocities: with enough iterations the step reaches the implicit Euler step
        of the springs, which Newton's method solves independently."""
        n, seg, mass, ke, dt = 6, 0.05, 0.01, 1.0e3, 0.01
        model = _chain(n=n, seg=seg, mass=mass, ke=ke).finalize()
        solver = SolverFBA(model, iterations=1600)
        s0, s1 = model.state(), model.state()
        x0 = s0.particle_q.numpy().astype(np.float64)
        v0 = np.random.default_rng(0).normal(scale=0.3, size=x0.shape)
        v0[0] = 0.0
        s0.particle_qd.assign(v0.astype(np.float32))
        solver.step(s0, s1, None, None, dt)
        masses = np.r_[0.0, np.full(n, mass)]
        springs = np.stack([np.arange(n), np.arange(1, n + 1)], axis=1)
        pinned = masses == 0.0
        expected = _implicit_euler(
            x0, v0, masses, springs, np.full(n, seg), np.full(n, ke), dt, solver.pin_stiffness, pinned
        )
        np.testing.assert_allclose(s1.particle_q.numpy(), expected, atol=1.0e-6)
        self.assertGreater(np.abs(expected - x0).max(), 5.0e-3)

    def test_reset_clears_the_selected_worlds_duals(self):
        """Keep the springs' duals across steps, clear them in the worlds a reset selects, and in all with none."""
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.add_world(_chain(n=3))
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s0, s1 = model.state(), model.state()
        for _ in range(3):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        dual = solver._spring_dual_d.numpy()
        self.assertTrue((np.abs(dual) > 0.0).any(axis=1).all())
        world = model.particle_world.numpy()[model.spring_indices.numpy().reshape(-1, 2)[:, 0]]
        solver.reset(s0, wp.array([False, True, False], dtype=wp.bool, device=model.device), flags=0)
        dual = solver._spring_dual_d.numpy()
        np.testing.assert_array_equal(dual[world == 1], 0.0)
        self.assertTrue((np.abs(dual[world == 0]) > 0.0).any(axis=1).all())
        solver.reset(s0, flags=0)
        np.testing.assert_array_equal(solver._spring_dual_d.numpy(), 0.0)
        np.testing.assert_array_equal(solver._spring_split_d.numpy(), 0.0)

    def test_springs_across_worlds_are_rejected(self):
        """Reject a spring that joins particles of different worlds."""
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.add_world(_chain(n=1))
        builder.add_spring(1, 3, ke=100.0, kd=0.0, control=0.0)
        with self.assertRaisesRegex(ValueError, "different worlds"):
            SolverFBA(builder.finalize())


if __name__ == "__main__":
    unittest.main(verbosity=2)
