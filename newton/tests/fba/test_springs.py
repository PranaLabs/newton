# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the springs of SolverFBA, ADMM stretch elements of the PD system."""

import itertools
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


def _implicit_euler(x0, v0, mass, springs, rest, ke, dt, pin_weight, pinned, damping=None):
    """One implicit Euler step of particles joined by springs, solved by Newton's method:
    ``M (x - x~) / dt^2 + grad U(x) + w (x_p - x0_p) = 0`` with ``x~ = x0 + dt v0 + dt^2 g`` for the free particles,
    ``U`` the springs' energy ``k/2 (|x_b - x_a| - l)^2`` and ``w`` the pins' weight. With ``damping`` ``c`` [N/m]
    per spring, each spring also pulls with ``c (|x_b - x_a| - r_0)`` along itself, ``r_0`` its length in ``x0``."""
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
        c_all = np.zeros(len(springs)) if damping is None else damping
        for (a, b), l0, k, c in zip(springs, rest, ke, c_all, strict=True):
            d = x[b] - x[a]
            length = np.linalg.norm(d)
            u = d / length
            r0 = np.linalg.norm(x0[b] - x0[a])
            f = (k * (length - l0) + c * (length - r0)) * u
            F[a] -= f
            F[b] += f
            # The force (k + c) (length - l*) u with l* = (k l0 + c r0) / (k + c).
            rest_eff = (k * l0 + c * r0) / (k + c)
            Ke = (k + c) * ((1.0 - rest_eff / length) * np.eye(3) + (rest_eff / length) * np.outer(u, u))
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

    def test_stiff_chain_converges_in_few_iterations(self):
        """Step a chain of 40 springs, a hundred times stiffer than their penalty, thrown down faster toward its free
        end: the chains' tension solve takes the step to the implicit Euler step in 10 iterations, where the dual
        steps alone leave stretch waves of several times the step."""
        n, seg, mass, ke, dt = 40, 0.025, 1.0e-3, 1.0e5, 0.01
        model = _chain(n=n, seg=seg, mass=mass, ke=ke).finalize()
        solver = SolverFBA(model, iterations=10)
        s0, s1 = model.state(), model.state()
        x0 = s0.particle_q.numpy().astype(np.float64)
        v0 = np.zeros_like(x0)
        v0[1:, 2] = np.linspace(0.0, -3.0, n)
        s0.particle_qd.assign(v0.astype(np.float32))
        solver.step(s0, s1, None, None, dt)
        masses = np.r_[0.0, np.full(n, mass)]
        springs = np.stack([np.arange(n), np.arange(1, n + 1)], axis=1)
        expected = _implicit_euler(
            x0, v0, masses, springs, np.full(n, seg), np.full(n, ke), dt, solver.pin_stiffness, masses == 0.0
        )
        np.testing.assert_allclose(s1.particle_q.numpy(), expected, atol=1.0e-6)
        self.assertGreater(np.abs(expected - x0).max(), 1.0e-3)

    def test_compressed_chain_keeps_its_springs_upright(self):
        """Throw a chain of 40 stiff springs up toward its pin at 4 m/s: holding it back takes duals that push harder
        than the penalty times the springs' lengths, and each split stays along its spring, so 10 iterations reach
        the implicit Euler step, where splits taken along the dual's side would turn springs over."""
        n, seg, mass, ke, dt = 40, 0.05, 1.0e-3, 1.0e5, 0.01
        model = _chain(n=n, seg=seg, mass=mass, ke=ke).finalize()
        solver = SolverFBA(model, iterations=10)
        s0, s1 = model.state(), model.state()
        x0 = s0.particle_q.numpy().astype(np.float64)
        v0 = np.zeros_like(x0)
        v0[1:, 2] = 4.0
        s0.particle_qd.assign(v0.astype(np.float32))
        solver.step(s0, s1, None, None, dt)
        masses = np.r_[0.0, np.full(n, mass)]
        springs = np.stack([np.arange(n), np.arange(1, n + 1)], axis=1)
        expected = _implicit_euler(
            x0, v0, masses, springs, np.full(n, seg), np.full(n, ke), dt, solver.pin_stiffness, masses == 0.0
        )
        np.testing.assert_allclose(s1.particle_q.numpy(), expected, atol=1.0e-6)

    def test_sagging_rope_settles_without_jitter(self):
        """Hang a rope of 80 stiff springs between two pins, every spring at rest length on an arc that sags 13 cm:
        at 10 iterations per step it swings out and settles, with its springs stretched by the weight they hold,
        as at 100 iterations; without the chains' tension solve its tension lags its swing and keeps it shaking at
        0.7 m/s with ten times the stretch."""
        n, length, span = 80, 1.0, 0.95
        half = 0.55191  # sin(half) / half = span / length
        radius = length / (2.0 * half)
        angle = np.linspace(-half, half, n + 1)
        sag = radius * (np.cos(angle) - np.cos(half))
        points = np.c_[0.5 * span + radius * np.sin(angle), 0.1 * sag, 1.0 - sag]
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        mass = 0.5 * 5.5e-2 * (np.r_[seg, 0.0] + np.r_[0.0, seg])
        mass[[0, -1]] = 0.0
        results = []
        for iterations in (10, 100):
            builder = newton.ModelBuilder()
            for p, m in zip(points, mass, strict=True):
                builder.add_particle(pos=tuple(p), vel=(0.0, 0.0, 0.0), mass=float(m), radius=0.0)
            for i, length_i in enumerate(seg):
                builder.add_spring(i, i + 1, ke=float(1.0e3 / length_i), kd=0.0, control=0.0)
            model = builder.finalize()
            solver = SolverFBA(model, iterations=iterations, rayleigh_alpha=2.0)
            s0, s1 = model.state(), model.state()
            speeds = []
            for _ in range(300):
                solver.step(s0, s1, None, None, 0.01)
                s0, s1 = s1, s0
                speeds.append(np.linalg.norm(s0.particle_qd.numpy(), axis=1).max())
            x = s0.particle_q.numpy().astype(np.float64)
            results.append((np.max(speeds[-50:]), np.linalg.norm(np.diff(x, axis=0), axis=1) / seg - 1.0))
        (speed, strain), (converged_speed, converged) = results
        self.assertLess(speed, 0.01)
        np.testing.assert_allclose(speed, converged_speed, rtol=0.1)
        np.testing.assert_allclose(strain, converged, atol=2.0e-5)

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

    @staticmethod
    def _dumbbell(stretch=0.0, spin=0.0, ke=100.0):
        """Two particles of 0.1 kg joined by a spring of rest length 0.2 m, without gravity: stretched by ``stretch``
        and spinning at ``spin`` [rad/s] about their center."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        half = 0.1 * (1.0 + stretch)
        for sign in (-1.0, 1.0):
            builder.add_particle(pos=(sign * half, 0.0, 0.0), vel=(0.0, sign * spin * half, 0.0), mass=0.1, radius=0.0)
        builder.add_spring(0, 1, ke=ke, kd=0.0, control=0.0)
        builder.spring_rest_length[0] = 0.2
        return builder.finalize()

    @staticmethod
    def _run(model, steps, **kwargs):
        solver = SolverFBA(model, iterations=50, **kwargs)
        s0, s1 = model.state(), model.state()
        history = []
        for _ in range(steps):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            history.append((s0.particle_q.numpy().astype(np.float64), s0.particle_qd.numpy().astype(np.float64)))
        return history

    def test_corotated_damping_spares_a_spinning_dumbbell(self):
        """Keep the spin of a dumbbell with ``rayleigh_corotated``, which damps the spring's length rate alone, where
        the isotropic stiffness damping brakes it."""
        model = self._dumbbell(spin=5.0)

        def spin(state):
            x, v = state
            return np.cross(x, v)[:, 2].sum() / np.sum(x[:, :2] ** 2)

        free = spin(self._run(model, 100)[-1])
        corotated = spin(self._run(model, 100, rayleigh_beta=0.05, rayleigh_corotated=True)[-1])
        isotropic = spin(self._run(model, 100, rayleigh_beta=0.05)[-1])
        np.testing.assert_allclose(corotated, free, rtol=1.0e-2)
        self.assertLess(isotropic, 0.2 * free)

    def test_damped_step_converges_to_implicit_euler(self):
        """Step a chain thrown with random velocities and damped along its springs: the step reaches the implicit
        Euler step with the springs' viscous force ``beta k (r - r_0) / dt`` along them."""
        n, seg, mass, ke, dt, beta = 6, 0.05, 0.01, 1.0e3, 0.01, 2.0e-3
        model = _chain(n=n, seg=seg, mass=mass, ke=ke).finalize()
        solver = SolverFBA(model, iterations=1600, rayleigh_beta=beta, rayleigh_corotated=True)
        s0, s1 = model.state(), model.state()
        x0 = s0.particle_q.numpy().astype(np.float64)
        v0 = np.random.default_rng(1).normal(scale=0.3, size=x0.shape)
        v0[0] = 0.0
        s0.particle_qd.assign(v0.astype(np.float32))
        solver.step(s0, s1, None, None, dt)
        masses = np.r_[0.0, np.full(n, mass)]
        springs = np.stack([np.arange(n), np.arange(1, n + 1)], axis=1)
        expected = _implicit_euler(
            x0,
            v0,
            masses,
            springs,
            np.full(n, seg),
            np.full(n, ke),
            dt,
            solver.pin_stiffness,
            masses == 0.0,
            damping=np.full(n, beta * ke / dt),
        )
        np.testing.assert_allclose(s1.particle_q.numpy(), expected, atol=1.0e-6)

    def test_spring_chains_follow_paths_and_bound_junctions(self):
        """Split the springs into chains along paths of particles with two springs: a rod, a Y junction cut into its
        three springs, a loop cut at one particle, two springs that meet head to head; a chain keeps the couplings
        of its inner particles, with the signs of the two springs' ends, and a particle whose couplings it drops
        weighs its inverse mass by its number of springs."""
        from newton._src.solvers.fba.linear_solver import spring_chains  # noqa: PLC0415

        indices = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [7, 5], [8, 9], [9, 10], [10, 8], [11, 12], [13, 12]]
        offsets, order, link, sign, scale = spring_chains(np.array(indices), 14)
        chains = [order[a:b].tolist() for a, b in itertools.pairwise(offsets)]
        self.assertEqual(chains, [[0, 1, 2], [3], [4], [5], [9, 10], [6, 7, 8]])
        np.testing.assert_array_equal(link, [1, 2, -1, -1, -1, -1, 12, -1, 9, 10, -1])
        np.testing.assert_array_equal(sign, [-1, -1, 0, 0, 0, 0, 1, 0, -1, -1, 0])
        np.testing.assert_array_equal(scale[[3, 4, 5]], [[1, 3], [3, 1], [1, 3]])
        np.testing.assert_array_equal(scale[[6, 8]], [[2, 1], [1, 2]])
        np.testing.assert_array_equal(scale[[0, 1, 2, 7, 9, 10]], 1.0)

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
