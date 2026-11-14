# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA tendons: inextensible strings between bodies (``set_tendons``)."""

import unittest

import numpy as np
import warp as wp

import newton

H = 0.05  # half size of the boxes [m]


def _boxes(positions, masses, gravity=-9.81):
    """Free boxes of side 2H at `positions` with `masses`, no ground."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    bodies = []
    for p, m in zip(positions, masses, strict=True):
        body = builder.add_body(xform=wp.transform(tuple(p), wp.quat_identity()))
        builder.add_shape_box(body, hx=H, hy=H, hz=H, cfg=newton.ModelBuilder.ShapeConfig(density=m / (2 * H) ** 3))
        bodies.append(body)
    return builder.finalize(), bodies


class TestTendons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _solver(self, model, sparse):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        return SolverFBA(model, iterations=10, nsn_sparse_storage=sparse, nsn_linear_iterations=10 if sparse else None)

    def _run(self, model, solver, steps, dt, tendons):
        """Step with the tendons that `tendons(state)` gives before each step; returns the states' trace."""
        s0, s1 = model.state(), model.state()
        trace = []
        for _ in range(steps):
            solver.set_tendons(*tendons(s0))
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            trace.append(s0.body_q.numpy().copy())
        return s0, np.array(trace)

    def test_pendulum_keeps_its_length(self):
        """A box hanging 0.5 m off a fixed point, let go 30 degrees out, swings on the taut string: its top stays at
        the string's length from the point, and the box swings down through the bottom."""
        anchor, length = np.array([0.0, 0.0, 1.0]), 0.5
        a = np.radians(30)
        top = anchor + length * np.array([np.sin(a), 0.0, -np.cos(a)])
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, (box,) = _boxes([top - (0.0, 0.0, H)], [0.2])
                solver = self._solver(model, sparse)

                def tendons(state, box=box):
                    return [[-1, box]], [[anchor, (0.0, 0.0, H)]], [[anchor, anchor]], [0.0], [length]

                _state, trace = self._run(model, solver, 120, 0.004, tendons)
                distances = [np.linalg.norm(_point_in(q, box, (0.0, 0.0, H)) - anchor) for q in trace]
                self.assertLess(max(distances), length + 2e-4)
                self.assertGreater(min(distances), length - 2e-3)
                xs = [_point_in(q, box, (0.0, 0.0, H))[0] for q in trace]
                self.assertLess(min(xs), 0.0)  # swung through the bottom to the other side

    def test_atwood_machine(self):
        """Two boxes of 20 g and 30 g on one string over two fixed points 0.2 m apart: they accelerate at
        (m2 - m1) g / (m1 + m2), the string's length kept."""
        m1, m2 = 0.02, 0.03
        A, B = np.array([-0.1, 0.0, 1.0]), np.array([0.1, 0.0, 1.0])
        drop = 0.4
        rest = 2 * drop + np.linalg.norm(B - A)
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, (b1, b2) = _boxes([A - (0, 0, drop + H), B - (0, 0, drop + H)], [m1, m2])
                solver = self._solver(model, sparse)

                def tendons(state, b1=b1, b2=b2):
                    return [[b1, b2]], [[(0, 0, H), (0, 0, H)]], [[A, B]], [np.linalg.norm(B - A)], [rest]

                dt, steps = 0.002, 150
                _state, trace = self._run(model, solver, steps, dt, tendons)
                t = (np.arange(steps) + 1) * dt
                z1, z2 = trace[:, b1, 2], trace[:, b2, 2]
                a = 2 * np.polyfit(t, 0.5 * (z1 - z2), 2)[0]
                theory = (m2 - m1) * 9.81 / (m1 + m2)
                self.assertAlmostEqual(a / theory, 1.0, delta=0.02)
                length = (
                    np.linalg.norm(_point_in(trace[-1], b1, (0, 0, H)) - A)
                    + np.linalg.norm(B - A)
                    + np.linalg.norm(_point_in(trace[-1], b2, (0, 0, H)) - B)
                )
                self.assertAlmostEqual(length, rest, delta=5e-4)

    def test_friction_holds_or_slows_the_slide(self):
        """The Atwood machine with a friction coefficient mu on the first end's slide, the 20 g box: its tension is
        (1 - mu) T, so the boxes accelerate at ((1 - mu) m2 - m1) g / (m1 + (1 - mu) m2) and hold from
        mu = 1 - m1 / m2 = 1/3 up."""
        m1, m2 = 0.02, 0.03
        A, B = np.array([-0.1, 0.0, 1.0]), np.array([0.1, 0.0, 1.0])
        drop = 0.4
        rest = 2 * drop + np.linalg.norm(B - A)
        for sparse in (False, True):
            for mu in (0.5, 0.2):
                with self.subTest(sparse=sparse, mu=mu):
                    model, (b1, b2) = _boxes([A - (0, 0, drop + H), B - (0, 0, drop + H)], [m1, m2])
                    solver = self._solver(model, sparse)

                    def tendons(state, b1=b1, b2=b2, mu=mu):
                        return (
                            [[b1, b2]],
                            [[(0, 0, H), (0, 0, H)]],
                            [[A, B]],
                            [np.linalg.norm(B - A)],
                            [rest],
                            [mu],
                        )

                    dt, steps = 0.002, 150
                    _state, trace = self._run(model, solver, steps, dt, tendons)
                    t = (np.arange(steps) + 1) * dt
                    z1, z2 = trace[:, b1, 2], trace[:, b2, 2]
                    a = 2 * np.polyfit(t, 0.5 * (z1 - z2), 2)[0]
                    theory = max((1 - mu) * m2 - m1, 0.0) * 9.81 / (m1 + (1 - mu) * m2)
                    if theory == 0.0:
                        self.assertLess(abs(z1[-1] - z1[0]), 5e-4)
                    else:
                        self.assertAlmostEqual(a / theory, 1.0, delta=0.03)

    def _atwood(self, sparse, mu, f0, steps=150, dt=0.002, v0=0.0, slack=0.0):
        """The 20 g / 30 g Atwood machine over two fixed points 0.2 m apart with friction on the 20 g box's slide,
        started at speed v0 (the 20 g box rising), its string ``slack`` longer than the path; returns the trace (z1,
        z2, vz1, vz2 per step) and the masses."""
        m1, m2 = 0.02, 0.03
        A, B = np.array([-0.1, 0.0, 1.0]), np.array([0.1, 0.0, 1.0])
        drop = 0.4
        rest = 2 * drop + np.linalg.norm(B - A) + slack
        model, (b1, b2) = _boxes([A - (0, 0, drop + H), B - (0, 0, drop + H)], [m1, m2])
        solver = self._solver(model, sparse)
        s0, s1 = model.state(), model.state()
        qd = s0.body_qd.numpy()
        qd[b1, 2], qd[b2, 2] = v0, -v0
        s0.body_qd.assign(qd)
        trace = []
        for _ in range(steps):
            solver.set_tendons(
                [[b1, b2]], [[(0, 0, H), (0, 0, H)]], [[A, B]], [np.linalg.norm(B - A)], [rest], [mu], [f0]
            )
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            trace.append(np.concatenate([s0.body_q.numpy()[[b1, b2], 2], s0.body_qd.numpy()[[b1, b2], 2]]))
        return np.array(trace), m1, m2

    def test_friction_force_holds_or_slows_the_slide(self):
        """A friction force F0 on the 20 g box's slide, with a coefficient mu: the boxes hold while
        ((1 - mu) m2 - m1) g <= F0 (0.15 N) and else accelerate at (((1 - mu) m2 - m1) g - F0) / (m1 + (1 - mu) m2),
        as a constant friction (0.05 N) and with both parts (mu 0.1, 0.03 N)."""
        dt = 0.002
        for sparse in (False, True):
            for mu, f0 in ((0.0, 0.15), (0.0, 0.05), (0.1, 0.03)):
                with self.subTest(sparse=sparse, mu=mu, f0=f0):
                    trace, m1, m2 = self._atwood(sparse, mu, f0, dt=dt)
                    t = (np.arange(len(trace)) + 1) * dt
                    z1, z2 = trace[:, 0], trace[:, 1]
                    a = 2 * np.polyfit(t, 0.5 * (z1 - z2), 2)[0]
                    theory = max(((1 - mu) * m2 - m1) * 9.81 - f0, 0.0) / (m1 + (1 - mu) * m2)
                    if theory == 0.0:
                        self.assertLess(abs(z1[-1] - z1[0]), 5e-4)
                    else:
                        self.assertAlmostEqual(a / theory, 1.0, delta=0.03)

    def test_friction_force_never_pushes(self):
        """The Atwood machine sliding at 0.3 m/s (the 20 g box rising) against a friction force of 1 N, far above the
        tension: the friction stops the slide but never exceeds the tension, so the string never pushes the 20 g box
        down faster than it falls freely, and the boxes come to rest."""
        dt = 0.002
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                trace, _m1, _m2 = self._atwood(sparse, 0.0, 1.0, steps=100, dt=dt, v0=0.3)
                a1 = np.diff(np.concatenate([[0.3], trace[:, 2]])) / dt
                self.assertGreater(a1.min(), -9.81 * 1.05)
                np.testing.assert_allclose(trace[-20:, 2:], 0.0, atol=2e-3)

    def test_friction_force_after_a_catch(self):
        """The Atwood machine with its string 2 cm longer than its path, released from rest, with a friction force of
        0.05 N: both boxes fall freely 1 cm until the string catches them at 0.443 m/s, which leaves the 20 g box
        rising at 0.089 m/s. The step of the catch is not held by the friction (the F0 part starts at the step after
        the string holds a tension), and the 20 g box then accelerates at ((m2 - m1) g - F0) / (m1 + m2)."""
        dt, f0 = 0.002, 0.05
        t_catch = np.sqrt(2 * 0.01 / 9.81)
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                trace, m1, m2 = self._atwood(sparse, 0.0, f0, steps=150, dt=dt, slack=0.02)
                v1 = trace[:, 2]
                caught = int(np.argmax(v1 > 0.0))
                self.assertGreater(caught, 0)
                self.assertGreater(v1[caught], 0.02)  # the catch step rises (it sticks at 0 when held)
                theory = 0.443 * (m2 - m1) / (m1 + m2) + ((m2 - m1) * 9.81 - f0) / (m1 + m2) * (0.3 - t_catch)
                self.assertAlmostEqual(v1[-1] / theory, 1.0, delta=0.04)

    def test_slack_tendon_carries_no_friction(self):
        """A 1 kg box without gravity, the first end of a 1 m string to a fixed point 0.3 m away, moves toward the
        point at 0.5 m/s: the slack string with a friction coefficient of 1 and a friction force of 1 N neither slows
        nor pushes it."""
        A = np.array([0.0, 0.0, 1.0])
        B = A + np.array([0.0, 0.0, 0.1])
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, (box,) = _boxes([A + np.array([0.3, 0.0, 0.0])], [1.0], gravity=0.0)
                solver = self._solver(model, sparse)
                state = model.state()
                qd = state.body_qd.numpy()
                qd[box, 0] = -0.5
                state.body_qd.assign(qd)
                s0, s1 = state, model.state()
                for _ in range(100):
                    solver.set_tendons([[box, -1]], [[(0.0, 0.0, 0.0), B]], [[A, A]], [0.0], [1.0], [1.0], [1.0])
                    solver.step(s0, s1, None, None, 0.004)
                    s0, s1 = s1, s0
                np.testing.assert_allclose(s0.body_qd.numpy()[box, :3], (-0.5, 0.0, 0.0), atol=1e-6)
                self.assertAlmostEqual(float(s0.body_q.numpy()[box, 0]), 0.3 - 0.5 * 0.4, delta=1e-5)

    def test_slack_string_pulls_nothing(self):
        """A box 0.1 m below a fixed point on a 0.3 m string falls freely until the string goes taut, then hangs."""
        anchor = np.array([0.0, 0.0, 1.0])
        model, (box,) = _boxes([anchor - (0.0, 0.0, 0.1 + H)], [0.1])
        solver = self._solver(model, True)

        def tendons(state):
            return [[-1, box]], [[anchor, (0.0, 0.0, H)]], [[anchor, anchor]], [0.0], [0.3]

        dt = 0.002
        _state, trace = self._run(model, solver, 200, dt, tendons)
        top = trace[:, box, 2] + H
        free = 0.9 - 0.5 * 9.81 * ((np.arange(200) + 1) * dt) ** 2
        falling = top > anchor[2] - 0.29
        np.testing.assert_allclose(top[falling], free[falling], atol=2e-3)  # free fall while slack
        self.assertAlmostEqual(top[-1], anchor[2] - 0.3, delta=2e-3)  # caught at the string's length

    def test_end_held_at_its_target(self):
        """A box whose top is tied by a string of zero length to a fixed point, pushed sideways: the end sits on its
        target, where the run's direction flips as the end passes it. With many iterations per step the box swings
        about the point and stays within about one step's travel (1.2 mm) of it, without the iterations running
        away."""
        anchor = np.array([0.0, 0.0, 1.0])
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, (box,) = _boxes([anchor - (0.0, 0.0, H + 1e-4)], [0.1])
                from newton.solvers import SolverFBA  # noqa: PLC0415

                solver = SolverFBA(
                    model, iterations=40, nsn_sparse_storage=sparse, nsn_linear_iterations=10 if sparse else None
                )
                state = model.state()
                qd = state.body_qd.numpy()
                qd[box, 0] = 0.3
                state.body_qd.assign(qd)
                s0, s1 = state, model.state()
                tops, speeds = [], []
                for _ in range(150):
                    solver.set_tendons([[-1, box]], [[anchor, (0.0, 0.0, H)]], [[anchor, anchor]], [0.0], [0.0])
                    solver.step(s0, s1, None, None, 0.004)
                    s0, s1 = s1, s0
                    tops.append(np.linalg.norm(_point_in(s0.body_q.numpy(), box, (0.0, 0.0, H)) - anchor))
                    speeds.append(np.linalg.norm(s0.body_qd.numpy()[box, :3]))
                self.assertLess(max(tops), 2e-3)
                self.assertLess(max(speeds), 1.0)

    def _hang(self, sparse, tau, steps, drop=0.0, dt=0.002):
        """A 0.1 kg box on a 0.3 m string from a fixed point, let go at rest with its top ``drop`` above the string's
        reach (0.3 m - drop below the point). Returns its top's depth below the point and its vertical speed per
        step."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        anchor, length = np.array([0.0, 0.0, 1.0]), 0.3
        model, (box,) = _boxes([anchor - (0.0, 0.0, length - drop + H)], [0.1])
        solver = SolverFBA(
            model,
            iterations=10,
            nsn_sparse_storage=sparse,
            nsn_linear_iterations=10 if sparse else None,
            tendon_time=tau,
        )
        s0, s1 = model.state(), model.state()
        depth, speed = [], []
        for _ in range(steps):
            solver.set_tendons([[-1, box]], [[anchor, (0.0, 0.0, H)]], [[anchor, anchor]], [0.0], [length])
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            depth.append(anchor[2] - (_point_in(s0.body_q.numpy(), box, (0.0, 0.0, H))[2]))
            speed.append(float(s0.body_qd.numpy()[box, 2]))
        return np.array(depth), np.array(speed)

    def test_soft_limit_stretches_by_its_time_constant(self):
        """With tendon_time tau = 4 ms a box hanging at rest stretches its string by g tau² (0.157 mm), the
        deflection of a spring of stiffness m / tau² under its weight; an inextensible string holds its length."""
        tau = 0.004
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                depth, _speed = self._hang(sparse, tau, 150)
                self.assertAlmostEqual((depth[-20:].mean() - 0.3) / (9.81 * tau * tau), 1.0, delta=0.05)
                depth, _speed = self._hang(sparse, 0.0, 60)
                self.assertLess(abs(depth[-20:].mean() - 0.3), 2e-6)

    def test_soft_limit_catches_over_several_steps(self):
        """A box falls 0.1 m on a slack string, freely until the string is taut (the soft limit pulls nothing while
        slack), and is caught at 1.4 m/s: the inextensible string stops it in one step, the soft one (10 ms,
        critically damped) over several, without a rebound, and then holds it g tau² (1 mm) lower."""
        dt = 0.002
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                hard_depth, hard_speed = self._hang(sparse, 0.0, 120, drop=0.1, dt=dt)
                tau = 0.01
                soft_depth, soft_speed = self._hang(sparse, tau, 120, drop=0.1, dt=dt)
                falling = hard_depth < 0.299  # the string still slack
                self.assertGreater(int(falling.sum()), 50)
                np.testing.assert_allclose(soft_depth[falling], hard_depth[falling], atol=1e-9)
                kick = lambda v: float(np.max(np.diff(v)))  # noqa: E731
                self.assertLess(kick(soft_speed), 0.5 * kick(hard_speed))
                self.assertLess(soft_speed.max(), 0.1 * np.sqrt(2 * 9.81 * 0.1))
                self.assertAlmostEqual(soft_depth[-1] - hard_depth[-1], 9.81 * tau * tau, delta=1e-4)

    def test_tendons_are_validated(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model, (box,) = _boxes([(0, 0, 0.5)], [0.1])
        solver = SolverFBA(model, iterations=1)
        with self.assertRaises(ValueError):
            solver.set_tendons([[-1, 3]], [[(0, 0, 0), (0, 0, 0)]], [[(0, 0, 1), (0, 0, 1)]], [0.0], [0.5])
        with self.assertRaises(ValueError):
            solver.set_tendons([[-1, box]], [[(0, 0, 0)]], [[(0, 0, 1), (0, 0, 1)]], [0.0], [0.5])
        for mu in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                solver.set_tendons([[-1, box]], [[(0, 0, 0), (0, 0, 0)]], [[(0, 0, 1), (0, 0, 1)]], [0.0], [0.5], [mu])
        with self.assertRaises(ValueError):
            solver.set_tendons(
                [[-1, box]], [[(0, 0, 0), (0, 0, 0)]], [[(0, 0, 1), (0, 0, 1)]], [0.0], [0.5], None, [-1.0]
            )
        solver.set_tendons(None, None, None, None, None)
        self.assertEqual(solver._tendon_count, 0)
        for kwargs in ({"tendon_time": -0.001}, {"tendon_damping": -1.0}):
            with self.assertRaises(ValueError):
                SolverFBA(model, iterations=1, **kwargs)


def _point_in(body_q, body, local):
    q = body_q[body]
    x, y, z, w = q[3:]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return q[:3] + R @ np.asarray(local)


if __name__ == "__main__":
    unittest.main(verbosity=2)
