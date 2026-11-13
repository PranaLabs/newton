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

    def test_slack_tendon_carries_no_friction(self):
        """A 1 kg box without gravity, the first end of a 1 m string to a fixed point 0.3 m away, moves toward the
        point at 0.5 m/s: the slack string with a friction coefficient of 1 neither slows nor pushes it."""
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
                    solver.set_tendons([[box, -1]], [[(0.0, 0.0, 0.0), B]], [[A, A]], [0.0], [1.0], [1.0])
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
        solver.set_tendons(None, None, None, None, None)
        self.assertEqual(solver._tendon_count, 0)


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
