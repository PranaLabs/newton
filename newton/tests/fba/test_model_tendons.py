# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA on the model's tendons (:meth:`newton.ModelBuilder.add_tendon`), which it routes itself."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.tendon import TendonGuideType

H = 0.05  # half size of the boxes [m]
G = 9.81
ANCHOR, PINHOLE, ROLLER = int(TendonGuideType.ANCHOR), int(TendonGuideType.PINHOLE), int(TendonGuideType.ROLLER)


def _builder(gravity=-G):
    from newton.solvers import SolverFBA  # noqa: PLC0415

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    SolverFBA.register_custom_attributes(builder)
    return builder


def _box(builder, position, mass):
    body = builder.add_body(xform=wp.transform(tuple(position), wp.quat_identity()))
    cfg = newton.ModelBuilder.ShapeConfig(density=mass / (2 * H) ** 3)
    builder.add_shape_box(body, hx=H, hy=H, hz=H, cfg=cfg)
    return body


def _fixed(builder, position=(0.0, 0.0, 0.0)):
    return builder.add_body(xform=wp.transform(tuple(position), wp.quat_identity()), mass=0.0, is_kinematic=True)


def _top(body):
    return newton.TendonGuide(body=body, guide_type=ANCHOR, offset=(0.0, 0.0, H))


def _solver(model, sparse, **kwargs):
    from newton.solvers import SolverFBA  # noqa: PLC0415

    return SolverFBA(
        model, iterations=10, nsn_sparse_storage=sparse, nsn_linear_iterations=10 if sparse else None, **kwargs
    )


def _run(model, solver, steps, dt, velocities=None):
    """Step from rest, or from the linear ``velocities`` of some bodies; returns the last state and the poses."""
    s0, s1 = model.state(), model.state()
    if velocities:
        qd = s0.body_qd.numpy()
        for body, v in velocities.items():
            qd[body, :3] = v
        s0.body_qd.assign(qd)
    trace = []
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        trace.append(s0.body_q.numpy().copy())
    return s0, np.array(trace)


def _atwood(m1, m2, drop=0.4, pinholes=True, mu=0.0, force=0.0, radius=0.05, slack=0.0):
    """Boxes of ``m1`` and ``m2`` hanging ``drop`` below two pinholes 0.2 m apart, or on both sides of one roller of
    ``radius`` (``pinholes=False``), on one string from the first box's top to the second's, at rest; with
    ``slack``, the pinholes' string is that much longer than its path."""
    builder = _builder()
    if pinholes:
        A, B = np.array([-0.1, 0.0, 1.0]), np.array([0.1, 0.0, 1.0])
    else:
        A, B = np.array([-radius, 0.0, 1.0]), np.array([radius, 0.0, 1.0])
    fixed = _fixed(builder)
    b1 = _box(builder, A - (0.0, 0.0, drop + H), m1)
    b2 = _box(builder, B - (0.0, 0.0, drop + H), m2)
    attributes = {"fba:guide_friction_force": force}
    if pinholes:
        rest = (drop, 0.2, drop + slack) if slack else (-1.0, -1.0, -1.0)
        middle = [
            newton.TendonGuide(
                body=fixed,
                guide_type=PINHOLE,
                offset=tuple(A),
                mu=mu,
                rest_length=rest[0],
                custom_attributes=attributes,
            ),
            newton.TendonGuide(body=fixed, guide_type=PINHOLE, offset=tuple(B), mu=mu, rest_length=rest[1]),
        ]
    else:
        middle = [
            newton.TendonGuide(
                body=fixed,
                guide_type=ROLLER,
                radius=radius,
                orientation=1,
                offset=(0.0, 0.0, 1.0),
                axis=(0.0, 1.0, 0.0),
                mu=mu,
                custom_attributes=attributes,
            )
        ]
    last = _top(b2)
    if pinholes and slack:
        last = newton.TendonGuide(body=b2, guide_type=ANCHOR, offset=(0.0, 0.0, H), rest_length=rest[2])
    builder.add_tendon([_top(b1), *middle, last])
    return builder.finalize(), b1, b2


def _acceleration(trace, b1, b2, dt):
    t = (np.arange(len(trace)) + 1) * dt
    z1, z2 = trace[:, b1, 2], trace[:, b2, 2]
    return 2 * np.polyfit(t, 0.5 * (z1 - z2), 2)[0]


class TestModelTendons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def test_atwood_machine(self):
        """Boxes of 20 g and 30 g on one string over two pinholes or over a roller accelerate at
        (m2 - m1) g / (m1 + m2), the string's routed length kept at its rest length."""
        m1, m2, dt, steps = 0.02, 0.03, 0.002, 150
        for pinholes in (True, False):
            for sparse in (False, True):
                with self.subTest(pinholes=pinholes, sparse=sparse):
                    model, b1, b2 = _atwood(m1, m2, pinholes=pinholes)
                    solver = _solver(model, sparse)
                    state, trace = _run(model, solver, steps, dt)
                    theory = (m2 - m1) * G / (m1 + m2)
                    self.assertAlmostEqual(_acceleration(trace, b1, b2, dt) / theory, 1.0, delta=0.02)
                    rest = float(solver._tendons[4].numpy()[0])
                    self.assertAlmostEqual(_routed_length(solver, state), rest, delta=5e-4)

    def test_rest_length_is_newtons_cable(self):
        """A tendon of one group rests at Newton's total cable length: its spans and its rollers' arcs."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model, _b1, _b2 = _atwood(0.02, 0.03, pinholes=False)
        solver = SolverFBA(model, iterations=1)
        np.testing.assert_allclose(solver._tendons[4].numpy(), solver.tendon_total_cable.numpy(), rtol=1e-6)
        self.assertAlmostEqual(float(solver._tendons[4].numpy()[0]), 2 * 0.4 + np.pi * 0.05, delta=1e-5)

    def test_capstan_friction_holds_or_slows_the_slide(self):
        """The roller's capstan coefficient mu over its half turn bounds the first box's slide by min(1, mu pi)
        times the tension: mu = 0.2 holds the boxes (1 - m1 / m2 = 1/3 needs 0.106), and mu = 0.05 slows them to
        ((1 - mu pi) m2 - m1) g / (m1 + (1 - mu pi) m2)."""
        m1, m2, dt, steps = 0.02, 0.03, 0.002, 150
        for mu in (0.2, 0.05):
            for sparse in (False, True):
                with self.subTest(mu=mu, sparse=sparse):
                    model, b1, b2 = _atwood(m1, m2, pinholes=False, mu=mu)
                    _state, trace = _run(model, _solver(model, sparse), steps, dt)
                    if mu * np.pi >= 1.0 - m1 / m2:
                        self.assertLess(abs(trace[-1, b1, 2] - trace[0, b1, 2]), 5e-4)
                    else:
                        k = 1.0 - mu * np.pi
                        theory = (k * m2 - m1) * G / (m1 + k * m2)
                        self.assertAlmostEqual(_acceleration(trace, b1, b2, dt) / theory, 1.0, delta=0.03)

    def test_guide_friction_force(self):
        """A guide's fba:guide_friction_force F0 holds the boxes at 0.15 N, above the weight difference of 0.098 N,
        and slows them to ((m2 - m1) g - F0) / (m1 + m2) at 0.05 N."""
        m1, m2, dt, steps = 0.02, 0.03, 0.002, 150
        for force in (0.15, 0.05):
            for sparse in (False, True):
                with self.subTest(force=force, sparse=sparse):
                    model, b1, b2 = _atwood(m1, m2, force=force)
                    _state, trace = _run(model, _solver(model, sparse), steps, dt)
                    if force > (m2 - m1) * G:
                        self.assertLess(abs(trace[-1, b1, 2] - trace[0, b1, 2]), 5e-4)
                    else:
                        theory = ((m2 - m1) * G - force) / (m1 + m2)
                        self.assertAlmostEqual(_acceleration(trace, b1, b2, dt) / theory, 1.0, delta=0.03)

    def test_interior_anchor_splits_the_tendon(self):
        """A tendon from a fixed point to box A and on to box B, both hanging, holds each part at its own length:
        the anchor on box A splits it into two rows."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = _builder()
                top = _fixed(builder, (0.0, 0.0, 1.0))
                a = _box(builder, (0.0, 0.0, 0.6 - H), 0.05)
                b = _box(builder, (0.0, 0.0, 0.3 - H), 0.05)
                builder.add_tendon(
                    [
                        newton.TendonGuide(body=top, guide_type=ANCHOR),
                        newton.TendonGuide(body=a, guide_type=ANCHOR, offset=(0.0, 0.0, H)),
                        newton.TendonGuide(body=a, guide_type=ANCHOR, offset=(0.0, 0.0, -H)),
                        newton.TendonGuide(body=b, guide_type=ANCHOR, offset=(0.0, 0.0, H)),
                    ]
                )
                model = builder.finalize()
                solver = _solver(model, sparse)
                self.assertEqual(solver._tendon_count, 3)
                state, _trace = _run(model, solver, 100, 0.004)
                q = state.body_q.numpy()
                self.assertAlmostEqual(1.0 - (q[a, 2] + H), 0.4, delta=5e-4)
                self.assertAlmostEqual((q[a, 2] - H) - (q[b, 2] + H), 0.2, delta=5e-4)

    def _hang(self, sparse, steps, drop=0.0, dt=0.002, compliance=(0.0, 0.0), damping=(0.0, 0.0), time=0.0):
        """A 0.1 kg box on a 0.3 m string, of two spans through a pinhole 0.1 m below a fixed point, let go at rest
        with its top ``drop`` above the string's reach. Returns its top's depth below the point and its speed."""
        builder = _builder()
        top = _fixed(builder, (0.0, 0.0, 1.0))
        box = _box(builder, (0.0, 0.0, 1.0 - 0.3 + drop - H), 0.1)
        builder.add_tendon(
            [
                newton.TendonGuide(body=top, guide_type=ANCHOR),
                newton.TendonGuide(
                    body=top,
                    guide_type=PINHOLE,
                    offset=(0.0, 0.0, -0.1),
                    compliance=compliance[0],
                    damping=damping[0],
                    rest_length=0.1,
                ),
                newton.TendonGuide(
                    body=box,
                    guide_type=ANCHOR,
                    offset=(0.0, 0.0, H),
                    compliance=compliance[1],
                    damping=damping[1],
                    rest_length=0.2,
                ),
            ],
            custom_attributes={"fba:tendon_time": time},
        )
        model = builder.finalize()
        solver = _solver(model, sparse)
        s0, s1 = model.state(), model.state()
        depth, speed = [], []
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            depth.append(1.0 - (s0.body_q.numpy()[box, 2] + H))
            speed.append(float(s0.body_qd.numpy()[box, 2]))
        return np.array(depth), np.array(speed)

    def test_compliance_stretches_the_string(self):
        """The spans' compliances add up: a 0.1 kg box hanging at rest stretches the string by m g (C1 + C2); an
        inextensible string holds its length."""
        c1, c2 = 1e-4, 2e-4
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                depth, _speed = self._hang(sparse, 200, compliance=(c1, c2), damping=(0.5, 0.5))
                self.assertAlmostEqual((depth[-20:].mean() - 0.3) / (0.1 * G * (c1 + c2)), 1.0, delta=0.05)
                depth, _speed = self._hang(sparse, 60)
                self.assertLess(abs(depth[-20:].mean() - 0.3), 2e-6)

    def test_time_constant_is_relative_to_the_mass(self):
        """fba:tendon_time tau = 4 ms stretches the string of a hanging box by g tau² (0.157 mm), the deflection of a
        spring m / tau² under its weight, whatever the spans' compliance."""
        tau = 0.004
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                depth, _speed = self._hang(sparse, 150, compliance=(1.0, 1.0), time=tau)
                self.assertAlmostEqual((depth[-20:].mean() - 0.3) / (G * tau * tau), 1.0, delta=0.05)

    def test_soft_string_catches_over_several_steps(self):
        """A box falls 0.1 m on a slack string, freely until it is taut, and is caught at 1.4 m/s: the inextensible
        string stops it in one step, a critically damped one of compliance m⁻¹ (10 ms)² over several, without a
        rebound, and then holds it g (10 ms)² lower."""
        dt, m, tau = 0.002, 0.1, 0.01
        c = tau * tau / m  # compliance of a 10 ms spring for the box
        d = 2.0 * m / tau  # its critical damping
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                hard_depth, hard_speed = self._hang(sparse, 120, drop=0.1, dt=dt)
                soft_depth, soft_speed = self._hang(
                    sparse, 120, drop=0.1, dt=dt, compliance=(c / 3, 2 * c / 3), damping=(d / 3, 2 * d / 3)
                )
                falling = hard_depth < 0.299
                self.assertGreater(int(falling.sum()), 50)
                np.testing.assert_allclose(soft_depth[falling], hard_depth[falling], atol=1e-9)
                kick = lambda v: float(np.max(np.diff(v)))  # noqa: E731
                self.assertLess(kick(soft_speed), 0.5 * kick(hard_speed))
                self.assertLess(soft_speed.max(), 0.1 * np.sqrt(2 * G * 0.1))
                self.assertAlmostEqual(soft_depth[-1] - hard_depth[-1], G * tau * tau, delta=1e-4)

    def test_dynamic_roller_engages(self):
        """A box swings on a string from a fixed point past a dynamic roller beside its path: the roller engages once
        the string wraps it, and the routed length stays at the rest length throughout."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = _builder()
                top = _fixed(builder, (0.0, 0.0, 1.0))
                box = _box(builder, (0.35, 0.0, 1.0 - H), 0.1)
                builder.add_tendon(
                    [
                        newton.TendonGuide(body=top, guide_type=ANCHOR, axis=(0.0, 1.0, 0.0)),
                        newton.TendonGuide(
                            body=top,
                            guide_type=ROLLER,
                            radius=0.01,
                            orientation=1,
                            dynamic=True,
                            offset=(0.05, 0.0, -0.2),
                            axis=(0.0, 1.0, 0.0),
                        ),
                        newton.TendonGuide(body=box, guide_type=ANCHOR, offset=(0.0, 0.0, H), axis=(0.0, 1.0, 0.0)),
                    ]
                )
                model = builder.finalize()
                solver = _solver(model, sparse)
                rest = float(solver._tendons[4].numpy()[0])
                s0, s1 = model.state(), model.state()
                engaged, lengths = False, []
                for _ in range(200):
                    solver.step(s0, s1, None, None, 0.002)
                    s0, s1 = s1, s0
                    lengths.append(_routed_length(solver, s0))
                    engaged |= bool(solver.tendon_guide_active.numpy()[1])
                self.assertTrue(engaged)
                self.assertLess(max(lengths), rest + 2e-3)

    def test_pendulum_keeps_its_length(self):
        """A box hanging 0.5 m off a fixed point, let go 30 degrees out, swings on the taut string: its top stays at
        the string's length from the point, and the box swings down through the bottom."""
        anchor, length = np.array([0.0, 0.0, 1.0]), 0.5
        a = np.radians(30)
        top = anchor + length * np.array([np.sin(a), 0.0, -np.cos(a)])
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = _builder()
                fixed = _fixed(builder, tuple(anchor))
                box = _box(builder, top - (0.0, 0.0, H), 0.2)
                builder.add_tendon([newton.TendonGuide(body=fixed, guide_type=ANCHOR), _top(box)])
                model = builder.finalize()
                _state, trace = _run(model, _solver(model, sparse), 120, 0.004)
                tops = np.array([_point(q[box], (0.0, 0.0, H)) for q in trace])
                distances = np.linalg.norm(tops - anchor, axis=1)
                self.assertLess(distances.max(), length + 2e-4)
                self.assertGreater(distances.min(), length - 2e-3)
                self.assertLess(tops[:, 0].min(), 0.0)  # swung through the bottom to the other side

    def test_friction_force_never_pushes(self):
        """The Atwood machine sliding at 0.3 m/s (the 20 g box rising) against a guide friction force of 1 N, far
        above the tension: the friction stops the slide but never exceeds the tension, so the string never pushes
        the 20 g box down faster than it falls freely, and the boxes come to rest."""
        dt = 0.002
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, b1, b2 = _atwood(0.02, 0.03, force=1.0)
                solver = _solver(model, sparse)
                s0, s1 = model.state(), model.state()
                qd = s0.body_qd.numpy()
                qd[b1, 2], qd[b2, 2] = 0.3, -0.3
                s0.body_qd.assign(qd)
                v1 = []
                for _ in range(100):
                    solver.step(s0, s1, None, None, dt)
                    s0, s1 = s1, s0
                    v1.append(float(s0.body_qd.numpy()[b1, 2]))
                a1 = np.diff(np.concatenate([[0.3], v1])) / dt
                self.assertGreater(a1.min(), -G * 1.05)
                np.testing.assert_allclose(s0.body_qd.numpy()[[b1, b2], 2], 0.0, atol=2e-3)

    def test_friction_force_after_a_catch(self):
        """The Atwood machine with its string 2 cm longer than its path, released from rest, with a friction force of
        0.05 N: both boxes fall freely 1 cm until the string catches them at 0.443 m/s, which leaves the 20 g box
        rising at 0.089 m/s. The step of the catch is not held by the friction (the F0 part starts at the step after
        the string holds a tension), and the 20 g box then accelerates at ((m2 - m1) g - F0) / (m1 + m2)."""
        dt, f0, m1, m2 = 0.002, 0.05, 0.02, 0.03
        t_catch = np.sqrt(2 * 0.01 / G)
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, b1, _b2 = _atwood(m1, m2, force=f0, slack=0.02)
                solver = _solver(model, sparse)
                s0, s1 = model.state(), model.state()
                v1 = []
                for _ in range(150):
                    solver.step(s0, s1, None, None, dt)
                    s0, s1 = s1, s0
                    v1.append(float(s0.body_qd.numpy()[b1, 2]))
                v1 = np.array(v1)
                caught = int(np.argmax(v1 > 0.0))
                self.assertGreater(caught, 0)
                self.assertGreater(v1[caught], 0.02)  # the catch step rises (it sticks at 0 when held)
                theory = 0.443 * (m2 - m1) / (m1 + m2) + ((m2 - m1) * G - f0) / (m1 + m2) * (0.3 - t_catch)
                self.assertAlmostEqual(v1[-1] / theory, 1.0, delta=0.04)

    def test_slack_tendon_carries_no_friction(self):
        """A 1 kg box without gravity, the first anchor of a 1 m string through a pinhole 0.3 m away, moves toward the
        pinhole at 0.5 m/s: the slack string with the pinhole's capstan mu 1 and a friction force of 1 N neither slows
        nor pushes it."""
        A = np.array([0.0, 0.0, 1.0])
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = _builder(gravity=0.0)
                fixed = _fixed(builder)
                box = _box(builder, A + np.array([0.3, 0.0, 0.0]), 1.0)
                builder.add_tendon(
                    [
                        newton.TendonGuide(body=box, guide_type=ANCHOR),
                        newton.TendonGuide(
                            body=fixed,
                            guide_type=PINHOLE,
                            offset=tuple(A),
                            mu=1.0,
                            rest_length=0.7,
                            custom_attributes={"fba:guide_friction_force": 1.0},
                        ),
                        newton.TendonGuide(
                            body=fixed, guide_type=ANCHOR, offset=tuple(A + np.array([0.0, 0.0, 0.1])), rest_length=0.3
                        ),
                    ]
                )
                model = builder.finalize()
                state, _trace = _run(model, _solver(model, sparse), 100, 0.004, {box: (-0.5, 0.0, 0.0)})
                np.testing.assert_allclose(state.body_qd.numpy()[box, :3], (-0.5, 0.0, 0.0), atol=1e-6)
                self.assertAlmostEqual(float(state.body_q.numpy()[box, 0]), 0.3 - 0.5 * 0.4, delta=1e-5)

    def test_slack_string_pulls_nothing(self):
        """A box 0.1 m below a fixed point on a 0.3 m string falls freely until the string goes taut, then hangs."""
        anchor, dt = np.array([0.0, 0.0, 1.0]), 0.002
        builder = _builder()
        fixed = _fixed(builder, tuple(anchor))
        box = _box(builder, anchor - (0.0, 0.0, 0.1 + H), 0.1)
        builder.add_tendon(
            [
                newton.TendonGuide(body=fixed, guide_type=ANCHOR),
                newton.TendonGuide(body=box, guide_type=ANCHOR, offset=(0.0, 0.0, H), rest_length=0.3),
            ]
        )
        model = builder.finalize()
        _state, trace = _run(model, _solver(model, True), 200, dt)
        top = trace[:, box, 2] + H
        free = 0.9 - 0.5 * G * ((np.arange(200) + 1) * dt) ** 2
        falling = top > anchor[2] - 0.29
        np.testing.assert_allclose(top[falling], free[falling], atol=2e-3)  # free fall while slack
        self.assertAlmostEqual(top[-1], anchor[2] - 0.3, delta=2e-3)  # caught at the string's length

    def test_end_held_at_its_target(self):
        """A box whose top is tied by a string of zero length to a fixed point, pushed sideways: the end sits on its
        target, where the run's direction flips as the end passes it. With many iterations per step the box swings
        about the point and stays within about one step's travel (1.2 mm) of it, without the iterations running
        away."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        anchor = np.array([0.0, 0.0, 1.0])
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                builder = _builder()
                fixed = _fixed(builder, tuple(anchor))
                box = _box(builder, anchor - (0.0, 0.0, H + 1e-4), 0.1)
                builder.add_tendon(
                    [
                        newton.TendonGuide(body=fixed, guide_type=ANCHOR),
                        newton.TendonGuide(body=box, guide_type=ANCHOR, offset=(0.0, 0.0, H), rest_length=0.0),
                    ]
                )
                model = builder.finalize()
                solver = SolverFBA(
                    model, iterations=40, nsn_sparse_storage=sparse, nsn_linear_iterations=10 if sparse else None
                )
                s0, s1 = model.state(), model.state()
                qd = s0.body_qd.numpy()
                qd[box, 0] = 0.3
                s0.body_qd.assign(qd)
                tops, speeds = [], []
                for _ in range(150):
                    solver.step(s0, s1, None, None, 0.004)
                    s0, s1 = s1, s0
                    q = s0.body_q.numpy()[box]
                    tops.append(np.linalg.norm(_point(q, (0.0, 0.0, H)) - anchor))
                    speeds.append(np.linalg.norm(s0.body_qd.numpy()[box, :3]))
                self.assertLess(max(tops), 2e-3)
                self.assertLess(max(speeds), 1.0)

    def test_model_tendons_are_validated(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        # a guide between the anchors on a body that SolverFBA integrates
        builder = _builder()
        top = _fixed(builder, (0.0, 0.0, 1.0))
        a = _box(builder, (0.0, 0.0, 0.6), 0.05)
        b = _box(builder, (0.0, 0.0, 0.3), 0.05)
        builder.add_tendon(
            [
                newton.TendonGuide(body=top, guide_type=ANCHOR),
                newton.TendonGuide(body=a, guide_type=PINHOLE),
                newton.TendonGuide(body=b, guide_type=ANCHOR),
            ]
        )
        with self.assertRaises(NotImplementedError):
            SolverFBA(builder.finalize(), iterations=1)
        model, _b1, _b2 = _atwood(0.02, 0.03)
        self.assertEqual(SolverFBA(model, iterations=1)._tendon_count, 1)
        with self.assertRaises(NotImplementedError):
            SolverFBA(model, iterations=1, nsn_schur_mode="lite")


def _point(q, local):
    """World position of the body-frame point ``local`` of a body at pose ``q``."""
    x, y, z, w = q[3:]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return q[:3] + R @ np.asarray(local)


def _routed_length(solver, state):
    """The length of tendon group 0 at ``state``, as its row measures it at a step's start: the solver routes it
    there, and the length is the middle length plus each end's run to its target."""
    solver._route_model_tendons(state)
    body = solver._tendons[0].numpy()[0]
    point = solver._tendons[1].numpy()[0]
    target = solver._tendons[2].numpy()[0]
    middle = float(solver._tendons[3].numpy()[0])
    body_q = state.body_q.numpy()
    length = middle
    for side in range(2):
        q = body_q[body[side]]
        x, y, z, w = q[3:]
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        length += np.linalg.norm(target[side] - (q[:3] + R @ point[side]))
    return length


if __name__ == "__main__":
    unittest.main(verbosity=2)
