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


def _run(model, solver, steps, dt):
    s0, s1 = model.state(), model.state()
    trace = []
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        trace.append(s0.body_q.numpy().copy())
    return s0, np.array(trace)


def _atwood(m1, m2, drop=0.4, pinholes=True, mu=0.0, force=0.0, radius=0.05):
    """Boxes of ``m1`` and ``m2`` hanging ``drop`` below two pinholes 0.2 m apart, or on both sides of one roller of
    ``radius`` (``pinholes=False``), on one string from the first box's top to the second's, at rest."""
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
        middle = [
            newton.TendonGuide(body=fixed, guide_type=PINHOLE, offset=tuple(A), mu=mu, custom_attributes=attributes),
            newton.TendonGuide(body=fixed, guide_type=PINHOLE, offset=tuple(B), mu=mu),
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
    builder.add_tendon([_top(b1), *middle, _top(b2)])
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
        model, b1, b2 = _atwood(0.02, 0.03)
        solver = SolverFBA(model, iterations=1)
        with self.assertRaises(ValueError):  # the model's tendons and set_tendons do not combine
            solver.set_tendons([[b1, b2]], [[(0, 0, H), (0, 0, H)]], [[(0, 0, 1), (0, 0, 1)]], [0.0], [1.0])
        with self.assertRaises(ValueError):
            solver.set_tendon_paths(wp.zeros((1, 2), dtype=wp.vec3), wp.zeros(1, dtype=wp.float64))
        with self.assertRaises(NotImplementedError):
            SolverFBA(model, iterations=1, nsn_schur_mode="lite")


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
