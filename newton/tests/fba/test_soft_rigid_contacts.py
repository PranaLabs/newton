# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA's soft rigid contacts (``rigid_contact_time``, ``rigid_contact_damping``)."""

import unittest

import numpy as np
import warp as wp

import newton

R, M = 0.0135, 0.010  # a 10 g ball of 13.5 mm radius
DT = 1.0 / 480.0


def _ball_over_a_box(height, gap=1e-4):
    """The ball with its bottom ``height`` above a static box, which it falls onto; the shapes report contacts within
    ``gap`` [m] of each other."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.4, density=0.0, margin=0.0, gap=gap)
    builder.add_shape_box(
        -1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.05, cfg=cfg
    )
    body = builder.add_body(xform=wp.transform((0.0, 0.0, R + height), wp.quat_identity()))
    ball = newton.ModelBuilder.ShapeConfig(mu=0.4, density=M / (4.0 / 3.0 * np.pi * R**3), margin=0.0, gap=gap)
    builder.add_shape_sphere(body, radius=R, cfg=ball)
    return builder.finalize(), body


def _drop(height, steps, sparse, gap=1e-4, **soft):
    """The ball's bottom height and vertical speed after each step."""
    from newton.solvers import SolverFBA  # noqa: PLC0415

    model, body = _ball_over_a_box(height, gap)
    solver = SolverFBA(
        model,
        iterations=20,
        nsn_sparse_storage=sparse,
        nsn_linear_iterations=10 if sparse else None,
        **soft,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    z, v = [], []
    for _ in range(steps):
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, DT)
        s0, s1 = s1, s0
        z.append(float(s0.body_q.numpy()[body, 2]) - R)
        v.append(float(s0.body_qd.numpy()[body, 2]))
    return np.array(z), np.array(v)


class TestSoftRigidContacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def test_rest_sinks_by_the_static_deflection(self):
        """The ball at rest on the box sinks g tau² (0.157 mm for tau = 4 ms), the deflection of a spring of
        stiffness m / tau² under its weight; hard contacts keep it on the surface."""
        tau = 0.004
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                z, _v = _drop(0.0, 60, sparse, rigid_contact_time=tau)
                self.assertAlmostEqual(-z[-20:].mean() / (9.81 * tau * tau), 1.0, delta=0.05)
                z, _v = _drop(0.0, 60, sparse)
                self.assertLess(abs(z[-20:].mean()), 2e-6)

    def test_impact_is_damped_and_spread(self):
        """The ball dropped from 30 mm hits the box at 0.77 m/s. With tau = 4 ms at critical damping it sinks a
        fraction of a millimetre, rebounds by less than a tenth of its drop's speed, and the impact takes several
        steps (its peak force is below the hard contact's one-step impulse); without damping it bounces back."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                hard_z, hard_v = _drop(0.03, 100, sparse)
                soft_z, soft_v = _drop(0.03, 100, sparse, rigid_contact_time=0.004, rigid_contact_damping=1.0)
                _elastic_z, elastic_v = _drop(0.03, 100, sparse, rigid_contact_time=0.004, rigid_contact_damping=0.0)
                impact = 0.767
                self.assertGreater(-soft_z.min(), 2e-4)
                self.assertLess(soft_v.max(), 0.1 * impact)
                self.assertGreater(elastic_v.max(), 0.3 * impact)
                peak = lambda v: float(np.max(np.diff(v)) / DT)  # noqa: E731
                self.assertLess(peak(soft_v), 0.7 * peak(hard_v))
                self.assertLess(abs(hard_z[-20:].mean()), 2e-6)

    def test_no_force_before_the_bodies_touch(self):
        """With a detection gap of 10 mm the box's contact exists while the ball still falls toward it; the damper acts
        in penetration only, so the ball falls freely until it touches and the impact is the one of a 0.1 mm gap."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                soft = {"rigid_contact_time": 0.004, "rigid_contact_damping": 1.0}
                near_z, near_v = _drop(0.03, 60, sparse, **soft)
                far_z, far_v = _drop(0.03, 60, sparse, gap=0.01, **soft)
                free = -9.81 * DT * (np.arange(len(far_v)) + 1)
                above = far_z > 1e-4
                np.testing.assert_allclose(far_v[above], free[above], atol=1e-3)  # no braking above the box
                self.assertAlmostEqual(float(far_z.min()), float(near_z.min()), delta=2e-5)
                self.assertAlmostEqual(float(far_v.max()), float(near_v.max()), delta=1e-3)

    def test_options_are_validated(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model, _body = _ball_over_a_box(0.0)
        self.assertEqual(SolverFBA(model, iterations=1).rigid_contact_time, 0.0)
        with self.assertRaises(ValueError):
            SolverFBA(model, iterations=1, rigid_contact_time=-0.001)
        with self.assertRaises(ValueError):
            SolverFBA(model, iterations=1, rigid_contact_damping=-1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
