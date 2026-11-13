# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the friction coefficient of SolverFBA's rigid contacts (``rigid_contact_friction``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.rigid_contact import gather_rigid_contacts

BOX_MU, GROUND_MU = 0.7, 0.1


def _scene(device, gravity=(0.0, 0.0, -9.81)):
    """A 1 m box resting on the ground, the box's friction 0.7 and the ground's 0.1."""
    cfg = newton.ModelBuilder.ShapeConfig
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=cfg(mu=GROUND_MU))
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(box, cfg=cfg(mu=BOX_MU))
    model = builder.finalize(device=device)
    model.gravity.assign([wp.vec3(*gravity)])
    return model


class TestGatherFriction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def _mu(self, **kwargs):
        model = _scene(self.device)
        state = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        slot = np.full(model.body_count, -1, dtype=np.int32)
        slot[0] = 0
        rc = gather_rigid_contacts(model, contacts, slot, self.device, **kwargs)
        self.assertGreaterEqual(rc.count, 4)
        return rc.mu.numpy()

    def test_mean_by_default(self):
        np.testing.assert_allclose(self._mu(), 0.5 * (BOX_MU + GROUND_MU), rtol=1e-6)

    def test_larger_with_friction_max(self):
        np.testing.assert_allclose(self._mu(friction_max=True), BOX_MU, rtol=1e-6)


class TestSolverFriction(unittest.TestCase):
    """Gravity tilted by atan(0.55) makes the ground an incline that the mean friction (0.4) cannot hold and the
    larger (0.7) holds."""

    H, STEPS = 0.01, 30

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _slide(self, **solver_args):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        theta = np.arctan(0.55)
        model = _scene("cuda:0", gravity=(9.81 * np.sin(theta), 0.0, -9.81 * np.cos(theta)))
        solver = SolverFBA(model, iterations=10, **solver_args)
        s0, s1 = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        for _ in range(self.STEPS):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, self.H)
            s0, s1 = s1, s0
        return float(s0.body_q.numpy()[0, 0])

    def test_mean_slides_and_max_holds(self):
        g, theta = 9.81, np.arctan(0.55)
        t = self.H * self.STEPS
        expected = 0.5 * g * (np.sin(theta) - 0.4 * np.cos(theta)) * t * t
        self.assertGreater(self._slide(), 0.5 * expected)
        self.assertLess(abs(self._slide(rigid_contact_friction="max")), 1e-3)

    def test_invalid_rule_refused(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        with self.assertRaises(ValueError):
            SolverFBA(_scene("cuda:0"), rigid_contact_friction="min")


if __name__ == "__main__":
    unittest.main(verbosity=2)
