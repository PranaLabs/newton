# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA's rigid iteration replayed from a CUDA graph (``rigid_iteration_graph``)."""

import math
import unittest

import numpy as np
import warp as wp

import newton


def _boxes_and_flap():
    """Three boxes dropped onto the ground, two of them onto each other, and a hinged flap they fall on."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=cfg)
    for x, z in ((0.0, 0.12), (0.02, 0.35), (0.6, 0.15)):
        body = builder.add_body(xform=wp.transform((x, 0.0, z), wp.quat_rpy(0.1, 0.2, 0.0)))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    flap = builder.add_link(xform=wp.transform((0.6, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(flap, hx=0.2, hy=0.05, hz=0.01, cfg=cfg)
    joint = builder.add_joint_revolute(
        -1,
        flap,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.4, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform((-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint])
    return builder.finalize(), {"nsn_sparse_storage": True, "nsn_linear_iterations": 10}


def _rod_on_a_box():
    """A particle rod dropped across a free box on the ground, the box's rows and the rod's in one solve."""
    from newton.solvers import SolverFBA  # noqa: PLC0415

    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    builder = newton.ModelBuilder()
    SolverFBA.register_custom_attributes(builder)
    builder.add_ground_plane(cfg=cfg)
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    radius = 0.002
    points = np.array([(x, 0.0, 0.11) for x in np.linspace(-0.15, 0.15, 31)])
    area = math.pi * radius * radius
    SolverFBA.add_particle_rod(builder, points, radius, 1e3 / area, density=0.002 / area, bending_stiffness=1e-5)
    model = builder.finalize()
    model.particle_mu = 0.5
    return model, {
        "nsn_sparse_storage": True,
        "nsn_linear_iterations": 10,
        "rayleigh_beta": 1e-4,
        "rayleigh_corotated": True,
    }


class TestRigidIterationGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _run(self, scene, graph, steps=60, dt=0.005):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model, kwargs = scene()
        solver = SolverFBA(model, iterations=10, rigid_iteration_graph=graph, **kwargs)
        s0, s1 = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.003)
        contacts = pipeline.contacts()
        for _ in range(steps):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, dt)
            s0, s1 = s1, s0
        return solver, s0

    def _check(self, scene, captured):
        eager, state_eager = self._run(scene, False)
        graph, state_graph = self._run(scene, True)
        self.assertIsNone(eager._rigid_iter_graph)
        if captured:
            self.assertNotIn(graph._rigid_iter_graph, (None, False))
        else:
            self.assertIsNone(graph._rigid_iter_graph)
        np.testing.assert_array_equal(state_graph.body_q.numpy(), state_eager.body_q.numpy())
        np.testing.assert_array_equal(state_graph.body_qd.numpy(), state_eager.body_qd.numpy())
        if state_eager.particle_q is not None:
            np.testing.assert_array_equal(state_graph.particle_q.numpy(), state_eager.particle_q.numpy())
        return state_graph

    def test_rows_built_anew_replay_bitwise(self):
        """A rod resting on a free box, whose rows are built anew at every step, replays its rigid iterations from
        the graph and steps bit for bit as with eager iterations."""
        state = self._check(_rod_on_a_box, captured=True)
        self.assertGreater(state.particle_q.numpy()[15, 2], 0.1)  # the rod's middle rests on the box's top

    def test_padded_rows_keep_their_own_graph(self):
        """Boxes falling onto each other and onto a hinged flap keep their padded rows, whose NSN step replays its
        own graph across steps: the option leaves them as they were."""
        state = self._check(_boxes_and_flap, captured=False)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())

    def test_off_by_default(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        self.assertFalse(SolverFBA(_boxes_and_flap()[0], iterations=1).rigid_iteration_graph)


if __name__ == "__main__":
    unittest.main(verbosity=2)
