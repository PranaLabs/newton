# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse storage of the full contact operator against the dense one, on the CPU and on CUDA."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA


def _cloth(builder, z):
    builder.add_cloth_grid(
        pos=wp.vec3(-0.08, -0.08, z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=8,
        dim_y=8,
        cell_x=0.02,
        cell_y=0.02,
        mass=0.005,
        tri_ke=1.0e3,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=1.0e-3,
        edge_kd=0.0,
        particle_radius=0.005,
    )


def _cloth_on_ground(device):
    builder = newton.ModelBuilder()
    _cloth(builder, 0.006)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _cloth_on_box(device):
    """A cloth falling onto a box that FBA integrates, on the ground."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    builder = newton.ModelBuilder()
    _cloth(builder, 0.106)
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return builder.finalize(device=device)


class TestFBASparseStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])

    def _run(self, make, device, steps, **options):
        model = make(device)
        solver = SolverFBA(model, iterations=6, **options)
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
        contacts = pipeline.contacts()
        a, b = model.state(), model.state()
        for _ in range(steps):
            pipeline.collide(a, contacts)
            solver.step(a, b, None, contacts, 1.0 / 120.0)
            a, b = b, a
        bodies = a.body_qd.numpy() if model.body_count else np.zeros((0, 6))
        return a.particle_q.numpy(), a.particle_qd.numpy(), bodies

    def test_matches_the_dense_operator(self):
        """Rest a cloth on the ground and drape one over a box that FBA integrates with the sparse storage as with the
        dense one, on every device: the cloth on the ground stays at rest, and the box stays still under the cloth."""
        for device in self.devices:
            for make in (_cloth_on_ground, _cloth_on_box):
                with self.subTest(device=device, scene=make.__name__):
                    dense = self._run(make, device, 30)
                    sparse = self._run(make, device, 30, nsn_sparse_storage=True)
                    np.testing.assert_allclose(sparse[0], dense[0], atol=1e-5)
                    if make is _cloth_on_ground:
                        self.assertLess(np.abs(sparse[1]).max(), 1e-3)
                    else:
                        self.assertLess(np.abs(sparse[2]).max(), 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
