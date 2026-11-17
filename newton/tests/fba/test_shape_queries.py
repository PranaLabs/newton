# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverFBA's own shape queries against Newton's particle contacts: barrels, switched flags, worlds and surface
velocities."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry import kernels as sdf_kernels
from newton._src.geometry.types import Axis, GeoType
from newton._src.solvers.fba.face_contact import SegmentContactQuery, shape_sdf
from newton.solvers import SolverFBA


@wp.kernel
def _barrel_distances(points: wp.array[wp.vec3], scale: wp.vec3, out: wp.array[wp.vec2]):
    i = wp.tid()
    x = points[i]
    fba = shape_sdf(int(GeoType.CYLINDER), scale, wp.uint64(0), x, 0.003)[0]
    reference = sdf_kernels.sdf_cylinder(x, scale[0], scale[1], int(Axis.Z), -1.0, scale[2])
    out[i] = wp.vec2(fba, reference)


class TestFBAShapeQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_barrel_cylinder(self):
        """Measure a barrel cylinder, wider at its middle, as Newton's particle contacts do."""
        builder = newton.ModelBuilder()
        shape = builder.add_shape_cylinder(-1, radius=0.1, half_height=0.1, barrel_radius=0.15)
        model = builder.finalize(device="cpu")
        scale = model.shape_scale.numpy()[shape]
        points = np.array([[r, 0.0, z] for r in (0.1, 0.12, 0.14, 0.16) for z in (0.0, 0.05)], dtype=np.float32)
        out = wp.zeros(len(points), dtype=wp.vec2, device="cpu")
        inputs = [wp.array(points, dtype=wp.vec3, device="cpu"), wp.vec3(*scale)]
        wp.launch(_barrel_distances, len(points), inputs, [out], device="cpu")
        np.testing.assert_allclose(out.numpy()[:, 0], out.numpy()[:, 1], atol=1e-6)

    def test_segment_contacts_read_switched_flags(self):
        """Take the segment contacts of a thin bar whose particle collisions are switched on after the first query,
        as a gripper switches its fingers' collisions."""
        builder = newton.ModelBuilder()
        turn = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2)
        bar = builder.add_shape_capsule(-1, xform=wp.transform(wp.vec3(0.0), turn), radius=0.001, half_height=0.02)
        for x in (-0.003, 0.003):
            builder.add_particle((x, 0.0, 0.0018), (0.0, 0.0, 0.0), 0.001, radius=0.001)
        model = builder.finalize(device="cpu")
        flags = model.shape_flags.numpy()
        query = SegmentContactQuery(np.array([[0, 1]]), "cpu")
        counts = []
        for on in (False, True, False):
            if on:
                flags[bar] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
            else:
                flags[bar] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
            model.shape_flags.assign(flags)
            counts.append(len(query.query(model, model.particle_q, None, 5.0e-4, 0.0)[0]))
        self.assertEqual(counts, [0, 1, 0])

    def test_global_particles_touch_local_shapes(self):
        """Find the face contacts of a global cloth on a box of a world, as Newton's particle contacts collide global
        particles with every shape."""
        builder = newton.ModelBuilder()
        world = newton.ModelBuilder()
        world.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.05)), hx=0.002, hy=0.2, hz=0.05)
        builder.add_world(world)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.05, -0.05, 0.002),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=1.0e3,
            particle_radius=0.001,
        )
        model = builder.finalize(device="cpu")
        self.assertEqual(int(model.particle_world.numpy().max()), -1)
        solver = SolverFBA(model, face_contacts=True, face_contact_gap=0.01)
        found = solver._face_query.query(
            model, model.particle_q, model.body_q, np.zeros(0, np.int32), np.zeros(0, np.int32), 0.003, 0.01
        )
        self.assertGreater(len(found[0]), 0)

    def test_surface_velocity_drags_the_cloth(self):
        """Carry a cloth resting on a static box at the surface velocity that its contacts report, as on a
        conveyor."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        cfg = newton.ModelBuilder.ShapeConfig(mu=1.0)
        builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.05)), hx=1.0, hy=1.0, hz=0.05, cfg=cfg)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.05, -0.05, 0.005),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=4,
            dim_y=4,
            cell_x=0.025,
            cell_y=0.025,
            mass=0.01,
            tri_ke=1.0e3,
            particle_radius=0.005,
        )
        model = builder.finalize(device="cpu")
        solver = SolverFBA(model, iterations=5)
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.005)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        for _ in range(60):
            pipeline.collide(s0, contacts)
            contacts.soft_contact_body_vel.fill_(wp.vec3(0.1, 0.0, 0.0))
            solver.step(s0, s1, None, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
        np.testing.assert_allclose(s0.particle_qd.numpy()[:, 0], 0.1, atol=5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
