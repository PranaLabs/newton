# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Shishi-odoshi FBA
#
# A shishi-odoshi: water from a spout fills the mouth of a bamboo tube on a
# hinge until the water's weight tips it; the tube pours the water into a
# basin, swings back and knocks its tail on a stone, and fills again. The
# water is SolverAPICWater; the tube is a body that SolverFBA integrates,
# its hinge a revolute joint, resting on the stone and the basin's rim
# through convex contacts. The two exchange the water's pressure once per
# step:
#
#   1. The FBA step applies the water's impulse on the tube over the
#      previous water step as a constant force.
#   2. The water step sees the tube where the FBA step left it, the closed
#      parts of the grid's faces moving with it.
#
# After the first frame, each frame's steps replay from one CUDA graph. The
# tube starts nearly full, so that it first pours within a second.
#
# Command: python -m newton.examples shishi_odoshi_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics import shishi_odoshi_fba_utils as geo
from newton.solvers import SolverAPICWater, SolverFBA

WATER_DENSITY = 1000.0
BAMBOO = (0.80, 0.66, 0.42)
STONE = (0.55, 0.47, 0.36)
GRANITE = (0.34, 0.34, 0.36)
WOOD = (0.36, 0.28, 0.20)
WATER = (0.25, 0.52, 0.92)
HIDDEN = (0.0, 0.0, -10.0)  # where the water particles wait before they leave the spout


def convex_hull(points):
    """The convex hull of ``points`` as a collision mesh, every point kept as a candidate vertex."""
    cloud = newton.Mesh(np.asarray(points, dtype=np.float32), [0, 1, 2], compute_inertia=False, maxhullvert=len(points))
    hull = cloud.compute_convex_hull()
    return newton.Mesh(hull.vertices, hull.indices, compute_inertia=False, maxhullvert=len(points))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10  # FBA steps per frame, each with one water step
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # The rigid side: the tube on its hinge, resting on the stone and stopped by the basin's back rim. The ground,
        # the basin and the stone hold the water too.
        builder = newton.ModelBuilder()
        contact = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0, gap=0.005, has_particle_collision=True)
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        builder.add_ground_plane(height=geo.GROUND_Z, cfg=contact, color=(0.16, 0.17, 0.15))
        boxes = geo.basin_boxes()
        for center, half in boxes:
            box = wp.transform(center, wp.quat_identity())
            builder.add_shape_box(-1, xform=box, hx=half[0], hy=half[1], hz=half[2], cfg=contact, color=GRANITE)
        stone_center, stone_points = geo.stone()
        stone = convex_hull(stone_points)
        builder.add_shape_convex_hull(-1, xform=wp.transform(tuple(stone_center), wp.quat_identity()), mesh=stone,
                                      cfg=contact, color=STONE)  # fmt: skip
        # The posts that hold the hinge pin, the pin and the spout's pipe, for show.
        for side in (-1.0, 1.0):
            builder.add_shape_box(-1, xform=wp.transform((0.0, side * 0.062, -0.10), wp.quat_identity()), hx=0.025,
                                  hy=0.015, hz=0.20, cfg=visual, color=WOOD)  # fmt: skip
        pin = wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi))
        builder.add_shape_cylinder(-1, xform=pin, radius=0.006, half_height=0.08, cfg=visual, color=(0.25, 0.22, 0.18))
        along = np.array([np.cos(geo.SPOUT_SLOPE), 0.0, -np.sin(geo.SPOUT_SLOPE)])
        pipe = wp.transform(
            tuple(geo.SPOUT_END - 0.247 * along), wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*along))
        )
        builder.add_shape_cylinder(-1, xform=pipe, radius=geo.SPOUT_RADIUS, half_height=0.25, cfg=visual,
                                   color=(0.50, 0.42, 0.28))  # fmt: skip
        # The tube: the mass of the hollow bamboo, a closed mesh that the water and the viewer see, and two convex
        # parts for its contacts with the stone and the basin.
        mass, com, inertia = geo.tube_mass_properties()
        rest = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(geo.REST))
        self.tube = builder.add_link(xform=wp.transform((0.0, 0.0, 0.0), rest), mass=mass, com=tuple(com),
                                     inertia=wp.mat33(inertia), lock_inertia=True)  # fmt: skip
        vertices, triangles = geo.tube_mesh()
        water_contact = newton.ModelBuilder.ShapeConfig(
            mu=0.0, density=0.0, has_shape_collision=False, has_particle_collision=True
        )
        builder.add_shape_mesh(self.tube, mesh=newton.Mesh(vertices, triangles.ravel(), compute_inertia=False),
                               cfg=water_contact, color=BAMBOO)  # fmt: skip
        hidden = newton.ModelBuilder.ShapeConfig(
            mu=0.0, density=0.0, gap=0.005, has_particle_collision=False, is_visible=False
        )
        for part in geo.tube_convex_parts():
            builder.add_shape_convex_hull(self.tube, mesh=convex_hull(part), cfg=hidden)
        builder.add_articulation([builder.add_joint_revolute(-1, self.tube, axis=(0.0, 1.0, 0.0))])
        self.model = builder.finalize()
        self.solver = SolverFBA(
            self.model, iterations=5, nsn_linear_iterations=10, nsn_sparse_storage=True, contact_warmstart=True
        )
        # The tube's few contacts: a small buffer keeps the steps that replay from a graph cheap.
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=64)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # The water side: particles 2.5 mm apart, two per 5 mm cell edge, filling the mouth to below the brim, and the
        # spout's stream of 17 mL/s, which takes particles from a pool of inactive ones.
        voxel = 0.005
        spacing = 0.5 * voxel
        flow = 17.0e-6
        prefill = geo.prefill_points(spacing, args.fill_level, np.random.default_rng(0))
        pool = int(args.water_seconds * flow / spacing**3)
        positions = np.concatenate([prefill, np.tile(HIDDEN, (pool, 1))])
        count = len(positions)
        water_builder = newton.ModelBuilder()
        SolverAPICWater.register_custom_attributes(water_builder)
        water_builder.add_particles(
            pos=[tuple(p) for p in positions],
            vel=[(0.0, 0.0, 0.0)] * count,
            mass=[WATER_DENSITY * spacing**3] * count,
            radius=[0.5 * spacing] * count,
            flags=[int(newton.ParticleFlags.ACTIVE)] * len(prefill) + [0] * pool,
        )
        self.water_model = water_builder.finalize()
        config = SolverAPICWater.Config(
            voxel_size=voxel, domain_lo=(-0.60, -0.20, geo.GROUND_Z - 0.02), domain_hi=(0.10, 0.20, 0.22)
        )
        self.water = SolverAPICWater(self.water_model, config)
        # The water meets the shapes that collide with particles: the ground, the basin, the stone and the tube.
        self.water.setup_colliders(self.model)
        along = np.array([np.cos(geo.SPOUT_SLOPE), 0.0, -np.sin(geo.SPOUT_SLOPE)])
        origin = geo.SPOUT_END - 0.006 * np.array([np.sin(geo.SPOUT_SLOPE), 0.0, np.cos(geo.SPOUT_SLOPE)])
        self.water.add_stream(origin, along, flow, side_count=3)
        self.water_state = self.water_model.state()
        # The water steps see the tube in state_0, which the FBA steps' results are copied into.
        self.water_state.body_q = self.state_0.body_q
        self.water_state.body_qd = self.state_0.body_qd
        self.graph = None

        self.lowest_angle = float(geo.REST)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.03, -0.91, 0.42), pitch=-30.0, yaw=90.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            # FBA: the tube under the water's impulse of the last step.
            self.state_0.clear_forces()
            self.water.add_body_forces(self.state_0.body_f, self.sim_dt)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            wp.copy(self.state_0.body_q, self.state_1.body_q)
            wp.copy(self.state_0.body_qd, self.state_1.body_qd)
            # The water against the tube where the step left it.
            self.water.step(self.water_state, self.water_state, None, None, self.sim_dt)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
            if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
                # The first frame, outside the graph, sets FBA up and reserves its contacts' buffers; the following
                # frames replay its steps.
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
        self.sim_time += self.frame_dt
        q = self.state_0.body_q.numpy()[self.tube]
        self.lowest_angle = min(self.lowest_angle, 2.0 * float(np.arctan2(q[4], q[6])))

    def test_post_step(self):
        q = self.state_0.body_q.numpy()[self.tube]
        assert np.all(np.isfinite(q)), f"non-finite tube pose at t={self.sim_time:.3f} s"
        # The tube stays on its hinge and between the stone and the basin's rim.
        assert np.linalg.norm(q[:3]) < 1.0e-3, f"the tube left its hinge by {np.linalg.norm(q[:3]):.4f} m"
        angle = 2.0 * float(np.arctan2(q[4], q[6]))
        assert geo.TILT - 0.05 < angle < geo.REST + 0.05, f"tube angle {np.degrees(angle):.1f} deg"

    def test_final(self):
        # The water tipped the tube onto the basin's rim.
        assert self.lowest_angle < geo.TILT + 0.1, f"the tube turned only to {np.degrees(self.lowest_angle):.1f} deg"
        q = self.water_state.particle_q.numpy()
        active = self.active()
        assert np.all(np.isfinite(q[active])), "non-finite water"
        x0, x1, y0, y1 = geo.BASIN_INNER
        in_basin = (q[:, 0] > x0) & (q[:, 0] < x1) & (q[:, 1] > y0) & (q[:, 1] < y1) & (q[:, 2] < -0.15) & active
        assert in_basin.sum() > 1000, f"only {in_basin.sum()} water particles reached the basin"

    def active(self):
        """Whether each water particle is in the water."""
        return (self.water_model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0

    def render(self):
        points = self.water_state.particle_q.numpy()[self.active()]
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if len(points):
            radius = 0.95 * float(self.water_model.particle_radius.numpy()[0]) * 2.0
            self.viewer.log_points(
                "/water",
                wp.array(points, dtype=wp.vec3),
                radius,
                wp.full(len(points), value=wp.vec3(*WATER), dtype=wp.vec3),
            )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--fill-level", type=float, default=0.076, help="World height [m] of the water the tube starts with."
    )
    parser.add_argument(
        "--water-seconds", type=float, default=40.0, help="Seconds of water that the pool of particles holds."
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
