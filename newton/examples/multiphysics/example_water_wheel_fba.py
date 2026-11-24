# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Water Wheel FBA
#
# An overshot water wheel: a stream from a bamboo pipe falls into the
# buckets between the paddles just past the wheel's top, and their weight
# turns the wheel; the buckets empty near the bottom into a trough that
# carries the water away. The wheel turns on a revolute joint whose
# passive damping (Model.joint_damping) is the wheel's load, so that it
# settles at the speed where the water's torque meets the load's.
#
# The water is SolverAPICWater; the wheel is a body that SolverFBA
# integrates. They exchange the water's pressure once per step: the FBA
# step applies the water's impulse on the wheel over the previous water
# step as a constant force, and the water step sees the wheel where the
# FBA step left it. After the first frame, each frame's steps replay from
# one CUDA graph.
#
# Command: python -m newton.examples water_wheel_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverAPICWater, SolverFBA

AXLE = np.array([0.0, 0.0, 0.30])  # [m], the wheel's axle runs along y through this point
RADIUS = 0.12  # [m], the wheel's outer radius
SOLE_RADIUS = 0.06  # [m], the hub's radius, the buckets' bottom
WIDTH = 0.08  # [m], between the wheel's side discs
WALL = 0.01  # [m], the discs' and the paddles' thickness
PADDLES = 12
PADDLE_LEAN = np.radians(15.0)  # how far each paddle's rim end trails its inner end, so that the buckets hold
WOOD_DENSITY = 600.0  # [kg/m^3]
WATER_DENSITY = 1000.0  # [kg/m^3]
FLOW = 80.0e-6  # [m^3/s], the pipe's flow
VOXEL = 0.005  # [m], the water's grid
SPOUT_SLOPE = np.radians(22.0)  # the pipe falls toward +x
SPOUT_ALONG = np.array([np.cos(SPOUT_SLOPE), 0.0, -np.sin(SPOUT_SLOPE)])
SPOUT_END = AXLE + np.array([-0.01, 0.0, RADIUS + 0.05])
SPOUT_RADIUS = 0.014  # [m]
TROUGH_SLOPE = np.radians(8.0)  # the trough's floor falls toward +x
WOOD = (0.55, 0.40, 0.25)
DARK_WOOD = (0.36, 0.25, 0.15)
GRANITE = (0.34, 0.34, 0.36)
WATER = (0.25, 0.52, 0.92)
HIDDEN = (0.0, 0.0, -10.0)  # where the water particles wait before they leave the pipe


def about_y(angle):
    """A rotation by ``angle`` [rad] about +y."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle))


def paddle(k):
    """Paddle ``k`` in the wheel's frame as ``(center, rotation, half length)``: a board from the hub at its angle,
    measured from +x toward +z, to the rim ``PADDLE_LEAN`` further on, behind it as the wheel turns."""
    angle = 2.0 * np.pi * k / PADDLES
    inner = SOLE_RADIUS * np.array([np.cos(angle), 0.0, np.sin(angle)])
    outer = RADIUS * np.array([np.cos(angle + PADDLE_LEAN), 0.0, np.sin(angle + PADDLE_LEAN)])
    d = outer - inner
    return tuple(0.5 * (inner + outer)), about_y(-np.arctan2(d[2], d[0])), 0.5 * float(np.linalg.norm(d))


def trough_floor_z(x):
    """The height of the trough floor's middle at ``x`` [m]."""
    return AXLE[2] - RADIUS - 0.05 - (x - 0.07) * np.tan(TROUGH_SLOPE)


def trough_boxes():
    """The trough under the wheel as boxes ``(center, rotation, half extents)``: its floor, falling toward +x, and
    its two side walls."""
    floor_z = trough_floor_z(0.07)
    tilt = about_y(TROUGH_SLOPE)
    half_y = 0.5 * WIDTH + WALL + 0.02
    return [
        ((0.07, 0.0, floor_z), tilt, (0.26, half_y + 0.01, 0.005)),
        ((0.07, half_y, floor_z + 0.04), tilt, (0.26, 0.005, 0.045)),
        ((0.07, -half_y, floor_z + 0.04), tilt, (0.26, 0.005, 0.045)),
    ]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10  # FBA steps per frame, each with one water step
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # The rigid side: the wheel on its axle, its load the joint's damping. The posts, the axle and the pipe are
        # for show, the trough holds the water alone; the wheel touches nothing rigid.
        builder = newton.ModelBuilder()
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        water_only = newton.ModelBuilder.ShapeConfig(
            density=0.0, has_shape_collision=False, has_particle_collision=True
        )
        wood = newton.ModelBuilder.ShapeConfig(
            mu=0.0, density=WOOD_DENSITY, has_shape_collision=False, has_particle_collision=True
        )
        builder.add_ground_plane(height=0.0, cfg=visual, color=(0.16, 0.17, 0.15))
        trough = trough_boxes()
        half_y = trough[0][2][1]
        for center, rotation, half in trough:
            builder.add_shape_box(-1, xform=wp.transform(center, rotation), hx=half[0], hy=half[1], hz=half[2],
                                  cfg=water_only, color=GRANITE)  # fmt: skip
        for x in (-0.13, 0.27):
            top = trough_floor_z(x) - 0.005
            builder.add_shape_box(-1, xform=wp.transform((x, 0.0, 0.5 * top), wp.quat_identity()), hx=0.015,
                                  hy=half_y, hz=0.5 * top, cfg=visual, color=GRANITE)  # fmt: skip
        for side in (-1.0, 1.0):
            post = wp.transform((0.0, side * (0.5 * WIDTH + WALL + 0.045), 0.5 * AXLE[2]), wp.quat_identity())
            builder.add_shape_box(-1, xform=post, hx=0.015, hy=0.01, hz=0.5 * AXLE[2], cfg=visual, color=WOOD)
        along_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)  # a cylinder's axis (z) onto y
        axle = wp.transform(tuple(AXLE), along_y)
        builder.add_shape_cylinder(-1, xform=axle, radius=0.008, half_height=0.5 * WIDTH + WALL + 0.06, cfg=visual,
                                   color=(0.25, 0.22, 0.18))  # fmt: skip
        pipe = wp.transform(tuple(SPOUT_END - 0.2 * SPOUT_ALONG),
                            wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*SPOUT_ALONG)))  # fmt: skip
        builder.add_shape_cylinder(-1, xform=pipe, radius=SPOUT_RADIUS, half_height=0.2, cfg=visual,
                                   color=(0.50, 0.42, 0.28))  # fmt: skip
        rest = SPOUT_END - 0.32 * SPOUT_ALONG  # the pipe rests on a post there
        height = rest[2] - SPOUT_RADIUS
        builder.add_shape_box(-1, xform=wp.transform((rest[0], 0.0, 0.5 * height), wp.quat_identity()), hx=0.012,
                              hy=0.012, hz=0.5 * height, cfg=visual, color=WOOD)  # fmt: skip

        # The wheel: a hub, two side discs and the paddles between them, the buckets the paddles and discs enclose;
        # dark ribs on the discs show where the paddles run.
        self.wheel = builder.add_link(xform=wp.transform(tuple(AXLE), wp.quat_identity()))
        builder.add_shape_cylinder(self.wheel, xform=wp.transform((0.0, 0.0, 0.0), along_y), radius=SOLE_RADIUS,
                                   half_height=0.5 * WIDTH + WALL, cfg=wood, color=WOOD)  # fmt: skip
        for side in (-1.0, 1.0):
            disc = wp.transform((0.0, side * 0.5 * (WIDTH + WALL), 0.0), along_y)
            builder.add_shape_cylinder(self.wheel, xform=disc, radius=RADIUS, half_height=0.5 * WALL, cfg=wood,
                                       color=WOOD)  # fmt: skip
        for k in range(PADDLES):
            center, rotation, half = paddle(k)
            builder.add_shape_box(self.wheel, xform=wp.transform(center, rotation), hx=half, hy=0.5 * WIDTH,
                                  hz=0.5 * WALL, cfg=wood, color=DARK_WOOD)  # fmt: skip
            for side in (-1.0, 1.0):
                rib = (center[0], side * (0.5 * WIDTH + WALL + 0.0015), center[2])
                builder.add_shape_box(self.wheel, xform=wp.transform(rib, rotation), hx=half, hy=0.0015,
                                      hz=0.5 * WALL, cfg=visual, color=DARK_WOOD)  # fmt: skip
        joint = builder.add_joint_revolute(
            -1, self.wheel, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform(tuple(AXLE), wp.quat_identity()),
            damping=args.load,
        )  # fmt: skip
        builder.add_articulation([joint])
        self.model = builder.finalize()
        self.solver = SolverFBA(self.model, iterations=5)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # The water side: the pipe's stream, 4 x 4 particles 2.5 mm apart across, from a pool of inactive particles
        # that the water leaving the grid's box returns to, in a grid around the wheel and the trough's upper end.
        spacing = 0.5 * VOXEL
        pool = args.pool
        water_builder = newton.ModelBuilder()
        SolverAPICWater.register_custom_attributes(water_builder)
        water_builder.add_particles(
            pos=[HIDDEN] * pool,
            vel=[(0.0, 0.0, 0.0)] * pool,
            mass=[WATER_DENSITY * spacing**3] * pool,
            radius=[0.5 * spacing] * pool,
            flags=[0] * pool,
        )
        self.water_model = water_builder.finalize()
        config = SolverAPICWater.Config(
            voxel_size=VOXEL,
            domain_lo=tuple(AXLE + np.array([-RADIUS - 0.06, -0.5 * WIDTH - WALL - 0.03, -RADIUS - 0.12])),
            domain_hi=tuple(AXLE + np.array([RADIUS + 0.20, 0.5 * WIDTH + WALL + 0.03, RADIUS + 0.13])),
        )
        self.water = SolverAPICWater(self.water_model, config)
        # The water meets the wheel and the trough, the shapes that collide with particles.
        self.water.setup_colliders(self.model)
        # The stream leaves the pipe a little below its axis.
        origin = SPOUT_END - 0.006 * np.array([np.sin(SPOUT_SLOPE), 0.0, np.cos(SPOUT_SLOPE)])
        self.water.add_stream(origin, SPOUT_ALONG, FLOW, side_count=4)
        self.water_state = self.water_model.state()
        # The water steps see the wheel in state_0, which the FBA steps' results are copied into.
        self.water_state.body_q = self.state_0.body_q
        self.water_state.body_qd = self.state_0.body_qd
        self.graph = None
        self.spin = []  # the wheel's angular velocity [rad/s] about +y at each frame

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.53, -0.67, 0.54), pitch=-18.0, yaw=128.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            # FBA: the wheel under the water's impulse of the last step and its load.
            self.state_0.clear_forces()
            self.water.add_body_forces(self.state_0.body_f, self.sim_dt)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            wp.copy(self.state_0.body_q, self.state_1.body_q)
            wp.copy(self.state_0.body_qd, self.state_1.body_qd)
            # The water against the wheel where the step left it.
            self.water.step(self.water_state, self.water_state, None, None, self.sim_dt)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
            if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
                # The first frame, outside the graph, sets FBA up; the following frames replay its steps.
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
        self.sim_time += self.frame_dt
        self.spin.append(float(self.state_0.body_qd.numpy()[self.wheel][4]))

    def test_post_step(self):
        q = self.state_0.body_q.numpy()[self.wheel]
        assert np.all(np.isfinite(q)), f"non-finite wheel pose at t={self.sim_time:.3f} s"
        offset = float(np.linalg.norm(q[:3] - AXLE))
        assert offset < 1.0e-3, f"the wheel left its axle by {offset:.4f} m at t={self.sim_time:.3f} s"

    def test_final(self):
        # The water past the top (+x) turns the wheel about +y, which lowers that side.
        spin = np.array(self.spin[-self.fps :])
        assert spin.mean() > 0.5, f"the wheel turns at {spin.mean():.3f} rad/s over the last second"
        q = self.water_state.particle_q.numpy()[self.active()]
        assert np.all(np.isfinite(q)), "non-finite water"

    def active(self):
        """Whether each water particle is in the water."""
        return (self.water_model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0

    def render(self):
        points = self.water_state.particle_q.numpy()[self.active()]
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if len(points):
            self.viewer.log_points(
                "/water",
                wp.array(points, dtype=wp.vec3),
                0.95 * VOXEL * 0.5,
                wp.full(len(points), value=wp.vec3(*WATER), dtype=wp.vec3),
            )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--load", type=float, default=0.01, help="Damping [N m s/rad] of the wheel's axle, its load.")
    parser.add_argument("--pool", type=int, default=20000, help="Water particles that the pipe recycles.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
