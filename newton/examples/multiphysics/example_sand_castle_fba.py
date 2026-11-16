# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sand Castle FBA
#
# A wrecking ball swings into a sand castle. The castle is damp sand of
# SolverImplicitMPM, its grains held together by a small cohesion on top
# of their friction; the ball hangs on a rod from a pivot, one body that
# SolverFBA integrates on a revolute joint. The ball waits while the
# castle settles, then swings down through the castle's front wall and
# keep, and the sand slows it.
#
# The two solvers exchange the sand's impulse on the ball once per FBA
# step, MPM seeing the ball as a collider that turns about the pivot with
# its moment of inertia about it (fba_mpm_coupling.HingeCoupling).
#
# Command: python -m newton.examples sand_castle_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.fba_mpm_coupling import HingeCoupling, ParticleDrain
from newton.solvers import SolverFBA, SolverImplicitMPM

PIVOT = np.array([0.0, 0.0, 0.45])  # [m], the ball swings about y through this point
ROD_LENGTH = 0.37  # [m], from the pivot to the ball's center
BALL_RADIUS = 0.045  # [m]
ROD_RADIUS = 0.005  # [m]
STEEL_DENSITY = 7800.0  # [kg/m^3]
SAND_DENSITY = 1600.0  # [kg/m^3]
SAND_FRICTION = 0.6
START_ANGLE = np.radians(75.0)  # the rod's angle from the vertical before the swing
VOXEL = 0.0075  # [m], the sand's grid
CASTLE = np.array([0.095, 0.0])  # [m], the castle's center on the ground
STEEL = (0.42, 0.44, 0.47)
WOOD = (0.45, 0.32, 0.20)
SAND = np.array([0.80, 0.66, 0.42])
HIDDEN = (0.0, 0.0, -10.0)  # where the sand particles go once they leave the grid


def box_sdf(p, center, half):
    """The signed distance of the points ``p`` to an axis-aligned box."""
    q = np.abs(p - np.asarray(center)) - np.asarray(half)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def cylinder_sdf(p, center, radius, z0, z1):
    """The signed distance of the points ``p`` to a vertical cylinder from ``z0`` to ``z1`` about ``center``."""
    radial = np.linalg.norm(p[:, :2] - np.asarray(center), axis=1) - radius
    axial = np.abs(p[:, 2] - 0.5 * (z0 + z1)) - 0.5 * (z1 - z0)
    q = np.stack([radial, axial], axis=1)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def castle_sdf(p):
    """The castle: a square keep with a tower at each corner and one in the middle, with merlons on top."""
    cx, cy = CASTLE
    parts = [box_sdf(p, (cx, cy, 0.035), (0.06, 0.06, 0.035))]
    towers = [((cx + sx * 0.06, cy + sy * 0.06), 0.028, 0.11) for sx in (-1, 1) for sy in (-1, 1)]
    towers.append(((cx, cy), 0.032, 0.15))
    for center, radius, height in towers:
        parts.append(cylinder_sdf(p, center, radius, 0.0, height))
        for k in range(4):
            a = 0.25 * np.pi + 0.5 * np.pi * k
            merlon = (center[0] + 0.65 * radius * np.cos(a), center[1] + 0.65 * radius * np.sin(a), height + 0.008)
            parts.append(box_sdf(p, merlon, (0.0085, 0.0085, 0.008)))
    for along in (-0.022, 0.022):  # merlons on the keep's walls, between the towers
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x = cx + sx * 0.052 + (along if sx == 0 else 0.0)
            y = cy + sy * 0.052 + (along if sy == 0 else 0.0)
            parts.append(box_sdf(p, (x, y, 0.078), (0.008, 0.008, 0.008)))
    return np.min(parts, axis=0)


def castle_points(spacing, rng):
    """Points ``spacing`` apart, jittered, that fill the castle."""
    lo = np.array([CASTLE[0] - 0.1, CASTLE[1] - 0.1, 0.0])
    counts = np.ceil(np.array([0.2, 0.2, 0.17]) / spacing).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"), axis=-1).reshape(-1, 3)
    p = lo + (grid + 0.5) * spacing
    p = p[castle_sdf(p) < 0.0]
    return p + rng.uniform(-0.2, 0.2, p.shape) * spacing


def box_mesh(center, half):
    """An axis-aligned box as a closed warp mesh for the MPM colliders."""
    box = newton.Mesh.create_box(*half, duplicate_vertices=False, compute_normals=False, compute_uvs=False,
                                 compute_inertia=False)  # fmt: skip
    points = wp.array((np.asarray(box.vertices) + np.asarray(center)).astype(np.float32), dtype=wp.vec3)
    return wp.Mesh(points, wp.array(np.asarray(box.indices).ravel(), dtype=int), wp.zeros_like(points))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16  # FBA steps per frame, each with one MPM step
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.release = args.release  # [s], when the ball starts to swing

        # The rigid side: the ball on its rod, swinging about the pivot; the frame that holds the pivot is for show,
        # and the ball touches nothing rigid.
        builder = newton.ModelBuilder()
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        steel = newton.ModelBuilder.ShapeConfig(
            mu=0.0, density=STEEL_DENSITY, has_shape_collision=False, has_particle_collision=True
        )
        builder.add_ground_plane(height=0.0, cfg=visual, color=(0.16, 0.17, 0.15))
        along_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)  # a cylinder's axis (z) onto y
        post = wp.transform((PIVOT[0], 0.13, 0.5 * PIVOT[2] + 0.008), wp.quat_identity())
        builder.add_shape_box(-1, xform=post, hx=0.016, hy=0.016, hz=0.5 * PIVOT[2] + 0.024, cfg=visual, color=WOOD)
        arm = wp.transform((PIVOT[0], 0.06, PIVOT[2]), along_y)  # from the post to the pivot
        builder.add_shape_cylinder(-1, xform=arm, radius=0.009, half_height=0.085, cfg=visual, color=STEEL)
        start = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(START_ANGLE))
        self.ball = builder.add_link(xform=wp.transform(tuple(PIVOT), start))
        builder.add_shape_cylinder(self.ball, xform=wp.transform((0.0, 0.0, -0.5 * ROD_LENGTH), wp.quat_identity()),
                                   radius=ROD_RADIUS, half_height=0.5 * ROD_LENGTH, cfg=steel, color=STEEL)  # fmt: skip
        builder.add_shape_sphere(self.ball, xform=wp.transform((0.0, 0.0, -ROD_LENGTH), wp.quat_identity()),
                                 radius=BALL_RADIUS, cfg=steel, color=STEEL)  # fmt: skip
        builder.add_shape_cylinder(self.ball, xform=wp.transform((0.0, 0.0, 0.0), along_y), radius=0.014,
                                   half_height=0.02, cfg=visual, color=STEEL)  # fmt: skip
        joint = builder.add_joint_revolute(
            -1, self.ball, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform(tuple(PIVOT), wp.quat_identity())
        )
        builder.joint_q[builder.joint_q_start[joint]] = float(START_ANGLE)
        builder.add_articulation([joint])
        self.model = builder.finalize()
        self.solver = SolverFBA(self.model, iterations=5)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # The sand side: the castle, two particles per voxel edge, in a grid around the castle and the ball's
        # path, sized by two particles at its corners that the solver's setup sees.
        spacing = 0.5 * VOXEL
        castle = castle_points(spacing, np.random.default_rng(0))
        self.domain_lo = np.array([-0.25, -0.35, -0.05])
        self.domain_hi = np.array([0.75, 0.35, 0.40])
        positions = np.concatenate([castle, [self.domain_lo, self.domain_hi]])
        count = len(positions)
        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        sand_builder.add_particles(
            pos=[tuple(p) for p in positions],
            vel=[(0.0, 0.0, 0.0)] * count,
            mass=[SAND_DENSITY * spacing**3] * count,
            radius=[0.5 * spacing] * count,
            flags=[int(newton.ParticleFlags.ACTIVE)] * count,
        )
        self.sand_model = sand_builder.finalize()
        # Damp sand: friction, and a cohesion that holds the walls up but lets the ball through.
        self.sand_model.mpm.friction.fill_(SAND_FRICTION)
        self.sand_model.mpm.yield_stress.fill_(args.cohesion)
        config = SolverImplicitMPM.Config()
        config.voxel_size = VOXEL
        config.grid_type = "fixed"
        config.grid_padding = 2
        config.max_active_cell_count = 1 << 15
        config.max_iterations = 50
        config.tolerance = 1.0e-4
        config.velocity_basis = "Q1"
        config.collider_basis = "S2"
        self.mpm = SolverImplicitMPM(self.sand_model, config=config)
        self.sand_state = self.sand_model.state()
        self.sand_state.mpm.particle_Jp.fill_(1.0)
        # Sand that leaves the scene 2 cm inside the grid's box is taken out.
        self.drain = ParticleDrain(
            self.mpm, self.sand_model, self.sand_state, self.domain_lo + 0.02, self.domain_hi - 0.02, HIDDEN
        )
        # The grid is sized: the corner particles leave.
        self.drain.retire([count - 2, count - 1])
        self.castle = castle
        self.sand_colors = np.clip(SAND * np.random.default_rng(1).uniform(0.88, 1.08, (count, 1)), 0.0, 1.0)

        # The ball and the rod as MPM's collider turn about the pivot with their moment of inertia about it; the
        # ground is a static collider.
        self.coupling = HingeCoupling(self.model, self.state_0, [(self.ball, tuple(PIVOT), (0.0, 1.0, 0.0))])
        self.coupling.attach(self.sand_state)
        self.sand_state.body_f = wp.zeros_like(self.state_0.body_f)
        self.ground_mesh = box_mesh((0.2, 0.0, -0.05), (1.0, 1.0, 0.05))  # the solver keeps the mesh id only
        self.mpm.setup_collider(
            collider_meshes=[None, self.ground_mesh],
            collider_body_ids=[self.ball, None],
            collider_friction=[0.3, SAND_FRICTION],
            collider_projection_threshold=[0.25 * VOXEL] * 2,
            model=self.model,
            **self.coupling.setup_collider_args(),
        )
        self.swing = []  # the rod's angle from the vertical [rad] at each frame
        self.settled = None  # the castle's particles when the ball starts to swing

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.44, -0.74, 0.36), pitch=-10.5, yaw=120.7)

    def simulate(self):
        for _ in range(self.sim_substeps):
            if self.sim_time >= self.release:
                # FBA: the ball under the sand's impulse of the last step.
                self.state_0.clear_forces()
                self.coupling.apply_impulses(self.state_0, self.sim_dt)
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            else:
                self.coupling.hold(self.state_0)  # the ball waits
            # MPM: the sand against the ball as it moved over the step.
            self.coupling.update_colliders(self.state_0)
            self.mpm.step(self.sand_state, self.sand_state, None, None, self.sim_dt)
            self.mpm.project_outside(self.sand_state, self.sand_state, self.sim_dt)
            self.coupling.collect_impulses(self.mpm, self.sand_state)
            self.drain.drain()

    def step(self):
        if self.settled is None and self.sim_time >= self.release:
            self.settled = self.sand_state.particle_q.numpy()[: len(self.castle)].copy()
        self.simulate()
        self.sim_time += self.frame_dt
        q = self.state_0.body_q.numpy()[self.ball]
        self.swing.append(float(2.0 * np.arctan2(q[4], q[6])))

    def test_post_step(self):
        q = self.state_0.body_q.numpy()[self.ball]
        assert np.all(np.isfinite(q)), f"non-finite ball pose at t={self.sim_time:.3f} s"
        offset = float(np.linalg.norm(q[:3] - PIVOT))
        assert offset < 1.0e-3, f"the ball left its pivot by {offset:.4f} m at t={self.sim_time:.3f} s"

    def test_final(self):
        # The castle stood while the ball waited.
        assert self.settled is not None, "the ball never swung"
        sag = np.linalg.norm(self.settled - self.castle, axis=1)
        assert np.percentile(sag, 99) < 0.01, f"the castle slumped by {np.percentile(sag, 99):.3f} m before the swing"
        # The ball went through the castle, and the sand took most of its swing.
        moved = np.linalg.norm(self.sand_state.particle_q.numpy()[: len(self.castle)] - self.settled, axis=1)
        assert (moved > 0.02).sum() > 1000, f"only {(moved > 0.02).sum()} grains moved more than 2 cm"
        lowest = float(np.min(self.swing))  # a free swing reaches -72 deg
        assert lowest > -0.85 * START_ANGLE, f"the ball swung through to {np.degrees(lowest):.1f} deg"
        q = self.sand_state.particle_q.numpy()[self.drain.active()]
        assert np.all(np.isfinite(q)), "non-finite sand"

    def render(self):
        active = self.drain.active()
        points = self.sand_state.particle_q.numpy()[active]
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if len(points):
            self.viewer.log_points(
                "/sand",
                wp.array(points, dtype=wp.vec3),
                0.95 * VOXEL * 0.25,
                wp.array(self.sand_colors[active], dtype=wp.vec3),
            )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--cohesion", type=float, default=700.0, help="The sand's cohesion [Pa], its yield stress.")
    parser.add_argument("--release", type=float, default=0.5, help="When the ball starts to swing [s].")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
