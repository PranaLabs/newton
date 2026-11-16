# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Batter Mixer FBA
#
# A mixer's paddle stirs thick batter, vanilla and chocolate, in a bowl.
# The paddle turns on a vertical shaft, a body that SolverFBA integrates on
# a revolute joint, under a motor's constant torque; the batter is a
# viscous fluid of SolverImplicitMPM that sticks to the bowl, and its drag
# holds the paddle at the speed where the two torques meet: thicker batter
# turns it slower. The paddle marbles the two batters as it turns.
#
# The two solvers exchange the batter's impulse on the paddle once per FBA
# step, MPM seeing the paddle as a collider that turns about its shaft with
# its moment of inertia about it (fba_mpm_coupling.HingeCoupling).
#
# Command: python -m newton.examples batter_mixer_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.fba_mpm_coupling import HingeCoupling, ParticleDrain
from newton.solvers import SolverFBA, SolverImplicitMPM

BOWL_RADIUS = 0.085  # [m], inside
BOWL_DEPTH = 0.08  # [m], inside
WALL = 0.006  # [m]
BATTER_DEPTH = 0.04  # [m]
SHAFT = np.array([0.0, 0.0, 0.0])  # [m], the shaft turns about z through this point
PADDLE_RADIUS = 0.06  # [m], the paddle's reach from the shaft
PADDLE_HEIGHT = 0.045  # [m]
PADDLE_LIFT = 0.006  # [m], between the paddle and the bowl's floor
STEEL_DENSITY = 7800.0  # [kg/m^3]
BATTER_DENSITY = 1100.0  # [kg/m^3]
BATTER_FRICTION = 1.0  # of the batter on the bowl and the paddle: it sticks to them
VOXEL = 0.005  # [m], the batter's grid
STEEL = (0.70, 0.72, 0.75)
BOWL = (0.86, 0.84, 0.80)
VANILLA = np.array([0.93, 0.80, 0.52])
CHOCOLATE = np.array([0.38, 0.22, 0.12])
HIDDEN = (0.0, 0.0, -10.0)  # where the batter particles go once they leave the grid


@wp.kernel
def motor_kernel(body: int, axis: wp.vec3, torque: float, body_f: wp.array[wp.spatial_vector]):
    """Add the motor's torque about the shaft to the paddle's forces."""
    body_f[body] = body_f[body] + wp.spatial_vector(wp.vec3(0.0), axis * torque)


def bowl_boxes(segments=24):
    """The bowl as boxes ``(center, rotation, half extents)``: a floor and a ring of wall segments."""
    boxes = [((0.0, 0.0, -0.5 * WALL), wp.quat_identity(), (BOWL_RADIUS + WALL, BOWL_RADIUS + WALL, 0.5 * WALL))]
    width = 2.0 * (BOWL_RADIUS + WALL) * np.tan(np.pi / segments) + 0.002
    for k in range(segments):
        a = 2.0 * np.pi * k / segments
        center = ((BOWL_RADIUS + 0.5 * WALL) * np.cos(a), (BOWL_RADIUS + 0.5 * WALL) * np.sin(a), 0.5 * BOWL_DEPTH)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(a))
        boxes.append((center, rotation, (0.5 * WALL, 0.5 * width, 0.5 * BOWL_DEPTH)))
    return boxes


def box_mesh(center, rotation, half):
    """A box as a closed warp mesh for the MPM colliders."""
    box = newton.Mesh.create_box(*half, duplicate_vertices=False, compute_normals=False, compute_uvs=False,
                                 compute_inertia=False)  # fmt: skip
    points = np.array([wp.quat_rotate(rotation, wp.vec3(*p)) for p in box.vertices]) + np.asarray(center)
    points = wp.array(points.astype(np.float32), dtype=wp.vec3)
    return wp.Mesh(points, wp.array(np.asarray(box.indices).ravel(), dtype=int), wp.zeros_like(points))


def batter_points(spacing, rng):
    """Points ``spacing`` apart, slightly jittered, that fill the bowl to the batter's depth."""
    n = int(np.ceil(2.0 * BOWL_RADIUS / spacing))
    m = int(np.ceil(BATTER_DEPTH / spacing))
    grid = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(m), indexing="ij"), axis=-1).reshape(-1, 3)
    p = np.array([-BOWL_RADIUS, -BOWL_RADIUS, 0.0]) + (grid + 0.5) * spacing
    p = p[np.hypot(p[:, 0], p[:, 1]) < BOWL_RADIUS - 0.5 * spacing]
    return p + rng.uniform(-0.15, 0.15, p.shape) * spacing


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16  # FBA steps per frame, each with one MPM step
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.torque = args.torque  # [N m], the motor's

        # The rigid side: the paddle on its shaft; the bowl is for show, and the paddle touches nothing rigid.
        builder = newton.ModelBuilder()
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        steel = newton.ModelBuilder.ShapeConfig(
            mu=0.0, density=STEEL_DENSITY, has_shape_collision=False, has_particle_collision=True
        )
        builder.add_ground_plane(height=-WALL, cfg=visual, color=(0.16, 0.17, 0.15))
        self.bowl = bowl_boxes()
        builder.add_shape_cylinder(-1, xform=wp.transform((0.0, 0.0, -0.5 * WALL), wp.quat_identity()),
                                   radius=BOWL_RADIUS + WALL, half_height=0.5 * WALL, cfg=visual, color=BOWL)  # fmt: skip
        for center, rotation, half in self.bowl[1:]:
            builder.add_shape_box(-1, xform=wp.transform(center, rotation), hx=half[0], hy=half[1], hz=half[2],
                                  cfg=visual, color=BOWL)  # fmt: skip
        top = BOWL_DEPTH + 0.25  # the shaft's top, out of view
        self.paddle = builder.add_link(xform=wp.transform(tuple(SHAFT), wp.quat_identity()))
        shaft_bottom = PADDLE_LIFT + 0.5 * PADDLE_HEIGHT
        shaft = wp.transform((0.0, 0.0, 0.5 * (shaft_bottom + top)), wp.quat_identity())
        builder.add_shape_cylinder(self.paddle, xform=shaft, radius=0.005, half_height=0.5 * (top - shaft_bottom),
                                   cfg=steel, color=STEEL)  # fmt: skip
        # The paddle: two blades across the shaft, each a frame of three bars around an open window.
        bar = 0.004
        z_mid = PADDLE_LIFT + 0.5 * PADDLE_HEIGHT
        for side in (-1.0, 1.0):
            for z in (PADDLE_LIFT + 0.5 * bar, PADDLE_LIFT + PADDLE_HEIGHT - 0.5 * bar):
                rail = wp.transform((side * 0.5 * PADDLE_RADIUS, 0.0, z), wp.quat_identity())
                builder.add_shape_box(self.paddle, xform=rail, hx=0.5 * PADDLE_RADIUS, hy=0.5 * bar, hz=0.5 * bar,
                                      cfg=steel, color=STEEL)  # fmt: skip
            for x in (PADDLE_RADIUS - 0.5 * bar, 0.5 * PADDLE_RADIUS):
                post = wp.transform((side * x, 0.0, z_mid), wp.quat_identity())
                builder.add_shape_box(self.paddle, xform=post, hx=0.5 * bar, hy=0.5 * bar, hz=0.5 * PADDLE_HEIGHT,
                                      cfg=steel, color=STEEL)  # fmt: skip
        joint = builder.add_joint_revolute(
            -1, self.paddle, axis=(0.0, 0.0, 1.0), parent_xform=wp.transform(tuple(SHAFT), wp.quat_identity())
        )
        builder.add_articulation([joint])
        self.model = builder.finalize()
        self.solver = SolverFBA(self.model, iterations=5)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # The batter side: the bowl's batter, two particles per voxel edge, in a grid around the bowl, sized by two
        # particles at its corners that the solver's setup sees.
        spacing = 0.5 * VOXEL
        batter = batter_points(spacing, np.random.default_rng(4))
        reach = BOWL_RADIUS + WALL + 0.02
        self.domain_lo = np.array([-reach, -reach, -WALL - 0.02])
        self.domain_hi = np.array([reach, reach, BOWL_DEPTH + 0.04])
        positions = np.concatenate([batter, [self.domain_lo, self.domain_hi]])
        count = len(positions)
        batter_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(batter_builder)
        batter_builder.add_particles(
            pos=[tuple(p) for p in positions],
            vel=[(0.0, 0.0, 0.0)] * count,
            mass=[BATTER_DENSITY * spacing**3] * count,
            radius=[0.5 * spacing] * count,
            flags=[int(newton.ParticleFlags.ACTIVE)] * count,
        )
        self.batter_model = batter_builder.finalize()
        # Batter: a viscous fluid, no friction between its grains, and pressure that holds in tension as in
        # compression.
        self.batter_model.mpm.friction.fill_(0.0)
        self.batter_model.mpm.tensile_yield_ratio.fill_(1.0)
        self.batter_model.mpm.viscosity.fill_(args.viscosity)
        config = SolverImplicitMPM.Config()
        config.voxel_size = VOXEL
        config.grid_type = "fixed"
        config.grid_padding = 2
        config.max_active_cell_count = 1 << 16
        config.max_iterations = 50
        config.tolerance = 1.0e-4
        config.strain_basis = "P1d"  # the default P0 lets a fluid at rest compress
        config.velocity_basis = "Q1"
        config.collider_basis = "S2"
        self.mpm = SolverImplicitMPM(self.batter_model, config=config)
        self.batter_state = self.batter_model.state()
        self.batter_state.mpm.particle_Jp.fill_(1.0)
        # Batter that would leave the grid is taken out, 1 cm inside the grid's box.
        self.drain = ParticleDrain(
            self.mpm, self.batter_model, self.batter_state, self.domain_lo + 0.01, self.domain_hi - 0.01, HIDDEN
        )
        # The grid is sized: the corner particles leave.
        self.drain.retire([count - 2, count - 1])
        self.batter_count = len(batter)
        # Vanilla and chocolate batter in alternate sectors, for the paddle to marble.
        sector = np.floor(np.arctan2(positions[:, 1], positions[:, 0]) / (0.25 * np.pi)).astype(int) % 2
        base = np.where(sector[:, None] == 0, VANILLA, CHOCOLATE)
        self.colors = np.clip(base * np.random.default_rng(5).uniform(0.95, 1.03, (count, 1)), 0.0, 1.0)

        # The paddle as MPM's collider turns about its shaft with its moment of inertia about it; the bowl is static.
        self.coupling = HingeCoupling(self.model, self.state_0, [(self.paddle, tuple(SHAFT), (0.0, 0.0, 1.0))])
        self.coupling.attach(self.batter_state)
        self.batter_state.body_f = wp.zeros_like(self.state_0.body_f)
        self.bowl_meshes = [box_mesh(*box) for box in self.bowl]  # the solver keeps the mesh ids only
        colliders = 1 + len(self.bowl_meshes)
        self.mpm.setup_collider(
            collider_meshes=[None, *self.bowl_meshes],
            collider_body_ids=[self.paddle] + [None] * len(self.bowl_meshes),
            collider_friction=[BATTER_FRICTION] * colliders,
            collider_projection_threshold=[0.25 * VOXEL] * colliders,
            model=self.model,
            **self.coupling.setup_collider_args(),
        )
        self.spin = []  # the paddle's angular velocity [rad/s] about +z at each frame

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.17, -0.22, 0.30), pitch=-45.2, yaw=127.7)

    def simulate(self):
        for _ in range(self.sim_substeps):
            # FBA: the paddle under the batter's impulse of the last step and the motor's torque.
            self.state_0.clear_forces()
            self.coupling.apply_impulses(self.state_0, self.sim_dt)
            wp.launch(motor_kernel, 1, inputs=[self.paddle, wp.vec3(0.0, 0.0, 1.0), self.torque, self.state_0.body_f])
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            # MPM: the batter against the paddle as it turned over the step.
            self.coupling.update_colliders(self.state_0)
            self.mpm.step(self.batter_state, self.batter_state, None, None, self.sim_dt)
            self.mpm.project_outside(self.batter_state, self.batter_state, self.sim_dt)
            self.coupling.collect_impulses(self.mpm, self.batter_state)
            self.drain.drain()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.spin.append(float(self.state_0.body_qd.numpy()[self.paddle][5]))

    def batter_spin(self):
        """The batter's mean angular velocity [rad/s] about the shaft."""
        q = self.batter_state.particle_q.numpy()[: self.batter_count]
        qd = self.batter_state.particle_qd.numpy()[: self.batter_count]
        r2 = np.maximum(q[:, 0] ** 2 + q[:, 1] ** 2, 1.0e-6)
        return float(np.mean((q[:, 0] * qd[:, 1] - q[:, 1] * qd[:, 0]) / r2))

    def test_post_step(self):
        q = self.state_0.body_q.numpy()[self.paddle]
        assert np.all(np.isfinite(q)), f"non-finite paddle pose at t={self.sim_time:.3f} s"
        offset = float(np.linalg.norm(q[:3] - SHAFT))
        assert offset < 1.0e-3, f"the paddle left its shaft by {offset:.4f} m at t={self.sim_time:.3f} s"

    def test_final(self):
        # The motor turns the paddle against the batter's drag, and the batter turns with it, in the bowl.
        spin = float(np.mean(self.spin[-self.fps // 2 :]))
        assert 2.0 < spin < 30.0, f"the paddle turns at {spin:.2f} rad/s over the last half second"
        assert self.batter_spin() > 1.0, f"the batter turns at only {self.batter_spin():.2f} rad/s"
        active = self.drain.active()
        assert active.sum() == self.batter_count, f"{self.batter_count - active.sum()} batter particles left the bowl"
        assert np.all(np.isfinite(self.batter_state.particle_q.numpy()[active])), "non-finite batter"

    def render(self):
        active = self.drain.active()
        points = self.batter_state.particle_q.numpy()[active]
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if len(points):
            self.viewer.log_points(
                "/batter", wp.array(points, dtype=wp.vec3), 0.3 * VOXEL, wp.array(self.colors[active], dtype=wp.vec3)
            )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--torque", type=float, default=0.25, help="The motor's torque [N m].")
    parser.add_argument("--viscosity", type=float, default=10.0, help="The batter's viscosity [Pa s].")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
