# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Shishi-odoshi FBA
#
# A shishi-odoshi: water from a spout fills the mouth of a bamboo tube on a
# hinge until the water's weight tips it; the tube pours the water into a
# basin, swings back and knocks its tail on a stone, and fills again. The
# water is SolverImplicitMPM; the tube is a body that SolverFBA integrates,
# its hinge a revolute joint, resting on the stone and the basin's rim
# through convex contacts. The two exchange impulses once per FBA step:
#
#   1. The FBA step applies the water's impulse of the previous step as a
#      constant force on the tube.
#   2. The MPM step sees the tube as a collider that turns about the hinge
#      with the tube's moment of inertia, at its pose at the start of the
#      step and with FBA's end velocity less the response to that impulse,
#      so that MPM solves the water and the tube's response together. The
#      collider impulses of the MPM step drive the next FBA step.
#
# The tube starts nearly full, so that it first pours after about two seconds.
#
# Command: python -m newton.examples shishi_odoshi_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics import shishi_odoshi_fba_utils as geo
from newton.examples.multiphysics.fba_mpm_coupling import HingeCoupling
from newton.solvers import SolverFBA, SolverImplicitMPM

WATER_DENSITY = 1000.0
BAMBOO = (0.80, 0.66, 0.42)
STONE = (0.55, 0.47, 0.36)
GRANITE = (0.34, 0.34, 0.36)
WOOD = (0.36, 0.28, 0.20)
WATER = (0.25, 0.52, 0.92)
HIDDEN = (0.0, 0.0, -10.0)  # where the water particles wait before they leave the spout, and after they drain


@wp.kernel
def escaping_kernel(
    q: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    lo: wp.vec3,
    hi: wp.vec3,
    # outputs
    count: wp.array[int],
):
    """Count the active water particles outside the box ``[lo, hi]`` or no longer finite."""
    i = wp.tid()
    if (flags[i] & newton.ParticleFlags.ACTIVE) == 0:
        return
    p = q[i]
    inside = p[0] > lo[0] and p[1] > lo[1] and p[2] > lo[2] and p[0] < hi[0] and p[1] < hi[1] and p[2] < hi[2]
    if not inside or not (wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])):
        wp.atomic_add(count, 0, 1)


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
        self.sim_substeps = 32  # FBA steps per frame, each with one MPM step
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # The rigid side: the tube on its hinge, resting on the stone and stopped by the basin's back rim.
        builder = newton.ModelBuilder()
        contact = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0, gap=0.005, has_particle_collision=False)
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        builder.add_ground_plane(height=geo.GROUND_Z, cfg=contact, color=(0.16, 0.17, 0.15))
        boxes = geo.basin_boxes()
        for center, half in boxes:
            builder.add_shape_box(-1, xform=wp.transform(center, wp.quat_identity()), hx=half[0], hy=half[1], hz=half[2],
                                  cfg=contact, color=GRANITE)  # fmt: skip
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
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # The water side: particles 2.5 mm apart, two per 5 mm voxel edge, filling the mouth to below the brim, and
        # the spout's stream of 17 mL/s, its particles waiting until they leave the pipe.
        voxel = 0.005
        spacing = 0.5 * voxel
        prefill = geo.prefill_points(spacing, args.fill_level, np.random.default_rng(0))
        stream, stream_time, self.stream_velocity = geo.stream_layout(spacing, 17.0e-6, args.water_seconds, 3)
        # A fixed grid around the water's reach, sized by two particles at its corners that the solver's setup sees.
        self.domain_lo = np.array([-0.60, -0.20, geo.GROUND_Z - 0.02])
        self.domain_hi = np.array([0.10, 0.20, 0.22])
        positions = np.concatenate([prefill, np.tile(HIDDEN, (len(stream), 1)), [self.domain_lo, self.domain_hi]])
        count = len(positions)
        active = int(newton.ParticleFlags.ACTIVE)
        water_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(water_builder)
        water_builder.add_particles(
            pos=[tuple(p) for p in positions],
            vel=[(0.0, 0.0, 0.0)] * count,
            mass=[WATER_DENSITY * spacing**3] * count,
            radius=[0.5 * spacing] * count,
            flags=[active] * len(prefill) + [0] * len(stream) + [active] * 2,
        )
        self.water_model = water_builder.finalize()
        # Water: no friction and no yield stress (no shear strength), and pressure that holds in tension as in
        # compression.
        self.water_model.mpm.friction.fill_(0.0)
        self.water_model.mpm.tensile_yield_ratio.fill_(1.0)
        self.water_model.mpm.viscosity.fill_(0.0)
        config = SolverImplicitMPM.Config()
        config.voxel_size = voxel
        config.grid_type = "fixed"
        config.grid_padding = 2
        config.max_active_cell_count = 1 << 14
        config.max_iterations = 50
        config.tolerance = 1.0e-4
        # Piecewise-linear strain: with the default piecewise-constant one, a pool at rest compresses and the tube
        # loses a quarter of its water's torque within 1.5 s, so it would never tip.
        config.strain_basis = "P1d"
        config.velocity_basis = "Q1"
        config.collider_basis = "S2"
        self.mpm = SolverImplicitMPM(self.water_model, config=config)

        self.water_state = self.water_model.state()
        self.water_state.mpm.particle_Jp.fill_(1.0)
        # The emission schedule on the host: each stream particle's time and place at the pipe's end.
        self.emit_time = np.concatenate([-np.ones(len(prefill)), stream_time, [-1.0, -1.0]])
        self.emit_position = np.concatenate([prefill, stream, [HIDDEN, HIDDEN]])
        self.particle_q = self.water_state.particle_q.numpy()
        self.particle_qd = self.water_state.particle_qd.numpy()
        self.particle_flags = self.water_model.particle_flags.numpy()
        # The grid is sized: the corner particles leave.
        self.particle_flags[-2:] = 0
        self.particle_q[-2:] = HIDDEN
        self.write_particles()

        # The tube as MPM's collider: it turns about the hinge with the tube's moment of inertia about it. The ground,
        # the basin and the stone are static colliders, each its own mesh, so that their signed distances combine as
        # a union.
        self.coupling = HingeCoupling(self.model, self.state_0, [(self.tube, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))])
        self.coupling.attach(self.water_state)
        self.water_state.body_f = wp.zeros_like(self.state_0.body_f)
        static = [((0.0, 0.0, geo.GROUND_Z - 0.05), (2.0, 2.0, 0.05)), *boxes]
        meshes = []
        for center, half in static:
            box = newton.Mesh.create_box(*half, duplicate_vertices=False, compute_normals=False, compute_uvs=False,
                                         compute_inertia=False)  # fmt: skip
            meshes.append((box.vertices + np.asarray(center, dtype=np.float32), box.indices))
        meshes.append((stone.vertices + stone_center.astype(np.float32), stone.indices))
        static_meshes = []
        for mesh_points, indices in meshes:
            points = wp.array(np.asarray(mesh_points, dtype=np.float32), dtype=wp.vec3)
            static_meshes.append(wp.Mesh(points, wp.array(np.ravel(indices), dtype=int), wp.zeros_like(points)))
        self.static_meshes = static_meshes  # the solver keeps the mesh ids only
        self.mpm.setup_collider(
            collider_meshes=[None, *static_meshes],
            collider_body_ids=[self.tube] + [None] * len(static_meshes),
            collider_friction=[0.0] * (1 + len(static_meshes)),
            collider_projection_threshold=[0.25 * voxel] * (1 + len(static_meshes)),
            model=self.model,
            **self.coupling.setup_collider_args(),
        )

        # Water leaves the scene 1 cm inside the grid's box, before it leaves the grid: a particle beyond the grid
        # would make the MPM step's collider impulses non-finite.
        self.drain_lo, self.drain_hi = self.domain_lo + 0.01, self.domain_hi - 0.01
        self.escaping = wp.zeros(1, dtype=int)
        self.lowest_angle = float(geo.REST)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.03, -0.91, 0.42), pitch=-30.0, yaw=90.0)

    def write_particles(self):
        """Hand the host's particle positions, velocities and flags to the water and let the solver see the flags."""
        self.water_state.particle_q.assign(self.particle_q)
        self.water_state.particle_qd.assign(self.particle_qd)
        self.water_model.particle_flags.assign(self.particle_flags)
        self.mpm.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

    def emit(self):
        """Let the stream particles due by the frame's end leave the pipe, placed and moving where their flight from
        the pipe's end puts them at the frame's start."""
        due = (self.emit_time >= 0.0) & (self.emit_time < self.sim_time + self.frame_dt)
        if not due.any():
            return
        self.particle_q = self.water_state.particle_q.numpy()
        self.particle_qd = self.water_state.particle_qd.numpy()
        tau = (self.sim_time - self.emit_time[due])[:, None]
        gravity = np.array([0.0, 0.0, -9.81])
        self.particle_q[due] = self.emit_position[due] + self.stream_velocity * tau + 0.5 * tau * tau * gravity
        self.particle_qd[due] = self.stream_velocity + tau * gravity
        self.particle_flags[due] |= int(newton.ParticleFlags.ACTIVE)
        self.emit_time[due] = -1.0
        self.write_particles()

    def drain(self):
        """Take out the water that left the drain box (or is no longer finite), once a step finds some."""
        self.escaping.zero_()
        inputs = [self.water_state.particle_q, self.water_model.particle_flags, wp.vec3(*self.drain_lo)]
        inputs.append(wp.vec3(*self.drain_hi))
        wp.launch(escaping_kernel, self.water_model.particle_count, inputs=inputs, outputs=[self.escaping])
        if not self.escaping.numpy()[0]:
            return
        self.particle_q = self.water_state.particle_q.numpy()
        self.particle_qd = self.water_state.particle_qd.numpy()
        q = self.particle_q
        inside = np.all((q > self.drain_lo) & (q < self.drain_hi), axis=1) & np.all(np.isfinite(q), axis=1)
        drained = ((self.particle_flags & int(newton.ParticleFlags.ACTIVE)) != 0) & ~inside
        self.particle_q[drained] = HIDDEN
        self.particle_qd[drained] = 0.0
        self.particle_flags[drained] &= ~int(newton.ParticleFlags.ACTIVE)
        self.write_particles()

    def simulate(self):
        for _ in range(self.sim_substeps):
            # FBA: the tube under the water's impulse of the last step.
            self.state_0.clear_forces()
            self.coupling.apply_impulses(self.state_0, self.sim_dt)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            # MPM: the water against the tube as it turned over the step.
            self.coupling.update_colliders(self.state_0)
            self.mpm.step(self.water_state, self.water_state, None, None, self.sim_dt)
            self.mpm.project_outside(self.water_state, self.water_state, self.sim_dt)
            self.coupling.collect_impulses(self.mpm, self.water_state)
            self.drain()

    def step(self):
        self.emit()
        self.simulate()
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
        active = (self.particle_flags & int(newton.ParticleFlags.ACTIVE)) != 0
        assert np.all(np.isfinite(q[active])), "non-finite water"
        x0, x1, y0, y1 = geo.BASIN_INNER
        in_basin = (q[:, 0] > x0) & (q[:, 0] < x1) & (q[:, 1] > y0) & (q[:, 1] < y1) & (q[:, 2] < -0.15) & active
        assert in_basin.sum() > 1000, f"only {in_basin.sum()} water particles reached the basin"

    def render(self):
        active = (self.particle_flags & int(newton.ParticleFlags.ACTIVE)) != 0
        points = self.water_state.particle_q.numpy()[active]
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
    parser.add_argument("--water-seconds", type=float, default=40.0, help="Seconds of water the spout gives.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
