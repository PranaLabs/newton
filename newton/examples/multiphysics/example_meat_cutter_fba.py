# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Meat Cutter FBA
#
# A lever cutter cuts a slab of pork belly in two. The blade turns on a
# hinge at the back of a board, a body that SolverFBA integrates on a
# revolute joint, pressed down by a hand: a constant torque, less the
# joint's damping, so that the push falls as the blade speeds up. Once the
# blade is on the board, the hand lifts it back. The pork belly is
# SolverImplicitMPM: layers of lean, fat and skin, each an elastic solid
# that yields past its own strength, which the blade's wedge presses down
# and parts.
#
# The two solvers exchange the meat's impulse on the blade once per FBA
# step, MPM seeing the blade as a collider that turns about the hinge with
# its moment of inertia about it (fba_mpm_coupling.HingeCoupling).
#
# Command: python -m newton.examples meat_cutter_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.fba_mpm_coupling import HingeCoupling, ParticleDrain
from newton.solvers import SolverFBA, SolverImplicitMPM

HINGE = np.array([-0.10, 0.0, 0.008])  # [m], the blade turns about y through this point
BLADE_LENGTH = 0.30  # [m], from the hinge to the handle
BLADE_DROP = 0.008  # [m], the blade's edge below the hinge's axis
BLADE_HEIGHT = 0.05  # [m]
BLADE_THICKNESS = 0.005  # [m]
BEVEL = 0.012  # [m], the height of the wedge that the edge is ground to
HANDLE_MASS = 0.5  # [kg]
LIFT = 10.0  # [N m], the hand's torque that lifts the blade back
START_ANGLE = np.radians(-24.0)  # the blade raised, its edge just above the meat
STEEL_DENSITY = 7800.0  # [kg/m^3]
MEAT_DENSITY = 1050.0  # [kg/m^3]
MEAT_LO = np.array([0.02, -0.04, 0.0])  # [m], the slab's corners
MEAT_HI = np.array([0.14, 0.04, 0.045])
# The layers from the board up: (top [m], kind), kinds 0 lean, 1 fat, 2 skin.
LAYERS = [(0.013, 0), (0.020, 1), (0.031, 0), (0.040, 1), (0.045, 2)]
# Per kind: Young's modulus [Pa], Poisson's ratio, yield stress [Pa], yield pressure [Pa], tensile yield ratio,
# friction. The yield stresses are low, for a blade whose edge the grid blunts to about a voxel's width, and the
# meat has little strength in tension, so that the cut stays open: MPM would weld two faces back where they touch.
MATERIALS = [
    (1.0e5, 0.40, 5.0e3, 3.0e5, 0.01, 0.3),  # lean
    (4.0e4, 0.45, 2.0e3, 1.0e5, 0.01, 0.2),  # fat
    (3.0e5, 0.40, 1.0e4, 5.0e5, 0.01, 0.3),  # skin
]
COLORS = np.array([(0.80, 0.32, 0.32), (0.96, 0.92, 0.86), (0.90, 0.74, 0.58)])  # raw pork: lean, fat, skin
VOXEL = 0.004  # [m], the meat's grid
STEEL = (0.62, 0.64, 0.68)
WOOD = (0.62, 0.45, 0.28)
HANDLE = (0.12, 0.10, 0.09)
HIDDEN = (0.0, 0.0, -10.0)  # where the meat particles go once they leave the grid


@wp.kernel
def push_kernel(body: int, axis: wp.vec3, torque: float, body_f: wp.array[wp.spatial_vector]):
    """Add the hand's torque about the hinge to the blade's forces."""
    body_f[body] = body_f[body] + wp.spatial_vector(wp.vec3(0.0), axis * torque)


def meat_points(spacing, rng):
    """Points ``spacing`` apart, slightly jittered, that fill the slab, and the kind of meat at each: the layers'
    boundaries wave a little."""
    counts = np.ceil((MEAT_HI - MEAT_LO) / spacing).astype(int)
    grid = np.stack(np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"), axis=-1).reshape(-1, 3)
    p = MEAT_LO + (grid + 0.5) * spacing
    kind = np.full(len(p), LAYERS[-1][1])
    for k in reversed(range(len(LAYERS) - 1)):
        top, layer = LAYERS[k]
        phase = rng.uniform(0.0, 2.0 * np.pi, 2)
        wave = (
            0.0015 * np.sin(2.0 * np.pi * p[:, 0] / 0.045 + phase[0]) * np.cos(2.0 * np.pi * p[:, 1] / 0.07 + phase[1])
        )
        kind[p[:, 2] < top + wave] = layer
    return p + rng.uniform(-0.15, 0.15, p.shape) * spacing, kind


def blade_mesh():
    """The blade in its body frame, from the hinge along +x: a plate whose lower edge is ground to a wedge."""
    half_t = 0.5 * BLADE_THICKNESS
    top, bevel, edge = BLADE_HEIGHT - BLADE_DROP, BEVEL - BLADE_DROP, -BLADE_DROP
    section = [(half_t, top), (-half_t, top), (-half_t, bevel), (-0.0003, edge), (0.0003, edge), (half_t, bevel)]
    points = np.array([(x, y, z) for x in (0.0, BLADE_LENGTH) for y, z in section], dtype=np.float32)
    cloud = newton.Mesh(points, [0, 1, 2], compute_inertia=False, maxhullvert=len(points))
    hull = cloud.compute_convex_hull()
    return newton.Mesh(hull.vertices, hull.indices, maxhullvert=len(points))


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
        self.release = args.release  # [s], when the hand starts to push
        self.push = args.push  # [N m], the hand's torque about the hinge, down
        self.phase = "wait"  # then "press" until the blade is through, "lift" back to its start, and "rest"
        self.down_time = None  # [s], when the blade reached the board

        # The rigid side: the board, and the blade on its hinge with a handle at its end.
        builder = newton.ModelBuilder()
        board = newton.ModelBuilder.ShapeConfig(mu=0.3, density=0.0, gap=0.002, has_particle_collision=False)
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        steel = newton.ModelBuilder.ShapeConfig(
            mu=0.3, density=STEEL_DENSITY, has_shape_collision=False, has_particle_collision=True
        )
        builder.add_ground_plane(height=-0.03, cfg=visual, color=(0.16, 0.17, 0.15))
        self.board_center, self.board_half = (0.06, 0.0, -0.015), (0.24, 0.10, 0.015)
        builder.add_shape_box(-1, xform=wp.transform(self.board_center, wp.quat_identity()), hx=self.board_half[0],
                              hy=self.board_half[1], hz=self.board_half[2], cfg=board, color=WOOD)  # fmt: skip
        along_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)  # a cylinder's axis (z) onto y
        for side in (-1.0, 1.0):  # the hinge's brackets
            bracket = wp.transform((HINGE[0], side * 0.012, 0.5 * HINGE[2] + 0.004), wp.quat_identity())
            builder.add_shape_box(-1, xform=bracket, hx=0.012, hy=0.004, hz=0.5 * HINGE[2] + 0.012, cfg=visual,
                                  color=STEEL)  # fmt: skip
        start = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(START_ANGLE))
        self.blade = builder.add_link(xform=wp.transform(tuple(HINGE), start))
        # The blade: a closed mesh that the meat and the viewer see, which gives its mass, and its convex hull for
        # its contacts with the board.
        hull = blade_mesh()
        builder.add_shape_mesh(self.blade, mesh=newton.Mesh(hull.vertices, hull.indices), cfg=steel, color=STEEL)
        hidden = newton.ModelBuilder.ShapeConfig(
            mu=0.3, density=0.0, gap=0.002, has_particle_collision=False, is_visible=False
        )
        builder.add_shape_convex_hull(self.blade, mesh=hull, cfg=hidden)
        handle_radius, handle_half = 0.013, 0.05
        handle_density = HANDLE_MASS / (np.pi * handle_radius**2 * 2.0 * handle_half)
        handle = newton.ModelBuilder.ShapeConfig(density=handle_density, has_shape_collision=False,
                                                 has_particle_collision=False)  # fmt: skip
        grip = wp.transform((BLADE_LENGTH, 0.0, BLADE_HEIGHT - BLADE_DROP - 0.01), along_y)
        builder.add_shape_cylinder(self.blade, xform=grip, radius=handle_radius, half_height=handle_half, cfg=handle,
                                   color=HANDLE)  # fmt: skip
        joint = builder.add_joint_revolute(
            -1, self.blade, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform(tuple(HINGE), wp.quat_identity()),
            damping=args.damping,
        )  # fmt: skip
        builder.joint_q[builder.joint_q_start[joint]] = float(START_ANGLE)
        builder.add_articulation([joint])
        self.model = builder.finalize()
        self.solver = SolverFBA(self.model, iterations=5)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()

        # The meat side: the slab, two particles per voxel edge, in a grid around it, sized by two particles at its
        # corners that the solver's setup sees.
        spacing = 0.5 * VOXEL
        meat, kind = meat_points(spacing, np.random.default_rng(2))
        self.domain_lo = np.array([-0.04, -0.12, -0.03])
        self.domain_hi = np.array([0.20, 0.12, 0.10])
        positions = np.concatenate([meat, [self.domain_lo, self.domain_hi]])
        kind = np.concatenate([kind, [0, 0]])
        count = len(positions)
        meat_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(meat_builder)
        meat_builder.add_particles(
            pos=[tuple(p) for p in positions],
            vel=[(0.0, 0.0, 0.0)] * count,
            mass=[MEAT_DENSITY * spacing**3] * count,
            radius=[0.5 * spacing] * count,
            flags=[int(newton.ParticleFlags.ACTIVE)] * count,
        )
        self.meat_model = meat_builder.finalize()
        # Each layer an elastic solid that flows past its yield stress and parts past its tensile strength.
        table = np.array(MATERIALS, dtype=np.float32)[kind]
        mpm = self.meat_model.mpm
        for column, name in enumerate(
            ("young_modulus", "poisson_ratio", "yield_stress", "yield_pressure", "tensile_yield_ratio", "friction")
        ):
            getattr(mpm, name).assign(table[:, column])
        config = SolverImplicitMPM.Config()
        config.voxel_size = VOXEL
        config.grid_type = "fixed"
        config.grid_padding = 2
        config.max_active_cell_count = 1 << 16
        config.max_iterations = 60
        config.tolerance = 1.0e-4
        config.strain_basis = "P1d"  # the default P0 locks a nearly incompressible solid
        config.velocity_basis = "Q1"
        config.collider_basis = "S2"
        self.mpm = SolverImplicitMPM(self.meat_model, config=config)
        self.meat_state = self.meat_model.state()
        self.meat_state.mpm.particle_Jp.fill_(1.0)
        self.drain = ParticleDrain(
            self.mpm, self.meat_model, self.meat_state, self.domain_lo + 0.008, self.domain_hi - 0.008, HIDDEN
        )
        self.drain.retire([count - 2, count - 1])
        self.meat = meat
        self.across = self.across_cut()  # the meat across the cut's plane before the cut
        self.colors = np.clip(COLORS[kind] * np.random.default_rng(3).uniform(0.94, 1.04, (count, 1)), 0.0, 1.0)

        # The blade as MPM's collider turns about the hinge with its moment of inertia about it; the board is
        # a static collider.
        self.coupling = HingeCoupling(self.model, self.state_0, [(self.blade, tuple(HINGE), (0.0, 1.0, 0.0))])
        self.coupling.attach(self.meat_state)
        self.meat_state.body_f = wp.zeros_like(self.state_0.body_f)
        self.board_mesh = box_mesh(self.board_center, self.board_half)  # the solver keeps the mesh id only
        self.mpm.setup_collider(
            collider_meshes=[None, self.board_mesh],
            collider_body_ids=[self.blade, None],
            collider_friction=[0.2, 0.6],
            collider_projection_threshold=[0.25 * VOXEL] * 2,
            model=self.model,
            **self.coupling.setup_collider_args(),
        )
        self.swing = []  # the blade's angle [rad] at each frame, negative above the board

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.22, -0.24, 0.16), pitch=-24.0, yaw=122.0)

    def simulate(self):
        torque = self.push if self.phase == "press" else -LIFT
        for _ in range(self.sim_substeps):
            if self.phase in ("press", "lift"):
                # FBA: the blade under the meat's impulse of the last step and the hand, against the board.
                self.state_0.clear_forces()
                self.coupling.apply_impulses(self.state_0, self.sim_dt)
                wp.launch(push_kernel, 1, inputs=[self.blade, wp.vec3(0.0, 1.0, 0.0), torque, self.state_0.body_f])
                self.pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            else:
                self.coupling.hold(self.state_0)  # the blade waits, or rests after the cut
            # MPM: the meat against the blade as it turned over the step.
            self.coupling.update_colliders(self.state_0)
            self.mpm.step(self.meat_state, self.meat_state, None, None, self.sim_dt)
            self.mpm.project_outside(self.meat_state, self.meat_state, self.sim_dt)
            self.coupling.collect_impulses(self.mpm, self.meat_state)
            self.drain.drain()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        q = self.state_0.body_q.numpy()[self.blade]
        self.swing.append(float(2.0 * np.arctan2(q[4], q[6])))
        # The hand: it pushes from the release until the blade is on the board (or for at most 4 s), then lifts the
        # blade back to where it started and lets go of it there.
        if self.phase == "wait" and self.sim_time >= self.release:
            self.phase = "press"
        elif self.phase == "press" and (self.swing[-1] > -0.01 or self.sim_time > self.release + 4.0):
            self.phase = "lift"
            self.down_time = self.sim_time
        elif self.phase == "lift" and self.swing[-1] < START_ANGLE:
            self.phase = "rest"

    def test_post_step(self):
        q = self.state_0.body_q.numpy()[self.blade]
        assert np.all(np.isfinite(q)), f"non-finite blade pose at t={self.sim_time:.3f} s"
        offset = float(np.linalg.norm(q[:3] - HINGE))
        assert offset < 1.0e-3, f"the blade left its hinge by {offset:.4f} m at t={self.sim_time:.3f} s"

    def across_cut(self):
        """How many particles of the slab's middle lie within 1.5 mm of the cut's plane, above the board's 5 mm."""
        q = self.meat_state.particle_q.numpy()[: len(self.meat)]
        middle = (q[:, 0] > MEAT_LO[0] + 0.01) & (q[:, 0] < MEAT_HI[0] - 0.01) & (q[:, 2] > 0.005)
        return int((middle & (np.abs(q[:, 1]) < 0.0015)).sum())

    def test_final(self):
        # The blade went down to the board and came back up, and the cut stays open.
        assert self.down_time is not None, f"the blade stopped at {np.degrees(self.swing[-1]):.1f} deg"
        assert self.phase in ("lift", "rest"), f"the hand is still in its {self.phase} phase"
        across = self.across_cut()
        assert across < 0.1 * self.across, f"{across} of {self.across} particles still lie across the cut"
        q = self.meat_state.particle_q.numpy()[self.drain.active()]
        assert np.all(np.isfinite(q)), "non-finite meat"

    def render(self):
        active = self.drain.active()
        points = self.meat_state.particle_q.numpy()[active]
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if len(points):
            self.viewer.log_points(
                "/meat", wp.array(points, dtype=wp.vec3), 0.3 * VOXEL, wp.array(self.colors[active], dtype=wp.vec3)
            )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--push", type=float, default=10.0, help="The hand's torque down about the hinge [N m].")
    parser.add_argument("--damping", type=float, default=20.0, help="The hinge's damping [N m s/rad].")
    parser.add_argument("--release", type=float, default=0.3, help="When the hand starts to push [s].")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
