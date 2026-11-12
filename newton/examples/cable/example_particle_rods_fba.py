# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Particle Rods FBA
#
# Elastic rods made of particles (SolverFBA.add_particle_rod): springs along
# a polyline that SolverFBA steps to their implicit Euler balance, and
# bending elements at its vertices. Three ropes, pinned at one end and of
# rising bending stiffness, swing down from the horizontal over a bar; a
# rod clamped in a block sags and swings; and a coil of rope drops onto the
# ground. The rods collide with the shapes through their particles, of the
# rods' radius, with each other as capsules, and render as tubes.
#
# Command: python -m newton.examples particle_rods_fba
###########################################################################

import itertools

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

RADIUS, DENSITY = 0.004, 1100.0  # radius [m] and density [kg/m^3] of the ropes
ROPE_YOUNG = 2.0e7  # the ropes' Young's modulus [Pa], for their axial stiffness
ROPE_BENDING = (2.0e-5, 2.0e-4, None)  # bending stiffness EI [N m^2]: a rope, a cord, and a wire (E pi r^4 / 4)
ROPE_LENGTH, ROPE_SEGMENTS = 1.0, 80
ROPE_PIN, ROPE_GAP = (-0.75, 1.35), 0.22  # the ropes' pinned ends (x, z) [m] and their spacing along y
BAR = (-0.15, 0.95, 0.03)  # the bar along y: x, z and radius [m]
ROD = (0.65, 0.5, 0.006, 1.0e9, 30)  # clamped rod: length [m], height [m], radius [m], Young's modulus [Pa], segments
COIL = (0.45, 0.45, 0.1, 3.0, 0.75, 120)  # coil: center x and y [m], radius [m], turns, height [m], segments
BLOCK = (1.0, -0.4)  # the block that clamps the rod: x and y [m]
TUBE_SIDES = 12


@wp.kernel
def rod_tube_kernel(
    x: wp.array[wp.vec3],
    rod_start: wp.array[wp.int32],
    rod_radius: wp.array[wp.float32],
    sides: int,
    # outputs
    points: wp.array[wp.vec3],
):
    """The rings of a tube around each rod, one thread per rod: a ring of ``sides`` points about each particle, in a
    frame carried along the rod by parallel transport so that the rings do not twist."""
    r = wp.tid()
    start = rod_start[r]
    end = rod_start[r + 1]
    radius = rod_radius[r]
    t = wp.normalize(x[start + 1] - x[start])
    ref = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(wp.dot(ref, t)) > 0.9:
        ref = wp.vec3(1.0, 0.0, 0.0)
    n = wp.normalize(wp.cross(t, ref))
    for i in range(start, end):
        i_prev = wp.max(i - 1, start)
        i_next = wp.min(i + 1, end - 1)
        t_new = wp.normalize(x[i_next] - x[i_prev])
        # Turn the normal by the rotation that takes the previous tangent to this one.
        axis = wp.cross(t, t_new)
        s = wp.length(axis)
        if s > 1.0e-8:
            n = wp.quat_rotate(wp.quat_from_axis_angle(axis / s, wp.atan2(s, wp.dot(t, t_new))), n)
        t = t_new
        n = wp.normalize(n - wp.dot(n, t) * t)
        b = wp.cross(t, n)
        for k in range(sides):
            angle = 2.0 * wp.pi * float(k) / float(sides)
            points[i * sides + k] = x[i] + radius * (wp.cos(angle) * n + wp.sin(angle) * b)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt, self.sim_substeps = 1.0 / 50.0, 2  # steps of 10 ms
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        builder.add_ground_plane()
        # The bar, a cylinder along y, and the block that clamps the rod.
        along_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * np.pi)
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform((BAR[0], 0.0, BAR[1]), along_y),
            radius=BAR[2],
            half_height=0.5,
            color=(0.6, 0.6, 0.6),
        )
        length, height, rod_radius, rod_young, rod_segments = ROD
        # The rod's clamped particles sit inside the block, which the rods pass through.
        block = newton.ModelBuilder.ShapeConfig(has_particle_collision=False)
        builder.add_shape_box(
            -1,
            xform=wp.transform((BLOCK[0], BLOCK[1], height), wp.quat_identity()),
            hx=0.05,
            hy=0.05,
            hz=0.08,
            cfg=block,
            color=(0.5, 0.5, 0.5),
        )

        self.rods, rod_radii = [], []
        # Three ropes, pinned at their first particle, laid out horizontally along x.
        for k, bending in enumerate(ROPE_BENDING):
            y = (k - 1) * ROPE_GAP
            x = np.linspace(ROPE_PIN[0], ROPE_PIN[0] + ROPE_LENGTH, ROPE_SEGMENTS + 1)
            points = np.c_[x, np.full_like(x, y), np.full_like(x, ROPE_PIN[1])]
            rope = SolverFBA.add_particle_rod(
                builder, points, RADIUS, ROPE_YOUNG, density=DENSITY, bending_stiffness=bending
            )
            builder.particle_mass[rope[0]] = 0.0
            self.rods.append(rope)
            rod_radii.append(RADIUS)
        # A stiff rod clamped by its first two particles in the block, along -x.
        x = np.linspace(BLOCK[0] - 0.01, BLOCK[0] - 0.01 - length, rod_segments + 1)
        points = np.c_[x, np.full_like(x, BLOCK[1]), np.full_like(x, height)]
        rod = SolverFBA.add_particle_rod(builder, points, rod_radius, rod_young, density=1200.0)
        builder.particle_mass[rod[0]] = 0.0
        builder.particle_mass[rod[1]] = 0.0
        self.rods.append(rod)
        rod_radii.append(rod_radius)
        # A coil of rope above the ground.
        cx, cy, coil_radius, turns, coil_height, coil_segments = COIL
        angle = np.linspace(0.0, 2.0 * np.pi * turns, coil_segments + 1)
        points = np.c_[
            cx + coil_radius * np.cos(angle),
            cy + coil_radius * np.sin(angle),
            coil_height + 0.02 * angle / (2.0 * np.pi),
        ]
        self.rods.append(
            SolverFBA.add_particle_rod(builder, points, RADIUS, ROPE_YOUNG, density=DENSITY, bending_stiffness=2.0e-4)
        )
        rod_radii.append(RADIUS)
        self.model = builder.finalize()

        # Ten LG iterations per 10 ms step, the rods' ADMM springs starting each step from the previous step's duals,
        # at most ten conjugate residual iterations per contact solve, the rods colliding with each other as capsules,
        # and light Rayleigh damping: mass-proportional
        # 0.5/s, and corotated stiffness damping of 2 ms on the rods' stretch and bending rates, which leaves their
        # swing alone.
        self.solver = SolverFBA(
            self.model,
            iterations=10,
            nsn_linear_iterations=10,
            rayleigh_alpha=0.5,
            rayleigh_beta=2.0e-3,
            rayleigh_corotated=True,
            rod_contact=True,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.02)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()

        # The tubes: rings of TUBE_SIDES points about each particle, quads between consecutive rings.
        starts = np.cumsum([0] + [len(r) for r in self.rods]).astype(np.int32)
        self.rod_start = wp.array(starts, dtype=wp.int32)
        self.rod_radius = wp.array(np.asarray(rod_radii, dtype=np.float32))
        self.tube_points = wp.zeros(self.model.particle_count * TUBE_SIDES, dtype=wp.vec3)
        quads = []
        for r0, r1 in itertools.pairwise(starts):
            for i in range(r0, r1 - 1):
                for k in range(TUBE_SIDES):
                    a, b = i * TUBE_SIDES + k, i * TUBE_SIDES + (k + 1) % TUBE_SIDES
                    c, d = a + TUBE_SIDES, b + TUBE_SIDES
                    quads += [a, b, d, a, d, c]
        self.tube_indices = wp.array(np.asarray(quads, dtype=np.int32))
        self.rest_length = self.model.spring_rest_length.numpy()
        self.springs = self.model.spring_indices.numpy().reshape(-1, 2)
        self.max_strain = 0.0

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(-0.65, -1.7, 1.25), pitch=-17.0, yaw=64.0)

    def simulate(self):
        # SolverFBA reads the contact count back to the host every step, so the loop is not graph-captured.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        x, v = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        assert np.isfinite(x).all(), f"non-finite positions at t={self.sim_time:.2f} s"
        speed = float(np.linalg.norm(v, axis=1).max())
        assert speed < 15.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        length = np.linalg.norm(x[self.springs[:, 1]] - x[self.springs[:, 0]], axis=1)
        self.max_strain = max(self.max_strain, float((length / self.rest_length - 1.0).max()))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        # The rods stretch by a few percent for a frame where they strike the bar or the ground, and settle back.
        assert self.max_strain < 0.08, f"the rods stretched by up to {100 * self.max_strain:.2f}%"
        length = np.linalg.norm(x[self.springs[:, 1]] - x[self.springs[:, 0]], axis=1)
        strain = float(np.abs(length / self.rest_length - 1.0).max())
        assert strain < 0.005, f"the rods are stretched by {100 * strain:.2f}% at t={self.sim_time:.2f} s"
        for rope in self.rods[:3]:
            np.testing.assert_allclose(x[rope[0]], self.model.particle_q.numpy()[rope[0]], atol=1.0e-4)
            assert x[rope[-1], 2] < ROPE_PIN[1] - 0.3, "a rope did not swing down"
        rod = self.rods[3]
        sag = ROD[1] - x[rod[-1], 2]
        assert 0.0 < sag < 0.1, f"the clamped rod's tip sags by {sag:.3f} m"
        if self.sim_time > 2.0:
            coil = self.rods[4]
            assert x[coil, 2].max() < 0.1, "the coil did not land"
            assert x[coil, 2].min() > 0.5 * RADIUS, f"the coil sank into the ground to z = {x[coil, 2].min():.4f} m"

    def render(self):
        wp.launch(
            rod_tube_kernel,
            dim=len(self.rods),
            inputs=[self.state_0.particle_q, self.rod_start, self.rod_radius, TUBE_SIDES],
            outputs=[self.tube_points],
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh("/rods", self.tube_points, self.tube_indices, color=(0.85, 0.45, 0.2))
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
