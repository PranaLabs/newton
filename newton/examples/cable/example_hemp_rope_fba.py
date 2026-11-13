# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Hemp Rope FBA
#
# A 3 m hemp rope, 12 mm three-strand, is laid down in a coil: a hand holds
# its upper end, lowers it toward the floor and circles, and the rope piles
# onto itself. The rope is a particle rod of SolverFBA
# (SolverFBA.add_particle_rod) with the stiffness, weight and bending
# stiffness of a natural fiber rope; its segments collide with each other as
# capsules (rod_contact=True) and with the floor through their particles. It
# renders as a three-strand rope.
#
# Command: python -m newton.examples hemp_rope_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

# 12 mm three-strand hemp (manila) rope: about 100 g/m, under 1% stretch at its working load, and a bending
# stiffness that lets 17 cm of it stand out from a hand before it bends down under its own weight.
ROPE_RADIUS = 0.006  # [m]
ROPE_LENGTH = 3.0  # [m]
ROPE_DENSITY = 884.0  # [kg/m^3], 0.1 kg/m
ROPE_YOUNG = 1.0e9  # [Pa], axial
ROPE_BENDING = 5.0e-3  # [N m^2]
ROPE_MU = 0.7  # rope on rope and on the floor
# The hand: it starts above the rope's hanging length, lowers at LOWER_SPEED and circles at CIRCLE_RADIUS once per
# CIRCLE_PERIOD, until it is HOLD_HEIGHT above the floor.
HAND_START = 3.08  # [m]
LOWER_SPEED = 0.42  # [m/s]
CIRCLE_RADIUS, CIRCLE_PERIOD = 0.09, 1.1  # [m], [s]
HOLD_HEIGHT = 0.2  # [m]
STRANDS, LAY_LENGTH = 3, 0.042  # strands of the rope and the length of one turn of their helix [m]
TUBE_SIDES = 18


@wp.kernel
def rope_tube_kernel(
    x: wp.array[wp.vec3],
    radius: float,
    sides: int,
    strands: int,
    lay: wp.array[wp.float32],
    # outputs
    points: wp.array[wp.vec3],
):
    """The rings of a three-strand rope around the particles, one thread for the rope: a ring of ``sides + 1``
    points about each particle (the last repeats the first for the texture seam), in a frame carried along the rope
    by parallel transport so that the rings do not twist. Each strand bulges out of the rope's circle, and the
    strands turn about the axis by ``lay`` [rad] at each particle."""
    n = x.shape[0]
    t = wp.normalize(x[1] - x[0])
    ref = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(wp.dot(ref, t)) > 0.9:
        ref = wp.vec3(1.0, 0.0, 0.0)
    nrm = wp.normalize(wp.cross(t, ref))
    for i in range(n):
        t_new = wp.normalize(x[wp.min(i + 1, n - 1)] - x[wp.max(i - 1, 0)])
        axis = wp.cross(t, t_new)
        s = wp.length(axis)
        if s > 1.0e-8:
            nrm = wp.quat_rotate(wp.quat_from_axis_angle(axis / s, wp.atan2(s, wp.dot(t, t_new))), nrm)
        t = t_new
        nrm = wp.normalize(nrm - wp.dot(nrm, t) * t)
        b = wp.cross(t, nrm)
        for k in range(sides + 1):
            angle = 2.0 * wp.pi * float(k) / float(sides)
            # Three lobes: the strands' cross-sections, a groove between each two.
            phase = float(strands) * (angle - lay[i]) * 0.5
            r = radius * (0.84 + 0.16 * wp.abs(wp.cos(phase)))
            points[i * (sides + 1) + k] = x[i] + r * (wp.cos(angle) * nrm + wp.sin(angle) * b)


def hemp_texture(width=64, height=256, seed=0):
    """A hemp rope's color over one lay length, ``height`` texels along and ``width`` around: tan fibers with dark
    grooves between the strands, which run diagonally across the texture."""
    rng = np.random.default_rng(seed)
    u = np.arange(width)[None, :] / width
    v = np.arange(height)[:, None] / height
    # Position across a strand, 0 and 1 at the grooves: the strands advance one turn per lay length.
    across = (STRANDS * (u - v)) % 1.0
    groove = 1.0 - np.exp(-((np.minimum(across, 1.0 - across) / 0.07) ** 2))
    # Fibers: streaks along the rope with slowly varying shade.
    fibers = np.cumsum(rng.normal(0.0, 1.0, size=(height, width)), axis=0) * 0.02
    fibers += rng.normal(0.0, 0.25, size=(height, width))
    shade = 0.62 + 0.38 * groove + 0.05 * np.tanh(fibers)
    rgb = np.clip(np.array([0.72, 0.58, 0.38]) * shade[..., None], 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt, self.sim_substeps = 1.0 / 50.0, 2  # steps of 10 ms
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = ROPE_MU
        builder.add_ground_plane()
        # The rope hangs straight down from the hand, its end just above the floor.
        segments = round(ROPE_LENGTH / (2.0 * ROPE_RADIUS))
        z = np.linspace(HAND_START, HAND_START - ROPE_LENGTH, segments + 1)
        points = np.c_[np.full_like(z, CIRCLE_RADIUS), np.zeros_like(z), z]
        self.rope = SolverFBA.add_particle_rod(
            builder, points, ROPE_RADIUS, ROPE_YOUNG, density=ROPE_DENSITY, bending_stiffness=ROPE_BENDING
        )
        builder.particle_mass[self.rope[0]] = 0.0
        self.model = builder.finalize()
        self.model.particle_mu = ROPE_MU

        # Ten LG iterations per 10 ms step, at most ten conjugate residual iterations per contact solve, the rope's
        # segments colliding as capsules, and light Rayleigh damping: mass-proportional 0.2/s, and corotated
        # stiffness damping of 1 ms that damps the rope's stretch and bending rates but not its swing.
        self.solver = SolverFBA(
            self.model,
            iterations=10,
            nsn_linear_iterations=10,
            rod_contact=True,
            rayleigh_alpha=0.2,
            rayleigh_beta=1.0e-3,
            rayleigh_corotated=True,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.01)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pin_targets = self.model.particle_q.numpy()

        # The rope's tube: rings of TUBE_SIDES + 1 points, quads between consecutive rings, and texture
        # coordinates: around the rope, and along it one texture height per lay length.
        n, ring = len(self.rope), TUBE_SIDES + 1
        rest = np.r_[0.0, np.cumsum(np.full(n - 1, ROPE_LENGTH / segments))]
        self.lay = wp.array((2.0 * np.pi * rest / LAY_LENGTH).astype(np.float32))
        self.tube_points = wp.zeros(n * ring, dtype=wp.vec3)
        quads = []
        for i in range(n - 1):
            for k in range(TUBE_SIDES):
                a, b = i * ring + k, i * ring + k + 1
                quads += [a, b, b + ring, a, b + ring, a + ring]
        self.tube_indices = wp.array(np.asarray(quads, dtype=np.int32))
        uv = np.stack(np.meshgrid(np.arange(ring) / TUBE_SIDES, rest / LAY_LENGTH), axis=-1).reshape(-1, 2)
        self.tube_uvs = wp.array(uv.astype(np.float32), dtype=wp.vec2)
        self.texture = hemp_texture()
        self.rest_length = self.model.spring_rest_length.numpy()
        self.springs = self.model.spring_indices.numpy().reshape(-1, 2)
        self.max_strain = 0.0

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(-0.75, -0.95, 0.75), pitch=-26.0, yaw=52.0)

    def hand(self, t):
        """The hand's position at time ``t`` [s]."""
        z = max(HAND_START - LOWER_SPEED * t, HOLD_HEIGHT)
        phase = 2.0 * np.pi * t / CIRCLE_PERIOD
        return np.array([CIRCLE_RADIUS * np.cos(phase), CIRCLE_RADIUS * np.sin(phase), z])

    def simulate(self):
        # SolverFBA reads the contact count back to the host every step, so the loop is not graph-captured.
        for k in range(self.sim_substeps):
            self.pin_targets[self.rope[0]] = self.hand(self.sim_time + (k + 1) * self.sim_dt)
            self.solver.set_pin_targets(self.pin_targets)
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
        assert speed < 10.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        length = np.linalg.norm(x[self.springs[:, 1]] - x[self.springs[:, 0]], axis=1)
        self.max_strain = max(self.max_strain, float((length / self.rest_length - 1.0).max()))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert self.max_strain < 0.01, f"the rope stretched by up to {100 * self.max_strain:.2f}%"
        assert x[:, 2].min() > 0.5 * ROPE_RADIUS, f"the rope sank into the floor to z = {x[:, 2].min():.4f} m"
        if self.sim_time > 6.0:
            lying = x[x[:, 2] < 6.0 * ROPE_RADIUS]
            assert len(lying) > 0.6 * len(x), "the rope did not pile up on the floor"
            spread = np.linalg.norm(lying[:, :2] - lying[:, :2].mean(axis=0), axis=1).max()
            assert spread < 0.5, f"the coil spread over {spread:.2f} m"

    def render(self):
        wp.launch(
            rope_tube_kernel,
            dim=1,
            inputs=[self.state_0.particle_q, ROPE_RADIUS, TUBE_SIDES, STRANDS, self.lay],
            outputs=[self.tube_points],
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/rope", self.tube_points, self.tube_indices, uvs=self.tube_uvs, texture=self.texture, roughness=0.9
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=400)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
