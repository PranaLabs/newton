# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Elastic Cords FBA
#
# Three cords hang from a bar, each held out level with a ball at its end
# and let go: an 8 mm bungee shock cord with a 0.5 kg iron ball, a 5 mm
# elastic cord with a 0.3 kg ball, and, for comparison, a 6 mm polyester
# rope with a 0.5 kg ball. The balls swing down; at the bottom of their
# swing the elastic cords stretch by about a fifth and bounce their balls
# along them, their rubber damping the bounces, while the rope holds its
# length. Each cord is a particle rod of SolverFBA
# (SolverFBA.add_particle_rod) with the axial and bending stiffness and the
# weight of the real cord, its ball a heavy particle on a short stiff hook,
# and the cords collide with themselves as capsules (rod_contact=True).
#
# Command: python -m newton.examples elastic_cords_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

BAR = (1.5, 0.015)  # height and radius [m] of the bar along x
CORD_LENGTH = 0.5  # [m]
# Cords: radius [m], line density [kg/m], axial stiffness EA [N], bending stiffness EI [N m^2], Rayleigh stiffness
# damping beta [s] of their stretch and bending rates, color, and the ball's mass [kg] and radius [m].
CORDS = [
    {
        "name": "bungee",
        "radius": 0.004,
        "line_density": 0.045,
        "axial": 80.0,
        "bending": 5.0e-5,
        "beta": 5.0e-3,
        "color": (0.05, 0.05, 0.06),
        "tracer": (0.85, 0.85, 0.82),
        "ball": (0.5, 0.025),
    },
    {
        "name": "elastic",
        "radius": 0.0025,
        "line_density": 0.02,
        "axial": 40.0,
        "bending": 1.5e-5,
        "beta": 5.0e-3,
        "color": (0.75, 0.1, 0.08),
        "tracer": (0.08, 0.08, 0.09),
        "ball": (0.3, 0.021),
    },
    {
        "name": "rope",
        "radius": 0.003,
        "line_density": 0.025,
        "axial": 3.0e4,
        "bending": 2.0e-4,
        "beta": 2.0e-4,
        "color": (0.9, 0.9, 0.86),
        "tracer": (0.15, 0.3, 0.75),
        "ball": (0.5, 0.025),
    },
]
CORD_SPACING = 0.28  # [m] along the bar
HOOK = 0.012  # length [m] of the hook between a cord's end and its ball's surface
HOOK_STIFFNESS = 2.0e5  # [N/m]
TUBE_SIDES = 12


@wp.kernel
def cord_tube_kernel(
    x: wp.array[wp.vec3],
    particles: wp.array[wp.int32],
    radius: float,
    sides: int,
    # outputs
    points: wp.array[wp.vec3],
):
    """The rings of a tube around a cord's particles, in a frame carried along the cord by parallel transport, each
    ring thinned as the rubber under it stretches (constant volume) and ``sides + 1`` points around."""
    n = particles.shape[0]
    t = wp.normalize(x[particles[1]] - x[particles[0]])
    ref = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(wp.dot(ref, t)) > 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    nrm = wp.normalize(wp.cross(t, ref))
    for i in range(n):
        p = x[particles[i]]
        t_new = wp.normalize(x[particles[wp.min(i + 1, n - 1)]] - x[particles[wp.max(i - 1, 0)]])
        axis = wp.cross(t, t_new)
        s = wp.length(axis)
        if s > 1.0e-8:
            nrm = wp.quat_rotate(wp.quat_from_axis_angle(axis / s, wp.atan2(s, wp.dot(t, t_new))), nrm)
        t = t_new
        nrm = wp.normalize(nrm - wp.dot(nrm, t) * t)
        b = wp.cross(t, nrm)
        for k in range(sides + 1):
            angle = 2.0 * wp.pi * float(k) / float(sides)
            points[i * (sides + 1) + k] = p + radius * (wp.cos(angle) * nrm + wp.sin(angle) * b)


def braid_texture(color, tracer, width=64, height=128, seed=1):
    """A braided sheath over one texture height along a cord and its circumference around: two sets of helical
    strands crossing over and under each other in ``color``, with two tracer strands in ``tracer``."""
    rng = np.random.default_rng(seed)
    u = np.arange(width)[None, :] / width
    v = np.arange(height)[:, None] / height
    carriers = 8
    a = (carriers * (u + v)) % 1.0  # strands winding one way
    b = (carriers * (u - v)) % 1.0  # and the other
    # Over-under: the strand on top alternates at each crossing.
    top_a = ((np.floor(carriers * (u + v)) + np.floor(carriers * (u - v))) % 2) == 0
    ridge_a = np.sin(np.pi * a) ** 0.6
    ridge_b = np.sin(np.pi * b) ** 0.6
    shade = np.where(top_a, 0.55 + 0.45 * ridge_a, 0.55 + 0.45 * ridge_b)
    shade *= 0.92 + 0.08 * rng.random((height, width))
    rgb = np.asarray(color)[None, None, :] * shade[..., None]
    strand = np.floor(carriers * (u + v)).astype(int) % carriers
    tracing = top_a & ((strand == 0) | (strand == carriers // 2))
    rgb = np.where(tracing[..., None], np.asarray(tracer)[None, None, :] * shade[..., None], rgb)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def icosphere(subdivisions=3):
    """Unit sphere vertices and triangles."""
    t = (1.0 + 5.0**0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]  # fmt: skip
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]  # fmt: skip
    vertices = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(subdivisions):
        cache, faces = {}, []

        def middle(a, b, cache=cache):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = vertices[a] + vertices[b]
                vertices.append(m / np.linalg.norm(m))
                cache[key] = len(vertices) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = faces
    return np.array(vertices), np.array(f, dtype=np.int32).reshape(-1)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        # Steps of 2 ms: a cord's penalty, 60 m / dt^2 for its particles' mass m, must hold the tension of its ball
        # at the bottom of the swing across its segments, which 10 ms steps leave too soft.
        self.frame_dt, self.sim_substeps = 1.0 / 50.0, 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        along_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform((0.0, 0.0, BAR[0]), along_x),
            radius=BAR[1],
            half_height=0.5,
            color=(0.7, 0.72, 0.75),
        )
        self.cords, self.balls, beta = [], [], []
        for k, c in enumerate(CORDS):
            x = (k - 1) * CORD_SPACING
            r = c["radius"]
            # The cord starts level from the bar's underside, its ball at its end.
            z = BAR[0] - BAR[1] - r
            y = np.linspace(0.0, CORD_LENGTH, round(CORD_LENGTH / (2.0 * r)) + 1)
            points = np.c_[np.full_like(y, x), y, np.full_like(y, z)]
            area = np.pi * r * r
            cord = SolverFBA.add_particle_rod(
                builder,
                points,
                r,
                c["axial"] / area,
                density=c["line_density"] / area,
                bending_stiffness=c["bending"],
            )
            builder.particle_mass[cord[0]] = 0.0
            mass, ball_radius = c["ball"]
            # The ball sits just beyond the cord's end, the hook between them.
            ball_center = points[-1] + (0.0, HOOK + ball_radius, 0.0)
            ball = builder.add_particle(pos=tuple(ball_center), vel=(0.0, 0.0, 0.0), mass=mass, radius=ball_radius)
            builder.add_spring(cord[-1], ball, ke=HOOK_STIFFNESS, kd=0.0, control=0.0)
            self.cords.append(cord)
            self.balls.append(ball)
            beta += [c["beta"]] * (len(cord) + 1)
        self.model = builder.finalize()
        self.model.particle_mu = 0.5

        # Ten LG iterations per step, the cords' segments colliding as capsules, and Rayleigh damping:
        # mass-proportional 0.05/s, and corotated stiffness damping per cord, the rubber's and the rope's, of the
        # cords' stretch and bending rates.
        # The contact solves keep their Delassus matrix in sparse storage, whose NSN step replays from step to step.
        self.solver = SolverFBA(
            self.model,
            iterations=10,
            nsn_linear_iterations=10,
            nsn_sparse_storage=True,
            rod_contact=True,
            rayleigh_alpha=0.05,
            rayleigh_beta=np.asarray(beta),
            rayleigh_corotated=True,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.02)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()

        # Rendering: a braided tube per cord, its texture stretching with the cord, and a sphere per ball.
        self.tubes = []
        for cord, ball, c in zip(self.cords, self.balls, CORDS, strict=True):
            # The tube runs on into its ball, which hides the hook.
            tube = [*cord, ball]
            n, ring = len(tube), TUBE_SIDES + 1
            quads = []
            for i in range(n - 1):
                for k in range(TUBE_SIDES):
                    a, b = i * ring + k, i * ring + k + 1
                    quads += [a, b, b + ring, a, b + ring, a + ring]
            # One texture height per four circumferences of rest length.
            x0 = self.model.particle_q.numpy()[tube]
            rest = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(x0, axis=0), axis=1))]
            v = rest / (4.0 * 2.0 * np.pi * c["radius"])
            uv = np.stack(np.meshgrid(np.arange(ring) / TUBE_SIDES, v), axis=-1).reshape(-1, 2)
            self.tubes.append(
                (
                    wp.array(np.asarray(tube, dtype=np.int32)),
                    wp.zeros(n * ring, dtype=wp.vec3),
                    wp.array(np.asarray(quads, dtype=np.int32)),
                    c,
                    wp.array(uv.astype(np.float32), dtype=wp.vec2),
                    braid_texture(c["color"], c["tracer"]),
                )
            )
        self.sphere_vertices, sphere_triangles = icosphere()
        self.sphere_triangles = wp.array(sphere_triangles)
        self.rest_length = self.model.spring_rest_length.numpy()
        self.springs = self.model.spring_indices.numpy().reshape(-1, 2)
        self.peak_stretch = np.zeros(len(CORDS))

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(-1.3, -0.55, 1.2), pitch=-2.0, yaw=22.0)

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

    def cord_lengths(self, x):
        return [np.linalg.norm(np.diff(x[cord], axis=0), axis=1).sum() for cord in self.cords]

    def test_post_step(self):
        x, v = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        assert np.isfinite(x).all(), f"non-finite positions at t={self.sim_time:.2f} s"
        speed = float(np.linalg.norm(v, axis=1).max())
        assert speed < 15.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        stretch = np.array(self.cord_lengths(x)) / CORD_LENGTH - 1.0
        self.peak_stretch = np.maximum(self.peak_stretch, stretch)

    def test_final(self):
        bungee, elastic, rope = self.peak_stretch
        if self.sim_time > 1.0:
            assert 0.1 < bungee < 0.6, f"the bungee cord stretched by {100 * bungee:.1f}%"
            assert 0.1 < elastic < 0.6, f"the elastic cord stretched by {100 * elastic:.1f}%"
        assert rope < 0.01, f"the rope stretched by {100 * rope:.1f}%"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        x = self.state_0.particle_q
        for k, (particles, points, quads, c, uvs, texture) in enumerate(self.tubes):
            wp.launch(cord_tube_kernel, dim=1, inputs=[x, particles, c["radius"], TUBE_SIDES], outputs=[points])
            self.viewer.log_mesh(f"/cords/{k}", points, quads, uvs=uvs, texture=texture, roughness=0.8)
        xh = x.numpy()
        for k, (ball, c) in enumerate(zip(self.balls, CORDS, strict=True)):
            world = xh[ball] + c["ball"][1] * self.sphere_vertices
            self.viewer.log_mesh(
                f"/balls/{k}",
                wp.array(world.astype(np.float32), dtype=wp.vec3),
                self.sphere_triangles,
                color=(0.18, 0.18, 0.2),
                roughness=0.35,
                metallic=0.9,
            )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
