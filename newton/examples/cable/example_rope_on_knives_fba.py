# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rope on Knives FBA
#
# A 0.71 m braided rope, 8 mm thick, drops onto five steel blades 1 mm
# thick. The rope is a particle rod of SolverFBA (SolverFBA.add_particle_rod)
# whose particles lie 12.5 mm apart, so a blade fits between the spheres of
# two of them. With rod_shape_contacts=True its segments also collide with
# the blades as capsules, and the rope comes to rest across their edges,
# sagging between them. With --no-segment-contacts it collides through its
# particles alone, and the blades pass between them: the rope falls through
# to the floor. It renders as a tube.
#
# Command: python -m newton.examples rope_on_knives_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

# 8 mm braided polyester rope: 40 g/m, stiff along its length and soft in bending.
ROPE_RADIUS = 0.004  # [m]
ROPE_DENSITY = 800.0  # [kg/m^3]
ROPE_YOUNG = 1.0e9  # [Pa], axial
ROPE_BENDING = 2.0e-4  # [N m^2]
ROPE_MU = 0.3  # rope on steel and on the floor
SEGMENT = 0.0125  # [m], 4.5 mm between the spheres of two neighboring particles
SEGMENTS = 57  # an odd count puts the middle of a segment at x = 0 and every 10 cm from it
# Five blades 1 mm thick, 12 cm wide and 20 cm tall, 10 cm apart across the rope, each under the middle of a segment.
BLADE_X = (-0.2, -0.1, 0.0, 0.1, 0.2)
BLADE_HALF = (0.0005, 0.06, 0.1)  # [m]
BLADE_TOP = 2.0 * BLADE_HALF[2]
DROP = 0.04  # [m], the rope's fall onto the blade edges
TUBE_SIDES = 16


@wp.kernel
def rope_tube_kernel(
    x: wp.array[wp.vec3],
    radius: float,
    sides: int,
    # outputs
    points: wp.array[wp.vec3],
):
    """The rings of a tube around the rope, one thread for the rope: a ring of ``sides`` points about each particle,
    in a frame carried along the rope by parallel transport so that the rings do not twist."""
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
        for k in range(sides):
            angle = 2.0 * wp.pi * float(k) / float(sides)
            points[i * sides + k] = x[i] + radius * (wp.cos(angle) * nrm + wp.sin(angle) * b)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt, self.sim_substeps = 1.0 / 50.0, 2  # steps of 10 ms
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.segment_contacts = not getattr(args, "no_segment_contacts", False)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = ROPE_MU
        builder.add_ground_plane()
        for x in BLADE_X:
            builder.add_shape_box(
                -1,
                xform=wp.transform((x, 0.0, BLADE_HALF[2]), wp.quat_identity()),
                hx=BLADE_HALF[0],
                hy=BLADE_HALF[1],
                hz=BLADE_HALF[2],
                color=(0.62, 0.65, 0.7),
            )
        # The rope lies straight across the blades, DROP above their edges.
        x = (np.arange(SEGMENTS + 1) - 0.5 * SEGMENTS) * SEGMENT
        points = np.c_[x, np.zeros_like(x), np.full_like(x, BLADE_TOP + ROPE_RADIUS + DROP)]
        self.rope = SolverFBA.add_particle_rod(
            builder, points, ROPE_RADIUS, ROPE_YOUNG, density=ROPE_DENSITY, bending_stiffness=ROPE_BENDING
        )
        self.model = builder.finalize()
        self.model.particle_mu = ROPE_MU

        # Ten LG iterations per 10 ms step, at most ten conjugate residual iterations per contact solve, and light
        # Rayleigh damping: mass-proportional 0.2/s, and corotated stiffness damping of 1 ms that damps the rope's
        # stretch and bending rates but not its swing. The segments find their contacts with the blades 1 cm out,
        # beyond the 9 mm that the rope falls in its last step before it lands. The contact solves keep their
        # Delassus matrix in sparse storage, whose NSN step replays from step to step.
        self.solver = SolverFBA(
            self.model,
            iterations=10,
            nsn_linear_iterations=10,
            nsn_sparse_storage=True,
            rod_shape_contacts=self.segment_contacts,
            rod_shape_contact_gap=0.01,
            rayleigh_alpha=0.2,
            rayleigh_beta=1.0e-3,
            rayleigh_corotated=True,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.01)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()

        # The rope's tube: rings of TUBE_SIDES points about each particle, quads between consecutive rings.
        n = len(self.rope)
        self.tube_points = wp.zeros(n * TUBE_SIDES, dtype=wp.vec3)
        quads = []
        for i in range(n - 1):
            for k in range(TUBE_SIDES):
                a, b = i * TUBE_SIDES + k, i * TUBE_SIDES + (k + 1) % TUBE_SIDES
                quads += [a, b, b + TUBE_SIDES, a, b + TUBE_SIDES, a + TUBE_SIDES]
        self.tube_indices = wp.array(np.asarray(quads, dtype=np.int32))

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(-0.3, -0.52, 0.34), pitch=-20.0, yaw=60.0)

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

    def rope_height_at(self, x_blade):
        """Height [m] of the rope's axis where it crosses the plane of a blade at ``x_blade``."""
        x = self.state_0.particle_q.numpy()[self.rope]
        i = int(np.flatnonzero((x[:-1, 0] - x_blade) * (x[1:, 0] - x_blade) <= 0.0)[0])
        w = (x_blade - x[i, 0]) / (x[i + 1, 0] - x[i, 0])
        return float((1.0 - w) * x[i, 2] + w * x[i + 1, 2])

    def test_post_step(self):
        x, v = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        assert np.isfinite(x).all(), f"non-finite positions at t={self.sim_time:.2f} s"
        # The rope lands on the blades at 0.9 m/s, and on the floor at 2.2 m/s when it falls through them.
        speed = float(np.linalg.norm(v, axis=1).max())
        assert speed < 3.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"

    def test_final(self):
        if not self.segment_contacts:
            return
        # The rope rests on every blade edge, its surface within 1 mm of the edge, and hangs clear of the floor.
        for x_blade in BLADE_X:
            z = self.rope_height_at(x_blade)
            assert z > BLADE_TOP + ROPE_RADIUS - 0.001, f"the blade at x={x_blade} cut into the rope to z={z:.4f} m"
        x = self.state_0.particle_q.numpy()
        assert x[:, 2].min() > ROPE_RADIUS, "the rope reached the floor"
        if self.sim_time > 2.0:
            speed = float(np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max())
            assert speed < 0.05, f"the rope still moves at {speed:.3f} m/s"

    def render(self):
        wp.launch(
            rope_tube_kernel,
            dim=1,
            inputs=[self.state_0.particle_q, ROPE_RADIUS, TUBE_SIDES],
            outputs=[self.tube_points],
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh("/rope", self.tube_points, self.tube_indices, color=(0.85, 0.45, 0.2), roughness=0.9)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--no-segment-contacts",
        action="store_true",
        help="Collide the rope with the blades through its particles only (rod_shape_contacts=False).",
    )
    parser.set_defaults(num_frames=200)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
