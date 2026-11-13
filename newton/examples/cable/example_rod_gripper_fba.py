# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rod Gripper FBA
#
# A rigid parallel-jaw gripper picks up a 1 m cotton rope from a table by its
# middle, lifts it, carries it aside and lays it down again. The rope is a
# particle rod of SolverFBA (SolverFBA.add_particle_rod); the gripper's pads
# and palm are kinematic bodies whose poses the example sets every step. Two
# ways to hold the rope, chosen with --grasp:
#
# - "friction": the pads squeeze the rope by 0.8 mm on each side, and their
#   contacts' friction alone carries it.
# - "pins": once the pads have closed, the rope's particles between them are
#   held at their places in the gripper's frame by particle constraint rows
#   (SolverFBA.set_particle_constraints), as the FBA cloth examples hold the
#   cloth they grasp, and let go when the pads open.
#
# Command: python -m newton.examples rod_gripper_fba --grasp friction
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

TABLE = 0.4  # height [m] of the table's top
# 10 mm cotton rope: 70 g/m, 2e4 N axial stiffness, 1.5e-3 N m^2 bending stiffness.
ROPE_RADIUS, ROPE_LENGTH, ROPE_LINE_DENSITY, ROPE_EA, ROPE_EI = 0.005, 1.0, 0.07, 2.0e4, 1.5e-3
PAD = (0.02, 0.006, 0.025)  # half extents [m] of a pad: along the rope, across it (closing), and up
SQUEEZE = 8.0e-4  # [m] the pads press into the rope on each side
OPEN = 0.03  # half gap [m] between the open pads' faces
# Key poses: time [s], the pads' center [m] and the half gap between their faces [m].
LOW = TABLE + 0.002 + PAD[2]
CLOSED = ROPE_RADIUS - SQUEEZE
KEYS = [
    (0.0, (0.0, 0.0, LOW + 0.18), OPEN),
    (1.0, (0.0, 0.0, LOW), OPEN),
    (1.6, (0.0, 0.0, LOW), CLOSED),
    (2.8, (0.0, 0.0, LOW + 0.32), CLOSED),
    (4.2, (0.27, 0.12, LOW + 0.32), CLOSED),
    (5.4, (0.27, 0.12, LOW + 0.002), CLOSED),
    (6.0, (0.27, 0.12, LOW + 0.002), OPEN),
    (7.0, (0.27, 0.12, LOW + 0.2), OPEN),
]
TUBE_SIDES = 12


def key_pose(t):
    """The pads' center and half gap at time ``t`` [s], eased between the key poses."""
    times = [k[0] for k in KEYS]
    if t >= times[-1]:
        return np.array(KEYS[-1][1]), KEYS[-1][2]
    i = int(np.searchsorted(times, t, side="right")) - 1
    (t0, c0, g0), (t1, c1, g1) = KEYS[i], KEYS[i + 1]
    s = (t - t0) / (t1 - t0)
    s = s * s * (3.0 - 2.0 * s)
    return (1.0 - s) * np.array(c0) + s * np.array(c1), (1.0 - s) * g0 + s * g1


@wp.func
def key_pose_device(t: float, key_t: wp.array[float], key_c: wp.array[wp.vec3], key_g: wp.array[float]):
    """:func:`key_pose` on the device."""
    n = key_t.shape[0]
    if t >= key_t[n - 1]:
        return key_c[n - 1], key_g[n - 1]
    i = int(0)
    for k in range(n - 1):
        if t >= key_t[k]:
            i = k
    s = (t - key_t[i]) / (key_t[i + 1] - key_t[i])
    s = s * s * (3.0 - 2.0 * s)
    return (1.0 - s) * key_c[i] + s * key_c[i + 1], (1.0 - s) * key_g[i] + s * key_g[i + 1]


@wp.kernel
def gripper_pose_kernel(
    t: float,
    dt: float,
    key_t: wp.array[float],
    key_c: wp.array[wp.vec3],
    key_g: wp.array[float],
    pads: wp.array[wp.int32],
    palm: int,
    pad_half_width: float,
    palm_height: float,
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Pose the pads and the palm at time ``t``, with the velocities of their motion over ``dt``."""
    center, gap = key_pose_device(t, key_t, key_c, key_g)
    ahead, gap_ahead = key_pose_device(t + dt, key_t, key_c, key_g)
    for k in range(2):
        side = 2.0 * float(k) - 1.0
        offset = wp.vec3(0.0, side * (gap + pad_half_width), 0.0)
        offset_ahead = wp.vec3(0.0, side * (gap_ahead + pad_half_width), 0.0)
        body_q[pads[k]] = wp.transform(center + offset, wp.quat_identity())
        body_qd[pads[k]] = wp.spatial_vector((ahead + offset_ahead - center - offset) / dt, wp.vec3(0.0))
    body_q[palm] = wp.transform(center + wp.vec3(0.0, 0.0, palm_height), wp.quat_identity())
    body_qd[palm] = wp.spatial_vector((ahead - center) / dt, wp.vec3(0.0))


@wp.kernel
def rope_tube_kernel(
    x: wp.array[wp.vec3],
    radius: float,
    sides: int,
    # outputs
    points: wp.array[wp.vec3],
):
    """The rings of a tube around the rope's particles, in a frame carried along it by parallel transport."""
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
            points[i * (sides + 1) + k] = x[i] + radius * (wp.cos(angle) * nrm + wp.sin(angle) * b)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.grasp = getattr(args, "grasp", "friction") if args is not None else "friction"
        self.frame_dt, self.sim_substeps = 1.0 / 50.0, 4  # steps of 5 ms
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.6
        builder.add_ground_plane()
        builder.add_shape_box(
            -1,
            xform=wp.transform((0.0, 0.0, TABLE - 0.02), wp.quat_identity()),
            hx=0.8,
            hy=0.45,
            hz=0.02,
            color=(0.55, 0.38, 0.22),
        )
        # The gripper: two rubber pads and a palm on kinematic bodies.
        pad_cfg = newton.ModelBuilder.ShapeConfig(mu=1.2)
        self.pads = []
        for _ in range(2):
            body = builder.add_body(
                xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0, is_kinematic=True
            )
            builder.add_shape_box(body, hx=PAD[0], hy=PAD[1], hz=PAD[2], cfg=pad_cfg, color=(0.15, 0.15, 0.17))
            self.pads.append(body)
        self.palm = builder.add_body(
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0, is_kinematic=True
        )
        palm_cfg = newton.ModelBuilder.ShapeConfig(has_particle_collision=False)
        builder.add_shape_box(self.palm, hx=0.03, hy=0.05, hz=0.012, cfg=palm_cfg, color=(0.7, 0.7, 0.72))
        # The rope lies on the table in a gentle S, across the pads' closing direction at its middle.
        s = np.linspace(-0.5, 0.5, 2001)
        curve = np.c_[0.95 * s, 0.08 * np.sin(2.0 * np.pi * s), np.full_like(s, TABLE + ROPE_RADIUS)]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        u = np.linspace(0.0, arc[-1], round(arc[-1] / (2.0 * ROPE_RADIUS)) + 1)
        points = np.stack([np.interp(u, arc, curve[:, k]) for k in range(3)], axis=1)
        area = np.pi * ROPE_RADIUS**2
        self.rope = SolverFBA.add_particle_rod(
            builder,
            points,
            ROPE_RADIUS,
            ROPE_EA / area,
            density=ROPE_LINE_DENSITY / area,
            bending_stiffness=ROPE_EI,
        )
        self.model = builder.finalize()
        self.model.particle_mu = 0.6

        # Ten LG iterations per 5 ms step, the rope's segments colliding as capsules, and light Rayleigh damping.
        # The contact solves keep their Delassus matrix in sparse storage, whose NSN step replays from step to step.
        self.solver = SolverFBA(
            self.model,
            iterations=10,
            nsn_linear_iterations=10,
            nsn_sparse_storage=True,
            rod_contact=True,
            rayleigh_alpha=0.5,
            rayleigh_beta=1.0e-3,
            rayleigh_corotated=True,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.01)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.held = None  # rope particles held by the "pins" grasp, and their places in the pads' frame
        # The key poses on the device, where the gripper is posed every step without a round trip through the host.
        self.key_t = wp.array([k[0] for k in KEYS], dtype=float)
        self.key_c = wp.array([k[1] for k in KEYS], dtype=wp.vec3)
        self.key_g = wp.array([k[2] for k in KEYS], dtype=float)
        self.pads_d = wp.array(self.pads, dtype=wp.int32)
        self.set_gripper(self.state_0, 0.0)
        self.set_gripper(self.state_1, 0.0)

        n, ring = len(self.rope), TUBE_SIDES + 1
        quads = []
        for i in range(n - 1):
            for k in range(TUBE_SIDES):
                a, b = i * ring + k, i * ring + k + 1
                quads += [a, b, b + ring, a, b + ring, a + ring]
        self.tube_points = wp.zeros(n * ring, dtype=wp.vec3)
        self.tube_indices = wp.array(np.asarray(quads, dtype=np.int32))
        self.rest_length = self.model.spring_rest_length.numpy()
        self.springs = self.model.spring_indices.numpy().reshape(-1, 2)
        self.max_strain = 0.0
        self.lifted = 0.0

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(0.15, -0.95, 0.8), pitch=-17.0, yaw=90.0)

    def set_gripper(self, state, t):
        """Pose the pads and the palm at time ``t``, with the velocities of their motion."""
        inputs = [float(t), float(self.sim_dt), self.key_t, self.key_c, self.key_g, self.pads_d, self.palm]
        inputs += [PAD[1], PAD[2] + 0.012]
        wp.launch(gripper_pose_kernel, 1, inputs, [state.body_q, state.body_qd])

    def update_pins(self, t):
        """The "pins" grasp: take the rope's particles between the pads once they have closed, hold them at their
        places in the pads' frame while they stay closed, and let them go as the pads open."""
        center, gap = key_pose(t)
        closed = gap <= CLOSED + 1.0e-4
        if closed and self.held is None:
            x = self.state_0.particle_q.numpy()[self.rope]
            d = x - center
            inside = (np.abs(d[:, 0]) < PAD[0]) & (np.abs(d[:, 1]) < gap + ROPE_RADIUS) & (np.abs(d[:, 2]) < PAD[2])
            taken = np.asarray(self.rope)[inside]
            self.held = (taken, d[inside]) if len(taken) else None
        elif not closed and self.held is not None:
            self.held = None
            self.solver.set_particle_constraints(None, None, None)
        if self.held is None:
            return
        taken, local = self.held
        targets = center + local
        rows = np.repeat(taken, 3)[:, None]
        directions = np.tile(np.eye(3), (len(taken), 1))
        self.solver.set_particle_constraints(rows, np.ones((len(rows), 1)), targets.reshape(-1), directions=directions)

    def simulate(self):
        # SolverFBA reads the contact count back to the host every step, so the loop is not graph-captured.
        for k in range(self.sim_substeps):
            t = self.sim_time + (k + 1) * self.sim_dt
            self.set_gripper(self.state_0, t - self.sim_dt)
            if self.grasp == "pins":
                self.update_pins(t)
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
        # The rope's stretch; between squeezing pads it also shortens a little, hidden there.
        self.max_strain = max(self.max_strain, float((length / self.rest_length - 1.0).max()))
        middle = x[self.rope[len(self.rope) // 2], 2]
        self.lifted = max(self.lifted, float(middle) - TABLE)

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert self.max_strain < 0.05, f"the rope stretched by up to {100 * self.max_strain:.2f}%"
        if self.sim_time > 3.0:
            assert self.lifted > 0.25, f"the gripper lifted the rope's middle by {self.lifted:.2f} m only"
        if self.sim_time > 6.5:
            middle = x[self.rope[len(self.rope) // 2]]
            np.testing.assert_allclose(middle[:2], KEYS[-1][1][:2], atol=0.05)
            assert middle[2] < TABLE + 0.03, "the rope did not come back down to the table"

    def render(self):
        wp.launch(
            rope_tube_kernel,
            dim=1,
            inputs=[self.state_0.particle_q, ROPE_RADIUS, TUBE_SIDES],
            outputs=[self.tube_points],
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh("/rope", self.tube_points, self.tube_indices, color=(0.86, 0.82, 0.72), roughness=0.95)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--grasp", choices=("friction", "pins"), default="friction", help="How the gripper holds the rope."
    )
    parser.set_defaults(num_frames=350)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
