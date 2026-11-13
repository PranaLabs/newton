# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Desk Cables FBA
#
# Three common cables at a laptop on a desk: an HDMI cable, a USB-C cable
# and a power cord, each a particle rod of SolverFBA
# (SolverFBA.add_particle_rod) with the diameter, weight and bending
# stiffness of the real cable, stiff and heavier at its plugs. The HDMI
# cable runs from the laptop over the desk's front edge to a hand, which
# lifts it out past the edge, carries it across the desk over the USB-C
# cable and lays it down; the power cord hangs over the back edge to the
# floor. The cables collide with each other as capsules (rod_contact=True)
# and with the desk, the laptop and the floor through their particles.
#
# Command: python -m newton.examples desk_cables_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

DESK_TOP, DESK_HALF = 0.75, (0.6, 0.35)  # height [m] of the desk's top and its half extents along x and y [m]
LAPTOP = ((0.05, 0.12), (0.155, 0.11), 0.018)  # the laptop base's center (x, y), half extents and thickness [m]
PORT_HEIGHT = DESK_TOP + 0.009
# Cables: radius [m], line density [kg/m], Young's modulus [Pa] for the axial stiffness, bending stiffness
# [N m^2], color, and plug: length [m], mass [kg] and half width [m] of its housing.
CABLES = {
    "hdmi": {
        "radius": 0.00365,
        "line_density": 0.06,
        "young": 1.5e8,
        "bending": 6.0e-3,
        "color": (0.06, 0.06, 0.065),
        "plug": (0.04, 0.016, 0.011),
    },
    "usb_c": {
        "radius": 0.002,
        "line_density": 0.025,
        "young": 1.5e8,
        "bending": 3.0e-4,
        "color": (0.92, 0.92, 0.9),
        "plug": (0.028, 0.006, 0.0065),
    },
    "power": {
        "radius": 0.00325,
        "line_density": 0.055,
        "young": 1.2e8,
        "bending": 2.0e-3,
        "color": (0.92, 0.92, 0.9),
        "plug": (0.045, 0.03, 0.017),
    },
}
# The metal parts at a plug's outer end: (half length along the plug, half width, half height, offset across) of
# each box beyond the housing's face [m]: an HDMI shell, a USB-C shell, and a power plug's two round pins.
PLUG_METAL = {
    "hdmi": [(0.0055, 0.007, 0.00225, 0.0)],
    "usb_c": [(0.0033, 0.0042, 0.00125, 0.0)],
    "power": [(0.0095, 0.002, 0.002, -0.0095), (0.0095, 0.002, 0.002, 0.0095)],
}
PLUG_BENDING_SCALE = 200.0  # a plug's housing bends this many times as stiffly as its cable
RELIEF_LENGTH, RELIEF_SCALE = 0.025, 6.0  # the strain relief behind each plug: its length [m] and stiffening
MU_DESK, MU_CABLE = 0.4, 0.35  # PVC jackets on wood, and on each other
# The hand that holds the HDMI cable's plug: waypoints (time [s], position [m]) between which it moves smoothly.
HAND = [
    (0.0, (0.26, -0.37, 0.42)),
    (1.0, (0.26, -0.37, 0.42)),
    (2.2, (0.24, -0.55, 0.9)),
    (3.2, (0.1, -0.28, 1.02)),
    (4.6, (-0.32, -0.18, 1.0)),
    (6.0, (-0.38, -0.2, 0.79)),
    (8.0, (-0.38, -0.2, 0.79)),
]
TUBE_SIDES = 12


def smooth_path(waypoints, spacing):
    """Points every ``spacing`` [m] along a Catmull-Rom curve through ``waypoints``."""
    p = np.asarray(waypoints, dtype=np.float64)
    p = np.concatenate([2.0 * p[:1] - p[1:2], p, 2.0 * p[-1:] - p[-2:-1]])
    t = np.linspace(0.0, 1.0, 64, endpoint=False)[:, None]
    dense = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        dense.append(
            0.5
            * (
                2.0 * p1
                + (p2 - p0) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
                + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t**3
            )
        )
    dense = np.concatenate([*dense, p[-2:-1]])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    s = np.linspace(0.0, arc[-1], max(2, round(arc[-1] / spacing) + 1))
    return np.stack([np.interp(s, arc, dense[:, k]) for k in range(3)], axis=1)


def cable_paths():
    """The cables' center lines, from the plug at the laptop (or the floor) to the free end."""
    r = {name: c["radius"] for name, c in CABLES.items()}
    (lx, ly), (hx, _), _ = LAPTOP
    front, back = -DESK_HALF[1], DESK_HALF[1]
    on_desk = {name: DESK_TOP + rad for name, rad in r.items()}
    hdmi = [
        (lx + hx + r["hdmi"] + 0.001, ly, PORT_HEIGHT),
        (lx + hx + 0.06, ly - 0.01, on_desk["hdmi"]),
        (lx + hx + 0.16, ly - 0.09, on_desk["hdmi"]),
        (0.39, -0.12, on_desk["hdmi"]),
        (0.31, -0.27, on_desk["hdmi"]),
        (0.27, front + 0.01, on_desk["hdmi"]),
        (0.265, front - 0.012, DESK_TOP - 0.02),
        (0.262, front - 0.016, 0.62),
        HAND[0][1],
    ]
    usb_c = [
        (lx - hx - r["usb_c"] - 0.001, ly - 0.02, PORT_HEIGHT),
        (lx - hx - 0.06, ly - 0.03, on_desk["usb_c"]),
        (-0.28, 0.0, on_desk["usb_c"]),
        (-0.3, -0.13, on_desk["usb_c"]),
        (-0.17, -0.2, on_desk["usb_c"]),
        (-0.02, -0.13, on_desk["usb_c"]),
        (0.07, -0.22, on_desk["usb_c"]),
    ]
    power = [
        (lx - hx - r["power"] - 0.001, ly + 0.07, PORT_HEIGHT),
        (lx - hx - 0.06, ly + 0.08, on_desk["power"]),
        (-0.22, back - 0.06, on_desk["power"]),
        (-0.24, back - 0.005, on_desk["power"]),
        (-0.245, back + 0.012, DESK_TOP - 0.02),
        (-0.25, back + 0.016, 0.4),
        (-0.25, back + 0.03, 0.06),
        (-0.24, back + 0.12, r["power"]),
        (-0.1, back + 0.32, r["power"]),
    ]
    return {"hdmi": hdmi, "usb_c": usb_c, "power": power}


@wp.kernel
def cable_tube_kernel(
    x: wp.array[wp.vec3],
    starts: wp.array[wp.int32],
    radius: wp.array[wp.float32],
    sides: int,
    # outputs
    points: wp.array[wp.vec3],
):
    """The rings of a tube around each cable's particles, one thread per cable, in a frame carried along the cable
    by parallel transport so that the rings do not twist; ``sides + 1`` points per ring, the last repeating the
    first."""
    c = wp.tid()
    start = starts[c]
    end = starts[c + 1]
    t = wp.normalize(x[start + 1] - x[start])
    ref = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(wp.dot(ref, t)) > 0.9:
        ref = wp.vec3(1.0, 0.0, 0.0)
    n = wp.normalize(wp.cross(t, ref))
    for i in range(start, end):
        t_new = wp.normalize(x[wp.min(i + 1, end - 1)] - x[wp.max(i - 1, start)])
        axis = wp.cross(t, t_new)
        s = wp.length(axis)
        if s > 1.0e-8:
            n = wp.quat_rotate(wp.quat_from_axis_angle(axis / s, wp.atan2(s, wp.dot(t, t_new))), n)
        t = t_new
        n = wp.normalize(n - wp.dot(n, t) * t)
        b = wp.cross(t, n)
        for k in range(sides + 1):
            angle = 2.0 * wp.pi * float(k) / float(sides)
            points[i * (sides + 1) + k] = x[i] + radius[c] * (wp.cos(angle) * n + wp.sin(angle) * b)


def box_mesh(half, center=(0.0, 0.0, 0.0)):
    """Vertices and triangles of a box of half extents ``half`` at ``center``, flat shaded."""
    hx, hy, hz = half
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = corners + np.asarray(center)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    vertices, triangles = [], []
    for f in faces:
        base = len(vertices)
        vertices += [corners[i] for i in f]
        triangles += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(vertices), np.array(triangles, dtype=np.int32)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt, self.sim_substeps = 1.0 / 50.0, 2  # steps of 10 ms
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = MU_DESK
        builder.add_ground_plane()
        # The desk: its top and four legs, and the laptop on it.
        (dx, dy), thickness = DESK_HALF, 0.03
        wood = (0.55, 0.38, 0.22)
        builder.add_shape_box(
            -1,
            xform=wp.transform((0.0, 0.0, DESK_TOP - 0.5 * thickness), wp.quat_identity()),
            hx=dx,
            hy=dy,
            hz=0.5 * thickness,
            color=wood,
        )
        leg_height = DESK_TOP - thickness
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform((sx * (dx - 0.04), sy * (dy - 0.04), 0.5 * leg_height), wp.quat_identity()),
                    hx=0.02,
                    hy=0.02,
                    hz=0.5 * leg_height,
                    color=wood,
                )
        (lx, ly), (hx, hy), lt = LAPTOP
        builder.add_shape_box(
            -1,
            xform=wp.transform((lx, ly, DESK_TOP + 0.5 * lt), wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=0.5 * lt,
            color=(0.55, 0.56, 0.58),
        )
        tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.3)
        builder.add_shape_box(
            -1,
            xform=wp.transform((lx, ly + hy + 0.03, DESK_TOP + lt + 0.105), tilt),
            hx=hx,
            hy=0.004,
            hz=0.105,
            color=(0.55, 0.56, 0.58),
        )

        # The cables: particle rods, plugs at both ends, the laptop's plugs held in their ports.
        self.cables, self.plugs, held, radii = {}, [], [], []
        for name, path in cable_paths().items():
            c = CABLES[name]
            r = c["radius"]
            points = smooth_path(path, 2.0 * r)
            points[:, 2] = np.maximum(points[:, 2], r)
            seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
            arc = np.r_[0.0, np.cumsum(seg)]
            length = arc[-1]
            # Bending stiffness per interior vertex: the plug housings, stiff strain reliefs behind them, the cable.
            plug_length = c["plug"][0]
            inner = arc[1:-1]
            to_end = np.minimum(inner, length - inner)
            bending = np.where(
                to_end < plug_length,
                c["bending"] * PLUG_BENDING_SCALE,
                np.where(to_end < plug_length + RELIEF_LENGTH, c["bending"] * RELIEF_SCALE, c["bending"]),
            )
            density = c["line_density"] / (np.pi * r * r)
            rod = SolverFBA.add_particle_rod(builder, points, r, c["young"], density=density, bending_stiffness=bending)
            # The plugs' masses spread over their particles.
            for end_particles in (
                rod[: round(plug_length / (2.0 * r)) + 1],
                rod[-round(plug_length / (2.0 * r)) - 1 :],
            ):
                for i in end_particles:
                    builder.particle_mass[i] += c["plug"][1] / len(end_particles)
                self.plugs.append((name, list(end_particles)))
            self.cables[name] = rod
            radii.append(r)
        # The laptop holds the plugs in its ports (the first plug of the HDMI and USB-C cables and of the power
        # cord), and the hand the HDMI cable's far plug by its end.
        for name in ("hdmi", "usb_c", "power"):
            plug = next(p for n, p in self.plugs if n == name)
            held += plug
        self.hand_particle = self.cables["hdmi"][-1]
        held.append(self.hand_particle)
        for i in held:
            builder.particle_mass[i] = 0.0
        self.model = builder.finalize()
        self.model.particle_mu = MU_CABLE

        # Ten LG iterations per 10 ms step, at most ten conjugate residual iterations per contact solve, the cables
        # colliding as capsules, and light Rayleigh damping: mass-proportional 0.3/s, and corotated stiffness
        # damping of 1 ms on the cables' stretch and bending rates.
        # The contact solves keep their Delassus matrix in sparse storage, whose NSN step replays from step to step.
        self.solver = SolverFBA(
            self.model,
            iterations=10,
            nsn_linear_iterations=10,
            nsn_sparse_storage=True,
            rod_contact=True,
            rayleigh_alpha=0.3,
            rayleigh_beta=1.0e-3,
            rayleigh_corotated=True,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.01)
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pin_targets = self.model.particle_q.numpy()

        # The tubes, one mesh per cable, and the plugs' housings, boxes posed by their particles.
        self.tubes = []
        for (name, rod), r in zip(self.cables.items(), radii, strict=True):
            n, ring = len(rod), TUBE_SIDES + 1
            quads = []
            for i in range(n - 1):
                for k in range(TUBE_SIDES):
                    a, b = i * ring + k, i * ring + k + 1
                    quads += [a, b, b + ring, a, b + ring, a + ring]
            self.tubes.append(
                (
                    name,
                    wp.array(np.asarray(rod, dtype=np.int32)),
                    wp.zeros(n * ring, dtype=wp.vec3),
                    wp.array(np.asarray(quads, dtype=np.int32)),
                    r,
                )
            )
        # Each plug: a plastic housing over its particles, and metal parts beyond its outer end, the cable's end.
        self.plug_meshes = []
        for k, (name, particles) in enumerate(self.plugs):
            length, _, half_width = CABLES[name]["plug"]
            outer = -1.0 if k % 2 == 0 else 1.0  # along the plug's particles, its first plug faces back
            housing = box_mesh((0.5 * length, half_width, 0.6 * half_width))
            vertices, triangles = [], []
            for half_length, half_w, half_h, across in PLUG_METAL[name]:
                center = (outer * (0.5 * length + half_length), across, 0.0)
                v, t = box_mesh((half_length, half_w, half_h), center)
                triangles.append(t + sum(len(a) for a in vertices))
                vertices.append(v)
            metal = (np.concatenate(vertices), np.concatenate(triangles))
            self.plug_meshes.append((name, particles, housing[0], wp.array(housing[1]), metal[0], wp.array(metal[1])))
        self.rest_length = self.model.spring_rest_length.numpy()
        self.springs = self.model.spring_indices.numpy().reshape(-1, 2)
        self.max_strain = 0.0

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(-0.55, -1.05, 1.15), pitch=-19.0, yaw=61.0)

    def hand(self, t):
        """The hand's position at time ``t`` [s], eased between its waypoints."""
        times = [w[0] for w in HAND]
        if t >= times[-1]:
            return np.array(HAND[-1][1])
        k = int(np.searchsorted(times, t, side="right")) - 1
        (t0, p0), (t1, p1) = HAND[k], HAND[k + 1]
        s = (t - t0) / (t1 - t0)
        s = s * s * (3.0 - 2.0 * s)
        return (1.0 - s) * np.array(p0) + s * np.array(p1)

    def simulate(self):
        # SolverFBA reads the contact count back to the host every step, so the loop is not graph-captured.
        for k in range(self.sim_substeps):
            self.pin_targets[self.hand_particle] = self.hand(self.sim_time + (k + 1) * self.sim_dt)
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
        self.max_strain = max(self.max_strain, float(np.abs(length / self.rest_length - 1.0).max()))

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        assert self.max_strain < 0.02, f"the cables stretched by up to {100 * self.max_strain:.2f}%"
        assert x[:, 2].min() > 0.0, f"a cable sank into the floor to z = {x[:, 2].min():.4f} m"
        np.testing.assert_allclose(x[self.hand_particle], self.hand(self.sim_time), atol=1.0e-3)
        if self.sim_time > 6.5:
            # The HDMI cable's far end lies on the desk on the USB-C cable's side.
            end = x[self.cables["hdmi"][-12:]]
            assert end[:, 0].max() < -0.1 and (end[:, 2] > DESK_TOP).all(), "the HDMI cable did not reach its place"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        x = self.state_0.particle_q
        for name, rod, points, indices, r in self.tubes:
            starts = wp.array(np.array([0, rod.shape[0]], dtype=np.int32))
            positions = wp.array(x.numpy()[rod.numpy()], dtype=wp.vec3)
            radius = wp.array(np.array([r], dtype=np.float32))
            wp.launch(cable_tube_kernel, dim=1, inputs=[positions, starts, radius, TUBE_SIDES], outputs=[points])
            self.viewer.log_mesh(f"/cables/{name}", points, indices, color=CABLES[name]["color"], roughness=0.55)
        xh = x.numpy()
        for k, (name, particles, housing, housing_tris, metal, metal_tris) in enumerate(self.plug_meshes):
            p = xh[particles]
            center = 0.5 * (p[0] + p[-1])
            axis = p[-1] - p[0]
            axis /= np.linalg.norm(axis)
            side = np.cross(axis, (0.0, 0.0, 1.0))
            side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1.0e-6 else np.array((0.0, 1.0, 0.0))
            up = np.cross(side, axis)
            frame = np.stack([axis, side, up])
            plastic = (0.12, 0.12, 0.13) if name == "hdmi" else (0.95, 0.95, 0.93)
            for part, vertices, triangles, color, metallic in (
                ("housing", housing, housing_tris, plastic, 0.0),
                ("metal", metal, metal_tris, (0.78, 0.78, 0.8), 1.0),
            ):
                world = center + vertices @ frame
                self.viewer.log_mesh(
                    f"/plugs/{k}/{part}",
                    wp.array(world.astype(np.float32), dtype=wp.vec3),
                    triangles,
                    color=color,
                    roughness=0.4 if metallic else 0.5,
                    metallic=metallic,
                )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=400)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
