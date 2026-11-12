# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Nonconvex Pile FBA
#
# A thousand rigid bodies of eight non-convex shapes drop into an open box
# and settle in a pile. Each body collides as convex parts: CoACD
# decomposes the meshes of a torus, a cup, a star prism and an open ring,
# and boxes compose a U channel, an L bracket, a jack and a stool.
# SolverFBA solves the contacts between the parts, the walls and the
# ground in one sparse Newton system per LG iteration, with the contact
# multipliers carried from step to step.
#
# Command: python -m newton.examples nonconvex_pile_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

SIZE = 0.08  # largest extent of every shape [m]
COLORS = (
    (0.95, 0.55, 0.20),
    (0.30, 0.62, 0.90),
    (0.95, 0.80, 0.20),
    (0.60, 0.40, 0.80),
    (0.35, 0.70, 0.35),
    (0.90, 0.35, 0.40),
    (0.30, 0.75, 0.75),
    (0.75, 0.55, 0.35),
)


def revolved(profile, sections=32):
    """The closed mesh of a profile ``[(r, z)]`` that starts and ends on the axis, revolved about z."""
    profile = np.asarray(profile, dtype=np.float64)
    inner = profile[1:-1]
    angles = 2.0 * np.pi * np.arange(sections) / sections
    ring = len(inner)
    vertices = [(0.0, 0.0, profile[0, 1]), (0.0, 0.0, profile[-1, 1])]
    for a in angles:
        vertices += [(r * np.cos(a), r * np.sin(a), z) for r, z in inner]
    faces = []
    for s in range(sections):
        t = (s + 1) % sections
        a, b = 2 + s * ring, 2 + t * ring
        faces += [(0, b, a), (1, a + ring - 1, b + ring - 1)]
        for k in range(ring - 1):
            faces += [(a + k, b + k, b + k + 1), (a + k, b + k + 1, a + k + 1)]
    return np.array(vertices), np.array(faces)


def swept_ring(angle, sections, major=1.0, minor=0.32, segments=12):
    """A torus of tube radius ``minor``, swept over ``angle`` about z, its open ends capped."""
    full = angle >= 2.0 * np.pi - 1e-9
    count = sections if full else sections + 1
    phi = 2.0 * np.pi * np.arange(segments) / segments
    vertices = []
    for s in range(count):
        a = angle * s / sections
        center = np.array([major * np.cos(a), major * np.sin(a), 0.0])
        out = np.array([np.cos(a), np.sin(a), 0.0])
        vertices += [center + minor * (np.cos(p) * out + np.sin(p) * np.array([0.0, 0.0, 1.0])) for p in phi]
    faces = []
    for s in range(sections):
        a, b = s * segments, ((s + 1) % count) * segments
        for k in range(segments):
            m = (k + 1) % segments
            faces += [(a + k, b + k, b + m), (a + k, b + m, a + m)]
    if not full:
        last = len(vertices)
        vertices += [np.array([major, 0.0, 0.0]), major * np.array([np.cos(angle), np.sin(angle), 0.0])]
        end = sections * segments
        for k in range(segments):
            m = (k + 1) % segments
            faces += [(last, k, m), (last + 1, end + m, end + k)]
    return np.array(vertices), np.array(faces)


def star_prism(points=5, outer=1.0, inner=0.45, height=0.5):
    """A star of ``points`` points extruded along z."""
    angles = np.pi / 2 + np.arange(2 * points) * np.pi / points
    radius = np.where(np.arange(2 * points) % 2 == 0, outer, inner)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    m = len(ring)
    vertices = [(0.0, 0.0, -height / 2), (0.0, 0.0, height / 2)]
    vertices += [(x, y, -height / 2) for x, y in ring] + [(x, y, height / 2) for x, y in ring]
    faces = []
    for i in range(m):
        j = (i + 1) % m
        faces += [(0, 2 + j, 2 + i), (1, 2 + m + i, 2 + m + j)]
        faces += [(2 + i, 2 + j, 2 + m + j), (2 + i, 2 + m + j, 2 + m + i)]
    return np.array(vertices), np.array(faces)


def scaled(vertices, faces):
    """``vertices`` centered and scaled to a largest extent of ``SIZE``, as a mesh of hulls of up to 32 vertices."""
    vertices = vertices - 0.5 * (vertices.max(axis=0) + vertices.min(axis=0))
    vertices = vertices * (SIZE / np.ptp(vertices, axis=0).max())
    return newton.Mesh(vertices.astype(np.float32), faces.astype(np.int32).flatten(), maxhullvert=32)


def curved_meshes():
    """The torus, the cup, the star prism and the open ring, which CoACD decomposes."""
    cup = [(0.0, 0.0), (0.42, 0.0), (0.5, 0.06), (0.5, 1.0), (0.42, 1.0), (0.42, 0.12), (0.0, 0.12)]
    return [
        scaled(*swept_ring(2.0 * np.pi, 24)),
        scaled(*revolved(cup)),
        scaled(*star_prism()),
        scaled(*swept_ring(1.5 * np.pi, 18)),
    ]


def box_compounds():
    """The U channel, the L bracket, the jack and the stool as boxes ``(center, half extents)``."""
    h, w = 0.5 * SIZE, 0.1 * SIZE
    u = [
        ((0.0, 0.0, -h + w), (h, 0.6 * h, w)),
        ((-h + w, 0.0, 0.0), (w, 0.6 * h, h)),
        ((h - w, 0.0, 0.0), (w, 0.6 * h, h)),
    ]
    l_bracket = [((0.0, 0.0, -h + w), (h, 0.7 * h, w)), ((-h + w, 0.0, 0.0), (w, 0.7 * h, h))]
    jack = [((0.0, 0.0, 0.0), (h, w, w)), ((0.0, 0.0, 0.0), (w, h, w)), ((0.0, 0.0, 0.0), (w, w, h))]
    stool = [((0.0, 0.0, 0.5 * h), (h, h, 0.1 * h))]
    stool += [
        ((x, y, -0.29 * h), (0.6 * w, 0.6 * w, 0.71 * h)) for x in (-0.8 * h, 0.8 * h) for y in (-0.8 * h, 0.8 * h)
    ]
    return [u, l_bracket, jack, stool]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        count = args.count

        rng = np.random.default_rng(args.seed)
        builder = newton.ModelBuilder()
        # Contacts are kept from 5 mm apart, as in convex_pile_fba.
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, density=1000.0, gap=0.005)
        builder.add_ground_plane(cfg=cfg)

        # An open square box about eight bodies deep for the pile.
        per_layer = int(np.ceil(np.sqrt(count / 8.0)))
        self.side = per_layer * 1.6 * SIZE
        wall_half, thickness = 5.0 * SIZE, 0.02
        offset = 0.5 * self.side + thickness
        walls = ((1, 0, thickness, offset), (-1, 0, thickness, offset))
        walls += ((0, 1, offset, thickness), (0, -1, offset, thickness))
        for x, y, hx, hy in walls:
            xform = wp.transform((x * offset, y * offset, wall_half), wp.quat_identity())
            builder.add_shape_box(-1, xform=xform, hx=hx, hy=hy, hz=wall_half, cfg=cfg, color=(0.82, 0.82, 0.86))

        # Layers of bodies above the box, turned at random: four curved meshes, decomposed below, and four compounds
        # of boxes.
        curved, compounds = curved_meshes(), box_compounds()
        kinds = len(curved) + len(compounds)
        spacing = 1.25 * SIZE
        columns = max(1, min(int(np.ceil(np.sqrt(count / 12.0))), int(self.side / spacing)))
        self.kind = np.arange(count) % kinds
        mesh_shapes = []
        for i in range(count):
            layer, k = divmod(i, columns * columns)
            gx, gy = divmod(k, columns)
            x = (gx - 0.5 * (columns - 1)) * spacing + rng.uniform(-0.1, 0.1) * SIZE
            y = (gy - 0.5 * (columns - 1)) * spacing + rng.uniform(-0.1, 0.1) * SIZE
            z = 0.8 * SIZE + layer * spacing
            body = builder.add_body(xform=wp.transform((x, y, z), wp.quat_rpy(*rng.uniform(-np.pi, np.pi, 3))))
            kind, color = self.kind[i], COLORS[self.kind[i]]
            if kind < len(curved):
                mesh_shapes.append(builder.add_shape_mesh(body, mesh=curved[kind], cfg=cfg, color=color))
            else:
                for center, half in compounds[kind - len(curved)]:
                    xform = wp.transform(center, wp.quat_identity())
                    builder.add_shape_box(body, xform=xform, hx=half[0], hy=half[1], hz=half[2], cfg=cfg, color=color)

        # CoACD decomposes each curved mesh once for all the bodies that share it, with its own search depth, and merges
        # the parts down to eight.
        import coacd  # noqa: PLC0415

        coacd.set_log_level("error")
        builder.approximate_meshes(
            "coacd",
            shape_indices=mesh_shapes,
            raise_on_failure=True,
            seed=0,
            threshold=0.05,
            max_convex_hull=8,
            merge=True,
            mcts_iterations=150,
            mcts_max_depth=3,
        )

        self.model = builder.finalize()
        # Five LG iterations and at most ten conjugate residual iterations per contact solve, the contact rows in
        # sparse storage, and each step's contact multipliers starting from the previous step's.
        self.solver = SolverFBA(
            self.model, iterations=5, nsn_linear_iterations=10, nsn_sparse_storage=True, contact_warmstart=True
        )
        # Sweep and prune: an explicit list of all pairs of the parts would hold tens of millions of pairs.
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=64 * count, broad_phase="sap")
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        distance = 1.7 * self.side
        self.viewer.set_camera(pos=wp.vec3(-0.55 * distance, -0.55 * distance, 1.15 * distance), pitch=-52.0, yaw=45.0)

    def simulate(self):
        # SolverFBA reads the contact count back to the host every step, so the loop is not graph-captured.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        # The bodies land at about 5 m/s.
        speed = np.abs(self.state_0.body_qd.numpy()[:, :3]).max()
        assert speed < 8.0, f"body speed {speed:.2f} m/s at t={self.sim_time:.3f} s"

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        velocity = self.state_0.body_qd.numpy()[:, :3]
        assert np.all(np.isfinite(body_q)) and np.all(np.isfinite(velocity)), "non-finite state"
        # Every body lies in the box, its parts on or above the ground, and the pile has settled.
        inside = 0.5 * self.side
        assert np.abs(body_q[:, :2]).max() < inside, f"a body left the box: {np.abs(body_q[:, :2]).max():.3f} m"
        lowest = self.lowest_point(body_q)
        assert lowest > -1.0e-3, f"a body sank into the ground: lowest point at z={lowest:.4f} m"
        speed = np.linalg.norm(velocity, axis=1)
        assert speed.mean() < 0.01, f"the pile still moves: mean speed {speed.mean():.4f} m/s"

    def lowest_point(self, body_q):
        """The lowest world height of the bodies' collision geometry: hull vertices and box corners."""
        model = self.model
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_xform = model.shape_transform.numpy()
        shape_scale = model.shape_scale.numpy()
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        low = np.inf
        for s in np.flatnonzero(shape_body >= 0):
            if shape_type[s] == newton.GeoType.BOX:
                local = corners * shape_scale[s]
            elif shape_type[s] == newton.GeoType.CONVEX_MESH:
                local = model.shape_source[s].vertices.astype(np.float64) * shape_scale[s]
            else:
                continue
            for q in (shape_xform[s], body_q[shape_body[s]]):
                xyz, w = q[3:6], q[6]
                t = 2.0 * np.cross(xyz, local)
                local = q[:3] + local + w * t + np.cross(xyz, t)
            low = min(low, local[:, 2].min())
        return low

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--count", type=int, default=1000, help="Number of bodies.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the poses and positions.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
