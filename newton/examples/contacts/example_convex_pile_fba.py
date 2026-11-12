# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Convex Pile FBA
#
# A thousand rigid bodies of eight convex shapes drop into an open box and
# settle in a pile. SolverFBA solves the contacts between the bodies, the
# walls and the ground in one sparse Newton system per LG iteration, with
# the contact multipliers carried from step to step.
#
# Command: python -m newton.examples convex_pile_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

HALF = 0.03  # body half size [m]
COLORS = (
    (0.90, 0.35, 0.25),
    (0.95, 0.70, 0.20),
    (0.35, 0.70, 0.35),
    (0.25, 0.55, 0.85),
    (0.60, 0.40, 0.80),
    (0.90, 0.50, 0.65),
    (0.30, 0.75, 0.75),
    (0.75, 0.75, 0.40),
)


def hull(points):
    """The convex hull of ``points`` as a collision mesh."""
    cloud = newton.Mesh(np.asarray(points, dtype=np.float32), [0, 1, 2], compute_inertia=False)
    shape = cloud.compute_convex_hull()
    return newton.Mesh(shape.vertices, shape.indices)


def body_meshes(rng):
    """A cube, a slab, a bar, a tetrahedron, an octahedron and three random polytopes."""
    h = HALF
    cube = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
    meshes = [hull(cube), hull(cube * (1.6, 1.2, 0.5)), hull(cube * (2.0, 0.6, 0.6))]
    meshes.append(hull(1.3 * np.array([[h, h, h], [-h, -h, h], [-h, h, -h], [h, -h, -h]])))
    meshes.append(hull(1.4 * h * np.vstack([np.eye(3), -np.eye(3)])))
    for _ in range(3):
        directions = rng.normal(size=(14, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        meshes.append(hull(1.2 * h * directions * rng.uniform(0.7, 1.0, (14, 1))))
    return meshes


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
        # Contacts are kept from 5 mm apart; the default gap, 10 cm, is larger than the bodies and would keep
        # tens of thousands of speculative contacts in the pile.
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, density=1000.0, gap=0.005)
        builder.add_ground_plane(cfg=cfg)

        # An open square box about eight bodies deep for the pile.
        per_layer = int(np.ceil(np.sqrt(count / 8.0)))
        self.side = per_layer * 3.2 * HALF
        wall_half, thickness = 10.0 * HALF, 0.02
        offset = 0.5 * self.side + thickness
        walls = ((1, 0, thickness, offset), (-1, 0, thickness, offset))
        walls += ((0, 1, offset, thickness), (0, -1, offset, thickness))
        for x, y, hx, hy in walls:
            xform = wp.transform((x * offset, y * offset, wall_half), wp.quat_identity())
            builder.add_shape_box(-1, xform=xform, hx=hx, hy=hy, hz=wall_half, cfg=cfg, color=(0.82, 0.82, 0.86))

        # Layers of bodies above the box, turned at random.
        meshes = body_meshes(rng)
        self.hull_points = [mesh.vertices for mesh in meshes]
        spacing = 3.0 * HALF
        columns = max(1, min(int(np.ceil(np.sqrt(count / 12.0))), int(self.side / spacing)))
        for i in range(count):
            layer, k = divmod(i, columns * columns)
            gx, gy = divmod(k, columns)
            x = (gx - 0.5 * (columns - 1)) * spacing + rng.uniform(-0.2, 0.2) * HALF
            y = (gy - 0.5 * (columns - 1)) * spacing + rng.uniform(-0.2, 0.2) * HALF
            z = 2.0 * HALF + layer * spacing
            body = builder.add_body(xform=wp.transform((x, y, z), wp.quat_rpy(*rng.uniform(-np.pi, np.pi, 3))))
            kind = i % len(meshes)
            builder.add_shape_convex_hull(body, mesh=meshes[kind], cfg=cfg, color=COLORS[kind])

        self.model = builder.finalize()
        # Five LG iterations and at most ten conjugate residual iterations per contact solve, the contact rows in
        # sparse storage, and each step's contact multipliers starting from the previous step's.
        self.solver = SolverFBA(
            self.model, iterations=5, nsn_linear_iterations=10, nsn_sparse_storage=True, contact_warmstart=True
        )
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=32 * count)
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
        # The bodies land at about 4 m/s.
        speed = np.abs(self.state_0.body_qd.numpy()[:, :3]).max()
        assert speed < 8.0, f"body speed {speed:.2f} m/s at t={self.sim_time:.3f} s"

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        velocity = self.state_0.body_qd.numpy()[:, :3]
        assert np.all(np.isfinite(body_q)) and np.all(np.isfinite(velocity)), "non-finite state"
        # Every body lies in the box, its hull on or above the ground, and the pile has settled.
        inside = 0.5 * self.side
        assert np.abs(body_q[:, :2]).max() < inside, f"a body left the box: {np.abs(body_q[:, :2]).max():.3f} m"
        lowest = min(self.lowest_point(body_q[k :: len(self.hull_points)], p) for k, p in enumerate(self.hull_points))
        assert lowest > -1.0e-3, f"a body sank into the ground: lowest hull point at z={lowest:.4f} m"
        speed = np.linalg.norm(velocity, axis=1)
        assert speed.mean() < 0.01, f"the pile still moves: mean speed {speed.mean():.4f} m/s"

    @staticmethod
    def lowest_point(body_q, points):
        """The lowest world height of ``points``, given in the frame of each of the bodies at ``body_q``."""
        xyz, w = body_q[:, None, 3:6], body_q[:, None, 6:7]
        t = 2.0 * np.cross(xyz, points[None])
        world = body_q[:, None, :3] + points[None] + w * t + np.cross(xyz, t)
        return world[..., 2].min()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--count", type=int, default=1000, help="Number of bodies.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the shapes, poses and positions.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
