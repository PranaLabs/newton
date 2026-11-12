# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Convex Hulls FBA
#
# Rigid bodies with convex hull colliders in SolverFBA: a stack of five
# cubes, a cube dropped into a U built from three hulls on one body, and a
# cube dropped into a concave U mesh that CoACD decomposes into convex
# hulls. The collision pipeline's convex contacts hold the hulls on the
# ground and on each other.
#
# Command: python -m newton.examples convex_hulls_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

HALF = 0.05  # cube half size [m]
STACK_X, COMPOUND_X, CONCAVE_X = -0.6, 0.0, 0.6
# The U's outline in its x-z plane, counterclockwise, extruded 0.3 m along y: a base 5 cm thick between two arms
# 5 cm wide and 25 cm high.
U_OUTLINE = (
    (-0.2, 0.0),
    (0.2, 0.0),
    (0.2, 0.25),
    (0.15, 0.25),
    (0.15, 0.05),
    (-0.15, 0.05),
    (-0.15, 0.25),
    (-0.2, 0.25),
)
U_DEPTH = 0.15  # half extent along y [m]
U_BASE = 0.05  # base thickness [m]


def box_hull(hx, hy, hz):
    """A box as a convex hull mesh."""
    return newton.Mesh.create_box(hx, hy, hz, duplicate_vertices=False, compute_normals=False, compute_uvs=False)


def concave_u():
    """The U as one closed concave mesh: its outline extruded along y, the caps split into three convex pieces."""
    n = len(U_OUTLINE)
    vertices = [(x, y, z) for y in (-U_DEPTH, U_DEPTH) for x, z in U_OUTLINE]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, n + j, j), (i, n + i, n + j)]
    for a, b, c in ((0, 1, 4), (0, 4, 5), (1, 2, 3), (1, 3, 4), (5, 6, 7), (5, 7, 0)):
        faces += [(a, b, c), (n + a, n + c, n + b)]
    return newton.Mesh(np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32).flatten())


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        # Contacts are kept from 1 cm apart, a tenth of a cube.
        cube_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, density=1000.0, gap=0.01)
        u_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, density=500.0, gap=0.01)
        builder.add_ground_plane(cfg=cube_cfg)
        cube = box_hull(HALF, HALF, HALF)

        # Five cubes stacked 2 mm apart.
        self.stack = []
        for i in range(5):
            body = builder.add_body(
                xform=wp.transform((STACK_X, 0.0, HALF + i * (2 * HALF + 0.002)), wp.quat_identity())
            )
            builder.add_shape_convex_hull(body, mesh=cube, cfg=cube_cfg, color=(0.85, 0.45, 0.25))
            self.stack.append(body)

        # A U of three hulls on one body: the base and the two arms.
        self.compound = builder.add_body(xform=wp.transform((COMPOUND_X, 0.0, 0.0), wp.quat_identity()))
        for center, half in (
            ((0.0, 0.0, 0.025), (0.2, U_DEPTH, 0.025)),
            ((-0.175, 0.0, 0.15), (0.025, U_DEPTH, 0.1)),
            ((0.175, 0.0, 0.15), (0.025, U_DEPTH, 0.1)),
        ):
            xform = wp.transform(center, wp.quat_identity())
            builder.add_shape_convex_hull(
                self.compound, xform=xform, mesh=box_hull(*half), cfg=u_cfg, color=(0.35, 0.6, 0.85)
            )

        # The same U as one concave mesh, decomposed into convex hulls by CoACD.
        import coacd  # noqa: PLC0415

        coacd.set_log_level("error")
        self.concave = builder.add_body(xform=wp.transform((CONCAVE_X, 0.0, 0.0), wp.quat_identity()))
        shape = builder.add_shape_mesh(self.concave, mesh=concave_u(), cfg=u_cfg, color=(0.45, 0.75, 0.45))
        builder.approximate_meshes("coacd", shape_indices=[shape], raise_on_failure=True, seed=0)

        # A cube dropped into each U, turned a little.
        self.dropped = []
        for x in (COMPOUND_X, CONCAVE_X):
            body = builder.add_body(xform=wp.transform((x, 0.0, 0.5), wp.quat_rpy(0.1, 0.2, 0.0)))
            builder.add_shape_convex_hull(body, mesh=cube, cfg=cube_cfg, color=(0.9, 0.75, 0.3))
            self.dropped.append(body)

        self.model = builder.finalize()
        # Five LG iterations and at most ten conjugate residual iterations per contact solve.
        self.solver = SolverFBA(self.model, iterations=5, nsn_linear_iterations=10, nsn_sparse_storage=True)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -2.0, 1.0), pitch=-26.0, yaw=90.0)

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
        # The dropped cubes land at about 3 m/s.
        speed = np.abs(self.state_0.body_qd.numpy()[:, :3]).max()
        assert speed < 5.0, f"body speed {speed:.2f} m/s at t={self.sim_time:.3f} s"

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(body_q)) and np.all(np.isfinite(body_qd)), "non-finite state"
        # The stack stands in its column, each cube on the one below.
        for i, body in enumerate(self.stack):
            np.testing.assert_allclose(body_q[body, :3], (STACK_X, 0.0, HALF + 2 * HALF * i), atol=2.0e-3)
        # The U's stand where they were, and each dropped cube lies on its U's base between the arms.
        for u, x in ((self.compound, COMPOUND_X), (self.concave, CONCAVE_X)):
            np.testing.assert_allclose(body_q[u, :3], (x, 0.0, 0.0), atol=5.0e-3)
        for body, x in zip(self.dropped, (COMPOUND_X, CONCAVE_X), strict=True):
            assert abs(body_q[body, 0] - x) < 0.15 - HALF, f"cube outside the U: x={body_q[body, 0]:.3f}"
            assert abs(body_q[body, 2] - (U_BASE + HALF)) < 5.0e-3, f"cube not on the base: z={body_q[body, 2]:.4f}"
        # The capped conjugate residual solves leave the resting bodies moving by a few mm/s while their poses hold.
        speed = np.linalg.norm(body_qd[:, :3], axis=1).max()
        assert speed < 0.02, f"bodies still moving: {speed:.4f} m/s"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
