# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Chain Net FBA
#
# A net of 1,080 interlocked steel chain links, after the huge chain net of
# Affine Body Dynamics (Lan et al. 2022, Fig. 13): flat links on a 20 x 20
# grid, its rim fixed, joined by standing links hooked through their ends and
# around their sides. Released, the net falls into a bowl 12 m deep and swings
# back. SolverFBA solves the contacts of the 1,008 free links in one sparse
# Newton system per LG iteration, with the contact multipliers carried from
# step to step; each link collides as 14 capsules along its centerline.
#
# Command: python -m newton.examples chain_net_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

# A steel chain link 1.5 m long and 1 m wide: rounded ends of centerline radius 0.4 m, 0.5 m apart, joined by straight
# sides, around a tube of radius 0.1 m.
END_RADIUS = 0.4  # [m]
END_OFFSET = 0.25  # [m], from the link's middle to each end's center
TUBE_RADIUS = 0.1  # [m]
END_SEGMENTS = 6  # capsules per rounded end
DENSITY = 7680.0  # [kg/m^3]
PITCH = (1.65, 1.6)  # [m], the flat links' spacing along x and y
HEIGHT = 15.0  # [m]
FIXED_COLOR = (0.35, 0.55, 0.85)
FREE_COLOR = (0.78, 0.78, 0.80)


def link_capsules():
    """The capsules of a link along its closed centerline, in the link's frame (long axis y, the link in the xy
    plane): ``(transform, half height)`` per capsule."""
    angles = np.pi * np.arange(END_SEGMENTS + 1) / END_SEGMENTS
    top = np.stack([END_RADIUS * np.cos(angles), END_OFFSET + END_RADIUS * np.sin(angles), 0.0 * angles], axis=1)
    centerline = np.vstack([top, -top])  # the bottom end is the top end turned by half a turn about z
    capsules = []
    for k in range(len(centerline)):
        p0, p1 = centerline[k], centerline[(k + 1) % len(centerline)]
        axis = (p1 - p0) / np.linalg.norm(p1 - p0)
        # The capsule's axis is its local z; turn z onto the segment.
        turn = np.cross((0.0, 0.0, 1.0), axis)
        q = wp.quat_from_axis_angle(wp.vec3(*(turn / np.linalg.norm(turn))), float(np.arccos(axis[2])))
        capsules.append((wp.transform(wp.vec3(*(0.5 * (p0 + p1))), q), 0.5 * float(np.linalg.norm(p1 - p0))))
    return capsules


def link_rotation(x_axis, long_axis):
    """The rotation that turns a link's local x axis to ``x_axis`` and its long axis (local y) to ``long_axis``."""
    x, y = np.asarray(x_axis, dtype=np.float64), np.asarray(long_axis, dtype=np.float64)
    return wp.quat_from_matrix(wp.mat33(np.stack([x, y, np.cross(x, y)], axis=1)))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        n = args.size

        builder = newton.ModelBuilder()
        # Frictionless steel; contacts are kept from 2 cm apart, a fifth of the tube radius.
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, density=DENSITY, gap=0.02)
        capsules = link_capsules()
        flat = link_rotation((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))  # lying in the xy plane, long axis along x
        standing_x = link_rotation((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))  # standing in the xz plane, long axis along x
        standing_y = link_rotation((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))  # standing in the yz plane, long axis along y

        def add_link(position, rotation, fixed):
            """A link at ``position``: its body, or -1 for a fixed link, whose capsules are static."""
            xform = wp.transform(wp.vec3(*position), rotation)
            body = -1 if fixed else builder.add_body(xform=xform)
            for local, half_height in capsules:
                builder.add_shape_capsule(
                    body,
                    xform=wp.transform_multiply(xform, local) if fixed else local,
                    radius=TUBE_RADIUS,
                    half_height=half_height,
                    cfg=cfg,
                    color=FIXED_COLOR if fixed else FREE_COLOR,
                )
            return body

        # The flat links on an n x n grid, its corners left out and its rim fixed. Between two flat links of a row
        # stands a link along x, its ends hooked through theirs; between two of a column, one along y, its ends
        # around their sides. Each standing link holds its two neighbors: (link, neighbor, neighbor's fixed center).
        self.span = (n - 1) * PITCH[0]
        self.hooks = []
        grid = {}
        for i in range(n):
            for j in range(n):
                if i in (0, n - 1) and j in (0, n - 1):
                    continue
                center = ((j - 0.5 * (n - 1)) * PITCH[0], (i - 0.5 * (n - 1)) * PITCH[1], HEIGHT)
                grid[i, j] = (add_link(center, flat, fixed=i in (0, n - 1) or j in (0, n - 1)), center)
        for (i, j), (_body, center) in list(grid.items()):
            for di, dj, rotation in ((0, 1, standing_x), (1, 0, standing_y)):
                if (i + di, j + dj) not in grid or (di == 0 and i in (0, n - 1)) or (dj == 0 and j in (0, n - 1)):
                    continue
                middle = (center[0] + 0.5 * dj * PITCH[0], center[1] + 0.5 * di * PITCH[1], HEIGHT)
                link = add_link(middle, rotation, fixed=False)
                for neighbor in ((i, j), (i + di, j + dj)):
                    self.hooks.append((link, *grid[neighbor]))
        self.free = np.array(sorted({link for link, _, _ in self.hooks} | {b for b, _ in grid.values() if b >= 0}))

        self.model = builder.finalize()
        # Five LG iterations and at most ten conjugate residual iterations per contact solve, the contact rows in
        # sparse storage, and each step's contact multipliers starting from the previous step's.
        self.solver = SolverFBA(
            self.model, iterations=5, nsn_linear_iterations=10, nsn_sparse_storage=True, contact_warmstart=True
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=64 * self.model.body_count, broad_phase="sap"
        )
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        distance = 1.25 * self.span
        self.viewer.set_camera(
            pos=wp.vec3(-0.62 * distance, -0.62 * distance, HEIGHT + 0.15 * distance), pitch=-19.0, yaw=45.0
        )

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
        # The links at the bottom of the net reach about 20 m/s as it falls into its bowl; as the net catches there,
        # at about 1.3 s, one standing link gets about 31 m/s for a frame from its contacts.
        speed = np.linalg.norm(self.state_0.body_qd.numpy()[:, :3], axis=1).max()
        assert speed < 40.0, f"link speed {speed:.2f} m/s at t={self.sim_time:.3f} s"

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(body_q)) and np.all(np.isfinite(self.state_0.body_qd.numpy())), "non-finite state"
        # Every standing link still holds both its neighbors: hooked links pulled taut are 1.3 m apart, center to
        # center, and one that came apart would be farther than that and a tube radius.
        reach = 2.0 * (END_OFFSET + END_RADIUS) + TUBE_RADIUS
        for link, neighbor, fixed_center in self.hooks:
            center = body_q[neighbor, :3] if neighbor >= 0 else np.asarray(fixed_center)
            distance = float(np.linalg.norm(body_q[link, :3] - center))
            assert distance < reach, f"link {link} came apart from its neighbor: {distance:.3f} m apart"
        # The net has fallen into a bowl.
        sag = HEIGHT - body_q[self.free, 2].min()
        assert sag > 2.0, f"the net sagged only {sag:.2f} m"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--size", type=int, default=20, help="Flat links per side of the net's grid.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
