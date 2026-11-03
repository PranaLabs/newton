# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PneuNet Gripper FBA
#
# A three-finger soft gripper of pneumatic networks (PneuNets), after Fig. 2
# of Macklin et al. 2019, "Non-Smooth Newton Methods for Deformable
# Multi-Body Dynamics". One tetrahedral body forms a flat star: each finger
# is a base layer with a row of rounded chambers on its back. The chambers
# are volume controlled: their tetrahedra take negative activations along
# their finger, which lengthen them to the volume that the gas pumped into
# each chamber fills, as the chambers of a PneuNet lengthen, and the
# lengthened backs curl the fingers. A hub holds the star's center. Partly
# inflated, the fingers hang open around a ball like an umbrella; the hub
# lowers them over it, and the chambers fill from the hub out and close the
# fingers around the ball: the chamber next to the hub bends each finger
# down past the ball's shoulder, and the others, less and less toward the
# tip, lay the finger along the ball's side. The hub lifts the ball, sets it
# down, and the deflated fingers let it go. SolverFBA integrates the ball as
# a rigid body and solves its contacts with the fingers and the ground.
#
# The tetrahedral mesh, assets/pneunet_gripper_fba.npz (7363 vertices,
# 27395 tetrahedra), was built for this example with gmsh, its proportions
# measured off the paper's Fig. 2 for a 6 cm ball.
#
# Command: python -m newton.examples pneunet_gripper_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

# The paper's silicone rubber. A 6 cm plastic ball, which the silicone grips with a friction coefficient near 1.
YOUNG, POISSON, DENSITY = 1.0e7, 0.47, 1200.0  # [Pa], [-], [kg/m^3]
RADIUS, BALL_MASS, MU = 0.03, 0.15, 1.0  # [m], [kg], [-]
# Asset geometry [m]: the star lies in z >= 0 with its grasping face at z = 0, the chambers above BASE and the hub
# mount, a cylinder of radius ROOT at the center, up to the chamber tops. Each finger's chambers start CHAMBER_START
# from the center, PITCH apart. The gripper is built at SCALE times the asset's size.
BASE, ROOT = 0.0035, 0.008
CHAMBER_START, PITCH, CHAMBERS = 0.011, 0.012, 6
SCALE = 0.8
# Chamber length along its finger over its rest length, which is also its volume ratio. Opened, the fingers hang
# around the ball like an umbrella. Closed, the chamber next to the hub bends each finger down past the ball's
# shoulder, and the others curl the rest of the finger tighter than the ball, so that it presses the ball against
# the palm, and the tighter the closer to the hub, where the moment of the contact forces is largest. A finger
# curled evenly tighter than the ball touches it with its tip alone and bows away from it in between.
OPEN = 1.15
CLOSED = (1.5, 1.42, 1.4, 1.38, 1.36, 1.34)  # from the hub out
# The gas fills the chambers from the hub out, each CLOSE_DELAY [s] after the one before, so each finger wraps the
# ball from its root to its tip.
CLOSE_DELAY = 0.2
CLEARANCE = 0.06  # grasping face over the ball before and after the grasp [m]
GRASP_GAP = 0.002  # grasping face over the ball while the fingers close [m]
LIFT = 0.05  # [m]
CONTACT_THICKNESS = 0.001  # [m]
CYCLE = 11.0  # [s]


def _smooth(t, t0, t1):
    """0 before t0, 1 after t1, a smoothstep in between."""
    u = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def schedule(t):
    """Hub height over its start [m], how far the fingers are opened, and how far each chamber, from the hub out, is
    closed, all in [0, 1], in an 11 s cycle: open the fingers, lower them around the ball, close them, lift, hold,
    lower, open, raise and deflate."""
    t = t % CYCLE
    height = (GRASP_GAP - CLEARANCE) * (_smooth(t, 1.0, 2.0) - _smooth(t, 9.5, 10.5))
    height += LIFT * (_smooth(t, 3.8, 5.3) - _smooth(t, 6.3, 7.8))
    opened = _smooth(t, 0.0, 1.0) - _smooth(t, 10.0, 11.0)
    start = 2.0 + CLOSE_DELAY * np.arange(CHAMBERS)
    closed = np.array([_smooth(t, t0, t0 + 0.6) for t0 in start]) - _smooth(t, 8.0, 9.0)
    return height, opened, closed


def _hub_mesh(radius, height, sides=8):
    """Vertices and triangles of a prism with a regular polygon base, standing on z = 0."""
    angles = 2.0 * np.pi * (np.arange(sides) + 0.5) / sides
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(sides)], axis=1)
    top = ring + np.array([0.0, 0.0, height])
    vertices = np.concatenate([ring, top, [[0.0, 0.0, 0.0], [0.0, 0.0, height]]])
    tris = []
    for k in range(sides):
        j = (k + 1) % sides
        tris += [[k, j, sides + j], [k, sides + j, sides + k], [2 * sides, j, k], [2 * sides + 1, sides + k, sides + j]]
    return vertices.astype(np.float32), np.array(tris, dtype=np.int32).reshape(-1)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        # At two substeps the ball rocks in the closed fingers, a little more with each swing.
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        mesh = newton.TetMesh.create_from_file(newton.examples.get_asset("pneunet_gripper_fba.npz"))
        rest = mesh.vertices
        top = float(rest[:, 2].max())
        self.top = SCALE * top  # mount top over the grasping face [m]
        self.start = 2.0 * RADIUS + CLEARANCE  # grasping face height before the grasp

        builder = newton.ModelBuilder()
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        self.ball = builder.add_body(xform=wp.transform((0.0, 0.0, RADIUS), wp.quat_identity()))
        ball_density = BALL_MASS / (4.0 / 3.0 * np.pi * RADIUS**3)
        builder.add_shape_sphere(
            self.ball,
            radius=RADIUS,
            cfg=newton.ModelBuilder.ShapeConfig(mu=MU, density=ball_density),
            color=(0.93, 0.78, 0.58),
        )
        mu = YOUNG / (2.0 * (1.0 + POISSON))
        lam = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, self.start),
            rot=wp.quat_identity(),
            scale=SCALE,
            vel=wp.vec3(0.0),
            mesh=mesh,
            density=DENSITY,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
            particle_radius=CONTACT_THICKNESS,
        )
        self.model = builder.finalize()

        # The hub holds the top face of the mount and moves it through set_pin_targets.
        self.x0 = self.model.particle_q.numpy()
        radial = np.hypot(rest[:, 0], rest[:, 1])
        self.hub = np.isclose(rest[:, 2], top) & (radial < ROOT)
        mass = np.where(self.hub, 0.0, self.model.particle_mass.numpy())
        self.model.particle_mass.assign(mass)
        self.model.particle_inv_mass.assign(np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0))

        # Chamber tetrahedra: above the base layer, outside the mount. Each grows along its finger, from the center
        # out, and takes the closed length of its chamber.
        self.tets = self.model.tet_indices.numpy().reshape(-1, 4)
        centers = rest[self.tets].mean(axis=1)
        radial = np.hypot(centers[:, 0], centers[:, 1])
        self.chambers = (centers[:, 2] > BASE) & (radial > ROOT)
        self.chamber_index = np.clip(np.floor((radial - CHAMBER_START) / PITCH), 0, CHAMBERS - 1).astype(np.int32)
        self.closed_volume = np.array(CLOSED)[self.chamber_index]
        finger = np.rint(np.arctan2(centers[:, 1], centers[:, 0]) / (2.0 * np.pi / 3.0)) * (2.0 * np.pi / 3.0)
        axis = np.stack([np.cos(finger), np.sin(finger), np.zeros_like(finger)], axis=1)
        directions = np.where(self.chambers[:, None], axis, 0.0)

        self.solver = SolverFBA(
            self.model,
            iterations=10,
            stretching_model="corotational",
            rayleigh_alpha=50.0,
            # The fingers press the ball onto the ground and close contact loops, whose Newton steps the default
            # regularization lets throw the ball.
            contact_regularization=0.1,
            nsn_linear_iterations=20,
            tet_activation_directions=directions,
        )
        # A substep moves the fingers by less than 0.3 mm. A 0.5 mm gap finds their contacts in time and leaves out the
        # particles farther from the ball, which would only enlarge the contact solve.
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.0005)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.surface = self.model.tri_indices.flatten()
        hub_vertices, self.hub_triangles = _hub_mesh(SCALE * 0.014, SCALE * 0.01)
        self.hub_rest = hub_vertices + np.array([0.0, 0.0, self.start + self.top], dtype=np.float32)
        self.hub_points = wp.array(self.hub_rest, dtype=wp.vec3)
        self.hub_triangles = wp.array(self.hub_triangles, dtype=wp.int32)
        self.lifted = 0.0

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.22, 0.12), pitch=-12.0, yaw=90.0)

    def _volumes(self, x):
        e = np.stack([x[self.tets[:, k]] - x[self.tets[:, 0]] for k in (1, 2, 3)], axis=2)
        return np.linalg.det(e) / 6.0

    def simulate(self):
        for _ in range(self.sim_substeps):
            height, opened, closed = schedule(self.sim_time + self.sim_dt)
            target = self.x0.copy()
            target[self.hub, 2] += height
            self.solver.set_pin_targets(target)
            volume = 1.0 + (OPEN - 1.0) * opened + (self.closed_volume - OPEN) * closed[self.chamber_index]
            # An activation a grows a tetrahedron to 1 - a times its rest volume, here 1 - a times its length.
            self.control.tet_activations.assign(np.where(self.chambers, 1.0 - volume, 0.0).astype(np.float32))
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()

    def test_post_step(self):
        x = self.state_0.particle_q.numpy()
        assert np.isfinite(x).all(), f"non-finite gripper positions at t={self.sim_time:.2f} s"
        volumes = self._volumes(x)
        assert (volumes > 0.0).all(), f"inverted gripper tetrahedron at t={self.sim_time:.2f} s"
        speed = np.abs(self.state_0.particle_qd.numpy()).max()
        assert speed < 1.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        ball = self.state_0.body_q.numpy()[self.ball]
        velocity = self.state_0.body_qd.numpy()[self.ball]
        ball_speed, ball_spin = np.linalg.norm(velocity[:3]), np.linalg.norm(velocity[3:])
        assert ball_speed < 1.0, f"ball speed {ball_speed:.2f} m/s at t={self.sim_time:.2f} s"
        assert ball_spin < 20.0, f"ball spin {ball_spin:.1f} rad/s at t={self.sim_time:.2f} s"
        assert ball[2] > RADIUS - 0.002, f"the ball sank {(RADIUS - ball[2]) * 1e3:.1f} mm into the ground"
        gap = np.linalg.norm(x - ball[:3], axis=1).min() - RADIUS
        assert gap > -0.5 * CONTACT_THICKNESS, f"a particle is {-gap * 1e3:.2f} mm inside the ball"
        self.lifted = max(self.lifted, float(ball[2]) - RADIUS)

    def test_final(self):
        assert self.lifted > 0.9 * LIFT, f"the ball rose by {self.lifted * 1e3:.1f} mm"

    def render(self):
        height, _, _ = schedule(self.sim_time)
        self.hub_points.assign(self.hub_rest + np.array([0.0, 0.0, height], dtype=np.float32))
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/gripper",
            self.state_0.particle_q,
            self.surface,
            color=(0.33, 0.78, 0.85),
            backface_culling=False,
        )
        self.viewer.log_mesh("/hub", self.hub_points, self.hub_triangles, color=(0.9, 0.9, 0.9))
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
