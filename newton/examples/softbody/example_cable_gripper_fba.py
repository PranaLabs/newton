# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Gripper FBA
#
# Two soft fingers hang from a fixed palm on either side of a rod. A cable
# runs down the inner side of each finger, embedded in its tetrahedra;
# pulling it shorter curls the finger around the rod, and releasing it
# lets the finger straighten, as in SOFA SoftRobots' cable-driven fingers.
# SolverFBA solves the cable rows with the contacts on the rod.
#
# Command: uv run -m newton.examples cable_gripper_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

CELL = 0.005  # [m], fingers of 4 x 4 x 20 cells
TOP = 0.2  # palm height [m]
ROD = (0.115, 0.02, 0.04)  # axis height, radius, half length along y [m]
CONTACT_THICKNESS = 0.002


def cable_ratio(t):
    """Commanded cable length over the rest length: pull for 1.5 s, hold 1 s, release for 1.5 s, rest 0.5 s."""
    t = t % 4.5
    pull = min(t / 1.5, 1.0, max(0.0, (4.0 - t) / 1.5))
    return 1.0 - 0.18 * pull


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # Silicone-like fingers: young 0.3 MPa, poisson 0.45, 1100 kg/m^3.
        young, poisson = 3.0e5, 0.45
        mu = young / (2.0 * (1.0 + poisson))
        lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
        along_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)
        rod = wp.transform((0.0, 0.0, ROD[0]), along_y)
        builder.add_shape_capsule(-1, xform=rod, radius=ROD[1], half_height=ROD[2], cfg=cfg)
        for side in (-1.0, 1.0):
            inner = side * (ROD[1] + 0.01)
            builder.add_soft_grid(
                wp.vec3(min(inner, inner + side * 4 * CELL), -2 * CELL, TOP - 20 * CELL),
                wp.quat_identity(),
                wp.vec3(0.0),
                *(4, 4, 20, CELL, CELL, CELL, 1100.0, mu, lam, 0.0),
                particle_radius=0.002,
            )
        self.model = builder.finalize()
        # The palm holds the top faces.
        x = self.model.particle_q.numpy()
        mass = np.where(np.isclose(x[:, 2], TOP), 0.0, self.model.particle_mass.numpy())
        self.model.particle_mass.assign(mass)
        self.model.particle_inv_mass.assign(np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0))

        # Each cable: nine 1 cm segments 3 mm inside the inner face.
        z = np.arange(TOP - 0.005, TOP - 20 * CELL, -0.01)
        self.cables = []
        for side in (-1.0, 1.0):
            points = np.array([[side * (ROD[1] + 0.013), 0.0, h] for h in z])
            self.cables.append(SolverFBA.Cable(self.model, np.stack([points[:-1], points[1:]], axis=1), min_ratio=0.8))

        self.solver = SolverFBA(
            self.model,
            iterations=10,
            stretching_model="neohookean",
            mu=mu,
            lam=lam,
            face_contacts=True,
            soft_contact_thickness=CONTACT_THICKNESS,
            face_contact_margin=0.01,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.01)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.07, -0.28, 0.2), pitch=-12.0, yaw=104.0)

    def simulate(self):
        # Ramp the cables like RealSim's controller: a sudden pull swings the chained segments slowly.
        ratio = cable_ratio(self.sim_time)
        rows = []
        for cable in self.cables:
            cable.length = ratio * cable.rest_length
            rows.append(cable.rows(self.state_0.particle_q))
        self.solver.set_particle_constraints(*(np.concatenate([r[k] for r in rows]) for k in range(4)))
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        speed = np.abs(self.state_0.particle_qd.numpy()).max()
        assert speed < 1.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"
        x = self.state_0.particle_q.numpy()
        gap = np.hypot(x[:, 0], x[:, 2] - ROD[0]).min() - ROD[1]
        assert gap > 0.5 * CONTACT_THICKNESS, f"a particle is {gap * 1e3:.2f} mm from the rod"

    def test_final(self):
        # After a full cycle the released fingers hang straight again.
        x, x0 = self.state_0.particle_q.numpy(), self.model.particle_q.numpy()
        assert np.all(np.isfinite(x)), "non-finite particle positions"
        assert np.abs(x - x0).max() < 5e-3, f"fingers off their rest shape by {np.abs(x - x0).max() * 1e3:.1f} mm"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
