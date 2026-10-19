# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Gripper FBA
#
# Two soft fingers hang from a palm on either side of a box on the ground.
# A cable runs down the inner side of each finger, embedded in its
# tetrahedra; pulling it shorter curls the fingers onto the box, the palm
# lifts the box and sets it down, and the released fingers open, as in
# SOFA SoftRobots' cable-driven grippers. SolverFBA integrates the box as a
# rigid body and solves the cable rows with the contacts on it.
#
# Command: uv run -m newton.examples cable_gripper_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

CELL = 0.005  # [m], fingers of 4 x 4 x 20 cells
BOTTOM = 0.003  # finger tips over the ground [m]
BOX = 0.02  # half extent of the box [m]
GAP = 0.008  # between the box and each finger [m]
LIFT = 0.05  # [m]
CONTACT_THICKNESS = 0.002


def schedule(t):
    """Cable length over the rest length and palm lift [m] in a 7.5 s cycle: close, lift, hold, lower, open."""
    t = t % 7.5
    close = min(t / 1.5, 1.0, max(0.0, 7.0 - t))
    lift = min(max(t - 2.0, 0.0) / 1.5, 1.0, max(0.0, (6.0 - t) / 1.5))
    return 1.0 - 0.15 * close, LIFT * lift


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
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        # A 64 g box; silicone grips with a friction coefficient near 1.
        self.box = builder.add_body(xform=wp.transform((0.0, 0.0, BOX), wp.quat_identity()))
        cfg = newton.ModelBuilder.ShapeConfig(mu=1.0, density=1000.0)
        builder.add_shape_box(self.box, hx=BOX, hy=BOX, hz=BOX, cfg=cfg)
        for side in (-1.0, 1.0):
            inner = side * (BOX + GAP)
            builder.add_soft_grid(
                pos=wp.vec3(min(inner, inner + side * 4 * CELL), -2 * CELL, BOTTOM),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=4,
                dim_y=4,
                dim_z=20,
                cell_x=CELL,
                cell_y=CELL,
                cell_z=CELL,
                density=1100.0,
                k_mu=mu,
                k_lambda=lam,
                k_damp=0.0,
                particle_radius=0.002,
            )
        self.model = builder.finalize()
        # The palm holds the top faces and moves them through set_pin_targets.
        self.x0 = self.model.particle_q.numpy()
        self.palm = np.isclose(self.x0[:, 2], BOTTOM + 20 * CELL)
        mass = np.where(self.palm, 0.0, self.model.particle_mass.numpy())
        self.model.particle_mass.assign(mass)
        self.model.particle_inv_mass.assign(np.divide(1.0, mass, out=np.zeros_like(mass), where=mass > 0))

        # Each cable: nine 1 cm segments 3 mm inside the inner face.
        z = np.arange(BOTTOM + 0.095, BOTTOM, -0.01)
        self.cables = []
        for side in (-1.0, 1.0):
            points = np.array([[side * (BOX + GAP + 0.003), 0.0, h] for h in z])
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
            # The box squeezed on the ground closes contact loops whose Newton steps rock the light box by
            # 41 rad/s at the default 1e-2; damping them by 0.1 leaves 1.3 rad/s.
            contact_regularization=0.1,
        )
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.01)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.07, -0.3, 0.1), pitch=-6.0, yaw=104.0)
        self.lifted = 0.0

    def simulate(self):
        # Ramp the cables like RealSim's cable controller and move the palm.
        ratio, lift = schedule(self.sim_time)
        target = self.x0.copy()
        target[self.palm, 2] += lift
        self.solver.set_pin_targets(target)
        rows = []
        for cable in self.cables:
            cable.length = ratio * cable.rest_length
            rows.append(cable.rows())
        self.solver.set_particle_constraints(*(np.concatenate([r[k] for r in rows]) for k in range(3)))
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
        body_q, body_qd = self.state_0.body_q.numpy()[self.box], self.state_0.body_qd.numpy()[self.box]
        assert np.abs(body_qd).max() < 5.0, f"box velocity {body_qd} at t={self.sim_time:.2f} s"
        # Signed distance of every particle to the box.
        rotation = np.array(wp.quat_to_matrix(wp.quat(*body_q[3:]))).reshape(3, 3)
        d = np.abs((self.state_0.particle_q.numpy() - body_q[:3]) @ rotation) - BOX
        gap = (np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)).min()
        assert gap > 0.5 * CONTACT_THICKNESS, f"a particle is {gap * 1e3:.2f} mm from the box"
        self.lifted = max(self.lifted, float(body_q[2]) - BOX)

    def test_final(self):
        assert self.lifted > 0.9 * LIFT, f"the box rose by {self.lifted * 1e3:.1f} mm"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
