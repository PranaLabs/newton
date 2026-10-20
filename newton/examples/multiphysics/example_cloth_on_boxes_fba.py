# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth on Boxes FBA
#
# A cloth drops onto two stacked boxes while a plank hinged next to them
# tips over onto the stack. SolverFBA integrates the cloth and the rigid
# bodies in one step: particle and face contacts push on the boxes, so box
# edges stay out of the cloth, the boxes rest on each other and on the
# ground, and the hinge holds the plank on its axis while it lies on the
# boxes.
#
# Command: python -m newton.examples cloth_on_boxes_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

HINGE = wp.vec3(1.1, 0.0, 0.05)
PLANK_HALF = wp.vec3(0.04, 0.3, 0.6)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
        builder.add_ground_plane(cfg=cfg)

        # Two stacked boxes, the upper one turned about the vertical axis.
        lower = builder.add_body(xform=wp.transform((0.0, 0.0, 0.25), wp.quat_identity()))
        builder.add_shape_box(lower, hx=0.4, hy=0.4, hz=0.25, cfg=cfg)
        turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4)
        upper = builder.add_body(xform=wp.transform((0.0, 0.0, 0.65), turn))
        builder.add_shape_box(upper, hx=0.25, hy=0.25, hz=0.15, cfg=cfg)

        # A plank hinged at its lower end, leaning toward the stack.
        lean = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.15)
        plank = builder.add_link(
            xform=wp.transform(HINGE + wp.quat_rotate(lean, wp.vec3(0.0, 0.0, PLANK_HALF[2])), lean)
        )
        builder.add_shape_box(plank, hx=PLANK_HALF[0], hy=PLANK_HALF[1], hz=PLANK_HALF[2], cfg=cfg)
        hinge = builder.add_joint_revolute(
            -1,
            plank,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(HINGE, wp.quat_identity()),
            child_xform=wp.transform((0.0, 0.0, -PLANK_HALF[2]), wp.quat_identity()),
        )
        builder.add_articulation([hinge])

        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 1.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=16,
            dim_y=16,
            cell_x=1.0 / 16.0,
            cell_y=1.0 / 16.0,
            mass=0.002,
            tri_ke=2500.0,
            tri_ka=5.0e3,
            edge_ke=0.1,
            particle_radius=3.0e-3,  # the cloth's contact thickness
        )

        self.model = builder.finalize()
        # Five LG iterations and at most ten conjugate residual iterations per contact solve.
        self.solver = SolverFBA(self.model, iterations=5, face_contacts=True, nsn_linear_iterations=10)
        self.pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.147)
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.2, -2.6, 1.8), pitch=-18.0, yaw=140.0)

    def simulate(self):
        # SolverFBA reads contacts back to the host every step, so the loop is not graph-captured.
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
        # The plank lands at about 2.5 rad/s; nothing in the scene moves faster.
        speed = np.abs(self.state_0.body_qd.numpy()).max()
        assert speed < 5.0, f"body speed {speed:.2f} at t={self.sim_time:.3f} s"

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(particle_q)) and np.all(np.isfinite(body_q)), "non-finite state"
        assert particle_q[:, 2].min() > -0.01, f"cloth below the ground: z_min={particle_q[:, 2].min():.4f}"
        assert particle_q[:, 2].max() > 0.5, "cloth fell off the stack"
        # The lower box stays on the ground and the hinge stays on its axis.
        np.testing.assert_allclose(body_q[0, :3], [0.0, 0.0, 0.25], atol=0.02)
        hinge = wp.transform_point(wp.transform(*np.split(body_q[2], [3])), wp.vec3(0.0, 0.0, -PLANK_HALF[2]))
        np.testing.assert_allclose(hinge, HINGE, atol=1e-3)
        assert np.abs(body_qd).max() < 1.0, f"bodies still moving: {np.abs(body_qd).max():.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
