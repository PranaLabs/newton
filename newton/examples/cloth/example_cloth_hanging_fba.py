# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Hanging FBA
#
# This simulation demonstrates a cloth hanging from its two top corners
# using the SolverFBA projective-dynamics solver. The cloth is pinned at
# the upper-left and upper-right corners and drapes under gravity.
#
# Command: python -m newton.examples cloth_hanging_fba
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA


class Example:
    def __init__(self, viewer, args):
        self.args = args
        self.viewer = viewer

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        dim = 32

        # Y-up world: cloth grid spans y from 2.0 (bottom row) to 2+dim*cell_y (top row).
        # Pin the two TOP corners so the cloth hangs under gravity (-Y).
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 2.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.1,
            tri_ke=5000.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=False,
        )
        # Top-row corners in the (dim+1)x(dim+1) grid: y-index = dim.
        top_left = dim * (dim + 1)
        top_right = dim * (dim + 1) + dim
        builder.particle_mass[top_left] = 0.0
        builder.particle_mass[top_right] = 0.0
        self.pins = [top_left, top_right]

        self.model = builder.finalize()
        self.solver = SolverFBA(self.model, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)
        # Toward the middle of the sheet, which hangs in the x-y plane.
        self.viewer.set_camera(pos=wp.vec3(2.0, 3.0, 3.0), pitch=-7.1, yaw=-111.8)

    def step(self):
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max()
        assert speed < 5.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"

    def test_final(self):
        # After 5 s the 1.6 m sheet hangs from its two top corners at y = 3.6 m down to y = 1.8 m.
        q, q0 = self.state_0.particle_q.numpy(), self.model.particle_q.numpy()
        assert np.all(np.isfinite(q)), "non-finite particle positions detected"
        assert np.abs(q[self.pins] - q0[self.pins]).max() < 1.0e-6, "pinned corners moved"
        assert 1.6 < q[:, 1].min() < 2.0, f"the sheet hangs down to y = {q[:, 1].min():.3f} m"
        assert q[:, 1].mean() < 3.0, f"the sheet's centroid is at y = {q[:, 1].mean():.3f} m"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
