# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Hanging FBA
#
# Volumetric soft body (tetrahedral grid) hanging from a pinned left face,
# simulated with SolverFBA projective-dynamics (ARAP tet stretching).
#
# Command: uv run -m newton.examples softbody_hanging_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()

        dim = 4
        cell = 0.1

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 1.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            dim_z=dim,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            density=1.0e3,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            fix_left=True,
        )

        self.model = builder.finalize()
        self.solver = SolverFBA(self.model, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)

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
        assert speed < 10.0, f"particle speed {speed:.2f} m/s at t={self.sim_time:.2f} s"

    def test_final(self):
        # After 5 s the soft 0.4 m cube, held by its left face, droops: its free face hangs 0.34 m low and
        # moves at 0.2 m/s.
        q, q0 = self.state_0.particle_q.numpy(), self.model.particle_q.numpy()
        assert np.all(np.isfinite(q)), "non-finite particle positions"
        fixed = self.model.particle_inv_mass.numpy() == 0.0
        assert np.abs(q[fixed] - q0[fixed]).max() < 1.0e-5, "the held face moved"
        tip = np.isclose(q0[:, 0], q0[:, 0].max())
        sag = float(np.mean(q0[tip, 2] - q[tip, 2]))
        assert 0.2 < sag < 0.45, f"the free face sags {sag:.3f} m"
        speed = np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max()
        assert speed < 1.0, f"the body still moves at {speed:.2f} m/s"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
