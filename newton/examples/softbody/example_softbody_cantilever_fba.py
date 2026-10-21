# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Cantilever FBA
#
# Two soft cantilever beams, 2.4 m long and clamped at one end, sag under
# gravity and swing. They weigh 1 kg each, with a Young's modulus of 1 MPa
# and a Poisson ratio of 0.3. The near one is undamped; the far one has
# Rayleigh stiffness damping, rayleigh_beta = 2e-5 s, and comes to rest
# within a few seconds while the near one keeps swinging. SolverFBA steps
# both with ARAP tetrahedra.
#
# Command: python -m newton.examples softbody_cantilever_fba
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverFBA

LENGTH, WIDTH = 2.4, 0.16  # the beams' length and square cross-section [m]
CELLS = (26, 13, 13)  # grid cells along the length and across
HEIGHT, OFFSET = 1.4, 0.45  # the beams' axes: height and distance from the middle [m]
MASS, YOUNG, POISSON = 1.0, 1.0e6, 0.3  # each beam's mass [kg], Young's modulus [Pa] and Poisson ratio
BETA = (0.0, 2.0e-5)  # Rayleigh stiffness damping of the near and the far beam [s]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt, self.sim_substeps = 1.0 / 50.0, 4  # steps of 5 ms
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        cell = (LENGTH / CELLS[0], WIDTH / CELLS[1], WIDTH / CELLS[2])
        # Every grid particle weighs density * cell volume; the clamped end, at x = 0 in the grid, weighs nothing.
        density = MASS / (CELLS[0] * (CELLS[1] + 1) * (CELLS[2] + 1) * np.prod(cell))
        k_mu = YOUNG / (2.0 * (1.0 + POISSON))
        k_lambda = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
        self.beams, beta = [], []
        for y, b, color in zip((-OFFSET, OFFSET), BETA, ((0.9, 0.55, 0.2), (0.25, 0.5, 0.85)), strict=True):
            start = builder.particle_count
            builder.add_soft_grid(
                pos=wp.vec3(-0.5 * LENGTH, y - 0.5 * WIDTH, HEIGHT - 0.5 * WIDTH),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=CELLS[0],
                dim_y=CELLS[1],
                dim_z=CELLS[2],
                cell_x=cell[0],
                cell_y=cell[1],
                cell_z=cell[2],
                density=density,
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=0.0,
                fix_left=True,
                color=color,
            )
            self.beams.append(np.arange(start, builder.particle_count))
            beta.append(np.full(builder.particle_count - start, b))
        builder.add_ground_plane()
        self.model = builder.finalize()

        # The damping coefficient of each particle: each beam's elements take their particles' value.
        self.solver = SolverFBA(self.model, iterations=10, rayleigh_beta=np.concatenate(beta))
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        # The particle nearest the middle of each beam's free end.
        x0 = self.model.particle_q.numpy()
        self.tips = []
        for beam, y in zip(self.beams, (-OFFSET, OFFSET), strict=True):
            end = beam[np.isclose(x0[beam, 0], x0[beam, 0].max())]
            self.tips.append(end[np.argmin(np.linalg.norm(x0[end, 1:] - [y, HEIGHT], axis=1))])
        self.speeds = []

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.0, -3.2, 2.1), pitch=-14.0, yaw=107.0)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def test_post_step(self):
        x, v = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        assert np.isfinite(x).all(), f"non-finite positions at t={self.sim_time:.2f} s"
        speeds = [float(np.linalg.norm(v[beam], axis=1).max()) for beam in self.beams]
        assert max(speeds) < 10.0, f"beam speeds {speeds} m/s at t={self.sim_time:.2f} s"
        self.speeds.append(speeds)

    def test_final(self):
        x = self.state_0.particle_q.numpy()
        sag = [HEIGHT - x[tip, 2] for tip in self.tips]
        assert min(sag) > 0.05, f"the beams' free ends sag by {sag} m"
        if self.sim_time > 5.0:
            # over the last second the damped beam has nearly come to rest while the undamped one swings
            last = np.array(self.speeds[-50:])
            assert last[:, 1].max() < 0.1 * last[:, 0].max(), f"last-second peak speeds {last.max(axis=0)} m/s"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
