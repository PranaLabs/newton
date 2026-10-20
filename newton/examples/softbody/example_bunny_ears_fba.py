# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Bunny Ears FBA
#
# A tetrahedral soft bunny hangs by the tips of its ears from two handles
# that lift it, pull the ears apart, swing it left and right, and bring the
# ears together and apart again, with self-contact throughout. Each handle
# holds one ear-tip vertex with three particle-constraint rows of the lite
# contact operator.
#
# Command: python -m newton.examples bunny_ears_fba
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
from newton.examples.pin_demo_utils import PinHandles, keyframed_handles
from newton.solvers import SolverFBA


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps, self.sim_time = 60, 0.0
        self.frame_dt, self.sim_substeps = 1.0 / self.fps, 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        stage = Usd.Stage.Open(newton.examples.get_asset("bunny_pin_fba.usd"))
        mesh = newton.TetMesh.create_from_usd(stage.GetPrimAtPath("/bunny"))
        vertices, self.tets = mesh.vertices, mesh.tet_indices.reshape(-1, 4)
        # Asset is Y-up. A 0.97 m bunny floats above the floor, held only at its ears.
        vertices = vertices * 0.6 + np.array([0.1, 0.55, 0.0])
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -2.0, 0.0))
        builder.default_particle_radius = 0.003
        builder.add_soft_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices.tolist(),
            indices=self.tets.reshape(-1).tolist(),
            density=30.0,
            k_mu=60000.0,
            k_lambda=100000.0,
            k_damp=0.0,
            particle_radius=0.003,
        )
        self.model = builder.finalize()
        self.rest = self.model.particle_q.numpy()
        # Hold one vertex per ear: a fixed-orientation patch can invert a tip element as the ear bends.
        groups = []
        for side in (self.rest[:, 0] < -0.17, self.rest[:, 0] > -0.11):
            candidates = np.flatnonzero(side & (self.rest[:, 1] > 1.35))
            tip = candidates[np.argmax(self.rest[candidates, 1])]
            groups.append(np.array([tip]))
        self.pins = PinHandles(self.rest, groups)
        self.solver = SolverFBA(
            self.model,
            iterations=16,
            nsn_schur_mode="lite",
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.018,
            particle_self_contact_gap=0.018,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.045,
            nsn_linear_iterations=16,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        a = self.pins.centers.copy()
        lifted = a + np.array([0, 0.12, 0])
        wide = lifted + np.array([[-0.16, 0.025, 0.0], [0.16, 0.025, 0.0]])
        center = lifted.mean(0)
        close = np.array([center + np.array([-0.035, 0.045, -0.015]), center + np.array([0.035, 0.045, 0.015])])
        self.keys = [
            (0, a, "REST"),
            (2, lifted, "LIFT"),
            (5, wide, "PULL EARS OUTWARD"),
            (7, wide + np.array([-0.16, 0.04, 0.10]), "SWING LEFT"),
            (9, wide + np.array([0.16, 0.04, -0.10]), "SWING RIGHT"),
            (11, lifted, "RELAX"),
            (14, close, "EARS TOGETHER"),
            (16, lifted, "OPEN AGAIN"),
            (18, lifted, "SETTLE"),
        ]
        self.phase = "REST"
        self.max_pin_error, self.max_speed = 0.0, 0.0
        self.triangles = self.model.tri_indices.flatten()
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(-1.5, 1.5, 1.95), pitch=-8.0, yaw=-53.0)

    def step(self):
        for _ in range(self.sim_substeps):
            centers, self.phase = keyframed_handles(self.sim_time + self.sim_dt, self.keys)
            self.pins.apply(self.solver, centers)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/bunny", self.state_0.particle_q, self.triangles, color=(0.86, 0.48, 0.20), backface_culling=False
        )
        self.pins.render(self.viewer, 0.026)
        self.viewer.end_frame()

    def test_post_step(self):
        p = self.state_0.particle_q.numpy()
        assert np.isfinite(p).all(), "non-finite bunny positions"
        v = p[self.tets]
        volumes = np.linalg.det(np.stack([v[:, i] - v[:, 0] for i in (1, 2, 3)], axis=2))
        assert volumes.min() > 0.0, f"inverted bunny tetrahedron at {self.sim_time:.3f}s"
        speed = float(np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max())
        error = self.pins.error(p)
        self.max_pin_error, self.max_speed = max(self.max_pin_error, error), max(self.max_speed, speed)
        assert speed < 6.0, f"bunny speed {speed:.3f} at {self.sim_time:.3f}s"
        assert error < 0.002, f"ear pin error {error:.5f}m"
        assert p[:, 1].min() > 0.10 and np.abs(p).max() < 3.0, (
            f"bunny bounds {p.min(0)} {p.max(0)} at {self.sim_time:.3f}s"
        )

    def test_final(self):
        self.test_post_step()
        assert self.model.tet_count > 0, "bunny must be volumetric"
        if self.sim_time >= 17.9:
            assert np.linalg.norm(self.state_0.particle_q.numpy() - self.rest, axis=1).mean() > 0.04


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
