# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Shirt Pins FBA
#
# A T-shirt hangs from two shoulder points that two handles move: it sways,
# tilts, gathers its shoulders together, crosses its front over its back and
# opens again, with cloth self-contact throughout; light Rayleigh stiffness
# damping calms its swing after each move. Each handle holds one vertex with
# three particle-constraint rows of the lite contact operator. Two substeps
# of five iterations leave some triangles crossed where the cloth folds
# sharply; the test checks every frame that at most 400 pairs intersect.
#
# Command: python -m newton.examples shirt_pins_fba
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.examples.cloth.pin_handle_utils import PinHandles, SurfaceIntersectionCheck, keyframed_handles
from newton.solvers import SolverFBA


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps, self.sim_time = 60, 0.0
        self.frame_dt, self.sim_substeps = 1.0 / self.fps, 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        stage = Usd.Stage.Open(newton.examples.get_asset("shirt_pin_fba.usd"))
        mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/shirt"))
        vertices, triangles = mesh.vertices.copy(), mesh.indices.reshape(-1, 3)
        vertices[:, 1] += 0.65
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0))
        builder.default_particle_radius = 0.004
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices.tolist(),
            indices=triangles.reshape(-1).tolist(),
            density=0.43,  # About 302 g over the shirt surface.
            tri_ke=80.0,  # FBA's ARAP weight is 2 * tri_ke = 160 per unit area.
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.000005,
            edge_kd=0.0,
            particle_radius=0.004,
        )
        self.model = builder.finalize()
        self.model.edge_rest_angle.zero_()
        self.model.particle_mu = 0.25
        self.rest = self.model.particle_q.numpy()
        groups = []
        for sign in (-1, 1):
            candidates = np.flatnonzero((sign * self.rest[:, 0] > 0.13) & (sign * self.rest[:, 0] < 0.23))
            tip = candidates[np.argmax(self.rest[candidates, 1])]
            groups.append(np.array([tip]))
        self.pins = PinHandles(self.rest, groups)
        self.solver = SolverFBA(
            self.model,
            iterations=5,
            nsn_schur_mode="lite",
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.009,
            particle_self_contact_gap=0.016,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.009,
            nsn_linear_iterations=10,
            # Light stiffness damping [s] halves the swing left after each move of the handles.
            rayleigh_beta=1.0e-4,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        # Contacts, though the shirt touches no shape, let the solver assemble its self contacts on the device;
        # a step without contacts takes the host path.
        self.contacts = newton.CollisionPipeline(self.model).contacts()
        a = self.pins.centers.copy()
        lift = a + np.array([0, 0.12, 0])
        mid = lift.mean(0)
        gather = np.array([mid + np.array([-0.045, 0, 0.02]), mid + np.array([0.045, 0, -0.02])])
        crossed = np.array([mid + np.array([0.105, 0.025, 0.09]), mid + np.array([-0.105, -0.025, -0.09])])
        self.keys = [
            (0, a, "HANG"),
            (2, a, "HANG / GRAVITY"),
            (3.5, lift + np.array([-0.15, 0.0, 0.07]), "SWAY LEFT"),
            (5, lift + np.array([0.15, 0.0, -0.07]), "SWAY RIGHT"),
            (6, lift + np.array([0.15, 0.0, -0.07]), "SWAY / SETTLE"),
            (8, lift + np.array([[-0.035, 0.10, 0.04], [0.035, -0.10, -0.04]]), "TILT"),
            (10, lift, "OPEN"),
            (13, gather, "GATHER / SELF CONTACT"),
            (16, crossed, "CROSS / FRONT AND BACK"),
            (18, gather, "UNCROSS"),
            (20, lift, "SPREAD"),
            (22, lift, "SETTLE"),
        ]
        self._surface_check = None
        self.triangles = self.model.tri_indices.flatten()
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(0.65, 1.21, 1.4), pitch=-8.0, yaw=-115.0)

    def step(self):
        for _ in range(self.sim_substeps):
            centers = keyframed_handles(self.sim_time + self.sim_dt, self.keys)
            self.pins.apply(self.solver, centers)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/shirt",
            self.state_0.particle_q,
            self.triangles,
            color=(0.12, 0.56, 0.68),
            roughness=0.95,
            metallic=0.0,
            backface_culling=False,
        )
        self.pins.render(self.viewer, 0.018)
        self.viewer.end_frame()

    def test_post_step(self):
        p = self.state_0.particle_q.numpy()
        assert np.isfinite(p).all(), "non-finite shirt positions"
        if self._surface_check is None:
            self._surface_check = SurfaceIntersectionCheck(self.model)
        intersections = self._surface_check.intersections(self.state_0.particle_q)
        assert intersections <= 400, f"{intersections} intersecting shirt triangle pairs at {self.sim_time:.3f}s"
        speed = float(np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max())
        error = self.pins.error(p)
        assert speed < 5.0, f"shirt speed {speed:.3f} at {self.sim_time:.3f}s"
        assert error < 0.001, f"shirt pin error {error:.5f}m"
        assert p[:, 1].min() > 0.10 and np.abs(p).max() < 2.5, "shirt escaped its stage"

    def test_final(self):
        self.test_post_step()
        if self.sim_time >= self.keys[-1][0] - 0.1:
            p = self.state_0.particle_q.numpy()
            assert p[:, 1].min() < self.pins.positions[:, 1].min() - 0.35, "shirt did not drape"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
