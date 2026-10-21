# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton

_SHEET = {
    "rot": wp.quat_identity(),
    "vel": wp.vec3(0.0),
    "mass": 0.02,
    "tri_ke": 5000.0,
    "tri_ka": 1.0e4,
    "edge_ke": 1.5e-3,
    "particle_radius": 0.008,
}


class TestFrictionAnchors(unittest.TestCase):
    """An 8 cm patch of 1 cm cells rests 12 mm over a sheet of 1 cm cells under gravity tilted along x, with friction
    0.5, solved with the lite operator: each step's solve leaves a little slip in the stacked contacts."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _stack(self, slope, *, box, sheet_cells=16, sheet_x=-0.08, patch_x=-0.04):
        """A sheet, pinned or lying on a box, and a patch on it, under gravity tilted by ``arctan(slope)``."""
        theta = np.arctan(slope)
        builder = newton.ModelBuilder(gravity=(9.81 * np.sin(theta), 0.0, -9.81 * np.cos(theta)))
        builder.default_shape_cfg.mu = 0.5
        if box:
            xform = wp.transform((0.0, 0.0, -0.05), wp.quat_identity())
            builder.add_shape_box(-1, xform=xform, hx=0.3, hy=0.3, hz=0.05)
        corner = wp.vec3(sheet_x, -0.08, 0.008)
        builder.add_cloth_grid(pos=corner, dim_x=sheet_cells, dim_y=16, cell_x=0.01, cell_y=0.01, **_SHEET)
        sheet = builder.particle_count
        if not box:
            for i in range(sheet):
                builder.particle_mass[i] = 0.0
        corner = wp.vec3(patch_x, -0.04, 0.02)
        builder.add_cloth_grid(pos=corner, dim_x=8, dim_y=8, cell_x=0.01, cell_y=0.01, **_SHEET)
        model = builder.finalize()
        model.particle_mu = 0.5
        return model, sheet

    def _drift(self, slope, anchors, *, box, steps=270, fps=180):
        """Mean distance [m] the sheet and the patch move down the slope from 0.5 s to the last step."""
        model, sheet = self._stack(slope, box=box)
        solver = newton.solvers.SolverFBA(
            model,
            iterations=5,
            nsn_schur_mode="lite",
            nsn_linear_iterations=10,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.012,
            particle_topological_contact_filter_threshold=1,
            friction_anchors=anchors,
        )
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.008) if box else None
        contacts = pipeline.contacts() if box else None
        s0, s1, start = model.state(), model.state(), None
        for k in range(steps):
            if box:
                pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / fps)
            s0, s1 = s1, s0
            if k + 1 == fps // 2:
                start = s0.particle_q.numpy()[:, 0]
        x = s0.particle_q.numpy()[:, 0] - start
        self.assertTrue(np.isfinite(x).all())
        return float(x[:sheet].mean()), float(x[sheet:].mean())

    def test_patch_holds_on_pinned_sheet_below_friction_angle(self):
        """Hold the patch on a pinned sheet tilted to a slope of 0.9 times the friction coefficient, which without
        anchors creeps down by millimeters in a second."""
        _, creep = self._drift(0.45, False, box=False)
        _, held = self._drift(0.45, True, box=False)
        self.assertGreater(creep, 3.0e-3)
        self.assertLess(abs(held), 5.0e-5)

    def test_sheet_and_patch_hold_on_box_below_friction_angle(self):
        """Hold a sheet on a static box and the patch on the sheet at a slope of 0.9 times the friction coefficient:
        the sheet keeps anchors on the box, the patch on the sheet. Without anchors both creep down."""
        creep = self._drift(0.45, False, box=True)
        held = self._drift(0.45, True, box=True)
        self.assertGreater(min(creep), 5.0e-4)
        self.assertLess(max(abs(v) for v in held), 5.0e-5)

    def test_patch_slides_above_friction_angle(self):
        """Slide the patch down a pinned sheet tilted beyond its friction angle by the Coulomb distance, the anchors
        trailing it."""
        slope, fps, steps = 0.8, 180, 27
        model, sheet = self._stack(slope, box=False, sheet_cells=60, sheet_x=-0.3, patch_x=-0.26)
        solver = newton.solvers.SolverFBA(
            model,
            iterations=5,
            nsn_schur_mode="lite",
            nsn_linear_iterations=10,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.012,
            particle_topological_contact_filter_threshold=1,
            friction_anchors=True,
        )
        s0, s1 = model.state(), model.state()
        x0 = s0.particle_q.numpy()[sheet:, 0].mean()
        for _ in range(steps):
            solver.step(s0, s1, None, None, 1.0 / fps)
            s0, s1 = s1, s0
        theta = np.arctan(slope)
        coulomb = 0.5 * 9.81 * (np.sin(theta) - 0.5 * np.cos(theta)) * (steps / fps) ** 2
        np.testing.assert_allclose(s0.particle_q.numpy()[sheet:, 0].mean() - x0, coulomb, rtol=0.25)

    def test_friction_anchors_need_friction(self):
        """Refuse friction anchors without friction."""
        model, _ = self._stack(0.0, box=False)
        with self.assertRaises(ValueError):
            newton.solvers.SolverFBA(model, friction=False, friction_anchors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
