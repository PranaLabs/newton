# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton

_YOUNG, _POISSON = 1.0e5, 0.3


def _lame(young=_YOUNG, poisson=_POISSON):
    return young / (2.0 * (1.0 + poisson)), young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))


def _cube(builder, size=0.1, cells=3):
    """A free soft cube of ``cells`` cells per side at the origin."""
    mu, lam = _lame()
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=cells,
        dim_y=cells,
        dim_z=cells,
        cell_x=size / cells,
        cell_y=size / cells,
        cell_z=size / cells,
        density=1000.0,
        k_mu=mu,
        k_lambda=lam,
        k_damp=0.0,
    )


def _volume(x, tets):
    e = np.stack([x[tets[:, k]] - x[tets[:, 0]] for k in (1, 2, 3)], axis=2)
    return np.linalg.det(e).sum() / 6.0


class TestTetActivation(unittest.TestCase):
    """Tetrahedral activations of SolverFBA, which grow each tetrahedron to ``1 - a`` times its rest volume,
    isotropically or along a direction."""

    def _settle(self, model, activations, steps=150, dt=0.01, stretching_model="corotational", **kwargs):
        solver = newton.solvers.SolverFBA(model, iterations=10, stretching_model=stretching_model, **kwargs)
        control = model.control()
        control.tet_activations.assign(np.broadcast_to(activations, (model.tet_count,)).astype(np.float32))
        s0, s1 = model.state(), model.state()
        for _ in range(steps):
            solver.step(s0, s1, control, None, dt)
            s0, s1 = s1, s0
        return s0, solver, control

    def _free_cube(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        _cube(builder)
        return builder.finalize()

    def test_zero_activations_change_nothing(self):
        """Step a sagging cantilever with zero activations exactly as without a control."""
        builder = newton.ModelBuilder()
        mu, lam = _lame()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=6,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1000.0,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0.0,
            fix_left=True,
        )
        model = builder.finalize()
        for stretching_model in ("arap", "corotational", "neohookean"):
            results = []
            for control in (None, model.control()):
                solver = newton.solvers.SolverFBA(model, iterations=10, stretching_model=stretching_model)
                s0, s1 = model.state(), model.state()
                for _ in range(20):
                    solver.step(s0, s1, control, None, 0.01)
                    s0, s1 = s1, s0
                results.append(s0.particle_q.numpy())
            np.testing.assert_array_equal(results[1], results[0], err_msg=stretching_model)

    def test_free_cube_grows_to_its_activation(self):
        """Bring a free cube with one activation to rest at ``1 - a`` times its volume and at ``(1 - a)^(1/3)`` times
        its edge lengths, for every stretching model and for growth and shrinkage."""
        model = self._free_cube()
        x0, tets = model.particle_q.numpy(), model.tet_indices.numpy().reshape(-1, 4)
        corner = np.argmax(x0.sum(axis=1))
        for stretching_model in ("arap", "corotational", "neohookean"):
            for a in (-0.5, 0.3):
                state, _, _ = self._settle(model, a, stretching_model=stretching_model, rayleigh_alpha=20.0)
                x = state.particle_q.numpy()
                ratio = _volume(x, tets) / _volume(x0, tets)
                self.assertAlmostEqual(ratio, 1.0 - a, delta=2.0e-3, msg=f"{stretching_model} a={a}")
                # Isotropic: the diagonal from the first corner to the opposite one scales by s.
                scale = np.linalg.norm(x[corner] - x[0]) / np.linalg.norm(x0[corner] - x0[0])
                self.assertAlmostEqual(scale, (1.0 - a) ** (1.0 / 3.0), delta=1.0e-3, msg=f"{stretching_model} a={a}")

    def test_directed_growth_stretches_along_the_direction(self):
        """Bring a free cube whose tetrahedra grow along x to rest at ``1 - a`` times its length along x and at its
        rest width and height, for every stretching model; the directions need not be unit vectors."""
        model = self._free_cube()
        x0, tets = model.particle_q.numpy(), model.tet_indices.numpy().reshape(-1, 4)
        directions = np.tile([2.0, 0.0, 0.0], (model.tet_count, 1))
        for stretching_model in ("arap", "corotational", "neohookean"):
            state, _, _ = self._settle(
                model,
                -0.5,
                stretching_model=stretching_model,
                rayleigh_alpha=20.0,
                tet_activation_directions=directions,
            )
            x = state.particle_q.numpy()
            extents = (x.max(axis=0) - x.min(axis=0)) / (x0.max(axis=0) - x0.min(axis=0))
            np.testing.assert_allclose(extents, [1.5, 1.0, 1.0], atol=2.0e-3, err_msg=stretching_model)
            self.assertAlmostEqual(_volume(x, tets) / _volume(x0, tets), 1.5, delta=3.0e-3, msg=stretching_model)

    def test_zero_directions_grow_isotropically(self):
        """Grow tetrahedra with zero directions exactly as without directions."""
        model = self._free_cube()
        plain, _, _ = self._settle(model, -0.5, steps=20)
        zero, _, _ = self._settle(model, -0.5, steps=20, tet_activation_directions=np.zeros((model.tet_count, 3)))
        np.testing.assert_array_equal(zero.particle_q.numpy(), plain.particle_q.numpy())

    def test_refuse_misshaped_directions(self):
        """Refuse growth directions without one finite row of 3 per tetrahedron."""
        model = self._free_cube()
        for directions in (np.zeros((model.tet_count, 2)), np.full((model.tet_count, 3), np.nan)):
            with self.assertRaises(ValueError):
                newton.solvers.SolverFBA(model, tet_activation_directions=directions)

    def test_activated_top_layer_bends_a_cantilever(self):
        """Curl a clamped bilayer strip toward its passive side when its top layer grows, and toward the top when it
        shrinks: the bending of a bimetal strip or a pneumatic bending actuator."""
        tips = {}
        for a in (-0.3, 0.0, 0.3):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            mu, lam = _lame()
            builder.add_soft_grid(
                pos=wp.vec3(0.0, 0.0, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=8,
                dim_y=1,
                dim_z=2,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.025,
                density=1000.0,
                k_mu=mu,
                k_lambda=lam,
                k_damp=0.0,
                fix_left=True,
            )
            model = builder.finalize()
            x0, tets = model.particle_q.numpy(), model.tet_indices.numpy().reshape(-1, 4)
            top = x0[tets].mean(axis=1)[:, 2] > 1.025
            activations = np.where(top, a, 0.0)
            state, _, _ = self._settle(model, activations, steps=200, rayleigh_alpha=20.0)
            tip = np.isclose(x0[:, 0], 0.4)
            tips[a] = state.particle_q.numpy()[tip, 2].mean() - x0[tip, 2].mean()
        self.assertAlmostEqual(tips[0.0], 0.0, delta=1.0e-6)
        self.assertLess(tips[-0.3], -0.05, "a growing top layer bends the strip down")
        self.assertGreater(tips[0.3], 0.05, "a shrinking top layer bends the strip up")

    def test_activations_change_between_steps(self):
        """Take up new activations at the next step, with the system as factored: a cube that settled at one
        activation settles at the next one."""
        model = self._free_cube()
        x0, tets = model.particle_q.numpy(), model.tet_indices.numpy().reshape(-1, 4)
        state, solver, control = self._settle(model, -0.2, rayleigh_alpha=20.0)
        control.tet_activations.fill_(-0.6)
        s0, s1 = state, model.state()
        for _ in range(150):
            solver.step(s0, s1, control, None, 0.01)
            s0, s1 = s1, s0
        self.assertAlmostEqual(_volume(s0.particle_q.numpy(), tets) / _volume(x0, tets), 1.6, delta=2.0e-3)

    def test_refuse_misshaped_activations(self):
        """Refuse activations without one value per tetrahedron."""
        model = self._free_cube()
        solver = newton.solvers.SolverFBA(model)
        control = model.control()
        control.tet_activations = wp.zeros(model.tet_count + 1, dtype=wp.float32)
        with self.assertRaises(ValueError):
            solver.step(model.state(), model.state(), control, None, 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
