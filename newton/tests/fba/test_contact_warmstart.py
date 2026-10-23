# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contact warm start of SolverFBA across steps."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import warm_start as WS

_NO_KEY = np.iinfo(np.int64).max


def _begin(group, keys):
    """Give ``group`` this step's keys."""
    keys = np.asarray(keys, dtype=np.int64)

    def write(out_keys, order):
        wp.launch(WS.index_keys_kernel, len(keys), [wp.array(keys, dtype=wp.int64)], [out_keys, order])

    group.begin(len(keys), write)


def _frames(t1, t2):
    return wp.array(np.asarray(t1, dtype=np.float32), dtype=wp.vec3), wp.array(
        np.asarray(t2, dtype=np.float32), dtype=wp.vec3
    )


class TestRowGroup(unittest.TestCase):
    """The previous step's rows, matched by key."""

    def setUp(self):
        scope = wp.ScopedDevice("cpu")
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_contacts_carry_over_by_key(self):
        """Start each contact from the previous step's contact of its key, with the friction force in the new
        tangents; new keys and keys that occur twice in either step start from zero."""
        group = WS.RowGroup(3, "cpu")
        lam, omega = np.arange(1.0, 16.0), np.linspace(0.1, 1.5, 15)
        e = np.eye(3)
        _begin(group, [30, 10, 20, 50, 50])
        group.store(wp.array(lam), wp.array(omega), 0, _frames(np.tile(e[0], (5, 1)), np.tile(e[1], (5, 1))))

        # The tangents of key 30 turn by 90 degrees about its normal: t1 becomes e1, t2 becomes -e0.
        t1, t2 = np.tile(e[0], (6, 1)), np.tile(e[1], (6, 1))
        t1[2], t2[2] = e[1], -e[0]
        _begin(group, [10, 40, 30, 20, 20, 50])
        out_lam, out_omega = wp.zeros(21, dtype=wp.float64), wp.zeros(21, dtype=wp.float64)
        group.restore(out_lam, out_omega, 3, _frames(t1, t2))

        got, prev = out_lam.numpy()[3:].reshape(6, 3), lam.reshape(5, 3)
        np.testing.assert_allclose(got[0], prev[1])  # key 10
        np.testing.assert_allclose(got[1], 0.0)  # key 40 is new
        np.testing.assert_allclose(got[2], [prev[0, 0], prev[0, 2], -prev[0, 1]], rtol=1e-6)  # key 30
        np.testing.assert_allclose(got[3:5], 0.0)  # key 20 twice in this step
        np.testing.assert_allclose(got[5], 0.0)  # key 50 twice in the previous step
        np.testing.assert_allclose(out_lam.numpy()[:3], 0.0)  # rows before the group's
        np.testing.assert_allclose(out_omega.numpy()[3:6], omega[3:6])

    def test_rows_carry_over_by_key(self):
        """Start single rows, such as joint rows, from the previous step's row of their key."""
        group = WS.RowGroup(1, "cpu")
        _begin(group, [4, 5, 9])
        group.store(wp.array([1.0, 2.0, 3.0]), wp.array([0.5, 0.6, 0.7]), 0)
        _begin(group, [5, 6, 9])
        lam, omega = wp.zeros(3, dtype=wp.float64), wp.zeros(3, dtype=wp.float64)
        group.restore(lam, omega, 0)
        np.testing.assert_array_equal(lam.numpy(), [2.0, 0.0, 3.0])
        np.testing.assert_array_equal(omega.numpy(), [0.6, 0.0, 0.7])

    def test_particle_contact_keys(self):
        """Key particle contacts by particle and shape; self contacts and contacts on several particles get none."""
        particle = wp.array([3, 3, 5, 7], dtype=wp.int32)
        shape = wp.array([1, 2, 1, -1], dtype=wp.int32)
        extra = wp.array([[-1, -1], [-1, -1], [8, 9], [4, 6]], dtype=wp.int32)
        keys, order = wp.empty(4, dtype=wp.int64), wp.empty(4, dtype=wp.int32)
        wp.launch(WS.particle_contact_keys_kernel, 4, [particle, shape, extra, 1, 4], [keys, order])
        np.testing.assert_array_equal(keys.numpy(), [3 * 4 + 1, 3 * 4 + 2, _NO_KEY, _NO_KEY])
        np.testing.assert_array_equal(order.numpy(), np.arange(4))

    def test_rigid_contact_keys(self):
        """Key rigid contacts by shape pair and their rank among the pair's contacts."""
        keys = WS.rigid_contact_keys(np.array([[0, 2], [0, 2], [0, 2], [1, 2]]), 3)
        np.testing.assert_array_equal(keys, np.array([2, 2, 2, 5]) * (1 << 20) + [0, 1, 2, 0])


class TestSolverFBAContactWarmStart(unittest.TestCase):
    """Contacts of resting cloth start each step from the forces they ended the previous one with."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    @staticmethod
    def _cloth(builder, z):
        grid = {"dim_x": 4, "dim_y": 4, "cell_x": 0.1, "cell_y": 0.1, "tri_ke": 500.0, "tri_ka": 1.0e3, "edge_ke": 1.0}
        builder.add_cloth_grid(
            pos=wp.vec3(-0.2, -0.2, z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=0.05,
            particle_radius=0.0,
            **grid,
        )

    def _restored_rows(self, model, rows, steps=60):
        """The rows ``rows(solver)`` at the end of the step before the last, and as the last step's warm start
        restores them, when the two steps have the same contacts."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        solver = SolverFBA(model, iterations=10, contact_warmstart=True)
        s0, s1 = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
        contacts = pipeline.contacts()
        restore, restored = solver._restore_warm_start, []

        def keep():
            restore()
            restored.append([a.numpy().copy() for a in rows(solver)])

        solver._restore_warm_start = keep
        ended, particles = [], []
        for _ in range(steps):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 0.005)
            s0, s1 = s1, s0
            ended.append([a.numpy().copy() for a in rows(solver)])
            particles.append(solver._contact_particle_d.numpy()[: solver._contact_count].copy())
        np.testing.assert_array_equal(particles[-1], particles[-2])
        return solver, ended[-2], restored[-1]

    def test_resting_cloth_starts_from_its_last_forces(self):
        """Start the contacts of a cloth resting on a static box from the multipliers and weights they ended the
        previous step with."""
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5)
        self._cloth(builder, 1.005)
        model = builder.finalize()

        def rows(solver):
            n = 3 * solver._contact_count
            return solver._lam_coulomb_persistent_d[:n], solver._omega_coulomb_persistent_d[:n]

        solver, ended, restored = self._restored_rows(model, rows)
        self.assertFalse(solver._two_way)
        self.assertGreater(np.abs(ended[0]).max(), 0.0)
        for a, b in zip(restored, ended, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6 * np.abs(b).max())

    def test_cloth_on_free_box_starts_from_its_last_forces(self):
        """Start the particle contacts of a cloth on a free box and the box's ground contacts, solved together, from
        the rows they ended the previous step with."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(box, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))
        self._cloth(builder, 1.005)
        model = builder.finalize()

        def rows(solver):
            return solver._rc_lam_d, solver._rc_omega_d

        solver, ended, restored = self._restored_rows(model, rows)
        self.assertTrue(solver._two_way)
        self.assertGreater(solver._rc_counts[1], 0)
        for a, b in zip(restored, ended, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6 * np.abs(b).max())

    def test_box_rests_at_its_height_with_one_iteration(self):
        """Hold a box on the ground at its height with one iteration per step: the warm start begins each step at the
        contact forces that carry the box, which one Newton step from zero falls short of."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        sink = []
        for warm in (False, True):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            builder.add_shape_box(builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity())))
            model = builder.finalize()
            solver = SolverFBA(model, iterations=1, contact_warmstart=warm)
            s0, s1 = model.state(), model.state()
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            for _ in range(100):
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, None, contacts, 0.01)
                s0, s1 = s1, s0
            sink.append(0.5 - s0.body_q.numpy()[0, 2])
        self.assertGreater(sink[0], 1e-4)
        self.assertLess(abs(sink[1]), 1e-6)

    def test_reset_and_new_rows_start_from_zero(self):
        """Start the contacts from zero after a reset and after a change of the time step, whose square divides the
        multipliers, and forget the constraint rows' multipliers once the rows change their particles."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5)
        self._cloth(builder, 1.005)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
        contacts = pipeline.contacts()
        for change in ("reset", "dt"):
            solver = SolverFBA(model, iterations=10, contact_warmstart=True)
            s0, s1, restored = model.state(), model.state(), []

            def keep(solver=solver, restore=solver._restore_warm_start, restored=restored):
                restore()
                n = 3 * solver._contact_count
                restored.append(np.abs(solver._lam_coulomb_persistent_d.numpy()[:n]).max(initial=0.0))

            solver._restore_warm_start = keep
            for k in range(40):
                if k == 39 and change == "reset":
                    solver.reset(s0)
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, None, contacts, 0.004 if k == 39 and change == "dt" else 0.005)
                s0, s1 = s1, s0
            self.assertGreater(restored[-2], 0.0, msg=change)
            self.assertEqual(restored[-1], 0.0, msg=change)
        rows = ([[0]] * 3, [[1.0]] * 3, s0.particle_q.numpy()[0])
        solver.set_particle_constraints(*rows, directions=np.eye(3))
        solver.step(s0, s1, None, contacts, 0.005)
        self.assertEqual(solver._warm_start.bilateral_count, 3)
        solver.set_particle_constraints([[1]] * 3, *rows[1:], directions=np.eye(3))
        self.assertEqual(solver._warm_start.bilateral_count, 0)

    def test_needs_the_full_operator_and_friction(self):
        """Refuse the warm start with the lite operator or without friction."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        self._cloth(builder, 1.0)
        model = builder.finalize()
        for kwargs in ({"nsn_schur_mode": "lite"}, {"friction": False}):
            with self.assertRaises(ValueError):
                SolverFBA(model, contact_warmstart=True, **kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
