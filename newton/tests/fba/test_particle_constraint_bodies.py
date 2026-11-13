# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA particle constraint rows on points of dynamic rigid bodies (``set_particle_constraints``
with ``bodies`` and ``body_points``)."""

import unittest

import numpy as np
import warp as wp

import newton


def _rotation(q):
    """Rotation matrix of a Warp quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _tie(solver, particles, body, points):
    """Three rows holding each of ``particles`` at its point of ``body`` (in the body frame)."""
    n = len(particles)
    solver.set_particle_constraints(
        np.repeat(np.asarray(particles)[:, None], 3, axis=0),
        np.ones((3 * n, 1)),
        np.zeros(3 * n),
        directions=np.tile(np.eye(3), (n, 1)),
        bodies=np.full(3 * n, body),
        body_points=np.repeat(np.asarray(points, dtype=float), 3, axis=0),
    )


def _rod(builder, start, masses):
    """A 1 cm rod of two particles along x from ``start``, with the given particle masses."""
    from newton.solvers import SolverFBA  # noqa: PLC0415

    points = np.array([start, np.asarray(start) + np.array([0.01, 0.0, 0.0])])
    rod = SolverFBA.add_particle_rod(builder, points, 0.001, 1e9, density=1e3)
    for i, m in zip(rod, masses, strict=True):
        builder.particle_mass[i] = m
    return list(rod), points


class TestBodyRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _step(self, solver, s0, s1, steps, dt):
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
        return s0

    def test_particles_follow_the_body_points(self):
        """Without gravity a box spins and drifts; a light rod tied to two points of it stays on those points."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        origin = np.array([0.0, 0.0, 1.0])
        v, w = np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.5, 2.0])
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        box = builder.add_body(xform=wp.transform(tuple(origin), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
        rod, points = _rod(builder, origin + np.array([0.15, 0.0, 0.05]), (0.001, 0.001))
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10, nsn_sparse_storage=True)
        local = points - origin
        _tie(solver, rod, box, local)
        s0, s1 = model.state(), model.state()
        s0.body_qd.assign([[*v, *w]])
        s0.particle_qd.assign(np.array([v + np.cross(w, p) for p in local], dtype=np.float32))
        s0 = self._step(solver, s0, s1, 100, 0.005)
        q = s0.body_q.numpy()[box]
        target = q[:3] + local @ _rotation(q[3:]).T
        np.testing.assert_allclose(s0.particle_q.numpy()[rod], target, atol=2e-4)
        self.assertGreater(np.linalg.norm(q[:3] - origin), 0.04)  # the box did move

    def test_body_takes_the_row_forces(self):
        """A box hinged at its centre of mass feels no torque from gravity; a heavy rod tied off its axis turns it at
        the rigid pendulum's rate from rest."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        m, r, h = 0.25, 0.2, 0.1
        origin = np.array([0.0, 0.0, 1.0])
        builder = newton.ModelBuilder()
        box = builder.add_link(xform=wp.transform(tuple(origin), wp.quat_identity()))
        builder.add_shape_box(box, hx=h, hy=h, hz=h)
        joint = builder.add_joint_revolute(
            -1, box, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform(tuple(origin), wp.quat_identity())
        )
        builder.add_articulation([joint])
        rod, points = _rod(builder, origin + np.array([r, 0.0, 0.0]), (m, m))
        model = builder.finalize()
        solver = SolverFBA(model, iterations=20, nsn_sparse_storage=True)
        _tie(solver, rod, box, points - origin)
        s0, s1 = model.state(), model.state()
        steps, dt = 10, 0.002
        s0 = self._step(solver, s0, s1, steps, dt)
        arms = points[:, 0] - origin[0]
        inertia = model.body_inertia.numpy()[box][1, 1]
        alpha = m * 9.81 * arms.sum() / (inertia + m * (arms**2).sum())
        omega = s0.body_qd.numpy()[box][4]
        # Positive about +y turns +x down: the tied side falls.
        self.assertAlmostEqual(omega / (alpha * steps * dt), 1.0, delta=0.05)
        self.assertLess(s0.particle_q.numpy()[rod[0]][2], origin[2])

    def test_rows_are_validated(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
        fixed = builder.add_body(xform=wp.transform((1.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(fixed, hx=0.1, hy=0.1, hz=0.1)
        rod, _ = _rod(builder, (0.15, 0.0, 1.0), (0.001, 0.001))
        solver = SolverFBA(builder.finalize(), iterations=1, nsn_sparse_storage=True)
        point = [(0.15, 0.0, 0.0)]
        with self.assertRaises(NotImplementedError):  # rows on bodies need directions
            solver.set_particle_constraints(
                np.array([[rod[0]]]), np.ones((1, 1)), np.zeros(1), bodies=[box], body_points=point
            )
        with self.assertRaises(ValueError):  # a kinematic body is not integrated
            _tie(solver, rod[:1], fixed, point)
        with self.assertRaises(ValueError):  # out of range
            _tie(solver, rod[:1], 7, point)
        with self.assertRaises(ValueError):  # one point per row
            solver.set_particle_constraints(
                np.full((3, 1), rod[0]),
                np.ones((3, 1)),
                np.zeros(3),
                directions=np.eye(3),
                bodies=np.full(3, box),
                body_points=point,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
