# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType
from newton._src.solvers.fba.face_contact import FaceContactQuery, face_contact_kernel

HALF = np.array([0.5, 0.5, 0.5])


def _box_sdf(x, half=HALF):
    q = np.abs(x) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def _draped_triangle(scale=0.15):
    """A triangle over the box's top edge at ``x = z = 0.5``: its vertices are outside, the edge pokes through."""
    corner = np.array([0.5, 0.0, 0.5])
    return corner + scale * (np.array([[0.3, 0.0, 0.52], [0.7, -0.3, 0.3], [0.7, 0.3, 0.33]]) - corner)


class TestFaceContactKernel(unittest.TestCase):
    """Deepest points of triangles in a box with half extents 0.5 m."""

    def _query(self, points, X_wb=None, X_bs=None):
        """Bary, depth, body_pos and normal of the contacts between one triangle and the box."""
        X_bs, body = (wp.transform_identity(), -1) if X_wb is None else (X_bs, 0)
        inputs = [(points, wp.vec3), ([[0, 1, 2]], wp.int32), ([0], wp.int32), ([0], wp.int32)]
        inputs += [([X_bs if X_wb is None else X_wb], wp.transform), ([X_bs], wp.transform)]
        inputs += [([body], wp.int32), ([int(GeoType.BOX)], wp.int32), ([HALF], wp.vec3)]
        outputs = [wp.zeros(1, dtype=t) for t in (wp.int32, wp.int32, wp.vec3, wp.float32, wp.vec3, wp.vec3)]
        inputs = [wp.array(a, dtype=t) for a, t in inputs]
        wp.launch(face_contact_kernel, dim=1, inputs=[*inputs, 32, 0.02], outputs=outputs)
        n = int(outputs[0].numpy()[0])
        return [a.numpy()[:n] for a in outputs[2:]]

    def test_edge_through_triangle(self):
        """The contact is the brute-force deepest point, anchored on the box surface."""
        points = _draped_triangle()
        bary, depth, body_pos, normal = self._query(points)
        self.assertEqual(len(depth), 1)
        u, v = np.mgrid[0.0:1.0:401j, 0.0:1.0:401j]
        grid = np.stack([1.0 - u - v, u, v], axis=-1).reshape(-1, 3)
        deepest = _box_sdf(grid[grid[:, 0] >= 0.0] @ points).min()
        self.assertLess(deepest, -5e-3)
        self.assertGreater(_box_sdf(points).min(), 0.0)
        self.assertAlmostEqual(float(depth[0]), deepest, delta=2e-4)
        self.assertAlmostEqual(float(_box_sdf(bary[0] @ points)), float(depth[0]), delta=1e-6)
        np.testing.assert_allclose(body_pos[0], bary[0] @ points - depth[0] * normal[0], atol=1e-6)
        self.assertAlmostEqual(float(_box_sdf(body_pos[0])), 0.0, delta=1e-6)

    def test_triangle_on_face_has_no_face_contact(self):
        """A triangle 1 mm above the top face is deepest at a vertex, which particle contacts cover."""
        points = np.array([[-0.2, -0.2, 0.501], [0.2, -0.2, 0.501], [0.0, 0.2, 0.502]])
        self.assertEqual(len(self._query(points)[1]), 0)

    def test_moving_body(self):
        """Moving the box with a body moves the contact with it; the anchor stays in the body frame."""
        local = _draped_triangle()
        X_wb = wp.transform((1.0, -2.0, 0.5), wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.8, 0.5)), 0.7))
        X_bs = wp.transform((0.1, 0.0, -0.2), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4))
        X_ws = wp.transform_multiply(X_wb, X_bs)
        world = np.array([wp.transform_point(X_ws, wp.vec3(*p)) for p in local])
        bary, depth, body_pos, normal = self._query(local)
        moved = self._query(world, X_wb, X_bs)
        np.testing.assert_allclose(moved[0], bary, atol=1e-5)
        np.testing.assert_allclose(moved[1], depth, atol=1e-5)
        np.testing.assert_allclose(moved[2][0], wp.transform_point(X_bs, wp.vec3(*body_pos[0])), atol=1e-5)
        np.testing.assert_allclose(moved[3][0], wp.transform_vector(X_ws, wp.vec3(*normal[0])), atol=1e-5)


class TestFaceContactQuery(unittest.TestCase):
    """Triangles folded up from an edge that passes 1 cm inside a sphere of radius 2 cm."""

    def _query(self, contact_particles, mirrored):
        builder = newton.ModelBuilder()
        center = np.array([0.0, 0.0, -0.01])
        builder.add_shape_sphere(-1, xform=wp.transform(center, wp.quat_identity()), radius=0.02)
        tent = np.array([[-0.05, 0.0, 0.04], [0.0, -0.05, 0.01], [0.0, 0.05, 0.01], [0.05, 0.0, 0.04]])
        for p in np.concatenate([tent, tent[:3] * [1.0, 1.0, -1.0]]) + center:
            builder.add_particle(wp.vec3(*p), wp.vec3(0.0), 0.01)
        builder.add_triangle(0, 1, 2)
        builder.add_triangle(*((4, 5, 6) if mirrored else (3, 2, 1)))
        model = builder.finalize()
        query = FaceContactQuery(model.tri_indices.numpy(), model.particle_count, model.device)
        particles = np.array(contact_particles)
        return query.query(model, model.particle_q, None, particles, np.zeros_like(particles))

    def test_shared_edge_reports_one_contact(self):
        """Both triangles find the optimum on their shared edge; one contact remains, on the sphere."""
        face, _shape, bary, body_pos, normal = self._query([1], mirrored=False)
        self.assertEqual(len(face), 1)
        self.assertEqual(int(np.count_nonzero(bary[0] == 0.0)), 1)
        self.assertAlmostEqual(float(np.linalg.norm(body_pos[0] - [0.0, 0.0, -0.01])), 0.02, delta=1e-6)
        self.assertGreater(float(normal[0][2]), 0.95)

    def test_candidates_come_from_particle_contacts(self):
        """A triangle reaching into the sphere is tested only when one of its particles has a contact."""
        self.assertEqual(self._query([1], mirrored=True)[0].tolist(), [0])
        self.assertEqual(self._query([1, 5], mirrored=True)[0].tolist(), [0, 1])


class TestContactRowsOnSeveralParticles(unittest.TestCase):
    """Two particle contacts and one contact on three particles against dense ``J A⁻¹ Jᵀ`` algebra."""

    def setUp(self):
        import scipy.sparse as sp

        from newton._src.solvers.fba.linear_solver import FBALinearSolver, factorize_and_sparse_inverse  # noqa: PLC0415

        rng = np.random.default_rng(7)
        n, off = 12, rng.uniform(-0.3, 0.3, 11)
        A = sp.diags([off, rng.uniform(5.0, 15.0, n), off], [-1, 0, 1], format="csr")
        self.A_inv = np.kron(np.linalg.inv(A.toarray()), np.eye(3))
        self.solver = FBALinearSolver(factorize_and_sparse_inverse(A), device=wp.get_device())
        particle, alpha = np.array([1, 5, 8]), np.array([1.0, 1.0, 0.5])
        extra, extra_alpha = np.array([[-1, -1], [-1, -1], [9, 3]]), np.array([[0.0, 0.0], [0.0, 0.0], [0.3, 0.2]])
        frames = np.linalg.qr(rng.standard_normal((3, 3, 3)))[0]  # rows: normal, t1, t2
        self.J = np.zeros((9, 3 * n))
        for c in range(3):
            for p, w in zip([particle[c], *extra[c]], [alpha[c], *extra_alpha[c]], strict=True):
                if p >= 0:
                    self.J[3 * c : 3 * c + 3, 3 * p : 3 * p + 3] += w * frames[c].T
        args = [(particle, wp.int32), (frames[:, :, 0], wp.vec3), (alpha, wp.float32)]
        args += [(frames[:, :, 1], wp.vec3), (frames[:, :, 2], wp.vec3)]
        self.args = [wp.array(a, dtype=t) for a, t in args]
        self.extra = [wp.array(extra, dtype=wp.int32), wp.array(extra_alpha, dtype=wp.float32)]
        self.solver.build_schur_complement(
            3, *self.args, download=False, j_extra_indices=self.extra[0], j_extra_alpha=self.extra[1]
        )

    def test_delassus(self):
        np.testing.assert_allclose(
            self.solver.W_device_view().numpy(), self.J @ self.A_inv @ self.J.T, rtol=1e-5, atol=1e-8
        )

    def test_correction(self):
        """``x = A⁻¹ (rhs + Jᵀ λ)`` gathers ``Jᵀ λ`` over every particle of a row."""
        rng = np.random.default_rng(3)
        rhs, lam = rng.standard_normal((12, 3)), rng.standard_normal(9)
        rhs_d, x_d = wp.array(rhs, dtype=wp.vec3), wp.zeros(12, dtype=wp.vec3)
        self.solver.apply_lambda_correction_combined(lam, rhs_d, x_d)
        np.testing.assert_allclose(
            x_d.numpy().ravel(), self.A_inv @ (rhs.ravel() + self.J.T @ lam), rtol=1e-4, atol=1e-6
        )

    def test_residual(self):
        """``offset - J x`` with the extra particles subtracted after the one-particle residual."""
        from newton._src.solvers.fba import kernels as K  # noqa: PLC0415

        rng = np.random.default_rng(5)
        x, offsets = rng.standard_normal((12, 3)), rng.standard_normal(9)
        x_d, r = wp.array(x, dtype=wp.vec3), wp.zeros(9, dtype=wp.float64)
        particle, normal, alpha, t1, t2 = self.args
        offset_d = [wp.array(offsets[k::3], dtype=wp.float64) for k in range(3)]
        wp.launch(K.compute_contact_residual_coulomb_kernel, 3, [*offset_d, alpha, normal, t1, t2, particle, x_d], [r])
        wp.launch(K.subtract_slot_residual_kernel, 3, [*self.extra, normal, t1, t2, x_d], [r])
        np.testing.assert_allclose(r.numpy(), offsets - self.J @ x.ravel(), rtol=1e-5, atol=1e-6)


class TestSolverFBAFaceContacts(unittest.TestCase):
    """A cloth of 5 cm cells on a static box with half extents 0.3 m."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _run(self, face_contacts, pos, cells, steps):
        """Final particle positions, the largest particle speed and the solver."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
        builder.add_shape_box(
            -1, xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.3, cfg=cfg
        )
        builder.add_cloth_grid(
            pos=wp.vec3(*pos),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=cells,
            dim_y=cells,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.002,
            tri_ke=5.0e3,
            tri_ka=5.0e3,
            edge_ke=0.1,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFBA(model, iterations=10, face_contacts=face_contacts)
        pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
        contacts, s0, s1, speed = pipeline.contacts(), model.state(), model.state(), 0.0
        for _ in range(steps):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / 120.0)
            s0, s1 = s1, s0
            speed = max(speed, float(np.abs(s0.particle_qd.numpy()).max()))
        return s0.particle_q.numpy(), speed, solver

    def test_box_edge_stays_out_of_the_faces(self):
        """Draped over the top edge, the cloth keeps the edge out of its faces, not just off its particles."""
        u, v = np.mgrid[0.0:1.0:21j, 0.0:1.0:21j]
        grid = np.stack([1.0 - u - v, u, v], axis=-1).reshape(-1, 3)
        grid = grid[grid[:, 0] >= 0.0]
        for face_contacts, bound in ((False, -5e-3), (True, -2.5e-3)):
            x, speed, solver = self._run(face_contacts, (0.0, -0.3, 0.65), 12, 240)
            faces = np.einsum("gk,tkd->tgd", grid, x[solver.model.tri_indices.numpy()])
            deepest = _box_sdf(faces - [0.0, 0.0, 0.3], np.full(3, 0.3)).min()
            if face_contacts:
                self.assertGreater(deepest, bound)
            else:  # particle contacts alone let the edge 5 mm or more into the faces
                self.assertLess(deepest, bound)
            self.assertLess(speed, 1.5)

    def test_flat_contact_is_unchanged(self):
        """On a flat face every triangle is deepest at a vertex: no face contact, the same trajectory."""
        runs = [self._run(face_contacts, (-0.1, -0.1, 0.62), 4, 40) for face_contacts in (False, True)]
        self.assertIsNone(runs[1][2]._contact_extra)
        np.testing.assert_array_equal(runs[0][0], runs[1][0])

    def test_runs_are_deterministic(self):
        """Two triangles can report the same edge optimum at the same depth; the kept one must not depend on emission order."""
        runs = [self._run(True, (0.0, -0.3, 0.65), 12, 60)[0] for _ in range(2)]
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_needs_the_full_coulomb_mode(self):
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
        )
        with self.assertRaises(ValueError):
            newton.solvers.SolverFBA(builder.finalize(), friction=False, face_contacts=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
