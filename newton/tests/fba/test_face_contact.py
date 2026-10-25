# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType
from newton._src.solvers.fba.face_contact import FaceContactQuery, _mesh_neighbors, face_contact_kernel

HALF = np.array([0.5, 0.5, 0.5])


def _box_sdf(x, half=HALF, edge_radius=0.0):
    q = np.abs(x) - (half - edge_radius)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0) - edge_radius


def _box_mesh(half):
    """The box as a closed triangle mesh with outward winding."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return newton.Mesh(corners, np.ravel([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]))


def _deepest_on(points, half=HALF):
    """Brute-force minimum of the box signed distance over a triangle."""
    u, v = np.mgrid[0.0:1.0:401j, 0.0:1.0:401j]
    grid = np.stack([1.0 - u - v, u, v], axis=-1).reshape(-1, 3)
    return _box_sdf(grid[grid[:, 0] >= 0.0] @ points, half).min()


def _pyramid_mesh():
    """A closed square pyramid 0.1 m tall with its tip at the origin."""
    verts = [[0.0, 0.0, 0.0], [-0.05, -0.05, -0.1], [0.05, -0.05, -0.1], [0.05, 0.05, -0.1], [-0.05, 0.05, -0.1]]
    return newton.Mesh(np.array(verts), np.ravel([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 3, 2), (1, 4, 3)]))


def _notch_mesh(half_width, segments=1):
    """A closed block 1 m wide and tall over ``|y| <= half_width`` with a 90 degree V notch from its top corners
    down to ``(x, z) = (0, 0.5)``: the three triangles around the notch split into ``segments**2``, wound outward."""
    bl, br, tr, notch, tl = np.array([(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (-0.5, 1.0)])  # (x, z)
    n, pts, tris = segments, [], []
    for a, b, c in ((notch, bl, br), (notch, br, tr), (notch, tl, bl)):  # counterclockwise in (x, z)
        index = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                index[i, j] = len(pts)
                pts.append((i * b + j * c + (n - i - j) * a) / n)
        for i in range(n):
            for j in range(n - i):
                tris.append((index[i, j], index[i + 1, j], index[i, j + 1]))
                if i + j < n - 1:
                    tris.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    pts, remap = np.unique(np.round(pts, 9), axis=0, return_inverse=True)
    tris = remap.reshape(-1)[np.array(tris)]
    # Counterclockwise in (x, z), so the -y cap's normals point along -y; its boundary edges run counterclockwise.
    edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    a, b = edges[counts[inverse.reshape(-1)] == 1].T
    v = len(pts)
    verts = np.array([(x, y, z) for y in (-half_width, half_width) for x, z in pts])
    sides = [np.stack([a, b + v, b], axis=1), np.stack([a, a + v, b + v], axis=1)]
    return newton.Mesh(verts, np.vstack([tris, tris[:, ::-1] + v, *sides]).ravel())


def _add_triangle_over_tip(builder, radius=0.002):
    """A triangle of circumradius 0.15 m centered 4 mm above the origin; its particles are 0.13 m from the pyramid."""
    for a in np.deg2rad([0.0, 120.0, 240.0]):
        builder.add_particle(wp.vec3(0.15 * np.cos(a), 0.15 * np.sin(a), 0.004), wp.vec3(0.0), 0.01, radius=radius)
    builder.add_triangle(0, 1, 2, tri_ke=50.0)


def _draped_triangle(scale=0.15):
    """A triangle over the box's top edge at ``x = z = 0.5``: its vertices are outside, the edge pokes through."""
    corner = np.array([0.5, 0.0, 0.5])
    return corner + scale * (np.array([[0.3, 0.0, 0.52], [0.7, -0.3, 0.3], [0.7, 0.3, 0.33]]) - corner)


class TestFaceContactKernel(unittest.TestCase):
    """Deepest points of triangles in a box with half extents 0.5 m."""

    def _query(self, points, X_wb=None, X_bs=None, edge_radius=0.0, half=HALF, mesh=None):
        """Bary, depth, body_pos and normal of the contacts between one triangle and the box or its ``mesh``."""
        X_bs, body = (wp.transform_identity(), -1) if X_wb is None else (X_bs, 0)
        geo, scale, mesh_id = (GeoType.BOX, half, 0) if mesh is None else (GeoType.MESH, np.ones(3), mesh.finalize())
        inputs = [(points, wp.vec3), ([[0, 1, 2]], wp.int32), ([0], wp.int32), ([0], wp.int32)]
        inputs += [([X_bs if X_wb is None else X_wb], wp.transform), ([X_bs], wp.transform)]
        inputs += [([body], wp.int32), ([int(geo)], wp.int32), ([scale], wp.vec3), ([mesh_id], wp.uint64)]
        outputs = [wp.zeros(1, dtype=t) for t in (wp.int32, wp.int32, wp.vec3, wp.float32, wp.vec3, wp.vec3)]
        inputs = [wp.array(a, dtype=t) for a, t in inputs]
        wp.launch(face_contact_kernel, dim=1, inputs=[*inputs, 128, edge_radius], outputs=outputs)
        n = int(outputs[0].numpy()[0])
        return [a.numpy()[:n] for a in outputs[2:]]

    def test_edge_through_triangle(self):
        """The contact is the brute-force deepest point, anchored on the box surface."""
        points = _draped_triangle()
        bary, depth, body_pos, normal = self._query(points)
        self.assertEqual(len(depth), 1)
        deepest = _deepest_on(points)
        self.assertLess(deepest, -5e-3)
        self.assertGreater(_box_sdf(points).min(), 0.0)
        self.assertAlmostEqual(float(depth[0]), deepest, delta=2e-4)
        self.assertAlmostEqual(float(_box_sdf(bary[0] @ points)), float(depth[0]), delta=1e-6)
        np.testing.assert_allclose(body_pos[0], bary[0] @ points - depth[0] * normal[0], atol=1e-6)
        self.assertAlmostEqual(float(_box_sdf(body_pos[0])), 0.0, delta=1e-6)

    def test_rounded_edge_keeps_the_normal_continuous(self):
        """Sunk into a sharp edge the normal is one face's; rounding the edge turns it between both faces."""
        points = _draped_triangle()
        for edge_radius in (0.0, 0.01):
            bary, depth, body_pos, normal = self._query(points, edge_radius=edge_radius)
            n = normal[0]
            if edge_radius == 0.0:
                self.assertAlmostEqual(float(np.abs(n).max()), 1.0, delta=1e-6)
            else:
                self.assertGreater(float(min(n[0], n[2])), 0.3)
            self.assertAlmostEqual(float(_box_sdf(body_pos[0], edge_radius=edge_radius)), 0.0, delta=1e-6)
            self.assertAlmostEqual(
                float(_box_sdf(bary[0] @ points, edge_radius=edge_radius)), float(depth[0]), delta=1e-6
            )

    def test_closed_mesh(self):
        """A closed mesh of the box gives the box's contact; around a 2 mm plate both sides are outside."""
        plate = np.array([[-0.05, -0.04, 0.204], [0.05, -0.04, 0.206], [0.0, 0.04, 0.208]])
        for half, points in ((HALF, _draped_triangle()), (np.array([0.2, 0.001, 0.2]), plate)):
            bary, depth, body_pos, normal = self._query(points, half=half, mesh=_box_mesh(half))
            self.assertEqual(len(depth), 1)
            self.assertAlmostEqual(float(depth[0]), _deepest_on(points, half), delta=2e-4)
            self.assertAlmostEqual(float(_box_sdf(bary[0] @ points, half)), float(depth[0]), delta=1e-6)
            self.assertAlmostEqual(float(_box_sdf(body_pos[0], half)), 0.0, delta=1e-6)
            np.testing.assert_allclose(body_pos[0], bary[0] @ points - depth[0] * normal[0], atol=1e-6)

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
        builder.add_triangle(0, 1, 2, tri_ke=50.0)
        builder.add_triangle(*((4, 5, 6) if mirrored else (3, 2, 1)), tri_ke=50.0)
        model = builder.finalize()
        query = FaceContactQuery(model.tri_indices.numpy(), model.particle_count, model.device)
        particles = np.array(contact_particles)
        return query.query(model, model.particle_q, None, particles, np.zeros_like(particles), 0.0)

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

    def test_margin_reaches_a_tip_inside_a_large_triangle(self):
        """No particle touches the pyramid under the triangle; only the margin finds its tip."""
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=_pyramid_mesh())
        _add_triangle_over_tip(builder)
        model = builder.finalize()
        query, none = FaceContactQuery(model.tri_indices.numpy(), model.particle_count, model.device), np.zeros(0, int)
        self.assertEqual(len(query.query(model, model.particle_q, None, none, none, 0.0)[0]), 0)
        face, _shape, _bary, body_pos, normal = query.query(model, model.particle_q, None, none, none, 0.0, 0.01)
        self.assertEqual(len(face), 1)
        np.testing.assert_allclose(body_pos[0], 0.0, atol=1e-5)
        self.assertGreater(float(normal[0][2]), 0.9)


class TestContactRowsOnSeveralParticles(unittest.TestCase):
    """Two particle contacts and one contact on eight particles against dense ``J A⁻¹ Jᵀ`` algebra."""

    def setUp(self):
        import scipy.sparse as sp

        from newton._src.solvers.fba.linear_solver import FBALinearSolver, factorize_and_sparse_inverse  # noqa: PLC0415

        rng = np.random.default_rng(7)
        n, off = 12, rng.uniform(-0.3, 0.3, 11)
        A = sp.diags([off, rng.uniform(5.0, 15.0, n), off], [-1, 0, 1], format="csr")
        self.A_inv = np.kron(np.linalg.inv(A.toarray()), np.eye(3))
        self.solver = FBALinearSolver(factorize_and_sparse_inverse(A), device=wp.get_device())
        particle, alpha = np.array([1, 5, 8]), np.array([1.0, 1.0, 0.5])
        extra = np.full((3, 7), -1)
        extra[2] = [9, 3, 5, 11, 2, 7, 4]  # shares particle 5 with the second contact
        extra_alpha = np.zeros((3, 7))
        extra_alpha[2] = [0.3, 0.2, -0.25, 0.1, -0.15, 0.05, 0.4]
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
        """Preserve multi-particle Schur values and active strides through bounded dense growth."""
        expected = self.J @ self.A_inv @ self.J.T
        np.testing.assert_allclose(self.solver.W_device_view().numpy(), expected, rtol=1e-5, atol=1e-8)

        # Repeat the same physical rows: the factorization stays 12x12, while W crosses 1024 rows.
        for M, capacity in ((334, 1002), (343, 1152), (385, 1280)):
            with self.subTest(contacts=M):
                contacts = np.arange(M) % 3
                args = [wp.array(a.numpy()[contacts], dtype=a.dtype, device=self.solver.device) for a in self.args]
                extra = [wp.array(a.numpy()[contacts], dtype=a.dtype, device=self.solver.device) for a in self.extra]
                self.solver.build_schur_complement(
                    M, *args, download=False, j_extra_indices=extra[0], j_extra_alpha=extra[1]
                )
                rows = (3 * contacts[:, None] + np.arange(3)).ravel()
                view = self.solver.W_device_view()
                np.testing.assert_allclose(view.numpy(), expected[np.ix_(rows, rows)], rtol=1e-5, atol=1e-8)
                self.assertEqual(view.shape, (3 * M, 3 * M))
                self.assertEqual(self.solver._W_device_d.shape, (capacity, capacity))
                self.assertEqual(view.strides, (capacity * 8, 8))

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

    def _run(self, face_contacts, pos, cells, steps, thickness=0.0, mesh_half=None, radius_overrides=None):
        """Final particle positions, the largest particle speed and the solver; ``mesh_half`` swaps the box for
        a closed mesh box with these half extents, its base also at ``z = 0``."""
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
        if mesh_half is None:
            builder.add_shape_box(
                -1, xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.3, cfg=cfg
            )
        else:
            xform = wp.transform((0.0, 0.0, mesh_half[2]), wp.quat_identity())
            builder.add_shape_mesh(-1, xform=xform, mesh=_box_mesh(mesh_half), cfg=cfg)
        builder.add_cloth_grid(
            pos=wp.vec3(*pos),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=np.atleast_1d(cells)[0],
            dim_y=np.atleast_1d(cells)[-1],
            cell_x=0.05,
            cell_y=0.05,
            mass=0.002,
            tri_ke=2500.0,
            tri_ka=5.0e3,
            edge_ke=0.1,
            particle_radius=thickness,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFBA(
            model, iterations=10, face_contacts=face_contacts, shape_particle_radius_override=radius_overrides
        )
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.15 - thickness)
        contacts, s0, s1, speed = pipeline.contacts(), model.state(), model.state(), 0.0
        for _ in range(steps):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / 120.0)
            s0, s1 = s1, s0
            speed = max(speed, float(np.abs(s0.particle_qd.numpy()).max()))
        return s0.particle_q.numpy(), speed, solver

    @staticmethod
    def _deepest(x, solver, half=(0.3, 0.3, 0.3)):
        """Signed distance of the deepest point of the box with its base at ``z = 0`` inside the cloth faces [m]."""
        u, v = np.mgrid[0.0:1.0:21j, 0.0:1.0:21j]
        grid = np.stack([1.0 - u - v, u, v], axis=-1).reshape(-1, 3)
        faces = np.einsum("gk,tkd->tgd", grid[grid[:, 0] >= 0.0], x[solver.model.tri_indices.numpy()])
        return _box_sdf(faces - [0.0, 0.0, half[2]], np.asarray(half)).min()

    def test_box_edge_stays_out_of_the_faces(self):
        """Draped over the top edge, the cloth keeps the edge out of its faces, not just off its particles.

        A third of the cloth hangs over the edge. With half of it hanging, the cloth slid off the edge in a
        third or more of starts moved by 10 um, and which ones changed with float32 rounding."""
        for face_contacts, bound in ((False, -3e-3), (True, -1e-3)):
            x, speed, solver = self._run(face_contacts, (-0.1, -0.3, 0.65), 12, 240)
            deepest = self._deepest(x, solver)
            if face_contacts:
                self.assertGreater(deepest, bound)
            else:  # particle contacts alone let the edge 4 mm or more into the faces
                self.assertLess(deepest, bound)
            self.assertLess(speed, 1.5)

    def test_contact_thickness(self):
        """The cloth rests the thickness above the top face, and draped over the edge it clears the edge."""
        x, _, _ = self._run(True, (-0.1, -0.1, 0.62), 4, 60, thickness=3e-3)
        np.testing.assert_allclose(x[:, 2] - 0.6, 3e-3, atol=1e-5)
        x, speed, solver = self._run(True, (0.0, -0.3, 0.65), 12, 240, thickness=3e-3)
        self.assertGreater(self._deepest(x, solver), 1e-3)
        self.assertLess(speed, 1.5)

    def test_reduced_shape_radius_preserves_particle_radii(self):
        """Keep cloth 2 mm from a selected support without changing its 28 mm particle radii."""
        x, _, solver = self._run(True, (-0.1, -0.1, 0.64), 4, 60, thickness=0.028, radius_overrides={0: 0.002})
        np.testing.assert_allclose(x[:, 2] - 0.6, 0.002, atol=1.0e-5)
        np.testing.assert_allclose(solver.model.particle_radius.numpy(), 0.028)
        x, _, _ = self._run(True, (-0.1, -0.1, 0.64), 4, 60, thickness=0.028)
        np.testing.assert_allclose(x[:, 2] - 0.6, 0.028, atol=1.0e-5)

    def test_thin_mesh_blade(self):
        """A closed 2 mm blade under the gap between two particle rows cuts through a cloth held by particle
        contacts alone; with face contacts the cloth folds over its edge and the edge stays out of its faces."""
        half = np.array([0.3, 0.001, 0.15])
        for face_contacts in (False, True):
            x, speed, solver = self._run(face_contacts, (-0.3, -0.325, 0.32), (12, 13), 120, 3e-3, mesh_half=half)
            if face_contacts:
                self.assertGreater(self._deepest(x, solver, half), 2e-3)
                self.assertGreater(x[:, 2].max(), 0.3)
                self.assertLess(speed, 1.5)
            else:
                self.assertLess(x[:, 2].max(), 0.0)

    def test_face_contact_margin(self):
        """A triangle falls onto a pyramid tip under its middle and rests on it only with the margin."""
        for margin in (None, 0.01):
            builder = newton.ModelBuilder()
            builder.add_shape_mesh(-1, mesh=_pyramid_mesh(), cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
            _add_triangle_over_tip(builder, radius=3e-3)
            model = builder.finalize()
            solver = newton.solvers.SolverFBA(model, iterations=10, face_contacts=True, face_contact_margin=margin)
            pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.009)
            contacts, s0, s1 = pipeline.contacts(), model.state(), model.state()
            for _ in range(30):
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, None, contacts, 1.0 / 120.0)
                s0, s1 = s1, s0
            x = s0.particle_q.numpy()
            normal = np.cross(x[1] - x[0], x[2] - x[0])
            height = float(np.dot(x[0], normal / np.linalg.norm(normal)))  # of the triangle plane above the tip
            if margin is None:
                self.assertLess(height, 0.0)
            else:
                self.assertGreater(height, 2e-3)

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
            tri_ke=50.0,
        )
        with self.assertRaises(ValueError):
            newton.solvers.SolverFBA(builder.finalize(), friction=False, face_contacts=True)


class TestMeshNeighbors(unittest.TestCase):
    def test_closed_mesh(self):
        """Across each edge of a triangle of a closed mesh lies a triangle with that edge, which lists it back."""
        builder = newton.ModelBuilder()
        builder.add_shape_sphere(-1, radius=0.1)
        builder.add_shape_mesh(-1, mesh=_notch_mesh(0.005, 3))
        model = builder.finalize(device="cpu")
        start, table = (a.numpy() for a in _mesh_neighbors(model, model.device))
        tri = np.asarray(model.shape_source[1].indices).reshape(-1, 3)
        rows = table[start[1] : start[1] + len(tri)]
        self.assertTrue((rows >= 0).all())
        for t, row in enumerate(rows):
            for (i, j), other in zip(((0, 1), (0, 2), (1, 2)), row, strict=True):
                self.assertNotEqual(other, t)
                self.assertTrue({tri[t, i], tri[t, j]} <= set(tri[other]))
                self.assertIn(t, rows[other])


class TestMeshContactMinima(unittest.TestCase):
    """Particle contacts of radius 5 mm on a block with a 90 degree V notch whose bottom runs along y."""

    NOTCH = np.array([0.0, 0.0, 0.5])
    RIGHT = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)  # outward normal of the face from the notch up to x = 0.5
    LEFT = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)

    def _minima(self, x, half_width=0.2, segments=1):
        """The pipeline's contacts of a particle at ``x`` and the other minima of its distance to the block."""
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=_notch_mesh(half_width, segments))
        builder.add_particle(wp.vec3(*x), wp.vec3(0.0), 0.01, radius=0.005)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.02)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        m = int(contacts.soft_contact_count.numpy()[0])
        fields = (contacts.soft_contact_particle, contacts.soft_contact_shape)
        fields += (contacts.soft_contact_body_pos, contacts.soft_contact_normal)
        primary = [a.numpy()[:m] for a in fields]
        query = FaceContactQuery(np.zeros((0, 3), dtype=np.int32), model.particle_count, model.device)
        return primary, query.mesh_minima(model, model.particle_q, None, *primary, alarm=0.02)

    def test_notch_holds_against_both_faces(self):
        """Above the notch and nearer its right face, the particle's closest point lies on that face alone; the
        other minimum adds the left face."""
        x = self.NOTCH + np.array([0.002, 0.0, 0.015])
        primary, (particle, shape, body_pos, normal, contact) = self._minima(x)
        np.testing.assert_allclose(primary[3], [self.RIGHT], atol=1e-5)
        np.testing.assert_array_equal(particle, [0])
        np.testing.assert_array_equal(contact, [0])
        np.testing.assert_array_equal(shape, primary[1])
        np.testing.assert_allclose(normal, [self.LEFT], atol=1e-5)
        np.testing.assert_allclose(body_pos, [x - np.dot(x - self.NOTCH, self.LEFT) * self.LEFT], atol=1e-6)

    def test_notch_of_a_thin_finely_split_block(self):
        """On the ridge of a block 1 cm thick whose edges are split in 20, the minimum on the left face is found
        among the many triangles near the notch."""
        x = self.NOTCH + np.array([0.0015, 0.0034, 0.0179])
        primary, (_, _, body_pos, normal, _) = self._minima(x, half_width=0.005, segments=20)
        np.testing.assert_allclose(primary[3], [self.RIGHT], atol=1e-5)
        np.testing.assert_allclose(normal, [self.LEFT], atol=1e-5)
        np.testing.assert_allclose(body_pos, [x - np.dot(x - self.NOTCH, self.LEFT) * self.LEFT], atol=1e-6)

    def test_flat_face_has_one_minimum(self):
        """Under the bottom face, 4 mm beside the diagonal between its two triangles, the other triangle's closest
        point on the diagonal lies within reach, but the triangle through it comes closer."""
        primary, minima = self._minima([0.0, 0.004, -0.01])
        self.assertEqual(len(primary[0]), 1)
        self.assertEqual(len(minima[0]), 0)

    def test_far_side_of_a_thin_block(self):
        """Beside a block 1 cm thick, its far side lies within reach but faces away from the particle."""
        primary, minima = self._minima([0.1, 0.015, 0.25], half_width=0.005)
        self.assertEqual(len(primary[0]), 1)
        self.assertEqual(len(minima[0]), 0)

    def test_particle_inside(self):
        """A particle inside the block keeps its one contact, which pushes it out through the nearest face."""
        primary, minima = self._minima([0.0, 0.001, 0.45], half_width=0.005)
        self.assertEqual(len(primary[0]), 1)
        self.assertEqual(len(minima[0]), 0)

    def test_triangle_slides_into_the_notch(self):
        """A frictionless triangle slides down the right face into the notch and stays a particle radius out of
        both faces; held by its closest points alone, its particles slid through the left face."""
        if not wp.is_cuda_available():
            self.skipTest("The SolverFBA contact NSN needs CUDA")
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=_notch_mesh(0.2), cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
        along = np.array([-1.0, 0.0, -1.0]) / np.sqrt(2.0)  # down the right face toward the notch
        center = self.NOTCH - 0.3 * along + 0.006 * self.RIGHT
        for offset in (0.02 * along, -0.01 * along + [0.0, 0.02, 0.0], -0.01 * along - [0.0, 0.02, 0.0]):
            builder.add_particle(wp.vec3(*(center + offset)), wp.vec3(0.0), 0.01, radius=0.005)
        builder.add_triangle(0, 1, 2, tri_ke=50.0)
        model = builder.finalize()
        solver = newton.solvers.SolverFBA(model, iterations=10, face_contacts=True, face_contact_margin=0.02)
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.02)
        contacts, s0, s1, lowest = pipeline.contacts(), model.state(), model.state(), np.inf
        observables = solver.observables({solver.ObservableKind.CONTACT_F}, contacts=contacts)
        for _ in range(200):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 0.005, observables=observables)
            s0, s1 = s1, s0
            gap = (s0.particle_q.numpy() - self.NOTCH) @ np.stack([self.LEFT, self.RIGHT], 1)
            lowest = min(lowest, float(gap.min()) - 0.005)
        self.assertGreater(lowest, -1.0e-3, f"a particle came {-lowest * 1e3:.1f} mm inside the contact surface")
        x = s0.particle_q.numpy()
        self.assertLess(np.abs(x[:, 0]).max(), 0.03, "the triangle did not settle in the notch")
        # Each row of the pipeline's particle contacts and of their other minima keeps its particle contact's index.
        count = int(contacts.soft_contact_count.numpy()[0])
        record = solver._contact_record_d.numpy()[: solver._contact_count]
        rows = record >= 0
        np.testing.assert_array_equal(np.unique(record[rows]), np.arange(count))
        self.assertGreater(rows.sum(), count, "no particle has a second minimum in the notch")
        pairs = np.stack([solver._contact_particle_d.numpy(), solver._contact_shape_d.numpy()])[
            :, : solver._contact_count
        ]
        pipeline_pairs = np.stack([contacts.soft_contact_particle.numpy(), contacts.soft_contact_shape.numpy()])
        np.testing.assert_array_equal(pairs[:, rows], pipeline_pairs[:, record[rows]])
        # The forces on the block, the second faces' included, balance the last step's momentum change and weight.
        soft = observables.contact_f.numpy()[contacts.rigid_contact_max :][:count]
        mass = model.particle_mass.numpy()
        change = mass @ (s0.particle_qd.numpy() - s1.particle_qd.numpy()) / 0.005
        weight = float(mass.sum()) * 9.81
        np.testing.assert_allclose(-soft[:, :3].sum(axis=0), change + np.array([0.0, 0.0, weight]), atol=1e-5 * weight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
