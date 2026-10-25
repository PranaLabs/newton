# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.kernels import _mat13d
from newton._src.solvers.fba.self_contact import SelfContact, particle_rings
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton.solvers import SolverFBA


def _closest_on_triangle(p, a, b, c):
    """Closest point of a triangle to ``p`` by dense sampling of barycentric coordinates [m]."""
    u, v = np.mgrid[0.0:1.0:201j, 0.0:1.0:201j]
    keep = u + v <= 1.0
    points = np.outer(1.0 - u[keep] - v[keep], a) + np.outer(u[keep], b) + np.outer(v[keep], c)
    return points[np.argmin(np.linalg.norm(points - p, axis=1))]


def _crossing_ribbons(gap, device=None, scale=1.0):
    """Two ribbons of nine 4 cm by 8 mm cells crossing at right angles in the middle of their long edges: the lower
    one along y in the plane z = 0, pinned, and the upper one along x ``gap`` over it. Particle ``2 i + j`` of a
    ribbon lies ``4 i`` cm along it and ``8 j`` mm across it, from -18 cm and -4 mm; the upper ribbon's particles
    follow the lower one's 20. ``scale`` scales all of it."""
    builder = newton.ModelBuilder()
    ribbons = (((-0.004, -0.18, 0.0), (0, 1, 0), (1, 0, 0)), ((-0.18, -0.004, gap), (1, 0, 0), (0, 1, 0)))
    for ribbon in ribbons:
        start, along, across = (np.asarray(v, dtype=float) for v in ribbon)
        vertices = [scale * (start + 0.04 * i * along + 0.008 * j * across) for i in range(10) for j in range(2)]
        indices = [k for i in range(9) for k in (2 * i, 2 * i + 2, 2 * i + 1, 2 * i + 1, 2 * i + 2, 2 * i + 3)]
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[wp.vec3(*v) for v in vertices],
            indices=indices,
            density=0.2,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            edge_ke=1.0e-3,
            particle_radius=0.0,
        )
        if builder.particle_count == 20:
            for k in range(20):
                builder.particle_mass[k] = 0.0
    return builder.finalize(device=device)


class TestSelfContactDetection(unittest.TestCase):
    """Two sheets of 2 cm cells, the second 1 cm over the first and shifted 3 mm along x and y."""

    def _sheets(self, device=None):
        builder = newton.ModelBuilder()
        for z, shift in ((0.0, 0.0), (0.01, 0.003)):
            builder.add_cloth_grid(
                pos=wp.vec3(shift, shift, z),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=10,
                dim_y=10,
                cell_x=0.02,
                cell_y=0.02,
                mass=0.01,
                tri_ke=50.0,
            )
        return builder.finalize(device=device)

    def test_vertex_face_takes_the_nearest_triangle(self):
        """Find for each particle one contact, with its nearest triangle of the other sheet, along the direction
        from the closest point to the particle, on the CPU, where a block has one thread, as on CUDA."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            model = self._sheets(device)
            q, tris = model.particle_q.numpy(), model.tri_indices.numpy()
            particles, alpha, normal = SelfContact(model, "vertex_face", 0.015).detect(model.particle_q)
            np.testing.assert_array_equal(particles[:, 0], np.arange(model.particle_count), err_msg=device)
            half = model.particle_count // 2
            for k in range(0, len(particles), 17):
                v = particles[k, 0]
                other = tris[(tris >= half).all(axis=1) if v < half else (tris < half).all(axis=1)]
                closest = [_closest_on_triangle(q[v], *q[t]) for t in other]
                nearest = closest[np.argmin([np.linalg.norm(q[v] - c) for c in closest])]
                point = -(alpha[k, 1:] @ q[particles[k, 1:]])
                np.testing.assert_allclose(point, nearest, atol=2e-4, err_msg=device)
                np.testing.assert_allclose(normal[k], (q[v] - point) / np.linalg.norm(q[v] - point), atol=1e-5)

    def test_vertex_face_grows_its_candidate_lists(self):
        """Search again with longer lists when particles have more candidate triangles than their lists hold, and find
        the same contacts."""
        model = self._sheets()
        contact = SelfContact(model, "vertex_face", 0.015)
        expected = contact.detect(model.particle_q)
        contact.candidates = wp.empty((model.particle_count, 1), dtype=wp.int32, device=model.device)
        found = contact.detect(model.particle_q)
        self.assertGreater(contact.candidates.shape[1], 1)
        for got, want in zip(found, expected, strict=True):
            np.testing.assert_array_equal(got, want)

    def test_group_margins(self):
        """Find the sheets 1 cm apart within a 1.2 cm margin between groups, with or without margins within groups, but
        not as one group with a 2 mm margin or without contacts within it; refuse margins without groups, and a
        triangle across two groups."""
        model = self._sheets()
        half = model.particle_count // 2
        two = np.repeat(np.arange(2, dtype=np.int32), half)
        common = {"particle_enable_self_contact": True, "particle_self_contact_margin": 0.012}
        common["particle_self_contact_gap"] = 0.001
        groups = {"particle_self_contact_group_ids": two, "particle_self_contact_group_margins": [0.002, 0.002]}
        self.assertEqual(
            SolverFBA(model, **common, **groups)._self_contact.find(model.particle_q), model.particle_count
        )
        groups = {"particle_self_contact_group_ids": 0 * two, "particle_self_contact_group_margins": [0.002]}
        self.assertEqual(SolverFBA(model, **common, **groups)._self_contact.find(model.particle_q), 0)
        found = SolverFBA(model, **common, particle_self_contact_group_ids=two)._self_contact.find(model.particle_q)
        self.assertEqual(found, model.particle_count)
        self.assertEqual(
            SolverFBA(model, **common, particle_self_contact_group_ids=0 * two)._self_contact.find(model.particle_q), 0
        )
        with self.assertRaises(ValueError):
            SolverFBA(model, **common, particle_self_contact_group_margins=[0.002])
        mixed = two.copy()
        mixed[0] = 1
        with self.assertRaises(ValueError):
            SolverFBA(
                model, **common, particle_self_contact_group_ids=mixed, particle_self_contact_group_margins=[1, 1]
            )

    def test_vertex_face_skips_the_interior_particles_of_solids(self):
        """Give a solid's interior particle no vertex-face contact, and its surface particles, a cloth's and a free
        particle's one; vertex-vertex contacts keep every particle. A second search clears the slots it skips."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                # Four tetrahedra around particle 4, 5 mm over the solid's bottom face.
                builder.add_soft_mesh(
                    pos=wp.vec3(0.0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(0.0),
                    vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.1, 0.1, 0.005)],
                    indices=[2, 1, 3, 4, 3, 0, 2, 4, 1, 0, 3, 4, 2, 0, 1, 4],
                    density=1.0,
                    k_mu=1.0,
                    k_lambda=1.0,
                    k_damp=0.0,
                    add_surface_mesh_edges=False,
                )
                interior = 4
                cloth = [
                    builder.add_particle(wp.vec3(p), wp.vec3(0.0), 1.0)
                    for p in [(-0.2, -0.2, -0.01), (0.2, -0.2, -0.01), (0.0, 0.3, -0.01)]
                ]
                builder.add_triangle(*cloth)
                free = builder.add_particle(wp.vec3(0.1, 0.1, -0.002), wp.vec3(0.0), 1.0)
                model = builder.finalize(device=device)
                q = model.particle_q.numpy()
                self.assertIn(interior, model.tet_indices.numpy())
                self.assertNotIn(interior, model.tri_indices.numpy())

                contact = SelfContact(model, "vertex_face", 0.02)
                particles, alpha, normal = contact.detect(model.particle_q)
                self.assertNotIn(interior, particles[:, 0])
                # The solid's corner over the cloth, and the free particle under the solid's bottom face.
                for v, point, direction in (
                    (0, (0.0, 0.0, -0.01), (0.0, 0.0, 1.0)),
                    (free, (0.1, 0.1, 0.0), (0.0, 0.0, -1.0)),
                ):
                    (row,) = np.flatnonzero(particles[:, 0] == v)
                    np.testing.assert_allclose(-(alpha[row, 1:] @ q[particles[row, 1:]]), point, atol=1e-7)
                    np.testing.assert_allclose(normal[row], direction, atol=1e-4)

                pairs, _, _ = SelfContact(model, "vertex_vertex", 0.02).detect(model.particle_q)
                self.assertIn((interior, free), {tuple(pair) for pair in pairs[:, :2]})

                contact.candidate_count.fill_(1)
                contact.found.fill_(1)
                for got, want in zip(contact.detect(model.particle_q), (particles, alpha, normal), strict=True):
                    np.testing.assert_array_equal(got, want)

    def _thin_solid(self, builder, pos):
        """A slab of 8 by 4 cells of 2 cm, 2 mm thick."""
        builder.add_soft_grid(
            pos=wp.vec3(*pos),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=8,
            dim_y=4,
            dim_z=1,
            cell_x=0.02,
            cell_y=0.02,
            cell_z=0.002,
            density=1.0,
            k_mu=1.0,
            k_lambda=1.0,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )

    def test_vertex_face_takes_a_thin_solid_as_one_body(self):
        """Keep the two faces of a 2 mm slab, which share its tetrahedra, from colliding at 3 mm, while two such slabs
        3 mm apart collide."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                self._thin_solid(builder, (0.0, 0.0, 0.0))
                single = builder.finalize(device=device)
                self.assertEqual(len(SelfContact(single, "vertex_face", 0.003).detect(single.particle_q)[0]), 0)

                split = single.particle_count
                self._thin_solid(builder, (0.003, 0.003, 0.003))
                pair = builder.finalize(device=device)
                particles, _, _ = SelfContact(pair, "vertex_face", 0.003).detect(pair.particle_q)
                self.assertGreater(len(particles), 20)
                for query, *triangle in particles:
                    self.assertTrue(all((query < split) != (v < split) for v in triangle))

    def test_vertex_vertex_takes_a_thin_solid_as_one_body(self):
        """Keep the two faces of a 2 mm slab, which share its tetrahedra, from colliding in vertex-vertex mode too,
        while the particles of two such slabs 3 mm apart collide."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                self._thin_solid(builder, (0.0, 0.0, 0.0))
                single = builder.finalize(device=device)
                self.assertEqual(len(SelfContact(single, "vertex_vertex", 0.005).detect(single.particle_q)[0]), 0)

                split = single.particle_count
                self._thin_solid(builder, (0.003, 0.003, 0.003))
                pair = builder.finalize(device=device)
                particles, _, _ = SelfContact(pair, "vertex_vertex", 0.005).detect(pair.particle_q)
                self.assertGreater(len(particles), 20)
                self.assertTrue(((particles[:, 0] < split) != (particles[:, 1] < split)).all())

    def test_vertex_face_keeps_the_folds_of_a_thin_solid(self):
        """Collide the two halves of a slab folded in half where they lie apart at rest."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                self._thin_solid(builder, (0.0, 0.0, 0.0))
                model = builder.finalize(device=device)
                rest = model.particle_q.numpy()
                folded = rest.copy()
                right = rest[:, 0] > 0.08
                folded[right, 0] = 0.16 - rest[right, 0]
                folded[right, 2] = 0.005 - rest[right, 2]
                contact = SelfContact(model, "vertex_face", 0.003)
                particles, alpha, _ = contact.detect(wp.array(folded, dtype=wp.vec3, device=device))
                self.assertGreater(len(particles), 20)
                at_rest = rest[particles[:, 0]] + np.einsum("ki,kij->kj", alpha[:, 1:], rest[particles[:, 1:]])
                self.assertGreater(np.count_nonzero(np.linalg.norm(at_rest, axis=1) > 0.04), 10)

    def test_edge_edge_takes_crossing_edges_once(self):
        """Find each pair of crossing edges once, at its closest points, after the vertex-face contacts.

        The long edges of two ribbons cross 3 mm apart in the middle of their 4 cm edges, 1.6 cm or more from every
        particle, so no vertex-face contact reaches them within the 1 cm query radius; and as much at 1/20 of that
        size, with 2 mm edges.
        """
        for device, scale in itertools.product(["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []), (1.0, 0.05)):
            model = _crossing_ribbons(0.003, device, scale)
            q = model.particle_q.numpy() / scale
            contact = SelfContact(model, "vertex_face", 0.01 * scale, rings=1, edge_edge=True)
            particles, alpha, normal = contact.detect(model.particle_q)
            edge = alpha[:, 1] > 0.0
            self.assertTrue(edge.all(), msg=device)
            pairs = {(tuple(p[:2]), tuple(p[2:])) for p in particles}
            self.assertEqual(len(pairs), len(particles), msg=device)
            for p, a, n in zip(particles, alpha, normal, strict=True):
                gap = a @ q[p]
                self.assertAlmostEqual(float(a.sum()), 0.0, places=6)
                self.assertTrue(0.0 < a[1] < 1.0 and 0.0 < -a[3] < 1.0, msg=device)
                self.assertLess(np.linalg.norm(gap), 0.01)
                np.testing.assert_allclose(n, gap / np.linalg.norm(gap), atol=1e-5, err_msg=device)
            # The lower ribbon's long edges x = -4 and 4 mm cross the upper one's y = -4 and 4 mm at (x, y), 3 mm under it.
            for lower, x in (((8, 10), -0.004), ((9, 11), 0.004)):
                for upper, y in (((28, 30), -0.004), ((29, 31), 0.004)):
                    k = [i for i, p in enumerate(particles) if (tuple(p[:2]), tuple(p[2:])) == (lower, upper)]
                    self.assertEqual(len(k), 1, msg=device)
                    s, u = (y + 0.02) / 0.04, (x + 0.02) / 0.04
                    np.testing.assert_allclose(alpha[k[0]], [1 - s, s, u - 1, -u], atol=1e-5, err_msg=device)
                    np.testing.assert_allclose(normal[k[0]], [0.0, 0.0, -1.0], atol=1e-5, err_msg=device)
            # Brought 1 cm under the upper ribbon's middle, the lower ribbon's particles take vertex-face contacts first.
            moved = q.copy()
            moved[8:12, 2] = 0.002
            moved[8:12, 1] *= 0.1
            particles, alpha, _ = contact.detect(wp.array(scale * moved, dtype=wp.vec3, device=device))
            edge = alpha[:, 1] > 0.0
            self.assertTrue((~edge).any(), msg=device)
            if edge.any():
                self.assertLess(np.flatnonzero(~edge).max(), np.flatnonzero(edge).min())

    def test_edge_edge_ratio_admits_edges_of_flat_layers(self):
        """Take no edge-edge contact between the flat sheets at the default ratio, whose vertex-face contacts hold the
        edges apart, and some at a ratio of 1.2, which admits edges as near as their particles' nearest triangles."""
        model = self._sheets()
        counts = []
        for ratio in (0.9, 1.2):
            contact = SelfContact(model, "vertex_face", 0.015, rings=1, edge_edge=True, edge_edge_ratio=ratio)
            _, alpha, _ = contact.detect(model.particle_q)
            counts.append(int((alpha[:, 1] > 0.0).sum()))
        self.assertEqual(counts[0], 0)
        self.assertGreater(counts[1], 0)
        solver = newton.solvers.SolverFBA(
            model,
            particle_enable_self_contact=True,
            particle_self_contact_edge_edge=True,
            particle_self_contact_edge_edge_ratio=1.2,
        )
        self.assertEqual(solver._self_contact.edge_edge_ratio, 1.2)
        with self.assertRaises(ValueError):
            newton.solvers.SolverFBA(
                model, particle_enable_self_contact=True, particle_self_contact_edge_edge_ratio=0.0
            )

    def test_vertex_vertex_pairs_once(self):
        """Make every pair of particles within the margin one contact.

        A sheet's own particles lie 2 cm apart, so all pairs join the two sheets.
        """
        model = self._sheets()
        q = model.particle_q.numpy()
        particles, alpha, normal = SelfContact(model, "vertex_vertex", 0.015).detect(model.particle_q)
        d = np.linalg.norm(q[:, None] - q[None], axis=2)
        expected = set(zip(*np.nonzero(np.triu(d < 0.015, 1)), strict=True))
        self.assertEqual({(a, b) for a, b in particles[:, :2]}, expected)
        np.testing.assert_array_equal(alpha[:, :2], np.tile([1.0, -1.0], (len(particles), 1)))
        expected_normal = (q[particles[:, 0]] - q[particles[:, 1]]) / d[tuple(particles[:, :2].T)][:, None]
        np.testing.assert_allclose(normal, expected_normal, atol=1e-6)

    def test_rings_exclude_near_particles(self):
        """Find only pairs more than two mesh edges apart on one sheet with a 5 cm margin."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=6,
            dim_y=6,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.01,
            tri_ke=50.0,
        )
        model = builder.finalize()
        tris = model.tri_indices.numpy()
        offsets, ring = particle_rings(tris, model.particle_count, 2)
        one = [set(tris[(tris == v).any(axis=1)].ravel()) | {v} for v in range(model.particle_count)]
        for v in range(model.particle_count):
            self.assertEqual(set(ring[offsets[v] : offsets[v + 1]]), set().union(*(one[u] for u in one[v])))
        particles, _, _ = SelfContact(model, "vertex_vertex", 0.05).detect(model.particle_q)
        self.assertGreater(len(particles), 0)
        for a, b in particles[:, :2]:
            self.assertNotIn(b, ring[offsets[a] : offsets[a + 1]])

    def test_rest_radius_drops_contacts_near_at_rest(self):
        """Drop the contacts between parts closer than the rest radius in the rest shape.

        On a flat sheet of 2 cm cells, particles find triangles outside their first ring within 3 cm, which a rest
        radius of 3 cm drops. Folded at its middle line 1 cm over itself, the sheet keeps the contacts between parts 3
        cm or more apart at rest.
        """
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=8,
            dim_y=4,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.01,
            tri_ke=50.0,
        )
        model = builder.finalize()
        rest, tris = model.particle_q.numpy(), model.tri_indices.numpy()
        self.assertGreater(len(SelfContact(model, "vertex_face", 0.03, rings=1).detect(model.particle_q)[0]), 0)
        contact = SelfContact(model, "vertex_face", 0.03, rings=1, rest_radius=0.03)
        self.assertEqual(len(contact.detect(model.particle_q)[0]), 0)
        folded = rest.copy()
        right = folded[:, 0] > 0.08
        folded[right, 0], folded[right, 2] = 0.16 - folded[right, 0], 0.01
        particles, _, _ = contact.detect(wp.array(folded, dtype=wp.vec3))
        self.assertGreater(len(particles), 10)
        for v, a, b, c in particles:
            t = next(i for i, tri in enumerate(tris) if {a, b, c} == set(tri))
            point = _closest_on_triangle(rest[v], *rest[tris[t]])
            self.assertGreaterEqual(np.linalg.norm(rest[v] - point), 0.03 - 2e-4)


class TestLiteDelassus(unittest.TestCase):
    """The sparse lite Delassus matrix ``W = H (dt^2 M^-1) H^T`` of rows on several particles, which
    ``FBALinearSolver.build_schur_lite_triplets`` merges from the slots of ``H`` sorted by particle."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def test_matches_the_dense_product(self):
        """Match the dense product for 300 rows of four slots over 49 particles, with unused slots, rows that take a
        particle twice, and pinned particles: each row keeps exactly the rows that share a particle with it, in
        order."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=6,
            dim_y=6,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.01,
            fix_left=True,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFBA(model, nsn_schur_mode="lite")
        solver.step(model.state(), model.state(), None, None, 0.01)
        ls, n, R = solver._linear_solver, model.particle_count, 300
        rng = np.random.default_rng(5)
        columns = rng.integers(-1, n, size=4 * R).astype(np.int32)
        columns[1::40] = columns[::40]  # a particle twice in a row
        values = rng.standard_normal((4 * R, 1, 3))
        arrays = [(columns, wp.int32), (values, _mat13d)]
        W = ls.build_schur_lite_triplets(R, *(wp.array(a, dtype=t) for a, t in arrays))
        offsets, nnz = W.offsets.numpy()[: R + 1], int(W.nnz)
        cols, vals = W.columns.numpy()[:nnz], W.values.numpy()[:nnz]
        dense = np.zeros((R, R))
        for i in range(R):
            row = cols[offsets[i] : offsets[i + 1]]
            self.assertTrue((np.diff(row) > 0).all())
            dense[i, row] = vals[offsets[i] : offsets[i + 1]]
        H = np.zeros((R, 3 * n))
        on = np.zeros((R, n), dtype=bool)
        for t in np.flatnonzero(columns >= 0):
            H[t // 4, 3 * columns[t] : 3 * columns[t] + 3] += values[t, 0]
            on[t // 4, columns[t]] = True
        s = np.repeat(ls._lite_s_d.numpy(), 3)
        reference = H @ (s[:, None] * H.T)
        np.testing.assert_allclose(dense, reference, rtol=1.0e-12, atol=1.0e-15 * np.abs(reference).max())
        stored = np.zeros((R, R), dtype=bool)
        for i in range(R):
            stored[i, cols[offsets[i] : offsets[i + 1]]] = True
        np.testing.assert_array_equal(stored, on.astype(int) @ on.T.astype(int) > 0)


class TestSelfContactSolve(unittest.TestCase):
    """A sheet of 2 cm cells, shifted 4 mm along x and y, falls from a few cm onto a like sheet on a box."""

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def _drop(self, start_gap, cells=10, fps=60, steps=60, mu=None, **solver_args):
        """Largest number of intersecting triangle pairs over the steps, then the gap between the sheets [m]."""
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.25), wp.quat_identity()), hx=0.25, hy=0.25, hz=0.25)
        grid = {
            "dim_x": cells,
            "dim_y": cells,
            "cell_x": 0.02,
            "cell_y": 0.02,
            "mass": 0.004,
            "tri_ke": 500.0,
            "particle_radius": 0.0,
        }
        for z, shift in ((0.5, 0.0), (0.5 + start_gap, 0.004)):
            corner = wp.vec3(shift - 0.01 * cells, shift - 0.01 * cells, z)
            builder.add_cloth_grid(pos=corner, rot=wp.quat_identity(), vel=wp.vec3(0.0), tri_ka=1e3, **grid)
        model = builder.finalize()
        if mu is not None:
            model.particle_mu = mu
        solver = newton.solvers.SolverFBA(model, iterations=10, **solver_args)
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.11)
        detector = TriMeshCollisionDetector(model)
        contacts, s0, s1, worst = pipeline.contacts(), model.state(), model.state(), 0
        for _ in range(steps):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / fps)
            s0, s1 = s1, s0
            detector.refit(s0.particle_q)
            detector.triangle_triangle_intersection_detection()
            worst = max(worst, int(detector.triangle_intersecting_triangles_count.numpy().sum()))
        if solver._self_contact is not None:
            # Each row of a particle-shape contact keeps its index in contacts; self contacts keep -1.
            n, count = solver._contact_count, int(contacts.soft_contact_count.numpy()[0])
            record, shape = solver._contact_record_d.numpy()[:n], solver._contact_shape_d.numpy()[:n]
            np.testing.assert_array_equal(record < 0, shape < 0)
            np.testing.assert_array_equal(np.sort(record[record >= 0]), np.arange(count))
            particle = solver._contact_particle_d.numpy()[:n][record >= 0]
            np.testing.assert_array_equal(particle, contacts.soft_contact_particle.numpy()[record[record >= 0]])
        self.solver = solver
        z = s0.particle_q.numpy()[:, 2]
        return worst, z[model.particle_count // 2 :].mean() - z[: model.particle_count // 2].mean()

    def test_vertex_face_holds_the_margin(self):
        """Rest the upper sheet 5 mm over the lower one with vertex-face contacts, without a crossing triangle.

        Without self contact it falls into the lower sheet. Contacts are found within 2.5 cm, the 5 mm margin plus the 2
        cm gap, more than the 1.25 cm the sheet falls in its last step.
        """
        self.assertGreater(self._drop(0.03)[0], 100)
        kw = {"particle_self_contact_margin": 0.005, "particle_self_contact_gap": 0.02}
        worst, gap = self._drop(0.03, particle_enable_self_contact=True, **kw)
        self.assertEqual(worst, 0)
        self.assertAlmostEqual(gap, 0.005, delta=2e-4)

    def test_groups_hold_their_margins(self):
        """Rest the upper sheet 8 mm over the lower one, the margin between the sheets' groups, with or without margins
        within the groups, or 5 mm as one group with that margin, through the device and the host contact assembly
        (face contacts); as one group without contacts within it, without margins or with a 0 margin, the upper sheet
        falls through the lower one."""
        kw = {"particle_enable_self_contact": True, "particle_self_contact_margin": 0.008}
        kw["particle_self_contact_gap"] = 0.02
        half = 11 * 11  # particles of a sheet
        for face_contacts in (False, True):
            for ids, margins, expected in (
                ([0] * half + [1] * half, [0.002, 0.002], 0.008),
                ([0] * half + [1] * half, None, 0.008),
                ([0] * 2 * half, [0.005], 0.005),
                ([0] * 2 * half, None, None),
                ([0] * 2 * half, [0.0], None),
            ):
                with self.subTest(face_contacts=face_contacts, margins=margins, expected=expected):
                    groups = {"particle_self_contact_group_ids": np.array(ids)}
                    groups["particle_self_contact_group_margins"] = margins
                    worst, gap = self._drop(0.03, face_contacts=face_contacts, **kw, **groups)
                    if expected is None:
                        self.assertGreater(worst, 100)
                    else:
                        self.assertEqual(worst, 0)
                        self.assertAlmostEqual(gap, expected, delta=2e-4)

    def test_edge_edge_leaves_flat_layers_to_vertex_face(self):
        """Rest the upper sheet 5 mm over the lower one with edge-edge contacts on, which flat layers barely take.

        Between flat layers, the vertex-face contacts hold the edges apart, and edge pairs take contacts only where
        they come nearer than their particles' nearest triangles.
        """
        kw = {"particle_self_contact_margin": 0.005, "particle_self_contact_gap": 0.02}
        worst, gap = self._drop(0.03, particle_enable_self_contact=True, particle_self_contact_edge_edge=True, **kw)
        self.assertEqual(worst, 0)
        self.assertAlmostEqual(gap, 0.005, delta=2e-4)
        n = self.solver._contact_count
        shape = self.solver._contact_shape_d.numpy()[:n]
        edge = (shape < 0) & (self.solver._contact_extra[1].numpy()[:n, 0] > 0.0)
        self.assertLess(edge.sum(), 0.05 * (shape < 0).sum())

    def _ribbon_rest(self, **solver_args):
        """Height [m] of the falling ribbon of :func:`_crossing_ribbons` over the pinned one where they cross after 1
        s, the solver, and its model."""
        model = _crossing_ribbons(0.01)
        model.particle_mu = 0.3
        kw = {"particle_self_contact_margin": 0.005, "particle_self_contact_gap": 0.01}
        kw |= {"particle_topological_contact_filter_threshold": 1}
        solver = newton.solvers.SolverFBA(model, iterations=10, particle_enable_self_contact=True, **kw, **solver_args)
        s0, s1 = model.state(), model.state()
        for _ in range(120):
            solver.step(s0, s1, None, None, 1.0 / 120.0)
            s0, s1 = s1, s0
        q = s0.particle_q.numpy()[20:]
        order = np.argsort(q[:, 0])
        return float(np.interp(0.0, q[order, 0], q[order, 2])), solver, model

    def test_edge_edge_holds_crossing_ribbons(self):
        """Rest a falling ribbon on a pinned one that it crosses in the middle of their edges, the 5 mm margin over it.

        The particles lie 1.6 cm or more from the crossing, beyond the reach of vertex-face contacts: without edge-edge
        contacts the ribbon falls through. The edge-edge contacts keep no friction anchors.
        """
        self.assertLess(self._ribbon_rest()[0], -0.1)
        height, solver, _ = self._ribbon_rest(particle_self_contact_edge_edge=True, friction_anchors=True)
        self.assertAlmostEqual(height, 0.005, delta=2e-4)
        self.assertFalse((solver._anchor_step_d.numpy() == solver._anchor_step).any())

    def test_set_particle_self_contact_thins_the_layers(self):
        """Settle the upper sheet 1 cm over a pinned one, then 4 mm over it once the margin changes to 4 mm."""
        builder = newton.ModelBuilder()
        grid = {"dim_x": 10, "dim_y": 10, "cell_x": 0.02, "cell_y": 0.02, "mass": 0.004, "tri_ke": 500.0}
        for z, shift in ((0.5, 0.0), (0.51, 0.004)):
            corner = wp.vec3(shift - 0.1, shift - 0.1, z)
            builder.add_cloth_grid(pos=corner, rot=wp.quat_identity(), vel=wp.vec3(0.0), tri_ka=1e3, **grid)
        model = builder.finalize()
        lower = model.particle_count // 2
        for array in (model.particle_mass, model.particle_inv_mass):
            values = array.numpy()
            values[:lower] = 0.0
            array.assign(values)
        kw = {"particle_self_contact_margin": 0.01, "particle_self_contact_edge_edge": True}
        solver = newton.solvers.SolverFBA(model, iterations=10, particle_enable_self_contact=True, **kw)
        s0, s1 = model.state(), model.state()
        detector = TriMeshCollisionDetector(model)
        for margin in (0.01, 0.004):
            solver.set_particle_self_contact(margin=margin)
            for _ in range(60):
                solver.step(s0, s1, None, None, 1.0 / 60.0)
                s0, s1 = s1, s0
            z = s0.particle_q.numpy()[:, 2]
            self.assertAlmostEqual(z[lower:].mean() - z[:lower].mean(), margin, delta=5e-4)
        detector.refit(s0.particle_q)
        detector.triangle_triangle_intersection_detection()
        self.assertEqual(int(detector.triangle_intersecting_triangles_count.numpy().sum()), 0)
        with self.assertRaises(ValueError):
            solver.set_particle_self_contact(margin=0.0)

    def test_refuse_edge_edge_without_vertex_face(self):
        """Refuse edge-edge contacts with vertex-vertex contacts, and a self-contact margin without self contact."""
        model = _crossing_ribbons(0.01)
        with self.assertRaises(ValueError):
            newton.solvers.SolverFBA(
                model,
                particle_enable_self_contact=True,
                particle_self_contact_mode="vertex_vertex",
                particle_self_contact_edge_edge=True,
            )
        with self.assertRaises(ValueError):
            newton.solvers.SolverFBA(model).set_particle_self_contact(margin=0.005)

    def test_self_contacts_hold_without_shape_contacts(self):
        """Keep the self contacts of a step that gets no contacts: a sheet falls onto a pinned sheet and rests on it.

        The step receives ``contacts=None``, so the self contacts are its only contacts. The upper sheet rests the 5 mm
        margin over the pinned lower one; without its self contacts it would fall through it.
        """
        builder = newton.ModelBuilder()
        grid = {"dim_x": 10, "dim_y": 10, "cell_x": 0.02, "cell_y": 0.02, "mass": 0.004, "tri_ke": 500.0}
        for z, shift in ((0.5, 0.0), (0.53, 0.004)):
            corner = wp.vec3(shift - 0.1, shift - 0.1, z)
            builder.add_cloth_grid(pos=corner, rot=wp.quat_identity(), vel=wp.vec3(0.0), tri_ka=1e3, **grid)
        model = builder.finalize()
        lower = model.particle_count // 2
        for array in (model.particle_mass, model.particle_inv_mass):
            values = array.numpy()
            values[:lower] = 0.0
            array.assign(values)
        kw = {"particle_self_contact_margin": 0.005, "particle_self_contact_gap": 0.02}
        solver = newton.solvers.SolverFBA(model, iterations=10, particle_enable_self_contact=True, **kw)
        s0, s1 = model.state(), model.state()
        for _ in range(60):
            solver.step(s0, s1, None, None, 1.0 / 60.0)
            s0, s1 = s1, s0
        z = s0.particle_q.numpy()[:, 2]
        self.assertAlmostEqual(z[lower:].mean() - z[:lower].mean(), 0.005, delta=5e-4)

    def test_vertex_vertex_holds_the_margin(self):
        """Nest the upper sheet between the lower particles, 2.65 to 3 cm over them, with frictionless vertex-vertex
        contacts.

        Particles 3 cm apart cover the 2.8 cm longest edges, so no triangle crosses. With friction, one NSN step per
        local-global iteration does not hold the tilted rows: the sheet sinks to 14 mm.
        """
        kw = {"particle_self_contact_mode": "vertex_vertex", "particle_self_contact_margin": 0.03}
        kw |= {"particle_self_contact_gap": 0.0015}
        worst, gap = self._drop(0.035, mu=0.0, particle_enable_self_contact=True, **kw)
        self.assertEqual(worst, 0)
        self.assertGreater(gap, 0.026)
        self.assertLess(gap, 0.030)

    def test_lite_nsn_holds_the_margin(self):
        """Hold the margin of both self-contact modes in the lite NSN as in the full NSN.

        Rows on several particles join a sparse solve, at steps of 1/600 s, where the lite Delassus approximation holds
        for this cloth.
        """
        vertex_face = {"particle_self_contact_margin": 0.005, "particle_self_contact_gap": 0.005}
        vertex_vertex = {"particle_self_contact_mode": "vertex_vertex", "particle_self_contact_margin": 0.03}
        vertex_vertex["particle_self_contact_gap"] = 0.0015
        for kw, start, low, high in ((vertex_face, 0.015, 0.0049, 0.0051), (vertex_vertex, 0.035, 0.0265, 0.030)):
            lite = {"particle_enable_self_contact": True, "nsn_schur_mode": "lite"}
            worst, gap = self._drop(start, cells=6, fps=600, steps=80, **kw, **lite)
            self.assertEqual(worst, 0)
            self.assertGreater(gap, low)
            self.assertLess(gap, high)

    def test_lite_accumulation_needs_the_sparse_lite_nsn(self):
        """Refuse ``nsn_lite_accumulate`` with the full operator or without friction, which never take the sparse lite
        NSN."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.004,
        )
        model = builder.finalize()
        for kw in ({"nsn_schur_mode": "full"}, {"nsn_schur_mode": "lite", "friction": False}):
            with self.subTest(**kw), self.assertRaises(ValueError):
                newton.solvers.SolverFBA(model, nsn_lite_accumulate=True, **kw)

    def _trampoline(self, **solver_args):
        """A 4 cm patch on a 20 cm sheet pinned along its edges, after 1 s at steps of 1/60 s in the lite NSN: the mean
        gap [m] between the patch and the sheet under it, the largest number of intersecting triangle pairs over the
        steps, and the final positions."""
        builder = newton.ModelBuilder()
        grid = {"rot": wp.quat_identity(), "vel": wp.vec3(0.0), "cell_x": 0.02, "cell_y": 0.02, "mass": 0.004}
        grid |= {"tri_ke": 500.0, "tri_ka": 1e3, "particle_radius": 0.0}
        builder.add_cloth_grid(pos=wp.vec3(-0.1, -0.1, 0.5), dim_x=10, dim_y=10, **grid)
        sheet = builder.particle_count
        for i in range(sheet):
            if i % 11 in (0, 10) or i // 11 in (0, 10):
                builder.particle_mass[i] = 0.0
        builder.add_cloth_grid(pos=wp.vec3(-0.016, -0.016, 0.506), dim_x=2, dim_y=2, **grid)
        model = builder.finalize()
        kw = {"particle_self_contact_margin": 0.005, "particle_self_contact_gap": 0.005, "nsn_linear_iterations": 10}
        solver = newton.solvers.SolverFBA(
            model, iterations=10, nsn_schur_mode="lite", particle_enable_self_contact=True, **kw, **solver_args
        )
        detector = TriMeshCollisionDetector(model)
        s0, s1, worst = model.state(), model.state(), 0
        for _ in range(60):
            solver.step(s0, s1, None, None, 1.0 / 60.0)
            s0, s1 = s1, s0
            detector.refit(s0.particle_q)
            detector.triangle_triangle_intersection_detection()
            worst = max(worst, int(detector.triangle_intersecting_triangles_count.numpy().sum()))
        q = s0.particle_q.numpy()
        gaps = []
        for p in q[sheet:]:
            # the sheet's height under p, from its three nearest particles weighted by inverse distance
            d = np.linalg.norm(q[:sheet, :2] - p[:2], axis=1)
            near = np.argsort(d)[:3]
            gaps.append(p[2] - np.average(q[near, 2], weights=1.0 / d[near]))
        self.solver = solver
        return float(np.mean(gaps)), worst, q

    def test_lite_accumulation_holds_the_margin_under_a_load(self):
        """Hold a patch the 5 mm margin over a pinned sheet that it presses down, at steps of 1/60 s, when the
        iterations of a step accumulate their contact forces.

        The lite Delassus approximation alone, which overrates the sheet's give under the patch by its stiffness, lets
        the patch sink 2 mm into the margin. The correction runs from one graph per step; launched without the graph,
        it moves the cloth alike.
        """
        self.assertLess(self._trampoline()[0], 0.004)
        gap, worst, replayed = self._trampoline(nsn_lite_accumulate=True)
        self.assertEqual(worst, 0)
        self.assertAlmostEqual(gap, 0.005, delta=1.0e-4)
        self.assertTrue(self.solver._lite_iter_graph)
        with mock.patch.object(wp, "ScopedCapture", side_effect=RuntimeError):
            _, _, launched = self._trampoline(nsn_lite_accumulate=True)
        self.assertIs(self.solver._lite_iter_graph, False)
        np.testing.assert_array_equal(launched, replayed)

    def test_lite_rows_hold_particles_among_self_contacts(self):
        """Lift a 2 x 2 corner of both 6-cell sheets 3 cm in 0.1 s with rows, the upper corner held 2 mm over the lower
        one.

        That is inside the 5 mm contact radius. Contacts on held particles carry no force, so the conflict does not
        throw the cloth, and the other contacts keep every triangle apart.
        """
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.25), wp.quat_identity()), hx=0.25, hy=0.25, hz=0.25)
        grid = {
            "dim_x": 6,
            "dim_y": 6,
            "cell_x": 0.02,
            "cell_y": 0.02,
            "mass": 0.004,
            "tri_ke": 500.0,
            "tri_ka": 1e3,
            "particle_radius": 0.0,
        }
        for z, shift in ((0.5, 0.0), (0.505, 0.004)):
            corner = wp.vec3(shift - 0.06, shift - 0.06, z)
            builder.add_cloth_grid(pos=corner, rot=wp.quat_identity(), vel=wp.vec3(0.0), **grid)
        model = builder.finalize()
        kw = {"particle_self_contact_margin": 0.005, "particle_self_contact_gap": 0.005}
        solver = newton.solvers.SolverFBA(
            model, iterations=10, nsn_schur_mode="lite", particle_enable_self_contact=True, **kw
        )
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.11)
        detector = TriMeshCollisionDetector(model)
        contacts, s0, s1 = pipeline.contacts(), model.state(), model.state()
        held = [0, 1, 7, 8, 49, 50, 56, 57]
        base = model.particle_q.numpy()[held]
        base[4:] = base[:4] + np.array([0.004, 0.004, 0.002])
        for k in range(120):
            target = base + np.array([0.0, 0.0, 0.03 * min(1.0, (k + 1) / 60.0)])
            directions = np.tile(np.eye(3), (8, 1))
            solver.set_particle_constraints(
                np.repeat(held, 3)[:, None], np.ones((24, 1)), target.reshape(-1), directions=directions
            )
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 1.0 / 600.0)
            s0, s1 = s1, s0
            if k % 10 == 9:
                detector.refit(s0.particle_q)
                detector.triangle_triangle_intersection_detection()
                self.assertEqual(int(detector.triangle_intersecting_triangles_count.numpy().sum()), 0)
        self.assertLess(float(np.abs(s0.particle_qd.numpy()).max()), 0.01)
        np.testing.assert_allclose(s0.particle_q.numpy()[held], target, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
