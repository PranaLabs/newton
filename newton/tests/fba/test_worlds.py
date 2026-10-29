# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolation of FBA worlds, including overlapping geometry and partial resets."""

import unittest

import numpy as np
import warp as wp
import warp.sparse as wps

import newton
from newton._src.solvers.fba.nsn_pcr_solver import NSNPCRSolver, group_world_rows
from newton._src.solvers.fba.self_contact import SelfContact, traversal_rank_kernel
from newton.solvers import SolverFBA


def _cloth(height=0.005, mass=0.01):
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, height),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.2, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        mass=mass,
        tri_ke=1000.0,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=0.01,
        edge_kd=0.0,
        particle_radius=0.01,
    )
    return builder


def _model(builders, device):
    builder = newton.ModelBuilder()
    for source in builders:
        builder.add_world(source)
    builder.add_ground_plane()
    return builder.finalize(device=device)


class TestFBAWorlds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])

    def test_pcr_independent_recurrences(self):
        """One truncated iteration, zero RHS and a refused system cannot change a neighboring world's step."""
        blocks = [np.array([[2.0, 0.5], [0.5, 1.0]]), np.array([[10.0, -2.0], [-2.0, 3.0]]), np.zeros((2, 2))]
        matrix = np.zeros((8, 8))
        worlds = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        for w, block in enumerate(blocks):
            ids = np.flatnonzero(worlds == w)
            matrix[np.ix_(ids, ids)] = block
        matrix[np.ix_([3, 7], [3, 7])] = np.eye(2)
        rhs = np.array([1.0, 300.0, 1.0, 0.0, 2.0, -700.0, 2.0, 0.0])
        for device in self.devices:
            for sparse in (False, True):
                with self.subTest(device=device, sparse=sparse):
                    solver = NSNPCRSolver(8, device, max_iter=1)
                    solver.world_rows = group_world_rows(worlds, device)
                    A = wp.array(matrix, dtype=wp.float64, device=device)
                    if sparse:
                        i, j = np.nonzero(matrix)
                        A = wps.bsr_zeros(8, 8, wp.float64, device=device)
                        wps.bsr_set_from_triplets(
                            A,
                            wp.array(i, dtype=wp.int32, device=device),
                            wp.array(j, dtype=wp.int32, device=device),
                            wp.array(matrix[i, j], dtype=wp.float64, device=device),
                        )
                    b = wp.array(rhs, dtype=wp.float64, device=device)
                    out = wp.empty(8, dtype=wp.float64, device=device)
                    solve = solver.solve_sparse_fused if sparse else solver.solve
                    solve(A, b, out)
                    for w in range(4):
                        ids = np.flatnonzero(worlds == w)
                        reference = NSNPCRSolver(2, device, max_iter=1)
                        expected = wp.empty(2, dtype=wp.float64, device=device)
                        reference.solve(
                            wp.array(matrix[np.ix_(ids, ids)], dtype=wp.float64, device=device),
                            wp.array(rhs[ids], dtype=wp.float64, device=device),
                            expected,
                        )
                        np.testing.assert_allclose(out.numpy()[ids], expected.numpy(), atol=1e-12, rtol=1e-12)

    def test_fused_world_pcr(self):
        """The fused sparse PCR of worlds over chunks of 256 rows takes each world's own steps, as the per-world
        recurrence does: interleaved worlds of three and two chunks with partial tails, a three-row world that
        runs out of rows before the ten iterations end, a zero right-hand side, refused indefinite and near-null
        directions and NaN and Inf right-hand sides beside them. A smaller system after a larger one and a replayed
        CUDA graph solve alike; runs checked for convergence, more chunks than the fused scratch holds and solves off
        CUDA keep the per-world recurrence."""
        rng = np.random.default_rng(5)

        def spd(n):
            S = np.where(rng.random((n, n)) < 0.02, rng.standard_normal((n, n)), 0.0)
            S = S + S.T
            np.fill_diagonal(S, 0.0)
            return S + np.diag(np.abs(S).sum(axis=1) + rng.uniform(0.5, 2.0, n))

        def system(blocks, rhs):
            """Block-diagonal worlds with their rows shuffled together."""
            worlds = np.repeat(np.arange(len(blocks)), [len(B) for B in blocks])
            order = rng.permutation(len(worlds))  # system row of each world row
            A, b = np.zeros((len(worlds),) * 2), np.zeros(len(worlds))
            start = 0
            for B, v in zip(blocks, rhs, strict=True):
                ids = order[start : start + len(B)]
                A[np.ix_(ids, ids)], b[ids] = B, v
                start += len(B)
            world = np.empty(len(worlds), dtype=int)
            world[order] = worlds
            return A, b, world, np.split(order, np.cumsum([len(B) for B in blocks])[:-1])

        healthy = [spd(600), spd(3), spd(257)]
        # positive definite, but its curvature along (1, -1) is 1e-8 of its diagonal's, under 1 / max_inverse_gain
        near_null = np.array([[1.0, 1.0 - 1e-8], [1.0 - 1e-8, 1.0]])
        blocks = [*healthy, spd(40), np.array([[1.0, 2.0], [2.0, 1.0]]), spd(20), spd(30), near_null]
        rhs = [rng.standard_normal(len(B)) for B in blocks]
        rhs[3][:] = 0.0
        rhs[4][:] = (1.0, -1.0)  # p A p < 0
        rhs[5][7], rhs[6][3] = np.nan, np.inf
        rhs[7][:] = (1.0, -1.0)
        large = system(blocks, rhs)
        small = system([spd(300), spd(5)], [rng.standard_normal(300), rng.standard_normal(5)])
        # 769 rows, but 513 worlds of two chunks each
        wide = system([spd(257), *([np.eye(1)] * 512)], [rng.standard_normal(257), *rng.standard_normal((512, 1))])

        def solve(device, A, b, world, method="solve_sparse_fused", solver=None, max_iter=10):
            n = len(b)
            i, j = np.nonzero(A)
            W = wps.bsr_from_triplets(
                n,
                n,
                wp.array(i, dtype=wp.int32, device=device),
                wp.array(j, dtype=wp.int32, device=device),
                wp.array(A[i, j], dtype=wp.float64, device=device),
            )
            solver = solver or NSNPCRSolver(len(large[1]), device, max_iter=max_iter)
            solver.world_rows = group_world_rows(world, device)
            x = wp.zeros(n, dtype=wp.float64, device=device)
            iterations = getattr(solver, method)(W, wp.array(b, dtype=wp.float64, device=device), x)
            return x.numpy(), iterations, (solver, W, x)

        for device in self.devices:
            with self.subTest(device=device):
                A, b, world, ids_of = large
                # off CUDA, the kernels of the fused path run through their method
                fused = "solve_sparse_fused" if wp.get_device(device).is_cuda else "_solve_worlds_fused"
                x, iterations, (solver, W, out) = solve(device, A, b, world, fused)
                x_ref, iterations_ref, _ = solve(device, A, b, world, "solve_sparse")
                self.assertEqual(iterations, iterations_ref)
                for w, B in enumerate(blocks):
                    ids = ids_of[w]
                    if w < len(healthy):
                        # the world alone, by the per-world recurrence with one world: its sums differ in order only
                        alone, _, _ = solve(device, B, rhs[w], np.zeros(len(B), dtype=int), "solve_sparse")
                        for result in (x_ref[ids], x[ids]):
                            np.testing.assert_allclose(result, alone, rtol=1e-10, atol=1e-12 * np.abs(alone).max())
                    else:
                        # stopped at the first iteration: no step, no NaN
                        np.testing.assert_array_equal(x[ids], 0.0)
                        np.testing.assert_array_equal(x_ref[ids], 0.0)
                # a smaller system on the same solver's scratch solves as on a new solver
                x_small, _, _ = solve(device, *small[:3], fused, solver)
                np.testing.assert_array_equal(x_small, solve(device, *small[:3], fused)[0])
                # runs checked for convergence and grids past the fused scratch take the per-world recurrence
                for case, max_iter in ((small, 30), (wide, 10)):
                    fused_x, fused_iterations, _ = solve(device, *case[:3], max_iter=max_iter)
                    ref_x, ref_iterations, _ = solve(device, *case[:3], "solve_sparse", max_iter=max_iter)
                    self.assertEqual(fused_iterations, ref_iterations)
                    np.testing.assert_array_equal(fused_x, ref_x)
                # mismatched shapes are refused before any kernel runs
                with self.assertRaises(ValueError):
                    solver.solve_sparse_fused(W, wp.zeros(len(b) - 1, dtype=wp.float64, device=device), out)
                # off CUDA the fused entry keeps the per-world recurrence
                if not wp.get_device(device).is_cuda:
                    np.testing.assert_array_equal(solve(device, A, b, world)[0], x_ref)
                    continue
                # a captured solve replays for new right-hand sides
                b_d = wp.array(b, dtype=wp.float64, device=device)
                solver.world_rows = group_world_rows(world, device)
                with wp.ScopedCapture(device=device) as capture:
                    solver.solve_sparse_fused(W, b_d, out)
                for scale in (1.0, -2.0):
                    b_d.assign(scale * b)
                    wp.capture_launch(capture.graph)
                    expected, _, _ = solve(device, A, scale * b, world)
                    np.testing.assert_array_equal(out.numpy(), expected)

    def test_fused_world_pcr_many_worlds(self):
        """More worlds than 1024 chunks of 256 rows take the fused sparse PCR within scratch sized by ``max_n``:
        1100 interleaved two-row worlds, one of a zero right-hand side and one refused as indefinite, step as the
        per-world recurrence does; on CUDA the public entry dispatches to them, replays a captured graph and reuses
        its scratch after a smaller system."""
        rng = np.random.default_rng(11)
        nw = 1100
        blocks = []
        for _ in range(nw):
            S = rng.standard_normal((2, 2))
            blocks.append(S @ S.T + np.eye(2))
        blocks[5] = np.array([[1.0, 2.0], [2.0, 1.0]])  # p A p < 0 along (1, -1)
        b = rng.standard_normal(2 * nw)
        b[2 * 9 : 2 * 9 + 2] = 0.0
        b[2 * 5 : 2 * 5 + 2] = (1.0, -1.0)
        order = rng.permutation(2 * nw)  # system row of each world row
        world = np.empty(2 * nw, dtype=int)
        world[order] = np.repeat(np.arange(nw), 2)
        local = order.reshape(nw, 2)
        i = np.concatenate([np.repeat(rows, 2) for rows in local])
        j = np.concatenate([np.tile(rows, 2) for rows in local])
        values = np.concatenate([B.ravel() for B in blocks])
        rhs = np.empty(2 * nw)
        rhs[order] = b

        def solve(device, method, solver=None, keep=(i, j, values, rhs, world)):
            i, j, values, rhs, world = keep
            n = len(rhs)
            W = wps.bsr_from_triplets(
                n,
                n,
                wp.array(i, dtype=wp.int32, device=device),
                wp.array(j, dtype=wp.int32, device=device),
                wp.array(values, dtype=wp.float64, device=device),
            )
            solver = solver or NSNPCRSolver(2 * nw, device, max_iter=10)
            solver.world_rows = group_world_rows(world, device)
            x = wp.zeros(n, dtype=wp.float64, device=device)
            iterations = getattr(solver, method)(W, wp.array(rhs, dtype=wp.float64, device=device), x)
            return x.numpy(), iterations, (solver, W, x)

        for device in self.devices:
            with self.subTest(device=device):
                solver = NSNPCRSolver(2 * nw, device, max_iter=10)
                # one chunk per world: a grid of 1100 blocks, and one rho, b . b and flag per world
                self.assertGreaterEqual(solver._fused_partials.shape[1], nw)
                for state in (solver._world_fused_rho.shape[1], solver._world_fused_b2.shape[0]):
                    self.assertGreaterEqual(state, nw)
                self.assertGreaterEqual(solver._world_fused_stopped.shape[0], nw)
                x, iterations, _ = solve(device, "_solve_worlds_fused", solver)
                x_ref, iterations_ref, _ = solve(device, "solve_sparse")
                self.assertEqual(iterations, iterations_ref)
                self.assertEqual(iterations, 2)
                np.testing.assert_allclose(x, x_ref, rtol=1e-10, atol=1e-12 * np.abs(x_ref).max())
                for w in (5, 9):  # refused at the first iteration, and a zero right-hand side: no step
                    np.testing.assert_array_equal(x[local[w]], 0.0)
                    np.testing.assert_array_equal(x_ref[local[w]], 0.0)
                healthy = np.setdiff1d(np.arange(nw), [5, 9])
                for w in healthy[:: nw // 10]:  # two iterations solve a two-row world
                    np.testing.assert_allclose(x[local[w]], np.linalg.solve(blocks[w], b[2 * w : 2 * w + 2]), rtol=1e-8)
                if not wp.get_device(device).is_cuda:
                    # off CUDA the public fused entry keeps the per-world recurrence
                    np.testing.assert_array_equal(solve(device, "solve_sparse_fused")[0], x_ref)
                    continue
                fused, _, (solver, W, out) = solve(device, "solve_sparse_fused")
                np.testing.assert_array_equal(fused, solve(device, "_solve_worlds_fused")[0])
                # a smaller system on the same scratch, then the large one again, solve as on new solvers
                small = (np.r_[0, 0, 1, 1], np.r_[0, 1, 0, 1], blocks[0].ravel(), b[:2], np.zeros(2, dtype=int))
                np.testing.assert_array_equal(
                    solve(device, "solve_sparse_fused", solver, small)[0],
                    solve(device, "solve_sparse_fused", keep=small)[0],
                )
                b_d = wp.array(rhs, dtype=wp.float64, device=device)
                solver.world_rows = group_world_rows(world, device)
                with wp.ScopedCapture(device=device) as capture:
                    solver.solve_sparse_fused(W, b_d, out)
                for scale in (1.0, -2.0):
                    b_d.assign(scale * rhs)
                    wp.capture_launch(capture.graph)
                    expected, _, _ = solve(device, "solve_sparse_fused", keep=(i, j, values, scale * rhs, world))
                    np.testing.assert_array_equal(out.numpy(), expected)

    def test_overlapping_self_contacts(self):
        """All three self-contact queries exclude nearby geometry belonging to another world."""
        for device in self.devices:
            model = _model([_cloth(0.0), _cloth(0.005)], device)
            for mode, edges in (("vertex_vertex", False), ("vertex_face", False), ("vertex_face", True)):
                with self.subTest(device=device, mode=mode, edges=edges):
                    query = SelfContact(model, mode, 0.02, edge_edge=edges)
                    self.assertEqual(query.find(model.particle_q), 0)
                    # The same geometry within one world really does collide.
                    model.particle_world.zero_()
                    self.assertGreater(query.find(model.particle_q), 0)
                    model.particle_world.assign(np.repeat(np.arange(2, dtype=np.int32), 4))

    def test_grouped_self_contacts_keep_original_selection(self):
        """Several worlds search their own worlds' trees, yet take the very contacts of the tree of all triangles:
        three layers per world, two of them coincident, so that particles meet triangles at equal distances, at
        rest, after the layers move, with margins of groups, rings and rest distance, more than 64 candidates,
        edge-edge contacts of crossing ribbons, a world without triangles and particles that change their worlds and back."""

        @wp.kernel
        def subtree(bvh: wp.uint64, roots: wp.array[wp.int32], items: wp.array2d[wp.int32]):
            w = wp.tid()
            query = wp.bvh_query_aabb(bvh, wp.vec3(-1.0e6), wp.vec3(1.0e6), roots[w])
            t = wp.int32(0)
            while wp.bvh_query_next(query, t):
                items[w, t] = 1

        def layers(dim, empty=False):
            builder = newton.ModelBuilder()
            if empty:  # a world of particles without triangles
                builder.add_particle(wp.vec3(0.01, 0.01, 0.0025), wp.vec3(0.0), 0.01)
                return builder
            for z in (0.0, 0.005, 0.005):
                builder.add_cloth_grid(
                    pos=wp.vec3(0.0, 0.0, z),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0),
                    dim_x=dim,
                    dim_y=dim,
                    cell_x=0.02,
                    cell_y=0.02,
                    mass=0.01,
                    particle_radius=0.002,
                )
            return builder

        def ribbons():
            """Two ribbons of 4 cm by 8 mm cells crossing at right angles, 3 mm apart."""
            builder = newton.ModelBuilder()
            for ribbon in (
                ((-0.004, -0.18, 0.0), (0, 1, 0), (1, 0, 0)),
                ((-0.18, -0.004, 0.003), (1, 0, 0), (0, 1, 0)),
            ):
                start, along, across = (np.asarray(v, dtype=float) for v in ribbon)
                vertices = [start + 0.04 * i * along + 0.008 * j * across for i in range(10) for j in range(2)]
                indices = [k for i in range(9) for k in (2 * i, 2 * i + 2, 2 * i + 1, 2 * i + 1, 2 * i + 2, 2 * i + 3)]
                builder.add_cloth_mesh(
                    pos=wp.vec3(0.0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(0.0),
                    vertices=[wp.vec3(*v) for v in vertices],
                    indices=indices,
                    density=0.2,
                )
            return builder

        def original(model, **options):
            """The query of the tree of all triangles, as before the worlds were grouped."""
            contact = SelfContact(model, "vertex_face", **options)
            contact.bvh = wp.Bvh(contact.lower, contact.upper)
            if contact.edge_edge:
                contact.edge_bvh = wp.Bvh(contact.edge_lower, contact.edge_upper)
            empty = wp.empty(0, dtype=wp.int32, device=model.device)
            contact.traversal_rank = contact.roots = contact.edge_roots = empty
            contact.initial_world = None
            return contact

        def pair(model, **options):
            """A grouped query and the query of the tree of all triangles, the grouped one ranking its triangles by
            that very tree, whose build need not repeat on every device; on the CPU it does, and the ranks of the
            grouped query's own tree of all triangles are the same."""
            grouped, reference = SelfContact(model, "vertex_face", **options), original(model, **options)
            rank = wp.empty(model.tri_count, dtype=wp.int32, device=model.device)
            wp.launch(traversal_rank_kernel, 1, [reference.bvh.id], [rank], device=model.device)
            np.testing.assert_array_equal(np.sort(rank.numpy()), np.arange(model.tri_count))
            if not model.device.is_cuda:
                np.testing.assert_array_equal(grouped.traversal_rank.numpy(), rank.numpy())
            grouped.traversal_rank = rank
            return grouped, reference

        def same(grouped, reference, q):
            n = grouped.find(q)
            self.assertEqual(n, reference.find(q))
            self.assertGreater(n, 0)
            for name in ("found_particles", "found_alpha", "found_normal"):
                np.testing.assert_array_equal(getattr(grouped, name).numpy()[:n], getattr(reference, name).numpy()[:n])
            # the same candidates, in another order
            count = grouped.candidate_count.numpy()
            np.testing.assert_array_equal(count, reference.candidate_count.numpy())
            a, b = grouped.candidates.numpy(), reference.candidates.numpy()
            for v in np.flatnonzero(count):
                np.testing.assert_array_equal(np.sort(a[v, : count[v]]), np.sort(b[v, : count[v]]))
            return n

        def check(device):
            naive_differs = False
            for worlds in (2, 4):
                with self.subTest(worlds=worlds):
                    model = _model([layers(3) for _ in range(worlds)] + [layers(3, empty=True)], device)
                    world, triangles = model.particle_world.numpy().copy(), model.tri_indices.numpy()
                    q = model.particle_q
                    options = {"query_radius": 0.01}
                    grouped, reference = pair(model, **options)
                    # each world's root reaches its own triangles only, and the world without triangles has none
                    roots = grouped.roots.numpy()
                    self.assertEqual(roots[-1], -1)
                    self.assertTrue(np.all(roots[:-1] >= 0))
                    items = wp.zeros((worlds, model.tri_count), dtype=wp.int32, device=device)
                    wp.launch(subtree, worlds, [grouped.bvh.id, grouped.roots], [items], device=device)
                    np.testing.assert_array_equal(
                        items.numpy(), world[triangles[:, 0]][None, :] == np.arange(worlds)[:, None]
                    )
                    same(grouped, reference, q)
                    self.assertEqual(grouped.candidate_count.numpy()[world == worlds], 0)
                    # without the ranks, the first of equals in the grouped query is another triangle
                    naive = SelfContact(model, "vertex_face", **options)
                    naive.traversal_rank = wp.empty(0, dtype=wp.int32, device=device)
                    n = naive.find(q)
                    naive_differs |= not np.array_equal(naive.found_alpha.numpy()[:n], grouped.found_alpha.numpy()[:n])
                    naive_differs |= not np.array_equal(
                        naive.found_particles.numpy()[:n], grouped.found_particles.numpy()[:n]
                    )
                    # the layers move: refit, with the ties kept
                    moved = q.numpy() + np.where(world[:, None] % 2 == 0, [0.003, -0.002, 0.0], [0.0, 0.001, 0.0])
                    moved = wp.array(moved.astype(np.float32), dtype=wp.vec3, device=device)
                    same(grouped, reference, moved)
                    # particles that change their worlds search the whole tree by the ranks, and come back
                    model.particle_world.zero_()
                    same(grouped, reference, q)
                    self.assertEqual(grouped.world_changed.numpy()[0], 1)
                    model.particle_world.assign(world)
                    same(grouped, reference, q)
                    self.assertEqual(grouped.world_changed.numpy()[0], 0)
                    # margins of groups, the layers' groups
                    groups = np.repeat(np.arange(3 * worlds + 1) % 3, [16] * 3 * worlds + [1])
                    both = pair(model, **options)
                    for contact in both:
                        contact.set_groups(groups, np.array([0.002, 0.0, 0.001]), 0.003, 0.004)
                    same(*both, q)
                    # rings and rest distance
                    extra = {"rings": 1, "rest_radius": 0.004}
                    same(*pair(model, **options, **extra), q)
            if not wp.get_device(device).is_cuda:  # the negative control: whether the grouped order of another
                self.assertTrue(naive_differs)  # backend's trees flips a tie depends on its build
            # edge-edge contacts of ribbons crossing 3 mm apart in co-located worlds, sorted by their edge pairs alike
            model = _model([ribbons() for _ in range(3)], device)
            options = {"query_radius": 0.01, "rings": 1, "edge_edge": True}
            grouped, reference = pair(model, **options)
            n = same(grouped, reference, model.particle_q)
            self.assertGreater(n, int(grouped.found.numpy().sum()))
            # more than 64 candidates of a particle grow the candidate lists alike
            model = _model([layers(6), layers(6)], device)
            grouped, reference = pair(model, query_radius=0.2)
            same(grouped, reference, model.particle_q)
            self.assertGreater(grouped.candidates.shape[1], 64)

        for device in self.devices:
            with self.subTest(device=device):
                check(device)

    def test_contact_steps_match_separate_worlds(self):
        """Heterogeneous overlapping worlds, including an empty-contact world, match separate solves."""
        for device in self.devices:
            sources = [_cloth(), _cloth(0.009, 0.03), _cloth(1.0)]
            for options in (
                {"nsn_schur_mode": "full"},
                {"nsn_sparse_storage": True},
                {"nsn_schur_mode": "lite"},
                {"friction": False},
            ):
                with self.subTest(device=device, options=options):
                    models = [_model(sources, device)] + [_model([source], device) for source in sources]
                    results = []
                    for model in models:
                        solver = SolverFBA(model, iterations=2, nsn_linear_iterations=2, **options)
                        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.03)
                        contacts = pipeline.contacts()
                        a, b = model.state(), model.state()
                        for _ in range(2):
                            pipeline.collide(a, contacts)
                            solver.step(a, b, None, contacts, 0.01)
                            a, b = b, a
                        results.append(a.particle_q.numpy())
                    np.testing.assert_allclose(results[0], np.concatenate(results[1:]), atol=2e-6, rtol=0.0)

    def test_per_world_gravity_and_empty_world(self):
        """Apply each world's own gravity, beside a world of only a static shape."""
        for device in self.devices:
            with self.subTest(device=device):
                empty_world = newton.ModelBuilder()
                empty_world.add_shape_sphere(-1, xform=wp.transform((10.0, 0.0, 0.0)), radius=0.1)
                model = _model([_cloth(1.0), empty_world, _cloth(1.0)], device)
                gravity = np.array([[1.0, 0.0, -2.0], [0.0, 0.0, 0.0], [0.0, -3.0, 1.0]], dtype=np.float32)
                model.gravity.assign(gravity)
                solver = SolverFBA(model, iterations=1)
                a, b = model.state(), model.state()
                solver.step(a, b, None, None, 0.01)
                expected = a.particle_q.numpy() + 0.01 * a.particle_qd.numpy()
                expected += 0.0001 * gravity[model.particle_world.numpy()]
                np.testing.assert_allclose(b.particle_q.numpy(), expected, atol=2e-7, rtol=0.0)

    def test_masked_reset_preserves_other_world_history(self):
        """Clear selected history with flags=0 without changing either world's physical state."""
        for device in self.devices:
            with self.subTest(device=device):
                model = _model([_cloth(), _cloth()], device)
                solver = SolverFBA(
                    model,
                    contact_warmstart=True,
                    friction_anchors=True,
                    nsn_sparse_storage=True,
                    iterations=2,
                    nsn_linear_iterations=2,
                )
                pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.03)
                contacts = pipeline.contacts()
                a, b = model.state(), model.state()
                pipeline.collide(a, contacts)
                solver.step(a, b, None, contacts, 0.01)
                group = solver._warm_start.particle
                before = group.prev_lam.numpy().copy()
                keys = group.prev_sorted.numpy()[: group.prev_count]
                order = group.prev_order.numpy()[: group.prev_count]
                world = model.particle_world.numpy()[keys // model.shape_count]
                rows = (3 * order[:, None] + np.arange(3)).reshape(-1)
                chosen = np.repeat(world == 0, 3)
                self.assertGreater(np.linalg.norm(before[rows[~chosen]]), 0.0)
                anchors = solver._anchor_previous_pressed_d.numpy().copy()
                anchor_keys = solver._anchor_previous_key_d.numpy()[: solver._anchor_count]
                anchor_world = model.particle_world.numpy()[anchor_keys // (model.shape_count + 1)]
                positions = b.particle_q.numpy().copy()
                solver.reset(b, wp.array([False, False, False], dtype=wp.bool, device=device))
                np.testing.assert_array_equal(group.prev_lam.numpy(), before)
                solver.reset(b, wp.array([True, False, False], dtype=wp.bool, device=device), flags=0)
                np.testing.assert_array_equal(group.prev_lam.numpy()[rows[chosen]], 0.0)
                np.testing.assert_array_equal(group.prev_lam.numpy()[rows[~chosen]], before[rows[~chosen]])
                after = solver._anchor_previous_pressed_d.numpy()[: solver._anchor_count]
                np.testing.assert_array_equal(after[anchor_world == 0], 0)
                np.testing.assert_array_equal(
                    after[anchor_world == 1], anchors[: solver._anchor_count][anchor_world == 1]
                )
                np.testing.assert_array_equal(b.particle_q.numpy(), positions)
                with self.assertRaises(ValueError):
                    solver.reset(b, wp.array([True, False], dtype=wp.bool, device=device))
                mask = wp.array([True, False, False], dtype=wp.bool, device=device)
                solver.reset(b, mask)
                pipeline.reset_contact_matching(mask)
                pipeline.collide(b, contacts)
                solver.step(b, a, None, contacts, 0.01)
                fresh = SolverFBA(
                    model,
                    contact_warmstart=True,
                    friction_anchors=True,
                    nsn_sparse_storage=True,
                    iterations=2,
                    nsn_linear_iterations=2,
                )
                expected = model.state()
                fresh.step(b, expected, None, contacts, 0.01)
                selected = model.particle_world.numpy() == 0
                np.testing.assert_allclose(
                    a.particle_q.numpy()[selected], expected.particle_q.numpy()[selected], atol=2e-6, rtol=0.0
                )

    def test_reset_state_flags(self):
        """Restore only selected worlds and requested fields, preserving inputs and reduced joint state."""
        source = _cloth()
        body = source.add_body(xform=wp.transform((1.0, 0.0, 0.5)))
        source.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        fields = {
            "body_q": newton.StateFlags.BODY_Q,
            "body_qd": newton.StateFlags.BODY_QD,
            "particle_q": newton.StateFlags.PARTICLE_Q,
            "particle_qd": newton.StateFlags.PARTICLE_QD,
        }
        for device in self.devices:
            model = _model([source, source], device)
            model.body_qd.assign(np.full((model.body_count, 6), 0.25, dtype=np.float32))
            solver = SolverFBA(model)
            state = model.state()
            mask = wp.array([True, False, False], dtype=wp.bool, device=device)
            baseline = {name: getattr(model, name).numpy().copy() for name in fields}
            dirty = {name: value + 1.0 for name, value in baseline.items()}
            cases = [
                None,
                0,
                *fields.values(),
                newton.StateFlags.JOINT_Q | newton.StateFlags.JOINT_QD,
                newton.StateFlags.BODY_F | newton.StateFlags.PARTICLE_F,
            ]
            for flags in cases:
                with self.subTest(device=device, flags=flags):
                    for name, value in dirty.items():
                        getattr(state, name).assign(value)
                    for name in ("body_f", "particle_f", "joint_q", "joint_qd"):
                        value = getattr(state, name)
                        if value is not None:
                            value.fill_(2.0)
                    preserved = {
                        name: getattr(state, name).numpy().copy()
                        for name in ("body_f", "particle_f", "joint_q", "joint_qd")
                        if getattr(state, name) is not None
                    }
                    solver.reset(state, mask, flags=flags)
                    for name, flag in fields.items():
                        expected = dirty[name].copy()
                        world = model.body_world if name.startswith("body") else model.particle_world
                        if flags is None or flags & flag:
                            selected = world.numpy() == 0
                            expected[selected] = baseline[name][selected]
                        np.testing.assert_array_equal(getattr(state, name).numpy(), expected)
                    for name, value in preserved.items():
                        np.testing.assert_array_equal(getattr(state, name).numpy(), value)
            solver.reset(state)
            for name, value in baseline.items():
                np.testing.assert_array_equal(getattr(state, name).numpy(), value)

    def test_reset_ownership_and_validation(self):
        """Preserve externally integrated bodies and validate requested arrays before any reset mutation."""
        source = _cloth()
        body = source.add_body(xform=wp.transform((1.0, 0.0, 0.5)))
        source.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        for device in self.devices:
            with self.subTest(device=device):
                model = _model([source, source], device)
                solver = SolverFBA(model)
                state = model.state()
                with self.assertRaises(ValueError):
                    solver.reset(None)
                state.body_q.assign(model.body_q.numpy() + 1.0)
                before = state.body_q.numpy().copy()
                valid_qd = state.particle_qd
                state.particle_qd = wp.zeros(model.particle_count - 1, dtype=wp.vec3, device=device)
                with self.assertRaises(ValueError):
                    solver.reset(state)
                np.testing.assert_array_equal(state.body_q.numpy(), before)
                # Unrequested and absent arrays do not participate in reset validation.
                solver.reset(state, flags=newton.StateFlags.PARTICLE_Q)
                state.particle_qd = valid_qd
                state.particle_q = None
                solver.reset(state, flags=newton.StateFlags.PARTICLE_Q)
                external = SolverFBA(model, integrate_with_external_rigid_solver=True)
                state = model.state()
                state.body_q.assign(before)
                state.particle_q.assign(model.particle_q.numpy() + 1.0)
                # Even a malformed body field belongs to the external integrator.
                state.body_qd = wp.zeros(0, dtype=wp.spatial_vector, device=device)
                external.reset(state)
                np.testing.assert_array_equal(state.body_q.numpy(), before)
                np.testing.assert_array_equal(state.particle_q.numpy(), model.particle_q.numpy())

    def test_reset_global_state(self):
        """Restore global entities only through the last mask entry, or as part of a full reset."""
        builder = newton.ModelBuilder()
        builder.add_builder(_cloth())
        builder.add_world(_cloth())
        for device in self.devices:
            with self.subTest(device=device):
                model = builder.finalize(device=device)
                solver = SolverFBA(model)
                state = model.state()
                initial = model.particle_q.numpy()
                dirty = initial + 1.0
                state.particle_q.assign(dirty)
                solver.reset(state, wp.array([False, True], dtype=wp.bool, device=device))
                expected = dirty.copy()
                selected = model.particle_world.numpy() == -1
                expected[selected] = initial[selected]
                np.testing.assert_array_equal(state.particle_q.numpy(), expected)
                solver.reset(state)
                np.testing.assert_array_equal(state.particle_q.numpy(), initial)

    def test_cross_world_constraints_are_rejected(self):
        """Reject particle constraint rows that join particles of different worlds."""
        model = _model([_cloth(), _cloth()], "cpu")
        solver = SolverFBA(model)
        with self.assertRaisesRegex(ValueError, "different worlds"):
            solver.set_particle_constraints([[0, 4]], [[1.0, -1.0]], [0.1])

    def test_bilateral_constraints_match_separate_worlds(self):
        """Step bilateral particle constraints in two worlds as each world steps alone."""
        for device in self.devices:
            for options in ({}, {"nsn_sparse_storage": True}, {"nsn_schur_mode": "lite"}):
                with self.subTest(device=device, options=options):
                    sources = [_cloth(1.0), _cloth(1.0, 0.03)]
                    results = []
                    for builders in (sources, *([s] for s in sources)):
                        model = _model(builders, device)
                        solver = SolverFBA(model, iterations=2, nsn_linear_iterations=1, **options)
                        count = len(builders)
                        solver.set_particle_constraints(
                            np.arange(0, 4 * count, 4)[:, None],
                            np.ones((count, 1)),
                            np.full(count, 0.05),
                            directions=np.tile([1.0, 0.0, 0.0], (count, 1)),
                            exact=True,
                        )
                        a, b = model.state(), model.state()
                        for _ in range(2):
                            solver.step(a, b, None, None, 0.01)
                            a, b = b, a
                        results.append(a.particle_q.numpy())
                    np.testing.assert_allclose(results[0], np.concatenate(results[1:]), atol=2e-6, rtol=0.0)

    def test_exact_rows_per_world_blocks(self):
        """Lite exact rows take one Cholesky block per world, of at most 64 rows each: 126 rows interleaved over two
        worlds, point rows and partial rows along one direction, and a third world without rows step as the worlds
        do alone, also after the targets and then the rows change; 65 rows in one world are refused."""

        def grid():
            builder = newton.ModelBuilder()
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, 0.0, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=4,
                dim_y=4,
                cell_x=0.05,
                cell_y=0.05,
                mass=0.01,
                tri_ke=1000.0,
                edge_ke=0.01,
                particle_radius=0.01,
            )
            return builder

        def rows(slot, world, shift, partial):
            """The rows of a world of 25 particles: 21 held at points, or 20 at points and two along z."""
            count = 20 if partial else 21
            p = np.repeat(np.arange(count), 3)
            d = np.tile(np.eye(3), (count, 1))
            if partial:
                p, d = np.r_[p, 21, 22], np.r_[d, [[0.0, 0.0, 1.0]] * 2]
            q = grid().particle_q
            target = (np.asarray(q)[p] * d).sum(axis=1) + shift * d[:, 2] + 0.002 * world * d[:, 0]
            return p + 25 * slot, d, target

        for device in self.devices:
            with self.subTest(device=device):
                results = []
                for worlds in ((0, 1, 2), (0,), (1,), (2,)):
                    model = _model([grid() for _ in worlds], device)
                    solver = SolverFBA(model, iterations=2, nsn_linear_iterations=1, nsn_schur_mode="lite")
                    a, b = model.state(), model.state()
                    for step in range(4):
                        partial = step >= 2  # the rows change after two steps
                        parts = [rows(k, w, 0.01 * step, partial) for k, w in enumerate(worlds) if w != 2]
                        if parts:
                            p, d, target = (np.concatenate(x) for x in zip(*parts, strict=True))
                            # interleave the worlds' rows
                            order = np.argsort(np.tile(np.arange(len(p) // len(parts)), len(parts)), kind="stable")
                            p, d, target = p[order], d[order], target[order]
                            solver.set_particle_constraints(
                                p[:, None], np.ones((len(p), 1)), target, directions=d, exact=True
                            )
                            if len(parts) == 2:
                                self.assertGreater(len(p), 64)
                        solver.step(a, b, None, None, 0.01)
                        a, b = b, a
                    results.append(a.particle_q.numpy())
                np.testing.assert_allclose(results[0], np.concatenate(results[1:]), atol=2e-6, rtol=0.0)
                # the held particles reach their targets and the world without rows falls freely
                q = results[0].reshape(3, 25, 3)
                self.assertTrue(np.all(q[2, :, 2] < 1.0))

                model = _model([grid(), grid()], device)
                solver = SolverFBA(model, nsn_schur_mode="lite")
                p = np.r_[np.repeat(np.arange(21), 3), 21, 22]
                with self.assertRaisesRegex(NotImplementedError, "per world"):
                    solver.set_particle_constraints(
                        p[:, None],
                        np.ones((65, 1)),
                        np.zeros(65),
                        directions=np.r_[np.tile(np.eye(3), (21, 1)), np.eye(3)[:2]],
                        exact=True,
                    )

    def test_exact_rows_packed_response_cache(self):
        """The cache of exact rows packs each world's columns of ``A^-1`` at its held particles over its own
        particles, 8 bytes per entry and no more than the columns over all particles in one world, and holds the
        complete ``J A^-1 J^T`` and the correction ``A^-1 J^T x`` of the dense formulas in the rows' order; it stays
        while only the targets change and rebuilds when the rows do."""
        from newton._src.solvers.fba import kernels as K  # noqa: PLC0415

        def grid():
            builder = newton.ModelBuilder()
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, 0.0, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=3,
                dim_y=3,
                cell_x=0.05,
                cell_y=0.05,
                mass=0.01,
                tri_ke=1000.0,
                edge_ke=0.01,
                particle_radius=0.01,
            )
            return builder

        def nbytes(a):
            return a.size * wp.types.type_size_in_bytes(a.dtype)

        def check(solver, p, d, c):
            model = solver.model
            n, world = model.particle_count, model.particle_world.numpy()
            a, b = model.state(), model.state()
            target = (a.particle_q.numpy()[p] * d).sum(axis=1)
            solver.set_particle_constraints(p[:, None], c[:, None], target, directions=d, exact=True)
            solver.step(a, b, None, None, 0.01)
            Ainv = np.empty((n, n))  # A^-1 of the solver's factor, column by column
            e, column = wp.zeros(n, dtype=wp.float64, device="cpu"), wp.empty(n, dtype=wp.float64, device="cpu")
            for i in range(n):
                e.zero_()
                e[i : i + 1].fill_(1.0)
                solver._linear_solver.solve_scalar(e, column)
                Ainv[:, i] = column.numpy()
            self.assertFalse(np.any(Ainv[world[:, None] != world[None, :]]))
            h = solver._held_rows[1]
            # the rows ordered by world, each world's in their own order
            rows = h.rows.numpy()
            np.testing.assert_array_equal(rows, np.argsort(world[p] if model.world_count > 1 else 0 * p, kind="stable"))
            slot, tile = h.slot.numpy(), h.tile
            hp, hc, hd = p[rows], c[rows], d[rows]
            # packed columns: sum over the worlds of U_w N_w entries of 8 bytes, U_w held particles of N_w
            held = [np.unique(hp[world[hp] == w]) for w in np.unique(world[hp])]
            packed = sum(len(u) * np.count_nonzero(world == world[u[0]]) for u in held)
            if model.world_count == 1:
                packed = sum(len(u) for u in held) * n
            allowance = 4 * (2 * n + 2 * len(held) + 1) + len(rows) * (5 * 4 + 3 * 8 + 24)
            # every array the cache allocates, whatever its layout, but the blocks' factors and multipliers
            unchanged = {"L", "rhs", "x"}
            total = sum(nbytes(v) for k, v in vars(h).items() if isinstance(v, wp.array) and k not in unchanged)
            self.assertLessEqual(total, 8 * packed + allowance)
            self.assertEqual(nbytes(h.response), 8 * packed)
            self.assertLessEqual(nbytes(h.response), 8 * len(np.unique(hp)) * n)  # the dense columns before
            self.assertLessEqual(total - nbytes(h.response), allowance)
            # the complete A^-1 J^T, unpacked
            block, local = h.particle_block.numpy(), h.particle_local.numpy()
            start, column_start = h.row_start.numpy(), h.column_start.numpy()
            stride, column, response = h.block_particles.numpy(), h.column.numpy(), h.response.numpy()
            dense = np.zeros((n, len(rows)))
            for i in np.flatnonzero(block >= 0):
                k = np.arange(start[block[i]], start[block[i] + 1])
                dense[i, k] = response[column_start[block[i]] + local[i] + column[k] * stride[block[i]]]
            np.testing.assert_array_equal(dense, Ainv[:, hp])
            # the complete J A^-1 J^T, bitwise as each world's block of the dense formula
            # the cache holds column q of A^-1 at particle p as entry (q, p), as the old formula of its columns at
            # their particles did; the computed inverse need not be bitwise symmetric, so transpose Ainv[target, source]
            W = Ainv[np.ix_(hp, hp)].T * (hd @ hd.T) * np.outer(hc, hc)
            L = h.L.numpy()
            blocks = slot // tile
            self.assertFalse(np.any(W[blocks[:, None] != blocks[None, :]]))
            for b in range(L.shape[0]):
                k = np.flatnonzero(blocks == b)
                np.testing.assert_array_equal(k, np.arange(k[0], k[0] + len(k)))
                np.testing.assert_array_equal(slot[k] % tile, np.arange(len(k)))
                m = len(k)
                np.testing.assert_array_equal(L[b, :m, :m], np.linalg.cholesky(W[np.ix_(k, k)]))
                np.testing.assert_allclose(L[b, :m, :m] @ L[b, :m, :m].T, W[np.ix_(k, k)], rtol=1e-12, atol=0)
                np.testing.assert_array_equal(L[b, m:, m:], np.eye(tile - m))
            self.assertEqual(L.shape[0], len(held))
            # q += A^-1 J^T x, summed in the rows' order
            x = np.random.default_rng(1).standard_normal(L.shape[0] * tile)
            q = wp.array(np.random.default_rng(2).standard_normal((n, 3)), dtype=wp.vec3, device="cpu")
            q0 = q.numpy().copy()
            inputs = [h.particle_block, h.particle_local, h.row_start, h.column_start, h.block_particles, h.column]
            inputs += [h.response, h.coefficient, h.direction, h.slot, wp.array(x, dtype=wp.float64, device="cpu")]
            wp.launch(K.held_row_correction_kernel, n, inputs, [q], device="cpu")
            dq = np.zeros((n, 3))
            for k in range(len(rows)):
                dq += Ainv[:, hp[k], None] * hc[k] * x[slot[k]] * hd[k]
            np.testing.assert_allclose(q.numpy(), (q0 + dq.astype(np.float32)), rtol=1e-6, atol=1e-6)
            untouched = ~np.isin(world, world[hp])  # a world without rows takes no correction
            np.testing.assert_array_equal(q.numpy()[untouched], q0[untouched])
            return h

        point = np.eye(3)
        # three co-located worlds of 16 particles: point pins of world 0, rows along one direction of world 1
        # interleaved with them, and world 2 without rows
        solver = SolverFBA(_model([grid() for _ in range(3)], "cpu"), iterations=1, nsn_schur_mode="lite")
        p = np.r_[np.repeat([0, 5], 3), 16 + 3, 16 + 9, 6]
        d = np.r_[np.tile(point, (2, 1)), [[0.0, 0.0, 1.0]] * 2, [[1.0, 0.0, 0.0]]]
        c = np.r_[np.ones(6), 2.0, 0.5, 1.0]
        h = check(solver, p, d, c)
        self.assertIs(check(solver, p, d, c), h)  # the targets alone keep the cache
        h2 = check(solver, p, d, 2.0 * c)  # new coefficients rebuild it
        self.assertIsNot(h2, h)
        h3 = check(solver, np.r_[p, 32 + 4], np.r_[d, [[0.0, 1.0, 0.0]]], np.r_[c, 1.0])  # world 2 adds its block
        self.assertEqual(h3.L.shape[0], 3)
        # four worlds, each pinning three particles at points: 12 columns of 16 entries instead of 64
        solver = SolverFBA(_model([grid() for _ in range(4)], "cpu"), iterations=1, nsn_schur_mode="lite")
        p = np.repeat(np.r_[0, 7, 15][None, :] + 16 * np.arange(4)[:, None], 3)
        h = check(solver, p, np.tile(point, (12, 1)), np.ones(36))
        self.assertEqual(h.response.size, 4 * 3 * 16)
        # one world pinning three particles at points keeps the three dense columns
        solver = SolverFBA(_model([grid()], "cpu"), iterations=1, nsn_schur_mode="lite")
        h = check(solver, np.repeat([0, 7, 15], 3), np.tile(point, (3, 1)), np.ones(9))
        self.assertEqual(h.response.size, 3 * 16)

    def test_exact_rows_one_world_mixed_global_particles(self):
        """In a model of one world, global particles joined to the world's own ones take the exact rows in one
        block, coupled through ``A^-1``, within one budget of 64 rows."""
        builder = newton.ModelBuilder()
        builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0), 0.01)
        builder.add_particle(wp.vec3(0.1, 0.0, 1.0), wp.vec3(0.0), 0.01)
        builder.begin_world()
        builder.add_particle(wp.vec3(0.0, 0.1, 1.0), wp.vec3(0.0), 0.01)
        for k in range(11):  # free particles of the world, 3 to 13, and global ones, 14 to 24
            builder.add_particle(wp.vec3(0.1 * k, 1.0, 1.0), wp.vec3(0.0), 0.01)
        builder.end_world()
        for k in range(11):
            builder.add_particle(wp.vec3(0.1 * k, 2.0, 1.0), wp.vec3(0.0), 0.01)
        builder.add_triangle(0, 1, 2, tri_ke=1000.0, tri_ka=0.0, tri_kd=0.0)
        for device in self.devices:
            with self.subTest(device=device):
                model = builder.finalize(device=device)
                self.assertEqual(model.world_count, 1)
                solver = SolverFBA(model, nsn_schur_mode="lite", iterations=1)
                directions = np.array([[0.0, 0.0, 1.0]] * 2)
                solver.set_particle_constraints(
                    [[0], [2]], [[1.0], [1.0]], [1.02, 0.98], directions=directions, exact=True
                )
                a, b = model.state(), model.state()
                solver.step(a, b, None, None, 0.01)
                np.testing.assert_allclose(b.particle_q.numpy()[[0, 2], 2], [1.02, 0.98], atol=2e-6, rtol=0.0)
                # 33 point rows of the world's particles and 33 of global ones
                p = np.repeat(np.arange(3, 25), 3)
                with self.assertRaisesRegex(NotImplementedError, "64 rows"):
                    solver.set_particle_constraints(
                        p[:, None], np.ones((66, 1)), np.ones(66), directions=np.tile(np.eye(3), (22, 1)), exact=True
                    )

    def test_failed_constraint_update_keeps_ownership(self):
        """Keep the previous constraint rows and their worlds after a refused update."""
        model = _model([_cloth(), _cloth()], "cpu")
        solver = SolverFBA(model, iterations=1)
        solver.set_particle_constraints([[0]], [[1.0]], [0.005], directions=[[0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            solver.set_particle_constraints([[4], [5]], [["bad"], ["bad"]], [0.005, 0.005])
        solver.step(model.state(), model.state(), None, None, 0.01)
        np.testing.assert_array_equal(solver._constraint_world_h, [0])

    def test_global_contacts_and_shape_ownership(self):
        """Let global particles contact and constrain a world's, and reject a shape outside its body's world."""
        builder = newton.ModelBuilder()
        builder.add_builder(_cloth(0.0))
        builder.add_world(_cloth(0.005))
        model = builder.finalize(device="cpu")
        self.assertEqual(model.world_count, 1)
        for mode in ("vertex_vertex", "vertex_face"):
            self.assertGreater(SelfContact(model, mode, 0.02).find(model.particle_q), 0)
        solver = SolverFBA(model)
        solver.set_particle_constraints([[0, 4]], [[1.0, -1.0]], [0.005])

        builder = newton.ModelBuilder()
        builder.begin_world()
        body = builder.add_body(mass=1.0)
        builder.end_world()
        builder.add_world(_cloth())
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        with self.assertRaisesRegex(ValueError, "body's world"):
            SolverFBA(builder.finalize(device="cpu"))

    def test_lite_dispatch_is_independent(self):
        """Choose each world's lite contact solve as it would alone, beside a world of another choice."""
        for boxes, mass in ((5, 0.01), (2, 1.0e10), (5, 1.0e10)):
            complex_world = _cloth()
            for _ in range(boxes):
                complex_world.add_shape_box(-1, hx=0.2, hy=0.2, hz=0.002)
            simple_world = _cloth(mass=mass)
            if mass == 0.01:
                simple_world.add_shape_plane(plane=(0.6, 0.0, 0.8, 0.0))
            sources = [simple_world, complex_world]
            for device in self.devices:
                for accumulate in (False, True):
                    with self.subTest(device=device, accumulate=accumulate, boxes=boxes, mass=mass):
                        results = []
                        for builders in (sources, *([s] for s in sources)):
                            model = _model(builders, device)
                            solver = SolverFBA(
                                model,
                                iterations=2,
                                nsn_linear_iterations=1,
                                nsn_schur_mode="lite",
                                nsn_lite_accumulate=accumulate,
                            )
                            pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.03)
                            contacts = pipeline.contacts()
                            a, b = model.state(), model.state()
                            for _ in range(2):
                                pipeline.collide(a, contacts)
                                solver.step(a, b, None, contacts, 0.01)
                                a, b = b, a
                            results.append(a.particle_q.numpy())
                        np.testing.assert_allclose(results[0], np.concatenate(results[1:]), atol=2e-6, rtol=0.0)

    def test_lite_self_contact_beside_plain_world(self):
        """Self-contacts between two cloth layers build the lite rows on the device; the layered world and a world of
        plain ground contacts beside it each match their separate solves."""
        layered = _cloth(0.01)
        layered.add_builder(_cloth(0.02))
        sources = [_cloth(), layered]
        for device in self.devices:
            with self.subTest(device=device):
                results = []
                for builders in (sources, *([s] for s in sources)):
                    model = _model(builders, device)
                    solver = SolverFBA(
                        model,
                        iterations=2,
                        nsn_linear_iterations=2,
                        nsn_schur_mode="lite",
                        particle_enable_self_contact=True,
                        particle_self_contact_margin=0.02,
                    )
                    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.03)
                    contacts = pipeline.contacts()
                    a, b = model.state(), model.state()
                    for _ in range(2):
                        pipeline.collide(a, contacts)
                        solver.step(a, b, None, contacts, 0.01)
                        a, b = b, a
                    results.append(a.particle_q.numpy())
                self.assertIsNotNone(solver._contact_extra)
                np.testing.assert_allclose(results[0], np.concatenate(results[1:]), atol=2e-6, rtol=0.0)

    def test_mixed_rigid_and_deformable_worlds(self):
        """A two-way contact in one environment must not change another environment's cloth solve."""
        cloth_box = _cloth(0.105)
        body = cloth_box.add_body(xform=wp.transform((0.05, 0.05, 0.05), wp.quat_identity()))
        cloth_box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05)
        rigid = newton.ModelBuilder()
        pose = wp.transform((0.05, 0.05, 0.05), wp.quat_identity())
        body = rigid.add_link(xform=pose)
        rigid.add_articulation([rigid.add_joint_prismatic(-1, body, axis=(0.0, 0.0, 1.0), parent_xform=pose)])
        rigid.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05)
        sources = [cloth_box, _cloth(), rigid]
        for device in self.devices:
            for sparse in (False, True):
                with self.subTest(device=device, sparse=sparse):
                    results = []
                    for builders in (sources, *([s] for s in sources)):
                        model = _model(builders, device)
                        solver = SolverFBA(
                            model,
                            iterations=2,
                            nsn_linear_iterations=2,
                            nsn_sparse_storage=sparse,
                            contact_warmstart=True,
                            friction_anchors=True,
                        )
                        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.03)
                        contacts = pipeline.contacts()
                        a, b = model.state(), model.state()
                        for _ in range(3):
                            pipeline.collide(a, contacts)
                            solver.step(a, b, None, contacts, 0.01)
                            if model.world_count > 1 and solver._anchor_count:
                                keys = solver._anchor_previous_key_d.numpy()[: solver._anchor_count]
                                self.assertTrue(np.all(keys[1:] >= keys[:-1]))
                            a, b = b, a
                        results.append(
                            tuple(
                                value.numpy() if value is not None else np.empty((0, width))
                                for value, width in ((a.particle_q, 3), (a.body_q, 7))
                            )
                        )
                    for field in range(2):
                        np.testing.assert_allclose(
                            results[0][field], np.concatenate([r[field] for r in results[1:]]), atol=2e-6, rtol=0.0
                        )


if __name__ == "__main__":
    unittest.main()
