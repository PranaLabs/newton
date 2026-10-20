# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import SolverFBA
from newton._src.solvers.fba.linear_solver import (
    build_particle_element_csr,
    build_pd_system,
)


def _projection_rhs(kernel, inputs, arity):
    """Run a local projection kernel over its elements and sum each element's contributions per particle, as the
    solver's gather does: the right-hand side the projection adds. ``inputs[0]`` holds the positions and
    ``inputs[1]`` the element vertex indices, ``arity`` per element."""
    positions, elements = inputs[0], inputs[1].numpy().reshape(-1, arity)
    contributions = wp.zeros(elements.shape, dtype=wp.vec3, device=positions.device)
    wp.launch(kernel, dim=elements.shape[0], inputs=inputs, outputs=[contributions], device=positions.device)
    rhs = np.zeros((positions.shape[0], 3), dtype=np.float32)
    np.add.at(rhs, elements, contributions.numpy())
    return rhs


class TestSolverFBAImport(unittest.TestCase):
    def test_import_solver_fba(self):
        """Import SolverFBA from the FBA package."""
        assert SolverFBA is not None


class TestBuildPDSystem(unittest.TestCase):
    """PD Hessian assembly on a 2-triangle mesh: sanity checks of its entries."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_two_triangle_cloth(self):
        # Two triangles sharing edge (1, 2):
        #     0 --- 1
        #     |   / |
        #     | /   |
        #     2 --- 3
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 1.0),
                wp.vec3(1.0, 0.0, 1.0),
            ],
            indices=[0, 1, 2, 1, 3, 2],
            density=1.0,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0,
            edge_kd=0.0,
        )
        return builder.finalize(device="cpu")

    def test_diagonal_includes_mass_over_dt_squared(self):
        """Assemble a symmetric PD matrix whose diagonal holds at least ``m / dt²`` for every free particle."""
        model = self._build_two_triangle_cloth()
        dt = 1.0 / 60.0
        A = build_pd_system(model, dt=dt, pin_stiffness=1e12)[0].toarray()

        self.assertEqual(A.shape, (model.particle_count, model.particle_count))
        # A is symmetric.
        diff = A - A.T
        self.assertLess(np.abs(diff).max(), 1e-10)

        # Diagonal of mass contribution: m_i/dt^2 for free particles.
        masses = model.particle_mass.numpy()
        inv_masses = model.particle_inv_mass.numpy()
        diag = A.diagonal()
        for i in range(model.particle_count):
            if inv_masses[i] > 0.0:
                # Contains at least m/dt^2 (other terms add on top).
                self.assertGreaterEqual(diag[i], masses[i] / (dt * dt) - 1e-9)

    def test_pin_stiffness_on_pinned_diagonal(self):
        """Add the pin stiffness to the diagonal of a particle pinned by a zero mass."""
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        # Manually pin particle 0 by setting mass to 0.
        builder.particle_mass[0] = 0.0
        model = builder.finalize(device="cpu")

        A = build_pd_system(model, dt=1.0 / 60.0, pin_stiffness=1e12)[0].toarray()
        # Pin contributes pin_stiffness to A[0,0].
        self.assertGreaterEqual(A[0, 0], 1e12 - 1e-3)


class TestBendingQMatchesReference(unittest.TestCase):
    """Verify _compute_isometric_bending_q against a hand-computed reference
    on an asymmetric quad, whose four cotangent weights all differ.
    """

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_bending_q_matches_reference(self):
        """Match the cotangent weights and the ``3 / (A0 + A1)`` scale of an asymmetric quad's interior edge with a hand
        computation."""
        # Asymmetric quad: triangles (0,1,2) and (1,3,2) sharing edge (1,2).
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),  # v0
                wp.vec3(2.0, 0.0, 0.0),  # v1
                wp.vec3(0.1, 0.0, 1.0),  # v2 (skewed)
                wp.vec3(1.9, 0.0, -1.0),  # v3 (different skew)
            ],
            indices=[0, 1, 2, 1, 3, 2],
            density=1.0,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0,
            edge_kd=0.0,
        )
        model = builder.finalize(device="cpu")

        # Inspect Newton's edge layout to find the interior edge and its stencil
        # permutation.  Newton stores [o0, o1, v1, v2]; reordering to
        # [v1, v2, o0, o1] gives the (v0, v1, v2, v3) stencil used internally.
        ei = model.edge_indices.numpy().reshape(-1, 4)
        interior_mask = (ei[:, 0] >= 0) & (ei[:, 1] >= 0)
        ei_interior = ei[interior_mask]
        # There must be exactly one interior edge for this two-triangle mesh.
        self.assertEqual(ei_interior.shape[0], 1)
        # After reorder: stencil = [v1=1, v2=2, o0=0, o1=3] -> (v0=1,v1=2,v2=0,v3=3)
        stencil = ei_interior[0, [2, 3, 0, 1]]  # [v1, v2, o0, o1]
        self.assertListEqual(list(stencil), [1, 2, 0, 3])

        _A, meta = build_pd_system(model, dt=1.0 / 60.0, pin_stiffness=1e12)

        # Hand-computed reference for stencil (v0=1,v1=2,v2=0,v3=3):
        #   x0=(2,0,0), x1=(0.1,0,1), x2=(0,0,0), x3=(1.9,0,-1)
        x0 = np.array([2.0, 0.0, 0.0])
        x1 = np.array([0.1, 0.0, 1.0])
        x2 = np.array([0.0, 0.0, 0.0])
        x3 = np.array([1.9, 0.0, -1.0])
        l01 = np.linalg.norm(x1 - x0)
        l02 = np.linalg.norm(x2 - x0)
        l12 = np.linalg.norm(x2 - x1)
        l03 = np.linalg.norm(x3 - x0)
        l13 = np.linalg.norm(x3 - x1)
        r0 = 0.5 * (l01 + l02 + l12)
        A0 = np.sqrt(max(r0 * (r0 - l01) * (r0 - l02) * (r0 - l12), 0.0))
        r1 = 0.5 * (l01 + l03 + l13)
        A1 = np.sqrt(max(r1 * (r1 - l01) * (r1 - l03) * (r1 - l13), 0.0))
        cot02 = (l01 * l01 - l02 * l02 + l12 * l12) / (4.0 * max(A0, 1e-20))
        cot12 = (l01 * l01 + l02 * l02 - l12 * l12) / (4.0 * max(A0, 1e-20))
        cot03 = (l01 * l01 - l03 * l03 + l13 * l13) / (4.0 * max(A1, 1e-20))
        cot13 = (l01 * l01 + l03 * l03 - l13 * l13) / (4.0 * max(A1, 1e-20))
        q_ref = np.array([cot02 + cot03, cot12 + cot13, -(cot02 + cot12), -(cot03 + cot13)])
        scale_ref = 3.0 / max(A0 + A1, 1e-20)

        self.assertTrue(
            np.allclose(meta["edge_quad_q"][0], q_ref, atol=1e-10),
            f"q mismatch: got {meta['edge_quad_q'][0]}, expected {q_ref}",
        )
        self.assertTrue(
            np.allclose(meta["edge_quad_scale"][0], scale_ref, atol=1e-10),
            f"scale mismatch: got {meta['edge_quad_scale'][0]}, expected {scale_ref}",
        )

    def test_bending_matches_style3d(self):
        """Assemble the bending matrix SolverStyle3D assembles: each interior edge adds
        ``edge_ke * 3 / (A0 + A1) * w w^T`` with the cotangent weights ``w`` that Style3D stores for the edge."""
        builder = newton.ModelBuilder()
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        # Cells of 10 x 20 cm: legs and diagonals of three lengths, and areas at which Style3D's 1e-8
        # regularization of areas and cotangents stays below 1e-6.
        newton.solvers.style3d.add_cloth_grid(
            builder,
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=5,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.2,
            mass=0.01,
        )
        model = builder.finalize(device="cpu")
        n = model.particle_count
        bending = model.edge_bending_properties.numpy().copy()  # a CPU array's numpy() is a view
        self.assertGreater(bending[:, 0].min(), 0.0)

        # Style3D's Hessian (style3d/builder.py add_bend_constraints); it skips boundary edges.
        edges = model.edge_indices.numpy()
        area, cot = model.style3d.edge_rest_area.numpy(), model.style3d.edge_bending_cot.numpy()
        w = np.stack([-cot[:, 0] - cot[:, 2], -cot[:, 1] - cot[:, 3], cot[:, 2] + cot[:, 3], cot[:, 0] + cot[:, 1]], 1)
        expected = np.zeros((n, n))
        for e in np.flatnonzero((edges[:, 0] >= 0) & (edges[:, 1] >= 0)):
            expected[np.ix_(edges[e], edges[e])] += bending[e, 0] * 3.0 / area[e] * np.outer(w[e], w[e])

        # FBA's bending part: its PD matrix with the model's edge_ke minus the one without bending.
        with_bending = build_pd_system(model, dt=1.0, pin_stiffness=0.0)[0].toarray()
        model.edge_bending_properties.assign(bending * np.array([0.0, 1.0], dtype=np.float32))
        without_bending = build_pd_system(model, dt=1.0, pin_stiffness=0.0)[0].toarray()
        np.testing.assert_allclose(
            with_bending - without_bending, expected, rtol=0.0, atol=1e-5 * np.abs(expected).max()
        )


class TestSparseInverse(unittest.TestCase):
    """compute_lower_inverse matches dense np.linalg.inv on small SPD matrices."""

    def test_random_spd_8x8(self):
        """Reproduce the permuted inverse of a tridiagonal SPD matrix as ``Sᵀ D⁻¹ S``."""
        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            _elimination_tree,
            compute_lower_inverse,
        )
        from newton._src.solvers.fba.sparse_ldlt import CsrMatrix, factor_ldlt  # noqa: PLC0415

        rng = np.random.default_rng(0)
        n = 8
        off = rng.uniform(-1.0, 1.0, n - 1)
        A = np.diag(rng.uniform(10.0, 20.0, n)) + np.diag(off, 1) + np.diag(off, -1)
        # Points along a line make the nested dissection reorder the unknowns.
        points = np.stack([np.arange(n), np.zeros(n), np.zeros(n)], axis=1).astype(np.float64)
        order, L, d = factor_ldlt(CsrMatrix.from_matrix(A), points, leaf_size=2)
        S = compute_lower_inverse(L, parent=_elimination_tree(L)).toarray()
        Ainv = np.linalg.inv(A)[np.ix_(order, order)]
        self.assertLess(np.abs(S.T @ np.diag(1.0 / d) @ S - Ainv).max(), 1e-12)


class TestFBALinearSolver(unittest.TestCase):
    """End-to-end linear solve and permutation round trip."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_solve_random_spd(self):
        """Solve a random tridiagonal SPD system for all three components to a relative residual of 1e-6."""
        import scipy.sparse as sp

        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            factorize_and_sparse_inverse,
        )

        rng = np.random.default_rng(42)
        n = 64
        diag = rng.uniform(10.0, 20.0, n)
        off = rng.uniform(-0.5, 0.5, n - 1)
        A = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr").astype(np.float64)

        fs = factorize_and_sparse_inverse(A)
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        solver = FBALinearSolver(fs, device=device)

        b_np = rng.standard_normal((n, 3))
        b = wp.array(b_np, dtype=wp.vec3, device=device)
        x = wp.empty(n, dtype=wp.vec3, device=device)

        solver.solve(b, x)
        x_np = x.numpy()

        # Verify A · x ≈ b for each component.
        for c in range(3):
            r = A.dot(x_np[:, c]) - b_np[:, c]
            self.assertLess(np.linalg.norm(r) / max(np.linalg.norm(b_np[:, c]), 1e-12), 1e-6)

    def test_solve_spd_with_dominant_off_diagonal(self):
        """Solve an SPD system whose off-diagonal entries exceed the diagonal.

        Off-diagonal pivoting would break the symmetric ``A⁻¹ = Pᵀ Sᵀ D⁻¹ S P`` form; bending stencils and rigid inertia
        produce such matrices.
        """
        import scipy.sparse as sp

        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            factorize_and_sparse_inverse,
        )

        A_dense = np.array([[1.0, 2.0, 0.0], [2.0, 5.0, 0.0], [0.0, 0.0, 3.0]])
        self.assertGreater(np.linalg.eigvalsh(A_dense).min(), 0.0)

        fs = factorize_and_sparse_inverse(sp.csr_matrix(A_dense))
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        solver = FBALinearSolver(fs, device=device)

        b_np = np.eye(3)
        b = wp.array(b_np, dtype=wp.vec3, device=device)
        x = wp.empty(3, dtype=wp.vec3, device=device)
        solver.solve(b, x)

        np.testing.assert_allclose(x.numpy(), np.linalg.solve(A_dense, b_np), rtol=1e-6, atol=1e-6)

    def test_permutation_roundtrip(self):
        """Permute a vector and permute it back with the inverse, as the linear solve does around the factor."""
        from newton._src.solvers.fba.kernels import apply_permutation_scalar_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        n = 32
        perm = np.random.default_rng(1).permutation(n).astype(np.int32)
        invperm = np.argsort(perm).astype(np.int32)

        x_np = np.random.default_rng(2).standard_normal(n)
        x = wp.array(x_np, dtype=wp.float64, device=device)
        y = wp.empty(n, dtype=wp.float64, device=device)
        z = wp.empty(n, dtype=wp.float64, device=device)
        perm_wp = wp.array(perm, dtype=wp.int32, device=device)
        invperm_wp = wp.array(invperm, dtype=wp.int32, device=device)

        wp.launch(apply_permutation_scalar_kernel, dim=n, inputs=[x, perm_wp], outputs=[y], device=device)
        wp.launch(apply_permutation_scalar_kernel, dim=n, inputs=[y, invperm_wp], outputs=[z], device=device)

        np.testing.assert_array_equal(y.numpy(), x_np[perm])
        np.testing.assert_array_equal(z.numpy(), x_np)


class TestARAPProjection(unittest.TestCase):
    """ARAP local projection on triangle stretching."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_projection_of_crushed_triangles(self):
        """Return the nearest 3x2 matrix with orthonormal columns from svd_3x2, also for crushed triangles.

        With one singular value at 1e-6 of the other, the former division by it turned float32 rounding of F^T F into
        projections of order 1e3 and blew up a cloth piled on a table.
        """
        from newton._src.solvers.fba.kernels import mat32, svd_3x2  # noqa: PLC0415

        @wp.kernel
        def project(F: wp.array[mat32], P: wp.array[mat32]):
            i = wp.tid()
            P[i] = svd_3x2(F[i])

        rng = np.random.default_rng(3)
        cases = [rng.standard_normal((3, 2)) for _ in range(8)]
        for s1 in (1.0e-3, 1.0e-6, 0.0):
            for _ in range(8):
                U = np.linalg.qr(rng.standard_normal((3, 2)))[0]
                V = np.linalg.qr(rng.standard_normal((2, 2)))[0]
                cases.append(U @ np.diag([rng.uniform(0.5, 2.0), s1]) @ V.T)
        cases.append(np.zeros((3, 2)))
        F = np.array(cases)
        out = wp.zeros(len(F), dtype=mat32)
        wp.launch(project, dim=len(F), inputs=[wp.array(F.astype(np.float32), dtype=mat32)], outputs=[out])
        P = out.numpy().astype(np.float64)
        np.testing.assert_allclose(
            np.einsum("nki,nkj->nij", P, P), np.broadcast_to(np.eye(2), (len(F), 2, 2)), atol=1e-5
        )
        U, sigma, Vt = np.linalg.svd(F, full_matrices=False)
        nearest = np.sum((sigma - 1.0) ** 2, axis=1)
        np.testing.assert_allclose(np.sum((F - P) ** 2, axis=(1, 2)), nearest, atol=1e-5)
        np.testing.assert_allclose(P[:8], (U @ Vt)[:8], atol=1e-5)

    def test_identity_F_zero_after_assembly_bias(self):
        """Contribute the analytic ``w·Dm_inv·P^T`` at rest, with ``Dm_inv = I₂``.

        For rest=current=(0,0,0),(1,0,0),(0,0,1), P (3x2) maps reference axes to
        the embedded triangle's axes, so the rows of ``w·Dm_inv·P^T`` are
            row0 = (1,0,0); row1 = (0,0,1)  (since P = embedding)
        and the particles receive:
            rhs[0] = -(row0+row1) = (-1, 0, -1)
            rhs[1] = (1, 0, 0)
            rhs[2] = (0, 0, 1)
        """
        from newton._src.solvers.fba.kernels import project_stretching_arap_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            dtype=wp.vec3,
            device=device,
        )
        tri_indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat22(1.0, 0.0, 0.0, 1.0)], dtype=wp.mat22, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)

        rhs = _projection_rhs(project_stretching_arap_kernel, [positions, tri_indices, Dm_inv, weight], 3)
        r = rhs
        np.testing.assert_allclose(r[1], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(r[2], [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(r[0], -(r[1] + r[2]), atol=1e-6)

    def test_non_symmetric_Dm_inv_exposes_transpose_bug(self):
        """Apply a non-symmetric Dm_inv (a skewed triangle's rest inverse) as Dm_inv, not Dm_inv^T.

        The wrong orientation gave a ~20% error.
        """
        from newton._src.solvers.fba.kernels import project_stretching_arap_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # Current configuration = rest, so P = embedding and the contributions w·Dm_inv·P^T are exact.
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(2, 0, 0), wp.vec3(1, 0, 1)],
            dtype=wp.vec3,
            device=device,
        )
        tri_indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        # For this triangle: e12=(2,0,0), e13=(1,0,1); the orthonormal basis is
        # n1=(1,0,0), n2=(0,0,1); Dm = basis^T·edges = [[2,1],[0,1]];
        # Dm_inv = [[0.5,-0.5],[0,1]].
        Dm_inv = wp.array(
            [wp.mat22(0.5, -0.5, 0.0, 1.0)],
            dtype=wp.mat22,
            device=device,
        )
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = _projection_rhs(project_stretching_arap_kernel, [positions, tri_indices, Dm_inv, weight], 3)
        r = rhs

        # Analytical: with current=rest and Dm_inv as above, P = embedding [[1,0],[0,0],[0,1]] (3x2);
        # P^T = [[1,0,0],[0,0,1]] (2x3). Dm_inv @ P^T = [[0.5, 0, -0.5],[0, 0, 1]].
        # So row0 (vertex 1's contribution) = (0.5, 0, -0.5), row1 (vertex 2) = (0, 0, 1).
        # rhs[1] = row0, rhs[2] = row1, rhs[0] = -(row0+row1).
        np.testing.assert_allclose(r[1], [0.5, 0.0, -0.5], atol=1e-5)
        np.testing.assert_allclose(r[2], [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(r[0], -(r[1] + r[2]), atol=1e-6)


class TestBendingProjection(unittest.TestCase):
    """Isometric bending projection. A flat rest gives zero RHS contribution."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_flat_rest_zero_rhs(self):
        """Contribute nothing on a flat rest stencil, whose rest curvature is zero."""
        from newton._src.solvers.fba.kernels import project_bending_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # Flat 4-vertex stencil: 2 triangles sharing edge (0,1).
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0.5, 0, 1), wp.vec3(0.5, 0, -1)],
            dtype=wp.vec3,
            device=device,
        )
        edge_indices = wp.array([[0, 1, 2, 3]], dtype=wp.int32, device=device)
        # q = [1,1,-1,-1] satisfies sum-to-zero; on the flat positions above
        # ``q·x_rest = (0,0,0)`` so ``edge_norm = 0`` and the kernel short-circuits.
        edge_quad_q = wp.array([wp.vec4(1.0, 1.0, -1.0, -1.0)], dtype=wp.vec4, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        edge_norm = wp.array([0.0], dtype=wp.float32, device=device)

        rhs = _projection_rhs(project_bending_kernel, [positions, edge_indices, edge_quad_q, weight, edge_norm], 4)
        # Flat-rest ``edge_norm = 0`` ⇒ the contributions are zero.
        np.testing.assert_allclose(rhs, np.zeros((4, 3)), atol=1e-6)


class SolverFBABendingCurvedRestTests(unittest.TestCase):
    """The bending local step restores the curvature *magnitude*, not its
    direction, on curved rest cloth.

    Restoring the *rest curvature vector*, ``w · q · (q · x_ref)``, the
    gradient of ``‖(q·x_cur) - (q·x_ref)‖²``, would penalise any deviation
    from it, including curvature flips (e.g. cloth bent against its rest
    direction). The local projection instead contributes
    ``w · q · ê · ‖q·x_rest‖`` where ``ê = (q·x_cur) / ‖q·x_cur‖``: the
    direction follows ``x_cur`` and only the magnitude is restored.

    The two formulas agree on flat rest (where ``‖q·x_rest‖ = 0``); they
    diverge on curved rest and disagree in sign whenever ``q·x_cur`` and
    ``q·x_rest`` point in opposite directions.
    """

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_scatter_follows_x_cur_not_x_rest(self) -> None:
        """Contribute along the current curvature direction (+z) of a stencil whose rest curvature points along -z.

        The contributions must be ``w · q · (0, 0, +norm_rest)`` (sign of
        ``q·x_cur``). Restoring the rest curvature vector would contribute
        ``w · q · (q·x_ref)`` ~ ``w · q · (0, 0, -norm_rest)``, the opposite sign.
        """
        from newton._src.solvers.fba.kernels import project_bending_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # Stencil chosen so ``q = [1, 1, -1, -1]`` satisfies sum-to-zero.
        # x_cur has only v1 lifted along +z by 1.0 so ``q · x_cur = (0, 0, +1)``.
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(0, 0, 1), wp.vec3(0, 0, 0), wp.vec3(0, 0, 0)],
            dtype=wp.vec3,
            device=device,
        )
        edge_indices = wp.array([[0, 1, 2, 3]], dtype=wp.int32, device=device)
        edge_quad_q = wp.array([wp.vec4(1.0, 1.0, -1.0, -1.0)], dtype=wp.vec4, device=device)
        weight = wp.array([2.0], dtype=wp.float32, device=device)
        # Rest curvature magnitude = 0.5; the rest direction is irrelevant:
        # the kernel only uses the magnitude.
        edge_norm = wp.array([0.5], dtype=wp.float32, device=device)

        rhs = _projection_rhs(project_bending_kernel, [positions, edge_indices, edge_quad_q, weight, edge_norm], 4)

        # Expected: ``w · q[a] · ê · norm_rest`` with
        # ``ê = q·x_cur / ‖q·x_cur‖ = (0, 0, 1)``.
        # rhs[0] = 2 * 1 * (0, 0, 1) * 0.5 = (0, 0, 1)
        # rhs[1] = 2 * 1 * (0, 0, 1) * 0.5 = (0, 0, 1)
        # rhs[2] = 2 * -1 * (0, 0, 1) * 0.5 = (0, 0, -1)
        # rhs[3] = 2 * -1 * (0, 0, 1) * 0.5 = (0, 0, -1)
        expected = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rhs, expected, atol=1e-6)

        # Restoring the rest curvature vector would contribute
        # ``w · q · (q · x_ref)`` with ``q · x_ref = (0, 0, -0.5)``, along -z;
        # the kernel restores only the magnitude, along +z.
        self.assertGreater(rhs[0, 2], 0.0)
        self.assertLess(rhs[2, 2], 0.0)

    def test_zero_curvature_x_cur_skips_scatter(self) -> None:
        """Contribute nothing when ``‖q·x_cur‖ ≈ 0``, without dividing by zero, whatever ``edge_norm`` is."""
        from newton._src.solvers.fba.kernels import project_bending_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # All positions zero ⇒ ``q · x_cur = 0``.
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(0, 0, 0), wp.vec3(0, 0, 0), wp.vec3(0, 0, 0)],
            dtype=wp.vec3,
            device=device,
        )
        edge_indices = wp.array([[0, 1, 2, 3]], dtype=wp.int32, device=device)
        edge_quad_q = wp.array([wp.vec4(1.0, 1.0, -1.0, -1.0)], dtype=wp.vec4, device=device)
        weight = wp.array([2.0], dtype=wp.float32, device=device)
        edge_norm = wp.array([0.5], dtype=wp.float32, device=device)

        rhs = _projection_rhs(project_bending_kernel, [positions, edge_indices, edge_quad_q, weight, edge_norm], 4)
        np.testing.assert_allclose(rhs, np.zeros((4, 3)), atol=0.0)

    def test_pre_bent_strip_preserves_curvature_magnitude(self) -> None:
        """Keep a 3x3 cloth strip with a curved rest shape near its rest shape under bending alone, without gravity.

        The local projection pulls only the curvature magnitude back. The rest configuration is an equilibrium, so the
        test mainly checks that the bending path runs end to end without NaN or runaway growth.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        grid = newton.ModelBuilder(up_axis=newton.Axis.Z)
        grid.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.5,
            cell_y=0.5,
            mass=1.0,
            tri_ke=50.0,
        )
        # Curve the rest: lift the centre vertex out of the plane, so the rest
        # angles make ``‖q·x_rest‖ > 0`` on every interior edge. The (2, 2)
        # grid has 9 particles; the centre vertex is at index 4.
        vertices = np.array(grid.particle_q)
        vertices[4, 2] += 0.2
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(v) for v in vertices],
            indices=np.array(grid.tri_indices).reshape(-1),
            density=1.0,
            tri_ke=50.0,
            tri_ka=1.0e2,
            tri_kd=0.0,
            edge_ke=10.0,
            edge_kd=0.0,
        )
        model = builder.finalize(device="cpu")

        # Pin all four corners so the strip is anchored but the interior
        # vertices stay free.
        inv_mass = model.particle_inv_mass.numpy().copy()
        for corner in (0, 2, 6, 8):
            inv_mass[corner] = 0.0
        model.particle_inv_mass = wp.array(inv_mass, dtype=wp.float32, device=model.device)
        mass = np.where(inv_mass > 0.0, 1.0 / np.maximum(inv_mass, 1.0e-30), 0.0).astype(np.float32)
        model.particle_mass = wp.array(mass, dtype=wp.float32, device=model.device)

        solver = SolverFBA(model, iterations=8, pin_stiffness=1e10)

        state_in = model.state()
        state_out = model.state()
        dt = 1.0 / 60.0
        for _ in range(20):
            solver.step(state_in, state_out, control=None, contacts=None, dt=dt)
            state_in, state_out = state_out, state_in

        pos_final = state_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(pos_final)), "positions diverged to non-finite values")
        # Centre vertex should remain lifted by roughly the rest curvature.
        # Restoring the rest curvature vector would satisfy this too (rest is
        # an equilibrium of both), but together with the unit tests above this
        # guards the integration.
        self.assertGreater(pos_final[4, 2], 0.1, "centre vertex collapsed below rest curvature")
        self.assertLess(pos_final[4, 2], 0.3, "centre vertex blew up past rest curvature")

    def test_rest_angle_sets_the_rest_curvature(self) -> None:
        """Keep a free hinge folded 60 deg at its rest angle, and unfold it once the rest angle is zeroed.

        Zeroing the rest angles is how a cloth modeled folded in its rest pose is made to rest flat.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        def angle_after(zero_rest: bool, steps: int) -> float:
            h, fold = np.sqrt(3.0) / 2.0, np.radians(60.0)
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0),
                vertices=[
                    wp.vec3(0.0, 0.0, 0.0),
                    wp.vec3(1.0, 0.0, 0.0),
                    wp.vec3(0.5, h, 0.0),
                    wp.vec3(0.5, -h * np.cos(fold), -h * np.sin(fold)),
                ],
                indices=[0, 1, 2, 1, 0, 3],
                density=1.0,
                tri_ke=500.0,
                tri_ka=1.0e3,
                edge_ke=1.0,
            )
            model = builder.finalize()
            if zero_rest:
                model.edge_rest_angle.zero_()
            solver = SolverFBA(model, iterations=10)
            s0, s1 = model.state(), model.state()
            for _ in range(steps):
                solver.step(s0, s1, None, None, 1.0 / 60.0)
                s0, s1 = s1, s0
            x = s0.particle_q.numpy()
            n0 = np.cross(x[1] - x[0], x[2] - x[0])
            n1 = np.cross(x[3] - x[0], x[1] - x[0])
            return float(np.degrees(np.arccos(np.dot(n0, n1) / np.linalg.norm(n0) / np.linalg.norm(n1))))

        self.assertAlmostEqual(angle_after(False, 0), 60.0, places=4)
        self.assertAlmostEqual(angle_after(False, 60), 60.0, delta=0.01)
        self.assertLess(angle_after(True, 10), 15.0)


class TestPublicAPI(unittest.TestCase):
    def test_solver_fba_in_newton_solvers(self):
        """Export SolverFBA from ``newton.solvers``."""
        from newton import solvers  # noqa: PLC0415

        self.assertTrue(hasattr(solvers, "SolverFBA"))
        self.assertIn("SolverFBA", solvers.__all__)


class TestSolverFBAIntegration(unittest.TestCase):
    """Integration tests of SolverFBA stepping cloth."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_hanging_cloth(self, dim: int = 32):
        # Y-up world: cloth grid spans y from 2.0 (bottom row, index 0..dim)
        # to 2+dim*cell_y (top row, index dim*(dim+1)..dim*(dim+1)+dim).
        # Pin the two TOP corners so the cloth hangs under gravity (-Y).
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 2, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=dim,
            dim_y=dim,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.1,
            tri_ke=5000.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=False,
        )
        # Top-row corners in the (dim+1)x(dim+1) grid: y-index=dim.
        top_left = dim * (dim + 1)
        top_right = dim * (dim + 1) + dim
        builder.particle_mass[top_left] = 0.0
        builder.particle_mass[top_right] = 0.0
        return builder.finalize()

    def test_hanging_cloth_stays_finite(self):
        """Step a 16x16 hanging cloth 50 times without non-finite positions."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._build_hanging_cloth(dim=16)
        solver = SolverFBA(model, iterations=8)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite values appeared")

    def test_hanging_cloth_settles(self):
        """Settle a 16x16 hanging cloth over 500 steps with its center between 1.9 m and 2.8 m high."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._build_hanging_cloth(dim=16)
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(500):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        # With realistic tri_ke=1e4, the cloth is stiff and drapes relatively
        # little. The center vertex (grid row dim//2, col dim//2) sits near
        # the midpoint of the hung cloth: observed center_y ≈ 2.35 after 500
        # steps. Bounds are deliberately loose to tolerate minor solver tweaks
        # while still verifying the cloth moved under gravity and stayed finite.
        center_idx = (16 // 2) * (16 + 1) + (16 // 2)
        self.assertLess(q[center_idx, 1], 2.8, "cloth did not fall under gravity")
        self.assertGreater(q[center_idx, 1], 1.9, "cloth fell past plausible drape")


class TestSolverFBAExtremeConfigs(unittest.TestCase):
    """Robustness tests covering extreme configurations: fully-free cloth,
    all-pinned cloth, single triangle, and zero edge stiffness."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_no_pin_free_cloth_remains_finite(self):
        """Keep an unpinned, freely falling cloth finite over 100 steps."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 2, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=8,
            dim_y=8,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=False,
        )
        # NO pinning.
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(100):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite values appeared in free-cloth simulation")

    def test_all_pinned_cloth_stays_put(self):
        """Keep a cloth whose particles are all pinned within 0.1 mm of its start over 50 steps."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=False,
        )
        for i in range(len(builder.particle_mass)):
            builder.particle_mass[i] = 0.0
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        x_initial = s_in.particle_q.numpy().copy()
        dt = 1.0 / 60.0
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        x_final = s_in.particle_q.numpy()
        drift = np.abs(x_final - x_initial).max()
        self.assertLess(drift, 1e-4, f"all-pinned cloth drifted {drift:.3e} m")

    def test_single_triangle_runs(self):
        """Step a single triangle with one pinned vertex, which stays at its pin."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_mesh(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0, 0, 0),
            vertices=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        builder.particle_mass[0] = 0.0  # pin vertex 0
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_allclose(q[0], [0, 0, 0], atol=1e-4)

    def test_zero_edge_stiffness_still_works(self):
        """Step a pinned cloth without bending stiffness to finite positions."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,  # no bending
            fix_left=True,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)))


class TestSolverFBAExternalForces(unittest.TestCase):
    """Robustness tests covering external force application and state buffer
    ping-pong behaviour."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_4x4_cloth(self):
        """Return a 4x4 cloth model with the top-left corner pinned."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        return builder.finalize()

    def test_external_force_displaces_particle(self):
        """Move a free particle differently after one step under a strong external force than without it.

        With ``fix_left=True`` on a 4x4 cloth, the left column (indices 0, 5, 10, 15, 20) is pinned and particle 7 is a
        free interior particle. The test catches ``state_in.particle_f`` not reaching the inertial prediction.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._build_4x4_cloth()
        solver = SolverFBA(model, iterations=5)
        dt = 1.0 / 60.0

        # Verify particle 7 is free (not pinned).
        self.assertGreater(model.particle_inv_mass.numpy()[7], 0.0, "particle 7 should be free")

        # --- No-force baseline ---
        s_in = model.state()
        s_out = model.state()
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, dt)
        q_no_force = s_out.particle_q.numpy().copy()

        # --- With external force on particle 7 (push in +z) ---
        s_in2 = model.state()
        s_out2 = model.state()
        forces_np = np.zeros((model.particle_count, 3), dtype=np.float32)
        forces_np[7] = [0.0, 0.0, 100.0]
        s_in2.particle_f.assign(forces_np)
        solver.step(s_in2, s_out2, None, None, dt)
        q_with_force = s_out2.particle_q.numpy().copy()

        displacement = np.linalg.norm(q_with_force[7] - q_no_force[7])
        self.assertGreater(
            displacement,
            1e-3,
            f"Force on particle 7 produced no measurable displacement: {displacement:.2e} m",
        )

    def test_state_ping_pong_no_leak(self):
        """Produce finite, changed positions over 100 steps that swap the input and output states.

        The test catches in-place changes of ``state_in`` and buffer aliasing.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._build_4x4_cloth()
        solver = SolverFBA(model, iterations=5)
        dt = 1.0 / 60.0

        s_in, s_out = model.state(), model.state()
        initial_q = s_in.particle_q.numpy().copy()

        for _ in range(100):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in

        q_final = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q_final)), "non-finite positions after 100-step ping-pong")
        # Cloth should have fallen (gravity), so at least one particle moved.
        self.assertFalse(
            np.allclose(q_final, initial_q, atol=1e-6),
            "Positions unchanged after 100 steps: cloth did not move",
        )

    def test_clear_forces_resets_between_steps(self):
        """Stop accelerating a particle once a one-time force is cleared, its elastic forces braking it.

        The test catches forces that accumulate across steps.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._build_4x4_cloth()
        solver = SolverFBA(model, iterations=5)
        dt = 1.0 / 60.0

        s_in, s_out = model.state(), model.state()

        # Step 0: apply impulse force on particle 3.
        forces_np = np.zeros((model.particle_count, 3), dtype=np.float32)
        forces_np[3] = [5.0, 0.0, 0.0]
        s_in.particle_f.assign(forces_np)
        solver.step(s_in, s_out, None, None, dt)
        s_in, s_out = s_out, s_in
        v_after_impulse = np.linalg.norm(s_in.particle_qd.numpy()[3])

        # Steps 1..50: clear forces, let cloth relax.
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in

        q_final = s_in.particle_q.numpy()
        v_final = np.linalg.norm(s_in.particle_qd.numpy()[3])

        self.assertTrue(np.all(np.isfinite(q_final)), "non-finite positions after impulse + relaxation")
        self.assertLess(
            v_final,
            v_after_impulse,
            f"Particle 3 did not decelerate: v_final={v_final:.4f} >= v_after_impulse={v_after_impulse:.4f}",
        )

    def test_alternating_ping_pong_buffer_ids(self):
        """Leave ``state_in`` unchanged and write only ``state_out`` in ``step()``."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s_a = model.state()
        s_b = model.state()
        initial_q = s_a.particle_q.numpy().copy()
        s_a.clear_forces()
        solver.step(s_a, s_b, None, None, 1.0 / 60.0)
        # state_in (s_a) should be unchanged.
        np.testing.assert_array_equal(s_a.particle_q.numpy(), initial_q)
        # state_out (s_b) should differ from initial.
        self.assertFalse(
            np.array_equal(s_b.particle_q.numpy(), initial_q),
            "state_out (s_b) was not updated after step()",
        )


class TestSolverFBAReconfigure(unittest.TestCase):
    """Robustness tests covering dt-change re-setup and notify_model_changed."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_4x4_cloth(self):
        """Return a 4x4 cloth model with the left column pinned."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        return builder.finalize()

    def test_dt_change_triggers_resetup(self):
        """Rebuild the linear solver when the time step changes."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s_in, s_out = model.state(), model.state()

        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, 1.0 / 60.0)
        self.assertAlmostEqual(solver._dt_setup, 1.0 / 60.0, places=10)
        linsolver_a = solver._linear_solver

        s_in, s_out = s_out, s_in
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, 1.0 / 120.0)
        self.assertAlmostEqual(solver._dt_setup, 1.0 / 120.0, places=10)
        linsolver_b = solver._linear_solver
        self.assertIsNot(linsolver_a, linsolver_b, "linear solver should have been rebuilt on dt change")

        q = s_out.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)))

    def test_same_dt_no_resetup(self):
        """Keep the linear solver across steps with the same time step."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s_in, s_out = model.state(), model.state()
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, 1.0 / 60.0)
        linsolver_a = solver._linear_solver
        s_in, s_out = s_out, s_in
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, 1.0 / 60.0)
        linsolver_b = solver._linear_solver
        self.assertIs(linsolver_a, linsolver_b, "linear solver should not rebuild when dt unchanged")

    def test_notify_shape_properties_forces_resetup(self):
        """Rebuild the linear solver at the next step after ``notify_model_changed(SHAPE_PROPERTIES)``."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s_in, s_out = model.state(), model.state()
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, 1.0 / 60.0)
        linsolver_a = solver._linear_solver
        self.assertIsNotNone(linsolver_a)

        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        self.assertIsNone(solver._linear_solver, "notify_model_changed should clear _linear_solver")
        self.assertIsNone(solver._dt_setup, "notify_model_changed should clear _dt_setup")

        s_in, s_out = s_out, s_in
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, 1.0 / 60.0)
        self.assertIsNotNone(solver._linear_solver)
        self.assertIsNot(solver._linear_solver, linsolver_a)
        q = s_out.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)))

    def test_notify_unrelated_flag_no_resetup(self):
        """Keep the linear solver after ``notify_model_changed(JOINT_PROPERTIES)``, which does not affect cloth."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=50.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s_in, s_out = model.state(), model.state()
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, 1.0 / 60.0)
        linsolver_a = solver._linear_solver

        # JOINT_PROPERTIES shouldn't affect FBA cloth.
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        self.assertIs(solver._linear_solver, linsolver_a, "FBA should ignore JOINT_PROPERTIES")


class TestSolverFBAStability(unittest.TestCase):
    """Long-run stability of a cloth pinned at two corners."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_pinned_cloth_stays_finite_for_500_steps(self):
        """Keep a 16x16 cloth pinned at two corners finite over 500 steps of 1/60 s with 10 iterations."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 2, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=16,
            dim_y=16,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.005,
            tri_ke=5000.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=False,
        )
        builder.particle_mass[0] = 0.0
        builder.particle_mass[16] = 0.0
        model = builder.finalize()
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        for _ in range(500):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, 1.0 / 60.0)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "documented-safe configuration went NaN")

    def test_large_cloth_dim64_realistic_stiffness_finite(self):
        """Keep a 64x64 hanging cloth with a realistic tri_ke of 1e4 finite over 500 steps with 5 iterations."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 2, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=64,
            dim_y=64,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.005,
            tri_ke=5000.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=False,
        )
        builder.particle_mass[0] = 0.0
        builder.particle_mass[64] = 0.0
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)
        s_in, s_out = model.state(), model.state()
        for _ in range(500):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, 1.0 / 60.0)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "64x64 cloth went NaN at realistic tri_ke=1e4")


class TestDynamicPin(unittest.TestCase):
    """Verify SolverFBA.set_pin_targets moves pinned particle to new targets."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_pin_follows_target(self):
        """Move a pinned particle with its target, 5 cm per step over 10 steps, to within 1 mm of it."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 1, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=5000.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=5)

        # Move pin 0 by 0.5 m in X over 10 steps.
        s_in, s_out = model.state(), model.state()
        x_ref = s_in.particle_q.numpy().copy()
        for _step in range(10):
            x_ref[0, 0] += 0.05  # 5 cm per step
            solver.set_pin_targets(x_ref)
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, 1.0 / 60.0)
            s_in, s_out = s_out, s_in

        # Pinned particle 0 should be close to its target.
        final = s_in.particle_q.numpy()[0]
        target = x_ref[0]
        self.assertLess(
            np.linalg.norm(final - target),
            1e-3,
            f"Pin should follow target; final={final}, target={target}",
        )

    def test_pinned_particles_with_mass_hold_their_targets(self):
        """Hold particles with zero inverse mass at their pin targets when they keep a mass."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 1.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=1.0,
            tri_ke=5000.0,
        )
        model = builder.finalize()
        model.particle_inv_mass.zero_()  # every particle pinned, with its mass kept
        solver = SolverFBA(model, iterations=5, pin_stiffness=1.0e6)
        s0, s1 = model.state(), model.state()
        x0 = s0.particle_q.numpy()
        for _ in range(10):
            solver.step(s0, s1, None, None, 1.0 / 60.0)
            s0, s1 = s1, s0
        np.testing.assert_allclose(s0.particle_q.numpy(), x0, atol=1.0e-6)

    def test_inactive_particles_hold_their_state(self):
        """Hold particles with mass but without the ACTIVE flag where the input state has them, as Newton's
        integrators do; particles with zero mass keep following their pin targets (see test_pin_follows_target)."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 1.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=5000.0,
            edge_ke=0.1,
        )
        model = builder.finalize()
        top = [20, 24]  # the corners of the top row
        flags = model.particle_flags.numpy()
        flags[top] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
        solver = SolverFBA(model, iterations=5)
        s0, s1 = model.state(), model.state()
        x0 = s0.particle_q.numpy()
        for step in range(30):
            if step == 15:  # move a held corner by writing the state
                q = s0.particle_q.numpy()
                self.assertLess(q[12, 1], x0[12, 1] - 1.0e-4, "the cloth does not hang from its corners")
                q[24, 2] += 0.1
                s0.particle_q.assign(q)
                moved = q[24].copy()
            solver.step(s0, s1, None, None, 1.0 / 60.0)
            s0, s1 = s1, s0
        q = s0.particle_q.numpy()
        np.testing.assert_array_equal(q[20], x0[20])
        np.testing.assert_array_equal(q[24], moved)
        np.testing.assert_array_equal(s0.particle_qd.numpy()[top], 0.0)


class SolverFBAConstructorOptionsTests(unittest.TestCase):
    """The models and contacts SolverFBA refuses."""

    def test_solver_init_rejects_no_geometry(self):
        """Refuse a model with no triangles."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize()
        with self.assertRaises(ValueError):
            SolverFBA(model, iterations=1)

    def test_tetrahedra_are_refused(self) -> None:
        """Refuse a model with tetrahedra, which SolverFBA does not simulate."""
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
        )
        with self.assertRaisesRegex(NotImplementedError, "cloth only"):
            SolverFBA(builder.finalize())

    def test_several_worlds_are_refused(self) -> None:
        """Refuse a model with two worlds, whose contact solve and self contacts would couple them."""
        cloth = newton.ModelBuilder()
        cloth.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            tri_ke=50.0,
        )
        builder = newton.ModelBuilder()
        builder.replicate(cloth, 2)
        with self.assertRaisesRegex(ValueError, "one world"):
            SolverFBA(builder.finalize())

    def test_rigid_bodies_are_refused(self) -> None:
        """Refuse a model with a rigid body, which SolverFBA does not simulate."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            tri_ke=50.0,
        )
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        with self.assertRaisesRegex(NotImplementedError, "rigid bodies"):
            SolverFBA(builder.finalize())

    def test_soft_contacts_are_refused(self) -> None:
        """Step with contacts that hold no soft contacts, and refuse contacts that hold some."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            tri_ke=50.0,
        )
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
        contacts = pipeline.contacts()
        solver = SolverFBA(model)
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, 1.0 / 60.0)
        q = s0.particle_q.numpy()
        q[:, 2] = 0.0
        s0.particle_q.assign(q)
        pipeline.collide(s0, contacts)
        with self.assertRaisesRegex(NotImplementedError, "does not solve contacts"):
            solver.step(s0, s1, None, contacts, 1.0 / 60.0)


class ParticleElementCsrTests(unittest.TestCase):
    """Adjacency CSR builder produces correct (offsets, element_idx, local_vertex_idx)."""

    def test_basic_four_vertex_adjacency(self) -> None:
        """Map each particle to its elements and local vertex indices for two four-vertex elements sharing three
        vertices."""
        quads = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
        offsets, elem_idx, local_v = build_particle_element_csr(quads, n_particles=5, n_verts_per_element=4)

        self.assertEqual(offsets.tolist(), [0, 1, 3, 5, 7, 8])
        # Vertex 1's entries: (element=0, local=1) and (element=1, local=0).
        v1_slice = slice(offsets[1], offsets[2])
        elem_v1 = sorted(zip(elem_idx[v1_slice].tolist(), local_v[v1_slice].tolist(), strict=True))
        self.assertEqual(elem_v1, [(0, 1), (1, 0)])

    def test_tri_adjacency(self) -> None:
        """Map each particle to its triangles and local vertex indices for two triangles sharing an edge."""
        tris = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        offsets, elem_idx, local_v = build_particle_element_csr(tris, n_particles=4, n_verts_per_element=3)
        self.assertEqual(offsets.tolist(), [0, 1, 3, 5, 6])
        v1_slice = slice(offsets[1], offsets[2])
        elem_v1 = sorted(zip(elem_idx[v1_slice].tolist(), local_v[v1_slice].tolist(), strict=True))
        self.assertEqual(elem_v1, [(0, 1), (1, 0)])

    def test_empty_adjacency(self) -> None:
        """Build an empty adjacency for no elements, with zero offsets for every particle."""
        empty = np.zeros((0, 4), dtype=np.int32)
        offsets, elem_idx, local_v = build_particle_element_csr(empty, n_particles=3, n_verts_per_element=4)
        self.assertEqual(offsets.tolist(), [0, 0, 0, 0])
        self.assertEqual(elem_idx.shape, (0,))
        self.assertEqual(local_v.shape, (0,))


class SolverFBATriDeterminismTests(unittest.TestCase):
    """The triangle stretching projections give bit-identical positions in two runs."""

    def _run_once(self) -> np.ndarray:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
        builder.add_cloth_grid(
            pos=wp.vec3(-0.2, -0.2, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=5,
            dim_y=5,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
            tri_ke=5000.0,
            tri_ka=0.0,
            tri_kd=0.0,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=1)
        s_in = model.state()
        s_out = model.state()
        for _ in range(3):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, 1.0 / 60.0)
            s_in, s_out = s_out, s_in
        return s_in.particle_q.numpy().copy()

    def test_arap_two_runs_identical(self) -> None:
        """Produce bit-identical triangle ARAP positions in two runs."""
        q1 = self._run_once()
        q2 = self._run_once()
        np.testing.assert_array_equal(q1, q2, err_msg="tri ARAP projection non-deterministic")


class SolverFBABendingDeterminismTests(unittest.TestCase):
    """The bending projection gives bit-identical positions in two runs."""

    def _run_once(self) -> np.ndarray:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
        # Cloth with bending edges enabled.
        builder.add_cloth_grid(
            pos=wp.vec3(-0.2, -0.2, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=5,
            dim_y=5,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
            tri_ke=5000.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-2,  # bending stiffness > 0
            edge_kd=0.0,
        )
        model = builder.finalize()
        solver = SolverFBA(model, iterations=1)
        s_in = model.state()
        s_out = model.state()
        for _ in range(3):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, 1.0 / 60.0)
            s_in, s_out = s_out, s_in
        return s_in.particle_q.numpy().copy()

    def test_two_runs_identical(self) -> None:
        """Produce bit-identical bent cloth positions in two runs."""
        q1 = self._run_once()
        q2 = self._run_once()
        np.testing.assert_array_equal(q1, q2, err_msg="bending projection non-deterministic")


if __name__ == "__main__":
    unittest.main()
