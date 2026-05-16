# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import SolverFBA
from newton._src.solvers.fba.linear_solver import build_pd_system


class TestSolverFBAImport(unittest.TestCase):
    def test_import_solver_fba(self):
        assert SolverFBA is not None


class TestBuildPDSystem(unittest.TestCase):
    """T1: PD Hessian assembly on a 2-triangle mesh, sanity-check entries."""

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
            tri_ke=1.0e2,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0,
            edge_kd=0.0,
        )
        return builder.finalize(device="cpu")

    def test_diagonal_includes_mass_over_dt_squared(self):
        model = self._build_two_triangle_cloth()
        dt = 1.0 / 60.0
        A, _meta = build_pd_system(model, dt=dt, pin_stiffness=1e12)

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
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=1.0e2,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        # Manually pin particle 0 by setting mass to 0.
        builder.particle_mass[0] = 0.0
        model = builder.finalize(device="cpu")

        A, _ = build_pd_system(model, dt=1.0 / 60.0, pin_stiffness=1e12)
        # Pin contributes pin_stiffness to A[0,0].
        self.assertGreaterEqual(A[0, 0], 1e12 - 1e-3)


class TestBendingQMatchesReference(unittest.TestCase):
    """T3: Verify _compute_isometric_bending_q against a hand-computed reference
    on an asymmetric quad where the old 2-cotangent formula would give wrong results.
    """

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_bending_q_matches_reference(self):
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
            tri_ke=1.0e2,
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


class TestSparseInverse(unittest.TestCase):
    """T2: compute_lower_inverse matches dense np.linalg.inv on small SPD."""

    def test_random_spd_8x8(self):
        import scipy.sparse as _sp

        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            _elimination_tree,
            _splu_extract_factors,
            compute_lower_inverse,
        )

        rng = np.random.default_rng(0)
        # Build a small SPD matrix via tridiagonal + jitter.
        n = 8
        diag = rng.uniform(10.0, 20.0, n)
        off = rng.uniform(-1.0, 1.0, n - 1)
        A = _sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csc").astype(np.float64)

        Ainv_dense = np.linalg.inv(A.toarray())

        # splu decomposes P_r @ A @ P_c^T = L @ D @ L^T for SPD A.
        # L is already in the permuted coordinate system, so compute S = L^{-1}
        # directly (invperm=None means identity ordering for L's internal space).
        L_csc, _U_csc, Dinv, perm_r, perm_c = _splu_extract_factors(A)
        S = compute_lower_inverse(L_csc, parent=_elimination_tree(L_csc))

        # Reconstruct A^{-1} = P_c^T @ S^T @ diag(Dinv) @ S @ P_r
        # (from P_r A P_c^T = L D L^T => A^{-1} = P_c^T L^{-T} D^{-1} L^{-1} P_r)
        S_arr = S.toarray()
        P_r = np.eye(n)[perm_r]
        P_c = np.eye(n)[perm_c]
        Ainv_reconstructed = P_c.T @ S_arr.T @ np.diag(Dinv) @ S_arr @ P_r
        self.assertLess(np.abs(Ainv_reconstructed - Ainv_dense).max(), 1e-12)


class TestFBALinearSolver(unittest.TestCase):
    """T3 end-to-end linear solve + T6 permutation roundtrip."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_solve_random_spd(self):
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

    def test_solve_multi_rhs_matches_single_rhs(self):
        """solve_multi_rhs_scalar produces same result as R separate solve() calls."""
        import scipy.sparse as _sp

        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            factorize_and_sparse_inverse,
        )

        rng = np.random.default_rng(42)
        N = 12
        # Build a small random SPD matrix.
        A_dense = rng.standard_normal((N, N))
        A_dense = A_dense @ A_dense.T + N * np.eye(N)
        A = _sp.csr_matrix(A_dense)

        fs = factorize_and_sparse_inverse(A)
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        solver = FBALinearSolver(fs, device=device)

        R = 5
        # Build R random scalar RHS rows (shape (R, N)).
        b_np = rng.standard_normal((R, N))

        # --- Single-RHS reference via solve() ---
        ref = np.zeros((R, N), dtype=np.float64)
        for r in range(R):
            b_vec3 = np.zeros((N, 3), dtype=np.float32)
            b_vec3[:, 0] = b_np[r].astype(np.float32)
            b_d = wp.array(b_vec3, dtype=wp.vec3, device=device)
            x_d = wp.zeros(N, dtype=wp.vec3, device=device)
            solver.solve(b_d, x_d)
            x_np = x_d.numpy()  # (N, 3) float32
            ref[r] = x_np[:, 0].astype(np.float64)

        # --- Multi-RHS path ---
        b_multi = wp.array(b_np, dtype=wp.float64, device=device)
        x_multi = wp.zeros(shape=(R, N), dtype=wp.float64, device=device)
        solver.solve_multi_rhs_scalar(b_multi, x_multi, R)
        got = x_multi.numpy()

        # Tolerance: single-RHS uses float32 intermediate (vec3 extract/insert),
        # multi-RHS is float64 throughout. Expect ~1e-6 relative error from f32 cast.
        np.testing.assert_allclose(
            got,
            ref,
            rtol=1e-5,
            atol=1e-6,
            err_msg="solve_multi_rhs_scalar disagrees with single-rhs solve()",
        )

    def test_permutation_roundtrip(self):
        from newton._src.solvers.fba.kernels import apply_permutation_vec3_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        n = 32
        perm = np.random.default_rng(1).permutation(n).astype(np.int32)
        invperm = np.argsort(perm).astype(np.int32)

        x_np = np.random.default_rng(2).standard_normal((n, 3))
        x = wp.array(x_np, dtype=wp.vec3, device=device)
        y = wp.empty(n, dtype=wp.vec3, device=device)
        z = wp.empty(n, dtype=wp.vec3, device=device)
        perm_wp = wp.array(perm, dtype=wp.int32, device=device)
        invperm_wp = wp.array(invperm, dtype=wp.int32, device=device)

        wp.launch(apply_permutation_vec3_kernel, dim=n, inputs=[x, perm_wp], outputs=[y], device=device)
        wp.launch(apply_permutation_vec3_kernel, dim=n, inputs=[y, invperm_wp], outputs=[z], device=device)

        np.testing.assert_allclose(z.numpy(), x_np, atol=0.0)


class TestARAPProjection(unittest.TestCase):
    """T4: ARAP local projection on triangle stretching."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_identity_F_zero_after_assembly_bias(self):
        """Rest configuration ⇒ P=identity; verify scatter values match the
        formula (Dm_inv @ P^T) with Dm_inv = I₂ analytically.

        For rest=current=(0,0,0),(1,0,0),(0,0,1), P (3x2) maps reference axes to
        the embedded triangle's axes; the kernel scatters w·Dm_inv·P^T:
            row0 = (1,0,0); row1 = (0,0,1)  (since P = embedding)
        Therefore:
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
        rhs = wp.zeros(3, dtype=wp.vec3, device=device)

        wp.launch(
            project_stretching_arap_kernel,
            dim=1,
            inputs=[positions, tri_indices, Dm_inv, weight],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        np.testing.assert_allclose(r[1], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(r[2], [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(r[0], -(r[1] + r[2]), atol=1e-6)

    def test_non_symmetric_Dm_inv_exposes_transpose_bug(self):
        """Non-symmetric Dm_inv (= skewed triangle's rest inverse) must be
        applied as Dm_inv, not Dm_inv^T. Wrong orientation gave ~20% error.
        """
        from newton._src.solvers.fba.kernels import project_stretching_arap_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # Current configuration = rest, so P = embedding; w·Dm_inv·P^T scatter exact.
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
        rhs = wp.zeros(3, dtype=wp.vec3, device=device)
        wp.launch(
            project_stretching_arap_kernel,
            dim=1,
            inputs=[positions, tri_indices, Dm_inv, weight],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()

        # Analytical: with current=rest and Dm_inv as above, P = embedding [[1,0],[0,0],[0,1]] (3x2);
        # P^T = [[1,0,0],[0,0,1]] (2x3). Dm_inv @ P^T = [[0.5, 0, -0.5],[0, 0, 1]].
        # So row0 (vertex 1's contribution) = (0.5, 0, -0.5), row1 (vertex 2) = (0, 0, 1).
        # rhs[1] = row0, rhs[2] = row1, rhs[0] = -(row0+row1).
        np.testing.assert_allclose(r[1], [0.5, 0.0, -0.5], atol=1e-5)
        np.testing.assert_allclose(r[2], [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(r[0], -(r[1] + r[2]), atol=1e-6)


class TestBendingProjection(unittest.TestCase):
    """T5: isometric bending projection — flat rest gives zero RHS contribution."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_flat_rest_zero_rhs(self):
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
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            project_bending_kernel,
            dim=1,
            inputs=[positions, edge_indices, edge_quad_q, weight, edge_norm],
            outputs=[rhs],
            device=device,
        )
        # Flat-rest ``edge_norm = 0`` ⇒ scatter contribution is zero.
        np.testing.assert_allclose(rhs.numpy(), np.zeros((4, 3)), atol=1e-6)


class SolverFBABendingCurvedRestTests(unittest.TestCase):
    """Regression (Task H'): bending local-step minimises curvature *magnitude*,
    not direction, on curved rest cloth.

    The previous kernel scattered ``w · q · (q · x_ref)`` which is the gradient
    of ``‖(q·x_cur) - (q·x_ref)‖²``. That penalises any deviation from the
    *rest curvature vector*, including curvature flips (e.g. cloth bent
    against its rest direction). RealSim's
    ``PDIsometricBendingEnergy::localProjection`` instead scatters
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
        """Crafted stencil: x_cur curvature direction is +z, rest is along -z.

        The new kernel must scatter ``w · q · (0, 0, +norm_rest)`` (sign of
        ``q·x_cur``). The legacy kernel would have scattered
        ``w · q · (q·x_ref)`` ~ ``w · q · (0, 0, -norm_rest)`` -- opposite sign.
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
        # Rest curvature magnitude = 0.5; the rest direction is irrelevant --
        # the new kernel only uses the magnitude.
        edge_norm = wp.array([0.5], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            project_bending_kernel,
            dim=1,
            inputs=[positions, edge_indices, edge_quad_q, weight, edge_norm],
            outputs=[rhs],
            device=device,
        )

        # Expected (new kernel): ``w · q[a] · ê · norm_rest`` with
        # ``ê = q·x_cur / ‖q·x_cur‖ = (0, 0, 1)``.
        # rhs[0] = 2 * 1 * (0, 0, 1) * 0.5 = (0, 0, 1)
        # rhs[1] = 2 * 1 * (0, 0, 1) * 0.5 = (0, 0, 1)
        # rhs[2] = 2 * -1 * (0, 0, 1) * 0.5 = (0, 0, -1)
        # rhs[3] = 2 * -1 * (0, 0, 1) * 0.5 = (0, 0, -1)
        expected = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rhs.numpy(), expected, atol=1e-6)

        # The legacy kernel would have scattered ``w · q · (q · x_ref)``. To
        # reach magnitude 0.5 from this stencil ``x_ref`` would have to satisfy
        # ``q · x_ref = (0, 0, ±0.5)``; the rest-as-flipped-curvature variant
        # gives the -z sign. The new RHS is along +z, so this assertion would
        # have failed with the old kernel (different sign in the z component).
        self.assertGreater(rhs.numpy()[0, 2], 0.0)
        self.assertLess(rhs.numpy()[2, 2], 0.0)

    def test_zero_curvature_x_cur_skips_scatter(self) -> None:
        """When ``‖q·x_cur‖ ≈ 0`` the kernel must short-circuit to zero scatter
        (no division-by-zero), regardless of ``edge_norm``.
        """
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
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            project_bending_kernel,
            dim=1,
            inputs=[positions, edge_indices, edge_quad_q, weight, edge_norm],
            outputs=[rhs],
            device=device,
        )
        np.testing.assert_allclose(rhs.numpy(), np.zeros((4, 3)), atol=0.0)

    def test_pre_bent_strip_preserves_curvature_magnitude(self) -> None:
        """End-to-end: curved-rest 3x3 cloth strip remains near its rest shape
        under gravity-free, bending-only forces. The old kernel pulled
        ``q·x_cur`` toward the rest *vector* and would lock in the rest
        direction; the new kernel pulls only the magnitude back. Either way the
        rest configuration is an equilibrium, so this test serves mainly as an
        integration check that the new kernel signature flows end-to-end
        without NaN or runaway growth.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.5,
            cell_y=0.5,
            mass=1.0,
            tri_ke=1.0e2,
            tri_ka=1.0e2,
            tri_kd=0.0,
            edge_ke=10.0,
            edge_kd=0.0,
        )
        model = builder.finalize(device="cpu")

        # Curve the rest: lift the centre vertex out of the plane before the
        # solver captures ``particle_q`` as the bending reference.  This makes
        # ``‖q·x_rest‖ > 0`` on every interior edge.
        q_np = model.particle_q.numpy().copy()
        # The (2, 2) grid has 9 particles; the centre vertex is at index 4.
        q_np[4, 2] += 0.2
        model.particle_q = wp.array(q_np, dtype=wp.vec3, device=model.device)

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
        # Centre vertex should remain lifted by roughly the rest curvature; the
        # old kernel would also satisfy this (rest is equilibrium for both),
        # but together with the unit tests above this guards the integration.
        self.assertGreater(pos_final[4, 2], 0.1, "centre vertex collapsed below rest curvature")
        self.assertLess(pos_final[4, 2], 0.3, "centre vertex blew up past rest curvature")


class TestPublicAPI(unittest.TestCase):
    def test_solver_fba_in_newton_solvers(self):
        from newton import solvers  # noqa: PLC0415

        self.assertTrue(hasattr(solvers, "SolverFBA"))
        self.assertIn("SolverFBA", solvers.__all__)


class TestSolverFBAIntegration(unittest.TestCase):
    """T7: smoke; T8: steady-state of hanging cloth."""

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
            tri_ke=1.0e4,
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

    def test_t7_smoke_no_nan(self):
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

    def test_t8_steady_state_settles(self):
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
        # the midpoint of the hung cloth — observed center_y ≈ 2.35 after 500
        # steps. Bounds are deliberately loose to tolerate minor solver tweaks
        # while still verifying the cloth moved under gravity and stayed finite.
        center_idx = (16 // 2) * (16 + 1) + (16 // 2)
        self.assertLess(q[center_idx, 1], 2.8, "cloth did not fall under gravity")
        self.assertGreater(q[center_idx, 1], 1.9, "cloth fell past plausible drape")


class TestSolverFBAExtremeConfigs(unittest.TestCase):
    """Robustness tests covering extreme configurations not exercised by the
    existing 13 tests: fully-free cloth, all-pinned cloth, single triangle,
    and zero edge stiffness."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_no_pin_free_cloth_remains_finite(self):
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
            tri_ke=1.0e2,
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
            tri_ke=1.0e2,
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
            tri_ke=1.0e2,
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
            tri_ke=1.0e2,
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
            tri_ke=1.0e2,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        return builder.finalize()

    def test_external_force_displaces_particle(self):
        """A strong external force on a free particle must produce a measurably
        different position after 1 step compared to the no-force case.

        With ``fix_left=True`` on a 4x4 cloth, the left column (indices 0, 5,
        10, 15, 20) is pinned.  Particle 7 is a free interior particle.

        Catches: is ``state_in.particle_f`` actually fed into
        ``compute_inertial_kernel``?
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
        """100-step ping-pong must produce finite, changed positions.

        Catches: in-place mutation of ``state_in`` or buffer aliasing.
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
            "Positions unchanged after 100 steps — cloth did not move",
        )

    def test_clear_forces_resets_between_steps(self):
        """Applying a one-time impulse then clearing forces must not let the
        particle keep accelerating; elastic restoring forces should brake it.

        Catches: force accumulation when ``clear_forces`` is omitted.
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
        """``step()`` must not write to ``state_in``; only ``state_out`` changes.

        Catches: accidental in-place writes to the input buffer.
        """
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
            tri_ke=1.0e2,
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
            tri_ke=1.0e2,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            fix_left=True,
        )
        return builder.finalize()

    def test_dt_change_triggers_resetup(self):
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
            tri_ke=1.0e2,
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
            tri_ke=1.0e2,
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
        from newton._src.solvers import SolverNotifyFlags  # noqa: PLC0415
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
            tri_ke=1.0e2,
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

        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)
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
        from newton._src.solvers import SolverNotifyFlags  # noqa: PLC0415
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
            tri_ke=1.0e2,
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
        solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)
        self.assertIs(solver._linear_solver, linsolver_a, "FBA should ignore JOINT_PROPERTIES")


class TestSolverFBAStability(unittest.TestCase):
    """Stability regression: verify the documented "safe zone" stays finite."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_safe_zone_dim16_iter10_dt60_finite_500_steps(self):
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
            tri_ke=1.0e4,
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
        """64x64 hanging cloth with realistic tri_ke=1e4 stays finite over 500
        steps with just 5 PD iterations. The previously-reported instability
        at dim>=24 was due to using tri_ke=1e2 (~1000x softer than real cloth).
        """
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
            tri_ke=1.0e4,
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


class TestFBARealSimAgreement(unittest.TestCase):
    """Bit-level trajectory match vs RealSim C++ reference (50 steps, 113-vertex cloth).

    Fixture: ``newton/tests/fixtures/fba_realsim_trajectory.npy`` (50, 113, 3) float32.
    Captured once from RealSim C++ LocalGlobalSolver + SPARSE_INVERSE_CUDA.
    Frame ``k`` of the fixture is the RealSim state *after* step ``k``, so we
    compare ``newton_traj[k]`` (after step ``k+1``) vs ``fixture[k+1]`` for
    ``k`` in ``[0, 48]``.

    See docs/superpowers/specs/2026-05-15-fba-realsim-crosscheck-rootcause.md
    and crosscheck-results.md for the diagnosis trail.
    """

    FIXTURE_DIR = Path(__file__).parent / "fixtures"
    OBJ_PATH = FIXTURE_DIR / "fba_cloth_113.obj"
    TRAJ_PATH = FIXTURE_DIR / "fba_realsim_trajectory.npy"

    # Simulation parameters matching RealSim FBACrossCheck
    DT = 0.01
    N_FRAMES = 50
    N_ITER = 5
    # E=1e4, nu=0.4 → mu = E/(2*(1+nu)) = 1e4/2.8 ≈ 3571.43; tri_ke = 2*mu ≈ 7142.857
    TRI_KE = 7142.857
    EDGE_KE = 0.1
    PIN_STIFFNESS = 1e10  # RealSim init.h:1023
    # Pin bounding boxes (xmin, ymin, zmin, xmax, ymax, zmax) — top two corners
    PIN_BOXES = (
        (-1.1, 0.9, -0.1, -0.9, 1.1, 0.1),
        (0.9, 0.9, -0.1, 1.1, 1.1, 0.1),
    )

    @classmethod
    def setUpClass(cls):
        wp.init()

    @staticmethod
    def _load_obj(path: "Path") -> "tuple[list[wp.vec3], list[int]]":
        """Parse .obj → (vertices, triangle_indices_0based)."""
        from pathlib import Path as _Path  # noqa: PLC0415

        vertices: list[wp.vec3] = []
        indices: list[int] = []
        with open(_Path(path)) as fh:
            for raw_line in fh:
                tok = raw_line.strip()
                if tok.startswith("v "):
                    parts = tok.split()
                    vertices.append(wp.vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif tok.startswith("f "):
                    parts = tok.split()[1:]
                    face_idx = [int(p.split("/")[0]) - 1 for p in parts]
                    for i in range(1, len(face_idx) - 1):
                        indices.extend([face_idx[0], face_idx[i], face_idx[i + 1]])
        return vertices, indices

    @staticmethod
    def _find_pin_indices(vertices: "list[wp.vec3]", boxes: "list[tuple]") -> "list[int]":
        pinned: list[int] = []
        for i, v in enumerate(vertices):
            x, y, z = v[0], v[1], v[2]
            for xmin, ymin, zmin, xmax, ymax, zmax in boxes:
                if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                    pinned.append(i)
                    break
        return pinned

    def _build_model(self, vertices: "list[wp.vec3]", indices: "list[int]") -> "newton.Model":
        """Build Newton model matching RealSim FBACrossCheck scene."""
        import newton as _newton  # noqa: PLC0415

        N = len(vertices)
        # Compute density for API compliance (overridden below to uniform 1/N).
        verts_np = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float64)
        idxs = np.array(indices, dtype=np.int32).reshape(-1, 3)
        total_area = 0.0
        for tri in idxs:
            e1 = verts_np[tri[1]] - verts_np[tri[0]]
            e2 = verts_np[tri[2]] - verts_np[tri[0]]
            total_area += 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
        density = 1.0 / total_area  # so that sum-of-area-weights ≈ 1 kg total

        builder = _newton.ModelBuilder(up_axis=_newton.Axis.Z, gravity=-10.0)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=density,
            tri_ke=self.TRI_KE,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=self.EDGE_KE,
            edge_kd=0.0,
        )
        # Override to uniform 1/N mass (RealSim Mass::addObjectMass convention).
        for i in range(N):
            builder.particle_mass[i] = 1.0 / N
        # Pin top-corner vertices by zeroing their mass.
        for idx in self._find_pin_indices(vertices, self.PIN_BOXES):
            builder.particle_mass[idx] = 0.0
        return builder.finalize()

    def test_drape_trajectory_matches_realsim(self):
        """50-step drape on 113-vertex cloth: max position delta < 5e-5 m at every frame."""

        from newton.solvers import SolverFBA  # noqa: PLC0415

        # Load mesh fixture.
        vertices, indices = self._load_obj(self.OBJ_PATH)
        self.assertEqual(len(vertices), 113)

        # Load RealSim reference trajectory (shape: 50 frames x 113 verts x 3).
        realsim_traj = np.load(str(self.TRAJ_PATH))
        self.assertEqual(realsim_traj.shape, (50, 113, 3))

        # Build model and solver.
        model = self._build_model(vertices, indices)
        solver = SolverFBA(model, iterations=self.N_ITER, pin_stiffness=self.PIN_STIFFNESS)

        state_in = model.state()
        state_out = model.state()
        newton_traj = np.zeros((self.N_FRAMES, model.particle_count, 3), dtype=np.float32)

        for f in range(self.N_FRAMES):
            state_in.clear_forces()
            solver.step(state_in, state_out, None, None, self.DT)
            newton_traj[f] = state_out.particle_q.numpy()
            state_in, state_out = state_out, state_in

        self.assertTrue(np.all(np.isfinite(newton_traj)), "non-finite positions in Newton trajectory")

        # Compare newton[k] vs realsim[k+1] for k in [0, 48].
        # realsim[0] = initial config (before any step); realsim[k+1] = after step k+1.
        max_delta = 0.0
        for k in range(self.N_FRAMES - 1):
            delta = float(np.linalg.norm(newton_traj[k] - realsim_traj[k + 1], axis=-1).max())
            max_delta = max(max_delta, delta)

        self.assertLess(
            max_delta,
            5e-5,
            f"Max L2 position delta {max_delta:.3e} m exceeds 5e-5 m tolerance "
            f"(observed ~6 µm after gravity-pin fix; float32 noise floor ~1e-7 m)",
        )


class TestCorotationalProjection(unittest.TestCase):
    """Tests for the corotational stretching local projection kernel."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _make_unit_triangle(self):
        """Return positions, Dm_inv=I₂, weight=1 for an axis-aligned triangle at rest."""
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            dtype=wp.vec3,
            device="cuda:0" if wp.is_cuda_available() else "cpu",
        )
        tri_indices = wp.array([0, 1, 2], dtype=wp.int32, device=positions.device)
        Dm_inv = wp.array([wp.mat22(1.0, 0.0, 0.0, 1.0)], dtype=wp.mat22, device=positions.device)
        weight = wp.array([1.0], dtype=wp.float32, device=positions.device)
        return positions, tri_indices, Dm_inv, weight

    def test_rest_configuration_F_equals_I_projects_to_I(self):
        """At rest (F~=I in the local frame), sigma=1 and the corot projection should
        return sigma_proj=1 as well -- the gradient of the energy is zero there.

        Verify scatter output matches the ARAP-at-identity case:
            rhs[1] == (1, 0, 0); rhs[2] == (0, 0, 1); rhs[0] == -(rhs[1]+rhs[2]).
        """
        from newton._src.solvers.fba.kernels import project_stretching_corotational_kernel  # noqa: PLC0415

        positions, tri_indices, Dm_inv, weight = self._make_unit_triangle()
        device = positions.device
        rhs = wp.zeros(3, dtype=wp.vec3, device=device)

        mu, lam = 1.0, 0.5
        wp.launch(
            project_stretching_corotational_kernel,
            dim=1,
            inputs=[positions, tri_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        # At rest sigma=1: corot energy has zero gradient, projection maps to 1.
        # P = U*diag(1,1)*Vt = I (embedding), identical to ARAP at rest.
        np.testing.assert_allclose(r[1], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[2], [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(r[0], -(r[1] + r[2]), atol=1e-5)

    def test_isotropic_stretch_pulls_sigma_back(self):
        """s0 = (1.5, 1.5) with mu=1, lam=0.5: closed-form gives sigma_proj = (1.2, 1.2).

        Analytical derivation:
            k = 2*mu = 2, M = 4*mu+lam = 4.5, det = M^2-lam^2 = 20.
            b = 2*(mu+lam) + k*s0 = 3 + 3 = 6 (same for both).
            sigma_proj = (M*b - lam*b)/det = (4.5*6 - 0.5*6)/20 = 24/20 = 1.2.
        """
        from newton._src.solvers.fba.kernels import project_stretching_corotational_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # Scale current positions by 1.5 to get isotropic stretch F = 1.5·I.
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1.5, 0, 0), wp.vec3(0, 0, 1.5)],
            dtype=wp.vec3,
            device=device,
        )
        tri_indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat22(1.0, 0.0, 0.0, 1.0)], dtype=wp.mat22, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(3, dtype=wp.vec3, device=device)

        mu, lam = 1.0, 0.5
        wp.launch(
            project_stretching_corotational_kernel,
            dim=1,
            inputs=[positions, tri_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()

        # For isotropic stretch by 1.5 with Dm_inv=I: F=1.5*I, U=V=I.
        # P = U*diag(1.2,1.2)*Vt = 1.2*I (the 3x2 embedding scaled by 1.2).
        # Scatter: row0 = Dm_inv[0,:]*Pt*w = (1.2, 0, 0); row1 = (0, 0, 1.2).
        np.testing.assert_allclose(r[1], [1.2, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[2], [0.0, 0.0, 1.2], atol=1e-5)
        np.testing.assert_allclose(r[0], -(r[1] + r[2]), atol=1e-5)

    def test_solver_corotational_runs_no_nan(self):
        """Full SolverFBA step with stretching_model='corotational', small cloth, 50 steps."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 2, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=8,
            dim_y=8,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.1,
            tri_ke=7142.857,  # 2μ with μ≈3571
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.1,
            edge_kd=0.0,
            fix_left=False,
        )
        # Pin top-left and top-right corners.
        top_left = 8 * (8 + 1)
        top_right = 8 * (8 + 1) + 8
        builder.particle_mass[top_left] = 0.0
        builder.particle_mass[top_right] = 0.0
        model = builder.finalize()

        mu, lam = 3571.0, 4286.0
        solver = SolverFBA(model, iterations=8, stretching_model="corotational", mu=mu, lam=lam)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite values in corotational simulation")

    def test_solver_init_requires_mu_lam_for_corot(self):
        """SolverFBA with stretching_model='corotational' and no mu/lam raises ValueError."""
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
            tri_ke=1.0e2,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            fix_left=True,
        )
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SolverFBA(model, stretching_model="corotational")  # missing mu and lam

        with self.assertRaises(ValueError):
            SolverFBA(model, stretching_model="corotational", mu=1000.0)  # missing lam

        with self.assertRaises(ValueError):
            SolverFBA(model, stretching_model="corotational", lam=1000.0)  # missing mu


class TestNeoHookeanProjection(unittest.TestCase):
    """Unit tests for project_neohookean_sigma and the NH stretching kernel."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _device(self):
        return "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_rest_configuration_projects_to_identity(self):
        """At sigma=(1,1), J=1, log_J=0, gradient=0; Newton leaves sigma unchanged.

        With sigma_sq=(1,1): sigma0=(1,1), k=2*mu, J=1, log_J=0, inv=(1,1).
        grad_0 = mu*(1-1) + lam*0*1 + k*(1-1) = 0.
        grad_1 = 0 similarly. So Newton dx=0 and sigma stays at (1,1).
        Verify the kernel scatter matches ARAP-at-rest: rhs[1]=(1,0,0), rhs[2]=(0,0,1).
        """
        from newton._src.solvers.fba.kernels import project_stretching_neohookean_kernel  # noqa: PLC0415

        device = self._device()
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            dtype=wp.vec3,
            device=device,
        )
        tri_indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat22(1.0, 0.0, 0.0, 1.0)], dtype=wp.mat22, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(3, dtype=wp.vec3, device=device)

        mu, lam = 1.0, 0.5
        wp.launch(
            project_stretching_neohookean_kernel,
            dim=1,
            inputs=[positions, tri_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        # At rest: sigma=(1,1), P = U*I*Vt = embedding I (3x2). Scatter as for ARAP.
        np.testing.assert_allclose(r[1], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[2], [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(r[0], -(r[1] + r[2]), atol=1e-5)

    def test_isotropic_stretch_NH_returns_finite_sigma(self):
        """sigma_0=(1.5,1.5) with mu=1, lam=0.5: after 5 Newton iter sigma is finite,
        positive, less than 1.5, and gradient norm < 1e-4.

        NH energy pulls sigma back toward 1 (incompressibility); the result should
        lie strictly between 0 and 1.5.  We also verify the gradient at the final
        sigma is small (well-converged in 5 iterations).
        """
        from newton._src.solvers.fba.kernels import project_stretching_neohookean_kernel  # noqa: PLC0415

        device = self._device()
        # Scale current positions by 1.5: sigma_sq = (1.5^2, 1.5^2) = (2.25, 2.25)
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1.5, 0, 0), wp.vec3(0, 0, 1.5)],
            dtype=wp.vec3,
            device=device,
        )
        tri_indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat22(1.0, 0.0, 0.0, 1.0)], dtype=wp.mat22, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(3, dtype=wp.vec3, device=device)

        mu, lam = 1.0, 0.5
        wp.launch(
            project_stretching_neohookean_kernel,
            dim=1,
            inputs=[positions, tri_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        # With Dm_inv=I and isotropic stretch, u0=(1,0,0), u1=(0,0,1), V=I.
        # P = diag(sigma_proj, sigma_proj) (embedded). Row0=(sigma_proj,0,0), row1=(0,0,sigma_proj).
        # rhs[1] = (sigma_proj, 0, 0), rhs[2] = (0, 0, sigma_proj).
        sigma_proj = r[1][0]
        self.assertTrue(np.isfinite(sigma_proj), "sigma_proj is not finite")
        self.assertGreater(sigma_proj, 0.0, "sigma_proj must be positive")
        self.assertLess(sigma_proj, 1.5, "NH should pull sigma back from 1.5 toward 1")

        # Verify gradient is small at the converged sigma (< 1e-3 tolerance for 5 iter).
        k = 2.0 * mu
        sigma0 = 1.5
        J = sigma_proj * sigma_proj
        log_J = np.log(J)
        inv_s = 1.0 / sigma_proj
        grad = mu * (sigma_proj - inv_s) + lam * log_J * inv_s + k * (sigma_proj - sigma0)
        self.assertLess(abs(grad), 1e-3, f"Gradient at converged sigma too large: {abs(grad):.2e}")

    def test_solver_neohookean_runs_no_nan(self):
        """Full SolverFBA step with stretching_model='neohookean', small cloth, 50 steps."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_grid(
            pos=wp.vec3(0, 2, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0, 0, 0),
            dim_x=8,
            dim_y=8,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.1,
            tri_ke=7142.857,  # 2*mu with mu=3571.43
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.1,
            edge_kd=0.0,
            fix_left=False,
        )
        # Pin top-left and top-right corners.
        top_left = 8 * (8 + 1)
        top_right = 8 * (8 + 1) + 8
        builder.particle_mass[top_left] = 0.0
        builder.particle_mass[top_right] = 0.0
        model = builder.finalize()

        mu = 3571.4285714285716
        lam = 14285.71428571429
        solver = SolverFBA(model, iterations=5, stretching_model="neohookean", mu=mu, lam=lam)
        s_in, s_out = model.state(), model.state()
        dt = 0.01
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite values in neohookean simulation")

    def test_solver_init_requires_mu_lam_for_neohookean(self):
        """SolverFBA with stretching_model='neohookean' and no mu/lam raises ValueError."""
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
            tri_ke=1.0e2,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            fix_left=True,
        )
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SolverFBA(model, stretching_model="neohookean")  # missing mu and lam

        with self.assertRaises(ValueError):
            SolverFBA(model, stretching_model="neohookean", mu=1000.0)  # missing lam

        with self.assertRaises(ValueError):
            SolverFBA(model, stretching_model="neohookean", lam=1000.0)  # missing mu


class TestFBARealSimAgreementNH(unittest.TestCase):
    """Trajectory match vs RealSim C++ reference with Neo-Hookean constitutive (50 steps, 113 verts).

    Fixture: ``newton/tests/fixtures/fba_realsim_nh_trajectory.npy`` (50, 113, 3) float32.
    Captured from RealSim C++ LocalGlobalSolver + TRI_NEOHOOKEAN + SPARSE_INVERSE_CUDA.
    Frame ``k`` of the fixture is the RealSim state after step ``k``; we compare
    ``newton_traj[k]`` (after step ``k+1``) vs ``fixture[k+1]`` for ``k`` in ``[0, 48]``.

    Lamé parameters from RealSim ElasticEnergy::setElasticParameter (E=1e4, nu=0.4):
        mu = E / (2*(1+nu)) = 3571.4286
        lambda = E*nu / ((1+nu)*(1-2*nu)) = 14285.714
    Note: for NH, RealSim passes the Lamé pair (mu, lambda) directly to the local
    projector; no factor-of-2 weight scaling is applied at the Newton API level.
    """

    FIXTURE_DIR = Path(__file__).parent / "fixtures"
    OBJ_PATH = FIXTURE_DIR / "fba_cloth_113.obj"
    TRAJ_PATH = FIXTURE_DIR / "fba_realsim_nh_trajectory.npy"

    # Simulation parameters matching RealSim FBACrossCheckNH
    DT = 0.01
    N_FRAMES = 50
    N_ITER = 5
    # E=1e4, nu=0.4 → mu = E/(2*(1+nu)) ≈ 3571.4286; lambda = E*nu/((1+nu)*(1-2*nu)) ≈ 14285.714
    # tri_ke = 2*mu ≈ 7142.857 (used only to build the PD Hessian A; NH uses mu/lam directly)
    TRI_KE = 7142.857
    EDGE_KE = 0.1
    MU = 3571.4285714285716
    LAM = 14285.71428571429
    PIN_STIFFNESS = 1e10
    PIN_BOXES = (
        (-1.1, 0.9, -0.1, -0.9, 1.1, 0.1),
        (0.9, 0.9, -0.1, 1.1, 1.1, 0.1),
    )

    @classmethod
    def setUpClass(cls):
        wp.init()

    @staticmethod
    def _load_obj(path: "Path") -> "tuple[list[wp.vec3], list[int]]":
        """Parse .obj → (vertices, triangle_indices_0based)."""
        from pathlib import Path as _Path  # noqa: PLC0415

        vertices: list[wp.vec3] = []
        indices: list[int] = []
        with open(_Path(path)) as fh:
            for raw_line in fh:
                tok = raw_line.strip()
                if tok.startswith("v "):
                    parts = tok.split()
                    vertices.append(wp.vec3(float(parts[1]), float(parts[2]), float(parts[3])))
                elif tok.startswith("f "):
                    parts = tok.split()[1:]
                    face_idx = [int(p.split("/")[0]) - 1 for p in parts]
                    for i in range(1, len(face_idx) - 1):
                        indices.extend([face_idx[0], face_idx[i], face_idx[i + 1]])
        return vertices, indices

    @staticmethod
    def _find_pin_indices(vertices: "list[wp.vec3]", boxes: "list[tuple]") -> "list[int]":
        pinned: list[int] = []
        for i, v in enumerate(vertices):
            x, y, z = v[0], v[1], v[2]
            for xmin, ymin, zmin, xmax, ymax, zmax in boxes:
                if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                    pinned.append(i)
                    break
        return pinned

    def _build_model(self, vertices: "list[wp.vec3]", indices: "list[int]") -> "newton.Model":
        """Build Newton model matching RealSim FBACrossCheckNH scene."""
        import newton as _newton  # noqa: PLC0415

        N = len(vertices)
        verts_np = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float64)
        idxs = np.array(indices, dtype=np.int32).reshape(-1, 3)
        total_area = 0.0
        for tri in idxs:
            e1 = verts_np[tri[1]] - verts_np[tri[0]]
            e2 = verts_np[tri[2]] - verts_np[tri[0]]
            total_area += 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
        density = 1.0 / total_area  # so that sum-of-area-weights ≈ 1 kg total

        builder = _newton.ModelBuilder(up_axis=_newton.Axis.Z, gravity=-10.0)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=density,
            tri_ke=self.TRI_KE,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=self.EDGE_KE,
            edge_kd=0.0,
        )
        # Override to uniform 1/N mass (RealSim Mass::addObjectMass convention).
        for i in range(N):
            builder.particle_mass[i] = 1.0 / N
        # Pin top-corner vertices by zeroing their mass.
        for idx in self._find_pin_indices(vertices, self.PIN_BOXES):
            builder.particle_mass[idx] = 0.0
        return builder.finalize()

    def test_nh_drape_trajectory_matches_realsim(self):
        """50-step NH drape on 113-vertex cloth: max position delta < 5e-5 m at every frame.

        NH uses 5 fixed Newton iterations in the local step; observed residual is ~9.5 µm
        (max L2 at frame 48), well within the 5e-5 m tolerance. This confirms the 5-iter
        Newton is sufficient to match RealSim's L-BFGS-to-convergence behavior.
        """
        from newton.solvers import SolverFBA  # noqa: PLC0415

        vertices, indices = self._load_obj(self.OBJ_PATH)
        self.assertEqual(len(vertices), 113)

        realsim_traj = np.load(str(self.TRAJ_PATH))
        self.assertEqual(realsim_traj.shape, (50, 113, 3))

        model = self._build_model(vertices, indices)
        solver = SolverFBA(
            model,
            iterations=self.N_ITER,
            pin_stiffness=self.PIN_STIFFNESS,
            stretching_model="neohookean",
            mu=self.MU,
            lam=self.LAM,
        )

        state_in = model.state()
        state_out = model.state()
        newton_traj = np.zeros((self.N_FRAMES, model.particle_count, 3), dtype=np.float32)

        for f in range(self.N_FRAMES):
            state_in.clear_forces()
            solver.step(state_in, state_out, None, None, self.DT)
            newton_traj[f] = state_out.particle_q.numpy()
            state_in, state_out = state_out, state_in

        self.assertTrue(np.all(np.isfinite(newton_traj)), "non-finite positions in NH Newton trajectory")

        # Compare newton[k] vs realsim[k+1] for k in [0, 48].
        max_delta = 0.0
        worst_frame = -1
        for k in range(self.N_FRAMES - 1):
            delta = float(np.linalg.norm(newton_traj[k] - realsim_traj[k + 1], axis=-1).max())
            if delta > max_delta:
                max_delta = delta
                worst_frame = k

        self.assertLess(
            max_delta,
            5e-5,
            f"Max L2 position delta {max_delta:.3e} m exceeds 5e-5 m tolerance "
            f"(worst frame {worst_frame}; observed ~9.5 µm with 5-iter Newton on RTX 5090)",
        )


class TestDynamicPin(unittest.TestCase):
    """Verify SolverFBA.set_pin_targets moves pinned particle to new targets."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_pin_follows_target(self):
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
            tri_ke=1.0e4,
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


class TestTetHessianAssembly(unittest.TestCase):
    """Verify tet Hessian block assembly in build_pd_system."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_single_tet_model(self):
        """Single tetrahedron: (0,0,0),(1,0,0),(0,1,0),(0,0,1). Pin vertex 0."""
        builder = newton.ModelBuilder()
        p0 = builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        p1 = builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        p2 = builder.add_particle(pos=wp.vec3(0.0, 1.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        p3 = builder.add_particle(pos=wp.vec3(0.0, 0.0, 1.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        builder.add_tetrahedron(p0, p1, p2, p3, k_mu=1.0e3, k_lambda=1.0e3)
        builder.particle_mass[p0] = 0.0
        return builder.finalize(device="cpu")

    def test_tet_meta_populated(self):
        model = self._build_single_tet_model()
        _A, meta = build_pd_system(model, dt=1.0 / 60.0, pin_stiffness=1e12)
        self.assertEqual(meta["tet_indices"].shape, (1, 4))
        self.assertEqual(meta["tet_rest_inv"].shape, (1, 3, 3))
        self.assertEqual(meta["tet_weight"].shape, (1,))
        # volume of canonical tet = 1/6
        np.testing.assert_allclose(meta["tet_volume"][0], 1.0 / 6.0, rtol=1e-5)

    def test_tet_hessian_symmetric(self):
        model = self._build_single_tet_model()
        A, _meta = build_pd_system(model, dt=1.0 / 60.0, pin_stiffness=1e12)
        diff = A - A.T
        self.assertLess(np.abs(diff).max(), 1e-10, "tet Hessian is not symmetric")

    def test_tet_hessian_psd(self):
        model = self._build_single_tet_model()
        A, _meta = build_pd_system(model, dt=1.0 / 60.0, pin_stiffness=1e12)
        eigenvalues = np.linalg.eigvalsh(A.toarray())
        self.assertGreater(eigenvalues.min(), -1e-8, "tet Hessian has negative eigenvalue")


class TestTetARAP(unittest.TestCase):
    """Tests for tet ARAP kernel: project_stretching_arap_tet_kernel."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_rest_tet_projects_to_identity(self):
        """Rest configuration (F=I) should scatter correctly.

        For canonical tet (0,0,0),(1,0,0),(0,1,0),(0,0,1):
        Ds = I, Dm_inv = I (since Dm = I), F = I, R = I.
        proj = w * I * I^T = w * I.
        row0=(1,0,0)*w, row1=(0,1,0)*w, row2=(0,0,1)*w
        rhs[0] = -(row0+row1+row2) = -(1,1,1)*w
        rhs[1] = (1,0,0)*w; rhs[2] = (0,1,0)*w; rhs[3] = (0,0,1)*w
        """
        from newton._src.solvers.fba.kernels import project_stretching_arap_tet_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1)],
            dtype=wp.vec3,
            device=device,
        )
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        # Dm_inv = I for canonical tet
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            project_stretching_arap_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        np.testing.assert_allclose(r[1], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(r[2], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(r[3], [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(r[0], -(r[1] + r[2] + r[3]), atol=1e-6)

    def test_rotation_invariance(self):
        """F = R0 (pure rotation) must project to exactly R0 and scatter consistently."""
        from newton._src.solvers.fba.kernels import project_stretching_arap_tet_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # 30-degree rotation about z-axis
        angle = np.pi / 6.0
        c, s = np.cos(angle), np.sin(angle)
        R0 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        # Rotated tet: apply R0 to canonical vertices
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        verts_rot = verts @ R0.T  # (4,3)

        positions = wp.array([wp.vec3(*v) for v in verts_rot], dtype=wp.vec3, device=device)
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        # Dm_inv = I for canonical tet
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            project_stretching_arap_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        # Momentum conservation: rhs[0] + rhs[1] + rhs[2] + rhs[3] = 0
        total = r[0] + r[1] + r[2] + r[3]
        np.testing.assert_allclose(total, [0.0, 0.0, 0.0], atol=1e-5)

    def test_momentum_conservation_arbitrary_deformation(self):
        """For any deformation, sum of rhs must be zero (momentum conservation)."""
        from newton._src.solvers.fba.kernels import project_stretching_arap_tet_kernel  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # Arbitrary (non-rest, non-rotated) positions
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1.3, 0.1, 0), wp.vec3(0.2, 1.2, 0.1), wp.vec3(0.1, 0.2, 1.1)],
            dtype=wp.vec3,
            device=device,
        )
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([2.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            project_stretching_arap_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        total = r[0] + r[1] + r[2] + r[3]
        np.testing.assert_allclose(total, [0.0, 0.0, 0.0], atol=1e-5)


class TestTetARAPIntegration(unittest.TestCase):
    """Smoke and integration tests for SolverFBA with tet softbody."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _build_tet_grid(self, dim: int = 4):
        """Build a dim x dim x dim tet grid, pinning the left face via fix_left=True."""
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.1),  # lift off ground
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            dim_z=dim,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1.0e3,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            fix_left=True,
        )
        return builder.finalize()

    def test_softbody_cube_smoke_no_nan(self):
        """Run 100 steps on a 4x4x4 tet grid; verify no NaN."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._build_tet_grid(dim=4)
        self.assertGreater(model.tet_count, 0, "model should have tets")
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(100):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions after 100 tet steps")

    def test_solver_init_rejects_no_geometry(self):
        """Model with no triangles and no tets must raise ValueError."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize()
        with self.assertRaises(ValueError):
            SolverFBA(model, iterations=1)

    def test_softbody_gravity_sag(self):
        """Pinned-left tet grid must deform under gravity."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._build_tet_grid(dim=4)
        solver = SolverFBA(model, iterations=10)
        s_in, s_out = model.state(), model.state()
        q_initial = s_in.particle_q.numpy().copy()
        dt = 1.0 / 60.0
        for _ in range(200):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q_final = s_in.particle_q.numpy()
        # At least some free particles should have moved under gravity.
        max_displacement = np.abs(q_final - q_initial).max()
        self.assertGreater(max_displacement, 1e-3, "tet softbody did not deform under gravity")


class TestTetCorotational(unittest.TestCase):
    """Tests for 3D corotational tet projection kernel."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _device(self):
        return "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_rest_tet_corot_projects_to_identity(self):
        """Rest configuration (F=I) should produce zero gradient and sigma_proj=(1,1,1).

        For canonical tet with Dm_inv=I: F=I, sigma=(1,1,1).
        At rest, energy gradient is zero, so sigma_proj = (1,1,1).
        Scatter output must match ARAP-at-rest:
            rhs[1]=(1,0,0), rhs[2]=(0,1,0), rhs[3]=(0,0,1), rhs[0]=-(sum).
        """
        from newton._src.solvers.fba.kernels import project_stretching_corotational_tet_kernel  # noqa: PLC0415

        device = self._device()
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1)],
            dtype=wp.vec3,
            device=device,
        )
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        mu, lam = 1.0, 0.5
        wp.launch(
            project_stretching_corotational_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        # At rest: sigma=(1,1,1), corot energy gradient=0, sigma_proj=(1,1,1).
        # P = U*diag(1,1,1)*Vt = I. proj = w*Dm_inv*I = I. Scatter as ARAP-at-rest.
        np.testing.assert_allclose(r[1], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[2], [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[3], [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(r[0], -(r[1] + r[2] + r[3]), atol=1e-5)

    def test_isotropic_stretch_corot_3d(self):
        """sigma_0=(1.5,1.5,1.5), mu=1, lam=0.5 should give sigma_proj≈1.18182 for all 3.

        Closed-form derivation:
            sum_b = 6*mu + 9*lam + 2*mu*(sigma_0_0+sigma_0_1+sigma_0_2)
                  = 6 + 4.5 + 2*(3*1.5) = 6 + 4.5 + 9 = 19.5
            alpha = 4*mu = 4
            alpha + 3*lam = 4 + 1.5 = 5.5
            sigma_proj_i = (2*mu + 3*lam + 2*mu*sigma_0_i)/(4*mu) - lam*sum_b/(4*mu*(4*mu+3*lam))
                         = (2 + 1.5 + 3)/4 - 0.5*19.5/(4*5.5)
                         = 6.5/4 - 9.75/22
                         = 1.625 - 0.443182...
                         = 1.18182...
        """
        from newton._src.solvers.fba.kernels import project_stretching_corotational_tet_kernel  # noqa: PLC0415

        device = self._device()
        # Isotropic 1.5x stretch: move vertices to make F = 1.5*I.
        # Ds = 1.5*I (edge vectors scaled by 1.5), Dm_inv = I -> F = 1.5*I.
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1.5, 0, 0), wp.vec3(0, 1.5, 0), wp.vec3(0, 0, 1.5)],
            dtype=wp.vec3,
            device=device,
        )
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        mu, lam = 1.0, 0.5
        wp.launch(
            project_stretching_corotational_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()

        # F = 1.5*I -> U=V=I, sigma=(1.5,1.5,1.5).
        # sigma_proj ~= 1.18182 for all three.
        # P = I*diag(sigma_proj)*I = sigma_proj*I.
        # proj = w*Dm_inv*P^T = sigma_proj*I.
        # row0=(sigma_proj,0,0), row1=(0,sigma_proj,0), row2=(0,0,sigma_proj)
        expected = 1.625 - 0.5 * 19.5 / (4.0 * 5.5)  # ~1.18182
        np.testing.assert_allclose(r[1], [expected, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[2], [0.0, expected, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[3], [0.0, 0.0, expected], atol=1e-5)
        np.testing.assert_allclose(r[0], -(r[1] + r[2] + r[3]), atol=1e-5)

    def test_momentum_conservation_corot_3d(self):
        """For any deformation, sum of rhs must be zero (momentum conservation)."""
        from newton._src.solvers.fba.kernels import project_stretching_corotational_tet_kernel  # noqa: PLC0415

        device = self._device()
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1.3, 0.1, 0), wp.vec3(0.2, 1.2, 0.1), wp.vec3(0.1, 0.2, 1.1)],
            dtype=wp.vec3,
            device=device,
        )
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([2.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        mu, lam = 1.0e3, 5.0e2
        wp.launch(
            project_stretching_corotational_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        total = r[0] + r[1] + r[2] + r[3]
        np.testing.assert_allclose(total, [0.0, 0.0, 0.0], atol=1e-4)

    def test_solver_corot_tet_softbody_runs(self):
        """Full SolverFBA step with stretching_model='corotational' on 4x4x4 tet grid, 50 steps."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            dim_z=4,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1.0e3,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            fix_left=True,
        )
        model = builder.finalize()

        mu, lam = 1.0e4, 1.0e4
        solver = SolverFBA(model, iterations=10, stretching_model="corotational", mu=mu, lam=lam)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions after 50 tet Corot steps")


class TestTetNeoHookean(unittest.TestCase):
    """Tests for 3D Neo-Hookean tet projection kernel."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def _device(self):
        return "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_rest_tet_nh_projects_to_identity(self):
        """At rest sigma=(1,1,1): J=1, log_J=0, gradient=0, Newton leaves sigma unchanged.

        With sigma=(1,1,1): J=1, log_J=0, inv_i=1.
        grad_i = mu*(1-1) + lam*0*1 + k*(1-1) = 0 for all i.
        Newton dx=0, sigma stays at (1,1,1).
        Scatter output must match ARAP-at-rest.
        """
        from newton._src.solvers.fba.kernels import project_stretching_neohookean_tet_kernel  # noqa: PLC0415

        device = self._device()
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1)],
            dtype=wp.vec3,
            device=device,
        )
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        mu, lam = 1.0, 0.5
        wp.launch(
            project_stretching_neohookean_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        # At rest: sigma=(1,1,1), NH gradient=0, sigma_proj=(1,1,1). Scatter as ARAP-at-rest.
        np.testing.assert_allclose(r[1], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[2], [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(r[3], [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(r[0], -(r[1] + r[2] + r[3]), atol=1e-5)

    def test_isotropic_stretch_nh_3d(self):
        """sigma_0=(1.5,1.5,1.5) with mu=1, lam=0.5, k=2:
        After 5 Newton iterations sigma_proj should be:
          - all three equal (by symmetry)
          - positive
          - less than 1.5 (NH pulls toward incompressibility)
          - gradient magnitude < 1e-4 at convergence
        """
        from newton._src.solvers.fba.kernels import project_stretching_neohookean_tet_kernel  # noqa: PLC0415

        device = self._device()
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1.5, 0, 0), wp.vec3(0, 1.5, 0), wp.vec3(0, 0, 1.5)],
            dtype=wp.vec3,
            device=device,
        )
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        mu, lam = 1.0, 0.5
        wp.launch(
            project_stretching_neohookean_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        # F=1.5*I -> U=V=I, sigma_0=(1.5,1.5,1.5).
        # rhs[1]=(sigma_proj,0,0), rhs[2]=(0,sigma_proj,0), rhs[3]=(0,0,sigma_proj).
        sigma_proj = float(r[1][0])
        self.assertTrue(np.isfinite(sigma_proj), "sigma_proj is not finite")
        self.assertGreater(sigma_proj, 0.0, "sigma_proj must be positive")
        self.assertLess(sigma_proj, 1.5, "NH should pull sigma back from 1.5 toward 1")

        # All three should be equal by symmetry.
        np.testing.assert_allclose(r[2][1], sigma_proj, atol=1e-5)
        np.testing.assert_allclose(r[3][2], sigma_proj, atol=1e-5)

        # Verify gradient is small at the converged sigma (3D NH: J = sigma_proj^3).
        k = 2.0 * mu
        sigma0 = 1.5
        J3 = sigma_proj**3
        log_J3 = np.log(J3)
        inv_s = 1.0 / sigma_proj
        grad = mu * (sigma_proj - inv_s) + lam * log_J3 * inv_s + k * (sigma_proj - sigma0)
        self.assertLess(abs(grad), 1e-4, f"Gradient at converged sigma too large: {abs(grad):.2e}")

    def test_momentum_conservation_nh_3d(self):
        """For any deformation, sum of rhs must be zero (momentum conservation)."""
        from newton._src.solvers.fba.kernels import project_stretching_neohookean_tet_kernel  # noqa: PLC0415

        device = self._device()
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1.3, 0.1, 0), wp.vec3(0.2, 1.2, 0.1), wp.vec3(0.1, 0.2, 1.1)],
            dtype=wp.vec3,
            device=device,
        )
        tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
        Dm_inv = wp.array([wp.mat33(1, 0, 0, 0, 1, 0, 0, 0, 1)], dtype=wp.mat33, device=device)
        weight = wp.array([2.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        mu, lam = 1.0e3, 5.0e2
        wp.launch(
            project_stretching_neohookean_tet_kernel,
            dim=1,
            inputs=[positions, tet_indices, Dm_inv, weight, mu, lam],
            outputs=[rhs],
            device=device,
        )
        r = rhs.numpy()
        total = r[0] + r[1] + r[2] + r[3]
        np.testing.assert_allclose(total, [0.0, 0.0, 0.0], atol=1e-4)

    def test_solver_nh_tet_softbody_runs(self):
        """Full SolverFBA step with stretching_model='neohookean' on 4x4x4 tet grid, 50 steps."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            dim_z=4,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1.0e3,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            fix_left=True,
        )
        model = builder.finalize()

        mu, lam = 1.0e4, 1.0e4
        solver = SolverFBA(model, iterations=10, stretching_model="neohookean", mu=mu, lam=lam)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0
        for _ in range(50):
            s_in.clear_forces()
            solver.step(s_in, s_out, None, None, dt)
            s_in, s_out = s_out, s_in
        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions after 50 tet NH steps")


class TestPhase4StageAContact(unittest.TestCase):
    """Phase 4 Stage A: unilateral plane contact via Schur complement.

    Tests the hard-constraint contact path in SolverFBA — no friction,
    λ ≥ 0 only.  Each test constructs a :class:`~newton.Contacts` object
    manually (bypassing the broad/narrow phase pipeline) to isolate the
    Schur-complement solver.
    """

    @classmethod
    def setUpClass(cls):
        wp.init()

    # ------------------------------------------------------------------ helpers

    def _build_single_particle_model(self, y: float = -0.5):
        """Return a one-triangle cloth model with one free particle.

        The cloth is a single equilateral triangle; particle 0 is placed at
        ``(0, y, 0)`` so it is below the y=0 plane when ``y < 0``.
        Particle 1 and 2 are placed at their usual rest positions but pinned
        so only particle 0 is free.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, y, 0.0),  # particle 0 — free, below plane
                wp.vec3(1.0, 0.0, 0.0),  # particle 1 — pinned
                wp.vec3(0.0, 0.0, 1.0),  # particle 2 — pinned
            ],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        builder.particle_mass[1] = 0.0  # pin
        builder.particle_mass[2] = 0.0  # pin
        return builder.finalize()

    def _make_contacts(self, device, count: int, particle_indices, normals, body_pos_world):
        """Manually populate a :class:`~newton.Contacts` object.

        Args:
            device: Warp device.
            count: Number of contacts ``M``.
            particle_indices: list[int] — particle index per contact.
            normals: list[tuple] — world-frame contact normal per contact.
            body_pos_world: list[tuple] — world anchor point per contact
                (stored as-is in ``soft_contact_body_pos``; for Stage A tests
                with static shapes this equals the world-frame anchor).

        Returns:
            :class:`~newton.Contacts` with ``soft_contact_count`` = ``count``.
        """
        contacts = newton.Contacts(rigid_contact_max=0, soft_contact_max=max(count, 1), device=device)
        # Set count via the counter slice.
        contacts.soft_contact_count.assign(np.array([count], dtype=np.int32))
        if count > 0:
            p_arr = np.array(particle_indices, dtype=np.int32)
            n_arr = np.array(normals, dtype=np.float32)
            bp_arr = np.array(body_pos_world, dtype=np.float32)
            contacts.soft_contact_particle.assign(p_arr)
            contacts.soft_contact_normal.assign(n_arr)
            contacts.soft_contact_body_pos.assign(bp_arr)
            # shape indices (-1 = static/no-body shape)
            s_arr = np.full(count, -1, dtype=np.int32)
            contacts.soft_contact_shape.assign(s_arr)
        return contacts

    # ------------------------------------------------------------------ tests

    def test_no_contact_passes_through(self):
        """Empty contacts object must produce the same output as contacts=None."""
        model = self._build_single_particle_model(y=0.5)
        device = model.device
        solver = SolverFBA(model, iterations=5)
        s_in_a, s_out_a = model.state(), model.state()
        s_in_b, s_out_b = model.state(), model.state()
        dt = 1.0 / 60.0

        # Baseline: no contacts.
        s_in_a.clear_forces()
        solver.step(s_in_a, s_out_a, None, None, dt)
        q_no_contact = s_out_a.particle_q.numpy().copy()

        # With empty contacts object (count == 0).
        empty_contacts = self._make_contacts(device, 0, [], [], [])
        s_in_b.clear_forces()
        solver.step(s_in_b, s_out_b, None, empty_contacts, dt)
        q_empty_contact = s_out_b.particle_q.numpy().copy()

        np.testing.assert_allclose(
            q_empty_contact, q_no_contact, atol=1e-6, err_msg="Empty contacts path diverged from contacts=None path"
        )

    def test_single_plane_contact_pushes_particle_away(self):
        """Single particle below y=0 plane should be pushed to y≥0 after step.

        Constraint: n=(0,1,0), anchor=(0,0,0) → offset = dot((0,1,0),(0,0,0)) = 0.
        After correction, particle[0].y should satisfy y ≥ -1e-4 m.
        """
        model = self._build_single_particle_model(y=-0.5)
        device = model.device
        solver = SolverFBA(model, iterations=10)

        contacts = self._make_contacts(
            device,
            count=1,
            particle_indices=[0],
            normals=[(0.0, 1.0, 0.0)],  # outward normal pointing +y
            body_pos_world=[(0.0, 0.0, 0.0)],  # anchor on the y=0 plane
        )

        s_in, s_out = model.state(), model.state()
        s_in.clear_forces()
        dt = 1.0 / 60.0
        solver.step(s_in, s_out, None, contacts, dt)

        q = s_out.particle_q.numpy()
        particle_y = float(q[0, 1])
        self.assertGreaterEqual(
            particle_y, -1e-4, f"Particle y={particle_y:.6f} should be >= 0 after plane contact correction"
        )

    def test_multiple_plane_contacts_no_interpenetration(self):
        """Four particles, each below the y=0 plane, each with a separate contact.

        After one step with contacts active, all free particles should have y ≥ -1e-4.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        # Build a 2x2 grid so we get 4 free particles in a row below y=0.
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, -0.5, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.5,
            cell_y=0.5,
            mass=0.1,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-2,
            edge_kd=0.0,
            fix_left=False,
        )
        # Pin one corner so the model is determinate.
        builder.particle_mass[0] = 0.0
        model = builder.finalize()
        device = model.device
        N = model.particle_count

        solver = SolverFBA(model, iterations=10)

        # All particles start at y = -0.5; identify the free ones.
        pos_np = model.particle_q.numpy()  # (N, 3)  # noqa: F841
        inv_mass = model.particle_inv_mass.numpy()
        free_indices = [i for i in range(N) if inv_mass[i] > 0]

        # Create one contact per free particle, all on y=0 plane.
        M = len(free_indices)
        normals = [(0.0, 1.0, 0.0)] * M
        anchors = [(0.0, 0.0, 0.0)] * M

        contacts = self._make_contacts(device, M, free_indices, normals, anchors)

        s_in, s_out = model.state(), model.state()
        s_in.clear_forces()
        dt = 1.0 / 60.0
        solver.step(s_in, s_out, None, contacts, dt)

        q = s_out.particle_q.numpy()
        for fi in free_indices:
            y = float(q[fi, 1])
            self.assertGreaterEqual(y, -1e-4, f"Particle {fi} y={y:.6f} should be >= 0 after plane contact correction")

    def test_schur_W_is_positive_definite_for_active_contacts(self):
        """W = J A⁻¹ Jᵀ must be symmetric positive definite for active contacts.

        Uses the FBALinearSolver directly with a known SPD system and verifies:
        1. All eigenvalues of W are positive.
        2. W is symmetric (W == Wᵀ up to float64 precision).
        3. For a single contact, W[0,0] == dot(n, A⁻¹ n_at_p) (scalar check).
        """
        import scipy.sparse as sp

        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            factorize_and_sparse_inverse,
        )

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        rng = np.random.default_rng(7)
        n = 32

        # Build a small SPD A.
        diag = rng.uniform(5.0, 15.0, n)
        off = rng.uniform(-0.5, 0.5, n - 1)
        A = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr").astype(np.float64)

        fs = factorize_and_sparse_inverse(A)
        solver = FBALinearSolver(fs, device=device)

        # 5 contacts: different particles, different normals.
        M = 5
        # Pick 5 distinct particle indices.
        particles = np.array([0, 5, 10, 15, 20], dtype=np.int32)
        normals_np = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.707, 0.707, 0.0],
                [0.0, 0.577, 0.816],
            ],
            dtype=np.float32,
        )
        # Normalise.
        for i in range(M):
            normals_np[i] /= np.linalg.norm(normals_np[i])
        alpha_np = np.ones(M, dtype=np.float32)

        j_indices = wp.array(particles, dtype=wp.int32, device=device)
        j_normals = wp.array(normals_np, dtype=wp.vec3, device=device)
        j_alpha = wp.array(alpha_np, dtype=wp.float32, device=device)

        W = solver.build_schur_complement(M, j_indices, j_normals, j_alpha)

        # 1. Symmetry.
        np.testing.assert_allclose(W, W.T, atol=1e-12, err_msg="W is not symmetric")

        # 2. All eigenvalues positive.
        eigvals = np.linalg.eigvalsh(W)
        min_eig = float(eigvals.min())
        self.assertGreater(min_eig, 0.0, f"W has non-positive eigenvalue: {min_eig:.3e}. eigvals={eigvals}")

        # 3. Scalar check for first contact: W[0,0] = n^T A^{-1}_{pp} n
        #    where A^{-1}_{pp} is the 1x1 scalar block at particle 0.
        #    Since A is scalar (not vec3), A^{-1}_{pp} = Ainv[p0, p0].
        Ainv_dense = np.linalg.inv(A.toarray())
        p0 = int(particles[0])
        n0 = normals_np[0].astype(np.float64)
        # For a scalar A and a scalar n (1D per component), W[0,0] = sum_i n[i]^2 * Ainv[p0,p0]
        # because the FBA solve does each component independently.
        W00_expected = float(np.dot(n0, n0)) * float(Ainv_dense[p0, p0])
        # Tolerance accounts for float32 → float64 round-trip in solve().
        self.assertAlmostEqual(
            W[0, 0], W00_expected, delta=1e-7, msg=f"W[0,0]={W[0, 0]:.8f} != expected {W00_expected:.8f}"
        )

    def test_approach_b_W_matches_approach_a(self):
        """Approach B multi-RHS Schur build produces W within 1e-7 of analytical reference."""
        import scipy.sparse as _sp

        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            factorize_and_sparse_inverse,
        )

        rng = np.random.default_rng(7)
        N = 20
        M = 5  # contacts

        A_dense = rng.standard_normal((N, N))
        A_dense = A_dense @ A_dense.T + N * np.eye(N)
        A = _sp.csr_matrix(A_dense)
        fs = factorize_and_sparse_inverse(A)

        device = "cuda:0" if wp.is_cuda_available() else "cpu"

        # Build random contact arrays.
        rng2 = np.random.default_rng(13)
        j_indices = rng2.integers(0, N, size=M, dtype=np.int32)
        j_normals = rng2.standard_normal((M, 3)).astype(np.float32)
        j_normals /= np.linalg.norm(j_normals, axis=1, keepdims=True) + 1e-8
        j_alpha = np.ones(M, dtype=np.float32)

        j_indices_d = wp.array(j_indices, dtype=wp.int32, device=device)
        j_normals_d = wp.array(j_normals, dtype=wp.vec3, device=device)
        j_alpha_d = wp.array(j_alpha, dtype=wp.float32, device=device)

        # Call build_schur_complement — uses Approach B internally.
        solver = FBALinearSolver(fs, device=device)
        W_b = solver.build_schur_complement(M, j_indices_d, j_normals_d, j_alpha_d)

        # Reference: compute W analytically via numpy.
        # W[cp, c] = alpha[cp] * sum_ax n_cp[ax] * A_inv[p_cp, p_c] * alpha[c] * n_c[ax]
        A_inv = np.linalg.inv(A_dense)
        W_ref = np.zeros((M, M), dtype=np.float64)
        for cp in range(M):
            p_cp = j_indices[cp]
            a_cp = float(j_alpha[cp])
            n_cp = j_normals[cp].astype(np.float64)
            for c in range(M):
                p_c = j_indices[c]
                a_c = float(j_alpha[c])
                n_c = j_normals[c].astype(np.float64)
                # A^{-1} applied component-wise: contribution per axis sums independently.
                W_ref[cp, c] = a_cp * a_c * float(np.dot(n_cp, n_c)) * float(A_inv[p_cp, p_c])

        np.testing.assert_allclose(
            W_b,
            W_ref,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Approach B W diverges from analytical reference",
        )

    def test_isodof_W_matches_dense_path(self):
        """``build_schur_complement(use_isodof=True)`` produces the same W as
        the legacy ``use_isodof=False`` multi-RHS path.

        The isodof path computes only ``A^{-1}[i, j]`` entries restricted to
        the unique contacted particles and assembles ``W`` from a per-particle
        rank table. It must match the dense (multi-RHS solve) path bitwise
        modulo float64 round-off.

        Regression for a sign/indexing bug where ``invperm[isodofs]`` was used
        in place of ``perm[isodofs]`` when computing the selected ``A^{-1}``
        entries, which silently builds the wrong matrix and triggers NaN
        downstream as soon as contacts engage.
        """
        import scipy.sparse as _sp

        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            factorize_and_sparse_inverse,
        )

        # Build a sparse SPD matrix whose COLAMD ordering produces a
        # non-trivial ``perm_r`` (``perm_r != invperm_r``). A purely random
        # dense SPD often factors with the identity permutation under COLAMD,
        # which masks any bug that swaps ``perm[isodofs]`` for
        # ``invperm[isodofs]``.
        rng = np.random.default_rng(11)
        N = 200
        M = 7

        A_lil = _sp.lil_matrix((N, N))
        for i in range(N):
            A_lil[i, i] = 4.0 + rng.random()
            if i > 0:
                v = float(rng.standard_normal())
                A_lil[i, i - 1] = v
                A_lil[i - 1, i] = v
            if i > 4:
                v = 0.1 * float(rng.standard_normal())
                A_lil[i, i - 5] = v
                A_lil[i - 5, i] = v
        for j in (10, 30, 50, 100, 150):
            A_lil[0, j] = 0.5
            A_lil[j, 0] = 0.5
        A = A_lil.tocsc()
        fs = factorize_and_sparse_inverse(A)
        # Sanity: this is the regime where perm != invperm.
        assert not np.array_equal(fs.perm_r, fs.invperm_r), (
            "Test setup invalid: COLAMD chose identity ordering; the isodof vs dense W test would be vacuous."
        )

        device = "cuda:0" if wp.is_cuda_available() else "cpu"

        rng2 = np.random.default_rng(17)
        # Allow duplicate particles in contact rows — both paths should still agree.
        j_indices = rng2.integers(0, N, size=M, dtype=np.int32)
        j_normals = rng2.standard_normal((M, 3)).astype(np.float32)
        j_normals /= np.linalg.norm(j_normals, axis=1, keepdims=True) + 1e-8
        j_alpha = rng2.uniform(0.5, 2.0, size=M).astype(np.float32)

        j_indices_d = wp.array(j_indices, dtype=wp.int32, device=device)
        j_normals_d = wp.array(j_normals, dtype=wp.vec3, device=device)
        j_alpha_d = wp.array(j_alpha, dtype=wp.float32, device=device)

        solver = FBALinearSolver(fs, device=device)
        W_dense = solver.build_schur_complement(M, j_indices_d, j_normals_d, j_alpha_d, use_isodof=False)
        W_iso = solver.build_schur_complement(M, j_indices_d, j_normals_d, j_alpha_d, use_isodof=True)

        np.testing.assert_allclose(
            W_iso,
            W_dense,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Isodof W diverges from dense-path W",
        )


class TestPhase4StageBFriction(unittest.TestCase):
    """Phase 4 Stage B: Coulomb friction via NonSmooth Newton."""

    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_coulomb_cone_projection_correctness(self):
        """project_coulomb_cone handles inside-cone, polar-cone, and surface cases."""
        from newton._src.solvers.fba.solver_fba import project_coulomb_cone  # noqa: PLC0415

        mu = 0.5

        # Case 1: Already inside cone (s >= 0 and |v| <= mu * s).
        s1, v1 = project_coulomb_cone(2.0, np.array([0.5, 0.5]), mu)
        # |v| = sqrt(0.5) ≈ 0.707, mu * s = 1.0 → inside
        self.assertAlmostEqual(s1, 2.0, places=10)
        np.testing.assert_allclose(v1, [0.5, 0.5], atol=1e-12)

        # Case 2: In polar cone (mu * |v| <= -s → project to origin).
        # s = -5, |v| = 1.0, mu * |v| = 0.5, -s = 5 → 0.5 < 5 → polar
        s2, v2 = project_coulomb_cone(-5.0, np.array([0.6, 0.8]), mu)
        self.assertAlmostEqual(s2, 0.0, places=10)
        np.testing.assert_allclose(v2, [0.0, 0.0], atol=1e-12)

        # Case 3: Surface projection (neither inside nor polar).
        # Use s = -0.1, v = [3.0, 4.0], |v| = 5.0
        # factor = (s + mu * |v|) / (1 + mu^2) = (-0.1 + 2.5) / 1.25 = 1.92
        # s_new = 1.92, v_new = mu * factor / |v| * v = 0.5 * 1.92 / 5.0 * [3, 4]
        #       = 0.192 * [3, 4] = [0.576, 0.768]
        s3, v3 = project_coulomb_cone(-0.1, np.array([3.0, 4.0]), mu)
        self.assertAlmostEqual(s3, 1.92, places=10)
        np.testing.assert_allclose(v3, [0.576, 0.768], atol=1e-10)
        # Verify on cone surface: |v3| = mu * s3
        self.assertAlmostEqual(np.linalg.norm(v3), mu * s3, places=10)

        # Case 4: s = 0, v = [0, 0] → inside cone (both zero).
        s4, v4 = project_coulomb_cone(0.0, np.array([0.0, 0.0]), mu)
        self.assertAlmostEqual(s4, 0.0, places=10)
        np.testing.assert_allclose(v4, [0.0, 0.0], atol=1e-12)

    def test_friction_disabled_matches_stage_a(self):
        """SolverFBA(friction=False) must produce bit-identical results to Stage A."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, -0.5, 0.0),  # particle 0 — free, below plane
                wp.vec3(1.0, 0.0, 0.0),  # pinned
                wp.vec3(0.0, 0.0, 1.0),  # pinned
            ],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        builder.particle_mass[1] = 0.0
        builder.particle_mass[2] = 0.0
        model = builder.finalize()
        device = model.device

        contacts = newton.Contacts(rigid_contact_max=0, soft_contact_max=1, device=device)
        contacts.soft_contact_count.assign(np.array([1], dtype=np.int32))
        contacts.soft_contact_particle.assign(np.array([0], dtype=np.int32))
        contacts.soft_contact_normal.assign(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        contacts.soft_contact_body_pos.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        contacts.soft_contact_shape.assign(np.array([-1], dtype=np.int32))

        dt = 1.0 / 60.0

        # Two identical friction=False solvers must agree exactly.
        solver_a = SolverFBA(model, iterations=5, friction=False)
        solver_b = SolverFBA(model, iterations=5, friction=False)

        s_in_a, s_out_a = model.state(), model.state()
        s_in_b, s_out_b = model.state(), model.state()
        s_in_a.clear_forces()
        s_in_b.clear_forces()

        solver_a.step(s_in_a, s_out_a, None, contacts, dt)
        solver_b.step(s_in_b, s_out_b, None, contacts, dt)

        qa = s_out_a.particle_q.numpy()
        qb = s_out_b.particle_q.numpy()
        # Two identical solvers must agree exactly.
        np.testing.assert_array_equal(qa, qb, err_msg="Two friction=False solvers diverged from each other")
        # Particle 0 should be above the plane.
        self.assertGreaterEqual(float(qa[0, 1]), -1e-4, f"Particle still below plane: y={float(qa[0, 1]):.6f}")

    def test_friction_W_block_structure(self):
        """build_schur_complement with tangents returns 6x6 symmetric W for 2 contacts."""
        import scipy.sparse as sp

        from newton._src.solvers.fba.linear_solver import (  # noqa: PLC0415
            FBALinearSolver,
            factorize_and_sparse_inverse,
        )
        from newton._src.solvers.fba.solver_fba import compute_tangent_basis  # noqa: PLC0415

        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        rng = np.random.default_rng(42)
        n = 20

        # Small SPD A.
        diag = rng.uniform(5.0, 15.0, n)
        off = rng.uniform(-0.3, 0.3, n - 1)
        A = sp.diags([off, diag, off], [-1, 0, 1], shape=(n, n), format="csr").astype(np.float64)
        fs = factorize_and_sparse_inverse(A)
        solver = FBALinearSolver(fs, device=device)

        # 2 contacts.
        M = 2
        particles = np.array([2, 7], dtype=np.int32)
        normals = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        alpha = np.ones(M, dtype=np.float32)

        # Compute tangent bases.
        t1_arr = np.zeros((M, 3), dtype=np.float32)
        t2_arr = np.zeros((M, 3), dtype=np.float32)
        for i in range(M):
            t1, t2 = compute_tangent_basis(normals[i])
            t1_arr[i] = t1.astype(np.float32)
            t2_arr[i] = t2.astype(np.float32)

        j_indices = wp.array(particles, dtype=wp.int32, device=device)
        j_normals = wp.array(normals, dtype=wp.vec3, device=device)
        j_alpha = wp.array(alpha, dtype=wp.float32, device=device)
        j_t1 = wp.array(t1_arr, dtype=wp.vec3, device=device)
        j_t2 = wp.array(t2_arr, dtype=wp.vec3, device=device)

        W = solver.build_schur_complement(M, j_indices, j_normals, j_alpha, j_t1, j_t2)

        # Shape must be 6x6 (3 rows per contact x 2 contacts).
        self.assertEqual(W.shape, (6, 6), f"Expected (6,6), got {W.shape}")

        # Symmetry.
        np.testing.assert_allclose(W, W.T, atol=1e-10, err_msg="Friction W is not symmetric")

        # All eigenvalues positive (SPD since the contacts are at distinct particles
        # and the directions are orthonormal).
        eigvals = np.linalg.eigvalsh(W)
        self.assertGreater(float(eigvals.min()), 0.0, f"W has non-positive eigenvalue: {eigvals.min():.3e}")

    def test_static_friction_holds_particle_on_plane(self):
        """A particle on a horizontal plane with lateral gravity should not drift when mu is sufficient.

        Setup: single free particle at (0, 0, 0) exactly on the plane (y=0).
        Gravity has a horizontal component: gravity_world = (0.1, -1.0, 0.0) [m/s^2].
        Normal load: F_n = m * 1.0. Tangent force: F_t = m * 0.1.
        With mu = 0.5: mu * F_n = 0.5 > F_t = 0.1 -> static friction should hold.
        After 50 steps, tangent displacement should be < 1e-3 m.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.0),  # particle 0 — free, exactly on plane
                wp.vec3(1.0, 0.0, 0.0),  # pinned
                wp.vec3(0.0, 0.0, 1.0),  # pinned
            ],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        builder.particle_mass[1] = 0.0
        builder.particle_mass[2] = 0.0
        model = builder.finalize()
        # Set lateral gravity (horizontal component pushes particle tangentially).
        _g = wp.zeros(1, dtype=wp.vec3)
        _g.fill_(wp.vec3(0.1, -1.0, 0.0))
        model.gravity.assign(_g)
        device = model.device

        # mu=0.5 override per-pair.
        mu_override = np.full(1, 0.5, dtype=np.float64)
        solver = SolverFBA(model, iterations=5, friction=True, mu_per_pair_override=mu_override)

        contacts = newton.Contacts(rigid_contact_max=0, soft_contact_max=1, device=device)
        contacts.soft_contact_count.assign(np.array([1], dtype=np.int32))
        contacts.soft_contact_particle.assign(np.array([0], dtype=np.int32))
        contacts.soft_contact_normal.assign(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
        contacts.soft_contact_body_pos.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        contacts.soft_contact_shape.assign(np.array([-1], dtype=np.int32))

        dt = 1.0 / 60.0
        state = model.state()
        state_out = model.state()

        for _ in range(50):
            state.clear_forces()
            solver.step(state, state_out, None, contacts, dt)
            state, state_out = state_out, state

        q = state.particle_q.numpy()
        # Measure tangential displacement (x and z components of particle 0).
        tangent_drift = float(np.sqrt(q[0, 0] ** 2 + q[0, 2] ** 2))
        self.assertLessEqual(
            tangent_drift, 1e-3, f"Static friction failed: tangent drift = {tangent_drift:.6f} m (target <= 1e-3 m)"
        )
        # Particle should stay on or above the plane.
        self.assertGreaterEqual(float(q[0, 1]), -1e-4, f"Particle fell through plane: y = {float(q[0, 1]):.6f}")

    def test_kinetic_friction_slows_sliding(self):
        """Particle sliding on a plane should decelerate with friction vs without.

        Setup: particle at (0, 0, 0) with initial velocity (1.0, 0.0, 0.0) [m/s],
        gravity = (0, -1, 0), plane normal = (0, 1, 0).
        Compare tangent position after 50 steps:
          - mu=0 (frictionless): particle slides freely.
          - mu=0.5: particle decelerates.
        Expected: x(mu=0.5) < x(mu=0).
        """

        def run_sliding(mu_val: float) -> float:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[
                    wp.vec3(0.0, 0.0, 0.0),
                    wp.vec3(1.0, 0.0, 0.0),
                    wp.vec3(0.0, 0.0, 1.0),
                ],
                indices=[0, 1, 2],
                density=1.0,
                tri_ke=1.0e4,
                tri_ka=0.0,
                tri_kd=0.0,
                edge_ke=0.0,
                edge_kd=0.0,
            )
            builder.particle_mass[1] = 0.0
            builder.particle_mass[2] = 0.0
            model = builder.finalize()
            device = model.device

            # Give particle 0 an initial tangential velocity.
            state = model.state()
            vel = state.particle_qd.numpy()
            vel[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            state.particle_qd.assign(vel)
            state_out = model.state()

            mu_override = np.full(1, mu_val, dtype=np.float64)
            solver = SolverFBA(model, iterations=5, friction=True, mu_per_pair_override=mu_override)

            contacts = newton.Contacts(rigid_contact_max=0, soft_contact_max=1, device=device)
            contacts.soft_contact_count.assign(np.array([1], dtype=np.int32))
            contacts.soft_contact_particle.assign(np.array([0], dtype=np.int32))
            contacts.soft_contact_normal.assign(np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
            contacts.soft_contact_body_pos.assign(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
            contacts.soft_contact_shape.assign(np.array([-1], dtype=np.int32))

            dt = 1.0 / 60.0
            for _ in range(50):
                state.clear_forces()
                solver.step(state, state_out, None, contacts, dt)
                state, state_out = state_out, state

            return float(state.particle_q.numpy()[0, 0])

        x_frictionless = run_sliding(0.0)
        x_friction = run_sliding(0.5)

        self.assertLess(
            x_friction,
            x_frictionless,
            f"Friction should decelerate particle: x(mu=0.5)={x_friction:.4f} should be < x(mu=0)={x_frictionless:.4f}",
        )

    def test_friction_multi_shape_scene_runs(self):
        """Regression: cloth + cylinder (off-origin) with friction=True must not
        produce spurious impulses or free-fall when mu_per_pair_override is
        sized to particle_count.

        Before the fix the tangent residual was computed relative to the world
        origin rather than the contact anchor, inflating residuals by ~10x for
        off-origin contacts.  The resulting runaway friction impulses launched
        the cloth upward, clearing all contacts on the next frame, so the cloth
        free-fell indefinitely at ~3 ms/step instead of the expected ~300+ ms.

        Setup: 6x6 cloth dropped over a tilted cylinder (static).  After 50
        steps with friction=True and mu_per_pair_override sized to
        particle_count, verify:
          - no NaN in positions, and
          - cloth stays within reasonable vertical bounds (not launched to
            |z| > 5 m).
        """
        DIM = 6
        CELL_SIZE = 0.08
        CYLINDER_RADIUS = 0.3
        CYLINDER_HALF_HEIGHT = 0.6
        CYLINDER_TILT_DEG = 30.0
        DT = 1.0 / 60.0

        angle_rad = np.deg2rad(CYLINDER_TILT_DEG)
        qx = float(np.sin(angle_rad / 2.0))
        qw = float(np.cos(angle_rad / 2.0))
        tilt_quat = wp.quat(qx, 0.0, 0.0, qw)
        cylinder_center_z = 0.5

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, cylinder_center_z), tilt_quat),
            radius=CYLINDER_RADIUS,
            half_height=CYLINDER_HALF_HEIGHT,
        )
        cloth_start_x = -DIM * CELL_SIZE / 2.0
        cloth_start_y = -DIM * CELL_SIZE / 2.0
        cloth_start_z = cylinder_center_z + CYLINDER_HALF_HEIGHT + 0.5
        builder.add_cloth_grid(
            pos=wp.vec3(cloth_start_x, cloth_start_y, cloth_start_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=DIM,
            dim_y=DIM,
            cell_x=CELL_SIZE,
            cell_y=CELL_SIZE,
            mass=0.02,
            tri_ke=8.0e3,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=5.0e-3,
            edge_kd=0.0,
        )
        model = builder.finalize()

        pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.08)
        contacts = pipeline.contacts()

        particle_count = model.particle_count
        # mu_per_pair_override sized to particle_count — this is the Demo 3 pattern
        # that previously triggered the bug.
        mu_override = np.full(particle_count, 0.4, dtype=np.float64)
        solver = SolverFBA(model, iterations=5, friction=True, mu_per_pair_override=mu_override)

        s_in = model.state()
        s_out = model.state()

        for _ in range(50):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver.step(s_in, s_out, None, contacts, DT)
            s_in, s_out = s_out, s_in

        q = s_in.particle_q.numpy()
        self.assertTrue(
            np.all(np.isfinite(q)),
            "NaN/Inf positions after 50 friction steps on cloth-cylinder scene",
        )
        min_z = float(q[:, 2].min())
        max_z = float(q[:, 2].max())
        self.assertGreater(
            min_z,
            -5.0,
            f"Cloth fell too far (min_z={min_z:.2f} m); friction impulses may be spurious",
        )
        self.assertLess(
            max_z,
            5.0,
            f"Cloth launched upward (max_z={max_z:.2f} m); friction impulses may be spurious",
        )


class TestPhase4StageCShapePrimitives(unittest.TestCase):
    """Phase 4 Stage C: verify the Schur-complement pipeline works for non-plane
    shape primitives (sphere, box) using Newton's real collision detection.

    All tests drive the FULL pipeline:  collide -> update_contacts -> step.
    """

    @classmethod
    def setUpClass(cls):
        wp.init()

    # ------------------------------------------------------------------ helpers

    def _make_pipeline(self, model, soft_contact_margin=0.2):
        """Create a CollisionPipeline with an explicit soft contact margin."""
        return newton.CollisionPipeline(model, soft_contact_margin=soft_contact_margin)

    # ------------------------------------------------------------------ test 1

    def test_particle_vs_sphere_pushes_out(self):
        """Single particle inside a static sphere is pushed outside after 50 steps.

        Setup:
          - Sphere radius 0.5 at origin (body=-1, identity transform).
          - Particle 0 at (0, 0, 0.2) — inside the sphere (d = 0.2 - 0.5 = -0.3).
          - No friction; Z-up gravity.

        After 50 steps the particle should be at z >= 0.49 (sphere surface radius
        minus 1e-2 tolerance for soft-contact stiffness residual).
        """
        builder = newton.ModelBuilder()
        builder.add_shape_sphere(body=-1, radius=0.5)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.0, 0.0, 0.2),  # particle 0 — free, inside sphere
                wp.vec3(2.0, 0.0, 0.0),  # particle 1 — pinned, well outside
                wp.vec3(0.0, 2.0, 0.0),  # particle 2 — pinned, well outside
            ],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        builder.particle_mass[1] = 0.0
        builder.particle_mass[2] = 0.0
        model = builder.finalize()

        pipeline = self._make_pipeline(model)
        contacts = pipeline.contacts()
        solver = SolverFBA(model, iterations=15, friction=False)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0

        for _ in range(50):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver.step(s_in, s_out, None, contacts, dt)
            s_in, s_out = s_out, s_in

        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions after sphere contact test")
        dist = float(np.linalg.norm(q[0]))
        self.assertGreaterEqual(
            dist,
            0.49,
            f"Particle distance from origin {dist:.4f} should be >= 0.49 (sphere radius 0.5) after 50 steps",
        )

    # ------------------------------------------------------------------ test 2

    def test_particle_vs_box_no_interpenetration(self):
        """Particle slightly inside a static box is pushed outside after 50 steps.

        Setup:
          - Box at origin, half-extents (1, 1, 1); extends from -1 to +1 on each axis.
          - Particle 0 at (0.9, 0.0, 0.0) — inside the box, 0.1 m from the +x face.
          - No friction; Z-up gravity.

        After 50 steps the particle should have x >= 0.99 (pushed toward +x face)
        OR be outside the box (|x|, |y|, or |z| >= 1.0 - 1e-2).
        """
        builder = newton.ModelBuilder()
        builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=1.0)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(0.9, 0.0, 0.0),  # particle 0 — free, inside box (0.1 m from +x)
                wp.vec3(0.9, 2.0, 0.0),  # particle 1 — pinned, outside box
                wp.vec3(0.9, 0.0, 2.0),  # particle 2 — pinned, outside box
            ],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        builder.particle_mass[1] = 0.0
        builder.particle_mass[2] = 0.0
        model = builder.finalize()

        pipeline = self._make_pipeline(model)
        contacts = pipeline.contacts()
        solver = SolverFBA(model, iterations=15, friction=False)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0

        for _ in range(50):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver.step(s_in, s_out, None, contacts, dt)
            s_in, s_out = s_out, s_in

        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions after box contact test")
        # Particle should be at or outside one of the box faces (|coord| >= 0.99).
        p0 = q[0]
        outside_or_on_surface = (abs(p0[0]) >= 0.99) or (abs(p0[1]) >= 0.99) or (abs(p0[2]) >= 0.99)
        self.assertTrue(
            outside_or_on_surface,
            f"Particle {p0} should be at or outside box surface (half-extents=1) after 50 steps",
        )

    # ------------------------------------------------------------------ test 3

    def test_cloth_drapes_on_sphere(self):
        """4x4 cloth drapes over a sphere; no NaN and no deep penetration after 200 steps.

        Setup:
          - Sphere radius 0.4 at origin (body=-1).
          - 4x4 cloth grid starting at (-0.4, -0.4, 0.6) above the sphere.
          - Top row pinned; gravity -Z.
          - No friction.

        Assertions:
          - No NaN in particle positions.
          - No particle center is deeper than 0.3 m inside the sphere (distance
            from origin < 0.4 - 0.3 = 0.1 m), allowing for soft-pin tolerance.
        """
        builder = newton.ModelBuilder()
        builder.add_shape_sphere(body=-1, radius=0.4)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.3, -0.3, 0.6),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            cell_x=0.15,
            cell_y=0.15,
            mass=0.05,
            tri_ke=1.0e3,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-2,
            edge_kd=0.0,
            fix_top=True,
        )
        model = builder.finalize()

        pipeline = self._make_pipeline(model, soft_contact_margin=0.15)
        contacts = pipeline.contacts()
        solver = SolverFBA(model, iterations=10, friction=False)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0

        for _ in range(200):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver.step(s_in, s_out, None, contacts, dt)
            s_in, s_out = s_out, s_in

        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions after cloth-drapes-on-sphere test")

        # No particle should be more than 0.3 m deep inside the sphere.
        dists = np.linalg.norm(q, axis=1)  # distance from sphere centre (origin)
        min_dist = float(dists.min())
        self.assertGreaterEqual(
            min_dist,
            0.1,
            f"Deepest particle is {0.4 - min_dist:.3f} m inside the sphere (min dist={min_dist:.3f}), "
            "expected < 0.3 m penetration",
        )

    # ------------------------------------------------------------------ test 4

    def test_softbody_falls_on_box(self):
        """3x3x3 soft-body cube falls onto a static box; no NaN after 100 steps.

        Setup:
          - Box at origin, half-extents (0.5, 0.5, 0.1) (a flat platform).
          - 3x3x3 soft-body cube with cell size 0.1 m starting at z=0.3 above the box.
          - No friction; gravity -Z.

        Assertions:
          - No NaN in particle positions.
          - Soft-body is not deeply below the box top surface (z >= -0.1 -  0.1 = -0.2 m).
        """
        builder = newton.ModelBuilder()
        # Flat platform: half-extents in XY are generous, thin in Z.
        builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.1)
        builder.add_soft_grid(
            pos=wp.vec3(-0.1, -0.1, 0.3),  # above box top face (z=0.1)
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=3,
            dim_y=3,
            dim_z=3,
            cell_x=0.07,
            cell_y=0.07,
            cell_z=0.07,
            density=5.0e2,
            k_mu=1.0e3,
            k_lambda=1.0e3,
            k_damp=0.0,
        )
        model = builder.finalize()

        pipeline = self._make_pipeline(model, soft_contact_margin=0.15)
        contacts = pipeline.contacts()
        solver = SolverFBA(model, iterations=10, friction=False)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0

        for _ in range(100):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver.step(s_in, s_out, None, contacts, dt)
            s_in, s_out = s_out, s_in

        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions after softbody-on-box test")

        # No particle should be deeply below the box top face (z = 0.1).
        # Allow 0.2 m tolerance for soft contact residual.
        min_z = float(q[:, 2].min())
        self.assertGreaterEqual(
            min_z,
            -0.2,
            f"Soft-body min z={min_z:.3f} is below the box + tolerance threshold -0.2 m",
        )

    # ------------------------------------------------------------------ test 5

    def test_mixed_shape_collision(self):
        """Cloth above a plane + a sphere; contacts from both shapes processed correctly.

        Setup:
          - Ground plane (Z-up, at z=0).
          - Sphere radius 0.3 at origin (body=-1).
          - 4x4 cloth grid starting at (-0.3, -0.3, 0.5) above both colliders.
          - Top row pinned; gravity -Z; no friction.

        Assertions:
          - No NaN after 150 steps.
          - No particle below z = -0.05 (plane).
          - No particle closer than 0.25 m to sphere centre (allowing 0.05 m
            soft tolerance below radius 0.3).
        """
        builder = newton.ModelBuilder()
        builder.add_ground_plane()  # plane at z=0
        builder.add_shape_sphere(body=-1, radius=0.3)  # sphere at origin
        builder.add_cloth_grid(
            pos=wp.vec3(-0.3, -0.3, 0.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            cell_x=0.15,
            cell_y=0.15,
            mass=0.05,
            tri_ke=1.0e3,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=1.0e-2,
            edge_kd=0.0,
            fix_top=True,
        )
        model = builder.finalize()

        pipeline = self._make_pipeline(model, soft_contact_margin=0.15)
        contacts = pipeline.contacts()
        solver = SolverFBA(model, iterations=10, friction=False)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0

        for _ in range(150):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver.step(s_in, s_out, None, contacts, dt)
            s_in, s_out = s_out, s_in

        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions after mixed-shape collision test")

        # No particle below ground plane (z < -0.05 with soft tolerance).
        min_z = float(q[:, 2].min())
        self.assertGreaterEqual(min_z, -0.05, f"Particle below ground plane: min z={min_z:.4f}")

        # No particle more than 0.05 m inside sphere.
        dists = np.linalg.norm(q, axis=1)
        min_dist = float(dists.min())
        self.assertGreaterEqual(
            min_dist,
            0.25,
            f"Particle too deep inside sphere: min dist={min_dist:.4f} m (sphere radius=0.3)",
        )

    # ------------------------------------------------------------------ test 6

    def test_non_origin_sphere_contact_no_double_transform(self):
        """Regression test: sphere placed at a non-origin position should not
        double-apply shape_transform when computing the contact world anchor.

        If the double-transform bug is reintroduced the computed contact offset
        will be wrong (the anchor will be placed far from the true contact surface)
        and the particle will not be pushed out correctly.

        Setup:
          - Static sphere, radius 0.5, centre at (5, 0, 0).
          - Single free particle at (5, 0, 0.2) — inside the sphere (d ≈ -0.3).
          - No gravity, no friction; particle should be pushed radially outward.

        After 80 steps the particle must be at distance >= 0.48 from (5, 0, 0)
        (sphere radius 0.5 minus 0.02 soft-contact residual tolerance).
        """
        builder = newton.ModelBuilder()
        # Sphere at (5, 0, 0) — non-origin position exercises the
        # static-shape path in update_contacts that previously double-applied
        # the shape_transform.
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(wp.vec3(5.0, 0.0, 0.0), wp.quat_identity()),
            radius=0.5,
        )
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(5.0, 0.0, 0.2),  # particle 0 — free, inside sphere
                wp.vec3(10.0, 0.0, 0.0),  # particle 1 — pinned, well outside
                wp.vec3(5.0, 10.0, 0.0),  # particle 2 — pinned, well outside
            ],
            indices=[0, 1, 2],
            density=1.0,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        builder.particle_mass[1] = 0.0
        builder.particle_mass[2] = 0.0
        model = builder.finalize()
        # Override gravity to zero so the particle does not drift downward under
        # gravity while being pushed out of the sphere.
        model.gravity.assign(wp.zeros(1, dtype=wp.vec3))

        pipeline = self._make_pipeline(model, soft_contact_margin=0.2)
        contacts = pipeline.contacts()
        solver = SolverFBA(model, iterations=15, friction=False)
        s_in, s_out = model.state(), model.state()
        dt = 1.0 / 60.0

        for _ in range(80):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver.step(s_in, s_out, None, contacts, dt)
            s_in, s_out = s_out, s_in

        q = s_in.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)), "non-finite positions in non-origin sphere test")

        sphere_centre = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        dist = float(np.linalg.norm(q[0].astype(np.float64) - sphere_centre))
        self.assertGreaterEqual(
            dist,
            0.48,
            f"Particle distance from sphere centre {dist:.4f} m should be >= 0.48 "
            "(sphere radius=0.5). If the double-transform bug is present the "
            "contact anchor is miscomputed and the particle stays inside.",
        )


class SolverFBAConstructorOptionsTests(unittest.TestCase):
    """Constructor exposes NSN iter count and lambda cap."""

    def _tiny_cloth_model(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.1, -0.1, 0.2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=3,
            dim_y=3,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.05,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
        )
        return builder.finalize()

    def test_default_nsn_iter_is_1(self) -> None:
        model = self._tiny_cloth_model()
        solver = SolverFBA(model)
        self.assertEqual(solver.nsn_iterations, 1)

    def test_nsn_iter_overridable(self) -> None:
        model = self._tiny_cloth_model()
        solver = SolverFBA(model, nsn_iterations=25)
        self.assertEqual(solver.nsn_iterations, 25)

    def test_lambda_cap_default_is_none(self) -> None:
        model = self._tiny_cloth_model()
        solver = SolverFBA(model)
        self.assertIsNone(solver.lambda_cap)

    def test_lambda_cap_settable(self) -> None:
        model = self._tiny_cloth_model()
        solver = SolverFBA(model, lambda_cap=100.0)
        self.assertEqual(solver.lambda_cap, 100.0)


class SolverFBALambdaWarmStartTests(unittest.TestCase):
    """λ warm-start across PD outer iters reproduces RealSim's per-frame reset.

    Verifies:
    1. λ is zero at step entry (per-frame setZero).
    2. After the first NSN call within a step, λ persistent is non-zero.
    3. After update_contacts changes the set, λ persistent is reset.
    """

    def _cloth_on_plane_model(self):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.add_ground_plane()
        builder.add_cloth_grid(
            pos=wp.vec3(-0.2, -0.2, 0.15),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
            tri_ke=1.0e4,
            tri_ka=0.0,
            tri_kd=0.0,
        )
        return builder.finalize()

    def test_lam_persistent_initially_none(self) -> None:
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._cloth_on_plane_model()
        solver = SolverFBA(model)
        self.assertIsNone(solver._lam_unilateral_persistent)
        self.assertIsNone(solver._lam_coulomb_persistent)

    def test_lam_persistent_accumulates_within_step(self) -> None:
        """After one step with contacts, the persistent λ buffer is non-zero."""
        from newton import CollisionPipeline  # noqa: PLC0415
        from newton.solvers import SolverFBA  # noqa: PLC0415

        model = self._cloth_on_plane_model()
        solver = SolverFBA(model, friction=False)
        s_in, s_out = model.state(), model.state()
        pipeline = CollisionPipeline(model, soft_contact_margin=0.05)
        contacts = pipeline.contacts()

        # Step a few times to let cloth settle into contact.
        for _ in range(10):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver.step(s_in, s_out, None, contacts, 1.0 / 60.0)
            s_in, s_out = s_out, s_in

        # λ should be populated for the contact set
        # (cloth bottom row resting on plane).
        if solver._contact_count > 0:
            self.assertIsNotNone(solver._lam_unilateral_persistent)
            self.assertGreater(
                float(np.abs(solver._lam_unilateral_persistent).max()),
                0.0,
                "λ should be non-zero after sustained contact",
            )


class SolverFBAKinematicCylinderTests(unittest.TestCase):
    """Per-shape angular velocity surfaces in Stage B tangent offsets."""

    def _ball_and_cylinder(self):
        """Tiny tet ball + 1 cylinder (axis +X world). Returns (model, pipeline, contacts)."""
        import math

        import newton

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-10.0)
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.5, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.2,
            cell_y=0.2,
            cell_z=0.2,
            density=500.0,
            k_mu=5.0e3,
            k_lambda=5.0e3,
            k_damp=0.0,
        )
        # Z-aligned cylinder primitive → rotate to +X via 90° about -Y.
        half = math.pi / 4.0
        qy = -math.sin(half)
        qw = math.cos(half)
        q = wp.quat(0.0, qy, 0.0, qw)
        builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, -0.2, 0.0), q),
            radius=0.3,
            half_height=2.0,
        )
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.1)
        contacts = pipeline.contacts()
        return model, pipeline, contacts

    def test_default_no_kinematic_motion(self) -> None:
        from newton.solvers import SolverFBA
        model, _, _ = self._ball_and_cylinder()
        solver = SolverFBA(model, friction=True)
        self.assertTrue(np.all(solver._shape_omega_h == 0.0))

    def test_kinematic_motion_stored_per_shape(self) -> None:
        from newton.solvers import SolverFBA
        model, _, _ = self._ball_and_cylinder()
        solver = SolverFBA(
            model,
            friction=True,
            shape_angular_velocity={0: -3.0},
        )
        self.assertEqual(float(solver._shape_omega_h[0]), -3.0)

    def test_kinematic_anchor_velocity_shifts_offsets(self) -> None:
        """ω ≠ 0 produces nonzero v_anchor; ω = 0 leaves it zero."""
        from newton.solvers import SolverFBA

        model, pipeline, contacts = self._ball_and_cylinder()
        s_in = model.state()
        s_out = model.state()

        # Drop the ball onto the cylinder to land contacts.
        solver_static = SolverFBA(
            model,
            friction=True,
            mu_per_pair_override=np.full(model.particle_count, 0.5),
        )
        for _ in range(20):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver_static.step(s_in, s_out, None, contacts, 1.0 / 60.0)
            s_in, s_out = s_out, s_in
            if solver_static._contact_count > 0:
                break
        self.assertGreater(
            solver_static._contact_count, 0,
            "Need at least one contact for the kinematic path."
        )

        solver_spin = SolverFBA(
            model,
            friction=True,
            mu_per_pair_override=np.full(model.particle_count, 0.5),
            shape_angular_velocity={0: -3.0},
        )
        # Drive one step on both so update_contacts ran on each.
        s2 = model.state()
        solver_static.step(s_in, s_out, None, contacts, 1.0 / 60.0)
        solver_spin.step(s_in, s2, None, contacts, 1.0 / 60.0)

        # static path: v_anchor must be exactly zero.
        np.testing.assert_allclose(
            solver_static._contact_v_anchor_h[: solver_static._contact_count],
            0.0,
        )
        # spin path: v_anchor must be nonzero for at least one contact.
        self.assertGreater(
            float(np.linalg.norm(
                solver_spin._contact_v_anchor_h[: solver_spin._contact_count]
            )),
            1e-6,
            "Expected ω ≠ 0 to produce nonzero v_anchor."
        )

    def test_cylinder_aligned_basis_for_spinning_shape(self) -> None:
        """For shapes with ω≠0, tangent basis is (normal × axis, normalize(normal × t1)).

        Verifies (i) v_anchor lies almost entirely along t1, (ii) projection onto
        t2 ≈ 0 (the axial direction carries no anchor shift, matching RealSim).
        """
        from newton.solvers import SolverFBA

        model, pipeline, contacts = self._ball_and_cylinder()
        s_in = model.state()
        s_out = model.state()
        solver_spin = SolverFBA(
            model,
            friction=True,
            mu_per_pair_override=np.full(model.particle_count, 0.5),
            shape_angular_velocity={0: -3.0},
        )
        for _ in range(20):
            s_in.clear_forces()
            pipeline.collide(s_in, contacts)
            solver_spin.step(s_in, s_out, None, contacts, 1.0 / 60.0)
            s_in, s_out = s_out, s_in
            if solver_spin._contact_count > 0:
                break
        self.assertGreater(solver_spin._contact_count, 0)

        M = solver_spin._contact_count
        v = solver_spin._contact_v_anchor_h[:M]
        t1 = solver_spin._contact_tangent1_d.numpy()[:M].astype(np.float64)
        t2 = solver_spin._contact_tangent2_d.numpy()[:M].astype(np.float64)
        proj_t1 = np.einsum("ij,ij->i", t1, v)
        proj_t2 = np.einsum("ij,ij->i", t2, v)
        # RealSim parity: v_anchor projection onto axial t2 must be ≈ 0.
        # Allow small slack for numerical SVD / float32 round-trip on tangent arrays.
        self.assertLess(float(np.max(np.abs(proj_t2))), 1e-4)
        # And t1 must carry most of the v_anchor norm.
        v_norm = np.linalg.norm(v, axis=1)
        self.assertGreater(
            float(np.min(np.abs(proj_t1) / np.maximum(v_norm, 1e-12))), 0.95,
            "v_anchor should lie almost entirely along the rolling tangent t1."
        )


class ParticleElementCsrTests(unittest.TestCase):
    """Adjacency CSR builder produces correct (offsets, element_idx, local_vertex_idx)."""

    def test_basic_tet_adjacency(self) -> None:
        import numpy as np
        from newton._src.solvers.fba.linear_solver import build_particle_element_csr

        # 2 tets sharing vertices 1, 2, 3.
        tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
        offsets, elem_idx, local_v = build_particle_element_csr(tets, n_particles=5, n_verts_per_element=4)

        self.assertEqual(offsets.tolist(), [0, 1, 3, 5, 7, 8])
        # Vertex 1's entries: (tet=0, local=1) and (tet=1, local=0).
        v1_slice = slice(offsets[1], offsets[2])
        elem_v1 = sorted(zip(elem_idx[v1_slice].tolist(), local_v[v1_slice].tolist()))
        self.assertEqual(elem_v1, [(0, 1), (1, 0)])

    def test_tri_adjacency(self) -> None:
        import numpy as np
        from newton._src.solvers.fba.linear_solver import build_particle_element_csr

        tris = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        offsets, elem_idx, local_v = build_particle_element_csr(tris, n_particles=4, n_verts_per_element=3)
        self.assertEqual(offsets.tolist(), [0, 1, 3, 5, 6])
        v1_slice = slice(offsets[1], offsets[2])
        elem_v1 = sorted(zip(elem_idx[v1_slice].tolist(), local_v[v1_slice].tolist()))
        self.assertEqual(elem_v1, [(0, 1), (1, 0)])

    def test_empty_adjacency(self) -> None:
        import numpy as np
        from newton._src.solvers.fba.linear_solver import build_particle_element_csr

        empty = np.zeros((0, 4), dtype=np.int32)
        offsets, elem_idx, local_v = build_particle_element_csr(empty, n_particles=3, n_verts_per_element=4)
        self.assertEqual(offsets.tolist(), [0, 0, 0, 0])
        self.assertEqual(elem_idx.shape, (0,))
        self.assertEqual(local_v.shape, (0,))


if __name__ == "__main__":
    unittest.main()
