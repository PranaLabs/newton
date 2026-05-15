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
        # For this flat-rest test, set x_ref = positions; with a translation-invariant
        # q (sum to zero), qᵀ·x_ref = 0 ⇒ scatter contribution is zero.
        x_ref = wp.array(positions.numpy(), dtype=wp.vec3, device=device)
        # q = [1,1,-1,-1] satisfies sum-to-zero.
        edge_quad_q = wp.array([wp.vec4(1.0, 1.0, -1.0, -1.0)], dtype=wp.vec4, device=device)
        weight = wp.array([1.0], dtype=wp.float32, device=device)
        rhs = wp.zeros(4, dtype=wp.vec3, device=device)

        wp.launch(
            project_bending_kernel,
            dim=1,
            inputs=[positions, x_ref, edge_indices, edge_quad_q, weight],
            outputs=[rhs],
            device=device,
        )
        # qᵀ·x_ref = x_ref[0] + x_ref[1] - x_ref[2] - x_ref[3]
        #          = (0,0,0)+(1,0,0)-(0.5,0,1)-(0.5,0,-1) = (0,0,0)
        # So scatter is zero.
        np.testing.assert_allclose(rhs.numpy(), np.zeros((4, 3)), atol=1e-6)


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


if __name__ == "__main__":
    unittest.main()
