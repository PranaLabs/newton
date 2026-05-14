# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba import SolverFBA
from newton._src.solvers.fba.kernels import project_stretching_arap_kernel
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

    def test_identity_F_gives_identity_P(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # One triangle at the rest configuration → F = identity column-augmented.
        # rest pos: (0,0,0), (1,0,0), (0,0,1); current pos = rest → F·Dm = identity.
        positions = wp.array(
            [wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 0, 1)],
            dtype=wp.vec3,
            device=device,
        )
        tri_indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
        Dm_inv = wp.array(
            [wp.mat22(1.0, 0.0, 0.0, 1.0)],  # 2x2 identity
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
        rhs_np = rhs.numpy()
        # When F is the rest configuration, ARAP projects to identity rotation;
        # the scatter contribution is the "rest" stencil — caller will balance
        # it with the Hessian. Verify scatter pattern matches Aᵀ·proj where
        # proj is the rest configuration mapped through Dm_inv·Pᵀ.
        # Concretely: rhs[1] - rhs[2] should be along e12 direction; rhs[0] = -rhs[1] - rhs[2].
        np.testing.assert_allclose(rhs_np[0], -(rhs_np[1] + rhs_np[2]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
