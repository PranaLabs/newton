# FBA Tet Corotational and Neo-Hookean Stretching Models

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3D Corotational and Neo-Hookean stretching kernels to SolverFBA's tetrahedral path, mirroring the existing cloth (triangle) Corot/NH implementations.

**Architecture:** Two new `@wp.func` projection functions (`project_corotational_sigma3d`, `project_neohookean_sigma3d`) implement closed-form 3D Corot and 5-iter Newton 3D NH; two new `@wp.kernel` functions (`project_stretching_corotational_tet_kernel`, `project_stretching_neohookean_tet_kernel`) wrap them with the same F = Ds·Dm_inv → SVD → scatter pattern as the existing `project_stretching_arap_tet_kernel`. The `SolverFBA.step()` tet dispatch block replaces the existing `NotImplementedError` raises with the new kernel launches.

**Tech Stack:** Python, Warp (wp.svd3, wp.mat33, wp.vec3), NumPy, Newton ModelBuilder. No new dependencies. Tests use `unittest` (Newton project standard).

---

## File Structure

- **Modify:** `newton/_src/solvers/fba/kernels.py` — add `project_corotational_sigma3d`, `project_stretching_corotational_tet_kernel`, `project_neohookean_sigma3d`, `project_stretching_neohookean_tet_kernel`
- **Modify:** `newton/_src/solvers/fba/solver_fba.py` — wire new kernels in `step()` tet dispatch + update import
- **Modify:** `newton/tests/test_solver_fba.py` — add `TestTetCorotational` and `TestTetNeoHookean` classes
- **Modify:** `scripts/fba_arap_vs_corot_bench.py` — extend to benchmark tet ARAP/Corot/NH on 8×8×8 cube

---

### Task A: Add `project_corotational_sigma3d` @wp.func and `project_stretching_corotational_tet_kernel` @wp.kernel

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py` (after the existing `project_stretching_arap_tet_kernel` block, before `project_stretching_arap_kernel`)

**Algorithm (Corotational 3D closed-form):**

Energy: `E(σ) = μ·||σ - I||² + (λ/2)·tr(σ-I)² + (k/2)·||σ - σ₀||²`, k = 2μ

Setting ∇E = 0 → matrix form `A·σ = b` where A = `4μ·I + λ·11ᵀ` (Sherman-Morrison):
```
α = 4μ
β = λ  
n = 3
inv(αI + β·11ᵀ) = (1/α)·I - β/(α·(α + n·β))·11ᵀ

sum_b = 6μ + 9λ + 2μ·(σ₀[0] + σ₀[1] + σ₀[2])
σ_proj[i] = (2μ + 3λ + 2μ·σ₀[i]) / (4μ) - λ·sum_b / (4μ·(4μ + 3λ))
```

**Verification (hand-check):** σ₀ = (1.5, 1.5, 1.5), μ=1, λ=0.5:
- sum_b = 6 + 4.5 + 2·4.5 = 19.5
- σ_proj[i] = (2+1.5+3)/4 - 0.5·19.5/(4·5.5) = 6.5/4 - 9.75/22 = 1.625 - 0.44318... = 1.18182...

The test must verify σ_proj ≈ 1.18182 (within 1e-5).

- [ ] **Step 1: Write the failing test for Corot 3D projection function**

Add `TestTetCorotational` class to `newton/tests/test_solver_fba.py` at the end of the file (before `if __name__ == "__main__":`):

```python
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
        # Ds = 1.5*I (edge vectors scaled by 1.5), Dm_inv = I → F = 1.5*I.
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

        # F = 1.5*I → U=V=I, sigma=(1.5,1.5,1.5).
        # sigma_proj ≈ 1.18182 for all three.
        # P = I*diag(sigma_proj)*I = sigma_proj*I.
        # proj = w*Dm_inv*P^T = sigma_proj*I.
        # row0=(sigma_proj,0,0), row1=(0,sigma_proj,0), row2=(0,0,sigma_proj)
        expected = 1.625 - 0.5 * 19.5 / (4.0 * 5.5)  # ≈1.18182
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
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run --extra dev -m newton.tests -k "TestTetCorotational"
```

Expected: `AttributeError` or `ImportError` — `project_stretching_corotational_tet_kernel` does not exist yet.

- [ ] **Step 3: Add `project_corotational_sigma3d` @wp.func to kernels.py**

In `newton/_src/solvers/fba/kernels.py`, insert after the closing `}` of `project_stretching_arap_tet_kernel` (after line 287) and before `@wp.kernel` of `project_stretching_arap_kernel` (line 289):

```python
@wp.func
def project_corotational_sigma3d(sigma: wp.vec3, mu: float, lam: float) -> wp.vec3:
    """Closed-form 3D corotational local projection on SVD singular values.

    Minimises E(σ) = μ·||σ-I||² + (λ/2)·tr(σ-I)² + (k/2)·||σ-σ₀||²
    with k = 2μ (PD penalty weight), I = (1,1,1) identity singular values.

    Setting ∇E = 0 yields the 3x3 linear system (4μ·I + λ·11ᵀ)·σ = b,
    solved in closed form via Sherman-Morrison:

        α = 4μ,  β = λ,  n = 3
        inv(αI + β·11ᵀ) = (1/α)·I - β/(α·(α + n·β))·11ᵀ
        b_i = 2μ + 3λ + 2μ·σ₀_i
        sum_b = 6μ + 9λ + 2μ·(σ₀[0]+σ₀[1]+σ₀[2])
        σ_proj[i] = b_i/α - β·sum_b / (α·(α + 3β))

    Note: uses the standard symmetric trace formula (NOT mimicking RealSim's
    Eigen .trace() bug on Vector2d which only reads index 0). The 2D cloth
    Corot in this codebase uses the same standard formulation; the 3D tet
    Corot here is the natural 3D extension of the same energy.

    Args:
        sigma: Singular values from ``wp.svd3(F)``, shape (3,), in descending order.
        mu: First Lamé parameter (shear modulus) [Pa].
        lam: Second Lamé parameter [Pa].

    Returns:
        Projected singular-value triple ``(σ_proj_0, σ_proj_1, σ_proj_2)``.
    """
    k = 2.0 * mu  # PD penalty weight (k = 2μ)
    alpha = 4.0 * mu  # = 2μ + λ/... wait — see derivation below
    # Full derivation:
    # Diagonal of system matrix: (2μ + λ + k) = 2μ + λ + 2μ = 4μ + λ
    # Off-diagonal: λ
    # But we absorb the λ off-diagonal into the rank-1 update:
    # A = (4μ + λ)·I - λ·I + λ·11ᵀ = 4μ·I + λ·11ᵀ... wait, A_ii = 4μ+λ, A_ij = λ
    # = 4μ·I + λ·11ᵀ  ← correct (diagonal contribution 4μ, off-diag λ)
    # Wait: A_ii = 4μ+λ = 4μ + λ·1 = (4μ·I + λ·11ᵀ)_ii ✓
    #        A_ij = λ = (4μ·I + λ·11ᵀ)_ij ✓
    # Sherman-Morrison with α=4μ, β=λ:
    alpha = 4.0 * mu
    beta = lam
    n = 3.0

    # RHS: b_i = 2μ + 3λ + 2μ·σ₀_i = 2(μ+λ) + λ + k·σ₀_i
    # (derived by differentiating E and setting = 0: (4μ+λ)σ_i + λ(σ_{i+1}+σ_{i+2}) = 2μ+3λ+k·σ₀_i
    #  which matches (2μ + λ + k)σ_i + λ Σ_{j≠i} σ_j = ... and collecting)
    b0 = 2.0 * mu + 3.0 * lam + k * sigma[0]
    b1 = 2.0 * mu + 3.0 * lam + k * sigma[1]
    b2 = 2.0 * mu + 3.0 * lam + k * sigma[2]
    sum_b = b0 + b1 + b2

    # σ_proj[i] = b_i/α - β·sum_b / (α·(α + n·β))
    scale = beta * sum_b / (alpha * (alpha + n * beta))
    proj0 = b0 / alpha - scale
    proj1 = b1 / alpha - scale
    proj2 = b2 / alpha - scale
    return wp.vec3(proj0, proj1, proj2)
```

- [ ] **Step 4: Add `project_stretching_corotational_tet_kernel` @wp.kernel to kernels.py**

Immediately after `project_corotational_sigma3d`, insert:

```python
@wp.kernel
def project_stretching_corotational_tet_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-tet corotational local projection scatter for PD softbody.

    Computes F = Ds * Dm_inv (3x3), extracts singular values via ``wp.svd3``,
    applies the closed-form 3D corotational projection to get ``sigma_proj``,
    reconstructs ``P = U * diag(sigma_proj) * Vt``, and scatters
    ``w * Dm_inv * P^T`` into the four stencil vertices via atomic_add.

    No reflection handling is needed: ``wp.svd3`` returns non-negative singular
    values, and the corotational projection preserves positivity; the
    reconstructed P naturally has the correct orientation.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        mu: First Lamé parameter [Pa].
        lam: Second Lamé parameter [Pa].
        rhs: Output RHS accumulator (atomic-add target), shape ``[particle_count]``.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    # Ds = [p1-p0 | p2-p0 | p3-p0]  (column-stack 3 edge vectors -> 3x3)
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0], e2[0], e3[0],
        e1[1], e2[1], e3[1],
        e1[2], e2[2], e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    U, sigma, V = wp.svd3(F)

    # Corotational projection: closed-form 3D solve on singular values.
    sigma_proj = project_corotational_sigma3d(wp.vec3(sigma[0], sigma[1], sigma[2]), mu, lam)

    # Reconstruct P = U * diag(sigma_proj) * V^T  (3x3).
    # P = U * S * V^T where S = diag(sigma_proj).
    # Column j of V^T is row j of V; P[i,j] = Σ_k U[i,k] * sigma_proj[k] * V[j,k]
    s0 = sigma_proj[0]
    s1 = sigma_proj[1]
    s2 = sigma_proj[2]
    # U columns scaled by sigma_proj; then multiply by V^T
    # P = (U * diag(s)) * V^T
    US = wp.mat33(
        U[0, 0] * s0, U[0, 1] * s1, U[0, 2] * s2,
        U[1, 0] * s0, U[1, 1] * s1, U[1, 2] * s2,
        U[2, 0] * s0, U[2, 1] * s1, U[2, 2] * s2,
    )
    P = US * wp.transpose(V)

    # proj = w * Dm_inv * P^T  (3x3)
    w = tet_weight[t]
    PT = wp.transpose(P)
    proj = w * (Dm_inv * PT)

    # Scatter stencil (same as ARAP tet):
    #   rhs[t[0]] += -proj.row(0) - proj.row(1) - proj.row(2)
    #   rhs[t[1]] += proj.row(0)
    #   rhs[t[2]] += proj.row(1)
    #   rhs[t[3]] += proj.row(2)
    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    wp.atomic_add(rhs, i0, -(row0 + row1 + row2))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)
    wp.atomic_add(rhs, i3, row2)
```

- [ ] **Step 5: Run test again to verify it passes**

```bash
uv run --extra dev -m newton.tests -k "TestTetCorotational"
```

Expected: all 4 tests in `TestTetCorotational` pass. Verify the `test_isotropic_stretch_corot_3d` value against the hand computation (≈1.18182).

- [ ] **Step 6: Run full test suite to verify no regressions**

```bash
uv run --extra dev -m newton.tests
```

Expected: 47 + 4 = 51 tests pass. (The tet dispatch in solver_fba.py still raises NotImplementedError for Corot/NH when used with tets, but the kernel itself can be tested directly — the integration test `test_solver_corot_tet_softbody_runs` will fail until Task B wires the dispatch. Run only non-integration tests here, or defer until Task B.)

> Note: `test_solver_corot_tet_softbody_runs` imports `SolverFBA` and calls `step()`. The dispatch in `step()` still raises `NotImplementedError` for tet+corot. So this specific test will fail until Task B. Either:
> (a) Add the dispatch wiring in this step (acceptable since it's 3 lines), or
> (b) Accept that test will fail until Task B. 
> 
> **Recommendation:** Do option (a) — wire the dispatch as part of this step to keep tests green.

- [ ] **Step 7: Wire Corot tet dispatch in solver_fba.py**

In `newton/_src/solvers/fba/solver_fba.py`:

1. Add `project_stretching_corotational_tet_kernel` to the import block in `step()`:

```python
        from .kernels import (  # noqa: PLC0415
            add_inertia_to_rhs_kernel,
            compute_inertial_kernel,
            project_bending_kernel,
            project_pin_kernel,
            project_stretching_arap_kernel,
            project_stretching_arap_tet_kernel,
            project_stretching_corotational_kernel,
            project_stretching_corotational_tet_kernel,
            project_stretching_neohookean_kernel,
            write_velocity_kernel,
            zero_vec3_kernel,
        )
```

2. Replace the tet dispatch block (currently lines 344-362) with:

```python
            # Tet stretching projection.
            if self._tet_indices_d is not None and model.tet_count > 0:
                if self.stretching_model == "arap":
                    wp.launch(
                        project_stretching_arap_tet_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
                elif self.stretching_model == "corotational":
                    wp.launch(
                        project_stretching_corotational_tet_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                            self._mu,
                            self._lam,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
                elif self.stretching_model == "neohookean":
                    raise NotImplementedError(
                        "stretching_model='neohookean' is not yet implemented for tets. "
                        "Only 'arap' and 'corotational' are supported for tetrahedral elements."
                    )
```

- [ ] **Step 8: Run all tests again**

```bash
uv run --extra dev -m newton.tests
```

Expected: 47 + 4 = 51 tests pass. (NH tet integration test not yet added.)

- [ ] **Step 9: Lint and format**

```bash
uvx pre-commit run -a
```

Fix any issues found.

- [ ] **Step 10: Commit**

```bash
git add newton/_src/solvers/fba/kernels.py newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "$(cat <<'EOF'
Add project_corotational_sigma3d + tet Corot kernel

Implements closed-form 3D corotational PD projection via Sherman-Morrison
(standard symmetric trace formula, not RealSim's Eigen .trace() bug).
Wires the new kernel into SolverFBA.step() tet dispatch.
EOF
)"
```

---

### Task B: Add `project_neohookean_sigma3d` @wp.func and `project_stretching_neohookean_tet_kernel` @wp.kernel

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py` (after the Corot tet kernel from Task A)
- Modify: `newton/_src/solvers/fba/solver_fba.py` (wire NH tet dispatch)
- Modify: `newton/tests/test_solver_fba.py` (add `TestTetNeoHookean` class)

**Algorithm (Neo-Hookean 3D Newton's method):**

Energy: `E_NH(σ) = (μ/2)(I₁ - 2·log(J) - 3) + (λ/2)·log(J)² + (k/2)||σ - σ₀||²`  
where I₁ = σ₀² + σ₁² + σ₂², J = σ₀·σ₁·σ₂, k = 2μ.

Gradient:
```
∇E_i = μ·(σ_i - 1/σ_i) + λ·log(J)/σ_i + k·(σ_i - σ₀_i)
```

Hessian (symmetric 3x3):
```
log_J = log(J)
diag_factor = μ + λ - λ·log_J
H_ii = μ + diag_factor/σ_i² + k
H_ij = λ/(σ_i·σ_j)   for i≠j
```

3x3 inverse via cofactors:
```
det_H = H00*(H11*H22 - H12²) - H01*(H01*H22 - H12*H02) + H02*(H01*H12 - H11*H02)
C00 = H11*H22 - H12²
C11 = H00*H22 - H02²
C22 = H00*H11 - H01²
C01 = -(H01*H22 - H12*H02)  # cofactor (0,1) = cofactor (1,0) by symmetry
C02 = H01*H12 - H11*H02     # cofactor (0,2) = cofactor (2,0)
C12 = -(H00*H12 - H01*H02)  # cofactor (1,2) = cofactor (2,1)
Δσ = [C00*g0 + C01*g1 + C02*g2,
      C01*g0 + C11*g1 + C12*g2,
      C02*g0 + C12*g1 + C22*g2] / det_H
σ ← σ - Δσ
```

5 fixed Newton iterations. Initialize from SVD singular values σ₀. Clamp σ > 1e-6 each iteration.

- [ ] **Step 1: Write the failing test for NH 3D projection**

Add `TestTetNeoHookean` class to `newton/tests/test_solver_fba.py` after `TestTetCorotational`:

```python
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
        # F=1.5*I → U=V=I, sigma_0=(1.5,1.5,1.5).
        # rhs[1]=(sigma_proj,0,0), rhs[2]=(0,sigma_proj,0), rhs[3]=(0,0,sigma_proj).
        sigma_proj = r[1][0]
        self.assertTrue(np.isfinite(sigma_proj), "sigma_proj is not finite")
        self.assertGreater(sigma_proj, 0.0, "sigma_proj must be positive")
        self.assertLess(sigma_proj, 1.5, "NH should pull sigma back from 1.5 toward 1")

        # All three should be equal by symmetry.
        np.testing.assert_allclose(r[2][1], sigma_proj, atol=1e-5)
        np.testing.assert_allclose(r[3][2], sigma_proj, atol=1e-5)

        # Verify gradient is small at the converged sigma.
        k = 2.0 * mu
        sigma0 = 1.5
        J3 = sigma_proj ** 3
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
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run --extra dev -m newton.tests -k "TestTetNeoHookean"
```

Expected: `AttributeError` or `ImportError` — `project_stretching_neohookean_tet_kernel` does not exist yet.

- [ ] **Step 3: Add `project_neohookean_sigma3d` @wp.func to kernels.py**

In `newton/_src/solvers/fba/kernels.py`, insert after `project_stretching_corotational_tet_kernel` (after Task A additions) and before `project_stretching_arap_kernel`:

```python
@wp.func
def project_neohookean_sigma3d(sigma: wp.vec3, mu: float, lam: float) -> wp.vec3:
    """Project SVD singular values to 3D Neo-Hookean PD equilibrium via 5 Newton iterations.

    Solves: min_σ  E_NH(σ) + (k/2)*||σ - σ₀||²
    where k = 2*mu (PD penalty weight) and
    E_NH(σ) = (μ/2)*(I₁ - 2*log(J) - 3) + (λ/2)*log(J)²
    with I₁ = σ₀² + σ₁² + σ₂², J = σ₀·σ₁·σ₂.

    Matches RealSim NHProjectionProblem3D::energy_density (using log_I3 = 2*log(J),
    so 0.125*λ*log_I3² = (λ/2)*log²(J)). Sign verified against RealSim reference.

    Gradient:
        ∇E_i = μ*(σ_i - 1/σ_i) + λ*log(J)/σ_i + k*(σ_i - σ₀_i)

    Hessian (symmetric 3x3):
        diag_factor = μ + λ - λ*log(J)
        H_ii = μ + diag_factor/σ_i² + k
        H_ij = λ/(σ_i*σ_j)  for i≠j

    Solved via closed-form 3x3 inverse (cofactor matrix / determinant).
    5 fixed Newton iterations. Clamp σ > 1e-6 each iteration.

    Args:
        sigma: Singular values from ``wp.svd3(F)``, shape (3,), in descending order.
        mu: First Lamé parameter (shear modulus) [Pa].
        lam: Second Lamé parameter [Pa].

    Returns:
        Projected sigma triple suitable for reconstructing P = U diag(sigma) V^T.
    """
    eps = 1.0e-6
    k = 2.0 * mu

    # Initialize Newton iterate from SVD singular values.
    s0 = wp.max(sigma[0], eps)
    s1 = wp.max(sigma[1], eps)
    s2 = wp.max(sigma[2], eps)
    # Store initial sigma_0 for penalty term.
    sigma0_0 = s0
    sigma0_1 = s1
    sigma0_2 = s2

    for _i in range(5):
        # Clamp before computing log/inv.
        s0 = wp.max(s0, eps)
        s1 = wp.max(s1, eps)
        s2 = wp.max(s2, eps)

        J = s0 * s1 * s2
        log_J = wp.log(J)
        inv0 = 1.0 / s0
        inv1 = 1.0 / s1
        inv2 = 1.0 / s2

        # Gradient.
        g0 = mu * (s0 - inv0) + lam * log_J * inv0 + k * (s0 - sigma0_0)
        g1 = mu * (s1 - inv1) + lam * log_J * inv1 + k * (s1 - sigma0_1)
        g2 = mu * (s2 - inv2) + lam * log_J * inv2 + k * (s2 - sigma0_2)

        # Hessian diagonal factor.
        diag_factor = mu + lam - lam * log_J
        H00 = mu + diag_factor * inv0 * inv0 + k
        H11 = mu + diag_factor * inv1 * inv1 + k
        H22 = mu + diag_factor * inv2 * inv2 + k
        H01 = lam * inv0 * inv1
        H02 = lam * inv0 * inv2
        H12 = lam * inv1 * inv2

        # Closed-form 3x3 inverse via cofactors.
        C00 = H11 * H22 - H12 * H12
        C11 = H00 * H22 - H02 * H02
        C22 = H00 * H11 - H01 * H01
        C01 = -(H01 * H22 - H12 * H02)
        C02 = H01 * H12 - H11 * H02
        C12 = -(H00 * H12 - H01 * H02)

        det_H = H00 * C00 + H01 * C01 + H02 * C02

        dx0 = (C00 * g0 + C01 * g1 + C02 * g2) / det_H
        dx1 = (C01 * g0 + C11 * g1 + C12 * g2) / det_H
        dx2 = (C02 * g0 + C12 * g1 + C22 * g2) / det_H

        s0 = s0 - dx0
        s1 = s1 - dx1
        s2 = s2 - dx2

    s0 = wp.max(s0, eps)
    s1 = wp.max(s1, eps)
    s2 = wp.max(s2, eps)
    return wp.vec3(s0, s1, s2)
```

- [ ] **Step 4: Add `project_stretching_neohookean_tet_kernel` @wp.kernel to kernels.py**

Immediately after `project_neohookean_sigma3d`, insert:

```python
@wp.kernel
def project_stretching_neohookean_tet_kernel(
    positions: wp.array[wp.vec3],
    tet_indices: wp.array[wp.int32],  # flat shape (4*T,)
    tet_rest_inv: wp.array[wp.mat33],
    tet_weight: wp.array[wp.float32],
    mu: float,
    lam: float,
    # output (atomic accumulator)
    rhs: wp.array[wp.vec3],
):
    """Per-tet Neo-Hookean local projection scatter for PD softbody.

    Computes F = Ds * Dm_inv (3x3), extracts singular values via ``wp.svd3``,
    applies 5-iteration Newton solve of the 3D Neo-Hookean PD projection to get
    ``sigma_proj``, reconstructs ``P = U * diag(sigma_proj) * Vt``, and scatters
    ``w * Dm_inv * P^T`` into the four stencil vertices via atomic_add.

    No reflection handling is needed: ``wp.svd3`` returns non-negative singular
    values, and the NH projection preserves positivity; the reconstructed P
    naturally has the correct orientation.

    Args:
        positions: Current particle positions [m], shape ``[particle_count]``.
        tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
        tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
        tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
        mu: First Lamé parameter [Pa].
        lam: Second Lamé parameter [Pa].
        rhs: Output RHS accumulator (atomic-add target), shape ``[particle_count]``.
    """
    t = wp.tid()
    i0 = tet_indices[4 * t + 0]
    i1 = tet_indices[4 * t + 1]
    i2 = tet_indices[4 * t + 2]
    i3 = tet_indices[4 * t + 3]

    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]

    # Ds = [p1-p0 | p2-p0 | p3-p0]  (column-stack 3 edge vectors -> 3x3)
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Ds = wp.mat33(
        e1[0], e2[0], e3[0],
        e1[1], e2[1], e3[1],
        e1[2], e2[2], e3[2],
    )
    Dm_inv = tet_rest_inv[t]
    F = Ds * Dm_inv

    U, sigma, V = wp.svd3(F)

    # Neo-Hookean projection: 5-iter Newton solve on singular values.
    sigma_proj = project_neohookean_sigma3d(wp.vec3(sigma[0], sigma[1], sigma[2]), mu, lam)

    # Reconstruct P = U * diag(sigma_proj) * V^T  (3x3).
    s0 = sigma_proj[0]
    s1 = sigma_proj[1]
    s2 = sigma_proj[2]
    US = wp.mat33(
        U[0, 0] * s0, U[0, 1] * s1, U[0, 2] * s2,
        U[1, 0] * s0, U[1, 1] * s1, U[1, 2] * s2,
        U[2, 0] * s0, U[2, 1] * s1, U[2, 2] * s2,
    )
    P = US * wp.transpose(V)

    # proj = w * Dm_inv * P^T  (3x3)
    w = tet_weight[t]
    PT = wp.transpose(P)
    proj = w * (Dm_inv * PT)

    # Scatter stencil.
    row0 = wp.vec3(proj[0, 0], proj[0, 1], proj[0, 2])
    row1 = wp.vec3(proj[1, 0], proj[1, 1], proj[1, 2])
    row2 = wp.vec3(proj[2, 0], proj[2, 1], proj[2, 2])
    wp.atomic_add(rhs, i0, -(row0 + row1 + row2))
    wp.atomic_add(rhs, i1, row0)
    wp.atomic_add(rhs, i2, row1)
    wp.atomic_add(rhs, i3, row2)
```

- [ ] **Step 5: Wire NH tet dispatch in solver_fba.py**

In `newton/_src/solvers/fba/solver_fba.py`:

1. Add `project_stretching_neohookean_tet_kernel` to the import block in `step()`:

```python
        from .kernels import (  # noqa: PLC0415
            add_inertia_to_rhs_kernel,
            compute_inertial_kernel,
            project_bending_kernel,
            project_pin_kernel,
            project_stretching_arap_kernel,
            project_stretching_arap_tet_kernel,
            project_stretching_corotational_kernel,
            project_stretching_corotational_tet_kernel,
            project_stretching_neohookean_kernel,
            project_stretching_neohookean_tet_kernel,
            write_velocity_kernel,
            zero_vec3_kernel,
        )
```

2. Replace the `elif self.stretching_model == "neohookean": raise NotImplementedError(...)` in the tet dispatch with:

```python
                elif self.stretching_model == "neohookean":
                    wp.launch(
                        project_stretching_neohookean_tet_kernel,
                        dim=model.tet_count,
                        inputs=[
                            self._x_cur,
                            self._tet_indices_d,
                            self._tet_rest_inv_d,
                            self._tet_weight_d,
                            self._mu,
                            self._lam,
                        ],
                        outputs=[self._rhs],
                        device=device,
                    )
```

- [ ] **Step 6: Run all tests**

```bash
uv run --extra dev -m newton.tests
```

Expected: 47 + 4 + 4 = 55 tests pass. (4 new Corot tet + 4 new NH tet; docstring note: the spec says 52 but we add 4+4=8 new tests.)

- [ ] **Step 7: Lint and format**

```bash
uvx pre-commit run -a
```

- [ ] **Step 8: Commit**

```bash
git add newton/_src/solvers/fba/kernels.py newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
git commit -m "$(cat <<'EOF'
Add project_neohookean_sigma3d + tet NH kernel

Implements 5-iter Newton 3D Neo-Hookean PD projection with closed-form
3x3 Hessian inverse via cofactors. Wires the new kernel into
SolverFBA.step() tet dispatch, completing tet Corot/NH support.
EOF
)"
```

---

### Task C: Extend benchmark script to cover tet ARAP/Corot/NH on 8×8×8 cube

**Files:**
- Modify: `scripts/fba_arap_vs_corot_bench.py`

- [ ] **Step 1: Add tet benchmark section to the existing bench script**

The existing script benchmarks cloth. We add a tet section that builds an 8×8×8 soft grid and benchmarks all three models. Extend `scripts/fba_arap_vs_corot_bench.py`:

After the existing `build_cloth` function and before `run_bench`, add:

```python
TET_DIM = 8  # 8x8x8 cube
NH_LAM = 14286.0  # E*nu / ((1+nu)(1-2nu)) for E=1e4, nu=0.4


def build_softbody(dim: int = TET_DIM) -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.1),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim,
        dim_y=dim,
        dim_z=dim,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1.0e3,
        k_mu=MU,
        k_lambda=MU,
        k_damp=0.0,
        fix_left=True,
    )
    return builder.finalize()
```

And in `main()`, after the cloth benchmark section, add:

```python
    print()
    print("=" * 70)
    print(f"TET SOFTBODY BENCHMARK — {TET_DIM}x{TET_DIM}x{TET_DIM} grid")
    print("=" * 70)
    model_tet_arap = build_softbody()
    model_tet_corot = build_softbody()
    model_tet_nh = build_softbody()

    from newton.solvers import SolverFBA as _SolverFBA  # noqa: PLC0415

    solver_tet_arap = _SolverFBA(model_tet_arap, iterations=10, stretching_model="arap")
    solver_tet_corot = _SolverFBA(
        model_tet_corot, iterations=10, stretching_model="corotational", mu=MU, lam=MU
    )
    solver_tet_nh = _SolverFBA(
        model_tet_nh, iterations=10, stretching_model="neohookean", mu=MU, lam=NH_LAM
    )

    print("Priming tet solvers...")
    dt = 1.0 / 60.0
    for solver, model in [
        (solver_tet_arap, model_tet_arap),
        (solver_tet_corot, model_tet_corot),
        (solver_tet_nh, model_tet_nh),
    ]:
        s_in, s_out = model.state(), model.state()
        s_in.clear_forces()
        solver.step(s_in, s_out, None, None, dt)
    print()

    print("--- Tet ARAP ---")
    times_tet_arap = run_bench(model_tet_arap, solver_tet_arap, "Tet-ARAP")
    print()
    print("--- Tet Corotational ---")
    times_tet_corot = run_bench(model_tet_corot, solver_tet_corot, "Tet-Corot")
    print()
    print("--- Tet Neo-Hookean ---")
    times_tet_nh = run_bench(model_tet_nh, solver_tet_nh, "Tet-NH")
    print()

    ms_tet_arap = times_tet_arap * 1000.0
    ms_tet_corot = times_tet_corot * 1000.0
    ms_tet_nh = times_tet_nh * 1000.0

    print("=" * 70)
    print("TET RESULTS")
    print("=" * 70)
    stats(ms_tet_arap, "Tet-ARAP")
    stats(ms_tet_corot, "Tet-Corot")
    stats(ms_tet_nh, "Tet-NH")
    print()
    for label, ms in [("Tet-Corot", ms_tet_corot), ("Tet-NH", ms_tet_nh)]:
        ratio = ms.mean() / ms_tet_arap.mean()
        overhead_pct = (ratio - 1.0) * 100.0
        print(f"  {label:10s} / Tet-ARAP ratio: {ratio:.3f}x  ({overhead_pct:+.1f}% overhead)")
    print("=" * 70)
```

- [ ] **Step 2: Run the benchmark (verify it runs without error)**

```bash
uv run scripts/fba_arap_vs_corot_bench.py
```

Expected: outputs cloth + tet benchmark results in ms. No NaN assertions triggered.

- [ ] **Step 3: Run full test suite one final time**

```bash
uv run --extra dev -m newton.tests
```

Expected: 55 tests pass (47 original + 4 Corot tet + 4 NH tet).

- [ ] **Step 4: Lint and format**

```bash
uvx pre-commit run -a
```

- [ ] **Step 5: Commit**

```bash
git add scripts/fba_arap_vs_corot_bench.py
git commit -m "$(cat <<'EOF'
Extend FBA bench to cover tet ARAP/Corot/NH on 8x8x8 cube

Adds tet softbody benchmark section alongside existing cloth section.
EOF
)"
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task Covering It |
|---|---|
| `project_corotational_sigma3d` closed-form 3D | Task A Step 3 |
| `project_stretching_corotational_tet_kernel` | Task A Step 4 |
| `project_neohookean_sigma3d` 5-iter Newton | Task B Step 3 |
| `project_stretching_neohookean_tet_kernel` | Task B Step 4 |
| Tet dispatch in `step()` wired | Task A Step 7, Task B Step 5 |
| `test_rest_tet_corot_projects_to_identity` | Task A Step 1 |
| `test_isotropic_stretch_corot_3d` (≈1.18182) | Task A Step 1 |
| `test_solver_corot_tet_softbody_runs` | Task A Step 1 |
| `test_rest_tet_nh_projects_to_identity` | Task B Step 1 |
| `test_isotropic_stretch_nh_3d` | Task B Step 1 |
| `test_solver_nh_tet_softbody_runs` | Task B Step 1 |
| Performance benchmark tet Corot/NH | Task C |
| Standard symmetric trace (not RealSim .trace() bug) | Task A Step 3 docstring |
| No new dependencies | N/A (uses existing Warp) |
| Existing 47 tests continue to pass | All tasks include full test run |

### Type / Signature Consistency Check

- `project_corotational_sigma3d(sigma: wp.vec3, mu: float, lam: float) -> wp.vec3` — used consistently in all call sites.
- `project_neohookean_sigma3d(sigma: wp.vec3, mu: float, lam: float) -> wp.vec3` — used consistently.
- Both tet kernels have identical signatures to each other (matching the NH cloth kernel with added `mu`/`lam` args and `tet_rest_inv` as `wp.mat33` instead of `wp.mat22`).
- Kernel call sites in `solver_fba.py` pass `self._mu`, `self._lam` — both already stored as float (since cloth Corot/NH already required them).

### Placeholder Scan

No TBD/TODO/placeholder text present. All code blocks are complete and self-contained.

### Reflection Handling Note

Both tet Corot and NH kernels use `P = U * diag(sigma_proj) * V^T` without reflection handling. This is correct because:
1. `wp.svd3` returns non-negative singular values by definition.
2. For Corot/NH, the projection converges to positive sigma values.
3. Reflection (det(U)*det(V) < 0) changes the orientation of U and V together, but since we reconstruct `P = U * diag(sigma_proj) * V^T` with the same U and V from SVD, the result automatically has the correct sign structure.
4. Only ARAP needs explicit reflection handling because it forcibly sets sigma to (1,1,1), discarding the sign information; Corot/NH keep sigma as a free variable that absorbs the sign.

This matches how RealSim's PDCorotationalTriangleEnergyParallel handles the 2D case (no explicit reflection check).
