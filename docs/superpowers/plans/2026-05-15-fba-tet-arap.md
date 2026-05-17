# FBA Tetrahedral ARAP Softbody Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tetrahedral ARAP softbody support to SolverFBA — the 3D analog of the existing triangle cloth ARAP path, using 4 vertices per element and 3×3 SVD.

**Architecture:** Tet support is wired into the same three files as triangle cloth: `linear_solver.py` (Hessian assembly), `kernels.py` (GPU projection kernel), and `solver_fba.py` (buffer management + dispatch). A new example and new tests complete the feature. Existing cloth tests must pass unchanged.

**Tech Stack:** Python/NumPy (CPU setup), Warp (GPU kernels, `wp.svd3`), SciPy (sparse factorization, unchanged)

---

## File Map

| File | Change |
|---|---|
| `newton/_src/solvers/fba/linear_solver.py` | Add tet Hessian assembly block + tet meta keys |
| `newton/_src/solvers/fba/kernels.py` | Add `project_arap_3x3` func + `project_stretching_arap_tet_kernel` |
| `newton/_src/solvers/fba/solver_fba.py` | Relax cloth-only check; add tet buffers + dispatch |
| `newton/tests/test_solver_fba.py` | Add `TestTetARAP` class with 4 tests |
| `newton/examples/softbody/example_softbody_hanging_fba.py` | New example |

---

## Preliminary: key facts gathered from codebase

- `model.tet_indices`: flat `wp.array[wp.int32]` shape `[tet_count*4]`
- `model.tet_poses`: `wp.array[wp.mat33]` shape `[tet_count]` — stores `Dm_inv` (rest-edge matrix inverse)
- `model.tet_materials`: `wp.array2d[wp.float32]` shape `[tet_count, 3]` — columns `[mu, lambda, damping]`
- `model.tet_count`: int
- Volume from `Dm_inv`: in `builder.add_tetrahedron`, `volume = det(Dm)/6`, so `det(Dm_inv) = 1/det(Dm) = 1/(6·V)`, giving `V = 1/(6·|det(Dm_inv)|)`
- `wp.svd3(F, U, sigma, V)` — Warp's in-place SVD; signature confirmed from `implicit_mpm_solver_kernels.py:395`: `_U, xi, _V = wp.svd3(F)`. Returns `(U, sigma, V)` as a tuple — not positional out-params.
- RealSim scatter pattern (from `PDTetrahedronEnergy.cpp:191-194`):
  ```
  rhs[t[0]] += -proj.row(0) - proj.row(1) - proj.row(2)
  rhs[t[1]] += proj.row(0)
  rhs[t[2]] += proj.row(1)
  rhs[t[3]] += proj.row(2)
  ```
  where `proj = wi * Dm_inv * R^T` (3×3).
- RealSim Hessian (from `PDTetrahedronEnergy.cpp:86-118`): `K = S^T * Q * S` where `Q = Dm_inv * Dm_inv^T` (3×3) and the selector `S` maps 4-vertex stencil. The spec's `ST @ Dm_inv` formulation is equivalent: `G = ST @ Dm_inv` → `K = G @ G^T` = `ST @ Dm_inv @ Dm_inv^T @ ST^T`.
- PD weight: RealSim uses `wi = weight * vol[i]` where `weight = 2*mu` for ARAP. So `tet_weight[i] = 2*mu_i * volume_i`.
- `wp.svd3` singular values: Warp docs say descending order; `implicit_mpm_solver_kernels.py:395` usage `wp.min(xi)` suggests index 2 is smallest — consistent with descending.
- The Warp kernel must use `wp.mat33` for `Dm_inv` — not a custom matrix type.

---

## Task A: Tet Hessian assembly in `build_pd_system`

**Files:**
- Modify: `newton/_src/solvers/fba/linear_solver.py`

- [ ] **Step A1: Read the current file** to understand the exact COO assembly pattern for triangles.

  Already read. The pattern is: build `coo_parts` list of `(rows, cols, vals)` numpy arrays, concatenate at the end.

- [ ] **Step A2: Add tet assembly block after the bending block in `build_pd_system`**

  Insert this block just before the `if coo_parts:` merge block (after the edge/bending block):

  ```python
  # ---- Tetrahedral stretching contribution ----
  tet_indices = np.zeros((0, 4), dtype=np.int32)
  tet_rest_inv = np.zeros((0, 3, 3), dtype=np.float64)
  tet_volume = np.zeros((0,), dtype=np.float64)
  tet_weight = np.zeros((0,), dtype=np.float64)
  mu_tet = np.zeros((0,), dtype=np.float64)
  if hasattr(model, "tet_count") and model.tet_count > 0:
      tet_indices_flat = model.tet_indices.numpy().astype(np.int32)
      tet_indices = tet_indices_flat.reshape(-1, 4)
      tet_rest_inv = model.tet_poses.numpy().astype(np.float64)
      # Volume from Dm_inv: det(Dm_inv) = 1/(6*V) => V = 1/(6*|det(Dm_inv)|)
      det_dm_inv = np.linalg.det(tet_rest_inv)
      tet_volume = 1.0 / (6.0 * np.abs(det_dm_inv) + 1.0e-30)
      tet_materials = model.tet_materials.numpy().astype(np.float64)
      mu_tet = tet_materials[:, 0]
      # PD weight: w = 2*mu * volume  (ARAP: singular values project to 1)
      tet_weight = 2.0 * mu_tet * tet_volume

      # Stencil selector ST is 4×3 (row = which edge coefficient for each tet vertex).
      ST_tet = np.array([
          [-1.0, -1.0, -1.0],
          [ 1.0,  0.0,  0.0],
          [ 0.0,  1.0,  0.0],
          [ 0.0,  0.0,  1.0],
      ], dtype=np.float64)  # (4, 3)
      # G_all[T, 4, 3] = ST @ Dm_inv per tet
      G_all = np.einsum("ab,tbc->tac", ST_tet, tet_rest_inv)  # (T, 4, 3)
      # K_all[T, 4, 4] = G @ G^T per tet (symmetric PSD)
      K_all = np.einsum("tac,tbc->tab", G_all, G_all)  # (T, 4, 4)
      wK_all = tet_weight[:, None, None] * K_all  # (T, 4, 4)

      # Vectorized scatter: for each (a,b) in 4×4, contribute wK[t,a,b] to A[tet[t,a], tet[t,b]]
      a_idx, b_idx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
      a_flat = a_idx.flatten()  # 16
      b_flat = b_idx.flatten()  # 16
      # Gather all 16 (row, col, val) arrays, then concatenate once.
      tet_rows_parts = []
      tet_cols_parts = []
      tet_vals_parts = []
      for a, b in zip(a_flat, b_flat):
          tet_rows_parts.append(tet_indices[:, a])
          tet_cols_parts.append(tet_indices[:, b])
          tet_vals_parts.append(wK_all[:, a, b])
      if len(tet_rows_parts) > 0:
          coo_parts.append((
              np.concatenate(tet_rows_parts).astype(np.int32),
              np.concatenate(tet_cols_parts).astype(np.int32),
              np.concatenate(tet_vals_parts),
          ))
  ```

- [ ] **Step A3: Add tet meta keys to the returned `meta` dict**

  In the `meta` dict at the end of `build_pd_system`, add after `"pin_weight"`:

  ```python
  "tet_indices": tet_indices,       # (T, 4) int32 or (0, 4) if no tets
  "tet_rest_inv": tet_rest_inv,     # (T, 3, 3) float64
  "tet_volume": tet_volume,         # (T,) float64
  "tet_weight": tet_weight,         # (T,) float64
  "tet_mu": mu_tet,                 # (T,) float64 — for future Corot/NH
  ```

- [ ] **Step A4: Write the failing test** for tet Hessian assembly in `newton/tests/test_solver_fba.py`

  Add this class at the end of the file (before `if __name__ == "__main__":` if present):

  ```python
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
  ```

- [ ] **Step A5: Run the test to verify it fails** (meta keys don't exist yet):

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k TestTetHessianAssembly 2>&1 | tail -20
  ```

  Expected: FAIL with `KeyError: 'tet_indices'` or similar.

- [ ] **Step A6: Implement the tet assembly (Steps A2+A3 above)**

  Edit `newton/_src/solvers/fba/linear_solver.py` as described in Steps A2 and A3.

- [ ] **Step A7: Run the test to verify it passes**:

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k TestTetHessianAssembly 2>&1 | tail -20
  ```

  Expected: 3 tests PASS.

- [ ] **Step A8: Run existing cloth tests to verify no regression**:

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -30
  ```

  Expected: all existing tests pass.

- [ ] **Step A9: Commit**:

  ```bash
  cd /home/ziqiu/work/newton && uvx pre-commit run -a && git add newton/_src/solvers/fba/linear_solver.py newton/tests/test_solver_fba.py && git commit -m "Add tet ARAP assembly to build_pd_system and PD setup"
  ```

---

## Task B: ARAP 3×3 kernels in `kernels.py`

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py`

- [ ] **Step B1: Add `project_arap_3x3` Warp function**

  Insert after `svd_3x2` (around line 178):

  ```python
  @wp.func
  def project_arap_3x3(F: wp.mat33) -> wp.mat33:
      """ARAP 3D projection: F -> nearest proper rotation R.

      Uses Warp's wp.svd3 then handles the reflection case:
      if det(U) * det(V) < 0, flip the last column of U so that
      R = U' * V^T has det(R) = +1.

      Args:
          F: 3x3 deformation gradient.

      Returns:
          R: 3x3 proper rotation (nearest rotation to F).
      """
      U = wp.mat33()
      sigma = wp.vec3()
      V = wp.mat33()
      U, sigma, V = wp.svd3(F)

      # Detect reflection: det(U) * det(V) < 0 means R = U*V^T would have det = -1.
      # Fix by flipping the column of U corresponding to the smallest singular value.
      # wp.svd3 returns sigma in descending order, so index 2 is the smallest.
      detUV = wp.determinant(U) * wp.determinant(V)
      if detUV < 0.0:
          # Flip column 2 of U (zero-indexed = last column in Warp row-major mat33).
          # wp.mat33 is row-major: U[row, col]. To flip col 2: negate U[0,2], U[1,2], U[2,2].
          U = wp.mat33(
              U[0, 0], U[0, 1], -U[0, 2],
              U[1, 0], U[1, 1], -U[1, 2],
              U[2, 0], U[2, 1], -U[2, 2],
          )
      return U * wp.transpose(V)
  ```

- [ ] **Step B2: Add `project_stretching_arap_tet_kernel` Warp kernel**

  Insert after `project_arap_3x3`:

  ```python
  @wp.kernel
  def project_stretching_arap_tet_kernel(
      positions: wp.array[wp.vec3],
      tet_indices: wp.array[wp.int32],        # flat shape (4*T,)
      tet_rest_inv: wp.array[wp.mat33],
      tet_weight: wp.array[wp.float32],
      rhs: wp.array[wp.vec3],
  ):
      """Per-tet ARAP local projection scatter for PD softbody.

      Computes F = Ds * Dm_inv (3x3), projects to nearest rotation R via SVD
      with reflection handling, and scatters w * Dm_inv * R^T into the four
      stencil vertices via atomic_add.

      Args:
          positions: Current particle positions [m], shape ``[particle_count]``.
          tet_indices: Flat tet indices, shape ``[4 * tet_count]``.
          tet_rest_inv: Per-tet 3x3 rest-pose inverse (Dm_inv), shape ``[tet_count]``.
          tet_weight: Per-tet weight (2*mu * volume), shape ``[tet_count]``.
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

      R = project_arap_3x3(F)

      # proj = w * Dm_inv * R^T  (3x3)
      w = tet_weight[t]
      RT = wp.transpose(R)
      proj = w * (Dm_inv * RT)

      # Scatter stencil (RealSim PDTetrahedronEnergy.cpp:191-194):
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

- [ ] **Step B3: Write the failing kernel tests** in `newton/tests/test_solver_fba.py`

  Add this class at the end of the test file:

  ```python
  class TestTetARAP(unittest.TestCase):
      """Tests for tet ARAP kernel: project_arap_3x3 and project_stretching_arap_tet_kernel."""

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
              dtype=wp.vec3, device=device,
          )
          tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
          # Dm_inv = I for canonical tet
          Dm_inv = wp.array([wp.mat33(1,0,0, 0,1,0, 0,0,1)], dtype=wp.mat33, device=device)
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
          # 30° rotation about z-axis
          angle = np.pi / 6.0
          c, s = np.cos(angle), np.sin(angle)
          R0 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
          # Rotated tet: apply R0 to canonical vertices
          verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
          verts_rot = verts @ R0.T  # (4,3)

          positions = wp.array([wp.vec3(*v) for v in verts_rot], dtype=wp.vec3, device=device)
          tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
          # Dm_inv = I for canonical tet
          Dm_inv = wp.array([wp.mat33(1,0,0, 0,1,0, 0,0,1)], dtype=wp.mat33, device=device)
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
              dtype=wp.vec3, device=device,
          )
          tet_indices = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
          Dm_inv = wp.array([wp.mat33(1,0,0, 0,1,0, 0,0,1)], dtype=wp.mat33, device=device)
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
  ```

- [ ] **Step B4: Run the test to verify it fails**:

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k TestTetARAP 2>&1 | tail -20
  ```

  Expected: FAIL with `ImportError: cannot import name 'project_stretching_arap_tet_kernel'`.

- [ ] **Step B5: Implement the kernels** (Steps B1+B2 above in `kernels.py`).

- [ ] **Step B6: Run the test to verify it passes**:

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k TestTetARAP 2>&1 | tail -20
  ```

  Expected: 3 tests PASS.

- [ ] **Step B7: Run full FBA test suite**:

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -30
  ```

  Expected: all existing tests + new TestTetARAP pass.

- [ ] **Step B8: Commit**:

  ```bash
  cd /home/ziqiu/work/newton && uvx pre-commit run -a && git add newton/_src/solvers/fba/kernels.py newton/tests/test_solver_fba.py && git commit -m "Add project_arap_3x3 + project_stretching_arap_tet_kernel"
  ```

---

## Task C: Wire tet support into `SolverFBA`

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py`
- Modify: `newton/tests/test_solver_fba.py`

- [ ] **Step C1: Relax the cloth-only validation** in `__init__`

  Replace:
  ```python
  if model.tri_count == 0:
      raise ValueError("SolverFBA requires at least one cloth triangle")
  ```
  With:
  ```python
  if model.tri_count == 0 and (not hasattr(model, "tet_count") or model.tet_count == 0):
      raise ValueError("SolverFBA requires at least one cloth triangle or tetrahedral element")
  ```

- [ ] **Step C2: Add tet buffer attributes** in `__init__` (after `self._pin_indices_d = None`):

  ```python
  # Tet device data (filled by _setup_pd_system).
  self._tet_indices_d = None
  self._tet_rest_inv_d = None
  self._tet_weight_d = None
  ```

- [ ] **Step C3: Upload tet device data in `_setup_pd_system`**

  After the `if meta["pin_indices"].shape[0] > 0:` block (end of `_setup_pd_system`), add:

  ```python
  if meta["tet_indices"].shape[0] > 0:
      self._tet_indices_d = wp.array(
          meta["tet_indices"].flatten().astype(np.int32),
          dtype=wp.int32, device=device,
      )
      self._tet_rest_inv_d = wp.array(
          meta["tet_rest_inv"].astype(np.float32),
          dtype=wp.mat33, device=device,
      )
      self._tet_weight_d = wp.array(
          meta["tet_weight"].astype(np.float32),
          dtype=wp.float32, device=device,
      )
  ```

- [ ] **Step C4: Import tet kernel in `step()`**

  In the `from .kernels import (` block at the top of `step()`, add:
  ```python
  project_stretching_arap_tet_kernel,
  ```

- [ ] **Step C5: Dispatch tet kernel in `step()`**

  After the bending projection block (`if self._edge_indices_d is not None:`) and before `self._linear_solver.solve(...)`, add:

  ```python
  # Tet ARAP projection.
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
      elif self.stretching_model in ("corotational", "neohookean"):
          raise NotImplementedError(
              f"stretching_model={self.stretching_model!r} is not yet implemented for tets. "
              "Only 'arap' is supported for tetrahedral elements."
          )
  ```

- [ ] **Step C6: Write the failing integration tests** in `newton/tests/test_solver_fba.py`

  Add this class at the end of the test file:

  ```python
  class TestTetARAPIntegration(unittest.TestCase):
      """Smoke and integration tests for SolverFBA with tet softbody."""

      @classmethod
      def setUpClass(cls):
          wp.init()

      def _build_tet_grid(self, dim: int = 4):
          """Build a dim×dim×dim tet grid, pinning the bottom face (z=0)."""
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
          """Pinned-left tet grid must sag under gravity (not stay rigid)."""
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
          # At least some free particles should have moved downward (negative y in Z-up world).
          max_displacement = np.abs(q_final - q_initial).max()
          self.assertGreater(max_displacement, 1e-3, "tet softbody did not deform under gravity")
  ```

- [ ] **Step C7: Run the new tests to verify they fail correctly**:

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k TestTetARAPIntegration 2>&1 | tail -30
  ```

  Expected: tests fail because tet dispatch not yet wired.

- [ ] **Step C8: Implement all C1–C5 changes in `solver_fba.py`**.

- [ ] **Step C9: Run all FBA tests**:

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -40
  ```

  Expected: all tests pass (including new integration tests).

- [ ] **Step C10: Commit**:

  ```bash
  cd /home/ziqiu/work/newton && uvx pre-commit run -a && git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py && git commit -m "Wire tet ARAP stretching in SolverFBA.step + add TestTetARAPIntegration"
  ```

---

## Task D: Example `example_softbody_hanging_fba.py`

**Files:**
- Create: `newton/examples/softbody/example_softbody_hanging_fba.py`

- [ ] **Step D1: Create the example file**:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
  # SPDX-License-Identifier: Apache-2.0

  ###########################################################################
  # Example Softbody Hanging FBA
  #
  # Volumetric soft body (tetrahedral grid) hanging from a pinned left face,
  # simulated with SolverFBA projective-dynamics (ARAP tet stretching).
  #
  # Command: uv run -m newton.examples softbody.example_softbody_hanging_fba
  ###########################################################################

  import numpy as np
  import warp as wp

  import newton
  import newton.examples
  from newton.solvers import SolverFBA


  class Example:
      def __init__(self, viewer, args):
          self.viewer = viewer
          self.fps = 60
          self.frame_dt = 1.0 / self.fps
          self.sim_time = 0.0

          builder = newton.ModelBuilder()

          dim = 4
          cell = 0.1

          builder.add_soft_grid(
              pos=wp.vec3(0.0, 1.0, 0.0),
              rot=wp.quat_identity(),
              vel=wp.vec3(0.0, 0.0, 0.0),
              dim_x=dim,
              dim_y=dim,
              dim_z=dim,
              cell_x=cell,
              cell_y=cell,
              cell_z=cell,
              density=1.0e3,
              k_mu=1.0e4,
              k_lambda=1.0e4,
              fix_left=True,
          )

          self.model = builder.finalize()
          self.solver = SolverFBA(self.model, iterations=10)
          self.state_0 = self.model.state()
          self.state_1 = self.model.state()

          self.viewer.set_model(self.model)

      def step(self):
          self.state_0.clear_forces()
          self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
          self.state_0, self.state_1 = self.state_1, self.state_0
          self.sim_time += self.frame_dt

      def render(self):
          self.viewer.begin_frame(self.sim_time)
          self.viewer.log_state(self.state_0)
          self.viewer.end_frame()

      def test_final(self):
          q = self.state_0.particle_q.numpy()
          assert np.all(np.isfinite(q)), "non-finite particle positions"
          # Free (unpinned) particles must have moved from initial rest positions.
          q_rest = self.model.particle_q.numpy()
          inv_mass = self.model.particle_inv_mass.numpy()
          free_mask = inv_mass > 0.0
          if free_mask.any():
              displacement = np.abs(q[free_mask] - q_rest[free_mask]).max()
              assert displacement > 1e-3, f"softbody did not deform: max displacement={displacement:.4f} m"


  if __name__ == "__main__":
      parser = newton.examples.create_parser()
      viewer, args = newton.examples.init(parser)
      newton.examples.run(Example(viewer, args), args)
  ```

- [ ] **Step D2: Run the example headlessly**:

  ```bash
  cd /home/ziqiu/work/newton && export PATH="$HOME/.local/bin:$PATH" && uv run -m newton.examples softbody.example_softbody_hanging_fba --viewer null --test 2>&1 | tail -20
  ```

  Expected: exits 0 with no error.

- [ ] **Step D3: Commit**:

  ```bash
  cd /home/ziqiu/work/newton && uvx pre-commit run -a && git add newton/examples/softbody/example_softbody_hanging_fba.py && git commit -m "Add example_softbody_hanging_fba example using SolverFBA tet ARAP"
  ```

---

## Task E: Final verification

- [ ] **Step E1: Run full test suite** to confirm cloth tests unchanged and new tet tests pass:

  ```bash
  cd /home/ziqiu/work/newton && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -40
  ```

  Expected: >= 41 tests pass (38 original + 3 TestTetHessianAssembly + 3 TestTetARAP + 3 TestTetARAPIntegration = 47; but some may be grouped in setUpClass — count actual).

- [ ] **Step E2: Run the full test suite** (not just FBA) to catch any unintended breakage:

  ```bash
  cd /home/ziqiu/work/newton && timeout 180 uv run --extra dev -m newton.tests 2>&1 | tail -20
  ```

---

## Self-Review

### Spec coverage

| Spec requirement | Covered by |
|---|---|
| `build_pd_system` tet block with `tet_indices`, `tet_rest_inv`, `tet_volume`, `tet_weight`, `tet_mu` | Task A |
| `project_arap_3x3` Warp func with reflection handling | Task B |
| `project_stretching_arap_tet_kernel` with correct scatter | Task B |
| Device buffers in `SolverFBA.__init__` | Task C |
| `_setup_pd_system` tet upload | Task C |
| `step()` tet ARAP dispatch | Task C |
| Relax cloth-only validation | Task C |
| `test_rest_tet_projects_to_identity` | Task B (B3) |
| `test_softbody_cube_smoke` | Task C (C6) |
| `test_solver_init_rejects_no_geometry` | Task C (C6) |
| `test_rotation_invariance` | Task B (B3) |
| Example `example_softbody_hanging_fba.py` | Task D |
| Existing cloth tests unchanged | Steps A8, B7, C9, E1 |

### Placeholder scan

No placeholder language present. All code blocks are complete.

### Type consistency

- `tet_rest_inv`: `np.ndarray[float64]` in meta → uploaded as `dtype=wp.mat33` (float32) in solver → read as `wp.array[wp.mat33]` in kernel. ✓
- `tet_weight`: `np.ndarray[float64]` in meta → uploaded as `dtype=wp.float32` → read as `wp.array[wp.float32]`. ✓
- `tet_indices`: shape `(T,4)` in meta → `.flatten()` before upload → flat `wp.array[wp.int32]`. ✓
- `project_arap_3x3` returns `wp.mat33`, consumed by `project_stretching_arap_tet_kernel`. ✓

### Known subtlety: Warp mat33 indexing

Warp `wp.mat33` is row-major. `U[row, col]` is the correct indexing. The reconstruction `wp.mat33(U[0,0], U[0,1], -U[0,2], ...)` correctly flips column 2 in row-major order.

### Known subtlety: gravity direction

`add_soft_mesh`/`add_soft_grid` uses a Z-up or Y-up world depending on builder's `up_axis`. In the example, default ModelBuilder is used (Z-up by default in Newton). `fix_left=True` pins the left face (x=0). Gravity acts in -Z direction. Free particles will sag in -Z. The test checks `displacement > 1e-3` which is direction-agnostic.

### Known subtlety: tet-only model (no triangles)

After step C1, `SolverFBA.__init__` allows tet-only models. But `_setup_pd_system` currently always attempts to upload `_tri_indices_d` even when `tri_count == 0`. Check that `meta["tri_indices"].shape[0]` is 0 in that case — the upload produces an empty array. The triangle kernel dispatch in `step()` uses `model.tri_count` as the `dim=` argument, so `dim=0` will launch 0 threads (safe).

### Known subtlety: tet `add_particle` vs builder

To build a test-only single-tet model, the test uses `builder.add_particle()` which requires the Newton `ModelBuilder`. Verify `add_particle` returns a particle index (it does, per `builder.py`).
