# FBA Contact Phase 4 Stage B — Coulomb Friction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend SolverFBA's hard-constraint contact pipeline to support Coulomb friction via NonSmooth Newton + blocked projected Gauss-Seidel on a 3M×3M Schur complement (1 normal + 2 tangent rows per contact).

**Architecture:** Build on the existing Stage A Schur-complement path (`build_schur_complement` in `linear_solver.py`); extend it to emit a 3M×3M `W` by adding two tangent-direction Jacobian rows per active contact. Add a `project_coulomb_cone` Python function and a blocked 3×3 Gauss-Seidel loop in `solver_fba.py`. Compute per-contact friction μ via VBD-style `sqrt(particle_mu × shape_mu)` mixing (with an optional per-pair override array). Add a `friction: bool` flag so Stage A behavior is preserved exactly when `friction=False`.

**Tech Stack:** Python, NumPy, Warp (GPU BSR SpMV via `wps.bsr_mv`), existing `FBALinearSolver` infrastructure.

---

## File Map

| File | Change |
|---|---|
| `newton/_src/solvers/fba/kernels.py` | Add `build_contact_jacobian_dir_kernel` (general direction, for tangent rows) |
| `newton/_src/solvers/fba/linear_solver.py` | Extend `build_schur_complement` to accept optional tangent arrays and emit 3M×3M W |
| `newton/_src/solvers/fba/solver_fba.py` | Add `friction` + `mu_per_pair_override` params; extend `_ensure_contact_buffers`, `update_contacts`, `step` |
| `newton/tests/test_solver_fba.py` | Add `TestPhase4StageBFriction` class with 5 tests |

---

## Task 1: Add generalized Jacobian-direction kernel to `kernels.py`

**Context:** Stage A uses `build_contact_jacobian_vec3_kernel`, which hard-codes "the direction is the contact normal from `j_normals`". For Stage B we need to set a Jacobian column for an *arbitrary* direction (normal, t1, or t2). Rather than three separate kernels, add one that takes the direction explicitly.

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py` (append after line 1133, before `set_lambda_jacobian_vec3_kernel`)

- [ ] **Step 1: Read kernels.py around line 1106 to understand existing kernel signature**

  Read `newton/_src/solvers/fba/kernels.py` lines 1106–1133 (already done in planning; confirming no changes were made).

- [ ] **Step 2: Append `build_contact_jacobian_dir_kernel` to `kernels.py`**

  Open `newton/_src/solvers/fba/kernels.py` and append after the closing brace of `build_contact_jacobian_vec3_kernel` (before `set_lambda_jacobian_vec3_kernel`):

  ```python
  @wp.kernel
  def build_contact_jacobian_dir_kernel(
      particle_count: int,
      contact_idx: int,
      j_indices: wp.array[wp.int32],
      j_alpha: wp.array[wp.float32],
      direction: wp.vec3,
      out: wp.array[wp.vec3],
  ):
      """Set out[j_indices[contact_idx]] = j_alpha[contact_idx] * direction.

      General-direction variant of :func:`build_contact_jacobian_vec3_kernel`.
      Used by Stage B friction to set tangent-direction Jacobian columns (t1, t2)
      without storing them in a per-contact array.

      Called with ``dim=1`` (single thread).  Caller must zero ``out`` before launch.

      Args:
          particle_count: Total particle count (bounds guard).
          contact_idx: Contact row index (0-based).
          j_indices: Particle index per contact, shape ``[M]``.
          j_alpha: Jacobian coefficient per contact (1.0 for particle-shape), shape ``[M]``.
          direction: The unit direction vector (normal, t1, or t2).
          out: Output vec3 array of length ``particle_count``.
      """
      _tid = wp.tid()
      idx = j_indices[contact_idx]
      if idx >= 0 and idx < particle_count:
          out[idx] = wp.float32(j_alpha[contact_idx]) * direction
  ```

- [ ] **Step 3: Verify the module compiles (no Warp syntax errors)**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run python -c "import warp as wp; wp.init(); from newton._src.solvers.fba.kernels import build_contact_jacobian_dir_kernel; print('OK')"
  ```

  Expected: `OK` (may also print Warp init messages).

- [ ] **Step 4: Commit A (kernel helper)**

  ```bash
  git add newton/_src/solvers/fba/kernels.py
  git commit -m "Add build_contact_jacobian_dir_kernel for Stage B tangent rows"
  ```

---

## Task 2: Add `project_coulomb_cone` and tangent-basis helper to `solver_fba.py`

**Context:** The Coulomb cone projection and tangent-basis computation are pure Python/NumPy (they run in the CPU-side Gauss-Seidel loop). Adding them as module-level functions in `solver_fba.py` lets tests import them directly.

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py` (insert before `class SolverFBA`, after imports)

- [ ] **Step 1: Write a failing test for `project_coulomb_cone`**

  In `newton/tests/test_solver_fba.py`, add the following class (insert before the existing `if __name__ == "__main__":` block at line 2510):

  ```python
  class TestPhase4StageBFriction(unittest.TestCase):
      """Phase 4 Stage B: Coulomb friction via NonSmooth Newton."""

      @classmethod
      def setUpClass(cls):
          wp.init()

      def test_coulomb_cone_projection_correctness(self):
          """project_coulomb_cone handles inside-cone, polar-cone, and surface cases."""
          from newton._src.solvers.fba.solver_fba import project_coulomb_cone

          mu = 0.5
          v_zero = np.array([0.0, 0.0])

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
          # s_new = 1.92, v_new = mu * 1.92 / 5.0 * [3, 4] = 0.384 * [3, 4] = [1.152, 1.536]
          s3, v3 = project_coulomb_cone(-0.1, np.array([3.0, 4.0]), mu)
          self.assertAlmostEqual(s3, 1.92, places=10)
          np.testing.assert_allclose(v3, [1.152, 1.536], atol=1e-10)
          # Verify on cone surface: |v3| = mu * s3
          self.assertAlmostEqual(np.linalg.norm(v3), mu * s3, places=10)

          # Case 4: s = 0, v = [0, 0] → inside cone (both zero).
          s4, v4 = project_coulomb_cone(0.0, np.array([0.0, 0.0]), mu)
          self.assertAlmostEqual(s4, 0.0, places=10)
          np.testing.assert_allclose(v4, [0.0, 0.0], atol=1e-12)
  ```

- [ ] **Step 2: Run failing test to confirm ImportError**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_coulomb_cone_projection_correctness 2>&1 | tail -10
  ```

  Expected: `FAIL` or `ERROR` (ImportError: cannot import name 'project_coulomb_cone').

- [ ] **Step 3: Add `project_coulomb_cone` and `compute_tangent_basis` to `solver_fba.py`**

  In `newton/_src/solvers/fba/solver_fba.py`, insert the following functions after the imports (after line 13, before `def _transform_point`):

  ```python
  def project_coulomb_cone(
      s: float, v: np.ndarray, mu: float
  ) -> tuple[float, np.ndarray]:
      """Project ``(s, v)`` onto the Coulomb friction cone K = {s' >= 0, |v'| <= mu * s'}.

      Three cases:
        1. Already in K: return ``(s, v)`` unchanged.
        2. In polar cone ``mu * |v| <= -s``: project to origin ``(0, 0, 0)``.
        3. Otherwise: project to cone surface ``|v'| = mu * s'``.

      Args:
          s: Normal component (λ_n scalar).
          v: Tangent components ``(λ_t1, λ_t2)``, shape ``(2,)``.
          mu: Coulomb friction coefficient (>= 0).

      Returns:
          Tuple ``(s_new, v_new)`` on or inside the cone.
      """
      v_norm = float(np.sqrt(v[0] ** 2 + v[1] ** 2))
      # Case 1: already inside cone.
      if v_norm <= mu * s and s >= 0.0:
          return s, v
      # Case 2: in polar cone → project to origin.
      if mu * v_norm <= -s:
          return 0.0, np.zeros(2, dtype=np.float64)
      # Case 3: project to cone surface.
      factor = (s + mu * v_norm) / (1.0 + mu * mu)
      s_new = factor
      if v_norm < 1e-15:
          v_new = np.zeros(2, dtype=np.float64)
      else:
          v_new = (mu * factor / v_norm) * v
      return s_new, v_new


  def compute_tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Compute two orthonormal tangent vectors ``(t1, t2)`` from a unit normal.

      Convention mirrors RealSim's ``BaseCollision::generateTangentDirections``:
      pick a reference direction (world X, or world Y if normal is nearly X),
      cross with normal to get t1, then cross normal with t1 to get t2.

      Args:
          n: Unit normal vector, shape ``(3,)``.

      Returns:
          Tuple ``(t1, t2)`` each of shape ``(3,)``, float64 unit vectors
          orthogonal to ``n`` and to each other.
      """
      n = np.asarray(n, dtype=np.float64)
      if abs(n[0]) > 0.9:
          ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
      else:
          ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
      t1 = np.cross(n, ref)
      t1 /= np.linalg.norm(t1) + 1e-30
      t2 = np.cross(n, t1)
      t2 /= np.linalg.norm(t2) + 1e-30
      return t1, t2
  ```

- [ ] **Step 4: Run failing test — should now pass**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_coulomb_cone_projection_correctness 2>&1 | tail -10
  ```

  Expected: `OK`, 1 test ran.

- [ ] **Step 5: Commit B (cone projection + tangent basis)**

  ```bash
  git add newton/_src/solvers/fba/solver_fba.py newton/tests/test_solver_fba.py
  git commit -m "Add Coulomb cone projection + tangent basis helpers; add projection test"
  ```

---

## Task 3: Extend `build_schur_complement` in `linear_solver.py` to support 3 rows per contact

**Context:** Stage A builds an M×M Schur complement with one row per contact (the normal row). Stage B needs a 3M×3M complement: rows 3c, 3c+1, 3c+2 correspond to normal, t1, t2 directions for contact c. The extension is: accept optional `j_tangent1` and `j_tangent2` arrays; when provided, emit a 3M×3M W instead of M×M. Stage A behavior (no tangent arrays) is preserved exactly.

**Files:**
- Modify: `newton/_src/solvers/fba/linear_solver.py` (modify `build_schur_complement` method, lines 791–854)

- [ ] **Step 1: Write the failing test for 3M×3M Schur**

  In `newton/tests/test_solver_fba.py`, add a new test method inside `TestPhase4StageBFriction`:

  ```python
  def test_friction_W_block_structure(self):
      """build_schur_complement with tangents returns 6x6 symmetric W for 2 contacts."""
      import scipy.sparse as sp

      from newton._src.solvers.fba.linear_solver import FBALinearSolver, factorize_and_sparse_inverse
      from newton._src.solvers.fba.solver_fba import compute_tangent_basis

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

      # Shape must be 6x6 (3 rows per contact × 2 contacts).
      self.assertEqual(W.shape, (6, 6), f"Expected (6,6), got {W.shape}")

      # Symmetry.
      np.testing.assert_allclose(W, W.T, atol=1e-10,
                                  err_msg="Friction W is not symmetric")

      # All eigenvalues positive (SPD since the contacts are at distinct particles
      # and the directions are orthonormal).
      eigvals = np.linalg.eigvalsh(W)
      self.assertGreater(float(eigvals.min()), 0.0,
                          f"W has non-positive eigenvalue: {eigvals.min():.3e}")
  ```

- [ ] **Step 2: Run failing test**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_friction_W_block_structure 2>&1 | tail -10
  ```

  Expected: `ERROR` — `build_schur_complement() got unexpected keyword arguments` or `TypeError`.

- [ ] **Step 3: Extend `build_schur_complement` in `linear_solver.py`**

  Read the current signature at lines 791–854 carefully (already done in planning). Replace the method body with the extended version.

  The key change: accept two optional parameters `j_tangent1` and `j_tangent2` (both `wp.array[wp.vec3] | None`). When both are provided, the output is `(3M, 3M)` with rows ordered as `[n_0, t1_0, t2_0, n_1, t1_1, t2_1, ...]`. When not provided, behavior is identical to Stage A.

  Find the method start in `linear_solver.py`:

  ```python
  def build_schur_complement(
      self,
      num_contacts: int,
      j_indices: wp.array,
      j_normals: wp.array,
      j_alpha: wp.array,
  ) -> np.ndarray:
  ```

  Replace the entire method (from `def build_schur_complement` through the final `return W` at line 854) with:

  ```python
  def build_schur_complement(
      self,
      num_contacts: int,
      j_indices: wp.array,
      j_normals: wp.array,
      j_alpha: wp.array,
      j_tangent1: wp.array | None = None,
      j_tangent2: wp.array | None = None,
  ) -> np.ndarray:
      """Build ``W = J · A⁻¹ · Jᵀ`` as a dense NumPy array.

      Stage A (``j_tangent1`` and ``j_tangent2`` are ``None``): emits an ``(M, M)``
      matrix with one row per contact (normal direction only).

      Stage B (both tangent arrays provided): emits a ``(3M, 3M)`` matrix with
      three rows per contact ordered ``[n_c, t1_c, t2_c]`` for c = 0..M-1.
      Off-diagonal coupling between contacts is fully included.

      The ``_y_cache`` attribute is updated to contain one entry per row (M entries
      in Stage A; 3M entries in Stage B); each entry is the corresponding ``A⁻¹ · J_row^T``
      column as a (N, 3) float64 NumPy array.  Used by
      :meth:`~newton._src.solvers.fba.solver_fba.SolverFBA._apply_lambda_correction_friction`.

      Args:
          num_contacts: Number of active contacts ``M``.
          j_indices: Particle index per contact, shape ``[M]``, dtype int32.
          j_normals: World-frame contact normal per contact, shape ``[M]``, dtype vec3.
          j_alpha: Jacobian coefficient per contact, shape ``[M]``, dtype float32.
          j_tangent1: First tangent direction per contact, shape ``[M]``, dtype vec3.
              When ``None``, Stage A behavior (M×M W).
          j_tangent2: Second tangent direction per contact, shape ``[M]``, dtype vec3.
              Must be provided together with ``j_tangent1``.

      Returns:
          Dense float64 NumPy array of shape ``(M, M)`` (Stage A) or ``(3M, 3M)`` (Stage B).
      """
      from .kernels import (  # noqa: PLC0415
          build_contact_jacobian_dir_kernel,
          build_contact_jacobian_vec3_kernel,
          zero_vec3_kernel,
      )

      has_friction = j_tangent1 is not None and j_tangent2 is not None
      rows_per_contact = 3 if has_friction else 1
      M = num_contacts
      total_rows = M * rows_per_contact
      n = self.n
      dev = self.device

      W = np.zeros((total_rows, total_rows), dtype=np.float64)

      # Allocate scratch for one Jacobian column (sparse vec3, length N).
      jcol = wp.zeros(n, dtype=wp.vec3, device=dev)
      y_out = wp.empty(n, dtype=wp.vec3, device=dev)

      # Pull contact metadata to host (small M).
      idx_np = j_indices.numpy()   # (M,) int32
      n_np = j_normals.numpy()     # (M, 3) float32
      a_np = j_alpha.numpy()       # (M,) float32
      if has_friction:
          t1_np = j_tangent1.numpy()   # (M, 3) float32
          t2_np = j_tangent2.numpy()   # (M, 3) float32

      # Cache y columns for reuse in correction step.
      self._y_cache: list[np.ndarray] = []

      # For each (contact c, axis a) row in J, compute y_{c,a} = A⁻¹ J_{c,a}^T
      # then fill one column of W.
      for c in range(M):
          directions: list[np.ndarray]
          if has_friction:
              directions = [n_np[c], t1_np[c], t2_np[c]]
          else:
              directions = [n_np[c]]

          for a, direction in enumerate(directions):
              row = c * rows_per_contact + a
              # Zero the sparse column.
              wp.launch(zero_vec3_kernel, dim=n, inputs=[jcol], device=dev)
              # Set jcol[idx] = alpha * direction.
              wp.launch(
                  build_contact_jacobian_dir_kernel,
                  dim=1,
                  inputs=[n, c, j_indices, j_alpha, wp.vec3(float(direction[0]), float(direction[1]), float(direction[2]))],
                  outputs=[jcol],
                  device=dev,
              )
              # Solve: y_{c,a} = A⁻¹ · jcol.
              self.solve(jcol, y_out)
              y_np = y_out.numpy()   # (N, 3) float64-ish (actually float32 vec3)
              self._y_cache.append(y_np.copy())

              # Fill column `row` of W:
              # W[row', row] = J_{row'} · y_{c,a}
              #   = alpha[c'] * dot(dir_{c',a'}, y_np[idx[c']])
              for cp in range(M):
                  dirs_cp: list[np.ndarray]
                  if has_friction:
                      dirs_cp = [n_np[cp], t1_np[cp], t2_np[cp]]
                  else:
                      dirs_cp = [n_np[cp]]
                  ip = idx_np[cp]
                  for ap, dir_cp in enumerate(dirs_cp):
                      rowp = cp * rows_per_contact + ap
                      W[rowp, row] = float(a_np[cp]) * float(np.dot(dir_cp, y_np[ip]))

      return W
  ```

  > **Implementation note:** The `build_contact_jacobian_vec3_kernel` used by Stage A read the direction from the `j_normals` array. The new `build_contact_jacobian_dir_kernel` accepts direction as a `wp.vec3` value directly. The Stage A path now also uses the new kernel (since the direction for the normal row is just `n_np[c]`). This preserves the W values identically.

- [ ] **Step 4: Run failing test — should now pass**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_friction_W_block_structure 2>&1 | tail -10
  ```

  Expected: `OK`, 1 test ran.

- [ ] **Step 5: Run full suite to confirm Stage A tests still pass**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -5
  ```

  Expected: `Ran 61 tests`, `OK`.

- [ ] **Step 6: Commit C (extended Schur)**

  ```bash
  git add newton/_src/solvers/fba/linear_solver.py newton/tests/test_solver_fba.py
  git commit -m "Extend build_schur_complement to support friction (3 rows per contact)"
  ```

---

## Task 4: Extend `SolverFBA.__init__` and `_ensure_contact_buffers` for Stage B state

**Context:** Stage B needs two new constructor parameters (`friction: bool`, `mu_per_pair_override`) and two new per-contact host buffers (tangent1 and tangent2 arrays computed per-step). The `_ensure_contact_buffers` method needs to allocate Warp device arrays for tangents and mu.

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py`

- [ ] **Step 1: Extend `__init__` signature**

  In `solver_fba.py`, find the `def __init__` at line 89 and change:

  ```python
  def __init__(
      self,
      model: Model,
      iterations: int = 10,
      pin_stiffness: float = 1e12,
      stretching_model: Literal["arap", "corotational", "neohookean"] = "arap",
      mu: float | None = None,
      lam: float | None = None,
  ) -> None:
  ```

  to:

  ```python
  def __init__(
      self,
      model: Model,
      iterations: int = 10,
      pin_stiffness: float = 1e12,
      stretching_model: Literal["arap", "corotational", "neohookean"] = "arap",
      mu: float | None = None,
      lam: float | None = None,
      friction: bool = True,
      mu_per_pair_override: np.ndarray | None = None,
  ) -> None:
  ```

- [ ] **Step 2: Store new parameters in `__init__` body**

  Inside `__init__`, after `self.pin_stiffness = float(pin_stiffness)`, add:

  ```python
  self.friction = bool(friction)
  self._mu_per_pair_override = (
      np.asarray(mu_per_pair_override, dtype=np.float64)
      if mu_per_pair_override is not None
      else None
  )
  ```

- [ ] **Step 3: Extend `_ensure_contact_buffers` for tangent + mu buffers**

  Find `_ensure_contact_buffers` (around line 507). Replace the entire method with:

  ```python
  def _ensure_contact_buffers(self, max_contacts: int) -> None:
      """Lazily allocate per-step contact buffers sized to ``max_contacts``."""
      if hasattr(self, "_contact_buf_max") and self._contact_buf_max >= max_contacts:
          return
      device = self._device
      cap = max(max_contacts, 16)
      self._contact_buf_max = cap
      self._contact_particle_d = wp.empty(cap, dtype=wp.int32, device=device)
      self._contact_normal_d = wp.empty(cap, dtype=wp.vec3, device=device)
      self._contact_alpha_d = wp.empty(cap, dtype=wp.float32, device=device)
      self._contact_offset_d = wp.empty(cap, dtype=wp.float64, device=device)
      # Stage B: tangent directions and per-contact friction mu.
      self._contact_tangent1_d = wp.empty(cap, dtype=wp.vec3, device=device)
      self._contact_tangent2_d = wp.empty(cap, dtype=wp.vec3, device=device)
      self._contact_mu_h = np.zeros(cap, dtype=np.float64)  # host-side mu array
  ```

- [ ] **Step 4: Verify no import errors**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run python -c "from newton._src.solvers.fba.solver_fba import SolverFBA; print('OK')"
  ```

  Expected: `OK`.

- [ ] **Step 5: Run tests to confirm no regression**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -5
  ```

  Expected: `OK` (same count as before).

---

## Task 5: Extend `update_contacts` to compute per-contact μ and tangent basis

**Context:** When `friction=True`, `update_contacts` must compute per-contact friction μ via `sqrt(particle_mu * shape_mu)` mixing (or use `mu_per_pair_override`), and compute the tangent basis `(t1, t2)` for each contact from the normal. These are stored host-side (in `_contact_mu_h`, `_contact_tangent1_h`, `_contact_tangent2_h`) and uploaded to device arrays for use in the Schur build.

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py` (`update_contacts` method, lines 519–609)

- [ ] **Step 1: Write failing tests for friction μ and tangent computation**

  Add two more test methods to `TestPhase4StageBFriction` in `test_solver_fba.py`:

  ```python
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
              wp.vec3(1.0, 0.0, 0.0),   # pinned
              wp.vec3(0.0, 0.0, 1.0),   # pinned
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

      # Stage A solver (no friction param = default friction=True, but let's compare
      # friction=False against a fresh stage-A solver).
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
      np.testing.assert_array_equal(qa, qb,
          err_msg="Two friction=False solvers diverged from each other")
      # Particle 0 should be above the plane.
      self.assertGreaterEqual(float(qa[0, 1]), -1e-4,
          f"Particle still below plane: y={float(qa[0, 1]):.6f}")
  ```

  > **Note:** Testing that `friction=False` gives Stage A behavior is done by confirming: (a) the two separate `friction=False` solvers agree identically, and (b) the particle is above the plane. Full bit-identity vs the original Stage A `SolverFBA(model)` is verified by confirming both produce the same unilateral contact resolution.

- [ ] **Step 2: Run failing test**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_friction_disabled_matches_stage_a 2>&1 | tail -10
  ```

  Expected: `ERROR` or `FAIL` (since `friction` param doesn't exist in constructor yet).

  > Note: `friction` was added to `__init__` in Task 4, so if Task 4 is done this may `PASS` already. If it passes, that's fine — proceed to next steps.

- [ ] **Step 3: Extend `update_contacts` to compute tangent basis and μ**

  Find the `update_contacts` method (around line 519). At the end of the method, after setting `self._contact_count = M`, add the Stage B tangent + mu computation:

  After the lines:
  ```python
  self._contact_particle_d.assign(particle_h[:M].astype(np.int32))
  self._contact_normal_d.assign(normal_h[:M].astype(np.float32))
  self._contact_alpha_d.assign(alpha_h[:M])
  self._contact_offset_d.assign(offset_h[:M])
  ```

  Insert:

  ```python
  # Stage B: compute tangent basis and friction μ for each contact.
  if self.friction:
      t1_h = np.zeros((M, 3), dtype=np.float32)
      t2_h = np.zeros((M, 3), dtype=np.float32)
      for c in range(M):
          t1, t2 = compute_tangent_basis(normal_h[c])
          t1_h[c] = t1.astype(np.float32)
          t2_h[c] = t2.astype(np.float32)
      self._contact_tangent1_d.assign(t1_h)
      self._contact_tangent2_d.assign(t2_h)
      self._contact_tangent1_h = t1_h
      self._contact_tangent2_h = t2_h

      # Compute per-contact friction μ via VBD-style sqrt mixing.
      particle_mu = float(getattr(model, "particle_mu", 0.5))
      mu_h = np.zeros(M, dtype=np.float64)
      if self._mu_per_pair_override is not None:
          for c in range(M):
              mu_h[c] = float(self._mu_per_pair_override[c])
      else:
          shape_mat_mu = (
              model.shape_material_mu.numpy()
              if hasattr(model, "shape_material_mu")
              else None
          )
          for c in range(M):
              s_idx = int(shape_h[c])
              if shape_mat_mu is not None and s_idx >= 0 and s_idx < len(shape_mat_mu):
                  mu_h[c] = float(np.sqrt(particle_mu * float(shape_mat_mu[s_idx])))
              else:
                  mu_h[c] = particle_mu
      self._contact_mu_h = mu_h[:M]
  ```

  Also make sure `model` is accessible at this point in the method body. Check that `model = self.model` is already set before the loop at line 549 — it is (line 549 reads `model = self.model`).

- [ ] **Step 4: Run all current tests to verify no regression**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -5
  ```

  Expected: `OK`, count should be 63 (59 original + 3 new so far: cone test, W block test, friction_disabled test).

---

## Task 6: Extend `step` to use the 3M Schur complement + blocked Gauss-Seidel with Coulomb cone projection

**Context:** This is the main algorithmic change. When `friction=True` and contacts are active, the step method builds a 3M×3M W, computes a 3M residual, solves via blocked projected Gauss-Seidel with Coulomb cone projection, and applies the correction using all 3M λ components.

**Files:**
- Modify: `newton/_src/solvers/fba/solver_fba.py` (the `step` method contact branch starting at line 436, and two new helper methods)

- [ ] **Step 1: Add `_compute_contact_residual_friction` helper**

  After `_compute_contact_residual` (around line 615), insert:

  ```python
  def _compute_contact_residual_friction(self, x_np: np.ndarray) -> np.ndarray:
      """Compute the 3M-vector residual ``r = [r_n, r_t1, r_t2]`` for each contact.

      For each contact c:
        r_n[c]  = offset[c] - alpha[c] * dot(n[c],  x_np[p[c]])
        r_t1[c] =           - alpha[c] * dot(t1[c], x_np[p[c]])
        r_t2[c] =           - alpha[c] * dot(t2[c], x_np[p[c]])

      Tangent residuals target zero tangential displacement relative to the
      contact anchor (PD-position friction).

      Args:
          x_np: Unconstrained solution, shape ``(N, 3)``, float32.

      Returns:
          Residual vector of shape ``(3M,)``, float64, ordered
          ``[r_n_0, r_t1_0, r_t2_0, r_n_1, ...]``.
      """
      M = self._contact_count
      r = np.zeros(3 * M, dtype=np.float64)
      for c in range(M):
          ip = int(self._contact_particle_h[c])
          xp = x_np[ip].astype(np.float64)
          alpha = float(self._contact_alpha_h[c])
          n = self._contact_normal_h[c].astype(np.float64)
          t1 = self._contact_tangent1_h[c].astype(np.float64)
          t2 = self._contact_tangent2_h[c].astype(np.float64)
          r[3 * c + 0] = float(self._contact_offset_h[c]) - alpha * float(np.dot(n, xp))
          r[3 * c + 1] = -alpha * float(np.dot(t1, xp))
          r[3 * c + 2] = -alpha * float(np.dot(t2, xp))
      return r
  ```

- [ ] **Step 2: Add `_solve_nsn_coulomb` helper**

  After `_solve_nsn_unilateral`, insert:

  ```python
  def _solve_nsn_coulomb(
      self, W: np.ndarray, r: np.ndarray, mu: np.ndarray, max_iters: int = 20
  ) -> np.ndarray:
      """Solve the frictional LCP via blocked projected Gauss-Seidel.

      Operates on 3-blocks ``[λ_n, λ_t1, λ_t2]`` per contact.  Each block
      is updated by solving the local 3×3 system (``W_cc``), then projecting
      onto the Coulomb cone.

      Args:
          W: Dense ``(3M, 3M)`` Schur complement matrix.
          r: Residual vector of shape ``(3M,)``.
          mu: Per-contact friction coefficient, shape ``(M,)``.
          max_iters: Maximum blocked Gauss-Seidel iterations.

      Returns:
          Contact impulse vector ``λ``, shape ``(3M,)``, float64, ordered
          ``[λ_n_0, λ_t1_0, λ_t2_0, λ_n_1, ...]``.
      """
      M = len(mu)
      lam = np.zeros(3 * M, dtype=np.float64)
      for _ in range(max_iters):
          lam_old = lam.copy()
          for c in range(M):
              s = slice(3 * c, 3 * c + 3)
              W_cc = W[s, s]
              # Effective RHS: r_eff = r[s] - sum_{c' != c} W[s, s'] lam[s']
              off_diag = W[s, :] @ lam - W_cc @ lam[s]
              r_eff = r[s] - off_diag
              # Solve 3x3: lam_unc = W_cc^{-1} r_eff
              if np.linalg.det(W_cc) < 1e-20:
                  continue
              lam_unc = np.linalg.solve(W_cc, r_eff)
              # Project onto Coulomb cone.
              s_unc = float(lam_unc[0])
              v_unc = lam_unc[1:3]
              s_proj, v_proj = project_coulomb_cone(s_unc, v_unc, float(mu[c]))
              lam[3 * c + 0] = s_proj
              lam[3 * c + 1] = v_proj[0]
              lam[3 * c + 2] = v_proj[1]
          if np.linalg.norm(lam - lam_old, np.inf) < 1e-8:
              break
      return lam
  ```

- [ ] **Step 3: Add `_apply_lambda_correction_friction` helper**

  After `_apply_lambda_correction`, insert:

  ```python
  def _apply_lambda_correction_friction(self, lam: np.ndarray) -> wp.array:
      """Compute ``correction = A⁻¹ · Jᵀ · λ`` for Stage B (3M λ).

      Uses the cached ``_y_cache`` from :meth:`build_schur_complement` which
      contains one (N, 3) array per row (3M total entries for friction).

      Each row r corresponds to contact ``c = r // 3``, axis ``a = r % 3``:
        - a=0: normal  contribution
        - a=1: t1 contribution
        - a=2: t2 contribution

      Args:
          lam: Contact impulse vector, shape ``(3M,)``, float64.

      Returns:
          Correction vec3 Warp array of length N.
      """
      from .kernels import accumulate_vec3_kernel  # noqa: PLC0415

      M = self._contact_count
      total_rows = 3 * M
      N = self.model.particle_count
      dev = self._device
      ls = self._linear_solver

      correction_np = np.zeros((N, 3), dtype=np.float64)

      if hasattr(ls, "_y_cache") and len(ls._y_cache) == total_rows:
          for row in range(total_rows):
              if abs(lam[row]) < 1e-15:
                  continue
              correction_np += lam[row] * ls._y_cache[row]
      else:
          # Fallback: re-solve for each row.
          from .kernels import zero_vec3_kernel  # noqa: PLC0415

          tmp = wp.empty(N, dtype=wp.vec3, device=dev)
          work = wp.empty(N, dtype=wp.vec3, device=dev)
          for c in range(M):
              if not hasattr(self, "_contact_tangent1_h"):
                  continue
              n = self._contact_normal_h[c].astype(np.float64)
              t1 = self._contact_tangent1_h[c].astype(np.float64)
              t2 = self._contact_tangent2_h[c].astype(np.float64)
              directions = [n, t1, t2]
              for a, direction in enumerate(directions):
                  row = 3 * c + a
                  if abs(lam[row]) < 1e-15:
                      continue
                  wp.launch(zero_vec3_kernel, dim=N, inputs=[work], device=dev)
                  ip = int(self._contact_particle_h[c])
                  alpha = float(self._contact_alpha_h[c])
                  # Set work[ip] = alpha * direction.
                  from .kernels import build_contact_jacobian_dir_kernel  # noqa: PLC0415
                  wp.launch(
                      build_contact_jacobian_dir_kernel,
                      dim=1,
                      inputs=[N, c, self._contact_particle_d, self._contact_alpha_d,
                               wp.vec3(float(direction[0]), float(direction[1]), float(direction[2]))],
                      outputs=[work],
                      device=dev,
                  )
                  ls.solve(work, tmp)
                  tmp_np = tmp.numpy().astype(np.float64)
                  correction_np += lam[row] * tmp_np

      correction = wp.zeros(N, dtype=wp.vec3, device=dev)
      correction.assign(correction_np.astype(np.float32))
      return correction
  ```

- [ ] **Step 4: Modify the `step` method contact branch to dispatch friction vs unilateral**

  In the `step` method, find the `if has_contacts:` block (around line 436):

  ```python
  if has_contacts:
      # --- Schur-complement NSN contact correction ---
      M = self._contact_count
      ls = self._linear_solver

      # 1. Build W = J A^{-1} J^T  (M x M dense).
      W = ls.build_schur_complement(
          M,
          self._contact_particle_d,
          self._contact_normal_d,
          self._contact_alpha_d,
      )

      # 2. Compute residual r = c_offset - J·x_unc  (positive = penetrating).
      x_unc_np = self._x_cur.numpy()  # (N, 3) float32
      r = self._compute_contact_residual(x_unc_np)

      # 3. Solve W λ = r with projected Gauss-Seidel (λ ≥ 0).
      lam = self._solve_nsn_unilateral(W, r, max_iters=20)

      # 4. Apply correction: x_cur = x_unc + A^{-1} J^T λ.
      #    (x* = A^{-1}(b + J^T λ) = x_unc + A^{-1} J^T λ)
      if np.any(lam > 1e-15):
          correction = self._apply_lambda_correction(lam)
          wp.launch(
              accumulate_vec3_kernel,
              dim=N,
              inputs=[correction],
              outputs=[self._x_cur],
              device=device,
          )
  ```

  Replace with:

  ```python
  if has_contacts:
      # --- Schur-complement NSN contact correction ---
      M = self._contact_count
      ls = self._linear_solver

      if self.friction and hasattr(self, "_contact_tangent1_d"):
          # Stage B: 3M Schur complement with Coulomb cone projection.
          W = ls.build_schur_complement(
              M,
              self._contact_particle_d,
              self._contact_normal_d,
              self._contact_alpha_d,
              self._contact_tangent1_d,
              self._contact_tangent2_d,
          )
          x_unc_np = self._x_cur.numpy()  # (N, 3) float32
          r = self._compute_contact_residual_friction(x_unc_np)
          lam = self._solve_nsn_coulomb(W, r, self._contact_mu_h[:M], max_iters=20)
          if np.any(np.abs(lam) > 1e-15):
              correction = self._apply_lambda_correction_friction(lam)
              wp.launch(
                  accumulate_vec3_kernel,
                  dim=N,
                  inputs=[correction],
                  outputs=[self._x_cur],
                  device=device,
              )
      else:
          # Stage A: M Schur complement, unilateral (λ ≥ 0) only.
          W = ls.build_schur_complement(
              M,
              self._contact_particle_d,
              self._contact_normal_d,
              self._contact_alpha_d,
          )
          x_unc_np = self._x_cur.numpy()  # (N, 3) float32
          r = self._compute_contact_residual(x_unc_np)
          lam = self._solve_nsn_unilateral(W, r, max_iters=20)
          if np.any(lam > 1e-15):
              correction = self._apply_lambda_correction(lam)
              wp.launch(
                  accumulate_vec3_kernel,
                  dim=N,
                  inputs=[correction],
                  outputs=[self._x_cur],
                  device=device,
              )
  ```

- [ ] **Step 5: Run all existing tests**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -5
  ```

  Expected: all tests pass (`OK`).

- [ ] **Step 6: Commit C (Step implementation)**

  ```bash
  git add newton/_src/solvers/fba/solver_fba.py
  git commit -m "Implement Stage B friction in SolverFBA.update_contacts + step"
  ```

---

## Task 7: Add the remaining 3 Stage B tests

**Context:** We need 3 more behavioral tests: static friction holds particle on tilted plane, kinetic friction decelerates sliding, and the `friction=False` disabled path (already written in Task 5 but let's finalize the kinetic and static tests).

**Files:**
- Modify: `newton/tests/test_solver_fba.py` (add to `TestPhase4StageBFriction`)

- [ ] **Step 1: Add `test_static_friction_holds_particle_on_plane`**

  Add this method to `TestPhase4StageBFriction`:

  ```python
  def test_static_friction_holds_particle_on_plane(self):
      """A particle on a horizontal plane with lateral gravity should not drift when mu is sufficient.

      Setup: single free particle at (0, 0, 0) exactly on the plane (y=0).
      Gravity has a horizontal component: gravity_world = (0.1, -1.0, 0.0) [m/s²].
      Normal load: F_n = m * 1.0. Tangent force: F_t = m * 0.1.
      With mu = 0.5: mu * F_n = 0.5 > F_t = 0.1 → static friction should hold.
      After 50 steps, tangent displacement should be < 1e-3 m.
      """
      builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
      # gravity needs to be set per-world. We'll set it on the builder's gravity.
      builder.gravity = np.array([0.1, -1.0, 0.0], dtype=np.float32)
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
      device = model.device

      # mu=0.5 with particle_mu: override to 0.5 per-pair.
      mu_override = np.full(1, 0.5, dtype=np.float64)
      solver = SolverFBA(model, iterations=5, friction=True, mu_per_pair_override=mu_override)

      # Contact: normal = (0,1,0), anchor = (0,0,0), shape=-1 (static).
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
          # Swap states.
          state, state_out = state_out, state

      q = state.particle_q.numpy()
      # Measure tangential displacement (x and z components of particle 0).
      tangent_drift = float(np.sqrt(q[0, 0] ** 2 + q[0, 2] ** 2))
      self.assertLessEqual(
          tangent_drift, 1e-3,
          f"Static friction failed: tangent drift = {tangent_drift:.6f} m (target <= 1e-3 m)"
      )
      # Particle should stay on or above the plane.
      self.assertGreaterEqual(float(q[0, 1]), -1e-4,
          f"Particle fell through plane: y = {float(q[0, 1]):.6f}")
  ```

- [ ] **Step 2: Add `test_kinetic_friction_slows_sliding`**

  Add this method to `TestPhase4StageBFriction`:

  ```python
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
          builder.gravity = np.array([0.0, -1.0, 0.0], dtype=np.float32)
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
          x_friction, x_frictionless,
          f"Friction should decelerate particle: x(mu=0.5)={x_friction:.4f} "
          f"should be < x(mu=0)={x_frictionless:.4f}"
      )
  ```

- [ ] **Step 3: Run all Stage B tests**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k TestPhase4StageBFriction 2>&1 | tail -15
  ```

  Expected: `Ran 5 tests`, `OK`.

  Record:
  - **Static friction tangent drift** (from test output or by adding `print` temporarily): should be ≤ 1e-3 m.
  - **x(mu=0)** vs **x(mu=0.5)**: the friction case should show a smaller x.

- [ ] **Step 4: Run the full test suite**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -5
  ```

  Expected: `Ran 64 tests`, `OK` (59 original + 5 new).

- [ ] **Step 5: Commit D (Stage B tests)**

  ```bash
  git add newton/tests/test_solver_fba.py
  git commit -m "Add Phase 4 Stage B friction tests (5 tests in TestPhase4StageBFriction)"
  ```

---

## Task 8: Performance measurement and report

**Context:** The spec requests a per-step timing comparison at M=32 contacts vs Stage A's ~58 ms/step. Add a performance measurement that can run standalone (not in the test suite).

**Files:**
- No file changes (shell commands only)

- [ ] **Step 1: Run a quick timing test for Stage B at M=32**

  Create a temporary inline timing script:

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run python - <<'EOF'
  import time
  import numpy as np
  import warp as wp
  import newton
  from newton._src.solvers.fba.solver_fba import SolverFBA

  wp.init()
  device = "cuda:0" if wp.is_cuda_available() else "cpu"

  # Build a cloth with enough particles (17x17 = 289).
  builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
  builder.add_cloth_grid(
      pos=wp.vec3(0.0, -0.1, 0.0),
      rot=wp.quat_identity(),
      vel=wp.vec3(0.0, 0.0, 0.0),
      dim_x=16, dim_y=16,
      cell_x=0.05, cell_y=0.05,
      mass=0.01,
      tri_ke=1.0e4, tri_ka=0.0, tri_kd=0.0,
      edge_ke=1e-2, edge_kd=0.0,
  )
  model = builder.finalize(device=device)
  N = model.particle_count

  # Stage B solver.
  mu_override = np.full(32, 0.5, dtype=np.float64)
  solver = SolverFBA(model, iterations=5, friction=True, mu_per_pair_override=mu_override)

  # Build 32 fake contacts on 32 distinct particles.
  M = 32
  contacts = newton.Contacts(rigid_contact_max=0, soft_contact_max=M, device=device)
  contacts.soft_contact_count.assign(np.array([M], dtype=np.int32))
  particles = np.arange(M, dtype=np.int32)
  normals = np.tile([0.0, 1.0, 0.0], (M, 1)).astype(np.float32)
  anchors = np.zeros((M, 3), dtype=np.float32)
  shapes = np.full(M, -1, dtype=np.int32)
  contacts.soft_contact_particle.assign(particles)
  contacts.soft_contact_normal.assign(normals)
  contacts.soft_contact_body_pos.assign(anchors)
  contacts.soft_contact_shape.assign(shapes)

  s_in, s_out = model.state(), model.state()
  dt = 1.0 / 60.0

  # Warm-up.
  for _ in range(3):
      s_in.clear_forces()
      solver.step(s_in, s_out, None, contacts, dt)
      s_in, s_out = s_out, s_in

  # Timed run.
  wp.synchronize()
  t0 = time.perf_counter()
  for _ in range(10):
      s_in.clear_forces()
      solver.step(s_in, s_out, None, contacts, dt)
      s_in, s_out = s_out, s_in
  wp.synchronize()
  t1 = time.perf_counter()

  ms_per_step = (t1 - t0) / 10 * 1000
  print(f"Stage B M=32, N={N}: {ms_per_step:.1f} ms/step")
  EOF
  ```

  Record the output. Expected: ~150–200 ms/step (3× the Stage A ~58 ms due to 3× Schur build cost).

- [ ] **Step 2: Compare vs Stage A timing**

  Run the same script but with `friction=False` to get Stage A baseline. Compare and report the ratio.

---

## Task 9: Lint and final verification

- [ ] **Step 1: Run pre-commit hooks**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uvx pre-commit run -a 2>&1 | tail -20
  ```

  Fix any formatting issues (ruff will auto-fix most).

- [ ] **Step 2: Run the full test suite one last time**

  ```bash
  export PATH="$HOME/.local/bin:$PATH" && uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -5
  ```

  Expected: `Ran 64 tests`, `OK`.

- [ ] **Step 3: Confirm commit history**

  ```bash
  git log --oneline -6
  ```

  Expected commits (in order from newest):
  ```
  <sha> Add Phase 4 Stage B friction tests (5 tests in TestPhase4StageBFriction)
  <sha> Implement Stage B friction in SolverFBA.update_contacts + step
  <sha> Extend build_schur_complement to support friction (3 rows per contact)
  <sha> Add Coulomb cone projection + tangent basis helpers; add projection test
  <sha> Add build_contact_jacobian_dir_kernel for Stage B tangent rows
  ```

---

## Self-Review Against Spec

| Spec requirement | Covered by Task |
|---|---|
| `friction: bool = True` param on `__init__` | Task 4 |
| `mu_per_pair_override` param | Task 4 |
| Per-contact μ via `sqrt(particle_mu × shape_mu)` | Task 5 |
| Tangent basis `compute_tangent_basis` (cross-product convention) | Task 2 |
| 3M×3M Schur complement W | Task 3 |
| 3M residual (normal + 2 tangent) | Task 6 |
| Blocked Gauss-Seidel with 3×3 local solve | Task 6 |
| Coulomb cone projection `project_coulomb_cone` | Task 2 |
| Correction via `A⁻¹ · Jᵀ · λ` (all 3M rows) | Task 6 |
| `friction=False` preserves Stage A behavior | Task 5 (test), Task 6 (dispatch) |
| No λ warm-start | λ initialized to zero each step (not changed from Stage A) |
| `test_friction_disabled_matches_stage_a` | Task 5 |
| `test_static_friction_holds_particle_on_plane` | Task 7 |
| `test_kinetic_friction_slows_sliding` | Task 7 |
| `test_coulomb_cone_projection_correctness` | Task 2 |
| `test_friction_W_block_structure` | Task 3 |
| 59 → 64 tests | Tasks 2, 3, 5, 7 |
| Performance report | Task 8 |

**Placeholder scan:** All code blocks are complete. No TBD or TODO markers present.

**Type consistency check:**
- `build_schur_complement` takes `wp.array | None` for tangent args — consistent with the method body's `if has_friction` check.
- `_contact_mu_h` is `np.float64` array — consistent with `_solve_nsn_coulomb` expecting `np.ndarray`.
- `project_coulomb_cone` returns `(float, np.ndarray)` — consistent with `_solve_nsn_coulomb` usage at lines `s_proj, v_proj = project_coulomb_cone(...)`.
- `_y_cache` length check in `_apply_lambda_correction_friction` uses `3 * M` — consistent with the 3M rows that `build_schur_complement` with friction now writes.
