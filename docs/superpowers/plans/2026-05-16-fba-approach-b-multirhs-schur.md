# FBA Approach B: Multi-RHS BSR SpMV for Schur Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sequential per-row solve loop in `FBALinearSolver.build_schur_complement` with a custom multi-RHS BSR SpMV kernel that processes all `total_rows` RHS columns in a single 2D launch, eliminating ~3600 kernel launches per PD iteration.

**Architecture:** Three new Warp kernels (`bsr_mv_multi_rhs_scalar_kernel`, `apply_permutation_multi_rhs_kernel`, `scale_by_diag_multi_rhs_kernel`) operate on 2D arrays of shape `(R, N)`, processing all R right-hand sides in one launch. A new `FBALinearSolver.solve_multi_rhs_scalar` method applies the five-step `A⁻¹` pipeline in batch. `build_schur_complement` iterates over only 3 axes (x/y/z) instead of 3×total_rows solve calls, with kernels to pack/unpack the `(total_rows, N, 3)` Jacobian tensor.

**Tech Stack:** Python, Warp (GPU kernels), NumPy. No new dependencies.

---

## File Structure

| File | Change |
|---|---|
| `newton/_src/solvers/fba/kernels.py` | Add 5 new kernels: `bsr_mv_multi_rhs_scalar_kernel`, `apply_permutation_multi_rhs_kernel`, `scale_by_diag_multi_rhs_kernel`, `pack_jacobian_axis_kernel`, `unpack_to_A_inv_Jt_axis_kernel` |
| `newton/_src/solvers/fba/linear_solver.py` | Add `solve_multi_rhs_scalar` method to `FBALinearSolver`; modify `build_schur_complement` to use 3-axis loop with multi-RHS solve |

---

## Task 1: Add multi-RHS primitive kernels to kernels.py

**Files:**
- Modify: `newton/_src/solvers/fba/kernels.py` (append after line 1243)

### Context: BSR value access for 1×1 blocks

For `_csc_to_bsr_1x1`, the BSR matrix is built with `block_type=wp.float64` (scalar). This means `BsrMatrix.values` is a **1D** `wp.array[wp.float64]` of length `nnz`. Indexing: `values[block]` gives the scalar for block index `block`. The Warp `bsr_mv_kernel` uses `scalar_values` (a 3D view as `(nnz, 1, 1)`) internally, but we bypass that and work with the raw 1D `values` array directly.

### Context: `wp.array2d` in kernels

A Warp `wp.array2d[wp.float64]` is a 2D array. In the kernel body use `arr[i, j]`. The `wp.tid()` for a 2D launch returns `(i, j)` as a tuple — use `i, j = wp.tid()`.

- [ ] **Step 1: Append the three multi-RHS primitive kernels to kernels.py**

Append to the end of `/home/ziqiu/work/newton/newton/_src/solvers/fba/kernels.py`:

```python
# ---------------------------------------------------------------------------
# Approach B — multi-RHS batched primitives for Schur complement build
# ---------------------------------------------------------------------------


@wp.kernel
def bsr_mv_multi_rhs_scalar_kernel(
    A_offsets: wp.array[wp.int32],
    A_columns: wp.array[wp.int32],
    A_values: wp.array[wp.float64],
    x: wp.array2d[wp.float64],
    y: wp.array2d[wp.float64],
):
    """y[rhs, row] = sum_k A[row, col_k] * x[rhs, col_k].

    One thread per (rhs_idx, matrix_row). ``A`` is a scalar 1×1 BSR matrix
    (i.e. A_values is a flat 1D array of length ``nnz``; block = scalar).

    Args:
        A_offsets: CSR row offsets, length ``nrow + 1``, int32.
        A_columns: Column indices, length ``nnz``, int32.
        A_values: Scalar values, length ``nnz``, float64.
        x: Input, shape ``(R, N)`` float64.
        y: Output, shape ``(R, N)`` float64; written in-place.
    """
    rhs_idx, row = wp.tid()
    v = wp.float64(0.0)
    beg = A_offsets[row]
    end = A_offsets[row + 1]
    for block in range(beg, end):
        col = A_columns[block]
        v = v + A_values[block] * x[rhs_idx, col]
    y[rhs_idx, row] = v


@wp.kernel
def apply_permutation_multi_rhs_kernel(
    src: wp.array2d[wp.float64],
    perm: wp.array[wp.int32],
    dst: wp.array2d[wp.float64],
):
    """dst[r, i] = src[r, perm[i]] — multi-RHS gather.

    One thread per (rhs_idx, element_idx).

    Args:
        src: Source array, shape ``(R, N)`` float64.
        perm: Permutation, length ``N``, int32.
        dst: Destination array, shape ``(R, N)`` float64.
    """
    rhs_idx, i = wp.tid()
    dst[rhs_idx, i] = src[rhs_idx, perm[i]]


@wp.kernel
def scale_by_diag_multi_rhs_kernel(
    src: wp.array2d[wp.float64],
    diag: wp.array[wp.float64],
    dst: wp.array2d[wp.float64],
):
    """dst[r, i] = diag[i] * src[r, i] — multi-RHS diagonal scaling.

    One thread per (rhs_idx, element_idx).

    Args:
        src: Source array, shape ``(R, N)`` float64.
        diag: Diagonal coefficients, length ``N``, float64.
        dst: Destination array, shape ``(R, N)`` float64.
    """
    rhs_idx, i = wp.tid()
    dst[rhs_idx, i] = diag[i] * src[rhs_idx, i]
```

- [ ] **Step 2: Append two pack/unpack kernels for Jacobian axis packing**

Continue appending to the end of `kernels.py`:

```python


@wp.kernel
def pack_jacobian_axis_kernel(
    axis: int,
    contact_particle: wp.array[wp.int32],
    contact_dir: wp.array[wp.vec3],
    contact_alpha: wp.array[wp.float32],
    b_multi: wp.array2d[wp.float64],
):
    """Pack one axis of J^T into b_multi for multi-RHS solve.

    For RHS row ``r`` (i.e. contact row ``r``):
        b_multi[r, :] = 0.0 for all particles except particle[r],
        b_multi[r, particle[r]] = alpha[r] * dir[r][axis].

    Called with ``dim = total_rows``. Caller must zero ``b_multi`` before launch.

    Args:
        axis: Spatial axis to extract (0=x, 1=y, 2=z).
        contact_particle: Particle index per row, shape ``[total_rows]``, int32.
        contact_dir: Contact direction per row, shape ``[total_rows]``, vec3.
        contact_alpha: Jacobian coefficient per row, shape ``[total_rows]``, float32.
        b_multi: Output RHS buffer, shape ``(total_rows, N)`` float64.
    """
    r = wp.tid()
    p = contact_particle[r]
    d = contact_dir[r]
    alpha = wp.float64(contact_alpha[r])
    b_multi[r, p] = alpha * wp.float64(d[axis])


@wp.kernel
def unpack_to_A_inv_Jt_axis_kernel(
    axis: int,
    y_multi: wp.array2d[wp.float64],
    A_inv_Jt: wp.array2d[wp.vec3],
):
    """Write one axis of y_multi back into A_inv_Jt.

    A_inv_Jt[r, i][axis] = float32(y_multi[r, i]).

    Called with ``dim = (total_rows, N)``.

    Args:
        axis: Spatial axis to write (0=x, 1=y, 2=z).
        y_multi: Solved result, shape ``(total_rows, N)`` float64.
        A_inv_Jt: Output buffer, shape ``(total_rows, N)`` vec3; only axis is written.
    """
    r, i = wp.tid()
    v = A_inv_Jt[r, i]
    if axis == 0:
        v[0] = wp.float32(y_multi[r, i])
    elif axis == 1:
        v[1] = wp.float32(y_multi[r, i])
    else:
        v[2] = wp.float32(y_multi[r, i])
    A_inv_Jt[r, i] = v
```

- [ ] **Step 3: Run the FBA solver tests to confirm no regressions from adding kernels**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -20
```

Expected: `Ran N tests in ...s` — all passing (same count as before, no failures from new kernel definitions).

---

## Task 2: Add `solve_multi_rhs_scalar` to `FBALinearSolver`

**Files:**
- Modify: `newton/_src/solvers/fba/linear_solver.py` (add method after `solve`, before `build_schur_complement`)

### Context: How `solve` works (current single-RHS path)

Current `solve` pipeline for each of 3 components:
1. Extract scalar component c from vec3 b → `_b_scalar` (1D, N)
2. Gather by invperm: `_b_perm[i] = _b_scalar[invperm[i]]`
3. SpMV: `_Sb = S * _b_perm`
4. Scale: `_DSb[i] = Dinv[i] * _Sb[i]`
5. SpMV: `_SDSb = S^T * _DSb`
6. Gather by perm: `_x_scalar[i] = _SDSb[perm[i]]`
7. Insert scalar into vec3 x component c

The multi-RHS version does the same 5 steps but on a `(R, N)` 2D array simultaneously, replacing steps 2–6 with 2D kernel launches. Steps 1 and 7 (component extraction) are handled by `pack_jacobian_axis_kernel` and `unpack_to_A_inv_Jt_axis_kernel` in the caller.

### Context: BSR offsets/columns/values access

```python
# For a 1x1 BSR matrix (block_type=wp.float64):
# bsr.offsets: wp.array[wp.int32], length nrow+1
# bsr.columns: wp.array[wp.int32], length nnz
# bsr.values:  wp.array[wp.float64], length nnz  (1D, scalar per block)
offsets = self._S_bsr.offsets
columns = self._S_bsr.columns
values = self._S_bsr.values  # 1D float64
```

### Context: Scratch buffer growth strategy

Allocate `(max_total_rows, N)` on first call, with `max_total_rows = model.soft_contact_max * 3` (if available) or just `total_rows`. Grow by ≥1.5× when a larger `total_rows` is encountered. Buffers: `_b_perm_multi`, `_Sb_multi`, `_DSb_multi`, `_SDSb_multi`, `_b_multi_tmp`.

- [ ] **Step 4: Add `solve_multi_rhs_scalar` method to `FBALinearSolver`**

Add the following method to `FBALinearSolver` in `linear_solver.py`, immediately before `build_schur_complement` (after `solve`):

```python
    def _ensure_multi_rhs_buffers(self, R: int) -> None:
        """Allocate or grow multi-RHS scratch buffers to hold R right-hand sides."""
        n = self.n
        dev = self.device
        if hasattr(self, "_multi_rhs_cap") and self._multi_rhs_cap >= R:
            return
        # Grow by at least 1.5x to amortize repeated allocation.
        new_cap = max(R, int(getattr(self, "_multi_rhs_cap", 0) * 1.5) + 1)
        self._multi_rhs_cap = new_cap
        self._b_perm_multi = wp.empty(shape=(new_cap, n), dtype=wp.float64, device=dev)
        self._Sb_multi = wp.empty(shape=(new_cap, n), dtype=wp.float64, device=dev)
        self._DSb_multi = wp.empty(shape=(new_cap, n), dtype=wp.float64, device=dev)
        self._SDSb_multi = wp.empty(shape=(new_cap, n), dtype=wp.float64, device=dev)

    def solve_multi_rhs_scalar(
        self,
        b_multi: wp.array2d,
        x_multi: wp.array2d,
        R: int,
    ) -> None:
        """Apply A^{-1} to R scalar RHS rows in a single batched pass.

        Each row of ``b_multi`` is one scalar RHS (length N). Each row of
        ``x_multi`` receives the corresponding scalar solution A^{-1} b[r].

        Pipeline (same as :meth:`solve` but batched over all R rows):

        1. ``b_perm[r, i] = b_multi[r, invperm[i]]``  (multi-RHS invperm gather)
        2. ``Sb[r, i]     = sum_j S[i, j] * b_perm[r, j]``  (multi-RHS SpMV with S)
        3. ``DSb[r, i]    = Dinv[i] * Sb[r, i]``  (multi-RHS diag scale)
        4. ``SDSb[r, i]   = sum_j S^T[i, j] * DSb[r, j]``  (multi-RHS SpMV with S^T)
        5. ``x_multi[r, i] = SDSb[r, perm[i]]``  (multi-RHS perm gather)

        Args:
            b_multi: Input RHS rows, shape ``(≥R, N)`` float64.
            x_multi: Output solution rows, shape ``(≥R, N)`` float64; written in-place.
            R: Number of active RHS rows to process (must be ≤ b_multi.shape[0]).
        """
        from .kernels import (  # noqa: PLC0415
            apply_permutation_multi_rhs_kernel,
            bsr_mv_multi_rhs_scalar_kernel,
            scale_by_diag_multi_rhs_kernel,
        )

        self._ensure_multi_rhs_buffers(R)
        n = self.n
        dev = self.device

        # View active sub-slices (Warp array slicing returns a view, no copy).
        b_perm_v = wp.array(
            ptr=self._b_perm_multi.ptr,
            dtype=wp.float64,
            shape=(R, n),
            device=dev,
        )
        Sb_v = wp.array(ptr=self._Sb_multi.ptr, dtype=wp.float64, shape=(R, n), device=dev)
        DSb_v = wp.array(ptr=self._DSb_multi.ptr, dtype=wp.float64, shape=(R, n), device=dev)
        SDSb_v = wp.array(ptr=self._SDSb_multi.ptr, dtype=wp.float64, shape=(R, n), device=dev)

        # Step 1: b_perm[r, i] = b_multi[r, invperm[i]]
        wp.launch(
            apply_permutation_multi_rhs_kernel,
            dim=(R, n),
            inputs=[b_multi, self._invperm],
            outputs=[b_perm_v],
            device=dev,
        )

        # Step 2: Sb[r, i] = sum_j S[i,j] * b_perm[r, j]
        wp.launch(
            bsr_mv_multi_rhs_scalar_kernel,
            dim=(R, n),
            inputs=[
                self._S_bsr.offsets,
                self._S_bsr.columns,
                self._S_bsr.values,
                b_perm_v,
            ],
            outputs=[Sb_v],
            device=dev,
        )

        # Step 3: DSb[r, i] = Dinv[i] * Sb[r, i]
        wp.launch(
            scale_by_diag_multi_rhs_kernel,
            dim=(R, n),
            inputs=[Sb_v, self._Dinv],
            outputs=[DSb_v],
            device=dev,
        )

        # Step 4: SDSb[r, i] = sum_j S^T[i,j] * DSb[r, j]
        wp.launch(
            bsr_mv_multi_rhs_scalar_kernel,
            dim=(R, n),
            inputs=[
                self._ST_bsr.offsets,
                self._ST_bsr.columns,
                self._ST_bsr.values,
                DSb_v,
            ],
            outputs=[SDSb_v],
            device=dev,
        )

        # Step 5: x_multi[r, i] = SDSb[r, perm[i]]
        wp.launch(
            apply_permutation_multi_rhs_kernel,
            dim=(R, n),
            inputs=[SDSb_v, self._perm],
            outputs=[x_multi],
            device=dev,
        )
```

**Important:** The `wp.array(ptr=..., shape=...)` call creates a view into the pre-allocated buffer without copying data. This is the idiomatic Warp way to slice a 2D buffer without allocation.

- [ ] **Step 5: Run the FBA solver tests again to confirm `solve_multi_rhs_scalar` doesn't break anything**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -20
```

Expected: same pass count as before.

---

## Task 3: Write a correctness test for `solve_multi_rhs_scalar`

**Files:**
- Modify: `newton/tests/test_solver_fba.py` (add test class near existing `FBALinearSolverTests`)

The test must verify that `solve_multi_rhs_scalar` produces results matching the per-row `solve` path to within 1e-10 (both operate in float64, so very tight tolerance is achievable).

- [ ] **Step 6: Find the right place to insert the test**

Read lines 181–280 of `newton/tests/test_solver_fba.py` to locate the `FBALinearSolverTests` class boundary.

- [ ] **Step 7: Add the multi-RHS correctness test**

Locate the `FBALinearSolverTests` class in `newton/tests/test_solver_fba.py` and add this test method inside it, after `test_solve_random_spd`:

```python
    def test_solve_multi_rhs_matches_single_rhs(self):
        """solve_multi_rhs_scalar produces same result as R separate solve() calls."""
        import scipy.sparse as _sp

        np.random.seed(42)
        N = 12
        # Build a small random SPD matrix.
        A_dense = np.random.randn(N, N)
        A_dense = A_dense @ A_dense.T + N * np.eye(N)
        A = _sp.csr_matrix(A_dense)

        from newton._src.solvers.fba.linear_solver import (
            FBALinearSolver,
            FactorizedSystem,
            factorize_and_sparse_inverse,
        )

        fs = factorize_and_sparse_inverse(A)
        try:
            device = wp.get_cuda_device()
        except Exception:
            device = wp.get_device("cpu")
        solver = FBALinearSolver(fs, device=device)

        R = 5
        # Build R random scalar RHS rows (shape (R, N)).
        b_np = np.random.randn(R, N)

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
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6,
                                   err_msg="solve_multi_rhs_scalar disagrees with single-rhs solve()")
```

- [ ] **Step 8: Run the new test alone to verify it passes**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_solve_multi_rhs_matches_single_rhs 2>&1 | tail -20
```

Expected output: `OK` with the test passing.

---

## Task 4: Hook multi-RHS solve into `build_schur_complement`

**Files:**
- Modify: `newton/_src/solvers/fba/linear_solver.py` (rewrite the per-row loop section of `build_schur_complement`)

### Context: Current flow (lines 884–907 in linear_solver.py)

The current per-row loop:
```python
for row in range(total_rows):
    # zero jcol, set one non-zero, call self.solve(jcol, y_out), wp.copy row
```
This fires ~15 kernel launches per row × `total_rows` rows = ~3600 launches for M=64 Stage B.

### Context: New flow

Replace the per-row loop with 3 axis passes:
1. Zero `b_multi` (all rows/cols).
2. Launch `pack_jacobian_axis_kernel(axis, ...)` with `dim=total_rows` — sets `b_multi[r, particle[r]] = alpha[r] * dir[r][axis]` for each row r.
3. Launch `solve_multi_rhs_scalar(b_multi, y_multi, total_rows)` — 5 kernels total.
4. Launch `unpack_to_A_inv_Jt_axis_kernel(axis, y_multi, _A_inv_Jt_d)` with `dim=(total_rows, n)`.

After 3 axis passes, `_A_inv_Jt_d` contains the same data as before. The rest of `build_schur_complement` (W device kernel + host pull) is unchanged.

### Context: Buffer management

The `b_multi` and `y_multi` 2D buffers are `(total_rows, N)` float64. Allocate lazily alongside `_A_inv_Jt_d`. Reuse `_ensure_multi_rhs_buffers` for the solve scratch; add separate `_b_multi_d` and `_y_multi_d` for the Jacobian pack/unpack.

- [ ] **Step 9: Add imports at the top of `build_schur_complement` body**

Locate the import block near line 833 of `linear_solver.py`:
```python
        from .kernels import (  # noqa: PLC0415
            build_contact_jacobian_dir_kernel,
            zero_vec3_kernel,
        )
```
Replace it with:
```python
        from .kernels import (  # noqa: PLC0415
            pack_jacobian_axis_kernel,
            unpack_to_A_inv_Jt_axis_kernel,
        )
```

- [ ] **Step 10: Replace the per-row solve loop and buffer setup**

Find and replace the section in `build_schur_complement` that:
1. Allocates/reuses `_A_inv_Jt_d`.
2. Allocates `jcol` and `y_out` scratch buffers.
3. Runs the per-row loop (`for row in range(total_rows): ...`).

Replace with:

```python
        # --- Allocate/reuse A_inv_Jt buffer: shape (total_rows, N) on device ---
        if not hasattr(self, "_A_inv_Jt_d") or self._A_inv_Jt_d.shape[0] < total_rows or self._A_inv_Jt_d.shape[1] != n:
            self._A_inv_Jt_d = wp.empty(shape=(total_rows, n), dtype=wp.vec3, device=dev)

        # --- Allocate/reuse 2D float64 buffers for multi-RHS pack/solve ---
        if (
            not hasattr(self, "_b_multi_d")
            or self._b_multi_d.shape[0] < total_rows
            or self._b_multi_d.shape[1] != n
        ):
            cap = max(total_rows, int(getattr(self, "_b_multi_cap", 0) * 1.5) + 1)
            self._b_multi_cap = cap
            self._b_multi_d = wp.zeros(shape=(cap, n), dtype=wp.float64, device=dev)
            self._y_multi_d = wp.zeros(shape=(cap, n), dtype=wp.float64, device=dev)

        # Active sub-view (no copy — pointer arithmetic into pre-allocated buffer).
        b_multi_v = wp.array(
            ptr=self._b_multi_d.ptr,
            dtype=wp.float64,
            shape=(total_rows, n),
            device=dev,
        )
        y_multi_v = wp.array(
            ptr=self._y_multi_d.ptr,
            dtype=wp.float64,
            shape=(total_rows, n),
            device=dev,
        )

        # --- Approach B: 3 axis passes (one multi-RHS solve per axis) ---
        for axis in range(3):
            # Zero b_multi rows for this axis pass.
            b_multi_v.zero_()
            # Pack J^T axis: b_multi[r, particle[r]] = alpha[r] * dir[r][axis].
            wp.launch(
                pack_jacobian_axis_kernel,
                dim=total_rows,
                inputs=[axis, row_particle_d, row_dir_d, row_alpha_d],
                outputs=[b_multi_v],
                device=dev,
            )
            # Batched solve: y_multi = A^{-1} * b_multi (all total_rows RHS at once).
            self._ensure_multi_rhs_buffers(total_rows)
            self.solve_multi_rhs_scalar(b_multi_v, y_multi_v, total_rows)
            # Unpack: A_inv_Jt[r, i][axis] = y_multi[r, i].
            wp.launch(
                unpack_to_A_inv_Jt_axis_kernel,
                dim=(total_rows, n),
                inputs=[axis],
                outputs=[y_multi_v, self._A_inv_Jt_d],
                device=dev,
            )
```

**Note on `unpack_to_A_inv_Jt_axis_kernel` argument order**: The kernel signature is `(axis, y_multi, A_inv_Jt)` with `axis` as an `int` input, `y_multi` as input array, and `A_inv_Jt` as output array. Verify that Warp launch call matches the kernel's `inputs` and `outputs` lists exactly.

- [ ] **Step 11: Run all FBA tests to verify correctness**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -30
```

Expected: 72+ tests, all passing. The critical ones:
- `test_no_contact_passes_through`
- `test_single_plane_contact_pushes_particle_away`
- `test_multiple_plane_contacts_no_interpenetration`
- `test_static_friction_holds_particle_on_plane`
- `test_friction_multi_shape_scene_runs`

If any test fails, diagnose with verbose output:
```bash
uv run --extra dev -m newton.tests -k <failing_test_name> 2>&1 | grep -A 20 "FAIL\|Error"
```

---

## Task 5: Verify W matrix bit-equivalence

**Files:**
- Modify: `newton/tests/test_solver_fba.py` (add one targeted test)

- [ ] **Step 12: Add a W-equivalence test comparing Approach A and Approach B outputs**

Add this test to the `PhaseContactSchurTests` class (near `test_schur_W_is_positive_definite_for_active_contacts`):

```python
    def test_approach_b_W_matches_approach_a(self):
        """Approach B multi-RHS schur build produces W within 1e-7 of Approach A."""
        import scipy.sparse as _sp
        from newton._src.solvers.fba.linear_solver import (
            FBALinearSolver,
            FactorizedSystem,
            factorize_and_sparse_inverse,
        )

        np.random.seed(7)
        N = 20
        M = 5  # contacts

        A_dense = np.random.randn(N, N)
        A_dense = A_dense @ A_dense.T + N * np.eye(N)
        A = _sp.csr_matrix(A_dense)
        fs = factorize_and_sparse_inverse(A)

        try:
            device = wp.get_cuda_device()
        except Exception:
            device = wp.get_device("cpu")

        # Build random contact arrays.
        np.random.seed(13)
        j_indices = np.random.randint(0, N, size=M, dtype=np.int32)
        j_normals = np.random.randn(M, 3).astype(np.float32)
        j_normals /= np.linalg.norm(j_normals, axis=1, keepdims=True) + 1e-8
        j_alpha = np.ones(M, dtype=np.float32)

        j_indices_d = wp.array(j_indices, dtype=wp.int32, device=device)
        j_normals_d = wp.array(j_normals, dtype=wp.vec3, device=device)
        j_alpha_d = wp.array(j_alpha, dtype=wp.float32, device=device)

        # Call build_schur_complement — this now uses Approach B internally.
        solver = FBALinearSolver(fs, device=device)
        W_b = solver.build_schur_complement(M, j_indices_d, j_normals_d, j_alpha_d)

        # Reference: compute W analytically via numpy (A^{-1} applied to each J^T column).
        A_inv = np.linalg.inv(A_dense)
        J = np.zeros((M, N), dtype=np.float64)
        for c in range(M):
            J[c, j_indices[c]] = float(j_alpha[c])
            # Multiply by direction component-wise — for Stage A W = J A^{-1} J^T
            # where J[c] only contributes via the normal direction, but since we
            # call build_schur_complement for Stage A (no tangents), each row's
            # contribution is alpha * n projected to scalar.
            # Actually Stage A: J is already built as alpha * dir inside the kernel.
            J[c, j_indices[c]] = float(j_alpha[c]) * float(np.linalg.norm(j_normals[c]))

        # Recompute properly: W_ref[cp, c] = alpha[cp] * dot(n[cp], (A^{-1} J^T)[c, particle[cp]])
        # where J^T column c = alpha[c] * dir[c] scattered at particle[c].
        Jt_cols = np.zeros((M, N), dtype=np.float64)  # each col is one J^T column
        for c in range(M):
            p = j_indices[c]
            for ax in range(3):
                Jt_cols[c, p] += float(j_alpha[c]) * float(j_normals[c, ax])  # not right

        # Simpler: compute directly via A_inv.
        # (A^{-1} J^T)[c, i] = A_inv @ e_p * alpha * n   projected at particle p
        # W_ref[cp, c] = alpha[cp] * dot(n[cp], A_inv[p_cp, :] * alpha[c] * n[c])  # no: J is vec3

        # Actually W is scalar (Stage A): W[cp, c] = J_cp · A^{-1} · J_c^T
        # where J_cp and J_c are row vectors with entry alpha * n_component at particle p.
        # For Stage A scalar W: both J rows are scalar (just one component used).
        # The kernel treats the full vec3 direction; the result is sum over 3 components.
        # W[cp, c] = sum_ax (alpha[cp] * n[cp, ax]) * (A_inv[p_cp, p_c] * alpha[c] * n[c, ax])
        W_ref = np.zeros((M, M), dtype=np.float64)
        for cp in range(M):
            p_cp = j_indices[cp]
            a_cp = float(j_alpha[cp])
            n_cp = j_normals[cp].astype(np.float64)
            for c in range(M):
                p_c = j_indices[c]
                a_c = float(j_alpha[c])
                n_c = j_normals[c].astype(np.float64)
                # A^{-1} applied to alpha*n scattered at p_c gives A_inv[:, p_c] * alpha_c * n_c[ax]
                # At particle p_cp: (A^{-1} J_c^T)[p_cp, ax] = A_inv[p_cp, p_c] * a_c * n_c[ax]
                # W[cp, c] = alpha[cp] * sum_ax n_cp[ax] * A_inv[p_cp, p_c] * a_c * n_c[ax]
                W_ref[cp, c] = a_cp * float(np.dot(n_cp, A_inv[p_cp, p_c] * a_c * n_c))

        np.testing.assert_allclose(W_b, W_ref, rtol=1e-6, atol=1e-7,
                                   err_msg="Approach B W diverges from analytical reference")
```

- [ ] **Step 13: Run the W-equivalence test**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_approach_b_W_matches_approach_a 2>&1 | tail -20
```

Expected: `OK`.

---

## Task 6: First commit

- [ ] **Step 14: Lint and format**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uvx pre-commit run -a 2>&1 | tail -30
```

If pre-commit fixes files, re-run tests to verify they still pass.

- [ ] **Step 15: Commit the kernel additions**

```bash
cd /home/ziqiu/work/newton
git add newton/_src/solvers/fba/kernels.py newton/tests/test_solver_fba.py
git commit -m "$(cat <<'EOF'
Add multi-RHS BSR SpMV, permutation, and pack/unpack kernels

Five new Warp kernels for Approach B batched Schur build:
- bsr_mv_multi_rhs_scalar_kernel: y[r,i] = sum_j A[i,j] x[r,j]
- apply_permutation_multi_rhs_kernel: dst[r,i] = src[r,perm[i]]
- scale_by_diag_multi_rhs_kernel: dst[r,i] = diag[i]*src[r,i]
- pack_jacobian_axis_kernel: packs one spatial axis of J^T
- unpack_to_A_inv_Jt_axis_kernel: writes solved axis back to vec3 buf

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Benchmark and performance report

- [ ] **Step 16: Run Demo 1 and record timings**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run python scripts/fba_demo1_cloth_on_plane.py 2>&1 | grep -E "Stage|mean|ms|step"
```

Record: Stage A mean step time, Stage B mean step time.

- [ ] **Step 17: Run Demo 3 and record timings**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run python scripts/fba_demo3_cloth_on_cylinder.py 2>&1 | grep -E "Stage|mean|ms|step"
```

Record: Stage A mean step time, Stage B mean step time.

- [ ] **Step 18: Compute speedup vs Approach A**

Approach A (from README.md):
- Demo 1 Stage A: 137.9 ms/step
- Demo 1 Stage B: 503.7 ms/step
- Demo 3 Stage B: ~1302 ms/step

Compute: Speedup = Approach_A_time / Approach_B_time for each.

---

## Task 8: Update README and second commit

**Files:**
- Modify: `scripts/contact_demos_out/README.md`

- [ ] **Step 19: Update README.md with Approach B numbers**

Edit `scripts/contact_demos_out/README.md` to:
1. Add a new "Approach B" row in each demo table with measured numbers.
2. Add a comparison section at the top if both approaches are documented.
3. Update the "Known Limitations" section to note that the per-row loop has been replaced.

The tables should keep both Approach A and B columns side-by-side:

| Variant | Approach A | Approach B | Speedup |
|---|---|---|---|
| Stage A (friction=False) | 137.9 ms | [measured] ms | [ratio]x |
| Stage B (friction=True, mu=0.3) | 503.7 ms | [measured] ms | [ratio]x |

- [ ] **Step 20: Commit hook and README update**

```bash
cd /home/ziqiu/work/newton
git add newton/_src/solvers/fba/linear_solver.py scripts/contact_demos_out/README.md
git commit -m "$(cat <<'EOF'
Hook multi-RHS solve into build_schur_complement (Approach B)

Replaces the sequential per-row solve loop (~3600 kernel launches for
M=64 Stage B) with 3 batched axis passes using solve_multi_rhs_scalar.
Each axis pass fires only 5 kernel launches regardless of total_rows,
reducing launch overhead by ~720x for the 3-axis (Stage B) path.

Includes FBALinearSolver.solve_multi_rhs_scalar with lazy-growth scratch
buffers (≥1.5x growth). Updates contact_demos_out/README.md with Approach B
performance numbers and speedup vs Approach A.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final test suite run

- [ ] **Step 21: Run all FBA tests for the final count**

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ziqiu/work/newton
uv run --extra dev -m newton.tests -k test_solver_fba 2>&1 | tail -10
```

Expected: `Ran N tests in Xs` with N ≥ 72 and 0 failures.

---

## Self-Review

### Spec Coverage Check

| Spec requirement | Task |
|---|---|
| Multi-RHS BSR SpMV kernel | Task 1, Step 1 |
| `apply_permutation_multi_rhs_kernel` | Task 1, Step 1 |
| `scale_by_diag_multi_rhs_kernel` | Task 1, Step 1 |
| `solve_multi_rhs_scalar` method | Task 2, Step 4 |
| Persistent scratch buffers with ≥1.5x growth | Task 2, Step 4 (`_ensure_multi_rhs_buffers`) |
| Pack/unpack kernels for axis-wise J^T | Task 1, Step 2 |
| Hook into `build_schur_complement` (3-axis loop) | Task 4, Steps 9–10 |
| All 71+ tests pass | Tasks 3, 5, 9 |
| W values within 1e-7 of Approach A | Task 5 |
| Benchmark Demo 1 + Demo 3 | Task 7 |
| Update README | Task 8 |
| Two commits (kernels, hook) | Tasks 6, 8 |

### Potential Issues

1. **`unpack_to_A_inv_Jt_axis_kernel` `inputs`/`outputs` split**: The kernel signature has `axis` (scalar int) and `y_multi` (read-only array) as inputs, and `A_inv_Jt` (read-write array) as output. Warp `wp.launch` requires `inputs` to be arguments the kernel reads, and `outputs` to be arguments it writes. For in-place read-write access, pass `A_inv_Jt` in `inputs` — Warp doesn't enforce the distinction at the Python level, it's just convention. **Decision**: put `y_multi` and `A_inv_Jt` both in `inputs` to avoid confusion, since `A_inv_Jt` is read-modify-write.

2. **`wp.array(ptr=..., shape=...)` for buffer views**: This is valid Warp idiom. If `ptr` isn't exposed, use `self._b_multi_d.numpy()` round-trip as fallback (slower but correct). Prefer the ptr approach.

3. **BSR `values` for 1x1 blocks is 1D**: Confirmed from `bsr_matrix_t` source — `values: wp.array(dtype=dtype)` where dtype=`wp.float64` gives a 1D scalar array. The `bsr_mv_kernel` accesses it via `scalar_values` (3D view), but our custom kernel accesses `values[block]` directly as a scalar.

4. **`b_multi_v.zero_()`**: Warp arrays have a `zero_()` method. Alternatively use `wp.launch(zero_scalar_kernel, dim=total_rows*n, inputs=[b_multi_v.flatten()])` if `.zero_()` isn't available on 2D arrays. Prefer `wp.zeros_like(b_multi_v)` style — but that allocates. Use `b_multi_v.fill_(0.0)` as fallback.
