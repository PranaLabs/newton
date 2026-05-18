# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional cupy + cuBLAS DGEMV bridge used by the NSN PCR solver.

Validates:
    * Graceful skip when cupy is not installed (no crash, no test failure).
    * Zero-copy interop: a Warp array and its cupy view share the same device
      buffer (modifications round-trip through both APIs).
    * Numerical parity with ``numpy``'s reference matvec at fp64 noise level.
    * Strided slice handling: passing a non-contiguous ``[:n, :n]`` slice of a
      larger preallocated buffer (as the PCR solver does in practice) still
      produces the correct result.

These tests cover the bridge itself; the PCR solver's end-to-end behavior is
already exercised by ``test_fba_nsn_pcr`` -- the existing PCR tests
automatically run through the cuBLAS path when cupy is installed.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba.cublas_interop import (
    cublas_dgemv,
    is_cublas_available,
    wp_to_cupy,
)


class _CublasInteropBase(unittest.TestCase):
    """Common setup: CUDA-only, cupy-or-skip."""

    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA device required for cuBLAS interop tests")
        if not is_cublas_available():
            raise unittest.SkipTest("cupy not installed; install with `pip install newton[cublas]`")
        cls.device = "cuda:0"


class TestCublasAvailability(unittest.TestCase):
    """Availability flag returns a bool either way (does not raise)."""

    def test_is_cublas_available_returns_bool(self):
        self.assertIsInstance(is_cublas_available(), bool)


class TestWpToCupyZeroCopy(_CublasInteropBase):
    """``wp_to_cupy`` yields a zero-copy view, not a copy."""

    def test_zero_copy_view_modify_via_cupy_visible_to_wp(self):
        # Allocate via Warp, view as cupy, modify via cupy, read via Warp.
        # If the buffer were copied at any point the Warp side would see the
        # original values and the assertion would fail.
        arr_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        wp_arr = wp.from_numpy(arr_np, dtype=wp.float64, device=self.device)
        cp_view = wp_to_cupy(wp_arr)
        # In-place +10 through cupy.
        cp_view += 10.0
        wp.synchronize_device(self.device)
        np.testing.assert_array_equal(wp_arr.numpy(), arr_np + 10.0)

    def test_zero_copy_view_modify_via_wp_visible_to_cupy(self):
        # Reverse: modify via Warp kernel-ish path (just .assign here), read
        # via cupy.  cupy reads the same device memory after a stream sync.
        wp_arr = wp.zeros(8, dtype=wp.float64, device=self.device)
        cp_view = wp_to_cupy(wp_arr)
        # Push a known pattern in via Warp's host->device copy.
        pattern = np.arange(8, dtype=np.float64)
        wp_arr.assign(pattern)
        wp.synchronize_device(self.device)
        # cupy view must read the same pattern.
        import cupy as cp  # noqa: PLC0415 — only runs when cupy is available

        np.testing.assert_array_equal(cp.asnumpy(cp_view), pattern)

    def test_zero_copy_view_2d_strided_slice(self):
        # The PCR solver passes a ``[:n, :n]`` slice of a larger preallocated
        # buffer.  Slice rows have stride ``max_n`` but length ``n``; cupy /
        # cuBLAS must respect those strides.  Verify the data pointer of the
        # cupy view matches the slice's CAI data pointer (no copy).
        max_n = 16
        n = 5
        full = wp.zeros((max_n, max_n), dtype=wp.float64, device=self.device)
        sl = full[:n, :n]
        cp_view = wp_to_cupy(sl)
        cai = sl.__cuda_array_interface__
        self.assertEqual(cp_view.data.ptr, cai["data"][0])
        # cupy strides are in bytes; slice's column stride is 8 (fp64).  Row
        # stride must be ``max_n * 8`` (whole-row stride of the parent buffer).
        self.assertEqual(cp_view.strides, (max_n * 8, 8))


class TestCublasDgemvNumerical(_CublasInteropBase):
    """``cublas_dgemv`` matches ``numpy.matmul`` at fp64 noise level."""

    def test_dgemv_matches_numpy_small(self):
        # Small dense system -- pure correctness, no perf consideration.
        m, n = 16, 16
        rng = np.random.default_rng(42)
        A_np = rng.standard_normal((m, n))
        x_np = rng.standard_normal(n)
        A = wp.array2d(A_np.astype(np.float64), dtype=wp.float64, device=self.device)
        x = wp.from_numpy(x_np.astype(np.float64), dtype=wp.float64, device=self.device)
        y = wp.zeros(m, dtype=wp.float64, device=self.device)
        cublas_dgemv(A, x, y, device=self.device)
        wp.synchronize_device(self.device)
        np.testing.assert_allclose(y.numpy(), A_np @ x_np, atol=1.0e-10)

    def test_dgemv_matches_numpy_demo_size(self):
        # n=900 mirrors Demo 5 Stage B 3M-sized Schur block.
        n = 900
        rng = np.random.default_rng(7)
        A_np = rng.standard_normal((n, n))
        # SPD-flavoured so the magnitude of A @ x is bounded.
        A_np = (A_np @ A_np.T) / n + np.eye(n)
        x_np = rng.standard_normal(n)
        A = wp.array2d(A_np.astype(np.float64), dtype=wp.float64, device=self.device)
        x = wp.from_numpy(x_np.astype(np.float64), dtype=wp.float64, device=self.device)
        y = wp.zeros(n, dtype=wp.float64, device=self.device)
        cublas_dgemv(A, x, y, device=self.device)
        wp.synchronize_device(self.device)
        # 1e-10 absolute tol on a fp64 GEMV at n=900 with bounded entries.
        np.testing.assert_allclose(y.numpy(), A_np @ x_np, atol=1.0e-10)

    def test_dgemv_strided_slice(self):
        # The PCR fast path passes ``A_view = A_full[:n, :n]`` -- a strided
        # slice view of a larger buffer.  cuBLAS must handle the stride.
        max_n = 64
        n = 13
        rng = np.random.default_rng(11)
        full_np = rng.standard_normal((max_n, max_n))
        full = wp.array2d(full_np.astype(np.float64), dtype=wp.float64, device=self.device)
        sl = full[:n, :n]
        x_np = rng.standard_normal(n)
        x = wp.from_numpy(x_np.astype(np.float64), dtype=wp.float64, device=self.device)
        y = wp.zeros(n, dtype=wp.float64, device=self.device)
        cublas_dgemv(sl, x, y, device=self.device)
        wp.synchronize_device(self.device)
        np.testing.assert_allclose(y.numpy(), full_np[:n, :n] @ x_np, atol=1.0e-10)

    def test_dgemv_alpha_beta(self):
        # General form y = alpha * A @ x + beta * y.  Verify with a known
        # initial y so beta != 0 actually contributes.
        n = 8
        rng = np.random.default_rng(13)
        A_np = rng.standard_normal((n, n))
        x_np = rng.standard_normal(n)
        y0_np = rng.standard_normal(n)
        A = wp.array2d(A_np.astype(np.float64), dtype=wp.float64, device=self.device)
        x = wp.from_numpy(x_np.astype(np.float64), dtype=wp.float64, device=self.device)
        y = wp.from_numpy(y0_np.astype(np.float64), dtype=wp.float64, device=self.device)
        alpha, beta = 2.5, -0.75
        cublas_dgemv(A, x, y, device=self.device, alpha=alpha, beta=beta)
        wp.synchronize_device(self.device)
        expected = alpha * (A_np @ x_np) + beta * y0_np
        np.testing.assert_allclose(y.numpy(), expected, atol=1.0e-10)


class TestPCRWithoutCublas(unittest.TestCase):
    """The PCR solver must still work when cupy is forcibly disabled.

    Validates the fallback branch.  ``use_cublas=False`` is an explicit
    constructor override -- the same branch the FBA solver hits on systems
    where the ``cublas`` extra is not installed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA device required for PCR fallback test")
        cls.device = "cuda:0"

    def test_pcr_solves_with_cublas_disabled(self):
        from newton._src.solvers.fba.nsn_pcr_solver import NSNPCRSolver  # noqa: PLC0415

        n = 50
        rng = np.random.default_rng(101)
        M = rng.standard_normal((n, n))
        A_np = (M @ M.T) / n + np.eye(n)
        b_np = rng.standard_normal(n)
        A = wp.array2d(A_np.astype(np.float64), dtype=wp.float64, device=self.device)
        b = wp.from_numpy(b_np.astype(np.float64), dtype=wp.float64, device=self.device)
        x_out = wp.zeros(n, dtype=wp.float64, device=self.device)
        # Force fallback path even if cupy is installed.
        solver = NSNPCRSolver(max_n=n, device=self.device, tol=1.0e-10, max_iter=500, use_cublas=False)
        iters, _ = solver.solve(A, b, x_out)
        self.assertGreater(iters, 0)
        rel_err = np.linalg.norm(x_out.numpy() - np.linalg.solve(A_np, b_np)) / np.linalg.norm(b_np)
        self.assertLess(rel_err, 1.0e-7)


if __name__ == "__main__":
    unittest.main()
