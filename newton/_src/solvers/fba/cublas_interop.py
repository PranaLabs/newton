# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional cupy + cuBLAS DGEMV path for the FBA NSN-Schur PCR solver.

This module is a thin bridge between Warp's device arrays and cuBLAS via
cupy's low-level :mod:`cupy_backends.cuda.libs.cublas` binding.  It exists
purely as a perf optimization: the in-house ``tile_matmul`` matvec kernel in
:mod:`nsn_pcr_solver` runs at ~12-90 us per call for the n ~ 600..3000 dense
fp64 Schur systems Demo 4/5 produce; cuBLAS DGEMV on the same hardware
(RTX 5090) is ~1.5-3.5x faster at n ≥ 2000 because NVIDIA's hand-tuned fp64
kernels hit higher SM occupancy at these sizes.

When ``cupy`` is not installed the FBA solver falls back to the pure-Warp
matvec path automatically (see :func:`is_cublas_available`).  cupy is a
purely-optional dependency listed under the ``cublas`` extra in
``pyproject.toml`` and is **not** required for the FBA solver to function.

Interop strategy:
    * Warp arrays expose ``__cuda_array_interface__`` (CAI) v2 as of Warp 1.x;
      :func:`cupy.asarray` accepts CAI directly and produces a zero-copy
      :class:`cupy.ndarray` view sharing the same device buffer.
    * Warp's per-device stream is bound to cuBLAS once at construction via
      :func:`cublas.setStream`; all subsequent DGEMV calls inherit Warp's
      stream ordering without any per-call ``with`` context overhead.

Why not :func:`cupy.matmul` or :func:`cupy.cublas.gemv`?
    Both wrappers internally fall back to a slow strided-gemv kernel when the
    matrix is a non-contiguous slice (e.g. ``A_full[:n, :n]``) -- which is
    *exactly* how the FBA solver passes ``A`` to PCR (the Schur LHS lives in
    a preallocated ``max_n x max_n`` buffer and only the leading ``[M, M]``
    block is active).  Measured on RTX 5090 fp64 (2026-05-18):

        n=3000  cupy.matmul on strided slice:  200 us
        n=3000  cupy.matmul on contiguous:      28 us
        n=3000  direct DGEMV (lda=max_n):       28 us

    Calling :func:`cublas.dgemv` directly with the correct leading dimension
    (``lda = parent_row_stride / dtype_size``) routes through the fast
    contiguous DGEMV kernel path even when the matrix is a sub-block of a
    larger buffer.  This is the canonical cuBLAS idiom and matches RealSim's
    ``CUDADenseJacobiPCRSolver`` C++ literal use of ``cublasDgemv``.

Row/column-major handling:
    cuBLAS is column-major (Fortran).  Warp's ``wp.array2d`` is row-major (C)
    -- viewing a row-major (m, n) buffer as a column-major matrix yields the
    *transpose* ``A_F = A_C^T`` (rows=n, cols=m, leading dim = parent row
    stride).  To compute ``y = A_C @ x`` in row-major semantics we tell
    cuBLAS to compute ``y = A_F^T @ x`` -- i.e. call DGEMV with op='T' and
    the (m, n) shape of ``A_C`` swapped to (n, m) for cuBLAS.  This restores
    the user-facing ``y = A_C @ x`` while letting cuBLAS use its fast
    contiguous-LDA kernel.
"""

from __future__ import annotations

import numpy as np
import warp as wp

try:
    import cupy as _cp
    from cupy_backends.cuda.libs import cublas as _cublas

    _CUPY_AVAILABLE = True
except ImportError:
    _cp = None
    _cublas = None
    _CUPY_AVAILABLE = False


def is_cublas_available() -> bool:
    """Whether cupy (and therefore cuBLAS) is importable.

    Returns:
        ``True`` if :mod:`cupy` imports successfully, else ``False``.
    """
    return _CUPY_AVAILABLE


def wp_to_cupy(arr: wp.array):
    """Return a zero-copy cupy view of a Warp device array.

    Args:
        arr: Warp array on a CUDA device.  Must expose
            ``__cuda_array_interface__`` (true for any Warp 1.x device array).

    Returns:
        A :class:`cupy.ndarray` sharing the same device buffer as ``arr``.
        Modifications through the cupy view are visible through the Warp
        array and vice versa -- no copy is performed.

    Raises:
        RuntimeError: if cupy is not installed.  Call :func:`is_cublas_available`
            first to gate the path.
    """
    if not _CUPY_AVAILABLE:
        raise RuntimeError("cupy is not installed; install with `pip install cupy-cuda12x`.")
    return _cp.asarray(arr)


def get_cublas_handle_for_stream(device: wp.Device | str):
    """Get cupy's cuBLAS handle and bind it to Warp's stream.

    cuBLAS operations queued via this handle inherit Warp's per-device stream
    ordering, so cuBLAS DGEMV calls interleave correctly with the surrounding
    Warp kernel launches without any explicit synchronization.

    Args:
        device: Warp device or device string (e.g. ``"cuda:0"``).

    Returns:
        Integer cuBLAS handle ready for :func:`cublas.dgemv` and friends.

    Raises:
        RuntimeError: if cupy is not installed.
    """
    if not _CUPY_AVAILABLE:
        raise RuntimeError("cupy is not installed; install with `pip install cupy-cuda12x`.")
    handle = _cp.cuda.device.get_cublas_handle()
    _cublas.setStream(handle, wp.get_stream(device).cuda_stream)
    return handle


def cublas_dgemv(
    A: wp.array2d[wp.float64],
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
    device: wp.Device | str | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    handle: int | None = None,
) -> None:
    """Compute ``y = alpha * (A @ x) + beta * y`` via cuBLAS DGEMV.

    Direct binding to ``cublasDgemv_v2``; bypasses :func:`cupy.matmul` and
    :func:`cupy.cublas.gemv` because both wrappers fall back to a slow strided
    kernel when ``A`` is a non-contiguous slice (see module docstring for
    measurements).

    The matrix is interpreted in row-major C order; the underlying cuBLAS
    call uses ``CUBLAS_OP_T`` on the column-major view to recover row-major
    semantics, with ``lda`` set from ``A``'s row stride.  This produces the
    expected ``y = A @ x`` for both contiguous and strided slices.

    Args:
        A: ``(m, n)`` row-major fp64 device matrix.  Strided slices (e.g.
            ``A_full[:m, :n]``) are supported and run on cuBLAS's fast DGEMV
            kernel -- the row stride of the *parent* buffer becomes the
            cuBLAS leading dimension.
        x: ``(n,)`` fp64 device input vector.  Must be unit-stride
            (contiguous).
        y: ``(m,)`` fp64 device output vector.  Must be unit-stride.  When
            ``beta != 0`` the prior contents are scaled into the result.
        device: Warp device for stream binding.  Defaults to the device the
            output array lives on.
        alpha: Scalar multiplier on ``A @ x``.
        beta: Scalar multiplier on the pre-existing contents of ``y``.
        handle: Optional cuBLAS handle (from :func:`get_cublas_handle_for_stream`).
            If ``None``, the per-call handle lookup adds ~2 us; callers in
            hot loops should cache the handle.

    Raises:
        RuntimeError: if cupy is not installed.
    """
    if not _CUPY_AVAILABLE:
        raise RuntimeError("cublas_dgemv requires cupy. Install with `pip install cupy-cuda12x`.")
    dev = device if device is not None else y.device
    if handle is None:
        handle = get_cublas_handle_for_stream(dev)

    A_cai = A.__cuda_array_interface__
    x_cai = x.__cuda_array_interface__
    y_cai = y.__cuda_array_interface__
    m, n_cols = A_cai["shape"]
    # Row stride / 8 = leading dim of the column-major view (lda).  For a
    # contiguous (m, n_cols) row-major buffer this equals n_cols; for a
    # strided slice it equals the parent buffer's column count.
    lda = A_cai["strides"][0] // 8
    # In cuBLAS column-major terms the buffer is (n_cols, m) with leading
    # dimension ``lda``.  CUBLAS_OP_T transposes that to recover the
    # row-major (m, n_cols) view we want.  After op='T', cuBLAS reads the
    # input vector ``x`` of length ``n_cols`` and writes the output ``y`` of
    # length ``m`` -- exactly ``y = A @ x``.
    alpha_p = np.array([alpha], dtype=np.float64)
    beta_p = np.array([beta], dtype=np.float64)
    _cublas.dgemv(
        handle,
        _cublas.CUBLAS_OP_T,
        n_cols,  # cuBLAS-col-major rows = row-major cols
        m,  # cuBLAS-col-major cols = row-major rows
        alpha_p.ctypes.data,
        A_cai["data"][0],
        lda,
        x_cai["data"][0],
        1,  # incx
        beta_p.ctypes.data,
        y_cai["data"][0],
        1,  # incy
    )
