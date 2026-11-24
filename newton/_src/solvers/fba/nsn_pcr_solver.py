# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Jacobi-preconditioned Conjugate Residual (PCR) solver for the NSN Schur LCP.

The solver is used inside the Non-Smooth Newton (NSN) outer loop to solve the
dense fp64 SPD reduced (Schur) system on every Newton step, and the sparse
Schur system of the lite mode.

Algorithm: conjugate residuals on the symmetrically preconditioned system
``P^1/2 A P^1/2``, with the Jacobi preconditioner ``P = diag(A)^-1``:

    Initial:
        x    = 0
        r    = b
        z    = P * r
        p    = z
        q    = A * p
        rho  = <z, q>

    While |r| > tol * |b| and iter < max_iter:
        t     = P * q
        den   = <q, t>
        alpha = rho / den
        x    += alpha * p
        r    -= alpha * q
        z    -= alpha * t
        t     = A * z
        rho_old = rho
        rho   = <z, t>
        beta  = rho / rho_old
        p     = z + beta * p
        q     = t + beta * q

Notes:
    * ``x = 0`` always: a solve takes no warm start.
    * ``max_iter`` is capped at ``n``.
    * A direction ``p`` whose curvature ``pᵀ A p / pᵀ diag(A) p`` falls below
      ``1 / max_inverse_gain`` would amplify the residual by more than
      ``max_inverse_gain``. From such a near-null direction on, and from a
      vanishing denominator on, ``alpha`` is zero and the iterate stays put.

Performance notes:
    * Matvec is a block-per-row tiled reduction (``_matvec_tiled_kernel``)
      using ``wp.tile_load`` + ``wp.tile_sum``: ~4-5x faster than a
      single-thread-per-row serial inner loop at n ~ 600..1500.
    * Dot products use ``wp.utils.array_inner`` (CUB-backed parallel reduction)
      instead of a single-thread accumulator: ~6x faster at n ~ 900.
    * The per-iteration scalars stay on the device: the host reads the
      residual norm once per ``check_every`` iterations, which saves 5-10
      syncs per PCR call.
    * The per-iter PCR body (matvec + dots + axpy + scale-add + precond)
      issues ~10 kernel launches per iter; at a typical 25 iters per
      ``solve()`` call that's ~250 Python-side launch dispatches. Each Warp
      ``wp.launch`` plus the CUDA driver call add ~3-6 us of host overhead;
      CUDA graphs replay that batch with ~1 us per call. :meth:`NSNPCRSolver.solve`
      captures one PCR iter into a :class:`warp.Graph` per ``n`` shape and
      caches it by the matrix's pointer and strides. Subsequent solves at the
      same shape replay the graph ``check_every`` times per convergence block,
      a single ``cuGraphLaunch`` per replay on the device side.
    * The matvec stays in Warp: cuBLAS calls fail inside graph capture
      (``cudaErrorStreamCaptureImplicit``), and the captured Warp path ran a
      step in 43.4 ms against ~99 ms with cuBLAS run eagerly, on an RTX 5090
      in fp64.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

from .capture import scoped_capture

wp.set_module_options({"enable_backward": False})


def group_world_rows(worlds: np.ndarray, device) -> tuple:
    """Stable indexed segments of independent worlds, without reordering the system's rows."""
    _, world, counts = np.unique(worlds, return_inverse=True, return_counts=True)
    arrays = (world, np.r_[0, np.cumsum(counts)], np.argsort(world, kind="stable"))
    return (*(wp.array(a, dtype=wp.int32, device=device) for a in arrays), int(counts.max(initial=0)))


@wp.kernel
def _world_dot_kernel(
    offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    a: wp.array[wp.float64],
    b: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    world, lane = wp.tid()
    value = wp.float64(0.0)
    for slot in range(offsets[world] + lane, offsets[world + 1], wp.block_dim()):
        i = rows[slot]
        value += a[i] * b[i]
    total = wp.tile_sum(wp.tile(value))[0]
    if lane == 0:
        out[world] = total


@wp.kernel
def _serial_dot_kernel(a: wp.array[wp.float64], b: wp.array[wp.float64], n: int, out: wp.array[wp.float64]):
    """``out[0] = a[:n] . b[:n]``, added in row order by one thread: the order of :func:`_world_dot_kernel` on the
    CPU, whatever the order of the native reduction of ``wp.utils.array_inner`` on the machine."""
    total = wp.float64(0.0)
    for i in range(n):
        total += a[i] * b[i]
    out[0] = total


@wp.kernel
def _world_dense_mv_kernel(
    A: wp.array2d[wp.float64],
    x: wp.array[wp.float64],
    world: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    out: wp.array[wp.float64],
):
    i, lane = wp.tid()
    w = world[i]
    value = wp.float64(0.0)
    # Do not even read another world's vector: zero times NaN is still NaN.
    for slot in range(offsets[w] + lane, offsets[w + 1], wp.block_dim()):
        j = rows[slot]
        value += A[i, j] * x[j]
    total = wp.tile_sum(wp.tile(value))[0]
    if lane == 0:
        out[i] = total


@wp.kernel
def _world_alpha_kernel(
    rho: wp.array[wp.float64],
    den: wp.array[wp.float64],
    curvature: wp.array[wp.float64],
    chart: wp.array[wp.float64],
    r2: wp.array[wp.float64],
    b2: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    iteration: int,
    check_every: int,
    tol2: wp.float64,
    eta: wp.float64,
    alpha: wp.array[wp.float64],
    stopped: wp.array[wp.int32],
):
    w = wp.tid()
    if iteration >= offsets[w + 1] - offsets[w] or b2[w] == wp.float64(0.0):
        stopped[w] = 1
    if iteration % check_every == 0 and r2[w] <= tol2 * b2[w]:
        stopped[w] = 1
    if not (den[w] > wp.float64(0.0) and curvature[w] > eta * chart[w] and wp.isfinite(rho[w])):
        stopped[w] = 1
    alpha[w] = wp.float64(0.0)
    if stopped[w] == 0:
        alpha[w] = rho[w] / den[w]


@wp.kernel
def _world_update_kernel(
    world: wp.array[wp.int32],
    alpha: wp.array[wp.float64],
    p: wp.array[wp.float64],
    q: wp.array[wp.float64],
    t: wp.array[wp.float64],
    x: wp.array[wp.float64],
    r: wp.array[wp.float64],
    z: wp.array[wp.float64],
):
    i = wp.tid()
    a = alpha[world[i]]
    if a != wp.float64(0.0):
        x[i] += a * p[i]
        r[i] -= a * q[i]
        z[i] -= a * t[i]


@wp.kernel
def _world_direction_kernel(
    world: wp.array[wp.int32],
    rho: wp.array[wp.float64],
    old: wp.array[wp.float64],
    z: wp.array[wp.float64],
    t: wp.array[wp.float64],
    p: wp.array[wp.float64],
    q: wp.array[wp.float64],
):
    i = wp.tid()
    w = world[i]
    beta = wp.float64(0.0)
    if old[w] != wp.float64(0.0):
        beta = rho[w] / old[w]
    p[i] = z[i] + beta * p[i]
    q[i] = t[i] + beta * q[i]


# ---------------------------------------------------------------------------
# Tile / block sizing for ``_matvec_tiled_kernel``
# ---------------------------------------------------------------------------
# K-dim tile width: each block accumulates ``TILE_K`` columns per inner step
# and uses ``wp.tile_sum`` to reduce them in shared memory.  ``128`` is the
# sweet spot for NSN Schur sizes n ~ 600..1500 on an RTX 5090 in fp64.  Smaller
# tiles win at n > 1500 where occupancy beats per-block re-use; larger tiles
# win at n < 400 where launch overhead dominates.
_PCR_TILE_K = wp.constant(128)
_PCR_BLOCK_DIM = 128
# Block size of the fused sparse iteration of :meth:`NSNPCRSolver.solve_sparse_fused`, and the most blocks whose
# partial sums each block reduces in one tile: up to 262144 rows.
_FUSED_BLOCK = wp.constant(256)
_FUSED_MAX_BLOCKS = wp.constant(1024)
# Up to this many blocks, every thread of the fused iteration adds the blocks' partial sums one by one, in the same
# order, where a tile reduction of all _FUSED_MAX_BLOCKS slots would cost the block several times more.
_SERIAL_BLOCKS = wp.constant(32)
# Threads per row of the tiled sparse product that the fused iteration takes for long rows.
_TILED_MV = 64
# Mean entries per row from which the products by W take the tiled product. warp.sparse picks it from 128 on, but
# the rows of a rod's contacts, about 100 entries each, read several times faster together than one by one.
_LONG_ROW = 64


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_precond_kernel(
    A: wp.array2d[wp.float64],
    precond: wp.array[wp.float64],
):
    """Compute the Jacobi (diagonal-inverse) preconditioner ``precond[i] = 1/A[i,i]``.

    A zero diagonal entry is converted to ``0.0`` (rather than ``inf``) so a
    silently-degenerate row produces a zero update instead of NaNs.
    """
    i = wp.tid()
    d = A[i, i]
    if d == wp.float64(0.0):
        precond[i] = wp.float64(0.0)
    else:
        precond[i] = wp.float64(1.0) / d


@wp.kernel
def _precond_from_bsr_diag_kernel(
    diag: wp.array[wp.float64],
    precond: wp.array[wp.float64],
):
    """Build ``precond[i] = 1/diag[i]`` for the LiteNSN sparse Schur.

    ``warp.sparse`` stores 1x1-block BSR values as scalar fp64, so
    :func:`warp.sparse.bsr_get_diag` returns an ``array[wp.float64]``
    rather than ``array[mat1x1]``.  The zero-guard mirrors the dense
    :func:`_compute_precond_kernel`.
    """
    i = wp.tid()
    d = diag[i]
    if d == wp.float64(0.0):
        precond[i] = wp.float64(0.0)
    else:
        precond[i] = wp.float64(1.0) / d


_DOT_BLOCK = wp.constant(256)
"""Rows per block of the dot products of :meth:`NSNPCRSolver.solve` with ``fixed_sums``."""


@wp.kernel
def _dot_partials_kernel(
    a: wp.array[wp.float64],
    b: wp.array[wp.float64],
    n: int,
    # outputs
    partials: wp.array[wp.float64],
):
    """The sum of ``a[i] b[i]`` over each block of :data:`_DOT_BLOCK` rows of the first ``n``, one block of threads
    each, by a tile reduction."""
    block, lane = wp.tid()
    i = block * _DOT_BLOCK + lane
    v = wp.float64(0.0)
    if i < n:
        v = a[i] * b[i]
    total = wp.tile_sum(wp.tile(v))[0]
    if lane == 0:
        partials[block] = total


@wp.kernel
def _dot_finish_kernel(
    partials: wp.array[wp.float64],
    blocks: int,
    # outputs
    out: wp.array[wp.float64],
):
    """The sum of the blocks' partial sums, in block order."""
    total = wp.float64(0.0)
    for k in range(blocks):
        total += partials[k]
    out[0] = total


@wp.kernel
def _matvec_tiled_kernel(
    A: wp.array2d[wp.float64],
    x: wp.array[wp.float64],
    n_pad: int,
    out: wp.array[wp.float64],
):
    """Dense matvec ``out = A @ x`` with one CUDA block per output row.

    Each block sums one row of ``A`` against ``x`` cooperatively: threads load a
    ``TILE_K`` strip via ``wp.tile_load`` (bounds-checked, so a partial last
    tile loads zero-padded), element-wise multiply, then reduce with
    ``wp.tile_sum``.

    Launched via ``wp.launch_tiled`` with ``dim=n`` (one block per active
    row) and ``block_dim=_PCR_BLOCK_DIM``.

    Args:
        A: ``(n, n)`` dense SPD matrix slice (may be a non-contiguous slice
            view; the per-row 1D tile_load on ``A[row]`` handles arbitrary
            row strides).  Entries beyond ``[n, n]`` are never read because
            ``tile_load`` bounds-checks against the slice's ``shape``.
        x: ``(n,)`` input vector slice.  Same bounds-check applies.
        n_pad: ``ceil(n / TILE_K) * TILE_K``. The inner loop iterates up to
            ``n_pad`` so the partial last tile is reduced through the same
            ``wp.tile_sum`` path; the OOB elements load as zero.
        out: ``(n,)`` output buffer.  Only entries ``[0, n)`` are written.
    """
    row = wp.tid()
    acc_s = wp.float64(0.0)
    for k_tile in range(0, n_pad, _PCR_TILE_K):
        a = wp.tile_load(A[row], shape=(_PCR_TILE_K,), offset=(k_tile,))
        b = wp.tile_load(x, shape=(_PCR_TILE_K,), offset=(k_tile,))
        prod = wp.tile_map(wp.mul, a, b)
        s = wp.tile_sum(prod)
        # ``tile_extract`` broadcasts the (1,) reduction result to all
        # threads in the block; folding into a scalar accumulator keeps the
        # cross-iter state out of shared memory (re-assigning a shared tile
        # broke the register-vs-shared promotion of Warp 1.14).
        acc_s = acc_s + wp.tile_extract(s, 0)
    # All threads in the block write the same value to ``out[row]``; CUDA
    # coalesces identical writes (equivalent to a single thread emit).
    out[row] = acc_s


@wp.kernel
def _matvec_masked_kernel(
    A: wp.array2d[wp.float64],
    x: wp.array[wp.float64],
    mask: wp.array2d[wp.int32],
    active: wp.array[wp.int32],
    per: int,
    s: wp.array[wp.float64],
    out: wp.array[wp.float64],
    sh: wp.array[wp.float64],
):
    """``out = A @ x`` with one block of ``TILE_K`` threads per row, and with a nonempty ``sh`` each row's
    ``s[row] * out[row]``, on the leading ``per * active[0]`` rows and columns of ``A`` (all for an empty ``active``),
    the other rows inert, whose products it sets to 0, and on the ``TILE_K``-square tiles that ``mask`` marks (all for
    an empty ``mask``), the others holding only zeros. ``x`` is 0 on the inert rows. Each thread sums its lane of the
    row's tiles and the block adds the sums once, where :func:`_matvec_tiled_kernel` adds each tile's: several times
    faster for rows of several tiles, its sums in another order."""
    row, lane = wp.tid()
    n = A.shape[0]
    m = n
    if active.shape[0] > 0:
        m = wp.min(per * active[0], n)
    acc = wp.float64(0.0)
    if row < m:
        for k_tile in range(0, m, _PCR_TILE_K):
            skip = bool(False)
            if mask.shape[0] > 0:
                skip = mask[row // _PCR_TILE_K, k_tile // _PCR_TILE_K] == 0
            if not skip:
                j = k_tile + lane
                if j < m:
                    acc += A[row, j] * x[j]
    total = wp.tile_extract(wp.tile_sum(wp.tile(acc)), 0)
    if lane == 0:
        out[row] = total
        if sh.shape[0] > 0:
            sh[row] = s[row] * total


@wp.kernel
def _fused_pcr_turn_kernel(
    sh: wp.array[wp.float64],
    nb: int,
    rho_buf: wp.array[wp.float64],
    slot: int,
    first: int,
    n: int,
    s: wp.array[wp.float64],
    precond: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    d: wp.array[wp.float64],
    q: wp.array[wp.float64],
    h: wp.array[wp.float64],
    partial_den: wp.array[wp.float64],
    partial_curvature: wp.array[wp.float64],
    partial_chart: wp.array[wp.float64],
):
    """:func:`_fused_pcr_direction_kernel` of the previous iteration and :func:`_fused_pcr_dots_kernel` of this one in
    one launch, with their sums: ``rho``, the ``nb`` block sums of the rows' terms ``sh = s h`` added in block order,
    into ``rho_buf[slot]``, and, past the ``first`` iteration, ``beta = rho / rho_old``, ``d = beta d + s`` and
    ``q = beta q + h``; then ``h = P q`` and the block sums of ``q . h``, ``d . q`` and ``d . diag(A) d`` from each
    thread's row. Every block forms ``rho`` alike, for at most ``_SERIAL_BLOCKS`` blocks."""
    b, lane = wp.tid()
    rho_new = wp.float64(0.0)
    for k in range(nb):
        rho_new += wp.tile_extract(wp.tile_sum(wp.tile_load(sh, shape=(_FUSED_BLOCK,), offset=(k * _FUSED_BLOCK,))), 0)
    i = b * _FUSED_BLOCK + lane
    zero = wp.float64(0.0)
    beta = zero
    if first == 0:
        rho = rho_buf[1 - slot]
        if rho != zero:
            beta = rho_new / rho
    if b == 0 and lane == 0:
        rho_buf[slot] = rho_new
    q_i = zero
    d_i = zero
    g_i = zero
    h_i = zero
    if i < n:
        q_i = q[i]
        d_i = d[i]
        if first == 0:
            d_i = beta * d_i + s[i]
            q_i = beta * q_i + h[i]
            d[i] = d_i
            q[i] = q_i
        g_i = diag[i]
        h_i = precond[i] * q_i
        h[i] = h_i
    den = wp.tile_sum(wp.tile(q_i * h_i))
    curvature = wp.tile_sum(wp.tile(d_i * q_i))
    chart = wp.tile_sum(wp.tile(d_i * (g_i * d_i)))
    if lane == 0:
        partial_den[b] = wp.tile_extract(den, 0)
        partial_curvature[b] = wp.tile_extract(curvature, 0)
        partial_chart[b] = wp.tile_extract(chart, 0)


@wp.kernel
def _apply_precond_kernel(
    r: wp.array[wp.float64],
    precond: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    """Apply Jacobi preconditioner: ``out = P * r`` (elementwise)."""
    i = wp.tid()
    out[i] = precond[i] * r[i]


@wp.kernel
def _axpy_arr_kernel(
    a_arr: wp.array[wp.float64],
    sign: wp.float64,
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
):
    """Vector AXPY in place reading the coefficient from a device array slot.

    ``y[i] = y[i] + sign * a_arr[0] * x[i]``.

    Used inside the per-iter PCR loop so ``alpha`` (computed from device-side
    dot products) does not need to round-trip through host memory.  ``sign``
    folds in the ``+/-`` so one kernel serves ``x += alpha * p``,
    ``r -= alpha * q`` and ``z -= alpha * t``.
    """
    i = wp.tid()
    y[i] = y[i] + sign * a_arr[0] * x[i]


@wp.kernel
def _scal_add_arr_kernel(
    a_arr: wp.array[wp.float64],
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
):
    """Combined scale-and-add reading the coefficient from device memory.

    ``y = a_arr[0] * y + x``, which computes ``p = z + beta*p`` and
    ``q = t + beta*q``. One kernel instead of a scale and an add halves the
    launch overhead, and it reads ``beta`` from device memory so the inner PCR
    loop never blocks on a host transfer.
    """
    i = wp.tid()
    y[i] = a_arr[0] * y[i] + x[i]


@wp.kernel
def _div_scalar_kernel(
    num: wp.array[wp.float64],
    den: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    """Device-side scalar divide ``out[0] = num[0] / den[0]``.

    Computes the PCR ``beta = rho / rho_old`` on the device, so the host never
    reads it. A zero ``den[0]`` gives ``0.0``.
    """
    d = den[0]
    if d == wp.float64(0.0):
        out[0] = wp.float64(0.0)
    else:
        out[0] = num[0] / d


@wp.kernel
def _dense_diag_kernel(A: wp.array2d[wp.float64], diag: wp.array[wp.float64]):
    """``diag = diag(A)``."""
    i = wp.tid()
    diag[i] = A[i, i]


@wp.kernel
def _diag_scale_kernel(
    A: wp.array2d[wp.float64],
    v: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    """``out = diag(A) v``."""
    i = wp.tid()
    out[i] = A[i, i] * v[i]


@wp.kernel
def _protected_alpha_kernel(
    rho: wp.array[wp.float64],
    den: wp.array[wp.float64],
    curvature: wp.array[wp.float64],
    chart_norm: wp.array[wp.float64],
    eta: wp.float64,
    alpha: wp.array[wp.float64],
    stopped: wp.array[wp.float64],
):
    """Step length ``rho / den``, or zero from the first refused direction on.

    In the Jacobi chart ``C = sqrt(diag A)`` the direction ``p`` has curvature
    ``pᵀ A p / pᵀ diag(A) p``. Below ``eta`` a step along it amplifies the
    residual by more than ``1 / eta``: such near-null directions, a vanishing
    denominator, and non-finite values freeze the iterate instead.
    """
    if stopped[0] == wp.float64(0.0):
        if not (den[0] > wp.float64(0.0) and curvature[0] > eta * chart_norm[0]):
            stopped[0] = wp.float64(1.0)
    if stopped[0] != wp.float64(0.0):
        alpha[0] = wp.float64(0.0)
    else:
        alpha[0] = rho[0] / den[0]


@wp.kernel
def _copy_scalar_kernel(
    src: wp.array[wp.float64],
    dst: wp.array[wp.float64],
):
    """Copy ``dst[0] = src[0]`` on device (used to snapshot ``rho_old``)."""
    dst[0] = src[0]


@wp.kernel
def csr_mv_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
):
    """``y = W x`` for a scalar CSR ``W``, one thread per row, summed in column order."""
    i = wp.tid()
    acc = wp.float64(0.0)
    for k in range(offsets[i], offsets[i + 1]):
        acc += values[k] * x[columns[k]]
    y[i] = acc


# Lanes of :func:`block3_mv_kernel` per block row.
BLOCK3_LANES = 32


@wp.kernel
def block3_mv_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat33d],
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
):
    """``y = W x`` for a matrix of 3x3 blocks in block CSR form, launched tiled with a block of lanes per block row,
    :data:`BLOCK3_LANES` on CUDA: lane ``l`` takes the block row's blocks ``l``, ``l + block_dim`` and so on, and the
    lanes' sums add up in a fixed order."""
    i, lane = wp.tid()
    acc = wp.vec3d()
    for k in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
        j = 3 * columns[k]
        acc += values[k] * wp.vec3d(x[j], x[j + 1], x[j + 2])
    total = wp.tile_extract(wp.tile_sum(wp.tile(acc, preserve_type=True)), 0)
    if lane == 0:
        y[3 * i] = total[0]
        y[3 * i + 1] = total[1]
        y[3 * i + 2] = total[2]


@wp.kernel
def block3_diag_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat33d],
    diag: wp.array[wp.float64],
):
    """The diagonal of a matrix of 3x3 blocks with its block columns in order in each block row, zero where a block
    row stores no diagonal block."""
    i = wp.tid()
    lo = offsets[i]
    hi = offsets[i + 1]
    end = hi
    while lo < hi:
        mid = (lo + hi) // 2
        if columns[mid] < i:
            lo = mid + 1
        else:
            hi = mid
    block = wp.mat33d()
    if lo < end:
        if columns[lo] == i:
            block = values[lo]
    for a in range(3):
        diag[3 * i + a] = block[a, a]


def block3_mv(W, x: wp.array[wp.float64], y: wp.array[wp.float64]) -> None:
    """``y = W x`` for a :class:`~newton._src.solvers.fba.linear_solver.Block3Matrix`."""
    inputs = [W.offsets, W.columns, W.values, x]
    wp.launch_tiled(block3_mv_kernel, dim=W.nrow, inputs=inputs, outputs=[y], block_dim=BLOCK3_LANES, device=x.device)


def block3_diag(W, diag: wp.array[wp.float64]) -> None:
    """The diagonal of a :class:`~newton._src.solvers.fba.linear_solver.Block3Matrix`."""
    wp.launch(block3_diag_kernel, W.nrow, inputs=[W.offsets, W.columns, W.values], outputs=[diag], device=diag.device)


@wp.kernel
def csr_diag_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    diag: wp.array[wp.float64],
):
    """The diagonal of a scalar CSR matrix with its columns in order in each row, zero where a row stores none."""
    i = wp.tid()
    lo = offsets[i]
    hi = offsets[i + 1]
    end = hi
    while lo < hi:
        mid = (lo + hi) // 2
        if columns[mid] < i:
            lo = mid + 1
        else:
            hi = mid
    d = wp.float64(0.0)
    if lo < end:
        if columns[lo] == i:
            d = values[lo]
    diag[i] = d


@wp.kernel
def _fused_pcr_dots_kernel(
    q: wp.array[wp.float64],
    precond: wp.array[wp.float64],
    d: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    h: wp.array[wp.float64],
    partial_den: wp.array[wp.float64],
    partial_curvature: wp.array[wp.float64],
    partial_chart: wp.array[wp.float64],
):
    """``h = P q`` and the block sums of ``q . h``, ``d . q`` and ``d . diag(A) d``."""
    b = wp.tid()
    off = b * _FUSED_BLOCK
    q_t = wp.tile_load(q, shape=(_FUSED_BLOCK,), offset=(off,))
    p_t = wp.tile_load(precond, shape=(_FUSED_BLOCK,), offset=(off,))
    d_t = wp.tile_load(d, shape=(_FUSED_BLOCK,), offset=(off,))
    g_t = wp.tile_load(diag, shape=(_FUSED_BLOCK,), offset=(off,))
    h_t = wp.tile_map(wp.mul, p_t, q_t)
    wp.tile_store(h, h_t, offset=(off,))
    wp.tile_store(partial_den, wp.tile_sum(wp.tile_map(wp.mul, q_t, h_t)), offset=(b,))
    wp.tile_store(partial_curvature, wp.tile_sum(wp.tile_map(wp.mul, d_t, q_t)), offset=(b,))
    dd_t = wp.tile_map(wp.mul, g_t, d_t)
    wp.tile_store(partial_chart, wp.tile_sum(wp.tile_map(wp.mul, d_t, dd_t)), offset=(b,))


@wp.kernel
def _fused_pcr_dots_given_kernel(
    q: wp.array[wp.float64],
    h: wp.array[wp.float64],
    d: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    partial_den: wp.array[wp.float64],
    partial_curvature: wp.array[wp.float64],
    partial_chart: wp.array[wp.float64],
):
    """The block sums of :func:`_fused_pcr_dots_kernel` for an ``h = P q`` that a block preconditioner formed."""
    b = wp.tid()
    off = b * _FUSED_BLOCK
    q_t = wp.tile_load(q, shape=(_FUSED_BLOCK,), offset=(off,))
    h_t = wp.tile_load(h, shape=(_FUSED_BLOCK,), offset=(off,))
    d_t = wp.tile_load(d, shape=(_FUSED_BLOCK,), offset=(off,))
    g_t = wp.tile_load(diag, shape=(_FUSED_BLOCK,), offset=(off,))
    wp.tile_store(partial_den, wp.tile_sum(wp.tile_map(wp.mul, q_t, h_t)), offset=(b,))
    wp.tile_store(partial_curvature, wp.tile_sum(wp.tile_map(wp.mul, d_t, q_t)), offset=(b,))
    dd_t = wp.tile_map(wp.mul, g_t, d_t)
    wp.tile_store(partial_chart, wp.tile_sum(wp.tile_map(wp.mul, d_t, dd_t)), offset=(b,))


@wp.func
def _blocks_total(partial: wp.array[wp.float64]) -> wp.float64:
    """The sum of the fused iteration's per-block partial sums ``partial``, one per block, the same in every thread:
    added in order up to :data:`_SERIAL_BLOCKS` blocks, by a tile reduction past them."""
    total = wp.float64(0.0)
    if partial.shape[0] <= _SERIAL_BLOCKS:
        for k in range(partial.shape[0]):
            total += partial[k]
    else:
        total = wp.tile_extract(wp.tile_sum(wp.tile_load(partial, shape=(_FUSED_MAX_BLOCKS,))), 0)
    return total


@wp.kernel
def _fused_pcr_step_kernel(
    partial_den: wp.array[wp.float64],
    partial_curvature: wp.array[wp.float64],
    partial_chart: wp.array[wp.float64],
    rho_buf: wp.array[wp.float64],
    slot: int,
    eta: wp.float64,
    n: int,
    d: wp.array[wp.float64],
    q: wp.array[wp.float64],
    h: wp.array[wp.float64],
    stopped: wp.array[wp.float64],
    x: wp.array[wp.float64],
    r: wp.array[wp.float64],
    s: wp.array[wp.float64],
    scaled: int,
    scale: wp.array[wp.float64],
    sw: wp.array[wp.float64],
):
    """The protected step length from the summed blocks, which every block forms alike, and ``x += alpha d``,
    ``r -= alpha q``, ``s -= alpha h``; with ``scaled``, also ``sw = scale s`` for a factored matrix."""
    b, lane = wp.tid()
    den = _blocks_total(partial_den)
    curvature = _blocks_total(partial_curvature)
    chart = _blocks_total(partial_chart)
    stop = stopped[0] != wp.float64(0.0)
    if not stop:
        if not (den > wp.float64(0.0) and curvature > eta * chart):
            stop = True
    alpha = wp.float64(0.0)
    if not stop:
        alpha = rho_buf[slot] / den
    if stop and b == 0 and lane == 0:
        stopped[0] = wp.float64(1.0)
    i = b * _FUSED_BLOCK + lane
    if i < n:
        x[i] = x[i] + alpha * d[i]
        r[i] = r[i] - alpha * q[i]
        s_i = s[i] - alpha * h[i]
        s[i] = s_i
        if scaled != 0:
            sw[i] = scale[i] * s_i


@wp.kernel
def _fused_pcr_mv_scaled_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    sw: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    shift: wp.array[wp.float64],
    s: wp.array[wp.float64],
    n: int,
    h: wp.array[wp.float64],
    partial_rho: wp.array[wp.float64],
):
    """:func:`_fused_pcr_mv_kernel` for the factored ``A = diag(scale) W diag(scale) + diag(shift)``:
    ``h = scale (W sw) + shift s`` from ``sw = scale s``, and the block sums of ``s . h``."""
    b, lane = wp.tid()
    i = b * _FUSED_BLOCK + lane
    y = wp.float64(0.0)
    si = wp.float64(0.0)
    if i < n:
        for k in range(offsets[i], offsets[i + 1]):
            y += values[k] * sw[columns[k]]
        si = s[i]
        y = scale[i] * y + shift[i] * si
        h[i] = y
    wp.tile_store(partial_rho, wp.tile_sum(wp.tile(y * si)), offset=(b,))


@wp.kernel
def _fused_pcr_rho_scaled_kernel(
    s: wp.array[wp.float64],
    scale: wp.array[wp.float64],
    shift: wp.array[wp.float64],
    n: int,
    h: wp.array[wp.float64],
    partial_rho: wp.array[wp.float64],
):
    """:func:`_fused_pcr_rho_kernel` for the factored matrix: ``h = scale h + shift s`` on an ``h = W sw`` that
    :func:`warp.sparse.bsr_mv` multiplied, and the block sums of ``s . h``."""
    b, lane = wp.tid()
    i = b * _FUSED_BLOCK + lane
    y = wp.float64(0.0)
    si = wp.float64(0.0)
    if i < n:
        si = s[i]
        y = scale[i] * h[i] + shift[i] * si
        h[i] = y
    wp.tile_store(partial_rho, wp.tile_sum(wp.tile(y * si)), offset=(b,))


@wp.kernel
def _scale_shift_kernel(
    scale: wp.array[wp.float64],
    shift: wp.array[wp.float64],
    v: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    """``out = scale out + shift v``, which completes the product of the factored matrix by ``v``."""
    i = wp.tid()
    out[i] = scale[i] * out[i] + shift[i] * v[i]


@wp.kernel
def _scale_kernel(scale: wp.array[wp.float64], v: wp.array[wp.float64], out: wp.array[wp.float64]):
    """``out = scale v``."""
    i = wp.tid()
    out[i] = scale[i] * v[i]


@wp.kernel
def _fused_pcr_mv_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    s: wp.array[wp.float64],
    n: int,
    h: wp.array[wp.float64],
    partial_rho: wp.array[wp.float64],
):
    """``h = W s`` and the block sums of ``s . h``."""
    b, lane = wp.tid()
    i = b * _FUSED_BLOCK + lane
    y = wp.float64(0.0)
    si = wp.float64(0.0)
    if i < n:
        for k in range(offsets[i], offsets[i + 1]):
            y += values[k] * s[columns[k]]
        h[i] = y
        si = s[i]
    wp.tile_store(partial_rho, wp.tile_sum(wp.tile(y * si)), offset=(b,))


@wp.kernel
def _fused_pcr_rho_kernel(s: wp.array[wp.float64], h: wp.array[wp.float64], partial_rho: wp.array[wp.float64]):
    """The block sums of ``s . h``, for an ``h = W s`` that :func:`warp.sparse.bsr_mv` multiplied."""
    b = wp.tid()
    off = b * _FUSED_BLOCK
    s_t = wp.tile_load(s, shape=(_FUSED_BLOCK,), offset=(off,))
    h_t = wp.tile_load(h, shape=(_FUSED_BLOCK,), offset=(off,))
    wp.tile_store(partial_rho, wp.tile_sum(wp.tile_map(wp.mul, s_t, h_t)), offset=(b,))


@wp.kernel
def _fused_pcr_direction_kernel(
    partial_rho: wp.array[wp.float64],
    rho_buf: wp.array[wp.float64],
    slot: int,
    n: int,
    s: wp.array[wp.float64],
    h: wp.array[wp.float64],
    d: wp.array[wp.float64],
    q: wp.array[wp.float64],
):
    """``beta = rho_new / rho`` from the summed blocks, ``d = beta d + s`` and ``q = beta q + h``; the first thread
    keeps ``rho_new`` in the other slot."""
    b, lane = wp.tid()
    rho_new = _blocks_total(partial_rho)
    rho = rho_buf[slot]
    beta = wp.float64(0.0)
    if rho != wp.float64(0.0):
        beta = rho_new / rho
    i = b * _FUSED_BLOCK + lane
    if i < n:
        d[i] = beta * d[i] + s[i]
        q[i] = beta * q[i] + h[i]
    if b == 0 and lane == 0:
        rho_buf[1 - slot] = rho_new


# The most rows that one block's iteration takes (:meth:`NSNPCRSolver._small_single`), and the most articulation blocks
# of its block preconditioner: every block takes a dense solve at each iteration.
_SMALL_ROWS = 512
_SMALL_JOINT_BLOCKS = 8
# The most rows that one block's dense iteration takes (:meth:`NSNPCRSolver._small_dense`): its product by the dense
# matrix runs on one multiprocessor, which beyond some 128 rows falls behind the launches' product on all of them.
_SMALL_DENSE_ROWS = 128
_SMALL_KERNELS = {}


def _small_pcr_kernel(tile: int, blocked: bool = True):
    """The whole fixed-count iteration of :meth:`NSNPCRSolver._fused_single` for a factored matrix of 3x3 blocks, or of
    scalar entries without ``blocked``, in one block of threads, one row per thread, with the joint rows' block solves
    of ``tile``-row Cholesky factors as its block preconditioner: one launch for the whole solve instead of a dozen
    per iteration."""
    key = (tile, blocked)
    if key in _SMALL_KERNELS:
        return _SMALL_KERNELS[key]

    @wp.func
    def precondition(
        v: wp.float64,
        pre: wp.float64,
        L: wp.array3d[wp.float64],
        blocks: int,
        my_block: int,
        my_local: int,
    ) -> wp.float64:
        """The Jacobi product, and on the joint rows the solve of their block's ``W``."""
        out = pre * v
        for k in range(blocks):
            rhs = wp.tile_zeros(shape=(tile,), dtype=wp.float64, storage="shared")
            mine = my_block == k
            wp.tile_scatter_masked(rhs, my_local, v, mine)
            y = wp.tile_cholesky_solve(wp.tile_load(L[k], shape=(tile, tile)), rhs)
            value = wp.tile_extract(y, my_local)
            if mine:
                out = value
        return out

    @wp.func
    def total(v: wp.float64) -> wp.float64:
        """The sum of every thread's ``v``, in every thread."""
        return wp.tile_extract(wp.tile_sum(wp.tile(v)), 0)

    @wp.func
    def totals(v: wp.vec3d) -> wp.vec3d:
        """The sums of every thread's three values at once, in every thread."""
        return wp.tile_extract(wp.tile_sum(wp.tile(v, preserve_type=True)), 0)

    if blocked:

        @wp.func
        def product(
            offsets: wp.array[wp.int32],
            columns: wp.array[wp.int32],
            values: wp.array[wp.mat33d],
            sw: wp.float64,
            row: int,
            active: bool,
        ) -> wp.float64:
            """Row ``row`` of ``W sw``, ``sw`` one value per thread."""
            sw_t = wp.tile(sw)
            y = wp.float64(0.0)
            if active:
                block = row // 3
                a = row - 3 * block
                for k in range(offsets[block], offsets[block + 1]):
                    c = 3 * columns[k]
                    m = values[k]
                    y += m[a, 0] * wp.tile_extract(sw_t, c)
                    y += m[a, 1] * wp.tile_extract(sw_t, c + 1)
                    y += m[a, 2] * wp.tile_extract(sw_t, c + 2)
            return y

    else:

        @wp.func
        def product(
            offsets: wp.array[wp.int32],
            columns: wp.array[wp.int32],
            values: wp.array[wp.float64],
            sw: wp.float64,
            row: int,
            active: bool,
        ) -> wp.float64:
            """Row ``row`` of ``W sw`` for scalar entries, summed in column order as :func:`csr_mv_kernel` sums."""
            sw_t = wp.tile(sw)
            y = wp.float64(0.0)
            if active:
                for k in range(offsets[row], offsets[row + 1]):
                    y += values[k] * wp.tile_extract(sw_t, columns[k])
            return y

    # The entries' type, 3x3 blocks or scalars, is that of ``product``: the kernel takes it from ``values``.
    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        offsets: wp.array[wp.int32],
        columns: wp.array[wp.int32],
        values: wp.array[Any],
        scale: wp.array[wp.float64],
        shift: wp.array[wp.float64],
        diag: wp.array[wp.float64],
        b: wp.array[wp.float64],
        n: int,
        max_iter: int,
        eta: wp.float64,
        L: wp.array3d[wp.float64],
        blocks: int,
        row_block: wp.array[wp.int32],
        row_local: wp.array[wp.int32],
        # outputs
        x_out: wp.array[wp.float64],
    ):
        _, i = wp.tid()
        active = i < n
        zero = wp.float64(0.0)
        dg = zero
        sc = zero
        sh = zero
        bi = zero
        my_block = int(-1)
        my_local = int(0)
        if active:
            dg = diag[i]
            sc = scale[i]
            sh = shift[i]
            bi = b[i]
            my_block = row_block[i]
            my_local = wp.max(row_local[i], 0)
        pre = zero
        if dg != zero:
            pre = wp.float64(1.0) / dg
        x = zero
        r = bi
        s = precondition(r, pre, L, blocks, my_block, my_local)
        d = s
        q = sc * product(offsets, columns, values, sc * d, i, active) + sh * d
        rho = total(s * q)
        stopped = bool(False)
        for _ in range(max_iter):
            h = precondition(q, pre, L, blocks, my_block, my_local)
            sums = totals(wp.vec3d(q * h, d * q, d * dg * d))
            den = sums[0]
            curvature = sums[1]
            chart = sums[2]
            if not stopped:
                if not (den > zero and curvature > eta * chart):
                    stopped = True
            alpha = zero
            if not stopped:
                alpha = rho / den
            x = x + alpha * d
            r = r - alpha * q
            s = s - alpha * h
            h = sc * product(offsets, columns, values, sc * s, i, active) + sh * s
            rho_new = total(s * h)
            beta = zero
            if rho != zero:
                beta = rho_new / rho
            d = beta * d + s
            q = beta * q + h
            rho = rho_new
        if active:
            x_out[i] = x

    _SMALL_KERNELS[key] = kernel
    return kernel


@wp.func
def _block_total(v: wp.float64) -> wp.float64:
    """The sum of every thread's ``v`` in a block of threads, in every thread."""
    return wp.tile_extract(wp.tile_sum(wp.tile(v)), 0)


@wp.func
def _block_totals(v: wp.vec3d) -> wp.vec3d:
    """The sums of every thread's three values at once in a block of threads, in every thread."""
    return wp.tile_extract(wp.tile_sum(wp.tile(v, preserve_type=True)), 0)


@wp.func
def _dense_row_product(A: wp.array2d[wp.float64], v: wp.float64, row: int, m: int) -> wp.float64:
    """Row ``row`` of ``A v`` over the leading ``m`` rows and columns, ``v`` one value per thread, summed in column
    order; 0 past them."""
    v_t = wp.tile(v)
    y = wp.float64(0.0)
    if row < m:
        for j in range(m):
            y += A[row, j] * wp.tile_extract(v_t, j)
    return y


@wp.kernel(enable_backward=False)
def _small_dense_pcr_kernel(
    A: wp.array2d[wp.float64],
    b: wp.array[wp.float64],
    n: int,
    active: wp.array[wp.int32],
    rows_per: int,
    max_iter: int,
    eta: wp.float64,
    block: int,
    # outputs
    x_out: wp.array[wp.float64],
):
    """The whole fixed-count iteration of :meth:`NSNPCRSolver._fused_dense` in one block of ``block`` threads, one
    row per thread, on the leading ``rows_per * active[0]`` rows (all ``n`` for an empty ``active``), at most
    ``block``, whose solution the other rows, inert rows of the identity and a zero right-hand side, leave unchanged:
    they keep ``x = 0``."""
    _, i = wp.tid()
    m = n
    if active.shape[0] > 0:
        m = wp.min(rows_per * active[0], n)
    zero = wp.float64(0.0)
    dg = zero
    bi = zero
    if i < m:
        dg = A[i, i]
        bi = b[i]
    pre = zero
    if dg != zero:
        pre = wp.float64(1.0) / dg
    x = zero
    r = bi
    s = pre * r
    d = s
    q = _dense_row_product(A, d, i, m)
    rho = _block_total(s * q)
    stopped = bool(False)
    for _ in range(max_iter):
        h = pre * q
        sums = _block_totals(wp.vec3d(q * h, d * q, d * dg * d))
        den = sums[0]
        curvature = sums[1]
        chart = sums[2]
        if not stopped:
            if not (den > zero and curvature > eta * chart):
                stopped = True
        alpha = zero
        if not stopped:
            alpha = rho / den
        x = x + alpha * d
        r = r - alpha * q
        s = s - alpha * h
        h = _dense_row_product(A, s, i, m)
        rho_new = _block_total(s * h)
        beta = zero
        if rho != zero:
            beta = rho_new / rho
        d = beta * d + s
        q = beta * q + h
        rho = rho_new
    if i < n:
        x_out[i] = x
    k = i + block
    while k < n:
        x_out[k] = zero
        k += block


@wp.kernel
def _small_dense_switch_kernel(
    active: wp.array[wp.int32],
    rows_per: int,
    n: int,
    most: int,
    # outputs
    flag: wp.array[wp.int32],
):
    """1 when the leading ``rows_per * active[0]`` rows of a system of ``n`` are at most ``most``, else 0."""
    flag[0] = wp.where(wp.min(rows_per * active[0], n) <= most, 1, 0)


@wp.kernel
def joint_row_places_kernel(
    rows: wp.array[wp.int32],
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    offset: int,
    # outputs
    place_block: wp.array[wp.int32],
    place_local: wp.array[wp.int32],
):
    """The articulation block and the row within it of each joint row ``rows[k]`` at row ``offset + k`` of a solve."""
    k = wp.tid()
    row = rows[k]
    if row >= 0:
        place_block[offset + k] = row_block[row]
        place_local[offset + k] = row_local[row]


# The fused iteration of independent worlds runs a grid of chunks of _FUSED_BLOCK rows: chunk c of world w is block
# w * chunks + c and takes the world's rows offsets[w] + c * _FUSED_BLOCK + lane, through the stable row index. Each
# block sums its world's chunk partials alike, so the worlds keep their own scalars and every block of a world agrees.


@wp.func
def _world_row(offsets: wp.array[wp.int32], rows: wp.array[wp.int32], w: int, c: int, k: int) -> int:
    """The system row of row ``k`` of chunk ``c`` of world ``w``, or -1 past the world's rows. A lane takes rows
    ``lane``, ``lane + block_dim`` and so on of its chunk: one on CUDA, all of them off it, whose tiled launches run a
    lane per block."""
    slot = offsets[w] + c * _FUSED_BLOCK + k
    if slot < offsets[w + 1]:
        return rows[slot]
    return -1


@wp.func
def _world_partial(partial: wp.array[wp.float64], w: int, chunks: int, lane: int) -> wp.float64:
    """A lane's share of the sum of world ``w``'s chunk partials."""
    value = wp.float64(0.0)
    for c in range(lane, chunks, wp.block_dim()):
        value += partial[w * chunks + c]
    return value


@wp.kernel
def _world_fused_start_kernel(
    offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    chunks: int,
    s: wp.array[wp.float64],
    q: wp.array[wp.float64],
    b: wp.array[wp.float64],
    partial_rho: wp.array[wp.float64],
    partial_b2: wp.array[wp.float64],
):
    """The chunk sums of ``s . q`` and ``b . b``."""
    block, lane = wp.tid()
    w = block // chunks
    sq = wp.float64(0.0)
    bb = wp.float64(0.0)
    for k in range(lane, _FUSED_BLOCK, wp.block_dim()):
        i = _world_row(offsets, rows, w, block - w * chunks, k)
        if i >= 0:
            sq += s[i] * q[i]
            bb += b[i] * b[i]
    wp.tile_store(partial_rho, wp.tile_sum(wp.tile(sq)), offset=(block,))
    wp.tile_store(partial_b2, wp.tile_sum(wp.tile(bb)), offset=(block,))


@wp.kernel
def _world_fused_sum_kernel(partial: wp.array[wp.float64], chunks: int, out: wp.array[wp.float64]):
    """Each world's sum of its chunk partials."""
    w, lane = wp.tid()
    total = wp.tile_sum(wp.tile(_world_partial(partial, w, chunks, lane)))[0]
    if lane == 0:
        out[w] = total


@wp.kernel
def _world_fused_dots_kernel(
    offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    chunks: int,
    q: wp.array[wp.float64],
    precond: wp.array[wp.float64],
    d: wp.array[wp.float64],
    diag: wp.array[wp.float64],
    h: wp.array[wp.float64],
    partial_den: wp.array[wp.float64],
    partial_curvature: wp.array[wp.float64],
    partial_chart: wp.array[wp.float64],
):
    """:func:`_fused_pcr_dots_kernel` over a world's chunk."""
    block, lane = wp.tid()
    w = block // chunks
    qh = wp.float64(0.0)
    dq = wp.float64(0.0)
    dd = wp.float64(0.0)
    for k in range(lane, _FUSED_BLOCK, wp.block_dim()):
        i = _world_row(offsets, rows, w, block - w * chunks, k)
        if i >= 0:
            hi = precond[i] * q[i]
            h[i] = hi
            qh += q[i] * hi
            dq += d[i] * q[i]
            dd += d[i] * (diag[i] * d[i])
    wp.tile_store(partial_den, wp.tile_sum(wp.tile(qh)), offset=(block,))
    wp.tile_store(partial_curvature, wp.tile_sum(wp.tile(dq)), offset=(block,))
    wp.tile_store(partial_chart, wp.tile_sum(wp.tile(dd)), offset=(block,))


@wp.kernel
def _world_fused_step_kernel(
    offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    chunks: int,
    partial_den: wp.array[wp.float64],
    partial_curvature: wp.array[wp.float64],
    partial_chart: wp.array[wp.float64],
    rho_buf: wp.array2d[wp.float64],
    b2: wp.array[wp.float64],
    slot: int,
    iteration: int,
    tol2: wp.float64,
    eta: wp.float64,
    d: wp.array[wp.float64],
    q: wp.array[wp.float64],
    h: wp.array[wp.float64],
    stopped: wp.array[wp.int32],
    x: wp.array[wp.float64],
    r: wp.array[wp.float64],
    s: wp.array[wp.float64],
):
    """Each world's protected step as :func:`_world_alpha_kernel` takes it, and its rows' update as
    :func:`_world_update_kernel` makes it."""
    block, lane = wp.tid()
    w = block // chunks
    c = block - w * chunks
    den = wp.tile_sum(wp.tile(_world_partial(partial_den, w, chunks, lane)))[0]
    curvature = wp.tile_sum(wp.tile(_world_partial(partial_curvature, w, chunks, lane)))[0]
    chart = wp.tile_sum(wp.tile(_world_partial(partial_chart, w, chunks, lane)))[0]
    rho = rho_buf[slot, w]
    stop = stopped[w] != 0
    if iteration >= offsets[w + 1] - offsets[w] or b2[w] == wp.float64(0.0):
        stop = True
    # the residual test of the first iteration, whose residual is b: the fused runs end before the next one
    if iteration == 0 and b2[w] <= tol2 * b2[w]:
        stop = True
    if not (den > wp.float64(0.0) and curvature > eta * chart and wp.isfinite(rho)):
        stop = True
    alpha = wp.float64(0.0)
    if not stop:
        alpha = rho / den
    if stop and c == 0 and lane == 0:
        stopped[w] = 1
    if alpha != wp.float64(0.0):
        for k in range(lane, _FUSED_BLOCK, wp.block_dim()):
            i = _world_row(offsets, rows, w, c, k)
            if i >= 0:
                x[i] += alpha * d[i]
                r[i] -= alpha * q[i]
                s[i] -= alpha * h[i]


@wp.kernel
def _world_fused_mv_kernel(
    offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    chunks: int,
    matrix_offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    s: wp.array[wp.float64],
    h: wp.array[wp.float64],
    partial_rho: wp.array[wp.float64],
):
    """:func:`_fused_pcr_mv_kernel` over a world's chunk."""
    block, lane = wp.tid()
    w = block // chunks
    sh = wp.float64(0.0)
    for k in range(lane, _FUSED_BLOCK, wp.block_dim()):
        i = _world_row(offsets, rows, w, block - w * chunks, k)
        if i >= 0:
            y = wp.float64(0.0)
            for e in range(matrix_offsets[i], matrix_offsets[i + 1]):
                y += values[e] * s[columns[e]]
            h[i] = y
            sh += s[i] * y
    wp.tile_store(partial_rho, wp.tile_sum(wp.tile(sh)), offset=(block,))


@wp.kernel
def _world_fused_direction_kernel(
    offsets: wp.array[wp.int32],
    rows: wp.array[wp.int32],
    chunks: int,
    partial_rho: wp.array[wp.float64],
    rho_buf: wp.array2d[wp.float64],
    slot: int,
    s: wp.array[wp.float64],
    h: wp.array[wp.float64],
    d: wp.array[wp.float64],
    q: wp.array[wp.float64],
):
    """Each world's ``beta`` and its rows' directions as :func:`_world_direction_kernel` forms them; the world's
    first chunk keeps ``rho_new`` in the other slot."""
    block, lane = wp.tid()
    w = block // chunks
    c = block - w * chunks
    rho_new = wp.tile_sum(wp.tile(_world_partial(partial_rho, w, chunks, lane)))[0]
    rho = rho_buf[slot, w]
    beta = wp.float64(0.0)
    if rho != wp.float64(0.0):
        beta = rho_new / rho
    for k in range(lane, _FUSED_BLOCK, wp.block_dim()):
        i = _world_row(offsets, rows, w, c, k)
        if i >= 0:
            d[i] = s[i] + beta * d[i]
            q[i] = h[i] + beta * q[i]
    if c == 0 and lane == 0:
        rho_buf[1 - slot, w] = rho_new


def csr_mv(W, x: wp.array, y: wp.array) -> None:
    """``y = W x`` for a scalar sparse matrix of ``warp.sparse``, read as CSR, through :func:`csr_mv_kernel`."""
    inputs = [W.offsets, W.columns, W.values, x]
    wp.launch(csr_mv_kernel, dim=int(W.nrow), inputs=inputs, outputs=[y], device=x.device)


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class NSNPCRSolver:
    """Jacobi-preconditioned Conjugate Residual solver for SPD systems.

    :meth:`solve` runs conjugate residuals on the symmetrically preconditioned
    ``P^1/2 A P^1/2``, with the step denominator ``(A p)ᵀ P (A p)``, which stays
    a Krylov method when the diagonal of ``A`` spans decades. It refuses
    near-null directions, so a singular ``A`` with an inconsistent right-hand
    side returns a bounded least-squares iterate. :meth:`solve_sparse` runs the
    same iteration on a sparse matrix, and :meth:`solve_sparse_fused` runs a
    fixed count of its iterations in four launches each. Used as the inner
    linear solver for the NSN Schur LCP system.

    The solver is *stateless* between ``solve`` calls: every call starts from
    ``x = 0``. Device buffers are pre-allocated for ``max_n`` so repeated calls
    at smaller sizes do not allocate.

    Args:
        max_n: Maximum system dimension the solver will handle. Internal
            buffers are sized to this.
        device: Warp device for buffer allocation (typically ``"cuda:0"``).
        tol: Relative tolerance: :meth:`solve` and :meth:`solve_sparse` stop
            once ``|b - A x| < tol |b|``.
        max_iter: Outer-loop cap. Effective limit is ``min(max_iter, n)``.
        check_every: Convergence-check batching factor.  PCR runs this many
            iterations on the device between two host reads of the residual
            norm and the stop flag.
        max_inverse_gain: Largest amplification a step of :meth:`solve` may
            carry; directions with chart curvature below its inverse are
            refused.
    """

    # Whether the fused iteration of a small factored system of 3x3 blocks runs in one block of threads
    # (:meth:`_small_single`). Its sums add up in another order than the iteration of several launches, whose sums
    # stay alike for any padding of the rows: comparisons bit for bit between paths that pad the same rows differently
    # turn it off.
    small_single = True

    def __init__(
        self,
        max_n: int,
        device: wp.Device | str,
        tol: float = 1.0e-5,
        max_iter: int = 100,
        check_every: int = 25,
        max_inverse_gain: float = 1.0e6,
    ) -> None:
        if max_n <= 0:
            raise ValueError(f"max_n must be positive, got {max_n}")

        self.max_n = int(max_n)
        self.device = device
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        # Rather than syncing every iteration, which stalls the host after every PCR step, test convergence
        # every ``check_every`` iterations. Up to ``check_every - 1`` iterations past the tolerance cost time
        # but not accuracy: each one lowers the preconditioned residual norm.
        self.check_every = max(1, int(check_every))
        self._eta = wp.float64(1.0 / max_inverse_gain)

        # Pre-allocate fp64 device buffers: ``_d`` holds the direction p,
        # ``_q`` A p, ``_s`` the preconditioned residual z and ``_h`` the
        # scratch t.
        self._x = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._r = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._d = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._q = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._h = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._s = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        # ``scale s`` of :meth:`solve_sparse_scaled`'s factored matrix.
        self._sw = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._precond = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        # Diagonal of the sparse matrix of :meth:`solve_sparse`.
        self._diag = wp.zeros(self.max_n, dtype=wp.float64, device=device)

        # Device-resident scalars for ``rho``, ``rho_old``, ``den``, ``alpha``,
        # ``beta``: keep the per-iter PCR coefficients on device so the inner
        # loop never blocks on a host transfer.  ``rho`` is synced exactly
        # once per ``solve()`` call at the end.
        self._rho_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._rho_old_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._den_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._alpha_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._beta_d = wp.zeros(1, dtype=wp.float64, device=device)
        # :meth:`solve` protection: chart curvature of each direction.
        self._dd = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._curvature_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._chart_norm_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._stopped_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._r_norm_d = wp.zeros(1, dtype=wp.float64, device=device)
        # Block sums of the fused sparse iteration of :meth:`solve_sparse_fused` and its double-buffered ``rho``.
        # The fused worlds take one block per chunk of each world, up to ``max_n`` blocks, so that a grid padded
        # to the longest world fits whenever it is no larger than the system's own bound; a single system keeps its
        # limit of 1024 blocks. Allocated once, so captured graphs keep their pointers.
        capacity = max(int(_FUSED_MAX_BLOCKS), self.max_n)
        self._fused_partials = wp.zeros((4, capacity), dtype=wp.float64, device=device)
        self._fused_rho = wp.zeros(2, dtype=wp.float64, device=device)
        # The single-block iteration of small factored systems (_small_single): each row's articulation block and row
        # in it, -1 off the joint rows, and a factor for runs without a block preconditioner.
        self._small_places = (
            wp.full(self.max_n, -1, dtype=wp.int32, device=device),
            wp.zeros(self.max_n, dtype=wp.int32, device=device),
        )
        self._small_no_factor = wp.zeros((1, 8, 8), dtype=wp.float64, device=device)
        # Whether a dense system's active rows take the single block (_fused_dense), read by a conditional node.
        self._small_dense_flag = wp.zeros(1, dtype=wp.int32, device=device)
        # The factors of an unscaled matrix, ``scale = 1`` and ``shift = 0``.
        rows = min(self.max_n, _SMALL_ROWS)
        self._small_unit = (
            wp.ones(rows, dtype=wp.float64, device=device),
            wp.zeros(rows, dtype=wp.float64, device=device),
        )
        # The fused iteration of independent worlds: each world's double-buffered ``rho``, ``b . b`` and stop flag.
        self._world_fused_rho = wp.zeros((2, capacity), dtype=wp.float64, device=device)
        self._world_fused_b2 = wp.zeros(capacity, dtype=wp.float64, device=device)
        self._world_fused_stopped = wp.zeros(capacity, dtype=wp.int32, device=device)

        # Cache one ``wp.Graph`` per active ``n`` shape so repeated
        # solves at the same shape replay a pre-recorded CUDA graph for the
        # inner PCR iter, eliminating per-launch Python+driver overhead.
        # Capture is attempted lazily on first solve at each shape; if it
        # fails we fall back to the eager loop and remember the failure so
        # we don't retry on every call.
        self._pcr_graph_cache: dict[int, wp.Graph | None] = {}
        self._block_precond = None
        # The dot products of solves with ``fixed_sums``: per-block partial sums, so that they allocate nothing.
        self._fixed_sums = False
        self._dot_partials = wp.zeros((self.max_n + int(_DOT_BLOCK) - 1) // int(_DOT_BLOCK), dtype=wp.float64,
                                      device=device)  # fmt: skip
        self.world_rows = None
        self._world_scalars = None

    def _solve_worlds(self, A, b, x_out, block_precond, *, sparse: bool) -> int:
        """Run the same PCR recurrence with independent reductions, budgets and refusal flags per world."""
        import warp.sparse as wps  # noqa: PLC0415

        world, offsets, rows, max_rows = self.world_rows
        n, nw, device = b.shape[0], offsets.shape[0] - 1, self.device
        if world.shape[0] != n or n > self.max_n or x_out.shape[0] != n or A.shape != (n, n):
            raise ValueError("world rows, matrix and vectors must describe the same system")
        if self._world_scalars is None:
            self._world_scalars = [wp.empty(self.max_n, dtype=wp.float64, device=device) for _ in range(8)]
            self._world_stopped = wp.zeros(self.max_n, dtype=wp.int32, device=device)
        rho, old, den, curvature, chart, alpha, r2, b2 = (a[:nw] for a in self._world_scalars)
        x, r, z, p, q, t, diag, precond, dd = (
            a[:n] for a in (self._x, self._r, self._s, self._d, self._q, self._h, self._diag, self._precond, self._dd)
        )

        def dot(a, b, out):
            wp.launch_tiled(
                _world_dot_kernel, dim=nw, inputs=[offsets, rows, a, b], outputs=[out], block_dim=128, device=device
            )

        def mv(a, out):
            if sparse:
                wps.bsr_mv(A, a, out)
            else:
                wp.launch_tiled(
                    _world_dense_mv_kernel,
                    dim=n,
                    inputs=[A, a, world, offsets, rows],
                    outputs=[out],
                    block_dim=128,
                    device=device,
                )

        def apply(a, out):
            wp.launch(_apply_precond_kernel, n, [a, precond], [out], device=device)
            if block_precond is not None:
                block_precond(a, out)

        x.zero_()
        self._world_stopped.zero_()
        if sparse:
            wps.bsr_get_diag(A, out=diag)
            wp.launch(_precond_from_bsr_diag_kernel, n, [diag], [precond], device=device)
        else:
            wp.launch(_compute_precond_kernel, n, [A], [precond], device=device)
        wp.copy(r, b)
        apply(r, z)
        wp.copy(p, z)
        mv(p, q)
        dot(z, q, rho)
        dot(b, b, b2)
        wp.copy(r2, b2)
        iterations = min(self.max_iter, max_rows)
        for k in range(iterations):
            apply(q, t)
            dot(q, t, den)
            if sparse:
                wp.launch(_apply_precond_kernel, n, [p, diag], [dd], device=device)
            else:
                wp.launch(_diag_scale_kernel, n, [A, p], [dd], device=device)
            dot(p, q, curvature)
            dot(p, dd, chart)
            if k and k % self.check_every == 0:
                dot(r, r, r2)
            inputs = [rho, den, curvature, chart, r2, b2, offsets, k, self.check_every]
            inputs += [wp.float64(self.tol**2), self._eta]
            wp.launch(_world_alpha_kernel, nw, inputs, [alpha, self._world_stopped], device=device)
            if k and k % self.check_every == 0 and not wp.get_device(device).is_capturing:
                if self._world_stopped[:nw].numpy().all():
                    iterations = k
                    break
            wp.launch(_world_update_kernel, n, [world, alpha, p, q, t], [x, r, z], device=device)
            mv(z, t)
            wp.copy(old, rho)
            dot(z, t, rho)
            wp.launch(_world_direction_kernel, n, [world, rho, old, z, t], [p, q], device=device)
        wp.copy(x_out, x)
        return iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        A: wp.array2d[wp.float64],
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        block_precond: Callable[[wp.array, wp.array], None] | None = None,
        fixed_sums: bool = False,
        active_rows: tuple[wp.array, int] | None = None,
        mask: wp.array2d[wp.int32] | None = None,
    ) -> tuple[int, float | None]:
        """Solve ``A @ x = b`` with Jacobi-preconditioned conjugate residual.

        Args:
            A: ``(n, n)`` dense SPD matrix, device-resident, fp64. ``n`` must
                satisfy ``n <= max_n``.
            b: ``(n,)`` right-hand side, device-resident, fp64.
            x_out: ``(n,)`` output buffer, device-resident, fp64. Filled with
                the solution on return.
            block_precond: Optional ``f(v, out)`` that overwrites rows of the
                Jacobi product ``out = P v`` with an exact block solve, such
                as the joint rows' Cholesky factor, a block preconditioner of
                the bilateral rows. The joint rows' solve
                (:class:`~newton._src.solvers.fba.joints.JointRowsPrecond`)
                records into the iteration's CUDA graph with the other
                launches; another callable runs the iterations eagerly.
            fixed_sums: For a solve of at most ``check_every`` iterations, take the dot products as per-block tile sums
                added in block order, which allocate nothing, as a CUDA graph's conditional node needs, in place of
                ``wp.utils.array_inner``: the sums differ by rounding.
            active_rows: With ``fixed_sums``, ``(count, per)``: the leading ``per * count[0]`` rows, a count on the
                device, hold the system, and the others are rows of the identity with a zero right-hand side, as the
                inert rows of a CUDA graph's contact buffer. When they are at most 128, the iterations take them
                alone in one block of threads (:meth:`_small_dense`), chosen by a conditional node on the count;
                otherwise the products by ``A`` take them alone.
            mask: With ``fixed_sums``, the 128-square tiles of ``A`` that hold nonzero entries, which its products
                take alone, the others holding only zeros.

        Returns:
            A ``(iterations_taken, final_rho)`` tuple where ``iterations_taken``
            counts the number of inner PCR iterations executed and
            ``final_rho`` is the last value of the preconditioned residual
            scalar ``rho = <z, A z>`` with ``z = P r`` (>= 0 for an SPD ``A``).
            A solve of at most ``check_every`` iterations runs them all without
            reading anything back to the host, and returns ``None`` for
            ``final_rho``.

        Raises:
            ValueError: if ``A``/``b``/``x_out`` shapes are inconsistent or
                exceed ``max_n``.
        """
        n = int(b.shape[0])
        if n <= 0:
            raise ValueError(f"system size must be positive, got n={n}")
        if n > self.max_n:
            raise ValueError(f"system size n={n} exceeds preallocated max_n={self.max_n}")
        if A.shape[0] != n or A.shape[1] != n:
            raise ValueError(f"A shape {tuple(A.shape)} incompatible with b shape ({n},)")
        if x_out.shape[0] != n:
            raise ValueError(f"x_out shape ({x_out.shape[0]},) incompatible with b shape ({n},)")

        if self.world_rows is not None:
            iterations = self._solve_worlds(A, b, x_out, block_precond, sparse=False)
            if min(self.max_iter, n) <= self.check_every:
                return iterations, None
            return iterations, float(self._world_scalars[0][: self.world_rows[1].shape[0] - 1].numpy().sum())

        device = self.device
        max_iter = min(self.max_iter, n)
        # A capped solve of at most one check block runs its iterations without reading back: the iterations past
        # convergence, and those of a zero right-hand side, leave the iterate unchanged.
        unchecked = max_iter <= self.check_every
        self._fixed_sums = fixed_sums and unchecked
        if self._fixed_sums and self.fuses_single(n):
            return self._fused_dense(A, b, x_out, block_precond, active_rows, mask), None
        tile_k = int(_PCR_TILE_K)
        n_pad = ((n + tile_k - 1) // tile_k) * tile_k

        # ----- Initialize x = 0 -----------------------------------------
        self._x.zero_()

        # ----- Early exit if b == 0 -------------------------------------
        # With b == 0, x = 0 is the exact solution.  ``array_inner`` is a
        # CUB-style parallel reduction, far cheaper than a single-thread
        # accumulator at the typical n ~ 600..1500 sizes.
        dot_b = 0.0
        if not unchecked:
            self._dot(b, b, self._r_norm_d, n)
            dot_b = float(self._r_norm_d.numpy()[0])
            if dot_b == 0.0:
                wp.copy(x_out, self._x, count=n)
                return (0, 0.0)

        # ----- Jacobi preconditioner: precond[i] = 1/A[i,i] -------------
        wp.launch(
            _compute_precond_kernel,
            dim=n,
            inputs=[A],
            outputs=[self._precond],
            device=device,
        )

        # ----- Symmetric PCR start ---------------------------------------
        # Buffers: ``_s`` holds z = P r, ``_d`` the direction p = z, ``_q``
        # holds A p, and ``_h`` the scratch t (P q, then A z).
        wp.copy(self._r, b, count=n)
        wp.launch(_apply_precond_kernel, dim=n, inputs=[self._r, self._precond], outputs=[self._s], device=device)
        self._block_precond = block_precond
        if block_precond is not None:
            block_precond(self._r, self._s)
        wp.copy(self._d, self._s, count=n)
        wp.launch_tiled(
            _matvec_tiled_kernel,
            dim=n,
            inputs=[A, self._d, n_pad],
            outputs=[self._q],
            block_dim=_PCR_BLOCK_DIM,
            device=device,
        )
        self._inner(self._s, self._q, self._rho_d, n)
        self._stopped_d.zero_()
        tol_sq = self.tol * self.tol * dot_b
        r_norm_sq = dot_b

        # ---- PCR main loop --------------------------------------------
        # Inner loop runs in ``check_every``-sized blocks fully on the device,
        # then syncs the residual norm once per block to test convergence.  All
        # per-iter scalars (``alpha``, ``beta``, ``rho``, ``den``) live in
        # device memory.  Each PCR iteration lowers the preconditioned residual
        # norm, so iterating past the tolerance between syncs is safe.
        # Iterations past convergence or past a refused direction leave the
        # iterate unchanged (see :func:`_protected_alpha_kernel`).
        #
        # Wrap one full PCR iter into a :class:`warp.Graph` and replay it
        # ``check_every`` times per block.  Replays issue a single
        # ``cuGraphLaunch`` on the device side instead of ~10 individual
        # ``wp.launch`` dispatches, so the Python+driver per-launch overhead
        # drops from ~30-50 us/iter to ~2 us/iter at n ~ 600..1500.
        graph = self._get_or_build_pcr_iter_graph(A, n, n_pad, device, block_precond)

        if unchecked:
            for _ in range(max_iter):
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    self._pcr_iter_eager(A, n, n_pad, device)
            wp.copy(x_out, self._x, count=n)
            return (max_iter, None)

        nb_iter = 0
        check_every = self.check_every
        while nb_iter < max_iter and r_norm_sq > tol_sq:
            block_end = min(nb_iter + check_every, max_iter)
            block_iters = block_end - nb_iter
            if graph is not None:
                # Graph fast path: replay the captured single-iter graph
                # ``block_iters`` times.  All PCR state lives in the pre-
                # allocated device buffers captured at recording time, so
                # the replays operate on the same memory as the eager path.
                for _ in range(block_iters):
                    wp.capture_launch(graph)
                nb_iter = block_end
            else:
                while nb_iter < block_end:
                    self._pcr_iter_eager(A, n, n_pad, device)
                    nb_iter += 1

            self._dot(self._r, self._r, self._r_norm_d, n)
            r_norm_sq = float(self._r_norm_d.numpy()[0])
            if self._stopped_d.numpy()[0] != 0.0:
                break
        rho = float(self._rho_d.numpy()[0])

        # Copy x back to caller-provided output buffer.
        wp.copy(x_out, self._x, count=n)

        return (nb_iter, rho)

    def solve_sparse(
        self,
        W_bsr,
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        block_precond: Callable[[wp.array, wp.array], None] | None = None,
    ) -> int:
        """Sparse PCR: ``W_bsr @ x = b`` for a BSR ``W`` with ``1x1`` blocks.

        Matches :meth:`solve` numerically: same termination criterion, same
        Jacobi preconditioner, same ``check_every`` host-sync cadence, but
        replaces the dense tiled matvec with :func:`warp.sparse.bsr_mv`. The
        sparse lite NSN takes it through :meth:`solve_sparse_fused` for runs
        that check the residual, and the sparse full NSN calls it directly.

        The residual is first checked after ``check_every`` iterations. When all
        ``max_iter`` iterations fit before it, the tolerance cannot end the run
        early, and they run without reading the device: a zero ``b`` then leaves
        ``x = 0`` through the protected step.

        Args:
            W_bsr: ``BsrMatrix`` of block shape ``(1, 1)`` and scalar type
                ``wp.float64``, such as the Newton matrix of the sparse lite NSN.
            b: ``(n,)`` RHS, fp64.
            x_out: ``(n,)`` output, fp64.
            block_precond: Optional exact block solve over the Jacobi
                product, as in :meth:`solve`.

        Returns:
            The iterations taken.

        Raises:
            ValueError: when shapes are inconsistent.
        """
        self._fixed_sums = False
        import warp.sparse as wps  # noqa: PLC0415

        n = int(b.shape[0])
        if n <= 0:
            raise ValueError(f"system size must be positive, got n={n}")
        if n > self.max_n:
            raise ValueError(f"system size n={n} exceeds preallocated max_n={self.max_n}")
        if int(W_bsr.shape[0]) != n or int(W_bsr.shape[1]) != n:
            raise ValueError(f"W_bsr shape {tuple(W_bsr.shape)} incompatible with b shape ({n},)")
        if x_out.shape[0] != n:
            raise ValueError(f"x_out shape ({x_out.shape[0]},) incompatible with b shape ({n},)")

        if self.world_rows is not None:
            return self._solve_worlds(W_bsr, b, x_out, block_precond, sparse=True)

        device = self.device
        max_iter = min(self.max_iter, n)
        unchecked = max_iter <= self.check_every

        # ----- x = 0 ----------------------------------------------------
        self._x.zero_()

        # ----- Early exit if b == 0 -------------------------------------
        if not unchecked:
            self._dot(b, b, self._r_norm_d, n)
            dot_b = float(self._r_norm_d.numpy()[0])
            if dot_b == 0.0:
                wp.copy(x_out, self._x, count=n)
                return 0

        # ----- Build Jacobi preconditioner from W's diagonal ------------
        diag_blocks = wps.bsr_get_diag(W_bsr, out=self._diag)
        wp.launch(
            _precond_from_bsr_diag_kernel,
            dim=n,
            inputs=[diag_blocks],
            outputs=[self._precond],
            device=device,
        )

        # ----- Symmetric PCR, as in :meth:`solve` -------------------------
        # ``wps.bsr_mv`` checks that x.shape[0] == W.cols, so the scratch
        # buffers (sized to max_n) are sliced to the active size ``n``.
        self._block_precond = block_precond
        wp.copy(self._r, b, count=n)
        wp.launch(_apply_precond_kernel, dim=n, inputs=[self._r, self._precond], outputs=[self._s], device=device)
        if block_precond is not None:
            block_precond(self._r, self._s)
        wp.copy(self._d, self._s, count=n)
        wps.bsr_mv(W_bsr, self._d[:n], self._q[:n])
        self._dot(self._s, self._q, self._rho_d, n)
        self._stopped_d.zero_()
        if unchecked:
            for _ in range(max_iter):
                self._pcr_iter_eager(None, n, n, device, W_bsr=W_bsr, diag=diag_blocks)
            wp.copy(x_out, self._x, count=n)
            return max_iter
        tol_sq = self.tol * self.tol * dot_b
        r_norm_sq = dot_b
        # The checked iterations read the residual norm from the device, so
        # they run eagerly; ``check_every`` bounds the host round trips.
        nb_iter = 0
        while nb_iter < max_iter and r_norm_sq > tol_sq:
            block_end = min(nb_iter + self.check_every, max_iter)
            while nb_iter < block_end:
                self._pcr_iter_eager(None, n, n, device, W_bsr=W_bsr, diag=diag_blocks)
                nb_iter += 1
            self._dot(self._r, self._r, self._r_norm_d, n)
            r_norm_sq = float(self._r_norm_d.numpy()[0])
            if self._stopped_d.numpy()[0] != 0.0:
                break
        wp.copy(x_out, self._x, count=n)
        return nb_iter

    def solve_sparse_fused(
        self,
        W_bsr,
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        block_precond: Callable[[wp.array, wp.array], None] | None = None,
    ) -> int:
        """:meth:`solve_sparse` for a fixed iteration count, in four or five launches per iteration that read nothing
        from the device, and the launches of ``block_precond`` besides.

        The scalars of an iteration reduce in blocks of rows, and every block sums the blocks' partial sums
        alike, so that the run is deterministic; the matrix products run one thread per row in column order, or,
        for rows of more than 128 entries on average, as :func:`warp.sparse.bsr_mv`'s tiled product. Sums
        run in another order than :meth:`solve_sparse`'s, so results differ by rounding. With :attr:`world_rows`,
        each world reduces its own chunks of rows and keeps its own scalars, row budget and stop flag, as in
        :meth:`solve_sparse`. Runs that :meth:`solve_sparse` checks for convergence, single systems of more than
        262144 rows, worlds of more chunks of 256 rows in all than ``max(1024, max_n)``, worlds with a block
        preconditioner, and solves off CUDA take :meth:`solve_sparse`.

        Args:
            W_bsr: Scalar ``BsrMatrix`` with its columns in order in each row.
            b: ``(n,)`` RHS, fp64.
            x_out: ``(n,)`` output, fp64.
            block_precond: Optional exact block solve over the Jacobi
                product, as in :meth:`solve`, whose launches read nothing
                from the device either.

        Returns:
            The iterations taken.
        """
        if self.world_rows is not None and block_precond is not None:
            return self.solve_sparse(W_bsr, b, x_out, block_precond=block_precond)
        if self.world_rows is not None:
            _, offsets, _, max_rows = self.world_rows
            chunks = (max_rows + _FUSED_BLOCK - 1) // _FUSED_BLOCK
            unchecked = min(self.max_iter, max_rows) <= self.check_every
            if (
                not unchecked
                or (offsets.shape[0] - 1) * chunks > self._fused_partials.shape[1]
                or not wp.get_device(self.device).is_cuda
            ):
                return self.solve_sparse(W_bsr, b, x_out)
            return self._solve_worlds_fused(W_bsr, b, x_out)
        if not self.fuses_alone(int(b.shape[0])):
            return self.solve_sparse(W_bsr, b, x_out, block_precond=block_precond)
        return self._fused_single(W_bsr, b, x_out, block_precond)

    def fuses_alone(self, n: int) -> bool:
        """Whether :meth:`solve_sparse_fused` runs a single system of ``n`` rows fused: without :attr:`world_rows`,
        a run it does not check, and on CUDA."""
        return self.world_rows is None and self.fuses_single(n)

    def fuses_single(self, n: int) -> bool:
        """Whether :meth:`solve_sparse_fused` runs a system of ``n`` rows of one world fused."""
        nb = (n + _FUSED_BLOCK - 1) // _FUSED_BLOCK
        unchecked = min(self.max_iter, n) <= self.check_every
        return unchecked and nb <= _FUSED_MAX_BLOCKS and wp.get_device(self.device).is_cuda

    def solve_sparse_scaled(
        self,
        W_bsr,
        scale: wp.array[wp.float64],
        shift: wp.array[wp.float64],
        diag: wp.array[wp.float64],
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        block_precond: Callable[[wp.array, wp.array], None] | None = None,
    ) -> int:
        """:meth:`solve_sparse_fused` for ``A = diag(scale) W diag(scale) + diag(shift)``, whose diagonal is ``diag``,
        without forming ``A``: each product by ``A`` multiplies by ``W`` between the scalings, which the step and the
        reductions of the iteration apply. Only for systems that :meth:`fuses_alone` fuses.

        Args:
            W_bsr: Scalar ``BsrMatrix`` with its columns in order in each row, or a ``Block3Matrix`` of 3x3 blocks
                with its block columns in order.
            scale: ``(n,)`` row and column scaling, fp64.
            shift: ``(n,)`` diagonal added after the scaling, fp64.
            diag: ``(n,)`` diagonal of ``A``, fp64, for the Jacobi preconditioner.
            b: ``(n,)`` RHS, fp64.
            x_out: ``(n,)`` output, fp64.
            block_precond: Optional exact block solve over the Jacobi product, as in :meth:`solve_sparse_fused`.

        Returns:
            The iterations taken.
        """
        if not self.fuses_alone(int(b.shape[0])):
            raise ValueError("solve_sparse_scaled takes systems that solve_sparse_fused fuses alone")
        return self._fused_single(W_bsr, b, x_out, block_precond, (scale, shift, diag))

    def _fused_single(self, W_bsr, b: wp.array[wp.float64], x_out: wp.array[wp.float64], block_precond, factors=None):
        """The fused iterations of a single system, of ``W_bsr`` or, with ``factors = (scale, shift, diag)``, of
        ``diag(scale) W_bsr diag(scale) + diag(shift)``."""
        import warp.sparse as wps  # noqa: PLC0415

        n = int(b.shape[0])
        nb = (n + _FUSED_BLOCK - 1) // _FUSED_BLOCK
        max_iter = min(self.max_iter, n)
        device = self.device
        if n <= 0 or n > self.max_n:
            raise ValueError(f"system size n={n} outside 1..{self.max_n}")
        if self._takes_small(W_bsr, n, block_precond, factors):
            return self._small_single(W_bsr, b, x_out, block_precond, factors, max_iter)
        pden, pcurv, pchart, prho = (self._fused_partials[k][:nb] for k in range(4))
        x, r, s, d, q, h, sw = (a[:n] for a in (self._x, self._r, self._s, self._d, self._q, self._h, self._sw))
        precond, diag = self._precond[:n], self._diag[:n]
        matrix = [W_bsr.offsets, W_bsr.columns, W_bsr.values]
        scaled = factors is not None
        scale, shift = (factors[0], factors[1]) if scaled else (s, s)
        # A matrix of 3x3 blocks comes factored only: its diagonal is given and its product is block3_mv's.
        blocked = getattr(W_bsr, "block3", False)
        if blocked and not scaled:
            raise ValueError("a matrix of 3x3 blocks takes solve_sparse_scaled")
        # Rows of more than _LONG_ROW entries on average take warp.sparse's tiled product: one block of threads per
        # row reads the row's entries together, where one thread per row reads them one by one.
        long_rows = int(W_bsr.nnz) > _LONG_ROW * n
        x.zero_()
        if scaled:
            wp.copy(diag, factors[2], count=n)
        else:
            wp.launch(csr_diag_kernel, n, matrix, [diag], device=device)
        wp.launch(_precond_from_bsr_diag_kernel, n, [diag], [precond], device=device)
        wp.copy(r, b, count=n)
        wp.launch(_apply_precond_kernel, n, [r, precond], [s], device=device)
        if block_precond is not None:
            block_precond(r, s)
        wp.copy(d, s, count=n)
        product_in = d
        if scaled:
            wp.launch(_scale_kernel, n, [scale, d], [sw], device=device)
            product_in = sw
        if blocked:
            block3_mv(W_bsr, product_in, q)
        elif long_rows:
            wps.bsr_mv(W_bsr, product_in, q, tile_size=_TILED_MV)
        else:
            wp.launch(csr_mv_kernel, n, [*matrix, product_in], [q], device=device)
        if scaled:
            wp.launch(_scale_shift_kernel, n, [scale, shift, d], [q], device=device)
        wp.utils.array_inner(s, q, out=self._fused_rho[0:1], count=n)
        self._stopped_d.zero_()
        for k in range(max_iter):
            slot = k % 2
            if block_precond is None:
                inputs = [q, precond, d, diag, h, pden, pcurv, pchart]
                wp.launch_tiled(_fused_pcr_dots_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            else:
                wp.launch(_apply_precond_kernel, n, [q, precond], [h], device=device)
                block_precond(q, h)
                inputs = [q, h, d, diag, pden, pcurv, pchart]
                wp.launch_tiled(
                    _fused_pcr_dots_given_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device
                )
            inputs = [pden, pcurv, pchart, self._fused_rho, slot, self._eta, n, d, q, h, self._stopped_d, x, r, s]
            inputs += [int(scaled), scale, sw]
            wp.launch_tiled(_fused_pcr_step_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            if blocked or long_rows:
                if blocked:
                    block3_mv(W_bsr, sw, h)
                else:
                    wps.bsr_mv(W_bsr, sw if scaled else s, h, tile_size=_TILED_MV)
                kernel = _fused_pcr_rho_scaled_kernel if scaled else _fused_pcr_rho_kernel
                inputs = [s, scale, shift, n, h, prho] if scaled else [s, h, prho]
                wp.launch_tiled(kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            elif scaled:
                inputs = [*matrix, sw, scale, shift, s, n, h, prho]
                wp.launch_tiled(
                    _fused_pcr_mv_scaled_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device
                )
            else:
                inputs = [*matrix, s, n, h, prho]
                wp.launch_tiled(_fused_pcr_mv_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            inputs = [prho, self._fused_rho, slot, n, s, h, d, q]
            wp.launch_tiled(_fused_pcr_direction_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
        wp.copy(x_out, x, count=n)
        return max_iter

    def _fused_dense(
        self,
        A: wp.array2d[wp.float64],
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        block_precond,
        active_rows: tuple[wp.array, int] | None = None,
        mask: wp.array2d[wp.int32] | None = None,
    ):
        """:meth:`solve` of a fixed iteration count with ``fixed_sums``, as :meth:`_fused_single` runs a sparse
        matrix, in launches that read nothing from the device: three per iteration without a block preconditioner
        (:meth:`_fused_dense_turns`), five with one besides its own, in place of some twenty. The dot products reduce
        in blocks of rows that every block sums alike, so that the run is deterministic; the product by ``A`` takes one
        block of threads per row, on the leading ``active_rows`` and the tiles of ``mask`` alone. Sums run in another
        order than the iteration of :meth:`_pcr_iter_eager`, so results differ by rounding. Without a block
        preconditioner, a system of at most :data:`_SMALL_DENSE_ROWS` rows takes :meth:`_small_dense` instead, and so
        does a larger one whose ``active_rows`` are as few, by a conditional node on their count. Returns the
        iterations taken."""
        n = int(b.shape[0])
        max_iter = min(self.max_iter, n)
        if self.small_single and block_precond is None:
            if n <= _SMALL_DENSE_ROWS:
                return self._small_dense(A, b, x_out, max_iter, active_rows)
            if active_rows is not None:
                count, per = active_rows
                inputs = [count, int(per), n, _SMALL_DENSE_ROWS]
                wp.launch(_small_dense_switch_kernel, 1, inputs, [self._small_dense_flag], device=self.device)
                wp.capture_if(
                    self._small_dense_flag,
                    on_true=lambda: self._small_dense(A, b, x_out, max_iter, active_rows),
                    on_false=lambda: self._fused_dense_launches(A, b, x_out, None, active_rows, mask),
                )
                return max_iter
        return self._fused_dense_launches(A, b, x_out, block_precond, active_rows, mask)

    def _fused_dense_launches(
        self,
        A: wp.array2d[wp.float64],
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        block_precond,
        active_rows: tuple[wp.array, int] | None = None,
        mask: wp.array2d[wp.int32] | None = None,
    ) -> int:
        """The launches of :meth:`_fused_dense`: :meth:`_fused_dense_turns` without a block preconditioner, five per
        iteration with one; with ``active_rows`` or ``mask``, the products by ``A`` take the leading rows and the
        marked tiles alone (:func:`_matvec_masked_kernel`). Returns the iterations taken."""
        n = int(b.shape[0])
        nb = (n + _FUSED_BLOCK - 1) // _FUSED_BLOCK
        max_iter = min(self.max_iter, n)
        device = self.device
        n_pad = ((n + int(_PCR_TILE_K) - 1) // int(_PCR_TILE_K)) * int(_PCR_TILE_K)
        pden, pcurv, pchart, prho = (self._fused_partials[k][:nb] for k in range(4))
        x, r, s, d, q, h = (a[:n] for a in (self._x, self._r, self._s, self._d, self._q, self._h))
        precond, diag = self._precond[:n], self._diag[:n]

        masked = active_rows is not None or mask is not None
        if masked:
            count, per = active_rows if active_rows is not None else (wp.empty(0, dtype=wp.int32, device=device), 1)
            tiles = mask if mask is not None else wp.empty((0, 0), dtype=wp.int32, device=device)

        if block_precond is None and nb <= int(_SERIAL_BLOCKS):
            return self._fused_dense_turns(A, b, x_out, active_rows, mask)

        def product(v, out):
            if masked:
                inputs, kernel, block = [A, v, tiles, count, int(per), s], _matvec_masked_kernel, int(_PCR_TILE_K)
                outputs = [out, wp.empty(0, dtype=wp.float64, device=device)]
            else:
                inputs, kernel, block = [A, v, n_pad], _matvec_tiled_kernel, _PCR_BLOCK_DIM
                outputs = [out]
            wp.launch_tiled(kernel, dim=n, inputs=inputs, outputs=outputs, block_dim=block, device=device)
            wp.launch_tiled(
                _fused_pcr_rho_kernel, dim=[nb], inputs=[s, out, prho], block_dim=_FUSED_BLOCK, device=device
            )

        x.zero_()
        wp.launch(_dense_diag_kernel, n, [A], [diag], device=device)
        wp.launch(_precond_from_bsr_diag_kernel, n, [diag], [precond], device=device)
        wp.copy(r, b, count=n)
        wp.launch(_apply_precond_kernel, n, [r, precond], [s], device=device)
        if block_precond is not None:
            block_precond(r, s)
        wp.copy(d, s, count=n)
        # q = A d and rho = s . q, with d = s.
        product(d, q)
        wp.launch(_dot_finish_kernel, 1, [prho, nb], [self._fused_rho[0:1]], device=device)
        self._stopped_d.zero_()
        for k in range(max_iter):
            slot = k % 2
            if block_precond is None:
                inputs = [q, precond, d, diag, h, pden, pcurv, pchart]
                wp.launch_tiled(_fused_pcr_dots_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            else:
                wp.launch(_apply_precond_kernel, n, [q, precond], [h], device=device)
                block_precond(q, h)
                inputs = [q, h, d, diag, pden, pcurv, pchart]
                wp.launch_tiled(
                    _fused_pcr_dots_given_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device
                )
            inputs = [pden, pcurv, pchart, self._fused_rho, slot, self._eta, n, d, q, h, self._stopped_d, x, r, s]
            inputs += [0, s, s]
            wp.launch_tiled(_fused_pcr_step_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            # h = A s and the block sums of s . h.
            product(s, h)
            inputs = [prho, self._fused_rho, slot, n, s, h, d, q]
            wp.launch_tiled(_fused_pcr_direction_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
        wp.copy(x_out, x, count=n)
        return max_iter

    def _fused_dense_turns(
        self,
        A: wp.array2d[wp.float64],
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        active_rows: tuple[wp.array, int] | None,
        mask: wp.array2d[wp.int32] | None,
    ) -> int:
        """:meth:`_fused_dense_launches` without a block preconditioner, for at most ``_SERIAL_BLOCKS`` blocks of
        rows, in three launches per iteration (:func:`_fused_pcr_turn_kernel`, :func:`_fused_pcr_step_kernel` and
        :func:`_matvec_masked_kernel`, whose product also writes each row's ``s h`` term), and without the last
        iteration's product, which no step reads: the sums of the launches with that product, in their order, bit for
        bit. Returns the iterations taken."""
        n = int(b.shape[0])
        nb = (n + _FUSED_BLOCK - 1) // _FUSED_BLOCK
        max_iter = min(self.max_iter, n)
        device = self.device
        pden, pcurv, pchart = (self._fused_partials[k][:nb] for k in range(3))
        x, r, s, d, q, h = (a[:n] for a in (self._x, self._r, self._s, self._d, self._q, self._h))
        precond, diag, sh = self._precond[:n], self._diag[:n], self._sw[:n]
        if active_rows is not None:
            count, per = active_rows
        else:
            count, per = wp.empty(0, dtype=wp.int32, device=device), 1
        tiles = mask if mask is not None else wp.empty((0, 0), dtype=wp.int32, device=device)

        def product(v, out):
            inputs = [A, v, tiles, count, int(per), s]
            outputs = [out, sh]
            wp.launch_tiled(
                _matvec_masked_kernel, dim=n, inputs=inputs, outputs=outputs, block_dim=int(_PCR_TILE_K), device=device
            )

        x.zero_()
        wp.launch(_dense_diag_kernel, n, [A], [diag], device=device)
        wp.launch(_precond_from_bsr_diag_kernel, n, [diag], [precond], device=device)
        wp.copy(r, b, count=n)
        wp.launch(_apply_precond_kernel, n, [r, precond], [s], device=device)
        wp.copy(d, s, count=n)
        # q = A d and the rows' terms of rho = s . q, with d = s.
        product(d, q)
        self._stopped_d.zero_()
        for k in range(max_iter):
            slot = k % 2
            inputs = [sh, nb, self._fused_rho, slot, int(k == 0), n, s, precond, diag, d, q, h, pden, pcurv, pchart]
            wp.launch_tiled(_fused_pcr_turn_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            inputs = [pden, pcurv, pchart, self._fused_rho, slot, self._eta, n, d, q, h, self._stopped_d, x, r, s]
            inputs += [0, s, s]
            wp.launch_tiled(_fused_pcr_step_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            if k + 1 < max_iter:
                # h = A s and the rows' terms of s . h, which the next turn adds.
                product(s, h)
        wp.copy(x_out, x, count=n)
        return max_iter

    def _small_dense(
        self,
        A: wp.array2d[wp.float64],
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        max_iter: int,
        active_rows: tuple[wp.array, int] | None,
    ) -> int:
        """:meth:`_fused_dense` in one launch: one block of :data:`_SMALL_DENSE_ROWS` threads, one row per thread,
        takes every iteration (:func:`_small_dense_pcr_kernel`), on the leading rows that ``active_rows`` counts
        alone, at most as many as the threads. Its sums reduce in one tile, in another order than the iteration's
        blocks, so that results differ by rounding."""
        n, device = int(b.shape[0]), self.device
        if active_rows is None:
            active, per = wp.empty(0, dtype=wp.int32, device=device), 1
        else:
            active, per = active_rows
        inputs = [A, b, n, active, int(per), max_iter, self._eta, _SMALL_DENSE_ROWS]
        wp.launch_tiled(
            _small_dense_pcr_kernel, dim=[1], inputs=inputs, outputs=[x_out], block_dim=_SMALL_DENSE_ROWS, device=device
        )
        return max_iter

    def _takes_small(self, W_bsr, n: int, block_precond, factors) -> bool:
        """Whether :meth:`_fused_single` runs its iteration in one block of threads (:meth:`_small_single`): a system
        of at most :data:`_SMALL_ROWS` rows, of a factored matrix of 3x3 blocks or of a scalar matrix, factored or not,
        whose rows hold at most :data:`_LONG_ROW` entries on average, without a block preconditioner or with joint rows'
        block solves (:class:`~newton._src.solvers.fba.joints.JointRowsPrecond`) of a few articulations. Longer rows
        take the iteration of several launches, whose tiled product reads a row with a block of threads, where the
        single block reads it with one thread."""
        blocked = getattr(W_bsr, "block3", False)
        if not self.small_single or (blocked and factors is None) or n > _SMALL_ROWS:
            return False
        if not blocked and int(W_bsr.nnz) > _LONG_ROW * n:
            return False
        if block_precond is None:
            return True
        system = getattr(block_precond, "system", None)
        return system is not None and system.L.shape[0] <= _SMALL_JOINT_BLOCKS

    def _small_single(self, W_bsr, b, x_out, block_precond, factors, max_iter: int) -> int:
        """:meth:`_fused_single` in one launch: one block of threads, one row per thread, takes every iteration
        (:func:`_small_pcr_kernel`). Its sums reduce in one tile, in another order than the iteration's blocks, so
        that results differ by rounding. An unscaled matrix takes the factors 1 and 0, which leave its products
        unchanged."""
        n, device = int(b.shape[0]), self.device
        blocked = getattr(W_bsr, "block3", False)
        if factors is None:
            wp.launch(csr_diag_kernel, n, [W_bsr.offsets, W_bsr.columns, W_bsr.values], [self._diag], device=device)
            factors = (self._small_unit[0][:n], self._small_unit[1][:n], self._diag[:n])
        scale, shift, diag = factors
        place_block, place_local = self._small_places
        place_block.fill_(-1)
        place_local.zero_()
        if block_precond is None:
            L, blocks, tile = self._small_no_factor, 0, 8
        else:
            system = block_precond.system
            L, blocks, tile = system.L, int(system.L.shape[0]), int(system.tile)
            rows = block_precond.rows
            inputs = [rows, system.row_block, system.row_local, int(block_precond.offset)]
            wp.launch(joint_row_places_kernel, rows.shape[0], inputs, [place_block, place_local], device=device)
        block_dim = 256 if n <= 256 else _SMALL_ROWS
        inputs = [W_bsr.offsets, W_bsr.columns, W_bsr.values, scale, shift, diag, b, n, max_iter, self._eta, L]
        inputs += [blocks, place_block, place_local]
        kernel = _small_pcr_kernel(tile, blocked)
        wp.launch_tiled(kernel, dim=[1], inputs=inputs, outputs=[x_out], block_dim=block_dim, device=device)
        return max_iter

    def _solve_worlds_fused(self, W_bsr, b: wp.array[wp.float64], x_out: wp.array[wp.float64]) -> int:
        """The recurrence of :meth:`_solve_worlds` for a fixed iteration count, fused as in
        :meth:`solve_sparse_fused` over each world's chunks of rows; it reads nothing from the device."""
        world, offsets, rows, max_rows = self.world_rows
        n, nw, device = int(b.shape[0]), offsets.shape[0] - 1, self.device
        if not 0 < n <= self.max_n:
            raise ValueError(f"system size n={n} outside 1..{self.max_n}")
        if world.shape[0] != n or x_out.shape[0] != n or tuple(int(k) for k in W_bsr.shape) != (n, n):
            raise ValueError("world rows, matrix and vectors must describe the same system")
        chunks = (max_rows + _FUSED_BLOCK - 1) // _FUSED_BLOCK
        grid = nw * chunks
        if grid > self._fused_partials.shape[1] or nw > self._world_fused_b2.shape[0]:
            raise ValueError(
                f"{nw} worlds of {chunks} chunks exceed the fused scratch of {self._fused_partials.shape[1]}"
            )
        pden, pcurv, pchart, prho = (self._fused_partials[k][:grid] for k in range(4))
        x, r, s, d, q, h = (a[:n] for a in (self._x, self._r, self._s, self._d, self._q, self._h))
        precond, diag = self._precond[:n], self._diag[:n]
        matrix = [W_bsr.offsets, W_bsr.columns, W_bsr.values]
        segments = [offsets, rows, chunks]
        rho, b2, stopped = self._world_fused_rho, self._world_fused_b2, self._world_fused_stopped
        x.zero_()
        wp.launch(csr_diag_kernel, n, matrix, [diag], device=device)
        wp.launch(_precond_from_bsr_diag_kernel, n, [diag], [precond], device=device)
        wp.copy(r, b, count=n)
        wp.launch(_apply_precond_kernel, n, [r, precond], [s], device=device)
        wp.copy(d, s, count=n)
        wp.launch(csr_mv_kernel, n, [*matrix, d], [q], device=device)
        inputs = [*segments, s, q, b]
        wp.launch_tiled(
            _world_fused_start_kernel,
            dim=[grid],
            inputs=inputs,
            outputs=[pden, pcurv],
            block_dim=_FUSED_BLOCK,
            device=device,
        )
        for partial, out in ((pden, rho[0]), (pcurv, b2)):
            inputs = [partial, chunks]
            wp.launch_tiled(
                _world_fused_sum_kernel, dim=[nw], inputs=inputs, outputs=[out], block_dim=_FUSED_BLOCK, device=device
            )
        stopped.zero_()
        iterations = min(self.max_iter, max_rows)
        tol2 = wp.float64(self.tol**2)
        for k in range(iterations):
            slot = k % 2
            inputs = [*segments, q, precond, d, diag, h, pden, pcurv, pchart]
            wp.launch_tiled(_world_fused_dots_kernel, dim=[grid], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            inputs = [*segments, pden, pcurv, pchart, rho, b2, slot, k, tol2, self._eta, d, q, h, stopped, x, r, s]
            wp.launch_tiled(_world_fused_step_kernel, dim=[grid], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            inputs = [*segments, *matrix, s, h, prho]
            wp.launch_tiled(_world_fused_mv_kernel, dim=[grid], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            inputs = [*segments, prho, rho, slot, s, h, d, q]
            wp.launch_tiled(
                _world_fused_direction_kernel, dim=[grid], inputs=inputs, block_dim=_FUSED_BLOCK, device=device
            )
        wp.copy(x_out, x, count=n)
        return iterations

    # ------------------------------------------------------------------
    # Per-iter body + graph capture
    # ------------------------------------------------------------------

    def _pcr_iter_eager(
        self,
        A: wp.array2d[wp.float64] | None,
        n: int,
        n_pad: int,
        device: wp.Device | str,
        W_bsr=None,
        diag: wp.array[wp.float64] | None = None,
    ) -> None:
        """Issue one symmetric PCR iteration of :meth:`solve` on the active stream.

        All launches are stream-ordered and safe to call inside a CUDA graph
        capture.

        With ``W_bsr`` and its ``diag``, the iteration of :meth:`solve_sparse`
        multiplies by the sparse matrix instead of ``A``.
        """
        # t = P q, den = qᵀ P q.
        wp.launch(_apply_precond_kernel, dim=n, inputs=[self._q, self._precond], outputs=[self._h], device=device)
        if self._block_precond is not None:
            self._block_precond(self._q, self._h)
        self._inner(self._q, self._h, self._den_d, n)
        # Chart curvature of p: pᵀ A p against pᵀ diag(A) p.
        if W_bsr is None:
            wp.launch(_diag_scale_kernel, dim=n, inputs=[A, self._d], outputs=[self._dd], device=device)
        else:
            wp.launch(_apply_precond_kernel, dim=n, inputs=[self._d, diag], outputs=[self._dd], device=device)
        self._inner(self._d, self._q, self._curvature_d, n)
        self._inner(self._d, self._dd, self._chart_norm_d, n)
        wp.launch(
            _protected_alpha_kernel,
            dim=1,
            inputs=[self._rho_d, self._den_d, self._curvature_d, self._chart_norm_d, self._eta],
            outputs=[self._alpha_d, self._stopped_d],
            device=device,
        )
        # x += alpha p, r -= alpha q, z -= alpha t.
        for vec, sign, out in ((self._d, 1.0, self._x), (self._q, -1.0, self._r), (self._h, -1.0, self._s)):
            wp.launch(
                _axpy_arr_kernel,
                dim=n,
                inputs=[self._alpha_d, wp.float64(sign), vec],
                outputs=[out],
                device=device,
            )
        # t = A z, rho = zᵀ A z, beta = rho / rho_old.
        if W_bsr is None:
            wp.launch_tiled(
                _matvec_tiled_kernel,
                dim=n,
                inputs=[A, self._s, n_pad],
                outputs=[self._h],
                block_dim=_PCR_BLOCK_DIM,
                device=device,
            )
        else:
            import warp.sparse as wps  # noqa: PLC0415

            wps.bsr_mv(W_bsr, self._s[:n], self._h[:n])
        wp.launch(_copy_scalar_kernel, dim=1, inputs=[self._rho_d], outputs=[self._rho_old_d], device=device)
        self._inner(self._s, self._h, self._rho_d, n)
        wp.launch(
            _div_scalar_kernel,
            dim=1,
            inputs=[self._rho_d, self._rho_old_d],
            outputs=[self._beta_d],
            device=device,
        )
        # p = z + beta p, q = t + beta q.
        wp.launch(_scal_add_arr_kernel, dim=n, inputs=[self._beta_d, self._s], outputs=[self._d], device=device)
        wp.launch(_scal_add_arr_kernel, dim=n, inputs=[self._beta_d, self._h], outputs=[self._q], device=device)

    def _inner(self, a: wp.array, b: wp.array, out: wp.array, n: int) -> None:
        """``out[0] = a[:n] . b[:n]``: by ``wp.utils.array_inner``, or with ``fixed_sums`` on CUDA by per-block tile
        sums added in block order, which allocate nothing (CUDA graph conditional bodies take no allocations)."""
        if not self._fixed_sums or not wp.get_device(self.device).is_cuda:
            self._dot(a, b, out, n)
            return
        blocks = (n + int(_DOT_BLOCK) - 1) // int(_DOT_BLOCK)
        wp.launch_tiled(
            _dot_partials_kernel,
            dim=blocks,
            inputs=[a, b, n],
            outputs=[self._dot_partials],
            block_dim=int(_DOT_BLOCK),
            device=self.device,
        )
        wp.launch(_dot_finish_kernel, 1, [self._dot_partials, blocks], [out], device=self.device)

    def _dot(self, a: wp.array, b: wp.array, out: wp.array, n: int) -> None:
        """``out[0] = a[:n] . b[:n]``: on the CPU in row order (:func:`_serial_dot_kernel`), as the solves of several
        worlds add each world's (:func:`_world_dot_kernel`), so that a world alone and among others take the same
        sums on every machine; on CUDA by ``wp.utils.array_inner``."""
        if wp.get_device(self.device).is_cpu:
            wp.launch(_serial_dot_kernel, 1, [a, b, n], [out], device=self.device)
        else:
            wp.utils.array_inner(a, b, out=out, count=n)

    def _get_or_build_pcr_iter_graph(
        self,
        A: wp.array2d[wp.float64],
        n: int,
        n_pad: int,
        device: wp.Device | str,
        block_precond: Callable[[wp.array, wp.array], None] | None = None,
    ) -> wp.Graph | None:
        """Look up (or capture on first miss) the single-PCR-iter CUDA graph.

        The graph is keyed by ``(n, A.ptr, A.strides)`` and the block preconditioner: ``n`` controls the launch
        dimensions, and the pointer and strides control which matrix block the captured matvec reads from. The
        joint rows' solve as block preconditioner (:class:`~newton._src.solvers.fba.joints.JointRowsPrecond`)
        records its launches into the graph too, keyed by its ``graph_key``, and the cache keeps it alive; another
        callable runs the iterations eagerly. When any of them changes between solve calls we re-capture; PCR is
        called repeatedly with the same matrix view and preconditioner within a Newton step so amortised capture
        overhead is near zero.

        Returns ``None`` if capture is impossible on the active device
        (e.g.  CPU device) so callers fall through to the eager loop.
        """
        # CUDA graphs require a CUDA device.
        dev_obj = wp.get_device(device) if isinstance(device, str) else device
        if not dev_obj.is_cuda:
            return None
        precond_key, precond_objects = None, ()
        if block_precond is not None:
            # The joint rows' solve (JointRowsPrecond) keys the graph by its graph_key; another callable runs the
            # iterations eagerly.
            precond_key = getattr(block_precond, "graph_key", None)
            if precond_key is None:
                return None
            precond_objects = (block_precond,)

        a_ptr = int(A.__cuda_array_interface__["data"][0])
        # Cache hash combines ``n`` (drives launch dims + n_pad) and the
        # matrix data pointer and strides: a reallocated buffer can return
        # at the same address with a different row pitch.
        key = (n, a_ptr, tuple(A.strides), precond_key, self._fixed_sums)
        if wp.get_device(device).is_capturing:
            # Inside another capture the iterations record into it; a capture here would fail and stick.
            return None
        cached = self._pcr_graph_cache.get(key, "missing")
        if cached == "missing":
            # Capture lazily. Failures are sticky: we cache ``None`` so
            # we don't retry capture every solve once it has failed (e.g.
            # if some kernel inside the iter body issues an implicit host
            # allocation that traps the capture). The preconditioner's objects stay alive with the graph, so that
            # no new object takes one of their identities.
            try:
                with scoped_capture(device) as cap:
                    self._pcr_iter_eager(A, n, n_pad, device)
                self._pcr_graph_cache = {key: cap.graph}
                self._pcr_graph_objects = precond_objects
                return cap.graph
            except RuntimeError:
                self._pcr_graph_cache = {key: None}
                self._pcr_graph_objects = precond_objects
                return None
        return cached
