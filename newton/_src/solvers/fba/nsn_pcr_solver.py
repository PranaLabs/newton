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

import warp as wp

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


@wp.kernel
def csr_diag_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    diag: wp.array[wp.float64],
):
    """The diagonal of a scalar CSR matrix, zero where a row stores none."""
    i = wp.tid()
    d = wp.float64(0.0)
    for k in range(offsets[i], offsets[i + 1]):
        if columns[k] == i:
            d = values[k]
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
):
    """The protected step length from the summed blocks, which every block forms alike, and ``x += alpha d``,
    ``r -= alpha q``, ``s -= alpha h``."""
    b, lane = wp.tid()
    den = wp.tile_extract(wp.tile_sum(wp.tile_load(partial_den, shape=(_FUSED_MAX_BLOCKS,))), 0)
    curvature = wp.tile_extract(wp.tile_sum(wp.tile_load(partial_curvature, shape=(_FUSED_MAX_BLOCKS,))), 0)
    chart = wp.tile_extract(wp.tile_sum(wp.tile_load(partial_chart, shape=(_FUSED_MAX_BLOCKS,))), 0)
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
        s[i] = s[i] - alpha * h[i]


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
    rho_new = wp.tile_extract(wp.tile_sum(wp.tile_load(partial_rho, shape=(_FUSED_MAX_BLOCKS,))), 0)
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
        self._fused_partials = wp.zeros((4, _FUSED_MAX_BLOCKS), dtype=wp.float64, device=device)
        self._fused_rho = wp.zeros(2, dtype=wp.float64, device=device)

        # Cache one ``wp.Graph`` per active ``n`` shape so repeated
        # solves at the same shape replay a pre-recorded CUDA graph for the
        # inner PCR iter, eliminating per-launch Python+driver overhead.
        # Capture is attempted lazily on first solve at each shape; if it
        # fails we fall back to the eager loop and remember the failure so
        # we don't retry on every call.
        self._pcr_graph_cache: dict[int, wp.Graph | None] = {}
        self._block_precond = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        A: wp.array2d[wp.float64],
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
        block_precond: Callable[[wp.array, wp.array], None] | None = None,
    ) -> tuple[int, float]:
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
                the bilateral rows. It runs eagerly, without the CUDA graph.

        Returns:
            A ``(iterations_taken, final_rho)`` tuple where ``iterations_taken``
            counts the number of inner PCR iterations executed and
            ``final_rho`` is the last value of the preconditioned residual
            scalar ``rho = <z, A z>`` with ``z = P r`` (>= 0 for an SPD ``A``).

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

        device = self.device
        max_iter = min(self.max_iter, n)
        tile_k = int(_PCR_TILE_K)
        n_pad = ((n + tile_k - 1) // tile_k) * tile_k

        # ----- Initialize x = 0 -----------------------------------------
        self._x.zero_()

        # ----- Early exit if b == 0 -------------------------------------
        # With b == 0, x = 0 is the exact solution.  ``array_inner`` is a
        # CUB-style parallel reduction, far cheaper than a single-thread
        # accumulator at the typical n ~ 600..1500 sizes.
        dot_b = float(wp.utils.array_inner(b, b, count=n))
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
        wp.utils.array_inner(self._s, self._q, out=self._rho_d, count=n)
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
        graph = None if block_precond is not None else self._get_or_build_pcr_iter_graph(A, n, n_pad, device)

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

            wp.utils.array_inner(self._r, self._r, out=self._r_norm_d, count=n)
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
    ) -> int:
        """LiteNSN sparse-Schur PCR: ``W_bsr @ x = b`` for a BSR ``W`` with
        ``1x1`` blocks.

        Matches :meth:`solve` numerically: same termination criterion, same
        Jacobi preconditioner, same ``check_every`` host-sync cadence, but
        replaces the dense tiled matvec with :func:`warp.sparse.bsr_mv`. The
        sparse lite NSN takes it through :meth:`solve_sparse_fused` for runs
        that check the residual.

        The residual is first checked after ``check_every`` iterations. When all
        ``max_iter`` iterations fit before it, the tolerance cannot end the run
        early, and they run without reading the device: a zero ``b`` then leaves
        ``x = 0`` through the protected step.

        Args:
            W_bsr: ``BsrMatrix`` of block shape ``(1, 1)`` and scalar type
                ``wp.float64``, such as the Newton matrix of the sparse lite NSN.
            b: ``(n,)`` RHS, fp64.
            x_out: ``(n,)`` output, fp64.

        Returns:
            The iterations taken.

        Raises:
            ValueError: when shapes are inconsistent.
        """
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

        device = self.device
        max_iter = min(self.max_iter, n)
        unchecked = max_iter <= self.check_every

        # ----- x = 0 ----------------------------------------------------
        self._x.zero_()

        # ----- Early exit if b == 0 -------------------------------------
        if not unchecked:
            dot_b = float(wp.utils.array_inner(b, b, count=n))
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
        self._block_precond = None
        wp.copy(self._r, b, count=n)
        wp.launch(_apply_precond_kernel, dim=n, inputs=[self._r, self._precond], outputs=[self._s], device=device)
        wp.copy(self._d, self._s, count=n)
        wps.bsr_mv(W_bsr, self._d[:n], self._q[:n])
        wp.utils.array_inner(self._s, self._q, out=self._rho_d, count=n)
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
            wp.utils.array_inner(self._r, self._r, out=self._r_norm_d, count=n)
            r_norm_sq = float(self._r_norm_d.numpy()[0])
            if self._stopped_d.numpy()[0] != 0.0:
                break
        wp.copy(x_out, self._x, count=n)
        return nb_iter

    def solve_sparse_fused(self, W_bsr, b: wp.array[wp.float64], x_out: wp.array[wp.float64]) -> int:
        """:meth:`solve_sparse` for a fixed iteration count, in four launches per iteration that read nothing from
        the device.

        The scalars of an iteration reduce in blocks of rows, and every block sums the blocks' partial sums
        alike, so that the run is deterministic; the matrix products run one thread per row in column order. Sums
        run in another order than :meth:`solve_sparse`'s, so results differ by rounding. Runs that
        :meth:`solve_sparse` checks for convergence, systems of more than 262144 rows, and solves off CUDA take
        :meth:`solve_sparse`.

        Args:
            W_bsr: Scalar ``BsrMatrix`` with its columns in order in each row.
            b: ``(n,)`` RHS, fp64.
            x_out: ``(n,)`` output, fp64.

        Returns:
            The iterations taken.
        """
        n = int(b.shape[0])
        nb = (n + _FUSED_BLOCK - 1) // _FUSED_BLOCK
        max_iter = min(self.max_iter, n)
        device = self.device
        if max_iter > self.check_every or nb > _FUSED_MAX_BLOCKS or not wp.get_device(device).is_cuda:
            return self.solve_sparse(W_bsr, b, x_out)
        if n <= 0 or n > self.max_n:
            raise ValueError(f"system size n={n} outside 1..{self.max_n}")
        pden, pcurv, pchart, prho = (self._fused_partials[k][:nb] for k in range(4))
        x, r, s, d, q, h = (a[:n] for a in (self._x, self._r, self._s, self._d, self._q, self._h))
        precond, diag = self._precond[:n], self._diag[:n]
        matrix = [W_bsr.offsets, W_bsr.columns, W_bsr.values]
        x.zero_()
        wp.launch(csr_diag_kernel, n, matrix, [diag], device=device)
        wp.launch(_precond_from_bsr_diag_kernel, n, [diag], [precond], device=device)
        wp.copy(r, b, count=n)
        wp.launch(_apply_precond_kernel, n, [r, precond], [s], device=device)
        wp.copy(d, s, count=n)
        wp.launch(csr_mv_kernel, n, [*matrix, d], [q], device=device)
        wp.utils.array_inner(s, q, out=self._fused_rho[0:1], count=n)
        self._stopped_d.zero_()
        for k in range(max_iter):
            slot = k % 2
            inputs = [q, precond, d, diag, h, pden, pcurv, pchart]
            wp.launch_tiled(_fused_pcr_dots_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            inputs = [pden, pcurv, pchart, self._fused_rho, slot, self._eta, n, d, q, h, self._stopped_d, x, r, s]
            wp.launch_tiled(_fused_pcr_step_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            inputs = [*matrix, s, n, h, prho]
            wp.launch_tiled(_fused_pcr_mv_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
            inputs = [prho, self._fused_rho, slot, n, s, h, d, q]
            wp.launch_tiled(_fused_pcr_direction_kernel, dim=[nb], inputs=inputs, block_dim=_FUSED_BLOCK, device=device)
        wp.copy(x_out, x, count=n)
        return max_iter

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
        wp.utils.array_inner(self._q, self._h, out=self._den_d, count=n)
        # Chart curvature of p: pᵀ A p against pᵀ diag(A) p.
        if W_bsr is None:
            wp.launch(_diag_scale_kernel, dim=n, inputs=[A, self._d], outputs=[self._dd], device=device)
        else:
            wp.launch(_apply_precond_kernel, dim=n, inputs=[self._d, diag], outputs=[self._dd], device=device)
        wp.utils.array_inner(self._d, self._q, out=self._curvature_d, count=n)
        wp.utils.array_inner(self._d, self._dd, out=self._chart_norm_d, count=n)
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
        wp.utils.array_inner(self._s, self._h, out=self._rho_d, count=n)
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

    def _get_or_build_pcr_iter_graph(
        self,
        A: wp.array2d[wp.float64],
        n: int,
        n_pad: int,
        device: wp.Device | str,
    ) -> wp.Graph | None:
        """Look up (or capture on first miss) the single-PCR-iter CUDA graph.

        The graph is keyed by ``(n, A.ptr, A.strides)``: ``n`` controls the
        launch dimensions, and the pointer and strides control which matrix
        block the captured matvec reads from.  When any of them changes
        between solve calls we re-capture; PCR is called repeatedly with the same matrix view
        within a Newton step so amortised capture overhead is near zero.

        Returns ``None`` if capture is impossible on the active device
        (e.g.  CPU device) so callers fall through to the eager loop.
        """
        # CUDA graphs require a CUDA device.
        dev_obj = wp.get_device(device) if isinstance(device, str) else device
        if not dev_obj.is_cuda:
            return None

        a_ptr = int(A.__cuda_array_interface__["data"][0])
        # Cache hash combines ``n`` (drives launch dims + n_pad) and the
        # matrix data pointer and strides: a reallocated buffer can return
        # at the same address with a different row pitch.
        key = (n, a_ptr, tuple(A.strides))
        cached = self._pcr_graph_cache.get(key, "missing")
        if cached == "missing":
            # Capture lazily. Failures are sticky: we cache ``None`` so
            # we don't retry capture every solve once it has failed (e.g.
            # if some kernel inside the iter body issues an implicit host
            # allocation that traps the capture).
            try:
                with wp.ScopedCapture(device=device) as cap:
                    self._pcr_iter_eager(A, n, n_pad, device)
                self._pcr_graph_cache = {key: cap.graph}
                return cap.graph
            except RuntimeError:
                self._pcr_graph_cache = {key: None}
                return None
        return cached
