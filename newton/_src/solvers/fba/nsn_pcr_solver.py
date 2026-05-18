# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Jacobi-preconditioned Conjugate Residual (PCR) solver for the NSN Schur LCP.

This module ports RealSim's ``CUDADenseJacobiPCRSolver`` to Newton/Warp. The
solver is used inside the Non-Smooth Newton (NSN) outer loop to solve the
dense fp64 SPD reduced (Schur) system on every Newton step.

References (RealSim ``realsim_py``):
    * ``src/Scomponent/linearsolver/iterative/CUDADenseCRSolver.cpp``
      ``solve_vec_gpu``: lines 121-238 — main PCR loop with cuBLAS calls.
    * ``src/Scomponent/linearsolver/iterative/CUDADenseJacobiPCRSolver.cpp``
      ``computePrecond_gpu`` / ``apply_precond_gpu``: lines 34-47 — Jacobi
      (diagonal-inverse) preconditioner.
    * ``include/Scomponent/linearsolver/iterative/CUDADenseJacobiPCRSolver.h``
      lines 13-17 — default constructor ``maxIter=100, tol=1e-5``.

Algorithm (matches RealSim ``CUDADenseCRSolver.cpp:121-238`` line-for-line):

    Initial:
        x_0  = 0                  (RealSim ``_x.setZero(n)`` line 136)
        r    = b                  (RealSim line 142)
        d    = P * r              (RealSim line 147, precond branch)
        q    = A * d              (RealSim line 156)
        h    = q                  (RealSim line 159)
        rho  = <r, h>             (RealSim line 162)
        tol2 = tol^2 * rho_init   (RealSim line 164)

    While rho > tol2 and iter <= max_iter:
        den  = <q, q>             (RealSim line 170)
        alpha= rho / den          (RealSim line 173)
        x   += alpha * d          (RealSim line 177)
        r   -= alpha * q          (RealSim line 181)
        s    = P * r              (RealSim line 186)
        h    = A * s              (RealSim line 189)
        rho_old = rho
        rho  = <r, h>             (RealSim line 193)
        beta = rho / rho_old      (RealSim line 195)
        d    = beta * d + s       (RealSim lines 198-201)
        q    = beta * q + h       (RealSim lines 204-207)

Notes on faithful port:
    * Convergence test is on ``rho`` (preconditioned), not ``||r||`` — this is
      RealSim's exact behaviour (line 166). ``rho`` is compared to
      ``tol^2 * rho_init`` so ``tol`` effectively bounds the
      preconditioned-residual *ratio* relative to the initial value.
    * ``x_0 = 0`` always — the ``warmStart`` flag on the C++ constructor is
      not honoured inside ``solve_vec_gpu`` (line 136). We replicate that.
    * ``max_iter`` is capped at ``n`` to mirror RealSim line 126.

Perf optimization notes (Perf #3, 2026-05):
    * Matvec is now a block-per-row tiled reduction (``_matvec_tiled_kernel``)
      using ``wp.tile_load`` + ``wp.tile_sum``: ~4-5x faster than the prior
      single-thread-per-row serial inner loop at n ~ 600..1500.
    * Dot products use ``wp.utils.array_inner`` (CUB-backed parallel reduction)
      instead of a single-thread accumulator: ~6x faster at n ~ 900.
    * Per-iter host syncs are gone — convergence is checked once at the end
      of the loop after a fixed iter count (still capped by ``rho > tol``).
      Saves 5-10 syncs per PCR call.

Perf optimization notes (Perf #4, 2026-05):
    * When the ``cublas`` extra (``pip install newton[cublas]``) is installed
      *and* the active system size ``n`` exceeds ``_CUBLAS_MIN_N``, the two
      per-iter dense matvecs ``q = A·d`` and ``h = A·s`` dispatch through
      cuBLAS DGEMV via the zero-copy
      :mod:`~newton._src.solvers.fba.cublas_interop` bridge.  cuBLAS DGEMV is
      ~1.5-3.5x faster than ``_matvec_tiled_kernel`` at n ≥ 2000 on RTX 5090
      fp64 (NVIDIA's hand-tuned kernels hit higher SM occupancy at these
      sizes than the in-house tile reduction).
    * We bind directly to ``cublas.dgemv`` via ``cupy_backends.cuda.libs``
      rather than going through ``cupy.matmul`` / ``cupy.cublas.gemv``: the
      higher-level wrappers fall back to a ~7x slower strided-gemv kernel
      when ``A`` is a non-contiguous slice of a larger preallocated buffer,
      which is exactly the FBA shape (the Schur LHS lives in a max_n²
      buffer and only the leading M² block is active).  Setting the
      cuBLAS leading dimension (``lda``) to the parent buffer's row stride
      routes through the fast contiguous DGEMV path.
    * Below the threshold the Python-side dispatch overhead (~10 us per
      call) outweighs the gain, so we stay on ``_matvec_tiled_kernel``.
      An RTX 5090 sweep on 2026-05-18 placed break-even at n ≈ 2000;
      smaller Newton-iter problems (Demo 4, Demo 5 Stage A) stay on
      tile_matmul and only Demo 5 Stage B (n ≈ 3M ≈ 3000) wins.
    * The fallback (pure Warp tile_matmul) path is preserved; FBA works
      identically without cupy installed.
"""

from __future__ import annotations

import warp as wp

from .cublas_interop import cublas_dgemv, get_cublas_handle_for_stream, is_cublas_available

# ---------------------------------------------------------------------------
# Tile / block sizing for ``_matvec_tiled_kernel``
# ---------------------------------------------------------------------------
# K-dim tile width: each block accumulates ``TILE_K`` columns per inner step
# and uses ``wp.tile_sum`` to reduce them in shared memory.  ``128`` is the
# sweet spot for n ~ 600..1500 on the NSN Schur sizes Demo 4/5 produce on an
# RTX 5090 fp64 pipeline (sweep run on 2026-05-18 — see commit log).  Smaller
# tiles win at n > 1500 where occupancy beats per-block re-use; larger tiles
# win at n < 400 where launch overhead dominates.
_PCR_TILE_K = wp.constant(128)
_PCR_BLOCK_DIM = 128

# Minimum system size at which cuBLAS DGEMV starts beating ``_matvec_tiled_kernel``.
# Empirical break-even on an RTX 5090 fp64 (RealSim hardware, 2026-05-18 sweep)
# at the PCR inner: tile_matmul runs at ~1.7 ms/solve for n ≤ 1500 (Python-side
# launch overhead dominates), while cuBLAS DGEMV per call adds a flat ~6 us of
# cupy dispatch.  Net wins start around n ≈ 2000 and the gap widens at larger n
# (tile_matmul scales as O(n²/SM_count); cuBLAS DGEMV is bandwidth-bound and
# scales as O(n²) at a lower constant).  Demo 4's M ≈ 86..258 contact counts
# therefore stay on the tile_matmul path, while Demo 5 Stage B (n = 3M ≈ 3000)
# goes through cuBLAS.  Set to 0 to force cuBLAS for all sizes (use only for
# benchmarking) or to a large value to disable.
_CUBLAS_MIN_N = 2000


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _compute_precond_kernel(
    A: wp.array2d[wp.float64],
    precond: wp.array[wp.float64],
):
    """Compute the Jacobi (diagonal-inverse) preconditioner.

    Faithful to RealSim ``CUDADenseJacobiPCRSolver.cpp:49-67`` and the GPU
    variant ``cudaJacobiPrecondition`` invoked at line 37: ``precond[i] = 1/A[i,i]``.

    A zero diagonal entry is converted to ``0.0`` (rather than ``inf``) so a
    silently-degenerate row produces a zero update instead of NaNs. RealSim's
    CPU fallback (line 56-60) also early-exits and disables the preconditioner
    in that case; we surface the issue at the Python layer through
    ``NSNPCRSolver.solve`` instead.
    """
    i = wp.tid()
    d = A[i, i]
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

    Replaces the prior single-thread row-wise accumulator.  Each block sums
    one row of ``A`` against ``x`` cooperatively: threads load a ``TILE_K``
    strip via ``wp.tile_load`` (bounds-checked, so a partial last tile loads
    zero-padded), element-wise multiply, then reduce with ``wp.tile_sum``.

    Launched via ``wp.launch_tiled`` with ``dim=n`` (one block per active
    row) and ``block_dim=_PCR_BLOCK_DIM``.

    Args:
        A: ``(n, n)`` dense SPD matrix slice (may be a non-contiguous slice
            view; the per-row 1D tile_load on ``A[row]`` handles arbitrary
            row strides).  Entries beyond ``[n, n]`` are never read because
            ``tile_load`` bounds-checks against the slice's ``shape``.
        x: ``(n,)`` input vector slice.  Same bounds-check applies.
        n_pad: ``ceil(n / TILE_K) * TILE_K`` — the inner loop iterates up to
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
        # cross-iter state out of shared memory (a shared-tile re-assign
        # currently breaks the Warp 1.14 register-vs-shared promotion).
        acc_s = acc_s + wp.tile_extract(s, 0)
    # All threads in the block write the same value to ``out[row]``; CUDA
    # coalesces identical writes — equivalent to a single thread emit.
    out[row] = acc_s


@wp.kernel
def _apply_precond_kernel(
    r: wp.array[wp.float64],
    precond: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    """Apply Jacobi preconditioner: ``out = P * r`` (elementwise).

    RealSim ``CUDADenseJacobiPCRSolver.cpp:42-47`` calls the helper
    ``cudaDiagMatrixVectorMultiplication``; this kernel is the direct equivalent.
    """
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
    folds in the ``+/-`` so we can reuse one kernel for both
    ``x += alpha * d`` and ``r -= alpha * q``.  Mirrors RealSim's
    ``cublasDaxpy`` updates (``CUDADenseCRSolver.cpp:177, 181``).
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

    ``y = a_arr[0] * y + x`` — mirrors RealSim's ``cublasDscal`` +
    ``cublasDaxpy`` pair used to compute ``d = beta*d + s`` and ``q = beta*q + h``
    (``CUDADenseCRSolver.cpp:198-207``).  Fused into one kernel to halve
    launch overhead, and reads ``beta`` from device memory so the inner PCR
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

    Used to compute PCR ``alpha = rho / den`` and ``beta = rho / rho_old`` on
    device, avoiding the host round-trip that previously cost two syncs per
    iter.  Guarded against ``den[0] == 0`` by emitting ``0.0`` (the caller's
    convergence check sees ``rho``/``rho_old`` and breaks before re-using a
    stale alpha).
    """
    d = den[0]
    if d == wp.float64(0.0):
        out[0] = wp.float64(0.0)
    else:
        out[0] = num[0] / d


@wp.kernel
def _copy_scalar_kernel(
    src: wp.array[wp.float64],
    dst: wp.array[wp.float64],
):
    """Copy ``dst[0] = src[0]`` on device (used to snapshot ``rho_old``)."""
    dst[0] = src[0]


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class NSNPCRSolver:
    """Jacobi-preconditioned Conjugate Residual solver for SPD systems.

    Ports RealSim ``CUDADenseJacobiPCRSolver`` (``CUDADenseCRSolver.cpp:121-238``)
    line-for-line. Used as the inner linear solver for the NSN Schur LCP system.

    The solver is *stateless* between ``solve`` calls (the C++ class is too —
    ``solve_vec_gpu`` always re-initializes ``x = 0``). Device buffers are
    pre-allocated for ``max_n`` so repeated calls at smaller sizes do not
    allocate.

    Args:
        max_n: Maximum system dimension the solver will handle. Internal
            buffers are sized to this.
        device: Warp device for buffer allocation (typically ``"cuda:0"``).
        tol: Relative tolerance on the *preconditioned* residual ``rho``.
            RealSim default is ``1e-5``
            (``CUDADenseJacobiPCRSolver.h:13``); the convergence test is
            ``rho < tol^2 * rho_init``.
        max_iter: Outer-loop cap. Effective limit is ``min(max_iter, n)``
            per RealSim ``CUDADenseCRSolver.cpp:126``.
        check_every: Convergence-check batching factor.  PCR runs this many
            iters fully asynchronously on the device before pulling ``rho``
            to host to test for convergence.  The default ``25`` matches the
            empirical iter count of the NSN Schur LCP on Demo 4/5 — a single
            sync per ``solve()`` call covers the typical run.
        use_cublas: Whether to dispatch the per-iter dense matvec through
            cuBLAS DGEMV (Perf #4).  Defaults to ``True`` when the ``cublas``
            extra is installed and ``False`` otherwise.  cuBLAS is only used
            when the active ``n`` per call exceeds ``_CUBLAS_MIN_N`` (~2000);
            below that the tile_matmul kernel is faster due to Python-side
            dispatch overhead.  Setting to ``False`` forces the Warp
            ``tile_matmul`` fallback for all sizes (useful for benchmarking).
    """

    def __init__(
        self,
        max_n: int,
        device: wp.Device | str,
        tol: float = 1.0e-5,
        max_iter: int = 100,
        check_every: int = 25,
        use_cublas: bool | None = None,
    ) -> None:
        if max_n <= 0:
            raise ValueError(f"max_n must be positive, got {max_n}")

        self.max_n = int(max_n)
        self.device = device
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        # Auto-enable cuBLAS when the optional ``cublas`` extra is installed.
        # ``use_cublas=True`` with no cupy raises so misconfigured environ-
        # ments fail loudly instead of silently falling back.  Even when
        # ``self._use_cublas`` is ``True`` the per-call gate ``n >=
        # _CUBLAS_MIN_N`` may still route a particular solve through the
        # tile_matmul kernel; the construction-time flag only enables the
        # cupy buffer caching below.
        if use_cublas is None:
            self._use_cublas = is_cublas_available()
        elif use_cublas and not is_cublas_available():
            raise RuntimeError("use_cublas=True requires cupy. Install with `pip install newton[cublas]`.")
        else:
            self._use_cublas = bool(use_cublas)
        # Bottleneck #2 fix: rather than syncing ``rho`` every iter (which
        # forces a CPU stall after every PCR step), only test convergence
        # every ``check_every`` iters.  Iterations between checks run fully
        # asynchronously on the device.  Worst-case we overshoot the tol by
        # ``check_every - 1`` iters; PCR's residual decay is monotone for SPD
        # systems so this never compromises correctness.  RealSim itself
        # syncs every iter via ``cublasDdot``; for the NSN inner this is the
        # dominant per-step overhead at large n.  Default raised from 5 to
        # 25 (Perf #3) — the NSN inner converges in ≤25 iters in practice so
        # this is effectively one host sync per PCR call.
        self.check_every = max(1, int(check_every))

        # Pre-allocate fp64 device buffers. Names mirror RealSim ``cuda_*``
        # buffers in ``CUDADenseCRSolver.h:44-48`` for ease of cross-reference.
        self._x = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._r = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._d = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._q = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._h = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._s = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._precond = wp.zeros(self.max_n, dtype=wp.float64, device=device)

        # Device-resident scalars for ``rho``, ``rho_old``, ``den``, ``alpha``,
        # ``beta`` — keep the per-iter PCR coefficients on device so the inner
        # loop never blocks on a host transfer.  ``rho`` is synced exactly
        # once per ``solve()`` call at the end.
        self._rho_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._rho_old_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._den_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._alpha_d = wp.zeros(1, dtype=wp.float64, device=device)
        self._beta_d = wp.zeros(1, dtype=wp.float64, device=device)

        # Bind cuBLAS to Warp's per-device stream once at construction.  All
        # subsequent ``cublas_dgemv`` calls inherit Warp's stream ordering at
        # zero per-call cost (no stream-context overhead, no ``cp.asarray``
        # round trips on the vector buffers).  We hold the handle as a plain
        # int so the inner loop avoids any Python attribute lookup.
        self._cublas_handle: int | None = None
        if self._use_cublas:
            self._cublas_handle = get_cublas_handle_for_stream(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        A: wp.array2d[wp.float64],
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
    ) -> tuple[int, float]:
        """Solve ``A @ x = b`` with Jacobi-preconditioned conjugate residual.

        Args:
            A: ``(n, n)`` dense SPD matrix, device-resident, fp64. ``n`` must
                satisfy ``n <= max_n``.
            b: ``(n,)`` right-hand side, device-resident, fp64.
            x_out: ``(n,)`` output buffer, device-resident, fp64. Filled with
                the solution on return.

        Returns:
            A ``(iterations_taken, final_rho)`` tuple where ``iterations_taken``
            counts the number of inner PCR iterations executed (matches
            RealSim's ``_nb_iter`` accounting at ``CUDADenseCRSolver.cpp:127``,
            ``234``) and ``final_rho`` is the last value of the preconditioned
            residual scalar ``rho = <r, h>`` (>= 0 since ``h`` mixes with ``r``
            through the SPD preconditioner).

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
        max_iter = min(self.max_iter, n)  # RealSim CUDADenseCRSolver.cpp:126
        tile_k = int(_PCR_TILE_K)
        n_pad = ((n + tile_k - 1) // tile_k) * tile_k

        # ----- cuBLAS dispatch decision ---------------------------------
        # Route through cuBLAS DGEMV when:
        #   1. cupy is installed (``self._use_cublas == True``), and
        #   2. ``n`` exceeds the empirical break-even ``_CUBLAS_MIN_N``.
        # Below the threshold, ``cublas_dgemv``'s ~10 us of Python-side
        # dispatch + ~3 us cuBLAS host launch is more than the entire
        # ``_matvec_tiled_kernel`` cost (~7-12 us at n ≤ 1500), so the
        # tile_matmul kernel wins.  Above it, cuBLAS DGEMV's ~28 us at
        # n ~ 3000 outperforms tile_matmul's ~90 us.
        use_cublas_this_solve = self._use_cublas and n >= _CUBLAS_MIN_N

        # ----- Initialize x = 0 -----------------------------------------
        # RealSim ``_x.setZero(n)`` (line 136). We zero only the active slice.
        self._x.zero_()

        # ----- Early exit if b == 0 -------------------------------------
        # RealSim guards with ``if(dot_b != 0.0)`` (line 138). With b == 0,
        # x = 0 is the exact solution.  ``array_inner`` is a CUB-style
        # parallel reduction — far cheaper than the prior single-thread
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

        # ----- r = b ----------------------------------------------------
        # RealSim line 142 (``cublasDcopy(b, r)``).
        wp.copy(self._r, b, count=n)

        # ----- d = P * r -----------------------------------------------
        # RealSim line 147.
        wp.launch(
            _apply_precond_kernel,
            dim=n,
            inputs=[self._r, self._precond],
            outputs=[self._d],
            device=device,
        )

        # ----- q = A * d -----------------------------------------------
        # RealSim line 156 (``cublasDgemv``).  Two paths:
        #   - n >= _CUBLAS_MIN_N and cupy installed: direct cuBLAS DGEMV via
        #     :func:`cublas_dgemv` (bypasses cupy.matmul's slow strided
        #     dispatch).  Matches RealSim's literal cuBLAS call.
        #   - Otherwise: block-per-row tile reduction with bounds-checked
        #     ``tile_load`` so OOB elements load zero (last partial K-tile).
        if use_cublas_this_solve:
            cublas_dgemv(A, self._d, self._q, device=device, handle=self._cublas_handle)
        else:
            wp.launch_tiled(
                _matvec_tiled_kernel,
                dim=n,
                inputs=[A, self._d, n_pad],
                outputs=[self._q],
                block_dim=_PCR_BLOCK_DIM,
                device=device,
            )

        # ----- h = q ---------------------------------------------------
        # RealSim line 159.
        wp.copy(self._h, self._q, count=n)

        # ----- rho = <r, h> --------------------------------------------
        # RealSim line 162.  ``array_inner`` writes directly to the device-
        # resident scalar — no host round-trip and the result feeds the next
        # iter's ``_div_scalar_kernel``.
        wp.utils.array_inner(self._r, self._h, out=self._rho_d, count=n)
        rho = float(self._rho_d.numpy()[0])

        # RealSim line 164: ``tol = _tol * _tol * rho`` — relative on rho.
        tol_sq = self.tol * self.tol * rho

        # ---- PCR main loop --------------------------------------------
        # Inner loop runs in ``check_every``-sized blocks fully on the device,
        # then syncs ``rho`` once per block to test convergence.  All per-iter
        # scalars (``alpha``, ``beta``, ``rho``, ``den``) live in device
        # memory.  PCR for SPD systems is monotone-convergent, so over-
        # iterating past the tol bound between syncs is safe.
        nb_iter = 0
        check_every = self.check_every
        while nb_iter < max_iter and rho > tol_sq:
            block_end = min(nb_iter + check_every, max_iter)
            while nb_iter < block_end:
                # ----- den = <q, q> ----------------------------------------
                # RealSim line 170.
                wp.utils.array_inner(self._q, self._q, out=self._den_d, count=n)

                # alpha = rho / den (on device).  RealSim line 173.
                wp.launch(
                    _div_scalar_kernel,
                    dim=1,
                    inputs=[self._rho_d, self._den_d],
                    outputs=[self._alpha_d],
                    device=device,
                )

                # ----- x += alpha * d --------------------------------------
                # RealSim line 177.  Coefficient sourced from ``_alpha_d[0]``.
                wp.launch(
                    _axpy_arr_kernel,
                    dim=n,
                    inputs=[self._alpha_d, wp.float64(1.0), self._d],
                    outputs=[self._x],
                    device=device,
                )

                # ----- r -= alpha * q --------------------------------------
                # RealSim line 181 (``coeff = -alpha``).
                wp.launch(
                    _axpy_arr_kernel,
                    dim=n,
                    inputs=[self._alpha_d, wp.float64(-1.0), self._q],
                    outputs=[self._r],
                    device=device,
                )

                # ----- s = P * r -------------------------------------------
                # RealSim line 186.
                wp.launch(
                    _apply_precond_kernel,
                    dim=n,
                    inputs=[self._r, self._precond],
                    outputs=[self._s],
                    device=device,
                )

                # ----- h = A * s -------------------------------------------
                # RealSim line 189.  See ``q = A * d`` above for the cuBLAS
                # vs tile_matmul dispatch rationale.
                if use_cublas_this_solve:
                    cublas_dgemv(A, self._s, self._h, device=device, handle=self._cublas_handle)
                else:
                    wp.launch_tiled(
                        _matvec_tiled_kernel,
                        dim=n,
                        inputs=[A, self._s, n_pad],
                        outputs=[self._h],
                        block_dim=_PCR_BLOCK_DIM,
                        device=device,
                    )

                # ----- rho_old = rho ; rho = <r, h> ------------------------
                # RealSim lines 191-193.  Snapshot ``rho`` into ``rho_old`` on
                # device, then recompute ``rho`` via the parallel reduction.
                wp.launch(
                    _copy_scalar_kernel,
                    dim=1,
                    inputs=[self._rho_d],
                    outputs=[self._rho_old_d],
                    device=device,
                )
                wp.utils.array_inner(self._r, self._h, out=self._rho_d, count=n)

                # beta = rho / rho_old (on device).  RealSim line 195.
                wp.launch(
                    _div_scalar_kernel,
                    dim=1,
                    inputs=[self._rho_d, self._rho_old_d],
                    outputs=[self._beta_d],
                    device=device,
                )

                # ----- d = beta * d + s ------------------------------------
                # RealSim lines 198-201.  Fused into one kernel.
                wp.launch(
                    _scal_add_arr_kernel,
                    dim=n,
                    inputs=[self._beta_d, self._s],
                    outputs=[self._d],
                    device=device,
                )

                # ----- q = beta * q + h ------------------------------------
                # RealSim lines 204-207.
                wp.launch(
                    _scal_add_arr_kernel,
                    dim=n,
                    inputs=[self._beta_d, self._h],
                    outputs=[self._q],
                    device=device,
                )

                nb_iter += 1

            # End-of-block convergence sync: pull rho once per ``check_every``
            # iters.  RealSim's CPU-side break on ``den == 0`` (line 172) is
            # not reproduced here because ``_div_scalar_kernel`` emits 0 for
            # a zero den and the next ``rho`` computation will not improve,
            # so the relative test below will keep ``rho`` constant and the
            # loop exits via ``max_iter`` instead.  For SPD A with well-
            # formed preconditioner this branch is unreachable in practice.
            rho = float(self._rho_d.numpy()[0])

        # Copy x back to caller-provided output buffer.
        wp.copy(x_out, self._x, count=n)

        return (nb_iter, rho)
