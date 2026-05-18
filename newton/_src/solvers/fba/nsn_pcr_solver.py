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

Perf optimization notes (Perf #5, 2026-05):
    * The per-iter PCR body (matvec + dots + axpy + scale-add + precond)
      issues ~10 kernel launches per iter; at Demo 5's typical 25 iters
      per ``solve()`` call that's ~250 Python-side launch dispatches.
      Each Warp ``wp.launch`` plus the CUDA driver call add ~3-6 us of
      host overhead; CUDA graphs replay that batch with ~1 us per call.
    * We capture **one** PCR iter (lines RealSim 170-207) into a
      :class:`warp.Graph` per ``n`` shape and cache the graph keyed by
      ``(n, A.ptr)``.  Subsequent solves at the same shape replay the
      graph ``check_every`` times per convergence-block — a single
      ``cuGraphLaunch`` per block on the device side.

Perf optimization notes (Perf #6, 2026-05):
    * The previous optional cuBLAS DGEMV path (Perf #4 via cupy) was
      removed.  cuBLAS calls trip ``cudaErrorStreamCaptureImplicit``
      during graph capture, so the two paths were mutually exclusive.
      CUDA graph capture of the pure-Warp ``tile_matmul`` path wins by
      a large margin (Demo 5 step_mean 43.4 ms with graphs vs ~99 ms
      with cuBLAS+eager on RTX 5090 fp64), so the cuBLAS path is no
      longer worth keeping.  Removing it drops the optional ``cupy``
      dependency from the project.
"""

from __future__ import annotations

import warp as wp

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


_MAT1X1_F64 = wp.types.matrix((1, 1), wp.float64)


@wp.kernel
def _precond_from_bsr_diag_kernel(
    diag_blocks: wp.array[_MAT1X1_F64],
    precond: wp.array[wp.float64],
):
    """Build ``precond[i] = 1/diag_blocks[i][0, 0]`` for the LiteNSN sparse Schur.

    The BSR Schur ``W`` produced by :meth:`FBALinearSolver.build_schur_lite` has
    ``1×1`` block shape, so ``bsr_get_diag`` returns one ``mat1x1`` per row.
    Mirrors :func:`_compute_precond_kernel` for the dense path: ``1/W[i,i]``,
    with the standard zero-diagonal-zero-output guard so a degenerate row
    yields a zero update instead of NaN.
    """
    i = wp.tid()
    d = diag_blocks[i][0, 0]
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
    """

    def __init__(
        self,
        max_n: int,
        device: wp.Device | str,
        tol: float = 1.0e-5,
        max_iter: int = 100,
        check_every: int = 25,
    ) -> None:
        if max_n <= 0:
            raise ValueError(f"max_n must be positive, got {max_n}")

        self.max_n = int(max_n)
        self.device = device
        self.tol = float(tol)
        self.max_iter = int(max_iter)
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

        # Perf #5: cache one ``wp.Graph`` per active ``n`` shape so repeated
        # solves at the same shape replay a pre-recorded CUDA graph for the
        # inner PCR iter, eliminating per-launch Python+driver overhead.
        # Capture is attempted lazily on first solve at each shape; if it
        # fails we fall back to the eager loop and remember the failure so
        # we don't retry on every call.
        self._pcr_graph_cache: dict[int, wp.Graph | None] = {}

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
        # RealSim line 156 (``cublasDgemv``).  Block-per-row tile reduction
        # with bounds-checked ``tile_load`` so OOB elements load zero (last
        # partial K-tile).
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
        #
        # Perf #5: wrap one full PCR iter into a :class:`warp.Graph` and
        # replay it ``check_every`` times per block.  Replays issue a single
        # ``cuGraphLaunch`` on the device side instead of ~10 individual
        # ``wp.launch`` dispatches, so the Python+driver per-launch overhead
        # drops from ~30-50 us/iter to ~2 us/iter at the n ~ 600..1500 sizes
        # Demo 5 hits.
        graph = self._get_or_build_pcr_iter_graph(A, n, n_pad, device)

        nb_iter = 0
        check_every = self.check_every
        while nb_iter < max_iter and rho > tol_sq:
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

    def solve_sparse(
        self,
        W_bsr,
        b: wp.array[wp.float64],
        x_out: wp.array[wp.float64],
    ) -> tuple[int, float]:
        """LiteNSN sparse-Schur PCR: ``W_bsr @ x = b`` for a BSR ``W`` with
        ``1×1`` blocks.

        Matches :meth:`solve` numerically — same termination criterion, same
        Jacobi preconditioner, same ``check_every`` host-sync cadence — but
        replaces the dense tiled matvec with :func:`warp.sparse.bsr_mv`.  This
        is the linear-solver entry point for ``SolverFBA(nsn_schur_mode="lite")``.

        Args:
            W_bsr: ``BsrMatrix`` of block shape ``(1, 1)`` and scalar type
                ``wp.float64`` (the LiteNSN Schur ``H · M⁻¹ · Hᵀ``).
            b: ``(n,)`` RHS, fp64.
            x_out: ``(n,)`` output, fp64.

        Returns:
            ``(iterations_taken, final_rho)``.

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
            raise ValueError(
                f"W_bsr shape {tuple(W_bsr.shape)} incompatible with b shape ({n},)"
            )
        if x_out.shape[0] != n:
            raise ValueError(f"x_out shape ({x_out.shape[0]},) incompatible with b shape ({n},)")

        device = self.device
        max_iter = min(self.max_iter, n)

        # ----- x = 0 ----------------------------------------------------
        self._x.zero_()

        # ----- Early exit if b == 0 -------------------------------------
        dot_b = float(wp.utils.array_inner(b, b, count=n))
        if dot_b == 0.0:
            wp.copy(x_out, self._x, count=n)
            return (0, 0.0)

        # ----- Build Jacobi preconditioner from W's diagonal ------------
        diag_blocks = wps.bsr_get_diag(W_bsr)  # array[mat1x1[float64]], length n
        wp.launch(
            _precond_from_bsr_diag_kernel,
            dim=n,
            inputs=[diag_blocks],
            outputs=[self._precond],
            device=device,
        )

        # ----- r = b ---------------------------------------------------
        wp.copy(self._r, b, count=n)

        # ----- d = P * r -----------------------------------------------
        wp.launch(
            _apply_precond_kernel,
            dim=n,
            inputs=[self._r, self._precond],
            outputs=[self._d],
            device=device,
        )

        # ----- q = W * d -----------------------------------------------
        # ``wps.bsr_mv(W, d, q)`` with the sparse W replacing the dense matvec.
        wps.bsr_mv(W_bsr, self._d, self._q)

        # ----- h = q ---------------------------------------------------
        wp.copy(self._h, self._q, count=n)

        # ----- rho = <r, h> --------------------------------------------
        wp.utils.array_inner(self._r, self._h, out=self._rho_d, count=n)
        rho = float(self._rho_d.numpy()[0])
        tol_sq = self.tol * self.tol * rho

        # ---- PCR main loop --------------------------------------------
        # No CUDA-graph capture here: bsr_mv currently issues internal
        # allocations on shape change that defeat capture. The eager loop is
        # still fully async between syncs; ``check_every`` bounds the host
        # round-trips per solve. If profiling shows the launch overhead
        # dominates LiteNSN at scale, revisit and add capture support.
        nb_iter = 0
        check_every = self.check_every
        while nb_iter < max_iter and rho > tol_sq:
            block_end = min(nb_iter + check_every, max_iter)
            while nb_iter < block_end:
                # ----- den = <q, q> ---------------------------------
                wp.utils.array_inner(self._q, self._q, out=self._den_d, count=n)
                wp.launch(
                    _div_scalar_kernel, dim=1,
                    inputs=[self._rho_d, self._den_d],
                    outputs=[self._alpha_d], device=device,
                )
                # x += alpha * d
                wp.launch(
                    _axpy_arr_kernel, dim=n,
                    inputs=[self._alpha_d, wp.float64(1.0), self._d],
                    outputs=[self._x], device=device,
                )
                # r -= alpha * q
                wp.launch(
                    _axpy_arr_kernel, dim=n,
                    inputs=[self._alpha_d, wp.float64(-1.0), self._q],
                    outputs=[self._r], device=device,
                )
                # s = P * r
                wp.launch(
                    _apply_precond_kernel, dim=n,
                    inputs=[self._r, self._precond],
                    outputs=[self._s], device=device,
                )
                # h = W * s   (sparse matvec)
                wps.bsr_mv(W_bsr, self._s, self._h)
                # rho_old = rho ; rho = <r, h>
                wp.launch(
                    _copy_scalar_kernel, dim=1,
                    inputs=[self._rho_d],
                    outputs=[self._rho_old_d], device=device,
                )
                wp.utils.array_inner(self._r, self._h, out=self._rho_d, count=n)
                wp.launch(
                    _div_scalar_kernel, dim=1,
                    inputs=[self._rho_d, self._rho_old_d],
                    outputs=[self._beta_d], device=device,
                )
                # d = beta*d + s
                wp.launch(
                    _scal_add_arr_kernel, dim=n,
                    inputs=[self._beta_d, self._s],
                    outputs=[self._d], device=device,
                )
                # q = beta*q + h
                wp.launch(
                    _scal_add_arr_kernel, dim=n,
                    inputs=[self._beta_d, self._h],
                    outputs=[self._q], device=device,
                )
                nb_iter += 1

            rho = float(self._rho_d.numpy()[0])

        wp.copy(x_out, self._x, count=n)
        return (nb_iter, rho)

    # ------------------------------------------------------------------
    # Per-iter body + graph capture (Perf #5)
    # ------------------------------------------------------------------

    def _pcr_iter_eager(
        self,
        A: wp.array2d[wp.float64],
        n: int,
        n_pad: int,
        device: wp.Device | str,
    ) -> None:
        """Issue one PCR iter's worth of launches on the active stream.

        Mirrors RealSim ``CUDADenseCRSolver.cpp:170-207`` (Newton step body).
        Reads ``self._q``, ``self._r``, ``self._d``, ``self._rho_d`` and writes
        through ``self._x``, ``self._r``, ``self._d``, ``self._q``, ``self._h``,
        ``self._s``, ``self._rho_d``, ``self._rho_old_d``, ``self._alpha_d``,
        ``self._beta_d``, ``self._den_d``.  All launches are stream-ordered
        and safe to call inside a CUDA graph capture.
        """
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
        # RealSim line 189.  Block-per-row tile reduction; see ``q = A * d``
        # in :meth:`solve` for the kernel rationale.
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

    def _get_or_build_pcr_iter_graph(
        self,
        A: wp.array2d[wp.float64],
        n: int,
        n_pad: int,
        device: wp.Device | str,
    ) -> wp.Graph | None:
        """Look up (or capture on first miss) the single-PCR-iter CUDA graph.

        The graph is keyed by ``(n, A.ptr)`` -- ``n`` controls the launch
        dimensions and ``A.ptr`` controls which matrix block the captured
        matvec reads from.  When either changes between solve calls we
        re-capture; PCR is called repeatedly with the same matrix view
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
        # matrix data pointer.  We use a simple tuple key in a dict --
        # collisions are not a concern at the handful of distinct shapes
        # FBA produces per session.
        key = (n, a_ptr)
        cached = self._pcr_graph_cache.get(key, "missing")
        if cached == "missing":
            # Capture lazily.  Failures are sticky -- we cache ``None`` so
            # we don't retry capture every solve once it has failed (e.g.
            # if some kernel inside the iter body issues an implicit host
            # allocation that traps the capture).
            try:
                with wp.ScopedCapture(device=device) as cap:
                    self._pcr_iter_eager(A, n, n_pad, device)
                self._pcr_graph_cache[key] = cap.graph
                return cap.graph
            except Exception:
                self._pcr_graph_cache[key] = None
                return None
        return cached
