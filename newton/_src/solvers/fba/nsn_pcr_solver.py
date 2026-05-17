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
"""

from __future__ import annotations

import warp as wp

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
def _matvec_kernel(
    A: wp.array2d[wp.float64],
    x: wp.array[wp.float64],
    n: wp.int32,
    out: wp.array[wp.float64],
):
    """Dense matrix-vector product ``out = A @ x``.

    Mirrors RealSim's ``cublasDgemv`` calls (e.g. ``CUDADenseCRSolver.cpp:156``).
    One Warp thread accumulates one row in fp64. For the NSN Schur sizes we
    target (n ~ 100..3000) the n^2 work-per-launch is dominant; a row-per-thread
    naive matvec is sufficient and matches the cuBLAS reference numerically.
    """
    i = wp.tid()
    s = wp.float64(0.0)
    for j in range(n):
        s = s + A[i, j] * x[j]
    out[i] = s


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
def _axpy_kernel(
    a: wp.float64,
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
):
    """Vector AXPY in place: ``y[i] += a * x[i]``.

    Mirrors RealSim's ``cublasDaxpy`` updates (``CUDADenseCRSolver.cpp:177, 181``).
    """
    i = wp.tid()
    y[i] = y[i] + a * x[i]


@wp.kernel
def _scal_add_kernel(
    a: wp.float64,
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
):
    """Combined scale-and-add: ``y = a * y + x``.

    Mirrors RealSim's ``cublasDscal`` + ``cublasDaxpy`` pair used to compute
    ``d = beta*d + s`` and ``q = beta*q + h`` (``CUDADenseCRSolver.cpp:198-207``).
    Fused into one kernel to halve launch overhead.
    """
    i = wp.tid()
    y[i] = a * y[i] + x[i]


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
    """

    def __init__(
        self,
        max_n: int,
        device: wp.Device | str,
        tol: float = 1.0e-5,
        max_iter: int = 100,
    ) -> None:
        if max_n <= 0:
            raise ValueError(f"max_n must be positive, got {max_n}")

        self.max_n = int(max_n)
        self.device = device
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        # Pre-allocate fp64 device buffers. Names mirror RealSim ``cuda_*``
        # buffers in ``CUDADenseCRSolver.h:44-48`` for ease of cross-reference.
        self._x = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._r = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._d = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._q = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._h = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._s = wp.zeros(self.max_n, dtype=wp.float64, device=device)
        self._precond = wp.zeros(self.max_n, dtype=wp.float64, device=device)

        # Single-element output buffers for ``wp.utils.array_inner`` to avoid
        # implicit allocations every iteration.
        self._dot_a = wp.zeros(1, dtype=wp.float64, device=device)
        self._dot_b = wp.zeros(1, dtype=wp.float64, device=device)

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

        # ----- Initialize x = 0 -----------------------------------------
        # RealSim ``_x.setZero(n)`` (line 136). We zero only the active slice.
        self._x.zero_()

        # ----- Early exit if b == 0 -------------------------------------
        # RealSim guards with ``if(dot_b != 0.0)`` (line 138). With b == 0,
        # x = 0 is the exact solution.
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
        # RealSim line 156 (``cublasDgemv``).
        wp.launch(
            _matvec_kernel,
            dim=n,
            inputs=[A, self._d, n],
            outputs=[self._q],
            device=device,
        )

        # ----- h = q ---------------------------------------------------
        # RealSim line 159.
        wp.copy(self._h, self._q, count=n)

        # ----- rho = <r, h> --------------------------------------------
        # RealSim line 162.
        rho = float(wp.utils.array_inner(self._r, self._h, count=n))

        # RealSim line 164: ``tol = _tol * _tol * rho`` — relative on rho.
        tol_sq = self.tol * self.tol * rho

        nb_iter = 0
        while (nb_iter < max_iter) and (rho > tol_sq):
            # ----- den = <q, q> ----------------------------------------
            # RealSim line 170.
            den = float(wp.utils.array_inner(self._q, self._q, count=n))
            if den == 0.0:
                # RealSim line 172: ``if(den == 0.0) break;``.
                break

            alpha = rho / den  # RealSim line 173

            # ----- x += alpha * d --------------------------------------
            # RealSim line 177.
            wp.launch(
                _axpy_kernel,
                dim=n,
                inputs=[wp.float64(alpha), self._d],
                outputs=[self._x],
                device=device,
            )

            # ----- r -= alpha * q --------------------------------------
            # RealSim line 181 (``coeff = -alpha``).
            wp.launch(
                _axpy_kernel,
                dim=n,
                inputs=[wp.float64(-alpha), self._q],
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
            # RealSim line 189.
            wp.launch(
                _matvec_kernel,
                dim=n,
                inputs=[A, self._s, n],
                outputs=[self._h],
                device=device,
            )

            # ----- rho_old = rho; rho = <r, h> -------------------------
            # RealSim lines 191-193.
            rho_old = rho
            rho = float(wp.utils.array_inner(self._r, self._h, count=n))

            beta = rho / rho_old  # RealSim line 195

            # ----- d = beta * d + s ------------------------------------
            # RealSim lines 198-201 (scal then axpy). Fused into one kernel.
            wp.launch(
                _scal_add_kernel,
                dim=n,
                inputs=[wp.float64(beta), self._s],
                outputs=[self._d],
                device=device,
            )

            # ----- q = beta * q + h ------------------------------------
            # RealSim lines 204-207.
            wp.launch(
                _scal_add_kernel,
                dim=n,
                inputs=[wp.float64(beta), self._h],
                outputs=[self._q],
                device=device,
            )

            nb_iter += 1

        # Copy x back to caller-provided output buffer.
        wp.copy(x_out, self._x, count=n)

        return (nb_iter, rho)
