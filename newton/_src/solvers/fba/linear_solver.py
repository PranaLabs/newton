# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cholesky + sparse-inverse linear solver for SolverFBA.

Setup phase (CPU):
    1. assemble scalar N×N PD Hessian `A` (caller-supplied).
    2. factor with `scipy.sparse.linalg.splu` (COLAMD ordering).
    3. compute `S = L⁻¹` keeping elimination-tree sparsity (ported from
       RealSim `LDLT_computeLowerInverse`, SparseLDLT.cpp:225–347).
    4. upload to device as Warp BSR matrices.

Runtime (GPU):
    `A⁻¹ b = Sᵀ · D⁻¹ · (S · b)` — two `bsr_mv` calls plus diagonal scaling.
"""
