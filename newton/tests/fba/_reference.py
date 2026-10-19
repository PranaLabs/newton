# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host reference implementations that the SolverFBA tests compare the solver against."""

import math

import numpy as np


def compute_tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute two orthonormal tangent vectors ``(t1, t2)`` from a unit normal.

    Pick a reference direction (world X, or world Y if the normal is nearly X),
    cross the normal with it to get t1, then the normal with t1 to get t2.

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


def quat_rotate_z_axis(q: np.ndarray) -> np.ndarray:
    """Apply quaternion ``q = (qx, qy, qz, qw)`` to the local +Z unit vector.

    Returns the world-frame direction of a shape's local +Z, used as the spin
    axis for Newton's cylinder primitive (which extends along local +Z).
    """
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dtype=np.float64,
    )


def transform_point(pos: np.ndarray, quat: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a rigid transform ``T = (pos, quat)`` to point ``p``.

    Warp stores transforms as ``[px, py, pz, qx, qy, qz, qw]``.

    Args:
        pos: Translation [m], shape ``(3,)``.
        quat: Quaternion ``[qx, qy, qz, qw]``.
        p: Point to transform, shape ``(3,)``.

    Returns:
        Transformed point, shape ``(3,)``.
    """
    qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    # Rotate p by quaternion then add translation.
    # Formula: q * p * q_conj where q = (qx, qy, qz, qw).
    # Using Rodrigues formula: v' = v + 2*qw*(q_vec x v) + 2*(q_vec x (q_vec x v))
    tx = 2.0 * (qy * pz - qz * py)
    ty = 2.0 * (qz * px - qx * pz)
    tz = 2.0 * (qx * py - qy * px)
    rx = px + qw * tx + (qy * tz - qz * ty)
    ry = py + qw * ty + (qz * tx - qx * tz)
    rz = pz + qw * tz + (qx * ty - qy * tx)
    return np.array([rx + float(pos[0]), ry + float(pos[1]), rz + float(pos[2])], dtype=np.float64)


def fb_unilateral_row(
    penetration: float,
    lam: float,
    precond: float,
    dt: float,
    pene0: float,
) -> tuple[float, float, float]:
    """Fischer-Burmeister evaluation for a unilateral (normal) contact row.

    Returns ``(omega, compliance, h)``.

    Args:
        penetration: ``J·q - pene0`` at current iterate (m).
        lam: Current normal lambda (N).
        precond: ``1 / W_ii`` (1/N).
        dt: Timestep (s).
        pene0: ``n·anchor`` (m).

    Returns:
        ``(omega, compliance, h)``:
        - ``omega``: per-row weighting ``1 - pene/√(pene² + (precond·lam)²)``.
          May exceed 1 in deep penetration (penetration < 0); intentional.
        - ``compliance``: diagonal regularization for the Schur LHS.
        - ``h``: row contribution to the Schur RHS,
          ``-(pene + precond·lam - root) + omega · (penetration + pene0)``.
    """
    pene = penetration
    plam = precond * lam
    root = math.sqrt(pene * pene + plam * plam)
    if root < 1e-30:
        # The residual vanishes at the origin, so the row takes no Newton step.
        return 0.0, precond / (dt * dt), 0.0
    omega = 1.0 - pene / root
    compliance = (1.0 - plam / root) * (precond / (dt * dt))
    h = -(pene + plam - root) + omega * (penetration + pene0)
    return omega, compliance, h


def fb_frictional_row(
    penetration: float,
    lam_t: float,
    lam_n: float,
    mu: float,
    precond: float,
    dt: float,
    pene0: float,
) -> tuple[float, float, float]:
    """Fischer-Burmeister evaluation for a frictional (tangent) contact row.

    Returns ``(omega, compliance, h)``.

    Inactive contact (``lam_n ≤ 0``): ``omega = 0``, ``compliance = 1/dt``,
    ``h = -dt · lam_t``.

    Active contact (``lam_n > 0``): smooth complementarity between slip speed
    ``|penetration|/dt`` and cone slack ``μ·lam_n - |lam_t|``. ``omega = 1``,
    compliance varies between near-zero (stick) and ``~1/dt`` (slip).
    """
    if lam_n <= 0.0:
        return 0.0, 1.0 / dt, -dt * lam_t

    abspenevel = abs(penetration / dt)
    tmp = precond * (mu * lam_n - abs(lam_t))
    root = math.sqrt(abspenevel * abspenevel + tmp * tmp)
    denom = abspenevel + mu * precond * lam_n - root
    if abs(denom) < 1e-30:
        compliance = 1.0 / dt
    else:
        compliance = ((root - tmp) / denom) * (precond / dt)
    h = -(dt * dt) * compliance * lam_t + pene0
    return 1.0, compliance, h


def solve_nsn_unilateral(
    W: np.ndarray,
    r: np.ndarray,
    pene0: np.ndarray,
    max_iters: int = 1,
    lam_init: np.ndarray | None = None,
    omega_init: np.ndarray | None = None,
    dt: float = 0.01,
    lambda_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host NSN driver for unilateral (frictionless) rows, the reference of the GPU driver.

    Returns ``(lam, omega, lam_apply)``:
      - ``lam`` (force units, for warm-start): the λ accumulated over the iterations.
      - ``omega``: per-row weighting from the last NSN iter.
      - ``lam_apply`` = ``dt² · ω · lam`` (position-LCP units): the value to
        pass into :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.apply_lambda_correction_combined`
        (which adds ``A⁻¹·Jᵀ·lam_apply`` to ``x_unc``).

    The two scales differ because the correction is
    ``Δq = dt² · A⁻¹ · Jᵀ · (ω · λ_force)``, whereas
    ``apply_lambda_correction_combined`` adds ``A⁻¹·Jᵀ·λ`` directly. ``omega``
    from the previous step is needed so that the next call can compute the
    correct iter-0 ``penetration = -r + dt²·W·(ω·λ)`` accounting for the
    previously-applied correction.
    """
    M = len(r)
    if M == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    if lam_init is not None and lam_init.shape == (M,):
        lam = lam_init.astype(np.float64, copy=True)
    else:
        lam = np.zeros(M, dtype=np.float64)
    if omega_init is not None and omega_init.shape == (M,):
        omega = omega_init.astype(np.float64, copy=True)
    else:
        omega = np.zeros(M, dtype=np.float64)

    diag_W = np.diag(W)
    # Preconditioner: precond[i] = dt² · W_ii for unilateral rows.
    precond = (dt * dt) * np.where(np.abs(diag_W) > 1e-12, np.maximum(diag_W, 1e-12), 1.0)

    for _ in range(max_iters):
        # penetration = -r + dt²·W·(ω·λ). With warm-start ω·λ this reflects
        # the gap at the position currently corrected by the previous iter.
        penetration = -r + (dt * dt) * (W @ (omega * lam))
        omega = np.zeros(M, dtype=np.float64)
        compliance = np.zeros(M, dtype=np.float64)
        h = np.zeros(M, dtype=np.float64)
        for c in range(M):
            om_n, cn, hn = fb_unilateral_row(
                penetration=penetration[c],
                lam=lam[c],
                precond=precond[c],
                dt=dt,
                pene0=pene0[c],
            )
            omega[c] = om_n
            compliance[c] = cn
            h[c] = hn

        A_schur = (omega[:, None] * omega[None, :]) * W + np.diag(compliance)
        # J·x_corrected = J·x_unc + dt²·W·(ω·λ) = (pene0 - r) + dt²·W·(ω·λ).
        J_x = (pene0 - r) + (dt * dt) * (W @ (omega * lam))
        rhs = (1.0 / (dt * dt)) * (h - omega * J_x)

        try:
            dlam = np.linalg.solve(A_schur, rhs)
        except np.linalg.LinAlgError:
            break
        lam = lam + dlam
        # No lam_n >= 0 clamp here: FB-Newton allows a transiently negative
        # lambda inside the inner solve and lets the FB residual converge. A
        # np.maximum(lam, 0.0) clamp would turn the Newton iteration into a
        # projected one.

    if lambda_cap is not None:
        cap_internal = lambda_cap / (dt * dt)
        np.clip(lam, -cap_internal, cap_internal, out=lam)
    lam_apply = (dt * dt) * omega * lam
    return lam, omega, lam_apply


def solve_nsn_coulomb(
    W: np.ndarray,
    r: np.ndarray,
    mu: np.ndarray,
    pene0: np.ndarray,
    max_iters: int = 1,
    lam_init: np.ndarray | None = None,
    omega_init: np.ndarray | None = None,
    dt: float = 0.01,
    lambda_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host NSN driver for frictional (Coulomb) rows, the reference of the GPU driver.

    Takes semismooth Newton steps ``dλ`` on the Fischer-Burmeister rows of
    :func:`fb_unilateral_row` and :func:`fb_frictional_row`, with omega
    weighting and position-LCP coupling. λ accumulates across ``max_iters``
    iterations.

    Args:
        W: ``(3M, 3M)`` Schur complement.
        r: ``(3M,)`` residual ``pene0 - alpha·J·x_unc``.
        mu: ``(M,)`` per-contact friction coefficient.
        pene0: ``(3M,)`` per-row anchor projection. Constant within step.
        max_iters: NSN iterations (default 1).
        lam_init: Optional warm-start ``(3M,)`` lambda.
        omega_init: Optional warm-start ``(3M,)`` omega from previous call.
        dt: Timestep (s).

    Returns:
        ``(lam, omega, lam_apply)``:
          - ``lam`` (force units, ``(3M,)``): the λ accumulated over the
            iterations, with layout ``[λ_n_0, λ_t1_0, λ_t2_0, ...]``. Use as warm-start.
          - ``omega``: per-row weighting from the last NSN iter.
          - ``lam_apply`` = ``dt² · ω · lam``: pass into
            :meth:`~newton._src.solvers.fba.linear_solver.FBALinearSolver.apply_lambda_correction_combined`.
    """
    M = len(mu)
    if M == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    size_total = 3 * M

    if lam_init is not None and lam_init.shape == (size_total,):
        lam = lam_init.astype(np.float64, copy=True)
    else:
        lam = np.zeros(size_total, dtype=np.float64)
    if omega_init is not None and omega_init.shape == (size_total,):
        omega = omega_init.astype(np.float64, copy=True)
    else:
        omega = np.zeros(size_total, dtype=np.float64)

    diag_W = np.diag(W)
    # Preconditioner: precond[3c]   = dt² · W_ii  (unilateral row)
    #                 precond[3c+1] = dt   · W_ii  (tangent rows)
    #                 precond[3c+2] = dt   · W_ii
    precond = np.zeros(size_total, dtype=np.float64)
    for c in range(M):
        w_n = max(float(abs(diag_W[3 * c])), 1e-12)
        w_t1 = max(float(abs(diag_W[3 * c + 1])), 1e-12)
        w_t2 = max(float(abs(diag_W[3 * c + 2])), 1e-12)
        precond[3 * c] = (dt * dt) * w_n
        precond[3 * c + 1] = dt * w_t1
        precond[3 * c + 2] = dt * w_t2

    for _ in range(max_iters):
        # penetration = -r + dt²·W·(ω·λ); on iter 0 with ω=0 this is -r.
        penetration = -r + (dt * dt) * (W @ (omega * lam))

        omega = np.zeros(size_total, dtype=np.float64)
        compliance = np.zeros(size_total, dtype=np.float64)
        h = np.zeros(size_total, dtype=np.float64)
        for c in range(M):
            idx_n = 3 * c
            idx_t1 = 3 * c + 1
            idx_t2 = 3 * c + 2
            om_n, cn, hn = fb_unilateral_row(
                penetration=penetration[idx_n],
                lam=lam[idx_n],
                precond=precond[idx_n],
                dt=dt,
                pene0=pene0[idx_n],
            )
            om_t1, ct1, ht1 = fb_frictional_row(
                penetration=penetration[idx_t1],
                lam_t=lam[idx_t1],
                lam_n=lam[idx_n],
                mu=mu[c],
                precond=precond[idx_t1],
                dt=dt,
                pene0=pene0[idx_t1],
            )
            om_t2, ct2, ht2 = fb_frictional_row(
                penetration=penetration[idx_t2],
                lam_t=lam[idx_t2],
                lam_n=lam[idx_n],
                mu=mu[c],
                precond=precond[idx_t2],
                dt=dt,
                pene0=pene0[idx_t2],
            )
            omega[idx_n] = om_n
            omega[idx_t1] = om_t1
            omega[idx_t2] = om_t2
            compliance[idx_n] = cn
            compliance[idx_t1] = ct1
            compliance[idx_t2] = ct2
            h[idx_n] = hn
            h[idx_t1] = ht1
            h[idx_t2] = ht2

        # Schur LHS.
        A_schur = (omega[:, None] * omega[None, :]) * W + np.diag(compliance)
        # J·x_corrected = J·x_unc + dt²·W·(ω·λ) = (pene0 - r) + dt²·W·(ω·λ).
        J_x = (pene0 - r) + (dt * dt) * (W @ (omega * lam))
        rhs = (1.0 / (dt * dt)) * (h - omega * J_x)

        try:
            dlam = np.linalg.solve(A_schur, rhs)
        except np.linalg.LinAlgError:
            break
        lam = lam + dlam

        # Signed cone clamp: lam_n is NOT clamped to >= 0 inside the inner
        # loop. The tangent clamp is two independent if-statements applied
        # in order; when lam_n < 0 this saturates lam_t to -mu*lam_n
        # (positive) rather than producing a degenerate np.clip with
        # low > high. The GPU kernel applies them in the same order:
        for c in range(M):
            lam_n_c = lam[3 * c]  # signed
            upper = mu[c] * lam_n_c
            lower = -mu[c] * lam_n_c
            if lam[3 * c + 1] > upper:
                lam[3 * c + 1] = upper
            if lam[3 * c + 1] < lower:
                lam[3 * c + 1] = lower
            if lam[3 * c + 2] > upper:
                lam[3 * c + 2] = upper
            if lam[3 * c + 2] < lower:
                lam[3 * c + 2] = lower

    if lambda_cap is not None:
        cap_internal = lambda_cap / (dt * dt)
        np.clip(lam, -cap_internal, cap_internal, out=lam)
    lam_apply = (dt * dt) * omega * lam
    return lam, omega, lam_apply
