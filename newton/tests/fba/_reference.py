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

    Port of RealSim ``NonSmoothNewton.cpp:332-341``. Returns
    ``(omega, compliance, h)``.

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

    Port of RealSim ``NonSmoothNewton.cpp:343-378``. Returns
    ``(omega, compliance, h)``.

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
