# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body joints for SolverFBA.

Each joint becomes bilateral rows on the integrated bodies, one per locked
direction: the complement of its free linear axes locks the anchor offset, and
the complement of its free angular axes locks the relative rotation. Directions
are expressed in the joint frame carried by the parent body.

Each free axis with passive damping (``Model.joint_damping``) adds a soft
row, a damper on the axis's motion over the step, and each finite limit of the
free axis of a revolute or prismatic joint (``Model.joint_limit_lower`` and
``Model.joint_limit_upper``) a candidate for a unilateral limit row, which
joins the step's rigid contact rows when the axis comes near the limit.
"""

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ...core.types import MAXVAL
from ...sim import JointType, Model
from .rigid import ROW_FORCE_LANES, row_force_rhs_kernel, row_incidence, slot_coefficients

_CONSTRAINT_JOINTS = (JointType.PRISMATIC, JointType.REVOLUTE, JointType.BALL, JointType.FIXED, JointType.D6)
_AXIS_JOINTS = (JointType.PRISMATIC, JointType.REVOLUTE)

ROW_LOCKED = wp.constant(0)
"""A row that locks a direction of a joint."""
ROW_DAMPER = wp.constant(1)
"""A soft row that damps the motion along a free axis of a joint."""


@dataclass
class JointRows:
    """Bilateral rows of the joints that constrain integrated bodies.

    Rows are grouped into independent blocks: connected components of the
    integrated bodies under these joints. Block ``k`` owns rows
    ``block_start[k]:block_start[k + 1]``.
    """

    parent: wp.array[wp.int32]
    """Parent body per joint, -1 for the world, shape [joint_count]."""
    child: wp.array[wp.int32]
    """Child body per joint, shape [joint_count]."""
    X_pj: wp.array[wp.transform]
    """Joint frame relative to the parent COM in the parent body frame, shape [joint_count]."""
    X_cj: wp.array[wp.transform]
    """Joint frame relative to the child COM in the child body frame, shape [joint_count]."""
    row_joint: wp.array[wp.int32]
    """Joint per row, shape [row_count]."""
    row_angular: wp.array[wp.int32]
    """1 for a rotational row, 0 for a translational row, shape [row_count]."""
    row_kind: wp.array[wp.int32]
    """:data:`ROW_LOCKED` or :data:`ROW_DAMPER` per row, shape [row_count]."""
    row_dir: wp.array[wp.vec3]
    """Locked direction, or the damped axis, in the joint frame, shape [row_count]."""
    row_damping: np.ndarray
    """Damping coefficient [N·s/m or N·m·s/rad] of each damper row, 0 for locked rows, shape [row_count]."""
    block_start: np.ndarray
    """Row range per independent block, shape [block_count + 1]."""
    joint_index: np.ndarray
    """Model joint of each joint of the rows, shape [joint_count]."""


def locked_directions(free_axes: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the complement of ``span(free_axes)`` in R³, shape [3 - rank, 3]."""
    if len(free_axes) == 0:
        return np.eye(3)
    _, s, vt = np.linalg.svd(np.atleast_2d(np.asarray(free_axes, dtype=np.float64)))
    return vt[int(np.sum(s > 1e-9 * s[0])) :]


def build_joint_rows(model: Model, body_slot: np.ndarray, device) -> JointRows | None:
    """Collect the rows of the enabled joints that touch an integrated body.

    Args:
        model: The Newton model.
        body_slot: Rigid DOF slot per model body, -1 for bodies the solver does
            not integrate; those keep their pose for the whole step.
        device: Warp device for the row arrays.

    Returns:
        The rows, or ``None`` if no joint constrains an integrated body.

    Raises:
        NotImplementedError: If such a joint is a distance or rod joint, or if the joints of a set of connected
            bodies lock more directions than the bodies have, as in kinematic loops and redundant joints.
    """
    if model.joint_count == 0:
        return None
    jtype, parent, child = model.joint_type.numpy(), model.joint_parent.numpy(), model.joint_child.numpy()
    pslot = np.where(parent >= 0, body_slot[np.maximum(parent, 0)], -1)
    cslot = body_slot[child]
    active = model.joint_enabled.numpy() & (jtype != JointType.FREE) & ((pslot >= 0) | (cslot >= 0))
    joints = np.flatnonzero(active)
    if joints.size == 0:
        return None
    for t in set(jtype[joints]) - set(_CONSTRAINT_JOINTS):
        # ROD shares its value with the deprecated CABLE, whose name the enum reports.
        name = "ROD" if t == JointType.ROD else JointType(t).name
        raise NotImplementedError(f"SolverFBA does not support {name} joints")

    # Union-find: each connected component of integrated bodies is one block.
    root = np.arange(body_slot.max() + 1)

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for j in joints[(pslot[joints] >= 0) & (cslot[joints] >= 0)]:
        root[find(pslot[j])] = find(cslot[j])

    axis, qd_start, dof_dim = model.joint_axis.numpy(), model.joint_qd_start.numpy(), model.joint_dof_dim.numpy()
    damping = model.joint_damping.numpy() if model.joint_damping is not None else np.zeros(len(axis))
    rows = []  # (block, local joint index, angular, direction, kind, damping)
    for k, j in enumerate(joints):
        n_lin, n_ang = dof_dim[j]
        axes = axis[qd_start[j] : qd_start[j] + n_lin + n_ang]
        block = find(max(pslot[j], cslot[j]))
        rows += [(block, k, 0, d, ROW_LOCKED, 0.0) for d in locked_directions(axes[:n_lin])]
        rows += [(block, k, 1, d, ROW_LOCKED, 0.0) for d in locked_directions(axes[n_lin:])]
        # Each free axis takes its passive damping as a soft row.
        for i in range(n_lin + n_ang):
            c = float(damping[qd_start[j] + i])
            if c > 0.0:
                d = np.asarray(axes[i], dtype=np.float64)
                rows.append((block, k, int(i >= n_lin), d / np.linalg.norm(d), ROW_DAMPER, c))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    blocks = np.array([r[0] for r in rows])
    block_start = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1], True]).astype(np.int32)
    slots = np.array([find(i) for i in range(len(root))])
    locked = np.array([r[4] == ROW_LOCKED for r in rows])
    for start, stop in itertools.pairwise(block_start):
        # The damper rows are soft: they keep W positive definite whatever they share with the locked rows.
        count = int(np.count_nonzero(locked[start:stop]))
        if count > 6 * np.count_nonzero(slots == blocks[start]):
            raise NotImplementedError(
                "SolverFBA does not support kinematic loops or redundant joints: the joints of a set of connected "
                f"bodies lock {count} directions of their {6 * np.count_nonzero(slots == blocks[start])}"
            )

    com = model.body_com.numpy()
    p, c = parent[joints], child[joints]
    X_pj, X_cj = model.joint_X_p.numpy()[joints], model.joint_X_c.numpy()[joints]
    X_pj[p >= 0, :3] -= com[p[p >= 0]]
    X_cj[:, :3] -= com[c]
    return JointRows(
        parent=wp.array(p, dtype=wp.int32, device=device),
        child=wp.array(c, dtype=wp.int32, device=device),
        X_pj=wp.array(X_pj, dtype=wp.transform, device=device),
        X_cj=wp.array(X_cj, dtype=wp.transform, device=device),
        row_joint=wp.array([r[1] for r in rows], dtype=wp.int32, device=device),
        row_angular=wp.array([r[2] for r in rows], dtype=wp.int32, device=device),
        row_kind=wp.array([r[4] for r in rows], dtype=wp.int32, device=device),
        row_dir=wp.array(np.array([r[3] for r in rows]), dtype=wp.vec3, device=device),
        row_damping=np.array([r[5] for r in rows], dtype=np.float64),
        block_start=block_start,
        joint_index=joints.astype(np.int32),
    )


LIMIT_MARGIN_ANGULAR = 0.02
"""Distance [rad] from a revolute limit within which, beyond the step's own motion, its row joins a step."""
LIMIT_MARGIN_LINEAR = 0.002
"""Distance [m] from a prismatic limit within which, beyond the step's own motion, its row joins a step."""


@dataclass
class JointLimits:
    """Limit candidates of the free axes of revolute and prismatic joints, one per finite limit.

    Each limit is a unilateral row ``g >= 0``: ``g = q - lower`` for a lower limit (``sign = 1``) and
    ``g = upper - q`` for an upper one (``sign = -1``), with ``q`` the axis coordinate of
    :func:`axis_coordinate`. A revolute coordinate is measured about the middle of its range, ``center``, so that the
    angle wraps on the far side of the range from the limits.
    """

    joint: wp.array[wp.int32]
    """Joint of each limit, an index into the per-joint arrays of :class:`JointRows`, shape [limit_count]."""
    angular: wp.array[wp.int32]
    """1 for a revolute limit, 0 for a prismatic one, shape [limit_count]."""
    axis: wp.array[wp.vec3]
    """Unit free axis in the joint frame, shape [limit_count]."""
    bound: wp.array[wp.float64]
    """Limit [rad or m], shape [limit_count]."""
    sign: wp.array[wp.float64]
    """1 for a lower limit, -1 for an upper one, shape [limit_count]."""
    center: wp.array[wp.float64]
    """Middle of a revolute axis's range [rad], 0 for a prismatic one, shape [limit_count]."""
    margin: wp.array[wp.float64]
    """:data:`LIMIT_MARGIN_ANGULAR` or :data:`LIMIT_MARGIN_LINEAR` per limit, shape [limit_count]."""
    slot: np.ndarray
    """Parent and child slots of each limit, -1 for a body that the solver does not integrate, shape
    [limit_count, 2]."""
    world: np.ndarray
    """World of each limit's child body, shape [limit_count]."""


def build_joint_limits(model: Model, rows: JointRows, body_slot: np.ndarray, device) -> JointLimits | None:
    """Collect the finite limits of the free axes of the revolute and prismatic joints of ``rows``.

    Args:
        model: The Newton model.
        rows: The joint rows on the integrated bodies.
        body_slot: Rigid DOF slot per model body, -1 for bodies the solver does not integrate.
        device: Warp device for the limit arrays.

    Returns:
        The limits, or ``None`` if no such axis has a finite limit.

    Raises:
        NotImplementedError: If a revolute axis has a single finite limit, or a range of 2 pi or more: its angle is
            known up to whole turns only.
        ValueError: If an axis's lower limit exceeds its upper one.
    """
    jtype, qd_start, axis = model.joint_type.numpy(), model.joint_qd_start.numpy(), model.joint_axis.numpy()
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    parent, child = rows.parent.numpy(), rows.child.numpy()
    world = model.body_world.numpy()
    limits = []  # (joint, angular, axis, bound, sign, center, margin)
    for k, j in enumerate(rows.joint_index):
        if jtype[j] not in _AXIS_JOINTS:
            continue
        dof = qd_start[j]
        lo, hi = float(lower[dof]), float(upper[dof])
        has_lo, has_hi = lo > -0.5 * MAXVAL, hi < 0.5 * MAXVAL
        if not (has_lo or has_hi):
            continue
        if has_lo and has_hi and lo > hi:
            raise ValueError(f"joint {j} has a lower limit {lo} above its upper limit {hi}")
        angular = int(jtype[j] == JointType.REVOLUTE)
        center, margin = 0.0, LIMIT_MARGIN_LINEAR
        if angular:
            if not (has_lo and has_hi) or hi - lo >= 2.0 * np.pi:
                raise NotImplementedError(
                    f"SolverFBA limits revolute joints within a range of less than 2 pi with both limits finite; "
                    f"joint {j} has limits ({lo}, {hi})"
                )
            center, margin = 0.5 * (lo + hi), LIMIT_MARGIN_ANGULAR
        a = np.asarray(axis[dof], dtype=np.float64)
        a = a / np.linalg.norm(a)
        if has_lo:
            limits.append((k, angular, a, lo, 1.0, center, margin))
        if has_hi:
            limits.append((k, angular, a, hi, -1.0, center, margin))
    if not limits:
        return None
    k = np.array([lim[0] for lim in limits])
    p, c = parent[k], child[k]
    slot = np.stack([np.where(p >= 0, body_slot[np.maximum(p, 0)], -1), body_slot[c]], axis=1).astype(np.int32)

    def column(i, dtype):
        return wp.array(np.array([lim[i] for lim in limits]), dtype=dtype, device=device)

    return JointLimits(
        joint=column(0, wp.int32),
        angular=column(1, wp.int32),
        axis=column(2, wp.vec3),
        bound=column(3, wp.float64),
        sign=column(4, wp.float64),
        center=column(5, wp.float64),
        margin=column(6, wp.float64),
        slot=slot,
        world=world[c].astype(np.int32),
    )


@wp.func
def _com_frame(
    b: int,
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
):
    """COM position and orientation of body ``b``: the per-slot pose for an
    integrated body, the state pose otherwise, and the origin for the world."""
    if b < 0:
        return wp.vec3d(0.0, 0.0, 0.0), wp.quatd(0.0, 0.0, 0.0, 1.0)
    s = body_slot[b]
    if s >= 0:
        return slot_x[s], slot_q[s]
    q = wp.normalize(wp.quatd(wp.transform_get_rotation(body_q[b])))
    return wp.vec3d(wp.transform_get_translation(body_q[b])) + wp.quat_rotate(q, wp.vec3d(body_com[b])), q


@wp.func
def _relative_rotation(q_ja: wp.quatd, q_jc: wp.quatd) -> wp.quatd:
    """Short-arc quaternion of ``R_Jaᵀ R_Jc``."""
    q = wp.quat_inverse(q_ja) * q_jc
    if q[3] < 0.0:
        q = q * wp.float64(-1.0)
    return q


@wp.func
def wrap_angle(angle: wp.float64) -> wp.float64:
    """``angle`` [rad] wrapped into ``(-pi, pi]``."""
    two_pi = wp.float64(2.0 * wp.pi)
    a = angle - two_pi * wp.floor((angle + wp.float64(wp.pi)) / two_pi)
    if a <= wp.float64(-wp.pi):
        a = a + two_pi
    return a


@wp.func
def axis_coordinate(
    angular: int,
    b: wp.vec3d,
    xa: wp.vec3d,
    qa: wp.quatd,
    xc: wp.vec3d,
    qc: wp.quatd,
    X_pj: wp.transform,
    X_cj: wp.transform,
) -> wp.float64:
    """Coordinate of a joint along its free axis ``b`` (joint frame) for the parent's and child's COM poses: the
    rotation angle about ``b`` [rad] in ``(-pi, pi]``, ``2 atan2(nu · b, eta)`` of the short-arc relative rotation
    ``(nu, eta)``, or the anchor offset along ``b`` [m]."""
    q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj))
    if angular == 1:
        q_rel = _relative_rotation(q_ja, qc * wp.quatd(wp.transform_get_rotation(X_cj)))
        return wp.float64(2.0) * wp.atan2(wp.dot(b, wp.vec3d(q_rel[0], q_rel[1], q_rel[2])), q_rel[3])
    x_ja = xa + wp.quat_rotate(qa, wp.vec3d(wp.transform_get_translation(X_pj)))
    x_jc = xc + wp.quat_rotate(qc, wp.vec3d(wp.transform_get_translation(X_cj)))
    return wp.dot(b, wp.quat_rotate_inv(q_ja, x_jc - x_ja))


@wp.kernel
def joint_jacobian_kernel(
    row_joint: wp.array[wp.int32],
    row_angular: wp.array[wp.int32],
    row_kind: wp.array[wp.int32],
    row_dir: wp.array[wp.vec3],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_pj: wp.array[wp.transform],
    X_cj: wp.array[wp.transform],
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
):
    """Row coefficients on the DOFs ``(chi, rho)`` of the parent and child body.

    Evaluated at the pinned frame. A translational row along ``d = R_Ja b`` acts
    on the anchor velocities; a rotational row uses the direction
    ``d = R_Ja (eta b + nu x b)`` of the relative rotation ``(eta, nu)``, which
    makes it the exact derivative of the residual ``2 nu · b``. A damper row
    on a rotation axis ``b`` uses ``d = R_Ja b``, the derivative of the angle
    about ``b`` while the other directions stay locked. World coefficients are
    rotated into each body's pinned frame.
    """
    r = wp.tid()
    j = row_joint[r]
    a = joint_parent[j]
    c = joint_child[j]
    _xa, qa = _com_frame(a, body_slot, slot_x, slot_q, body_q, body_com)
    _xc, qc = _com_frame(c, body_slot, slot_x, slot_q, body_q, body_com)
    q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj[j]))
    b = wp.vec3d(row_dir[r])
    lin = wp.vec3d(0.0, 0.0, 0.0)
    ang_a = wp.vec3d(0.0, 0.0, 0.0)
    ang_c = wp.vec3d(0.0, 0.0, 0.0)
    if row_angular[r] == 1:
        d = wp.quat_rotate(q_ja, b)
        if row_kind[r] == ROW_LOCKED:
            q_rel = _relative_rotation(q_ja, qc * wp.quatd(wp.transform_get_rotation(X_cj[j])))
            d = wp.quat_rotate(q_ja, q_rel[3] * b + wp.cross(wp.vec3d(q_rel[0], q_rel[1], q_rel[2]), b))
        ang_a = -d
        ang_c = d
    else:
        lin = wp.quat_rotate(q_ja, b)
        ang_a = -wp.cross(wp.quat_rotate(qa, wp.vec3d(wp.transform_get_translation(X_pj[j]))), lin)
        ang_c = wp.cross(wp.quat_rotate(qc, wp.vec3d(wp.transform_get_translation(X_cj[j]))), lin)
    sa = -1
    if a >= 0:
        sa = body_slot[a]
    row_slot[r, 0] = sa
    row_slot[r, 1] = body_slot[c]
    row_jac[r, 0] = slot_coefficients(-wp.quat_rotate_inv(qa, lin), wp.quat_rotate_inv(qa, ang_a))
    row_jac[r, 1] = slot_coefficients(wp.quat_rotate_inv(qc, lin), wp.quat_rotate_inv(qc, ang_c))


@wp.kernel
def joint_residual_kernel(
    row_joint: wp.array[wp.int32],
    row_angular: wp.array[wp.int32],
    row_kind: wp.array[wp.int32],
    row_dir: wp.array[wp.vec3],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_pj: wp.array[wp.transform],
    X_cj: wp.array[wp.transform],
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    row_start: wp.array[wp.float64],
    row_q_start: wp.array[wp.quatd],
    # outputs
    residual: wp.array[wp.float64],
):
    """Violation of each row at the live pose: the anchor offset ``R_Jaᵀ (x_Jc - x_Ja)``
    or the rotation ``2 nu`` along the locked direction, zero when satisfied. A
    damper row holds its axis's motion since the step start
    (:func:`joint_axis_start_kernel`): along a linear axis the change of the
    offset, about an angular axis ``b`` the angle ``2 atan2(nu · b, eta)`` of
    the relative rotation's increment ``(nu, eta)`` in the parent's joint
    frame, which for a revolute joint is the change of its angle."""
    r = wp.tid()
    j = row_joint[r]
    xa, qa = _com_frame(joint_parent[j], body_slot, slot_x, slot_q, body_q, body_com)
    xc, qc = _com_frame(joint_child[j], body_slot, slot_x, slot_q, body_q, body_com)
    q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj[j]))
    b = wp.vec3d(row_dir[r])
    if row_kind[r] == ROW_DAMPER:
        if row_angular[r] == 1:
            q_rel = _relative_rotation(q_ja, qc * wp.quatd(wp.transform_get_rotation(X_cj[j])))
            q_step = q_rel * wp.quat_inverse(row_q_start[r])
            if q_step[3] < 0.0:
                q_step = q_step * wp.float64(-1.0)
            nu = wp.vec3d(q_step[0], q_step[1], q_step[2])
            residual[r] = wp.float64(2.0) * wp.atan2(wp.dot(nu, b), q_step[3])
        else:
            residual[r] = axis_coordinate(0, b, xa, qa, xc, qc, X_pj[j], X_cj[j]) - row_start[r]
    elif row_angular[r] == 1:
        q_rel = _relative_rotation(q_ja, qc * wp.quatd(wp.transform_get_rotation(X_cj[j])))
        residual[r] = wp.float64(2.0) * wp.dot(b, wp.vec3d(q_rel[0], q_rel[1], q_rel[2]))
    else:
        x_ja = xa + wp.quat_rotate(qa, wp.vec3d(wp.transform_get_translation(X_pj[j])))
        x_jc = xc + wp.quat_rotate(qc, wp.vec3d(wp.transform_get_translation(X_cj[j])))
        residual[r] = wp.dot(b, wp.quat_rotate_inv(q_ja, x_jc - x_ja))


@wp.kernel
def joint_axis_start_kernel(
    row_joint: wp.array[wp.int32],
    row_angular: wp.array[wp.int32],
    row_kind: wp.array[wp.int32],
    row_dir: wp.array[wp.vec3],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_pj: wp.array[wp.transform],
    X_cj: wp.array[wp.transform],
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    row_start: wp.array[wp.float64],
    row_q_start: wp.array[wp.quatd],
):
    """Each damper row's state at the given poses, from which its residual measures the step's motion: a linear
    row's axis offset (:func:`axis_coordinate`) in ``row_start``, an angular row's relative rotation
    ``R_Jaᵀ R_Jc`` in ``row_q_start``. For the step start, ``body_slot`` is -1 for every body, so that the poses come
    from ``body_q``: the pinned frame of an integrated body is already turned by its start velocity."""
    r = wp.tid()
    value = wp.float64(0.0)
    q_rel = wp.quatd(0.0, 0.0, 0.0, 1.0)
    if row_kind[r] == ROW_DAMPER:
        j = row_joint[r]
        xa, qa = _com_frame(joint_parent[j], body_slot, slot_x, slot_q, body_q, body_com)
        xc, qc = _com_frame(joint_child[j], body_slot, slot_x, slot_q, body_q, body_com)
        if row_angular[r] == 1:
            q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj[j]))
            q_rel = _relative_rotation(q_ja, qc * wp.quatd(wp.transform_get_rotation(X_cj[j])))
        else:
            value = axis_coordinate(0, wp.vec3d(row_dir[r]), xa, qa, xc, qc, X_pj[j], X_cj[j])
    row_start[r] = value
    row_q_start[r] = q_rel


@wp.func
def limit_gap(
    k: int,
    joint: wp.array[wp.int32],
    angular: wp.array[wp.int32],
    axis: wp.array[wp.vec3],
    bound: wp.array[wp.float64],
    sign: wp.array[wp.float64],
    center: wp.array[wp.float64],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_pj: wp.array[wp.transform],
    X_cj: wp.array[wp.transform],
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
) -> wp.float64:
    """Gap ``g`` of limit ``k`` at the given poses: ``sign (q - bound)``, a revolute angle measured from the middle of
    its range."""
    j = joint[k]
    xa, qa = _com_frame(joint_parent[j], body_slot, slot_x, slot_q, body_q, body_com)
    xc, qc = _com_frame(joint_child[j], body_slot, slot_x, slot_q, body_q, body_com)
    q = axis_coordinate(angular[k], wp.vec3d(axis[k]), xa, qa, xc, qc, X_pj[j], X_cj[j])
    if angular[k] == 1:
        q = center[k] + wrap_angle(q - center[k])
    return sign[k] * (q - bound[k])


@wp.kernel
def joint_limit_near_kernel(
    joint: wp.array[wp.int32],
    angular: wp.array[wp.int32],
    axis: wp.array[wp.vec3],
    bound: wp.array[wp.float64],
    sign: wp.array[wp.float64],
    center: wp.array[wp.float64],
    margin: wp.array[wp.float64],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_pj: wp.array[wp.transform],
    X_cj: wp.array[wp.transform],
    body_slot: wp.array[wp.int32],
    no_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    near: wp.array[wp.int32],
):
    """1 for each limit whose row joins the step, else 0: one whose gap at the step start (``body_q``, read for every
    body through ``no_slot``, -1 per body) or after the step's free flight (``slot_x``, ``slot_q``) falls within its
    margin plus the free flight's change of the gap."""
    k = wp.tid()
    g0 = limit_gap(
        k, joint, angular, axis, bound, sign, center, joint_parent, joint_child, X_pj, X_cj, no_slot,
        slot_x, slot_q, body_q, body_com,
    )  # fmt: skip
    g1 = limit_gap(
        k, joint, angular, axis, bound, sign, center, joint_parent, joint_child, X_pj, X_cj, body_slot,
        slot_x, slot_q, body_q, body_com,
    )  # fmt: skip
    near[k] = wp.where(wp.min(g0, g1) < margin[k] + wp.abs(g1 - g0), 1, 0)


@wp.kernel
def joint_limit_jacobian_kernel(
    rows: wp.array[wp.int32],
    joint: wp.array[wp.int32],
    angular: wp.array[wp.int32],
    axis: wp.array[wp.vec3],
    sign: wp.array[wp.float64],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_pj: wp.array[wp.transform],
    X_cj: wp.array[wp.transform],
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    row_offset: int,
    # outputs
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
):
    """Row ``row_offset + 3i`` of the step's limit ``rows[i]`` at the pinned frame: ``sign`` times the derivative of
    its axis coordinate, the rows of :func:`joint_jacobian_kernel` along the free axis (a damper row's for a
    revolute axis). Its other two rows stay inert, as a frictionless contact's, with no slot and zero
    coefficients."""
    i = wp.tid()
    k = rows[i]
    j = joint[k]
    a = joint_parent[j]
    c = joint_child[j]
    _xa, qa = _com_frame(a, body_slot, slot_x, slot_q, body_q, body_com)
    _xc, qc = _com_frame(c, body_slot, slot_x, slot_q, body_q, body_com)
    q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj[j]))
    d = wp.quat_rotate(q_ja, wp.vec3d(axis[k])) * sign[k]
    lin = wp.vec3d(0.0, 0.0, 0.0)
    ang_a = -d
    ang_c = d
    if angular[k] == 0:
        lin = d
        ang_a = -wp.cross(wp.quat_rotate(qa, wp.vec3d(wp.transform_get_translation(X_pj[j]))), lin)
        ang_c = wp.cross(wp.quat_rotate(qc, wp.vec3d(wp.transform_get_translation(X_cj[j]))), lin)
    r = row_offset + 3 * i
    sa = -1
    if a >= 0:
        sa = body_slot[a]
    row_slot[r, 0] = sa
    row_slot[r, 1] = body_slot[c]
    row_jac[r, 0] = slot_coefficients(-wp.quat_rotate_inv(qa, lin), wp.quat_rotate_inv(qa, ang_a))
    row_jac[r, 1] = slot_coefficients(wp.quat_rotate_inv(qc, lin), wp.quat_rotate_inv(qc, ang_c))


@wp.kernel
def joint_limit_gap_kernel(
    rows: wp.array[wp.int32],
    joint: wp.array[wp.int32],
    angular: wp.array[wp.int32],
    axis: wp.array[wp.vec3],
    bound: wp.array[wp.float64],
    sign: wp.array[wp.float64],
    center: wp.array[wp.float64],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_pj: wp.array[wp.transform],
    X_cj: wp.array[wp.transform],
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    row_offset: int,
    # outputs
    value: wp.array[wp.float64],
):
    """Row values of the step's limit ``rows[i]`` at the live pose: its gap (:func:`limit_gap`), then the inert
    rows' zeros."""
    i = wp.tid()
    r = row_offset + 3 * i
    value[r] = limit_gap(
        rows[i], joint, angular, axis, bound, sign, center, joint_parent, joint_child, X_pj, X_cj, body_slot,
        slot_x, slot_q, body_q, body_com,
    )  # fmt: skip
    value[r + 1] = wp.float64(0.0)
    value[r + 2] = wp.float64(0.0)


@wp.kernel
def joint_limit_carry_kernel(
    rows: wp.array[wp.int32],
    src: wp.array[wp.float64],
    src_offset: int,
    to_rows: int,
    dst_offset: int,
    # outputs
    dst: wp.array[wp.float64],
):
    """Carry the three NSN values of each step limit ``rows[i]`` between the step's rows (from ``src_offset + 3i``)
    and their per-limit slots (``3 rows[i]``): into the step's rows (from ``dst_offset + 3i``) for ``to_rows``,
    else out of them."""
    i = wp.tid()
    k = rows[i]
    for lane in range(3):
        if to_rows == 1:
            dst[dst_offset + 3 * i + lane] = src[3 * k + lane]
        else:
            dst[3 * k + lane] = src[src_offset + 3 * i + lane]


@wp.kernel
def bilateral_delassus_kernel(
    block_start: wp.array[wp.int32],
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
    slot_inv_mass: wp.array[wp.spatial_matrixd],
    compliance: wp.array[wp.float64],
    # outputs
    W: wp.array3d[wp.float64],
):
    """Delassus block ``J A⁻¹ Jᵀ`` per independent block, padded with identity, plus the rows' ``compliance`` on its
    diagonal where given (an empty array for none)."""
    k, i, j = wp.tid()
    n = block_start[k + 1] - block_start[k]
    if i >= n or j >= n:
        W[k, i, j] = wp.where(i == j, wp.float64(1.0), wp.float64(0.0))
        return
    r = block_start[k] + i
    s = block_start[k] + j
    w = wp.float64(0.0)
    for x in range(2):
        for y in range(2):
            sx = row_slot[r, x]
            if sx >= 0 and sx == row_slot[s, y]:
                w += wp.dot(row_jac[r, x], slot_inv_mass[sx] * row_jac[s, y])
    if i == j and compliance.shape[0] > 0:
        w += compliance[r]
    W[k, i, j] = w


@wp.kernel
def bilateral_rhs_kernel(
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    residual: wp.array[wp.float64],
    compliance: wp.array[wp.float64],
    force: wp.array[wp.float64],
    # outputs
    rhs: wp.array2d[wp.float64],
):
    """``-(residual + compliance force)`` per row, the soft rows' ``compliance`` where given (an empty array for
    none): a soft row holds ``residual + compliance force = 0``."""
    r = wp.tid()
    value = residual[r]
    if compliance.shape[0] > 0:
        value += compliance[r] * force[r]
    rhs[row_block[r], row_local[r]] = -value


@wp.kernel
def bilateral_update_kernel(
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    df: wp.array2d[wp.float64],
    # outputs
    force: wp.array[wp.float64],
):
    r = wp.tid()
    force[r] = force[r] + df[row_block[r], row_local[r]]


@wp.kernel
def block_gather_kernel(
    rows: wp.array[wp.int32],
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    v: wp.array[wp.float64],
    offset: int,
    # outputs
    rhs: wp.array2d[wp.float64],
):
    """``rhs[block, local] = v[offset + k]`` for joint row ``rows[k]``, none where ``rows[k]`` is -1."""
    k = wp.tid()
    row = rows[k]
    if row >= 0:
        rhs[row_block[row], row_local[row]] = v[offset + k]


@wp.kernel
def block_scatter_kernel(
    rows: wp.array[wp.int32],
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    x: wp.array2d[wp.float64],
    offset: int,
    # outputs
    out: wp.array[wp.float64],
):
    """``out[offset + k] = x[block, local]`` for joint row ``rows[k]``, none where ``rows[k]`` is -1."""
    k = wp.tid()
    row = rows[k]
    if row >= 0:
        out[offset + k] = x[row_block[row], row_local[row]]


_TILE_KERNELS = {}


def _tile_kernels(tile: int):
    """Cholesky factor and solve kernels for blocks padded to ``tile`` rows."""
    if tile not in _TILE_KERNELS:
        # No adjoints: the one of a 64-row factor needs 65792 bytes of static shared memory, over the 48 KiB limit.

        @wp.kernel(module="unique", enable_backward=False)
        def factor(W: wp.array3d[wp.float64], L: wp.array3d[wp.float64]):
            k = wp.tid()
            wp.tile_store(L[k], wp.tile_cholesky(wp.tile_load(W[k], shape=(tile, tile))))

        @wp.kernel(module="unique", enable_backward=False)
        def solve(L: wp.array3d[wp.float64], rhs: wp.array2d[wp.float64], x: wp.array2d[wp.float64]):
            k = wp.tid()
            L_k = wp.tile_load(L[k], shape=(tile, tile))
            wp.tile_store(x[k], wp.tile_cholesky_solve(L_k, wp.tile_load(rhs[k], shape=tile)))

        _TILE_KERNELS[tile] = (factor, solve)
    return _TILE_KERNELS[tile]


class BilateralSystem:
    """Newton system ``W df = -residual`` of the joint rows, ``W = J A⁻¹ Jᵀ``.

    ``W`` is block diagonal over the independent blocks of :class:`JointRows`;
    each block is Cholesky-factored once per step. ``slot_inv_mass`` holds the
    inverse 6x6 diagonal blocks of the rigid system matrix.

    A damper row of coefficient ``c`` is soft: over a step ``dt`` it holds its
    axis's motion ``g`` and force ``f`` at ``g + (dt / c) f = 0``, the backward
    Euler step of the damper, so its diagonal of ``W`` gains the compliance
    ``dt / c`` and its residual the term ``(dt / c) f``.
    """

    def __init__(self, rows: JointRows, body_slot: np.ndarray, slot_inv_mass: wp.array, device):
        sizes = np.diff(rows.block_start)
        self.tile = max(8, 1 << int(np.ceil(np.log2(sizes.max()))))
        # An fp64 tile Cholesky of 128 rows needs more shared memory than GPUs provide.
        if self.tile > 64:
            raise NotImplementedError(f"SolverFBA supports at most 64 joint rows per articulation, got {sizes.max()}")
        n_blocks, R = len(sizes), int(rows.block_start[-1])
        self.slot_inv_mass = slot_inv_mass

        row_block = np.repeat(np.arange(n_blocks), sizes)
        self.row_block = wp.array(row_block, dtype=wp.int32, device=device)
        self.row_local = wp.array(np.arange(R) - rows.block_start[row_block], dtype=wp.int32, device=device)
        joint = rows.row_joint.numpy()
        parent, child = rows.parent.numpy()[joint], rows.child.numpy()[joint]
        slots = np.stack([np.where(parent >= 0, body_slot[np.maximum(parent, 0)], -1), body_slot[child]], axis=1)
        start, entry_row, entry_side = row_incidence(slots, slot_inv_mass.shape[0])
        self.slot_entry_start = wp.array(start, dtype=wp.int32, device=device)
        self.entry_row = wp.array(entry_row, dtype=wp.int32, device=device)
        self.entry_side = wp.array(entry_side, dtype=wp.int32, device=device)

        self.block_start = wp.array(rows.block_start, dtype=wp.int32, device=device)
        self.W = wp.empty((n_blocks, self.tile, self.tile), dtype=wp.float64, device=device)
        self.L = wp.empty_like(self.W)
        self.rhs = wp.zeros((n_blocks, self.tile), dtype=wp.float64, device=device)
        self.df = wp.zeros_like(self.rhs)
        self.device = device
        # The damper rows' coefficients, and each row's compliance for the step (dt / c, zero on locked rows): an
        # empty array without damper rows.
        self.damping = np.asarray(rows.row_damping, dtype=np.float64)
        has_dampers = bool((self.damping > 0.0).any())
        self.compliance = wp.zeros(R if has_dampers else 0, dtype=wp.float64, device=device)
        self._compliance_dt = None

    def set_dt(self, dt: float) -> None:
        """Set the damper rows' compliance ``dt / c`` for steps of ``dt``."""
        if self.compliance.shape[0] == 0 or self._compliance_dt == dt:
            return
        damped = self.damping > 0.0
        self.compliance.assign(np.where(damped, dt / np.where(damped, self.damping, 1.0), 0.0))
        self._compliance_dt = dt

    def factor(self, row_slot: wp.array2d[wp.int32], row_jac: wp.array2d[wp.spatial_vectord]) -> None:
        """Assemble and factor ``W`` for this step's Jacobian, the damper rows' compliance (:meth:`set_dt`) on its
        diagonal."""
        wp.launch(
            bilateral_delassus_kernel,
            dim=self.W.shape,
            inputs=[self.block_start, row_slot, row_jac, self.slot_inv_mass, self.compliance],
            outputs=[self.W],
            device=self.device,
        )
        wp.launch_tiled(
            _tile_kernels(self.tile)[0], dim=self.W.shape[0], inputs=[self.W, self.L], block_dim=64, device=self.device
        )

    def update(self, residual: wp.array[wp.float64], force: wp.array[wp.float64]) -> None:
        """Newton step on the rows: ``force += W⁻¹ (-residual)``, the damper rows' soft term included."""
        R = residual.shape[0]
        wp.launch(
            bilateral_rhs_kernel,
            dim=R,
            inputs=[self.row_block, self.row_local, residual, self.compliance, force],
            outputs=[self.rhs],
            device=self.device,
        )
        wp.launch_tiled(
            _tile_kernels(self.tile)[1],
            dim=self.W.shape[0],
            inputs=[self.L, self.rhs, self.df],
            block_dim=64,
            device=self.device,
        )
        wp.launch(
            bilateral_update_kernel,
            dim=R,
            inputs=[self.row_block, self.row_local, self.df],
            outputs=[force],
            device=self.device,
        )

    def solve_rows(self, rows: wp.array[wp.int32], offset: int, v: wp.array, out: wp.array) -> None:
        """Block solve ``out = W⁻¹ v`` on joint rows ``rows`` placed at ``offset + k`` in both vectors.

        ``rows`` must cover whole blocks; entries of -1 take no part, and their rows of ``out`` stay as they were.
        The factor comes from :meth:`factor`.
        """
        wp.launch(
            block_gather_kernel,
            dim=rows.shape[0],
            inputs=[rows, self.row_block, self.row_local, v, offset],
            outputs=[self.rhs],
            device=self.device,
        )
        wp.launch_tiled(
            _tile_kernels(self.tile)[1],
            dim=self.W.shape[0],
            inputs=[self.L, self.rhs, self.df],
            block_dim=64,
            device=self.device,
        )
        wp.launch(
            block_scatter_kernel,
            dim=rows.shape[0],
            inputs=[rows, self.row_block, self.row_local, self.df, offset],
            outputs=[out],
            device=self.device,
        )

    def apply_force(self, row_jac, force, b_const, b) -> None:
        """``b = b_const + Jᵀ force``."""
        wp.launch_tiled(
            row_force_rhs_kernel,
            dim=self.slot_entry_start.shape[0] - 1,
            inputs=[self.slot_entry_start, self.entry_row, self.entry_side, row_jac, force, b_const],
            outputs=[b],
            block_dim=ROW_FORCE_LANES,
            device=self.device,
        )


@wp.kernel
def select_rows_kernel(
    index: wp.array[wp.int32],
    src_slot: wp.array2d[wp.int32],
    src_jac: wp.array2d[wp.spatial_vectord],
    offset: int,
    # outputs
    dst_slot: wp.array2d[wp.int32],
    dst_jac: wp.array2d[wp.spatial_vectord],
):
    """Copy rows ``index`` of one row set to rows ``offset + k`` of another."""
    k = wp.tid()
    for side in range(2):
        dst_slot[offset + k, side] = src_slot[index[k], side]
        dst_jac[offset + k, side] = src_jac[index[k], side]


@wp.kernel
def gather_values_kernel(
    src: wp.array[wp.float64],
    index: wp.array[wp.int32],
    offset: int,
    # outputs
    dst: wp.array[wp.float64],
):
    """``dst[offset + k] = src[index[k]]``, zero where ``index[k]`` is -1."""
    k = wp.tid()
    value = wp.float64(0.0)
    if index[k] >= 0:
        value = src[index[k]]
    dst[offset + k] = value


@wp.kernel
def gather_rows_kernel(
    src: wp.array[Any],
    index: wp.array[wp.int32],
    # outputs
    dst: wp.array[Any],
):
    """``dst[k] = src[index[k]]``."""
    k = wp.tid()
    dst[k] = src[index[k]]


@wp.kernel
def gather_rows_2d_kernel(
    src: wp.array2d[Any],
    index: wp.array[wp.int32],
    # outputs
    dst: wp.array2d[Any],
):
    """``dst[k, j] = src[index[k], j]``."""
    k, j = wp.tid()
    dst[k, j] = src[index[k], j]


@wp.kernel
def scatter_values_kernel(
    src: wp.array[wp.float64],
    offset: int,
    index: wp.array[wp.int32],
    # outputs
    dst: wp.array[wp.float64],
):
    """``dst[index[k]] = src[offset + k]``, none where ``index[k]`` is -1."""
    k = wp.tid()
    if index[k] >= 0:
        dst[index[k]] = src[offset + k]
