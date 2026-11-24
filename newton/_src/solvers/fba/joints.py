# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body joints for SolverFBA.

Each joint becomes bilateral rows on the integrated bodies, one per locked
direction: the complement of its free linear axes locks the anchor offset, and
the complement of its free angular axes locks the relative rotation. Directions
are expressed in the joint frame carried by the parent body.

Each free axis with passive damping (``Model.joint_damping``), a drive
(``Model.joint_target_ke`` or ``Model.joint_target_kd``), armature
(``Model.joint_armature``), joint friction (``Model.joint_friction``) or, on a
revolute or prismatic joint, finite limits (``Model.joint_limit_lower`` and
``Model.joint_limit_upper``) adds one row on the axis's motion over the step,
whose force is the axis's total force: the backward Euler step of the drive,
damping and armature forces, the drive bounded by its effort limit
(``Model.joint_effort_limit``) when below 1e6, the dry friction, and the
limits' forces. The relation between the row's motion and force is monotone and
piecewise linear: a row with an effort limit, friction or limits takes, at each
Newton step, the piece that one step of the row alone would reach
(:func:`joint_axis_law_kernel`).

A mimic joint (``Model.joint_mimic_joint``) holds each coordinate at ``c0 + c1``
times its leader's, as a pair of rows on the two axes, the follower's and the
leader's, combined into one row on their Newton system: the follower's row takes
the leader's minus ``c1`` times, and the leader's row stays inert, its force
``-c1`` times the follower's.
"""

import itertools
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ...core.types import MAXVAL
from ...exceptions import NewtonWarning
from ...sim import JointType, Model
from ...sim.articulation import transform_2d_rotational_axes
from .rigid import ROW_FORCE_LANES, row_force_rhs_kernel, row_incidence, slot_coefficients

wp.set_module_options({"enable_backward": False})

_CONSTRAINT_JOINTS = (JointType.PRISMATIC, JointType.REVOLUTE, JointType.BALL, JointType.FIXED, JointType.D6)
_AXIS_JOINTS = (JointType.PRISMATIC, JointType.REVOLUTE)

ROW_LOCKED = wp.constant(0)
"""A row that locks a direction of a joint."""
ROW_DAMPER = wp.constant(1)
"""A soft row on the motion along a free axis of a joint, carried by the parent's joint frame: its passive damping,
drive and armature."""
ROW_HOOKE = wp.constant(2)
"""A row that locks the twist of a D6 joint with two free angular axes, ``a0`` carried by the parent and ``a1`` by
the child, as Newton composes the joint's rotation ``q(a0, q0) q(a1, q1)``: it holds the angle between the two
axes."""
ROW_DAMPER_CHILD = wp.constant(3)
"""A soft row on the rotation about a free axis carried by the child's joint frame: the second axis of a D6 joint
with two free angular axes."""
ROW_MIMIC = wp.constant(4)
"""A row on the coordinate of a free axis, half of a mimic pair: the follower's row, combined with its leader's, holds
the follower's coordinate at ``c0 + c1`` times the leader's."""

LAW_SOFT = wp.constant(0)
"""A row whose compliance and target motion hold for the step (:func:`joint_soft_row_kernel`)."""
LAW_AXIS = wp.constant(1)
"""The row of a free axis whose force law has several pieces: a drive bounded by its effort limit, joint friction or
limits. :func:`joint_axis_law_kernel` linearizes it at each Newton step."""
LAW_FREEZE = wp.constant(1.0e6)
"""Compliance, relative to ``W_ii``, of an axis row whose force does not change with its motion on its piece, such as
a drive at its effort limit or a sliding friction: a Newton step lands the force on its value up to the step's own
motion over this compliance."""
LAW_HOLD = wp.constant(1.0e-4)
"""Compliance, relative to ``W_ii``, of an axis row held at a motion, by static friction or at a limit: it regularizes
the Newton step alone, the row's target motion following its force."""
LAW_FAR = wp.constant(1.0e30)
"""The motion bound [rad or m] of an axis without that limit."""
_FINITE_LIMIT = wp.constant(0.5 * MAXVAL)
"""Limits beyond this magnitude [rad or m] are absent."""
EFFORT_UNLIMITED = 1.0e6
"""Effort limits at or above this value, the builder's default, leave a drive unbounded."""


@wp.func
def _is_damper(kind: wp.int32) -> bool:
    return kind == ROW_DAMPER or kind == ROW_DAMPER_CHILD


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
    """:data:`ROW_LOCKED`, :data:`ROW_DAMPER`, :data:`ROW_HOOKE` or :data:`ROW_DAMPER_CHILD` per row, shape
    [row_count]."""
    row_dir: wp.array[wp.vec3]
    """Locked direction, the damped axis, or a :data:`ROW_HOOKE` row's first axis, in the joint frame, shape
    [row_count]."""
    row_axis2: wp.array[wp.vec3]
    """A :data:`ROW_HOOKE` row's second axis in the joint frame, zero for the other rows, shape [row_count]."""
    row_damping: np.ndarray
    """Passive damping coefficient [N·s/m or N·m·s/rad] of each soft row, 0 for locked rows, shape [row_count]."""
    row_soft: np.ndarray
    """Whether each row is a soft row (:data:`ROW_DAMPER` or :data:`ROW_DAMPER_CHILD`), shape [row_count]."""
    row_dof: wp.array[wp.int32]
    """Model DOF of each soft row, -1 for the other rows, shape [row_count]."""
    row_coord: wp.array[wp.int32]
    """Model coordinate of each soft row's position target, -1 for the other rows and for axes without a position
    drive, shape [row_count]."""
    row_law: wp.array[wp.int32]
    """:data:`LAW_SOFT` or :data:`LAW_AXIS` per row, shape [row_count]."""
    row_limited: wp.array[wp.int32]
    """1 for the axis row of a revolute or prismatic joint with limits, which its law holds, shape [row_count]."""
    mimic_rows: np.ndarray
    """The follower's and the leader's row of each mimic pair, shape [pair_count, 2]."""
    mimic_coeffs: np.ndarray
    """The offset ``c0`` and multiplier ``c1`` of each mimic pair, shape [pair_count, 2]."""
    block_start: np.ndarray
    """Row range per independent block, shape [block_count + 1]."""
    joint_index: np.ndarray
    """Model joint of each joint of the rows, shape [joint_count]."""


# A kernel of its own module, so that the rows' setup does not compile this module for the CPU.
@wp.kernel(module="unique")
def _hooke_axes_kernel(axis_0: wp.vec3, axis_1: wp.vec3, out: wp.array[wp.vec3]):
    a0, a1 = transform_2d_rotational_axes(axis_0, axis_1, 0.0)
    out[0] = a0
    out[1] = a1


def hooke_axes(axis_0: np.ndarray, axis_1: np.ndarray) -> np.ndarray:
    """The two free angular axes of a D6 joint as Newton composes its rotation, orthonormalized, shape [2, 3]."""
    out = wp.empty(2, dtype=wp.vec3, device="cpu")
    wp.launch(_hooke_axes_kernel, 1, [wp.vec3(*axis_0), wp.vec3(*axis_1)], [out], device="cpu")
    return out.numpy().astype(np.float64)


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

    # A mimic joint and its leader share their block.
    local = np.full(model.joint_count, -1)
    local[joints] = np.arange(joints.size)
    mimics = []
    if model.joint_mimic_joint is not None and (model.joint_mimic_joint.numpy()[joints] >= 0).any():
        from ...sim.joint_mimic import has_supported_joint_mimics  # noqa: PLC0415

        if has_supported_joint_mimics(model, "SolverFBA"):
            leaders, coeffs = model.joint_mimic_joint.numpy(), model.joint_mimic_coeffs.numpy()
            supported = (JointType.PRISMATIC, JointType.REVOLUTE, JointType.D6)
            for j in joints:
                lead = leaders[j]
                if lead < 0 or jtype[j] not in supported or jtype[lead] not in supported:
                    continue
                if local[lead] < 0:
                    raise NotImplementedError(
                        f"SolverFBA needs the leader {lead} of mimic joint {j} to be a joint on the bodies it integrates"
                    )
                root[find(max(pslot[j], cslot[j]))] = find(max(pslot[lead], cslot[lead]))
                mimics.append((j, lead, float(coeffs[j][0]), float(coeffs[j][1])))

    axis, qd_start, dof_dim = model.joint_axis.numpy(), model.joint_qd_start.numpy(), model.joint_dof_dim.numpy()
    q_start = model.joint_q_start.numpy()

    def per_dof(values):
        return values.numpy().astype(np.float64) if values is not None else np.zeros(len(axis))

    damping, armature = per_dof(model.joint_damping), per_dof(model.joint_armature)
    target_ke, target_kd = per_dof(model.joint_target_ke), per_dof(model.joint_target_kd)
    friction = per_dof(getattr(model, "joint_friction", None))
    effort = per_dof(getattr(model, "joint_effort_limit", None))
    if getattr(model, "joint_effort_limit", None) is None:
        effort[:] = EFFORT_UNLIMITED
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    zero = np.zeros(3)
    # (block, local joint index, angular, direction, kind, damping, second axis, dof, coordinate, mimic pair and side,
    # law, limited)
    rows = []
    turns = []
    for k, j in enumerate(joints):
        n_lin, n_ang = dof_dim[j]
        axes = np.array(axis[qd_start[j] : qd_start[j] + n_lin + n_ang], dtype=np.float64)
        block = find(max(pslot[j], cslot[j]))
        rows += [
            (block, k, 0, d, ROW_LOCKED, 0.0, zero, -1, -1, -1, LAW_SOFT, 0) for d in locked_directions(axes[:n_lin])
        ]
        kinds = [ROW_DAMPER] * (n_lin + n_ang)
        if n_ang == 2:
            # The rotation q(a0, q0) q(a1, q1) keeps the angle between a0 in the parent's joint frame and a1 in the
            # child's, and turns about a1 in the child's frame; the axes as Newton orthonormalizes them.
            axes[n_lin:] = hooke_axes(axes[n_lin], axes[n_lin + 1])
            rows.append((block, k, 1, axes[n_lin], ROW_HOOKE, 0.0, axes[n_lin + 1], -1, -1, -1, LAW_SOFT, 0))
            kinds[n_lin + 1] = ROW_DAMPER_CHILD
        else:
            rows += [
                (block, k, 1, d, ROW_LOCKED, 0.0, zero, -1, -1, -1, LAW_SOFT, 0)
                for d in locked_directions(axes[n_lin:])
            ]
        # Each free axis takes its passive damping, drive, armature, friction and limits as one row.
        for i in range(n_lin + n_ang):
            dof = qd_start[j] + i
            d = axes[i] / np.linalg.norm(axes[i])
            limited = 0
            if jtype[j] in _AXIS_JOINTS:
                lo, hi = float(lower[dof]), float(upper[dof])
                has_lo, has_hi = lo > -0.5 * MAXVAL, hi < 0.5 * MAXVAL
                if has_lo and has_hi and lo > hi:
                    raise ValueError(f"joint {j} has a lower limit {lo} above its upper limit {hi}")
                if jtype[j] == JointType.REVOLUTE and has_lo != has_hi:
                    raise NotImplementedError(
                        f"SolverFBA limits revolute joints with both limits finite; joint {j} has limits ({lo}, {hi})"
                    )
                if jtype[j] == JointType.REVOLUTE and has_lo and hi - lo >= 2.0 * np.pi:
                    # An angle known up to whole turns lies within such a range at every turn.
                    turns.append(int(j))
                else:
                    limited = int(has_lo or has_hi)
            if max(damping[dof], armature[dof], target_ke[dof], target_kd[dof], friction[dof], limited) <= 0.0:
                continue
            coord = -1
            if target_ke[dof] > 0.0:
                if i >= n_lin and n_ang > 1:
                    raise NotImplementedError(
                        f"SolverFBA drives the angle of a joint with one free angular axis only; joint {j} has "
                        f"{n_ang} and a position gain on DOF {dof}"
                    )
                coord = q_start[j] + i
            bounded = max(target_ke[dof], target_kd[dof]) > 0.0 and effort[dof] < EFFORT_UNLIMITED
            law = LAW_AXIS if bounded or friction[dof] > 0.0 or limited else LAW_SOFT
            rows.append(
                (block, k, int(i >= n_lin), d, kinds[i], float(damping[dof]), zero, dof, coord, -1, law, limited)
            )
    # Each coordinate of a mimic joint takes a pair of rows: the follower's axis and the leader's.
    pair_coeffs = []
    for j, lead, c0, c1 in mimics:
        dims = (tuple(dof_dim[j]), tuple(dof_dim[lead]))
        if dims[0] != dims[1] or (dims[0][1] > 1):
            raise NotImplementedError(
                f"SolverFBA's mimic joints need the follower {j} and the leader {lead} to have the same axes, with "
                f"at most one angular axis; got {dims[0]} and {dims[1]}"
            )
        block = find(max(pslot[j], cslot[j]))
        n_lin = dims[0][0]
        for i in range(sum(dims[0])):
            pair = len(pair_coeffs)
            for side, joint in enumerate((j, lead)):
                d = axis[qd_start[joint] + i] / np.linalg.norm(axis[qd_start[joint] + i])
                rows.append(
                    (
                        block,
                        local[joint],
                        int(i >= n_lin),
                        d,
                        ROW_MIMIC,
                        0.0,
                        zero,
                        -1,
                        -1,
                        2 * pair + side,
                        LAW_SOFT,
                        0,
                    )
                )
            pair_coeffs.append((c0, c1))
    if turns:
        warnings.warn(
            f"SolverFBA does not enforce the limits of revolute joints {turns}, whose ranges span a full turn or more: "
            "it knows their angles up to whole turns",
            NewtonWarning,
            stacklevel=2,
        )
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    blocks = np.array([r[0] for r in rows])
    block_start = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1], True]).astype(np.int32)
    slots = np.array([find(i) for i in range(len(root))])
    # A mimic pair locks one direction, its follower's row; the leader's row is inert.
    locked = np.array([r[4] in (ROW_LOCKED, ROW_HOOKE) or (r[4] == ROW_MIMIC and r[9] % 2 == 0) for r in rows])
    # The soft rows: the axis rows, which take a compliance on W's diagonal.
    soft = np.array([r[4] in (ROW_DAMPER, ROW_DAMPER_CHILD) for r in rows])
    mimic_rows = np.zeros((len(pair_coeffs), 2), dtype=np.int32)
    for index, r in enumerate(rows):
        if r[9] >= 0:
            mimic_rows[r[9] // 2, r[9] % 2] = index
    for start, stop in itertools.pairwise(block_start):
        # The soft rows keep W positive definite whatever they share with the locked rows.
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
        row_axis2=wp.array(np.array([r[6] for r in rows]), dtype=wp.vec3, device=device),
        row_damping=np.array([r[5] for r in rows], dtype=np.float64),
        row_soft=soft,
        row_dof=wp.array([r[7] for r in rows], dtype=wp.int32, device=device),
        row_coord=wp.array([r[8] for r in rows], dtype=wp.int32, device=device),
        row_law=wp.array([r[10] for r in rows], dtype=wp.int32, device=device),
        row_limited=wp.array([r[11] for r in rows], dtype=wp.int32, device=device),
        block_start=block_start,
        joint_index=joints.astype(np.int32),
        mimic_rows=mimic_rows,
        mimic_coeffs=np.array(pair_coeffs, dtype=np.float64).reshape(-1, 2),
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


@wp.kernel
def advance_unintegrated_kernel(
    body_slot: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    dt: float,
    # outputs
    body_q_end: wp.array[wp.transform],
):
    """The poses at the step's end of the bodies that the solver does not integrate: a kinematic body moves on at its
    prescribed velocity, its center of mass at ``body_qd``'s linear part and its orientation at the angular part
    (both in the world frame), as Newton prescribes ``q + qd dt``; a static one stays. The integrated bodies, whose
    poses come from their slots, keep their state poses."""
    b = wp.tid()
    tf = body_q[b]
    if body_slot[b] >= 0:
        body_q_end[b] = tf
        return
    v = wp.spatial_top(body_qd[b])
    w = wp.spatial_bottom(body_qd[b])
    q = wp.transform_get_rotation(tf)
    com = wp.transform_point(tf, body_com[b])
    angle = wp.length(w) * dt
    if angle > 0.0:
        q = wp.normalize(wp.quat_from_axis_angle(wp.normalize(w), angle) * q)
    body_q_end[b] = wp.transform(com + v * dt - wp.quat_rotate(q, body_com[b]), q)


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
    row_axis2: wp.array[wp.vec3],
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
    on the anchor velocities, and a locked one also on the parent's turn of
    ``d`` by the anchors' offset; a rotational row uses the direction
    ``d = R_Ja (eta b + nu x b)`` of the relative rotation ``(eta, nu)``, which
    makes it the exact derivative of the residual ``2 nu · b``. A damper row
    on a rotation axis ``b`` uses ``d = R_Ja b``, the derivative of the angle
    about ``b`` while the other directions stay locked, or ``d = R_Jc b`` for
    an axis the child carries. A :data:`ROW_HOOKE` row on the axes
    ``u = R_Ja a0`` and ``v = R_Jc a1`` takes ``u x v`` on the parent and
    ``v x u`` on the child, the derivative of ``u · v``. World coefficients are
    rotated into each body's pinned frame.
    """
    r = wp.tid()
    j = row_joint[r]
    a = joint_parent[j]
    c = joint_child[j]
    xa, qa = _com_frame(a, body_slot, slot_x, slot_q, body_q, body_com)
    xc, qc = _com_frame(c, body_slot, slot_x, slot_q, body_q, body_com)
    q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj[j]))
    b = wp.vec3d(row_dir[r])
    lin = wp.vec3d(0.0, 0.0, 0.0)
    ang_a = wp.vec3d(0.0, 0.0, 0.0)
    ang_c = wp.vec3d(0.0, 0.0, 0.0)
    if row_kind[r] == ROW_HOOKE:
        q_jc = qc * wp.quatd(wp.transform_get_rotation(X_cj[j]))
        n = wp.cross(wp.quat_rotate(q_ja, b), wp.quat_rotate(q_jc, wp.vec3d(row_axis2[r])))
        ang_a = n
        ang_c = -n
    elif row_angular[r] == 1:
        d = wp.quat_rotate(q_ja, b)
        if row_kind[r] == ROW_LOCKED:
            q_rel = _relative_rotation(q_ja, qc * wp.quatd(wp.transform_get_rotation(X_cj[j])))
            d = wp.quat_rotate(q_ja, q_rel[3] * b + wp.cross(wp.vec3d(q_rel[0], q_rel[1], q_rel[2]), b))
        elif row_kind[r] == ROW_DAMPER_CHILD:
            d = wp.quat_rotate(qc * wp.quatd(wp.transform_get_rotation(X_cj[j])), b)
        ang_a = -d
        ang_c = d
    else:
        lin = wp.quat_rotate(q_ja, b)
        r_a = wp.quat_rotate(qa, wp.vec3d(wp.transform_get_translation(X_pj[j])))
        r_c = wp.quat_rotate(qc, wp.vec3d(wp.transform_get_translation(X_cj[j])))
        ang_a = -wp.cross(r_a, lin)
        ang_c = wp.cross(r_c, lin)
        if row_kind[r] == ROW_LOCKED or row_kind[r] == ROW_MIMIC:
            # The locked direction turns with the parent, so the parent's turn also moves the row by the anchors'
            # offset along it: zero for coincident anchors, the slide of a prismatic joint across it.
            ang_a = ang_a + wp.cross(lin, (xc + r_c) - (xa + r_a))
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
    row_axis2: wp.array[wp.vec3],
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
    row_target: wp.array[wp.float64],
    # outputs
    residual: wp.array[wp.float64],
):
    """Violation of each row at the live pose: the anchor offset ``R_Jaᵀ (x_Jc - x_Ja)``
    or the rotation ``2 nu`` along the locked direction, or for a
    :data:`ROW_HOOKE` row ``(R_Ja a0) · (R_Jc a1) - a0 · a1``, zero when
    satisfied. A soft row holds its axis's motion since the step start
    (:func:`joint_axis_start_kernel`) less its target motion ``row_target``
    (:func:`joint_soft_row_kernel`): along a linear axis the change of the
    offset, about an angular axis ``b`` the angle ``2 atan2(nu · b, eta)`` of
    the relative rotation's increment ``(nu, eta)`` in the parent's joint
    frame, which for a revolute joint is the change of its angle, or in the
    child's joint frame for an axis the child carries."""
    r = wp.tid()
    j = row_joint[r]
    xa, qa = _com_frame(joint_parent[j], body_slot, slot_x, slot_q, body_q, body_com)
    xc, qc = _com_frame(joint_child[j], body_slot, slot_x, slot_q, body_q, body_com)
    q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj[j]))
    b = wp.vec3d(row_dir[r])
    if row_kind[r] == ROW_HOOKE:
        q_jc = qc * wp.quatd(wp.transform_get_rotation(X_cj[j]))
        a1 = wp.vec3d(row_axis2[r])
        residual[r] = wp.dot(wp.quat_rotate(q_ja, b), wp.quat_rotate(q_jc, a1)) - wp.dot(b, a1)
    elif row_kind[r] == ROW_MIMIC:
        # The axis's coordinate, which mimic_combine_kernel turns into the pair's violation.
        residual[r] = axis_coordinate(row_angular[r], b, xa, qa, xc, qc, X_pj[j], X_cj[j])
    elif _is_damper(row_kind[r]):
        if row_angular[r] == 1:
            q_rel = _relative_rotation(q_ja, qc * wp.quatd(wp.transform_get_rotation(X_cj[j])))
            q_step = q_rel * wp.quat_inverse(row_q_start[r])
            if row_kind[r] == ROW_DAMPER_CHILD:
                q_step = wp.quat_inverse(row_q_start[r]) * q_rel
            if q_step[3] < 0.0:
                q_step = q_step * wp.float64(-1.0)
            nu = wp.vec3d(q_step[0], q_step[1], q_step[2])
            residual[r] = wp.float64(2.0) * wp.atan2(wp.dot(nu, b), q_step[3]) - row_target[r]
        else:
            residual[r] = axis_coordinate(0, b, xa, qa, xc, qc, X_pj[j], X_cj[j]) - row_start[r] - row_target[r]
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
    row_axis2: wp.array[wp.vec3],
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
    if _is_damper(row_kind[r]):
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


@wp.kernel
def joint_soft_row_kernel(
    row_joint: wp.array[wp.int32],
    row_angular: wp.array[wp.int32],
    row_kind: wp.array[wp.int32],
    row_dir: wp.array[wp.vec3],
    row_dof: wp.array[wp.int32],
    row_coord: wp.array[wp.int32],
    row_law: wp.array[wp.int32],
    row_limited: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    X_pj: wp.array[wp.transform],
    X_cj: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    joint_damping: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
    joint_target_ke: wp.array[wp.float32],
    joint_target_kd: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
    joint_target_qd: wp.array[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    dt: wp.float64,
    # outputs
    compliance: wp.array[wp.float64],
    target: wp.array[wp.float64],
    terms: wp.array2d[wp.float64],
):
    """Each axis row's compliance and target motion over a step of ``dt`` from the poses and velocities at the
    step's start, 0 for the other rows.

    An axis of coordinate ``q`` takes the drive force ``ke (q* - q) + kd (q_dot* - q_dot)``, the passive damping
    ``-c q_dot`` and the armature's inertial force ``-I_a q_ddot``, each at the step's end: with the axis's motion
    ``g`` over the step, ``q_dot = g / dt`` and ``q_ddot = (g / dt - q_dot0) / dt``, the total force is
    ``f = b - k g`` with ``k = ke + (kd + c) / dt + I_a / dt²`` and ``b = ke (q* - q0) + kd q_dot* + I_a q_dot0 / dt``.
    The row holds ``g - g* + C f = 0`` with the compliance ``C = 1 / k`` and the target motion ``g* = b / k``. A
    damper alone keeps ``C = dt / c``. ``q0`` and ``q_dot0`` are the axis's coordinate and rate at the start: the
    angle about an angular axis ``b`` of the relative rotation in the parent's joint frame and the relative angular
    velocity along it (along the child's frame axis for an axis the child carries), or the anchor offset along a
    linear axis and its rate in the parent's joint frame.

    A row of :data:`LAW_AXIS` also keeps, for :func:`joint_axis_law_kernel`, ``terms[r] = (k_a, b_a, k_p, b_p, C,
    g*, g_lo, g_up)``: its drive's part ``b_a - k_a g`` (``k_a = ke + kd / dt``, ``b_a = ke (q* - q0) + kd q_dot*``),
    the rest ``b_p - k_p g``, and the motions ``lower - q0`` and ``upper - q0`` at which it meets its limits
    (``∓LAW_FAR`` without them), a revolute angle measured about the middle of its range.
    """
    r = wp.tid()
    if not _is_damper(row_kind[r]):
        compliance[r] = wp.float64(0.0)
        target[r] = wp.float64(0.0)
        return
    dof = row_dof[r]
    law = row_law[r] == LAW_AXIS
    c = wp.float64(joint_damping[dof])
    arm = wp.float64(joint_armature[dof])
    ke = wp.float64(joint_target_ke[dof])
    kd = wp.float64(joint_target_kd[dof])
    if ke == wp.float64(0.0) and arm == wp.float64(0.0) and kd == wp.float64(0.0) and not law:
        # A damper alone, or a row whose coefficients were all zeroed, which then takes no force.
        compliance[r] = wp.where(c > wp.float64(0.0), dt / c, wp.float64(1.0e30))
        target[r] = wp.float64(0.0)
        return
    j = row_joint[r]
    a = joint_parent[j]
    ch = joint_child[j]
    xa = wp.vec3d(0.0, 0.0, 0.0)
    qa = wp.quatd(0.0, 0.0, 0.0, 1.0)
    va = wp.vec3d(0.0, 0.0, 0.0)
    wa = wp.vec3d(0.0, 0.0, 0.0)
    if a >= 0:
        xa = wp.vec3d(wp.transform_point(body_q[a], body_com[a]))
        qa = wp.quatd(wp.transform_get_rotation(body_q[a]))
        va = wp.vec3d(wp.spatial_top(body_qd[a]))
        wa = wp.vec3d(wp.spatial_bottom(body_qd[a]))
    xc = wp.vec3d(wp.transform_point(body_q[ch], body_com[ch]))
    qc = wp.quatd(wp.transform_get_rotation(body_q[ch]))
    vc = wp.vec3d(wp.spatial_top(body_qd[ch]))
    wc = wp.vec3d(wp.spatial_bottom(body_qd[ch]))
    q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj[j]))
    b = wp.vec3d(row_dir[r])
    rate = wp.float64(0.0)
    if row_angular[r] == 1:
        d = wp.quat_rotate(q_ja, b)
        if row_kind[r] == ROW_DAMPER_CHILD:
            d = wp.quat_rotate(qc * wp.quatd(wp.transform_get_rotation(X_cj[j])), b)
        rate = wp.dot(wc - wa, d)
    else:
        r_a = wp.quat_rotate(qa, wp.vec3d(wp.transform_get_translation(X_pj[j])))
        r_c = wp.quat_rotate(qc, wp.vec3d(wp.transform_get_translation(X_cj[j])))
        offset = (xc + r_c) - (xa + r_a)
        v_rel = (vc + wp.cross(wc, r_c)) - (va + wp.cross(wa, r_a)) - wp.cross(wa, offset)
        rate = wp.dot(b, wp.quat_rotate_inv(q_ja, v_rel))
    force = arm * rate / dt
    drive = wp.float64(0.0)
    if kd > wp.float64(0.0):
        velocity_term = kd * wp.float64(joint_target_qd[dof])
        force += velocity_term
        drive += velocity_term
    if row_coord[r] >= 0:
        q0 = axis_coordinate(row_angular[r], b, xa, qa, xc, qc, X_pj[j], X_cj[j])
        position_term = ke * (wp.float64(joint_target_q[row_coord[r]]) - q0)
        force += position_term
        drive += position_term
    stiffness = ke * dt * dt + (kd + c) * dt + arm
    if stiffness > wp.float64(0.0):
        compliance[r] = dt * dt / stiffness
        target[r] = force * dt * dt / stiffness
    else:
        # An axis with friction or limits alone: no force off them.
        compliance[r] = wp.float64(1.0e30)
        target[r] = wp.float64(0.0)
    if law:
        terms[r, 0] = ke + kd / dt
        terms[r, 1] = drive
        terms[r, 2] = c / dt + arm / (dt * dt)
        terms[r, 3] = arm * rate / dt
        terms[r, 4] = compliance[r]
        terms[r, 5] = target[r]
        lo = wp.float64(-LAW_FAR)
        hi = wp.float64(LAW_FAR)
        if row_limited[r] != 0:
            lower = wp.float64(joint_limit_lower[dof])
            upper = wp.float64(joint_limit_upper[dof])
            q = axis_coordinate(row_angular[r], b, xa, qa, xc, qc, X_pj[j], X_cj[j])
            if row_angular[r] == 1:
                middle = wp.float64(0.5) * (lower + upper)
                q = middle + wrap_angle(q - middle)
            if lower > -wp.float64(_FINITE_LIMIT):
                lo = lower - q
            if upper < wp.float64(_FINITE_LIMIT):
                hi = upper - q
        terms[r, 6] = lo
        terms[r, 7] = hi


@wp.kernel
def mimic_combine_kernel(
    rows: wp.array2d[wp.int32],
    coeffs: wp.array2d[wp.float64],
    offset: int,
    with_offset: int,
    # outputs
    values: wp.array[wp.float64],
):
    """Combine the row values of each mimic pair, the follower's row at ``offset + rows[p, 0]`` and the leader's at
    ``offset + rows[p, 1]``: the follower's takes ``-c1`` times the leader's (and ``-c0`` with ``with_offset``), the
    pair's violation ``q_f - c0 - c1 q_l`` from the coordinates, and the leader's becomes 0."""
    p = wp.tid()
    a = offset + rows[p, 0]
    b = offset + rows[p, 1]
    value = values[a] - coeffs[p, 1] * values[b]
    if with_offset != 0:
        value -= coeffs[p, 0]
    values[a] = value
    values[b] = wp.float64(0.0)


@wp.kernel
def mimic_force_kernel(
    rows: wp.array2d[wp.int32],
    coeffs: wp.array2d[wp.float64],
    offset: int,
    # outputs
    force: wp.array[wp.float64],
):
    """The leader's row of each mimic pair takes ``-c1`` times the follower's force, so that the pair applies the
    combined row's force to both joints."""
    p = wp.tid()
    force[offset + rows[p, 1]] = -coeffs[p, 1] * force[offset + rows[p, 0]]


@wp.kernel
def mimic_matrix_rows_kernel(
    block: wp.array[wp.int32],
    local: wp.array2d[wp.int32],
    coeffs: wp.array2d[wp.float64],
    # outputs
    W: wp.array3d[wp.float64],
):
    """The first step of ``T W Tᵀ`` for the mimic pairs: the follower's row of block ``block[p]`` takes ``-c1`` times
    the leader's."""
    p, j = wp.tid()
    k = block[p]
    W[k, local[p, 0], j] = W[k, local[p, 0], j] - coeffs[p, 1] * W[k, local[p, 1], j]


@wp.kernel
def mimic_matrix_columns_kernel(
    block: wp.array[wp.int32],
    local: wp.array2d[wp.int32],
    coeffs: wp.array2d[wp.float64],
    # outputs
    W: wp.array3d[wp.float64],
):
    """The second step of ``T W Tᵀ``: the follower's column takes ``-c1`` times the leader's."""
    p, i = wp.tid()
    k = block[p]
    W[k, i, local[p, 0]] = W[k, i, local[p, 0]] - coeffs[p, 1] * W[k, i, local[p, 1]]


@wp.kernel
def mimic_matrix_inert_kernel(
    block: wp.array[wp.int32],
    local: wp.array2d[wp.int32],
    # outputs
    W: wp.array3d[wp.float64],
):
    """The last step: the leader's row and column become those of the identity, an inert row."""
    p, i = wp.tid()
    k = block[p]
    b = local[p, 1]
    if i == b:
        W[k, b, b] = wp.float64(1.0)
    else:
        W[k, b, i] = wp.float64(0.0)
        W[k, i, b] = wp.float64(0.0)


def combine_mimic_matrix(W: wp.array3d, block: wp.array, local: wp.array, coeffs: wp.array, device) -> None:
    """``W <- T W Tᵀ`` for the mimic pairs in blocks ``block`` at the local rows ``local``, the leader's rows
    inert."""
    pairs, size = local.shape[0], W.shape[1]
    if not pairs:
        return
    wp.launch(mimic_matrix_rows_kernel, (pairs, size), [block, local, coeffs], [W], device=device)
    wp.launch(mimic_matrix_columns_kernel, (pairs, size), [block, local, coeffs], [W], device=device)
    wp.launch(mimic_matrix_inert_kernel, (pairs, size), [block, local], [W], device=device)


@wp.kernel
def joint_wrench_kernel(
    joint_type: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    joint_f: wp.array[wp.float32],
    # outputs
    body_f: wp.array[wp.spatial_vector],
):
    """Add each joint's generalized forces ``joint_f`` to its bodies' wrenches about their centers of mass, as
    :class:`~newton.solvers.SolverSemiImplicit` applies them: a FREE or DISTANCE joint's world wrench to the child,
    and along each axis, in the parent's joint frame, the force or torque to the child at its joint anchor and the
    opposite to the parent at its own."""
    j = wp.tid()
    if not joint_enabled[j]:
        return
    t = joint_type[j]
    c = joint_child[j]
    start = joint_qd_start[j]
    if t == JointType.FREE or t == JointType.DISTANCE:
        f = wp.vec3(joint_f[start], joint_f[start + 1], joint_f[start + 2])
        tau = wp.vec3(joint_f[start + 3], joint_f[start + 4], joint_f[start + 5])
        wp.atomic_add(body_f, c, wp.spatial_vector(f, tau))
        return
    p = joint_parent[j]
    X_wp = joint_X_p[j]
    r_p = wp.vec3(0.0)
    if p >= 0:
        X_wp = wp.transform_multiply(body_q[p], X_wp)
        r_p = wp.transform_get_translation(X_wp) - wp.transform_point(body_q[p], body_com[p])
    X_wc = wp.transform_multiply(body_q[c], joint_X_c[j])
    r_c = wp.transform_get_translation(X_wc) - wp.transform_point(body_q[c], body_com[c])
    n_lin = joint_dof_dim[j, 0]
    n_ang = joint_dof_dim[j, 1]
    f = wp.vec3(0.0)
    tau = wp.vec3(0.0)
    for i in range(n_lin):
        f += joint_f[start + i] * wp.transform_vector(X_wp, joint_axis[start + i])
    for i in range(n_ang):
        tau += joint_f[start + n_lin + i] * wp.transform_vector(X_wp, joint_axis[start + n_lin + i])
    if wp.length_sq(f) == 0.0 and wp.length_sq(tau) == 0.0:
        return
    if p >= 0:
        wp.atomic_sub(body_f, p, wp.spatial_vector(f, tau + wp.cross(r_p, f)))
    wp.atomic_add(body_f, c, wp.spatial_vector(f, tau + wp.cross(r_c, f)))


@wp.kernel
def bilateral_delassus_kernel(
    block_start: wp.array[wp.int32],
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
    slot_inv_mass: wp.array[wp.spatial_matrixd],
    # outputs
    W: wp.array3d[wp.float64],
):
    """Delassus block ``J A⁻¹ Jᵀ`` per independent block, padded with identity."""
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
    W[k, i, j] = w


@wp.kernel
def bilateral_diagonal_kernel(
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    compliance: wp.array[wp.float64],
    # outputs
    diagonal: wp.array2d[wp.float64],
):
    """Each row's compliance at its place in its block's diagonal, the soft rows' (:func:`joint_soft_row_kernel`)
    and the axis rows' (:func:`joint_axis_law_kernel`); none without soft rows (an empty ``compliance``)."""
    r = wp.tid()
    if compliance.shape[0] > 0:
        diagonal[row_block[r], row_local[r]] = compliance[r]


@wp.kernel
def joint_law_start_kernel(
    law_rows: wp.array[wp.int32],
    row_dof: wp.array[wp.int32],
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    W: wp.array3d[wp.float64],
    terms: wp.array2d[wp.float64],
    friction: wp.array[wp.float32],
    # outputs
    compliance: wp.array[wp.float64],
):
    """The compliance of each axis row ``law_rows[i]`` at the step's first factor, before its law
    (:func:`joint_axis_law_kernel`) sets it: held, :data:`LAW_HOLD` times its ``W_ii``, with friction; its drive's,
    damping's and armature's, ``terms[r, 4]``, with these; else :data:`LAW_FREEZE` times ``W_ii``, no force."""
    i = wp.tid()
    r = law_rows[i]
    d = W[row_block[r], row_local[r], row_local[r]]
    if friction[row_dof[r]] > wp.float32(0.0):
        compliance[r] = wp.float64(LAW_HOLD) * d
    elif terms[r, 0] + terms[r, 2] > wp.float64(0.0):
        compliance[r] = terms[r, 4]
    else:
        compliance[r] = wp.float64(LAW_FREEZE) * d


@wp.func
def _axis_piece(g: wp.float64, k_a: wp.float64, b_a: wp.float64, effort: wp.float64, k_p: wp.float64,
                b_p: wp.float64, friction: wp.float64):  # fmt: skip
    """The linear piece ``f = alpha - beta g`` of an axis's force law about the motion ``g`` off its jumps: the drive
    ``b_a - k_a g`` within ``[-effort, effort]``, the rest ``b_p - k_p g``, and the friction ``-friction sign(g)``.
    Returns ``(alpha, beta)``."""
    alpha = b_p
    beta = k_p
    drive = b_a - k_a * g
    if drive > effort:
        alpha += effort
    elif drive < -effort:
        alpha -= effort
    else:
        alpha += b_a
        beta += k_a
    if g < wp.float64(0.0):
        alpha += friction
    elif g > wp.float64(0.0):
        alpha -= friction
    return alpha, beta


@wp.kernel
def joint_axis_law_kernel(
    law_rows: wp.array[wp.int32],
    row_dof: wp.array[wp.int32],
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    W: wp.array3d[wp.float64],
    terms: wp.array2d[wp.float64],
    effort_limit: wp.array[wp.float32],
    friction: wp.array[wp.float32],
    force: wp.array[wp.float64],
    # outputs
    compliance: wp.array[wp.float64],
    target: wp.array[wp.float64],
    residual: wp.array[wp.float64],
    diagonal: wp.array2d[wp.float64],
    refactor: wp.array[wp.int32],
):
    """The law of each axis row ``law_rows[i]`` for a Newton step: the piece of its force law that one step of the
    row alone reaches, linearized as the row's compliance and target motion, which its ``residual`` then measures
    against, so that the step solves ``W df + c df = -(g - g* + c f)``.

    The row's force ``f`` and its axis's motion ``g`` since the step's start (its ``residual`` plus ``target``)
    relate by a monotone decreasing law (:func:`joint_soft_row_kernel`'s ``terms``): the drive ``b_a - k_a g``
    within the effort limit ``[-E, E]``, the damping and armature ``b_p - k_p g``, the friction ``-F sign(g)``, which
    holds the axis at ``g = 0`` with a force within ``F`` of the rest, and the limits, which hold ``g`` within
    ``[g_lo, g_up]`` with any force pushing it inside. One step of the row alone moves along ``g' = g + d (f' - f)``
    (``d = W_ii``); where that line meets the law, the row takes the law's piece there: a linear piece
    ``f = alpha - beta g``, as the compliance ``1 / k`` and the target ``g + (alpha - beta g) / k``, with ``k`` at
    least ``1 / (LAW_FREEZE d)`` (the drive's own soft row where it is within its limit without friction), or a hold
    at a motion ``p`` (static friction, a limit), as the compliance ``c = LAW_HOLD d`` and the target ``p + c f``.
    A row whose compliance changes marks its block for :meth:`BilateralSystem.refactor`."""
    i = wp.tid()
    r = law_rows[i]
    k = row_block[r]
    local = row_local[r]
    d = W[k, local, local]
    old_target = target[r]
    g = residual[r] + old_target
    f = force[r]
    dof = row_dof[r]
    k_a = terms[r, 0]
    b_a = terms[r, 1]
    k_p = terms[r, 2]
    b_p = terms[r, 3]
    lo = terms[r, 6]
    hi = terms[r, 7]
    effort = wp.float64(LAW_FAR)
    if k_a > wp.float64(0.0) and wp.float64(effort_limit[dof]) < wp.float64(EFFORT_UNLIMITED):
        effort = wp.float64(effort_limit[dof])
    fr = wp.float64(friction[dof])
    # The law's breakpoints within (lo, hi), ascending: the drive's saturations and, with friction, g = 0.
    p0 = wp.float64(LAW_FAR)
    p1 = wp.float64(LAW_FAR)
    p2 = wp.float64(LAW_FAR)
    if k_a > wp.float64(0.0) and effort < wp.float64(LAW_FAR):
        p0 = (b_a - effort) / k_a
        p2 = (b_a + effort) / k_a
    if fr > wp.float64(0.0):
        p1 = wp.float64(0.0)
    if p1 < p0:
        t = p0
        p0 = p1
        p1 = t
    if p2 < p1:
        t = p1
        p1 = p2
        p2 = t
    if p1 < p0:
        t = p0
        p0 = p1
        p1 = t
    # Scan the law's segments from lo up: the line g' = g + d (f' - f), along which the step of the row alone moves,
    # rises across the decreasing law once.
    hold = 0
    point = wp.float64(0.0)
    alpha = wp.float64(0.0)
    beta = wp.float64(0.0)
    found = 0
    started = 0
    x = lo
    for n in range(4):
        if found == 0:
            y = hi
            if n == 0:
                y = wp.min(p0, hi)
            elif n == 1:
                y = wp.min(p1, hi)
            elif n == 2:
                y = wp.min(p2, hi)
            if y > x:
                # The segment (x, y) is one linear piece: its coefficients from a motion inside it.
                probe = wp.float64(0.5) * (x + y)
                if x <= -wp.float64(LAW_FAR):
                    probe = wp.where(y >= wp.float64(LAW_FAR), wp.float64(0.0), y - wp.float64(1.0))
                elif y >= wp.float64(LAW_FAR):
                    probe = x + wp.float64(1.0)
                a1, b1 = _axis_piece(probe, k_a, b_a, effort, k_p, b_p, fr)
                if started == 0 and x > -wp.float64(LAW_FAR):
                    # At the lower limit, which pushes up with any force: the line meets it there if it lies above
                    # the law just past it.
                    if f + (x - g) / d >= a1 - b1 * x:
                        hold = 1
                        point = x
                        found = 1
                started = 1
                if found == 0:
                    root = (d * (a1 - f) + g) / (d * b1 + wp.float64(1.0))
                    if root <= y or y >= wp.float64(LAW_FAR):
                        alpha = a1
                        beta = b1
                        found = 1
                    elif y >= hi:
                        # Past the last piece below the upper limit, which pushes down with any force.
                        hold = 1
                        point = hi
                        found = 1
                    elif fr > wp.float64(0.0) and y == wp.float64(0.0):
                        # The friction's jump at g = 0: the line meets it unless it lies below the law just past it.
                        a2, _b2 = _axis_piece(wp.float64(1.0e-300), k_a, b_a, effort, k_p, b_p, fr)
                        if f - g / d >= a2:
                            hold = 1
                            point = wp.float64(0.0)
                            found = 1
                x = y
    c = wp.float64(0.0)
    goal = wp.float64(0.0)
    if hold == 1:
        c = wp.float64(LAW_HOLD) * d
        goal = point + c * f
    else:
        if fr == wp.float64(0.0) and beta == k_a + k_p and alpha == b_a + b_p and beta > wp.float64(0.0):
            # The drive within its limit, without friction: the row's own soft law.
            c = terms[r, 4]
            goal = terms[r, 5]
        else:
            stiffness = wp.max(beta, wp.float64(1.0) / (wp.float64(LAW_FREEZE) * d))
            c = wp.float64(1.0) / stiffness
            goal = g + (alpha - beta * g) / stiffness
    if c != compliance[r]:
        refactor[k] = 1
    compliance[r] = c
    diagonal[k, local] = c
    if goal != old_target:
        target[r] = goal
        residual[r] = g - goal


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
        def factor(
            W: wp.array3d[wp.float64],
            diagonal: wp.array2d[wp.float64],
            refactor: wp.array[wp.int32],
            L: wp.array3d[wp.float64],
        ):
            # The blocks marked in ``refactor`` alone, which it then clears.
            k = wp.tid()
            if refactor[k] != 0:
                A = wp.tile_diag_add(wp.tile_load(W[k], shape=(tile, tile)), wp.tile_load(diagonal[k], shape=tile))
                wp.tile_store(L[k], wp.tile_cholesky(A))
                refactor[k] = 0

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

    A soft row holds its axis's motion ``g`` and force ``f`` at
    ``g - g* + C f = 0``, the backward Euler step of its damper, drive and
    armature (:func:`joint_soft_row_kernel`), so its diagonal of ``W`` gains
    the compliance ``C`` and its residual the term ``C f``. An axis row of
    :data:`LAW_AXIS` (an effort limit, joint friction, limits) takes its
    compliance and target motion from its law at each Newton step
    (:meth:`axis_law`), and the blocks whose compliance changes are factored
    again.
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
        # Each row's compliance for the step, zero on locked rows, set by joint_soft_row_kernel: an empty array
        # without soft rows. It joins W's diagonal at the factor, from ``diagonal``.
        self.compliance = wp.zeros(R if bool(rows.row_soft.any()) else 0, dtype=wp.float64, device=device)
        self.diagonal = wp.zeros_like(self.rhs)
        # The blocks the next factor takes: all of them at the step's start, those whose axis rows changed pieces at a
        # Newton step.
        self.refactor_blocks = wp.zeros(n_blocks, dtype=wp.int32, device=device)
        # The axis rows of LAW_AXIS, whose law sets their compliance at each Newton step, their model DOFs, and their
        # terms (joint_soft_row_kernel).
        self.row_dof = rows.row_dof
        law = rows.row_law.numpy()
        self.law_rows = wp.array(np.flatnonzero(law == LAW_AXIS).astype(np.int32), dtype=wp.int32, device=device)
        self.law_terms = wp.zeros((R if self.law_rows.shape[0] else 0, 8), dtype=wp.float64, device=device)
        # The mimic pairs: their rows, coefficients, block and rows within it.
        pairs = np.asarray(rows.mimic_rows, dtype=np.int32).reshape(-1, 2)
        self.mimic_rows = wp.array(pairs, dtype=wp.int32, device=device)
        self.mimic_coeffs = wp.array(np.asarray(rows.mimic_coeffs).reshape(-1, 2), dtype=wp.float64, device=device)
        self.mimic_block = wp.array(row_block[pairs[:, 0]], dtype=wp.int32, device=device)
        self.mimic_local = wp.array(
            (pairs - rows.block_start[row_block[pairs]]).astype(np.int32), dtype=wp.int32, device=device
        )

    def factor(
        self,
        row_slot: wp.array2d[wp.int32],
        row_jac: wp.array2d[wp.spatial_vectord],
        friction: wp.array | None = None,
    ) -> None:
        """Assemble ``W`` for this step's Jacobian, with the mimic pairs combined (:func:`combine_mimic_matrix`), and
        factor it with the soft rows' compliance on its diagonal, the axis rows' before their laws set it
        (:func:`joint_law_start_kernel`, from the DOFs' ``friction``)."""
        wp.launch(
            bilateral_delassus_kernel,
            dim=self.W.shape,
            inputs=[self.block_start, row_slot, row_jac, self.slot_inv_mass],
            outputs=[self.W],
            device=self.device,
        )
        combine_mimic_matrix(self.W, self.mimic_block, self.mimic_local, self.mimic_coeffs, self.device)
        n_law = self.law_rows.shape[0]
        if n_law:
            inputs = [self.law_rows, self.row_dof, self.row_block, self.row_local, self.W, self.law_terms, friction]
            wp.launch(joint_law_start_kernel, n_law, inputs, [self.compliance], device=self.device)
        R = self.row_block.shape[0]
        inputs = [self.row_block, self.row_local, self.compliance]
        wp.launch(bilateral_diagonal_kernel, R, inputs, [self.diagonal], device=self.device)
        self.refactor_blocks.fill_(1)
        self.refactor()

    def refactor(self) -> None:
        """Factor ``W`` with the rows' compliance on its diagonal, in the blocks marked in ``refactor_blocks``."""
        wp.launch_tiled(
            _tile_kernels(self.tile)[0],
            dim=self.W.shape[0],
            inputs=[self.W, self.diagonal, self.refactor_blocks],
            outputs=[self.L],
            block_dim=64,
            device=self.device,
        )

    def axis_law(
        self,
        residual: wp.array[wp.float64],
        target: wp.array[wp.float64],
        force: wp.array[wp.float64],
        effort_limit: wp.array[wp.float32],
        friction: wp.array[wp.float32],
    ) -> None:
        """Set the axis rows' compliance and target motion for a Newton step from their forces ``force`` and the
        ``residual`` of their target motion ``target``, both updated, by :func:`joint_axis_law_kernel`, and factor the
        blocks whose compliance changed again."""
        n_law = self.law_rows.shape[0]
        if not n_law:
            return
        inputs = [self.law_rows, self.row_dof, self.row_block, self.row_local, self.W, self.law_terms]
        inputs += [effort_limit, friction, force]
        outputs = [self.compliance, target, residual, self.diagonal, self.refactor_blocks]
        wp.launch(joint_axis_law_kernel, n_law, inputs, outputs, device=self.device)
        self.refactor()

    def update(self, residual: wp.array[wp.float64], force: wp.array[wp.float64]) -> None:
        """Newton step on the rows: ``force += W⁻¹ (-residual)``, the soft rows' term included, from the residual with
        its mimic pairs combined (:func:`mimic_combine_kernel`); the leaders' rows take ``-c1`` times their
        followers' forces."""
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
        if self.mimic_rows.shape[0]:
            inputs = [self.mimic_rows, self.mimic_coeffs, 0]
            wp.launch(mimic_force_kernel, self.mimic_rows.shape[0], inputs, [force], device=self.device)

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
