# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body joints for SolverFBA.

Each joint becomes bilateral rows on the integrated bodies, one per locked
direction: the complement of its free linear axes locks the anchor offset, and
the complement of its free angular axes locks the relative rotation. Directions
are expressed in the joint frame carried by the parent body.

Each free axis with passive damping (``Model.joint_damping``), a drive
(``Model.joint_target_ke`` or ``Model.joint_target_kd``) or armature
(``Model.joint_armature``) adds a soft row on the axis's motion over the step,
the backward Euler step of their forces, and each finite limit of the free axis
of a revolute or prismatic joint (``Model.joint_limit_lower`` and
``Model.joint_limit_upper``) a candidate for a unilateral limit row, which
joins the step's rigid contact rows when the axis comes near the limit.

Two kinds of rows hold a force within a box, by a box law in each Newton step
(:func:`joint_box_law_kernel`): the soft row of a driven axis whose effort limit
(``Model.joint_effort_limit``) is below 1e6 bounds its drive's force, and each
free axis with joint friction (``Model.joint_friction``) adds a row on its
motion whose force the friction bounds.

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

BOX_NONE = wp.constant(0)
"""A row without a box law."""
BOX_EFFORT = wp.constant(1)
"""A soft row whose drive force ``ke (q* - q) + kd (q_dot* - q_dot)`` the DOF's effort limit bounds; its damping and
armature stay unbounded."""
BOX_FRICTION = wp.constant(2)
"""A row on the motion along a free axis that holds it still, with the force the DOF's friction bounds."""
BOX_FREEZE = wp.constant(1.0e6)
"""Compliance of a row held at the bound of its box, relative to its ``W_ii`` plus its compliance off the bound: a
Newton step lands its force on the bound."""
FRICTION_STICK = wp.constant(1.0e-4)
"""Compliance of a friction row that holds its axis, relative to its ``W_ii``: next to a limit row on the same axis,
the two rows' system stays nonsingular."""
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
    row_box: wp.array[wp.int32]
    """:data:`BOX_NONE`, :data:`BOX_EFFORT` or :data:`BOX_FRICTION` per row, shape [row_count]."""
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
    zero = np.zeros(3)
    # (block, local joint index, angular, direction, kind, damping, second axis, dof, coordinate, mimic pair and side,
    # box law)
    rows = []
    for k, j in enumerate(joints):
        n_lin, n_ang = dof_dim[j]
        axes = np.array(axis[qd_start[j] : qd_start[j] + n_lin + n_ang], dtype=np.float64)
        block = find(max(pslot[j], cslot[j]))
        rows += [(block, k, 0, d, ROW_LOCKED, 0.0, zero, -1, -1, -1, BOX_NONE) for d in locked_directions(axes[:n_lin])]
        kinds = [ROW_DAMPER] * (n_lin + n_ang)
        if n_ang == 2:
            # The rotation q(a0, q0) q(a1, q1) keeps the angle between a0 in the parent's joint frame and a1 in the
            # child's, and turns about a1 in the child's frame; the axes as Newton orthonormalizes them.
            axes[n_lin:] = hooke_axes(axes[n_lin], axes[n_lin + 1])
            rows.append((block, k, 1, axes[n_lin], ROW_HOOKE, 0.0, axes[n_lin + 1], -1, -1, -1, BOX_NONE))
            kinds[n_lin + 1] = ROW_DAMPER_CHILD
        else:
            rows += [
                (block, k, 1, d, ROW_LOCKED, 0.0, zero, -1, -1, -1, BOX_NONE) for d in locked_directions(axes[n_lin:])
            ]
        # Each free axis takes its passive damping, drive and armature as a soft row, the drive bounded by an effort
        # limit, and its friction as a row of its own.
        for i in range(n_lin + n_ang):
            dof = qd_start[j] + i
            d = axes[i] / np.linalg.norm(axes[i])
            if friction[dof] > 0.0:
                rows.append((block, k, int(i >= n_lin), d, kinds[i], 0.0, zero, dof, -1, -1, BOX_FRICTION))
            if max(damping[dof], armature[dof], target_ke[dof], target_kd[dof]) <= 0.0:
                continue
            coord = -1
            if target_ke[dof] > 0.0:
                if i >= n_lin and n_ang > 1:
                    raise NotImplementedError(
                        f"SolverFBA drives the angle of a joint with one free angular axis only; joint {j} has "
                        f"{n_ang} and a position gain on DOF {dof}"
                    )
                coord = q_start[j] + i
            driven = max(target_ke[dof], target_kd[dof]) > 0.0
            box = BOX_EFFORT if driven and effort[dof] < EFFORT_UNLIMITED else BOX_NONE
            rows.append((block, k, int(i >= n_lin), d, kinds[i], float(damping[dof]), zero, dof, coord, -1, box))
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
                    (block, local[joint], int(i >= n_lin), d, ROW_MIMIC, 0.0, zero, -1, -1, 2 * pair + side, BOX_NONE)
                )
            pair_coeffs.append((c0, c1))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    blocks = np.array([r[0] for r in rows])
    block_start = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1], True]).astype(np.int32)
    slots = np.array([find(i) for i in range(len(root))])
    # A mimic pair locks one direction, its follower's row; the leader's row is inert.
    locked = np.array([r[4] in (ROW_LOCKED, ROW_HOOKE) or (r[4] == ROW_MIMIC and r[9] % 2 == 0) for r in rows])
    # The soft rows: the dampers and the friction rows, which take a compliance on W's diagonal.
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
        row_box=wp.array([r[10] for r in rows], dtype=wp.int32, device=device),
        block_start=block_start,
        joint_index=joints.astype(np.int32),
        mimic_rows=mimic_rows,
        mimic_coeffs=np.array(pair_coeffs, dtype=np.float64).reshape(-1, 2),
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

    Revolute limits whose range spans a full turn or more, such as the +-2 pi of many arm descriptions, are left out
    with a :class:`~newton.exceptions.NewtonWarning`: the axis's angle is known up to whole turns only, and lies within such a
    range at every turn.

    Raises:
        NotImplementedError: If a revolute axis has a single finite limit: its angle is known up to whole turns only.
        ValueError: If an axis's lower limit exceeds its upper one.
    """
    jtype, qd_start, axis = model.joint_type.numpy(), model.joint_qd_start.numpy(), model.joint_axis.numpy()
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    parent, child = rows.parent.numpy(), rows.child.numpy()
    world = model.body_world.numpy()
    limits = []  # (joint, angular, axis, bound, sign, center, margin)
    turns = []
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
            if not (has_lo and has_hi):
                raise NotImplementedError(
                    f"SolverFBA limits revolute joints with both limits finite; joint {j} has limits ({lo}, {hi})"
                )
            if hi - lo >= 2.0 * np.pi:
                # An angle known up to whole turns lies within such a range at every turn.
                turns.append(int(j))
                continue
            center, margin = 0.5 * (lo + hi), LIMIT_MARGIN_ANGULAR
        a = np.asarray(axis[dof], dtype=np.float64)
        a = a / np.linalg.norm(a)
        if has_lo:
            limits.append((k, angular, a, lo, 1.0, center, margin))
        if has_hi:
            limits.append((k, angular, a, hi, -1.0, center, margin))
    if turns:
        warnings.warn(
            f"SolverFBA does not enforce the limits of revolute joints {turns}, whose ranges span a full turn or more: "
            "it knows their angles up to whole turns",
            NewtonWarning,
            stacklevel=2,
        )
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
    row_box: wp.array[wp.int32],
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
    dt: wp.float64,
    # outputs
    compliance: wp.array[wp.float64],
    target: wp.array[wp.float64],
    terms: wp.array2d[wp.float64],
):
    """Each soft row's compliance and target motion over a step of ``dt`` from the poses and velocities at the
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

    A row of :data:`BOX_EFFORT` also keeps its drive's part ``b_a - k_a g`` (``k_a = ke + kd / dt``,
    ``b_a = ke (q* - q0) + kd q_dot*``) and the rest ``b_p - k_p g`` in ``terms[r] = (k_a, b_a, k_p, b_p, C, g*)``
    for :func:`joint_box_law_kernel`; a friction row (:data:`BOX_FRICTION`) takes its compliance and target there.
    """
    r = wp.tid()
    if not _is_damper(row_kind[r]) or row_box[r] == BOX_FRICTION:
        compliance[r] = wp.float64(0.0)
        target[r] = wp.float64(0.0)
        return
    dof = row_dof[r]
    c = wp.float64(joint_damping[dof])
    arm = wp.float64(joint_armature[dof])
    ke = wp.float64(joint_target_ke[dof])
    kd = wp.float64(joint_target_kd[dof])
    if ke == wp.float64(0.0) and arm == wp.float64(0.0) and kd == wp.float64(0.0):
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
    compliance[r] = dt * dt / stiffness
    target[r] = force * dt * dt / stiffness
    if row_box[r] == BOX_EFFORT:
        terms[r, 0] = ke + kd / dt
        terms[r, 1] = drive
        terms[r, 2] = c / dt + arm / (dt * dt)
        terms[r, 3] = arm * rate / dt
        terms[r, 4] = compliance[r]
        terms[r, 5] = target[r]


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
    body_q_end: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # outputs
    near: wp.array[wp.int32],
):
    """1 for each limit whose row joins the step, else 0: one whose gap at the step start (``body_q``, read for every
    body through ``no_slot``, -1 per body) or after the step's free flight (``slot_x``, ``slot_q``, and the other
    bodies at their step-end poses ``body_q_end``) falls within its margin plus the free flight's change of the
    gap."""
    k = wp.tid()
    g0 = limit_gap(
        k, joint, angular, axis, bound, sign, center, joint_parent, joint_child, X_pj, X_cj, no_slot,
        slot_x, slot_q, body_q, body_com,
    )  # fmt: skip
    g1 = limit_gap(
        k, joint, angular, axis, bound, sign, center, joint_parent, joint_child, X_pj, X_cj, body_slot,
        slot_x, slot_q, body_q_end, body_com,
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
    and the box rows' (:func:`joint_box_law_kernel`); none without soft rows (an empty ``compliance``)."""
    r = wp.tid()
    if compliance.shape[0] > 0:
        diagonal[row_block[r], row_local[r]] = compliance[r]


@wp.kernel
def joint_box_start_kernel(
    box_rows: wp.array[wp.int32],
    row_box: wp.array[wp.int32],
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    W: wp.array3d[wp.float64],
    # outputs
    compliance: wp.array[wp.float64],
    state: wp.array[wp.int32],
):
    """Start each box row ``box_rows[i]`` of a step off its bounds: a friction row holding its axis, at the
    compliance :data:`FRICTION_STICK` times its ``W_ii`` and no target motion (:func:`joint_soft_row_kernel`'s), a
    drive at the compliance and target of :func:`joint_soft_row_kernel`."""
    i = wp.tid()
    r = box_rows[i]
    state[r] = 0
    if row_box[r] == BOX_FRICTION:
        compliance[r] = wp.float64(FRICTION_STICK) * W[row_block[r], row_local[r], row_local[r]]


@wp.kernel
def joint_box_law_kernel(
    box_rows: wp.array[wp.int32],
    row_box: wp.array[wp.int32],
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
    state: wp.array[wp.int32],
    diagonal: wp.array2d[wp.float64],
    refactor: wp.array[wp.int32],
):
    """The box law of each box row ``box_rows[i]`` for a Newton step from its force ``f`` and its axis's motion ``g``
    since the step's start (its ``residual`` plus ``target``): the row's compliance ``c`` and target motion ``g*``,
    which its ``residual`` then measures against, so that the step solves ``W df + c df = -(g - g* + c f)``.

    With ``d = W_ii``, one step of the row alone predicts its bounded force ``y`` off the bounds: a friction row's
    ``f - g / (d + C)`` (``C = FRICTION_STICK d``), a drive's ``b_a - k_a g'`` at the motion
    ``g' = g - d (g - g0* + C0 f) / (d + C0)`` (``terms = (k_a, b_a, k_p, b_p, C0, g0*)`` of
    :func:`joint_soft_row_kernel`). Within ``[-F, F]`` the row keeps its law off the bound: a friction row holds its
    axis (``c = C``, ``g* = C f``, so that the step drives ``g`` to zero, ``C`` regularizing only the step), a drive
    its soft row (``C0``, ``g0*``). Past ``s F`` (``s = ±1``) the row takes its law at the bound, the force
    ``s F + b_p - k_p g`` (``b_p = k_p = 0`` for friction): ``c = 1 / k`` with ``k = max(k_p, 1 / (BOX_FREEZE (d + C)))``
    and ``g* = g + (s F + b_p - k_p g) / k``, so that the step lands the force on ``s F + b_p - k_p g`` up to the
    step's own motion over ``BOX_FREEZE (d + C)``. A row whose side changes marks its block for
    :meth:`BilateralSystem.refactor`."""
    i = wp.tid()
    r = box_rows[i]
    k = row_block[r]
    local = row_local[r]
    d = W[k, local, local]
    old_target = target[r]
    motion = residual[r] + old_target
    f = force[r]
    side = 0
    soft = wp.float64(0.0)
    offset = wp.float64(0.0)
    passive = wp.float64(0.0)
    c = wp.float64(0.0)
    goal = wp.float64(0.0)
    if row_box[r] == BOX_FRICTION:
        bound = wp.float64(friction[row_dof[r]])
        soft = wp.float64(FRICTION_STICK) * d
        y = f - motion / (d + soft)
        c = soft
        goal = soft * f
    else:
        bound = wp.float64(effort_limit[row_dof[r]])
        soft = terms[r, 4]
        c = soft
        goal = terms[r, 5]
        y = terms[r, 1] - terms[r, 0] * (motion - d * (motion - goal + soft * f) / (d + soft))
        passive = terms[r, 2]
        offset = terms[r, 3]
    if y > bound:
        side = 1
    elif y < -bound:
        side = -1
    if side != 0:
        stiffness = wp.max(passive, wp.float64(1.0) / (wp.float64(BOX_FREEZE) * (d + soft)))
        c = wp.float64(1.0) / stiffness
        goal = motion + (wp.float64(side) * bound + offset - passive * motion) / stiffness
    compliance[r] = c
    diagonal[k, local] = c
    if goal != old_target:
        target[r] = goal
        residual[r] = motion - goal
    if side != state[r]:
        state[r] = side
        refactor[k] = 1


@wp.kernel
def joint_friction_clamp_kernel(
    box_rows: wp.array[wp.int32],
    row_box: wp.array[wp.int32],
    row_dof: wp.array[wp.int32],
    friction: wp.array[wp.float32],
    # outputs
    force: wp.array[wp.float64],
):
    """Clamp the force of each friction row among ``box_rows`` to its box ``[-F, F]``."""
    i = wp.tid()
    r = box_rows[i]
    if row_box[r] == BOX_FRICTION:
        bound = wp.float64(friction[row_dof[r]])
        force[r] = wp.clamp(force[r], -bound, bound)


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
    the compliance ``C`` and its residual the term ``C f``. A box row (a drive
    with an effort limit, a friction row) takes its compliance and target motion
    from its box law at each Newton step (:meth:`box_law`), and the blocks whose
    box rows change sides are factored again.
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
        # The blocks the next factor takes: all of them at the step's start, those whose box rows changed sides at a
        # Newton step.
        self.refactor_blocks = wp.zeros(n_blocks, dtype=wp.int32, device=device)
        # The box rows: their law, model DOF, side of their box (-1, 0, 1) and drive terms (joint_soft_row_kernel).
        self.row_box, self.row_dof = rows.row_box, rows.row_dof
        box = rows.row_box.numpy()
        self.box_rows = wp.array(np.flatnonzero(box != BOX_NONE).astype(np.int32), dtype=wp.int32, device=device)
        self.has_friction = bool((box == BOX_FRICTION).any())
        self.box_state = wp.zeros(R if self.box_rows.shape[0] else 0, dtype=wp.int32, device=device)
        self.box_terms = wp.zeros((R if (box == BOX_EFFORT).any() else 0, 6), dtype=wp.float64, device=device)
        # The mimic pairs: their rows, coefficients, block and rows within it.
        pairs = np.asarray(rows.mimic_rows, dtype=np.int32).reshape(-1, 2)
        self.mimic_rows = wp.array(pairs, dtype=wp.int32, device=device)
        self.mimic_coeffs = wp.array(np.asarray(rows.mimic_coeffs).reshape(-1, 2), dtype=wp.float64, device=device)
        self.mimic_block = wp.array(row_block[pairs[:, 0]], dtype=wp.int32, device=device)
        self.mimic_local = wp.array(
            (pairs - rows.block_start[row_block[pairs]]).astype(np.int32), dtype=wp.int32, device=device
        )

    def factor(self, row_slot: wp.array2d[wp.int32], row_jac: wp.array2d[wp.spatial_vectord]) -> None:
        """Assemble ``W`` for this step's Jacobian, with the mimic pairs combined (:func:`combine_mimic_matrix`), and
        factor it with the soft rows' compliance on its diagonal, the box rows off their bounds
        (:func:`joint_box_start_kernel`)."""
        wp.launch(
            bilateral_delassus_kernel,
            dim=self.W.shape,
            inputs=[self.block_start, row_slot, row_jac, self.slot_inv_mass],
            outputs=[self.W],
            device=self.device,
        )
        combine_mimic_matrix(self.W, self.mimic_block, self.mimic_local, self.mimic_coeffs, self.device)
        n_box = self.box_rows.shape[0]
        if n_box:
            inputs = [self.box_rows, self.row_box, self.row_block, self.row_local, self.W]
            outputs = [self.compliance, self.box_state]
            wp.launch(joint_box_start_kernel, n_box, inputs, outputs, device=self.device)
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

    def box_law(
        self,
        residual: wp.array[wp.float64],
        target: wp.array[wp.float64],
        force: wp.array[wp.float64],
        effort_limit: wp.array[wp.float32],
        friction: wp.array[wp.float32],
    ) -> None:
        """Set the box rows' compliance and target motion for a Newton step from their forces ``force`` and the
        ``residual`` of their target motion ``target``, both updated, by :func:`joint_box_law_kernel`, and factor the
        blocks whose box rows changed sides again."""
        n_box = self.box_rows.shape[0]
        if not n_box:
            return
        inputs = [self.box_rows, self.row_box, self.row_dof, self.row_block, self.row_local, self.W, self.box_terms]
        inputs += [effort_limit, friction, force]
        outputs = [self.compliance, target, residual, self.box_state, self.diagonal, self.refactor_blocks]
        wp.launch(joint_box_law_kernel, n_box, inputs, outputs, device=self.device)
        self.refactor()

    def clamp_friction(self, force: wp.array[wp.float64], friction: wp.array[wp.float32]) -> None:
        """Clamp the friction rows' forces ``force`` to their boxes."""
        if self.has_friction:
            inputs = [self.box_rows, self.row_box, self.row_dof, friction]
            wp.launch(joint_friction_clamp_kernel, self.box_rows.shape[0], inputs, [force], device=self.device)

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
