# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body joints for SolverFBA.

Each joint becomes bilateral rows on the integrated bodies, one per locked
direction: the complement of its free linear axes locks the anchor offset, and
the complement of its free angular axes locks the relative rotation. Directions
are expressed in the joint frame carried by the parent body.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import JointType, Model

_CONSTRAINT_JOINTS = (JointType.PRISMATIC, JointType.REVOLUTE, JointType.BALL, JointType.FIXED, JointType.D6)


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
    row_dir: wp.array[wp.vec3]
    """Locked direction in the joint frame, shape [row_count]."""
    block_start: np.ndarray
    """Row range per independent block, shape [block_count + 1]."""


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
        NotImplementedError: If such a joint is a distance or cable joint.
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
        raise NotImplementedError(f"SolverFBA does not support {JointType(t).name} joints")

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
    rows = []  # (block, local joint index, angular, direction)
    for k, j in enumerate(joints):
        n_lin, n_ang = dof_dim[j]
        axes = axis[qd_start[j] : qd_start[j] + n_lin + n_ang]
        block = find(max(pslot[j], cslot[j]))
        rows += [(block, k, 0, d) for d in locked_directions(axes[:n_lin])]
        rows += [(block, k, 1, d) for d in locked_directions(axes[n_lin:])]
    rows.sort(key=lambda r: r[0])
    blocks = np.array([r[0] for r in rows])
    block_start = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1], True]).astype(np.int32)

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
        row_dir=wp.array(np.array([r[3] for r in rows]), dtype=wp.vec3, device=device),
        block_start=block_start,
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


@wp.kernel
def joint_residual_kernel(
    row_joint: wp.array[wp.int32],
    row_angular: wp.array[wp.int32],
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
    residual: wp.array[wp.float64],
):
    """Violation of each row at the live pose: the anchor offset ``R_Jaᵀ (x_Jc - x_Ja)``
    or the rotation ``2 nu`` along the locked direction, zero when satisfied."""
    r = wp.tid()
    j = row_joint[r]
    xa, qa = _com_frame(joint_parent[j], body_slot, slot_x, slot_q, body_q, body_com)
    xc, qc = _com_frame(joint_child[j], body_slot, slot_x, slot_q, body_q, body_com)
    q_ja = qa * wp.quatd(wp.transform_get_rotation(X_pj[j]))
    b = wp.vec3d(row_dir[r])
    if row_angular[r] == 1:
        q_rel = _relative_rotation(q_ja, qc * wp.quatd(wp.transform_get_rotation(X_cj[j])))
        residual[r] = wp.float64(2.0) * wp.dot(b, wp.vec3d(q_rel[0], q_rel[1], q_rel[2]))
    else:
        x_ja = xa + wp.quat_rotate(qa, wp.vec3d(wp.transform_get_translation(X_pj[j])))
        x_jc = xc + wp.quat_rotate(qc, wp.vec3d(wp.transform_get_translation(X_cj[j])))
        residual[r] = wp.dot(b, wp.quat_rotate_inv(q_ja, x_jc - x_ja))
