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
from .rigid import row_force_rhs_kernel, row_incidence, slot_coefficients

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
def joint_jacobian_kernel(
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
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
):
    """Row coefficients on the DOFs ``(chi, rho)`` of the parent and child body.

    Evaluated at the pinned frame. A translational row along ``d = R_Ja b`` acts
    on the anchor velocities; a rotational row uses the direction
    ``d = R_Ja (eta b + nu x b)`` of the relative rotation ``(eta, nu)``, which
    makes it the exact derivative of the residual ``2 nu · b``. World
    coefficients are rotated into each body's pinned frame.
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
def bilateral_rhs_kernel(
    row_block: wp.array[wp.int32],
    row_local: wp.array[wp.int32],
    residual: wp.array[wp.float64],
    # outputs
    rhs: wp.array2d[wp.float64],
):
    r = wp.tid()
    rhs[row_block[r], row_local[r]] = -residual[r]


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
    """``rhs[block, local] = v[offset + k]`` for joint row ``rows[k]``."""
    k = wp.tid()
    rhs[row_block[rows[k]], row_local[rows[k]]] = v[offset + k]


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
    """``out[offset + k] = x[block, local]`` for joint row ``rows[k]``."""
    k = wp.tid()
    out[offset + k] = x[row_block[rows[k]], row_local[rows[k]]]


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

    def factor(self, row_slot: wp.array2d[wp.int32], row_jac: wp.array2d[wp.spatial_vectord]) -> None:
        """Assemble and factor ``W`` for this step's Jacobian."""
        wp.launch(
            bilateral_delassus_kernel,
            dim=self.W.shape,
            inputs=[self.block_start, row_slot, row_jac, self.slot_inv_mass],
            outputs=[self.W],
            device=self.device,
        )
        wp.launch_tiled(
            _tile_kernels(self.tile)[0], dim=self.W.shape[0], inputs=[self.W, self.L], block_dim=64, device=self.device
        )

    def update(self, residual: wp.array[wp.float64], force: wp.array[wp.float64]) -> None:
        """Newton step on hard rows: ``force += W⁻¹ (-residual)``."""
        R = residual.shape[0]
        wp.launch(
            bilateral_rhs_kernel,
            dim=R,
            inputs=[self.row_block, self.row_local, residual],
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

        ``rows`` must cover whole blocks; the factor comes from :meth:`factor`.
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
        wp.launch(
            row_force_rhs_kernel,
            dim=self.slot_entry_start.shape[0] - 1,
            inputs=[self.slot_entry_start, self.entry_row, self.entry_side, row_jac, force, b_const],
            outputs=[b],
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
    """``dst[offset + k] = src[index[k]]``."""
    k = wp.tid()
    dst[offset + k] = src[index[k]]


@wp.kernel
def scatter_values_kernel(
    src: wp.array[wp.float64],
    offset: int,
    index: wp.array[wp.int32],
    # outputs
    dst: wp.array[wp.float64],
):
    """``dst[index[k]] = src[offset + k]``."""
    k = wp.tid()
    dst[index[k]] = src[offset + k]
