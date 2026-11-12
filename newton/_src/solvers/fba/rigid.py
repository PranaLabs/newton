# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body kernels for SolverFBA.

Each integrated body owns six DOFs ``u = (chi, rho)`` in the PD system: ``chi``
is the COM position and ``rho = dt * omega`` the step rotation vector, both in
the body frame pinned once per step, ``R_pin = exp(dt [omega_prev]x) R_prev``.
In that frame the rigid block ``diag(m I3, I) / dt^2`` stays constant (see
:func:`~newton._src.solvers.fba.linear_solver.build_rigid_system`). Poses are
retracted from the step-start pose. Velocities follow Newton's convention,
``body_qd = (v_com, omega)`` in world frame, and ``body_q`` is the body origin.

A constraint row acts on at most two integrated bodies. Side ``k`` of row ``r``
names the body slot ``row_slot[r, k]`` (-1 for none) and its coefficients
``row_jac[r, k]`` on that slot's DOFs.
"""

import numpy as np
import warp as wp

# Stability clamp on the rotation per step, |omega| * dt <= 0.5 [rad].
_MAX_STEP_ROTATION = wp.constant(wp.float64(0.5))
# Step rotations with squared norm below this are treated as numerical noise.
_ROTATION_NOISE_SQ = wp.constant(wp.float64(1.0e-28))


@wp.func
def _clamp_step_rotation(omega: wp.vec3d, dt: wp.float64) -> wp.vec3d:
    n = wp.length(omega)
    if n * dt > _MAX_STEP_ROTATION:
        return omega * (_MAX_STEP_ROTATION / (n * dt))
    return omega


@wp.func
def _rotate_by(omega: wp.vec3d, dt: wp.float64, q: wp.quatd) -> wp.quatd:
    """Apply the world-frame rotation ``exp(dt [omega]x)`` to ``q``."""
    angle = wp.length(omega) * dt
    dq = wp.quat_from_axis_angle(omega / wp.length(omega), angle)
    return wp.normalize(dq * q)


@wp.func
def slot_coefficients(lin: wp.vec3d, ang: wp.vec3d) -> wp.spatial_vectord:
    """Row coefficients on the DOFs ``(chi, rho)`` of a slot from their two halves."""
    return wp.spatial_vectord(lin[0], lin[1], lin[2], ang[0], ang[1], ang[2])


@wp.kernel
def prepare_rigid_step_kernel(
    bodies: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33],
    body_world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    dt: wp.float64,
    # outputs
    q_pin: wp.array[wp.quatd],
    x_prev: wp.array[wp.vec3d],
    b: wp.array[wp.float64],
):
    """Pin each body's solve frame and assemble its step-constant RHS.

    ``b`` is ``A_rigid · u_free`` for the free-flight step ``u_free``:
    ``b_trans = R_pinᵀ (m (x + dt v + dt² g) / dt² + f)`` and
    ``b_rot = I w / dt - cross(w, I w) + R_pinᵀ τ`` with ``w = R_pinᵀ ω`` in
    the body frame, where ``-cross(w, I w)`` is the explicit gyroscopic torque.
    """
    k = wp.tid()
    i = bodies[k]
    tf = body_q[i]
    q0 = wp.normalize(wp.quatd(wp.transform_get_rotation(tf)))
    x0 = wp.vec3d(wp.transform_get_translation(tf)) + wp.quat_rotate(q0, wp.vec3d(body_com[i]))
    v0 = wp.vec3d(wp.spatial_top(body_qd[i]))
    w0 = wp.vec3d(wp.spatial_bottom(body_qd[i]))

    w_pin = _clamp_step_rotation(w0, dt)
    qp = q0
    if dt * dt * wp.length_sq(w_pin) >= _ROTATION_NOISE_SQ:
        qp = _rotate_by(w_pin, dt, q0)
    q_pin[k] = qp
    x_prev[k] = x0

    g = wp.vec3d(gravity[wp.max(body_world[i], 0)])
    f = wp.vec3d(wp.spatial_top(body_f[i]))
    tau = wp.vec3d(wp.spatial_bottom(body_f[i]))
    m = wp.float64(body_mass[i])
    inertia = wp.mat33d(body_inertia[i])

    b_trans = wp.quat_rotate_inv(qp, m / (dt * dt) * (x0 + dt * v0 + dt * dt * g) + f)
    w = wp.quat_rotate_inv(qp, w0)
    b_rot = inertia * w / dt - wp.cross(w, inertia * w) + wp.quat_rotate_inv(qp, tau)
    for c in range(3):
        b[6 * k + c] = b_trans[c]
        b[6 * k + 3 + c] = b_rot[c]


@wp.kernel
def retract_rigid_kernel(
    bodies: wp.array[wp.int32],
    u: wp.array[wp.float64],
    q_pin: wp.array[wp.quatd],
    x_prev: wp.array[wp.vec3d],
    body_q_prev: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    dt: wp.float64,
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    x_live: wp.array[wp.vec3d],
    q_live: wp.array[wp.quatd],
):
    """Map the solved DOFs back to body poses and velocities.

    The COM moves to ``R_pin chi``; the orientation is retracted from the
    step-start one, ``q = exp(R_pin rho) q_prev``, with the rotation clamped
    like the pinned frame. Velocities are ``v = (x - x_prev) / dt`` and
    ``ω = R_pin rho / dt``. The COM pose is also kept in float64 per slot.
    """
    k = wp.tid()
    i = bodies[k]
    qp = q_pin[k]
    chi = wp.vec3d(u[6 * k], u[6 * k + 1], u[6 * k + 2])
    rho = wp.vec3d(u[6 * k + 3], u[6 * k + 4], u[6 * k + 5])

    x = wp.quat_rotate(qp, chi)
    q = wp.normalize(wp.quatd(wp.transform_get_rotation(body_q_prev[i])))
    omega = wp.vec3d(0.0, 0.0, 0.0)
    if wp.length_sq(rho) >= _ROTATION_NOISE_SQ:
        omega = wp.quat_rotate(qp, _clamp_step_rotation(rho / dt, dt))
        q = _rotate_by(omega, dt, q)

    v = (x - x_prev[k]) / dt
    x_live[k] = x
    q_live[k] = q
    origin = x - wp.quat_rotate(q, wp.vec3d(body_com[i]))
    body_q[i] = wp.transform(wp.vec3(origin), wp.quat(q))
    body_qd[i] = wp.spatial_vector(wp.vec3(v), wp.vec3(omega))


def row_incidence(row_slot: np.ndarray, slot_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows touching each body slot, as a CSR over ``(row, side)`` entries.

    Args:
        row_slot: Body slot of both sides of each row, -1 for none, shape [row_count, 2].
        slot_count: Number of body slots.

    Returns:
        ``(slot_entry_start, entry_row, entry_side)``; the entries of slot ``k``
        are ``slot_entry_start[k]:slot_entry_start[k + 1]``.
    """
    n = row_slot.shape[0]
    slots, rows, sides = row_slot.ravel(), np.repeat(np.arange(n), 2), np.tile([0, 1], n)
    keep = slots >= 0
    order = np.argsort(slots[keep], kind="stable")
    start = np.searchsorted(slots[keep][order], np.arange(slot_count + 1))
    return start.astype(np.int32), rows[keep][order].astype(np.int32), sides[keep][order].astype(np.int32)


@wp.kernel
def solve_rigid_blocks_kernel(
    slot_inv_mass: wp.array[wp.spatial_matrixd],
    b: wp.array[wp.float64],
    # outputs
    x: wp.array[wp.float64],
):
    """``x = A⁻¹ b`` for the block diagonal rigid system, one body slot per thread."""
    k = wp.tid()
    rhs = wp.spatial_vectord()
    for c in range(6):
        rhs[c] = b[6 * k + c]
    out = slot_inv_mass[k] * rhs
    for c in range(6):
        x[6 * k + c] = out[c]


class RigidBlockSolver:
    """Solves the rigid system of :func:`~.linear_solver.build_rigid_system`, whose bodies do not couple, with the
    inverse of each body's 6x6 block."""

    def __init__(self, slot_inv_mass: wp.array[wp.spatial_matrixd]):
        self.slot_inv_mass = slot_inv_mass

    def solve_scalar(self, b: wp.array[wp.float64], x: wp.array[wp.float64]) -> None:
        """Compute ``x = A⁻¹ b``; ``x`` must not alias ``b``."""
        inv = self.slot_inv_mass
        wp.launch(solve_rigid_blocks_kernel, inv.shape[0], inputs=[inv, b], outputs=[x], device=inv.device)


@wp.kernel
def _slot_entries_kernel(
    row_slot: wp.array2d[wp.int32],
    slot_count: int,
    distinct: int,
    # outputs
    keys: wp.array[wp.int32],
    values: wp.array[wp.int32],
):
    """Entry ``2 r + side`` of each row side keyed by its body slot, or by ``slot_count`` for a side without a slot
    and, with ``distinct``, for a second side on the first side's slot."""
    i = wp.tid()
    r = i // 2
    side = i - 2 * r
    s = row_slot[r, side]
    if s < 0 or (distinct != 0 and side == 1 and s == row_slot[r, 0]):
        s = slot_count
    keys[i] = s
    values[i] = i


@wp.kernel
def _slot_starts_kernel(keys: wp.array[wp.int32], count: int, starts: wp.array[wp.int32]):
    """The first of the ``count`` sorted entries whose slot is at least ``k``."""
    k = wp.tid()
    lo = int(0)
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < k:
            lo = mid + 1
        else:
            hi = mid
    starts[k] = lo


@wp.kernel
def _split_entries_kernel(
    values: wp.array[wp.int32],
    # outputs
    entry_row: wp.array[wp.int32],
    entry_side: wp.array[wp.int32],
):
    i = wp.tid()
    row = values[i] // 2
    entry_row[i] = row
    entry_side[i] = values[i] - 2 * row


def slot_entries(
    row_slot: wp.array2d[wp.int32], rows: int, slot_count: int, distinct: bool = False
) -> tuple[wp.array, wp.array, wp.array]:
    """:func:`row_incidence` of the first ``rows`` rows of a device ``row_slot``, computed on its device.

    The entries of slot ``k`` are ``start[k]:start[k + 1]``, ascending by row and then side, as
    :func:`row_incidence` orders them; ``entry_row`` and ``entry_side`` hold ``2 * rows`` entries, of which those from
    ``start[slot_count]`` on belong to no slot. With ``distinct``, a row enters each slot's entries once, as the
    rows of a group do.
    """
    device = row_slot.device
    start = wp.zeros(slot_count + 1, dtype=wp.int32, device=device)
    count = 2 * rows
    entry_row = wp.empty(max(count, 1), dtype=wp.int32, device=device)
    entry_side = wp.empty(max(count, 1), dtype=wp.int32, device=device)
    if count == 0:
        return start, entry_row, entry_side
    # radix_sort_pairs sorts in place in arrays twice the sorted length.
    keys = wp.empty(2 * count, dtype=wp.int32, device=device)
    values = wp.empty(2 * count, dtype=wp.int32, device=device)
    inputs = [row_slot, slot_count, int(distinct)]
    wp.launch(_slot_entries_kernel, count, inputs=inputs, outputs=[keys, values], device=device)
    wp.utils.radix_sort_pairs(keys, values, count)
    wp.launch(_slot_starts_kernel, slot_count + 1, inputs=[keys, count], outputs=[start], device=device)
    wp.launch(_split_entries_kernel, count, inputs=[values], outputs=[entry_row, entry_side], device=device)
    return start, entry_row, entry_side


# Lanes of :func:`row_force_rhs_kernel` per body slot.
ROW_FORCE_LANES = 32


# Most body slots of a block of three rows: two per row.
BLOCK_GROUPS = wp.constant(6)


@wp.kernel
def block_slot_groups_kernel(
    row_slot: wp.array2d[wp.int32],
    # outputs
    groups: wp.array2d[wp.int32],
):
    """The body slots of each block of three rows, once each in their order, -1 after them."""
    b = wp.tid()
    count = int(0)
    for a in range(3):
        for x in range(2):
            s = row_slot[3 * b + a, x]
            if s >= 0:
                seen = int(0)
                for g in range(count):
                    if groups[b, g] == s:
                        seen = 1
                if seen == 0:
                    groups[b, count] = s
                    count += 1
    for g in range(count, BLOCK_GROUPS):
        groups[b, g] = -1


@wp.kernel
def _group_entries_kernel(
    groups: wp.array2d[wp.int32],
    slot_count: int,
    # outputs
    keys: wp.array[wp.int32],
    values: wp.array[wp.int32],
):
    """Entry ``G r + g`` of each row's groups keyed by its group, or by ``slot_count`` where it has none."""
    i = wp.tid()
    width = groups.shape[1]
    r = i // width
    s = groups[r, i - width * r]
    if s < 0:
        s = slot_count
    keys[i] = s
    values[i] = i


@wp.kernel
def _entry_rows_kernel(values: wp.array[wp.int32], width: int, rows: wp.array[wp.int32]):
    i = wp.tid()
    rows[i] = values[i] // width


def group_entries(groups: wp.array2d[wp.int32], slot_count: int) -> tuple[wp.array, wp.array]:
    """The rows of each group of ``groups``, ``(rows, G)`` with each row's groups distinct and -1 where unused,
    computed on its device: ``(start, rows)``, the rows of group ``k`` ascending in ``rows[start[k]:start[k + 1]]``,
    as :func:`slot_entries` gives them for rows of two sides."""
    device = groups.device
    count = groups.shape[0] * groups.shape[1]
    start = wp.zeros(slot_count + 1, dtype=wp.int32, device=device)
    rows = wp.empty(max(count, 1), dtype=wp.int32, device=device)
    if count == 0:
        return start, rows
    # radix_sort_pairs sorts in place in arrays twice the sorted length.
    keys = wp.empty(2 * count, dtype=wp.int32, device=device)
    values = wp.empty(2 * count, dtype=wp.int32, device=device)
    wp.launch(_group_entries_kernel, count, inputs=[groups, slot_count], outputs=[keys, values], device=device)
    wp.utils.radix_sort_pairs(keys, values, count)
    wp.launch(_slot_starts_kernel, slot_count + 1, inputs=[keys, count], outputs=[start], device=device)
    wp.launch(_entry_rows_kernel, count, inputs=[values, groups.shape[1]], outputs=[rows], device=device)
    return start, rows


@wp.kernel
def row_force_rhs_kernel(
    slot_entry_start: wp.array[wp.int32],
    entry_row: wp.array[wp.int32],
    entry_side: wp.array[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
    force: wp.array[wp.float64],
    b_const: wp.array[wp.float64],
    # outputs
    b: wp.array[wp.float64],
):
    """``b = b_const + Jᵀ f`` per body slot, gathered over the rows touching it, launched tiled with a block of lanes
    per slot, :data:`ROW_FORCE_LANES` on CUDA: lane ``l`` sums the slot's entries ``l``, ``l + block_dim`` and so on,
    and the lanes' sums add up in a fixed order."""
    k, lane = wp.tid()
    acc = wp.spatial_vectord()
    for e in range(slot_entry_start[k] + lane, slot_entry_start[k + 1], wp.block_dim()):
        acc += force[entry_row[e]] * row_jac[entry_row[e], entry_side[e]]
    total = wp.tile_extract(wp.tile_sum(wp.tile(acc, preserve_type=True)), 0)
    if lane == 0:
        for c in range(6):
            b[6 * k + c] = b_const[6 * k + c] + total[c]


@wp.kernel
def rigid_delassus_kernel(
    W_soft: wp.array2d[wp.float64],
    n_soft: int,
    soft_tail: int,
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
    slot_inv_mass: wp.array[wp.spatial_matrixd],
    # outputs
    W: wp.array2d[wp.float64],
):
    """Delassus matrix ``W = J A⁻¹ Jᵀ`` of a set of constraint rows.

    ``A`` is block diagonal, so ``W`` is the deformable block ``W_soft`` on the
    first ``n_soft`` rows and on the rows from ``soft_tail`` on, which follow
    them in ``W_soft``, plus the response ``c_iᵀ A_s⁻¹ c_j`` of every body
    slot ``s`` that rows ``i`` and ``j`` share.
    """
    i, j = wp.tid()
    w = wp.float64(0.0)
    si, sj = i, j
    if i >= soft_tail:
        si = n_soft + i - soft_tail
    if j >= soft_tail:
        sj = n_soft + j - soft_tail
    if (i < n_soft or i >= soft_tail) and (j < n_soft or j >= soft_tail):
        w = W_soft[si, sj]
    for x in range(2):
        s = row_slot[i, x]
        if s >= 0:
            for y in range(2):
                if row_slot[j, y] == s:
                    w += wp.dot(row_jac[i, x], slot_inv_mass[s] * row_jac[j, y])
    W[i, j] = w


# Lanes of :func:`rigid_delassus_rows_kernel` per row.
DELASSUS_LANES = 32


@wp.kernel
def rigid_row_minv_jac_kernel(
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
    slot_inv_mass: wp.array[wp.spatial_matrixd],
    # outputs
    row_minv_jac: wp.array2d[wp.spatial_vectord],
):
    """``A⁻¹ Jᵀ`` of each row's side: its slot's block inverse times its Jacobian, zero on a side without a slot."""
    i, x = wp.tid()
    v = wp.spatial_vectord()
    s = row_slot[i, x]
    if s >= 0:
        v = slot_inv_mass[s] * row_jac[i, x]
    row_minv_jac[i, x] = v


@wp.kernel
def rigid_delassus_rows_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    soft_offsets: wp.array[wp.int32],
    soft_columns: wp.array[wp.int32],
    soft_values: wp.array[wp.float64],
    n_soft: int,
    soft_tail: int,
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
    row_minv_jac: wp.array2d[wp.spatial_vectord],
    # outputs
    values: wp.array[wp.float64],
):
    """The entries of :func:`rigid_delassus_kernel`'s ``W`` at a CSR pattern, from the deformable block ``W_soft`` in
    CSR form without its zero entries and the rows' ``A⁻¹ Jᵀ`` of :func:`rigid_row_minv_jac_kernel`. Launched tiled
    with a block of lanes per row, :data:`DELASSUS_LANES` on CUDA: lane ``l`` takes the row's entries ``l``,
    ``l + block_dim`` and so on."""
    i, lane = wp.tid()
    for k in range(offsets[i] + lane, offsets[i + 1], wp.block_dim()):
        j = columns[k]
        w = wp.float64(0.0)
        if (i < n_soft or i >= soft_tail) and (j < n_soft or j >= soft_tail):
            si, sj = i, j
            if i >= soft_tail:
                si = n_soft + i - soft_tail
            if j >= soft_tail:
                sj = n_soft + j - soft_tail
            lo = soft_offsets[si]
            hi = soft_offsets[si + 1]
            while lo < hi:
                mid = (lo + hi) // 2
                if soft_columns[mid] < sj:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < soft_offsets[si + 1]:
                if soft_columns[lo] == sj:
                    w = soft_values[lo]
        for x in range(2):
            s = row_slot[i, x]
            if s >= 0:
                for y in range(2):
                    if row_slot[j, y] == s:
                        w += wp.dot(row_jac[i, x], row_minv_jac[j, y])
        values[k] = w


@wp.kernel
def rigid_delassus_blocks_kernel(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
    row_minv_jac: wp.array2d[wp.spatial_vectord],
    # outputs
    values: wp.array[wp.mat33d],
):
    """The 3x3 blocks of the rigid rows' ``W = J A⁻¹ Jᵀ`` at a block CSR pattern of blocks of three rows, each entry
    as :func:`rigid_delassus_rows_kernel` forms it. Launched tiled with a block of lanes per block row,
    :data:`DELASSUS_LANES` on CUDA: lane ``l`` takes the block row's blocks ``l``, ``l + block_dim`` and so on."""
    bi, lane = wp.tid()
    for k in range(offsets[bi] + lane, offsets[bi + 1], wp.block_dim()):
        bj = columns[k]
        block = wp.mat33d()
        for a in range(3):
            i = 3 * bi + a
            for c in range(3):
                j = 3 * bj + c
                w = wp.float64(0.0)
                for x in range(2):
                    s = row_slot[i, x]
                    if s >= 0:
                        for y in range(2):
                            if row_slot[j, y] == s:
                                w += wp.dot(row_jac[i, x], row_minv_jac[j, y])
                block[a, c] = w
        values[k] = block


@wp.kernel
def rigid_row_residual_kernel(
    value: wp.array[wp.float64],
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
    u: wp.array[wp.float64],
    u_free: wp.array[wp.float64],
    # outputs
    r: wp.array[wp.float64],
):
    """NSN residual ``r = -value + J (u - u_free)`` of rows on rigid bodies.

    ``value`` is each row's constraint value at the live pose of ``u``; the
    shift moves it back to the free-flight DOFs ``u_free`` along the pinned
    ``J``, where the NSN linear model starts.
    """
    i = wp.tid()
    acc = -value[i]
    for side in range(2):
        s = row_slot[i, side]
        if s >= 0:
            for c in range(6):
                acc += row_jac[i, side][c] * (u[6 * s + c] - u_free[6 * s + c])
    r[i] = acc
