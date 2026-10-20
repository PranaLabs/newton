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
    """``b = b_const + Jᵀ f`` per body slot, gathered over the rows touching it."""
    k = wp.tid()
    acc = wp.spatial_vectord()
    for e in range(slot_entry_start[k], slot_entry_start[k + 1]):
        acc += force[entry_row[e]] * row_jac[entry_row[e], entry_side[e]]
    for c in range(6):
        b[6 * k + c] = b_const[6 * k + c] + acc[c]


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
