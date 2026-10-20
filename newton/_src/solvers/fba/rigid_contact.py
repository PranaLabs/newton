# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body contacts for SolverFBA.

Each contact touching a body the solver integrates becomes three rows
``[n, t1, t2]``: a unilateral row on the gap ``n · (a1 - a0)`` between the
friction anchors of its two sides, and two friction rows on their relative
tangential displacement over the step. Anchors are Newton's contact points
shifted by the contact offsets, so the gap already subtracts both margins.

Within the step an anchor on an integrated body moves with the body to first
order in its rotation, ``a = x + r0 + theta x r0``, with ``r0`` its lever at the
step start and ``theta`` the body's rotation since then, and ``r0`` is also the
lever of the rows. A contact force thus acts at the contact point the collision
pipeline found, and a ball keeps rolling at its speed: a lever turned with the
step's rotation would let the normal force spin the ball up.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import Contacts, Model
from .rigid import slot_coefficients


@dataclass
class RigidContacts:
    """Rigid contacts that touch an integrated body, in a deterministic order.

    Side ``k`` of contact ``c`` is on model body ``body[c, k]``, -1 for a static
    shape, and has body slot ``slot[c, k]``, -1 if the solver does not integrate
    that body.
    """

    slot: np.ndarray
    """Body slot per side, -1 if not integrated, shape [count, 2]."""
    body: wp.array2d[wp.int32]
    """Model body per side, -1 for a static shape, shape [count, 2]."""
    anchor: wp.array2d[wp.vec3]
    """Friction anchor per side, relative to the COM in the body frame, or in
    the world frame for a static shape [m], shape [count, 2]."""
    normal: wp.array[wp.vec3]
    """World contact normal from side 0 to side 1, shape [count]."""
    mu: wp.array[wp.float64]
    """Friction coefficient, the mean of both shapes', shape [count]."""

    @property
    def count(self) -> int:
        return self.slot.shape[0]


def gather_rigid_contacts(model: Model, contacts: Contacts, body_slot: np.ndarray, device) -> RigidContacts | None:
    """Collect the rigid contacts that touch a body the solver integrates.

    Contacts within one body or between bodies the solver does not integrate
    are dropped. The rest are sorted by shape pair and anchor so the rows do not
    depend on the order the collision pipeline emitted them in.

    Args:
        model: The Newton model.
        contacts: Contacts populated by :meth:`~newton.Model.collide`.
        body_slot: Rigid DOF slot per model body, -1 if not integrated.
        device: Warp device for the contact arrays.

    Returns:
        The contacts, or ``None`` if no contact touches an integrated body.
    """
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return None
    shape = np.stack([contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]], axis=1)
    body = np.where(shape >= 0, model.shape_body.numpy()[np.maximum(shape, 0)], -1)
    slot = np.where(body >= 0, body_slot[np.maximum(body, 0)], -1)
    keep = np.flatnonzero((shape >= 0).all(axis=1) & (body[:, 0] != body[:, 1]) & (slot >= 0).any(axis=1))
    if keep.size == 0:
        return None
    point = np.stack([contacts.rigid_contact_point0.numpy(), contacts.rigid_contact_point1.numpy()], axis=1)[keep]
    offset = np.stack([contacts.rigid_contact_offset0.numpy(), contacts.rigid_contact_offset1.numpy()], axis=1)[keep]
    shape, body, slot = shape[keep], body[keep], slot[keep]
    com = np.where(body[..., None] >= 0, model.body_com.numpy()[np.maximum(body, 0)], 0.0)
    anchor = point.astype(np.float64) + offset - com
    normal = contacts.rigid_contact_normal.numpy()[keep]

    order = np.lexsort((anchor[:, 0, 2], anchor[:, 0, 1], anchor[:, 0, 0], shape[:, 1], shape[:, 0]))
    mu = model.shape_material_mu.numpy().astype(np.float64)[shape[order]].mean(axis=1)
    return RigidContacts(
        slot=slot[order],
        body=wp.array(body[order], dtype=wp.int32, device=device),
        anchor=wp.array(anchor[order], dtype=wp.vec3, device=device),
        normal=wp.array(normal[order], dtype=wp.vec3, device=device),
        mu=wp.array(mu, dtype=wp.float64, device=device),
    )


@wp.func
def _row_direction(n: wp.vec3d, k: int) -> wp.vec3d:
    """Direction of row ``k`` of a contact: the normal, then two tangents."""
    if k == 0:
        return n
    a = wp.vec3d(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.5:
        a = wp.vec3d(0.0, 1.0, 0.0)
    t1 = wp.normalize(wp.cross(n, a))
    if k == 1:
        return t1
    return wp.cross(n, t1)


@wp.func
def _rotation_vector(q: wp.quatd) -> wp.vec3d:
    """Rotation vector of the unit quaternion ``q``, the shorter way round."""
    v = wp.vec3d(q[0], q[1], q[2])
    w = q[3]
    if w < wp.float64(0.0):
        v = -v
        w = -w
    s = wp.length(v)
    if s < wp.float64(1.0e-12):
        return wp.float64(2.0) * v
    return (wp.float64(2.0) * wp.atan2(s, w) / s) * v


@wp.func
def _moved_anchor(x: wp.vec3d, q: wp.quatd, q0: wp.quatd, r0: wp.vec3d) -> wp.vec3d:
    """Anchor of lever ``r0`` at the step start on a body now at COM ``x`` and turned from ``q0`` to ``q``, to first
    order in the turn."""
    return x + r0 + wp.cross(_rotation_vector(q * wp.quat_inverse(q0)), r0)


@wp.kernel
def rigid_contact_jacobian_kernel(
    body: wp.array2d[wp.int32],
    anchor: wp.array2d[wp.vec3],
    normal: wp.array[wp.vec3],
    body_slot: wp.array[wp.int32],
    q_pin: wp.array[wp.quatd],
    row_offset: int,
    body_q0: wp.array[wp.transform],
    # outputs
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
):
    """Coefficients of rows ``row_offset + 3c + k`` on the DOFs ``(chi, rho)`` of both sides.

    The row along the world direction ``d`` measures ``d · (a1 - a0)``. At the
    pinned frame, side ``k`` contributes ``±(R_pinᵀ d, R_pinᵀ r0 x R_pinᵀ d)``,
    negative for side 0, with ``r0`` the lever of its anchor relative to the COM
    at the step-start pose ``body_q0``.
    """
    c, k = wp.tid()
    d = _row_direction(wp.vec3d(normal[c]), k)
    r = row_offset + 3 * c + k
    for side in range(2):
        s = wp.int32(-1)
        if body[c, side] >= 0:
            s = body_slot[body[c, side]]
        row_slot[r, side] = s
        coeffs = wp.spatial_vectord()
        if s >= 0:
            d_loc = wp.quat_rotate_inv(q_pin[s], d)
            q0 = wp.normalize(wp.quatd(wp.transform_get_rotation(body_q0[body[c, side]])))
            lever = wp.quat_rotate_inv(q_pin[s], wp.quat_rotate(q0, wp.vec3d(anchor[c, side])))
            coeffs = slot_coefficients(d_loc, wp.cross(lever, d_loc))
            if side == 0:
                coeffs = -coeffs
        row_jac[r, side] = coeffs


@wp.func
def _side_anchor(
    b: int,
    r: wp.vec3d,
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    slot_x0: wp.array[wp.vec3d],
    body_q0: wp.array[wp.transform],
    body_qd0: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    dt: wp.float64,
):
    """World anchor at the step start, at the live pose, and its friction target.

    On an integrated body the anchor moves to first order in the body's
    rotation since the step start. A body the solver does not integrate keeps
    its pose; only its friction target follows the body velocity, as for
    deformable contacts.
    """
    if b < 0:
        return r, r, r
    q0 = wp.normalize(wp.quatd(wp.transform_get_rotation(body_q0[b])))
    s = body_slot[b]
    if s >= 0:
        r0 = wp.quat_rotate(q0, r)
        a = _moved_anchor(slot_x[s], slot_q[s], q0, r0)
        return slot_x0[s] + r0, a, a
    x0 = wp.vec3d(wp.transform_get_translation(body_q0[b])) + wp.quat_rotate(q0, wp.vec3d(body_com[b]))
    a0 = x0 + wp.quat_rotate(q0, r)
    v = wp.vec3d(wp.spatial_top(body_qd0[b])) + wp.cross(wp.vec3d(wp.spatial_bottom(body_qd0[b])), a0 - x0)
    return a0, a0, a0 + dt * v


@wp.kernel
def rigid_contact_gap_kernel(
    body: wp.array2d[wp.int32],
    anchor: wp.array2d[wp.vec3],
    normal: wp.array[wp.vec3],
    body_slot: wp.array[wp.int32],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    slot_x0: wp.array[wp.vec3d],
    body_q0: wp.array[wp.transform],
    body_qd0: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    dt: wp.float64,
    row_offset: int,
    # outputs
    gap: wp.array[wp.float64],
):
    """Row values at the live pose: the normal gap ``n · (a1 - a0)`` and the
    tangential slip ``t · ((a1 - a1⁰) - (a0 - a0⁰))`` over the step [m]."""
    c = wp.tid()
    n = wp.vec3d(normal[c])
    start0, live0, target0 = _side_anchor(
        body[c, 0], wp.vec3d(anchor[c, 0]), body_slot, slot_x, slot_q, slot_x0, body_q0, body_qd0, body_com, dt
    )
    start1, live1, target1 = _side_anchor(
        body[c, 1], wp.vec3d(anchor[c, 1]), body_slot, slot_x, slot_q, slot_x0, body_q0, body_qd0, body_com, dt
    )
    slip = (target1 - start1) - (target0 - start0)
    gap[row_offset + 3 * c] = wp.dot(n, live1 - live0)
    gap[row_offset + 3 * c + 1] = wp.dot(_row_direction(n, 1), slip)
    gap[row_offset + 3 * c + 2] = wp.dot(_row_direction(n, 2), slip)


@wp.func
def _particle_row_direction(c: int, k: int, normal: wp.array[wp.vec3], t1: wp.array[wp.vec3], t2: wp.array[wp.vec3]):
    if k == 0:
        return wp.vec3d(normal[c])
    if k == 1:
        return wp.vec3d(t1[c])
    return wp.vec3d(t2[c])


@wp.func
def _particle_contact_body(c: int, contact_shape: wp.array[wp.int32], shape_body: wp.array[wp.int32]) -> wp.int32:
    """Body of a particle contact's shape, -1 for a static shape."""
    s = contact_shape[c]
    if s < 0:
        return wp.int32(-1)
    return shape_body[s]


@wp.kernel
def particle_contact_body_jacobian_kernel(
    contact_shape: wp.array[wp.int32],
    contact_body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    tangent1: wp.array[wp.vec3],
    tangent2: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_slot: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    q_pin: wp.array[wp.quatd],
    body_q0: wp.array[wp.transform],
    # outputs
    row_slot: wp.array2d[wp.int32],
    row_jac: wp.array2d[wp.spatial_vectord],
):
    """Body side of rows ``3c + k`` of particle contacts, at the pinned frame.

    A row along ``d`` measures ``alpha d · x - d · a`` for the particle ``x``
    and the anchor ``a`` on the shape's body, so an integrated body adds
    ``-(R_pinᵀ d, R_pinᵀ r0 x R_pinᵀ d)`` with ``r0`` the lever of the anchor
    relative to its COM at the step-start pose ``body_q0``. Contacts on other
    shapes get no body side.
    """
    c, k = wp.tid()
    b = _particle_contact_body(c, contact_shape, shape_body)
    s = wp.int32(-1)
    if b >= 0:
        s = body_slot[b]
    row = 3 * c + k
    row_slot[row, 0] = s
    row_slot[row, 1] = -1
    coeffs = wp.spatial_vectord()
    if s >= 0:
        d_loc = wp.quat_rotate_inv(q_pin[s], _particle_row_direction(c, k, normal, tangent1, tangent2))
        q0 = wp.normalize(wp.quatd(wp.transform_get_rotation(body_q0[b])))
        lever = wp.quat_rotate_inv(q_pin[s], wp.quat_rotate(q0, wp.vec3d(contact_body_pos[c] - body_com[b])))
        coeffs = -slot_coefficients(d_loc, wp.cross(lever, d_loc))
    row_jac[row, 0] = coeffs
    row_jac[row, 1] = wp.spatial_vectord()


@wp.kernel
def particle_contact_body_residual_kernel(
    contact_shape: wp.array[wp.int32],
    contact_body_pos: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    tangent1: wp.array[wp.vec3],
    tangent2: wp.array[wp.vec3],
    world_anchor: wp.array[wp.vec3],
    v_anchor: wp.array[wp.vec3d],
    shape_body: wp.array[wp.int32],
    body_slot: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    slot_x: wp.array[wp.vec3d],
    slot_q: wp.array[wp.quatd],
    body_q0: wp.array[wp.transform],
    row_jac: wp.array2d[wp.spatial_vectord],
    u: wp.array[wp.float64],
    u_free: wp.array[wp.float64],
    dt: wp.float64,
    # outputs
    r: wp.array[wp.float64],
):
    """Move the anchor of particle contacts on integrated bodies with the body.

    The deformable residual measures from the step-start anchor ``a0``, moved
    by ``dt v`` in the friction rows. On an integrated body the anchor moves
    with the live pose instead, to first order in the body's rotation since the
    step-start pose ``body_q0``, ``r += d · (a - a0) - dt d · v`` in friction
    rows, plus the shift ``J_body (u - u_free)`` back to the free-flight DOFs.
    """
    c, k = wp.tid()
    b = _particle_contact_body(c, contact_shape, shape_body)
    if b < 0:
        return
    s = body_slot[b]
    if s < 0:
        return
    d = _particle_row_direction(c, k, normal, tangent1, tangent2)
    q0 = wp.normalize(wp.quatd(wp.transform_get_rotation(body_q0[b])))
    a = _moved_anchor(slot_x[s], slot_q[s], q0, wp.quat_rotate(q0, wp.vec3d(contact_body_pos[c] - body_com[b])))
    delta = wp.dot(d, a - wp.vec3d(world_anchor[c]))
    if k > 0:
        delta -= dt * wp.dot(d, v_anchor[c])
    row = 3 * c + k
    for i in range(6):
        delta += row_jac[row, 0][i] * (u[6 * s + i] - u_free[6 * s + i])
    r[row] += delta
