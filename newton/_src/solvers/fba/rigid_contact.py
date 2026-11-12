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
from .kernels import host_head, stored_contact_count
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
    shape: np.ndarray
    """Model shape per side, shape [count, 2]."""
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


_DROPPED_PAIR = wp.constant(wp.int64(0x7FFFFFFFFFFFFFFF))


@wp.func
def _lookup(table: wp.array[wp.int32], i: wp.int32) -> wp.int32:
    """``table[i]``, or -1 for a negative ``i``."""
    if i < 0:
        return -1
    return table[i]


@wp.kernel
def _rigid_contact_keys_kernel(
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    offset0: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_slot: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    shape_count: wp.int64,
    # outputs
    pair_key: wp.array[wp.int64],
    anchor_key: wp.array2d[wp.float64],
    order: wp.array[wp.int32],
    table: wp.array[wp.int32],
):
    """The sort keys of each emitted contact: its shape pair, or a key past every pair for a contact the solver
    drops, and the side-0 anchor relative to the COM; counts the kept contacts in ``table[0]``."""
    c = wp.tid()
    order[c] = c
    s0, s1 = shape0[c], shape1[c]
    b0, b1 = _lookup(shape_body, s0), _lookup(shape_body, s1)
    slot0, slot1 = _lookup(body_slot, b0), _lookup(body_slot, b1)
    keep = s0 >= 0 and s1 >= 0 and b0 != b1 and (slot0 >= 0 or slot1 >= 0)
    a = wp.vec3d(point0[c]) + wp.vec3d(offset0[c])
    if b0 >= 0:
        a -= wp.vec3d(body_com[b0])
    for k in range(3):
        # -0 and +0 compare equal in the order; the radix sort keys alone would split them.
        v = a[k]
        if v == wp.float64(0.0):
            v = wp.float64(0.0)
        anchor_key[k, c] = v
    if keep:
        pair_key[c] = wp.int64(s0) * shape_count + wp.int64(s1)
        wp.atomic_add(table, 0, 1)
    else:
        pair_key[c] = _DROPPED_PAIR


@wp.kernel
def _gather_int64_kernel(src: wp.array[wp.int64], order: wp.array[wp.int32], dst: wp.array[wp.int64]):
    i = wp.tid()
    dst[i] = src[order[i]]


@wp.kernel
def _gather_float64_kernel(src: wp.array[wp.float64], order: wp.array[wp.int32], dst: wp.array[wp.float64]):
    i = wp.tid()
    dst[i] = src[order[i]]


@wp.kernel
def _rigid_contact_gather_kernel(
    order: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    offset0: wp.array[wp.vec3],
    offset1: wp.array[wp.vec3],
    normal_in: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_slot: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    shape_mu: wp.array[wp.float32],
    # outputs
    body: wp.array2d[wp.int32],
    anchor: wp.array2d[wp.vec3],
    normal: wp.array[wp.vec3],
    mu: wp.array[wp.float64],
    table: wp.array[wp.int32],
):
    """Contact ``i`` of the sorted order: its bodies, anchors, normal and friction, and its slots and shapes in row
    ``i`` of ``table[1:]``."""
    i = wp.tid()
    c = order[i]
    s = wp.vec2i(shape0[c], shape1[c])
    p = wp.vec3d(point0[c]) + wp.vec3d(offset0[c])
    q = wp.vec3d(point1[c]) + wp.vec3d(offset1[c])
    for k in range(2):
        b = _lookup(shape_body, s[k])
        slot = _lookup(body_slot, b)
        a = p
        if k == 1:
            a = q
        if b >= 0:
            a -= wp.vec3d(body_com[b])
        body[i, k] = b
        anchor[i, k] = wp.vec3(a)
        table[1 + 4 * i + k] = slot
        table[3 + 4 * i + k] = s[k]
    normal[i] = normal_in[c]
    mu_sum = wp.float64(0.0)
    if s[0] >= 0 and s[1] >= 0:
        mu_sum = wp.float64(shape_mu[s[0]]) + wp.float64(shape_mu[s[1]])
    mu[i] = mu_sum * wp.float64(0.5)


class RigidContactGather:
    """Device buffers of :func:`gather_rigid_contacts`, kept across steps and grown with the contact count."""

    def __init__(self, device):
        self.device = device
        self.capacity = 0

    def reserve(self, n: int) -> None:
        if n <= self.capacity:
            return
        cap = max(n, int(1.5 * self.capacity), 64)
        d = self.device
        # radix_sort_pairs works in place on arrays twice the sorted length.
        self.pair_key = wp.empty(2 * cap, dtype=wp.int64, device=d)
        self.anchor_key = wp.empty((3, cap), dtype=wp.float64, device=d)
        self.key64 = wp.empty(2 * cap, dtype=wp.int64, device=d)
        self.keyf = wp.empty(2 * cap, dtype=wp.float64, device=d)
        self.order = wp.empty(2 * cap, dtype=wp.int32, device=d)
        self.table = wp.empty(1 + 4 * cap, dtype=wp.int32, device=d)
        self.body = wp.empty((cap, 2), dtype=wp.int32, device=d)
        self.anchor = wp.empty((cap, 2), dtype=wp.vec3, device=d)
        self.normal = wp.empty(cap, dtype=wp.vec3, device=d)
        self.mu = wp.empty(cap, dtype=wp.float64, device=d)
        self.capacity = cap


class RigidRowStorage:
    """The NSN rows of the rigid contacts kept across steps, for a capacity of contacts that changes only when the
    step's contacts outgrow it or would take less than half of it afresh, followed by ``joint_rows`` rows for the
    joint rows that join the contacts.

    The rows past the step's contacts, and past its joint rows, stay inert: no body slot, no entries of ``W``, and
    zero residual, anchor projection, multipliers and friction, so that the NSN step of the capacity solves the
    step's rows alone and its graph can replay from step to step. :attr:`joint_select` holds the step's joint rows in
    the order of their NSN rows, -1 past them.
    """

    def __init__(self, device, joint_rows: int = 0):
        self.device = device
        self.contacts = 0
        self.joint_rows = joint_rows
        self.joint_select = wp.full(max(joint_rows, 1), -1, dtype=wp.int32, device=device)

    def select_joint_rows(self, rows: np.ndarray) -> None:
        """Take the joint rows ``rows`` for the step's NSN rows that follow the contacts' capacity."""
        select = np.full(self.joint_select.shape[0], -1, dtype=np.int32)
        select[: len(rows)] = rows
        self.joint_select.assign(select)

    def reserve(self, m: int) -> None:
        """Hold the rows of ``m`` contacts."""
        from .linear_solver import CsrStorage  # noqa: PLC0415

        # A tenth more than the contacts, in steps of 64 contacts.
        contacts = 64 * ((m + m // 10 + 63) // 64)
        if m <= self.contacts and 2 * contacts >= self.contacts:
            return
        n, d = 3 * contacts + self.joint_rows, self.device
        self.row_slot = wp.empty((n, 2), dtype=wp.int32, device=d)
        self.row_jac = wp.empty((n, 2), dtype=wp.spatial_vectord, device=d)
        self.pene0 = wp.zeros(n, dtype=wp.float64, device=d)
        self.mu = wp.empty(contacts, dtype=wp.float64, device=d)
        self.value = wp.empty(n, dtype=wp.float64, device=d)
        self.r = wp.empty(n, dtype=wp.float64, device=d)
        self.lam = wp.empty(n, dtype=wp.float64, device=d)
        self.omega = wp.empty(n, dtype=wp.float64, device=d)
        self.csr = CsrStorage(n, d)
        self.contacts = contacts

    def begin(self) -> None:
        """Clear the rows for a step; the rows of its contacts are written after."""
        self.row_slot.fill_(-1)
        self.row_jac.zero_()
        for a in (self.mu, self.r, self.lam, self.omega):
            a.zero_()


def gather_rigid_contacts(
    model: Model,
    contacts: Contacts,
    body_slot: np.ndarray | wp.array,
    device,
    buffers: RigidContactGather | None = None,
) -> RigidContacts | None:
    """Collect the rigid contacts that touch a body the solver integrates.

    Contacts within one body or between bodies the solver does not integrate
    are dropped. The rest are sorted by shape pair and anchor so the rows do not
    depend on the order the collision pipeline emitted them in. The selection
    and the sort run on the device; the host reads the contact count and the
    kept contacts' slots and shapes.

    Args:
        model: The Newton model.
        contacts: Contacts populated by :meth:`~newton.Model.collide`.
        body_slot: Rigid DOF slot per model body, -1 if not integrated.
        device: Warp device for the contact arrays.
        buffers: Device buffers to reuse across calls. The arrays of the
            returned contacts are views into them, valid until the next call.

    Returns:
        The contacts, or ``None`` if no contact touches an integrated body.
    """
    n = stored_contact_count(contacts.rigid_contact_count, contacts.rigid_contact_shape0.shape[0], "rigid")
    if n == 0:
        return None
    if buffers is None:
        buffers = RigidContactGather(device)
    buffers.reserve(n)
    B = buffers
    if not isinstance(body_slot, wp.array):
        body_slot = wp.array(np.asarray(body_slot, dtype=np.int32), dtype=wp.int32, device=device)
    c = contacts
    B.table.zero_()
    wp.launch(
        _rigid_contact_keys_kernel,
        n,
        inputs=[
            c.rigid_contact_shape0,
            c.rigid_contact_shape1,
            c.rigid_contact_point0,
            c.rigid_contact_offset0,
            model.shape_body,
            body_slot,
            model.body_com,
            wp.int64(max(model.shape_count, 1)),
        ],
        outputs=[B.pair_key, B.anchor_key, B.order, B.table],
        device=device,
    )
    # Stable least-significant-key-first passes: anchor z, y, x, then the shape pair, as ``np.lexsort`` orders them.
    for k in (2, 1, 0):
        wp.launch(_gather_float64_kernel, n, inputs=[B.anchor_key[k], B.order], outputs=[B.keyf], device=device)
        wp.utils.radix_sort_pairs(B.keyf, B.order, n)
    wp.launch(_gather_int64_kernel, n, inputs=[B.pair_key, B.order], outputs=[B.key64], device=device)
    wp.utils.radix_sort_pairs(B.key64, B.order, n)
    wp.launch(
        _rigid_contact_gather_kernel,
        n,
        inputs=[
            B.order,
            c.rigid_contact_shape0,
            c.rigid_contact_shape1,
            c.rigid_contact_point0,
            c.rigid_contact_point1,
            c.rigid_contact_offset0,
            c.rigid_contact_offset1,
            c.rigid_contact_normal,
            model.shape_body,
            body_slot,
            model.body_com,
            model.shape_material_mu,
        ],
        outputs=[B.body, B.anchor, B.normal, B.mu, B.table],
        device=device,
    )
    table = host_head(B.table, 1 + 4 * n)
    m = int(table[0])
    if m == 0:
        return None
    rows = table[1:].reshape(n, 4)[:m]
    return RigidContacts(
        slot=np.ascontiguousarray(rows[:, :2]),
        shape=np.ascontiguousarray(rows[:, 2:]),
        body=B.body[:m],
        anchor=B.anchor[:m],
        normal=B.normal[:m],
        mu=B.mu[:m],
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
def rigid_contact_tangents_kernel(
    normal: wp.array[wp.vec3],
    # outputs
    tangent1: wp.array[wp.vec3],
    tangent2: wp.array[wp.vec3],
):
    """Directions of the two friction rows of each rigid contact."""
    c = wp.tid()
    n = wp.vec3d(normal[c])
    tangent1[c] = wp.vec3(_row_direction(n, 1))
    tangent2[c] = wp.vec3(_row_direction(n, 2))


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
