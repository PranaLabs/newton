# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body contacts for SolverFBA.

Each contact touching a body the solver integrates becomes three rows
``[n, t1, t2]``: a unilateral row on the gap ``n · (a1 - a0)`` between the
friction anchors of its two sides, and two friction rows on their relative
tangential displacement over the step. Anchors are Newton's contact points
shifted by the contact offsets, so the gap already subtracts both margins.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import Contacts, Model


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

    A body the solver does not integrate keeps its pose; only its friction
    target follows the body velocity, as for deformable contacts.
    """
    if b < 0:
        return r, r, r
    q0 = wp.normalize(wp.quatd(wp.transform_get_rotation(body_q0[b])))
    s = body_slot[b]
    if s >= 0:
        a = slot_x[s] + wp.quat_rotate(slot_q[s], r)
        return slot_x0[s] + wp.quat_rotate(q0, r), a, a
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
