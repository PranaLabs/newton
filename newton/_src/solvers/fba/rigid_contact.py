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
