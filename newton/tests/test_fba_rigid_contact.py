# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA rigid contacts."""

import types
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.rigid_contact import gather_rigid_contacts

_FIELDS = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal")


def _slots(model):
    dynamic = (model.body_flags.numpy() & int(newton.BodyFlags.DYNAMIC)) != 0
    slot = np.full(model.body_count, -1, dtype=np.int32)
    slot[dynamic] = np.arange(np.count_nonzero(dynamic))
    return slot


def _box_scene(device):
    """A dynamic box resting on the ground, its COM off the body origin, and a kinematic box beside it."""
    cfg = newton.ModelBuilder.ShapeConfig
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=cfg(mu=0.2))
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(box, xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()), cfg=cfg(mu=0.8))
    kinematic = builder.add_body(xform=wp.transform((3.0, 0.0, 0.5), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(kinematic)
    model = builder.finalize(device=device)
    state = model.state()
    return model, state, model.collide(state)


class TestGatherRigidContacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0" if wp.is_cuda_available() else "cpu"

    def test_box_on_ground(self):
        model, state, contacts = _box_scene(self.device)
        rc = gather_rigid_contacts(model, contacts, _slots(model), self.device)
        self.assertGreaterEqual(rc.count, 4)
        # Only the dynamic box: the kinematic box on the ground has nothing to integrate.
        body = rc.body.numpy()
        np.testing.assert_array_equal(np.sort(body, axis=1), np.tile([-1, 0], (rc.count, 1)))
        np.testing.assert_array_equal(rc.slot, np.where(body >= 0, 0, -1))
        np.testing.assert_allclose(rc.mu.numpy(), 0.5, rtol=1e-6)

        # The world anchors of both sides coincide on the box bottom.
        tf, com = state.body_q.numpy()[0], model.body_com.numpy()[0]
        world = rc.anchor.numpy().astype(np.float64)
        on_box = body >= 0
        world[on_box] = [
            wp.transform_point(wp.transform(*np.split(tf, [3])), wp.vec3(*(a + com))) for a in world[on_box]
        ]
        np.testing.assert_allclose(world[:, 0], world[:, 1], atol=1e-5)
        np.testing.assert_allclose(world[:, 0, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(np.abs(rc.normal.numpy()), np.tile([0.0, 0.0, 1.0], (rc.count, 1)), atol=1e-6)

    def test_order_does_not_depend_on_emission(self):
        model, _, contacts = _box_scene(self.device)
        n = int(contacts.rigid_contact_count.numpy()[0])
        perm = np.random.default_rng(0).permutation(n)
        shuffled = types.SimpleNamespace(rigid_contact_count=contacts.rigid_contact_count)
        for f in _FIELDS:
            a = getattr(contacts, f"rigid_contact_{f}")
            setattr(shuffled, f"rigid_contact_{f}", wp.array(a.numpy()[:n][perm], dtype=a.dtype, device=self.device))
        slot = _slots(model)
        ref, out = (gather_rigid_contacts(model, c, slot, self.device) for c in (contacts, shuffled))
        np.testing.assert_array_equal(out.slot, ref.slot)
        for f in ("body", "anchor", "normal", "mu"):
            np.testing.assert_array_equal(getattr(out, f).numpy(), getattr(ref, f).numpy())

    def test_no_integrated_body(self):
        model, _, contacts = _box_scene(self.device)
        self.assertIsNone(gather_rigid_contacts(model, contacts, np.full(model.body_count, -1), self.device))


if __name__ == "__main__":
    unittest.main(verbosity=2)
