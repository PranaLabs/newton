# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reduction of SolverFBA's rigid contacts by bins on the device (``gather_rigid_contacts`` with a
``bin_size``): the contacts it keeps must be those of the reduction on the host it replaces, the deepest of each bin
(of equally deep ones the first), in their order."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.rigid_contact import gather_rigid_contacts

TILES, TILE, HALF = 12, 0.02, 0.1


def _scene(device):
    """A 0.2 m box tilted a little on a floor of 12 x 12 static 2 cm tiles, so that its contacts' depths differ."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.01)
    builder = newton.ModelBuilder()
    for i in range(TILES):
        for j in range(TILES):
            centre = ((i - (TILES - 1) / 2) * TILE, (j - (TILES - 1) / 2) * TILE, -TILE / 2)
            builder.add_shape_box(
                -1, xform=wp.transform(centre, wp.quat_identity()), hx=TILE / 2, hy=TILE / 2, hz=TILE / 2, cfg=cfg
            )
    tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 0.0)), 0.02)
    box = builder.add_body(xform=wp.transform((0.003, -0.002, HALF - 0.001), tilt), com=wp.vec3(0.01, 0.0, -0.02))
    builder.add_shape_box(box, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
    return builder.finalize(device=device)


def _host_reduction(rc, body_q, body_com, size):
    """The kept contacts' indices, as the reduction on the host computed them."""
    m = rc.count
    body = rc.body.numpy()
    anchor, normal = rc.anchor.numpy().astype(np.float64), rc.normal.numpy().astype(np.float64)
    q = body_q.astype(np.float64)
    world = np.empty((m, 2, 3))
    for k in range(2):
        b = body[:, k]
        dyn = b >= 0
        world[~dyn, k] = anchor[~dyn, k]
        if dyn.any():
            bq = q[b[dyn]]
            x, y, z, w = bq[:, 3], bq[:, 4], bq[:, 5], bq[:, 6]
            R = np.stack(
                [
                    np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                    np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                    np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
                ],
                1,
            )
            world[dyn, k] = bq[:, :3] + np.einsum("nij,nj->ni", R, body_com[b[dyn]] + anchor[dyn, k])
    gap = np.einsum("ij,ij->i", world[:, 1] - world[:, 0], normal)
    cell = np.floor(anchor[:, 0] / size).astype(np.int64)
    direction = np.rint(normal * 1.4).astype(np.int64)
    keys = np.concatenate([body.astype(np.int64), cell, direction], axis=1)
    order = np.lexsort((np.arange(m), gap, *keys.T[::-1]))
    k_sorted = keys[order]
    first = np.ones(m, dtype=bool)
    first[1:] = (k_sorted[1:] != k_sorted[:-1]).any(axis=1)
    return np.sort(order[first])


class TestRigidContactBinDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The contact gather sorts on CUDA")

    def test_keeps_the_host_reduction_contacts(self):
        """For bins of 1, 2 and 5 cm the device keeps the same contacts as the host reduction, deepest first, in
        their order, with their slots, shapes, anchors, normals and friction."""
        device = "cuda:0"
        model = _scene(device)
        state = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        slot = np.full(model.body_count, -1, dtype=np.int32)
        slot[0] = 0
        full = gather_rigid_contacts(model, contacts, slot, device)
        self.assertGreater(full.count, 100)
        com = model.body_com.numpy().astype(np.float64)
        for size in (0.01, 0.02, 0.05):
            with self.subTest(size=size):
                keep = _host_reduction(full, state.body_q.numpy(), com, size)
                self.assertLess(len(keep), full.count)
                kept = gather_rigid_contacts(model, contacts, slot, device, bin_size=size, body_q=state.body_q)
                self.assertEqual(kept.count, len(keep))
                np.testing.assert_array_equal(kept.slot, full.slot[keep])
                np.testing.assert_array_equal(kept.shape, full.shape[keep])
                for name in ("body", "anchor", "normal", "mu"):
                    np.testing.assert_array_equal(getattr(kept, name).numpy(), getattr(full, name).numpy()[keep])

    def test_without_contacts_on_bodies(self):
        """A step whose contacts all lie between static shapes keeps none."""
        device = "cuda:0"
        model = _scene(device)
        state = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        slot = np.full(model.body_count, -1, dtype=np.int32)  # the box is not integrated
        self.assertIsNone(gather_rigid_contacts(model, contacts, slot, device, bin_size=0.02, body_q=state.body_q))


if __name__ == "__main__":
    unittest.main(verbosity=2)
