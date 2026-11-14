# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reduction of SolverFBA's rigid contacts by bins (``rigid_contact_bin``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.rigid_contact import gather_rigid_contacts

TILES, TILE, HALF = 10, 0.02, 0.1


def _scene(device):
    """A 0.2 m box resting on a floor of 10 x 10 static 2 cm tiles: four contacts on every tile under it."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    builder = newton.ModelBuilder()
    for i in range(TILES):
        for j in range(TILES):
            centre = ((i - (TILES - 1) / 2) * TILE, (j - (TILES - 1) / 2) * TILE, -TILE / 2)
            builder.add_shape_box(
                -1, xform=wp.transform(centre, wp.quat_identity()), hx=TILE / 2, hy=TILE / 2, hz=TILE / 2, cfg=cfg
            )
    box = builder.add_body(xform=wp.transform((0.0, 0.0, HALF), wp.quat_identity()))
    builder.add_shape_box(box, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
    return builder.finalize(device=device)


def _keys(rc, size):
    """Each contact's bin: its bodies, the cell of its side-0 anchor and its normal rounded to 26 directions (in
    double precision, as the reduction bins them)."""
    cell = np.floor(rc.anchor.numpy().astype(np.float64)[:, 0] / size).astype(np.int64)
    direction = np.rint(rc.normal.numpy().astype(np.float64) * 1.4).astype(np.int64)
    return [tuple(k) for k in np.concatenate([rc.body.numpy().astype(np.int64), cell, direction], axis=1)]


class TestRigidContactBin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("The SolverFBA contact NSN needs CUDA")

    def test_one_contact_per_bin(self):
        """The reduction keeps one contact of each bin, in their order."""
        size = 0.05
        model = _scene("cuda:0")
        state = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        slot = np.full(model.body_count, -1, dtype=np.int32)
        slot[0] = 0
        full = gather_rigid_contacts(model, contacts, slot, "cuda:0")
        kept = gather_rigid_contacts(model, contacts, slot, "cuda:0", bin_size=size, body_q=state.body_q)
        self.assertGreater(full.count, 4 * kept.count)
        all_keys, kept_keys = _keys(full, size), _keys(kept, size)
        self.assertEqual(sorted(kept_keys), sorted(set(all_keys)))
        # Kept in their order: the kept contacts are a subsequence of the full set.
        rows = np.concatenate([full.anchor.numpy().reshape(full.count, -1), full.normal.numpy()], axis=1)
        kept_rows = np.concatenate([kept.anchor.numpy().reshape(kept.count, -1), kept.normal.numpy()], axis=1)
        index = [int(np.flatnonzero((rows == row).all(axis=1))[0]) for row in kept_rows]
        self.assertEqual(index, sorted(index))

    def test_box_rests_as_without(self):
        """The box rests on the tiles as it does with every contact, on far fewer contact rows."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        def rest(**kwargs):
            model = _scene("cuda:0")
            solver = SolverFBA(model, iterations=10, **kwargs)
            s0, s1 = model.state(), model.state()
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            for _ in range(30):
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, None, contacts, 0.01)
                s0, s1 = s1, s0
            return s0.body_q.numpy()[0], solver._rc_counts[1]

        q_all, rows_all = rest()
        q_bin, rows_bin = rest(rigid_contact_bin=0.05)
        self.assertLess(4 * rows_bin, rows_all)
        np.testing.assert_allclose(q_bin[:3], q_all[:3], atol=2e-4)
        np.testing.assert_allclose(np.abs(q_bin[3:]), np.abs(q_all[3:]), atol=1e-4)

    def test_negative_bin_refused(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        with self.assertRaises(ValueError):
            SolverFBA(_scene("cuda:0"), rigid_contact_bin=-0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
