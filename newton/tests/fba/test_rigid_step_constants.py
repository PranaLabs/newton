# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the step-constant parts of SolverFBA's rigid contact and tendon rows.

A row's value at the live pose of an iteration takes each side's anchor at the step start, its lever and friction
target, formed once per step (``rigid_contact_anchor_kernel``, ``tendon_anchor_kernel``), and the turn of each body
since the step start, which the retraction keeps (``retract_rigid_kernel``). The values must be those of the anchors
moved to first order in that turn.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.fba import rigid_contact as RC
from newton._src.solvers.fba.rigid import retract_rigid_kernel

DT = 1.0 / 480.0
DEVICE = "cpu"


def _rotation(q):
    """Rotation matrix of the quaternion ``(x, y, z, w)``."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_vector(q):
    """Rotation vector of the unit quaternion ``(x, y, z, w)``, the shorter way round."""
    v, w = np.asarray(q[:3]), q[3]
    if w < 0:
        v, w = -v, -w
    s = np.linalg.norm(v)
    return 2.0 * v if s < 1e-12 else 2.0 * np.arctan2(s, w) / s * v


def _product(a, b):
    """Quaternion product ``a b`` of ``(x, y, z, w)`` quaternions."""
    av, aw, bv, bw = np.asarray(a[:3]), a[3], np.asarray(b[:3]), b[3]
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - av @ bv]


def _direction(n, k):
    """Row ``k`` of a contact of normal ``n``: the normal, then two tangents."""
    if k == 0:
        return n
    a = np.array([0.0, 1.0, 0.0]) if abs(n[0]) > 0.5 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    return t1 if k == 1 else np.cross(n, t1)


class TestRigidStepConstants(unittest.TestCase):
    """Six bodies, five integrated (slots 0-4) and one the solver does not integrate, moved by a retraction."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.rng = rng
        cls.slot = np.array([0, 1, 2, -1, 3, 4], np.int32)
        bodies = np.flatnonzero(cls.slot >= 0).astype(np.int32)
        q = rng.standard_normal((6, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        cls.body_q = np.concatenate([rng.uniform(-1, 1, (6, 3)), q], axis=1).astype(np.float32)
        cls.body_qd = rng.standard_normal((6, 6)).astype(np.float32)
        cls.com = rng.uniform(-0.05, 0.05, (6, 3)).astype(np.float32)
        q_pin = rng.standard_normal((len(bodies), 4))
        q_pin /= np.linalg.norm(q_pin, axis=1, keepdims=True)
        cls.x0 = rng.uniform(-1, 1, (len(bodies), 3))
        u = rng.standard_normal(6 * len(bodies)) * 0.02
        cls.d = {
            "slot": wp.array(cls.slot, dtype=wp.int32, device=DEVICE),
            "body_q": wp.array(cls.body_q, dtype=wp.transform, device=DEVICE),
            "body_qd": wp.array(cls.body_qd, dtype=wp.spatial_vector, device=DEVICE),
            "com": wp.array(cls.com, dtype=wp.vec3, device=DEVICE),
            "x0": wp.array(cls.x0, dtype=wp.vec3d, device=DEVICE),
        }
        live = [wp.empty(len(bodies), dtype=t, device=DEVICE) for t in (wp.vec3d, wp.quatd, wp.vec3d)]
        scratch = [wp.empty(6, dtype=wp.transform, device=DEVICE), wp.empty(6, dtype=wp.spatial_vector, device=DEVICE)]
        wp.launch(
            retract_rigid_kernel,
            len(bodies),
            inputs=[
                wp.array(bodies, dtype=wp.int32, device=DEVICE),
                wp.array(u, dtype=wp.float64, device=DEVICE),
                wp.array(q_pin, dtype=wp.quatd, device=DEVICE),
                cls.d["x0"],
                cls.d["body_q"],
                cls.d["com"],
                wp.float64(DT),
            ],
            outputs=[*scratch, *live],
            device=DEVICE,
        )
        cls.d["x"], cls.d["q"], cls.d["turn"] = live
        cls.x, cls.q, cls.turn = (a.numpy().astype(np.float64) for a in live)

    def _q0(self, b):
        q = self.body_q[b, 3:].astype(np.float64)
        return q / np.linalg.norm(q)

    def _side(self, b, r):
        """Reference start, live anchor and friction target of a side with anchor ``r`` on body ``b``."""
        r = np.asarray(r, dtype=np.float64)
        if b < 0:
            return r, r, r
        q0 = self._q0(b)
        R0 = _rotation(q0)
        s = self.slot[b]
        if s >= 0:
            r0 = R0 @ r
            live = self.x[s] + r0 + np.cross(self.turn[s], r0)
            return self.x0[s] + r0, live, live
        c0 = self.body_q[b, :3] + R0 @ self.com[b]
        a0 = c0 + R0 @ r
        v = self.body_qd[b, :3] + np.cross(self.body_qd[b, 3:], a0 - c0)
        return a0, a0, a0 + DT * v

    def test_turn_is_the_rotation_since_the_step_start(self):
        """The retraction keeps each slot's turn ``q q0⁻¹`` as a rotation vector."""
        for b in np.flatnonzero(self.slot >= 0):
            s = self.slot[b]
            q0 = self._q0(b)
            q0_inv = np.r_[-q0[:3], q0[3]]
            np.testing.assert_allclose(self.turn[s], _rotation_vector(_product(self.q[s], q0_inv)), atol=1e-12)
            self.assertGreater(np.linalg.norm(self.turn[s]), 1e-6)

    def test_contact_rows(self):
        """The normal gap ``n · (a1 - a0)`` and the tangential slips over the step, for contacts between static
        shapes, integrated bodies and a body the solver does not integrate."""
        m = 64
        body = self.rng.integers(-1, 6, size=(m, 2)).astype(np.int32)
        anchor = self.rng.uniform(-0.1, 0.1, (m, 2, 3)).astype(np.float32)
        normal = self.rng.standard_normal((m, 3))
        normal = (normal / np.linalg.norm(normal, axis=1, keepdims=True)).astype(np.float32)
        d = self.d
        args = [
            wp.array(body, dtype=wp.int32, device=DEVICE),
            wp.array(anchor, dtype=wp.vec3, device=DEVICE),
            wp.array(normal, dtype=wp.vec3, device=DEVICE),
        ]
        parts = [wp.empty((m, 2), dtype=wp.vec3d, device=DEVICE) for _ in range(4)]
        wp.launch(
            RC.rigid_contact_anchor_kernel,
            m,
            inputs=[*args, d["slot"], d["x0"], d["body_q"], d["body_qd"], d["com"], wp.float64(DT)],
            outputs=parts,
            device=DEVICE,
        )
        gap = wp.zeros(3 * m + 3, dtype=wp.float64, device=DEVICE)
        wp.launch(
            RC.rigid_contact_gap_kernel,
            m,
            inputs=[args[0], args[2], *parts, d["slot"], d["x"], d["turn"], 3],
            outputs=[gap],
            device=DEVICE,
        )
        gap = gap.numpy()
        np.testing.assert_array_equal(gap[:3], 0.0)  # rows before the offset stay untouched
        for c in range(m):
            n = normal[c].astype(np.float64)
            (s0, l0, t0), (s1, l1, t1) = (self._side(body[c, k], anchor[c, k]) for k in range(2))
            slip = (t1 - s1) - (t0 - s0)
            expected = [n @ (l1 - l0), _direction(n, 1) @ slip, _direction(n, 2) @ slip]
            np.testing.assert_allclose(gap[3 + 3 * c : 6 + 3 * c], expected, atol=1e-12)

    def test_tendon_rows(self):
        """A tendon's slack ``L0 - L`` with each end's run measured along its step-start direction, and the first
        end's slide along it where the tendon has friction."""
        t_count = 16
        body = self.rng.integers(-1, 6, size=(t_count, 2)).astype(np.int32)
        point = self.rng.uniform(-0.1, 0.1, (t_count, 2, 3)).astype(np.float32)
        target = self.rng.uniform(-1, 1, (t_count, 2, 3)).astype(np.float32)
        middle, rest = self.rng.uniform(0.1, 0.5, t_count), self.rng.uniform(0.5, 2.0, t_count)
        mu = np.where(np.arange(t_count) % 2 == 0, 0.3, 0.0)
        d = self.d
        b_d, t_d = wp.array(body, dtype=wp.int32, device=DEVICE), wp.array(target, dtype=wp.vec3, device=DEVICE)
        ends = [wp.empty((t_count, 2), dtype=wp.vec3d, device=DEVICE) for _ in range(3)]
        wp.launch(
            RC.tendon_anchor_kernel,
            t_count,
            inputs=[
                *(b_d, wp.array(point, dtype=wp.vec3, device=DEVICE), t_d),
                *(d["slot"], d["x0"], d["body_q"], d["body_qd"], d["com"], wp.float64(DT)),
            ],
            outputs=ends,
            device=DEVICE,
        )
        gap = wp.zeros(3 * t_count, dtype=wp.float64, device=DEVICE)
        f64 = [wp.array(a, dtype=wp.float64, device=DEVICE) for a in (middle, rest, mu)]
        wp.launch(
            RC.tendon_gap_kernel,
            t_count,
            inputs=[b_d, t_d, *f64, *ends, d["slot"], d["x"], d["turn"], 0],
            outputs=[gap],
            device=DEVICE,
        )
        gap = gap.numpy().reshape(t_count, 3)
        for t in range(t_count):
            length, slide = middle[t], 0.0
            for k in range(2):
                b = body[t, k]
                p = point[t, k].astype(np.float64)
                if b >= 0:
                    R0 = _rotation(self._q0(b))
                    lever = p - self.com[b]
                    p0 = self.body_q[b, :3] + R0 @ self.com[b] + R0 @ lever
                    start, live, _ = self._side(b, lever)
                else:
                    p0 = start = live = p
                goal = target[t, k].astype(np.float64)
                e = (goal - p0) / np.linalg.norm(goal - p0)
                length += e @ (goal - live)
                if k == 0 and mu[t] > 0:
                    slide = e @ (live - start)
            np.testing.assert_allclose(gap[t], [rest[t] - length, slide, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
