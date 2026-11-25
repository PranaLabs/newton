# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rod contacts of SolverFBA: springs that collide with each other as capsules."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.rod_contact import RodContact
from newton.solvers import SolverFBA

ROD_CAPACITY_ATOL = 1.0e-6


def _rod(builder, start, end, segments, radius, young=1.0e7, bending_stiffness=None):
    """A particle rod of ``segments`` equal segments from ``start`` to ``end``."""
    points = np.linspace(np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64), segments + 1)
    return SolverFBA.add_particle_rod(
        builder, points, radius, young, density=1000.0, bending_stiffness=bending_stiffness
    )


def _segment_distance(p0, p1, q0, q1):
    """Distance between the segments ``p0 p1`` and ``q0 q1``, from the closest points of their lines clamped to
    the segments and the four endpoint-to-segment distances."""

    def point(x, a, b):
        d = b - a
        return np.linalg.norm(x - (a + np.clip(np.dot(x - a, d) / np.dot(d, d), 0.0, 1.0) * d))

    best = min(point(p0, q0, q1), point(p1, q0, q1), point(q0, p0, p1), point(q1, p0, p1))
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, b, e = np.dot(d1, d1), np.dot(d1, d2), np.dot(d2, d2)
    denom = a * e - b * b
    if denom > 1.0e-12 * a * e:
        s = (b * np.dot(d2, r) - np.dot(d1, r) * e) / denom
        t = (a * np.dot(d2, r) - b * np.dot(d1, r)) / denom
        if 0.0 < s < 1.0 and 0.0 < t < 1.0:
            best = min(best, np.linalg.norm(p0 + s * d1 - q0 - t * d2))
    return best


def _contacts(model, gap=0.002):
    return RodContact(model, model.spring_indices.numpy().reshape(-1, 2), gap).detect(model.particle_q)


def _settle(model, steps=200, **kwargs):
    solver = SolverFBA(model, iterations=10, nsn_linear_iterations=10, rayleigh_alpha=2.0, **kwargs)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    for _ in range(steps):
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, 0.005)
        s0, s1 = s1, s0
    return s0.particle_q.numpy().astype(np.float64)


class TestRodContactDetection(unittest.TestCase):
    """The contacts that :class:`RodContact` finds."""

    def test_straight_rod_does_not_touch_itself(self):
        """Find no contact on a straight rod whose segments are shorter than its radius: its particles within the
        sum of two radii plus the gap of each other along the rod never collide."""
        builder = newton.ModelBuilder()
        _rod(builder, (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), 40, 0.005)
        particles, _, _, _ = _contacts(builder.finalize(device="cpu"))
        self.assertEqual(len(particles), 0)

    def test_folded_rod_touches_itself_between_its_legs(self):
        """Fold a rod into a hairpin whose straight legs lie 1.8 radii apart: each particle of a leg's straight
        part takes contacts with the other leg alone, with the sum of the two radii as margin and the normal
        across the legs."""
        radius, half = 0.004, 0.0036
        angle = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 9)
        leg = np.linspace(0.1, 0.0, 21)
        points = np.concatenate(
            [
                np.c_[leg, np.full(21, -half), np.zeros(21)],
                np.c_[-half * np.cos(angle[1:-1]), half * np.sin(angle[1:-1]), np.zeros(7)],
                np.c_[leg[::-1], np.full(21, half), np.zeros(21)],
            ]
        )
        builder = newton.ModelBuilder()
        SolverFBA.add_particle_rod(builder, points, radius, 1.0e7, density=1000.0)
        particles, alpha, normal, margin = _contacts(builder.finalize(device="cpu"))
        y = points[:, 1]
        self.assertGreater(len(particles), 0)
        for row, n in zip(particles, normal, strict=True):
            used = row[row >= 0]
            # Each contact joins the two legs, across them where it lies on their straight parts.
            self.assertTrue((y[used] < 0.0).any() and (y[used] > 0.0).any())
            if (np.abs(y[used]) == half).all():
                np.testing.assert_allclose(np.abs(n), [0.0, 1.0, 0.0], atol=1.0e-5)
        np.testing.assert_allclose(margin, 2.0 * radius, rtol=1.0e-6)
        np.testing.assert_allclose(alpha.sum(axis=1), 0.0, atol=1.0e-6)
        # Every particle of the straight legs away from the bend takes a contact.
        straight = {i for i in range(len(points)) if abs(y[i]) == half and points[i, 0] > 0.03}
        self.assertTrue(straight <= set(particles[particles >= 0].tolist()))

    def test_particle_over_a_vertex_takes_one_contact(self):
        """Hold a particle right above an interior vertex of a rod: it is nearest to the vertex on both segments
        that share it, and takes one contact with it."""
        builder = newton.ModelBuilder()
        below = _rod(builder, (-0.02, 0.0, 0.0), (0.02, 0.0, 0.0), 4, 0.003)
        above = _rod(builder, (0.0, 0.0, 0.007), (0.0, 0.0, 0.03), 1, 0.005)
        particles, alpha, normal, margin = _contacts(builder.finalize(device="cpu"))
        pair = {below[2], above[0]}
        rows = [k for k, (row, w) in enumerate(zip(particles, alpha, strict=True)) if set(row[w != 0.0]) == pair]
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(margin[rows[0]], 0.008, rtol=1.0e-6)
        np.testing.assert_allclose(np.abs(normal[rows[0]]), [0.0, 0.0, 1.0], atol=1.0e-6)


class TestRodContactSimulation(unittest.TestCase):
    """Rods that SolverFBA keeps apart."""

    def test_crossing_rods_keep_their_radii_apart(self):
        """Drop a thick rod across a thin rod pinned at both ends: it comes to rest draped over it, the segments of
        the two nearest each other the sum of the radii apart; without rod contacts it falls through to the
        ground."""
        r_thin, r_thick = 0.003, 0.008
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        thin = _rod(builder, (-0.1, 0.0, 0.1), (0.1, 0.0, 0.1), 20, r_thin, young=1.0e9)
        builder.particle_mass[thin[0]] = 0.0
        builder.particle_mass[thin[-1]] = 0.0
        thick = _rod(builder, (0.025, -0.1, 0.12), (0.025, 0.1, 0.12), 20, r_thick, bending_stiffness=1.0e-4)
        model = builder.finalize()
        for rod_contact in (True, False):
            x = _settle(model, rod_contact=rod_contact)
            middle = x[thick[10]]
            below = np.interp(middle[0], x[thin, 0], x[thin, 2])
            if rod_contact:
                gap = min(_segment_distance(x[i], x[i + 1], x[j], x[j + 1]) for i in thin[:-1] for j in thick[:-1])
                np.testing.assert_allclose(gap, r_thin + r_thick, atol=5.0e-5)
                self.assertGreater(middle[2], below + r_thin + r_thick - 5.0e-5)
            else:
                self.assertLess(middle[2], below)

    def test_parallel_rods_rest_along_their_length(self):
        """Drop a rod onto a parallel rod pinned at both ends: all along its middle it rests the sum of the radii
        above the pinned rod, which carries it through the vertex-segment contacts of every particle."""
        r_low, r_top = 0.003, 0.004
        builder = newton.ModelBuilder()
        low = _rod(builder, (-0.1, 0.0, 0.1), (0.1, 0.0, 0.1), 20, r_low, young=1.0e9)
        builder.particle_mass[low[0]] = 0.0
        builder.particle_mass[low[-1]] = 0.0
        top = _rod(builder, (-0.05, 0.0, 0.11), (0.05, 0.0, 0.11), 10, r_top, bending_stiffness=1.0e-4)
        model = builder.finalize()
        x = _settle(model, rod_contact=True)
        middle = np.asarray(top[2:-2])
        below = np.interp(x[middle, 0], x[low, 0], x[low, 2])
        np.testing.assert_allclose(x[middle, 2] - below, r_low + r_top, atol=2.0e-4)
        np.testing.assert_allclose(x[middle, 1], 0.0, atol=1.0e-4)

    def test_rod_contact_options_are_validated(self):
        """Reject rod contacts on a model without springs or without friction, and a negative gap."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
        )
        with self.assertRaisesRegex(ValueError, "rod_contact needs springs"):
            SolverFBA(builder.finalize(device="cpu"), rod_contact=True)
        builder = newton.ModelBuilder()
        _rod(builder, (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), 4, 0.005)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "rod_contact needs friction=True"):
            SolverFBA(model, rod_contact=True, friction=False)
        with self.assertRaisesRegex(ValueError, "rod_contact_gap must be >= 0"):
            SolverFBA(model, rod_contact=True, rod_contact_gap=-1.0)


def _crossed_rods(face=False):
    """A thick rod dropped across a thin rod pinned at both ends, over the ground, with rod contacts on the full
    operator; with ``face``, beside a triangle on a pyramid's tip, which only the face contacts find."""
    from newton.tests.fba.test_face_contact import _add_triangle_over_tip, _pyramid_mesh  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    x = 0.0
    if face:
        _add_triangle_over_tip(builder, radius=3e-3)
        builder.add_shape_mesh(-1, mesh=_pyramid_mesh(), cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))
        x = 0.5
    thin = _rod(builder, (x - 0.1, 0.0, 0.1), (x + 0.1, 0.0, 0.1), 20, 0.003, young=1.0e9)
    builder.particle_mass[thin[0]] = 0.0
    builder.particle_mass[thin[-1]] = 0.0
    _rod(builder, (x + 0.025, -0.1, 0.12), (x + 0.025, 0.1, 0.12), 20, 0.008, bending_stiffness=1.0e-4)
    model = builder.finalize()
    options = {"face_contacts": True, "face_contact_gap": 0.01} if face else {}
    solver = SolverFBA(model, iterations=10, nsn_linear_iterations=10, rayleigh_alpha=2.0, rod_contact=True, **options)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.01)
    return model, solver, pipeline, pipeline.contacts(), options


class TestRodContactCapacity(unittest.TestCase):
    """Rod contacts on the full operator's capacity path, which steps recorded into a graph take: found, sorted and
    laid out on the device, on every device."""

    def test_capacity_path_matches_the_count_path(self):
        """Step the thick rod onto the pinned thin one with two solvers from the same states, one through the
        capacity path and one through the host's rows, with the same rows, the rod contacts after the particle
        contacts and, beside a triangle on a pyramid's tip, after its face contacts, and alike to rounding: bit for
        bit on the CPU. Each step starts both from the host path's state, and each solver keeps its own springs'
        duals, which carry over steps."""
        dt = 0.005
        for face in (False, True):
            with self.subTest(face=face):
                model, host, pipeline, contacts, options = _crossed_rods(face)
                capacity = SolverFBA(
                    model, iterations=10, nsn_linear_iterations=10, rayleigh_alpha=2.0, rod_contact=True, **options
                )
                s0, s1, s2 = model.state(), model.state(), model.state()
                rods = faces = 0
                worst = 0.0
                for k in range(60):
                    pipeline.collide(s0, contacts)
                    host.step(s0, s1, None, contacts, dt)
                    if k == 1:
                        # The steps recorded into a graph take the capacity path; this takes it without a graph.
                        capacity._recording = lambda: True
                    capacity.step(s0, s2, None, contacts, dt)
                    if k:
                        self.assertEqual(int(capacity._capacity_active_d.numpy()[0]), host._contact_count)
                        first = int(capacity._rod_contact.first.numpy()[0])
                        found = int(capacity._rod_contact.sorted_count.numpy()[0])
                        rods += found > 0
                        if face:
                            faces += found > 0 and first > int(capacity._capacity_face.first.numpy()[0])
                    worst = max(worst, float(np.abs(s2.particle_q.numpy() - s1.particle_q.numpy()).max()))
                    s0, s1 = s1, s0
                self.assertGreater(rods, 40)
                if face:
                    self.assertGreater(faces, 10)
                if wp.get_device(model.device).is_cpu:
                    self.assertEqual(worst, 0.0)
                self.assertLess(worst, ROD_CAPACITY_ATOL)
                self.assertEqual(capacity.dropped_contacts(), 0)


@unittest.skipUnless(wp.is_cuda_available(), "graphs record on CUDA devices")
class TestRodContactGraphs(unittest.TestCase):
    """Steps of rod contacts recorded into a CUDA graph."""

    def test_rod_contacts_replay_from_a_graph(self):
        """Record the collision and two steps of the rods into a graph after a step outside it, replay it, and reach
        the states of the same steps taken through the capacity path without a graph, bit for bit."""
        dt, results = 0.005, []
        for graph in (False, True):
            model, solver, pipeline, contacts, _ = _crossed_rods()
            s0, s1 = model.state(), model.state()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, dt)
            s0, s1 = s1, s0
            if not graph:
                solver._recording = lambda: True

            def two_steps(s0=s0, s1=s1, solver=solver, pipeline=pipeline, contacts=contacts):
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, None, contacts, dt)
                pipeline.collide(s1, contacts)
                solver.step(s1, s0, None, contacts, dt)

            if graph:
                with wp.ScopedCapture() as capture:
                    two_steps()
                for _ in range(30):
                    wp.capture_launch(capture.graph)
            else:
                for _ in range(30):
                    two_steps()
            results.append(s0.particle_q.numpy())
        self.assertGreater(float(np.abs(results[0] - results[0][0]).max()), 0.0)
        np.testing.assert_array_equal(results[1], results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
