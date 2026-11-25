# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverFBA's segment contacts of rods with shapes (``rod_shape_contacts``)."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.fba.face_contact import SegmentContactQuery

BAR_R, ROD_R = 0.001, 0.001


def _bar_and_segment(device, x0, x1, z):
    """A static 1 mm bar along y at the origin and a segment of two 1 mm particles from (x0, 0, z) to (x1, 0, z)."""
    builder = newton.ModelBuilder()
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2)  # capsule axis z -> world y
    builder.add_shape_capsule(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), q), radius=BAR_R, half_height=0.02)
    for x in (x0, x1):
        builder.add_particle((x, 0.0, z), (0.0, 0.0, 0.0), 0.001, radius=ROD_R)
    return builder.finalize(device=device)


class TestSegmentContactQuery(unittest.TestCase):
    def _query(self, x0, x1, z, gap=0.0):
        model = _bar_and_segment("cpu", x0, x1, z)
        query = SegmentContactQuery(np.array([[0, 1]]), "cpu")
        return query.query(model, model.particle_q, None, 5e-4, gap)

    def test_bar_between_the_particles(self):
        """A segment lying across the bar, its particles clear of it to either side, takes one contact at its middle,
        its normal up from the bar."""
        segment, shape, weights, _, normal = self._query(-0.003, 0.003, BAR_R + ROD_R - 0.0002)
        self.assertEqual(len(segment), 1)
        self.assertEqual(int(shape[0]), 0)
        np.testing.assert_allclose(weights[0], [0.5, 0.5], atol=1e-3)
        self.assertAlmostEqual(float(weights[0].sum()), 1.0, places=6)
        np.testing.assert_allclose(normal[0], [0.0, 0.0, 1.0], atol=1e-3)

    def test_clear_segment_takes_none(self):
        """A segment more than its radius and the gap above the bar takes no contact."""
        segment, *_ = self._query(-0.003, 0.003, BAR_R + ROD_R + 0.001, gap=0.0005)
        self.assertEqual(len(segment), 0)

    def test_deepest_at_a_particle_is_the_particle_contact(self):
        """A segment whose deepest point is a particle (one end over the bar) takes no segment contact."""
        segment, *_ = self._query(0.0, 0.006, BAR_R + ROD_R - 0.0002)
        self.assertEqual(len(segment), 0)


ZB = 0.4  # height [m] of the bar of _string_on_bar


def _string_on_bar(segments=True, held=(), **options):
    """A string with 6 mm segments (4 mm between its particles' spheres) over a static bar of 1 mm radius at height
    ``ZB``, 10 g on one end and 15 g on the other, the particles ``held`` (indices along it) pinned, and its solver
    with segment contacts and ``options``, pipeline and contacts."""
    from newton.solvers import SolverFBA  # noqa: PLC0415

    seg, zb = 0.006, ZB
    rc = BAR_R + ROD_R + 0.0005
    pts = [(-rc, 0.0, z) for z in np.arange(zb - 0.15, zb, seg)]
    pts += [(rc * math.cos(a), 0.0, zb + rc * math.sin(a)) for a in np.linspace(math.pi, 0.0, 5)[1:-1]]
    pts += [(rc, 0.0, z) for z in np.arange(zb, zb - 0.15, -seg)]
    builder = newton.ModelBuilder()
    area = math.pi * ROD_R**2
    rope = SolverFBA.add_particle_rod(
        builder, np.array(pts), ROD_R, 1e3 / area, density=0.002 / area, bending_stiffness=1e-7
    )
    builder.particle_mass[rope[0]] += 0.010
    builder.particle_mass[rope[-1]] += 0.015
    for k in held:
        builder.particle_mass[rope[k]] = 0.0
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2)
    builder.add_shape_capsule(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, zb), q),
        radius=BAR_R,
        half_height=0.02,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, gap=0.003),
    )
    model = builder.finalize()
    solver = SolverFBA(
        model,
        iterations=20,
        nsn_linear_iterations=10,
        rayleigh_beta=1e-4,
        rayleigh_corotated=True,
        rod_shape_contacts=segments,
        face_contact_edge_radius=5e-4,
        **options,
    )
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.003)
    return model, rope, solver, pipeline, pipeline.contacts()


class TestRodOverThinBar(unittest.TestCase):
    """The string of :func:`_string_on_bar`: without segment contacts the bar passes between two particles and the
    string drops off; with them it slides over the bar and stays on."""

    def _run(self, segments, **options):
        model, rope, solver, pipeline, contacts = _string_on_bar(segments, **options)
        s0, s1 = model.state(), model.state()
        for _ in range(300):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, None, contacts, 2e-3)
            s0, s1 = s1, s0
        x = s0.particle_q.numpy()[rope]
        return bool(x[:, 2].max() > ZB and x[0, 0] < 0.0 < x[-1, 0])

    def test_segments_keep_the_string_on_the_bar(self):
        self.assertFalse(self._run(False))
        self.assertTrue(self._run(True))

    def test_segments_with_rod_contacts(self):
        """Keep the string on the bar with rod contacts too, found between its legs within a 4 mm gap: the rows of
        the segment contacts, with one further particle, join those of the rod contacts, with three."""
        self.assertTrue(self._run(True, rod_contact=True, rod_contact_gap=0.004))


class TestRodShapeContactCapacity(unittest.TestCase):
    """Segment contacts on the full operator's capacity path, which steps recorded into a graph take: found and laid
    out on the device, on every device."""

    def test_capacity_path_matches_the_count_path(self):
        """Step the string over the bar with two solvers from the same states, one through the capacity path and one
        through the host's rows, with the same rows, the segment contacts after the particle contacts and before the
        rod contacts, and alike to rounding: bit for bit on the CPU. With the two particles at the bar's top held, the
        segment between them, on held particles alone, drops its contacts on both paths. Each step starts both from
        the host path's state, and each solver keeps its own springs' duals, which carry over steps."""
        from newton.solvers import SolverFBA  # noqa: PLC0415

        dt = 2e-3
        for case, held, options in (
            ("segments", (), {}),
            ("rods", (), {"rod_contact": True, "rod_contact_gap": 0.004}),
            ("held", (26, 27), {}),
        ):
            with self.subTest(case=case):
                model, _, host, pipeline, contacts = _string_on_bar(True, held, **options)
                capacity = SolverFBA(
                    model,
                    iterations=20,
                    nsn_linear_iterations=10,
                    rayleigh_beta=1e-4,
                    rayleigh_corotated=True,
                    rod_shape_contacts=True,
                    face_contact_edge_radius=5e-4,
                    **options,
                )
                s0, s1, s2 = model.state(), model.state(), model.state()
                found = dropped = 0
                worst = 0.0
                for k in range(150):
                    pipeline.collide(s0, contacts)
                    host.step(s0, s1, None, contacts, dt)
                    if k == 1:
                        # The steps recorded into a graph take the capacity path; this takes it without a graph.
                        capacity._recording = lambda: True
                    capacity.step(s0, s2, None, contacts, dt)
                    if k:
                        self.assertEqual(int(capacity._capacity_active_d.numpy()[0]), host._contact_count)
                        kept = int(capacity._segment_query._cap.found.numpy()[0])
                        query = host._segment_query.query(model, s0.particle_q, None, 5e-4, host.rod_shape_contact_gap)
                        found += kept > 0
                        dropped += len(query[0]) - kept
                    worst = max(worst, float(np.abs(s2.particle_q.numpy() - s1.particle_q.numpy()).max()))
                    s0, s1 = s1, s0
                self.assertGreater(found, 100)
                if held:
                    self.assertGreater(dropped, 100)
                else:
                    self.assertEqual(dropped, 0)
                if wp.get_device(model.device).is_cpu:
                    self.assertEqual(worst, 0.0)
                self.assertLess(worst, 1.0e-6)
                self.assertEqual(capacity.dropped_contacts(), 0)


@unittest.skipUnless(wp.is_cuda_available(), "graphs record on CUDA devices")
class TestRodShapeContactGraphs(unittest.TestCase):
    """Steps of segment contacts recorded into a CUDA graph."""

    def test_segment_contacts_replay_from_a_graph(self):
        """Record the collision and two steps of the string over the bar into a graph after a step outside it, replay
        it, and reach the states of the same steps taken through the capacity path without a graph, bit for bit."""
        dt, results = 2e-3, []
        for graph in (False, True):
            model, _, solver, pipeline, contacts = _string_on_bar()
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
                for _ in range(60):
                    wp.capture_launch(capture.graph)
            else:
                for _ in range(60):
                    two_steps()
            results.append(s0.particle_q.numpy())
        np.testing.assert_array_equal(results[0], results[1])
        self.assertEqual(solver.dropped_contacts(), 0)


class TestRodShapeContactOptions(unittest.TestCase):
    def test_options_are_validated(self):
        from newton.solvers import SolverFBA  # noqa: PLC0415

        builder = newton.ModelBuilder()
        builder.add_particle((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.001, radius=0.001)
        with self.assertRaises(ValueError):  # no springs to collide
            SolverFBA(builder.finalize(device="cpu"), rod_shape_contacts=True)
        builder = newton.ModelBuilder()
        SolverFBA.add_particle_rod(builder, np.array([(0.0, 0.0, 1.0), (0.01, 0.0, 1.0)]), 0.001, 1e9, density=1e3)
        model = builder.finalize(device="cpu")
        with self.assertRaises(ValueError):  # the lite operator
            SolverFBA(model, rod_shape_contacts=True, nsn_schur_mode="lite")
        with self.assertRaises(ValueError):
            SolverFBA(model, rod_shape_contacts=True, rod_shape_contact_gap=-0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
