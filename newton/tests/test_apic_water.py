# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`~newton.solvers.SolverAPICWater`: water at rest, buoyancy, a dam break against Martin and Moyce's
measurements, particles leaving the grid's box, streams, CUDA graph replay and the configuration errors."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverAPICWater
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DX = 0.01
SPACING = 0.5 * DX
DENSITY = 1000.0
ACTIVE = int(newton.ParticleFlags.ACTIVE)


def _lattice(lo, hi, spacing=SPACING):
    """Points ``spacing`` apart filling the box ``[lo, hi]``, half a spacing in from its sides."""
    axes = [np.arange(a + 0.5 * spacing, b - 1.0e-9, spacing) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def _water(device, points, pool=0, velocity=(0.0, 0.0, 0.0)):
    """A model of water particles at ``points``, active, followed by ``pool`` inactive ones."""
    builder = newton.ModelBuilder()
    SolverAPICWater.register_custom_attributes(builder)
    count = len(points) + pool
    positions = np.concatenate([points, np.zeros((pool, 3))]) if pool else points
    builder.add_particles(
        pos=[tuple(p) for p in positions],
        vel=[tuple(velocity)] * len(points) + [(0.0, 0.0, 0.0)] * pool,
        mass=[DENSITY * SPACING**3] * count,
        radius=[0.5 * SPACING] * count,
        flags=[ACTIVE] * len(points) + [0] * pool,
    )
    return builder.finalize(device=device)


def _tank(device, size, wall=0.02, body_box=None):
    """A rigid model: the ground plane and four walls, static boxes around ``[0, size]``, and optionally a body with a
    box ``body_box = (center, half extents)``, all colliding with particles."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_particle_collision=True, has_shape_collision=False)
    builder.add_ground_plane(height=0.0, cfg=cfg)
    sx, sy, sz = size
    walls = (
        ((-wall, 0.5 * sy, 0.5 * sz), (wall, 0.5 * sy + 2 * wall, 0.5 * sz)),
        ((sx + wall, 0.5 * sy, 0.5 * sz), (wall, 0.5 * sy + 2 * wall, 0.5 * sz)),
        ((0.5 * sx, -wall, 0.5 * sz), (0.5 * sx + 2 * wall, wall, 0.5 * sz)),
        ((0.5 * sx, sy + wall, 0.5 * sz), (0.5 * sx + 2 * wall, wall, 0.5 * sz)),
    )
    for center, half in walls:
        xform = wp.transform(center, wp.quat_identity())
        builder.add_shape_box(-1, xform=xform, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    body = None
    if body_box is not None:
        center, half = body_box
        body = builder.add_body(xform=wp.transform(center, wp.quat_identity()))
        body_cfg = newton.ModelBuilder.ShapeConfig(
            density=500.0, has_particle_collision=True, has_shape_collision=False
        )
        builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=body_cfg)
    return builder.finalize(device=device), body


def _solver(water, lo, hi, **config):
    return SolverAPICWater(water, SolverAPICWater.Config(voxel_size=DX, domain_lo=lo, domain_hi=hi, **config))


def _active(water):
    return (water.particle_flags.numpy() & ACTIVE) != 0


class TestAPICWater(unittest.TestCase):
    pass


def test_still_pool_stays_at_rest(test, device):
    """A pool at rest in a tank of box shapes, its walls and floor in node planes, stays at rest with its depth and
    inside the tank, stepped eagerly and replayed from a CUDA graph alike."""
    water = _water(device, _lattice((0.0, 0.0, 0.0), (0.1, 0.08, 0.06)))
    rigid, _ = _tank(device, (0.1, 0.08, 0.2))
    solver = _solver(water, (-0.04, -0.04, -0.04), (0.14, 0.12, 0.12))
    solver.setup_colliders(rigid)
    state = water.state()
    z0 = float(state.particle_q.numpy()[:, 2].mean())
    for _ in range(100):
        solver.step(state, state, None, None, 1.0 / 300.0)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, state, None, None, 1.0 / 300.0)
    for _ in range(100):
        wp.capture_launch(capture.graph)
    q, v = state.particle_q.numpy(), state.particle_qd.numpy()
    test.assertTrue(_active(water).all())
    test.assertLess(float(np.abs(v).max()), 0.01)
    test.assertAlmostEqual(float(q[:, 2].mean()), z0, delta=5.0e-4)
    test.assertTrue(np.all(q > -1.0e-6) and np.all(q[:, :2] < np.array([0.1, 0.08]) + 1.0e-6))
    test.assertEqual(solver.dropped_fluid_cells(), 0)


def test_buoyancy_on_a_fixed_box(test, device):
    """A fixed box body half in a pool takes the pressure's impulse of the displaced water's weight, upward, without
    a sideways force or a torque about its center of mass."""
    half = np.array([0.05, 0.03, 0.03])
    center = np.array([0.15, 0.1, 0.1])
    points = _lattice((0.0, 0.0, 0.0), (0.3, 0.2, 0.1))
    points = points[~np.all(np.abs(points - center) < half + 0.25 * SPACING, axis=1)]
    water = _water(device, points)
    rigid, body = _tank(device, (0.3, 0.2, 0.3), body_box=(tuple(center), tuple(half)))
    solver = _solver(water, (-0.05, -0.05, -0.05), (0.35, 0.25, 0.3))
    solver.setup_colliders(rigid)
    rigid_state = rigid.state()
    state = water.state()
    state.body_q, state.body_qd = rigid_state.body_q, rigid_state.body_qd
    dt = 1.0 / 300.0
    impulses = []
    for k in range(240):
        solver.step(state, state, None, None, dt)
        if k >= 180:
            impulses.append(solver.body_impulse.numpy()[body].copy())
    wrench = np.mean(impulses, axis=0) / dt
    # The pool keeps its depth of 0.1 m: the box displaces the water it replaced.
    buoyancy = DENSITY * 9.81 * (2.0 * half[0]) * (2.0 * half[1]) * half[2]
    test.assertAlmostEqual(float(wrench[2]), buoyancy, delta=0.02 * buoyancy)
    test.assertLess(float(np.abs(wrench[:2]).max()), 1.0e-3 * buoyancy)
    test.assertLess(float(np.abs(wrench[3:]).max()), 1.0e-4 * buoyancy)
    # The impulse becomes the body's force over the next step.
    body_f = wp.zeros(rigid.body_count, dtype=wp.spatial_vector, device=device)
    solver.add_body_forces(body_f, dt)
    np.testing.assert_allclose(body_f.numpy()[body], impulses[-1] / dt, rtol=1.0e-6)


def test_dam_break_front(test, device):
    """A column of water ``a`` wide and high collapses along a tank: its front at ``T = t sqrt(2 g / a)`` lies where
    Martin and Moyce (1952) measured it, ``Z = x / a`` of about 1.5 at ``T = 1.14`` and 2.6 at ``T = 2.29``."""
    a = 0.15
    water = _water(device, _lattice((0.0, 0.0, 0.0), (a, 0.1, a)))
    rigid, _ = _tank(device, (0.6, 0.1, 0.4))
    solver = _solver(water, (-0.05, -0.05, -0.05), (0.65, 0.15, 0.3))
    solver.setup_colliders(rigid)
    state = water.state()
    dt = 1.0 / 400.0
    fronts = []
    for steps in (40, 40):
        for _ in range(steps):
            solver.step(state, state, None, None, dt)
        fronts.append(float(state.particle_q.numpy()[:, 0].max()) / a)
    T = [0.1 * math.sqrt(2.0 * 9.81 / a), 0.2 * math.sqrt(2.0 * 9.81 / a)]
    test.assertAlmostEqual(T[1], 2.29, delta=0.01)
    test.assertTrue(1.35 < fronts[0] < 1.75, fronts)
    test.assertTrue(2.35 < fronts[1] < 2.85, fronts)
    test.assertTrue(_active(water).all())


def test_particles_leave_through_the_box(test, device):
    """Particles that fly out of the grid's box leave the water: inactive, at rest, out of the way; the others keep
    going."""
    points = _lattice((0.05, 0.05, 0.05), (0.09, 0.09, 0.09))
    water = _water(device, points, velocity=(2.0, 0.0, 0.0))
    solver = _solver(water, (0.0, 0.0, 0.0), (0.2, 0.15, 0.15))
    state = water.state()
    for _ in range(30):
        solver.step(state, state, None, None, 1.0 / 400.0)
    active = _active(water)
    test.assertFalse(active.any())
    test.assertTrue(np.all(state.particle_qd.numpy() == 0.0))
    test.assertTrue(np.all(state.particle_q.numpy()[:, 0] < -100.0))


def test_stream_emits_whole_layers(test, device):
    """A stream brings inactive particles in, a 3 x 3 layer per interval, each placed along its flight from the
    origin, and waits when the inactive particles run short of a layer; particles that leave go back to it."""
    pool = 9 * 20 + 4
    water = _water(device, np.zeros((0, 3)), pool=pool)
    solver = _solver(water, (0.0, 0.0, 0.0), (0.2, 0.2, 0.3))
    flow = 9 * SPACING**3 / 0.01  # a layer every 10 ms
    origin = np.array([0.1, 0.1, 0.25])
    solver.add_stream(origin, (0.0, 0.0, -1.0), flow, side_count=3)
    state = water.state()
    dt = 1.0 / 400.0
    counts = []
    for _ in range(42):
        solver.step(state, state, None, None, dt)
        counts.append(int(_active(water).sum()))
    # Layers leave at 0, 10, 20 ... ms; by 0.105 s eleven have left, every one whole.
    test.assertEqual(counts[-1], 99)
    test.assertTrue(all(c % 9 == 0 for c in counts))
    q = state.particle_q.numpy()[_active(water)]
    test.assertTrue(np.all(np.abs(q[:, :2] - origin[:2]) < 2.5 * SPACING))
    test.assertTrue(np.all(q[:, 2] < origin[2] + 1.0e-6))
    for _ in range(200):
        solver.step(state, state, None, None, dt)
    # The water falls out of the box's bottom about 0.17 s after it leaves, back into the pool, which holds 20 layers
    # and four more particles: the stream keeps flowing.
    count = int(_active(water).sum())
    test.assertTrue(9 * 12 <= count <= 9 * 20, count)


def test_configuration_errors(test, device):
    """The solver needs CUDA, the custom attributes and an open box; colliders need a signed distance."""
    builder = newton.ModelBuilder()
    builder.add_particles(pos=[(0.0, 0.0, 0.0)], vel=[(0.0, 0.0, 0.0)], mass=[1.0e-4], radius=[1.0e-3])
    plain = builder.finalize(device=device)
    with test.assertRaisesRegex(ValueError, "custom attributes"):
        SolverAPICWater(plain)
    water = _water(device, np.zeros((1, 3)))
    with test.assertRaisesRegex(ValueError, "empty grid box"):
        _solver(water, (0.0, 0.0, 0.0), (0.0, 0.1, 0.1))
    water_cpu = _water("cpu", np.zeros((1, 3)))
    with test.assertRaisesRegex(ValueError, "CUDA"):
        SolverAPICWater(water_cpu)
    rigid_builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(has_particle_collision=True)
    heightfield = newton.Heightfield(
        data=np.zeros((3, 3), dtype=np.float32), nrow=3, ncol=3, hx=0.1, hy=0.1, min_z=0.0, max_z=0.0
    )
    rigid_builder.add_shape_heightfield(heightfield=heightfield, cfg=cfg)
    rigid = rigid_builder.finalize(device=device)
    solver = _solver(water, (0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    with test.assertRaisesRegex(ValueError, "HFIELD"):
        solver.setup_colliders(rigid)


devices = get_cuda_test_devices()
for name, function in (
    ("test_still_pool_stays_at_rest", test_still_pool_stays_at_rest),
    ("test_buoyancy_on_a_fixed_box", test_buoyancy_on_a_fixed_box),
    ("test_dam_break_front", test_dam_break_front),
    ("test_particles_leave_through_the_box", test_particles_leave_through_the_box),
    ("test_stream_emits_whole_layers", test_stream_emits_whole_layers),
    ("test_configuration_errors", test_configuration_errors),
):
    add_function_test(TestAPICWater, name, function, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
