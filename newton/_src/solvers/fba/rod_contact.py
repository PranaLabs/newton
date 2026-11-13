# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contacts of springs with each other as capsules, such as the segments of particle rods, for
:class:`~newton.solvers.SolverFBA`."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ...sim import Model
from .kernels import host_head
from .self_contact import _in_ring, iota_kernel


@wp.kernel
def capsule_bounds_kernel(
    q: wp.array[wp.vec3],
    segments: wp.array[wp.vec2i],
    radius: wp.array[wp.float32],
    half_gap: float,
    # outputs
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    """The bounds of each segment's capsule, grown by half the detection gap."""
    e = wp.tid()
    a = segments[e][0]
    b = segments[e][1]
    r = wp.vec3(wp.max(radius[a], radius[b]) + half_gap)
    lower[e] = wp.min(q[a], q[b]) - r
    upper[e] = wp.max(q[a], q[b]) + r


@wp.func
def _store(
    k: wp.int32,
    key: wp.int64,
    p0: wp.int32,
    p1: wp.int32,
    p2: wp.int32,
    p3: wp.int32,
    w: wp.vec4,
    n: wp.vec3,
    reach: float,
    keys: wp.array[wp.int64],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    margin: wp.array[wp.float32],
):
    if k < particles.shape[0]:
        keys[k] = key
        particles[k, 0] = p0
        particles[k, 1] = p1
        particles[k, 2] = p2
        particles[k, 3] = p3
        for j in range(4):
            alpha[k, j] = w[j]
        normal[k] = n
        margin[k] = reach


@wp.func
def _in_vertex_region(
    x: wp.vec3,
    v: wp.int32,
    q: wp.array[wp.vec3],
    segments: wp.array[wp.vec2i],
    adjacent_offsets: wp.array[wp.int32],
    adjacent: wp.array[wp.int32],
) -> bool:
    """Whether ``x`` projects onto none of the segments at particle ``v`` beyond ``v``, so that ``v`` is the
    closest point of each of them."""
    for k in range(adjacent_offsets[v], adjacent_offsets[v + 1]):
        e = adjacent[k]
        w = segments[e][0]
        if w == v:
            w = segments[e][1]
        if wp.dot(x - q[v], q[w] - q[v]) > 0.0:
            return False
    return True


@wp.kernel
def vertex_segment_contacts_kernel(
    q: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    segments: wp.array[wp.vec2i],
    radius: wp.array[wp.float32],
    owner: wp.array[wp.int32],
    adjacent_offsets: wp.array[wp.int32],
    adjacent: wp.array[wp.int32],
    bvh: wp.uint64,
    ring_offsets: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    gap: float,
    # outputs
    count: wp.array[wp.int32],
    keys: wp.array[wp.int64],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    margin: wp.array[wp.float32],
):
    """Each particle of a segment and each segment whose capsule its sphere comes within ``gap`` of, with no
    particles within each other's rings: a contact on the particle and the segment's two particles. A closest point
    at the segment's end is a particle: the two particles take a contact where each is the closest point of every
    segment at the other, once, from the lower particle and the other's first segment, ``owner``; elsewhere the
    contacts with the segments' insides hold them."""
    p = wp.tid()
    if owner[p] < 0:
        return
    x = q[p]
    r = radius[p]
    query = wp.bvh_query_aabb(bvh, x - wp.vec3(r + 0.5 * gap), x + wp.vec3(r + 0.5 * gap))
    e = wp.int32(0)
    n_segments = segments.shape[0]
    while wp.bvh_query_next(query, e):
        b0 = segments[e][0]
        b1 = segments[e][1]
        if world[p] >= 0 and world[b0] >= 0 and world[p] != world[b0]:
            continue
        d1 = q[b1] - q[b0]
        length2 = wp.dot(d1, d1)
        u = float(0.0)
        if length2 > 0.0:
            u = wp.clamp(wp.dot(x - q[b0], d1) / length2, 0.0, 1.0)
        c = q[b0] + u * d1
        d = wp.length(x - c)
        reach = r + (1.0 - u) * radius[b0] + u * radius[b1]
        if d >= reach + gap or d < 1.0e-6 * reach:
            continue
        if u <= 0.0 or u >= 1.0:
            b = b0
            if u >= 1.0:
                b = b1
            if b <= p or owner[b] != e:
                continue
            if not _in_vertex_region(x, b, q, segments, adjacent_offsets, adjacent):
                continue
            if not _in_vertex_region(q[b], p, q, segments, adjacent_offsets, adjacent):
                continue
        if _in_ring(ring_offsets, ring, p, b0) or _in_ring(ring_offsets, ring, p, b1):
            continue
        k = wp.atomic_add(count, 0, 1)
        key = wp.int64(p) * wp.int64(n_segments) + wp.int64(e)
        w = wp.vec4(1.0, u - 1.0, -u, 0.0)
        _store(k, key, p, b0, b1, -1, w, (x - c) / d, reach, keys, particles, alpha, normal, margin)


@wp.kernel
def segment_segment_contacts_kernel(
    q: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    segments: wp.array[wp.vec2i],
    radius: wp.array[wp.float32],
    bvh: wp.uint64,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    ring_offsets: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    gap: float,
    key_base: wp.int64,
    # outputs
    count: wp.array[wp.int32],
    keys: wp.array[wp.int64],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    margin: wp.array[wp.float32],
):
    """Each pair of segments, at least 1.8 degrees apart in direction, whose capsules come within ``gap`` of each
    other at closest points inside both segments, with no particles within each other's rings, once: a contact on
    their four particles with the weights of the two closest points. Closest points at a particle, and those of
    nearly parallel segments, are left to the vertex-segment contacts."""
    ea = wp.tid()
    a0 = segments[ea][0]
    a1 = segments[ea][1]
    pa0 = q[a0]
    pa1 = q[a1]
    query = wp.bvh_query_aabb(bvh, lower[ea], upper[ea])
    eb = wp.int32(0)
    n_segments = segments.shape[0]
    while wp.bvh_query_next(query, eb):
        if eb <= ea:
            continue
        b0 = segments[eb][0]
        b1 = segments[eb][1]
        if world[a0] >= 0 and world[b0] >= 0 and world[a0] != world[b0]:
            continue
        pb0 = q[b0]
        pb1 = q[b1]
        # Nearly parallel segments have no well-defined closest points; their particles' contacts hold them apart.
        da = pa1 - pa0
        db = pb1 - pb0
        if wp.length_sq(wp.cross(da, db)) < 1.0e-3 * wp.length_sq(da) * wp.length_sq(db):
            continue
        scale = wp.max(radius[a0], radius[a1]) + wp.max(radius[b0], radius[b1])
        st = wp.closest_point_edge_edge(pa0, pa1, pb0, pb1, 1.0e-6 * scale * scale)
        s = st[0]
        u = st[1]
        d = st[2]
        if s <= 1.0e-3 or s >= 1.0 - 1.0e-3 or u <= 1.0e-3 or u >= 1.0 - 1.0e-3:
            continue
        reach = (1.0 - s) * radius[a0] + s * radius[a1] + (1.0 - u) * radius[b0] + u * radius[b1]
        if d >= reach + gap or d < 1.0e-6 * reach:
            continue
        if (
            _in_ring(ring_offsets, ring, a0, b0)
            or _in_ring(ring_offsets, ring, a0, b1)
            or _in_ring(ring_offsets, ring, a1, b0)
            or _in_ring(ring_offsets, ring, a1, b1)
        ):
            continue
        k = wp.atomic_add(count, 0, 1)
        key = key_base + wp.int64(ea) * wp.int64(n_segments) + wp.int64(eb)
        w = wp.vec4(1.0 - s, s, u - 1.0, -u)
        n = (pa0 + s * (pa1 - pa0) - (pb0 + u * (pb1 - pb0))) / d
        _store(k, key, a0, a1, b0, b1, w, n, reach, keys, particles, alpha, normal, margin)


@wp.kernel
def gather_capsule_contacts_kernel(
    order: wp.array[wp.int32],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    margin: wp.array[wp.float32],
    # outputs
    out_particles: wp.array2d[wp.int32],
    out_alpha: wp.array2d[wp.float32],
    out_normal: wp.array[wp.vec3],
    out_margin: wp.array[wp.float32],
):
    """The contacts in the order of their keys."""
    k = wp.tid()
    i = order[k]
    for j in range(4):
        out_particles[k, j] = particles[i, j]
        out_alpha[k, j] = alpha[i, j]
    out_normal[k] = normal[i]
    out_margin[k] = margin[i]


def segment_rings(segments: np.ndarray, particle_count: int, rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted CSR of the particles within ``rings`` segments of each particle along the segments, the particle
    included."""
    neighbors = [[] for _ in range(particle_count)]
    for a, b in segments.tolist():
        neighbors[a].append(b)
        neighbors[b].append(a)
    reached = []
    for v in range(particle_count):
        reach, front = {v}, {v}
        for _ in range(rings):
            front = {w for u in front for w in neighbors[u]} - reach
            reach |= front
        reached.append(sorted(reach))
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in reached])]).astype(np.int32)
    return offsets, np.concatenate(reached).astype(np.int32)


class RodContact:
    """Contacts of segments, such as the springs of particle rods, with each other as capsules around them, found on
    the positions at the start of a step.

    A segment joins two particles, and its capsule's radius runs linearly between theirs. Each particle of a segment
    takes a contact with each segment whose capsule its sphere comes within ``gap`` of where its closest point lies
    inside the segment, two particles take one where each is the closest point of all segments at the other, and
    two segments whose closest points lie inside both take one on their four particles. A contact keeps its points the sum of the radii
    there apart, along the direction between them at detection. Particles within ``rings`` segments of each other
    along the segments never collide: by default as many as the largest sum of two radii plus the gap spans at the
    shortest segment, so that a straight rod never touches itself.

    Args:
        model: Model with the particles.
        segments: Particle pairs of the segments, shape ``(S, 2)``.
        gap: Detection distance [m] beyond the capsules' surfaces.
        rings: Segments within which particles do not collide, or ``None`` for the default.
    """

    def __init__(self, model: Model, segments: np.ndarray, gap: float, rings: int | None = None):
        segments = np.asarray(segments, dtype=np.int32).reshape(-1, 2)
        self.device, self.gap = model.device, float(gap)
        self.world, self.radius = model.particle_world, model.particle_radius
        self.segments = wp.array(segments, dtype=wp.vec2i, device=self.device)
        n = segments.shape[0]
        if rings is None:
            q = model.particle_q.numpy()
            r = model.particle_radius.numpy()
            length = np.linalg.norm(q[segments[:, 1]] - q[segments[:, 0]], axis=1)
            rings = max(1, math.ceil((2.0 * float(r[segments].max()) + self.gap) / float(length.min())))
        self.rings = int(rings)
        offsets, ring = segment_rings(segments, model.particle_count, self.rings)
        self.ring_offsets, self.ring = (wp.array(a, dtype=wp.int32, device=self.device) for a in (offsets, ring))
        # The first segment of each particle, -1 for particles on none.
        owner = np.full(model.particle_count, -1, dtype=np.int32)
        for e in range(n - 1, -1, -1):
            owner[segments[e]] = e
        self.owner = wp.array(owner, dtype=wp.int32, device=self.device)
        # The segments at each particle, as a CSR.
        order = np.argsort(segments.ravel(), kind="stable")
        counts = np.bincount(segments.ravel(), minlength=model.particle_count)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        self.adjacent_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)
        self.adjacent = wp.array((order // 2).astype(np.int32), dtype=wp.int32, device=self.device)
        self.lower = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.upper = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._bounds(model.particle_q)
        self.bvh = wp.Bvh(self.lower, self.upper)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._allocate(4 * n)

    def _bounds(self, q: wp.array) -> None:
        inputs = [q, self.segments, self.radius, 0.5 * self.gap]
        wp.launch(capsule_bounds_kernel, self.segments.shape[0], inputs, [self.lower, self.upper], device=self.device)

    def _allocate(self, capacity: int) -> None:
        # The sort takes twice the keys it sorts.
        self.keys = wp.empty(2 * capacity, dtype=wp.int64, device=self.device)
        self.order = wp.empty(2 * capacity, dtype=wp.int32, device=self.device)
        self.particles = wp.empty((capacity, 4), dtype=wp.int32, device=self.device)
        self.alpha = wp.empty((capacity, 4), dtype=wp.float32, device=self.device)
        self.normal = wp.empty(capacity, dtype=wp.vec3, device=self.device)
        self.margin = wp.empty(capacity, dtype=wp.float32, device=self.device)

    def detect(self, q: wp.array) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Particles and coefficients, shape ``(C, 4)`` with particle ``-1`` in unused slots, normals, shape
        ``(C, 3)``, and margins [m], shape ``(C,)``, of the contacts at ``q``: the vertex-segment contacts in the
        order of their particles and segments, then the segment-segment contacts in the order of their pairs."""
        self._bounds(q)
        self.bvh.refit()
        n_segments, n_particles = self.segments.shape[0], q.shape[0]
        while True:
            self.count.zero_()
            outputs = [self.count, self.keys, self.particles, self.alpha, self.normal, self.margin]
            inputs = [q, self.world, self.segments, self.radius, self.owner, self.adjacent_offsets, self.adjacent]
            inputs += [self.bvh.id, self.ring_offsets, self.ring, self.gap]
            wp.launch(vertex_segment_contacts_kernel, n_particles, inputs, outputs, device=self.device)
            inputs = [q, self.world, self.segments, self.radius, self.bvh.id, self.lower, self.upper]
            inputs += [self.ring_offsets, self.ring, self.gap, wp.int64(n_particles * n_segments)]
            wp.launch(segment_segment_contacts_kernel, n_segments, inputs, outputs, device=self.device)
            n = int(self.count.numpy()[0])
            if n <= self.particles.shape[0]:
                break
            self._allocate(2 * n)
        if n == 0:
            return (
                np.zeros((0, 4), dtype=np.int32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        wp.launch(iota_kernel, n, outputs=[self.order], device=self.device)
        wp.utils.radix_sort_pairs(self.keys, self.order, n)
        out = [
            wp.empty((n, 4), dtype=wp.int32, device=self.device),
            wp.empty((n, 4), dtype=wp.float32, device=self.device),
            wp.empty(n, dtype=wp.vec3, device=self.device),
            wp.empty(n, dtype=wp.float32, device=self.device),
        ]
        inputs = [self.order, self.particles, self.alpha, self.normal, self.margin]
        wp.launch(gather_capsule_contacts_kernel, n, inputs, out, device=self.device)
        return tuple(host_head(a, n) for a in out)
