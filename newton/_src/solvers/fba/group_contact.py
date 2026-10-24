# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Per-object vertex-face clearances for disconnected soft meshes."""

from types import MethodType

import numpy as np
import warp as wp

from newton._src.geometry.kernels import triangle_closest_point
from newton._src.solvers.fba.self_contact import (
    _LANES,
    SelfContact,
    _in_ring,
    compact_contacts_kernel,
    triangle_candidates_kernel,
)


@wp.kernel
def pair_margin_vertex_face_kernel(
    q: wp.array[wp.vec3],
    tri: wp.array2d[wp.int32],
    candidates: wp.array2d[wp.int32],
    candidate_count: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring: wp.array[wp.int32],
    rest: wp.array[wp.vec3],
    rest_radius: float,
    query_radius: float,
    component: wp.array[wp.int32],
    same_margin: wp.array[wp.float32],
    different_margin: float,
    gap: float,
    count: wp.array[wp.int32],
    particles: wp.array2d[wp.int32],
    alpha: wp.array2d[wp.float32],
    normal: wp.array[wp.vec3],
    found: wp.array[wp.int32],
):
    v, lane = wp.tid()
    p = q[v]
    best = gap
    best_k = wp.int32(2147483647)
    best_t = wp.int32(-1)
    best_bary = wp.vec3(0.0)
    best_point = wp.vec3(0.0)
    best_distance = query_radius
    for k in range(lane, wp.min(candidate_count[v], candidates.shape[1]), wp.block_dim()):
        t = candidates[v, k]
        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        point, bary, _feature = triangle_closest_point(q[a], q[b], q[c], p)
        distance = wp.length(p - point)
        margin = different_margin
        if component[v] == component[a]:
            margin = same_margin[component[v]]
        clearance = distance - margin
        if clearance >= best:
            continue
        if (
            _in_ring(ring_offsets, ring, v, a)
            or _in_ring(ring_offsets, ring, v, b)
            or _in_ring(ring_offsets, ring, v, c)
        ):
            continue
        if rest_radius > 0.0:
            rp, _rb, _rf = triangle_closest_point(rest[a], rest[b], rest[c], rest[v])
            if wp.length(rest[v] - rp) < rest_radius:
                continue
        best, best_k, best_t = clearance, k, t
        best_bary, best_point, best_distance = bary, point, distance
    nearest = wp.tile_min(wp.tile(best))[0]
    first = wp.tile_min(wp.tile(wp.where(best == nearest, best_k, wp.int32(2147483647))))[0]
    nearest_distance = wp.tile_min(wp.tile(wp.where(best_k == first, best_distance, wp.float32(1.0e30))))[0]
    hit = first != wp.int32(2147483647) and nearest_distance >= 1.0e-6 * query_radius
    if lane == 0:
        found[v] = wp.where(hit, 1, 0)
    if not hit or best_k != first:
        return
    wp.atomic_add(count, 0, 1)
    particles[v, 0] = v
    alpha[v, 0] = 1.0
    for i in range(3):
        particles[v, i + 1] = tri[best_t, i]
        alpha[v, i + 1] = -best_bary[i]
    normal[v] = (p - best_point) / best_distance


@wp.kernel
def set_pair_margin_offsets(
    particle: wp.array[wp.int32],
    shape: wp.array[wp.int32],
    extra_particle: wp.array2d[wp.int32],
    component: wp.array[wp.int32],
    same_margin: wp.array[wp.float32],
    different_margin: float,
    normal: wp.array[wp.vec3],
    tangent1: wp.array[wp.vec3],
    tangent2: wp.array[wp.vec3],
    body_pos: wp.array[wp.vec3],
    world_anchor: wp.array[wp.vec3],
    normal_offset: wp.array[wp.float64],
    tangent1_offset: wp.array[wp.float64],
    tangent2_offset: wp.array[wp.float64],
):
    c = wp.tid()
    # Only VF soft-soft rows: rigid particle/face contacts retain their own skins.
    if shape[c] >= 0:
        return
    p, other = particle[c], extra_particle[c, 0]
    if p < 0 or other < 0:
        return
    margin = different_margin
    if component[p] == component[other]:
        margin = same_margin[component[p]]
    anchor = margin * normal[c]
    body_pos[c] = anchor
    world_anchor[c] = anchor
    # Match original fp32 dot -> fp64 conversion used by FBA's anchor kernels.
    normal_offset[c] = wp.float64(wp.dot(normal[c], anchor))
    tangent1_offset[c] = wp.float64(wp.dot(tangent1[c], anchor))
    tangent2_offset[c] = wp.float64(wp.dot(tangent2[c], anchor))


class PairMarginSelfContact(SelfContact):
    def __init__(self, model, component, same_margin, different_margin, gap, rings, rest_radius):
        query = max(float(different_margin), float(np.max(same_margin))) + float(gap)
        super().__init__(model, "vertex_face", query, rings, rest_radius)
        self.component = wp.array(component, dtype=wp.int32, device=self.device)
        self.same_margin = wp.array(same_margin, dtype=wp.float32, device=self.device)
        self.different_margin, self.gap = float(different_margin), float(gap)

    def find(self, q):
        self._bounds(q)
        self.bvh.refit()
        while True:
            self.count.zero_()
            wp.launch(
                triangle_candidates_kernel,
                q.shape[0],
                inputs=[q, self.query_particle_mask, self.bvh.id, self.query_radius],
                outputs=[self.candidates, self.candidate_count, self.count],
                device=self.device,
            )
            wp.launch_tiled(
                pair_margin_vertex_face_kernel,
                dim=[q.shape[0]],
                inputs=[
                    q,
                    self.tri_indices,
                    self.candidates,
                    self.candidate_count,
                    self.ring_offsets,
                    self.ring,
                    self.rest,
                    self.rest_radius,
                    self.query_radius,
                    self.component,
                    self.same_margin,
                    self.different_margin,
                    self.gap,
                ],
                outputs=[self.count, self.particles, self.alpha, self.normal, self.found],
                block_dim=_LANES,
                device=self.device,
            )
            wp.utils.array_scan(self.found, self.position, inclusive=False)
            wp.launch(
                compact_contacts_kernel,
                q.shape[0],
                inputs=[self.found, self.position, self.particles, self.alpha, self.normal],
                outputs=[self.found_particles, self.found_alpha, self.found_normal],
                device=self.device,
            )
            n, most = (int(x) for x in self.count.numpy())
            if most > self.candidates.shape[1]:
                self.candidates = wp.empty((q.shape[0], 2 * most), dtype=wp.int32, device=self.device)
            else:
                return n


def configure_pair_margins(solver, particle_components, same_component_margins, different_component_margin):
    """Configure only this solver, without changing physical state.

    Triangles must belong to a single component. Original self-contact mode,
    topology rings and rest exclusion remain active. Call before the first step.
    """
    old = solver._self_contact
    if old is None or old.mode != "vertex_face":
        raise ValueError("Pair margins require the enabled vertex-face self-contact path")
    if not solver.friction:
        raise ValueError("This scene helper expects Coulomb-friction contacts")
    raw = np.asarray(particle_components)
    if not np.issubdtype(raw.dtype, np.integer) or raw.size == 0:
        raise ValueError("Particle group IDs must be a nonempty integer array")
    component = np.asarray(raw, dtype=np.int32)
    margins = np.asarray(same_component_margins, dtype=np.float32)
    if component.shape != (solver.model.particle_count,) or np.any(component < 0):
        raise ValueError("Each physical particle needs a valid component ID")
    if (
        margins.ndim != 1
        or len(margins) <= int(component.max())
        or not np.isfinite(margins).all()
        or np.any(margins <= 0)
    ):
        raise ValueError("Every component must retain a positive finite self margin")
    if not np.isfinite(different_component_margin) or different_component_margin <= 0:
        raise ValueError("Inter-component margin must remain positive and finite")
    triangles = solver.model.tri_indices.numpy()
    if np.any(component[triangles] != component[triangles[:, :1]]):
        raise ValueError("A boundary triangle must belong to a single soft object")
    # Use the model's original rest configuration, never a settled pose.
    gap = float(old.query_radius - solver.particle_self_contact_margin)
    sc = PairMarginSelfContact(solver.model, component, margins, different_component_margin, gap, 0, old.rest_radius)
    sc.ring_offsets, sc.ring, sc.rest = old.ring_offsets, old.ring, old.rest
    original_ingest = solver._ingest_contacts

    def ingest_with_pair_offsets(self, contacts, state=None):
        original_ingest(contacts, state)
        if self._contact_count == 0 or self._contact_extra is None:
            return
        wp.launch(
            set_pair_margin_offsets,
            self._contact_count,
            inputs=[
                self._contact_particle_d,
                self._contact_shape_d,
                self._contact_extra[0],
                sc.component,
                sc.same_margin,
                sc.different_margin,
                self._contact_normal_d,
                self._contact_tangent1_d,
                self._contact_tangent2_d,
            ],
            outputs=[
                self._contact_body_pos_d,
                self._contact_world_anchor_d,
                self._contact_offset_d,
                self._contact_tangent1_offset_d,
                self._contact_tangent2_offset_d,
            ],
            device=self._device,
        )

    solver._self_contact = sc
    solver._ingest_contacts = MethodType(ingest_with_pair_offsets, solver)
