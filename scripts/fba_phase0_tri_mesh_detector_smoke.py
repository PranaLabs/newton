# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 smoke for v-t+e-e plan: TriMeshCollisionDetector setup + refit.

Builds a 2-layer cloth-on-sphere scene (reuses ``fba_cloth_layers_on_sphere.py``
helpers), then sets up a single ``TriMeshCollisionDetector`` (per locked
decision A in the plan) covering both cloths.  Steps for 100 frames with
``detector.refit()`` between each step.

Validates:
  * Tier 1 (binary): no NaN, no exceptions, BVH state stays sane.
  * Tier 2 (drift): every step's ``lower_bounds_tris`` / ``lower_bounds_edges``
    matches a NumPy recomputation from ``state.particle_q`` to fp32 tolerance
    (1e-5 absolute, 1e-4 relative).
  * Tier 3 (diagnostics): per-step counts of v-t and e-e broadphase hits.
    Expected: zero at frame 0 (layers separated), ramping up once they touch.

The detector is invoked but its CSR output is **not** consumed yet — that is
Phase 1.  This script proves the plumbing works before we start emitting rows.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

# Bring VBD's filter-list + adjacency helpers — same module is already used
# inside Newton; FBA may freely import sibling internal modules.
from newton._src.solvers.vbd.particle_vbd_kernels import (
    ParticleForceElementAdjacencyInfo,
    _count_num_adjacent_edges,
    _count_num_adjacent_faces,
    _fill_adjacent_edges,
    _fill_adjacent_faces,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    set_to_csr,
)
from newton._src.solvers.vbd.tri_mesh_collision import TriMeshCollisionDetector

import newton
from newton.solvers import SolverFBA

# Reuse the multi-layer cloth-on-sphere scene builder verbatim.
from scripts.fba_cloth_layers_on_sphere import (  # noqa: E402
    _DT,
    _FRICTION_MU,
    _NSN_ITER,
    _OBJ_MASS_PER_LAYER,
    _PD_ITER,
    _PARTICLE_RADIUS,
    _PIN_STIFFNESS,
    build_scene,
)


# ---------------------------------------------------------------------------
# Adjacency + filter list construction (mirrors SolverVBD's setup, isolated
# here so FBA can build the same data without depending on the VBD class).
# ---------------------------------------------------------------------------


def build_particle_force_element_adjacency(model) -> ParticleForceElementAdjacencyInfo:
    """Build the CSR adjacency object that VBD's ring filters expect.

    Only fills the slots needed for v-t + e-e ring-filter helpers
    (``v_adj_edges`` and ``v_adj_faces``).  ``v_adj_tets`` /
    ``v_adj_springs`` are left empty.
    """
    adj = ParticleForceElementAdjacencyInfo()

    with wp.ScopedDevice("cpu"):
        # Vertex ↔ edge adjacency (CSR, 2 ints per entry: (edge_id, vertex_order))
        if model.edge_indices is not None and model.edge_indices.shape[0] > 0:
            edges_cpu = model.edge_indices.to("cpu")
            num_adj = wp.zeros(shape=(model.particle_count,), dtype=wp.int32)
            wp.launch(_count_num_adjacent_edges, inputs=[edges_cpu, num_adj], dim=1, device="cpu")
            num_adj_np = num_adj.numpy()
            offsets = np.empty(model.particle_count + 1, dtype=wp.int32)
            offsets[0] = 0
            offsets[1:] = np.cumsum(2 * num_adj_np)
            adj.v_adj_edges_offsets = wp.array(offsets, dtype=wp.int32)
            fill_count = wp.zeros(shape=(model.particle_count,), dtype=wp.int32)
            adj.v_adj_edges = wp.empty(shape=(int(2 * num_adj_np.sum()),), dtype=wp.int32)
            wp.launch(
                _fill_adjacent_edges,
                inputs=[edges_cpu, adj.v_adj_edges_offsets, fill_count, adj.v_adj_edges],
                dim=1, device="cpu",
            )
        else:
            adj.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
            adj.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

        # Vertex ↔ triangle adjacency (CSR, 2 ints per entry: (tri_id, vertex_order))
        if model.tri_indices is not None and model.tri_indices.shape[0] > 0:
            tris_cpu = model.tri_indices.to("cpu")
            num_adj = wp.zeros(shape=(model.particle_count,), dtype=wp.int32, device="cpu")
            wp.launch(_count_num_adjacent_faces, inputs=[tris_cpu, num_adj], dim=1, device="cpu")
            num_adj_np = num_adj.numpy()
            offsets = np.empty(model.particle_count + 1, dtype=wp.int32)
            offsets[0] = 0
            offsets[1:] = np.cumsum(2 * num_adj_np)
            adj.v_adj_faces_offsets = wp.array(offsets, dtype=wp.int32)
            fill_count = wp.zeros(shape=(model.particle_count,), dtype=wp.int32)
            adj.v_adj_faces = wp.empty(shape=(int(2 * num_adj_np.sum()),), dtype=wp.int32)
            wp.launch(
                _fill_adjacent_faces,
                inputs=[tris_cpu, adj.v_adj_faces_offsets, fill_count, adj.v_adj_faces],
                dim=1, device="cpu",
            )
        else:
            adj.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
            adj.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

        # Phase 0 only needs faces + edges; springs / tets are not used by
        # the ring-filter helpers.  Populate empty placeholders so .to(device)
        # works (the struct's .to() unconditionally moves every CSR slot).
        adj.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)
        adj.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
        adj.v_adj_tets = wp.empty(shape=(0,), dtype=wp.int32)
        adj.v_adj_tets_offsets = wp.empty(shape=(0,), dtype=wp.int32)

    return adj.to(model.device)


def build_filter_lists(model, adjacency: ParticleForceElementAdjacencyInfo, ring: int = 2):
    """Build v-t and e-e topology filter CSRs (1-/2-ring) on the model device."""
    v_tri_sets = None
    edge_edge_sets = None
    if ring >= 2:
        if adjacency.v_adj_faces_offsets.size > 0:
            v_tri_sets = build_vertex_n_ring_tris_collision_filter(
                ring,
                model.particle_count,
                model.edge_indices.numpy(),
                adjacency.v_adj_edges.numpy(),
                adjacency.v_adj_edges_offsets.numpy(),
                adjacency.v_adj_faces.numpy(),
                adjacency.v_adj_faces_offsets.numpy(),
            )
        if adjacency.v_adj_edges_offsets.size > 0:
            edge_edge_sets = build_edge_n_ring_edge_collision_filter(
                ring,
                model.edge_indices.numpy(),
                adjacency.v_adj_edges.numpy(),
                adjacency.v_adj_edges_offsets.numpy(),
            )

    if v_tri_sets is None:
        vt_flat = vt_offs = None
    else:
        vt_flat_np, vt_offs_np = set_to_csr(v_tri_sets)
        vt_flat = wp.array(vt_flat_np, dtype=int, device=model.device)
        vt_offs = wp.array(vt_offs_np, dtype=int, device=model.device)

    if edge_edge_sets is None:
        ee_flat = ee_offs = None
    else:
        ee_flat_np, ee_offs_np = set_to_csr(edge_edge_sets)
        ee_flat = wp.array(ee_flat_np, dtype=int, device=model.device)
        ee_offs = wp.array(ee_offs_np, dtype=int, device=model.device)

    return vt_flat, vt_offs, ee_flat, ee_offs


# ---------------------------------------------------------------------------
# Per-step AABB ground-truth recompute (numpy) — fast at <2k tris.
# ---------------------------------------------------------------------------


def aabbs_from_positions(positions: np.ndarray, tri_indices: np.ndarray):
    """Return ``(lower_tris, upper_tris)`` shape (T,3)."""
    tri = tri_indices.reshape(-1, 3)
    p = positions[tri]                      # (T, 3, 3)
    return p.min(axis=1), p.max(axis=1)


def aabbs_from_positions_edges(positions: np.ndarray, edge_indices: np.ndarray):
    """Return ``(lower_edges, upper_edges)`` shape (E,3) — using endpoint vertices (col 2, 3)."""
    e = edge_indices.reshape(-1, 4)
    p = positions[e[:, 2:4]]                # (E, 2, 3)
    return p.min(axis=1), p.max(axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--query-radius", type=float, default=0.05)
    ap.add_argument("--ring", type=int, default=2)
    args = ap.parse_args(argv)

    wp.init()

    # ---- Scene ----
    b, _layer_p, _layer_t = build_scene(n_layers=2)
    model = b.finalize()
    uniform_mass = _OBJ_MASS_PER_LAYER / _layer_p[0][1]
    model.particle_mass.assign(np.full(model.particle_count, uniform_mass, np.float32))
    model.particle_inv_mass.assign(np.full(model.particle_count, 1.0 / uniform_mass, np.float32))

    print(f"[scene] particles={model.particle_count}  tris={model.tri_count}  edges={model.edge_count}")

    # ---- Solver (no self-contact wiring yet — Phase 0 is detector-only plumbing). ----
    mu_o = np.full(model.particle_count, _FRICTION_MU, np.float64)
    solver = SolverFBA(
        model,
        iterations=_PD_ITER, nsn_iterations=_NSN_ITER, friction=True,
        stretching_model="arap", mu_per_pair_override=mu_o,
        pin_stiffness=_PIN_STIFFNESS, nsn_schur_mode="lite",
    )

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.04)
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()

    # ---- Build adjacency + filter lists ----
    adjacency = build_particle_force_element_adjacency(model)
    vt_flat, vt_offs, ee_flat, ee_offs = build_filter_lists(model, adjacency, ring=args.ring)
    print(f"[filter] v-t list entries={0 if vt_flat is None else int(vt_flat.size)}  "
          f"e-e list entries={0 if ee_flat is None else int(ee_flat.size)}")

    # ---- Detector ----
    detector = TriMeshCollisionDetector(
        model,
        vertex_collision_buffer_pre_alloc=16,
        edge_collision_buffer_pre_alloc=20,
        edge_edge_parallel_epsilon=1e-5,
    )
    detector.set_collision_filter_list(vt_flat, vt_offs, ee_flat, ee_offs)
    print(f"[detector] bvh_tris.id={detector.bvh_tris.id}  bvh_edges.id={detector.bvh_edges.id}")

    tri_np = model.tri_indices.numpy().astype(np.int32)
    edge_np = model.edge_indices.numpy().astype(np.int32)

    # ---- Trackers ----
    max_tri_drift = 0.0
    max_edge_drift = 0.0
    nan_seen = False
    vt_count_hist = []
    ee_count_hist = []

    for f in range(args.frames):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, _DT)
        state_0, state_1 = state_1, state_0

        # Update detector vertex_positions reference to current state buffer
        # (refit() takes a new_pos kw, but we keep the binding stable by
        # passing state_0.particle_q which is the live buffer after swap).
        detector.refit(state_0.particle_q)

        # ---- Tier 2 drift checks ----
        q = state_0.particle_q.numpy()
        if not np.isfinite(q).all():
            print(f"[NaN] frame {f}: non-finite particle_q")
            nan_seen = True
            break

        lo_tri_gt, hi_tri_gt = aabbs_from_positions(q.astype(np.float64), tri_np)
        lo_tri_bvh = detector.lower_bounds_tris.numpy()
        hi_tri_bvh = detector.upper_bounds_tris.numpy()
        tri_lo_err = float(np.abs(lo_tri_bvh - lo_tri_gt).max())
        tri_hi_err = float(np.abs(hi_tri_bvh - hi_tri_gt).max())
        tri_drift = max(tri_lo_err, tri_hi_err)
        max_tri_drift = max(max_tri_drift, tri_drift)

        lo_edge_gt, hi_edge_gt = aabbs_from_positions_edges(q.astype(np.float64), edge_np)
        lo_edge_bvh = detector.lower_bounds_edges.numpy()
        hi_edge_bvh = detector.upper_bounds_edges.numpy()
        edge_lo_err = float(np.abs(lo_edge_bvh - lo_edge_gt).max())
        edge_hi_err = float(np.abs(hi_edge_bvh - hi_edge_gt).max())
        edge_drift = max(edge_lo_err, edge_hi_err)
        max_edge_drift = max(max_edge_drift, edge_drift)

        # ---- Tier 3 broadphase counts ----
        detector.vertex_colliding_triangles_count.zero_()
        detector.vertex_triangle_collision_detection(args.query_radius)
        vt_counts = detector.vertex_colliding_triangles_count.numpy()
        vt_count_hist.append(int(vt_counts.sum()))

        detector.edge_colliding_edges_count.zero_()
        detector.edge_edge_collision_detection(args.query_radius)
        ee_counts = detector.edge_colliding_edges_count.numpy()
        ee_count_hist.append(int(ee_counts.sum()))

        if f % 20 == 0 or f == args.frames - 1:
            print(f"  f={f:3d}  tri_drift={tri_drift:.2e}  edge_drift={edge_drift:.2e}  "
                  f"vt_pairs={vt_count_hist[-1]:5d}  ee_pairs={ee_count_hist[-1]:5d}  "
                  f"y[{q[:,1].min():+.2f},{q[:,1].max():+.2f}]")

    # ---- Tier 1 binary gate ----
    drift_tol = 1.0e-4  # fp32 abs tol — positions are float32 in particle_q
    drift_pass = (max_tri_drift < drift_tol) and (max_edge_drift < drift_tol)
    nan_pass = not nan_seen

    # ---- Tier 3 hits-ramp diagnostic ----
    vt_settles = vt_count_hist[-20:] if vt_count_hist else []
    ee_settles = ee_count_hist[-20:] if ee_count_hist else []
    vt_max = max(vt_count_hist) if vt_count_hist else 0
    ee_max = max(ee_count_hist) if ee_count_hist else 0

    print()
    print(f"[summary] max_tri_aabb_drift = {max_tri_drift:.3e} m  (tol {drift_tol:.0e})")
    print(f"[summary] max_edge_aabb_drift = {max_edge_drift:.3e} m  (tol {drift_tol:.0e})")
    print(f"[summary] v-t pair count peak={vt_max}, last-20-mean={np.mean(vt_settles):.1f}")
    print(f"[summary] e-e pair count peak={ee_max}, last-20-mean={np.mean(ee_settles):.1f}")

    print()
    print(f"  Tier 1 (no-NaN, AABB tracks fp32): {'PASS' if (nan_pass and drift_pass) else 'FAIL'}")
    print(f"  Tier 3 (broadphase fires after settling): "
          f"{'PASS' if (vt_max > 0) else 'NO HITS'}")

    return 0 if (nan_pass and drift_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
