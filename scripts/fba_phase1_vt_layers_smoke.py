# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 smoke for v-t+e-e plan: TriMeshCollisionDetector + 4-slot Jacobian.

Two-layer cloth stacking on the sphere with v-t self-contact enabled.  Runs
600 frames; every 50 frames, computes:
  * NaN check on particle positions.
  * Min/max y of the particles.
  * Number of penetration events (layer-A vertices inside layer-B triangles
    with signed distance < 0).

Acceptance:
  Tier 1 (binary): no NaN, finite throughout.
  Tier 2 (per-frame penetration count): ≤ 5 over frames 200–600.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA

from scripts.fba_cloth_layers_on_sphere import (  # noqa: E402
    _DT,
    _FRICTION_MU,
    _NSN_ITER,
    _OBJ_MASS_PER_LAYER,
    _PARTICLE_RADIUS,
    _PD_ITER,
    _PIN_STIFFNESS,
    build_scene,
)


def _closest_point_on_triangle_bary(a, b, c, p):
    """NumPy port of triangle_closest_point_barycentric (per-row vectorised).

    a/b/c/p shape (N, 3). Returns (bary (N, 3), closest (N, 3)).
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("nk,nk->n", ab, ap)
    d2 = np.einsum("nk,nk->n", ac, ap)

    bp = p - b
    d3 = np.einsum("nk,nk->n", ab, bp)
    d4 = np.einsum("nk,nk->n", ac, bp)

    cp = p - c
    d5 = np.einsum("nk,nk->n", ab, cp)
    d6 = np.einsum("nk,nk->n", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    # Default: interior face point.
    safe_denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
    v_face = vb / safe_denom
    w_face = vc / safe_denom
    bary = np.stack([1.0 - v_face - w_face, v_face, w_face], axis=1)

    # Vertex A: d1 <= 0 and d2 <= 0
    region_a = (d1 <= 0.0) & (d2 <= 0.0)
    bary[region_a] = np.array([1.0, 0.0, 0.0])
    # Vertex B: d3 >= 0 and d4 <= d3
    region_b = (d3 >= 0.0) & (d4 <= d3)
    bary[region_b] = np.array([0.0, 1.0, 0.0])
    # Edge AB: vc <= 0, d1 >= 0, d3 <= 0
    region_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v_ab = d1[region_ab] / np.where((d1 - d3)[region_ab] != 0, (d1 - d3)[region_ab], 1.0)
    bary[region_ab] = np.stack([1.0 - v_ab, v_ab, np.zeros_like(v_ab)], axis=1)
    # Vertex C: d6 >= 0, d5 <= d6
    region_c = (d6 >= 0.0) & (d5 <= d6)
    bary[region_c] = np.array([0.0, 0.0, 1.0])
    # Edge AC: vb <= 0, d2 >= 0, d6 <= 0
    region_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w_ac = d2[region_ac] / np.where((d2 - d6)[region_ac] != 0, (d2 - d6)[region_ac], 1.0)
    bary[region_ac] = np.stack([1.0 - w_ac, np.zeros_like(w_ac), w_ac], axis=1)
    # Edge BC: va <= 0, (d4 - d3) >= 0, (d5 - d6) >= 0
    region_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    denom_bc = (d4 - d3 + d5 - d6)
    w_bc = (d4 - d3)[region_bc] / np.where(denom_bc[region_bc] != 0, denom_bc[region_bc], 1.0)
    bary[region_bc] = np.stack([np.zeros_like(w_bc), 1.0 - w_bc, w_bc], axis=1)

    closest = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    return bary, closest


def count_penetrations(
    positions: np.ndarray,
    tris: np.ndarray,
    layer_A_particles: np.ndarray,
    layer_B_tris: np.ndarray,
    threshold: float,
) -> int:
    """Counts (vertex_in_layer_A, tri_in_layer_B) penetration events.

    Penetration is signed: we compute closest-point on the triangle, signed
    distance = sign(dot(p - closest, tri_normal)) * ||p - closest||. Negative
    signed distance with magnitude > threshold and < some sane upper bound
    indicates the vertex has passed through the triangle plane.
    """
    if layer_A_particles.size == 0 or layer_B_tris.size == 0:
        return 0
    # Triangle normals.
    a = positions[tris[layer_B_tris, 0]]
    b = positions[tris[layer_B_tris, 1]]
    c = positions[tris[layer_B_tris, 2]]
    n_tri = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n_tri, axis=1, keepdims=True)
    n_tri = n_tri / np.where(n_norm > 1e-12, n_norm, 1.0)
    # Triangle bbox for filtering.
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)

    count = 0
    p_all = positions[layer_A_particles]
    # Brute force, but cheap at < 2k particles * < 5k tris × tri-bbox cull.
    # Use bbox filter per-particle to reduce work.
    for vi, p in enumerate(p_all):
        # Inflated bbox by threshold so we don't miss "just below the plane" cases.
        mask = np.all(p[None] >= lo - threshold, axis=1) & np.all(
            p[None] <= hi + threshold, axis=1
        )
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        a_sub = a[idx]
        b_sub = b[idx]
        c_sub = c[idx]
        n_sub = n_tri[idx]
        _bary, closest = _closest_point_on_triangle_bary(
            a_sub, b_sub, c_sub, np.broadcast_to(p, a_sub.shape)
        )
        diff = p[None] - closest
        dist = np.linalg.norm(diff, axis=1)
        signed = np.einsum("nk,nk->n", diff, n_sub)
        # Penetration: vertex on the negative-normal side AND within range.
        # A vertex of layer A that lies "below" a layer-B triangle plane is
        # penetrating that triangle plane.  Combine with closest-point dist
        # < threshold (i.e. genuinely close, not on the other side of the
        # mesh).
        pen_mask = (signed < -1.0e-4) & (dist < threshold)
        count += int(pen_mask.sum())
    return count


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--n-layers", type=int, default=2)
    args = ap.parse_args(argv)

    wp.init()

    b, layer_p, layer_t = build_scene(n_layers=args.n_layers)
    model = b.finalize()
    print(f"[scene] layers={args.n_layers}  particles={model.particle_count}  tris={model.tri_count}")

    uniform_mass = _OBJ_MASS_PER_LAYER / layer_p[0][1]
    model.particle_mass.assign(np.full(model.particle_count, uniform_mass, np.float32))
    model.particle_inv_mass.assign(np.full(model.particle_count, 1.0 / uniform_mass, np.float32))

    mu_o = np.full(model.particle_count, _FRICTION_MU, np.float64)
    solver = SolverFBA(
        model,
        iterations=_PD_ITER, nsn_iterations=_NSN_ITER, friction=True,
        stretching_model="arap", mu_per_pair_override=mu_o,
        pin_stiffness=_PIN_STIFFNESS, nsn_schur_mode="lite",
        particle_contact_radius=_PARTICLE_RADIUS * 2.0,
        particle_contact_margin=0.04,
        particle_contact_friction=0.25,
        particle_contact_topology_ring=2,
        particle_contact_rest_exclusion_radius=0.1,
        particle_contact_mode="vt",
        # This smoke is the v-t-only acceptance test; e-e (Phase 3) has its
        # own dedicated smoke (``fba_phase3_ee_xcross_smoke``).  Disable
        # here so the regression matches the original Phase 1 acceptance
        # criteria.
        particle_contact_ee=False,
    )

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.04)
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()

    tris_np = model.tri_indices.numpy().astype(np.int32)

    # Layer index ranges.
    pA_start, pA_count = layer_p[0]
    pB_start, pB_count = layer_p[1] if args.n_layers >= 2 else (0, 0)
    tA_start, tA_count = layer_t[0]
    tB_start, tB_count = layer_t[1] if args.n_layers >= 2 else (0, 0)
    particles_A = np.arange(pA_start, pA_start + pA_count, dtype=np.int32)
    particles_B = np.arange(pB_start, pB_start + pB_count, dtype=np.int32)
    tris_A_ids = np.arange(tA_start, tA_start + tA_count, dtype=np.int32)
    tris_B_ids = np.arange(tB_start, tB_start + tB_count, dtype=np.int32)

    pen_thresh = _PARTICLE_RADIUS * 2.0 + 0.04  # match contact threshold
    pen_history = {}
    nan_seen = False

    for f in range(args.frames):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, _DT)
        state_0, state_1 = state_1, state_0

        if (f + 1) % args.report_every == 0 or f == 0:
            q = state_0.particle_q.numpy()
            if not np.isfinite(q).all():
                print(f"  [NaN] frame {f+1}")
                nan_seen = True
                break
            pen_AB = count_penetrations(q.astype(np.float64), tris_np, particles_A, tris_B_ids, pen_thresh)
            pen_BA = count_penetrations(q.astype(np.float64), tris_np, particles_B, tris_A_ids, pen_thresh)
            pen_total = pen_AB + pen_BA
            pen_history[f + 1] = pen_total
            # Inspect the last contact count emitted by the detector path.
            if solver._pc_contact_count_d is not None:
                last_contacts = int(solver._pc_contact_count_d.numpy()[0])
            else:
                last_contacts = -1
            print(
                f"  f={f+1:4d}  contacts={last_contacts:5d}  pen_AB={pen_AB:4d}  pen_BA={pen_BA:4d}  "
                f"total={pen_total:4d}  y[{q[:,1].min():+.3f},{q[:,1].max():+.3f}]"
            )

    # ---- Verdict ----
    print()
    if nan_seen:
        print("  Tier 1 (no-NaN): FAIL")
        return 1
    print("  Tier 1 (no-NaN): PASS")

    eval_frames = [fid for fid in pen_history if 200 <= fid <= args.frames]
    if not eval_frames:
        print("  Tier 2 (penetration count): NO DATA")
        return 1
    max_pen = max(pen_history[f] for f in eval_frames)
    print(f"  Tier 2 (penetration count, frames 200-{args.frames}): peak={max_pen}  threshold=5")
    print(f"          {'PASS' if max_pen <= 5 else 'FAIL'}")
    return 0 if max_pen <= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
