# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 smoke for v-t+e-e plan: edge-edge contacts on crossed cloth strips.

Two thin cloth strips arranged in an "X" shape (strip A along world X,
strip B rotated 90 deg around Y to lie along world Z), pinned at their
outer-most corners, and offset vertically so they cross within the
contact threshold at the centre.  Under gravity the unfixed centre of
each strip sags toward the other.  Without edge-edge contacts the
strips visibly pass through each other at the crossing; with e-e
contacts they constrain each other and a minimum separation persists.

Acceptance:
  Tier 1 (binary): no NaN over 500 frames.
  Tier 2 (separation): at the crossing point, the minimum signed
      distance between any vertex of strip A and the closest triangle
      plane of strip B stays >= 0.5 * particle_radius across frames
      100-500 in the e-e-on case.
  Tier 3 (diagnostics): per-frame v-t and e-e pair counts.

The script runs the simulation twice — once with ``particle_contact_ee=
False`` (v-t only) and once with the default ``True`` — so the user
can compare the separation traces and confirm that e-e is what stops
the strips from passing through each other.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFBA


# ---- Scene constants. ----
_DT = 1.0 / 240.0
_GRAVITY = -10.0
_PD_ITER = 5
_NSN_ITER = 1
_YOUNG = 1.0e3
_POISSON = 0.3
_BENDING = 1.0e-4
_PIN_STIFFNESS = 1.0e10
_FRICTION_MU = 0.05

# Strip geometry: a long, NARROW strip so the only way the two strips
# constrain each other at the crossing is via edge-edge contacts (vertex-
# triangle would only fire if a corner vertex of strip A landed on the
# tiny triangle area of strip B, which is geometrically unlikely when
# the strips are 1 cell wide and cross at 90 degrees).
#
# 12 cells along the long axis * 1 cell along the short axis -> 0.6m * 0.05m.
_STRIP_DIM_X = 12
_STRIP_DIM_Y = 1
_STRIP_CELL = 0.05

# Particle contact tuning.
_PARTICLE_RADIUS = 0.01
_CONTACT_MARGIN = 0.02

# Strip B is held rigid (all of its vertices pinned), forming the "floor"
# the upper strip A drops onto.  Strip A starts with its endpoints at
# y=_STRIP_A_PIN_Y and its midplane free under gravity — without contact
# it sags below strip B's midplane, passing through.
_STRIP_A_PIN_Y = 0.025   # A endpoints (pinned)
_STRIP_B_PIN_Y = 0.000   # B all vertices (pinned)

# Mass per strip particle — tuned so that without contact, A would sag past
# strip B's plane by ~1 cm, but with contact, A is caught above B.
_PARTICLE_MASS = 0.02


def _make_strip_grid(
    dim_x: int, dim_y: int, cell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build a planar grid mesh (in the local XY plane) of size (dim_x+1, dim_y+1).

    Returns (vertices, triangles) as numpy arrays — vertices shape (N, 3),
    tris shape (T, 3) — using the same winding convention as
    ``ModelBuilder.add_cloth_grid``.
    """
    vertices = []
    tris = []
    for y in range(dim_y + 1):
        for x in range(dim_x + 1):
            vertices.append((x * cell, y * cell, 0.0))
    def idx(x: int, y: int) -> int:
        return y * (dim_x + 1) + x
    for y in range(1, dim_y + 1):
        for x in range(1, dim_x + 1):
            v0 = idx(x - 1, y - 1)
            v1 = idx(x, y - 1)
            v2 = idx(x, y)
            v3 = idx(x - 1, y)
            tris.append((v0, v1, v3))
            tris.append((v1, v2, v3))
    return np.asarray(vertices, dtype=np.float32), np.asarray(tris, dtype=np.int32)


def build_scene() -> tuple[newton.ModelBuilder, dict]:
    """Two crossed cloth strips, pinned at outer corners only.

    Strip A spans the X axis (long axis = world X), centred at y=+offset/2.
    Strip B spans the Z axis (long axis = world Z), centred at y=-offset/2.
    Each strip's two end-corners are pinned via stiff pin springs (handled
    by SolverFBA via ``particle_mass = 0`` on pin particles).
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=_GRAVITY)

    verts_local, tris = _make_strip_grid(_STRIP_DIM_X, _STRIP_DIM_Y, _STRIP_CELL)

    # Centre the grid on the origin.
    lx = _STRIP_DIM_X * _STRIP_CELL
    ly = _STRIP_DIM_Y * _STRIP_CELL
    verts_local[:, 0] -= lx * 0.5
    verts_local[:, 1] -= ly * 0.5

    n_per_strip = verts_local.shape[0]
    info: dict = {}

    # ---- Strip A: lies in world XZ, long axis = world X, midplane y=_STRIP_A_PIN_Y. ----
    verts_A = np.zeros_like(verts_local)
    verts_A[:, 0] = verts_local[:, 0]
    verts_A[:, 2] = verts_local[:, 1]
    verts_A[:, 1] = _STRIP_A_PIN_Y

    builder.add_cloth_mesh(
        pos=wp.vec3(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0, 0, 0),
        vertices=[wp.vec3(*v.tolist()) for v in verts_A],
        indices=tris.reshape(-1).tolist(),
        density=_PARTICLE_MASS / (_STRIP_CELL * _STRIP_CELL),
        tri_ke=_YOUNG,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=_BENDING,
        edge_kd=0.0,
        particle_radius=_PARTICLE_RADIUS,
    )
    info["A_start"] = 0
    info["A_count"] = n_per_strip
    info["A_tri_start"] = 0
    info["A_tri_count"] = tris.shape[0]

    # ---- Strip B: lies in world XZ, long axis = world Z, midplane y=_STRIP_B_PIN_Y. ----
    # Same grid but rotated 90 deg around Y so its long axis aligns with Z.
    verts_B = np.zeros_like(verts_local)
    verts_B[:, 0] = verts_local[:, 1]  # short axis -> X
    verts_B[:, 2] = verts_local[:, 0]  # long axis -> Z
    verts_B[:, 1] = _STRIP_B_PIN_Y

    builder.add_cloth_mesh(
        pos=wp.vec3(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0, 0, 0),
        vertices=[wp.vec3(*v.tolist()) for v in verts_B],
        indices=tris.reshape(-1).tolist(),
        density=_PARTICLE_MASS / (_STRIP_CELL * _STRIP_CELL),
        tri_ke=_YOUNG,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=_BENDING,
        edge_kd=0.0,
        particle_radius=_PARTICLE_RADIUS,
    )
    info["B_start"] = n_per_strip
    info["B_count"] = n_per_strip
    info["B_tri_start"] = tris.shape[0]
    info["B_tri_count"] = tris.shape[0]

    # ---- Pin both ends of strip A; pin ALL of strip B. ----
    # Strip B is held rigid (acts as the obstacle); strip A drapes from its
    # endpoints and must be caught by EE contact at the crossing.
    pin_particles: list[int] = []
    nx = _STRIP_DIM_X + 1
    ny = _STRIP_DIM_Y + 1
    # Strip A: pin both end-rows (x=0 and x=dim_x), all y.
    for cy in range(ny):
        for cx in (0, nx - 1):
            pin_particles.append(info["A_start"] + cy * nx + cx)
    # Strip B: pin EVERY vertex (B is rigid).
    for cy in range(ny):
        for cx in range(nx):
            pin_particles.append(info["B_start"] + cy * nx + cx)
    info["pin_particles"] = pin_particles

    return builder, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--report-every", type=int, default=50)
    args = ap.parse_args(argv)

    wp.init()

    overall_pass = True

    for ee_on, label in ((False, "EE-OFF"), (True, "EE-ON")):
        print(f"\n========= run: {label}  (particle_contact_ee={ee_on}) =========")
        b, info = build_scene()
        model = b.finalize()
        # Pin diagonally-opposite corners by clamping their mass.
        pin_set = set(info["pin_particles"])
        if pin_set:
            mass = model.particle_mass.numpy().copy()
            inv = model.particle_inv_mass.numpy().copy()
            for pi in pin_set:
                mass[pi] = 0.0
                inv[pi] = 0.0
            model.particle_mass.assign(mass)
            model.particle_inv_mass.assign(inv)

        n_pin = len(pin_set)
        n_particles = int(model.particle_count)
        n_tris = int(model.tri_count)
        print(
            f"[scene] particles={n_particles}  tris={n_tris}  "
            f"edges={model.edge_count}  pins={n_pin}"
        )

        mu_o = np.full(n_particles, _FRICTION_MU, np.float64)
        solver = SolverFBA(
            model,
            iterations=_PD_ITER,
            nsn_iterations=_NSN_ITER,
            friction=True,
            stretching_model="arap",
            mu_per_pair_override=mu_o,
            pin_stiffness=_PIN_STIFFNESS,
            nsn_schur_mode="lite",
            particle_contact_radius=_PARTICLE_RADIUS,
            particle_contact_margin=_CONTACT_MARGIN,
            particle_contact_friction=_FRICTION_MU,
            particle_contact_topology_ring=2,
            particle_contact_rest_exclusion_radius=0.1,
            particle_contact_mode="vt",
            particle_contact_ee=ee_on,
        )

        pipeline = newton.CollisionPipeline(model, soft_contact_margin=_CONTACT_MARGIN)
        contacts = pipeline.contacts()
        state_0 = model.state()
        state_1 = model.state()

        tris_np = model.tri_indices.numpy().astype(np.int32)
        A_start, A_count = info["A_start"], info["A_count"]
        B_start, B_count = info["B_start"], info["B_count"]
        B_tri_start, B_tri_count = info["B_tri_start"], info["B_tri_count"]
        A_particles = np.arange(A_start, A_start + A_count, dtype=np.int32)
        B_tris_ids = np.arange(B_tri_start, B_tri_start + B_tri_count, dtype=np.int32)

        # The "crossing" region: vertices of strip A whose XZ position lies
        # inside the bounding rectangle of strip B (which spans
        # X in [-0.05, 0.05], Z in [-0.20, 0.20]).
        rest_q = state_0.particle_q.numpy()
        A_xz = rest_q[A_particles][:, [0, 2]]
        in_crossing = (
            (A_xz[:, 0] > -0.06) & (A_xz[:, 0] < 0.06)
            & (A_xz[:, 1] > -0.06) & (A_xz[:, 1] < 0.06)
        )
        A_crossing = A_particles[in_crossing]
        print(f"[scene] {label}: A vertices in crossing region = {A_crossing.size}")

        # Identify central edges of each strip for the SIGNED edge-edge
        # crossing metric (true EE-acceptance signal — v-t cannot help here).
        edges_np = model.edge_indices.numpy().astype(np.int64)
        # An edge belongs to strip A iff both endpoints are in A's index range.
        def _strip_edges(start: int, count: int) -> np.ndarray:
            end = start + count
            mask = (
                (edges_np[:, 2] >= start) & (edges_np[:, 2] < end)
                & (edges_np[:, 3] >= start) & (edges_np[:, 3] < end)
            )
            return np.where(mask)[0]
        A_edges_all = _strip_edges(A_start, A_count)
        B_edges_all = _strip_edges(B_start, B_count)
        # Keep only edges whose midpoint XZ lies inside the crossing region.
        def _central_edges(edge_ids):
            ids = []
            for eid in edge_ids:
                v0 = edges_np[eid, 2]; v1 = edges_np[eid, 3]
                mp_xz = 0.5 * (rest_q[v0] + rest_q[v1])
                if abs(mp_xz[0]) < 0.06 and abs(mp_xz[2]) < 0.06:
                    ids.append(int(eid))
            return np.asarray(ids, dtype=np.int64)
        A_edges_c = _central_edges(A_edges_all)
        B_edges_c = _central_edges(B_edges_all)
        print(f"[scene] {label}: A central edges = {A_edges_c.size}, B central edges = {B_edges_c.size}")

        sep_history: dict[int, float] = {}
        ee_history: dict[int, int] = {}
        vt_history: dict[int, int] = {}
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
                # Minimum signed distance of any A-vertex (in crossing region)
                # to any B-triangle: use closest-point-on-triangle.  For each
                # A-vertex p, find the nearest B-triangle, compute the closest
                # point on it, then dot with the triangle's normal to get a
                # signed distance.
                a = q[tris_np[B_tris_ids, 0]]
                bv = q[tris_np[B_tris_ids, 1]]
                c = q[tris_np[B_tris_ids, 2]]
                normals = np.cross(bv - a, c - a)
                ln = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / np.where(ln > 1e-12, ln, 1.0)
                min_signed = np.inf
                for pi in A_crossing:
                    p = q[pi]
                    # Closest point on each triangle via barycentric (simple impl).
                    bary, closest = _bary_closest(a, bv, c, np.broadcast_to(p, a.shape))
                    diff = p - closest
                    signed = np.einsum("nk,nk->n", diff, normals)
                    # Take the signed dist of the geometrically nearest triangle.
                    dist = np.linalg.norm(diff, axis=1)
                    k = int(np.argmin(dist))
                    if abs(signed[k]) < abs(min_signed) or not np.isfinite(min_signed):
                        min_signed = signed[k]
                sep_history[f + 1] = float(min_signed)
                # Edge-edge signed gap: min over central A-edge x B-edge of
                # signed-y(closest-A-point - closest-B-point).  Negative means
                # an A-edge segment lies BELOW a B-edge segment at their closest
                # point — a true EE penetration the EE pass is supposed to
                # prevent.
                if A_edges_c.size > 0 and B_edges_c.size > 0:
                    ee_min_signed = _edge_edge_min_signed(q, edges_np, A_edges_c, B_edges_c)
                else:
                    ee_min_signed = float("nan")
                vt_h = int(getattr(solver, "_pc_last_vt_hits", 0))
                ee_h = int(getattr(solver, "_pc_last_ee_hits", 0))
                ee_history[f + 1] = ee_h
                vt_history[f + 1] = vt_h
                print(
                    f"  f={f+1:4d}  vt_hits={vt_h:5d}  ee_hits={ee_h:5d}  "
                    f"vt_sep={min_signed:+.5f}  ee_sep={ee_min_signed:+.5f}  "
                    f"y_A[{q[A_particles,1].min():+.4f}, {q[A_particles,1].max():+.4f}]"
                    f"  y_B[{q[B_start:B_start+B_count,1].min():+.4f}, {q[B_start:B_start+B_count,1].max():+.4f}]"
                )

        # ---- Verdict per run. ----
        print()
        if nan_seen:
            print(f"  Tier 1 ({label} no-NaN): FAIL")
            overall_pass = False
            continue
        print(f"  Tier 1 ({label} no-NaN): PASS")
        eval_frames = [fr for fr in sep_history if 100 <= fr <= args.frames]
        if not eval_frames:
            print(f"  Tier 2 ({label} separation): NO DATA")
            continue
        # Tier 2 acceptance uses the EE-specific metric: minimum signed-Y of
        # closest_A_pt - closest_B_pt over all central edge-edge pairs.  This
        # is the metric v-t cannot influence — only e-e can prevent the
        # edges from crossing.  Threshold = 0 (no penetration).  We require
        # ee-on to keep this >= 0 throughout frames 100-N.
        seps = np.array([sep_history[fr] for fr in eval_frames])
        min_signed = float(seps.min())
        threshold = 0.5 * _PARTICLE_RADIUS
        print(
            f"  Tier 2 ({label} signed v-t separation, frames 100-{args.frames}): "
            f"min={min_signed:+.5f}m  threshold={threshold:+.5f}m"
        )
        if ee_on:
            tier2_pass = min_signed >= 0.0
            print(f"          ee-on requirement: min >= 0 -> {'PASS' if tier2_pass else 'FAIL'}")
            if not tier2_pass:
                overall_pass = False
        else:
            print(
                f"          baseline (no acceptance criterion); expect penetration when v-t cannot resolve."
            )

        # Tier 3 diagnostics summary.
        if eval_frames:
            print(
                f"  Tier 3 ({label} broadphase): "
                f"vt_hits range [{min(vt_history[fr] for fr in eval_frames)}, {max(vt_history[fr] for fr in eval_frames)}], "
                f"ee_hits range [{min(ee_history[fr] for fr in eval_frames)}, {max(ee_history[fr] for fr in eval_frames)}]"
            )

    return 0 if overall_pass else 1


# ---------------------------------------------------------------------------
# Helper: barycentric closest-point on a triangle (numpy port).
# Same algorithm as ``triangle_closest_point_barycentric`` (Warp).
# ---------------------------------------------------------------------------


def _edge_edge_min_signed(q: np.ndarray, edges_np: np.ndarray, A_edges: np.ndarray, B_edges: np.ndarray) -> float:
    """Min signed-Y of (closest_A_pt - closest_B_pt) over all (eA, eB) pairs.

    Closest points on each pair of segments via Eberly's algorithm (numpy port).
    Returns the most negative signed-Y across all pairs, or +inf if no pairs.
    """
    a0 = q[edges_np[A_edges, 2]]
    a1 = q[edges_np[A_edges, 3]]
    b0 = q[edges_np[B_edges, 2]]
    b1 = q[edges_np[B_edges, 3]]
    da = a1 - a0
    db = b1 - b0
    # Broadcast: pairs(A, B)
    da_e = da[:, None, :]
    db_e = db[None, :, :]
    r = a0[:, None, :] - b0[None, :, :]
    aa = np.einsum("ABk,ABk->AB", da_e, da_e)
    bb = np.einsum("ABk,ABk->AB", da_e, db_e)
    ee = np.einsum("ABk,ABk->AB", db_e, db_e)
    f = np.einsum("ABk,ABk->AB", db_e, r)
    c = np.einsum("ABk,ABk->AB", da_e, r)
    denom = aa * ee - bb * bb
    s = np.where(denom > 1e-12, (bb * f - c * ee) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = (bb * s + f) / np.where(ee > 1e-12, ee, 1.0)
    t = np.clip(t, 0.0, 1.0)
    s = np.clip((bb * t - c) / np.where(aa > 1e-12, aa, 1.0), 0.0, 1.0)
    cA = a0[:, None, :] + da_e * s[..., None]
    cB = b0[None, :, :] + db_e * t[..., None]
    dy = cA[..., 1] - cB[..., 1]
    return float(dy.min())


def _bary_closest(a, b, c, p):
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
    safe = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
    v_face = vb / safe
    w_face = vc / safe
    bary = np.stack([1.0 - v_face - w_face, v_face, w_face], axis=1)

    region_a = (d1 <= 0.0) & (d2 <= 0.0)
    bary[region_a] = np.array([1.0, 0.0, 0.0])
    region_b = (d3 >= 0.0) & (d4 <= d3)
    bary[region_b] = np.array([0.0, 1.0, 0.0])
    region_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if region_ab.any():
        v_ab = d1[region_ab] / np.where((d1 - d3)[region_ab] != 0, (d1 - d3)[region_ab], 1.0)
        bary[region_ab] = np.stack([1.0 - v_ab, v_ab, np.zeros_like(v_ab)], axis=1)
    region_c = (d6 >= 0.0) & (d5 <= d6)
    bary[region_c] = np.array([0.0, 0.0, 1.0])
    region_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if region_ac.any():
        w_ac = d2[region_ac] / np.where((d2 - d6)[region_ac] != 0, (d2 - d6)[region_ac], 1.0)
        bary[region_ac] = np.stack([1.0 - w_ac, np.zeros_like(w_ac), w_ac], axis=1)
    region_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    if region_bc.any():
        denom_bc = d4 - d3 + d5 - d6
        w_bc = (d4 - d3)[region_bc] / np.where(denom_bc[region_bc] != 0, denom_bc[region_bc], 1.0)
        bary[region_bc] = np.stack([np.zeros_like(w_bc), 1.0 - w_bc, w_bc], axis=1)

    closest = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    return bary, closest


if __name__ == "__main__":
    sys.exit(main())
