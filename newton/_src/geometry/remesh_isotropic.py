# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isotropic remeshing of triangle surfaces after Botsch and Kobbelt, "A Remeshing Approach to Multiresolution
Modeling" (2004)."""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _closest_point_kernel(mesh: wp.uint64, points: wp.array[wp.vec3d], out: wp.array[wp.vec3d]):
    i = wp.tid()
    p = wp.vec3(points[i])
    q = wp.mesh_query_point_no_sign(mesh, p, 1.0e10)
    if q.result:
        out[i] = wp.vec3d(wp.mesh_eval_position(mesh, q.face, q.u, q.v))
    else:
        out[i] = points[i]


# Boundary corners turn by more than 30 degrees: 2 sin(15 degrees).
_CORNER = 0.52


def _topology(f: np.ndarray):
    """Half-edges ``(u, w)`` of the faces, in face order, the unique edges, the edge of each half-edge, and the
    number of faces at each edge."""
    half = np.stack([f, np.roll(f, -1, axis=1)], axis=2).reshape(-1, 2)
    edges, edge_of, count = np.unique(np.sort(half, axis=1), axis=0, return_inverse=True, return_counts=True)
    return half, edges, edge_of.ravel(), count


def _split(v: np.ndarray, f: np.ndarray, high: float):
    """Split edges longer than ``high`` at their midpoints, each face's longest one per round."""
    while True:
        _, edges, edge_of, count = _topology(f)
        length = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        if length.max() <= high:
            return v, f
        h_len = np.where(length[edge_of] > high, length[edge_of], -1.0).reshape(-1, 3)
        h = 3 * np.arange(len(f)) + h_len.argmax(axis=1)
        h = h[h_len.max(axis=1) > 0.0]
        split = np.flatnonzero(np.bincount(edge_of[h], minlength=len(edges)) == count)
        if not len(split):
            split = np.array([length.argmax()])
        h = np.flatnonzero(np.isin(edge_of, split))
        k, fid = h % 3, h // 3
        m = len(v) + np.searchsorted(split, edge_of[h])
        v = np.vstack([v, 0.5 * (v[edges[split, 0]] + v[edges[split, 1]])])
        u, w, x = f[fid, k], f[fid, (k + 1) % 3], f[fid, (k + 2) % 3]
        f = f.copy()
        f[fid] = np.stack([u, m, x], axis=1)
        f = np.vstack([f, np.stack([m, w, x], axis=1)])


def _normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])


def _boundary(v: np.ndarray, edges: np.ndarray, count: np.ndarray):
    """Whether each vertex is on the boundary, and how sharply the boundary turns there: ``2 sin(angle / 2)``."""
    b = edges[count == 1]
    on_b = np.zeros(len(v), dtype=bool)
    on_b[b.ravel()] = True
    d = v[b[:, 1]] - v[b[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    turn = np.zeros_like(v)
    np.add.at(turn, b[:, 0], d)
    np.add.at(turn, b[:, 1], -d)
    return on_b, np.linalg.norm(turn, axis=1)


def _collapse(v: np.ndarray, f: np.ndarray, low: float, high: float):
    """Collapse edges shorter than ``low`` where the result keeps the surface manifold, its faces unflipped, and its
    edges shorter than ``high``; independent collapses per round. A collapse keeps a boundary vertex in place, the
    sharper turn of the boundary on a boundary edge, and keeps corners where the boundary turns by over 30 degrees."""
    while True:
        _, edges, _, count = _topology(f)
        length = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        on_b, turn = _boundary(v, edges, count)
        ring = [set() for _ in range(len(v))]
        for a, b in edges:
            ring[a].add(b)
            ring[b].add(a)
        faces_of = [[] for _ in range(len(v))]
        for i, tri in enumerate(f):
            for x in tri:
                faces_of[x].append(i)
        normal, locked, merge = _normals(v, f), np.zeros(len(v), dtype=bool), {}
        for e in np.argsort(length):
            if length[e] >= low:
                break
            a, b = edges[e]
            if locked[a] or locked[b] or (on_b[a] and on_b[b] and count[e] != 1):
                continue
            if on_b[b] and (not on_b[a] or turn[b] > turn[a]):
                a, b = b, a
            if turn[b] > _CORNER:
                continue
            p = v[a] if on_b[a] else 0.5 * (v[a] + v[b])
            near = list(ring[a] | ring[b])
            if len(ring[a] & ring[b]) != count[e] or np.linalg.norm(v[near] - p, axis=1).max() >= high:
                continue
            tris = [t for t in set(faces_of[a] + faces_of[b]) if not (a in f[t] and b in f[t])]
            moved = v[f[tris]]
            moved[np.isin(f[tris], (a, b))] = p
            moved_normal = _normals(moved.reshape(-1, 3), np.arange(3 * len(tris)).reshape(-1, 3))
            if (np.einsum("ij,ij->i", moved_normal, normal[tris]) <= 0).any():
                continue
            merge[b] = (a, p)
            locked[list(ring[a] | ring[b] | {a, b})] = True
        if not merge:
            return v, f
        v = v.copy()
        target = np.arange(len(v))
        for b, (a, p) in merge.items():
            target[b], v[a] = a, p
        f = target[f]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]


def _flip(v: np.ndarray, f: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Flip interior edges toward valence 6, 4 on the boundary, where the flip keeps both faces unflipped;
    independent flips per round."""
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        half, edges, edge_of, count = _topology(f)
        on_b = np.zeros(len(v), dtype=bool)
        on_b[edges[count == 1].ravel()] = True
        valence = np.bincount(edges.ravel(), minlength=len(v))
        target = np.where(on_b, 4, 6)
        # The two half-edges of each interior edge: (a, b) in face f0 with third vertex c, (b, a) in f1 with d.
        inner = np.flatnonzero(count[edge_of] == 2)
        order = inner[np.argsort(edge_of[inner], kind="stable")]
        h0, h1 = order[0::2], order[1::2]
        f0, f1 = h0 // 3, h1 // 3
        a, b = half[h0, 0], half[h0, 1]
        c, d = f[f0, (h0 % 3 + 2) % 3], f[f1, (h1 % 3 + 2) % 3]
        quad = np.stack([a, b, c, d], axis=1)
        before = ((valence[quad] - target[quad]) ** 2).sum(axis=1)
        after = ((valence[quad] + [-1, -1, 1, 1] - target[quad]) ** 2).sum(axis=1)
        absent = ~np.isin(np.minimum(c, d) * len(v) + np.maximum(c, d), edges[:, 0] * len(v) + edges[:, 1])
        n = _normals(v, f[f0]) + _normals(v, f[f1])
        new0, new1 = np.stack([a, d, c], axis=1), np.stack([d, b, c], axis=1)
        same = (np.einsum("ij,ij->i", _normals(v, new0), n) > 0) & (np.einsum("ij,ij->i", _normals(v, new1), n) > 0)
        cand = (after < before) & absent & same & (c != d)
        if not cand.any():
            break
        priority = np.where(cand, before - after + rng.random(len(cand)), -1.0)
        best = np.full(len(v), -1.0)
        for col in quad.T:
            np.maximum.at(best, col[cand], priority[cand])
        win = cand & np.all(best[quad] == priority[:, None], axis=1)
        f = f.copy()
        f[f0[win]], f[f1[win]] = new0[win], new1[win]
    return f


def _relax(v: np.ndarray, f: np.ndarray, surface: wp.Mesh, seg: np.ndarray) -> np.ndarray:
    """Move each vertex halfway to the mean of its neighbors, in its tangent plane, and project it onto the input
    surface; boundary vertices move toward their two boundary neighbors and onto the input boundary ``seg``."""
    _, edges, _, count = _topology(f)
    on_b, turn = _boundary(v, edges, count)
    pair = np.vstack([edges, edges[:, ::-1]])
    keep = ~on_b[pair[:, 0]] | (np.r_[count, count] == 1)
    mean = np.zeros_like(v)
    np.add.at(mean, pair[keep, 0], v[pair[keep, 1]])
    mean /= np.maximum(np.bincount(pair[keep, 0], minlength=len(v)), 1)[:, None]
    normal, face_normal = np.zeros_like(v), _normals(v, f)
    for k in range(3):
        np.add.at(normal, f[:, k], face_normal)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    step = mean - v
    step[~on_b] -= np.einsum("ij,ij->i", step, normal)[~on_b, None] * normal[~on_b]
    used = np.zeros(len(v), dtype=bool)
    used[f.ravel()] = True
    used[turn > _CORNER] = False
    v = np.where(used[:, None], v + 0.5 * step, v)
    out = wp.empty(len(v), dtype=wp.vec3d, device=surface.device)
    wp.launch(_closest_point_kernel, len(v), [surface.id, wp.array(v, dtype=wp.vec3d, device=surface.device)], [out])
    projected = out.numpy()
    b = np.flatnonzero(on_b)
    if len(b):
        d = seg[:, 1] - seg[:, 0]
        t = np.clip(np.einsum("bsk,sk->bs", v[b, None] - seg[None, :, 0], d) / np.einsum("sk,sk->s", d, d), 0.0, 1.0)
        near = seg[None, :, 0] + t[..., None] * d[None]
        projected[b] = near[np.arange(len(b)), np.linalg.norm(near - v[b, None], axis=2).argmin(axis=1)]
    return np.where(used[:, None], projected, v)


def remesh_isotropic(
    vertices: np.ndarray, faces: np.ndarray, edge_length: float | None = None, iterations: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Remesh a triangle surface to edges of about ``edge_length``.

    No edge of the result is longer than 4/3 of it. Each iteration splits the longer edges, collapses those
    shorter than 4/5 of it, flips edges
    toward valence 6, or 4 on the boundary, and moves each vertex in its tangent plane to the mean of its
    neighbors, projected back onto the input surface. Boundary vertices move along the boundary, projected back
    onto the input boundary, so the boundary keeps its shape but its corners round off. As vertices go to the
    nearest point of the input, parts of the surface closer than the edge length may exchange vertices.

    Args:
        vertices: Vertex positions, shape ``(N, 3)``.
        faces: Vertex indices of the triangles, shape ``(M, 3)``, consistently oriented.
        edge_length: Target edge length; the mean edge length of the input when ``None``.
        iterations: Number of iterations.

    Returns:
        A tuple ``(vertices, faces)`` of the remeshed surface.
    """
    v0, f0 = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    _, edges, _, count = _topology(f0)
    if edge_length is None:
        edge_length = float(np.linalg.norm(v0[edges[:, 0]] - v0[edges[:, 1]], axis=1).mean())
    seg = v0[edges[count == 1]]
    surface = wp.Mesh(wp.array(v0, dtype=wp.vec3), wp.array(f0.ravel(), dtype=wp.int32))
    v, f = v0, f0
    for _ in range(iterations):
        v, f = _split(v, f, 4.0 / 3.0 * edge_length)
        v, f = _collapse(v, f, 0.8 * edge_length, 4.0 / 3.0 * edge_length)
        f = _flip(v, f)
        v = _relax(v, f, surface, seg)
    v, f = _split(v, f, 4.0 / 3.0 * edge_length)
    used = np.unique(f)
    index = np.full(len(v), -1)
    index[used] = np.arange(len(used))
    return v[used].astype(np.float32), index[f].astype(np.int32)
