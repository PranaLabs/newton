# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isotropic remeshing of triangle surfaces after Botsch and Kobbelt, "A Remeshing Approach to Multiresolution
Modeling" (2004): edge splits and collapses toward a target edge length."""

from __future__ import annotations

import numpy as np

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
