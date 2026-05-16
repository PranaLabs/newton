# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Medit ``.mesh`` parser (text format, 1-based tet indices)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_medit_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Medit ``.mesh`` file.

    Args:
        path: Path to ``foo_volume_NP.mesh``.

    Returns:
        ``(verts, tets)`` where ``verts`` is ``(V, 3) float64`` and ``tets`` is
        ``(T, 4) int32`` with zero-based indices.
    """
    vertices: list[tuple[float, float, float]] = []
    tets: list[tuple[int, int, int, int]] = []
    state: str | None = None
    n_verts = 0
    n_tets = 0
    vert_count = 0
    tet_count_read = 0

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Vertices"):
                state = "vcount"
                continue
            if line.startswith("Tetrahedra"):
                state = "tcount"
                continue
            if line in ("End", "End\n"):
                break
            if line.startswith(("MeshVersionFormatted", "Dimension", "Triangles", "Edges", "Normals")):
                state = "skip_section" if line.startswith(("Triangles", "Edges", "Normals")) else "header"
                continue

            if state == "vcount":
                n_verts = int(line)
                state = "verts"
                continue
            if state == "tcount":
                n_tets = int(line)
                state = "tets"
                continue
            if state == "verts" and vert_count < n_verts:
                parts = line.split()
                vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                vert_count += 1
                if vert_count == n_verts:
                    state = None
                continue
            if state == "tets" and tet_count_read < n_tets:
                parts = line.split()
                tets.append((int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1))
                tet_count_read += 1
                if tet_count_read == n_tets:
                    state = None
                continue
            if state == "skip_section":
                try:
                    int(line)
                except ValueError:
                    pass
                continue

    return np.array(vertices, dtype=np.float64), np.array(tets, dtype=np.int32)
