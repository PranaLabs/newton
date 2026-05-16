# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pin selection helpers — AABB in object-local coords -> particle index list."""

from __future__ import annotations

import numpy as np


def select_in_aabb(
    verts: np.ndarray,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
) -> np.ndarray:
    """Return ``int32`` indices of vertices whose coordinates are within ``[lo, hi]``.

    Args:
        verts: ``(V, 3)`` vertex positions.
        lo: Lower AABB bound ``(x, y, z)``.
        hi: Upper AABB bound ``(x, y, z)``.

    Returns:
        ``(K,)`` array of selected vertex indices in ascending order.
    """
    lo_arr = np.asarray(lo, dtype=verts.dtype)
    hi_arr = np.asarray(hi, dtype=verts.dtype)
    mask = np.all((verts >= lo_arr) & (verts <= hi_arr), axis=1)
    return np.flatnonzero(mask).astype(np.int32)
