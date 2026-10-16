# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cables through tetrahedral bodies, after RealSim's ``CableController`` and SOFA SoftRobots' cable constraint."""

import numpy as np


class Cable:
    """A cable through a tetrahedral body that pulls its embedded points together.

    The cable is a list of segments: segment ``s`` joins ``points[2 * s]`` and ``points[2 * s + 1]``, each
    embedded at the rest pose in the tetrahedron that contains it. :meth:`rows` turns every segment into
    one row for :meth:`SolverFBA.set_particle_constraints` along the segment's current direction, whose
    value is the segment's share of :attr:`length` in proportion to the rest lengths. The rows are hard and
    their multipliers are the segment tensions. :attr:`length` is clamped to ``[min_ratio, 1]`` times the
    rest length, so the cable pulls but does not push. Concatenate the rows of several cables to drive
    them together.

    Args:
        model: The model whose tetrahedra hold the cable.
        points: Segment end points [m] at the rest pose, shape ``(2 * S, 3)``; repeat a point to chain segments.
        min_ratio: Shortest length as a fraction of the rest length.
    """

    def __init__(self, model, points, min_ratio: float = 0.85):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0 or len(points) % 2:
            raise ValueError("a cable needs pairs of points")
        x = model.particle_q.numpy().astype(np.float64)
        tets = model.tet_indices.numpy().reshape(-1, 4)
        v0 = x[tets[:, 0]]
        inv = np.linalg.inv(np.stack([x[tets[:, k]] - v0 for k in (1, 2, 3)], axis=2))
        self.particles = np.empty((len(points), 4), dtype=np.int32)
        self.weights = np.empty((len(points), 4))
        for i, p in enumerate(points):
            b = np.einsum("tij,tj->ti", inv, p - v0)
            bary = np.column_stack([1.0 - b.sum(axis=1), b])
            best = int(bary.min(axis=1).argmax())
            if bary[best].min() < -1e-6:
                raise ValueError(f"cable point {i} lies outside the tetrahedra")
            self.particles[i], self.weights[i] = tets[best], bary[best]
        rest = np.linalg.norm(points[1::2] - points[0::2], axis=1)
        self.rest_length = float(rest.sum())
        """Sum of the segment lengths at the rest pose [m]."""
        self.share = rest / self.rest_length
        self.min_ratio = float(min_ratio)
        self.length = self.rest_length
        """Commanded cable length [m]."""

    def rows(self, particle_q):
        """Rows ``(particles, coefficients, directions, targets)`` of the segments at the positions ``particle_q``."""
        x = np.asarray(particle_q.numpy() if hasattr(particle_q, "numpy") else particle_q, dtype=np.float64)
        p = np.einsum("pk,pkd->pd", self.weights, x[self.particles])
        direction = p[1::2] - p[0::2]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        length = np.clip(self.length, self.min_ratio * self.rest_length, self.rest_length)
        particles = np.hstack([self.particles[0::2], self.particles[1::2]])
        coefficients = np.hstack([-self.weights[0::2], self.weights[1::2]])
        return particles, coefficients, direction, length * self.share
