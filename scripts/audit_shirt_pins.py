# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure actual cloth intersections independently of FBA's contact candidate count."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.fba.self_contact import SelfContact
from newton.examples.cloth.example_shirt_pins_fba import Example
from newton.examples.pin_demo_utils import SurfaceIntersectionCheck
from newton.viewer import ViewerNull


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1320)
    parser.add_argument("--mode", choices=["full", "lite"])
    parser.add_argument("--substeps", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--stretching", type=float)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--bending", type=float)
    parser.add_argument("--rest-radius", type=float)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--rings", type=int)
    parser.add_argument("--linear-iterations", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    ex = Example(ViewerNull(), None)
    if args.mode:
        ex.solver.nsn_schur_mode = args.mode
    if args.rings is not None:
        old = ex.solver._self_contact
        ex.solver._self_contact = SelfContact(ex.model, old.mode, old.query_radius, args.rings, old.rest_radius)
    if args.linear_iterations:
        ex.solver.nsn_linear_iterations = args.linear_iterations
    if args.substeps:
        ex.sim_substeps = args.substeps
        ex.sim_dt = ex.frame_dt / args.substeps
    if args.iterations:
        ex.solver.iterations = args.iterations
    if args.radius is not None:
        ex.solver.particle_self_contact_margin = args.radius
    if args.stretching is not None:
        props = ex.model.tri_materials.numpy()
        props[:, 0] = args.stretching
        ex.model.tri_materials.assign(props)
    if args.bending is not None:
        bends = ex.model.edge_bending_properties.numpy()
        bends[:, 0] = args.bending
        ex.model.edge_bending_properties.assign(bends)
    if args.rest_radius is not None:
        ex.solver._self_contact.rest_radius = args.rest_radius
    if args.margin is not None:
        ex.solver._self_contact.query_radius = args.margin
    detector = SurfaceIntersectionCheck(ex.model)
    tris = ex.model.tri_indices.numpy()
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    rest = ex.rest
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    print(
        "mesh edges percentiles",
        np.percentile(lengths, [0, 50, 95, 100]),
        "mass",
        ex.model.particle_mass.numpy().sum(),
        flush=True,
    )
    history = []
    snapshots = []
    worst = 0
    start = time.perf_counter()
    for frame in range(args.frames + 1):
        if frame:
            ex.step()
        intersections = detector.intersections(ex.state_0.particle_q)
        worst = max(worst, intersections)
        p = ex.state_0.particle_q.numpy()
        ratio = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1) / lengths
        history.append([frame, intersections, float(np.percentile(ratio, 99)), float(ratio.max())])
        if frame % 60 == 0:
            snapshots.append(p)
            print(
                f"{frame}: intersections={intersections}, edge strain p99={np.percentile(ratio, 99):.3f}, max={ratio.max():.3f}, pin error={ex.pins.error(p):.5f}, solver contacts={ex.solver._contact_count}, elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
        if frame % 120 == 0 or frame == args.frames:
            np.savez_compressed(
                args.out / "audit.npz", history=np.array(history), snapshots=np.array(snapshots), triangles=tris
            )
    result = {
        "worst_intersecting_pairs": worst,
        "intersecting_frames": sum(x[1] > 0 for x in history),
        "max_edge_ratio": max(x[3] for x in history),
        "frames": args.frames,
        "seconds": time.perf_counter() - start,
    }
    (args.out / "audit.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == "__main__":
    main()
