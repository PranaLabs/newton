# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Multi-layer cloth-on-sphere stacking demo.

Same single-env sphere as ``example_cloth_on_sphere_fba`` but with ``N``
cloth instances dropped from staggered heights so they land sequentially
on top of each other (or pass through each other when self-contact is
off, which is the default — that's the point of the baseline).

Usage::

    uv run python scripts/fba_cloth_layers_on_sphere.py \
        --n-layers 3 --num-frames 600 \
        --render-out /tmp/cloth_layers_baseline.mp4

    # With self-contact enabled (using Phase 3 path):
    uv run python scripts/fba_cloth_layers_on_sphere.py \
        --n-layers 3 --num-frames 600 --self-contact \
        --render-out /tmp/cloth_layers_self.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton.solvers import SolverFBA

_MESH_PATH = Path("/home/ziqiu/work/RealSim_py/realsim_py/resources/mesh/cloth/square_1013P.obj")

# Single-env cloth-on-sphere defaults (mirror example_cloth_on_sphere_fba).
_DT = 0.01
_PD_ITER = 5
_NSN_ITER = 1
_GRAVITY = -10.0
_OBJ_MASS_PER_LAYER = 1000.0
_YOUNG = 1.0e5
_POISSON = 0.4
_BENDING = 20.0
_FRICTION_MU = 0.3
_PIN_STIFFNESS = 1.0e10

_CLOTH_SCALE = 3.5
_CLOTH_ROT_X_DEG = 90.0  # lay flat on XZ plane
# Stagger each layer ``_LAYER_DY`` metres higher than the previous so they
# fall sequentially and approximately settle at the same time on the sphere.
_LAYER_DY = 1.5
_LAYER_Y0 = 3.5  # bottom-most layer's centre height

_SPHERE_RADIUS = 3.0
_SPHERE_VISUAL_RADIUS = 2.90
_SPHERE_CENTER = (0.0, 0.0, 0.0)
_SPHERE_COLOR = (0.55, 0.55, 0.65)

_PARTICLE_RADIUS = 0.04
_CONTACT_MARGIN = 0.05

# Per-layer colours so the rendered stacking is legible.  Cycled if
# ``n_layers`` exceeds the palette.
_LAYER_COLORS = [
    (0.78, 0.32, 0.36),  # muted red
    (0.32, 0.55, 0.72),  # steel blue
    (0.78, 0.62, 0.32),  # warm tan
    (0.45, 0.70, 0.42),  # sage green
    (0.62, 0.42, 0.70),  # plum
]


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, tris = [], []
    with open(path) as f:
        for ln in f:
            tok = ln.split()
            if not tok:
                continue
            if tok[0] == "v":
                verts.append([float(x) for x in tok[1:4]])
            elif tok[0] == "f":
                tris.append([int(x.split("/")[0]) - 1 for x in tok[1:4]])
    return np.array(verts, np.float32), np.array(tris, np.int32)


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _lame(y: float, n: float) -> tuple[float, float]:
    mu = y / (2 * (1 + n))
    lam = y * n / ((1 + n) * (1 - 2 * n))
    return mu, lam


def build_scene(n_layers: int):
    verts_local, tris = _load_obj(_MESH_PATH)
    R = _rot_x(_CLOTH_ROT_X_DEG)
    n_per_layer = len(verts_local)
    tris_per_layer = len(tris)

    b = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=_GRAVITY)

    mu, _lam = _lame(_YOUNG, _POISSON)
    layer_tri_ranges = []  # (tri_start, tri_count) per layer, for rendering
    layer_particle_ranges = []

    for layer in range(n_layers):
        # Scale, rotate, then lift to layer height.  Per-layer tiny X jitter so
        # layers don't perfectly overlap (helps the broadphase de-tangle when
        # self-contact is on).
        verts_world = (verts_local * _CLOTH_SCALE) @ R.T
        verts_world[:, 1] += _LAYER_Y0 + layer * _LAYER_DY
        verts_world[:, 0] += 0.01 * layer  # break perfect overlap
        verts_world = verts_world.astype(np.float32)

        e1 = verts_world[tris[:, 1]] - verts_world[tris[:, 0]]
        e2 = verts_world[tris[:, 2]] - verts_world[tris[:, 0]]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        density = _OBJ_MASS_PER_LAYER / max(float(area.sum()), 1e-12)

        layer_particle_start = b.particle_count
        layer_tri_start = b.tri_count
        b.add_cloth_mesh(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0, 0, 0),
            vertices=[wp.vec3(*v.tolist()) for v in verts_world],
            indices=tris.reshape(-1).tolist(),
            density=density,
            tri_ke=2.0 * mu,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=_BENDING,
            edge_kd=0.0,
            particle_radius=_PARTICLE_RADIUS,
        )
        layer_particle_ranges.append((layer_particle_start, n_per_layer))
        layer_tri_ranges.append((layer_tri_start, tris_per_layer))

    # Sphere collider — split visual / collision (mirrors cloth_on_sphere).
    xform_world = wp.transform(wp.vec3(*_SPHERE_CENTER), wp.quat_identity())

    cfg_collision = b.default_shape_cfg.copy()
    cfg_collision.mu = _FRICTION_MU
    cfg_collision.is_visible = False
    b.add_shape_sphere(
        body=-1, xform=xform_world, radius=_SPHERE_RADIUS, cfg=cfg_collision,
    )
    cfg_visual = b.default_site_cfg.copy()
    b.add_shape_sphere(
        body=-1, xform=xform_world, radius=_SPHERE_VISUAL_RADIUS,
        cfg=cfg_visual, as_site=True, color=_SPHERE_COLOR,
    )

    return b, layer_particle_ranges, layer_tri_ranges


def run(args):
    wp.init()

    b, layer_particle_ranges, layer_tri_ranges = build_scene(args.n_layers)
    model = b.finalize()
    print(f"layers={args.n_layers}  particles={model.particle_count}  shapes={model.shape_count}")

    uniform_mass = _OBJ_MASS_PER_LAYER / layer_particle_ranges[0][1]
    model.particle_mass.assign(
        np.full(model.particle_count, uniform_mass, dtype=np.float32)
    )
    model.particle_inv_mass.assign(
        np.full(model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
    )
    mu_o = np.full(model.particle_count, _FRICTION_MU, dtype=np.float64)

    solver_kwargs = dict(
        iterations=_PD_ITER,
        nsn_iterations=_NSN_ITER,
        friction=True,
        stretching_model="arap",
        mu_per_pair_override=mu_o,
        pin_stiffness=_PIN_STIFFNESS,
        nsn_schur_mode="lite",
    )
    if args.self_contact:
        solver_kwargs["particle_contact_radius"] = _PARTICLE_RADIUS * 2.0
        solver_kwargs["particle_contact_margin"] = 0.04
        solver_kwargs["particle_contact_friction"] = 0.25
        solver_kwargs["particle_contact_topology_ring"] = 2
        solver_kwargs["particle_contact_rest_exclusion_radius"] = 0.1
        solver_kwargs["particle_contact_ee"] = not args.no_ee
    solver = SolverFBA(model, **solver_kwargs)

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=_CONTACT_MARGIN)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()

    # ---- Viewer ----
    viewer = None
    frame_buf = None
    out_dir = None
    layer_tri_indices_d = []
    layer_particle_idx_d = []
    if args.render_out is not None:
        viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
        viewer.show_triangles = False
        viewer.set_model(model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        # Per-layer triangle index arrays.  ``model.tri_indices`` is laid out
        # by layer order since we called add_cloth_mesh sequentially.
        all_tri = model.tri_indices.numpy()
        for (tri_start, tri_count) in layer_tri_ranges:
            layer_tris = all_tri[tri_start : tri_start + tri_count].flatten()
            layer_tri_indices_d.append(wp.array(layer_tris, dtype=wp.int32, device=viewer.device))
        if hasattr(viewer, "camera"):
            viewer.camera.pos = viewer.camera._as_vec3(wp.vec3(6.5, 4.0, 6.5))
            viewer.camera.look_at(wp.vec3(0.0, 1.0, 0.0))
        frame_buf = wp.empty(shape=(args.height, args.width, 3), dtype=wp.uint8, device=viewer.device)
        out_dir = Path(str(args.render_out.with_suffix("")) + "_frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("frame_*.png"):
            stale.unlink()

    # ---- Run ----
    step_ms = np.empty(args.num_frames, dtype=np.float32)
    sim_time = 0.0
    t_start = time.perf_counter()
    for f in range(args.num_frames):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        wp.synchronize_device()
        t = time.perf_counter()
        solver.step(state_0, state_1, None, contacts, _DT)
        wp.synchronize_device()
        step_ms[f] = (time.perf_counter() - t) * 1000.0
        state_0, state_1 = state_1, state_0
        sim_time += _DT

        if viewer is not None:
            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            for i, ((p_start, p_count), tri_indices) in enumerate(zip(
                layer_particle_ranges, layer_tri_indices_d
            )):
                color = _LAYER_COLORS[i % len(_LAYER_COLORS)]
                viewer.log_mesh(
                    f"/cloth_layer_{i}",
                    state_0.particle_q,
                    tri_indices,
                    color=color,
                    backface_culling=False,
                )
            viewer.end_frame()
            viewer.get_frame(target_image=frame_buf)
            png = out_dir / f"frame_{f:06d}.png"
            try:
                import imageio.v2 as imageio  # noqa: PLC0415
                imageio.imwrite(png, frame_buf.numpy())
            except ImportError:
                from PIL import Image  # noqa: PLC0415
                Image.fromarray(frame_buf.numpy(), mode="RGB").save(png)

        if f % 50 == 0 or f == args.num_frames - 1:
            q = state_0.particle_q.numpy()
            nc = int(contacts.soft_contact_count.numpy()[0]) if hasattr(contacts, "soft_contact_count") else 0
            print(f"  frame {f}: y=[{q[:,1].min():.3f}, {q[:,1].max():.3f}] "
                  f"n_contacts={nc} step_ms={step_ms[f]:.2f}")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone {args.num_frames} frames in {elapsed:.1f}s.  "
          f"step_mean={step_ms.mean():.2f} ms  "
          f"settled step_mean={step_ms[100:].mean():.2f} ms")

    if args.render_out is not None:
        print(f"Stitching MP4 -> {args.render_out}")
        try:
            import imageio.v2 as imageio  # noqa: PLC0415
            w = imageio.get_writer(
                str(args.render_out), fps=100, codec="libx264",
                quality=8, macro_block_size=1,
            )
            for png in sorted(out_dir.glob("frame_*.png")):
                w.append_data(imageio.imread(str(png)))
            w.close()
        except ImportError:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "100",
                "-i", str(out_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(args.render_out),
            ], check=True)
        size_mb = args.render_out.stat().st_size / 1024 / 1024
        print(f"Wrote {args.render_out}  ({size_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--num-frames", type=int, default=600)
    p.add_argument("--render-out", type=Path, default=Path("/tmp/cloth_layers.mp4"))
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--self-contact", action="store_true",
                   help="Enable Phase 3 cloth self-contact (BSR + sparse-PCR).")
    p.add_argument("--no-ee", action="store_true",
                   help="Disable edge-edge contacts (v-t only).  Default: e-e ON.")
    args = p.parse_args()
    if args.no_render:
        args.render_out = None
    return run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
