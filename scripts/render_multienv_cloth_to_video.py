# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render the N-env cloth-on-sphere FBA scene to a video / PNG sequence.

Lays out ``N`` cloth-on-sphere worlds in a flat ``side × side`` grid via
``builder.replicate(spacing=(s, 0, s))`` so every env is visible from a
single elevated camera.  Steps with ``SolverFBA(nsn_schur_mode="lite")``
and pulls each rendered frame off the ``ViewerGL`` headless surface to PNG.

Usage::

    # 25 envs (5×5), 300 frames at 1920x1080 -> MP4 at 100 FPS
    uv run python scripts/render_multienv_cloth_to_video.py \
        --n-envs 25 --num-frames 300 \
        --out-dir /tmp/multienv_cloth_frames \
        --out-mp4 /tmp/multienv_cloth.mp4

    # smaller grid for quick iteration
    uv run python scripts/render_multienv_cloth_to_video.py \
        --n-envs 4 --num-frames 100 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton.examples.cloth.example_cloth_on_sphere_fba import (
    _DT,
    _FRICTION_MU,
    _NSN_ITER,
    _OBJ_MASS,
    _PD_ITER,
    _PIN_STIFFNESS,
    _CHECKER_TILES,
    _CLOTH_COLOR,
    _SPHERE_COLOR,
    _make_checker_texture,
    build_cloth_on_sphere_builder,
)
from newton.solvers import SolverFBA


def _write_png(arr_u8: np.ndarray, path: Path) -> None:
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        imageio.imwrite(path, arr_u8)
        return
    except ImportError:
        pass
    from PIL import Image  # noqa: PLC0415

    Image.fromarray(arr_u8, mode="RGB").save(path)


def _stitch_mp4(frame_dir: Path, mp4_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        writer = imageio.get_writer(
            str(mp4_path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
        )
        for png in sorted(frame_dir.glob("frame_*.png")):
            writer.append_data(imageio.imread(str(png)))
        writer.close()
        return
    except ImportError:
        pass
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(mp4_path),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-envs", type=int, default=25,
                        help="Number of worlds. Laid out in a flat ceil(sqrt(N))×ceil(sqrt(N)) grid.")
    parser.add_argument("--spacing", type=float, default=10.0,
                        help="Per-axis grid spacing in meters (XZ; Y always 0 to keep envs at ground).")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/multienv_cloth_frames"))
    parser.add_argument("--out-mp4", type=Path, default=Path("/tmp/multienv_cloth.mp4"))
    parser.add_argument("--fps", type=int, default=100,
                        help="Output FPS.  Default 100 = 1/dt for real-time playback (dt=0.01).")
    parser.add_argument("--no-mp4", action="store_true")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--mode", choices=["full", "lite"], default="lite",
                        help="SolverFBA mode.  Default 'lite' — required for N≥16 to fit on a 32 GB GPU.")
    parser.add_argument("--hero-env", type=int, default=None,
                        help="World index to frame as the focal env (camera looks at it from "
                             "close range).  Other envs recede into the background.  Defaults to "
                             "the env nearest the grid corner closest to the camera.")
    parser.add_argument("--hero-distance", type=float, default=12.0,
                        help="Camera distance from the hero env in meters.")
    parser.add_argument("--hero-elev", type=float, default=6.0,
                        help="Camera height above the hero env in meters.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("frame_*.png"):
        stale.unlink()

    wp.init()

    # ---- Build the replicated model ------------------------------------
    sub = build_cloth_on_sphere_builder()
    n_particles_per_env = sub.particle_count
    n_tris_per_env = len(sub.tri_indices) // 3 if sub.tri_indices else 0
    print(f"[render] N={args.n_envs} mode={args.mode} frames={args.num_frames}")
    print(f"  particles/env={n_particles_per_env}  tris/env={n_tris_per_env}")

    if args.n_envs == 1:
        main_b = sub
    else:
        main_b = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=sub.gravity)
        # Flat XZ grid — Y spacing is zero so all cloths sit on their own
        # sphere at the same height (the camera frames a single horizontal
        # plane instead of a stacked tower).
        main_b.replicate(
            sub,
            world_count=args.n_envs,
            spacing=(args.spacing, 0.0, args.spacing),
        )
    model = main_b.finalize()
    print(f"  model: particle_count={model.particle_count} shape_count={model.shape_count} "
          f"world_count={model.world_count}")

    # ---- Mass + solver -------------------------------------------------
    uniform_mass = _OBJ_MASS / n_particles_per_env
    model.particle_mass.assign(
        np.full(model.particle_count, uniform_mass, dtype=np.float32)
    )
    model.particle_inv_mass.assign(
        np.full(model.particle_count, 1.0 / uniform_mass, dtype=np.float32)
    )
    mu_override = np.full(model.particle_count, _FRICTION_MU, dtype=np.float64)
    solver = SolverFBA(
        model,
        iterations=_PD_ITER,
        nsn_iterations=_NSN_ITER,
        friction=True,
        stretching_model="arap",
        mu_per_pair_override=mu_override,
        pin_stiffness=_PIN_STIFFNESS,
        nsn_schur_mode=args.mode,
    )
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()

    # ---- Viewer --------------------------------------------------------
    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    viewer.show_triangles = False  # we'll log a textured cloth mesh ourselves
    viewer.set_model(model)
    # ``set_model`` auto-computes a viewer-side world-offset grid (via
    # ``_auto_compute_world_offsets``).  We physically replicated worlds via
    # ``builder.replicate(spacing=...)``, so ``model.shape_transform`` and
    # ``state.particle_q`` are already in correct world coordinates.  Leaving
    # the auto offsets in place would double-offset every shape relative to
    # the cloth we log via ``log_mesh(particle_q)`` — which is exactly the
    # cloth-vs-sphere misalignment in earlier multi-env videos.  Reset to zero.
    viewer.set_world_offsets((0.0, 0.0, 0.0))

    # Combined tri-index array — one big strip containing every world's
    # triangles, particle indices already offset per-world by replicate().
    cloth_tri_indices = model.tri_indices.flatten()

    # Planar UVs from rest positions, with per-env modulo so each env gets
    # its own tiled checker (otherwise the seam-running texture coords
    # across worlds would alias into a single huge sheet).
    verts_rest = state_0.particle_q.numpy()
    # Convert world-space coords back into per-env local coords for UV tiling.
    # particle_world maps each particle to its world index.
    p_world = model.particle_world.numpy().astype(np.int64) if model.particle_world is not None else np.zeros(model.particle_count, dtype=np.int64)
    # Per-world XZ bounding box (in world coords) so per-env UVs line up.
    uv = np.zeros((model.particle_count, 2), dtype=np.float32)
    for w in range(int(model.world_count)):
        mask = p_world == w
        if not mask.any():
            continue
        xs = verts_rest[mask, 0]
        zs = verts_rest[mask, 2]
        x_range = max(float(xs.max() - xs.min()), 1e-6)
        z_range = max(float(zs.max() - zs.min()), 1e-6)
        uv[mask, 0] = ((xs - xs.min()) / x_range) * _CHECKER_TILES
        uv[mask, 1] = ((zs - zs.min()) / z_range) * _CHECKER_TILES
    cloth_uvs = wp.array(uv, dtype=wp.vec2)
    cloth_texture = _make_checker_texture()

    # ---- Camera: hero-focused with the rest of the grid receding ------
    # ``compute_world_offsets`` centers the grid around the origin, so per-env
    # world positions are at ``(d1·s - cx, 0, d0·s - cz)`` for d0/d1 in
    # ``0..side-1`` with ``cx = cz = (side-1)·s/2``.  The hero env defines
    # what the camera looks at; the rest of the grid stays in frame because
    # we place the camera *outside* the hero env on the side facing away
    # from the grid centroid, so all peripheral envs sit behind the hero.
    from newton._src.utils import compute_world_offsets  # noqa: PLC0415
    side = int(math.ceil(math.sqrt(max(args.n_envs, 1))))
    offsets_np = np.asarray(
        compute_world_offsets(args.n_envs, (args.spacing, 0.0, args.spacing), newton.Axis.Y)
    )

    # Default hero: the env closest to ``(-extent/2, 0, -extent/2)`` so the
    # camera placed on its outside face naturally points back through the
    # whole grid.  Sweep over offsets so this works for non-square N too.
    if args.hero_env is None:
        target_corner = np.array([offsets_np[:, 0].min(), 0.0, offsets_np[:, 2].min()])
        hero = int(np.argmin(np.linalg.norm(offsets_np - target_corner, axis=1)))
    else:
        hero = int(args.hero_env) % args.n_envs
    hero_pos = offsets_np[hero].copy()

    # Camera direction: away from the grid centroid (origin) through the hero.
    # That way the hero is the closest env to the camera and every other env
    # sits behind it, naturally receding into the frame.
    grid_center = offsets_np.mean(axis=0).astype(np.float32)
    dir_vec = hero_pos - grid_center
    if np.linalg.norm(dir_vec[[0, 2]]) < 1e-6:
        # Hero is the grid centroid — fall back to a fixed diagonal so the
        # camera still has a defined direction (target stays the hero).
        dir_vec = np.array([-1.0, 0.0, -1.0], dtype=np.float32)
    dir_xz = dir_vec.copy()
    dir_xz[1] = 0.0
    dir_xz /= max(float(np.linalg.norm(dir_xz)), 1e-6)
    cam_xz = hero_pos[[0, 2]] + dir_xz[[0, 2]] * args.hero_distance
    cam_pos = wp.vec3(float(cam_xz[0]), float(args.hero_elev), float(cam_xz[1]))
    cam_target = wp.vec3(float(hero_pos[0]), 1.0, float(hero_pos[2]))
    if hasattr(viewer, "camera"):
        viewer.camera.pos = viewer.camera._as_vec3(cam_pos)
        viewer.camera.look_at(cam_target)
    print(f"  grid: side={side} extent={(side - 1) * args.spacing:.1f}m centered at origin")
    print(f"  hero env: idx={hero} at xz=({hero_pos[0]:.1f}, {hero_pos[2]:.1f})")
    print(f"  camera pos={tuple(cam_pos)}, target={tuple(cam_target)}")

    # ---- Pre-allocate frame buffer + run loop --------------------------
    frame_buf = wp.empty(shape=(args.height, args.width, 3), dtype=wp.uint8, device=viewer.device)
    sim_time = 0.0
    t_start = time.perf_counter()
    written = 0
    for f in range(args.num_frames):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, _DT)
        state_0, state_1 = state_1, state_0
        sim_time += _DT

        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_mesh(
            "/cloth",
            state_0.particle_q,
            cloth_tri_indices,
            uvs=cloth_uvs,
            texture=cloth_texture,
            color=_CLOTH_COLOR,
            backface_culling=False,
        )
        viewer.end_frame()

        if f % args.render_every == 0:
            viewer.get_frame(target_image=frame_buf)
            img_np = frame_buf.numpy()
            path = args.out_dir / f"frame_{written:06d}.png"
            _write_png(img_np, path)
            written += 1
        if f % 50 == 0 and f > 0:
            elapsed = time.perf_counter() - t_start
            fps = (f + 1) / elapsed
            print(f"  frame {f}/{args.num_frames}  fps={fps:.1f}  png_written={written}")

    elapsed = time.perf_counter() - t_start
    print(f"Done. {written} PNGs in {elapsed:.1f}s -> {args.out_dir}")

    if not args.no_mp4:
        out_fps = max(1, args.fps // max(1, args.render_every))
        print(f"Stitching MP4 -> {args.out_mp4} @ {out_fps} fps")
        _stitch_mp4(args.out_dir, args.out_mp4, out_fps)
        size_mb = args.out_mp4.stat().st_size / 1024 / 1024
        print(f"Wrote {args.out_mp4}  ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
