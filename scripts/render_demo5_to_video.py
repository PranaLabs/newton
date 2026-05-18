# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render the SqueezingBall FBA example to a video / PNG sequence.

Drives ``newton.examples.softbody.example_squeezing_ball_fba.Example`` with a
headless ``ViewerGL`` and grabs each rendered frame via ``viewer.get_frame``.
Frames are written as PNGs and (optionally) stitched to an MP4 via ffmpeg.

Usage::

    # 600 frames -> PNG sequence + MP4 at /tmp/squeezing_ball.mp4
    uv run python scripts/render_demo5_to_video.py \
        --num-frames 600 \
        --out-dir /tmp/squeezing_ball_frames \
        --out-mp4 /tmp/squeezing_ball.mp4 \
        --width 1280 --height 720

The MP4 stitch uses ``imageio[ffmpeg]`` if available, otherwise falls back to
shelling out to ``ffmpeg``. Pass ``--no-mp4`` to only emit the PNG sequence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp


def _write_png(arr_u8: np.ndarray, path: Path) -> None:
    """Write a uint8 HxWx3 array to PNG (PIL fallback if imageio absent)."""
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        imageio.imwrite(path, arr_u8)
        return
    except ImportError:
        pass
    from PIL import Image  # noqa: PLC0415

    Image.fromarray(arr_u8, mode="RGB").save(path)


def _stitch_mp4(frame_dir: Path, mp4_path: Path, fps: int) -> None:
    """Encode the PNG sequence in ``frame_dir`` to an MP4 at ``fps``.

    Prefers imageio's ffmpeg writer; falls back to a direct ffmpeg subprocess.
    """
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        writer = imageio.get_writer(
            str(mp4_path),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        for png in sorted(frame_dir.glob("frame_*.png")):
            writer.append_data(imageio.imread(str(png)))
        writer.close()
        return
    except ImportError:
        pass
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(mp4_path),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-frames", type=int, default=600, help="Number of frames to render.")
    parser.add_argument("--width", type=int, default=1920, help="Output frame width in pixels.")
    parser.add_argument("--height", type=int, default=1080, help="Output frame height in pixels.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/squeezing_ball_frames"),
        help="Directory for the PNG sequence (created if absent).",
    )
    parser.add_argument(
        "--out-mp4",
        type=Path,
        default=Path("/tmp/squeezing_ball.mp4"),
        help="MP4 output path.  Ignored when --no-mp4 is set.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=100,
        help="Output video FPS (default 100 = 1/dt for real-time playback at dt=0.01).",
    )
    parser.add_argument("--no-mp4", action="store_true", help="Skip MP4 stitching; PNGs only.")
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every Nth sim step (use >1 for slow-motion-free fast renders).",
    )
    parser.add_argument(
        "--example",
        type=str,
        default="squeezing_ball_fba",
        help="Newton example module under newton.examples.softbody to drive "
        "(e.g. squeezing_ball_fba, pulling_wooper_fba).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("frame_*.png"):
        stale.unlink()

    wp.init()
    import newton.viewer  # noqa: PLC0415

    viewer = newton.viewer.ViewerGL(
        width=args.width,
        height=args.height,
        headless=True,
    )
    print(f"ViewerGL headless {args.width}x{args.height}")

    ex_args = argparse.Namespace(
        num_frames=args.num_frames,
        headless=True,
        rerun_address="",
        viewer="gl",
        output_path=None,
    )
    import importlib  # noqa: PLC0415
    # The example modules live under newton.examples.<category>; probe the
    # common categories rather than hard-coding ``softbody``.
    mod = None
    for category in ("softbody", "cloth", "rigid", "robot"):
        try:
            mod = importlib.import_module(f"newton.examples.{category}.example_{args.example}")
            break
        except ModuleNotFoundError:
            continue
    if mod is None:
        raise ModuleNotFoundError(f"newton.examples.*.example_{args.example} not found")
    Example = mod.Example
    ex = Example(viewer, ex_args)
    print(f"Model: particle_count={ex.model.particle_count}, shape_count={ex.model.shape_count}")

    # Pre-allocate the read-back buffer (avoids realloc each frame).
    frame_buf = wp.empty(shape=(args.height, args.width, 3), dtype=wp.uint8, device=viewer.device)

    t_start = time.perf_counter()
    written = 0
    for f in range(args.num_frames):
        ex.step()
        ex.render()
        if f % args.render_every == 0:
            viewer.get_frame(target_image=frame_buf)
            img_np = frame_buf.numpy()
            # ``get_frame`` already returns top-left origin (per its docstring:
            # "Origin is top-left (OpenGL's bottom-left is flipped)"), so no
            # extra vertical flip is needed.
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
        print(f"Wrote {args.out_mp4}  ({args.out_mp4.stat().st_size / 1024 / 1024:.1f} MB)")

    sys.exit(0)


if __name__ == "__main__":
    main()
