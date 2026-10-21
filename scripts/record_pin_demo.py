# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record and validate the pin demos without a display.

uv run --no-sync python scripts/record_pin_demo.py bunny --frames 1080
uv run --no-sync python scripts/record_pin_demo.py shirt --frames 1320
Requires imageio-ffmpeg only for encoding. Saves MP4, preview JPG and metrics JSON.
"""

import argparse
import importlib
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import warp as wp

import newton.viewer


def main():
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser()
    parser.add_argument("demo", choices=["bunny", "shirt"])
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/pin_demos"))
    args = parser.parse_args()
    frames = args.frames or (1080 if args.demo == "bunny" else 1320)
    module = "softbody.example_bunny_ears_fba" if args.demo == "bunny" else "cloth.example_shirt_pins_fba"
    title = "BUNNY / TWO EAR PINS" if args.demo == "bunny" else "SHIRT / TWO SHOULDER PINS"
    args.out.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    viewer = newton.viewer.ViewerGL(width=1280, height=720, headless=True)
    viewer.show_ui = False
    example = importlib.import_module("newton.examples." + module).Example(viewer, None)
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(fontpath, 22)
    big = ImageFont.truetype(fontpath, 28)
    small = ImageFont.truetype(fontpath, 16)
    video = args.out / (args.demo + "_pins.mp4")
    encoder = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x720",
            "-r",
            str(example.fps / args.every),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        stdin=subprocess.PIPE,
        stderr=(args.out / (args.demo + "_encode.log")).open("w"),
    )
    start = time.perf_counter()
    max_contacts = 0
    samples = []
    shots = []
    max_tet_inversion = 0
    try:
        example.test_post_step()
        for frame in range(frames + 1):
            if frame:
                example.step()
                example.test_post_step()
            if frame % args.every == 0:
                example.render()
                example.render()
                im = Image.fromarray(viewer.get_frame().numpy().astype(np.uint8))
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 1280, 79), fill=(13, 22, 33))
                draw.text((28, 13), title, font=big, fill=(237, 243, 249))
                draw.text(
                    (30, 49),
                    "Newton SolverFBA  /  GPU simulation  /  self-collision ON",
                    font=small,
                    fill=(155, 178, 192),
                )
                draw.ellipse((916, 27, 928, 39), fill=(31, 217, 255))
                draw.text((937, 22), "PIN A", font=small, fill="white")
                draw.ellipse((1024, 27, 1036, 39), fill=(255, 89, 31))
                draw.text((1045, 22), "PIN B", font=small, fill="white")
                draw.rectangle((0, 650, 1280, 720), fill=(13, 22, 33))
                draw.text((28, 665), example.phase, font=font, fill=(235, 241, 247))
                draw.text(
                    (1000, 669),
                    f"{example.sim_time:05.2f} / {frames / example.fps:.0f} s",
                    font=font,
                    fill=(155, 178, 192),
                )
                draw.rectangle((0, 716, int(1280 * frame / max(frames, 1)), 719), fill=(38, 192, 209))
                encoder.stdin.write(np.asarray(im).tobytes())
                if frame % 120 == 0 or frame == frames:
                    shot = args.out / f"{args.demo}_{frame:04d}.jpg"
                    im.save(shot, quality=92)
                    shots.append(shot)
            if frame % 60 == 0:
                sc = example.solver._self_contact
                count = int(sc.find(example.state_0.particle_q))
                max_contacts = max(max_contacts, count)
                pos = example.state_0.particle_q.numpy()
                if args.demo == "bunny":
                    e = pos[example.tets]
                    det = np.linalg.det(np.stack((e[:, 1] - e[:, 0], e[:, 2] - e[:, 0], e[:, 3] - e[:, 0]), axis=2))
                    max_tet_inversion = max(max_tet_inversion, int(np.count_nonzero(det <= 0)))
                samples.append(
                    {
                        "time": example.sim_time,
                        "phase": example.phase,
                        "self_contact_candidates": count,
                        "pin_error_m": example.pins.error(pos),
                    }
                )
                print(
                    f"{args.demo}: {frame}/{frames}, self-contact candidates={count}, elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )
        example.test_final()
    finally:
        encoder.stdin.close()
        code = encoder.wait()
        viewer.close()
    assert code == 0, "video encoder failed"
    metrics = {
        "demo": args.demo,
        "frames": frames,
        "duration_s": frames / example.fps,
        "particles": example.model.particle_count,
        "self_collision": True,
        "max_self_contact_candidates": max_contacts,
        "max_pin_error_m": example.max_pin_error,
        "max_speed_m_s": example.max_speed,
        "max_inverted_tetrahedra": max_tet_inversion,
        "elapsed_s": time.perf_counter() - start,
        "checks_passed": True,
        "samples": samples,
    }
    if args.demo == "shirt":
        metrics.update(
            max_self_intersections=example.max_self_intersections,
            max_edge_ratio=example.max_edge_ratio,
            max_edge_ratio_p99=example.max_edge_ratio_p99,
            validation="Non-adjacent triangle intersections checked at every 60 Hz frame, including the initial surface",
            total_mass_kg=float(example.model.particle_mass.numpy().sum()),
            substeps=example.sim_substeps,
            solver_iterations=example.solver.iterations,
            contact_radius_m=example.solver.particle_self_contact_margin,
            contact_margin_m=example.solver._self_contact.query_radius,
            stretch_stiffness=float(example.model.tri_materials.numpy()[0, 0]),
            bending_stiffness=float(example.model.edge_bending_properties.numpy()[0, 0]),
        )
    (args.out / (args.demo + "_metrics.json")).write_text(json.dumps(metrics, indent=2))
    # One contact sheet for checking the entire motion at a glance.
    thumbs = []
    for path in shots:
        im = Image.open(path)
        im.thumbnail((426, 240))
        thumbs.append(im.copy())
    rows = (len(thumbs) + 2) // 3
    sheet = Image.new("RGB", (1278, rows * 240), (13, 22, 33))
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % 3) * 426, (k // 3) * 240))
    sheet.save(args.out / (args.demo + "_storyboard.jpg"), quality=90)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
