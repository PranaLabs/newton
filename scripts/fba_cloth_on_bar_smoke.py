# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.0 baseline: cloth-on-horizontal-bar without self-contact.

A 1013-particle TRI_ARAP cloth (2×2 m, no pins) drops onto a static horizontal
cylinder bar.  Without self-contact, the two cloth halves on either side of
the bar will pass through each other below the bar — this is the reference
'broken physics' baseline we'll compare against once Phase 3 self-contact is
in.

Usage::

    uv run python scripts/fba_cloth_on_bar_smoke.py \
        --num-frames 300 \
        --render-out /tmp/cloth_on_bar_baseline.mp4
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
_DT = 0.01
_PD_ITER = 5
_NSN_ITER = 1
_GRAVITY = -10.0
_OBJ_MASS = 20.0
_YOUNG = 1.0e5
_POISSON = 0.4
# Bending tuned empirically below — anything ≥ 1.0 destabilises the
# implicit Euler solve at this drop height; 0.5 keeps cloth drape soft.
_BENDING = 0.5
_FRICTION_MU = 0.05
_PIN_STIFFNESS = 1.0e10

# Cloth: 3×3 m → larger flaps, drape > bar radius by a wide margin.
_CLOTH_SCALE = 3.0
# Drop from JUST above the bar — at this height the contact velocity is
# ~1.5 m/s, well within the soft_contact_margin (5 cm) per timestep,
# so no tunneling.  Higher drops cause the cloth to fall through the bar
# in a single dt step (per-step displacement > margin).
_CLOTH_INIT_Y = 0.35
# Small +X offset breaks the perfectly-balanced equilibrium that would
# otherwise let friction hold the cloth on top of the bar.
_CLOTH_INIT_XZ = (0.05, 0.0)

# Bar: thinner so the cloth wraps tighter and dangles further below.
_BAR_RADIUS = 0.2
_BAR_HALF_HEIGHT = 2.0
_BAR_CENTER = (0.0, 0.0, 0.0)

# Per-particle contact radius — defaults are too small for our cloth size,
# leading to tunneling at moderate velocities.  Match cloth_on_sphere's
# implicit radius (it uses Newton's default but the cloth there starts close
# to the obstacle so tunneling never fires).
_PARTICLE_RADIUS = 0.04
_CONTACT_MARGIN = 0.05  # back to the cloth_on_sphere default — larger margins
                        # generate too many spurious contacts in this scene
                        # (every cloth particle ends up registered against
                        # the bar's bounding box because cloth size ≫ margin).


def _load_obj(p):
    v, t = [], []
    with open(p) as f:
        for ln in f:
            t_ = ln.split()
            if not t_:
                continue
            if t_[0] == "v":
                v.append([float(x) for x in t_[1:4]])
            elif t_[0] == "f":
                t.append([int(x.split("/")[0]) - 1 for x in t_[1:4]])
    return np.array(v, np.float32), np.array(t, np.int32)


def _rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _lame(y, n):
    mu = y / (2 * (1 + n))
    lam = y * n / ((1 + n) * (1 - 2 * n))
    return mu, lam


def build_scene():
    verts_local, tris = _load_obj(_MESH_PATH)
    # Rx(90°) to lay the mesh flat on the XZ plane, then scale, then lift up.
    verts_world = ((verts_local * _CLOTH_SCALE) @ _rot_x(90.0).T).astype(np.float32)
    verts_world[:, 1] += _CLOTH_INIT_Y
    verts_world[:, 0] += _CLOTH_INIT_XZ[0]
    verts_world[:, 2] += _CLOTH_INIT_XZ[1]

    b = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=_GRAVITY)

    e1 = verts_world[tris[:, 1]] - verts_world[tris[:, 0]]
    e2 = verts_world[tris[:, 2]] - verts_world[tris[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    density = _OBJ_MASS / max(float(area.sum()), 1e-12)
    mu, _lam = _lame(_YOUNG, _POISSON)
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

    # Horizontal cylinder along world +Z.  Newton's add_shape_cylinder uses
    # local +Z as the cylinder axis, so no rotation needed — the bar already
    # lies along world +Z.
    cfg_collision = b.default_shape_cfg.copy()
    cfg_collision.mu = _FRICTION_MU
    cfg_collision.is_visible = False  # collision-only; see split visual below
    b.add_shape_cylinder(
        body=-1,
        xform=wp.transform(wp.vec3(*_BAR_CENTER), wp.quat_identity()),
        radius=_BAR_RADIUS,
        half_height=_BAR_HALF_HEIGHT,
        cfg=cfg_collision,
    )
    # Visual: slightly smaller cylinder (mirrors the cloth_on_sphere
    # visual-vs-collision split so the cloth visually rests on the rendered
    # surface).
    cfg_visual = b.default_site_cfg.copy()
    b.add_shape_cylinder(
        body=-1,
        xform=wp.transform(wp.vec3(*_BAR_CENTER), wp.quat_identity()),
        radius=_BAR_RADIUS * 0.97,
        half_height=_BAR_HALF_HEIGHT,
        cfg=cfg_visual,
        as_site=True,
        color=wp.vec3(0.55, 0.55, 0.65),
    )

    return b


def run(num_frames: int, render_out: Path | None, width: int = 1280, height: int = 720) -> int:
    wp.init()

    b = build_scene()
    model = b.finalize()
    uniform_mass = _OBJ_MASS / model.particle_count
    model.particle_mass.assign(np.full(model.particle_count, uniform_mass, np.float32))
    model.particle_inv_mass.assign(np.full(model.particle_count, 1.0 / uniform_mass, np.float32))
    mu_o = np.full(model.particle_count, _FRICTION_MU, np.float64)
    solver_kwargs = dict(
        iterations=_PD_ITER,
        nsn_iterations=_NSN_ITER,
        friction=True,
        stretching_model="arap",
        mu_per_pair_override=mu_o,
        pin_stiffness=_PIN_STIFFNESS,
        nsn_schur_mode="lite",
    )
    if _ENABLE_SELF_CONTACT:
        # 2× particle radius — empirically the smallest value where the
        # FB-unilateral activation kicks in early enough to overcome the
        # cloth halves' inward swing velocity within a single timestep.
        # Smaller (1×r) lets penetration build up between broadphase
        # registrations; larger (3×r) collides with the rest-pose exclusion
        # and ends up with zero candidate pairs.
        solver_kwargs["self_contact_radius"] = _PARTICLE_RADIUS * 2.0
        solver_kwargs["self_contact_margin"] = 0.04
        solver_kwargs["self_contact_friction"] = 0.25
        solver_kwargs["self_contact_topology_ring"] = 2
        solver_kwargs["self_contact_rest_exclusion_radius"] = 0.1
    solver = SolverFBA(model, **solver_kwargs)

    pipeline = newton.CollisionPipeline(model, soft_contact_margin=_CONTACT_MARGIN)
    contacts = pipeline.contacts()
    state_0 = model.state()
    state_1 = model.state()

    # ---- Viewer + render setup (headless) ----
    viewer = None
    frame_buf = None
    out_dir = None
    if render_out is not None:
        viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
        viewer.show_triangles = False
        viewer.set_model(model)
        viewer.set_world_offsets((0.0, 0.0, 0.0))
        cloth_tri = model.tri_indices.flatten()
        # Camera: front-right, framing the bar + cloth's eventual drape extent
        # (cloth halves can sag below the bar by ~1.0 m on each side, so
        # target y ≈ -0.3 with some elevation).
        if hasattr(viewer, "camera"):
            viewer.camera.pos = viewer.camera._as_vec3(wp.vec3(4.0, 2.0, 4.0))
            viewer.camera.look_at(wp.vec3(0.0, -0.3, 0.0))
        frame_buf = wp.empty(shape=(height, width, 3), dtype=wp.uint8, device=viewer.device)
        out_dir = render_out.with_suffix("")  # use as PNG dir basename
        out_dir = Path(str(out_dir) + "_frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("frame_*.png"):
            stale.unlink()

    step_ms = np.empty(num_frames, dtype=np.float32)
    sim_time = 0.0
    t_start = time.perf_counter()
    for f in range(num_frames):
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
            viewer.log_mesh(
                "/cloth", state_0.particle_q, cloth_tri,
                color=(0.78, 0.32, 0.36), backface_culling=False,
            )
            viewer.end_frame()
            viewer.get_frame(target_image=frame_buf)
            img = frame_buf.numpy()
            png = out_dir / f"frame_{f:06d}.png"
            try:
                import imageio.v2 as imageio  # noqa: PLC0415
                imageio.imwrite(png, img)
            except ImportError:
                from PIL import Image  # noqa: PLC0415
                Image.fromarray(img, mode="RGB").save(png)
        if f % 50 == 0 or f == num_frames - 1:
            q = state_0.particle_q.numpy()
            nc = int(contacts.soft_contact_count.numpy()[0]) if hasattr(contacts, "soft_contact_count") else 0
            print(f"  frame {f}: y range [{q[:,1].min():.3f}, {q[:,1].max():.3f}] "
                  f"n_contacts={nc} step_ms={step_ms[f]:.2f}")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone {num_frames} frames in {elapsed:.1f}s.  "
          f"step_mean={step_ms.mean():.2f} ms  "
          f"settled step_mean={step_ms[100:].mean():.2f} ms")

    if render_out is not None:
        print(f"Stitching MP4 -> {render_out}")
        try:
            import imageio.v2 as imageio  # noqa: PLC0415
            w = imageio.get_writer(str(render_out), fps=100, codec="libx264",
                                    quality=8, macro_block_size=1)
            for png in sorted(out_dir.glob("frame_*.png")):
                w.append_data(imageio.imread(str(png)))
            w.close()
        except ImportError:
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "100",
                "-i", str(out_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(render_out),
            ], check=True)
        print(f"Wrote {render_out} ({render_out.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-frames", type=int, default=300)
    p.add_argument("--render-out", type=Path, default=Path("/tmp/cloth_on_bar_baseline.mp4"))
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--self-contact", action="store_true",
                   help="Enable Phase 3 self-contact (BSR + sparse-PCR pass).")
    args = p.parse_args()
    global _ENABLE_SELF_CONTACT
    _ENABLE_SELF_CONTACT = args.self_contact
    return run(args.num_frames, None if args.no_render else args.render_out, args.width, args.height)


_ENABLE_SELF_CONTACT = False


if __name__ == "__main__":
    sys.exit(main())
