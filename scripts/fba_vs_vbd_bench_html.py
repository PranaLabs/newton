#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prana Lab
"""Generate HTML synthesis report for the FBA vs VBD cloth benchmark.

Reads npy trajectories and JSON metadata from --out-dir, writes three new
figures and a self-contained report.html into the same directory.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(out_dir: Path):
    x_fba = np.load(out_dir / "x_FBA_ref.npy")   # (800, 1089, 3)
    x_vbd = np.load(out_dir / "x_VBD_ref.npy")   # (800, 1089, 3)

    with open(out_dir / "anchor.json") as f:
        anchor = json.load(f)
    with open(out_dir / "calibration.json") as f:
        calibration = json.load(f)
    with open(out_dir / "vbd_ref.json") as f:
        vbd_ref = json.load(f)
    with open(out_dir / "results.json") as f:
        results = json.load(f)

    return x_fba, x_vbd, anchor, calibration, vbd_ref, results


def shared_axis_limits(xy_a, xy_b, pad_frac=0.05):
    """Return (xmin, xmax, ymin, ymax) covering both point clouds with padding."""
    x_all = np.concatenate([xy_a[:, 0], xy_b[:, 0]])
    y_all = np.concatenate([xy_a[:, 1], xy_b[:, 1]])
    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()
    xpad = (xmax - xmin) * pad_frac
    ypad = (ymax - ymin) * pad_frac
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


# ---------------------------------------------------------------------------
# Figure A: converged_shapes_side_by_side.png
# ---------------------------------------------------------------------------

def fig_side_by_side(out_dir: Path, x_fba, x_vbd, calibration):
    alpha_star = calibration["alpha_star"]
    beta_star = calibration["beta_star"]

    xy_fba = x_fba[-1, :, 0:2]  # (1089, 2)
    xy_vbd = x_vbd[-1, :, 0:2]

    xmin, xmax, ymin, ymax = shared_axis_limits(xy_fba, xy_vbd)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    ax.scatter(xy_fba[:, 0], xy_fba[:, 1], s=6, color="steelblue")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("FBA  (iter=200)")

    ax = axes[1]
    ax.scatter(xy_vbd[:, 0], xy_vbd[:, 1], s=6, color="darkorange")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"VBD  (substeps=40, iter=40 at α*={alpha_star:.2f}, β*={beta_star:.2f})"
    )

    fig.tight_layout()
    out_path = out_dir / "converged_shapes_side_by_side.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure B: converged_shapes_overlay.png
# ---------------------------------------------------------------------------

def fig_overlay(out_dir: Path, x_fba, x_vbd):
    xy_fba = x_fba[-1, :, 0:2]
    xy_vbd = x_vbd[-1, :, 0:2]

    xmin, xmax, ymin, ymax = shared_axis_limits(xy_fba, xy_vbd)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xy_fba[:, 0], xy_fba[:, 1], s=8, color="steelblue", alpha=0.55,
               label="FBA iter=200")
    ax.scatter(xy_vbd[:, 0], xy_vbd[:, 1], s=8, color="darkorange", alpha=0.55,
               label="VBD substeps=40, iter=40")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Terminal-frame positions, both solvers overlaid")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = out_dir / "converged_shapes_overlay.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure C: converged_diff_heatmap.png
# ---------------------------------------------------------------------------

def fig_diff_heatmap(out_dir: Path, x_fba, x_vbd):
    diff = x_fba[-1] - x_vbd[-1]          # (1089, 3)
    d = np.linalg.norm(diff, axis=-1)      # (1089,)

    xy = x_fba[-1, :, 0:2]

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=d, cmap="magma", s=10)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("‖x_FBA − x_VBD‖ (m)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Per-vertex distance between FBA and VBD terminal frames\n"
        f"(max = {d.max():.3e} m, mean = {d.mean():.3e} m)"
    )

    fig.tight_layout()
    out_path = out_dir / "converged_diff_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    return float(d.max()), float(d.mean())


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def fmt_num(v, sig=4):
    """Format a float as scientific notation with sig significant figures."""
    return f"{v:.{sig}e}"


def table_rows(rows, header):
    cells = "".join(f"<th>{h}</th>" for h in header)
    html = f"<thead><tr>{cells}</tr></thead>\n<tbody>\n"
    for i, row in enumerate(rows):
        cls = ' class="even"' if i % 2 == 0 else ""
        tds = "".join(f"<td>{c}</td>" for c in row)
        html += f"<tr{cls}>{tds}</tr>\n"
    html += "</tbody>"
    return f"<table>\n{html}\n</table>"


def img_fig(src, caption):
    return (
        f'<figure>\n'
        f'  <img src="{src}" alt="{caption}">\n'
        f'  <figcaption>{caption}</figcaption>\n'
        f'</figure>\n'
    )


def side_by_side(*figs):
    inner = "\n".join(figs)
    return f'<div class="side-by-side">\n{inner}\n</div>\n'


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------

CSS = """\
body {
  max-width: 960px;
  margin: auto;
  padding: 1.5em;
  font-family: system-ui, -apple-system, sans-serif;
  line-height: 1.55;
  color: #222;
}
h1, h2, h3, h4 {
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.2em;
}
h1 { font-size: 1.7em; }
h2 { font-size: 1.35em; margin-top: 2em; }
h3 { font-size: 1.1em; margin-top: 1.5em; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
  font-size: 0.9em;
}
th {
  background: #f0f0f0;
  text-align: left;
  padding: 0.45em 0.7em;
  border-bottom: 2px solid #bbb;
}
td {
  padding: 0.35em 0.7em;
  font-variant-numeric: tabular-nums;
  border-bottom: 1px solid #e4e4e4;
}
tr.even td { background: #f8f8f8; }
figure { margin: 1em 0; }
figcaption { font-style: italic; text-align: center; font-size: 0.88em; color: #555; margin-top: 0.3em; }
.side-by-side { display: flex; gap: 1em; flex-wrap: wrap; }
.side-by-side > figure { flex: 1; min-width: 320px; }
img { max-width: 100%; height: auto; }
pre {
  background: #f5f5f5;
  border: 1px solid #ddd;
  padding: 1em;
  overflow-x: auto;
  font-size: 0.82em;
  border-radius: 3px;
}
a { color: #0057a8; }
"""


def build_html(out_dir: Path, anchor, calibration, vbd_ref, results,
               diff_max, diff_mean):
    import datetime
    today = datetime.date.today().isoformat()

    alpha_star = calibration["alpha_star"]
    beta_star = calibration["beta_star"]
    max_sag = calibration["max_sag"]
    floor_rms = calibration["floor_rms"]

    sweep = results["sweep"]

    # --- Section 1 table: anchor parameters ---
    anchor_rows = [
        ("n_frames", anchor["n_frames"]),
        ("iterations (FBA anchor)", anchor["iterations"]),
        ("mu", f"{anchor['mu']:.0f}"),
        ("lam", f"{anchor['lam']:.0f}"),
        ("edge_ke", f"{anchor['edge_ke']:.1f}"),
        ("frame_dt (s)", f"{anchor['frame_dt']:.4f}"),
        ("max_sag (m)", f"{max_sag:.5f}"),
        ("mean ms/frame (anchor)", f"{anchor['mean_ms_per_frame']:.3f}"),
    ]
    anchor_table = table_rows(anchor_rows, ["Parameter", "Value"])

    # --- Section 2 calibration table ---
    cal_rows = [
        ("α* (tri_ke scale)", f"{alpha_star:.4f}"),
        ("β* (edge_ke scale)", f"{beta_star:.4f}"),
        ("2D search triggered", str(calibration["upgraded_to_2d"])),
        ("Candidate substeps", calibration["candidate_substeps"]),
        ("Candidate iterations", calibration["candidate_iterations"]),
        ("Verify substeps", calibration["verify_substeps"]),
        ("Verify iterations", calibration["verify_iterations"]),
        ("Floor RMS (high-fid VBD vs FBA anchor, m)", f"{floor_rms:.6e}"),
        ("Floor threshold (frac of max_sag)", f"{calibration['floor_threshold_frac']:.4f}"),
        ("max_sag (m)", f"{max_sag:.6f}"),
    ]
    cal_table = table_rows(cal_rows, ["Parameter", "Value"])

    # --- Section 3 self-conv table ---
    self_rows = []
    for e in sweep:
        self_rows.append((
            e["label"],
            f"{e['mean_ms']:.3f}",
            f"{e['terminal_rel_err']:.4e}",
        ))
    self_table = table_rows(self_rows,
                            ["Config", "mean ms/frame", "terminal rel_err"])

    # --- Appendix sweep table (vs FBA ref) ---
    sweep_rows = []
    for e in sweep:
        sweep_rows.append((
            e["label"],
            f"{e['mean_ms']:.3f}",
            f"{e['p50_ms']:.3f}",
            f"{e['p95_ms']:.3f}",
            f"{e['terminal_rms']:.4e}",
            f"{e['terminal_rel_err']:.4e}",
        ))
    sweep_table = table_rows(
        sweep_rows,
        ["Config", "mean ms/frame", "p50 ms", "p95 ms",
         "terminal RMS vs FBA ref (m)", "rel_err"],
    )

    # --- VBD ref timing ---
    vbd_ref_rows = [
        ("substeps", vbd_ref["substeps"]),
        ("iterations", vbd_ref["iterations"]),
        ("α*", f"{vbd_ref['alpha_star']:.4f}"),
        ("β*", f"{vbd_ref['beta_star']:.4f}"),
        ("n_frames", vbd_ref["n_frames"]),
        ("elapsed_s", f"{vbd_ref['elapsed_s']:.2f}"),
        ("mean ms/frame", f"{vbd_ref['mean_ms_per_frame']:.3f}"),
    ]
    vbd_ref_table = table_rows(vbd_ref_rows, ["Parameter", "Value"])

    # --- Raw JSON appendix ---
    anchor_json_str = json.dumps(anchor, indent=2)
    vbd_ref_json_str = json.dumps(vbd_ref, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FBA vs VBD — 32×32 Hanging Cloth Benchmark</title>
<style>
{CSS}
</style>
</head>
<body>

<h1>FBA vs VBD — 32×32 Hanging Cloth Benchmark</h1>
<p><em>Generated {today}</em></p>

<h2>1. Setup</h2>

<p>Scene: 32×32 hanging cloth, cell size 0.05 m (rest configuration 1.6×1.6 m),
Y-up gravity, two top corners pinned (particle_mass = 0), all other particles
mass 0.1 kg. Constitutive model: Stable Neo-Hookean (Smith 2018),
μ = λ = 10 000 Pa. Edge springs: edge_ke = 1.0. Time integration: implicit
Euler, dt = 1/100 s, 800 frames (8 s total).</p>

{anchor_table}

<h2>2. Method</h2>

<h3>Step 1 — FBA anchor</h3>
<p>FBA is run with iter = 200 for 800 frames, producing the trajectory
<code>x_FBA_ref</code> (800 frames × 1 089 particles × 3).</p>

<h3>Step 2 — VBD calibration</h3>
<p>A 1-D grid search over α (scaling tri_ke in VBD) minimises the RMS distance
between VBD's terminal shape at moderate budget and <code>x_FBA_ref[-1]</code>.
If the 1-D minimum exceeds 0.5 % × max_sag, the search is extended to a 2-D
grid over (α, β) where β scales edge_ke.</p>

{cal_table}

{img_fig("calibration.png",
         "Calibration loss surface. Colour encodes terminal RMS (m) between "
         "VBD at candidate budget and x_FBA_ref[-1]. Star marks (α*, β*).")}

<h3>Step 3 — VBD self-reference</h3>
<p>VBD is run at calibrated (α*, β*) with substeps = 40, iter = 40 for 800
frames → <code>x_VBD_ref</code>. Timing below.</p>

{vbd_ref_table}

<h3>Step 4 — Sweep</h3>
<p>Four FBA configs (iter ∈ {5, 10, 20, 40}) and four VBD configs
((substeps, iter) ∈ {(2,10),(5,10),(10,10),(10,20)}) are each run at
calibrated (α*, β*) for 800 frames. Per-frame wall-clock time and per-frame
RMS error against both references are recorded.</p>

<h2>3. Convergence rate</h2>

<p>Each solver is compared against its own high-iteration reference: FBA sweep
configs versus <code>x_FBA_ref</code> (iter=200); VBD sweep configs versus
<code>x_VBD_ref</code> (substeps=40, iter=40 at α*={alpha_star:.2f},
β*={beta_star:.2f}). Relative error = per-frame RMS divided by max_sag of the
FBA anchor ({max_sag:.4f} m).</p>

{self_table}

{img_fig("pareto_self.png",
         "Pareto plot: mean ms/frame vs terminal relative error (vs own reference). "
         "FBA configs in blue; VBD configs in orange.")}

{img_fig("rms_over_time_self.png",
         "Relative RMS error over 800 frames, each config vs its own high-iteration reference.")}

<h3>Spatial residual structure</h3>
<p>Per-vertex terminal-frame residual, each solver vs its own converged reference
(best-budget sweep config for each solver).</p>

{side_by_side(
    img_fig("error_heatmap_fba_self.png",
            "Per-vertex |x_FBA_best − x_FBA_ref| at terminal frame (magma colormap)."),
    img_fig("error_heatmap_vbd_self.png",
            "Per-vertex |x_VBD_best − x_VBD_ref| at terminal frame (magma colormap)."),
)}

<h2>4. Converged shapes</h2>

<p>Terminal-frame particle positions (x–y plane) at each solver's high-iteration
reference run.</p>

{img_fig("converged_shapes_side_by_side.png",
         "Left: FBA terminal frame (iter=200). "
         f"Right: VBD terminal frame (substeps=40, iter=40, α*={alpha_star:.2f}, β*={beta_star:.2f}). "
         "Identical axis limits across both panels.")}

{img_fig("converged_shapes_overlay.png",
         "FBA (blue) and VBD (orange) terminal-frame positions overlaid on the same axes.")}

<h3>Per-vertex distance between converged solutions</h3>

{img_fig("converged_diff_heatmap.png",
         f"Per-vertex Euclidean distance ‖x_FBA_ref[-1] − x_VBD_ref[-1]‖ "
         f"(max = {diff_max:.3e} m, mean = {diff_mean:.3e} m). "
         "Positions from FBA terminal frame.")}

<p>RMS distance between FBA and VBD terminal frames: {floor_rms:.4f} m.
Max sag of FBA anchor: {max_sag:.4f} m.</p>

<h2>Appendix — Sweep against shared FBA reference</h2>

<p>For completeness, every sweep config is also compared against the FBA
reference (<code>x_FBA_ref</code>), i.e. not the self-convergence view from
Section 3.</p>

{sweep_table}

{img_fig("pareto.png",
         "Pareto plot: mean ms/frame vs terminal RMS against x_FBA_ref. "
         "FBA configs in blue; VBD configs in orange.")}

{img_fig("rms_over_time.png",
         "RMS error over 800 frames, each sweep config vs x_FBA_ref.")}

{side_by_side(
    img_fig("error_heatmap_fba.png",
            "Per-vertex |x_FBA_best − x_FBA_ref| at terminal frame."),
    img_fig("error_heatmap_vbd.png",
            "Per-vertex |x_VBD_best − x_FBA_ref| at terminal frame."),
)}

<h3>Visual replay</h3>

<p>
  <a href="three_up.mp4">three_up.mp4</a> — side-by-side video:
  FBA-best | VBD-best | reference (FBA iter=200) over 8 s. 50 fps × 800 frames.
</p>

<h3>Anchor and VBD-ref metadata (raw JSON)</h3>

<h4>anchor.json</h4>
<pre>{anchor_json_str}</pre>

<h4>vbd_ref.json</h4>
<pre>{vbd_ref_json_str}</pre>

</body>
</html>
"""
    out_path = out_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate figures and HTML report for FBA vs VBD benchmark."
    )
    parser.add_argument(
        "--out-dir",
        default="scripts/fba_vs_vbd_bench_out",
        help="Directory containing npy+json inputs; figures and HTML are written here.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"Output directory not found: {out_dir}")

    x_fba, x_vbd, anchor, calibration, vbd_ref, results = load_data(out_dir)

    fig_side_by_side(out_dir, x_fba, x_vbd, calibration)
    fig_overlay(out_dir, x_fba, x_vbd)
    diff_max, diff_mean = fig_diff_heatmap(out_dir, x_fba, x_vbd)

    build_html(out_dir, anchor, calibration, vbd_ref, results, diff_max, diff_mean)


if __name__ == "__main__":
    main()
