# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build a standalone HTML report covering only Section 3 (Convergence rate)
of the FBA vs VBD bench, with PNGs embedded as base64 data URIs so the result
is a single shareable file.

Consumes existing outputs under ``scripts/fba_vs_vbd_bench_out/`` (the four
self-convergence PNGs). Re-run the bench script first if those are missing.

Run::

    uv run python scripts/fba_vs_vbd_bench_section3_html.py
"""

from __future__ import annotations

import base64
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "fba_vs_vbd_bench_out"
DST = OUT_DIR / "report_section3.html"


def b64(png_name: str) -> str:
    data = (OUT_DIR / png_name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


pareto_self_uri = b64("pareto_self.png")
rms_self_uri = b64("rms_over_time_self.png")
hm_fba_uri = b64("error_heatmap_fba_self.png")
hm_vbd_uri = b64("error_heatmap_vbd_self.png")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FBA vs VBD — Convergence Rate</title>
<style>
body {{
  max-width: 960px;
  margin: auto;
  padding: 1.5em;
  font-family: system-ui, -apple-system, sans-serif;
  line-height: 1.55;
  color: #222;
}}
h1, h2, h3 {{
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.2em;
}}
h1 {{ font-size: 1.55em; }}
h3 {{ font-size: 1.1em; margin-top: 1.5em; }}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
  font-size: 0.9em;
}}
th {{
  background: #f0f0f0;
  text-align: left;
  padding: 0.45em 0.7em;
  border-bottom: 2px solid #bbb;
}}
td {{
  padding: 0.35em 0.7em;
  font-variant-numeric: tabular-nums;
  border-bottom: 1px solid #e4e4e4;
}}
tr.even td {{ background: #f8f8f8; }}
figure {{ margin: 1em 0; }}
figcaption {{ font-style: italic; text-align: center; font-size: 0.88em; color: #555; margin-top: 0.3em; }}
.side-by-side {{ display: flex; gap: 1em; flex-wrap: wrap; }}
.side-by-side > figure {{ flex: 1; min-width: 320px; }}
img {{ max-width: 100%; height: auto; }}
code {{ background: #f3f3f3; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.92em; }}
</style>
</head>
<body>

<h1>FBA vs VBD — Convergence rate (32×32 hanging cloth)</h1>

<p>Each solver is compared against its own high-iteration reference: FBA sweep
configs versus <code>x_FBA_ref</code> (iter=200); VBD sweep configs versus
<code>x_VBD_ref</code> (substeps=40, iter=40 at α*=0.50,
β*=2.00). Relative error = per-frame RMS divided by max_sag of the
FBA anchor (0.3505 m).</p>

<table>
<thead><tr><th>Config</th><th>mean ms/frame</th><th>terminal rel_err</th></tr></thead>
<tbody>
<tr class="even"><td>FBA iter=5</td><td>3.419</td><td>1.8714e-04</td></tr>
<tr><td>FBA iter=10</td><td>6.835</td><td>8.8995e-06</td></tr>
<tr class="even"><td>FBA iter=20</td><td>13.557</td><td>2.0509e-06</td></tr>
<tr><td>FBA iter=40</td><td>27.021</td><td>2.4476e-06</td></tr>
<tr class="even"><td>VBD 2x10</td><td>2.440</td><td>3.7428e-01</td></tr>
<tr><td>VBD 5x10</td><td>6.057</td><td>4.3466e-02</td></tr>
<tr class="even"><td>VBD 10x10</td><td>12.005</td><td>1.0468e-02</td></tr>
<tr><td>VBD 10x20</td><td>23.306</td><td>1.0306e-02</td></tr>
</tbody>
</table>

<figure>
  <img src="{pareto_self_uri}" alt="Pareto plot: mean ms/frame vs terminal relative error (vs own reference). FBA configs in blue; VBD configs in orange.">
  <figcaption>Pareto plot: mean ms/frame vs terminal relative error (vs own reference). FBA configs in blue; VBD configs in orange.</figcaption>
</figure>

<figure>
  <img src="{rms_self_uri}" alt="Relative RMS error over 800 frames, each config vs its own high-iteration reference.">
  <figcaption>Relative RMS error over 800 frames, each config vs its own high-iteration reference.</figcaption>
</figure>

<h3>Spatial residual structure</h3>

<p>Per-vertex terminal-frame residual, each solver vs its own converged reference
(best-budget sweep config for each solver).</p>

<div class="side-by-side">
<figure>
  <img src="{hm_fba_uri}" alt="Per-vertex |x_FBA_best − x_FBA_ref| at terminal frame (magma colormap).">
  <figcaption>Per-vertex |x_FBA_best − x_FBA_ref| at terminal frame (magma colormap).</figcaption>
</figure>

<figure>
  <img src="{hm_vbd_uri}" alt="Per-vertex |x_VBD_best − x_VBD_ref| at terminal frame (magma colormap).">
  <figcaption>Per-vertex |x_VBD_best − x_VBD_ref| at terminal frame (magma colormap).</figcaption>
</figure>
</div>

</body>
</html>
"""

DST.write_text(HTML)
print(f"Wrote {DST} ({DST.stat().st_size / 1024:.1f} KB)")
