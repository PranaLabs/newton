# CudaTests Twisting Bar 4-Way Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the CudaTests twisting bar comparison to a full 4-way visual+performance table: Newton ARAP, Newton Corot, Newton NH (already done), and RealSim NH rendered from .abc.

**Architecture:** All changes live in a single script `scripts/fba_twisting_bar_cudatests.py`. The script gains a new `--energy all` mode that runs ARAP and Corot in sequence (NH already run), plus a new `--realsim` mode that extracts RealSim NH positions from .abc via the `/tmp/dump_abc_traj` binary and renders them with the same `render_frame()`. A `--montage` mode assembles the 4-column PNG grid. `--perf-update` rewrites `perf_summary.txt` with all four rows. A top-level `--all` flag chains everything.

**Tech Stack:** Python / NumPy / Matplotlib / Warp / Newton SolverFBA, Alembic `/tmp/dump_abc_traj` binary for .abc extraction.

---

## Pre-flight checks

- [ ] Confirm `/tmp/dump_abc_traj` binary exists and is executable

```bash
/tmp/dump_abc_traj /home/ziqiu/work/RealSim_py/realsim_py/simulation/output_abc/TwistingBarNH/output_obj_0.abc 2>&1 | head -3
# Expected: "810\n11340\n-1 -2 -1"
```

- [ ] Confirm existing NH frames are present

```bash
ls /home/ziqiu/work/newton/scripts/twisting_bar_out_cudatests/neohookean/frame_*.png | wc -l
# Expected: 8  (old snapshot set; we will re-render for the 10-frame set below)
```

---

## Shared constants to align across all tasks

Snapshot frames **for this comparison** (10 frames, matching RealSim ABC):

```python
SNAPSHOT_FRAMES = [0, 8, 16, 24, 50, 100, 200, 400, 600, 809]
```

Energy → sub-directory mapping:

```
arap        → scripts/twisting_bar_out_cudatests/arap/
corotational → scripts/twisting_bar_out_cudatests/corotational/
neohookean  → scripts/twisting_bar_out_cudatests/neohookean/   (re-render)
realsim     → scripts/twisting_bar_out_cudatests/realsim/
```

Montage output: `scripts/twisting_bar_out_cudatests/summary.png`
Perf summary: `scripts/twisting_bar_out_cudatests/perf_summary.txt`

---

## Task 1: Update SNAPSHOT_FRAMES and run Newton ARAP + Corot

**Files:**
- Modify: `scripts/fba_twisting_bar_cudatests.py`

The existing script already has `--energy {neohookean,arap,corotational}`. We only need to:
1. Change `SNAPSHOT_FRAMES` to the 10-frame set.
2. Re-render NH with new frames (run `--energy neohookean` again, it overwrites).
3. Run ARAP and Corot.

- [ ] **Step 1: Update SNAPSHOT_FRAMES in the script**

In `scripts/fba_twisting_bar_cudatests.py`, find:

```python
SNAPSHOT_FRAMES = [0, 8, 16, 50, 100, 400, 800, 809]
```

Replace with:

```python
SNAPSHOT_FRAMES = [0, 8, 16, 24, 50, 100, 200, 400, 600, 809]
```

- [ ] **Step 2: Re-run Newton NH to regenerate frames for new snapshot set**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_twisting_bar_cudatests.py --energy neohookean
```

Expected terminal output (last lines):
```
  [neohookean] Done. mean=7.xx ms  median=7.xx ms  p95=8.xx ms  stable
=== Total wall time: NNs (N.N min) ===
```

Check new frames exist:
```bash
ls scripts/twisting_bar_out_cudatests/neohookean/frame_*.png
# Should include frame_0024.png, frame_0200.png, frame_0600.png
```

- [ ] **Step 3: Run Newton ARAP**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_twisting_bar_cudatests.py --energy arap
```

Expected terminal (last lines):
```
  [arap] Done. mean=X.XX ms  median=X.XX ms  p95=X.XX ms  stable
```

Check output:
```bash
ls scripts/twisting_bar_out_cudatests/arap/frame_*.png | wc -l
# Expected: 10
```

- [ ] **Step 4: Run Newton Corot**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_twisting_bar_cudatests.py --energy corotational
```

Check output:
```bash
ls scripts/twisting_bar_out_cudatests/corotational/frame_*.png | wc -l
# Expected: 10
```

---

## Task 2: Add RealSim NH ABC rendering

**Files:**
- Modify: `scripts/fba_twisting_bar_cudatests.py` — add `render_realsim_nh()` function and `--realsim` CLI flag

The `/tmp/dump_abc_traj` binary streams all 810 frames as text to stdout:

```
810           <- numSamples
11340         <- nVerts for frame 0
x0 y0 z0
x1 y1 z1
...           <- 11340 lines
11340         <- nVerts for frame 1
...
```

We only need frames in `SNAPSHOT_FRAMES`. To avoid holding 810 × 11340 × 3 floats in RAM simultaneously, we parse frame-by-frame.

- [ ] **Step 1: Add `render_realsim_nh()` to the script**

Add this function after `render_frame()` (around line 224 of the original):

```python
# ---------------------------------------------------------------------------
# 5b. Extract and render RealSim NH .abc trajectory
# ---------------------------------------------------------------------------
ABC_PATH = Path(
    "/home/ziqiu/work/RealSim_py/realsim_py/simulation/output_abc"
    "/TwistingBarNH/output_obj_0.abc"
)
DUMP_ABC_BIN = Path("/tmp/dump_abc_traj")


def render_realsim_nh(out_subdir: Path) -> None:
    """Stream RealSim NH .abc via dump_abc_traj, render SNAPSHOT_FRAMES."""
    import subprocess

    snapshot_set = set(SNAPSHOT_FRAMES)
    out_subdir.mkdir(parents=True, exist_ok=True)

    print(f"  [RealSim NH] Streaming {ABC_PATH.name} via {DUMP_ABC_BIN} ...", flush=True)
    proc = subprocess.Popen(
        [str(DUMP_ABC_BIN), str(ABC_PATH)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None

    num_samples = int(proc.stdout.readline().strip())
    print(f"  [RealSim NH] {num_samples} frames in .abc", flush=True)

    for f in range(num_samples):
        n_verts = int(proc.stdout.readline().strip())
        verts = np.empty((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            parts = proc.stdout.readline().split()
            verts[i, 0] = float(parts[0])
            verts[i, 1] = float(parts[1])
            verts[i, 2] = float(parts[2])

        if f in snapshot_set:
            img_path = out_subdir / f"frame_{f:04d}.png"
            render_frame(verts, f, img_path, "RealSim NH")
            print(f"  [RealSim NH] frame {f} rendered -> {img_path.name}", flush=True)

        if f % 100 == 0 and f > 0:
            print(f"  [RealSim NH] parsed frame {f}/{num_samples}", flush=True)

    proc.wait()
    if proc.returncode != 0:
        print(f"  [RealSim NH] WARNING: dump_abc_traj exited with code {proc.returncode}", flush=True)
    print("  [RealSim NH] Done.", flush=True)
```

- [ ] **Step 2: Add `--realsim` flag and wire it in `main()`**

In `main()`, after the existing `--energy` argparse argument, add:

```python
parser.add_argument(
    "--realsim",
    action="store_true",
    help="Render RealSim NH .abc trajectory to PNGs (requires /tmp/dump_abc_traj)",
)
```

After the Newton simulation block (after the perf table write), add:

```python
if args.realsim:
    print("\n=== Rendering RealSim NH .abc trajectory ===")
    realsim_out = OUT_DIR / "realsim"
    render_realsim_nh(realsim_out)
```

- [ ] **Step 3: Run the RealSim render**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_twisting_bar_cudatests.py --energy neohookean --realsim
# (--energy neohookean is still required by argparse; it will run NH again
#  but that's fine — NH is fast and ensures perf stats are fresh)
```

Expected output includes:
```
=== Rendering RealSim NH .abc trajectory ===
  [RealSim NH] Streaming output_obj_0.abc via /tmp/dump_abc_traj ...
  [RealSim NH] 810 frames in .abc
  [RealSim NH] frame 0 rendered -> frame_0000.png
  ...
  [RealSim NH] Done.
```

Check:
```bash
ls scripts/twisting_bar_out_cudatests/realsim/frame_*.png | wc -l
# Expected: 10
```

---

## Task 3: Build 4-column summary montage

**Files:**
- Modify: `scripts/fba_twisting_bar_cudatests.py` — add `build_montage()` and `--montage` flag

Layout: 10 rows × 4 columns. Column order: ARAP | Corot | NH | RealSim NH. Each cell is one 5×6 inch PNG (400×480 px at dpi=80). Montage uses `matplotlib.image.imread` to avoid PIL dependency.

- [ ] **Step 1: Add `build_montage()` to the script**

Add after `render_realsim_nh()`:

```python
# ---------------------------------------------------------------------------
# 6. Build 4-column summary montage
# ---------------------------------------------------------------------------
MONTAGE_COLS = [
    ("arap",         "Newton ARAP"),
    ("corotational", "Newton Corot"),
    ("neohookean",   "Newton NH"),
    ("realsim",      "RealSim NH"),
]


def build_montage() -> Path:
    """Assemble 10-row × 4-col montage from per-energy frame PNGs."""
    rows = SNAPSHOT_FRAMES
    ncols = len(MONTAGE_COLS)
    nrows = len(rows)

    # Load all images first to get shape
    cells: dict[tuple[int, int], np.ndarray] = {}
    for c, (energy_dir, _) in enumerate(MONTAGE_COLS):
        for r, frame_idx in enumerate(rows):
            p = OUT_DIR / energy_dir / f"frame_{frame_idx:04d}.png"
            if p.exists():
                cells[(r, c)] = plt.imread(str(p))
            else:
                print(f"  [montage] WARNING: missing {p}", flush=True)

    # Infer cell size from first available image
    sample = next(iter(cells.values()))
    h, w = sample.shape[:2]

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * w / 80, nrows * h / 80),
        dpi=80,
    )
    # If only 1 row, axes is 1-D; normalise to 2-D
    if nrows == 1:
        axes = axes[np.newaxis, :]

    col_labels = [label for _, label in MONTAGE_COLS]
    for c, col_label in enumerate(col_labels):
        axes[0, c].set_title(col_label, fontsize=8, fontweight="bold")

    for r, frame_idx in enumerate(rows):
        axes[r, 0].set_ylabel(f"f={frame_idx}", fontsize=7, rotation=0, labelpad=28)
        for c in range(ncols):
            ax = axes[r, c]
            if (r, c) in cells:
                ax.imshow(cells[(r, c)])
            else:
                ax.set_facecolor("lightgrey")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    fig.tight_layout(pad=0.3)
    out_path = OUT_DIR / "summary.png"
    plt.savefig(str(out_path), dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  [montage] Saved {out_path}", flush=True)
    return out_path
```

- [ ] **Step 2: Add `--montage` flag and wire in `main()`**

In the argparse block:

```python
parser.add_argument(
    "--montage",
    action="store_true",
    help="Build 4-column summary montage from existing per-energy frames",
)
```

After the `--realsim` block in `main()`:

```python
if args.montage:
    print("\n=== Building 4-column montage ===")
    montage_path = build_montage()
    print(f"  Montage: {montage_path}")
```

- [ ] **Step 3: Generate the montage**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_twisting_bar_cudatests.py --energy neohookean --montage
```

Check output:
```bash
ls -lh scripts/twisting_bar_out_cudatests/summary.png
# Should be > 1 MB
```

Open visually (optional):
```bash
xdg-open scripts/twisting_bar_out_cudatests/summary.png 2>/dev/null || true
```

---

## Task 4: Full perf_summary.txt with all 4 rows

**Files:**
- Modify: `scripts/fba_twisting_bar_cudatests.py` — update `write_perf_summary()` logic

Currently `main()` writes the perf file only after a single energy run. We need it to accumulate results across energies and always include the fixed RealSim NH numbers.

Strategy: after each Newton energy run, the script appends stats to a JSON sidecar (`perf_summary.json`). The `--perf-update` flag reads that JSON plus hardcoded RealSim NH numbers and rewrites the human-readable `.txt`.

- [ ] **Step 1: Add JSON accumulation helper**

Add this function near the top of the script (after imports):

```python
import json

PERF_JSON = OUT_DIR / "perf_stats.json"

# Hardcoded RealSim NH stats from the completed run
REALSIM_NH_STATS = {
    "frames": 810,
    "stable": True,
    "mean_ms": 50.23,
    "median_ms": 50.68,
    "p95_ms": 55.92,
}


def load_perf_json() -> dict:
    if PERF_JSON.exists():
        with open(str(PERF_JSON)) as fh:
            return json.load(fh)
    return {}


def save_perf_json(energy: str, stats: dict) -> None:
    data = load_perf_json()
    data[energy] = stats
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(PERF_JSON), "w") as fh:
        json.dump(data, fp=fh, indent=2)
```

- [ ] **Step 2: Call `save_perf_json` after each Newton run**

In `main()`, after `run_newton_energy()` returns `stats`:

```python
save_perf_json(args.energy, stats)
```

- [ ] **Step 3: Add `write_full_perf_summary()` and `--perf-update` flag**

Add this function:

```python
def write_full_perf_summary(device: str) -> Path:
    """Write combined Newton + RealSim perf table to perf_summary.txt."""
    data = load_perf_json()

    energies_newton = [
        ("arap",         "Newton ARAP"),
        ("corotational", "Newton Corot"),
        ("neohookean",   "Newton NH"),
    ]

    header_note = (
        "Newton FBA CudaTests params: E=1e9, nu=0.45, 810 frames, dt=0.01, PD_iter=5\n"
        "Mesh: cube_volume_11340P.mesh (11340 verts, ~58956 tets)\n"
        f"Device: {device}\n\n"
    )

    col_w = 28
    num_w = 10

    header = (
        f"{'':>{col_w}}"
        f"{'setup(ms)':>{num_w}}"
        f"{'mean(ms)':>{num_w}}"
        f"{'median(ms)':>{num_w}}"
        f"{'p95(ms)':>{num_w}}"
        f"{'frames':>{num_w}}"
        f"{'stable':>16}"
    )
    sep = "-" * len(header)

    lines = [header_note, header, sep]

    for key, label in energies_newton:
        s = data.get(key, {})
        if s:
            nan_f = s.get("nan_frame")
            stable_str = f"NaN@{nan_f}" if nan_f is not None else "YES"
            row = (
                f"  {label:<{col_w - 2}}"
                f"{s.get('setup_ms', 0):>{num_w}.1f}"
                f"{s.get('mean_ms', 0):>{num_w}.2f}"
                f"{s.get('median_ms', 0):>{num_w}.2f}"
                f"{s.get('p95_ms', 0):>{num_w}.2f}"
                f"{s.get('n_steps', 0):>{num_w}}"
                f"{stable_str:>16}"
            )
        else:
            row = f"  {label:<{col_w - 2}}{'(not yet run)':>{num_w}}"
        lines.append(row)

    lines.append(sep)

    # RealSim NH row (no setup time reported)
    rs = REALSIM_NH_STATS
    stable_rs = "YES" if rs["stable"] else "NO"
    rs_row = (
        f"  {'RealSim NH':<{col_w - 2}}"
        f"{'N/A':>{num_w}}"
        f"{rs['mean_ms']:>{num_w}.2f}"
        f"{rs['median_ms']:>{num_w}.2f}"
        f"{rs['p95_ms']:>{num_w}.2f}"
        f"{rs['frames']:>{num_w}}"
        f"{stable_rs:>16}"
    )
    lines.append(rs_row)
    lines.append(sep)

    # Speedup rows vs RealSim
    lines.append("")
    lines.append("Speedup (Newton / RealSim NH median):")
    for key, label in energies_newton:
        s = data.get(key, {})
        if s and s.get("median_ms", 0) > 0:
            spd = rs["median_ms"] / s["median_ms"]
            lines.append(f"  {label}: {spd:.2f}x")

    out_path = OUT_DIR / "perf_summary.txt"
    with open(str(out_path), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Perf summary written: {out_path}", flush=True)
    return out_path
```

Add the argparse flag:

```python
parser.add_argument(
    "--perf-update",
    action="store_true",
    help="Rewrite perf_summary.txt with all accumulated Newton + RealSim stats",
)
```

Wire in `main()` after the `--montage` block:

```python
if args.perf_update:
    print("\n=== Writing full perf summary ===")
    write_full_perf_summary(device)
```

Also call `save_perf_json` right after the Newton run always (unconditionally).

- [ ] **Step 4: Run perf update after all three energies have been run**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_twisting_bar_cudatests.py --energy neohookean --perf-update
```

Expected: `perf_summary.txt` now has all three Newton rows + RealSim NH row.

```bash
cat scripts/twisting_bar_out_cudatests/perf_summary.txt
```

---

## Task 5: End-to-end driver — run everything in one shot

Rather than a new flag, document the three-command sequence that produces all artifacts:

- [ ] **Step 1: Run all three Newton energies**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_twisting_bar_cudatests.py --energy arap
uv run python scripts/fba_twisting_bar_cudatests.py --energy corotational
uv run python scripts/fba_twisting_bar_cudatests.py --energy neohookean
```

- [ ] **Step 2: Render RealSim NH + build montage + update perf table**

```bash
cd /home/ziqiu/work/newton
uv run python scripts/fba_twisting_bar_cudatests.py --energy neohookean --realsim --montage --perf-update
```

Expected terminal summary:
```
=== Rendering RealSim NH .abc trajectory ===
  [RealSim NH] 810 frames in .abc
  [RealSim NH] Done.
=== Building 4-column montage ===
  [montage] Saved .../summary.png
=== Writing full perf summary ===
  Perf summary written: .../perf_summary.txt
```

- [ ] **Step 3: Verify all artifacts**

```bash
ls /home/ziqiu/work/newton/scripts/twisting_bar_out_cudatests/
# Expected dirs: arap/ corotational/ neohookean/ realsim/
# Expected files: summary.png  perf_summary.txt  perf_stats.json

for d in arap corotational neohookean realsim; do
  echo "$d: $(ls /home/ziqiu/work/newton/scripts/twisting_bar_out_cudatests/$d/frame_*.png 2>/dev/null | wc -l) frames"
done
# Expected: 10 10 10 10
```

---

## Task 6: Commit

- [ ] **Step 1: Stage only script + perf text (not PNGs/npz)**

```bash
cd /home/ziqiu/work/newton
git add scripts/fba_twisting_bar_cudatests.py
git add scripts/twisting_bar_out_cudatests/perf_summary.txt
```

- [ ] **Step 2: Create commit**

```bash
git commit -m "$(cat <<'EOF'
Add ARAP/Corot runs to CudaTests twisting bar + RealSim render

Extend fba_twisting_bar_cudatests.py with:
- Newton ARAP and Corot runs at CudaTests params (E=1e9, 810 frames)
- RealSim NH .abc extraction + rendering via dump_abc_traj
- 4-column visual montage (summary.png)
- Full perf table with all 3 Newton energies + RealSim NH baseline

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify commit**

```bash
git log --oneline -3
git show --stat HEAD
```

---

## Self-review checklist

- [x] SNAPSHOT_FRAMES updated to 10-frame set everywhere
- [x] ARAP uses no mu/lam (per existing `run_newton_energy` logic)
- [x] Corot + NH use MU=3.448e8, LAM=3.103e9 from E=1e9, ν=0.45
- [x] RealSim ABC parse handles 810-frame stream, only stores requested frames
- [x] `render_frame()` reused for RealSim — consistent visual style
- [x] Montage col order: ARAP | Corot | NH | RealSim NH
- [x] perf_summary.txt includes all 4 rows with correct RealSim NH hardcoded values
- [x] No PNGs or .npz committed
- [x] Existing NH trajectory.npz is overwritten (not committed anyway)
