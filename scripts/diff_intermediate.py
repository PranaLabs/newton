# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare per-PD-outer-iter intermediate dumps between Newton FBA and RealSim.

Usage:
    uv run python scripts/diff_intermediate.py <fba.npz> <rs_prefix>

The FBA dump is a single ``.npz`` written by ``SolverFBA.configure_diagnostic_dump``.
The RealSim dump is a directory of ``<prefix>_<name>.bin`` raw float64 files plus a
``<prefix>_meta.json`` describing array shapes.

For each shared variable, prints max/mean/median |diff|, argmax particle index,
and the FBA/RS values at that index. Sorted by max-diff descending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_realsim(prefix: str) -> dict[str, np.ndarray]:
    """Load RealSim ``.bin`` dumps via the sibling ``_meta.json``.

    Args:
        prefix: Path prefix used in RealSim's dump (``REALSIM_DIAG_OUT``).

    Returns:
        Dict mapping variable name (e.g. ``"x_pre"``, ``"rhs_k0"``) to an
        ``(N, 3)`` float64 array.
    """
    meta_path = Path(f"{prefix}_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"RealSim meta.json missing: {meta_path}")
    with meta_path.open() as f:
        meta = json.load(f)
    N = int(meta["N"])
    arrays = {}
    for name in meta["arrays"]:
        bin_path = Path(f"{prefix}_{name}.bin")
        if not bin_path.exists():
            print(f"  [warn] RealSim bin missing: {bin_path}", file=sys.stderr)
            continue
        raw = np.fromfile(str(bin_path), dtype=np.float64)
        if raw.size != N * 3:
            print(
                f"  [warn] RealSim bin {bin_path.name} size {raw.size} != N*3 ({N * 3}); skipping",
                file=sys.stderr,
            )
            continue
        arrays[name] = raw.reshape(N, 3)
    return arrays


def _load_fba(npz_path: str) -> dict[str, np.ndarray]:
    """Load Newton FBA ``.npz`` dump, filtering meta-only entries."""
    data = np.load(npz_path)
    arrays: dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("_meta_"):
            continue
        arr = data[key]
        if arr.ndim == 2 and arr.shape[1] == 3:
            arrays[key] = arr.astype(np.float64)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff FBA vs RealSim PD intermediate dumps")
    parser.add_argument("fba_npz", help="Newton FBA diagnostic .npz path")
    parser.add_argument("rs_prefix", help="RealSim dump path prefix (without _<name>.bin suffix)")
    args = parser.parse_args()

    fba = _load_fba(args.fba_npz)
    rs = _load_realsim(args.rs_prefix)

    fba_keys = set(fba.keys())
    rs_keys = set(rs.keys())
    common = sorted(fba_keys & rs_keys)
    only_fba = sorted(fba_keys - rs_keys)
    only_rs = sorted(rs_keys - fba_keys)

    print(f"FBA dump:    {args.fba_npz}  ({len(fba_keys)} arrays)")
    print(f"RS prefix:   {args.rs_prefix}  ({len(rs_keys)} arrays)")
    if only_fba:
        print(f"  only in FBA: {only_fba}")
    if only_rs:
        print(f"  only in RS:  {only_rs}")
    if not common:
        print("No common variables to diff.")
        return

    rows = []
    for name in common:
        a = fba[name]
        b = rs[name]
        if a.shape != b.shape:
            print(f"  [skip] shape mismatch for {name}: FBA={a.shape} RS={b.shape}")
            continue
        d = np.abs(a - b)
        per_particle = d.max(axis=1)  # max |diff| over x,y,z per particle
        max_d = float(per_particle.max())
        mean_d = float(d.mean())
        median_d = float(np.median(d))
        argmax_i = int(np.argmax(per_particle))
        rows.append(
            {
                "name": name,
                "max": max_d,
                "mean": mean_d,
                "median": median_d,
                "argmax_i": argmax_i,
                "fba_vals": a[argmax_i].tolist(),
                "rs_vals": b[argmax_i].tolist(),
            }
        )

    rows.sort(key=lambda r: r["max"], reverse=True)

    print()
    header = f"{'variable':<24} {'max_diff':>12} {'mean_diff':>12} {'median_diff':>12} {'argmax_i':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<24} {r['max']:>12.4e} {r['mean']:>12.4e} {r['median']:>12.4e} {r['argmax_i']:>10d}"
        )
        fa = r["fba_vals"]
        ra = r["rs_vals"]
        print(
            f"  fba @ argmax = [{fa[0]:+.6e}, {fa[1]:+.6e}, {fa[2]:+.6e}]\n"
            f"  rs  @ argmax = [{ra[0]:+.6e}, {ra[1]:+.6e}, {ra[2]:+.6e}]"
        )


if __name__ == "__main__":
    main()
