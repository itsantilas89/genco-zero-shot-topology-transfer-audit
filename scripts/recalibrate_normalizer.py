"""Normaliser variants (README section 6).

graphkit's `--normalizer_stats` file is a dict keyed by network name, and it is only used
when the evaluated network's name is a key; otherwise graphkit silently refits on the
target. So every stats file written here is keyed by the *target* grid name.

  A  source stats : the checkpoint's own stats, re-keyed to the target (per-run file).
  B  target, inputs only : baseMVA = 95th percentile of non-zero target Pd and Qd,
                 vn_kv_max = max target vn_kv. No labels used. One file per target in
                 normalizer_stats_target/<grid>.pt.
  C  target, full refit : graphkit's default fit_on_train (includes Pg and Qg, i.e.
                 labels); needs no file, the run simply omits --normalizer_stats.

Run as a script to (re)build the variant-B files and a summary table.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from common import (RESULTS_DIR, RUNS_DIR, STATS_TARGET_DIR, ROOT, load_settings, raw_dir,
                    target_names)

BASE_MVA_ORIG = 100.0


def _p95_nonzero(*series: pd.Series) -> float:
    vals = pd.concat([s[s != 0] for s in series])
    return float(np.percentile(vals, 95))


def target_stats(grid: str, include_labels: bool = False) -> dict:
    """Variant B (include_labels=False) or an all-scenario stand-in for C (True)."""
    bus = pd.read_parquet(raw_dir(grid) / "bus_data.parquet",
                          columns=["Pd", "Qd", "Qg", "vn_kv"])
    series = [bus["Pd"], bus["Qd"]]
    if include_labels:
        gen = pd.read_parquet(raw_dir(grid) / "gen_data.parquet", columns=["p_mw"])
        series += [gen["p_mw"], bus["Qg"]]
    return {"baseMVA_orig": torch.tensor(BASE_MVA_ORIG),
            "baseMVA": torch.tensor(_p95_nonzero(*series), dtype=torch.float),
            "vn_kv_max": torch.tensor(float(bus["vn_kv"].max()), dtype=torch.float)}


def variant_b_path(grid: str):
    return STATS_TARGET_DIR / f"{grid}.pt"


def write_variant_b(grid: str):
    STATS_TARGET_DIR.mkdir(exist_ok=True)
    torch.save({grid: target_stats(grid)}, variant_b_path(grid))
    return variant_b_path(grid)


def variant_a_path(manifest_row, target: str):
    """Source checkpoint's stats re-keyed to `target`; written to a per-pair scratch file."""
    src = torch.load(ROOT / manifest_row.normalizer_stats_path, map_location="cpu",
                     weights_only=True)
    if manifest_row.source_grid not in src:
        raise KeyError(f"{manifest_row.ckpt_id}: stats file has keys {list(src)}")
    out_dir = RUNS_DIR / "_stats_A"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest_row.ckpt_id}__as__{target}.pt"
    torch.save({target: src[manifest_row.source_grid]}, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*")
    args = ap.parse_args()
    rows = []
    for grid in args.targets or target_names(load_settings()):
        write_variant_b(grid)
        b = target_stats(grid)
        c = target_stats(grid, include_labels=True)
        rows.append({"target": grid, "B_baseMVA": float(b["baseMVA"]),
                     "B_vn_kv_max": float(b["vn_kv_max"]),
                     "Cstandin_baseMVA": float(c["baseMVA"])})
        print(f"{grid}: B baseMVA={rows[-1]['B_baseMVA']:.2f} vn_kv_max={rows[-1]['B_vn_kv_max']:g}"
              f" (all-scenario refit incl. labels: {rows[-1]['Cstandin_baseMVA']:.2f})")
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "normalizer_stats_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
