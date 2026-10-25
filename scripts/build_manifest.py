"""Build manifest.csv from the downloaded checkpoint folders (README section 2).

One row per checkpoint: the 53 single-grid checkpoints from gridfm/genco-pf-datakit plus
the subgrid-pretrained model from gridfm/genco-pf-transfer. Also verifies that every
config shares the model dimensions that make cross-grid transfer possible.
"""
from __future__ import annotations

import re
import sys

import pandas as pd
import yaml

from common import (CKPT_DIR, MANIFEST_PATH, PRETRAIN_DIR, ROOT, SHARED_MODEL_KEYS,
                    SUBGRID_ID, SUBGRID_SOURCE)

PRETRAIN_CFG = "HGNS_PreTrain_subnets_100.yaml"
METRIC_MAP = {
    "Avg. active res. (MW)": "id_active_res_MW",
    "Avg. reactive res. (MVar)": "id_reactive_res_MVar",
    "PBE Mean": "id_pbe_mean",
}


def rel(path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_id_metrics(folder) -> dict:
    csv = folder / "metrics.csv"
    out = {v: float("nan") for v in METRIC_MAP.values()}
    if csv.exists():
        df = pd.read_csv(csv).set_index("Metric")["Value"]
        for key, col in METRIC_MAP.items():
            if key in df.index:
                out[col] = float(df[key])
    return out


def main() -> int:
    rows = []
    pat = re.compile(r"(case\d+_[a-z]+)/(base|small|tiny)/seed(\d+)$")
    for cfg in sorted(CKPT_DIR.glob("case*/*/seed*/*.yaml")):
        folder = cfg.parent
        m = pat.search(folder.relative_to(CKPT_DIR).as_posix())
        if not m:
            continue
        grid, size, seed = m.group(1), m.group(2), int(m.group(3))
        weights = folder / ("last.pt" if grid == "case10000_goc" else "best_model_state_dict.pt")
        stats = folder / "normalizer_stats.pt"
        with open(cfg) as f:
            mc = yaml.safe_load(f)["model"]
        rows.append({
            "ckpt_id": f"{grid}__{size}__seed{seed}",
            "source_grid": grid, "size": size, "seed": seed,
            "hidden_size": mc["hidden_size"],
            "weights_path": rel(weights), "config_path": rel(cfg),
            "normalizer_stats_path": rel(stats) if stats.exists() else "",
            **read_id_metrics(folder),
        })

    pre_cfg = PRETRAIN_DIR / PRETRAIN_CFG
    if pre_cfg.exists():
        with open(pre_cfg) as f:
            mc = yaml.safe_load(f)["model"]
        rows.append({
            "ckpt_id": SUBGRID_ID, "source_grid": SUBGRID_SOURCE, "size": "base",
            "seed": yaml.safe_load(open(pre_cfg))["seed"], "hidden_size": mc["hidden_size"],
            "weights_path": rel(PRETRAIN_DIR / "best_model_state_dict.pt"),
            "config_path": rel(pre_cfg), "normalizer_stats_path": "",
            **{v: float("nan") for v in METRIC_MAP.values()},
        })
    else:
        print(f"WARNING: {pre_cfg} not found; subgrid-pretrained model left out", file=sys.stderr)

    df = pd.DataFrame(rows)
    problems = []
    for r in df.itertuples():
        for p in (r.weights_path, r.config_path):
            if not (ROOT / p).exists():
                problems.append(f"{r.ckpt_id}: missing {p}")
        with open(ROOT / r.config_path) as f:
            mc = yaml.safe_load(f)["model"]
        for k, v in SHARED_MODEL_KEYS.items():
            if mc.get(k) != v:
                problems.append(f"{r.ckpt_id}: {k}={mc.get(k)} (expected {v})")
    # the size label must map to a single hidden size
    for size, g in df.groupby("size"):
        if g["hidden_size"].nunique() != 1:
            problems.append(f"size {size} has hidden sizes {sorted(g['hidden_size'].unique())}")

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"wrote {MANIFEST_PATH} ({len(df)} checkpoints)")
    print(df.groupby(["source_grid", "size"]).size().unstack(fill_value=0))
    if problems:
        print("\nPROBLEMS:", *problems, sep="\n  ")
        return 1
    print("all configs share input_bus_dim/input_gen_dim/edge_dim/num_layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
