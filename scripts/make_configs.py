"""Generate one evaluation YAML per (checkpoint, target) pair (README section 5).

For each pair the source config is copied and only the data section is retargeted:
  * data.networks / data.scenarios point at the target (scenarios = number present).
  * test_ratio, val_ratio and split_by_load_scenario_idx stay as in the source config.
  * seed: the source seed on the diagonal (target == source grid), `eval_seed` from
    settings.yaml everywhere else, so every checkpoint sees the identical test set on a
    given target.
  * data.mask_ref_vm = true (needed by all checkpoints released before October 2026).
  * verbose = false: no diagnostic plots, they are not used.
The subgrid-pretrained checkpoint is converted from the PreTraining task to PowerFlow.
"""
from __future__ import annotations

import argparse
import copy
import sys

import pandas as pd
import yaml

from common import (CONFIG_DIR, MANIFEST_PATH, ROOT, SUBGRID_ID, load_settings,
                    n_scenarios_present, pair_id, target_names)

PRETRAIN_ONLY_DATA_KEYS = ["mask_p_high", "mask_p_low", "training_networks",
                           "validation_networks", "test_networks"]


def pf_reference_training(manifest: pd.DataFrame) -> dict:
    """`training` section of a Base-size PowerFlow config (used for the subgrid model)."""
    row = manifest[(manifest["size"] == "base") & (manifest["source_grid"] == "case118_ieee")].iloc[0]
    with open(ROOT / row.config_path) as f:
        return copy.deepcopy(yaml.safe_load(f)["training"])


def build_config(row, target: str, n_scen: int, eval_seed: int, ref_training: dict) -> dict:
    with open(ROOT / row.config_path) as f:
        cfg = yaml.safe_load(f)

    if row.ckpt_id == SUBGRID_ID:
        cfg["task"] = {"task_name": "PowerFlow"}
        for k in PRETRAIN_ONLY_DATA_KEYS:
            cfg["data"].pop(k, None)
        cfg["training"] = ref_training
        cfg["data"].setdefault("test_ratio", 0.1)
        cfg["data"].setdefault("val_ratio", 0.1)
        cfg["data"].setdefault("split_by_load_scenario_idx", True)

    data = cfg["data"]
    data["networks"] = [target]
    data["scenarios"] = [n_scen]
    data["mask_ref_vm"] = True
    cfg["seed"] = int(row.seed) if row.source_grid == target else int(eval_seed)
    cfg["verbose"] = False
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--targets", nargs="*", help="restrict to these targets")
    args = ap.parse_args()

    settings = load_settings()
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    ref_training = pf_reference_training(manifest)
    targets = args.targets or target_names(settings)

    CONFIG_DIR.mkdir(exist_ok=True)
    n = 0
    for target in targets:
        n_scen = n_scenarios_present(target)
        for row in manifest.itertuples():
            cfg = build_config(row, target, n_scen, settings["eval_seed"], ref_training)
            path = CONFIG_DIR / f"{pair_id(row.ckpt_id, target)}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            n += 1
    print(f"wrote {n} configs to {CONFIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
