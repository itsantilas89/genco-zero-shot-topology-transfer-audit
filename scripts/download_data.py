"""Download the target datasets (README section 3).

For each target only the first K `scenario_partition=*` partitions of every table are
fetched (K from settings.yaml; null = whole dataset), into data/<grid>/raw/. Afterwards
the scenario IDs are checked to be contiguous from 0 and the count is stored in
data/<grid>/n_scenarios_present.json, which make_configs.py reads. The dataset's HF commit
hash is recorded there too, for the provenance table.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

from common import DATA_DIR, load_settings, raw_dir

SCENARIOS_PER_PARTITION = 200


def patterns(tables: list[str], partitions: int | None) -> list[str]:
    pats = ["n_scenarios.txt", "stats.parquet/*", "README.md"]
    for t in tables:
        if partitions is None:
            pats.append(f"{t}.parquet/*")
        else:
            pats += [f"{t}.parquet/scenario_partition={k}/*" for k in range(partitions)]
    return pats


def check_contiguous(grid: str, tables: list[str]) -> int:
    """Return the number of scenarios; raise if IDs are not 0..n-1 in every table."""
    n_ref = None
    for t in tables:
        ids = pd.read_parquet(raw_dir(grid) / f"{t}.parquet", columns=["scenario"])["scenario"]
        uniq = ids.unique()
        n = len(uniq)
        if ids.min() != 0 or ids.max() != n - 1:
            raise RuntimeError(f"{grid}/{t}: scenario IDs not contiguous from 0 "
                               f"(min={ids.min()}, max={ids.max()}, unique={n})")
        if n_ref is None:
            n_ref = n
        elif n != n_ref:
            raise RuntimeError(f"{grid}: {t} has {n} scenarios, expected {n_ref}")
    return int(n_ref)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", help="default: all targets in settings.yaml")
    ap.add_argument("--partitions", type=int, help="override K for every selected target")
    ap.add_argument("--with-ybus", action="store_true", help="also fetch y_bus_data (extension)")
    ap.add_argument("--no-scale-only", action="store_true",
                    help="skip the one-partition fetch of the source-only grids")
    args = ap.parse_args()

    settings = load_settings()
    tables = list(settings["tables"]) + (settings["optional_tables"] if args.with_ybus else [])
    api = HfApi()
    jobs = [(g, settings["targets"][g], tables, args.partitions)
            for g in (args.targets or list(settings["targets"]))]
    if not args.targets and not args.no_scale_only:
        jobs += [(g, spec, ["bus_data"], None) for g, spec in settings.get("scale_only", {}).items()]
    for grid, spec, grid_tables, k_override in jobs:
        k = k_override if k_override is not None else spec["partitions"]
        rev = api.dataset_info(spec["hf_repo"]).sha
        print(f"[{grid}] {spec['hf_repo']}@{rev[:8]}  partitions={'all' if k is None else k}")
        raw_dir(grid).mkdir(parents=True, exist_ok=True)
        snapshot_download(spec["hf_repo"], repo_type="dataset", revision=rev,
                          local_dir=str(raw_dir(grid)), allow_patterns=patterns(grid_tables, k))
        n = check_contiguous(grid, grid_tables)
        info = {"grid": grid, "hf_repo": spec["hf_repo"], "revision": rev,
                "partitions": k, "n_scenarios": n}
        (DATA_DIR / grid / "n_scenarios_present.json").write_text(json.dumps(info, indent=2))
        print(f"[{grid}] {n} contiguous scenarios present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
