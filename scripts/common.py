"""Shared paths, constants and small helpers for the audit scripts."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "settings.yaml"

CKPT_DIR = ROOT / "genco-pf-datakit"
PRETRAIN_DIR = ROOT / "genco-pf-transfer" / "pretraining"
GRAPHKIT_DIR = ROOT / "gridfm-graphkit"
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
STATS_TARGET_DIR = ROOT / "normalizer_stats_target"
RESULTS_DIR = Path(os.environ.get("AUDIT_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("AUDIT_FIGURES_DIR", ROOT / "figures"))
RUNS_DIR = ROOT / "runs"          # scratch (MLflow stores, predictions); gitignored
MANIFEST_PATH = ROOT / "manifest.csv"

SIZES = ["base", "small", "tiny"]
SUBGRID_ID = "pretrain_subgrid100"   # ckpt_id of the subgrid-pretrained model
SUBGRID_SOURCE = "subgrid100"        # its `source_grid` value
VARIANTS = ["A", "B", "C"]

# Dimensions that must be identical across checkpoints for transfer to be possible.
SHARED_MODEL_KEYS = {"input_bus_dim": 15, "input_gen_dim": 6, "edge_dim": 10, "num_layers": 12}


def load_settings() -> dict:
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f)


def target_names(settings: dict | None = None) -> list[str]:
    return list((settings or load_settings())["targets"])


def raw_dir(grid: str) -> Path:
    return DATA_DIR / grid / "raw"


def n_scenarios_present(grid: str) -> int:
    """Number of scenarios actually present for `grid` (written by download_data.py)."""
    path = DATA_DIR / grid / "n_scenarios_present.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/download_data.py first")
    return int(json.loads(path.read_text())["n_scenarios"])


def pair_id(ckpt_id: str, target: str) -> str:
    return f"{ckpt_id}__{target}"


def run_name(ckpt_id: str, target: str, variant: str) -> str:
    return f"{ckpt_id}__{target}__{variant}"


def python_exe() -> str:
    return sys.executable


def graphkit_cmd() -> list[str]:
    return [python_exe(), "-m", "gridfm_graphkit"]


def env_for_runs() -> dict:
    env = dict(os.environ)
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    return env
