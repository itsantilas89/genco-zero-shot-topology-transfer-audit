"""Per-run metrics from saved predictions, and collection into the result tables.

`process_run` is called by run_matrix.py right after each evaluation, so bulky prediction
files can be deleted straight away. `python scripts/postprocess.py collect` then merges all
results/per_run/*.json with manifest.csv into results_raw.csv and results.csv
(README sections 7 and 8).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from common import DATA_DIR, MANIFEST_PATH, RESULTS_DIR, RUNS_DIR, raw_dir

PER_RUN_DIR = RESULTS_DIR / "per_run"
BUS_TYPES = ["PQ", "PV", "REF"]


# ---------------------------------------------------------------- helpers
def artifacts_test_dir(run_name: str):
    """The single MLflow run folder created for `run_name` (run_matrix gives each run its
    own --log_dir, so there is exactly one)."""
    hits = list((RUNS_DIR / run_name / "mlruns").glob("*/*/artifacts/test"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one artifacts/test dir under runs/{run_name}, found {len(hits)}")
    return hits[0]


def read_metric_csv(path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["Metric"], pd.to_numeric(df["Value"], errors="coerce")))


def calibration(pred: np.ndarray, true: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[ok], true[ok]
    if len(t) < 3 or np.std(t) == 0:
        return {"slope": np.nan, "r2": np.nan, "r2_identity": np.nan}
    slope = float(np.polyfit(t, p, 1)[0])
    r = float(np.corrcoef(t, p)[0, 1])
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {"slope": slope, "r2": r * r,
            "r2_identity": float(1 - ((p - t) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan}


def metrics_from_predictions(df: pd.DataFrame) -> dict:
    """Accuracy, bound violations and calibration from <grid>_predictions.parquet."""
    out: dict = {}
    vm_err = (df["Vm_pred"] - df["Vm_target"]).abs()
    va_err = (df["Va_pred"] - df["Va_target"]).abs()
    out["vm_mae_all"], out["va_mae_all"] = float(vm_err.mean()), float(va_err.mean())
    for t in BUS_TYPES:
        m = df[t].astype(bool)
        out[f"vm_mae_{t}"] = float(vm_err[m].mean()) if m.any() else np.nan
        out[f"va_mae_{t}"] = float(va_err[m].mean()) if m.any() else np.nan

    has_bounds = (df["Vm_max"] > df["Vm_min"]) & (df["Vm_min"] > 0)
    viol = (df["Vm_pred"] < df["Vm_min"]) | (df["Vm_pred"] > df["Vm_max"])
    out["vm_violation_rate"] = float(viol[has_bounds].mean()) if has_bounds.any() else np.nan
    true_viol = (df["Vm_target"] < df["Vm_min"]) | (df["Vm_target"] > df["Vm_max"])
    out["vm_true_violation_rate"] = float(true_viol[has_bounds].mean()) if has_bounds.any() else np.nan

    for name, p, t in [("vm", "Vm_pred", "Vm_target"), ("va", "Va_pred", "Va_target")]:
        for k, v in calibration(df[p].to_numpy(), df[t].to_numpy()).items():
            out[f"{name}_cal_{k}"] = v
    out["pred_active_res_MW"] = float(df["active res. (MW)"].mean())
    out["pred_reactive_res_MVar"] = float(df["reactive res. (MVar)"].mean())
    out["mean_bus_apparent_power_test_MVA"] = float(np.hypot(df["Pd"], df["Qd"]).mean())
    out["n_test_scenarios"] = int(df["scenario"].nunique())
    out["n_buses"] = int(df["bus"].nunique())
    return out


def mean_bus_apparent_power(grid: str) -> float:
    """Mean over (scenario, bus) of sqrt(Pd^2 + Qd^2) in MVA, from the raw target data."""
    bus = pd.read_parquet(raw_dir(grid) / "bus_data.parquet", columns=["Pd", "Qd"])
    return float(np.hypot(bus["Pd"], bus["Qd"]).mean())


# ---------------------------------------------------------------- per run
def process_run(run_name: str, meta: dict, wall_s: float, keep_predictions: bool = False) -> dict:
    """Read one finished evaluation, write results/per_run/<run_name>.json, return the row."""
    grid = meta["target"]
    tdir = artifacts_test_dir(run_name)
    base = read_metric_csv(tdir / f"{grid}_metrics.csv")
    acdc = read_metric_csv(tdir / f"{grid}_ac_dc_metrics.csv")
    if not base:
        raise RuntimeError(f"no {grid}_metrics.csv in {tdir}")

    row = dict(meta)
    row.update({
        "run_name": run_name, "wall_s": round(wall_s, 2),
        "genco_active_res_MW": base.get("Avg. active res. (MW)"),
        "genco_reactive_res_MVar": base.get("Avg. reactive res. (MVar)"),
        "genco_pbe_mean": base.get("PBE Mean"), "genco_pbe_max": base.get("PBE Max"),
        "ac_truth_active_res_MW": acdc.get("AC Avg. active res. (MW)"),
        "ac_truth_reactive_res_MVar": acdc.get("AC Avg. reactive res. (MVar)"),
        "dc_active_res_MW": acdc.get("DC Avg. active res. (MW)"),
        "dc_nan_scenarios": acdc.get("DC DC NaN scenarios"),
    })
    pred_path = tdir / f"{grid}_predictions.parquet"
    if pred_path.exists():
        row.update(metrics_from_predictions(pd.read_parquet(pred_path)))
    else:
        row["error_note"] = "predictions parquet missing"

    PER_RUN_DIR.mkdir(parents=True, exist_ok=True)
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
    (PER_RUN_DIR / f"{run_name}.json").write_text(json.dumps(clean, default=float, indent=1))

    if not keep_predictions:
        for f in tdir.glob("*.parquet"):
            f.unlink()
    return row


def write_grid_scales() -> dict:
    """Mean bus apparent power for every grid that has bus data on disk (targets and the
    source-only grids); used to express residuals relative to grid size."""
    scale = {p.parent.parent.name: mean_bus_apparent_power(p.parent.parent.name)
             for p in sorted(DATA_DIR.glob("*/raw/bus_data.parquet"))}
    pd.DataFrame({"grid": list(scale), "mean_bus_apparent_power_MVA": list(scale.values())}
                 ).to_csv(RESULTS_DIR / "grid_scales.csv", index=False)
    return scale


# ---------------------------------------------------------------- collection
RAW_COLS = ["run_name", "ckpt_id", "source_grid", "size", "seed", "hidden_size", "target",
            "variant", "is_diagonal", "genco_active_res_MW", "genco_reactive_res_MVar",
            "genco_pbe_mean", "genco_pbe_max", "ac_truth_active_res_MW",
            "ac_truth_reactive_res_MVar", "dc_active_res_MW", "dc_nan_scenarios", "wall_s",
            "id_active_res_MW", "id_reactive_res_MVar", "id_pbe_mean"]


def collect() -> pd.DataFrame:
    files = sorted(PER_RUN_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"no per-run results in {PER_RUN_DIR}")
    df = pd.DataFrame([json.loads(f.read_text()) for f in files])
    man = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    id_cols = ["id_active_res_MW", "id_reactive_res_MVar", "id_pbe_mean"]
    df = df.drop(columns=[c for c in id_cols if c in df], errors="ignore").merge(
        man[["ckpt_id"] + id_cols], on="ckpt_id", how="left")
    df["is_diagonal"] = df["source_grid"] == df["target"]

    RESULTS_DIR.mkdir(exist_ok=True)
    df[[c for c in RAW_COLS if c in df]].sort_values("run_name").to_csv(
        RESULTS_DIR / "results_raw.csv", index=False)

    scale = write_grid_scales()
    df["target_mean_S_MVA"] = df["target"].map(scale)
    df["rel_active_res"] = df["genco_active_res_MW"] / df["target_mean_S_MVA"]
    df["rel_reactive_res"] = df["genco_reactive_res_MVar"] / df["target_mean_S_MVA"]
    df["genco_over_dc"] = df["genco_active_res_MW"] / df["dc_active_res_MW"]
    df["beats_dc"] = df["genco_over_dc"] < 1
    df.sort_values("run_name").to_csv(RESULTS_DIR / "results.csv", index=False)
    print(f"wrote results_raw.csv and results.csv ({len(df)} runs)")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["collect"])
    ap.parse_args()
    collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
