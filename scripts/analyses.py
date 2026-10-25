"""Analyses A1-A6 (README section 9). Reads results/results.csv, writes results/analyses/*.csv
and results/summary.json (the numbers quoted in the write-up).

Conventions
  * Residual metric: rel_active_res = mean active residual / mean bus apparent power of the
    target grid (dimensionless; comparable across grids).
  * "Zero-shot" = off-diagonal pairs (target != source grid). Primary variant per checkpoint:
    A (its own normaliser stats) if it ships a stats file, else B (the subgrid model).
  * In-distribution (ID) reference = the checkpoint's own metrics.csv (original held-out test
    split), divided by the mean bus apparent power of its source grid. Our diagonal runs are
    not used for ID numbers: on partition subsets their test split overlaps the original
    training data.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from common import MANIFEST_PATH, RESULTS_DIR, SUBGRID_ID

OUT = RESULTS_DIR / "analyses"
N_BOOT = 5000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- data loading
def n_buses_of(grid: str) -> float:
    m = re.match(r"case(\d+)_", grid)
    return float(m.group(1)) if m else np.nan


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    res = pd.read_csv(RESULTS_DIR / "results.csv")
    man = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    scales = pd.read_csv(RESULTS_DIR / "grid_scales.csv").set_index("grid")["mean_bus_apparent_power_MVA"].to_dict()
    for c in ["id_active_res_MW", "id_reactive_res_MVar", "id_pbe_mean"]:
        man[c] = pd.to_numeric(man[c], errors="coerce")   # blank for the subgrid model
    man["has_stats"] = man["normalizer_stats_path"] != ""
    man["source_scale"] = man["source_grid"].map(scales)
    man["id_rel_active_res"] = man["id_active_res_MW"] / man["source_scale"]
    man["source_n_buses"] = man["source_grid"].map(n_buses_of)
    return res, man, scales


def primary_zero_shot(res: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    """Off-diagonal rows under each checkpoint's primary variant."""
    has_stats = man.set_index("ckpt_id")["has_stats"]
    zs = res[~res["is_diagonal"]].copy()
    want = zs["ckpt_id"].map(has_stats).map({True: "A", False: "B"})
    return zs[zs["variant"] == want].copy()


# ---------------------------------------------------------------- statistics
def spearman(x, y) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def bootstrap_rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    n = len(x)
    if n < 4:
        return spearman(x, y) if n >= 3 else np.nan, np.nan, np.nan
    rhos = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, n, n)
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            rhos.append(spearman(x[i], y[i]))
    lo, hi = np.percentile(rhos, [2.5, 97.5]) if rhos else (np.nan, np.nan)
    return spearman(x, y), float(lo), float(hi)


def permutation_p(id_res: np.ndarray, zs: np.ndarray, y: str, n_perm: int = 5000) -> float:
    """Two-sided p-value for rho(ID, y) under the null that zero-shot residuals are unrelated
    to ID residuals (zero-shot values shuffled across checkpoints). For y = degradation ratio
    the ratio is recomputed after shuffling, so the mechanical negative correlation that comes
    from ID sitting in the denominator is part of the null."""
    if len(id_res) < 4:
        return np.nan
    def stat(z):
        return spearman(id_res, z if y == "zs_rel" else z / id_res)
    obs = stat(zs)
    null = np.array([stat(RNG.permutation(zs)) for _ in range(n_perm)])
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def cluster_bootstrap(df: pd.DataFrame, cluster: str, stat) -> tuple[float, float, float]:
    """Point estimate and 95 % CI of stat(df), resampling whole clusters (checkpoints)."""
    groups = [g for _, g in df.groupby(cluster)]
    est = stat(df)
    boots = []
    for _ in range(N_BOOT):
        pick = RNG.integers(0, len(groups), len(groups))
        boots.append(stat(pd.concat([groups[i] for i in pick])))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


# ---------------------------------------------------------------- A1
def a1_transfer_matrix(res, man, zs) -> pd.DataFrame:
    dc_mw = res[~res["is_diagonal"]].groupby("target")["dc_active_res_MW"].median()
    rows = []
    zcell = zs.groupby(["size", "source_grid", "target", "ckpt_id"]).agg(
        rel=("rel_active_res", "first"), mw=("genco_active_res_MW", "first")).reset_index()
    for (size, src, tgt), g in zcell.groupby(["size", "source_grid", "target"]):
        rows.append({"size": size, "source_grid": src, "target": tgt, "kind": "zero_shot",
                     "n_seeds": len(g), "median_rel_active_res": g["rel"].median(),
                     "median_genco_over_dc": (g["mw"] / dc_mw[tgt]).median()})
    scale_t = res.groupby("target")["target_mean_S_MVA"].first()
    diag = man[man["source_grid"].isin(scale_t.index)]
    for (size, src), g in diag.groupby(["size", "source_grid"]):
        rows.append({"size": size, "source_grid": src, "target": src, "kind": "id_reference",
                     "n_seeds": len(g), "median_rel_active_res": (g["id_active_res_MW"] / scale_t[src]).median(),
                     "median_genco_over_dc": (g["id_active_res_MW"] / dc_mw[src]).median()})
    out = pd.DataFrame(rows)
    out["beats_dc"] = out["median_genco_over_dc"] < 1
    return out


# ---------------------------------------------------------------- A2 (H1)
def a2_topology_overfitting(man, zs) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = zs[zs["ckpt_id"] != SUBGRID_ID].groupby("ckpt_id").agg(
        zs_rel=("rel_active_res", "mean"), n_targets=("target", "nunique")).reset_index()
    ck = man.merge(per, on="ckpt_id")
    ck = ck[ck["ckpt_id"] != SUBGRID_ID].dropna(subset=["id_rel_active_res", "zs_rel"]).copy()
    ck["degradation_ratio"] = ck["zs_rel"] / ck["id_rel_active_res"]

    rows = []
    strata = [("all", "all", ck)]
    strata += [("size", s, g) for s, g in ck.groupby("size")]
    strata += [("source_grid", s, g) for s, g in ck.groupby("source_grid")]
    for scope, name, g in strata:
        for y in ["zs_rel", "degradation_ratio"]:
            rho, lo, hi = bootstrap_rho(g["id_rel_active_res"].to_numpy(), g[y].to_numpy())
            p = permutation_p(g["id_rel_active_res"].to_numpy(), g["zs_rel"].to_numpy(), y)
            rows.append({"scope": scope, "stratum": name, "x": "id_rel_active_res", "y": y,
                         "n": len(g), "spearman_rho": rho, "ci95_lo": lo, "ci95_hi": hi,
                         "permutation_p": p})
    return ck, pd.DataFrame(rows)


def a2_grid_clustered(ck: pd.DataFrame) -> dict:
    """H1 with the source grid as the unit of resampling. The 53 checkpoints come from only 7
    source grids, so the per-checkpoint bootstrap above overstates the effective sample size."""
    rng = np.random.default_rng(1)          # own generator: leaves the other bootstraps unchanged
    grids = ck["source_grid"].unique()
    by = {g: ck[ck["source_grid"] == g] for g in grids}

    def rho(d):
        return spearman(d["id_rel_active_res"], d["zs_rel"])

    boots = []
    for _ in range(N_BOOT):
        d = pd.concat([by[g] for g in rng.choice(grids, len(grids))])
        if d["source_grid"].nunique() > 1:
            boots.append(rho(d))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per_grid = ck.groupby("source_grid")[["id_rel_active_res", "zs_rel"]].median()
    loo = {g: rho(ck[ck["source_grid"] != g]) for g in grids}
    return {"n_source_grids": int(len(grids)), "rho": rho(ck), "ci95_lo": float(lo), "ci95_hi": float(hi),
            "grid_level_rho_of_medians": rho(per_grid),
            "leave_one_grid_out_rho_min": min(loo.values()), "leave_one_grid_out_rho_max": max(loo.values())}


# ---------------------------------------------------------------- A3 (H2)
def a3_source_diversity(res, man, zs) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = zs.groupby(["ckpt_id", "source_grid", "size", "seed"]).agg(
        zs_rel=("rel_active_res", "mean")).reset_index()
    per["source_n_buses"] = per["source_grid"].map(n_buses_of)

    # like-for-like on every target: Base single-grid vs the Base subgrid-pretrained model
    base_single = res[(res["size"] == "base") & ~res["is_diagonal"] & (res["ckpt_id"] != SUBGRID_ID)]
    rows = []
    for tgt in sorted(res["target"].unique()):
        sub = res[(res["ckpt_id"] == SUBGRID_ID) & (res["target"] == tgt) & (res["variant"] == "B")]
        if sub.empty:
            continue
        sub_rel = float(sub["rel_active_res"].iloc[0])
        row = {"target": tgt, "subgrid_rel_B": sub_rel}
        for v in ("A", "B"):
            cells = base_single[(base_single["target"] == tgt) & (base_single["variant"] == v)]
            by_src = cells.groupby("source_grid")["rel_active_res"].median()
            row[f"base_single_best_{v}"] = by_src.min() if len(by_src) else np.nan
            row[f"base_single_median_{v}"] = by_src.median() if len(by_src) else np.nan
            row[f"frac_single_sources_beaten_{v}"] = float((by_src > sub_rel).mean()) if len(by_src) else np.nan
            row[f"n_single_sources_{v}"] = len(by_src)
        rows.append(row)
    return per, pd.DataFrame(rows)


# ---------------------------------------------------------------- A4 (H3)
def a4_recalibration(res) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = res.pivot_table(index=["ckpt_id", "source_grid", "size", "target", "is_diagonal"],
                           columns="variant", values="rel_active_res").reset_index()
    for v in "ABC":
        if v not in wide:
            wide[v] = np.nan
    pairs = wide.dropna(subset=["A", "B"]).copy()
    pairs["gain_A_to_B"] = pairs["A"] / pairs["B"]          # >1: recalibration helped
    pairs["improved"] = pairs["B"] < pairs["A"]

    rows = []
    for name, g in [("zero_shot", pairs[~pairs["is_diagonal"]]), ("diagonal", pairs[pairs["is_diagonal"]])]:
        if g.empty:
            continue
        frac = cluster_bootstrap(g, "ckpt_id", lambda d: d["improved"].mean())
        med = cluster_bootstrap(g, "ckpt_id", lambda d: np.log2(d["gain_A_to_B"]).median())
        rows.append({"scope": name, "n_pairs": len(g), "n_checkpoints": g["ckpt_id"].nunique(),
                     "frac_improved": frac[0], "frac_ci_lo": frac[1], "frac_ci_hi": frac[2],
                     "median_log2_gain_A_over_B": med[0], "log2_ci_lo": med[1], "log2_ci_hi": med[2]})
    gap = wide.dropna(subset=["B", "C"]).copy()
    gap["B_over_C"] = gap["B"] / gap["C"]
    return pairs, pd.DataFrame(rows), gap


# ---------------------------------------------------------------- A5
def a5_safety(zs) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = zs[["run_name", "ckpt_id", "source_grid", "size", "target", "variant", "rel_active_res",
            "genco_over_dc", "vm_violation_rate", "vm_true_violation_rate", "vm_cal_slope",
            "va_cal_slope", "vm_cal_r2", "va_cal_r2"]].copy()
    d["violation_excess"] = d["vm_violation_rate"] - d["vm_true_violation_rate"]
    low_res = (d["rel_active_res"] <= d["rel_active_res"].median()) | (d["genco_over_dc"] < 1)
    collapsed = (d["vm_cal_slope"].abs() < 0.3) | (d["va_cal_slope"].abs() < 0.3)
    d["flag_low_residual_collapsed_calibration"] = low_res & collapsed
    return d, d[d["flag_low_residual_collapsed_calibration"]]


# ---------------------------------------------------------------- A6
def a6_mxgps(zs) -> pd.DataFrame:
    ref = pd.read_csv("reference/mxgps_case14.csv")
    rows = [{"model": r.model, "mae_vm_pu": r.mae_vm_pu, "bvr_percent": r.bvr_percent,
             "origin": "published (MxGPS Table IV, zero-shot on 14-bus)"} for r in ref.itertuples()]
    c14 = zs[zs["target"] == "case14_ieee"]
    for label, g in [("GENCO single-grid (mean over checkpoints)", c14[c14["ckpt_id"] != SUBGRID_ID]),
                     ("GENCO subgrid-pretrained", c14[c14["ckpt_id"] == SUBGRID_ID])]:
        if g.empty:
            continue
        rows.append({"model": label, "mae_vm_pu": g["vm_mae_all"].mean(),
                     "bvr_percent": 100 * g["vm_violation_rate"].mean(),
                     "origin": f"this audit, case14_ieee, n={len(g)} checkpoints"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res, man, _ = load()
    zs = primary_zero_shot(res, man)
    summary: dict = {"n_runs": int(len(res)), "n_zero_shot_runs": int(len(zs))}

    a1 = a1_transfer_matrix(res, man, zs)
    a1.to_csv(OUT / "a1_transfer_matrix.csv", index=False)

    ck, cor = a2_topology_overfitting(man, zs)
    ck.to_csv(OUT / "a2_checkpoints.csv", index=False)
    cor.to_csv(OUT / "a2_correlations.csv", index=False)
    head = cor[(cor.scope == "all")].set_index("y")
    summary["H1"] = {
        "n_checkpoints": int(len(ck)),
        "rho_id_vs_zero_shot_residual": head.loc["zs_rel", ["spearman_rho", "ci95_lo", "ci95_hi", "permutation_p"]].to_dict() if len(ck) else None,
        "rho_id_vs_degradation_ratio": head.loc["degradation_ratio", ["spearman_rho", "ci95_lo", "ci95_hi", "permutation_p"]].to_dict() if len(ck) else None,
        "note": ("H1 (topology overfitting) predicts that low in-distribution (ID) residual goes "
                 "with large degradation. rho(ID, degradation ratio) is negative under H1 but is "
                 "also negative by construction (ID is the ratio's denominator): judge it by its "
                 "permutation_p, which uses that mechanical effect as the null. rho(ID, zero-shot "
                 "residual) > 0 (CI above 0) would mean better ID accuracy goes with better "
                 "transfer, i.e. against H1."),
    }

    summary["H1"]["grid_clustered_bootstrap"] = a2_grid_clustered(ck)

    per, a3 = a3_source_diversity(res, man, zs)
    per.to_csv(OUT / "a3_zero_shot_by_source.csv", index=False)
    a3.to_csv(OUT / "a3_subgrid_vs_base.csv", index=False)
    if not a3.empty:
        summary["H2"] = {"targets": int(len(a3)),
                         "mean_frac_single_sources_beaten_vs_A": float(a3["frac_single_sources_beaten_A"].mean()),
                         "mean_frac_single_sources_beaten_vs_B": float(a3["frac_single_sources_beaten_B"].mean()),
                         "targets_where_subgrid_beats_best_single_A": int((a3["subgrid_rel_B"] < a3["base_single_best_A"]).sum()),
                         "note": "one subgrid checkpoint, so no seed uncertainty on its side"}

    pairs, a4, gap = a4_recalibration(res)
    pairs.to_csv(OUT / "a4_pairs.csv", index=False)
    a4.to_csv(OUT / "a4_summary.csv", index=False)
    gap.to_csv(OUT / "a4_B_vs_C.csv", index=False)
    summary["H3"] = a4.set_index("scope").to_dict("index") if not a4.empty else None

    a5, flagged = a5_safety(zs)
    a5.to_csv(OUT / "a5_safety.csv", index=False)
    flagged.to_csv(OUT / "a5_flagged.csv", index=False)
    summary["A5"] = {"n_flagged": int(len(flagged)), "of": int(len(a5)),
                     "median_violation_excess": float(a5["violation_excess"].median())}

    a6 = a6_mxgps(zs)
    a6.to_csv(OUT / "a6_mxgps_reference.csv", index=False)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
