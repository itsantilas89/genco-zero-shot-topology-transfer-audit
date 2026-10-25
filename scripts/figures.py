"""Figures for the write-up (README section 10). Reads results/analyses/*.csv; run
analyses.py first. Every figure is saved with an opaque white background and a
colour-blind-safe palette (Okabe-Ito) so it reads the same on light and dark pages.
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES_DIR, RESULTS_DIR, SIZES, SUBGRID_SOURCE
from analyses import n_buses_of

A = RESULTS_DIR / "analyses"
SIZE_COLOR = {"base": "#0072B2", "small": "#E69F00", "tiny": "#009E73"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.edgecolor": "#333333",
                     "axes.labelcolor": "#111111", "xtick.color": "#111111",
                     "ytick.color": "#111111", "text.color": "#111111", "font.size": 9})


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}.png")


def grid_order(grids) -> list[str]:
    return sorted(grids, key=lambda g: (g == SUBGRID_SOURCE, n_buses_of(g)))


# ------------------------------------------------------------------ fig 1
def fig_transfer_matrix() -> None:
    a1 = pd.read_csv(A / "a1_transfer_matrix.csv")
    targets = grid_order(a1["target"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False, constrained_layout=True)
    vals = np.log10(a1["median_rel_active_res"].clip(lower=1e-9))
    vmin, vmax = np.floor(vals.min()), np.ceil(vals.max())
    im = None
    for ax, size in zip(axes, SIZES):
        d = a1[a1["size"] == size]
        sources = grid_order(d["source_grid"].unique())
        mat = np.full((len(sources), len(targets)), np.nan)
        for r in d.itertuples():
            mat[sources.index(r.source_grid), targets.index(r.target)] = np.log10(max(r.median_rel_active_res, 1e-9))
        im = ax.imshow(mat, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(targets)), [t.replace("_", "\n") for t in targets], fontsize=7)
        ax.set_yticks(range(len(sources)), [s.split("_")[0] for s in sources], fontsize=7)
        ax.set_xlabel("target grid")
        ax.set_title(f"{size.capitalize()}  (n = {int(d['n_seeds'].sum())} checkpoint-pairs)")
        for r in d.itertuples():
            i, j = sources.index(r.source_grid), targets.index(r.target)
            v = np.log10(max(r.median_rel_active_res, 1e-9))
            ax.text(j, i, f"{r.median_rel_active_res:.1e}" + ("\n★" if r.beats_dc else ""),
                    ha="center", va="center", fontsize=6.5,
                    color="white" if (v - vmin) / (vmax - vmin + 1e-9) < 0.5 else "black")
            if r.kind == "id_reference":
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.6))
    axes[0].set_ylabel("source (training) grid")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("log10 relative active residual (dimensionless)")
    fig.suptitle("A1  Transfer matrix. Black frame = in-distribution reference; ★ = GENCO residual below DC-PF; "
                 "median over seeds", fontsize=9.5)
    save(fig, "fig1_transfer_matrix")


# ------------------------------------------------------------------ fig 2 (headline)
def fig_h1_scatter() -> None:
    ck = pd.read_csv(A / "a2_checkpoints.csv")
    cor = pd.read_csv(A / "a2_correlations.csv")
    if ck.empty:
        return
    grids = grid_order(ck["source_grid"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, y, ylabel in [(axes[0], "zs_rel", "mean zero-shot relative active residual (dimensionless)"),
                          (axes[1], "degradation_ratio", "degradation ratio  zero-shot / in-distribution (dimensionless)")]:
        for g_i, g in enumerate(grids):
            for size in SIZES:
                d = ck[(ck.source_grid == g) & (ck["size"] == size)]
                ax.scatter(d["id_rel_active_res"], d[y], s=34, marker=MARKERS[g_i % len(MARKERS)],
                           color=SIZE_COLOR[size], edgecolor="black", linewidth=0.4, alpha=.9,
                           label=f"{g.split('_')[0][4:]}-bus, {size}" if size == "base" else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("in-distribution relative active residual (dimensionless)")
        ax.set_ylabel(ylabel)
        row = cor[(cor.scope == "all") & (cor.y == y)].iloc[0]
        pv = "< 0.001" if row.permutation_p < 1e-3 else f"= {row.permutation_p:.3f}"
        ax.set_title(f"Spearman ρ = {row.spearman_rho:.2f}  [95 % CI {row.ci95_lo:.2f}, {row.ci95_hi:.2f}]\n"
                     f"n = {int(row.n)} checkpoints, permutation p {pv}", fontsize=8.5)
        ax.grid(alpha=.25)
    handles = [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], ls="", color="#777777",
                          mec="black", label=f"{g.split('_')[0][4:]}-bus source") for i, g in enumerate(grids)]
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, mec="black", label=s) for s, c in SIZE_COLOR.items()]
    axes[1].legend(handles=handles, fontsize=7, loc="best", ncol=2)
    fig.suptitle("A2 (H1)  Do checkpoints with lower in-distribution error degrade more on unseen grids?", fontsize=10)
    save(fig, "fig2_h1_id_vs_zero_shot")


# ------------------------------------------------------------------ fig 3
def fig_source_scale() -> None:
    per = pd.read_csv(A / "a3_zero_shot_by_source.csv")
    cmp_ = pd.read_csv(A / "a3_subgrid_vs_base.csv")
    sub = per[per["source_grid"] == SUBGRID_SOURCE]
    single = per[per["source_grid"] != SUBGRID_SOURCE]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for size in SIZES:
        d = single[single["size"] == size]
        ax.scatter(d["source_n_buses"], d["zs_rel"], s=30, color=SIZE_COLOR[size], edgecolor="black",
                   linewidth=.4, label=f"{size} (single grid)")
    if not sub.empty:
        ax.axhline(float(sub["zs_rel"].iloc[0]), color="#CC79A7", lw=1.8, ls="--",
                   label="subgrid-pretrained (base, 100 topologies)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of buses in the source (training) grid")
    ax.set_ylabel("mean zero-shot relative active residual (dimensionless)")
    n = len(single)
    ax.set_title(f"A3 (H2)  Zero-shot residual vs source size  (n = {n} checkpoints"
                 + (f", subgrid beats {cmp_['frac_single_sources_beaten_A'].mean():.0%} of Base single-grid sources"
                    if not cmp_.empty else "") + ")", fontsize=8.5)
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    save(fig, "fig3_source_scale_and_subgrid")


# ------------------------------------------------------------------ fig 4
def fig_recalibration() -> None:
    pairs = pd.read_csv(A / "a4_pairs.csv")
    summ = pd.read_csv(A / "a4_summary.csv").set_index("scope")
    if pairs.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(-6, 6, 49)
    for name, color, label in [("zero_shot", "#0072B2", "zero-shot pairs"),
                               ("diagonal", "#D55E00", "in-distribution pairs")]:
        d = pairs[pairs["is_diagonal"] == (name == "diagonal")]
        if d.empty:
            continue
        ax.hist(np.log2(d["gain_A_to_B"]).clip(-6, 6), bins=bins, alpha=.6, color=color,
                label=f"{label}: n = {len(d)}, improved {summ.loc[name, 'frac_improved']:.0%} "
                      f"[{summ.loc[name, 'frac_ci_lo']:.0%}, {summ.loc[name, 'frac_ci_hi']:.0%}]")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("log2 (residual with source stats A / residual with target-input stats B)   → right = B better")
    ax.set_ylabel("number of (checkpoint, target) pairs")
    ax.set_title("A4 (H3)  Gain from recalibrating the normaliser on target inputs (values clipped to ±6)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.25)
    save(fig, "fig4_recalibration_gain")


# ------------------------------------------------------------------ fig 5
def fig_safety() -> None:
    d = pd.read_csv(A / "a5_safety.csv")
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = axes[0].scatter(d["vm_cal_slope"], d["rel_active_res"], c=100 * d["violation_excess"],
                         cmap="magma_r", s=22, edgecolor="black", linewidth=.3)
    f = d[d["flag_low_residual_collapsed_calibration"]]
    axes[0].scatter(f["vm_cal_slope"], f["rel_active_res"], s=90, facecolor="none", edgecolor="#D55E00", lw=1.5,
                    label=f"low residual, collapsed calibration (n = {len(f)})")
    axes[0].set_yscale("log"); axes[0].axvline(1, color="#555555", lw=.8, ls=":")
    axes[0].set_xlabel("Vm calibration slope, predicted vs true (ideal = 1)")
    axes[0].set_ylabel("relative active residual (dimensionless)")
    axes[0].legend(fontsize=7)
    fig.colorbar(sc, ax=axes[0]).set_label("Vm boundary violation rate above ground truth (percentage points)")
    axes[1].hist(100 * d["vm_violation_rate"], bins=40, color="#0072B2", alpha=.8)
    axes[1].set_xlabel("Vm boundary violation rate (% of bus predictions outside [Vmin, Vmax])")
    axes[1].set_ylabel("number of zero-shot runs")
    fig.suptitle(f"A5  Safety of zero-shot predictions  (n = {len(d)} runs)", fontsize=10)
    save(fig, "fig5_safety")


def main() -> int:
    for fn in (fig_transfer_matrix, fig_h1_scatter, fig_source_scale, fig_recalibration, fig_safety):
        try:
            fn()
        except Exception as e:                 # noqa: BLE001
            print(f"{fn.__name__} failed: {e!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
