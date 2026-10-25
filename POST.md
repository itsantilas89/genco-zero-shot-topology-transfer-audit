# Where does GENCO break? A zero-shot topology-transfer audit of the released checkpoints

## Motivation

The GENCO paper names zero-shot transfer to unseen grids as an open problem. This diagnostic study (audit) asks, using only the released checkpoints and no training, how they behave on grids they were never trained on and which factors are associated with the degradation. We look at three candidate factors: 1. topology overfitting (a hypothesis taken from [MxGPS](https://arxiv.org/pdf/2607.13763)), 2. source-grid diversity and scale, and 3. normaliser statistics.

## Contact
- **Name:** Ioannis Tsantilas
- **Email:** itsantilas@epu.ntua.gr
- **Affiliation:** EPU - NTUA, Greece
- **GitHub:** [Genco Zero-Shot Topology Transfer](https://github.com/itsantilas89/genco-zero-shot-topology-transfer/)

## GENCO Background

The terms used below are:

**The checkpoints**
- **Base / Small / Tiny** are three sizes of the same 12-layer network. They differ only in the hidden dimension: 48 (Base), 24 (Small) and 12 (Tiny).
- **Seed** (0, 1, 42): the random seed of a training run, so checkpoints with the same grid and size but different seeds are repeats of the same recipe.
- **Source grid** is the grid a checkpoint was trained on; **target grid** is the grid we evaluate it on. IEEE 14, 30, 57 and 118 are standard test systems named after their number of buses; GOC 500, 2000 and 10000 are larger test cases from the PGLib benchmark library (GOC: Grid Optimization Competition). Each single-grid checkpoint is trained on one grid only.
- **In-distribution** means evaluated on the held-out test split of its own source grid. **Zero-shot** means evaluated on a different grid with no fine-tuning and no training on that grid.
- **Subgrid-pretrained model**: a single Base-size model that the GENCO authors pretrained on 100 *subgrids*, i.e. smaller networks (100 to 1,000 buses) cut out of ten large PGLib grids (4,619 to 30,000 buses) by random walks, with 9,000 scenarios each. It is meant to give the model exposure to many topologies instead of one. None of the ten base grids is one of our targets. It is the only released checkpoint of this kind, and it ships without normaliser statistics.

**Quantities we report**
- **Power-balance residual** (active in MW, reactive in MVar): at each bus, how far the injected power is from the power flowing out through the lines, computed from the predicted voltages. Zero means physically consistent. We average its absolute value over buses and scenarios.
- **Relative active residual**: that mean active residual divided by the mean bus apparent power of the grid, so grids of different size are comparable. A value of 1 means the average mismatch per bus is as large as the average load.
- **Normaliser statistics**: graphkit rescales powers and voltages before they enter the network, using constants fitted on the training grid (`baseMVA`, the 95th percentile of the non-zero power values, and `vn_kv_max`, the highest nominal voltage). A checkpoint ships the constants of its own training grid. Variants A, B and C in the Setup section are three ways of choosing them on a new grid.
- **Boundary violation rate**: the share of buses whose predicted Vm falls outside that bus's allowed range [Vmin, Vmax].
- **Calibration slope**: the slope of a line fitted to predicted against true values. A slope near 1 is ideal; a slope near 0 means the predictions barely follow the truth (close to a constant), even if their average error looks small.

**The three hypotheses**
- **H1, topology overfitting:** checkpoints with the lowest in-distribution error degrade the most on unseen grids, an idea we developed on [MxGPS](https://arxiv.org/pdf/2607.13763).
- **H2, source diversity:** the subgrid-pretrained model transfers better than single-grid checkpoints of the same size.
- **H3, normaliser mismatch:** part of the zero-shot gap comes from the per-grid normaliser statistics, and recalibrating them on target inputs recovers part of it.

## Relation to the GENCO publication

The GENCO [publication paper](https://arxiv.org/pdf/2608.09921) contains two zero-shot results and one adaptation result:

- *Task 3.1 (Table 4):* models trained on IEEE 118 are evaluated on the unseen IEEE 57 and GOC 500 grids. GENCO has the lowest error among the neural solvers, but all degrade substantially: its mean power-balance loss is about equal to the mean apparent power on IEEE 57 and about 14 times larger on GOC 500.
- *Section 5.5.3, Fig. 15:* the pretrained subgrid model, used zero-shot on IEEE 118, has a mean active residual of 11.07 MW against 2.30 MW for DC-PF.
- *Same section:* fine-tuning that model with about 1,000 grid-specific samples already beats DC-PF (1.93 MW against 2.30 MW), while training from scratch on the same samples does not (3.81 MW). The paper concludes that zero-shot transfer remains unsolved.

These are the same model or a close analogue, so they serve as a cross-check of our pipeline:

- The released subgrid-pretrained model on IEEE 118 gives **9.6 MW** (normaliser variant B) and **11.7 MW** (variant C) against a DC-PF residual of **2.34 MW**, to be compared with 11.07 MW and 2.30 MW in the paper. We do not know which normaliser variant the paper used.
- Our Base checkpoints trained on IEEE 118 give a median relative residual of about 2.5 on IEEE 57 and about 9 on GOC 500, against about 1 and about 14 in the paper. These are different checkpoints (the paper's Task 3.1 models are trained on its PF∆ benchmark dataset, ours on the released datakit data), so we read this as the same order of magnitude and pattern, not as a replication, therefore confirming the zero-shot transfer claim.

However, the paper does not report the following, and the released checkpoints let us measure them without training:

1. A full **source × target matrix** over 7 source grids (IEEE 14, 30, 57, 118; GOC 500, 2000, 10000), three sizes and several seeds, on 5 target grids: 585 evaluations in total, each compared with DC-PF on the same test split.
2. A test of whether **lower in-distribution error goes with worse zero-shot transfer** (hypothesis H1), with uncertainty estimates and a check of how much the result depends on the source grid.
3. How **source-grid size and diversity** relate to zero-shot error, including the subgrid-pretrained model on the same axes as the single-grid checkpoints (H2).
4. The effect of **normaliser statistics** under three variants (H3).
5. **Zero-shot voltage-magnitude boundary violation rates and calibration slopes**. The paper's voltage-violation results are in-distribution.
6. **Public code, configs, results and provenance** to re-run all of it on the released checkpoints.

## Setup

- **Checkpoints:** the 53 single-grid power-flow checkpoints of `gridfm/genco-pf-datakit` (IEEE 14, 30, 57, 118 and GOC 500: Base, Small, Tiny, seeds 0, 1, 42; GOC 2000 and 10000: Small and Tiny, two seeds each), plus the subgrid-pretrained Base model of `gridfm/genco-pf-transfer/pretraining`, converted from the PreTraining to the PowerFlow task.
- **Targets:** `case14_ieee`, `case30_ieee`, `case57_ieee`, `case118_ieee`, `case500_goc`. Grids 2000 and 10000 are sources only.
- **Evaluation sample:** the first 20,000 scenarios of each target (10,000 for case500), evaluated on the 10 % test split: 2,002 scenarios for 14, 30 and 118, about 2,000 for 57 and about 990 for 500. In all zero-shot pairs every checkpoint sees the identical test set on a given target.
- **Normaliser variants:** **A** the checkpoint's own statistics (nothing from the target); **B** `baseMVA` and `vn_kv_max` recomputed from target inputs only (Pd, Qd, vn_kv); **C** graphkit's default refit, which uses labels (Pg, Qg), as an upper reference on the source grid and for the subgrid model. The subgrid model ships no statistics, so it is run under B and C only.
- **Metric:** relative active residual = mean active power-balance residual ÷ mean bus apparent power of the target grid (dimensionless). The in-distribution reference is each checkpoint's shipped `metrics.csv` over the same scale. We also report the DC-PF residual on the same test split, the Vm boundary violation rate, and the calibration slope of predicted against true values.
- **Total:** 585 evaluations (265 variant A, 270 B, 50 C) on one NVIDIA L40S, graphkit commit `f104900`, torch 2.11.0 + CUDA 13.0.
- **Reproduction gate:** `case118_ieee/small/seed0` on its own grid gives 0.073 MW active and 0.0106 MVar reactive residual on our 2,002-scenario subset, against 0.125 MW and 0.0136 MVar shipped. This is the same range, not an exact match (different test split). The agreement with the paper's zero-shot numbers above is a second, independent check.

## Results

### A1. Transfer matrix

![Transfer matrix](https://raw.githubusercontent.com/itsantilas89/genco-zero-shot-topology-transfer-audit/main/figures/fig1_transfer_matrix.png)

In-distribution the median relative residual is 0.0013 (range 1.2e-4 to 0.11); zero-shot it is **0.94** (IQR 0.46 to 2.5), roughly 700 times higher. At that level the mean power mismatch per bus is of the order of the load itself. The median GENCO / DC-PF residual ratio is **13.8**, and 4 of 225 zero-shot runs have a residual below DC-PF, all on the 14-bus target. Sources of 118 buses or fewer have a median zero-shot residual of 2.8 against 0.64 for sources of 500 buses or more, and transfer from small grids to case500 is the worst case (3 to 11).

### A2. Topology overfitting (H1)

![In-distribution vs zero-shot](https://raw.githubusercontent.com/itsantilas89/genco-zero-shot-topology-transfer-audit/main/figures/fig2_h1_id_vs_zero_shot.png)

Across the 53 single-grid checkpoints, the Spearman correlation between in-distribution and mean zero-shot relative residual is **ρ = -0.53** (95 % CI -0.71 to -0.28 resampling checkpoints; permutation p < 0.001): checkpoints with lower in-distribution error tend to transfer worse. The evidence is weaker than that interval suggests:

- The 53 checkpoints come from only 7 source grids. Resampling source grids instead gives a 95 % CI of **-0.75 to +0.01**. Leaving out any one source grid keeps ρ between -0.64 and -0.33, and the correlation of the 7 per-grid medians is -0.64.
- It is mostly a between-grid effect: the small grids have both the lowest in-distribution error and the worst transfer. Within one source grid (9 checkpoints each) the intervals mostly include zero, and for GOC 500 the sign reverses (ρ = +0.75).
- By size, ρ is -0.32 for Base (n = 15, CI -0.74 to +0.36), -0.68 for Small (n = 19, CI -0.86 to -0.27) and -0.45 for Tiny (n = 19, CI -0.77 to +0.02).

We therefore read this as a negative association that is consistent with H1 but not firmly established, and we do not claim that better-fitted checkpoints of the same grid overfit more. The correlation with the degradation ratio (ρ = -0.94) is negative by construction, since the in-distribution residual is its denominator, so we do not use it as evidence.

### A3. Source diversity and scale (H2)

![Source size and subgrid model](https://raw.githubusercontent.com/itsantilas89/genco-zero-shot-topology-transfer-audit/main/figures/fig3_source_scale_and_subgrid.png)

The subgrid-pretrained model (100 topologies) has a median zero-shot residual of **0.50** over its five target runs (mean **0.41**), against 0.98 for the single-grid runs pooled. Against the Base single-grid checkpoints of equal size it does better in 18 of 20 (source, target) pairs (these pairs are not independent) and on 4 of 5 targets against the best single source; the exception is case14, where the case30 source is better. Its Vm violation rate is 0.4 % (median) against 13 % for single-grid checkpoints and 2.9 % in the ground truth. Its residual is still 3 to 9 times the DC-PF residual.

This does not separate diversity from scale. Per checkpoint, averaging the residual over the targets, the Small checkpoints trained on one 2,000-bus or one 10,000-bus grid reach 0.50 and 0.57 (median over seeds), and the Small GOC 500 checkpoint 0.58, close to the subgrid model's 0.41. No Base checkpoint exists for the 2,000- and 10,000-bus grids, so the capacity-matched comparison covers the five smaller sources only. The subgrid result is a single checkpoint, so it carries no seed uncertainty.

### A4. Normaliser recalibration (H3)

![Recalibration gain](https://raw.githubusercontent.com/itsantilas89/genco-zero-shot-topology-transfer-audit/main/figures/fig4_recalibration_gain.png)

Replacing the source statistics (A) by target-input statistics (B) does **not** help zero-shot: the median log2 gain is 0.00 (95 % CI -0.02 to +0.02) and 49 % of the 220 pairs improve (CI 42 % to 55 %; checkpoint-clustered bootstrap). On the checkpoint's own grid, B is about 2,000 times worse than A (median log2 gain -11), while the full refit C matches A. On the subgrid model B and C give similar residuals (0.50 and 0.52 median). So this particular input-only recalibration is not a faithful stand-in for the original statistics. A possible explanation is that it leaves Pg and Qg out of the `baseMVA` fit, which the models may depend on, and that zero-shot errors are too large for the scale shift to matter; we did not test either. We read this as: H3 is not supported for this variant, and the data say nothing about other recalibrations.

### A5. Safety

![Safety](https://raw.githubusercontent.com/itsantilas89/genco-zero-shot-topology-transfer-audit/main/figures/fig5_safety.png)

The median zero-shot Vm violation rate is 13 % for single-grid checkpoints against 2.9 % in the ground truth, with some runs above 80 %. 33 of 225 runs combine a below-median residual (or one below DC-PF) with a collapsed calibration slope (|slope| < 0.3 for Vm or Va). Such a pattern can indicate near-mean predictions, so a low residual alone was not a reliable indicator in these runs.

## Limitations

- No retraining and no fine-tuning; every result is for the released checkpoints used zero-shot.
- Evaluation uses about 2,000 test scenarios per target (about 990 for case500), and some strata have only 4 checkpoints. The checkpoints come from 7 source grids, which limits what can be concluded about H1 and H2.
- Grids 2000 and 10000 are sources only, and the subgrid model is a single checkpoint.
- Source grid size and family (IEEE or GOC) are confounded with in-distribution error, so H1 and H2 are correlational.
- Variant C uses labels and is an upper reference only; variant B is one specific recalibration.
- The in-distribution reference comes from the shipped `metrics.csv`. The reproduction gate matches in range only, and the cross-check with the paper's numbers is for two cells.

## Practical recommendation

These follow from the zero-shot results only; we did not test fine-tuning.

1. Do not rely on zero-shot output on a grid the checkpoint was not trained on: expect residuals well above DC-PF, and compare against DC-PF on your own data.
2. Check the Vm boundary violation rate and the calibration slope of predicted against true values; a low residual with a collapsed slope can indicate mean prediction.
3. Keep the checkpoint's own normaliser statistics: recomputing them from target inputs (variant B) did not help zero-shot and degraded the model on its own grid.
4. If you adapt a model, the subgrid-pretrained model and checkpoints trained on large grids transferred best zero-shot here; the GENCO paper reports that fine-tuning the subgrid model helps, which we did not test.

## References

- Code, configs, results and figures: https://github.com/itsantilas89/genco-zero-shot-topology-transfer-audit (`results/results.csv`, `results/summary.json`, `figures/`)
- Reproduction: `scripts/setup_remote.sh` then `scripts/run_all.sh`; commit hashes and dataset revisions in `results/provenance.json`
- Checkpoints: [gridfm/genco-pf-datakit](https://huggingface.co/gridfm/genco-pf-datakit), [gridfm/genco-pf-transfer](https://huggingface.co/gridfm/genco-pf-transfer)
- [GENCO](https://arxiv.org/pdf/2608.09921) · [MxGPS](https://arxiv.org/pdf/2607.13763)
