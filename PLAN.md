> This is the original project plan. Setup and run instructions are in [README.md](README.md),
> which also lists where the implementation deviates from this plan.

# GENCO Zero-Shot Topology Transfer Audit: Workroad

## 0. Scope and framing

- **Question:** how do released GENCO power-flow checkpoints behave on grids they were never trained on, and what drives the degradation?
- **Constraint:** no model training. All results come from public checkpoints, evaluated through `gridfm-graphkit`, on data produced by `gridfm-datakit`.
- **MxGPS's role:** it is cited as the source of one hypothesis, topology overfitting: models with the lowest in-distribution error degrade the most on unseen topologies. It is not a competitor and is not benchmarked head-to-head.
- **Hypotheses:**
  - **H1, topology overfitting:** across checkpoints, in-distribution residual is negatively correlated with zero-shot robustness.
  - **H2, source diversity:** the subgrid-pretrained checkpoint, trained on 100 topologies, transfers better than any checkpoint trained on a single grid of similar capacity.
  - **H3, normaliser mismatch:** part of the zero-shot gap comes from per-grid normalisation statistics, and recalibrating them on the target grid without labels recovers part of it.
- **Done when:** the transfer matrix, the three hypothesis analyses, the figures, a public repo and a Community-tab post exist.

### Useful Links
- [GENCO - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis](https://arxiv.org/pdf/2608.09921)
- [GridFM Datakit](https://github.com/gridfm/gridfm-datakit)
- [GridFM Graphkit](https://github.com/gridfm/gridfm-graphkit)
- [GridFM GENCO Paper](https://github.com/gridfm/gridfm-paper-genco)
- [MxGPS: Multiplex Graph Transformers for a Power Grid Foundation Model](https://arxiv.org/pdf/2607.13763)

## 1. Environment

1. Clone `gridfm-graphkit`.
   - Pin it to commit `f104900`, the last commit where `main` matched the paper branch.
   - Fall back to the `genco-paper-repro` branch if the checkpoints fail to load.
2. Install the package and its extra dependencies:
   - `pip install -e .`
   - `torch-scatter` and `torch-sparse` matched to the torch/CUDA build.
   - `huggingface_hub[cli]`.
3. Set `MLFLOW_ALLOW_FILE_STORE=true`.
4. Freeze the environment with `pip freeze > requirements.lock`, and record the GPU model, driver and CUDA version.

## 2. Checkpoint inventory

1. Download all of `gridfm/genco-pf-datakit`.
   - Grids 14, 30, 57, 118 and 500: Base, Small and Tiny, seeds 0, 1 and 42.
   - Grids 2000 and 10000: Small and Tiny only, two seeds each.
2. Download `gridfm/genco-pf-transfer/pretraining/`.
   - This is a Base-size model trained with the `PreTraining` task.
   - It ships with no `normalizer_stats.pt`.
3. Build `manifest.csv` with these columns:

   | Column | Content |
   |---|---|
   | `ckpt_id` | unique identifier |
   | `source_grid` | grid the checkpoint was trained on |
   | `size` | Base, Small or Tiny |
   | `seed` | training seed |
   | `weights_path` | checkpoint file; `last.pt` for 10000 |
   | `config_path` | training YAML |
   | `normalizer_stats_path` | stats file, or empty |
   | `id_active_res_MW`, `id_reactive_res_MVar`, `id_pbe_mean` | in-distribution reference values, copied from each `metrics.csv` |

4. Check that every config shares the same dimensions: `input_bus_dim=15`, `input_gen_dim=6`, `edge_dim=10`, `num_layers=12`. Only `hidden_size` should differ. A mismatch means a checkpoint cannot be applied to other grids.

## 3. Target data

1. Targets: `case14_ieee`, `case30_ieee`, `case57_ieee`, `case118_ieee` and `case500_goc`.
2. Do not use 2000 or 10000 as targets; their datasets are 100 GB or more. They stay in the study as sources only.
3. For 14, 30 and 57, download the full `gridfm/pf_small_<grid>` datasets into `data/<grid>/raw/`.
4. For 118 and 500, download the same `scenario_partition=*` subset for every table:
   - Tables: `bus_data.parquet`, `gen_data.parquet`, `branch_data.parquet`, `y_bus_data.parquet`.
   - Include `n_scenarios.txt` and `stats.parquet`.
   - Use `hf download --repo-type dataset --include "<table>/scenario_partition=<k>/*"` for each table and partition.
5. Check that the downloaded scenario IDs are contiguous from 0, then set `data.scenarios` to the number of scenarios actually present.
6. Fix one evaluation sample size per target. Report it, and keep it identical across all checkpoints.

## 4. Reproduction gate

1. Evaluate `case118_ieee/small/seed0` on `case118_ieee` using its own config and normaliser stats:

   ```bash
   gridfm_graphkit evaluate \
     --config_conversion auto_inline \
     --config genco-pf-datakit/case118_ieee/small/seed0/case118_ieee_small_seed0.yaml \
     --data_path data \
     --model_path genco-pf-datakit/case118_ieee/small/seed0/best_model_state_dict.pt \
     --normalizer_stats genco-pf-datakit/case118_ieee/small/seed0/normalizer_stats.pt \
     --compute_dc_ac_metrics --save_output
   ```

2. Compare against `metrics.csv`. Expected: average active residual ≈ 0.125 MW, average reactive residual ≈ 0.014 MVar.
3. If the full dataset was not downloaded, repeat the check on the partition subset. The numbers should land in the same range.
4. **Gate:** do not continue until this reproduces. Most later failures would trace back to a mismatch here.

## 5. Config generator

1. Write `make_configs.py`. For each `(checkpoint, target)` pair, it copies the source YAML and overrides:
   - `data.networks: [<target>]` and `data.scenarios: [<N_target>]`.
   - `test_ratio`, `val_ratio`, `split_by_load_scenario_idx: true` and `seed`, kept as in the source config. This makes the diagonal entry, where target equals source, the original held-out test split.
   - `data.mask_ref_vm: true`, which all pre-October-2026 checkpoints need.
2. For the subgrid-pretrained checkpoint:
   - Set `task.task_name: PowerFlow` and drop the pretraining-only keys.
   - Keep `hidden_size: 48` and the other model keys.
3. Output: `configs/<ckpt_id>__<target>.yaml`, about 270 files.

## 6. Normaliser handling

The normaliser's `fit_on_train` statistics depend on the grid:

- `baseMVA` is the 95th percentile of non-zero Pd, Qd, Pg and Qg.
- `vn_kv_max` is the maximum nominal voltage.

Each zero-shot pair is evaluated under up to three variants:

| Variant | Statistics | Labels used |
|---|---|---|
| **A: source** | the checkpoint's own `normalizer_stats.pt` | none |
| **B: target, inputs only** | `baseMVA` from target Pd and Qd only (95th percentile of non-zero values); `vn_kv_max` from target `vn_kv`. Write the dict `{baseMVA_orig, baseMVA, vn_kv_max}` to `normalizer_stats_target/<grid>.pt` | none |
| **C: target, full refit** | the default fit, which includes Pg and Qg | Qg, which is an output; this is an upper reference only |

1. Confirm that passing `--normalizer_stats` skips `fit()`. If it does not, the contiguous-scenario assertion in `HeteroDataMVANormalizer.fit` will fire on partition subsets.
2. The subgrid-pretrained checkpoint has no stats file, so it runs only under variants B and C. State this in the write-up.

## 7. Matrix runs

1. Write `run_matrix.py`. It iterates over `manifest.csv` × targets × normaliser variants (A, B; C on the diagonal and the subgrid model only), and calls:

   ```bash
   gridfm_graphkit evaluate --config <cfg> --model_path <w> --normalizer_stats <stats> \
     --data_path data --compute_dc_ac_metrics --save_output \
     --exp_name zs_audit --run_name <ckpt_id>__<target>__<variant>
   ```

2. Make it resumable: skip runs that already have results, and log failures to `failures.csv` together with the error text.
3. Collect results into `results_raw.csv`:
   - From MLflow: active and reactive residuals, PBE mean and max, the DC-PF residual reference, and runtime.
   - Plus the run metadata from the manifest.
4. Smoke-test on the 14 and 30 targets before launching 118 and 500.

## 8. Metrics from saved predictions

From each `<grid>_predictions.parquet`, compute:

1. Vm MAE and Va MAE, split by bus type (PQ, PV, REF).
2. Boundary violation rate: the fraction of predicted Vm outside `[Vmin, Vmax]`.
3. Calibration: slope and R² of predicted against true, for Vm and Va.
4. Relative active residual: the mean residual divided by the mean bus apparent power on that grid. This makes the numbers comparable across grids and matches the normalisation in the GENCO paper.
5. GENCO / DC-PF residual ratio. A value below 1 means GENCO beats DC-PF even zero-shot.

Write the results to `results.csv`, with one row per (checkpoint, target, variant).

## 9. Analyses

1. **A1, transfer matrix.**
   - Heatmap of source grid × target grid, coloured by the median over seeds of the relative active residual. One panel per size: Base, Small, Tiny.
   - Mark the cells where GENCO beats DC-PF.
2. **A2, topology overfitting (H1).**
   - For each checkpoint: in-distribution relative residual against mean zero-shot relative residual, and against the degradation ratio (zero-shot ÷ in-distribution).
   - Spearman ρ with a bootstrap 95% CI over checkpoints.
   - Repeat stratified by size and by source grid.
   - Report the result whichever way it comes out.
3. **A3, source diversity and scale (H2).**
   - Zero-shot residual against source grid size.
   - Place the subgrid-pretrained model on the same axes.
   - Compare models of equal capacity: Base single-grid against Base subgrid-pretrained.
4. **A4, normaliser recalibration (H3).**
   - Per pair: the gain from variant A to variant B. Show the distribution and the fraction of pairs that improve.
   - Show the gap from variant B to variant C to separate how much comes from input-only versus label-informed statistics.
5. **A5, safety.**
   - Zero-shot boundary violation rate and calibration slope for each pair.
   - Flag pairs with low residual but collapsed calibration (slope near 0), which point to mean prediction.
6. **A6, MxGPS reference.**
   - One sentence and one small table: GENCO zero-shot on `case14_ieee` against the published MxGPS zero-shot numbers on case14 (Vm MAE, violation rate).
   - Label it indicative only, because the training data, sources and metric protocol all differ.

## 10. Figures

1. The transfer-matrix heatmaps from A1.
2. The in-distribution vs zero-shot scatter with ρ and CI from A2. This is the headline figure.
3. Zero-shot residual against source size, including the subgrid model, from A3.
4. The recalibration gain distribution from A4.
5. The violation-rate and calibration panel from A5.
6. Optional: a per-bus residual map for one bad zero-shot pair, showing where errors concentrate (slack-adjacent buses, PV buses, long branches).

Every figure needs labelled units, a stated sample size and a light/dark-safe palette.

## 11. Optional extension: hybrid warm start

Only start this if every step above is complete.

1. Build a Ybus Newton–Raphson solver from `branch_data.parquet` and `y_bus_data.parquet`, and check it against the datakit AC solution.
2. For zero-shot pairs, compare three initialisations: flat start, DC start, and GENCO zero-shot output.
3. Report iterations to convergence, convergence rate and wall time.
4. This converts zero-shot error into a practical result for the "hybrid solvers" direction.

## 12. Release

1. Repository layout:

   ```
   README.md                 # question, results summary, exact reproduction commands
   requirements.lock
   manifest.csv
   scripts/  make_configs.py  run_matrix.py  recalibrate_normalizer.py  postprocess.py  analyses.py  figures.py
   configs/                  # generated YAMLs
   normalizer_stats_target/  # variant-B stats
   results/  results_raw.csv  results.csv  failures.csv
   figures/
   LICENSE                   # Apache-2.0
   ```

2. Do not redistribute the weights. Link the Hugging Face repos with the exact paths used.
3. Record the graphkit commit, the dataset revisions (HF commit hashes) and the scenario partitions used.
4. Add SHA-256 checksums for the result CSVs.
5. Make one command reproduce everything from the downloads: `python scripts/run_matrix.py && python scripts/analyses.py`.

## 13. Write-up for the Community tab

1. **Title**, for example: *Where does GENCO break? A zero-shot topology-transfer audit of the released checkpoints*.
2. **Motivation:** the GENCO paper names zero-shot transfer as an open problem; prior work (MxGPS) proposes topology overfitting as the mechanism; this entry tests it on GENCO.
3. **Setup:** the checkpoints, targets, normaliser variants, metrics and sample sizes.
4. **Results:** A1–A5, each with a figure and one quantitative claim.
5. **Limitations:**
   - No retraining.
   - Partition subsets on 118 and 500.
   - The 2000 and 10000 grids are sources only.
   - Label leakage in variant C.
   - The MxGPS comparison is indicative.
6. **Practical recommendation:** what a GENCO user should do before deploying on a new grid. Examples: recalibrate the normaliser, prefer the subgrid-pretrained model as a fine-tuning start, check the violation rate.
7. **Links:** repo, results CSV, figures.
8. **Contact:** email, team member names and affiliations.

## 14. Submission checklist

- [ ] Reproduction gate passed and documented
- [ ] `results.csv` complete; failures listed with causes
- [ ] H1, H2 and H3 each answered with a number and an uncertainty estimate
- [ ] All figures render; units and sample sizes stated
- [ ] Repo public, README reproduces end to end on a fresh environment
- [ ] Post contains a title, write-up, visuals, links, contact email, names and affiliations
- [ ] Posted to `huggingface.co/spaces/gridfm/genco-challenge/discussions` before end of 25 October AoE

## Descoping order

If time runs short, drop these in order:

1. §11, the hybrid warm start.
2. Variant C.
3. Seed 42, keeping seeds 0 and 1.
4. Tiny checkpoints.
5. The 500 target.

Never drop the reproduction gate, A2, A4 or the public repo.