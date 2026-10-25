# GENCO zero-shot topology transfer audit

How do the released GENCO power-flow checkpoints behave on grids they were never trained on, and what drives the degradation? No model is trained: every number comes from public checkpoints, evaluated through [`gridfm-graphkit`](https://github.com/gridfm/gridfm-graphkit) on data produced by [`gridfm-datakit`](https://github.com/gridfm/gridfm-datakit).

- **H1, topology overfitting:** checkpoints with the lowest in-distribution residual degrade the most on unseen grids.
- **H2, source diversity:** the subgrid-pretrained checkpoint (100 topologies) transfers better than single-grid checkpoints of the same size.
- **H3, normaliser mismatch:** part of the zero-shot gap comes from per-grid normalisation statistics, and recalibrating them on target inputs recovers part of it.

The full project plan, with the reasoning behind each step, is in [PLAN.md](PLAN.md). The differences between the plan and this implementation are listed under [Deviations](#deviations-from-the-plan).

**Status:** the full matrix has been run (585 evaluations, no failures, reproduction gate passed). Results are summarised [below](#results); the Community post draft is in [POST.md](POST.md).

---

## Setup (GPU server)

You need Linux, an NVIDIA driver whose CUDA version (top right of `nvidia-smi`) is at least 12.6, `git`, Python 3.10–3.12, about 15 GB of free disk and internet access to github.com, huggingface.co, download.pytorch.org and data.pyg.org.

```bash
git clone <URL-OF-THIS-REPO> genco-zero-shot-topology-transfer
cd genco-zero-shot-topology-transfer
conda create -n <venv> python=3.10
conda activate <venv>
CUDA=cu130 bash ./scripts/setup_remote.sh       # this will take a while to setup
```

`setup_remote.sh` creates `.venv`, installs torch 2.11.0 for the chosen CUDA flavour (`cu126`, `cu128`, `cu130`, or `cpu`) with the prebuilt `torch-scatter` wheel, clones `gridfm-graphkit` at the pinned commit `f104900`, installs it editable, checks that the CUDA scatter kernels work, and writes `requirements.lock` and `results/environment.txt` (GPU, driver, CUDA, Python). If the sanity check fails, nothing downstream is worth running.

> **Note:** Ignore the following error, if appeared:
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gridfm-graphkit 0.9.0.3 requires torchaudio>=2.10, which is not installed.
gridfm-graphkit 0.9.0.3 requires torchvision>=0.25, which is not installed.
```
You can ignore the warning by running:
```bash
pip install torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

Why torch 2.11.0: PyG publishes no `torch-scatter` wheel for torch 2.13 and only some CUDA flavours for 2.12, while compiling it needs a matching `nvcc`. Torch 2.11.0 has a wheel on every flavour and is inside graphkit's `torch>=2.10,<2.14` range.

## Run

```bash
tmux new -s <venv>                      # the full run takes hours; keep it alive if you disconnect
conda activate <venv>                    # tmux starts a fresh shell, so activate again inside it
JOBS=4 bash ./scripts/run_all.sh 2>&1 | tee run_all.log
```

`run_all.sh` is resumable. Finished matrix runs are skipped, so after an interruption or a fix, run the same command again. It performs, in order:

1. `download_checkpoints.py`: the released checkpoints (about 2 GB) and their Hugging Face revisions.
2. `build_manifest.py`: `manifest.csv`, one row per checkpoint, with a check that all configs share the model dimensions.
3. `download_data.py`: the target datasets (the first K partitions per grid, see `settings.yaml`).
4. `recalibrate_normalizer.py`: normaliser variant B statistics per target.
5. `make_configs.py`: one evaluation YAML per (checkpoint, target) pair, about 270 files.
6. **`repro_gate.py`**: evaluates `case118_ieee/small/seed0` on its own grid and compares with the shipped `metrics.csv`. The script stops here if it does not reproduce. Do not override this; it is the guard against silent setup errors.
7. `run_matrix.py --jobs $JOBS`: about 580 evaluations (variants A and B for every pair, C on the diagonal and for the subgrid model).
8. `postprocess.py collect`, `analyses.py`, `figures.py`, `finalize.py`: result tables, analyses A1–A6, figures, provenance and checksums.

Useful variations:

```bash
python scripts/repro_gate.py                                   # just the gate
python scripts/run_matrix.py --dry-run                         # list pending runs
python scripts/run_matrix.py --targets case14_ieee --jobs 2    # one target first (smoke test)
python scripts/run_matrix.py --limit 5                         # first five pending runs only
python scripts/run_matrix.py --retry-failed                    # after fixing a failure
python scripts/run_matrix.py --gpus 0,1 --jobs 8               # spread runs over two GPUs
SKIP_GATE=1 JOBS=8 bash scripts/run_all.sh                     # re-run after the gate passed
```

`--jobs` is the number of evaluation processes sharing the GPU(s). Each process holds a small model and a data loader, so 4–8 per GPU is a reasonable start; lower `run.num_workers` in `settings.yaml` if the CPU is the bottleneck. The first run per target is executed alone because it builds the processed-dataset cache that the others reuse.

I did not time the matrix on a GPU. On a laptop CPU one evaluation of a 2,000-scenario case14 slice takes about 35 s.

### Bring the results back

```bash
git add results figures manifest.csv configs normalizer_stats_target requirements.lock run_all.log
git commit -m "Results of the full matrix"
git push
```

`results/per_run/` (raw per-run JSON), `runs/`, the datasets and the checkpoints are git-ignored. `results/results.csv` carries everything from the per-run files.

Check `results/failures.csv` first: it lists every failed run with the error text. An empty or missing file means all runs finished.

---

## What the scripts do

| Script | Purpose |
|---|---|
| `settings.yaml` | Pinned commit, HF repos, targets and partition counts, evaluation seed, gate expectations |
| `scripts/common.py` | Paths and helpers |
| `scripts/download_checkpoints.py` | Checkpoints from `gridfm/genco-pf-datakit` and `gridfm/genco-pf-transfer/pretraining` |
| `scripts/build_manifest.py` | `manifest.csv` (54 checkpoints: 53 single-grid plus the subgrid model) |
| `scripts/download_data.py` | Target datasets, contiguity check, dataset revisions |
| `scripts/recalibrate_normalizer.py` | Normaliser variants A, B and the target-keyed stats files |
| `scripts/make_configs.py` | Evaluation YAMLs |
| `scripts/repro_gate.py` | Reproduction gate |
| `scripts/run_matrix.py` | Resumable, parallel evaluation matrix; failures to `results/failures.csv` |
| `scripts/postprocess.py` | Per-run metrics from predictions; `collect` writes `results_raw.csv` and `results.csv` |
| `scripts/analyses.py` | A1–A6 to `results/analyses/*.csv` and `results/summary.json` |
| `scripts/figures.py` | Figures 1–5 to `figures/` (PNG and PDF) |
| `scripts/finalize.py` | `results/provenance.json` and `results/SHA256SUMS` |
| `scripts/synthetic_results.py` | Random results with the real schema, to test `analyses.py` and `figures.py` |
| `reference/mxgps_case14.csv` | Published MxGPS zero-shot numbers on the 14-bus case (Table IV of arXiv:2607.13763), for A6 |

### Normaliser variants

| Variant | Statistics | Labels used | Runs for |
|---|---|---|---|
| **A** | the checkpoint's own `normalizer_stats.pt` | none | every pair, except the subgrid model (it ships none) |
| **B** | `baseMVA` from target Pd and Qd only, `vn_kv_max` from target `vn_kv` | none | every pair |
| **C** | graphkit's default refit, which includes Pg and Qg | yes (upper reference) | diagonal pairs and the subgrid model |

The zero-shot reference for single-grid checkpoints is variant A (nothing from the target is used); for the subgrid model it is B.

### Metrics

Per run: the GENCO active and reactive residuals and PBE (from graphkit), the DC-PF and AC ground-truth residuals on the same test split, Vm and Va MAE per bus type, Vm boundary violation rate (and the same rate for the ground truth, as a baseline), calibration slope and R² of predicted against true Vm and Va, and wall time. The relative active residual is the mean active residual divided by the mean bus apparent power of the target grid, `mean(sqrt(Pd² + Qd²))` over buses and scenarios.

---

## Deviations from the plan

Found while reading graphkit at `f104900` and testing the pipeline.

1. **Normaliser stats must be keyed by the target grid.** `--normalizer_stats` takes a dict keyed by network name, and graphkit only uses it when the evaluated network's name is a key. A source checkpoint's stats file is keyed by its source grid, so passing it for another target makes graphkit silently refit on the target (the plan, section 6.1, assumed it skips the fit). Variant A therefore re-keys the source stats to the target, and the variant B files in `normalizer_stats_target/` are written as `{grid: {...}}`.
2. **`runtime_data.parquet` is needed.** `--compute_dc_ac_metrics` reads it; the table list in the plan omits it.
3. **Evaluation size.** Every target uses the first K partitions (`settings.yaml`: 100 partitions = 20,000 scenarios, so about 2,000 test scenarios, for 14, 30, 57 and 118; 50 partitions for case500), not the full datasets for 14, 30 and 57. This keeps the evaluation sample fixed and the runtime bounded. Increase `partitions`, or set it to `null`, to use more.
4. **Split seed.** Off-diagonal pairs use `eval_seed` (0) for the split, so every checkpoint is evaluated on the identical test set of a target. The plan kept the source seed, which would give seed-0, seed-1 and seed-42 checkpoints different test sets on the same target. Diagonal pairs keep the source seed.
5. **In-distribution reference.** A2 and A1's diagonal use each checkpoint's shipped `metrics.csv`, divided by the mean bus apparent power of its source grid. The diagonal runs are still executed (they check the pipeline and give variant B and C on the source grid), but on a partition subset their test split overlaps the original training data, so they are not used as the in-distribution reference. Grids 2000 and 10000 get their scale from one downloaded partition of `bus_data` (`scale_only` in `settings.yaml`).
6. **H1 statistic.** Besides Spearman ρ with a bootstrap CI between in-distribution residual and the degradation ratio (zero-shot ÷ in-distribution), A2 reports ρ against the zero-shot residual itself and a permutation p-value. The ratio contains the in-distribution residual in its denominator, so it correlates negatively with it even for unrelated models; the permutation null accounts for that.
7. **Predictions are not kept.** After each run the metrics are extracted and the prediction files are deleted, which keeps disk use small. `--keep-predictions` retains them.
8. **Not implemented:** the optional per-bus residual map (figure 6) and the hybrid warm-start extension (section 11).

## Limitations to carry into the write-up

No retraining; partition subsets of the target data; grids 2000 and 10000 are sources only; variant C uses labels; the MxGPS comparison is indicative only (different training data, sources and metric protocol); one subgrid-pretrained checkpoint, so no seed uncertainty on that side of H2.

## Results

585 evaluations (265 variant A, 270 B, 50 C) of 54 checkpoints on 5 targets, about 2,000 test scenarios per target (about 990 for case500), on one NVIDIA L40S with graphkit `f104900` and torch 2.11.0 + CUDA 13.0. The reproduction gate passed in the same range, not exactly: `case118_ieee/small/seed0` gives 0.073 MW / 0.0106 MVar against 0.125 MW / 0.0136 MVar shipped. The numbers below are from `results/summary.json` and `results/analyses/`; the full write-up with figures is [POST.md](POST.md).

| Question | Result |
|---|---|
| Zero-shot vs in-distribution | Median relative active residual 0.94 zero-shot against 0.0013 in-distribution (about 700 times higher). Median GENCO / DC-PF residual ratio 13.8; 4 of 225 zero-shot runs beat DC-PF, all on the 14-bus target. |
| **H1** topology overfitting | Spearman ρ(in-distribution, zero-shot residual) = -0.53, 95 % CI [-0.71, -0.28], permutation p < 0.001, n = 53. Mostly a between-grid effect: within one source grid the intervals mostly include 0 (GOC 500: ρ = +0.75). |
| **H2** source diversity | Subgrid-pretrained model: median zero-shot residual 0.50 against 0.98 for single-grid checkpoints; better than the equal-size Base single-grid sources in 18 of 20 (source, target) pairs and than the best single source on 4 of 5 targets. One checkpoint, no seed uncertainty. |
| **H3** normaliser mismatch | Not supported: variant B vs A median log2 gain 0.00, 95 % CI [-0.02, 0.02], 49 % of 220 pairs improve. On the source grid B is about 2,000 times worse than A; C matches A. |
| Safety | Median Vm violation rate 13 % (single-grid) against 2.9 % in the ground truth; 33 of 225 runs have low residual with collapsed calibration. |
| MxGPS (indicative) | 14-bus zero-shot Vm MAE: GENCO single-grid 0.018, subgrid model 0.004, MxGPS 0.035 to 0.038 p.u.; protocols differ. |

Figures: `figures/fig1_transfer_matrix`, `fig2_h1_id_vs_zero_shot` (headline), `fig3_source_scale_and_subgrid`, `fig4_recalibration_gain`, `fig5_safety` (PNG and PDF).

## Repository layout

```
README.md  PLAN.md  POST.md  LICENSE  settings.yaml  requirements.txt  requirements.lock  manifest.csv
scripts/                    # pipeline (see table above) and setup_remote.sh / run_all.sh
configs/                    # generated evaluation YAMLs
normalizer_stats_target/    # variant B statistics, one file per target
reference/                  # published numbers used in A6
results/                    # results_raw.csv results.csv failures.csv summary.json analyses/ provenance.json SHA256SUMS
figures/
```

Weights and datasets are not redistributed. They come from [`gridfm/genco-pf-datakit`](https://huggingface.co/gridfm/genco-pf-datakit), [`gridfm/genco-pf-transfer`](https://huggingface.co/gridfm/genco-pf-transfer) (`pretraining/` only) and the `gridfm/pf_small_<grid>` datasets; the exact revisions are in `results/provenance.json` after a run.

## Links

- [GENCO: a unified neural solver embedded in a development framework for steady-state grid analysis](https://arxiv.org/pdf/2608.09921)
- [GridFM Datakit](https://github.com/gridfm/gridfm-datakit) · [GridFM Graphkit](https://github.com/gridfm/gridfm-graphkit) · [GridFM GENCO paper repo](https://github.com/gridfm/gridfm-paper-genco)
- [MxGPS: multiplex graph transformers for a power grid foundation model](https://arxiv.org/pdf/2607.13763)

Licensed under Apache-2.0 (see [LICENSE](LICENSE)).
