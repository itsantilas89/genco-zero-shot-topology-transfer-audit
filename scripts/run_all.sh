#!/usr/bin/env bash
# Whole pipeline, resumable: re-running skips finished steps and finished matrix runs.
#   JOBS=4   parallel evaluation processes sharing the GPU(s) (default 4)
#   SKIP_GATE=1 to skip the reproduction gate on a re-run
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_ALLOW_FILE_STORE=true
JOBS="${JOBS:-4}"

python scripts/download_checkpoints.py
python scripts/build_manifest.py
python scripts/download_data.py
python scripts/recalibrate_normalizer.py
python scripts/make_configs.py

if [ -z "${SKIP_GATE:-}" ]; then python scripts/repro_gate.py; fi   # exits 1 -> pipeline stops

python scripts/run_matrix.py --jobs "$JOBS" || echo "some runs failed: see results/failures.csv, rerun to retry"
python scripts/postprocess.py collect
python scripts/analyses.py
python scripts/figures.py
python scripts/finalize.py
