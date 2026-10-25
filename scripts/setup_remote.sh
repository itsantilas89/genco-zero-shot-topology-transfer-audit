#!/usr/bin/env bash
# One-time environment setup on the GPU server (README: "Setup").
#   CUDA=cu126|cu128|cu130|cpu   PyTorch wheel flavour (default cu126; must be <= driver's CUDA)
#   PYTHON=python3.11            interpreter for the venv it creates (3.10-3.12 supported by graphkit)
# If a conda env or venv is already active, it is used as is and PYTHON is ignored.
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA="${CUDA:-cu126}"
PYTHON="${PYTHON:-python3.11}"
TORCH=2.11.0   # newest torch for which PyG ships torch-scatter wheels on every CUDA flavour
COMMIT=$(python3 -c "import yaml;print(yaml.safe_load(open('settings.yaml'))['graphkit']['commit'])" 2>/dev/null || echo f104900)

echo "== GPU =="; nvidia-smi || echo "WARNING: nvidia-smi not found, installing for ${CUDA} anyway"

if [ -n "${VIRTUAL_ENV:-}" ] || { [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != "base" ]; }; then
  echo "== using the active environment: ${VIRTUAL_ENV:-$CONDA_PREFIX} =="
else
  echo "== venv (.venv, ${PYTHON}) =="
  if command -v uv >/dev/null; then uv venv --python "${PYTHON#python}" .venv; else "${PYTHON}" -m venv .venv; fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PIP="pip"; pip install -U pip   # plain pip: the PyTorch index must serve torch's own dependencies

echo "== torch ${TORCH} (${CUDA}) + torch-scatter / torch-sparse =="
$PIP install "torch==${TORCH}" --index-url "https://download.pytorch.org/whl/${CUDA}"
$PIP install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
# torch-sparse is optional in graphkit (only used for RRWP positional encodings)
$PIP install torch-sparse -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html" || echo "torch-sparse not installed (optional)"

echo "== gridfm-graphkit @ ${COMMIT} =="
if [ ! -d gridfm-graphkit/.git ]; then
  git clone https://github.com/gridfm/gridfm-graphkit.git gridfm-graphkit
fi
git -C gridfm-graphkit checkout "${COMMIT}"
$PIP install --no-deps -e gridfm-graphkit
$PIP install -r requirements.txt

echo "== sanity check =="
python - <<'PY'
import torch, torch_scatter, gridfm_graphkit
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count())
assert torch.cuda.is_available() or "cpu" in torch.__version__, "CUDA build installed but no GPU visible"
if torch.cuda.is_available():  # the scatter CUDA kernels must work, not just import
    import torch_scatter as ts
    x = torch.ones(4, device="cuda"); i = torch.tensor([0, 0, 1, 1], device="cuda")
    assert ts.scatter_add(x, i).tolist() == [2.0, 2.0]
print("graphkit import OK")
PY

mkdir -p results
{ echo "setup $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  nvidia-smi | grep "CUDA Version" || true; python --version; } > results/environment.txt
pip freeze > requirements.lock
echo "== done. Next: activate the environment, then bash scripts/run_all.sh =="
