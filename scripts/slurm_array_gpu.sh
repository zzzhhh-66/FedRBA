#!/usr/bin/env bash
#SBATCH --job-name=fedrba
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
MATRIX_FILE="${MATRIX_FILE:-$PROJECT_DIR/scripts/experiments.txt}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-HyperET}"
cd "$PROJECT_DIR"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "Cannot find Conda initialization under $CONDA_ROOT" >&2
  exit 2
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the Slurm GPU job.")
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

COMMAND="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MATRIX_FILE")"
if [[ -z "$COMMAND" ]]; then
  echo "No command at line $SLURM_ARRAY_TASK_ID in $MATRIX_FILE" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] $COMMAND"
bash -c "$COMMAND"
