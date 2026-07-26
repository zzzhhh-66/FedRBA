#!/usr/bin/env bash
#SBATCH --job-name=fedrba-shard
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
MATRIX_FILE="${MATRIX_FILE:-$PROJECT_DIR/scripts/experiments_pending.txt}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-HyperET}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array.}"
SHARD_COUNT="${MATRIX_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

cd "$PROJECT_DIR" || exit 2
if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "Cannot find Conda initialization under $CONDA_ROOT" >&2
  exit 2
fi
if [[ ! -f "$MATRIX_FILE" ]]; then
  echo "Cannot find matrix file $MATRIX_FILE" >&2
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
if [[ $? -ne 0 ]]; then
  exit 2
fi

TOTAL_COMMANDS="$(wc -l < "$MATRIX_FILE")"
STATUS_FILE="$PROJECT_DIR/slurm_logs/${SLURM_ARRAY_JOB_ID}_${SHARD_ID}_status.tsv"
printf "matrix_line\tstatus\tcommand\n" > "$STATUS_FILE"
failures=0

for ((index=SHARD_ID; index<=TOTAL_COMMANDS; index+=SHARD_COUNT)); do
  command="$(sed -n "${index}p" "$MATRIX_FILE")"
  if [[ -z "$command" ]]; then
    continue
  fi
  echo "[$(date --iso-8601=seconds)] matrix_line=$index START $command"
  if bash -c "$command"; then
    printf "%s\tCOMPLETED\t%s\n" "$index" "$command" >> "$STATUS_FILE"
    echo "[$(date --iso-8601=seconds)] matrix_line=$index COMPLETED"
  else
    code=$?
    printf "%s\tFAILED(%s)\t%s\n" "$index" "$code" "$command" >> "$STATUS_FILE"
    echo "[$(date --iso-8601=seconds)] matrix_line=$index FAILED($code)" >&2
    failures=$((failures + 1))
  fi
done

echo "Shard $SHARD_ID/$SHARD_COUNT finished with $failures failure(s)."
if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
