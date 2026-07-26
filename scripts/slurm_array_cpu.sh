#!/usr/bin/env bash
#SBATCH --job-name=fedrba-cpu
#SBATCH --array=1-90%12
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
MATRIX_FILE="${MATRIX_FILE:-$PROJECT_DIR/scripts/experiments.txt}"
cd "$PROJECT_DIR"
mkdir -p slurm_logs

if [[ -n "${VIRTUAL_ENV_PATH:-}" ]]; then
  source "$VIRTUAL_ENV_PATH/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

COMMAND="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MATRIX_FILE")"
if [[ -z "$COMMAND" ]]; then
  echo "No command at line $SLURM_ARRAY_TASK_ID in $MATRIX_FILE" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] $COMMAND"
srun bash -lc "$COMMAND --device cpu"
