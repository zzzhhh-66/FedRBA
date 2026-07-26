# Slurm/HPC execution

The supplied launchers are templates. Cluster partition names, GPU resource
syntax, accounts, modules, and time limits vary by site; edit the `#SBATCH`
header before submission.

## 1. Prepare the environment

```bash
cd /path/to/FedRBA
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests -q
mkdir -p slurm_logs
```

The GPU launchers expect Conda by default. Override these variables when needed:

```bash
export CONDA_ROOT=/path/to/miniconda3
export CONDA_ENV_NAME=fedrba
```

## 2. Build and verify a matrix

```bash
python scripts/build_experiment_matrix.py \
  --profile full \
  --seeds 1 2 3 \
  --output scripts/experiments_main.txt

wc -l scripts/experiments_main.txt
sed -n '1,5p' scripts/experiments_main.txt
```

The main matrix must contain 90 non-empty lines.

## 3. Submit as a sharded array

Sharding lets one GPU job execute several matrix commands sequentially and
keeps the number of submitted jobs within common QOS limits.

```bash
sbatch \
  --array=1-4%4 \
  --export=ALL,PROJECT_DIR="$PWD",MATRIX_FILE="$PWD/scripts/experiments_main.txt",CONDA_ROOT="$CONDA_ROOT",CONDA_ENV_NAME="$CONDA_ENV_NAME",MATRIX_SHARDS=4 \
  scripts/slurm_sharded_gpu.sh
```

For a cluster with a stricter limit, reduce both the array range and
`MATRIX_SHARDS`. Do not request more concurrent GPUs than the user's QOS allows.

## 4. Monitor

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,AllocTRES
tail -n 50 slurm_logs/JOB_ID_TASK.out
tail -n 50 slurm_logs/JOB_ID_TASK.err
```

Each shard writes `slurm_logs/JOBID_TASK_status.tsv`. A successful matrix has no
`FAILED` entry.

## 5. Resume only missing runs

The training entry point protects existing outputs. Use
`scripts/build_pending_matrix.py` to construct a matrix containing only missing
or incomplete runs. Inspect that generated file before submission.

Never add `--overwrite` to a batch matrix unless the target output directories
have been audited and replacement is intentional.

## 6. Collect results

Keep raw `outputs/`, checkpoints, and Slurm logs on protected storage. Commit
only aggregate tables and the small audited paper artifacts. This repository's
`.gitignore` enforces that boundary.

