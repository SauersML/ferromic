#!/bin/bash -l
#SBATCH --job-name=refsim
#SBATCH --partition=msismall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160g
#SBATCH --time=6:00:00
#SBATCH --output=logs/refsim_%A_%a.out
# Reference recurrence pipeline (simulations/refsim) over a sharded grid.
# Submit with:  sbatch --array=0-7 --export=TASK=flux refsim.sbatch
set -euo pipefail

ROOT=/projects/standard/hsiehph/sauer354/refsim
cd "$ROOT"
mkdir -p logs out

module load iqtree2/2.1.2-gcc-13.1.0-xsnl63s
export IQTREE_BIN=$(command -v iqtree2)
export PYTHONPATH="$ROOT"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMP_NUM_THREADS=1

NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

srun ./env/bin/python run_grid.py \
    --task "$TASK" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --procs "${SLURM_CPUS_PER_TASK}" \
    --out "out/${TASK}_shard${SHARD}.csv"
