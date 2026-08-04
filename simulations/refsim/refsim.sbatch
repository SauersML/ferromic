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
#
# FLUX_SCOPE (leaves|all) selects where between-orientation flux acts; TAG names
# the output files, so a second scope can be run without overwriting the first:
#   sbatch --array=0-7 --export=ALL,TASK=flux,FLUX_SCOPE=all,TAG=fluxall refsim.sbatch
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
FLUX_SCOPE=${FLUX_SCOPE:-leaves}
TAG=${TAG:-$TASK}

srun ./env/bin/python run_grid.py \
    --task "$TASK" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --procs "${SLURM_CPUS_PER_TASK}" \
    --flux-scope "$FLUX_SCOPE" \
    --out "out/${TAG}_shard${SHARD}.csv"
