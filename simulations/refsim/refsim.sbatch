#!/bin/bash -l
#SBATCH --job-name=refsim
#SBATCH --account=hsiehph
#SBATCH --partition=sioux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160g
#SBATCH --time=6:00:00
#SBATCH --output=logs/refsim_%A_%a.out
# Reference recurrence pipeline (simulations/refsim) over a sharded grid.
# Submit with:  sbatch --array=0-7 --export=TASK=fluxsweep refsim.sbatch
#
# FLUX_SCOPE (leaves|all) selects where between-orientation flux acts; TAG names
# the output files, so a second scope can be run without overwriting the first:
#   sbatch --array=0-7 --export=ALL,TASK=flux,FLUX_SCOPE=all,TAG=fluxall refsim.sbatch
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
cd "$ROOT"
mkdir -p logs out

module load iqtree2/2.1.2-gcc-13.1.0-xsnl63s
export IQTREE_BIN
IQTREE_BIN=$(command -v iqtree2)
test "$(sha256sum "$IQTREE_BIN" | cut -d' ' -f1)" = \
  81a16f2958e63d25cf6d67e2b64222b2ba24b34aa76069a6225a2df36ce166a2
export PYTHONPATH="$ROOT"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMP_NUM_THREADS=1

NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
FLUX_SCOPE=${FLUX_SCOPE:-leaves}
TAG=${TAG:-$TASK}

extra_args=()
if [[ -n ${REPS:-} ]]; then
    extra_args+=(--reps "$REPS")
fi
if [[ -n ${SCENARIOS:-} ]]; then
    extra_args+=(--scenarios "$SCENARIOS")
fi
if [[ -n ${DEPTHS:-} ]]; then
    extra_args+=(--depths "$DEPTHS")
fi
if [[ -n ${RHOS:-} ]]; then
    extra_args+=(--rhos "$RHOS")
fi
if [[ -n ${FLUX:-} ]]; then
    extra_args+=(--flux "$FLUX")
fi

srun ./env/bin/python run_grid.py \
    --task "$TASK" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --procs "${SLURM_CPUS_PER_TASK}" \
    --flux-scope "$FLUX_SCOPE" \
    "${extra_args[@]}" \
    --out "out/${TAG}_shard${SHARD}.csv"
