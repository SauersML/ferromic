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
# Submit with:  sbatch --array=0-7 --export=TASK=gene_flux refsim.sbatch
#
# Gene flux acts between opposite orientations throughout every interval in
# which both populations exist. TAG optionally names the output files.
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
export OMP_NUM_THREADS=1

NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAG:-$TASK}
RUN_SCRATCH="$ROOT/scratch/refsim_${SLURM_JOB_ID}_${SHARD}"
mkdir -p "$RUN_SCRATCH"
trap 'rm -rf -- "$RUN_SCRATCH"' EXIT

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
    --scratch "$RUN_SCRATCH" \
    "${extra_args[@]}" \
    --out "out/${TAG}_shard${SHARD}.csv"
