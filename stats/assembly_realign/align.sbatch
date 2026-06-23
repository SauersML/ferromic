#!/usr/bin/env bash
#SBATCH --job-name=inv_align
#SBATCH --partition=agsmall
#SBATCH --array=0-11
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=2:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/di/polarize/logs/align_%a.out
set -euo pipefail
module load minimap2/2.30
WD=/projects/standard/hsiehph/sauer354/di/polarize
mapfile -t ASMS < <(ls $WD/asm/*.fna.gz | sort)
ASM=${ASMS[$SLURM_ARRAY_TASK_ID]}
tag=$(basename "$ASM" | cut -d. -f1)
echo "task $SLURM_ARRAY_TASK_ID: $tag (ape=ref, regions=query)"
# ape assembly is REFERENCE (indexed once), hg38 regions are QUERY (small -> bounded mem)
minimap2 -x asm20 -c --secondary=yes -N 20 -p 0.1 -t 16 \
  "$ASM" $WD/ref/hg38_regions.fa > $WD/paf/${tag}.paf 2> $WD/logs/mm2_${tag}.err
echo "done $tag: $(wc -l < $WD/paf/${tag}.paf) PAF lines"
