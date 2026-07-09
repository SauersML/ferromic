#!/bin/bash
#SBATCH --job-name=hg38_mmi
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=msismall
#SBATCH -o /projects/standard/hsiehph/sauer354/di/svbyeye/logs/build_index.%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/di/svbyeye/logs/build_index.%j.err
set -euo pipefail
module load minimap2/2.30
REF=/projects/standard/hsiehph/sauer354/di/hg38.no_alt.fa
OUT=/projects/standard/hsiehph/sauer354/di/svbyeye/ref/hg38.asm20.mmi
minimap2 -x asm20 -t 8 -d "$OUT" "$REF"
echo "INDEX_DONE $OUT"; ls -la "$OUT"
