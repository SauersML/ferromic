#!/bin/bash
#SBATCH --job-name=hg38_vs_chimp
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=msismall
#SBATCH -o /projects/standard/hsiehph/sauer354/di/svbyeye/logs/hg38_vs_chimp_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/di/svbyeye/logs/hg38_vs_chimp_%j.err
set -euo pipefail
module load minimap2/2.30
REF=/projects/standard/hsiehph/sauer354/di/svbyeye/ref/panTro6.fa.gz
HG=/projects/standard/hsiehph/sauer354/di/hg38.no_alt.fa
OUT=/projects/standard/hsiehph/sauer354/di/svbyeye/paf_chimp/hg38_vs_panTro6.paf.gz
mkdir -p /projects/standard/hsiehph/sauer354/di/svbyeye/paf_chimp
# target = chimp, query = hg38  -> alignments oriented w.r.t. chimp (ancestral ref)
minimap2 -x asm20 -c --eqx --secondary=no -s 25000 -K 1G -t 12 "$REF" "$HG" | gzip -c > "$OUT"
echo "DONE hg38 vs chimp -> $OUT"; zcat "$OUT" | wc -l
