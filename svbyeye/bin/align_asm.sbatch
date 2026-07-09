#!/bin/bash
#SBATCH --job-name=asm2hg38
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --partition=msismall
#SBATCH -o /projects/standard/hsiehph/sauer354/di/svbyeye/logs/align_%A_%a.out
#SBATCH -e /projects/standard/hsiehph/sauer354/di/svbyeye/logs/align_%A_%a.err
set -euo pipefail
module load minimap2/2.30
module load samtools/1.21
MANIFEST=/projects/standard/hsiehph/sauer354/di/svbyeye/inputs/manifest.tab
IDX=/projects/standard/hsiehph/sauer354/di/svbyeye/ref/hg38.asm20.mmi
OUTDIR=/projects/standard/hsiehph/sauer354/di/svbyeye/paf
mkdir -p "$OUTDIR"
# pick the SLURM_ARRAY_TASK_ID-th data line (1-based, skip header)
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
sample=$(echo "$line" | cut -f1)
asm=$(echo "$line" | cut -f2)
out="$OUTDIR/${sample}.hg38.paf.gz"
if [[ -s "$out" ]]; then echo "SKIP $sample (exists)"; exit 0; fi
echo "ALIGN sample=$sample asm=$asm"
tmp="$OUTDIR/${sample}.hg38.paf.tmp"
minimap2 -x asm20 -c --eqx --secondary=no -s 25000 -K 1G -t 12 \
    "$IDX" "$asm" > "$tmp"
nrec=$(wc -l < "$tmp")
gzip -c "$tmp" > "$out" && rm -f "$tmp"
echo "DONE $sample records=$nrec -> $out"
