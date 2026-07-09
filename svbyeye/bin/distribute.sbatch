#!/bin/bash
#SBATCH --job-name=distpaf
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=msismall
#SBATCH -o /projects/standard/hsiehph/sauer354/di/svbyeye/logs/dist_%A_%a.out
#SBATCH -e /projects/standard/hsiehph/sauer354/di/svbyeye/logs/dist_%A_%a.err
set -euo pipefail
BASE=/projects/standard/hsiehph/sauer354/di/svbyeye
MANIFEST=$BASE/inputs/manifest.tab
INV=$BASE/inputs/inversions.tsv
OUTROOT=$BASE/stageb
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
sample=$(echo "$line" | cut -f1)
paf=$BASE/paf/${sample}.hg38.paf.gz
if [[ ! -s "$paf" ]]; then echo "SKIP missing PAF $paf"; exit 0; fi
python3 $BASE/bin/distribute_paf.py "$INV" "$paf" "$sample" "$OUTROOT"
echo "DIST_DONE $sample"
