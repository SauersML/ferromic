#!/bin/bash
#SBATCH --job-name=split_hap
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=msismall
#SBATCH -o /projects/standard/hsiehph/sauer354/di/svbyeye/logs/split_%A_%a.out
#SBATCH -e /projects/standard/hsiehph/sauer354/di/svbyeye/logs/split_%A_%a.err
set -euo pipefail
module load samtools/1.21
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "/projects/standard/hsiehph/sauer354/di/svbyeye/inputs/manifest.tab")
sample=$(echo "$line" | cut -f1); asm=$(echo "$line" | cut -f2)
[[ "$sample" == panTro6_chimp ]] && { echo "skip chimp"; exit 0; }
bash /projects/standard/hsiehph/sauer354/di/svbyeye/bin/split_one.sh "$sample" "$asm" "/projects/standard/hsiehph/sauer354/di/pav_run/assemblies"
