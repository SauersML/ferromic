#!/bin/bash
#SBATCH --job-name=plotinv
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=msismall
#SBATCH -o /projects/standard/hsiehph/sauer354/di/svbyeye/logs/plot_%A_%a.out
#SBATCH -e /projects/standard/hsiehph/sauer354/di/svbyeye/logs/plot_%A_%a.err
set -euo pipefail
BASE=/projects/standard/hsiehph/sauer354/di/svbyeye
RS=/projects/standard/hsiehph/shared/conda_shared/envs/svbyeye_pipeline/bin/Rscript
INV=$BASE/inputs/inversions.tsv
row=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$INV")
IFS=$'\t' read -r inv_id chrom start end size af recur disease wstart wend flank <<< "$row"
plotdir=$BASE/plots; mkdir -p "$plotdir"
srcdir=$BASE/stageb/$inv_id
combined=$BASE/work/${inv_id}.combined.paf
if compgen -G "$srcdir/*.paf" > /dev/null; then
  cat "$srcdir"/*.paf > "$combined"
else
  : > "$combined"
fi
$RS $BASE/bin/visualize_inv.R "$combined" "$plotdir/$inv_id" "$inv_id" \
    "$chrom" "$start" "$end" "$wstart" "$wend" "$af" "$recur" "$disease" 2>&1
# directional gradient track (third view)
$RS $BASE/bin/track_gradient.R "$combined" "$plotdir/${inv_id}.grad.png" "$inv_id" \
    "$chrom" "$start" "$end" "$wstart" "$wend" "$af" 2>&1 | grep -E "WROTE|Error" || true
rm -f "$combined"
echo "PLOT_DONE $inv_id"
