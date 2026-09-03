#!/bin/bash -l
#SBATCH --job-name=svbyeye-replot
#SBATCH --array=1-93
#SBATCH --time=1:00:00
#SBATCH --mem=16g
#SBATCH --cpus-per-task=1
#SBATCH --partition=msismall
#SBATCH --output=/projects/standard/hsiehph/sauer354/svbyeye-consensus-93/replot/logs/replot_%A_%a.out

# One array task per consensus locus: render the panTro6-vs-GRCh38 page from the
# per-locus slice of the combined alignment. No realignment; the combined PAF is
# the same one the published bundle came from.

set -euo pipefail

E=${SVBYEYE_DIR:-/projects/standard/hsiehph/sauer354/svbyeye-consensus-93}
W=$E/replot
R=${SVBYEYE_RSCRIPT:-/projects/standard/hsiehph/shared/conda_shared/envs/svbyeye_pipeline/bin/Rscript}

# The manifest is CRLF; a trailing carriage return in the label reaches the plot
# title as a line break.
row=$(tail -n +2 "$E/manifest.tsv" | tr -d '\r' | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$row" ]; then
  echo "No manifest row ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
IFS=$'\t' read -r inv_id chrom inv_start inv_end window_start window_end recurrence label <<< "$row"

echo "task ${SLURM_ARRAY_TASK_ID}: $inv_id ($chrom:$inv_start-$inv_end)"
"$R" "$E/plot_chimp_hires.R" \
  "$W/paf/$inv_id.paf" \
  "$W/plots/$inv_id" \
  "$inv_id" "$chrom" "$inv_start" "$inv_end" "$window_start" "$label"

for suffix in .pdf .png .orientation.tsv; do
  out="$W/plots/$inv_id$suffix"
  if [ ! -s "$out" ]; then
    echo "missing output: $out" >&2
    exit 1
  fi
done
