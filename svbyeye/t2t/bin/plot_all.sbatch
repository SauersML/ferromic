#!/bin/bash
#SBATCH --job-name=slco-svbyeye-plot
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=a100
set -euo pipefail
# Array over inversions. For each locus, render the T2T-chimp miropeat (the
# drop-in replacement) and the panTro6 miropeat (for the diff), from the
# per-locus PAFs written by align_chimp.sbatch/distribute_paf.py.
ART=${ART:?set ART to your svbyeye/t2t working directory (holds ref/ paf/ stageb/ plots/ pipeline/)}
ENVP=$ART/env/svbyeye; PIPE=${PIPE:-$(dirname "$0")}
INV=$ART/inputs/inversions.tsv
STAGEB=$ART/stageb; PLOTS=$ART/plots
RS=$ENVP/bin/Rscript
mkdir -p "$PLOTS" "$ART/qc"
export PATH="$ENVP/bin:$PATH"

row=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$INV")
IFS=$'\t' read -r inv_id chrom start end size af recur disease wstart wend flank <<< "$row"
[[ -z "${inv_id:-}" ]] && { echo "no row $SLURM_ARRAY_TASK_ID"; exit 0; }
qc=$ART/qc/${inv_id}.qc.tsv; : > "$qc"

for ref in chimpT2T panTro6; do
  paf=$STAGEB/$inv_id/${ref}.paf
  out=$PLOTS/${inv_id}.${ref}.png
  $RS "$PIPE/plot_chimp_miro.R" "$paf" "$out" "$inv_id" "$chrom" \
      "$start" "$end" "$wstart" "$wend" "$af" "$disease" "$ref" 2>&1 \
      | grep -E "^QC|WROTE|PLACEHOLDER|Error" | tee -a "$qc" || true
done
echo "PLOT_DONE $inv_id"
