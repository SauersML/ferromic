#!/bin/bash
#SBATCH --job-name=slco-smoke-svbyeye
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=a100
set -euo pipefail
# Exact-entrypoint smoke: run the SAME minimap2 params, distribute_paf.py and
# plot_chimp_miro.R the full run uses, but against a 2-chromosome hg38 target
# (chr21+chr22) so it is cheap. Validates SVbyEye rendering + PAF plumbing +
# diff end-to-end on real data before the whole-genome alignment.
ART=${ART:?set ART to your svbyeye/t2t working directory (holds ref/ paf/ stageb/ plots/ pipeline/)}
ENVP=$ART/env/svbyeye; REF=$ART/ref; PIPE=${PIPE:-$(dirname "$0")}
SM=$ART/smoke
export PATH="$ENVP/bin:$PATH"
mkdir -p "$SM/stageb" "$SM/plots"

echo "=== build small hg38 target (chr21+chr22) ==="
[[ -s "$SM/hg38.small.fa" ]] || samtools faidx "$REF/hg38.analysisSet.fa" chr21 chr22 > "$SM/hg38.small.fa"
[[ -s "$SM/hg38.small.mmi" ]] || minimap2 -x asm20 -t 16 -d "$SM/hg38.small.mmi" "$SM/hg38.small.fa"

echo "=== inversion table restricted to chr21/chr22 ==="
head -1 "$ART/inputs/inversions.tsv" > "$SM/inv.small.tsv"
awk -F'\t' '$2=="chr21"||$2=="chr22"' "$ART/inputs/inversions.tsv" >> "$SM/inv.small.tsv"
echo "smoke loci:"; tail -n +2 "$SM/inv.small.tsv" | wc -l

for ref in chimpT2T panTro6; do
  qfa=$REF/${ref}.relabel.fa.gz
  paf=$SM/${ref}.smallpaf.gz
  echo "=== align $ref -> hg38(chr21/22) [exact params] ==="
  minimap2 -x asm20 -c --eqx --secondary=no -s 25000 -K 1G -t 16 "$SM/hg38.small.mmi" "$qfa" \
      | gzip -c > "$paf"
  echo "$ref records: $(zcat "$paf" | wc -l)"
  python3 "$PIPE/distribute_paf.py" "$SM/inv.small.tsv" "$paf" "$ref" "$SM/stageb"
done

echo "=== plot each smoke locus (both refs) ==="
RS=$ENVP/bin/Rscript
while IFS=$'\t' read -r inv_id chrom start end size af recur disease wstart wend flank; do
  [[ "$inv_id" == "inv_id" ]] && continue
  for ref in chimpT2T panTro6; do
    $RS "$PIPE/plot_chimp_miro.R" "$SM/stageb/$inv_id/${ref}.paf" \
        "$SM/plots/${inv_id}.${ref}.png" "$inv_id" "$chrom" \
        "$start" "$end" "$wstart" "$wend" "$af" "$disease" "$ref" 2>&1 \
        | grep -E "^QC|WROTE|PLACEHOLDER|Error" || true
  done
done < "$SM/inv.small.tsv"

echo "=== diff on smoke set ==="
python3 "$PIPE/diff_chimp.py" "$SM/inv.small.tsv" "$SM/stageb" \
    "$SM/diff_chimp.smoke.tsv" "$SM/diff_chimp.smoke.json"
echo "=== smoke plots produced ==="
ls -la "$SM/plots" | head
echo "SMOKE_DONE"
