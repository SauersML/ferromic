#!/bin/bash
#SBATCH --job-name=slco-svbyeye-align
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --partition=a100
set -euo pipefail
# Array: task 1 = T2T chimp, task 2 = panTro6. Each aligns its chimp assembly
# (query) -> hg38 (target) with the SAME minimap2 params as the original panTro6
# pipeline, then distributes overlapping records into stageb/<inv_id>/<ref>.paf.
ART=${ART:?set ART to your svbyeye/t2t working directory (holds ref/ paf/ stageb/ plots/ pipeline/)}
ENVP=$ART/env/svbyeye
REF=$ART/ref; PIPE=${PIPE:-$(dirname "$0")}
IDX=$REF/hg38.asm20.mmi
INV=$ART/inputs/inversions.tsv
OUTDIR=$ART/paf; STAGEB=$ART/stageb
export PATH="$ENVP/bin:$PATH"
mkdir -p "$OUTDIR" "$STAGEB"

case "${SLURM_ARRAY_TASK_ID:-1}" in
  1) SAMPLE=chimpT2T; QFA=$REF/chimpT2T.relabel.fa.gz ;;
  2) SAMPLE=panTro6;  QFA=$REF/panTro6.relabel.fa.gz ;;
  *) echo "bad array id"; exit 1 ;;
esac

OUT=$OUTDIR/${SAMPLE}.hg38.paf.gz
if [[ -s "$OUT" ]]; then
  echo "SKIP align $SAMPLE (exists)"
else
  echo "=== ALIGN $SAMPLE ($QFA) -> hg38 ==="
  tmp=$OUTDIR/${SAMPLE}.hg38.paf.tmp
  # Matched params: -x asm20 -c --eqx --secondary=no -s 25000 -K 1G
  minimap2 -x asm20 -c --eqx --secondary=no -s 25000 -K 1G -t "${SLURM_CPUS_PER_TASK:-32}" \
      "$IDX" "$QFA" > "$tmp"
  nrec=$(wc -l < "$tmp")
  gzip -c "$tmp" > "$OUT" && rm -f "$tmp"
  echo "DONE align $SAMPLE records=$nrec -> $OUT"
fi

echo "=== DISTRIBUTE $SAMPLE into per-locus stageb ==="
python3 "$PIPE/distribute_paf.py" "$INV" "$OUT" "$SAMPLE" "$STAGEB"
echo "ALIGN_DISTRIBUTE_DONE $SAMPLE"
