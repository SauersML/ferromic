#!/bin/bash
# Align mPanTro3 to the GRCh38 windows of every ledger locus and render the
# SVbyEye pages. Run after prepare_t2t_chimp.sh and build_ledger_windows.py.
# One minimap2 run covers all 292 windows; the 93 consensus loci are then
# rendered through render_chimp_consensus.py (the bundle that CI consumes) and
# the remaining loci through the same renderer for the orientation ledger.
set -euo pipefail
D=/projects/standard/hsiehph/sauer354/chimp-t2t-20260909
cd "$D"
R=/projects/standard/hsiehph/shared/conda_shared/envs/svbyeye_pipeline/bin/Rscript
HG=/projects/standard/hsiehph/shared/DIR_homes/caffe029/hg38.no_alt.fa
# The node python is 3.6; the render helpers need 3.9+.
PY=/projects/standard/hsiehph/sauer354/.venvs/demesdraw_cli/bin/python
until grep -q PREP_DONE logs/prep_full.log; do sleep 30; done
module load minimap2/2.30 samtools/1.21
# 1. windows for all 292 ledger loci (clipped to chromosome length); reused when already built
if [ ! -s win/all292.windows.fa.fai ] || [ "$(grep -c . win/all292.windows.fa.fai)" -ne 292 ]; then
  : > win/all292.windows.fa
  tail -n +2 windows292.tsv | while IFS=$'\t' read -r inv_id chrom s e ws we label src; do
    len=$(awk -v c="$chrom" '$1==c{print $2}' ref/hg38.no_alt.fa.fai)
    [ "$we" -gt "$len" ] && we=$len
    samtools faidx --fai-idx ref/hg38.no_alt.fa.fai "$HG" "$chrom:$((ws+1))-$we" | sed "1s/.*/>$inv_id/" >> win/all292.windows.fa
  done
  samtools faidx win/all292.windows.fa
  echo WINDOWS_DONE $(grep -c '>' win/all292.windows.fa)
fi
# 2. one alignment: target = hg38 windows, query = mPanTro3 (alias-named).
# -p 0.1 keeps secondaries down to a tenth of the primary score so a single
# chimpanzee copy of a tandemly duplicated block is reported against both
# human copies instead of only the best-scoring one.
# -K 4G loads the whole assembly as one batch: minimap2 hands out whole query
# sequences to threads, so the default 500 Mb batch keeps only two or three
# chromosomes in flight.
if [ ! -s paf/all292.mPanTro3.paf ]; then
/usr/bin/time -v minimap2 -x asm20 -c --eqx --secondary=yes -N 50 -p 0.1 -K 4G -t 20 win/all292.windows.fa ref/mPanTro3.fa > paf/all292.mPanTro3.paf 2> logs/mm2_all292.log
fi
echo ALIGN_DONE $(wc -l < paf/all292.mPanTro3.paf)
# 3. consensus 93 via the committed splitter/renderer
tail -n +2 manifest.clean.tsv | cut -f1 | sort > logs/ids93.txt
awk -F'\t' 'NR==FNR{k[$1]=1;next} ($6 in k)' logs/ids93.txt paf/all292.mPanTro3.paf > paf/consensus93.mPanTro3.paf
export SVBYEYE_CHIMP_ASSEMBLY=mPanTro3
"$PY" render_chimp_consensus.py --manifest manifest.clean.tsv --paf paf/consensus93.mPanTro3.paf --plot-script plot_chimp_hires.R --rscript "$R" --output-dir out93 --jobs 8 > logs/render93.log 2>&1
echo RENDER93_DONE
# 4. the other 199 loci with the same renderer, for orientation calls
mkdir -p wide199/paf wide199/plots
tail -n +2 windows292.tsv | awk -F'\t' '$8=="ledger"' > logs/ledger199.tsv
while IFS=$'\t' read -r inv_id chrom s e ws we label src; do
  awk -F'\t' -v t="$inv_id" '$6==t' paf/all292.mPanTro3.paf > wide199/paf/$inv_id.paf
done < logs/ledger199.tsv
render_one() {
  IFS=$'\t' read -r inv_id chrom s e ws we label src <<< "$1"
  if [ -s wide199/paf/$inv_id.paf ]; then
    "$R" plot_chimp_hires.R wide199/paf/$inv_id.paf wide199/plots/$inv_id "$inv_id" "$chrom" "$s" "$e" "$ws" "$label" > wide199/plots/$inv_id.log 2>&1 || echo "FAIL $inv_id"
  else
    echo "NOPAF $inv_id"
  fi
}
export -f render_one; export R
xargs -a logs/ledger199.tsv -d '\n' -P 8 -I{} bash -c 'render_one "$@"' _ {} > logs/render199.log 2>&1
echo RENDER199_DONE
