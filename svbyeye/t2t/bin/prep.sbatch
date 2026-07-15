#!/bin/bash
#SBATCH --job-name=slco-svbyeye-prep
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=a100
set -euo pipefail
ART=${ART:?set ART to your svbyeye/t2t working directory (holds ref/ paf/ stageb/ plots/ pipeline/)}
ENVP=$ART/env/svbyeye
REF=$ART/ref
PIPE=${PIPE:-$(dirname "$0")}
export PATH="$ENVP/bin:$PATH"

echo "=== [1] hg38 .fai ==="
[[ -s "$REF/hg38.analysisSet.fa.fai" ]] || samtools faidx "$REF/hg38.analysisSet.fa"
head -30 "$REF/hg38.analysisSet.fa.fai" | cut -f1,2

echo "=== [2] hg38 minimap2 asm20 index ==="
[[ -s "$REF/hg38.asm20.mmi" ]] || minimap2 -x asm20 -t 16 -d "$REF/hg38.asm20.mmi" "$REF/hg38.analysisSet.fa"

echo "=== [3] inversion table + windows ==="
mkdir -p "$ART/inputs"
python3 "$PIPE/make_inv_table.py" "$PIPE/inputs/inv_properties.tsv" \
    "$REF/hg38.analysisSet.fa.fai" "$ART/inputs/inversions.tsv"
echo "n inversions:"; tail -n +2 "$ART/inputs/inversions.tsv" | wc -l

echo "=== [4] relabel chimp FASTAs (clean chromosome-named query ids) ==="
python3 "$PIPE/relabel_chimp.py" --in "$REF/t2t_chimp.mPanTro3v2.fna.gz" \
    --out "$REF/chimpT2T.relabel.fa.gz" --prefix chimpT2T \
    --report "$REF/t2t_chimp.assembly_report.txt" --map-out "$REF/chimpT2T.namemap.tsv"
python3 "$PIPE/relabel_chimp.py" --in "$REF/panTro6.fa.gz" \
    --out "$REF/panTro6.relabel.fa.gz" --prefix panTro6 --map-out "$REF/panTro6.namemap.tsv"
echo "chimp query id samples:"
( zcat "$REF/chimpT2T.relabel.fa.gz" | grep '^>' | head -5 ) || true
( zcat "$REF/panTro6.relabel.fa.gz" | grep '^>' | head -5 ) || true
echo "PREP_DONE"
