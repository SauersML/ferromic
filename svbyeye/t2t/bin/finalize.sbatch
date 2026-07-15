#!/bin/bash
#SBATCH --job-name=slco-svbyeye-finalize
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=a100
set -euo pipefail
ART=${ART:?set ART to your svbyeye/t2t working directory (holds ref/ paf/ stageb/ plots/ pipeline/)}
ENVP=$ART/env/svbyeye; PIPE=${PIPE:-$(dirname "$0")}
export PATH="$ENVP/bin:$PATH"
INV=$ART/inputs/inversions.tsv; STAGEB=$ART/stageb; PLOTS=$ART/plots
mkdir -p "$ART/results"

echo "=== whole-genome diff (T2T vs panTro6) ==="
python3 "$PIPE/diff_chimp.py" "$INV" "$STAGEB" \
    "$ART/results/diff_chimp.tsv" "$ART/results/diff_chimp.json"

echo "=== assemble per-locus QC ==="
{ echo -e "ref\tinv_id\tchrom\tstart\tend\tfrac_rev\tcov_pct\torient";
  cat "$ART"/qc/*.qc.tsv 2>/dev/null | grep '^QC' | cut -f2-; } > "$ART/results/plot_qc.tsv" || true

echo "=== HTML gallery ==="
# copy plots next to the gallery for a self-contained artifact
mkdir -p "$ART/results/gallery/plots"
cp -f "$PLOTS"/*.png "$ART/results/gallery/plots/" 2>/dev/null || true
python3 "$PIPE/make_gallery.py" "$ART/results/diff_chimp.json" \
    "$ART/results/gallery/plots" "$ART/results/gallery/index.html" plots

echo "=== counts ==="
echo "t2t plots:    $(ls "$PLOTS"/*.chimpT2T.png 2>/dev/null | wc -l)"
echo "panTro6 plots:$(ls "$PLOTS"/*.panTro6.png 2>/dev/null | wc -l)"
echo "FINALIZE_DONE"
