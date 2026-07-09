#!/bin/bash
#SBATCH --job-name=svbe_final
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=msismall
#SBATCH -o /projects/standard/hsiehph/sauer354/di/svbyeye/logs/finalize_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/di/svbyeye/logs/finalize_%j.err
set -uo pipefail
BASE=/projects/standard/hsiehph/sauer354/di/svbyeye
mkdir -p $BASE/qc $BASE/results
echo "=== orientation QC ==="
python3 $BASE/bin/orient_qc.py $BASE/inputs/inversions.tsv $BASE/stageb $BASE/qc/orientation_qc.tsv
echo "=== gallery ==="
python3 $BASE/bin/make_gallery.py $BASE/qc/orientation_qc.tsv $BASE/plots $BASE/plots/index.html
echo "=== counts ==="
echo "PAFs:    $(ls $BASE/paf/*.gz 2>/dev/null | wc -l)/66"
echo "PNGs:    $(ls $BASE/plots/*.png 2>/dev/null | wc -l)/292"
echo "PDFs:    $(ls $BASE/plots/*.pdf 2>/dev/null | wc -l)/292"
echo "DONE finalize"
