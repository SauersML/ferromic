#!/bin/bash
#SBATCH --job-name=slco-svbyeye-dl
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=a100
set -euo pipefail
ART=${ART:?set ART to your svbyeye/t2t working directory (holds ref/ paf/ stageb/ plots/ pipeline/)}
REF=$ART/ref
mkdir -p "$REF"; cd "$REF"

NCBI=https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/028/858/775/GCA_028858775.2_NHGRI_mPanTro3-v2.0_pri
UCSC=https://hgdownload.soe.ucsc.edu/goldenPath

echo "=== [1/3] T2T chimpanzee (mPanTro3 v2.0, GCA_028858775.2) ==="
wget -q -O t2t_chimp.mPanTro3v2.fna.gz          "$NCBI/GCA_028858775.2_NHGRI_mPanTro3-v2.0_pri_genomic.fna.gz"
wget -q -O t2t_chimp.assembly_report.txt        "$NCBI/GCA_028858775.2_NHGRI_mPanTro3-v2.0_pri_assembly_report.txt"
echo "b1d6f7965d27505d090cf48071c0ad9d  t2t_chimp.mPanTro3v2.fna.gz" | md5sum -c -

echo "=== [2/3] panTro6 (UCSC, the assembly being replaced) ==="
wget -q -O panTro6.fa.gz "$UCSC/panTro6/bigZips/panTro6.fa.gz"

echo "=== [3/3] GRCh38 no-alt analysis set (target reference) ==="
wget -q -O hg38.analysisSet.fa.gz "$UCSC/hg38/bigZips/analysisSet/hg38.analysisSet.fa.gz"
gunzip -kf hg38.analysisSet.fa.gz
# .fai is built in the prep step once the env's samtools is ready

echo "=== record checksums + sizes ==="
md5sum *.gz > CHECKSUMS.md5
ls -la
echo "=== chimp FASTA header sample (T2T) ==="
( zcat t2t_chimp.mPanTro3v2.fna.gz | grep '^>' | head -3 ) || true
echo "=== panTro6 header sample ==="
( zcat panTro6.fa.gz | grep '^>' | head -3 ) || true
echo "DL_DONE"
