#!/bin/bash
# Prepare the T2T chimpanzee assembly (mPanTro3 hap1, GCA_028858775.2) and the
# 93 consensus GRCh38 windows for the SVbyEye alignments.
#
# Inputs fetched once on the MSI login node (compute nodes have no internet):
#   curl -LO https://hgdownload.soe.ucsc.edu/hubs/GCA/028/858/775/GCA_028858775.2/GCA_028858775.2.2bit
#   curl -LO https://hgdownload.soe.ucsc.edu/hubs/GCA/028/858/775/GCA_028858775.2/GCA_028858775.2.chromAlias.txt
#   curl -LO https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/twoBitToFa
# The 2bit names sequences by GenBank accession; chromAlias maps them to the
# assembly names (CM054453.2 -> chr19_hap1_hsa17, the chr17 homolog), which is
# what the plots label. manifest.clean.tsv is the manifest from the published
# consensus-93 bundle.
set -euo pipefail
D=/projects/standard/hsiehph/sauer354/chimp-t2t-20260909
cd "$D"
module load samtools/1.21
test -s ref/chromAlias.txt
./dl/twoBitToFa dl/mPanTro3.GCA_028858775.2.2bit ref/mPanTro3.acc.fa
python3 - <<'PY'
alias = {}
for line in open("ref/chromAlias.txt"):
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) >= 2:
        alias[f[0]] = f[1]
out = open("ref/mPanTro3.fa", "w")
for line in open("ref/mPanTro3.acc.fa"):
    if line.startswith(">"):
        acc = line[1:].split()[0]
        out.write(">" + alias.get(acc, acc) + "\n")
    else:
        out.write(line)
out.close()
PY
rm ref/mPanTro3.acc.fa
samtools faidx ref/mPanTro3.fa
HG=/projects/standard/hsiehph/shared/DIR_homes/caffe029/hg38.no_alt.fa
: > win/consensus93.windows.fa
tail -n +2 manifest.clean.tsv | while IFS=$'\t' read -r inv_id chrom inv_start inv_end ws we rec label; do
  samtools faidx --fai-idx ref/hg38.no_alt.fa.fai "$HG" "$chrom:$((ws+1))-$we" | sed "1s/.*/>$inv_id/" >> win/consensus93.windows.fa
done
samtools faidx win/consensus93.windows.fa
echo PREP_DONE
