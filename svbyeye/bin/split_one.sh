#!/bin/bash
# Split one HGSVC3 combined assembly into haplotype1 / haplotype2 FASTAs for PAV.
# Usage: split_one.sh <sample> <asm.fa.gz> <outdir>
set -euo pipefail
sample="$1"; asm="$2"; out="$3"
mkdir -p "$out"
h1="$out/${sample}_h1.fa"; h2="$out/${sample}_h2.fa"
if [[ -s "$h1.gz" && -s "$h2.gz" ]]; then echo "SKIP $sample (exists)"; exit 0; fi
# Stream fasta; route each record to h1/h2 by the haplotype tag in its header.
# Drop unassigned/unplaced contigs (no haplotype tag).
zcat "$asm" | awk -v f1="$h1" -v f2="$h2" '
  /^>/ { keep=0
         if (index($0,"haplotype1")>0) { dst=f1; keep=1 }
         else if (index($0,"haplotype2")>0) { dst=f2; keep=1 } }
  { if (keep) print >> dst }
'
bgzip -f -@4 "$h1"; bgzip -f -@4 "$h2"
samtools faidx "$h1.gz"; samtools faidx "$h2.gz"
n1=$(zcat "$h1.gz" | grep -c '^>' || true)
n2=$(zcat "$h2.gz" | grep -c '^>' || true)
echo "DONE $sample h1=$n1 h2=$n2 contigs"
