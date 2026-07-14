#!/bin/bash
# Fetch the public reference data the coding analysis needs. Set DEST to your data root
# (the same path you pass as FUNCTIONAL_DATA_ROOT). Downloads are large; run once.
#
# Coordinates: GRCh38 / hg38 throughout for the coding arm.
set -euo pipefail
DEST="${1:-${FUNCTIONAL_DATA_ROOT:?set FUNCTIONAL_DATA_ROOT or pass a dest dir}}/reference"
mkdir -p "$DEST"
cd "$DEST"

# GRCh38 primary assembly + GENCODE v47 annotation
wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/GRCh38.primary_assembly.genome.fa.gz
gunzip -kf GRCh38.primary_assembly.genome.fa.gz
mv -f GRCh38.primary_assembly.genome.fa GRCh38.primary_assembly.genome.fa 2>/dev/null || true
wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.annotation.gtf.gz

# AlphaMissense hg38 (Cheng et al., Science 2023; Zenodo). ~1 GB.
wget -c "https://zenodo.org/records/8208688/files/AlphaMissense_hg38.tsv.gz"
# optional: bgzip + tabix index for fast lookup (else the scorer does one filtered linear scan)
#   zcat AlphaMissense_hg38.tsv.gz | bgzip > AlphaMissense_hg38.bgz && tabix -s1 -b2 -e2 AlphaMissense_hg38.bgz

# ClinVar GRCh38 VCF (positive-control gate)
wget -c https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz -O clinvar_GRCh38.vcf.gz

echo "Reference data in $DEST"
echo "Geuvadis genotypes/expression/splicing (E-GEUV-1) and GTEx v10 sQTLs are fetched"
echo "separately; see functional/regulatory/README.md."
