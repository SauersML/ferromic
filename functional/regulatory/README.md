# `functional/regulatory/` — measured regulatory-QTL analyses

Do inversion haplotypes change cis gene expression and splicing in real RNA-seq? Measured cis
eQTL and sQTL by inversion-tag dosage in the Geuvadis LCL panel (462 samples) plus GTEx v10,
integrated into one per-locus table. Associational, haplotype-level.

## Modules

- **`common.py`** — shared machinery: tag-SNP ALT dosage from the Geuvadis genotypes
  (`pgenlib`), the covariate design matrix (intercept + sex + population one-hots + top-K
  genome-wide expression PCs, a PEER proxy since the matrices are library-depth-normalised but
  not PEER-corrected), OLS with a two-sided t on the dosage coefficient, and BH-FDR.
- **`eqtl.py`** — Geuvadis cis-eQTL: for each gene within 1 Mb of an inversion, regress
  `log2(RPKM+1)` on inversion-tag dosage + covariates. Reports beta (per-ALT log2 fold change),
  direction, analytic p, a per-locus permuted-dosage empirical null, and a genome-wide BH-q.
- **`sqtl_geuvadis.py`** — Geuvadis cis differential splicing: LeafCutter-style intron-excision
  ratios (junction / cluster-sum, clusters = shared-splice-site junctions in a gene), exon PSI,
  or transcript usage, regressed on dosage + covariates.
- **`sqtl_gtex.py`** — GTEx v10 multi-tissue sQTL lookup at each tag SNP, plus the enrichment
  test: is a tag SNP more often a measured sVariant than a **MAF- and gene-proximity-matched**
  background of common variants (one-sided Fisher)? Matching on both MAF and proximity is
  essential (tag SNPs are common and gene-dense); the enrichment is modest and goes null under a
  looser match, so it is reported as borderline, not robust.
- **`integrate.py`** — join measured eQTL + measured sQTL/splicing + measured GTEx eQTL +
  AlphaGenome predicted splice into one per-locus table with a `measured_molecular_any` flag.

**No genome-wide "eQTL enrichment vs background" claim is made** — cis-eQTL is ubiquitous for
common variants, so that would be trivially true. eQTL results are used mechanistically (which
gene, which direction).

## Run

```bash
# Geuvadis cis-eQTL (deterministic, seed 42)
python -m functional.regulatory.cli eqtl \
    --inversions functional/data/inversions.tsv \
    --out functional/results/regulatory/arm_eqtl.tsv \
    --gene-rpkm $FUNCTIONAL_GEUVADIS_GENE_RPKM --pgen $FUNCTIONAL_GEUVADIS_PGEN

# Geuvadis cis splicing-QTL (junction / exon / transcript)
python -m functional.regulatory.cli sqtl-geuvadis --pheno junction \
    --inversions functional/data/inversions.tsv \
    --out functional/results/regulatory/armA_junction.tsv

# Integrate measured + predicted per locus
python -m functional.regulatory.cli integrate \
    --eqtl functional/results/regulatory/arm_eqtl.tsv \
    --sqtl-master functional/data/regulatory/per_inversion_master.tsv \
    --gtex-eqtl functional/data/regulatory/gtex_eqtls.tsv \
    --ag-splice functional/data/regulatory/per_inversion_table.csv \
    --inversions functional/data/inversions.tsv \
    --ensg-symbol functional/data/ensg_symbol.tsv.gz \
    --out functional/results/regulatory/regulatory_per_locus.tsv
```

Input paths resolve via `functional/paths.py` (`--<name>` flag, `FUNCTIONAL_*` env var, or
`FUNCTIONAL_DATA_ROOT`). The 1000G panel + inversion tables ship under `functional/data/`.

## Outputs

- `../results/regulatory/arm_eqtl.tsv` (+ `_summary.json`) — per gene×locus cis-eQTL.
- `../results/regulatory/regulatory_per_locus.tsv` (+ `regulatory_summary.json`) — integrated
  per-locus measured + predicted consequences.
- `armA_{junction,exon,transcript}.tsv` — Geuvadis splicing-QTL per phenotype (when run).

## Provenance

Geuvadis (E-GEUV-1), hg19/GENCODE v12 matrices; genotypes as PLINK2 pgen. GTEx v10 via the
portal API. Each CLI writes a `*_provenance.json` recording the resolved input paths + seed.
