# ScoreInvHap concordance benchmark (Reviewer 2, comment 3)

Benchmarks our imputed inversion dosages against the published genotyping tool
**scoreInvHap** (Ruiz-Arenas et al., *PLOS Genetics* 2019;
[isglobal-brge/scoreInvHap](https://github.com/isglobal-brge/scoreInvHap))
for the two inversions scoreInvHap ships reference objects for, **8p23.1** and
**17q21.31**, in a large public cohort (1000 Genomes).

## What runs where

Each method is run on its own native genome build, then the two call sets are
intersected on the shared 1000 Genomes sample IDs:

| Method | Inversion IDs | Cohort / build |
| --- | --- | --- |
| scoreInvHap (`inv8_001`, `inv17_007`) | 8p23.1, 17q21.31 | 1000G phase-3, GRCh37 |
| Our PLS models (`data/models/`) | `chr8-7301025-INV-5297356`, `chr17-45585160-INV-706887` | 1000G high-coverage, GRCh38 |

Inversion IDs were verified by coordinate against `data/inv_properties.tsv`:
8p23.1 = `chr8-7301025-INV-5297356` (chr8:7,301,024-12,598,379) and
17q21.31 = `chr17-45585160-INV-706887` (chr17:45,585,159-46,292,045).

## Files

- `run_scoreinvhap.R` — genotype an inversion with scoreInvHap from a VCF.
- `infer_dosage_public.py` — run our trained models on the prepared genotype
  matrices, filling missing genotypes with the per-SNP cohort mean (no
  All-of-Us ancestry priors; public data only).
- `compute_concordance.py` — join both methods per sample; emit `r`, `r²`,
  Spearman ρ, hard-call concordance and inverted-allele frequencies, plus a
  scatter PDF.

The whole benchmark is wired as
`.github/workflows/scoreinvhap_concordance.yml` and writes
`data/scoreinvhap_concordance.tsv` and `data/scoreinvhap_concordance.pdf`.

## Reviewer expectations

8p23.1 is expected to impute near-perfectly (r² ≈ 1) and 17q21.31 has perfect
tag SNPs, so both inversions should show very high concordance with
scoreInvHap.
