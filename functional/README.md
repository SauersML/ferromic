# `functional/` — functional-consequence analyses for polymorphic inversions

Reproducible analysis code + result tables behind the functional-consequence supplement:
what measurable and predicted molecular consequences do inversion haplotypes carry at the
coding, splicing, and regulatory levels, and how much of a predicted effect is the
orientation flip itself versus the SNVs linked to it? Four independent analyses, one per
subdirectory.

**Scope:** analysis code and the result tables it produces. There is no interpretive prose
here — the "what it means" lives with the study. All results are **associational and
haplotype-level**: a tag-SNP dosage indexes the inversion haplotype, not the inversion per se.

## The three analyses

| Subdir | Analysis | Model / data | Key output |
|---|---|---|---|
| [`coding/`](coding/README.md) | Coding functional scoring of orientation-differentiating missense variants | AlphaMissense + ESM C + Evo 2 7B zero-shot | `results/coding/arm1_coding_calls.tsv` |
| [`splice/`](splice/README.md) | Gene-localised AlphaGenome splice-disruption + GTEx-sQTL validation | AlphaGenome API | `results/splice/per_inversion_splice.tsv` |
| [`regulatory/`](regulatory/README.md) | Measured cis eQTL & sQTL by inversion-tag dosage | Geuvadis LCL RNA-seq + GTEx v10 | `results/regulatory/{arm_eqtl,regulatory_per_locus}.tsv` |
| [`structural/`](structural/README.md) | Structure-vs-linked-SNV decomposition of the predicted splice effect | AlphaGenome counterfactual on real phased haplotypes | `results/structural/{master_table.csv,debias_summary.json}` |

Each analysis has a runnable CLI (`python -m functional.<analysis>.cli ...`); see the
per-module READMEs for exact commands.

## Layout

```
functional/
  paths.py            input-path resolution + provenance recording (no hard-coded paths)
  featured_loci.py    canonical featured-locus definitions (shared)
  coding/             AlphaMissense / ESM C / Evo 2 scoring + cross-method consolidation
  splice/             AlphaGenome per-gene splice formulation + GTEx validation
  regulatory/         Geuvadis/GTEx eQTL + sQTL + per-locus integration
  structural/         AlphaGenome structure-vs-SNV counterfactual decomposition
  data/               small committed reference inputs (inversions, tag variants, panel, ...)
  results/            committed recorded result tables (the reference outputs)
  tests/              reproduction + QC tests (see "Verification")
```

## Dependencies

Pure-Python + scientific stack; no private/internal libraries. Install the analysis extras:

```bash
pip install numpy scipy pandas          # core (all analyses)
pip install pysam pyfaidx                # coding: AlphaMissense lookup + protein extraction
pip install torch fair-esm              # coding: ESM C (GPU)
pip install evo2                         # coding: Evo 2 7B zero-shot (GPU)
pip install alphagenome                  # splice: AlphaGenome API
pip install Pgenlib pyliftover           # regulatory: genotype dosage + hg19->hg38 liftover
```

## Input data (not committed)

Large inputs resolve through `functional/paths.py`, in order: an explicit `--<name>` flag, a
`FUNCTIONAL_<NAME>` env var, or `<FUNCTIONAL_DATA_ROOT>/<relpath>`. Sources:

| Key | What / where to fetch |
|---|---|
| `reference_fasta`, `gencode_gtf` | GRCh38 primary assembly + GENCODE v47 (gencodegenes.org) |
| `alphamissense` | AlphaMissense hg38 (Zenodo 10.5281/zenodo.8208688) |
| `clinvar_vcf` | ClinVar GRCh38 VCF (NCBI ClinVar FTP) — positive-control gate |
| `geuvadis_pgen` | Geuvadis (E-GEUV-1) LCL genotypes, PLINK2 pgen (hg19) |
| `geuvadis_gene_rpkm` | Geuvadis gene RPKM matrix (ArrayExpress E-GEUV-1) |
| `geuvadis_junction` / `_exon` / `_transcript` | Geuvadis splicing matrices (E-GEUV-1) |
| `gtex_eqtls` | GTEx v10 cis-eQTL at tag SNPs (GTEx portal API) |
| `alphagenome_scores` | per-inversion AlphaGenome `.npz` (produced by `functional.splice.cli score`) |

`data/download_reference.sh` documents the reference/AlphaMissense/ClinVar fetch commands.
Small inputs (inversion coordinates, tag variants, the 1000G panel, ENSG→symbol map) and the
frozen per-method score tables ship under `data/`.

## Verification

Every result is traceable to a script + input, and the packaged code is checked against the
recorded outputs:

- **Pure-logic reproductions (run in CI, no external data):**
  - `tests/test_coding_combine.py` — the documented method thresholds recreate the recorded
    per-method flags + `n_methods_flag`, and `consolidate()` reproduces `arm1_coding_calls.tsv`.
  - `tests/test_mapping_qc.py` — CDS→residue mapping: codon assembly, strand handling,
    reverse-complement round-trip, and the REF-match silent-bug guard.
  - `tests/test_integrate_reproduction.py` — `integrate()` reproduces `regulatory_per_locus.tsv`
    from the committed reference inputs.
- **Data-dependent reproductions (skip unless the inputs are configured):**
  - `tests/test_eqtl_reproduction.py` — the Geuvadis cis-eQTL scan (17q21, seed 42) matches the
    recorded per-gene betas and analytic p in `arm_eqtl.tsv`.
  - `tests/test_splice_reproduction.py` — the gene-localised splice formulation recomputes each
    inversion's top-splice gene + max splice score to match `per_inversion_table.csv`.

```bash
pip install pytest
python -m pytest functional/tests -q          # pure tests run; data-dependent ones skip
# with inputs configured (FUNCTIONAL_DATA_ROOT / FUNCTIONAL_* env vars), all run
```

Determinism: every stochastic step (eQTL/sQTL permutation nulls, expression-PC construction)
is seeded (default 42). Provenance: analysis CLIs write a `*_provenance.json` next to their
output recording the resolved absolute input paths + a timestamp.
