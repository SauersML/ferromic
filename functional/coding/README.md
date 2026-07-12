# `functional/coding/` — coding functional scoring

Score the orientation-differentiating missense fixed differences between inversion haplotypes
with three orthogonal variant-effect predictors, then consolidate into a per-CDS-site
functional-call table.

## Pipeline

1. **`map_consequences.py`** — map each ferromic CDS fixed difference (gene, transcript, CDS
   position, direct/inverted allele) to a genomic coordinate + amino-acid consequence using the
   GENCODE r47 model + GRCh38. Emits a REF-match flag: the reconstructed reference CDS base must
   equal the reported `direct_allele`, else the row is flagged and excluded from scoring
   (reported, not silently dropped). This is the silent-bug guard.
2. **`score_alphamissense.py`** — AlphaMissense pathogenicity lookup (genomic match) + a
   *same-gene matched-null percentile* (exact percentile within the gene's full missense
   spectrum) + per-transcript reference-protein extraction (input to ESM C). AM likely-pathogenic
   cutoff: `am_pathogenicity >= 0.564` (Cheng et al., Science 2023).
3. **`score_esmc.py`** — ESM C (`esmc_300m`) masked-marginal LLR at the variant residue
   (`log P(alt) − log P(ref)`); more negative = more deleterious. GPU.
4. **`score_evo2.py`** — Evo 2 7B zero-shot delta-likelihood over a genomic window around the
   variant; more negative = more disruptive. GPU. Also the shared ClinVar positive-control scorer.
5. **`combine.py`** — per-method damaging flags + cross-method concordance `n_methods_flag`, then
   `consolidate()` joins the flags back onto the full consequence annotation and assigns a
   `coding_call` per site (`functional 3/3`, `likely functional 2/3`, `sequence-model-only 1/3`,
   `benign 0/3`, `synonymous`, `missense unscored`).

Thresholds used by `combine`: AlphaMissense `>= 0.564`; Evo 2 ΔLL `<= -10.0`; ESM C LLR
`<= -5.0`. These reproduce the recorded per-method flags and `n_methods_flag` exactly (see
`../tests/test_coding_combine.py`).

## Run

The consolidation (no GPU, deterministic) from the frozen per-method scores:

```bash
python -m functional.coding.cli consolidate \
    --variants functional/data/coding/arm1_coding_variants.tsv \
    --scores   functional/data/coding/arm1_final_3method.tsv \
    --out      functional/results/coding/arm1_coding_calls.tsv
```

The upstream per-method scoring (needs the reference genome/GTF, AlphaMissense, and GPU for
ESM C / Evo 2) is driven from the module functions; each `score_*` module exposes importable
scoring functions plus a `score_table(...)`.

## Outputs

- `../results/coding/arm1_coding_calls.tsv` — per CDS site: consequence, 3-method scores,
  `n_methods_flag`, `coding_call`, featured-locus tag.
- `../results/coding/arm1_coding_summary.json` — counts + the functional/likely calls.

## Provenance

The scores were computed against GRCh38 + GENCODE v47, AlphaMissense hg38, ESM C `esmc_300m`,
and Evo 2 `arcinstitute/evo2_7b`. The positive-control gate (AlphaMissense AUROC 0.97, Evo 2
zero-shot 0.76 on ClinVar missense in these genes) is a separate ClinVar run, not committed here.
