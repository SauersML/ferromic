# `functional/structural/` — structure-vs-SNV decomposition of an inversion's effect

Split an inversion's *predicted* molecular consequence into a **structural** (orientation-flip)
component and a **linked-SNV** component, using a sequence model (AlphaGenome) to run the
counterfactual that data cannot: hold the SNVs fixed and change only the orientation. Companion
population-genetic and conditional-QTL readouts quantify how separable structure and linked SNVs
are *from data alone*.

**All results are haplotype-level and model-based.** This bounds the structure-vs-linked-SNV
split of the *AlphaGenome-predicted* splice-usage disruption; it does not by itself prove the
inversion is the causal allele. There is no interpretive prose here — the "what it means" lives
with the study.

## Method (per locus)

Build four sequences on a fixed GRCh38 window around each breakpoint (see `reconstruct.py`):

| sequence | orientation | SNVs |
|---|---|---|
| `ref-direct`    | direct   | reference |
| `ref-inverted`  | inverted | reference |
| `full-inverted` | inverted | phased **inverted-background** SNVs |
| `full-direct`   | direct   | phased **direct-background** SNVs |

Score `SPLICE_SITE_USAGE` for each with AlphaGenome. The per-position/per-track difference field
decomposes exactly (linear identity, verified to float precision as a QC):

```
d_total  = SSU(full_inverted) - SSU(full_direct)
d_struct = SSU(ref_inverted)  - SSU(ref_direct)          # pure orientation flip
d_snv    = d_total - d_struct                            # differential linked SNVs
fraction_structural = ||d_struct|| / (||d_struct|| + ||d_snv||)      # flank-restricted L1
```

Norms are restricted to positions **outside** the flipped segment (frame-consistent: identical
genomic base identity in every sequence there, so a change is a genuine context-mediated
splicing perturbation). Per locus, windows are combined disruption-weighted.

**Consensus vs de-biased.** `score-consensus` uses a common-variant major-allele consensus per
background; the resulting fraction is an **upper bound** (consensus under-counts the linked-SNV
load). `score-perhap` re-runs the decomposition with **true per-individual phased haplotypes**
(full SNV content, rare + common) and reports the **de-biased** fraction plus a per-individual
range. Orientation per haplotype is from the tag SNP, validated against the Porubsky
assembly-based orientation calls; primary claims use only tag-reliable loci
(tag↔Porubsky concordance ≥ 0.90).

## Modules

- `reconstruct.py` — build the four counterfactual sequences from real phased data (2bit
  reference + region-streamed 1000G panel). Deterministic; 0-based half-open internally.
- `score_consensus.py` — consensus decomposition via the AlphaGenome API → `ag_decomp.json`
  (+ optional per-position arrays).
- `score_perhap.py` — **de-biased** per-haplotype decomposition (seed 42, K=16 haps/background)
  → `perhap_debiased.json`.
- `bg_stats.py` — background genetics: between-background divergence, within-background diversity,
  and tag↔cis-SNV LD (collinearity/separability).
- `anchor.py` — model-free structural anchor: a breakpoint inside a protein-coding gene is
  structural by construction (needs GENCODE).
- `recurrence.py` — recurrent-vs-single contrasts (recurrence as a natural experiment).
- `integrate.py` — per-locus verdicts from the handles above.
- `decompose.py` — the pure flank-restricted L1 arithmetic (shared by `summarize` + the tests).
- `summarize.py` — assemble `master_table.csv`, `qc_summary.json`, `debias_summary.json` from the
  per-window JSONs (pure; no API).

## Run

```bash
# Reproducible-from-cache: recompute the tables from the committed per-window JSONs (no API)
python -m functional.structural.cli summarize

# Upstream build/score (needs ALPHAGENOME_API_KEY + network + reference/panel via functional.paths)
python -m functional.structural.cli score-consensus --loci-file <loci.json> --out ag_decomp.json --arraydir <dir>
python -m functional.structural.cli score-perhap    --out perhap_debiased.json
python -m functional.structural.cli bg-stats
python -m functional.structural.cli anchor
python -m functional.structural.cli recurrence
python -m functional.structural.cli integrate
```

## Inputs

Small derived inputs are committed under `../data/structural/` (target table, analysis-locus
list, per-window gene spans, `config.json`). Large inputs resolve through `functional/paths.py`
(no hard-coded paths):

| Key | What |
|---|---|
| `reference_2bit` | GRCh38 2bit (UCSC `hg38.2bit`) |
| `thousand_genomes_panel_dir` | 1000G high-coverage GRCh38 3202-sample phased panel (EBI FTP or a local mirror; region-streamed). Override the default EBI URL with `FUNCTIONAL_THOUSAND_GENOMES_PANEL_DIR`. |
| `gencode_gtf` | GENCODE GTF (protein-coding gene spans, for `anchor`) |

Backgrounds also use ferromic's Porubsky phased inversion callset (hom-carrier sample lists,
baked into the committed target table) and the experiment-#8 tag SNPs.

## Outputs (`../results/structural/`)

- `master_table.csv` — per locus: recurrence, AF, tag concordance/reliability,
  `fraction_structural`, background divergence, tagging r², breakpoint-in-gene, verdict.
- `qc_summary.json` — decomposition reconciliation error, reverse-complement round-trip,
  ref-match mismatches, tag↔AF agreement, tag-reliable count + median fraction.
- `debias_summary.json` — de-biased median fraction + consensus upper bound + per-individual range.
- `perhap_debiased.json`, `ag_decomp_full.json`, `bg_stats.json`, `structural_anchor.json`,
  `handle2_recurrence.json`, `handle3_conditional.json`, `locus_verdicts.json`.

## Verification

`tests/test_structural_reproduction.py` (pure, runs in CI): recomputes each locus's
`fraction_structural` from the committed norms and matches `master_table.csv`; reproduces the
QC + de-biased summaries; and asserts the headline (9 tag-reliable loci, all structure-dominant,
consensus median ~0.85, de-biased median ~0.80).

Determinism: `score-perhap` fixes seed 42. Provenance: the target table records the exact
per-sample hom-carrier lists and tag SNPs; large inputs are logged by `functional/paths.py`.
