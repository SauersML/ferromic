# Chimp polarization: `inverted == derived`

This repository polarizes every inversion's orientation against the chimpanzee
(and, where chimp is uninformative, the gorilla / orangutan / macaque) outgroup.

## The convention (project-wide)

For every orientation-dependent quantity:

| label | meaning |
|-------|---------|
| group / allele **0**, `0_*`, `group0`, `pi_direct`, PAML branch `#1`, tagging "direct" | **direct == ANCESTRAL** orientation (shared with the primate outgroups) |
| group / allele **1**, `1_*`, `group1`, `pi_inverted`, PAML branch `#2`, tagging "inverted" | **inverted == DERIVED** orientation w.r.t. chimp |

This **replaces** the historical convention, where "inverted" merely meant "not
the hg38 reference arrangement." `Inverted_AF` / `derived_af` are now the
frequency of the **derived** orientation.

## Source of truth: `data/inversion_polarity.tsv`

Produced by `stats/polarize_orientation.py` (multi-outgroup, **no-drop**):

1. `collect` — stream the UCSC hg38-vs-{panTro6,gorGor6,ponAbe3,rheMac10} net AXT
   and `all.chain` alignments, caching per-locus, per-outgroup strand evidence
   (breakpoint-trimmed interior; dominant orthologous chain to suppress
   paralog/SD noise).
2. `decide` — per outgroup, call the reference arrangement collinear vs inverted;
   root by **phylogenetically weighted parsimony** across outgroups. Congruent
   outgroups → high/moderate confidence; discordant outgroups (a positive
   recurrence / ILS signal, e.g. 17q21.31, 8p23.1) are **flagged**, not dropped.
   Loci with no usable ape orthology get `confidence=assumed` (reference assumed
   ancestral) — the only fallback, and it is labelled as such.

Key column: **`flip_ref_polarity`** — `1` when the hg38 reference orientation is
itself the derived one, so the raw `0/1` encoding must be swapped.

## Where the flip is applied

The Rust pipeline (`src/`) and the per-CDS `.phy` files remain in the **raw
hg38-reference encoding** (group0 = reference). Polarization is applied:

- **Tables** (`output.csv`, `inv_properties.tsv`, divergence, recurrence,
  tagging, gene tables, …): re-keyed in place by `scripts/apply_polarity.py`
  (swap paired `_direct`/`_inverted` columns + `0_`/`1_` columns, complement
  `*_AF`, swap `0|1` genotypes in `callset.tsv`). Pure data consumers (e.g.
  `stats/inv_dir_recur_model.py`, `stats/replicate_manuscript_statistics.py`)
  then get the new meaning for free.
- **`.phy` consumers** read the raw group files and swap `group0<->group1` for
  flipped loci at read time via `stats/_inv_common.is_flipped()`:
  `cds/combine_phy.py` (PAML branch labels), `stats/find_tagging_snps.py`,
  `stats/four_fold_pi.py`, `stats/pin_pis.py`.

`is_flipped(chrom, start, end, orig_id=...)` (in `stats/_inv_common.py`) is the
single lookup; it tolerates ±1 bp coordinate jitter and falls back to OrigID.

## Automation (CI)

- `.github/workflows/ancestral_orientation.yml` regenerates
  `data/inversion_polarity.tsv` (multi-outgroup) and commits it on dispatch.
- `.github/workflows/upload_latest_manual_run_artifacts.yml` polarizes a freshly
  produced raw `output.csv` (`apply_polarity.py --only output.csv`) before
  committing it — gated so an already-polarized file is never double-flipped.
- PAML (`analysis_pipeline.yml`) and tagging-SNP workflows regenerate their
  outputs from the now polarity-aware consumers.

The migration is idempotent by construction: the ephemeral
`data/.polarity_applied` marker (gitignored) guards the dataset-wide migration,
and the CI `--only` path operates solely on freshly produced raw artifacts.
