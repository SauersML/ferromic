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

2. **Gold-standard overlay** (`stats/integrate_gold_polarity.py`): chain synteny is
   only ~39% concordant with the deep-outgroup direction, so paralog-immune
   assembly/Strand-seq evidence is layered on top by fixed precedence —
   2025 Complete Ape Genomes T2T (`data/apes2025_t2t_polarity.tsv`,
   `stats/polarize_apes2025_t2t.py`), Porubsky 2020 Strand-seq + deep-outgroup
   (`data/strandseq_polarity.tsv`, `stats/polarize_strandseq.py`). The overlay is
   **idempotent**, and CI (`ancestral_orientation.yml`) runs it right after the v2
   caller so regeneration reproduces — never reverts — the gold source of truth.

Key columns:
- **`flip_ref_polarity`** — `1` when the hg38 reference orientation is itself the
  derived one, so the raw `0/1` encoding must be swapped.
- **`evidence_tier`** — `gold_t2t_apes` > `gold_deepOG` > `gold_strandseq` >
  `recurrent_{t2t_apes,strandseq}` > `synteny`.
- **`resolution_status`** — honest trust bucket for EVERY locus (quarantine, never a
  silent reference-ancestral fallback): `resolved` (gold assembly/Strand-seq),
  `recurrent` (toggles between orientations), `provisional` (chain-synteny only — low
  trust), `unresolved` (no ape orthology — not polarizable by any outgroup method).
  Only `resolved` + `recurrent` carry trustworthy polarity; `provisional`/`unresolved`
  flips are best-effort estimates and should be treated as such downstream.

## The remaining gap (`provisional` + `unresolved`) — what has been exhausted

The 212 non-gold loci (38 of the 93-locus consensus set) have been checked against every
tractable orthogonal source, none of which resolves them:

- **2025 Complete Ape Genomes T2T (6 species, SYRI/PAV)** — no overlapping ape inversion
  call at these loci, by either reciprocal (>0.3) **or containment** (smaller feature ≥50%
  covered, deep-ape HOM) overlap. Containment re-check added **0** loci.
- **Strand-seq (Porubsky 2020, n=1,069)** — no clean deep-outgroup signal.
- **Yunis & Prakash cytogenetic inversions (supp. table)** — only Mb-scale *species-fixed*
  pericentric inversions; not applicable to these (mostly <100 kb) human polymorphisms.
- **Chain synteny (v2)** — paralog-confounded; only ~39% concordant with the deep-gold
  direction, so it is reported as `provisional`, never trusted as gold.

These loci are mostly small/young human-specific inversions and SD-rich regions where no
ape assembly carries a resolvable orthologous inversion.

- **UCSC 16-way diploid Cactus chains** (all 12 ape haplotypes, hg38-referenced;
  `stats/polarize_cactus_apes.py`) were downloaded and run — they cover 336/399 loci, but
  agree with the Strand-seq gold standard only **~50%** even for high-confidence unanimous
  calls. Precomputed chains get forced onto one paralog copy across inversion junctions in
  SDs (they call 8p23.1 wrong), so they are **not integrated**. This is why the published
  consortium SYRI/PAV calls (synteny-block-aware, 88% vs Strand-seq) are the reliable
  assembly tier and raw chains are not.

The only remaining rigorous method is **raw local realignment of breakpoint-flanking
windows** (`minimap2 -x asm20 --secondary=yes -N 50`, keeping secondary alignments so SD
paralogs are not collapsed) against the ape assembly FASTAs — which needs the ~32 GB
assemblies + an aligner on real compute (MSI `acn116`, or a per-assembly CI job). On this
network MSI is unreachable (UMN VPN gateways blocked at :443; GCP relay billing closed) and
local disk is insufficient, so it is deferred. Until then these loci stay honestly
quarantined rather than assigned fabricated confidence.

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
