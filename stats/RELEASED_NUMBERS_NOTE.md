# What the committed artifacts actually say

Several numbers differ between the manuscript draft and this repository. For each
one below, the value given is what the committed code and data produce, verified
by inspecting or rerunning the released pipeline. Where the manuscript disagrees,
the manuscript is the thing to change — the code-availability statement promises
that these tables reproduce from GitHub.

## Sample count — 44 individuals / 88 haplotypes

`data/callset.tsv` has 54 columns, of which 10 are metadata (`seqnames`, `start`,
`end`, `width`, `inv_id`, `arbigent_genotype`, `misorient_info`,
`orthog_tech_support`, `inversion_category`, `inv_AF`) and **44 are sample
columns**, giving 88 phased haplotypes. This matches
`replicate_manuscript_statistics.txt` ("44 phased individuals … 88 potential
phased haplotypes") and the per-region haplotype maxima it reports (max 81 direct
at a locus).

The draft's 41 / 82 is not reproducible from the committed callset.

## Phecode count — 1,090

`data/phewas_results.tsv` contains 1,090 unique values of `Phenotype`, matching
the replication log. The manuscript's 1,089 is off by one.

## Internal decay Spearman — ρ = 0.500, and the reported n is wrong in both

Resolved by `stats/decay_spearman_variants.py`, which recomputes the decay
independently from the committed per-site tracks.

**The released value reproduces exactly**: within-locus mean, across-locus median,
restricted to consensus-classified loci > 100 kbp gives **ρ = 0.5003, p = 2.16e-04**
over 60 contributing series — matching the log's 0.500, p = 2.156e-04 to four
significant figures.

**The quoted `n = 60` is a mislabel, in the manuscript as well as the log.** In
`_calc_spearman`, `n` is `len(window_data)`, the number of contributing
inversion × orientation series; the correlation itself runs over the 2 kbp bins
spanning 0–100 kbp, i.e. 50 points. Both published ρ values are consistent with 50
bins and neither with 60 (ρ = 0.500 at 50 bins → p = 2.2e-04; ρ = 0.451 → p = 1.0e-03,
which is exactly the manuscript's stated p). Report the bin count, or drop n.

**ρ = 0.451 is not reproducible.** Across all four combinations of within-locus
(mean/median) × across-locus (mean/median) aggregation the values are 0.500, 0.554,
and NaN twice — the two median-within variants are undefined because per-site π is
mostly zero, which is also why the log's "median within 2kb bins" line reads NA.
Nothing on the current data yields 0.451, so it is from an earlier data state.
Quote ρ = 0.500 with its rule stated.

## CDS conservation — the GLM p-values should not be quoted at all

Superseding the item below. `robust_cds_reanalysis.py` shows the weighted-binomial
GLM overstates the information in the data by treating C(k,2) overlapping haplotype
pairs as independent trials: nominal N is 486,275 where there are 26 inversions, and
the single-event coefficient is identified off 7 of them. Treating the inversion as
the independent unit, pairing within gene, and using exact randomisation:

| contrast | GLM | corrected |
|---|---|---|
| orientation, single-event | p = 0.0078 | +8.26 pts, exact p = 0.094 |
| orientation, recurrent | p = 0.44 | +3.36 pts, p = 0.426 |
| **recurrence × orientation** | **p = 0.0045** | **+4.90 pts, p = 0.413** |
| after background-diversity adjustment | — | −1.33 pts, p = 0.770 |

The interaction — the claim the manuscript makes — does not survive. The single-event
effect is marginal and fully accounted for by the lower background diversity of those
haplotypes (observed and predicted differences correlate at Spearman ρ = 0.690).
Power at the real unit count is 39.6% for the observed effect, ~14 points are needed
for 80%, and the exact test cannot return p below 2/128 = 0.0156 whatever the effect
size. Report this descriptively, with the power statement.

## CDS pairwise contrasts — the log, not the old table

Resolved and regenerated: `data/cds_pairwise_adjusted.tsv` was stale and had the
orientation levels assigned the other way round (it put Single/Inverted *below*
Single/Direct). Rerunning `stats/CDS_identical_model.py` reproduces the log
exactly. Recurrent/Inverted vs Single/Inverted is **Δlogit = −3.626,
p = 0.002529, q = 0.0148**, not q = 0.664. All four CDS EMM/pairwise tables have
been regenerated.

## Adjusted π interaction — p = 0.00039

See `PI_ADJUSTMENT_NOTE.md`. Report the covariate-adjusted row of
`recurrence_controls_summary.tsv`; Model C conditions on the exposure itself and
is not an adjusted estimate of the same effect.

## PAML ω direction — the manuscript is right; the table was stale

**An earlier version of this note said FDFT1 and BLK were reversed in the text.
That was wrong.** The manuscript is in the raw GRCh38 reference frame and always
matched the original PAML output.

`data/GRAND_PAML_RESULTS.tsv` had been rewritten into the *ancestral/derived*
frame by the chimp-polarization cutover of 2026-06-22 (`b4506bca`) and its
successors, which swapped the two ω columns for every gene at a locus whose
reference orientation is derived — including all of chr8:7301024-12598379, hence
FDFT1 and BLK. The cutover was reverted on 2026-07-01 (`202e6af3`), but this file
is a CI-regenerated output and was left stale in the polarized frame.

The swap is provable and purely mechanical: across all 194 genes, the polarized
and pre-polarization tables are identical except for 44 exactly-swapped ω pairs,
and **every** likelihood, p-value, q-value, proportion and κ is byte-identical.
The clade-model LRT is symmetric in the two clades, so a label swap cannot move
them. The table has been restored to the raw frame, and it now reads FDFT1
ω_direct = 0 / ω_inverted = 59.2 and BLK ω_direct = 156.4 / ω_inverted = 0.0001 —
the manuscript's values.

## 4-fold correlations — same root cause, numbers change

`data/four_fold_pi_by_inversion.tsv` was stale in the same way (58 rows, 30
identical, 28 exactly swapped) and has also been restored. The drafted ρ = 0.560
and ρ = 0.759 were computed in the polarized frame. In the raw frame the same
comparisons give:

| comparison (n = 26, consensus-classified) | polarized (drafted) | raw (correct) |
|---|---|---|
| whole-locus vs 4-fold | ρ = 0.560, p = 0.003 | **ρ = 0.542, p = 0.0042** |
| 4-fold vs whole-CDS | ρ = 0.759, p = 7.1e-06 | **ρ = 0.689, p = 9.9e-05** |

Same direction and conclusion, but the quoted values need updating.

## Still stale elsewhere?

`data/pin_pis_by_inversion.tsv` was in the same state and has been restored.
Everything else the revert flagged has either been regenerated on raw orientation
(`recurrence_controls`, `cds_conservation_calibration`) or was restored by the
revert itself (`callset.tsv`, `inv_properties.tsv`,
`balanced_recurrence_results.tsv`). `data/cds_identical_proportions.tsv` predates
the whole polarization episode (last touched 2025-11-23), so the CDS analyses are
unaffected.

## Flux rate units — per lineage per generation

The flux term is a symmetric migration rate between opposite-orientation demes,
so its unit is **migrants per lineage per generation**, not per base pair per
generation. The repository uses that wording consistently
(`simulations/refsim/README.md`, `make_report.py` axis label); the Methods edit
that says "per base pair per generation" for the flux sweep should be corrected.
The recombination rates ρ (0, 1×10⁻⁸, 1×10⁻⁶) *are* per base pair per
generation — that sentence is right, which is probably how the two got conflated.

## Imputation summary

`replicate_manuscript_statistics.txt` reports 158 models evaluated with 21 at
r² > 0.3 and BH p < 0.05; the manuscript quotes 12 of 93 at r² > 0.5. These are
different thresholds over different denominators rather than a contradiction, but
only one pair of numbers should appear in the paper, and it should be the pair
the released tables support.

## Recurrence simulation false-positive rate — still unresolved

Two successive claims here were wrong and are both withdrawn: first that the
manuscript's "< 5%" was unreproducible (that was our mis-specified single-event
model), then that a corrected model put it at 1.6% (that was a non-comparable
subset with an effectively older inversion).

The corrected single-event model — the Methods' own one-divergence model at
t_inv ∈ {50, 100, 250} kya — over 2,700 loci gives an overall false-positive rate of
**0.106**. At m = 0 the rate is 0.533 / 0.333 / 0.000 across ρ = 0 / 10⁻⁸ / 10⁻⁶ at
50 kya, 0.133 / 0.017 / 0.000 at 100 kya, and 0.000 throughout at 250 kya.

The age ordering matches the manuscript (worst at 50 kya, its stated maximum). The
magnitude at 50 kya (0.53 vs 0.04) and the ordering in ρ (we get zero at 10⁻⁶, where
the manuscript reports its maximum) do not. Our direction follows from the coalescent:
at ρ = 0 the locus is a single genealogy and ILS in a young, small inverted deme breaks
monophyly, whereas at ρ = 10⁻⁶ the ML tree averages over many genealogies and separates
the orientations cleanly.

**The flux claim itself is fully supported.** Over the corrected 5,400-locus sweep the
single-origin FPR goes 0.113 → 0.106 (z = −0.39, p = 0.70) and power 0.891 → 0.907
(z = 0.91, p = 0.36) across m = 0 → 10⁻⁶ — neither moves detectably. The response
letter's wording ("the false positive rate under the single-event models did not
increase, and the power to detect recurrent loci was not reduced") stands as written. An
intermediate note here that the FPR rose at p = 0.033 came from the mis-specified
single-event model and is withdrawn.

Quote the flux result as a relative statement across flux levels; do not anchor it to a
"< 5%" baseline until the ρ-ordering question above is resolved.

## AGES multiple-testing correction — say which one

Two different corrections are in play and they differ by three orders of magnitude:

* `data/best_tagging_snps_qvalues.tsv` — Benjamini-Hochberg **across the inversion
  candidate set only**. This is what the manuscript quotes, and it reproduces exactly
  (8p23.1 q = 4.29e-04, 12q13.11 q = 0.005, 10q22.3 q = 4.29e-04, 7p11.2 q = 0.033).
* `ages_FDR` in `data/ages_multi_tag_snps.tsv` — AGES's **own genome-wide** correction
  across every SNP in their scan. For the same SNP, rs4268452 at 10q22.3 has
  P_X = 7.2e-05 and AGES FDR = 0.37.

Both are legitimate for their own family. The Methods must name which is reported,
because a reviewer pulling the SNP straight from the AGES database will see 0.37.

## Not checked here

Nothing outstanding in this section.
