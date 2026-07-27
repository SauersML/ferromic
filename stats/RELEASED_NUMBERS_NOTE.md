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

## Imputation summary — both reconcile, with one off-by-one

Resolved by `stats/imputation_threshold_summary.py`. The two counts are not in
conflict: they use different thresholds over **different denominators**, and both
reproduce. 158 is every model fit; only 75 of those sit at a locus in the
93-locus consensus set.

| subset | n | r² > 0.3 | + BH | r² > 0.5 | + BH | r² > 0.7 | + BH |
|---|---|---|---|---|---|---|---|
| all models fit | 158 | 35 | **21** | 18 | 16 | 12 | 12 |
| consensus 93-locus set | 75 | 20 | 14 | **12** | 11 | 9 | 9 |

* The log's "158 models evaluated, 21 with r² > 0.3 and BH p < 0.05" reproduces
  exactly, over all models fit.
* The manuscript's "12 of the 93" reproduces as the count at **r² > 0.5 alone**
  within the consensus set.

**The off-by-one:** the manuscript writes "12 … (r² > 0.5 and BH p < 0.05)", but
12 is the r²-only count; requiring BH p < 0.05 as well gives **11**. Either drop
the BH clause from that sentence or change 12 to 11.

## Recurrence simulation false-positive rate — reconciled

Two earlier claims here were withdrawn along the way; this is the settled version,
from the corrected one-divergence single-event model over 11,250 loci.

**Overall single-origin false-positive rate 0.097, recurrent power 0.805.**

By inversion age: **0.250 at 50 kya, 0.034 at 100 kya, 0.006 at 250 kya.**
By recombination: 0.179 at ρ = 0, 0.110 at ρ = 10⁻⁸, 0.0005 at ρ = 10⁻⁶.

What matches the manuscript: the rate is driven by inversion age in the direction
it reports — worst at the shallowest depth, which is where it states its maximum —
and it is under 5% at every depth except the 50-kya model. Averaged over the older
depths the rule is comfortably inside "< 5%".

What still does not: the magnitude at 50 kya (0.25 against 4%), and the ordering in
ρ — we get essentially zero at ρ = 10⁻⁶ where the manuscript reports its maximum.
The direction we see follows from the coalescent rather than from an implementation
choice: at ρ = 0 the locus is a single genealogy and incomplete lineage sorting in a
young, small inverted deme breaks inverted monophyly, whereas at ρ = 10⁻⁶ the ML
tree averages over many genealogies and separates the orientations cleanly.

The practical consequence for the write-up is small: the flux claim is a comparison
across flux levels within one scenario, so the baseline cancels, and it holds
(see above). Only a sentence anchoring the classifier to a specific "< 5%" figure at
the youngest depth would need softening.


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

## eQTL evidence at the inversion tagging SNPs

`stats/inversion_eqtl_lookup.py` queries GTEx v8 for every single-tissue eQTL at the
tagging SNPs — 1,212 records across six loci.

The manuscript makes no PRAG1 claim. A PRAG1 result was raised separately as having no
committed code behind it; for the record, **PRAG1 returns no eQTL at any 8p23.1 tagging
SNP in GTEx v8** (0 records, against 22 genes that do: AF131215.9, CLDN23, ERI1,
MFHAS1, MSRA, MTMR9, PPP1R3B, PRSS55, RP1L1, TNKS, …). So if such a result exists it
does not come from this source, and the source would need naming.

The table *does* support two claims that are made: the 8p23.1
haplotype has a thyroid eQTL (AF131215.9, Thyroid, p = 3.1e-17), matching the thyroid
phenotype cluster; and 17q21.31 has very strong expression effects (KANSL1-AS1
p = 6.2e-158, LRRC37A4P p = 1.9e-174), consistent with the Geuvadis KANSL1 result the
response letter cites.

## Not checked here

Nothing outstanding in this section.
