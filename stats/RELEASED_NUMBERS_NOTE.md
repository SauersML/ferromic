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

## Internal decay Spearman — ρ = 0.500, p = 2.156e-04, n = 60

`replicate_manuscript_statistics.txt` records ρ = 0.500 for the overall internal
decay of diversity against distance from the locus start (first 100 kb, loci
≥ 100 kb, n = 60), with the per-locus values in
`data/spearman_decay_points.tsv`. The manuscript's ρ = 0.451, p = 0.001 is not
produced by any committed script; n = 60 agrees, so the two are the same
comparison computed differently.

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

## PAML ω direction — FDFT1 and BLK are reversed in the text

See the commit adding `stats/paml_extreme_omega_check.py`. The pipeline is
self-consistent (`cds/pipeline_lib.py` marks pure direct branches `#1`, and the
parser maps PAML branch type 1 → `cmc_omega2_direct`), so
`winner_omega2_direct = 59.2` for FDFT1 really is the direct clade. The draft
states the opposite for both named genes.

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

## Recurrence simulation false-positive rate

The manuscript's "< 5%" is not reproduced by the upstream pipeline itself. Scoring
11,250 loci through `simulations/refsim/` — a step-for-step port of
hsiehphLab/inversionSimulation — the single-origin false-positive rate of
`minMutHomoplasy ≥ 2` is **0.155** overall and **0.149** at zero flux, which is
upstream's own model. Replacing the previous NJ trees with the reference IQ-TREE
pipeline moved this *up*, not toward 5%, so the tree method is not the explanation.
The rate is also highest at ρ = 0 (0.266) and near zero at ρ = 10⁻⁶ (0.004), the
opposite ordering to the manuscript's reported 4% at ρ = 10⁻⁶.

What does hold, and is now measured with the manuscript's own classifier: across
m_flux 0 → 10⁻⁶ the false-positive rate moves 0.149 → 0.156 and power 0.804 → 0.816.
The extreme-flux extension locates the breakdown between 10⁻⁵ and 10⁻⁴, above the
whole swept range, which is why the sweep is flat.

The most likely explanation for the remaining gap is a rejection criterion applied on
top of the event count in the original pipeline that is not present in its published
code. Until that is identified, the flux result should be reported as a relative
statement (flux does not degrade the classifier) rather than against a "< 5%" baseline.

## Not checked here

The AGES q-values in `data/best_tagging_snps_qvalues.tsv` reproduce the
manuscript exactly (8p23.1 q = 4.29e-04, 12q13.11 q = 0.005, 10q22.3
q = 4.29e-04, 7p11.2 q = 0.033) and need no action.
