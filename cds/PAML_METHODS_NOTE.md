# PAML methods reconciliation note

This note documents exactly what the committed PAML analysis is, supplies
corrected Methods/Results wording for the manuscript (which is not in this
repo), and folds in the relevant reviewer comments. It accompanies the
code-documentation changes in `cds/pipeline_lib.py`, `cds/finalize_results.py`,
and `stats/paml_agg.py`.

Audit reference: code-audit issue #8; Reviewer 1 comment 5; Reviewer 2
comment 2.

## 1. What the committed analysis actually is

The executable analysis and the committed results
(`data/GRAND_PAML_RESULTS.tsv`) are a **Clade Model C (CmC) test, partitioned
by inversion orientation** — they are **not** a single-branch dN/dS test on the
branch leading into the inverted haplotype, and not a branch-site test.

Concretely, in `cds/pipeline_lib.py`:

- `RUN_CLADE_MODEL_TEST = True`, `RUN_BRANCH_MODEL_TEST = False`
  (lines ~180). The branch-model two-ratio test (`model = 2`, `NSsites = 0`)
  is disabled and is opt-in only; it produced none of the committed results.
- codeml is run with `model = 3`, `NSsites = 2`, `ncatG = 3`
  (`analyze_single_gene` → `_build_attempts`), i.e. Clade Model C.
- Branch labelling (`create_paml_tree_files`):
  - **H1 (alternative):** every *pure direct* internal/terminal branch is
    marked `#1` and every *pure inverted* branch is marked `#2`. Branches whose
    descendants mix orientations, plus the outgroup, are background. This gives
    three CmC partitions: background, pooled-direct clade, pooled-inverted
    clade. It pools **all** direct branches and **all** inverted branches — it
    does not single out the one branch that leads into the inversion.
  - **H0 (null):** both pure direct and pure inverted branches collapse to a
    single foreground class `#1`, forcing the two orientations to share one
    divergent-class dN/dS.
- The reported statistic is the H1-vs-H0 likelihood-ratio test, `df = 1`
  (`cmc_p_value` / `overall_p_value`, FDR-corrected to `cmc_q_value` /
  `overall_q_value`). Each hypothesis is optimised from a tournament of four
  starting (kappa, omega) seeds and the best log-likelihood is kept.

Committed output columns reflect CmC, not a branch model:
`winner_p0/p1/p2` (site-class proportions), `winner_omega0` (the conserved/
purifying class shared across branches), and `winner_omega2_direct` /
`winner_omega2_inverted` (the divergent site-class dN/dS in the pooled direct
vs pooled inverted clades). The file contains **no** `bm_*` (branch-model)
columns.

So the H1-vs-H0 LRT answers one question: **does the divergent site class have
a different dN/dS between the pooled direct and the pooled inverted clades?** It
is a test of orientation-associated *differentiation* in selective regime, not
a test for positive selection on any individual lineage.

## 2. What the committed numbers show (47 usable genes)

- 47 genes have complete H0+H1 likelihoods ("Usable: complete data");
  10 more have H1 fitting worse than H0; the rest are excluded/uncomputable.
- The smallest raw LRT p-value is ~1.3e-3, but **no gene survives
  Benjamini–Hochberg FDR correction** (smallest `overall_q_value` ≈ 0.063;
  zero genes with q < 0.05).
- Per-class omega > 1 is nonetheless common in the point estimates: ~20/47
  genes show `winner_omega2_inverted` > 1 and ~26/47 show
  `winner_omega2_direct` > 1.

The juxtaposition is the key honest result: elevated divergent-class omega
estimates are frequent, yet the formal orientation-difference test detects
nothing significant after multiple-testing correction. Those high omega point
estimates are expected sampling behaviour on short, low-information branches and
are **not** evidence of positive selection (see §4).

## 3. Corrected Methods wording (ready to paste)

> Selection on protein-coding sequence was tested with codeml from PAML using
> Clade Model C (CmC; `model = 3`, `NSsites = 2`, `ncatG = 3`). For each gene we
> partitioned branches of the gene tree by inversion orientation: all branches
> ancestral only to direct haplotypes formed one foreground clade and all
> branches ancestral only to inverted haplotypes formed a second foreground
> clade, with branches subtending both orientations (and the chimpanzee
> outgroup) as background. The alternative model (H1) allowed the divergent CmC
> site class to take a separate dN/dS in the direct and inverted clades; the
> null model (H0) constrained the two orientations to share a single
> divergent-class dN/dS. Support for an orientation-specific shift in selective
> regime was assessed by a likelihood-ratio test between H1 and H0 (1 degree of
> freedom), with each model optimised from four independent starting points and
> the best likelihood retained. P-values were corrected across genes by the
> Benjamini–Hochberg procedure. This design contrasts the *pooled* direct and
> *pooled* inverted clades; it is not a branch-specific test of the single
> lineage leading into the inverted arrangement, and a per-class dN/dS estimate
> above one does not by itself constitute a test for positive selection.

Corrected Results wording (suggested):

> Across the 47 genes with complete CmC fits, no gene showed a significant
> difference in divergent-class dN/dS between direct and inverted clades after
> FDR correction (all q > 0.05). Although point estimates of the divergent-class
> dN/dS exceeded one for many genes in one or both orientations, these estimates
> are imprecise on the short internal branches available here and do not
> constitute evidence of positive selection.

## 4. Reviewer points incorporated

**Reviewer 1, comment 5 / Reviewer 2, comment 2 — omega>1 on short branches;
"pervasive differentiation" overstated.**

- codeml dN/dS estimates are unstable and upward-biased on short,
  low-substitution branches; a point estimate of omega > 1 for an individual
  gene/branch (e.g. genes such as FDFT1 or BLK cited by the reviewers) is a
  noisy estimate, not a statistical detection of positive selection. The only
  selection claim the committed analysis supports is the CmC H1-vs-H0 LRT, and
  here that test is non-significant for every gene after FDR correction.
- The phrase **"pervasive differentiation"** should be softened. Recommended
  replacements: "we did not detect significant orientation-associated shifts in
  selective constraint after multiple-testing correction," or, if describing
  point estimates, "point estimates were variable and frequently above one but
  were not statistically distinguishable between orientations." Avoid wording
  that implies widespread or pervasive positive/diversifying selection.

## 5. Optional branch-model capability (not enabled, no results claimed)

The repository retains a disabled branch-model two-ratio path (`model = 2`,
`NSsites = 0`) behind `RUN_BRANCH_MODEL_TEST` / the `run_branch_model`
argument. It is opt-in only, is **off** in the committed configuration, and
produced **none** of the committed results (no `bm_*` columns exist in
`data/GRAND_PAML_RESULTS.tsv`). It is documented and clearly marked as dead/
opt-in in the code so it cannot be mistaken for the active analysis. No genuine
single-branch / branch-site test on the inversion-leading branch was run; a full
re-run (via `combine_paml.yml` on GitHub Actions) would be required to produce
one and is out of scope for this reconciliation.
