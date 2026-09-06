# Within-ancestry-PC PheWAS sensitivity results

These six tables are the final All of Us v8 sensitivity analyses for the 37
phenotypes implicated by the pooled PheWAS. Each phenotype was tested against
all seven inversions within one All of Us genetic-ancestry group after replacing
the global projected components with 16 principal components fitted inside that
group.

The groups are European (EUR), African (AFR), Admixed American (AMR), East Asian
(EAS), South Asian (SAS), and Middle Eastern (MID). Models required at least 100
cases and 100 controls. Consequently, EUR, AFR, and AMR contain all 259 possible
inversion-phenotype combinations; smaller groups contain only evaluable models.
Thirty EAS models for `chr6-141867315-INV-29159` are retained as explicitly
invalid rows because the inversion dosage was removed during design-matrix
pruning.

The phenotype set was selected from the original pooled findings. Therefore,
within-table q-values describe multiplicity within this sensitivity set and are
not independent replication statistics. The intended analysis is effect-size
correspondence with the existing ancestry-stratified estimates in
`data/phewas_results.tsv`. Run:

```bash
python stats/phewas_within_ancestry_figures.py
```

This writes the comparison table to
`data/phewas_within_ancestry_correspondence.tsv` and the figures and summary to
`results/phewas_within_ancestry/`.


## Confidence intervals

Rows whose `CI_Method` is `lrt_quadratic` carry 95% intervals derived from the
likelihood-ratio statistic of the model, `exp(beta ± 1.96·|beta|/z)` with `z`
the standard-normal quantile of the likelihood-ratio p-value. These replace the
`profile` intervals originally exported by the pipeline, whose constrained refit
did not reach the constrained maximum (fixed in phewas/models.py, c364c189).
The rebuild is performed and verified by `stats/rebuild_phewas_profile_cis.py`,
which also keeps `data/within_ancestry_pc_phewas_results.tsv` in step with the
six per-ancestry tables. `wald_mle` and `profile_penalized` (Firth) intervals
never used the defective refit and are unchanged.
