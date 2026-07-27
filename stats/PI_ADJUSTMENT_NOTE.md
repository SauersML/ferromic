# Which adjusted π model to report

Two committed artifacts give opposite answers for the orientation-by-recurrence
interaction in nucleotide diversity:

| source | adjustment set | interaction | p |
|---|---|---|---|
| `data/recurrence_controls_summary.tsv` (covariate-adjusted) | inversion length, inverted AF, SNP density, CDS density | fold-change 4.438 (95% CI 1.947–10.119) | **0.000394** |
| `data/recurrence_controls_summary.tsv` (matched) | matched on length + AF | fold-change 3.922 (95% CI 1.868–8.606) | 0.00360 |
| `data/replicate_manuscript_statistics.txt`, Model A (unadjusted) | none | fold-change 4.149 (95% CI 2.086–8.254) | 5.02e-05 |
| `data/replicate_manuscript_statistics.txt`, **Model C** | ln1p(`Number_recurrent_events`), ln(`Size_.kbp.`), `Inverted_AF`, ln(`Formation_rate_per_generation`) | fold-change 0.996 (95% CI 0.403–2.464) | **0.993** |

**Report the covariate-adjusted row of `recurrence_controls_summary.tsv`
(p = 0.00039). Do not report Model C as an adjusted estimate of the same effect.**

## Why Model C is not estimating the same quantity

The contrast of interest is the difference in the orientation effect between
recurrent and single-event inversions. Recurrence status is the exposure. Model C
conditions on `Number_recurrent_events` and on `Formation_rate_per_generation` —
which are not confounders of that contrast, they are the exposure itself in
continuous form and a quantity derived from it. A locus is classified recurrent
*because* its inferred event count exceeds one, so `Recurrent` and
ln1p(`Number_recurrent_events`) encode the same variable at two resolutions.

The fit shows exactly what that does. In Model C the interaction collapses to
fold-change 0.996 (p = 0.993) while the signal reappears on the covariates:

```
Covariate: Number_recurrent_events (ln1p, z):    fold-change 0.479, p = 0.092
Covariate: Formation_rate_per_generation (ln, z): fold-change 3.618, p = 0.028
```

Conditioning on the event count leaves the `Recurrent` indicator with almost no
independent variation, so its interaction is estimated off residual noise. This
is over-adjustment for the exposure, not a robustness check that the effect
fails; note also that Model C's *marginal* orientation effects stay strongly
negative in both classes (single 0.355, p = 1.0e-04; recurrent 0.353,
p = 7.1e-03), which is not the behaviour of a model in which the diversity
signal has gone away.

## Why the `recurrence_controls_summary.tsv` adjustment is the right one

Reviewer 3's comment 3 asks whether the recurrent-vs-single comparisons control
for genomic architecture — inversion length, allele frequency, local SNP density,
gene density. Those are properties that plausibly differ between the two classes
and plausibly affect diversity, i.e. genuine confounders, and none of them is a
restatement of recurrence status. That is precisely the adjustment set in
`recurrence_controls_summary.tsv`, and the interaction survives it
(p = 0.00039), as it does under matching on length and AF (p = 0.0036).

The same table carries the corresponding adjusted results for the other two
outcomes, which should be quoted from the same source for consistency:

```
Hudson FST (Recurrent - Single)   covariate-adjusted  -0.2252  p = 0.00597
                                  matched             -0.1978  p = 0.0132
da = Dxy - pi_avg                 covariate-adjusted  -3.70e-4 p = 0.0161
                                  matched             -2.88e-4 p = 0.0814
```

## Action

`stats/inv_dir_recur_model.py` still fits Model C. Either drop it from the
released scripts or keep it explicitly labelled as a collinearity demonstration —
what it shows is that recurrence status carries no information once the event
count is conditioned on, which is a statement about the two variables, not about
diversity.
