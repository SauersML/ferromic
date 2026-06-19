# Between-orientation flux and the recurrence classifier (Reviewer 1.1 / Reviewer 3.2)

The structured-coalescent recurrence classifier used in the manuscript (and in
Porubsky et al. 2022) assumes **no exchange between inverted and direct
orientations**. Reviewers noted that gene conversion and double crossover in
inversion heterozygotes produce real between-orientation flux, and asked whether
ignoring it undermines the single-event vs recurrent classification.

This directory holds a simulation experiment that answers that question by adding
a flux term to the model and sweeping its rate.

## Model

Faithful msprime (1.4.2) reproduction of `hsiehphLab/inversionSimulation`
(`recurrentINV_m1.2pop.py`): ancestral `Ne = 6000`, `mu = 1.25e-8`, generation
time 25 y, locus 200 kbp. An inversion event = divergence creating an isolated
opposite-orientation deme with a 90% bottleneck; gene flow (`mig_const`) is
**same-orientation only** — that is the assumption being tested. The no-flux
baseline reproduces the manuscript behavior (single-event -> 1 inferred event;
3-recurrent -> ~3).

**Flux** is added as symmetric inverted<->direct deme migration at rate `m` (the
coalescent dual of a gene-conversion / double-crossover tract spanning the
inversion), swept over `m ∈ {0, 1e-9, 1e-8, 1e-7, 1e-6}` plus an extreme
extension `{3e-6, 1e-5, 3e-5, 1e-4}`.

**Classifier**: minimum number of orientation state changes on the haplotype tree
by Fitch parsimony; called recurrent if score ≥ 2 (the Porubsky et al. 2022
criterion). Single-event FPR = recurrent-call rate under the single-event model;
recurrent power = recurrent-call rate under the 3-event model. 60 reps/cell,
crossed with depth (young/recent/old) and recombination (0, 1e-8, 1e-6).

## Result

Over the **biologically plausible** flux range (`m ≤ 1e-6`), flux does **not**
materially change either the false-positive rate or power — both stay within
sampling noise of the no-flux baseline at every depth and recombination rate.
Classification accuracy is governed by divergence depth and recombination, not by
flux. Example (rho = 1e-8, recent depth):

- single-event FPR vs `m` (0/1e-9/1e-8/1e-7/1e-6): 0.35 / 0.27 / 0.33 / 0.30 / 0.32 — flat
- recurrent power vs `m`: 0.83 / 0.78 / 0.77 / 0.80 / 0.82 — flat

The high baseline FPR at "recent" depth is a tree-resolution effect (it vanishes
at rho = 1e-6, FPR = 0 everywhere), not a flux effect. The classifier only breaks
down under **implausibly extreme** flux, and then mainly by inflating the inferred
*number* of events: single-event young FPR 0.02 (1e-6) -> 0.40 (1e-5) -> 0.97
(1e-4); mean inferred events ~7 vs true 3 at m = 1e-4.

Full per-cell tables: `flux_results_tables.md`, `flux_results.csv`.

## Caveat

For tractability of the large sweep, trees are built by neighbor-joining and the
recurrence call uses Fitch parsimony, rather than the pipeline's IQ-TREE ML path.
The parsimony score is topology-driven and the no-flux baseline reproduces the
expected single/recurrent behavior, so the conclusion is robust; a
pipeline-identical IQ-TREE re-run is possible but much slower.

## Reproducing

Requires `msprime`. With a venv that has it:

```bash
python run_sweep_par.py --reps 60 --depths young,recent,old \
    --rhos 0,1e-8,1e-6 --scenarios single,recurrent --procs 7 --out sweep_full.json
python make_report.py        # -> flux_results.csv, flux_results_tables.md, flux_fpr_power.png
python run_extreme.py && python make_extreme_fig.py   # extreme-flux extension + flux_breakdown.png
```

Figures (`*.png`) are gitignored by repo convention and regenerate from the JSON
results via the report scripts.
