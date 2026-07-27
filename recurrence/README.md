# `recurrence/` — validated pop-gen recurrence classifier for polymorphic inversions

Reproducible analysis code + recorded result tables for classifying whether a
polymorphic inversion arose **once** (single origin) or **recurrently** (multiple
independent origins) from per-inversion population-genetic signals, validated against
structured-coalescent simulation ground truth and applied to the balanced inversion set.

**Scope:** analysis code and the tables it produces. There is no interpretive prose here
— the "what it means" lives with the study. Every reported number is a recorded output
of the code in this directory, reproducible from committed inputs.

## Method

Per-inversion features (see `features.py`) → a deterministic logistic classifier with a
partial-AUC-at-low-FPR refinement (see `classifier.py`), fit and evaluated on labelled
structured-coalescent simulations, then applied to the real inversions.

Two feature sets / classifiers are fit from the same simulations and split:

| Classifier | Features | Role |
|---|---|---|
| **full** (13-feature) | haplotype-tree parsimony origin count, tag-SNP r², π ratio / inverted π, inverted-lineage dispersion, pairwise-distance structure, Hudson F<sub>ST</sub>, segregating-site count | the headline simulation-validated model |
| **transferable** (8-feature) | the diversity/differentiation features computable identically on simulations and on ferromic `output.csv` (π, θ, Hudson F<sub>ST</sub>/d<sub>xy</sub>, segregating sites, inversion frequency) | the model applied to the real inversions (no domain shift in feature definitions) |

The classifier objective maximizes a smooth surrogate of the partial AUC over the
FPR ≤ 0.10 region, L2-anchored to a logistic warm start.

The tree-origin-count feature is **the reference recurrence classifier itself**: the
Fitch small-parsimony minimum number of orientation state changes on an IQ-TREE
maximum-likelihood haplotype tree with the ancestral outgroup collapsed — i.e.
`minMutHomoplasy` as computed by
[`hsiehphLab/inversionSimulation`](https://github.com/hsiehphLab/inversionSimulation),
the pipeline behind the manuscript's Fig. 1G. The pipeline lives in
[`simulations/refsim/`](../simulations/refsim/), which documents the file-by-file
correspondence with upstream; `parsimony.py` is the entry point for callers that hold a
genotype matrix rather than a simulated tree sequence. Scoring a locus therefore needs
Biopython and an IQ-TREE binary; the committed training set carries the counts already,
so `fit` and `score` do not.

## Recorded results (committed, `results/`)

Held-out simulation metrics (30% of 11,250 labelled loci, spanning every axis cell):

| Classifier | AUC | Power @ FPR ≤ 0.10 | Brier |
|---|---|---|---|
| full (13-feature) | 0.913 | 0.837 | 0.105 |
| transferable (8-feature) | 0.914 | 0.840 | — |
| reference parsimony rule (≥ 2 origins) | 0.831 | **0.000** | — |

The reference rule has zero power in the low-FPR region the classifier targets, because
its own false-positive rate on these simulations is **0.155**, well above 0.10 — see
below. The learned model is what recovers power there.

### What the reference classifier does on this grid

Scoring all 11,250 loci through the upstream pipeline (`simulations/refsim/`):

| | single-origin FPR | recurrent power |
|---|---|---|
| overall | **0.155** | **0.805** |

by recombination rate (single-origin FPR):

| depth | ρ = 0 | ρ = 10⁻⁸ | ρ = 10⁻⁶ |
|---|---|---|---|
| recent | 0.502 | 0.459 | 0.013 |
| young | 0.243 | 0.117 | 0.000 |
| old | 0.051 | 0.008 | 0.000 |

and across between-orientation flux:

| m_flux | 0 | 10⁻⁹ | 10⁻⁸ | 10⁻⁷ | 10⁻⁶ |
|---|---|---|---|---|---|
| single-origin FPR | 0.149 | 0.158 | 0.159 | 0.152 | 0.156 |
| recurrent power | 0.804 | 0.789 | 0.806 | 0.808 | 0.816 |

Two things follow.

* **Flux does not degrade the classifier.** On the corrected 5,400-locus sweep the
  single-origin false-positive rate goes 0.113 → 0.106 (p = 0.70) and power 0.891 → 0.907
  (p = 0.36) across m = 0 → 10⁻⁶ — neither moves detectably. This is the manuscript's
  claim, measured with the manuscript's own classifier.
* **The false-positive rates above are from a mis-specified single-event scenario and
  are being regenerated.** Constraining only the inverted-deme admixture proportion does
  not produce a single-origin sample: direct haplotypes could still be drawn from the
  deme sister to the inverted one, and those lineages force extra parsimony steps. On a
  genuine single-origin sample the rate is 1.6%, consistent with the manuscript's < 5%.
  See `simulations/refsim/README.md`.

## Layout

```
recurrence/
  classifier.py     deterministic logistic + partial-AUC-at-low-FPR refinement + metrics
  features.py       per-inversion pop-gen feature extractor (full 13-feature set + per-bp stats)
  transferable.py   the 8 sim/real-transferable features (from sim stats or output.csv)
  parsimony.py      reference origin count (IQ-TREE ML tree + Fitch) for a genotype matrix
  simulate.py       stage 1: training-set generation / shard merge (uses simulations/refsim)
  fit.py            stage 2: fit + validate both classifiers on the sims
  apply.py          stage 3: score the real inversions + consensus concordance
  cli.py            unified `python -m recurrence.cli {simulate,fit,score}` entrypoint
  paths.py          committed-input/result locations + provenance recording
  data/             committed inputs (see below)
  results/          committed recorded result tables (the reference outputs)
  tests/            reproduction / QC tests (see "Verification")
```

## Inputs (committed, `data/`)

| File | What |
|---|---|
| `sim_features.csv.gz` | labelled simulation training set (11,250 loci × features); regenerable via the `simulate` stage |
| `inv_properties.tsv` | balanced inversion set + consensus single/recurrent labels |
| `output.csv` | ferromic canonical per-inversion π / θ / F<sub>ST</sub> / d<sub>xy</sub> (the real-data feature source) |

## Reproduce

```bash
pip install -e ".[recurrence]"          # numpy, scipy, scikit-learn, pandas, pytest

# stage 2 — fit + validate both classifiers on the committed training set (no msprime):
python -m recurrence.cli fit            # -> results/{model,transferable_model,sim_metrics,tf_sim_metrics}.json + sim_test_pred*.csv.gz

# stage 3 — score the real inversions + consensus concordance (no msprime):
python -m recurrence.cli score          # -> results/{real_scores.csv, concordance.json, concordance_disagreements.csv}

# stage 1 (optional) — regenerate the labelled training set.
# Needs msprime, Biopython and an IQ-TREE binary; run the grid sharded on a cluster
# (see simulations/refsim/README.md), then merge the shards:
pip install -e ".[recurrence-sim]"
python -m recurrence.cli simulate --merge 'simulations/refsim/out/trainset_shard*.csv'
```

`fit` and `score` are deterministic (fixed split `seed % 10 ∈ {7,8,9}` = test;
classifier `random_state=42`); the `simulate` stage seeds every grid cell from its grid
index, and each locus's IQ-TREE search from that same seed.

Each generated output writes a provenance sidecar (`*_provenance.json`) recording
the resolved inputs and library versions.

## Verification

```bash
pytest recurrence/tests -q
```

The reproduction / QC tests (run in CI, `.github/workflows/recurrence.yml`, from
committed data only — no msprime, no network):

- **coefficient recovery** — re-fitting from the committed training set recovers the
  recorded model coefficients (warm-logistic to 1e-4; pAUC-refined to 1e-2);
- **held-out AUC** — the recorded 0.913 full / 0.914 transferable held-out simulation
  AUC and 0.837 power @ FPR ≤ 0.10 reproduce;
- **calibration argument order** — the calibration table is computed as
  `_calibration(labels, scores)` (binned by the continuous scores, observed label
  frequency tracking predicted score), and the swapped `(scores, labels)` call is
  detectably wrong (collapses to ≤ 2 label bins); the committed calibration tables
  reproduce from the training set;
- **parsimony baseline** — the reference parsimony rule reproduces at 0.000 power
  @ FPR ≤ 0.10 (its own FPR on these simulations is 0.155);
- **determinism / order-invariance** — `classify()` returns the same origin count under
  arbitrary haplotype re-ordering (rows are canonicalized before the alignment is built,
  since IQ-TREE's search depends on sequence order). These tests are skipped where no
  IQ-TREE binary or Biopython is available;
- **feature reproduction** — the extractor reproduces pinned golden feature values on a fixture;
- **score reproduction** — re-scoring the real inversions reproduces the recorded
  `real_scores.csv` (per-inversion score to 1e-6) and `concordance.json` summary.
