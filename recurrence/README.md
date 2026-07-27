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
| full (13-feature) | 0.927 | 0.826 | 0.102 |
| transferable (8-feature) | 0.890 | 0.810 | 0.117 |
| parsimony-count rule (≥ 2 origins) | 0.794 | **0.000** | 0.206 |

The parsimony-count baseline has zero power in the low-FPR region the classifier
targets; the learned model is what recovers it.

Real-inversion application (`results/real_scores.csv`): a continuous recurrence score +
binary call for all 292 balanced inversions (180 usable given ≥ 2 haplotypes per
orientation; 120 non-consensus loci scored beyond the labelled set), each with a
`low_confidence` flag when an orientation has < 4 informative haplotypes. Concordance
with the manuscript's 60 usable consensus calls (`results/concordance.json`,
AUC 0.71, κ 0.42) is recorded as a **consistency check, not independent validation** —
the consensus labels share signals with several classifier features (the caveat is
carried in the JSON). The non-circular validation is the simulation ground truth.

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
- **held-out AUC** — the recorded 0.927 full / 0.890 transferable held-out simulation
  AUC and 0.826 power @ FPR ≤ 0.10 reproduce;
- **calibration argument order** — the calibration table is computed as
  `_calibration(labels, scores)` (binned by the continuous scores, observed label
  frequency tracking predicted score), and the swapped `(scores, labels)` call is
  detectably wrong (collapses to ≤ 2 label bins); the committed calibration tables
  reproduce from the training set;
- **parsimony baseline** — the parsimony-count rule reproduces at 0.000 power @ FPR ≤ 0.10;
- **determinism / order-invariance** — `classify()` returns the same origin count under
  arbitrary haplotype re-ordering (rows are canonicalized before the alignment is built,
  since IQ-TREE's search depends on sequence order). These tests are skipped where no
  IQ-TREE binary or Biopython is available;
- **feature reproduction** — the extractor reproduces pinned golden feature values on a fixture;
- **score reproduction** — re-scoring the real inversions reproduces the recorded
  `real_scores.csv` (per-inversion score to 1e-6) and `concordance.json` summary.
