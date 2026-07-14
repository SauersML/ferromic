"""Stage 2 (fit + validate): fit the recurrence classifier on the simulation training
set and evaluate it against simulation ground truth on a held-out split, stratified by
the manuscript's Fig-1 axes. Deterministic.

Two classifiers are fit from the same training set and split:

* full          -- all 13 pop-gen features (features.FEATURE_NAMES); the headline
                   simulation-validated model.
* transferable  -- the 8 diversity/differentiation features computable identically on
                   simulations and on the real inversions (transferable.TRANSFERABLE_FEATURES);
                   this is the model applied to real data in the ``score`` stage.

Held-out split: test = rows with (seed % 10) in {7, 8, 9} (30%), spanning every axis
cell; fit on the remaining 70%. The full model is compared against its logistic warm
start and the parsimony-count rule (tree_n_events >= 2) as baselines.

Outputs (to results/): model.json, transferable_model.json, sim_metrics.json,
tf_sim_metrics.json, sim_test_pred.csv.gz, tf_sim_test_pred.csv.gz, and provenance.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import classifier as C
from . import paths
from .features import FEATURE_NAMES
from .transferable import TRANSFERABLE_FEATURES

TEST_FOLDS = [7, 8, 9]


def _split(df):
    fold = df["seed"] % 10
    test = df[fold.isin(TEST_FOLDS)].copy()
    train = df[~fold.isin(TEST_FOLDS)].copy()
    return train, test


def _strat_metrics(df, score_col):
    rows = []
    for axis in ["depth", "rho", "m_flux", "inv_freq"]:
        if axis not in df.columns:
            continue
        for val, g in df.groupby(axis):
            if g["label"].nunique() < 2:
                continue
            rows.append(dict(axis=axis, value=val, **C.evaluate(g["label"].values, g[score_col].values)))
    return rows


def _calibration(y, s, nbins=10):
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(s, bins) - 1, 0, nbins - 1)
    out = []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 0:
            out.append(dict(bin_lo=float(bins[b]), bin_hi=float(bins[b + 1]),
                            mean_pred=float(s[m].mean()), obs_freq=float(y[m].mean()),
                            n=int(m.sum())))
    return out


def fit_full(train, test):
    model = C.fit(train[FEATURE_NAMES].values, train["label"].values, FEATURE_NAMES)
    for part in (train, test):
        part["score"] = model.score(part[FEATURE_NAMES].values)
        part["warm_score"] = model.warm_score(part[FEATURE_NAMES].values)
        part["parsimony"] = (part["tree_n_events"] >= 2).astype(float)
    metrics = {
        "feature_set": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "fmax": C.FMAX,
        "train": C.evaluate(train.label.values, train.score.values),
        "test": C.evaluate(test.label.values, test.score.values),
        "test_warm_logistic": C.evaluate(test.label.values, test.warm_score.values),
        "test_parsimony_rule": C.evaluate(test.label.values, test.parsimony.values),
        "calibration_test": _calibration(test.label.values, test.score.values),
        "strata_test": _strat_metrics(test, "score"),
        "weights_pauc": {n: float(w) for n, w in zip(FEATURE_NAMES, model.w)},
        "weights_warm_logistic": {n: float(w) for n, w in zip(FEATURE_NAMES, model.warm_w)},
    }
    return model, metrics, test


def fit_transferable(train, test):
    cols = ["tf_" + f for f in TRANSFERABLE_FEATURES]
    model = C.fit(train[cols].values, train["label"].values, TRANSFERABLE_FEATURES)
    for part in (train, test):
        part["tf_score"] = model.score(part[cols].values)
    metrics = {
        "feature_set": TRANSFERABLE_FEATURES,
        "fmax": C.FMAX,
        "train": C.evaluate(train.label.values, train.tf_score.values),
        "test": C.evaluate(test.label.values, test.tf_score.values),
        "weights": {n: float(w) for n, w in zip(TRANSFERABLE_FEATURES, model.w)},
        "calibration_test": _calibration(test.label.values, test.tf_score.values),
        "strata_test": _strat_metrics(test.assign(score=test.tf_score), "score"),
    }
    return model, metrics


def run(sims=None, outdir=None):
    sims = sims or paths.SIM_FEATURES
    outdir = outdir or paths.RESULTS
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(sims)
    train, test = _split(df)
    print(f"train {len(train)} / test {len(test)}  (pos frac train={train.label.mean():.3f})")

    full_model, full_metrics, full_test = fit_full(train.copy(), test.copy())
    tf_model, tf_metrics = fit_transferable(train.copy(), test.copy())

    with open(os.path.join(outdir, "model.json"), "w") as fh:
        json.dump(asdict(full_model), fh, indent=2)
    with open(os.path.join(outdir, "transferable_model.json"), "w") as fh:
        json.dump(asdict(tf_model), fh, indent=2)
    with open(os.path.join(outdir, "sim_metrics.json"), "w") as fh:
        json.dump(full_metrics, fh, indent=2)
    with open(os.path.join(outdir, "tf_sim_metrics.json"), "w") as fh:
        json.dump(tf_metrics, fh, indent=2)

    full_cols = ["scenario", "label", "depth", "rho", "m_flux", "inv_freq", "seed",
                 "score", "warm_score", "parsimony"] + FEATURE_NAMES
    full_test[full_cols].to_csv(os.path.join(outdir, "sim_test_pred.csv.gz"), index=False)
    tf_cols = ["tf_" + f for f in TRANSFERABLE_FEATURES]
    tf_test = test.copy()
    tf_test["score"] = tf_model.score(tf_test[tf_cols].values)
    tf_test[["scenario", "label", "depth", "rho", "inv_freq", "seed", "score"] + tf_cols].to_csv(
        os.path.join(outdir, "tf_sim_test_pred.csv.gz"), index=False)

    paths.write_provenance(os.path.join(outdir, "fit_provenance.json"),
                           {"sim_features": sims},
                           extra={"test_folds": TEST_FOLDS})

    t = full_metrics["test"]
    p = full_metrics["test_parsimony_rule"]
    tf = tf_metrics["test"]
    print("\n=== held-out sim metrics ===")
    print(f"  full pAUC classifier : AUC={t['auc']:.3f}  pAUC@{C.FMAX}={t['pauc_fmax']:.3f}  "
          f"power@FPR<={C.FMAX}={t['power_at_fpr10']:.3f}  Brier={t['brier']:.3f}")
    print(f"  parsimony rule (>=2) : AUC={p['auc']:.3f}  power@FPR<={C.FMAX}={p['power_at_fpr10']:.3f}")
    print(f"  transferable         : AUC={tf['auc']:.3f}  power@FPR<={C.FMAX}={tf['power_at_fpr10']:.3f}")
    return full_metrics, tf_metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit + validate the recurrence classifier on sims.")
    ap.add_argument("--sims", default=None, help="training set CSV (default: committed data/sim_features.csv.gz)")
    ap.add_argument("--outdir", default=None, help="output dir (default: recurrence/results)")
    args = ap.parse_args(argv)
    run(sims=args.sims, outdir=args.outdir)


if __name__ == "__main__":
    main()
