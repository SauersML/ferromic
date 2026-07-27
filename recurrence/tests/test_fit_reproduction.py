"""Reproduction QC for the fit stage: re-fit both classifiers from the committed
training set and assert the recorded model coefficients and held-out simulation metrics
reproduce. Runs from committed data only (no msprime)."""
import functools
import json

import numpy as np
import pandas as pd

from recurrence import classifier as C
from recurrence import fit, paths
from recurrence.features import FEATURE_NAMES
from recurrence.transferable import TRANSFERABLE_FEATURES


@functools.lru_cache(maxsize=1)
def _refit():
    df = pd.read_csv(paths.SIM_FEATURES)
    train, test = fit._split(df)
    full_model, full_metrics, _ = fit.fit_full(train.copy(), test.copy())
    tf_model, tf_metrics = fit.fit_transferable(train.copy(), test.copy())
    return full_model, full_metrics, tf_model, tf_metrics


def test_full_coefficients_reproduce():
    model, _, _, _ = _refit()
    ref = json.load(open(paths.MODEL_FULL))
    assert ref["feature_names"] == FEATURE_NAMES
    # warm-start logistic is convex -> reproduces very tightly
    assert np.max(np.abs(np.array(model.warm_w) - np.array(ref["warm_w"]))) < 1e-4
    # pAUC L-BFGS-B refinement -> looser tolerance across BLAS/platforms
    assert np.max(np.abs(np.array(model.w) - np.array(ref["w"]))) < 1e-2
    assert abs(model.b - ref["b"]) < 1e-2


def test_held_out_sim_auc_reproduces():
    _, m, _, _ = _refit()
    ref = json.load(open(paths.SIM_METRICS))
    # headline validated numbers, from the reference-classifier training set:
    # AUC ~0.913, power@FPR<=0.10 ~0.837, Brier ~0.105
    assert abs(m["test"]["auc"] - 0.913) < 3e-3
    assert abs(m["test"]["auc"] - ref["test"]["auc"]) < 3e-3
    assert abs(m["test"]["power_at_fpr10"] - 0.837) < 2e-2
    assert abs(m["test"]["brier"] - 0.105) < 5e-3


def test_parsimony_baseline_zero_power():
    """The parsimony-count rule has no power in the low-FPR region the classifier
    targets -- the contrast that motivates the learned model."""
    _, m, _, _ = _refit()
    assert m["test_parsimony_rule"]["power_at_fpr10"] == 0.0


def test_transferable_reproduces():
    _, _, tf_model, tf_metrics = _refit()
    ref = json.load(open(paths.MODEL_TRANSFERABLE))
    assert ref["feature_names"] == TRANSFERABLE_FEATURES
    assert abs(tf_metrics["test"]["auc"] - 0.914) < 3e-3
    assert abs(tf_metrics["test"]["power_at_fpr10"] - 0.840) < 2e-2


def test_fit_is_deterministic():
    """Two independent fits from the same training set give identical coefficients."""
    df = pd.read_csv(paths.SIM_FEATURES)
    train, test = fit._split(df)
    m1 = C.fit(train[FEATURE_NAMES].values, train["label"].values, FEATURE_NAMES)
    m2 = C.fit(train[FEATURE_NAMES].values, train["label"].values, FEATURE_NAMES)
    assert np.allclose(np.array(m1.w), np.array(m2.w), atol=1e-10)


def _is_binary(vals):
    return set(np.unique(np.asarray(vals))).issubset({0, 1})


def test_calibration_argument_order():
    """Guard against the (label, score) -> (score, score) argument swap in the
    calibration table: `_calibration(y, s)` must bin by the continuous scores `s`
    (many populated bins, monotone-increasing mean_pred that observed frequency
    tracks), NOT by the binary labels `y` (which would populate <= 2 bins)."""
    df = pd.read_csv(paths.SIM_FEATURES)
    train, test = fit._split(df)
    _, full_metrics, _ = fit.fit_full(train.copy(), test.copy())
    calib = full_metrics["calibration_test"]

    # binning the continuous scores populates many bins; binning the 0/1 labels
    # (the swapped call) would populate at most 2.
    assert len(calib) >= 5
    preds = [r["mean_pred"] for r in calib]
    obs = [r["obs_freq"] for r in calib]
    assert preds == sorted(preds), "mean_pred not monotone -> not binned by score"
    # every bin's mean predicted score lies inside that bin's score range
    for r in calib:
        assert r["bin_lo"] <= r["mean_pred"] <= r["bin_hi"]
    # observed label frequency tracks predicted score (calibration is real, not swapped)
    assert np.corrcoef(preds, obs)[0, 1] > 0.9

    # explicit swap detector: recompute scores and feed _calibration in both orders.
    # correct (labels, scores) bins by scores -> many bins and equals the committed
    # table; swapped (scores, labels) bins the 0/1 labels -> collapses to <= 2 bins.
    y = test["label"].values
    m = C.fit(train[FEATURE_NAMES].values, train["label"].values, FEATURE_NAMES)
    sc = m.score(test[FEATURE_NAMES].values)
    assert _is_binary(y)
    correct = fit._calibration(y, sc)
    swapped = fit._calibration(sc, y)
    assert len(correct) >= 5
    assert len(swapped) <= 2, "swapped (score, label) call should collapse to <=2 label bins"
    assert correct == calib  # the committed table uses the correct order


def test_committed_calibration_matches_recompute():
    """The committed calibration_test tables reproduce from the committed training set."""
    _, full_metrics, _, tf_metrics = _refit()
    for recomputed, ref_path in ((full_metrics, paths.SIM_METRICS),
                                 (tf_metrics, paths.TF_SIM_METRICS)):
        ref = json.load(open(ref_path))
        got, want = recomputed["calibration_test"], ref["calibration_test"]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert abs(g["mean_pred"] - w["mean_pred"]) < 1e-6
            assert abs(g["obs_freq"] - w["obs_freq"]) < 1e-6
            assert g["n"] == w["n"]
