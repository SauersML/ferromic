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
    # headline validated numbers: AUC ~0.927, power@FPR<=0.10 ~0.826, Brier ~0.102
    assert abs(m["test"]["auc"] - 0.927) < 3e-3
    assert abs(m["test"]["auc"] - ref["test"]["auc"]) < 3e-3
    assert abs(m["test"]["power_at_fpr10"] - 0.826) < 2e-2
    assert abs(m["test"]["brier"] - 0.102) < 5e-3


def test_parsimony_baseline_zero_power():
    """The parsimony-count rule has no power in the low-FPR region the classifier
    targets -- the contrast that motivates the learned model."""
    _, m, _, _ = _refit()
    assert m["test_parsimony_rule"]["power_at_fpr10"] == 0.0


def test_transferable_reproduces():
    _, _, tf_model, tf_metrics = _refit()
    ref = json.load(open(paths.MODEL_TRANSFERABLE))
    assert ref["feature_names"] == TRANSFERABLE_FEATURES
    assert abs(tf_metrics["test"]["auc"] - 0.890) < 3e-3
    assert abs(tf_metrics["test"]["power_at_fpr10"] - 0.810) < 2e-2


def test_fit_is_deterministic():
    """Two independent fits from the same training set give identical coefficients."""
    df = pd.read_csv(paths.SIM_FEATURES)
    train, test = fit._split(df)
    m1 = C.fit(train[FEATURE_NAMES].values, train["label"].values, FEATURE_NAMES)
    m2 = C.fit(train[FEATURE_NAMES].values, train["label"].values, FEATURE_NAMES)
    assert np.allclose(np.array(m1.w), np.array(m2.w), atol=1e-10)
