"""Reproduction QC for the score stage: re-score the real inversions from the committed
transferable model + committed per-inversion tables and assert the recorded score table
and concordance summary reproduce. Runs from committed data only."""
import json

import numpy as np
import pandas as pd

from recurrence import apply, paths


def test_real_scores_reproduce():
    real, conc = apply.run(outdir="/tmp/recurrence_score_test")
    ref = pd.read_csv(paths.REAL_SCORES)
    assert len(real) == len(ref)
    # continuous score reproduces per inversion (usable rows)
    got = real.set_index("inv_id")["recurrence_score"]
    want = ref.set_index("inv_id")["recurrence_score"]
    common = got.dropna().index.intersection(want.dropna().index)
    assert len(common) > 100
    assert np.max(np.abs(got.loc[common].values - want.loc[common].values)) < 1e-6


def test_concordance_summary_reproduces():
    _, conc = apply.run(outdir="/tmp/recurrence_score_test")
    ref = json.load(open(paths.CONCORDANCE))
    assert conc["n_consensus_scored"] == ref["n_consensus_scored"]
    assert conc["coverage"] == ref["coverage"]
    assert abs(conc["auc_vs_consensus"] - ref["auc_vs_consensus"]) < 1e-6
    assert abs(conc["threshold_fpr_fmax"] - ref["threshold_fpr_fmax"]) < 1e-6
    assert conc["n_disagreements"] == ref["n_disagreements"]


def test_low_confidence_flag_present():
    real, _ = apply.run(outdir="/tmp/recurrence_score_test")
    # every non-usable inversion is flagged low-confidence, and the flag column exists
    assert "low_confidence" in real.columns
    non_usable = real[real["usable"] != True]  # noqa: E712
    assert bool(non_usable["low_confidence"].all())
