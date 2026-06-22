"""Regression tests for BUG #3: zero-coverage / impossible-dosage guard.

An imputation model whose predictor matrix is all-missing (-127) used to be run anyway:
missing entries were mean-filled, the undefined column means became 0, and the model
predicted from an all-zero vector, emitting impossible dosages outside [0, 2]. The fix
fails closed on zero coverage and clamps any out-of-range predictions into [0, 2].
"""
import os
import sys

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import infer_dosage  # noqa: E402
from infer_dosage import assess_coverage, MISSING_VALUE_CODE  # noqa: E402


def test_assess_coverage_all_missing():
    X = np.full((10, 4), MISSING_VALUE_CODE, dtype=np.int8)
    covered_mask, n_covered, overall = assess_coverage(X, min_snp_call_rate=0.01)
    assert n_covered == 0
    assert overall == 0.0
    assert not covered_mask.any()


def test_assess_coverage_partial():
    X = np.full((10, 3), MISSING_VALUE_CODE, dtype=np.int8)
    X[:, 0] = 1            # fully covered column
    X[:5, 1] = 0           # 50% covered column
    # column 2 stays all-missing
    covered_mask, n_covered, overall = assess_coverage(X, min_snp_call_rate=0.1)
    assert covered_mask.tolist() == [True, True, False]
    assert n_covered == 2


def _fit_out_of_range_model(n_snps):
    """A linear model whose prediction on an all-zero vector is far outside [0, 2]."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, n_snps))
    # Large intercept => all-zero input predicts ~ -40 (impossible dosage).
    y = -40.0 + X[:, 0] * 5.0
    return LinearRegression().fit(X, y)


def _run_worker(tmpdir, model_name, X_matrix, model):
    geno_dir = os.path.join(tmpdir, "genotype_matrices")
    model_dir = os.path.join(tmpdir, "impute")
    temp_dir = os.path.join(tmpdir, "temp_dosages")
    for d in (geno_dir, model_dir, temp_dir):
        os.makedirs(d, exist_ok=True)

    np.save(os.path.join(geno_dir, f"{model_name}.genotypes.npy"), X_matrix)
    joblib.dump(model, os.path.join(model_dir, f"{model_name}.model.joblib"))

    # Point module globals at our temp dirs.
    infer_dosage.GENOTYPE_DIR = geno_dir
    infer_dosage.MODEL_DIR = model_dir
    infer_dosage.TEMP_RESULT_DIR = temp_dir
    infer_dosage.MODEL_SOURCE_DIR = ""

    n_samples = X_matrix.shape[0]
    ancestry_indices = np.zeros(n_samples, dtype=np.int8)
    inv_anc_map = {0: "eur", 7: "unk"}
    args = (model_name, n_samples, ancestry_indices, 7, inv_anc_map)
    res = infer_dosage._process_model_batched(args)
    out_npy = os.path.join(temp_dir, f"{model_name}.npy")
    preds = np.load(out_npy) if os.path.exists(out_npy) else None
    return res, preds


def test_all_missing_matrix_is_skipped(tmp_path):
    """All-missing predictor matrix must fail closed (skipped, no dosages emitted)."""
    n_samples, n_snps = 20, 5
    X = np.full((n_samples, n_snps), MISSING_VALUE_CODE, dtype=np.int8)
    model = _fit_out_of_range_model(n_snps)
    res, preds = _run_worker(str(tmp_path), "chrTEST-1-INV-1", X, model)
    assert res["status"] == "skipped_low_coverage"
    # No dosage file should have been written.
    assert preds is None


def test_out_of_range_predictions_are_clamped(tmp_path):
    """A covered model whose raw predictions fall outside [0, 2] must be clamped,
    never emitted as impossible dosages."""
    n_samples, n_snps = 30, 5
    # Fully covered genotypes (all zeros are valid 0-dosage calls, not missing).
    X = np.zeros((n_samples, n_snps), dtype=np.int8)
    model = _fit_out_of_range_model(n_snps)

    # Sanity: the raw model really does predict an impossible value here.
    raw = float(model.predict(np.zeros((1, n_snps)))[0])
    assert raw < infer_dosage.DOSAGE_MIN or raw > infer_dosage.DOSAGE_MAX

    res, preds = _run_worker(str(tmp_path), "chrTEST-2-INV-1", X, model)
    assert res["status"] == "ok"
    assert preds is not None
    assert len(preds) == n_samples
    # Every emitted dosage is within the valid range.
    assert np.all(preds >= infer_dosage.DOSAGE_MIN)
    assert np.all(preds <= infer_dosage.DOSAGE_MAX)
    # And the clamp actually fired.
    assert res.get("n_clamped", 0) > 0
