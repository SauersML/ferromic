"""Coverage and biological-range guards for dosage inference."""

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from imputation import infer_dosage, pls_patch


def _model(n_snps: int, outcome_offset: float = 1.0) -> Pipeline:
    rng = np.random.default_rng(0)
    predictors = rng.integers(0, 3, size=(80, n_snps)).astype(np.float64)
    outcome = outcome_offset + predictors[:, 0] * 0.1
    return Pipeline(
        [("pls", pls_patch.PLSRegression(n_components=min(2, n_snps)))]
    ).fit(predictors, outcome)


def test_all_missing_matrix_fails_closed(tmp_path):
    matrix = np.full((20, 5), infer_dosage.MISSING_INT8, dtype=np.int8)
    matrix_path = tmp_path / "matrix.npy"
    model_path = tmp_path / "model.joblib"
    np.save(matrix_path, matrix)
    joblib.dump(_model(5), model_path)
    ancestry = np.zeros(matrix.shape[0], dtype=np.int8)

    with pytest.raises(RuntimeError, match="no predictors"):
        infer_dosage._predict(
            "test-model",
            matrix_path,
            model_path,
            ancestry,
            batch_size=10,
            threads=1,
        )


def test_out_of_range_predictions_are_clamped(tmp_path):
    matrix = np.zeros((30, 5), dtype=np.int8)
    matrix_path = tmp_path / "matrix.npy"
    model_path = tmp_path / "model.joblib"
    np.save(matrix_path, matrix)
    model = _model(5, outcome_offset=-40.0)
    joblib.dump(model, model_path)
    ancestry = np.zeros(matrix.shape[0], dtype=np.int8)

    raw = float(model.predict(np.zeros((1, 5)))[0])
    assert raw < infer_dosage.DOSAGE_MIN
    predictions, report = infer_dosage._predict(
        "test-model",
        matrix_path,
        model_path,
        ancestry,
        batch_size=10,
        threads=1,
    )

    assert np.all(predictions >= infer_dosage.DOSAGE_MIN)
    assert np.all(predictions <= infer_dosage.DOSAGE_MAX)
    assert report["clamped_predictions"] == len(predictions)


def test_partial_coverage_uses_training_mean_for_absent_predictors(tmp_path):
    matrix = np.array(
        [[0, infer_dosage.MISSING_INT8], [1, infer_dosage.MISSING_INT8]] * 10,
        dtype=np.int8,
    )
    matrix_path = tmp_path / "matrix.npy"
    model_path = tmp_path / "model.joblib"
    np.save(matrix_path, matrix)
    model = _model(2)
    joblib.dump(model, model_path)
    ancestry = np.zeros(matrix.shape[0], dtype=np.int8)

    predictions, report = infer_dosage._predict(
        "test-model",
        matrix_path,
        model_path,
        ancestry,
        batch_size=10,
        threads=1,
    )

    expected_input = matrix.astype(np.float32)
    expected_input[:, 1] = model.named_steps["pls"]._x_mean[1]
    expected = np.clip(
        np.asarray(model.predict(expected_input)).reshape(-1),
        infer_dosage.DOSAGE_MIN,
        infer_dosage.DOSAGE_MAX,
    )
    np.testing.assert_allclose(predictions, expected, rtol=1e-6, atol=1e-6)
    assert report["covered_predictors"] == 1
    assert report["training_mean_predictors"] == 1
