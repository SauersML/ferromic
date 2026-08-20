"""Coverage and biological-range guards for dosage inference."""

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from imputation import infer_dosage


def _out_of_range_model(n_snps: int) -> LinearRegression:
    rng = np.random.default_rng(0)
    predictors = rng.normal(size=(50, n_snps))
    outcome = -40.0 + predictors[:, 0] * 5.0
    return LinearRegression().fit(predictors, outcome)


def test_all_missing_matrix_fails_closed(tmp_path):
    matrix = np.full((20, 5), infer_dosage.MISSING_INT8, dtype=np.int8)
    matrix_path = tmp_path / "matrix.npy"
    model_path = tmp_path / "model.joblib"
    np.save(matrix_path, matrix)
    joblib.dump(_out_of_range_model(5), model_path)
    ancestry = np.zeros(matrix.shape[0], dtype=np.int8)

    with pytest.raises(RuntimeError, match="usable coverage"):
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
    model = _out_of_range_model(5)
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
