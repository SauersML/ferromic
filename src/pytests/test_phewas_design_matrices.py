import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phewas import models


@pytest.fixture
def worker_environment(tmp_path, monkeypatch):
    n = 5
    index = pd.Index([f"id{i}" for i in range(n)], name="person_id")
    cols = pd.Index(["const", "target", "sex", "AGE_c"])
    X_all = np.arange(n * len(cols), dtype=np.float64).reshape(n, len(cols))

    allowed_mask = np.array([False, True, True, False, False], dtype=bool)
    finite_mask = np.array([True, False, True, True, True], dtype=bool)
    case_idx = np.array([0, 3], dtype=np.int32)
    case_mask = np.zeros(n, dtype=bool)
    case_mask[case_idx] = True
    valid_mask = (allowed_mask | case_mask) & finite_mask

    monkeypatch.setattr(models, "N_core", n, raising=False)
    monkeypatch.setattr(models, "worker_core_df_index", index, raising=False)
    monkeypatch.setattr(models, "worker_core_df_cols", cols, raising=False)
    monkeypatch.setattr(models, "col_ix", {c: i for i, c in enumerate(cols)}, raising=False)
    monkeypatch.setattr(models, "X_all", X_all, raising=False)
    monkeypatch.setattr(models, "finite_mask_worker", finite_mask, raising=False)
    monkeypatch.setattr(models, "allowed_mask_by_cat", {"catA": allowed_mask}, raising=False)

    allowed_fp = models._index_fingerprint(index[np.flatnonzero(allowed_mask & finite_mask)])
    monkeypatch.setattr(models, "allowed_fp_by_cat", {"catA": allowed_fp}, raising=False)

    anc = pd.Series(["eur", "afr", "eur", "eur", "afr"], index=index).str.lower()
    monkeypatch.setattr(models, "worker_anc_series", anc, raising=False)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ctx: Dict[str, Any] = {
        "CACHE_DIR": str(cache_dir),
        "RESULTS_CACHE_DIR": str(tmp_path / "results"),
        "LRT_OVERALL_CACHE_DIR": str(tmp_path / "lrt_overall"),
        "BOOT_OVERALL_CACHE_DIR": str(tmp_path / "boot_overall"),
        "LRT_FOLLOWUP_CACHE_DIR": str(tmp_path / "lrt_followup"),
        "NUM_PCS": 0,
        "MIN_CASES_FILTER": 0,
        "MIN_CONTROLS_FILTER": 0,
        "MIN_NEFF_FILTER": 0.0,
        "RIDGE_L2_BASE": 1.0,
        "SEX_RESTRICT_MODE": "none",
        "SEX_RESTRICT_PROP": 1.0,
        "SEX_RESTRICT_MAX_OTHER_CASES": 0,
        "CTX_TAG": "test",
        "CACHE_VERSION_TAG": "test",
        "MODE": "test",
        "SELECTION": "test",
        "BOOTSTRAP_SEQ_ALPHA": 0.01,
        "BOOTSTRAP_B_MAX": 10,
        "PER_ANC_MIN_CASES": 0,
        "PER_ANC_MIN_CONTROLS": 0,
    }
    monkeypatch.setattr(models, "CTX", ctx, raising=False)

    for key in [
        "RESULTS_CACHE_DIR",
        "LRT_OVERALL_CACHE_DIR",
        "BOOT_OVERALL_CACHE_DIR",
        "LRT_FOLLOWUP_CACHE_DIR",
    ]:
        os.makedirs(ctx[key], exist_ok=True)

    task = {
        "name": "test_pheno",
        "category": "catA",
        "target": "target",
        "cdr_codename": "unit",
        "case_idx": case_idx,
        "case_fp": "fp",
    }

    pheno_df = pd.DataFrame({"is_case": case_mask.astype(np.int8)}, index=index)
    pheno_path = cache_dir / f"pheno_{task['name']}_{task['cdr_codename']}.parquet"
    pheno_df.to_parquet(pheno_path)

    return {
        "task": task,
        "valid_mask": valid_mask,
        "case_mask": case_mask,
    }


def _patch_and_invoke(worker_fn, env, monkeypatch):
    valid_mask = env["valid_mask"]
    case_mask = env["case_mask"]

    def check_design_matrix(X_base, y_series):
        base_cols = list(X_base.columns)
        base_ix = [models.col_ix[c] for c in base_cols]
        expected_df = pd.DataFrame(
            models.X_all[valid_mask][:, base_ix],
            index=models.worker_core_df_index[valid_mask],
            columns=base_cols,
        ).astype(np.float64, copy=False)
        pd.testing.assert_frame_equal(X_base, expected_df)
        expected_y = pd.Series(
            np.where(case_mask[valid_mask], 1, 0),
            index=expected_df.index,
            dtype=np.int8,
        )
        pd.testing.assert_series_equal(y_series, expected_y)
        return X_base, y_series, None, "unit_test_skip"

    monkeypatch.setattr(models, "_apply_sex_restriction", check_design_matrix)
    worker_fn(env["task"])


def test_lrt_overall_design_matrix_respects_mask(worker_environment, monkeypatch):
    _patch_and_invoke(models.lrt_overall_worker, worker_environment, monkeypatch)


def test_bootstrap_overall_design_matrix_respects_mask(worker_environment, monkeypatch):
    _patch_and_invoke(models.bootstrap_overall_worker, worker_environment, monkeypatch)


def test_followup_design_matrix_respects_mask(worker_environment, monkeypatch):
    _patch_and_invoke(models.lrt_followup_worker, worker_environment, monkeypatch)
