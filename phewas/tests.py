import os
import sys
import time
import json
import tempfile
import threading
import contextlib
from pathlib import Path
import shutil
import queue
import platform
import resource
from unittest.mock import patch, MagicMock
import warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from google.cloud import bigquery
    bigquery.Client = MagicMock()
except Exception:
    pass

try:
    from phewas import iox
    iox.load_related_to_remove = lambda *_, **__: set()
except Exception:
    pass

# Add the current directory to the path to allow absolute imports of phewas modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import phewas.run as run
import phewas.iox as io
import phewas.pheno as pheno
import phewas.models as models
from scipy.special import expit as sigmoid

pytestmark = pytest.mark.timeout(30)

# --- Test Constants ---
TEST_TARGET_INVERSION = 'chr_test-1-INV-1'
TEST_CDR_CODENAME = "dataset"

# --- Global Test Helpers & Fixtures ---

@contextlib.contextmanager
def temp_workspace():
    """Creates a temporary workspace, sets it as CWD, and cleans up."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            os.environ["WORKSPACE_CDR"] = f"test.project.{TEST_CDR_CODENAME}"
            os.environ["GOOGLE_PROJECT"] = "local-project"
            for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
                os.environ[v] = "1"
            yield Path(tmpdir)
        finally:
            os.chdir(original_dir)

@contextlib.contextmanager
def preserve_run_globals():
    keys = ["MIN_CASES_FILTER","MIN_CONTROLS_FILTER","FDR_ALPHA","LRT_SELECT_ALPHA",
            "TARGET_INVERSION","PHENOTYPE_DEFINITIONS_URL","INVERSION_DOSAGES_FILE"]
    snapshot = {k: getattr(run, k) for k in keys if hasattr(run, k)}
    try:
        yield
    finally:
        for k,v in snapshot.items(): setattr(run, k, v)

def write_parquet(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

def write_tsv(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False)

def make_synth_cohort(N=200, NUM_PCS=10, seed=42):
    rng = np.random.default_rng(seed)
    person_ids = [f"p{i:07d}" for i in range(1, N + 1)]

    demographics = pd.DataFrame({"AGE": rng.uniform(30, 75, N)}, index=pd.Index(person_ids, name="person_id"))
    demographics["AGE_sq"] = demographics["AGE"]**2
    demographics['AGE_c'] = demographics['AGE'] - demographics['AGE'].mean()
    demographics['AGE_c_sq'] = demographics['AGE_c'] ** 2
    sex = pd.DataFrame({"sex": rng.binomial(1, 0.55, N).astype(float)}, index=demographics.index)
    pcs = pd.DataFrame(rng.normal(0, 0.01, (N, NUM_PCS)), index=demographics.index, columns=[f"PC{i}" for i in range(1, NUM_PCS + 1)])
    inversion_main = pd.DataFrame({TEST_TARGET_INVERSION: np.clip(rng.normal(0, 0.5, N), -2, 2)}, index=demographics.index)
    inversion_const = pd.DataFrame({TEST_TARGET_INVERSION: np.zeros(N)}, index=demographics.index)
    ancestry = pd.DataFrame({"ANCESTRY": rng.choice(["eur", "afr"], N, p=[0.6, 0.4])}, index=demographics.index)

    p_a = sigmoid(1.0 * inversion_main[TEST_TARGET_INVERSION] + 0.02 * (demographics["AGE"] - 50) - 0.2 * sex["sex"])
    p_c = sigmoid(0.6 * inversion_main[TEST_TARGET_INVERSION] - 0.01 * (demographics["AGE"] - 50))
    cases_a = set(demographics.index[rng.random(N) < p_a])
    cases_b = set(rng.choice(person_ids, 6, replace=False))
    cases_c = set(demographics.index[rng.random(N) < p_c])

    phenos = {
        "A_strong_signal": {"disease": "A strong signal", "category": "cardio", "cases": cases_a},
        "B_insufficient": {"disease": "B insufficient", "category": "cardio", "cases": cases_b},
        "C_moderate_signal": {"disease": "C moderate signal", "category": "neuro", "cases": cases_c},
    }

    core_data = {
        "demographics": demographics, "sex": sex, "pcs": pcs,
        "inversion_main": inversion_main, "inversion_const": inversion_const,
        "ancestry": ancestry, "related_to_remove": set()
    }
    return core_data, phenos


def _init_lrt_worker_from_df(df, masks, anc_series, ctx):
    arr = df.to_numpy(dtype=np.float32, copy=True)
    meta, shm = io.create_shared_from_ndarray(arr, readonly=True)
    models.init_lrt_worker(meta, list(df.columns), df.index.astype(str), masks, anc_series, ctx)
    return shm

def prime_all_caches_for_run(core_data, phenos, cdr_codename, target_inversion, cache_dir="./phewas_cache"):
    os.makedirs(cache_dir, exist_ok=True)

    write_parquet(Path(cache_dir) / f"demographics_{cdr_codename}.parquet", core_data["demographics"])
    num_pcs = core_data["pcs"].shape[1]
    gcp_project = os.environ.get("GOOGLE_PROJECT", "")
    pcs_path = Path(cache_dir) / f"pcs_{num_pcs}_{run._source_key(gcp_project, run.PCS_URI, num_pcs)}.parquet"
    sex_path = Path(cache_dir) / f"genetic_sex_{run._source_key(gcp_project, run.SEX_URI)}.parquet"
    anc_path = Path(cache_dir) / f"ancestry_labels_{run._source_key(gcp_project, run.PCS_URI)}.parquet"
    dosages_resolved = os.path.abspath(run.INVERSION_DOSAGES_FILE)
    inv_safe = models.safe_basename(target_inversion)
    inv_path = Path(cache_dir) / f"inversion_{inv_safe}_{run._source_key(dosages_resolved, target_inversion)}.parquet"

    write_parquet(inv_path, core_data["inversion_main"])
    write_parquet(pcs_path, core_data["pcs"])
    write_parquet(sex_path, core_data["sex"])
    write_parquet(anc_path, core_data["ancestry"])

    pheno_defs_list = []
    for s_name, p_data in phenos.items():
        p_path = Path(cache_dir) / f"pheno_{s_name}_{cdr_codename}.parquet"
        case_df = pd.DataFrame({"is_case": 1}, index=pd.Index(list(p_data["cases"]), name="person_id"), dtype=np.int8)
        write_parquet(p_path, case_df)
        pheno_defs_list.append({
            "disease": p_data["disease"], "disease_category": p_data["category"],
            "sanitized_name": s_name, "icd9_codes": "1.1", "icd10_codes": "A1.1"
        })

    pan_cases = {"cardio": phenos["A_strong_signal"]["cases"] | phenos["B_insufficient"]["cases"], "neuro": phenos["C_moderate_signal"]["cases"]}
    pd.to_pickle(pan_cases, Path(cache_dir) / f"pan_category_cases_{cdr_codename}.pkl")

    for d in ["results_atomic", "lrt_overall", "lrt_followup"]:
        os.makedirs(Path(cache_dir) / d, exist_ok=True)

    return pd.DataFrame(pheno_defs_list)

def make_local_pheno_defs_tsv(pheno_defs_df, tmpdir) -> Path:
    path = Path(tmpdir) / "local_defs.tsv"
    write_tsv(path, pheno_defs_df[["disease", "disease_category", "icd9_codes", "icd10_codes"]])
    return path

def read_rss_bytes():
    if PSUTIL_AVAILABLE:
        return psutil.Process().memory_info().rss
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    try:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r * 1024 if platform.system() == "Linux" else r)
    except Exception:
        pass
    raise RuntimeError("Cannot measure RSS on this platform without psutil")

@pytest.fixture
def test_ctx():
    return {
        "NUM_PCS": 10, "MIN_CASES_FILTER": 10, "MIN_CONTROLS_FILTER": 10,
        "FDR_ALPHA": 0.2, "PER_ANC_MIN_CASES": 5, "PER_ANC_MIN_CONTROLS": 5,
        "LRT_SELECT_ALPHA": 0.2, "CACHE_DIR": "./phewas_cache",
        "RESULTS_CACHE_DIR": "./phewas_cache/results_atomic",
        "LRT_OVERALL_CACHE_DIR": "./phewas_cache/lrt_overall",
        "LRT_FOLLOWUP_CACHE_DIR": "./phewas_cache/lrt_followup",
        "BOOT_OVERALL_CACHE_DIR": "./phewas_cache/boot_overall",
        "RIDGE_L2_BASE": 1.0,
        # Disable new filters for tests by default.
        # We will override these in specific tests that check the filters.
        "MIN_NEFF_FILTER": 0,
        "MLE_REFIT_MIN_NEFF": 0,
        "CACHE_VERSION_TAG": io.CACHE_VERSION_TAG,
        "CTX_TAG": "test_ctx",
    }

# --- Unit Tests ---
def test_io_demographics_cache_validation():
    with temp_workspace():
        good_df = pd.DataFrame({"AGE": [40, 50], "AGE_sq": [1600, 2500]}, index=pd.Index(["p1", "p2"], name="person_id"))
        cache_path = Path("./phewas_cache") / f"demographics_{TEST_CDR_CODENAME}.parquet"
        write_parquet(cache_path, good_df)
        def fail_gen(): raise AssertionError("Generator should not be called")
        res = io.get_cached_or_generate(str(cache_path), fail_gen)
        pd.testing.assert_frame_equal(res, good_df)

        bad_df = good_df.copy(); bad_df["AGE_sq"] = [0, 0]
        write_parquet(cache_path, bad_df)
        def regen_func(): return good_df
        res = io.get_cached_or_generate(str(cache_path), regen_func)
        pd.testing.assert_frame_equal(res, good_df)

def test_index_fingerprint_is_order_insensitive():
    fp1 = models._index_fingerprint(pd.Index(["p1", "p3", "p2"]))
    fp2 = models._index_fingerprint(pd.Index(["p2", "p1", "p3"]))
    assert fp1 == fp2 and fp1.endswith(":3")

def test_atomic_write_json_is_atomic():
    with temp_workspace():
        path, exceptions = "test.json", []
        def writer(payload):
            try: io.atomic_write_json(path, payload)
            except Exception as e: exceptions.append(e)
        threads = [threading.Thread(target=writer, args=({"val": i},)) for i in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not exceptions
        with open(path, 'r') as f: assert "val" in json.load(f)

def test_should_skip_meta_equivalence(test_ctx):
    with temp_workspace():
        core_df = pd.DataFrame(np.ones((10, 2)), columns=['const', TEST_TARGET_INVERSION])
        allowed_fp = "dummy_allowed_fp"
        # Define the metadata for the test
        meta = {
            "model_columns": list(core_df.columns),
            "num_pcs": 10,
            "min_cases": test_ctx["MIN_CASES_FILTER"],
            "min_ctrls": test_ctx["MIN_CONTROLS_FILTER"],
            "min_neff": test_ctx["MIN_NEFF_FILTER"],
            "target": TEST_TARGET_INVERSION,
            "category": "cat",
            "core_index_fp": models._index_fingerprint(core_df.index),
            "case_idx_fp": "dummy_fp",
            "allowed_mask_fp": allowed_fp,
            "ridge_l2_base": test_ctx["RIDGE_L2_BASE"],
            "ctx_tag": test_ctx["CTX_TAG"],
            "cache_version_tag": test_ctx["CACHE_VERSION_TAG"],
        }
        # Write the metadata to a JSON file
        io.write_meta_json("test.meta.json", meta)
        models.CTX = test_ctx
        # Check that the skip function returns True when the context is the same
        core_index_fp = models._index_fingerprint(core_df.index)
        thresholds = {
            "min_cases": test_ctx["MIN_CASES_FILTER"],
            "min_ctrls": test_ctx["MIN_CONTROLS_FILTER"],
            "min_neff": test_ctx["MIN_NEFF_FILTER"],
        }
        assert models._should_skip(
            "test.meta.json",
            core_df.columns,
            core_index_fp,
            "dummy_fp",
            "cat",
            TEST_TARGET_INVERSION,
            allowed_fp,
            thresholds=thresholds,
        )
        # Change the context
        test_ctx_changed = test_ctx.copy()
        test_ctx_changed["MIN_CASES_FILTER"] = 11
        models.CTX = test_ctx_changed
        # Check that the skip function returns False when the context is different
        thresholds_changed = {
            "min_cases": test_ctx_changed["MIN_CASES_FILTER"],
            "min_ctrls": test_ctx_changed["MIN_CONTROLS_FILTER"],
            "min_neff": test_ctx_changed["MIN_NEFF_FILTER"],
        }
        assert not models._should_skip(
            "test.meta.json",
            core_df.columns,
            core_index_fp,
            "dummy_fp",
            "cat",
            TEST_TARGET_INVERSION,
            allowed_fp,
            thresholds=thresholds_changed,
        )

def test_pheno_cache_loader_returns_correct_indices():
    with temp_workspace():
        core_index = pd.Index([f"p{i}" for i in range(10)])
        case_ids = ["p2", "p5", "p8"]
        pheno_info = {"sanitized_name": "test_pheno", "disease_category": "test_cat"}
        cache_path = Path(f"./phewas_cache/pheno_{pheno_info['sanitized_name']}_{TEST_CDR_CODENAME}.parquet")
        write_parquet(cache_path, pd.DataFrame(index=pd.Index(case_ids, name="person_id"), data={"is_case": 1}))
        res = pheno._load_single_pheno_cache(pheno_info, core_index, TEST_CDR_CODENAME, "./phewas_cache")
        np.testing.assert_array_equal(res["case_idx"], np.array([2, 5, 8], dtype=np.int32))

def test_worker_constant_dosage_emits_nan(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_const'],
        ], axis=1)
        X = sm.add_constant(core_df)
        anc = core_data['ancestry']['ANCESTRY']
        anc_series = anc.str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        X = X.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        shm = _init_lrt_worker_from_df(X, {"cardio": np.ones(len(X), dtype=bool)}, anc, test_ctx)
        task = {
            "name": "A_strong_signal",
            "category": "cardio",
            "cdr_codename": TEST_CDR_CODENAME,
            "target": TEST_TARGET_INVERSION,
        }
        def fake_ladder(X, y, **kwargs):
            class _Dummy:
                pass

            res = _Dummy()
            if hasattr(X, "columns"):
                res.params = pd.Series(np.zeros(len(X.columns)), index=X.columns)
            else:
                res.params = np.zeros(X.shape[1])
            setattr(res, "_used_ridge", True)
            setattr(res, "_used_firth", False)
            setattr(res, "_final_is_mle", False)
            setattr(res, "llf", np.nan)
            return res, "ridge_only"

        orig_fit = models._fit_logit_ladder
        orig_firth = models._firth_refit
        orig_score = models._score_test_from_reduced
        orig_score_boot = models._score_bootstrap_from_reduced
        try:
            models._fit_logit_ladder = fake_ladder
            models._firth_refit = lambda *args, **kwargs: None
            models._score_test_from_reduced = lambda *args, **kwargs: (np.nan, np.nan)
            models._score_bootstrap_from_reduced = lambda *args, **kwargs: np.nan
            models.lrt_overall_worker(task)
        finally:
            models._fit_logit_ladder = orig_fit
            models._firth_refit = orig_firth
            models._score_test_from_reduced = orig_score
            models._score_bootstrap_from_reduced = orig_score_boot
        result_path = Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json"
        assert result_path.exists()
        with open(result_path) as f: res = json.load(f)
        assert all(pd.isna(res.get(k)) for k in ["Beta", "OR", "P_Value"])
        shm.close(); shm.unlink()

def test_worker_insufficient_counts_skips(test_ctx):
    # This test specifically checks the insufficient counts filter, so we
    # override the default-disabled test context.
    test_ctx = test_ctx.copy()
    test_ctx["MIN_NEFF_FILTER"] = 100

    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        X = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main'],
        ], axis=1))
        anc = core_data['ancestry']['ANCESTRY']
        anc_series = anc.str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        X = X.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        shm = _init_lrt_worker_from_df(
            X,
            {"cardio": np.ones(len(X), dtype=bool)},
            anc,
            test_ctx,
        )
        models.lrt_overall_worker({
            "name": "B_insufficient",
            "category": "cardio",
            "cdr_codename": TEST_CDR_CODENAME,
            "target": TEST_TARGET_INVERSION,
        })
        result_path = Path(test_ctx["RESULTS_CACHE_DIR"]) / "B_insufficient.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)
        assert res["Skip_Reason"].startswith("insufficient_counts")
        shm.close(); shm.unlink()

def test_lrt_rank_and_df_positive(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        anc_series = core_data['ancestry']['ANCESTRY'].str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        core_df_with_const = core_df_with_const.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        shm = _init_lrt_worker_from_df(
            core_df_with_const,
            {"cardio": np.ones(len(core_df), dtype=bool)},
            core_data['ancestry']['ANCESTRY'],
            test_ctx,
        )
        task = {"name": "A_strong_signal", "category": "cardio", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        models.lrt_overall_worker(task)
        result_path = Path(test_ctx["LRT_OVERALL_CACHE_DIR"]) / "A_strong_signal.json"
        assert result_path.exists()
        shm.close(); shm.unlink()

def test_followup_includes_ancestry_levels_and_splits(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        anc_series = core_data['ancestry']['ANCESTRY'].str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        core_df_with_const = core_df_with_const.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        shm = _init_lrt_worker_from_df(core_df_with_const, {"neuro": np.ones(len(core_df), dtype=bool)}, core_data['ancestry']['ANCESTRY'], test_ctx)
        task = {"name": "C_moderate_signal", "category": "neuro", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        models.lrt_followup_worker(task)
        result_path = Path(test_ctx["LRT_FOLLOWUP_CACHE_DIR"]) / "C_moderate_signal.json"
        assert result_path.exists()
        shm.close(); shm.unlink()

def test_safe_basename():
    assert models.safe_basename("endo/../../weird:thing") == "endo_.._.._weird_thing"
    assert models.safe_basename("normal_name-1.0") == "normal_name-1.0"

def test_cache_idempotency_on_mask_change(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        X = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main'],
        ], axis=1))
        anc = core_data['ancestry']['ANCESTRY']
        anc_series = anc.str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        X = X.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        shm = _init_lrt_worker_from_df(
            X,
            {"cardio": np.ones(len(X), dtype=bool)},
            anc,
            test_ctx,
        )
        task = {
            "name": "A_strong_signal",
            "category": "cardio",
            "cdr_codename": TEST_CDR_CODENAME,
            "target": TEST_TARGET_INVERSION,
        }
        models.lrt_overall_worker(task)
        mtime1 = (Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json").stat().st_mtime
        time.sleep(0.1)
        models.lrt_overall_worker(task)
        mtime2 = (Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json").stat().st_mtime
        assert mtime1 == mtime2
        new_mask = np.ones(len(X), dtype=bool); new_mask[:10] = False
        shm.close(); shm.unlink()
        shm = _init_lrt_worker_from_df(
            X,
            {"cardio": new_mask},
            anc,
            test_ctx,
        )
        models.lrt_overall_worker(task)
        mtime3 = (Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json").stat().st_mtime
        assert mtime2 < mtime3
        shm.close(); shm.unlink()

def test_ridge_intercept_is_zero(test_ctx):
    with temp_workspace():
        X = pd.DataFrame({'const': 1.0, 'x1': [0, 0, 1, 1]}, index=pd.RangeIndex(4))
        y = pd.Series([0, 0, 1, 1])
        with patch('statsmodels.api.Logit') as mock_logit:
            mock_logit.return_value.fit.side_effect = PerfectSeparationWarning()
            models.CTX = test_ctx
            models._fit_logit_ladder(X, y, ridge_ok=True)
            assert mock_logit.return_value.fit_regularized.called
            args, kwargs = mock_logit.return_value.fit_regularized.call_args
            assert 'alpha' in kwargs
            assert isinstance(kwargs['alpha'], float)
            assert kwargs['alpha'] > 0.0

def test_lrt_collinear_df_is_zero(test_ctx):
    with temp_workspace():
        core_data, _ = make_synth_cohort()
        X_base = pd.concat([core_data['demographics'][['AGE_c']], core_data['sex']], axis=1)
        X_red = sm.add_constant(X_base)
        X_full = X_red.copy(); X_full['collinear'] = X_full['AGE_c'] * 2
        assert (X_full.shape[1] - X_red.shape[1]) == 1
        rank_full = np.linalg.matrix_rank(X_full)
        rank_red = np.linalg.matrix_rank(X_red)
        assert (rank_full - rank_red) == 0

def test_sex_restriction_policy(test_ctx):
    X = pd.DataFrame({'sex': [0,0,0,1,1,1]}); y = pd.Series([1,1,0,0,0,0])
    X_res, y_res, note, skip = models._apply_sex_restriction(X, y)
    assert skip is None and 'sex_restricted' in note and len(X_res) == 3 and 'sex' not in X_res.columns
    X = pd.DataFrame({'sex': [0,0,1,1,1,1]}); y = pd.Series([1,1,0,0,0,0])
    _, _, _, skip = models._apply_sex_restriction(X.loc[y.index != 2], y.loc[y.index != 2])
    assert skip is not None

def test_penalized_fit_ci_and_pval_suppression(test_ctx):
    """Verifies that CIs and P-values are suppressed for penalized (ridge) fits."""
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        cases = list(phenos["A_strong_signal"]["cases"])
        core_data['pcs'].loc[cases, 'PC1'] = 1000
        core_data['pcs'].loc[~core_data['pcs'].index.isin(cases), 'PC1'] = -1000
        core_df = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c']],
            core_data['pcs'],
            core_data['inversion_main'],
        ], axis=1))
        anc = core_data['ancestry']['ANCESTRY']
        anc_series = anc.str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        core_df = core_df.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        models.CTX = test_ctx
        shm = _init_lrt_worker_from_df(
            core_df,
            {"cardio": np.ones(len(core_df), dtype=bool)},
            anc,
            test_ctx,
        )
        task = {
            "name": "A_strong_signal",
            "category": "cardio",
            "cdr_codename": TEST_CDR_CODENAME,
            "target": TEST_TARGET_INVERSION,
        }
        def fake_ladder(X, y, **kwargs):
            class _Dummy:
                pass

            res = _Dummy()
            n_params = X.shape[1] if hasattr(X, "shape") else len(X[0])
            res.params = np.zeros(n_params, dtype=float)
            res._used_ridge = True
            res._used_firth = False
            res._final_is_mle = False
            res._path_reasons = ["ridge_only"]
            res.llf = np.nan
            return res, "ridge_only"

        orig_fit = models._fit_logit_ladder
        orig_firth = models._firth_refit
        orig_score = models._score_test_from_reduced
        orig_score_boot = models._score_bootstrap_from_reduced
        try:
            models._fit_logit_ladder = fake_ladder
            models._firth_refit = lambda *args, **kwargs: None
            models._score_test_from_reduced = lambda *args, **kwargs: (np.nan, np.nan)
            models._score_bootstrap_from_reduced = lambda *args, **kwargs: np.nan
            models.lrt_overall_worker(task)
        finally:
            models._fit_logit_ladder = orig_fit
            models._firth_refit = orig_firth
            models._score_test_from_reduced = orig_score
            models._score_bootstrap_from_reduced = orig_score_boot
        res = json.load(open(Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json"))
        assert res['Used_Ridge'] is True
        assert res['Inference_Type'] == 'none'
        assert res['OR_CI95'] is None
        assert pd.isna(res['P_Value'])
        shm.close(); shm.unlink()

def test_firth_fit_keeps_inference(test_ctx):
    """Firth fits triggered by ridge should retain valid inference."""
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        cases = list(phenos["A_strong_signal"]["cases"])
        # Force a separation scenario that promotes the ridge ladder to use Firth.
        core_data['pcs'].loc[cases, 'PC1'] = 1000
        core_data['pcs'].loc[~core_data['pcs'].index.isin(cases), 'PC1'] = -1000
        core_df = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c']],
            core_data['pcs'],
            core_data['inversion_main'],
        ], axis=1))
        anc = core_data['ancestry']['ANCESTRY']
        anc_series = anc.str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        core_df = core_df.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        models.CTX = test_ctx
        shm = _init_lrt_worker_from_df(
            core_df,
            {"cardio": np.ones(len(core_df), dtype=bool)},
            anc,
            test_ctx,
        )
        task = {
            "name": "A_strong_signal",
            "category": "cardio",
            "cdr_codename": TEST_CDR_CODENAME,
            "target": TEST_TARGET_INVERSION,
        }

        models.lrt_overall_worker(task)

        result_path = Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)

        assert res['Inference_Type'] == 'firth'
        assert res['Used_Ridge'] is True  # Ridge was used in the ladder, but inference should remain valid
        assert np.isfinite(res['P_Value']) and res['P_Value'] > 0
        assert res['P_Valid'] is True
        assert res['OR_CI95'] is not None
        assert res['CI_Valid'] is True
        shm.close(); shm.unlink()

def test_perfect_separation_promoted_to_ridge(test_ctx):
    X = pd.DataFrame({'const': 1, 'x': [0, 0, 1, 1]}); y = pd.Series([0, 0, 1, 1])
    models.CTX = test_ctx
    with patch('statsmodels.api.Logit') as mock_logit:
        mock_logit.return_value.fit.side_effect = [PerfectSeparationWarning(), PerfectSeparationWarning()]
        mock_logit.return_value.fit_regularized.return_value = "ridge_fit"
        fit, reason = models._fit_logit_ladder(X, y)
        assert mock_logit.return_value.fit_regularized.called

def test_worker_reports_n_used_after_sex_restriction(test_ctx):
    """Verifies that N_*_Used fields are correctly reported after sex restriction."""
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        male_ids = core_data['sex'][core_data['sex']['sex'] == 1.0].index
        cases = set(np.random.default_rng(1).choice(male_ids, 20, replace=False))
        phenos['sex_restricted_pheno'] = {'disease': 'sex_restricted', 'category': 'endo', 'cases': cases}

        # write caches so Stage-1 worker can load phenotype cases
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)

        core_df = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main']
        ], axis=1))
        anc = core_data['ancestry']['ANCESTRY']
        anc_series = anc.str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        core_df = core_df.join(A, how="left").fillna({c: 0.0 for c in A.columns})

        allowed_mask_arr = ~core_df.index.isin(list(cases))
        allowed_mask = {"endo": allowed_mask_arr}
        shm = _init_lrt_worker_from_df(core_df, allowed_mask, anc, test_ctx)

        models.lrt_overall_worker({
            "name": "sex_restricted_pheno",
            "category": "endo",
            "cdr_codename": TEST_CDR_CODENAME,
            "target": TEST_TARGET_INVERSION,
        })

        result_path = Path(test_ctx["RESULTS_CACHE_DIR"]) / "sex_restricted_pheno.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)

        assert 'sex_majority_restricted_to_1' in res['Model_Notes']
        assert res['N_Cases'] == len(cases)
        assert res['N_Total_Used'] == len(male_ids)
        assert res['N_Cases_Used'] == len(cases)
        assert res['N_Controls_Used'] == len(male_ids) - len(cases)
        shm.close(); shm.unlink()

def test_lrt_overall_firth_fit_keeps_inference(test_ctx):
    """Stage-1 Firth refits triggered by ridge should retain valid inference."""
    with temp_workspace():
        core_data, phenos = make_synth_cohort(N=100)
        cases = list(phenos["A_strong_signal"]["cases"])
        core_data['pcs'].loc[cases, 'PC1'] = 1000
        core_data['pcs'].loc[~core_data['pcs'].index.isin(cases), 'PC1'] = -1000

        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)

        core_df = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1)
        core_df = sm.add_constant(core_df)
        anc_cols = pd.get_dummies(core_data['ancestry']['ANCESTRY'], prefix='ANC', drop_first=True, dtype=np.float64)
        core_df = core_df.join(anc_cols)

        shm = _init_lrt_worker_from_df(core_df, {}, core_data['ancestry']['ANCESTRY'], test_ctx)

        task = {"name": "A_strong_signal", "category": "cardio", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerfectSeparationWarning)
            models.lrt_overall_worker(task)

        result_path = Path(test_ctx["LRT_OVERALL_CACHE_DIR"]) / "A_strong_signal.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)

        assert res['Inference_Type'] == 'firth'
        assert np.isfinite(res['P_LRT_Overall'])
        assert res['P_Overall_Valid'] is True
        assert res.get('LRT_Overall_Reason') in (None, '',)

        atomic_path = Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json"
        assert atomic_path.exists()
        with open(atomic_path) as f:
            atomic_res = json.load(f)

        assert atomic_res['Inference_Type'] == 'firth'
        assert atomic_res['Used_Ridge'] is True
        assert np.isfinite(atomic_res['P_Value'])
        shm.close(); shm.unlink()

def test_lrt_followup_firth_fit_keeps_ci(test_ctx):
    """Stage-2 per-ancestry Firth refits should emit valid inference outputs."""
    with temp_workspace():
        rng = np.random.default_rng(42)
        N=300
        core_data, phenos = make_synth_cohort(N=N)
        core_data['ancestry']['ANCESTRY'] = rng.choice(['eur', 'afr', 'amr'], N)

        afr_ids = core_data['ancestry'][core_data['ancestry']['ANCESTRY'] == 'afr'].index
        cases = list(phenos["C_moderate_signal"]["cases"])
        afr_cases = [pid for pid in cases if pid in afr_ids]
        afr_non_cases = [pid for pid in afr_ids if pid not in cases]

        core_data['pcs'].loc[afr_cases, 'PC1'] = 1000
        core_data['pcs'].loc[afr_non_cases, 'PC1'] = -1000

        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1)
        core_df = sm.add_constant(core_df)

        shm = _init_lrt_worker_from_df(core_df, {}, core_data['ancestry']['ANCESTRY'], test_ctx)

        task = {"name": "C_moderate_signal", "category": "neuro", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerfectSeparationWarning)
            models.lrt_followup_worker(task)

        result_path = Path(test_ctx["LRT_FOLLOWUP_CACHE_DIR"]) / "C_moderate_signal.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)

        assert res['AFR_Inference_Type'] == 'firth'
        assert np.isfinite(res['AFR_P'])
        assert res['AFR_P_Valid'] is True
        assert res['AFR_CI95'] is not None
        assert 'EUR_CI95' in res
        assert 'AMR_CI95' in res
        assert res.get('AFR_REASON') in (None, '')
        shm.close(); shm.unlink()

# --- Integration Tests ---
def test_fetcher_producer_drains_cache_only():
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        pheno_defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_index = pd.Index([f"p{i:07d}" for i in range(1, 201)], name="person_id")
        q = queue.Queue(maxsize=100)
        fetcher_thread = threading.Thread(
            target=pheno.phenotype_fetcher_worker,
            args=(q, pheno_defs_df, None, None, TEST_CDR_CODENAME, core_index, "./phewas_cache", 128, 4)
        )
        fetcher_thread.start()
        results = []
        for _ in range(len(phenos) + 1):
            item = q.get()
            if item is None: break
            results.append(item)
        fetcher_thread.join()
        assert len(results) == len(phenos)
        assert {r['name'] for r in results} == set(phenos.keys())

def test_lrt_worker_creates_atomic_results(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort(seed=42)
        pheno_defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        X = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main'],
        ], axis=1))
        anc = core_data['ancestry']['ANCESTRY']
        anc_series = anc.str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        X = X.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        allowed_mask_by_cat = {
            category: np.ones(len(X), dtype=bool)
            for category in {p['category'] for p in phenos.values()}
        }
        _ = _init_lrt_worker_from_df(
            X,
            allowed_mask_by_cat,
            anc,
            test_ctx,
        )
        for s_name, p_data in phenos.items():
            models.lrt_overall_worker({
                "name": s_name,
                "category": p_data['category'],
                "cdr_codename": TEST_CDR_CODENAME,
                "target": TEST_TARGET_INVERSION,
            })
        result_files = os.listdir(test_ctx["RESULTS_CACHE_DIR"])
        assert len(result_files) >= 2 # B_insufficient is skipped
        with open(Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json") as f: res = json.load(f)
        assert res["OR"] > 1.0 and res["P_Value"] < 0.1

def test_cache_equivalence_skips_work(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        X = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main'],
        ], axis=1))
        anc = core_data['ancestry']['ANCESTRY']
        anc_series = anc.str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        X = X.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        allowed_mask_by_cat = {
            "cardio": np.ones(len(X), dtype=bool),
            "neuro": np.ones(len(X), dtype=bool),
        }
        _ = _init_lrt_worker_from_df(
            X,
            allowed_mask_by_cat,
            anc,
            test_ctx,
        )
        for s_name, p_data in phenos.items():
            models.lrt_overall_worker({
                "name": s_name,
                "category": p_data['category'],
                "cdr_codename": TEST_CDR_CODENAME,
                "target": TEST_TARGET_INVERSION,
            })
        mtimes = {f: f.stat().st_mtime for f in Path(test_ctx["RESULTS_CACHE_DIR"]).glob("*.json")}
        time.sleep(1)
        for s_name, p_data in phenos.items():
            models.lrt_overall_worker({
                "name": s_name,
                "category": p_data['category'],
                "cdr_codename": TEST_CDR_CODENAME,
                "target": TEST_TARGET_INVERSION,
            })
        mtimes_after = {f: f.stat().st_mtime for f in Path(test_ctx["RESULTS_CACHE_DIR"]).glob("*.json")}
        assert mtimes == mtimes_after

def test_lrt_overall_meta_idempotency(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        X_base = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1)
        X = sm.add_constant(X_base)
        anc_series = core_data['ancestry']['ANCESTRY'].str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        X = X.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        shm = _init_lrt_worker_from_df(
            X,
            {"cardio": np.ones(len(X), dtype=bool), "neuro": np.ones(len(X), dtype=bool)},
            core_data['ancestry']['ANCESTRY'],
            test_ctx,
        )
        task = {"name": "A_strong_signal", "category": "cardio", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        models.lrt_overall_worker(task)
        f = Path(test_ctx["LRT_OVERALL_CACHE_DIR"]) / "A_strong_signal.json"
        m0 = f.stat().st_mtime
        time.sleep(1)
        models.lrt_overall_worker(task)
        assert f.stat().st_mtime == m0
        shm.close(); shm.unlink()

def test_final_results_has_ci_and_ancestry_fields():
    with temp_workspace() as tmpdir, preserve_run_globals():
        core_data, phenos = make_synth_cohort()
        run.INVERSION_DOSAGES_FILE = "dummy.tsv"
        defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        local_defs = make_local_pheno_defs_tsv(defs_df, tmpdir)
        orig_bootstrap = models._score_bootstrap_from_reduced

        def safe_score_bootstrap(*args, **kwargs):
            res = orig_bootstrap(*args, **kwargs)
            if isinstance(res, tuple) and len(res) == 2:
                return (*res, 0, 0)
            return res

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('phewas.run.bigquery.Client'))
            stack.enter_context(patch('phewas.run.io.load_related_to_remove', return_value=set()))
            stack.enter_context(patch('phewas.run.supervisor_main', lambda *a, **k: run._pipeline_once()))
            stack.enter_context(patch('phewas.models._score_bootstrap_from_reduced', safe_score_bootstrap))
            stack.enter_context(patch('phewas.run.io.load_pcs', lambda gcp_project, PCS_URI, NUM_PCS, _core=core_data: _core['pcs'].iloc[:, :NUM_PCS]))
            stack.enter_context(patch('phewas.run.io.load_genetic_sex', lambda gcp_project, SEX_URI, _core=core_data: _core['sex']))
            stack.enter_context(patch('phewas.run.io.load_ancestry_labels', lambda gcp_project, LABELS_URI, _core=core_data: _core['ancestry']))

            run.TARGET_INVERSIONS = [TEST_TARGET_INVERSION]
            run.MASTER_RESULTS_CSV = "master_results.csv"
            run.MIN_CASES_FILTER = run.MIN_CONTROLS_FILTER = 10
            run.NUM_PCS = core_data['pcs'].shape[1]
            run.FDR_ALPHA = run.LRT_SELECT_ALPHA = 0.4
            run.PHENOTYPE_DEFINITIONS_URL = str(local_defs)
            write_tsv(run.INVERSION_DOSAGES_FILE, core_data["inversion_main"].reset_index().rename(columns={'person_id':'SampleID'}))
            run.main()

        output_path = Path(run.MASTER_RESULTS_CSV)
        assert output_path.exists()
        df = pd.read_csv(output_path, sep='\t')
        assert "OR_CI95" in df.columns and "FINAL_INTERPRETATION" in df.columns and "Q_GLOBAL" in df.columns

def test_memory_envelope_relative():
    with temp_workspace():
        base_rss = read_rss_bytes()
        n_phenos, n_participants = (100, 10000)
        envelope_gb = 1.0
        core_data, phenos_base = make_synth_cohort(N=n_participants)
        phenos = {f"pheno_{i}": phenos_base["A_strong_signal"] for i in range(n_phenos)}
        phenos.update(phenos_base)
        pheno_defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        local_defs_path = make_local_pheno_defs_tsv(pheno_defs_df, Path("."))
        with preserve_run_globals():
            run.MIN_CASES_FILTER, run.MIN_CONTROLS_FILTER = 10, 10
            run.PHENOTYPE_DEFINITIONS_URL = str(local_defs_path)
            run.TARGET_INVERSIONS = [TEST_TARGET_INVERSION]
            run.INVERSION_DOSAGES_FILE = "dummy.tsv"
            write_tsv(run.INVERSION_DOSAGES_FILE, core_data["inversion_main"].reset_index().rename(columns={'person_id':'SampleID'}))
            peak_rss = [base_rss]
            stop_event = threading.Event()
            def poll_mem():
                while not stop_event.is_set():
                    peak_rss[0] = max(peak_rss[0], read_rss_bytes())
                    time.sleep(0.1)
            poll_thread = threading.Thread(target=poll_mem)
            poll_thread.start()
            try: run.main()
            finally: stop_event.set(); poll_thread.join()
            peak_delta_gb = (peak_rss[0] - base_rss) / (1024**3)
            assert peak_delta_gb < envelope_gb, f"Peak memory delta {peak_delta_gb:.3f} GB exceeded envelope"

def test_multi_inversion_pipeline_produces_master_file():
    """
    Integration test for the primary new feature: running two inversions, applying
    a global FDR, and producing a single master result file.
    """
    with temp_workspace() as tmpdir, preserve_run_globals():
        # 1. Define two inversions and their synthetic data
        INV_A, INV_B = 'chr_test-A-INV-1', 'chr_test-B-INV-2'
        core_data, phenos = make_synth_cohort()
        rng = np.random.default_rng(101)
        core_data['inversion_A'] = pd.DataFrame({INV_A: np.clip(rng.normal(0.8, 0.5, 200), -2, 2)}, index=core_data['demographics'].index)
        core_data['inversion_B'] = pd.DataFrame({INV_B: np.zeros(200)}, index=core_data['demographics'].index)

        # Re-generate the 'strong signal' phenotype to be associated with INV_A
        p_a = sigmoid(2.5 * core_data['inversion_A'][INV_A] + 0.02 * (core_data["demographics"]["AGE"] - 50) - 0.2 * core_data["sex"]["sex"])
        cases_a = set(core_data["demographics"].index[rng.random(200) < p_a])
        phenos['A_strong_signal']['cases'] = cases_a

        # 2. Prime caches for both inversions
        run.INVERSION_DOSAGES_FILE = "dummy_dosages.tsv"
        defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, INV_A)
        dosages_resolved = os.path.abspath(run.INVERSION_DOSAGES_FILE)
        inv_a_path = Path("./phewas_cache") / f"inversion_{models.safe_basename(INV_A)}_{run._source_key(dosages_resolved, INV_A)}.parquet"
        inv_b_path = Path("./phewas_cache") / f"inversion_{models.safe_basename(INV_B)}_{run._source_key(dosages_resolved, INV_B)}.parquet"
        write_parquet(inv_a_path, core_data["inversion_A"])
        write_parquet(inv_b_path, core_data["inversion_B"])

        local_defs = make_local_pheno_defs_tsv(defs_df, tmpdir)
        orig_bootstrap = models._score_bootstrap_from_reduced

        def safe_score_bootstrap(*args, **kwargs):
            res = orig_bootstrap(*args, **kwargs)
            if isinstance(res, tuple) and len(res) == 2:
                return (*res, 0, 0)
            return res

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('phewas.run.bigquery.Client'))
            stack.enter_context(patch('phewas.run.io.load_related_to_remove', return_value=set()))
            stack.enter_context(patch('phewas.run.supervisor_main', lambda *a, **k: run._pipeline_once()))
            stack.enter_context(patch('phewas.models._score_bootstrap_from_reduced', safe_score_bootstrap))
            stack.enter_context(patch('phewas.run.io.load_pcs', lambda gcp_project, PCS_URI, NUM_PCS, _core=core_data: _core['pcs'].iloc[:, :NUM_PCS]))
            stack.enter_context(patch('phewas.run.io.load_genetic_sex', lambda gcp_project, SEX_URI, _core=core_data: _core['sex']))
            stack.enter_context(patch('phewas.run.io.load_ancestry_labels', lambda gcp_project, LABELS_URI, _core=core_data: _core['ancestry']))

            # 3. Configure and run the main pipeline
            run.TARGET_INVERSIONS = [INV_A, INV_B]
            run.MASTER_RESULTS_CSV = "multi_inversion_master.csv"
            run.MIN_CASES_FILTER = run.MIN_CONTROLS_FILTER = 10
            run.NUM_PCS = core_data['pcs'].shape[1]
            run.FDR_ALPHA = 0.9  # High alpha to ensure we get some hits
            run.PHENOTYPE_DEFINITIONS_URL = str(local_defs)
            dummy_dosage_df = pd.DataFrame({
                'SampleID': core_data['demographics'].index,
                INV_A: core_data['inversion_A'][INV_A],
                INV_B: core_data['inversion_B'][INV_B],
            })
            write_tsv(run.INVERSION_DOSAGES_FILE, dummy_dosage_df)

            run.main()

        # 4. Assert correctness of outputs
        output_path = Path(run.MASTER_RESULTS_CSV)
        assert output_path.exists(), "Master CSV file was not created"

        df = pd.read_csv(output_path, sep='\t')

        # Assert per-inversion directories were created
        assert (Path("./phewas_cache") / models.safe_basename(INV_A)).is_dir()
        assert (Path("./phewas_cache") / models.safe_basename(INV_B)).is_dir()

        # Assert results from both inversions are in the file
        assert set(df['Inversion'].unique()) == {INV_A, INV_B}

        # Assert global Q value was computed correctly
        assert 'Q_GLOBAL' in df.columns
        # All valid (non-NA) p-values should have been included in a single correction run
        valid_ps = df['P_LRT_Overall'].notna()
        assert df.loc[valid_ps, 'Q_GLOBAL'].nunique() >= 1 # Should have some q-values

        # A_strong_signal should be a hit for INV_A but not INV_B
        strong_hit_a = df[(df['Phenotype'] == 'A_strong_signal') & (df['Inversion'] == INV_A)]
        strong_hit_b = df[(df['Phenotype'] == 'A_strong_signal') & (df['Inversion'] == INV_B)]
        assert strong_hit_a['P_LRT_Overall'].iloc[0] < 0.1
        assert pd.isna(strong_hit_b['P_LRT_Overall'].iloc[0]), "P-value for constant inversion should be NaN"

def test_demographics_age_clipping():
    """Tests that age is correctly clipped to [0, 120] in io.load_demographics_with_stable_age."""
    with temp_workspace():
        mock_bq_client = MagicMock()
        yob_df = pd.DataFrame({'person_id': ['p1', 'p2', 'p3'], 'year_of_birth': [2000, 1900, 2020]})
        obs_df = pd.DataFrame({'person_id': ['p1', 'p2', 'p3'], 'obs_end_year': [2200, 2000, 2000]})
        mock_bq_client.query.side_effect = [
            MagicMock(to_dataframe=MagicMock(return_value=yob_df)),
            MagicMock(to_dataframe=MagicMock(return_value=obs_df))
        ]
        demographics_df = io.load_demographics_with_stable_age(mock_bq_client, "dummy_cdr_id")
        assert demographics_df.loc['p1', 'AGE'] == 120
        assert demographics_df.loc['p2', 'AGE'] == 100
        assert demographics_df.loc['p3', 'AGE'] == 0
        pd.testing.assert_series_equal(demographics_df['AGE_sq'], demographics_df['AGE']**2, check_names=False)


def test_ridge_seeded_refit_matches_mle():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({'const': 1.0,
                      'x1': rng.normal(size=n),
                      'x2': rng.normal(size=n)})
    beta = np.array([-0.2, 1.1, -0.6])
    p = 1/(1+np.exp(-(X.values @ beta)))
    y = pd.Series(rng.binomial(1, p))

    fit_mle = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=200)

    import phewas.models as models
    # This test does not use the test_ctx fixture, so we must set the context manually
    # to disable the n_eff gate that would otherwise cause this test to fail.
    models.CTX = {"MLE_REFIT_MIN_NEFF": 0, "RIDGE_L2_BASE": 1.0}
    orig = models._logit_fit
    def flaky(model, method, **kw):
        if method in ('newton','bfgs') and not kw.get('_already_failed', False):
            from statsmodels.tools.sm_exceptions import PerfectSeparationError
            raise PerfectSeparationError('force ridge seed')
        return orig(model, method, **{**kw, '_already_failed': True})
    try:
        models._logit_fit = flaky
        fit, reason = models._fit_logit_ladder(X, y, ridge_ok=True)
        assert reason in ('ridge_seeded_refit',)
        np.testing.assert_allclose(fit.params.values, fit_mle.params.values, rtol=1e-3, atol=1e-3)
        assert abs(fit.llf - fit_mle.llf) < 1e-3
    finally:
        models._logit_fit = orig


def test_lrt_allows_when_ridge_seeded_but_final_is_mle(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        X = sm.add_constant(pd.concat([core_data['demographics'][['AGE_c','AGE_c_sq']],
                                       core_data['sex'], core_data['pcs'],
                                       core_data['inversion_main']], axis=1))
        anc = pd.get_dummies(core_data['ancestry']['ANCESTRY'], prefix='ANC', drop_first=True, dtype=np.float64)
        X = X.join(anc)

        shm = _init_lrt_worker_from_df(X, {}, core_data['ancestry']['ANCESTRY'], test_ctx)

        from phewas import models as M
        orig = M._logit_fit
        def flaky(model, method, **kw):
            if method in ('newton','bfgs') and not kw.get('_already_failed', False):
                from statsmodels.tools.sm_exceptions import PerfectSeparationError
                raise PerfectSeparationError('force ridge seed')
            return orig(model, method, **{**kw, '_already_failed': True})
        try:
            M._logit_fit = flaky
            task = {"name": "A_strong_signal", "category": "cardio",
                    "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
            M.lrt_overall_worker(task)
            res = json.load(open(Path(test_ctx["LRT_OVERALL_CACHE_DIR"]) / "A_strong_signal.json"))
            assert np.isfinite(res['P_LRT_Overall'])
            assert res.get('LRT_Overall_Reason') in (None, '',) or pd.isna(res['LRT_Overall_Reason'])
        finally:
            M._logit_fit = orig
            shm.close(); shm.unlink()
