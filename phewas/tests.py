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
import math
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
import phewas.pipes as pipes
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


def _init_worker_from_df(df, masks, ctx):
    """Utility to initialize model worker using shared memory."""
    arr = df.to_numpy(dtype=np.float32, copy=True)
    meta, shm = io.create_shared_from_ndarray(arr, readonly=True)
    models.init_worker(meta, list(df.columns), df.index.astype(str), masks, ctx)
    return shm


def _init_lrt_worker_from_df(df, masks, anc_series, ctx):
    arr = df.to_numpy(dtype=np.float32, copy=True)
    meta, shm = io.create_shared_from_ndarray(arr, readonly=True)
    models.init_lrt_worker(meta, list(df.columns), df.index.astype(str), masks, anc_series, ctx)
    return shm


@contextlib.contextmanager
def bootstrap_test_ctx(**overrides):
    """Temporarily override bootstrap-related CTX keys for deterministic tests."""
    original = models.CTX
    snapshot = dict(original)
    defaults = {
        "BOOTSTRAP_B": 512,
        "BOOTSTRAP_B_MAX": 8192,
        "BOOTSTRAP_SEQ_ALPHA": 0.005,
        "BOOTSTRAP_CHUNK": models.BOOTSTRAP_CHUNK,
        "BOOTSTRAP_STREAM_TARGET_BYTES": models.BOOTSTRAP_STREAM_TARGET_BYTES,
        "BOOT_MULTIPLIER": "normal",
        "FDR_ALPHA": 0.05,
        "PROFILE_MAX_ABS_BETA": models.PROFILE_MAX_ABS_BETA,
        "BOOT_SEED_BASE": 1234,
    }
    temp = dict(snapshot)
    temp.update(defaults)
    temp.update(overrides)
    models.CTX = temp
    try:
        yield models.CTX
    finally:
        models.CTX = original


def make_bootstrap_inputs(beta=0.6, seed=0, n=240):
    """Construct a simple dataset for score/bootstrap helper tests."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X_full = pd.DataFrame({"const": np.ones(n), "x": x})
    eta = -0.3 + beta * x
    p = sigmoid(eta)
    y = pd.Series(rng.binomial(1, p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.Logit(y, X_full).fit(disp=0, maxiter=200)
    X_red = X_full[["const"]]
    x_target = X_full["x"].to_numpy(dtype=np.float64, copy=False)
    beta_hat = float(fit.params["x"])
    return X_full, X_red, y, x_target, beta_hat


def _beta_to_or_bounds(lo, hi):
    if lo == -np.inf:
        lo_or = 0.0
    elif np.isfinite(lo):
        lo_or = float(np.exp(lo))
    else:
        lo_or = np.nan
    if hi == np.inf:
        hi_or = np.inf
    elif np.isfinite(hi):
        hi_or = float(np.exp(hi))
    else:
        hi_or = np.nan
    return lo_or, hi_or


def _bh_qvalues(p_values):
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks
    rev = order[::-1]
    q[rev] = np.minimum.accumulate(q[rev])
    return np.clip(q, 0.0, 1.0)


def _bh_threshold(p_values, alpha):
    p = np.sort(np.asarray(p_values, dtype=float))
    m = p.size
    if m == 0:
        return float("nan")
    thresholds = alpha * (np.arange(1, m + 1, dtype=float) / m)
    hits = p <= thresholds
    if np.any(hits):
        idx = int(np.max(np.nonzero(hits)[0]))
        return float(thresholds[idx])
    return float(thresholds[0])

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
            "model_columns": list(core_df.columns), "num_pcs": 10, "min_cases": 10, "min_ctrls": 10,
            "target": TEST_TARGET_INVERSION, "category": "cat", "core_index_fp": models._index_fingerprint(core_df.index),
            "case_idx_fp": "dummy_fp", "allowed_mask_fp": allowed_fp, "ridge_l2_base": 1.0
        }
        # Write the metadata to a JSON file
        io.write_meta_json("test.meta.json", meta)
        models.CTX = test_ctx
        # Check that the skip function returns True when the context is the same
        core_index_fp = models._index_fingerprint(core_df.index)
        assert models._should_skip("test.meta.json", core_df.columns, core_index_fp, "dummy_fp", "cat", TEST_TARGET_INVERSION, allowed_fp)
        # Change the context
        test_ctx_changed = test_ctx.copy()
        test_ctx_changed["MIN_CASES_FILTER"] = 11
        models.CTX = test_ctx_changed
        # Check that the skip function returns False when the context is different
        assert not models._should_skip("test.meta.json", core_df.columns, core_index_fp, "dummy_fp", "cat", TEST_TARGET_INVERSION, allowed_fp)

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
        core_df = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_const']], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        shm = _init_worker_from_df(core_df_with_const, {"cardio": np.ones(len(core_df), dtype=bool)}, test_ctx)
        case_idx = core_df_with_const.index.get_indexer(list(phenos["A_strong_signal"]["cases"]))
        pheno_data = {"name": "A_strong_signal", "category": "cardio", "case_idx": case_idx[case_idx >= 0].astype(np.int32)}
        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])
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
        core_df = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        shm = _init_worker_from_df(core_df_with_const, {"cardio": np.ones(len(core_df), dtype=bool)}, test_ctx)
        case_idx = core_df_with_const.index.get_indexer(list(phenos["B_insufficient"]["cases"]))
        pheno_data = {"name": "B_insufficient", "category": "cardio", "case_idx": case_idx[case_idx != -1]}
        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])
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
        shm = _init_worker_from_df(core_df_with_const, {"cardio": np.ones(len(core_df), dtype=bool)}, test_ctx)
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
        core_df = sm.add_constant(pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1))
        shm = _init_worker_from_df(core_df, {"cardio": np.ones(len(core_df), dtype=bool)}, test_ctx)
        case_idx = core_df.index.get_indexer(list(phenos["A_strong_signal"]["cases"]))
        pheno_data = {"name": "A_strong_signal", "category": "cardio", "case_idx": case_idx[case_idx >= 0]}
        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])
        mtime1 = (Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json").stat().st_mtime
        time.sleep(0.1)
        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])
        mtime2 = (Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json").stat().st_mtime
        assert mtime1 == mtime2
        new_mask = np.ones(len(core_df), dtype=bool); new_mask[:10] = False
        shm.close(); shm.unlink()
        shm = _init_worker_from_df(core_df, {"cardio": new_mask}, test_ctx)
        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])
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
        cases = list(phenos["A_strong_signal"]["cases"])
        core_data['pcs'].loc[cases, 'PC1'] = 1000
        core_data['pcs'].loc[~core_data['pcs'].index.isin(cases), 'PC1'] = -1000
        core_df = sm.add_constant(pd.concat([core_data['demographics'][['AGE_c']], core_data['pcs'], core_data['inversion_main']], axis=1))
        shm = _init_worker_from_df(core_df, {"cardio": np.ones(len(core_df), dtype=bool)}, test_ctx)
        case_idx = core_df.index.get_indexer(cases)
        pheno_data = {"name": "A_strong_signal", "category": "cardio", "case_idx": case_idx[case_idx >= 0]}
        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])
        res = json.load(open(Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json"))
        assert res['Used_Ridge'] is True
        assert res['OR_CI95'] is None
        assert pd.isna(res['P_Value'])
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

        core_df = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main']
        ], axis=1))

        allowed_mask_arr = ~core_df.index.isin(list(cases))
        allowed_mask = {"endo": allowed_mask_arr}
        shm = _init_worker_from_df(core_df, allowed_mask, test_ctx)
        case_idx = core_df.index.get_indexer(list(cases))
        pheno_data = {"name": "sex_restricted_pheno", "category": "endo", "case_idx": case_idx[case_idx >= 0]}

        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])

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

def test_lrt_overall_invalidated_by_penalized_fit(test_ctx):
    """Verifies Stage-1 LRT is skipped if a penalized fit is required."""
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

        assert res['LRT_Overall_Reason'] == 'penalized_fit_in_path'
        assert pd.isna(res['P_LRT_Overall'])
        shm.close(); shm.unlink()

def test_lrt_followup_penalized_fit_omits_ci(test_ctx):
    """Verifies Stage-2 per-ancestry CI is omitted for penalized fits."""
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

        assert 'AFR_CI95' not in res
        assert 'EUR_CI95' in res
        assert 'AMR_CI95' in res
        assert 'EUR_REASON' not in res
        shm.close(); shm.unlink()

# --- Integration Tests ---


def _run_multi_inversion_pipeline(tmpdir):
    INV_A, INV_B = 'chr_test-A-INV-1', 'chr_test-B-INV-2'
    with preserve_run_globals(), \
         patch('phewas.run.bigquery.Client'), \
         patch('phewas.run.io.load_related_to_remove', return_value=set()), \
         patch('phewas.run.supervisor_main', lambda *a, **k: run._pipeline_once()):
        core_data, phenos = make_synth_cohort()
        rng = np.random.default_rng(101)
        core_data['inversion_A'] = pd.DataFrame({INV_A: np.clip(rng.normal(0.8, 0.5, 200), -2, 2)}, index=core_data['demographics'].index)
        core_data['inversion_B'] = pd.DataFrame({INV_B: np.zeros(200)}, index=core_data['demographics'].index)

        p_a = sigmoid(2.5 * core_data['inversion_A'][INV_A] + 0.02 * (core_data['demographics']['AGE'] - 50) - 0.2 * core_data['sex']['sex'])
        cases_a = set(core_data['demographics'].index[rng.random(200) < p_a])
        phenos['A_strong_signal']['cases'] = cases_a

        defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, INV_A)
        run.INVERSION_DOSAGES_FILE = str(Path(tmpdir) / "dummy_dosages.tsv")
        dosages_resolved = os.path.abspath(run.INVERSION_DOSAGES_FILE)
        inv_a_path = Path("./phewas_cache") / f"inversion_{models.safe_basename(INV_A)}_{run._source_key(dosages_resolved, INV_A)}.parquet"
        inv_b_path = Path("./phewas_cache") / f"inversion_{models.safe_basename(INV_B)}_{run._source_key(dosages_resolved, INV_B)}.parquet"
        write_parquet(inv_a_path, core_data['inversion_A'])
        write_parquet(inv_b_path, core_data['inversion_B'])

        local_defs = make_local_pheno_defs_tsv(defs_df, tmpdir)
        run.TARGET_INVERSIONS = [INV_A, INV_B]
        run.MASTER_RESULTS_CSV = str(Path(tmpdir) / "multi_inversion_master.csv")
        run.MIN_CASES_FILTER = run.MIN_CONTROLS_FILTER = 10
        run.FDR_ALPHA = 0.9
        run.PHENOTYPE_DEFINITIONS_URL = str(local_defs)

        dummy_dosage_df = pd.DataFrame({
            'SampleID': core_data['demographics'].index,
            INV_A: core_data['inversion_A'][INV_A],
            INV_B: core_data['inversion_B'][INV_B],
        })
        write_tsv(run.INVERSION_DOSAGES_FILE, dummy_dosage_df)

        run.main()
        output_path = Path(run.MASTER_RESULTS_CSV)
        assert output_path.exists(), "Master CSV file was not created"
        df = pd.read_csv(output_path, sep='\t')
    return df, INV_A, INV_B


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

def test_pipes_run_fits_creates_atomic_results(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort(seed=42)
        pheno_defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        pan_cases = {"cardio": phenos["A_strong_signal"]["cases"] | phenos["B_insufficient"]["cases"], "neuro": phenos["C_moderate_signal"]["cases"]}
        allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(core_df_with_const.index, pan_cases, np.ones(len(core_df_with_const), dtype=bool))
        q = queue.Queue()
        for s_name, p_data in phenos.items():
            case_idx = core_df_with_const.index.get_indexer(list(p_data['cases']))
            q.put({"name": s_name, "category": p_data['category'], "case_idx": case_idx[case_idx != -1]})
        q.put(None)
        pipes.run_fits(q, core_df_with_const, allowed_mask_by_cat, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"], test_ctx, 4.0)
        result_files = os.listdir(test_ctx["RESULTS_CACHE_DIR"])
        assert len(result_files) >= 2 # B_insufficient is skipped
        with open(Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json") as f: res = json.load(f)
        assert res["OR"] > 1.0 and res["P_Value"] < 0.1

def test_cache_equivalence_skips_work(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([core_data['demographics'][['AGE_c', 'AGE_c_sq']], core_data['sex'], core_data['pcs'], core_data['inversion_main']], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        allowed_mask_by_cat = {"cardio": np.ones(len(core_df), dtype=bool), "neuro": np.ones(len(core_df), dtype=bool)}
        q = queue.Queue()
        for s_name, p_data in phenos.items():
            case_idx = core_df_with_const.index.get_indexer(list(p_data['cases']))
            q.put({"name": s_name, "category": p_data['category'], "case_idx": case_idx[case_idx != -1]})
        q.put(None)
        pipes.run_fits(q, core_df_with_const, allowed_mask_by_cat, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"], test_ctx, 4.0)
        mtimes = {f: f.stat().st_mtime for f in Path(test_ctx["RESULTS_CACHE_DIR"]).glob("*.json")}
        time.sleep(1)
        q2 = queue.Queue()
        for s_name, p_data in phenos.items():
            case_idx = core_df_with_const.index.get_indexer(list(p_data['cases']))
            q2.put({"name": s_name, "category": p_data['category'], "case_idx": case_idx[case_idx != -1]})
        q2.put(None)
        pipes.run_fits(q2, core_df_with_const, allowed_mask_by_cat, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"], test_ctx, 4.0)
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
        shm = _init_worker_from_df(X, {"cardio": np.ones(len(X), bool), "neuro": np.ones(len(X), bool)}, test_ctx)
        task = {"name": "A_strong_signal", "category": "cardio", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        models.lrt_overall_worker(task)
        f = Path(test_ctx["LRT_OVERALL_CACHE_DIR"]) / "A_strong_signal.json"
        m0 = f.stat().st_mtime
        time.sleep(1)
        models.lrt_overall_worker(task)
        assert f.stat().st_mtime == m0
        shm.close(); shm.unlink()

def test_final_results_has_ci_and_ancestry_fields():
    with temp_workspace() as tmpdir, preserve_run_globals(), \
         patch('phewas.run.bigquery.Client'), \
         patch('phewas.run.io.load_related_to_remove', return_value=set()), \
         patch('phewas.run.supervisor_main', lambda *a, **k: run._pipeline_once()):
        core_data, phenos = make_synth_cohort()
        defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        local_defs = make_local_pheno_defs_tsv(defs_df, tmpdir)
        run.TARGET_INVERSIONS = [TEST_TARGET_INVERSION]  # Now a list
        run.MASTER_RESULTS_CSV = "master_results.csv"
        run.MIN_CASES_FILTER = run.MIN_CONTROLS_FILTER = 10
        run.FDR_ALPHA = run.LRT_SELECT_ALPHA = 0.4
        run.PHENOTYPE_DEFINITIONS_URL = str(local_defs)
        run.INVERSION_DOSAGES_FILE = "dummy.tsv"
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
    """Integration test to ensure two inversions flow through to a single master file."""
    with temp_workspace() as tmpdir:
        df, inv_a, inv_b = _run_multi_inversion_pipeline(tmpdir)

        assert (Path("./phewas_cache") / models.safe_basename(inv_a)).is_dir()
        assert (Path("./phewas_cache") / models.safe_basename(inv_b)).is_dir()

        assert set(df['Inversion'].unique()) == {inv_a, inv_b}
        assert 'Q_GLOBAL' in df.columns
        valid_ps = df['P_LRT_Overall'].notna()
        assert df.loc[valid_ps, 'Q_GLOBAL'].nunique() >= 1

        strong_hit_a = df[(df['Phenotype'] == 'A_strong_signal') & (df['Inversion'] == inv_a)]
        strong_hit_b = df[(df['Phenotype'] == 'A_strong_signal') & (df['Inversion'] == inv_b)]
        assert strong_hit_a['P_LRT_Overall'].iloc[0] < 0.1
        assert pd.isna(strong_hit_b['P_LRT_Overall'].iloc[0]), "P-value for constant inversion should be NaN"


def test_bh_matches_reference():
    with temp_workspace() as tmpdir:
        df, _, _ = _run_multi_inversion_pipeline(tmpdir)
        valid_mask = df['P_LRT_Overall'].notna()
        observed_q = df.loc[valid_mask, 'Q_GLOBAL'].to_numpy(dtype=float)
        reference_q = _bh_qvalues(df.loc[valid_mask, 'P_LRT_Overall'])
        assert np.allclose(observed_q, reference_q, rtol=0.0, atol=1e-12)

        selected_observed = set(df.index[valid_mask][observed_q <= 0.05])
        selected_reference = set(df.index[valid_mask][reference_q <= 0.05])
        assert selected_observed == selected_reference


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


def test_score_bootstrap_rng_determinism():
    X_full, X_red, y, x_target, beta_hat = make_bootstrap_inputs()
    with bootstrap_test_ctx(BOOT_SEED_BASE=2024, BOOTSTRAP_B=512, BOOTSTRAP_B_MAX=4096) as ctx:
        p1, _, draws1, exceed1 = models._score_bootstrap_from_reduced(
            X_red, y, x_target, seed_key=("unit", "p"), B=ctx["BOOTSTRAP_B"]
        )
        ci1 = models._score_boot_ci_beta(
            X_red,
            y,
            x_target,
            beta_hat,
            seed_key=("unit", "ci"),
            p_at_zero=p1,
        )
        p2, _, draws2, exceed2 = models._score_bootstrap_from_reduced(
            X_red, y, x_target, seed_key=("unit", "p"), B=ctx["BOOTSTRAP_B"]
        )
        ci2 = models._score_boot_ci_beta(
            X_red,
            y,
            x_target,
            beta_hat,
            seed_key=("unit", "ci"),
            p_at_zero=p2,
        )
    assert np.isclose(p1, p2)
    assert draws1 == draws2 and exceed1 == exceed2
    assert ci1["draws_max"] == ci2["draws_max"]
    assert ci1.get("lo") == ci2.get("lo") and ci1.get("hi") == ci2.get("hi")

    with bootstrap_test_ctx(BOOT_SEED_BASE=3030, BOOTSTRAP_B=512, BOOTSTRAP_B_MAX=4096):
        p3, _, _, _ = models._score_bootstrap_from_reduced(
            X_red, y, x_target, seed_key=("unit", "p"), B=512
        )
    assert not np.isclose(p1, p3)


def test_score_bootstrap_ci_coherence():
    seeds = [101, 303]
    for seed_base in seeds:
        for beta in (0.7, 0.0):
            X_full, X_red, y, x_target, beta_hat = make_bootstrap_inputs(beta=beta, seed=seed_base)
            with bootstrap_test_ctx(BOOT_SEED_BASE=seed_base, BOOTSTRAP_B=768, BOOTSTRAP_B_MAX=8192):
                p_val, _, _, _ = models._score_bootstrap_from_reduced(
                    X_red, y, x_target, seed_key=("coherence", beta, "p"), B=768
                )
                ci = models._score_boot_ci_beta(
                    X_red,
                    y,
                    x_target,
                    beta_hat,
                    seed_key=("coherence", beta, "ci"),
                    p_at_zero=p_val,
                )
            lo_or, hi_or = _beta_to_or_bounds(ci.get("lo"), ci.get("hi"))
            if np.isfinite(p_val) and p_val < 0.05:
                assert ci.get("valid") is True
                assert not (np.isfinite(lo_or) and np.isfinite(hi_or) and lo_or <= 1.0 <= hi_or)
            else:
                if ci.get("valid"):
                    assert lo_or <= 1.0 <= hi_or or (lo_or == 0.0 and np.isinf(hi_or))


def test_score_bootstrap_bits_magnitude():
    rng = np.random.default_rng(7)
    n = 180
    x = rng.normal(size=n)
    X_red = np.ones((n, 1))
    xt = x.astype(np.float64)
    eta_small = 0.2 * x
    eta_big = 1.2 * x
    y_small = rng.binomial(1, sigmoid(eta_small))
    y_big = rng.binomial(1, sigmoid(eta_big))
    bits_small = models._score_bootstrap_bits(X_red, y_small, xt, 0.0, kind="mle")
    bits_big = models._score_bootstrap_bits(X_red, y_big, xt, 0.0, kind="mle")
    assert bits_small is not None and bits_big is not None
    assert np.isfinite(bits_small["T_obs"]) and np.isfinite(bits_big["T_obs"]) and bits_small["den"] > 0
    assert bits_big["T_obs"] > bits_small["T_obs"]


def test_score_bootstrap_variance_shrinkage():
    X_full, X_red, y, x_target, _ = make_bootstrap_inputs(beta=0.5, seed=11)
    bits = models._score_bootstrap_bits(X_red.to_numpy(dtype=np.float64), y.to_numpy(), x_target, 0.0, kind="mle")
    with bootstrap_test_ctx(BOOT_SEED_BASE=909):
        rng_small = models._bootstrap_rng(("var", "seed"))
        detail_small = models._score_bootstrap_p_from_bits(bits, B=1024, rng=rng_small, return_detail=True)
        rng_large = models._bootstrap_rng(("var", "seed"))
        detail_large = models._score_bootstrap_p_from_bits(bits, B=32768, rng=rng_large, return_detail=True)
    assert detail_large["draws"] > detail_small["draws"]
    assert abs(detail_small["p"] - detail_large["p"]) < 0.02


def test_score_boot_ci_boundary_label():
    X_full, X_red, y, x_target, beta_hat = make_bootstrap_inputs(beta=1.1, seed=21)
    with bootstrap_test_ctx(BOOT_SEED_BASE=505):
        ci = models._score_boot_ci_beta(
            X_red,
            y,
            x_target,
            beta_hat,
            max_abs_beta=0.25,
            seed_key=("boundary", "ci"),
        )
    assert ci.get("valid") is True
    assert ci.get("sided") == "one"
    assert "boundary" in ci.get("label", "")


def test_bootstrap_multiplier_modes():
    h = np.ones(12)
    threshold = 0.5

    class RademacherRNG:
        def __init__(self):
            self.samples = []

        def choice(self, values, size=None):
            arr = np.random.default_rng(0).choice(values, size=size)
            self.samples.append(arr)
            return arr

        def standard_normal(self, *_, **__):
            raise AssertionError("standard_normal should not be used for rademacher")

    with bootstrap_test_ctx(BOOT_MULTIPLIER="rademacher"):
        rng = RademacherRNG()
        models._bootstrap_chunk_exceed(h, threshold, rng, reps=4, target_bytes=256)
        assert rng.samples, "Rademacher generator should have been used"
        for arr in rng.samples:
            assert np.isin(arr, [-1.0, 1.0]).all()

    class NormalRNG:
        def __init__(self):
            self.called = False

        def standard_normal(self, size=None):
            self.called = True
            return np.zeros(size, dtype=np.float64)

        def choice(self, *args, **kwargs):
            raise AssertionError("choice should not be used for normal multipliers")

    with bootstrap_test_ctx(BOOT_MULTIPLIER="normal"):
        rng = NormalRNG()
        models._bootstrap_chunk_exceed(h, threshold, rng, reps=4, target_bytes=256)
        assert rng.called is True


def test_adaptive_bootstrap_planner_triggers_near_cutoff():
    with temp_workspace():
        results_dir = Path("./phewas_cache/results_atomic")
        results_dir.mkdir(parents=True, exist_ok=True)
        ctx = {
            "FDR_ALPHA": 0.05,
            "BOOTSTRAP_SEQ_ALPHA": 0.01,
            "BOOTSTRAP_B_MAX": 64000,
        }
        records = [
            {"Phenotype": "very_sig", "P_Value": 0.0005, "Inference_Type": "mle"},
            {
                "Phenotype": "borderline",
                "P_Value": (1 + 34) / (2000 + 1),
                "Inference_Type": "score_boot",
                "Boot_Total": 2000,
                "Boot_Exceed": 34,
            },
            {"Phenotype": "nullish", "P_Value": 0.3, "Inference_Type": "mle"},
        ]
        for rec in records:
            payload = {**rec}
            payload.setdefault("Boot_Total", None)
            payload.setdefault("Boot_Exceed", None)
            io.atomic_write_json(results_dir / f"{rec['Phenotype']}.json", payload)

        plan = models.plan_score_bootstrap_refinement(str(results_dir), ctx, safety_factor=8.0)
        assert plan, "Expected refinement plan to schedule at least one phenotype"
        assert {entry["name"] for entry in plan} == {"borderline"}
        entry = plan[0]
        assert entry["min_total"] > records[1]["Boot_Total"]
        assert entry["min_total"] <= ctx["BOOTSTRAP_B_MAX"]
        lo, hi = models._clopper_pearson_interval(34, 2000, alpha=ctx["BOOTSTRAP_SEQ_ALPHA"])
        assert lo < entry["alpha_target"] < hi
        t_star = _bh_threshold([rec["P_Value"] for rec in records], ctx["FDR_ALPHA"])
        assert math.isclose(entry["alpha_target"], t_star, rel_tol=0.0, abs_tol=1e-12)


def test_adaptive_bootstrap_round_trip_updates_draws_and_keeps_CI_coherent(test_ctx):
    test_ctx = test_ctx.copy()
    test_ctx.update({"BOOTSTRAP_B": 256, "BOOTSTRAP_B_MAX": 4096, "BOOT_MULTIPLIER": "normal", "FDR_ALPHA": 0.05})
    with temp_workspace():
        core_data, phenos = make_synth_cohort(N=160)
        core_df = sm.add_constant(pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main'],
        ], axis=1))
        masks = {
            "cardio": np.ones(len(core_df), dtype=bool),
            "neuro": np.ones(len(core_df), dtype=bool),
        }
        shm = _init_worker_from_df(core_df, masks, test_ctx)
        cat_map = {name: data["category"] for name, data in phenos.items()}

        target_names = ["A_strong_signal", "C_moderate_signal"]
        patchers = [
            patch.object(models, "_ok_mle_fit", return_value=False),
            patch.object(models, "_firth_refit", return_value=None),
            patch.object(models, "_score_test_from_reduced", return_value=(np.nan, None)),
        ]
        try:
            for p in patchers:
                p.start()
            for name in target_names:
                case_idx = core_df.index.get_indexer(list(phenos[name]["cases"]))
                pheno_data = {
                    "name": name,
                    "category": cat_map[name],
                    "case_idx": case_idx[case_idx >= 0].astype(np.int32),
                }
                models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])
        finally:
            for p in reversed(patchers):
                p.stop()

        results_dir = Path(test_ctx["RESULTS_CACHE_DIR"])
        baseline = {}
        for name in target_names:
            with open(results_dir / f"{name}.json") as fh:
                baseline[name] = json.load(fh)
                assert baseline[name]["Inference_Type"] == "score_boot"

        border_path = results_dir / f"{target_names[0]}.json"
        border_record = baseline[target_names[0]]
        border_record.update({
            "P_Value": (1 + 20) / (1024 + 1),
            "Boot_Total": 1024,
            "Boot_Exceed": 20,
        })
        io.atomic_write_json(border_path, border_record)

        plan_ctx = dict(test_ctx)
        plan = models.plan_score_bootstrap_refinement(str(results_dir), plan_ctx, safety_factor=8.0)
        assert plan, "Expected refinement plan"
        planned = {entry["name"] for entry in plan}
        assert target_names[0] in planned

        before_totals = {name: baseline[name]["Boot_Total"] for name in target_names}
        lookup = {entry["name"]: entry for entry in plan}
        p_values = []
        for name in target_names:
            with open(results_dir / f"{name}.json") as fh:
                rec = json.load(fh)
                p_values.append(rec.get("P_Value", np.nan))
        t_star = _bh_threshold(p_values, plan_ctx["FDR_ALPHA"])
        assert math.isclose(lookup[target_names[0]]["alpha_target"], t_star, rel_tol=0.0, abs_tol=1e-12)

        patchers = [
            patch.object(models, "_ok_mle_fit", return_value=False),
            patch.object(models, "_firth_refit", return_value=None),
            patch.object(models, "_score_test_from_reduced", return_value=(np.nan, None)),
        ]
        try:
            for p in patchers:
                p.start()
            for name in planned:
                case_idx = core_df.index.get_indexer(list(phenos[name]["cases"]))
                pheno_data = {
                    "name": name,
                    "category": cat_map[name],
                    "case_idx": case_idx[case_idx >= 0].astype(np.int32),
                    "min_total": lookup[name]["min_total"],
                    "refine_round": 1,
                }
                models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])
        finally:
            for p in reversed(patchers):
                p.stop()

        updated = {}
        for name in target_names:
            with open(results_dir / f"{name}.json") as fh:
                updated[name] = json.load(fh)

        for name in planned:
            assert updated[name]["Boot_Total"] >= lookup[name]["min_total"]
            assert updated[name]["Boot_Total"] > before_totals[name]
        for name in set(target_names) - planned:
            assert updated[name]["Boot_Total"] == before_totals[name]

        for rec in updated.values():
            if rec.get("P_Source") == "score_boot" and rec.get("P_Value", 1.0) < 0.05 and rec.get("CI_Valid"):
                assert not (rec.get("CI_LO_OR") <= 1.0 <= rec.get("CI_HI_OR"))
            if rec.get("P_Source") == "score_boot":
                notes = rec.get("Model_Notes", "")
                for token in [
                    "inference=score_boot",
                    "ci=score_boot_multiplier",
                    "boot=",
                    "boot_seq_alpha=",
                    "boot_multiplier=",
                ]:
                    assert token in notes

        shm.close()
        shm.unlink()


def test_empirical_fdr_control_at_alpha_point05():
    rng = np.random.default_rng(12345)
    m = 120
    n = 350
    alt_count = int(0.4 * m)
    is_alt = np.zeros(m, dtype=bool)
    is_alt[:alt_count] = True
    rng.shuffle(is_alt)
    intercept = -0.3
    beta_alt = 0.7
    datasets = []

    with bootstrap_test_ctx(BOOTSTRAP_B=512, BOOTSTRAP_B_MAX=8192, BOOTSTRAP_SEQ_ALPHA=0.01, BOOT_SEED_BASE=2025, FDR_ALPHA=0.05) as ctx:
        pvals = np.empty(m, dtype=float)
        draws = np.zeros(m, dtype=int)
        ones_col = np.ones((n, 1))
        for idx in range(m):
            x = rng.normal(size=n)
            beta = beta_alt if is_alt[idx] else 0.0
            eta = intercept + beta * x
            p = sigmoid(eta)
            y = rng.binomial(1, p)
            datasets.append((x, y))
            p_boot, _, draws_i, _ = models._score_bootstrap_from_reduced(
                ones_col,
                y,
                x,
                seed_key=("empirical_fdr", idx),
                B=ctx["BOOTSTRAP_B"],
                B_max=ctx["BOOTSTRAP_B_MAX"],
            )
            pvals[idx] = p_boot
            draws[idx] = draws_i
        assert np.all(np.isfinite(pvals))

        t_star = _bh_threshold(pvals, ctx["FDR_ALPHA"])
        refine_min = int(min(ctx["BOOTSTRAP_B_MAX"], math.ceil(8.0 / max(t_star, 1e-12))))
        band_mask = (pvals >= max(0.5 * t_star, 1e-12)) & (pvals <= 2.0 * t_star)
        for idx in np.where(band_mask)[0]:
            x, y = datasets[idx]
            p_refine, _, draws_ref, _ = models._score_bootstrap_from_reduced(
                ones_col,
                y,
                x,
                seed_key=("empirical_fdr", idx),
                B=ctx["BOOTSTRAP_B"],
                B_max=ctx["BOOTSTRAP_B_MAX"],
                min_total=refine_min,
            )
            pvals[idx] = p_refine
            draws[idx] = draws_ref
            assert draws_ref >= refine_min

    qvals = _bh_qvalues(pvals)
    selected = np.where(qvals <= 0.05)[0]
    assert selected.size > 0
    V = int(np.sum(~is_alt[selected]))
    S = int(np.sum(is_alt[selected]))
    fdr_observed = V / selected.size
    power = S / max(np.sum(is_alt), 1)
    assert fdr_observed <= 0.08
    assert power >= 0.60


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
