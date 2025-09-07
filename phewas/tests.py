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
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add the current directory to the path to allow absolute imports of phewas modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import run, iox as io, pheno, models, pipes
from scipy.special import expit as sigmoid

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

def prime_all_caches_for_run(core_data, phenos, cdr_codename, target_inversion, cache_dir="./phewas_cache"):
    os.makedirs(cache_dir, exist_ok=True)

    write_parquet(Path(cache_dir) / f"demographics_{cdr_codename}.parquet", core_data["demographics"])
    write_parquet(Path(cache_dir) / f"inversion_{target_inversion}.parquet", core_data["inversion_main"])
    write_parquet(Path(cache_dir) / "pcs_10.parquet", core_data["pcs"])
    write_parquet(Path(cache_dir) / "genetic_sex.parquet", core_data["sex"])
    write_parquet(Path(cache_dir) / "ancestry_labels.parquet", core_data["ancestry"])

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
        # On Linux, getrusage returns KB. On macOS, it returns bytes.
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
        meta = {"model_columns": list(core_df.columns), "num_pcs": 10, "min_cases": 10, "min_ctrls": 10,
                "target": TEST_TARGET_INVERSION, "category": "cat", "core_index_fp": models._index_fingerprint(core_df.index),
                "case_idx_fp": "dummy_fp"}
        io.write_meta_json("test.meta.json", meta)

        models.CTX = test_ctx
        assert models._should_skip("test.meta.json", core_df, "dummy_fp", "cat", TEST_TARGET_INVERSION)

        test_ctx_changed = test_ctx.copy(); test_ctx_changed["MIN_CASES_FILTER"] = 11
        models.CTX = test_ctx_changed
        assert not models._should_skip("test.meta.json", core_df, "dummy_fp", "cat", TEST_TARGET_INVERSION)

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
        core_df = pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_const']
        ], axis=1)
        core_df_with_const = sm.add_constant(core_df)

        Path(test_ctx["RESULTS_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        models.init_worker(core_df_with_const, {"cardio": np.ones(len(core_df), dtype=bool)}, test_ctx)

        case_idx = core_df_with_const.index.get_indexer(list(phenos["A_strong_signal"]["cases"]))
        pheno_data = {"name": "A_strong_signal", "category": "cardio", "case_idx": case_idx[case_idx >= 0].astype(np.int32)}
        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])

        result_path = Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)
        assert all(pd.isna(res[k]) for k in ["Beta", "OR", "P_Value"])

def test_worker_insufficient_counts_skips(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        core_df = pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main']
        ], axis=1)
        core_df_with_const = sm.add_constant(core_df)

        Path(test_ctx["RESULTS_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        models.init_worker(core_df_with_const, {"cardio": np.ones(len(core_df), dtype=bool)}, test_ctx)

        case_idx = core_df_with_const.index.get_indexer(list(phenos["B_insufficient"]["cases"]))
        pheno_data = {"name": "B_insufficient", "category": "cardio", "case_idx": case_idx[case_idx != -1]}
        models.run_single_model_worker(pheno_data, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"])

        result_path = Path(test_ctx["RESULTS_CACHE_DIR"]) / "B_insufficient.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)
        assert res["Skip_Reason"] == "insufficient_cases_or_controls"

def test_lrt_rank_and_df_positive(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main']
        ], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        anc_series = core_data['ancestry']['ANCESTRY'].str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        core_df_with_const = core_df_with_const.join(A, how="left").fillna({c: 0.0 for c in A.columns})

        Path(test_ctx["LRT_OVERALL_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        models.init_worker(core_df_with_const, {"cardio": np.ones(len(core_df), dtype=bool)}, test_ctx)

        task = {"name": "A_strong_signal", "category": "cardio", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        models.lrt_overall_worker(task)

        result_path = Path(test_ctx["LRT_OVERALL_CACHE_DIR"]) / "A_strong_signal.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)
        assert res["LRT_df_Overall"] >= 1
        assert 0 < res["P_LRT_Overall"] <= 1

def test_followup_includes_ancestry_levels_and_splits(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main']
        ], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        anc_series = core_data['ancestry']['ANCESTRY'].str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        core_df_with_const = core_df_with_const.join(A, how="left").fillna({c: 0.0 for c in A.columns})

        Path(test_ctx["LRT_FOLLOWUP_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        models.init_lrt_worker(core_df_with_const, {"neuro": np.ones(len(core_df), dtype=bool)}, core_data['ancestry']['ANCESTRY'], test_ctx)

        task = {"name": "C_moderate_signal", "category": "neuro", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        models.lrt_followup_worker(task)

        result_path = Path(test_ctx["LRT_FOLLOWUP_CACHE_DIR"]) / "C_moderate_signal.json"
        assert result_path.exists()
        with open(result_path) as f:
            res = json.load(f)
        assert "eur" in res["LRT_Ancestry_Levels"] and "afr" in res["LRT_Ancestry_Levels"]
        assert res["EUR_N"] > 0 and res["AFR_N"] > 0
        assert pd.notna(res["EUR_OR"])

# --- Integration Tests ---

def test_fetcher_producer_drains_cache_only():
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        pheno_defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_index = pd.Index([f"p{i:07d}" for i in range(1, 201)], name="person_id")
        q = queue.Queue(maxsize=100)

        fetcher_thread = threading.Thread(target=pheno.phenotype_fetcher_worker,
            args=(q, pheno_defs_df, None, None, {}, TEST_CDR_CODENAME, core_index, "./phewas_cache", 128, 4))
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
        core_data, phenos = make_synth_cohort()
        pheno_defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main']
        ], axis=1)
        core_df_with_const = sm.add_constant(core_df)

        pan_cases = {"cardio": phenos["A_strong_signal"]["cases"] | phenos["B_insufficient"]["cases"], "neuro": phenos["C_moderate_signal"]["cases"]}
        allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(core_df_with_const.index, pan_cases, np.ones(len(core_df_with_const), dtype=bool))

        q = queue.Queue()
        for s_name, p_data in phenos.items():
            case_idx = core_df_with_const.index.get_indexer(list(p_data['cases']))
            q.put({"name": s_name, "category": p_data['category'], "case_idx": case_idx[case_idx != -1]})
        q.put(None)

        pipes.run_fits(q, core_df_with_const, allowed_mask_by_cat, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"], test_ctx)

        result_files = os.listdir(test_ctx["RESULTS_CACHE_DIR"])
        assert len(result_files) >= len(phenos)

        with open(Path(test_ctx["RESULTS_CACHE_DIR"]) / "A_strong_signal.json") as f:
            res = json.load(f)
        assert res["OR"] > 1.0 and res["P_Value"] < 0.1

def test_cache_equivalence_skips_work(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        core_df = pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main']
        ], axis=1)
        core_df_with_const = sm.add_constant(core_df)
        allowed_mask_by_cat = {"cardio": np.ones(len(core_df), dtype=bool), "neuro": np.ones(len(core_df), dtype=bool)}

        q = queue.Queue()
        for s_name, p_data in phenos.items():
            case_idx = core_df_with_const.index.get_indexer(list(p_data['cases']))
            q.put({"name": s_name, "category": p_data['category'], "case_idx": case_idx[case_idx != -1]})
        q.put(None)

        pipes.run_fits(q, core_df_with_const, allowed_mask_by_cat, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"], test_ctx)

        mtimes = {f: f.stat().st_mtime for f in Path(test_ctx["RESULTS_CACHE_DIR"]).glob("*.json")}
        time.sleep(1)

        q2 = queue.Queue()
        for s_name, p_data in phenos.items():
            case_idx = core_df_with_const.index.get_indexer(list(p_data['cases']))
            q2.put({"name": s_name, "category": p_data['category'], "case_idx": case_idx[case_idx != -1]})
        q2.put(None)
        pipes.run_fits(q2, core_df_with_const, allowed_mask_by_cat, TEST_TARGET_INVERSION, test_ctx["RESULTS_CACHE_DIR"], test_ctx)

        mtimes_after = {f: f.stat().st_mtime for f in Path(test_ctx["RESULTS_CACHE_DIR"]).glob("*.json")}
        assert mtimes == mtimes_after

def test_lrt_overall_meta_idempotency(test_ctx):
    with temp_workspace():
        core_data, phenos = make_synth_cohort()
        prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        X_base = pd.concat([
            core_data['demographics'][['AGE_c', 'AGE_c_sq']],
            core_data['sex'],
            core_data['pcs'],
            core_data['inversion_main']
        ], axis=1)
        X = sm.add_constant(X_base)
        anc_series = core_data['ancestry']['ANCESTRY'].str.lower()
        A = pd.get_dummies(pd.Categorical(anc_series), prefix='ANC', drop_first=True, dtype=np.float64)
        X = X.join(A, how="left").fillna({c: 0.0 for c in A.columns})
        Path(test_ctx["LRT_OVERALL_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        models.init_worker(X, {"cardio": np.ones(len(X), bool), "neuro": np.ones(len(X), bool)}, test_ctx)
        task = {"name": "A_strong_signal", "category": "cardio", "cdr_codename": TEST_CDR_CODENAME, "target": TEST_TARGET_INVERSION}
        models.lrt_overall_worker(task)
        f = Path(test_ctx["LRT_OVERALL_CACHE_DIR"]) / "A_strong_signal.json"
        m0 = f.stat().st_mtime
        time.sleep(1)
        models.lrt_overall_worker(task)
        assert f.stat().st_mtime == m0

def test_final_results_has_ci_and_ancestry_fields():
    with temp_workspace() as tmpdir, preserve_run_globals(), \
         patch('run.bigquery.Client'), \
         patch('run.io.load_related_to_remove', return_value=set()):
        core_data, phenos = make_synth_cohort()
        defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        local_defs = make_local_pheno_defs_tsv(defs_df, tmpdir)
        run.TARGET_INVERSION = TEST_TARGET_INVERSION
        run.MIN_CASES_FILTER = run.MIN_CONTROLS_FILTER = 10
        run.FDR_ALPHA = run.LRT_SELECT_ALPHA = 0.4
        run.PHENOTYPE_DEFINITIONS_URL = str(local_defs)
        run.INVERSION_DOSAGES_FILE = "dummy.tsv"
        write_tsv(run.INVERSION_DOSAGES_FILE, core_data["inversion_main"].reset_index().rename(columns={'person_id':'SampleID'}))
        run.main()

        df = pd.read_csv(f"phewas_results_{TEST_TARGET_INVERSION}.csv")
        assert (df["OR_CI95"].astype(str).str.contains(",")).any()
        any_lrts = [c for c in df.columns if c.endswith("_P_FDR")]
        assert len(any_lrts) > 0

def test_memory_envelope_relative():
    if not os.environ.get("RUN_SLOW"):
        pytest.skip("Memory test is slow, requires RUN_SLOW=1")

    with temp_workspace():
        base_rss = read_rss_bytes()

        n_phenos, n_participants = (100, 10000) if os.environ.get("RUN_SLOW") == "1" else (20, 2000)
        envelope_gb = 1.0 if os.environ.get("RUN_SLOW") == "1" else 0.25

        core_data, phenos_base = make_synth_cohort(N=n_participants)
        phenos = {f"pheno_{i}": phenos_base["A_strong_signal"] for i in range(n_phenos)}

        pheno_defs_df = prime_all_caches_for_run(core_data, phenos, TEST_CDR_CODENAME, TEST_TARGET_INVERSION)
        local_defs_path = make_local_pheno_defs_tsv(pheno_defs_df, Path("."))

        with preserve_run_globals():
            run.MIN_CASES_FILTER, run.MIN_CONTROLS_FILTER = 10, 10
            run.PHENOTYPE_DEFINITIONS_URL = str(local_defs_path)
            run.TARGET_INVERSION = TEST_TARGET_INVERSION
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
            try:
                run.main()
            finally:
                stop_event.set()
                poll_thread.join()

            peak_delta_gb = (peak_rss[0] - base_rss) / (1024**3)
            assert peak_delta_gb < envelope_gb, f"Peak memory delta {peak_delta_gb:.3f} GB exceeded envelope"
