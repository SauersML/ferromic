import os
import re
import time
import ast
import warnings
import gc
import psutil
from datetime import datetime
import threading
import queue
from functools import partial
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pandas.api.types import is_numeric_dtype

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from google.cloud import bigquery
from statsmodels.stats.multitest import multipletests
import faulthandler
faulthandler.enable()

def _global_excepthook(exc_type, exc, tb):
    """
    Uncaught exception hook that prints a full stack trace immediately across threads and subprocesses.
    """
    import traceback, sys
    print("[TRACEBACK] Uncaught exception:", flush=True)
    traceback.print_exception(exc_type, exc, tb)
    sys.stderr.flush()

import sys
sys.excepthook = _global_excepthook
def _thread_excepthook(args):
    _global_excepthook(args.exc_type, args.exc_value, args.exc_traceback)
threading.excepthook = _thread_excepthook

# Ensure line-buffered, real-time stdout/stderr for consistent progress bars and diagnostics across threads and subprocesses.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message=r"^overflow encountered in exp",
    category=RuntimeWarning,
    module=r"^statsmodels\.discrete\.discrete_model$",
)
warnings.filterwarnings(
    "ignore",
    message=r"^divide by zero encountered in log",
    category=RuntimeWarning,
    module=r"^statsmodels\.discrete\.discrete_model$",
)

# --- Configuration ---
TARGET_INVERSION = 'chr17-45585160-INV-706887'
PHENOTYPE_DEFINITIONS_URL = "https://github.com/SauersML/ferromic/raw/refs/heads/main/data/significant_heritability_diseases.tsv"

# --- Performance & Memory Tuning ---
QUEUE_MAX_SIZE = cpu_count() * 4
LOADER_THREADS = 32

LOADER_CHUNK_SIZE = 128

# --- Data sources and caching ---
CACHE_DIR = "./phewas_cache"
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results_atomic")
# Per-phenotype Likelihood Ratio Test caches for resume-safe execution
LRT_OVERALL_CACHE_DIR = os.path.join(CACHE_DIR, "lrt_overall")
LRT_FOLLOWUP_CACHE_DIR = os.path.join(CACHE_DIR, "lrt_followup")
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
RELATEDNESS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"

# --- Model parameters ---
NUM_PCS = 10
MIN_CASES_FILTER = 1000
MIN_CONTROLS_FILTER = 500
FDR_ALPHA = 0.05

# --- Per-ancestry thresholds and multiple-testing for ancestry splits ---
PER_ANC_MIN_CASES = 50
PER_ANC_MIN_CONTROLS = 50
ANCESTRY_ALPHA = 0.05
ANCESTRY_P_ADJ_METHOD = "fdr_bh"
LRT_SELECT_ALPHA = 0.05


# --- Regularization strength for ridge fallback in unstable fits ---
RIDGE_L2_BASE = 1.0

# --- Suppress pandas warnings ---
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Worker Initializer for Multiprocessing ---
worker_core_df = None

def init_worker(df_to_share, masks):
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"^overflow encountered in exp",
        category=RuntimeWarning,
        module=r"^statsmodels\.discrete\.discrete_model$",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"^divide by zero encountered in log",
        category=RuntimeWarning,
        module=r"^statsmodels\.discrete\.discrete_model$",
    )
    """Sends the large core_df and precomputed masks to each worker process once in a read-only fashion."""
    global worker_core_df, allowed_mask_by_cat, N_core
    worker_core_df = df_to_share
    allowed_mask_by_cat = masks
    N_core = len(df_to_share)
    print(f"[Worker-{os.getpid()}] Initialized and received shared core dataframe and masks.", flush=True)

class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

def _index_fingerprint(index) -> str:
    """Order-insensitive fingerprint of a person_id index."""
    s = '\n'.join(sorted(map(str, index)))
    return hashlib.sha256(s.encode()).hexdigest()[:16] + f":{len(index)}"

def _bytes_fp(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def _read_meta_json(path) -> dict | None:
    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        return None

def _write_meta_json(path, meta: dict):
    pd.Series(meta).to_json(path)

def rss_gb():
    """Returns the resident set size of the current process in gigabytes for lightweight memory instrumentation."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def sanitize_name(name):
    """Cleans a disease name to be a valid identifier."""
    name = re.sub(r'[\*\(\)\[\]\/\']', '', name)
    name = re.sub(r'[\s,-]+', '_', name.strip())
    return name

def parse_icd_codes(code_string):
    """Parses a semi-colon delimited string of ICD codes into a clean set."""
    if pd.isna(code_string) or not isinstance(code_string, str): return set()
    return {code.strip().strip('"') for code in code_string.split(';') if code.strip()}

def get_cached_or_generate(cache_path, generation_func, *args, **kwargs):
    """
    Generic caching wrapper with validation. Compatible with pre-existing caches.
    If the existing file fails shape/schema/NA checks, regenerate it by calling generation_func.
    """
    def _valid_demographics(df):
        ok = all(c in df.columns for c in ["AGE", "AGE_sq"])
        ok = ok and is_numeric_dtype(df["AGE"]) and is_numeric_dtype(df["AGE_sq"])
        if not ok: return False
        # AGE_sq consistency (allow minor float noise)
        return np.nanmax(np.abs(df["AGE_sq"].to_numpy() - (df["AGE"].to_numpy() ** 2))) < 1e-6

    def _valid_inversion(df):
        # exactly one column: the current TARGET_INVERSION; numeric; no NA-only rows
        cols = list(df.columns)
        if cols != [TARGET_INVERSION]: 
            return False
        return is_numeric_dtype(df[TARGET_INVERSION]) and df[TARGET_INVERSION].notna().any()

    def _valid_pcs(df):
        expected = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
        if list(df.columns) != expected:
            return False
        # all numeric, no all-NA columns
        return all(is_numeric_dtype(df[c]) and df[c].notna().any() for c in expected)

    def _valid_sex(df):
        if list(df.columns) != ["sex"]:
            return False
        if not is_numeric_dtype(df["sex"]):
            return False
        # allow only 0/1 (with possible missing filtered at join time)
        uniq = set(pd.unique(df["sex"].dropna()))
        return uniq.issubset({0, 1})

    def _needs_validation(path):
        bn = os.path.basename(path)
        return (
            bn.startswith("demographics_")
            or bn.startswith("inversion_")
            or bn.startswith("pcs_")
            or bn == "genetic_sex.parquet"
        )

    def _validate(path, df):
        bn = os.path.basename(path)
        if bn.startswith("demographics_"):
            return _valid_demographics(df)
        if bn.startswith("inversion_"):
            return _valid_inversion(df)
        if bn.startswith("pcs_"):
            return _valid_pcs(df)
        if bn == "genetic_sex.parquet":
            return _valid_sex(df)
        # everything else: accept as-is
        return True

    if os.path.exists(cache_path):
        print(f"  -> Found cache, loading from '{cache_path}'...")
        try:
            df = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"  -> Cache unreadable ({e}); regenerating...")
            df = generation_func(*args, **kwargs)
            df.to_parquet(cache_path)
            return df

        # Basic index hygiene for joins
        if df.index.name != "person_id":
            try:
                df.index = df.index.astype(str)
                df.index.name = "person_id"
            except Exception:
                pass

        # Only validate known core covariates; regenerate if invalid
        if _needs_validation(cache_path) and not _validate(cache_path, df):
            print(f"  -> Cache at '{cache_path}' failed validation; regenerating...")
            df = generation_func(*args, **kwargs)
            df.to_parquet(cache_path)
        return df

    print(f"  -> No cache found at '{cache_path}'. Generating data...")
    data = generation_func(*args, **kwargs)
    data.to_parquet(cache_path)
    return data

def _load_inversions():
    """Loads the target inversion dosage."""
    try:
        df = pd.read_csv(INVERSION_DOSAGES_FILE, sep="\t", usecols=["SampleID", TARGET_INVERSION])
        df['SampleID'] = df['SampleID'].astype(str)
        return df.set_index('SampleID').rename_axis('person_id')
    except Exception as e:
        raise RuntimeError(f"Failed to load inversion data: {e}")

def _load_pcs(gcp_project):
    """Loads genetic PCs."""
    try:
        raw_pcs = pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True})
        pc_mat = pd.DataFrame(
            raw_pcs["pca_features"].apply(lambda s: ast.literal_eval(s) if pd.notna(s) else [np.nan]*16).tolist(),
            columns=[f"PC{i}" for i in range(1, 17)]
        )
        pc_df = pc_mat.assign(person_id=raw_pcs["research_id"].astype(str)).set_index("person_id")
        return pc_df[[f'PC{i}' for i in range(1, NUM_PCS + 1)]]
    except Exception as e:
        raise RuntimeError(f"Failed to load PCs: {e}")

def _load_genetic_sex(gcp_project):
    """Loads genetically-inferred sex and encodes it as a numeric variable."""
    print("    -> Loading genetically-inferred sex (ploidy)...")
    sex_df = pd.read_csv(SEX_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True},
                         usecols=['research_id', 'dragen_sex_ploidy'])
    
    sex_df['sex'] = np.nan
    sex_df.loc[sex_df['dragen_sex_ploidy'] == 'XX', 'sex'] = 0
    sex_df.loc[sex_df['dragen_sex_ploidy'] == 'XY', 'sex'] = 1
    
    sex_df = sex_df.rename(columns={'research_id': 'person_id'})
    sex_df['person_id'] = sex_df['person_id'].astype(str)
    
    return sex_df[['person_id', 'sex']].dropna().set_index('person_id')

def _load_ancestry_labels(gcp_project):
    """Loads predicted ancestry labels for each person."""
    print("    -> Loading genetic ancestry labels...")
    raw = pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True},
                      usecols=['research_id', 'ancestry_pred'])
    df = raw.rename(columns={'research_id': 'person_id', 'ancestry_pred': 'ANCESTRY'})
    df['person_id'] = df['person_id'].astype(str)
    df = df.dropna(subset=['ANCESTRY'])
    return df.set_index('person_id')[['ANCESTRY']]

def _load_related_to_remove(gcp_project):
    """Loads the pre-computed list of related individuals to prune."""
    print("    -> Loading list of related individuals to exclude...")
    related_df = pd.read_csv(RELATEDNESS_URI, sep="\t", header=None, names=['person_id'],
                             storage_options={"project": gcp_project, "requester_pays": True})
    
    # Return a set for extremely fast filtering
    return set(related_df['person_id'].astype(str))

def _load_demographics_with_stable_age(bq_client, cdr_id):
    """
    Loads demographics, calculating a stable and reproducible age for each participant
    based on their last observation date in the dataset.
    """
    print("    -> Generating stable, reproducible age covariate...")
    
    # Query 1: Get year of birth
    yob_q = f"SELECT person_id, year_of_birth FROM `{cdr_id}.person`"
    yob_df = bq_client.query(yob_q).to_dataframe()
    yob_df['person_id'] = yob_df['person_id'].astype(str)

    # Query 2: Get the year of the last observation for each person
    obs_q = f"""
        SELECT person_id, EXTRACT(YEAR FROM MAX(observation_period_end_date)) AS obs_end_year
        FROM `{cdr_id}.observation_period`
        GROUP BY person_id
    """
    obs_df = bq_client.query(obs_q).to_dataframe()
    obs_df['person_id'] = obs_df['person_id'].astype(str)

    # Merge the two data sources
    demographics = pd.merge(yob_df, obs_df, on='person_id', how='inner')
    
    # Calculate age and age-squared, handling potential data errors gracefully
    demographics['year_of_birth'] = pd.to_numeric(demographics['year_of_birth'], errors='coerce')
    demographics['AGE'] = demographics['obs_end_year'] - demographics['year_of_birth']
    demographics['AGE_sq'] = demographics['AGE'] ** 2
    
    # Set index and select final columns, dropping anyone with missing age info
    final_df = demographics[['person_id', 'AGE', 'AGE_sq']].dropna().set_index('person_id')
    
    print(f"    -> Successfully calculated stable age for {len(final_df):,} participants.")
    return final_df

# --- High-Performance Pipeline Functions ---

def _load_single_pheno_cache(pheno_info, core_index, cdr_codename):
    """THREAD WORKER: Loads one cached phenotype file from disk and returns integer case indices."""
    s_name, category = pheno_info['sanitized_name'], pheno_info['disease_category']
    pheno_cache_path = os.path.join(CACHE_DIR, f"pheno_{s_name}_{cdr_codename}.parquet")
    try:
        ph = pd.read_parquet(pheno_cache_path, columns=['is_case'])
        case_ids = ph.index[ph['is_case'] == 1].astype(str)
        case_idx = core_index.get_indexer(case_ids)
        case_idx = case_idx[case_idx >= 0].astype(np.int32)
        return {"name": s_name, "category": category, "case_idx": case_idx}
    except Exception as e:
        print(f"[CacheLoader] - [FAIL] Failed to load '{s_name}': {e}", flush=True)
        return None


def phenotype_fetcher_worker(pheno_queue, pheno_defs, bq_client, cdr_id, category_to_pan_cases, cdr_codename, core_index):
    """PRODUCER: High-performance, memory-stable data loader that works in chunks without constructing per-phenotype controls."""
    print("[Fetcher]  - Categorizing phenotypes into cached vs. uncached...")
    phenos_to_load_from_cache = [row.to_dict() for _, row in pheno_defs.iterrows() if os.path.exists(os.path.join(CACHE_DIR, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))]
    phenos_to_query_from_bq = [row.to_dict() for _, row in pheno_defs.iterrows() if not os.path.exists(os.path.join(CACHE_DIR, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))]
    print(f"[Fetcher]  - Found {len(phenos_to_load_from_cache)} cached phenotypes to fast-load.")
    print(f"[Fetcher]  - Found {len(phenos_to_query_from_bq)} uncached phenotypes to queue.")

    # ---  STAGE 1 - PACED PARALLEL CACHE LOADING IN CHUNKS ---
    num_chunks = (len(phenos_to_load_from_cache) + LOADER_CHUNK_SIZE - 1) // LOADER_CHUNK_SIZE
    for i in range(0, len(phenos_to_load_from_cache), LOADER_CHUNK_SIZE):
        chunk = phenos_to_load_from_cache[i:i + LOADER_CHUNK_SIZE]
        chunk_num = (i // LOADER_CHUNK_SIZE) + 1
        print(f"[Fetcher]  - Processing chunk {chunk_num} of {num_chunks} ({len(chunk)} phenotypes)...", flush=True)
        with ThreadPoolExecutor(max_workers=LOADER_THREADS) as executor:
            future_to_pheno = {executor.submit(_load_single_pheno_cache, p_info, core_index, cdr_codename): p_info for p_info in chunk}
            for future in as_completed(future_to_pheno):
                result = future.result()
                if result:
                    pheno_queue.put(result)
        print(f"[Mem] RSS after chunk {chunk_num}/{num_chunks}: {rss_gb():.2f} GB", flush=True)
    print("[Fetcher]  - Finished all parallel cache loading chunks.")

    # STAGE 2: SLOW SEQUENTIAL BIGQUERY QUERIES
    for pheno_info in phenos_to_query_from_bq:
        s_name, category, all_codes = pheno_info['sanitized_name'], pheno_info['disease_category'], pheno_info['all_codes']
        print(f"[Fetcher]  - [BQ] Querying '{s_name}'...", flush=True)

        if not all_codes:
            case_idx = np.empty(0, dtype=np.int32)
        else:
            formatted_codes = ','.join([repr(c) for c in all_codes])
            q = f"SELECT DISTINCT person_id FROM `{cdr_id}.condition_occurrence` WHERE condition_source_value IN ({formatted_codes})"
            try:
                df_ids = bq_client.query(q).to_dataframe()
                pids = df_ids["person_id"].astype(str)
                idx = core_index.get_indexer(pids)
                idx = idx[idx >= 0].astype(np.int32)
                case_idx = idx
            except Exception as e:
                print(f"[Fetcher]  - [FAIL] BQ query failed for {s_name}. Error: {str(e)[:150]}", flush=True)
                case_idx = np.empty(0, dtype=np.int32)

        print(f"[Fetcher]  - Caching {len(case_idx):,} new cases for '{s_name}'", flush=True)
        pheno_cache_path = os.path.join(CACHE_DIR, f"pheno_{s_name}_{cdr_codename}.parquet")
        # Cache the full set of case person_ids from BQ (not just the current core intersection)
        pids_for_cache = pd.Index(pids if 'pids' in locals() else [], dtype=str, name='person_id')
        df_to_cache = pd.DataFrame({'is_case': 1}, index=pids_for_cache, dtype=np.int8)
        df_to_cache.to_parquet(pheno_cache_path)


        pheno_queue.put({"name": s_name, "category": category, "case_idx": case_idx})

    pheno_queue.put(None)
    print("[Fetcher]  - All phenotypes fetched. Producer thread finished.")

def _should_skip(meta_path, core_df, case_idx_fp, category, target):
    meta = _read_meta_json(meta_path)
    if not meta:
        return False  # no meta = cannot prove equivalence -> recompute
    same_cols = meta.get("model_columns") == list(core_df.columns)
    same_params = (
        meta.get("num_pcs") == NUM_PCS and
        meta.get("min_cases") == MIN_CASES_FILTER and
        meta.get("min_ctrls") == MIN_CONTROLS_FILTER and
        meta.get("target") == target and
        meta.get("category") == category
    )
    same_core = meta.get("core_index_fp") == _index_fingerprint(core_df.index)
    same_case = meta.get("case_idx_fp") == case_idx_fp
    return all([same_cols, same_params, same_core, same_case])

def run_single_model_worker(pheno_data, target_inversion, results_cache_dir):
    """CONSUMER: Runs a single model. Executed in a separate process using integer indices and precomputed masks."""
    global worker_core_df, allowed_mask_by_cat, N_core
    s_name = pheno_data["name"]
    category = pheno_data["category"]
    case_idx = pheno_data["case_idx"]
    result_path = os.path.join(results_cache_dir, f"{s_name}.json")
    meta_path = result_path + ".meta.json"
    # Order-insensitive fingerprint of the case set based on person_id values to ensure stable caching across runs.
    case_ids_for_fp = worker_core_df.index[case_idx] if case_idx.size > 0 else pd.Index([], name=worker_core_df.index.name)
    case_idx_fp = _index_fingerprint(case_ids_for_fp)
    if os.path.exists(result_path) and _should_skip(meta_path, worker_core_df, case_idx_fp, category, target_inversion):
        return

    try:
        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True

        # Ensure modeling matrix contains only finite values across all covariates to prevent silent NaN/Inf propagation.
        finite_mask_worker = np.isfinite(worker_core_df.to_numpy()).all(axis=1)
        valid_mask = (allowed_mask_by_cat[category] | case_mask) & finite_mask_worker


        n_total = int(valid_mask.sum())
        if n_total == 0:
            print(f"[Worker-{os.getpid()}] - [SKIP] {s_name:<40s} | Reason=no_valid_rows_after_mask", flush=True)
            return

        # Construct response aligned to valid rows.
        y = np.zeros(n_total, dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0:
            y[case_positions] = 1

        # Harden design matrix for numeric stability and emit immediate diagnostics if any non-finite values slip through.
        X_clean = worker_core_df[valid_mask].copy().astype(np.float64, copy=False)
        try:
            _X = X_clean.copy()
            _num_cols = [c for c in _X.columns if pd.api.types.is_numeric_dtype(_X[c])]
            _X = _X[_num_cols].astype(np.float64, copy=False)
            _t = target_inversion
            _tvec = _X[_t].to_numpy()

            # Basic stats for target
            _u = pd.Series(_tvec).nunique(dropna=False)
            _mn = float(np.nanmin(_tvec)) if _u > 0 else float('nan')
            _mx = float(np.nanmax(_tvec)) if _u > 0 else float('nan')
            _sd = float(np.nanstd(_tvec))
        
            # How many unique AGE values? (quadratic degeneracy check)
            _age_unique = _X['AGE'].nunique(dropna=True) if 'AGE' in _X.columns else np.nan
            _age_flag = (_age_unique <= 2) if pd.notna(_age_unique) else False
        
            # Columns equal/affine to target (exact within float tol)
            _eq_cols = []
            for c in _X.columns:
                if c == _t: 
                    continue
                v = _X[c].to_numpy()
                if np.allclose(v, _tvec, equal_nan=True):
                    _eq_cols.append(c)
                else:
                    A = np.vstack([v, np.ones_like(v)]).T
                    try:
                        coef, *_ = np.linalg.lstsq(A, _tvec, rcond=None)
                        resid = _tvec - (coef[0]*v + coef[1])
                        if np.nanmax(np.abs(resid)) < 1e-10:
                            _eq_cols.append(c + "[affine]")
                    except Exception:
                        pass
        
            # R^2 of target explained by other covariates
            _others = [c for c in _X.columns if c not in ['const', _t]]
            _r2 = np.nan
            if len(_others) > 0:
                _A = _X[_others].to_numpy()
                try:
                    coef, *_ = np.linalg.lstsq(_A, _tvec, rcond=None)
                    pred = _A.dot(coef)
                    ss_res = float(np.nansum(( _tvec - pred )**2))
                    ss_tot = float(np.nansum(( _tvec - np.nanmean(_tvec))**2))
                    _r2 = 1.0 - (ss_res/ss_tot) if ss_tot > 0 else np.nan
                except Exception:
                    pass
        
            # Matrix rank & condition number
            try:
                _arr = _X.to_numpy()
                _rank = int(np.linalg.matrix_rank(_arr))
                _s = np.linalg.svd(_arr, compute_uv=False)
                _cond = float((_s[0] / _s[-1]) if (_s[-1] != 0) else np.inf)
            except Exception:
                _rank = -1
                _cond = np.nan
        
            # Zero-variance columns (after valid_mask)
            _zv = [c for c in _X.columns if c not in ['const', _t] and pd.Series(_X[c]).nunique(dropna=False) <= 1]
        
            # Print concise one-line summary
            _age_note = " AGE_DEGENERATE(const,AGE,AGE_sq collinear!)" if _age_flag else ""
            print(f"[DEBUG_COLLINEARITY] {s_name} N={len(_X)} rank={_rank} cond={_cond:.2e} "
                  f"target_unique={_u} min={_mn:.4g} max={_mx:.4g} sd={_sd:.4g} "
                  f"R2(target~others)={_r2:.6f} zero_var={_zv} eq_cols={_eq_cols}{_age_note}",
                  flush=True)
        except Exception as _e:
            print(f"[DEBUG_COLLINEARITY] {s_name} failed: {type(_e).__name__}: {_e}", flush=True)
        
        if not np.isfinite(X_clean.to_numpy()).all():
            bad_cols = [c for c in X_clean.columns if not np.isfinite(X_clean[c].to_numpy()).all()]
            bad_rows_mask = ~np.isfinite(X_clean.to_numpy()).all(axis=1)
            bad_idx_sample = X_clean.index[bad_rows_mask][:10].tolist()
            import traceback, sys
            print(f"[TRACEBACK] Non-finite in worker design for phenotype '{s_name}'", flush=True)
            print(f"[DEBUG] columns={','.join(bad_cols)} sample_rows={bad_idx_sample}", flush=True)
            traceback.print_stack()
            sys.stderr.flush()
        y_clean = pd.Series(y, index=X_clean.index, name='is_case')

        # The following block prints detailed statistics for the predictor and outcome variables
        # immediately before model fitting. This is crucial for debugging systematic model failures,
        # such as observing a standard deviation of zero in the predictor, which would correctly
        # result in a beta coefficient of zero.
        try:
            # Prepare the value counts string for the predictor to avoid printing thousands of lines.
            predictor_counts = X_clean[target_inversion].value_counts(dropna=False)
            if len(predictor_counts) > 10:
                # If there are many unique values, show only the first and last 5 for brevity.
                counts_str = (
                    f"{predictor_counts.head(5).to_string()}\n"
                    f"...\n"
                    f"{predictor_counts.tail(5).to_string()}"
                )
            else:
                # If there are few unique values, show them all.
                counts_str = predictor_counts.to_string()

            print(
                f"\n--- [DEBUG] Pre-fit diagnostics for phenotype '{s_name}' in Worker-{os.getpid()} ---"
                f"\n[DEBUG] Total rows in design matrix: {len(X_clean):,}"
                f"\n[DEBUG] Predictor '{target_inversion}' statistics:\n{X_clean[target_inversion].describe().to_string()}"
                f"\n[DEBUG] Predictor '{target_inversion}' value counts (truncated):\n{counts_str}"
                f"\n[DEBUG] Outcome 'is_case' value counts:\n{y_clean.value_counts().to_string()}"
                "\n--- [DEBUG] End of diagnostics ---\n",
                flush=True
            )
        except Exception as diag_e:
            print(f"[DEBUG] Diagnostic printing failed for '{s_name}': {diag_e}", flush=True)

        n_cases = int(y_clean.sum())
        n_ctrls = int(n_total - n_cases)
        if n_cases < MIN_CASES_FILTER or n_ctrls < MIN_CONTROLS_FILTER:
            # write result+meta so _should_skip() will skip this phenotype next run
            result_data = {
                "Phenotype": s_name,
                "N_Total": n_total,
                "N_Cases": n_cases,
                "N_Controls": n_ctrls,
                "Beta": float('nan'),
                "OR": float('nan'),
                "P_Value": float('nan'),
                "Skip_Reason": "insufficient_cases_or_controls"
            }
            _atomic_write_json(result_path, result_data)
            _atomic_write_json(meta_path, {
                "kind": "phewas_result",
                "s_name": s_name,
                "category": category,
                "model": "Logit",
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": case_idx_fp,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "skip_reason": "insufficient_cases_or_controls"
            })

            print(
                f"[Worker-{os.getpid()}] - [SKIP] {s_name:<40s} | N={n_total:,} Cases={n_cases:,} Ctrls={n_ctrls:,} "
                f"| Reason=insufficient_cases_or_controls thresholds(cases>={MIN_CASES_FILTER}, ctrls>={MIN_CONTROLS_FILTER})",
                flush=True
            )
            return

        # Drop any zero-variance covariates within the valid set, but never drop the intercept or target.
        drop_candidates = [c for c in X_clean.columns if c not in ('const', target_inversion)]
        zero_var_cols = [c for c in drop_candidates if X_clean[c].nunique(dropna=False) <= 1]
        if zero_var_cols:
            X_clean = X_clean.drop(columns=zero_var_cols)

        # If the target inversion dosage is constant in the valid set, persist a NA result and return.
        if X_clean[target_inversion].nunique(dropna=False) <= 1:
            result_data = {
                "Phenotype": s_name,
                "N_Total": n_total,
                "N_Cases": n_cases,
                "N_Controls": n_ctrls,
                "Beta": float('nan'),
                "OR": float('nan'),
                "P_Value": float('nan')
            }
            _atomic_write_json(result_path, result_data)
            _atomic_write_json(meta_path, {
                "kind": "phewas_result",
                "s_name": s_name,
                "category": category,
                "model": "Logit",
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": case_idx_fp,
                "created_at": datetime.utcnow().isoformat() + "Z",
            })
            uvals = X_clean[target_inversion].dropna().unique().tolist()
            print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta=nan | OR=nan | P=nan | CONSTANT_DOSAGE unique_values={uvals}", flush=True)
            return

        # Helper to check convergence across statsmodels fit result variants.
        def _converged(fit_obj):
            try:
                if hasattr(fit_obj, "mle_retvals") and isinstance(fit_obj.mle_retvals, dict):
                    return bool(fit_obj.mle_retvals.get("converged", False))
                if hasattr(fit_obj, "converged"):
                    return bool(fit_obj.converged)
                return False
            except Exception:
                return False

        # Apply automatic stabilization locally in the worker: handle sex separation, attempt standard MLE, then ridge fallback.
        X_work = X_clean
        y_work = y_clean

        # Detect sex-by-case separation and restrict or drop sex when needed.
        model_notes_worker = []
        if 'sex' in X_work.columns:
            try:
                tab = pd.crosstab(X_work['sex'], y_work).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
                valid_sexes = []
                for s in [0.0, 1.0]:
                    if s in tab.index:
                        has_ctrl = bool(tab.loc[s, 0] > 0)
                        has_case = bool(tab.loc[s, 1] > 0)
                        if has_ctrl and has_case:
                            valid_sexes.append(s)
                if len(valid_sexes) == 1:
                    mask = X_work['sex'].isin(valid_sexes)
                    X_work = X_work.loc[mask]
                    y_work = y_work.loc[X_work.index]
                    model_notes_worker.append("sex_restricted")
                elif len(valid_sexes) == 0:
                    X_work = X_work.drop(columns=['sex'])
                    model_notes_worker.append("sex_dropped_for_separation")
            except Exception:
                pass

        fit = None
        fit_reason = ""

        try:
            fit_try = sm.Logit(y_work, X_work).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
            if _converged(fit_try):
                setattr(fit_try, "_model_note", ";".join(model_notes_worker) if model_notes_worker else "")
                setattr(fit_try, "_used_ridge", False)
                fit = fit_try
        except Exception as e:
            import traceback
            print("[TRACEBACK] run_single_model_worker newton failed:", flush=True)
            traceback.print_exc()
            fit_reason = f"newton_exception:{type(e).__name__}:{e}"

        if fit is None:
            try:
                fit_try = sm.Logit(y_work, X_work).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                if _converged(fit_try):
                    setattr(fit_try, "_model_note", ";".join(model_notes_worker) if model_notes_worker else "")
                    setattr(fit_try, "_used_ridge", False)
                    fit = fit_try
                else:
                    fit_reason = "bfgs_not_converged"
            except Exception as e:
                import traceback
                print("[TRACEBACK] run_single_model_worker bfgs failed:", flush=True)
                traceback.print_exc()
                fit_reason = f"bfgs_exception:{type(e).__name__}:{e}"

        if fit is None:
            try:
                p = X_work.shape[1] - (1 if 'const' in X_work.columns else 0)
                n = max(1, X_work.shape[0])
                alpha = max(RIDGE_L2_BASE * (float(p) / float(n)), 1e-6)
                ridge_fit = sm.Logit(y_work, X_work).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=800)
                try:
                    refit = sm.Logit(y_work, X_work).fit(disp=0, method='newton', maxiter=400, tol=1e-8, start_params=ridge_fit.params, warn_convergence=True)
                    if _converged(refit):
                        model_notes_worker.append("ridge_seeded_refit")
                        setattr(refit, "_model_note", ";".join(model_notes_worker))
                        setattr(refit, "_used_ridge", True)
                        fit = refit
                    else:
                        model_notes_worker.append("ridge_only")
                        setattr(ridge_fit, "_model_note", ";".join(model_notes_worker))
                        setattr(ridge_fit, "_used_ridge", True)
                        fit = ridge_fit
                except Exception:
                    model_notes_worker.append("ridge_only")
                    setattr(ridge_fit, "_model_note", ";".join(model_notes_worker))
                    setattr(ridge_fit, "_used_ridge", True)
                    fit = ridge_fit
            except Exception as e:
                fit = None
                fit_reason = f"ridge_exception:{type(e).__name__}:{e}"

        if fit is None or target_inversion not in fit.params:
            result_data = {
                "Phenotype": s_name,
                "N_Total": n_total,
                "N_Cases": n_cases,
                "N_Controls": n_ctrls,
                "Beta": float('nan'),
                "OR": float('nan'),
                "P_Value": float('nan')
            }
            _atomic_write_json(result_path, result_data)
            _atomic_write_json(meta_path, {
                "kind": "phewas_result",
                "s_name": s_name,
                "category": category,
                "model": "Logit",
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": case_idx_fp,
                "created_at": datetime.utcnow().isoformat() + "Z",
            })
            what = f"AUTO_FIT_FAILED:{fit_reason}" if fit is None else "COEFFICIENT_MISSING_IN_FIT_PARAMS"
            print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta=nan | OR=nan | P=nan | {what}", flush=True)
            return

        beta = float(fit.params[target_inversion])
        try:
            pval = float(fit.pvalues[target_inversion])
            pval_reason = ""
        except Exception as e:
            pval = float('nan')
            pval_reason = f"pvalue_unavailable({type(e).__name__})"
        
        # Compute a 95% CI for the overall cohort using the model-based standard error when available.
        se = None
        if hasattr(fit, "bse"):
            try:
                se = float(fit.bse[target_inversion])
            except Exception:
                se = None
        or_ci95_str = None
        if se is not None and np.isfinite(se) and se > 0.0:
            lo = float(np.exp(beta - 1.96 * se))
            hi = float(np.exp(beta + 1.96 * se))
            or_ci95_str = f"{lo:.3f},{hi:.3f}"
        
        suffix = f" | P_CAUSE={pval_reason}" if (not np.isfinite(pval) and pval_reason) else ""
        print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta={beta:+.3f} | OR={np.exp(beta):.3f} | P={pval:.2e}{suffix}", flush=True)

        result_data = {
            "Phenotype": s_name,
            "N_Total": n_total,
            "N_Cases": n_cases,
            "N_Controls": n_ctrls,
            "Beta": beta,
            "OR": float(np.exp(beta)),
            "P_Value": pval,
            "OR_CI95": or_ci95_str
        }
        _atomic_write_json(result_path, result_data)

        _atomic_write_json(meta_path, {
            "kind": "phewas_result",
            "s_name": s_name,
            "category": category,
            "model": "Logit",
            "model_columns": list(worker_core_df.columns),
            "num_pcs": NUM_PCS,
            "min_cases": MIN_CASES_FILTER,
            "min_ctrls": MIN_CONTROLS_FILTER,
            "target": target_inversion,
            "core_index_fp": _index_fingerprint(worker_core_df.index),
            "case_idx_fp": case_idx_fp,
            "created_at": datetime.utcnow().isoformat() + "Z",
        })

    except Exception as e:
        import traceback, sys
        print(f"[Worker-{os.getpid()}] - [FAIL] {s_name:<40s} | Error occurred. Full traceback follows:", flush=True)
        traceback.print_exc()
        sys.stderr.flush()

    finally:
        if 'pheno_data' in locals():
            del pheno_data
        if 'y' in locals():
            del y
        if 'X_clean' in locals():
            del X_clean
        if 'y_clean' in locals():
            del y_clean
        gc.collect()

# Atomic JSON writer used by workers to guarantee on-disk completeness even under interruption.
def _atomic_write_json(path, data_obj):
    """
    Writes JSON atomically by first writing to a unique temp path and then moving it into place.
    Accepts either a dict-like object or a pandas Series.
    """
    tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    try:
        if isinstance(data_obj, pd.Series):
            data_obj.to_json(tmp_path)
        else:
            pd.Series(data_obj).to_json(tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# Lightweight LRT meta equivalence check. This keeps invariants small and stable.
def _lrt_meta_should_skip(meta_path, core_df_cols, core_index_fp, case_idx_fp, category, target):
    meta = _read_meta_json(meta_path)
    if not meta:
        return False
    same = (
        meta.get("model_columns") == list(core_df_cols) and
        meta.get("num_pcs") == NUM_PCS and
        meta.get("min_cases") == MIN_CASES_FILTER and
        meta.get("min_ctrls") == MIN_CONTROLS_FILTER and
        meta.get("target") == target and
        meta.get("category") == category and
        meta.get("core_index_fp") == core_index_fp and
        meta.get("case_idx_fp") == case_idx_fp
    )
    return bool(same)

# Dedicated initializer for LRT pools that also provides ancestry labels to workers for Stage-2.
worker_anc_series = None
def init_lrt_worker(df_to_share, masks, anc_series):
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"^overflow encountered in exp",
        category=RuntimeWarning,
        module=r"^statsmodels\.discrete\.discrete_model$",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"^divide by zero encountered in log",
        category=RuntimeWarning,
        module=r"^statsmodels\.discrete\.discrete_model$",
    )
    global worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series
    worker_core_df = df_to_share
    allowed_mask_by_cat = masks
    N_core = len(df_to_share)
    # Align and normalize ancestry once per worker to avoid repeated per-task work.
    worker_anc_series = anc_series.reindex(df_to_share.index).str.lower()
    print(f"[Worker-{os.getpid()}] Initialized and received shared core dataframe, masks, and ancestry.", flush=True)

def lrt_overall_worker(task):
    """
    Worker for Stage-1 overall LRT. Reads case set from per-phenotype cache, builds reduced and full models
    on the same rows, computes df via rank difference, and writes result + meta atomically.
    """
    try:
        s_name = task["name"]
        category = task["category"]
        cdr_codename = task["cdr_codename"]
        target_inversion = task["target"]
        result_path = os.path.join(LRT_OVERALL_CACHE_DIR, f"{s_name}.json")
        meta_path = result_path + ".meta.json"

        # Case fingerprint based on person_id values for resume-safe equivalence.
        pheno_cache_path = os.path.join(CACHE_DIR, f"pheno_{s_name}_{cdr_codename}.parquet")
        if not os.path.exists(pheno_cache_path):
            _atomic_write_json(result_path, {
                "Phenotype": s_name,
                "P_LRT_Overall": float('nan'),
                "LRT_df_Overall": float('nan'),
                "LRT_Overall_Reason": "missing_case_cache"
            })
            _atomic_write_json(meta_path, {
                "kind": "lrt_overall",
                "s_name": s_name,
                "category": category,
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": "",
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=missing_case_cache", flush=True)
            return

        ph = pd.read_parquet(pheno_cache_path, columns=['is_case'])
        case_ids = ph.index[ph['is_case'] == 1].astype(str)
        core_index = worker_core_df.index
        case_idx = core_index.get_indexer(case_ids)
        case_idx = case_idx[case_idx >= 0].astype(np.int32)
        case_ids_for_fp = core_index[case_idx] if case_idx.size > 0 else pd.Index([], name=core_index.name)
        case_fp = _index_fingerprint(case_ids_for_fp)

        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path,
            worker_core_df.columns,
            _index_fingerprint(core_index),
            case_fp,
            category,
            target_inversion
        ):
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} CACHE_HIT", flush=True)
            return

        # Build masks and design
        finite_mask = np.isfinite(worker_core_df.to_numpy()).all(axis=1)
        allowed_mask = allowed_mask_by_cat.get(category, np.ones(len(core_index), dtype=bool))
        case_mask = np.zeros(len(core_index), dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask
        n_valid = int(valid_mask.sum())
        y = np.zeros(n_valid, dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0:
            y[case_positions] = 1

        # Assemble base X and enforce float64
        pc_cols_local = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
        base_cols = ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE']
        X_base = worker_core_df.loc[valid_mask, base_cols].astype(np.float64, copy=False)
        y_series = pd.Series(y, index=X_base.index, name='is_case')

        # Sex handling mirrored across reduced and full
        if 'sex' in X_base.columns:
            try:
                tab = pd.crosstab(X_base['sex'], y_series).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
                valid_sexes = []
                for s in [0.0, 1.0]:
                    if s in tab.index:
                        if bool(tab.loc[s, 0] > 0) and bool(tab.loc[s, 1] > 0):
                            valid_sexes.append(s)
                if len(valid_sexes) == 1:
                    keep = X_base['sex'].isin(valid_sexes)
                    X_base = X_base.loc[keep]
                    y_series = y_series.loc[X_base.index]
                elif len(valid_sexes) == 0:
                    X_base = X_base.drop(columns=['sex'])
            except Exception:
                pass

        # Drop zero-variance columns except const and target
        zvars = [c for c in X_base.columns if c not in ['const', target_inversion] and pd.Series(X_base[c]).nunique(dropna=False) <= 1]
        if len(zvars) > 0:
            X_base = X_base.drop(columns=zvars)

        n_cases = int(y_series.sum())
        n_ctrls = int(len(y_series) - n_cases)
        if n_cases < MIN_CASES_FILTER or n_ctrls < MIN_CONTROLS_FILTER:
            _atomic_write_json(result_path, {
                "Phenotype": s_name,
                "P_LRT_Overall": float('nan'),
                "LRT_df_Overall": float('nan'),
                "LRT_Overall_Reason": "insufficient_counts"
            })
            _atomic_write_json(meta_path, {
                "kind": "lrt_overall",
                "s_name": s_name,
                "category": category,
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(core_index),
                "case_idx_fp": case_fp,
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=insufficient_counts", flush=True)
            return

        # Require variability in target
        if target_inversion not in X_base.columns or pd.Series(X_base[target_inversion]).nunique(dropna=False) <= 1:
            _atomic_write_json(result_path, {
                "Phenotype": s_name,
                "P_LRT_Overall": float('nan'),
                "LRT_df_Overall": float('nan'),
                "LRT_Overall_Reason": "target_constant"
            })
            _atomic_write_json(meta_path, {
                "kind": "lrt_overall",
                "s_name": s_name,
                "category": category,
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(core_index),
                "case_idx_fp": case_fp,
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=target_constant", flush=True)
            return

        # Fit reduced and full using a hardened local helper to keep parity with main.
        def _design_rank(X):
            try:
                return int(np.linalg.matrix_rank(X.values))
            except Exception:
                return int(np.linalg.matrix_rank(np.asarray(X)))

        def _fit_logit_hardened(X, y_in, require_target):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X = X.astype(np.float64, copy=False)
            y_arr = np.asarray(y_in, dtype=np.float64).reshape(-1)
            if require_target:
                if target_inversion not in X.columns or pd.Series(X[target_inversion]).nunique(dropna=False) <= 1:
                    return None, "target_constant"
            try:
                fit_try = sm.Logit(y_arr, X).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
                return fit_try, ""
            except Exception:
                pass
            try:
                fit_try = sm.Logit(y_arr, X).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                return fit_try, ""
            except Exception:
                pass
            try:
                p = X.shape[1] - (1 if 'const' in X.columns else 0)
                n = max(1, X.shape[0])
                alpha = max(RIDGE_L2_BASE * (float(p) / float(n)), 1e-6)
                ridge_fit = sm.Logit(y_arr, X).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=800)
                return ridge_fit, ""
            except Exception as e:
                return None, f"fit_exception:{type(e).__name__}"

        X_full = X_base.copy()
        X_red = X_base.drop(columns=[target_inversion])

        fit_red, r_reason = _fit_logit_hardened(X_red, y_series, require_target=False)
        fit_full, f_reason = _fit_logit_hardened(X_full, y_series, require_target=True)

        out = {
            "Phenotype": s_name,
            "P_LRT_Overall": float('nan'),
            "LRT_df_Overall": float('nan'),
            "LRT_Overall_Reason": ""
        }

        if (fit_red is not None) and (fit_full is not None) and hasattr(fit_full, "llf") and hasattr(fit_red, "llf") and (fit_full.llf >= fit_red.llf):
            rank_red = _design_rank(X_red)
            rank_full = _design_rank(X_full)
            df_lrt = int(max(0, rank_full - rank_red))
            if df_lrt > 0:
                from scipy import stats as _stats
                llr = 2.0 * (fit_full.llf - fit_red.llf)
                out["P_LRT_Overall"] = float(_stats.chi2.sf(llr, df_lrt))
                out["LRT_df_Overall"] = df_lrt
                out["LRT_Overall_Reason"] = ""
                print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} df={df_lrt} p={out['P_LRT_Overall']:.3e}", flush=True)
            else:
                out["LRT_df_Overall"] = 0
                out["LRT_Overall_Reason"] = "no_df"
                print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=no_df", flush=True)
        else:
            reasons = []
            if fit_red is None:
                reasons.append(f"reduced_model_failed:{r_reason}")
            if fit_full is None:
                reasons.append(f"full_model_failed:{f_reason}")
            if (fit_red is not None) and (fit_full is not None) and (fit_full.llf < fit_red.llf):
                reasons.append("full_llf_below_reduced_llf")
            out["LRT_Overall_Reason"] = ";".join(reasons) if reasons else "fit_failed"
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason={out['LRT_Overall_Reason']}", flush=True)

        _atomic_write_json(result_path, out)
        _atomic_write_json(meta_path, {
            "kind": "lrt_overall",
            "s_name": s_name,
            "category": category,
            "model_columns": list(worker_core_df.columns),
            "num_pcs": NUM_PCS,
            "min_cases": MIN_CASES_FILTER,
            "min_ctrls": MIN_CONTROLS_FILTER,
            "target": target_inversion,
            "core_index_fp": _index_fingerprint(core_index),
            "case_idx_fp": case_fp,
            "created_at": datetime.utcnow().isoformat() + "Z"
        })
    except Exception:
        import traceback, sys
        print(f"[LRT-Stage1-Worker-{os.getpid()}] {task.get('name','?')} FAILED with exception, writing error stub", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        s_name = task.get("name", "unknown")
        _atomic_write_json(os.path.join(LRT_OVERALL_CACHE_DIR, f"{s_name}.json"), {
            "Phenotype": s_name,
            "P_LRT_Overall": float('nan'),
            "LRT_df_Overall": float('nan'),
            "LRT_Overall_Reason": "exception"
        })

def lrt_followup_worker(task):
    """
    Worker for Stage-2 ancestry×dosage LRT and per-ancestry splits for selected phenotypes.
    Caches one JSON per phenotype in the follow-up cache directory.
    """
    try:
        s_name = task["name"]
        category = task["category"]
        cdr_codename = task["cdr_codename"]
        target_inversion = task["target"]
        result_path = os.path.join(LRT_FOLLOWUP_CACHE_DIR, f"{s_name}.json")
        meta_path = result_path + ".meta.json"

        # Case indices
        pheno_cache_path = os.path.join(CACHE_DIR, f"pheno_{s_name}_{cdr_codename}.parquet")
        if not os.path.exists(pheno_cache_path):
            _atomic_write_json(result_path, {
                'Phenotype': s_name,
                'P_LRT_AncestryxDosage': float('nan'),
                'LRT_df': float('nan'),
                'LRT_Ancestry_Levels': "",
                'LRT_Reason': "missing_case_cache"
            })
            _atomic_write_json(meta_path, {
                "kind": "lrt_followup",
                "s_name": s_name,
                "category": category,
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": "",
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} SKIP reason=missing_case_cache", flush=True)
            return

        ph = pd.read_parquet(pheno_cache_path, columns=['is_case'])
        case_ids = ph.index[ph['is_case'] == 1].astype(str)
        core_index = worker_core_df.index
        case_idx = core_index.get_indexer(case_ids)
        case_idx = case_idx[case_idx >= 0].astype(np.int32)
        case_ids_for_fp = core_index[case_idx] if case_idx.size > 0 else pd.Index([], name=core_index.name)
        case_fp = _index_fingerprint(case_ids_for_fp)

        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path,
            worker_core_df.columns,
            _index_fingerprint(core_index),
            case_fp,
            category,
            target_inversion
        ):
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} CACHE_HIT", flush=True)
            return

        # Masks and base X/Y
        finite_mask = np.isfinite(worker_core_df.to_numpy()).all(axis=1)
        allowed_mask = allowed_mask_by_cat.get(category, np.ones(len(core_index), dtype=bool))
        case_mask = np.zeros(len(core_index), dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask
        if int(valid_mask.sum()) == 0:
            _atomic_write_json(result_path, {
                'Phenotype': s_name,
                'P_LRT_AncestryxDosage': float('nan'),
                'LRT_df': float('nan'),
                'LRT_Ancestry_Levels': "",
                'LRT_Reason': "no_valid_rows_after_mask"
            })
            _atomic_write_json(meta_path, {
                "kind": "lrt_followup",
                "s_name": s_name,
                "category": category,
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(core_index),
                "case_idx_fp": case_fp,
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} SKIP reason=no_valid_rows_after_mask", flush=True)
            return

        pc_cols_local = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
        base_cols = ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE']
        X_base = worker_core_df.loc[valid_mask, base_cols].astype(np.float64, copy=False)
        y = np.zeros(X_base.shape[0], dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0:
            y[case_positions] = 1
        y_series = pd.Series(y, index=X_base.index, name='is_case')

        anc_vec = worker_anc_series.loc[X_base.index]
        anc_levels_local = anc_vec.dropna().unique().tolist()
        if 'eur' in anc_levels_local:
            anc_levels_local = ['eur'] + [a for a in anc_levels_local if a != 'eur']

        out = {
            'Phenotype': s_name,
            'P_LRT_AncestryxDosage': float('nan'),
            'LRT_df': float('nan'),
            'LRT_Ancestry_Levels': ",".join(anc_levels_local),
            'LRT_Reason': ""
        }

        if len(anc_levels_local) < 2:
            out['LRT_Reason'] = "only_one_ancestry_level"
            _atomic_write_json(result_path, out)
            _atomic_write_json(meta_path, {
                "kind": "lrt_followup",
                "s_name": s_name,
                "category": category,
                "model_columns": list(worker_core_df.columns),
                "num_pcs": NUM_PCS,
                "min_cases": MIN_CASES_FILTER,
                "min_ctrls": MIN_CONTROLS_FILTER,
                "target": target_inversion,
                "core_index_fp": _index_fingerprint(core_index),
                "case_idx_fp": case_fp,
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} SKIP reason=only_one_ancestry_level", flush=True)
            return

        keep = anc_vec.notna()
        X_base = X_base.loc[keep]
        y_series = y_series.loc[keep]
        anc_keep = anc_vec.loc[keep]
        anc_keep = pd.Series(pd.Categorical(anc_keep, categories=anc_levels_local, ordered=False), index=anc_keep.index, name='ANCESTRY')
        A = pd.get_dummies(anc_keep, prefix='ANC', drop_first=True, dtype=np.float64)
        X_red = pd.concat([X_base, A], axis=1, join='inner').astype(np.float64, copy=False)
        X_full = X_red.copy()
        for c in A.columns:
            X_full[f"{target_inversion}:{c}"] = X_full[target_inversion] * X_full[c]

        # Fit both designs (hardened)
        def _fit_logit(X, y_in, require_target):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X = X.astype(np.float64, copy=False)
            y_arr = np.asarray(y_in, dtype=np.float64).reshape(-1)
            if require_target:
                if target_inversion not in X.columns or pd.Series(X[target_inversion]).nunique(dropna=False) <= 1:
                    return None, "target_constant"
            try:
                fit_try = sm.Logit(y_arr, X).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
                return fit_try, ""
            except Exception:
                pass
            try:
                fit_try = sm.Logit(y_arr, X).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                return fit_try, ""
            except Exception:
                pass
            try:
                p = X.shape[1] - (1 if 'const' in X.columns else 0)
                n = max(1, X.shape[0])
                alpha = max(RIDGE_L2_BASE * (float(p) / float(n)), 1e-6)
                ridge_fit = sm.Logit(y_arr, X).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=800)
                return ridge_fit, ""
            except Exception as e:
                return None, f"fit_exception:{type(e).__name__}"

        fit_red, rr = _fit_logit(X_red, y_series, require_target=False)
        fit_full, fr = _fit_logit(X_full, y_series, require_target=True)

        def _rank(X):
            try:
                return int(np.linalg.matrix_rank(X.values))
            except Exception:
                return int(np.linalg.matrix_rank(np.asarray(X)))

        if (fit_red is not None) and (fit_full is not None) and hasattr(fit_full, "llf") and hasattr(fit_red, "llf") and (fit_full.llf >= fit_red.llf):
            from scipy import stats as _stats
            df_lrt = int(max(0, _rank(X_full) - _rank(X_red)))
            if df_lrt > 0:
                llr = 2.0 * (fit_full.llf - fit_red.llf)
                out['P_LRT_AncestryxDosage'] = float(_stats.chi2.sf(llr, df_lrt))
                out['LRT_df'] = df_lrt
                out['LRT_Reason'] = ""
                print(f"[Ancestry-Worker-{os.getpid()}] {s_name} df={df_lrt} p={out['P_LRT_AncestryxDosage']:.3e}", flush=True)
            else:
                out['LRT_Reason'] = "no_interaction_df"
        else:
            reasons = []
            if fit_red is None:
                reasons.append(f"reduced_model_failed:{rr}")
            if fit_full is None:
                reasons.append(f"full_model_failed:{fr}")
            if (fit_red is not None) and (fit_full is not None) and (fit_full.llf < fit_red.llf):
                reasons.append("full_llf_below_reduced_llf")
            out['LRT_Reason'] = ";".join(reasons) if reasons else "fit_failed"

        # Per-ancestry simple splits (counts always included; OR/P only when stratum large enough)
        for anc in anc_levels_local:
            group_mask = valid_mask & worker_anc_series.eq(anc).to_numpy()
            if not group_mask.any():
                out[f"{anc.upper()}_N"] = 0
                out[f"{anc.upper()}_N_Cases"] = 0
                out[f"{anc.upper()}_N_Controls"] = 0
                out[f"{anc.upper()}_OR"] = float('nan')
                out[f"{anc.upper()}_CI95"] = float('nan')
                out[f"{anc.upper()}_P"] = float('nan')
                out[f"{anc.upper()}_REASON"] = "no_rows_in_group"
                continue
            X_g = worker_core_df.loc[group_mask, ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE']].astype(np.float64, copy=False)
            y_g = np.zeros(X_g.shape[0], dtype=np.int8)
            case_positions_g = np.nonzero(case_mask[group_mask])[0]
            if case_positions_g.size > 0:
                y_g[case_positions_g] = 1
            n_cases_g = int(y_g.sum())
            n_tot_g = int(len(y_g))
            n_ctrl_g = n_tot_g - n_cases_g
            out[f"{anc.upper()}_N"] = n_tot_g
            out[f"{anc.upper()}_N_Cases"] = n_cases_g
            out[f"{anc.upper()}_N_Controls"] = n_ctrl_g
            if (n_cases_g < PER_ANC_MIN_CASES) or (n_ctrl_g < PER_ANC_MIN_CONTROLS):
                out[f"{anc.upper()}_OR"] = float('nan')
                out[f"{anc.upper()}_CI95"] = float('nan')
                out[f"{anc.upper()}_P"] = float('nan')
                out[f"{anc.upper()}_REASON"] = "insufficient_stratum_counts"
                continue
            # Fit per-ancestry effect
            try:
                fit_g = sm.Logit(y_g, X_g).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
            except Exception:
                try:
                    fit_g = sm.Logit(y_g, X_g).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                except Exception:
                    fit_g = None
            if (fit_g is None) or (target_inversion not in getattr(fit_g, "params", {})):
                out[f"{anc.upper()}_OR"] = float('nan')
                out[f"{anc.upper()}_CI95"] = float('nan')
                out[f"{anc.upper()}_P"] = float('nan')
                out[f"{anc.upper()}_REASON"] = "subset_fit_failed"
                continue
            beta = float(fit_g.params[target_inversion])
            or_val = float(np.exp(beta))
            if hasattr(fit_g, "bse"):
                try:
                    se = float(fit_g.bse[target_inversion])
                    lo = float(np.exp(beta - 1.96 * se))
                    hi = float(np.exp(beta + 1.96 * se))
                    out[f"{anc.upper()}_CI95"] = f"{lo:.3f},{hi:.3f}"
                except Exception:
                    out[f"{anc.upper()}_CI95"] = float('nan')
            else:
                out[f"{anc.upper()}_CI95"] = float('nan')
            out[f"{anc.upper()}_OR"] = or_val
            try:
                out[f"{anc.upper()}_P"] = float(fit_g.pvalues[target_inversion])
            except Exception:
                out[f"{anc.upper()}_P"] = float('nan')
            out[f"{anc.upper()}_REASON"] = ""

        _atomic_write_json(result_path, out)
        _atomic_write_json(meta_path, {
            "kind": "lrt_followup",
            "s_name": s_name,
            "category": category,
            "model_columns": list(worker_core_df.columns),
            "num_pcs": NUM_PCS,
            "min_cases": MIN_CASES_FILTER,
            "min_ctrls": MIN_CONTROLS_FILTER,
            "target": target_inversion,
            "core_index_fp": _index_fingerprint(core_index),
            "case_idx_fp": case_fp,
            "created_at": datetime.utcnow().isoformat() + "Z"
        })
    except Exception:
        import traceback, sys
        print(f"[Ancestry-Worker-{os.getpid()}] {task.get('name','?')} FAILED with exception, writing error stub", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        s_name = task.get("name", "unknown")
        _atomic_write_json(os.path.join(LRT_FOLLOWUP_CACHE_DIR, f"{s_name}.json"), {
            'Phenotype': s_name,
            'P_LRT_AncestryxDosage': float('nan'),
            'LRT_df': float('nan'),
            'LRT_Ancestry_Levels': "",
            'LRT_Reason': "exception"
        })

def main():
    script_start_time = time.time()
    print("=" * 70)
    print(" Starting Robust, Memory-Stable PheWAS Pipeline (Chunked Producer)")
    print("=" * 70)

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)

    try:
        with Timer() as t_setup:
            print("\n--- PART 1: SETUP & SHARED DATA LOADING ---")
            print("[Setup]    - Loading phenotype definitions...")
            pheno_defs_df = pd.read_csv(PHENOTYPE_DEFINITIONS_URL, sep="\t")
            pheno_defs_df["sanitized_name"] = pheno_defs_df["disease"].apply(sanitize_name)
            pheno_defs_df["all_codes"] = pheno_defs_df.apply(
                lambda row: parse_icd_codes(row["icd9_codes"]).union(parse_icd_codes(row["icd10_codes"])),
                axis=1,
            )

            print("[Setup]    - Setting up BigQuery client...")
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            cdr_codename = cdr_dataset_id.split(".")[-1]

            print("[Setup]    - Loading shared covariates (Demographics, Inversions, PCs, Sex)...")
            demographics_df = get_cached_or_generate(
                os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"),
                _load_demographics_with_stable_age,
                bq_client=bq_client,
                cdr_id=cdr_dataset_id,
            )
            inversion_df = get_cached_or_generate(
                os.path.join(CACHE_DIR, f"inversion_{TARGET_INVERSION}.parquet"), _load_inversions
            )
            pc_df = get_cached_or_generate(
                os.path.join(CACHE_DIR, "pcs_10.parquet"), _load_pcs, gcp_project=gcp_project
            )
            sex_df = get_cached_or_generate(
                os.path.join(CACHE_DIR, "genetic_sex.parquet"), _load_genetic_sex, gcp_project=gcp_project
            )
            ancestry_labels_df = get_cached_or_generate(
                os.path.join(CACHE_DIR, "ancestry_labels.parquet"), _load_ancestry_labels, gcp_project=gcp_project
            )

            # Load related individuals to remove.
            related_ids_to_remove = _load_related_to_remove(gcp_project=gcp_project)

            print("[Setup]    - Standardizing covariate indexes for robust joining...")
            demographics_df.index = demographics_df.index.astype(str)
            inversion_df.index = inversion_df.index.astype(str)
            pc_df.index = pc_df.index.astype(str)
            sex_df.index = sex_df.index.astype(str)
            ancestry_labels_df.index = ancestry_labels_df.index.astype(str)

            pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
            # covariate_cols = [TARGET_INVERSION] + ["sex"] + pc_cols + ["AGE", "AGE_sq"]
            covariate_cols = [TARGET_INVERSION] + ["sex"] + pc_cols + ["AGE"]

            core_df = (
                demographics_df.join(inversion_df, how="inner")
                .join(pc_df, how="inner")
                .join(sex_df, how="inner")
            )

            print(f"[Setup]    - Pre-filter cohort size: {len(core_df):,}")
            core_df = core_df[~core_df.index.isin(related_ids_to_remove)]
            print(f"[Setup]    - Post-filter unrelated cohort size: {len(core_df):,}")

            core_df = core_df[covariate_cols]
            core_df_with_const = sm.add_constant(core_df, prepend=True)

            print("\n--- [DIAGNOSTIC] Testing matrix condition number ---")
            try:
                cols = ['const', 'sex', 'AGE', TARGET_INVERSION] + [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                mat = core_df_with_const[cols].dropna().to_numpy()
                cond = np.linalg.cond(mat)
                print(f"[DIAGNOSTIC] Condition number (current model cols): {cond:,.2f}")
            except Exception as e:
                print(f"[DIAGNOSTIC] Could not compute condition number. Error: {e}")
            print("--- [DIAGNOSTIC] End of test ---\n")

            del core_df, demographics_df, inversion_df, pc_df
            gc.collect()
            print(f"[Setup]    - Core covariate DataFrame ready. Shape: {core_df_with_const.shape}")
            
            if core_df_with_const.shape[0] == 0:
                raise RuntimeError(
                    "FATAL: Core covariate DataFrame has 0 rows after join. Check input data alignment."
                )

            core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")

            # Use a non-finite-aware mask to guarantee rows are fully numeric and finite across all covariates.
            global_notnull_mask = np.isfinite(core_df_with_const.to_numpy()).all(axis=1)

            print(f"[Mem] RSS after core covariates assembly: {rss_gb():.2f} GB")

            print("[Setup]    - Pre-calculating pan-category case sets...")
            category_cache_path = os.path.join(CACHE_DIR, f"pan_category_cases_{cdr_codename}.pkl")
            if os.path.exists(category_cache_path):
                category_to_pan_cases = pd.read_pickle(category_cache_path)
            else:
                category_to_pan_cases = {}
                for category, group in pheno_defs_df.groupby("disease_category"):
                    pan_codes = set.union(*group["all_codes"])
                    if pan_codes:
                        q = (
                            f"SELECT DISTINCT person_id FROM `{cdr_dataset_id}.condition_occurrence` "
                            f"WHERE condition_source_value IN ({','.join([repr(c) for c in pan_codes])})"
                        )
                        category_to_pan_cases[category] = set(
                            bq_client.query(q).to_dataframe()["person_id"].astype(str)
                        )
                    else:
                        category_to_pan_cases[category] = set()
                pd.to_pickle(category_to_pan_cases, category_cache_path)

            print("[Setup]    - Building allowed-control masks per category without constructing per-phenotype controls...")
            allowed_mask_by_cat = {}
            n_core = len(core_index)
            for category, pan_cases in category_to_pan_cases.items():
                pan_idx = core_index.get_indexer(list(pan_cases))
                pan_idx = pan_idx[pan_idx >= 0]
                mask = np.ones(n_core, dtype=bool)
                if pan_idx.size > 0:
                    mask[pan_idx] = False
                mask &= global_notnull_mask
                allowed_mask_by_cat[category] = mask
            print(f"[Mem] RSS after allowed-mask preprocessing: {rss_gb():.2f} GB")

        print(f"\n--- Total Setup Time: {t_setup.duration:.2f}s ---")

        # --- PART 2: RUNNING THE PIPELINE ---
        pheno_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        fetcher_thread = threading.Thread(
            target=phenotype_fetcher_worker,
            args=(
                pheno_queue,
                pheno_defs_df,
                bq_client,
                cdr_dataset_id,
                category_to_pan_cases,
                cdr_codename,
                core_index,
            ),
        )
        fetcher_thread.start()

        worker_func = partial(
            run_single_model_worker, target_inversion=TARGET_INVERSION, results_cache_dir=RESULTS_CACHE_DIR
        )

        print(f"\n--- Starting parallel model fitting with {cpu_count()} worker processes ---")
        with Pool(
            processes=max(1, min(cpu_count(), 8)),
            initializer=init_worker,
            initargs=(core_df_with_const, allowed_mask_by_cat),
            maxtasksperchild=50,
        ) as pool:
            bar_len = 40
            queued = 0
            done = 0
            lock = threading.Lock()
        
            def _print_bar(q, d):
                q = int(q); d = int(d)
                pct = int((d * 100) / q) if q else 0
                filled = int(bar_len * (d / q)) if q else 0
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                print(f"\r[Fit] {bar} {d}/{q} ({pct}%)", end="", flush=True)
        
            def _cb(_):
                nonlocal done, queued
                with lock:
                    done += 1
                    _print_bar(queued, done)
        
            # Drain queue → submit jobs with a completion callback
            while True:
                pheno_data = pheno_queue.get()
                if pheno_data is None:
                    break
                queued += 1
                pool.apply_async(worker_func, (pheno_data,), callback=_cb)
                _print_bar(queued, done)  # show progress while queuing too
        
            pool.close()
            pool.join()  # callbacks keep updating the bar while we wait
            _print_bar(queued, done)  # ensure 100% line
            print("")  # newline after the bar


        fetcher_thread.join()
        print("\n--- All models finished. ---")

        # --- PART 3: CONSOLIDATE & ANALYZE RESULTS ---
        print("\n--- Consolidating results from atomic files ---")
        all_results_from_disk = []
        result_files = [f for f in os.listdir(RESULTS_CACHE_DIR) if f.endswith(".json") and not f.endswith(".meta.json")]
        total_files = len(result_files)
        bar_len = 30
        for i, filename in enumerate(result_files, start=1):
            try:
                result = pd.read_json(os.path.join(RESULTS_CACHE_DIR, filename), typ="series")
                all_results_from_disk.append(result.to_dict())
            except Exception as e:
                print(f"Warning: Could not read corrupted result file: {filename}, Error: {e}")
            filled = int(bar_len * i / total_files) if total_files > 0 else bar_len
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            pct = int(i * 100 / total_files) if total_files > 0 else 100
            print(f"\r[Consolidate] {bar} {i}/{total_files} ({pct}%)", end="", flush=True)
        if total_files > 0:
            print("")

        if not all_results_from_disk:
            print("No results found to process.")
        else:
            # Ensure LRT cache directories exist
            os.makedirs(LRT_OVERALL_CACHE_DIR, exist_ok=True)
            os.makedirs(LRT_FOLLOWUP_CACHE_DIR, exist_ok=True)

            results_df = pd.DataFrame(all_results_from_disk)
            print(f"Successfully consolidated {len(results_df)} results.")

            # Stage-1 FDR is computed later using the overall LRT; initialize the working results frame.
            df = results_df.copy()

            # === Ancestry follow-up on FDR-significant phenotypes ===
            from scipy import stats

            # Compute overall-cohort 95% CI for OR in the consolidated results using cached Beta and P_Value when needed.
            if "OR_CI95" not in df.columns:
                df["OR_CI95"] = np.nan

            def _compute_overall_or_ci(beta_val, p_val):
                """
                Returns a formatted 'lo,hi' string for the overall OR 95% CI using a Wald approximation
                derived from the cached Beta and two-sided P_Value. Falls back to NaN when unavailable.
                """
                if pd.isna(beta_val) or pd.isna(p_val):
                    return np.nan
                try:
                    b = float(beta_val)
                    p = float(p_val)
                    if not np.isfinite(b) or not np.isfinite(p):
                        return np.nan
                    if not (0.0 < p < 1.0):
                        return np.nan
                    z = float(stats.norm.ppf(1.0 - p / 2.0))
                    if not np.isfinite(z) or z == 0.0:
                        return np.nan
                    se = abs(b) / z
                    lo = float(np.exp(b - 1.96 * se))
                    hi = float(np.exp(b + 1.96 * se))
                    return f"{lo:.3f},{hi:.3f}"
                except Exception:
                    return np.nan

            missing_ci_mask = df["OR_CI95"].isna() | (df["OR_CI95"].astype(str) == "") | (df["OR_CI95"].astype(str).str.lower() == "nan")
            df.loc[missing_ci_mask, "OR_CI95"] = df.loc[missing_ci_mask, ["Beta", "P_Value"]].apply(lambda r: _compute_overall_or_ci(r["Beta"], r["P_Value"]), axis=1)

            # Align ancestry labels to the core covariate index used in modeling.
            anc_series = ancestry_labels_df.reindex(core_df_with_const.index)["ANCESTRY"].str.lower()
            # Global ancestry alignment diagnostics to surface upstream issues before per-phenotype tests.
            total_core = int(len(core_df_with_const.index))
            known_anc = int(anc_series.notna().sum())
            missing_anc = total_core - known_anc
            core_dup = int(core_df_with_const.index.duplicated(keep=False).sum())
            core_idx_dtype = str(core_df_with_const.index.dtype)
            anc_levels_global = ",".join(sorted(pd.Series(anc_series.dropna().unique()).astype(str)))
            print(f"[DEBUG] ancestry_align total_core={total_core} known={known_anc} missing={missing_anc} core_index_dtype={core_idx_dtype} core_index_dup_count={core_dup} levels={anc_levels_global}", flush=True)

            # ---- PARALLEL STAGE-1 OVERALL LRT WITH PER-PHENOTYPE ATOMIC CACHING ----
            name_to_cat = pheno_defs_df.set_index('sanitized_name')['disease_category'].to_dict()
            phenos_list = df["Phenotype"].astype(str).tolist()
            tasks = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": TARGET_INVERSION} for s in phenos_list]

            print(f"[LRT-Stage1] Scheduling {len(tasks)} phenotypes for overall LRT with atomic caching.", flush=True)
            bar_len = 40
            queued = 0
            done = 0
            lock = threading.Lock()

            def _print_bar(q, d, label):
                q = int(q); d = int(d)
                pct = int((d * 100) / q) if q else 0
                filled = int(bar_len * (d / q)) if q else 0
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                print(f"\r[{label}] {bar} {d}/{q} ({pct}%)", end="", flush=True)

            with Pool(
                processes=max(1, min(cpu_count(), 8)),
                initializer=init_worker,
                initargs=(core_df_with_const, allowed_mask_by_cat),
                maxtasksperchild=50,
            ) as pool:

                def _cb(_):
                    nonlocal done, queued
                    with lock:
                        done += 1
                        _print_bar(queued, done, "LRT-Stage1")

                for task in tasks:
                    queued += 1
                    pool.apply_async(lrt_overall_worker, (task,), callback=_cb)
                    _print_bar(queued, done, "LRT-Stage1")
                pool.close()
                pool.join()
                _print_bar(queued, done, "LRT-Stage1")
                print("")

            # Read cached overall LRT results
            overall_records = []
            files_overall = [f for f in os.listdir(LRT_OVERALL_CACHE_DIR) if f.endswith(".json") and not f.endswith(".meta.json")]
            total_ov = len(files_overall)
            bar_len = 30
            for i, filename in enumerate(files_overall, start=1):
                try:
                    rec = pd.read_json(os.path.join(LRT_OVERALL_CACHE_DIR, filename), typ="series")
                    overall_records.append(rec.to_dict())
                except Exception as e:
                    print(f"Warning: Could not read LRT overall file: {filename}, Error: {e}")
                filled = int(bar_len * i / total_ov) if total_ov > 0 else bar_len
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                pct = int(i * 100 / total_ov) if total_ov > 0 else 100
                print(f"\r[LRT-Stage1-Collect] {bar} {i}/{total_ov} ({pct}%)", end="", flush=True)
            if total_ov > 0:
                print("")

            if len(overall_records) > 0:
                overall_df = pd.DataFrame(overall_records)
                df = df.merge(overall_df, on="Phenotype", how="left")
                mask_overall = pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna()
                m_total = int(mask_overall.sum())
                print(f"[LRT-Stage1] m_total_non_nan={m_total}", flush=True)
                if m_total > 0:
                    _, p_adj_overall, _, _ = multipletests(df.loc[mask_overall, "P_LRT_Overall"], alpha=FDR_ALPHA, method="fdr_bh")
                    df.loc[mask_overall, "P_FDR"] = p_adj_overall
                df["Sig_FDR"] = df["P_FDR"] < FDR_ALPHA
                R_selected = int(pd.to_numeric(df["Sig_FDR"], errors="coerce").fillna(False).astype(bool).sum())
                print(f"[LRT-Stage1] Stage-1 BH complete. R_selected={R_selected}", flush=True)
            else:
                print("[LRT-Stage1] No overall LRT records found on disk.", flush=True)
                R_selected = 0
                m_total = 0

            # ---- PARALLEL STAGE-2 FOLLOW-UP WITH ATOMIC CACHING ----
            hit_names = df.loc[df["Sig_FDR"] == True, "Phenotype"].astype(str).tolist()
            print(f"[Ancestry] Scheduling follow-up for {len(hit_names)} FDR-significant phenotypes.", flush=True)
            if len(hit_names) > 0:
                tasks_follow = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": TARGET_INVERSION} for s in hit_names]
                bar_len = 40
                queued = 0
                done = 0
                lock = threading.Lock()
                with Pool(
                    processes=max(1, min(cpu_count(), 8)),
                    initializer=init_lrt_worker,
                    initargs=(core_df_with_const, allowed_mask_by_cat, anc_series),
                    maxtasksperchild=50,
                ) as pool:
                    def _cb2(_):
                        nonlocal done, queued
                        with lock:
                            done += 1
                            _print_bar(queued, done, "Ancestry")

                    for task in tasks_follow:
                        queued += 1
                        pool.apply_async(lrt_followup_worker, (task,), callback=_cb2)
                        _print_bar(queued, done, "Ancestry")
                    pool.close()
                    pool.join()
                    _print_bar(queued, done, "Ancestry")
                    print("")

            # Read follow-up cache and merge
            follow_records = []
            files_follow = [f for f in os.listdir(LRT_FOLLOWUP_CACHE_DIR) if f.endswith(".json") and not f.endswith(".meta.json")]
            total_fw = len(files_follow)
            bar_len = 30
            for i, filename in enumerate(files_follow, start=1):
                try:
                    rec = pd.read_json(os.path.join(LRT_FOLLOWUP_CACHE_DIR, filename), typ="series")
                    follow_records.append(rec.to_dict())
                except Exception as e:
                    print(f"Warning: Could not read LRT follow-up file: {filename}, Error: {e}")
                filled = int(bar_len * i / total_fw) if total_fw > 0 else bar_len
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                pct = int(i * 100 / total_fw) if total_fw > 0 else 100
                print(f"\r[FollowUp-Collect] {bar} {i}/{total_fw} ({pct}%)", end="", flush=True)
            if total_fw > 0:
                print("")

            if len(follow_records) > 0:
                follow_df = pd.DataFrame(follow_records)
                df = df.merge(follow_df, on="Phenotype", how="left")

            # Benjamini–Bogomolov adjustment for ancestry-specific tests
            m_total = int(pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna().sum())
            R_selected = int(pd.to_numeric(df["Sig_FDR"], errors="coerce").fillna(False).astype(bool).sum())
            alpha_within = (FDR_ALPHA * (R_selected / m_total)) if m_total > 0 else 0.0

            if R_selected > 0 and alpha_within > 0.0:
                selected_idx = df.index[df["Sig_FDR"] == True].tolist()
                for idx in selected_idx:
                    p_lrt = df.at[idx, "P_LRT_AncestryxDosage"] if "P_LRT_AncestryxDosage" in df.columns else np.nan
                    if (not pd.notna(p_lrt)) or (p_lrt >= LRT_SELECT_ALPHA):
                        continue
                    levels_str = df.at[idx, "LRT_Ancestry_Levels"] if "LRT_Ancestry_Levels" in df.columns else ""
                    anc_levels = [s for s in str(levels_str).split(",") if s]
                    anc_upper = [s.upper() for s in anc_levels]

                    pvals = []
                    keys = []
                    for anc in anc_upper:
                        pcol = f"{anc}_P"
                        rcol = f"{anc}_REASON"
                        if (pcol in df.columns) and (rcol in df.columns):
                            pval = df.at[idx, pcol]
                            reason_val = df.at[idx, rcol]
                            if pd.notna(pval) and reason_val != "insufficient_stratum_counts" and reason_val != "not_selected_by_LRT":
                                pvals.append(float(pval))
                                keys.append(anc)

                    if len(pvals) == 0:
                        continue

                    rej, p_adj_vals, _, _ = multipletests(pvals, alpha=alpha_within, method="fdr_bh")
                    for anc_key, adj_val in zip(keys, p_adj_vals):
                        outcol = f"{anc_key}_P_FDR"
                        df.at[idx, outcol] = float(adj_val)

            # Drop legacy column if present.
            if "EUR_P_Source" in df.columns:
                df = df.drop(columns=["EUR_P_Source"], errors="ignore")

            # FINAL_INTERPRETATION from BB-adjusted ancestry p-values
            if "Sig_FDR" in df.columns:
                df["FINAL_INTERPRETATION"] = ""
                for idx in df.index.tolist():
                    try:
                        if not bool(df.at[idx, "Sig_FDR"]):
                            df.at[idx, "FINAL_INTERPRETATION"] = ""
                            continue
                    except Exception:
                        df.at[idx, "FINAL_INTERPRETATION"] = ""
                        continue

                    p_lrt = df.at[idx, "P_LRT_AncestryxDosage"] if "P_LRT_AncestryxDosage" in df.columns else np.nan
                    if (not pd.notna(p_lrt)) or (p_lrt >= LRT_SELECT_ALPHA):
                        df.at[idx, "FINAL_INTERPRETATION"] = "overall"
                        continue

                    levels_str = df.at[idx, "LRT_Ancestry_Levels"] if "LRT_Ancestry_Levels" in df.columns else ""
                    anc_levels = [s for s in str(levels_str).split(",") if s]
                    anc_upper = [s.upper() for s in anc_levels]

                    sig_groups = []
                    for anc in anc_upper:
                        adj_col = f"{anc}_P_FDR"
                        rcol = f"{anc}_REASON"
                        if adj_col in df.columns:
                            p_adj_val = df.at[idx, adj_col]
                            reason_val = df.at[idx, rcol] if rcol in df.columns else ""
                            if pd.notna(p_adj_val) and (p_adj_val < alpha_within) and reason_val != "insufficient_stratum_counts" and reason_val != "not_selected_by_LRT":
                                sig_groups.append(anc)

                    if len(sig_groups) == 0:
                        df.at[idx, "FINAL_INTERPRETATION"] = "unable to determine"
                    else:
                        df.at[idx, "FINAL_INTERPRETATION"] = ",".join(sig_groups)

            output_filename = f"phewas_results_{TARGET_INVERSION}.csv"
            print(f"\n--- Saving final results to '{output_filename}' ---")
            df.to_csv(output_filename, index=False)

            # Filter for FDR-significant results before printing
            out_df = df[df['Sig_FDR'] == True].copy()

            # Format integer-like counts safely; print blanks when values are missing.
            for col in ["N_Total", "N_Cases", "N_Controls"]:
                out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
                out_df[col] = out_df[col].apply(lambda v: "" if pd.isna(v) else f"{int(v):,}")

            # Format floats safely; print blanks when values are missing.
            out_df["Beta"] = out_df["Beta"].apply(lambda v: "" if pd.isna(v) else f"{float(v):+0.4f}")
            out_df["OR"] = out_df["OR"].apply(lambda v: "" if pd.isna(v) else f"{float(v):0.3f}")
            out_df["P_Value"] = out_df["P_Value"].apply(lambda v: "" if pd.isna(v) else f"{float(v):.3e}")
            out_df["P_FDR"] = out_df["P_FDR"].apply(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")

            # Ensure missing Sig_FDR stays unmarked.
            out_df["Sig_FDR"] = out_df["Sig_FDR"].fillna(False).map(lambda x: "✓" if bool(x) else "")

            print(out_df.to_string(index=False))

    except Exception as e:
        import traceback
        print("\nSCRIPT HALTED DUE TO A CRITICAL ERROR:", flush=True)
        traceback.print_exc()


    finally:
        script_duration = time.time() - script_start_time
        print("\n" + "=" * 70)
        print(f" Script finished in {script_duration:.2f} seconds.")
        print("=" * 70)

if __name__ == "__main__":
    main()
