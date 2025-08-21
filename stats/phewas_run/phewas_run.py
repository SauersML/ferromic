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
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
RELATEDNESS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"

# --- Model parameters ---
NUM_PCS = 10
MIN_CASES_FILTER = 1000
MIN_CONTROLS_FILTER = 500
FDR_ALPHA = 0.05

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
        df_to_cache = pd.DataFrame(index=core_index[case_idx], data={'is_case': 1}, dtype=np.int8)
        df_to_cache.index.name = 'person_id'
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
            _t = TARGET_INVERSION
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
            pd.Series(result_data).to_json(result_path)
            _write_meta_json(meta_path, {
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
            pd.Series(result_data).to_json(result_path)
            _write_meta_json(meta_path, {
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

        fit = None

        # First attempt: Newton-Raphson with a higher iteration cap.
        try:
            fit_try = sm.Logit(y_clean, X_clean).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
            fit = fit_try if _converged(fit_try) else None
        except Exception as e:
            import traceback
            print("[TRACEBACK] run_single_model_worker newton failed:", flush=True)
            traceback.print_exc()
            fit = None

        # Second attempt: BFGS with an even higher iteration cap.
        if fit is None:
            try:
                fit_try = sm.Logit(y_clean, X_clean).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                fit = fit_try if _converged(fit_try) else None
            except Exception as e:
                import traceback
                print("[TRACEBACK] run_single_model_worker bfgs failed:", flush=True)
                traceback.print_exc()
                fit = None

        # If all fitting attempts failed, persist NA result and return.
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
            pd.Series(result_data).to_json(result_path)
            _write_meta_json(meta_path, {
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
            what = "MODEL_DID_NOT_CONVERGE" if fit is None else "COEFFICIENT_MISSING_IN_FIT_PARAMS"
            print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta=nan | OR=nan | P=nan | {what} (attempts=lbfgs,bfgs)", flush=True)
            return


        beta = float(fit.params[target_inversion])
        try:
            pval = float(fit.pvalues[target_inversion])
            pval_reason = ""
        except Exception as e:
            pval = float('nan')
            pval_reason = f"pvalue_unavailable({type(e).__name__})"
        
        suffix = f" | P_CAUSE={pval_reason}" if (not np.isfinite(pval) and pval_reason) else ""
        print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta={beta:+.3f} | OR={np.exp(beta):.3f} | P={pval:.2e}{suffix}", flush=True)

        result_data = {
            "Phenotype": s_name,
            "N_Total": n_total,
            "N_Cases": n_cases,
            "N_Controls": n_ctrls,
            "Beta": beta,
            "OR": float(np.exp(beta)),
            "P_Value": pval
        }
        pd.Series(result_data).to_json(result_path)

        _write_meta_json(meta_path, {
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

            # This block calculates and prints the condition number of the design matrix both with and
            # without the 'AGE_sq' term. A high condition number (e.g., > 1,000) indicates severe
            # multicollinearity, which can lead to numerically unstable model fits. A dramatic increase
            # in this number upon adding 'AGE_sq' would confirm it as the source of the instability.
            print("\n--- [DIAGNOSTIC] Testing matrix condition number ---")
            try:
                # Define columns for a model that excludes the squared term. This model is expected to be numerically stable.
                stable_cols = ['const', 'sex', 'AGE', TARGET_INVERSION] + [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                # Drop any rows with NA values before computing the condition number to avoid errors.
                stable_matrix = core_df_with_const[stable_cols].dropna().to_numpy()
                stable_cond = np.linalg.cond(stable_matrix)
                print(f"[DIAGNOSTIC] Condition number WITHOUT AGE_sq: {stable_cond:,.2f}")

                # Define columns for the full model as used in the workers. This model is hypothesized to be unstable.
                unstable_cols = ['const', 'sex', 'AGE', 'AGE_sq', TARGET_INVERSION] + [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                # Drop any rows with NA values before computing the condition number.
                unstable_matrix = core_df_with_const[unstable_cols].dropna().to_numpy()
                unstable_cond = np.linalg.cond(unstable_matrix)
                print(f"[DIAGNOSTIC] Condition number WITH AGE_sq:    {unstable_cond:,.2e}")
            except Exception as e:
                print(f"[DIAGNOSTIC] Could not compute condition numbers. Error: {e}")
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
            processes=cpu_count(),
            initializer=init_worker,
            initargs=(core_df_with_const, allowed_mask_by_cat),
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
            results_df = pd.DataFrame(all_results_from_disk)
            print(f"Successfully consolidated {len(results_df)} results.")

            df = results_df.sort_values("P_Value").reset_index(drop=True)
            _, p_adj, _, _ = multipletests(df["P_Value"].dropna(), alpha=FDR_ALPHA, method="fdr_bh")
            df.loc[df["P_Value"].notna(), "P_FDR"] = p_adj
            df["Sig_FDR"] = df["P_FDR"] < FDR_ALPHA

            # === Ancestry follow-up on FDR-significant phenotypes ===
            from scipy import stats

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
            
            # Helper: build y and X matrices given masks, always using PC1–PC10.
            pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
            # base_cols = ['const', TARGET_INVERSION, 'sex'] + pc_cols + ['AGE', 'AGE_sq']
            base_cols = ['const', TARGET_INVERSION, 'sex'] + pc_cols + ['AGE']

            def _build_y_X(valid_mask, case_mask):
                X = core_df_with_const.loc[valid_mask, base_cols].copy()
                # Force all covariates to plain float64 for stable numeric behavior across pandas and statsmodels.
                X = X.astype(np.float64, copy=False)
                y = np.zeros(X.shape[0], dtype=np.int8)
                case_positions = np.nonzero(case_mask[valid_mask])[0]
                if case_positions.size > 0:
                    y[case_positions] = 1
                return pd.Series(y, index=X.index, name='is_case'), X

            def _safe_fit_logit(X, y):
                """
                Fits a logistic regression with robust numeric hygiene.
                All design matrices are coerced to float64 and validated for finiteness to prevent object-dtype arrays.
                Returns a tuple of (fit_result or None, reason string).
                """
                # Coerce X to DataFrame and ensure numeric dtypes only.
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                # Convert any non-numeric columns to numeric, coercing invalid values to NaN for explicit detection.
                non_numeric_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
                if len(non_numeric_cols) > 0:
                    X = X.copy()
                    for c in non_numeric_cols:
                        X[c] = pd.to_numeric(X[c], errors="coerce")
                # Upcast to float64 to avoid pandas extension dtypes and mixed dtypes.
                X = X.astype(np.float64, copy=False)
            
                # Validate response vector and coerce to float64 1-D array.
                y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
                if y_arr.ndim != 1:
                    return None, "response_not_1d"
            
                # Fail fast on NaN or Inf in design or response with detailed diagnostics and a full stack trace.
                if not np.isfinite(X.to_numpy()).all():
                    import traceback, sys
                    arr = X.to_numpy()
                    bad_rows_mask = ~np.isfinite(arr).all(axis=1)
                    bad_row_count = int(bad_rows_mask.sum())

                    # Column-level diagnostics: NaN, +Inf, -Inf counts.
                    nan_counts = {c: int(pd.isna(X[c]).sum()) for c in X.columns}
                    posinf_counts = {c: int(np.isposinf(X[c].to_numpy()).sum()) for c in X.columns}
                    neginf_counts = {c: int(np.isneginf(X[c].to_numpy()).sum()) for c in X.columns}

                    # Partition diagnostics to pinpoint source when ancestry dummies are present.
                    anc_cols = [c for c in X.columns if c.startswith('ANC_')]
                    inter_cols = [c for c in X.columns if f"{TARGET_INVERSION}:" in c]
                    base_cols = [c for c in X.columns if c not in anc_cols + inter_cols]

                    def _allfinite_block(cols):
                        if len(cols) == 0:
                            return np.ones(len(X), dtype=bool)
                        return np.isfinite(X[cols].to_numpy()).all(axis=1)

                    only_anc = bad_rows_mask & _allfinite_block(base_cols) & (~_allfinite_block(anc_cols))
                    only_base = bad_rows_mask & (~_allfinite_block(base_cols)) & _allfinite_block(anc_cols)
                    both_blocks = bad_rows_mask & (~_allfinite_block(base_cols)) & (~_allfinite_block(anc_cols))

                    # Index diagnostics.
                    dup_count = int(X.index.duplicated(keep=False).sum())
                    idx_dtype = str(X.index.dtype)
                    sample_bad = list(map(str, X.index[bad_rows_mask][:5].tolist()))

                    print("[TRACEBACK] _safe_fit_logit detected non-finite values in design matrix", flush=True)
                    print(f"[DEBUG] design_shape={X.shape} index_dtype={idx_dtype} index_dup_count={dup_count}", flush=True)
                    print(f"[DEBUG] bad_row_count={bad_row_count} bad_rows_only_anc={int(only_anc.sum())} bad_rows_only_base={int(only_base.sum())} bad_rows_both_blocks={int(both_blocks.sum())}", flush=True)
                    print(f"[DEBUG] non_finite_columns={','.join([c for c in X.columns if nan_counts[c]>0 or posinf_counts[c]>0 or neginf_counts[c]>0])}", flush=True)
                    print(f"[DEBUG] nan_counts={{{', '.join([f'{k}:{v}' for k,v in nan_counts.items() if v>0])}}}", flush=True)
                    print(f"[DEBUG] posinf_counts={{{', '.join([f'{k}:{v}' for k,v in posinf_counts.items() if v>0])}}}", flush=True)
                    print(f"[DEBUG] neginf_counts={{{', '.join([f'{k}:{v}' for k,v in neginf_counts.items() if v>0])}}}", flush=True)
                    if bad_row_count > 0:
                        print(f"[DEBUG] bad_row_index_sample={','.join(sample_bad)}", flush=True)
                    traceback.print_stack()
                    sys.stderr.flush()
                    return None, "non_finite_in_design"
                if not np.isfinite(y_arr).all():
                    import traceback, sys
                    y_nan = int(np.isnan(y_arr).sum())
                    y_posinf = int(np.isposinf(y_arr).sum())
                    y_neginf = int(np.isneginf(y_arr).sum())
                    print("[TRACEBACK] _safe_fit_logit detected non-finite values in response vector", flush=True)
                    print(f"[DEBUG] y_nan={y_nan} y_posinf={y_posinf} y_neginf={y_neginf} y_len={len(y_arr)}", flush=True)
                    traceback.print_stack()
                    sys.stderr.flush()
                    return None, "non_finite_in_response"
            
                # Drop zero-variance covariates within the current design, but never drop the intercept or target.
                cols_to_check = [c for c in X.columns if c not in ['const', TARGET_INVERSION]]
                zvar = [c for c in cols_to_check if pd.Series(X[c]).nunique(dropna=False) <= 1]
                if len(zvar) > 0:
                    X = X.drop(columns=zvar)
            
                # If the target inversion dosage is constant within this subset, the effect is not estimable.
                if TARGET_INVERSION not in X.columns or pd.Series(X[TARGET_INVERSION]).nunique(dropna=False) <= 1:
                    return None, "target_constant"
            
                def _conv(f):
                    try:
                        if hasattr(f, "mle_retvals") and isinstance(f.mle_retvals, dict):
                            return bool(f.mle_retvals.get("converged", False))
                        if hasattr(f, "converged"):
                            return bool(f.converged)
                        return False
                    except Exception:
                        return False
            
                last_reason = ""
                try:
                    fit_try = sm.Logit(y_arr, X).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
                    if _conv(fit_try):
                        return fit_try, ""
                    last_reason = "lbfgs_not_converged"
                except Exception as e:
                    import traceback
                    print("[TRACEBACK] _safe_fit_logit newton failed:", flush=True)
                    traceback.print_exc()
                    last_reason = f"lbfgs_exception:{type(e).__name__}:{e}"
            
                try:
                    fit_try = sm.Logit(y_arr, X).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                    if _conv(fit_try):
                        return fit_try, ""
                    last_reason = "bfgs_not_converged"
                except Exception as e:
                    import traceback
                    print("[TRACEBACK] _safe_fit_logit bfgs failed:", flush=True)
                    traceback.print_exc()
                    last_reason = f"bfgs_exception:{type(e).__name__}:{e}"
            
                try:
                    fit_try = sm.Logit(y_arr, X).fit(disp=0, method='newton', maxiter=400, tol=1e-8, warn_convergence=True)
                    if _conv(fit_try):
                        return fit_try, ""
                    last_reason = "newton_not_converged"
                except Exception as e:
                    import traceback
                    print("[TRACEBACK] _safe_fit_logit newton failed:", flush=True)
                    traceback.print_exc()
                    last_reason = f"newton_exception:{type(e).__name__}:{e}"
                return None, last_reason

            def _or_ci_pair(fit, coef_name):
                beta = float(fit.params[coef_name])
                se_val = None
                if hasattr(fit, "bse"):
                    try:
                        se_val = float(fit.bse[coef_name])
                    except Exception:
                        se_val = None
                or_val = float(np.exp(beta))
                if se_val is None or not np.isfinite(se_val) or se_val == 0.0:
                    return or_val, float('nan'), float('nan')
                lo = float(np.exp(beta - 1.96 * se_val))
                hi = float(np.exp(beta + 1.96 * se_val))
                return or_val, lo, hi

            def _design_matrix_rank(X):
                """
                Computes the numerical rank of a design matrix for robust LRT df calculation.
                Uses numpy.linalg.matrix_rank on the concrete array backing the DataFrame.
                """
                try:
                    return int(np.linalg.matrix_rank(X.values))
                except Exception:
                    return int(np.linalg.matrix_rank(np.asarray(X)))

            def _convergence_flag(fit):
                """
                Returns True when the statsmodels Logit fit converged. Falls back to the
                'converged' attribute if mle_retvals is unavailable.
                """
                try:
                    if hasattr(fit, "mle_retvals") and isinstance(fit.mle_retvals, dict):
                        return bool(fit.mle_retvals.get("converged", False))
                    if hasattr(fit, "converged"):
                        return bool(fit.converged)
                    return False
                except Exception:
                    return False

            # Prepare phenotype-to-category mapping for masks.
            name_to_cat = pheno_defs_df.set_index('sanitized_name')['disease_category'].to_dict()

            # Collect follow-up results keyed by phenotype.
            follow_rows = []

            hit_names = df.loc[df["Sig_FDR"] == True, "Phenotype"].tolist()
            if len(hit_names) > 0:
                for s_name in hit_names:
                    category = name_to_cat.get(s_name, None)
                    if category is None or category not in allowed_mask_by_cat:
                        continue

                    # Read case indices from the per-phenotype cache file created earlier.
                    pheno_cache_path = os.path.join(CACHE_DIR, f"pheno_{s_name}_{cdr_codename}.parquet")
                    if not os.path.exists(pheno_cache_path):
                        continue
                    ph = pd.read_parquet(pheno_cache_path, columns=['is_case'])
                    case_ids = ph.index[ph['is_case'] == 1].astype(str)
                    case_idx = core_index.get_indexer(case_ids)
                    case_idx = case_idx[case_idx >= 0].astype(np.int32)

                    case_mask = np.zeros(len(core_index), dtype=bool)
                    if case_idx.size > 0:
                        case_mask[case_idx] = True

                    # Valid modeling mask mirrors the worker logic for this phenotype.
                    valid_mask_all = (allowed_mask_by_cat[category] | case_mask) & global_notnull_mask
                    if int(valid_mask_all.sum()) == 0:
                        continue

                    # Interaction LRT: reduced model with ancestry main effects, full adds dosage×ancestry.
                    y_all, X_base = _build_y_X(valid_mask_all, case_mask)
                    anc_vec = anc_series.loc[X_base.index]
                    anc_levels_local = anc_vec.dropna().unique().tolist()
                    if 'eur' in anc_levels_local:
                        anc_levels_local = ['eur'] + [a for a in anc_levels_local if a != 'eur']
                    anc_cat = pd.Categorical(anc_vec, categories=anc_levels_local, ordered=False)
                    # Build ancestry design for the LRT. Require at least two ancestry levels to ensure nonzero df.
                    if len(anc_levels_local) < 2:
                        A = pd.DataFrame(index=X_base.index)
                        X_red = X_base.copy()
                        X_full = X_base.copy()
                        fit_red = None
                        fit_full = None
                        p_lrt = np.nan
                        lrt_df = np.nan
                        ancestry_dummy_cols = []
                        interaction_cols = []
                        varying_interactions = []
                    else:
                        # Encode ancestry as numeric dummy variables with a plain NumPy dtype.
                        # Restrict to rows with known ancestry and enforce exact index alignment to avoid NaNs from outer alignment.
                        keep = anc_vec.notna()
                        dropped_unknown_anc = int((~keep).sum())
                        if dropped_unknown_anc > 0:
                            print(f"[DEBUG] LRT ancestry_missing_rows phenotype={s_name} dropped={dropped_unknown_anc} base_n_before={len(X_base)}", flush=True)
                        X_base = X_base.loc[keep].astype(np.float64, copy=False)
                        # make response match the filtered design (critical for shape/index alignment)
                        y_all = y_all.loc[keep]

                        anc_vec_keep = anc_vec.loc[keep]
                        anc_vec_keep = pd.Series(
                            pd.Categorical(anc_vec_keep, categories=anc_levels_local, ordered=False),
                            index=anc_vec_keep.index,
                            name='ANCESTRY'
                        )
                        A = pd.get_dummies(anc_vec_keep, prefix='ANC', drop_first=True, dtype=np.float64)
                        if not X_base.index.equals(A.index):
                            print(f"[DEBUG] Forcing ancestry-dummy reindex (base_n={len(X_base)} anc_n={len(A)})", flush=True)
                            A = A.reindex(X_base.index)

                        # Detailed index-alignment diagnostics before concatenation.
                        base_dup = int(X_base.index.duplicated(keep=False).sum())

                        anc_dup = int(A.index.duplicated(keep=False).sum())
                        print(f"[DEBUG] LRT index_alignment phenotype={s_name} base_n={len(X_base)} anc_n={len(A)} base_dup={base_dup} anc_dup={anc_dup} base_index_dtype={X_base.index.dtype} anc_index_dtype={A.index.dtype} indices_equal={X_base.index.equals(A.index)}", flush=True)
                        if (not X_base.index.equals(A.index)) or base_dup > 0 or anc_dup > 0:
                            import traceback, sys
                            print("[TRACEBACK] LRT index mismatch or duplicates detected during ancestry design assembly", flush=True)
                            traceback.print_stack()
                            sys.stderr.flush()

                        # Build reduced and full design matrices with inner alignment to prevent NaNs from misalignment.
                        X_red = pd.concat([X_base, A], axis=1, join='inner').astype(np.float64, copy=False)
                        lost_rows = len(X_base) - len(X_red)
                        if lost_rows != 0:
                            import traceback, sys
                            print(f"[DEBUG] LRT concat_inner_rows_lost phenotype={s_name} lost={lost_rows}", flush=True)
                            print("[TRACEBACK] LRT inner-join dropped rows while aligning ancestry dummies with base covariates", flush=True)
                            traceback.print_stack()
                            sys.stderr.flush()

                        X_full = X_red.copy()
                        for col in A.columns:
                            X_full[f"{TARGET_INVERSION}:{col}"] = X_full[TARGET_INVERSION] * X_full[col]
                        X_full = X_full.astype(np.float64, copy=False)

                        ancestry_dummy_cols = list(A.columns)
                        interaction_cols = [f"{TARGET_INVERSION}:{col}" for col in ancestry_dummy_cols]

                        # Drop zero-variance interaction columns from the full model before fitting.
                        zero_var_inters = [c for c in interaction_cols if X_full[c].var() == 0]
                        if len(zero_var_inters) > 0:
                            X_full = X_full.drop(columns=zero_var_inters)

                        varying_interactions = [c for c in interaction_cols if c in X_full.columns]

                        # Fit with hardened numeric design matrices to prevent object-dtype casting errors.
                        fit_red, fit_red_reason = _safe_fit_logit(X_red, y_all)
                        fit_full, fit_full_reason = _safe_fit_logit(X_full, y_all)

                        p_lrt = np.nan
                        lrt_df = np.nan

                    lrt_converged_red = False
                    lrt_converged_full = False

                    if fit_red is not None:
                        lrt_converged_red = _convergence_flag(fit_red)
                        if not lrt_converged_red:
                            print(f"[FollowUp] LRT reduced model did not converge for phenotype '{s_name}'.")

                    if fit_full is not None:
                        lrt_converged_full = _convergence_flag(fit_full)
                        if not lrt_converged_full:
                            print(f"[FollowUp] LRT full model did not converge for phenotype '{s_name}'.")

                    if (fit_red is not None) and (fit_full is not None) and (fit_full.llf >= fit_red.llf):
                        rank_red = _design_matrix_rank(X_red)
                        rank_full = _design_matrix_rank(X_full)
                        lrt_df = int(max(0, rank_full - rank_red))
                        if lrt_df > 0:
                            llr = 2.0 * (fit_full.llf - fit_red.llf)
                            p_lrt = float(stats.chi2.sf(llr, lrt_df))
                        else:
                            print(f"[FollowUp] LRT degrees of freedom computed as zero for phenotype '{s_name}'. Skipping p-value.")

                    # --- Reason for NaN LRT ---
                    lrt_reason = []
                    if len(anc_levels_local) < 2:
                        lrt_reason.append("only_one_ancestry_level")
                    elif fit_red is None:
                        lrt_reason.append(f"reduced_model_failed:{fit_red_reason}")
                    elif fit_full is None:
                        lrt_reason.append(f"full_model_failed:{fit_full_reason}")
                    elif not pd.notna(lrt_df) or int(lrt_df) == 0:
                        lrt_reason.append("no_interaction_df")
                    elif (fit_red is not None) and (fit_full is not None) and (fit_full.llf < fit_red.llf):
                        lrt_reason.append("full_llf_below_reduced_llf")
                    lrt_reason_str = ";".join(lrt_reason) if lrt_reason else ""
                    
                    out = {
                        'Phenotype': s_name,
                        'P_LRT_AncestryxDosage': p_lrt,
                        'LRT_df': lrt_df,
                        'LRT_Ancestry_Levels': ",".join(anc_levels_local),
                        'LRT_Reason': lrt_reason_str
                    }
                    for anc in anc_levels_local:
                        group_mask = valid_mask_all & anc_series.eq(anc).reindex(core_df_with_const.index).fillna(False).to_numpy()
                        if not group_mask.any():
                            out[f"{anc.upper()}_OR"] = np.nan
                            out[f"{anc.upper()}_CI95"] = np.nan
                            out[f"{anc.upper()}_REASON"] = "no_rows_in_group"
                            if anc == 'eur':
                                out["EUR_P"] = np.nan
                                out["EUR_P_Source"] = "EUR-only"
                                out["EUR_N"] = 0
                                out["EUR_N_Cases"] = 0
                                out["EUR_N_Controls"] = 0
                            continue

                        y_g, X_g = _build_y_X(group_mask, case_mask)
                        n_cases = int(y_g.sum())
                        n_tot = int(len(y_g))
                        n_ctrl = n_tot - n_cases
                        fit_g, fit_g_reason = _safe_fit_logit(X_g, y_g)
                        
                        if fit_g is None or TARGET_INVERSION not in fit_g.params:
                            out[f"{anc.upper()}_OR"] = np.nan
                            out[f"{anc.upper()}_CI95"] = np.nan
                            out[f"{anc.upper()}_REASON"] = ("subset_fit_failed" + (f":{fit_g_reason}" if fit_g_reason else "")) if fit_g is None else "coef_missing"

                            if anc == 'eur':
                                out["EUR_P"] = np.nan
                                out["EUR_P_Source"] = "EUR-only"
                                out["EUR_N"] = n_tot
                                out["EUR_N_Cases"] = n_cases
                                out["EUR_N_Controls"] = n_ctrl
                            continue

                        conv_g = _convergence_flag(fit_g)
                        if not conv_g:
                            print(f"[FollowUp] Per-ancestry model did not converge for phenotype '{s_name}' in ancestry '{anc}'.")
                        or_val, lo, hi = _or_ci_pair(fit_g, TARGET_INVERSION)
                        out[f"{anc.upper()}_OR"] = or_val
                        out[f"{anc.upper()}_CI95"] = f"{lo:.3f},{hi:.3f}"
                        if anc == 'eur':
                            try:
                                eur_p = float(fit_g.pvalues[TARGET_INVERSION])
                            except Exception:
                                eur_p = np.nan
                            out["EUR_P"] = eur_p
                            out["EUR_P_Source"] = "EUR-only"
                            out["EUR_N"] = n_tot
                            out["EUR_N_Cases"] = n_cases
                            out["EUR_N_Controls"] = n_ctrl
                    follow_rows.append(out)
                    try:
                        # Structured immediate summary for ancestry follow-up on this phenotype.
                        lrt_val = out.get('P_LRT_AncestryxDosage')
                        lrt_str = f"P_LRT={lrt_val:.3e}" if pd.notna(lrt_val) else "P_LRT=nan"
                        df_val = out.get('LRT_df')
                        df_str = f"df={int(df_val)}" if pd.notna(df_val) else "df=nan"
                        levels_str = out.get('LRT_Ancestry_Levels', '')
                        # Compose per-ancestry snippets in a stable order.
                        anc_snippets = []
                        for anc in anc_levels_local:
                            k_or = f"{anc.upper()}_OR"
                            k_ci = f"{anc.upper()}_CI95"
                            or_val = out.get(k_or, np.nan)
                            ci_val = out.get(k_ci, np.nan)
                            if pd.isna(or_val):
                                reason = out.get(f"{anc.upper()}_REASON", "")
                                rs = f" REASON={reason}" if reason else ""
                                anc_snippets.append(f"{anc.upper()}: OR=nan{rs}")
                            else:
                                if isinstance(ci_val, str):
                                    anc_snippets.append(f"{anc.upper()}: OR={float(or_val):.3f} CI95=({ci_val})")
                                else:
                                    anc_snippets.append(f"{anc.upper()}: OR={float(or_val):.3f}")

                        eur_detail = ""
                        if 'EUR_P' in out:
                            eur_p = out['EUR_P']
                            eur_n = out['EUR_N']
                            eur_nc = out['EUR_N_Cases']
                            eur_nctrl = out['EUR_N_Controls']
                            eur_p_str = f"P={float(eur_p):.3e}" if pd.notna(eur_p) else "P=nan"
                            eur_detail = f" | EUR N={eur_n} Cases={eur_nc} Controls={eur_nctrl} {eur_p_str}"
                        reason_suffix = f" | LRT_REASON={out.get('LRT_Reason','')}" if (not pd.notna(lrt_val) or not pd.notna(df_val)) else ""
                        print(f"[Ancestry] {s_name} | {lrt_str} {df_str}{reason_suffix} | Levels={levels_str} | " + " ; ".join(anc_snippets) + eur_detail, flush=True)
                    except Exception:
                        # Never allow reporting to break analysis.
                        pass
                        
            # Merge follow-up columns into main df so the final CSV includes them for hits.
            if len(follow_rows) > 0:
                follow_df = pd.DataFrame(follow_rows)
                df = df.merge(follow_df, on="Phenotype", how="left")

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
