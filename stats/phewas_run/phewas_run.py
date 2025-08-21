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
    case_idx_fp = _bytes_fp(case_idx.tobytes())
    if os.path.exists(result_path) and _should_skip(meta_path, worker_core_df, case_idx_fp, category, target_inversion):
        return

    try:
        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True

        # Ensure modeling matrix contains no missing covariates by applying a non-null mask.
        nonnull_mask = ~worker_core_df.isna().any(axis=1).to_numpy()
        valid_mask = (allowed_mask_by_cat[category] | case_mask) & nonnull_mask

        n_total = int(valid_mask.sum())
        if n_total == 0:
            print(f"[Worker-{os.getpid()}] - [SKIP] {s_name:<40s} | Reason=no_valid_rows_after_mask", flush=True)
            return

        # Construct response aligned to valid rows.
        y = np.zeros(n_total, dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0:
            y[case_positions] = 1

        X_clean = worker_core_df[valid_mask].copy()
        y_clean = pd.Series(y, index=X_clean.index, name='is_case')

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

        # First attempt: LBFGS with a higher iteration cap.
        try:
            fit_try = sm.Logit(y_clean, X_clean).fit(disp=0, maxiter=400, method='lbfgs')
            fit = fit_try if _converged(fit_try) else None
        except Exception:
            fit = None

        # Second attempt: BFGS with an even higher iteration cap.
        if fit is None:
            try:
                fit_try = sm.Logit(y_clean, X_clean).fit(disp=0, maxiter=800, method='bfgs')
                fit = fit_try if _converged(fit_try) else None
            except Exception:
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


        print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta={beta:+.3f} | OR={np.exp(beta):.3f} | P={pval:.2e}", flush=True)

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
        print(f"[Worker-{os.getpid()}] - [FAIL] {s_name:<40s} | Error: {str(e)[:100]}", flush=True)
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
            covariate_cols = [TARGET_INVERSION] + ["sex"] + pc_cols + ["AGE", "AGE_sq"]

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
            del core_df, demographics_df, inversion_df, pc_df
            gc.collect()
            print(f"[Setup]    - Core covariate DataFrame ready. Shape: {core_df_with_const.shape}")
            if core_df_with_const.shape[0] == 0:
                raise RuntimeError(
                    "FATAL: Core covariate DataFrame has 0 rows after join. Check input data alignment."
                )

            core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")
            global_notnull_mask = ~core_df_with_const.isna().any(axis=1).to_numpy()
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
            while True:
                pheno_data = pheno_queue.get()
                if pheno_data is None:
                    break
                pool.apply_async(worker_func, (pheno_data,))
            pool.close()
            pool.join()

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
            
            # Helper: build y and X matrices given masks, always using PC1–PC10.
            pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
            base_cols = ['const', TARGET_INVERSION, 'sex'] + pc_cols + ['AGE', 'AGE_sq']

            def _build_y_X(valid_mask, case_mask):
                X = core_df_with_const.loc[valid_mask, base_cols].copy()
                y = np.zeros(X.shape[0], dtype=np.int8)
                case_positions = np.nonzero(case_mask[valid_mask])[0]
                if case_positions.size > 0:
                    y[case_positions] = 1
                return pd.Series(y, index=X.index, name='is_case'), X

            def _safe_fit_logit(X, y):
                # Drop zero-variance covariates within the current design, but never drop the intercept or target.
                cols_to_check = [c for c in X.columns if c not in ['const', TARGET_INVERSION]]
                zvar = [c for c in cols_to_check if X[c].nunique(dropna=False) <= 1]
                if len(zvar) > 0:
                    X = X.drop(columns=zvar)
                # If the target inversion dosage is constant within this subset, the effect is not estimable.
                if X[TARGET_INVERSION].nunique(dropna=False) <= 1:
                    return None

                def _conv(f):
                    try:
                        if hasattr(f, "mle_retvals") and isinstance(f.mle_retvals, dict):
                            return bool(f.mle_retvals.get("converged", False))
                        if hasattr(f, "converged"):
                            return bool(f.converged)
                        return False
                    except Exception:
                        return False

                fit_loc = None
                try:
                    fit_try = sm.Logit(y, X).fit(disp=0, maxiter=400, method='lbfgs')
                    fit_loc = fit_try if _conv(fit_try) else None
                except Exception:
                    fit_loc = None

                if fit_loc is None:
                    try:
                        fit_try = sm.Logit(y, X).fit(disp=0, maxiter=800, method='bfgs')
                        fit_loc = fit_try if _conv(fit_try) else None
                    except Exception:
                        fit_loc = None

                if fit_loc is None:
                    try:
                        fit_loc = sm.Logit(y, X).fit_regularized(method='lbfgs', maxiter=2000, alpha=1.0, L1_wt=0.0)
                    except Exception:
                        fit_loc = None

                return fit_loc

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
                        A = pd.get_dummies(anc_cat, prefix='ANC', drop_first=True)

                        X_red = pd.concat([X_base, A], axis=1)
                        X_full = X_red.copy()
                        for col in A.columns:
                            X_full[f"{TARGET_INVERSION}:{col}"] = X_full[TARGET_INVERSION] * X_full[col]

                        ancestry_dummy_cols = list(A.columns)
                        interaction_cols = [f"{TARGET_INVERSION}:{col}" for col in ancestry_dummy_cols]

                        # Drop zero-variance interaction columns from the full model before fitting.
                        zero_var_inters = [c for c in interaction_cols if X_full[c].var() == 0]
                        if len(zero_var_inters) > 0:
                            X_full = X_full.drop(columns=zero_var_inters)

                        varying_interactions = [c for c in interaction_cols if c in X_full.columns]

                        fit_red = _safe_fit_logit(X_red, y_all)
                        fit_full = _safe_fit_logit(X_full, y_all)

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
                        lrt_reason.append("reduced_model_failed")
                    elif fit_full is None:
                        lrt_reason.append("full_model_failed")
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
                        fit_g = _safe_fit_logit(X_g, y_g)

                        if fit_g is None or TARGET_INVERSION not in fit_g.params:
                            out[f"{anc.upper()}_OR"] = np.nan
                            out[f"{anc.upper()}_CI95"] = np.nan
                            out[f"{anc.upper()}_REASON"] = "subset_fit_failed" if fit_g is None else "coef_missing"
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

            out_df = df.copy()

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
        print(f"\nSCRIPT HALTED DUE TO A CRITICAL ERROR:\n{e}")

    finally:
        script_duration = time.time() - script_start_time
        print("\n" + "=" * 70)
        print(f" Script finished in {script_duration:.2f} seconds.")
        print("=" * 70)

if __name__ == "__main__":
    main()
