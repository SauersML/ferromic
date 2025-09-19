import os
import re
import hashlib
from functools import lru_cache
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt, copysign
from collections import defaultdict
from itertools import combinations
from typing import Optional, Dict, Set, Tuple, List
from scipy.stats import chi2_contingency
import json
import tempfile

from . import iox as io
from . import pipes
from . import models

LOCK_MAX_AGE_SEC = 360000 # 100h

# --- BigQuery batch fetch tuning ---
PHENO_BUCKET_SERIES = [1, 4, 16, 64]  # escalate result sharding if needed
BQ_PAGE_ROWS = 50_000  # page size for streaming results
BQ_BATCH_PHENOS = 80  # max phenotypes per batch
BQ_BATCH_MAX_CODES = 8_000  # cap total codes per batch
BQ_BATCH_WORKERS = 2  # concurrent batch queries


# --- Dedup ---
PHENO_DEDUP_ENABLE = True
PHI_THRESHOLD = 0.70
SHARE_THRESHOLD = 0.70
PHENO_PROTECT: Set[str] = set()
PHENO_DEDUP_CAP_PER_PERSON: Optional[int] = 64

# --- Prevalence cap ---
# Drop phenotypes with extremely high absolute case counts before pairwise deduplication.
EXCLUDE_ABS_CASES = 90_000

_CASE_CACHE_MAX = int(os.environ.get("PHENO_CASE_CACHE_MAX", "512"))


def configure_from_ctx(ctx: Dict) -> None:
    """Override module-level knobs from CTX (if provided)."""
    global BQ_PAGE_ROWS, BQ_BATCH_PHENOS, BQ_BATCH_MAX_CODES, BQ_BATCH_WORKERS
    global PHENO_DEDUP_ENABLE, PHENO_DEDUP_CAP_PER_PERSON, PHI_THRESHOLD, SHARE_THRESHOLD, PHENO_PROTECT

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() not in ("0", "false", "no", "off")

    BQ_PAGE_ROWS = int(ctx.get("BQ_PAGE_ROWS", BQ_PAGE_ROWS))
    BQ_BATCH_PHENOS = int(ctx.get("BQ_BATCH_PHENOS", BQ_BATCH_PHENOS))
    BQ_BATCH_MAX_CODES = int(ctx.get("BQ_BATCH_MAX_CODES", BQ_BATCH_MAX_CODES))
    BQ_BATCH_WORKERS = int(ctx.get("BQ_BATCH_WORKERS", BQ_BATCH_WORKERS))
    PHENO_DEDUP_ENABLE = _as_bool(ctx.get("PHENO_DEDUP_ENABLE", PHENO_DEDUP_ENABLE))
    cap = ctx.get("PHENO_DEDUP_CAP_PER_PERSON", PHENO_DEDUP_CAP_PER_PERSON)
    if cap is None:
        PHENO_DEDUP_CAP_PER_PERSON = None
    else:
        _cap = int(cap)
        PHENO_DEDUP_CAP_PER_PERSON = _cap if _cap > 0 else None
    PHI_THRESHOLD = float(ctx.get("PHI_THRESHOLD", PHI_THRESHOLD))
    SHARE_THRESHOLD = float(ctx.get("SHARE_THRESHOLD", SHARE_THRESHOLD))
    PHENO_PROTECT = set(ctx.get("PHENO_PROTECT", PHENO_PROTECT))

def _prequeue_should_run(pheno_info, core_index, allowed_mask_by_cat, sex_vec,
                         min_cases, min_ctrls, sex_mode="majority", sex_prop=0.99, max_other=0, min_neff=None):
    """
    Decide, without loading X, whether this phenotype should be queued.
    Uses cached case indices, allowed control mask, and sex restriction rule.
    Returns True if min cases/controls (and optional Neff) are satisfiable after restriction.
    """
    category = pheno_info['disease_category']

    # 1) get case indices in core index (fast parquet read of 'is_case')
    case_idx = _load_single_pheno_cache(pheno_info, core_index,
                                        pheno_info.get('cdr_codename', ''),
                                        pheno_info.get('cache_dir', ''))  # may return None
    if not case_idx or (case_idx.get("case_idx") is None):
        return False
    case_ix_raw = case_idx["case_idx"]
    case_ix_arr = np.asarray(case_ix_raw)
    if not np.issubdtype(case_ix_arr.dtype, np.integer):
        pos = core_index.get_indexer(case_ix_arr)
        case_ix = pos[pos >= 0]
    else:
        case_ix = case_ix_arr
    if case_ix.size == 0:
        return False

    # 2) allowed control indices for this category (fallback: all allowed)
    allowed_mask = allowed_mask_by_cat.get(category, None)
    if allowed_mask is None:
        allowed_mask = np.ones(core_index.size, dtype=bool)
    ctrl_base_ix = np.flatnonzero(allowed_mask)
    if ctrl_base_ix.size == 0:
        return False

    # 3) apply sex restriction logically
    sex_cases = sex_vec[case_ix]
    n_f_case = int(np.sum(sex_cases == 0.0))
    n_m_case = int(np.sum(sex_cases == 1.0))
    total_cases = n_f_case + n_m_case
    if total_cases == 0:
        return False

    if sex_mode == "strict":
        if n_f_case > 0 and n_m_case == 0:
            dom = 0.0
        elif n_m_case > 0 and n_f_case == 0:
            dom = 1.0
        else:
            dom = None
    else:
        if n_f_case >= n_m_case:
            dom, prop, other = 0.0, n_f_case / total_cases, n_m_case
        else:
            dom, prop, other = 1.0, n_m_case / total_cases, n_f_case
        if not (prop >= sex_prop or other <= max_other):
            dom = None

    ctrl_ix = np.setdiff1d(ctrl_base_ix, case_ix, assume_unique=False)
    if dom is None:
        eff_cases = total_cases
        eff_ctrls = ctrl_ix.size
    else:
        eff_cases = n_f_case if dom == 0.0 else n_m_case
        eff_ctrls = int(np.sum(sex_vec[ctrl_ix] == dom))

    if (eff_cases < min_cases) or (eff_ctrls < min_ctrls):
        return False

    if min_neff is not None:
        neff_ub = 1.0 / (1.0/eff_cases + 1.0/eff_ctrls)
        if neff_ub < float(min_neff):
            return False

    return True

def sanitize_name(name):
    """Cleans a disease name to be a valid identifier."""
    name = re.sub(r'[\*\(\)\[\]\/\']', '', name)
    name = re.sub(r'[\s,-]+', '_', name.strip())
    return name

def parse_icd_codes(code_string):
    """Parses a semi-colon delimited string of ICD codes into a clean set."""
    if pd.isna(code_string) or not isinstance(code_string, str): return set()
    return {code.strip().strip('"') for code in code_string.split(';') if code.strip()}

def load_definitions(url) -> pd.DataFrame:
    """Copies the snippet from run: read TSV, add `sanitized_name`, compute `all_codes` using `parse_icd_codes`."""
    print("[Setup]    - Loading phenotype definitions...")
    pheno_defs_df = pd.read_csv(url, sep="\t")

    sanitized_names = pheno_defs_df["disease"].apply(sanitize_name)

    if sanitized_names.duplicated().any():
        print("[defs WARN] Sanitized name collisions detected. Appending short hash to duplicates.")
        dupes = sanitized_names[sanitized_names.duplicated()].unique()

        for d in dupes:
            idx = pheno_defs_df.index[sanitized_names == d]
            for i in idx:
                original_name = pheno_defs_df.loc[i, "disease"]
                short_hash = hashlib.sha256(original_name.encode()).hexdigest()[:6]
                sanitized_names[i] = f"{sanitized_names[i]}_{short_hash}"

    pheno_defs_df["sanitized_name"] = sanitized_names

    if pheno_defs_df["sanitized_name"].duplicated().any():
        dupes = pheno_defs_df["sanitized_name"][pheno_defs_df["sanitized_name"].duplicated()].unique()
        raise RuntimeError(
            "Sanitized name collisions persist after hashing: " + ", ".join(map(str, dupes[:10]))
        )

    pheno_defs_df["all_codes"] = pheno_defs_df.apply(
        lambda row: parse_icd_codes(row["icd9_codes"]).union(parse_icd_codes(row["icd10_codes"])),
        axis=1,
    )
    return pheno_defs_df

def build_pan_category_cases(defs, bq_client, cdr_id, cache_dir, cdr_codename) -> dict:
    print("[Setup]    - Pre-calculating pan-category case sets...")
    category_cache_path = os.path.join(cache_dir, f"pan_category_cases_{cdr_codename}.pkl")
    if os.path.exists(category_cache_path):
        try:
            return pd.read_pickle(category_cache_path)
        except Exception:
            pass

    from google.cloud import bigquery
    category_to_pan_cases = {}
    for category, group in defs.groupby("disease_category"):
        # union of sets -> sorted list of UPPER() codes
        code_sets = list(group["all_codes"])
        pan_codes = set.union(*code_sets) if code_sets else set()
        codes_upper = sorted({str(c).upper() for c in pan_codes if str(c).strip()})
        if not codes_upper:
            category_to_pan_cases[category] = set(); continue

        sql = f"""
          SELECT DISTINCT CAST(person_id AS STRING) AS person_id
          FROM `{cdr_id}.condition_occurrence`
          WHERE UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
        """
        job_cfg = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("codes", "STRING", codes_upper)]
        )
        df = bq_client.query(sql, job_config=job_cfg).to_dataframe()
        category_to_pan_cases[category] = set(df["person_id"].astype(str))

    io.atomic_write_pickle(category_cache_path, category_to_pan_cases)
    return category_to_pan_cases

def build_allowed_mask_by_cat(core_index, category_to_pan_cases, global_notnull_mask) -> dict:
    """Moves the “Building allowed-control masks…” block here unchanged."""
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
    return allowed_mask_by_cat

def populate_caches_prepass(pheno_defs_df, bq_client, cdr_id, core_index, cache_dir, cdr_codename, max_lock_age_sec=LOCK_MAX_AGE_SEC):
    """
    Populates phenotype caches deterministically using a single-writer, per-phenotype lock protocol.
    This function is safe to re-run and is resilient to crashes.
    """
    print("[Prepass]  - Starting crash-safe phenotype cache prepass.", flush=True)
    lock_dir = os.path.join(cache_dir, "locks")
    os.makedirs(lock_dir, exist_ok=True)

    # 1. Pre-calculate pan-category cases with locking
    category_cache_path = os.path.join(cache_dir, f"pan_category_cases_{cdr_codename}.pkl")
    category_lock_path = os.path.join(lock_dir, f"pan_category_cases_{cdr_codename}.lock")
    if not os.path.exists(category_cache_path):
        if io.ensure_lock(category_lock_path, max_lock_age_sec):
            try:
                if not os.path.exists(category_cache_path):  # Check again inside lock
                    print("[Prepass]  - Generating pan-category case sets...", flush=True)
                    build_pan_category_cases(pheno_defs_df, bq_client, cdr_id, cache_dir, cdr_codename)
            finally:
                io.release_lock(category_lock_path)
        else:
            print("[Prepass]  - Waiting for another process to generate pan-category cases...", flush=True)
            while not os.path.exists(category_cache_path):
                if io.ensure_lock(category_lock_path, max_lock_age_sec):
                    try:
                        if not os.path.exists(category_cache_path):
                            print("[Prepass]  - [LOCK] Reclaiming pan-category generation...", flush=True)
                            build_pan_category_cases(pheno_defs_df, bq_client, cdr_id, cache_dir, cdr_codename)
                    finally:
                        io.release_lock(category_lock_path)
                    break
                time.sleep(5)  # Wait for the other process to finish

    # 2. Process per-phenotype caches
    missing = [row.to_dict() for _, row in pheno_defs_df.iterrows() if not os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))]
    print(f"[Prepass]  - Found {len(missing)} missing phenotype caches.", flush=True)
    if not missing:
        return

    def _process_one(pheno_info):
        s_name = pheno_info['sanitized_name']
        try:
            # Fast exit if cache already exists
            ph_path = os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")
            if os.path.exists(ph_path):
                return None

            # Prepass must NEVER block on locks. Try once; if locked, skip.
            res = _query_single_pheno_bq(
                pheno_info, cdr_id, core_index, cache_dir, cdr_codename,
                bq_client=bq_client, non_blocking=True
            )
            return (res or {}).get("name")
        except Exception as e:
            print(f"[Prepass]  - [FAIL] Failed to process '{s_name}': {e}", flush=True)
            return None

    # Use a thread pool to process phenotypes in parallel, respecting locks
    with ThreadPoolExecutor(max_workers=BQ_BATCH_WORKERS * 2) as executor:
        futures = [executor.submit(_process_one, p_info) for p_info in missing]
        completed_count = 0
        for fut in as_completed(futures):
            if fut.result():
                completed_count += 1
        print(f"[Prepass]  - Successfully populated {completed_count} new caches.")

    print("[Prepass]  - Phenotype cache prepass complete.", flush=True)

@lru_cache(maxsize=_CASE_CACHE_MAX)
def _case_ids_cached(s_name: str, cdr_codename: str, cache_dir: str) -> tuple:
    """
    Read the per-phenotype parquet ONCE per process and return the case person_ids as a tuple of str.
    Pure read-only; never writes or deletes any on-disk cache.
    """
    pheno_cache_path = os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")
    ph = pd.read_parquet(pheno_cache_path, columns=['is_case'])
    case_ids = ph.index[ph['is_case'] == 1].astype(str)
    return tuple(case_ids)

def _load_single_pheno_cache(pheno_info, core_index, cdr_codename, cache_dir):
    """THREAD WORKER: Loads one cached phenotype (via memoized IDs) and returns integer case indices."""
    s_name = pheno_info['sanitized_name']
    category = pheno_info['disease_category']
    try:
        # 1) Fast path: memoized person_id strings (no repeat disk I/O on cache hit)
        case_ids = _case_ids_cached(s_name, cdr_codename, cache_dir)

        # 2) Map those person_ids to positions in THIS inversion's core_index
        if not case_ids:
            return None
        pos = core_index.get_indexer(pd.Index(case_ids))
        case_idx = pos[pos >= 0].astype(np.int32)

        if case_idx.size == 0:
            return None
        return {"name": s_name, "category": category, "case_idx": case_idx}
    except Exception as e:
        print(f"[CacheLoader] - [FAIL] Failed to load '{s_name}': {e}", flush=True)
        return None

def _query_single_pheno_bq(pheno_info, cdr_id, core_index, cache_dir, cdr_codename, bq_client=None, non_blocking=False):
    """THREAD WORKER: Queries one phenotype from BigQuery, caches it, and returns a descriptor."""
    from google.cloud import bigquery
    if bq_client is None:
        bq_client = bigquery.Client()
    s_name, category, all_codes = pheno_info['sanitized_name'], pheno_info['disease_category'], pheno_info['all_codes']
    codes_upper = sorted({str(c).upper() for c in (all_codes or set()) if str(c).strip()})

    pheno_cache_path = os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")
    lock_dir = os.path.join(cache_dir, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"pheno_{s_name}_{cdr_codename}.lock")

    # Non-blocking prepass: skip immediately if another process is working on it
    if non_blocking:
        if os.path.exists(pheno_cache_path):
            return {"name": s_name, "category": category, "codes_n": len(codes_upper)}
        if not io.ensure_lock(lock_path, LOCK_MAX_AGE_SEC):
            print(f"[Prepass]  - Skipping '{s_name}', another process has the lock.", flush=True)
            return None
        try:
            # do the work without re-locking later
            if os.path.exists(pheno_cache_path):
                return {"name": s_name, "category": category, "codes_n": len(codes_upper)}
            print(f"[Fetcher]  - [BQ] Querying '{s_name}'...", flush=True)
            pids = []
            if codes_upper:
                sql = f"""
                  SELECT DISTINCT CAST(person_id AS STRING) AS person_id
                  FROM `{cdr_id}.condition_occurrence`
                  WHERE condition_source_value IS NOT NULL
                    AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
                """
                try:
                    job_cfg = bigquery.QueryJobConfig(
                        query_parameters=[bigquery.ArrayQueryParameter("codes", "STRING", codes_upper)]
                    )
                    df_ids = bq_client.query(sql, job_config=job_cfg).to_dataframe()
                    pids = df_ids["person_id"].astype(str)
                except Exception as e:
                    print(f"[Fetcher]  - [FAIL] BQ query failed for {s_name}. Error: {str(e)[:150]}", flush=True)
                    pids = []
            pids_for_cache = pd.Index(sorted(pids), dtype=str, name='person_id')
            df_to_cache = pd.DataFrame({'is_case': 1}, index=pids_for_cache, dtype=np.int8)
            io.atomic_write_parquet(pheno_cache_path, df_to_cache, compression="snappy")
            print(f"[Fetcher]  - Cached {len(pids_for_cache):,} new cases for '{s_name}'", flush=True)
            return {"name": s_name, "category": category, "codes_n": len(codes_upper)}
        finally:
            io.release_lock(lock_path)

    # Default (blocking) path for normal callers; includes stale-lock breaker
    while True:
        if io.ensure_lock(lock_path, LOCK_MAX_AGE_SEC):
            try:
                if os.path.exists(pheno_cache_path):
                    return {"name": s_name, "category": category, "codes_n": len(codes_upper)}
                print(f"[Fetcher]  - [BQ] Querying '{s_name}'...", flush=True)
                pids = []
                if codes_upper:
                    sql = f"""
                      SELECT DISTINCT CAST(person_id AS STRING) AS person_id
                      FROM `{cdr_id}.condition_occurrence`
                      WHERE condition_source_value IS NOT NULL
                        AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
                    """
                    try:
                        job_cfg = bigquery.QueryJobConfig(
                            query_parameters=[bigquery.ArrayQueryParameter("codes", "STRING", codes_upper)]
                        )
                        df_ids = bq_client.query(sql, job_config=job_cfg).to_dataframe()
                        pids = df_ids["person_id"].astype(str)
                    except Exception as e:
                        print(f"[Fetcher]  - [FAIL] BQ query failed for {s_name}. Error: {str(e)[:150]}", flush=True)
                        pids = []
                pids_for_cache = pd.Index(sorted(pids), dtype=str, name='person_id')
                df_to_cache = pd.DataFrame({'is_case': 1}, index=pids_for_cache, dtype=np.int8)
                io.atomic_write_parquet(pheno_cache_path, df_to_cache, compression="snappy")
                print(f"[Fetcher]  - Cached {len(pids_for_cache):,} new cases for '{s_name}'", flush=True)
                return {"name": s_name, "category": category, "codes_n": len(codes_upper)}
            finally:
                io.release_lock(lock_path)
        else:
            # If another process holds the lock, break stale locks to avoid infinite wait
            try:
                if os.path.exists(lock_path):
                    st = os.stat(lock_path)
                    if (time.time() - st.st_mtime) > LOCK_MAX_AGE_SEC:
                        print(f"[Fetcher]  - [LOCK] Breaking stale lock for '{s_name}'", flush=True)
                        io.release_lock(lock_path)
                        continue
            except FileNotFoundError:
                pass
            if os.path.exists(pheno_cache_path):
                return {"name": s_name, "category": category, "codes_n": len(codes_upper)}
            time.sleep(0.5)

def _batch_pheno_defs(phenos_to_query_from_bq, max_phenos, max_codes):
    """
    Yield lists of pheno rows such that each batch respects both:
      - <= max_phenos phenotypes
      - <= max_codes total ICD codes across the batch
    """
    batch, code_tally = [], 0
    for row in phenos_to_query_from_bq:
        n_codes = len(row.get("all_codes") or [])
        # start new batch if limits would be exceeded
        if batch and (len(batch) >= max_phenos or (code_tally + n_codes) > max_codes):
            yield batch
            batch, code_tally = [], 0
        batch.append(row)
        code_tally += n_codes
    if batch:
        yield batch

def _query_batch_bq(batch_infos, bq_client, cdr_id, core_index, cache_dir, cdr_codename):
    """
    THREAD WORKER: Queries MANY phenotypes in one scan using an Array<STRUCT<code STRING, pheno STRING>> parameter.
    Streams results page-by-page and shards by person_id buckets when needed to bound output size.

    Returns: list of {"name": sanitized_name, "category": disease_category, "codes_n": int}
    and writes per-phenotype parquet caches (is_case=1).
    """
    from google.cloud import bigquery  # local import to not affect unit tests that bypass BQ

    lock_dir = os.path.join(cache_dir, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    locks = {}
    filtered_infos = []
    for row in batch_infos:
        s_name = row["sanitized_name"]
        lock_path = os.path.join(lock_dir, f"pheno_{s_name}_{cdr_codename}.lock")
        pheno_cache_path = os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")

        # If cache already exists, no need to lock/include
        if os.path.exists(pheno_cache_path):
            continue

        # Try to acquire the per-pheno lock
        if io.ensure_lock(lock_path, LOCK_MAX_AGE_SEC):
            # Double-check cache inside the lock
            if os.path.exists(pheno_cache_path):
                io.release_lock(lock_path)
                continue
            locks[s_name] = lock_path
            filtered_infos.append(row)
            continue

        # Another process holds the lock. If it's stale, break it and try once.
        try:
            st = os.stat(lock_path)
            if (time.time() - st.st_mtime) > LOCK_MAX_AGE_SEC:
                print(f"[Fetcher]  - [LOCK] Breaking stale lock for '{s_name}'", flush=True)
                io.release_lock(lock_path)  # break stale lock
                if io.ensure_lock(lock_path, LOCK_MAX_AGE_SEC):
                    if os.path.exists(pheno_cache_path):
                        io.release_lock(lock_path)
                        continue
                    locks[s_name] = lock_path
                    filtered_infos.append(row)
                    continue
        except FileNotFoundError:
            # Lock disappeared between checks; let the next loop/worker handle it
            pass

        # Fresh lock held by someone else: skip this phenotype (do not spin)
        print(f"[Fetcher]  - [Batch] Skipping '{s_name}', another process has the lock.", flush=True)

    batch_infos = filtered_infos
    try:
        codes_list = []
        phenos_list = []
        meta = {}
        for row in batch_infos:
            s_name = row["sanitized_name"]
            category = row["disease_category"]
            codes = list((row.get("all_codes") or set()))
            codes_upper = sorted({str(c).upper() for c in codes if str(c).strip()})
            meta[s_name] = {"category": category, "codes": codes_upper}
            for c in codes_upper:
                codes_list.append(c)
                phenos_list.append(s_name)
        if not codes_list:
            out = []
            for s_name, m in meta.items():
                pheno_cache_path = os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")
                io.atomic_write_parquet(pheno_cache_path,
                    pd.DataFrame({'is_case': []}, index=pd.Index([], name='person_id'), dtype=np.int8),
                    compression="snappy")
                out.append({"name": s_name, "category": m["category"], "codes_n": len(m["codes"])})
            return out

        sql = f"""
      WITH code_pairs AS (
        SELECT code, pheno
        FROM UNNEST(@codes) AS code WITH OFFSET off
        JOIN UNNEST(@phenos) AS pheno WITH OFFSET off2
        ON off = off2
      )
      SELECT DISTINCT CAST(co.person_id AS STRING) AS person_id, cp.pheno AS pheno
      FROM `{cdr_id}.condition_occurrence` AS co
      JOIN code_pairs AS cp
        ON co.condition_source_value IS NOT NULL
       AND UPPER(TRIM(co.condition_source_value)) = cp.code
      WHERE MOD(ABS(FARM_FINGERPRINT(CAST(co.person_id AS STRING))), @bucket_count) = @bucket_id
    """

        pheno_to_pids = {s_name: set() for s_name in meta.keys()}
        succeeded = False
        for bucket_count in PHENO_BUCKET_SERIES:
            try:
                pheno_to_pids_attempt = {s_name: set() for s_name in meta.keys()}
                for bucket_id in range(bucket_count):
                    job_cfg = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ArrayQueryParameter("codes", "STRING", codes_list),
                            bigquery.ArrayQueryParameter("phenos", "STRING", phenos_list),
                            bigquery.ScalarQueryParameter("bucket_count", "INT64", bucket_count),
                            bigquery.ScalarQueryParameter("bucket_id", "INT64", bucket_id),
                        ])
                    job = bq_client.query(sql, job_config=job_cfg)
                    for page in job.result(page_size=BQ_PAGE_ROWS).pages:
                        for row in page:
                            pheno_to_pids_attempt[row.pheno].add(str(row.person_id))
                pheno_to_pids = pheno_to_pids_attempt
                succeeded = True
                break
            except Exception as e:
                print(f"[Fetcher]  - [WARN] Batch failed at {bucket_count} buckets: {str(e)[:200]}", flush=True)
                pheno_to_pids = {s_name: set() for s_name in meta.keys()}

        if not succeeded:
            print(f"[Fetcher]  - [FAIL] Batch could not be fetched after {PHENO_BUCKET_SERIES} buckets. Falling back to per-phenotype queries.", flush=True)
            for lp in locks.values():
                io.release_lock(lp)
            locks.clear()
            results = []
            for row in batch_infos:
                try:
                    results.append(_query_single_pheno_bq(row, cdr_id, core_index, cache_dir, cdr_codename, bq_client=bq_client))
                except Exception as e:
                    print(f"[Fetcher]  - [FAIL] Fallback single query failed for {row['sanitized_name']}: {str(e)[:200]}", flush=True)
                    results.append({"name": row["sanitized_name"], "category": row["disease_category"], "codes_n": len(row.get("all_codes") or [])})
            return results

        results = []
        for s_name, m in meta.items():
            pids = pheno_to_pids[s_name] if succeeded else set()
            pheno_cache_path = os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")
            if os.path.exists(pheno_cache_path):
                results.append({"name": s_name, "category": m["category"], "codes_n": len(m["codes"])})
                continue
            idx_for_cache = pd.Index(sorted(list(pids)), name='person_id')
            df_to_cache = pd.DataFrame({'is_case': 1}, index=idx_for_cache, dtype=np.int8)
            io.atomic_write_parquet(pheno_cache_path, df_to_cache, compression="snappy")
            print(f"[Fetcher]  - Cached {len(df_to_cache):,} cases for '{s_name}' (batched)", flush=True)
            results.append({"name": s_name, "category": m["category"], "codes_n": len(m["codes"])})

        return results
    finally:
        for lock_path in locks.values():
            io.release_lock(lock_path)

def phenotype_fetcher_worker(pheno_queue,
                             pheno_defs: pd.DataFrame,
                             bq_client,
                             cdr_id: str,
                             cdr_codename: str,
                             core_index: pd.Index,
                             cache_dir: str,
                             loader_chunk_size: int,
                             loader_threads: int,
                             allow_bq: bool = True,
                             allowed_mask_by_cat: Optional[dict] = None,
                             sex_vec: Optional[np.ndarray] = None,
                             min_cases: int = 1000,
                             min_ctrls: int = 1000,
                             sex_mode: str = "majority",
                             sex_prop: float = 0.99,
                             max_other: int = 0,
                             min_neff: Optional[int] = None):
    """
    PRODUCER: High-performance, memory-stable data loader that now optionally performs *global phenotype deduplication*
    before enqueueing work. Dedup uses two rules: phi > PHI_THRESHOLD or directional case-share >= SHARE_THRESHOLD.
    """
    print("[Fetcher]  - Categorizing phenotypes into cached vs. uncached...", flush=True)
    phenos_to_load_from_cache = [
        row.to_dict() for _, row in pheno_defs.iterrows()
        if os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))
    ]
    phenos_to_query_from_bq = [
        row.to_dict() for _, row in pheno_defs.iterrows()
        if not os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))
    ]
    print(f"[Fetcher]  - Found {len(phenos_to_load_from_cache)} cached phenotypes to fast-load.", flush=True)
    print(f"[Fetcher]  - Found {len(phenos_to_query_from_bq)} uncached phenotypes.", flush=True)

    keep_set: Optional[Set[str]] = None

    if PHENO_DEDUP_ENABLE:
        # Phase 0: ensure caches exist for all phenotypes so dedup has full visibility (coarse lock to avoid duplicate scans)
        if phenos_to_query_from_bq:
            if allow_bq and (bq_client is not None) and (cdr_id is not None):
                prec_lock = os.path.join(cache_dir, f"dedup_precache_{cdr_codename}.lock")
                if io.ensure_lock(prec_lock, LOCK_MAX_AGE_SEC):
                    try:
                        print(f"[Fetcher]  - [Dedup] Pre-caching {len(phenos_to_query_from_bq)} uncached phenotypes via BigQuery...", flush=True)
                        _precache_all_missing_phenos(pheno_defs, bq_client, cdr_id, core_index, cache_dir, cdr_codename)
                    except Exception as e:
                        print(f"[Fetcher]  - [WARN] Pre-cache failed: {str(e)[:200]}", flush=True)
                    finally:
                        io.release_lock(prec_lock)
                else:
                    print("[Fetcher]  - [Dedup] Another worker is pre-caching; proceeding with currently cached subset.", flush=True)
            else:
                print("[Fetcher]  - [WARN] Dedup requested but BigQuery is disabled or unavailable; proceeding with cached subset only.", flush=True)

        # Refresh cached list after any precache pass
        phenos_to_load_from_cache = [
            row.to_dict() for _, row in pheno_defs.iterrows()
            if os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))
        ]
        phenos_to_query_from_bq = []  # everything needed for dedup has been cached (or we proceed with cached-only)

        # Phase 1: compute/use manifest once per cohort (fingerprinted by the cohort index)
        try:
            cohort_fp = str(models._index_fingerprint(core_index))
            manifest_path = os.path.join(cache_dir, f"pheno_dedup_manifest_{cdr_codename}_{cohort_fp}.json")
            manifest_lock = manifest_path + ".lock"

            keep_set = None
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r") as fh:
                        manifest = json.load(fh)
                    ks = manifest.get("kept", [])
                    if isinstance(ks, list):
                        keep_set = set(ks)
                        print(f"[Dedup] using existing manifest: kept {len(keep_set)}, dropped {len(manifest.get('dropped', []))}.", flush=True)
                except Exception:
                    keep_set = None

            if keep_set is None:
                if io.ensure_lock(manifest_lock, LOCK_MAX_AGE_SEC):
                    try:
                        # Check again after acquiring the lock
                        if os.path.exists(manifest_path):
                            with open(manifest_path, "r") as fh:
                                manifest = json.load(fh)
                            ks = manifest.get("kept", [])
                            keep_set = set(ks) if isinstance(ks, list) else None
                        if keep_set is None:
                            ded = deduplicate_phenotypes(
                                pheno_defs_df=pheno_defs,
                                core_index=core_index,
                                cdr_codename=cdr_codename,
                                cache_dir=cache_dir,
                                min_cases=min_cases,
                                phi_threshold=PHI_THRESHOLD,
                                share_threshold=SHARE_THRESHOLD,
                                protect=PHENO_PROTECT
                            )
                            keep_set = set(ded.get("kept", set()))
                            print(f"[Dedup] kept {len(keep_set)} phenotypes; dropped {len(ded.get('dropped', []))}.", flush=True)
                    finally:
                        io.release_lock(manifest_lock)
                else:
                    # Another worker is computing; wait briefly then read
                    time.sleep(3)
                    try:
                        with open(manifest_path, "r") as fh:
                            manifest = json.load(fh)
                        ks = manifest.get("kept", [])
                        keep_set = set(ks) if isinstance(ks, list) else None
                    except Exception:
                        keep_set = None
        except Exception as e:
            print(f"[Dedup WARN] Falling back to no-dedup due to error: {e}", flush=True)
            keep_set = None

    # STAGE 1: enqueue cached phenotypes (optionally filtered by dedup keep_set)
    enqueued = 0
    for row in phenos_to_load_from_cache:
        sname = row['sanitized_name']
        if (keep_set is not None) and (sname not in keep_set):
            continue

        row['cdr_codename'] = cdr_codename
        row['cache_dir'] = cache_dir

        if (allowed_mask_by_cat is not None) and (sex_vec is not None):
            if not _prequeue_should_run(row, core_index, allowed_mask_by_cat, sex_vec,
                                        min_cases, min_ctrls, sex_mode, sex_prop, max_other, min_neff):
                continue

        pheno_queue.put({
            "name": sname,
            "category": row['disease_category'],
            "codes_n": len(row.get('all_codes') or []),
            "cdr_codename": cdr_codename
        })
        enqueued += 1

    # STAGE 2: handle uncached (only if dedup disabled AND BigQuery allowed)
    if (not PHENO_DEDUP_ENABLE) and phenos_to_query_from_bq:
        if allow_bq and (bq_client is not None) and (cdr_id is not None):
            print(f"[Fetcher]  - Processing {len(phenos_to_query_from_bq)} uncached phenotypes from BQ in batches...", flush=True)
            phenos_to_query_from_bq.sort(key=lambda r: len(r.get("all_codes") or []), reverse=True)
            phenos_to_query_from_bq = [
                r for r in phenos_to_query_from_bq
                if not os.path.exists(os.path.join(cache_dir, f"pheno_{r['sanitized_name']}_{cdr_codename}.parquet"))
            ]
            batches = list(_batch_pheno_defs(phenos_to_query_from_bq, BQ_BATCH_PHENOS, BQ_BATCH_MAX_CODES))
            print(f"[Fetcher]  - Created {len(batches)} batches (<= {BQ_BATCH_PHENOS} phenos and <= {BQ_BATCH_MAX_CODES} codes per batch).", flush=True)

            with ThreadPoolExecutor(max_workers=min(BQ_BATCH_WORKERS, len(batches))) as executor:
                inflight = set()
                for batch in batches:
                    fut = executor.submit(_query_batch_bq, batch, bq_client, cdr_id, core_index, cache_dir, cdr_codename)
                    inflight.add(fut)
                for fut in as_completed(inflight):
                    try:
                        results = fut.result()
                        for r in results:
                            info = {
                                'sanitized_name': r['name'],
                                'disease_category': r['category'],
                                'cdr_codename': cdr_codename,
                                'cache_dir': cache_dir,
                            }
                            if (allowed_mask_by_cat is not None) and (sex_vec is not None):
                                if not _prequeue_should_run(info, core_index, allowed_mask_by_cat, sex_vec,
                                                            min_cases, min_ctrls, sex_mode, sex_prop, max_other, min_neff):
                                    continue
                            pheno_queue.put({
                                "name": r['name'],
                                "category": r['category'],
                                "codes_n": int(r.get("codes_n") or 0),
                                "cdr_codename": cdr_codename
                            })
                            enqueued += 1
                    except Exception as e:
                        print(f"[Fetcher]  - [FAIL] Batch query failed: {str(e)[:200]}", flush=True)
        else:
            print(f"[Fetcher]  - [WARN] Skipping {len(phenos_to_query_from_bq)} uncached phenotypes because allow_bq=False or client/cdr missing.", flush=True)

    pheno_queue.put(None)
    print(f"[Fetcher]  - All phenotypes fetched. Producer thread finished. Enqueued={enqueued}.", flush=True)


# --- Dedup functions ---

def phi_from_2x2(n11: int, n10: int, n01: int, n00: int) -> float:
    """
    Compute the phi coefficient (Pearson correlation for two binary variables)
    using SciPy's chi2_contingency for robustness, with the correct sign.
    """
    n = n11 + n10 + n01 + n00
    if n <= 0:
        return 0.0
    try:
        chi2, _, _, _ = chi2_contingency([[n11, n10], [n01, n00]], correction=False)
        mag = sqrt(max(0.0, chi2 / n))
        sign_num = (n11 * n00) - (n10 * n01)
        return copysign(mag, sign_num)
    except Exception:
        # On any numerical issue, fall back to 0.0 which cannot trigger a drop.
        return 0.0


def _build_pair_overlap_counts(cases_by_pheno: Dict[str, np.ndarray],
                               cap_per_person: Optional[int] = PHENO_DEDUP_CAP_PER_PERSON) -> Dict[Tuple[str, str], int]:
    """
    Builds sparse co-occurrence counts k(A,B)=|cases(A) ∩ cases(B)| by streaming per-person membership.
    Only pairs that actually co-occur are produced.

    :param cases_by_pheno: mapping of phenotype name -> sorted np.ndarray[int] of person indices
    :param cap_per_person: optional cap of pairings per person to mitigate combinatorial explosion
    :return: dict with keys (min(name), max(name)) -> intersection count
    """
    person_to_phenos: Dict[int, List[str]] = defaultdict(list)
    for name, idx in cases_by_pheno.items():
        # idx is sorted int32 array of person positions
        for p in idx:
            person_to_phenos[int(p)].append(name)

    pair_k: Dict[Tuple[str, str], int] = defaultdict(int)
    for phenos in person_to_phenos.values():
        if len(phenos) < 2:
            continue
        # Optional cap: limit the number of combinations contributed by this "hub" person
        if cap_per_person is not None and len(phenos) > 1:
            phenos = sorted(phenos)[:cap_per_person]
        else:
            phenos = sorted(phenos)
        for a, b in combinations(phenos, 2):
            pair_k[(a, b)] += 1
    return pair_k


def _tiebreak_drop(a_name: str, n1a: int, n_codes_a: int,
                   b_name: str, n1b: int, n_codes_b: int) -> str:
    """
    Decide which phenotype to drop when a rule fires.
    Primary: smaller case count; Secondary: fewer ICD codes; Tertiary: lexicographically larger name.
    Returns the name to DROP.
    """
    if n1a < n1b:
        return a_name
    if n1b < n1a:
        return b_name
    if n_codes_a < n_codes_b:
        return a_name
    if n_codes_b < n_codes_a:
        return b_name
    # final tiebreak: drop lexicographically larger to keep determinism
    return b_name if b_name > a_name else a_name


def _write_dedup_manifest(path: str, manifest: dict) -> None:
    """
    Atomically write dedup manifest JSON to the given path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix="pheno_dedup_manifest.", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(manifest, fh, separators=(",", ":"), ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _precache_all_missing_phenos(pheno_defs_df: pd.DataFrame,
                                 bq_client,
                                 cdr_id: str,
                                 core_index: pd.Index,
                                 cache_dir: str,
                                 cdr_codename: str) -> None:
    """
    Ensure *all* phenotypes in pheno_defs_df have on-disk parquet caches before dedup runs.
    Uses existing batching helpers to minimize BigQuery scans.
    """
    # Coarse guard so only one precache runner works a subset at a time
    lock_dir = os.path.join(cache_dir, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    prec_guard = os.path.join(lock_dir, f"precache_subset_{cdr_codename}.lock")
    if not io.ensure_lock(prec_guard, LOCK_MAX_AGE_SEC):
        print("[Dedup]     - Pre-caching already running elsewhere; skipping.", flush=True)
        return
    try:
        missing = [
            row.to_dict() for _, row in pheno_defs_df.iterrows()
            if not os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))
        ]
        if not missing:
            return

        missing.sort(key=lambda r: len(r.get("all_codes") or []), reverse=True)
        batches = list(_batch_pheno_defs(missing, BQ_BATCH_PHENOS, BQ_BATCH_MAX_CODES))
        print(f"[Dedup]     - Pre-caching: {len(missing)} phenotypes in {len(batches)} batches.", flush=True)

        with ThreadPoolExecutor(max_workers=min(BQ_BATCH_WORKERS, max(1, len(batches)))) as executor:
            futs = [executor.submit(_query_batch_bq, batch, bq_client, cdr_id, core_index, cache_dir, cdr_codename)
                    for batch in batches]
            for fut in as_completed(futs):
                try:
                    _ = fut.result()
                except Exception as e:
                    print(f"[Dedup]     - [WARN] Batch pre-cache failed: {str(e)[:200]}", flush=True)
    finally:
        io.release_lock(prec_guard)


def deduplicate_phenotypes(pheno_defs_df: pd.DataFrame,
                           core_index: pd.Index,
                           cdr_codename: str,
                           cache_dir: str,
                           min_cases: int,
                           phi_threshold: float = PHI_THRESHOLD,
                           share_threshold: float = SHARE_THRESHOLD,
                           protect: Optional[Set[str]] = None) -> dict:
    """
    Greedy, deterministic phenotype deduplication using two rules:
      1) phi > phi_threshold  -> drop the one with fewer cases (ties: more codes wins, then name)
      2) directional share >= share_threshold in *either* direction -> drop the smaller

    Prints a message every time a phenotype is dropped, with the reason and relevant statistics.

    Returns:
      {
        "config": {...},
        "kept": [names...],
        "dropped": [{"name":..., "reason":..., "with":..., "k":..., "n1_self":..., "n1_other":..., "phi":..., "share_self":..., "share_other":...}, ...]
      }
    """
    protect = protect or set()
    N = int(len(core_index))

    # Cohort-specific manifest path
    cohort_fp = str(models._index_fingerprint(core_index))
    manifest_path = os.path.join(cache_dir, f"pheno_dedup_manifest_{cdr_codename}_{cohort_fp}.json")

    # 1) Materialize case indices per phenotype from cache; filter by min_cases
    all_codes_map = pheno_defs_df.set_index("sanitized_name")["all_codes"].to_dict()
    cat_map = pheno_defs_df.set_index("sanitized_name")["disease_category"].to_dict()

    cases_by_pheno: Dict[str, np.ndarray] = {}
    n1_map: Dict[str, int] = {}
    n_codes_map: Dict[str, int] = {}
    kept: Set[str] = set()
    dropped_names: Set[str] = set()
    dropped_records: List[dict] = []

    for s_name in pheno_defs_df["sanitized_name"]:
        try:
            case_ids = _case_ids_cached(s_name, cdr_codename, cache_dir)
        except Exception:
            continue
        if not case_ids:
            continue
        pos = core_index.get_indexer(pd.Index(case_ids))
        idx = pos[pos >= 0].astype(np.int32, copy=False)
        n1 = int(idx.size)
        if n1 < int(min_cases):
            continue
        # Prevalence cap: exclude ultra-common phenotypes up front
        if n1 >= EXCLUDE_ABS_CASES:
            print(f"Dropped '{s_name}' due to prevalence cap: n1={n1} >= {EXCLUDE_ABS_CASES} of N={N}")
            dropped_records.append({
                "name": s_name,
                "reason": "prevalence_cap",
                "with": None,
                "k": 0,
                "n1_self": n1,
                "n1_other": N,
                "share_self": (n1 / float(N)) if N else 0.0,
                "share_other": None
            })
            dropped_names.add(s_name)
            continue
        cases_by_pheno[s_name] = np.sort(idx, kind="mergesort")
        n1_map[s_name] = n1
        n_codes_map[s_name] = len(all_codes_map.get(s_name) or [])

    if not cases_by_pheno:
        manifest = {
            "config": {"min_cases": min_cases, "phi_threshold": phi_threshold, "share_threshold": share_threshold, "N": N},
            "kept": [],
            "dropped": dropped_records
        }
        _write_dedup_manifest(manifest_path, manifest)
        return {"kept": set(), "dropped": dropped_records}

    # 2) Build sparse overlap counts only for co-occurring pairs
    pair_k = _build_pair_overlap_counts(cases_by_pheno, cap_per_person=PHENO_DEDUP_CAP_PER_PERSON)

    # 3) Prepare deterministic order: cases desc, codes desc, name asc
    items = [(name, n1_map[name], n_codes_map.get(name, 0)) for name in cases_by_pheno.keys()]
    items.sort(key=lambda t: (-t[1], -t[2], t[0]))

    # Build neighbor map from pair_k to avoid scanning non-overlapping pairs
    neighbors: Dict[str, Set[str]] = defaultdict(set)
    for (a, b), _k in pair_k.items():
        neighbors[a].add(b)
        neighbors[b].add(a)

    # Helper to form a consistent pair key
    def _pair_key(x: str, y: str) -> Tuple[str, str]:
        return (x, y) if x < y else (y, x)

    # Helper to explain tiebreak decision in human-readable form
    def _tiebreak_explanation(a_name: str, n1a: int, n_codes_a: int,
                              b_name: str, n1b: int, n_codes_b: int,
                              dropped: str) -> str:
        if n1a != n1b:
            # smaller case count drops
            if n1a < n1b and dropped == a_name:
                return f"tiebreak: smaller case count ({n1a} < {n1b})"
            if n1b < n1a and dropped == b_name:
                return f"tiebreak: smaller case count ({n1b} < {n1a})"
        if n_codes_a != n_codes_b:
            # fewer ICD codes drops
            if n_codes_a < n_codes_b and dropped == a_name:
                return f"tiebreak: fewer ICD codes ({n_codes_a} < {n_codes_b})"
            if n_codes_b < n_codes_a and dropped == b_name:
                return f"tiebreak: fewer ICD codes ({n_codes_b} < {n_codes_a})"
        # final: lexicographically larger name drops
        larger = b_name if b_name > a_name else a_name
        if dropped == larger:
            return f"tiebreak: lexicographically larger name ('{larger}')"
        return "tiebreak: deterministic rule"

    # 4) Greedy sweep
    for name_a, n1a, n_codes_a in items:
        if name_a in dropped_names:
            continue
        kept.add(name_a)

        for name_b in sorted(neighbors.get(name_a, [])):
            if name_b in dropped_names:
                continue
            n1b = n1_map.get(name_b, 0)
            n_codes_b = n_codes_map.get(name_b, 0)

            # Intersection size from sparse map (skip if zero or missing)
            k = pair_k.get(_pair_key(name_a, name_b), 0)
            if k <= 0:
                continue

            share_ab = k / float(n1a) if n1a > 0 else 0.0
            share_ba = k / float(n1b) if n1b > 0 else 0.0

            # Rule 2: directional share threshold
            dropped_now = False
            if (share_ab >= share_threshold) or (share_ba >= share_threshold):
                drop = _tiebreak_drop(name_a, n1a, n_codes_a, name_b, n1b, n_codes_b)
                if drop not in protect:
                    other = name_b if drop == name_a else name_a
                    # Explain which direction(s) triggered
                    trigger_dirs = []
                    if share_ab >= share_threshold:
                        trigger_dirs.append(f"{name_a}->{name_b} ({share_ab:.4f}≥{share_threshold})")
                    if share_ba >= share_threshold:
                        trigger_dirs.append(f"{name_b}->{name_a} ({share_ba:.4f}≥{share_threshold})")
                    tie_expl = _tiebreak_explanation(name_a, n1a, n_codes_a, name_b, n1b, n_codes_b, drop)
                    print(
                        f"Dropped '{drop}' because directional share>=threshold with '{other}'; "
                        f"triggers: {', '.join(trigger_dirs)}; "
                        f"k={k}, n1_self={n1_map.get(drop, 0)}, n1_other={n1_map.get(other, 0)}, "
                        f"share_self={(share_ba if drop == name_b else share_ab):.4f}, "
                        f"share_other={(share_ab if drop == name_b else share_ba):.4f}; {tie_expl}"
                    )

                    dropped_records.append({
                        "name": drop,
                        "reason": "share>=threshold",
                        "with": other,
                        "k": int(k),
                        "n1_self": int(n1_map.get(drop, 0)),
                        "n1_other": int(n1_map.get(other, 0)),
                        "share_self": float(share_ba if drop == name_b else share_ab),
                        "share_other": float(share_ab if drop == name_b else share_ba)
                    })
                    dropped_names.add(drop)
                    if drop == name_a:
                        kept.discard(name_a)
                        dropped_now = True
            if dropped_now:
                break  # A was dropped; stop scanning its neighbors

            # Rule 1: phi coefficient threshold
            n10 = n1a - k
            n01 = n1b - k
            n00 = N - (k + n10 + n01)
            phi = phi_from_2x2(k, n10, n01, n00)
            if phi > phi_threshold:
                drop = _tiebreak_drop(name_a, n1a, n_codes_a, name_b, n1b, n_codes_b)
                if drop in protect:
                    continue
                other = name_b if drop == name_a else name_a
                tie_expl = _tiebreak_explanation(name_a, n1a, n_codes_a, name_b, n1b, n_codes_b, drop)
                print(
                    f"Dropped '{drop}' because phi>{phi_threshold} with '{other}'; "
                    f"k={k}, n1_self={n1_map.get(drop, 0)}, n1_other={n1_map.get(other, 0)}, "
                    f"phi={phi:.4f}; {tie_expl}"
                )

                dropped_records.append({
                    "name": drop,
                    "reason": "phi>threshold",
                    "with": other,
                    "k": int(k),
                    "n1_self": int(n1_map.get(drop, 0)),
                    "n1_other": int(n1_map.get(other, 0)),
                    "phi": float(phi)
                })
                dropped_names.add(drop)
                if drop == name_a:
                    kept.discard(name_a)
                    break  # A was dropped; move on to next A

    manifest = {
        "config": {
            "min_cases": min_cases,
            "phi_threshold": phi_threshold,
            "share_threshold": share_threshold,
            "N": N
        },
        "kept": sorted(list(kept)),
        "dropped": dropped_records
    }
    _write_dedup_manifest(manifest_path, manifest)
    return {"kept": kept, "dropped": dropped_records}
