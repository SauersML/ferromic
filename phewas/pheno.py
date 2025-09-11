import os
import re
import hashlib
from functools import lru_cache
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import iox as io
from . import pipes

LOCK_MAX_AGE_SEC = 360000 # 100h

# --- BigQuery batch fetch tuning ---
PHENO_BUCKET_SERIES = [1, 4, 16, 64]  # escalate result sharding if needed
BQ_PAGE_ROWS = int(os.getenv("BQ_PAGE_ROWS", "50000"))  # page size for streaming results
BQ_BATCH_PHENOS = int(os.getenv("BQ_BATCH_PHENOS", "80"))  # max phenotypes per batch
BQ_BATCH_MAX_CODES = int(os.getenv("BQ_BATCH_MAX_CODES", "8000"))  # cap total codes per batch
BQ_BATCH_WORKERS = int(os.getenv("BQ_BATCH_WORKERS", "2"))  # concurrent batch queries

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
        print("[defs ERROR] Sanitized name collisions persist after hashing.")

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
                if not os.path.exists(category_cache_path): # Check again inside lock
                    print("[Prepass]  - Generating pan-category case sets...", flush=True)
                    build_pan_category_cases(pheno_defs_df, bq_client, cdr_id, cache_dir, cdr_codename)
            finally:
                io.release_lock(category_lock_path)
        else:
            print("[Prepass]  - Waiting for another process to generate pan-category cases...", flush=True)
            while not os.path.exists(category_cache_path):
                time.sleep(5) # Wait for the other process to finish

    # 2. Process per-phenotype caches
    missing = [row.to_dict() for _, row in pheno_defs_df.iterrows() if not os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))]
    print(f"[Prepass]  - Found {len(missing)} missing phenotype caches.", flush=True)
    if not missing:
        return

    def _process_one(pheno_info):
        s_name = pheno_info['sanitized_name']
        lock_path = os.path.join(lock_dir, f"pheno_{s_name}_{cdr_codename}.lock")

        if not io.ensure_lock(lock_path, max_lock_age_sec):
            print(f"[Prepass]  - Skipping '{s_name}', another process has the lock.", flush=True)
            return None

        try:
            # Check again inside the lock in case another process finished
            if os.path.exists(os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")):
                return None

            # Use the single-phenotype query function which is simpler and sufficient here
            _query_single_pheno_bq(pheno_info, cdr_id, core_index, cache_dir, cdr_codename, bq_client=bq_client)
            return s_name
        except Exception as e:
            print(f"[Prepass]  - [FAIL] Failed to process '{s_name}': {e}", flush=True)
            return None
        finally:
            io.release_lock(lock_path)

    # Use a thread pool to process phenotypes in parallel, respecting locks
    with ThreadPoolExecutor(max_workers=BQ_BATCH_WORKERS * 2) as executor:
        futures = [executor.submit(_process_one, p_info) for p_info in missing]
        completed_count = 0
        for fut in as_completed(futures):
            if fut.result():
                completed_count += 1
        print(f"[Prepass]  - Successfully populated {completed_count} new caches.")

    print("[Prepass]  - Phenotype cache prepass complete.", flush=True)

@lru_cache(maxsize=4096)
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

def _query_single_pheno_bq(pheno_info, cdr_id, core_index, cache_dir, cdr_codename, bq_client=None):
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
        while True:
            if io.ensure_lock(lock_path, LOCK_MAX_AGE_SEC):
                if os.path.exists(pheno_cache_path):
                    io.release_lock(lock_path)
                    break
                locks[s_name] = lock_path
                filtered_infos.append(row)
                break
            else:
                if os.path.exists(pheno_cache_path):
                    break
                time.sleep(0.5)

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

def phenotype_fetcher_worker(pheno_queue, pheno_defs, bq_client, cdr_id, cdr_codename,
                             core_index, cache_dir, loader_chunk_size, loader_threads, allow_bq=True,
                             allowed_mask_by_cat=None, sex_vec=None,
                             min_cases=1000, min_ctrls=1000,
                             sex_mode="majority", sex_prop=0.99, max_other=0, min_neff=None):
    """PRODUCER: High-performance, memory-stable data loader that works in chunks without constructing per-phenotype controls."""
    print("[Fetcher]  - Categorizing phenotypes into cached vs. uncached...")
    phenos_to_load_from_cache = [row.to_dict() for _, row in pheno_defs.iterrows() if os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))]
    phenos_to_query_from_bq = [row.to_dict() for _, row in pheno_defs.iterrows() if not os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))]
    print(f"[Fetcher]  - Found {len(phenos_to_load_from_cache)} cached phenotypes to fast-load.")
    print(f"[Fetcher]  - Found {len(phenos_to_query_from_bq)} uncached phenotypes to queue.")

    for row in phenos_to_load_from_cache:
        row['cdr_codename'] = cdr_codename
        row['cache_dir'] = cache_dir
        if (allowed_mask_by_cat is not None) and (sex_vec is not None):
            if not _prequeue_should_run(row, core_index, allowed_mask_by_cat, sex_vec,
                                        min_cases, min_ctrls, sex_mode, sex_prop, max_other, min_neff):
                continue
        pheno_queue.put({"name": row['sanitized_name'], "category": row['disease_category'], "codes_n": len(row.get('all_codes') or []), "cdr_codename": cdr_codename})

    # STAGE 2: CONCURRENT BIGQUERY QUERIES (BATCHED)
    if phenos_to_query_from_bq:
        if allow_bq and bq_client is not None and cdr_id is not None:
            print(f"[Fetcher]  - Processing {len(phenos_to_query_from_bq)} uncached phenotypes from BQ in batches...", flush=True)
            # Sort by number of codes to make batches more uniform in size
            phenos_to_query_from_bq.sort(key=lambda r: len(r.get("all_codes") or []), reverse=True)
            phenos_to_query_from_bq = [r for r in phenos_to_query_from_bq if not os.path.exists(os.path.join(cache_dir, f"pheno_{r['sanitized_name']}_{cdr_codename}.parquet"))]
            # Build batches respecting phenotype and code-count caps
            batches = list(_batch_pheno_defs(phenos_to_query_from_bq, BQ_BATCH_PHENOS, BQ_BATCH_MAX_CODES))
            print(f"[Fetcher]  - Created {len(batches)} batches "
                  f"(<= {BQ_BATCH_PHENOS} phenos and <= {BQ_BATCH_MAX_CODES} codes per batch).", flush=True)

            # Use conservative concurrency because each batch scans the table once
            with ThreadPoolExecutor(max_workers=min(BQ_BATCH_WORKERS, len(batches))) as executor:
                inflight = set()
                for batch in batches:
                    while (pipes.BUDGET.remaining_gb() < pipes.BUDGET.floor_gb()) or (len(inflight) >= 2 * BQ_BATCH_WORKERS):
                        done = {f for f in inflight if f.done()}
                        for f in list(done):
                            inflight.remove(f)
                            try:
                                results = f.result()
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
                                    r["cdr_codename"] = cdr_codename
                                    pheno_queue.put(r)
                            except Exception as e:
                                print(f"[Fetcher]  - [FAIL] Batch query failed: {str(e)[:200]}", flush=True)
                        time.sleep(0.2)
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
                            r["cdr_codename"] = cdr_codename
                            pheno_queue.put(r)
                    except Exception as e:
                        print(f"[Fetcher]  - [FAIL] Batch query failed: {str(e)[:200]}", flush=True)
        else:
            if phenos_to_query_from_bq:
                print(f"[Fetcher]  - [WARN] Skipping {len(phenos_to_query_from_bq)} uncached phenotypes because allow_bq=False. Sample: {phenos_to_query_from_bq[0]['sanitized_name']}", flush=True)

    pheno_queue.put(None)
    print("[Fetcher]  - All phenotypes fetched. Producer thread finished.")
