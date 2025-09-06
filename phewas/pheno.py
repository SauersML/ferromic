import os
import re
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from phewas import io

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
    pheno_defs_df["sanitized_name"] = pheno_defs_df["disease"].apply(sanitize_name)
    pheno_defs_df["all_codes"] = pheno_defs_df.apply(
        lambda row: parse_icd_codes(row["icd9_codes"]).union(parse_icd_codes(row["icd10_codes"])),
        axis=1,
    )
    return pheno_defs_df

def build_pan_category_cases(defs, bq_client, cdr_id, cache_dir, cdr_codename) -> dict:
    """Moves the “pre-calculating pan-category case sets…” block into here unchanged."""
    print("[Setup]    - Pre-calculating pan-category case sets...")
    category_cache_path = os.path.join(cache_dir, f"pan_category_cases_{cdr_codename}.pkl")
    if os.path.exists(category_cache_path):
        category_to_pan_cases = pd.read_pickle(category_cache_path)
    else:
        category_to_pan_cases = {}
        for category, group in defs.groupby("disease_category"):
            codes = list(group["all_codes"])
            pan_codes = set.union(*codes) if codes else set()
            if pan_codes:
                q = (
                    f"SELECT DISTINCT person_id FROM `{cdr_id}.condition_occurrence` "
                    f"WHERE condition_source_value IN ({','.join([repr(c) for c in pan_codes])})"
                )
                category_to_pan_cases[category] = set(
                    bq_client.query(q).to_dataframe()["person_id"].astype(str)
                )
            else:
                category_to_pan_cases[category] = set()
        pd.to_pickle(category_to_pan_cases, category_cache_path)
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

def _load_single_pheno_cache(pheno_info, core_index, cdr_codename, cache_dir):
    """THREAD WORKER: Loads one cached phenotype file from disk and returns integer case indices."""
    s_name, category = pheno_info['sanitized_name'], pheno_info['disease_category']
    pheno_cache_path = os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")
    try:
        ph = pd.read_parquet(pheno_cache_path, columns=['is_case'])
        case_ids = ph.index[ph['is_case'] == 1].astype(str)
        case_idx = core_index.get_indexer(case_ids)
        case_idx = case_idx[case_idx >= 0].astype(np.int32)
        return {"name": s_name, "category": category, "case_idx": case_idx}
    except Exception as e:
        print(f"[CacheLoader] - [FAIL] Failed to load '{s_name}': {e}", flush=True)
        return None

def phenotype_fetcher_worker(pheno_queue, pheno_defs, bq_client, cdr_id, category_to_pan_cases, cdr_codename, core_index, cache_dir, loader_chunk_size, loader_threads):
    """PRODUCER: High-performance, memory-stable data loader that works in chunks without constructing per-phenotype controls."""
    print("[Fetcher]  - Categorizing phenotypes into cached vs. uncached...")
    phenos_to_load_from_cache = [row.to_dict() for _, row in pheno_defs.iterrows() if os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))]
    phenos_to_query_from_bq = [row.to_dict() for _, row in pheno_defs.iterrows() if not os.path.exists(os.path.join(cache_dir, f"pheno_{row['sanitized_name']}_{cdr_codename}.parquet"))]
    print(f"[Fetcher]  - Found {len(phenos_to_load_from_cache)} cached phenotypes to fast-load.")
    print(f"[Fetcher]  - Found {len(phenos_to_query_from_bq)} uncached phenotypes to queue.")

    # ---  STAGE 1 - PACED PARALLEL CACHE LOADING IN CHUNKS ---
    num_chunks = (len(phenos_to_load_from_cache) + loader_chunk_size - 1) // loader_chunk_size
    for i in range(0, len(phenos_to_load_from_cache), loader_chunk_size):
        chunk = phenos_to_load_from_cache[i:i + loader_chunk_size]
        chunk_num = (i // loader_chunk_size) + 1
        print(f"[Fetcher]  - Processing chunk {chunk_num} of {num_chunks} ({len(chunk)} phenotypes)...", flush=True)
        with ThreadPoolExecutor(max_workers=loader_threads) as executor:
            future_to_pheno = {executor.submit(_load_single_pheno_cache, p_info, core_index, cdr_codename, cache_dir): p_info for p_info in chunk}
            for future in as_completed(future_to_pheno):
                result = future.result()
                if result:
                    pheno_queue.put(result)
        print(f"[Mem] RSS after chunk {chunk_num}/{num_chunks}: {io.rss_gb():.2f} GB", flush=True)
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
        pheno_cache_path = os.path.join(cache_dir, f"pheno_{s_name}_{cdr_codename}.parquet")
        # Cache the full set of case person_ids from BQ (not just the current core intersection)
        pids_for_cache = pd.Index(pids if 'pids' in locals() else [], dtype=str, name='person_id')
        df_to_cache = pd.DataFrame({'is_case': 1}, index=pids_for_cache, dtype=np.int8)
        df_to_cache.to_parquet(pheno_cache_path)


        pheno_queue.put({"name": s_name, "category": category, "case_idx": case_idx})

    pheno_queue.put(None)
    print("[Fetcher]  - All phenotypes fetched. Producer thread finished.")
