import os
import re
import time
import ast
import warnings
import gc
from datetime import datetime
import threading
import queue
from functools import partial
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import statsmodels.api as sm
from google.cloud import bigquery
from statsmodels.stats.multitest import multipletests

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

def init_worker(df_to_share):
    """Sends the large core_df to each worker process ONCE to save memory and time."""
    global worker_core_df
    worker_core_df = df_to_share
    print(f"[Worker-{os.getpid()}] Initialized and received shared core dataframe.", flush=True)


class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


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
    """Generic caching wrapper."""
    if os.path.exists(cache_path):
        print(f"  -> Found cache, loading from '{cache_path}'...")
        return pd.read_parquet(cache_path)
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

def _load_single_pheno_cache(pheno_info, base_ids, category_to_pan_cases, cdr_codename):
    """THREAD WORKER: Loads one cached phenotype file from disk."""
    s_name, category = pheno_info['sanitized_name'], pheno_info['disease_category']
    pheno_cache_path = os.path.join(CACHE_DIR, f"pheno_{s_name}_{cdr_codename}.parquet")
    try:
        pheno_data_df = pd.read_parquet(pheno_cache_path)
        cases = set(pheno_data_df[pheno_data_df['is_case'] == 1].index.astype(str))
        controls = base_ids - category_to_pan_cases[category]
        return {"name": s_name, "cases": cases, "controls": controls}
    except Exception as e:
        print(f"[CacheLoader] - [FAIL] Failed to load '{s_name}': {e}", flush=True)
        return None

def phenotype_fetcher_worker(pheno_queue, pheno_defs, bq_client, cdr_id, base_ids, category_to_pan_cases, cdr_codename):
    """PRODUCER: High-performance, memory-stable data loader that works in chunks."""
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
        
        # A new, temporary thread pool is created for each chunk.
        # This ensures memory from completed futures is garbage collected between chunks.
        with ThreadPoolExecutor(max_workers=LOADER_THREADS) as executor:
            future_to_pheno = {executor.submit(_load_single_pheno_cache, p_info, base_ids, category_to_pan_cases, cdr_codename): p_info for p_info in chunk}
            for future in as_completed(future_to_pheno):
                result = future.result()
                if result:
                    pheno_queue.put(result) # Will block if main queue is full
    print("[Fetcher]  - Finished all parallel cache loading chunks.")

    # STAGE 2: SLOW SEQUENTIAL BIGQUERY QUERIES
    for pheno_info in phenos_to_query_from_bq:
        s_name, category, all_codes = pheno_info['sanitized_name'], pheno_info['disease_category'], pheno_info['all_codes']
        print(f"[Fetcher]  - [BQ] Querying '{s_name}'...", flush=True)

        # Check if there are any codes to query for this phenotype
        if not all_codes:
            cases = set()
        else:
            # Format codes for the SQL IN clause
            formatted_codes = ','.join([repr(c) for c in all_codes])
            q = f"SELECT DISTINCT person_id FROM `{cdr_id}.condition_occurrence` WHERE condition_source_value IN ({formatted_codes})"
            
            try:
                # Execute query and get a set of person_ids as strings, intersected with the base population
                df_ids = bq_client.query(q).to_dataframe()
                cases = set(df_ids["person_id"].astype(str)).intersection(base_ids)
            except Exception as e:
                print(f"[Fetcher]  - [FAIL] BQ query failed for {s_name}. Error: {str(e)[:150]}", flush=True)
                cases = set()
        
        # Cache the newly fetched data (even if it's empty) so we don't query again next time.
        print(f"[Fetcher]  - Caching {len(cases):,} new cases for '{s_name}'", flush=True)
        pheno_cache_path = os.path.join(CACHE_DIR, f"pheno_{s_name}_{cdr_codename}.parquet")
        # Create a DataFrame to cache the case IDs
        df_to_cache = pd.DataFrame(index=list(cases), data={'is_case': 1}, dtype=np.int8)
        df_to_cache.index.name = 'person_id'
        df_to_cache.to_parquet(pheno_cache_path)
        
        # Define controls for this phenotype's category
        controls = base_ids - category_to_pan_cases[category]
        
        # Put the complete data onto the queue for the consumer processes
        pheno_queue.put({"name": s_name, "cases": cases, "controls": controls})

    pheno_queue.put(None) # Sentinel value to signal completion
    print("[Fetcher]  - All phenotypes fetched. Producer thread finished.")


def run_single_model_worker(pheno_data, target_inversion, results_cache_dir):
    """CONSUMER: Runs a single model. Executed in a separate process."""
    global worker_core_df
    s_name = pheno_data["name"]
    result_path = os.path.join(results_cache_dir, f"{s_name}.json")
    
    if os.path.exists(result_path):
        return

    try:
        y = pd.Series(0, index=list(pheno_data["cases"].union(pheno_data["controls"])), name='is_case', dtype=np.int8)
        y.loc[list(pheno_data["cases"])] = 1
        
        combined = pd.concat([y, worker_core_df], axis=1, join='inner').dropna()
        y_clean, X_clean = combined['is_case'], combined.drop(columns=['is_case'])
        
        n_cases, n_ctrls = y_clean.sum(), len(y_clean) - y_clean.sum()

        if n_cases < MIN_CASES_FILTER or n_ctrls < MIN_CONTROLS_FILTER:
            return

        model = sm.Logit(y_clean, X_clean)
        fit = model.fit(disp=0, maxiter=200)
        beta, pval = fit.params[target_inversion], fit.pvalues[target_inversion]
        
        print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={len(y_clean):,} | Cases={n_cases:,} | Beta={beta:+.3f} | OR={np.exp(beta):.3f} | P={pval:.2e}", flush=True)

        result_data = {
            "Phenotype": s_name, "N_Total": int(len(y_clean)), "N_Cases": int(n_cases), 
            "N_Controls": int(n_ctrls), "Beta": float(beta), "OR": float(np.exp(beta)), "P_Value": float(pval)
        }
        pd.Series(result_data).to_json(result_path)

    except Exception as e:
        print(f"[Worker-{os.getpid()}] - [FAIL] {s_name:<40s} | Error: {str(e)[:100]}", flush=True)
    finally:
        del pheno_data, y
        if 'combined' in locals(): del combined
        if 'y_clean' in locals(): del y_clean
        if 'X_clean' in locals(): del X_clean
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
            pheno_defs_df = pd.read_csv(PHENOTYPE_DEFINITIONS_URL, sep='\t')
            pheno_defs_df['sanitized_name'] = pheno_defs_df['disease'].apply(sanitize_name)
            pheno_defs_df['all_codes'] = pheno_defs_df.apply(lambda row: parse_icd_codes(row['icd9_codes']).union(parse_icd_codes(row['icd10_codes'])), axis=1)
            
            print("[Setup]    - Setting up BigQuery client...")
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            cdr_codename = cdr_dataset_id.split('.')[-1]
            
            print("[Setup]    - Loading shared covariates (Demographics, Inversions, PCs, Sex)...")
            demographics_df = get_cached_or_generate(
                os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"),
                _load_demographics_with_stable_age, bq_client=bq_client, cdr_id=cdr_dataset_id
            )
            inversion_df = get_cached_or_generate(os.path.join(CACHE_DIR, f"inversion_{TARGET_INVERSION}.parquet"), _load_inversions)
            pc_df = get_cached_or_generate(os.path.join(CACHE_DIR, "pcs_10.parquet"), _load_pcs, gcp_project=gcp_project)
            sex_df = get_cached_or_generate(os.path.join(CACHE_DIR, "genetic_sex.parquet"), _load_genetic_sex, gcp_project=gcp_project)
            
            # Load related individuals to remove. No caching needed as it's a small file and a fast operation.
            related_ids_to_remove = _load_related_to_remove(gcp_project=gcp_project)
            
            print("[Setup]    - Standardizing covariate indexes for robust joining...")
            demographics_df.index = demographics_df.index.astype(str)
            inversion_df.index = inversion_df.index.astype(str)
            pc_df.index = pc_df.index.astype(str)
            sex_df.index = sex_df.index.astype(str)
            
            pc_cols = [f'PC{i}' for i in range(1, NUM_PCS + 1)]
            # Add 'sex' to the list of model covariates
            covariate_cols = [TARGET_INVERSION] + ['sex'] + pc_cols + ['AGE', 'AGE_sq']
            
            # Add sex_df to the join chain
            core_df = demographics_df.join(inversion_df, how='inner') \
                                     .join(pc_df, how='inner') \
                                     .join(sex_df, how='inner')
            
            print(f"[Setup]    - Pre-filter cohort size: {len(core_df):,}")
            # Apply the relatedness filter
            core_df = core_df[~core_df.index.isin(related_ids_to_remove)]
            print(f"[Setup]    - Post-filter unrelated cohort size: {len(core_df):,}")
    
            # Now select the final columns
            core_df = core_df[covariate_cols]
            
            core_df_with_const = sm.add_constant(core_df, prepend=True)
            del core_df, demographics_df, inversion_df, pc_df; gc.collect()
            print(f"[Setup]    - Core covariate DataFrame ready. Shape: {core_df_with_const.shape}")
            if core_df_with_const.shape[0] == 0:
                raise RuntimeError("FATAL: Core covariate DataFrame has 0 rows after join. Check input data alignment.")

            print("[Setup]    - Pre-calculating pan-category case sets...")
            category_cache_path = os.path.join(CACHE_DIR, f"pan_category_cases_{cdr_codename}.pkl")
            if os.path.exists(category_cache_path):
                category_to_pan_cases = pd.read_pickle(category_cache_path)
            else:
                # The base population is defined later from the covariate-eligible cohort to ensure consistent denominators.
                unused_base_ids_df = None
                # Placeholder assignment to maintain variable existence; the definitive assignment occurs after covariate assembly.
                base_ids = set()
                category_to_pan_cases = {}
                for category, group in pheno_defs_df.groupby('disease_category'):
                    pan_codes = set.union(*group['all_codes'])
                    if pan_codes:
                        q = f"SELECT DISTINCT person_id FROM `{cdr_dataset_id}.condition_occurrence` WHERE condition_source_value IN ({','.join([repr(c) for c in pan_codes])})"
                        category_to_pan_cases[category] = set(bq_client.query(q).to_dataframe()['person_id'].astype(str))
                    else:
                        category_to_pan_cases[category] = set()
                pd.to_pickle(category_to_pan_cases, category_cache_path)
            print("[Setup]    - Defining base population as covariate-eligible cohort...")
            base_ids = set(core_df_with_const.index.astype(str))
            print(f"[Setup]    - Base population size (covariate-eligible): {len(base_ids):,}")


        print(f"\n--- Total Setup Time: {t_setup.duration:.2f}s ---")

        # --- PART 2: RUNNING THE PIPELINE ---
        pheno_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        fetcher_thread = threading.Thread(target=phenotype_fetcher_worker, args=(pheno_queue, pheno_defs_df, bq_client, cdr_dataset_id, base_ids, category_to_pan_cases, cdr_codename))
        fetcher_thread.start()
        
        worker_func = partial(run_single_model_worker, target_inversion=TARGET_INVERSION, results_cache_dir=RESULTS_CACHE_DIR)
        
        print(f"\n--- Starting parallel model fitting with {cpu_count()} worker processes ---")
        with Pool(processes=cpu_count(), initializer=init_worker, initargs=(core_df_with_const,)) as pool:
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
        for filename in os.listdir(RESULTS_CACHE_DIR):
            if filename.endswith('.json'):
                try:
                    result = pd.read_json(os.path.join(RESULTS_CACHE_DIR, filename), typ='series')
                    all_results_from_disk.append(result.to_dict())
                except Exception as e:
                    print(f"Warning: Could not read corrupted result file: {filename}, Error: {e}")
        
        if not all_results_from_disk:
            print("No results found to process.")
        else:
            results_df = pd.DataFrame(all_results_from_disk)
            print(f"Successfully consolidated {len(results_df)} results.")
            
            print("\n--- FINAL RESULTS (sorted by P-value) ---")
            df = results_df.sort_values("P_Value").reset_index(drop=True)
            _, p_adj, _, _ = multipletests(df['P_Value'].dropna(), alpha=FDR_ALPHA, method="fdr_bh")
            df.loc[df['P_Value'].notna(), "P_FDR"] = p_adj
            df["Sig_FDR"] = df["P_FDR"] < FDR_ALPHA

            output_filename = f"phewas_results_{TARGET_INVERSION}.csv"
            print(f"\n--- Saving final results to '{output_filename}' ---")
            df.to_csv(output_filename, index=False)
            
            out_df = df.copy()
            for col in ['N_Total', 'N_Cases', 'N_Controls']: out_df[col] = out_df[col].astype(int).map('{:,.0f}'.format)
            out_df['Beta'] = out_df['Beta'].map('{:+.4f}'.format); out_df['OR'] = out_df['OR'].map('{:.3f}'.format)
            out_df['P_Value'] = out_df['P_Value'].map('{:.3e}'.format); out_df['P_FDR'] = out_df['P_FDR'].map('{:.3f}'.format)
            out_df['Sig_FDR'] = out_df['Sig_FDR'].map(lambda x: "✓" if x else "")
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
