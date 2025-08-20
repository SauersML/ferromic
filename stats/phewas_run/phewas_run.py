import os
import re
import time
import ast
import warnings
from datetime import datetime
import threading
import queue
import numpy as np
import pandas as pd
import statsmodels.api as sm
from google.cloud import bigquery
from statsmodels.stats.multitest import multipletests

# --- Configuration ---
TARGET_INVERSION = 'chr17-45585160-INV-706887'
PHENOTYPE_DEFINITIONS_URL = "https://github.com/SauersML/ferromic/raw/refs/heads/main/data/significant_heritability_diseases.tsv"

# Data sources and caching
CACHE_DIR = "./phewas_cache"
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"

# Model parameters
NUM_PCS = 10
MIN_CASES_FILTER = 1000
MIN_CONTROLS_FILTER = 500
FDR_ALPHA = 0.05

# Suppress pandas warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)


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

# --- Data Loading and Caching Helpers ---

def get_cached_or_generate(cache_path, generation_func, *args, **kwargs):
    """Generic caching wrapper for large, single files."""
    if os.path.exists(cache_path):
        print(f"  -> Found cache, loading from '{cache_path}'...")
        return pd.read_parquet(cache_path)
    print(f"  -> No cache found at '{cache_path}'. Generating data...")
    data = generation_func(*args, **kwargs)
    print(f"  -> Saving new cache to '{cache_path}'...")
    data.to_parquet(cache_path)
    return data

def _load_inversions():
    """Efficiently loads the target inversion dosage with a string index."""
    try:
        df = pd.read_csv(
            INVERSION_DOSAGES_FILE, sep="\t",
            usecols=["SampleID", TARGET_INVERSION]
        )
        df['SampleID'] = df['SampleID'].astype(str)
        df = df.set_index('SampleID').rename_axis('person_id')
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load inversion data: {e}")

def _load_pcs(gcp_project):
    """Efficiently loads genetic PCs with a string index."""
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

# --- Producer-Consumer Pipeline Functions ---

def phenotype_fetcher_worker(pheno_queue, pheno_defs, bq_client, cdr_id, base_ids, category_to_pan_cases, cdr_codename):
    """PRODUCER THREAD: Fetches data for one phenotype at a time and puts it onto the queue."""
    def query_person_ids(icd_codes):
        if not icd_codes: return set()
        formatted_codes = ','.join([repr(c) for c in icd_codes])
        if not formatted_codes: return set()
        
        q = f"SELECT DISTINCT person_id FROM `{cdr_id}.condition_occurrence` WHERE condition_source_value IN ({formatted_codes})"
        try:
            df_ids = bq_client.query(q).to_dataframe()
            return set(df_ids["person_id"].astype(str)).intersection(base_ids)
        except Exception as e:
            print(f"[Fetcher]  - BQ query failed. Error: {str(e)[:150]}")
            return set()

    for _, row in pheno_defs.iterrows():
        s_name, category = row['sanitized_name'], row['disease_category']
        pheno_cache_path = os.path.join(CACHE_DIR, f"pheno_{s_name}_{cdr_codename}.parquet")
        
        if os.path.exists(pheno_cache_path):
            print(f"[Fetcher]  - [CACHE] Loading '{s_name}'")
            pheno_data_df = pd.read_parquet(pheno_cache_path)
            cases = set(pheno_data_df[pheno_data_df['is_case'] == 1].index.astype(str))
        else:
            print(f"[Fetcher]  - [BQ] Querying '{s_name}'...")
            cases = query_person_ids(row['all_codes'])
            temp_df = pd.DataFrame(index=list(cases), dtype=np.int8)
            temp_df['is_case'] = 1
            temp_df.index.name = 'person_id'
            print(f"[Fetcher]  - Caching {len(cases):,} cases for '{s_name}'")
            temp_df.to_parquet(pheno_cache_path)

        controls = base_ids - category_to_pan_cases[category]
        pheno_queue.put({"name": s_name, "cases": cases, "controls": controls})

    pheno_queue.put(None)
    print("[Fetcher]  - All phenotypes fetched. Worker thread finished.")

def run_phewas_pipeline(core_df, pheno_queue):
    """CONSUMER: Takes phenotype data from the queue, runs the model, and collects results."""
    print("\n--- PART 2: RUNNING ASSOCIATION MODELS (Consumer starting) ---")
    pc_cols = [f'PC{i}' for i in range(1, NUM_PCS + 1)]
    covariate_cols = [TARGET_INVERSION] + pc_cols + ['AGE', 'AGE_sq']
    X_core = sm.add_constant(core_df[covariate_cols].copy(), prepend=True)
    
    all_results = []
    phenos_tested, phenos_skipped = 0, 0

    while True:
        pheno_data = pheno_queue.get()
        if pheno_data is None:
            pheno_queue.task_done()
            break

        s_name = pheno_data["name"]
        analysis_ids = pheno_data["cases"].union(pheno_data["controls"])
        y = pd.Series(0, index=list(analysis_ids), name='is_case', dtype=np.int8)
        y.loc[list(pheno_data["cases"])] = 1
        
        combined = pd.concat([y, X_core], axis=1, join='inner').dropna()
        y_clean, X_clean = combined['is_case'], combined.drop(columns=['is_case'])
        
        n_cases, n_ctrls = y_clean.sum(), len(y_clean) - y_clean.sum()

        if n_cases < MIN_CASES_FILTER or n_ctrls < MIN_CONTROLS_FILTER:
            phenos_skipped += 1
            pheno_queue.task_done()
            continue
        
        phenos_tested += 1
        try:
            model = sm.Logit(y_clean, X_clean)
            fit = model.fit(disp=0, maxiter=200)
            beta, pval = fit.params[TARGET_INVERSION], fit.pvalues[TARGET_INVERSION]
            
            print(f"[Modeler]  - [OK] {s_name:<40s} | N={len(y_clean):,}, cases={n_cases:,} | beta={beta:+.4f}, OR={np.exp(beta):.3f}, p={pval:.2e}")
            all_results.append({
                "Phenotype": s_name, "N_Total": len(y_clean), "N_Cases": n_cases, 
                "N_Controls": n_ctrls, "Beta": beta, "OR": np.exp(beta), "P_Value": pval
            })
        except Exception as e:
            print(f"[Modeler]  - [FAIL] {s_name:<40s} | Error: {str(e)[:100]}")
        
        pheno_queue.task_done()

    print(f"\n[Modeler]  - Consumer finished. Tested {phenos_tested} eligible phenotypes, skipped {phenos_skipped}.")
    return pd.DataFrame(all_results)


def main():
    script_start_time = time.time()
    print("=" * 70)
    print(" Starting Concurrent & Memory-Efficient PheWAS Pipeline (v3 - Fixed)")
    print("=" * 70)
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    try:
        with Timer() as t_setup:
            print("\n[Setup]    - Loading and processing phenotype definitions from URL...")
            pheno_defs_df = pd.read_csv(PHENOTYPE_DEFINITIONS_URL, sep='\t')
            pheno_defs_df['sanitized_name'] = pheno_defs_df['disease'].apply(sanitize_name)
            pheno_defs_df['all_codes'] = pheno_defs_df.apply(lambda row: parse_icd_codes(row['icd9_codes']).union(parse_icd_codes(row['icd10_codes'])), axis=1)
            print(f"[Setup]    - Loaded {len(pheno_defs_df)} phenotype definitions.")
            
            print("[Setup]    - Setting up BigQuery client...")
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            cdr_codename = cdr_dataset_id.split('.')[-1]
            
            print("[Setup]    - Fetching base population...")
            persons_df = bq_client.query(f"SELECT person_id FROM `{cdr_dataset_id}.person`").to_dataframe()
            base_ids = set(persons_df["person_id"].astype(str))

            print("[Setup]    - Loading shared covariates (Demographics, Inversions, PCs)...")
            demographics_df = get_cached_or_generate(
                os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"),
                lambda: bq_client.query(f"SELECT person_id, year_of_birth FROM `{cdr_dataset_id}.person`").to_dataframe().assign(person_id=lambda d: d.person_id.astype(str)).set_index('person_id').assign(AGE=datetime.now().year - pd.to_numeric(lambda d: d.year_of_birth), AGE_sq=lambda d: d.AGE**2)[['AGE', 'AGE_sq']]
            )
            inversion_df = get_cached_or_generate(os.path.join(CACHE_DIR, f"inversion_{TARGET_INVERSION}.parquet"), _load_inversions)
            pc_df = get_cached_or_generate(os.path.join(CACHE_DIR, "pcs_10.parquet"), _load_pcs, gcp_project=gcp_project)
            
            # --- FIX: Rigorously ensure all indexes are strings before joining ---
            demographics_df.index = demographics_df.index.astype(str)
            inversion_df.index = inversion_df.index.astype(str)
            pc_df.index = pc_df.index.astype(str)
            
            core_df = demographics_df.join(inversion_df, how='inner').join(pc_df, how='inner')
            if core_df.empty:
                raise RuntimeError("Core covariate DataFrame is empty after joins. This should not happen with corrected index types.")
            print(f"[Setup]    - Core covariate DataFrame ready. Shape: {core_df.shape}")

            print("[Setup]    - Pre-calculating pan-category case sets for control definitions...")
            category_cache_path = os.path.join(CACHE_DIR, f"pan_category_cases_{cdr_codename}.pkl")
            if os.path.exists(category_cache_path):
                print("  -> Found cache for pan-category cases.")
                category_to_pan_cases = pd.read_pickle(category_cache_path)
            else:
                category_to_pan_cases = {}
                for category, group in pheno_defs_df.groupby('disease_category'):
                    print(f"  -> Querying all cases for category: {category}...")
                    pan_codes = set.union(*group['all_codes'])
                    if pan_codes:
                        q = f"SELECT DISTINCT person_id FROM `{cdr_dataset_id}.condition_occurrence` WHERE condition_source_value IN ({','.join([repr(c) for c in pan_codes])})"
                        category_to_pan_cases[category] = set(bq_client.query(q).to_dataframe()['person_id'].astype(str))
                    else:
                        category_to_pan_cases[category] = set()
                print("  -> Saving pan-category cases to cache.")
                pd.to_pickle(category_to_pan_cases, category_cache_path)

        print(f"\n--- Total Setup Time: {t_setup.duration:.2f}s ---")

        pheno_queue = queue.Queue(maxsize=10)
        fetcher_thread = threading.Thread(target=phenotype_fetcher_worker, args=(pheno_queue, pheno_defs_df, bq_client, cdr_dataset_id, base_ids, category_to_pan_cases, cdr_codename))
        fetcher_thread.start()
        
        results_df = run_phewas_pipeline(core_df, pheno_queue)
        fetcher_thread.join()

        if not results_df.empty:
            print("\n--- PART 3: FINAL RESULTS (sorted by P-value) ---")
            df = results_df.sort_values("P_Value").copy()
            _, p_adj, _, _ = multipletests(df['P_Value'].dropna(), alpha=FDR_ALPHA, method="fdr_bh")
            df.loc[df['P_Value'].notna(), "P_FDR"] = p_adj
            df["Sig_FDR"] = df["P_FDR"] < FDR_ALPHA

            output_filename = f"phewas_results_{TARGET_INVERSION}.csv"
            print(f"\n--- Saving results to '{output_filename}' ---")
            df.to_csv(output_filename, index=False)
            
            out_df = df.copy()
            for col in ['N_Total', 'N_Cases', 'N_Controls']: out_df[col] = out_df[col].map('{:,.0f}'.format)
            out_df['Beta'] = out_df['Beta'].map('{:+.4f}'.format); out_df['OR'] = out_df['OR'].map('{:.3f}'.format)
            out_df['P_Value'] = out_df['P_Value'].map('{:.3e}'.format); out_df['P_FDR'] = out_df['P_FDR'].map('{:.3f}'.format)
            out_df['Sig_FDR'] = out_df['Sig_FDR'].map(lambda x: "✓" if x else "")
            print(out_df.to_string(index=False))

    except (FileNotFoundError, EnvironmentError, RuntimeError, ValueError) as e:
        print(f"\nSCRIPT HALTED DUE TO A CRITICAL ERROR:\n{e}")
    
    script_duration = time.time() - script_start_time
    print("\n" + "=" * 70)
    print(f" Script finished in {script_duration:.2f} seconds.")
    print("=" * 70)

if __name__ == "__main__":
    main()
