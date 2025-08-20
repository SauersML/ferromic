import os
import time
import ast
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from google.cloud import bigquery
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# --- Configuration ---
TARGET_INVERSION = 'chr17-45585160-INV-706887'
TARGET_PHENOTYPES = ['COPD', 'Osteoarthritis']

# Data sources and caching
CACHE_DIR = "./phewas_cache"
# CORRECTED: Updated file path as requested
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"

# Model parameters
NUM_PCS = 10
MIN_CASES_FILTER = 500      # Phenotypes must have at least this many cases
MIN_CONTROLS_FILTER = 500   # Phenotypes must have at least this many controls
FDR_ALPHA = 0.05

# Suppress pandas warnings for cleaner output
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)


class Timer:
    """Context manager for timing blocks of code."""
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


def get_cached_or_generate(cache_path, generation_func, *args, **kwargs):
    """
    Generic caching wrapper. If cache_path exists, load it.
    Otherwise, run generation_func, save the result to cache_path, and return it.
    """
    if os.path.exists(cache_path):
        print(f"  -> Found cache, loading from '{cache_path}'...")
        return pd.read_parquet(cache_path)
    
    print(f"  -> No cache found at '{cache_path}'. Generating data...")
    data = generation_func(*args, **kwargs)
    
    print(f"  -> Saving new cache to '{cache_path}'...")
    data.to_parquet(cache_path)
    return data


def setup_data():
    """
    Handles all data loading, querying, and merging. Caches results aggressively.
    """
    print("\n--- PART 1: DATA SETUP AND LOADING ---")
    
    # --- 1.1: BigQuery Client Setup ---
    with Timer() as t:
        print("\n[1.1] Setting up BigQuery client...")
        try:
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            print(f"  ✓ Using CDR: {cdr_dataset_id} | Project: {gcp_project}")
            cdr_codename = cdr_dataset_id.split('.')[-1]
        except KeyError as e:
            raise EnvironmentError(f"FATAL: Missing environment variable: {e}.") from e
    print(f"  Done in {t.duration:.2f}s")

    # --- 1.2: Base Population ---
    with Timer() as t:
        print("\n[1.2] Fetching base population...")
        persons_df = bq_client.query(f"SELECT person_id FROM `{cdr_dataset_id}.person`").to_dataframe()
        persons_df["person_id"] = persons_df["person_id"].astype(str)
        base_ids = set(persons_df["person_id"])
        print(f"  ✓ Base population size: {len(base_ids):,}")
    print(f"  Done in {t.duration:.2f}s")

    # --- 1.3: Demographics (Age) - Caching this step ---
    with Timer() as t:
        print("\n[1.3] Fetching Demographics (Age)...")
        def _load_demographics():
            query = f"SELECT person_id, year_of_birth FROM `{cdr_dataset_id}.person`"
            df = bq_client.query(query).to_dataframe()
            df['person_id'] = df['person_id'].astype(str)
            current_year = datetime.now().year
            df['AGE'] = current_year - df['year_of_birth']
            df['AGE_sq'] = df['AGE'] ** 2
            return df[['person_id', 'AGE', 'AGE_sq']].set_index('person_id')

        demographics_cache_path = os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet")
        demographics_df = get_cached_or_generate(demographics_cache_path, _load_demographics)
        print(f"  ✓ Demographics loaded for {len(demographics_df):,} individuals.")
    print(f"  Done in {t.duration:.2f}s")
    
    # --- 1.4: Phenotypes (Cached) ---
    with Timer() as t:
        print("\n[1.4] Fetching phenotype data from BigQuery...")
        def _load_phenotypes():
            # CORRECTED: Moved definitions and print statements INSIDE the cached function
            # This ensures progress is shown when the cache is being built.
            print("    (Building phenotype cache from BigQuery...)")
            phenotype_definitions = {
                "COPD": {"icd_patterns": ["J44.%"]},
                "Osteoarthritis": {"icd_patterns": ["M15.%", "M16.%", "M17.%", "M18.%", "M19.%"]},
            }
            case_sets = {}
            for pheno, meta in phenotype_definitions.items():
                print(f"    - Querying cases for {pheno}...")
                def build_icd_sql_condition(patterns, alias="co"):
                    parts = [f"LOWER({alias}.condition_source_value) LIKE LOWER('{p}')" for p in patterns]
                    return "(" + " OR ".join(parts) + ")"
                q = f"""
                    SELECT DISTINCT person_id FROM `{cdr_dataset_id}.condition_occurrence` co
                    WHERE {build_icd_sql_condition(meta['icd_patterns'], alias="co")}
                """
                df_ids = bq_client.query(q).to_dataframe()
                df_ids["person_id"] = df_ids["person_id"].astype(str)
                case_sets[pheno] = set(df_ids["person_id"]).intersection(base_ids)
                print(f"      -> Found {len(case_sets[pheno]):,} cases.")
            
            pheno_df = pd.DataFrame(index=pd.Index(sorted(list(base_ids)), name="person_id", dtype=str))
            for pheno_name in TARGET_PHENOTYPES:
                case_ids = case_sets[pheno_name]
                pheno_df[f"is_case_{pheno_name}"] = pheno_df.index.isin(case_ids).astype(int)
                pheno_df[f"is_control_{pheno_name}"] = (~pheno_df.index.isin(case_ids)).astype(int)
            return pheno_df

        phenotype_cache_path = os.path.join(CACHE_DIR, f"phenotypes_{cdr_codename}.parquet")
        phenotype_df = get_cached_or_generate(phenotype_cache_path, _load_phenotypes)
    print(f"  Phenotype setup done in {t.duration:.2f}s")
    
    # --- 1.5: Inversion Dosages (Cached) ---
    with Timer() as t:
        print(f"\n[1.5] Loading inversion dosages for '{TARGET_INVERSION}'...")
        def _load_inversions():
            try:
                df = pd.read_csv(INVERSION_DOSAGES_FILE, sep="\t", index_col="SampleID", usecols=["SampleID", TARGET_INVERSION])
                df.index = df.index.astype(str)
                return df
            except FileNotFoundError:
                raise FileNotFoundError(f"Inversion dosage file not found: {INVERSION_DOSAGES_FILE}")
            except ValueError as e:
                raise ValueError(f"'{TARGET_INVERSION}' not found in dosage file. Error: {e}")

        inversion_cache_path = os.path.join(CACHE_DIR, "inversion_dosages.parquet")
        inversion_dosages_df = get_cached_or_generate(inversion_cache_path, _load_inversions)
    print(f"  Done in {t.duration:.2f}s")

    # --- 1.6: Genetic PCs (Cached) ---
    with Timer() as t:
        print("\n[1.6] Loading genetic principal components (PCs)...")
        def _load_pcs():
            try:
                raw_pcs = pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True})
                pc_cols_all = [f"PC{i}" for i in range(1, 17)]
                pc_mat = pd.DataFrame(raw_pcs["pca_features"].apply(lambda s: ast.literal_eval(s) if pd.notna(s) else [np.nan]*16).tolist(), columns=pc_cols_all)
                pc_df = pc_mat.assign(person_id=raw_pcs["research_id"].astype(str)).set_index("person_id")
                return pc_df[[f'PC{i}' for i in range(1, NUM_PCS + 1)]]
            except Exception as e:
                raise RuntimeError(f"Error loading PCs: {e}") from e

        pcs_cache_path = os.path.join(CACHE_DIR, "genetic_pcs.parquet")
        pc_df = get_cached_or_generate(pcs_cache_path, _load_pcs)
    print(f"  Done in {t.duration:.2f}s")

    # --- 1.7: Merge to Master DataFrame ---
    with Timer() as t:
        print("\n[1.7] Merging to master analysis DataFrame...")
        master_df = phenotype_df.join(demographics_df, how="inner").join(inversion_dosages_df, how="inner").join(pc_df, how="inner")
        if master_df.empty:
            raise ValueError("Master DataFrame is empty after merging. Check person_id alignment.")
        print(f"  ✓ MASTER DataFrame ready. Shape: {master_df.shape}")
    print(f"  Done in {t.duration:.2f}s")
    
    return master_df


def run_phewas(master_df):
    """
    Runs association tests with Age, Age^2, and PC controls.
    Filters out phenotypes with insufficient cases or controls.
    """
    print("\n--- PART 2: RUNNING ASSOCIATION MODELS ---")
    
    with Timer() as t:
        print(f"Testing inversion '{TARGET_INVERSION}' against {len(TARGET_PHENOTYPES)} phenotypes.")
        
        pc_cols_to_use = [f'PC{i}' for i in range(1, NUM_PCS + 1)]
        covariate_cols = [TARGET_INVERSION] + pc_cols_to_use + ['AGE', 'AGE_sq']
        X_full = master_df[covariate_cols].copy()
        X_full = sm.add_constant(X_full, prepend=True)
        
        all_results = []
        
        for phenotype in TARGET_PHENOTYPES:
            case_col = f'is_case_{phenotype}'
            
            # Everyone with a defined phenotype status is included initially
            y = master_df[case_col]
            X = X_full

            combined = pd.concat([y, X], axis=1).dropna()
            y_clean = combined[case_col]
            X_clean = combined.drop(columns=[case_col])

            n_total, n_cases = len(y_clean), int(y_clean.sum())
            n_ctrls = n_total - n_cases

            # IMPROVEMENT: A single, clear filtering step
            if n_cases < MIN_CASES_FILTER or n_ctrls < MIN_CONTROLS_FILTER:
                print(f"[SKIP] {phenotype:<15s} | Fails count filter (cases={n_cases:,}, controls={n_ctrls:,}).")
                continue
            if X_clean[TARGET_INVERSION].nunique() < 2:
                 print(f"[SKIP] {phenotype:<15s} | Inversion dosage has no variation in this subset.")
                 continue

            try:
                model = sm.Logit(y_clean, X_clean)
                fit = model.fit(disp=0, maxiter=200)

                beta, pval = fit.params[TARGET_INVERSION], fit.pvalues[TARGET_INVERSION]
                OR = np.exp(beta)
                
                print(f"[OK]   {phenotype:<15s} | N={n_total:,}, cases={n_cases:,} | beta={beta:+.4f}, OR={OR:.3f}, p={pval:.2e}")
                
                all_results.append({
                    "Phenotype": phenotype, "N_Total": n_total, "N_Cases": n_cases,
                    "Beta": beta, "OR": OR, "P_Value": pval,
                })
            except (PerfectSeparationError, np.linalg.LinAlgError):
                print(f"[FAIL] {phenotype:<15s} | Perfect separation or singular matrix.")
            except Exception as e:
                print(f"[FAIL] {phenotype:<15s} | Error: {str(e)[:100]}")

    print(f"--- Association analysis done in {t.duration:.2f}s ---")
    return pd.DataFrame(all_results)


def display_results(results_df):
    """Formats and prints the final results with FDR correction."""
    print("\n--- PART 3: FINAL RESULTS ---")
    
    if results_df.empty:
        print("No models were run or all were skipped. Exiting results display.")
        return
        
    df = results_df.copy().sort_values("P_Value")

    p_values = df["P_Value"].dropna()
    if not p_values.empty:
        _, p_adj, _, _ = multipletests(p_values, alpha=FDR_ALPHA, method="fdr_bh")
        df.loc[p_values.index, "P_FDR"] = p_adj
        df["Sig_FDR"] = df["P_FDR"] < FDR_ALPHA
    else:
        df["P_FDR"], df["Sig_FDR"] = np.nan, False

    out_df = df.copy()
    out_df['N_Total'] = out_df['N_Total'].map('{:,.0f}'.format)
    out_df['N_Cases'] = out_df['N_Cases'].map('{:,.0f}'.format)
    out_df['Beta'] = out_df['Beta'].map('{:+.4f}'.format)
    out_df['OR'] = out_df['OR'].map('{:.3f}'.format)
    out_df['P_Value'] = out_df['P_Value'].map('{:.3e}'.format)
    out_df['P_FDR'] = out_df['P_FDR'].map('{:.3f}'.format)
    out_df['Sig_FDR'] = out_df['Sig_FDR'].map(lambda x: "✓" if x else "")

    print(f"\nAssociation results for inversion: {TARGET_INVERSION}")
    print(out_df.to_string(index=False))


def main():
    """Main execution function."""
    script_start_time = time.time()
    print("=" * 60)
    print(" Starting PheWAS Script with Age/PC Controls & Filtering")
    print("=" * 60)
    
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        with Timer() as t_setup:
            master_df = setup_data()
        print(f"\n--- Total Data Setup Time: {t_setup.duration:.2f}s ---")
        
        results_df = run_phewas(master_df)
        
        display_results(results_df)

    except (FileNotFoundError, EnvironmentError, RuntimeError, ValueError) as e:
        print(f"\nSCRIPT HALTED DUE TO A CRITICAL ERROR:\n{e}")
    
    script_duration = time.time() - script_start_time
    print("\n" + "=" * 60)
    print(f" Script finished in {script_duration:.2f} seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
