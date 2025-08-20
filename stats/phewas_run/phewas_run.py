import os
import re
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
PHENOTYPE_DEFINITIONS_URL = "https://github.com/SauersML/ferromic/raw/refs/heads/main/data/significant_heritability_diseases.tsv"

# Data sources and caching
CACHE_DIR = "./phewas_cache"
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"

# Model parameters
NUM_PCS = 10
MIN_CASES_FILTER = 1000     # NEW: Require at least 1,000 cases
MIN_CONTROLS_FILTER = 500   # Controls are now category-wide
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
    """Generic caching wrapper."""
    if os.path.exists(cache_path):
        print(f"  -> Found cache, loading from '{cache_path}'...")
        return pd.read_parquet(cache_path)
    
    print(f"  -> No cache found at '{cache_path}'. Generating data...")
    data = generation_func(*args, **kwargs)
    
    print(f"  -> Saving new cache to '{cache_path}'...")
    data.to_parquet(cache_path)
    return data


def sanitize_name(name):
    """Cleans a disease name to be a valid Python/DataFrame identifier."""
    name = re.sub(r'[\*\(\)]', '', name)  # Remove special characters
    name = re.sub(r'\s+', '_', name.strip()) # Replace whitespace with underscore
    return name


def parse_icd_codes(code_string):
    """Parses a semi-colon delimited string of ICD codes into a clean set."""
    if pd.isna(code_string) or not isinstance(code_string, str):
        return set()
    # Split by semicolon, strip quotes and whitespace, filter out empty strings
    codes = {code.strip().strip('"') for code in code_string.split(';') if code.strip()}
    return codes


def setup_data():
    """Handles all data loading, querying, and merging."""
    print("\n--- PART 1: DATA SETUP AND LOADING ---")
    
    # --- 1.1: Load Phenotype Definitions from URL ---
    with Timer() as t:
        print("\n[1.1] Loading and processing phenotype definitions from URL...")
        try:
            pheno_defs_df = pd.read_csv(PHENOTYPE_DEFINITIONS_URL, sep='\t')
            pheno_defs_df['sanitized_name'] = pheno_defs_df['disease'].apply(sanitize_name)
            pheno_defs_df['icd9_set'] = pheno_defs_df['icd9_codes'].apply(parse_icd_codes)
            pheno_defs_df['icd10_set'] = pheno_defs_df['icd10_codes'].apply(parse_icd_codes)
            pheno_defs_df['all_codes'] = pheno_defs_df.apply(lambda row: row['icd9_set'].union(row['icd10_set']), axis=1)
            print(f"  ✓ Loaded and processed {len(pheno_defs_df)} phenotype definitions.")
        except Exception as e:
            raise RuntimeError(f"FATAL: Could not load or process phenotype definitions from URL. Error: {e}") from e
    print(f"  Done in {t.duration:.2f}s")
    
    # --- 1.2: BigQuery Client Setup ---
    with Timer() as t:
        print("\n[1.2] Setting up BigQuery client...")
        try:
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            cdr_codename = cdr_dataset_id.split('.')[-1]
            print(f"  ✓ Using CDR: {cdr_dataset_id} | Project: {gcp_project}")
        except KeyError as e:
            raise EnvironmentError(f"FATAL: Missing environment variable: {e}.") from e
    print(f"  Done in {t.duration:.2f}s")

    # --- 1.3: Base Population ---
    with Timer() as t:
        print("\n[1.3] Fetching base population...")
        persons_df = bq_client.query(f"SELECT person_id FROM `{cdr_dataset_id}.person`").to_dataframe()
        persons_df["person_id"] = persons_df["person_id"].astype(str)
        base_ids = set(persons_df["person_id"])
        print(f"  ✓ Base population size: {len(base_ids):,}")
    print(f"  Done in {t.duration:.2f}s")
    
    # --- 1.4: Demographics (Age) ---
    with Timer() as t:
        print("\n[1.4] Fetching Demographics (Age)...")
        def _load_demographics():
            query = f"SELECT person_id, year_of_birth FROM `{cdr_dataset_id}.person`"
            df = bq_client.query(query).to_dataframe()
            df['person_id'] = df['person_id'].astype(str)
            df['AGE'] = datetime.now().year - df['year_of_birth']
            df['AGE_sq'] = df['AGE'] ** 2
            return df[['person_id', 'AGE', 'AGE_sq']].set_index('person_id')
        demographics_cache_path = os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet")
        demographics_df = get_cached_or_generate(demographics_cache_path, _load_demographics)
    print(f"  Done in {t.duration:.2f}s")
    
    # --- 1.5: Phenotypes (with Category-Exclusive Controls) ---
    with Timer() as t:
        print("\n[1.5] Fetching phenotype data from BigQuery with category-exclusive controls...")
        def _load_phenotypes_with_logic(pheno_defs, bq_c, cdr_id, base_pids):
            print("    (Building phenotype cache from BigQuery...)")
            
            def build_icd_sql_condition(codes):
                # Format codes for a SQL IN clause
                formatted_codes = [f"'{code}'" for code in codes if code]
                if not formatted_codes: return "FALSE"
                # Use both source_value and standard_code_1 for broader matching
                return f"""(co.condition_source_value IN ({','.join(formatted_codes)}) OR
                             std.concept_code IN ({','.join(formatted_codes)}))"""

            def query_person_ids(icd_codes):
                if not icd_codes: return set()
                q = f"""
                    SELECT DISTINCT co.person_id
                    FROM `{cdr_id}.condition_occurrence` co
                    LEFT JOIN `{cdr_id}.concept` std ON co.condition_source_concept_id = std.concept_id
                    WHERE {build_icd_sql_condition(icd_codes)}
                """
                df_ids = bq_c.query(q).to_dataframe()
                return set(df_ids["person_id"].astype(str)).intersection(base_pids)

            # Step 1: Get pan-category cases (for defining controls)
            category_to_pan_cases = {}
            for category, group in pheno_defs.groupby('disease_category'):
                print(f"    - Querying all cases for category: {category}...")
                pan_category_codes = set.union(*group['all_codes'])
                category_to_pan_cases[category] = query_person_ids(pan_category_codes)
                print(f"      -> Found {len(category_to_pan_cases[category]):,} total cases in category.")

            # Step 2: Get specific cases for each disease
            disease_to_specific_cases = {}
            for _, row in pheno_defs.iterrows():
                sanitized_name = row['sanitized_name']
                print(f"    - Querying specific cases for: {sanitized_name}...")
                disease_to_specific_cases[sanitized_name] = query_person_ids(row['all_codes'])
                print(f"      -> Found {len(disease_to_specific_cases[sanitized_name]):,} cases.")
            
            # Step 3: Assemble the final phenotype DataFrame
            pheno_df = pd.DataFrame(index=pd.Index(sorted(list(base_pids)), name="person_id", dtype=str))
            for _, row in pheno_defs.iterrows():
                s_name, category = row['sanitized_name'], row['disease_category']
                cases = disease_to_specific_cases[s_name]
                # Controls are everyone in the base pop MINUS all cases from the same category
                controls = base_pids - category_to_pan_cases[category]
                pheno_df[f'is_case_{s_name}'] = pheno_df.index.isin(cases).astype(int)
                pheno_df[f'is_control_{s_name}'] = pheno_df.index.isin(controls).astype(int)
            return pheno_df
        
        phenotype_cache_path = os.path.join(CACHE_DIR, f"phenotypes_dynamic_{cdr_codename}.parquet")
        phenotype_df = get_cached_or_generate(
            phenotype_cache_path, _load_phenotypes_with_logic,
            pheno_defs=pheno_defs_df, bq_c=bq_client, cdr_id=cdr_dataset_id, base_pids=base_ids
        )
    print(f"  Phenotype setup done in {t.duration:.2f}s")
    
    # --- 1.6-1.8: Load Covariates and Merge ---
    # (No changes to these steps, just renumbering)
    print("\n[1.6] Loading inversion dosages...")
    inversion_dosages_df = get_cached_or_generate(
        os.path.join(CACHE_DIR, "inversion_dosages.parquet"),
        lambda: pd.read_csv(INVERSION_DOSAGES_FILE, sep="\t", index_col="SampleID", usecols=["SampleID", TARGET_INVERSION]).set_index(pd.Index(pd.read_csv(INVERSION_DOSAGES_FILE, sep="\t")['SampleID'].astype(str)))
    )
    print("\n[1.7] Loading genetic PCs...")
    pc_df = get_cached_or_generate(
        os.path.join(CACHE_DIR, "genetic_pcs.parquet"),
        lambda: pd.DataFrame(pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True})["pca_features"].apply(lambda s: ast.literal_eval(s) if pd.notna(s) else [np.nan]*16).tolist(), columns=[f"PC{i}" for i in range(1, 17)]).assign(person_id=pd.read_csv(PCS_URI, sep="\t")["research_id"].astype(str)).set_index("person_id")[[f'PC{i}' for i in range(1, NUM_PCS + 1)]]
    )

    print("\n[1.8] Merging to master analysis DataFrame...")
    master_df = phenotype_df.join(demographics_df, how="inner").join(inversion_dosages_df, how="inner").join(pc_df, how="inner")
    if master_df.empty: raise ValueError("Master DataFrame is empty after merging.")
    print(f"  ✓ MASTER DataFrame ready. Shape: {master_df.shape}")
    
    return master_df, pheno_defs_df


def run_phewas(master_df, pheno_defs_df):
    """
    Runs association tests with Age, Age^2, and PC controls, applying new filters.
    """
    print("\n--- PART 2: RUNNING ASSOCIATION MODELS ---")
    
    phenotypes_to_test = pheno_defs_df['sanitized_name'].tolist()
    print(f"Preparing to test inversion '{TARGET_INVERSION}' against up to {len(phenotypes_to_test)} phenotypes.")
    
    with Timer() as t:
        pc_cols = [f'PC{i}' for i in range(1, NUM_PCS + 1)]
        covariate_cols = [TARGET_INVERSION] + pc_cols + ['AGE', 'AGE_sq']
        X_full = sm.add_constant(master_df[covariate_cols].copy(), prepend=True)
        
        all_results = []
        eligible_phenos = 0
        
        for phenotype in phenotypes_to_test:
            case_col, ctrl_col = f'is_case_{phenotype}', f'is_control_{phenotype}'
            
            # Define analysis subset based on having either case or control status
            analysis_mask = (master_df[case_col] == 1) | (master_df[ctrl_col] == 1)
            y = master_df.loc[analysis_mask, case_col]
            X = X_full.loc[analysis_mask]

            # Drop rows with any NAs across outcome or predictors for this specific analysis
            combined = pd.concat([y, X], axis=1).dropna()
            y_clean, X_clean = combined[case_col], combined.drop(columns=[case_col])

            n_total, n_cases, n_ctrls = len(y_clean), int(y_clean.sum()), len(y_clean) - int(y_clean.sum())

            # Apply the new, stricter filtering
            if n_cases < MIN_CASES_FILTER or n_ctrls < MIN_CONTROLS_FILTER:
                continue # Silently skip phenotypes that don't meet the count threshold
            
            eligible_phenos += 1
            if X_clean[TARGET_INVERSION].nunique() < 2:
                 print(f"[SKIP] {phenotype:<40s} | Inversion dosage has no variation in this subset.")
                 continue

            try:
                model = sm.Logit(y_clean, X_clean)
                fit = model.fit(disp=0, maxiter=200)
                beta, pval = fit.params[TARGET_INVERSION], fit.pvalues[TARGET_INVERSION]
                print(f"[OK]   {phenotype:<40s} | N={n_total:,}, cases={n_cases:,} | beta={beta:+.4f}, OR={np.exp(beta):.3f}, p={pval:.2e}")
                all_results.append({ "Phenotype": phenotype, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls, "Beta": beta, "OR": np.exp(beta), "P_Value": pval })
            except Exception as e:
                print(f"[FAIL] {phenotype:<40s} | Error: {str(e)[:100]}")

    print(f"\nTested {eligible_phenos} eligible phenotypes.")
    print(f"--- Association analysis done in {t.duration:.2f}s ---")
    return pd.DataFrame(all_results)


def display_results(results_df):
    """Formats and prints the final results with FDR correction."""
    print("\n--- PART 3: FINAL RESULTS (sorted by P-value) ---")
    
    if results_df.empty:
        print("No models were run or none passed filters. Exiting results display.")
        return
        
    df = results_df.sort_values("P_Value").copy()
    
    if 'P_Value' in df.columns and df['P_Value'].notna().any():
        _, p_adj, _, _ = multipletests(df['P_Value'].dropna(), alpha=FDR_ALPHA, method="fdr_bh")
        df.loc[df['P_Value'].notna(), "P_FDR"] = p_adj
        df["Sig_FDR"] = df["P_FDR"] < FDR_ALPHA
    else:
        df["P_FDR"], df["Sig_FDR"] = np.nan, False

    out_df = df.copy()
    for col in ['N_Total', 'N_Cases', 'N_Controls']:
        out_df[col] = out_df[col].map('{:,.0f}'.format)
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
    print("=" * 70)
    print(" Starting Dynamic PheWAS with Category-Exclusive Controls")
    print("=" * 70)
    
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        with Timer() as t_setup:
            master_df, pheno_defs_df = setup_data()
        print(f"\n--- Total Data Setup Time: {t_setup.duration:.2f}s ---")
        
        results_df = run_phewas(master_df, pheno_defs_df)
        
        display_results(results_df)

    except (FileNotFoundError, EnvironmentError, RuntimeError, ValueError) as e:
        print(f"\nSCRIPT HALTED DUE TO A CRITICAL ERROR:\n{e}")
    
    script_duration = time.time() - script_start_time
    print("\n" + "=" * 70)
    print(f" Script finished in {script_duration:.2f} seconds.")
    print("=" * 70)

if __name__ == "__main__":
    main()
