# example so far below, will get updated later.

import os
import time
import ast
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from google.cloud import bigquery
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# --- Configuration ---
TARGET_INVERSION = 'chr17-45585160-INV-706887'
TARGET_PHENOTYPES = ['COPD', 'Osteoarthritis']

# Data sources
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"

# Model parameters
NUM_PCS = 10
MIN_MODEL_N = 100
MIN_CASES_MODEL = 20
FDR_ALPHA = 0.05

# Suppress pandas chained assignment warning
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_data():
    """
    Handles all data loading, querying, and merging.
    This corresponds to CELL 1 in the original notebook.
    """
    print("\n--- PART 1: DATA SETUP AND LOADING ---")
    t_start = time.time()

    # ============================================================================================================
    # 1.1) Phenotype definitions
    # ============================================================================================================
    print("\n[1.1] Defining phenotypes...")
    
    # We define the full dictionary and then filter it to keep the structure clear.
    ALL_PHENOTYPES_DEFINITIONS = {
        "COPD": {
            "group": "PULMONARY",
            "icd_patterns": ["J44.%"],
            "source": "ICD-10 J44.x COPD"
        },
        "Osteoarthritis": {
            "group": "MUSCULOSKELETAL",
            "icd_patterns": ["M15.%", "M16.%", "M17.%", "M18.%", "M19.%"],
            "source": "ICD-10 M15–M19 Osteoarthritis"
        }
    }
    
    # Filter for only the target phenotypes
    PHENOTYPES = {name: data for name, data in ALL_PHENOTYPES_DEFINITIONS.items() if name in TARGET_PHENOTYPES}
    print(f"  ✓ Will analyze {len(PHENOTYPES)} phenotypes: {', '.join(PHENOTYPES.keys())}")


    # ============================================================================================================
    # 1.2) Environment / BigQuery client
    # ============================================================================================================
    print("\n[1.2] Setting up BigQuery client...")
    try:
        cdr_dataset_id = os.environ["WORKSPACE_CDR"]
        gcp_project = os.environ["GOOGLE_PROJECT"]
        bq_client = bigquery.Client(project=gcp_project)
        print(f"  ✓ Using CDR: {cdr_dataset_id} | Project: {gcp_project}")
    except KeyError as e:
        raise EnvironmentError(
            f"FATAL: Required environment variable not set: {e}. "
            "Please set GOOGLE_PROJECT and WORKSPACE_CDR."
        ) from e
    except Exception as e:
        raise RuntimeError(f"FATAL: Could not initialize BigQuery client — {e}") from e

    # ============================================================================================================
    # 1.3) Fetch base population and phenotypes from BigQuery
    # ============================================================================================================
    print("\n[1.3] Fetching base population and phenotype cases...")
    persons_df = bq_client.query(f"SELECT person_id FROM `{cdr_dataset_id}.person`").to_dataframe()
    persons_df["person_id"] = persons_df["person_id"].astype(str)
    base_ids = set(persons_df["person_id"])
    print(f"  ✓ Base population size: {len(base_ids):,}")

    def build_icd_sql_condition(patterns, alias="co"):
        parts = [f"LOWER({alias}.condition_source_value) LIKE LOWER('{p}')" for p in patterns]
        return "(" + " OR ".join(parts) + ")"

    case_sets = {}
    control_sets = {}
    for pheno, meta in PHENOTYPES.items():
        print(f"  - Querying cases for {pheno}...")
        q = f"""
            SELECT DISTINCT person_id
            FROM `{cdr_dataset_id}.condition_occurrence` co
            WHERE {build_icd_sql_condition(meta['icd_patterns'], alias="co")}
        """
        df_ids = bq_client.query(q).to_dataframe()
        df_ids["person_id"] = df_ids["person_id"].astype(str)
        case_ids = set(df_ids["person_id"]).intersection(base_ids)
        case_sets[pheno] = case_ids
        
        # Controls are everyone in the base population who is not a case for this specific phenotype.
        control_sets[pheno] = base_ids - case_ids
        print(f"    - Found {len(case_ids):,} cases and {len(control_sets[pheno]):,} potential controls.")

    # ============================================================================================================
    # 1.4) Assemble phenotype indicator matrix
    # ============================================================================================================
    print("\n[1.4] Building phenotype indicator DataFrame...")
    phenotype_df = pd.DataFrame(index=pd.Index(sorted(list(base_ids)), name="person_id", dtype=str))
    
    for pheno in PHENOTYPES.keys():
        phenotype_df[f"is_case_{pheno}"] = 0
        phenotype_df[f"is_control_{pheno}"] = 0
        phenotype_df.loc[list(case_sets[pheno]), f"is_case_{pheno}"] = 1
        phenotype_df.loc[list(control_sets[pheno]), f"is_control_{pheno}"] = 1

    # Print summary table
    summary_rows = [
        {
            "Phenotype": pheno,
            "N_Cases": int(phenotype_df[f"is_case_{pheno}"].sum()),
            "N_Controls": int(phenotype_df[f"is_control_{pheno}"].sum())
        }
        for pheno in PHENOTYPES.keys()
    ]
    summary_df = pd.DataFrame(summary_rows)
    print("  ✓ Phenotype Counts:")
    print(summary_df.to_string(index=False))
    
    # ============================================================================================================
    # 1.5) Load inversion dosages
    # ============================================================================================================
    print(f"\n[1.5] Loading inversion dosages for '{TARGET_INVERSION}'...")
    try:
        inversion_dosages_df = pd.read_csv(
            INVERSION_DOSAGES_FILE, 
            sep="\t", 
            index_col="SampleID", 
            usecols=["SampleID", TARGET_INVERSION]
        )
        inversion_dosages_df.index = inversion_dosages_df.index.astype(str)
        print(f"  ✓ Inversion file loaded: {inversion_dosages_df.shape[0]:,} samples")
    except FileNotFoundError:
        raise FileNotFoundError(f"Inversion dosage file not found: {INVERSION_DOSAGES_FILE}")
    except ValueError as e:
        raise ValueError(f"'{TARGET_INVERSION}' not found in the dosage file columns. Error: {e}")

    # ============================================================================================================
    # 1.6) Load genetic PCs
    # ============================================================================================================
    print("\n[1.6] Loading genetic principal components (PCs)...")
    try:
        raw_pcs = pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True})
        pc_cols_all = [f"PC{i}" for i in range(1, 17)]
        pc_mat = pd.DataFrame(
            raw_pcs["pca_features"].apply(lambda s: ast.literal_eval(s) if pd.notna(s) else [np.nan]*16).tolist(),
            columns=pc_cols_all
        )
        pc_df = pc_mat.assign(person_id=raw_pcs["research_id"].astype(str)).set_index("person_id")
        pc_df = pc_df[[f'PC{i}' for i in range(1, NUM_PCS + 1)]] # Keep only the required PCs
        print(f"  ✓ PCs loaded: {pc_df.shape[1]} PCs x {pc_df.shape[0]:,} samples")
    except Exception as e:
        raise RuntimeError(f"Error loading PCs: {e}") from e

    # ============================================================================================================
    # 1.7) Merge to master analysis frame
    # ============================================================================================================
    print("\n[1.7] Merging to master analysis DataFrame...")
    master_df = (
        phenotype_df
        .join(inversion_dosages_df, how="inner")
        .join(pc_df, how="inner")
    )
    if master_df.empty:
        raise ValueError("Master DataFrame is empty after merging. Check person_id alignment and inputs.")
    
    print(f"  ✓ MASTER DataFrame ready. Shape: {master_df.shape}")
    print(f"--- Data setup done in {time.time() - t_start:.2f}s ---")
    return master_df


def run_phewas(master_df):
    """
    Runs the association tests for the target phenotypes and inversion.
    This corresponds to CELL 2 in the original notebook.
    """
    print("\n--- PART 2: RUNNING ASSOCIATION MODELS ---")
    t_start = time.time()
    
    # --- Formula parts ---
    pc_cols_to_use = [f'PC{i}' for i in range(1, NUM_PCS + 1)]
    pc_formula_part = " + ".join(pc_cols_to_use)
    
    print(f"Testing inversion '{TARGET_INVERSION}' against {len(TARGET_PHENOTYPES)} phenotypes.")
    
    # --- Main loop ---
    all_results = []
    
    for phenotype in TARGET_PHENOTYPES:
        case_col = f'is_case_{phenotype}'
        ctrl_col = f'is_control_{phenotype}'
        
        # Subset with labeled case/control and drop rows with any NAs in predictors/outcome
        sub = master_df[(master_df[case_col] == 1) | (master_df[ctrl_col] == 1)]
        keep_cols = [case_col, TARGET_INVERSION, *pc_cols_to_use]
        sub = sub.dropna(subset=keep_cols)

        # Data sufficiency checks
        n_total = len(sub)
        n_cases = int(sub[case_col].sum())
        n_ctrls = n_total - n_cases

        if n_total < MIN_MODEL_N:
            print(f"[SKIP] {phenotype}: N={n_total} is less than minimum {MIN_MODEL_N}.")
            continue
        if n_cases < MIN_CASES_MODEL:
            print(f"[SKIP] {phenotype}: Cases={n_cases} is less than minimum {MIN_CASES_MODEL}.")
            continue
        if sub[case_col].nunique() < 2:
            print(f"[SKIP] {phenotype}: Outcome is single-class after NA drop.")
            continue
        if sub[TARGET_INVERSION].nunique(dropna=True) < 2:
            print(f"[SKIP] {phenotype}: Predictor (inversion dosage) has no variation.")
            continue
            
        # Build formula; use patsy Q("...") to safely reference the inversion column name
        formula = f"{case_col} ~ Q(\"{TARGET_INVERSION}\") + {pc_formula_part}"
        
        try:
            fit = smf.logit(formula, data=sub).fit(disp=0, maxiter=200)

            param_name = f'Q("{TARGET_INVERSION}")'
            if param_name not in fit.params.index:
                 raise RuntimeError(f"Parameter for inversion not found in fit.")
            
            beta = float(fit.params[param_name])
            pval = float(fit.pvalues[param_name])
            OR = float(np.exp(beta))
            
            print(
                f"[OK]   {phenotype:<15s} | N={n_total}, cases={n_cases} | "
                f"beta={beta:+.4f}, OR={OR:.3f}, p={pval:.2e}"
            )
            
            all_results.append({
                "Inversion": TARGET_INVERSION, "Phenotype": phenotype,
                "N_Total": n_total, "N_Cases": n_cases,
                "Beta": beta, "OR": OR, "P_Value": pval,
                "Converged": True, "Error": None
            })
            
        except PerfectSeparationError:
            msg = "Perfect separation detected."
            print(f"[FAIL] {phenotype:<15s} | {msg}")
            all_results.append({
                "Inversion": TARGET_INVERSION, "Phenotype": phenotype,
                "N_Total": n_total, "N_Cases": n_cases,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan,
                "Converged": False, "Error": msg
            })
        except Exception as e:
            msg = str(e)[:100] # Truncate long error messages
            print(f"[FAIL] {phenotype:<15s} | {msg}")
            all_results.append({
                "Inversion": TARGET_INVERSION, "Phenotype": phenotype,
                "N_Total": n_total, "N_Cases": n_cases,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan,
                "Converged": False, "Error": msg
            })

    results_df = pd.DataFrame(all_results)
    print(f"--- Association analysis done in {time.time() - t_start:.2f}s ---")
    return results_df


def display_results(results_df):
    """
    Formats and prints the final results with FDR correction.
    This corresponds to CELL 3 in the original notebook.
    """
    print("\n--- PART 3: FINAL RESULTS ---")
    
    if results_df.empty:
        print("No results were generated. Exiting.")
        return
        
    df = results_df.copy()

    # --- FDR Correction ---
    # Note: FDR over only 2 tests is not very meaningful but included for completeness.
    p_values = df["P_Value"].dropna()
    if not p_values.empty:
        reject, p_adj, _, _ = multipletests(p_values, alpha=FDR_ALPHA, method="fdr_bh")
        df["P_FDR"] = np.nan
        df.loc[p_values.index, "P_FDR"] = p_adj
        df["Sig_FDR"] = False
        df.loc[p_values.index, "Sig_FDR"] = reject
    else:
        df["P_FDR"] = np.nan
        df["Sig_FDR"] = False

    # --- Format for Printing ---
    show_cols = [
        "Phenotype", "N_Total", "N_Cases", "Beta", "OR", "P_Value", "P_FDR", "Sig_FDR"
    ]
    
    # Create a formatted copy for display
    out_df = df.reindex(columns=show_cols).sort_values("P_Value").reset_index(drop=True)
    out_df['N_Total'] = out_df['N_Total'].map('{:,.0f}'.format)
    out_df['N_Cases'] = out_df['N_Cases'].map('{:,.0f}'.format)
    out_df['Beta'] = out_df['Beta'].map('{:+.4f}'.format)
    out_df['OR'] = out_df['OR'].map('{:.3f}'.format)
    out_df['P_Value'] = out_df['P_Value'].map('{:.3e}'.format)
    out_df['P_FDR'] = out_df['P_FDR'].map('{:.3f}'.format)
    out_df['Sig_FDR'] = out_df['Sig_FDR'].map(lambda x: "✓" if x else "")

    print("\nAssociation results for inversion:", TARGET_INVERSION)
    print(out_df.to_string(index=False))


def main():
    """Main execution function."""
    script_start_time = time.time()
    print("=====================================================")
    print(" Starting Simplified PheWAS Script ")
    print("=====================================================")
    
    try:
        # Part 1: Load and prepare all data
        master_df = setup_data()
        
        # Part 2: Run the logistic regression models
        results_df = run_phewas(master_df)
        
        # Part 3: Display the final, formatted results
        display_results(results_df)

    except (FileNotFoundError, EnvironmentError, RuntimeError, ValueError) as e:
        print(f"\nSCRIPT HALTED DUE TO AN ERROR:\n{e}")
    
    script_duration = time.time() - script_start_time
    print("\n=====================================================")
    print(f" Script finished in {script_duration:.2f} seconds.")
    print("=====================================================")


if __name__ == "__main__":
    main()
