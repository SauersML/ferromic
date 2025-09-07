import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from datetime import datetime
import time
import warnings
import gc
import threading
import queue
import faulthandler
import sys
import traceback

import numpy as np
import pandas as pd
import statsmodels.api as sm
from google.cloud import bigquery
from statsmodels.stats.multitest import multipletests
from scipy import stats

import iox as io
import pheno
import pipes
import models

faulthandler.enable()

def _global_excepthook(exc_type, exc, tb):
    """
    Uncaught exception hook that prints a full stack trace immediately across threads and subprocesses.
    """
    print("[TRACEBACK] Uncaught exception:", flush=True)
    traceback.print_exception(exc_type, exc, tb)
    sys.stderr.flush()

sys.excepthook = _global_excepthook

def _thread_excepthook(args):
    _global_excepthook(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = _thread_excepthook

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# --- Configuration ---
TARGET_INVERSIONS = ['chr17-45585160-INV-706887']
PHENOTYPE_DEFINITIONS_URL = "https://github.com/SauersML/ferromic/raw/refs/heads/main/data/significant_heritability_diseases.tsv"
MASTER_RESULTS_CSV = f"phewas_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.tsv"

# --- Performance & Memory Tuning ---
MIN_AVAILABLE_MEMORY_GB = 4.0
QUEUE_MAX_SIZE = os.cpu_count() * 4
LOADER_THREADS = 32
LOADER_CHUNK_SIZE = 128

# --- Data sources and caching ---
CACHE_DIR = "./phewas_cache"
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
RELATEDNESS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"

# --- Model parameters ---
NUM_PCS = 10
MIN_CASES_FILTER = 1000
MIN_CONTROLS_FILTER = 500
FDR_ALPHA = 0.05

# --- Per-ancestry thresholds and multiple-testing for ancestry splits ---
PER_ANC_MIN_CASES = 50
PER_ANC_MIN_CONTROLS = 50
ANCESTRY_ALPHA = 0.05
ANCESTRY_P_ADJ_METHOD = "fdr_bh"
LRT_SELECT_ALPHA = 0.05

# --- Regularization strength for ridge fallback in unstable fits ---
RIDGE_L2_BASE = 1.0

# --- Suppress pandas warnings ---
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

def main():
    import run
    script_start_time = time.time()
    print("=" * 70)
    print(" Starting Robust, Memory-Stable PheWAS Pipeline (Chunked Producer)")
    print("=" * 70)

    # --- Backward compatibility: if a string is given for TARGET_INVERSIONS, wrap it in a list ---
    if isinstance(run.TARGET_INVERSIONS, str):
        run.TARGET_INVERSIONS = [run.TARGET_INVERSIONS]

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        # --- PART 1: SETUP & SHARED DATA LOADING (ONCE) ---
        with Timer() as t_setup:
            print("\n--- Loading shared data (Demographics, PCs, Sex)... ---")
            pheno_defs_df = pheno.load_definitions(PHENOTYPE_DEFINITIONS_URL)

            print("[Setup]    - Setting up BigQuery client...")
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            cdr_codename = cdr_dataset_id.split(".")[-1]

            demographics_df = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"),
                io.load_demographics_with_stable_age, bq_client=bq_client, cdr_id=cdr_dataset_id,
            )
            pc_df = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, "pcs_10.parquet"),
                io.load_pcs, gcp_project, PCS_URI, NUM_PCS, validate_num_pcs=NUM_PCS,
            )
            sex_df = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, "genetic_sex.parquet"),
                io.load_genetic_sex, gcp_project, SEX_URI,
            )
            related_ids_to_remove = io.load_related_to_remove(gcp_project=gcp_project, RELATEDNESS_URI=RELATEDNESS_URI)

            print("[Setup]    - Standardizing covariate indexes...")
            demographics_df.index, pc_df.index, sex_df.index = [df.index.astype(str) for df in (demographics_df, pc_df, sex_df)]

            # Pre-join shared covariates
            shared_covariates_df = demographics_df.join(pc_df, how="inner").join(sex_df, how="inner")
            shared_covariates_df = shared_covariates_df[~shared_covariates_df.index.isin(related_ids_to_remove)]
            print(f"[Setup]    - Shared covariates ready for {len(shared_covariates_df):,} unrelated subjects.")

            # Add ancestry labels for Stage-1 adjustment (shared across all models)
            ancestry = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, "ancestry_labels.parquet"),
                io.load_ancestry_labels, gcp_project, PCS_URI
            )
            anc_series = ancestry.reindex(shared_covariates_df.index)["ANCESTRY"].str.lower()

        print(f"\n--- Total Shared Setup Time: {t_setup.duration:.2f}s ---")

        # --- Main loop to process each inversion ---
        for target_inversion in run.TARGET_INVERSIONS:
            print("\n" + "=" * 70)
            print(f" PROCESSING INVERSION: {target_inversion}")
            print("=" * 70)

            # --- Create per-inversion cache directories to prevent collisions ---
            inversion_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion))
            results_cache_dir = os.path.join(inversion_cache_dir, "results_atomic")
            lrt_overall_cache_dir = os.path.join(inversion_cache_dir, "lrt_overall")
            lrt_followup_cache_dir = os.path.join(inversion_cache_dir, "lrt_followup")
            os.makedirs(results_cache_dir, exist_ok=True)
            os.makedirs(lrt_overall_cache_dir, exist_ok=True)
            os.makedirs(lrt_followup_cache_dir, exist_ok=True)

            # --- Create the context dictionary with per-inversion paths ---
            ctx = {
                "NUM_PCS": NUM_PCS, "MIN_CASES_FILTER": MIN_CASES_FILTER, "MIN_CONTROLS_FILTER": MIN_CONTROLS_FILTER,
                "FDR_ALPHA": FDR_ALPHA, "PER_ANC_MIN_CASES": PER_ANC_MIN_CASES, "PER_ANC_MIN_CONTROLS": PER_ANC_MIN_CONTROLS,
                "LRT_SELECT_ALPHA": LRT_SELECT_ALPHA, "CACHE_DIR": CACHE_DIR, "RIDGE_L2_BASE": RIDGE_L2_BASE,
                "RESULTS_CACHE_DIR": results_cache_dir,
                "LRT_OVERALL_CACHE_DIR": lrt_overall_cache_dir,
                "LRT_FOLLOWUP_CACHE_DIR": lrt_followup_cache_dir,
            }

            # --- Load inversion-specific data and build the final core dataframe ---
            inversion_df = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, f"inversion_{target_inversion}.parquet"),
                io.load_inversions, target_inversion, INVERSION_DOSAGES_FILE, validate_target=target_inversion,
            )
            inversion_df.index = inversion_df.index.astype(str)

            core_df = shared_covariates_df.join(inversion_df, how="inner")
            print(f"[Setup-{target_inversion}] - Post-join cohort size: {len(core_df):,}")

            age_mean = core_df['AGE'].mean()
            core_df['AGE_c'] = core_df['AGE'] - age_mean
            core_df['AGE_c_sq'] = core_df['AGE_c'] ** 2

            pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
            covariate_cols = [target_inversion] + ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]
            core_df_subset = core_df[covariate_cols]
            core_df_with_const = sm.add_constant(core_df_subset, prepend=True)

            anc_cat = pd.Categorical(anc_series.reindex(core_df_with_const.index))
            A = pd.get_dummies(anc_cat, prefix='ANC', drop_first=True, dtype=np.float64)
            core_df_with_const = core_df_with_const.join(A, how="left").fillna({c: 0.0 for c in A.columns})
            print(f"[Setup-{target_inversion}] - Added {len(A.columns)} ancestry columns for adjustment.")

            core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")
            global_notnull_mask = np.isfinite(core_df_with_const.to_numpy()).all(axis=1)

            category_to_pan_cases = pheno.build_pan_category_cases(pheno_defs_df, bq_client, cdr_dataset_id, CACHE_DIR, cdr_codename)
            allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(core_index, category_to_pan_cases, global_notnull_mask)

            # --- PART 2: RUNNING THE PIPELINE (per inversion) ---
            pheno_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
            fetcher_thread = threading.Thread(
                target=pheno.phenotype_fetcher_worker,
                args=(
                    pheno_queue, pheno_defs_df, bq_client, cdr_dataset_id,
                    category_to_pan_cases, cdr_codename, core_index,
                    CACHE_DIR, LOADER_CHUNK_SIZE, LOADER_THREADS
                ),
            )
            fetcher_thread.start()

            pipes.run_fits(pheno_queue, core_df_with_const, allowed_mask_by_cat, target_inversion, results_cache_dir, ctx, MIN_AVAILABLE_MEMORY_GB)
            fetcher_thread.join()
            print(f"\n--- All models finished for {target_inversion}. ---")

            name_to_cat = pheno_defs_df.set_index('sanitized_name')['disease_category'].to_dict()
            result_files = [f for f in os.listdir(results_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
            phenos_list = [f.replace(".json", "") for f in result_files]
            print(f"[LRT-Stage1] Found {len(phenos_list)} model results to schedule for overall LRT.")
            pipes.run_lrt_overall(core_df_with_const, allowed_mask_by_cat, phenos_list, name_to_cat, cdr_codename, target_inversion, ctx, MIN_AVAILABLE_MEMORY_GB)

        # --- PART 3: CONSOLIDATE & ANALYZE RESULTS (ACROSS ALL INVERSIONS) ---
        print("\n" + "=" * 70)
        print(" Part 3: Consolidating final results across all inversions")
        print("=" * 70)

        all_results_from_disk = []
        for target_inversion in run.TARGET_INVERSIONS:
            inversion_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion))
            results_cache_dir = os.path.join(inversion_cache_dir, "results_atomic")
            result_files = [f for f in os.listdir(results_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
            for filename in result_files:
                try:
                    result = pd.read_json(os.path.join(results_cache_dir, filename), typ="series").to_dict()
                    result['Inversion'] = target_inversion
                    all_results_from_disk.append(result)
                except Exception as e:
                    print(f"Warning: Could not read corrupted result file: {filename}, Error: {e}")

        if not all_results_from_disk:
            print("No results found to process.")
        else:
            df = pd.DataFrame(all_results_from_disk)
            print(f"Successfully consolidated {len(df)} results across {len(run.TARGET_INVERSIONS)} inversions.")

            if "OR_CI95" not in df.columns: df["OR_CI95"] = np.nan
            def _compute_overall_or_ci(beta_val, p_val):
                if pd.isna(beta_val) or pd.isna(p_val): return np.nan
                try:
                    b, p = float(beta_val), float(p_val)
                    if not (np.isfinite(b) and np.isfinite(p) and 0.0 < p < 1.0): return np.nan
                    z = float(stats.norm.ppf(1.0 - p / 2.0));
                    if not np.isfinite(z) or z == 0.0: return np.nan
                    se = abs(b) / z
                    return f"{float(np.exp(b - 1.96 * se)):.3f},{float(np.exp(b + 1.96 * se)):.3f}"
                except Exception: return np.nan
            missing_ci_mask = (df["OR_CI95"].isna() | (df["OR_CI95"].astype(str) == "") | (df["OR_CI95"].astype(str).str.lower() == "nan"))
            if "Used_Ridge" in df.columns:
                missing_ci_mask &= (df["Used_Ridge"] == False)
            df.loc[missing_ci_mask, "OR_CI95"] = df.loc[missing_ci_mask, ["Beta", "P_Value"]].apply(lambda r: _compute_overall_or_ci(r["Beta"], r["P_Value"]), axis=1)

            # Collect all Stage-1 LRT results from their per-inversion cache directories
            print("\n--- Collecting Stage-1 LRT results ---")
            overall_records = []
            for target_inversion in run.TARGET_INVERSIONS:
                lrt_overall_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion), "lrt_overall")
                if not os.path.isdir(lrt_overall_cache_dir): continue
                files_overall = [f for f in os.listdir(lrt_overall_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                for filename in files_overall:
                    try:
                        rec = pd.read_json(os.path.join(lrt_overall_cache_dir, filename), typ="series").to_dict()
                        rec['Inversion'] = target_inversion
                        overall_records.append(rec)
                    except Exception as e:
                        print(f"Warning: Could not read LRT overall file: {filename}, Error: {e}")

            if overall_records:
                overall_df = pd.DataFrame(overall_records)
                print(f"Collected {len(overall_df)} Stage-1 LRT records across {len(run.TARGET_INVERSIONS)} inversions.")
                # Merge LRT results into the main dataframe
                df = df.merge(overall_df, on=["Phenotype", "Inversion"], how="left")
            else:
                print("No Stage-1 LRT records found.")

            # Perform global FDR correction on the merged dataframe
            mask_overall = pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna()
            m_total = int(mask_overall.sum())
            df["Q_GLOBAL"] = np.nan
            if m_total > 0:
                print(f"\n--- Applying global BH-FDR correction to {m_total} valid P-values ---")
                _, q_adj_global, _, _ = multipletests(df.loc[mask_overall, "P_LRT_Overall"], alpha=FDR_ALPHA, method="fdr_bh")
                df.loc[mask_overall, "Q_GLOBAL"] = q_adj_global

            df["Sig_Global"] = df["Q_GLOBAL"] < FDR_ALPHA

            # --- PART 4: SCHEDULE AND RUN STAGE-2 FOLLOW-UPS ---
            print("\n" + "=" * 70)
            print(" Part 4: Running Stage-2 Follow-up Analyses for Global Hits")
            print("=" * 70)
            name_to_cat = pheno_defs_df.set_index('sanitized_name')['disease_category'].to_dict()

            for target_inversion in run.TARGET_INVERSIONS:
                # Re-create the inversion-specific context and data to ensure correct follow-up
                inversion_df = io.get_cached_or_generate(
                    os.path.join(CACHE_DIR, f"inversion_{target_inversion}.parquet"),
                    io.load_inversions, target_inversion, INVERSION_DOSAGES_FILE, validate_target=target_inversion,
                )
                inversion_df.index = inversion_df.index.astype(str)
                core_df = shared_covariates_df.join(inversion_df, how="inner")
                age_mean = core_df['AGE'].mean()
                core_df['AGE_c'] = core_df['AGE'] - age_mean
                core_df['AGE_c_sq'] = core_df['AGE_c'] ** 2
                pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                covariate_cols = [target_inversion] + ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]
                core_df_subset = core_df[covariate_cols]
                core_df_with_const = sm.add_constant(core_df_subset, prepend=True)
                anc_cat = pd.Categorical(anc_series.reindex(core_df_with_const.index))
                A = pd.get_dummies(anc_cat, prefix='ANC', drop_first=True, dtype=np.float64)
                core_df_with_const = core_df_with_const.join(A, how="left").fillna({c: 0.0 for c in A.columns})
                core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")
                global_notnull_mask = np.isfinite(core_df_with_const.to_numpy()).all(axis=1)
                category_to_pan_cases = pheno.build_pan_category_cases(pheno_defs_df, bq_client, cdr_dataset_id, CACHE_DIR, cdr_codename)
                allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(core_index, category_to_pan_cases, global_notnull_mask)

                inversion_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion))
                ctx = {
                    "NUM_PCS": NUM_PCS, "MIN_CASES_FILTER": MIN_CASES_FILTER, "MIN_CONTROLS_FILTER": MIN_CONTROLS_FILTER,
                    "FDR_ALPHA": FDR_ALPHA, "PER_ANC_MIN_CASES": PER_ANC_MIN_CASES, "PER_ANC_MIN_CONTROLS": PER_ANC_MIN_CONTROLS,
                    "LRT_SELECT_ALPHA": LRT_SELECT_ALPHA, "CACHE_DIR": CACHE_DIR, "RIDGE_L2_BASE": RIDGE_L2_BASE,
                    "RESULTS_CACHE_DIR": os.path.join(inversion_cache_dir, "results_atomic"),
                    "LRT_OVERALL_CACHE_DIR": os.path.join(inversion_cache_dir, "lrt_overall"),
                    "LRT_FOLLOWUP_CACHE_DIR": os.path.join(inversion_cache_dir, "lrt_followup"),
                }

                # Select hits for the current inversion and run follow-up
                hit_phenos = df.loc[(df["Sig_Global"] == True) & (df["Inversion"] == target_inversion), "Phenotype"].astype(str).tolist()
                if hit_phenos:
                    print(f"--- Running follow-up for {len(hit_phenos)} hits in {target_inversion} ---")
                    pipes.run_lrt_followup(core_df_with_const, allowed_mask_by_cat, anc_series, hit_phenos, name_to_cat, cdr_codename, target_inversion, ctx, MIN_AVAILABLE_MEMORY_GB)

            # Consolidate all follow-up results
            print("\n--- Consolidating all Stage-2 follow-up results ---")
            follow_records = []
            for target_inversion in run.TARGET_INVERSIONS:
                lrt_followup_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion), "lrt_followup")
                if not os.path.isdir(lrt_followup_cache_dir): continue
                files_follow = [f for f in os.listdir(lrt_followup_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                for filename in files_follow:
                    try:
                        rec = pd.read_json(os.path.join(lrt_followup_cache_dir, filename), typ="series").to_dict()
                        rec['Inversion'] = target_inversion
                        follow_records.append(rec)
                    except Exception as e:
                        print(f"Warning: Could not read LRT follow-up file: {filename}, Error: {e}")

            if follow_records:
                follow_df = pd.DataFrame(follow_records)
                print(f"Collected {len(follow_df)} follow-up records.")
                df = df.merge(follow_df, on=["Phenotype", "Inversion"], how="left")

            m_total = int(pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna().sum())
            R_selected = int(pd.to_numeric(df["Sig_Global"], errors="coerce").fillna(False).astype(bool).sum())
            alpha_within = (FDR_ALPHA * (R_selected / m_total)) if m_total > 0 else 0.0

            if R_selected > 0 and alpha_within > 0.0:
                selected_idx = df.index[df["Sig_Global"] == True].tolist()
                for idx in selected_idx:
                    p_lrt = df.at[idx, "P_LRT_AncestryxDosage"] if "P_LRT_AncestryxDosage" in df.columns else np.nan
                    if (not pd.notna(p_lrt)) or (p_lrt >= LRT_SELECT_ALPHA): continue
                    levels_str = df.at[idx, "LRT_Ancestry_Levels"] if "LRT_Ancestry_Levels" in df.columns else ""
                    anc_levels = [s for s in str(levels_str).split(",") if s]
                    anc_upper = [s.upper() for s in anc_levels]
                    pvals, keys = [], []
                    for anc in anc_upper:
                        pcol, rcol = f"{anc}_P", f"{anc}_REASON"
                        if pcol in df.columns and rcol in df.columns:
                            pval, reason = df.at[idx, pcol], df.at[idx, rcol]
                            if pd.notna(pval) and reason != "insufficient_stratum_counts" and reason != "not_selected_by_LRT":
                                pvals.append(float(pval)); keys.append(anc)
                    if len(pvals) > 0:
                        _, p_adj_vals, _, _ = multipletests(pvals, alpha=alpha_within, method="fdr_bh")
                        for anc_key, adj_val in zip(keys, p_adj_vals):
                            df.at[idx, f"{anc_key}_P_FDR"] = float(adj_val)

            if "EUR_P_Source" in df.columns: df = df.drop(columns=["EUR_P_Source"], errors="ignore")

            if "Sig_Global" in df.columns:
                df["FINAL_INTERPRETATION"] = ""
                for idx in df.index[df['Sig_Global'] == True].tolist():
                    p_lrt = df.at[idx, "P_LRT_AncestryxDosage"] if "P_LRT_AncestryxDosage" in df.columns else np.nan
                    if pd.isna(p_lrt) or p_lrt >= LRT_SELECT_ALPHA:
                        df.at[idx, "FINAL_INTERPRETATION"] = "overall"
                        continue

                    levels_str = df.at[idx, "LRT_Ancestry_Levels"] if "LRT_Ancestry_Levels" in df.columns else ""
                    anc_levels = [s.upper() for s in str(levels_str).split(",") if s]
                    sig_groups = []
                    for anc in anc_levels:
                        adj_col, rcol = f"{anc}_P_FDR", f"{anc}_REASON"
                        if adj_col in df.columns:
                            p_adj, reason = df.at[idx, adj_col], df.at[idx, rcol] if rcol in df.columns else ""
                            if pd.notna(p_adj) and p_adj < alpha_within and reason != "insufficient_stratum_counts" and reason != "not_selected_by_LRT":
                                sig_groups.append(anc)
                    df.at[idx, "FINAL_INTERPRETATION"] = ",".join(sig_groups) if sig_groups else "unable to determine"

            print(f"\n--- Saving final results to '{MASTER_RESULTS_CSV}' ---")
            df.to_csv(MASTER_RESULTS_CSV, index=False, sep='\t')

            out_df = df[df['Sig_Global'] == True].copy()
            if not out_df.empty:
                print("\n--- Top Hits Summary ---")
                for col in ["N_Total", "N_Cases", "N_Controls"]:
                    if col in out_df.columns:
                        out_df[col] = pd.to_numeric(out_df[col], errors="coerce").apply(lambda v: f"{int(v):,}" if pd.notna(v) else "")
                for col, fmt in {"Beta": "+0.4f", "OR": "0.3f", "P_Value": ".3e", "Q_GLOBAL": ".3f"}.items():
                    if col in out_df.columns:
                        out_df[col] = pd.to_numeric(out_df[col], errors="coerce").apply(lambda v: f"{v:{fmt}}" if pd.notna(v) else "")
                out_df["Sig_Global"] = out_df["Sig_Global"].fillna(False).map(lambda x: "✓" if bool(x) else "")
                print(out_df.to_string(index=False))

    except Exception as e:
        print("\nSCRIPT HALTED DUE TO A CRITICAL ERROR:", flush=True)
        traceback.print_exc()

    finally:
        script_duration = time.time() - script_start_time
        print("\n" + "=" * 70)
        print(f" Script finished in {script_duration:.2f} seconds.")
        print("=" * 70)

if __name__ == "__main__":
    main()
