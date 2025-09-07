import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
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
QUEUE_MAX_SIZE = os.cpu_count() * 4
LOADER_THREADS = 32
LOADER_CHUNK_SIZE = 128

# --- Data sources and caching ---
CACHE_DIR = "./phewas_cache"
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results_atomic")
LRT_OVERALL_CACHE_DIR = os.path.join(CACHE_DIR, "lrt_overall")
LRT_FOLLOWUP_CACHE_DIR = os.path.join(CACHE_DIR, "lrt_followup")
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
    script_start_time = time.time()
    print("=" * 70)
    print(" Starting Robust, Memory-Stable PheWAS Pipeline (Chunked Producer)")
    print("=" * 70)

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)
    os.makedirs(LRT_OVERALL_CACHE_DIR, exist_ok=True)
    os.makedirs(LRT_FOLLOWUP_CACHE_DIR, exist_ok=True)

    # Context dictionary to pass constants to workers without circular imports.
    ctx = {
        "NUM_PCS": NUM_PCS,
        "MIN_CASES_FILTER": MIN_CASES_FILTER,
        "MIN_CONTROLS_FILTER": MIN_CONTROLS_FILTER,
        "FDR_ALPHA": FDR_ALPHA,
        "PER_ANC_MIN_CASES": PER_ANC_MIN_CASES,
        "PER_ANC_MIN_CONTROLS": PER_ANC_MIN_CONTROLS,
        "LRT_SELECT_ALPHA": LRT_SELECT_ALPHA,
        "CACHE_DIR": CACHE_DIR,
        "RESULTS_CACHE_DIR": RESULTS_CACHE_DIR,
        "LRT_OVERALL_CACHE_DIR": LRT_OVERALL_CACHE_DIR,
        "LRT_FOLLOWUP_CACHE_DIR": LRT_FOLLOWUP_CACHE_DIR,
        "RIDGE_L2_BASE": RIDGE_L2_BASE
    }

    try:
        with Timer() as t_setup:
            print("\n--- PART 1: SETUP & SHARED DATA LOADING ---")
            pheno_defs_df = pheno.load_definitions(PHENOTYPE_DEFINITIONS_URL)

            print("[Setup]    - Setting up BigQuery client...")
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            cdr_codename = cdr_dataset_id.split(".")[-1]

            print("[Setup]    - Loading shared covariates (Demographics, Inversions, PCs, Sex)...")
            demographics_df = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"),
                io.load_demographics_with_stable_age,
                bq_client=bq_client,
                cdr_id=cdr_dataset_id,
            )
            inversion_df = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, f"inversion_{TARGET_INVERSION}.parquet"),
                io.load_inversions,
                TARGET_INVERSION,
                INVERSION_DOSAGES_FILE,
                validate_target=TARGET_INVERSION,
            )
            pc_df = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, "pcs_10.parquet"),
                io.load_pcs,
                gcp_project,
                PCS_URI,
                NUM_PCS,
                validate_num_pcs=NUM_PCS,
            )
            sex_df = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, "genetic_sex.parquet"),
                io.load_genetic_sex,
                gcp_project,
                SEX_URI,
            )

            related_ids_to_remove = io.load_related_to_remove(gcp_project=gcp_project, RELATEDNESS_URI=RELATEDNESS_URI)

            print("[Setup]    - Standardizing covariate indexes for robust joining...")
            demographics_df.index = demographics_df.index.astype(str)
            inversion_df.index = inversion_df.index.astype(str)
            pc_df.index = pc_df.index.astype(str)
            sex_df.index = sex_df.index.astype(str)

            pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]

            core_df = (
                demographics_df.join(inversion_df, how="inner")
                .join(pc_df, how="inner")
                .join(sex_df, how="inner")
            )

            print(f"[Setup]    - Pre-filter cohort size: {len(core_df):,}")
            core_df = core_df[~core_df.index.isin(related_ids_to_remove)]
            print(f"[Setup]    - Post-filter unrelated cohort size: {len(core_df):,}")

            # Center age and create squared term for better model stability
            age_mean = core_df['AGE'].mean()
            core_df['AGE_c'] = core_df['AGE'] - age_mean
            core_df['AGE_c_sq'] = core_df['AGE_c'] ** 2
            print(f"[Setup]    - Age centered around mean ({age_mean:.2f}). AGE_c and AGE_c_sq created.")

            covariate_cols = [TARGET_INVERSION] + ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]
            core_df = core_df[covariate_cols]
            core_df_with_const = sm.add_constant(core_df, prepend=True)

            print("\n--- [DIAGNOSTIC] Testing matrix condition number ---")
            try:
                cols = ['const', 'sex', 'AGE_c', 'AGE_c_sq', TARGET_INVERSION] + [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                mat = core_df_with_const[cols].dropna().to_numpy()
                cond = np.linalg.cond(mat)
                print(f"[DIAGNOSTIC] Condition number (current model cols): {cond:,.2f}")
            except Exception as e:
                print(f"[DIAGNOSTIC] Could not compute condition number. Error: {e}")
            print("--- [DIAGNOSTIC] End of test ---\n")

            del core_df, demographics_df, inversion_df, pc_df, sex_df
            gc.collect()
            print(f"[Setup]    - Core covariate DataFrame ready. Shape: {core_df_with_const.shape}")

            if core_df_with_const.shape[0] == 0:
                raise RuntimeError("FATAL: Core covariate DataFrame has 0 rows after join. Check input data alignment.")

            # Add ancestry main effects to adjust for population structure in Stage-1 LRT
            print("[Setup]    - Loading ancestry labels for Stage-1 model adjustment...")
            ancestry = io.get_cached_or_generate(
                os.path.join(CACHE_DIR, "ancestry_labels.parquet"),
                io.load_ancestry_labels, gcp_project, PCS_URI
            )
            anc_series = ancestry.reindex(core_df_with_const.index)["ANCESTRY"].str.lower()
            anc_cat = pd.Categorical(anc_series)
            A = pd.get_dummies(anc_cat, prefix='ANC', drop_first=True, dtype=np.float64)
            core_df_with_const = core_df_with_const.join(A, how="left").fillna({c: 0.0 for c in A.columns})
            print(f"[Setup]    - Added {len(A.columns)} ancestry columns for adjustment: {list(A.columns)}")

            core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")
            global_notnull_mask = np.isfinite(core_df_with_const.to_numpy()).all(axis=1)
            print(f"[Mem] RSS after core covariates assembly: {io.rss_gb():.2f} GB")

            category_to_pan_cases = pheno.build_pan_category_cases(pheno_defs_df, bq_client, cdr_dataset_id, CACHE_DIR, cdr_codename)
            allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(core_index, category_to_pan_cases, global_notnull_mask)
            print(f"[Mem] RSS after allowed-mask preprocessing: {io.rss_gb():.2f} GB")

        print(f"\n--- Total Setup Time: {t_setup.duration:.2f}s ---")

        # --- PART 2: RUNNING THE PIPELINE ---
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

        pipes.run_fits(pheno_queue, core_df_with_const, allowed_mask_by_cat, TARGET_INVERSION, RESULTS_CACHE_DIR, ctx)

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
            df = results_df.copy()

            if "OR_CI95" not in df.columns: df["OR_CI95"] = np.nan

            def _compute_overall_or_ci(beta_val, p_val):
                if pd.isna(beta_val) or pd.isna(p_val): return np.nan
                try:
                    b, p = float(beta_val), float(p_val)
                    if not (np.isfinite(b) and np.isfinite(p) and 0.0 < p < 1.0): return np.nan
                    z = float(stats.norm.ppf(1.0 - p / 2.0))
                    if not np.isfinite(z) or z == 0.0: return np.nan
                    se = abs(b) / z
                    return f"{float(np.exp(b - 1.96 * se)):.3f},{float(np.exp(b + 1.96 * se)):.3f}"
                except Exception: return np.nan

            if "Used_Ridge" not in df.columns:
                df["Used_Ridge"] = False
            df["Used_Ridge"] = df["Used_Ridge"].fillna(False)

            missing_ci_mask = (
                (df["OR_CI95"].isna() | (df["OR_CI95"].astype(str) == "") | (df["OR_CI95"].astype(str).str.lower() == "nan")) &
                (df["Used_Ridge"] == False)
            )
            df.loc[missing_ci_mask, "OR_CI95"] = df.loc[missing_ci_mask, ["Beta", "P_Value"]].apply(lambda r: _compute_overall_or_ci(r["Beta"], r["P_Value"]), axis=1)

            total_core = int(len(core_df_with_const.index))
            known_anc = int(anc_series.notna().sum())
            missing_anc = total_core - known_anc
            core_dup = int(core_df_with_const.index.duplicated(keep=False).sum())
            core_idx_dtype = str(core_df_with_const.index.dtype)
            anc_levels_global = ",".join(sorted(pd.Series(anc_series.dropna().unique()).astype(str)))
            print(f"[DEBUG] ancestry_align total_core={total_core} known={known_anc} missing={missing_anc} core_index_dtype={core_idx_dtype} core_index_dup_count={core_dup} levels={anc_levels_global}", flush=True)

            name_to_cat = pheno_defs_df.set_index('sanitized_name')['disease_category'].to_dict()
            phenos_list = df["Phenotype"].astype(str).tolist()

            pipes.run_lrt_overall(core_df_with_const, allowed_mask_by_cat, phenos_list, name_to_cat, cdr_codename, TARGET_INVERSION, ctx)

            overall_records = []
            files_overall = [f for f in os.listdir(LRT_OVERALL_CACHE_DIR) if f.endswith(".json") and not f.endswith(".meta.json")]
            total_ov = len(files_overall)
            bar_len = 30
            for i, filename in enumerate(files_overall, start=1):
                try:
                    rec = pd.read_json(os.path.join(LRT_OVERALL_CACHE_DIR, filename), typ="series")
                    overall_records.append(rec.to_dict())
                except Exception as e:
                    print(f"Warning: Could not read LRT overall file: {filename}, Error: {e}")
                filled = int(bar_len * i / total_ov) if total_ov > 0 else bar_len
                bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                pct = int(i * 100 / total_ov) if total_ov > 0 else 100
                print(f"\r[LRT-Stage1-Collect] {bar} {i}/{total_ov} ({pct}%)", end="", flush=True)
            if total_ov > 0: print("")

            if len(overall_records) > 0:
                overall_df = pd.DataFrame(overall_records)
                df = df.merge(overall_df, on="Phenotype", how="left")
                mask_overall = pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna()
                m_total = int(mask_overall.sum())
                df["P_FDR"] = np.nan
                if m_total > 0:
                    _, p_adj_overall, _, _ = multipletests(df.loc[mask_overall, "P_LRT_Overall"], alpha=FDR_ALPHA, method="fdr_bh")
                    df.loc[mask_overall, "P_FDR"] = p_adj_overall
                df["Sig_FDR"] = df["P_FDR"] < FDR_ALPHA
            else:
                df["P_FDR"] = np.nan
                df["Sig_FDR"] = False

            hit_names = df.loc[df["Sig_FDR"] == True, "Phenotype"].astype(str).tolist()
            pipes.run_lrt_followup(core_df_with_const, allowed_mask_by_cat, anc_series, hit_names, name_to_cat, cdr_codename, TARGET_INVERSION, ctx)

            follow_records = []
            files_follow = [f for f in os.listdir(LRT_FOLLOWUP_CACHE_DIR) if f.endswith(".json") and not f.endswith(".meta.json")]
            total_fw = len(files_follow)
            bar_len = 30
            if len(files_follow) > 0:
                for i, filename in enumerate(files_follow, start=1):
                    try:
                        rec = pd.read_json(os.path.join(LRT_FOLLOWUP_CACHE_DIR, filename), typ="series")
                        follow_records.append(rec.to_dict())
                    except Exception as e:
                        print(f"Warning: Could not read LRT follow-up file: {filename}, Error: {e}")
                    filled = int(bar_len * i / total_fw) if total_fw > 0 else bar_len
                    bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
                    pct = int(i * 100 / total_fw) if total_fw > 0 else 100
                    print(f"\r[FollowUp-Collect] {bar} {i}/{total_fw} ({pct}%)", end="", flush=True)
                if total_fw > 0: print("")

                if len(follow_records) > 0:
                    follow_df = pd.DataFrame(follow_records)
                    df = df.merge(follow_df, on="Phenotype", how="left")

            m_total = int(pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna().sum())
            R_selected = int(pd.to_numeric(df["Sig_FDR"], errors="coerce").fillna(False).astype(bool).sum())
            alpha_within = (FDR_ALPHA * (R_selected / m_total)) if m_total > 0 else 0.0

            if R_selected > 0 and alpha_within > 0.0:
                selected_idx = df.index[df["Sig_FDR"] == True].tolist()
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

            if "Sig_FDR" in df.columns:
                df["FINAL_INTERPRETATION"] = ""
                for idx in df.index[df['Sig_FDR'] == True].tolist():
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

            safe_inversion_id = TARGET_INVERSION.replace(":", "_").replace("-", "_")
            output_filename = f"phewas_results_{safe_inversion_id}.csv"
            print(f"\n--- Saving final results to '{output_filename}' ---")
            df.to_csv(output_filename, index=False)

            out_df = df[df['Sig_FDR'] == True].copy()
            for col in ["N_Total", "N_Cases", "N_Controls"]:
                out_df[col] = pd.to_numeric(out_df[col], errors="coerce").apply(lambda v: f"{int(v):,}" if pd.notna(v) else "")
            for col, fmt in {"Beta": "+0.4f", "OR": "0.3f", "P_Value": ".3e", "P_FDR": ".3f"}.items():
                out_df[col] = pd.to_numeric(out_df[col], errors="coerce").apply(lambda v: f"{v:{fmt}}" if pd.notna(v) else "")
            out_df["Sig_FDR"] = out_df["Sig_FDR"].fillna(False).map(lambda x: "✓" if bool(x) else "")
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
