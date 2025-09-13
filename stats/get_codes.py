import io
import gzip
import math
import urllib.request
import numpy as np
import pandas as pd

# =============================================================================
# PHASE 1: SCRIPT CONFIGURATION
# =============================================================================

# Primary PheCodeX mapping file.
PHECODE_MAP_URL = "https://raw.githubusercontent.com/nhgritctran/PheTK/main/src/PheTK/phecode/phecodeX.csv"

# UK Biobank "phenocode" mappings for enrichment.
UKBB_ICD10_MAP_URL = "https://raw.githubusercontent.com/atgu/ukbb_pan_ancestry/refs/heads/master/data/UKB_PHENOME_ICD10_PHECODE_MAP_20200109.txt"
UKBB_ICD9_MAP_URL  = "https://raw.githubusercontent.com/atgu/ukbb_pan_ancestry/refs/heads/master/data/UKB_PHENOME_ICD9_PHECODE_MAP_20200109.txt"

# UK Biobank heritability manifest for further enrichment (bgzip).
H2_MANIFEST_URL = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_release/h2_manifest.tsv.bgz"

# Define the name for the final, comprehensive output file.
OUTPUT_FILENAME = "phecodex_with_phenocode_and_h2_mappings.tsv"

MISSING_PHECODES_TSV = "phecodes_without_usable_h2.tsv"

# Define the delimiter for joining lists into a single string.
LIST_DELIMITER = ";"

# --- Statistical knobs ---
H2_THRESHOLD = 0.05   # 5% heritability boundary
FDR_Q        = 0.05   # BH-FDR across phenocodes
USE_QC       = False  # set True to require QC flags (defined_h2 & in_bounds_h2), if available

# =============================================================================
# Utility helpers (pandas/numpy only)
# =============================================================================

def read_bgz_tsv(url: str, usecols=None) -> pd.DataFrame:
    """
    Robustly read a (b)gzipped TSV from URL without pyarrow/duckdb/scipy.
    Tries pandas with compression='infer' first; falls back to urllib+gzip.
    """
    try:
        return pd.read_csv(url, sep="\t", compression="infer", engine="python",
                           usecols=(usecols if usecols is None else (lambda c: c in usecols)))
    except Exception:
        with urllib.request.urlopen(url) as resp:
            gzdata = io.BytesIO(resp.read())
        with gzip.GzipFile(fileobj=gzdata, mode="rb") as gz:
            return pd.read_csv(gz, sep="\t", engine="python",
                               usecols=(usecols if usecols is None else (lambda c: c in usecols)))

def list_join_safe(x):
    """Join a list of strings with LIST_DELIMITER or return empty string if list/values are missing."""
    if not isinstance(x, (list, tuple, np.ndarray)):
        return ""
    if len(x) == 0:
        return ""
    return LIST_DELIMITER.join(map(str, x))

def acat(pvals: np.ndarray) -> float:
    """
    ACAT combiner for one-sided p-values.
    Robust to dependence; good power for sparse signals.
    """
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    tans = np.tan((0.5 - p) * np.pi)
    T = np.mean(tans)
    p_acat = 0.5 - np.arctan(T) / np.pi
    if not np.isfinite(p_acat):
        return np.nan
    return float(np.clip(p_acat, 0.0, 1.0))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg q-values (FDR). Returns q-values in the original order.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = np.empty_like(p_sorted)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (p_sorted[i] * n) / rank
        if val > prev:
            val = prev
        prev = val
        q_sorted[i] = val
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q

def combine_codes_unique_sorted(series: pd.Series) -> list:
    """Helper to deduplicate and sort ICD code lists pulled from a series."""
    if series is None or series.empty:
        return []
    vals = series.dropna().astype(str).unique()
    return sorted(vals)

# =============================================================================
# Main
# =============================================================================

def main():
    """
    Download, merge PheCodeX, UKBB phenocode (PheCodes), and heritability data
    into a single comprehensive mapping file using ACAT + BH-FDR for '>5% in any population'.
    """
    print("Starting combined PheCodeX, UKBB Phenocode, and Heritability mapping process...")

    # =========================================================================
    # PHASE 2: DATA ACQUISITION AND LOADING
    # =========================================================================
    try:
        # --- 2a: Load Primary PheCodeX Data ---
        print(f"Downloading PheCodeX data from: {PHECODE_MAP_URL}")
        phecodex_df = pd.read_csv(
            PHECODE_MAP_URL,
            dtype={
                "phecode": "string",
                "ICD": "string",
                "flag": "Int64",
                "phecode_string": "string",
                "category_num": "string",
                "phecode_category": "string",
                "sex": "string",
                "icd10_only": "Int64",
                "code_val": "string",
            }
        )[["phecode", "ICD", "flag", "phecode_string", "phecode_category"]]

        # --- 2b: Load UK Biobank Phenocode (PheCode) Data ---
        print("Downloading UKBB ICD-to-Phenocode maps...")
        icd10_map = pd.read_csv(
            UKBB_ICD10_MAP_URL, sep="\t", engine="python",
            dtype={"ICD10": "string", "phecode": "string"}
        )[["ICD10", "phecode"]].rename(columns={"ICD10": "ICD", "phecode": "ukbb_phenocode"})

        icd9_map = pd.read_csv(
            UKBB_ICD9_MAP_URL, sep="\t", engine="python",
            dtype={"ICD9": "string", "phecode": "string"}
        )[["ICD9", "phecode"]].rename(columns={"ICD9": "ICD", "phecode": "ukbb_phenocode"})

        ukbb_lookup_df = pd.concat([icd9_map, icd10_map], ignore_index=True).drop_duplicates()
        print("Created a unified UKBB lookup table.")

        # --- 2c: Load and Pre-process Heritability Data ---
        print(f"Downloading heritability data from: {H2_MANIFEST_URL}")

        usecols = [
            "trait_type", "phenocode", "pop",
            "estimates.final.h2_liability", "estimates.final.h2_liability_se",
            "estimates.final.h2_observed", "estimates.final.h2_observed_se",
            "qcflags.defined_h2", "qcflags.in_bounds_h2", "qcflags.pass_all"
        ]
        h2_raw_df = read_bgz_tsv(H2_MANIFEST_URL, usecols=usecols)
        
        # Keep only phecode traits (so 'phenocode' matches PheCodes)
        h2_raw_df = h2_raw_df[h2_raw_df["trait_type"] == "phecode"].copy()
        
        # Preserve a pre-QC snapshot for diagnostics
        h2_pre_qc_df = h2_raw_df.copy()
        
        # Optional QC filters
        if USE_QC:
            have_defined = "qcflags.defined_h2" in h2_raw_df.columns
            have_inbounds = "qcflags.in_bounds_h2" in h2_raw_df.columns
            if have_defined and have_inbounds:
                h2_raw_df = h2_raw_df[
                    (h2_raw_df["qcflags.defined_h2"] == True) &
                    (h2_raw_df["qcflags.in_bounds_h2"] == True)
                ].copy()


        # Ensure columns exist
        for col in [
            "estimates.final.h2_liability", "estimates.final.h2_liability_se",
            "estimates.final.h2_observed", "estimates.final.h2_observed_se"
        ]:
            if col not in h2_raw_df.columns:
                h2_raw_df[col] = np.nan

        # Prefer liability; fallback to observed
        h2_raw_df["h2"] = np.where(
            h2_raw_df["estimates.final.h2_liability"].notna(),
            h2_raw_df["estimates.final.h2_liability"],
            h2_raw_df["estimates.final.h2_observed"]
        )
        h2_raw_df["se"] = np.where(
            h2_raw_df["estimates.final.h2_liability_se"].notna(),
            h2_raw_df["estimates.final.h2_liability_se"],
            h2_raw_df["estimates.final.h2_observed_se"]
        )

        # Drop rows with missing/invalid SE or h2
        h2_rows = h2_raw_df[
            h2_raw_df["h2"].notna() & h2_raw_df["se"].notna() & (h2_raw_df["se"] > 0)
        ][["phenocode", "pop", "h2", "se"]].copy()

        # One-sided p-values vs threshold 0.05
        # p = 0.5 * erfc( z / sqrt(2) ), z = (h2 - 0.05)/se
        z = (h2_rows["h2"].to_numpy(dtype=float) - H2_THRESHOLD) / h2_rows["se"].to_numpy(dtype=float)
        p_one = np.array([0.5 * math.erfc(val / math.sqrt(2.0)) for val in z], dtype=float)
        p_one = np.clip(p_one, 1e-15, 1.0 - 1e-15)
        h2_rows["p_one"] = p_one

        # ACAT + Bonferroni per phenocode
        def acat_group(g: pd.DataFrame) -> pd.Series:
            ps = g["p_one"].to_numpy(dtype=float)
            p_ac = acat(ps)
            p_bf = min(1.0, float(ps.min() * len(ps)))
            h2_max = float(np.nanmax(g["h2"].to_numpy(dtype=float))) if len(g) else np.nan
            pop_min = g.loc[g["p_one"].idxmin(), "pop"] if len(g) else np.nan
            return pd.Series({
                "p_any_acat": p_ac,
                "p_any_bonf": p_bf,
                "h2_max_any_pop": h2_max,
                "pop_min_p": pop_min
            })

        per_pheno = (
            h2_rows.groupby("phenocode")
            .apply(acat_group)
            .reset_index()  # <-- keep 'phenocode' as a column
        )

        # BH-FDR across phenocodes using ACAT p-values
        valid_mask = per_pheno["p_any_acat"].notna()
        per_pheno["q_bh"] = np.nan
        if valid_mask.any():
            qvals = bh_fdr(per_pheno.loc[valid_mask, "p_any_acat"].to_numpy(dtype=float))
            per_pheno.loc[valid_mask, "q_bh"] = qvals

        # Phenocode-level flag (FDR)
        per_pheno["phenocode_is_gt5_fdr"] = (
            (per_pheno["q_bh"].notna()) & (per_pheno["q_bh"] <= FDR_Q)
        ).astype("int64")

        # EUR descriptive summary (mean EUR h2 across rows; already liability-preferred)
        eur_rows = h2_rows[h2_rows["pop"] == "EUR"].copy()
        eur_h2 = eur_rows.groupby("phenocode", as_index=False)["h2"].mean()
        eur_h2 = eur_h2.rename(columns={"h2": "eur_h2_mean"})

        # Merge per-phenocode stats
        per_pheno = per_pheno.merge(eur_h2, on="phenocode", how="left")

        # --- Build report of mapped PheCodes lacking usable h2/SE and why ---
        # Set of mapped PheCodes actually referenced by PheCodeX ICDs (restrict to ICDs present in PheCodeX)
        mapped_set = set(
            ukbb_lookup_df.loc[
                ukbb_lookup_df["ICD"].isin(phecodex_df["ICD"].dropna().astype(str)),
                "ukbb_phenocode"
            ].dropna().astype(str).unique().tolist()
        )


        # Phenocodes with usable p-values (i.e., considered for ACAT)
        considered_set = set(per_pheno.loc[per_pheno["p_any_acat"].notna(), "phenocode"].astype(str).unique().tolist())

        # Phenocode presence in manifest (pre-QC and post-QC)
        manifest_preqc_set = set(h2_pre_qc_df["phenocode"].dropna().astype(str).unique().tolist())
        manifest_postqc_set = set(h2_raw_df["phenocode"].dropna().astype(str).unique().tolist())

        # Candidates to report = mapped but not considered
        missing_set = sorted(mapped_set - considered_set)

        # Helper: summarize per-phenocode row counts & issues
        def summarize_issues_one(phenocode: str) -> dict:
            # rows in manifest pre/post QC
            pre = h2_pre_qc_df[h2_pre_qc_df["phenocode"].astype(str) == phenocode]
            post = h2_raw_df[h2_raw_df["phenocode"].astype(str) == phenocode]

            # Choose liability/observed (same logic as main path) for diagnostics
            for df in (pre, post):
                if "estimates.final.h2_liability" not in df.columns: df["estimates.final.h2_liability"] = np.nan
                if "estimates.final.h2_liability_se" not in df.columns: df["estimates.final.h2_liability_se"] = np.nan
                if "estimates.final.h2_observed" not in df.columns: df["estimates.final.h2_observed"] = np.nan
                if "estimates.final.h2_observed_se" not in df.columns: df["estimates.final.h2_observed_se"] = np.nan
                df["_h2_pref"] = np.where(
                    df["estimates.final.h2_liability"].notna(),
                    df["estimates.final.h2_liability"],
                    df["estimates.final.h2_observed"],
                )
                df["_se_pref"] = np.where(
                    df["estimates.final.h2_liability_se"].notna(),
                    df["estimates.final.h2_liability_se"],
                    df["estimates.final.h2_observed_se"],
                )

            # Counts
            n_pops_preqc = int(pre.shape[0])
            n_pops_postqc = int(post.shape[0])
            n_h2_preqc = int(pre["_h2_pref"].notna().sum())
            n_se_preqc = int(pre["_se_pref"].notna().sum())
            n_sepos_postqc = int((post["_se_pref"].notna() & (post["_se_pref"] > 0)).sum())

            # Which pops exist pre/post & with usable SE>0
            pops_preqc = ";".join(sorted(pre["pop"].dropna().astype(str).unique().tolist())) if n_pops_preqc else ""
            pops_postqc = ";".join(sorted(post["pop"].dropna().astype(str).unique().tolist())) if n_pops_postqc else ""
            pops_sepos = ";".join(sorted(post.loc[(post["_se_pref"].notna()) & (post["_se_pref"] > 0), "pop"].dropna().astype(str).unique().tolist()))

            # Primary reason classification (exclusive, ordered)
            if phenocode not in manifest_preqc_set:
                reason = "absent_from_manifest"
            elif n_pops_postqc == 0 and USE_QC:
                reason = "removed_by_qc_all_pops"
            elif n_sepos_postqc == 0:
                # present after QC but SE unsuitable everywhere
                if n_se_preqc == 0:
                    reason = "missing_se_all_pops"
                else:
                    reason = "nonpositive_or_missing_se_all_pops"
            else:
                # Should not happen if it's in missing_set; fallback label
                reason = "other_not_considered"

            # Descriptive: max h2 (pre-QC) if any
            h2_vals = pre["_h2_pref"].astype(float)
            h2_max_preqc = (float(np.nanmax(h2_vals)) if h2_vals.notna().any() else np.nan)

            return dict(
                phenocode=phenocode,
                reason=reason,
                n_pops_manifest_preqc=n_pops_preqc,
                n_pops_manifest_postqc=n_pops_postqc,
                n_rows_with_h2_preqc=n_h2_preqc,
                n_rows_with_se_preqc=n_se_preqc,
                n_rows_with_sepos_postqc=n_sepos_postqc,
                pops_manifest_preqc=pops_preqc,
                pops_manifest_postqc=pops_postqc,
                pops_with_sepos_postqc=pops_sepos,
                h2_max_preqc=h2_max_preqc,
            )

        missing_rows = [summarize_issues_one(pc) for pc in missing_set]
        missing_df = pd.DataFrame(missing_rows, columns=[
            "phenocode","reason",
            "n_pops_manifest_preqc","n_pops_manifest_postqc",
            "n_rows_with_h2_preqc","n_rows_with_se_preqc","n_rows_with_sepos_postqc",
            "pops_manifest_preqc","pops_manifest_postqc","pops_with_sepos_postqc",
            "h2_max_preqc"
        ])

        # Write TSV of “mapped but not considered” phenocodes with reasons
        try:
            missing_df.to_csv(MISSING_PHECODES_TSV, sep="\t", index=False)
            print(f"Wrote detail on missing/unsuitable PheCodes to: {MISSING_PHECODES_TSV}")
        except Exception as _e:
            print(f"Warning: failed to write {MISSING_PHECODES_TSV}: {_e}")

        print("Successfully processed heritability data at the phenocode level (ACAT + BH-FDR).")

    except Exception as e:
        print("------------------------------------------------------------")
        print("FATAL ERROR: Failed to download or parse a required data file.")
        print(f"Error details: {e}")
        return

    # =========================================================================
    # PHASE 3: MERGING AND AGGREGATION
    # =========================================================================
    print("Enriching data with UKBB Phenocodes and Heritability stats...")

    # Map ICD -> ukbb_phenocode (PheCodes), left-join into PheCodeX rows
    base_df = phecodex_df.merge(ukbb_lookup_df, on="ICD", how="left")

    grouping_cols = ["phecode", "phecode_string", "phecode_category"]

    # Aggregate to PheCodeX disease level with deduplicated code lists and PheCodes
    def agg_disease(g: pd.DataFrame) -> pd.Series:
        icd9 = combine_codes_unique_sorted(g.loc[g["flag"] == 9, "ICD"])
        icd10 = combine_codes_unique_sorted(g.loc[g["flag"] == 10, "ICD"])
        phecodes = combine_codes_unique_sorted(g["ukbb_phenocode"])
        return pd.Series({
            "ICD9_Mappings": icd9,
            "ICD10_Mappings": icd10,
            "ukbb_phenocode": phecodes
        })

    aggregated_df = (
        base_df.groupby(grouping_cols)
        .apply(agg_disease)
        .reset_index()  # <-- keep grouping keys as columns
    )

    # Explode phenocode list and join to phenocode-level heritability signals
    long_df = aggregated_df.explode("ukbb_phenocode")
    long_df["ukbb_phenocode"] = long_df["ukbb_phenocode"].astype("string")
    per_pheno["phenocode"] = per_pheno["phenocode"].astype("string")

    long_joined = long_df.merge(
        per_pheno[["phenocode", "phenocode_is_gt5_fdr", "eur_h2_mean"]],
        left_on="ukbb_phenocode",
        right_on="phenocode",
        how="left"
    )

    # Aggregate back to disease level: OR of flags, mean EUR h2 across mapped phenocodes
    disease_signals = (
        long_joined.groupby(grouping_cols, as_index=False)
        .agg({
            "phenocode_is_gt5_fdr": "max",
            "eur_h2_mean": "mean"
        })
        .rename(columns={
            "phenocode_is_gt5_fdr": "is_h2_significant_in_any_ancestry",
            "eur_h2_mean": "h2_eur_avg"  # keep legacy name for downstream compatibility
        })
    )

    # Merge signals back to aggregated_df; diseases with no mapped phenocodes will get NaNs
    final_df = aggregated_df.merge(disease_signals, on=grouping_cols, how="left")

    # Fill missing flag with 0; keep h2_eur_avg as float (round on write)
    final_df["is_h2_significant_in_any_ancestry"] = (
        final_df["is_h2_significant_in_any_ancestry"].fillna(0).astype("int64")
    )

    # --- Reporting: phenocode-level and disease-level summaries ---
    # Phenocode-level (mapped to PheCodeX) coverage and significance
    mapped_codes = aggregated_df["ukbb_phenocode"].explode().dropna().astype(str).unique()
    considered_codes = per_pheno.loc[per_pheno["p_any_acat"].notna(), "phenocode"].astype(str).unique()
    mapped_set = set(mapped_codes.tolist()) if hasattr(mapped_codes, "tolist") else set(mapped_codes)
    considered_set = set(considered_codes.tolist()) if hasattr(considered_codes, "tolist") else set(considered_codes)
    have_p_set = mapped_set & considered_set
    n_mapped = len(mapped_set)
    n_have_p = len(have_p_set)
    n_sig = int(per_pheno.loc[per_pheno["phenocode"].astype(str).isin(have_p_set), "phenocode_is_gt5_fdr"].fillna(0).sum())
    n_nonsig = n_have_p - n_sig
    n_no_data = n_mapped - n_have_p

    print("Phenocode-level summary (mapped to PheCodeX):")
    print(f"  Total mapped PheCodes: {n_mapped}")
    print(f"  With usable h2/SE (considered): {n_have_p}")
    print(f"  BH–FDR ≤ {FDR_Q:.2f} significant (> {H2_THRESHOLD*100:.0f}% in any pop): {n_sig}")
    print(f"  Not significant (among considered): {n_nonsig}")
    print(f"  No significance data / not used: {n_no_data}")

    # Disease-level summary
    n_diseases = int(final_df.shape[0])
    n_diseases_sig = int(final_df["is_h2_significant_in_any_ancestry"].sum())
    print("Disease-level summary:")
    print(f"  Total PheCodeX diseases: {n_diseases}")
    print(f"  With > {H2_THRESHOLD*100:.0f}% any-pop signal (via mapped PheCodes, FDR): {n_diseases_sig}")
    print(f"  Without: {n_diseases - n_diseases_sig}")

    # =========================================================================
    # PHASE 4: FINAL FORMATTING AND OUTPUT GENERATION
    # =========================================================================
    print("Formatting final data and preparing for output...")

    # Convert lists to delimited strings; keep ukbb_phenocode column name unchanged
    final_df["icd9_codes"] = final_df["ICD9_Mappings"].apply(list_join_safe)
    final_df["icd10_codes"] = final_df["ICD10_Mappings"].apply(list_join_safe)
    final_df["ukbb_phenocode"] = final_df["ukbb_phenocode"].apply(list_join_safe)

    # Round h2_eur_avg to 4 decimals; empty string if NaN (compat with prior outputs)
    if "h2_eur_avg" not in final_df.columns:
        final_df["h2_eur_avg"] = np.nan
    final_df["h2_eur_avg"] = final_df["h2_eur_avg"].round(4)
    final_df["h2_eur_avg"] = final_df["h2_eur_avg"].apply(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    # Select and order final columns
    final_df = final_df[[
        "phecode",
        "ukbb_phenocode",
        "phecode_string",
        "phecode_category",
        "is_h2_significant_in_any_ancestry",
        "h2_eur_avg",
        "icd9_codes",
        "icd10_codes"
    ]].rename(columns={
        "phecode_string": "disease",
        "phecode_category": "disease_category"
    })

    # Write TSV
    try:
        final_df.to_csv(OUTPUT_FILENAME, sep="\t", index=False)
        print("-" * 50)
        print("PROCESS COMPLETE!")
        print(f"Successfully created the enriched mapping file: {OUTPUT_FILENAME}")
        print(f"Found and processed {len(final_df)} unique diseases.")
        print("-" * 50)
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")


if __name__ == "__main__":
    main()
