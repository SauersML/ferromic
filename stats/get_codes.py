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
H2_THRESHOLD = 0.10   # 10% heritability boundary
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

import math
import numpy as np
import pandas as pd

def ivw_fixed(h2_array, se_array):
    """
    Fixed-effect IVW combine across PheCodes within a population.
    Returns: (theta_hat, se_hat, w_sum, n_used)
    """
    h2 = np.asarray(h2_array, dtype=float)
    se = np.asarray(se_array, dtype=float)
    mask = np.isfinite(h2) & np.isfinite(se) & (se > 0)
    h2, se = h2[mask], se[mask]
    n_used = int(h2.size)
    if n_used == 0:
        return (np.nan, np.nan, 0.0, 0)
    w = 1.0 / np.square(se)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return (np.nan, np.nan, 0.0, 0)
    theta_hat = float(np.sum(w * h2) / w_sum)
    se_hat = float(np.sqrt(1.0 / w_sum))
    return (theta_hat, se_hat, w_sum, n_used)

def cochran_q_i2(h2_array, se_array, theta_hat):
    """
    Cochran's Q and I^2 (in %) for heterogeneity across PheCodes within a population.
    """
    h2 = np.asarray(h2_array, dtype=float)
    se = np.asarray(se_array, dtype=float)
    mask = np.isfinite(h2) & np.isfinite(se) & (se > 0)
    h2, se = h2[mask], se[mask]
    k = h2.size
    if (k <= 1) or (not np.isfinite(theta_hat)):
        return (np.nan, 0.0)
    w = 1.0 / np.square(se)
    Q = float(np.sum(w * np.square(h2 - theta_hat)))
    df = k - 1
    if Q <= 0:
        return (Q, 0.0)
    I2 = max(0.0, (Q - df) / Q) * 100.0
    return (Q, float(I2))

def _dl_tau2(theta, v):
    """
    DerSimonian–Laird tau^2 estimator (non-negative).
    theta: array of per-pop estimates
    v:     array of per-pop variances (se^2)
    """
    w = 1.0 / v
    sum_w = np.sum(w)
    sum_w2 = np.sum(np.square(w))
    mu_fe = np.sum(w * theta) / sum_w
    Q = np.sum(w * np.square(theta - mu_fe))
    df = theta.size - 1
    c = sum_w - (sum_w2 / sum_w)
    tau2 = max(0.0, (Q - df) / max(c, 1e-12))
    return tau2, Q, df, mu_fe, sum_w

def re_meta_pop(theta_array, se_array, method="DL", kh=True):
    """
    Random-effects meta-analysis across populations for one disease.
    Inputs:
      - theta_array: per-pop IVW point estimates
      - se_array:    per-pop IVW standard errors
      - method:      "DL" (DerSimonian–Laird); REML not implemented here
      - kh:          if True, use t-like critical value (we use normal fallback without SciPy)
    Returns dict with:
      mu (point), se_mu, ci95_l, ci95_u, tau2, I2, pred_int_l, pred_int_u, n_pops_used
    """
    theta = np.asarray(theta_array, dtype=float)
    se = np.asarray(se_array, dtype=float)
    mask = np.isfinite(theta) & np.isfinite(se) & (se > 0)
    theta, se = theta[mask], se[mask]
    k = theta.size
    out = dict(mu=np.nan, se_mu=np.nan, ci95_l=np.nan, ci95_u=np.nan,
               tau2=np.nan, I2=np.nan, pred_int_l=np.nan, pred_int_u=np.nan,
               n_pops_used=int(k))
    if k == 0:
        return out
    v = np.square(se)

    # If only one population: return FE value
    if k == 1:
        out["mu"] = float(theta[0])
        out["se_mu"] = float(se[0])
        out["ci95_l"] = out["mu"] - 1.96 * out["se_mu"]
        out["ci95_u"] = out["mu"] + 1.96 * out["se_mu"]
        out["tau2"] = 0.0
        out["I2"] = 0.0
        # predictive interval undefined with one pop; leave NaNs
        return out

    # DL tau^2 (reverts to FE if Q<=df)
    tau2, Q, df, mu_fe, sum_w = _dl_tau2(theta, v)
    w_re = 1.0 / (v + tau2)
    sum_w_re = np.sum(w_re)
    mu_re = float(np.sum(w_re * theta) / sum_w_re)
    se_mu = float(np.sqrt(1.0 / sum_w_re))

    # I^2 across pops
    I2 = 0.0
    if Q > df:
        I2 = max(0.0, (Q - df) / Q) * 100.0

    # CIs (use normal 1.96; if you add SciPy, replace with t-crit for KH)
    crit = 1.96
    ci_l = mu_re - crit * se_mu
    ci_u = mu_re + crit * se_mu

    # Predictive interval for a new population (approx)
    pred_sd = math.sqrt(tau2 + se_mu**2) if np.isfinite(tau2) else np.nan
    pred_l = mu_re - crit * pred_sd if np.isfinite(pred_sd) else np.nan
    pred_u = mu_re + crit * pred_sd if np.isfinite(pred_sd) else np.nan

    out.update(dict(
        mu=mu_re, se_mu=se_mu, ci95_l=ci_l, ci95_u=ci_u,
        tau2=float(tau2), I2=float(I2),
        pred_int_l=pred_l, pred_int_u=pred_u
    ))
    return out

def flag_scale_mix(scale_used_series):
    """
    Returns 1 if any contributing row used observed-scale fallback; else 0.
    """
    if scale_used_series is None or len(scale_used_series) == 0:
        return 0
    vals = pd.Series(scale_used_series).astype(str).str.lower()
    return int(any(v != "liability" for v in vals if pd.notna(v)))

def build_disease_pop_estimates(disease_pheno_pop_long, grouping_cols):
    """
    Stage A: For each (disease, pop), IVW across mapped PheCodes with usable (h2, se).
    Returns a dataframe with one row per (disease, pop):
      - theta_pop, se_pop, n_phecodes_used_pop, i2_within_pop, scale_mix_pop
    """
    req = grouping_cols + ["pop", "h2", "se", "scale_used"]
    df = disease_pheno_pop_long[req].copy()

    # Drop rows with missing h2 or se here (IVW will also check; this keeps groups small)
    df = df[df["h2"].notna() & df["se"].notna() & (df["se"] > 0)]

    if df.empty:
        cols = grouping_cols + ["pop", "theta_pop", "se_pop", "n_phecodes_used_pop", "i2_within_pop", "scale_mix_pop"]
        return pd.DataFrame(columns=cols)

    def _one_group(g: pd.DataFrame) -> pd.Series:
        theta_hat, se_hat, wsum, n_used = ivw_fixed(g["h2"].to_numpy(), g["se"].to_numpy())
        Q, I2 = cochran_q_i2(g["h2"].to_numpy(), g["se"].to_numpy(), theta_hat)
        mix = flag_scale_mix(g["scale_used"])
        return pd.Series({
            "theta_pop": theta_hat,
            "se_pop": se_hat,
            "n_phecodes_used_pop": int(n_used),
            "i2_within_pop": I2,
            "scale_mix_pop": int(mix),
        })

    out = (
        df.groupby(grouping_cols + ["pop"])
          .apply(_one_group)
          .reset_index()
    )
    return out

def build_disease_overall_estimates(disease_pop_df, grouping_cols):
    """
    Stage B: Across populations (per disease) random-effects meta-analysis.
    Outputs one row per disease with:
      h2_overall_RE, se_overall_RE, ci95_l_overall_RE, ci95_u_overall_RE,
      tau2_between_pops, I2_between_pops, pred_int_l, pred_int_u,
      n_pops_used, any_scale_mix_flag
    """
    if disease_pop_df is None or disease_pop_df.empty:
        cols = grouping_cols + [
            "h2_overall_RE","se_overall_RE","ci95_l_overall_RE","ci95_u_overall_RE",
            "tau2_between_pops","I2_between_pops","pred_int_l","pred_int_u",
            "n_pops_used","any_scale_mix_flag"
        ]
        return pd.DataFrame(columns=cols)

    def _one_disease(g: pd.DataFrame) -> pd.Series:
        thetas = g["theta_pop"].to_numpy(dtype=float)
        ses    = g["se_pop"].to_numpy(dtype=float)
        meta   = re_meta_pop(thetas, ses, method="DL", kh=True)
        any_mix = int((g["scale_mix_pop"] == 1).any())
        return pd.Series({
            "h2_overall_RE": meta["mu"],
            "se_overall_RE": meta["se_mu"],
            "ci95_l_overall_RE": meta["ci95_l"],
            "ci95_u_overall_RE": meta["ci95_u"],
            "tau2_between_pops": meta["tau2"],
            "I2_between_pops": meta["I2"],
            "pred_int_l": meta["pred_int_l"],
            "pred_int_u": meta["pred_int_u"],
            "n_pops_used": int(meta["n_pops_used"]),
            "any_scale_mix_flag": any_mix
        })

    out = (
        disease_pop_df.groupby(grouping_cols)
                      .apply(_one_disease)
                      .reset_index()
    )
    return out

# =============================================================================
# Main
# =============================================================================

def main():
    """
    Build the master TSV with:
      1) ANY-pop detection (ACAT + BH) at a >H2_THRESHOLD boundary (unchanged),
      2) PLUS a principled 'overall' heritability estimate per disease using:
         - Stage A: IVW across mapped PheCodes within each population
         - Stage B: random-effects meta-analysis across populations
         The final overall point estimate is stored as 'h2_overall_RE' (float).

    Assumes the following helpers are defined in this module:
      - read_bgz_tsv, list_join_safe, acat, bh_fdr, combine_codes_unique_sorted
      - ivw_fixed, cochran_q_i2, re_meta_pop
      - build_disease_pop_estimates, build_disease_overall_estimates
    """
    import io
    import gzip
    import math
    import urllib.request
    import numpy as np
    import pandas as pd

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

        # Ensure key columns exist
        for col in [
            "estimates.final.h2_liability", "estimates.final.h2_liability_se",
            "estimates.final.h2_observed", "estimates.final.h2_observed_se"
        ]:
            if col not in h2_raw_df.columns:
                h2_raw_df[col] = np.nan

        # Prefer liability; fallback to observed; also record scale_used for diagnostics
        liab  = h2_raw_df["estimates.final.h2_liability"]
        liab_se = h2_raw_df["estimates.final.h2_liability_se"]
        obs   = h2_raw_df["estimates.final.h2_observed"]
        obs_se  = h2_raw_df["estimates.final.h2_observed_se"]

        use_liab = liab.notna() & liab_se.notna()
        h2_raw_df["h2"] = np.where(use_liab, liab, obs)
        h2_raw_df["se"] = np.where(use_liab, liab_se, obs_se)
        h2_raw_df["scale_used"] = np.where(use_liab, "liability", "observed")

        # Drop rows with missing/invalid SE or h2
        h2_rows = h2_raw_df[
            h2_raw_df["h2"].notna() & h2_raw_df["se"].notna() & (h2_raw_df["se"] > 0)
        ][["phenocode", "pop", "h2", "se", "scale_used"]].copy()

        # One-sided p-values vs threshold H2_THRESHOLD
        z = (h2_rows["h2"].to_numpy(dtype=float) - H2_THRESHOLD) / h2_rows["se"].to_numpy(dtype=float)
        p_one = np.array([0.5 * math.erfc(val / math.sqrt(2.0)) for val in z], dtype=float)
        p_one = np.clip(p_one, 1e-15, 1.0 - 1e-15)
        h2_rows["p_one"] = p_one

        # ACAT + Bonferroni per phenocode (for detection)
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
            .reset_index()
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
        mapped_set = set(
            ukbb_lookup_df.loc[
                ukbb_lookup_df["ICD"].isin(phecodex_df["ICD"].dropna().astype(str)),
                "ukbb_phenocode"
            ].dropna().astype(str).unique().tolist()
        )
        considered_set = set(per_pheno.loc[per_pheno["p_any_acat"].notna(), "phenocode"].astype(str).unique().tolist())
        manifest_preqc_set = set(h2_pre_qc_df["phenocode"].dropna().astype(str).unique().tolist())
        manifest_postqc_set = set(h2_raw_df["phenocode"].dropna().astype(str).unique().tolist())
        missing_set = sorted(mapped_set - considered_set)

        def summarize_issues_one(phenocode: str) -> dict:
            pre = h2_pre_qc_df[h2_pre_qc_df["phenocode"].astype(str) == phenocode]
            post = h2_raw_df[h2_raw_df["phenocode"].astype(str) == phenocode]

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

            n_pops_preqc = int(pre.shape[0])
            n_pops_postqc = int(post.shape[0])
            n_h2_preqc = int(pre["_h2_pref"].notna().sum())
            n_se_preqc = int(pre["_se_pref"].notna().sum())
            n_sepos_postqc = int((post["_se_pref"].notna() & (post["_se_pref"] > 0)).sum())

            pops_preqc = ";".join(sorted(pre["pop"].dropna().astype(str).unique().tolist())) if n_pops_preqc else ""
            pops_postqc = ";".join(sorted(post["pop"].dropna().astype(str).unique().tolist())) if n_pops_postqc else ""
            pops_sepos = ";".join(sorted(post.loc[(post["_se_pref"].notna()) & (post["_se_pref"] > 0), "pop"].dropna().astype(str).unique().tolist()))

            if phenocode not in manifest_preqc_set:
                reason = "absent_from_manifest"
            elif n_pops_postqc == 0 and USE_QC:
                reason = "removed_by_qc_all_pops"
            elif n_sepos_postqc == 0:
                if n_se_preqc == 0:
                    reason = "missing_se_all_pops"
                else:
                    reason = "nonpositive_or_missing_se_all_pops"
            else:
                reason = "other_not_considered"

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
    # PHASE 3: MERGING, DETECTION, AND ESTIMATION
    # =========================================================================
    print("Enriching data with UKBB PheCodes and Heritability stats...")

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
        .reset_index()
    )

    # -------- Detection branch (unchanged logic) --------
    long_df = aggregated_df.explode("ukbb_phenocode")
    long_df["ukbb_phenocode"] = long_df["ukbb_phenocode"].astype("string")
    per_pheno["phenocode"] = per_pheno["phenocode"].astype("string")

    long_joined = long_df.merge(
        per_pheno[["phenocode", "phenocode_is_gt5_fdr", "eur_h2_mean"]],
        left_on="ukbb_phenocode",
        right_on="phenocode",
        how="left"
    )

    disease_signals = (
        long_joined.groupby(grouping_cols, as_index=False)
        .agg({
            "phenocode_is_gt5_fdr": "max",
            "eur_h2_mean": "mean"
        })
        .rename(columns={
            "phenocode_is_gt5_fdr": "is_h2_significant_in_any_ancestry",
            "eur_h2_mean": "h2_eur_avg"
        })
    )

    # Merge detection signals back
    final_df = aggregated_df.merge(disease_signals, on=grouping_cols, how="left")
    final_df["is_h2_significant_in_any_ancestry"] = (
        final_df["is_h2_significant_in_any_ancestry"].fillna(0).astype("int64")
    )

    # -------- Estimation branch (new: Stage A + Stage B) --------
    # Build a long table of (disease keys, phenocode) x (pop, h2, se, scale_used)
    disease_pheno_pop_long = (
        long_df.merge(
            h2_rows.rename(columns={"phenocode": "ukbb_phenocode"}),
            on="ukbb_phenocode",
            how="left"
        )
    )
    # Note: rows with no matched h2/se will carry NaNs and be dropped in Stage A

    # Stage A: IVW across mapped PheCodes within each population (per disease × pop)
    disease_pop_df = build_disease_pop_estimates(
        disease_pheno_pop_long,
        grouping_cols=grouping_cols
    )

    # Stage B: Random-effects meta across populations (per disease)
    disease_overall_df = build_disease_overall_estimates(
        disease_pop_df,
        grouping_cols=grouping_cols
    )
    # Merge overall estimate into final_df
    final_df = final_df.merge(disease_overall_df, on=grouping_cols, how="left")

    # --- Reporting summaries ---
    n_diseases = int(final_df.shape[0])
    n_diseases_sig = int(final_df["is_h2_significant_in_any_ancestry"].sum())
    n_overall = int(final_df["h2_overall_RE"].notna().sum()) if "h2_overall_RE" in final_df.columns else 0

    print("Disease-level summary:")
    print(f"  Total PheCodeX diseases: {n_diseases}")
    print(f"  With > {H2_THRESHOLD*100:.0f}% any-pop signal (via mapped PheCodes, FDR): {n_diseases_sig}")
    print(f"  With overall heritability computed (Stage B): {n_overall}")

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

    # Keep 'h2_overall_RE' as a numeric float in the master file (Program Two will pick it up)
    # If you prefer rounded display, uncomment:
    # final_df["h2_overall_RE"] = final_df["h2_overall_RE"].round(4)

    # Select and order final columns (add h2_overall_RE so Program Two can surface it)
    final_df = final_df[[
        "phecode",
        "ukbb_phenocode",
        "phecode_string",
        "phecode_category",
        "is_h2_significant_in_any_ancestry",
        "h2_eur_avg",
        "icd9_codes",
        "icd10_codes",
        "h2_overall_RE"  # <-- new column, numeric
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
