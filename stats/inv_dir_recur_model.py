import sys
import math
import warnings
from typing import Tuple, Dict, Iterable, Optional
import re
import itertools

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.contingency_tables import mcnemar


from scipy.stats import wilcoxon

# ------------------------- FILE PATHS -------------------------

OUTPUT_CSV  = "./output.csv"
INVINFO_TSV = "./inv_info.tsv"

# Save outputs
SAVE_TABLES = True
OUT_MODEL_A_TABLE = "modelA_effects.csv"
OUT_MODEL_B_TABLE = "modelB_effects.csv"
OUT_FLOOR_SWEEP   = "floor_sweep.csv"
OUT_INFLUENCE     = "influence_top.csv"
OUT_DFBETAS       = "influence_dfbetas.csv"
OUT_TOST          = "tost_recurrent.csv"

# ------------------------- SETTINGS --------------------------

# Floor (epsilon) for logs (STRICT: only quantile method; no other fallbacks)
FLOOR_QUANTILE = 0.01
MIN_FLOOR      = 1e-8

# Sensitivities (these are analyses, not fallbacks)
RUN_NONZERO_SENSITIVITY = True
RUN_PERMUTATION_TEST    = True
N_PERMUTATIONS          = 10000
PERM_SEED               = 2025
RUN_PERM_STRATIFIED     = True
PERM_STRATA_COL         = "chr_std"     # must exist or stratified test is skipped with error

RUN_FLOOR_SWEEP         = True
SWEEP_QUANTILES         = [0.005, 0.01, 0.02, 0.05, 0.10]
EXTRA_FLOORS            = [2e-4]        # very large epsilon to prove robustness

RUN_TOST                = True
TOST_MARGIN_RATIO       = 1.20          # ±20% equivalence window on ratio scale

SHOW_TOP_INFLUENCERS    = 10
RATIO_DISPLAY_FLOOR     = 1e-3

# ------------------------- FORMATTING -------------------------

def _fmt_p(p: float) -> str:
    if not np.isfinite(p): return "NA"
    if p < 1e-99: return "<1e-99"
    if p < 1e-3:  return f"{p:.1e}"
    return f"{p:.3f}"

def _fmt_ratio(r: float) -> str:
    if not np.isfinite(r): return "NA"
    if r < RATIO_DISPLAY_FLOOR:
        return f"<{RATIO_DISPLAY_FLOOR:.3f}×"
    return f"{r:.3f}×"

def _fmt_pct(r: float) -> str:
    return "NA" if not np.isfinite(r) else f"{(r-1.0)*100.0:+.1f}%"

def _fmt_ci(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)): return "[NA, NA]"
    return f"[{lo:.3f}, {hi:.3f}]"

def _print_header(s: str):
    print("\n" + s)
    print("-" * len(s))

# ------------------------- HELPERS ---------------------------

def _standardize_chr(val: str) -> str:
    s = str(val).strip()
    return s[3:] if s.lower().startswith("chr") else s

def _linear_combo(res, weights: Dict[str, float]) -> Tuple[float, float, float]:
    """Linear contrast of params, returns (est, se, p) for H0: L*beta = 0."""
    pnames = list(res.params.index)
    L = np.zeros((1, len(pnames)), dtype=float)
    for k, w in weights.items():
        if k not in pnames:
            raise KeyError(f"Parameter '{k}' not in model. Available: {pnames}")
        L[0, pnames.index(k)] = float(w)
    ttest = res.t_test(L)
    est = float(np.squeeze(ttest.effect))
    se  = float(np.squeeze(ttest.sd))
    p   = float(np.squeeze(ttest.pvalue))
    return est, se, p

def _pack_effect_row(label: str, est: float, se: float) -> Dict[str, float]:
    lo, hi = est - 1.96*se, est + 1.96*se
    ratio, lo_r, hi_r = math.exp(est), math.exp(lo), math.exp(hi)
    return dict(effect=label, ratio=ratio, ci_low=lo_r, ci_high=hi_r, pct=(ratio-1.0)*100.0)

# ------------------------- FLOOR (EPSILON) -------------------

def choose_floor_from_quantile(pi_all: np.ndarray, q: float, min_floor: float) -> float:
    pos = pi_all[np.isfinite(pi_all) & (pi_all > 0)]
    if pos.size == 0:
        raise ValueError("All π values are non-positive; cannot choose floor from quantile.")
    return max(float(np.quantile(pos, q)) * 0.5, min_floor)

# ------------------------- LOADING & STRICT MATCHING --------

def load_and_match(output_csv: str, invinfo_tsv: str) -> pd.DataFrame:
    """
    STRICT loader:
      - Requires inv_info.tsv has columns: Chromosome, Start, End, 0_single_1_recur_consensus
      - Crashes if inv_info has duplicate keys (chr_std, Start, End)
      - Builds 9 candidate (Start,End) per region with ±1 bp tolerance
      - Keeps only true matches; for each region, picks the minimal match_priority
        and requires exactly ONE inv row at that best priority.
      - Returns matched table with both π values present and finite.
    """
    df  = pd.read_csv(output_csv)
    inv = pd.read_csv(invinfo_tsv, sep='\t')

    # enforce required columns
    need_df = ["chr", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"]
    miss_df = [c for c in need_df if c not in df.columns]
    if miss_df:
        raise KeyError(f"{output_csv} missing columns: {miss_df}")

    need_inv = ["Chromosome", "Start", "End", "0_single_1_recur_consensus"]
    miss_inv = [c for c in need_inv if c not in inv.columns]
    if miss_inv:
        raise KeyError(f"{invinfo_tsv} missing columns: {miss_inv}")

    # standardize chromosomes
    df["chr_std"]  = df["chr"].apply(_standardize_chr)
    inv["chr_std"] = inv["Chromosome"].apply(_standardize_chr)

    # check duplicates in inv_info keys → CRASH if any
    dup_keys = inv.duplicated(subset=["chr_std", "Start", "End"], keep=False)
    if dup_keys.any():
        bad = inv.loc[dup_keys, ["chr_std", "Start", "End"]].drop_duplicates()
        raise ValueError(f"inv_info.tsv contains duplicate (chr,Start,End) keys. Offending keys:\n{bad.to_string(index=False)}")

    # compact df
    df_small = df[["chr_std", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"]].rename(
        columns={"0_pi_filtered": "pi_direct", "1_pi_filtered": "pi_inverted"}
    ).copy()
    df_small["region_start"] = df_small["region_start"].astype(int)
    df_small["region_end"]   = df_small["region_end"].astype(int)

    # build ±1 bp candidate keys (9 per region)
    cands = []
    for ds in (-1, 0, 1):
        for de in (-1, 0, 1):
            tmp = df_small.copy()
            tmp["Start"] = tmp["region_start"] + ds
            tmp["End"]   = tmp["region_end"]   + de
            tmp["match_priority"] = abs(ds) + abs(de)  # 0 (exact), 1, or 2
            cands.append(tmp)
    df_cand = pd.concat(cands, ignore_index=True)

    inv_small = inv[["chr_std", "Start", "End", "0_single_1_recur_consensus"]].copy()
    merged = df_cand.merge(inv_small, on=["chr_std", "Start", "End"], how="inner")  # keep only true matches

    if merged.empty:
        raise RuntimeError("No regions matched inv_info under ±1 bp tolerance.")

    # For each region (chr_std, region_start, region_end) select ONE row:
    # - minimal match_priority present
    # - after deduplicating inv targets (Start,End), require exactly one → else CRASH
    key = ["chr_std", "region_start", "region_end"]

    def pick_one(g: pd.DataFrame) -> pd.DataFrame:
        mp = int(g["match_priority"].min())
        gg = g[g["match_priority"] == mp].drop_duplicates(subset=["Start","End"]).copy()
        if gg.shape[0] != 1:
            # Real ambiguity at best priority → CRASH
            raise ValueError(
                "Ambiguous inv mapping at best priority for region "
                f"{g.name[0]}:{int(g.name[1])}-{int(g.name[2])} ; "
                f"candidates={gg[['Start','End','0_single_1_recur_consensus']].to_dict(orient='records')}"
            )
        return gg.iloc[[0]]

    best = (merged.groupby(key, group_keys=True)
                  .apply(pick_one, include_groups=False)
                  .droplevel(-1)
                  .reset_index())

    if best.empty:
        raise RuntimeError("After strict selection, no regions remained. (This should not happen.)")

    # Map recurrence
    best["Recurrence"] = pd.to_numeric(best["0_single_1_recur_consensus"], errors="coerce").map({0:"Single-event", 1:"Recurrent"})
    best = best[~best["Recurrence"].isna()].copy()

    # numeric cleanup and π requirements
    best["pi_direct"]   = pd.to_numeric(best["pi_direct"],   errors="coerce")
    best["pi_inverted"] = pd.to_numeric(best["pi_inverted"], errors="coerce")
    best = best.dropna(subset=["pi_direct","pi_inverted"])
    best = best[np.isfinite(best["pi_direct"]) & np.isfinite(best["pi_inverted"])].copy()

    if best.empty:
        raise RuntimeError("No region retained both finite π values after matching.")

    # attach region_id
    best["region_id"] = (
        best["chr_std"].astype(str) + ":" +
        best["region_start"].astype(int).astype(str) + "-" +
        best["region_end"].astype(int).astype(str)
    )

    # final columns
    cols = ["region_id","Recurrence","chr_std","region_start","region_end","Start","End","pi_direct","pi_inverted"]
    return best[cols].copy()

# ------------------------- MODEL A (PRIMARY) -----------------

def run_model_A(matched: pd.DataFrame, eps: float, nonzero_only: bool=False):
    """
    Δ-logπ model: log((π_inv + eps)/(π_dir + eps)) ~ Recurrent, HC3 SEs.
    No weighting, no fallbacks.
    """
    dfA = matched.copy()
    if nonzero_only:
        keep = (dfA["pi_direct"] > 0) & (dfA["pi_inverted"] > 0)
        dfA = dfA.loc[keep].copy()

    dfA["logFC"] = np.log(dfA["pi_inverted"].to_numpy(float) + eps) \
                 - np.log(dfA["pi_direct"  ].to_numpy(float) + eps)
    dfA["Recurrent"] = (dfA["Recurrence"] == "Recurrent").astype(int)

    X = sm.add_constant(dfA[["Recurrent"]])
    res = sm.OLS(dfA["logFC"], X).fit(cov_type="HC3")

    # Contrasts (coding-invariant)
    est_SE,  se_SE,  p_SE  = _linear_combo(res, {"const":1.0})
    est_RE,  se_RE,  p_RE  = _linear_combo(res, {"const":1.0, "Recurrent":1.0})
    est_INT, se_INT, p_INT = _linear_combo(res, {"Recurrent":1.0})

    # Overall pooled Δ-logπ
    res_all = sm.OLS(dfA["logFC"], np.ones((dfA.shape[0],1))).fit(cov_type="HC3")
    est_ALL = float(res_all.params.iloc[0]); se_ALL = float(res_all.bse.iloc[0]); p_ALL = float(res_all.pvalues.iloc[0])

    tab = pd.DataFrame([
        {**_pack_effect_row("Single-event: Inverted vs Direct", est_SE, se_SE), "p": p_SE},
        {**_pack_effect_row("Recurrent: Inverted vs Direct",    est_RE, se_RE), "p": p_RE},
        {**_pack_effect_row("Interaction (difference between those two)", est_INT, se_INT), "p": p_INT},
        {**_pack_effect_row("Overall inversion effect (pooled Δ-logπ)",    est_ALL, se_ALL), "p": p_ALL},
    ])

    return res, tab, dfA

# ------------------------- MODEL B (CONFIRMATORY) -----------

def run_model_B(matched: pd.DataFrame, eps: float):
    """
    Fixed-effects confirmation (no random effects; no try/except):
      log_pi ~ Inverted + Inverted:Recurrent + C(region_id), cluster-robust by region
    """
    rows = []
    for _, r in matched.iterrows():
        rows.append({"region_id": r["region_id"], "Recurrence": r["Recurrence"], "status":"Direct",   "pi": r["pi_direct"]})
        rows.append({"region_id": r["region_id"], "Recurrence": r["Recurrence"], "status":"Inverted", "pi": r["pi_inverted"]})
    d = pd.DataFrame(rows)

    d["log_pi"]   = np.log(d["pi"].to_numpy(float) + float(eps))
    d["Inverted"] = (d["status"] == "Inverted").astype(int)
    d["Recurrent"]= (d["Recurrence"] == "Recurrent").astype(int)

    # Recurrence main effect is absorbed by C(region_id) and intentionally omitted
    res = smf.ols("log_pi ~ Inverted + Inverted:Recurrent + C(region_id)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["region_id"]}
    )

    est_SE,  se_SE,  p_SE  = _linear_combo(res, {"Inverted":1.0})
    est_RE,  se_RE,  p_RE  = _linear_combo(res, {"Inverted":1.0, "Inverted:Recurrent":1.0})
    est_INT, se_INT, p_INT = _linear_combo(res, {"Inverted:Recurrent":1.0})

    tab = pd.DataFrame([
        {**_pack_effect_row("Single-event: Inverted vs Direct", est_SE, se_SE), "p": p_SE},
        {**_pack_effect_row("Recurrent: Inverted vs Direct",    est_RE, se_RE), "p": p_RE},
        {**_pack_effect_row("Interaction (difference between those two)", est_INT, se_INT), "p": p_INT},
    ])

    # Overall pooled paired effect (ignoring Recurrence), same fixed structure
    res_overall = smf.ols("log_pi ~ Inverted + C(region_id)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["region_id"]}
    )

    return res, tab, d, res_overall


# ==== MODEL C: helpers + run_model_C ====

# New output filename for Model C (used by the rewritten main)
OUT_MODEL_C_TABLE = "modelC_effects.csv"

def _parse_percent_to_prop(val) -> float:
    """
    Convert strings like '95%' to 0.95.
    If already numeric in [0,1] or [0,100], normalize accordingly.
    Non-parsable -> NaN.
    """
    if val is None:
        return float("nan")
    s = str(val).strip()
    if s == "" or s.upper() == "NA":
        return float("nan")
    try:
        if s.endswith("%"):
            num = float(s[:-1].strip())
            return num / 100.0
        # already numeric?
        x = float(s)
        if 0.0 <= x <= 1.0:
            return x
        if 1.0 < x <= 100.0:
            return x / 100.0
        return float("nan")
    except Exception:
        return float("nan")

_CI_LEAD_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)(?:\s|\[|$)", re.IGNORECASE)

def _parse_ci_point(val) -> float:
    """
    Extract the leading numeric point estimate from strings like:
      '1.02e-04 [2.72e-05 ,1.21e-04]'  -> 1.02e-04
    Also accepts plain numerics. Non-parsable -> NaN.
    """
    if val is None:
        return float("nan")
    s = str(val).strip()
    if s == "" or s.upper() == "NA":
        return float("nan")
    m = _CI_LEAD_RE.match(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

def _to_float(val) -> float:
    """Coerce to float; NA on failure."""
    try:
        x = float(val)
        if np.isfinite(x):
            return x
        return float("nan")
    except Exception:
        return float("nan")

def _to_int(val) -> float:
    """Coerce to integer (returned as float for modeling); NA on failure."""
    try:
        x = int(float(val))
        return float(x)
    except Exception:
        return float("nan")

def _safe_log_pos(x: pd.Series) -> pd.Series:
    """Natural log for strictly positive x; else NaN."""
    x = pd.to_numeric(x, errors="coerce")
    return np.log(x.where(x > 0.0, np.nan))

def _ln1p_count(x: pd.Series) -> pd.Series:
    """ln(1 + count) for counts ≥0; NA if negative/non-numeric."""
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x >= 0.0, np.nan)
    return np.log1p(x)

def _zscore_series(s: pd.Series) -> pd.Series:
    """Z-score (mean 0, sd 1) ignoring NaNs; returns NaNs where input NaN; drops constants to 0."""
    mu = np.nanmean(s.to_numpy(float))
    sd = np.nanstd(s.to_numpy(float), ddof=0)
    if not np.isfinite(sd) or sd == 0.0:
        # If constant/degenerate, return zeros (kept so model stays stable but contributes nothing)
        return s.apply(lambda v: 0.0 if np.isfinite(v) else float("nan"))
    return (s - mu) / sd

def _impute_by_recurrence_transformed(df: pd.DataFrame, col: str, rec_col: str = "Recurrence") -> Tuple[pd.Series, pd.Series]:
    """
    Impute transformed feature 'col' by Recurrence-class median.
    Returns (imputed_series, missing_indicator)
    """
    miss = df[col].isna().astype(int)
    # per-class medians in transformed scale
    meds = df.groupby(rec_col)[col].median()
    overall = np.nanmedian(df[col].to_numpy(float))
    def fill_row(row):
        if not np.isfinite(row[col]):
            # class median
            cm = meds.get(row[rec_col], np.nan)
            if not np.isfinite(cm):
                cm = overall
            if not np.isfinite(cm):
                cm = 0.0
            return float(cm)
        return float(row[col])
    filled = df.apply(fill_row, axis=1).astype(float)
    return filled, miss

def _prepare_invinfo_covariates(invinfo_path: Optional[str]) -> pd.DataFrame:
    """
    Load inv_info.tsv, filter to 0/1 in consensus, compute cleaned & transformed covariates.
    """
    path = invinfo_path if invinfo_path is not None else INVINFO_TSV
    inv = pd.read_csv(path, sep="\t")

    # Standardize chromosome label
    inv["chr_std"] = inv["Chromosome"].apply(_standardize_chr)

    # Enforce presence & 0/1 filter on consensus
    if "0_single_1_recur_consensus" not in inv.columns:
        raise KeyError(f"{path} missing column: 0_single_1_recur_consensus")
    inv["_cons"] = pd.to_numeric(inv["0_single_1_recur_consensus"], errors="coerce")
    inv = inv[inv["_cons"].isin([0, 1])].copy()

    # Check for duplicate keys (strict)
    dup = inv.duplicated(subset=["chr_std", "Start", "End"], keep=False)
    if dup.any():
        bad = inv.loc[dup, ["chr_std", "Start", "End"]].drop_duplicates()
        raise ValueError(f"{path} contains duplicate (chr,Start,End) keys.\n{bad.to_string(index=False)}")

    # Raw numeric extractions
    inv["_nr_events"]      = inv["Number_recurrent_events"].apply(_to_int) if "Number_recurrent_events" in inv.columns else float("nan")
    inv["_size_kbp"]       = pd.to_numeric(inv.get("Size_.kbp.", np.nan), errors="coerce")
    inv["_inverted_af"]    = pd.to_numeric(inv.get("Inverted_AF", np.nan), errors="coerce")
    inv["_form_rate"]      = inv.get("Formation_rate_per_generation_.95..C.I..", np.nan)
    inv["_form_rate"]      = inv["_form_rate"].apply(_parse_ci_point)

    # Transformations
    inv["T_nr_events"]        = _ln1p_count(inv["_nr_events"])
    inv["T_ln_size_kbp"]      = _safe_log_pos(inv["_size_kbp"])
    inv["T_inverted_af"]      = pd.to_numeric(inv["_inverted_af"], errors="coerce")  # keep on raw scale (can be negative)
    inv["T_ln_form_rate"]     = _safe_log_pos(inv["_form_rate"])

    # Assemble transformed columns
    tcols = [
        ("T_nr_events",        "Z_nr_events"),
        ("T_ln_size_kbp",      "Z_ln_size_kbp"),
        ("T_inverted_af",      "Z_inverted_af"),
        ("T_ln_form_rate",     "Z_ln_formation_rate"),
    ]

    tidy = inv[["chr_std", "Start", "End", "_cons"]].copy()
    tidy.rename(columns={"_cons": "consensus_numeric"}, inplace=True)

    for tcol, zcol in tcols:
        tidy[zcol] = _zscore_series(inv[tcol])
        # No NA indicators; missing values are disallowed downstream.


    return tidy

def run_model_C(matched: pd.DataFrame, invinfo_path: Optional[str], eps: float, nonzero_only: bool=False):
    """
    MODEL C:
      - Outcome: logFC = log(pi_inverted+eps) - log(pi_direct+eps)
      - Predictors: Recurrent + z-scored covariates from inv_info.tsv (with missingness dummies)
      - SEs: HC3 robust
      - Reporting: rows for SE, RE, Interaction, plus each covariate (per +1 SD)
    Returns:
      resC (RegressionResultsWrapper), tabC (effects table), dfC (model frame), used_covariates (list)
    """
    # Start from matched (already 0/1 consensus-only, finite π)
    dfC = matched.copy()

    # Outcome and group
    if nonzero_only:
        keep = (dfC["pi_direct"] > 0) & (dfC["pi_inverted"] > 0)
        dfC = dfC.loc[keep].copy()

    dfC["logFC"] = np.log(dfC["pi_inverted"].to_numpy(float) + float(eps)) \
                 - np.log(dfC["pi_direct"  ].to_numpy(float) + float(eps))
    dfC["Recurrent"] = (dfC["Recurrence"] == "Recurrent").astype(int)

    # Bring covariates
    cov = _prepare_invinfo_covariates(invinfo_path)

    # Merge by strict keys chosen earlier
    dfC = dfC.merge(cov, on=["chr_std", "Start", "End"], how="left", validate="m:1")

    # Build covariate lists
    z_covs = [
        "Z_nr_events",
        "Z_ln_size_kbp",
        "Z_inverted_af",
        "Z_ln_formation_rate",
    ]

    # No imputation; enforce completeness across all covariates
    missing_mask = dfC[z_covs].isna().any(axis=1)
    if bool(missing_mask.any()):
            bad = dfC.loc[missing_mask, ["region_id"] + z_covs].head(10)
            raise ValueError(f"MODEL C: Missing covariate values after merge; rows={int(missing_mask.sum())}. Example:\n{bad.to_string(index=False)}")
    
    # Drop covariates that are constant (all zeros after z-scoring) and verify finiteness
    used_covariates = []
    for zc in list(z_covs):
            col = pd.to_numeric(dfC[zc], errors="coerce")
            if not np.isfinite(col.to_numpy(float)).all():
                raise ValueError(f"MODEL C: Non-finite values detected in covariate '{zc}'.")
            if np.nanstd(col.to_numpy(float), ddof=0) == 0.0:
                z_covs.remove(zc)
            else:
                used_covariates.append(zc)
    
    na_covs = []  # ensure no NA indicators are used


    # Design matrix: const + Recurrent + Z-covariates + NA indicators
    X_parts = [dfC[["Recurrent"]]]
    if z_covs:
            X_parts.append(dfC[z_covs])
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X)


    # Fit with HC3 robust SEs
    resC = sm.OLS(dfC["logFC"], X).fit(cov_type="HC3")

    # Build effects table
    rows = []

    # 1) Single-event (adjusted): const
    est_SE, se_SE, p_SE = _linear_combo(resC, {"const": 1.0})
    rows.append({**_pack_effect_row("Single-event: Inverted vs Direct (adjusted)", est_SE, se_SE), "p": p_SE})

    # 2) Recurrent (adjusted): const + Recurrent
    est_RE, se_RE, p_RE = _linear_combo(resC, {"const": 1.0, "Recurrent": 1.0})
    rows.append({**_pack_effect_row("Recurrent: Inverted vs Direct (adjusted)", est_RE, se_RE), "p": p_RE})

    # 3) Interaction (difference between those two): Recurrent
    est_INT, se_INT, p_INT = _linear_combo(resC, {"Recurrent": 1.0})
    rows.append({**_pack_effect_row("Interaction (difference between those two)", est_INT, se_INT), "p": p_INT})

    # 4) Covariate main effects (per +1 SD increase on transformed scale)
    pretty = {
        "Z_nr_events": "Covariate: Number_recurrent_events (ln1p, z)",
        "Z_ln_size_kbp": "Covariate: Size_.kbp. (ln, z)",
        "Z_inverted_af": "Covariate: Inverted_AF (z)",
        "Z_ln_formation_rate": "Covariate: Formation_rate_per_generation (ln, z)",
    }
    for zc in used_covariates:
        est, se, p = _linear_combo(resC, {zc: 1.0})
        label = pretty.get(zc, f"Covariate: {zc}")
        rows.append({**_pack_effect_row(label, est, se), "p": p})

    tabC = pd.DataFrame(rows)
    return resC, tabC, dfC, used_covariates



# ==== MODEL D: Shapley attribution of the attenuation in the Recurrence effect (Model A → Model C) ====

# output filename for Model D
OUT_MODEL_D_TABLE = "modelD_attribution.csv"

def _add_const(df_or_arr: pd.DataFrame) -> pd.DataFrame:
    """Add intercept column; keeps column names stable."""
    return sm.add_constant(df_or_arr, has_constant='add')

def _fit_beta_R_HC3(y: np.ndarray, g: np.ndarray, Z: Optional[pd.DataFrame]) -> float:
    """
    Fit OLS with HC3 and return the coefficient on G ('Recurrent') using design: [const, G, Z_S].
    Coefficient estimate is identical under robust vs classic OLS; HC3 only affects SEs.
    """
    parts = [pd.Series(g, name="Recurrent")]
    if Z is not None and Z.shape[1] > 0:
        parts.append(Z)
    X = _add_const(pd.concat(parts, axis=1))
    res = sm.OLS(y, X).fit(cov_type="HC3")
    if "Recurrent" not in res.params.index:
        raise RuntimeError("MODEL D: 'Recurrent' coefficient missing from design—check inputs.")
    return float(res.params["Recurrent"])

def _residualize_on_S(vec: np.ndarray, Z: Optional[pd.DataFrame]) -> np.ndarray:
    """
    Residualize a vector on Z (with intercept). Uses plain OLS (robustness irrelevant for projection).
    Returns residuals with an intercept included in the regression.
    """
    if Z is None or Z.shape[1] == 0:
        # Only intercept: residuals are vec - mean(vec)
        return vec - np.mean(vec)
    X = _add_const(Z)
    # Use np.linalg.lstsq for numerical stability
    beta, *_ = np.linalg.lstsq(X.to_numpy(float), vec.astype(float), rcond=None)
    fitted = X.to_numpy(float) @ beta
    return vec.astype(float) - fitted

def _subset_key(cols: Iterable[str]) -> frozenset:
    return frozenset(cols)

def run_model_D(dfC: pd.DataFrame, used_covariates: Iterable[str]):
    """
    MODEL D (exact, order-free Shapley decomposition):
      - Operates on the SAME analysis rows and ε as Model C (consume dfC and used_covariates from run_model_C).
      - Attributes the total attenuation in the 'Recurrent' coefficient from:
            A0:  Y ~ Recurrent        (on Model-C rows)   → β_R(∅)
        to  C:   Y ~ Recurrent + Z     (full Model C set)  → β_R(Z)
        where Y = logFC, G = Recurrent (0/1), and Z = used_covariates (z-scored).

      - Returns:
          tabD : DataFrame with per-covariate Shapley contributions (log units), ratio, % of total drop,
                 and a split into covariance vs variance channels (FWL).
          summaryD : dict with beta_A0, beta_C, delta_beta, n_covariates.
    """
    if dfC is None or used_covariates is None:
        raise ValueError("MODEL D requires dfC and used_covariates from Model C.")
    Z_all = list(used_covariates)
    p = len(Z_all)
    if p == 0:
        raise ValueError("MODEL D: No covariates available in Model C (used_covariates empty).")

    # Extract core vectors/matrix from dfC (SAME rows / ε as Model C)
    y = dfC["logFC"].to_numpy(float)
    g = dfC["Recurrent"].to_numpy(int).astype(float)
    Zmat = dfC[Z_all].copy()

    # Baseline A0 (on Model C rows): Y ~ Recurrent
    beta_A0 = _fit_beta_R_HC3(y, g, Z=None)

    # Full C: Y ~ Recurrent + Z_all
    beta_C  = _fit_beta_R_HC3(y, g, Z=Zmat)

    delta_beta = beta_A0 - beta_C

    # Precompute β_R(S), C(S), V(S) for all subsets S ⊆ Z
    # Use dictionary keyed by frozenset of column names
    beta_map: Dict[frozenset, float] = {}
    C_map:    Dict[frozenset, float] = {}
    V_map:    Dict[frozenset, float] = {}

    # Helper: build Z for a subset
    def Z_of(S: frozenset) -> Optional[pd.DataFrame]:
        if not S:
            return None
        return Zmat[list(S)]

    n = float(len(y))
    tiny = 1e-12

    # Enumerate all subsets once (2^p)
    all_cols = Z_all
    for r in range(0, p + 1):
        for comb in itertools.combinations(all_cols, r):
            S = _subset_key(comb)
            ZS = Z_of(S)

            # β_R(S)
            beta_S = _fit_beta_R_HC3(y, g, ZS)
            beta_map[S] = beta_S

            # FWL residuals to get C(S) and V(S)
            rY = _residualize_on_S(y, ZS)
            rG = _residualize_on_S(g, ZS)

            # With an intercept in the residualization, residuals should be mean-zero; use population moments
            C_map[S] = float(np.dot(rG, rY) / n)
            V_map[S] = float(np.dot(rG, rG) / n)

    # Shapley contributions for each covariate k
    fact = math.factorial
    denom = float(fact(p))
    rows = []

    for k in Z_all:
        phi_total = 0.0
        phi_cov   = 0.0
        phi_var   = 0.0
        w_sum     = 0.0

        others = [c for c in Z_all if c != k]

        # All S ⊆ (Z\{k})
        for r in range(0, p):
            for comb in itertools.combinations(others, r):
                S = _subset_key(comb)
                Sk = _subset_key(set(comb) | {k})

                w = (fact(len(S)) * fact(p - len(S) - 1)) / denom

                beta_S  = beta_map[S]
                beta_Sk = beta_map[Sk]

                VS  = V_map[S]
                VSk = V_map[Sk]
                CS  = C_map[S]
                CSk = C_map[Sk]

                # Guard against degenerate residualized G variance
                if VSk <= tiny or VS <= tiny:
                    continue

                # Total marginal change in β_R when adding k to S
                d_beta = beta_S - beta_Sk

                # FWL channel split (exact algebra):
                # β(S) - β(Sk) = [C(S) - C(Sk)] / V(Sk)  +  β(S) * [V(Sk) - V(S)] / V(Sk)
                cov_term = (CS - CSk) / VSk
                var_term = beta_S * (VSk - VS) / VSk

                phi_total += w * d_beta
                phi_cov   += w * cov_term
                phi_var   += w * var_term
                w_sum     += w

        if w_sum <= 0.0:
            raise RuntimeError(f"MODEL D: All Shapley terms degenerate for covariate '{k}' (residualized variance ~ 0).")

        # Normalize if any degenerate subsets were skipped
        phi_total /= w_sum
        phi_cov   /= w_sum
        phi_var   /= w_sum

        rows.append({
            "covariate": k,
            "phi_log":   phi_total,
            "phi_ratio": float(math.exp(phi_total)),
            "phi_cov":   phi_cov,
            "phi_var":   phi_var,
        })

    tabD = pd.DataFrame(rows)
    # Add % of drop; handle near-zero delta safely
    if abs(delta_beta) < 1e-15:
        tabD["pct_of_drop"] = float("nan")
    else:
        tabD["pct_of_drop"] = 100.0 * tabD["phi_log"] / delta_beta

    # Order by absolute contribution
    tabD.sort_values(by="phi_log", key=lambda s: np.abs(s), ascending=False, inplace=True)
    tabD.reset_index(drop=True, inplace=True)

    summaryD = {
        "beta_A0": beta_A0,
        "beta_C":  beta_C,
        "delta_beta": delta_beta,
        "n_covariates": p,
    }
    return tabD, summaryD


# ------------------------- PERMUTATION TESTS ----------------

def perm_test_interaction(dfA: pd.DataFrame, n: int, seed: int) -> Tuple[float, float]:
    """Two-sided permutation on Recurrence labels for Δ-logπ difference."""
    rng = np.random.default_rng(seed)
    y = dfA["logFC"].to_numpy(float)
    g = dfA["Recurrent"].to_numpy(int)
    obs = float(np.nanmean(y[g==1]) - np.nanmean(y[g==0]))
    diffs = np.empty(n, dtype=float)
    for i in range(n):
        gp = rng.permutation(g)
        diffs[i] = float(np.nanmean(y[gp==1]) - np.nanmean(y[gp==0]))
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n + 1)
    return obs, p

def perm_test_interaction_stratified(dfA: pd.DataFrame, strata_col: str, n: int, seed: int) -> Tuple[float, float]:
    """Stratified permutation: permute Recurrence within each stratum."""
    if strata_col not in dfA.columns:
        raise KeyError(f"Strata column '{strata_col}' not found in dfA.")
    rng = np.random.default_rng(seed)
    y = dfA["logFC"].to_numpy(float)
    g = dfA["Recurrent"].to_numpy(int)
    strata = dfA[strata_col].astype("category").cat.codes.to_numpy(int)

    obs = float(np.nanmean(y[g==1]) - np.nanmean(y[g==0]))
    diffs = np.empty(n, float)
    for i in range(n):
        gp = g.copy()
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            gp[idx] = rng.permutation(gp[idx])
        diffs[i] = float(np.nanmean(y[gp==1]) - np.nanmean(y[gp==0]))
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n + 1)
    return obs, p

# ------------------------- MCNEMAR --------------------------

def mcnemar_by_class(matched: pd.DataFrame):
    """Paired zero vs >0 test within each class (Single-event, Recurrent)."""
    _print_header("MCNEMAR (paired zero vs >0) within class")
    for grp in ["Single-event", "Recurrent"]:
        sub = matched.loc[matched["Recurrence"] == grp, ["pi_direct","pi_inverted"]].dropna()
        if sub.empty:
            print(f"  {grp:<13}  no data")
            continue
        direct_pos   = (sub["pi_direct"  ].to_numpy(float) > 0)
        inverted_pos = (sub["pi_inverted"].to_numpy(float) > 0)

        a = int(np.sum( direct_pos &  inverted_pos))  # both >0
        b = int(np.sum( direct_pos & ~inverted_pos))  # direct >0, inv == 0
        c = int(np.sum(~direct_pos &  inverted_pos))  # direct == 0, inv >0
        d = int(np.sum(~direct_pos & ~inverted_pos))  # both == 0
        tbl = np.array([[a, b], [c, d]], dtype=int)

        exact = (b + c) <= 25
        res = mcnemar(tbl, exact=exact, correction=not exact)
        p = float(getattr(res, 'pvalue', np.nan))
        print(f"  {grp:<13}  table=[[both>0, direct>0&inv=0],[direct=0&inv>0, both=0]]={tbl.tolist()}  p={_fmt_p(p)}  (exact={exact})")

# ------------------------- DIAGNOSTICS ----------------------

def cooks_distance_top(X: pd.DataFrame, y: pd.Series, k=SHOW_TOP_INFLUENCERS) -> pd.DataFrame:
    ols = sm.OLS(y, X).fit()
    infl = OLSInfluence(ols)
    cd = infl.cooks_distance[0]
    out = pd.DataFrame({"region_id": X.index, "cooks_d": cd})
    out.sort_values("cooks_d", ascending=False, inplace=True)
    return out.head(k).reset_index(drop=True)

def dfbetas_table(X: pd.DataFrame, y: pd.Series, colnames: Iterable[str]) -> pd.DataFrame:
    ols = sm.OLS(y, X).fit()
    infl = OLSInfluence(ols)
    dfb = pd.DataFrame(infl.dfbetas, index=X.index, columns=list(colnames))
    dfb["region_id"] = X.index
    return dfb

def print_diagnostics(matched: pd.DataFrame, dfA: pd.DataFrame, floor_used: float, resA, resB, resOverall):
    _print_header("DATA DIAGNOSTICS")
    n_regions = matched.shape[0]
    n_single  = int((matched["Recurrence"] == "Single-event").sum())
    n_recur   = int((matched["Recurrence"] == "Recurrent").sum())
    print(f"Paired regions kept: {n_regions} (Single-event: {n_single}, Recurrent: {n_recur})")

    def zero_counts(df, col): return int((df[col] <= 0).sum())
    z_dir_all = zero_counts(matched, "pi_direct")
    z_inv_all = zero_counts(matched, "pi_inverted")
    print(f"Zeros / nonpositive π  —  Direct: {z_dir_all},  Inverted: {z_inv_all}")
    for grp in ["Single-event", "Recurrent"]:
        sub = matched.loc[matched["Recurrence"] == grp]
        print(f"  {grp:<13}  zeros — Direct: {zero_counts(sub,'pi_direct')},  Inverted: {zero_counts(sub,'pi_inverted')}")

    print(f"Detection floor used for logs (applied equally to both arms): {floor_used:.3g}")
    touched = ((dfA["pi_direct"] < floor_used) | (dfA["pi_inverted"] < floor_used)).mean()
    print(f"Fraction of pairs touched by floor (either arm < ε): {touched:.3f}")

    def q(a):
        a = a[np.isfinite(a)]
        if a.size == 0: return "NA"
        qs = np.percentile(a, [0, 25, 50, 75, 100])
        return f"min={qs[0]:.3g}, Q1={qs[1]:.3g}, median={qs[2]:.3g}, Q3={qs[3]:.3g}, max={qs[4]:.3g}"
    print("π (Direct)   summary:", q(matched["pi_direct"].to_numpy(float)))
    print("π (Inverted) summary:", q(matched["pi_inverted"].to_numpy(float)))
    print("Δ-logπ summary (logFC):", q(dfA["logFC"].to_numpy(float)))

    print("\nPaired Wilcoxon (Direct vs Inverted) within each class:")
    for grp in ["Single-event", "Recurrent"]:
        sub = matched.loc[matched["Recurrence"] == grp, ["pi_direct","pi_inverted"]].dropna()
        pval = float("nan")
        if len(sub) >= 2:
            try:
                _, pval = wilcoxon(sub["pi_direct"].to_numpy(float),
                                   sub["pi_inverted"].to_numpy(float),
                                   alternative="two-sided", zero_method="wilcox")
            except Exception:
                try:
                    _, pval = wilcoxon(sub["pi_direct"].to_numpy(float),
                                       sub["pi_inverted"].to_numpy(float),
                                       alternative="two-sided", zero_method="zsplit")
                except Exception:
                    pval = float("nan")
        print(f"  {grp:<13} p = {_fmt_p(pval)}")

    _print_header("AGREEMENT CHECK (Model A vs Model B), log scale")
    a_SE = float(_linear_combo(resA, {"const":1.0})[0])
    b_SE = float(_linear_combo(resB, {"Inverted":1.0})[0])
    a_RE = float(_linear_combo(resA, {"const":1.0,"Recurrent":1.0})[0])
    b_RE = float(_linear_combo(resB, {"Inverted":1.0,"Inverted:Recurrent":1.0})[0])
    print(f"  Single-event  (A vs B): {a_SE:.6f} vs {b_SE:.6f}")
    print(f"  Recurrent     (A vs B): {a_RE:.6f} vs {b_RE:.6f}")

    _print_header("OVERALL INVERSION EFFECT (paired across all regions)")
    estO, seO, _ = _linear_combo(resOverall, {"Inverted":1.0})
    ratioO, loO, hiO = math.exp(estO), math.exp(estO-1.96*seO), math.exp(estO+1.96*seO)
    print(f"  Overall (pooled): {_fmt_ratio(ratioO)}  CI={_fmt_ci(loO, hiO)}  change={_fmt_pct(ratioO)}")

    _print_header("INFLUENCE (Model A) — top regions by Cook's distance")
    X = sm.add_constant(dfA[["Recurrent"]]); X.index = dfA["region_id"]
    y = dfA["logFC"]; y.index = dfA["region_id"]
    top = cooks_distance_top(X, y, k=SHOW_TOP_INFLUENCERS)
    if top.empty:
        print("  No influence results.")
    else:
        for i, row in top.iterrows():
            print(f"  {i+1:>2}. {row['region_id']:<20}  Cook's D = {row['cooks_d']:.4g}")
    try:
        if SAVE_TABLES:
            top.to_csv(OUT_INFLUENCE, index=False)
            dfb = dfbetas_table(X, y, ["const","Recurrent"])
            dfb.to_csv(OUT_DFBETAS, index=False)
    except Exception:
        pass

# ------------------------- FLOOR SWEEP ----------------------

def floor_sweep(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_pi = np.r_[matched["pi_direct"].to_numpy(float), matched["pi_inverted"].to_numpy(float)]

    floors: Dict[str, float] = {}
    for q in SWEEP_QUANTILES:
        floors[f"quantile_{q:.3%}"] = choose_floor_from_quantile(all_pi, q=q, min_floor=MIN_FLOOR)
    for v in EXTRA_FLOORS:
        floors[f"extra_{v:.0e}"] = float(v)

    for label, eps in floors.items():
        dfA = matched.copy()
        dfA["logFC"] = np.log(dfA["pi_inverted"] + eps) - np.log(dfA["pi_direct"] + eps)
        dfA["Recurrent"] = (dfA["Recurrence"] == "Recurrent").astype(int)

        res = sm.OLS(dfA["logFC"], sm.add_constant(dfA[["Recurrent"]])).fit(cov_type="HC3")
        est_SE, se_SE, p_SE = _linear_combo(res, {"const":1})
        est_RE, se_RE, p_RE = _linear_combo(res, {"const":1,"Recurrent":1})
        est_I,  se_I,  p_I  = _linear_combo(res, {"Recurrent":1})

        touched = ((dfA["pi_direct"] < eps) | (dfA["pi_inverted"] < eps)).mean()

        rows.append({
            "floor_label": label, "floor_value": eps, "frac_pairs_touched": float(touched),
            "SE_log": est_SE, "SE_ratio": float(np.exp(est_SE)), "SE_p": p_SE,
            "RE_log": est_RE, "RE_ratio": float(np.exp(est_RE)), "RE_p": p_RE,
            "INT_log": est_I,  "INT_ratio": float(np.exp(est_I)),  "INT_p": p_I,
        })
    return pd.DataFrame(rows)

# ------------------------- TOST (EQUIVALENCE) ---------------

def tost_equivalence_on_recurrent(resA, margin_ratio: float = TOST_MARGIN_RATIO) -> Tuple[float, float, float, float]:
    """
    TOST for recurrent inversion effect from Model A (log scale).
    Robust-SE context → use normal reference by design (no df games).
    """
    est_RE, se_RE, _ = _linear_combo(resA, {"const":1.0, "Recurrent":1.0})
    delta = math.log(float(margin_ratio))

    # Normal CDF
    from math import erf, sqrt
    cdf = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
    t1 = (est_RE + delta) / se_RE  # > -delta
    t2 = (delta - est_RE) / se_RE  # < +delta
    p_equiv = max(1 - cdf(t1), 1 - cdf(t2))
    return est_RE, se_RE, p_equiv, delta

# ------------------------- MAIN -----------------------------

def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 120)

    # 1) Load & STRICT match
    matched = load_and_match(OUTPUT_CSV, INVINFO_TSV)
    n_single = (matched['Recurrence'] == 'Single-event').sum()
    n_recur  = (matched['Recurrence'] == 'Recurrent').sum()
    print(f"Matched paired regions (STRICT): {matched.shape[0]}  (Single-event: {n_single}, Recurrent: {n_recur})")

    # 2) Choose epsilon (quantile-only, no fallback) — same rule as before
    all_pi = np.r_[matched["pi_direct"].to_numpy(float), matched["pi_inverted"].to_numpy(float)]
    floor_used = choose_floor_from_quantile(all_pi, q=FLOOR_QUANTILE, min_floor=MIN_FLOOR)

    # 3) Model A (primary)
    _print_header("MODEL A — Δ-logπ ~ Recurrence (HC3)")
    resA, tabA, dfA = run_model_A(matched, eps=floor_used, nonzero_only=False)
    for _, r in tabA.iterrows():
        print(f"{r['effect']:<52}  ratio={_fmt_ratio(r['ratio'])}  CI={_fmt_ci(r['ci_low'], r['ci_high'])}  change={_fmt_pct(r['ratio'])}  p={_fmt_p(r['p'])}")
    if SAVE_TABLES:
        try: tabA.to_csv(OUT_MODEL_A_TABLE, index=False)
        except Exception: pass

    # 4) Model B (confirmatory FE + cluster by region)
    _print_header("MODEL B — logπ ~ Inverted + Inverted:Recurrence + C(region_id)  (cluster-robust by region)")
    resB, tabB, longB, resOverall = run_model_B(matched, eps=floor_used)
    for _, r in tabB.iterrows():
        print(f"{r['effect']:<52}  ratio={_fmt_ratio(r['ratio'])}  CI={_fmt_ci(r['ci_low'], r['ci_high'])}  change={_fmt_pct(r['ratio'])}  p={_fmt_p(r['p'])}")
    if SAVE_TABLES:
        try: tabB.to_csv(OUT_MODEL_B_TABLE, index=False)
        except Exception: pass

    # 5) Model C (covariate-adjusted)
    _print_header("MODEL C — Δ-logπ ~ Recurrence + covariates from inv_info.tsv (HC3)")
    resC, tabC, dfC, used_covs = run_model_C(matched, invinfo_path=INVINFO_TSV, eps=floor_used, nonzero_only=False)
    for _, r in tabC.iterrows():
        print(f"{r['effect']:<52}  ratio={_fmt_ratio(r['ratio'])}  CI={_fmt_ci(r['ci_low'], r['ci_high'])}  change={_fmt_pct(r['ratio'])}  p={_fmt_p(r['p'])}")
    if SAVE_TABLES:
        try: tabC.to_csv(OUT_MODEL_C_TABLE, index=False)
        except Exception: pass
    
    # 5b) Model D — Shapley attribution of Recurrence effect attenuation (Model A → Model C)
    _print_header("MODEL D — Shapley attribution of Recurrence effect attenuation (Model A → Model C)")
    try:
        tabD, summaryD = run_model_D(dfC, used_covs)
        # Summary of attenuation
        betaA0 = summaryD["beta_A0"]; betaC = summaryD["beta_C"]; dB = summaryD["delta_beta"]
        print(f"Baseline A0 (on Model-C rows): β_R={betaA0:.6f}  ratio={_fmt_ratio(math.exp(betaA0))}")
        print(f"Full Model C:                 β_R={betaC:.6f}  ratio={_fmt_ratio(math.exp(betaC))}")
        print(f"Total attenuation Δβ_R (A→C):  {dB:.6f}  fold={_fmt_ratio(math.exp(dB))}")
    
        # Contributions (sorted by absolute log contribution)
        for _, r in tabD.iterrows():
            share = "NA" if not np.isfinite(r.get("pct_of_drop", np.nan)) else f"{r['pct_of_drop']:.1f}%"
            print(f"  {r['covariate']:<40}  contribution(log)={r['phi_log']:+.6f}  fold={_fmt_ratio(r['phi_ratio'])}  share={share}  "
                  f"[channels: cov={r['phi_cov']:+.6f}, var={r['phi_var']:+.6f}]")
        if SAVE_TABLES:
            try: tabD.to_csv(OUT_MODEL_D_TABLE, index=False)
            except Exception: pass
    except Exception as e:
        print(f"  MODEL D attribution skipped: {e}")


    # 6) Permutation tests for Model A interaction, plus stratified variant if available
    if RUN_PERMUTATION_TEST:
        _print_header(f"PERMUTATION TEST (Model A interaction) — {N_PERMUTATIONS} shuffles")
        obs, pperm = perm_test_interaction(dfA, n=N_PERMUTATIONS, seed=PERM_SEED)
        print(f"Observed Δ(mean log-ratio) (Recurrent − Single-event): {obs:.6f}")
        print(f"Two-sided permutation p-value: {_fmt_p(pperm)}")
        if RUN_PERM_STRATIFIED:
            try:
                obs_s, pperm_s = perm_test_interaction_stratified(dfA, strata_col=PERM_STRATA_COL,
                                                                   n=N_PERMUTATIONS, seed=PERM_SEED)
                print(f"Stratified (by {PERM_STRATA_COL}) — observed: {obs_s:.6f}, p={_fmt_p(pperm_s)}")
            except Exception as e:
                print(f"  Stratified permutation skipped: {e}")

    # 7) McNemar within class (paired zeros)
    mcnemar_by_class(matched)

    # 8) Diagnostics & agreement (A vs B)
    print_diagnostics(matched, dfA, floor_used, resA, resB, resOverall)

    # 9) Nonzero-only sensitivity (Model A)
    if RUN_NONZERO_SENSITIVITY:
        _print_header("NONZERO-ONLY SENSITIVITY — Model A (drop any pair with π=0 on either arm)")
        resA_nz, tabA_nz, _ = run_model_A(matched, eps=floor_used, nonzero_only=True)
        for _, r in tabA_nz.iterrows():
            print(f"{r['effect']:<52}  ratio={_fmt_ratio(r['ratio'])}  CI={_fmt_ci(r['ci_low'], r['ci_high'])}  change={_fmt_pct(r['ratio'])}  p={_fmt_p(r['p'])}")

        # Also run the analogous sensitivity for Model C
        _print_header("NONZERO-ONLY SENSITIVITY — Model C (drop any pair with π=0 on either arm)")
        try:
            resC_nz, tabC_nz, _, _ = run_model_C(matched, invinfo_path=INVINFO_TSV, eps=floor_used, nonzero_only=True)
            for _, r in tabC_nz.iterrows():
                print(f"{r['effect']:<52}  ratio={_fmt_ratio(r['ratio'])}  CI={_fmt_ci(r['ci_low'], r['ci_high'])}  change={_fmt_pct(r['ratio'])}  p={_fmt_p(r['p'])}")
        except Exception as e:
            print(f"  Model C nonzero-only sensitivity skipped: {e}")

    # 10) Floor sweep (Model A across floors)
    if RUN_FLOOR_SWEEP:
        _print_header("FLOOR SENSITIVITY SWEEP — Model A across floors")
        sweep = floor_sweep(matched)
        cols = ["floor_label", "floor_value", "frac_pairs_touched",
                "SE_ratio","SE_p","RE_ratio","RE_p","INT_ratio","INT_p"]
        print(sweep[cols].to_string(index=False))
        if SAVE_TABLES:
            try: sweep.to_csv(OUT_FLOOR_SWEEP, index=False)
            except Exception: pass

    # 11) TOST equivalence (±20%) for recurrent effect — both Model A and Model C
    if RUN_TOST:
        _print_header(f"EQUIVALENCE (TOST) — recurrent inversion effect within ±{int((TOST_MARGIN_RATIO-1)*100)}%")
        # Model A
        est_RE_A, se_RE_A, p_equiv_A, deltaA = tost_equivalence_on_recurrent(resA, margin_ratio=TOST_MARGIN_RATIO)
        ratio_A = math.exp(est_RE_A)
        lo_A, hi_A = math.exp(est_RE_A - 1.96*se_RE_A), math.exp(est_RE_A + 1.96*se_RE_A)
        print(f"Model A — Recurrent effect: ratio={_fmt_ratio(ratio_A)}  CI={_fmt_ci(lo_A, hi_A)}  TOST p_equiv={_fmt_p(p_equiv_A)}  (delta={_fmt_ci(math.exp(-deltaA), math.exp(deltaA))})")

        # Model C
        try:
            est_RE_C, se_RE_C, p_equiv_C, deltaC = tost_equivalence_on_recurrent(resC, margin_ratio=TOST_MARGIN_RATIO)
            ratio_C = math.exp(est_RE_C)
            lo_C, hi_C = math.exp(est_RE_C - 1.96*se_RE_C), math.exp(est_RE_C + 1.96*se_RE_C)
            print(f"Model C — Recurrent effect (adjusted): ratio={_fmt_ratio(ratio_C)}  CI={_fmt_ci(lo_C, hi_C)}  TOST p_equiv={_fmt_p(p_equiv_C)}  (delta={_fmt_ci(math.exp(-deltaC), math.exp(deltaC))})")
            if SAVE_TABLES:
                try:
                    pd.DataFrame([{
                        "recurrent_log_est": est_RE_C, "recurrent_log_se": se_RE_C,
                        "recurrent_ratio": ratio_C, "ci_low": lo_C, "ci_high": hi_C,
                        "tost_delta_log": deltaC, "tost_margin_ratio_low": math.exp(-deltaC),
                        "tost_margin_ratio_high": math.exp(deltaC), "p_equiv": p_equiv_C
                    }]).to_csv("tost_recurrent_modelC.csv", index=False)
                except Exception:
                    pass
        except Exception as e:
            print(f"  TOST (Model C) skipped: {e}")

    # 12) Save influence tables for Model A (existing)
    try:
        X = sm.add_constant(dfA[["Recurrent"]]); X.index = dfA["region_id"]
        y = dfA["logFC"]; y.index = dfA["region_id"]
        top = cooks_distance_top(X, y, k=SHOW_TOP_INFLUENCERS)
        if SAVE_TABLES:
            top.to_csv(OUT_INFLUENCE, index=False)
            dfb = dfbetas_table(X, y, ["const","Recurrent"])
            dfb.to_csv(OUT_DFBETAS, index=False)
    except Exception:
        pass

    # 13) Influence diagnostics for Model C
    # Build the X,y that were used in Model C
    # Recover columns: const already added in fit; rebuild to align indices
    z_covs = list(used_covs)
    Xc = sm.add_constant(pd.concat([dfC[["Recurrent"]], dfC[z_covs]], axis=1))
    Xc.index = dfC["region_id"]
    yc = dfC["logFC"]; yc.index = dfC["region_id"]
    topC = cooks_distance_top(Xc, yc, k=SHOW_TOP_INFLUENCERS)
    if SAVE_TABLES:
        topC.to_csv("influence_top_modelC.csv", index=False)
        dfbC = dfbetas_table(Xc, yc, Xc.columns)
        dfbC.to_csv("influence_dfbetas_modelC.csv", index=False)
 
if __name__ == "__main__":
    main()
