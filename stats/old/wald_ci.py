import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import chi2

# ------------------ HARDCODED PATHS ------------------ #
IN_PATH  = "phewas_results.tsv"
OUT_PATH = "phewas_results.tsv"   # overwrite input after backing it up
WALD_COL = "Wald_OR_CI95"         # overwrite if it already exists

# ------------------ helpers ------------------ #

def pick_col(df, candidates):
    """Return the first existing column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def as_bool_series(raw, index):
    """Interpret common truthy strings as booleans; if column is missing, return all-False."""
    if raw is None:
        return pd.Series(False, index=index)
    s = pd.Series(raw, index=index, copy=False).astype(str).str.strip().str.upper()
    return s.isin(["TRUE", "T", "1", "YES"])

def numeric_copy(raw):
    """Numeric view of a string/object column (original left untouched)."""
    return pd.to_numeric(raw, errors="coerce")

def format_ci_pair(lo, hi):
    # Compact report-style formatting (3 significant figures): e.g., "0.869,0.927"
    return f"{lo:.3g},{hi:.3g}"

def next_backup_path(path):
    """Return a backup path like *_old.tsv; if it exists, *_old2.tsv; then *_old3.tsv, ..."""
    base, ext = os.path.splitext(path)
    candidates = [f"{base}_old{ext}", f"{base}_old2{ext}"]
    i = 3
    while os.path.exists(candidates[-1]):
        candidates.append(f"{base}_old{i}{ext}")
        i += 1
    return candidates[-1]

# ------------------ main ------------------ #

def main():
    # --- load ---
    if not os.path.isfile(IN_PATH):
        print(f"ERROR: file not found: {IN_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read as strings so existing columns are preserved verbatim on write.
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str, keep_default_na=False)
    idx = df.index

    # --- identify columns (robust to slight variants) ---
    col_beta        = pick_col(df, ["Beta", "BETA", "beta"])
    if not col_beta:
        print("ERROR: required column 'Beta' not found.", file=sys.stderr)
        sys.exit(1)

    col_p           = pick_col(df, ["P_LRT_Overall", "P_Value", "P_EMP", "P"])
    if not col_p:
        print("ERROR: no p-value column found (looked for P_LRT_Overall, P_Value, P_EMP, P).", file=sys.stderr)
        sys.exit(1)

    col_p_valid     = pick_col(df, ["P_Overall_Valid", "P_Valid", "P_Is_Valid"])
    col_inference   = pick_col(df, ["Inference_Type", "Inference_Type_x", "Inference"])
    col_used_ridge  = pick_col(df, ["Used_Ridge", "Ridge_Used"])
    col_used_firth  = pick_col(df, ["Used_Firth", "Firth_Used"])
    col_ci_valid    = pick_col(df, ["CI_Valid", "CI_Is_Valid"])
    col_p_source    = pick_col(df, ["P_Source", "P_Source_x", "P_Method", "P_Test"])

    # --- numeric/boolean views for computation ---
    beta = numeric_copy(df[col_beta])
    pval = numeric_copy(df[col_p])

    p_valid      = as_bool_series(df[col_p_valid], idx)     if col_p_valid    else pd.Series(True,  index=idx)
    used_ridge   = as_bool_series(df[col_used_ridge], idx)  if col_used_ridge else pd.Series(False, index=idx)
    used_firth   = as_bool_series(df[col_used_firth], idx)  if col_used_firth else pd.Series(False, index=idx)  # informative only
    ci_is_valid  = as_bool_series(df[col_ci_valid], idx)    if col_ci_valid   else pd.Series(False, index=idx)
    inference    = (pd.Series(df[col_inference], index=idx).astype(str).str.strip().str.lower()
                    if col_inference else pd.Series("mle", index=idx))

    # --- p-value source allowance ---
    # Allow Wald-from-p when the p-value came from MLE LRT, Firth LRT, or a score chi-square test.
    if col_p_source:
        p_source = pd.Series(df[col_p_source], index=idx).astype(str).str.strip().str.lower()
        allowed_sources = {"lrt", "lrt_mle", "lrt_firth", "score", "score_chi2", "score-chi2"}
        # also accept variants like "lrt_firth_refit" etc.
        source_allowed = p_source.apply(
            lambda s: any(s == a or s.startswith(a + "_") or s.endswith("_" + a) for a in allowed_sources)
        )
    else:
        # If no source column, be permissive.
        source_allowed = pd.Series(True, index=idx)

    # --- eligibility mask for computing Wald CI from p ---
    beta_ok  = beta.replace([np.inf, -np.inf], np.nan).notna()
    p_clean  = pval.replace([np.inf, -np.inf], np.nan)
    p_in_01  = (p_clean > 0) & (p_clean < 1)

    # Penalized MLE (ridge + labeled as MLE) -> do NOT do Wald-from-p.
    penalized_mle = used_ridge & (inference == "mle")

    # We purposely DO allow Firth (works for your HPV example with 'lrt_firth' and 'ci=profile_penalized').
    needs_wald = (
        (~ci_is_valid) &               # no valid CI present
        p_valid &                      # p-value flagged valid
        source_allowed &               # p-value came from an allowed source
        (~penalized_mle) &             # exclude penalized MLE case
        beta_ok &                      # finite beta
        p_in_01                        # proper p in (0,1)
    )

    # --- prepare/overwrite output column ---
    wald_col = pd.Series([""] * len(df), index=idx, dtype=object)

    if needs_wald.any():
        sel = needs_wald[needs_wald].index

        p_sel     = p_clean.loc[sel].clip(lower=1e-300, upper=1.0 - 1e-16).astype(float)
        beta_sel  = beta.loc[sel].astype(float)
        abs_beta  = beta_sel.abs()

        # For df=1, chi2 = z^2, so z = sqrt(chi2.isf(p, 1)).
        z_sel = pd.Series(np.sqrt(chi2.isf(p_sel.to_numpy(), df=1)), index=sel)
        z_sel = z_sel.replace([np.inf, -np.inf], np.nan)

        # SE = |beta| / z  (two-sided Wald from |Z|); guard against z -> 0
        se_sel = abs_beta / np.maximum(z_sel, 1e-12)

        lo_beta = beta_sel - 1.96 * se_sel
        hi_beta = beta_sel + 1.96 * se_sel
        lo_or   = np.exp(lo_beta)
        hi_or   = np.exp(hi_beta)

        finite_endpoints = lo_or.replace([np.inf, -np.inf], np.nan).notna() & \
                           hi_or.replace([np.inf, -np.inf], np.nan).notna()
        fill_idx = finite_endpoints[finite_endpoints].index

        wald_col.loc[fill_idx] = [
            format_ci_pair(lo, hi) for lo, hi in zip(lo_or.loc[fill_idx], hi_or.loc[fill_idx])
        ]

    # Overwrite (or create) the output column
    df[WALD_COL] = wald_col

    # --- backup & write ---
    backup_path = next_backup_path(IN_PATH)

    try:
        # Move original file out of the way, then write the new one
        os.replace(IN_PATH, backup_path)
    except Exception as e:
        print(f"ERROR moving input to backup ({backup_path}): {e}", file=sys.stderr)
        sys.exit(1)

    # Write updated file to OUT_PATH (same as original by default)
    df.to_csv(OUT_PATH, sep="\t", index=False, na_rep="")

    # --- summary ---
    n_total = len(df)
    n_added = int((df[WALD_COL] != "").sum())
    n_eligible = int(needs_wald.sum())

    print(f"Backed up original to: {backup_path}")
    print(f"Wrote updated file: {OUT_PATH}")
    print(f"Column '{WALD_COL}' overwritten/created.")
    print(f"Wald CI computed for {n_added} rows (eligible: {n_eligible} of {n_total}).")
    print("Inclusion rules: !CI_Valid & P_Valid & p in (0,1) & source in {LRT (MLE/Firth), score} & not penalized MLE.")

if __name__ == "__main__":
    main()
