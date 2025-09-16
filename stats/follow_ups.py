# ===================== GLOBALS =====================
import os
PROJECT_ID        = os.getenv("GOOGLE_PROJECT")                  # required for BQ
CDR_DATASET_ID    = os.getenv("WORKSPACE_CDR")                   # required for SQL formatting
INVERSION_FILE    = "../imputed_inversion_dosages.tsv"           # TSV with SampleID + inversion cols
OUTPUT_DIR        = "./assoc_outputs"
CACHE_DIR         = ".bq_cache"
QUANTILE_BINS     = 3

INVERSION_17      = "chr17-45585160-INV-706887"
# ===================================================

"""
This script builds PERSONAL and FAMILY history cohorts from survey data, merges them
with inversion dosages, runs association tests, and plots.

  • PERSONAL history cohorts EXCLUDE anyone who has ANY EHR data (in any OMOP EHR domain).
  • FAMILY history cohorts DO NOT exclude for EHR presence.

We also print detailed statistics per phenotype on EHR coverage, removals, and remaining counts.
"""

import hashlib, json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as _
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# --- plotting theme ---
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titleweight": "semibold",
    "axes.labelweight": "regular",
})

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")

# -------------------- SQL DEFINITIONS --------------------
# Phenotype survey definitions (unchanged)
PHENO_SQL = {
    "Breast Cancer": {
        "universe": "SELECT DISTINCT person_id FROM `{CDR}.ds_survey` WHERE question LIKE '%breast cancer%'",
        "personal_case": """
            SELECT DISTINCT person_id FROM (
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Including yourself, who in your family has had breast cancer%' AND answer LIKE '% - Self'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you still seeing a doctor or health care provider for breast cancer?%' AND answer LIKE '% - Yes'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you currently prescribed medications and/or receiving treatment for breast cancer?%' AND answer LIKE '% - Yes'
            )
        """,
        "family_case": """
            SELECT DISTINCT person_id FROM `{CDR}.ds_survey`
            WHERE question LIKE 'Have you or anyone in your family ever been diagnosed with the following cancer conditions%Select all that apply.'
              AND answer LIKE '% - Breast cancer'
        """,
    },
    "Obesity": {
        "universe": "SELECT DISTINCT person_id FROM `{CDR}.ds_survey` WHERE question LIKE '%obesity%'",
        "personal_case": """
            SELECT DISTINCT person_id FROM (
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Including yourself, who in your family has had obesity%' AND answer LIKE '% - Self'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you still seeing a doctor or health care provider for obesity?%' AND answer LIKE '% - Yes'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you currently prescribed medications and/or receiving treatment for obesity?%' AND answer LIKE '% - Yes'
            )
        """,
        "family_case": """
            SELECT DISTINCT person_id FROM `{CDR}.ds_survey`
            WHERE question LIKE 'Including yourself, who in your family has had obesity%' AND answer LIKE '% - %' AND answer NOT LIKE '% - Self'
        """,
    },
    "Heart Failure": {
        "universe": "SELECT DISTINCT person_id FROM `{CDR}.ds_survey` WHERE question LIKE '%congestive heart failure%'",
        "personal_case": """
            SELECT DISTINCT person_id FROM (
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Including yourself, who in your family has had congestive heart failure%' AND answer LIKE '% - Self'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you still seeing a doctor or health care provider for congestive heart failure?%' AND answer LIKE '% - Yes'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you currently prescribed medications and/or receiving treatment for congestive heart failure?%' AND answer LIKE '% - Yes'
            )
        """,
        "family_case": """
            SELECT DISTINCT person_id FROM `{CDR}.ds_survey`
            WHERE question LIKE 'Including yourself, who in your family has had congestive heart failure%' AND answer LIKE '% - %' AND answer NOT LIKE '% - Self'
        """,
    },
    "Cognitive Impairment": {
        "universe": """
            SELECT DISTINCT person_id FROM `{CDR}.ds_survey` WHERE 
                question LIKE '%dementia%' OR
                question LIKE '%memory loss or impairment%' OR
                question LIKE '%difficulty concentrating, remembering or making decisions%' OR
                question LIKE '%difficulty doing errands alone%' OR
                question LIKE '%difficulty dressing or bathing%'
        """,
        "personal_case": """
            SELECT DISTINCT person_id FROM (
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Including yourself, who in your family has had dementia%' AND answer LIKE '% - Self'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Including yourself, who in your family has had memory loss or impairment%' AND answer LIKE '% - Self'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you still seeing a doctor%for dementia%' AND answer LIKE '% - Yes'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you currently prescribed medications%for dementia%' AND answer LIKE '% - Yes'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you still seeing a doctor%for memory loss or impairment%' AND answer LIKE '% - Yes'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you currently prescribed medications%for memory loss or impairment%' AND answer LIKE '% - Yes'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE '%difficulty concentrating, remembering or making decisions%' AND answer NOT LIKE 'PMI: Skip'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE '%difficulty doing errands alone%' AND answer NOT LIKE 'PMI: Skip'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE '%difficulty dressing or bathing%' AND answer NOT LIKE 'PMI: Skip'
            )
        """,
        "family_case": """
            SELECT DISTINCT person_id FROM (
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Including yourself, who in your family has had dementia%' AND answer LIKE '% - %' AND answer NOT LIKE '% - Self'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Including yourself, who in your family has had memory loss or impairment%' AND answer LIKE '% - %' AND answer NOT LIKE '% - Self'
            )
        """,
    },
    "Migraine": {
        "universe": "SELECT DISTINCT person_id FROM `{CDR}.ds_survey` WHERE question LIKE '%migraine headaches%'",
        "personal_case": """
            SELECT DISTINCT person_id FROM (
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Including yourself, who in your family has had migraine headaches%' AND answer LIKE '% - Self'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you still seeing a doctor or health care provider for migraine headaches?%' AND answer LIKE '% - Yes'
                UNION DISTINCT
                SELECT person_id FROM `{CDR}.ds_survey` WHERE question LIKE 'Are you currently prescribed medications and/or receiving treatment for migraine headaches?%' AND answer LIKE '% - Yes'
            )
        """,
        "family_case": """
            SELECT DISTINCT person_id FROM `{CDR}.ds_survey`
            WHERE question LIKE 'Including yourself, who in your family has had migraine headaches%' AND answer LIKE '% - %' AND answer NOT LIKE '% - Self'
        """,
    },
}

# -------------------- CACHE + BIGQUERY --------------------
def _cache_key(sql: str) -> str:
    payload = json.dumps({"sql": sql, "cdr": CDR_DATASET_ID}, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()  # non-crypto; fine for caching

def _cache_path(key: str) -> Path:
    return Path(CACHE_DIR) / f"{key}.parquet"

def execute_query(sql: str, desc: str) -> pd.DataFrame:
    from pandas import read_gbq  # assume installed
    key = _cache_key(sql)
    path = _cache_path(key)
    if path.exists():
        print(f"INFO | {desc:16s} | cache -> {path.name}")
        df = pd.read_parquet(path)
    else:
        print(f"INFO | {desc:16s} | BigQuery")
        df = read_gbq(sql, project_id=PROJECT_ID, progress_bar_type=None)
        df.to_parquet(path, index=False)
    n = len(df)
    uniq = df["person_id"].nunique() if "person_id" in df.columns else None
    msg = f"DEBUG| {desc:16s} | rows={n:,}" + (f" unique_person_id={uniq:,}" if uniq is not None else "")
    print(msg)
    return df

# -------------------- EHR PRESENCE --------------------
def get_all_ehr_person_ids(cdr: str) -> pd.DataFrame:
    """
    Returns DISTINCT person_id for participants with any EHR visits.
    We anchor EHR presence on visit_occurrence only (excludes survey-only participants).
    """
    sql = f"""
    SELECT DISTINCT person_id
    FROM `{cdr}.visit_occurrence`
    """
    df = execute_query(sql, "EHR from visits")
    # normalize dtype
    if "person_id" in df.columns:
        df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["person_id"]).astype({"person_id": "int64"})
    return df[["person_id"]].drop_duplicates()

# -------------------- PHENOTYPE COHORTS --------------------
def _print_exclusion_stats(label: str, u0: pd.DataFrame, c0: pd.DataFrame, u: pd.DataFrame, c: pd.DataFrame,
                           ehr_df: pd.DataFrame | None):
    n_u0 = len(u0); n_c0 = len(c0); n_ctrl0 = n_u0 - n_c0
    n_u  = len(u);  n_c  = len(c);  n_ctrl  = n_u  - n_c
    rem_u = n_u0 - n_u; rem_c = n_c0 - n_c; rem_ctrl = n_ctrl0 - n_ctrl
    print(f"INFO | Exclusion ({label}) | universe_removed={rem_u:,} cases_removed={rem_c:,} controls_removed={rem_ctrl:,} remaining_universe={n_u:,} remaining_cases={n_c:,} remaining_controls={n_ctrl:,}")

    if ehr_df is not None and not ehr_df.empty:
        u0_in_ehr = int(u0["person_id"].isin(ehr_df["person_id"]).sum())
        u0_not_ehr = n_u0 - u0_in_ehr
        print(f"INFO | EHR coverage      | universe_with_EHR={u0_in_ehr:,} universe_without_EHR={u0_not_ehr:,}")

def get_phenotypes(universe_sql: str, case_sql: str,
                   exclude_ids: pd.DataFrame | None = None,
                   exclusion_label: str | None = None,
                   ehr_df_for_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build a phenotype cohort (cases/controls) from survey-based universe + case SQL.
    If exclude_ids is provided, remove those person_ids from both universe and cases (used for PERSONAL history).
    """
    u0 = execute_query(universe_sql, "Universe")
    c0 = execute_query(case_sql,     "Cases")

    # Standardize to int64 for stable merges
    for df in (u0, c0):
        if "person_id" in df.columns:
            df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").astype("Int64")
            df.dropna(subset=["person_id"], inplace=True)
            df["person_id"] = df["person_id"].astype("int64")

    if exclude_ids is not None and not exclude_ids.empty:
        # Exclude from both universe and cases
        ex = exclude_ids["person_id"].astype("int64")
        u = u0[~u0["person_id"].isin(ex)].copy()
        c = c0[~c0["person_id"].isin(ex)].copy()
        _print_exclusion_stats(exclusion_label or "Excluded", u0, c0, u, c, ehr_df_for_stats)
    else:
        u, c = u0, c0
        # Still print EHR coverage for transparency
        _print_exclusion_stats("No exclusion", u0, c0, u, c, ehr_df_for_stats)

    cases = c.assign(status=1) if not c.empty else pd.DataFrame(columns=["person_id","status"])
    ctrls = u[~u.person_id.isin(c.person_id)].assign(status=0) if not c.empty else u.assign(status=0)
    pheno = pd.concat([cases, ctrls], ignore_index=True)[["person_id","status"]]
    print(f"INFO | Cohort            | universe={len(u):,} cases={pheno['status'].sum():,} controls={(len(pheno)-pheno['status'].sum()):,}")
    return pheno

# -------------------- TESTS (no covariates) --------------------
def _fit_logit(y: pd.Series, x: pd.Series):
    # Coerce + drop rows with NaNs in either vector
    x = pd.to_numeric(x, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(int)
    mask = ~(x.isna() | y.isna())
    x, y = x[mask], y[mask]

    # Degeneracy checks: constant predictor or single-class outcome → skip logistic
    if pd.Series(x).nunique(dropna=True) < 2 or pd.Series(y).nunique(dropna=True) < 2:
        return None

    # Add intercept and fit with a robust optimizer; guard against numeric failures
    X = sm.add_constant(x, has_constant="add")
    try:
        return sm.Logit(y, X).fit(disp=0, method="lbfgs", maxiter=200)
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, RuntimeError):
        return None

def _lrt_p(llf_alt: float, llf_null: float, df_diff: int = 1) -> float:
    from scipy.stats import chi2
    return float(chi2.sf(2*(llf_alt-llf_null), df_diff))

def cochran_armitage(y: pd.Series, g: pd.Series):
    from scipy.stats import norm
    t = (pd.DataFrame({"y":y.astype(int),"g":g.round().clip(0,2).astype(int)})
           .groupby("g").agg(cases=("y","sum"), n=("y","size")).reindex([0,1,2], fill_value=0))
    scores = np.array([0.,1.,2.])
    n = float(t["n"].sum()); p = float(t["cases"].sum()/max(n,1))
    w = t["n"].to_numpy(); sbar = np.average(scores, weights=w)
    num = float(np.sum(scores*t["cases"].to_numpy()) - p*np.sum(scores*w))
    den = math.sqrt(max(p*(1-p)*np.sum(w*(scores-sbar)**2), 1e-12))
    z = num/den if den>0 else 0.0
    return z, float(2*norm.sf(abs(z)))

def run_tests(df: pd.DataFrame, inv: str) -> dict:
    y = df["status"].astype(int)
    x = df[inv].astype(float)
    out = {"wald_p": np.nan, "lrt_p": np.nan, "or": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "pseudo_r2": np.nan, "trend_z": np.nan, "trend_p": np.nan}

    # Try logistic; if it fails (or predictor is constant) _fit_logit returns None
    m = _fit_logit(y, x)
    if m is not None:
        out["wald_p"]   = float(m.pvalues.get(inv, np.nan))
        out["or"]       = float(np.exp(m.params.get(inv, np.nan)))
        if inv in m.params.index:
            ci = m.conf_int().loc[inv]
            out["ci_low"], out["ci_high"] = float(np.exp(ci[0])), float(np.exp(ci[1]))
        out["pseudo_r2"] = float(1 - (m.llf / m.llnull)) if m.llnull != 0 else np.nan
        out["lrt_p"]     = float(_lrt_p(m.llf, m.llnull, 1))
    else:
        if pd.Series(x).dropna().nunique() < 2:
            print(f"WARN | {inv}: predictor constant → logistic skipped; reporting trend only.")
        else:
            print(f"WARN | {inv}: logistic failed numerically; reporting trend only.")

    # Always provide the Cochran–Armitage trend test as a stable fallback
    z, p_trend = cochran_armitage(y, x)
    out["trend_z"], out["trend_p"] = float(z), float(p_trend)
    return out

# -------------------- PLOTS (Relative Risk) --------------------
def _relative_risk(series_status: pd.Series, by: pd.Series) -> pd.DataFrame:
    base = float(series_status.mean())  # cohort prevalence as baseline
    g = (pd.DataFrame({"bin": by, "status": series_status.astype(int)})
         .groupby("bin", observed=True, dropna=False)
         .agg(n=("status","size"), risk=("status","mean"))
         .reset_index())

    g["rr"] = g["risk"] / max(base, 1e-12)
    # simple SE for p (binomial), propagate to RR (denom treated as constant)
    g["se_rr"] = np.sqrt(g["risk"]*(1-g["risk"])/g["n"].clip(lower=1)) / max(base, 1e-12)
    g["rr_low"] = (g["rr"] - 1.96*g["se_rr"]).clip(lower=0)
    g["rr_high"] = g["rr"] + 1.96*g["se_rr"]
    g["base_risk"] = base
    return g

def plot_quantiles(df: pd.DataFrame, inv: str, phenotype: str, test_type: str):
    series = df[inv].astype(float)
    q = pd.qcut(series, q=QUANTILE_BINS, duplicates="drop")
    # midpoints as pure float series to avoid categorical reductions
    mid = pd.Series(q).map(lambda b: (b.left + b.right) / 2).astype(float)
    g = _relative_risk(df["status"], q)
    # per-bin midpoint (mean is safe since 'mid' is float)
    mids = (
        pd.DataFrame({"bin": q, "mid": mid})
          .groupby("bin", observed=True, dropna=False)
          .agg(mid=("mid", "mean"))
          .reset_index()
    )
    g = g.merge(mids, on="bin", how="left").sort_values("mid")

    fig = plt.figure(figsize=(7.5, 4.6))
    plt.plot(g["mid"], g["rr"], marker="o", linewidth=2.2)
    plt.fill_between(g["mid"], g["rr_low"], g["rr_high"], alpha=0.15, linewidth=0)
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel(f"{inv} dosage (quantile centers)")
    plt.ylabel("Relative risk vs cohort prevalence")
    plt.title(f"{phenotype} — {test_type}: RR by {QUANTILE_BINS}-quantile of {inv}")
    plt.tight_layout()
    out = Path(OUTPUT_DIR)/f"rr_quantiles_{phenotype.replace(' ','_')}_{test_type.replace(' ','_')}_{inv}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return str(out)

def plot_rounded_bins(df: pd.DataFrame, inv: str, phenotype: str, test_type: str):
    b = df[inv].astype(float).round().clip(0,2)
    g = _relative_risk(df["status"], b).sort_values("bin")
    fig = plt.figure(figsize=(6.8, 4.2))
    plt.bar(g["bin"].astype(str), g["rr"])
    for i, (rr, n) in enumerate(zip(g["rr"], g["n"])):
        plt.text(i, rr, f"{rr:.2f}\n(n={n})", ha="center", va="bottom")
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Rounded dosage bin {0,1,2}")
    plt.ylabel("Relative risk vs cohort prevalence")
    plt.title(f"{phenotype} — {test_type}: RR by rounded bins of {inv}")
    plt.tight_layout()
    out = Path(OUTPUT_DIR)/f"rr_bins_{phenotype.replace(' ','_')}_{test_type.replace(' ','_')}_{inv}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return str(out)

# -------------------- CORR DIAGNOSTICS --------------------
def corr_chr17_vs_all(dosage: pd.DataFrame):
    num_cols = dosage.select_dtypes(include=[np.number]).columns
    s = dosage[num_cols].corr(method="pearson")[INVERSION_17].sort_values(ascending=False)
    out = Path(OUTPUT_DIR)/"chr17_correlations.csv"; s.to_csv(out, header=["pearson_corr_with_chr17"])
    print("INFO | Top correlations with chr17:")
    for k, v in s.head(10).round(4).items():
        print(f"      {k:>36s}  {v:+.4f}")
    return str(out)

# -------------------- ORCHESTRATION --------------------
def run_for_phenotype(name: str, sqls: dict, dosage: pd.DataFrame, inversions: list,
                      ehr_people: pd.DataFrame):
    print("\n" + "#"*80)
    print(f"INFO | Phenotype: {name}")
    print("#"*80)

    # PERSONAL: exclude EHR
    ph = get_phenotypes(
        sqls["universe"],
        sqls["personal_case"],
        exclude_ids=ehr_people,
        exclusion_label="EHR present → EXCLUDED (Personal)",
        ehr_df_for_stats=ehr_people
    )
    # FAMILY: no exclusion
    fh = get_phenotypes(
        sqls["universe"],
        sqls["family_case"],
        exclude_ids=None,
        exclusion_label=None,
        ehr_df_for_stats=ehr_people
    )

    # Merge with dosages
    dfp = ph.merge(dosage, on="person_id", how="inner")
    dff = fh.merge(dosage, on="person_id", how="inner")
    print(f"INFO | Merge (Personal) | rows={len(dfp):,} cases={int(dfp['status'].sum()):,}")
    print(f"INFO | Merge (Family)   | rows={len(dff):,} cases={int(dff['status'].sum()):,}")

    rows = []
    for label, d in [("Personal History", dfp), ("Family History", dff)]:
        for inv in inversions:
            if inv not in d.columns:
                print(f"WARN | {inv} not in data; skip."); continue
            sub = d[["status", inv]].dropna()
            if sub.empty:
                print(f"WARN | {label}: no non-null rows after dropna for {inv}; skip.")
                continue
            desc = sub[inv].describe(percentiles=[.01,.1,.5,.9,.99]).round(4)
            bins = sub[inv].round().clip(0,2).value_counts().reindex([0,1,2], fill_value=0).astype(int).tolist()
            print(f"DEBUG| {label:16s} | {inv} nonnull={len(sub):,} min={desc['min']:.3f} p50={desc['50%']:.3f} max={desc['max']:.3f} bins(0/1/2)={bins}")

            # Skip logistic if predictor or outcome has no variation; fall back to trend only
            if sub[inv].dropna().nunique() < 2 or sub["status"].dropna().nunique() < 2:
                print(f"WARN | {label}: {inv} or outcome has no variation → skipping logistic; trend only.")
                z, p_trend = cochran_armitage(sub["status"], sub[inv])
                stats = {
                    "wald_p": np.nan, "lrt_p": np.nan, "or": np.nan,
                    "ci_low": np.nan, "ci_high": np.nan, "pseudo_r2": np.nan,
                    "trend_z": float(z), "trend_p": float(p_trend),
                }
            else:
                stats = run_tests(sub, inv)

            qplot = plot_quantiles(d, inv, name, label)
            bplot = plot_rounded_bins(d, inv, name, label)

            rows.append({
                "phenotype": name, "test_type": label, "inversion": inv,
                "n": int(len(d)), "n_cases": int(d["status"].sum()),
                "n_controls": int(len(d)-d["status"].sum()),
                **{k: (None if (isinstance(v,float) and np.isnan(v)) else v) for k,v in stats.items()},
                "quantile_plot": qplot, "rounded_bin_plot": bplot,
            })
    return rows

def main():
    if not PROJECT_ID or not CDR_DATASET_ID:
        raise RuntimeError("GOOGLE_PROJECT and WORKSPACE_CDR environment variables are required.")

    # Fill CDR in SQL
    sqls = {k: {kk: vv.format(CDR=CDR_DATASET_ID) for kk, vv in v.items()} for k, v in PHENO_SQL.items()}

    # Load dosages
    print(f"INFO | Loading dosages: {INVERSION_FILE}")
    d = pd.read_csv(INVERSION_FILE, sep="\t").rename(columns={"SampleID":"person_id"})
    d["person_id"] = pd.to_numeric(d["person_id"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["person_id"]).astype({"person_id":"int64"})
    inv_cols = [c for c in d.columns if c.startswith("chr") and "INV" in c]
    print(f"INFO | Dosage shape={d.shape} inversions={len(inv_cols)} sample={inv_cols[:4]}")

    # chr17 for all
    inv_map = {name: [INVERSION_17] for name in sqls}

    # NEW: build EHR presence set once
    ehr_people = get_all_ehr_person_ids(CDR_DATASET_ID)
    n_ehr = len(ehr_people)
    n_dosage_overlap_with_ehr = int(d["person_id"].isin(ehr_people["person_id"]).sum())
    print(f"INFO | EHR global        | persons_with_any_EHR={n_ehr:,}")
    print(f"INFO | EHR∩Dosage        | dosage_persons_with_EHR={n_dosage_overlap_with_ehr:,} out_of_dosage_n={len(d):,}")

    # Run per phenotype
    all_rows = []
    for name in sqls:
        all_rows += run_for_phenotype(name, sqls[name], d, inv_map[name], ehr_people)

    # Report table
    rep = pd.DataFrame(all_rows).sort_values(["phenotype","test_type","inversion"])
    out_csv = Path(OUTPUT_DIR)/"association_results.csv"; rep.to_csv(out_csv, index=False)

    print("\n" + "="*100)
    if not rep.empty:
        view = rep[["phenotype","test_type","inversion","n","n_cases","n_controls","or","ci_low","ci_high","wald_p","lrt_p","trend_p","pseudo_r2"]]
        print("INFO | Final results (OR per dosage unit):")
        print(view.to_string(index=False))
    else:
        print("INFO | No results produced.")
    print("="*100)

    # chr17 correlations
    corr_path = corr_chr17_vs_all(d)
    print(f"INFO | chr17 correlation CSV -> {corr_path}")
    print(f"INFO | Outputs -> {Path(OUTPUT_DIR).resolve()}")

if __name__ == "__main__":
    main()
