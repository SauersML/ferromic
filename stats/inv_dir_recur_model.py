import sys
import math
import warnings
from typing import Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.contingency_tables import mcnemar


from scipy.stats import wilcoxon

# ------------------------- FILE PATHS -------------------------

OUTPUT_CSV  = "./output.csv"
INVINFO_CSV = "./inv_info.csv"

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

def load_and_match(output_csv: str, invinfo_csv: str) -> pd.DataFrame:
    """
    STRICT loader:
      - Requires inv_info.csv has columns: Chromosome, Start, End, 0_single_1_recur_consensus
      - Crashes if inv_info has duplicate keys (chr_std, Start, End)
      - Builds 9 candidate (Start,End) per region with ±1 bp tolerance
      - Keeps only true matches; for each region, picks the minimal match_priority
        and requires exactly ONE inv row at that best priority.
      - Returns matched table with both π values present and finite.
    """
    df  = pd.read_csv(output_csv)
    inv = pd.read_csv(invinfo_csv)

    # enforce required columns
    need_df = ["chr", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"]
    miss_df = [c for c in need_df if c not in df.columns]
    if miss_df:
        raise KeyError(f"{output_csv} missing columns: {miss_df}")

    need_inv = ["Chromosome", "Start", "End", "0_single_1_recur_consensus"]
    miss_inv = [c for c in need_inv if c not in inv.columns]
    if miss_inv:
        raise KeyError(f"{invinfo_csv} missing columns: {miss_inv}")

    # standardize chromosomes
    df["chr_std"]  = df["chr"].apply(_standardize_chr)
    inv["chr_std"] = inv["Chromosome"].apply(_standardize_chr)

    # check duplicates in inv_info keys → CRASH if any
    dup_keys = inv.duplicated(subset=["chr_std", "Start", "End"], keep=False)
    if dup_keys.any():
        bad = inv.loc[dup_keys, ["chr_std", "Start", "End"]].drop_duplicates()
        raise ValueError(f"inv_info.csv contains duplicate (chr,Start,End) keys. Offending keys:\n{bad.to_string(index=False)}")

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
    else:
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
    matched = load_and_match(OUTPUT_CSV, INVINFO_CSV)
    n_single = (matched['Recurrence']=='Single-event').sum()
    n_recur  = (matched['Recurrence']=='Recurrent').sum()
    print(f"Matched paired regions (STRICT): {matched.shape[0]}  (Single-event: {n_single}, Recurrent: {n_recur})")

    # 2) Choose epsilon (quantile-only, no fallback)
    all_pi = np.r_[matched["pi_direct"].to_numpy(float), matched["pi_inverted"].to_numpy(float)]
    floor_used = choose_floor_from_quantile(all_pi, q=FLOOR_QUANTILE, min_floor=MIN_FLOOR)

    # 3) Model A (primary)
    _print_header("MODEL A — Δ-logπ ~ Recurrence (HC3)")
    resA, tabA, dfA = run_model_A(matched, eps=floor_used, nonzero_only=False)
    for _, r in tabA.iterrows():
        print(f"{r['effect']:<44}  ratio={_fmt_ratio(r['ratio'])}  CI={_fmt_ci(r['ci_low'], r['ci_high'])}  "
              f"change={_fmt_pct(r['ratio'])}  p={_fmt_p(r['p'])}")
    if SAVE_TABLES:
        tabA.to_csv(OUT_MODEL_A_TABLE, index=False)

    # 4) Model B (confirmatory, FE + cluster by region)
    _print_header("MODEL B — logπ ~ Inverted + Inverted:Recurrence + C(region_id)  (cluster-robust by region)")
    resB, tabB, longB, resOverall = run_model_B(matched, eps=floor_used)
    for _, r in tabB.iterrows():
        print(f"{r['effect']:<44}  ratio={_fmt_ratio(r['ratio'])}  CI={_fmt_ci(r['ci_low'], r['ci_high'])}  "
              f"change={_fmt_pct(r['ratio'])}  p={_fmt_p(r['p'])}")
    if SAVE_TABLES:
        tabB.to_csv(OUT_MODEL_B_TABLE, index=False)

    # 5) Permutation interaction test(s)
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

    # 6) McNemar within class (paired zeros)
    mcnemar_by_class(matched)

    # 7) Diagnostics & agreement
    print_diagnostics(matched, dfA, floor_used, resA, resB, resOverall)

    # 8) Nonzero-only sensitivity
    if RUN_NONZERO_SENSITIVITY:
        _print_header("NONZERO-ONLY SENSITIVITY — drop any pair with π=0 on either arm")
        resA_nz, tabA_nz, _ = run_model_A(matched, eps=floor_used, nonzero_only=True)
        for _, r in tabA_nz.iterrows():
            print(f"{r['effect']:<44}  ratio={_fmt_ratio(r['ratio'])}  CI={_fmt_ci(r['ci_low'], r['ci_high'])}  "
                  f"change={_fmt_pct(r['ratio'])}  p={_fmt_p(r['p'])}")

    # 9) Floor sweep (quantiles + big epsilon), and save
    if RUN_FLOOR_SWEEP:
        _print_header("FLOOR SENSITIVITY SWEEP — Model A across floors")
        sweep = floor_sweep(matched)
        cols = ["floor_label", "floor_value", "frac_pairs_touched",
                "SE_ratio","SE_p","RE_ratio","RE_p","INT_ratio","INT_p"]
        print(sweep[cols].to_string(index=False))
        if SAVE_TABLES:
            sweep.to_csv(OUT_FLOOR_SWEEP, index=False)

    # 10) TOST (equivalence) for recurrent effect
    if RUN_TOST:
        _print_header(f"EQUIVALENCE (TOST) — recurrent inversion effect within ±{int((TOST_MARGIN_RATIO-1)*100)}%")
        est_RE, se_RE, p_equiv, delta = tost_equivalence_on_recurrent(resA, margin_ratio=TOST_MARGIN_RATIO)
        ratio = math.exp(est_RE)
        lo, hi = math.exp(est_RE - 1.96*se_RE), math.exp(est_RE + 1.96*se_RE)
        print(f"Recurrent effect: ratio={_fmt_ratio(ratio)}  CI={_fmt_ci(lo, hi)}  "
              f"TOST p_equiv={_fmt_p(p_equiv)}  (delta={_fmt_ci(math.exp(-delta), math.exp(delta))})")
        if SAVE_TABLES:
            pd.DataFrame([{
                "recurrent_log_est": est_RE, "recurrent_log_se": se_RE,
                "recurrent_ratio": ratio, "ci_low": lo, "ci_high": hi,
                "tost_delta_log": delta, "tost_margin_ratio_low": math.exp(-delta),
                "tost_margin_ratio_high": math.exp(delta), "p_equiv": p_equiv
            }]).to_csv(OUT_TOST, index=False)

    # 11) Save influence table for Model A
    try:
        X = sm.add_constant(dfA[["Recurrent"]]); X.index = dfA["region_id"]
        y = dfA["logFC"]; y.index = dfA["region_id"]
        top = cooks_distance_top(X, y, k=SHOW_TOP_INFLUENCERS)
        if SAVE_TABLES:
            top.to_csv(OUT_INFLUENCE, index=False)
    except Exception:
        pass

if __name__ == "__main__":
    main()
