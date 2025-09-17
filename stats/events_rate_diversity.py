from typing import Tuple, Optional, Dict
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from scipy.stats import (
    spearmanr,
    kendalltau,
    permutation_test,
    bootstrap,
    theilslopes,
)

# ------------------------- FILE PATHS -------------------------

OUTPUT_CSV  = "./output.csv"
INVINFO_TSV = "./inv_info.tsv"

# ------------------------- OUTPUT ARTIFACTS -------------------

OUT_RESULTS_CSV                 = "explore_results.csv"
OUT_LOGFC_VS_FORMRATE_FIG       = "logfc_vs_formation_rate.pdf"
OUT_LOGFC_VS_NRECUR_FIG         = "logfc_vs_nrecur.pdf"
OUT_FST_VS_FORMRATE_FIG         = "fst_vs_formation_rate.pdf"
OUT_FST_VS_NRECUR_FIG           = "fst_vs_nrecur.pdf"

# ------------------------- SETTINGS --------------------------

FLOOR_QUANTILE = 0.01
MIN_FLOOR      = 1e-8

N_PERMUTATIONS = 10000
PERM_SEED      = 2025

N_BOOTSTRAP    = 2000
BOOT_SEED      = 2026
CONF_LEVEL     = 0.95  # for bootstrap CIs

LOWESS_FRAC    = 0.6   # smoothing span for FST plots
JITTER_X       = 0.10  # jitter amplitude for integer x in count+FST plots

# ------------------------- GLOBAL PLOT STYLE ------------------

COLOR_SE       = "#0b3d91"  # dark blue
COLOR_RE       = "#9b2c9b"  # reddish purple

# ------------------------- HELPERS ---------------------------

def _standardize_chr(val: str) -> str:
    s = str(val).strip()
    return s[3:] if s.lower().startswith("chr") else s

def choose_floor_from_quantile(pi_all: np.ndarray, q: float, min_floor: float) -> float:
    pos = pi_all[np.isfinite(pi_all) & (pi_all > 0)]
    assert pos.size > 0, "All π values are non-positive; cannot choose ε from a positive quantile."
    return max(float(np.quantile(pos, q)) * 0.5, min_floor)

_CI_LEAD_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)(?:\s|\[|$)", re.IGNORECASE)

def parse_ci_lead_number(val) -> float:
    """
    Extract the leading numeric point estimate from strings like
      '1.02e-04 [2.72e-05 ,1.21e-04]'.
    - Returns np.nan for 'NA'/blank/non-parsable.
    """
    if val is None:
        return np.nan
    s = str(val).strip()
    if s == "" or s.upper() == "NA":
        return np.nan
    m = _CI_LEAD_RE.match(s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _lowess_xy(x: np.ndarray, y: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted x and corresponding LOWESS-smoothed y (for FST panels)."""
    mask = np.isfinite(x) & np.isfinite(y)
    assert mask.sum() >= 2, "LOWESS requires at least 2 finite points."
    x1, y1 = x[mask], y[mask]
    order = np.argsort(x1)
    x1, y1 = x1[order], y1[order]
    smth = lowess(y1, x1, frac=frac, return_sorted=True)
    xs = smth[:, 0]
    ys = smth[:, 1]
    return xs, ys

def _fmt_p(p: float) -> str:
    if not np.isfinite(p): return "NA"
    if p < 1e-99: return "<1e-99"
    if p < 1e-3:  return f"{p:.1e}"
    return f"{p:.3f}"

# --------- Stats wrappers (SciPy / statsmodels) ---------

def spearman_perm_p(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided permutation p-value for Spearman correlation (permute pairings)."""
    def stat(a, b):
        return spearmanr(a, b).correlation
    res = permutation_test(
        (x, y),
        stat,
        permutation_type="pairings",
        n_resamples=n_perm,
        alternative="two-sided",
        random_state=seed,
        vectorized=False
    )
    return float(res.pvalue)

def kendall_perm_p(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided permutation p-value for Kendall tau (permute pairings)."""
    def stat(a, b):
        return kendalltau(a, b).statistic
    res = permutation_test(
        (x, y),
        stat,
        permutation_type="pairings",
        n_resamples=n_perm,
        alternative="two-sided",
        random_state=seed,
        vectorized=False
    )
    return float(res.pvalue)

def spearman_boot_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, conf_level: float) -> Tuple[float, float]:
    """Bootstrap BCa CI for Spearman ρ using scipy.stats.bootstrap (paired resampling)."""
    def stat(a, b, axis=0):
        return spearmanr(a, b).correlation
    res = bootstrap(
        data=(x, y),
        statistic=stat,
        vectorized=False,
        paired=True,
        n_resamples=n_boot,
        confidence_level=conf_level,
        method="BCa",
        random_state=seed
    )
    ci = res.confidence_interval
    return float(ci.low), float(ci.high)

def ols_hc3_slope(y: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """OLS slope with HC3 robust SEs: y ~ const + x. Returns dict with slope, CI, p, R²."""
    assert y.shape[0] == x.shape[0] and y.ndim == 1 and x.ndim == 1
    X = pd.DataFrame({"const": np.ones_like(x, dtype=float), "x": x.astype(float)})
    res = sm.OLS(y.astype(float), X.values).fit(cov_type="HC3")
    beta = float(res.params[1])
    ci_arr = np.asarray(res.conf_int(alpha=0.05))
    ci_low = float(ci_arr[1, 0])
    ci_high = float(ci_arr[1, 1])
    p = float(res.pvalues[1])
    r2 = float(res.rsquared)
    return dict(beta=beta, ci_low=ci_low, ci_high=ci_high, p=p, r2=r2)

# ------------------------- STRICT MATCHING -------------------

def load_and_match(output_csv: str, invinfo_tsv: str) -> pd.DataFrame:
    """
    STRICT loader + matcher:
      - output.csv must contain: 'chr','region_start','region_end',
                                 '0_pi_filtered','1_pi_filtered',
                                 'hudson_fst_hap_group_0v1'
      - inv_info.tsv must contain: 'Chromosome','Start','End','0_single_1_recur_consensus',
                                   'Number_recurrent_events','Formation_rate_per_generation_.95..C.I..'
      - Build ±1 bp candidate (Start,End) and select a UNIQUE inv row per region at minimal priority.
      - DROP rows with non-finite π in either arm (remove NaN and ±Inf).
      - Keep only rows with 0/1 in consensus → {'Single-event','Recurrent'}.
      - Extract formation_rate point estimate (may be NaN) and Number_recurrent_events (may be NaN).
    """
    df  = pd.read_csv(output_csv)
    inv = pd.read_csv(invinfo_tsv, sep="\t")

    need_df  = ["chr", "region_start", "region_end",
                "0_pi_filtered", "1_pi_filtered",
                "hudson_fst_hap_group_0v1"]
    need_inv = ["Chromosome", "Start", "End",
                "0_single_1_recur_consensus",
                "Number_recurrent_events",
                "Formation_rate_per_generation_.95..C.I.."]
    for c in need_df:
        assert c in df.columns, f"{output_csv} missing column: {c}"
    for c in need_inv:
        assert c in inv.columns, f"{invinfo_tsv} missing column: {c}"

    df["chr_std"]  = df["chr"].apply(_standardize_chr)
    inv["chr_std"] = inv["Chromosome"].apply(_standardize_chr)

    # Check duplicates in inv keys
    dup_keys = inv.duplicated(subset=["chr_std", "Start", "End"], keep=False)
    assert not dup_keys.any(), "inv_info.tsv contains duplicate (chr,Start,End) keys."

    # Compact df
    df_small = df[["chr_std", "region_start", "region_end",
                   "0_pi_filtered", "1_pi_filtered",
                   "hudson_fst_hap_group_0v1"]].rename(
        columns={"0_pi_filtered": "pi_direct",
                 "1_pi_filtered": "pi_inverted",
                 "hudson_fst_hap_group_0v1": "fst_hudson"}
    ).copy()
    df_small["region_start"] = df_small["region_start"].astype(int)
    df_small["region_end"]   = df_small["region_end"].astype(int)

    # Build ±1 bp candidate keys (9 per region)
    cands = []
    for ds in (-1, 0, 1):
        for de in (-1, 0, 1):
            tmp = df_small.copy()
            tmp["Start"] = tmp["region_start"] + ds
            tmp["End"]   = tmp["region_end"]   + de
            tmp["match_priority"] = abs(ds) + abs(de)  # 0 (exact), 1, or 2
            cands.append(tmp)
    df_cand = pd.concat(cands, ignore_index=True)

    inv_small = inv[["chr_std", "Start", "End",
                     "0_single_1_recur_consensus",
                     "Number_recurrent_events",
                     "Formation_rate_per_generation_.95..C.I.."]].copy()
    merged = df_cand.merge(inv_small, on=["chr_std", "Start", "End"], how="inner")
    assert not merged.empty, "No regions matched inv_info under ±1 bp tolerance."

    # Select one inv row per region at minimal priority with uniqueness
    key = ["chr_std", "region_start", "region_end"]
    rows = []
    for _, g in merged.groupby(key, sort=False):
        mp = int(g["match_priority"].min())
        gg = g[g["match_priority"] == mp].drop_duplicates(subset=["Start", "End"]).copy()
        assert gg.shape[0] == 1, (
            f"Ambiguous inv mapping at best priority for region "
            f"{g.iloc[0]['chr_std']}:{int(g.iloc[0]['region_start'])}-{int(g.iloc[0]['region_end'])}"
        )
        rows.append(gg.iloc[0])
    best = pd.DataFrame(rows)

    # Recurrence label: only keep 0/1
    rec_num = pd.to_numeric(best["0_single_1_recur_consensus"], errors="coerce")
    mask_rec = rec_num.isin([0, 1])
    best = best.loc[mask_rec].copy()
    assert best.shape[0] > 0, "No rows with valid 0/1 in 0_single_1_recur_consensus."
    best["Recurrence"] = rec_num.loc[mask_rec].map({0: "Single-event", 1: "Recurrent"}).to_numpy()

    # π and F_ST numeric conversions
    best["pi_direct"]   = pd.to_numeric(best["pi_direct"],   errors="coerce")
    best["pi_inverted"] = pd.to_numeric(best["pi_inverted"], errors="coerce")
    best["fst_hudson"]  = pd.to_numeric(best["fst_hudson"],  errors="coerce")

    # DROP rows with non-finite π in either arm (remove NaN, +Inf, -Inf)
    mask_pi = np.isfinite(best["pi_direct"].to_numpy(float)) & np.isfinite(best["pi_inverted"].to_numpy(float))
    best = best.loc[mask_pi].copy()
    assert best.shape[0] > 0, "After dropping non-finite π rows, no regions remain."

    # Extract formation rate (point estimate only); allow NaN here
    best["formation_rate"] = best["Formation_rate_per_generation_.95..C.I.."].apply(parse_ci_lead_number)

    # Parse number of recurrent events; allow NaN here
    best["n_recur_events"] = pd.to_numeric(best["Number_recurrent_events"], errors="coerce")

    # region_id
    best["region_id"] = (
        best["chr_std"].astype(str) + ":" +
        best["region_start"].astype(int).astype(str) + "-" +
        best["region_end"].astype(int).astype(str)
    )

    cols = ["region_id", "chr_std", "region_start", "region_end",
            "Recurrence", "pi_direct", "pi_inverted", "fst_hudson",
            "n_recur_events", "formation_rate", "Start", "End"]
    return best[cols].copy()

# ------------------------- DATA PREP -------------------------

def build_logfc_table(matched: pd.DataFrame, eps: float) -> pd.DataFrame:
    """
    Create one row per region with:
      region_id, Recurrence, logFC, x_formrate = log10(formation_rate), x_nrecur = ln(1 + n_recur_events),
      plus raw predictors for reference.
    Rows with formation_rate <= 0 will have x_formrate = NaN.
    """
    t = matched.copy()
    t["logFC"] = np.log(t["pi_inverted"].to_numpy(float) + float(eps)) \
               - np.log(t["pi_direct"  ].to_numpy(float) + float(eps))

    fr = t["formation_rate"].to_numpy(float)
    nr = t["n_recur_events"].to_numpy(float)

    x_form = np.where(np.isfinite(fr) & (fr > 0.0), np.log10(fr), np.nan)
    x_nrec = np.where(np.isfinite(nr) & (nr >= 0.0), np.log1p(nr), np.nan)

    out = pd.DataFrame({
        "region_id": t["region_id"],
        "Recurrence": t["Recurrence"],
        "logFC": t["logFC"].astype(float),
        "x_formrate": x_form.astype(float),
        "x_nrecur": x_nrec.astype(float),
        "formation_rate": fr.astype(float),
        "n_recur_events": nr.astype(float),
    })
    return out

# ------------------------- PLOTTING / TEST UTILS -------------------------

def annotate_line_panel(ax, n: int,
                        ols: Optional[Dict[str, float]],
                        ts: Optional[Tuple[float, float, float, float]],
                        sp: Optional[Tuple[float, float, float]],
                        kd: Optional[Tuple[float, float]]):
    """
    ax: matplotlib axis
    n: sample size
    ols: dict(beta, ci_low, ci_high, p, r2)
    ts:  (slope, intercept, lo_slope, hi_slope)
    sp:  (rho, ci_lo, ci_hi)
    kd:  (tau, p_perm)
    """
    lines = [f"n={n}"]
    if ols is not None:
        lines.append(f"OLS β={ols['beta']:+.3f} [{ols['ci_low']:+.3f},{ols['ci_high']:+.3f}], p={_fmt_p(ols['p'])}, R²={ols['r2']:.3f}")
    if ts is not None:
        slope, intercept, lo, hi = ts
        lines.append(f"Theil–Sen slope={slope:+.3f} [{lo:+.3f},{hi:+.3f}]")
    if sp is not None:
        rho, lo, hi = sp
        lines.append(f"Spearman ρ={rho:+.3f} [{lo:+.3f},{hi:+.3f}]")
    if kd is not None:
        tau, p = kd
        lines.append(f"Kendall τ={tau:+.3f}, perm p={_fmt_p(p)}")
    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, ha="left", va="top",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, linewidth=0.0))

# ------------------------- MAIN -----------------------

def main():
    # Load & match (strict)
    matched = load_and_match(OUTPUT_CSV, INVINFO_TSV)

    # ε for π logs (use all π across both arms)
    all_pi = np.r_[matched["pi_direct"].to_numpy(float), matched["pi_inverted"].to_numpy(float)]
    eps = choose_floor_from_quantile(all_pi, q=FLOOR_QUANTILE, min_floor=MIN_FLOOR)

    # Build logFC table
    logfc = build_logfc_table(matched, eps=eps)

    # ---------- π ANALYSES: logFC vs formation rate (SE, RE) ----------
    fig1, axes1 = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for i, (cat, color) in enumerate([("Single-event", COLOR_SE), ("Recurrent", COLOR_RE)]):
        ax = axes1[i]
        sub = logfc.loc[logfc["Recurrence"] == cat].copy()
        x_raw = sub["x_formrate"].to_numpy(float)
        y_raw = sub["logFC"].to_numpy(float)

        mfin = np.isfinite(x_raw) & np.isfinite(y_raw)
        x = x_raw[mfin]; y = y_raw[mfin]
        n = int(mfin.sum())
        assert n >= 3, f"No sufficient data (>=3) for {cat} panel: logFC vs formation rate."

        ax.scatter(x, y, s=32, alpha=0.82, c=color)
        ax.set_title(f"logFC vs formation rate — {cat}", fontsize=16)
        ax.set_xlabel("log10(Formation rate)", fontsize=14)
        ax.set_ylabel("logFC = log(π_inv+ε) − log(π_dir+ε)", fontsize=14)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', labelsize=13)

        # Theil–Sen line (single straight line)
        slope, intercept, lo_slope, hi_slope = theilslopes(y, x, alpha=0.05)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, intercept + slope*xs, linewidth=2.2, c=color)

        # OLS-HC3 inference
        ols = ols_hc3_slope(y, x)

        # Spearman + Kendall + permutation + bootstrap CI
        rho, _ = spearmanr(x, y)
        p_rho_perm = spearman_perm_p(x, y, n_perm=N_PERMUTATIONS, seed=PERM_SEED + i*11 + 1)
        ci_lo, ci_hi = spearman_boot_ci(x, y, n_boot=N_BOOTSTRAP, seed=BOOT_SEED + i*11 + 1, conf_level=CONF_LEVEL)

        tau, _ = kendalltau(x, y)
        p_tau_perm = kendall_perm_p(x, y, n_perm=N_PERMUTATIONS, seed=PERM_SEED + i*11 + 101)

        annotate_line_panel(
            ax, n=n,
            ols=ols,
            ts=(float(slope), float(intercept), float(lo_slope), float(hi_slope)),
            sp=(float(rho), float(ci_lo), float(ci_hi)),
            kd=(float(tau), float(p_tau_perm))
        )
    fig1.suptitle("Paired π outcome: logFC vs formation rate", fontsize=16)
    fig1.savefig(OUT_LOGFC_VS_FORMRATE_FIG, bbox_inches="tight")

    # ---------- π ANALYSES: logFC vs # recurrent events (SE, RE) ----------
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for i, (cat, color) in enumerate([("Single-event", COLOR_SE), ("Recurrent", COLOR_RE)]):
        ax = axes2[i]
        sub = logfc.loc[logfc["Recurrence"] == cat].copy()
        x_raw = sub["x_nrecur"].to_numpy(float)
        y_raw = sub["logFC"].to_numpy(float)

        mfin = np.isfinite(x_raw) & np.isfinite(y_raw)
        x = x_raw[mfin]; y = y_raw[mfin]
        n = int(mfin.sum())
        assert n >= 3, f"No sufficient data (>=3) for {cat} panel: logFC vs #recurrent."

        # If predictor is constant in this class, fail (per spec we require working panels)
        assert np.nanstd(x) > 0.0, f"Predictor constant in {cat} panel (#recurrent)."

        ax.scatter(x, y, s=32, alpha=0.82, c=color)
        ax.set_title(f"logFC vs # recurrent events — {cat}", fontsize=16)
        ax.set_xlabel("ln(1 + Number of recurrent events)", fontsize=14)
        ax.set_ylabel("logFC = log(π_inv+ε) − log(π_dir+ε)", fontsize=14)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', labelsize=13)

        # Theil–Sen line
        slope, intercept, lo_slope, hi_slope = theilslopes(y, x, alpha=0.05)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, intercept + slope*xs, linewidth=2.2, c=color)

        # OLS-HC3 inference
        ols = ols_hc3_slope(y, x)

        # Spearman + Kendall + permutation + bootstrap CI
        rho, _ = spearmanr(x, y)
        p_rho_perm = spearman_perm_p(x, y, n_perm=N_PERMUTATIONS, seed=PERM_SEED + i*17 + 2)
        ci_lo, ci_hi = spearman_boot_ci(x, y, n_boot=N_BOOTSTRAP, seed=BOOT_SEED + i*17 + 2, conf_level=CONF_LEVEL)

        tau, _ = kendalltau(x, y)
        p_tau_perm = kendall_perm_p(x, y, n_perm=N_PERMUTATIONS, seed=PERM_SEED + i*17 + 102)

        annotate_line_panel(
            ax, n=n,
            ols=ols,
            ts=(float(slope), float(intercept), float(lo_slope), float(hi_slope)),
            sp=(float(rho), float(ci_lo), float(ci_hi)),
            kd=(float(tau), float(p_tau_perm))
        )
    fig2.suptitle("Paired π outcome: logFC vs number of recurrent events", fontsize=16)
    fig2.savefig(OUT_LOGFC_VS_NRECUR_FIG, bbox_inches="tight")

    # ---------- FST ANALYSES: formation rate vs FST; #recur vs FST ----------
    # Formation rate vs FST (two categories)
    fig3, axes3 = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for i, (cat, color) in enumerate([("Single-event", COLOR_SE), ("Recurrent", COLOR_RE)]):
        ax = axes3[i]
        sub = matched.loc[matched["Recurrence"] == cat].copy()
        x_raw = pd.to_numeric(sub["formation_rate"], errors="coerce").to_numpy(float)
        y_raw = pd.to_numeric(sub["fst_hudson"], errors="coerce").to_numpy(float)

        mfin = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0.0)
        x = x_raw[mfin]; y = y_raw[mfin]
        n = int(mfin.sum())
        assert n >= 3, f"No sufficient data (>=3) for {cat} panel: FST vs formation rate."

        Xp = np.log10(x)
        ax.scatter(Xp, y, s=32, alpha=0.80, c=color)

        xs, ys = _lowess_xy(Xp, y, frac=LOWESS_FRAC)
        ax.plot(xs, ys, linewidth=2.2, c=color)

        ax.set_title(f"FST (0v1) vs formation rate — {cat}", fontsize=16)
        ax.set_xlabel("log10(Formation rate)", fontsize=14)
        ax.set_ylabel("Hudson F_ST (hap 0 vs 1)", fontsize=14)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', labelsize=13)

        rho, _ = spearmanr(x, y)
        tau, _ = kendalltau(x, y)
        ax.text(0.02, 0.98, f"n={n}\nSpearman ρ={rho:+.3f}\nKendall τ={tau:+.3f}",
                ha="left", va="top", transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, linewidth=0.0))
    fig3.suptitle("FST vs formation rate (two categories)", fontsize=16)
    fig3.savefig(OUT_FST_VS_FORMRATE_FIG, bbox_inches="tight")

    # #recurrent vs FST (two categories)
    fig4, axes4 = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for i, (cat, color) in enumerate([("Single-event", COLOR_SE), ("Recurrent", COLOR_RE)]):
        ax = axes4[i]
        sub = matched.loc[matched["Recurrence"] == cat].copy()
        x_raw = pd.to_numeric(sub["n_recur_events"], errors="coerce").to_numpy(float)
        y_raw = pd.to_numeric(sub["fst_hudson"], errors="coerce").to_numpy(float)

        mfin = np.isfinite(x_raw) & np.isfinite(y_raw)
        x = x_raw[mfin]; y = y_raw[mfin]
        n = int(mfin.sum())
        assert n >= 3, f"No sufficient data (>=3) for {cat} panel: FST vs #recurrent."

        rng = np.random.default_rng(73 + i)
        xj = x + rng.uniform(-JITTER_X, +JITTER_X, size=n)

        ax.scatter(xj, y, s=32, alpha=0.80, c=color)

        xs, ys = _lowess_xy(x, y, frac=LOWESS_FRAC)
        ax.plot(xs, ys, linewidth=2.2, c=color)

        ax.set_title(f"FST (0v1) vs # recurrent events — {cat}", fontsize=16)
        ax.set_xlabel("Number of recurrent events", fontsize=14)
        ax.set_ylabel("Hudson F_ST (hap 0 vs 1)", fontsize=14)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', labelsize=13)

        rho, _ = spearmanr(x, y)
        tau, _ = kendalltau(x, y)
        ax.text(0.02, 0.98, f"n={n}\nSpearman ρ={rho:+.3f}\nKendall τ={tau:+.3f}",
                ha="left", va="top", transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, linewidth=0.0))
    fig4.suptitle("FST vs number of recurrent events (two categories)", fontsize=16)
    fig4.savefig(OUT_FST_VS_NRECUR_FIG, bbox_inches="tight")

    # ---------- Save results table ----------
    rows = []

    # logFC vs formation rate (two cats)
    for i, cat in enumerate(["Single-event", "Recurrent"]):
        s = logfc.loc[logfc["Recurrence"] == cat]
        x = s["x_formrate"].to_numpy(float)
        y = s["logFC"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        assert n >= 3, f"No sufficient data (>=3) for export: logFC vs formation — {cat}."
        xm, ym = x[m], y[m]
        ols = ols_hc3_slope(ym, xm)
        slope, intercept, lo_slope, hi_slope = theilslopes(ym, xm, alpha=0.05)
        rho, _ = spearmanr(xm, ym)
        p_rho = spearman_perm_p(xm, ym, n_perm=N_PERMUTATIONS, seed=PERM_SEED + 810 + i)
        ci_lo, ci_hi = spearman_boot_ci(xm, ym, n_boot=N_BOOTSTRAP, seed=BOOT_SEED + 810 + i, conf_level=CONF_LEVEL)
        tau, _ = kendalltau(xm, ym)
        p_tau = kendall_perm_p(xm, ym, n_perm=N_PERMUTATIONS, seed=PERM_SEED + 910 + i)
        rows.append(dict(panel="logFC_vs_formrate", category=cat, n=n,
                         ols_beta=ols["beta"], ols_ci_low=ols["ci_low"], ols_ci_high=ols["ci_high"],
                         ols_p=ols["p"], ols_r2=ols["r2"],
                         theilsen_slope=slope, theilsen_lo=lo_slope, theilsen_hi=hi_slope,
                         spearman_rho=rho, spearman_ci_low=ci_lo, spearman_ci_high=ci_hi, spearman_p_perm=p_rho,
                         kendall_tau=tau, kendall_p_perm=p_tau))

    # logFC vs #recurrent (two cats)
    for i, cat in enumerate(["Single-event", "Recurrent"]):
        s = logfc.loc[logfc["Recurrence"] == cat]
        x = s["x_nrecur"].to_numpy(float)
        y = s["logFC"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        assert n >= 3, f"No sufficient data (>=3) for export: logFC vs #recurrent — {cat}."
        xm, ym = x[m], y[m]
        assert np.nanstd(xm) > 0.0, f"Predictor constant in export: #recurrent — {cat}."
        ols = ols_hc3_slope(ym, xm)
        slope, intercept, lo_slope, hi_slope = theilslopes(ym, xm, alpha=0.05)
        rho, _ = spearmanr(xm, ym)
        p_rho = spearman_perm_p(xm, ym, n_perm=N_PERMUTATIONS, seed=PERM_SEED + 820 + i)
        ci_lo, ci_hi = spearman_boot_ci(xm, ym, n_boot=N_BOOTSTRAP, seed=BOOT_SEED + 820 + i, conf_level=CONF_LEVEL)
        tau, _ = kendalltau(xm, ym)
        p_tau = kendall_perm_p(xm, ym, n_perm=N_PERMUTATIONS, seed=PERM_SEED + 920 + i)
        rows.append(dict(panel="logFC_vs_nrecur", category=cat, n=n,
                         ols_beta=ols["beta"], ols_ci_low=ols["ci_low"], ols_ci_high=ols["ci_high"],
                         ols_p=ols["p"], ols_r2=ols["r2"],
                         theilsen_slope=slope, theilsen_lo=lo_slope, theilsen_hi=hi_slope,
                         spearman_rho=rho, spearman_ci_low=ci_lo, spearman_ci_high=ci_hi, spearman_p_perm=p_rho,
                         kendall_tau=tau, kendall_p_perm=p_tau))

    # FST vs formation (two cats)
    for i, cat in enumerate(["Single-event", "Recurrent"]):
        s = matched.loc[matched["Recurrence"] == cat]
        x = pd.to_numeric(s["formation_rate"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(s["fst_hudson"], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
        n = int(m.sum())
        assert n >= 3, f"No sufficient data (>=3) for export: FST vs formation — {cat}."
        xm, ym = x[m], y[m]
        rho, _ = spearmanr(xm, ym)
        tau, _ = kendalltau(xm, ym)
        rows.append(dict(panel="fst_vs_formrate", category=cat, n=n,
                         spearman_rho=rho, kendall_tau=tau))

    # FST vs #recurrent (two cats)
    for i, cat in enumerate(["Single-event", "Recurrent"]):
        s = matched.loc[matched["Recurrence"] == cat]
        x = pd.to_numeric(s["n_recur_events"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(s["fst_hudson"], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        assert n >= 3, f"No sufficient data (>=3) for export: FST vs #recurrent — {cat}."
        xm, ym = x[m], y[m]
        rho, _ = spearmanr(xm, ym)
        tau, _ = kendalltau(xm, ym)
        rows.append(dict(panel="fst_vs_nrecur", category=cat, n=n,
                         spearman_rho=rho, kendall_tau=tau))

    res_df = pd.DataFrame(rows)
    assert not res_df.empty, "No results produced."
    res_df.to_csv(OUT_RESULTS_CSV, index=False)

if __name__ == "__main__":
    main()
