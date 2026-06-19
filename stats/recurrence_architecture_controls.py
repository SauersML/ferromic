#!/usr/bin/env python3
"""
Recurrence effects controlling for genomic architecture (Reviewer 3, comment 3).

The recurrent-vs-single-event evolutionary contrasts in the manuscript could in
principle be confounded by genomic architecture: inversion length, allele
frequency, local SNP density, and gene/CDS density all differ systematically
between recurrent and single-event inversions and could drive the diversity and
divergence contrasts. This script shows the recurrence effects survive both
(i) regression adjustment for those covariates and (ii) covariate matching of
recurrent vs single-event inversions.

Two outcomes are tested:
  (A) Per-locus orientation diversity contrast — the Δlogπ interaction
      (Recurrent vs Single-event difference in inverted-vs-direct log π), the
      same quantity estimated by Model A/C in stats/inv_dir_recur_model.py.
  (B) The recurrent-vs-single-event divergence contrasts: Hudson FST and
      da (= Dxy - mean within-group π) between orientation groups.

Covariates, all derivable from existing repo data:
  - ln(inversion length, kbp)        from inv_properties.tsv  (Size_.kbp.)
  - inverted allele frequency        from inv_properties.tsv  (Inverted_AF)
  - SNP density (segregating sites / kbp, both orientations pooled, filtered)
                                      from output.csv segregating-site columns
  - CDS density (# CDS segments per locus / kbp)
                                      from phy_metadata.tsv overlapping each locus

NOTE on recombination rate: a genetic map is external data and is deliberately
NOT used here. Local SNP density serves as a within-locus proxy for the local
mutational/recombinational background; residual confounding by recombination
rate is acknowledged as a limitation in the response to reviewers.

Strict policy: uses ONLY data already in the repo. No new datasets.

Outputs (written to ../data):
  recurrence_controls_summary.tsv      -- all effect estimates (unadj/adj/matched)
  recurrence_controls_covariates.tsv   -- per-locus covariate table used
  recurrence_controls.pdf              -- forest-style summary figure
"""

import os
import sys
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- PATHS -------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.abspath(os.path.join(HERE, "..", "data"))


def _resolve_input(name: str) -> str:
    """Prefer a fresh copy in the CWD (CI working dir), else fall back to data/."""
    for base in (os.getcwd(), DATA_DIR):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, name)


OUTPUT_CSV   = _resolve_input("output.csv")
INVINFO_TSV  = _resolve_input("inv_properties.tsv")
PHYMETA_TSV  = _resolve_input("phy_metadata.tsv")

OUT_SUMMARY  = os.path.join(DATA_DIR, "recurrence_controls_summary.tsv")
OUT_COVTAB   = os.path.join(DATA_DIR, "recurrence_controls_covariates.tsv")
OUT_FIG      = os.path.join(DATA_DIR, "recurrence_controls.pdf")

# ------------------------- SETTINGS -------------------------
FLOOR_QUANTILE = 0.01      # same epsilon rule as inv_dir_recur_model.py
MIN_FLOOR      = 1e-8
N_BOOT         = 5000      # bootstrap reps for matched contrasts
N_PERM         = 10000     # permutation reps for matched contrasts
SEED           = 2025
CALIPER_SD     = 1.0       # nearest-neighbour caliper, in SD of the matching distance

RNG = np.random.default_rng(SEED)


# ------------------------- HELPERS -------------------------
def _standardize_chr(val) -> str:
    s = str(val).strip()
    return s[3:] if s.lower().startswith("chr") else s


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0.0:
        return pd.Series(np.where(np.isfinite(x), 0.0, np.nan), index=s.index)
    return pd.Series((x - mu) / sd, index=s.index)


def choose_floor(pi_all: np.ndarray) -> float:
    pos = pi_all[np.isfinite(pi_all) & (pi_all > 0)]
    if pos.size == 0:
        raise ValueError("All pi values non-positive.")
    return max(float(np.quantile(pos, FLOOR_QUANTILE)) * 0.5, MIN_FLOOR)


# ------------------------- LOADING & MATCHING -------------------------
def load_loci() -> pd.DataFrame:
    """
    Build one row per classified (consensus 0/1) inversion locus, carrying:
      - orientation pi (direct/inverted), Hudson FST, Dxy, mean pi -> da
      - segregating-site counts (for SNP density)
      - Size_.kbp., Inverted_AF, recurrence class
    Coordinate matching to output.csv uses the same +/-1bp strategy as
    inv_dir_recur_model.py / overall_fst_by_type.py.
    """
    df  = pd.read_csv(OUTPUT_CSV)
    inv = pd.read_csv(INVINFO_TSV, sep="\t")

    df["chr_std"]  = df["chr"].apply(_standardize_chr)
    inv["chr_std"] = inv["Chromosome"].apply(_standardize_chr)

    inv["cons_int"] = pd.to_numeric(inv["0_single_1_recur_consensus"], errors="coerce")
    inv = inv[inv["cons_int"].isin([0, 1])].copy()
    inv["Start"] = pd.to_numeric(inv["Start"], errors="coerce").astype("Int64")
    inv["End"]   = pd.to_numeric(inv["End"],   errors="coerce").astype("Int64")

    # output.csv columns we need
    keep_cols = {
        "0_pi_filtered": "pi_direct",
        "1_pi_filtered": "pi_inverted",
        "hudson_fst_hap_group_0v1": "fst",
        "hudson_dxy_hap_group_0v1": "dxy",
        "hudson_pi_avg_hap_group_0v1": "pi_avg",
        "0_segregating_sites_filtered": "seg0",
        "1_segregating_sites_filtered": "seg1",
        "inversion_freq_filter": "inv_freq_out",
    }
    base = ["chr_std", "region_start", "region_end"]
    d = df[base + list(keep_cols)].rename(columns=keep_cols).copy()
    d["region_start"] = pd.to_numeric(d["region_start"], errors="coerce").fillna(-1).astype(int)
    d["region_end"]   = pd.to_numeric(d["region_end"],   errors="coerce").fillna(-1).astype(int)

    # +/-1bp candidate expansion
    cands = []
    for ds in (-1, 0, 1):
        for de in (-1, 0, 1):
            tmp = d.copy()
            tmp["Start"] = tmp["region_start"] + ds
            tmp["End"]   = tmp["region_end"]   + de
            tmp["prio"]  = abs(ds) + abs(de)
            cands.append(tmp)
    cand = pd.concat(cands, ignore_index=True)
    cand["Start"] = cand["Start"].astype("Int64")
    cand["End"]   = cand["End"].astype("Int64")

    inv_small = inv[["chr_std", "Start", "End", "cons_int", "Size_.kbp.", "Inverted_AF"]].copy()
    merged = cand.merge(inv_small, on=["chr_std", "Start", "End"], how="inner")

    # best (closest) match per locus
    key = ["chr_std", "region_start", "region_end"]
    merged = merged.sort_values("prio")
    best = merged.drop_duplicates(subset=key, keep="first").copy()

    best["pi_direct"]   = pd.to_numeric(best["pi_direct"], errors="coerce")
    best["pi_inverted"] = pd.to_numeric(best["pi_inverted"], errors="coerce")
    best = best[np.isfinite(best["pi_direct"]) & np.isfinite(best["pi_inverted"])].copy()

    best["Recurrence"] = best["cons_int"].map({0: "Single-event", 1: "Recurrent"})
    best["recur"] = best["cons_int"].astype(int)
    best["region_id"] = (best["chr_std"].astype(str) + ":" +
                         best["region_start"].astype(str) + "-" +
                         best["region_end"].astype(str))

    # ---- covariates ----
    best["size_kbp"]   = pd.to_numeric(best["Size_.kbp."], errors="coerce")
    best["inv_af"]     = pd.to_numeric(best["Inverted_AF"], errors="coerce")
    span_kbp           = (best["region_end"] - best["region_start"]).clip(lower=1) / 1000.0
    best["span_kbp"]   = span_kbp
    seg_tot            = pd.to_numeric(best["seg0"], errors="coerce").fillna(0) + \
                         pd.to_numeric(best["seg1"], errors="coerce").fillna(0)
    best["snp_density"] = seg_tot / span_kbp   # segregating sites per kbp (both orientations)

    # divergence outcomes
    best["fst"]    = pd.to_numeric(best["fst"], errors="coerce")
    best["dxy"]    = pd.to_numeric(best["dxy"], errors="coerce")
    best["pi_avg"] = pd.to_numeric(best["pi_avg"], errors="coerce")
    best["da"]     = best["dxy"] - best["pi_avg"]   # net divergence (Nei's da)

    # CDS density from phy_metadata (count CDS segments overlapping the locus / kbp)
    best["cds_density"] = _cds_density(best)

    return best


def _cds_density(loci: pd.DataFrame) -> pd.Series:
    """# distinct CDS records overlapping each locus, per kbp of locus span."""
    if not os.path.exists(PHYMETA_TSV):
        warnings.warn("phy_metadata.tsv not found; CDS density set to NaN.")
        return pd.Series(np.nan, index=loci.index)
    phy = pd.read_csv(PHYMETA_TSV, sep="\t")
    phy["chr_std"] = phy["chromosome"].apply(_standardize_chr)
    phy["cs"] = pd.to_numeric(phy["overall_cds_start_1based"], errors="coerce")
    phy["ce"] = pd.to_numeric(phy["overall_cds_end_1based"], errors="coerce")
    # one record per (transcript, hap) is double-counting orientation; collapse to
    # unique transcript span so density reflects gene content, not haplotype rows.
    phy_u = phy.dropna(subset=["cs", "ce"]).drop_duplicates(
        subset=["chr_std", "transcript_id", "cs", "ce"])
    out = np.full(len(loci), np.nan)
    by_chr = {c: g for c, g in phy_u.groupby("chr_std")}
    for i, (_, r) in enumerate(loci.iterrows()):
        g = by_chr.get(r["chr_std"])
        span_kbp = max((r["region_end"] - r["region_start"]) / 1000.0, 1e-6)
        if g is None:
            out[i] = 0.0
            continue
        ov = (g["cs"] <= r["region_end"]) & (g["ce"] >= r["region_start"])
        out[i] = float(ov.sum()) / span_kbp
    return pd.Series(out, index=loci.index)


# ------------------------- OUTCOME (A): Delta-log pi interaction -------------------------
def fit_dlogpi(loci: pd.DataFrame, covs: Optional[List[str]], eps: float) -> Dict:
    """
    Outcome (A): logFC = log(pi_inv+eps) - log(pi_dir+eps) ~ Recurrent [+ covs], HC3.
    The recurrence effect of interest is the 'Recurrent' coefficient (the
    interaction = difference in inverted-vs-direct log pi between classes).
    Returns the recurrence effect on log and ratio scale, plus n.
    """
    d = loci.copy()
    d["logFC"] = np.log(d["pi_inverted"].to_numpy(float) + eps) - \
                 np.log(d["pi_direct"].to_numpy(float) + eps)
    d["Recurrent"] = d["recur"].astype(int)

    pred = ["Recurrent"] + (covs or [])
    sub = d.dropna(subset=["logFC"] + pred).copy()
    X = sm.add_constant(sub[pred])
    res = sm.OLS(sub["logFC"], X).fit(cov_type="HC3")

    b = float(res.params["Recurrent"])
    se = float(res.bse["Recurrent"])
    p = float(res.pvalues["Recurrent"])
    return dict(est_log=b, se_log=se, ratio=math.exp(b),
                ci_lo=math.exp(b - 1.96 * se), ci_hi=math.exp(b + 1.96 * se),
                p=p, n=int(sub.shape[0]),
                n_recur=int((sub["Recurrent"] == 1).sum()),
                n_single=int((sub["Recurrent"] == 0).sum()))


# ------------------------- OUTCOME (B): FST / da contrast -------------------------
def fit_divergence(loci: pd.DataFrame, outcome: str, covs: Optional[List[str]]) -> Dict:
    """
    Outcome (B): outcome (fst or da) ~ Recurrent [+ covs], HC3.
    Recurrence effect = additive difference (Recurrent - Single-event) in the
    outcome, adjusted for covariates when provided.
    """
    d = loci.copy()
    d["Recurrent"] = d["recur"].astype(int)
    pred = ["Recurrent"] + (covs or [])
    sub = d.dropna(subset=[outcome] + pred).copy()
    X = sm.add_constant(sub[pred])
    res = sm.OLS(sub[outcome], X).fit(cov_type="HC3")
    b = float(res.params["Recurrent"])
    se = float(res.bse["Recurrent"])
    p = float(res.pvalues["Recurrent"])
    # Mann-Whitney on the raw (unadjusted) contrast, for reference, when no covs
    mwu_p = np.nan
    if not covs:
        r = sub.loc[sub["Recurrent"] == 1, outcome].to_numpy(float)
        s = sub.loc[sub["Recurrent"] == 0, outcome].to_numpy(float)
        if len(r) and len(s):
            mwu_p = float(mannwhitneyu(r, s, alternative="two-sided").pvalue)
    return dict(est=b, se=se, ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                p=p, mwu_p=mwu_p, n=int(sub.shape[0]),
                n_recur=int((sub["Recurrent"] == 1).sum()),
                n_single=int((sub["Recurrent"] == 0).sum()))


# ------------------------- MATCHING -------------------------
def nn_match(loci: pd.DataFrame, match_cols: List[str], caliper_sd: float) -> pd.DataFrame:
    """
    Greedy 1:1 nearest-neighbour matching of recurrent (treated) to single-event
    (control) loci on z-scored match_cols (Mahalanobis-ish Euclidean on z-scores),
    without replacement, within a caliper. Returns matched pairs.
    """
    d = loci.dropna(subset=match_cols).copy().reset_index(drop=True)
    Z = np.column_stack([_zscore(d[c]).to_numpy(float) for c in match_cols])
    treat_idx = np.where(d["recur"].to_numpy() == 1)[0]
    ctrl_idx  = list(np.where(d["recur"].to_numpy() == 0)[0])

    # caliper on the matching distance distribution (all treated-control distances)
    dists_all = []
    for t in treat_idx:
        for c in ctrl_idx:
            dists_all.append(np.linalg.norm(Z[t] - Z[c]))
    cal = caliper_sd * np.std(dists_all) if dists_all else np.inf

    pairs = []
    used = set()
    # order treated by isolation (fewest close controls first) for stability
    order = sorted(treat_idx, key=lambda t: min(np.linalg.norm(Z[t] - Z[c]) for c in ctrl_idx))
    for t in order:
        best_c, best_d = None, np.inf
        for c in ctrl_idx:
            if c in used:
                continue
            dd = np.linalg.norm(Z[t] - Z[c])
            if dd < best_d:
                best_d, best_c = dd, c
        if best_c is not None and best_d <= cal:
            used.add(best_c)
            pairs.append((t, best_c, best_d))

    rows = []
    for pid, (t, c, dd) in enumerate(pairs):
        for idx, role in ((t, "Recurrent"), (c, "Single-event")):
            r = d.iloc[idx].to_dict()
            r["pair_id"] = pid
            r["match_role"] = role
            r["match_dist"] = dd
            rows.append(r)
    return pd.DataFrame(rows)


def matched_dlogpi(matched: pd.DataFrame, eps: float) -> Dict:
    """Paired Delta-logpi interaction within matched pairs: bootstrap + permutation."""
    m = matched.copy()
    m["logFC"] = np.log(m["pi_inverted"].to_numpy(float) + eps) - \
                 np.log(m["pi_direct"].to_numpy(float) + eps)
    piv = m.pivot_table(index="pair_id", columns="match_role", values="logFC")
    piv = piv.dropna()
    diff = (piv["Recurrent"] - piv["Single-event"]).to_numpy(float)  # paired, log scale
    return _paired_inference(diff, log_scale=True)


def matched_divergence(matched: pd.DataFrame, outcome: str) -> Dict:
    m = matched.copy()
    piv = m.pivot_table(index="pair_id", columns="match_role", values=outcome).dropna()
    diff = (piv["Recurrent"] - piv["Single-event"]).to_numpy(float)
    return _paired_inference(diff, log_scale=False)


def _paired_inference(diff: np.ndarray, log_scale: bool) -> Dict:
    n = diff.size
    obs = float(np.mean(diff))
    # bootstrap CI
    boots = np.array([np.mean(RNG.choice(diff, size=n, replace=True)) for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # sign-flip permutation p-value (paired)
    cnt = 0
    for _ in range(N_PERM):
        signs = RNG.choice([-1.0, 1.0], size=n)
        if abs(np.mean(signs * diff)) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (N_PERM + 1)
    out = dict(n_pairs=n, est=obs, ci_lo=float(lo), ci_hi=float(hi), p=float(p))
    if log_scale:
        out.update(ratio=math.exp(obs), ratio_lo=math.exp(lo), ratio_hi=math.exp(hi))
    return out


# ------------------------- REPORTING -------------------------
def _fmt_p(p):
    if p != p:
        return "NA"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def main():
    print(f"Reading data from {DATA_DIR}")
    loci = load_loci()
    n_rec = int((loci["recur"] == 1).sum())
    n_sin = int((loci["recur"] == 0).sum())
    print(f"Classified loci matched to output.csv: {loci.shape[0]} "
          f"(Recurrent={n_rec}, Single-event={n_sin})")

    all_pi = np.r_[loci["pi_direct"].to_numpy(float), loci["pi_inverted"].to_numpy(float)]
    eps = choose_floor(all_pi)
    print(f"Detection floor (epsilon) for log pi: {eps:.3g}")

    # z-scored covariates
    loci["z_lnsize"] = _zscore(np.log(loci["size_kbp"].clip(lower=1e-6)))
    loci["z_af"]     = _zscore(loci["inv_af"])
    loci["z_snpden"] = _zscore(np.log1p(loci["snp_density"]))
    loci["z_cdsden"] = _zscore(np.log1p(loci["cds_density"]))
    COVS = ["z_lnsize", "z_af", "z_snpden", "z_cdsden"]

    # covariate balance (recurrent vs single, raw scale)
    print("\nCovariate means by class (raw):")
    for c, lbl in [("size_kbp", "Size (kbp)"), ("inv_af", "Inverted AF"),
                   ("snp_density", "SNP density /kbp"), ("cds_density", "CDS density /kbp")]:
        mr = loci.loc[loci.recur == 1, c].mean()
        ms = loci.loc[loci.recur == 0, c].mean()
        print(f"  {lbl:<18} Recurrent={mr:.4g}  Single={ms:.4g}")

    rows = []

    # ===== OUTCOME A: Delta-log pi interaction =====
    print("\n" + "=" * 70)
    print("OUTCOME A: Delta-log pi recurrence interaction (ratio scale)")
    print("=" * 70)
    a_un  = fit_dlogpi(loci, covs=None, eps=eps)
    a_adj = fit_dlogpi(loci, covs=COVS, eps=eps)
    print(f"  Unadjusted : ratio={a_un['ratio']:.3f} "
          f"[{a_un['ci_lo']:.3f},{a_un['ci_hi']:.3f}] p={_fmt_p(a_un['p'])} "
          f"(n={a_un['n']}, R={a_un['n_recur']}/S={a_un['n_single']})")
    print(f"  Adjusted   : ratio={a_adj['ratio']:.3f} "
          f"[{a_adj['ci_lo']:.3f},{a_adj['ci_hi']:.3f}] p={_fmt_p(a_adj['p'])} (n={a_adj['n']})")

    matched_af = nn_match(loci, ["z_lnsize", "z_af"], CALIPER_SD)
    a_mat = matched_dlogpi(matched_af, eps=eps)
    print(f"  Matched(len+AF): ratio={a_mat['ratio']:.3f} "
          f"[{a_mat['ratio_lo']:.3f},{a_mat['ratio_hi']:.3f}] p={_fmt_p(a_mat['p'])} "
          f"(pairs={a_mat['n_pairs']})")

    rows += [
        dict(outcome="Delta-log pi interaction (ratio)", control="unadjusted",
             effect=a_un['ratio'], ci_lo=a_un['ci_lo'], ci_hi=a_un['ci_hi'],
             p=a_un['p'], n=a_un['n'], n_recur=a_un['n_recur'], n_single=a_un['n_single'],
             scale="ratio"),
        dict(outcome="Delta-log pi interaction (ratio)", control="covariate-adjusted",
             effect=a_adj['ratio'], ci_lo=a_adj['ci_lo'], ci_hi=a_adj['ci_hi'],
             p=a_adj['p'], n=a_adj['n'], n_recur=a_adj['n_recur'], n_single=a_adj['n_single'],
             scale="ratio"),
        dict(outcome="Delta-log pi interaction (ratio)", control="matched (length+AF)",
             effect=a_mat['ratio'], ci_lo=a_mat['ratio_lo'], ci_hi=a_mat['ratio_hi'],
             p=a_mat['p'], n=2 * a_mat['n_pairs'], n_recur=a_mat['n_pairs'],
             n_single=a_mat['n_pairs'], scale="ratio"),
    ]

    # ===== OUTCOME B: FST and da contrasts =====
    for outcome, label in [("fst", "Hudson FST (Recurrent - Single)"),
                           ("da", "da = Dxy - pi_avg (Recurrent - Single)")]:
        print("\n" + "=" * 70)
        print(f"OUTCOME B: {label}")
        print("=" * 70)
        b_un  = fit_divergence(loci, outcome, covs=None)
        b_adj = fit_divergence(loci, outcome, covs=COVS)
        print(f"  Unadjusted : diff={b_un['est']:+.4g} "
              f"[{b_un['ci_lo']:+.4g},{b_un['ci_hi']:+.4g}] p={_fmt_p(b_un['p'])} "
              f"(MWU p={_fmt_p(b_un['mwu_p'])}; n={b_un['n']}, "
              f"R={b_un['n_recur']}/S={b_un['n_single']})")
        print(f"  Adjusted   : diff={b_adj['est']:+.4g} "
              f"[{b_adj['ci_lo']:+.4g},{b_adj['ci_hi']:+.4g}] p={_fmt_p(b_adj['p'])} (n={b_adj['n']})")

        b_mat = matched_divergence(matched_af, outcome)
        print(f"  Matched(len+AF): diff={b_mat['est']:+.4g} "
              f"[{b_mat['ci_lo']:+.4g},{b_mat['ci_hi']:+.4g}] p={_fmt_p(b_mat['p'])} "
              f"(pairs={b_mat['n_pairs']})")

        rows += [
            dict(outcome=label, control="unadjusted", effect=b_un['est'],
                 ci_lo=b_un['ci_lo'], ci_hi=b_un['ci_hi'], p=b_un['p'], n=b_un['n'],
                 n_recur=b_un['n_recur'], n_single=b_un['n_single'], scale="difference"),
            dict(outcome=label, control="covariate-adjusted", effect=b_adj['est'],
                 ci_lo=b_adj['ci_lo'], ci_hi=b_adj['ci_hi'], p=b_adj['p'], n=b_adj['n'],
                 n_recur=b_adj['n_recur'], n_single=b_adj['n_single'], scale="difference"),
            dict(outcome=label, control="matched (length+AF)", effect=b_mat['est'],
                 ci_lo=b_mat['ci_lo'], ci_hi=b_mat['ci_hi'], p=b_mat['p'],
                 n=2 * b_mat['n_pairs'], n_recur=b_mat['n_pairs'],
                 n_single=b_mat['n_pairs'], scale="difference"),
        ]

    # ---- save tables ----
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT_SUMMARY, sep="\t", index=False, float_format="%.6g")
    print(f"\nWrote summary: {OUT_SUMMARY}")

    covtab = loci[["region_id", "chr_std", "region_start", "region_end", "Recurrence",
                   "recur", "size_kbp", "inv_af", "snp_density", "cds_density",
                   "pi_direct", "pi_inverted", "fst", "dxy", "pi_avg", "da"]].copy()
    covtab.to_csv(OUT_COVTAB, sep="\t", index=False, float_format="%.6g")
    print(f"Wrote covariate table: {OUT_COVTAB}")

    make_figure(summ, OUT_FIG)
    print(f"Wrote figure: {OUT_FIG}")


def make_figure(summ: pd.DataFrame, path: str):
    """Forest-style panels: one per outcome, three rows (unadj/adj/matched)."""
    outcomes = list(dict.fromkeys(summ["outcome"]))
    fig, axes = plt.subplots(len(outcomes), 1, figsize=(7.2, 2.1 * len(outcomes)),
                             squeeze=False)
    ctrl_order = ["unadjusted", "covariate-adjusted", "matched (length+AF)"]
    colors = {"unadjusted": "#4C72B0", "covariate-adjusted": "#DD8452",
              "matched (length+AF)": "#55A868"}
    for ai, oc in enumerate(outcomes):
        ax = axes[ai][0]
        sub = summ[summ["outcome"] == oc].set_index("control")
        is_ratio = sub["scale"].iloc[0] == "ratio"
        null = 1.0 if is_ratio else 0.0
        ys = list(range(len(ctrl_order)))[::-1]
        for y, ctrl in zip(ys, ctrl_order):
            if ctrl not in sub.index:
                continue
            r = sub.loc[ctrl]
            ax.errorbar(r["effect"], y,
                        xerr=[[r["effect"] - r["ci_lo"]], [r["ci_hi"] - r["effect"]]],
                        fmt="o", color=colors[ctrl], capsize=3, ms=6)
            ax.annotate(f"p={_fmt_p(r['p'])} (n={int(r['n'])})",
                        (r["ci_hi"], y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.axvline(null, color="grey", ls="--", lw=1)
        if is_ratio:
            ax.set_xscale("log")
        ax.set_yticks(ys)
        ax.set_yticklabels(ctrl_order, fontsize=9)
        ax.set_title(oc, fontsize=10)
        ax.set_ylim(-0.6, len(ctrl_order) - 0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Recurrence effects under architecture controls", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
