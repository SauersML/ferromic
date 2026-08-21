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

Covariates:
  - ln(inversion length, kbp)        from inv_properties.tsv  (Size_.kbp.)
  - inverted allele frequency        from inv_properties.tsv  (Inverted_AF)
  - SNP density (segregating sites / kbp, both orientations pooled, filtered)
                                      from output.csv segregating-site columns
  - CDS density (# CDS segments per locus / kbp)
                                      from phy_metadata.tsv overlapping each locus
  - local recombination rate            from the Beagle GRCh38 genetic map
  - chromosome-arm position            from UCSC hg38 cytobands

The external references are checksum-pinned in
reproducibility/manuscript_sources.json and converted to the small per-locus
table by stats/inversion_architecture_covariates.py.

Outputs (written to ../data):
  recurrence_controls_summary.tsv      -- all effect estimates (unadj/adj/matched)
  recurrence_controls_covariates.tsv   -- per-locus covariate table used
  recurrence_controls.pdf              -- forest-style summary figure
"""

import os
import sys
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
ARCH_TSV     = _resolve_input("inversion_architecture_covariates.tsv")
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


def union_segregating_from_tracks(g0, g1) -> int:
    """Count positions segregating in *either* orientation, as a UNION (no double count).

    ``g0``/``g1`` are per-position tracks for the two orientations: ``np.nan`` marks an
    uncallable base and a finite value ``> 0`` marks a segregating base (e.g. per-site pi).
    A position segregating in BOTH orientations contributes once, not twice -- the previous
    pooled ``seg0 + seg1`` double-counted shared positions, inflating the SNP-density
    covariate with exactly the cross-orientation sharing that co-varies with recurrence.

    Returns the number of positions where either orientation is callable and segregating.
    Raises ValueError if the two tracks have different lengths.
    """
    a = np.asarray(g0, dtype=float)
    b = np.asarray(g1, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"track length mismatch: {a.shape} vs {b.shape}")
    seg0 = np.isfinite(a) & (a > 0.0)
    seg1 = np.isfinite(b) & (b > 0.0)
    return int(np.count_nonzero(seg0 | seg1))


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
    # NOTE: output.csv exposes only the per-orientation aggregate counts seg0/seg1, NOT the
    # per-position pi tracks, so the true union of segregating positions cannot be recovered
    # here (a position segregating in both orientations is counted once per orientation).
    # When per-position tracks are available, use union_segregating_from_tracks() instead of
    # this sum to avoid double-counting cross-orientation shared sites.
    seg_tot            = pd.to_numeric(best["seg0"], errors="coerce").fillna(0) + \
                         pd.to_numeric(best["seg1"], errors="coerce").fillna(0)
    best["snp_density"] = seg_tot / span_kbp   # per-orientation segregating sites per kbp (pooled)

    # divergence outcomes
    best["fst"]    = pd.to_numeric(best["fst"], errors="coerce")
    best["dxy"]    = pd.to_numeric(best["dxy"], errors="coerce")
    best["pi_avg"] = pd.to_numeric(best["pi_avg"], errors="coerce")
    best["da"]     = best["dxy"] - best["pi_avg"]   # net divergence (Nei's da)

    # CDS density from phy_metadata (count CDS segments overlapping the locus / kbp)
    best["cds_density"] = _cds_density(best)

    # recombination landscape and genomic compartment (see
    # stats/inversion_architecture_covariates.py for how these are derived)
    best = _attach_architecture(best)

    return best


def _attach_architecture(loci: pd.DataFrame) -> pd.DataFrame:
    """Merge in per-locus recombination rate and centromere-relative position.

    Reviewer 3 lists "recombination landscape" and "genomic compartment" among
    the possible confounders. Both need reference data outside the repo's
    variant tables, so stats/inversion_architecture_covariates.py derives them
    from checksum-pinned references before this analysis runs.
    """
    loci = loci.copy()
    for c in ("recomb_cM_per_Mb", "recomb_cM_per_Mb_flank", "rel_arm_position",
              "dist_to_centromere"):
        loci[c] = np.nan
    if not os.path.exists(ARCH_TSV):
        raise FileNotFoundError(
            "Missing inversion_architecture_covariates.tsv. Run "
            "stats/inversion_architecture_covariates.py from the pinned "
            "Beagle map and UCSC cytoband sources first."
        )
    arch = pd.read_csv(ARCH_TSV, sep="\t")
    arch["chr_std"] = arch["Chromosome"].apply(_standardize_chr)
    arch["Start"] = pd.to_numeric(arch["Start"], errors="coerce")
    arch["End"] = pd.to_numeric(arch["End"], errors="coerce")
    cols = ["recomb_cM_per_Mb", "recomb_cM_per_Mb_flank", "rel_arm_position",
            "dist_to_centromere"]

    # match each locus to the architecture row with the greatest reciprocal
    # overlap on the same chromosome: coordinates in output.csv are the analysed
    # span and can differ from inv_properties by a few bp at the breakpoints
    for i, r in loci.iterrows():
        cand = arch[arch["chr_std"] == r["chr_std"]]
        if cand.empty:
            continue
        s, e = float(r["region_start"]), float(r["region_end"])
        ov = np.minimum(cand["End"], e) - np.maximum(cand["Start"], s)
        frac = ov / np.maximum(cand["End"] - cand["Start"], e - s)
        j = frac.idxmax()
        if frac.loc[j] <= 0.5:
            continue
        for c in cols:
            loci.at[i, c] = cand.at[j, c]
    n_ok = int(loci["recomb_cM_per_Mb_flank"].notna().sum())
    print(f"Architecture covariates matched for {n_ok}/{len(loci)} loci")
    if n_ok < 0.8 * len(loci):
        raise ValueError(
            f"Architecture covariates resolved for only {n_ok}/{len(loci)} loci; "
            "at least 80% are required."
        )
    return loci


def _cds_density(loci: pd.DataFrame) -> pd.Series:
    """# distinct CDS records overlapping each locus, per kbp of locus span."""
    if not os.path.exists(PHYMETA_TSV):
        raise FileNotFoundError(
            "Missing phy_metadata.tsv required to calculate CDS density."
        )
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
    """Paired inference where the interval and the p-value come from ONE test.

    Previously the confidence interval was a percentile bootstrap of the mean
    while the p-value was a sign-flip permutation test. Those are different
    procedures with different nulls, and on the matched da contrast they
    disagreed outright: the interval excluded zero while the p-value was 0.081.
    A reader checking one against the other has no way to reconcile that.

    The interval is now obtained by inverting the sign-flip test: it is the set
    of shifts delta for which the paired differences are not significantly
    asymmetric about delta at alpha. The reported p-value is that same test at
    delta = 0, so the interval excludes zero exactly when p < alpha.
    """
    n = diff.size
    obs = float(np.mean(diff))

    def perm_p(delta: float) -> float:
        d = diff - delta
        stat = abs(float(np.mean(d)))
        signs = RNG.choice([-1.0, 1.0], size=(N_PERM, n))
        null = np.abs(signs.dot(d) / n)
        return float((np.count_nonzero(null >= stat) + 1) / (N_PERM + 1))

    p = perm_p(0.0)

    # invert the test by bisection on each side of the observed mean
    alpha = 0.05
    spread = float(np.std(diff, ddof=1)) if n > 1 else abs(obs)
    step = max(spread * 4.0, abs(obs) * 4.0, 1e-12)

    def bound(direction: int) -> float:
        far = obs + direction * step
        while perm_p(far) > alpha and abs(far - obs) < step * 64:
            far += direction * step
        near = obs
        for _ in range(40):
            mid = 0.5 * (near + far)
            if perm_p(mid) > alpha:
                near = mid
            else:
                far = mid
        return 0.5 * (near + far)

    lo, hi = bound(-1), bound(+1)
    out = dict(n_pairs=n, est=obs, ci_lo=float(lo), ci_hi=float(hi), p=float(p))
    if log_scale:
        out.update(ratio=math.exp(obs), ratio_lo=math.exp(lo), ratio_hi=math.exp(hi))
    return out


# ------------------------- REPORTING -------------------------
def _fmt_p_math(p):
    """p for the figure: proper scientific notation rather than e-notation."""
    if p != p:
        return "NA"
    if p >= 1e-3:
        return f"{p:.3f}"
    exp = int(math.floor(math.log10(p)))
    mant = p / (10.0 ** exp)
    return f"{mant:.1f} \\times 10^{{{exp}}}"


def _fmt_p(p):
    if p != p:
        return "NA"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"



# ---------------- CONDITIONAL RANDOMIZATION (primary inference) ----------------
# The null in dispute is  outcome _||_ recurrence | architecture.  Rather than
# trusting an asymptotic variance formula (which assumes the covariates enter
# linearly and correctly) or matching (which deletes two thirds of the loci and,
# as stats/recurrence_controls_calibration.py measures, is anti-conservative at
# 16.6% for the diversity outcome), the reference distribution is built by
# redrawing recurrence labels from P(recurrent | architecture) with the outcomes
# held fixed.  Measured false positive rate: 0.042-0.053 across the three
# outcomes, against a nominal 0.05.
CRT_DRAWS = 4999
CRT_ALPHA = 0.05


def _ridge_propensity(Z, r, ridge=1.0, iters=100):
    X = np.column_stack([np.ones(len(r)), Z]) if Z.size else np.ones((len(r), 1))
    b = np.zeros(X.shape[1])
    pen = ridge * np.eye(X.shape[1]); pen[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(pr * (1 - pr), 1e-6, None)
        step = np.linalg.solve(X.T @ (X * W[:, None]) + pen,
                               X.T @ (r - pr) - pen @ b)
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = np.clip(X @ b, -30, 30)
    return np.clip(1.0 / (1.0 + np.exp(-eta)), 0.02, 0.98)


def crt_inference(y, r, Z, log_scale=False, rng=None):
    """Effect, randomization p-value, and an interval from inverting that test.

    Statistic is the covariate-adjusted recurrence coefficient via Frisch-Waugh,
    T = (r' M y) / (r' M r) with M the residual maker for the covariates.  For a
    candidate effect b the null hypothesis is  y - b*r  _||_  r | Z, and both the
    observed statistic and every null draw are linear in b, so the whole
    inversion is closed form and the interval agrees with the p-value by
    construction.
    """
    rng = rng or np.random.default_rng(20260817)
    y = np.asarray(y, float); r = np.asarray(r, float)
    keep = np.isfinite(y) & np.isfinite(r)
    if Z is not None and Z.size:
        keep &= np.isfinite(Z).all(axis=1)
    y, r = y[keep], r[keep]
    Zk = Z[keep] if (Z is not None and Z.size) else np.empty((keep.sum(), 0))
    n = y.size
    X = np.column_stack([np.ones(n), Zk]) if Zk.shape[1] else np.ones((n, 1))
    M = np.eye(n) - X @ np.linalg.pinv(X.T @ X) @ X.T

    obs = float((r @ M @ y) / (r @ M @ r))
    ps = _ridge_propensity(Zk, r) if Zk.shape[1] else np.full(n, r.mean())
    R = (rng.random((CRT_DRAWS, n)) < ps[None, :]).astype(float)
    MY, MR = M @ y, M @ r
    a = R @ MY
    c = R @ MR
    d = np.einsum("ij,ij->i", R, R @ M)
    ok = np.abs(d) > 1e-12
    a, c, d = a[ok], c[ok], d[ok]

    def pval(b):
        return (np.count_nonzero(np.abs((a - b * c) / d) >= abs(obs - b)) + 1) / (d.size + 1)

    p = pval(0.0)
    spread = float(np.std(y, ddof=1)) or 1.0
    step = max(4 * spread, 4 * abs(obs), 1e-12)

    def bound(direction):
        far = obs + direction * step
        while pval(far) > CRT_ALPHA and abs(far - obs) < step * 64:
            far += direction * step
        near = obs
        for _ in range(60):
            mid = 0.5 * (near + far)
            if pval(mid) > CRT_ALPHA:
                near = mid
            else:
                far = mid
        return 0.5 * (near + far)

    lo, hi = bound(-1), bound(+1)
    out = dict(n=int(n), est=obs, ci_lo=float(lo), ci_hi=float(hi), p=float(p))
    if log_scale:
        out.update(ratio=math.exp(obs), ratio_lo=math.exp(lo), ratio_hi=math.exp(hi))
    return out


def build_crt_summary(loci, eps, COVS, COVS_EXT):
    """One test, three conditioning sets, for each of the three outcomes."""
    rng = np.random.default_rng(20260817)
    r = loci["recur"].to_numpy(float)
    dlogpi = (np.log(loci["pi_inverted"].to_numpy(float) + eps)
              - np.log(loci["pi_direct"].to_numpy(float) + eps))
    levels = [("no conditioning", []),
              ("conditioned on length, allele frequency,\nvariant density and coding density", COVS),
              ("conditioned on length, allele frequency, variant density,\ncoding density, recombination rate\nand chromosome arm position", COVS_EXT)]
    outcomes = [("Delta-log pi interaction (ratio)", dlogpi, True),
                ("Hudson FST (Recurrent - Single)", loci["fst"].to_numpy(float), False),
                ("da = Dxy - pi_avg (Recurrent - Single)", loci["da"].to_numpy(float), False)]
    rows = []
    for oname, y, is_ratio in outcomes:
        for lname, cols in levels:
            if cols is None:
                continue
            Z = loci[cols].to_numpy(float) if cols else np.empty((len(loci), 0))
            res = crt_inference(y, r, Z, log_scale=is_ratio, rng=rng)
            rows.append(dict(
                outcome=oname, control=lname,
                effect=res["ratio"] if is_ratio else res["est"],
                ci_lo=res["ratio_lo"] if is_ratio else res["ci_lo"],
                ci_hi=res["ratio_hi"] if is_ratio else res["ci_hi"],
                p=res["p"], n=res["n"], n_recur=int(r.sum()),
                n_single=int((1 - r).sum()),
                scale="ratio" if is_ratio else "difference"))
    return pd.DataFrame(rows)


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

    # extended set: + recombination landscape and genomic compartment
    have_arch = loci["recomb_cM_per_Mb_flank"].notna().sum() >= 0.8 * len(loci)
    if not have_arch:
        raise ValueError("Extended architecture covariates are incomplete.")
    loci["z_recomb"] = _zscore(np.log1p(loci["recomb_cM_per_Mb_flank"]))
    loci["z_arm"] = _zscore(loci["rel_arm_position"])
    COVS_EXT = COVS + ["z_recomb", "z_arm"]

    # covariate balance (recurrent vs single, raw scale)
    print("\nCovariate means by class (raw):")
    bal = [("size_kbp", "Size (kbp)"), ("inv_af", "Inverted AF"),
           ("snp_density", "SNP density /kbp"), ("cds_density", "CDS density /kbp")]
    if have_arch:
        bal += [("recomb_cM_per_Mb_flank", "Recomb cM/Mb (flank)"),
                ("rel_arm_position", "Rel. arm position")]
    for c, lbl in bal:
        mr = loci.loc[loci.recur == 1, c].mean()
        ms = loci.loc[loci.recur == 0, c].mean()
        try:
            _, pdiff = mannwhitneyu(loci.loc[loci.recur == 1, c].dropna(),
                                    loci.loc[loci.recur == 0, c].dropna(),
                                    alternative="two-sided")
        except ValueError:
            pdiff = float("nan")
        print(f"  {lbl:<22} Recurrent={mr:<10.4g} Single={ms:<10.4g} "
              f"MWU p={_fmt_p(pdiff)}")

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

    a_ext = fit_dlogpi(loci, covs=COVS_EXT, eps=eps) if COVS_EXT else None
    if a_ext:
        print(f"  Adjusted+  : ratio={a_ext['ratio']:.3f} "
              f"[{a_ext['ci_lo']:.3f},{a_ext['ci_hi']:.3f}] "
              f"p={_fmt_p(a_ext['p'])} (n={a_ext['n']}) "
              f"[+recombination rate, arm position]")

    matched_af = nn_match(loci, ["z_lnsize", "z_af"], CALIPER_SD)
    a_mat = matched_dlogpi(matched_af, eps=eps)
    print(f"  Matched(len+AF): ratio={a_mat['ratio']:.3f} "
          f"[{a_mat['ratio_lo']:.3f},{a_mat['ratio_hi']:.3f}] p={_fmt_p(a_mat['p'])} "
          f"(pairs={a_mat['n_pairs']})")

    matched_ext = (nn_match(loci, ["z_lnsize", "z_af", "z_recomb"], CALIPER_SD)
                   if COVS_EXT else None)
    a_mat2 = matched_dlogpi(matched_ext, eps=eps) if matched_ext is not None \
        and len(matched_ext) else None
    if a_mat2:
        print(f"  Matched(len+AF+recomb): ratio={a_mat2['ratio']:.3f} "
              f"[{a_mat2['ratio_lo']:.3f},{a_mat2['ratio_hi']:.3f}] "
              f"p={_fmt_p(a_mat2['p'])} (pairs={a_mat2['n_pairs']})")

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
    if a_ext:
        rows.append(dict(
            outcome="Delta-log pi interaction (ratio)",
            control="adjusted + recombination/compartment",
            effect=a_ext['ratio'], ci_lo=a_ext['ci_lo'], ci_hi=a_ext['ci_hi'],
            p=a_ext['p'], n=a_ext['n'], n_recur=a_ext['n_recur'],
            n_single=a_ext['n_single'], scale="ratio"))
    if a_mat2:
        rows.append(dict(
            outcome="Delta-log pi interaction (ratio)",
            control="matched (length+AF+recomb)",
            effect=a_mat2['ratio'], ci_lo=a_mat2['ratio_lo'],
            ci_hi=a_mat2['ratio_hi'], p=a_mat2['p'], n=2 * a_mat2['n_pairs'],
            n_recur=a_mat2['n_pairs'], n_single=a_mat2['n_pairs'], scale="ratio"))

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

        b_ext = fit_divergence(loci, outcome, covs=COVS_EXT) if COVS_EXT else None
        if b_ext:
            print(f"  Adjusted+  : diff={b_ext['est']:+.4g} "
                  f"[{b_ext['ci_lo']:+.4g},{b_ext['ci_hi']:+.4g}] "
                  f"p={_fmt_p(b_ext['p'])} (n={b_ext['n']}) "
                  f"[+recombination rate, arm position]")

        b_mat = matched_divergence(matched_af, outcome)
        print(f"  Matched(len+AF): diff={b_mat['est']:+.4g} "
              f"[{b_mat['ci_lo']:+.4g},{b_mat['ci_hi']:+.4g}] p={_fmt_p(b_mat['p'])} "
              f"(pairs={b_mat['n_pairs']})")

        b_mat2 = (matched_divergence(matched_ext, outcome)
                  if matched_ext is not None and len(matched_ext) else None)
        if b_mat2:
            print(f"  Matched(len+AF+recomb): diff={b_mat2['est']:+.4g} "
                  f"[{b_mat2['ci_lo']:+.4g},{b_mat2['ci_hi']:+.4g}] "
                  f"p={_fmt_p(b_mat2['p'])} (pairs={b_mat2['n_pairs']})")

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
        if b_ext:
            rows.append(dict(
                outcome=label, control="adjusted + recombination/compartment",
                effect=b_ext['est'], ci_lo=b_ext['ci_lo'], ci_hi=b_ext['ci_hi'],
                p=b_ext['p'], n=b_ext['n'], n_recur=b_ext['n_recur'],
                n_single=b_ext['n_single'], scale="difference"))
        if b_mat2:
            rows.append(dict(
                outcome=label, control="matched (length+AF+recomb)",
                effect=b_mat2['est'], ci_lo=b_mat2['ci_lo'], ci_hi=b_mat2['ci_hi'],
                p=b_mat2['p'], n=2 * b_mat2['n_pairs'],
                n_recur=b_mat2['n_pairs'], n_single=b_mat2['n_pairs'],
                scale="difference"))

    # ---- save tables ----
    # legacy rows (asymptotic + matched) are retained for the calibration
    # comparison in stats/recurrence_controls_calibration.py, but the reported
    # inference is the conditional randomization test.
    legacy = pd.DataFrame(rows)
    legacy.to_csv(os.path.join(DATA_DIR, "recurrence_controls_legacy_tests.tsv"),
                  sep="\t", index=False, float_format="%.6g")
    summ = build_crt_summary(loci, eps, COVS, COVS_EXT)
    summ.to_csv(OUT_SUMMARY, sep="\t", index=False, float_format="%.6g")
    print(f"\nWrote summary: {OUT_SUMMARY}")

    covtab = loci[["region_id", "chr_std", "region_start", "region_end", "Recurrence",
                   "recur", "size_kbp", "inv_af", "snp_density", "cds_density",
                   "recomb_cM_per_Mb", "recomb_cM_per_Mb_flank",
                   "rel_arm_position", "dist_to_centromere",
                   "pi_direct", "pi_inverted", "fst", "dxy", "pi_avg", "da"]].copy()
    covtab.to_csv(OUT_COVTAB, sep="\t", index=False, float_format="%.6g")
    print(f"Wrote covariate table: {OUT_COVTAB}")

    make_figure(summ, OUT_FIG)
    print(f"Wrote figure: {OUT_FIG}")


def make_figure(summ: pd.DataFrame, path: str):
    """Monochrome forest panels: one per outcome, one row per control strategy.

    Deliberately black-only. The five control strategies are already separated
    by vertical position and named on the axis, so colour would encode nothing
    that position does not; keeping it black also survives greyscale printing
    and keeps the eye on the interval widths, which are the point.
    """
    outcomes = list(dict.fromkeys(summ["outcome"]))
    # rows come from the summary itself now that there is one test with three
    # conditioning sets, rather than a fixed list of differing procedures
    ctrl_order = list(dict.fromkeys(summ["control"]))
    ctrl_label = {c: c for c in ctrl_order}

    # short x-axis labels; the long outcome string is not repeated as a title
    xlab = {"Delta-log pi interaction (ratio)":
            "orientation by recurrence effect on nucleotide diversity  (ratio, logarithmic scale)",
            "Hudson FST (Recurrent - Single)":
            "difference in $F_{ST}$,  recurrent $-$ single-event",
            "da = Dxy - pi_avg (Recurrent - Single)":
            "difference in net divergence $d_a$,  recurrent $-$ single-event"}

    n_rows = len(ctrl_order)
    fig, axes = plt.subplots(len(outcomes), 1,
                             figsize=(19.0, 0.74 * n_rows * len(outcomes) + 1.2),
                             squeeze=False)
    for ai, oc in enumerate(outcomes):
        ax = axes[ai][0]
        sub = summ[summ["outcome"] == oc].set_index("control")
        is_ratio = sub["scale"].iloc[0] == "ratio"
        null = 1.0 if is_ratio else 0.0
        ys = list(range(n_rows))[::-1]
        lo_all, hi_all = [], []
        for y, ctrl in zip(ys, ctrl_order):
            if ctrl not in sub.index:
                continue
            r = sub.loc[ctrl]
            ax.errorbar(r["effect"], y,
                        xerr=[[r["effect"] - r["ci_lo"]],
                              [r["ci_hi"] - r["effect"]]],
                        fmt="o", color="black", ecolor="black",
                        elinewidth=2.2, capsize=5, capthick=2.2,
                        ms=11, mfc="black",
                        mec="black", mew=2.2, zorder=3)
            lo_all.append(r["ci_lo"]); hi_all.append(r["ci_hi"])
            # one string, so the two fields can never collide when a long
            # y-label narrows the axes and shifts the axes-fraction positions
            ax.annotate(f"$p = {_fmt_p_math(r['p'])}$,   {int(r['n'])} loci",
                        (1.03, y), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=14, color="black",
                        annotation_clip=False)
            lo_all.append(r["ci_lo"]); hi_all.append(r["ci_hi"])
        ax.axvline(null, color="black", ls=(0, (5, 5)), lw=1.6, zorder=1)
        if is_ratio:
            ax.set_xscale("log")
        if lo_all and hi_all:
            lo, hi = min(lo_all), max(hi_all)
            if is_ratio:
                ax.set_xlim(lo / 1.9, hi * 1.6)
            else:
                pad = (hi - lo) if hi > lo else abs(hi) or 1.0
                ax.set_xlim(lo - 0.10 * pad, hi + 0.12 * pad)
        ax.set_yticks(ys)
        ax.set_yticklabels([ctrl_label.get(c, c) for c in ctrl_order],
                           fontsize=13, color="black")
        xlabel = xlab.get(oc, oc)
        if not is_ratio:
            # small da values ran their tick labels together. Use a sparse
            # locator, and where the values are tiny fold the power of ten into
            # the axis label rather than leaving a floating offset that lands on
            # top of it.
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
            span = max(abs(v) for v in (list(lo_all) + list(hi_all))) if lo_all else 1.0
            if span < 0.01:
                exp = int(np.floor(np.log10(span)))
                scale = 10.0 ** exp
                ax.xaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda v, _p, sc=scale: f"{v / sc:g}"))
                xlabel = f"{xlabel}   ($\\times 10^{{{exp}}}$)"
        ax.set_xlabel(xlabel, fontsize=16, color="black", labelpad=8)
        ax.tick_params(axis="x", labelsize=14, colors="black", width=1.4,
                       length=6)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, n_rows - 0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("black")
            ax.spines[sp].set_linewidth(1.4)
    fig.tight_layout(h_pad=1.6, rect=[0, 0, 0.76, 1])
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
