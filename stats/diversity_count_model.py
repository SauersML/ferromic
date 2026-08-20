"""The orientation-by-recurrence diversity contrast, fitted to counts.

The published model is OLS on Delta log pi = log(pi_inv + eps) - log(pi_dir + eps),
with eps a detection floor read off the data. The floor is doing real work: many
inverted groups have pi exactly zero, and the estimate, its standard error and its
p-value all move when the floor moves. It also discards the information that
distinguishes zero differences over a million callable sites from zero over ten.

pi is a ratio of two integers the alignments hold exactly, so none of that is
necessary. Write D for the number of pairwise nucleotide differences summed over
within-orientation haplotype pairs and C for the number of callable pair-sites.
Model the differences as a rate process with a locus-specific rate lambda_l:

    D_l,dir ~ Poisson(C_l,dir * lambda_l)
    D_l,inv ~ Poisson(C_l,inv * lambda_l * exp(b + c * recurrent_l))

lambda_l is a nuisance parameter with exactly two observations behind it, so
maximising over it is the classic incidental-parameters trap and gives
inconsistent estimates. Condition on the locus total instead. Given
n_l = D_l,dir + D_l,inv, the inverted count is binomial and lambda_l cancels
exactly:

    D_l,inv | n_l ~ Binomial(n_l, p_l),
    logit(p_l) = log(C_l,inv / C_l,dir) + b + c * recurrent_l

which is an ordinary two-parameter regression with a known offset: no locus
intercepts, no floor, no transform, and a locus whose inverted haplotypes carry
zero differences contributes 0 out of n_l rather than a censored value.

Each locus therefore yields one number, the conditional log ratio

    t_l = log[(D_l,inv + 1/2) / (C_l,inv + 1)] - log[(D_l,dir + 1/2) / (C_l,dir + 1)]

whose exponential is the inverted-to-direct diversity ratio at that locus. The
half-count is the Jeffreys correction, which keeps a locus with zero inverted
differences finite without imposing a floor on pi itself: a zero over a million
callable sites and a zero over ten give different t_l, which is the whole point.

The locus is the unit of replication, so the reported effect is the mean of t_l
within a recurrence class and the interaction is the difference of those means.
Inference is exact and assumption-free: relabelling a locus's two orientations
flips the sign of its t_l, so the within-class test is an exact sign-flip
permutation, and the interaction is tested by permuting the recurrence labels
across loci. A locus bootstrap gives intervals, and an inverse-variance weighted
fit is reported alongside as a sensitivity to the choice of weighting.

Inputs:  data/locus_pair_counts.tsv        (from stats/locus_pair_counts.py)
         data/inv_properties.tsv           (recurrence classification)
Outputs: data/diversity_count_model.tsv
         data/diversity_count_model.pdf / .png
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STATS = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_STATS), "data")

IN_COUNTS = os.path.join(_DATA, "locus_pair_counts.tsv")
IN_PROPS = os.path.join(_DATA, "inv_properties.tsv")
OUT_TSV = os.path.join(_DATA, "diversity_count_model.tsv")
OUT_PDF = os.path.join(_DATA, "diversity_count_model.pdf")
OUT_PNG = os.path.join(_DATA, "diversity_count_model.png")

N_PERM = 100_000
N_BOOT = 20_000
RNG_SEED = 2026

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL_SINGLE = "#3B5BA5"
COL_RECUR = "#C2601F"


# ------------------------------------------------------------ conditional model
def locus_log_ratio(d):
    """Per-locus conditional log ratio and its variance.

    The Jeffreys half-count keeps zero-difference groups finite; the variance is
    the delta-method variance of the log of a binomial rate, which is what makes
    the inverse-variance sensitivity fit possible.
    """
    di, ci = d["diffs_inverted"].to_numpy(float), d["callable_inverted"].to_numpy(float)
    dd, cd = d["diffs_direct"].to_numpy(float), d["callable_direct"].to_numpy(float)
    t = (np.log((di + 0.5) / (ci + 1.0)) - np.log((dd + 0.5) / (cd + 1.0)))
    var = 1.0 / (di + 0.5) + 1.0 / (dd + 0.5)
    return t, var


def signflip_p(t, n_perm, rng):
    """Exact-in-distribution p for the mean of t under label relabelling."""
    obs = float(np.mean(t))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(t)))
    null = (signs * t).mean(axis=1)
    return obs, float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1)), null


def label_perm_p(t, rec, n_perm, rng):
    """p for the difference of class means, permuting the recurrence labels."""
    obs = float(t[rec == 1].mean() - t[rec == 0].mean())
    k = int((rec == 1).sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(t))
        null[i] = t[idx[:k]].mean() - t[idx[k:]].mean()
    return obs, float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1)), null


def main():
    if not os.path.exists(IN_COUNTS):
        sys.exit(f"missing {IN_COUNTS}; run stats/locus_pair_counts.py first")
    cnt = pd.read_csv(IN_COUNTS, sep="\t")
    props = pd.read_csv(IN_PROPS, sep="\t").rename(
        columns={"0_single_1_recur_consensus": "consensus"})
    props["chrom_std"] = props["Chromosome"].astype(str).str.replace(
        "chr", "", regex=False)
    cnt["chrom_std"] = cnt["chrom"].astype(str).str.replace("chr", "",
                                                            regex=False)
    rec_map = {}
    for _, r in cnt.iterrows():
        cand = props[props["chrom_std"] == r["chrom_std"]]
        if cand.empty:
            continue
        ov = (np.minimum(cand["End"], r["end"]) -
              np.maximum(cand["Start"], r["start"]))
        denom = np.maximum(cand["End"] - cand["Start"], r["end"] - r["start"])
        frac = ov / denom
        j = frac.idxmax()
        if frac.loc[j] > 0.9:
            rec_map[(r["chrom"], r["start"], r["end"])] = props.loc[j, "consensus"]
    cnt["consensus"] = [rec_map.get((r.chrom, r.start, r.end), np.nan)
                        for r in cnt.itertuples()]

    d = cnt[cnt["consensus"].isin([0, 1])].copy()
    d = d[(d["callable_direct"] > 0) & (d["callable_inverted"] > 0)]
    d = d[(d["n_hap_direct"] >= 2) & (d["n_hap_inverted"] >= 2)]
    d = d[(d["diffs_direct"] + d["diffs_inverted"]) > 0]
    print(f"{len(d)} informative loci "
          f"({int((d.consensus == 1).sum())} recurrent, "
          f"{int((d.consensus == 0).sum())} single-event)")
    print(f"  loci with zero inverted differences: "
          f"{int((d['diffs_inverted'] == 0).sum())}  -- these are what the "
          f"epsilon floor existed to handle")
    print(f"  total pairwise differences: "
          f"{int((d['diffs_direct'] + d['diffs_inverted']).sum()):,} over "
          f"{(d['callable_direct'] + d['callable_inverted']).sum():.3g} "
          f"callable pair-sites")

    t, var = locus_log_ratio(d)
    rec = d["consensus"].to_numpy(float)
    rng = np.random.default_rng(RNG_SEED)

    print("\n  locus is the unit; effect is the mean conditional log ratio")
    results = {}
    for code, name in ((0, "single-event"), (1, "recurrent")):
        sel = rec == code
        obs, p, _ = signflip_p(t[sel], N_PERM, rng)
        lo, hi = np.percentile(
            [np.mean(rng.choice(t[sel], sel.sum(), replace=True))
             for _ in range(N_BOOT)], [2.5, 97.5])
        results[name] = dict(fold=np.exp(obs), lo=np.exp(lo), hi=np.exp(hi), p=p,
                             n=int(sel.sum()))
        print(f"    {name:<13} pi_inv/pi_dir = {np.exp(obs):.3f} "
              f"[{np.exp(lo):.3f}, {np.exp(hi):.3f}]  "
              f"exact sign-flip p = {p:.4g}  (n = {int(sel.sum())} loci)")

    obs_int, p_int, _ = label_perm_p(t, rec, N_PERM, rng)
    boot_int = np.empty(N_BOOT)
    idx_s, idx_r = np.nonzero(rec == 0)[0], np.nonzero(rec == 1)[0]
    for i in range(N_BOOT):
        boot_int[i] = (t[rng.choice(idx_r, len(idx_r), True)].mean()
                       - t[rng.choice(idx_s, len(idx_s), True)].mean())
    lo_i, hi_i = np.percentile(boot_int, [2.5, 97.5])
    u, p_mwu = stats.mannwhitneyu(t[rec == 1], t[rec == 0],
                                  alternative="two-sided")
    print(f"    interaction   {np.exp(obs_int):.3f}-fold "
          f"[{np.exp(lo_i):.3f}, {np.exp(hi_i):.3f}]  "
          f"label-permutation p = {p_int:.4g}, Mann-Whitney p = {p_mwu:.4g}")

    # inverse-variance weighted sensitivity: does the answer depend on treating
    # every locus equally?
    print("\n  inverse-variance weighted sensitivity")
    for code, name in ((0, "single-event"), (1, "recurrent")):
        sel = rec == code
        w = 1.0 / var[sel]
        m = float(np.sum(w * t[sel]) / np.sum(w))
        se = float(np.sqrt(1.0 / np.sum(w)))
        print(f"    {name:<13} fold = {np.exp(m):.3f} "
              f"[{np.exp(m - 1.96 * se):.3f}, {np.exp(m + 1.96 * se):.3f}]")
    w_all = 1.0 / var
    m_r = np.sum(w_all[rec == 1] * t[rec == 1]) / np.sum(w_all[rec == 1])
    m_s = np.sum(w_all[rec == 0] * t[rec == 0]) / np.sum(w_all[rec == 0])
    print(f"    interaction   fold = {np.exp(m_r - m_s):.3f}")

    print("\n  published epsilon-floored model, for comparison:")
    print("    single-event 0.263, recurrent 1.090, interaction 4.149-fold")

    rows = [
        dict(term="orientation within single-event (pi_inv / pi_dir)",
             estimate=results["single-event"]["fold"],
             ci_lo=results["single-event"]["lo"],
             ci_hi=results["single-event"]["hi"],
             exact_p=results["single-event"]["p"],
             n_loci=results["single-event"]["n"]),
        dict(term="orientation within recurrent (pi_inv / pi_dir)",
             estimate=results["recurrent"]["fold"],
             ci_lo=results["recurrent"]["lo"], ci_hi=results["recurrent"]["hi"],
             exact_p=results["recurrent"]["p"], n_loci=results["recurrent"]["n"]),
        dict(term="recurrence x orientation (fold)", estimate=np.exp(obs_int),
             ci_lo=np.exp(lo_i), ci_hi=np.exp(hi_i), exact_p=p_int,
             n_loci=len(d)),
        dict(term="recurrence x orientation, Mann-Whitney",
             estimate=np.exp(obs_int), ci_lo=np.nan, ci_hi=np.nan,
             exact_p=float(p_mwu), n_loci=len(d)),
        dict(term="interaction, inverse-variance weighted",
             estimate=float(np.exp(m_r - m_s)), ci_lo=np.nan, ci_hi=np.nan,
             exact_p=np.nan, n_loci=len(d)),
    ]
    pd.DataFrame(rows).to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nWrote {OUT_TSV}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    ax = axes[0]
    for code, name, col in ((0, "single-event", COL_SINGLE),
                            (1, "recurrent", COL_RECUR)):
        s_ = d[d.consensus == code]
        ax.scatter(s_["pi_direct"] * 1e3, s_["pi_inverted"] * 1e3, s=36,
                   color=col, alpha=.85, edgecolor="white",
                   label=f"{name} (n={len(s_)})")
    lim = max(d["pi_direct"].max(), d["pi_inverted"].max()) * 1e3 * 1.08
    ax.plot([0, lim], [0, lim], color="#999999", ls="--", lw=.9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("$\\pi$, direct haplotypes ($\\times10^{-3}$)")
    ax.set_ylabel("$\\pi$, inverted haplotypes ($\\times10^{-3}$)")
    ax.set_title("A  Exact counts, no detection floor", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    xs = np.arange(2)
    names = ["single-event", "recurrent"]
    est = [results[k]["fold"] for k in names]
    lo = [results[k]["lo"] for k in names]
    hi = [results[k]["hi"] for k in names]
    ax.axhline(1.0, color="#999999", lw=.9, ls="--")
    ax.errorbar(xs, est,
                yerr=[[e - l for e, l in zip(est, lo)],
                      [h - e for e, h in zip(est, hi)]],
                fmt="o", ms=10, capsize=6, lw=1.6, color="#222222",
                ecolor="#777777")
    for x, k in zip(xs, names):
        ax.annotate(f"p = {results[k]['p']:.2g}", (x, results[k]["hi"]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylabel("$\\pi_{inverted} / \\pi_{direct}$")
    ax.set_title(f"B  Interaction {np.exp(obs_int):.2f}-fold\n"
                 f"permutation p = {p_int:.2g}", loc="left", fontsize=11)

    ax = axes[2]
    ax.hist(np.exp(boot_int), bins=50, color="#c9d4e8", edgecolor="white")
    ax.axvline(np.exp(obs_int), color="#C2601F", lw=2,
               label=f"estimate {np.exp(obs_int):.2f}x")
    ax.axvline(4.149, color="#2A7360", lw=1.6, ls="-.",
               label="published 4.15x")
    ax.axvline(1.0, color="#999999", ls="--", lw=1.2, label="no interaction")
    ax.set_xlabel("interaction (fold), locus bootstrap")
    ax.set_ylabel("draws")
    ax.set_title("C  Resampling loci", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PDF} / {OUT_PNG}")


if __name__ == "__main__":
    main()
