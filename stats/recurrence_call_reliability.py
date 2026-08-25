"""How trustworthy is a recurrence call at each real locus?

The gene-flux sweep measures the reference classifier's false-positive rate over a
grid of recombination rates, divergence depths and inversion frequencies. Two of
those axes are observable at the real loci: the local recombination rate and the
inverted allele frequency. This script projects each of the 93 analysed
inversions onto the simulated surface using only those two — deliberately *not*
divergence or diversity, which are the quantities the downstream comparisons are
about and would make the exercise circular — and reports the simulated
false-positive rate in the cell the locus falls into.

The output is a per-locus reliability flag: at a locus whose recombination rate
puts it where the classifier is calibrated, a recurrent call can be taken at face
value; at a locus in the low-recombination regime, where the simulated
false-positive rate is high, a recurrent call is weak evidence on its own.

It also asks the aggregate question that matters for the manuscript: are the
observed recurrent calls concentrated in the part of parameter space where the
classifier is unreliable? If they were, the recurrent/single-event contrasts would
be partly an artefact of where the classifier fails.

Inputs:  simulations/refsim/gene_flux_results.csv
         data/inversion_architecture_covariates.tsv
         data/inv_properties.tsv
Outputs: data/recurrence_call_reliability.tsv
         data/recurrence_call_reliability.pdf / .png
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STATS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_STATS)
_DATA = os.path.join(_REPO, "data")
_SIM = os.path.join(_REPO, "simulations", "refsim")

SWEEP = os.path.join(_SIM, "gene_flux_results.csv")
ARCH = os.path.join(_DATA, "inversion_architecture_covariates.tsv")
INVPROPS = os.path.join(_DATA, "inv_properties.tsv")
OUT_TSV = os.path.join(_DATA, "recurrence_call_reliability.tsv")
OUT_PDF = os.path.join(_DATA, "recurrence_call_reliability.pdf")
OUT_PNG = os.path.join(_DATA, "recurrence_call_reliability.png")

SINGLE_SCENARIO_PREFIX = "single"
RHO_LEVELS = None          # taken from the sweep

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL_SINGLE = "#3B5BA5"
COL_RECUR = "#C2601F"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def cm_per_mb_to_rho(cm_per_mb):
    """Convert cM/Mb to a per-base recombination probability per generation."""
    return cm_per_mb / 100.0 / 1e6


def main():
    sweep_path = SWEEP
    if not os.path.exists(sweep_path):
        sys.exit("gene_flux_results.csv is missing; run the production sweep")
    print(f"sweep: {os.path.basename(sweep_path)}")
    sw = pd.read_csv(sweep_path)
    single = sw[sw["scenario"].str.startswith(SINGLE_SCENARIO_PREFIX)]
    if single.empty:
        sys.exit("no single-origin scenario in the sweep")

    # False-positive rate by recombination rate, marginalised over depth,
    # inversion frequency and flux (flux has no effect; see the sweep report).
    by_rho = single.groupby("rho").agg(n=("reps", "sum"), k=("n_called", "sum"))
    by_rho["fpr"] = by_rho["k"] / by_rho["n"]
    by_rho[["lo", "hi"]] = [wilson(r.k, r.n) for r in by_rho.itertuples()]
    print("\nsimulated false-positive rate by recombination rate:")
    for rho, r in by_rho.iterrows():
        print(f"  rho = {rho:.0e}  FPR = {r.fpr:.3f} [{r.lo:.3f}, {r.hi:.3f}]"
              f"  ({int(r.n)} replicates)")

    # the same, split by divergence depth, for the reliability note
    by_rho_depth = (single.groupby(["rho", "depth"])
                    .agg(n=("reps", "sum"), k=("n_called", "sum")))
    by_rho_depth["fpr"] = by_rho_depth["k"] / by_rho_depth["n"]

    arch = pd.read_csv(ARCH, sep="\t")
    props = pd.read_csv(INVPROPS, sep="\t")
    props = props.rename(columns={"0_single_1_recur_consensus": "consensus"})
    m = arch.merge(props[["OrigID", "consensus", "Inverted_AF", "Size_.kbp."]],
                   on="OrigID", how="left")
    m = m[m["consensus"].isin([0, 1])].copy()
    print(f"\n{len(m)} classified loci with a recombination estimate "
          f"({int((m.consensus == 1).sum())} recurrent, "
          f"{int((m.consensus == 0).sum())} single-event)")

    # Each locus is assigned the simulated rho level closest to its own
    # recombination rate on a log scale; loci above the top simulated level are
    # assigned to it, which is conservative (the classifier is best there).
    rhos = np.array(sorted(by_rho.index))
    m["rho_locus"] = cm_per_mb_to_rho(m["recomb_cM_per_Mb_flank"])
    def nearest(v):
        if not np.isfinite(v):
            return np.nan
        lv = np.log10(np.maximum(v, 1e-12))
        cand = np.array([np.log10(max(r, 1e-12)) for r in rhos])
        return rhos[int(np.argmin(np.abs(cand - lv)))]
    m["rho_bin"] = m["rho_locus"].map(nearest)
    m["simulated_fpr"] = m["rho_bin"].map(by_rho["fpr"])
    m["simulated_fpr_lo"] = m["rho_bin"].map(by_rho["lo"])
    m["simulated_fpr_hi"] = m["rho_bin"].map(by_rho["hi"])
    # At a given recombination rate the false-positive rate still depends on how
    # long the arrangements have been separated, which is not observable without
    # using divergence. So report the range across depths rather than one number.
    depth_order = ["very_recent", "recent", "young", "old"]
    fpr_by_depth = {(rho, dep): by_rho_depth.loc[(rho, dep), "fpr"]
                    for rho, dep in by_rho_depth.index}
    for dep in depth_order:
        m[f"fpr_{dep}"] = [fpr_by_depth.get((r, dep), np.nan)
                           for r in m["rho_bin"]]
    m["fpr_min_over_depth"] = m[[f"fpr_{d}" for d in depth_order]].min(axis=1)
    m["fpr_max_over_depth"] = m[[f"fpr_{d}" for d in depth_order]].max(axis=1)
    m["reliability"] = np.where(
        m["fpr_max_over_depth"] < 0.05, "calibrated at every depth",
        np.where(m["fpr_min_over_depth"] < 0.05,
                 "calibrated only for old divergences",
                 "weak at every depth"))

    print("\nloci by the reliability of a recurrence call at their "
          "recombination rate:")
    tab = pd.crosstab(m["reliability"],
                      m["consensus"].map({0: "single-event", 1: "recurrent"}))
    print(tab.to_string())

    # Are recurrent calls concentrated where the classifier fails?
    rec = m.loc[m.consensus == 1, "recomb_cM_per_Mb_flank"].dropna()
    sin = m.loc[m.consensus == 0, "recomb_cM_per_Mb_flank"].dropna()
    u, p_mwu = stats.mannwhitneyu(rec, sin, alternative="two-sided")
    print(f"\nrecombination rate, recurrent vs single-event: "
          f"median {rec.median():.3f} vs {sin.median():.3f} cM/Mb, "
          f"Mann-Whitney p = {p_mwu:.3g}")
    weak = m["simulated_fpr"] >= 0.20
    ct = pd.crosstab(weak, m["consensus"])
    if ct.shape == (2, 2):
        odds, p_fis = stats.fisher_exact(ct.to_numpy())
        print(f"recurrent calls in the weak-reliability regime: "
              f"{int(((m.consensus == 1) & weak).sum())}/{int((m.consensus == 1).sum())}"
              f" vs single-event {int(((m.consensus == 0) & weak).sum())}/"
              f"{int((m.consensus == 0).sum())}; Fisher p = {p_fis:.3g}, "
              f"odds ratio = {odds:.2f}")
    else:
        p_fis, odds = np.nan, np.nan

    out = m[["Chromosome", "Start", "End", "OrigID", "consensus",
             "Inverted_AF", "Size_.kbp.", "recomb_cM_per_Mb_flank",
             "rho_locus", "rho_bin", "simulated_fpr", "simulated_fpr_lo",
             "simulated_fpr_hi", "fpr_very_recent", "fpr_recent", "fpr_young",
             "fpr_old", "reliability"]].copy()
    out["consensus"] = out["consensus"].map({0: "single-event", 1: "recurrent"})
    out = out.rename(columns={
        "consensus": "Recurrence class (Porubsky consensus)",
        "recomb_cM_per_Mb_flank": "Recombination rate, 1 Mb flanks (cM/Mb)",
        "rho_locus": "Implied per-base recombination rate",
        "rho_bin": "Nearest simulated recombination rate",
        "simulated_fpr": "Simulated false-positive rate at that rate",
        "simulated_fpr_lo": "95% CI lower", "simulated_fpr_hi": "95% CI upper",
        "reliability": "Reliability of a recurrence call here"})
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nWrote {OUT_TSV}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    ax = axes[0]
    x = np.arange(len(by_rho))
    ax.bar(x, by_rho["fpr"], color="#8C8F99", width=.6)
    ax.errorbar(x, by_rho["fpr"],
                yerr=[by_rho["fpr"] - by_rho["lo"], by_rho["hi"] - by_rho["fpr"]],
                fmt="none", ecolor="#333333", capsize=4, lw=1)
    ax.axhline(0.05, color="#A8342A", ls="--", lw=1.2)
    ax.text(len(x) - .55, 0.055, "5%", color="#A8342A", fontsize=9, va="bottom",
            ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0e}" for r in by_rho.index])
    ax.set_xlabel("recombination rate (per bp per generation)")
    ax.set_ylabel("simulated false-positive rate")
    ax.set_title("A  Where the classifier is calibrated", loc="left", fontsize=11)

    ax = axes[1]
    for code, name, col in ((0, "single-event", COL_SINGLE),
                            (1, "recurrent", COL_RECUR)):
        v = m.loc[m.consensus == code, "recomb_cM_per_Mb_flank"].dropna()
        ax.hist(v, bins=np.linspace(0, max(3.0, m["recomb_cM_per_Mb_flank"].max()),
                                    22),
                alpha=.6, color=col, label=f"{name} (n={len(v)})")
    ax.set_xlabel("recombination rate, 1 Mb flanks (cM/Mb)")
    ax.set_ylabel("inversion loci")
    ax.set_title(f"B  Where the real loci sit\nMann-Whitney p = {p_mwu:.2f}",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    depths = [d for d in ["very_recent", "recent", "young", "old"]
              if d in {i[1] for i in by_rho_depth.index}]
    grid = np.array([[fpr_by_depth.get((r, d), np.nan) for d in depths]
                     for r in by_rho.index])
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=max(0.7, np.nanmax(grid)),
                   aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.45 else "#222222")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([d.replace("_", " ") for d in depths], fontsize=9)
    ax.set_yticks(range(len(by_rho)))
    ax.set_yticklabels([f"{r:.0e}" for r in by_rho.index], fontsize=9)
    ax.set_xlabel("divergence depth")
    ax.set_ylabel("recombination rate (per bp per generation)")
    # mark the row the real loci actually occupy
    row = list(by_rho.index).index(m["rho_bin"].mode().iloc[0])
    ax.add_patch(plt.Rectangle((-0.5, row - 0.5), len(depths), 1, fill=False,
                               edgecolor="#111111", lw=2.4))
    ax.set_title("C  Simulated false-positive rate\n"
                 "(boxed row = where the real loci sit)", loc="left",
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PDF} / {OUT_PNG}")


if __name__ == "__main__":
    main()
