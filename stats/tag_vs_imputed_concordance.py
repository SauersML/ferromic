"""Do the imputed dosages give the same PheWAS as a directly genotyped tag SNP?

Reviewer 2 questioned the 17q21.31 imputation on the grounds that its pooled
cross-validated r-squared is low. The direct answer is not another accuracy metric
but the association results themselves: the inversion has published perfect tag
SNPs, so the same phenome can be scanned twice, once with the imputed dosage and
once with a tag SNP, and the two sets of results compared. If the imputation were
failing at this locus the two scans would disagree.

Two panels:
  A  -log10 p from the two scans, across every phecode tested in both.
  B  effect sizes (log odds ratio) from the two scans.

Inputs:  data/phewas_results.tsv        (imputed dosage, all inversions)
         data/all_pop_phewas_tag.tsv    (tagging SNP, 17q21.31)
Outputs: data/tag_vs_imputed_concordance.pdf / .png
         data/tag_vs_imputed_concordance.tsv
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
_DATA = os.path.join(os.path.dirname(_STATS), "data")

IN_IMPUTED = os.path.join(_DATA, "phewas_results.tsv")
IN_TAG = os.path.join(_DATA, "all_pop_phewas_tag.tsv")
OUT_TSV = os.path.join(_DATA, "tag_vs_imputed_concordance.tsv")
OUT_PDF = os.path.join(_DATA, "tag_vs_imputed_concordance.pdf")
OUT_PNG = os.path.join(_DATA, "tag_vs_imputed_concordance.png")

INVERSION = "chr17-45585160-INV-706887"
LOCUS = "17q21.31"
ALPHA = 0.05

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(path, inversion=None):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if inversion is not None and "Inversion" in df.columns:
        df = df[df["Inversion"] == inversion]
    p = pd.to_numeric(df.get("P_LRT_Overall", df.get("P_Value_x")),
                      errors="coerce")
    out = pd.DataFrame({
        "Phenotype": df["Phenotype"],
        "p": p,
        "q": pd.to_numeric(df.get("Q_GLOBAL"), errors="coerce"),
        "or_": pd.to_numeric(df.get("OR"), errors="coerce"),
        "n_cases": pd.to_numeric(df.get("N_Cases"), errors="coerce"),
    })
    return out[np.isfinite(out["p"]) & (out["p"] > 0)].drop_duplicates("Phenotype")


def main():
    for p in (IN_IMPUTED, IN_TAG):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    imp = load(IN_IMPUTED, INVERSION)
    tag = load(IN_TAG)
    m = imp.merge(tag, on="Phenotype", suffixes=("_imputed", "_tag"))
    print(f"{LOCUS}: {len(imp)} phecodes from imputed dosage, {len(tag)} from the "
          f"tag SNP, {len(m)} in common")

    m["logp_imputed"] = -np.log10(m["p_imputed"])
    m["logp_tag"] = -np.log10(m["p_tag"])
    ok_or = (m["or__imputed"] > 0) & (m["or__tag"] > 0)
    m.loc[ok_or, "beta_imputed"] = np.log(m.loc[ok_or, "or__imputed"])
    m.loc[ok_or, "beta_tag"] = np.log(m.loc[ok_or, "or__tag"])

    rho_p, p_rho = stats.spearmanr(m["logp_imputed"], m["logp_tag"])
    r_p, _ = stats.pearsonr(m["logp_imputed"], m["logp_tag"])
    b = m[ok_or]
    rho_b, p_rho_b = stats.spearmanr(b["beta_imputed"], b["beta_tag"])
    r_b, _ = stats.pearsonr(b["beta_imputed"], b["beta_tag"])
    slope = np.polyfit(b["beta_imputed"], b["beta_tag"], 1)[0]
    same_dir = float(np.mean(np.sign(b["beta_imputed"]) == np.sign(b["beta_tag"])))

    sig_i = m["q_imputed"] < ALPHA
    sig_t = m["q_tag"] < ALPHA
    both = int((sig_i & sig_t).sum())
    only_i = int((sig_i & ~sig_t).sum())
    only_t = int((~sig_i & sig_t).sum())
    m["significance_group"] = "BH q >= 0.05 in both analyses"
    m.loc[sig_i & ~sig_t, "significance_group"] = "Significant with imputed dosage only"
    m.loc[~sig_i & sig_t, "significance_group"] = "Significant with tagging SNP only"
    m.loc[sig_i & sig_t, "significance_group"] = "Significant in both analyses"
    b = m.loc[ok_or].copy()

    print(f"  -log10 p    : Spearman rho = {rho_p:.3f} (p = {p_rho:.3g}), "
          f"Pearson r = {r_p:.3f}")
    print(f"  log OR      : Spearman rho = {rho_b:.3f} (p = {p_rho_b:.3g}), "
          f"Pearson r = {r_b:.3f}, slope = {slope:.3f}, "
          f"same direction {100 * same_dir:.1f}%")
    print(f"  BH q < {ALPHA}: {both} in both, {only_i} imputed only, "
          f"{only_t} tag only")

    pd.DataFrame([
        dict(quantity="-log10 p", spearman_rho=rho_p, spearman_p=p_rho,
             pearson_r=r_p, n=len(m)),
        dict(quantity="log odds ratio", spearman_rho=rho_b, spearman_p=p_rho_b,
             pearson_r=r_b, n=int(ok_or.sum())),
        dict(quantity="effect slope (tag on imputed)", spearman_rho=np.nan,
             spearman_p=np.nan, pearson_r=slope, n=int(ok_or.sum())),
        dict(quantity="same direction of effect", spearman_rho=np.nan,
             spearman_p=np.nan, pearson_r=same_dir, n=int(ok_or.sum())),
        dict(quantity="BH q < 0.05 in both analyses", spearman_rho=np.nan,
             spearman_p=np.nan, pearson_r=both, n=len(m)),
    ]).to_csv(OUT_TSV, sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))

    significance_styles = [
        ("BH q >= 0.05 in both analyses", "#9AA9BF", 16, .55),
        ("Significant with imputed dosage only", "#3B5BA5", 24, .9),
        ("Significant with tagging SNP only", "#C2601F", 24, .9),
        ("Significant in both analyses", "#2A7360", 28, .95),
    ]

    ax = axes[0]
    lim = max(m["logp_imputed"].max(), m["logp_tag"].max()) * 1.06
    for group, color, size, alpha in significance_styles:
        selected = m["significance_group"] == group
        ax.scatter(
            m.loc[selected, "logp_imputed"],
            m.loc[selected, "logp_tag"],
            s=size,
            alpha=alpha,
            color=color,
            edgecolor="none",
            label=group,
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"{LOCUS} imputed inversion dosage association,  $-\\log_{{10}}p$")
    ax.set_ylabel(f"{LOCUS} tagging SNP association,  $-\\log_{{10}}p$")
    ax.set_title("A", loc="left", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:], labels[1:], frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    lo = min(b["beta_imputed"].min(), b["beta_tag"].min())
    hi = max(b["beta_imputed"].max(), b["beta_tag"].max())
    pad = .06 * (hi - lo)
    for group, color, size, alpha in significance_styles:
        selected = b["significance_group"] == group
        ax.scatter(
            b.loc[selected, "beta_imputed"],
            b.loc[selected, "beta_tag"],
            s=size,
            alpha=alpha,
            color=color,
            edgecolor="none",
        )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"Effect using {LOCUS} imputed inversion dosage,  $\\log$OR")
    ax.set_ylabel(f"Effect using {LOCUS} tagging SNP dosage,  $\\log$OR")
    ax.set_title("B", loc="left", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"\nWrote {OUT_TSV}\n      {OUT_PDF} / {OUT_PNG}")


if __name__ == "__main__":
    main()
