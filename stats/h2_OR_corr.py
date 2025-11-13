import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 22

phewas_url = (
    "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/"
    "phewas_results.tsv"
)
h2_url = (
    "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/"
    "significant_heritability_diseases.tsv"
)

phewas = pd.read_csv(phewas_url, sep="\t", low_memory=False)
h2 = pd.read_csv(h2_url, sep="\t", low_memory=False)

h2["h2_overall_REML"] = pd.to_numeric(h2["h2_overall_REML"], errors="coerce")


def canon_name(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    return "".join(ch for ch in s if ch.isalnum())


phewas["canon"] = phewas["Phenotype"].map(canon_name)
h2["canon"] = h2["disease"].map(canon_name)

h2_grouped = (
    h2.dropna(subset=["canon"])
      .groupby("canon", as_index=False)["h2_overall_REML"]
      .max()
      .rename(columns={"h2_overall_REML": "h2"})
)

phewas["OR"] = pd.to_numeric(phewas["OR"], errors="coerce")

merged = phewas.merge(h2_grouped, on="canon", how="inner")
merged = merged.dropna(subset=["h2", "OR"])
merged = merged[merged["OR"] > 0]

groups = merged.groupby("canon")

rows = []
for canon, g in groups:
    h2_val = float(g["h2"].iloc[0])
    ors = g["OR"].to_numpy(dtype=float)

    max_or = float(np.max(ors))
    max_abs_1_minus_or = float(np.max(np.abs(1.0 - ors)))

    rows.append(
        {
            "canon": canon,
            "h2": h2_val,
            "max_or": max_or,
            "max_abs_1_minus_or": max_abs_1_minus_or,
        }
    )

summary = pd.DataFrame(rows)

metrics = ["max_or", "max_abs_1_minus_or"]
spearman_results = {}
for m in metrics:
    mask = summary[m].notna() & summary["h2"].notna()
    rho, pval = spearmanr(summary.loc[mask, "h2"], summary.loc[mask, m])
    spearman_results[m] = (rho, pval)

fig, (axA, axB) = plt.subplots(2, 1, figsize=(18, 10))

scatter_kwargs = dict(
    alpha=0.2,
    color="blue",
    edgecolors="none",
    s=120,
)

x_label = "Phenotype heritability (h\u00b2, SNP-REML)"


def plot_panel(ax, x, y, panel_label, y_label, rho, pval):
    mask = np.isfinite(x) & np.isfinite(y)
    x_plot = x[mask]
    y_plot = y[mask]

    ax.scatter(x_plot, y_plot, **scatter_kwargs)
    ax.set_xlabel(x_label, fontweight="normal")
    ax.set_ylabel(y_label, fontweight="normal")

    ax.text(
        0.02,
        0.98,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.text(
        0.98,
        0.98,
        f"Spearman \u03c1 = {rho:.3f}\n p = {pval:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=18,
    )


rho, pval = spearman_results["max_or"]
plot_panel(
    axA,
    summary["h2"].to_numpy(),
    summary["max_or"].to_numpy(),
    "A",
    "Maximum odds ratio",
    rho,
    pval,
)

rho, pval = spearman_results["max_abs_1_minus_or"]
plot_panel(
    axB,
    summary["h2"].to_numpy(),
    summary["max_abs_1_minus_or"].to_numpy(),
    "B",
    "Maximum |1 \u2212 OR|",
    rho,
    pval,
)

fig.tight_layout()
fig.subplots_adjust(hspace=0.45)

fig.savefig("heritability_inversion_OR_relationships.png", dpi=300)
