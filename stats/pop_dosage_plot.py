import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# Change this to download from GitHub: https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/inversion_population_frequencies.tsv
tsv_path = "inversion_population_frequencies.tsv"

# Load table
df = pd.read_csv(tsv_path, sep="\t")

# Case-insensitive column lookup
lower_to_actual = {c.lower(): c for c in df.columns}

def get_col(possible_names):
    for name in possible_names:
        key = name.lower()
        if key in lower_to_actual:
            return lower_to_actual[key]
    raise KeyError(f"None of {possible_names} found in columns: {list(df.columns)}")

# Identify needed columns (names are flexible)
pop_col = get_col(["Population", "population", "Pop"])
inv_col = get_col(["Inversion", "inversion", "Inv"])
mean_col = get_col(["Mean_dosage", "Mean_Dosage", "mean_dosage"])
n_col = get_col(["N", "n", "count"])
std_col = get_col(["Dosage_STD_All", "Dosage_STD", "dosage_std_all", "dosage_std"])

# Normalize column names
df = df.rename(columns={
    pop_col: "Population",
    inv_col: "Inversion",
    mean_col: "Mean_dosage",
    n_col: "N",
    std_col: "Dosage_STD_All",
})

# Map population codes to display labels
pop_display_map = {
    "ALL": "Overall",
    "afr": "AFR",
    "amr": "AMR",
    "eas": "EAS",
    "eur": "EUR",
    "mid": "MID",
    "sas": "SAS",
}

df["Population_display"] = df["Population"].map(pop_display_map).fillna(df["Population"])

# Keep rows with valid dosage summary stats (and ignore any allele-frequency CI columns)
df = df[(df["N"] > 1) & df["Dosage_STD_All"].notna()].copy()

# Compute standard error and 95% CI for dosage using the t distribution
df["SE"] = df["Dosage_STD_All"] / np.sqrt(df["N"])
df["t_crit"] = t.ppf(0.975, df["N"] - 1)
df["margin"] = df["t_crit"] * df["SE"]
df["CI_lower"] = df["Mean_dosage"] - df["margin"]
df["CI_upper"] = df["Mean_dosage"] + df["margin"]

# Order inversions (x-axis) and populations (grouping)
inversions = np.sort(df["Inversion"].unique())

pop_order = ["Overall", "AFR", "AMR", "EAS", "EUR", "MID", "SAS"]
pop_order = [p for p in pop_order if p in df["Population_display"].unique()]

# Color palette without pink/purple
color_map = {
    "Overall": "#1f77b4",  # blue
    "AFR": "#ff7f0e",      # orange
    "AMR": "#2ca02c",      # green
    "EAS": "#bcbd22",      # olive
    "EUR": "#17becf",      # cyan
    "MID": "#8c564b",      # brown
    "SAS": "#d62728",      # red
}

# Global typography
plt.rcParams.update({
    "axes.labelsize": 22,
    "xtick.labelsize": 14,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
})

num_inv = len(inversions)
fig_width = max(18, num_inv * 0.6)
fig, ax = plt.subplots(figsize=(fig_width, 9))

# Horizontal positions for inversion groups and population offsets
x_base = np.arange(num_inv)
group_width = 0.7
offset_step = group_width / max(len(pop_order), 1)
offset_start = -group_width / 2 + offset_step / 2

# Plot mean + 95% CI for each population, with large semi-transparent points
for idx, pop in enumerate(pop_order):
    sub = df[df["Population_display"] == pop].set_index("Inversion")

    means = sub.reindex(inversions)["Mean_dosage"]
    lower = sub.reindex(inversions)["CI_lower"]
    upper = sub.reindex(inversions)["CI_upper"]

    yerr = np.vstack((means - lower, upper - means))
    x_positions = x_base + offset_start + idx * offset_step

    ax.errorbar(
        x_positions,
        means,
        yerr=yerr,
        fmt="o",
        markersize=10,
        elinewidth=1.6,
        capsize=4,
        color=color_map.get(pop, "#333333"),
        label=pop,
        linestyle="none",
        alpha=0.5,
    )

# Axes styling
ax.set_xlabel("")  # no x-axis title
ax.set_ylabel("Mean dosage")
ax.grid(False)

ax.set_xticks(x_base)
ax.set_xticklabels(inversions, rotation=60, ha="right")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Spacing and legend (no legend title)
fig.subplots_adjust(right=0.82, bottom=0.3)
ax.legend(
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.0,
)

plt.show()
