import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import ticker

# Ensure text remains editable in PDFs (embed TrueType fonts)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ---------- Load & prepare data ----------
imp = pd.read_csv("imputation_results.tsv", sep="\t", dtype=str)
inv = pd.read_csv("inv_info.tsv", sep="\t", dtype=str)

# Trim whitespace in headers & values
imp.columns = imp.columns.str.strip()
inv.columns = inv.columns.str.strip()

required_imp_cols = {"id", "unbiased_pearson_r2"}
missing_imp = required_imp_cols.difference(imp.columns)
if missing_imp:
    raise RuntimeError(f"Missing columns in imputation_results.tsv: {missing_imp}")

# Ensure we have an OrigID in inv_info (or derive from 'OrigIDSize_.kbp.')
if "OrigID" not in inv.columns:
    if "OrigIDSize_.kbp." in inv.columns:
        inv["OrigID"] = inv["OrigIDSize_.kbp."].astype(str).str.strip().str.split().str[0]
    else:
        raise RuntimeError("Neither 'OrigID' nor 'OrigIDSize_.kbp.' found in inv_info.tsv")

cons_col = "0_single_1_recur_consensus"
for col in ["Chromosome", "Start", "End", cons_col]:
    if col not in inv.columns:
        raise RuntimeError(f"Column '{col}' not found in inv_info.tsv")

# Normalize types for merge
imp["id"] = imp["id"].astype(str).str.strip()
inv["OrigID"] = inv["OrigID"].astype(str).str.strip()

# Merge id ↔ OrigID
merged = (
    imp.merge(
        inv[["OrigID", "Chromosome", "Start", "End", cons_col]],
        left_on="id",
        right_on="OrigID",
        how="inner",
    ).drop(columns=["OrigID"])
)

# Convert to numeric
merged["unbiased_pearson_r2"] = pd.to_numeric(merged["unbiased_pearson_r2"], errors="coerce")
merged[cons_col] = pd.to_numeric(merged[cons_col], errors="coerce")
merged["Start"] = pd.to_numeric(merged["Start"], errors="coerce")
merged["End"] = pd.to_numeric(merged["End"], errors="coerce")

# Filter: only consensus exactly 0 or 1
plot_df = merged[merged[cons_col].isin([0, 1])].copy()
plot_df = plot_df.dropna(subset=["unbiased_pearson_r2", "Chromosome", "Start", "End"])

# Compute r from r^2
plot_df["r"] = np.sqrt(np.clip(plot_df["unbiased_pearson_r2"].values, 0, None))

# Build chr:start-end label with thousands separators
def fmt_coord(row):
    chrom = str(row["Chromosome"]).strip()
    s = int(row["Start"])
    e = int(row["End"])
    return f"{chrom}:{s:,}-{e:,}"

plot_df["label"] = plot_df.apply(fmt_coord, axis=1)

# Sort by r (descending)
plot_df = plot_df.sort_values("r", ascending=False).reset_index(drop=True)
if plot_df.empty:
    raise RuntimeError("No inversions with consensus 0/1 were found after filtering and merging.")

# ---------- Plot styling ----------
# Colors (white background)
BLUE_DARK  = "#1e4fb3"   # single-event
BLUE_LIGHT = "#6fb6ff"   # recurrent

# Positions and values
x = np.arange(len(plot_df))
y = plot_df["r"].values
labels = plot_df["label"].tolist()
consensus = plot_df[cons_col].values  # 0 = single-event, 1 = recurrent
mask_single = (consensus == 0)
mask_recur  = (consensus == 1)

# Figure size scales with number of bars
fig_w = max(14.0, min(36.0, 0.40 * len(plot_df) + 8.0))
fig_h = 9.0

# Bigger fonts overall
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 22,      # y-axis label will be large
    "xtick.labelsize": 14,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,     # BIG legend text
})

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# NO gridlines
ax.grid(False)

# Draw bars
barw = 0.82

# SINGLE-EVENT: dark blue + diagonal stripes
if mask_single.any():
    ax.bar(
        x[mask_single],
        y[mask_single],
        width=barw,
        color=BLUE_DARK,
        edgecolor="black",     # hatch color uses edgecolor
        linewidth=0.7,
        hatch="//",
        label="Single-event",
    )

# RECURRENT: lighter blue + white dots overlay
if mask_recur.any():
    # base
    ax.bar(
        x[mask_recur],
        y[mask_recur],
        width=barw,
        color=BLUE_LIGHT,
        edgecolor=BLUE_LIGHT,
        linewidth=0.6,
        label="Recurrent",
    )
    # dots overlay
    ax.bar(
        x[mask_recur],
        y[mask_recur],
        width=barw,
        color=(1, 1, 1, 0),    # transparent face
        edgecolor="white",     # hatch color
        linewidth=0.8,
        hatch="..",
    )

# Labels
ax.set_ylabel("r", labelpad=14, color="black")

# X-axis label — WAY BIGGER
ax.set_xlabel("Inversion (sorted by r)", labelpad=16, color="black", fontsize=26)

# NO title
ax.set_title("")

# X tick labels: tilted/skewed
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right")
for lbl in ax.get_xticklabels():
    lbl.set_color("black")
    lbl.set_fontsize(14)
    lbl.set_style("italic")   # subtle skew

# Y ticks
ax.tick_params(axis="y", colors="black")
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

# Spines for a clean look
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(0.8)

# Legend ("key") TOP RIGHT and WAY BIGGER
legend_handles = [
    Patch(facecolor=BLUE_DARK, edgecolor="black", hatch="//", label="Single-event"),
    Patch(facecolor=BLUE_LIGHT, edgecolor="white", hatch="..", label="Recurrent"),
]
leg = ax.legend(
    handles=legend_handles,
    loc="upper right",
    frameon=True,
    borderpad=0.6,
    labelspacing=0.6,
    handlelength=2.0,
    handleheight=1.2,
)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("black")

# Extra margins so nothing gets cut off
plt.margins(x=0.02, y=0.06)
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.22)

# Save as vectorized PDF with editable text
out_pdf = "inversion_r_plot.pdf"
plt.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.6, transparent=False)
