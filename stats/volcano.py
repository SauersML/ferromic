import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

INPUT_FILE = "phewas_results.tsv"
OUTPUT_PDF = "phewas_volcano.pdf"

# --------------------------- Appearance & sizing ---------------------------

plt.rcParams.update({
    "figure.figsize": (13, 8.5),            # big & readable
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
})

# --------------------------- Color utilities ---------------------------

def non_orange_colors(n, seed=21):
    """
    Generate n distinct colors while EXCLUDING orange-ish hues.
    We sample hues in HSV and skip an orange band (~20°–45°).

    Implementation details:
    - Allowed hue ranges (in [0,1]): [0, 0.055) U (0.125, 1.0)
    - We alternate saturation/value to improve differentiability when n is large.
    - Returns RGB tuples.
    """
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)

    gaps = [(0.0, 0.055), (0.125, 1.0)]
    total = sum(b - a for a, b in gaps)

    # A few (sat, val) combos to cycle through for separation
    sv = [(0.80, 0.85), (0.65, 0.90), (0.75, 0.70), (0.55, 0.80)]

    cols = []
    for i in range(n):
        t = (i + 0.5) / n * total
        for a, b in gaps:
            w = b - a
            if t <= w:
                h = a + t
                break
            t -= w
        s, v = sv[i % len(sv)]
        rgb = mcolors.hsv_to_rgb((h, s, v))
        cols.append(tuple(rgb))
    return cols

# --------------------------- Stats utilities ---------------------------

def bh_fdr_cutoff(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR cutoff: returns the *p* threshold (largest p-value declared significant).
    If no discoveries, returns np.nan.
    """
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    m = p.size
    if m == 0:
        return np.nan
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    crit = ranks / m * alpha
    ok = p_sorted <= crit
    if not np.any(ok):
        return np.nan
    return p_sorted[np.where(ok)[0].max()]

# --------------------------- Data IO & prep ---------------------------

def load_and_prepare(path):
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: '{path}' not found in current directory.")

    df = pd.read_csv(path, sep="\t", dtype=str)

    # Required columns
    needed = ["OR", "P_LRT_Overall"]
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"ERROR: Required column '{c}' is missing in {path}.")

    # Optional columns
    if "Inversion" not in df.columns:
        df["Inversion"] = "Unknown"
    if "Phenotype" not in df.columns:
        df["Phenotype"] = ""

    # Coerce types
    df["OR"] = pd.to_numeric(df["OR"], errors="coerce")
    df["P_LRT_Overall"] = pd.to_numeric(df["P_LRT_Overall"], errors="coerce")

    # Drop rows with missing or non-positive p-values
    df = df[np.isfinite(df["P_LRT_Overall"].to_numpy()) & (df["P_LRT_Overall"] > 0)].copy()

    # Axes transforms
    df["lnOR"] = np.log(df["OR"])                 # normalized OR (centered at 0)
    df["neglog10p"] = -np.log10(df["P_LRT_Overall"])

    # Keep only finite
    df = df[np.isfinite(df["lnOR"]) & np.isfinite(df["neglog10p"])].copy()


    # Clean Inversion
    df["Inversion"] = df["Inversion"].fillna("Unknown").astype(str)

    return df

# --------------------------- Plotting helpers ---------------------------

def make_or_ticks(xlim_ln):
    """
    Build human-readable x-ticks labeled in OR-space, placed in ln(OR)-space.
    We target nice multipliers around 1× on a symmetric grid.
    """
    # Candidate ORs (both sides): tweak as needed
    or_vals = np.array([0.25, 0.33, 0.5, 0.67, 0.83, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0])
    pos = np.log(or_vals)

    # Keep ticks within current xlim
    mask = (pos >= xlim_ln[0]) & (pos <= xlim_ln[1])
    pos = pos[mask]
    or_vals = or_vals[mask]

    # Pretty labels (× for fold-change; 1× at center)
    labels = []
    for v in or_vals:
        if np.isclose(v, 1.0):
            labels.append("1×")
        elif v < 1.0:
            labels.append(f"{v:.2g}×")   # e.g., 0.67×
        else:
            labels.append(f"{v:.2g}×")   # e.g., 1.5×
    return pos, labels

def assign_colors_and_markers(levels):
    """
    Assign distinct colors (no orange) PLUS a cycle of markers for extra separability.
    """
    n = len(levels)
    colors = non_orange_colors(n)
    # A marker cycle that avoids '^' (reserved for arrows), mixes filled/open shapes for contrast
    marker_cycle = ['o', 's', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', 'H', 'd']
    marker_map = {lvl: marker_cycle[i % len(marker_cycle)] for i, lvl in enumerate(levels)}
    color_map = {lvl: colors[i] for i, lvl in enumerate(levels)}
    return color_map, marker_map

# --------------------------- Main plotting ---------------------------

def plot_volcano(df, out_pdf):
    if df.empty:
        raise SystemExit("ERROR: No valid rows after cleaning; nothing to plot.")

    # Extreme handling: up-arrows for y > 300
    EXTREME_Y = 300.0
    df["is_extreme"] = df["neglog10p"] > EXTREME_Y

    if (~df["is_extreme"]).any():
        ymax_nonextreme = df.loc[~df["is_extreme"], "neglog10p"].max()
        arrow_y = ymax_nonextreme * 1.10  # 10% higher than highest non-extreme
        if not (np.isfinite(arrow_y) and arrow_y > 0):
            arrow_y = EXTREME_Y * 1.10
    else:
        # If all are extreme, still put them slightly above threshold
        arrow_y = EXTREME_Y * 1.10

    df["y_plot"] = np.where(df["is_extreme"], arrow_y, df["neglog10p"])

    # Color & marker by Inversion
    inv_levels = sorted(df["Inversion"].unique())
    color_map, marker_map = assign_colors_and_markers(inv_levels)

    # FDR line (BH 0.05)
    p_cut = bh_fdr_cutoff(df["P_LRT_Overall"].to_numpy(), alpha=0.05)
    y_fdr = -np.log10(p_cut) if (isinstance(p_cut, (int, float)) and p_cut > 0 and np.isfinite(p_cut)) else np.nan

    # Smart x-limits (symmetric in lnOR), using a high percentile to avoid extreme domination
    xabs = np.abs(df["lnOR"].to_numpy())
    if xabs.size == 0:
        xmax = 1.0
    else:
        x99 = np.nanpercentile(xabs, 99.5)
        xmax = max(0.5, float(x99))
        xmax = min(max(xabs.max(), 0.5), xmax * 1.2)  # keep sane but inclusive
    xlim = (-xmax, xmax)

    # Prepare figure
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.78)  # room for legend

    # Use symlog to: (a) open up the middle and (b) compress far tails
    # linthresh controls the width of the central linear region around 0
    ax.set_xscale('symlog', linthresh=2, linscale=1.25, base=10)
    ax.set_xlim(xlim)

    # y-limits with a bit of headroom
    ymax = max(df["y_plot"].max(), y_fdr if np.isfinite(y_fdr) else 0.0)
    ax.set_ylim(0, ymax * 1.06 if ymax > 0 else 10)

    # Grid & baseline
    ax.grid(alpha=0.3, linewidth=0.7)
    ax.axvline(0.0, color='k', linewidth=1.0)

    # FDR line (if defined)
    if np.isfinite(y_fdr):
        ax.axhline(y_fdr, linestyle="--", linewidth=1.2)
        ax.text(0.99, (y_fdr - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "BH FDR 0.05", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # Order for nicer layering (least significant first)
    df = df.sort_values("y_plot", ascending=True)

    # Draw points per inversion
    N = df.shape[0]
    rasterize = N > 60000

    for inv in inv_levels:
        sub = df[df["Inversion"] == inv]
        # Normal points
        norm = sub[~sub["is_extreme"]]
        if not norm.empty:
            ax.scatter(
                norm["lnOR"].to_numpy(),
                norm["y_plot"].to_numpy(),
                s=22, alpha=0.75,
                marker=marker_map[inv],
                facecolor=color_map[inv],
                edgecolor="black",
                linewidth=0.3,
                rasterized=rasterize,
            )
        # Extreme points → UP ARROW (not triangle)
        ext = sub[sub["is_extreme"]]
        if not ext.empty:
            ax.scatter(
                ext["lnOR"].to_numpy(),
                ext["y_plot"].to_numpy(),
                s=90, alpha=0.95,
                marker=r'$\uparrow$',
                facecolor=color_map[inv],
                edgecolor="black",
                linewidth=0.4,
                rasterized=rasterize,
            )

    # Axis labels & title
    ax.set_ylabel(r"$-\log_{10}(\mathrm{p})$")
    ax.set_xlabel("")
    ax.set_title("")

    # Human-readable OR ticks (labels in “×” space, positioned in ln(OR) space)
    xticks, xlabels = make_or_ticks(ax.get_xlim())
    if len(xticks) >= 3:  # avoid too sparse
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

    # Legend (color+marker for Inversion) on the right, multi-column if many
    handles = []
    for inv in inv_levels:
        handles.append(
            Line2D([], [], linestyle='None',
                   marker=marker_map[inv], markersize=9,
                   markerfacecolor=color_map[inv], markeredgecolor="black", markeredgewidth=0.6,
                   label=str(inv))
        )

    n_inv = len(inv_levels)
    if n_inv <= 18:
        ncols = 1
    elif n_inv <= 40:
        ncols = 2
    elif n_inv <= 70:
        ncols = 3
    else:
        ncols = 4

    fig.legend(
        handles=handles,
        title="Inversion",
        loc="center left",
        bbox_to_anchor=(0.805, 0.5),
        frameon=False,
        ncol=ncols,
        borderaxespad=0.0,
        handlelength=1.2,
        columnspacing=1.0,
        labelspacing=0.6,
    )

    # Save
    with PdfPages(OUTPUT_PDF) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PDF}")

# --------------------------- Entrypoint ---------------------------

def main():
    df = load_and_prepare(INPUT_FILE)
    plot_volcano(df, OUTPUT_PDF)

if __name__ == "__main__":
    main()
