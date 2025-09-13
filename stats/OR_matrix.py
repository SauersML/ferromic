from pathlib import Path
import math
import textwrap
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# =========================
# Global configuration
# =========================
INPUT_PATH = "phewas_results.tsv"
OUT_PREFIX = "phewas_heatmap"

# Significance thresholds
P_THRESHOLD = 0.05
Q_THRESHOLD = 0.05

# Colormap & scaling
COLORMAP = "RdBu_r"
PERCENTILE_CAP = 99.5          # clip color range to this percentile of |ln(OR)|

# Figure sizing (inches)
CELL = 0.55                     # size per cell; keep squares feeling roomy
EXTRA_W = 5.2                   # room for y labels, colorbar, legend
EXTRA_H = 2.2                   # base room for title (x-label space is added dynamically)
MIN_W, MAX_W = 12.0, 80.0
MIN_H, MAX_H = 7.0, 60.0

# Y-axis label density (to avoid overlap there)
MAX_YLABELS = 80

# X-axis label styling & layout (staggered, non-overlapping)
X_LABEL_FONTSIZE = 9
WRAP_CHARS = 22                 # wrap long phenotype labels for narrower width
LABEL_BASE_DY_PT = 8            # first tier: points below the axis baseline
LABEL_TIER_STEP_PT = 10         # additional points per tier
LABEL_MAX_TIERS = 10            # up to this many tiers of labels
LABEL_PADDING_PX = 3            # horizontal padding when testing for overlaps
LEADER_LW = 0.5                 # leader (connector) line width
LEADER_COLOR = "black"

# Cell separation (natural whitespace between squares)
DRAW_CELL_EDGES = True
EDGE_LW = 0.7                   # thin white edges become "gaps"
EDGE_COLOR = "white"

# Outline styling (thin, crisp)
DASH_PATTERN = (0, (1.0, 1.0))  # short dash/short gap (looks dotted)
LW_DASHED = 0.5                 # thin dashed outline (nominal p)
LW_SOLID  = 0.9                 # slightly thicker solid outline (BH q)

# Fonts / rc
mpl.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,   # keep text editable in vector editors
    "ps.fonttype": 42,
})


# =========================
# Helpers
# =========================
def validate_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_figsize(n_rows: int, n_cols: int) -> Tuple[float, float]:
    """Choose a size that keeps cells looking like squares and avoids crowding."""
    w = clamp(n_cols * CELL + EXTRA_W, MIN_W, MAX_W)
    h = clamp(n_rows * CELL + EXTRA_H, MIN_H, MAX_H)
    return (w, h)


def sparse_y_ticks(n: int, max_labels: int) -> np.ndarray:
    """Return y tick indices to keep so that ≤ max_labels are shown."""
    if n <= max_labels:
        return np.arange(n)
    step = int(math.ceil(n / max_labels))
    return np.arange(0, n, step)


def _measure_one_line_width_px(fig, text: str, fontsize: float) -> float:
    """Measure width of a single-line string in display pixels."""
    renderer = fig.canvas.get_renderer()
    t = mpl.text.Text(x=0, y=0, text=text, fontsize=fontsize)
    t.set_figure(fig)  # attach figure so get_window_extent can access dpi
    bbox = t.get_window_extent(renderer=renderer)
    return bbox.width


def measure_label_width_px(fig, label: str, fontsize: float) -> float:
    """
    Measure width of a (possibly multi-line) label in pixels as
    the max width across lines.
    """
    # Ensure renderer exists
    fig.canvas.draw()
    lines = str(label).split("\n")
    if not lines:
        return 0.0
    return max(_measure_one_line_width_px(fig, ln, fontsize) for ln in lines)


def compute_staggered_layout(fig, ax, labels: List[str], x_centers: np.ndarray) -> Tuple[List[Tuple[float,int,float]], float]:
    """
    Compute a non-overlapping, staggered placement for x-axis labels.

    Returns:
      placements: list of tuples (x_center_data, tier_index, dy_points)
      max_dy_pt: maximum dy in points used (for margin calculation)
    """
    # Ensure renderer ready
    fig.canvas.draw()

    # Transform to display pixels for x positions of column centers
    x_disp = ax.transData.transform(np.column_stack([x_centers, np.zeros_like(x_centers)]))[:, 0]

    # Prepare tiers: list of occupied intervals in pixels per tier
    tiers: List[List[Tuple[float, float]]] = [[] for _ in range(LABEL_MAX_TIERS)]
    placements: List[Tuple[float, int, float]] = []
    max_dy_pt = 0.0

    # Measure each label's width (max line width) in pixels
    widths_px = [measure_label_width_px(fig, lab, X_LABEL_FONTSIZE) for lab in labels]

    for j, (xc_px, w_px) in enumerate(zip(x_disp, widths_px)):
        half = 0.5 * w_px + LABEL_PADDING_PX
        interval = (xc_px - half, xc_px + half)

        placed = False
        for tier_idx in range(LABEL_MAX_TIERS):
            occupied = tiers[tier_idx]
            # Check overlap with any existing interval in this tier
            overlaps = any(not (interval[1] < a or interval[0] > b) for (a, b) in occupied)
            if not overlaps:
                occupied.append(interval)
                dy_pt = LABEL_BASE_DY_PT + tier_idx * LABEL_TIER_STEP_PT
                placements.append((x_centers[j], tier_idx, dy_pt))
                max_dy_pt = max(max_dy_pt, dy_pt)
                placed = True
                break

        if not placed:
            # Densest fallback: place on last tier anyway
            tier_idx = LABEL_MAX_TIERS - 1
            tiers[tier_idx].append(interval)
            dy_pt = LABEL_BASE_DY_PT + tier_idx * LABEL_TIER_STEP_PT
            placements.append((x_centers[j], tier_idx, dy_pt))
            max_dy_pt = max(max_dy_pt, dy_pt)

    return placements, max_dy_pt


def add_staggered_xlabels(fig, ax, labels: List[str], x_centers: np.ndarray):
    """
    Add staggered, non-overlapping x labels below the axis with leader lines.
    Assumes layout (subplots_adjust) is already finalized.
    """
    placements, _ = compute_staggered_layout(fig, ax, labels, x_centers)

    # Draw labels + leaders
    for (x, tier, dy_pt), text in zip(placements, labels):
        ann = ax.annotate(
            text,
            xy=(x, 0), xycoords=ax.get_xaxis_transform(),  # start at axis baseline
            xytext=(0, -dy_pt), textcoords="offset points",
            ha="center", va="top",
            fontsize=X_LABEL_FONTSIZE,
            arrowprops=dict(
                arrowstyle="-",
                lw=LEADER_LW,
                color=LEADER_COLOR,
                shrinkA=0, shrinkB=2,
                mutation_scale=1.0,
            ),
            clip_on=False,
        )
        ann.set_multialignment("center")


def main():
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path, sep="\t", dtype=str)
    validate_columns(df, ["Inversion", "Phenotype", "OR", "P_Value", "BH_q"])

    # Coerce numerics, clean
    for c in ["OR", "P_Value", "BH_q"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Inversion", "Phenotype", "OR"])
    df = df[df["OR"] > 0]

    # Normed OR: ln(OR), symmetric around 0
    df["normed_OR"] = np.log(df["OR"].values)

    # If duplicates, keep the "best": lowest BH_q, then lowest P, then largest |effect|
    df["_abs_effect"] = df["normed_OR"].abs()
    df["_BH_q_sort"] = df["BH_q"].fillna(np.inf)
    df["_P_sort"] = df["P_Value"].fillna(np.inf)
    df = df.sort_values(
        by=["_BH_q_sort", "_P_sort", "_abs_effect"],
        ascending=[True, True, False]
    ).drop_duplicates(subset=["Inversion", "Phenotype"], keep="first")

    # Sort columns (phenotypes) by *mean absolute* effect across inversions
    col_strength = df.groupby("Phenotype")["normed_OR"].apply(lambda s: np.nanmean(np.abs(s)))
    col_order = col_strength.sort_values(ascending=False).index.tolist()

    # Preserve row (inversion) order as first-seen in file
    row_order = pd.unique(df["Inversion"].values).tolist()

    # Build matrices
    pv = df.pivot(index="Inversion", columns="Phenotype", values="normed_OR").reindex(index=row_order, columns=col_order)
    pmat = df.pivot(index="Inversion", columns="Phenotype", values="P_Value").reindex(index=row_order, columns=col_order)
    qmat = df.pivot(index="Inversion", columns="Phenotype", values="BH_q").reindex(index=row_order, columns=col_order)

    # Labels (phenotypes: underscores → spaces)
    raw_col_labels = [str(c) for c in pv.columns]
    col_labels = [textwrap.fill(lbl.replace("_", " "), width=WRAP_CHARS) for lbl in raw_col_labels]
    row_labels = list(pv.index)

    data = pv.values
    n_rows, n_cols = data.shape

    # Color limits (symmetric about 0), robust to outliers
    finite_abs = np.abs(data[np.isfinite(data)])
    vmax = np.nanpercentile(finite_abs, PERCENTILE_CAP) if finite_abs.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    # Figure
    fig_w, fig_h = compute_figsize(n_rows, n_cols)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)

    # Heatmap (vectorized): pcolormesh over a unit grid
    x = np.arange(n_cols + 1)
    y = np.arange(n_rows + 1)
    masked = np.ma.masked_invalid(data)

    # Colormap (Matplotlib 3.7+)
    cmap = mpl.colormaps.get_cmap(COLORMAP)
    # Make a version with explicit 'bad' color for NaNs
    cmap = cmap.with_extremes(bad="#D9D9D9")

    edgecolors = EDGE_COLOR if DRAW_CELL_EDGES else "face"
    linewidth = EDGE_LW if DRAW_CELL_EDGES else 0.0

    mesh = ax.pcolormesh(
        x, y, masked,
        cmap=cmap, vmin=-vmax, vmax=vmax,
        edgecolors=edgecolors, linewidth=linewidth, shading="flat"
    )
    ax.invert_yaxis()  # top row at top

    # Axis labels & title
    ax.set_xlabel("Phenotype (phecode)")
    ax.set_ylabel("Inversion")
    ax.set_title("Inversion–Phenotype Associations (normed OR = ln(OR))", pad=12)

    # Ticks: centers of cells for both axes
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)

    # Y labels: sparsify to avoid overlap (phenotype labels handled via custom layout)
    y_keep = sparse_y_ticks(n_rows, MAX_YLABELS)
    y_ticklabels = [row_labels[i] if i in set(y_keep) else "" for i in range(n_rows)]
    ax.set_yticklabels(y_ticklabels)

    # Hide default x tick labels; we'll add staggered labels with leaders
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)

    # Aesthetics
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normed OR (ln scale)\nnegative: decreased risk · positive: increased risk")

    # Significance outlines (draw after heatmap for visibility)
    row_lookup = {inv: i for i, inv in enumerate(pv.index)}
    col_lookup = {ph: j for j, ph in enumerate(pv.columns)}

    def outline_cell(i, j, kind):
        xy = (j, i)  # bottom-left of cell
        if kind == "q":  # strong (solid, slightly thicker)
            rect = Rectangle(
                xy, 1, 1, fill=False, lw=LW_SOLID, linestyle="solid", edgecolor="black"
            )
        elif kind == "p":  # nominal (short-dashed, thin)
            rect = Rectangle(
                xy, 1, 1, fill=False, lw=LW_DASHED, linestyle=DASH_PATTERN, edgecolor="black"
            )
        else:
            return
        ax.add_patch(rect)

    # Iterate rows once (df is deduplicated on (Inversion, Phenotype))
    for inv, ph, pval, qval in zip(df["Inversion"], df["Phenotype"], df["P_Value"], df["BH_q"]):
        i = row_lookup.get(inv, None)
        j = col_lookup.get(ph, None)
        if i is None or j is None:
            continue
        if pd.notna(qval) and qval < Q_THRESHOLD:
            outline_cell(i, j, "q")
        elif pd.notna(pval) and pval < P_THRESHOLD:
            outline_cell(i, j, "p")

    # Legend for outlines
    legend_elems = [
        Line2D([0], [0], color="black", lw=LW_SOLID, linestyle="solid", label=f"BH q < {Q_THRESHOLD:g}"),
        Line2D([0], [0], color="black", lw=LW_DASHED, linestyle=DASH_PATTERN, label=f"p < {P_THRESHOLD:g}"),
    ]
    ax.legend(
        handles=legend_elems,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="Significance"
    )

    # ---------- Custom, staggered x labels with leaders ----------
    # First pass: modest bottom margin so we can measure/render
    base_bottom = 0.16
    fig.subplots_adjust(left=0.16, right=0.86, bottom=base_bottom, top=0.90)

    # Column centers in data coords
    x_centers = np.arange(n_cols) + 0.5

    # Compute layout (tiers & offsets) and the max vertical extent needed
    _, max_dy_pt = compute_staggered_layout(fig, ax, col_labels, x_centers)

    # Convert needed label space (points) to figure fraction and update bottom margin
    extra_bottom_in = (max_dy_pt + 12) / 72.0   # add a small cushion (12pt)
    extra_frac = extra_bottom_in / fig.get_size_inches()[1]
    new_bottom = base_bottom + extra_frac
    new_bottom = clamp(new_bottom, 0.16, 0.50)  # cap so we don't overgrow
    fig.subplots_adjust(bottom=new_bottom)

    # Draw labels + leaders in final layout
    add_staggered_xlabels(fig, ax, col_labels, x_centers)

    # Ensure vector output (no rasterization of the QuadMesh)
    mesh.set_rasterized(False)

    out_svg = f"{OUT_PREFIX}.svg"
    out_pdf = f"{OUT_PREFIX}.pdf"
    fig.savefig(out_svg)
    fig.savefig(out_pdf)

    print(f"Saved: {out_svg}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
