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

# OR-Tools CP-SAT
from ortools.sat.python import cp_model

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
EXTRA_H = 2.2                   # base room for title (x-label space computed later)
MIN_W, MAX_W = 12.0, 80.0
MIN_H, MAX_H = 7.0, 60.0

# Y-axis label density (to avoid overlap there)
MAX_YLABELS = 80

# X-axis labels: text + routing setup (CP-SAT)
X_LABEL_FONTSIZE = 9
WRAP_CHARS = 22                 # wrap phenotype names to normalize widths

# Leader routing geometry (in pixels; converted to axes-fraction for drawing)
ROUTE_OFFSET_PX = 10            # vertical distance from axis baseline to the horizontal routing line
LABEL_BOX_PAD_PX = 4            # extra horizontal padding added to measured label widths (visual breathing room)
LABEL_GAP_MIN_PX = 6            # minimum horizontal gap between adjacent labels on the SAME tier
EPS_ORDER_PX = 1                # minimal strictly-increasing separation for X order
TIER_COUNT = 10                 # maximum tiers available (solver will minimize usage)

# Solver objective weights
W_ANCHOR = 1                    # weight for sum of |X - anchor|
W_TIERS  = 1000                 # weight for max tier used (keeps labels close to axis)
W_SPREAD = 1                    # weight to encourage larger min-gap between neighbors

# CP-SAT parameters
SOLVER_TIME_LIMIT_SEC = 15.0

# Cell separation (natural whitespace between squares)
DRAW_CELL_EDGES = True
EDGE_LW = 0.7                   # thin white edges become "gaps"
EDGE_COLOR = "white"

# Outline styling (thin, crisp)
DASH_PATTERN = (0, (1.0, 1.0))  # short dash/short gap (looks dotted)
LW_DASHED = 0.5                 # thin dashed outline (nominal p)
LW_SOLID  = 0.9                 # slightly thicker solid outline (BH q)

# Leader styling
LEADER_LW = 0.5
LEADER_COLOR = "black"

# Label text box (optional subtle bounding box; set facecolor=None for no box)
LABEL_BBOX = dict(facecolor=None, edgecolor=None, boxstyle=None, pad=0.0)

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


def measure_text_bbox_px(fig, text: str, fontsize: float) -> Tuple[float, float]:
    """
    Measure multi-line text bounding box (width, height) in display pixels.
    """
    fig.canvas.draw()
    t = mpl.text.Text(x=0, y=0, text=text, fontsize=fontsize, multialignment="center")
    t.set_figure(fig)
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return bbox.width, bbox.height


def data_x_to_px(ax, x_data: float) -> float:
    """Map data x in [0..n_cols] to display pixel x using axes window extent."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    x0, w = bbox.x0, bbox.width
    # Data x range is [0, n_cols]; we assume axis limits are set accordingly
    x_px = x0 + (x_data / (ax.get_xlim()[1] - ax.get_xlim()[0])) * w
    return x_px


def px_to_axes_frac(ax, x_px: float, y_px: float) -> Tuple[float, float]:
    """Convert display pixel coords to axes-fraction coords (0..1 inside, can be <0 below)."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    xf = (x_px - bbox.x0) / bbox.width
    yf = (y_px - bbox.y0) / bbox.height
    return xf, yf


def axes_frac_x_from_px(ax, x_px: float) -> float:
    """Convert display pixel X to axes-fraction X (0..1 inside)."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    return (x_px - bbox.x0) / bbox.width


def get_axes_bbox_px(ax):
    """Return axes bbox (x0, y0, width, height) in display pixels."""
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    return bbox.x0, bbox.y0, bbox.width, bbox.height


# =========================
# CP-SAT label layout
# =========================
def solve_label_layout_cp_sat(
    anchors_px: List[int],
    widths_px: List[int],
    tier_count: int,
    x_min_px: int,
    x_max_px: int,
    min_gap_px: int,
    eps_order_px: int,
    w_anchor: int,
    w_tiers: int,
    w_spread: int,
    time_limit_sec: float,
):
    """
    CP-SAT model:
      Variables:
        X_j ∈ [x_min_px .. x_max_px] (int), strictly increasing
        t_j ∈ {0..tier_count-1} (int)
        d_j = |X_j - anchors_px[j]|
        s_j ∈ {0,1} indicates (t_j == t_{j+1})
        δ ≥ 0 min-gap across same-tier neighbors
        M_t ≥ 0 max tier used

      Constraints:
        Order preserving: X_{j+1} - X_j ≥ eps_order_px
        Same-tier channeling: (t_j == t_{j+1}) ↔ s_j
        Same-tier nonoverlap (adjacent): (X_{j+1} - X_j) ≥ (W_j+W_{j+1})/2 + min_gap_px, enforced if s_j
        Bound X_j (left/right), plus half-width margins are absorbed by widths_px in the inequality
        t_j ≤ M_t

      Objective:
        Minimize   w_anchor * Σ d_j + w_tiers * M_t  − w_spread * δ
    """
    n = len(anchors_px)
    model = cp_model.CpModel()

    # Variables
    X = [model.NewIntVar(x_min_px, x_max_px, f"X_{j}") for j in range(n)]
    T = [model.NewIntVar(0, tier_count - 1, f"T_{j}") for j in range(n)]
    D = [model.NewIntVar(0, x_max_px - x_min_px, f"D_{j}") for j in range(n)]
    S = [model.NewBoolVar(f"S_{j}") for j in range(n - 1)]  # same tier flags for adjacent pairs
    delta = model.NewIntVar(0, x_max_px - x_min_px, "delta")
    M_t = model.NewIntVar(0, tier_count - 1, "M_t")

    # |X - anchor|
    for j in range(n):
        model.AddAbsEquality(D[j], X[j] - anchors_px[j])

    # Order preserving
    for j in range(n - 1):
        model.Add(X[j + 1] - X[j] >= eps_order_px)

    # Channel S_j ↔ (T_j == T_{j+1})
    for j in range(n - 1):
        model.Add(T[j] == T[j + 1]).OnlyEnforceIf(S[j])
        model.Add(T[j] != T[j + 1]).OnlyEnforceIf(S[j].Not())

    # Same-tier nonoverlap for adjacent neighbors
    # X_{j+1} - X_j >= (W_j + W_{j+1})/2 + min_gap_px, enforced if same tier
    for j in range(n - 1):
        min_sep = (widths_px[j] + widths_px[j + 1]) // 2 + min_gap_px
        model.Add(X[j + 1] - X[j] >= min_sep).OnlyEnforceIf(S[j])

    # Encourage global spacing: X_{j+1} - X_j - halfwidthsum >= delta (if same tier)
    for j in range(n - 1):
        halfsum = (widths_px[j] + widths_px[j + 1]) // 2
        model.Add(X[j + 1] - X[j] - halfsum >= delta).OnlyEnforceIf(S[j])

    # Max tier usage
    for j in range(n):
        model.Add(T[j] <= M_t)

    # Objective
    # Minimize w_anchor * sum(D) + w_tiers * M_t  - w_spread * delta
    obj_terms = []
    if w_anchor:
        obj_terms.append(w_anchor * sum(D))
    if w_tiers:
        obj_terms.append(w_tiers * M_t)
    if w_spread:
        obj_terms.append(-w_spread * delta)
    model.Minimize(sum(obj_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = 8  # parallelize if possible

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT could not find a feasible label layout.")

    X_sol = [int(solver.Value(X[j])) for j in range(n)]
    T_sol = [int(solver.Value(T[j])) for j in range(n)]
    M_t_sol = int(solver.Value(M_t))
    delta_sol = int(solver.Value(delta))

    return X_sol, T_sol, M_t_sol, delta_sol


# =========================
# Main
# =========================
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
    label_texts = [textwrap.fill(lbl.replace("_", " "), width=WRAP_CHARS) for lbl in raw_col_labels]
    row_labels = list(pv.index)

    data = pv.values
    n_rows, n_cols = data.shape

    # Color limits (symmetric about 0), robust to outliers
    finite_abs = np.abs(data[np.isfinite(data)])
    vmax = np.nanpercentile(finite_abs, PERCENTILE_CAP) if finite_abs.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    # Figure (initial)
    fig_w, fig_h = compute_figsize(n_rows, n_cols)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)

    # Heatmap (vectorized): pcolormesh over a unit grid
    x = np.arange(n_cols + 1)
    y = np.arange(n_rows + 1)
    masked = np.ma.masked_invalid(data)

    # Colormap (Matplotlib 3.7+)
    cmap = mpl.colormaps.get_cmap(COLORMAP).with_extremes(bad="#D9D9D9")

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
    ax.set_xlim(0, n_cols)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)

    # Y labels: sparsify to avoid overlap (phenotype labels handled via optimized layout)
    y_keep = sparse_y_ticks(n_rows, MAX_YLABELS)
    y_ticklabels = [row_labels[i] if i in set(y_keep) else "" for i in range(n_rows)]
    ax.set_yticklabels(y_ticklabels)

    # Hide default x tick labels; we’ll draw optimized ones ourselves
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
        xy = (j, i)  # bottom-left of cell in data coords
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

    # Iterate once (df deduplicated on (Inversion, Phenotype))
    for inv, ph, pval, qval in zip(df["Inversion"], df["Phenotype"], df["P_Value"], df["BH_q"]):
        i = row_lookup.get(inv, None)
        j = col_lookup.get(ph, None)
        if i is None or j is None:
            continue
        if pd.notna(qval) and qval < Q_THRESHOLD:
            outline_cell(i, j, "q")
        elif pd.notna(pval) and pval < P_THRESHOLD:
            outline_cell(i, j, "p")

    # ---------- Optimized phenotype labels with CP-SAT ----------
    # Layout prep: fix left/right margins first so axes width is stable
    base_bottom = 0.16
    fig.subplots_adjust(left=0.16, right=0.86, bottom=base_bottom, top=0.90)
    fig.canvas.draw()

    # Anchors in display pixels (center of each column)
    x_centers_data = np.arange(n_cols) + 0.5
    anchors_px = [int(round(data_x_to_px(ax, xc))) for xc in x_centers_data]

    # Measure label sizes in pixels (multi-line)
    label_sizes = [measure_text_bbox_px(fig, txt, X_LABEL_FONTSIZE) for txt in label_texts]
    widths_px = [int(math.ceil(w + 2 * LABEL_BOX_PAD_PX)) for (w, h) in label_sizes]
    heights_px = [int(math.ceil(h)) for (w, h) in label_sizes]

    # Tier height: uniform spacing based on tallest label + vertical padding
    TIER_HEIGHT_PX = max(heights_px) + 6

    # X bounds for label centers (keep labels within axes width, respecting half-widths)
    ax_x0, ax_y0, ax_w, ax_h = get_axes_bbox_px(ax)
    x_min_px = int(math.floor(ax_x0 + widths_px[0] / 2 + 2))   # small left buffer
    x_max_px = int(math.ceil(ax_x0 + ax_w - widths_px[-1] / 2 - 2))

    # Solve CP-SAT layout
    X_sol_px, T_sol, M_t_sol, delta_sol = solve_label_layout_cp_sat(
        anchors_px=anchors_px,
        widths_px=widths_px,
        tier_count=TIER_COUNT,
        x_min_px=x_min_px,
        x_max_px=x_max_px,
        min_gap_px=LABEL_GAP_MIN_PX,
        eps_order_px=EPS_ORDER_PX,
        w_anchor=W_ANCHOR,
        w_tiers=W_TIERS,
        w_spread=W_SPREAD,
        time_limit_sec=SOLVER_TIME_LIMIT_SEC,
    )

    # Compute required bottom margin (in figure fraction) for the labels area
    max_tier_used = M_t_sol
    needed_px = ROUTE_OFFSET_PX + (max_tier_used + 1) * TIER_HEIGHT_PX + 12  # + small cushion
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    new_bottom = clamp(base_bottom + needed_px / fig_h_px, 0.16, 0.60)
    fig.subplots_adjust(bottom=new_bottom)
    fig.canvas.draw()

    # Convert solved X to axes-fraction (for drawing)
    X_sol_frac = [axes_frac_x_from_px(ax, xpx) for xpx in X_sol_px]
    A_frac = [axes_frac_x_from_px(ax, apx) for apx in anchors_px]

    # Y positions in axes-fraction units (negative values extend into bottom margin)
    # Axis baseline (inside axes) is y=0 in ax.transAxes. Below-axis positions are negative.
    route_y_frac = -ROUTE_OFFSET_PX / ax_h
    tier_y_tops = [-(ROUTE_OFFSET_PX + (t + 0) * TIER_HEIGHT_PX) / ax_h for t in range(TIER_COUNT)]
    # Label anchor position (top of the text box) per label:
    label_top_y_frac = [-(ROUTE_OFFSET_PX + (T_sol[j]) * TIER_HEIGHT_PX) / ax_h for j in range(n_cols)]

    # Draw leaders: 3-segment orthogonal polylines (no crossings by construction)
    for j in range(n_cols):
        # Segment 1: vertical from (A_frac[j], 0) to (A_frac[j], route_y_frac)
        ax.add_line(Line2D(
            [A_frac[j], A_frac[j]], [0.0, route_y_frac],
            transform=ax.transAxes, lw=LEADER_LW, color=LEADER_COLOR, clip_on=False
        ))
        # Segment 2: horizontal from (A_frac[j], route_y_frac) to (X_sol_frac[j], route_y_frac)
        ax.add_line(Line2D(
            [A_frac[j], X_sol_frac[j]], [route_y_frac, route_y_frac],
            transform=ax.transAxes, lw=LEADER_LW, color=LEADER_COLOR, clip_on=False
        ))
        # Segment 3: vertical from (X_sol_frac[j], route_y_frac) down to top of label box
        ax.add_line(Line2D(
            [X_sol_frac[j], X_sol_frac[j]], [route_y_frac, label_top_y_frac[j]],
            transform=ax.transAxes, lw=LEADER_LW, color=LEADER_COLOR, clip_on=False
        ))

    # Draw labels (top-aligned at label_top_y_frac)
    for j, txt in enumerate(label_texts):
        ax.text(
            X_sol_frac[j], label_top_y_frac[j],
            txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=X_LABEL_FONTSIZE,
            bbox=LABEL_BBOX,
            clip_on=False,
        )

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
