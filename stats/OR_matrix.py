#!/usr/bin/env python3
from pathlib import Path
import math
import time
import textwrap
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRectangle
from matplotlib.lines import Line2D

# Optional: try to import skimage route. We also provide a deterministic fallback path.
try:
    from skimage.graph import route_through_array as skimage_route_through_array
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

# =========================
# Global configuration (BREATHING ROOM MODE)
# =========================
INPUT_PATH = "phewas_results.tsv"
OUT_PREFIX = "phewas_heatmap"

# Significance thresholds
P_THRESHOLD = 0.05
Q_THRESHOLD = 0.05

# Colormap & scaling
COLORMAP = "RdBu_r"
PERCENTILE_CAP = 99.5          # clip color range to this percentile of |ln(OR)|

# Figure sizing (inches) — allow VERY WIDE figures if needed
CELL = 0.60
EXTRA_W = 10.0
EXTRA_H = 3.0
MIN_W, MAX_W = 16.0, 120.0
MIN_H, MAX_H = 8.0, 60.0

# Y-axis label density
MAX_YLABELS = 80

# Label text
X_LABEL_FONTSIZE = 9
WRAP_CHARS = 24

# Label packing (STRICT no-overlap)
GAP_PX = 24.0                  # min horizontal whitespace between labels (px) — BIG
ROW_GAP_PX = 24.0              # extra vertical gap between tiers (px) — BIG
BASE_OFFSET_PX = 100.0         # distance from axis baseline to first tier top (px) — BIG
MAX_TIERS = 8000               # safety

# Global “free band” right under the x-axis to make routing trivial
TOP_BAND_PX = 80.0             # tall, continuous, always free band under baseline
GOAL_GAP_PX = 8.0              # how far above label top a goal sits (toward baseline)
Y_JITTER_PX = 2.0              # small y jitter for labels (kept non-overlapping)

# Routing grid (FULL AXES)
GRID_PX = 4.0                  # coarser cell to shrink memory/time (4 px)
LINE_TUBE_PX = 1.0             # not used to block lines anymore (we allow line overlaps)
CLEAR_START_GOAL_RADIUS_CELLS = 3  # carve bigger halos around endpoints
MARGIN_PX = 128.0              # generous grid margin
GRID_DEPTH_SCALE = 4.0         # grid vertical depth multiplier

# Smoothing (optional — we keep light)
SMOOTH_STEP_PX = 4.0
DP_EPS_PX = 3.0

# Rendering
LEADER_LW = 0.8
LEADER_COLOR = "black"
DRAW_CELL_EDGES = True
EDGE_LW = 0.7
EDGE_COLOR = "white"
DASH_PATTERN = (0, (1.0, 1.0))
LW_DASHED = 0.5
LW_SOLID = 0.9
LABEL_BBOX = dict(facecolor=None, edgecolor=None, boxstyle=None, pad=0.0)

# Matplotlib rc
mpl.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# Utilities
# =========================
def validate_columns(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def compute_figsize(n_rows: int, n_cols: int) -> Tuple[float, float]:
    # Allow very large width; we’ll also widen right/left room so axis grows
    w = clamp(n_cols * CELL + EXTRA_W, MIN_W, MAX_W)
    h = clamp(n_rows * CELL + EXTRA_H, MIN_H, MAX_H)
    return (w, h)

def sparse_y_ticks(n: int, max_labels: int) -> np.ndarray:
    if n <= max_labels:
        return np.arange(n)
    step = int(math.ceil(n / max_labels))
    return np.arange(0, n, step)

def measure_text_bbox_px(fig, text: str, fontsize: float) -> Tuple[float, float]:
    fig.canvas.draw()
    t = mpl.text.Text(x=0, y=0, text=text, fontsize=fontsize, multialignment="center")
    t.set_figure(fig)
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return bbox.width, bbox.height

def data_x_to_px(ax, x_data: float) -> float:
    fig = ax.figure
    fig.canvas.draw()
    xy_px = ax.transData.transform((x_data, 0.0))
    return xy_px[0]

def axes_window_px(ax) -> Tuple[float,float,float,float]:
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    return bbox.x0, bbox.y0, bbox.width, bbox.height

# =========================
# Label tiering & x-positioning (STRICT, NO OVERLAP)
# =========================
def shelf_pack_tiers(anchors: List[float], widths: List[float], gap_px: float) -> List[int]:
    """
    Greedy shelf packing by anchor x; tiers are horizontal rows to ensure vertical separation.
    """
    n = len(anchors)
    order = sorted(range(n), key=lambda i: anchors[i])
    tier_last_x = []
    tier_last_w = []
    assign = [-1] * n
    for j in order:
        A = anchors[j]; W = widths[j]
        placed = False
        for t in range(len(tier_last_x)):
            need = (tier_last_w[t] + W) / 2.0 + gap_px
            if A >= tier_last_x[t] + need:
                assign[j] = t
                tier_last_x[t] = A
                tier_last_w[t] = W
                placed = True
                break
        if not placed:
            assign[j] = len(tier_last_x)
            tier_last_x.append(A)
            tier_last_w.append(W)
        if len(tier_last_x) > MAX_TIERS:
            raise RuntimeError("Exceeded MAX_TIERS — labels too dense. Enlarge figure width or reduce font size.")
    return assign

def isotonic_with_spacing(
    idxs: List[int],
    anchors: List[float],
    widths: List[float],
    x_lo: float, x_hi: float,
    gap_px: float
) -> List[float]:
    """
    Order-preserving x-placement with hard separation. With lots of width and big gap_px,
    this stays very conservative to avoid shoulder-to-shoulder crowding.
    """
    if not idxs:
        return []
    A = [anchors[i] for i in idxs]
    W = [widths[i] for i in idxs]
    m = len(idxs)
    LB = [x_lo + W[k]/2.0 for k in range(m)]
    UB = [x_hi - W[k]/2.0 for k in range(m)]
    s = [ (W[k] + W[k+1])/2.0 + gap_px for k in range(m-1) ]

    E = [0.0]*m
    E[0] = max(LB[0], A[0])
    for k in range(1, m):
        E[k] = max(LB[k], E[k-1] + s[k-1], A[k])

    L = [0.0]*m
    L[m-1] = min(UB[m-1], A[m-1])
    for k in range(m-2, -1, -1):
        L[k] = min(UB[k], L[k+1] - s[k], A[k])

    for k in range(m):
        if E[k] > L[k]:
            mid = (E[k] + L[k]) / 2.0
            E[k] = mid; L[k] = mid

    X = [clamp(A[k], E[k], L[k]) for k in range(m)]
    X[m-1] = clamp(X[m-1], E[m-1], L[m-1])
    for k in range(m-2, -1, -1):
        X[k] = min(X[k], X[k+1] - s[k])
        X[k] = clamp(X[k], E[k], L[k])
    for k in range(1, m):
        X[k] = max(X[k], X[k-1] + s[k-1])
        X[k] = clamp(X[k], E[k], L[k])

    return X

def verify_no_label_overlap(
    tiers: List[int],
    Xc_px: List[float],
    widths_px: List[float],
    heights_px: List[float],
    axis_baseline_y: float,
    tier_pitch_px: float
):
    """
    HARD assertion: no label rectangles overlap. Sweep per tier by left edge.
    """
    K = max(tiers) + 1 if tiers else 0
    per_tier = [[] for _ in range(K)]
    for i, t in enumerate(tiers):
        top_y = axis_baseline_y - (BASE_OFFSET_PX + t * tier_pitch_px)
        left = Xc_px[i] - widths_px[i]/2.0
        right = Xc_px[i] + widths_px[i]/2.0
        bottom = top_y - heights_px[i]
        per_tier[t].append((left, right, top_y, bottom, i))
    for t in range(K):
        row = sorted(per_tier[t], key=lambda r: r[0])
        for a, b in zip(row, row[1:]):
            if a[1] + 1e-6 > b[0]:
                raise RuntimeError("Label placement produced overlap — enlarge figure width or increase GAP_PX/ROW_GAP_PX.")

# =========================
# Grid utilities & routing helpers
# =========================
def snap_down(v: float, grid_px: float) -> float:
    return math.floor(v / grid_px) * grid_px

def snap_up(v: float, grid_px: float) -> float:
    return math.ceil(v / grid_px) * grid_px

def build_global_grid(ax_x0: float,
                      ax_y0: float,
                      ax_w: float,
                      ax_h: float,
                      label_rects_px: List[Tuple[float, float, float, float]],
                      grid_px: float,
                      scale: float,
                      margin_px: float) -> Tuple[float, float, int, int, float, float]:
    """
    Build a global grid that fully covers:
      * entire axes width (with margins) and entire x-span of labels
      * vertical span from the x-axis baseline down to the deeper of (scale×axes height) or the deepest label bottom
    Returns (gx0, gy0, gw, gh, gx1, gy1) where (gx0,gy0) bottom-left in px, (gx1,gy1) top-right in px.
    """
    if label_rects_px:
        x_min_labels = min(L for (L, R, T, B) in label_rects_px)
        x_max_labels = max(R for (L, R, T, B) in label_rects_px)
        y_min_labels = min(B for (_, _, _, B) in label_rects_px)
    else:
        x_min_labels = ax_x0
        x_max_labels = ax_x0 + ax_w
        y_min_labels = ax_y0 - ax_h

    x_left_desired  = min(ax_x0,            x_min_labels) - margin_px
    x_right_desired = max(ax_x0 + ax_w,     x_max_labels) + margin_px

    y_bottom_desired = min(ax_y0 - scale * ax_h, y_min_labels - margin_px)
    y_top_desired    = ax_y0  # baseline

    gx0 = snap_down(x_left_desired, grid_px)
    gx1 = snap_up(x_right_desired, grid_px)
    gy0 = snap_down(y_bottom_desired, grid_px)
    gy1 = snap_up(y_top_desired, grid_px)

    width_px  = max(grid_px, gx1 - gx0)
    height_px = max(grid_px, gy1 - gy0)

    gw = int(math.ceil(width_px  / grid_px)) + 1
    gh = int(math.ceil(height_px / grid_px)) + 1

    print(f"[INFO] Grid builder: scale={scale:.2f}, margin={margin_px:.1f}px, GRID_PX={grid_px:.1f}px")
    print(f"[INFO] Grid X-range px: left gx0={gx0:.1f}, right={gx1:.1f}, width={width_px:.1f}  "
          f"(axes [{ax_x0:.1f},{ax_x0+ax_w:.1f}], labels [{x_min_labels:.1f},{x_max_labels:.1f}])")
    print(f"[INFO] Grid Y-range px: bottom gy0={gy0:.1f}, top={gy1:.1f}, depth={height_px:.1f}  "
          f"(baseline {ax_y0:.1f}, deepest label {y_min_labels:.1f})")
    print(f"[INFO] Grid size cells: gw×gh = {gw}×{gh}")
    return gx0, gy0, gw, gh, gx1, gy1

def rasterize_labels_into(occ: np.ndarray, gx0: float, gy0: float, grid_px: float,
                          label_rects_px: List[Tuple[float,float,float,float]]):
    """
    Paint labels (rectangles) as BLOCKED cells.
    """
    gh, gw = occ.shape
    for (L, R, T, B) in label_rects_px:
        il0 = int((L - gx0) / grid_px)
        ir1 = int((R - gx0) / grid_px)
        jb0 = int((B - gy0) / grid_px)
        jt1 = int((T - gy0) / grid_px)
        il0 = clamp(il0, 0, gw-1); ir1 = clamp(ir1, 0, gw-1)
        jb0 = clamp(jb0, 0, gh-1); jt1 = clamp(jt1, 0, gh-1)
        if il0 <= ir1 and jb0 <= jt1:
            occ[jb0:jt1+1, il0:ir1+1] = 1

def carve_top_band_free(occ: np.ndarray, ax_y0: float, gx0: float, gy0: float, grid_px: float, band_px: float):
    """
    Carve a continuous FREE band just under the baseline across the full grid width.
    """
    gh, gw = occ.shape
    band_top_y = ax_y0
    band_bot_y = ax_y0 - band_px
    j0 = int((band_bot_y - gy0) / grid_px)
    j1 = int((band_top_y - gy0) / grid_px)
    j0 = clamp(j0, 0, gh-1)
    j1 = clamp(j1, 0, gh-1)
    if j1 < j0:
        j0, j1 = j1, j0
    occ[j0:j1+1, :] = 0
    print(f"[INFO] Carved FREE top band: y∈[{band_bot_y:.1f},{band_top_y:.1f}] (~{band_px:.1f}px) → rows {j0}..{j1}")

def carve_vertical_slit_for_goal(occ: np.ndarray,
                                 gx0: float, gy0: float, grid_px: float,
                                 goal_px: Tuple[float,float],
                                 up_to_y: float,
                                 half_width_cells: int = 1):
    """
    Carve a thin vertical slit from 'up_to_y' down to the goal y to ensure connectivity.
    """
    gh, gw = occ.shape
    xg, yg = goal_px
    ci = int((xg - gx0) / grid_px)
    jt = int((up_to_y - gy0) / grid_px)
    jb = int((yg - gy0) / grid_px)
    ci = clamp(ci, 0, gw-1)
    jt = clamp(jt, 0, gh-1)
    jb = clamp(jb, 0, gh-1)
    if jb < jt:
        jt, jb = jb, jt
    il0 = clamp(ci - half_width_cells, 0, gw-1)
    ir1 = clamp(ci + half_width_cells, 0, gw-1)
    occ[jt:jb+1, il0:ir1+1] = 0

def supercover_cells(p0: Tuple[float,float], p1: Tuple[float,float],
                     gx0: float, gy0: float, grid_px: float) -> List[Tuple[int,int]]:
    """
    Amanatides & Woo 2D grid traversal.
    """
    x0 = (p0[0] - gx0) / grid_px
    y0 = (p0[1] - gy0) / grid_px
    x1 = (p1[0] - gx0) / grid_px
    y1 = (p1[1] - gy0) / grid_px

    i = int(math.floor(x0))
    j = int(math.floor(y0))
    i_end = int(math.floor(x1))
    j_end = int(math.floor(y1))

    cells = [(i, j)]
    dx = x1 - x0; dy = y1 - y0
    step_i = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_j = 0 if dy == 0 else (1 if dy > 0 else -1)
    tDelta_i = float("inf") if dx == 0 else abs(1.0 / dx)
    tDelta_j = float("inf") if dy == 0 else abs(1.0 / dy)

    def frac(a): return a - math.floor(a)

    if step_i > 0: tMax_i = (1.0 - frac(x0)) * tDelta_i
    elif step_i < 0: tMax_i = frac(x0) * tDelta_i
    else: tMax_i = float("inf")
    if step_j > 0: tMax_j = (1.0 - frac(y0)) * tDelta_j
    elif step_j < 0: tMax_j = frac(y0) * tDelta_j
    else: tMax_j = float("inf")

    max_iter = 4 * (abs(i_end - i) + abs(j_end - j) + 2)
    iters = 0
    while (i != i_end or j != j_end) and iters < max_iter:
        iters += 1
        if tMax_i < tMax_j:
            i += step_i; tMax_i += tDelta_i; cells.append((i, j))
        elif tMax_j < tMax_i:
            j += step_j; tMax_j += tDelta_j; cells.append((i, j))
        else:
            i += step_i; j += step_j
            tMax_i += tDelta_i; tMax_j += tDelta_j
            cells.append((i, j))
    return cells

def los_free(occ: np.ndarray, p0: Tuple[float,float], p1: Tuple[float,float],
             gx0: float, gy0: float, grid_px: float) -> bool:
    gh, gw = occ.shape
    for (ci, cj) in supercover_cells(p0, p1, gx0, gy0, grid_px):
        if not (0 <= cj < gh and 0 <= ci < gw):
            return False
        if occ[cj, ci]:
            return False
    return True

def greedy_shortcut(poly: List[Tuple[float,float]],
                    occ: np.ndarray, gx0: float, gy0: float, grid_px: float) -> List[Tuple[float,float]]:
    if len(poly) <= 2:
        return poly
    out = [poly[0]]
    i = 0
    while i < len(poly) - 1:
        k = len(poly) - 1
        while k > i + 1 and not los_free(occ, poly[i], poly[k], gx0, gy0, grid_px):
            k -= 1
        out.append(poly[k])
        i = k
    return out

def douglas_peucker(points: List[Tuple[float,float]], eps: float) -> List[Tuple[float,float]]:
    if len(points) < 3:
        return points
    def point_seg_dist(p, a, b):
        (px, py), (ax, ay), (bx, by) = p, a, b
        vx, vy = bx - ax, by - ay
        if vx == 0 and vy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * vx + (py - ay) * vy) / (vx*vx + vy*vy)
        t = clamp(t, 0.0, 1.0)
        qx = ax + t * vx; qy = ay + t * vy
        return math.hypot(px - qx, py - qy)
    def recurse(pts, i0, i1, keep):
        if i1 <= i0 + 1: return
        a = pts[i0]; b = pts[i1]
        idx, dmax = -1, -1.0
        for i in range(i0+1, i1):
            d = point_seg_dist(pts[i], a, b)
            if d > dmax:
                dmax = d; idx = i
        if dmax > eps:
            keep.add(idx)
            recurse(pts, i0, idx, keep)
            recurse(pts, idx, i1, keep)
    keep = {0, len(points) - 1}
    recurse(points, 0, len(points) - 1, keep)
    return [points[i] for i in sorted(keep)]

def simple_manhattan_fallback(start: Tuple[float,float],
                              goal: Tuple[float,float],
                              band_y: float) -> List[Tuple[float,float]]:
    """
    Deterministic fallback path: start (on axis) → down to band_y → horizontal to goal.x → down to goal.y.
    Assumes band_y is above label tops (true by construction).
    """
    sx, sy = start
    gx, gy = goal
    return [(sx, sy),
            (sx, band_y - 1.0),          # slightly inside the band
            (gx, band_y - 1.0),
            (gx, gy)]

# =========================
# Main
# =========================
def main():
    t_start = time.perf_counter()

    # ---------- Read & preprocess ----------
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

    # Normed OR
    df["normed_OR"] = np.log(df["OR"].values)

    # Deduplicate per (Inversion, Phenotype)
    df["_abs_effect"] = df["normed_OR"].abs()
    df["_BH_q_sort"] = df["BH_q"].fillna(np.inf)
    df["_P_sort"] = df["P_Value"].fillna(np.inf)
    df = df.sort_values(
        by=["_BH_q_sort", "_P_sort", "_abs_effect"],
        ascending=[True, True, False]
    ).drop_duplicates(subset=["Inversion", "Phenotype"], keep="first")

    # Sort columns by mean absolute effect
    col_strength = df.groupby("Phenotype")["normed_OR"].apply(lambda s: np.nanmean(np.abs(s)))
    col_order = col_strength.sort_values(ascending=False).index.tolist()

    # Row order
    row_order = pd.unique(df["Inversion"].values).tolist()

    # Build matrices
    pv = df.pivot(index="Inversion", columns="Phenotype", values="normed_OR").reindex(index=row_order, columns=col_order)
    pmat = df.pivot(index="Inversion", columns="Phenotype", values="P_Value").reindex(index=row_order, columns=col_order)
    qmat = df.pivot(index="Inversion", columns="Phenotype", values="BH_q").reindex(index=row_order, columns=col_order)

    # Labels (phenotypes)
    raw_col_labels = [str(c) for c in pv.columns]
    label_texts = [textwrap.fill(lbl.replace("_", " "), width=WRAP_CHARS) for lbl in raw_col_labels]
    row_labels = list(pv.index)

    data = pv.values
    n_rows, n_cols = data.shape

    # Color scaling
    finite_abs = np.abs(data[np.isfinite(data)])
    vmax = np.nanpercentile(finite_abs, PERCENTILE_CAP) if finite_abs.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    # ---------- Figure & heatmap ----------
    fig_w, fig_h = compute_figsize(n_rows, n_cols)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)

    # VERY generous left/right so axes width grows a lot; also leave big top to fit title/colorbar
    left_frac, right_frac, top_frac = 0.12, 0.88, 0.92
    base_bottom = 0.16
    fig.subplots_adjust(left=left_frac, right=right_frac, bottom=base_bottom, top=top_frac)
    fig.canvas.draw()

    x = np.arange(n_cols + 1)
    y = np.arange(n_rows + 1)
    masked = np.ma.masked_invalid(data)

    cmap = mpl.colormaps.get_cmap(COLORMAP).with_extremes(bad="#D9D9D9")

    edgecolors = EDGE_COLOR if DRAW_CELL_EDGES else "face"
    linewidth = EDGE_LW if DRAW_CELL_EDGES else 0.0

    mesh = ax.pcolormesh(
        x, y, masked,
        cmap=cmap, vmin=-vmax, vmax=vmax,
        edgecolors=edgecolors, linewidth=linewidth, shading="flat"
    )
    ax.invert_yaxis()

    ax.set_xlabel("Phenotype (phecode)")
    ax.set_ylabel("Inversion")
    ax.set_title("Inversion–Phenotype Associations (normed OR = ln(OR))", pad=12)

    ax.set_xlim(0, n_cols)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)

    # Y tick labels sparse
    y_keep = sparse_y_ticks(n_rows, MAX_YLABELS)
    y_ticklabels = [row_labels[i] if i in set(y_keep) else "" for i in range(n_rows)]
    ax.set_yticklabels(y_ticklabels)

    # Hide default x ticks
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normed OR (ln scale)\nnegative: decreased risk · positive: increased risk")

    # Significance outlines
    row_lookup = {inv: i for i, inv in enumerate(pv.index)}
    col_lookup = {jph: j for j, jph in enumerate(pv.columns)}

    def outline_cell(i, j, kind):
        xy = (j, i)
        if kind == "q":
            rect = MplRectangle(xy, 1, 1, fill=False, lw=LW_SOLID, linestyle="solid", edgecolor="black")
        elif kind == "p":
            rect = MplRectangle(xy, 1, 1, fill=False, lw=LW_DASHED, linestyle=DASH_PATTERN, edgecolor="black")
        else:
            return
        ax.add_patch(rect)

    for inv, ph, pval, qval in zip(df["Inversion"], df["Phenotype"], df["P_Value"], df["BH_q"]):
        i = row_lookup.get(inv, None)
        j = col_lookup.get(ph, None)
        if i is None or j is None:
            continue
        if pd.notna(qval) and qval < Q_THRESHOLD:
            outline_cell(i, j, "q")
        elif pd.notna(pval) and pval < P_THRESHOLD:
            outline_cell(i, j, "p")

    # ---------- Prepare label geometry (LOTS of spacing + small y-jitter, STRICT NO-OVERLAP) ----------
    fig.canvas.draw()
    ax_x0, ax_y0, ax_w, ax_h = axes_window_px(ax)
    axis_baseline_y = ax_y0

    # Anchors (px) directly under each column center
    x_centers_data = np.arange(n_cols) + 0.5
    anchors_px = [data_x_to_px(ax, xc) for xc in x_centers_data]

    print(f"[INFO] Columns: {n_cols}, Rows: {n_rows}, Figure DPI: {fig.dpi}, Axes px: {ax_w:.1f}×{ax_h:.1f}")

    # Measure label sizes in px
    label_sizes = [measure_text_bbox_px(fig, txt, X_LABEL_FONTSIZE) for txt in label_texts]
    widths_px = [w for (w, h) in label_sizes]
    heights_px = [h for (w, h) in label_sizes]
    max_h_px = max(heights_px) if heights_px else 0.0

    # Assign to horizontal tiers to avoid overlap (by x, with big gaps)
    tiers = shelf_pack_tiers(anchors_px, widths_px, GAP_PX)
    K = max(tiers) + 1 if tiers else 0
    tier_indices = [[] for _ in range(K)]
    for j, t in enumerate(tiers):
        tier_indices[t].append(j)

    # Place centers within [ax_x0+4, ax_x0+ax_w-4]
    Xc_px = [0.0] * n_cols
    for t in range(K):
        idxs = tier_indices[t]
        X_t = isotonic_with_spacing(
            idxs=idxs,
            anchors=anchors_px,
            widths=widths_px,
            x_lo=ax_x0 + 4.0, x_hi=ax_x0 + ax_w - 4.0,
            gap_px=GAP_PX
        )
        for k_local, j in enumerate(idxs):
            Xc_px[j] = X_t[k_local]

    # Vertical placement: big tier pitch + tiny jitter, but keep a guaranteed FREE TOP BAND above all labels
    rng = np.random.RandomState(123)
    tier_pitch_px = max_h_px + ROW_GAP_PX
    label_rects_px: List[Tuple[float,float,float,float]] = []
    Ytop_abs_px = [0.0]*n_cols
    for j in range(n_cols):
        t = tiers[j]
        jitter = float(rng.uniform(-Y_JITTER_PX, Y_JITTER_PX))
        top_y = axis_baseline_y - (BASE_OFFSET_PX + t * tier_pitch_px) + jitter
        # Ensure label top is BELOW the guaranteed top band
        top_y = min(top_y, axis_baseline_y - (TOP_BAND_PX + 6.0))
        Ytop_abs_px[j] = top_y
        L = Xc_px[j] - widths_px[j]/2.0
        R = Xc_px[j] + widths_px[j]/2.0
        T = top_y
        B = top_y - heights_px[j]
        label_rects_px.append((L, R, T, B))

    # STRICT per-tier overlap check (tiers + big pitch makes cross-tier overlap impossible here)
    verify_no_label_overlap(tiers, Xc_px, widths_px, heights_px, axis_baseline_y, tier_pitch_px)
    print(f"[INFO] Label tiers: {K}, tier pitch: {tier_pitch_px:.1f}px, GAP_PX={GAP_PX:.1f}px, jitter≤{Y_JITTER_PX:.1f}px")

    # Compute bottom margin to fully show labels
    deepest_bottom = min(B for (_, _, _, B) in label_rects_px) if label_rects_px else axis_baseline_y - BASE_OFFSET_PX
    needed_extra = (axis_baseline_y - deepest_bottom) + 20.0
    new_bottom = clamp(base_bottom + needed_extra / (fig.get_size_inches()[1] * fig.dpi), 0.16, 0.88)
    fig.subplots_adjust(bottom=new_bottom)
    fig.canvas.draw()
    print(f"[INFO] Bottom margin set to {new_bottom:.3f} (needed extra: {needed_extra:.1f}px)")

    # ---------- SAVE LABELS-ONLY PLOT (before any pathfinding) ----------
    def save_labels_only_plot():
        left, right, bottom, top = left_frac, right_frac, new_bottom, top_frac
        w_in, h_in = fig.get_size_inches()
        fig2, ax2 = plt.subplots(figsize=(w_in, h_in), constrained_layout=False)
        fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        fig2.canvas.draw()
        ax2.set_axis_off()
        # Use the original axes' pixel box to convert absolute px to axes fractions
        x0, y0, w, h = ax.get_window_extent(renderer=fig.canvas.get_renderer()).bounds
        def to_axes_frac(px: float, py: float) -> Tuple[float, float]:
            return ((px - x0) / w, (py - y0) / h)
        for j, txt in enumerate(label_texts):
            xf, yf = to_axes_frac(Xc_px[j], Ytop_abs_px[j])
            ax2.text(xf, yf, txt, transform=ax2.transAxes, ha="center", va="top",
                     fontsize=X_LABEL_FONTSIZE, clip_on=False)
        out_svg = f"{OUT_PREFIX}_labels_only.svg"
        out_pdf = f"{OUT_PREFIX}_labels_only.pdf"
        fig2.savefig(out_svg)
        fig2.savefig(out_pdf)
        plt.close(fig2)
        print(f"[SAVED] {out_svg}")
        print(f"[SAVED] {out_pdf}")

    save_labels_only_plot()

    # ---------- Global grid (CAVERNOUS, EASY) ----------
    gx0, gy0, gw, gh, gx1, gy1 = build_global_grid(
        ax_x0=ax_x0, ax_y0=ax_y0, ax_w=ax_w, ax_h=ax_h,
        label_rects_px=label_rects_px,
        grid_px=GRID_PX, scale=GRID_DEPTH_SCALE, margin_px=MARGIN_PX
    )
    # Build occupancy: labels only (we allow line overlaps → no line blocking)
    global_occ_labels = np.zeros((gh, gw), dtype=np.uint8)
    rasterize_labels_into(global_occ_labels, gx0, gy0, GRID_PX, label_rects_px)

    # Carve FREE top band under baseline
    carve_top_band_free(global_occ_labels, ax_y0=ax_y0, gx0=gx0, gy0=gy0, grid_px=GRID_PX, band_px=TOP_BAND_PX)

    # Precompute endpoints
    starts_all = [(anchors_px[j], axis_baseline_y) for j in range(n_cols)]
    # Goals are above label top (toward baseline), safely within the free slit zone we’ll carve
    goals_all  = [(Xc_px[j], Ytop_abs_px[j] + GOAL_GAP_PX) for j in range(n_cols)]

    print(f"[INFO] Routing window px: x[{gx0:.1f},{gx1:.1f}]  y[{gy0:.1f},{ax_y0:.1f}]")
    print(f"[INFO] Global grid: {gw}×{gh} cells @ {GRID_PX:.1f}px; labels blocked cells: {int(global_occ_labels.sum())}")

    # ---------- Route ALL labels (sequential, robust) ----------
    committed_paths: Dict[int, List[Tuple[float,float]]] = {}

    def pxtog(xp: float, yp: float) -> Tuple[int,int]:
        c = int((xp - gx0) / GRID_PX)
        r = int((yp - gy0) / GRID_PX)
        return r, c

    band_mid_y = ax_y0 - (TOP_BAND_PX * 0.5)

    for j in range(n_cols):
        start = starts_all[j]
        goal  = goals_all[j]

        # Build a fresh working occupancy snapshot (labels only, plus free band)
        occ = global_occ_labels.copy()

        # Carve a vertical slit from band down to the goal for this label
        carve_vertical_slit_for_goal(occ, gx0, gy0, GRID_PX,
                                     goal_px=goal,
                                     up_to_y=band_mid_y,
                                     half_width_cells=1)

        # Carve halos at start & goal
        sr, sc = pxtog(*start)
        gr, gc = pxtog(*goal)
        r = CLEAR_START_GOAL_RADIUS_CELLS
        gh_occ, gw_occ = occ.shape
        occ[max(0, sr - r):min(gh_occ, sr + r + 1), max(0, sc - r):min(gw_occ, sc + r + 1)] = 0
        occ[max(0, gr - r):min(gh_occ, gr + r + 1), max(0, gc - r):min(gw_occ, gc + r + 1)] = 0

        # Try shortest path if available; else deterministic fallback
        poly: List[Tuple[float,float]]
        used_fallback = False
        if HAVE_SKIMAGE:
            try:
                cost = np.ones_like(occ, dtype=np.float64)
                cost[occ != 0] = np.inf
                s_rc = (sr, sc); g_rc = (gr, gc)
                path_rc, _ = skimage_route_through_array(
                    cost, s_rc, g_rc, fully_connected=True, geometric=True
                )
                poly = [(gx0 + (c + 0.5) * GRID_PX, gy0 + (r + 0.5) * GRID_PX) for (r, c) in path_rc]
            except Exception:
                used_fallback = True
        else:
            used_fallback = True

        if used_fallback:
            poly = simple_manhattan_fallback(start, goal, band_y=band_mid_y)

        # Light simplification (respecting current occupancy)
        poly2 = greedy_shortcut(poly, occ, gx0, gy0, GRID_PX)
        poly3 = douglas_peucker(poly2, DP_EPS_PX)

        committed_paths[j] = poly3

    print(f"[INFO] Routed {len(committed_paths)} / {n_cols} label leaders (lines may overlap each other; labels are strictly avoided).")

    # ---------- DRAW: Curves & labels ----------
    ax_x0, ax_y0, ax_w, ax_h = axes_window_px(ax)
    def to_axes_frac(p: Tuple[float,float]) -> Tuple[float,float]:
        return ((p[0] - ax_x0) / ax_w, (p[1] - ax_y0) / ax_h)

    # Draw lines sorted by x to reduce visual overdraw noise
    for j in np.argsort(anchors_px):
        poly = committed_paths.get(j)
        if not poly:
            continue
        pts_axes = [to_axes_frac(p) for p in poly]
        xs = [p[0] for p in pts_axes]; ys = [p[1] for p in pts_axes]
        ax.add_line(Line2D(xs, ys, transform=ax.transAxes, lw=LEADER_LW, color=LEADER_COLOR, clip_on=False))

    # Draw labels
    for j, txt in enumerate(label_texts):
        xf = (Xc_px[j] - ax_x0) / ax_w
        yf = (Ytop_abs_px[j] - ax_y0) / ax_h
        ax.text(xf, yf, txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=X_LABEL_FONTSIZE, bbox=LABEL_BBOX, clip_on=False)

    mesh.set_rasterized(False)

    out_svg = f"{OUT_PREFIX}.svg"
    out_pdf = f"{OUT_PREFIX}.pdf"
    fig.savefig(out_svg)
    fig.savefig(out_pdf)

    t_end = time.perf_counter()
    print(f"\n[SAVED] {out_svg}")
    print(f"[SAVED] {out_pdf}")
    print(f"[DONE] Total time: {t_end - t_start:.2f}s | labels: {n_cols}, rows: {n_rows}, grid: {GRID_PX}px")
    print(f"[STATS] Label blocks: {int(global_occ_labels.sum())}; paths drawn: {len(committed_paths)}")

if __name__ == "__main__":
    main()
