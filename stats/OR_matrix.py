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

from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage.graph import route_through_array as skimage_route_through_array

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
CELL = 0.55
EXTRA_W = 5.2
EXTRA_H = 2.2
MIN_W, MAX_W = 12.0, 80.0
MIN_H, MAX_H = 7.0, 60.0

# Y-axis label density
MAX_YLABELS = 80

# Label text
X_LABEL_FONTSIZE = 9
WRAP_CHARS = 22

# Label packing (STRICT no-overlap; NO “windows” in pathfinding sense)
GAP_PX = 8.0                    # min horizontal whitespace between labels (px)
ROW_GAP_PX = 10.0               # extra vertical gap between tiers (px)
BASE_OFFSET_PX = 18.0           # distance from axis baseline to first tier top (px)
MAX_TIERS = 4000                # high cap for safety

# Routing grid (FULL AXES, NO CORRIDORS)
GRID_PX = 2.0                   # cell size in pixels (small → finer; keep ≤ 2 for tight channels)
SAFETY_GAP_PX = 1.0             # lines end this many px above label top
LINE_TUBE_PX = 1.0              # raster tube radius for already-committed lines (exact “width”)
CLEAR_START_GOAL_RADIUS_CELLS = 1  # carve free cells around start/goal (cell radius)

# Smoothing (always apply; validated against occupancy)
SMOOTH_STEP_PX = 2.0            # sampling step for smoothed curve
DP_EPS_PX = 2.0                 # Douglas–Peucker epsilon for polyline simplification

# Rendering
LEADER_LW = 0.6
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
    Greedy shelf packing by anchor x; tiers are just horizontal rows to ensure vertical separation.
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
    Order-preserving x-placement with minimum movement and hard separation.
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
# Grid utilities & pathfinding (FULL-AXES, NO CORRIDORS)
# =========================
def build_global_grids(ax_x0: float,
                       ax_y0: float,
                       ax_w: float,
                       ax_h: float,
                       label_rects_px: List[Tuple[float, float, float, float]],
                       grid_px: float,
                       scale: float = 3.0,
                       margin_px: float = 32.0) -> Tuple[float, float, int, int]:
    """
    Build the global occupancy grid so it FULLY contains the horizontal span of all labels
    (with margin) and the vertical span from the x-axis baseline down to the deeper of:
      - 'scale' × axes height, or
      - the deepest label bottom,
    again with margin. Snap the grid bounds to the GRID_PX lattice so label corners never
    quantize outside due to float -> int rounding.

    Returns (gx0, gy0, gw, gh), where (gx0, gy0) is the bottom-left of the grid (in px).
    """
    # ----------------------------
    # Compute required horizontal span (labels may extend left/right of axes)
    # ----------------------------
    if label_rects_px:
        x_min_labels = min(L for (L, R, T, B) in label_rects_px)
        x_max_labels = max(R for (L, R, T, B) in label_rects_px)
    else:
        x_min_labels = ax_x0
        x_max_labels = ax_x0 + ax_w

    # Desired (continuous) pixel bounds before snapping
    x_left_desired  = min(ax_x0,            x_min_labels - margin_px)
    x_right_desired = max(ax_x0 + ax_w,     x_max_labels + margin_px)

    # ----------------------------
    # Compute required vertical span (top at baseline)
    # ----------------------------
    if label_rects_px:
        y_min_labels = min(B for (_, _, _, B) in label_rects_px)  # smallest y (deepest)
    else:
        y_min_labels = ax_y0 - ax_h

    y_min_scaled = ax_y0 - scale * ax_h
    y_bottom_desired = min(y_min_scaled, y_min_labels - margin_px)  # bottom (smaller y)
    y_top_desired    = ax_y0                                        # top (baseline)

    # ----------------------------
    # SNAP bounds to the grid to avoid off-by-one from int() truncation
    # (Ensure labels/endpoints quantize INSIDE the grid)
    # ----------------------------
    def snap_down(v: float) -> float:
        return math.floor(v / grid_px) * grid_px

    def snap_up(v: float) -> float:
        return math.ceil(v / grid_px) * grid_px

    gx0 = snap_down(x_left_desired)
    gx1 = snap_up(x_right_desired)
    gy0 = snap_down(y_bottom_desired)
    gy1 = snap_up(y_top_desired)

    # Safety: ensure at least 1 cell span in each dimension
    width_px  = max(grid_px, gx1 - gx0)
    height_px = max(grid_px, gy1 - gy0)

    # Grid size in cells (+1 so last index is addressable after center-offset)
    gw = int(math.ceil(width_px  / grid_px)) + 1
    gh = int(math.ceil(height_px / grid_px)) + 1

    # ----------------------------
    # Debug prints
    # ----------------------------
    print(f"[INFO] Grid builder: scale={scale:.2f}, margin={margin_px:.1f}px, GRID_PX={grid_px:.1f}px")
    print(f"[INFO] Grid X-range px: left gx0={gx0:.1f}, right={gx1:.1f}, width={width_px:.1f}  "
          f"(axes [{ax_x0:.1f},{ax_x0+ax_w:.1f}], labels [{x_min_labels:.1f},{x_max_labels:.1f}])")
    print(f"[INFO] Grid Y-range px: bottom gy0={gy0:.1f}, top={gy1:.1f}, depth={height_px:.1f}  "
          f"(baseline {ax_y0:.1f}, deepest label {y_min_labels:.1f})")
    print(f"[INFO] Grid size cells: gw×gh = {gw}×{gh}")

    return gx0, gy0, gw, gh

def rasterize_labels_into(occ: np.ndarray, gx0: float, gy0: float, grid_px: float,
                          label_rects_px: List[Tuple[float,float,float,float]]):
    """
    Paint labels (rectangles) as BLOCKED cells (no inflation).
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

def supercover_cells(p0: Tuple[float,float], p1: Tuple[float,float],
                     gx0: float, gy0: float, grid_px: float) -> List[Tuple[int,int]]:
    """
    Amanatides & Woo 2D supercover traversal in grid coords.
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

def rasterize_polyline_tube_into(occ: np.ndarray, poly: List[Tuple[float,float]],
                                 gx0: float, gy0: float, grid_px: float, tube_px: float):
    """
    Paint a polyline tube into occ (set to 1). Tube radius in pixels (>= 0).
    """
    gh, gw = occ.shape
    r2 = (tube_px / grid_px) ** 2 + 0.25
    Rint = max(0, int(math.ceil(tube_px / grid_px)))
    disk_offsets = [(di, dj) for dj in range(-Rint-1, Rint+2)
                    for di in range(-Rint-1, Rint+2) if (di*di + dj*dj) <= r2]
    for p0, p1 in zip(poly, poly[1:]):
        cells = supercover_cells(p0, p1, gx0, gy0, grid_px)
        for (ci, cj) in cells:
            if 0 <= cj < gh and 0 <= ci < gw:
                if Rint == 0:
                    occ[cj, ci] = 1
                else:
                    for di, dj in disk_offsets:
                        ii = ci + di; jj = cj + dj
                        if 0 <= jj < gh and 0 <= ii < gw:
                            occ[jj, ii] = 1

def los_free(occ: np.ndarray, p0: Tuple[float,float], p1: Tuple[float,float],
             gx0: float, gy0: float, grid_px: float) -> bool:
    """Line-of-sight on grid: all supercover cells must be free."""
    gh, gw = occ.shape
    for (ci, cj) in supercover_cells(p0, p1, gx0, gy0, grid_px):
        if not (0 <= cj < gh and 0 <= ci < gw):  # outside grid → treat as blocked
            return False
        if occ[cj, ci]:
            return False
    return True

def greedy_shortcut(poly: List[Tuple[float,float]],
                    occ: np.ndarray, gx0: float, gy0: float, grid_px: float) -> List[Tuple[float,float]]:
    """Remove intermediate points when there is direct LOS."""
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
    """Standard DP simplification (epsilon in pixels)."""
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

def sample_catmull_rom(points: List[Tuple[float,float]], step_px: float) -> List[Tuple[float,float]]:
    """Catmull–Rom spline resampling with ~step_px spacing."""
    if len(points) <= 2:
        return points
    P = [points[0]] + points + [points[-1]]
    out = [points[0]]
    def cr(p0,p1,p2,p3,t):
        t2=t*t; t3=t2*t
        x0,y0=p0; x1,y1=p1; x2,y2=p2; x3,y3=p3
        x = 0.5*((-x0 + 3*x1 - 3*x2 + x3)*t3 + (2*x0 - 5*x1 + 4*x2 - x3)*t2 + (-x0 + x2)*t + 2*x1)
        y = 0.5*((-y0 + 3*y1 - 3*y2 + y3)*t3 + (2*y0 - 5*y1 + 4*y2 - y3)*t2 + (-y0 + y2)*t + 2*y1)
        return (x,y)
    for i in range(1, len(points)):
        p0 = P[i-1]; p1=P[i]; p2=P[i+1]; p3=P[i+2]
        seg_len = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        steps = max(2, int(seg_len / max(1.0, step_px)))
        for s in range(1, steps+1):
            out.append(cr(p0,p1,p2,p3,s/steps))
    return out

# ---------- Pathfinding with scikit-image (FULL GRID) ----------
def _route_one_candidate(args):
    """
    Top-level function for ProcessPool.
    Compute a candidate path on FULL occupancy using labels + prior-batch lines.
    STRICT endpoint handling: endpoints MUST be inside the grid (no clamping).
    """
    (label_id, start_px, goal_px, global_occ_labels, global_occ_lines_prev,
     gx0, gy0, gw, gh, grid_px, clear_r_cells) = args

    # Build combined occupancy snapshot (labels + prior batches' lines), copy so we can carve
    occ = (global_occ_labels | global_occ_lines_prev).astype(np.uint8).copy()

    # Map display px -> grid (row, col). STRICT in-bounds (no clamping).
    def pxtog_strict(xp: float, yp: float) -> Tuple[int, int]:
        c = int((xp - gx0) / grid_px)
        r = int((yp - gy0) / grid_px)
        if not (0 <= r < gh and 0 <= c < gw):
            raise RuntimeError(
                f"Endpoint out of grid for label {label_id}: "
                f"(x={xp:.1f}, y={yp:.1f}) -> (r={r}, c={c}), grid size {gh}×{gw}, "
                f"gx0={gx0:.1f}, gy0={gy0:.1f}, GRID_PX={grid_px:.1f}"
            )
        return (r, c)

    s_rc = pxtog_strict(*start_px)
    g_rc = pxtog_strict(*goal_px)
    sr, sc = s_rc; gr, gc = g_rc

    # Clear small halos at start & goal to ensure connectivity to exact endpoints
    r = clear_r_cells
    occ[max(0, sr - r):min(gh, sr + r + 1), max(0, sc - r):min(gw, sc + r + 1)] = 0
    occ[max(0, gr - r):min(gh, gr + r + 1), max(0, gc - r):min(gw, gc + r + 1)] = 0

    # Cost array: 1.0 for free, +inf for blocked
    cost = np.ones_like(occ, dtype=np.float64)
    cost[occ != 0] = np.inf

    # Route (8-connected, Euclidean edge weights)
    path_rc, total_cost = skimage_route_through_array(
        cost, s_rc, g_rc, fully_connected=True, geometric=True
    )

    # Convert path to pixel centers
    poly = [(gx0 + (c + 0.5) * grid_px, gy0 + (r + 0.5) * grid_px) for (r, c) in path_rc]

    return (label_id, poly, float(total_cost), occ.shape)

def assert_grid_covers_labels_and_endpoints(label_rects_px: List[Tuple[float,float,float,float]],
                                            starts_px: List[Tuple[float,float]],
                                            goals_px: List[Tuple[float,float]],
                                            gx0: float, gy0: float, gw: int, gh: int,
                                            grid_px: float) -> None:
    """
    Fail fast if ANY label rectangle or ANY start/goal lies outside the grid.
    """
    def pxtog(xp: float, yp: float) -> Tuple[int, int]:
        c = int((xp - gx0) / grid_px)
        r = int((yp - gy0) / grid_px)
        return r, c

    # Check labels
    for idx, (L, R, T, B) in enumerate(label_rects_px):
        rL, cL = pxtog(L, T)     # top-left
        rR, cR = pxtog(R, B)     # bottom-right (lower y)
        # All four corners must map strictly inside
        corners = [pxtog(L, T), pxtog(R, T), pxtog(L, B), pxtog(R, B)]
        for (rr, cc) in corners:
            if not (0 <= rr < gh and 0 <= cc < gw):
                raise RuntimeError(
                    f"Label {idx} out of grid: corner (r={rr}, c={cc}) outside 0..{gh-1} × 0..{gw-1}. "
                    f"LRTB=({L:.1f},{R:.1f},{T:.1f},{B:.1f}), gx0={gx0:.1f}, gy0={gy0:.1f}, GRID_PX={grid_px:.1f}"
                )

    # Check endpoints
    for k, (xp, yp) in enumerate(starts_px):
        r, c = pxtog(xp, yp)
        if not (0 <= r < gh and 0 <= c < gw):
            raise RuntimeError(
                f"Start {k} out of grid: (x={xp:.1f}, y={yp:.1f}) -> (r={r}, c={c}). "
                f"Grid {gh}×{gw}, gy0={gy0:.1f}, gx0={gx0:.1f}, GRID_PX={grid_px:.1f}"
            )
    for k, (xp, yp) in enumerate(goals_px):
        r, c = pxtog(xp, yp)
        if not (0 <= r < gh and 0 <= c < gw):
            raise RuntimeError(
                f"Goal {k} out of grid: (x={xp:.1f}, y={yp:.1f}) -> (r={r}, c={c}). "
                f"Grid {gh}×{gw}, gy0={gy0:.1f}, gx0={gx0:.1f}, GRID_PX={grid_px:.1f}"
            )

    print("[INFO] Grid coverage assertion passed: all labels and all endpoints are inside the routing grid.")


def save_labels_only_plot(fig_ref: plt.Figure,
                          ax_ref: plt.Axes,
                          Xc_px: List[float],
                          Ytop_abs_px: List[float],
                          label_texts: List[str],
                          fontsize: float,
                          out_prefix: str,
                          margins: Tuple[float,float,float,float]) -> None:
    """
    Save a separate figure that shows ONLY the labels (no lines, no heatmap).
    Positions are taken from absolute pixel locations relative to ax_ref.
    """
    left, right, bottom, top = margins  # fractions
    # New figure sized like the reference
    w_in, h_in = fig_ref.get_size_inches()
    fig2, ax2 = plt.subplots(figsize=(w_in, h_in), constrained_layout=False)
    fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    # Compute axes window for transform to axes fraction
    fig2.canvas.draw()
    ax2.set_axis_off()  # hide spines/ticks; "just labels"

    # Use the reference axes window for consistent normalization
    x0, y0, w, h = ax_ref.get_window_extent(renderer=fig_ref.canvas.get_renderer()).bounds

    def to_axes_frac(px: float, py: float) -> Tuple[float, float]:
        return ((px - x0) / w, (py - y0) / h)

    # Draw texts
    for j, txt in enumerate(label_texts):
        xf, yf = to_axes_frac(Xc_px[j], Ytop_abs_px[j])
        ax2.text(xf, yf, txt, transform=ax2.transAxes, ha="center", va="top",
                 fontsize=fontsize, clip_on=False)

    out_svg = f"{out_prefix}_labels_only.svg"
    out_pdf = f"{out_prefix}_labels_only.pdf"
    fig2.savefig(out_svg)
    fig2.savefig(out_pdf)
    plt.close(fig2)
    print(f"[SAVED] {out_svg}")
    print(f"[SAVED] {out_pdf}")

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

    # ---------- Prepare label geometry ----------
    base_bottom = 0.16
    fig.subplots_adjust(left=0.16, right=0.86, bottom=base_bottom, top=0.90)
    fig.canvas.draw()

    ax_x0, ax_y0, ax_w, ax_h = axes_window_px(ax)
    axis_baseline_y = ax_y0

    # Anchors (px)
    x_centers_data = np.arange(n_cols) + 0.5
    anchors_px = [data_x_to_px(ax, xc) for xc in x_centers_data]

    print(f"[INFO] Columns: {n_cols}, Rows: {n_rows}, Figure DPI: {fig.dpi}, Axes px: {ax_w:.1f}×{ax_h:.1f}")

    # Measure label sizes in px
    label_sizes = [measure_text_bbox_px(fig, txt, X_LABEL_FONTSIZE) for txt in label_texts]
    widths_px = [w for (w, h) in label_sizes]
    heights_px = [h for (w, h) in label_sizes]
    max_h_px = max(heights_px) if heights_px else 0.0

    # Tier assignment and per-tier x placement (STRICT NO OVERLAP)
    tiers = shelf_pack_tiers(anchors_px, widths_px, GAP_PX)
    K = max(tiers) + 1 if tiers else 0
    tier_indices = [[] for _ in range(K)]
    for j, t in enumerate(tiers):
        tier_indices[t].append(j)

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

    tier_pitch_px = max_h_px + ROW_GAP_PX
    verify_no_label_overlap(tiers, Xc_px, widths_px, heights_px, axis_baseline_y, tier_pitch_px)
    print(f"[INFO] Label tiers: {K}, tier pitch: {tier_pitch_px:.1f}px, GAP_PX: {GAP_PX:.1f}px")

    # Build label rectangles (px)
    label_rects_px: List[Tuple[float,float,float,float]] = []
    Ytop_abs_px = [0.0]*n_cols
    for j in range(n_cols):
        t = tiers[j]
        top_y = axis_baseline_y - (BASE_OFFSET_PX + t * tier_pitch_px)
        Ytop_abs_px[j] = top_y
        L = Xc_px[j] - widths_px[j]/2.0
        R = Xc_px[j] + widths_px[j]/2.0
        T = top_y
        B = top_y - heights_px[j]
        label_rects_px.append((L, R, T, B))

    # Compute bottom margin to fully show labels
    deepest_bottom = min(B for (_, _, _, B) in label_rects_px) if label_rects_px else axis_baseline_y - BASE_OFFSET_PX
    needed_extra = (axis_baseline_y - deepest_bottom) + 14.0
    new_bottom = clamp(base_bottom + needed_extra / (fig.get_size_inches()[1] * fig.dpi), 0.16, 0.85)
    fig.subplots_adjust(bottom=new_bottom)
    fig.canvas.draw()
    print(f"[INFO] Bottom margin set to {new_bottom:.3f} (needed extra: {needed_extra:.1f}px)")

    # ---------- SAVE LABELS-ONLY PLOT (before any pathfinding) ----------
    save_labels_only_plot(
        fig_ref=fig,
        ax_ref=ax,
        Xc_px=Xc_px,
        Ytop_abs_px=Ytop_abs_px,
        label_texts=label_texts,
        fontsize=X_LABEL_FONTSIZE,
        out_prefix=OUT_PREFIX,
        margins=(0.16, 0.86, new_bottom, 0.90)
    )

    # ---------- Global grids (TRIPLE depth & cover all labels) ----------
    gx0, gy0, gw, gh = build_global_grids(
        ax_x0=ax_x0, ax_y0=ax_y0, ax_w=ax_w, ax_h=ax_h,
        label_rects_px=label_rects_px,
        grid_px=GRID_PX, scale=3.0, margin_px=32.0
    )

    # Post-grid: assert that labels and endpoints are inside the grid
    starts_all = [(anchors_px[j], axis_baseline_y) for j in range(n_cols)]
    goals_all  = [(Xc_px[j], Ytop_abs_px[j] - SAFETY_GAP_PX) for j in range(n_cols)]
    assert_grid_covers_labels_and_endpoints(
        label_rects_px=label_rects_px,
        starts_px=starts_all,
        goals_px=goals_all,
        gx0=gx0, gy0=gy0, gw=gw, gh=gh,
        grid_px=GRID_PX
    )

    # Build global occupancies
    global_occ_labels = np.zeros((gh, gw), dtype=np.uint8)
    rasterize_labels_into(global_occ_labels, gx0, gy0, GRID_PX, label_rects_px)
    global_occ_lines = np.zeros_like(global_occ_labels, dtype=np.uint8)  # committed line tubes

    print(f"[INFO] Routing window px: x[{ax_x0:.1f},{ax_x0+ax_w:.1f}]  y[{gy0:.1f},{ax_y0:.1f}]")
    print(f"[INFO] Global grid: {gw}×{gh} cells @ {GRID_PX:.1f}px; labels blocked cells: {int(global_occ_labels.sum())}")

    # ---------- Route order: 4 batches by anchor quartiles ----------
    order = np.argsort(anchors_px)  # indices sorted by anchor x
    qsize = max(1, n_cols // 4)
    batches = [
        list(order[0:qsize]),
        list(order[qsize:2*qsize]),
        list(order[2*qsize:3*qsize]),
        list(order[3*qsize:]),
    ]
    batches = [b for b in batches if len(b) > 0]

    committed_paths: Dict[int, List[Tuple[float,float]]] = {}

    # Acceptance collision check WITH symmetric endpoint carve-outs
    def path_cells_blocked(poly: List[Tuple[float,float]],
                           carve_start: Optional[Tuple[float,float]] = None,
                           carve_goal: Optional[Tuple[float,float]] = None,
                           carve_r_cells: int = CLEAR_START_GOAL_RADIUS_CELLS) -> bool:
        occ = (global_occ_labels | global_occ_lines).astype(np.uint8).copy()

        def pxtog(xp: float, yp: float) -> Tuple[int,int]:
            c = int((xp - gx0) / GRID_PX)
            r = int((yp - gy0) / GRID_PX)
            return r, c

        if carve_start is not None:
            sr, sc = pxtog(*carve_start)
            if 0 <= sr < gh and 0 <= sc < gw:
                r = carve_r_cells
                occ[max(0,sr-r):min(gh,sr+r+1), max(0,sc-r):min(gw,sc+r+1)] = 0
        if carve_goal is not None:
            gr, gc = pxtog(*carve_goal)
            if 0 <= gr < gh and 0 <= gc < gw:
                r = carve_r_cells
                occ[max(0,gr-r):min(gh,gr+r+1), max(0,gc-r):min(gw,gc+r+1)] = 0

        for p0, p1 in zip(poly, poly[1:]):
            for (ci, cj) in supercover_cells(p0, p1, gx0, gy0, GRID_PX):
                if not (0 <= cj < gh and 0 <= ci < gw):
                    return True
                if occ[cj, ci]:
                    return True
        return False

    # ---------- Process batches ----------
    for bi, batch in enumerate(batches, 1):
        print(f"\n[INFO] === Batch {bi}/{len(batches)}: {len(batch)} labels ===")
        starts = {j: (anchors_px[j], axis_baseline_y) for j in batch}
        goals  = {j: (Xc_px[j], Ytop_abs_px[j] - SAFETY_GAP_PX) for j in batch}

        remaining = batch.copy()
        round_id = 0
        while remaining:
            round_id += 1
            t_round = time.perf_counter()
            print(f"[INFO] Batch {bi} Round {round_id}: remaining {len(remaining)} | building candidates in parallel...")

            # Build snapshot of lines so far (prior batches + already accepted in earlier rounds)
            args_list = []
            for j in remaining:
                args_list.append((
                    j,
                    starts[j], goals[j],
                    global_occ_labels,        # labels only
                    global_occ_lines,         # committed lines so far
                    gx0, gy0, gw, gh, GRID_PX,
                    CLEAR_START_GOAL_RADIUS_CELLS
                ))

            candidates: Dict[int, Tuple[List[Tuple[float,float]], float, Tuple[int,int]]] = {}
            with ProcessPoolExecutor(max_workers=min(8, max(1, len(remaining)//4))) as ex:
                futs = {ex.submit(_route_one_candidate, a): a[0] for a in args_list}
                for fut in as_completed(futs):
                    jid = futs[fut]
                    try:
                        label_id, poly, cost_val, shape = fut.result()
                        candidates[label_id] = (poly, cost_val, shape)
                    except Exception as e:
                        print(f"[WARN] Candidate routing failed for label {jid}: {e}")
                        candidates[label_id] = (None, float("inf"), (gh, gw))

            print(f"[INFO] Batch {bi} Round {round_id}: candidates built in {time.perf_counter()-t_round:.2f}s")

            accepted = []
            rejected = []
            for j in remaining:
                cand = candidates.get(j, (None, float("inf"), (gh, gw)))[0]
                if cand is None:
                    rejected.append(j)
                    continue
                if path_cells_blocked(cand, carve_start=starts[j], carve_goal=goals[j]):
                    rejected.append(j)
                else:
                    # Validate simplified/smoothed variants against ACTIVE occupancy
                    occ_active = (global_occ_labels | global_occ_lines).astype(np.uint8).copy()
                    poly2 = greedy_shortcut(cand, occ_active, gx0, gy0, GRID_PX)
                    simp = douglas_peucker(poly2, DP_EPS_PX)
                    ok = True
                    for p0, p1 in zip(simp, simp[1:]):
                        if not los_free(occ_active, p0, p1, gx0, gy0, GRID_PX):
                            ok = False; break
                    if ok:
                        poly2 = simp
                    if len(poly2) >= 3:
                        smooth = sample_catmull_rom(poly2, SMOOTH_STEP_PX)
                        ok2 = True
                        for p0, p1 in zip(smooth, smooth[1:]):
                            if not los_free(occ_active, p0, p1, gx0, gy0, GRID_PX):
                                ok2 = False; break
                        if ok2:
                            poly2 = smooth
                    if path_cells_blocked(poly2, carve_start=starts[j], carve_goal=goals[j]):
                        rejected.append(j)
                    else:
                        rasterize_polyline_tube_into(global_occ_lines, poly2, gx0, gy0, GRID_PX, LINE_TUBE_PX)
                        committed_paths[j] = poly2
                        accepted.append(j)

            print(f"[INFO] Batch {bi} Round {round_id}: accepted {len(accepted)}, rejected {len(rejected)}")

            if rejected:
                next_remaining = []
                for j in rejected:
                    occ = (global_occ_labels | global_occ_lines).astype(np.uint8).copy()
                    def pxtog(xp, yp) -> Tuple[int,int]:
                        c = int((xp - gx0) / GRID_PX)
                        r = int((yp - gy0) / GRID_PX)
                        return (r, c)
                    sr, sc = pxtog(*starts[j]); gr, gc = pxtog(*goals[j])
                    r = CLEAR_START_GOAL_RADIUS_CELLS
                    occ[max(0,sr-r):min(gh,sr+r+1), max(0,sc-r):min(gw,sc+r+1)] = 0
                    occ[max(0,gr-r):min(gh,gr+r+1), max(0,gc-r):min(gw,gc+r+1)] = 0
                    cost = np.ones_like(occ, dtype=np.float64)
                    cost[occ != 0] = np.inf
                    try:
                        path_rc, _ = skimage_route_through_array(
                            cost, (sr, sc), (gr, gc), fully_connected=True, geometric=True
                        )
                        poly = [(gx0 + (c + 0.5) * GRID_PX, gy0 + (r + 0.5) * GRID_PX) for (r, c) in path_rc]
                        if not path_cells_blocked(poly, carve_start=starts[j], carve_goal=goals[j]):
                            occ_active = (global_occ_labels | global_occ_lines).astype(np.uint8).copy()
                            poly2 = greedy_shortcut(poly, occ_active, gx0, gy0, GRID_PX)
                            simp = douglas_peucker(poly2, DP_EPS_PX)
                            ok = True
                            for p0, p1 in zip(simp, simp[1:]):
                                if not los_free(occ_active, p0, p1, gx0, gy0, GRID_PX):
                                    ok = False; break
                            if ok: poly2 = simp
                            if len(poly2) >= 3:
                                smooth = sample_catmull_rom(poly2, SMOOTH_STEP_PX)
                                ok2 = True
                                for p0, p1 in zip(smooth, smooth[1:]):
                                    if not los_free(occ_active, p0, p1, gx0, gy0, GRID_PX):
                                        ok2 = False; break
                                if ok2: poly2 = smooth
                            if not path_cells_blocked(poly2, carve_start=starts[j], carve_goal=goals[j]):
                                rasterize_polyline_tube_into(global_occ_lines, poly2, gx0, gy0, GRID_PX, LINE_TUBE_PX)
                                committed_paths[j] = poly2
                            else:
                                next_remaining.append(j)
                        else:
                            next_remaining.append(j)
                    except Exception as e:
                        print(f"[WARN] Sequential re-route failed for label {j}: {e}")
                        next_remaining.append(j)

                remaining = next_remaining
            else:
                remaining = []

            print(f"[INFO] Batch {bi} Round {round_id} done in {time.perf_counter()-t_round:.2f}s, still remaining: {len(remaining)}")

        print(f"[INFO] Batch {bi} committed lines so far: {len(committed_paths)}; blocked cells: {int(global_occ_lines.sum())}")

    # ---------- DRAW: Curves & labels ----------
    ax_x0, ax_y0, ax_w, ax_h = axes_window_px(ax)
    def to_axes_frac(p: Tuple[float,float]) -> Tuple[float,float]:
        return ((p[0] - ax_x0) / ax_w, (p[1] - ax_y0) / ax_h)

    for j in np.argsort(anchors_px):
        poly = committed_paths.get(j)
        if not poly:
            continue
        pts_axes = [to_axes_frac(p) for p in poly]
        xs = [p[0] for p in pts_axes]; ys = [p[1] for p in pts_axes]
        ax.add_line(Line2D(xs, ys, transform=ax.transAxes, lw=LEADER_LW, color=LEADER_COLOR, clip_on=False))

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
    print(f"[STATS] Final blocked cells — labels: {int(global_occ_labels.sum())}, lines: {int(global_occ_lines.sum())}, total: {int((global_occ_labels|global_occ_lines).sum())}")

if __name__ == "__main__":
    main()
