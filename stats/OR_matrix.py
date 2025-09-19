from pathlib import Path
import math
import textwrap
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRectangle
from matplotlib.transforms import blended_transform_factory

from _inv_common import map_inversion_series

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
PERCENTILE_CAP = 99.5  # clip color range to this percentile of |ln(OR)|

# Base figure sizing (inches). Width scales up to avoid label label collisions.
CELL_W = 0.16          # base width per column (inches) — we scale this up as needed
CELL_H = 0.55          # height per row (inches)
EXTRA_W = 6.0          # extra width (colorbar, side margins)
EXTRA_H = 2.0          # extra height (title, axes labels)
MIN_W, MAX_W = 18.0, 220.0
MIN_H, MAX_H = 7.0, 60.0

# Axes margins (fractions). Bottom is adjusted dynamically, starting from this.
LEFT_FRAC   = 0.14
RIGHT_FRAC  = 0.86
TOP_FRAC    = 0.90
BOTTOM_BASE = 0.20      # starting point; will be increased just enough to avoid cutoffs

# Y-axis tick label density
MAX_YLABELS = 80

# X label appearance (single level, steep angle near x-axis)
X_LABEL_FONTSIZE = 9
X_LABEL_WRAP = 200            # effectively keep labels as one line (just in case)
X_LABEL_ROT_DEG = 68          # steep rotation
X_LABEL_SAFETY_PAD_PX = 2.0   # inflate w/h by this many px for collision clearance
LABEL_OFFSET = 0.05           # axes fraction below baseline (close to axis)

# Ensure labels are inside the canvas
BOTTOM_FIG_PAD_PX = 8.0       # min pixel distance from lowest label pixel to figure bottom

# Leader lines (from baseline to label)
LEADER_LW = 0.7
LEADER_COLOR = "black"

# Cell edges
DRAW_CELL_EDGES = True
EDGE_LW = 0.7
EDGE_COLOR = "white"

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
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def compute_figsize(n_rows: int, n_cols: int, width_scale: float) -> Tuple[float, float]:
    w = n_cols * CELL_W * width_scale + EXTRA_W
    h = n_rows * CELL_H + EXTRA_H
    w = clamp(w, MIN_W, MAX_W)
    h = clamp(h, MIN_H, MAX_H)
    return (w, h)

def sparse_y_ticks(n: int, max_labels: int) -> np.ndarray:
    if n <= max_labels:
        return np.arange(n)
    step = int(math.ceil(n / max_labels))
    return np.arange(0, n, step)

# ---------- Geometry helpers for rotated OBB collision ----------
def rotate_point(x: float, y: float, theta_rad: float) -> Tuple[float, float]:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return (c * x - s * y, s * x + c * y)

def quad_from_anchor_topright_px(anchor_px: Tuple[float, float],
                                 w_px: float, h_px: float,
                                 theta_rad: float) -> np.ndarray:
    """
    Build a quadrilateral (4×2 array) for a rectangle of size (w,h) with its
    TOP-RIGHT corner at the anchor, then rotated by theta around the anchor.

    Local (unrotated) corners relative to anchor (0,0) with top-right at (0,0):
        P0 = (0, 0)          (top-right)
        P1 = (-w, 0)         (top-left)
        P2 = (-w, -h)        (bottom-left)
        P3 = (0, -h)         (bottom-right)
    """
    ax, ay = anchor_px
    local = np.array([
        [ 0.0,  0.0],
        [-w_px, 0.0],
        [-w_px, -h_px],
        [ 0.0, -h_px],
    ], dtype=float)
    rot = np.zeros_like(local)
    for i, (x, y) in enumerate(local):
        rx, ry = rotate_point(x, y, theta_rad)
        rot[i, 0] = ax + rx
        rot[i, 1] = ay + ry
    return rot  # (4,2)

def _project_poly_on_axis(poly: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    unit = axis / np.linalg.norm(axis)
    dots = poly @ unit
    return float(np.min(dots)), float(np.max(dots))

def polygons_intersect_SAT(polyA: np.ndarray, polyB: np.ndarray) -> bool:
    """
    Separating Axis Theorem for convex polygons (here 4-vertex OBBs).
    Returns True if polygons intersect.
    """
    def axes(poly):
        ax_list = []
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i+1) % len(poly)]
            edge = p1 - p0
            axis = np.array([-edge[1], edge[0]], dtype=float)
            if axis[0] == 0.0 and axis[1] == 0.0:
                continue
            ax_list.append(axis)
        return ax_list

    for axis in axes(polyA) + axes(polyB):
        a_min, a_max = _project_poly_on_axis(polyA, axis)
        b_min, b_max = _project_poly_on_axis(polyB, axis)
        if a_max < b_min or b_max < a_min:
            return False
    return True

def any_rotated_collisions(quads: List[np.ndarray]) -> bool:
    """
    Check OBB (quad) collisions pairwise using SAT. O(n^2) but fine for a few hundred labels.
    """
    if not quads:
        return False
    idx = sorted(range(len(quads)), key=lambda i: float(np.min(quads[i][:, 0])))
    for a_pos, ia in enumerate(idx):
        A = quads[ia]
        ax_max = float(np.max(A[:, 0]))
        for ib in idx[a_pos+1:]:
            B = quads[ib]
            if float(np.min(B[:, 0])) > ax_max:
                break
            if polygons_intersect_SAT(A, B):
                return True
    return False

def build_label_quads_px(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    labels: List[str],
    fontsize: float,
    rot_deg: float,
    offset_axes: float,
    pad_px: float
) -> List[np.ndarray]:
    """
    For each label, compute its OBB quad (in DISPLAY PIXELS) matching the final draw settings:
      - anchor at (x=j+0.5, y=-offset_axes) in blended (data, axes) transform
      - rotation about the anchor by rot_deg (CCW)
      - top-right aligned (ha='right', va='top')
      - text measured unrotated at 'fontsize', then inflated by 'pad_px' in both w/h.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    sizes = []
    for raw in labels:
        txt = textwrap.fill(raw.replace("_", " "), width=X_LABEL_WRAP)
        t = mpl.text.Text(x=0, y=0, text=txt, fontsize=fontsize, rotation=0.0)
        t.set_figure(fig)
        bbox = t.get_window_extent(renderer=renderer)
        sizes.append((bbox.width + 2.0 * pad_px, bbox.height + 2.0 * pad_px))

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    theta = math.radians(rot_deg)

    quads = []
    for j, (w, h) in enumerate(sizes):
        x_data = j + 0.5
        y_axes = -offset_axes
        ax_disp, ay_disp = trans.transform((x_data, y_axes))
        quad = quad_from_anchor_topright_px((ax_disp, ay_disp), w, h, theta)
        quads.append(quad)
    return quads

def draw_all_labels_one_level(
    ax: mpl.axes.Axes,
    labels: List[str],
    fontsize: float,
    rot_deg: float,
    offset_axes: float
) -> None:
    """
    Draw ALL labels on ONE tier under the axis, rotated by rot_deg.
    """
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for j, raw in enumerate(labels):
        lbl = textwrap.fill(raw.replace("_", " "), width=X_LABEL_WRAP)
        x_data = j + 0.5
        y_axes = -offset_axes
        ax.text(
            x_data, y_axes, lbl,
            transform=trans,
            rotation=rot_deg,
            rotation_mode="anchor",
            ha="right", va="top",
            fontsize=fontsize,
            clip_on=False
        )

def draw_leaders(
    ax: mpl.axes.Axes,
    n_cols: int,
    offset_axes: float,
    lw: float,
    color: str
) -> None:
    """
    Straight vertical leaders from axis baseline (axes y=0) to label tier (axes y=-offset_axes).
    """
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for j in range(n_cols):
        x = j + 0.5
        ax.plot([x, x], [0.0, -offset_axes], transform=trans, lw=lw, color=color, clip_on=False)

# =========================
# Main
# =========================
def main():
    # ---------- Read & preprocess ----------
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path, sep="\t", dtype=str)
    for c in ["OR", "P_Value", "BH_q"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Inversion", "Phenotype", "OR"])
    df = df[df["OR"] > 0]

    df["Inversion"] = df["Inversion"].astype(str)
    df["Inversion"] = map_inversion_series(df["Inversion"])

    # Normed OR
    df["normed_OR"] = np.log(df["OR"].values)

    # Deduplicate per (Inversion, Phenotype)
    df["_abs_effect"] = df["normed_OR"].abs()
    df["_BH_q_sort"] = df["BH_q"].fillna(np.inf)
    df["_P_sort"]     = df["P_Value"].fillna(np.inf)
    df = df.sort_values(
        by=["_BH_q_sort", "_P_sort", "_abs_effect"],
        ascending=[True, True, False]
    ).drop_duplicates(subset=["Inversion", "Phenotype"], keep="first")

    # Order: columns by MAX absolute effect across inversions (desc), rows by appearance
    col_strength = df.groupby("Phenotype")["normed_OR"].apply(lambda s: np.nanmax(np.abs(s)))
    col_order = col_strength.sort_values(ascending=False).index.tolist()
    row_order = pd.unique(df["Inversion"].values).tolist()

    pv   = df.pivot(index="Inversion", columns="Phenotype", values="normed_OR").reindex(index=row_order, columns=col_order)
    pmat = df.pivot(index="Inversion", columns="Phenotype", values="P_Value").reindex(index=row_order, columns=col_order)
    qmat = df.pivot(index="Inversion", columns="Phenotype", values="BH_q").reindex(index=row_order, columns=col_order)

    data = pv.values
    n_rows, n_cols = data.shape
    print(f"[INFO] Columns: {n_cols}, Rows: {n_rows}")

    # Color scaling
    finite_abs = np.abs(data[np.isfinite(data)])
    vmax = np.nanpercentile(finite_abs, PERCENTILE_CAP) if finite_abs.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    col_labels = [str(c) for c in pv.columns]
    row_labels = list(pv.index)

    # ---------- Iterative layout loop ----------
    width_scale = 1.0
    attempt = 0
    bottom_frac = BOTTOM_BASE
    final_fig = None
    final_ax  = None

    while True:
        attempt += 1
        fig_w, fig_h = compute_figsize(n_rows, n_cols, width_scale=width_scale)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
        fig.subplots_adjust(left=LEFT_FRAC, right=RIGHT_FRAC, bottom=bottom_frac, top=TOP_FRAC)

        # Heatmap
        x = np.arange(n_cols + 1)
        y = np.arange(n_rows + 1)
        masked = np.ma.masked_invalid(data)
        cmap = mpl.colormaps.get_cmap(COLORMAP).with_extremes(bad="#D9D9D9")
        edgecolors = EDGE_COLOR if DRAW_CELL_EDGES else "face"
        linewidth  = EDGE_LW if DRAW_CELL_EDGES else 0.0

        mesh = ax.pcolormesh(
            x, y, masked,
            cmap=cmap, vmin=-vmax, vmax=vmax,
            edgecolors=edgecolors, linewidth=linewidth, shading="flat"
        )
        ax.invert_yaxis()

        ax.set_xlabel("Phenotype (phecode)")
        # Remove y-axis label entirely per request:
        ax.set_ylabel("")  # <-- deleted the y-axis label
        ax.set_title("Inversion–Phenotype Associations (normed OR = ln(OR))", pad=12)

        ax.set_xlim(0, n_cols)
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_yticks(np.arange(n_rows) + 0.5)

        # Sparse y tick labels
        y_keep = sparse_y_ticks(n_rows, MAX_YLABELS)
        y_ticklabels = [row_labels[i] if i in set(y_keep) else "" for i in range(n_rows)]
        ax.set_yticklabels(y_ticklabels)

        # Hide default x tick labels; we draw our own angled labels later
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

        # Cosmetic spines off
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colorbar
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Normed OR (ln scale)\nnegative: decreased risk · positive: increased risk")

        # Significance outlines
        row_lookup = {inv: i for i, inv in enumerate(pv.index)}
        col_lookup = {jph: j for j, jph in enumerate(pv.columns)}
        for inv, ph, pval, qval in zip(df["Inversion"], df["Phenotype"], df["P_Value"], df["BH_q"]):
            i = row_lookup.get(inv, None)
            j = col_lookup.get(ph, None)
            if i is None or j is None:
                continue
            xy = (j, i)
            if pd.notna(qval) and qval < Q_THRESHOLD:
                rect = MplRectangle(xy, 1, 1, fill=False, lw=0.9, linestyle="solid", edgecolor="black")
                ax.add_patch(rect)
            elif pd.notna(pval) and pval < P_THRESHOLD:
                rect = MplRectangle(xy, 1, 1, fill=False, lw=0.5, linestyle=(0, (1.0, 1.0)), edgecolor="black")
                ax.add_patch(rect)

        # Build rotated label quads (DISPLAY pixels) for collision + bottom-padding checks
        fig.canvas.draw()
        quads = build_label_quads_px(
            fig=fig,
            ax=ax,
            labels=col_labels,
            fontsize=X_LABEL_FONTSIZE,
            rot_deg=X_LABEL_ROT_DEG,
            offset_axes=LABEL_OFFSET,
            pad_px=X_LABEL_SAFETY_PAD_PX
        )

        # --- Ensure labels are not cut off at the bottom of the figure ---
        # Find the minimum display Y among all quad vertices.
        min_y_px = min(float(np.min(q[:, 1])) for q in quads) if quads else 1e9
        fig_h_px = fig.get_size_inches()[1] * fig.dpi
        # If any part is below the canvas (min_y_px < desired padding), increase bottom margin.
        if min_y_px < BOTTOM_FIG_PAD_PX and bottom_frac < 0.50:
            missing = BOTTOM_FIG_PAD_PX - min_y_px
            add_frac = missing / fig_h_px
            # push axes up just enough; cap reasonably
            new_bottom = clamp(bottom_frac + add_frac, bottom_frac + 1e-4, 0.50)
            print(f"[INFO] Attempt {attempt}: increasing bottom margin {bottom_frac:.3f} -> {new_bottom:.3f} "
                  f"(needed {missing:.1f}px)")
            bottom_frac = new_bottom
            plt.close(fig)
            continue

        # --- Collision test (rotated OBBs) ---
        collided = any_rotated_collisions(quads)
        print(f"[INFO] Attempt {attempt}: fig_w={fig_w:.1f}in, bottom={bottom_frac:.3f} — collisions: {collided}")

        if not collided or fig_w >= MAX_W - 1e-6:
            # Draw labels + leaders once layout is valid (or at max width)
            draw_all_labels_one_level(
                ax=ax,
                labels=col_labels,
                fontsize=X_LABEL_FONTSIZE,
                rot_deg=X_LABEL_ROT_DEG,
                offset_axes=LABEL_OFFSET
            )
            draw_leaders(
                ax=ax,
                n_cols=n_cols,
                offset_axes=LABEL_OFFSET,
                lw=LEADER_LW,
                color=LEADER_COLOR
            )
            final_fig, final_ax = fig, ax
            if collided:
                print("[WARN] Reached MAX_W; residual label collisions may remain.")
            break

        # Otherwise: widen and retry
        plt.close(fig)
        width_scale *= 1.10  # +10% each iteration

    # ---------- Save outputs (tight bbox to be extra-safe) ----------
    out_svg = f"{OUT_PREFIX}.svg"
    out_pdf = f"{OUT_PREFIX}.pdf"
    final_fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    final_fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"[SAVED] {out_svg}")
    print(f"[SAVED] {out_pdf}")

if __name__ == "__main__":
    main()
