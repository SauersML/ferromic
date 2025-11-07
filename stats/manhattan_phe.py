import os, re, sys, math, subprocess
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import warnings

try:  # optional dependency used for nicer label placement
    from adjustText import adjust_text as ADJUST_TEXT
    _ADJUST_TEXT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional soft dependency
    _ADJUST_TEXT_AVAILABLE = False

    def ADJUST_TEXT(*args, **kwargs):  # type: ignore[override]
        """Fallback when adjustText is unavailable (no-op)."""

        return []

    warnings.warn(
        "adjustText is not installed; proceeding without label collision adjustment.",
        RuntimeWarning,
        stacklevel=2,
    )

matplotlib.use('Agg')

from _inv_common import map_inversion_series

# ---------- Config ----------
INFILE = "data/phewas_results.tsv"
PHECODE_FILE = "phecodeX_R_labels.csv"
PHECODE_URL = "https://raw.githubusercontent.com/PheWAS/PhecodeX/refs/heads/main/phecodeX_R_labels.csv"
OUTDIR = "phewas_plots"
INV_MAPPING_FILE = "data/balanced_recurrence_results.tsv"

PHENO_COL = "Phenotype"
P_Q_COL   = "Q_GLOBAL"
OR_COL    = "OR"
BETA_COL  = "Beta"
INV_COL   = "Inversion"
SIG_COL   = "Sig_Global"

UNCAT_NAME = "Uncategorized"

# Sizing
MIN_WIDTH       = 14.0
MAX_WIDTH       = 26.0
WIDTH_PER_100   = 0.40
FIG_HEIGHT      = 7.8

# Axes placement (figure fractions) to enforce constant drawable width
# Increased margins: left, bottom, width, height - giving 25% more room on all sides
AXES_BBOX       = (0.12, 0.12, 0.76, 0.76)  # left, bottom, width, height

# Markers & style
TRI_BASE_SIZE   = 260.0    # triangle area (pt^2) when OR = 1.0 (larger baseline)
TRI_OR_MIN      = 0.67     # minimum OR for scaling (protective effects)
TRI_OR_MAX      = 1.5      # maximum OR for scaling (risk effects)
POINT_EDGE_LW   = 0.45
POINT_ALPHA_SIG = 0.95
POINT_ALPHA_NONSIG = 0.30


# Risk direction palette
INCOLOR_HEX     = "#2B6CB0"
DECOLOR_HEX     = "#C53030"
SIG_DARKEN      = 0.10     # darken amount for significant points
NON_SIG_LIGHTEN = 0.25     # lighten amount for non-significant points

# Label/legend
LABEL_FONTSZ    = 22.0     # phenotype labels
AX_LABEL_FONTSZ = 22.0     # axis labels (q, x-axis)
TICK_FONTSZ     = 20.0     # tick labels
TITLE_FONTSZ    = 26.0     # plot title
LEGEND_FONTSZ   = 20.0     # legend text
LEGEND_TITLE_SZ = 22.0     # legend title
ANNOTATE_Q_THRESH = 0.05

# Single linebreak rule
MIN_WORDS_BREAK = 6
MIN_WORDS_SIDE  = 3

# Pixel-based margins/headroom
X_PAD_PX        = 18       # left/right padding in pixels (converted to data)
Y_TOP_PAD_FRAC  = 0.08

# adjustText tuning
ADJ_EXPAND_TEXT = (1.06, 1.28)
ADJ_EXPAND_PNTS = (1.03, 1.16)
ADJ_FORCE_PNTS  = (0.07, 0.32)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---------- Data prep ----------
def ensure_phecode_file(path: str, url: str) -> None:
    """Ensure the PheCode category file exists locally, downloading if needed."""
    if os.path.exists(path):
        return

    print(f"[INFO] Downloading {url} → {path}")
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
    except Exception as exc:  # pragma: no cover - network failure
        sys.exit(f"ERROR: Failed to download PheCodeX labels from {url}: {exc}")

    try:
        with open(path, "wb") as fh:
            fh.write(data)
    except OSError as exc:  # pragma: no cover - filesystem failure
        sys.exit(f"ERROR: Could not write downloaded PheCodeX labels to {path}: {exc}")

# ---------- Helpers ----------
def canonicalize_name(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)): return ""
    s = str(s).replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def pretty_text(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)): return ""
    return str(s).replace("_", " ")

def balanced_linebreak(s: str, min_words_each_side=MIN_WORDS_SIDE) -> str:
    words = s.split()
    if len(words) < MIN_WORDS_BREAK: return s
    best_i, best_diff = None, float("inf")
    for i in range(min_words_each_side, len(words) - min_words_each_side + 1):
        L = " ".join(words[:i]); R = " ".join(words[i:])
        diff = abs(len(L) - len(R))
        if diff < best_diff: best_i, best_diff = i, diff
    if best_i is None: return s
    return " ".join(words[:best_i]) + "\n" + " ".join(words[best_i:])

def truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes","y"})

def open_file(path: str) -> None:
    try:
        if sys.platform.startswith("darwin"): subprocess.Popen(["open", path])
        elif os.name == "nt": os.startfile(path)  # type: ignore[attr-defined]
        else: subprocess.Popen(["xdg-open", path])
    except Exception:
        pass

def compute_width(n_points: int) -> float:
    width = MIN_WIDTH + WIDTH_PER_100 * (n_points / 100.0)
    return float(max(MIN_WIDTH, min(MAX_WIDTH, width)))

def sanitize_filename(s: str) -> str:
    s = str(s) if s is not None else "NA"
    s = re.sub(r"[^\w.\-]+", "_", s.strip())
    return s[:200] if s else "NA"

# Palette & shading
def lighten_color(hex_color: str, amount: float) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    r = min(1.0, r + (1.0 - r) * amount)
    g = min(1.0, g + (1.0 - g) * amount)
    b = min(1.0, b + (1.0 - b) * amount)
    return mcolors.to_hex((r, g, b))


def darken_color(hex_color: str, amount: float) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    r = max(0.0, r * (1.0 - amount))
    g = max(0.0, g * (1.0 - amount))
    b = max(0.0, b * (1.0 - amount))
    return mcolors.to_hex((r, g, b))


def scale_all_sizes(or_values: pd.Series) -> np.ndarray:
    """Scale marker sizes linearly with the odds ratio (clamped to [0.67, 1.5])."""
    arr = pd.to_numeric(or_values, errors="coerce").to_numpy()
    arr = np.nan_to_num(arr, nan=1.0, posinf=TRI_OR_MAX, neginf=TRI_OR_MIN)
    arr[arr <= 0] = 1.0

    # Clamp to [0.67, 1.5] range first
    arr = np.clip(arr, TRI_OR_MIN, TRI_OR_MAX)

    # Linear scaling: OR=1.0 keeps baseline size, OR<1 shrinks, OR>1 enlarges
    return TRI_BASE_SIZE * arr

def pts_to_px(fig, pts):  # points -> pixels
    return pts * (fig.dpi / 72.0)

def tri_radius_px(fig, s_pt2: float) -> float:
    """
    Contact radius approximation for triangle: use equivalent circle radius
    r = sqrt(s/pi), slightly deflated so line touches the triangle edge visually.
    """
    r_pt = math.sqrt(max(s_pt2, 1e-9) / math.pi) * 0.95
    return pts_to_px(fig, r_pt)

# Rect/point geometry in pixel space
def closest_point_on_rect(bb, pxy):
    x = min(max(pxy[0], bb.x0), bb.x1)
    y = min(max(pxy[1], bb.y0), bb.y1)
    return np.array([x, y], dtype=float)

def rect_point_dist(bb, pxy):
    q = closest_point_on_rect(bb, pxy)
    return float(np.hypot(*(pxy - q))), q

def texts_bboxes_px(ax, texts):
    fig = ax.get_figure(); fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out=[]
    for t in texts:
        # Get bounding box directly from text, not from patch (which may be None)
        bb = t.get_window_extent(renderer=renderer).expanded(1.01, 1.06)
        out.append((t, bb))
    return out, renderer

def px_step_to_data(ax, dx_px, dy_px):
    inv = ax.transData.inverted()
    x0,y0 = ax.transData.transform((0,0))
    x1,y1 = x0+dx_px, y0+dy_px
    xd, yd = inv.transform((x1,y1)) - inv.transform((x0,y0))
    return float(xd), float(yd)

# ---------- Category mapping ----------
def load_inversion_mapping(mapping_file: str) -> dict:
    """Load inversion ID to chr:start-end mapping from balanced_recurrence_results.tsv"""
    if not os.path.exists(mapping_file):
        print(f"[WARN] Inversion mapping file not found: {mapping_file}", file=sys.stderr)
        return {}
    
    df = pd.read_csv(mapping_file, sep="\t", dtype=str)
    mapping = {}
    for _, row in df.iterrows():
        inv_id = row.get("Inversion_ID", "")
        chrom = row.get("Chromosome", "")
        start = row.get("Start", "")
        end = row.get("End", "")
        if inv_id and chrom and start and end:
            # Create chr:start-end format
            mapped_label = f"{chrom}:{start}-{end}"
            mapping[inv_id] = mapped_label
    
    return mapping

def load_category_map(phecode_csv: str) -> pd.DataFrame:
    if not os.path.exists(phecode_csv): sys.exit(f"ERROR: Cannot find {phecode_csv}")
    pc = pd.read_csv(phecode_csv, dtype=str)

    column_set = set(pc.columns)
    legacy_cols = {"phecode_string", "phecode_category", "category_num"}
    phecodex_cols = {"description", "group", "groupnum"}

    if legacy_cols.issubset(column_set):
        pass
    elif phecodex_cols.issubset(column_set):
        pc = pc.rename(
            columns={
                "description": "phecode_string",
                "group": "phecode_category",
                "groupnum": "category_num",
            }
        )
    else:
        need = sorted(legacy_cols)
        alt = sorted(phecodex_cols)
        sys.exit(
            f"ERROR: {phecode_csv} must contain columns {need} or the PheCodeX columns {alt}"
        )

    pc["clean_name"] = pc["phecode_string"].map(canonicalize_name)
    grp = pc.groupby("clean_name", dropna=False)[["phecode_category","category_num"]]
    rows=[]
    for key, sub in grp:
        pairs = list(zip(sub["phecode_category"], sub["category_num"]))
        if not pairs: continue
        cat, num = Counter(pairs).most_common(1)[0][0]
        rows.append({"clean_name": key, "phecode_category": cat, "category_num": num})
    cmap = pd.DataFrame(rows)
    cmap["category_num_num"] = pd.to_numeric(cmap["category_num"], errors="coerce")
    return cmap

# ---------- Collision resolution (second pass) ----------
def resolve_overlaps_strict(ax, texts, points_px, point_rad_px, max_iter=450, step_px=2.5):
    """
    Remove residual overlaps (label–label AND label–marker) in pixel space.
    Moves labels in both x and y by small px steps, iteratively.
    """
    if not texts: return
    fig = ax.get_figure()

    def labels_bboxes():
        fig.canvas.draw()
        return texts_bboxes_px(ax, texts)

    for _ in range(max_iter):
        moved=False
        bbs, renderer = labels_bboxes()

        # 1) label–label separation
        for i in range(len(bbs)):
            ti, bi = bbs[i]
            for j in range(i+1, len(bbs)):
                tj, bj = bbs[j]
                overlap = not (bi.x1 < bj.x0 or bi.x0 > bj.x1 or bi.y1 < bj.y0 or bi.y0 > bj.y1)
                if overlap:
                    ci = np.array([(bi.x0+bi.x1)/2.0, (bi.y0+bi.y1)/2.0])
                    cj = np.array([(bj.x0+bj.x1)/2.0, (bj.y0+bj.y1)/2.0])
                    v = ci - cj
                    if np.allclose(v, 0): v = np.array([1.0, 0.0])
                    v = v / np.linalg.norm(v)
                    dx, dy = v * step_px
                    xdi, ydi = px_step_to_data(ax, dx, dy)
                    xdj, ydj = px_step_to_data(ax, -dx, -dy)
                    xi, yi = ti.get_position(); ti.set_position((xi+xdi, yi+ydi))
                    xj, yj = tj.get_position(); tj.set_position((xj+xdj, yj+ydj))
                    moved=True

        # 2) label–marker separation (vs nearest violating marker)
        fig.canvas.draw()
        bbs, renderer = labels_bboxes()
        for t, bb in bbs:
            centers = points_px
            # distance from label bbox edge to each point center
            dists = []
            qs = []
            for c in centers:
                d, q = rect_point_dist(bb, c)
                dists.append(d); qs.append(q)
            dists = np.asarray(dists)
            # Find any violation: dist < radius + cushion
            cushion = 2.0
            viol = dists < (point_rad_px + cushion)
            if viol.any():
                k = int(np.argmin(dists - point_rad_px))  # closest offender
                # move away from offending point along outward normal (from point to label center)
                center = np.array([(bb.x0+bb.x1)/2.0, (bb.y0+bb.y1)/2.0])
                v = center - centers[k]
                if np.allclose(v, 0): v = np.array([0.0, -1.0])
                v = v / np.linalg.norm(v)
                dx, dy = v * step_px
                xd, yd = px_step_to_data(ax, dx, dy)
                x0, y0 = t.get_position()
                t.set_position((x0+xd, y0+yd))
                moved=True

        if not moved:
            break

# ---------- Connector drawing ----------
def draw_connectors(ax, ann_rows, texts, color_by_rowid, size_by_rowid):
    """
    Connector from label-box edge to triangle edge, in pixel space (exact),
    then transformed back to data coords. Color matches the triangle.
    """
    if not texts: return
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    # per-row point pixels & marker radii
    pt_px = {}
    tri_rad_px = {}
    for idx, r in ann_rows.iterrows():
        pxy = ax.transData.transform((float(r["x"]), float(r["y"])))
        pt_px[idx] = np.array(pxy)
        size_val = float(size_by_rowid.get(idx, TRI_BASE_SIZE))
        tri_rad_px[idx] = tri_radius_px(fig, size_val)

    for t in texts:
        rowid = getattr(t, "_rowid", None)
        if rowid is None or rowid not in pt_px: continue

        # Get bounding box directly from text (no patch since we removed bbox)
        bb = t.get_window_extent(renderer=renderer)

        p = pt_px[rowid]
        q = closest_point_on_rect(bb, p)

        v = p - q; L = np.linalg.norm(v)
        if L < 1e-6: v = np.array([0.0, -1.0]); L = 1.0
        e = p - (v / L) * tri_rad_px[rowid]  # triangle edge

        qd = inv.transform(q)
        ed = inv.transform(e)
        color = color_by_rowid[rowid]
        ax.add_patch(FancyArrowPatch(
            posA=qd, posB=ed, arrowstyle="-", mutation_scale=1,
            linewidth=1.0, color=color, zorder=3.2, shrinkA=0.0, shrinkB=0.0
        ))

# ---------- Plot per inversion ----------
def plot_one_inversion(
    df_group: pd.DataFrame,
    inversion_label: str,
    global_ymin: float | None = None,
    global_ymax: float | None = None,
    global_xlim: tuple[float, float] | None = None,
    global_fig_width: float | None = None,
    global_xrange: float | None = None,
) -> str | None:
    g = df_group.copy()
    g[P_Q_COL] = pd.to_numeric(g[P_Q_COL], errors="coerce")
    g[BETA_COL] = pd.to_numeric(g[BETA_COL], errors="coerce")
    g[OR_COL]   = pd.to_numeric(g[OR_COL], errors="coerce")

    g = g[g[PHENO_COL].notna() & g[P_Q_COL].notna()]
    if g.empty: return None

    tiny = np.nextafter(0, 1)
    g.loc[g[P_Q_COL] <= 0, P_Q_COL] = tiny

    # display
    g["Phen_display"] = g[PHENO_COL].map(pretty_text)
    g["Phen_wrapped"] = g["Phen_display"].map(lambda s: balanced_linebreak(s, MIN_WORDS_SIDE))
    g["y"] = -np.log10(g[P_Q_COL])
    g["risk_dir"] = np.where(g[BETA_COL].fillna(0) >= 0, "inc", "dec")

    # categories
    g["cat_name"] = g["phecode_category"].fillna(UNCAT_NAME)
    g["cat_num"]  = g["category_num_num"].fillna(9999)

    cat_counts = g.groupby("cat_name").size()
    cat_meta = (
        g[["cat_name", "cat_num"]]
        .drop_duplicates()
        .set_index("cat_name")
    )
    cat_info: list[tuple[str, int, float]] = []
    for cat, count in cat_counts.items():
        meta_val = float(cat_meta.at[cat, "cat_num"]) if cat in cat_meta.index else 9999.0
        cat_info.append((cat, int(count), meta_val))
    cat_info.sort(key=lambda x: (x[1], x[2], x[0]))

    cat_order: list[str] = []
    lo, hi = 0, len(cat_info) - 1
    take_small = True
    while lo <= hi:
        if take_small:
            cat_order.append(cat_info[lo][0])
            lo += 1
        else:
            cat_order.append(cat_info[hi][0])
            hi -= 1
        take_small = not take_small

    # x positions within category:
    # left side (dec) sorted by q; right side (inc) sorted by q
    pieces, centers, ticklabels = [], [], []
    start = 0
    cat_order_plotted: list[str] = []
    for cat in cat_order:
        cat_df = g[g["cat_name"] == cat].copy()
        dec_df = cat_df[cat_df["risk_dir"] == "dec"].sort_values(P_Q_COL, kind="mergesort")
        inc_df = cat_df[cat_df["risk_dir"] == "inc"].sort_values(P_Q_COL, kind="mergesort")

        n_dec = len(dec_df)
        n_inc = len(inc_df)
        n_tot = n_dec + n_inc

        # Left block: dec
        if n_dec > 0:
            dec_df = dec_df.copy()
            dec_df["x"] = np.arange(start, start + n_dec, dtype=float)

        # Right block: inc
        if n_inc > 0:
            inc_df = inc_df.copy()
            inc_df["x"] = np.arange(start + n_dec, start + n_tot, dtype=float)

        block = pd.concat([dec_df, inc_df], axis=0)
        if not block.empty:
            pieces.append(block)
            centers.append(start + (n_tot - 1)/2.0)
            ticklabels.append(cat)
            start += n_tot
            cat_order_plotted.append(cat)

    if not pieces: return None
    g = pd.concat(pieces, ignore_index=False).sort_values("x")
    m = len(g)

    if SIG_COL in g.columns:
        sig_mask_full = truthy_series(g[SIG_COL])
    else:
        sig_mask_full = pd.Series(False, index=g.index)

    base_color_lookup = {"inc": INCOLOR_HEX, "dec": DECOLOR_HEX}
    plot_colors: list[str] = []
    for idx, row in g.iterrows():
        base_color = base_color_lookup.get(row.get("risk_dir"), INCOLOR_HEX)
        if bool(sig_mask_full.get(idx, False)):
            plot_colors.append(darken_color(base_color, SIG_DARKEN))
        else:
            plot_colors.append(lighten_color(base_color, NON_SIG_LIGHTEN))
    g["plot_color"] = plot_colors

    # Scale ALL points by odds ratio (not just significant ones)
    size_array = scale_all_sizes(g[OR_COL])
    g["plot_size"] = size_array

    # figure
    if global_fig_width is not None:
        fig_w = float(global_fig_width)
    else:
        fig_w = compute_width(m)
    fig = plt.figure(figsize=(fig_w, FIG_HEIGHT))
    ax = fig.add_axes(AXES_BBOX)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    obstacles = []
    inc = g["risk_dir"] == "inc"
    dec = ~inc
    inc_sig = inc & sig_mask_full
    inc_nonsig = inc & ~sig_mask_full
    dec_sig = dec & sig_mask_full
    dec_nonsig = dec & ~sig_mask_full

    tri_inc_sig = tri_inc_nonsig = tri_dec_sig = tri_dec_nonsig = None

    if inc_sig.any():
        tri_inc_sig = ax.scatter(
            g.loc[inc_sig, "x"], g.loc[inc_sig, "y"],
            s=g.loc[inc_sig, "plot_size"], marker="^",
            c=g.loc[inc_sig, "plot_color"], edgecolors="black",
            linewidths=POINT_EDGE_LW, alpha=POINT_ALPHA_SIG, zorder=2.0,
            label="Risk increasing"
        )
        obstacles.append(tri_inc_sig)
    if inc_nonsig.any():
        tri_inc_nonsig = ax.scatter(
            g.loc[inc_nonsig, "x"], g.loc[inc_nonsig, "y"],
            s=g.loc[inc_nonsig, "plot_size"], marker="^",
            c=g.loc[inc_nonsig, "plot_color"], edgecolors="black",
            linewidths=POINT_EDGE_LW, alpha=POINT_ALPHA_NONSIG, zorder=2.0,
            label="Risk increasing" if tri_inc_sig is None else None
        )
        obstacles.append(tri_inc_nonsig)

    if dec_sig.any():
        tri_dec_sig = ax.scatter(
            g.loc[dec_sig, "x"], g.loc[dec_sig, "y"],
            s=g.loc[dec_sig, "plot_size"], marker="v",
            c=g.loc[dec_sig, "plot_color"], edgecolors="black",
            linewidths=POINT_EDGE_LW, alpha=POINT_ALPHA_SIG, zorder=2.0,
            label="Risk decreasing"
        )
        obstacles.append(tri_dec_sig)
    if dec_nonsig.any():
        tri_dec_nonsig = ax.scatter(
            g.loc[dec_nonsig, "x"], g.loc[dec_nonsig, "y"],
            s=g.loc[dec_nonsig, "plot_size"], marker="v",
            c=g.loc[dec_nonsig, "plot_color"], edgecolors="black",
            linewidths=POINT_EDGE_LW, alpha=POINT_ALPHA_NONSIG, zorder=2.0,
            label="Risk decreasing" if tri_dec_sig is None else None
        )
        obstacles.append(tri_dec_nonsig)

    # establish consistent x-limits prior to layout/annotation work
    raw_xmin = float(g["x"].min())
    raw_xmax = float(g["x"].max())
    if global_xlim is not None:
        base_xmin, base_xmax = map(float, global_xlim)
    else:
        base_xmin, base_xmax = raw_xmin, raw_xmax
    if not np.isfinite(base_xmin) or not np.isfinite(base_xmax):
        base_xmin, base_xmax = raw_xmin, raw_xmax
    if base_xmax <= base_xmin:
        base_xmax = base_xmin + 1.0
    ax.set_xlim(base_xmin, base_xmax)
    fig.canvas.draw()
    xpad_data = px_step_to_data(ax, X_PAD_PX, 0)[0]
    final_xmin = base_xmin - xpad_data
    final_xmax = base_xmax + xpad_data
    ax.set_xlim(final_xmin, final_xmax)
    fig.canvas.draw()

    # annotations: q < 0.1 OR FDR significant
    annotate_mask = (g[P_Q_COL] < ANNOTATE_Q_THRESH)
    if SIG_COL in g.columns: annotate_mask |= truthy_series(g[SIG_COL])
    ann_rows = g.loc[annotate_mask].sort_values([ "cat_num", "x" ])

    # initial placement: natural side
    texts=[]
    # side-aware offset (right for inc, left for dec)
    if global_xrange is not None and np.isfinite(global_xrange):
        x_range = float(global_xrange)
    else:
        x_range = float(g["x"].max() - g["x"].min()) if m > 1 else 1.0
    x_range = max(1.0, x_range)
    dx_side = 0.02 * x_range
    for idx, r in ann_rows.iterrows():
        place_right = (r["risk_dir"] == "inc")
        x0 = r["x"] + (dx_side if place_right else -dx_side)
        ha = "left" if place_right else "right"
        t = ax.text(
            x0, r["y"], balanced_linebreak(r["Phen_wrapped"]),
            fontsize=LABEL_FONTSZ, ha=ha, va="bottom", zorder=3.6,
            bbox=None  # Remove box around labels
        )
        t._rowid = idx  # persistent binding for connectors
        texts.append(t)

    # let adjustText move labels freely (natural)
    if texts:
        ADJUST_TEXT(
            texts, ax=ax,
            expand_text=ADJ_EXPAND_TEXT,
            expand_points=ADJ_EXPAND_PNTS,
            force_points=ADJ_FORCE_PNTS,
            add_objects=[ob for ob in obstacles if ob is not None],
            arrowprops=None
        )

    # strict second pass: remove any residual overlaps (labels vs labels and vs markers)
    fig.canvas.draw()
    # per-point px centers and collision radii (max of triangle & circle if sig)
    pts_px = []
    rad_px = []
    tri_radius_by_rowid = {idx: tri_radius_px(fig, float(g.at[idx, "plot_size"])) for idx in g.index}
    for i, r in g.iterrows():
        px = ax.transData.transform((float(r["x"]), float(r["y"])))
        pts_px.append(np.array(px))
        tri_r = tri_radius_by_rowid.get(i, tri_radius_px(fig, TRI_BASE_SIZE))
        rad_px.append(tri_r)
    pts_px = np.vstack(pts_px)
    rad_px = np.array(rad_px)
    resolve_overlaps_strict(ax, texts, pts_px, rad_px, max_iter=450, step_px=2.5)

    # margins & headroom (x-limits already established above)

    if global_ymax is not None and np.isfinite(global_ymax):
        base_min = float(global_ymin) if (global_ymin is not None and np.isfinite(global_ymin)) else 0.0
        y_bottom = min(0.0, base_min)
        margin = abs(float(global_ymax)) * 0.05
        y_top = float(global_ymax) + margin
        if not np.isfinite(y_top):
            y_top = float(global_ymax)
        if y_top <= y_bottom:
            fallback = abs(y_bottom) * 0.05
            if fallback <= 0:
                fallback = 0.05
            y_top = y_bottom + fallback
        ax.set_ylim(y_bottom, y_top)
    else:
        ymin, ymax = g["y"].min(), g["y"].max()
        ax.set_ylim(ymin, ymax + max(0.25, (ymax - ymin) * Y_TOP_PAD_FRAC))

    # q = 0.05 reference line
    q05_y = -math.log10(0.05)
    ax.axhline(q05_y, color="#666666", linestyle="--", linewidth=1.0, label="q = 0.05")

    # axes / ticks / title
    ax.set_title(str(inversion_label), fontsize=TITLE_FONTSZ, pad=10, fontweight="semibold")
    ax.set_ylabel(r"$-\log_{10}(q)$", fontsize=AX_LABEL_FONTSZ)
    ax.set_xticks(centers)
    ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=TICK_FONTSZ)
    ax.tick_params(axis="x", pad=3, labelsize=TICK_FONTSZ)
    ax.tick_params(axis="y", labelsize=TICK_FONTSZ)

    # category separators
    cum = np.cumsum([len(g[g["cat_name"] == c]) for c in cat_order_plotted])
    for x0 in cum[:-1]:
        ax.axvline(x=x0 - 0.5, color="#e6e6ee", linestyle="-", linewidth=0.7, zorder=1)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    legend1 = None
    if handles:
        legend1 = ax.legend(handles, labels, fontsize=LEGEND_FONTSZ, loc="upper right", frameon=False)

    or_levels = [0.67, 1.0, 1.5]
    or_labels = [f"{val:.2f}" if val < 1.0 else f"{val:.1f}" for val in or_levels]
    sample_color = darken_color(INCOLOR_HEX, SIG_DARKEN)
    size_handles = [
        ax.scatter(
            [], [],
            s=scale_all_sizes(pd.Series([val]))[0],
            marker="^", facecolors=sample_color,
            edgecolors="black", linewidths=POINT_EDGE_LW,
            alpha=POINT_ALPHA_SIG
        )
        for val in or_levels
    ]
    legend2 = ax.legend(
        size_handles, or_labels,
        title="Odds ratio",
        fontsize=LEGEND_FONTSZ, title_fontsize=LEGEND_TITLE_SZ,
        loc="upper left", frameon=False,
        borderaxespad=0.8
    )
    if legend1 is not None:
        ax.add_artist(legend1)
    ax.add_artist(legend2)

    # connectors (AFTER final layout; color by exact rowid)
    fig.canvas.draw()
    color_by_rowid = g["plot_color"].to_dict()
    size_by_rowid = g["plot_size"].to_dict()
    draw_connectors(ax, ann_rows, texts, color_by_rowid, size_by_rowid)

    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, f"phewas_{sanitize_filename(str(inversion_label))}")
    # Save both PDF and PNG
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.5)
    fig.savefig(f"{base}.png", format="png", dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    return base

# ---------- Main ----------
def main():
    # Check for input file in current directory first, then try data/ directory
    infile_path = INFILE
    if not os.path.exists(infile_path):
        # Try alternative location (current directory if INFILE has data/ prefix, or data/ prefix if it doesn't)
        if infile_path.startswith("data/"):
            alt_path = infile_path.replace("data/", "", 1)
        else:
            alt_path = os.path.join("data", infile_path)

        if os.path.exists(alt_path):
            infile_path = alt_path
        else:
            sys.exit(f"ERROR: Cannot find {INFILE} or {alt_path}")

    ensure_phecode_file(PHECODE_FILE, PHECODE_URL)

    if not _ADJUST_TEXT_AVAILABLE:
        print(
            "[WARN] adjustText package not available; proceeding without label adjustment.",
            file=sys.stderr,
        )

    df = pd.read_csv(infile_path, sep="\t", dtype=str)
    for col in [PHENO_COL, INV_COL, P_Q_COL]:
        if col not in df.columns: sys.exit(f"ERROR: {INFILE} missing required column '{col}'")

    # Merge categories
    cmap = load_category_map(PHECODE_FILE)
    df["Phen_clean"] = df[PHENO_COL].map(canonicalize_name)
    df = df.merge(
        cmap[["clean_name","phecode_category","category_num","category_num_num"]],
        how="left", left_on="Phen_clean", right_on="clean_name"
    )

    df[INV_COL] = df[INV_COL].fillna("").astype(str)
    inv_mask = df[INV_COL].str.strip() != ""
    df = df[inv_mask].copy()
    if df.empty: sys.exit("No rows with a non-empty Inversion value.")

    df[INV_COL] = map_inversion_series(df[INV_COL])
    
    # Load inversion ID mapping for proper titles
    inv_mapping = load_inversion_mapping(INV_MAPPING_FILE)

    q_numeric_all = pd.to_numeric(df[P_Q_COL], errors="coerce")
    valid_mask = df[PHENO_COL].notna() & q_numeric_all.notna()
    if valid_mask.any():
        tiny = np.nextafter(0, 1)
        q_valid = q_numeric_all.loc[valid_mask].astype(float).copy()
        q_valid[q_valid <= 0] = tiny
        y_vals = -np.log10(q_valid)
        global_ymin = float(y_vals.min()) if not y_vals.empty else None
        global_ymax = float(y_vals.max()) if not y_vals.empty else None
    else:
        global_ymin = None
        global_ymax = None

    counts_series = (
        df.loc[valid_mask]
        .groupby(INV_COL, dropna=False)
        .size()
    )
    if not counts_series.empty:
        max_points = int(counts_series.max())
        if max_points > 0:
            xmax_val = float(max_points - 1)
            global_xlim = (0.0, xmax_val)
            global_fig_width = compute_width(max_points)
            global_xrange = xmax_val - 0.0
        else:
            global_xlim = None
            global_fig_width = None
            global_xrange = None
    else:
        global_xlim = None
        global_fig_width = None
        global_xrange = None

    made, to_open = [], []
    for inv, grp in df.groupby(INV_COL, dropna=False):
        # Use mapped label if available, otherwise use original
        display_label = inv_mapping.get(inv, inv)
        out = plot_one_inversion(
            grp,
            inversion_label=display_label,
            global_ymin=global_ymin,
            global_ymax=global_ymax,
            global_xlim=global_xlim,
            global_fig_width=global_fig_width,
            global_xrange=global_xrange,
        )
        if out:
            made.append(out)
            if (SIG_COL in grp.columns) and truthy_series(grp[SIG_COL]).any():
                to_open.append(out)

    if not made:
        print("No plots produced (no valid phenotypes or Q_GLOBAL values)."); return

    print(f"Wrote {len(made)} PDF(s) to: {OUTDIR}")
    for p in made: print("  -", p)

    if to_open:
        print(f"Auto-opening {len(to_open)} plot(s) with ≥1 FDR significant hit:")
        for p in to_open:
            print("    *", p); open_file(p)
    else:
        print("No inversions had an FDR significant hit — nothing auto-opened.")

if __name__ == "__main__":
    main()

