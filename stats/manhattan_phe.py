import os, re, sys, math, subprocess
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from adjustText import adjust_text as ADJUST_TEXT  # required
matplotlib.use('Agg')

from _inv_common import map_inversion_series

# ---------- Config ----------
INFILE = "phewas_results.tsv"
PHECODE_FILE = "phecodeX.csv"
OUTDIR = "phewas_plots"

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
AXES_BBOX       = (0.18, 0.14, 0.64, 0.78)  # left, bottom, width, height

# Markers & style
TRI_SIZE        = 82.0     # triangle area (pt^2)
CIRCLE_SIZE     = 300.0    # FDR circle area (pt^2)
POINT_EDGE_LW   = 0.45
POINT_ALPHA     = 0.98
CIRCLE_EDGE_LW  = 1.3

# Label/legend
LABEL_FONTSZ    = 14.2
AX_LABEL_FONTSZ = 12
TICK_FONTSZ     = 10.5
TITLE_FONTSZ    = 15
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
def build_palette(cat_order):
    okabe_ito = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999",
    ]
    tableau10 = list(mcolors.TABLEAU_COLORS.values())
    tab20 = [mcolors.to_hex(c) for c in plt.cm.tab20.colors]
    base = okabe_ito + tableau10 + tab20
    if len(base) < len(cat_order):
        def lighten(h, amt=0.25):
            r,g,b = mcolors.to_rgb(h)
            r=min(1,r+(1-r)*amt); g=min(1,g+(1-g)*amt); b=min(1,b+(1-b)*amt)
            return mcolors.to_hex((r,g,b))
        base += [lighten(c) for c in base]
    return {c: base[i % len(base)] for i, c in enumerate(cat_order)}

def shade_with_norm(base_hex: str, norm: float, l_light=0.86, l_dark=0.28) -> str:
    r,g,b = mcolors.to_rgb(base_hex)
    import colorsys
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    l_new = l_light - norm * (l_light - l_dark)
    s_new = max(0.52, s)
    r2,g2,b2 = colorsys.hls_to_rgb(h,l_new,s_new)
    return mcolors.to_hex((r2,g2,b2))

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
        patch = t.get_bbox_patch()
        bb = patch.get_window_extent(renderer=renderer).expanded(1.01, 1.06)
        out.append((t, bb))
    return out, renderer

def px_step_to_data(ax, dx_px, dy_px):
    inv = ax.transData.inverted()
    x0,y0 = ax.transData.transform((0,0))
    x1,y1 = x0+dx_px, y0+dy_px
    xd, yd = inv.transform((x1,y1)) - inv.transform((x0,y0))
    return float(xd), float(yd)

# ---------- Category mapping ----------
def load_category_map(phecode_csv: str) -> pd.DataFrame:
    if not os.path.exists(phecode_csv): sys.exit(f"ERROR: Cannot find {phecode_csv}")
    pc = pd.read_csv(phecode_csv, dtype=str)
    need = {"phecode_string","phecode_category","category_num"}
    if not need.issubset(set(pc.columns)):
        sys.exit(f"ERROR: {phecode_csv} must contain {sorted(need)}")
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
def draw_connectors(ax, ann_rows, texts, color_by_rowid, tri_size_pt2):
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
        tri_rad_px[idx] = tri_radius_px(fig, tri_size_pt2)

    for t in texts:
        rowid = getattr(t, "_rowid", None)
        if rowid is None or rowid not in pt_px: continue

        patch = t.get_bbox_patch()
        bb = patch.get_window_extent(renderer=renderer)

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
    cat_order = (
        g[["cat_name","cat_num"]]
        .drop_duplicates()
        .sort_values(["cat_num","cat_name"], kind="mergesort")
        .reset_index(drop=True)
    )["cat_name"].tolist()
    cat_to_base = build_palette(cat_order)

    # shade by |log(OR)| normalized
    or_vals = g[OR_COL].fillna(1.0).astype(float).clip(lower=np.nextafter(0,1))
    mag = np.abs(np.log(or_vals))
    p95 = np.nanpercentile(mag, 95) if np.isfinite(np.nanpercentile(mag, 95)) else 1.0
    denom = p95 if p95 > 0 else (mag.max() if mag.max() > 0 else 1.0)
    norm_all = np.clip(mag / denom, 0, 1)

    # x positions within category:
    # left side (dec) sorted by q; right side (inc) sorted by q
    pieces, centers, ticklabels = [], [], []
    start = 0
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
            # color shading (per point) using category hue
            base = cat_to_base[cat]
            idxs = block.index.tolist()
            block["color"] = [shade_with_norm(base, float(norm_all.loc[i])) for i in idxs]
            pieces.append(block)
            centers.append(start + (n_tot - 1)/2.0)
            ticklabels.append(cat)
            start += n_tot

    if not pieces: return None
    g = pd.concat(pieces, ignore_index=False).sort_values("x")
    m = len(g)

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
    # FDR circles (behind triangles)
    circ=None
    if SIG_COL in g.columns:
        sig = truthy_series(g[SIG_COL])
        if sig.any():
            circ = ax.scatter(
                g.loc[sig,"x"], g.loc[sig,"y"],
                s=CIRCLE_SIZE, marker="o",
                facecolors="white", edgecolors="black",
                linewidths=CIRCLE_EDGE_LW, zorder=1.5, alpha=1.0,
                label="FDR significant"
            )
            obstacles.append(circ)

    # triangles
    inc = g["risk_dir"]=="inc"
    dec = ~inc
    tri_inc = ax.scatter(
        g.loc[inc,"x"], g.loc[inc,"y"],
        s=TRI_SIZE, marker="^",
        c=g.loc[inc,"color"], edgecolors="black",
        linewidths=POINT_EDGE_LW, alpha=POINT_ALPHA, zorder=2.0,
        label="Risk increasing"
    ) if inc.any() else None
    tri_dec = ax.scatter(
        g.loc[dec,"x"], g.loc[dec,"y"],
        s=TRI_SIZE, marker="v",
        c=g.loc[dec,"color"], edgecolors="black",
        linewidths=POINT_EDGE_LW, alpha=POINT_ALPHA, zorder=2.0,
        label="Risk decreasing"
    ) if dec.any() else None
    if tri_inc is not None: obstacles.append(tri_inc)
    if tri_dec is not None: obstacles.append(tri_dec)

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
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#333333", linewidth=0.35, alpha=0.4)
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
    sig_mask_full = truthy_series(g[SIG_COL]) if (SIG_COL in g.columns) else pd.Series(False, index=g.index)
    tri_r = tri_radius_px(fig, TRI_SIZE)
    circ_r = tri_radius_px(fig, CIRCLE_SIZE)  # consistent scale conversion
    for i, r in g.iterrows():
        px = ax.transData.transform((float(r["x"]), float(r["y"])))
        pts_px.append(np.array(px))
        rad_px.append(max(tri_r, circ_r if bool(sig_mask_full.get(i, False)) else tri_r))
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
    cum = np.cumsum([len(g[g["cat_name"]==c]) for c in cat_order])
    for x0 in cum[:-1]:
        ax.axvline(x=x0 - 0.5, color="#e6e6ee", linestyle="-", linewidth=0.7, zorder=1)

    # legend
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(fontsize=9, loc="upper right", frameon=False)

    # connectors (AFTER final layout; color by exact rowid)
    fig.canvas.draw()
    color_by_rowid = g["color"].to_dict()
    draw_connectors(ax, ann_rows, texts, color_by_rowid, tri_size_pt2=TRI_SIZE)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"phewas_{sanitize_filename(str(inversion_label))}.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out

# ---------- Main ----------
def main():
    if not os.path.exists(INFILE): sys.exit(f"ERROR: Cannot find {INFILE}")
    if not os.path.exists(PHECODE_FILE): sys.exit(f"ERROR: Cannot find {PHECODE_FILE}")

    df = pd.read_csv(INFILE, sep="\t", dtype=str)
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
        out = plot_one_inversion(
            grp,
            inversion_label=inv,
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
