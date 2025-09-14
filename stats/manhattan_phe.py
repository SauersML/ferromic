import os, re, sys, math, subprocess
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from adjustText import adjust_text as ADJUST_TEXT

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
FIG_HEIGHT      = 7.6

# Markers & style
TRI_SIZE        = 80.0     # triangle area (pt^2)
CIRCLE_SIZE     = 300.0    # FDR circle area (pt^2)
POINT_EDGE_LW   = 0.45
POINT_ALPHA     = 0.98
CIRCLE_EDGE_LW  = 1.3

# Label/legend
LABEL_FONTSZ    = 9.2
AX_LABEL_FONTSZ = 12
TICK_FONTSZ     = 10.5
TITLE_FONTSZ    = 15
ANNOTATE_Q_THRESH = 0.1

# Linebreak
MIN_WORDS_BREAK = 6
MIN_WORDS_SIDE  = 3

# Inner margins & headroom
X_PAD_PX       = 18        # add ~px padding left/right (converted to data)
Y_TOP_PAD_FRAC = 0.08

# adjustText tuning
ADJ_EXPAND_TEXT = (1.06, 1.26)
ADJ_EXPAND_PNTS = (1.03, 1.14)
ADJ_FORCE_PNTS  = (0.07, 0.30)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.6,
    "legend.frameon": False,
})

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
    Visual contact radius for triangle given scatter 's' (pt^2).
    Use equivalent circle radius r = sqrt(s/pi) with slight deflate to meet edge.
    """
    r_pt = math.sqrt(max(s_pt2, 1e-9) / math.pi) * 0.95   # tuned factor
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
def resolve_overlaps_strict(ax, texts, points_px, point_rad_px, max_iter=400, step_px=2.5):
    """
    Remove any residual overlaps:
      - label–label (bbox vs bbox)
      - label–marker (bbox vs circle of radius point_rad_px around each point)
    Move labels in pixel space (both x and y), smallest nudges first.
    """
    if not texts: return
    fig = ax.get_figure()
    inv = ax.transData.inverted()

    for _ in range(max_iter):
        fig.canvas.draw()
        bbs, renderer = texts_bboxes_px(ax, texts)
        moved=False

        # 1) label–label
        for i in range(len(bbs)):
            ti, bi = bbs[i]
            for j in range(i+1, len(bbs)):
                tj, bj = bbs[j]
                overlap = not (bi.x1 < bj.x0 or bi.x0 > bj.x1 or bi.y1 < bj.y0 or bi.y0 > bj.y1)
                if overlap:
                    # push apart horizontally by step_px toward opposite directions
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

        # 2) label–marker
        fig.canvas.draw()
        bbs, renderer = texts_bboxes_px(ax, texts)
        for t, bb in bbs:
            # nearest point
            centers = points_px
            dists = np.hypot(centers[:,0]- (bb.x0+bb.x1)/2.0, centers[:,1]- (bb.y0+bb.y1)/2.0)
            k = int(np.argmin(dists))
            dist_to_edge, q = rect_point_dist(bb, centers[k])
            if dist_to_edge < point_rad_px[k] + 2.0:  # 2px cushion
                # move label away from the point along outward normal
                v = ( (bb.x0+bb.x1)/2.0 - centers[k][0], (bb.y0+bb.y1)/2.0 - centers[k][1] )
                if np.allclose(v, 0): v = (0.0, -1.0)
                vx, vy = np.array(v) / np.linalg.norm(v)
                dx, dy = vx*step_px, vy*step_px
                xd, yd = px_step_to_data(ax, dx, dy)
                x0, y0 = t.get_position()
                t.set_position((x0+xd, y0+yd))
                moved=True

        if not moved:
            break

# ---------- Connector drawing ----------
def draw_connectors(ax, ann_rows, texts, color_by_rowid, tri_size_pt2):
    """
    For each label, draw a connector from the label-box edge to the triangle edge,
    computed in pixel space and transformed back to data.
    """
    if not texts: return
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    # Precompute point pixels and triangle radii per annotated row
    pt_px = {}
    tri_rad_px = {}
    for idx, r in ann_rows.iterrows():
        pxy = ax.transData.transform((float(r["x"]), float(r["y"])))
        pt_px[idx] = np.array(pxy)
        tri_rad_px[idx] = tri_radius_px(fig, tri_size_pt2)

    for t in texts:
        rowid = getattr(t, "_rowid", None)
        if rowid is None or rowid not in pt_px:
            continue

        # label bbox in pixels (use actual drawn patch)
        patch = t.get_bbox_patch()
        bb = patch.get_window_extent(renderer=renderer)

        # closest contact point on label box to the point
        p = pt_px[rowid]
        q = closest_point_on_rect(bb, p)

        # move along vector from label to point, stopping at triangle edge
        v = p - q; L = np.linalg.norm(v)
        if L < 1e-6:
            v = np.array([0.0, -1.0]); L = 1.0
        e = p - (v / L) * tri_rad_px[rowid]   # triangle-edge contact

        # Convert to data coords and draw
        qd = inv.transform(q)
        ed = inv.transform(e)
        color = color_by_rowid[rowid]
        ax.add_patch(FancyArrowPatch(
            posA=qd, posB=ed, arrowstyle="-", mutation_scale=1,
            linewidth=1.0, color=color, zorder=3.2, shrinkA=0.0, shrinkB=0.0
        ))

# ---------- Plotting per inversion ----------
def plot_one_inversion(df_group: pd.DataFrame, inversion_label: str) -> str | None:
    g = df_group.copy()
    g[P_Q_COL] = pd.to_numeric(g[P_Q_COL], errors="coerce")
    g[BETA_COL] = pd.to_numeric(g[BETA_COL], errors="coerce")
    g[OR_COL]   = pd.to_numeric(g[OR_COL], errors="coerce")

    g = g[g[PHENO_COL].notna() & g[P_Q_COL].notna()]
    if g.empty: return None

    tiny = np.nextafter(0, 1)
    g.loc[g[P_Q_COL] <= 0, P_Q_COL] = tiny

    # display fields
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

    # shade by |log(OR)| normalized (p95)
    or_vals = g[OR_COL].fillna(1.0).astype(float).clip(lower=np.nextafter(0,1))
    mag = np.abs(np.log(or_vals))
    p95 = np.nanpercentile(mag, 95) if np.isfinite(np.nanpercentile(mag, 95)) else 1.0
    denom = p95 if p95 > 0 else (mag.max() if mag.max() > 0 else 1.0)
    norm_all = np.clip(mag / denom, 0, 1)

    # x positions grouped by category
    pieces, centers, ticklabels = [], [], []
    start=0
    for cat in cat_order:
        block = g[g["cat_name"]==cat].sort_values([P_Q_COL,"Phen_display"], kind="mergesort").copy()
        n=len(block)
        block["x"] = np.arange(start, start+n, dtype=float)
        idxs = block.index.tolist()
        block["color"] = [shade_with_norm(cat_to_base[cat], float(norm_all.loc[idx])) for idx in idxs]
        pieces.append(block)
        centers.append(start + (n-1)/2.0)
        ticklabels.append(cat)
        start += n
    g = pd.concat(pieces, ignore_index=False).sort_values("x")
    m = len(g)

    # figure
    fig_w = compute_width(m)
    fig, ax = plt.subplots(figsize=(fig_w, FIG_HEIGHT))
    ax.set_facecolor("#ffffff")

    obstacles = []
    # FDR circles
    circ = None
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

    # annotations: q < 0.1 OR FDR significant
    annotate_mask = (g[P_Q_COL] < ANNOTATE_Q_THRESH)
    if SIG_COL in g.columns: annotate_mask |= truthy_series(g[SIG_COL])
    ann_rows = g.loc[annotate_mask].sort_values("x")
    texts=[]
    # initial small horizontal offsets alternating
    x_range = (g["x"].max() - g["x"].min()) if m>1 else 1.0
    dx = 0.02 * max(1.0, x_range)
    for i, (idx, r) in enumerate(ann_rows.iterrows()):
        x0 = r["x"] + (dx if (i % 2 == 0) else -dx)
        ha = "left" if (i % 2 == 0) else "right"
        t = ax.text(
            x0, r["y"], balanced_linebreak(r["Phen_wrapped"]),
            fontsize=LABEL_FONTSZ, ha=ha, va="bottom", zorder=3.6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#333333", linewidth=0.35, alpha=0.98)
        )
        t._rowid = idx  # bind permanently for exact pairing
        texts.append(t)

    # let adjustText move in both axes (no constraints)
    if texts:
        ADJUST_TEXT(
            texts, ax=ax,
            expand_text=ADJ_EXPAND_TEXT,
            expand_points=ADJ_EXPAND_PNTS,
            force_points=ADJ_FORCE_PNTS,
            add_objects=[ob for ob in obstacles if ob is not None],
            arrowprops=None
        )

    # second-pass strict overlap resolver (labels vs labels AND labels vs markers)
    # prepare point centers (px) and radii (px) for all points
    fig.canvas.draw()
    pts_px = []
    rad_px = []
    for _, r in g.iterrows():
        px = ax.transData.transform((float(r["x"]), float(r["y"])))
        pts_px.append(np.array(px))
        rad_px.append(tri_radius_px(fig, TRI_SIZE))
    pts_px = np.vstack(pts_px)
    rad_px = np.array(rad_px)

    resolve_overlaps_strict(ax, texts, pts_px, rad_px, max_iter=400, step_px=2.0)

    # after all moves & any layout, finalize limits/margins and compute connectors last
    # inner margin in pixels → data units
    x0_px, _ = ax.transData.transform((0, 0))
    xpad_data = px_step_to_data(ax, X_PAD_PX, 0)[0]
    xmin, xmax = g["x"].min(), g["x"].max()
    ax.set_xlim(xmin - xpad_data, xmax + xpad_data)

    ymin, ymax = g["y"].min(), g["y"].max()
    ax.set_ylim(ymin, ymax + max(0.25, (ymax - ymin) * Y_TOP_PAD_FRAC))

    # axes / ticks / title
    ax.set_title(str(inversion_label), fontsize=TITLE_FONTSZ, pad=10)
    ax.set_ylabel(r"$-\log_{10}(q)$", fontsize=AX_LABEL_FONTSZ)
    ax.set_xticks(centers)
    ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=TICK_FONTSZ)
    ax.tick_params(axis="x", pad=3)

    # category separators
    cum = np.cumsum([len(g[g["cat_name"]==c]) for c in cat_order])
    for x0 in cum[:-1]:
        ax.axvline(x=x0 - 0.5, color="#e6e6ee", linestyle="-", linewidth=0.7, zorder=1)

    # legend
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(fontsize=9, loc="upper right")

    # connectors (color by exact rowid)
    color_by_rowid = g["color"].to_dict()
    fig.canvas.draw()  # ensure final renderer
    draw_connectors(ax, ann_rows, texts, color_by_rowid, tri_size_pt2=TRI_SIZE)

    fig.tight_layout()
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

    inv_mask = df[INV_COL].astype(str).str.strip() != ""
    df = df[inv_mask].copy()
    if df.empty: sys.exit("No rows with a non-empty Inversion value.")

    made, to_open = [], []
    for inv, grp in df.groupby(INV_COL, dropna=False):
        out = plot_one_inversion(grp, inversion_label=inv)
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
