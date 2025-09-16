import os
import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------- Configuration ---------------------------

INPUT_FILE       = "phewas_results.tsv"
OUT_PDF          = "phewas_forest.pdf"
OUT_PNG          = "phewas_forest.png"

# Layout: explicit box model (data-units on the y-axis)
HEADER_BOX_H     = 1.80   # height of the header box per inversion
ROW_BOX_H        = 1.70   # height of each phenotype row box
BLOCK_GAP_H      = 0.50   # vertical gap after a section (breathing room)

# Panel widths: [label panel, main plot panel]
LEFT_RIGHT       = (0.4, 0.6)

# Text & styling
WRAP_WIDTH       = 42      # phenotype label wrapping width (characters)
POINT_SIZE_PT2   = 14.0    # tiny, uniform point size (pt^2)
POINT_EDGE_LW    = 0.55
GRID_ALPHA       = 0.28
BAND_ALPHA       = 0.06
HEADER_UL_ALPHA  = 0.16

# Header label placement (initial seed) and movement limits
HEADER_X_SHIFT        = 0.08      # x-position (axes-fraction on left panel) for header text
HEADER_TOP_PAD_FRAC   = 0.10      # initial offset *below* the header-box top edge (fraction of HEADER_BOX_H)
HEADER_MIN_PAD_FRAC   = 0.02      # minimal allowed offset below the top edge (safety margin)
HEADER_LIFT_STEP_FRAC = 0.06      # how much to move UP per iteration if overlap is detected (fraction of HEADER_BOX_H)
HEADER_MAX_ITERS      = 40        # max iterations while trying to resolve overlap

# Row label positioning (near BOTTOM of the row box)

# Axis warp to reduce bunching around OR=1:  x = sign(ln(OR)) * |ln(OR)|^GAMMA
GAMMA            = 0.50           # 0<GAMMA<1 expands center; 0.5 is a gentle, readable default

# Fixed x ticks (subset to range)
TICK_OR_VALUES   = [0.2, 0.5, 0.9, 1.1, 1.5, 2.0]

# Matplotlib defaults
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 12.0,
    "axes.labelsize": 12.0,
    "axes.linewidth": 1.05,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.7,
    "grid.alpha": GRID_ALPHA,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --------------------------- Helpers ---------------------------

def wrap_label(s: str, width=WRAP_WIDTH) -> str:
    s = "" if pd.isna(s) else str(s).replace("_", " ").strip()
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=True)) if s else ""

def format_plain_decimal(x: float, max_decimals=18) -> str:
    """Plain decimal formatter (no scientific/engineering notation)."""
    if not (isinstance(x, (int, float, np.floating)) and np.isfinite(x)):
        return ""
    if x == 0:
        return "0"
    if x >= 1:
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    d = int(math.ceil(-math.log10(x))) + 2
    d = min(max_decimals, max(2, d))
    s = f"{x:.{d}f}".rstrip("0").rstrip(".")
    return s if s else "0"

def palette_cool_no_yellow_or_orange_or_brown(n: int):
    """
    Cool-toned palette excluding yellow/orange/brown.
    (Blues/teals/purples/greens/grays only.)
    """
    base = [
        "#4C78A8", "#72B7B2", "#54A24B", "#7B7C9E",  # blue, teal, green, slate
        "#A0CBE8", "#8C6BB1", "#1B9E77", "#2C7FB8",  # light blue, purple, sea green, blue
        "#6BAED6", "#3182BD", "#08519C", "#8DA0CB",  # blues / periwinkle
        "#66C2A5", "#99C794", "#A5ADD6", "#6A51A3",  # teal/green/purples
        "#9E9AC8", "#7FC97F", "#B2ABD2", "#5DA5DA",  # purples/greens/blues
        "#B39DDB", "#9575CD", "#26A69A", "#666666",  # purples/teal/grays
        "#888888", "#999999", "#444444", "#8E44AD"   # grays/magenta-purple
    ]
    def lighten(hexcolor, amt=0.18):
        r,g,b = mpl.colors.to_rgb(hexcolor)
        r=min(1,r+(1-r)*amt); g=min(1,g+(1-g)*amt); b=min(1,b+(1-b)*amt)
        return mpl.colors.to_hex((r,g,b))
    out = list(base)
    i = 0
    while len(out) < n:
        out.append(lighten(base[i % len(base)], 0.22))
        i += 1
    return out[:n]

def warp_or_to_axis(or_vals):
    """Map OR to warped axis coordinate via ln(OR) -> sign*|ln(OR)|^GAMMA."""
    x = np.asarray(or_vals, dtype=float)
    x = np.where(~np.isfinite(x) | (x <= 0), np.nan, x)
    z = np.log(x)
    s = np.sign(z)
    m = np.power(np.abs(z), GAMMA, where=np.isfinite(z))
    return s * m

def parse_or_ci95(series: pd.Series):
    """
    Parse OR_CI95 strings like '0.893,0.942' into (lo, hi) floats.
    Returns two numpy arrays aligned with series.
    """
    los = np.full(len(series), np.nan, dtype=float)
    his = np.full(len(series), np.nan, dtype=float)
    for i, v in enumerate(series.astype(str).fillna("")):
        parts = [p.strip() for p in v.split(",")]
        if len(parts) == 2:
            try:
                lo = float(parts[0]); hi = float(parts[1])
                if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0:
                    los[i] = lo; his[i] = hi
            except Exception:
                pass
    return los, his

def bboxes_overlap(bb1, bb2):
    """True if two Matplotlib Bbox objects overlap (strict)."""
    return not (bb1.x1 <= bb2.x0 or bb1.x0 >= bb2.x1 or bb1.y1 <= bb2.y0 or bb1.y0 >= bb2.y1)

# --------------------------- Data prep ---------------------------

def load_and_prepare(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: '{path}' not found.")

    df = pd.read_csv(path, sep="\t", dtype=str)

    required = ["Phenotype", "Inversion", "OR", "Q_GLOBAL", "OR_CI95"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: missing required column(s): {', '.join(missing)}")

    # Types
    df["Phenotype"] = df["Phenotype"].fillna("").astype(str)
    df["Inversion"] = df["Inversion"].fillna("").astype(str)
    df["OR"]        = pd.to_numeric(df["OR"], errors="coerce")
    df["Q_GLOBAL"]  = pd.to_numeric(df["Q_GLOBAL"], errors="coerce")

    # Parse CI from OR_CI95
    lo, hi = parse_or_ci95(df["OR_CI95"])
    df["OR_lo"] = lo
    df["OR_hi"] = hi

    # Keep only valid & significant rows
    good = (
        df["Phenotype"].str.strip().ne("") &
        np.isfinite(df["OR"]) & (df["OR"] > 0) &
        np.isfinite(df["Q_GLOBAL"]) &
        np.isfinite(df["OR_lo"]) & (df["OR_lo"] > 0) &
        np.isfinite(df["OR_hi"]) & (df["OR_hi"] > 0)
    )
    df = df[good].copy()
    if df.empty:
        raise SystemExit("No valid rows after cleaning (check OR, Q_GLOBAL, OR_CI95).")

    df = df[df["Q_GLOBAL"] <= 0.05].copy()
    if df.empty:
        raise SystemExit("No FDR-significant hits at q <= 0.05 in Q_GLOBAL.")

    # Within each inversion: order by q ascending, then |lnOR| descending, then Phenotype
    df["lnOR_abs"] = np.abs(np.log(df["OR"]))
    inv_order = pd.unique(df["Inversion"])
    df["Inversion"] = pd.Categorical(df["Inversion"], categories=inv_order, ordered=True)
    df.sort_values(["Inversion", "Q_GLOBAL", "lnOR_abs", "Phenotype"],
                   ascending=[True, True, False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --------------------------- Layout (explicit boxes) ---------------------------

def build_layout(df: pd.DataFrame):
    """
    Build non-overlapping vertical boxes for each inversion section:
      - Header box [head_y0, head_y1)
      - Row boxes for each phenotype (list of dicts with y0, y1, yc)
    Returns:
      sections: list of dicts
      y_max:    max y for axis range
    """
    sections = []
    y_cursor = 0.0  # grows downward (we will invert axis later)

    for inv, sub in df.groupby("Inversion", sort=False):
        sub = sub.copy()

        # Header box (top of section)
        head_y0 = y_cursor
        head_y1 = head_y0 + HEADER_BOX_H

        # Row boxes
        rows = []
        row_y0 = head_y1
        for _, r in sub.iterrows():
            row_y1 = row_y0 + ROW_BOX_H
            rows.append({
                "row_y0": row_y0,
                "row_y1": row_y1,
                "row_yc": 0.5 * (row_y0 + row_y1),
                "Phenotype": r["Phenotype"],
                "Q_GLOBAL": float(r["Q_GLOBAL"]),
                "OR": float(r["OR"]),
                "OR_lo": float(r["OR_lo"]),
                "OR_hi": float(r["OR_hi"]),
                "Inversion": str(r["Inversion"]),
            })
            row_y0 = row_y1

        sec_y0 = head_y0
        sec_y1 = row_y0
        y_cursor = sec_y1 + BLOCK_GAP_H

        sections.append({
            "Inversion": str(inv),
            "head_y0": head_y0,
            "head_y1": head_y1,
            "rows": rows,
            "sec_y0": sec_y0,
            "sec_y1": sec_y1,
        })

    y_max = y_cursor
    return sections, y_max

# --------------------------- Plot ---------------------------

def plot_forest(df: pd.DataFrame, out_pdf=OUT_PDF, out_png=OUT_PNG):
    sections, y_max = build_layout(df)

    # X-limits from CI extremes ONLY, then ±10%
    all_los = [row["OR_lo"] for sec in sections for row in sec["rows"]]
    all_his = [row["OR_hi"] for sec in sections for row in sec["rows"]]
    leftmost  = float(np.min(all_los))
    rightmost = float(np.max(all_his))
    x_or_left  = max(0.0001, leftmost * 0.90)   # 10% further left of min CI
    x_or_right = rightmost * 1.10               # 10% further right of max CI

    # Figure sizing
    n_rows_total = sum(len(sec["rows"]) for sec in sections)
    fig_h = max(8.0, min(28.0, 2.3 + n_rows_total * 0.24 + len(sections) * 0.70))
    fig_w = 17.0

    # Panels
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=LEFT_RIGHT, wspace=0.05)
    axL = fig.add_subplot(gs[0, 0])  # label panel
    axR = fig.add_subplot(gs[0, 1])  # main plot panel

    # Shared y (top -> bottom)
    axL.set_ylim(y_max, 0.0)
    axR.set_ylim(y_max, 0.0)

    # Left panel styling
    axL.set_facecolor("white")
    axL.set_xticks([]); axL.set_yticks([])
    for sp in axL.spines.values():
        sp.set_visible(False)
    yxf = axL.get_yaxis_transform()  # x in axes fraction [0..1], y in data units

    # Right panel styling
    axR.set_yticks([])
    axR.set_xlabel("Odds ratio")

    # Warped x-limits (not forced symmetric)
    x_left_warp  = float(warp_or_to_axis([x_or_left])[0])
    x_right_warp = float(warp_or_to_axis([x_or_right])[0])
    xmin_warp, xmax_warp = (min(x_left_warp, x_right_warp), max(x_left_warp, x_right_warp))
    axR.set_xlim(xmin_warp, xmax_warp)

    # OR=1 line if inside range
    if xmin_warp < 0.0 < xmax_warp:
        axR.axvline(0.0, color="#333333", linestyle="-", linewidth=1.0, alpha=0.9, zorder=1.1)

    # Colors per inversion (no yellow/orange/brown)
    inv_names = [sec["Inversion"] for sec in sections]
    inv_colors = palette_cool_no_yellow_or_orange_or_brown(len(inv_names))
    inv_to_color = {inv: inv_colors[i] for i, inv in enumerate(inv_names)}

    # Alternating background bands per section
    xr0, xr1 = axR.get_xlim()
    for i, sec in enumerate(sections):
        if i % 2 == 0:
            y0, y1 = sec["sec_y0"], sec["sec_y1"]
            axL.add_patch(Rectangle((0, y0), 1, (y1 - y0), transform=yxf,
                                    color="#2f4f4f", alpha=BAND_ALPHA, zorder=0))
            axR.add_patch(Rectangle((xr0, y0), (xr1 - xr0), (y1 - y0),
                                    transform=axR.transData, color="#2f4f4f", alpha=BAND_ALPHA, zorder=0))

    # --- Place texts and graphics; keep references for overlap detection ---
    header_texts = []           # list of dicts: {"text": Text, "sec": sec}
    row_texts_by_section = {}   # sec_id -> [Text, Text, ...] for phenotype + q labels

    # 1) Headers (initial positions near TOP of header box)
    for sec_id, sec in enumerate(sections):
        inv = sec["Inversion"]
        c   = inv_to_color[inv]
        head_y0 = sec["head_y0"]

        # initial y for header label: a bit below the top edge; anchor with va='top'
        y_header = head_y0 + HEADER_BOX_H * HEADER_TOP_PAD_FRAC
        t = axL.text(HEADER_X_SHIFT, y_header, inv, ha="left", va="top",
                     fontsize=13.5, fontweight="semibold", color=c,
                     transform=yxf, zorder=3)
        header_texts.append({"text": t, "sec": sec, "color": c})

        # underline across main plot at the same y
        axR.hlines(y=y_header, xmin=xr0, xmax=xr1, color=c, alpha=HEADER_UL_ALPHA,
                   linewidth=1.0, linestyles="-", zorder=1.0)

    # 2) Rows (labels at BOTTOM of row box; CI & point at CENTER)
    for sec_id, sec in enumerate(sections):
        c = inv_to_color[sec["Inversion"]]
        row_texts_by_section[sec_id] = []

        for row in sec["rows"]:
            y0, y1, yc = row["row_y0"], row["row_y1"], row["row_yc"]

            # Left labels CENTERED vertically within the row box
            y_label = yc  # center of the row box
            
            phen = wrap_label(row["Phenotype"])
            t1 = axL.text(0.02,  y_label, phen, ha="left",  va="center",
                          color="#111111", transform=yxf, zorder=3)
            t2 = axL.text(0.985, y_label, f"q = {format_plain_decimal(row['Q_GLOBAL'])}",
                          ha="right", va="center", color="#444444",
                          transform=yxf, zorder=3)

            row_texts_by_section[sec_id].extend([t1, t2])

            # Right panel: CI line & point at center
            or_lo, or_hi, or_pt = row["OR_lo"], row["OR_hi"], row["OR"]
            x_lo  = float(warp_or_to_axis([or_lo])[0])
            x_hi  = float(warp_or_to_axis([or_hi])[0])
            x_pt  = float(warp_or_to_axis([or_pt])[0])

            axR.hlines(y=yc, xmin=x_lo, xmax=x_hi, color=c, linewidth=2.0, alpha=0.95, zorder=2.2)
            axR.plot([x_lo, x_lo], [yc-0.12, yc+0.12], color=c, linewidth=1.6, alpha=0.95, zorder=2.25)
            axR.plot([x_hi, x_hi], [yc-0.12, yc+0.12], color=c, linewidth=1.6, alpha=0.95, zorder=2.25)
            axR.scatter([x_pt], [yc], s=POINT_SIZE_PT2, facecolor=c, edgecolor="black",
                        linewidth=POINT_EDGE_LW, alpha=0.97, zorder=3.0)

    # 3) Fixed OR ticks within [x_or_left, x_or_right]
    tick_ors = [v for v in TICK_OR_VALUES if (x_or_left <= v <= x_or_right)]
    tick_pos = warp_or_to_axis(tick_ors)
    tick_lbl = [f"{v:g}×" for v in tick_ors]
    axR.set_xticks(tick_pos)
    axR.set_xticklabels(tick_lbl)

    # --- Overlap detection & iterative lift for header labels ---
    # Move each header UP in small steps until it no longer overlaps any row text in its section,
    # or until it reaches the minimal top pad.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for sec_id, hdr in enumerate(header_texts):
        t_header = hdr["text"]
        sec      = hdr["sec"]
        c        = hdr["color"]

        head_y0  = sec["head_y0"]
        y_min    = head_y0 + HEADER_BOX_H * HEADER_MIN_PAD_FRAC  # cannot go above this (i.e., too close to the top edge)
        step     = HEADER_BOX_H * HEADER_LIFT_STEP_FRAC

        # Helper to compute overlap between header text and any row texts in the same section
        def header_overlaps_rows() -> bool:
            bb_h = t_header.get_window_extent(renderer=renderer)
            for txt in row_texts_by_section.get(sec_id, []):
                if not txt.get_visible():
                    continue
                bb_r = txt.get_window_extent(renderer=renderer)
                if bboxes_overlap(bb_h, bb_r):
                    return True
            return False

        # Iteratively nudge header upward if overlapping row text
        moved = False
        iters = 0
        while True:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            if not header_overlaps_rows():
                break
            # current header y (in data units; remember left text uses blended transform)
            # Retrieve current y by storing it ourselves: use t_header.get_position()[1] in data coords
            x_axes_frac, y_data = t_header.get_position()  # (x in axes-fraction, y in data units)
            new_y = max(y_min, y_data - step)  # move up (smaller y) but not above y_min
            if abs(new_y - y_data) < 1e-9:
                # Can't move further; break to avoid infinite loop
                break
            t_header.set_position((x_axes_frac, new_y))
            # Also lift the underline to match the new header y
            # First, remove the old underline by redrawing a fresh one on top (cheap & simple):
            axR.hlines(y=new_y, xmin=xr0, xmax=xr1, color=c, alpha=HEADER_UL_ALPHA,
                       linewidth=1.0, linestyles="-", zorder=1.0)
            moved = True
            iters += 1
            if iters >= HEADER_MAX_ITERS:
                break

    # Final layout & save (no legend)
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.99])
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  - {OUT_PDF}\n  - {OUT_PNG}")

# --------------------------- Entrypoint ---------------------------

def main():
    df = load_and_prepare(INPUT_FILE)
    plot_forest(df, out_pdf=OUT_PDF, out_png=OUT_PNG)

if __name__ == "__main__":
    main()
