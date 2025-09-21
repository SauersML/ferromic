import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib import colors as mcolors

from _inv_common import map_inversion_series

INPUT_FILE = "phewas_results.tsv"
OUTPUT_PDF = "phewas_volcano.pdf"

# Ensure PDF/SVG outputs remain vectorized with editable text
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# --------------------------- Color / markers ---------------------------

def non_orange_colors(n, seed=21):
    """Generate n distinct colors excluding orange hues."""
    if n <= 0:
        return []
    gaps = [(0.0, 0.055), (0.125, 1.0)]  # skip ~20°–45° (orange) in HSV
    total = sum(b - a for a, b in gaps)
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
        cols.append(mcolors.hsv_to_rgb((h, s, v)))
    return [tuple(c) for c in cols]

def assign_colors_and_markers(levels):
    n = len(levels)
    colors = non_orange_colors(n)
    if n > 1:
        colors = colors[1:] + colors[:1]
    marker_cycle = ['o', 's', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', 'H', 'd']
    marker_map = {lvl: marker_cycle[i % len(marker_cycle)] for i, lvl in enumerate(levels)}
    color_map  = {lvl: colors[i] for i, lvl in enumerate(levels)}
    return color_map, marker_map

# --------------------------- Stats ---------------------------

def bh_fdr_cutoff(pvals, alpha=0.05):
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

# --------------------------- Data ---------------------------

def load_and_prepare(path):
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: '{path}' not found in current directory.")
    df = pd.read_csv(path, sep="\t", dtype=str)

    need = ["OR", "P_LRT_Overall"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"ERROR: missing required column '{c}' in {path}")

    df["Inversion"] = df.get("Inversion", "Unknown").fillna("Unknown").astype(str)
    df["Inversion"] = map_inversion_series(df["Inversion"])
    df["Phenotype"] = df.get("Phenotype", "").fillna("").astype(str)

    df["OR"] = pd.to_numeric(df["OR"], errors="coerce")
    df["P_LRT_Overall"] = pd.to_numeric(df["P_LRT_Overall"], errors="coerce")

    # Keep only finite, positive p
    df = df[np.isfinite(df["P_LRT_Overall"].to_numpy()) & (df["P_LRT_Overall"] > 0)].copy()

    df["lnOR"] = np.log(df["OR"]) / np.log(2.2)   # log_base3(OR)

    df["neglog10p"] = -np.log10(df["P_LRT_Overall"])

    df = df[np.isfinite(df["lnOR"]) & np.isfinite(df["neglog10p"])].copy()

    # Drop empty labels
    df = df[df["Phenotype"].str.strip() != ""].copy()

    # Stabilize indices (used later for label bookkeeping)
    df.reset_index(drop=True, inplace=True)
    return df

# --------------------------- Axis ticks ---------------------------

def make_or_ticks_sparse(xlim_ln):
    candidates = np.array([0.1, 0.2, 0.33, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    ln_pos = np.log(candidates) / np.log(2.2)

    # Within current xlim
    in_range = (ln_pos >= xlim_ln[0]) & (ln_pos <= xlim_ln[1])

    # Thin the middle but keep 1×
    keep = in_range & ~(((candidates > 0.8) & (candidates < 1.25)) & (np.abs(candidates - 1.0) > 1e-12))

    pos = ln_pos[keep]
    vals = candidates[keep]

    # Labels
    labels = ["1×" if np.isclose(v, 1.0) else f"{v:.2g}×" for v in vals]
    return pos.tolist(), labels

# --------------------------- Label helpers ---------------------------

def _px_to_data(ax, dx_px, dy_px):
    inv = ax.transData.inverted()
    x0, y0 = ax.transData.transform((0, 0))
    x1, y1 = x0 + dx_px, y0 + dy_px
    (xd, yd) = inv.transform((x1, y1)) - inv.transform((x0, y0))
    return float(xd), float(yd)

def _bbox_dict(ax, texts, expand=(1.0, 1.0)):
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = {}
    for key, t in texts.items():
        if (t is not None) and t.get_visible():
            bb = t.get_window_extent(renderer=renderer)
            if expand != (1.0, 1.0):
                bb = bb.expanded(expand[0], expand[1])
            out[key] = bb
    return out

def _overlap(bb1, bb2):
    return not (bb1.x1 <= bb2.x0 or bb1.x0 >= bb2.x1 or bb1.y1 <= bb2.y0 or bb1.y0 >= bb2.y1)

def _find_overlapping_pairs(bboxes_dict):
    items = list(bboxes_dict.items())
    pairs = set()
    n = len(items)
    for a in range(n):
        ia, bba = items[a]
        for b in range(a + 1, n):
            ib, bbb = items[b]
            if _overlap(bba, bbb):
                pairs.add((ia, ib))
    return pairs

def _thin_by_significance(df, texts, keys_subset=None, expand=(1.02, 1.08)):
    """
    Delete labels until NO overlaps remain.
    Keep the more significant (greater neglog10p). Tie-break: larger |lnOR|, then smaller index.
    """
    if not texts:
        return False

    vis_keys = []
    for k, t in texts.items():
        if t is None or (not t.get_visible()):
            continue
        if (keys_subset is None) or (k in keys_subset):
            vis_keys.append(k)
    if len(vis_keys) <= 1:
        return False

    any_text = next(iter(texts.values()))
    ax = any_text.axes

    bboxes = _bbox_dict(ax, {k: texts[k] for k in vis_keys}, expand=expand)
    if len(bboxes) <= 1:
        return False

    pairs = _find_overlapping_pairs(bboxes)
    if not pairs:
        return False

    losers = set()
    for i, j in pairs:
        yi = float(df.loc[i, "neglog10p"])
        yj = float(df.loc[j, "neglog10p"])
        if yi == yj:
            xi = abs(float(df.loc[i, "lnOR"]))
            xj = abs(float(df.loc[j, "lnOR"]))
            if xi == xj:
                drop = max(i, j)  # deterministic
            else:
                drop = i if xi < xj else j  # drop closer to center
        else:
            drop = i if yi < yj else j  # drop less significant
        losers.add(drop)

    changed = False
    for k in losers:
        t = texts.get(k, None)
        if t is not None and t.get_visible():
            t.set_visible(False)
            changed = True
    return changed

def _prune_out_of_bounds(ax, texts, df, eps_px=1.0):
    """
    Remove labels that:
      - exit the axes area;
      - are on the wrong side of the y-axis (x=0) relative to their point;
      - extend below the x-axis (y=0).
    """
    if not texts:
        return False

    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axbb = ax.get_window_extent(renderer=renderer)
    yaxis_x_px = ax.transData.transform((0, 0))[0]
    xaxis_y_px = ax.transData.transform((0, 0))[1]

    changed = False
    for idx, t in list(texts.items()):
        if t is None or (not t.get_visible()):
            continue
        bb = t.get_window_extent(renderer=renderer)
        # Outside axes?
        if (bb.x0 < axbb.x0 - eps_px or bb.x1 > axbb.x1 + eps_px or
            bb.y0 < axbb.y0 - eps_px or bb.y1 > axbb.y1 + eps_px):
            t.set_visible(False); changed = True; continue
        # Wrong side of y-axis?
        x = float(df.loc[idx, "lnOR"])
        if x >= 0:
            if bb.x0 < yaxis_x_px - eps_px:
                t.set_visible(False); changed = True; continue
        else:
            if bb.x1 > yaxis_x_px + eps_px:
                t.set_visible(False); changed = True; continue
        # Below x-axis?
        if bb.y0 < xaxis_y_px - eps_px:
            t.set_visible(False); changed = True; continue
    return changed

def _add_connector(ax, text_artist, px_point, color, linewidth=0.9, alpha=0.9):
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = text_artist.get_window_extent(renderer=renderer)
    cx = min(max(px_point[0], bb.x0), bb.x1)
    cy = min(max(px_point[1], bb.y0), bb.y1)
    inv = ax.transData.inverted()
    xA, yA = inv.transform((cx, cy))
    xB, yB = inv.transform(tuple(px_point))
    ax.add_patch(FancyArrowPatch(
        (xA, yA), (xB, yB),
        arrowstyle="-", mutation_scale=1,
        linewidth=linewidth, color=color, alpha=alpha,
        shrinkA=0.0, shrinkB=0.0, zorder=3.4
    ))

# --------------------------- Plot ---------------------------

def plot_volcano(df, out_pdf):
    if df.empty:
        raise SystemExit("ERROR: No valid rows after cleaning; nothing to plot.")

    # Up-arrow handling for ultra-small p-values
    EXTREME_Y = 300.0
    df["is_extreme"] = df["neglog10p"] > EXTREME_Y
    if (~df["is_extreme"]).any():
        ymax_nonextreme = df.loc[~df["is_extreme"], "neglog10p"].max()
        arrow_y = ymax_nonextreme * 1.10 if (np.isfinite(ymax_nonextreme) and ymax_nonextreme > 0) else EXTREME_Y * 1.10
    else:
        arrow_y = EXTREME_Y * 1.10
    df["y_plot"] = np.where(df["is_extreme"], arrow_y, df["neglog10p"])

    # Colors/markers
    inv_levels = sorted(df["Inversion"].unique())
    color_map, marker_map = assign_colors_and_markers(inv_levels)

    # FDR threshold (BH 0.05)
    p_cut = bh_fdr_cutoff(df["P_LRT_Overall"].to_numpy(), alpha=0.05)
    y_fdr = -np.log10(p_cut) if (isinstance(p_cut, (int, float)) and np.isfinite(p_cut) and p_cut > 0) else np.nan
    fdr_label = f"BH FDR 0.05 (p ≤ {p_cut:.2e})" if np.isfinite(y_fdr) else "BH FDR 0.05"

    # Dynamically set the labeling threshold to a q-value alpha
    p_cut_labeling = bh_fdr_cutoff(df["P_LRT_Overall"].to_numpy(), alpha=0.06)
    if p_cut_labeling is not None and np.isfinite(p_cut_labeling) and p_cut_labeling > 0:
        LABEL_MIN_Y = -np.log10(p_cut_labeling)
    else:
        # If no points meet the threshold, set an impossibly high value to label nothing.
        LABEL_MIN_Y = np.inf

    # X-limits: show ALL data (entirely in log space via lnOR; symmetric around 0 == OR 1)
    xabs = np.abs(df["lnOR"].to_numpy())
    xmax = np.nanmax(xabs) if xabs.size else 1.0
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    xpad = xmax * 0.06
    xlim = (-xmax - xpad, xmax + xpad)

    fig, ax = plt.subplots(figsize=(13, 8.5))

    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=15, width=1.2)

    # FULL LOG REGION: we already plot ln(OR) on a linear axis, so the entire axis is logarithmic in OR with no linear band.
    ax.set_xlim(xlim)

    ymax = df["y_plot"].max()
    ax.set_ylim(0, (ymax * 1.06) if (np.isfinite(ymax) and ymax > 0) else 10)

    # Baseline + FDR line
    ax.axvline(0.0, color='k', linewidth=1.0)
    if np.isfinite(y_fdr):
        ax.axhline(y_fdr, linestyle=":", color="black", linewidth=1.2)

    # Draw points
    N = df.shape[0]
    rasterize = N > 60000
    for inv in inv_levels:
        sub = df[df["Inversion"] == inv]
        norm = sub[~sub["is_extreme"]]
        if not norm.empty:
            ax.scatter(
                norm["lnOR"].to_numpy(), norm["y_plot"].to_numpy(),
                s=22, alpha=0.75, marker=marker_map[inv],
                facecolor=color_map[inv], edgecolor="black", linewidth=0.3,
                rasterized=rasterize
            )
        ext = sub[sub["is_extreme"]]
        if not ext.empty:
            ax.scatter(
                ext["lnOR"].to_numpy(), ext["y_plot"].to_numpy(),
                s=90, alpha=0.95, marker=r'$\uparrow$',
                facecolor=color_map[inv], edgecolor="black", linewidth=0.4,
                rasterized=rasterize
            )

    # Axis labels & ticks
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=18)  # italic p
    ax.set_xlabel("", fontsize=18)                  # remove x-axis title
    xticks, xlabels = make_or_ticks_sparse(ax.get_xlim())
    if len(xticks) >= 3:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=15)

    # Legend inside top-right; include dotted FDR sample
    inv_handles = [
        Line2D([], [], linestyle='None', marker=marker_map[inv], markersize=9,
               markerfacecolor=color_map[inv], markeredgecolor="black", markeredgewidth=0.6,
               label=str(inv))
        for inv in inv_levels
    ]
    fdr_handle = Line2D([], [], linestyle=':', color='black', linewidth=1.2, label=fdr_label)
    handles = inv_handles + ([fdr_handle] if np.isfinite(y_fdr) else [])
    n_inv = len(inv_levels)
    ncol = 1 if n_inv <= 12 else (2 if n_inv <= 30 else 3)
    ax.legend(
        handles=handles, title="Key",
        loc="upper right", frameon=False, ncol=ncol,
        borderaxespad=0.8, handlelength=1.6, columnspacing=1.0, labelspacing=0.6,
        fontsize=13, title_fontsize=15
    )

    # -------------------- BINNED LABELING --------------------
    # 10 bins by |lnOR|; 1 = most extreme
    abs_ln = np.abs(df["lnOR"].to_numpy())
    try:
        df["__bin_tmp"] = pd.qcut(abs_ln, q=10, labels=False, duplicates="drop")
        max_lbl = int(df["__bin_tmp"].max())
        df["bin10"] = (max_lbl - df["__bin_tmp"]).astype(int) + 1  # 1..10 (1 = most extreme)
    except Exception:
        rk = pd.Series(abs_ln).rank(method="average", pct=True).to_numpy()
        df["bin10"] = (10 - np.ceil(rk * 10).astype(int)) + 1

    DX_LABEL_PX = 8.0
    DY_LABEL_PX = 2.0
    dx_data, dy_data = _px_to_data(ax, DX_LABEL_PX, DY_LABEL_PX)

    texts = {}  # idx -> Text

    # Place labels per bin (extreme->inward), prune until stable
    for b in sorted(df["bin10"].unique()):
        bin_rows = df[(df["bin10"] == b) & (df["neglog10p"] >= LABEL_MIN_Y)]
        if bin_rows.empty:
            continue

        for idx, r in bin_rows.iterrows():
            if idx in texts:
                continue
            x, y = float(r["lnOR"]), float(r["y_plot"])
            label_text = str(r["Phenotype"]).replace("_", " ")  # underscores → spaces
            if x >= 0:
                tx, ha = x + dx_data, "left"
            else:
                tx, ha = x - dx_data, "right"
            t = ax.text(tx, y, label_text, fontsize=13.0, ha=ha, va="bottom",
                        color="black", zorder=3.5)
            texts[idx] = t

        # Strict per-bin thinning + out-of-bounds pruning
        while True:
            removed1 = _thin_by_significance(df, texts, keys_subset=set(bin_rows.index), expand=(1.02, 1.08))
            removed2 = _prune_out_of_bounds(ax, texts, df, eps_px=1.0)
            if not (removed1 or removed2):
                break

    # Final global cleanup
    while True:
        removed1 = _thin_by_significance(df, texts, keys_subset=None, expand=(1.02, 1.08))
        removed2 = _prune_out_of_bounds(ax, texts, df, eps_px=1.0)
        if not (removed1 or removed2):
            break

    # Connectors
    fig.canvas.draw()
    point_px = {}
    for idx, r in df.iterrows():
        px = ax.transData.transform((float(r["lnOR"]), float(r["y_plot"])))
        point_px[idx] = (float(px[0]), float(px[1]))
    for idx, t in texts.items():
        if t is None or (not t.get_visible()):
            continue
        inv = str(df.loc[idx, "Inversion"])
        col = color_map.get(inv, (0, 0, 0))
        _add_connector(ax, t, point_px[idx], color=col, linewidth=0.9, alpha=0.9)

    # Save
    with PdfPages(OUTPUT_PDF) as pdf:
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")

# --------------------------- Entrypoint ---------------------------

def main():
    df = load_and_prepare(INPUT_FILE)
    plot_volcano(df, OUTPUT_PDF)

if __name__ == "__main__":
    main()
