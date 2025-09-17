import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerPatch

from scipy.stats import wilcoxon
SCIPY_OK = True

# -----------------------------------------------------------------------------
# Configuration / aesthetics
# -----------------------------------------------------------------------------
OUTPUT_CSV = Path("./output.csv")
INVINFO_TSV = Path("./inv_info.tsv")

rng = np.random.default_rng(2025)  # reproducible jitter

# Core colors
COLOR_DIRECT   = "#1f3b78"   # dark blue
COLOR_INVERTED = "#8c2d7e"   # reddish purple

# Overlay (hatch) colors
OVERLAY_SINGLE = "#d9d9d9"   # very light dots
OVERLAY_RECUR  = "#4a4a4a"   # dark gray diagonals

AX_TEXT = "#333333"          # labels/ticks

# Geometry and styles
VIOLIN_WIDTH  = 0.9
POINT_SIZE    = 28
LINE_WIDTH    = 1.6
ALPHA_VIOLIN  = 0.72
ALPHA_POINTS  = 0.60   # 60% transparent dots (change requested)

# Slim internal box plot width
BOXPLOT_WIDTH = 0.18

# X positions (clear gap between sections)
POS = {
    ("Single-event", "Direct"):   0.00,
    ("Single-event", "Inverted"): 1.00,
    ("Recurrent", "Direct"):      3.00,
    ("Recurrent", "Inverted"):    4.00,
}
SECTION_CENTERS = {"Single-event": 0.50, "Recurrent": 3.50}

# -----------------------------------------------------------------------------
# Legend square handler (kept for compatibility, though we now draw the key manually)
# -----------------------------------------------------------------------------
class SquareHandler(HandlerPatch):
    """Force legend patches to render as squares (not stretched)."""
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        sz = min(width, height) * 0.9
        x = xdescent + (width  - sz) / 2
        y = ydescent + (height - sz) / 2
        p = Rectangle(
            (x, y), sz, sz, transform=trans,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            hatch=orig_handle.get_hatch(),
            linewidth=0.0,
            alpha=orig_handle.get_alpha()
        )
        return [p]

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _standardize_chr(val):
    """Strip leading 'chr' for consistent matching (e.g., 'chr1' -> '1')."""
    s = str(val).strip()
    return s[3:] if s.lower().startswith("chr") else s

def _sci_notation_tex(p):
    """
    Format p in scientific notation with unicode superscripts, e.g., 1.23×10⁻⁶.
    """
    if p is None or not np.isfinite(p):
        return "NA"
    if p == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(p))))
    mant = p / (10 ** exp)
    mant_str = f"{mant:.3g}"
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp_sup = str(exp).translate(superscripts)
    return f"{mant_str}×10{exp_sup}"

def _load_and_match(output_csv: Path, invinfo_tsv: Path) -> pd.DataFrame:
    """Load CSV/TSV files and match regions with ±1 bp tolerance on start/end."""
    if not output_csv.exists():
        raise FileNotFoundError(f"Cannot find {output_csv.resolve()}")
    if not invinfo_tsv.exists():
        raise FileNotFoundError(f"Cannot find {invinfo_tsv.resolve()}")

    df  = pd.read_csv(output_csv)
    inv = pd.read_csv(invinfo_tsv, sep="\t")

    # Standardize chromosomes
    df["chr_std"]  = df["chr"].apply(_standardize_chr)
    inv["chr_std"] = inv["Chromosome"].apply(_standardize_chr)

    df_small = df[["chr_std", "region_start", "region_end", "0_pi_filtered", "1_pi_filtered"]].copy()
    df_small.rename(columns={"0_pi_filtered": "pi_direct", "1_pi_filtered": "pi_inverted"}, inplace=True)

    # Build permissive candidates
    cands = []
    for ds in (-1, 0, 1):
        for de in (-1, 0, 1):
            tmp = df_small.copy()
            tmp["Start"] = tmp["region_start"].astype(int) + ds
            tmp["End"]   = tmp["region_end"].astype(int) + de
            tmp["match_priority"] = abs(ds) + abs(de)  # prefer exact match first
            cands.append(tmp)
    df_cand = pd.concat(cands, ignore_index=True)

    # Recurrence column (consensus preferred, fallback alternate)
    recur_col = None
    for c in ["0_single_1_recur_consensus", "0_single_1_recur"]:
        if c in inv.columns:
            recur_col = c
            break
    if recur_col is None:
        raise KeyError("inv_info.tsv lacks '0_single_1_recur_consensus' and '0_single_1_recur'.")

    inv_small = inv[["chr_std", "Start", "End", recur_col]].copy()
    merged = df_cand.merge(inv_small, on=["chr_std", "Start", "End"], how="left")

    # Keep best match per original row
    merged.sort_values(["chr_std", "region_start", "region_end", "match_priority"], inplace=True)
    merged = merged.drop_duplicates(subset=["chr_std", "region_start", "region_end"], keep="first")

    # Map recurrence to labels
    merged["Recurrence"] = pd.to_numeric(merged[recur_col], errors="coerce").map({0: "Single-event", 1: "Recurrent"})
    merged = merged[~merged["Recurrence"].isna()].copy()

    # π cleanup
    merged["pi_direct"]   = pd.to_numeric(merged["pi_direct"], errors="coerce")
    merged["pi_inverted"] = pd.to_numeric(merged["pi_inverted"], errors="coerce")

    # require at least one π present
    merged = merged.dropna(subset=["pi_direct", "pi_inverted"], how="all").copy()
    return merged

def _prepare_for_violins(matched: pd.DataFrame):
    """Prepare data lists (in order) and positions/keys for violin/box plotting."""
    se = matched.loc[matched["Recurrence"] == "Single-event"]
    re = matched.loc[matched["Recurrence"] == "Recurrent"]

    vals = [
        se["pi_direct"].dropna().values,     # x=0
        se["pi_inverted"].dropna().values,   # x=1
        re["pi_direct"].dropna().values,     # x=3
        re["pi_inverted"].dropna().values,   # x=4
    ]
    pos = [
        POS[("Single-event", "Direct")],
        POS[("Single-event", "Inverted")],
        POS[("Recurrent", "Direct")],
        POS[("Recurrent", "Inverted")],
    ]
    keys = [
        ("Single-event", "Direct"),
        ("Single-event", "Inverted"),
        ("Recurrent", "Direct"),
        ("Recurrent", "Inverted"),
    ]
    return vals, pos, keys

def _draw_half_violins(ax, vals, pos, keys):
    """
    Draw full violins and clip to outside halves.
    Overlay style per group (dots for Single-event, diagonals for Recurrent) on top of base colors.
    """
    v = ax.violinplot(
        dataset=vals,
        positions=pos,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    bodies = v["bodies"]

    # Use current y-limits (already pre-set) for robust clipping
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    xmin -= 1.0
    xmax += 1.0

    for body, (grp, ori), x in zip(bodies, keys, pos):
        base_color = COLOR_DIRECT if ori == "Direct" else COLOR_INVERTED
        body.set_facecolor(base_color)
        body.set_edgecolor("none")
        body.set_alpha(ALPHA_VIOLIN)

        # Clip to half (outside half only)
        if ori == "Direct":
            clip_rect = Rectangle((xmin, ymin), x - 1e-6 - xmin, ymax - ymin, transform=ax.transData)
        else:
            clip_rect = Rectangle((x + 1e-6, ymin), xmax - (x + 1e-6), ymax - ymin, transform=ax.transData)
        body.set_clip_path(clip_rect)

        # Overlay style per group
        if grp == "Single-event":
            body.set_hatch(".")
            body.set_edgecolor(OVERLAY_SINGLE)
            body.set_linewidth(0.0)
        else:
            body.set_hatch("//")
            body.set_edgecolor(OVERLAY_RECUR)
            body.set_linewidth(0.0)

    return bodies

def _overlay_boxplots(ax, vals, pos):
    """Draw slim boxplots centered at each position (inside the pair)."""
    for data, x in zip(vals, pos):
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        if data.size == 0:
            continue
        bp = ax.boxplot(
            [data],
            positions=[x],
            widths=BOXPLOT_WIDTH,
            vert=True,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            boxprops=dict(facecolor="white", edgecolor="#111111", linewidth=1.1),
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color="#111111", linewidth=1.0),
            capprops=dict(color="#111111", linewidth=1.0),
        )
        # Ensure boxplot is visible atop violin
        for part in ["boxes", "medians", "whiskers", "caps"]:
            for artist in bp[part]:
                artist.set_zorder(5)

def _paired_lines_and_points(ax, matched: pd.DataFrame, cmap, norm):
    """For each region, draw jittered points and a connecting line (colored by log2FC)."""
    def jitter():
        # inward jitter to keep connections tidy
        return float(rng.uniform(0.06, 0.20))

    EPS = 1e-12

    for grp in ["Single-event", "Recurrent"]:
        sub = matched.loc[matched["Recurrence"] == grp].copy()
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            d = row["pi_direct"]
            i = row["pi_inverted"]
            if not (np.isfinite(d) and np.isfinite(i)):
                continue
            log2fc = math.log2((d + EPS) / (i + EPS))
            color = cmap(norm(log2fc))

            j = jitter()
            x_d = POS[(grp, "Direct")] + j    # Direct jitter to the right
            x_i = POS[(grp, "Inverted")] - j  # Inverted jitter to the left

            ax.plot([x_d, x_i], [d, i], color=color, linewidth=LINE_WIDTH, alpha=0.98, zorder=3, solid_capstyle="round")
            ax.scatter([x_d], [d], s=POINT_SIZE, c=COLOR_DIRECT,   edgecolors="black", linewidths=0.5, alpha=ALPHA_POINTS, zorder=6)
            ax.scatter([x_i], [i], s=POINT_SIZE, c=COLOR_INVERTED, edgecolors="black", linewidths=0.5, alpha=ALPHA_POINTS, zorder=6)

def _wilcoxon_and_annotate(ax, matched: pd.DataFrame, y_for_brackets: float):
    """Compute paired Wilcoxon tests and place compact p-value annotations above each section."""
    def annotate(center_x, label, pval):
        y = y_for_brackets
        # bracket
        ax.plot([center_x - 0.55, center_x + 0.55], [y, y], color="#444444", linewidth=1.3)
        ax.plot([center_x - 0.55, center_x - 0.55], [y, y * 0.998], color="#444444", linewidth=1.3)
        ax.plot([center_x + 0.55, center_x + 0.55], [y, y * 0.998], color="#444444", linewidth=1.3)
        # label (omit “signed-rank”)
        ptxt = _sci_notation_tex(pval) if np.isfinite(pval) else "NA"
        ax.text(center_x, y * 1.008, f"{label} Wilcoxon p = {ptxt}",
                ha="center", va="bottom", fontsize=11, color=AX_TEXT)

    for grp in ["Single-event", "Recurrent"]:
        sub = matched.loc[matched["Recurrence"] == grp].dropna(subset=["pi_direct", "pi_inverted"]).copy()
        pval = float("nan")
        if SCIPY_OK and len(sub) >= 2:
            try:
                _, pval = wilcoxon(sub["pi_direct"].values, sub["pi_inverted"].values,
                                   alternative="two-sided", zero_method="wilcox")
            except Exception:
                try:
                    _, pval = wilcoxon(sub["pi_direct"].values, sub["pi_inverted"].values,
                                       alternative="two-sided", zero_method="zsplit")
                except Exception:
                    pval = float("nan")
        annotate(SECTION_CENTERS[grp], grp, pval)

def _draw_right_key(rax):
    """
    Draw a manual, large-square group key on the right panel (top area).
    Squares are ~4× typical legend patch area. We use axis-fraction coordinates.
    """
    rax.set_xlim(0, 1)
    rax.set_ylim(0, 1)
    rax.axis("off")

    # Square size in axis-fraction units — bigger (~4× typical legend size).
    S = 0.16  # side length of the square (adjusted for visibility)
    X0 = 0.08
    YS = [0.82, 0.58, 0.34, 0.10]  # vertical placements (top to bottom)

    # Build the four subgroup squares
    entries = [
        ("Single-event — Direct",   COLOR_DIRECT,   ".",  OVERLAY_SINGLE),
        ("Single-event — Inverted", COLOR_INVERTED, ".",  OVERLAY_SINGLE),
        ("Recurrent — Direct",      COLOR_DIRECT,   "//", OVERLAY_RECUR),
        ("Recurrent — Inverted",    COLOR_INVERTED, "//", OVERLAY_RECUR),
    ]

    for (label, face, hatch, edge), y in zip(entries, YS):
        sq = Rectangle((X0, y), S, S, transform=rax.transAxes,
                       facecolor=face, edgecolor=edge, hatch=hatch,
                       linewidth=0.0, alpha=ALPHA_VIOLIN, clip_on=False)
        rax.add_patch(sq)
        rax.text(X0 + S + 0.06, y + S/2, label, transform=rax.transAxes,
                 ha="left", va="center", fontsize=12, color=AX_TEXT)

def main():
    # Fonts: professional, Linux-friendly
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Liberation Sans", "Ubuntu", "Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": AX_TEXT,
        "xtick.color": AX_TEXT,
        "ytick.color": AX_TEXT,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Load & match
    matched = _load_and_match(OUTPUT_CSV, INVINFO_TSV)
    if matched.empty:
        raise SystemExit("No matched regions found with ±1 bp tolerance.")

    vals, pos, keys = _prepare_for_violins(matched)

    # Y limits with headroom for brackets — set BEFORE violins so clipping works
    all_points = np.concatenate([a for a in vals if isinstance(a, (list, np.ndarray)) and len(a) > 0]) \
                 if any(len(a) for a in vals) else np.array([0.0, 1.0])
    ymin = max(0.0, float(np.nanmin(all_points)) * 0.98)
    ymax = float(np.nanmax(all_points)) if np.isfinite(np.nanmax(all_points)) else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    # Figure with a right-side column for key (top) and color bar (under it)
    fig = plt.figure(figsize=(14.6, 6.6))
    gs  = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.42], wspace=0.04)
    ax  = fig.add_subplot(gs[0, 0])

    # Right column split vertically (legend/key above, color bar below)
    # Make bar thin by keeping its panel short; move it UP a bit with a manual shift later.
    side = gs[0, 1].subgridspec(nrows=2, ncols=1, height_ratios=[0.72, 0.28], hspace=0.08)
    rax = fig.add_subplot(side[0, 0])  # key container (top-right)
    cax = fig.add_subplot(side[1, 0])  # colorbar container (under key)
    rax.axis("off")

    # Main axis setup
    ax.set_ylim(ymin, ymax * 1.17)
    ax.set_axisbelow(True)

    # Half-violins (outside halves)
    _ = _draw_half_violins(ax, vals, pos, keys)

    # Overlay narrow proper box plots
    _overlay_boxplots(ax, vals, pos)

    # Color normalization for log2 fold-change lines
    EPS = 1e-12
    log2fcs = np.log2((matched["pi_direct"].to_numpy(dtype=float) + EPS) /
                      (matched["pi_inverted"].to_numpy(dtype=float) + EPS))
    finite_vals = log2fcs[np.isfinite(log2fcs)]
    if finite_vals.size == 0:
        v = 1.0
    else:
        v = float(np.nanpercentile(np.abs(finite_vals), 98))
        if not np.isfinite(v) or v <= 0:
            v = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 1.0
            if not np.isfinite(v) or v <= 0:
                v = 1.0
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    cmap = plt.get_cmap("coolwarm")

    # Paired lines & points
    _paired_lines_and_points(ax, matched, cmap=cmap, norm=norm)

    # Axes labels and ticks
    ax.set_ylabel("Nucleotide diversity (π)", color=AX_TEXT, fontsize=16)
    ax.set_xticks([
        POS[("Single-event", "Direct")],
        POS[("Single-event", "Inverted")],
        POS[("Recurrent", "Direct")],
        POS[("Recurrent", "Inverted")],
    ])
    ax.set_xticklabels(["Direct", "Inverted", "Direct", "Inverted"], fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)

    # Section labels under each pair (move up, closer to x-axis)
    ax.text(SECTION_CENTERS["Single-event"], -0.06, "Single-event", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=14, fontweight="bold", color=AX_TEXT)
    ax.text(SECTION_CENTERS["Recurrent"], -0.06, "Recurrent", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=14, fontweight="bold", color=AX_TEXT)

    # Subtle separator between sections
    ax.axvline(x=(POS[("Single-event", "Inverted")] + POS[("Recurrent", "Direct")]) / 2,
               color="#dddddd", linewidth=1.0, zorder=1)

    # P-values (brackets & text; omit “signed-rank” wording)
    _wilcoxon_and_annotate(ax, matched, y_for_brackets=ax.get_ylim()[1] / 1.15)

    # ---------- RIGHT COLUMN ----------
    # Large-square group key (manual) at top-right
    _draw_right_key(rax)

    # Thin horizontal color bar for log2 fold-change — under the key, moved UP slightly
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(
        r"$\log_{2}\!\left(\pi_{\mathrm{Direct}} \,/\, \pi_{\mathrm{Inverted}}\right)$",
        color=AX_TEXT,
        fontsize=13,
    )
    cbar.ax.tick_params(color=AX_TEXT, labelcolor=AX_TEXT, labelsize=12)
    for spine in cax.spines.values():
        spine.set_visible(False)

    # Make the bar a thinner rectangle and shift it up a bit
    pos = cax.get_position(fig)
    new_height = pos.height * 0.50      # thin horizontal bar height for readability
    new_y      = pos.y0 + pos.height * 0.22  # lift the bar to sit closer under the key
    new_x      = pos.x0 + pos.width * 0.10   # shift the bar further to the right within the right column
    new_w      = pos.width * 0.86            # preserve right margin and keep the bar compact
    cax.set_position([new_x, new_y, new_w, new_height])

    # Save
    fig.savefig("pi_comparison_violins.pdf", bbox_inches="tight")

    # (Optional) show if interactive backend is available
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
