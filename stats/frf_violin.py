
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# ---------- Configuration ----------
# Colors for regions and overlay hatches for groups
EDGE_COLOR = "#15616d"       # teal for Edge region
MIDDLE_COLOR = "#d1495b"     # rose for Middle region
OVERLAY_SINGLE = "#d9d9d9"   # dots hatch color for Single-event
OVERLAY_RECUR  = "#4a4a4a"   # diagonal hatch color for Recurrent

VIOLIN_WIDTH = 0.9
ALPHA_VIOLIN = 0.72
POINT_SIZE   = 28
LINE_WIDTH   = 1.4
ALPHA_POINTS = 0.65

# Fixed x-positions for a clear gap between groups
POS = {
    ("Single-event", "Edge region"):   0.00,
    ("Single-event", "Middle region"): 1.00,
    ("Recurrent",    "Edge region"):   3.00,
    ("Recurrent",    "Middle region"): 4.00,
}
SECTION_CENTERS = {"Single-event": 0.50, "Recurrent": 3.50}

# Reproducible jitter so the connecting line uses the same offset for the pair
RNG = np.random.default_rng(2025)


def _find_column(df: pd.DataFrame, options: list[str]) -> str:
    for col in options:
        if col in df.columns:
            return col
    raise KeyError(f"Missing required column; looked for one of: {options!r}")


def load_data() -> pd.DataFrame:
    # Attempt to locate the input file locally
    filename = "per_inversion_frf_effects.tsv"
    candidates = [
        filename,
        os.path.join("data", filename),
        os.path.join(os.path.dirname(__file__), "..", "data", filename),
    ]

    local_path = None
    for cand in candidates:
        if os.path.exists(cand):
            local_path = cand
            break

    if not local_path:
        raise FileNotFoundError(f"Could not find {filename}. Ensure it is present in CWD or data/.")

    df = pd.read_csv(local_path, sep="\t")
    group_col = _find_column(df, ["STATUS", "0_single_1_recur_consensus"])
    edge_col  = _find_column(df, ["frf_mu_edge"])
    mid_col   = _find_column(df, ["frf_mu_mid"])

    df[group_col] = pd.to_numeric(df[group_col], errors="coerce")
    df[edge_col]  = pd.to_numeric(df[edge_col], errors="coerce")
    df[mid_col]   = pd.to_numeric(df[mid_col], errors="coerce")

    keep = df[group_col].isin([0, 1]) & df[edge_col].notna() & df[mid_col].notna()
    sub = df.loc[keep, [group_col, edge_col, mid_col]].copy()

    sub.rename(columns={edge_col: "Edge region", mid_col: "Middle region"}, inplace=True)
    sub["Group"] = sub[group_col].map({0: "Single-event", 1: "Recurrent"})
    return sub[["Group", "Edge region", "Middle region"]]


def prepare_violin_arrays(sub: pd.DataFrame):
    order = [("Single-event", "Edge region"),
             ("Single-event", "Middle region"),
             ("Recurrent",    "Edge region"),
             ("Recurrent",    "Middle region")]
    values = []
    positions = []
    keys = []
    for grp, region in order:
        arr = sub.loc[sub["Group"] == grp, region].to_numpy(dtype=float)
        values.append(arr[~np.isnan(arr)])
        positions.append(POS[(grp, region)])
        keys.append((grp, region))
    return values, positions, keys


def draw_half_violins(ax, values, positions, keys):
    v = ax.violinplot(
        dataset=values,
        positions=positions,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Use current axis limits to build clipping rectangles
    xmin, xmax = ax.get_xlim()
    ymin_ax, ymax_ax = ax.get_ylim()
    xmin -= 1.0
    xmax += 1.0

    for body, (grp, region), x in zip(v["bodies"], keys, positions):
        base_color = EDGE_COLOR if region == "Edge region" else MIDDLE_COLOR
        body.set_facecolor(base_color)
        body.set_edgecolor("none")
        body.set_alpha(ALPHA_VIOLIN)

        # Clip to outside half so the two regions are visually separated
        if region == "Edge region":
            clip_rect = Rectangle((xmin, ymin_ax), x - 1e-6 - xmin, ymax_ax - ymin_ax, transform=ax.transData)
        else:
            clip_rect = Rectangle((x + 1e-6, ymin_ax), xmax - (x + 1e-6), ymax_ax - ymin_ax, transform=ax.transData)
        body.set_clip_path(clip_rect)

        # Hatch overlay by group
        if grp == "Single-event":
            body.set_hatch(".")
            body.set_edgecolor(OVERLAY_SINGLE)
        else:
            body.set_hatch("//")
            body.set_edgecolor(OVERLAY_RECUR)
        body.set_linewidth(0.0)

    return v


def overlay_boxplots(ax, values, positions):
    for data, x in zip(values, positions):
        d = np.asarray(data, dtype=float)
        d = d[~np.isnan(d)]
        if d.size == 0:
            continue
        bp = ax.boxplot(
            [d],
            positions=[x],
            widths=0.18,
            vert=True,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            boxprops=dict(facecolor="white", edgecolor="#111111", linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.6),
            whiskerprops=dict(color="#111111", linewidth=1.0),
            capprops=dict(color="#111111", linewidth=1.0),
        )
        for part in ["boxes", "medians", "whiskers", "caps"]:
            for artist in bp[part]:
                artist.set_zorder(5)


def paired_lines_and_points(ax, sub: pd.DataFrame):
    for grp in ["Single-event", "Recurrent"]:
        g = sub.loc[sub["Group"] == grp]
        if g.empty:
            continue
        offset = RNG.uniform(0.06, 0.20, size=len(g))  # inward jitter
        x_edge = POS[(grp, "Edge region")] + offset
        x_mid  = POS[(grp, "Middle region")] - offset
        y_edge = g["Edge region"].to_numpy()
        y_mid  = g["Middle region"].to_numpy()

        for xe, xm, ye, ym in zip(x_edge, x_mid, y_edge, y_mid):
            ax.plot([xe, xm], [ye, ym], color="#555555", linewidth=LINE_WIDTH, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.scatter(x_edge, y_edge, s=POINT_SIZE, c=EDGE_COLOR,   edgecolors="black", linewidths=0.5, alpha=ALPHA_POINTS, zorder=6)
        ax.scatter(x_mid,  y_mid,  s=POINT_SIZE, c=MIDDLE_COLOR, edgecolors="black", linewidths=0.5, alpha=ALPHA_POINTS, zorder=6)


def main():
    sub = load_data()
    values, positions, keys = prepare_violin_arrays(sub)

    all_vals = np.concatenate([a for a in values if a.size > 0]) if any(a.size for a in values) else np.array([0.0, 1.0])
    ymin = float(np.nanmin(all_vals)) if np.isfinite(np.nanmin(all_vals)) else 0.0
    ymax = float(np.nanmax(all_vals)) if np.isfinite(np.nanmax(all_vals)) else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    plt.figure(figsize=(12.8, 6.4))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.set_ylim(min(0.0, ymin * 1.02), ymax * 1.12)

    draw_half_violins(ax, values, positions, keys)
    overlay_boxplots(ax, values, positions)
    paired_lines_and_points(ax, sub)

    ax.set_ylabel("Fixation index", fontsize=14)
    ax.set_xticks([POS[("Single-event", "Edge region")], POS[("Single-event", "Middle region")],
                   POS[("Recurrent", "Edge region")], POS[("Recurrent", "Middle region")]])
    ax.set_xticklabels(["Edge region", "Middle region", "Edge region", "Middle region"], fontsize=12)

    ax.text(SECTION_CENTERS["Single-event"], -0.06, "Single-event", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=13, fontweight="bold", color="#333333")
    ax.text(SECTION_CENTERS["Recurrent"], -0.06, "Recurrent", ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=13, fontweight="bold", color="#333333")

    ax.axvline(x=(POS[("Single-event", "Middle region")] + POS[("Recurrent", "Edge region")]) / 2,
               color="#dddddd", linewidth=1.0, zorder=1)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    # Legend (key) in the top right
    legend_handles = [
        Patch(facecolor=EDGE_COLOR,   edgecolor="black", label="Edge region",   alpha=ALPHA_VIOLIN),
        Patch(facecolor=MIDDLE_COLOR, edgecolor="black", label="Middle region", alpha=ALPHA_VIOLIN),
        Patch(facecolor="white", edgecolor=OVERLAY_SINGLE, hatch=".",  label="Single-event"),
        Patch(facecolor="white", edgecolor=OVERLAY_RECUR,   hatch="//", label="Recurrent"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)

    plt.tight_layout()
    out_dir = Path(__file__).resolve().parents[1] / "frf"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "frf_violin_from_url.png", dpi=150)
    plt.savefig(out_dir / "frf_violin_from_url.pdf", bbox_inches="tight")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
