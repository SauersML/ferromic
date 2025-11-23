"""
Precision-weighted strip+box plot for FRF breakpoint enrichment by inversion status.

X-axis: inversion status (Single-event vs Recurrent)
Y-axis: frf_delta_centered (edge – middle FRF minus null expectation)
Point size: precision weight share within each category (1 / frf_var_delta)
"""

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "data/per_inversion_frf_effects.tsv"
COLOR_MAP = {0: "#2ca02c", 1: "#ff7f0e"}
LABELS = {0: "Single-event", 1: "Recurrent"}


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cumulative = np.cumsum(w_sorted)
    cutoff = 0.5 * np.nansum(w_sorted)
    idx = np.searchsorted(cumulative, cutoff)
    idx = min(idx, v_sorted.size - 1)
    return float(v_sorted[idx])


def main():
    df = pd.read_csv(DATA_PATH, sep="\t")
    mask = (
        df["usable_for_meta"].astype(bool)
        & df["STATUS"].isin([0, 1])
        & np.isfinite(df["frf_delta_centered"])
        & np.isfinite(df["frf_var_delta"])
        & (df["frf_var_delta"] > 0.0)
    )
    d = df.loc[mask].copy()
    if d.empty:
        raise RuntimeError("No usable rows after filtering.")

    d["w"] = 1.0 / d["frf_var_delta"]

    # Weight share within each category for visual scaling
    shares = []
    for g in (0, 1):
        g_weights = d.loc[d["STATUS"] == g, "w"]
        total = float(np.nansum(g_weights.to_numpy(dtype=float)))
        if total > 0.0:
            shares.append(g_weights / total)
        else:
            shares.append(pd.Series(np.zeros_like(g_weights), index=g_weights.index))
    d["share_within"] = pd.concat(shares).sort_index()

    rng = np.random.default_rng(seed=1)
    jitter = rng.normal(loc=0.0, scale=0.06, size=d.shape[0])
    d["x"] = d["STATUS"] + jitter

    area_per_unit_share = 5000.0
    min_area = 18.0
    d["area"] = np.maximum(min_area, d["share_within"] * area_per_unit_share)
    d["color"] = d["STATUS"].map(COLOR_MAP)

    fig, ax = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)

    # Box plots
    box_data = [d.loc[d["STATUS"] == g, "frf_delta_centered"].to_numpy() for g in (0, 1)]
    bp = ax.boxplot(
        box_data,
        positions=[0, 1],
        widths=0.28,
        patch_artist=True,
        showcaps=True,
        medianprops={"color": "#2f2f2f", "linewidth": 1.4},
        whiskerprops={"color": "#4f4f4f", "linewidth": 1.2},
        capprops={"color": "#4f4f4f", "linewidth": 1.2},
        boxprops={"linewidth": 1.2, "facecolor": "#f5f5f5", "edgecolor": "#4f4f4f"},
    )

    # Recolor boxes to match categories
    for patch, g in zip(bp["boxes"], (0, 1)):
        patch.set_facecolor(COLOR_MAP[g])
        patch.set_alpha(0.24)

    # Jittered points
    for g in (1, 0):  # plot recurrent first, single-event on top
        gdf = d.loc[d["STATUS"] == g]
        ax.scatter(
            gdf["x"].to_numpy(),
            gdf["frf_delta_centered"].to_numpy(),
            s=gdf["area"].to_numpy(),
            c=gdf["color"].to_numpy(),
            linewidths=0.5,
            edgecolors="white",
            alpha=0.9,
            zorder=(2 if g == 0 else 1),
            label=LABELS[g],
        )

    # Weighted medians as diamonds
    for g, x in zip((0, 1), (0, 1)):
        gdf = d.loc[d["STATUS"] == g]
        wm = weighted_median(gdf["frf_delta_centered"], gdf["w"])
        ax.scatter(
            [x], [wm],
            marker="D", s=90,
            color=COLOR_MAP[g], edgecolors="#2f2f2f", linewidths=0.9,
            zorder=3,
        )

    # Reference line at zero
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", zorder=0)

    # Significance bracket
    y_max = float(np.nanmax(d["frf_delta_centered"].to_numpy()))
    y_min = float(np.nanmin(d["frf_delta_centered"].to_numpy()))
    bracket_y = y_max + 0.12 * (y_max - y_min if y_max != y_min else 1.0)
    text_y = bracket_y + 0.02 * (y_max - y_min if y_max != y_min else 1.0)
    ax.plot([0, 0, 1, 1], [bracket_y - 0.005, bracket_y, bracket_y, bracket_y - 0.005], color="#4f4f4f", linewidth=1.2)
    ax.text(0.5, text_y, r"$p = 3.0\times10^{-6}$", ha="center", va="bottom", fontsize=11, color="#2f2f2f")

    # Axes and styling
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABELS[0], LABELS[1]])
    ax.set_ylabel("FST breakpoint enrichment (edge - middle, null-corrected)")
    ax.set_xlabel("Inversion type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Size legend for precision share within category
    from matplotlib.lines import Line2D

    share_levels = np.array([0.01, 0.05, 0.10], dtype=float)
    size_areas = np.maximum(min_area, share_levels * area_per_unit_share)
    size_labels = [f"{int(round(s * 100))}%" for s in share_levels]
    size_handles = [
        Line2D(
            [], [], linestyle="",
            marker="o", markersize=math.sqrt(a),
            markerfacecolor="#AAAAAA", markeredgecolor="#FFFFFF", markeredgewidth=0.6,
            label=lab,
        )
        for a, lab in zip(size_areas, size_labels)
    ]
    ax.legend(
        handles=size_handles,
        title="Weight share within category",
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        borderpad=0.6,
        labelspacing=0.8,
        handletextpad=0.8,
        handlelength=1.0,
        fontsize=9,
        title_fontsize=9,
    )

    out_dir = Path(__file__).resolve().parents[2] / "frf"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "frf_delta_strip_box.png", dpi=300)
    fig.savefig(out_dir / "frf_delta_strip_box.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
