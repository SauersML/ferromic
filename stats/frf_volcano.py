# Volcano plot for FRF (centered) with within-category weight scaling.
# Downloads the data file, builds:
#   x = FST difference minus null expectation (frf_delta_centered)
#   y = -log10(two-sided normal p from z = delta_centered / SE)
# Point AREA ∝ weight share WITHIN each category (Single-event vs Recurrent).
# Single-event = GREEN, Recurrent = ORANGE, both circles.

import os
import math
import pathlib

import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import numpy as np
import pandas as pd
from math import erfc, isfinite, sqrt

def main():
    # Attempt to locate the input file locally (provided by replicate_figures.py or existing in data/)
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

    # Filter to usable rows and required fields
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

    # Standard errors and z-scores
    d["se"] = np.sqrt(d["frf_var_delta"])
    z = d["frf_delta_centered"] / d["se"]

    # Two-sided normal p using math.erfc to avoid numpy.<ufunc> dispatch on pandas
    p_vals = []
    root2 = sqrt(2.0)
    for t in z.to_numpy():
        tv = float(t)
        if isfinite(tv):
            p_vals.append(erfc(abs(tv) / root2))
        else:
            p_vals.append(np.nan)
    p = np.array(p_vals, dtype=float)
    p = np.clip(p, 1e-300, 1.0)  # guard for -log10
    d["neglog10p"] = -np.log10(p)

    # Precision weights and within-category shares
    d["w"] = 1.0 / d["frf_var_delta"]

    shares_parts = []
    for g in (0, 1):
        gmask = (d["STATUS"] == g)
        w_series = d.loc[gmask, "w"]
        # Convert to numpy before summation to avoid pandas keyword incompatibilities
        wsum = float(np.nansum(w_series.to_numpy(dtype=float)))
        if wsum > 0.0:
            shares_parts.append((w_series / wsum))
        else:
            shares_parts.append(pd.Series(np.zeros(gmask.sum(), dtype=float), index=w_series.index))
    d["share_within"] = pd.concat(shares_parts).sort_index()

    # Point area ∝ within-category share (scatter 's' expects points^2)
    area_per_unit_share = 5000.0
    min_area = 16.0
    d["area"] = np.maximum(min_area, d["share_within"] * area_per_unit_share)

    # Colors: 0=Single-event (green), 1=Recurrent (orange)
    color_map = {0: "#2ca02c", 1: "#ff7f0e"}
    d["color"] = d["STATUS"].map(color_map)

    fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)

    # Plot recurrent first, then single-event on top; both circles; 75% opaque
    for g in (1, 0):
        gdf = d.loc[d["STATUS"] == g]
        ax.scatter(
            gdf["frf_delta_centered"].to_numpy(),
            gdf["neglog10p"].to_numpy(),
            s=gdf["area"].to_numpy(),
            c=gdf["color"].to_numpy(),
            marker="o",
            linewidths=0.4,
            edgecolors="white",
            alpha=0.75,
            label=("Recurrent" if g == 1 else "Single-event"),
            zorder=(2 if g == 0 else 1),
        )

    # Reference line at zero effect
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="-", zorder=0)

    # Axis labels (no code variable names on the figure)
    ax.set_xlabel("FST difference minus null expectation")
    ax.set_ylabel(r"$-\log_{10}$ p-value")

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Symmetric x-limits around zero if possible
    x_abs = np.nanmax(np.abs(d["frf_delta_centered"].to_numpy()))
    if np.isfinite(x_abs) and x_abs > 0:
        ax.set_xlim(-1.05 * x_abs, 1.05 * x_abs)

    from matplotlib.lines import Line2D

    # Color (category) legend in upper-left with standardized marker size
    cat_handles = [
        Line2D(
            [], [], linestyle="",
            marker="o", markersize=9,
            markerfacecolor=color_map[0], markeredgecolor="white", markeredgewidth=0.6,
            label="Single-event",
        ),
        Line2D(
            [], [], linestyle="",
            marker="o", markersize=9,
            markerfacecolor=color_map[1], markeredgecolor="white", markeredgewidth=0.6,
            label="Recurrent",
        ),
    ]
    cat_legend = ax.legend(
        handles=cat_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        title=None,
        borderpad=0.6,
        labelspacing=0.5,
        handletextpad=0.6,
        handlelength=1.0,
        fontsize=10,
    )
    ax.add_artist(cat_legend)

    # Size legend (weight share within category) in upper-right
    share_levels = np.array([0.01, 0.05, 0.10], dtype=float)
    size_areas = np.maximum(min_area, share_levels * area_per_unit_share)
    size_labels = [f"{int(round(s*100))}%" for s in share_levels]

    size_handles = [
        Line2D(
            [], [], linestyle="",
            marker="o",
            markersize=math.sqrt(a),  # sqrt(area) -> legend marker size in points
            markerfacecolor="#AAAAAA",
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.6,
            label=lab,
        )
        for a, lab in zip(size_areas, size_labels)
    ]
    size_legend = ax.legend(
        handles=size_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        title="Weight share within category",
        borderpad=0.6,
        labelspacing=0.9,
        handletextpad=0.8,
        handlelength=1.0,
        fontsize=10,
        title_fontsize=10,
    )
    ax.add_artist(size_legend)

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "frf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "frf_volcano.png"
    out_pdf = out_dir / "frf_volcano.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

if __name__ == "__main__":
    main()
