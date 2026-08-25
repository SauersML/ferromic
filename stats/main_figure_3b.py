#!/usr/bin/env python3
"""Render the corrected main-manuscript Figure 3B.

The panel shows the gene-level difference in CDS pair identity between inverted
and direct haplotypes. P-values and direct permutation-FDR q-values come from
the inversion-level joint-label permutation analysis in
``data/cds_permutation_joint_control.tsv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from _figstyle import NEUTRAL, apply


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "cds_permutation_joint_control.tsv"
DEFAULT_PAIR_COUNTS = ROOT / "data" / "per_gene_cds_permutation.tsv"
DEFAULT_OUTPUT = ROOT / "data" / "main_figure_3b.pdf"
FDR_ALPHA = 0.05
EXPECTED_GENES = 130
EXPECTED_SIGNIFICANT = 13


def spread_labels(values: np.ndarray, lower: float, upper: float, gap: float) -> np.ndarray:
    """Return ordered, bounded label positions with a minimum vertical gap."""
    order = np.argsort(values)
    placed = np.clip(np.asarray(values, dtype=float), lower, upper)
    for previous, current in zip(order[:-1], order[1:]):
        placed[current] = max(placed[current], placed[previous] + gap)
    overflow = placed[order[-1]] - upper
    if overflow > 0:
        placed[order] -= overflow
    for previous, current in zip(order[-2::-1], order[:0:-1]):
        placed[previous] = min(placed[previous], placed[current] - gap)
    underflow = lower - placed[order[0]]
    if underflow > 0:
        placed[order] += underflow
    return placed


def load_results(path: Path, pair_counts_path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t")
    required = {"gene_name", "recurrence", "delta", "joint_p", "direct_fdr_q"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    if len(data) != EXPECTED_GENES:
        raise ValueError(f"Expected {EXPECTED_GENES} tested genes, found {len(data)}")
    if data["joint_p"].isna().any() or (data["joint_p"] <= 0).any():
        raise ValueError("Every plotted gene must have a finite, positive permutation p-value")
    significant = data["direct_fdr_q"] < FDR_ALPHA
    if int(significant.sum()) != EXPECTED_SIGNIFICANT:
        raise ValueError(
            f"Expected {EXPECTED_SIGNIFICANT} genes at FDR q < {FDR_ALPHA}, "
            f"found {int(significant.sum())}"
        )

    pair_counts = pd.read_csv(pair_counts_path, sep="\t")
    pair_columns = {"gene_name", "inv_id", "k_direct", "k_inverted"}
    missing_pair_columns = pair_columns - set(pair_counts.columns)
    if missing_pair_columns:
        raise ValueError(
            f"Missing pair-count columns in {pair_counts_path}: "
            f"{sorted(missing_pair_columns)}"
        )
    direct_n = pd.to_numeric(pair_counts["k_direct"], errors="raise")
    inverted_n = pd.to_numeric(pair_counts["k_inverted"], errors="raise")
    pair_counts["total_pairs"] = (
        direct_n * (direct_n - 1) / 2 + inverted_n * (inverted_n - 1) / 2
    )
    data = data.merge(
        pair_counts[["gene_name", "inv_id", "total_pairs"]],
        on=["gene_name", "inv_id"],
        how="left",
        validate="one_to_one",
    )
    if data["total_pairs"].isna().any():
        missing_genes = sorted(data.loc[data["total_pairs"].isna(), "gene_name"].unique())
        raise ValueError(f"Missing CDS pair counts for genes: {missing_genes}")
    return data


def render(data: pd.DataFrame, output: Path, *, composite: bool = False) -> None:
    apply(base_size=12)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    plotted = data.copy()
    plotted["minus_log10_p"] = -np.log10(plotted["joint_p"])
    significant = plotted["direct_fdr_q"] < FDR_ALPHA
    original_colors = {"single-event": "#006400", "recurrent": "#FF6F00"}
    colors = plotted["recurrence"].map(original_colors).fillna(NEUTRAL)
    threshold_p = float(plotted.loc[significant, "joint_p"].max())
    threshold_y = -np.log10(threshold_p)

    pair_max = float(plotted["total_pairs"].max())
    point_sizes = 2 * (60 + 400 * plotted["total_pairs"] / pair_max)

    fig, ax = plt.subplots(
        figsize=(6.0, 4.2) if composite else (8.4, 6.2),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhspan(threshold_y, plotted["minus_log10_p"].max() + 0.65,
               color="#f5f5f5", zorder=0)
    ax.axhline(threshold_y, color="#777777", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color="#777777", lw=0.9, zorder=1)

    ax.scatter(
        plotted.loc[~significant, "delta"],
        plotted.loc[~significant, "minus_log10_p"],
        c=colors[~significant],
        s=point_sizes[~significant],
        alpha=0.36,
        edgecolors="white",
        linewidths=0.65,
        rasterized=False,
        zorder=2,
    )
    ax.scatter(
        plotted.loc[significant, "delta"],
        plotted.loc[significant, "minus_log10_p"],
        c=colors[significant],
        s=point_sizes[significant],
        alpha=0.70,
        edgecolors="white",
        linewidths=0.75,
        zorder=4,
    )

    x_min = float(plotted["delta"].min()) - 0.16
    x_max = float(plotted["delta"].max()) + 0.16
    y_max = float(plotted["minus_log10_p"].max()) + (2.0 if composite else 0.62)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.12, y_max)

    labelled = plotted.loc[significant].copy()
    x_range = x_max - x_min
    for side in (-1, 1):
        group = labelled.loc[np.where(labelled["delta"] < 0, -1, 1) == side].copy()
        if group.empty:
            continue
        label_y = spread_labels(
            group["minus_log10_p"].to_numpy(),
            lower=threshold_y + (0.4 if composite else 0.25),
            upper=y_max - (1.25 if composite else 0.16),
            gap=0.55 if composite else 0.34,
        )
        label_x_offset = side * x_range * 0.038
        for (_, row), y_text in zip(group.iterrows(), label_y):
            ax.annotate(
                row["gene_name"],
                xy=(row["delta"], row["minus_log10_p"]),
                xytext=(row["delta"] + label_x_offset, y_text),
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=10.2,
                fontweight="semibold",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#aaaaaa",
                    "lw": 0.65,
                    "shrinkA": 1,
                    "shrinkB": 4,
                },
                zorder=5,
            )

    ax.text(
        x_min + 0.025 * x_range,
        threshold_y + 0.09,
        "FDR q < 0.05",
        color="#666666",
        fontsize=10.5,
        va="bottom",
    )
    ax.set_xlabel("Δ proportion identical (Inverted − Direct)")
    ax.set_ylabel("−log₁₀(permutation p)")
    ax.text(-0.055, 1.02 if composite else 1.105, "B", transform=ax.transAxes,
            fontsize=22,
            fontweight="bold", ha="left", va="top")
    recurrence_legend = ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=original_colors["single-event"],
                   markersize=8, label="Single-event"),
            Line2D([], [], marker="o", ls="", color=original_colors["recurrent"],
                   markersize=8, label="Recurrent"),
        ],
        loc="upper left" if composite else "lower left",
        bbox_to_anchor=(0.04, 0.99 if composite else 1.015),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.2,
    )

    legend_levels = [0.35 * pair_max, 0.65 * pair_max, pair_max]
    size_handles = [
        ax.scatter([], [], s=2 * (60 + 400 * value / pair_max),
                   color="#b9b7c9", alpha=0.75, edgecolor="white", linewidth=0.7)
        for value in legend_levels
    ]
    size_legend = ax.legend(
        size_handles,
        [f"{int(round(value)):,} pairs" for value in legend_levels],
        title="Total pairs (circle size)",
        loc="upper left" if composite else "lower left",
        bbox_to_anchor=(0.47, 0.99 if composite else 1.015),
        frameon=False,
        handletextpad=0.7,
        labelspacing=0.8,
    )
    ax.add_artist(recurrence_legend)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pair-counts", type=Path, default=DEFAULT_PAIR_COUNTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--composite", action="store_true")
    args = parser.parse_args()
    render(load_results(args.input, args.pair_counts), args.output, composite=args.composite)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
