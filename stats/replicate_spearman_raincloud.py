"""Generate the Spearman decay raincloud plot from saved statistics."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _load_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Missing Spearman decay points. Run stats/replicate_manuscript_statistics.py first "
            f"to populate {path}."
        )

    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise ValueError("Spearman decay points table is empty; nothing to plot.")

    df["label"] = df["recurrence_flag"].map({0: "Single-event", 1: "Recurrent"})
    df["is_significant"] = df["q_value"].apply(lambda q: pd.notna(q) and q < 0.05)
    df["alpha"] = np.where(df["is_significant"], 0.6, 0.3)
    df["size"] = np.where(df["is_significant"], 320, 160)
    return df


def _plot(df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    palette = {"Recurrent": "#f28e2b", "Single-event": "#59a14f"}

    sns.set_theme(context="talk", style="white")
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.kdeplot(
        data=df,
        x="rho",
        hue="label",
        fill=True,
        common_norm=False,
        alpha=0.35,
        palette=palette,
        linewidth=3.2,
        ax=ax,
    )

    y_jitter = -0.035 + np.random.normal(loc=0.0, scale=0.005, size=len(df))
    sig = df["is_significant"].to_numpy()

    ax.scatter(
        df.loc[~sig, "rho"],
        y_jitter[~sig],
        c=df.loc[~sig, "label"].map(palette),
        alpha=df.loc[~sig, "alpha"],
        s=df.loc[~sig, "size"],
        edgecolors="black",
        linewidths=1.1,
        marker="o",
        zorder=5,
    )

    ax.scatter(
        df.loc[sig, "rho"],
        y_jitter[sig],
        c=df.loc[sig, "label"].map(palette),
        alpha=df.loc[sig, "alpha"],
        s=df.loc[sig, "size"],
        edgecolors="black",
        linewidths=1.3,
        marker="x",
        zorder=6,
    )


    ax.set_xlabel("Spearman correlation", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylim(bottom=min(ax.get_ylim()[0], -0.08))
    ax.grid(False)

    ax.axhline(0, color="gray", lw=1.0, alpha=0.6)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    x_min, x_max = ax.get_xlim()
    y_max = ax.get_ylim()[1]
    arrow_y = y_max * 0.92
    ax.annotate(
        "Higher FST near breakpoints ",
        xy=(x_max * 0.95, arrow_y),
        xytext=(x_min + (x_max - x_min) * 0.55, arrow_y),
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 2.2},
        ha="right",
        va="center",
        fontsize=17,
        color="gray",
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Recurrent",
            markerfacecolor=palette["Recurrent"],
            markeredgecolor="black",
            markersize=18,
            alpha=0.6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Single-event",
            markerfacecolor=palette["Single-event"],
            markeredgecolor="black",
            markersize=18,
            alpha=0.6,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            label="FDR < 0.05",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=22,
            alpha=0.8,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        title="Key",
        frameon=True,
        fontsize=15,
        title_fontsize=16,
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points_path = DATA_DIR / "spearman_decay_points.tsv"
    df = _load_points(points_path)
    png_path = points_path.with_name("spearman_decay_raincloud.png")
    pdf_path = points_path.with_name("spearman_decay_raincloud.pdf")
    _plot(df, png_path, pdf_path)
    print(f"Saved raincloud plot to {png_path} and {pdf_path}.")


if __name__ == "__main__":
    main()
