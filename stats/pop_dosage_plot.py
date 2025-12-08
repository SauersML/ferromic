from __future__ import annotations

import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

DATA_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/inversion_population_frequencies.tsv"
DATA_PATH = Path("data/inversion_population_frequencies.tsv")
OUTPUT_PATH = Path("special/pop_dosage_plot.pdf")

plt.rcParams.update({
    "axes.labelsize": 22,
    "xtick.labelsize": 14,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _ensure_data_file() -> Path:
    """Ensure the inversion dosage summary table is available locally."""

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return DATA_PATH

    print(f"Downloading inversion dosage table from {DATA_URL}...")
    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request) as response, DATA_PATH.open("wb") as handle:
            if response.status != 200:
                raise HTTPError(DATA_URL, response.status, "Bad status", response.headers, None)
            shutil.copyfileobj(response, handle)
    except (URLError, HTTPError) as exc:
        if DATA_PATH.exists():
            DATA_PATH.unlink()
        raise RuntimeError(f"Failed to download {DATA_URL}") from exc

    return DATA_PATH


def _load_and_normalize() -> pd.DataFrame:
    """Load the population dosage table and harmonize column names."""

    tsv_path = _ensure_data_file()
    df = pd.read_csv(tsv_path, sep="\t")

    lower_to_actual = {c.lower(): c for c in df.columns}

    def get_col(possible_names):
        for name in possible_names:
            key = name.lower()
            if key in lower_to_actual:
                return lower_to_actual[key]
        raise KeyError(f"None of {possible_names} found in columns: {list(df.columns)}")

    pop_col = get_col(["Population", "population", "Pop"])
    inv_col = get_col(["Inversion", "inversion", "Inv"])
    mean_col = get_col(["Mean_dosage", "Mean_Dosage", "mean_dosage"])
    n_col = get_col(["N", "n", "count"])
    std_col = get_col(["Dosage_STD_All", "Dosage_STD", "dosage_std_all", "dosage_std"])

    df = df.rename(
        columns={
            pop_col: "Population",
            inv_col: "Inversion",
            mean_col: "Mean_dosage",
            n_col: "N",
            std_col: "Dosage_STD_All",
        }
    )

    return df


def _prepare_dataframe() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = _load_and_normalize()

    pop_display_map = {
        "ALL": "Overall",
        "afr": "AFR",
        "amr": "AMR",
        "eas": "EAS",
        "eur": "EUR",
        "mid": "MID",
        "sas": "SAS",
    }

    df["Population_display"] = df["Population"].map(pop_display_map).fillna(df["Population"])
    df = df[(df["N"] > 1) & df["Dosage_STD_All"].notna()].copy()

    df["SE"] = df["Dosage_STD_All"] / np.sqrt(df["N"])
    df["t_crit"] = t.ppf(0.975, df["N"] - 1)
    df["margin"] = df["t_crit"] * df["SE"]
    df["CI_lower"] = df["Mean_dosage"] - df["margin"]
    df["CI_upper"] = df["Mean_dosage"] + df["margin"]

    inversions = np.sort(df["Inversion"].unique())
    pop_order = ["Overall", "AFR", "AMR", "EAS", "EUR", "MID", "SAS"]
    pop_order = [p for p in pop_order if p in df["Population_display"].unique()]

    return df, inversions, pop_order


def _plot(df: pd.DataFrame, inversions: np.ndarray, pop_order: list[str]) -> plt.Figure:
    color_map = {
        "Overall": "#1f77b4",  # blue
        "AFR": "#ff7f0e",  # orange
        "AMR": "#2ca02c",  # green
        "EAS": "#bcbd22",  # olive
        "EUR": "#17becf",  # cyan
        "MID": "#8c564b",  # brown
        "SAS": "#d62728",  # red
    }

    num_inv = len(inversions)
    fig_width = max(18, num_inv * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    x_base = np.arange(num_inv)
    group_width = 0.7
    offset_step = group_width / max(len(pop_order), 1)
    offset_start = -group_width / 2 + offset_step / 2

    for idx, pop in enumerate(pop_order):
        sub = df[df["Population_display"] == pop].set_index("Inversion")

        means = sub.reindex(inversions)["Mean_dosage"]
        lower = sub.reindex(inversions)["CI_lower"]
        upper = sub.reindex(inversions)["CI_upper"]

        yerr = np.vstack((means - lower, upper - means))
        x_positions = x_base + offset_start + idx * offset_step

        ax.errorbar(
            x_positions,
            means,
            yerr=yerr,
            fmt="o",
            markersize=10,
            elinewidth=1.6,
            capsize=4,
            color=color_map.get(pop, "#333333"),
            label=pop,
            linestyle="none",
            alpha=0.5,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Mean dosage")
    ax.grid(False)

    ax.set_xticks(x_base)
    ax.set_xticklabels(inversions, rotation=60, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(right=0.82, bottom=0.3)
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    return fig


def main() -> None:
    df, inversions, pop_order = _prepare_dataframe()
    fig = _plot(df, inversions, pop_order)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved population dosage plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
