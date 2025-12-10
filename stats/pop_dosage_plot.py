from __future__ import annotations

import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/inversion_population_frequencies.tsv"
DATA_PATH = Path("data/inversion_population_frequencies.tsv")
INV_PROPERTIES_PATH = Path("data/inv_properties.tsv")
# Keep this basename aligned with scripts/replicate_figures.py outputs to satisfy
# the run-analysis artifact checks.
OUTPUT_BASE = Path("special/pop_dosage_plot")

plt.rcParams.update({
    "axes.labelsize": 22,
    "xtick.labelsize": 14,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _ensure_data_file() -> Path:
    """Ensure the inversion allele frequency table is available locally."""

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return DATA_PATH

    print(f"Downloading inversion allele frequency table from {DATA_URL}...")
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


def _load_inv_properties() -> pd.DataFrame:
    """Load inversion properties with recurrence and coordinate information."""

    df = pd.read_csv(INV_PROPERTIES_PATH, sep="\t")

    lower_to_actual = {c.lower(): c for c in df.columns}

    def get_col(possible_names: list[str]) -> str:
        for name in possible_names:
            key = name.lower()
            if key in lower_to_actual:
                return lower_to_actual[key]
        raise KeyError(
            f"None of {possible_names} found in columns: {list(df.columns)}"
        )

    chrom_col = get_col(["Chromosome", "chrom", "chr"])
    start_col = get_col(["Start", "start"])
    end_col = get_col(["End", "end"])
    inversion_col = get_col(["OrigID", "Inversion", "Inv", "inversion_id"])
    recurrence_col = get_col(["0_single_1_recur_consensus", "recurrence"])

    df[recurrence_col] = pd.to_numeric(df[recurrence_col], errors="coerce")
    df = df.rename(
        columns={
            chrom_col: "Chromosome",
            start_col: "Start",
            end_col: "End",
            inversion_col: "Inversion",
            recurrence_col: "Recurrence",
        }
    )

    df = df[df["Recurrence"].isin([0, 1])].copy()

    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"] = pd.to_numeric(df["End"], errors="coerce")
    df = df[df[["Start", "End"]].notna().all(axis=1)]

    return df[["Inversion", "Chromosome", "Start", "End"]]


def _load_and_normalize() -> pd.DataFrame:
    """Load the population allele frequency table and harmonize column names."""

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
    allele_freq_col = get_col(["Allele_Freq", "allele_freq", "Allele_freq"])
    ci_lower_col = get_col(["CI95_Lower", "ci95_lower", "ci_lower"])
    ci_upper_col = get_col(["CI95_Upper", "ci95_upper", "ci_upper"])
    n_col = get_col(["N", "n", "count"])

    df = df.rename(
        columns={
            pop_col: "Population",
            inv_col: "Inversion",
            mean_col: "Mean_dosage",
            allele_freq_col: "Allele_Freq",
            ci_lower_col: "CI95_Lower",
            ci_upper_col: "CI95_Upper",
            n_col: "N",
        }
    )

    return df


def _prepare_dataframe() -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    df = _load_and_normalize()
    inv_properties = _load_inv_properties()

    df = df.merge(inv_properties, on="Inversion", how="inner", validate="m:1")
    df["Label"] = (
        df["Chromosome"].astype(str)
        + ":"
        + df["Start"].astype(str)
        + "-"
        + df["End"].astype(str)
    )

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
    required_cols = [
        "AF_Q1",
        "AF_Median",
        "AF_Q3",
        "AF_Lower_Whisker",
        "AF_Upper_Whisker",
    ]

    df = df[(df["N"] > 1)].copy()
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]

    overall = df[df["Population_display"] == "Overall"].set_index("Inversion")
    if overall.empty:
        raise ValueError("Overall population (ALL) is required for inversion ordering")

    inversions = (
        overall.sort_values("AF_Median", ascending=False, na_position="last")
        .index.to_numpy()
    )
    label_map = df.drop_duplicates("Inversion").set_index("Inversion")["Label"]
    labels = [label_map.get(inv, inv) for inv in inversions]
    pop_order = ["Overall", "AFR", "AMR", "EAS", "EUR", "MID", "SAS"]
    pop_order = [p for p in pop_order if p in df["Population_display"].unique()]

    return df, inversions, labels, pop_order


def _plot(
    df: pd.DataFrame, inversions: np.ndarray, labels: list[str], pop_order: list[str]
) -> plt.Figure:
    color_map = {
        "Overall": "#1f77b4",  # blue
        "AFR": "#ff7f0e",  # orange
        "AMR": "#2ca02c",  # green
        "EAS": "#bcbd22",  # olive
        "EUR": "#17becf",  # cyan
        "MID": "#8c564b",  # brown
        "SAS": "#d62728",  # red
    }

    for col in [
        "AF_Q1",
        "AF_Median",
        "AF_Q3",
        "AF_Lower_Whisker",
        "AF_Upper_Whisker",
    ]:
        if col not in df.columns:
            raise KeyError(f"Required column missing from dataframe: {col}")

    num_inv = len(inversions)
    fig_width = max(18, num_inv * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    x_base = np.arange(num_inv)
    group_width = 0.7
    offset_step = group_width / max(len(pop_order), 1)
    offset_start = -group_width / 2 + offset_step / 2

    for idx, pop in enumerate(pop_order):
        sub = df[df["Population_display"] == pop].set_index("Inversion")

        aligned = sub.reindex(inversions)

        stats = []
        positions = []

        for x_pos, inv in zip(x_base, inversions):
            row = aligned.loc[inv]

            q1 = row["AF_Q1"]
            med = row["AF_Median"]
            q3 = row["AF_Q3"]
            whislo = row["AF_Lower_Whisker"]
            whishi = row["AF_Upper_Whisker"]

            if np.any(pd.isna([q1, med, q3, whislo, whishi])):
                continue

            stats.append(
                {
                    "med": float(med),
                    "q1": float(q1),
                    "q3": float(q3),
                    "whislo": float(whislo),
                    "whishi": float(whishi),
                    "fliers": [],
                }
            )
            positions.append(x_pos + offset_start + idx * offset_step)

        if not stats:
            continue

        ax.bxp(
            stats,
            positions=positions,
            widths=offset_step * 0.8,
            showfliers=False,
            showcaps=False,
            patch_artist=True,
            boxprops=dict(
                facecolor=color_map.get(pop, "#333333"),
                linewidth=1.4,
                alpha=0.5,
            ),
            medianprops=dict(
                color="black",
                linewidth=1.6,
            ),
            whiskerprops=dict(
                linewidth=0,
            ),
            capprops=dict(
                linewidth=0,
            ),
        )

        ax.plot(
            [],
            [],
            linestyle="",
            marker="s",
            markersize=10,
            markerfacecolor=color_map.get(pop, "#333333"),
            markeredgecolor="none",
            label=pop,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Allele frequency")
    ax.set_ylim(0, 1)
    ax.grid(False)

    ax.set_xticks(x_base)
    ax.set_xticklabels(labels, rotation=60, ha="right")

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
    df, inversions, labels, pop_order = _prepare_dataframe()
    fig = _plot(df, inversions, labels, pop_order)

    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_BASE.with_suffix(".pdf")
    png_path = OUTPUT_BASE.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved population dosage plot to {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
