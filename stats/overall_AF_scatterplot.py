"""Generate allele frequency scatterplots combining callset and imputed cohorts.

This script merges inversion metadata, allele frequencies from the Porubsky et al.
2022 callset, and imputed frequencies from the All of Us cohort. It filters to
inversions with consensus recurrence values of 0 or 1 and produces two scatter
plots: one with point estimates only and one including 95% confidence intervals
for both axes.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Matplotlib default blue/orange palette for consistency across figures.
# Keep these inline (instead of importing a shared style module) to avoid
# introducing a dependency that is absent in the CI/replication environment.
CALLSET_POINT_COLOR = "#1f77b4"
IMPUTED_POINT_COLOR = "#ff7f0e"
# Used for inversions with low imputation performance.
LOW_R2_COLOR = "#7f7f7f"

# Distinguish single-event vs recurrent inversions.
SINGLE_EVENT_COLOR = "#1b9e77"  # darker green
RECURRENT_EVENT_COLOR = "#d95f02"  # darker orange

# Reduced transparency and larger markers to improve visibility.
POINT_ALPHA = 0.9
LOW_R2_ALPHA = 0.7
SCATTER_SIZE = 90
ERRORBAR_MARKERSIZE = 8

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# File locations
CALLSET_PATH = "data/2AGRCh38_unifiedCallset - 2AGRCh38_unifiedCallset.tsv"
INV_PROPERTIES_PATH = "data/inv_properties.tsv"
AOU_FREQUENCIES_PATH = "data/inversion_population_frequencies.tsv"
OUTPUT_BASE = Path("special/overall_AF_scatterplot")
IMPUTATION_RESULTS_PATH = "data/imputation_results_merged.tsv"

# Allowed diploid genotypes and the number of alternate alleles they carry
ALLOWED_GENOTYPES = {
    "1|1": 2,
    "1|0": 1,
    "0|1": 1,
    "0|0": 0,
}


def _ensure_exists(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1 + z**2 / trials
    center = phat + z**2 / (2 * trials)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * trials)) / trials)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def load_inv_properties() -> pd.DataFrame:
    df = pd.read_csv(_ensure_exists(INV_PROPERTIES_PATH), sep="\t")
    required_cols = [
        "OrigID",
        "Chromosome",
        "Start",
        "End",
        "0_single_1_recur_consensus",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"inv_properties.tsv missing columns: {missing}")

    df["0_single_1_recur_consensus"] = pd.to_numeric(
        df["0_single_1_recur_consensus"], errors="coerce"
    )
    keep = df["0_single_1_recur_consensus"].isin([0, 1])
    df = df.loc[keep, required_cols].copy()
    df = df.drop_duplicates(subset=["OrigID"])
    df["coordinate"] = df.apply(lambda r: f"{r['Chromosome']}:{int(r['Start'])}-{int(r['End'])}", axis=1)
    df.rename(columns={"0_single_1_recur_consensus": "recurrence_consensus"}, inplace=True)
    return df


def compute_callset_af(
    row: pd.Series, genotype_cols: Iterable[str]
) -> tuple[float, float, float, int, int]:
    alt_alleles = 0
    valid_calls = 0

    for col in genotype_cols:
        gt = row[col]
        if pd.isna(gt):
            continue
        gt_str = str(gt).strip()
        if gt_str.lower() == "nan":
            continue
        if gt_str in ALLOWED_GENOTYPES:
            alt_alleles += ALLOWED_GENOTYPES[gt_str]
            valid_calls += 1

    if valid_calls == 0:
        return (float("nan"), float("nan"), float("nan"), 0, 0)

    trials = 2 * valid_calls
    af = alt_alleles / trials
    ci_low, ci_high = wilson_ci(alt_alleles, trials)
    return (af, ci_low, ci_high, alt_alleles, trials)


def load_callset_afs(allowed_ids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(_ensure_exists(CALLSET_PATH), sep="\t")
    if "inv_id" not in df.columns:
        raise KeyError("Callset file missing 'inv_id' column")

    df = df.drop_duplicates(subset=["inv_id"])
    non_genotype_cols = {
        "seqnames",
        "start",
        "end",
        "width",
        "inv_id",
        "arbigent_genotype",
        "misorient_info",
        "orthog_tech_support",
        "inversion_category",
        "inv_AF",
    }
    genotype_cols = [c for c in df.columns if c not in non_genotype_cols]

    records = []
    for _, row in df.iterrows():
        inv_id = row["inv_id"]
        if inv_id not in allowed_ids:
            continue
        af, ci_low, ci_high, alt_alleles, trials = compute_callset_af(
            row, genotype_cols
        )

        records.append({
            "OrigID": inv_id,
            "callset_af": af,
            "callset_ci_low": ci_low,
            "callset_ci_high": ci_high,
        })

    return pd.DataFrame.from_records(records)


def load_aou_frequencies() -> pd.DataFrame:
    df = pd.read_csv(_ensure_exists(AOU_FREQUENCIES_PATH), sep="\t")
    required_cols = ["Inversion", "Population", "Allele_Freq", "CI95_Lower", "CI95_Upper"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"inversion_population_frequencies.tsv missing columns: {missing}")

    df = df.loc[df["Population"] == "ALL", required_cols].copy()
    df.rename(
        columns={
            "Inversion": "OrigID",
            "Allele_Freq": "aou_af",
            "CI95_Lower": "aou_ci_low",
            "CI95_Upper": "aou_ci_high",
        },
        inplace=True,
    )

    for col in ("aou_af", "aou_ci_low", "aou_ci_high"):
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    df = df.drop_duplicates(subset=["OrigID"])
    return df


def load_imputation_results() -> pd.DataFrame:
    df = pd.read_csv(_ensure_exists(IMPUTATION_RESULTS_PATH), sep="\t")
    required_cols = ["OrigID", "unbiased_pearson_r2"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "imputation_results_merged.tsv missing columns: " f"{missing}"
        )

    raw_r2 = df["unbiased_pearson_r2"]
    df["unbiased_pearson_r2"] = pd.to_numeric(raw_r2, errors="coerce")
    invalid_mask = raw_r2.notna() & df["unbiased_pearson_r2"].isna()
    if invalid_mask.any():
        invalid_ids = df.loc[invalid_mask, "OrigID"].tolist()
        print(
            "Warning: non-numeric unbiased_pearson_r2 for OrigID(s): "
            f"{', '.join(map(str, invalid_ids))}. Omitting these from plots."
        )

    df = df.loc[:, required_cols].drop_duplicates(subset=["OrigID"])
    return df


def _clamp_to_unit(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=1)
    return df


def _recurrence_colors(recurrence: pd.Series) -> pd.Series:
    return recurrence.map({0: SINGLE_EVENT_COLOR, 1: RECURRENT_EVENT_COLOR})


def plot_scatter(data: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    high_r2 = data.loc[data["unbiased_pearson_r2"] > 0.5]
    low_r2 = data.loc[data["unbiased_pearson_r2"] <= 0.5]

    def _plot_subset(
        subset: pd.DataFrame, alpha: float, marker: str
    ) -> tuple[float, Line2D | None]:
        if subset.empty:
            return float("nan"), None

        colors = _recurrence_colors(subset["recurrence_consensus"])
        scatter = ax.scatter(
            subset["callset_af"],
            subset["aou_af"],
            s=SCATTER_SIZE,
            c=colors,
            alpha=alpha,
            marker=marker,
            edgecolor="black",
            linewidth=0.4,
        )

        if len(subset) >= 2:
            slope, intercept = np.polyfit(
                subset["callset_af"], subset["aou_af"], 1
            )
            x_vals = np.linspace(
                subset["callset_af"].min(), subset["callset_af"].max(), 100
            )
            ax.plot(
                x_vals, slope * x_vals + intercept, color="black", linestyle="--"
            )
            r_value = np.corrcoef(subset["callset_af"], subset["aou_af"])[0, 1]
        else:
            r_value = float("nan")
        return r_value, scatter

    r_high, _ = _plot_subset(high_r2, POINT_ALPHA, "^")
    r_low, _ = _plot_subset(low_r2, LOW_R2_ALPHA, "o")

    ax.annotate(
        (
            f"r = {r_high:.2f} (N = {len(high_r2)})"
            if not math.isnan(r_high)
            else f"r = NA (N = {len(high_r2)})"
        ),
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=16,
        color=CALLSET_POINT_COLOR,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    ax.annotate(
        (
            f"r = {r_low:.2f} (N = {len(low_r2)})"
            if not math.isnan(r_low)
            else f"r = NA (N = {len(low_r2)})"
        ),
        xy=(0.05, 0.86),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=16,
        color=LOW_R2_COLOR,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    recurrence_legend = [
        Line2D(
            [],
            [],
            marker="o",
            color="white",
            markerfacecolor=SINGLE_EVENT_COLOR,
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label="Single-event",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="white",
            markerfacecolor=RECURRENT_EVENT_COLOR,
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label="Recurrent",
        ),
    ]

    quality_legend = [
        Line2D(
            [], [], marker="^", color="black", linestyle="", markersize=9, label="r² > 0.5"
        ),
        Line2D(
            [], [], marker="o", color="black", linestyle="", markersize=9, label="r² ≤ 0.5"
        ),
    ]

    leg1 = ax.legend(
        handles=recurrence_legend,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        title="Recurrence",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=quality_legend,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        title="Imputation",
    )

    ax.set_xlabel("Porubsky et al. 2022 Callset Allele Frequency", fontsize=16)
    ax.set_ylabel("All of Us Cohort Imputed Allele Frequency", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=14)

    output_base = Path(filename)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_with_ci(data: pd.DataFrame, filename: str) -> None:
    # Ensure CI columns are ordered low→high so matplotlib receives non-negative
    # error bar lengths even if the inputs were flipped. This mirrors the
    # behaviour of `wilson_ci`, which always returns (lower, upper).
    for low_col, high_col in (
        ("callset_ci_low", "callset_ci_high"),
        ("aou_ci_low", "aou_ci_high"),
    ):
        lows = data[[low_col, high_col]].min(axis=1)
        highs = data[[low_col, high_col]].max(axis=1)
        data[low_col] = lows
        data[high_col] = highs

    fig, ax = plt.subplots(figsize=(6, 6))
    xerr = np.maximum(
        0,
        np.vstack(
            [
                data["callset_af"] - data["callset_ci_low"],
                data["callset_ci_high"] - data["callset_af"],
            ]
        ),
    )
    yerr = np.maximum(
        0,
        np.vstack(
            [
                data["aou_af"] - data["aou_ci_low"],
                data["aou_ci_high"] - data["aou_af"],
            ]
        ),
    )

    high_r2 = data.loc[data["unbiased_pearson_r2"] > 0.5]
    low_r2 = data.loc[data["unbiased_pearson_r2"] <= 0.5]

    def _errorbar_subset(
        subset: pd.DataFrame, alpha: float, marker: str
    ) -> tuple[float, Line2D | None]:
        if subset.empty:
            return float("nan"), None

        subset_xerr = xerr[:, subset.index]
        subset_yerr = yerr[:, subset.index]
        subset_positions = pd.Index(subset.index)

        err_lines: Line2D | None = None
        for rec_value, rec_color in (
            (0, SINGLE_EVENT_COLOR),
            (1, RECURRENT_EVENT_COLOR),
        ):
            rec_points = subset.loc[subset["recurrence_consensus"] == rec_value]
            if rec_points.empty:
                continue

            rec_positions = subset_positions.get_indexer(rec_points.index)
            rec_xerr = subset_xerr[:, rec_positions]
            rec_yerr = subset_yerr[:, rec_positions]

            (err_lines, _, _) = ax.errorbar(
                rec_points["callset_af"],
                rec_points["aou_af"],
                xerr=rec_xerr,
                yerr=rec_yerr,
                fmt=marker,
                markersize=ERRORBAR_MARKERSIZE,
                ecolor="#4a4a4a",
                elinewidth=1.0,
                capsize=3,
                color=rec_color,
                markerfacecolor=rec_color,
                markeredgecolor="black",
                alpha=alpha,
            )

        if len(subset) >= 2:
            slope, intercept = np.polyfit(
                subset["callset_af"], subset["aou_af"], 1
            )
            x_vals = np.linspace(
                subset["callset_af"].min(), subset["callset_af"].max(), 100
            )
            ax.plot(
                x_vals, slope * x_vals + intercept, color="black", linestyle="--"
            )
            r_value = np.corrcoef(subset["callset_af"], subset["aou_af"])[0, 1]
        else:
            r_value = float("nan")
        return r_value, err_lines

    r_high, err_high = _errorbar_subset(high_r2, POINT_ALPHA, "^")
    r_low, err_low = _errorbar_subset(low_r2, LOW_R2_ALPHA, "o")

    ax.annotate(
        (
            f"r = {r_high:.2f} (N = {len(high_r2)})"
            if not math.isnan(r_high)
            else f"r = NA (N = {len(high_r2)})"
        ),
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=16,
        color=CALLSET_POINT_COLOR,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    ax.annotate(
        (
            f"r = {r_low:.2f} (N = {len(low_r2)})"
            if not math.isnan(r_low)
            else f"r = NA (N = {len(low_r2)})"
        ),
        xy=(0.05, 0.86),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=16,
        color=LOW_R2_COLOR,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    recurrence_legend = [
        Line2D(
            [],
            [],
            marker="o",
            color="white",
            markerfacecolor=SINGLE_EVENT_COLOR,
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label="Single-event",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="white",
            markerfacecolor=RECURRENT_EVENT_COLOR,
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label="Recurrent",
        ),
    ]

    quality_legend = [
        Line2D(
            [], [], marker="^", color="black", linestyle="", markersize=9, label="r² > 0.5"
        ),
        Line2D(
            [], [], marker="o", color="black", linestyle="", markersize=9, label="r² ≤ 0.5"
        ),
    ]

    leg1 = ax.legend(
        handles=recurrence_legend,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        title="Recurrence",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=quality_legend,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        title="Imputation",
    )

    ax.set_xlabel("Porubsky et al. 2022 Callset Allele Frequency", fontsize=16)
    ax.set_ylabel("All of Us Cohort Imputed Allele Frequency", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=14)

    output_base = Path(filename)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    inv_props = load_inv_properties()
    allowed_ids = set(inv_props["OrigID"])

    callset_df = load_callset_afs(allowed_ids)
    aou_df = load_aou_frequencies()
    imputation_df = load_imputation_results()

    merged = (
        inv_props[["OrigID", "coordinate", "recurrence_consensus"]]
        .merge(callset_df, on="OrigID", how="inner")
        .merge(aou_df, on="OrigID", how="inner")
        .merge(imputation_df, on="OrigID", how="left")
    )

    merged = merged.dropna(subset=["callset_af", "aou_af"])

    merged = _clamp_to_unit(
        merged,
        ["callset_af", "callset_ci_low", "callset_ci_high", "aou_af", "aou_ci_low", "aou_ci_high"],
    )

    valid_r2_mask = merged["unbiased_pearson_r2"].notna()
    if not valid_r2_mask.all():
        missing_r2_ids = merged.loc[~valid_r2_mask, "OrigID"].tolist()
        if missing_r2_ids:
            print(
                "Warning: missing or non-numeric unbiased_pearson_r2 for OrigID(s): "
                f"{', '.join(map(str, missing_r2_ids))}. Omitting these from plots."
            )
    merged = merged.loc[valid_r2_mask]

    if merged.empty:
        raise ValueError("No overlapping inversions after filtering by consensus and availability.")

    plot_scatter(merged, str(OUTPUT_BASE))
    plot_scatter_with_ci(merged, str(OUTPUT_BASE.with_name(f"{OUTPUT_BASE.name}_with_ci")))


if __name__ == "__main__":
    main()