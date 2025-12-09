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

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# File locations
CALLSET_PATH = "data/2AGRCh38_unifiedCallset - 2AGRCh38_unifiedCallset.tsv"
INV_PROPERTIES_PATH = "data/inv_properties.tsv"
AOU_FREQUENCIES_PATH = "data/inversion_population_frequencies.tsv"
OUTPUT_BASE = Path("special/overall_AF_scatterplot")

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
    required_cols = ["OrigID", "Chromosome", "Start", "End", "0_single_1_recur_consensus"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"inv_properties.tsv missing columns: {missing}")

    df["0_single_1_recur_consensus"] = pd.to_numeric(df["0_single_1_recur_consensus"], errors="coerce")
    keep = df["0_single_1_recur_consensus"].isin([0, 1])
    df = df.loc[keep, required_cols].copy()
    df = df.drop_duplicates(subset=["OrigID"])
    df["coordinate"] = df.apply(lambda r: f"{r['Chromosome']}:{int(r['Start'])}-{int(r['End'])}", axis=1)
    return df


def compute_callset_af(row: pd.Series, genotype_cols: Iterable[str]) -> tuple[float, float, float]:
    alt_alleles = 0
    valid_calls = 0

    for col in genotype_cols:
        gt = row[col]
        if pd.isna(gt):
            continue
        gt_str = str(gt)
        if gt_str.lower() == "nan":
            continue
        if gt_str in ALLOWED_GENOTYPES:
            alt_alleles += ALLOWED_GENOTYPES[gt_str]
            valid_calls += 1

    if valid_calls == 0:
        return (float("nan"), float("nan"), float("nan"))

    trials = 2 * valid_calls
    af = alt_alleles / trials
    ci_low, ci_high = wilson_ci(alt_alleles, trials)
    return (af, ci_low, ci_high)


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
        af, ci_low, ci_high = compute_callset_af(row, genotype_cols)
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
    df.rename(columns={
        "Inversion": "OrigID",
        "Allele_Freq": "aou_af",
        "CI95_Lower": "aou_ci_low",
        "CI95_Upper": "aou_ci_high",
    }, inplace=True)
    df = df.drop_duplicates(subset=["OrigID"])
    return df


def plot_scatter(data: pd.DataFrame, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data["callset_af"], data["aou_af"], s=70, color="#1f77b4")

    r = np.corrcoef(data["callset_af"], data["aou_af"])[0, 1]
    ax.annotate(f"r = {r:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top", fontsize=16)

    ax.set_xlabel("Porubsky et al. 2022 Callset Allele Frequency", fontsize=16)
    ax.set_ylabel("All of Us Cohort Imputed Allele Frequency", fontsize=16)
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

    ax.errorbar(
        data["callset_af"],
        data["aou_af"],
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        markersize=6,
        ecolor="#4a4a4a",
        elinewidth=1.0,
        capsize=3,
        color="#2ca02c",
    )

    r = np.corrcoef(data["callset_af"], data["aou_af"])[0, 1]
    ax.annotate(f"r = {r:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top", fontsize=16)

    ax.set_xlabel("Porubsky et al. 2022 Callset Allele Frequency", fontsize=16)
    ax.set_ylabel("All of Us Cohort Imputed Allele Frequency", fontsize=16)
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

    merged = (
        inv_props[["OrigID", "coordinate"]]
        .merge(callset_df, on="OrigID", how="inner")
        .merge(aou_df, on="OrigID", how="inner")
    )

    merged = merged.dropna(subset=["callset_af", "aou_af"])

    if merged.empty:
        raise ValueError("No overlapping inversions after filtering by consensus and availability.")

    plot_scatter(merged, str(OUTPUT_BASE))
    plot_scatter_with_ci(merged, str(OUTPUT_BASE.with_name(f"{OUTPUT_BASE.name}_with_ci")))


if __name__ == "__main__":
    main()
