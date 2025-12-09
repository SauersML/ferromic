"""Compute per-population inversion allele frequencies from dosage data."""
from __future__ import annotations

import argparse
import math
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import iox
from . import run

CI_Z = 1.96


ID_CANDIDATE_COLUMNS: Sequence[str] = (
    "SampleID",
    "sample_id",
    "person_id",
    "research_id",
    "participant_id",
    "ID",
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _resolve_dosages_path(dosages_file: str | None) -> str:
    if dosages_file:
        return dosages_file
    return run._find_upwards(run.INVERSION_DOSAGES_FILE)


def _identify_id_column(header: Sequence[str]) -> str:
    match = next((col for col in ID_CANDIDATE_COLUMNS if col in header), None)
    if match is None:
        raise RuntimeError(
            "No identifier column found in the inversion dosages file. "
            f"Looked for {list(ID_CANDIDATE_COLUMNS)}."
        )
    return match


def load_all_inversion_dosages(dosages_file: str) -> pd.DataFrame:
    """Load all inversion dosages indexed by person_id with numeric columns."""
    df = pd.read_csv(dosages_file, sep="\t")
    id_col = _identify_id_column(df.columns)
    df[id_col] = df[id_col].astype(str)

    if not df[id_col].is_unique:
        df = df.drop_duplicates(subset=id_col, keep="first")

    inversion_cols = [col for col in df.columns if col != id_col]
    dosages_only = df[inversion_cols].apply(pd.to_numeric, errors="coerce")

    result = dosages_only.copy()
    result.index = df[id_col].astype(str)
    result.index.name = "person_id"

    return result


def _available_inversions(dosages: pd.DataFrame, *, include_all: bool) -> list[str]:
    return list(dosages.columns)


def _compute_frequency_stats(series: pd.Series) -> tuple[float, float, float, float, int]:
    clean = series.dropna()
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    mean_dosage = float(clean.mean())
    std_dosage = float(clean.std(ddof=1)) if n > 1 else float("nan")
    if not np.isfinite(std_dosage):
        std_dosage = float("nan")

    se_dosage = std_dosage / math.sqrt(n) if n > 1 else float("nan")
    af = mean_dosage / 2.0
    if np.isfinite(se_dosage):
        margin = CI_Z * (se_dosage / 2.0)
        lower = _clamp(af - margin)
        upper = _clamp(af + margin)
    else:
        lower = float("nan")
        upper = float("nan")
    return mean_dosage, af, lower, upper, n


def _compute_box_plot_stats(series: pd.Series) -> dict[str, float]:
    clean = series.dropna().to_numpy()
    if clean.size == 0:
        return {
            "Q1": float("nan"),
            "Median": float("nan"),
            "Q3": float("nan"),
            "Lower_Whisker": float("nan"),
            "Upper_Whisker": float("nan"),
        }

    q1, median, q3 = np.percentile(clean, [25, 50, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    sorted_vals = np.sort(clean)
    lower_whisker = float(sorted_vals[sorted_vals >= lower_bound][0])
    upper_whisker = float(sorted_vals[sorted_vals <= upper_bound][-1])

    return {
        "Q1": float(q1),
        "Median": float(median),
        "Q3": float(q3),
        "Lower_Whisker": lower_whisker,
        "Upper_Whisker": upper_whisker,
    }


def summarize_population_frequencies(
    dosages: pd.DataFrame,
    ancestry: pd.Series,
    *,
    include_all_inversions: bool = False,
) -> pd.DataFrame:
    """Return allele frequency summary per inversion and population."""
    merged = dosages.join(ancestry.rename("Population"), how="inner")
    populations: Iterable[str] = sorted({pop for pop in merged["Population"].dropna().unique()})
    populations = list(populations) + ["ALL"]

    inversion_cols = _available_inversions(merged.drop(columns=["Population"]), include_all=include_all_inversions)

    records: list[dict[str, object]] = []
    for inversion in inversion_cols:
        overall_std = float(merged[inversion].std(ddof=1)) if len(merged[inversion].dropna()) > 1 else float("nan")
        if not np.isfinite(overall_std):
            overall_std = float("nan")
        for population in populations:
            if population == "ALL":
                subset = merged[inversion]
            else:
                subset = merged.loc[merged["Population"] == population, inversion]

            mean_dosage, af, lower, upper, n = _compute_frequency_stats(subset)
            dosage_box = _compute_box_plot_stats(subset)
            allele_box = _compute_box_plot_stats(subset / 2.0)
            records.append(
                {
                    "Inversion": inversion,
                    "Population": population,
                    "N": n,
                    "Mean_Dosage": mean_dosage,
                    "Allele_Freq": af,
                    "CI95_Lower": lower,
                    "CI95_Upper": upper,
                    "Dosage_STD_All": overall_std,
                    "Dosage_Q1": dosage_box["Q1"],
                    "Dosage_Median": dosage_box["Median"],
                    "Dosage_Q3": dosage_box["Q3"],
                    "Dosage_Lower_Whisker": dosage_box["Lower_Whisker"],
                    "Dosage_Upper_Whisker": dosage_box["Upper_Whisker"],
                    "AF_Q1": allele_box["Q1"],
                    "AF_Median": allele_box["Median"],
                    "AF_Q3": allele_box["Q3"],
                    "AF_Lower_Whisker": allele_box["Lower_Whisker"],
                    "AF_Upper_Whisker": allele_box["Upper_Whisker"],
                }
            )

    result = pd.DataFrame.from_records(records)
    result["Inversion"] = result["Inversion"].astype(str)
    result["Population"] = result["Population"].astype(str)
    result = result.sort_values(["Inversion", "Population"]).reset_index(drop=True)
    return result


def _load_ancestry_and_filter_related(
    *,
    ancestry_uri: str,
    relatedness_uri: str,
    gcp_project: str,
    dosages_index: pd.Index,
    keep_related: bool,
) -> pd.Series:
    ancestry_df = iox.load_ancestry_labels(gcp_project=gcp_project, LABELS_URI=ancestry_uri)
    ancestry_df = ancestry_df.reindex(dosages_index)

    if keep_related:
        return ancestry_df["ANCESTRY"].dropna()

    related = iox.load_related_to_remove(gcp_project=gcp_project, RELATEDNESS_URI=relatedness_uri)
    return ancestry_df.loc[~ancestry_df.index.isin(related), "ANCESTRY"].dropna()


def write_population_frequencies(
    *,
    dosages_file: str | None,
    output_file: str,
    ancestry_uri: str,
    relatedness_uri: str,
    keep_related: bool = False,
    include_all_inversions: bool = False,
) -> pd.DataFrame:
    dosages_path = _resolve_dosages_path(dosages_file)
    dosages_df = load_all_inversion_dosages(dosages_path)

    if "GOOGLE_PROJECT" not in os.environ:
        raise EnvironmentError("GOOGLE_PROJECT environment variable is required to load ancestry labels.")
    gcp_project = os.environ["GOOGLE_PROJECT"]

    ancestry_series = _load_ancestry_and_filter_related(
        ancestry_uri=ancestry_uri,
        relatedness_uri=relatedness_uri,
        gcp_project=gcp_project,
        dosages_index=dosages_df.index,
        keep_related=keep_related,
    )

    aligned = dosages_df.loc[dosages_df.index.intersection(ancestry_series.index)]
    freq_df = summarize_population_frequencies(
        aligned,
        ancestry_series,
        include_all_inversions=include_all_inversions,
    )
    freq_df.to_csv(output_file, sep="\t", index=False)
    return freq_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute per-population inversion allele frequencies.")
    parser.add_argument(
        "--dosages-file",
        type=str,
        default=None,
        help=(
            "Path to the inversion dosage TSV. Defaults to searching for "
            f"{run.INVERSION_DOSAGES_FILE} in the current or parent directories."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inversion_population_frequencies.tsv",
        help="Destination TSV path for the allele frequency summary.",
    )
    parser.add_argument(
        "--ancestry-uri",
        type=str,
        default=run.PCS_URI,
        help="URI for the ancestry labels TSV used in the main PheWAS pipeline.",
    )
    parser.add_argument(
        "--relatedness-uri",
        type=str,
        default=run.RELATEDNESS_URI,
        help="URI for the relatedness list used to exclude related individuals.",
    )
    parser.add_argument(
        "--keep-related",
        action="store_true",
        help="Keep related individuals instead of excluding them from the analysis.",
    )
    parser.add_argument(
        "--all-inversions",
        action="store_true",
        help=(
            "Summarize all inversions present in the dosages file instead of restricting "
            "to the configured TARGET_INVERSIONS set."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    freq_df = write_population_frequencies(
        dosages_file=args.dosages_file,
        output_file=args.output,
        ancestry_uri=args.ancestry_uri,
        relatedness_uri=args.relatedness_uri,
        keep_related=args.keep_related,
        include_all_inversions=args.all_inversions,
    )
    print(f"Wrote {len(freq_df)} rows to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
