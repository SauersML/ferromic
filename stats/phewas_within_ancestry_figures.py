"""Summarize and plot the within-ancestry-PC PheWAS sensitivity analysis.

The existing ancestry-stratified estimates in ``data/phewas_results.tsv`` used
the 16 global All of Us genetic principal components.  The six tables under
``data/phewas_within_ancestry`` repeat the analysis after fitting 16 components
separately inside each All of Us genetic-ancestry group.  This script compares
those two otherwise matched estimates for the 39 pooled FDR-significant
inversion-phenotype associations (37 unique phenotypes).

Outputs
-------
data/phewas_within_ancestry_correspondence.tsv
results/phewas_within_ancestry/correspondence_summary.tsv
results/phewas_within_ancestry/correspondence_summary.json
results/phewas_within_ancestry/effect_correspondence.pdf and .png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _figstyle import CATEGORICAL, apply


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
INPUT_DIR = DATA / "phewas_within_ancestry"
RESULT_DIR = REPO / "results" / "phewas_within_ancestry"

ORIGINAL_RESULTS = DATA / "phewas_results.tsv"
OUT_COMPARISON = DATA / "phewas_within_ancestry_correspondence.tsv"
OUT_SUMMARY = RESULT_DIR / "correspondence_summary.tsv"
OUT_JSON = RESULT_DIR / "correspondence_summary.json"
OUT_CORRESPONDENCE_PDF = RESULT_DIR / "effect_correspondence.pdf"
OUT_CORRESPONDENCE_PNG = RESULT_DIR / "effect_correspondence.png"

POPULATIONS = (
    ("EUR", "European"),
    ("AFR", "African"),
    ("AMR", "Admixed American"),
    ("EAS", "East Asian"),
    ("SAS", "South Asian"),
    ("MID", "Middle Eastern"),
)

POPULATION_COLORS = {
    code: CATEGORICAL[index]
    for index, (code, _) in enumerate(POPULATIONS)
}

LOCUS_LABELS = {
    "chr8-7301025-INV-5297356": "8p23.1",
    "chr17-45585160-INV-706887": "17q21.31",
}

REQUIRED_NEW_COLUMNS = {
    "Phenotype",
    "Inversion",
    "Beta",
    "OR",
    "P_Value",
    "P_Valid",
    "Q_GLOBAL",
    "N_Total",
    "N_Cases",
    "N_Controls",
    "CI_Valid",
    "CI_LO_OR",
    "CI_HI_OR",
}


def _boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().eq("true")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def load_original_hits() -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_RESULTS, sep="\t", low_memory=False)
    required = {"Phenotype", "Inversion", "Sig_Global", "OR", "Q_GLOBAL"}
    missing = required - set(original.columns)
    if missing:
        raise ValueError(f"{ORIGINAL_RESULTS} is missing {sorted(missing)}")

    hits = original[_boolean(original["Sig_Global"])].copy()
    duplicated = hits.duplicated(["Inversion", "Phenotype"], keep=False)
    if duplicated.any():
        raise ValueError("Original FDR-significant associations are not unique")
    if len(hits) != 39 or hits["Phenotype"].nunique() != 37:
        raise ValueError(
            "Expected 39 original associations spanning 37 phenotypes; found "
            f"{len(hits)} associations and {hits['Phenotype'].nunique()} phenotypes"
        )
    return hits


def load_new_population(population: str) -> pd.DataFrame:
    path = INPUT_DIR / f"phewas_{population.lower()}_within_ancestry_pcs.tsv"
    table = pd.read_csv(path, sep="\t", low_memory=False)
    missing = REQUIRED_NEW_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    if table.duplicated(["Inversion", "Phenotype"]).any():
        raise ValueError(f"{path} has duplicate inversion-phenotype rows")
    if not (
        pd.to_numeric(table["N_Total"], errors="coerce")
        == pd.to_numeric(table["N_Cases"], errors="coerce")
        + pd.to_numeric(table["N_Controls"], errors="coerce")
    ).all():
        raise ValueError(f"{path} contains inconsistent sample counts")
    return table


def build_comparison() -> pd.DataFrame:
    hits = load_original_hits()
    records = []

    for population, population_label in POPULATIONS:
        new = load_new_population(population)
        old_columns = [
            "Inversion",
            "Phenotype",
            "OR",
            "Q_GLOBAL",
            f"{population}_OR",
            f"{population}_P",
            f"{population}_CI_LO_OR",
            f"{population}_CI_HI_OR",
        ]
        missing = set(old_columns) - set(hits.columns)
        if missing:
            raise ValueError(f"{ORIGINAL_RESULTS} is missing {sorted(missing)}")

        old = hits[old_columns].rename(
            columns={
                "OR": "pooled_or",
                "Q_GLOBAL": "pooled_q",
                f"{population}_OR": "existing_or",
                f"{population}_P": "existing_p",
                f"{population}_CI_LO_OR": "existing_ci_lo_or",
                f"{population}_CI_HI_OR": "existing_ci_hi_or",
            }
        )

        new_columns = [
            "Inversion",
            "Phenotype",
            "Beta",
            "OR",
            "P_Value",
            "P_Valid",
            "Q_GLOBAL",
            "N_Total",
            "N_Cases",
            "N_Controls",
            "CI_Valid",
            "CI_Sided",
            "CI_LO_OR",
            "CI_HI_OR",
        ]
        if "Skip_Reason" in new.columns:
            new_columns.append("Skip_Reason")
        new = new[new_columns].rename(
            columns={
                "Beta": "within_beta",
                "OR": "within_or",
                "P_Value": "within_p",
                "P_Valid": "within_p_valid",
                "Q_GLOBAL": "within_q_selected_set",
                "N_Total": "within_n_total",
                "N_Cases": "within_n_cases",
                "N_Controls": "within_n_controls",
                "CI_Valid": "within_ci_valid",
                "CI_Sided": "within_ci_sided",
                "CI_LO_OR": "within_ci_lo_or",
                "CI_HI_OR": "within_ci_hi_or",
                "Skip_Reason": "within_skip_reason",
            }
        )

        joined = old.merge(
            new,
            on=["Inversion", "Phenotype"],
            how="left",
            validate="one_to_one",
        )
        joined.insert(0, "population", population)
        joined.insert(1, "population_label", population_label)
        joined.insert(
            4,
            "locus",
            joined["Inversion"].map(LOCUS_LABELS).fillna(joined["Inversion"]),
        )

        numeric_columns = (
            "pooled_or",
            "pooled_q",
            "existing_or",
            "existing_p",
            "existing_ci_lo_or",
            "existing_ci_hi_or",
            "within_beta",
            "within_or",
            "within_p",
            "within_q_selected_set",
            "within_n_total",
            "within_n_cases",
            "within_n_controls",
            "within_ci_lo_or",
            "within_ci_hi_or",
        )
        _numeric(joined, numeric_columns)

        joined["within_p_valid"] = _boolean(joined["within_p_valid"])
        joined["within_ci_valid"] = _boolean(joined["within_ci_valid"])
        joined["existing_beta"] = np.log(joined["existing_or"])

        joined["evaluable"] = (
            np.isfinite(joined["existing_beta"])
            & np.isfinite(joined["within_beta"])
            & joined["within_p_valid"]
        )
        joined["direction_concordant"] = np.where(
            joined["evaluable"],
            np.sign(joined["existing_beta"]) == np.sign(joined["within_beta"]),
            pd.NA,
        )
        joined["beta_shift_within_minus_existing"] = np.where(
            joined["evaluable"],
            joined["within_beta"] - joined["existing_beta"],
            np.nan,
        )
        joined["absolute_beta_shift"] = joined[
            "beta_shift_within_minus_existing"
        ].abs()

        reason = pd.Series("", index=joined.index, dtype=object)
        reason.loc[~np.isfinite(joined["existing_beta"])] = (
            "existing_stratified_estimate_unavailable"
        )
        missing_new = joined["within_beta"].isna()
        reason.loc[missing_new & reason.eq("")] = (
            "not_evaluated_minimum_cases_or_controls"
        )
        if "within_skip_reason" in joined:
            explicit = joined["within_skip_reason"].fillna("").astype(str)
            reason.loc[explicit.ne("")] = explicit.loc[explicit.ne("")]
        reason.loc[~joined["within_p_valid"] & reason.eq("")] = (
            "within_ancestry_model_invalid"
        )
        joined["not_evaluable_reason"] = reason
        records.append(joined)

    comparison = pd.concat(records, ignore_index=True)
    return comparison.sort_values(
        ["population", "pooled_q", "Inversion", "Phenotype"],
        kind="stable",
    ).reset_index(drop=True)


def summarize(comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    evaluable = comparison[comparison["evaluable"]].copy()
    overall_r = float(stats.pearsonr(
        evaluable["existing_beta"], evaluable["within_beta"]
    ).statistic)
    same = int(evaluable["direction_concordant"].sum())

    rows = []
    for population, population_label in POPULATIONS:
        group = evaluable[evaluable["population"] == population]
        r = float(
            stats.pearsonr(group["existing_beta"], group["within_beta"]).statistic
        )
        concordant = int(group["direction_concordant"].sum())
        rows.append(
            {
                "population": population,
                "population_label": population_label,
                "n_evaluable": len(group),
                "n_direction_concordant": concordant,
                "direction_concordance_percent": float(100 * concordant / len(group)),
                "pearson_r_log_or": r,
                "median_absolute_beta_shift": float(
                    group["absolute_beta_shift"].median()
                ),
                "maximum_absolute_beta_shift": float(
                    group["absolute_beta_shift"].max()
                ),
            }
        )
    table = pd.DataFrame(rows)

    summary = {
        "original_associations": 39,
        "original_unique_phenotypes": 37,
        "population_groups": len(POPULATIONS),
        "possible_association_population_comparisons": 39 * len(POPULATIONS),
        "evaluable_association_population_comparisons": len(evaluable),
        "direction_concordant_comparisons": same,
        "direction_concordance_percent": float(100 * same / len(evaluable)),
        "pearson_r_log_or": overall_r,
        "median_absolute_beta_shift": float(
            evaluable["absolute_beta_shift"].median()
        ),
        "percentile_95_absolute_beta_shift": float(
            evaluable["absolute_beta_shift"].quantile(0.95)
        ),
        "nonconcordant_comparisons": comparison.loc[
            comparison["evaluable"]
            & ~comparison["direction_concordant"].astype("boolean").fillna(False),
            [
                "population",
                "locus",
                "Phenotype",
                "existing_or",
                "within_or",
                "within_p",
                "within_q_selected_set",
            ],
        ].to_dict(orient="records"),
    }
    return table, summary


def plot_correspondence(comparison: pd.DataFrame, summary: dict) -> None:
    evaluable = comparison[comparison["evaluable"]]
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.7), constrained_layout=True)

    for ax, (population, population_label) in zip(axes.flat, POPULATIONS):
        group = evaluable[evaluable["population"] == population]
        limit = 1.10 * max(
            0.15,
            group["existing_beta"].abs().max(),
            group["within_beta"].abs().max(),
        )
        r = stats.pearsonr(group["existing_beta"], group["within_beta"]).statistic
        color = POPULATION_COLORS[population]

        ax.plot([-limit, limit], [-limit, limit], color="#888888", lw=0.9, zorder=0)
        ax.scatter(
            group["existing_beta"],
            group["within_beta"],
            s=25,
            color=color,
            alpha=0.76,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{population_label} ({population})", loc="left", fontweight="bold")
        ax.text(
            0.03,
            0.97,
            f"r = {r:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    fig.supxlabel("Global-PC log(OR)")
    fig.supylabel("Within-ancestry-PC log(OR)")
    fig.suptitle(
        f"Pearson r = {summary['pearson_r_log_or']:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUT_CORRESPONDENCE_PDF)
    fig.savefig(OUT_CORRESPONDENCE_PNG, dpi=240)
    plt.close(fig)


def main() -> None:
    apply(base_size=9.0, dpi=300)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison()
    summary_table, summary = summarize(comparison)
    comparison.to_csv(OUT_COMPARISON, sep="\t", index=False)
    summary_table.to_csv(OUT_SUMMARY, sep="\t", index=False)
    with OUT_JSON.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    plot_correspondence(comparison, summary)

    print(
        f"Compared {summary['evaluable_association_population_comparisons']} "
        "evaluable association-population estimates: "
        f"r={summary['pearson_r_log_or']:.4f}, "
        f"{summary['direction_concordance_percent']:.1f}% same direction, "
        f"median |delta beta|={summary['median_absolute_beta_shift']:.4f}."
    )
    print(f"Wrote {OUT_COMPARISON}")
    print(f"Wrote {OUT_SUMMARY} and {OUT_JSON}")
    print(f"Wrote {OUT_CORRESPONDENCE_PDF} and {OUT_CORRESPONDENCE_PNG}")


if __name__ == "__main__":
    main()
