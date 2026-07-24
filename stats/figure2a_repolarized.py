#!/usr/bin/env python3
"""Regenerate manuscript Figure 2A using manually reviewed chimp polarity.

The population-genetic output is encoded relative to GRCh38:

* ``0_pi_filtered`` is diversity among reference/direct haplotypes.
* ``1_pi_filtered`` is diversity among alternate/inverted haplotypes.

The review ledger identifies which human arrangement is shared with chimp.
For ``direct`` calls, group 0 is ancestral and group 1 is derived. For
``inverted`` calls, the two groups are swapped. ``na`` calls are unresolved
and are excluded from this polarity-dependent analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "output.csv"
DEFAULT_PROPERTIES = REPO_ROOT / "data" / "inv_properties.tsv"
DEFAULT_POLARITY = REPO_ROOT / "data" / "chimp_alignment_responses.json"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "figure2a_repolarized"

COLORS = {"Ancestral": "#1f3b78", "Derived": "#8c2d7e"}
POSITIONS = {
    ("Single-event", "Ancestral"): 0.0,
    ("Single-event", "Derived"): 1.0,
    ("Recurrent", "Ancestral"): 3.0,
    ("Recurrent", "Derived"): 4.0,
}


def normalize_chromosome(value: object) -> str:
    chrom = str(value).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_polarity(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("dataset") != "chimp_vs_hg38_inversion_alignments":
        raise ValueError(f"Unexpected polarity dataset in {path}")
    responses = payload.get("responses")
    if not isinstance(responses, list):
        raise ValueError(f"{path} has no responses list")

    calls: dict[str, str] = {}
    for response in responses:
        inv_id = str(response.get("inv_id", "")).strip()
        call = str(response.get("classification", "")).strip().lower()
        if not inv_id or call not in {"direct", "inverted", "na"}:
            raise ValueError(f"Invalid response: {response!r}")
        if inv_id in calls:
            raise ValueError(f"Duplicate response for {inv_id}")
        calls[inv_id] = call
    return calls, payload


def build_audit(
    output_path: Path,
    properties_path: Path,
    polarity_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return one audited row per recurrence-classified manuscript locus."""
    output = pd.read_csv(output_path)
    properties = pd.read_csv(properties_path, sep="\t")
    calls, polarity_payload = load_polarity(polarity_path)

    required_output = {
        "chr",
        "region_start",
        "region_end",
        "0_pi_filtered",
        "1_pi_filtered",
    }
    required_properties = {
        "Chromosome",
        "Start",
        "End",
        "OrigID",
        "0_single_1_recur_consensus",
    }
    if missing := required_output - set(output.columns):
        raise KeyError(f"{output_path} missing columns: {sorted(missing)}")
    if missing := required_properties - set(properties.columns):
        raise KeyError(f"{properties_path} missing columns: {sorted(missing)}")

    output = output.copy()
    output["_chrom"] = output["chr"].map(normalize_chromosome)
    output["_start"] = pd.to_numeric(output["region_start"], errors="raise").astype(int)
    output["_end"] = pd.to_numeric(output["region_end"], errors="raise").astype(int)

    properties = properties.copy()
    properties["_recurrence"] = pd.to_numeric(
        properties["0_single_1_recur_consensus"], errors="coerce"
    )
    targets = properties.loc[properties["_recurrence"].isin([0, 1])].copy()
    if targets["OrigID"].duplicated().any():
        duplicates = targets.loc[targets["OrigID"].duplicated(), "OrigID"].tolist()
        raise ValueError(f"Duplicate manuscript inversion IDs: {duplicates}")

    missing_calls = sorted(set(targets["OrigID"].astype(str)) - set(calls))
    if missing_calls:
        raise ValueError(
            f"Polarity ledger lacks {len(missing_calls)} manuscript loci: {missing_calls}"
        )

    rows: list[dict[str, object]] = []
    for _, locus in targets.iterrows():
        inv_id = str(locus["OrigID"])
        chrom = normalize_chromosome(locus["Chromosome"])
        start = int(locus["Start"])
        end = int(locus["End"])
        recurrence = (
            "Recurrent" if int(locus["_recurrence"]) == 1 else "Single-event"
        )
        call = calls[inv_id]

        candidates = output.loc[
            (output["_chrom"] == chrom)
            & ((output["_start"] - start).abs() <= 1)
            & ((output["_end"] - end).abs() <= 1)
        ].copy()

        base = {
            "inv_id": inv_id,
            "chrom": f"chr{chrom}",
            "start": start,
            "end": end,
            "recurrence": recurrence,
            "chimp_call": call,
            "flip_ref_polarity": 1 if call == "inverted" else 0 if call == "direct" else pd.NA,
            "included_in_plot": False,
            "included_in_model": False,
            "plot_exclusion_reason": "",
            "model_exclusion_reason": "",
            "raw_pi_group0": np.nan,
            "raw_pi_group1": np.nan,
            "pi_ancestral": np.nan,
            "pi_derived": np.nan,
        }

        if candidates.empty:
            base["plot_exclusion_reason"] = "no_output_coordinate_match"
            base["model_exclusion_reason"] = "no_output_coordinate_match"
            rows.append(base)
            continue

        candidates["_distance"] = (
            (candidates["_start"] - start).abs()
            + (candidates["_end"] - end).abs()
        )
        nearest = candidates.loc[candidates["_distance"] == candidates["_distance"].min()]
        if len(nearest) != 1:
            raise ValueError(
                f"Ambiguous ±1 bp output match for {inv_id}: "
                f"{nearest[['chr', 'region_start', 'region_end']].to_dict('records')}"
            )
        match = nearest.iloc[0]
        pi0 = finite_number(match["0_pi_filtered"])
        pi1 = finite_number(match["1_pi_filtered"])
        base["output_region_start"] = int(match["_start"])
        base["output_region_end"] = int(match["_end"])
        base["raw_pi_group0"] = np.nan if pi0 is None else pi0
        base["raw_pi_group1"] = np.nan if pi1 is None else pi1

        if call == "na":
            base["plot_exclusion_reason"] = "unresolved_chimp_orientation"
            base["model_exclusion_reason"] = "unresolved_chimp_orientation"
        else:
            if call == "direct":
                ancestral, derived = pi0, pi1
            else:
                ancestral, derived = pi1, pi0
            base["pi_ancestral"] = np.nan if ancestral is None else ancestral
            base["pi_derived"] = np.nan if derived is None else derived
            if ancestral is None and derived is None:
                base["plot_exclusion_reason"] = "no_finite_pi"
                base["model_exclusion_reason"] = "no_finite_pi"
            else:
                base["included_in_plot"] = True
                if ancestral is not None and derived is not None:
                    base["included_in_model"] = True
                else:
                    base["model_exclusion_reason"] = "one_orientation_missing_pi"
        rows.append(base)

    audit = pd.DataFrame(rows).sort_values(
        ["chrom", "start", "end"], kind="stable"
    )
    metadata = {
        "polarity_dataset": polarity_payload["dataset"],
        "polarity_updated_at": polarity_payload.get("updated_at"),
        "n_reviewed_total": len(calls),
        "n_manuscript_loci": len(targets),
    }
    return audit, metadata


def choose_floor(values: np.ndarray) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    if not positive.size:
        raise ValueError("No positive nucleotide-diversity values")
    return max(float(np.quantile(positive, 0.01)) * 0.5, 1e-8)


def linear_contrast(
    result: object, weights: dict[str, float]
) -> tuple[float, float, float]:
    names = list(result.params.index)
    vector = np.zeros((1, len(names)))
    for name, weight in weights.items():
        vector[0, names.index(name)] = weight
    test = result.t_test(vector)
    return (
        float(np.squeeze(test.effect)),
        float(np.squeeze(test.sd)),
        float(np.squeeze(test.pvalue)),
    )


def fit_model(model_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    values = np.r_[
        model_data["pi_ancestral"].to_numpy(float),
        model_data["pi_derived"].to_numpy(float),
    ]
    epsilon = choose_floor(values)
    model_data = model_data.copy()
    model_data["log_ratio"] = np.log(model_data["pi_derived"] + epsilon) - np.log(
        model_data["pi_ancestral"] + epsilon
    )
    model_data["is_recurrent"] = (
        model_data["recurrence"] == "Recurrent"
    ).astype(int)
    design = sm.add_constant(model_data[["is_recurrent"]])
    fitted = sm.OLS(model_data["log_ratio"], design).fit(cov_type="HC3")

    contrasts = [
        (
            "derived_vs_ancestral_single_event",
            {"const": 1.0},
        ),
        (
            "derived_vs_ancestral_recurrent",
            {"const": 1.0, "is_recurrent": 1.0},
        ),
        (
            "orientation_by_recurrence_interaction",
            {"is_recurrent": 1.0},
        ),
    ]
    rows = []
    for label, weights in contrasts:
        estimate, standard_error, p_value = linear_contrast(fitted, weights)
        rows.append(
            {
                "contrast": label,
                "log_ratio": estimate,
                "hc3_standard_error": standard_error,
                "ratio": math.exp(estimate),
                "ratio_ci_low": math.exp(estimate - 1.96 * standard_error),
                "ratio_ci_high": math.exp(estimate + 1.96 * standard_error),
                "p_value_two_sided_wald": p_value,
            }
        )

    summary: dict[str, object] = {
        "epsilon": epsilon,
        "model": "log((pi_derived + epsilon)/(pi_ancestral + epsilon)) ~ recurrent",
        "covariance": "HC3",
        "n_included": int(len(model_data)),
    }
    for recurrence in ("Single-event", "Recurrent"):
        subset = model_data.loc[model_data["recurrence"] == recurrence]
        key = recurrence.lower().replace("-", "_")
        summary[f"n_{key}"] = int(len(subset))
        summary[f"mean_pi_ancestral_{key}"] = float(subset["pi_ancestral"].mean())
        summary[f"mean_pi_derived_{key}"] = float(subset["pi_derived"].mean())
        summary[f"median_pi_ancestral_{key}"] = float(subset["pi_ancestral"].median())
        summary[f"median_pi_derived_{key}"] = float(subset["pi_derived"].median())
    return pd.DataFrame(rows), summary


def p_label(value: float) -> str:
    return f"P = {value:.2e}" if value < 0.001 else f"P = {value:.3f}"


def draw_figure(
    plot_data: pd.DataFrame,
    effects: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rng = np.random.default_rng(2025)
    groups = [
        ("Single-event", "Ancestral"),
        ("Single-event", "Derived"),
        ("Recurrent", "Ancestral"),
        ("Recurrent", "Derived"),
    ]
    values = [
        plot_data.loc[plot_data["recurrence"] == recurrence, f"pi_{orientation.lower()}"]
        .dropna()
        .to_numpy(float)
        for recurrence, orientation in groups
    ]

    fig, ax = plt.subplots(figsize=(8.2, 6.5))
    violin = ax.violinplot(
        values,
        positions=[POSITIONS[group] for group in groups],
        widths=0.9,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, (_, orientation) in zip(violin["bodies"], groups):
        body.set_facecolor(COLORS[orientation])
        body.set_edgecolor("none")
        body.set_alpha(0.55)

    for values_for_box, group in zip(values, groups):
        if not len(values_for_box):
            continue
        ax.boxplot(
            [values_for_box],
            positions=[POSITIONS[group]],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": "#111111"},
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "#111111"},
            capprops={"color": "#111111"},
        )

    log2_ratio = np.log2(
        (plot_data["pi_ancestral"].to_numpy(float) + 1e-12)
        / (plot_data["pi_derived"].to_numpy(float) + 1e-12)
    )
    finite_log2_ratio = log2_ratio[np.isfinite(log2_ratio)]
    max_abs = float(np.percentile(np.abs(finite_log2_ratio), 98))
    max_abs = max(max_abs, 1e-12)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    cmap = plt.get_cmap("coolwarm")
    for (_, row), fold_change in zip(plot_data.iterrows(), log2_ratio):
        recurrence = row["recurrence"]
        jitter = float(rng.uniform(0.06, 0.20))
        x_ancestral = POSITIONS[(recurrence, "Ancestral")] + jitter
        x_derived = POSITIONS[(recurrence, "Derived")] - jitter
        ancestral = finite_number(row["pi_ancestral"])
        derived = finite_number(row["pi_derived"])
        if ancestral is not None and derived is not None:
            ax.plot(
                [x_ancestral, x_derived],
                [ancestral, derived],
                color=cmap(norm(fold_change)),
                linewidth=1.25,
                alpha=0.85,
                zorder=2,
            )
        for x, value, orientation in (
            (x_ancestral, ancestral, "Ancestral"),
            (x_derived, derived, "Derived"),
        ):
            if value is not None:
                ax.scatter(
                    [x],
                    [value],
                    c=[COLORS[orientation]],
                    s=25,
                    edgecolors="black",
                    linewidths=0.4,
                    alpha=0.72,
                    zorder=3,
                )

    p_single = float(
        effects.loc[
            effects["contrast"] == "derived_vs_ancestral_single_event",
            "p_value_two_sided_wald",
        ].iloc[0]
    )
    p_recurrent = float(
        effects.loc[
            effects["contrast"] == "derived_vs_ancestral_recurrent",
            "p_value_two_sided_wald",
        ].iloc[0]
    )
    ymax = max(float(np.nanmax(np.concatenate(values))), 1e-8)
    ax.set_ylim(0, ymax * 1.23)
    for left, right, p_value in ((0, 1, p_single), (3, 4, p_recurrent)):
        y = ymax * 1.08
        h = ymax * 0.025
        ax.plot([left, left, right, right], [y, y + h, y + h, y], color="#222222")
        ax.text((left + right) / 2, y + h * 1.25, p_label(p_value), ha="center")

    ax.axvline(2, color="#dddddd", linewidth=1)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["Ancestral", "Derived", "Ancestral", "Derived"])
    ax.text(0.5, -0.11, "Single-event", transform=ax.get_xaxis_transform(), ha="center", fontweight="bold")
    ax.text(3.5, -0.11, "Recurrent", transform=ax.get_xaxis_transform(), ha="center", fontweight="bold")
    ax.set_ylabel("Nucleotide diversity (π)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.03, fraction=0.05)
    colorbar.set_label(r"$\log_{2}(\pi_{\mathrm{ancestral}}/\pi_{\mathrm{derived}})$")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inv-properties", type=Path, default=DEFAULT_PROPERTIES)
    parser.add_argument("--polarity-json", type=Path, default=DEFAULT_POLARITY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    audit, source_metadata = build_audit(
        args.output_csv, args.inv_properties, args.polarity_json
    )
    plot_data = audit.loc[audit["included_in_plot"]].copy()
    model_data = audit.loc[audit["included_in_model"]].copy()
    if plot_data.empty or model_data.empty:
        raise RuntimeError("No loci survived repolarization and quality filters")
    effects, model_summary = fit_model(model_data)

    audit.to_csv(args.outdir / "figure2a_locus_audit.tsv", sep="\t", index=False)
    effects.to_csv(args.outdir / "figure2a_model_effects.tsv", sep="\t", index=False)
    summary = {
        **source_metadata,
        **model_summary,
        "n_plot_loci": int(audit["included_in_plot"].sum()),
        "n_model_loci": int(audit["included_in_model"].sum()),
        "n_excluded_from_plot": int((~audit["included_in_plot"]).sum()),
        "n_excluded_from_model": int((~audit["included_in_model"]).sum()),
        "plot_exclusion_counts": {
            str(key): int(value)
            for key, value in audit.loc[
                ~audit["included_in_plot"], "plot_exclusion_reason"
            ]
            .value_counts()
            .items()
        },
        "model_exclusion_counts": {
            str(key): int(value)
            for key, value in audit.loc[
                ~audit["included_in_model"], "model_exclusion_reason"
            ]
            .value_counts()
            .items()
        },
        "chimp_calls_in_manuscript_loci": {
            str(key): int(value)
            for key, value in audit["chimp_call"].value_counts().items()
        },
        "input_files": {
            "output_csv": {
                "path": str(args.output_csv.resolve()),
                "sha256": sha256(args.output_csv),
            },
            "inv_properties": {
                "path": str(args.inv_properties.resolve()),
                "sha256": sha256(args.inv_properties),
            },
            "polarity_json": {
                "path": str(args.polarity_json.resolve()),
                "sha256": sha256(args.polarity_json),
            },
        },
    }
    with (args.outdir / "figure2a_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    draw_figure(
        plot_data,
        effects,
        args.outdir / "figure2a_repolarized.pdf",
        args.outdir / "figure2a_repolarized.png",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
