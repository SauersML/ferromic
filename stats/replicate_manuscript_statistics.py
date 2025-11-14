"""Replicate key manuscript statistics using the local summary tables.

This script aggregates statistics from the curated TSV/CSV files under the
``data`` directory.  The goal is to provide a single entry point that prints a
human-readable report and saves the same text to
``stats/replicate_manuscript_statistics.txt``.

The implementation focuses on metrics that can be recomputed from the staged
summary tables (e.g., inversion annotations, π estimates, FST estimates,
imputation summaries, and PheWAS association results).  Each section is
implemented as a pure function so that it can be tested in isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats import inv_dir_recur_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPORT_PATH = Path(__file__).with_suffix(".txt")


def _fmt_number(value: float, digits: int = 3) -> str:
    """Return a compact string for either integers or floating values."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    abs_v = abs(float(value))
    if 0 < abs_v < 10 ** -(digits - 1) or abs_v >= 10 ** (digits + 1):
        return f"{float(value):.{digits}e}"
    return f"{float(value):.{digits}f}"


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else float("nan")


def _load_inv_properties() -> pd.DataFrame:
    path = DATA_DIR / "inv_properties.tsv"
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={
        "Chromosome": "chromosome",
        "Start": "start",
        "End": "end",
        "0_single_1_recur_consensus": "recurrence_flag",
        "OrigID": "inversion_id",
    })
    df["chromosome"] = df["chromosome"].astype(str).str.replace("^chr", "", regex=True)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["recurrence_flag"] = pd.to_numeric(df["recurrence_flag"], errors="coerce")
    df = df[df["recurrence_flag"].isin([0, 1])].copy()
    df["recurrence_label"] = df["recurrence_flag"].map({0: "Single-event", 1: "Recurrent"})
    return df


def summarize_recurrence() -> List[str]:
    df = _load_inv_properties()
    total = len(df)
    recurrent = int((df["recurrence_flag"] == 1).sum())
    single = int((df["recurrence_flag"] == 0).sum())
    recurrence_pct = _safe_div(recurrent, total) * 100
    lines = [
        "Chromosome inversion recurrence summary:",
        f"  High-quality inversions with consensus labels: {total} (single-event = {single}, recurrent = {recurrent}).",
        f"  Fraction recurrent = {_fmt_number(recurrence_pct, digits=2)}%.",
    ]
    return lines


def _load_pi_summary() -> pd.DataFrame:
    pi_path = DATA_DIR / "output.csv"
    pi_df = pd.read_csv(pi_path, low_memory=False)
    pi_df["chr"] = pi_df["chr"].astype(str).str.replace("^chr", "", regex=True)
    inv_df = _load_inv_properties()
    merged = pi_df.merge(
        inv_df[["chromosome", "start", "end", "recurrence_flag", "recurrence_label", "inversion_id"]],
        left_on=["chr", "region_start", "region_end"],
        right_on=["chromosome", "start", "end"],
        how="inner",
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["0_pi_filtered", "1_pi_filtered"])
    return merged


def summarize_diversity() -> List[str]:
    df = _load_pi_summary()
    lines: List[str] = ["Nucleotide diversity (π) comparisons:"]
    total_sites = len(df)
    lines.append(f"  Loci with finite π for both orientations: {total_sites}.")

    inv_mean = df["1_pi_filtered"].mean()
    dir_mean = df["0_pi_filtered"].mean()
    ttest = stats.ttest_rel(df["1_pi_filtered"], df["0_pi_filtered"])
    lines.append(
        "  Across all loci: mean π(inverted) = "
        f"{_fmt_number(inv_mean, 6)}, mean π(direct) = {_fmt_number(dir_mean, 6)}."
    )
    lines.append(
        "    Paired t-test for π(inverted) vs π(direct): "
        f"t = {_fmt_number(ttest.statistic, 3)}, p = {_fmt_number(ttest.pvalue, 3)}."
    )

    for flag, label in [(0, "Single-event"), (1, "Recurrent")]:
        sub = df[df["recurrence_flag"] == flag]
        if sub.empty:
            continue
        lines.append(
            f"  {label} inversions: median π(inverted) = {_fmt_number(sub['1_pi_filtered'].median(), 6)}, "
            f"median π(direct) = {_fmt_number(sub['0_pi_filtered'].median(), 6)}."
        )

    inv_only = df[["recurrence_flag", "1_pi_filtered"]].copy()
    inv_only_grouped = inv_only.groupby("recurrence_flag")["1_pi_filtered"].median()
    if not inv_only_grouped.empty:
        lines.append(
            "  Within inverted haplotypes:"
            f" recurrent median π = {_fmt_number(inv_only_grouped.get(1, float('nan')), 6)};"
            f" single-event median π = {_fmt_number(inv_only_grouped.get(0, float('nan')), 6)}."
        )
    return lines


def summarize_linear_model() -> List[str]:
    df = _load_pi_summary()
    inv_ids = df.get("inversion_id")
    if inv_ids is None:
        inv_ids = pd.Series([""] * len(df))
    else:
        inv_ids = inv_ids.fillna("")
    matched = pd.DataFrame(
        {
            "pi_direct": df["0_pi_filtered"].astype(float),
            "pi_inverted": df["1_pi_filtered"].astype(float),
            "Recurrence": df["recurrence_flag"].map({0: "Single-event", 1: "Recurrent"}),
            "region_id": inv_ids,
            "chromosome": df["chromosome"],
            "start": df["start"],
            "end": df["end"],
        }
    )
    matched["region_id"] = matched.apply(
        lambda row: row["region_id"]
        if isinstance(row["region_id"], str) and row["region_id"]
        else f"chr{row['chromosome']}:{int(row['start'])}-{int(row['end'])}",
        axis=1,
    )
    matched = matched.dropna(subset=["pi_direct", "pi_inverted", "Recurrence"])
    all_pi = np.r_[matched["pi_direct"].to_numpy(float), matched["pi_inverted"].to_numpy(float)]
    eps = inv_dir_recur_model.choose_floor_from_quantile(
        all_pi, q=inv_dir_recur_model.FLOOR_QUANTILE, min_floor=inv_dir_recur_model.MIN_FLOOR
    )
    _, effects, _ = inv_dir_recur_model.run_model_A(matched, eps=eps, nonzero_only=False)
    lines = ["Orientation × recurrence linear model on log π ratios (Model A):"]
    lines.append(f"  Detection floor applied before logs: ε = {_fmt_number(eps, 6)}.")
    for effect in effects.itertuples():
        lines.append(
            f"  {effect.effect}: fold-change = {_fmt_number(effect.ratio, 3)} "
            f"(95% CI {_fmt_number(effect.ci_low, 3)}–{_fmt_number(effect.ci_high, 3)}), "
            f"p = {_fmt_number(effect.p, 3)}."
        )
    return lines


def summarize_fst() -> List[str]:
    df = _load_pi_summary()
    if "hudson_fst_hap_group_0v1" not in df.columns:
        return ["Hudson's FST column not present; skipping differentiation summary."]
    fst = df.dropna(subset=["hudson_fst_hap_group_0v1"])
    lines = ["Hudson's FST between orientations:"]
    for flag, label in [(0, "Single-event"), (1, "Recurrent")]:
        sub = fst[fst["recurrence_flag"] == flag]
        if sub.empty:
            continue
        lines.append(
            f"  {label}: median FST = {_fmt_number(sub['hudson_fst_hap_group_0v1'].median(), 3)} (n = {len(sub)})."
        )
    if not fst.empty:
        if fst["recurrence_flag"].nunique() > 1:
            utest = stats.mannwhitneyu(
                fst.loc[fst["recurrence_flag"] == 0, "hudson_fst_hap_group_0v1"],
                fst.loc[fst["recurrence_flag"] == 1, "hudson_fst_hap_group_0v1"],
                alternative="two-sided",
            )
            lines.append(
                "  Mann–Whitney U test (single-event vs recurrent): "
                f"U = {_fmt_number(utest.statistic, 3)}, p = {_fmt_number(utest.pvalue, 3)}."
            )
        counts = fst["hudson_fst_hap_group_0v1"].to_numpy()
        lines.append(
            "  Highly differentiated loci: "
            f"{int((counts > 0.2).sum())} with FST > 0.2; {int((counts > 0.5).sum())} with FST > 0.5."
        )
    return lines


def summarize_imputation() -> List[str]:
    path = DATA_DIR / "imputation_results.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={
        "unbiased_pearson_r2": "r2",
        "p_fdr_bh": "bh_p",
    })
    usable = df[(df["r2"] > 0.3) & (df["bh_p"] < 0.05)]
    lines = ["Imputation performance summary:"]
    lines.append(f"  Models evaluated: {len(df)}; models with r² > 0.3 and BH p < 0.05: {len(usable)}.")
    if "Use" in df.columns:
        lines.append(f"  Models flagged for downstream PheWAS (Use == True): {int(df['Use'].eq(True).sum())}.")
    return lines


def summarize_phewas_scale() -> List[str]:
    results_path = DATA_DIR / "phewas_results.tsv"
    if not results_path.exists():
        return [f"PheWAS results table not found at {results_path}."]

    results = pd.read_csv(results_path, sep="\t", low_memory=False)
    required_cols = {"Phenotype", "N_Cases", "N_Controls", "Inversion"}
    if not required_cols.issubset(results.columns):
        missing = ", ".join(sorted(required_cols - set(results.columns)))
        return [f"PheWAS results missing required columns: {missing}."]

    lines = ["PheWAS scale summary:"]
    lines.append(f"  Unique phenotypes tested: {results['Phenotype'].nunique()}.")
    lines.append(
        "  Case counts span "
        f"{_fmt_number(results['N_Cases'].min(), 0)} to {_fmt_number(results['N_Cases'].max(), 0)}; "
        f"controls span {_fmt_number(results['N_Controls'].min(), 0)}–{_fmt_number(results['N_Controls'].max(), 0)}."
    )
    inv_counts = results.groupby("Inversion")["Phenotype"].nunique().sort_values(ascending=False)
    lines.append(
        "  Phenotype coverage per inversion (top 5): "
        + ", ".join(f"{inv}: {count}" for inv, count in inv_counts.head(5).items())
        + ("; ..." if len(inv_counts) > 5 else "")
    )

    sig_col = results.get("Sig_Global")
    if sig_col is not None:
        sig_mask = sig_col.astype(str).str.upper() == "TRUE"
        sig_inversions = results.loc[sig_mask, "Inversion"].nunique()
        lines.append(
            f"  Inversions with ≥1 BH-significant phenotype: {sig_inversions} of {results['Inversion'].nunique()}."
        )

    cat_path = DATA_DIR / "phewas v2 - categories.tsv"
    if cat_path.exists():
        categories = pd.read_csv(cat_path, sep="\t", low_memory=False)
        if {"Inversion", "Category", "P_GBJ", "P_GLS"}.issubset(categories.columns):
            sig_categories = categories[(categories["P_GBJ"] < 0.05) | (categories["P_GLS"] < 0.05)]
            lines.append(
                f"  Category-level tests with q<0.05: {len(sig_categories)} across "
                f"{sig_categories['Inversion'].nunique()} inversions."
            )
        else:
            lines.append("  Category summary present but missing required columns; skipped detailed counts.")
    else:
        lines.append("  Category-level summary table not found; skipping.")

    return lines


def build_report() -> List[str]:
    sections: List[Tuple[str, Iterable[str]]] = [
        ("Recurrence", summarize_recurrence()),
        ("Diversity", summarize_diversity()),
        ("Linear model", summarize_linear_model()),
        ("Differentiation", summarize_fst()),
        ("Imputation", summarize_imputation()),
        ("PheWAS scale", summarize_phewas_scale()),
    ]
    report_lines: List[str] = []
    for title, content in sections:
        report_lines.append(title.upper())
        if isinstance(content, Iterable):
            for line in content:
                report_lines.append(line)
        else:
            report_lines.append(str(content))
        report_lines.append("")
    return report_lines


def main() -> None:
    report_lines = build_report()
    text = "\n".join(report_lines).strip() + "\n"
    print(text)
    REPORT_PATH.write_text(text)
    print(f"\nSaved report to {REPORT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
