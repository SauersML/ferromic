"""Replicate manuscript metrics and tests.
"""
from __future__ import annotations

import math
import sys
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats import CDS_identical_model, inv_dir_recur_model, recur_breakpoint_tests  # noqa: E402
from stats._inv_common import map_inversion_series, map_inversion_value

DATA_DIR = REPO_ROOT / "data"
ANALYSIS_DOWNLOAD_DIR = REPO_ROOT / "analysis_downloads"
REPORT_PATH = Path(__file__).with_suffix(".txt")


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------


def _fmt(value: float | int | None, digits: int = 3) -> str:
    """Format floating-point numbers with sensible scientific notation.

    Integers are rendered without decimal places. Very small or very large
    values fall back to scientific notation so the printed report stays
    readable.
    """

    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(val) or math.isinf(val):
        return "NA"
    if 0 < abs(val) < 10 ** -(digits - 1) or abs(val) >= 10 ** (digits + 1):
        return f"{val:.{digits}e}"
    return f"{val:.{digits}f}"


def _safe_mean(series: pd.Series) -> float | None:
    if series is None:
        return None
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return None
    return float(vals.mean())


def _safe_median(series: pd.Series) -> float | None:
    if series is None:
        return None
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return None
    return float(vals.median())


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_repo_artifact(basename: str) -> Path | None:
    """Search common locations for derived analysis artefacts."""

    search_dirs = [
        REPO_ROOT,
        DATA_DIR,
        REPO_ROOT / "cds",
        REPO_ROOT / "stats",
        ANALYSIS_DOWNLOAD_DIR,
        ANALYSIS_DOWNLOAD_DIR / "public_internet",
    ]
    for directory in search_dirs:
        if directory is None or not directory.exists():
            continue
        candidate = directory / basename
        if candidate.exists():
            return candidate
    return None


@contextmanager
def _temporary_workdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------


def _load_inv_properties() -> pd.DataFrame:
    path = DATA_DIR / "inv_properties.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing inversion annotation table: {path}")

    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(
        columns={
            "Chromosome": "chromosome",
            "Start": "start",
            "End": "end",
            "OrigID": "inversion_id",
            "0_single_1_recur_consensus": "recurrence_flag",
        }
    )
    df["chromosome"] = df["chromosome"].astype(str).str.replace("^chr", "", regex=True)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["recurrence_flag"] = pd.to_numeric(df["recurrence_flag"], errors="coerce")
    df = df[df["recurrence_flag"].isin([0, 1])].copy()
    df["recurrence_label"] = df["recurrence_flag"].map({0: "Single-event", 1: "Recurrent"})
    return df


def _load_pi_summary() -> pd.DataFrame:
    pi_path = DATA_DIR / "output.csv"
    if not pi_path.exists():
        raise FileNotFoundError(f"Missing per-inversion diversity summary: {pi_path}")

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


def _load_fst_table() -> pd.DataFrame | None:
    fst_candidates = [
        DATA_DIR / "hudson_fst_results.tsv.gz",
        DATA_DIR / "hudson_fst_results.tsv",
    ]
    fst_path = next((path for path in fst_candidates if path.exists()), None)
    if fst_path is None:
        return None
    fst = pd.read_csv(fst_path, sep="\t", low_memory=False, compression="infer")
    required = {"chr", "region_start_0based", "region_end_0based", "FST"}
    if not required.issubset(fst.columns):
        return None
    fst = fst.rename(
        columns={
            "chr": "chromosome",
            "region_start_0based": "start",
            "region_end_0based": "end",
            "FST": "fst",
        }
    )
    fst["chromosome"] = fst["chromosome"].astype(str)
    fst["start"] = pd.to_numeric(fst["start"], errors="coerce")
    fst["end"] = pd.to_numeric(fst["end"], errors="coerce")
    fst["fst"] = pd.to_numeric(fst["fst"], errors="coerce")
    fst = fst.replace([np.inf, -np.inf], np.nan).dropna(subset=["start", "end", "fst"])
    inv_df = _load_inv_properties()
    out = fst.merge(
        inv_df[["chromosome", "start", "end", "recurrence_flag", "recurrence_label", "inversion_id"]],
        on=["chromosome", "start", "end"],
        how="inner",
    )
    return out


# ---------------------------------------------------------------------------
# Section 1. Recurrence and sample size summaries
# ---------------------------------------------------------------------------


def summarize_recurrence() -> List[str]:
    inv_df = _load_inv_properties()
    total = len(inv_df)
    recurrent = int((inv_df["recurrence_flag"] == 1).sum())
    single = int((inv_df["recurrence_flag"] == 0).sum())
    frac = (recurrent / total * 100) if total else float("nan")
    lines = ["Chromosome inversion recurrence summary:"]
    lines.append(
        "  High-quality inversions with consensus labels: "
        f"{_fmt(total, 0)} (single-event = {_fmt(single, 0)}, recurrent = {_fmt(recurrent, 0)})."
    )
    lines.append(f"  Fraction recurrent = {_fmt(frac, 2)}%." if total else "  Fraction recurrent unavailable.")
    return lines


def summarize_sample_sizes() -> List[str]:
    lines: List[str] = ["Sample sizes for diversity analyses:"]

    callset_path = DATA_DIR / "callset.tsv"
    if callset_path.exists():
        header = pd.read_csv(callset_path, sep="\t", nrows=0)
        meta_cols = {
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
        sample_cols = [c for c in header.columns if c not in meta_cols]
        n_samples = len(sample_cols)
        lines.append(
            "  Inversion callset columns indicate "
            f"{_fmt(n_samples, 0)} phased individuals (sample columns)."
        )
        lines.append(
            "  Reporting haplotypes as twice the sample count yields "
            f"{_fmt(2 * n_samples, 0)} potential phased haplotypes."
        )
    else:
        lines.append(f"  Callset not found at {callset_path}; sample counts unavailable.")

    pi_df = _load_pi_summary()
    usable = pi_df[(pi_df["0_num_hap_filter"] >= 2) & (pi_df["1_num_hap_filter"] >= 2)]
    lines.append(
        "  Loci with ≥2 haplotypes per orientation for π: "
        f"{_fmt(len(usable), 0)} (from output.csv)."
    )

    return lines


# ---------------------------------------------------------------------------
# Section 2. Diversity and linear model
# ---------------------------------------------------------------------------


def summarize_diversity() -> List[str]:
    df = _load_pi_summary()
    lines: List[str] = ["Nucleotide diversity (π) by orientation and recurrence:"]
    lines.append(f"  Total loci with finite π estimates: {_fmt(len(df), 0)}.")

    inv_mean = df["1_pi_filtered"].mean()
    dir_mean = df["0_pi_filtered"].mean()
    ttest = stats.ttest_rel(df["1_pi_filtered"], df["0_pi_filtered"])
    lines.append(
        "  Across all loci: mean π(inverted) = "
        f"{_fmt(inv_mean, 6)}, mean π(direct) = {_fmt(dir_mean, 6)}."
    )
    lines.append(
        "    Two-sided paired t-test comparing orientations: "
        f"t = {_fmt(ttest.statistic, 3)}, p = {_fmt(ttest.pvalue, 3)}."
    )

    for flag, label in [(0, "Single-event"), (1, "Recurrent")]:
        sub = df[df["recurrence_flag"] == flag]
        if sub.empty:
            continue
        lines.append(
            f"  {label} inversions: median π(inverted) = {_fmt(sub['1_pi_filtered'].median(), 6)}, "
            f"median π(direct) = {_fmt(sub['0_pi_filtered'].median(), 6)}."
        )

    inv_only = df[["recurrence_flag", "1_pi_filtered"]]
    grouped = inv_only.groupby("recurrence_flag")["1_pi_filtered"].median()
    lines.append(
        "  Within inverted haplotypes: recurrent median π = "
        f"{_fmt(grouped.get(1, np.nan), 6)}; single-event median π = {_fmt(grouped.get(0, np.nan), 6)}."
    )
    return lines


def summarize_linear_model() -> List[str]:
    df = _load_pi_summary()
    inv_ids = df.get("inversion_id")
    if inv_ids is None:
        inv_ids = pd.Series(["" for _ in range(len(df))])
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
        all_pi,
        q=inv_dir_recur_model.FLOOR_QUANTILE,
        min_floor=inv_dir_recur_model.MIN_FLOOR,
    )

    _, effects, _ = inv_dir_recur_model.run_model_A(matched, eps=eps, nonzero_only=False)
    lines = ["Orientation × recurrence linear model on log π ratios (Model A):"]
    lines.append(f"  Detection floor applied before logs: ε = {_fmt(eps, 6)}.")
    for effect in effects.itertuples():
        lines.append(
            f"  {effect.effect}: fold-change = {_fmt(effect.ratio, 3)} "
            f"(95% CI {_fmt(effect.ci_low, 3)}–{_fmt(effect.ci_high, 3)}), p = {_fmt(effect.p, 3)}."
        )
    return lines


def summarize_cds_conservation_glm() -> List[str]:
    lines: List[str] = [
        "CDS conservation GLM (proportion of identical CDS pairs):"
    ]

    pairwise_df: pd.DataFrame | None = None
    source_label: str | None = None
    errors: List[str] = []

    cds_input = _resolve_repo_artifact("cds_identical_proportions.tsv")
    if cds_input is not None:
        try:
            with _temporary_workdir(cds_input.parent):
                cds_df = CDS_identical_model.load_data()
                res = CDS_identical_model.fit_glm_binom(cds_df, include_covariates=True)
                _, pairwise_df = CDS_identical_model.emms_and_pairs(
                    res, cds_df, include_covariates=True
                )
            source_label = "recomputed from cds_identical_proportions.tsv"
        except SystemExit as exc:  # stats/CDS_identical_model exits on missing inputs
            errors.append(f"CDS GLM exited early: {exc}")
        except FileNotFoundError as exc:
            errors.append(f"Missing CDS supporting file: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Failed to recompute GLM: {exc}")

    if pairwise_df is None:
        pairwise_path = _resolve_repo_artifact("cds_pairwise_adjusted.tsv")
        if pairwise_path is not None:
            try:
                pairwise_df = pd.read_csv(pairwise_path, sep="\t", low_memory=False)
                source_label = f"loaded from {_relative_to_repo(pairwise_path)}"
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"Unable to read {pairwise_path}: {exc}")

    if pairwise_df is None:
        lines.append(
            "  CDS GLM inputs unavailable (expected cds_identical_proportions.tsv or cds_pairwise_adjusted.tsv)."
        )
        lines.extend(f"  {msg}" for msg in errors)
        return lines

    if source_label:
        lines.append(f"  Source: {source_label}.")

    required = {
        "A",
        "B",
        "diff_logit",
        "diff_prob",
        "p_value",
        "q_value_fdr",
    }
    if not required.issubset(pairwise_df.columns):
        missing = ", ".join(sorted(required - set(pairwise_df.columns)))
        lines.append(f"  Pairwise contrast table missing required columns: {missing}.")
        return lines

    target_label = "Single/Inverted"
    comparisons = [
        (target_label, "Single/Direct"),
        (target_label, "Recurrent/Inverted"),
        (target_label, "Recurrent/Direct"),
    ]

    def _extract_contrast(a: str, b: str) -> pd.Series | None:
        mask = (
            ((pairwise_df["A"] == a) & (pairwise_df["B"] == b))
            | ((pairwise_df["A"] == b) & (pairwise_df["B"] == a))
        )
        subset = pairwise_df.loc[mask]
        if subset.empty:
            return None
        return subset.iloc[0]

    found_any = False
    for target, other in comparisons:
        row = _extract_contrast(target, other)
        if row is None:
            lines.append(f"  Contrast {target} vs {other} not present in CDS pairwise table.")
            continue
        found_any = True
        diff_prob = float(row["diff_prob"])
        diff_logit = float(row["diff_logit"])
        if row["A"] != target:
            diff_prob *= -1
            diff_logit *= -1
        lines.append(
            "  "
            + f"{target} vs {other}: Δlogit = {_fmt(diff_logit, 3)}, Δp = {_fmt(diff_prob, 3)}, "
            + f"p = {_fmt(row['p_value'], 3)}, BH q = {_fmt(row['q_value_fdr'], 3)}."
        )

    if not found_any:
        lines.append("  No pairwise contrasts reported for Single/Inverted haplotypes.")

    return lines


# ---------------------------------------------------------------------------
# Section 3. Differentiation and breakpoint enrichment
# ---------------------------------------------------------------------------


def summarize_fst() -> List[str]:
    df = _load_pi_summary()
    if "hudson_fst_hap_group_0v1" not in df.columns:
        return ["Hudson's FST column missing from output.csv; skipping differentiation summary."]

    fst = df.dropna(subset=["hudson_fst_hap_group_0v1"])
    if fst.empty:
        return ["No finite Hudson's FST values available."]

    fst = fst.rename(columns={"hudson_fst_hap_group_0v1": "fst"})
    lines = ["Differentiation between orientations (Hudson's FST):"]
    for flag, label in [(0, "Single-event"), (1, "Recurrent")]:
        sub = fst[fst["recurrence_flag"] == flag]
        if sub.empty:
            continue
        lines.append(
            f"  {label}: median FST = {_fmt(sub['fst'].median(), 3)} (n = {_fmt(len(sub), 0)})."
        )

    if fst["recurrence_flag"].nunique() > 1:
        utest = stats.mannwhitneyu(
            fst.loc[fst["recurrence_flag"] == 0, "fst"],
            fst.loc[fst["recurrence_flag"] == 1, "fst"],
            alternative="two-sided",
        )
        lines.append(
            "  Mann–Whitney U test (single-event vs recurrent): "
            f"U = {_fmt(utest.statistic, 3)}, p = {_fmt(utest.pvalue, 3)}."
        )

    counts = fst["fst"].to_numpy()
    lines.append(
        "  Highly differentiated loci: "
        f"{_fmt(int((counts > 0.2).sum()), 0)} with FST > 0.2 and {_fmt(int((counts > 0.5).sum()), 0)} with FST > 0.5."
    )
    return lines


def summarize_frf() -> List[str]:
    frf_path = DATA_DIR / "per_inversion_frf_effects.tsv"
    if not frf_path.exists():
        return ["Breakpoint FRF results not found; skipping enrichment analysis."]

    frf = pd.read_csv(frf_path, sep="\t", low_memory=False)
    frf = frf.rename(columns={"frf_delta": "edge_minus_middle", "usable_for_meta": "usable"})
    if {"chrom", "start", "end"}.issubset(frf.columns):
        frf["chromosome_norm"] = frf["chrom"].astype(str).str.replace("^chr", "", regex=True)
        inv_df = _load_inv_properties()
        frf = frf.merge(
            inv_df[["chromosome", "start", "end", "recurrence_flag", "recurrence_label", "inversion_id"]],
            left_on=["chromosome_norm", "start", "end"],
            right_on=["chromosome", "start", "end"],
            how="left",
            suffixes=("", "_inv"),
        )
    lines: List[str] = ["Breakpoint enrichment (flat–ramp–flat model deltas):"]

    usable = frf[frf.get("usable", True) == True]
    if "recurrence_flag" not in usable.columns:
        lines.append("  Recurrence annotations missing; cannot stratify FRF deltas.")
        return lines

    vecs: dict[int, np.ndarray] = {}
    deltas: dict[int, float] = {}
    counts: dict[int, int] = {}
    for flag in [0, 1]:
        subset = usable[usable["recurrence_flag"] == flag]
        if subset.empty:
            continue
        vec = subset["edge_minus_middle"].dropna().to_numpy(dtype=float)
        if vec.size == 0:
            continue
        vecs[flag] = vec
        deltas[flag] = float(np.mean(vec))
        counts[flag] = int(vec.size)
        label = "Single-event" if flag == 0 else "Recurrent"
        lines.append(
            f"  {label}: mean(FST_flank - FST_middle) = {_fmt(deltas[flag], 3)} (n = {_fmt(counts[flag], 0)})."
        )

    if set(deltas) == {0, 1}:
        diff = deltas[0] - deltas[1]
        lines.append(
            "  Difference-of-differences (single-event minus recurrent): "
            f"{_fmt(diff, 3)}."
        )

    if set(vecs) == {0, 1}:
        if len(vecs[0]) > 0 and len(vecs[1]) > 0:
            res = recur_breakpoint_tests.directional_energy_test(
                vecs[0],
                vecs[1],
                n_perm=10000,
                random_state=2025,
            )
            lines.append(
                "  Global permutation test (Energy distance, Single-event > Recurrent): "
                f"p = {_fmt(res['p_value_0gt1'], 3)}."
            )

    return lines


# ---------------------------------------------------------------------------
# Section 4. PheWAS breadth and highlights
# ---------------------------------------------------------------------------


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
        f"{_fmt(results['N_Cases'].min(), 0)} to {_fmt(results['N_Cases'].max(), 0)}; "
        f"controls span {_fmt(results['N_Controls'].min(), 0)}–{_fmt(results['N_Controls'].max(), 0)}."
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

    return lines


@dataclass
class AssocSpec:
    inversion: str
    label: str
    search_terms: Tuple[str, ...]
    table_name: str = "phewas_results.tsv"


def _format_or(row: pd.Series) -> str:
    or_col = None
    for candidate in ["OR", "Odds_Ratio", "OR_overall"]:
        if candidate in row.index:
            or_col = candidate
            break
    if or_col is None:
        return "Odds ratio unavailable"

    or_value = row.get(or_col)
    lo = None
    hi = None
    for lo_candidate in [
        "CI_Lower",
        "CI95_Lower",
        "CI_Lower_Overall",
        "CI_LO_OR",
        "CI_Lower_DISPLAY",
    ]:
        if lo_candidate in row.index and not pd.isna(row.get(lo_candidate)):
            lo = row.get(lo_candidate)
            break
    for hi_candidate in [
        "CI_Upper",
        "CI95_Upper",
        "CI_Upper_Overall",
        "CI_HI_OR",
        "CI_Upper_DISPLAY",
    ]:
        if hi_candidate in row.index and not pd.isna(row.get(hi_candidate)):
            hi = row.get(hi_candidate)
            break
    if lo is not None and hi is not None:
        return f"OR = {_fmt(or_value, 3)} (95% CI {_fmt(lo, 3)}–{_fmt(hi, 3)})"
    return f"OR = {_fmt(or_value, 3)}"


def summarize_key_associations() -> List[str]:
    SOURCE_LABELS = {
        "phewas_results.tsv": "MAIN IMPUTED",
        "all_pop_phewas_tag.tsv": "TAG SNP",
        "PGS_controls.tsv": "PGS CONTROL",
    }

    targets = [
        AssocSpec(
            "chr10-79542902-INV-674513",
            "Positive DNA test for high-risk HPV types",
            ("hpv", "dna", "positive"),
        ),
        AssocSpec(
            "chr6-141867315-INV-29159",
            "Laryngitis and tracheitis",
            ("laryngitis", "tracheitis"),
        ),
        AssocSpec(
            "chr12-46897663-INV-16289",
            "Conjunctivitis",
            ("conjunct",),
        ),
        AssocSpec(
            "chr12-46897663-INV-16289",
            "Migraine",
            ("migraine",),
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Morbid obesity (Main Imputed)",
            ("morbid", "obesity"),
            table_name="phewas_results.tsv",
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Morbid obesity (Tag SNP)",
            ("morbid", "obesity"),
            table_name="all_pop_phewas_tag.tsv",
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Breast lump or abnormal exam",
            ("lump", "breast"),
            table_name="all_pop_phewas_tag.tsv",
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Abnormal mammogram",
            ("mammogram",),
            table_name="all_pop_phewas_tag.tsv",
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Mild cognitive impairment",
            ("mild", "cognitive"),
            table_name="all_pop_phewas_tag.tsv",
        ),
    ]

    table_names = sorted({spec.table_name for spec in targets})
    tables: dict[str, pd.DataFrame] = {}
    missing_tables: List[str] = []
    inv_meta_path = DATA_DIR / "inv_properties.tsv"
    for name in table_names:
        path = DATA_DIR / name
        if not path.exists():
            missing_tables.append(name)
            continue
        df = pd.read_csv(path, sep="\t", low_memory=False)
        if "Phenotype" not in df.columns or "Inversion" not in df.columns:
            missing_tables.append(name)
            continue
        df["Phenotype"] = df["Phenotype"].astype(str)
        df["Inversion"] = df["Inversion"].astype(str)
        df["Inversion"] = map_inversion_series(df["Inversion"], inv_info_path=str(inv_meta_path))
        tables[name] = df

    if not tables:
        missing_desc = f" ({', '.join(sorted(set(missing_tables)))})" if missing_tables else ""
        return [
            "Per-phenotype association tables not found; skipping highlights" + missing_desc + "."
        ]

    available_sources = list(tables.keys())
    lines: List[str] = [
        "Selected inversion–phenotype associations (logistic regression with LRT p-values):",
        "  Available source tables: " + ", ".join(available_sources) + ".",
    ]
    if missing_tables:
        lines.append("  Missing source tables: " + ", ".join(sorted(set(missing_tables))) + ".")

    for spec in targets:
        table = tables.get(spec.table_name)
        if table is None:
            lines.append(
                f"  {spec.inversion}: source table {spec.table_name} not available locally; "
                f"cannot summarize {spec.label}."
            )
            continue

        target_inv_id = map_inversion_value(spec.inversion, inv_info_path=str(inv_meta_path))
        subset = table[table["Inversion"].str.strip() == target_inv_id]
        if subset.empty:
            lines.append(
                f"  {spec.inversion}: no PheWAS results available locally for {spec.label}."
            )
            continue

        mask = np.ones(len(subset), dtype=bool)
        norm_labels = subset["Phenotype"].astype(str).str.lower()
        for term in spec.search_terms:
            mask &= norm_labels.str.contains(term, na=False)
        candidates = subset[mask]

        if candidates.empty:
            lines.append(
                f"  {spec.inversion} × {spec.label}: matching phenotype not found in {spec.table_name}."
            )
            continue

        sort_columns = [
            col
            for col in ["P_Value", "P_Value_y", "P_Value_x", "P_LRT_Overall"]
            if col in candidates.columns
        ]
        if sort_columns:
            r = candidates.sort_values(sort_columns).iloc[0]
        else:
            r = candidates.iloc[0]

        pval = None
        for col in [
            "P_Value",
            "P_Value_y",
            "P_Value_x",
            "P_LRT_Overall",
            "P_Value_LRT_Bootstrap",
        ]:
            value = r.get(col)
            if value is not None and not pd.isna(value):
                pval = value
                break

        bh = None
        for col in ["Q_GLOBAL", "BH_FDR_Q"]:
            value = r.get(col)
            if value is not None and not pd.isna(value):
                bh = value
                break
        if bh is None:
            bh = pval
        parts = _format_or(r)
        source_lbl = SOURCE_LABELS.get(spec.table_name, "UNKNOWN SOURCE")
        lines.append(
            f"  [{source_lbl}] {spec.inversion} vs {spec.label}: {parts}, "
            f"BH-adjusted p ≈ {_fmt(bh, 3)} (raw p = {_fmt(pval, 3)})."
        )
    return lines


def summarize_category_tests() -> List[str]:
    cat_path = DATA_DIR / "phewas v2 - categories.tsv"
    if not cat_path.exists():
        return ["Phecode category-level omnibus results not found; skipping summary."]

    categories = pd.read_csv(cat_path, sep="\t", low_memory=False)
    required = {"Inversion", "Category", "P_GBJ", "P_GLS", "Q_GBJ", "Q_GLS"}
    if not required.issubset(categories.columns):
        missing = ", ".join(sorted(required - set(categories.columns)))
        return [f"Category table missing required columns: {missing}."]

    lines = ["Phecode category omnibus and directional tests:"]
    for inv, group in categories.groupby("Inversion"):
        sig = group[(group["Q_GBJ"] < 0.05) | (group["Q_GLS"] < 0.05)]
        if sig.empty:
            continue
        summaries = []
        for row in sig.itertuples():
            gbj_q = _fmt(row.Q_GBJ, 3) if not pd.isna(row.Q_GBJ) else "NA"
            gls_q = _fmt(row.Q_GLS, 3) if not pd.isna(row.Q_GLS) else "NA"
            gbj_p = _fmt(row.P_GBJ, 3) if not pd.isna(row.P_GBJ) else "NA"
            gls_p = _fmt(row.P_GLS, 3) if not pd.isna(row.P_GLS) else "NA"
            summaries.append(
                f"{row.Category}: GBJ q = {gbj_q} (p = {gbj_p}), GLS q = {gls_q} (p = {gls_p})"
            )
        lines.append(f"  {inv}: " + "; ".join(summaries))

    if len(lines) == 1:
        lines.append("  No categories reached the significance threshold (q < 0.05).")
    return lines


# ---------------------------------------------------------------------------
# Section 5. Imputation performance
# ---------------------------------------------------------------------------


def summarize_imputation() -> List[str]:
    path = DATA_DIR / "imputation_results.tsv"
    if not path.exists():
        return [f"Imputation summary not found at {path}."]

    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"unbiased_pearson_r2": "r2", "p_fdr_bh": "bh_p"})
    usable = df[(df["r2"] > 0.3) & (df["bh_p"] < 0.05)]
    lines = ["Imputation performance summary:"]
    lines.append(
        f"  Models evaluated: {_fmt(len(df), 0)}; models with r² > 0.3 and BH p < 0.05: {_fmt(len(usable), 0)}."
    )
    if "Use" in df.columns:
        lines.append(
            f"  Models flagged for downstream PheWAS (Use == True): {_fmt(int(df['Use'].eq(True).sum()), 0)}."
        )
    return lines


# ---------------------------------------------------------------------------
# Section 6. PGS covariate sensitivity and selection
# ---------------------------------------------------------------------------


def summarize_pgs_controls() -> List[str]:
    candidates = [
        (DATA_DIR / "pgs_sensitivity.tsv", {}),
        (
            DATA_DIR / "PGS_controls.tsv",
            {
                "P_Value_NoCustomControls": "p_nominal",
                "P_Value": "p_with_pgs",
            },
        ),
    ]

    pgs: pd.DataFrame | None = None
    source = None
    for path, rename_map in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", low_memory=False)
        if rename_map:
            df = df.rename(columns=rename_map)
        required = {"Inversion", "Phenotype", "p_nominal", "p_with_pgs"}
        if not required.issubset(df.columns):
            continue
        pgs = df
        source = path.name
        break

    if pgs is None:
        return ["Polygenic-score sensitivity table not found; skipping summary."]

    pgs = pgs.replace([np.inf, -np.inf], np.nan)
    pgs = pgs.dropna(subset=["p_nominal", "p_with_pgs"])
    if pgs.empty:
        return ["PGS sensitivity table empty after filtering p-values."]

    pgs["fold_change"] = pgs["p_with_pgs"] / pgs["p_nominal"].replace(0, np.nan)
    largest = pgs.sort_values("fold_change", ascending=False).iloc[0]

    lines = [
        "[PGS CONTROL] Sensitivity of PheWAS associations to regional PGS covariates:",
        f"  Source table: {source}.",
    ]
    lines.append(
        f"  Largest p-value inflation: inversion {largest.Inversion} × {largest.Phenotype} "
        f"(p_nominal = {_fmt(largest.p_nominal, 3)}, p_with_pgs = {_fmt(largest.p_with_pgs, 3)}, "
        f"fold-change = {_fmt(largest.fold_change, 3)})."
    )
    return lines


def summarize_family_history() -> List[str]:
    fam_path = REPO_ROOT / "assoc_outputs" / "assoc_family_groups_gee_single_beta.csv"

    if not fam_path.exists():
        return [
            "Family history validation results not found (run stats/extra/family.py first)."
        ]

    try:
        df = pd.read_csv(fam_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        return [f"Error reading family history results: {exc}"]

    if "phenotype" not in df.columns:
        return [
            "Family history validation file missing 'phenotype' column; cannot summarize results."
        ]

    lines = ["Family History Validation (GEE Model):"]
    key_phenos = ["Breast Cancer", "Obesity", "Heart Failure", "Cognitive Impairment"]

    found_any = False
    for pheno in key_phenos:
        mask = df["phenotype"].astype(str).str.contains(pheno, case=False, na=False)
        row = df[mask]
        if row.empty:
            continue
        found_any = True
        r = row.iloc[0]
        or_val = r.get("OR")
        ci_lo = r.get("CI_low")
        ci_hi = r.get("CI_high")
        p_val = r.get("p")
        lines.append(
            f"  [FAMILY FOLLOW-UP] {pheno}: OR = {_fmt(or_val, 3)} "
            f"(95% CI {_fmt(ci_lo, 3)}–{_fmt(ci_hi, 3)}), p = {_fmt(p_val, 3)}."
        )

    if not found_any:
        lines.append("  No manuscript phenotypes recovered from family history validation table.")
    return lines


def _largest_window_change(dates: pd.Series, values: pd.Series, window: float = 1000.0) -> Tuple[float, float, float] | None:
    mask = dates.notna() & values.notna()
    if mask.sum() < 2:
        return None

    filtered_dates = dates[mask].to_numpy()
    filtered_values = values[mask].to_numpy()
    sorted_idx = np.argsort(filtered_dates)
    sorted_dates = filtered_dates[sorted_idx]
    sorted_values = filtered_values[sorted_idx]

    min_date = float(sorted_dates[0])
    max_date = float(sorted_dates[-1])
    if max_date - min_date < window:
        return None

    start_points = np.arange(min_date, max_date - window + 1, 1.0)
    if start_points.size == 0:
        return None
    end_points = start_points + window

    start_vals = np.interp(start_points, sorted_dates, sorted_values)
    end_vals = np.interp(end_points, sorted_dates, sorted_values)
    deltas = np.abs(end_vals - start_vals)
    idx = int(np.argmax(deltas))
    return float(start_points[idx]), float(end_points[idx]), float(deltas[idx])


def _plain_number(value: float | int | None) -> str:
    """Render numbers without scientific notation or rounding."""

    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(val) or math.isinf(val):
        return "NA"
    rounded = round(val)
    if abs(val - rounded) < 1e-9:
        return str(int(rounded))
    text = f"{val:.15f}".rstrip("0").rstrip(".")
    return text if text else "0"


def summarize_selection() -> List[str]:
    trajectory_path = DATA_DIR / "Trajectory-12_47296118_A_G.tsv"
    if not trajectory_path.exists():
        return ["Trajectory data not found; skipping summary."]

    traj = pd.read_csv(trajectory_path, sep="\t", low_memory=False)
    numeric_cols = [
        "date_left",
        "date_right",
        "date_center",
        "num_allele",
        "num_alt_allele",
        "af",
        "af_low",
        "af_up",
        "pt",
        "pt_low",
        "pt_up",
    ]
    for col in numeric_cols:
        if col in traj.columns:
            traj[col] = pd.to_numeric(traj[col], errors="coerce")

    value_col = "af" if "af" in traj.columns else "pt"
    traj = traj.dropna(subset=["date_center", value_col])
    if traj.empty:
        return ["AGES trajectory table is empty after filtering numeric values."]

    traj = traj.sort_values("date_center")
    present = traj.iloc[0]
    ancient = traj.iloc[-1]
    change = present[value_col] - ancient[value_col]
    value_min = traj[value_col].min()
    value_max = traj[value_col].max()
    sample_median = _safe_median(traj.get("num_allele"))
    window_summary = _largest_window_change(traj["date_center"], traj[value_col], window=1000.0)

    lines = [
        "Allele frequency trajectory summary (12_47296118_A_G):",
        f"  Windows analyzed: {_fmt(len(traj), 0)} spanning {_fmt(traj['date_center'].min(), 0)}–{_fmt(traj['date_center'].max(), 0)} years before present.",
        f"  Observed allele-frequency ranges {_fmt(value_min, 3)}–{_fmt(value_max, 3)}; net change from {_fmt(ancient.date_center, 0)} to {_fmt(present.date_center, 0)} years BP is {_fmt(change, 3)}.",
    ]
    if sample_median is not None:
        lines.append(
            f"  Median haploid sample size per window ≈ {_fmt(sample_median, 0)} alleles."
        )
    if window_summary is not None:
        start, end, delta = window_summary
        lines.append(
            "  Largest ~1,000-year change: "
            f"Δf = {_plain_number(delta)} between {_plain_number(start)} and {_plain_number(end)} years BP."
        )
    return lines


# ---------------------------------------------------------------------------
# Master report builder
# ---------------------------------------------------------------------------


def build_report() -> List[str]:
    sections: List[Tuple[str, Iterable[str]]] = [
        ("Recurrence", summarize_recurrence()),
        ("Sample sizes", summarize_sample_sizes()),
        ("Diversity", summarize_diversity()),
        ("Linear model", summarize_linear_model()),
        ("CDS conservation", summarize_cds_conservation_glm()),
        ("Differentiation", summarize_fst()),
        ("Breakpoint FRF", summarize_frf()),
        ("Imputation", summarize_imputation()),
        ("PheWAS scale", summarize_phewas_scale()),
        ("Key associations", summarize_key_associations()),
        ("Category tests", summarize_category_tests()),
        ("PGS controls", summarize_pgs_controls()),
        ("Family History", summarize_family_history()),
        ("Selection", summarize_selection()),
    ]

    output: List[str] = []
    for title, content in sections:
        output.append(title.upper())
        if isinstance(content, Iterable):
            for line in content:
                output.append(line)
        else:
            output.append(str(content))
        output.append("")
    return output


def main() -> None:
    lines = build_report()
    text = "\n".join(lines).strip() + "\n"
    print(text)
    REPORT_PATH.write_text(text)
    print(f"\nSaved report to {REPORT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()

