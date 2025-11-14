"""Replicate manuscript metrics and tests from staged summary tables.

This script consolidates the computations used throughout the manuscript into a
single reproducible entry point.  It mirrors the statistics described in the
Results text, prints a human-readable report, and writes the same content to
``stats/replicate_full_metrics.txt`` for archival purposes.

Each section below draws on tables or intermediate outputs that already live in
the ``data`` directory.  When a statistic depends on a previously published
analysis module, we re-use the corresponding helper (for example the
orientation × recurrence linear model in ``stats.inv_dir_recur_model``).  If a
required data product is missing, the script records this explicitly so the
reader understands which metrics could not be regenerated in the current
checkout.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats import inv_dir_recur_model  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
REPORT_PATH = Path(__file__).with_suffix(".txt")


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------


def _fmt(value: float | int | None, digits: int = 3) -> str:
    """Format floating-point numbers with sensible scientific notation.

    Integers are rendered without decimal places.  Very small or very large
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
    fst_path = DATA_DIR / "hudson_fst_results.tsv"
    if not fst_path.exists():
        return None
    fst = pd.read_csv(fst_path, sep="\t", low_memory=False)
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
# Section 1. Sample sizes and diversity summaries
# ---------------------------------------------------------------------------


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

    inv_df = _load_inv_properties()
    high_quality = inv_df
    lines.append(
        "  High-quality inversions with consensus recurrence calls: "
        f"{_fmt(len(high_quality), 0)} total (single-event = {_fmt(int((high_quality['recurrence_flag']==0).sum()), 0)}, "
        f"recurrent = {_fmt(int((high_quality['recurrence_flag']==1).sum()), 0)})."
    )

    return lines


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
            "region_id": inv_ids.fillna(""),
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

    all_pi = np.concatenate([matched["pi_direct"].to_numpy(), matched["pi_inverted"].to_numpy()])
    eps = inv_dir_recur_model.choose_floor_from_quantile(
        all_pi, q=inv_dir_recur_model.FLOOR_QUANTILE, min_floor=inv_dir_recur_model.MIN_FLOOR
    )
    _, effects, _ = inv_dir_recur_model.run_model_A(matched, eps=eps, nonzero_only=False)

    lines = ["Orientation × recurrence linear model on log π ratios (Model A):"]
    lines.append(f"  Detection floor ε before log transform: {_fmt(eps, 6)}.")
    for effect in effects.itertuples():
        lines.append(
            f"  {effect.effect}: fold-change = {_fmt(effect.ratio, 3)} "
            f"(95% CI {_fmt(effect.ci_low, 3)}–{_fmt(effect.ci_high, 3)}), p = {_fmt(effect.p, 3)}."
        )
    return lines


# ---------------------------------------------------------------------------
# Section 2. Differentiation and breakpoint enrichment
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

    deltas: dict[int, float] = {}
    for flag, label in [(0, "Single-event"), (1, "Recurrent")]:
        sub = usable[usable["recurrence_flag"] == flag]
        if sub.empty:
            lines.append(f"  {label}: no usable inversions with FRF estimates.")
            continue
        mean_delta = float(sub["edge_minus_middle"].mean())
        deltas[flag] = mean_delta
        lines.append(
            f"  {label}: mean flank–middle FST difference = {_fmt(mean_delta, 3)} (n = {_fmt(len(sub), 0)})."
        )

    if 0 in deltas and 1 in deltas:
        diff = deltas[0] - deltas[1]
        lines.append(
            "  Difference in edge enrichment (single-event − recurrent): "
            f"Δ = {_fmt(diff, 3)}."
        )
    return lines


# ---------------------------------------------------------------------------
# Section 3. PheWAS-scale summaries and association highlights
# ---------------------------------------------------------------------------


def _load_phewas_results() -> pd.DataFrame | None:
    path = DATA_DIR / "phewas_results.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["Phenotype"] = df["Phenotype"].astype(str)
    return df


def summarize_phewas_scale() -> List[str]:
    results = _load_phewas_results()
    if results is None or results.empty:
        return ["PheWAS summary table unavailable."]

    lines = ["PheWAS scale and coverage:"]
    lines.append(f"  Unique phecodes tested: {_fmt(results['Phenotype'].nunique(), 0)}.")
    lines.append(
        "  Case counts span "
        f"{_fmt(results['N_Cases'].min(), 0)}–{_fmt(results['N_Cases'].max(), 0)}; "
        f"controls span {_fmt(results['N_Controls'].min(), 0)}–{_fmt(results['N_Controls'].max(), 0)}."
    )
    inv_counts = results.groupby("Inversion")["Phenotype"].nunique().sort_values(ascending=False)
    top = ", ".join(f"{inv}: {count}" for inv, count in inv_counts.head(5).items())
    lines.append(f"  Phenotype coverage per inversion (top five): {top}{'; ...' if len(inv_counts) > 5 else ''}.")

    sig_col = results.get("Sig_Global")
    if sig_col is not None:
        sig_mask = sig_col.astype(str).str.upper() == "TRUE"
        lines.append(
            "  Inversions with ≥1 BH-significant association: "
            f"{_fmt(results.loc[sig_mask, 'Inversion'].nunique(), 0)} of {_fmt(results['Inversion'].nunique(), 0)}."
        )
    return lines


@dataclass
class AssocSpec:
    inversion: str
    label: str
    search_terms: Tuple[str, ...]


def _format_or(row: pd.Series) -> str:
    or_val = row.get("OR") or row.get("OR_MLE") or row.get("OR_x")
    lo = row.get("CI_LO_OR_DISPLAY") or row.get("OR_95CI_Lower")
    hi = row.get("CI_HI_OR_DISPLAY") or row.get("OR_95CI_Upper")
    parts = [f"OR = {_fmt(or_val, 3)}"]
    if pd.notna(lo) and pd.notna(hi):
        parts.append(f"95% CI {_fmt(lo, 3)}–{_fmt(hi, 3)}")
    return ", ".join(parts)


def summarize_key_associations() -> List[str]:
    path = DATA_DIR / "all_pop_phewas_tag.tsv"
    if not path.exists():
        return ["Per-phenotype association table not found; skipping highlights."]

    assoc = pd.read_csv(path, sep="\t", low_memory=False)
    assoc["Phenotype"] = assoc["Phenotype"].astype(str)
    assoc["Inversion"] = assoc["Inversion"].astype(str)

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
            "Morbid obesity",
            ("morbid", "obesity"),
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Breast lump or abnormal exam",
            ("lump", "breast"),
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Abnormal mammogram",
            ("mammogram",),
        ),
    ]

    lines: List[str] = ["Selected inversion–phenotype associations (logistic regression with LRT p-values):"]
    for spec in targets:
        subset = assoc[assoc["Inversion"].str.strip() == spec.inversion]
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
                f"  {spec.inversion} × {spec.label}: matching phenotype not found in local table."
            )
            continue

        r = candidates.sort_values("P_Value_y").iloc[0]
        pval = r.get("P_Value_y") or r.get("P_Value") or r.get("P_LRT_Overall")
        bh = r.get("Q_GLOBAL") or r.get("P_Value_y")
        parts = _format_or(r)
        lines.append(
            f"  {spec.inversion} vs {spec.label}: {parts}, "
            f"BH-adjusted p ≈ {_fmt(bh, 3)} (raw p = {_fmt(pval, 3)})."
        )
    return lines


def summarize_category_tests() -> List[str]:
    path = DATA_DIR / "phewas v2 - categories.tsv"
    if not path.exists():
        return ["Category-level omnibus tests not available."]

    cat = pd.read_csv(path, sep="\t", low_memory=False)
    lines = ["Phecode category omnibus tests (GBJ/GLS):"]

    for inv in sorted(cat["Inversion"].unique()):
        sub = cat[cat["Inversion"] == inv]
        if sub.empty:
            continue
        sig = sub[(sub["P_GBJ"] < 0.05) | (sub["P_GLS"] < 0.05)]
        if sig.empty:
            continue
        parts = []
        for _, row in sig.iterrows():
            pg = row.get("P_GBJ")
            gls = row.get("P_GLS")
            if pd.notna(pg) and pd.notna(gls):
                parts.append(f"{row['Category']} (GBJ p={_fmt(pg, 3)}, GLS p={_fmt(gls, 3)})")
            elif pd.notna(pg):
                parts.append(f"{row['Category']} (GBJ p={_fmt(pg, 3)})")
            elif pd.notna(gls):
                parts.append(f"{row['Category']} (GLS p={_fmt(gls, 3)})")
        if parts:
            lines.append(f"  {inv}: {', '.join(parts)}")
    return lines


def summarize_pgs_controls() -> List[str]:
    path = DATA_DIR / "PGS_controls.tsv"
    if not path.exists():
        return ["Regional PGS covariate comparison not found (PGS_controls.tsv missing)."]

    df = pd.read_csv(path, sep="\t", low_memory=False)
    lines = ["Effect of regional polygenic score controls:"]

    df["fold_change"] = df.apply(
        lambda row: np.nan
        if pd.isna(row.get("P_Value")) or pd.isna(row.get("P_Value_NoCustomControls")) or row["P_Value_NoCustomControls"] == 0
        else abs(float(row["P_Value_NoCustomControls"])) / abs(float(row["P_Value"])) ,
        axis=1,
    )

    for _, row in df.sort_values("P_Value").iterrows():
        lines.append(
            f"  {row['Phenotype']} ({row['Inversion']}): p without PGS = {_fmt(row['P_Value_NoCustomControls'], 3)}, "
            f"with PGS = {_fmt(row['P_Value'], 3)}; OR with controls = {_fmt(row['OR'], 3)}."
        )

    if df["fold_change"].notna().any():
        max_change = df.loc[df["fold_change"].idxmax()]
        lines.append(
            "  Largest p-value magnitude change: "
            f"{max_change['Phenotype']} (fold-change ≈ {_fmt(max_change['fold_change'], 3)})."
        )
    return lines


# ---------------------------------------------------------------------------
# Section 4. Selection trajectories
# ---------------------------------------------------------------------------


def summarize_selection() -> List[str]:
    traj_path = DATA_DIR / "Trajectory-12_47296118_A_G.tsv"
    lines: List[str] = ["Ancient DNA selection summary (AGES trajectories):"]
    if not traj_path.exists():
        lines.append(f"  Trajectory file not found at {traj_path}.")
        return lines

    traj = pd.read_csv(traj_path, sep="\t")
    if "date_center" not in traj.columns or "pt" not in traj.columns:
        lines.append("  Trajectory columns incomplete; cannot compute change summaries.")
        return lines

    dates = traj["date_center"].to_numpy(float)
    pt = traj["pt"].to_numpy(float)
    order = np.argsort(dates)
    dates = dates[order]
    pt = pt[order]

    # Compute largest change over 1,000-year window (approximate)
    window = 1000
    best_change = None
    best_start = None
    best_end = None
    for start_idx in range(len(dates)):
        start = dates[start_idx]
        end = start + window
        end_idx = np.searchsorted(dates, end, side="right") - 1
        if end_idx <= start_idx:
            continue
        change = abs(pt[end_idx] - pt[start_idx])
        if best_change is None or change > best_change:
            best_change = change
            best_start = start
            best_end = dates[end_idx]
    if best_change is not None:
        lines.append(
            "  Largest predicted allele-frequency change over a 1,000-year window occurs between "
            f"{_fmt(best_start, 0)} and {_fmt(best_end, 0)} years BP (Δ ≈ {_fmt(best_change, 3)})."
        )
    return lines


# ---------------------------------------------------------------------------
# Master report builder
# ---------------------------------------------------------------------------


def build_report() -> List[str]:
    sections: List[Tuple[str, Iterable[str]]] = [
        ("Sample sizes", summarize_sample_sizes()),
        ("Diversity", summarize_diversity()),
        ("Linear model", summarize_linear_model()),
        ("Differentiation", summarize_fst()),
        ("Breakpoint FRF", summarize_frf()),
        ("PheWAS scale", summarize_phewas_scale()),
        ("Key associations", summarize_key_associations()),
        ("Category tests", summarize_category_tests()),
        ("PGS controls", summarize_pgs_controls()),
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
