"""Print which inversions appear in each frequency/imputation plot.

This helper mirrors the filtering logic in:
- stats/imputation_plot.py
- stats/pop_dosage_plot.py
- stats/overall_AF_scatterplot.py

The r plot starts from imputation metrics and only drops entries that
lack consensus, coordinates, or r² values. The population dosage and
overall AF plots first intersect inversion metadata with the population
frequency table derived from the dosage file, and then (for the AF
scatterplot) also require a Porubsky callset entry. Because the All of
Us frequencies are sourced from the same dosage-derived table, missing
rows there reflect limited dosage coverage rather than missing WGS data.
This script reports the resulting counts and lists inversions that are
present in the r plot but absent from the other figures.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONS_COL = "0_single_1_recur_consensus"


def load_inversion_r_plot_set(data_dir: Path, debug: bool = False) -> set[str]:
    imp = pd.read_csv(data_dir / "imputation_results_merged.tsv", sep="\t", dtype=str)
    inv = pd.read_csv(data_dir / "inv_properties.tsv", sep="\t", dtype=str)
    imp.columns = imp.columns.str.strip()
    inv.columns = inv.columns.str.strip()

    merged = imp.merge(
        inv[["OrigID", "Chromosome", "Start", "End", CONS_COL]],
        left_on="id",
        right_on="OrigID",
        how="inner",
        suffixes=("_imp", "_inv"),
    )

    chrom_col = "Chromosome_inv" if "Chromosome_inv" in merged.columns else "Chromosome"
    start_col = "Start_inv" if "Start_inv" in merged.columns else "Start"
    end_col = "End_inv" if "End_inv" in merged.columns else "End"
    cons_out = f"{CONS_COL}_inv" if f"{CONS_COL}_inv" in merged.columns else CONS_COL

    merged = merged[["id", "unbiased_pearson_r2", chrom_col, start_col, end_col, cons_out]].rename(
        columns={
            chrom_col: "Chromosome",
            start_col: "Start",
            end_col: "End",
            cons_out: CONS_COL,
        }
    )

    merged["unbiased_pearson_r2"] = pd.to_numeric(merged["unbiased_pearson_r2"], errors="coerce")
    merged[CONS_COL] = pd.to_numeric(merged[CONS_COL], errors="coerce")
    merged["Start"] = pd.to_numeric(merged["Start"], errors="coerce")
    merged["End"] = pd.to_numeric(merged["End"], errors="coerce")

    # Match stats/imputation_plot.py: only consensus 0/1 plus non-null r^2 and
    # coordinates. No q-value filter is applied in the plotting script.
    plot_df = merged[merged[CONS_COL].isin([0, 1])].dropna(
        subset=["unbiased_pearson_r2", "Chromosome", "Start", "End"]
    )

    if debug:
        print("[debug] imputation_results_merged.tsv rows:", len(imp), "unique ids:", imp["id"].nunique())
        print("[debug] after merge+filter for inversion_r_plot:", len(plot_df), "unique ids:", plot_df["id"].nunique())
    return set(plot_df["id"])


def load_pop_dosage_set(
    data_dir: Path, debug: bool = False
) -> tuple[set[str], dict[str, set[str]]]:
    freq = pd.read_csv(data_dir / "inversion_population_frequencies.tsv", sep="\t")
    freq_raw_overall = freq[freq["Population"] == "ALL"][["Inversion"]].drop_duplicates()
    inv_props = pd.read_csv(data_dir / "inv_properties.tsv", sep="\t")
    freq.columns = freq.columns.str.strip()
    inv_props.columns = inv_props.columns.str.strip()

    inv_props[CONS_COL] = pd.to_numeric(inv_props[CONS_COL], errors="coerce")
    inv_props = inv_props[inv_props[CONS_COL].isin([0, 1])].copy()
    inv_props["Start"] = pd.to_numeric(inv_props["Start"], errors="coerce")
    inv_props["End"] = pd.to_numeric(inv_props["End"], errors="coerce")
    inv_props = inv_props[inv_props[["Start", "End"]].notna().all(axis=1)]
    inv_props = inv_props.rename(columns={"OrigID": "Inversion"})

    freq["N"] = pd.to_numeric(freq["N"], errors="coerce")
    freq = freq[freq["N"] > 1]
    for col in [
        "AF_Q1",
        "AF_Median",
        "AF_Q3",
        "AF_Lower_Whisker",
        "AF_Upper_Whisker",
    ]:
        freq = freq[freq[col].notna()]

    freq = freq.merge(
        inv_props[["Inversion", "Chromosome", "Start", "End"]],
        on="Inversion",
        how="inner",
    )

    overall = freq[freq["Population"] == "ALL"].set_index("Inversion")
    overall_ids = set(overall.index)
    overall_raw_ids = set(freq_raw_overall["Inversion"])

    if debug:
        print(
            "[debug] inversion_population_frequencies.tsv rows:",
            len(freq),
            "unique inversions:",
            freq["Inversion"].nunique(),
        )
        print(
            "[debug] after filters for pop_dosage_plot (ALL pop):",
            len(overall),
            "unique ids:",
            overall.index.nunique(),
        )
    return set(overall.sort_values("AF_Median", ascending=False).index), {
        "pop_overall_ids": overall_ids,
        "pop_overall_raw_ids": overall_raw_ids,
    }


def load_overall_af_set(
    data_dir: Path, debug: bool = False
) -> tuple[set[str], dict[str, set[str]]]:
    callset = pd.read_csv(
        data_dir / "2AGRCh38_unifiedCallset - 2AGRCh38_unifiedCallset.tsv", sep="\t"
    )
    inv_props = pd.read_csv(data_dir / "inv_properties.tsv", sep="\t")
    inv_props.columns = inv_props.columns.str.strip()
    inv_props[CONS_COL] = pd.to_numeric(inv_props[CONS_COL], errors="coerce")
    inv_props = inv_props[inv_props[CONS_COL].isin([0, 1])][
        ["OrigID", "Chromosome", "Start", "End"]
    ].drop_duplicates("OrigID")

    allowed = set(inv_props["OrigID"])
    cs = callset.drop_duplicates("inv_id")
    genotype_cols = [
        c
        for c in cs.columns
        if c
        not in {
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
            "recurrence",
            "start1",
            "end1",
            "start2",
            "end2",
            "chrom",
            "AF_filtered",
            "event_type",
            "AF",
            "inversion_type",
            "id_left",
            "id_right",
            "tbi",
            "fai",
            "bam",
            "vcf",
            "external_comments",
            "freq",
        }
        and not c.startswith("Unnamed")
    ]
    allowed_genotypes = {"1|1": 2, "1|0": 1, "0|1": 1, "0|0": 0}

    records = []
    missing_callset_freq: set[str] = set()
    callset_present_allowed: set[str] = set()
    for _, row in cs.iterrows():
        inv_id = row["inv_id"]
        if inv_id not in allowed:
            continue
        callset_present_allowed.add(inv_id)
        alt = valid = 0
        for col in genotype_cols:
            gt = row[col]
            if pd.isna(gt):
                continue
            gt = str(gt)
            if gt.lower() == "nan":
                continue
            if gt in allowed_genotypes:
                alt += allowed_genotypes[gt]
                valid += 1
        if valid == 0:
            missing_callset_freq.add(inv_id)
            continue
        af = alt / (2 * valid)
        records.append((inv_id, af))

    call_df = pd.DataFrame(records, columns=["OrigID", "callset_af"])
    freq = pd.read_csv(data_dir / "inversion_population_frequencies.tsv", sep="\t")
    freq.columns = freq.columns.str.strip()
    aou = freq[freq["Population"] == "ALL"][
        ["Inversion", "Allele_Freq", "CI95_Lower", "CI95_Upper"]
    ].copy()
    aou = aou.rename(
        columns={
            "Inversion": "OrigID",
            "Allele_Freq": "aou_af",
            "CI95_Lower": "aou_ci_low",
            "CI95_Upper": "aou_ci_high",
        }
    )
    aou = aou.drop_duplicates("OrigID")

    merged = inv_props[["OrigID"]].merge(call_df, on="OrigID", how="inner").merge(
        aou, on="OrigID", how="inner"
    )
    merged = merged.dropna(subset=["callset_af", "aou_af"])

    callset_ids = set(call_df["OrigID"])
    aou_ids = set(aou["OrigID"])

    if debug:
        print(
            "[debug] callset rows:",
            len(callset),
            "unique inv_ids:",
            callset["inv_id"].nunique(),
        )
        print(
            "[debug] callset-derived AF records after filters:",
            len(call_df),
            "unique ids:",
            call_df["OrigID"].nunique(),
        )
        print(
            "[debug] callset rows skipped because no diploid genotype codes were",\
            " present (cannot derive AF):",
            len(missing_callset_freq),
        )
        print(
            "[debug] AoU ALL-pop rows:",
            len(aou),
            "unique ids:",
            aou["OrigID"].nunique(),
        )
        print(
            "[debug] merged rows for overall_AF_scatterplot_with_ci:",
            len(merged),
            "unique ids:",
            merged["OrigID"].nunique(),
        )
    return set(merged["OrigID"]), {
        "callset_ids": callset_ids,
        "callset_present_allowed": callset_present_allowed,
        "aou_ids": aou_ids,
        "aou_source_ids": set(aou["OrigID"]),
        "missing_callset_freq": missing_callset_freq,
    }


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print intermediate counts pulled directly from data/*.tsv",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Directory containing the input TSVs (default: ./data)",
    )
    return parser


def main(data_dir: Path, debug: bool = False) -> None:
    r_plot = load_inversion_r_plot_set(data_dir=data_dir, debug=debug)
    pop_dosage, pop_details = load_pop_dosage_set(data_dir=data_dir, debug=debug)
    overall_af, af_details = load_overall_af_set(data_dir=data_dir, debug=debug)

    print(f"inversion_r_plot: {len(r_plot)} inversions")
    print(f"pop_dosage_plot: {len(pop_dosage)} inversions")
    print(f"overall_AF_scatterplot_with_ci: {len(overall_af)} inversions")

    missing_pop = sorted(r_plot - pop_dosage)
    missing_af = sorted(r_plot - overall_af)
    missing_either = sorted(r_plot - (pop_dosage | overall_af))

    if debug:
        print(
            "\n[debug] reason breakdown vs pop_dosage_plot (ALL population entries)",
            f"missing from frequency table (ALL rows absent): {len(r_plot - pop_details['pop_overall_raw_ids'])}",
            f"| present in ALL rows but filtered by QC: {len((r_plot & pop_details['pop_overall_raw_ids']) - pop_details['pop_overall_ids'])}",
        )
        print(
            "[debug] reason breakdown vs overall_AF_scatterplot_with_ci",
            f"missing any callset row: {len(r_plot - af_details['callset_present_allowed'])}",
            "| callset rows lack diploid 0/1 genotypes:",
            len(r_plot & af_details["missing_callset_freq"]),
            "| missing AoU ALL-pop frequencies:",
            len(r_plot - af_details["aou_ids"]),
        )
        if len(r_plot - af_details["aou_ids"]) and not (r_plot - af_details["callset_present_allowed"]):
            print(
                "[debug] note: overall_AF exclusions stem from the dosage-derived",
                "AoU frequency table when the callset has diploid data.",
            )

    def pop_reason(inv: str) -> str:
        if inv in pop_dosage:
            return ""
        if inv not in pop_details["pop_overall_raw_ids"]:
            return "not present in inversion_population_frequencies.tsv (ALL population)"
        if inv not in pop_details["pop_overall_ids"]:
            return "excluded by population frequency QC (N<=1 or missing AF quartiles)"
        return "excluded for unknown population-frequency reason"

    def af_reason(inv: str) -> str:
        if inv in overall_af:
            return ""
        if inv not in af_details["callset_present_allowed"]:
            return "no callset row for this inversion"
        if inv in af_details["missing_callset_freq"]:
            return "callset row lacks diploid 0/1 genotypes so AF is NA"
        if inv not in af_details["aou_ids"]:
            return "no AoU ALL-population frequency row"
        return "excluded for unknown AF merge reason"

    if missing_either:
        print("\nPer-inversion exclusion reasons (imputation plot only):")
        rows = []
        for inv in missing_either:
            rows.append(
                {
                    "Inversion": inv,
                    "missing_pop_dosage_plot": pop_reason(inv),
                    "missing_overall_AF_plot": af_reason(inv),
                }
            )
        detail_df = pd.DataFrame(rows)
        print(detail_df.to_string(index=False))

    print("\nInversion_r_plot entries not in pop_dosage_plot:")
    print("\n".join(missing_pop))

    print("\nInversion_r_plot entries not in overall_AF_scatterplot_with_ci:")
    print("\n".join(missing_af))

    print("\nInversion_r_plot entries missing from both other plots:")
    print("\n".join(missing_either))


if __name__ == "__main__":
    args = parse_args().parse_args()
    main(data_dir=args.data_dir, debug=args.debug)
