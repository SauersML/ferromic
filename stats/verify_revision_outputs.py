#!/usr/bin/env python3
"""Verify and inventory the non-PheWAS numerical claims used in the revision."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = DATA / "revision_claim_audit.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []
        self.failures: list[str] = []

    def number(
        self,
        claim: str,
        observed: float | int | str,
        expected: float | int,
        tolerance: float,
        source: Path,
    ) -> None:
        value = float(observed)
        ok = math.isfinite(value) and abs(value - float(expected)) <= tolerance
        self.rows.append(
            {
                "claim": claim,
                "observed": f"{value:.15g}",
                "expected": f"{float(expected):.15g}",
                "tolerance": f"{tolerance:.3g}",
                "source": str(source.relative_to(ROOT)),
                "source_sha256": digest(source),
                "status": "PASS" if ok else "FAIL",
            }
        )
        if not ok:
            self.failures.append(
                f"{claim}: observed {value:.15g}, expected {expected} +/- {tolerance}"
            )

    def write(self) -> None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "claim",
                    "observed",
                    "expected",
                    "tolerance",
                    "source",
                    "source_sha256",
                    "status",
                ),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(self.rows)
        if self.failures:
            raise SystemExit("revision output audit failed:\n- " + "\n- ".join(self.failures))
        print(f"verified {len(self.rows)} non-PheWAS revision claims -> {OUT}")


def main() -> None:
    audit = Audit()

    exclusions_path = DATA / "table_s5_exclusion_reasons.tsv"
    exclusions = read_tsv(exclusions_path)
    audit.number("Porubsky loci reviewed", len(exclusions), 292, 0, exclusions_path)
    audit.number(
        "consensus recurrence loci analysed",
        sum(row["analysed"] == "yes" for row in exclusions),
        93,
        0,
        exclusions_path,
    )

    sd_path = DATA / "recurrence_sd_summary.tsv"
    sd = {row["quantity"]: row for row in read_tsv(sd_path)}
    audit.number(
        "SD hard-rule agreement with consensus",
        sd["hard-rule agreement with consensus"]["value"],
        0.7957,
        5e-5,
        sd_path,
    )
    for label, expected, tolerance in (
        ("[SD hard rule (primary)] Single-event: Inverted vs Direct", 0.000296111, 5e-9),
        ("[SD hard rule (primary)] Recurrent: Inverted vs Direct", 0.63832, 5e-6),
        ("[SD hard rule (primary)] Interaction (difference between those two)", 0.000355745, 5e-9),
    ):
        audit.number(f"SD recurrence sensitivity p: {label}", sd[label]["p"], expected, tolerance, sd_path)

    four_path = DATA / "four_fold_pi_correlations.tsv"
    four = read_tsv(four_path)

    def four_row(measure_x: str, measure_y: str) -> dict[str, str]:
        found = [
            row
            for row in four
            if row["subset"] == "recurrence_classified"
            and row["measure_x"] == measure_x
            and row["measure_y"] == measure_y
            and row["statistic"] == "spearman_rho"
        ]
        if len(found) != 1:
            raise SystemExit(f"expected one four-fold row for {measure_x}/{measure_y}")
        return found[0]

    for measures, rho, p_value in (
        (("wholeLocus", "fourfold"), 0.501267, 0.00908858),
        (("fourfold", "wholeCDS"), 0.628613, 0.000583291),
    ):
        row = four_row(*measures)
        name = f"4-fold concordance {measures[0]} vs {measures[1]}"
        audit.number(f"{name} rho", row["value"], rho, 5e-7, four_path)
        audit.number(f"{name} p", row["p_value"], p_value, 5e-9, four_path)
        audit.number(f"{name} loci", row["n_loci"], 26, 0, four_path)

    div_path = DATA / "divergence_da_dxy_by_type_stats.tsv"
    div = {row["metric"]: row for row in read_tsv(div_path)}
    audit.number("Dxy recurrence comparison p", div["dxy"]["p_two_sided"], 0.9173878965, 5e-10, div_path)
    audit.number("da recurrent median", div["da"]["median_recurrent"], 4.25e-5, 5e-12, div_path)
    audit.number("da single-event median", div["da"]["median_single_event"], 1.285e-4, 5e-12, div_path)
    audit.number("da recurrence comparison p", div["da"]["p_two_sided"], 0.0373039468, 5e-10, div_path)
    audit.number(
        "Hudson FST recurrence comparison p",
        div["hudson_fst_hap_group_0v1"]["p_two_sided"],
        0.00499931182,
        5e-11,
        div_path,
    )

    chimp_path = RESULTS / "figure2a_repolarized" / "figure2a_model_effects.tsv"
    chimp = {row["contrast"]: row for row in read_tsv(chimp_path)}
    for contrast, ratio, p_value in (
        ("derived_vs_ancestral_single_event", 0.2027584654, 0.00260253284),
        ("derived_vs_ancestral_recurrent", 1.012834655, 0.9402941444),
        ("orientation_by_recurrence_interaction", 4.995276784, 0.0038556859),
    ):
        audit.number(f"chimp-polarized {contrast} ratio", chimp[contrast]["ratio"], ratio, 5e-9, chimp_path)
        audit.number(
            f"chimp-polarized {contrast} p",
            chimp[contrast]["p_value_two_sided_wald"],
            p_value,
            5e-10,
            chimp_path,
        )

    pin_path = DATA / "pin_pis_tests.tsv"
    pin = read_tsv(pin_path)
    for category, n, direct, inverted in (
        ("single", 2, 0.2032670766, 0.1284566519),
        ("recurrent", 9, 0.2157557084, 0.1366646670),
    ):
        row = next(
            item
            for item in pin
            if item["metric"] == "piN/piS"
            and item["test"] == "paired Wilcoxon (inverted vs direct)"
            and item["category"] == category
        )
        audit.number(f"piN/piS {category} loci", row["n"], n, 0, pin_path)
        audit.number(f"piN/piS {category} direct median", row["median_direct"], direct, 5e-10, pin_path)
        audit.number(f"piN/piS {category} inverted median", row["median_inverted"], inverted, 5e-10, pin_path)

    paml_path = DATA / "paml_extreme_omega_check.tsv"
    paml = {row["gene"]: row for row in read_tsv(paml_path)}
    for gene, direct, inverted, p_value in (
        ("FDFT1", 0.0, 59.2218, 0.0013448),
        ("BLK", 156.364, 0.0001, 0.0175552),
    ):
        audit.number(f"{gene} omega direct", paml[gene]["omega2_direct"], direct, 5e-5, paml_path)
        audit.number(f"{gene} omega inverted", paml[gene]["omega2_inverted"], inverted, 5e-5, paml_path)
        audit.number(f"{gene} unadjusted p", paml[gene]["overall_p_value"], p_value, 5e-8, paml_path)
    audit.number(
        "PAML genes passing BH q < 0.05",
        sum(float(row["overall_q_value"]) < 0.05 for row in paml.values()),
        0,
        0,
        paml_path,
    )

    architecture_path = DATA / "recurrence_controls_summary.tsv"
    architecture = read_tsv(architecture_path)
    expected_architecture = {
        "Delta-log pi interaction (ratio)": (0.0002, 0.0004, 0.0036),
        "Hudson FST (Recurrent - Single)": (0.0006, 0.0008, 0.0034),
        "da = Dxy - pi_avg (Recurrent - Single)": (0.0042, 0.0014, 0.0034),
    }
    for outcome, expected_values in expected_architecture.items():
        rows = [row for row in architecture if row["outcome"] == outcome]
        if len(rows) != 3:
            raise SystemExit(f"expected three architecture rows for {outcome}")
        for row, expected in zip(rows, expected_values):
            audit.number(
                f"architecture control p: {outcome}; {row['control'].replace(chr(10), ' ')}",
                row["p"],
                expected,
                5e-7,
                architecture_path,
            )

    decay_path = DATA / "decay_spearman_variants.tsv"
    decay = next(
        row
        for row in read_tsv(decay_path)
        if row["within_locus"] == "mean" and row["across_locus"] == "median"
    )
    audit.number("breakpoint decay Spearman rho", decay["rho"], 0.500324, 5e-7, decay_path)
    audit.number("breakpoint decay Spearman p", decay["p_value"], 0.00021564, 5e-9, decay_path)
    audit.number("breakpoint decay bins", decay["n_bins"], 50, 0, decay_path)
    audit.number("breakpoint decay series", decay["n_series"], 60, 0, decay_path)

    imputation_path = DATA / "imputation_threshold_summary.tsv"
    imputation = read_tsv(imputation_path)
    row = next(
        item
        for item in imputation
        if item["subset"] == "consensus_93_locus_set" and float(item["r2_threshold"]) == 0.5
    )
    audit.number("consensus imputation models with r2 > 0.5", row["n_passing_r2"], 12, 0, imputation_path)
    audit.number(
        "consensus imputation models with r2 > 0.5 and BH p < 0.05",
        row["n_passing_r2_and_bh"],
        11,
        0,
        imputation_path,
    )

    hsinv_path = DATA / "imputation_benchmark_HsInv0284_summary.tsv"
    hsinv = next(row for row in read_tsv(hsinv_path) if row["group"] == "ALL")
    audit.number("6q24.1 (HsInv0284) benchmark samples", hsinv["n"], 517, 0, hsinv_path)
    audit.number("6q24.1 (HsInv0284) dosage r2", hsinv["r2"], 0.9428699596, 5e-10, hsinv_path)
    audit.number(
        "6q24.1 (HsInv0284) hard-call concordance",
        hsinv["concordance"],
        0.9961315280,
        5e-10,
        hsinv_path,
    )

    scoreinv_path = DATA / "scoreinvhap_concordance.tsv"
    scoreinv = {row["inversion"]: row for row in read_tsv(scoreinv_path)}
    for inversion, n, r2, concordance in (
        ("8p23.1", 500, 0.7575, 0.758),
        ("17q21.31", 500, 0.9443, 0.976),
    ):
        row = scoreinv[inversion]
        audit.number(f"{inversion} benchmark samples", row["n_samples"], n, 0, scoreinv_path)
        audit.number(f"{inversion} dosage r2", row["r2"], r2, 5e-7, scoreinv_path)
        audit.number(
            f"{inversion} hard-call concordance",
            row["hardcall_concordance"],
            concordance,
            5e-7,
            scoreinv_path,
        )

    cds_path = DATA / "robust_cds_reanalysis_results.tsv"
    cds = {row["hypothesis"] + "|" + row["method"]: row for row in read_tsv(cds_path)}
    primary_key = (
        "Inverted vs Direct within single-event inversions|"
        "Primary: paired inversion-level mean; exact sign-flip t"
    )
    interaction_key = (
        "Single-event minus recurrent orientation effect|"
        "Exact Welch-studentised recurrence-label permutation"
    )
    audit.number("CDS single-event orientation exact p", cds[primary_key]["p_two_sided"], 0.09375, 5e-9, cds_path)
    audit.number("CDS recurrence-by-orientation exact p", cds[interaction_key]["p_two_sided"], 0.41305564, 5e-9, cds_path)

    audit.write()


if __name__ == "__main__":
    main()
