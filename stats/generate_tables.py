#!/usr/bin/env python3
"""Generate the supplementary tables Excel workbook.

This utility orchestrates the steps required to build the manuscript
supplementary tables. It performs the following operations:

1. Curates the inversion catalog from ``data/inv_properties.tsv``.
2. Ensures the CDS conservation test results are produced by running the
   ``stats/per_gene_cds_differences_jackknife.py`` pipeline and filters the
   BH FDR results (q < 0.05).
3. Aggregates the published TSV artefacts into a single Excel workbook with a
   "Read me" worksheet that explains each tab.

The resulting ``supplementary_tables.xlsx`` file is saved under the Next.js
public directory so the web site can link to it directly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
NEXT_PUBLIC_DIR = REPO_ROOT / "web" / "figures-site" / "public"
DEFAULT_OUTPUT = NEXT_PUBLIC_DIR / "downloads" / "supplementary_tables.xlsx"

PUBLIC_BASE_URL = "https://sharedspace.s3.msi.umn.edu/public_internet/"

INV_COLUMNS_KEEP: List[str] = [
    "Chromosome",
    "Start",
    "End",
    "Number_recurrent_events",
    "OrigID",
    "Size_.kbp.",
    "Inverted_AF",
    "verdictRecurrence_hufsah",
    "verdictRecurrence_benson",
    "0_single_1_recur_consensus",
]

INV_RENAME_MAP: Dict[str, str] = {
    "Number_recurrent_events": "number recurrent events",
    "OrigID": "Inversion ID",
    "Size_.kbp.": "Size (kbp)",
    "Inverted_AF": "Inversion allele frequency",
}


INVERSION_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Chromosome", "The chromosome number (GRCh38 reference)."),
        ("Start", "The 1-based start coordinate of the inversion (GRCh38)."),
        ("End", "The 1-based end coordinate of the inversion (GRCh38)."),
        (
            "number recurrent events",
            "The estimated number of independent inversion recurrence events based on coalescent simulations.",
        ),
        ("Inversion ID", "The unique identifier assigned to the inversion (format: chr-start-inv-id)."),
        ("Size (kbp)", "The length of the inverted segment in kilobase pairs."),
        (
            "Inversion allele frequency",
            "The frequency of the inverted allele observed in the phased reference panel (n=88 haplotypes).",
        ),
        ("verdictRecurrence_hufsah", "Recurrence classification based on the Hufsah algorithm."),
        ("verdictRecurrence_benson", "Recurrence classification based on the Benson algorithm."),
        (
            "0_single_1_recur_consensus",
            "Consensus recurrence status used throughout this study: 0 indicates a Single-event inversion (evolved via a single historical mutational event), 1 indicates a Recurrent inversion (evolved via multiple independent events).",
        ),
    ]
)

GENE_CONSERVATION_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Gene", "HGNC gene symbol."),
        ("Transcript", "Ensembl transcript ID used for the CDS analysis."),
        ("Inversion ID", "The identifier of the inversion overlapping this gene."),
        (
            "Orientation more conserved",
            "Indicates which haplotype orientation (Inverted or Direct) has a higher proportion of identical CDS pairs based on the sign of Δ.",
        ),
        (
            "Direct identical pair proportion",
            "The fraction of pairwise comparisons among direct haplotypes that resulted in 100% identical amino acid sequences.",
        ),
        (
            "Inverted identical pair proportion",
            "The fraction of pairwise comparisons among inverted haplotypes that resulted in 100% identical amino acid sequences.",
        ),
        (
            "Δ (inverted − direct)",
            "The difference in identical pair proportions (Inverted minus Direct). Positive values indicate higher conservation in the inverted orientation.",
        ),
        ("SE(Δ)", "Standard error of the difference (Δ), calculated via leave-one-haplotype-out jackknife."),
        ("p-value", "Nominal p-value testing the null hypothesis that conservation is equal between orientations."),
        ("q-value", "Benjamini-Hochberg false discovery rate (FDR) adjusted p-value."),
    ]
)

PHEWAS_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Phenotype", "The phecode string representing the disease phenotype."),
        ("Inversion", "The Inversion ID being tested."),
        ("Q_GLOBAL", "The q-value (FDR) derived from the global heterogeneity test across all phenotypes."),
        ("N_Controls", "Number of control participants."),
        ("OR", "Odds Ratio per unit of inversion dosage."),
        ("CI_LO_OR", "Lower bound of the 95% confidence interval for the Odds Ratio."),
        ("CI_HI_OR", "Upper bound of the 95% confidence interval for the Odds Ratio."),
        ("N_Total", "Total number of participants (Cases + Controls)."),
        ("N_Cases", "Number of case participants."),
        ("P_Value_unadjusted", "Nominal p-value from the logistic regression likelihood ratio test."),
        ("P_Source_x", "The source of the p-value (e.g., LRT, Score)."),
        ("CI_Method", "Method used to calculate confidence intervals (e.g., Profile-Likelihood or Wald)."),
        ("Inference_Type", "Type of statistical inference performed."),
        ("Model_Notes", "Notes regarding model convergence or covariate selection."),
        ("Sig_Global", "Boolean indicating global significance after correction."),
        ("P_LRT_AncestryxDosage", "P-value for the interaction term between dosage and ancestry."),
        ("P_Stage2_Valid", "Boolean indicating if the stage 2 (ancestry-specific) analysis was valid."),
        ("Stage2_P_Source", "Source of the stage 2 p-value."),
        ("Stage2_Inference_Type", "Inference type for stage 2 analysis."),
        ("Stage2_Model_Notes", "Notes regarding the stage 2 model."),
        ("EUR_N", "Total participants (European ancestry)."),
        ("EUR_N_Cases", "Number of cases (European ancestry)."),
        ("EUR_N_Controls", "Number of controls (European ancestry)."),
        ("EUR_OR", "Odds Ratio (European ancestry)."),
        ("EUR_P", "P-value (European ancestry)."),
        ("EUR_P_Source", "P-value source (European ancestry)."),
        ("EUR_Inference_Type", "Inference type (European ancestry)."),
        ("EUR_CI_Method", "CI calculation method (European ancestry)."),
        ("EUR_CI_LO_OR", "Lower 95% CI bound (European ancestry)."),
        ("EUR_CI_HI_OR", "Upper 95% CI bound (European ancestry)."),
        ("AFR_N", "Total participants (African ancestry)."),
        ("AFR_N_Cases", "Number of cases (African ancestry)."),
        ("AFR_N_Controls", "Number of controls (African ancestry)."),
        ("AFR_OR", "Odds Ratio (African ancestry)."),
        ("AFR_P", "P-value (African ancestry)."),
        ("AFR_P_Source", "P-value source (African ancestry)."),
        ("AFR_Inference_Type", "Inference type (African ancestry)."),
        ("AFR_CI_Method", "CI calculation method (African ancestry)."),
        ("AFR_CI_LO_OR", "Lower 95% CI bound (African ancestry)."),
        ("AFR_CI_HI_OR", "Upper 95% CI bound (African ancestry)."),
        ("AMR_N", "Total participants (Admixed American ancestry)."),
        ("AMR_N_Cases", "Number of cases (Admixed American ancestry)."),
        ("AMR_N_Controls", "Number of controls (Admixed American ancestry)."),
        ("AMR_OR", "Odds Ratio (Admixed American ancestry)."),
        ("AMR_P", "P-value (Admixed American ancestry)."),
        ("AMR_P_Source", "P-value source (Admixed American ancestry)."),
        ("AMR_Inference_Type", "Inference type (Admixed American ancestry)."),
        ("AMR_CI_Method", "CI calculation method (Admixed American ancestry)."),
        ("AMR_CI_LO_OR", "Lower 95% CI bound (Admixed American ancestry)."),
        ("AMR_CI_HI_OR", "Upper 95% CI bound (Admixed American ancestry)."),
        ("SAS_N", "Total participants (South Asian ancestry)."),
        ("SAS_N_Cases", "Number of cases (South Asian ancestry)."),
        ("SAS_N_Controls", "Number of controls (South Asian ancestry)."),
        ("SAS_OR", "Odds Ratio (South Asian ancestry)."),
        ("SAS_P", "P-value (South Asian ancestry)."),
        ("SAS_P_Source", "P-value source (South Asian ancestry)."),
        ("SAS_Inference_Type", "Inference type (South Asian ancestry)."),
        ("SAS_CI_Method", "CI calculation method (South Asian ancestry)."),
        ("SAS_CI_LO_OR", "Lower 95% CI bound (South Asian ancestry)."),
        ("SAS_CI_HI_OR", "Upper 95% CI bound (South Asian ancestry)."),
        ("EAS_N", "Total participants (East Asian ancestry)."),
        ("EAS_N_Cases", "Number of cases (East Asian ancestry)."),
        ("EAS_N_Controls", "Number of controls (East Asian ancestry)."),
        ("EAS_OR", "Odds Ratio (East Asian ancestry)."),
        ("EAS_P", "P-value (East Asian ancestry)."),
        ("EAS_P_Source", "P-value source (East Asian ancestry)."),
        ("EAS_Inference_Type", "Inference type (East Asian ancestry)."),
        ("EAS_CI_Method", "CI calculation method (East Asian ancestry)."),
        ("EAS_CI_LO_OR", "Lower 95% CI bound (East Asian ancestry)."),
        ("EAS_CI_HI_OR", "Upper 95% CI bound (East Asian ancestry)."),
        ("MID_N", "Total participants (Middle Eastern ancestry)."),
        ("MID_N_Cases", "Number of cases (Middle Eastern ancestry)."),
        ("MID_N_Controls", "Number of controls (Middle Eastern ancestry)."),
        ("MID_OR", "Odds Ratio (Middle Eastern ancestry)."),
        ("MID_P", "P-value (Middle Eastern ancestry)."),
        ("MID_P_Source", "P-value source (Middle Eastern ancestry)."),
        ("MID_Inference_Type", "Inference type (Middle Eastern ancestry)."),
        ("MID_CI_Method", "CI calculation method (Middle Eastern ancestry)."),
        ("MID_CI_LO_OR", "Lower 95% CI bound (Middle Eastern ancestry)."),
        ("MID_CI_HI_OR", "Upper 95% CI bound (Middle Eastern ancestry)."),
    ]
)

TAG_PHEWAS_COLUMN_DEFS: Dict[str, str] = PHEWAS_COLUMN_DEFS.copy()

CATEGORY_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Inversion", "The Inversion ID."),
        ("Category", "The phecode category being tested."),
        ("Phenotypes in category", "Total number of phenotypes in this category."),
        ("Phenotypes included in GBJ", "Number of phenotypes passing QC that were included in the omnibus test."),
        ("GLS test statistic", "Test statistic for the Generalized Least Squares directional meta-analysis."),
        ("P_GBJ", "P-value for the GBJ omnibus test (testing if any signal exists in the category)."),
        ("Q_GBJ", "FDR q-value for the GBJ test."),
        (
            "Direction",
            "The aggregate direction of effect (Increased Risk or Decreased Risk) if the GLS test is significant.",
        ),
    ]
)

IMPUTATION_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("Inversion", "The Inversion ID."),
        ("n_components", "Number of PLS components selected via cross-validation."),
        (
            "unbiased_pearson_r2",
            "Pearson r² correlation between imputed and true dosage in held-out cross-validation folds.",
        ),
        ("p_value", "P-value comparing the trained model against a null intercept-only model."),
        ("p_fdr_bh", "FDR adjusted p-value."),
        (
            "Use",
            "Boolean flag indicating if the inversion met the quality threshold (r² > 0.3 and q < 0.05) for inclusion in the PheWAS.",
        ),
    ]
)

TRAJECTORY_COLUMN_DEFS: Dict[str, str] = OrderedDict(
    [
        ("date_center", "Estimated time point (Years Before Present)."),
        ("af", "Allele frequency of the reference allele."),
        ("af_low", "Lower bound of the allele frequency confidence interval."),
        ("af_up", "Upper bound of the allele frequency confidence interval."),
        ("selection_coefficient", "Estimated strength of selection acting on the locus."),
    ]
)

GENE_RESULTS_SCRIPT = REPO_ROOT / "stats" / "per_gene_cds_differences_jackknife.py"
GENE_RESULTS_TSV = REPO_ROOT / "gene_inversion_direct_inverted.tsv"
CDS_SUMMARY_TSV = REPO_ROOT / "cds_identical_proportions.tsv"

PHEWAS_RESULTS = DATA_DIR / "phewas_results.tsv"
PHEWAS_TAGGING_RESULTS = DATA_DIR / "all_pop_phewas_tag.tsv"
CATEGORIES_RESULTS_CANDIDATES = (
    DATA_DIR / "categories.tsv",
    DATA_DIR / "phewas v2 - categories.tsv",
)
IMPUTATION_RESULTS = DATA_DIR / "imputation_results.tsv"
INV_PROPERTIES = DATA_DIR / "inv_properties.tsv"
TRAJECTORY_DATA = DATA_DIR / "Trajectory-12_47296118_A_G.tsv"


@dataclass
class SheetInfo:
    name: str
    description: str
    column_defs: Dict[str, str]
    loader: Callable[[], pd.DataFrame]


class SupplementaryTablesError(RuntimeError):
    """Raised for unrecoverable supplementary table failures."""


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response:  # noqa: S310 (trusted host maintained by project)
            data = response.read()
    except HTTPError as exc:  # pragma: no cover - network failure edge case
        raise SupplementaryTablesError(
            f"Unable to download required resource from {url} (HTTP {exc.code})."
        ) from exc
    except URLError as exc:  # pragma: no cover - network failure edge case
        raise SupplementaryTablesError(
            f"Unable to reach {url}: {exc.reason}."
        ) from exc

    destination.write_bytes(data)


def _prune_columns(df: pd.DataFrame, column_defs: Dict[str, str], sheet_name: str) -> pd.DataFrame:
    expected_cols = list(column_defs.keys())
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise SupplementaryTablesError(
            f"Sheet '{sheet_name}' is missing required columns: {', '.join(missing)}"
        )

    return df.loc[:, expected_cols].copy()


def ensure_cds_summary() -> Path:
    """Ensure cds_identical_proportions.tsv exists, generating it if .phy files are available."""
    if CDS_SUMMARY_TSV.exists():
        return CDS_SUMMARY_TSV

    # Check if we have .phy files to run the pipeline
    phy_files = list(REPO_ROOT.glob("*.phy"))
    if len(phy_files) >= 100:  # Arbitrary threshold indicating we have the dataset
        print(f"Found {len(phy_files)} .phy files. Running cds_differences.py to generate summary...")
        try:
            cds_diff_script = REPO_ROOT / "stats" / "cds_differences.py"
            if not cds_diff_script.exists():
                raise SupplementaryTablesError(f"CDS differences script not found: {cds_diff_script}")
            
            # Run cds_differences.py from repo root
            result = subprocess.run(
                [sys.executable, str(cds_diff_script)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"cds_differences.py stderr:\n{result.stderr}", file=sys.stderr)
                raise SupplementaryTablesError(
                    f"cds_differences.py failed with exit code {result.returncode}"
                )
            
            if not CDS_SUMMARY_TSV.exists():
                raise SupplementaryTablesError(
                    "cds_differences.py completed but did not produce cds_identical_proportions.tsv"
                )
            
            print(f"✅ Generated {CDS_SUMMARY_TSV.name}")
            return CDS_SUMMARY_TSV
            
        except subprocess.TimeoutExpired:
            raise SupplementaryTablesError("cds_differences.py timed out after 1 hour")
        except Exception as e:
            print(f"Failed to run cds_differences.py: {e}", file=sys.stderr)
            print("Falling back to downloading pre-computed results...")
    
    # Fallback: download pre-computed results
    url = PUBLIC_BASE_URL + CDS_SUMMARY_TSV.name
    print(f"Downloading CDS summary table from {url} ...")
    _download_file(url, CDS_SUMMARY_TSV)
    return CDS_SUMMARY_TSV


def ensure_gene_results() -> Path:
    """Ensure gene_inversion_direct_inverted.tsv exists, generating it if CDS summary is available."""
    if GENE_RESULTS_TSV.exists():
        return GENE_RESULTS_TSV

    # First ensure we have the CDS summary
    cds_summary = ensure_cds_summary()
    
    # Check if we have pairs files to run the per-gene analysis
    pairs_files = list(REPO_ROOT.glob("pairs_CDS__*.tsv"))
    if len(pairs_files) >= 100:  # Threshold indicating we have the dataset
        print(f"Found {len(pairs_files)} pairs files. Running per_gene_cds_differences_jackknife.py...")
        try:
            gene_script = REPO_ROOT / "stats" / "per_gene_cds_differences_jackknife.py"
            if not gene_script.exists():
                raise SupplementaryTablesError(f"Per-gene script not found: {gene_script}")
            
            # Run per_gene_cds_differences_jackknife.py from repo root
            result = subprocess.run(
                [sys.executable, str(gene_script)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"per_gene_cds_differences_jackknife.py stderr:\n{result.stderr}", file=sys.stderr)
                raise SupplementaryTablesError(
                    f"per_gene_cds_differences_jackknife.py failed with exit code {result.returncode}"
                )
            
            if not GENE_RESULTS_TSV.exists():
                raise SupplementaryTablesError(
                    "per_gene_cds_differences_jackknife.py completed but did not produce gene_inversion_direct_inverted.tsv"
                )
            
            print(f"✅ Generated {GENE_RESULTS_TSV.name}")
            return GENE_RESULTS_TSV
            
        except subprocess.TimeoutExpired:
            raise SupplementaryTablesError("per_gene_cds_differences_jackknife.py timed out after 1 hour")
        except Exception as e:
            print(f"Failed to run per_gene_cds_differences_jackknife.py: {e}", file=sys.stderr)
            print("Falling back to downloading pre-computed results...")
    
    # Fallback: download pre-computed results
    url = PUBLIC_BASE_URL + GENE_RESULTS_TSV.name
    print(f"Downloading gene-level CDS results from {url} ...")
    _download_file(url, GENE_RESULTS_TSV)
    return GENE_RESULTS_TSV


def _load_inversion_catalog() -> pd.DataFrame:
    if not INV_PROPERTIES.exists():
        raise SupplementaryTablesError(f"Inversion properties TSV not found: {INV_PROPERTIES}")

    df = pd.read_csv(INV_PROPERTIES, sep="\t", dtype=str, low_memory=False)
    keepable = [c for c in df.columns if str(c).strip()]
    df = df.loc[:, keepable]

    missing = [col for col in INV_COLUMNS_KEEP if col not in df.columns]
    if missing:
        raise SupplementaryTablesError(
            "Inversion properties TSV is missing required columns: " + ", ".join(missing)
        )

    df = df[INV_COLUMNS_KEEP].copy()
    df = df.rename(columns=INV_RENAME_MAP)
    return _prune_columns(df, INVERSION_COLUMN_DEFS, "Inversion catalog")


def _load_gene_conservation() -> pd.DataFrame:
    tsv_path = ensure_gene_results()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    numeric_cols = ["p_direct", "p_inverted", "delta", "se_delta", "p_value", "q_value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def orientation(row: pd.Series) -> str:
        delta = row.get("delta")
        if pd.isna(delta):
            return "Unknown"
        if delta > 0:
            return "Inverted"
        if delta < 0:
            return "Direct"
        return "Tie"

    df["Orientation more conserved"] = df.apply(orientation, axis=1)

    rename_map = {
        "gene_name": "Gene",
        "transcript_id": "Transcript",
        "inv_id": "Inversion ID",
        "p_direct": "Direct identical pair proportion",
        "p_inverted": "Inverted identical pair proportion",
        "delta": "Δ (inverted − direct)",
        "se_delta": "SE(Δ)",
        "p_value": "p-value",
        "q_value": "q-value",
    }

    df = df.rename(columns=rename_map)
    df = _prune_columns(df, GENE_CONSERVATION_COLUMN_DEFS, "CDS conservation genes")
    df = df.sort_values("q-value", kind="mergesort").reset_index(drop=True)
    return df


def _load_simple_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SupplementaryTablesError(f"Required TSV not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def _clean_phewas_df(
    df: pd.DataFrame, sheet_name: str, column_defs: Dict[str, str]
) -> pd.DataFrame:
    # Check for P_Value_x and P_Value_y columns
    if "P_Value_x" in df.columns and "P_Value_y" in df.columns:
        # Convert to numeric for comparison
        p_x = pd.to_numeric(df["P_Value_x"], errors="coerce")
        p_y = pd.to_numeric(df["P_Value_y"], errors="coerce")

        both_nan = p_x.isna() & p_y.isna()
        both_equal = p_x == p_y
        all_match = (both_nan | both_equal).all()

        if not all_match:
            diff_mask = ~(both_nan | both_equal)
            first_diff_idx = diff_mask.idxmax() if diff_mask.any() else None
            raise SupplementaryTablesError(
                f"P_Value_x and P_Value_y columns have different values. "
                f"First difference at row {first_diff_idx}: "
                f"P_Value_x={df.loc[first_diff_idx, 'P_Value_x']}, "
                f"P_Value_y={df.loc[first_diff_idx, 'P_Value_y']}"
            )

        df = df.drop(columns=["P_Value_y"])
        df = df.rename(columns={"P_Value_x": "P_Value_unadjusted"})

    if "P_Value_unadjusted" not in df.columns and "P_Value" in df.columns:
        df = df.rename(columns={"P_Value": "P_Value_unadjusted"})

    empty_cols = [
        col for col in df.columns if df[col].isna().all() or (df[col].astype(str).str.strip() == "").all()
    ]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    return _prune_columns(df, column_defs, sheet_name)


def _load_phewas_results() -> pd.DataFrame:
    df = _load_simple_tsv(PHEWAS_RESULTS)
    return _clean_phewas_df(df, "PheWAS results", PHEWAS_COLUMN_DEFS)


def _load_categories() -> pd.DataFrame:
    for candidate in CATEGORIES_RESULTS_CANDIDATES:
        if candidate.exists():
            df = _load_simple_tsv(candidate)
            # Remove Z_Cap and Dropped columns if present
            columns_to_drop = ["Z_Cap", "Dropped"]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            # Rename columns for clarity
            rename_map = {
                "K_Total": "Phenotypes in category",
                "K_GBJ": "Phenotypes included in GBJ",
                "T_GLS": "GLS test statistic",
            }
            df = df.rename(columns=rename_map)

            return _prune_columns(df, CATEGORY_COLUMN_DEFS, "Phenotype categories")
    raise SupplementaryTablesError("Unable to locate categories TSV in the data directory.")


def _load_phewas_tagging() -> pd.DataFrame:
    if PHEWAS_TAGGING_RESULTS.exists():
        df = _load_simple_tsv(PHEWAS_TAGGING_RESULTS)
        return _clean_phewas_df(df, "17q21 tagging PheWAS", TAG_PHEWAS_COLUMN_DEFS)

    url = PUBLIC_BASE_URL + PHEWAS_TAGGING_RESULTS.name
    print(f"Attempting to download PheWAS tagging results from {url} ...")
    try:
        _download_file(url, PHEWAS_TAGGING_RESULTS)
    except SupplementaryTablesError as exc:
        raise SupplementaryTablesError(
            "PheWAS tagging results were not found locally and could not be downloaded."
        ) from exc

    return _clean_phewas_df(
        _load_simple_tsv(PHEWAS_TAGGING_RESULTS), "17q21 tagging PheWAS", TAG_PHEWAS_COLUMN_DEFS
    )


def _load_imputation_results() -> pd.DataFrame:
    df = _load_simple_tsv(IMPUTATION_RESULTS)
    # Remove unnamed columns (Column 6 and Column 9)
    columns_to_drop = ["Column 6", "Column 9"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return _prune_columns(df, IMPUTATION_COLUMN_DEFS, "Imputation results")


def _load_trajectory_data() -> pd.DataFrame:
    if not TRAJECTORY_DATA.exists():
        raise SupplementaryTablesError(f"Trajectory data not found: {TRAJECTORY_DATA}")
    df = pd.read_csv(TRAJECTORY_DATA, sep="\t", dtype=str, low_memory=False)
    return _prune_columns(df, TRAJECTORY_COLUMN_DEFS, "AGES trajectory 12_47296118")


def build_workbook(output_path: Path) -> None:
    sheet_infos: List[SheetInfo] = []
    sheet_frames: List[pd.DataFrame] = []

    def register(sheet: SheetInfo) -> None:
        sheet_infos.append(sheet)
        print(f"Preparing sheet: {sheet.name}")
        df = sheet.loader()
        sheet_frames.append(df)

    register(
        SheetInfo(
            name="Inversion catalog",
            description=(
                "A comprehensive catalog of the 93 balanced human chromosomal inversions analyzed in this study. "
                "Inversion calls, coordinates, and recurrence classifications are derived from Porubsky et al. (2022) "
                "using Strand-seq and long-read sequencing on the 1000 Genomes Project panel (GRCh38 coordinates)."
            ),
            column_defs=INVERSION_COLUMN_DEFS,
            loader=_load_inversion_catalog,
        )
    )

    register(
        SheetInfo(
            name="CDS conservation genes",
            description=(
                "Analysis of protein-coding gene conservation within inversion loci. Tests quantify differences in the "
                "proportion of identical Coding Sequence (CDS) pairs between inverted and direct haplotypes, identifying genes "
                "where the inverted orientation maintains significantly higher (or lower) sequence conservation."
            ),
            column_defs=GENE_CONSERVATION_COLUMN_DEFS,
            loader=_load_gene_conservation,
        )
    )

    register(
        SheetInfo(
            name="PheWAS results",
            description=(
                "Phenome-wide association study (PheWAS) results linking imputed inversion dosages to electronic health record "
                "(EHR) phenotypes in the NIH All of Us cohort (v8). Association tests were performed using logistic regression "
                "adjusted for age, sex, 16 genetic principal components, and ancestry categories."
            ),
            column_defs=PHEWAS_COLUMN_DEFS,
            loader=_load_phewas_results,
        )
    )

    register(
        SheetInfo(
            name="17q21 tagging PheWAS",
            description=(
                "Validation PheWAS for the 17q21 inversion locus using a tagging SNP (rs105255341) instead of imputed dosage. "
                "This ensures that the pleiotropic effects observed (e.g., obesity vs. breast cancer protection) are robust to "
                "the method of genotype determination."
            ),
            column_defs=TAG_PHEWAS_COLUMN_DEFS,
            loader=_load_phewas_tagging,
        )
    )

    register(
        SheetInfo(
            name="Phenotype categories",
            description=(
                "Aggregate statistical tests assessing whether specific inversions are associated with entire categories of "
                "phenotypes (e.g., 'Dermatologic'). Uses the Generalized Berk-Jones (GBJ) test for set-based significance and "
                "Generalized Least Squares (GLS) for directional effects."
            ),
            column_defs=CATEGORY_COLUMN_DEFS,
            loader=_load_categories,
        )
    )

    register(
        SheetInfo(
            name="Imputation results",
            description=(
                "Performance metrics for the machine learning models (Partial Least Squares regression) used to impute inversion "
                "dosage from flanking SNP genotypes. Models were trained on the 88 phased haplotypes from the reference panel."
            ),
            column_defs=IMPUTATION_COLUMN_DEFS,
            loader=_load_imputation_results,
        )
    )

    register(
        SheetInfo(
            name="AGES trajectory 12_47296118",
            description=(
                "Allele frequency trajectory data for SNP 12_47296118_A_G (tagging the 12q13 inversion) derived from the "
                "Ancient Genome Edge Selection (AGES) database. Represents frequency changes over the last ~14,000 years in "
                "West Eurasia."
            ),
            column_defs=TRAJECTORY_COLUMN_DEFS,
            loader=_load_trajectory_data,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        readme_ws = workbook.add_worksheet("Read me")

        header_fmt = workbook.add_format({"bold": True, "font_size": 14, "bottom": 1})
        desc_fmt = workbook.add_format({"italic": True, "text_wrap": True})
        col_name_fmt = workbook.add_format({"bold": True, "text_wrap": True, "bg_color": "#EEEEEE"})
        col_def_fmt = workbook.add_format({"text_wrap": True})

        readme_ws.set_column(0, 0, 32)
        readme_ws.set_column(1, 1, 120)

        row = 0
        for sheet_info in sheet_infos:
            readme_ws.write(row, 0, sheet_info.name, header_fmt)
            row += 1

            readme_ws.merge_range(row, 0, row, 1, sheet_info.description, desc_fmt)
            row += 1

            readme_ws.write(row, 0, "Column", col_name_fmt)
            readme_ws.write(row, 1, "Definition", col_name_fmt)
            row += 1

            for col_name, definition in sheet_info.column_defs.items():
                readme_ws.write(row, 0, col_name, col_name_fmt)
                readme_ws.write(row, 1, definition, col_def_fmt)
                row += 1

            row += 2

        for sheet_info, df in zip(sheet_infos, sheet_frames):
            df.to_excel(writer, index=False, sheet_name=sheet_info.name)

    print(f"Supplementary tables written to {output_path}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supplementary tables workbook.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the Excel workbook (default: web/figures-site/public/downloads).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    try:
        build_workbook(args.output.resolve())
    except SupplementaryTablesError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive guardrail
        print(f"ERROR: Unexpected failure while generating tables: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
