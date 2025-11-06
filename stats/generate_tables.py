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
from datetime import datetime, timezone
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


@dataclass
class SheetInfo:
    name: str
    description: str
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
    return df


def _load_gene_conservation() -> pd.DataFrame:
    tsv_path = ensure_gene_results()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    numeric_cols = ["p_direct", "p_inverted", "delta", "se_delta", "p_value", "q_value"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["q_value"].notna() & (df["q_value"] < 0.05)].copy()

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

    ordered_cols = [
        "Gene",
        "Transcript",
        "Inversion ID",
        "Orientation more conserved",
        "Direct identical pair proportion",
        "Inverted identical pair proportion",
        "Δ (inverted − direct)",
        "SE(Δ)",
        "p-value",
        "q-value",
    ]

    df = df.rename(columns=rename_map)
    df = df.reindex(columns=ordered_cols)
    df = df.sort_values("q-value", kind="mergesort").reset_index(drop=True)
    return df


def _load_simple_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SupplementaryTablesError(f"Required TSV not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def _load_phewas_results() -> pd.DataFrame:
    df = _load_simple_tsv(PHEWAS_RESULTS)
    
    # Check for P_Value_x and P_Value_y columns
    if "P_Value_x" in df.columns and "P_Value_y" in df.columns:
        # Convert to numeric for comparison
        p_x = pd.to_numeric(df["P_Value_x"], errors="coerce")
        p_y = pd.to_numeric(df["P_Value_y"], errors="coerce")
        
        # Check if values are the same (accounting for NaN)
        # Two NaNs are considered equal, but any difference in non-NaN values is an error
        both_nan = p_x.isna() & p_y.isna()
        both_equal = (p_x == p_y)
        all_match = (both_nan | both_equal).all()
        
        if not all_match:
            # Find first differing row for error message
            diff_mask = ~(both_nan | both_equal)
            first_diff_idx = diff_mask.idxmax() if diff_mask.any() else None
            raise SupplementaryTablesError(
                f"P_Value_x and P_Value_y columns have different values. "
                f"First difference at row {first_diff_idx}: "
                f"P_Value_x={df.loc[first_diff_idx, 'P_Value_x']}, "
                f"P_Value_y={df.loc[first_diff_idx, 'P_Value_y']}"
            )
        
        # Values match - drop P_Value_y and rename P_Value_x
        df = df.drop(columns=["P_Value_y"])
        df = df.rename(columns={"P_Value_x": "P_Value_unadjusted"})
    
    # Remove columns that have no data (all values are empty/null)
    empty_cols = [col for col in df.columns if df[col].isna().all() or (df[col].astype(str).str.strip() == "").all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    
    return df


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
                "K_GLS": "Phenotypes included in GLS",
                "T_GLS": "GLS test statistic",
            }
            df = df.rename(columns=rename_map)
            
            return df
    raise SupplementaryTablesError("Unable to locate categories TSV in the data directory.")


def _load_phewas_tagging() -> pd.DataFrame:
    if PHEWAS_TAGGING_RESULTS.exists():
        return _load_simple_tsv(PHEWAS_TAGGING_RESULTS)

    url = PUBLIC_BASE_URL + PHEWAS_TAGGING_RESULTS.name
    print(f"Attempting to download PheWAS tagging results from {url} ...")
    try:
        _download_file(url, PHEWAS_TAGGING_RESULTS)
    except SupplementaryTablesError as exc:
        raise SupplementaryTablesError(
            "PheWAS tagging results were not found locally and could not be downloaded."
        ) from exc

    return _load_simple_tsv(PHEWAS_TAGGING_RESULTS)


def _load_imputation_results() -> pd.DataFrame:
    df = _load_simple_tsv(IMPUTATION_RESULTS)
    # Remove unnamed columns (Column 6 and Column 9)
    columns_to_drop = ["Column 6", "Column 9"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df


def build_workbook(output_path: Path) -> None:
    sheets = OrderedDict()
    readme_rows: List[Dict[str, str]] = []

    generated_at = datetime.now(timezone.utc).isoformat()
    readme_rows.append(
        {
            "Tab": "Generated at",
            "Description": generated_at,
            "Status / Notes": "Timestamps are in UTC.",
        }
    )

    def register(sheet: SheetInfo) -> None:
        print(f"Preparing sheet: {sheet.name}")
        status = "Available"
        try:
            df = sheet.loader()
        except Exception as exc:  # pragma: no cover - defensive logging
            status = f"Error: {exc}"
            df = pd.DataFrame({"error": [str(exc)]})
        sheets[sheet.name] = df
        readme_rows.append(
            {
                "Tab": sheet.name,
                "Description": sheet.description,
                "Status / Notes": status,
            }
        )

    register(
        SheetInfo(
            name="Inversion catalog",
            description=(
                "The catalog of 93 balanced human inversions used throughout the study. "
                "Coordinates, recurrence calls, and consensus annotations are derived "
                "from data/inv_properties.tsv."
            ),
            loader=_load_inversion_catalog,
        )
    )

    register(
        SheetInfo(
            name="CDS conservation genes",
            description=(
                "Protein-coding genes with significant CDS conservation differences "
                "between direct and inverted haplotypes (BH q < 0.05). Generated via "
                "stats/per_gene_cds_differences_jackknife.py."
            ),
            loader=_load_gene_conservation,
        )
    )

    register(
        SheetInfo(
            name="PheWAS results",
            description="Full genome-wide PheWAS association statistics (data/phewas_results.tsv).",
            loader=_load_phewas_results,
        )
    )

    register(
        SheetInfo(
            name="17q21 tagging PheWAS",
            description=(
                "PheWAS results for tagging variants linked to the 17q21 inversion "
                "(data/all_pop_phewas_tag.tsv)."
            ),
            loader=_load_phewas_tagging,
        )
    )

    register(
        SheetInfo(
            name="Phenotype categories",
            description="Category-level PheWAS summaries (data/categories.tsv).",
            loader=_load_categories,
        )
    )

    register(
        SheetInfo(
            name="Imputation results",
            description="Dosage imputation performance metrics (data/imputation_results.tsv).",
            loader=_load_imputation_results,
        )
    )

    readme_df = pd.DataFrame(readme_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        readme_df.to_excel(writer, index=False, sheet_name="Read me")
        for sheet_name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)

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
