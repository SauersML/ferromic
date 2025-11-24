"""Replicate manuscript metrics and tests.
"""
from __future__ import annotations

import math
import sys
import os
import re
from contextlib import contextmanager
import gzip
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Tuple
import tempfile

import shutil
import numpy as np
import pandas as pd
import requests
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats import (
    CDS_identical_model,
    fst_edge_decay,
    inv_dir_recur_model,
    recur_breakpoint_tests,
    per_inversion_breakpoint_metric,
    cds_differences,
    per_gene_cds_differences_jackknife,
)  # noqa: E402
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


def _fmt_pvalue(value: float | int | None) -> str:
    """Render p-values without aggressive rounding.

    Uses up to 15 significant digits so extremely small probabilities stay
    visible instead of collapsing to 0.000. Falls back to scientific notation
    automatically for tiny values.
    """

    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(val) or math.isinf(val):
        return "NA"
    if val == 0.0:
        return "0"
    return f"{val:.15g}"


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


def download_latest_artifacts():
    """
    Automatically downloads the required artifacts from the latest successful
    'Manual Run VCF Pipeline' (manual_run_vcf.yml) execution.
    """
    print("\n" + "=" * 80)
    print(">>> ARTIFACT RETRIEVAL: FETCHING LATEST DATA FROM GITHUB ACTIONS <<<")
    print("=" * 80)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print("WARNING: GITHUB_TOKEN or GITHUB_REPOSITORY not set. Skipping auto-download.")
        print("Ensure you have manually placed the required files in data/ if running locally.")
        return

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    api_root = "https://api.github.com"
    workflow_file = "manual_run_vcf.yml"

    # 1. Find latest successful run
    print(f"Finding latest successful run of {workflow_file}...")
    try:
        url = f"{api_root}/repos/{repo}/actions/workflows/{workflow_file}/runs"
        params = {"status": "success", "per_page": 1, "exclude_pull_requests": "true"}
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        runs = resp.json().get("workflow_runs", [])
        if not runs:
            print("No successful runs found. Skipping download.")
            return
        run_id = runs[0]["id"]
        print(f"Found Run ID: {run_id}")
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return

    # 2. List artifacts
    print("Listing artifacts...")
    try:
        url = f"{api_root}/repos/{repo}/actions/runs/{run_id}/artifacts"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        artifacts = resp.json().get("artifacts", [])
    except Exception as e:
        print(f"Error fetching artifacts: {e}")
        return

    # Define mapping: Artifact Name -> (Target Filename in data/, Unzip Logic)
    # Logic options:
    #   - 'copy_inner_zip': copy nested zip archive as-is
    #   - 'extract_file': extract specific file unchanged
    #   - 'extract_renamed': extract and rename
    #   - 'extract_and_gunzip': extract gzipped file and store decompressed contents
    # Since GHA artifacts are ALWAYS zip files, we download the zip and then process.
    artifact_map = {
        "run-vcf-phy-outputs": {"target": "phy_outputs.zip", "action": "copy_inner_zip"},
        "run-vcf-falsta": {"target": "per_site_diversity_output.falsta.gz", "action": "extract_file"},
        "run-vcf-hudson-fst": {"target": "FST_data.tsv", "action": "extract_and_gunzip"},
        # IMPORTANT: Do NOT download run-vcf-metadata to inv_properties.tsv.
        # run-vcf-metadata contains phy_metadata.tsv, which is different from inv_properties.tsv.
        "run-vcf-metadata": {"target": "phy_metadata.tsv", "action": "extract_renamed"},
        "run-vcf-output-csv": {"target": "output.csv", "action": "extract_file"},
    }

    # Specific internal filenames expected inside the artifacts
    internal_names = {
        "run-vcf-falsta": "per_site_diversity_output.falsta.gz",
        "run-vcf-hudson-fst": "hudson_fst_results.tsv.gz",
        "run-vcf-metadata": "phy_metadata.tsv",
        "run-vcf-output-csv": "output.csv",
        "run-vcf-phy-outputs": "phy_outputs.zip"
    }

    # Ensure DATA_DIR is defined and exists (using global variable defined at module level)
    DATA_DIR.mkdir(exist_ok=True)

    for artifact in artifacts:
        name = artifact["name"]
        if name not in artifact_map:
            continue

        spec = artifact_map[name]
        target_path = DATA_DIR / spec["target"]
        download_url = artifact["archive_download_url"]

        print(f"Downloading {name} -> {target_path.name}...")
        try:
            # Stream download to a temporary file to avoid memory issues with large artifacts
            with tempfile.TemporaryFile() as tmp_file:
                with requests.get(download_url, headers=headers, stream=True) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)

                tmp_file.seek(0)
                with zipfile.ZipFile(tmp_file) as z:
                    # Perform action
                    if spec["action"] == "copy_inner_zip":
                        # The artifact contains a zip file (e.g. phy_outputs.zip)
                        # We extract that inner zip to data/
                        inner_name = internal_names[name]
                        with z.open(inner_name) as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)

                    elif spec["action"] == "extract_file":
                        # Extract specific file as is
                        inner_name = internal_names[name]
                        with z.open(inner_name) as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)

                    elif spec["action"] == "extract_renamed":
                        # Extract file but rename it (e.g. phy_metadata.tsv -> inv_properties.tsv)
                        inner_name = internal_names[name]
                        with z.open(inner_name) as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)

                    elif spec["action"] == "extract_and_gunzip":
                        # Extract gzipped file, decompress it, and save the decompressed payload
                        inner_name = internal_names[name]
                        with z.open(inner_name) as src:
                            with gzip.open(src) as gz_src:
                                data = gz_src.read()
                        target_path.write_bytes(data)

            print(f"  Success: {target_path.name} updated.")

        except Exception as e:
            print(f"  FAILED to process {name}: {e}")
            # We don't exit here, try to get other files


def _stage_cds_inputs() -> list[Path]:
    """Prepare required inputs for cds_differences in the working directory."""

    staged_paths: list[Path] = []

    metadata_src = DATA_DIR / "inv_properties.tsv"
    if not metadata_src.exists():
        raise FileNotFoundError(
            "Missing metadata: expected data/inv_properties.tsv to stage inv_properties.tsv"
        )

    metadata_dest = Path("inv_properties.tsv")
    shutil.copy2(metadata_src, metadata_dest)
    staged_paths.append(metadata_dest)

    zip_archives = sorted(DATA_DIR.glob("*.zip"))
    if not zip_archives:
        raise FileNotFoundError("No .zip archives found in data/ for PHYLIP extraction")

    extracted_any = False
    for archive_path in zip_archives:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".phy") or name.endswith(".phy.gz")
                ]
                if not members:
                    continue
                extracted_any = True
                for member in members:
                    target_name = Path(member).name
                    if target_name.endswith(".gz"):
                        target_name = target_name[:-3]
                    target_path = Path(target_name)
                    with archive.open(member) as zipped_member:
                        if member.endswith(".gz"):
                            with gzip.open(zipped_member) as gz_member:
                                data = gz_member.read()
                        else:
                            data = zipped_member.read()
                    target_path.write_bytes(data)
                    staged_paths.append(target_path)
        except zipfile.BadZipFile:
            print(f"WARNING: '{archive_path.name}' is not a valid zip file. Skipping.")
            continue

    if not extracted_any:
        raise FileNotFoundError(
            "No .phy or .phy.gz files found inside data/*.zip archives"
        )

    return staged_paths


def run_fresh_cds_pipeline():
    """
    Force regeneration of CDS statistics from raw .phy files.
    """

    # Always start from a clean slate for CDS summary outputs
    for filename in [
        "cds_identical_proportions.tsv",
        "gene_inversion_direct_inverted.tsv",
        "region_identical_proportions.tsv",
        "skipped_details.tsv",
    ]:
        target = DATA_DIR / filename
        if target.exists():
            target.unlink()

    print("\n" + "=" * 80)
    print(">>> PIPELINE: REGENERATING CDS DATA FROM RAW .PHY FILES <<<")
    print("=" * 80)

    with _temporary_workdir(REPO_ROOT):
        # 1. Clean up old intermediate files to ensure we are using raw data
        print("... Cleaning old summary tables to ensure fresh run ...")
        for f in Path(".").glob("cds_identical_proportions.tsv"):
            f.unlink()
        for f in Path(".").glob("pairs_CDS__*.tsv"):
            f.unlink()
        for f in Path(".").glob("gene_inversion_direct_inverted.tsv"):
            f.unlink()

        staged_paths: list[Path] = []
        try:
            # 2. Stage required inputs for cds_differences.py
            print("... Staging metadata and PHYLIP archives from data/ ...")
            staged_paths = _stage_cds_inputs()

            # 3. Run the Raw Processor (equivalent to running stats/cds_differences.py)
            print("\n[Step 1/2] Parsing raw PHYLIP files (cds_differences.py)...")
            try:
                cds_differences.main()
            except Exception as e:
                print(f"FATAL: Raw .phy processing failed: {e}")
                sys.exit(1)

            # 4. Run the Jackknife Analysis (equivalent to stats/per_gene_cds_differences_jackknife.py)
            print("\n[Step 2/2] Running Jackknife statistics (per_gene_cds_differences_jackknife.py)...")
            try:
                per_gene_cds_differences_jackknife.main()
            except Exception as e:
                print(f"FATAL: Jackknife analysis failed: {e}")
                sys.exit(1)

            print("... Copying generated TSV files to data/ ...")
            for filename in [
                "cds_identical_proportions.tsv",
                "gene_inversion_direct_inverted.tsv",
                "region_identical_proportions.tsv",
                "skipped_details.tsv",
            ]:
                src = Path(filename)
                if src.exists():
                    shutil.copy2(src, DATA_DIR / filename)
                    print(f"  Copied {filename} to data/")
                else:
                    print(f"  WARNING: {filename} not found, skipping copy.")

            print("\n>>> PIPELINE: GENERATION COMPLETE. Proceeding to manuscript report...\n")

        except Exception as e:
            print(f"FATAL: CDS generation pipeline failed: {e}")
            sys.exit(1)

        finally:
            if staged_paths:
                print("... Cleaning staged metadata and PHYLIP files ...")
                for path in staged_paths:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass


# ---------------------------------------------------------------------------
# π structure helpers
# ---------------------------------------------------------------------------


@dataclass
class SpearmanResult:
    rho: float | None
    p_value: float | None
    n: int


@dataclass
class SpearmanPoint:
    rho: float | None
    p_value: float | None
    recurrence_flag: int
    group: int
    region: tuple[str, int, int] | None = None
    q_value: float | None = None
    bins_used: int = 0


@dataclass
class EdgeMiddleStats:
    flank_mean: float | None
    middle_mean: float | None
    entries: int
    total_haplotypes: int
    mean_haplotypes: float | None
    median_haplotypes: float | None
    min_haplotypes: int | None
    max_haplotypes: int | None


@dataclass
class PiStructureMetrics:
    # Edge vs Middle Metrics (≥40kb)
    # Direct (Group 0)
    dir_stats: EdgeMiddleStats

    # Inverted (Group 1)
    inv_stats: EdgeMiddleStats

    # Overall (Group 0 + 1)
    all_stats: EdgeMiddleStats

    # Edge/Middle subgroup breakdown (group, recurrence)
    subgroup_edge_middle: dict[tuple[int, int], EdgeMiddleStats]

    # Spearman Decay Metrics (≥100kb, first 100kb)
    spearman_overall: SpearmanResult
    spearman_single_inv: SpearmanResult  # Group 1, Recur 0
    spearman_recur_dir: SpearmanResult   # Group 0, Recur 1
    spearman_recur_inv: SpearmanResult   # Group 1, Recur 1
    spearman_single_dir: SpearmanResult  # Group 0, Recur 0

    # Spearman Decay Metrics (Median within 2kb bins)
    spearman_overall_median: SpearmanResult
    spearman_single_inv_median: SpearmanResult
    spearman_recur_dir_median: SpearmanResult
    spearman_recur_inv_median: SpearmanResult
    spearman_single_dir_median: SpearmanResult

    spearman_points: list[SpearmanPoint]

    unique_inversions: int


class _MetricAccumulator:
    """Accumulates Pi data for stats."""
    def __init__(self):
        self.flank_means: list[float] = []
        self.middle_means: list[float] = []
        self.haplotype_counts: list[int] = []

    def add_edge_middle(self, values: np.ndarray, hap_count: int | None = None) -> None:
        # Expects values length >= 40,000 checked by caller
        flanks = np.r_[values[:10_000], values[-10_000:]]
        flank_mean = float(np.nanmean(flanks))
        if np.isfinite(flank_mean):
            self.flank_means.append(flank_mean)

        middle_start = max((values.size - 20_000) // 2, 0)
        middle_slice = values[middle_start : middle_start + 20_000]
        if middle_slice.size == 20_000:
            middle_mean = float(np.nanmean(middle_slice))
            if np.isfinite(middle_mean):
                self.middle_means.append(middle_mean)

        if hap_count is not None:
            self.haplotype_counts.append(int(hap_count))


def _calc_spearman(
    window_data: list[np.ndarray],
    agg_func: Callable[[np.ndarray, int], np.ndarray] = np.nanmedian,
    min_inv_per_bin: int = 5,
) -> SpearmanResult:
    """Calculate Spearman rho using aggregated 2kb windows across inversions.

    The manuscript aggregates diversity values per 2kb bin across inversions and
    applies a minimum inversion-per-bin threshold before computing the
    correlation. This mirrors the visualization logic (50 bins spanning 0–100kb).
    """

    if not window_data:
        return SpearmanResult(rho=None, p_value=None, n=0)

    window_matrix = np.vstack(window_data)
    base_distances = np.arange(0, 100_000, 2_000, dtype=float)

    # Aggregate per-bin values across inversions, masking bins with sparse data.
    per_bin_counts = np.sum(np.isfinite(window_matrix), axis=0)
    aggregated = agg_func(window_matrix, axis=0)
    aggregated = np.where(per_bin_counts >= min_inv_per_bin, aggregated, np.nan)

    mask = np.isfinite(aggregated)
    if mask.sum() < 2:
        return SpearmanResult(rho=None, p_value=None, n=len(window_data))

    rho_val, p_val = stats.spearmanr(base_distances[mask], aggregated[mask])

    rho = float(rho_val) if np.isfinite(rho_val) else None
    p = float(p_val) if np.isfinite(p_val) else None

    return SpearmanResult(rho=rho, p_value=p, n=len(window_data))


def _distance_fold_binning(
    values: np.ndarray, *, bin_size: int = 2_000, max_distance: int = 100_000
) -> tuple[np.ndarray, np.ndarray]:
    """Fold values around the inversion midpoint and bin by distance to edge.

    Distances are computed as ``min(idx, len(values) - 1 - idx)`` so that both
    edges map to 0 and bins beyond half the inversion length remain empty. The
    function returns per-bin means and medians over *sites*, leaving bins with
    no coverage as NaN.
    """

    if values.size == 0:
        bins = max_distance // bin_size
        return np.full(bins, np.nan), np.full(bins, np.nan)

    n_bins = max_distance // bin_size
    folded_distances = np.minimum(
        np.arange(values.size, dtype=int), values.size - 1 - np.arange(values.size, dtype=int)
    )

    finite_mask = np.isfinite(values)
    usable_mask = finite_mask & (folded_distances < max_distance)
    if not np.any(usable_mask):
        return np.full(n_bins, np.nan), np.full(n_bins, np.nan)

    dist_subset = folded_distances[usable_mask]
    val_subset = values[usable_mask]
    bin_indices = (dist_subset // bin_size).astype(int)

    bin_means = np.full(n_bins, np.nan, dtype=float)
    bin_medians = np.full(n_bins, np.nan, dtype=float)

    for idx in range(n_bins):
        mask = bin_indices == idx
        if not np.any(mask):
            continue
        bin_vals = val_subset[mask]
        bin_means[idx] = np.nanmean(bin_vals)
        bin_medians[idx] = np.nanmedian(bin_vals)

    return bin_means, bin_medians


def _calc_pi_structure_metrics() -> PiStructureMetrics:
    """Parse per-site diversity tracks to replicate π structure metrics.

    Filters for consensus inversions (0/1) and computes stats for Direct, Inverted, and Overall.
    """

    falsta_candidates = [
        DATA_DIR / "per_site_diversity_output.falsta",
        DATA_DIR / "per_site_diversity_output.falsta.gz",
    ]
    falsta_path = next((path for path in falsta_candidates if path.exists()), None)
    if falsta_path is None:
        raise FileNotFoundError(
            "Missing per-site diversity FALSTA: per_site_diversity_output.falsta(.gz)"
        )

    # Load inversion whitelist, recurrence mapping, and haplotype counts
    try:
        inv_df = _load_inv_properties()
        # Map (chrom, start, end) -> recurrence_flag
        recurrence_map = {
            (str(row.chromosome), int(row.start), int(row.end)): int(row.recurrence_flag)
            for row in inv_df.itertuples(index=False)
        }

        pi_df = _load_pi_summary(drop_na_pi=False)
        pi_df["chr"] = pi_df["chr"].astype(str).str.replace("^chr", "", regex=True)
        hap_map: dict[tuple[str, int, int], tuple[int | None, int | None]] = {}
        for _, row in pi_df.iterrows():
            region = (str(row["chr"]), int(row["region_start"]), int(row["region_end"]))

            def _safe_int(val):
                try:
                    return int(val) if pd.notna(val) else None
                except Exception:
                    return None

            hap_map[region] = (
                _safe_int(row.get("0_num_hap_filter")),
                _safe_int(row.get("1_num_hap_filter")),
            )
    except Exception:
        raise

    # Edge/Middle Accumulators (Group 0, Group 1)
    acc_em_0 = _MetricAccumulator()
    acc_em_1 = _MetricAccumulator()

    # Edge/Middle Accumulators by subgroup (group, recurrence)
    acc_em_subgroup: dict[tuple[int, int], _MetricAccumulator] = {
        (0, 0): _MetricAccumulator(),
        (0, 1): _MetricAccumulator(),
        (1, 0): _MetricAccumulator(),
        (1, 1): _MetricAccumulator(),
    }

    # Spearman Accumulators (Group, Recurrence) -> list of window arrays
    # Keys: (0, 0), (0, 1), (1, 0), (1, 1)
    acc_spearman_mean: dict[tuple[int, int], list[np.ndarray]] = {
        (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []
    }
    acc_spearman_median: dict[tuple[int, int], list[np.ndarray]] = {
        (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []
    }

    spearman_points: list[SpearmanPoint] = []
    spearman_distances = np.arange(0, 100_000, 2_000, dtype=float)

    qualifying_regions: set[tuple[str, int, int]] = set()
    processed_entries: set[tuple[tuple[str, int, int], int]] = set()

    header_pattern = re.compile(
        r"chr[_:=]*(?P<chrom>[^_]+).*?start[_:=]*(?P<start>\d+).*?end[_:=]*(?P<end>\d+)",
        re.IGNORECASE,
    )

    def _parse_values(body_lines: list[str]) -> np.ndarray:
        if not body_lines:
            return np.array([], dtype=float)
        body_text = "".join(body_lines).strip()
        if not body_text:
            return np.array([], dtype=float)
        clean_text = re.sub(r"\bNA\b", "nan", body_text)
        try:
            return np.fromstring(clean_text, sep=",")
        except ValueError:
            return np.array([], dtype=float)

    def _process_record(header: str | None, body_lines: list[str]) -> None:
        if not header or not body_lines or not header.startswith(">filtered_pi"):
            return

        match = header_pattern.search(header)
        if not match:
            return
        chrom = match.group("chrom")
        start = int(match.group("start"))
        end = int(match.group("end"))

        # FILTER: Check against allowed list and get recurrence
        region = (chrom, start, end)
        if region not in recurrence_map:
            return
        recur_flag = recurrence_map[region]

        # Check group
        group_match = re.search(r"_group_(?P<grp>\d+)", header)
        if not group_match:
            return
        group_id = int(group_match.group("grp"))
        if group_id not in (0, 1):
            return

        # Duplicate check (warn but still process to mirror manuscript counts)
        if (region, group_id) in processed_entries:
            print(
                f"WARNING: Duplicate inversion {region} in group {group_id}; processing again."
            )
        processed_entries.add((region, group_id))

        values = _parse_values(body_lines)
        if values.size == 0:
            return

        # --- Logic for Edge/Middle (Threshold 40kb) ---
        if values.size >= 40_000:
            qualifying_regions.add(region)
            hap_counts = hap_map.get(region, (None, None)) if "hap_map" in locals() else (None, None)
            hap_count = hap_counts[group_id] if hap_counts else None
            if group_id == 0:
                acc_em_0.add_edge_middle(values, hap_count=hap_count)
            else:
                acc_em_1.add_edge_middle(values, hap_count=hap_count)

            subgroup_acc = acc_em_subgroup.get((group_id, recur_flag))
            if subgroup_acc is not None:
                subgroup_acc.add_edge_middle(values, hap_count=hap_count)

        # --- Logic for Spearman (Threshold 100kb) ---
        # "folded" 100 kbp ... (average of start and reversed end)
        if values.size >= 100_000:
            window_means, window_medians = _distance_fold_binning(values)

            point_rho = None
            point_p = None
            mask = np.isfinite(window_means)
            bins_used = int(np.sum(mask))
            if bins_used >= 5:
                rho_val, p_val = stats.spearmanr(
                    spearman_distances[mask], window_means[mask]
                )
                point_rho = float(rho_val) if np.isfinite(rho_val) else None
                point_p = float(p_val) if np.isfinite(p_val) else None

            spearman_points.append(
                SpearmanPoint(
                    rho=point_rho,
                    p_value=point_p,
                    recurrence_flag=recur_flag,
                    group=group_id,
                    region=region,
                    bins_used=bins_used,
                )
            )

            key = (group_id, recur_flag)
            if key in acc_spearman_mean:
                acc_spearman_mean[key].append(window_means)
                # Use per-bin means for both statistics to mirror the manuscript logic
                # (median of per-bin means across inversions).
                acc_spearman_median[key].append(window_means)

    current_header: str | None = None
    sequence_lines: list[str] = []
    if falsta_path.suffix == ".gz":
        handle_factory = lambda: gzip.open(falsta_path, "rt", encoding="utf-8")
    else:
        handle_factory = lambda: falsta_path.open("r", encoding="utf-8")

    with handle_factory() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _process_record(current_header, sequence_lines)
                current_header = line
                sequence_lines = []
            else:
                sequence_lines.append(line)
    _process_record(current_header, sequence_lines)

    # --- Compile Edge/Middle Metrics ---
    def _em_stats(acc: _MetricAccumulator) -> EdgeMiddleStats:
        fm = float(np.mean(acc.flank_means)) if acc.flank_means else None
        mm = float(np.mean(acc.middle_means)) if acc.middle_means else None
        n = len(acc.flank_means)

        hap_total = 0
        hap_mean = None
        hap_median = None
        hap_min = None
        hap_max = None

        if acc.haplotype_counts:
            haps = np.array(acc.haplotype_counts)
            hap_total = int(np.sum(haps))
            hap_mean = float(np.mean(haps))
            hap_median = float(np.median(haps))
            hap_min = int(np.min(haps))
            hap_max = int(np.max(haps))

        return EdgeMiddleStats(
            flank_mean=fm,
            middle_mean=mm,
            entries=n,
            total_haplotypes=hap_total,
            mean_haplotypes=hap_mean,
            median_haplotypes=hap_median,
            min_haplotypes=hap_min,
            max_haplotypes=hap_max
        )

    stats_0 = _em_stats(acc_em_0)
    stats_1 = _em_stats(acc_em_1)

    # Overall Edge/Middle
    acc_em_all = _MetricAccumulator()
    acc_em_all.flank_means = acc_em_0.flank_means + acc_em_1.flank_means
    acc_em_all.middle_means = acc_em_0.middle_means + acc_em_1.middle_means
    acc_em_all.haplotype_counts = acc_em_0.haplotype_counts + acc_em_1.haplotype_counts
    stats_all = _em_stats(acc_em_all)

    subgroup_stats: dict[tuple[int, int], EdgeMiddleStats] = {}
    for key, acc in acc_em_subgroup.items():
        subgroup_stats[key] = _em_stats(acc)

    p_for_fdr = [pt.p_value for pt in spearman_points if pt.p_value is not None]
    if p_for_fdr:
        try:
            from statsmodels.stats.multitest import multipletests

            _, q_vals, _, _ = multipletests(p_for_fdr, alpha=0.05, method="fdr_bh")
            idx = 0
            for pt in spearman_points:
                if pt.p_value is None:
                    continue
                pt.q_value = float(q_vals[idx])
                idx += 1
        except Exception:
            # If statsmodels is unavailable or the correction fails, leave q-values as None
            pass

    # --- Compile Spearman Metrics ---
    # 1. Overall (All 4 subgroups)
    all_spearman_data = (
        acc_spearman_mean[(0, 0)] + acc_spearman_mean[(0, 1)] +
        acc_spearman_mean[(1, 0)] + acc_spearman_mean[(1, 1)]
    )
    res_overall = _calc_spearman(all_spearman_data, agg_func=np.nanmedian)
    all_spearman_median_data = (
        acc_spearman_median[(0, 0)] + acc_spearman_median[(0, 1)] +
        acc_spearman_median[(1, 0)] + acc_spearman_median[(1, 1)]
    )
    res_overall_median = _calc_spearman(all_spearman_median_data, agg_func=np.nanmedian)

    # 2. Single-Inv (G1, R0)
    res_single_inv = _calc_spearman(acc_spearman_mean[(1, 0)], agg_func=np.nanmedian)
    res_single_inv_median = _calc_spearman(acc_spearman_median[(1, 0)], agg_func=np.nanmedian)

    # 3. Recur-Dir (G0, R1)
    res_recur_dir = _calc_spearman(acc_spearman_mean[(0, 1)], agg_func=np.nanmedian)
    res_recur_dir_median = _calc_spearman(acc_spearman_median[(0, 1)], agg_func=np.nanmedian)

    # 4. Recur-Inv (G1, R1)
    res_recur_inv = _calc_spearman(acc_spearman_mean[(1, 1)], agg_func=np.nanmedian)
    res_recur_inv_median = _calc_spearman(acc_spearman_median[(1, 1)], agg_func=np.nanmedian)

    # 5. Single-Dir (G0, R0)
    res_single_dir = _calc_spearman(acc_spearman_mean[(0, 0)], agg_func=np.nanmedian)
    res_single_dir_median = _calc_spearman(acc_spearman_median[(0, 0)], agg_func=np.nanmedian)

    return PiStructureMetrics(
        dir_stats=stats_0,
        inv_stats=stats_1,
        all_stats=stats_all,

        subgroup_edge_middle=subgroup_stats,

        spearman_overall=res_overall,
        spearman_single_inv=res_single_inv,
        spearman_recur_dir=res_recur_dir,
        spearman_recur_inv=res_recur_inv,
        spearman_single_dir=res_single_dir,

        spearman_overall_median=res_overall_median,
        spearman_single_inv_median=res_single_inv_median,
        spearman_recur_dir_median=res_recur_dir_median,
        spearman_recur_inv_median=res_recur_inv_median,
        spearman_single_dir_median=res_single_dir_median,

        spearman_points=spearman_points,

        unique_inversions=len(qualifying_regions),
    )


def _save_spearman_points(points: list[SpearmanPoint]) -> Path | None:
    if not points:
        return None

    df = pd.DataFrame(
        [
            {
                "rho": p.rho,
                "p_value": p.p_value,
                "q_value": p.q_value,
                "recurrence_flag": p.recurrence_flag,
                "group": p.group,
                "region_chr": p.region[0] if p.region else None,
                "region_start": p.region[1] if p.region else None,
                "region_end": p.region[2] if p.region else None,
                "bins_used": p.bins_used,
            }
            for p in points
            if p.rho is not None
            and p.p_value is not None
            and np.isfinite(p.p_value)
            and p.bins_used >= 5
        ]
    )

    if df.empty:
        return None

    output_path = DATA_DIR / "spearman_decay_points.tsv"
    df.to_csv(output_path, sep="\t", index=False)
    return output_path


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


def _load_pi_summary(drop_na_pi: bool = True) -> pd.DataFrame:
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
    if drop_na_pi:
        merged = merged.dropna(subset=["0_pi_filtered", "1_pi_filtered"])
    return merged


def _load_fst_table() -> pd.DataFrame | None:
    fst_candidates = [
        DATA_DIR / "FST_data.tsv",
        DATA_DIR / "FST_data.tsv.gz",
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

    # Load with drop_na_pi=False to get haplotype counts for all loci,
    # even those where pi could not be calculated for one orientation.
    pi_df_unfiltered = _load_pi_summary(drop_na_pi=False)

    # Count loci with at least two haplotypes, regardless of orientation
    # Using fillna(0) because NaN implies 0 valid haplotypes for that orientation
    h0 = pi_df_unfiltered["0_num_hap_filter"].fillna(0)
    h1 = pi_df_unfiltered["1_num_hap_filter"].fillna(0)
    total_haps = h0 + h1

    num_with_two_haps_total = (total_haps >= 2).sum()
    num_dir_ge_2 = (h0 >= 2).sum()
    num_inv_ge_2 = (h1 >= 2).sum()
    num_both_ge_2 = ((h0 >= 2) & (h1 >= 2)).sum()
    num_either_ge_2 = ((h0 >= 2) | (h1 >= 2)).sum()

    lines.append(
        "  Number of loci with at least two haplotypes (total across orientations): "
        f"{_fmt(num_with_two_haps_total, 0)}."
    )
    lines.append(
        "  Number of loci with at least two direct haplotypes: "
        f"{_fmt(num_dir_ge_2, 0)}."
    )
    lines.append(
        "  Number of loci with at least two inverted haplotypes: "
        f"{_fmt(num_inv_ge_2, 0)}."
    )
    lines.append(
        "  Number of loci with at least two haplotypes in either orientation (union): "
        f"{_fmt(num_either_ge_2, 0)}."
    )
    lines.append(
        "  Number of loci with at least two haplotypes in each orientation (intersection): "
        f"{_fmt(num_both_ge_2, 0)}."
    )

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
    inv_median = _safe_median(df["1_pi_filtered"])
    dir_median = _safe_median(df["0_pi_filtered"])
    fold_lower = None
    if inv_median not in (None, 0) and dir_median is not None:
        fold_lower = dir_median / inv_median
    ttest = stats.ttest_rel(df["1_pi_filtered"], df["0_pi_filtered"])
    lines.append(
        "  Across all loci: mean π(inverted) = "
        f"{_fmt(inv_mean, 6)}, mean π(direct) = {_fmt(dir_mean, 6)}."
    )
    lines.append(
        "    Median π(inverted) = "
        f"{_fmt(inv_median, 6)}, median π(direct) = {_fmt(dir_median, 6)}."
    )
    if fold_lower is not None:
        if fold_lower >= 1:
            lines.append(
                "    Median π(inverted) is "
                f"{_fmt(fold_lower, 3)}-fold lower than median π(direct)."
            )
        else:
            lines.append(
                "    Median π(inverted) is "
                f"{_fmt(1 / fold_lower, 3)}-fold higher than median π(direct)."
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


def summarize_pi_structure() -> List[str]:
    try:
        metrics = _calc_pi_structure_metrics()
    except FileNotFoundError as exc:
        return [f"Pi structure inputs unavailable: {exc}"]
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        return [f"Pi structure summary failed: {exc}"]

    lines = [
        (
            "Nucleotide diversity structure (Edge vs Middle and Internal Decay), "
            "filtered by consensus inversion status:"
        ),
        f"  Qualifying regions (≥40kbp): {_fmt(metrics.unique_inversions, 0)} unique inversions.",
    ]

    def _fmt_hap(stats: EdgeMiddleStats) -> str:
        if stats.entries == 0:
            return "n=0, Haplotypes=NA"

        s = f"n={stats.entries} regions. Haplotypes (N): Total={stats.total_haplotypes}"
        if stats.mean_haplotypes is not None:
            s += f", Mean={_fmt(stats.mean_haplotypes, 1)}"
            s += f" [Min {stats.min_haplotypes}, Max {stats.max_haplotypes}]"
        return s

    # Direct
    lines.append(
        f"  [Edge vs Middle] Direct/Group 0: {_fmt_hap(metrics.dir_stats)}. "
        f"Flank Mean = {_fmt(metrics.dir_stats.flank_mean)}, Middle Mean = {_fmt(metrics.dir_stats.middle_mean)}."
    )

    # Inverted
    lines.append(
        f"  [Edge vs Middle] Inverted/Group 1: {_fmt_hap(metrics.inv_stats)}. "
        f"Flank Mean = {_fmt(metrics.inv_stats.flank_mean)}, Middle Mean = {_fmt(metrics.inv_stats.middle_mean)}."
    )

    # Overall
    lines.append(
        f"  [Edge vs Middle] Overall: {_fmt_hap(metrics.all_stats)}. "
        f"Flank Mean = {_fmt(metrics.all_stats.flank_mean)}, Middle Mean = {_fmt(metrics.all_stats.middle_mean)}."
    )

    subgroup_labels = {
        (1, 0): "Single-event Inverted (Group 1, Recurrence 0)",
        (0, 1): "Recurrent Direct (Group 0, Recurrence 1)",
        (0, 0): "Single-event Direct (Group 0, Recurrence 0)",
        (1, 1): "Recurrent Inverted (Group 1, Recurrence 1)",
    }
    for key, label in subgroup_labels.items():
        stats = metrics.subgroup_edge_middle.get(key)
        if stats is None:
            continue
        lines.append(
            f"    - {label}: {_fmt_hap(stats)}. "
            f"Flank Mean = {_fmt(stats.flank_mean)}, Middle Mean = {_fmt(stats.middle_mean)}."
        )

    # Spearman Decay
    lines.append("")
    lines.append("Internal decay (Spearman's ρ of diversity vs distance from start for first 100kb, loci ≥100kb):")

    def _fmt_spearman(r, label):
        return (
            f"  {label}: ρ = {_fmt(r.rho, 3)} "
            f"(p(two-sided) = {_fmt(r.p_value, 3)}, n = {_fmt(r.n, 0)})."
        )

    lines.append(_fmt_spearman(metrics.spearman_overall, "Overall (All Consensus 0+1)"))
    lines.append(_fmt_spearman(metrics.spearman_single_inv, "Single-Event Inverted (G1, R0)"))
    lines.append(_fmt_spearman(metrics.spearman_recur_dir, "Recurrent Direct (G0, R1)"))
    lines.append(_fmt_spearman(metrics.spearman_recur_inv, "Recurrent Inverted (G1, R1)"))
    lines.append(_fmt_spearman(metrics.spearman_single_dir, "Single-Event Direct (G0, R0)"))

    lines.append("")
    lines.append("Internal decay (median within 2kb bins, Spearman's ρ for first 100kb, loci ≥100kb):")
    lines.append(_fmt_spearman(metrics.spearman_overall_median, "Overall (All Consensus 0+1)"))
    lines.append(_fmt_spearman(metrics.spearman_single_inv_median, "Single-Event Inverted (G1, R0)"))
    lines.append(_fmt_spearman(metrics.spearman_recur_dir_median, "Recurrent Direct (G0, R1)"))
    lines.append(_fmt_spearman(metrics.spearman_recur_inv_median, "Recurrent Inverted (G1, R1)"))
    lines.append(_fmt_spearman(metrics.spearman_single_dir_median, "Single-Event Direct (G0, R0)"))

    saved_points = _save_spearman_points(metrics.spearman_points)
    if saved_points:
        lines.append("")
        lines.append(
            "  Saved Spearman decay points (including q-values) to: "
            f"{_relative_to_repo(saved_points)}."
        )

    return lines


def summarize_fst_edge_decay() -> List[str]:
    header = (
        "Folded Hudson FST decay (two-sided Spearman testing for changes from edge to center):"
    )
    try:
        results = fst_edge_decay.compute_fst_edge_decay(DATA_DIR)
    except FileNotFoundError as exc:
        return [f"Hudson FST decay inputs unavailable: {exc}"]
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        return [f"Hudson FST decay summary failed: {exc}"]

    if not results:
        return [header, "  No qualifying inversions (length > 100kbp with recurrence labels)."]

    lines = [header]
    lines.append(
        "  Reporting only inversions with ≥5 usable bins and finite two-sided p-values; others are omitted."
    )
    results = sorted(
        results,
        key=lambda r: (r.chrom, r.start, r.end, r.recurrence_flag),
    )
    for res in results:
        lines.append(f"  {res.inv_label} ({res.recurrence_label}):")
        lines.append(
            "    "
            f"rho={_fmt(res.rho)}, "
            f"p(two-sided)={_fmt_pvalue(res.p_two_sided)}, "
            f"q={_fmt_pvalue(res.q_value)}, "
            f"bins={_fmt(res.bins_used, 0)}"
        )
    return lines


def summarize_linear_model() -> List[str]:
    pi_path = DATA_DIR / "output.csv"
    inv_path = DATA_DIR / "inv_properties.tsv"

    # load_and_match expects string paths and handles strict matching logic.
    try:
        matched = inv_dir_recur_model.load_and_match(str(pi_path), str(inv_path))
    except Exception as exc:
        return [f"Strict data loading failed: {exc}"]

    # Calculate epsilon floor exactly as in the modeling script
    all_pi = np.r_[matched["pi_direct"].to_numpy(float), matched["pi_inverted"].to_numpy(float)]
    eps = inv_dir_recur_model.choose_floor_from_quantile(
        all_pi,
        q=inv_dir_recur_model.FLOOR_QUANTILE,
        min_floor=inv_dir_recur_model.MIN_FLOOR,
    )

    lines = ["Orientation × recurrence linear models (replicated strict logic):"]
    lines.append(

        "  Model definitions (mirroring stats/inv_dir_recur_model.py):"
    )
    lines.append(
        "    [Model A] Outcome Δlogπ = log(π_inverted+ε) − log(π_direct+ε); "
        "predictor is a Recurrent indicator (Single-event baseline); HC3 "
        "robust SEs; contrasts report single-event, recurrent, interaction, "
        "and pooled inversion effects."
    )
    lines.append(
        "    [Model B] Rows duplicated per orientation with outcome log(π+ε); "
        "OLS with design log_pi ~ Inverted + Inverted:Recurrent + C(region_id); "
        "cluster-robust by region_id; recurrence main effect absorbed by "
        "fixed effects; contrasts compare orientation within recurrence "
        "groups and their interaction."
    )
    lines.append(
        "    [Model C] Outcome Δlogπ as in Model A with predictors Recurrent "
        "+ z-scored covariates ln1p(Number_recurrent_events), ln(Size_kbp), "
        "Inverted_AF (raw z), ln(Formation_rate_per_generation); HC3 robust "
        "SEs; rows with missing covariates are dropped and effects are per +1 SD."
    )
    lines.append(f"  Detection floor applied before logs: ε = {_fmt(eps, 6)}.")

    # Model A (Basic)
    lines.append(
        "  [Model A] Δ-logπ = log(π_inv+ε) – log(π_dir+ε) ~ 1 + Recurrent (HC3 SEs). "
        "No weights or covariates; effects reported for single-event, recurrent, "
        "interaction, and pooled Δ-logπ."
    )
    try:
        _, tabA, dfA = inv_dir_recur_model.run_model_A(matched, eps=eps, nonzero_only=False)
        for row in tabA.itertuples():
            lines.append(
                f"    {row.effect}: fold-change = {_fmt(row.ratio, 3)} "
                f"(95% CI {_fmt(row.ci_low, 3)}–{_fmt(row.ci_high, 3)}), p = {_fmt(row.p, 3)}."
            )
    except Exception as exc:
        lines.append(f"    Model A failed: {exc}")

    # Model B (Fixed Effects)
    lines.append(
        "  [Model B] log(π+ε) ~ Inverted + Inverted:Recurrent + C(region_id); "
        "cluster-robust by region_id. Recurrence main effect absorbed by fixed "
        "effects; contrasts give single-event, recurrent, and interaction pairs."
    )
    try:
        _, tabB, _, _ = inv_dir_recur_model.run_model_B(matched, eps=eps)
        for row in tabB.itertuples():
            lines.append(
                f"    {row.effect}: fold-change = {_fmt(row.ratio, 3)} "
                f"(95% CI {_fmt(row.ci_low, 3)}–{_fmt(row.ci_high, 3)}), p = {_fmt(row.p, 3)}."
            )
    except Exception as exc:
        lines.append(f"    Model B failed: {exc}")

    # Model C (Covariate Adjusted)
    lines.append(
        "  [Model C] Δ-logπ ~ 1 + Recurrent + z-scored covariates from inv_properties.tsv "
        "(Number_recurrent_events ln1p, Size_.kbp. ln, Inverted_AF, Formation_rate_per_generation ln). "
        "HC3 SEs; rows require complete covariates with missingness dummies excluded."
    )
    try:
        _, tabC, _, _ = inv_dir_recur_model.run_model_C(
            matched, invinfo_path=str(inv_path), eps=eps, nonzero_only=False
        )
        covariate_rows = tabC.iloc[3:]
        lines.append(
            "    Covariates included in fit: "
            + (", ".join(covariate_rows.effect) if not covariate_rows.empty else "None (dropped as constant)")
        )
        for row in tabC.itertuples():
            lines.append(
                f"    {row.effect}: fold-change = {_fmt(row.ratio, 3)} "
                f"(95% CI {_fmt(row.ci_low, 3)}–{_fmt(row.ci_high, 3)}), p = {_fmt(row.p, 3)}."
            )
    except Exception as exc:
        lines.append(f"    Model C failed: {exc}")

    # Permutation Test
    lines.append(
        f"  [Permutation] Model A interaction (n={_fmt(inv_dir_recur_model.N_PERMUTATIONS, 0)}):"
    )
    try:
        obs, p_perm = inv_dir_recur_model.perm_test_interaction(
            dfA,
            n=inv_dir_recur_model.N_PERMUTATIONS,
            seed=inv_dir_recur_model.PERM_SEED,
        )
        lines.append(f"    Observed Δ(mean log-ratio) = {_fmt(obs, 4)}, p = {_fmt(p_perm, 4)}.")
    except Exception as exc:
        lines.append(f"    Permutation test failed: {exc}")

    return lines

def summarize_cds_conservation_glm() -> List[str]:
    lines: List[str] = [
        "CDS conservation GLM (proportion of identical CDS pairs):",
        "  Model definition: Binomial GLM with logit link and frequency weights = n_pairs, "
        "cluster-robust by inversion; formula prop ~ C(consensus) * C(phy_group) + "
        "log_m + log_L + log_k (log of n_sites, inversion length, and n_sequences).",
        "  Categories use Single/Recurrent × Direct/Inverted encoding; estimated marginal "
        "means are standardized with equal inversion weight and covariates set to their "
        "weighted means before pairwise contrasts.",
    ]

    res_nocov: object | None = None
    res_adj: object | None = None
    emm_nocov: pd.DataFrame | None = None
    emm_adj: pd.DataFrame | None = None
    pw_nocov: pd.DataFrame | None = None
    pw_adj: pd.DataFrame | None = None
    source_label: str | None = None
    errors: List[str] = []

    cds_input = _resolve_repo_artifact("cds_identical_proportions.tsv")

    if cds_input and cds_input.exists():
        try:
            with _temporary_workdir(cds_input.parent):
                cds_df = CDS_identical_model.load_data()
                res_nocov = CDS_identical_model.fit_glm_binom(
                    cds_df, include_covariates=False
                )
                emm_nocov, pw_nocov = CDS_identical_model.emms_and_pairs(
                    res_nocov, cds_df, include_covariates=False
                )
                res_adj = CDS_identical_model.fit_glm_binom(
                    cds_df, include_covariates=True
                )
                emm_adj, pw_adj = CDS_identical_model.emms_and_pairs(
                    res_adj, cds_df, include_covariates=True
                )
            source_label = f"loaded from {_relative_to_repo(cds_input)}"
        except SystemExit as exc:
            errors.append(f"CDS GLM exited early: {exc}")
        except Exception as exc:
            errors.append(f"Failed to compute GLM: {exc}")
    else:
        errors.append("cds_identical_proportions.tsv not found (Pipeline failure?)")

    if pw_adj is None and pw_nocov is None:
        lines.append(
            "  FATAL: CDS GLM inputs unavailable. The pipeline should have generated cds_identical_proportions.tsv."
        )
        lines.extend(f"  {msg}" for msg in errors)
        # Return lines but likely this indicates a critical failure
        return lines

    if source_label:
        lines.append(f"  Source: {source_label}.")

    if errors:
        lines.extend(f"  WARNING: {msg}" for msg in errors)

    if "cds_df" in locals():
        lines.append(
            "  Input summary: "
            + f"n_rows = {_fmt(len(cds_df), 0)}, n_inversions = {_fmt(cds_df['inv_id'].nunique(), 0)}."
        )

    def _append_means(label: str, emm: pd.DataFrame | None):
        if emm is None:
            lines.append(f"  {label}: marginal means unavailable.")
            return
        lines.append(f"  {label}: standardized marginal means (equal inversion weight):")
        for row in emm.sort_values("p_hat", ascending=False).itertuples():
            lines.append(
                f"    {row.category}: p̂ = {row.p_hat * 100:.4f}% "
                f"(95% CI {row.p_lcl95 * 100:.4f}–{row.p_ucl95 * 100:.4f}%)."
            )

    def _append_pairs(label: str, pairwise: pd.DataFrame | None):
        if pairwise is None:
            lines.append(f"  {label}: pairwise contrasts unavailable.")
            return
        required = {
            "A",
            "B",
            "diff_logit",
            "diff_prob",
            "p_value",
            "q_value_fdr",
        }
        if not required.issubset(pairwise.columns):
            missing = ", ".join(sorted(required - set(pairwise.columns)))
            lines.append(f"  {label}: contrast table missing required columns: {missing}.")
            return

        lines.append(f"  {label}: pairwise contrasts (BH-FDR):")
        for row in pairwise.itertuples():
            lines.append(
                "    "
                + f"{row.A} vs {row.B}: Δlogit = {_fmt(row.diff_logit, 3)}, "
                + f"Δp = {row.diff_prob * 100:.1f}%, p = {_fmt(row.p_value, 3)}, "
                + f"BH q = {_fmt(row.q_value_fdr, 3)}."
            )

    def _append_fit(label: str, res) -> None:
        if res is None:
            lines.append(f"  {label}: model fit unavailable.")
            return
        pseudo_r2 = None
        try:
            pseudo_r2 = 1 - float(res.deviance) / float(res.null_deviance)
        except Exception:
            pseudo_r2 = None
        lines.append(
            "  "
            + f"{label}: logLik = {_fmt(getattr(res, 'llf', None), 3)}, "
            + f"null deviance = {_fmt(getattr(res, 'null_deviance', None), 3)}, "
            + f"residual deviance = {_fmt(getattr(res, 'deviance', None), 3)}, "
            + f"pseudo-R² = {_fmt(pseudo_r2, 3)}."
        )

    _append_fit("Unadjusted model", res_nocov)
    _append_means("Unadjusted model", emm_nocov)
    _append_pairs("Unadjusted model", pw_nocov)

    _append_fit("Adjusted model", res_adj)
    _append_means("Adjusted model", emm_adj)
    _append_pairs("Adjusted model", pw_adj)

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
        frf_path = REPO_ROOT / "per_inversion_breakpoint_tests" / "per_inversion_frf_effects.tsv"
        if not frf_path.exists():
            return ["Breakpoint FRF results not found; skipping enrichment analysis."]

    frf = pd.read_csv(frf_path, sep="\t", low_memory=False)

    if "STATUS" in frf.columns and "recurrence_flag" not in frf.columns:
        frf["recurrence_flag"] = frf["STATUS"]

    frf = frf.rename(columns={"frf_delta": "edge_minus_middle", "usable_for_meta": "usable"})
    
    if {"chrom", "start", "end"}.issubset(frf.columns):
        frf["chromosome_norm"] = frf["chrom"].astype(str).str.replace("^chr", "", regex=True)
        try:
            inv_df = _load_inv_properties()
            frf = frf.merge(
                inv_df[["chromosome", "start", "end", "recurrence_label", "inversion_id"]],
                left_on=["chromosome_norm", "start", "end"],
                right_on=["chromosome", "start", "end"],
                how="left",
                suffixes=("", "_inv"),
            )
        except Exception:
            pass

    lines: List[str] = ["Breakpoint enrichment (Flat–Ramp–Flat Model):"]

    if "usable" in frf.columns:
        usable_mask = frf["usable"].fillna(False).astype(bool) | \
                      frf["usable"].astype(str).str.lower().isin(["true", "1"])
        usable = frf[usable_mask].copy()
    else:
        usable = frf[np.isfinite(frf["frf_var_delta"]) & (frf["frf_var_delta"] > 0)].copy()

    if "recurrence_flag" not in usable.columns:
        lines.append("  Recurrence annotations missing (no 'STATUS' or 'recurrence_flag' column).")
        return lines
    
    # --- Descriptive Stats (Unweighted Levels) ---
    if {"frf_mu_edge", "frf_mu_mid"}.issubset(usable.columns):
        lines.append("  [Descriptive Levels] Raw FST averages (Unweighted):")
        for flag, label in [(0, "Single-event"), (1, "Recurrent")]:
            sub = usable[usable["recurrence_flag"] == flag]
            if not sub.empty:
                mean_edge = _safe_mean(sub["frf_mu_edge"])
                mean_mid = _safe_mean(sub["frf_mu_mid"])
                lines.append(f"    {label} (n={len(sub)}): Edge={_fmt(mean_edge)}, Middle={_fmt(mean_mid)}.")
    lines.append("")

    # --- Old Method (Unweighted Delta) ---
    lines.append("  [Unweighted Delta Analysis]")
    vecs = {}
    deltas = {}
    for flag, label in [(0, "Single-event"), (1, "Recurrent")]:
        sub = usable[usable["recurrence_flag"] == flag]
        vec = sub["edge_minus_middle"].dropna().to_numpy(dtype=float)
        if vec.size > 0:
            vecs[flag] = vec
            deltas[flag] = float(np.mean(vec))
            lines.append(f"    {label} mean delta: {_fmt(deltas[flag], 3)}.")

    if 0 in vecs and 1 in vecs:
        diff = deltas[0] - deltas[1]
        lines.append(f"    Diff-of-diffs (Single - Recurrent): {_fmt(diff, 3)}.")
        res = recur_breakpoint_tests.directional_energy_test(
            vecs[0], vecs[1], n_perm=10000, random_state=2025
        )
        lines.append(f"    Energy Test p-value (Single > Recurrent): {_fmt(res['p_value_0gt1'], 3)}.")
    lines.append("")

    # --- New Method (Precision-Weighted Meta-Analysis) ---
    lines.append("  [Precision-Weighted Meta-Analysis]")
    
    y = usable["edge_minus_middle"].to_numpy(dtype=float)
    s2 = usable["frf_var_delta"].to_numpy(dtype=float)
    group = usable["recurrence_flag"].to_numpy(dtype=int)

    if len(y) == 0 or len(s2) == 0:
        lines.append("    Insufficient data for weighted analysis.")
        return lines

    weights = per_inversion_breakpoint_metric.compute_meta_weights_from_s2(s2)
    n_perm = 20_000_000
    n_workers = os.cpu_count() or 1

    # Weighted Median
    d_med, med_s, med_r = per_inversion_breakpoint_metric.weighted_median_difference(y, weights, group)
    perm_med = per_inversion_breakpoint_metric.meta_permutation_pvalue(
        y, weights, group, n_perm=n_perm, chunk=1000, base_seed=2025, n_workers=n_workers
    )
    lines.append(f"    Weighted Median Delta: Single={_fmt(med_s)}, Recurrent={_fmt(med_r)}, Diff={_fmt(d_med)}.")
    lines.append(f"    Median P-value (Two-sided): {_fmt(perm_med['p_perm_two_sided'], 4)}.")

    # Weighted Mean
    d_mean, mean_s, mean_r = per_inversion_breakpoint_metric.weighted_mean_difference(y, weights, group)
    perm_mean = per_inversion_breakpoint_metric.meta_permutation_pvalue_mean(
        y, weights, group, n_perm=n_perm, chunk=1000, base_seed=2026, n_workers=n_workers
    )
    lines.append(f"    Weighted Mean Delta:   Single={_fmt(mean_s)}, Recurrent={_fmt(mean_r)}, Diff={_fmt(d_mean)}.")
    lines.append(f"    Mean P-value (Two-sided):   {_fmt(perm_mean['p_perm_two_sided'], 4)}.")

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
    ancestry_targets: List[str] = field(default_factory=list)


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
            ancestry_targets=["AFR"],
        ),
        AssocSpec(
            "chr12-46897663-INV-16289",
            "Conjunctivitis",
            ("conjunct",),
        ),
        AssocSpec(
            "chr12-46897663-INV-16289",
            "Acne",
            ("acne",),
        ),
        AssocSpec(
            "chr12-46897663-INV-16289",
            "Epidermal thickening",
            ("epidermal", "thicken"),
        ),
        AssocSpec(
            "chr12-46897663-INV-16289",
            "Inflammation of the eye",
            ("inflamm", "eye"),
        ),
        AssocSpec(
            "chr12-46897663-INV-16289",
            "Migraine",
            ("migraine",),
        ),
        AssocSpec(
            "chr12-46897663-INV-16289",
            "Disorder of nervous system",
            ("disorder", "nervous"),
        ),
        AssocSpec(
            # The main imputed Morbid obesity signal comes from the 17q21 inversion
            # (chr17-45585160-INV-706887). The tag-SNP analyses use a synthetic
            # identifier (chr17-45974480-INV-29218), so we keep the tag-SNP
            # entries below on that ID but point the main PheWAS summary at the
            # true inversion identifier so the odds ratios reported here match
            # the imputed results discussed in the manuscript.
            "chr17-45585160-INV-706887",
            "Morbid obesity (Main Imputed)",
            ("morbid", "obesity"),
            table_name="phewas_results.tsv",
            ancestry_targets=["EUR", "AFR"],
        ),
        AssocSpec(
            "chr17-45585160-INV-706887",
            "Breast lump or abnormal exam (Main Imputed)",
            ("lump", "breast"),
            table_name="phewas_results.tsv",
            ancestry_targets=["EUR"],
        ),
        AssocSpec(
            "chr17-45585160-INV-706887",
            "Abnormal mammogram (Main Imputed)",
            ("mammogram",),
            table_name="phewas_results.tsv",
            ancestry_targets=["EUR"],
        ),
        AssocSpec(
            "chr17-45585160-INV-706887",
            "Mild cognitive impairment (Main Imputed)",
            ("mild", "cognitive"),
            table_name="phewas_results.tsv",
            ancestry_targets=["EUR", "AMR"],
        ),
        AssocSpec(
            "chr17-45585160-INV-706887",
            "Abnormal Papanicolaou smear",
            ("papanicolaou", "smear"),
            table_name="phewas_results.tsv",
        ),
        AssocSpec(
            "chr17-45585160-INV-706887",
            "Melanocytic nevi",
            ("melanocytic", "nevi"),
            table_name="phewas_results.tsv",
        ),
        AssocSpec(
            "chr17-45585160-INV-706887",
            "Benign neoplasm of the skin",
            ("benign", "neoplasm", "skin"),
            table_name="phewas_results.tsv",
        ),
        AssocSpec(
            "chr17-45585160-INV-706887",
            "Diastolic Heart Failure",
            ("diastolic", "heart", "failure"),
            table_name="phewas_results.tsv",
        ),
        AssocSpec(
            "chr17-45585160-INV-706887",
            "Breast Cancer (Malignant neoplasm)",
            ("malignant", "neoplasm", "breast"),
            table_name="phewas_results.tsv",
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Morbid obesity (Tag SNP)",
            ("morbid", "obesity"),
            table_name="all_pop_phewas_tag.tsv",
            ancestry_targets=["EUR", "AFR"],
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Breast lump or abnormal exam (Tag SNP)",
            ("lump", "breast"),
            table_name="all_pop_phewas_tag.tsv",
            ancestry_targets=["EUR"],
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Abnormal mammogram (Tag SNP)",
            ("mammogram",),
            table_name="all_pop_phewas_tag.tsv",
            ancestry_targets=["EUR"],
        ),
        AssocSpec(
            "chr17-45974480-INV-29218",
            "Mild cognitive impairment (Tag SNP)",
            ("mild", "cognitive"),
            table_name="all_pop_phewas_tag.tsv",
            ancestry_targets=["EUR", "AMR"],
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

        interaction_col = "P_LRT_AncestryxDosage"
        interaction_val = r.get(interaction_col) if interaction_col in r.index else None
        if interaction_val is not None and not pd.isna(interaction_val):
            lines.append(
                f"    Interaction (Ancestry × Dosage): p = {_fmt(interaction_val, 3)}."
            )

        for anc in spec.ancestry_targets:
            p_col = f"{anc}_P"
            or_col = f"{anc}_OR"
            lo_col = f"{anc}_CI_LO_OR"
            hi_col = f"{anc}_CI_HI_OR"
            p_val = r.get(p_col)
            if p_val is None or pd.isna(p_val):
                continue
            line = f"    [{anc}] p = {_fmt(p_val, 3)}"
            or_val = r.get(or_col)
            if or_val is not None and not pd.isna(or_val):
                lo_val = r.get(lo_col)
                hi_val = r.get(hi_col)
                if (
                    lo_val is not None
                    and hi_val is not None
                    and not pd.isna(lo_val)
                    and not pd.isna(hi_val)
                ):
                    line += (
                        f", OR = {_fmt(or_val, 3)} (95% CI {_fmt(lo_val, 3)}–{_fmt(hi_val, 3)})"
                    )
                else:
                    line += f", OR = {_fmt(or_val, 3)}"
            lines.append(line + ".")
    return lines


def summarize_category_tests() -> List[str]:
    cat_path = DATA_DIR / "phewas v2 - categories.tsv"
    if not cat_path.exists():
        return ["Phecode category-level omnibus results not found; skipping summary."]

    categories = pd.read_csv(cat_path, sep="\t", low_memory=False)
    required = {
        "Inversion",
        "Category",
        "Direction",
        "P_GBJ",
        "P_GLS",
        "Q_GBJ",
        "Q_GLS",
    }
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
            direction_label: str | None
            raw_direction = getattr(row, "Direction", None)
            if isinstance(raw_direction, str):
                normalized = raw_direction.strip().lower()
                if normalized == "increase":
                    direction_label = "Increased risk"
                elif normalized == "decrease":
                    direction_label = "Decreased risk"
                else:
                    direction_label = raw_direction.strip() or None
            else:
                direction_label = None
            if direction_label:
                category_name = f"{row.Category} ({direction_label})"
            else:
                category_name = f"{row.Category}"
            summaries.append(
                f"{category_name}: GBJ q = {gbj_q} (p = {gbj_p}), GLS q = {gls_q} (p = {gls_p})"
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

    # Additional specific reporting for manuscript diseases
    target_terms = ["Breast", "Obesity", "Heart", "Cognitive", "MCI", "Alzheimer"]

    # Create a mask for phenotypes containing any of the target terms
    mask = pgs["Phenotype"].astype(str).apply(
        lambda x: any(term.lower() in x.lower() for term in target_terms)
    )

    relevant_rows = pgs[mask].copy()
    if not relevant_rows.empty:
        # Sort by fold change to be consistent with "largest inflation" logic or just by name
        relevant_rows = relevant_rows.sort_values("fold_change", ascending=False)

        lines.append("  Specific disease statistics:")
        for row in relevant_rows.itertuples():
             lines.append(
                f"    {row.Phenotype}: p_nominal = {_fmt(row.p_nominal, 3)} -> p_with_pgs = {_fmt(row.p_with_pgs, 3)} "
                f"(fold-change = {_fmt(row.fold_change, 3)})"
            )

    return lines


def summarize_family_history() -> List[str]:
    fam_path = DATA_DIR / "family_phewas.tsv"

    if not fam_path.exists():
        return [
            "Family history validation results not found; expected data/family_phewas.tsv."
        ]

    try:
        df = pd.read_csv(fam_path, sep="\t", low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive logging
        return [f"Error reading family history results: {exc}"]

    if "phenotype" not in df.columns:
        return [
            "Family history validation file missing 'phenotype' column; cannot summarize results."
        ]

    df["phenotype"] = df["phenotype"].astype(str).str.strip()

    lines = ["Family History Validation (Family-based PheWAS):"]
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
        ("Pi Structure", summarize_pi_structure()),
        ("Hudson FST edge decay", summarize_fst_edge_decay()),
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
    download_latest_artifacts()
    run_fresh_cds_pipeline()
    lines = build_report()
    text = "\n".join(lines).strip() + "\n"
    print(text)
    REPORT_PATH.write_text(text)
    print(f"\nSaved report to {REPORT_PATH.relative_to(Path.cwd())}")
    shutil.copy2(REPORT_PATH, DATA_DIR / "replicate_manuscript_statistics.txt")
    print(f"Copied report to {DATA_DIR / 'replicate_manuscript_statistics.txt'}")


if __name__ == "__main__":
    main()
