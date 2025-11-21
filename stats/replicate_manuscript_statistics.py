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
from typing import Iterable, List, Tuple
import tempfile

import shutil
import requests
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats import (
    CDS_identical_model,
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

    print("\n" + "=" * 80)
    print(">>> PIPELINE: REGENERATING CDS DATA FROM RAW .PHY FILES <<<")
    print("=" * 80)

    expected_outputs = [
        "cds_identical_proportions.tsv",
        "gene_inversion_direct_inverted.tsv",
        "region_identical_proportions.tsv",
        "skipped_details.tsv",
    ]

    # If all outputs already exist (either in the repo root or data/), reuse them
    # instead of attempting to regenerate from the raw PHYLIP archives. This is
    # helpful in environments without the large .phy bundles (for example when
    # Git LFS artifacts are not available).
    missing_outputs: list[str] = []
    for filename in expected_outputs:
        root_path = REPO_ROOT / filename
        data_path = DATA_DIR / filename
        if root_path.exists():
            continue
        if data_path.exists():
            print(f"... {filename} found in data/, using cached copy ...")
            continue
        missing_outputs.append(filename)

    if not missing_outputs:
        print("All CDS summary outputs already present; skipping regeneration.")
        return

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
class PiStructureMetrics:
    # Edge vs Middle Metrics (≥40kb)
    # Direct (Group 0)
    dir_flank_mean: float | None
    dir_middle_mean: float | None
    dir_entries: int

    # Inverted (Group 1)
    inv_flank_mean: float | None
    inv_middle_mean: float | None
    inv_entries: int

    # Overall (Group 0 + 1)
    all_flank_mean: float | None
    all_middle_mean: float | None
    all_entries: int

    # Spearman Decay Metrics (≥100kb, first 100kb)
    spearman_overall: SpearmanResult
    spearman_single_inv: SpearmanResult  # Group 1, Recur 0
    spearman_recur_dir: SpearmanResult   # Group 0, Recur 1
    spearman_recur_inv: SpearmanResult   # Group 1, Recur 1
    spearman_single_dir: SpearmanResult  # Group 0, Recur 0

    unique_inversions: int


class _MetricAccumulator:
    """Accumulates Pi data for stats."""
    def __init__(self):
        self.flank_means: list[float] = []
        self.middle_means: list[float] = []

    def add_edge_middle(self, values: np.ndarray) -> None:
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


def _calc_spearman(window_data: list[np.ndarray]) -> SpearmanResult:
    if not window_data:
        return SpearmanResult(None, None, 0)

    # 50 windows per entry (100kb / 2kb)
    window_values = np.concatenate(window_data)
    base_distances = np.arange(0, 100_000, 2_000, dtype=float)
    distances = np.tile(base_distances, len(window_data))

    mask = np.isfinite(window_values)
    if mask.sum() < 2:
        return SpearmanResult(None, None, len(window_data))

    rho_val, p_val = stats.spearmanr(distances[mask], window_values[mask])

    rho = float(rho_val) if np.isfinite(rho_val) else None
    p = float(p_val) if np.isfinite(p_val) else None

    return SpearmanResult(rho, p, len(window_data))


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

    # Load inversion whitelist and recurrence mapping
    try:
        inv_df = _load_inv_properties()
        # Map (chrom, start, end) -> recurrence_flag
        recurrence_map = {
            (str(row.chromosome), int(row.start), int(row.end)): int(row.recurrence_flag)
            for row in inv_df.itertuples(index=False)
        }
    except Exception:
        raise

    # Edge/Middle Accumulators (Group 0, Group 1)
    acc_em_0 = _MetricAccumulator()
    acc_em_1 = _MetricAccumulator()

    # Spearman Accumulators (Group, Recurrence) -> list of window arrays
    # Keys: (0, 0), (0, 1), (1, 0), (1, 1)
    acc_spearman: dict[tuple[int, int], list[np.ndarray]] = {
        (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []
    }

    qualifying_regions: set[tuple[str, int, int]] = set()

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

        values = _parse_values(body_lines)
        if values.size == 0 or not np.isfinite(values).any():
            return

        # Filter Logic:
        # Must have finite bases check?
        # Original code used `finite_bases < 40_000` as a hard reject for everything.
        # Now we have two thresholds.
        # Let's check finite bases relative to thresholds.
        finite_mask = np.isfinite(values)
        finite_bases = int(finite_mask.sum())

        # --- Logic for Edge/Middle (Threshold 40kb) ---
        if finite_bases >= 40_000 and values.size >= 40_000:
            qualifying_regions.add(region)
            if group_id == 0:
                acc_em_0.add_edge_middle(values)
            else:
                acc_em_1.add_edge_middle(values)

        # --- Logic for Spearman (Threshold 100kb) ---
        # "first 100 kbp ... total length greater than 100 kbp"
        if finite_bases >= 100_000 and values.size >= 100_000:
            first_100k = values[:100_000]
            # Reshape to 2kb windows (100,000 / 2,000 = 50 windows)
            reshaped = first_100k.reshape(50, 2_000)
            window_means = np.nanmean(reshaped, axis=1)

            key = (group_id, recur_flag)
            if key in acc_spearman:
                acc_spearman[key].append(window_means)

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
    def _em_stats(acc):
        fm = float(np.mean(acc.flank_means)) if acc.flank_means else None
        mm = float(np.mean(acc.middle_means)) if acc.middle_means else None
        n = len(acc.flank_means)
        return fm, mm, n

    d_fm, d_mm, d_n = _em_stats(acc_em_0)
    i_fm, i_mm, i_n = _em_stats(acc_em_1)

    # Overall Edge/Middle
    acc_em_all = _MetricAccumulator()
    acc_em_all.flank_means = acc_em_0.flank_means + acc_em_1.flank_means
    acc_em_all.middle_means = acc_em_0.middle_means + acc_em_1.middle_means
    a_fm, a_mm, a_n = _em_stats(acc_em_all)

    # --- Compile Spearman Metrics ---
    # 1. Overall (All 4 subgroups)
    all_spearman_data = (
        acc_spearman[(0, 0)] + acc_spearman[(0, 1)] +
        acc_spearman[(1, 0)] + acc_spearman[(1, 1)]
    )
    res_overall = _calc_spearman(all_spearman_data)

    # 2. Single-Inv (G1, R0)
    res_single_inv = _calc_spearman(acc_spearman[(1, 0)])

    # 3. Recur-Dir (G0, R1)
    res_recur_dir = _calc_spearman(acc_spearman[(0, 1)])

    # 4. Recur-Inv (G1, R1)
    res_recur_inv = _calc_spearman(acc_spearman[(1, 1)])

    # 5. Single-Dir (G0, R0)
    res_single_dir = _calc_spearman(acc_spearman[(0, 0)])

    return PiStructureMetrics(
        dir_flank_mean=d_fm,
        dir_middle_mean=d_mm,
        dir_entries=d_n,

        inv_flank_mean=i_fm,
        inv_middle_mean=i_mm,
        inv_entries=i_n,

        all_flank_mean=a_fm,
        all_middle_mean=a_mm,
        all_entries=a_n,

        spearman_overall=res_overall,
        spearman_single_inv=res_single_inv,
        spearman_recur_dir=res_recur_dir,
        spearman_recur_inv=res_recur_inv,
        spearman_single_dir=res_single_dir,

        unique_inversions=len(qualifying_regions),
    )


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

    # Direct
    lines.append(
        f"  [Edge vs Middle] Direct/Group 0 (n={metrics.dir_entries}): "
        f"Flank Mean = {_fmt(metrics.dir_flank_mean)}, Middle Mean = {_fmt(metrics.dir_middle_mean)}."
    )

    # Inverted
    lines.append(
        f"  [Edge vs Middle] Inverted/Group 1 (n={metrics.inv_entries}): "
        f"Flank Mean = {_fmt(metrics.inv_flank_mean)}, Middle Mean = {_fmt(metrics.inv_middle_mean)}."
    )

    # Overall
    lines.append(
        f"  [Edge vs Middle] Overall (n={metrics.all_entries}): "
        f"Flank Mean = {_fmt(metrics.all_flank_mean)}, Middle Mean = {_fmt(metrics.all_middle_mean)}."
    )

    # Spearman Decay
    lines.append("")
    lines.append("Internal decay (Spearman's ρ of diversity vs distance from start for first 100kb, loci ≥100kb):")

    def _fmt_spearman(r, label):
        return (
            f"  {label}: ρ = {_fmt(r.rho, 3)} "
            f"(p = {_fmt(r.p_value, 3)}, n = {_fmt(r.n, 0)})."
        )

    lines.append(_fmt_spearman(metrics.spearman_overall, "Overall (All Consensus 0+1)"))
    lines.append(_fmt_spearman(metrics.spearman_single_inv, "Single-Event Inverted (G1, R0)"))
    lines.append(_fmt_spearman(metrics.spearman_recur_dir, "Recurrent Direct (G0, R1)"))
    lines.append(_fmt_spearman(metrics.spearman_recur_inv, "Recurrent Inverted (G1, R1)"))
    lines.append(_fmt_spearman(metrics.spearman_single_dir, "Single-Event Direct (G0, R0)"))

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

    pairwise_df: pd.DataFrame | None = None
    source_label: str | None = None
    errors: List[str] = []

    cds_input = _resolve_repo_artifact("cds_identical_proportions.tsv")

    if cds_input and cds_input.exists():
        try:
            with _temporary_workdir(cds_input.parent):
                cds_df = CDS_identical_model.load_data()
                res = CDS_identical_model.fit_glm_binom(cds_df, include_covariates=True)
                _, pairwise_df = CDS_identical_model.emms_and_pairs(
                    res, cds_df, include_covariates=True
                )
            source_label = f"loaded from {_relative_to_repo(cds_input)}"
        except SystemExit as exc:
            errors.append(f"CDS GLM exited early: {exc}")
        except Exception as exc:
            errors.append(f"Failed to compute GLM: {exc}")
    else:
         errors.append("cds_identical_proportions.tsv not found (Pipeline failure?)")

    if pairwise_df is None:
        lines.append(
            "  FATAL: CDS GLM inputs unavailable. The pipeline should have generated cds_identical_proportions.tsv."
        )
        lines.extend(f"  {msg}" for msg in errors)
        # Return lines but likely this indicates a critical failure
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
    n_perm = 1_000_000
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
