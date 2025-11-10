"""
Per-inversion metric: how much higher or lower are Hudson FST values at breakpoints vs middle?

We use a flat-ramp-flat (FRF) model: two plateaus at the edges and center,
connected by a linear ramp. A shared block-permutation null preserves spatial
structure while estimating significance.

Sign convention:
  POSITIVE = FST higher at breakpoints
  NEGATIVE = FST higher in middle
"""

from __future__ import annotations
import logging
import sys
import time
import os
import re
import io
import zipfile
import hashlib
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import multiprocessing as mp

import numpy as np
import pandas as pd
import requests

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("per_inversion_breakpoint_test")

# ------------------------- CONFIG -------------------------

GITHUB_REPO = "SauersML/ferromic"
WORKFLOW_NAME = "manual_run_vcf.yml"
ARTIFACT_NAME_FALSTA = "run-vcf-falsta"

# Local paths
OUTDIR = Path("per_inversion_breakpoint_tests")
FALSTA_CACHE = Path("per_site_fst_output.falsta")  # Check current directory first

# Window parameters
WINDOW_SIZE_BP = 1_000
MIN_INVERSION_LENGTH = 0  # Disabled - allow all inversions (can change later if needed)
MIN_WINDOWS_PER_INVERSION = 20

# Permutation parameters
N_PERMUTATIONS = 3_000
DEFAULT_BLOCK_SIZE_WINDOWS = 5  # Fallback block size (windows) if autocorr unavailable
PERMUTATION_BATCH_SIZE = 256    # Number of permutations processed per batch
MAX_INNER_THREADS = max(1, os.cpu_count() or 1)

FRF_MIN_EDGE_WINDOWS = 1
FRF_MIN_MID_WINDOWS = 1

# Permutation validity threshold
MIN_BLOCKS_FOR_PERMUTATION = 5
AUTOCORR_MIN_PAIRS = 5
AUTOCORR_TARGET = 0.3  # target correlation level for block size selection
FRF_CANDIDATE_CHUNK_SIZE = 2048
TOTAL_CPUS = max(1, os.cpu_count() or 1)
_ACTIVE_COUNTER = None
_TOTAL_CPUS_SHARED = TOTAL_CPUS

# Hudson FST constants
EPS_DENOM = 1e-12

# Regex for parsing Hudson FST headers
_RE_HUD = re.compile(
    r">.*?hudson_pairwise_fst.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)",
    re.IGNORECASE,
)


# ------------------------- REPRODUCIBLE SEEDING -------------------------

def stable_seed_from_key(key: str) -> int:
    """
    Generate a stable random seed from a string key.

    Uses MD5 hash to ensure reproducibility across runs and machines,
    unlike Python's hash() which is process-dependent.
    """
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def init_worker(active_counter, total_cpus):
    global _ACTIVE_COUNTER, _TOTAL_CPUS_SHARED
    _ACTIVE_COUNTER = active_counter
    _TOTAL_CPUS_SHARED = total_cpus


@contextmanager
def worker_activity():
    if _ACTIVE_COUNTER is None:
        yield
        return
    with _ACTIVE_COUNTER.get_lock():
        _ACTIVE_COUNTER.value += 1
    try:
        yield
    finally:
        with _ACTIVE_COUNTER.get_lock():
            _ACTIVE_COUNTER.value = max(0, _ACTIVE_COUNTER.value - 1)


def current_inner_threads() -> int:
    if _ACTIVE_COUNTER is None or _TOTAL_CPUS_SHARED <= 1:
        return 1
    with _ACTIVE_COUNTER.get_lock():
        active = max(1, _ACTIVE_COUNTER.value)
    return max(1, min(MAX_INNER_THREADS, _TOTAL_CPUS_SHARED // active))


# ------------------------- GITHUB ARTIFACT DOWNLOAD -------------------------

def download_latest_artifact(
    repo: str,
    workflow_name: str,
    artifact_name: str,
    output_dir: Path
) -> Optional[Path]:
    """Download the latest artifact from a GitHub Actions workflow."""
    log.info(f"Fetching latest artifact '{artifact_name}' from {repo}/{workflow_name}...")

    # Check for GitHub token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.error("GITHUB_TOKEN environment variable required to download artifacts")
        log.error("Cannot proceed without authentication")
        return None

    # Create session with proper headers (matching workflow)
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })

    try:
        # Get latest successful workflow run
        runs_url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_name}/runs"
        response = session.get(runs_url, params={
            "status": "success",
            "exclude_pull_requests": "true",
            "per_page": 1
        })
        response.raise_for_status()

        runs = response.json().get("workflow_runs", [])
        if not runs:
            log.error(f"No successful runs found for workflow {workflow_name}")
            return None

        run = runs[0]
        run_id = run["id"]
        log.info(f"Using artifacts from run {run_id} ({run['html_url']})")

        # Get artifacts from that run
        artifacts_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
        response = session.get(artifacts_url, params={"per_page": 100})
        response.raise_for_status()

        artifacts = {
            artifact["name"]: artifact
            for artifact in response.json().get("artifacts", [])
        }

        if artifact_name not in artifacts:
            log.error(f"Artifact '{artifact_name}' not found in run {run_id}")
            return None

        # Download the artifact
        artifact = artifacts[artifact_name]
        download_url = artifact["archive_download_url"]
        log.info(f"Downloading artifact: {artifact_name}")

        response = session.get(download_url)
        response.raise_for_status()

        # Extract directly from memory (matching workflow approach)
        output_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            extracted_files = []
            for member in zf.namelist():
                if member.endswith(".falsta"):
                    target = output_dir / Path(member).name
                    with target.open("wb") as fh:
                        fh.write(zf.read(member))
                    extracted_files.append(target)
                    log.info(f"Extracted {target.name}")

            if not extracted_files:
                log.error("No .falsta files found in artifact")
                return None

            # Return the FST file specifically
            for f in extracted_files:
                if "fst" in f.name.lower():
                    log.info(f"Successfully downloaded to {f}")
                    return f

            # If no FST file found, return first file
            return extracted_files[0]

    except Exception as e:
        log.error(f"Error downloading artifact: {e}")
        return None


# ------------------------- DATA STRUCTURES -------------------------

@dataclass
class Window:
    """A genomic window with Hudson FST data."""
    position: int  # Midpoint position
    numerator_sum: float
    denominator_sum: float
    n_sites: int

    @property
    def fst(self) -> float:
        """Compute Hudson FST for this window."""
        if self.denominator_sum <= EPS_DENOM:
            return np.nan
        return self.numerator_sum / self.denominator_sum


@dataclass
class Inversion:
    """An inversion with windowed FST data."""
    chrom: str
    start: int
    end: int
    length: int
    windows: List[Window]

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def inv_key(self) -> str:
        return f"{self.chrom}_{self.start}_{self.end}"


@dataclass
class TestResults:
    """Flat-ramp-flat test results for one inversion."""
    inv_key: str
    chrom: str
    start: int
    end: int
    length: int
    n_windows: int
    n_sites: int
    n_blocks: int
    block_size_windows: int
    corr_length_windows: float
    autocorr_max_lag: int
    autocorr_lags_evaluated: int
    frf_permutation_valid: bool

    frf_mu_edge: float
    frf_mu_mid: float
    frf_delta: float          # mu_edge - mu_mid (FST units)
    frf_a: float
    frf_b: float
    frf_p: float


# ------------------------- DATA LOADING -------------------------

def normalize_chromosome(chrom: str) -> str:
    """Normalize chromosome names."""
    chrom = str(chrom).strip().lower()
    if chrom.startswith("chr_"):
        chrom = chrom[4:]
    elif chrom.startswith("chr"):
        chrom = chrom[3:]
    return f"chr{chrom}"


def parse_hudson_header(header: str) -> Optional[Dict[str, Any]]:
    """Parse Hudson FST header to extract coordinates and component type."""
    match = _RE_HUD.search(header)
    if not match:
        return None

    chrom_raw, start_str, end_str = match.groups()
    chrom = normalize_chromosome(chrom_raw)
    start = int(start_str)
    end = int(end_str)

    # Determine if numerator or denominator
    header_lower = header.lower()
    if "numerator" in header_lower:
        component = "numerator"
    elif "denominator" in header_lower:
        component = "denominator"
    else:
        return None

    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "component": component
    }


def parse_data_line(line: str) -> np.ndarray:
    """Parse comma-separated FST data, handling NA values."""
    clean = line.strip()
    if not clean:
        return np.array([], dtype=np.float64)

    # Normalize NA tokens to "nan" for fast parsing
    clean = re.sub(r"\bna\b", "nan", clean, flags=re.IGNORECASE)
    arr = np.fromstring(clean, sep=",", dtype=np.float64)

    if arr.size == 0 and clean:
        # Fallback for pathological strings that np.fromstring cannot parse
        tokens = clean.split(",")
        values = []
        for token in tokens:
            token_stripped = token.strip()
            if not token_stripped or token_stripped.lower() == "na":
                values.append(np.nan)
            else:
                try:
                    values.append(float(token_stripped))
                except ValueError:
                    values.append(np.nan)
        arr = np.array(values, dtype=np.float64)

    return arr


def load_hudson_data(falsta_path: Path) -> List[Inversion]:
    """Load Hudson FST numerator/denominator pairs and create windowed inversions."""
    log.info(f"Loading Hudson FST data from {falsta_path}...")

    # First pass: collect all numerator/denominator pairs
    pairs_by_coords = {}

    current_header = None
    current_data_lines = []

    with open(falsta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Process previous record
                if current_header and current_data_lines:
                    parsed = parse_hudson_header(current_header)
                    if parsed:
                        data = parse_data_line("".join(current_data_lines))
                        key = (parsed["chrom"], parsed["start"], parsed["end"])

                        if key not in pairs_by_coords:
                            pairs_by_coords[key] = {}

                        pairs_by_coords[key][parsed["component"]] = data

                # Start new record
                current_header = line
                current_data_lines = []
            else:
                current_data_lines.append(line)

        # Process last record
        if current_header and current_data_lines:
            parsed = parse_hudson_header(current_header)
            if parsed:
                data = parse_data_line("".join(current_data_lines))
                key = (parsed["chrom"], parsed["start"], parsed["end"])

                if key not in pairs_by_coords:
                    pairs_by_coords[key] = {}

                pairs_by_coords[key][parsed["component"]] = data

    log.info(f"Found {len(pairs_by_coords)} unique coordinate regions")

    # Second pass: create windowed inversions
    inversions = []

    for (chrom, start, end), components in pairs_by_coords.items():
        if "numerator" not in components or "denominator" not in components:
            continue

        numerator = components["numerator"]
        denominator = components["denominator"]

        if len(numerator) != len(denominator):
            log.warning(f"Length mismatch for {chrom}:{start}-{end}, skipping")
            continue

        length = end - start
        if length < MIN_INVERSION_LENGTH:
            continue

        n_sites = len(numerator)
        if n_sites == 0:
            continue

        n_windows = max(1, (length + WINDOW_SIZE_BP - 1) // WINDOW_SIZE_BP)

        site_offsets = np.linspace(0, length, n_sites, endpoint=False)
        window_idx = np.clip((site_offsets // WINDOW_SIZE_BP).astype(int), 0, n_windows - 1)

        finite_num = np.isfinite(numerator)
        finite_den = np.isfinite(denominator)
        valid_mask = finite_num & finite_den
        num_clean = np.where(finite_num, numerator, 0.0)
        den_clean = np.where(finite_den, denominator, 0.0)

        num_sums = np.bincount(window_idx, weights=num_clean, minlength=n_windows)
        den_sums = np.bincount(window_idx, weights=den_clean, minlength=n_windows)
        n_valid_sites = np.bincount(window_idx, weights=valid_mask.astype(np.float64), minlength=n_windows).astype(int)

        window_starts = start + np.arange(n_windows) * WINDOW_SIZE_BP
        window_ends = np.minimum(window_starts + WINDOW_SIZE_BP, end)
        window_positions = ((window_starts + window_ends) // 2).astype(int)

        windows = []
        for idx in range(n_windows):
            den_sum = float(den_sums[idx])
            n_valid = int(n_valid_sites[idx])
            if den_sum > EPS_DENOM and n_valid >= 1:
                windows.append(Window(
                    position=int(window_positions[idx]),
                    numerator_sum=float(num_sums[idx]),
                    denominator_sum=den_sum,
                    n_sites=n_valid
                ))

        # Windows are already in order by construction, but sort for safety
        windows.sort(key=lambda w: w.position)

        if len(windows) >= MIN_WINDOWS_PER_INVERSION:
            inversions.append(Inversion(
                chrom=chrom,
                start=start,
                end=end,
                length=length,
                windows=windows
            ))

    log.info(f"Created {len(inversions)} inversions with ≥{MIN_WINDOWS_PER_INVERSION} windows")
    return inversions


# ------------------------- DISTANCE FOLDING -------------------------

def compute_folded_distances(inversion: Inversion) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute folded distance from breakpoints and prepare data for testing.

    Returns:
        x_normalized: Distance from nearest breakpoint, normalized to [0, 1]
        fst_values: FST value for each window
        weights: Denominator sum (weight) for each window
    """
    positions = np.array([w.position for w in inversion.windows])
    fst_values = np.array([w.fst for w in inversion.windows])
    weights = np.array([w.denominator_sum for w in inversion.windows])

    # Distance from nearest breakpoint
    dist_from_start = positions - inversion.start
    dist_from_end = inversion.end - positions
    dist_from_nearest = np.minimum(dist_from_start, dist_from_end)

    # Normalize to [0, 1]: 0 at breakpoint, 1 at center
    max_dist = inversion.length / 2.0
    x_normalized = dist_from_nearest / max_dist

    return x_normalized, fst_values, weights


# ------------------------- AUTOCORRELATION / BLOCK SIZE -------------------------

def estimate_correlation_length(
    fst: np.ndarray,
    weights: np.ndarray,
    max_global_block_size: int = DEFAULT_BLOCK_SIZE_WINDOWS
) -> Tuple[int, float, int, int]:
    """
    Estimate per-inversion correlation length (in windows) from FST fluctuations.

    Returns:
        block_size_windows: Suggested block size (>=2 windows when possible)
        corr_length_windows: Estimated correlation length (float, NaN if unavailable)
        max_lag_considered: Largest lag evaluated
        lags_evaluated: Number of lag values with valid estimates
    """
    valid = np.isfinite(fst) & np.isfinite(weights)
    values = fst[valid]
    w = weights[valid]
    n = len(values)

    if n == 0:
        return max(1, max_global_block_size), np.nan, 0, 0
    if n == 1:
        block_size = max(1, min(max_global_block_size, 1))
        return block_size, np.nan, 0, 0

    if np.sum(w > 0) > 0:
        mean = np.average(values, weights=w)
    else:
        mean = float(np.mean(values))
    fluct = values - mean

    max_lag_candidate = n - 1
    if max_lag_candidate < 1:
        block_size = max(1, min(max_global_block_size, n))
        return block_size, np.nan, 0, 0

    autocorr_vals: List[float] = []
    lags: List[int] = []

    for lag in range(1, max_lag_candidate + 1):
        v1 = fluct[:-lag]
        v2 = fluct[lag:]
        if len(v1) < AUTOCORR_MIN_PAIRS:
            break

        num = float(np.dot(v1, v2)) / len(v1)
        denom = np.sqrt(
            (np.dot(v1, v1) / len(v1)) *
            (np.dot(v2, v2) / len(v2))
        )
        if denom <= 1e-12:
            corr = 0.0
        else:
            corr = num / denom
        if not np.isfinite(corr):
            corr = 0.0
        corr = float(np.clip(corr, -1.0, 1.0))

        autocorr_vals.append(corr)
        lags.append(lag)

    if not autocorr_vals:
        block_size = max(1, min(max_global_block_size, n))
        return block_size, np.nan, 0 if not lags else lags[-1], 0

    autocorr_array = np.array(autocorr_vals)
    monotone = np.minimum.accumulate(autocorr_array)

    target_idx = np.where(monotone <= AUTOCORR_TARGET)[0]
    if len(target_idx) > 0:
        corr_length = float(lags[target_idx[0]])
    else:
        corr_length = float(lags[-1])

    block_size = int(round(max(1.0, corr_length)))
    block_size = max(1, block_size)
    block_size = min(block_size, n)

    return block_size, corr_length, lags[-1], len(lags)


# ------------------------- PERMUTATION NULL -------------------------

def precompute_block_structure(n: int, block_size: int) -> List[np.ndarray]:
    """
    Precompute block index arrays once.

    Returns list of index arrays, one per block.
    """
    if n <= 0:
        return []
    if n <= block_size:
        return [np.arange(n)]

    n_blocks = (n + block_size - 1) // block_size
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        blocks.append(np.arange(start, end))
    return blocks


def generate_block_permutation_indices(
    blocks: List[np.ndarray],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate one block-permuted index array.

    Shuffles blocks and concatenates them.
    """
    shuffled_blocks = blocks.copy()
    rng.shuffle(shuffled_blocks)
    return np.concatenate(shuffled_blocks)


def build_exhaustive_frf_candidates(
    x_sorted: np.ndarray,
    min_edge_windows: int,
    min_mid_windows: int
) -> Dict[str, np.ndarray]:
    """Enumerate every coherent FRF split in folded-distance order."""
    n = len(x_sorted)
    min_edge_windows = max(1, int(min_edge_windows))
    min_mid_windows = max(1, int(min_mid_windows))

    if n == 0:
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return {
            "edge_end": empty_int,
            "mid_start": empty_int,
            "ramp_start": empty_int,
            "ramp_end": empty_int,
            "a_rel": empty_float,
            "b_rel": empty_float,
        }

    max_edge_end = n - min_mid_windows - 1
    if max_edge_end < min_edge_windows - 1:
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return {
            "edge_end": empty_int,
            "mid_start": empty_int,
            "ramp_start": empty_int,
            "ramp_end": empty_int,
            "a_rel": empty_float,
            "b_rel": empty_float,
        }

    max_mid_start = n - min_mid_windows

    edge_candidates = np.arange(min_edge_windows - 1, max_edge_end + 1, dtype=int)
    mid_counts = max_mid_start - (edge_candidates + 1) + 1
    mid_counts = np.clip(mid_counts, 0, None)
    valid_edges = mid_counts > 0

    if not np.any(valid_edges):
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return {
            "edge_end": empty_int,
            "mid_start": empty_int,
            "ramp_start": empty_int,
            "ramp_end": empty_int,
            "a_rel": empty_float,
            "b_rel": empty_float,
        }

    edge_candidates = edge_candidates[valid_edges]
    mid_counts = mid_counts[valid_edges]

    edge_end_arr = np.repeat(edge_candidates, mid_counts)

    mid_segments = [
        np.arange(edge + 1, edge + 1 + count, dtype=int)
        for edge, count in zip(edge_candidates, mid_counts)
    ]
    mid_start_arr = np.concatenate(mid_segments, dtype=int) if mid_segments else np.array([], dtype=int)

    ramp_start_arr = edge_end_arr + 1
    ramp_end_arr = mid_start_arr

    a_rel_arr = x_sorted[edge_end_arr].astype(float)
    b_rel_arr = x_sorted[mid_start_arr].astype(float)

    return {
        "edge_end": edge_end_arr,
        "mid_start": mid_start_arr,
        "ramp_start": ramp_start_arr,
        "ramp_end": ramp_end_arr,
        "a_rel": a_rel_arr,
        "b_rel": b_rel_arr,
    }


def _prefix_with_zero(arr: np.ndarray) -> np.ndarray:
    """Prefix sum with a leading zero column for 2D arrays."""
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    out = np.zeros((arr.shape[0], arr.shape[1] + 1), dtype=arr.dtype)
    np.cumsum(arr, axis=1, out=out[:, 1:])
    return out


def run_frf_search(
    fst_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    x_sorted: np.ndarray,
    candidates: Dict[str, np.ndarray],
    half_length_bp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the exhaustive FRF search for one or more samples."""
    fst_matrix = np.asarray(fst_matrix, dtype=float)
    weight_matrix = np.asarray(weight_matrix, dtype=float)

    if fst_matrix.ndim != 2 or weight_matrix.ndim != 2:
        raise ValueError("fst_matrix and weight_matrix must be 2-dimensional")
    if fst_matrix.shape != weight_matrix.shape:
        raise ValueError("fst_matrix and weight_matrix must have the same shape")

    n_samples, n_windows = fst_matrix.shape
    if n_windows == 0 or candidates["edge_end"].size == 0 or half_length_bp <= 0:
        nan = np.full(n_samples, np.nan)
        return nan, nan, nan, nan, nan

    x_row = x_sorted[np.newaxis, :]
    wf = weight_matrix * fst_matrix
    wf2 = weight_matrix * (fst_matrix ** 2)
    wx = weight_matrix * x_row
    wx2 = weight_matrix * (x_row ** 2)
    wfx = weight_matrix * fst_matrix * x_row

    prefix_w = _prefix_with_zero(weight_matrix)
    prefix_wf = _prefix_with_zero(wf)
    prefix_wf2 = _prefix_with_zero(wf2)
    prefix_wx = _prefix_with_zero(wx)
    prefix_wx2 = _prefix_with_zero(wx2)
    prefix_wfx = _prefix_with_zero(wfx)

    total_w = prefix_w[:, -1]
    total_wf = prefix_wf[:, -1]
    total_wf2 = prefix_wf2[:, -1]

    edge_end = candidates["edge_end"]
    mid_start = candidates["mid_start"]
    ramp_start = candidates["ramp_start"]
    ramp_end = candidates["ramp_end"]
    a_rel = candidates["a_rel"]
    b_rel = candidates["b_rel"]

    n_candidates = edge_end.size
    eps = 1e-12
    row_idx = np.arange(n_samples)

    best_sse = np.full(n_samples, np.inf)
    best_mu_edge = np.full(n_samples, np.nan)
    best_mu_mid = np.full(n_samples, np.nan)
    best_delta = np.full(n_samples, np.nan)
    best_a_bp = np.full(n_samples, np.nan)
    best_b_bp = np.full(n_samples, np.nan)

    chunk_size = max(1, min(FRF_CANDIDATE_CHUNK_SIZE, n_candidates))

    for start in range(0, n_candidates, chunk_size):
        end = min(start + chunk_size, n_candidates)
        edge_chunk = edge_end[start:end]
        mid_chunk = mid_start[start:end]
        ramp_start_chunk = ramp_start[start:end]
        ramp_end_chunk = ramp_end[start:end]
        a_chunk = a_rel[start:end]
        b_chunk = b_rel[start:end]

        edge_idx = edge_chunk + 1
        mid_idx = mid_chunk

        edge_sum_w = np.take(prefix_w, edge_idx, axis=1)
        edge_sum_wf = np.take(prefix_wf, edge_idx, axis=1)
        edge_sum_wf2 = np.take(prefix_wf2, edge_idx, axis=1)

        mid_prefix_w = np.take(prefix_w, mid_idx, axis=1)
        mid_prefix_wf = np.take(prefix_wf, mid_idx, axis=1)
        mid_prefix_wf2 = np.take(prefix_wf2, mid_idx, axis=1)

        mid_sum_w = total_w[:, None] - mid_prefix_w
        mid_sum_wf = total_wf[:, None] - mid_prefix_wf
        mid_sum_wf2 = total_wf2[:, None] - mid_prefix_wf2

        mu_edge = edge_sum_wf / np.maximum(edge_sum_w, eps)
        mu_mid = mid_sum_wf / np.maximum(mid_sum_w, eps)

        edge_sse = edge_sum_wf2 - np.where(edge_sum_w > eps, (edge_sum_wf ** 2) / np.maximum(edge_sum_w, eps), 0.0)
        mid_sse = mid_sum_wf2 - np.where(mid_sum_w > eps, (mid_sum_wf ** 2) / np.maximum(mid_sum_w, eps), 0.0)

        ramp_sum_w = np.take(prefix_w, ramp_end_chunk, axis=1) - np.take(prefix_w, ramp_start_chunk, axis=1)
        ramp_sum_wf = np.take(prefix_wf, ramp_end_chunk, axis=1) - np.take(prefix_wf, ramp_start_chunk, axis=1)
        ramp_sum_wf2 = np.take(prefix_wf2, ramp_end_chunk, axis=1) - np.take(prefix_wf2, ramp_start_chunk, axis=1)
        ramp_sum_wx = np.take(prefix_wx, ramp_end_chunk, axis=1) - np.take(prefix_wx, ramp_start_chunk, axis=1)
        ramp_sum_wx2 = np.take(prefix_wx2, ramp_end_chunk, axis=1) - np.take(prefix_wx2, ramp_start_chunk, axis=1)
        ramp_sum_wfx = np.take(prefix_wfx, ramp_end_chunk, axis=1) - np.take(prefix_wfx, ramp_start_chunk, axis=1)

        delta = np.maximum(b_chunk - a_chunk, 1e-6)
        slope = (mu_mid - mu_edge) / delta[np.newaxis, :]
        intercept = mu_edge - slope * a_chunk[np.newaxis, :]

        ramp_sse = (
            ramp_sum_wf2
            - 2 * intercept * ramp_sum_wf
            - 2 * slope * ramp_sum_wfx
            + (intercept ** 2) * ramp_sum_w
            + 2 * intercept * slope * ramp_sum_wx
            + (slope ** 2) * ramp_sum_wx2
        )

        total_sse = edge_sse + mid_sse + ramp_sse

        chunk_best_idx = np.argmin(total_sse, axis=1)
        chunk_best_sse = total_sse[row_idx, chunk_best_idx]
        update_mask = chunk_best_sse < best_sse

        if np.any(update_mask):
            best_sse[update_mask] = chunk_best_sse[update_mask]

            selected_mu_edge = mu_edge[row_idx, chunk_best_idx]
            selected_mu_mid = mu_mid[row_idx, chunk_best_idx]
            best_mu_edge[update_mask] = selected_mu_edge[update_mask]
            best_mu_mid[update_mask] = selected_mu_mid[update_mask]
            best_delta[update_mask] = (selected_mu_edge - selected_mu_mid)[update_mask]

            a_bp_vals = a_chunk[chunk_best_idx] * half_length_bp
            b_bp_vals = b_chunk[chunk_best_idx] * half_length_bp
            best_a_bp[update_mask] = a_bp_vals[update_mask]
            best_b_bp[update_mask] = b_bp_vals[update_mask]

    return best_mu_edge, best_mu_mid, best_delta, best_a_bp, best_b_bp


def permutation_test(
    inversion: Inversion,
    n_permutations: int,
    block_size: int
) -> TestResults:
    """
    Run the flat-ramp-flat test with a shared block-permutation null distribution.

    Optimized version:
    - Pre-filters data once for statistics computation
    - Pre-allocates null arrays
    - Precomputes block structure on FULL spatial grid (preserves spatial correlation)
    - Precomputes FRF region indices
    - Vectorizes FRF across permutations
    """
    log.info(f"Testing {inversion.inv_key} ({inversion.n_windows} windows)...")

    inner_threads = current_inner_threads()

    # Prepare data (full arrays, may contain NaN)
    x_full, fst_full, w_full = compute_folded_distances(inversion)
    n_all = len(x_full)

    # Identify valid windows
    valid = np.isfinite(x_full) & np.isfinite(fst_full) & np.isfinite(w_full)
    n_valid = int(np.sum(valid))

    # Compressed valid-only arrays for computing statistics
    x_v = x_full[valid]
    fst_v = fst_full[valid]
    w_v = w_full[valid]

    # If insufficient data, tests will return NaN - don't skip the inversion
    if n_valid < 3:
        log.warning(f"  Very few valid windows ({n_valid}) for {inversion.inv_key}, results may be NaN")

    half_length = inversion.length / 2.0

    # Estimate correlation length -> per-inversion block size
    block_size_inv, corr_length_est, autocorr_max_lag, autocorr_lags_eval = estimate_correlation_length(
        fst_v, w_v, max_global_block_size=block_size
    )

    # Sort by folded distance for exhaustive FRF search
    order = np.argsort(x_v)
    x_sorted = x_v[order]
    fst_sorted = fst_v[order]
    w_sorted = w_v[order]

    # Enumerate all coherent FRF candidates
    frf_candidates = build_exhaustive_frf_candidates(
        x_sorted,
        FRF_MIN_EDGE_WINDOWS,
        FRF_MIN_MID_WINDOWS,
    )

    # Observed FRF statistics (computed on valid data only)
    obs_mu_edge_arr, obs_mu_mid_arr, obs_delta_arr, obs_a_bp_arr, obs_b_bp_arr = run_frf_search(
        fst_sorted[np.newaxis, :],
        w_sorted[np.newaxis, :],
        x_sorted,
        frf_candidates,
        half_length,
    )

    obs_frf_edge = float(obs_mu_edge_arr[0])
    obs_frf_mid = float(obs_mu_mid_arr[0])
    obs_frf_delta = float(obs_delta_arr[0])
    obs_frf_a = float(obs_a_bp_arr[0])
    obs_frf_b = float(obs_b_bp_arr[0])

    # Precompute block structure on FULL spatial grid (not compressed)
    # This preserves spatial correlation structure in the null
    blocks = precompute_block_structure(n_all, block_size_inv)
    n_blocks = len(blocks)
    enough_blocks = n_blocks >= MIN_BLOCKS_FOR_PERMUTATION

    has_candidates = frf_candidates["edge_end"].size > 0

    can_permute = (
        enough_blocks
        and n_valid > 0
        and has_candidates
    )
    null_frf_delta = None
    p_frf = np.nan

    if can_permute:
        # Create mapping from original (full) indices to compressed (valid-only) indices
        orig_to_comp = np.full(n_all, -1, dtype=int)
        orig_to_comp[np.where(valid)[0]] = np.arange(n_valid)

        rng = np.random.default_rng(seed=stable_seed_from_key(inversion.inv_key))
        batch_size = min(PERMUTATION_BATCH_SIZE, n_permutations)
        batch_count = math.ceil(n_permutations / batch_size)
        null_frf_delta = np.empty(n_permutations, dtype=float)

        executor = None
        futures = []
        if inner_threads > 1:
            executor = ThreadPoolExecutor(max_workers=inner_threads)

        try:
            generated = 0
            batch_index = 0

            while generated < n_permutations:
                batch = min(batch_size, n_permutations - generated)
                perm_indices = np.empty((batch, n_valid), dtype=int)

                for j in range(batch):
                    idx_full = generate_block_permutation_indices(blocks, rng)
                    idx_comp = orig_to_comp[idx_full]
                    perm_indices[j, :] = idx_comp[idx_comp >= 0]

                fst_perm_batch = fst_v[perm_indices][:, order]
                w_perm_batch = w_v[perm_indices][:, order]

                if executor:
                    fut = executor.submit(
                        run_frf_search,
                        fst_perm_batch,
                        w_perm_batch,
                        x_sorted,
                        frf_candidates,
                        half_length,
                    )
                    futures.append((batch_index, generated, batch, fut))
                else:
                    _, _, batch_deltas, _, _ = run_frf_search(
                        fst_perm_batch,
                        w_perm_batch,
                        x_sorted,
                        frf_candidates,
                        half_length,
                    )
                    null_frf_delta[generated:generated + batch] = batch_deltas

                generated += batch
                batch_index += 1

            if executor:
                for batch_idx, offset, batch_len, fut in sorted(futures, key=lambda x: x[0]):
                    _, _, batch_deltas, _, _ = fut.result()
                    null_frf_delta[offset:offset + batch_len] = batch_deltas
        finally:
            if executor:
                executor.shutdown(wait=True)

        def compute_p_value(obs, null_array):
            if not np.isfinite(obs):
                return np.nan
            valid_null = null_array[np.isfinite(null_array)]
            if len(valid_null) == 0:
                return np.nan
            count = np.sum(np.abs(valid_null) >= np.abs(obs))
            return (count + 1) / (len(valid_null) + 1)

        p_frf = compute_p_value(obs_frf_delta, null_frf_delta)
    else:
        if not enough_blocks:
            log.warning(
                f"  Only {n_blocks} block(s); need >= {MIN_BLOCKS_FOR_PERMUTATION} for permutation p-values. "
                "Reporting FRF effect size only."
            )
        elif not has_candidates:
            log.warning("  No valid FRF candidates (insufficient edge/mid coverage); p-value unavailable.")
        elif n_valid == 0:
            log.warning("  No valid windows after filtering; p-value unavailable.")

    # Compile results
    n_sites_total = sum(w.n_sites for w in inversion.windows)

    return TestResults(
        inv_key=inversion.inv_key,
        chrom=inversion.chrom,
        start=inversion.start,
        end=inversion.end,
        length=inversion.length,
        n_windows=inversion.n_windows,
        n_sites=n_sites_total,
        n_blocks=n_blocks,
        block_size_windows=block_size_inv,
        corr_length_windows=corr_length_est,
        autocorr_max_lag=autocorr_max_lag,
        autocorr_lags_evaluated=autocorr_lags_eval,
        frf_permutation_valid=can_permute,
        frf_mu_edge=obs_frf_edge,
        frf_mu_mid=obs_frf_mid,
        frf_delta=obs_frf_delta,
        frf_a=obs_frf_a,
        frf_b=obs_frf_b,
        frf_p=p_frf,
    )


def permutation_test_worker(inversion, n_permutations, block_size):
    with worker_activity():
        return permutation_test(inversion, n_permutations, block_size)


# ------------------------- MAIN -------------------------

def main():
    log.info("=" * 80)
    log.info("Per-Inversion Breakpoint vs Middle FST Test")
    log.info("=" * 80)
    log.info("")
    log.info("Sign convention:")
    log.info("  POSITIVE = FST higher at breakpoints")
    log.info("  NEGATIVE = FST higher in middle")
    log.info("")

    # Look for falsta file
    falsta_path = None

    # Check cache locations
    if FALSTA_CACHE.exists():
        log.info(f"Found cached FST data: {FALSTA_CACHE}")
        falsta_path = FALSTA_CACHE
    elif (OUTDIR / "per_site_fst_output.falsta").exists():
        falsta_path = OUTDIR / "per_site_fst_output.falsta"
        log.info(f"Found FST data: {falsta_path}")

    # Try to download if not found locally
    if not falsta_path:
        log.info("FST data not found locally, attempting to download from GitHub Actions...")
        falsta_path = download_latest_artifact(
            GITHUB_REPO,
            WORKFLOW_NAME,
            ARTIFACT_NAME_FALSTA,
            OUTDIR
        )

    if not falsta_path:
        log.error("")
        log.error("=" * 80)
        log.error("FST DATA NOT FOUND")
        log.error("=" * 80)
        log.error("")
        log.error("Please download per_site_fst_output.falsta from GitHub Actions:")
        log.error(f"  1. Go to: https://github.com/{GITHUB_REPO}/actions/workflows/{WORKFLOW_NAME}")
        log.error(f"  2. Click on the most recent successful run")
        log.error(f"  3. Download the '{ARTIFACT_NAME_FALSTA}' artifact")
        log.error(f"  4. Extract per_site_fst_output.falsta to current directory")
        log.error("")
        sys.exit(1)

    # Load data
    inversions = load_hudson_data(falsta_path)

    if not inversions:
        log.error("No inversions loaded. Exiting.")
        sys.exit(1)

    log.info(f"Loaded {len(inversions)} inversions for testing")
    log.info(f"Running {N_PERMUTATIONS} permutations with per-inversion block sizes (fallback={DEFAULT_BLOCK_SIZE_WINDOWS} windows)")
    log.info(f"Minimum blocks required for FRF p-values: {MIN_BLOCKS_FOR_PERMUTATION}")
    log.info("")

    # Create output directory
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Run tests in parallel
    all_results = []
    start_time = time.time()

    # Determine number of workers
    n_workers = min(os.cpu_count() or 1, len(inversions))
    log.info(f"Using {n_workers} parallel workers")
    log.info("")

    active_counter = mp.Value('i', 0)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(active_counter, TOTAL_CPUS)
    ) as executor:
        # Submit all jobs
        future_to_inv = {
            executor.submit(permutation_test_worker, inv, N_PERMUTATIONS, DEFAULT_BLOCK_SIZE_WINDOWS): inv
            for inv in inversions
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_inv):
            inv = future_to_inv[future]
            completed += 1

            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)

                    # Log summary
                    log.info(f"[{completed}/{len(inversions)}] {result.inv_key}")
                    if np.isfinite(result.frf_p):
                        p_text = f"{result.frf_p:.4f}"
                    elif not result.frf_permutation_valid:
                        p_text = f"NA (blocks={result.n_blocks})"
                    else:
                        p_text = "NA"
                    frf_info = f"Δ={result.frf_delta:+.4f}, p={p_text}"
                    if np.isfinite(result.frf_a) and np.isfinite(result.frf_b):
                        frf_info += f" [ramp: {result.frf_a/1000:.0f}-{result.frf_b/1000:.0f}kb]"
                    frf_info += f" [block_size={result.block_size_windows}, blocks={result.n_blocks}]"
                    log.info(f"  Flat-ramp-flat: {frf_info}")
                    log.info("")
                else:
                    log.warning(f"[{completed}/{len(inversions)}] {inv.inv_key} - Insufficient data")

            except Exception as e:
                log.error(f"[{completed}/{len(inversions)}] {inv.inv_key} - Error: {e}", exc_info=True)

    elapsed = time.time() - start_time

    # Save results
    if all_results:
        df = pd.DataFrame([vars(r) for r in all_results])

        # Sort by strongest evidence (lowest FRF p-value, NaNs last)
        df = df.sort_values('frf_p', na_position='last')

        output_tsv = OUTDIR / "per_inversion_breakpoint_test_results.tsv"
        df.to_csv(output_tsv, sep='\t', index=False)

        log.info("")
        log.info("=" * 80)
        log.info("SUMMARY")
        log.info("=" * 80)
        log.info(f"Total inversions tested: {len(all_results)}")
        log.info(f"Time elapsed: {elapsed:.1f}s ({elapsed/len(all_results):.1f}s per inversion)")
        log.info("")

        n_valid_perm = int(np.sum(df['frf_permutation_valid']))
        n_invalid_perm = len(df) - n_valid_perm

        sig_mask = (df['frf_p'] < 0.05).fillna(False)
        sig_frf = int(sig_mask.sum())
        valid_denominator = max(n_valid_perm, 1)

        log.info(f"Valid permutation tests (>= {MIN_BLOCKS_FOR_PERMUTATION} blocks): {n_valid_perm}")
        if n_invalid_perm > 0:
            log.info(f"Inversions without valid permutations (p=NA): {n_invalid_perm}")
        log.info(f"Significant at p<0.05:")
        log.info(f"  Flat-ramp-flat: {sig_frf:3d} ({sig_frf/valid_denominator*100:5.1f}% of testable inversions)")

        # Direction consistency among significant results
        if sig_mask.sum() > 0:
            edges_higher = np.sum(df.loc[sig_mask, 'frf_delta'] > 0)
            middle_higher = np.sum(df.loc[sig_mask, 'frf_delta'] < 0)

            log.info(f"Among FRF-significant inversions (n={sig_mask.sum()}):")
            log.info(f"  Edges higher:  {edges_higher}")
            log.info(f"  Middle higher: {middle_higher}")
            log.info("")

        log.info(f"Results saved to: {output_tsv}")
        log.info("")

    log.info("Done!")


if __name__ == "__main__":
    main()
