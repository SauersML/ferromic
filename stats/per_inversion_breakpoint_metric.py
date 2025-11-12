from __future__ import annotations

import logging
import sys
import os
import time
import threading
import atexit

# Ensure BLAS-style math libraries do not oversubscribe threads when this module
# is imported. Environment variables are respected only if they are unset so
# that callers can still override the behaviour explicitly.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import re
import io
import zipfile
import hashlib
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Iterable, Sequence
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Empty
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is optional but recommended
    njit = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("frf_meta_analysis")

# ------------------------- CONFIG -------------------------

GITHUB_REPO = "SauersML/ferromic"
WORKFLOW_NAME = "manual_run_vcf.yml"
ARTIFACT_NAME_FALSTA = "run-vcf-falsta"

OUTDIR = Path("per_inversion_breakpoint_tests")
FALSTA_CACHE = Path("per_site_fst_output.falsta")

INV_PROPERTIES_PATH = Path("inv_properties.tsv")
CHR_COL_INV = "Chromosome"
START_COL_INV = "Start"
END_COL_INV = "End"
STATUS_COL = "0_single_1_recur_consensus"

WINDOW_SIZE_BP = 1_000
MIN_INVERSION_LENGTH = 0
MIN_WINDOWS_PER_INVERSION = 25

N_PERMUTATIONS = 3_000
DEFAULT_BLOCK_SIZE_WINDOWS = 10
USE_GLOBAL_AUTOCORR_BLOCK_SIZE_OVERRIDE = False
GLOBAL_AUTOCORR_BLOCK_SIZE_WINDOWS = 3
PERMUTATION_CHUNK_SIZE = 8192

FRF_MIN_EDGE_WINDOWS = 1
FRF_MIN_MID_WINDOWS = 1

# ------------------------- ROTATION NULL CONFIG -------------------------

# Rotate by a random offset k in {1,...,n-1}; optionally also flip orientation.
ROTATION_ALLOW_FLIP = False

# Disallow seams too close to the ends of the ordered series (in windows).
# 0 means allow all k. Small integers like 1–2 are a cautious choice.
ROTATION_SEAM_EXCLUDE_WINDOWS = 0

# If True and n-1 is small, sample each valid offset without replacement per chunk until exhausted.
ROTATION_ENUMERATE_OFFSETS = True

# If True, rotate residuals around the fitted FRF trend and then add the trend back.
# This guards against broad nonstationarity but is optional; default off.
ROTATION_USE_RESIDUALS = False

FRF_CANDIDATE_CHUNK_SIZE = 8192
# Upper bound for the number of prefix "cells" (rows * columns) we materialize
# at once when evaluating cached FRF candidates. This caps transient allocations
# such as prefix arrays so that very wide inversions do not request multi-GB
# buffers inside worker processes.
FRF_PREFIX_CHUNK_CELLS = 16_000_000

META_PERMUTATIONS = 3_000_000_000
META_PERM_CHUNK = 1000
META_PERM_BASE_SEED = 2025

TOTAL_CPUS = max(1, os.cpu_count() or 1)
EPS_DENOM = 1e-12

AUTOCORR_MIN_PAIRS = 3
AUTOCORR_TARGET = 0.4

_RE_HUD = re.compile(
    r">.*?hudson_pairwise_fst.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)",
    re.IGNORECASE,
)

# ------------------------- REPRODUCIBLE SEEDING -------------------------

def stable_seed_from_key(key: str) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

# ------------------------- GITHUB ARTIFACT DOWNLOAD -------------------------

def download_latest_artifact(
    repo: str,
    workflow_name: str,
    artifact_name: str,
    output_dir: Path
) -> Optional[Path]:
    log.info(f"Fetching latest artifact '{artifact_name}' from {repo}/{workflow_name}...")

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.error("GITHUB_TOKEN environment variable required to download artifacts")
        return None

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })

    runs_url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_name}/runs"
    response = session.get(
        runs_url,
        params={
            "status": "success",
            "per_page": 1,
        },
    )
    if not response.ok:
        log.error("Failed to fetch workflow runs")
        return None

    runs = response.json().get("workflow_runs", [])
    if not runs:
        log.error(f"No successful runs found for workflow {workflow_name}")
        return None

    run = runs[0]
    run_id = run["id"]
    log.info(f"Using artifacts from run {run_id} ({run.get('html_url', '')})")

    artifacts_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
    response = session.get(artifacts_url, params={"per_page": 100})
    if not response.ok:
        log.error("Failed to list artifacts")
        return None

    artifacts = {a["name"]: a for a in response.json().get("artifacts", [])}
    if artifact_name not in artifacts:
        log.error(f"Artifact '{artifact_name}' not found in run {run_id}")
        return None

    artifact = artifacts[artifact_name]
    download_url = artifact["archive_download_url"]

    response = session.get(download_url)
    if not response.ok:
        log.error("Failed to download artifact archive")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        extracted_files: List[Path] = []
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

    for f in extracted_files:
        if "fst" in f.name.lower():
            log.info(f"Using FST file {f}")
            return f

    return extracted_files[0]

# ------------------------- DATA STRUCTURES -------------------------


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"

@dataclass
class Window:
    position: int
    numerator_sum: float
    denominator_sum: float
    n_sites: int
    @property
    def fst(self) -> float:
        if self.denominator_sum <= EPS_DENOM:
            return np.nan
        return self.numerator_sum / self.denominator_sum

@dataclass
class Inversion:
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
class PermutationPlan:
    inv_key: str
    base_seed: int
    n_permutations: int
    chunk_size: int
    n_valid: int
    order: np.ndarray
    fst_values: np.ndarray
    weight_values: np.ndarray
    weight_sorted: np.ndarray
    x_sorted: np.ndarray
    candidates: Dict[str, np.ndarray]
    half_length: float
    observed_delta: float
    valid_offsets: np.ndarray
    allow_flip: bool
    use_residuals: bool
    rotation_basis: np.ndarray
    trend_physical: Optional[np.ndarray]
    plan_id: int = -1
    offset_step: int = 1
    offset_seed: int = 0

    @property
    def n_chunks(self) -> int:
        if self.n_permutations <= 0:
            return 0
        return (self.n_permutations + self.chunk_size - 1) // self.chunk_size


@dataclass
class PreparedInversion:
    result: "FRFResult"
    plan: Optional[PermutationPlan]


@dataclass
class PermutationChunkResult:
    inv_key: str
    chunk_index: int
    n_finite: int
    sum_delta: float
    sum_sq_delta: float
    count_ge_observed: int


@dataclass(frozen=True)
class SharedArrayDescriptor:
    name: str
    shape: Tuple[int, ...]
    dtype_str: str


def _create_shared_descriptor(
    array: Optional[np.ndarray],
) -> Tuple[Optional[SharedArrayDescriptor], Optional[shared_memory.SharedMemory]]:
    if array is None:
        return None, None
    arr = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    return SharedArrayDescriptor(shm.name, arr.shape, arr.dtype.str), shm


def _attach_shared_descriptor(descriptor: SharedArrayDescriptor) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(name=descriptor.name)
    array = np.ndarray(descriptor.shape, dtype=np.dtype(descriptor.dtype_str), buffer=shm.buf)
    return array, shm


@dataclass
class FRFResult:
    inv_key: str
    chrom: str
    start: int
    end: int
    length: int
    n_windows: int
    n_sites: int
    block_size_windows: int
    n_blocks: int
    frf_mu_edge: float
    frf_mu_mid: float
    frf_delta: float
    frf_null_delta_mean: float
    frf_a: float
    frf_b: float
    frf_var_delta: float
    frf_se_delta: float
    usable_for_meta: bool


def estimate_inversion_workload(inv: Inversion) -> Tuple[int, int, int, int]:
    """Estimate work for scheduling purposes.

    We approximate cost by prioritizing inversions with the most valid sites,
    windows, and physical span (proxy for candidate count).
    """

    valid_windows = sum(
        1
        for w in inv.windows
        if w.denominator_sum > EPS_DENOM and w.n_sites > 0
    )
    total_sites = sum(w.n_sites for w in inv.windows)
    approx_candidates = max(0, valid_windows * max(0, valid_windows - 1) // 2)
    return (
        approx_candidates,
        total_sites,
        inv.n_windows,
        inv.length,
    )


def _choose_coprime_step(length: int, rng: np.random.Generator) -> int:
    if length <= 1:
        return 1
    max_attempts = min(32, length)
    for _ in range(max_attempts):
        candidate = int(rng.integers(1, length))
        if math.gcd(candidate, length) == 1:
            return candidate
    # Fallback to 1 if we somehow fail to sample a coprime step.
    return 1


def _offset_positions(
    cycle_size: int,
    chunk_start: int,
    size: int,
    *,
    offset_seed: int,
    offset_step: int,
) -> np.ndarray:
    if cycle_size <= 0 or size <= 0:
        return np.array([], dtype=np.int64)
    start = int(offset_seed % cycle_size)
    step = int(offset_step % cycle_size) or 1
    draws = np.arange(chunk_start, chunk_start + size, dtype=np.int64)
    return (start + step * draws) % cycle_size

# ------------------------- DATA LOADING -------------------------

def normalize_chromosome(chrom: str) -> str:
    c = str(chrom).strip().lower()
    if c.startswith("chr_"):
        c = c[4:]
    elif c.startswith("chr"):
        c = c[3:]
    return f"chr{c}"

def parse_hudson_header(header: str) -> Optional[Dict[str, Any]]:
    m = _RE_HUD.search(header)
    if not m:
        return None
    chrom_raw, start_str, end_str = m.groups()
    chrom = normalize_chromosome(chrom_raw)
    start = int(start_str)
    end = int(end_str)
    h = header.lower()
    if "numerator" in h:
        component = "numerator"
    elif "denominator" in h:
        component = "denominator"
    else:
        return None
    return {"chrom": chrom, "start": start, "end": end, "component": component}

def parse_data_line(line: str) -> np.ndarray:
    clean = line.strip()
    if not clean:
        return np.array([], dtype=np.float64)
    clean = re.sub(r"\bna\b", "nan", clean, flags=re.IGNORECASE)
    arr = np.fromstring(clean, sep=",", dtype=np.float64)
    if arr.size == 0 and clean:
        tokens = clean.split(",")
        vals: List[float] = []
        for t in tokens:
            s = t.strip()
            if not s or s.lower() == "na":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(s))
                except ValueError:
                    vals.append(np.nan)
        arr = np.array(vals, dtype=np.float64)
    return arr

def load_hudson_data(falsta_path: Path) -> List[Inversion]:
    log.info(f"Loading Hudson FST data from {falsta_path}...")
    pairs_by_coords: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}
    current_header: Optional[str] = None
    current_data_lines: List[str] = []
    with falsta_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header and current_data_lines:
                    parsed = parse_hudson_header(current_header)
                    if parsed:
                        data = parse_data_line(",".join(current_data_lines))
                        key = (parsed["chrom"], parsed["start"], parsed["end"])
                        if key not in pairs_by_coords:
                            pairs_by_coords[key] = {}
                        pairs_by_coords[key][parsed["component"]] = data
                current_header = line
                current_data_lines = []
            else:
                current_data_lines.append(line)
        if current_header and current_data_lines:
            parsed = parse_hudson_header(current_header)
            if parsed:
                data = parse_data_line(",".join(current_data_lines))
                key = (parsed["chrom"], parsed["start"], parsed["end"])
                if key not in pairs_by_coords:
                    pairs_by_coords[key] = {}
                pairs_by_coords[key][parsed["component"]] = data
    log.info(f"Found {len(pairs_by_coords)} unique coordinate regions")

    inversions: List[Inversion] = []
    for (chrom, start, end), components in pairs_by_coords.items():
        if "numerator" not in components or "denominator" not in components:
            continue
        numerator = components["numerator"]
        denominator = components["denominator"]
        if len(numerator) != len(denominator):
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
        
        num_clean = np.where(valid_mask, numerator, 0.0)
        den_clean = np.where(valid_mask, denominator, 0.0)

        num_sums = np.bincount(window_idx, weights=num_clean, minlength=n_windows)
        den_sums = np.bincount(window_idx, weights=den_clean, minlength=n_windows)
        n_valid_sites = np.bincount(window_idx, weights=valid_mask.astype(np.float64), minlength=n_windows).astype(int)

        window_starts = start + np.arange(n_windows) * WINDOW_SIZE_BP
        window_ends = np.minimum(window_starts + WINDOW_SIZE_BP, end)
        window_positions = ((window_starts + window_ends) // 2).astype(int)

        windows: List[Window] = []
        for idx in range(n_windows):
            den_sum = float(den_sums[idx])
            n_valid = int(n_valid_sites[idx])
            if den_sum > EPS_DENOM and n_valid >= 1:
                windows.append(Window(
                    position=int(window_positions[idx]),
                    numerator_sum=float(num_sums[idx]),
                    denominator_sum=den_sum,
                    n_sites=n_valid,
                ))
        windows.sort(key=lambda w: w.position)
        if len(windows) >= MIN_WINDOWS_PER_INVERSION:
            inversions.append(Inversion(chrom=chrom, start=start, end=end, length=length, windows=windows))

    log.info(f"Created {len(inversions)} inversions with ≥{MIN_WINDOWS_PER_INVERSION} windows")
    return inversions

# ------------------------- FRF UTILITIES -------------------------

def compute_folded_distances(inversion: Inversion) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array([w.position for w in inversion.windows], dtype=float)
    fst_values = np.array([w.fst for w in inversion.windows], dtype=float)
    weights = np.array([w.denominator_sum for w in inversion.windows], dtype=float)
    dist_from_start = positions - float(inversion.start)
    dist_from_end = float(inversion.end) - positions
    dist_from_nearest = np.minimum(dist_from_start, dist_from_end)
    max_dist = inversion.length / 2.0
    if max_dist <= 0:
        x_normalized = np.full_like(positions, np.nan)
    else:
        x_normalized = dist_from_nearest / max_dist
    return x_normalized, fst_values, weights

def _resolve_autocorr_block_size(n: int, default_block_size: int) -> int:
    if USE_GLOBAL_AUTOCORR_BLOCK_SIZE_OVERRIDE:
        candidate = GLOBAL_AUTOCORR_BLOCK_SIZE_WINDOWS
    else:
        candidate = default_block_size

    if not math.isfinite(candidate) or candidate < 1:
        candidate = DEFAULT_BLOCK_SIZE_WINDOWS

    if n > 0:
        candidate = min(int(candidate), n)

    return max(1, int(candidate))


def estimate_correlation_length(
    fst: np.ndarray,
    weights: np.ndarray,
    default_block_size: int = DEFAULT_BLOCK_SIZE_WINDOWS,
) -> Tuple[int, float, int, int]:
    valid = np.isfinite(fst) & np.isfinite(weights)
    if USE_GLOBAL_AUTOCORR_BLOCK_SIZE_OVERRIDE:
        n = int(np.sum(valid))
        block_size = _resolve_autocorr_block_size(n, default_block_size)
        return block_size, float("nan"), 0, 0

    values = fst[valid]
    w = weights[valid]
    n = len(values)
    if n <= 1:
        block_size = _resolve_autocorr_block_size(n, default_block_size)
        return block_size, float("nan"), 0, 0

    if np.sum(w > 0) > 0:
        mean = float(np.average(values, weights=w))
    else:
        mean = float(np.mean(values))

    fluct = values - mean
    max_lag_candidate = n - 1
    if max_lag_candidate < 1:
        block_size = _resolve_autocorr_block_size(n, default_block_size)
        return block_size, float("nan"), 0, 0

    autocorr_vals: List[float] = []
    lags: List[int] = []
    for lag in range(1, max_lag_candidate + 1):
        v1 = fluct[:-lag]
        v2 = fluct[lag:]
        if len(v1) < AUTOCORR_MIN_PAIRS:
            break

        num = float(np.dot(v1, v2)) / len(v1)
        denom = math.sqrt((np.dot(v1, v1) / len(v1)) * (np.dot(v2, v2) / len(v2)))
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
        block_size = _resolve_autocorr_block_size(n, default_block_size)
        last_lag = lags[-1] if lags else 0
        return block_size, float("nan"), last_lag, 0

    autocorr_array = np.array(autocorr_vals)
    monotone = np.minimum.accumulate(autocorr_array)
    target_idx = np.where(monotone <= AUTOCORR_TARGET)[0]
    if len(target_idx) > 0:
        corr_length = float(lags[target_idx[0]])
    else:
        corr_length = float(lags[-1])

    block_size = int(round(max(1.0, corr_length)))
    block_size = max(1, min(block_size, n))
    return block_size, corr_length, lags[-1], len(lags)

def precompute_block_structure(n: int, block_size: int) -> List[np.ndarray]:
    if n <= 0:
        return []
    if n <= block_size:
        return [np.arange(n, dtype=int)]
    n_blocks = (n + block_size - 1) // block_size
    blocks: List[np.ndarray] = []
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        blocks.append(np.arange(start, end, dtype=int))
    return blocks

def generate_block_permutation_indices(
    blocks: List[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = list(blocks)
    rng.shuffle(shuffled)
    return np.concatenate(shuffled)

def build_exhaustive_frf_candidates(
    x_sorted: np.ndarray,
    min_edge_windows: int,
    min_mid_windows: int,
) -> Dict[str, np.ndarray]:
    n = len(x_sorted)
    min_edge_windows = max(1, int(min_edge_windows))
    min_mid_windows = max(1, int(min_mid_windows))
    if n == 0:
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return {"edge_end": empty_int, "mid_start": empty_int, "ramp_start": empty_int, "ramp_end": empty_int, "a_rel": empty_float, "b_rel": empty_float}
    max_edge_end = n - min_mid_windows - 1
    if max_edge_end < min_edge_windows - 1:
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return {"edge_end": empty_int, "mid_start": empty_int, "ramp_start": empty_int, "ramp_end": empty_int, "a_rel": empty_float, "b_rel": empty_float}
    max_mid_start = n - min_mid_windows
    edge_candidates = np.arange(min_edge_windows - 1, max_edge_end + 1, dtype=int)
    mid_counts = max_mid_start - (edge_candidates + 1) + 1
    mid_counts = np.clip(mid_counts, 0, None)
    valid_edges = mid_counts > 0
    if not np.any(valid_edges):
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return {"edge_end": empty_int, "mid_start": empty_int, "ramp_start": empty_int, "ramp_end": empty_int, "a_rel": empty_float, "b_rel": empty_float}
    edge_candidates = edge_candidates[valid_edges]
    mid_counts = mid_counts[valid_edges]
    edge_end_arr = np.repeat(edge_candidates, mid_counts)
    mid_segments: List[np.ndarray] = []
    for edge, count in zip(edge_candidates, mid_counts):
        mid_segments.append(np.arange(edge + 1, edge + 1 + count, dtype=int))
    mid_start_arr = np.concatenate(mid_segments, dtype=int) if mid_segments else np.array([], dtype=int)
    ramp_start_arr = edge_end_arr + 1
    ramp_end_arr = mid_start_arr
    a_rel_arr = x_sorted[edge_end_arr].astype(float)
    b_rel_arr = x_sorted[mid_start_arr].astype(float)
    return {"edge_end": edge_end_arr, "mid_start": mid_start_arr, "ramp_start": ramp_start_arr, "ramp_end": ramp_end_arr, "a_rel": a_rel_arr, "b_rel": b_rel_arr}

def _prefix_with_zero(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    out = np.zeros((arr.shape[0], arr.shape[1] + 1), dtype=arr.dtype)
    np.cumsum(arr, axis=1, out=out[:, 1:])
    return out

@dataclass
class FRFStaticTerms:
    edge_idx: np.ndarray
    mid_idx: np.ndarray
    ramp_start_idx: np.ndarray
    ramp_end_idx: np.ndarray
    edge_sum_w: np.ndarray
    mid_sum_w: np.ndarray
    ramp_sum_w: np.ndarray
    ramp_sum_wx: np.ndarray
    ramp_sum_wx2: np.ndarray
    total_w: float
    a_rel: np.ndarray
    b_rel: np.ndarray


def _prepare_frf_static_terms(
    weight_sorted: np.ndarray,
    x_sorted: np.ndarray,
    candidates: Dict[str, np.ndarray],
) -> FRFStaticTerms:
    weight_sorted = np.asarray(weight_sorted, dtype=float)
    x_sorted = np.asarray(x_sorted, dtype=float)
    prefix_w = np.concatenate(([0.0], np.cumsum(weight_sorted, dtype=float)))
    prefix_wx = np.concatenate(([0.0], np.cumsum(weight_sorted * x_sorted, dtype=float)))
    prefix_wx2 = np.concatenate(([0.0], np.cumsum(weight_sorted * (x_sorted ** 2), dtype=float)))

    edge_idx = np.asarray(candidates["edge_end"], dtype=int) + 1
    mid_idx = np.asarray(candidates["mid_start"], dtype=int)
    ramp_start_idx = np.asarray(candidates["ramp_start"], dtype=int)
    ramp_end_idx = np.asarray(candidates["ramp_end"], dtype=int)

    total_w = float(prefix_w[-1])
    edge_sum_w = prefix_w[edge_idx]
    mid_sum_w = total_w - prefix_w[mid_idx]
    ramp_sum_w = prefix_w[ramp_end_idx] - prefix_w[ramp_start_idx]
    ramp_sum_wx = prefix_wx[ramp_end_idx] - prefix_wx[ramp_start_idx]
    ramp_sum_wx2 = prefix_wx2[ramp_end_idx] - prefix_wx2[ramp_start_idx]

    return FRFStaticTerms(
        edge_idx=edge_idx,
        mid_idx=mid_idx,
        ramp_start_idx=ramp_start_idx,
        ramp_end_idx=ramp_end_idx,
        edge_sum_w=edge_sum_w,
        mid_sum_w=mid_sum_w,
        ramp_sum_w=ramp_sum_w,
        ramp_sum_wx=ramp_sum_wx,
        ramp_sum_wx2=ramp_sum_wx2,
        total_w=total_w,
        a_rel=np.asarray(candidates["a_rel"], dtype=float),
        b_rel=np.asarray(candidates["b_rel"], dtype=float),
    )


def run_frf_search(
    fst_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    x_sorted: np.ndarray,
    candidates: Dict[str, np.ndarray],
    half_length_bp: float,
    *,
    static_terms: Optional[FRFStaticTerms] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fst_matrix = np.asarray(fst_matrix, dtype=float)
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    if fst_matrix.ndim != 2 or weight_matrix.ndim != 2:
        raise ValueError("fst_matrix and weight_matrix must be 2D")
    if fst_matrix.shape != weight_matrix.shape:
        raise ValueError("fst_matrix and weight_matrix must have same shape")
    n_samples, n_windows = fst_matrix.shape
    if n_windows == 0 or candidates["edge_end"].size == 0 or half_length_bp <= 0:
        nan = np.full(n_samples, np.nan)
        return nan, nan, nan, nan, nan

    eps = 1e-12
    row_idx = np.arange(n_samples)

    if static_terms is None:
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
        if n_candidates == 0:
            nan = np.full(n_samples, np.nan)
            return nan, nan, nan, nan, nan

        best_sse = np.full(n_samples, np.inf)
        best_mu_edge = np.full(n_samples, np.nan)
        best_mu_mid = np.full(n_samples, np.nan)
        best_delta = np.full(n_samples, np.nan)
        best_a_bp = np.full(n_samples, np.nan)
        best_b_bp = np.full(n_samples, np.nan)

        chunk_size = max(1, min(FRF_CANDIDATE_CHUNK_SIZE, n_candidates))
        for start_idx in range(0, n_candidates, chunk_size):
            end_idx = min(start_idx + chunk_size, n_candidates)
            edge_chunk = edge_end[start_idx:end_idx]
            mid_chunk = mid_start[start_idx:end_idx]
            ramp_start_chunk = ramp_start[start_idx:end_idx]
            ramp_end_chunk = ramp_end[start_idx:end_idx]
            a_chunk = a_rel[start_idx:end_idx]
            b_chunk = b_rel[start_idx:end_idx]

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

            edge_sse = edge_sum_wf2 - np.where(
                edge_sum_w > eps, (edge_sum_wf ** 2) / np.maximum(edge_sum_w, eps), 0.0
            )
            mid_sse = mid_sum_wf2 - np.where(
                mid_sum_w > eps, (mid_sum_wf ** 2) / np.maximum(mid_sum_w, eps), 0.0
            )

            ramp_sum_w = np.take(prefix_w, ramp_end_chunk, axis=1) - np.take(
                prefix_w, ramp_start_chunk, axis=1
            )
            ramp_sum_wf = np.take(prefix_wf, ramp_end_chunk, axis=1) - np.take(
                prefix_wf, ramp_start_chunk, axis=1
            )
            ramp_sum_wf2 = np.take(prefix_wf2, ramp_end_chunk, axis=1) - np.take(
                prefix_wf2, ramp_start_chunk, axis=1
            )
            ramp_sum_wx = np.take(prefix_wx, ramp_end_chunk, axis=1) - np.take(
                prefix_wx, ramp_start_chunk, axis=1
            )
            ramp_sum_wx2 = np.take(prefix_wx2, ramp_end_chunk, axis=1) - np.take(
                prefix_wx2, ramp_start_chunk, axis=1
            )
            ramp_sum_wfx = np.take(prefix_wfx, ramp_end_chunk, axis=1) - np.take(
                prefix_wfx, ramp_start_chunk, axis=1
            )

            delta_rel = np.maximum(b_chunk - a_chunk, 1e-6)
            slope = (mu_mid - mu_edge) / delta_rel[np.newaxis, :]
            intercept = mu_edge - slope * a_chunk[np.newaxis, :]

            ramp_sse = (
                ramp_sum_wf2
                - 2.0 * intercept * ramp_sum_wf
                - 2.0 * slope * ramp_sum_wfx
                + (intercept ** 2) * ramp_sum_w
                + 2.0 * intercept * slope * ramp_sum_wx
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

    # Cached path
    n_candidates = static_terms.edge_idx.size
    if n_candidates == 0:
        nan = np.full(n_samples, np.nan)
        return nan, nan, nan, nan, nan

    target_cells = 2_000_000
    best_sse = np.full(n_samples, np.inf)
    best_mu_edge = np.full(n_samples, np.nan)
    best_mu_mid = np.full(n_samples, np.nan)
    best_delta = np.full(n_samples, np.nan)
    best_a_bp = np.full(n_samples, np.nan)
    best_b_bp = np.full(n_samples, np.nan)

    prefix_row_chunk = max(
        1,
        min(
            n_samples,
            max(1, FRF_PREFIX_CHUNK_CELLS // max(1, n_windows + 1)),
        ),
    )

    for row_start in range(0, n_samples, prefix_row_chunk):
        row_end = min(row_start + prefix_row_chunk, n_samples)
        row_slice = slice(row_start, row_end)
        sub_weights = weight_matrix[row_slice]
        sub_fst = fst_matrix[row_slice]
        sub_rows = row_end - row_start
        row_idx_sub = np.arange(sub_rows)

        prefix_wf = _prefix_with_zero(sub_weights * sub_fst)
        prefix_wf2 = _prefix_with_zero(sub_weights * (sub_fst ** 2))
        prefix_wfx = _prefix_with_zero(sub_weights * sub_fst * x_sorted[np.newaxis, :])

        total_wf = prefix_wf[:, -1]
        total_wf2 = prefix_wf2[:, -1]

        chunk_size = max(1, min(n_candidates, target_cells // max(1, sub_rows)))

        best_sse_sub = np.full(sub_rows, np.inf)
        best_mu_edge_sub = np.full(sub_rows, np.nan)
        best_mu_mid_sub = np.full(sub_rows, np.nan)
        best_delta_sub = np.full(sub_rows, np.nan)
        best_a_bp_sub = np.full(sub_rows, np.nan)
        best_b_bp_sub = np.full(sub_rows, np.nan)

        for start in range(0, n_candidates, chunk_size):
            end = min(start + chunk_size, n_candidates)

            edge_idx = static_terms.edge_idx[start:end]
            mid_idx = static_terms.mid_idx[start:end]

            edge_sum_w = static_terms.edge_sum_w[np.newaxis, start:end]
            mid_sum_w = static_terms.mid_sum_w[np.newaxis, start:end]

            edge_sum_wf = np.take(prefix_wf, edge_idx, axis=1)
            edge_sum_wf2 = np.take(prefix_wf2, edge_idx, axis=1)
            mu_edge = edge_sum_wf / np.maximum(edge_sum_w, eps)

            mid_prefix_wf = np.take(prefix_wf, mid_idx, axis=1)
            mid_prefix_wf2 = np.take(prefix_wf2, mid_idx, axis=1)
            mid_sum_wf = total_wf[:, None] - mid_prefix_wf
            mid_sum_wf2 = total_wf2[:, None] - mid_prefix_wf2
            mu_mid = mid_sum_wf / np.maximum(mid_sum_w, eps)

            edge_sse = edge_sum_wf2 - np.where(
                edge_sum_w > eps,
                (edge_sum_wf ** 2) / np.maximum(edge_sum_w, eps),
                0.0,
            )
            mid_sse = mid_sum_wf2 - np.where(
                mid_sum_w > eps,
                (mid_sum_wf ** 2) / np.maximum(mid_sum_w, eps),
                0.0,
            )

            rs = static_terms.ramp_start_idx[start:end]
            re = static_terms.ramp_end_idx[start:end]
            ramp_sum_w = static_terms.ramp_sum_w[np.newaxis, start:end]
            ramp_sum_wx = static_terms.ramp_sum_wx[np.newaxis, start:end]
            ramp_sum_wx2 = static_terms.ramp_sum_wx2[np.newaxis, start:end]

            ramp_sum_wf = np.take(prefix_wf, re, axis=1) - np.take(prefix_wf, rs, axis=1)
            ramp_sum_wf2 = np.take(prefix_wf2, re, axis=1) - np.take(prefix_wf2, rs, axis=1)
            ramp_sum_wfx = np.take(prefix_wfx, re, axis=1) - np.take(prefix_wfx, rs, axis=1)

            a_rel = static_terms.a_rel[start:end]
            b_rel = static_terms.b_rel[start:end]
            denom = np.maximum(b_rel - a_rel, 1e-6)[np.newaxis, :]
            slope = (mu_mid - mu_edge) / denom
            intercept = mu_edge - slope * a_rel[np.newaxis, :]

            ramp_sse = (
                ramp_sum_wf2
                - 2.0 * intercept * ramp_sum_wf
                - 2.0 * slope * ramp_sum_wfx
                + (intercept ** 2) * ramp_sum_w
                + 2.0 * intercept * slope * ramp_sum_wx
                + (slope ** 2) * ramp_sum_wx2
            )

            total_sse = edge_sse + mid_sse + ramp_sse
            local_idx = np.argmin(total_sse, axis=1)
            local_best = total_sse[row_idx_sub, local_idx]
            update_mask = local_best < best_sse_sub
            if np.any(update_mask):
                best_sse_sub[update_mask] = local_best[update_mask]
                sel_mu_edge = mu_edge[row_idx_sub, local_idx]
                sel_mu_mid = mu_mid[row_idx_sub, local_idx]
                best_mu_edge_sub[update_mask] = sel_mu_edge[update_mask]
                best_mu_mid_sub[update_mask] = sel_mu_mid[update_mask]
                delta_vals = sel_mu_edge - sel_mu_mid
                best_delta_sub[update_mask] = delta_vals[update_mask]
                sel_a = a_rel[local_idx] * half_length_bp
                sel_b = b_rel[local_idx] * half_length_bp
                best_a_bp_sub[update_mask] = sel_a[update_mask]
                best_b_bp_sub[update_mask] = sel_b[update_mask]

        best_sse[row_slice] = best_sse_sub
        best_mu_edge[row_slice] = best_mu_edge_sub
        best_mu_mid[row_slice] = best_mu_mid_sub
        best_delta[row_slice] = best_delta_sub
        best_a_bp[row_slice] = best_a_bp_sub
        best_b_bp[row_slice] = best_b_bp_sub

    return best_mu_edge, best_mu_mid, best_delta, best_a_bp, best_b_bp

# ------------------------- ROTATION HELPERS -------------------------

def _frf_predict_series(
    x_sorted: np.ndarray,
    mu_edge: float,
    mu_mid: float,
    a_bp: float,
    b_bp: float,
    half_length_bp: float,
) -> np.ndarray:
    if not (
        np.isfinite(mu_edge)
        and np.isfinite(mu_mid)
        and np.isfinite(a_bp)
        and np.isfinite(b_bp)
        and half_length_bp > 0
    ):
        return np.full_like(x_sorted, np.nan, dtype=float)
    a_rel = float(a_bp) / float(half_length_bp)
    b_rel = float(b_bp) / float(half_length_bp)
    a_rel, b_rel = float(min(a_rel, b_rel)), float(max(a_rel, b_rel))
    out = np.empty_like(x_sorted, dtype=float)
    left = x_sorted <= a_rel
    right = x_sorted >= b_rel
    mid = ~(left | right)
    out[left] = mu_edge
    out[right] = mu_mid
    if np.any(mid):
        denom = max(1e-12, b_rel - a_rel)
        out[mid] = mu_edge + (mu_mid - mu_edge) * (x_sorted[mid] - a_rel) / denom
    return out


def _mad_standardize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = mad if mad > 1e-12 else np.nanstd(v) or 1.0
    return (v - med) / max(scale, 1e-12)

# ------------------------- PER-INVERSION FRF + NULL -------------------------

def prepare_inversion_frf_and_permutation(
    inversion: Inversion,
    n_permutations: int,
    default_block_size: int,
) -> PreparedInversion:
    x_full, fst_full, w_full = compute_folded_distances(inversion)
    n_all = len(x_full)

    valid = np.isfinite(x_full) & np.isfinite(fst_full) & np.isfinite(w_full)
    n_valid = int(np.sum(valid))
    n_sites_total = sum(w.n_sites for w in inversion.windows)

    if n_valid < 3 or inversion.length <= 0:
        result = FRFResult(
            inv_key=inversion.inv_key,
            chrom=inversion.chrom,
            start=inversion.start,
            end=inversion.end,
            length=inversion.length,
            n_windows=inversion.n_windows,
            n_sites=n_sites_total,
            block_size_windows=0,
            n_blocks=0,
            frf_mu_edge=float("nan"),
            frf_mu_mid=float("nan"),
            frf_delta=float("nan"),
            frf_null_delta_mean=float("nan"),
            frf_a=float("nan"),
            frf_b=float("nan"),
            frf_var_delta=float("nan"),
            frf_se_delta=float("nan"),
            usable_for_meta=False,
        )
        return PreparedInversion(result=result, plan=None)

    x_v = x_full[valid]
    fst_v = fst_full[valid]
    w_v = w_full[valid]
    half_length = inversion.length / 2.0
    order = np.argsort(x_v)
    x_sorted = x_v[order]
    fst_sorted = fst_v[order]
    w_sorted = w_v[order]

    candidates = build_exhaustive_frf_candidates(
        x_sorted, FRF_MIN_EDGE_WINDOWS, FRF_MIN_MID_WINDOWS
    )
    if candidates["edge_end"].size == 0:
        result = FRFResult(
            inv_key=inversion.inv_key,
            chrom=inversion.chrom,
            start=inversion.start,
            end=inversion.end,
            length=inversion.length,
            n_windows=inversion.n_windows,
            n_sites=n_sites_total,
            block_size_windows=0,
            n_blocks=0,
            frf_mu_edge=float("nan"),
            frf_mu_mid=float("nan"),
            frf_delta=float("nan"),
            frf_null_delta_mean=float("nan"),
            frf_a=float("nan"),
            frf_b=float("nan"),
            frf_var_delta=float("nan"),
            frf_se_delta=float("nan"),
            usable_for_meta=False,
        )
        return PreparedInversion(result=result, plan=None)

    static_terms = _prepare_frf_static_terms(w_sorted, x_sorted, candidates)
    mu_edge_arr, mu_mid_arr, delta_arr, a_bp_arr, b_bp_arr = run_frf_search(
        fst_sorted[np.newaxis, :],
        w_sorted[np.newaxis, :],
        x_sorted,
        candidates,
        half_length,
        static_terms=static_terms,
    )
    frf_mu_edge = float(mu_edge_arr[0])
    frf_mu_mid = float(mu_mid_arr[0])
    frf_delta = float(delta_arr[0])
    frf_a = float(a_bp_arr[0])
    frf_b = float(b_bp_arr[0])

    result = FRFResult(
        inv_key=inversion.inv_key,
        chrom=inversion.chrom,
        start=inversion.start,
        end=inversion.end,
        length=inversion.length,
        n_windows=inversion.n_windows,
        n_sites=n_sites_total,
        block_size_windows=0,
        n_blocks=0,
        frf_mu_edge=frf_mu_edge,
        frf_mu_mid=frf_mu_mid,
        frf_delta=frf_delta,
        frf_null_delta_mean=float("nan"),
        frf_a=frf_a,
        frf_b=frf_b,
        frf_var_delta=float("nan"),
        frf_se_delta=float("nan"),
        usable_for_meta=False,
    )

    if n_permutations <= 0:
        return PreparedInversion(result=result, plan=None)

    n = int(x_sorted.size)
    if n < 2:
        return PreparedInversion(result=result, plan=None)

    valid_offsets = np.arange(1, n, dtype=int)
    if ROTATION_SEAM_EXCLUDE_WINDOWS > 0:
        ex = int(ROTATION_SEAM_EXCLUDE_WINDOWS)
        valid_offsets = valid_offsets[(valid_offsets >= ex) & (valid_offsets <= n - ex)]
    if valid_offsets.size == 0:
        return PreparedInversion(result=result, plan=None)

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)

    if ROTATION_USE_RESIDUALS:
        trend_sorted = _frf_predict_series(
            x_sorted, frf_mu_edge, frf_mu_mid, frf_a, frf_b, half_length
        )
        residual_sorted = fst_sorted - trend_sorted
        rotation_basis_physical = _mad_standardize(residual_sorted)[inverse_order]
        trend_physical = trend_sorted[inverse_order]
        use_residuals = True
    else:
        rotation_basis_physical = fst_v.astype(float, copy=False)
        trend_physical = None
        use_residuals = False

    rotation_basis_physical = np.asarray(rotation_basis_physical, dtype=float)
    if trend_physical is not None:
        trend_physical = np.asarray(trend_physical, dtype=float)

    plan = PermutationPlan(
        inv_key=inversion.inv_key,
        base_seed=stable_seed_from_key(inversion.inv_key),
        n_permutations=n_permutations,
        chunk_size=PERMUTATION_CHUNK_SIZE,
        n_valid=n,
        order=order,
        fst_values=fst_v,
        weight_values=w_v,
        weight_sorted=w_sorted,
        x_sorted=x_sorted,
        candidates=candidates,
        half_length=half_length,
        observed_delta=frf_delta,
        valid_offsets=valid_offsets,
        allow_flip=bool(ROTATION_ALLOW_FLIP),
        use_residuals=use_residuals,
        rotation_basis=rotation_basis_physical,
        trend_physical=trend_physical,
    )
    rng_offsets = np.random.default_rng(plan.base_seed)
    plan.offset_seed = int(rng_offsets.integers(0, max(1, plan.valid_offsets.size)))
    plan.offset_step = _choose_coprime_step(plan.valid_offsets.size, rng_offsets)
    return PreparedInversion(result=result, plan=plan)


def fit_inversion_worker(args) -> PreparedInversion:
    inversion, n_permutations, default_block_size = args
    return prepare_inversion_frf_and_permutation(inversion, n_permutations, default_block_size)


def _build_permutation_plan_payload(
    plans: Sequence[PermutationPlan],
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    with _PERM_SHARED_LOCK:
        if _PERM_PARENT_HANDLES:
            for existing in list(_PERM_PARENT_HANDLES.keys()):
                handles = _PERM_PARENT_HANDLES.pop(existing, [])
                for shm in handles:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        continue

        for plan in plans:
            arrays: Dict[str, Optional[SharedArrayDescriptor]] = {}
            handles: List[shared_memory.SharedMemory] = []

            def add_array(
                key: str,
                array: Optional[np.ndarray],
                *,
                dtype: Optional[np.dtype] = None,
            ) -> None:
                if array is None:
                    arrays[key] = None
                    return
                arr = np.asarray(array, dtype=dtype if dtype is not None else array.dtype)
                descriptor, shm = _create_shared_descriptor(arr)
                arrays[key] = descriptor
                if shm is not None:
                    handles.append(shm)

            add_array("rotation_basis", plan.rotation_basis, dtype=np.float64)
            add_array("trend_physical", plan.trend_physical, dtype=np.float64)
            add_array("order", plan.order, dtype=np.int32)
            add_array("weight_sorted", plan.weight_sorted, dtype=np.float64)
            add_array("x_sorted", plan.x_sorted, dtype=np.float64)
            add_array("valid_offsets", plan.valid_offsets, dtype=np.int32)

            cand = plan.candidates
            add_array("candidate_edge_end", cand.get("edge_end"), dtype=np.int32)
            add_array("candidate_mid_start", cand.get("mid_start"), dtype=np.int32)
            add_array("candidate_ramp_start", cand.get("ramp_start"), dtype=np.int32)
            add_array("candidate_ramp_end", cand.get("ramp_end"), dtype=np.int32)
            add_array("candidate_a_rel", cand.get("a_rel"), dtype=np.float64)
            add_array("candidate_b_rel", cand.get("b_rel"), dtype=np.float64)

            payload.append(
                {
                    "plan_id": int(plan.plan_id),
                    "inv_key": plan.inv_key,
                    "base_seed": int(plan.base_seed),
                    "chunk_size": int(plan.chunk_size),
                    "n_permutations": int(plan.n_permutations),
                    "n_valid": int(plan.n_valid),
                    "half_length": float(plan.half_length),
                    "observed_delta": float(plan.observed_delta),
                    "allow_flip": bool(plan.allow_flip),
                    "use_residuals": bool(plan.use_residuals),
                    "offset_step": int(plan.offset_step),
                    "offset_seed": int(plan.offset_seed),
                    "arrays": arrays,
                }
            )

            _PERM_PARENT_HANDLES[int(plan.plan_id)] = handles

    return payload


def _release_permutation_plan_shared_memory(plan_ids: Iterable[int]) -> None:
    with _PERM_SHARED_LOCK:
        for pid in plan_ids:
            handles = _PERM_PARENT_HANDLES.pop(int(pid), [])
            for shm in handles:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except FileNotFoundError:
                    continue


def _permutation_worker_setup(plan_payload: Sequence[Dict[str, Any]]) -> None:
    plans_state: Dict[int, Dict[str, Any]] = {}
    shared_handles: List[shared_memory.SharedMemory] = []

    for spec in plan_payload:
        arrays: Dict[str, Optional[np.ndarray]] = {}
        for key, descriptor in spec["arrays"].items():
            if descriptor is None:
                arrays[key] = None
                continue
            arr, shm = _attach_shared_descriptor(descriptor)
            arrays[key] = arr
            shared_handles.append(shm)

        candidates = {
            "edge_end": arrays["candidate_edge_end"],
            "mid_start": arrays["candidate_mid_start"],
            "ramp_start": arrays["candidate_ramp_start"],
            "ramp_end": arrays["candidate_ramp_end"],
            "a_rel": arrays["candidate_a_rel"],
            "b_rel": arrays["candidate_b_rel"],
        }

        static_terms = _prepare_frf_static_terms(
            arrays["weight_sorted"], arrays["x_sorted"], candidates
        )

        plan_id = int(spec["plan_id"])
        n_valid = int(spec["n_valid"])
        plans_state[plan_id] = {
            "meta": {
                "inv_key": spec["inv_key"],
                "base_seed": int(spec["base_seed"]),
                "chunk_size": int(spec["chunk_size"]),
                "n_permutations": int(spec["n_permutations"]),
                "n_valid": n_valid,
                "half_length": float(spec["half_length"]),
                "observed_delta": float(spec["observed_delta"]),
                "allow_flip": bool(spec["allow_flip"]),
                "use_residuals": bool(spec["use_residuals"]),
                "offset_step": int(spec["offset_step"]),
                "offset_seed": int(spec["offset_seed"]),
            },
            "arrays": arrays,
            "candidates": candidates,
            "static_terms": static_terms,
            "weight_template": arrays["weight_sorted"][np.newaxis, :],
            "base_indices": np.arange(n_valid, dtype=np.int64),
            "flipped_indices": np.arange(n_valid - 1, -1, -1, dtype=np.int64),
        }

    _PERM_SHARED_STATE["plans"] = plans_state
    _PERM_SHARED_STATE["shared_handles"] = shared_handles

    def _cleanup_shared_handles() -> None:
        handles = _PERM_SHARED_STATE.pop("shared_handles", [])
        for shm in handles:
            try:
                shm.close()
            except Exception:
                pass

    atexit.register(_cleanup_shared_handles)


def run_permutation_chunk(args) -> PermutationChunkResult:
    plan_id, chunk_index = args
    plans = _PERM_SHARED_STATE.get("plans")
    if not plans or plan_id not in plans:
        raise RuntimeError("Permutation plan shared state is not initialized in worker")

    state = plans[plan_id]
    meta = state["meta"]
    arrays = state["arrays"]
    static_terms = state["static_terms"]
    candidates = state["candidates"]

    chunk_size = meta["chunk_size"]
    start = chunk_index * chunk_size
    remaining = meta["n_permutations"] - start
    if remaining <= 0:
        return PermutationChunkResult(meta["inv_key"], chunk_index, 0, 0.0, 0.0, 0)

    size = min(chunk_size, remaining)

    n = meta["n_valid"]
    valid_offsets = arrays["valid_offsets"]
    if n <= 1 or valid_offsets is None or valid_offsets.size == 0:
        return PermutationChunkResult(meta["inv_key"], chunk_index, 0, 0.0, 0.0, 0)

    rng = np.random.default_rng(meta["base_seed"] + chunk_index)

    offset_positions = _offset_positions(
        valid_offsets.size,
        start,
        size,
        offset_seed=meta["offset_seed"],
        offset_step=meta["offset_step"],
    )
    offsets = valid_offsets[offset_positions.astype(int)]

    rotation_basis = arrays["rotation_basis"]
    trend = arrays["trend_physical"]
    order = arrays["order"].astype(np.int64, copy=False)
    weight_template = state["weight_template"]
    x_sorted = arrays["x_sorted"]

    base_indices = state["base_indices"]
    flipped_indices = state["flipped_indices"]

    count = 0
    sum_delta = 0.0
    sum_sq_delta = 0.0
    count_ge = 0

    batch_limit = max(1, min(1024, size))
    allow_flip = meta["allow_flip"]
    use_residuals = meta["use_residuals"] and trend is not None

    for batch_start in range(0, size, batch_limit):
        batch_end = min(batch_start + batch_limit, size)
        batch_offsets = offsets[batch_start:batch_end]
        batch_size = batch_offsets.size
        if batch_size == 0:
            continue

        orient = np.empty((batch_size, n), dtype=np.int64)
        orient[:] = base_indices
        if allow_flip:
            flip_flags = rng.integers(0, 2, size=batch_size, dtype=np.int8)
            if np.any(flip_flags):
                orient[flip_flags == 1] = flipped_indices

        idx = (orient + batch_offsets[:, None]) % n
        rotated = rotation_basis[idx]
        if use_residuals:
            fst_perm_physical = rotated + trend[np.newaxis, :]
        else:
            fst_perm_physical = rotated

        fst_perm_sorted = fst_perm_physical[:, order]
        weight_batch = np.broadcast_to(weight_template, (batch_size, n))

        _, _, deltas, _, _ = run_frf_search(
            fst_perm_sorted,
            weight_batch,
            x_sorted,
            candidates,
            meta["half_length"],
            static_terms=static_terms,
        )

        deltas = np.asarray(deltas, dtype=float)
        finite = np.isfinite(deltas)
        if not np.any(finite):
            continue
        valid = deltas[finite]
        count += int(valid.size)
        sum_delta += float(np.sum(valid))
        sum_sq_delta += float(np.sum(valid * valid))
        count_ge += int(np.sum(valid >= meta["observed_delta"]))

    return PermutationChunkResult(
        meta["inv_key"],
        chunk_index,
        count,
        sum_delta,
        sum_sq_delta,
        count_ge,
    )

# ------------------------- PRECISION-WEIGHTED MEDIAN ANALYSIS -------------------------

CONDITION_NUMBER_LIMIT = 1e12


def _solve_weighted_normal_equations(
    XtWX: np.ndarray,
    XtWy: np.ndarray,
    *,
    compute_inverse: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Solve XtWX beta = XtWy while guarding against ill-conditioning."""

    try:
        cond = np.linalg.cond(XtWX)
    except np.linalg.LinAlgError as exc:
        raise exc
    if not np.isfinite(cond) or cond > CONDITION_NUMBER_LIMIT:
        raise np.linalg.LinAlgError("XtWX is ill-conditioned")

    sign, logdet = np.linalg.slogdet(XtWX)
    if sign <= 0.0 or not math.isfinite(logdet):
        raise np.linalg.LinAlgError("XtWX is not positive definite")

    beta_hat = np.linalg.solve(XtWX, XtWy)
    inv_XtWX = np.linalg.solve(XtWX, np.eye(XtWX.shape[0])) if compute_inverse else None
    return beta_hat, inv_XtWX, logdet


def reml_negloglik_tau2(
    tau2: float,
    y: np.ndarray,
    s2: np.ndarray,
    X: np.ndarray,
) -> float:
    if tau2 < 0.0:
        return 1e300
    v = s2 + tau2
    if np.any(v <= 0.0):
        return 1e300
    w = 1.0 / v
    XtW = X.T * w
    XtWX = XtW @ X
    XtWy = XtW @ y
    try:
        beta_hat, _, logdetXtWX = _solve_weighted_normal_equations(
            XtWX, XtWy, compute_inverse=False
        )
    except np.linalg.LinAlgError:
        return 1e300
    resid = y - X @ beta_hat
    sse = float(np.sum(w * resid * resid))
    logdetV = float(np.sum(np.log(v)))
    return 0.5 * (logdetV + sse + logdetXtWX)

def estimate_tau2_reml(
    y: np.ndarray,
    s2: np.ndarray,
    X: np.ndarray,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> float:
    if y.size < 3:
        return 0.0
    var_y = float(np.var(y)) if y.size > 1 else 0.0
    mean_s2 = float(np.mean(s2))
    base_upper = max(1e-8, var_y + mean_s2 * 10.0)

    def golden_section_search(upper: float) -> Tuple[float, bool]:
        if upper <= 0.0:
            return 0.0, False
        a = 0.0
        b = upper
        invphi = 0.6180339887498949
        c = b - invphi * (b - a)
        d = a + invphi * (b - a)
        fc = reml_negloglik_tau2(c, y, s2, X)
        fd = reml_negloglik_tau2(d, y, s2, X)
        for _ in range(max_iter):
            if abs(b - a) < tol * (1.0 + a + b):
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - invphi * (b - a)
                fc = reml_negloglik_tau2(c, y, s2, X)
            else:
                a = c
                c = d
                fc = fd
                d = a + invphi * (b - a)
                fd = reml_negloglik_tau2(d, y, s2, X)
        tau2_est = max(0.0, (a + b) * 0.5)
        hits_upper = tau2_est >= 0.95 * upper
        return tau2_est, hits_upper

    upper = base_upper
    tau2_est = 0.0
    for _ in range(6):
        tau2_est, hits_upper = golden_section_search(upper)
        if not hits_upper:
            break
        upper *= 4.0
    return tau2_est


def estimate_tau2_descriptive(y: np.ndarray, s2: np.ndarray) -> float:
    X = np.ones((y.size, 1), dtype=float)
    tau2 = estimate_tau2_reml(y, s2, X)
    if not math.isfinite(tau2) or tau2 < 0.0:
        tau2 = 0.0
    return float(tau2)


def compute_meta_weights_from_s2(s2: np.ndarray) -> np.ndarray:
    """Precision weights for the robust meta statistic derived from permutation variances."""

    s2 = np.asarray(s2, dtype=float)
    weights = np.zeros_like(s2, dtype=float)
    finite_pos = np.isfinite(s2) & (s2 > 0.0)
    weights[finite_pos] = 1.0 / s2[finite_pos]

    if not np.any(weights > 0.0):
        weights[:] = 1.0

    return weights


def precision_weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return float("nan")
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    total_weight = float(np.sum(w_sorted))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return float("nan")
    cumulative = np.cumsum(w_sorted)
    cutoff = 0.5 * total_weight
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    if idx >= v_sorted.size:
        return float(v_sorted[-1])
    if cumulative[idx] == cutoff and idx + 1 < v_sorted.size:
        return float(0.5 * (v_sorted[idx] + v_sorted[idx + 1]))
    return float(v_sorted[idx])


def weighted_median_difference(
    y: np.ndarray,
    weights: np.ndarray,
    group: np.ndarray,
) -> Tuple[float, float, float]:
    mask_single = group == 0
    mask_recurrent = group == 1
    if not (np.any(mask_single) and np.any(mask_recurrent)):
        return float("nan"), float("nan"), float("nan")
    median_single = precision_weighted_median(y[mask_single], weights[mask_single])
    median_recurrent = precision_weighted_median(y[mask_recurrent], weights[mask_recurrent])
    if not (math.isfinite(median_single) and math.isfinite(median_recurrent)):
        return float("nan"), float(median_single), float(median_recurrent)
    delta = median_single - median_recurrent
    return float(delta), float(median_single), float(median_recurrent)


def weighted_mean_difference(
    y: np.ndarray,
    weights: np.ndarray,
    group: np.ndarray,
) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    group = np.asarray(group, dtype=int)

    mask_valid = np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask_valid):
        return float("nan"), float("nan"), float("nan")

    y = y[mask_valid]
    weights = weights[mask_valid]
    group = group[mask_valid]

    mask_single = group == 0
    mask_recurrent = group == 1

    if not (np.any(mask_single) and np.any(mask_recurrent)):
        return float("nan"), float("nan"), float("nan")

    w_single = float(np.add.reduce(weights[mask_single], dtype=np.float64))
    w_recurrent = float(np.add.reduce(weights[mask_recurrent], dtype=np.float64))

    if w_single <= 0.0 or w_recurrent <= 0.0:
        return float("nan"), float("nan"), float("nan")

    mean_single = float(
        np.add.reduce(weights[mask_single] * y[mask_single], dtype=np.float64) / w_single
    )
    mean_recurrent = float(
        np.add.reduce(weights[mask_recurrent] * y[mask_recurrent], dtype=np.float64) / w_recurrent
    )

    delta = mean_single - mean_recurrent
    return float(delta), float(mean_single), float(mean_recurrent)


def prepare_meta_analysis_inputs(
    df: pd.DataFrame,
) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str]]:
    mask = (
        df["STATUS"].isin([0, 1])
        & df["usable_for_meta"]
        & np.isfinite(df.get("frf_delta_centered", df["frf_delta"]))
        & np.isfinite(df["frf_var_delta"])
        & (df["frf_var_delta"] > 0.0)
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        log.warning("No inversions with usable FRF variance and group labels")
        return None
    if "frf_delta_centered" in sub.columns:
        y = sub["frf_delta_centered"].to_numpy(dtype=float)
        y_label = "frf_delta_centered"
    else:
        y = sub["frf_delta"].to_numpy(dtype=float)
        y_label = "frf_delta"
    s2 = sub["frf_var_delta"].to_numpy(dtype=float)
    group = sub["STATUS"].to_numpy(dtype=int)
    if np.all(group == group[0]):
        log.warning("Only one group present among usable inversions")
        return None
    return sub, y, s2, group, y_label


def run_precision_weighted_median_analysis(
    sub: pd.DataFrame,
    y: np.ndarray,
    s2: np.ndarray,
    group: np.ndarray,
    y_label: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    tau2 = estimate_tau2_descriptive(y, s2)
    weights = compute_meta_weights_from_s2(s2)
    delta, median_single, median_recurrent = weighted_median_difference(y, weights, group)

    return {
        "tau2_descriptive": float(tau2),
        "median_recurrent": float(median_recurrent),
        "median_single": float(median_single),
        "delta_median": float(delta),
        "n_total": float(sub.shape[0]),
        "n_single": float(int(np.sum(group == 0))),
        "n_recurrent": float(int(np.sum(group == 1))),
        "y_label": y_label,
    }, weights

# ------------------------- META-LEVEL PERMUTATION -------------------------

_META_SHARED_STATE: Dict[str, Any] = {}

if njit is not None:

    @njit(nogil=True)
    def _meta_perm_scan_numba(
        y_sorted: np.ndarray,
        w_sorted: np.ndarray,
        stamps: np.ndarray,
        gen: int,
        total_group0_weight: float,
        total_weight: float,
    ) -> Tuple[float, float, float]:
        if not np.isfinite(total_group0_weight) or total_group0_weight <= 0.0:
            return np.nan, np.nan, np.nan
        total_group1_weight = total_weight - total_group0_weight
        if not np.isfinite(total_group1_weight) or total_group1_weight <= 0.0:
            return np.nan, np.nan, np.nan

        target0 = 0.5 * total_group0_weight
        target1 = 0.5 * total_group1_weight

        median0 = np.nan
        median1 = np.nan
        last0 = np.nan
        last1 = np.nan
        cum0 = 0.0
        cum1 = 0.0
        n = y_sorted.size

        for idx in range(n):
            wt = w_sorted[idx]
            val = y_sorted[idx]
            if stamps[idx] == gen:
                cum0 += wt
                last0 = val
                if not np.isfinite(median0):
                    if cum0 > target0:
                        median0 = val
                    elif cum0 == target0:
                        next_val = np.nan
                        for j in range(idx + 1, n):
                            if stamps[j] == gen:
                                next_val = y_sorted[j]
                                break
                        if np.isfinite(next_val):
                            median0 = 0.5 * (val + next_val)
                        else:
                            median0 = val
            else:
                cum1 += wt
                last1 = val
                if not np.isfinite(median1):
                    if cum1 > target1:
                        median1 = val
                    elif cum1 == target1:
                        next_val = np.nan
                        for j in range(idx + 1, n):
                            if stamps[j] != gen:
                                next_val = y_sorted[j]
                                break
                        if np.isfinite(next_val):
                            median1 = 0.5 * (val + next_val)
                        else:
                            median1 = val
            if np.isfinite(median0) and np.isfinite(median1):
                break

        if not np.isfinite(median0):
            median0 = last0
        if not np.isfinite(median1):
            median1 = last1
        if not np.isfinite(median0) or not np.isfinite(median1):
            return np.nan, np.nan, np.nan
        return median0 - median1, median0, median1

else:  # pragma: no cover - numba optional
    _meta_perm_scan_numba = None  # type: ignore

_PERM_SHARED_STATE: Dict[str, Any] = {}
_PERM_SHARED_LOCK = threading.Lock()
_PERM_PARENT_HANDLES: Dict[int, List[shared_memory.SharedMemory]] = {}


def _meta_perm_worker_setup(
    shm_name_y: str,
    shm_name_w: str,
    length: int,
    dtype_str: str,
    n_group0: int,
    total_weight: float,
    T_obs: float,
    progress_queue: Optional[Any],
    total_weighted_y: Optional[float] = None,
) -> None:
    """Initializer that maps shared arrays into the worker process."""

    dtype = np.dtype(dtype_str)
    shm_y = shared_memory.SharedMemory(name=shm_name_y)
    shm_w = shared_memory.SharedMemory(name=shm_name_w)

    _META_SHARED_STATE["y_shm"] = shm_y
    _META_SHARED_STATE["w_shm"] = shm_w
    _META_SHARED_STATE["y"] = np.ndarray((length,), dtype=dtype, buffer=shm_y.buf)
    _META_SHARED_STATE["w"] = np.ndarray((length,), dtype=dtype, buffer=shm_w.buf)
    _META_SHARED_STATE["n"] = int(length)
    _META_SHARED_STATE["n_group0"] = int(n_group0)
    _META_SHARED_STATE["total_weight"] = float(total_weight)
    _META_SHARED_STATE["T_obs"] = float(T_obs)
    _META_SHARED_STATE["abs_T_obs"] = abs(float(T_obs))
    if progress_queue is not None:
        _META_SHARED_STATE["progress_queue"] = progress_queue
    else:
        _META_SHARED_STATE.pop("progress_queue", None)
    if total_weighted_y is not None and math.isfinite(total_weighted_y):
        _META_SHARED_STATE["total_weighted_y"] = float(total_weighted_y)
    else:
        _META_SHARED_STATE.pop("total_weighted_y", None)


def _resolve_weighted_median_tie(
    idx: int,
    *,
    is_group0: bool,
    gen: int,
    stamps: np.ndarray,
    y_sorted: np.ndarray,
) -> float:
    """Resolve a tie by averaging with the next value from the same group."""

    current = float(y_sorted[idx])
    n = y_sorted.size
    for j in range(idx + 1, n):
        in_group0 = stamps[j] == gen
        if in_group0 is is_group0:
            return float(0.5 * (current + float(y_sorted[j])))
    return current


def _meta_perm_single_stat(
    stamps: np.ndarray,
    gen: int,
    total_group0_weight: float,
) -> Tuple[float, float, float]:
    y_sorted = _META_SHARED_STATE["y"]
    w_sorted = _META_SHARED_STATE["w"]
    total_weight = _META_SHARED_STATE["total_weight"]

    if _meta_perm_scan_numba is not None:
        delta, med0, med1 = _meta_perm_scan_numba(
            y_sorted,
            w_sorted,
            stamps,
            int(gen),
            float(total_group0_weight),
            float(total_weight),
        )
        return float(delta), float(med0), float(med1)

    if not math.isfinite(total_group0_weight) or total_group0_weight <= 0.0:
        return float("nan"), float("nan"), float("nan")

    total_group1_weight = float(total_weight - total_group0_weight)
    if not math.isfinite(total_group1_weight) or total_group1_weight <= 0.0:
        return float("nan"), float("nan"), float("nan")

    target0 = 0.5 * total_group0_weight
    target1 = 0.5 * total_group1_weight

    median0 = float("nan")
    median1 = float("nan")
    last0 = float("nan")
    last1 = float("nan")
    cum0 = 0.0
    cum1 = 0.0

    n = y_sorted.size
    for idx in range(n):
        wt = float(w_sorted[idx])
        val = float(y_sorted[idx])
        if stamps[idx] == gen:
            cum0 += wt
            last0 = val
            if not math.isfinite(median0):
                if cum0 > target0:
                    median0 = val
                elif cum0 == target0:
                    median0 = _resolve_weighted_median_tie(
                        idx,
                        is_group0=True,
                        gen=gen,
                        stamps=stamps,
                        y_sorted=y_sorted,
                    )
        else:
            cum1 += wt
            last1 = val
            if not math.isfinite(median1):
                if cum1 > target1:
                    median1 = val
                elif cum1 == target1:
                    median1 = _resolve_weighted_median_tie(
                        idx,
                        is_group0=False,
                        gen=gen,
                        stamps=stamps,
                        y_sorted=y_sorted,
                    )

        if math.isfinite(median0) and math.isfinite(median1):
            break

    if not math.isfinite(median0):
        median0 = last0
    if not math.isfinite(median1):
        median1 = last1

    if not math.isfinite(median0) or not math.isfinite(median1):
        return float("nan"), float("nan"), float("nan")

    return float(median0 - median1), float(median0), float(median1)


def _maybe_set_affinity(cpu_affinity: Optional[List[int]]) -> None:
    if not cpu_affinity:
        return
    try:
        os.sched_setaffinity(0, set(cpu_affinity))
    except (AttributeError, PermissionError, OSError):
        pass


def _meta_perm_worker_run(
    draws_target: int,
    seed: int,
    report_every: int,
    cpu_affinity: Optional[List[int]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = _META_SHARED_STATE["n"]
    n_group0 = _META_SHARED_STATE["n_group0"]
    T_obs = _META_SHARED_STATE["T_obs"]
    abs_T_obs = _META_SHARED_STATE["abs_T_obs"]
    w_sorted = _META_SHARED_STATE["w"]
    progress_queue = _META_SHARED_STATE.get("progress_queue")

    _maybe_set_affinity(cpu_affinity)

    stamps = np.zeros(n, dtype=np.int32)
    gen = 1
    gen_max = np.iinfo(np.int32).max
    draws_done = 0
    invalid = 0
    count_upper = 0
    count_lower = 0
    count_two = 0
    sum_T = 0.0
    sumsq_T = 0.0

    report_every = max(1, int(report_every))
    reported_since_flush = 0
    while draws_done < draws_target:
        if gen >= gen_max:
            stamps.fill(0)
            gen = 1
        sample = rng.choice(n, size=n_group0, replace=False, shuffle=False)
        stamps[sample] = gen
        total0 = float(np.add.reduce(w_sorted[sample], dtype=np.float64))

        delta, _, _ = _meta_perm_single_stat(stamps, gen, total0)
        if not math.isfinite(delta):
            invalid += 1
            gen += 1
            continue

        if delta >= T_obs:
            count_upper += 1
        if delta <= T_obs:
            count_lower += 1
        if abs(delta) >= abs_T_obs:
            count_two += 1
        sum_T += delta
        sumsq_T += delta * delta
        draws_done += 1
        reported_since_flush += 1

        if progress_queue is not None and reported_since_flush >= report_every:
            progress_queue.put(int(reported_since_flush))
            reported_since_flush = 0

        if draws_done < draws_target:
            delta_flip = -delta
            if delta_flip >= T_obs:
                count_upper += 1
            if delta_flip <= T_obs:
                count_lower += 1
            if abs(delta_flip) >= abs_T_obs:
                count_two += 1
            sum_T += delta_flip
            sumsq_T += delta_flip * delta_flip
            draws_done += 1
            reported_since_flush += 1

            if progress_queue is not None and reported_since_flush >= report_every:
                progress_queue.put(int(reported_since_flush))
                reported_since_flush = 0

        gen += 1

    if progress_queue is not None and reported_since_flush > 0:
        progress_queue.put(int(reported_since_flush))

    return {
        "draws": draws_done,
        "invalid": invalid,
        "count_upper": count_upper,
        "count_lower": count_lower,
        "count_two": count_two,
        "sum_T": sum_T,
        "sumsq_T": sumsq_T,
    }


def _meta_perm_worker_run_mean(
    draws_target: int,
    seed: int,
    report_every: int,
    cpu_affinity: Optional[List[int]],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = _META_SHARED_STATE["n"]
    n_group0 = _META_SHARED_STATE["n_group0"]
    T_obs = _META_SHARED_STATE["T_obs"]
    abs_T_obs = _META_SHARED_STATE["abs_T_obs"]
    y_sorted = _META_SHARED_STATE["y"]
    w_sorted = _META_SHARED_STATE["w"]
    total_weight = float(_META_SHARED_STATE["total_weight"])
    progress_queue = _META_SHARED_STATE.get("progress_queue")
    total_weighted_y = float(
        _META_SHARED_STATE.get(
            "total_weighted_y",
            float(np.add.reduce(w_sorted * y_sorted, dtype=np.float64)),
        )
    )
    _META_SHARED_STATE["total_weighted_y"] = total_weighted_y

    _maybe_set_affinity(cpu_affinity)

    draws_done = 0
    invalid = 0
    count_upper = 0
    count_lower = 0
    count_two = 0
    sum_T = 0.0
    sumsq_T = 0.0

    report_every = max(1, int(report_every))
    reported_since_flush = 0

    while draws_done < draws_target:
        sample = rng.choice(n, size=n_group0, replace=False, shuffle=False)
        w0 = float(np.add.reduce(w_sorted[sample], dtype=np.float64))
        if not math.isfinite(w0) or w0 <= 0.0:
            invalid += 1
            continue
        w1 = total_weight - w0
        if not math.isfinite(w1) or w1 <= 0.0:
            invalid += 1
            continue

        wy0 = float(np.add.reduce(w_sorted[sample] * y_sorted[sample], dtype=np.float64))
        wy1 = total_weighted_y - wy0

        if not (math.isfinite(wy0) and math.isfinite(wy1)):
            invalid += 1
            continue

        delta = float(wy0 / w0 - wy1 / w1)

        if not math.isfinite(delta):
            invalid += 1
            continue

        if delta >= T_obs:
            count_upper += 1
        if delta <= T_obs:
            count_lower += 1
        if abs(delta) >= abs_T_obs:
            count_two += 1
        sum_T += delta
        sumsq_T += delta * delta
        draws_done += 1
        reported_since_flush += 1

        if progress_queue is not None and reported_since_flush >= report_every:
            progress_queue.put(int(reported_since_flush))
            reported_since_flush = 0

        if draws_done < draws_target:
            delta_flip = -delta
            if delta_flip >= T_obs:
                count_upper += 1
            if delta_flip <= T_obs:
                count_lower += 1
            if abs(delta_flip) >= abs_T_obs:
                count_two += 1
            sum_T += delta_flip
            sumsq_T += delta_flip * delta_flip
            draws_done += 1
            reported_since_flush += 1

            if progress_queue is not None and reported_since_flush >= report_every:
                progress_queue.put(int(reported_since_flush))
                reported_since_flush = 0

    if progress_queue is not None and reported_since_flush > 0:
        progress_queue.put(int(reported_since_flush))

    return {
        "draws": draws_done,
        "invalid": invalid,
        "count_upper": count_upper,
        "count_lower": count_lower,
        "count_two": count_two,
        "sum_T": sum_T,
        "sumsq_T": sumsq_T,
    }

def meta_permutation_pvalue(
    y: np.ndarray,
    weights: np.ndarray,
    group: np.ndarray,
    n_perm: int,
    chunk: int,
    base_seed: int,
    n_workers: int,
) -> Dict[str, float | int]:
    n = y.size
    weights_obs = np.asarray(weights, dtype=float)
    group_obs = np.asarray(group, dtype=int)
    T_obs, _, _ = weighted_median_difference(y, weights_obs, group_obs)

    if not math.isfinite(T_obs):
        return {
            "p_perm_one_sided_upper": float("nan"),
            "p_perm_one_sided_lower": float("nan"),
            "p_perm_two_sided": float("nan"),
            "T_obs": float(T_obs),
            "total_draws": int(0),
            "count_upper": int(0),
            "count_lower": int(0),
            "count_two": int(0),
        }

    mask_group0 = group_obs == 0
    n_group0 = int(np.sum(mask_group0))
    n_group1 = n - n_group0
    if n_group0 == 0 or n_group1 == 0 or n_perm <= 0:
        return {
            "p_perm_one_sided_upper": float("nan"),
            "p_perm_one_sided_lower": float("nan"),
            "p_perm_two_sided": float("nan"),
            "T_obs": float(T_obs),
            "total_draws": int(0),
            "count_upper": int(0),
            "count_lower": int(0),
            "count_two": int(0),
        }

    order = np.argsort(y)
    y_sorted = np.ascontiguousarray(y[order], dtype=np.float64)
    w_sorted = np.ascontiguousarray(weights_obs[order], dtype=np.float64)
    total_weight = float(np.sum(w_sorted))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return {
            "p_perm_one_sided_upper": float("nan"),
            "p_perm_one_sided_lower": float("nan"),
            "p_perm_two_sided": float("nan"),
            "T_obs": float(T_obs),
            "total_draws": int(0),
            "count_upper": int(0),
            "count_lower": int(0),
            "count_two": int(0),
        }

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()

    shm_y = shared_memory.SharedMemory(create=True, size=y_sorted.nbytes)
    shm_w = shared_memory.SharedMemory(create=True, size=w_sorted.nbytes)
    try:
        np.ndarray(y_sorted.shape, dtype=y_sorted.dtype, buffer=shm_y.buf)[:] = y_sorted
        np.ndarray(w_sorted.shape, dtype=w_sorted.dtype, buffer=shm_w.buf)[:] = w_sorted

        try:
            import psutil  # type: ignore

            physical_cores = psutil.cpu_count(logical=False) or 1
        except Exception:
            physical_cores = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )

        n_workers = max(1, min(n_workers, physical_cores))
        n_workers = min(n_workers, n_perm)
        if n_workers <= 0:
            n_workers = 1

        if hasattr(os, "sched_getaffinity"):
            available_cpus = sorted(os.sched_getaffinity(0))  # type: ignore[arg-type]
        else:
            available_cpus = list(range(os.cpu_count() or 1))
        if not available_cpus:
            available_cpus = list(range(n_workers))

        cpu_plan: List[List[int]] = [
            [available_cpus[idx % len(available_cpus)]] for idx in range(n_workers)
        ]

        draws_per_worker = [n_perm // n_workers] * n_workers
        remainder = n_perm % n_workers
        for i in range(remainder):
            draws_per_worker[i] += 1

        seeds = [base_seed + 1 + i for i in range(n_workers)]

        progress_queue = ctx.Queue()
        disable_bar = not sys.stdout.isatty()
        results: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_meta_perm_worker_setup,
            initargs=(
                shm_y.name,
                shm_w.name,
                y_sorted.size,
                y_sorted.dtype.str,
                n_group0,
                total_weight,
                T_obs,
                progress_queue,
                float(np.add.reduce(w_sorted * y_sorted, dtype=np.float64)),
            ),
        ) as pool:
            futures = []
            for worker_idx, draws in enumerate(draws_per_worker):
                if draws <= 0:
                    continue
                futures.append(
                    pool.submit(
                        _meta_perm_worker_run,
                        int(draws),
                        int(seeds[worker_idx]),
                        int(max(1, chunk)),
                        cpu_plan[worker_idx],
                    )
                )

            consumer_stop = threading.Event()
            consumer_thread: Optional[threading.Thread] = None

            with tqdm(
                total=n_perm,
                unit="draw",
                dynamic_ncols=True,
                smoothing=0.0,
                mininterval=0.5,
                ascii=True,
                disable=disable_bar,
            ) as pbar:

                if futures:
                    def _drain_progress() -> None:
                        while not consumer_stop.is_set():
                            try:
                                delta = progress_queue.get(timeout=0.5)
                            except Empty:
                                continue
                            if delta is None:
                                break
                            if delta:
                                pbar.update(int(delta))
                        while True:
                            try:
                                delta = progress_queue.get_nowait()
                            except Empty:
                                break
                            if not delta:
                                continue
                            pbar.update(int(delta))

                    consumer_thread = threading.Thread(
                        target=_drain_progress,
                        name="meta-perm-progress",
                        daemon=True,
                    )
                    consumer_thread.start()

                try:
                    results = [f.result() for f in futures]
                finally:
                    consumer_stop.set()
                    progress_queue.put(None)
                    if consumer_thread is not None:
                        consumer_thread.join()

        progress_queue.close()
        progress_queue.join_thread()

        total_draws = sum(res["draws"] for res in results)
        if total_draws == 0:
            return {
                "p_perm_one_sided_upper": float("nan"),
                "p_perm_one_sided_lower": float("nan"),
                "p_perm_two_sided": float("nan"),
                "T_obs": float(T_obs),
                "total_draws": int(0),
                "count_upper": int(0),
                "count_lower": int(0),
                "count_two": int(0),
            }

        total_upper = sum(res["count_upper"] for res in results)
        total_lower = sum(res["count_lower"] for res in results)
        total_two = sum(res["count_two"] for res in results)

        p_one_upper = (1.0 + float(total_upper)) / (1.0 + float(total_draws))
        p_one_lower = (1.0 + float(total_lower)) / (1.0 + float(total_draws))
        p_two = (1.0 + float(total_two)) / (1.0 + float(total_draws))
    finally:
        shm_y.close()
        shm_w.close()
        shm_y.unlink()
        shm_w.unlink()

    return {
        "p_perm_one_sided_upper": float(p_one_upper),
        "p_perm_one_sided_lower": float(p_one_lower),
        "p_perm_two_sided": float(p_two),
        "T_obs": float(T_obs),
        "total_draws": int(total_draws),
        "count_upper": int(total_upper),
        "count_lower": int(total_lower),
        "count_two": int(total_two),
    }


def meta_permutation_pvalue_mean(
    y: np.ndarray,
    weights: np.ndarray,
    group: np.ndarray,
    n_perm: int,
    chunk: int,
    base_seed: int,
    n_workers: int,
) -> Dict[str, float | int]:
    n = y.size
    weights_obs = np.asarray(weights, dtype=float)
    group_obs = np.asarray(group, dtype=int)
    T_obs, _, _ = weighted_mean_difference(y, weights_obs, group_obs)

    if not math.isfinite(T_obs):
        return {
            "p_perm_one_sided_upper": float("nan"),
            "p_perm_one_sided_lower": float("nan"),
            "p_perm_two_sided": float("nan"),
            "T_obs": float(T_obs),
            "total_draws": int(0),
            "count_upper": int(0),
            "count_lower": int(0),
            "count_two": int(0),
        }

    mask_group0 = group_obs == 0
    n_group0 = int(np.sum(mask_group0))
    n_group1 = n - n_group0
    if n_group0 == 0 or n_group1 == 0 or n_perm <= 0:
        return {
            "p_perm_one_sided_upper": float("nan"),
            "p_perm_one_sided_lower": float("nan"),
            "p_perm_two_sided": float("nan"),
            "T_obs": float(T_obs),
            "total_draws": int(0),
            "count_upper": int(0),
            "count_lower": int(0),
            "count_two": int(0),
        }

    order = np.argsort(y)
    y_sorted = np.ascontiguousarray(y[order], dtype=np.float64)
    w_sorted = np.ascontiguousarray(weights_obs[order], dtype=np.float64)
    total_weight = float(np.sum(w_sorted))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return {
            "p_perm_one_sided_upper": float("nan"),
            "p_perm_one_sided_lower": float("nan"),
            "p_perm_two_sided": float("nan"),
            "T_obs": float(T_obs),
            "total_draws": int(0),
            "count_upper": int(0),
            "count_lower": int(0),
            "count_two": int(0),
        }

    total_weighted_y = float(np.add.reduce(w_sorted * y_sorted, dtype=np.float64))

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()

    shm_y = shared_memory.SharedMemory(create=True, size=y_sorted.nbytes)
    shm_w = shared_memory.SharedMemory(create=True, size=w_sorted.nbytes)
    try:
        np.ndarray(y_sorted.shape, dtype=y_sorted.dtype, buffer=shm_y.buf)[:] = y_sorted
        np.ndarray(w_sorted.shape, dtype=w_sorted.dtype, buffer=shm_w.buf)[:] = w_sorted

        try:
            import psutil  # type: ignore

            physical_cores = psutil.cpu_count(logical=False) or 1
        except Exception:
            physical_cores = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )

        n_workers = max(1, min(n_workers, physical_cores))
        n_workers = min(n_workers, n_perm)
        if n_workers <= 0:
            n_workers = 1

        if hasattr(os, "sched_getaffinity"):
            available_cpus = sorted(os.sched_getaffinity(0))  # type: ignore[arg-type]
        else:
            available_cpus = list(range(os.cpu_count() or 1))
        if not available_cpus:
            available_cpus = list(range(n_workers))

        cpu_plan: List[List[int]] = [
            [available_cpus[idx % len(available_cpus)]] for idx in range(n_workers)
        ]

        draws_per_worker = [n_perm // n_workers] * n_workers
        remainder = n_perm % n_workers
        for i in range(remainder):
            draws_per_worker[i] += 1

        seeds = [base_seed + 1 + i for i in range(n_workers)]

        progress_queue = ctx.Queue()
        disable_bar = not sys.stdout.isatty()
        results: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_meta_perm_worker_setup,
            initargs=(
                shm_y.name,
                shm_w.name,
                y_sorted.size,
                y_sorted.dtype.str,
                n_group0,
                total_weight,
                T_obs,
                progress_queue,
                total_weighted_y,
            ),
        ) as pool:
            futures = []
            for worker_idx, draws in enumerate(draws_per_worker):
                if draws <= 0:
                    continue
                futures.append(
                    pool.submit(
                        _meta_perm_worker_run_mean,
                        int(draws),
                        int(seeds[worker_idx]),
                        int(max(1, chunk)),
                        cpu_plan[worker_idx],
                    )
                )

            consumer_stop = threading.Event()
            consumer_thread: Optional[threading.Thread] = None

            with tqdm(
                total=n_perm,
                desc="Meta permutations (mean)",
                unit="draw",
                dynamic_ncols=True,
                smoothing=0.0,
                mininterval=0.5,
                ascii=True,
                disable=disable_bar,
            ) as pbar:

                if futures:

                    def _drain_progress() -> None:
                        while not consumer_stop.is_set():
                            try:
                                delta = progress_queue.get(timeout=0.5)
                            except Empty:
                                continue
                            if delta is None:
                                break
                            if delta:
                                pbar.update(int(delta))
                        while True:
                            try:
                                delta = progress_queue.get_nowait()
                            except Empty:
                                break
                            if not delta:
                                continue
                            pbar.update(int(delta))

                    consumer_thread = threading.Thread(
                        target=_drain_progress,
                        name="meta-perm-progress-mean",
                        daemon=True,
                    )
                    consumer_thread.start()

                try:
                    results = [f.result() for f in futures]
                finally:
                    consumer_stop.set()
                    progress_queue.put(None)
                    if consumer_thread is not None:
                        consumer_thread.join()

        progress_queue.close()
        progress_queue.join_thread()

        total_draws = sum(res["draws"] for res in results)
        if total_draws == 0:
            return {
                "p_perm_one_sided_upper": float("nan"),
                "p_perm_one_sided_lower": float("nan"),
                "p_perm_two_sided": float("nan"),
                "T_obs": float(T_obs),
                "total_draws": int(0),
                "count_upper": int(0),
                "count_lower": int(0),
                "count_two": int(0),
            }

        total_upper = sum(res["count_upper"] for res in results)
        total_lower = sum(res["count_lower"] for res in results)
        total_two = sum(res["count_two"] for res in results)

        p_one_upper = (1.0 + float(total_upper)) / (1.0 + float(total_draws))
        p_one_lower = (1.0 + float(total_lower)) / (1.0 + float(total_draws))
        p_two = (1.0 + float(total_two)) / (1.0 + float(total_draws))
    finally:
        shm_y.close()
        shm_w.close()
        shm_y.unlink()
        shm_w.unlink()

    return {
        "p_perm_one_sided_upper": float(p_one_upper),
        "p_perm_one_sided_lower": float(p_one_lower),
        "p_perm_two_sided": float(p_two),
        "T_obs": float(T_obs),
        "total_draws": int(total_draws),
        "count_upper": int(total_upper),
        "count_lower": int(total_lower),
        "count_two": int(total_two),
    }

# ------------------------- MAIN -------------------------

def main():
    log.info("=" * 80)
    log.info("Flat–Ramp–Flat Breakpoint Enrichment: Precision-Weighted Median + Permutation")
    log.info("=" * 80)
    log.info("")
    log.info("Sign convention: frf_delta = mu_edge - mu_mid")
    log.info("  POSITIVE = FST higher at breakpoints")
    log.info("  NEGATIVE = FST higher in middle")
    log.info("")

    falsta_path: Optional[Path] = None
    if FALSTA_CACHE.exists():
        falsta_path = FALSTA_CACHE
        log.info(f"Found cached FST data: {falsta_path}")
    elif (OUTDIR / "per_site_fst_output.falsta").exists():
        falsta_path = OUTDIR / "per_site_fst_output.falsta"
        log.info(f"Found FST data: {falsta_path}")

    if falsta_path is None:
        log.info("FST data not found locally, attempting download from GitHub Actions...")
        falsta_path = download_latest_artifact(GITHUB_REPO, WORKFLOW_NAME, ARTIFACT_NAME_FALSTA, OUTDIR)

    if falsta_path is None or not falsta_path.exists():
        log.error("")
        log.error("=" * 80)
        log.error("FST DATA NOT FOUND")
        log.error("=" * 80)
        log.error("Obtain per_site_fst_output.falsta and place it in the current directory or OUTDIR.")
        sys.exit(1)

    inversions = load_hudson_data(falsta_path)
    if not inversions:
        log.error("No inversions loaded. Exiting.")
        sys.exit(1)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info(f"Running FRF fitting and null permutations for {len(inversions)} inversions")
    log.info(f"Permutations per inversion: {N_PERMUTATIONS}")
    log.info("")

    n_workers = min(os.cpu_count() or 1, len(inversions))
    prepared_results: List[PreparedInversion] = []

    prioritized_inversions = sorted(
        inversions,
        key=lambda inv: estimate_inversion_workload(inv),
        reverse=True,
    )

    with ProcessPoolExecutor(
        max_workers=n_workers,
    ) as executor:
        futures = {}
        for inv in prioritized_inversions:
            future = executor.submit(
                fit_inversion_worker,
                (inv, N_PERMUTATIONS, DEFAULT_BLOCK_SIZE_WINDOWS),
            )
            futures[future] = inv
        completed = 0
        for future in as_completed(futures):
            inv = futures[future]
            prepared = future.result()
            prepared_results.append(prepared)
            completed += 1
            delta = prepared.result.frf_delta
            delta_str = f"{delta:+.4f}" if np.isfinite(delta) else "nan"
            plan_msg = "(rotation plan ready)" if prepared.plan is not None else "(no permutation plan)"
            log.info(
                f"[{completed}/{len(inversions)}] {inv.inv_key} "
                f"Δ={delta_str} {plan_msg}"
            )

    if not prepared_results:
        log.error("No FRF results obtained. Exiting.")
        sys.exit(1)

    plans = [prep.plan for prep in prepared_results if prep.plan is not None]
    results_by_key = {prep.result.inv_key: prep.result for prep in prepared_results}

    for plan_id, plan in enumerate(plans):
        plan.plan_id = plan_id

    chunk_tasks: List[Tuple[int, int]] = []
    for plan in plans:
        for chunk_index in range(plan.n_chunks):
            chunk_tasks.append((plan.plan_id, chunk_index))

    stats_by_inv = {
        plan.inv_key: {"n": 0, "sum": 0.0, "sum_sq": 0.0, "count_ge": 0}
        for plan in plans
    }

    if chunk_tasks:
        total_chunks = len(chunk_tasks)
        log.info(
            f"Running {total_chunks} permutation chunks across {len(plans)} inversions"
        )
        perm_workers = min(TOTAL_CPUS, total_chunks)
        perm_workers = max(1, perm_workers)
        inv_chunk_totals = {plan.inv_key: plan.n_chunks for plan in plans}
        inv_progress = {key: 0 for key in inv_chunk_totals}
        completed_inversions: set[str] = set()
        start_time = time.time()
        plan_payload = _build_permutation_plan_payload(plans)
        try:
            with ProcessPoolExecutor(
                max_workers=perm_workers,
                initializer=_permutation_worker_setup,
                initargs=(plan_payload,),
            ) as executor:
                future_map = {
                    executor.submit(run_permutation_chunk, task): task
                    for task in chunk_tasks
                }

                disable_bar = not sys.stdout.isatty()
                with tqdm(
                    total=len(chunk_tasks),
                    unit="chunk",
                    dynamic_ncols=True,
                    smoothing=0.0,
                    mininterval=0.5,
                    ascii=True,
                    disable=disable_bar,
                ) as pbar:
                    for future in as_completed(future_map):
                        chunk_result = future.result()

                        inv_key = chunk_result.inv_key
                        inv_progress[inv_key] = inv_progress.get(inv_key, 0) + 1

                        stats = stats_by_inv.get(inv_key)
                        if stats is not None:
                            stats["n"] += chunk_result.n_finite
                            stats["sum"] += chunk_result.sum_delta
                            stats["sum_sq"] += chunk_result.sum_sq_delta
                            stats["count_ge"] += chunk_result.count_ge_observed

                        pbar.update(1)

                        inv_total = inv_chunk_totals.get(inv_key, 0)
                        if (
                            inv_total
                            and inv_progress[inv_key] == inv_total
                            and inv_key not in completed_inversions
                        ):
                            completed_inversions.add(inv_key)
                            pbar.set_postfix_str(
                                f"inversions {len(completed_inversions)}/{len(inv_chunk_totals)}"
                            )
            total_elapsed = time.time() - start_time
            overall_rate = total_chunks / total_elapsed if total_elapsed > 0 else float("nan")
            log.info(
                "Permutation chunks finished in %s (average rate %s chunks/s)",
                _format_duration(total_elapsed),
                f"{overall_rate:.2f}" if math.isfinite(overall_rate) and overall_rate > 0 else "--",
            )
        finally:
            _release_permutation_plan_shared_memory(plan.plan_id for plan in plans)
    else:
        log.info("No permutation work required (insufficient valid windows or permutations set to zero)")

    for plan in plans:
        res = results_by_key[plan.inv_key]
        stats = stats_by_inv.get(plan.inv_key, {})
        n = stats.get("n", 0)
        if n > 1:
            sum_delta = stats["sum"]
            sum_sq_delta = stats["sum_sq"]
            mean_delta = float(sum_delta / n)
            res.frf_null_delta_mean = mean_delta
            variance = (sum_sq_delta - (sum_delta ** 2) / n) / (n - 1)
            variance = float(max(0.0, variance))
            res.frf_var_delta = variance
            res.frf_se_delta = float(math.sqrt(variance)) if variance > 0.0 else float("nan")
        else:
            res.frf_null_delta_mean = float("nan")
            res.frf_var_delta = float("nan")
            res.frf_se_delta = float("nan")
        res.usable_for_meta = (
            np.isfinite(res.frf_delta)
            and np.isfinite(res.frf_var_delta)
            and res.frf_var_delta > 0.0
        )

    all_results: List[FRFResult] = [prep.result for prep in prepared_results]

    for idx, res in enumerate(all_results, start=1):
        delta_str = f"{res.frf_delta:+.4f}" if np.isfinite(res.frf_delta) else "nan"
        var_str = f"{res.frf_var_delta:.3e}" if np.isfinite(res.frf_var_delta) else "nan"
        meta_flag = "usable" if res.usable_for_meta else "not-usable"
        log.info(
            f"[{idx}/{len(all_results)}] {res.inv_key} "
            f"Δ={delta_str} "
            f"(var={var_str}) "
            f"[{meta_flag}]"
        )
    if not all_results:
        log.error("No FRF results obtained. Exiting.")
        sys.exit(1)

    df = pd.DataFrame([vars(r) for r in all_results])

    inv_props = pd.read_csv(INV_PROPERTIES_PATH, sep="\t").copy()
    had_cols = set(inv_props.columns)
    if CHR_COL_INV not in had_cols:
        log.error(f"{INV_PROPERTIES_PATH} missing column: {CHR_COL_INV}")
        sys.exit(1)
    if START_COL_INV not in had_cols or END_COL_INV not in had_cols:
        log.error(f"{INV_PROPERTIES_PATH} missing Start/End columns")
        sys.exit(1)
    if STATUS_COL not in had_cols:
        log.error(f"{INV_PROPERTIES_PATH} missing column: {STATUS_COL}")
        sys.exit(1)

    inv_props["chrom_norm"] = inv_props[CHR_COL_INV].apply(normalize_chromosome)
    df["chrom_norm"] = df["chrom"]
    merged = pd.merge(
        df,
        inv_props[[CHR_COL_INV, START_COL_INV, END_COL_INV, STATUS_COL, "chrom_norm"]],
        left_on=["chrom_norm", "start", "end"],
        right_on=["chrom_norm", START_COL_INV, END_COL_INV],
        how="inner",
        validate="1:1",
    )
    merged = merged.rename(columns={STATUS_COL: "STATUS"})
    merged.drop(columns=["chrom_norm"], inplace=True)

    n_matched = merged.shape[0]
    log.info("")
    log.info(f"Matched {n_matched} inversions to inv_properties.tsv by chrom/start/end")

    merged["usable_for_meta"] = merged["usable_for_meta"].astype(bool)
    if "frf_null_delta_mean" in merged.columns:
        merged["frf_delta_centered"] = merged["frf_delta"] - merged["frf_null_delta_mean"]
    else:
        merged["frf_delta_centered"] = merged["frf_delta"]
    per_inv_out = OUTDIR / "per_inversion_frf_effects.tsv"
    merged.to_csv(per_inv_out, sep="\t", index=False)
    log.info(f"Per-inversion FRF results (with group labels) written to: {per_inv_out}")

    meta_inputs = prepare_meta_analysis_inputs(merged)
    if meta_inputs is None:
        log.warning("Meta-analysis could not be performed with available data.")
        return
    sub_meta, y_meta, s2_meta, group_meta, y_label = meta_inputs

    meta_results, meta_weights = run_precision_weighted_median_analysis(
        sub_meta,
        y_meta,
        s2_meta,
        group_meta,
        y_label,
    )

    log.info("")
    log.info("=" * 80)
    log.info("PRECISION-WEIGHTED MEDIAN ANALYSIS: group 0 (single) vs group 1 (recurrent)")
    log.info("=" * 80)

    tau2_desc = meta_results["tau2_descriptive"]
    median_recurrent = meta_results["median_recurrent"]
    median_single = meta_results["median_single"]
    delta_median = meta_results["delta_median"]

    n_total = int(meta_results["n_total"])
    n_single = int(meta_results["n_single"])
    n_recurrent = int(meta_results["n_recurrent"])

    log.info(f"Usable inversions for meta-analysis: {n_total}")
    log.info(f"  group 0 (single-event): {n_single}")
    log.info(f"  group 1 (recurrent):    {n_recurrent}")
    log.info(
        "Descriptive tau^2 from random-effects mean model (not used in weights): %.4e",
        tau2_desc,
    )
    log.info("Precision weights = 1 / s_i^2 with s_i^2 from inversion-level permutations")
    log.info("")
    log.info(
        f"Weighted median {y_label} (single,   group 0): {median_single:+.4f}"
    )
    log.info(
        f"Weighted median {y_label} (recurrent, group 1): {median_recurrent:+.4f}"
    )
    log.info(f"Median difference (single - recurrent):   {delta_median:+.4f}")
    log.info("")

    # Meta-level permutation p-values
    n_meta_workers = min(TOTAL_CPUS, max(1, (META_PERMUTATIONS + META_PERM_CHUNK - 1) // META_PERM_CHUNK))
    perm_out = meta_permutation_pvalue(
        y=y_meta,
        weights=meta_weights,
        group=group_meta,
        n_perm=META_PERMUTATIONS,
        chunk=META_PERM_CHUNK,
        base_seed=stable_seed_from_key("meta-permutation") + META_PERM_BASE_SEED,
        n_workers=n_meta_workers,
    )
    p_perm_one_upper = perm_out["p_perm_one_sided_upper"]
    p_perm_one_lower = perm_out["p_perm_one_sided_lower"]
    p_perm_two = perm_out["p_perm_two_sided"]
    log.info(
        "Exceedances (>=T_obs, <=T_obs, two-sided): %d/%d, %d/%d, %d/%d",
        int(perm_out["count_upper"]),
        int(perm_out["total_draws"]),
        int(perm_out["count_lower"]),
        int(perm_out["total_draws"]),
        int(perm_out["count_two"]),
        int(perm_out["total_draws"]),
    )
    log.info(
        "Permutation p (one-sided, single > recurrent): %r  [ (1+%d)/(1+%d) ]",
        p_perm_one_upper,
        int(perm_out["count_upper"]),
        int(perm_out["total_draws"]),
    )
    log.info(
        "Permutation p (one-sided, recurrent > single): %r  [ (1+%d)/(1+%d) ]",
        p_perm_one_lower,
        int(perm_out["count_lower"]),
        int(perm_out["total_draws"]),
    )
    log.info(
        "Permutation p (two-sided): %r  [ (1+%d)/(1+%d) ]",
        p_perm_two,
        int(perm_out["count_two"]),
        int(perm_out["total_draws"]),
    )
    log.info("")

    delta_mean, mean_single, mean_recurrent = weighted_mean_difference(
        y_meta, meta_weights, group_meta
    )
    if math.isfinite(mean_single) and math.isfinite(mean_recurrent) and math.isfinite(delta_mean):
        log.info("[MEAN] Weighted mean %s (single,   group 0): %+.4f", y_label, mean_single)
        log.info("[MEAN] Weighted mean %s (recurrent, group 1): %+.4f", y_label, mean_recurrent)
        log.info("[MEAN] Mean difference (single - recurrent):   %+.4f", delta_mean)
    else:
        log.warning("[MEAN] Unable to compute finite weighted means for both groups")

    perm_out_mean = meta_permutation_pvalue_mean(
        y=y_meta,
        weights=meta_weights,
        group=group_meta,
        n_perm=META_PERMUTATIONS,
        chunk=META_PERM_CHUNK,
        base_seed=stable_seed_from_key("meta-permutation-mean") + META_PERM_BASE_SEED,
        n_workers=n_meta_workers,
    )
    log.info(
        "[MEAN] Exceedances (>=T_obs, <=T_obs, two-sided): %d/%d, %d/%d, %d/%d",
        int(perm_out_mean["count_upper"]),
        int(perm_out_mean["total_draws"]),
        int(perm_out_mean["count_lower"]),
        int(perm_out_mean["total_draws"]),
        int(perm_out_mean["count_two"]),
        int(perm_out_mean["total_draws"]),
    )
    log.info(
        "[MEAN] Permutation p (one-sided, single > recurrent): %r  [ (1+%d)/(1+%d) ]",
        perm_out_mean["p_perm_one_sided_upper"],
        int(perm_out_mean["count_upper"]),
        int(perm_out_mean["total_draws"]),
    )
    log.info(
        "[MEAN] Permutation p (one-sided, recurrent > single): %r  [ (1+%d)/(1+%d) ]",
        perm_out_mean["p_perm_one_sided_lower"],
        int(perm_out_mean["count_lower"]),
        int(perm_out_mean["total_draws"]),
    )
    log.info(
        "[MEAN] Permutation p (two-sided): %r  [ (1+%d)/(1+%d) ]",
        perm_out_mean["p_perm_two_sided"],
        int(perm_out_mean["count_two"]),
        int(perm_out_mean["total_draws"]),
    )
    log.info("")

if __name__ == "__main__":
    main()
