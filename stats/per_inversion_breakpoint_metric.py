from __future__ import annotations

import logging
import sys
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
MIN_WINDOWS_PER_INVERSION = 20

N_PERMUTATIONS = 3_000
DEFAULT_BLOCK_SIZE_WINDOWS = 5
PERMUTATION_BATCH_SIZE = 256
MAX_INNER_THREADS = max(1, os.cpu_count() or 1)

FRF_MIN_EDGE_WINDOWS = 1
FRF_MIN_MID_WINDOWS = 1
MIN_BLOCKS_FOR_PERMUTATION = 5

AUTOCORR_MIN_PAIRS = 5
AUTOCORR_TARGET = 0.3
FRF_CANDIDATE_CHUNK_SIZE = 2048

META_PERMUTATIONS = 10000
META_PERM_CHUNK = 1000
META_PERM_BASE_SEED = 777

TOTAL_CPUS = max(1, os.cpu_count() or 1)
_ACTIVE_COUNTER = None
_TOTAL_CPUS_SHARED = TOTAL_CPUS

EPS_DENOM = 1e-12

_RE_HUD = re.compile(
    r">.*?hudson_pairwise_fst.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)",
    re.IGNORECASE,
)

# ------------------------- REPRODUCIBLE SEEDING -------------------------

def stable_seed_from_key(key: str) -> int:
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
    response = session.get(runs_url, params={
        "status": "success",
        "exclude_pull_requests": "true",
        "per_page": 1,
    })
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
    frf_a: float
    frf_b: float
    frf_var_delta: float
    frf_se_delta: float
    usable_for_meta: bool

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
                        data = parse_data_line("".join(current_data_lines))
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
                data = parse_data_line("".join(current_data_lines))
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

        num_clean = np.where(finite_num, numerator, 0.0)
        den_clean = np.where(finite_den, denominator, 0.0)

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

def estimate_correlation_length(
    fst: np.ndarray,
    weights: np.ndarray,
    max_global_block_size: int = DEFAULT_BLOCK_SIZE_WINDOWS,
) -> Tuple[int, float, int, int]:
    valid = np.isfinite(fst) & np.isfinite(weights)
    values = fst[valid]
    w = weights[valid]
    n = len(values)
    if n <= 1:
        block_size = max(1, min(max_global_block_size, max(n, 1)))
        return block_size, float("nan"), 0, 0
    if np.sum(w > 0) > 0:
        mean = float(np.average(values, weights=w))
    else:
        mean = float(np.mean(values))
    fluct = values - mean
    max_lag_candidate = n - 1
    if max_lag_candidate < 1:
        block_size = max(1, min(max_global_block_size, n))
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
        block_size = max(1, min(max_global_block_size, n))
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

def run_frf_search(
    fst_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    x_sorted: np.ndarray,
    candidates: Dict[str, np.ndarray],
    half_length_bp: float,
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

    eps = 1e-12
    row_idx = np.arange(n_samples)

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

        edge_sse = edge_sum_wf2 - np.where(edge_sum_w > eps, (edge_sum_wf ** 2) / np.maximum(edge_sum_w, eps), 0.0)
        mid_sse = mid_sum_wf2 - np.where(mid_sum_w > eps, (mid_sum_wf ** 2) / np.maximum(mid_sum_w, eps), 0.0)

        ramp_sum_w = np.take(prefix_w, ramp_end_chunk, axis=1) - np.take(prefix_w, ramp_start_chunk, axis=1)
        ramp_sum_wf = np.take(prefix_wf, ramp_end_chunk, axis=1) - np.take(prefix_wf, ramp_start_chunk, axis=1)
        ramp_sum_wf2 = np.take(prefix_wf2, ramp_end_chunk, axis=1) - np.take(prefix_wf2, ramp_start_chunk, axis=1)
        ramp_sum_wx = np.take(prefix_wx, ramp_end_chunk, axis=1) - np.take(prefix_wx, ramp_start_chunk, axis=1)
        ramp_sum_wx2 = np.take(prefix_wx2, ramp_end_chunk, axis=1) - np.take(prefix_wx2, ramp_start_chunk, axis=1)
        ramp_sum_wfx = np.take(prefix_wfx, ramp_end_chunk, axis=1) - np.take(prefix_wfx, ramp_start_chunk, axis=1)

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

# ------------------------- PER-INVERSION FRF + NULL -------------------------

def fit_inversion_frf_and_null(
    inversion: Inversion,
    n_permutations: int,
    default_block_size: int,
) -> FRFResult:
    with worker_activity():
        x_full, fst_full, w_full = compute_folded_distances(inversion)
        n_all = len(x_full)

        valid = np.isfinite(x_full) & np.isfinite(fst_full) & np.isfinite(w_full)
        n_valid = int(np.sum(valid))
        n_sites_total = sum(w.n_sites for w in inversion.windows)

        if n_valid < 3 or inversion.length <= 0:
            return FRFResult(
                inv_key=inversion.inv_key, chrom=inversion.chrom, start=inversion.start, end=inversion.end,
                length=inversion.length, n_windows=inversion.n_windows, n_sites=n_sites_total,
                block_size_windows=1, n_blocks=0,
                frf_mu_edge=float("nan"), frf_mu_mid=float("nan"), frf_delta=float("nan"),
                frf_a=float("nan"), frf_b=float("nan"),
                frf_var_delta=float("nan"), frf_se_delta=float("nan"),
                usable_for_meta=False,
            )

        x_v = x_full[valid]
        fst_v = fst_full[valid]
        w_v = w_full[valid]
        half_length = inversion.length / 2.0

        block_size_inv, _, _, _ = estimate_correlation_length(
            fst_v, w_v, max_global_block_size=default_block_size
        )

        order = np.argsort(x_v)
        x_sorted = x_v[order]
        fst_sorted = fst_v[order]
        w_sorted = w_v[order]

        candidates = build_exhaustive_frf_candidates(
            x_sorted, FRF_MIN_EDGE_WINDOWS, FRF_MIN_MID_WINDOWS
        )
        if candidates["edge_end"].size == 0:
            return FRFResult(
                inv_key=inversion.inv_key, chrom=inversion.chrom, start=inversion.start, end=inversion.end,
                length=inversion.length, n_windows=inversion.n_windows, n_sites=n_sites_total,
                block_size_windows=block_size_inv, n_blocks=0,
                frf_mu_edge=float("nan"), frf_mu_mid=float("nan"), frf_delta=float("nan"),
                frf_a=float("nan"), frf_b=float("nan"),
                frf_var_delta=float("nan"), frf_se_delta=float("nan"),
                usable_for_meta=False,
            )

        mu_edge_arr, mu_mid_arr, delta_arr, a_bp_arr, b_bp_arr = run_frf_search(
            fst_sorted[np.newaxis, :],
            w_sorted[np.newaxis, :],
            x_sorted,
            candidates,
            half_length,
        )
        frf_mu_edge = float(mu_edge_arr[0])
        frf_mu_mid = float(mu_mid_arr[0])
        frf_delta = float(delta_arr[0])
        frf_a = float(a_bp_arr[0])
        frf_b = float(b_bp_arr[0])

        blocks = precompute_block_structure(n_all, block_size_inv)
        n_blocks = len(blocks)
        can_permute = n_permutations > 0 and n_blocks >= MIN_BLOCKS_FOR_PERMUTATION

        frf_var_delta = float("nan")
        frf_se_delta = float("nan")

        if can_permute:
            orig_to_comp = np.full(n_all, -1, dtype=int)
            orig_to_comp[np.where(valid)[0]] = np.arange(n_valid, dtype=int)
            base_seed = stable_seed_from_key(inversion.inv_key)
            total = n_permutations
            batch_size = PERMUTATION_BATCH_SIZE
            n_batches = (total + batch_size - 1) // batch_size

            inner_threads = current_inner_threads()
            deltas_accum: List[np.ndarray] = []

            def run_batch(batch_index: int) -> np.ndarray:
                rng = np.random.default_rng(base_seed + 1 + batch_index)
                size = batch_size if (batch_index + 1) * batch_size <= total else (total - batch_index * batch_size)
                perm_indices = np.empty((size, n_valid), dtype=int)
                for j in range(size):
                    idx_full = generate_block_permutation_indices(blocks, rng)
                    idx_comp = orig_to_comp[idx_full]
                    row = idx_comp[idx_comp >= 0]
                    perm_indices[j, :] = row
                fst_perm_batch = fst_v[perm_indices][:, order]
                w_perm_batch = w_v[perm_indices][:, order]
                _, _, batch_deltas, _, _ = run_frf_search(
                    fst_perm_batch, w_perm_batch, x_sorted, candidates, half_length
                )
                return batch_deltas

            if inner_threads > 1 and n_batches > 1:
                with ThreadPoolExecutor(max_workers=inner_threads) as pool:
                    futures = {pool.submit(run_batch, b): b for b in range(n_batches)}
                    for fut in as_completed(futures):
                        deltas_accum.append(fut.result())
            else:
                for b in range(n_batches):
                    deltas_accum.append(run_batch(b))

            null_deltas = np.concatenate(deltas_accum, axis=0)[:total]
            finite_mask = np.isfinite(null_deltas)
            if np.sum(finite_mask) > 1:
                frf_var_delta = float(np.var(null_deltas[finite_mask], ddof=1))
                if frf_var_delta > 0.0:
                    frf_se_delta = float(math.sqrt(frf_var_delta))

        usable_for_meta = (
            np.isfinite(frf_delta) and np.isfinite(frf_var_delta) and frf_var_delta > 0.0
        )
        return FRFResult(
            inv_key=inversion.inv_key, chrom=inversion.chrom, start=inversion.start, end=inversion.end,
            length=inversion.length, n_windows=inversion.n_windows, n_sites=n_sites_total,
            block_size_windows=block_size_inv, n_blocks=n_blocks,
            frf_mu_edge=frf_mu_edge, frf_mu_mid=frf_mu_mid, frf_delta=frf_delta,
            frf_a=frf_a, frf_b=frf_b,
            frf_var_delta=frf_var_delta, frf_se_delta=frf_se_delta,
            usable_for_meta=bool(usable_for_meta),
        )

def fit_inversion_worker(args) -> FRFResult:
    inversion, n_permutations, default_block_size = args
    return fit_inversion_frf_and_null(inversion, n_permutations, default_block_size)

# ------------------------- RANDOM-EFFECTS META-REGRESSION -------------------------

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

def z_to_p_one_sided_greater(z: float) -> float:
    if not math.isfinite(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2.0))

def z_to_p_two_sided(z: float) -> float:
    if not math.isfinite(z):
        return float("nan")
    return math.erfc(abs(z) / math.sqrt(2.0))

def compute_meta_group_effect(y: np.ndarray, s2: np.ndarray, is_single: np.ndarray) -> Tuple[float, float, float, float]:
    X = np.column_stack([np.ones_like(is_single), is_single])
    tau2 = estimate_tau2_reml(y, s2, X)
    v = s2 + tau2
    w = 1.0 / v
    XtW = X.T * w
    XtWX = XtW @ X
    XtWy = XtW @ y
    try:
        beta_hat, inv_XtWX, _ = _solve_weighted_normal_equations(
            XtWX, XtWy, compute_inverse=True
        )
    except np.linalg.LinAlgError:
        return tau2, float("nan"), float("nan"), float("nan")
    beta_group = float(beta_hat[1])
    se_group = float(inv_XtWX[1, 1]) ** 0.5 if inv_XtWX[1, 1] > 0.0 else float("nan")
    z = beta_group / se_group if (math.isfinite(se_group) and se_group > 0.0) else float("nan")
    return tau2, beta_group, se_group, z

def run_random_effects_meta_regression(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    mask = (
        df["STATUS"].isin([0, 1])
        & df["usable_for_meta"]
        & np.isfinite(df["frf_delta"])
        & np.isfinite(df["frf_var_delta"])
        & (df["frf_var_delta"] > 0.0)
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        log.warning("No inversions with usable FRF variance and group labels")
        return None
    y = sub["frf_delta"].to_numpy(dtype=float)
    s2 = sub["frf_var_delta"].to_numpy(dtype=float)
    is_single = (sub["STATUS"] == 0).to_numpy(dtype=float)
    if np.all(is_single == 0.0) or np.all(is_single == 1.0):
        log.warning("Only one group present among usable inversions")
        return None
    tau2, beta_group, se_group, z = compute_meta_group_effect(y, s2, is_single)
    # Compute group means explicitly from the fitted model
    X = np.column_stack([np.ones_like(is_single), is_single])
    v = s2 + tau2
    w = 1.0 / v
    XtW = X.T * w
    XtWX = XtW @ X
    XtWy = XtW @ y
    
    try:
        # Solve the normal equations using stable linear solver
        beta_hat = np.linalg.solve(XtWX, XtWy)
        beta0 = float(beta_hat[0])
        mu_recurrent = beta0
        mu_single = beta0 + beta_group
    except np.linalg.LinAlgError:
        mu_recurrent = float("nan")
        mu_single = float("nan")

    p_one_sided = z_to_p_one_sided_greater(z)
    p_two_sided = z_to_p_two_sided(z)

    return {
        "tau2": float(tau2),
        "mu_recurrent": mu_recurrent,
        "mu_single": mu_single,
        "beta_group": float(beta_group),
        "se_group": float(se_group),
        "z_group": float(z),
        "p_one_sided_single_gt_recurrent": float(p_one_sided),
        "p_two_sided": float(p_two_sided),
        "n_total": float(sub.shape[0]),
        "n_single": float(int(np.sum(sub["STATUS"] == 0))),
        "n_recurrent": float(int(np.sum(sub["STATUS"] == 1))),
    }

# ------------------------- META-LEVEL PERMUTATION -------------------------

def _meta_perm_chunk_worker(args) -> np.ndarray:
    """Worker function for meta-level permutation - must be module-level for pickling."""
    start_idx, size, seed, y, s2, is_single, use_stat = args
    rng = np.random.default_rng(seed)
    stats = np.empty(size, dtype=float)
    for i in range(size):
        perm_labels = rng.permutation(is_single)
        _, b, se, z = compute_meta_group_effect(y, s2, perm_labels)
        stats[i] = b if use_stat == "beta" else z
    return stats

def meta_permutation_pvalue(
    y: np.ndarray,
    s2: np.ndarray,
    is_single: np.ndarray,
    n_perm: int,
    chunk: int,
    base_seed: int,
    n_workers: int,
    use_stat: str = "beta",
) -> Dict[str, float]:
    n = y.size
    n_workers = max(1, min(n_workers, (n_perm + chunk - 1) // chunk))
    obs_tau2, obs_beta, obs_se, obs_z = compute_meta_group_effect(y, s2, is_single)
    if use_stat == "beta":
        T_obs = obs_beta
    else:
        T_obs = obs_z

    tasks = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        idx = 0
        while idx < n_perm:
            take = min(chunk, n_perm - idx)
            seed = base_seed + 1 + (idx // chunk)
            args = (idx, take, seed, y, s2, is_single, use_stat)
            tasks.append(pool.submit(_meta_perm_chunk_worker, args))
            idx += take
        perm_stats: List[np.ndarray] = []
        for fut in as_completed(tasks):
            perm_stats.append(fut.result())

    T = np.concatenate(perm_stats, axis=0)[:n_perm]
    finite = np.isfinite(T)
    T = T[finite]
    if T.size == 0 or not math.isfinite(T_obs):
        return {
            "p_perm_one_sided": float("nan"),
            "p_perm_two_sided": float("nan"),
            "T_obs": float(T_obs),
        }
    p_one = (1.0 + float(np.sum(T >= T_obs))) / (1.0 + float(T.size))
    p_two = (1.0 + float(np.sum(np.abs(T) >= abs(T_obs)))) / (1.0 + float(T.size))
    return {"p_perm_one_sided": float(p_one), "p_perm_two_sided": float(p_two), "T_obs": float(T_obs)}

# ------------------------- MAIN -------------------------

def main():
    log.info("=" * 80)
    log.info("Flat–Ramp–Flat Breakpoint Enrichment: Random-Effects Meta-Analysis + Permutation")
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
    active_counter = mp.Value('i', 0)
    all_results: List[FRFResult] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(active_counter, TOTAL_CPUS),
    ) as executor:
        futures = {
            executor.submit(
                fit_inversion_worker,
                (inv, N_PERMUTATIONS, DEFAULT_BLOCK_SIZE_WINDOWS),
            ): inv
            for inv in inversions
        }
        completed = 0
        for future in as_completed(futures):
            inv = futures[future]
            res = future.result()
            all_results.append(res)
            completed += 1
            meta_flag = "usable" if res.usable_for_meta else "not-usable"
            log.info(
                f"[{completed}/{len(inversions)}] {res.inv_key} "
                f"Δ={res.frf_delta:+.4f} "
                f"(var={res.frf_var_delta:.3e}) "
                f"[blocks={res.n_blocks}, block_size={res.block_size_windows}] "
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
    per_inv_out = OUTDIR / "per_inversion_frf_effects.tsv"
    merged.to_csv(per_inv_out, sep="\t", index=False)
    log.info(f"Per-inversion FRF results (with group labels) written to: {per_inv_out}")

    meta_results = run_random_effects_meta_regression(merged)

    log.info("")
    log.info("=" * 80)
    log.info("RANDOM-EFFECTS META-ANALYSIS: group 0 (single) vs group 1 (recurrent)")
    log.info("=" * 80)

    if meta_results is None:
        log.warning("Meta-analysis could not be performed with available data.")
        return

    tau2 = meta_results["tau2"]
    mu_recurrent = meta_results["mu_recurrent"]
    mu_single = meta_results["mu_single"]
    beta_group = meta_results["beta_group"]
    se_group = meta_results["se_group"]
    z_group = meta_results["z_group"]
    p_one = meta_results["p_one_sided_single_gt_recurrent"]
    p_two = meta_results["p_two_sided"]

    n_total = int(meta_results["n_total"])
    n_single = int(meta_results["n_single"])
    n_recurrent = int(meta_results["n_recurrent"])

    log.info(f"Usable inversions for meta-analysis: {n_total}")
    log.info(f"  group 0 (single-event): {n_single}")
    log.info(f"  group 1 (recurrent):    {n_recurrent}")
    log.info(f"Estimated between-inversion variance tau^2: {tau2:.4e}")
    log.info("")
    log.info(f"Mean frf_delta (recurrent, group 1): {mu_recurrent:+.4f}")
    log.info(f"Mean frf_delta (single,   group 0): {mu_single:+.4f}")
    log.info(f"Difference (single - recurrent):   {beta_group:+.4f}")
    log.info(f"SE(diff): {se_group:.4f}")
    log.info(f"z-statistic: {z_group:.3f}")
    log.info(f"One-sided p (normal theory): {p_one:.4g}")
    log.info(f"Two-sided p (normal theory): {p_two:.4g}")
    log.info("")

    # Meta-level permutation p-values
    mask = (
        merged["STATUS"].isin([0, 1])
        & merged["usable_for_meta"]
        & np.isfinite(merged["frf_delta"])
        & np.isfinite(merged["frf_var_delta"])
        & (merged["frf_var_delta"] > 0.0)
    )
    sub = merged.loc[mask].copy()
    y = sub["frf_delta"].to_numpy(dtype=float)
    s2 = sub["frf_var_delta"].to_numpy(dtype=float)
    is_single = (sub["STATUS"] == 0).to_numpy(dtype=float)

    n_meta_workers = min(TOTAL_CPUS, max(1, (META_PERMUTATIONS + META_PERM_CHUNK - 1) // META_PERM_CHUNK))
    perm_out = meta_permutation_pvalue(
        y=y,
        s2=s2,
        is_single=is_single,
        n_perm=META_PERMUTATIONS,
        chunk=META_PERM_CHUNK,
        base_seed=stable_seed_from_key("meta-permutation") + META_PERM_BASE_SEED,
        n_workers=n_meta_workers,
        use_stat="beta",
    )
    p_perm_one = perm_out["p_perm_one_sided"]
    p_perm_two = perm_out["p_perm_two_sided"]
    log.info(f"Permutation p (one-sided, single > recurrent): {p_perm_one:.4g}")
    log.info(f"Permutation p (two-sided): {p_perm_two:.4g}")
    log.info("")

    if math.isfinite(p_perm_one) and p_perm_one < 0.05:
        log.info("Conclusion: Meta-analysis with label-exchange permutation supports higher breakpoint enrichment")
        log.info("             for group 0 (single-event) inversions than for group 1 (recurrent).")
    else:
        log.info("Conclusion: Meta-analysis with permutation does not provide strong evidence that")
        log.info("             group 0 (single-event) frf_delta exceeds group 1 (recurrent).")

if __name__ == "__main__":
    main()
