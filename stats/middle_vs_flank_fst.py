import re
import sys
import time
import math
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import shapiro, mannwhitneyu

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, to_rgb, to_hex
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fst_flanking_analysis_exact_mf_quadrants")

# ------------------------------------------------------------------------------
# Matplotlib base config (embed fonts, no TeX)
# ------------------------------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,      # Embed TrueType in PDF (Type 42)
    "ps.fonttype": 42,
    "text.usetex": False,    # DO NOT use TeX/mathtext; we draw exponents manually
    "axes.unicode_minus": True,
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# ------------------------------------------------------------------------------
# Window specifications (exact windows; no overlap)
# ------------------------------------------------------------------------------
def _format_bp_to_kb(value: int) -> str:
    value_kb = value / 1_000
    if value % 1_000 == 0:
        return f"{int(value_kb)} kb"
    formatted = f"{value_kb:.3f}".rstrip("0").rstrip(".")
    return f"{formatted} kb"


def _format_bp_slug(value: int) -> str:
    value_kb = value / 1_000
    if value % 1_000 == 0:
        return f"{int(value_kb)}kb"
    formatted = f"{value_kb:.3f}".rstrip("0").rstrip(".")
    return f"{formatted.replace('.', 'p')}kb"


@dataclass(frozen=True)
class WindowSpec:
    total_length: int
    flank_size: int
    middle_size: int

    def __post_init__(self) -> None:
        if self.total_length <= 0:
            raise ValueError("Total length must be positive.")
        if self.flank_size <= 0:
            raise ValueError("Flank size must be positive.")
        if self.middle_size <= 0:
            raise ValueError("Middle size must be positive.")
        if (2 * self.flank_size + self.middle_size) != self.total_length:
            raise ValueError(
                "Window sizes must satisfy: total = 2 * flank + middle."
            )

    @property
    def slug(self) -> str:
        return _format_bp_slug(self.total_length)

    @property
    def total_label(self) -> str:
        return _format_bp_to_kb(self.total_length)

    @property
    def flank_label(self) -> str:
        return _format_bp_to_kb(self.flank_size)

    @property
    def middle_label(self) -> str:
        return _format_bp_to_kb(self.middle_size)

    @property
    def description(self) -> str:
        return (
            f"{self.total_label} total "
            f"(flanks {self.flank_label} each, middle {self.middle_label})"
        )


TOTAL_WINDOW_LENGTHS = [20_000, 40_000, 60_000, 80_000, 100_000, 200_000]
WINDOW_SPECS: List[WindowSpec] = []
for total in TOTAL_WINDOW_LENGTHS:
    if total % 4 != 0:
        raise ValueError(
            f"Total window length {total:,} bp must be divisible by 4 to define symmetric flanks and middle."
        )
    flank = total // 4
    middle = total // 2
    WINDOW_SPECS.append(
        WindowSpec(total_length=total, flank_size=flank, middle_size=middle)
    )

# Output directories are organized by total window size (e.g., total_20kb)
def ensure_output_dir(spec: WindowSpec) -> Path:
    out = OUTPUT_BASE_DIR / f"total_{spec.slug}"
    out.mkdir(parents=True, exist_ok=True)
    return out

# ------------------------------------------------------------------------------
# Other Constants & File Paths
# ------------------------------------------------------------------------------
PERMUTATIONS = 10_000

FST_DATA_FILE = "per_site_fst_output.falsta"
INVERSION_FILE = "inv_info.tsv"
OUTPUT_BASE_DIR = Path("fst_analysis_results_exact_mf_quadrants")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

EPS_DENOM = 1e-12

# ------------------------------------------------------------------------------
# Category Mappings & Order
# ------------------------------------------------------------------------------
CATEGORY_ORDER = [
    "Single-event",
    "Recurrent",
]
CATEGORY_KEYS = [
    "single_event",
    "recurrent",
]

# ------------------------------------------------------------------------------
# Plotting Style & Colors
# ------------------------------------------------------------------------------
# Core colors for Single-event/Recurrent categories
COLOR_SINGLE_EVENT = "#1f3b78"   # dark blue
COLOR_RECURRENT    = "#8c2d7e"   # reddish purple

# Overlay (hatch) properties for Single-event/Recurrent status
OVERLAY_SINGLE_HATCH = "..." # Light gray dots (density adjusted from previous '. ')
OVERLAY_SINGLE_EDGE  = "#d9d9d9" # Light gray for hatch edge

OVERLAY_RECUR_HATCH  = "//"  # Dark gray diagonals
OVERLAY_RECUR_EDGE   = "#4a4a4a" # Dark gray for hatch edge

# Lightness factors for Flank (darker) and Middle (lighter)
FLANK_LIGHTNESS_FACTOR  = 0.75  # Make 25% darker
MIDDLE_LIGHTNESS_FACTOR = 1.25  # Make 25% lighter

# Other plot styles
AX_TEXT = "#333333"          # labels/ticks
POINT_SIZE    = 28
LINE_WIDTH    = 1.6
ALPHA_POINTS  = 0.60
VIOLIN_WIDTH  = 0.90
ALPHA_VIOLIN  = 0.72
# VIOLIN_EDGE and VIOLIN_EWIDTH are no longer used for violin bodies directly,
# but can remain for boxplots if desired.
VIOLIN_EDGE   = "#111111"
VIOLIN_EWIDTH = 0.6

BOXPLOT_WIDTH = 0.20
JITTER_MIN = 0.06
JITTER_MAX = 0.20
POS_X = {"Flank": 0.0, "Middle": 1.0}

# Helper to adjust color lightness
def adjust_lightness(color, factor):
    """
    Adjusts the lightness of a given color (hex or RGB tuple).
    Factor < 1 for darker, > 1 for lighter.
    """
    rgb = to_rgb(color)
    hls = mcolors.rgb_to_hsv(rgb)
    new_l = max(0.0, min(1.0, hls[1] * factor)) # Ensure lightness stays between 0 and 1
    new_rgb = mcolors.hsv_to_rgb((hls[0], new_l, hls[2]))
    return to_hex(new_rgb)

# ------------------------------------------------------------------------------
def normalize_chromosome(chrom: str) -> Optional[str]:
    if not isinstance(chrom, str):
        chrom = str(chrom)
    chrom = chrom.strip().lower()
    if chrom.startswith("chr_"):
        chrom_part = chrom[4:]
    elif chrom.startswith("chr"):
        chrom_part = chrom[3:]
    else:
        chrom_part = chrom
    if chrom_part.isalnum() or chrom_part in ("x", "y", "m", "w", "z") or "_" in chrom_part:
        return f"chr{chrom_part}"
    logger.warning(f"Could not normalize chromosome: '{chrom}'. Invalid format.")
    return None


def extract_fst_coordinates_from_header(header: str) -> Optional[dict]:
    if "hudson_pairwise_fst" not in header.lower():
        return None
    pattern = re.compile(
        r">.*?hudson_pairwise_fst.*?_chr_?([\w\.\-]+)_start_(\d+)_end_(\d+)",
        re.IGNORECASE,
    )
    match = pattern.search(header)
    if not match:
        logger.warning(f"Failed to extract coordinates from hudson_pairwise_fst header: {header[:70]}...")
        return None

    chrom_part, start_str, end_str = match.groups()
    chrom = normalize_chromosome(chrom_part)
    start = int(start_str) if start_str is not None else None
    end = int(end_str) if end_str is not None else None

    if chrom is None or start is None or end is None:
        logger.warning(f"Chromosome normalization or coordinate extraction failed for header: {header[:70]}...")
        return None
    if start >= end:
        logger.warning(f"Start >= End in header: {header[:70]}... ({start} >= {end})")
        return None

    return {"chrom": chrom, "start": start, "end": end}


def map_regions_to_inversions(inversion_df: pd.DataFrame) -> Tuple[dict, dict]:
    logger.info("Creating inversion region mappings...")
    recurrent_regions: Dict[str, List[Tuple[int, int]]] = {}
    single_event_regions: Dict[str, List[Tuple[int, int]]] = {}

    inversion_df["Start"] = pd.to_numeric(inversion_df["Start"], errors="coerce")
    inversion_df["End"] = pd.to_numeric(inversion_df["End"], errors="coerce")
    inversion_df["0_single_1_recur_consensus"] = pd.to_numeric(
        inversion_df["0_single_1_recur_consensus"], errors="coerce"
    )
    inversion_df["Chromosome"] = inversion_df["Chromosome"].astype(str)

    original_rows = len(inversion_df)
    inversion_df = inversion_df.dropna(
        subset=["Chromosome", "Start", "End", "0_single_1_recur_consensus"]
    )
    dropped_rows = original_rows - len(inversion_df)
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows from inversion info due to missing values.")

    for _, row in inversion_df.iterrows():
        chrom = normalize_chromosome(row["Chromosome"])
        if chrom is None:
            continue
        start = int(row["Start"])
        end = int(row["End"])
        is_recurrent = int(row["0_single_1_recur_consensus"]) == 1
        target = recurrent_regions if is_recurrent else single_event_regions
        target.setdefault(chrom, []).append((start, end))

    logger.info(
        f"Mapped {sum(len(v) for v in recurrent_regions.values())} recurrent and "
        f"{sum(len(v) for v in single_event_regions.values())} single-event regions."
    )
    return recurrent_regions, single_event_regions


def is_overlapping(s1: int, e1: int, s2: int, e2: int) -> bool:
    # Overlap / adjacency / <=1bp separation (inclusive coords)
    return (e1 + 2) >= s2 and (e2 + 2) >= s1


def determine_inversion_type(coords: dict, recurrent_regions: dict, single_event_regions: dict) -> str:
    chrom, start, end = coords.get("chrom"), coords.get("start"), coords.get("end")
    if not all([chrom, isinstance(start, int), isinstance(end, int)]):
        return "unknown"
    is_recur = any(is_overlapping(start, end, rs, re) for rs, re in recurrent_regions.get(chrom, []))
    is_single = any(is_overlapping(start, end, rs, re) for rs, re in single_event_regions.get(chrom, []))
    if is_recur and is_single:
        return "ambiguous"
    if is_recur:
        return "recurrent"
    if is_single:
        return "single_event"
    return "unknown"


def paired_permutation_test(
    x: np.ndarray, y: np.ndarray, num_permutations: int = PERMUTATIONS, use_median: bool = False
) -> float:
    if len(x) != len(y):
        logger.error(f"Input arrays x ({len(x)}) and y ({len(y)}) have different lengths for paired test.")
        return np.nan
    valid = ~np.isnan(x) & ~np.isnan(y)
    diffs = x[valid] - y[valid]
    n = len(diffs)
    if n < 2:
        logger.warning(f"Cannot perform permutation test: only {n} valid pairs found after NaN removal.")
        return np.nan

    stat_func = np.median if use_median else np.mean
    obs = stat_func(diffs)
    if np.isclose(obs, 0):
        return 1.0
    obs_abs = abs(obs)

    count = 0
    for _ in range(num_permutations):
        signs = np.random.choice([1, -1], size=n, replace=True)
        perm = stat_func(diffs * signs)
        if abs(perm) >= obs_abs:
            count += 1
    return count / num_permutations


def parse_falsta_data_line(line: str) -> Optional[np.ndarray]:
    try:
        values = line.split(",")
        data = np.full(len(values), np.nan, dtype=np.float32)
        for i, x in enumerate(values):
            val_str = x.strip()
            if val_str and val_str.upper() != "NA":
                data[i] = float(val_str)
        if np.all(np.isnan(data)):
            logger.debug("Parsed data line resulted in all NaNs, skipping.")
            return None
        return data
    except ValueError as e:
        logger.warning(f"ValueError parsing data line: {e}. Skipping line segment: {line[:50]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing data line: {e}. Skipping segment: {line[:50]}...", exc_info=True)
        return None


def _identify_fst_component(header: str) -> Optional[str]:
    lower = header.lower()
    if "numerator" in lower:
        return "numerator"
    if "denominator" in lower:
        return "denominator"
    return None


def load_fst_data(file_path: str | Path, min_total_length: int) -> List[dict]:
    logger.info(f"Loading Hudson FST data from {file_path}")
    logger.info(
        "Expecting paired numerator/denominator records with matching coordinates "
        f"and sequence length ≥ {min_total_length:,}."
    )
    start_time = time.time()

    file_path = Path(file_path)
    sequences_by_coord: Dict[Tuple[str, int, int], dict] = {}

    headers_read = 0
    sequences_parsed = 0
    skipped_not_fst = 0
    skipped_coord_error = 0
    skipped_component_error = 0
    skipped_data_error = 0
    skipped_short_total = 0
    skipped_length_mismatch = 0

    current_header: Optional[str] = None
    current_sequence_parts: List[str] = []
    current_coords: Optional[dict] = None
    current_component: Optional[str] = None
    header_is_valid = False

    def _finalize_current_sequence() -> None:
        nonlocal sequences_parsed, skipped_data_error, skipped_short_total, skipped_length_mismatch
        if not (header_is_valid and current_header and current_sequence_parts and current_coords and current_component):
            return
        full = "".join(current_sequence_parts)
        data = parse_falsta_data_line(full)
        if data is None:
            skipped_data_error += 1
            return
        length = len(data)
        if length < min_total_length:
            skipped_short_total += 1
            return

        key = (current_coords["chrom"], current_coords["start"], current_coords["end"])
        entry = sequences_by_coord.setdefault(
            key,
            {
                "coords": current_coords,
                "numerator": None,
                "denominator": None,
                "headers": {},
                "length": length,
            },
        )

        existing_length = entry.get("length")
        if existing_length is not None and existing_length != length:
            skipped_length_mismatch += 1
            return

        entry["length"] = length
        entry["headers"][current_component] = current_header
        entry[current_component] = data
        sequences_parsed += 1

    try:
        with file_path.open("r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    headers_read += 1
                    _finalize_current_sequence()

                    current_header = line
                    current_sequence_parts = []
                    current_coords = None
                    current_component = None
                    header_is_valid = False

                    if "hudson_pairwise_fst" not in current_header.lower():
                        skipped_not_fst += 1
                        current_header = None
                        continue

                    component = _identify_fst_component(current_header)
                    if component is None:
                        skipped_component_error += 1
                        current_header = None
                        continue

                    coords = extract_fst_coordinates_from_header(current_header)
                    if coords is None:
                        skipped_coord_error += 1
                        current_header = None
                        continue

                    current_coords = coords
                    current_component = component
                    header_is_valid = True
                elif header_is_valid and current_header:
                    current_sequence_parts.append(line)

            _finalize_current_sequence()

    except FileNotFoundError:
        logger.error(f"Fatal Error: FST data file not found at {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}", exc_info=True)
        return []

    skipped_incomplete_pairs = 0
    fst_sequences: List[dict] = []
    for (chrom, start, end), entry in sequences_by_coord.items():
        num = entry.get("numerator")
        den = entry.get("denominator")
        if num is None or den is None:
            skipped_incomplete_pairs += 1
            continue
        length = entry.get("length")
        if length is None:
            skipped_length_mismatch += 1
            continue
        if len(num) != len(den):
            skipped_length_mismatch += 1
            continue

        fst_sequences.append(
            {
                "header": f"{entry['headers'].get('numerator', '')}|{entry['headers'].get('denominator', '')}",
                "coords": {"chrom": chrom, "start": start, "end": end},
                "length": int(length),
                "numerator": num,
                "denominator": den,
                "headers": entry.get("headers", {}).copy(),
            }
        )

    elapsed_time = time.time() - start_time
    logger.info(
        f"Read {headers_read} headers, parsed {sequences_parsed} sequences (components) in {elapsed_time:.2f}s."
    )
    logger.info(
        f"Assembled {len(fst_sequences)} numerator/denominator pairs with length ≥ {min_total_length:,}."
    )
    logger.info(
        "Skipped counts: "
        f"Not 'hudson_pairwise_fst'={skipped_not_fst}, "
        f"Bad component tag={skipped_component_error}, "
        f"Coord Err={skipped_coord_error}, "
        f"Too Short={skipped_short_total}, "
        f"Data Parse Err={skipped_data_error}, "
        f"Length mismatch={skipped_length_mismatch}, "
        f"Incomplete pairs={skipped_incomplete_pairs}"
    )
    return fst_sequences


def _ratio_of_sums(numerator: np.ndarray, denominator: np.ndarray) -> Tuple[float, float, float, int]:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (den > EPS_DENOM)
    if not np.any(valid):
        return (np.nan, 0.0, 0.0, 0)
    num_sum = float(np.nansum(num[valid]))
    den_sum = float(np.nansum(den[valid]))
    if den_sum <= EPS_DENOM:
        return (np.nan, num_sum, den_sum, int(np.sum(valid)))
    ratio = num_sum / den_sum
    return (float(ratio), num_sum, den_sum, int(np.sum(valid)))


def calculate_flanking_stats(fst_sequences: List[dict], spec: WindowSpec) -> List[dict]:
    """Compute Hudson FST ratios for flanking and middle windows defined by ``spec``."""
    logger.info(
        "Calculating flanking and middle FST ratios for %d sequences (total=%s, flank=%s, middle=%s)...",
        len(fst_sequences),
        spec.total_label,
        spec.flank_label,
        spec.middle_label,
    )
    start_time = time.time()
    results: List[dict] = []
    skipped_too_short_total = 0
    skipped_nan_middle = 0
    skipped_nan_flanks = 0

    for seq in fst_sequences:
        num = seq.get("numerator")
        den = seq.get("denominator")
        if num is None or den is None:
            continue

        L = len(num)
        if L != len(den):
            logger.warning("Numerator/denominator length mismatch for %s; skipping.", seq.get("header", "<unknown>"))
            continue
        if L < spec.total_length:
            skipped_too_short_total += 1
            continue

        beginning_num = num[:spec.flank_size]
        beginning_den = den[:spec.flank_size]
        ending_num = num[-spec.flank_size:]
        ending_den = den[-spec.flank_size:]

        start_middle = (L - spec.middle_size) // 2
        end_middle = start_middle + spec.middle_size
        middle_num = num[start_middle:end_middle]
        middle_den = den[start_middle:end_middle]

        if not (
            spec.flank_size <= start_middle
            and end_middle <= (L - spec.flank_size)
        ):
            logger.warning(
                f"Middle window overlaps flanks (len={L}, start_mid={start_middle}, end_mid={end_middle}). Skipping."
            )
            continue

        begin_ratio, begin_num_sum, begin_den_sum, begin_cnt = _ratio_of_sums(beginning_num, beginning_den)
        end_ratio, end_num_sum, end_den_sum, end_cnt = _ratio_of_sums(ending_num, ending_den)
        middle_ratio, middle_num_sum, middle_den_sum, middle_cnt = _ratio_of_sums(middle_num, middle_den)

        flank_num_sum = begin_num_sum + end_num_sum
        flank_den_sum = begin_den_sum + end_den_sum
        flank_cnt = begin_cnt + end_cnt
        flanking_ratio = np.nan
        if flank_den_sum > EPS_DENOM:
            flanking_ratio = flank_num_sum / flank_den_sum

        if np.isnan(middle_ratio):
            skipped_nan_middle += 1
            continue
        if np.isnan(flanking_ratio):
            skipped_nan_flanks += 1
            continue

        stats = {
            "header": seq.get("header"),
            "coords": seq.get("coords"),
            "length": seq.get("length", L),
            "window_total_bp": spec.total_length,
            "window_flank_bp": spec.flank_size,
            "window_middle_bp": spec.middle_size,
            "beginning_ratio": begin_ratio,
            "ending_ratio": end_ratio,
            "middle_ratio": middle_ratio,
            "flanking_ratio": flanking_ratio,
            "beginning_informative_sites": begin_cnt,
            "ending_informative_sites": end_cnt,
            "middle_informative_sites": middle_cnt,
            "flanking_informative_sites": flank_cnt,
            "beginning_num_sum": begin_num_sum,
            "beginning_den_sum": begin_den_sum,
            "ending_num_sum": end_num_sum,
            "ending_den_sum": end_den_sum,
            "middle_num_sum": middle_num_sum,
            "middle_den_sum": middle_den_sum,
            "flanking_num_sum": flank_num_sum,
            "flanking_den_sum": flank_den_sum,
        }

        results.append(stats)

    elapsed_time = time.time() - start_time
    logger.info(f"Calculated stats for {len(results)} sequences in {elapsed_time:.2f}s.")
    if skipped_too_short_total:
        logger.warning(
            f"Skipped {skipped_too_short_total} sequences too short for exact windows (need ≥ {spec.total_length:,})."
        )
    if skipped_nan_middle:
        logger.warning(f"Skipped {skipped_nan_middle} sequences (no informative middle sites).")
    if skipped_nan_flanks:
        logger.warning(f"Skipped {skipped_nan_flanks} sequences (no informative flanking sites).")
    return results


def categorize_sequences(flanking_stats: List[dict], recurrent_regions: dict, single_event_regions: dict) -> dict:
    logger.info("Categorizing sequences based on overlap with inversion regions...")
    categories = {
        "single_event": [],
        "recurrent": [],
    }

    for seq_stats in flanking_stats:
        coords = seq_stats.get("coords")
        if not coords:
            seq_stats["inv_class"] = "unknown"
            continue

        inv_type = determine_inversion_type(coords, recurrent_regions, single_event_regions)
        seq_stats["inv_class"] = inv_type

        if inv_type == "single_event":
            categories["single_event"].append(seq_stats)
        elif inv_type == "recurrent":
            categories["recurrent"].append(seq_stats)
        # ambiguous/unknown not included in these bins

    for disp, key in zip(CATEGORY_ORDER, CATEGORY_KEYS):
        logger.info(f"  {disp}: {len(categories.get(key, []))}")
    return categories


def _extract_flank_minus_middle_diffs(seq_stats: List[dict]) -> np.ndarray:
    """Return an array of (flank - middle) ratios for sequences with finite values."""
    diffs: List[float] = []
    for stats in seq_stats:
        flank = stats.get("flanking_ratio")
        middle = stats.get("middle_ratio")
        if flank is None or middle is None:
            continue
        if not (np.isfinite(flank) and np.isfinite(middle)):
            continue
        diffs.append(float(flank) - float(middle))
    if not diffs:
        return np.asarray([], dtype=float)
    return np.asarray(diffs, dtype=float)


def single_vs_recurrent_one_sided_test(categories: dict) -> dict:
    """One-sided Mann-Whitney U test on (flank - middle) differences."""
    se_diffs = _extract_flank_minus_middle_diffs(categories.get("single_event", []))
    rec_diffs = _extract_flank_minus_middle_diffs(categories.get("recurrent", []))

    result = {
        "n_single_event": int(se_diffs.size),
        "n_recurrent": int(rec_diffs.size),
        "single_event_mean_diff": float(np.nanmean(se_diffs)) if se_diffs.size else np.nan,
        "recurrent_mean_diff": float(np.nanmean(rec_diffs)) if rec_diffs.size else np.nan,
        "mannwhitney_u": np.nan,
        "p_one_sided": np.nan,
        "alternative": "single_event>recurrent",
    }

    if se_diffs.size and rec_diffs.size:
        try:
            stat, p = mannwhitneyu(se_diffs, rec_diffs, alternative="greater")
            result["mannwhitney_u"] = float(stat)
            result["p_one_sided"] = float(p)
        except ValueError:
            pass

    return result


def pooled_group_estimates(categories: dict, reps: int = 2000, seed: int = 1337) -> pd.DataFrame:
    """Compute pooled Hudson FST (ratio-of-sums) per group and window with bootstrap SE."""

    def _collect(seq_stats: List[dict], window: str) -> Tuple[np.ndarray, np.ndarray]:
        if window == "Flank":
            num = np.asarray([s.get("flanking_num_sum", np.nan) for s in seq_stats], dtype=float)
            den = np.asarray([s.get("flanking_den_sum", np.nan) for s in seq_stats], dtype=float)
        else:
            num = np.asarray([s.get("middle_num_sum", np.nan) for s in seq_stats], dtype=float)
            den = np.asarray([s.get("middle_den_sum", np.nan) for s in seq_stats], dtype=float)
        valid = np.isfinite(num) & np.isfinite(den) & (den > EPS_DENOM)
        return num[valid], den[valid]

    def _pooled(num: np.ndarray, den: np.ndarray) -> Tuple[float, float, float, int]:
        if num.size == 0:
            return (np.nan, 0.0, 0.0, 0)
        num_sum = float(np.sum(num))
        den_sum = float(np.sum(den))
        if den_sum <= EPS_DENOM:
            return (np.nan, num_sum, den_sum, int(num.size))
        return (num_sum / den_sum, num_sum, den_sum, int(num.size))

    def _bootstrap_se(num: np.ndarray, den: np.ndarray, reps: int, seed: int) -> float:
        if num.size == 0:
            return float("nan")
        rng = np.random.default_rng(seed)
        estimates = np.empty(reps, dtype=float)
        for i in range(reps):
            indices = rng.integers(0, num.size, size=num.size)
            num_sum = float(np.sum(num[indices]))
            den_sum = float(np.sum(den[indices]))
            if den_sum <= EPS_DENOM:
                estimates[i] = np.nan
            else:
                estimates[i] = num_sum / den_sum
        finite = estimates[np.isfinite(estimates)]
        if finite.size < 2:
            return float("nan")
        return float(np.std(finite, ddof=1))

    groups = {
        "Single-event": list(categories.get("single_event", [])),
        "Recurrent": list(categories.get("recurrent", [])),
        "Overall": list(categories.get("single_event", [])) + list(categories.get("recurrent", [])),
    }

    rows = []
    for group_name, seqs in groups.items():
        for window in ("Flank", "Middle"):
            num_vals, den_vals = _collect(seqs, window)
            pooled, num_sum, den_sum, n_inv = _pooled(num_vals, den_vals)
            se = _bootstrap_se(num_vals, den_vals, reps=reps, seed=seed)
            rows.append({
                "group": group_name,
                "window": window,
                "pooled_fst": pooled,
                "bootstrap_se": se,
                "numerator_sum": num_sum,
                "denominator_sum": den_sum,
                "n_inversions_contributing": n_inv,
            })

    return pd.DataFrame(rows)


def perform_statistical_tests(categories: dict, all_sequences_stats: List[dict]) -> dict:
    logger.info("Performing paired permutation tests (Middle vs Flanking FST; ratio diffs)...")
    test_results: Dict[str, dict] = {}

    # Build map with display names
    display_map = {
        "Single-event": categories.get("single_event", []),
        "Recurrent": categories.get("recurrent", []),
        "Overall": all_sequences_stats,
    }

    for name, seqs in display_map.items():
        test_results[name] = {"mean_p": np.nan, "mean_normality_p": np.nan, "n_valid_pairs": 0}
        if len(seqs) < 2:
            continue
        f_means = np.array([s["flanking_ratio"] for s in seqs], dtype=float)
        m_means = np.array([s["middle_ratio"] for s in seqs], dtype=float)
        valid = ~np.isnan(f_means) & ~np.isnan(m_means)
        n_valid = int(np.sum(valid))
        test_results[name]["n_valid_pairs"] = n_valid
        if n_valid < 2:
            continue

        p = paired_permutation_test(m_means[valid], f_means[valid], use_median=False)
        test_results[name]["mean_p"] = p

        if n_valid >= 3:
            diffs = m_means[valid] - f_means[valid]
            if len(np.unique(diffs)) > 1:
                try:
                    _, sh_p = shapiro(diffs)
                    test_results[name]["mean_normality_p"] = sh_p
                except ValueError:
                    pass

    return test_results


def print_summary_statistics(sequences: List[dict], group_name: str):
    """Calculates and prints summary statistics for a given group of sequences."""
    if not sequences:
        logger.info(f"\n--- Summary Statistics for {group_name} (n=0) ---")
        logger.info("  No sequences in this group.")
        return

    flank_means = [s['flanking_ratio'] for s in sequences]
    middle_means = [s['middle_ratio'] for s in sequences]

    n = len(sequences)

    logger.info(f"\n--- Summary Statistics for {group_name} (n={n}) ---")
    logger.info(f"  Flank Region (Hudson FST):  Mean={np.nanmean(flank_means):.8f}, Median={np.nanmedian(flank_means):.8f}")
    logger.info(f"  Middle Region (Hudson FST): Mean={np.nanmean(middle_means):.8f}, Median={np.nanmedian(middle_means):.8f}")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def format_plain_no_e(x: float, max_decimals: int = 8) -> str:
    if not np.isfinite(x):
        return ""
    s = f"{x:.{max_decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"

def _format_p_value_for_annotation(p: float) -> str:
    """
    Format a p-value for compact axis annotation.
    Returns 'NA' if not finite, '<1e-6' for very small values, otherwise a plain decimal.
    """
    if not np.isfinite(p):
        return "NA"
    if p < 1e-6:
        return "<1e-6"
    return format_plain_no_e(p, max_decimals=8)

# ------------------------------------------------------------------------------
def _split_forced_sci(x: float, sig: int = 3) -> Tuple[str, Optional[str]]:
    if not np.isfinite(x):
        return ("N/A", None)
    ax = abs(x)
    if ax == 0:
        return ("0", None)
    if 1e-3 <= ax < 1e4:
        s = f"{x:.{sig}g}"
        try:
            if np.isclose(float(s), round(float(s)), atol=10**-(sig+1)):
                s = str(int(round(float(s))))
        except Exception:
            pass
        return (s, None)
    exp = int(np.floor(np.log10(ax)))
    mant = x / (10 ** exp)
    mant_str = f"{mant:.{sig}g}"
    try:
        if np.isclose(float(mant_str), round(float(mant_str)), atol=10**-(sig+1)):
            mant_str = str(int(round(float(mant_str))))
    except Exception:
        pass
    base = f"{mant_str} × 10"
    return (base, f"{exp:d}")

# ------------------------------------------------------------------------------
# Plot helpers for the MF (Middle vs Flank) quadrant layout
# ------------------------------------------------------------------------------
def _deterministic_rng(seed_key: str) -> np.random.RandomState:
    seed = int.from_bytes(hashlib.md5(hashlib.md5(seed_key.encode("utf-8")).digest()).digest()[:4], "little") # Double hash for extra robustness
    return np.random.RandomState(seed)

def _inward_jitter(rng: np.random.RandomState) -> float:
    return float(rng.uniform(JITTER_MIN, JITTER_MAX))

def _overlay_boxplots(ax, vals, pos):
    """Slim boxplots centered at each position."""
    for data, x in zip(vals, pos):
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        bp = ax.boxplot(
            [arr],
            positions=[x],
            widths=BOXPLOT_WIDTH,
            vert=True,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            boxprops=dict(facecolor="white", edgecolor=VIOLIN_EDGE, linewidth=1.1),
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color=VIOLIN_EDGE, linewidth=1.0),
            capprops=dict(color=VIOLIN_EDGE, linewidth=1.0),
        )
        for part in ["boxes", "medians", "whiskers", "caps"]:
            for artist in bp[part]:
                artist.set_zorder(5)

def _draw_two_violins(ax, vals, pos, flank_color, middle_color, hatch_pattern, hatch_edge_color):
    """
    Draws two half-violins (Flank left half, Middle right half) with specific
    base colors and a hatch pattern.
    """
    v = ax.violinplot(
        dataset=vals,
        positions=pos,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Get the axis limits, which MUST be set before this function is called.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    plot_height = ymax - ymin

    # The violin bodies are returned in order: [Flank, Middle]
    for i, body in enumerate(v["bodies"]):
        # Apply the common hatch and alpha properties first
        body.set_hatch(hatch_pattern)
        body.set_edgecolor(hatch_edge_color)
        body.set_linewidth(0.0)  # No outer edge for the violin body itself
        body.set_alpha(ALPHA_VIOLIN)

        if i == 0:  # This is the Flank violin at pos[0]
            body.set_facecolor(flank_color)
            # Create a clipping rectangle that covers the LEFT side of the plot,
            # ending at the center of this violin.
            clip_width = pos[0] - xmin
            clip_rect = Rectangle((xmin, ymin), clip_width, plot_height, transform=ax.transData)
            body.set_clip_path(clip_rect)

        elif i == 1:  # This is the Middle violin at pos[1]
            body.set_facecolor(middle_color)
            # Create a clipping rectangle that covers the RIGHT side of the plot,
            # starting from the center of this violin.
            clip_start_x = pos[1]
            clip_width = xmax - clip_start_x
            clip_rect = Rectangle((clip_start_x, ymin), clip_width, plot_height, transform=ax.transData)
            body.set_clip_path(clip_rect)

def _safe_log2_ratio(m: float, f: float, eps: float = 1e-12) -> float:
    """Safely compute log2((m + eps) / (f + eps)) allowing for negative Hudson FST values."""
    if not (np.isfinite(m) and np.isfinite(f)):
        return float("nan")
    if (m + eps) <= 0 or (f + eps) <= 0:
        return float("nan")
    try:
        return math.log2((m + eps) / (f + eps))
    except ValueError:
        return float("nan")


def _paired_lines_and_points(ax, rows, cmap, norm, category_key: str, flank_point_color, middle_point_color):
    """Per sequence: draw two points (Flank, Middle) and the connecting line colored by log2(M/MF)."""
    EPS = 1e-12
    for r in rows:
        f = r["flanking_ratio"]
        m = r["middle_ratio"]
        if not (np.isfinite(f) and np.isfinite(m)):
            continue
        rng = _deterministic_rng(r.get("header", "") + "|" + category_key)
        j = _inward_jitter(rng)

        x_f = POS_X["Flank"]  + j    # jitter rightwards
        x_m = POS_X["Middle"] - j    # jitter leftwards

        log2fc = _safe_log2_ratio(m, f, EPS)
        if not np.isfinite(log2fc):
            log2fc = 0.0
        c = cmap(norm(log2fc))

        ax.plot([x_f, x_m], [f, m], color=c, linewidth=LINE_WIDTH, alpha=0.98, zorder=3, solid_capstyle="round")
        # Use specific point colors derived from the base color
        ax.scatter([x_f], [f], s=POINT_SIZE, c=flank_point_color, edgecolors="black", linewidths=0.5, alpha=ALPHA_POINTS, zorder=6)
        ax.scatter([x_m], [m], s=POINT_SIZE, c=middle_point_color,   edgecolors="black", linewidths=0.5, alpha=ALPHA_POINTS, zorder=6)


def _prepare_category_bins(categories: dict) -> Dict[str, List[dict]]:
    """Ensure the bins are present for the two requested panels (single-event, recurrent)."""
    bins = {
        "single_event": categories.get("single_event", []),
        "recurrent": categories.get("recurrent", []),
    }
    return bins

def _collect_all_pairs_for_scale(categories: dict) -> np.ndarray:
    """Collect log2 ratios across all bins for a stable, shared color scale."""
    EPS = 1e-12
    vals = []
    for key in ["single_event", "recurrent"]:
        for s in categories.get(key, []):
            f = s.get("flanking_ratio", np.nan)
            m = s.get("middle_ratio", np.nan)
            if np.isfinite(f) and np.isfinite(m):
                log2fc = _safe_log2_ratio(m, f, EPS)
                if np.isfinite(log2fc):
                    vals.append(log2fc)
    return np.asarray(vals, dtype=float)

def _draw_right_key(rax):
    """Right-side legend showing category and region encodings."""
    rax.set_xlim(0, 1)
    rax.set_ylim(0, 1)
    rax.axis("off")

    legend_elements = [
        {"label": "Single-event", "facecolor": COLOR_SINGLE_EVENT, "hatch": OVERLAY_SINGLE_HATCH, "edgecolor": OVERLAY_SINGLE_EDGE, "alpha": ALPHA_VIOLIN, "y": 0.78},
        {"label": "Recurrent",    "facecolor": COLOR_RECURRENT,    "hatch": OVERLAY_RECUR_HATCH,  "edgecolor": OVERLAY_RECUR_EDGE,  "alpha": ALPHA_VIOLIN, "y": 0.64},
        {"label": "Flanking Region", "facecolor": adjust_lightness(COLOR_SINGLE_EVENT, FLANK_LIGHTNESS_FACTOR), "hatch": None, "edgecolor": "none", "alpha": ALPHA_VIOLIN, "y": 0.38},
        {"label": "Middle Region",   "facecolor": adjust_lightness(COLOR_SINGLE_EVENT, MIDDLE_LIGHTNESS_FACTOR), "hatch": None, "edgecolor": "none", "alpha": ALPHA_VIOLIN, "y": 0.24},
    ]

    S = 0.09
    X0 = 0.08

    rax.text(X0, 0.90, "Inversion Category:", ha="left", va="bottom", fontsize=12, color=AX_TEXT, fontweight="bold")
    rax.text(X0, 0.48, "Region Type:", ha="left", va="bottom", fontsize=12, color=AX_TEXT, fontweight="bold")

    for entry in legend_elements:
        hatch_to_pass = entry["hatch"] if entry["hatch"] else None
        sq = Rectangle((X0, entry["y"]), S, S, transform=rax.transAxes,
                       facecolor=entry["facecolor"], edgecolor=entry["edgecolor"],
                       hatch=hatch_to_pass, linewidth=0.0, alpha=entry["alpha"], clip_on=False)
        rax.add_patch(sq)
        rax.text(X0 + S + 0.02, entry["y"] + S / 2, entry["label"], transform=rax.transAxes,
                 ha="left", va="center", fontsize=11, color=AX_TEXT)


def _annotate_p_bracket(ax, x1: float, x2: float, y: float, label: str) -> None:
    """
    Draw a bracket between x1 and x2 at height y and center the label above the bracket.
    The bracket consists of two short vertical ticks joined by a horizontal line.
    """
    ymin, ymax = ax.get_ylim()
    span = float(ymax - ymin)
    tick = span * 0.015
    y0 = y - tick
    ax.plot([x1, x1, x2, x2], [y0, y, y, y0], color=AX_TEXT, linewidth=1.1, zorder=10, clip_on=False)
    ax.text((x1 + x2) * 0.5, y + tick * 0.6, label, ha="center", va="bottom", color=AX_TEXT, fontsize=11, zorder=10, clip_on=False)

def create_mf_quadrant_violins(
    categories: dict, test_results: dict, output_dir: Path, spec: WindowSpec
) -> Optional[plt.Figure]:
    """Build a 1×2 panel of middle vs flank violins for Hudson FST."""
    logger.info(
        "Creating Middle vs Flank FST violins for %s...",
        spec.description,
    )
    start_time = time.time()

    bins = _prepare_category_bins(categories)

    all_vals = []
    for key in bins:
        all_vals.extend([s.get("flanking_ratio", np.nan) for s in bins[key]])
        all_vals.extend([s.get("middle_ratio",  np.nan) for s in bins[key]])
    all_vals = np.asarray(all_vals, dtype=float)
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size == 0:
        y_lo, y_hi = 0.0, 1.0
    else:
        y_min, y_max = np.nanmin(finite), np.nanmax(finite)
        pad = (y_max - y_min) * 0.06 if y_max > y_min else 0.1
        y_lo, y_hi = max(0.0, y_min - pad), y_max + pad

    all_log2 = _collect_all_pairs_for_scale(bins)
    finite_log2 = all_log2[np.isfinite(all_log2)]
    if finite_log2.size == 0:
        v = 1.0
    else:
        v = float(np.nanpercentile(np.abs(finite_log2), 98))
        if not np.isfinite(v) or v <= 0:
            vmax = float(np.nanmax(np.abs(finite_log2))) if finite_log2.size else 1.0
            v = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(12, 6.6))
    fig.suptitle(
        f"Middle vs Flank Hudson FST — {spec.total_label} total (flanks {spec.flank_label} each, middle {spec.middle_label})",
        fontsize=16,
        color=AX_TEXT,
        y=0.97,
    )
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.65],
        height_ratios=[1.0, 0.22],
        wspace=0.24,
        hspace=0.32,
    )

    ax_map = {
        "single_event": fig.add_subplot(gs[0, 0]),
        "recurrent": fig.add_subplot(gs[0, 1]),
    }
    rax = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[1, 0:3])

    title_map = {
        "single_event": "Single-event",
        "recurrent": "Recurrent",
    }

    for key in ["single_event", "recurrent"]:
        ax = ax_map[key]
        rows = bins.get(key, [])

        if key == "single_event":
            base_color = COLOR_SINGLE_EVENT
            hatch_pattern = OVERLAY_SINGLE_HATCH
            hatch_edge_color = OVERLAY_SINGLE_EDGE
        else:
            base_color = COLOR_RECURRENT
            hatch_pattern = OVERLAY_RECUR_HATCH
            hatch_edge_color = OVERLAY_RECUR_EDGE

        flank_color = adjust_lightness(base_color, FLANK_LIGHTNESS_FACTOR)
        middle_color = adjust_lightness(base_color, MIDDLE_LIGHTNESS_FACTOR)

        v_flank = [s.get("flanking_ratio", np.nan) for s in rows]
        v_mid = [s.get("middle_ratio", np.nan) for s in rows]
        vals = [np.asarray(v_flank, dtype=float), np.asarray(v_mid, dtype=float)]
        pos = [POS_X["Flank"], POS_X["Middle"]]

        ax.set_title(title_map[key], fontsize=14, color=AX_TEXT, pad=18)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(y_lo, y_hi)
        _draw_two_violins(ax, vals, pos, flank_color, middle_color, hatch_pattern, hatch_edge_color)
        _overlay_boxplots(ax, vals, pos)
        _paired_lines_and_points(
            ax,
            rows,
            cmap=cmap,
            norm=norm,
            category_key=key,
            flank_point_color=flank_color,
            middle_point_color=middle_color,
        )

        ax.set_xticks([POS_X["Flank"], POS_X["Middle"]])
        ax.set_xticklabels(["Flank", "Middle"], fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_plain_no_e(v, max_decimals=8)))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        disp_name = title_map[key]
        tr = test_results.get(disp_name, {})
        perm_p = _format_p_value_for_annotation(tr.get("mean_p", np.nan))
        label = f"p={perm_p}"

        has_vals = (np.isfinite(vals[0]).any() or np.isfinite(vals[1]).any())
        data_max = np.nanmax(np.concatenate([vals[0], vals[1]])) if has_vals else y_hi
        y_range = y_hi - y_lo
        y_bracket = min(y_hi - 0.02 * y_range, data_max + 0.06 * y_range)
        _annotate_p_bracket(ax, POS_X["Flank"], POS_X["Middle"], y_bracket, label)

    ax_map["single_event"].set_ylabel("Hudson FST", fontsize=16, color=AX_TEXT)

    _draw_right_key(rax)

    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.set_xticks([])
    cax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$\log_{2}\!\left(FST_{\mathrm{Middle}} / FST_{\mathrm{Flank}}\right)$",
                   color=AX_TEXT, fontsize=13)
    cbar.ax.tick_params(color=AX_TEXT, labelcolor=AX_TEXT, labelsize=12)
    cbar.outline.set_visible(False)

    pos = cax.get_position(fig)
    new_h = pos.height * 0.45
    new_y = pos.y0 + (pos.height - new_h) * 0.55
    new_x = pos.x0 + pos.width * 0.12
    new_w = pos.width * 0.76
    cax.set_position([new_x, new_y, new_w, new_h])

    out = output_dir / f"fst_mf_quadrant_violins_total_{spec.slug}.pdf"
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    except Exception:
        pass
    try:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {out}")
    except Exception as e:
        logger.error(f"Failed to save figure: {e}")

    logger.info(f"MF plot built in {time.time() - start_time:.2f}s.")
    return fig


# ------------------------------------------------------------------------------
def main():
    overall_start = time.time()
    logger.info("--- Starting Hudson FST Flanking Regions Analysis ---")
    logger.info("Configured window specifications:")
    for spec in WINDOW_SPECS:
        logger.info("  • %s", spec.description)

    # Log font resolution (debug)
    try:
        font_path = fm.findfont("DejaVu Sans")
        logger.info(f"Using font: {font_path}")
    except Exception as e:
        logger.warning(f"Could not resolve 'DejaVu Sans': {e}")

    # Load inversion info once
    inv_file_path = Path(INVERSION_FILE)
    if not inv_file_path.is_file():
        logger.error(f"Inversion info file not found: {inv_file_path}")
        return
    try:
        inv_df = pd.read_csv(inv_file_path, sep="\t")
    except Exception as e:
        logger.error(f"Failed reading inversion file: {e}")
        return
    recurrent_regions, single_event_regions = map_regions_to_inversions(inv_df)

    fst_path = Path(FST_DATA_FILE)
    if not fst_path.is_file():
        logger.error(f"FST data file not found: {fst_path}")
        return

    successful_runs = 0
    for spec in WINDOW_SPECS:
        run_start = time.time()
        logger.info("\n%s", "=" * 95)
        logger.info(">>> Analyzing window specification: %s", spec.description)
        output_dir = ensure_output_dir(spec)

        fst_sequences = load_fst_data(fst_path, spec.total_length)
        if not fst_sequences:
            logger.error(
                "No valid sequences after filtering for %s. Skipping.",
                spec.description,
            )
            continue

        flanking_stats = calculate_flanking_stats(fst_sequences, spec)
        if not flanking_stats:
            logger.error(
                "No sequences after flanking stats (NaN/length issues) for %s. Skipping.",
                spec.description,
            )
            continue

        categories = categorize_sequences(
            flanking_stats, recurrent_regions, single_event_regions
        )

        filtered_flanking_stats: List[dict] = []
        for key in CATEGORY_KEYS:
            filtered_flanking_stats.extend(categories.get(key, []))
        logger.info(
            "Filtered out ambiguous/unknown sequences. Kept %d for final analysis.",
            len(filtered_flanking_stats),
        )

        pooled_df = pooled_group_estimates(categories, reps=2_000, seed=1337)
        pooled_csv = output_dir / f"fst_mf_pooled_group_estimates_{spec.slug}.csv"
        try:
            pooled_df.to_csv(pooled_csv, index=False)
            logger.info("Saved pooled across-inversions FST estimates to %s", pooled_csv)
        except Exception as e:
            logger.error("Failed to write pooled estimates: %s", e)

        if not pooled_df.empty:
            try:
                for group_name in ["Single-event", "Recurrent", "Overall"]:
                    for window in ("Middle", "Flank"):
                        row = pooled_df[
                            (pooled_df.group == group_name) & (pooled_df.window == window)
                        ]
                        if row.empty:
                            continue
                        entry = row.iloc[0]
                        logger.info(
                            "[Pooled %s | %s] FST=%s ± %s  (n_inv=%d, num_sum=%s, den_sum=%s)",
                            group_name,
                            window,
                            format_plain_no_e(entry.pooled_fst) or "NA",
                            format_plain_no_e(entry.bootstrap_se) or "NA",
                            int(entry.n_inversions_contributing),
                            format_plain_no_e(entry.numerator_sum) or "NA",
                            format_plain_no_e(entry.denominator_sum) or "NA",
                        )
            except Exception:
                logger.exception("Failed logging pooled summary")

        tests = perform_statistical_tests(categories, filtered_flanking_stats)
        diff_test = single_vs_recurrent_one_sided_test(categories)
        tests["Single_vs_Recurrent_FlankMinusMiddle"] = diff_test

        logger.info("\n=== Summary Statistics (%s) ===", spec.description)
        print_summary_statistics(filtered_flanking_stats, "Overall")
        for name, key in zip(CATEGORY_ORDER, CATEGORY_KEYS):
            print_summary_statistics(categories.get(key, []), name)

        logger.info("\n--- Permutation Test Results (%s) ---", spec.description)
        for name in [
            "Single-event",
            "Recurrent",
            "Overall",
        ]:
            tr = tests.get(name, {})
            logger.info(
                f"[{name}] n={tr.get('n_valid_pairs', 0)}  "
                f"perm_p={_format_p_value_for_annotation(tr.get('mean_p', np.nan))}  "
                f"shapiro_p={_format_p_value_for_annotation(tr.get('mean_normality_p', np.nan))}"
            )

        logger.info(
            "\n--- Single-event vs Recurrent Δ(Flank − Middle) Test (%s) ---",
            spec.description,
        )
        se_mean = diff_test.get("single_event_mean_diff", np.nan)
        rec_mean = diff_test.get("recurrent_mean_diff", np.nan)
        se_mean_str = format_plain_no_e(se_mean) or "NA"
        rec_mean_str = format_plain_no_e(rec_mean) or "NA"
        logger.info(
            "[Diff Summary] n_single=%d mean_single=%s  n_recurrent=%d mean_recurrent=%s",
            diff_test.get("n_single_event", 0),
            se_mean_str,
            diff_test.get("n_recurrent", 0),
            rec_mean_str,
        )
        mw_u = diff_test.get("mannwhitney_u", np.nan)
        mw_u_str = format_plain_no_e(mw_u) or "NA"
        logger.info(
            "[Mann-Whitney U] U=%s  p_one_sided=%s  alternative=%s",
            mw_u_str,
            _format_p_value_for_annotation(diff_test.get("p_one_sided", np.nan)),
            diff_test.get("alternative", "single_event>recurrent"),
        )

        tests_path = output_dir / f"fst_mf_permtest_results_{spec.slug}.csv"
        try:
            pd.DataFrame.from_dict(tests, orient="index").to_csv(
                tests_path, index_label="group"
            )
            logger.info(f"Saved test results to {tests_path}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

        fig = create_mf_quadrant_violins(categories, tests, output_dir, spec)

        logger.info("\n--- Analysis Summary (%s) ---", spec.description)
        logger.info(f"Input FST File: {FST_DATA_FILE}")
        logger.info(f"Input Inversion File: {INVERSION_FILE}")
        logger.info(
            f"Total sequences in final analysis: {len(filtered_flanking_stats)}"
        )
        logger.info("--- Finished %s in %.2fs ---", spec.description, time.time() - run_start)

        if fig:
            plt.close(fig)

        successful_runs += 1

    logger.info("\n%s", "=" * 95)
    logger.info(
        "Completed %d of %d window specifications in %.2fs.",
        successful_runs,
        len(WINDOW_SPECS),
        time.time() - overall_start,
    )

if __name__ == "__main__":
    main()