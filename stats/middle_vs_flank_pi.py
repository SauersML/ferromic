import os
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
from scipy.stats import shapiro

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, to_rgb, to_hex
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib import transforms as mtransforms
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
logger = logging.getLogger("pi_flanking_analysis_exact_mf_quadrants")

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

PI_DATA_FILE = "per_site_diversity_output.falsta"
INVERSION_FILE = "inv_info.tsv"
OUTPUT_BASE_DIR = Path("pi_analysis_results_exact_mf_quadrants")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Category Mappings & Order
# ------------------------------------------------------------------------------
CAT_MAPPING = {
    "Recurrent Inverted": "recurrent_inverted",
    "Recurrent Direct": "recurrent_direct",
    "Single-event Inverted": "single_event_inverted",
    "Single-event Direct": "single_event_direct",
}
REVERSE_CAT_MAPPING = {v: k for k, v in CAT_MAPPING.items()}

CATEGORY_ORDER = [
    "Single-event Inverted",
    "Single-event Direct",
    "Recurrent Inverted",
    "Recurrent Direct",
]
CATEGORY_KEYS = [
    "single_event_inverted",
    "single_event_direct",
    "recurrent_inverted",
    "recurrent_direct",
]

# ------------------------------------------------------------------------------
# Plotting Style & Colors
# ------------------------------------------------------------------------------
# Core colors for Direct/Inverted status
COLOR_DIRECT   = "#1f3b78"   # dark blue
COLOR_INVERTED = "#8c2d7e"   # reddish purple

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


def extract_coordinates_from_header(header: str) -> Optional[dict]:
    if "filtered_pi" not in header.lower():
        return None
    pattern = re.compile(
        r">.*?filtered_pi.*?_chr_?([\w\.\-]+)_start_(\d+)_end_(\d+)(?:_group_([01]))?",
        re.IGNORECASE,
    )
    match = pattern.search(header)
    if not match:
        logger.warning(f"Failed to extract coordinates from filtered_pi header: {header[:70]}...")
        return None

    chrom_part, start_str, end_str, group_str = match.groups()
    chrom = normalize_chromosome(chrom_part)
    start = int(start_str) if start_str is not None else None
    end = int(end_str) if end_str is not None else None
    group = int(group_str) if group_str is not None else None

    if chrom is None or start is None or end is None:
        logger.warning(f"Chromosome normalization or coordinate extraction failed for header: {header[:70]}...")
        return None
    if start >= end:
        logger.warning(f"Start >= End in header: {header[:70]}... ({start} >= {end})")
        return None

    return {"chrom": chrom, "start": start, "end": end, "group": group}


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


def parse_pi_data_line(line: str) -> Optional[np.ndarray]:
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


def load_pi_data(file_path: str | Path, min_total_length: int) -> List[dict]:
    logger.info(f"Loading pi data from {file_path}")
    logger.info(
        f"Applying filters: Header must contain 'filtered_pi', Sequence length ≥ {min_total_length:,} (total)"
    )
    start_time = time.time()

    pi_sequences: List[dict] = []
    sequences_processed = 0
    headers_read = 0
    skipped_short_total = 0
    skipped_not_filtered_pi = 0
    skipped_coord_error = 0
    skipped_data_error = 0
    skipped_missing_group = 0

    current_header: Optional[str] = None
    current_sequence_parts: List[str] = []
    is_current_header_valid = False

    try:
        with open(file_path, "r") as f:
            for _, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    headers_read += 1
                    if is_current_header_valid and current_header and current_sequence_parts:
                        sequences_processed += 1
                        full = "".join(current_sequence_parts)
                        pi_data = parse_pi_data_line(full)
                        if pi_data is not None:
                            length = len(pi_data)
                            if length >= min_total_length:
                                coords = extract_coordinates_from_header(current_header)
                                if coords:
                                    if coords.get("group") is not None:
                                        pi_sequences.append(
                                            {
                                                "header": current_header,
                                                "coords": coords,
                                                "data": pi_data,
                                                "length": length,
                                                "is_inverted": coords["group"] == 1,
                                            }
                                        )
                                    else:
                                        skipped_missing_group += 1
                                else:
                                    skipped_coord_error += 1
                            else:
                                skipped_short_total += 1
                        else:
                            skipped_data_error += 1

                    current_header = line
                    current_sequence_parts = []
                    is_current_header_valid = False

                    if "filtered_pi" in current_header.lower():
                        coords_check = extract_coordinates_from_header(current_header)
                        if coords_check:
                            is_current_header_valid = True
                        else:
                            skipped_coord_error += 1
                            current_header = None
                    else:
                        skipped_not_filtered_pi += 1
                        current_header = None

                elif is_current_header_valid and current_header:
                    current_sequence_parts.append(line)

            # process trailing seq
            if is_current_header_valid and current_header and current_sequence_parts:
                sequences_processed += 1
                full = "".join(current_sequence_parts)
                pi_data = parse_pi_data_line(full)
                if pi_data is not None:
                    length = len(pi_data)
                    if length >= min_total_length:
                        coords = extract_coordinates_from_header(current_header)
                        if coords:
                            if coords.get("group") is not None:
                                pi_sequences.append(
                                    {
                                        "header": current_header,
                                        "coords": coords,
                                        "data": pi_data,
                                        "length": length,
                                        "is_inverted": coords["group"] == 1,
                                    }
                                )
                            else:
                                skipped_missing_group += 1
                        else:
                            skipped_coord_error += 1
                    else:
                        skipped_short_total += 1
                else:
                    skipped_data_error += 1

    except FileNotFoundError:
        logger.error(f"Fatal Error: Pi data file not found at {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}", exc_info=True)
        return []

    elapsed_time = time.time() - start_time
    logger.info(
        f"Read {headers_read} headers, processed {sequences_processed} candidate 'filtered_pi' sequences in {elapsed_time:.2f}s."
    )
    logger.info(
        f"Loaded {len(pi_sequences)} valid sequences (filtered_pi, length ≥ {min_total_length:,}, valid coords, group)."
    )
    logger.info("Skipped counts: "
                f"Not 'filtered_pi'={skipped_not_filtered_pi}, "
                f"Too Short (total length)={skipped_short_total}, "
                f"Coord Err={skipped_coord_error}, "
                f"Missing Group={skipped_missing_group}, "
                f"Data Parse Err={skipped_data_error}")
    return pi_sequences


def calculate_flanking_stats(pi_sequences: List[dict], spec: WindowSpec) -> List[dict]:
    """
    EXACT windows defined by ``spec``:
      - Left flank:  first  ``spec.flank_size`` bases
      - Middle:      centered window of ``spec.middle_size`` bases
      - Right flank: last   ``spec.flank_size`` bases
    Total length must be ≥ ``spec.total_length``. Windows never overlap.
    """
    logger.info(
        "Calculating flanking and middle statistics for %d sequences (total=%s, flank=%s, middle=%s)...",
        len(pi_sequences),
        spec.total_label,
        spec.flank_label,
        spec.middle_label,
    )
    start_time = time.time()
    results: List[dict] = []
    skipped_too_short_total = 0
    skipped_nan_middle = 0
    skipped_nan_flanks = 0

    for seq in pi_sequences:
        data = seq["data"]
        L = len(data)
        if L < spec.total_length:
            skipped_too_short_total += 1
            continue

        # Exact windows
        beginning_flank = data[:spec.flank_size]
        ending_flank = data[-spec.flank_size:]

        # Centered middle window
        start_middle = (L - spec.middle_size) // 2
        end_middle = start_middle + spec.middle_size
        middle_region = data[start_middle:end_middle]

        if not (
            spec.flank_size <= start_middle
            and end_middle <= (L - spec.flank_size)
        ):
            logger.warning(
                f"Middle window overlaps flanks (len={L}, start_mid={start_middle}, end_mid={end_middle}). Skipping."
            )
            continue

        stats = {
            "header": seq["header"],
            "coords": seq["coords"],
            "is_inverted": seq["is_inverted"],
            "length": seq["length"],
            "window_total_bp": spec.total_length,
            "window_flank_bp": spec.flank_size,
            "window_middle_bp": spec.middle_size,
            "beginning_mean": np.nanmean(beginning_flank),
            "ending_mean": np.nanmean(ending_flank),
            "middle_mean": np.nanmean(middle_region),
            "beginning_median": np.nanmedian(beginning_flank),
            "ending_median": np.nanmedian(ending_flank),
            "middle_median": np.nanmedian(middle_region),
        }

        stats["flanking_mean"] = np.nanmean([stats["beginning_mean"], stats["ending_mean"]])
        stats["flanking_median"] = np.nanmean([stats["beginning_median"], stats["ending_median"]])

        if np.isnan(stats["middle_mean"]):
            skipped_nan_middle += 1
            continue
        if np.isnan(stats["flanking_mean"]):
            skipped_nan_flanks += 1
            continue

        results.append(stats)

    elapsed_time = time.time() - start_time
    logger.info(f"Calculated stats for {len(results)} sequences in {elapsed_time:.2f}s.")
    if skipped_too_short_total:
        logger.warning(
            f"Skipped {skipped_too_short_total} sequences too short for exact windows (need ≥ {spec.total_length:,})."
        )
    if skipped_nan_middle:
        logger.warning(f"Skipped {skipped_nan_middle} sequences (NaN middle mean).")
    if skipped_nan_flanks:
        logger.warning(f"Skipped {skipped_nan_flanks} sequences (both flanks NaN).")
    return results


def categorize_sequences(flanking_stats: List[dict], recurrent_regions: dict, single_event_regions: dict) -> dict:
    logger.info("Categorizing sequences based on overlap with inversion regions...")
    categories = {
        "single_event_inverted": [],
        "single_event_direct": [],
        "recurrent_inverted": [],
        "recurrent_direct": [],
    }

    for seq_stats in flanking_stats:
        coords = seq_stats.get("coords")
        if not coords or "is_inverted" not in seq_stats:
            seq_stats["inv_class"] = "unknown"
            continue

        inv_type = determine_inversion_type(coords, recurrent_regions, single_event_regions)
        seq_stats["inv_class"] = inv_type

        if inv_type == "single_event":
            key = "single_event_inverted" if seq_stats["is_inverted"] else "single_event_direct"
            categories[key].append(seq_stats)
        elif inv_type == "recurrent":
            key = "recurrent_inverted" if seq_stats["is_inverted"] else "recurrent_direct"
            categories[key].append(seq_stats)
        # ambiguous/unknown not included in these four bins

    for disp, key in zip(CATEGORY_ORDER, CATEGORY_KEYS):
        logger.info(f"  {disp}: {len(categories[key])}")
    return categories


def perform_statistical_tests(categories: dict, all_sequences_stats: List[dict]) -> dict:
    logger.info("Performing paired permutation tests (Middle vs Flanking; mean diffs)...")
    test_results: Dict[str, dict] = {}

    # Build map with display names
    display_map = {
        "Single-event Inverted": categories["single_event_inverted"],
        "Single-event Direct": categories["single_event_direct"],
        "Recurrent Inverted": categories["recurrent_inverted"],
        "Recurrent Direct": categories["recurrent_direct"],
        "Overall": all_sequences_stats,
    }

    for name, seqs in display_map.items():
        test_results[name] = {"mean_p": np.nan, "mean_normality_p": np.nan, "n_valid_pairs": 0}
        if len(seqs) < 2:
            continue
        f_means = np.array([s["flanking_mean"] for s in seqs], dtype=float)
        m_means = np.array([s["middle_mean"] for s in seqs], dtype=float)
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

    flank_means = [s['flanking_mean'] for s in sequences]
    middle_means = [s['middle_mean'] for s in sequences]

    n = len(sequences)

    logger.info(f"\n--- Summary Statistics for {group_name} (n={n}) ---")
    logger.info(f"  Flank Region (π means):  Mean={np.nanmean(flank_means):.8f}, Median={np.nanmedian(flank_means):.8f}")
    logger.info(f"  Middle Region (π means): Mean={np.nanmean(middle_means):.8f}, Median={np.nanmedian(middle_means):.8f}")

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

def _paired_lines_and_points(ax, rows, cmap, norm, category_key: str, flank_point_color, middle_point_color):
    """Per sequence: draw two points (Flank, Middle) and the connecting line colored by log2(M/MF)."""
    EPS = 1e-12
    for r in rows:
        f = r["flanking_mean"]
        m = r["middle_mean"]
        if not (np.isfinite(f) and np.isfinite(m)):
            continue
        rng = _deterministic_rng(r.get("header", "") + "|" + category_key)
        j = _inward_jitter(rng)

        x_f = POS_X["Flank"]  + j    # jitter rightwards
        x_m = POS_X["Middle"] - j    # jitter leftwards

        log2fc = math.log2((m + EPS) / (f + EPS))
        c = cmap(norm(log2fc))

        ax.plot([x_f, x_m], [f, m], color=c, linewidth=LINE_WIDTH, alpha=0.98, zorder=3, solid_capstyle="round")
        # Use specific point colors derived from the base color
        ax.scatter([x_f], [f], s=POINT_SIZE, c=flank_point_color, edgecolors="black", linewidths=0.5, alpha=ALPHA_POINTS, zorder=6)
        ax.scatter([x_m], [m], s=POINT_SIZE, c=middle_point_color,   edgecolors="black", linewidths=0.5, alpha=ALPHA_POINTS, zorder=6)


def _prepare_category_bins(categories: dict) -> Dict[str, List[dict]]:
    """Ensure the bins are present for the four requested panels."""
    bins = {
        "single_event_inverted": categories.get("single_event_inverted", []),
        "single_event_direct":   categories.get("single_event_direct", []),
        "recurrent_inverted":    categories.get("recurrent_inverted", []),
        "recurrent_direct":      categories.get("recurrent_direct", []),
    }
    return bins

def _collect_all_pairs_for_scale(categories: dict) -> np.ndarray:
    """Collect log2 ratios across all bins for a stable, shared color scale."""
    EPS = 1e-12
    vals = []
    for key in ["single_event_inverted", "single_event_direct", "recurrent_inverted", "recurrent_direct"]:
        for s in categories.get(key, []):
            f = s.get("flanking_mean", np.nan)
            m = s.get("middle_mean", np.nan)
            if np.isfinite(f) and np.isfinite(m):
                vals.append(math.log2((m + EPS) / (f + EPS)))
    return np.asarray(vals, dtype=float)

def _draw_right_key(rax):
    """Right-side legend showing region colors and patterns compositionally."""
    rax.set_xlim(0, 1)
    rax.set_ylim(0, 1)
    rax.axis("off")

    # Define legend entries for compositional elements
    # Using 'none' for edgecolor/hatch indicates it's not applicable for that specific visual aspect of the legend
    # For hatches, we want to show the hatch itself clearly, so facecolor can be white for better contrast
    legend_elements = [
        # Inversion Type (Direct/Inverted)
        {"label": "Direct Orientation",   "facecolor": COLOR_DIRECT,   "hatch": None,   "edgecolor": "none", "alpha": ALPHA_VIOLIN, "y": 0.88},
        {"label": "Inverted Orientation", "facecolor": COLOR_INVERTED, "hatch": None,   "edgecolor": "none", "alpha": ALPHA_VIOLIN, "y": 0.76},
        # Recurrence Type (Single-event/Recurrent)
        {"label": "Single-event",         "facecolor": "#FFFFFF",      "hatch": OVERLAY_SINGLE_HATCH, "edgecolor": OVERLAY_SINGLE_EDGE, "alpha": 1.0, "y": 0.54},
        {"label": "Recurrent",            "facecolor": "#FFFFFF",      "hatch": OVERLAY_RECUR_HATCH,  "edgecolor": OVERLAY_RECUR_EDGE,  "alpha": 1.0, "y": 0.42},
        # Region Type (Flank/Middle) - using an arbitrary base color (e.g., COLOR_DIRECT) to show lightness
        {"label": "Flanking Region",      "facecolor": adjust_lightness(COLOR_DIRECT, FLANK_LIGHTNESS_FACTOR),  "hatch": None, "edgecolor": "none", "alpha": ALPHA_VIOLIN, "y": 0.20},
        {"label": "Middle Region",        "facecolor": adjust_lightness(COLOR_DIRECT, MIDDLE_LIGHTNESS_FACTOR), "hatch": None, "edgecolor": "none", "alpha": ALPHA_VIOLIN, "y": 0.08},
    ]

    S = 0.08 # Smaller square size for compositional key
    X0 = 0.08 # X position for squares

    rax.text(X0, 0.96, "Inversion Status:", ha="left", va="bottom", fontsize=12, color=AX_TEXT, fontweight="bold")
    rax.text(X0, 0.62, "Recurrence Type:", ha="left", va="bottom", fontsize=12, color=AX_TEXT, fontweight="bold")
    rax.text(X0, 0.28, "Region Type:", ha="left", va="bottom", fontsize=12, color=AX_TEXT, fontweight="bold")


    for entry in legend_elements:
        # Pass hatch=None if not applicable to avoid matplotlib warnings/errors
        hatch_to_pass = entry["hatch"] if entry["hatch"] else None
        sq = Rectangle((X0, entry["y"]), S, S, transform=rax.transAxes,
                       facecolor=entry["facecolor"], edgecolor=entry["edgecolor"],
                       hatch=hatch_to_pass, linewidth=0.0, alpha=entry["alpha"], clip_on=False)
        rax.add_patch(sq)
        rax.text(X0 + S + 0.02, entry["y"] + S/2, entry["label"], transform=rax.transAxes,
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
    """
    Build a 2x2 grid of subplots:
      [Single-event Inverted]  [Single-event Direct]
      [Recurrent Inverted]     [Recurrent Direct]
    Each subplot: two violins (Flank, Middle), slim boxplots, paired lines/points, and
    an annotation box with n, permutation p-value, and Shapiro–Wilk p-value.
    Right column: legend (top) + thin horizontal colorbar (bottom).
    """
    logger.info(
        "Creating Middle vs Flank quadrant violins for %s...",
        spec.description,
    )
    start_time = time.time()

    bins = _prepare_category_bins(categories)

    # Shared y-limits across all panes
    all_vals = []
    for key in bins:
        all_vals.extend([s.get("flanking_mean", np.nan) for s in bins[key]])
        all_vals.extend([s.get("middle_mean",  np.nan) for s in bins[key]])
    all_vals = np.asarray(all_vals, dtype=float)
    finite = all_vals[np.isfinite(all_vals)]
    y_min, y_max = (0.0, 1.0) if finite.size == 0 else (np.nanmin(finite), np.nanmax(finite))
    pad = (y_max - y_min) * 0.06 if y_max > y_min else 0.1
    y_lo, y_hi = max(0.0, y_min - pad), y_max + pad

    # Shared color scale from all log2 ratios
    all_log2 = _collect_all_pairs_for_scale(bins)
    finite_log2 = all_log2[np.isfinite(all_log2)]
    if finite_log2.size == 0:
        v = 1.0
    else:
        v = float(np.nanpercentile(np.abs(finite_log2), 98))
        if not np.isfinite(v) or v <= 0:
            v = float(np.nanmax(np.abs(finite_log2))) if finite_log2.size else 1.0
            if not np.isfinite(v) or v <= 0:
                v = 1.0
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    cmap = plt.get_cmap("coolwarm")
    
    # Layout: 2 rows x 3 columns (last col used for legend & colorbar)
    fig = plt.figure(figsize=(12, 8.0))
    fig.suptitle(
        f"Middle vs Flank π — {spec.total_label} total (flanks {spec.flank_label} each, middle {spec.middle_label})",
        fontsize=16,
        color=AX_TEXT,
        y=0.98,
    )
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.72],
        height_ratios=[1.0, 1.0],
        wspace=0.22, hspace=0.28
    )
    
    ax_map = {
        "single_event_inverted": fig.add_subplot(gs[0, 0]),
        "single_event_direct":   fig.add_subplot(gs[0, 1]),
        "recurrent_inverted":    fig.add_subplot(gs[1, 0]),
        "recurrent_direct":      fig.add_subplot(gs[1, 1]),
    }
    
    rax = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[1, 2])


    # Draw panels
    title_map = {
        "single_event_inverted": "Single-event Inverted",
        "single_event_direct":   "Single-event Direct",
        "recurrent_inverted":    "Recurrent Inverted",
        "recurrent_direct":      "Recurrent Direct",
    }

    for key in ["single_event_inverted", "single_event_direct", "recurrent_inverted", "recurrent_direct"]:
        ax = ax_map[key]
        rows = bins[key]

        if "direct" in key:
            base_color = COLOR_DIRECT
            inversion_status = "Direct" # For potential debugging/logging
        else: # "inverted" in key
            base_color = COLOR_INVERTED
            inversion_status = "Inverted"

        if "single_event" in key:
            hatch_pattern = OVERLAY_SINGLE_HATCH
            hatch_edge_color = OVERLAY_SINGLE_EDGE
            recurrence_type = "Single-event" # For potential debugging/logging
        else: # "recurrent" in key
            hatch_pattern = OVERLAY_RECUR_HATCH
            hatch_edge_color = OVERLAY_RECUR_EDGE
            recurrence_type = "Recurrent"

        flank_color = adjust_lightness(base_color, FLANK_LIGHTNESS_FACTOR)
        middle_color = adjust_lightness(base_color, MIDDLE_LIGHTNESS_FACTOR)

        # Violin data in order [Flank, Middle]
        v_flank = [s.get("flanking_mean", np.nan) for s in rows]
        v_mid   = [s.get("middle_mean",  np.nan) for s in rows]
        vals = [np.asarray(v_flank, dtype=float), np.asarray(v_mid, dtype=float)]
        pos  = [POS_X["Flank"], POS_X["Middle"]]
        
        ax.set_title(title_map[key], fontsize=14, color=AX_TEXT, pad=20)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(y_lo, y_hi)        
        _draw_two_violins(ax, vals, pos, flank_color, middle_color, hatch_pattern, hatch_edge_color)
        
        _overlay_boxplots(ax, vals, pos)
        
        _paired_lines_and_points(ax, rows, cmap=cmap, norm=norm, category_key=key, 
                                 flank_point_color=flank_color, middle_point_color=middle_color)


        ax.set_xticks([POS_X["Flank"], POS_X["Middle"]])
        ax.set_xticklabels(["Flank", "Middle"], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: format_plain_no_e(v, max_decimals=8)))
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

    ax_map["single_event_inverted"].set_ylabel("Mean Nucleotide Diversity (π)", fontsize=16, color=AX_TEXT)
    ax_map["recurrent_inverted"].set_ylabel("Mean Nucleotide Diversity (π)", fontsize=16, color=AX_TEXT)

    _draw_right_key(rax)

    for spine in cax.spines.values():
        spine.set_visible(False)
    cax.set_xticks([]); cax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$\log_{2}\!\left(\pi_{\mathrm{Middle}} \,/\, \pi_{\mathrm{Flank}}\right)$",
                   color=AX_TEXT, fontsize=13)
    cbar.ax.tick_params(color=AX_TEXT, labelcolor=AX_TEXT, labelsize=12)
    cbar.outline.set_visible(False)

    pos = cax.get_position(fig)
    new_h = pos.height * 0.45
    new_y = pos.y0 + (pos.height - new_h) * 0.50
    new_x = pos.x0 + pos.width * 0.08
    new_w = pos.width * 0.84
    cax.set_position([new_x, new_y, new_w, new_h])

    out = output_dir / f"pi_mf_quadrant_violins_total_{spec.slug}.pdf"
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    except Exception:
        pass
    try:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {out}")
    except Exception as e:
        logger.error(f"Failed to save figure: {e}")

    logger.info(f"MF quadrant plot built in {time.time() - start_time:.2f}s.")
    return fig


# ------------------------------------------------------------------------------
def main():
    overall_start = time.time()
    logger.info("--- Starting Pi Flanking Regions Analysis ---")
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

    pi_path = Path(PI_DATA_FILE)
    if not pi_path.is_file():
        logger.error(f"Pi data file not found: {pi_path}")
        return

    successful_runs = 0
    for spec in WINDOW_SPECS:
        run_start = time.time()
        logger.info("\n%s", "=" * 95)
        logger.info(">>> Analyzing window specification: %s", spec.description)
        output_dir = ensure_output_dir(spec)

        pi_sequences = load_pi_data(pi_path, spec.total_length)
        if not pi_sequences:
            logger.error(
                "No valid sequences after filtering for %s. Skipping.",
                spec.description,
            )
            continue

        flanking_stats = calculate_flanking_stats(pi_sequences, spec)
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

        tests = perform_statistical_tests(categories, filtered_flanking_stats)

        logger.info("\n=== Summary Statistics (%s) ===", spec.description)
        print_summary_statistics(filtered_flanking_stats, "Overall")
        for name, key in zip(CATEGORY_ORDER, CATEGORY_KEYS):
            print_summary_statistics(categories.get(key, []), name)

        logger.info("\n--- Permutation Test Results (%s) ---", spec.description)
        for name in [
            "Single-event Inverted",
            "Single-event Direct",
            "Recurrent Inverted",
            "Recurrent Direct",
            "Overall",
        ]:
            tr = tests.get(name, {})
            logger.info(
                f"[{name}] n={tr.get('n_valid_pairs', 0)}  "
                f"perm_p={_format_p_value_for_annotation(tr.get('mean_p', np.nan))}  "
                f"shapiro_p={_format_p_value_for_annotation(tr.get('mean_normality_p', np.nan))}"
            )

        tests_path = output_dir / f"mf_permtest_results_{spec.slug}.csv"
        try:
            pd.DataFrame.from_dict(tests, orient="index").to_csv(
                tests_path, index_label="group"
            )
            logger.info(f"Saved test results to {tests_path}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

        fig = create_mf_quadrant_violins(categories, tests, output_dir, spec)

        logger.info("\n--- Analysis Summary (%s) ---", spec.description)
        logger.info(f"Input Pi File: {PI_DATA_FILE}")
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
