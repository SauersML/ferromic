import os
import re
import sys
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import shapiro

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm

import seaborn as sns

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pi_flanking_analysis_filtered_pi_length")

# ------------------------------------------------------------------------------
# Matplotlib font/embedding configuration to avoid missing-glyph boxes
# ------------------------------------------------------------------------------
# We need a font that actually includes superscripts and the multiplication sign.
# DejaVu Sans ships with Matplotlib and has the needed glyphs.
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42,      # Embed TrueType glyphs in PDF (Type 42)
    "ps.fonttype": 42,
    "text.usetex": False,    # DO NOT use TeX; we draw Unicode directly
    "axes.unicode_minus": True,
})

# Optional: if you want to force a specific local DejaVuSans.ttf file, you can load it:
# from matplotlib import font_manager as fm
# ttf_path = Path(mpl.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
# prop = fm.FontProperties(fname=str(ttf_path))
# mpl.rcParams['font.family'] = prop.get_name()

# ------------------------------------------------------------------------------
# Constants & File Paths
# ------------------------------------------------------------------------------
MIN_LENGTH = 150_000          # Min sequence length for analysis
FLANK_SIZE = 50_000           # Flanking size at each end
PERMUTATIONS = 10_000         # Permutations for significance testing

PI_DATA_FILE = "per_site_diversity_output.falsta"
INVERSION_FILE = "inv_info.tsv"
OUTPUT_DIR = Path("pi_analysis_results_filtered_pi_length")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If True and DejaVu Sans isn't detected, fall back to plain 'e' notation
# instead of Unicode superscripts (prevents "missing glyph" boxes in hostile envs).
USE_SAFE_FALLBACK = True

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
    "Recurrent Inverted",
    "Recurrent Direct",
    "Single-event Inverted",
    "Single-event Direct",
]
CATEGORY_ORDER_WITH_OVERALL = CATEGORY_ORDER + ["Overall"]

# ------------------------------------------------------------------------------
# Plotting Style & Colors
# ------------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-ticks")

# Violin fill colors
COLOR_PALETTE = plt.cm.tab10.colors
FLANKING_COLOR = COLOR_PALETTE[0]  # Blue
MIDDLE_COLOR  = COLOR_PALETTE[1]   # Orange

# Base point colors (by sequence type)
INVERTED_BASE_COLOR = "#A23B72"     # Reddish-purple
DIRECT_BASE_COLOR   = "#0D3B66"     # Dark blue

# Overlay colors
RECURRENT_OVERLAY_COLOR     = "#4A4A4A"  # Dark gray “X”
RECURRENT_OVERLAY_ALPHA     = 0.7        # Semi-transparent X
SINGLE_EVENT_OVERLAY_COLOR  = "#CFCFCF"  # Light gray dot

# Transparency & z-ordering
SCATTER_ALPHA = 0.35                 # base points
SCATTER_SIZE_PT2 = 18.0              # scatter size in points^2
LINE_ALPHA = 1.00                    # connecting lines fully opaque
LINE_WIDTH = 0.9
VIOLIN_LINEWIDTH = 1.0
VIOLIN_ALPHA = 0.25                  # violins very transparent
MEDIAN_LINE_COLOR = "k"
MEDIAN_LINE_WIDTH = 1.5
DEFAULT_LINE_COLOR = "grey"
PLOT_COLORMAP = cm.coolwarm

# Deterministic jitter
JITTER_WIDTH = 0.15                      # +/- horizontal jitter
X_POS = {"Flanking": 0.0, "Middle": 1.0} # category x positions

# ------------------------------------------------------------------------------
# Helpers: chromosome normalization & parsing
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


def load_pi_data(file_path: str | Path) -> List[dict]:
    logger.info(f"Loading pi data from {file_path}")
    logger.info(f"Applying filters: Header must contain 'filtered_pi', Sequence length >= {MIN_LENGTH}")
    start_time = time.time()

    pi_sequences: List[dict] = []
    sequences_processed = 0
    headers_read = 0
    skipped_short = 0
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
                            if length >= MIN_LENGTH:
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
                                        logger.warning(f"Skipping: Missing group in header '{current_header[:70]}...'")
                                        skipped_missing_group += 1
                                else:
                                    logger.error(f"Header marked valid but coord extraction failed later: {current_header[:70]}...")
                                    skipped_coord_error += 1
                            else:
                                skipped_short += 1
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
                    if length >= MIN_LENGTH:
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
                        skipped_short += 1
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
        f"Loaded {len(pi_sequences)} valid sequences (filtered_pi, length ≥ {MIN_LENGTH}, valid coords, group)."
    )
    logger.info("Skipped counts: "
                f"Not 'filtered_pi'={skipped_not_filtered_pi}, "
                f"Too Short={skipped_short}, "
                f"Coord Err={skipped_coord_error}, "
                f"Missing Group={skipped_missing_group}, "
                f"Data Parse Err={skipped_data_error}")
    return pi_sequences


def calculate_flanking_stats(pi_sequences: List[dict]) -> List[dict]:
    logger.info(f"Calculating flanking and middle statistics for {len(pi_sequences)} sequences...")
    start_time = time.time()
    results: List[dict] = []
    skipped_too_short_for_flanks = 0
    skipped_nan_middle = 0
    skipped_nan_flanks = 0

    for i, seq in enumerate(pi_sequences):
        data = seq["data"]
        min_req_len = 2 * FLANK_SIZE + 1
        if len(data) < min_req_len:
            skipped_too_short_for_flanks += 1
            continue

        beginning_flank = data[:FLANK_SIZE]
        ending_flank = data[-FLANK_SIZE:]
        middle_region = data[FLANK_SIZE:-FLANK_SIZE]

        stats = {
            "header": seq["header"],
            "coords": seq["coords"],
            "is_inverted": seq["is_inverted"],
            "length": seq["length"],
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
    if skipped_too_short_for_flanks:
        logger.warning(f"Skipped {skipped_too_short_for_flanks} sequences too short for flank analysis.")
    if skipped_nan_middle:
        logger.warning(f"Skipped {skipped_nan_middle} sequences (NaN middle mean).")
    if skipped_nan_flanks:
        logger.warning(f"Skipped {skipped_nan_flanks} sequences (both flanks NaN).")
    return results


def categorize_sequences(flanking_stats: List[dict], recurrent_regions: dict, single_event_regions: dict) -> dict:
    logger.info("Categorizing sequences based on overlap with inversion regions...")
    categories = {CAT_MAPPING[name]: [] for name in CATEGORY_ORDER}

    for seq_stats in flanking_stats:
        coords = seq_stats.get("coords")
        if not coords or "is_inverted" not in seq_stats:
            seq_stats["inv_class"] = "unknown"
            continue

        inv_type = determine_inversion_type(coords, recurrent_regions, single_event_regions)
        seq_stats["inv_class"] = inv_type
        key = None
        if inv_type == "recurrent":
            key = "recurrent_inverted" if seq_stats["is_inverted"] else "recurrent_direct"
        elif inv_type == "single_event":
            key = "single_event_inverted" if seq_stats["is_inverted"] else "single_event_direct"
        # ambiguous/unknown are not put in categories used for tests, but inv_class is kept for overlays

        if key:
            categories[key].append(seq_stats)

    for disp in CATEGORY_ORDER:
        logger.info(f"  {disp}: {len(categories[CAT_MAPPING[disp]])}")
    return categories


def perform_statistical_tests(categories: dict, all_sequences_stats: List[dict]) -> dict:
    logger.info("Performing paired permutation tests (Middle vs Flanking; mean diffs)...")
    test_results: Dict[str, dict] = {}

    category_data_map = {REVERSE_CAT_MAPPING[k]: v for k, v in categories.items()}
    category_data_map["Overall"] = all_sequences_stats

    for name in CATEGORY_ORDER_WITH_OVERALL:
        seqs = category_data_map.get(name, [])
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

# ------------------------------------------------------------------------------
# Unicode superscripts (no TeX), with optional safe fallback
# ------------------------------------------------------------------------------

_SUPERSCRIPTS = str.maketrans({
    "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
})

def _active_font_is_dejavu() -> bool:
    try:
        path = fm.findfont("DejaVu Sans")
        return "dejavu" in os.path.basename(path).lower()
    except Exception:
        return False

def _superscript_int(n: int) -> str:
    return str(n).translate(_SUPERSCRIPTS).replace("-", "⁻")

def _trim_trailing_zeros(s: str) -> str:
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def sci_notation_superscript(x: float, sig: int = 3, allow_fallback: bool = True) -> str:
    """Return '1.23 × 10⁻⁴' using Unicode superscripts.
       If DejaVu Sans isn’t active and allow_fallback, return plain '1.23e-4'."""
    if not np.isfinite(x):
        return "N/A"
    if x == 0:
        return "0"
    if (not _active_font_is_dejavu()) and USE_SAFE_FALLBACK and allow_fallback:
        return f"{x:.{sig}e}"  # safe ASCII fallback

    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10 ** exp)
    mant_str = _trim_trailing_zeros(f"{mant:.{sig}g}")
    try:
        if np.isclose(float(mant_str), round(float(mant_str)), atol=10**-(sig+1)):
            mant_str = str(int(round(float(mant_str))))
    except Exception:
        pass
    return f"{mant_str} × 10{_superscript_int(exp)}"

def format_number_superscript(x: float, sig: int = 3) -> str:
    """Plain for mid-range (trim zeros), scientific with Unicode superscripts for small/large."""
    if not np.isfinite(x):
        return "N/A"
    ax = abs(x)
    if ax == 0:
        return "0"
    if 1e-3 <= ax < 1e4:
        s = f"{x:.{sig}g}"
        try:
            if np.isclose(float(s), round(float(s)), atol=10**-(sig+1)):
                s = str(int(round(float(s))))
        except Exception:
            pass
        return s
    return sci_notation_superscript(x, sig=sig, allow_fallback=True)

def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "p = N/A"
    # show scientific if very small
    if p_value < 0.001:
        return f"p = {sci_notation_superscript(p_value, sig=2, allow_fallback=True)}"
    return f"p = {format_number_superscript(p_value, sig=3)}"

def _tick_formatter(val, pos):
    return format_number_superscript(val, sig=3)

# ------------------------------------------------------------------------------
# Plotting with overlays
# ------------------------------------------------------------------------------
def _deterministic_jitter(pair_id: str, region_type: str, base_x: float, width: float = JITTER_WIDTH) -> float:
    """Deterministic jitter per (pair_id, region_type) so lines and points meet exactly."""
    key = f"{pair_id}|{region_type}"
    seed = int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:4], "little")
    rng = np.random.RandomState(seed)
    return base_x + rng.uniform(-width, width)

def create_paired_violin_with_overlays(all_sequences_stats: List[Dict], test_results: Dict) -> Optional[plt.Figure]:
    stat_type = "mean"
    logger.info(f"Creating paired violin with overlays for {stat_type} π...")
    start_time = time.time()

    flanking_field = f"flanking_{stat_type}"
    middle_field  = f"middle_{stat_type}"

    plot_data = []
    paired_list = []

    for i, s in enumerate(all_sequences_stats):
        f_val = s.get(flanking_field)
        m_val = s.get(middle_field)
        is_inverted = s.get("is_inverted")
        inv_class = s.get("inv_class", "unknown")
        pair_id = f"pair_{i}"
        if is_inverted is None:
            is_inverted = False
        if pd.notna(f_val) and pd.notna(m_val):
            plot_data.append({"pair_id": pair_id, "region_type": "Flanking", "pi_value": f_val,
                              "is_inverted": is_inverted, "inv_class": inv_class})
            plot_data.append({"pair_id": pair_id, "region_type": "Middle", "pi_value": m_val,
                              "is_inverted": is_inverted, "inv_class": inv_class})
            paired_list.append({"pair_id": pair_id, "Flanking": f_val, "Middle": m_val})

    if len(paired_list) < 2:
        logger.warning(f"Insufficient valid pairs ({len(paired_list)}) with non-NaN values. Skipping plot.")
        return None

    df_long = pd.DataFrame(plot_data)
    df_paired = pd.DataFrame(paired_list)

    # L2FC for line colors
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df_paired["Middle"] / df_paired["Flanking"]
        df_paired["L2FC"] = np.log2(ratio)
        df_paired.loc[np.isneginf(df_paired["L2FC"]) | pd.isna(df_paired["L2FC"]), "L2FC"] = np.nan

    l2fc_finite = df_paired["L2FC"].replace([np.inf, -np.inf], np.nan).dropna()
    scalar_mappable = None
    if not l2fc_finite.empty:
        q_low = np.nanpercentile(l2fc_finite, 2)
        q_high = np.nanpercentile(l2fc_finite, 98)
        max_abs = max(abs(q_low), abs(q_high))
        min_range = 0.1
        vmin, vmax = (-min_range / 2, min_range / 2) if (max_abs < (min_range / 2) or np.isclose(max_abs, 0)) else (-max_abs, max_abs)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=PLOT_COLORMAP)
        logger.info(f"L2FC normalization range: [{vmin:.3f}, {vmax:.3f}]")
    else:
        logger.warning("No finite L2FC values; lines will use default color and no colorbar will be drawn.")

    # Tall/skinnier fig
    fig, ax = plt.subplots(figsize=(6.0, 10.0))

    # --- VIOLINS (very transparent) ---
    region_order = ["Flanking", "Middle"]
    pre_cols = len(ax.collections)
    sns.violinplot(
        data=df_long,
        x="region_type",
        y="pi_value",
        order=region_order,
        palette=[FLANKING_COLOR, MIDDLE_COLOR],
        inner=None,
        linewidth=VIOLIN_LINEWIDTH,
        width=0.8,
        cut=0,
        scale="width",
        ax=ax,
        zorder=5,
    )
    for col in ax.collections[pre_cols:]:
        if isinstance(col, PolyCollection):
            col.set_alpha(VIOLIN_ALPHA)
            col.set_zorder(5)

    # Deterministic jitter positions (used for BOTH lines and points)
    df_long["x_jit"] = [
        _deterministic_jitter(r["pair_id"], r["region_type"], X_POS[r["region_type"]], JITTER_WIDTH)
        for _, r in df_long.iterrows()
    ]

    # --- CONNECTING LINES (opaque; above violins, under points) ---
    for _, row in df_paired.iterrows():
        pid = row["pair_id"]
        x_flank_jit  = _deterministic_jitter(pid, "Flanking", X_POS["Flanking"], JITTER_WIDTH)
        x_middle_jit = _deterministic_jitter(pid, "Middle",  X_POS["Middle"],  JITTER_WIDTH)
        y_flank, y_mid = row["Flanking"], row["Middle"]
        l2fc_val = row["L2FC"]
        line_color = scalar_mappable.to_rgba(l2fc_val) if (scalar_mappable is not None and pd.notna(l2fc_val) and np.isfinite(l2fc_val)) else DEFAULT_LINE_COLOR
        ax.plot([x_flank_jit, x_middle_jit], [y_flank, y_mid],
                color=line_color, alpha=LINE_ALPHA, lw=LINE_WIDTH, zorder=9)

    # --- BASE POINTS (above lines) ---
    facecolors = np.where(df_long["is_inverted"].values, INVERTED_BASE_COLOR, DIRECT_BASE_COLOR)
    ax.scatter(
        df_long["x_jit"].values,
        df_long["pi_value"].values,
        s=SCATTER_SIZE_PT2,
        c=facecolors,
        alpha=SCATTER_ALPHA,
        edgecolors="white",
        linewidths=0.2,
        zorder=10,
    )

    # --- OVERLAYS (on top of points) ---
    df_recur  = df_long[df_long["inv_class"] == "recurrent"]
    df_single = df_long[df_long["inv_class"] == "single_event"]
    if not df_recur.empty:
        ax.scatter(
            df_recur["x_jit"].values, df_recur["pi_value"].values,
            s=SCATTER_SIZE_PT2 * 0.95, c=RECURRENT_OVERLAY_COLOR,
            marker="x", linewidths=1.0, alpha=RECURRENT_OVERLAY_ALPHA, zorder=13,
        )
    if not df_single.empty:
        ax.scatter(
            df_single["x_jit"].values, df_single["pi_value"].values,
            s=SCATTER_SIZE_PT2 * 0.5, c=SINGLE_EVENT_OVERLAY_COLOR,
            marker="o", linewidths=0.0, alpha=1.0, zorder=12,
        )

    # --- MEDIANS (topmost) ---
    medians = df_long.groupby("region_type", observed=False)["pi_value"].median()
    for region, median_val in medians.items():
        x_center = X_POS[region]
        xmin = x_center - 0.15 / 2
        xmax = x_center + 0.15 / 2
        ax.hlines(
            y=median_val, xmin=xmin, xmax=xmax,
            color=MEDIAN_LINE_COLOR, linestyle="-", linewidth=MEDIAN_LINE_WIDTH,
            zorder=14, alpha=0.9,
        )

    # --- Axes cosmetics ---
    ax.set_ylabel("Mean Nucleotide Diversity (π)", fontsize=12)
    ax.set_xticks([X_POS["Flanking"], X_POS["Middle"]])
    ax.set_xticklabels(["Flanking", "Middle"], fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("")   # NO 'region_type'
    ax.set_title("")    # No main title
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, offset=5, trim=False)
    ax.yaxis.set_major_formatter(FuncFormatter(_tick_formatter))

    # Y-limits + buffer
    all_pi = df_long["pi_value"].dropna()
    if not all_pi.empty:
        lo, hi = all_pi.min(), all_pi.max()
        pad = (hi - lo) * 0.05 if hi > lo else 0.1
        ax.set_ylim(bottom=max(0, lo - pad), top=hi + pad)
    else:
        ax.set_ylim(0, 1)

    # --- Layout: reserve right margin for legends + colorbar ---
    try:
        fig.tight_layout(rect=[0.06, 0.05, 0.78, 0.98])  # ~22% for right side
    except Exception as e:
        logger.error(f"tight_layout error: {e}", exc_info=True)

    # --- Colorbar (right, lower area) ---
    cbar = None
    cax = None
    if scalar_mappable is not None:
        ax_pos = ax.get_position()
        cax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0 + 0.10, 0.03, ax_pos.height * 0.55])
        cbar = fig.colorbar(scalar_mappable, cax=cax)
        cbar.set_label("Log2 (π Middle / π Flanking)", rotation=270, labelpad=14, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_tick_formatter))

    # --- Annotation (top-left inside axes) ---
    overall = test_results.get("Overall", {})
    p_val = overall.get("mean_p", np.nan)
    n_pairs = overall.get("n_valid_pairs", len(paired_list))
    mean_diff_val = float(np.nanmean(df_paired["Middle"] - df_paired["Flanking"]))
    note = (
        f"N = {n_pairs} pairs\n"
        f"Mean Diff (Middle − Flank): {format_number_superscript(mean_diff_val, sig=4)}\n"
        f"{format_p_value(p_val)} (Permutation Test)"
    )
    ax.text(
        0.03, 0.97, note, transform=ax.transAxes,
        ha="left", va="top", fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="grey"),
        zorder=20,
    )

    # --- Legends (far right, above colorbar) ---
    overlay_handles = [
        Line2D([0], [0], marker="x", color=RECURRENT_OVERLAY_COLOR, label="Recurrent",
               markersize=np.sqrt(SCATTER_SIZE_PT2) * 0.95, linestyle="None", alpha=RECURRENT_OVERLAY_ALPHA),
        Line2D([0], [0], marker="o", color=SINGLE_EVENT_OVERLAY_COLOR, label="Single-event",
               markersize=np.sqrt(SCATTER_SIZE_PT2) * 0.75, linestyle="None"),
    ]
    seq_handles = [
        Line2D([0], [0], marker="o", color="w", label="Direct",
               markerfacecolor=DIRECT_BASE_COLOR, markersize=np.sqrt(SCATTER_SIZE_PT2),
               markeredgewidth=0.2, markeredgecolor="white", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", label="Inverted",
               markerfacecolor=INVERTED_BASE_COLOR, markersize=np.sqrt(SCATTER_SIZE_PT2),
               markeredgewidth=0.2, markeredgecolor="white", linestyle="None"),
    ]

    fig_trans = fig.transFigure
    if cax is not None:
        cbpos = cax.get_position()
        legend_x = cbpos.x0                    # align with cbar left
        overlay_y = min(0.97, cbpos.y0 + cbpos.height + 0.14)
        seq_y     = min(0.95, cbpos.y0 + cbpos.height + 0.05)
    else:
        axpos = ax.get_position()
        legend_x = axpos.x1 + 0.02
        overlay_y = min(0.97, axpos.y1)
        seq_y     = max(0.80, overlay_y - 0.12)

    fig.legend(
        handles=overlay_handles, title="Inversion Class",
        fontsize=9, title_fontsize=10, loc="lower left",
        bbox_to_anchor=(legend_x, overlay_y), bbox_transform=fig_trans,
        borderaxespad=0.2, frameon=True, framealpha=0.9,
    )
    fig.legend(
        handles=seq_handles, title="Sequence Type",
        fontsize=9, title_fontsize=10, loc="lower left",
        bbox_to_anchor=(legend_x, seq_y), bbox_transform=fig_trans,
        borderaxespad=0.2, frameon=True, framealpha=0.9,
    )

    # --- Save ---
    out = OUTPUT_DIR / f"pi_overall_{stat_type}_violin_paired_L2FC_overlays_unicode.pdf"
    try:
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure to {out}")
    except Exception as e:
        logger.error(f"Failed to save figure: {e}")

    logger.info(f"Plot built in {time.time() - start_time:.2f}s.")
    return fig

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    t0 = time.time()
    logger.info("--- Starting Pi Flanking Regions Analysis ---")
    logger.info("--- Mode: Filtered Pi & Length Only ---")

    # Quick sanity: confirm DejaVu Sans resolution path
    try:
        font_path = fm.findfont("DejaVu Sans")
        logger.info(f"Using font: {font_path}")
    except Exception as e:
        logger.warning(f"Could not resolve 'DejaVu Sans' via font manager: {e}")

    # Load inversion info
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

    # Load pi data
    pi_path = Path(PI_DATA_FILE)
    if not pi_path.is_file():
        logger.error(f"Pi data file not found: {pi_path}")
        return
    pi_sequences = load_pi_data(pi_path)
    if not pi_sequences:
        logger.error("No valid sequences after filtering. Exiting.")
        return

    # Stats
    flanking_stats = calculate_flanking_stats(pi_sequences)
    if not flanking_stats:
        logger.error("No sequences after flanking stats (NaN/length issues). Exiting.")
        return

    # Categorize & tests
    categories = categorize_sequences(flanking_stats, recurrent_regions, single_event_regions)
    test_results = perform_statistical_tests(categories, flanking_stats)

    # Plot
    fig = create_paired_violin_with_overlays(flanking_stats, test_results)

    # Summary log
    logger.info("\n--- Analysis Summary ---")
    logger.info(f"Input Pi File: {PI_DATA_FILE}")
    logger.info(f"Input Inversion File: {INVERSION_FILE}")
    logger.info(f"Total sequences in final analysis: {len(flanking_stats)}")

    logger.info("\nPaired Test Results (Middle vs Flanking - mean difference):")
    logger.info("-" * 95)
    logger.info(f"{'Category':<25} {'N Valid Pairs':<15} {'Mean Diff (M−F)':<22} {'Permutation p':<20} {'Normality p (Diffs)':<20}")
    logger.info("-" * 95)

    for cat in CATEGORY_ORDER_WITH_OVERALL:
        res = test_results.get(cat, {})
        n_valid = res.get("n_valid_pairs", 0)
        if cat == "Overall":
            seqs = flanking_stats
        else:
            seqs = categories.get(CAT_MAPPING.get(cat, ""), [])
        mean_diff = np.nan
        if seqs and n_valid > 0:
            mm = np.array([s["middle_mean"] for s in seqs], dtype=float)
            ff = np.array([s["flanking_mean"] for s in seqs], dtype=float)
            valid = ~np.isnan(mm) & ~np.isnan(ff)
            if np.sum(valid) > 0:
                mean_diff = np.mean(mm[valid] - ff[valid])

        mean_p = res.get("mean_p", np.nan)
        norm_p = res.get("mean_normality_p", np.nan)
        md_str = format_number_superscript(float(mean_diff), sig=4) if pd.notna(mean_diff) else "N/A"
        p_str  = format_p_value(mean_p)
        np_str = format_number_superscript(float(norm_p), sig=3) if pd.notna(norm_p) else ("N/A" if n_valid < 3 else "Const")

        logger.info(f"{cat:<25} {n_valid:<15} {md_str:<22} {p_str:<20} {np_str:<20}")

    logger.info("-" * 95)
    logger.info(f"--- Finished in {time.time() - t0:.2f}s ---")

    if fig:
        plt.close(fig)

if __name__ == "__main__":
    main()
