import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('pi_flanking_analysis_filtered_pi_length') #  logger name

# Constants
MIN_LENGTH = 150_000  # Minimum sequence length for analysis
FLANK_SIZE = 50_000  # Size of flanking regions (each end)
PERMUTATIONS = 10000 # Number of permutations for significance testing
BAR_ALPHA = 0.3      # Transparency for plot bars (Used previously, kept for reference)

# File paths (Update these paths if necessary)
PI_DATA_FILE = 'per_site_diversity_output.falsta'
INVERSION_FILE = 'inv_info.tsv'
OUTPUT_DIR = Path('pi_analysis_results_filtered_pi_length') #  output directory name

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Category mapping for consistency
CAT_MAPPING = {
    'Recurrent Inverted': 'recurrent_inverted',
    'Recurrent Direct': 'recurrent_direct',
    'Single-event Inverted': 'single_event_inverted',
    'Single-event Direct': 'single_event_direct'
}
REVERSE_CAT_MAPPING = {v: k for k, v in CAT_MAPPING.items()}
# Define the specific order for categories in plots and tables
CATEGORY_ORDER = ['Recurrent Inverted', 'Recurrent Direct', 'Single-event Inverted', 'Single-event Direct']
CATEGORY_ORDER_WITH_OVERALL = CATEGORY_ORDER + ['Overall']
INTERNAL_CATEGORIES = list(CAT_MAPPING.values())

# Plotting Style
plt.style.use('seaborn-v0_8-ticks') #  style for better appearance
# Using Tableau 10 color scheme for better distinction in general
COLOR_PALETTE = plt.cm.tab10.colors
FLANKING_COLOR = COLOR_PALETTE[0] # Blue for Flanking
MIDDLE_COLOR = COLOR_PALETTE[1]  # Orange for Middle
# Paired plot specific styles
SCATTER_ALPHA = 0.35 # Slightly more transparent points
SCATTER_SIZE = 3.0   # Smaller points for less overlap
LINE_ALPHA = 0.15    # More transparent lines
LINE_WIDTH = 0.9     # Thinner lines
VIOLIN_ALPHA = 0.2   # More transparent violin body
MEDIAN_LINE_COLOR = 'k' # Black median line
MEDIAN_LINE_WIDTH = 1.5 # Clear median line width
DEFAULT_LINE_COLOR = 'grey' # Color for lines if L2FC calculation fails
PLOT_COLORMAP = cm.coolwarm # Colormap for paired lines based on L2FC

# --- Helper Functions ---

def normalize_chromosome(chrom: str) -> str | None:
    """
    Normalize chromosome name to 'chrN' format (e.g., chr1, chrX).
    Handles 'chr_', 'chr', or just number/letter inputs.
    Returns None if input seems invalid.
    """
    if not isinstance(chrom, str):
        chrom = str(chrom) # Attempt to convert if not string (e.g., from pandas)
    chrom = chrom.strip().lower()
    if chrom.startswith('chr_'):
        chrom_part = chrom[4:]
    elif chrom.startswith('chr'):
        chrom_part = chrom[3:]
    else:
        chrom_part = chrom

    # Basic check if the remaining part seems valid (number or X/Y/M/W/Z)
    # Allow combinations like chrUn_xxx found in some assemblies
    if chrom_part.isalnum() or chrom_part in ('x', 'y', 'm', 'w', 'z') or '_' in chrom_part:
        return f"chr{chrom_part}"
    else:
        logger.warning(f"Could not normalize chromosome: '{chrom}'. Invalid format.")
        return None

def extract_coordinates_from_header(header: str) -> dict | None:
    """
    Extract genomic coordinates and group from pi sequence header using regex.
    Example: ">filtered_pi_chr_1_start_13104251_end_13122521_group_1"
    Returns None if parsing fails. Avoids try-except.
    Only processes headers containing 'filtered_pi'.
    """
    # Require 'filtered_pi' in the header for extraction
    if 'filtered_pi' not in header.lower():
        return None

    # Regex to capture chromosome, start, end, and group
    # Allows 'chr' or 'chr_' prefix for chromosome part
    pattern = re.compile(
        # start of line or '>' precedes 'filtered_pi' to avoid partial matches
        # Then look for the standard coordinate pattern
        r'>.*?filtered_pi.*?_chr_?([\w\.\-]+)_start_(\d+)_end_(\d+)(?:_group_([01]))?',
        re.IGNORECASE # Ignore case for keywords
    )
    match = pattern.search(header)

    if match:
        chrom_part, start_str, end_str, group_str = match.groups()

        chrom = normalize_chromosome(chrom_part)
        # Use explicit None check because start/end could be 0
        start = int(start_str) if start_str is not None else None
        end = int(end_str) if end_str is not None else None
        # Group 1 = inverted, Group 0 = direct. Default to None if not present.
        group = int(group_str) if group_str is not None else None

        if chrom is None or start is None or end is None: # Normalization failed or coords missing
            logger.warning(f"Chromosome normalization or coordinate extraction failed for header: {header[:70]}...")
            return None

        # Basic sanity check on coordinates
        if start >= end:
             logger.warning(f"Start >= End coordinate in header: {header[:70]}... ({start} >= {end})")
             return None

        result = {
            'chrom': chrom,
            'start': start,
            'end': end,
            'group': group # Will be None if group wasn't found
        }
        return result
    else:
        # Log failure only if it was supposed to be a filtered_pi header
        # (This check is technically redundant with the initial 'if' but adds clarity)
        if 'filtered_pi' in header.lower():
            logger.warning(f"Failed to extract coordinates using regex from filtered_pi header: {header[:70]}...")
        return None

def map_regions_to_inversions(inversion_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Create mapping dictionaries for recurrent and single-event inversion regions.
    Returns two dictionaries: {chromosome: [(start, end), ...]}
    Handles potential NaN or non-numeric data gracefully.
    """
    logger.info("Creating inversion region mappings...")
    recurrent_regions = {}
    single_event_regions = {}

    # Correct data types, coercing errors to NaT/NaN
    inversion_df['Start'] = pd.to_numeric(inversion_df['Start'], errors='coerce')
    inversion_df['End'] = pd.to_numeric(inversion_df['End'], errors='coerce')
    inversion_df['0_single_1_recur_consensus'] = pd.to_numeric(inversion_df['0_single_1_recur_consensus'], errors='coerce')
    # Chromosome is string for normalization
    inversion_df['Chromosome'] = inversion_df['Chromosome'].astype(str)


    # Drop rows where essential information is missing
    original_rows = len(inversion_df)
    inversion_df = inversion_df.dropna(subset=['Chromosome', 'Start', 'End', '0_single_1_recur_consensus'])
    dropped_rows = original_rows - len(inversion_df)
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows from inversion info due to missing values.")

    for _, row in inversion_df.iterrows():
        chrom = normalize_chromosome(row['Chromosome'])
        if chrom is None:
            continue # Skip if chromosome name is invalid

        start = int(row['Start']) # Already numeric and not NaN
        end = int(row['End'])
        is_recurrent = int(row['0_single_1_recur_consensus']) == 1

        target_dict = recurrent_regions if is_recurrent else single_event_regions

        if chrom not in target_dict:
            target_dict[chrom] = []
        target_dict[chrom].append((start, end))

    recurrent_count = sum(len(regions) for regions in recurrent_regions.values())
    single_event_count = sum(len(regions) for regions in single_event_regions.values())

    logger.info(f"Mapped {recurrent_count} recurrent regions across {len(recurrent_regions)} chromosomes.")
    logger.info(f"Mapped {single_event_count} single-event regions across {len(single_event_regions)} chromosomes.")
    if recurrent_regions:
        logger.debug(f"Recurrent chromosomes: {sorted(list(recurrent_regions.keys()))}")
    if single_event_regions:
        logger.debug(f"Single-event chromosomes: {sorted(list(single_event_regions.keys()))}")

    return recurrent_regions, single_event_regions

def is_overlapping(region1_start: int, region1_end: int, region2_start: int, region2_end: int) -> bool:
    """
    Check if two genomic regions overlap or are separated by at most 1 base pair
    (inclusive coordinates). Returns True if they overlap, are adjacent,
    or have exactly a 1 bp gap between them.
    """
    # Assumes coordinates are 1-based inclusive.
    # Condition checks if region 1 doesn't end more than 1 bp before region 2 starts,
    # AND if region 2 doesn't end more than 1 bp before region 1 starts.
    # This covers actual overlap, adjacency (0 bp gap), and a 1 bp gap.
    return (region1_end + 2) >= region2_start and (region2_end + 2) >= region1_start

def determine_inversion_type(coords: dict, recurrent_regions: dict, single_event_regions: dict) -> str:
    """
    Determine inversion type based on overlap with mapped regions.
    Returns 'recurrent', 'single_event', 'ambiguous', or 'unknown'.
    """
    chrom = coords.get('chrom')
    start = coords.get('start')
    end = coords.get('end')

    if not all([chrom, isinstance(start, int), isinstance(end, int)]):
        return 'unknown' # Invalid coordinates

    is_recurrent = False
    if chrom in recurrent_regions:
        for region_start, region_end in recurrent_regions[chrom]:
            if is_overlapping(start, end, region_start, region_end):
                is_recurrent = True
                break

    is_single_event = False
    if chrom in single_event_regions:
        for region_start, region_end in single_event_regions[chrom]:
            if is_overlapping(start, end, region_start, region_end):
                is_single_event = True
                break

    if is_recurrent and is_single_event:
        return 'ambiguous'
    elif is_recurrent:
        return 'recurrent'
    elif is_single_event:
        return 'single_event'
    else:
        return 'unknown'

def paired_permutation_test(x: np.ndarray, y: np.ndarray, num_permutations: int = PERMUTATIONS, use_median: bool = False) -> float:
    """
    Perform a paired permutation test using mean or median of differences.
    Handles potential NaN values in the input arrays robustly.
    Returns the two-tailed p-value.
    """
    if len(x) != len(y):
         # This should ideally not happen if data processing is correct
         logger.error(f"Input arrays x ({len(x)}) and y ({len(y)}) have different lengths for paired test.")
         return np.nan # Return NaN if lengths differ

    # Calculate differences, ignoring pairs with NaN in *either* x or y
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    differences = x[valid_indices] - y[valid_indices]
    num_valid_pairs = len(differences)

    # Need at least 2 valid pairs to perform the test
    if num_valid_pairs < 2:
        logger.warning(f"Cannot perform permutation test: only {num_valid_pairs} valid pairs found after NaN removal.")
        return np.nan

    # Choose the test statistic function (mean or median)
    stat_func = np.median if use_median else np.mean

    # Calculate the observed statistic on the valid differences
    observed_stat = stat_func(differences)
    observed_abs_stat = abs(observed_stat)

    # If observed statistic is exactly zero, p-value is 1 (cannot be more extreme)
    # Check with tolerance due to potential floating point inaccuracies
    if np.isclose(observed_stat, 0):
        return 1.0

    count_extreme_or_more = 0
    for _ in range(num_permutations):
        # Randomly flip signs of the *valid* differences
        signs = np.random.choice([1, -1], size=num_valid_pairs, replace=True)
        permuted_diffs = differences * signs
        permuted_stat = stat_func(permuted_diffs)

        # Two-tailed test: count if permuted statistic's absolute value
        # is greater than or equal to the observed absolute statistic
        if abs(permuted_stat) >= observed_abs_stat:
            count_extreme_or_more += 1

    p_value = count_extreme_or_more / num_permutations

    return p_value

# --- Data Loading and Processing ---

def parse_pi_data_line(line: str) -> np.ndarray | None:
    """Parses a comma-separated line of pi values into a numpy array."""
    # Faster parsing by avoiding list comprehension with conditional
    try:
        values = line.split(',')
        data = np.full(len(values), np.nan, dtype=np.float32)
        for i, x in enumerate(values):
            val_str = x.strip()
            if val_str and val_str.upper() != 'NA':
                 # Attempt conversion, will raise ValueError if invalid
                 data[i] = float(val_str)
        # Check if all values are NaN after conversion (e.g., line was just commas or NAs)
        if np.all(np.isnan(data)):
             logger.debug("Parsed data line resulted in all NaNs, skipping.")
             return None
        return data
    except ValueError as e:
        # Handle cases where a value cannot be converted to float and isn't NA
        # Extract the problematic value more reliably
        problem_val = 'unknown'
        try:
            # Try to identify the specific value causing the error
            float(val_str)
        except ValueError:
            problem_val = f"'{val_str}'"
        logger.warning(f"ValueError parsing data line value {problem_val}: {e}. Skipping line segment: {line[:50]}...")
        return None
    except Exception as e: # Catch other unexpected errors during parsing
        logger.error(f"Unexpected error parsing data line: {e}. Skipping line segment: {line[:50]}...", exc_info=True)
        return None

def load_pi_data(file_path: str | Path) -> list[dict]:
    """
    Load filtered pi data from a FASTA-like file.
    Assumes header lines start with '>' and data follows on the next line(s).
    Handles multi-line sequences by concatenating.
    Filters sequences based on MIN_LENGTH.
    Only processes sequences with headers containing 'filtered_pi'.
    Extracts coordinates and group info from these headers.
    """
    logger.info(f"Loading pi data from {file_path}")
    logger.info(f"Applying filters: Header must contain 'filtered_pi', Sequence length >= {MIN_LENGTH}")
    start_time = time.time()

    pi_sequences = []
    sequences_processed = 0
    headers_read = 0
    skipped_short = 0
    skipped_not_filtered_pi = 0 # Explicitly track skipped non-'filtered_pi'
    skipped_coord_error = 0
    skipped_data_error = 0
    skipped_missing_group = 0 # Track missing group specifically

    current_header = None
    current_sequence_parts = []
    is_current_header_valid = False # Track if the current header is one we care about

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                if line.startswith('>'):
                    headers_read += 1
                    # --- Process the *previous* sequence if one was being read ---
                    if is_current_header_valid and current_header and current_sequence_parts:
                        sequences_processed += 1
                        full_sequence_line = "".join(current_sequence_parts)
                        pi_data = parse_pi_data_line(full_sequence_line)

                        if pi_data is not None:
                            actual_length = len(pi_data) # Use actual length after parsing
                            if actual_length >= MIN_LENGTH:
                                # Coords should already be extracted and checked when header was read
                                coords = extract_coordinates_from_header(current_header) # Re-extract for safety, though inefficient
                                if coords:
                                    if coords.get('group') is not None: # Check group info exists
                                        is_inverted = coords['group'] == 1
                                        pi_sequences.append({
                                            'header': current_header,
                                            'coords': coords,
                                            'data': pi_data,
                                            'length': actual_length,
                                            'is_inverted': is_inverted
                                        })
                                    else:
                                        logger.warning(f"Skipping sequence: Missing group info in header '{current_header[:70]}...'")
                                        skipped_missing_group += 1
                                else:
                                    # This case should be rare if the logic below is correct
                                    logger.error(f"Internal Logic Error: Header '{current_header[:70]}...' was marked valid but coordinate extraction failed later.")
                                    skipped_coord_error += 1
                            else:
                                logger.debug(f"Skipping sequence (Too Short: {actual_length} < {MIN_LENGTH}): Header '{current_header[:70]}...'")
                                skipped_short += 1
                        else:
                            logger.warning(f"Skipping sequence (Data Parse Error): Header '{current_header[:70]}...'")
                            skipped_data_error +=1 # Parsing the data string failed

                    # --- Start processing the *new* sequence header ---
                    # Reset state for the new header
                    current_header = line
                    current_sequence_parts = []
                    is_current_header_valid = False # Assume invalid until checked

                    # Check if this new header is 'filtered_pi'
                    if 'filtered_pi' in current_header.lower():
                         # Attempt to extract coordinates immediately to validate header structure
                         coords_check = extract_coordinates_from_header(current_header)
                         if coords_check:
                             is_current_header_valid = True # Header looks ok, start collecting data
                         else:
                             logger.warning(f"Skipping sequence: 'filtered_pi' header failed coordinate/format check '{current_header[:70]}...'")
                             skipped_coord_error += 1
                             current_header = None # data isn't collected for this invalid header
                    else:
                        # It's not a 'filtered_pi' header, mark as skipped and ignore data lines
                        skipped_not_filtered_pi += 1
                        current_header = None # data isn't collected

                elif is_current_header_valid and current_header:
                    # Append data line only if the current header is valid ('filtered_pi' and coords ok)
                    current_sequence_parts.append(line)

            # --- Process the very last sequence in the file after loop ends ---
            if is_current_header_valid and current_header and current_sequence_parts:
                sequences_processed += 1
                full_sequence_line = "".join(current_sequence_parts)
                pi_data = parse_pi_data_line(full_sequence_line)

                if pi_data is not None:
                    actual_length = len(pi_data)
                    if actual_length >= MIN_LENGTH:
                        coords = extract_coordinates_from_header(current_header)
                        if coords:
                            if coords.get('group') is not None:
                                is_inverted = coords['group'] == 1
                                pi_sequences.append({
                                    'header': current_header,
                                    'coords': coords,
                                    'data': pi_data,
                                    'length': actual_length,
                                    'is_inverted': is_inverted
                                })
                            else:
                                logger.warning(f"Skipping last sequence: Missing group info in header '{current_header[:70]}...'")
                                skipped_missing_group += 1
                        else:
                             logger.error(f"Internal Logic Error: Last header '{current_header[:70]}...' was marked valid but coordinate extraction failed.")
                             skipped_coord_error += 1
                    else:
                        logger.debug(f"Skipping last sequence (Too Short: {actual_length} < {MIN_LENGTH}): Header '{current_header[:70]}...'")
                        skipped_short += 1
                else:
                    logger.warning(f"Skipping last sequence (Data Parse Error): Header '{current_header[:70]}...'")
                    skipped_data_error += 1

    except FileNotFoundError:
        logger.error(f"Fatal Error: Pi data file not found at {file_path}")
        return [] # Return empty list if file not found
    except Exception as e: # Catch other potential file reading errors
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}", exc_info=True)
        return []

    elapsed_time = time.time() - start_time
    logger.info(f"Read {headers_read} headers, processed {sequences_processed} potential 'filtered_pi' sequences in {elapsed_time:.2f} seconds.")
    logger.info(f"Loaded {len(pi_sequences)} valid sequences meeting all criteria ('filtered_pi', length >= {MIN_LENGTH}, valid coords, group info).")
    logger.info(f"Skipped sequences breakdown:")
    logger.info(f"  - Not 'filtered_pi': {skipped_not_filtered_pi}")
    logger.info(f"  - Too Short (< {MIN_LENGTH} bp): {skipped_short}")
    logger.info(f"  - Coordinate/Format Error in Header: {skipped_coord_error}")
    logger.info(f"  - Missing Group Info in Header: {skipped_missing_group}")
    logger.info(f"  - Data Parsing Error: {skipped_data_error}")

    if pi_sequences:
         # Log chromosome distribution
        chrom_counts = {}
        for seq in pi_sequences:
            chrom = seq['coords'].get('chrom', 'Unknown')
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
        logger.info(f"Chromosome distribution of loaded sequences: {chrom_counts}")
    else:
        logger.warning("No valid pi sequences were loaded that met all specified criteria.")

    return pi_sequences


def calculate_flanking_stats(pi_sequences: list[dict]) -> list[dict]:
    """
    Calculate mean and median pi for beginning, middle, and ending regions.
    Handles NaN values within regions. Returns list of dicts with stats.
    Skips sequences too short for flank analysis or where essential stats (middle) are NaN.
    """
    logger.info(f"Calculating flanking and middle statistics for {len(pi_sequences)} sequences...")
    start_time = time.time()
    results = []
    skipped_too_short_for_flanks = 0
    skipped_nan_middle = 0
    skipped_nan_flanks = 0

    for i, seq in enumerate(pi_sequences):
        data = seq['data']
        seq_id = seq['header'][:70] # For logging

        # Check if sequence is long enough for two flanks and at least one middle point
        min_req_len = 2 * FLANK_SIZE + 1
        if len(data) < min_req_len:
            logger.debug(f"Sequence {i} ({seq_id}...) length {len(data)} < {min_req_len}, skipping stat calculation.")
            skipped_too_short_for_flanks += 1
            continue

        # Extract regions
        beginning_flank = data[:FLANK_SIZE]
        ending_flank = data[-FLANK_SIZE:]
        middle_region = data[FLANK_SIZE:-FLANK_SIZE]

        # Calculate stats, ignoring NaNs using nan-aware functions
        stats = {
            'header': seq['header'],
            'coords': seq['coords'],
            'is_inverted': seq['is_inverted'],
            'length': seq['length'], # Keep length info
            'beginning_mean': np.nanmean(beginning_flank),
            'ending_mean': np.nanmean(ending_flank),
            'middle_mean': np.nanmean(middle_region),
            'beginning_median': np.nanmedian(beginning_flank), # Keep median calculation for potential future use
            'ending_median': np.nanmedian(ending_flank),
            'middle_median': np.nanmedian(middle_region)
        }

        # Calculate combined flanking stats AFTER individual stats are computed
        b_mean, e_mean = stats['beginning_mean'], stats['ending_mean']
        stats['flanking_mean'] = np.nanmean([b_mean, e_mean]) # nanmean handles if one is NaN

        b_med, e_med = stats['beginning_median'], stats['ending_median']
        stats['flanking_median'] = np.nanmean([b_med, e_med]) # Use nanmean of medians as combined measure

        # Check if any critical calculation resulted in NaN
        # Middle region MUST have a valid mean.
        # Flanking regions: at least one flank must have a valid mean.
        middle_is_nan = np.isnan(stats['middle_mean'])
        flanking_is_nan = np.isnan(stats['flanking_mean']) # Combined flanking mean is NaN only if BOTH individual flanks are NaN

        if middle_is_nan:
             logger.warning(f"Sequence {i} ({seq_id}...) middle region resulted in NaN mean stat. Skipping this sequence.")
             skipped_nan_middle += 1
             continue
        if flanking_is_nan:
            logger.warning(f"Sequence {i} ({seq_id}...) BOTH flanking regions resulted in NaN mean stats. Skipping this sequence.")
            skipped_nan_flanks += 1
            continue

        results.append(stats)

    elapsed_time = time.time() - start_time
    logger.info(f"Successfully calculated statistics for {len(results)} sequences in {elapsed_time:.2f} seconds.")
    if skipped_too_short_for_flanks > 0:
        logger.warning(f"Skipped {skipped_too_short_for_flanks} sequences too short (< {min_req_len} bp) for flank analysis.")
    if skipped_nan_middle > 0:
        logger.warning(f"Skipped {skipped_nan_middle} sequences due to NaN in middle region mean.")
    if skipped_nan_flanks > 0:
        logger.warning(f"Skipped {skipped_nan_flanks} sequences due to NaN in BOTH flanking region means.")

    return results

def categorize_sequences(flanking_stats: list[dict], recurrent_regions: dict, single_event_regions: dict) -> dict:
    """
    Categorize sequences based on inversion type and haplotype status.
    Returns a dictionary where keys are internal category names and values are lists of sequence stat dicts.
    Sequences overlapping both recurrent and single-event regions ('ambiguous') or
    not overlapping any defined region ('unknown') are logged but not included in categories.
    """
    logger.info("Categorizing sequences based on overlap with inversion regions...")
    start_time = time.time()
    # Initialize categories using the defined order
    categories = {CAT_MAPPING[cat_name]: [] for cat_name in CATEGORY_ORDER}
    ambiguous_count = 0
    unknown_count = 0
    sequences_without_coords = 0 # Should be low if load_pi_data/calculate_flanking_stats worked

    for seq_stats in flanking_stats:
        coords = seq_stats.get('coords')
        is_inverted = seq_stats.get('is_inverted') # Should exist if passed calculate_stats

        if coords is None or is_inverted is None:
             logger.warning(f"Sequence missing coordinates or inversion status during categorization: {seq_stats.get('header', 'Unknown Header')[:70]}...")
             sequences_without_coords += 1
             continue

        inversion_type = determine_inversion_type(coords, recurrent_regions, single_event_regions)

        category_key = None
        if inversion_type == 'recurrent':
            category_key = 'recurrent_inverted' if is_inverted else 'recurrent_direct'
        elif inversion_type == 'single_event':
            category_key = 'single_event_inverted' if is_inverted else 'single_event_direct'
        elif inversion_type == 'ambiguous':
            ambiguous_count += 1
            logger.debug(f"Sequence categorized as ambiguous (overlaps both recurrent and single): {seq_stats['header'][:70]}...")
        else: # 'unknown'
            unknown_count += 1
            logger.debug(f"Sequence categorized as unknown (no overlap with defined regions): {seq_stats['header'][:70]}...")

        if category_key and category_key in categories:
            categories[category_key].append(seq_stats)
        elif category_key:
            # This case should not happen if CAT_MAPPING is correct
             logger.error(f"Internal Error: Generated category key '{category_key}' not found in initialized categories for header {seq_stats['header'][:70]}...")

    elapsed_time = time.time() - start_time
    logger.info(f"Finished categorization in {elapsed_time:.2f} seconds.")
    logger.info("Category counts (sequences assigned):")
    total_categorized = 0
    for display_cat in CATEGORY_ORDER:
        internal_cat = CAT_MAPPING[display_cat]
        count = len(categories[internal_cat])
        logger.info(f"  {display_cat}: {count}")
        total_categorized += count
    logger.info(f"Total sequences categorized: {total_categorized}")
    logger.info(f"Sequences not categorized (Ambiguous overlap): {ambiguous_count}")
    logger.info(f"Sequences not categorized (Unknown overlap / Not in inv file): {unknown_count}")
    if sequences_without_coords > 0:
        logger.warning(f"Sequences skipped during categorization due to missing coords/group in stats data: {sequences_without_coords}")

    # Check for empty categories which might indicate issues
    for display_cat in CATEGORY_ORDER:
        internal_cat = CAT_MAPPING[display_cat]
        if not categories[internal_cat]:
            logger.warning(f"Category '{display_cat}' is empty after categorization.")

    return categories

# --- Statistical Testing ---

def perform_statistical_tests(categories: dict, all_sequences_stats: list[dict]) -> dict:
    """
    Performs paired permutation tests and Shapiro-Wilk normality tests
    for mean differences between flanking and middle regions for each category
    and overall.

    Args:
        categories (dict): Dictionary mapping internal category names to lists of sequence stats.
        all_sequences_stats (list): List containing stats for ALL sequences that passed previous steps.

    Returns:
        dict: Nested dictionary with p-values and valid pair counts.
              {'CategoryName': {'mean_p': float, 'mean_normality_p': float, 'n_valid_pairs': int}}.
    """
    logger.info("Performing statistical tests (Middle vs Flanking - mean difference)...")
    test_results = {}

    # Create a map including 'Overall' using the display names for consistency
    category_data_map = {REVERSE_CAT_MAPPING[k]: v for k, v in categories.items()}
    category_data_map['Overall'] = all_sequences_stats

    for category_name in CATEGORY_ORDER_WITH_OVERALL: # Iterate in defined order
        seq_list = category_data_map.get(category_name)
        if seq_list is None: # Should not happen with Overall, but check for safety
            logger.warning(f"  Category {category_name} not found in data map for testing. Skipping.")
            continue

        logger.info(f"  Testing category: {category_name} ({len(seq_list)} sequences)")
        # Initialize results for this category
        test_results[category_name] = {'mean_p': np.nan, 'mean_normality_p': np.nan, 'n_valid_pairs': 0}

        if len(seq_list) < 2: # Need at least 2 sequences for any comparison
            logger.warning(f"    Skipping tests for {category_name}: < 2 sequences available.")
            continue

        # Extract paired mean values, handling potential NaNs from calculations
        flanking_means = np.array([s['flanking_mean'] for s in seq_list], dtype=float)
        middle_means = np.array([s['middle_mean'] for s in seq_list], dtype=float)

        # Determine number of valid pairs (non-NaN in both flanking and middle)
        valid_indices = ~np.isnan(flanking_means) & ~np.isnan(middle_means)
        num_valid_pairs = int(np.sum(valid_indices)) # Cast to int for reporting
        test_results[category_name]['n_valid_pairs'] = num_valid_pairs

        if num_valid_pairs >= 2:
             # --- Permutation Test (mean differences) ---
             # Test: Middle vs Flanking (using the valid pairs)
             mean_perm_p = paired_permutation_test(
                 middle_means[valid_indices], flanking_means[valid_indices], use_median=False
             )
             test_results[category_name]['mean_p'] = mean_perm_p
             logger.info(f"    Permutation test (mean): p = {mean_perm_p:.4g} ({num_valid_pairs} valid pairs)")

             # --- Normality Test (on the mean Differences of valid pairs) ---
             if num_valid_pairs >= 3: # Shapiro-Wilk requires at least 3 samples
                 differences = middle_means[valid_indices] - flanking_means[valid_indices]
                 # Check for constant data which causes Shapiro-Wilk error
                 if len(np.unique(differences)) > 1:
                     try:
                         shapiro_stat, shapiro_p = shapiro(differences)
                         test_results[category_name]['mean_normality_p'] = shapiro_p
                         # Log normality result clearly
                         logger.info(f"    Normality test (mean Diffs): Shapiro-Wilk W={shapiro_stat:.4f}, p={shapiro_p:.4g}")
                     except ValueError as e:
                         # Should be less likely after unique check, but catch just in case
                         logger.warning(f"    Could not perform normality test for {category_name} (ValueError): {e}")
                 else:
                     logger.warning(f"    Skipping normality test for {category_name}: All differences are identical ({num_valid_pairs} pairs).")
                     test_results[category_name]['mean_normality_p'] = np.nan # Indicate test not performed meaningfully

             else:
                 logger.info(f"    Skipping normality test for {category_name}: < 3 valid pairs ({num_valid_pairs}).")

        else:
            logger.warning(f"    Skipping tests for {category_name}: < 2 valid pairs found ({num_valid_pairs}).")

    return test_results

# --- Diagnostic Function (Optional but potentially useful) ---

def diagnose_single_event_discrepancy(
    pi_file_path: str | Path,
    single_event_regions: dict,
    min_length: int,
    inv_info_path: str | Path
    ) -> None:
    """
    Diagnoses why counts for single-event direct vs inverted might differ *after filtering*.

    Checks each defined single-event region against the pi data headers
    to see if direct/inverted sequences are found and why they might be
    filtered out by `load_pi_data` (Not Found, Not 'filtered_pi', Too Short, Coord/Group Error).

    Args:
        pi_file_path: Path to the per_site_diversity_output.falsta file.
        single_event_regions: Dict mapping {chrom: [(start, end), ...]} for SINGLE events.
        min_length: The minimum sequence length threshold used in load_pi_data.
        inv_info_path: Path to the inversion info file (for logging context).
    """
    logger.info("--- STARTING DIAGNOSIS: Single-Event Discrepancy (Post-Filtering) ---")
    logger.info(f"Analyzing Pi file: {pi_file_path}")
    logger.info(f"Based on Single-Event regions from: {inv_info_path}")
    logger.info(f"Using Filters: Header contains 'filtered_pi', Min Length >= {min_length}")

    if not single_event_regions:
        logger.warning("DIAGNOSIS: No single-event regions were defined. Skipping diagnosis.")
        logger.info("--- ENDING DIAGNOSIS ---")
        return

    # Initialize tracking for each defined single-event region
    region_status = {}
    defined_region_count = 0
    for chrom, regions in single_event_regions.items():
        for start, end in regions:
            region_key = (chrom, start, end)
            region_status[region_key] = {
                'direct_status': 'Region Not Matched', 'inverted_status': 'Region Not Matched',
                'direct_details': '', 'inverted_details': ''
            }
            defined_region_count += 1
    logger.info(f"Tracking {defined_region_count} defined single-event regions.")

    processed_headers = 0
    potential_matches = 0
    headers_analyzed = 0 # Count headers passing initial '>' check

    try:
        with open(pi_file_path, 'r') as f:
            current_header = None
            line_num = 0
            while True: # Process header and potential data line(s)
                header_line = f.readline()
                line_num += 1
                if not header_line: break # End of file
                header_line = header_line.strip()
                if not header_line or not header_line.startswith('>'): continue

                headers_analyzed += 1
                current_header = header_line

                # 1. Check if header contains 'filtered_pi'
                if 'filtered_pi' not in current_header.lower():
                    # This header would be skipped by load_pi_data - check if it overlaps a region
                    coords_raw = extract_coordinates_from_header(current_header.replace('filtered_pi', 'filtered_pi')) # Hacky way to maybe parse coords
                    if coords_raw: # Parsed coords even without 'filtered_pi'? Unlikely with current regex but check
                        if coords_raw['chrom'] in single_event_regions:
                             for r_start, r_end in single_event_regions[coords_raw['chrom']]:
                                 if is_overlapping(coords_raw['start'], coords_raw['end'], r_start, r_end):
                                     region_key = (coords_raw['chrom'], r_start, r_end)
                                     target_key = 'direct_status' if coords_raw.get('group') == 0 else 'inverted_status'
                                     target_details = 'direct_details' if coords_raw.get('group') == 0 else 'inverted_details'
                                     if region_status[region_key][target_key] == 'Region Not Matched':
                                         region_status[region_key][target_key] = "Filtered (Not 'filtered_pi')"
                                         region_status[region_key][target_details] = f"Header: {current_header[:70]}"
                                     break
                    continue # Skip to next header

                # 2. Check if header has valid coordinates and group
                coords = extract_coordinates_from_header(current_header)
                if not coords or coords.get('group') is None:
                     # Header is 'filtered_pi' but fails coord/group extraction
                     # Check if the *intended* region (if parsable somehow) matches a defined SE region
                     # This is complex, log the failure for now
                     logger.debug(f"DIAGNOSIS: Header '{current_header[:70]}' is 'filtered_pi' but failed coord/group parse.")
                     # We cannot reliably link this to a region without coords. It contributes to `skipped_coord_error` or `skipped_missing_group`.
                     continue # Skip to next header

                # Header is 'filtered_pi' and has valid coords/group
                header_chrom = coords['chrom']
                header_start = coords['start']
                header_end = coords['end']
                header_group = coords['group'] # 0 or 1

                # 3. Check sequence length (requires reading next line(s))
                # Simplified: Read only the *next* line. Assumes data is on one line for diagnosis.
                data_line = f.readline()
                line_num += 1
                actual_length = 0
                if data_line:
                    data_line = data_line.strip()
                    try:
                        values = data_line.split(',')
                        actual_length = sum(1 for x in values if x.strip() and x.strip().upper() != 'NA')
                    except Exception:
                         actual_length = 0 # Cannot determine length if parsing fails

                is_too_short = actual_length < min_length

                # 4. Check for overlap with defined single-event regions
                matched_region = False
                if header_chrom in single_event_regions:
                    for region_start, region_end in single_event_regions[header_chrom]:
                        if is_overlapping(header_start, header_end, region_start, region_end):
                            potential_matches += 1
                            matched_region = True
                            region_key = (header_chrom, region_start, region_end)
                            target_status_key = 'direct_status' if header_group == 0 else 'inverted_status'
                            target_details_key = 'direct_details' if header_group == 0 else 'inverted_details'

                            status = "Error"
                            details = f"Header: {current_header[:70]}... Len={actual_length}"
                            if is_too_short:
                                status = f"Filtered (Too Short: {actual_length})"
                            else:
                                status = "Found & Passed Filters"

                            # Update status, prioritizing "Passed" if found, then "Filtered", then "Not Matched"
                            current_region_s = region_status[region_key][target_status_key]
                            if current_region_s == 'Region Not Matched' or status == "Found & Passed Filters" or "Filtered" in status:
                                # Prioritize 'Passed' if already 'Filtered'
                                if status == "Found & Passed Filters" or current_region_s == 'Region Not Matched':
                                     region_status[region_key][target_status_key] = status
                                     region_status[region_key][target_details_key] = details
                                # If current is 'Passed', don't overwrite with 'Filtered' from another overlapping header
                                elif "Filtered" in status and "Filtered" not in current_region_s and current_region_s != "Found & Passed Filters":
                                     region_status[region_key][target_status_key] = status
                                     region_status[region_key][target_details_key] = details


                            break # Assume header maps to only one defined SE region for simplicity


    except FileNotFoundError:
        logger.error(f"DIAGNOSIS: Pi data file not found at {pi_file_path}")
        logger.info("--- ENDING DIAGNOSIS ---")
        return
    except Exception as e:
        logger.error(f"DIAGNOSIS: An error occurred reading {pi_file_path}: {e}", exc_info=True)
        logger.info("--- ENDING DIAGNOSIS ---")
        return

    logger.info(f"Diagnosis scan complete. Analyzed {headers_analyzed} '>' headers.")
    logger.info(f"Found {potential_matches} headers potentially overlapping defined single-event regions.")

    # Summarize the findings based on the *first* reason a region's haplotype might be excluded
    summary_counts = {
        'direct': {'Region Not Matched': 0, "Filtered (Not 'filtered_pi')": 0, 'Filtered (Too Short)': 0, 'Filtered (Coord/Group Err)': 0, 'Found & Passed Filters': 0},
        'inverted': {'Region Not Matched': 0, "Filtered (Not 'filtered_pi')": 0, 'Filtered (Too Short)': 0, 'Filtered (Coord/Group Err)': 0, 'Found & Passed Filters': 0}
    }
    detailed_examples = {'direct_filtered_short': [], 'inverted_filtered_short': []}

    # Re-calculate counts based on final status
    for region_key, status_dict in region_status.items():
        d_stat = status_dict['direct_status']
        if "Region Not Matched" in d_stat: summary_counts['direct']['Region Not Matched'] += 1
        elif "Not 'filtered_pi'" in d_stat: summary_counts['direct']["Filtered (Not 'filtered_pi')"] += 1
        elif "Too Short" in d_stat:
            summary_counts['direct']['Filtered (Too Short)'] += 1
            if len(detailed_examples['direct_filtered_short']) < 5: detailed_examples['direct_filtered_short'].append(f"{region_key}: {status_dict['direct_details']}")
        elif "Passed Filters" in d_stat: summary_counts['direct']['Found & Passed Filters'] += 1
        # Note: Coord/Group Error is harder to track back to a specific region if coords fail, so this count might be underrepresented here. It's captured in load_pi_data logs.

        i_stat = status_dict['inverted_status']
        if "Region Not Matched" in i_stat: summary_counts['inverted']['Region Not Matched'] += 1
        elif "Not 'filtered_pi'" in i_stat: summary_counts['inverted']["Filtered (Not 'filtered_pi')"] += 1
        elif "Too Short" in i_stat:
            summary_counts['inverted']['Filtered (Too Short)'] += 1
            if len(detailed_examples['inverted_filtered_short']) < 5: detailed_examples['inverted_filtered_short'].append(f"{region_key}: {status_dict['inverted_details']}")
        elif "Passed Filters" in i_stat: summary_counts['inverted']['Found & Passed Filters'] += 1

    # Adjust counts: Total regions = defined_region_count. Summing statuses should equal this.
    # If a region wasn't matched by any header, it remains 'Region Not Matched'.
    # We need to other categories don't double-count.
    # The logic above tries to capture the *reason* a haplotype matching the region might be missing.

    logger.info("\n--- DIAGNOSIS SUMMARY (Single-Event Regions) ---")
    logger.info(f"Total Defined Single-Event Regions: {defined_region_count}")
    logger.info(f"\nStatus for Direct Haplotype (per defined region):")
    for status, count in summary_counts['direct'].items(): logger.info(f"  {status}: {count}")
    logger.info(f"\nStatus for Inverted Haplotype (per defined region):")
    for status, count in summary_counts['inverted'].items(): logger.info(f"  {status}: {count}")

    logger.info(f"\n-> Expected 'Single-event Direct' count passing filters: {summary_counts['direct']['Found & Passed Filters']}")
    logger.info(f"-> Expected 'Single-event Inverted' count passing filters: {summary_counts['inverted']['Found & Passed Filters']}")

    if summary_counts['direct']['Filtered (Too Short)'] > 0 or summary_counts['inverted']['Filtered (Too Short)'] > 0:
        logger.info("\nExamples of sequences Filtered (Too Short):")
        if detailed_examples['direct_filtered_short']:
             logger.info("  Direct:")
             for ex in detailed_examples['direct_filtered_short']: logger.info(f"    {ex}")
        if detailed_examples['inverted_filtered_short']:
             logger.info("  Inverted:")
             for ex in detailed_examples['inverted_filtered_short']: logger.info(f"    {ex}")

    final_direct_count = summary_counts['direct']['Found & Passed Filters']
    final_inverted_count = summary_counts['inverted']['Found & Passed Filters']

    if final_direct_count != final_inverted_count:
        logger.warning("DIAGNOSIS: Counts for 'Found & Passed Filters' DIFFER between Direct and Inverted.")
        logger.warning("DIAGNOSIS: This suggests the filtering applied by `load_pi_data` (checking for 'filtered_pi', length, valid coords/group) affects the haplotypes differently.")
        logger.warning("DIAGNOSIS: Check the status counts above to see the primary reasons for discrepancy (e.g., more 'Too Short' sequences for one haplotype).")
    else:
        logger.info("DIAGNOSIS: Counts for 'Found & Passed Filters' MATCH.")
        logger.info("DIAGNOSIS: If final category counts in the main analysis differ, the issue might be later (e.g., NaN stats in `calculate_flanking_stats` - check main logs, or ambiguous/unknown overlap in `categorize_sequences`).")

    logger.info("--- ENDING DIAGNOSIS ---")


# --- Plotting ---

def format_p_value(p_value: float) -> str:
    """Formats p-value for display on plots."""
    if pd.isna(p_value):
        return "p = N/A"
    elif p_value < 0.001:
        return "p < 0.001"
    elif p_value < 1e-6:
         # Use scientific notation for very small p-values
         return f"p = {p_value:.2e}"
    else:
        # Use general format with reasonable precision for other cases
        return f"p = {p_value:.3g}"

def create_kde_plot(all_sequences_stats: List[Dict], test_results: Dict) -> Optional[plt.Figure]:
    stat_type = "mean"
    DIRECT_LINEWIDTH = 0.6
    DIRECT_EDGECOLOR = 'darkgrey'
    DIRECT_LINESTYLE = ':'
    INVERTED_LINEWIDTH = 1.0
    INVERTED_EDGECOLOR = 'black'
    INVERTED_LINESTYLE = '-'
    connecting_line_alpha = 0.25

    logger.info(f"Creating Overall Paired Violin Plot for {stat_type.capitalize()} Pi (Continuous L2FC, Manual Norm)...")
    start_time = time.time()

    flanking_field = f"flanking_{stat_type}"
    middle_field = f"middle_{stat_type}"

    plot_data = []
    paired_list = []

    sequences_with_inversion_info = 0
    sequences_missing_inversion_info = 0
    for i, s in enumerate(all_sequences_stats):
        f_val = s.get(flanking_field)
        m_val = s.get(middle_field)
        is_inverted = s.get('is_inverted')
        pair_id = f'pair_{i}'

        if is_inverted is None:
             logger.warning(f"Sequence at index {i} (Header: {s.get('header', 'Unknown')[:50]}...) is missing 'is_inverted' status. Will use Direct outline.")
             sequences_missing_inversion_info += 1
             is_inverted = False
        else:
             sequences_with_inversion_info += 1

        if pd.notna(f_val) and pd.notna(m_val):
            plot_data.append({'pair_id': pair_id, 'region_type': 'Flanking', 'pi_value': f_val, 'is_inverted': is_inverted})
            plot_data.append({'pair_id': pair_id, 'region_type': 'Middle', 'pi_value': m_val, 'is_inverted': is_inverted})
            paired_list.append({'pair_id': pair_id, 'Flanking': f_val, 'Middle': m_val})

    if sequences_missing_inversion_info > 0:
        logger.warning(f"Found {sequences_missing_inversion_info} sequences missing 'is_inverted' status out of {len(all_sequences_stats)}.")

    n_valid_pairs_plot = len(paired_list)

    overall_results = test_results.get('Overall', {})
    overall_p_value = overall_results.get('mean_p', np.nan)
    n_reported_test = overall_results.get('n_valid_pairs', n_valid_pairs_plot)

    if n_valid_pairs_plot < 2:
        logger.warning(f"Insufficient valid pairs ({n_valid_pairs_plot}) with non-NaN Flanking and Middle means. Skipping Overall plot.")
        return None

    df_long = pd.DataFrame(plot_data)
    df_paired = pd.DataFrame(paired_list)

    if df_long.empty:
        logger.warning("Plotting DataFrame 'df_long' is empty. Cannot create plot.")
        return None

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df_paired['Middle'] / df_paired['Flanking']
        df_paired['L2FC'] = np.log2(ratio)
        mask_to_replace = np.isneginf(df_paired['L2FC']) | pd.isna(df_paired['L2FC'])
        df_paired.loc[mask_to_replace, 'L2FC'] = np.nan

    l2fc_finite_values = df_paired['L2FC'].replace([np.inf, -np.inf], np.nan).dropna()
    can_draw_colorbar = not l2fc_finite_values.empty
    scalar_mappable = None
    manual_vmin = np.nan
    manual_vmax = np.nan

    if can_draw_colorbar:
        q_low = np.nanpercentile(l2fc_finite_values, 2)
        q_high = np.nanpercentile(l2fc_finite_values, 98)
        max_abs_l2fc = max(abs(q_low), abs(q_high))
        min_range_magnitude = 0.1
        if max_abs_l2fc < (min_range_magnitude / 2) or np.isclose(max_abs_l2fc, 0):
             manual_vmin = -min_range_magnitude / 2
             manual_vmax = min_range_magnitude / 2
        else:
            manual_vmin = -max_abs_l2fc
            manual_vmax = max_abs_l2fc
        norm = mcolors.Normalize(vmin=manual_vmin, vmax=manual_vmax, clip=True)
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=PLOT_COLORMAP)
        logger.info(f"Setting manual L2FC normalization range: [{manual_vmin:.3f}, {manual_vmax:.3f}]")
    else:
        logger.warning("No finite L2FC values found. Cannot create L2FC colorbar. Paired lines will use default color.")

    fig, ax = plt.subplots(figsize=(9, 7))

    region_palette = {'Flanking': FLANKING_COLOR, 'Middle': MIDDLE_COLOR}
    region_order = ['Flanking', 'Middle']
    x_coords_cat = {'Flanking': 0, 'Middle': 1}

    for _, row in df_paired.iterrows():
        l2fc_val = row['L2FC']
        x_flank = x_coords_cat['Flanking']
        x_middle = x_coords_cat['Middle']
        y_flank = row['Flanking']
        y_middle = row['Middle']
        if scalar_mappable is not None and pd.notna(l2fc_val) and np.isfinite(l2fc_val):
             line_color = scalar_mappable.to_rgba(l2fc_val)
        elif np.isinf(l2fc_val):
             line_color = DEFAULT_LINE_COLOR
        else:
             line_color = DEFAULT_LINE_COLOR
        ax.plot([x_flank, x_middle], [y_flank, y_middle],
                color=line_color, alpha=connecting_line_alpha, lw=LINE_WIDTH, zorder=1)

    violin_width = 0.8
    sns.violinplot(data=df_long, x='region_type', y='pi_value', order=region_order,
                   hue='region_type', palette=region_palette,
                   inner=None, linewidth=1.2, width=violin_width,
                   cut=0, density_norm='width', alpha=VIOLIN_ALPHA,
                   legend=False, ax=ax, zorder=10)

    common_stripplot_args = {
        'x': 'region_type', 'y': 'pi_value', 'order': region_order,
        'hue': 'region_type', 'palette': region_palette,
        'size': SCATTER_SIZE, 'alpha': SCATTER_ALPHA,
        'jitter': 0.15,
        'legend': False,
        'ax': ax, 'zorder': 11
    }

    df_direct = df_long[~df_long['is_inverted']]
    direct_plotted = False
    if not df_direct.empty:
        try:
             sns.stripplot(data=df_direct, **common_stripplot_args,
                           linewidth=DIRECT_LINEWIDTH,
                           edgecolor=DIRECT_EDGECOLOR,
                           marker='o'
                          )
             logger.info(f"Plotted {len(df_direct)} Direct points.")
             direct_plotted = True
        except Exception as e:
             logger.error(f"Error plotting Direct stripplot: {e}. Skipping Direct points.", exc_info=True)

    df_inverted = df_long[df_long['is_inverted']]
    inverted_plotted = False
    if not df_inverted.empty:
        try:
            sns.stripplot(data=df_inverted, **common_stripplot_args,
                          linewidth=INVERTED_LINEWIDTH,
                          edgecolor=INVERTED_EDGECOLOR,
                          marker='o',
                         )
            logger.info(f"Plotted {len(df_inverted)} Inverted points.")
            inverted_plotted = True
        except Exception as e:
             logger.error(f"Error plotting Inverted stripplot: {e}. Skipping Inverted points.", exc_info=True)

    median_values = df_long.groupby('region_type', observed=False)['pi_value'].median()
    median_line_width_on_plot = 0.15
    for region, median_val in median_values.items():
        x_center = x_coords_cat[region]
        xmin = x_center - median_line_width_on_plot / 2
        xmax = x_center + median_line_width_on_plot / 2
        ax.hlines(y=median_val, xmin=xmin, xmax=xmax,
                  color=MEDIAN_LINE_COLOR, linestyle='-', linewidth=MEDIAN_LINE_WIDTH,
                  zorder=12, alpha=0.8)

    colorbar_width_adjustment = 0.97
    if scalar_mappable is not None:
        cbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02, aspect=25, shrink=0.6)
        cbar.set_label('Log2 (π Middle / π Flanking)', rotation=270, labelpad=18, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)
        colorbar_width_adjustment = 0.90

    p_text = format_p_value(overall_p_value)
    mean_diff = np.nanmean(df_paired['Middle'] - df_paired['Flanking'])
    diff_text = f"Mean Diff (Middle - Flank): {mean_diff:.4g}"
    n_text = f"N = {n_reported_test} pairs"
    annotation_text = f"{n_text}\n{diff_text}\n{p_text} (Permutation Test)"
    ax.text(0.03, 0.97, annotation_text, transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='black',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8, ec='grey'))

    ax.set_ylabel(f'Mean Nucleotide Diversity (π)', fontsize=12)
    ax.set_xlabel('Region Type', fontsize=12)
    ax.set_xticks(list(x_coords_cat.values()))
    ax.set_xticklabels(list(x_coords_cat.keys()), fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(f'Overall Comparison of Mean π: Middle vs. Flanking Regions', fontsize=14, pad=20)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    ax.xaxis.grid(False)
    sns.despine(ax=ax, offset=5, trim=False)

    all_pi_values = df_long['pi_value'].dropna()
    if not all_pi_values.empty:
        min_val = all_pi_values.min()
        max_val = all_pi_values.max()
        y_range = max_val - min_val
        y_buffer = y_range * 0.05 if y_range > 0 else 0.1
        ax.set_ylim(bottom=max(0, min_val - y_buffer), top=max_val + y_buffer)
    else:
        ax.set_ylim(0, 1)

    legend_marker_fill = 'lightgrey'
    legend_elements = []
    if direct_plotted:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label='Direct',
                       markerfacecolor=legend_marker_fill, markersize=np.sqrt(30),
                       linestyle='None',
                       markeredgewidth=DIRECT_LINEWIDTH, markeredgecolor=DIRECT_EDGECOLOR)
        )
    if inverted_plotted:
         legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label='Inverted',
                       markerfacecolor=legend_marker_fill, markersize=np.sqrt(30),
                       linestyle='None',
                       markeredgewidth=INVERTED_LINEWIDTH, markeredgecolor=INVERTED_EDGECOLOR)
        )

    if legend_elements:
        ax.legend(handles=legend_elements, title="Sequence Type", fontsize=9, title_fontsize=10,
                  loc='upper right', bbox_to_anchor=(1.15, 1.01),
                  frameon=True, framealpha=0.9)
        colorbar_width_adjustment = min(colorbar_width_adjustment, 0.85)

    try:
        fig.tight_layout(rect=[0.03, 0.03, colorbar_width_adjustment, 0.93])
    except Exception as e:
        logger.error(f"Error during tight_layout adjustment: {e}", exc_info=True)

    plot_filename = OUTPUT_DIR / f"pi_overall_{stat_type}_violin_paired_L2FC_inversion_outline.png"
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved overall styled paired violin plot to {plot_filename}")
    except ValueError as ve:
         logger.error(f"ValueError saving plot: {ve}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to save styled violin plot to {plot_filename}: {e}", exc_info=True)

    elapsed_time = time.time() - start_time
    logger.info(f"Created and saved styled violin plot in {elapsed_time:.2f} seconds.")

    return fig


# --- Main Execution ---

def main():
    total_start_time = time.time()
    logger.info("--- Starting Pi Flanking Regions Analysis ---")
    logger.info("--- Mode: Filtered Pi & Length Only ---")

    # --- Load Inversion Info ---
    inv_file_path = Path(INVERSION_FILE)
    if not inv_file_path.is_file():
        logger.error(f"Inversion info file not found: {inv_file_path}. Cannot proceed with categorization.")
        return
    logger.info(f"Loading inversion info from {inv_file_path}")
    try:
        inversion_df = pd.read_csv(inv_file_path, sep='\t')
        logger.info(f"Loaded {inversion_df.shape[0]} rows from inversion file.")
        recurrent_regions, single_event_regions = map_regions_to_inversions(inversion_df)
    except Exception as e:
        logger.error(f"Failed to load or process inversion file {inv_file_path}: {e}", exc_info=True)
        return

    # --- Load Pi Data (with Filtering) ---
    pi_file_path = Path(PI_DATA_FILE)
    if not pi_file_path.is_file():
         logger.error(f"Pi data file not found: {pi_file_path}. Cannot run analysis.")
         return

    # Optional: Run diagnosis before main loading if needed
    if single_event_regions:
         diagnose_single_event_discrepancy(
             pi_file_path=pi_file_path,
             single_event_regions=single_event_regions,
             min_length=MIN_LENGTH,
             inv_info_path=inv_file_path
         )
    else:
         logger.warning("Skipping single-event discrepancy diagnosis as no single-event regions were loaded/mapped.")

    pi_sequences = load_pi_data(pi_file_path)
    if not pi_sequences:
        logger.error("No valid sequences loaded after filtering ('filtered_pi', length, coords, group). Exiting.")
        return

    # --- Calculate Flanking Stats ---
    flanking_stats = calculate_flanking_stats(pi_sequences)
    if not flanking_stats:
        logger.error("No sequences remained after calculating flanking statistics (check logs for NaN/length issues). Exiting.")
        return

    # --- REMOVED FILTERS ---
    # The following filters are EXPLICITLY REMOVED based on the request:
    # - filter_sequences_by_region_completeness
    # - find_and_filter_strict_pairs
    # The analysis proceeds with all sequences that passed `load_pi_data` and `calculate_flanking_stats`.
    logger.info("Skipping Haplotype Completeness and Strict Pairing filters as requested.")
    sequences_for_analysis = flanking_stats # Use the direct output

    # --- Categorize Sequences ---
    categories = categorize_sequences(sequences_for_analysis, recurrent_regions, single_event_regions)

    # --- Perform Statistical Tests ---
    # Pass the full set of sequences for 'Overall' comparison
    test_results = perform_statistical_tests(categories, sequences_for_analysis)

    # --- Generate Plot ---
    fig_mean = create_kde_plot(sequences_for_analysis, test_results)

    # --- Generate Summary ---
    logger.info("\n--- Analysis Summary (Filtered Pi & Length Only) ---")
    logger.info(f"Input Pi File: {PI_DATA_FILE}")
    logger.info(f"Input Inversion File: {INVERSION_FILE}")
    logger.info(f"Filters Applied: Header contains 'filtered_pi', Min Length >= {MIN_LENGTH}, Valid Coords/Group, Calculable Stats (Flanks/Middle)")
    logger.info(f"Total Sequences Used in Final Analysis: {len(sequences_for_analysis)}")

    logger.info("\nPaired Test Results (Middle vs Flanking - mean difference):")
    logger.info("-" * 85) # Adjusted width
    logger.info(f"{'Category':<25} {'N Valid Pairs':<15} {'Mean Diff (M-F)':<18} {'Permutation p':<15} {'Normality p (Diffs)':<15}")
    logger.info("-" * 85)

    summary_data = []
    # Iterate through categories in the defined order + Overall
    for cat in CATEGORY_ORDER_WITH_OVERALL:
        results_for_cat = test_results.get(cat, {})
        n_valid_pairs = results_for_cat.get('n_valid_pairs', 0)

        # Calculate mean difference from the input data for this category
        mean_diff = np.nan
        if cat == 'Overall':
            seq_list = sequences_for_analysis
        else:
            internal_cat = CAT_MAPPING.get(cat)
            seq_list = categories.get(internal_cat, []) if internal_cat else []

        if seq_list and n_valid_pairs > 0: # there are pairs to calculate diff from
            m_means = np.array([s['middle_mean'] for s in seq_list])
            f_means = np.array([s['flanking_mean'] for s in seq_list])
            valid_indices = ~np.isnan(m_means) & ~np.isnan(f_means)
            # Recalculate diff using only the valid pairs identified by the test function
            if np.sum(valid_indices) == n_valid_pairs: # Safety check
                 mean_diff = np.mean(m_means[valid_indices] - f_means[valid_indices])
            else:
                 logger.warning(f"Mismatch in valid pair count for {cat} between summary ({np.sum(valid_indices)}) and test ({n_valid_pairs}). Using test count.")
                 # Attempt calculation anyway if possible
                 if np.sum(valid_indices) > 0:
                     mean_diff = np.mean(m_means[valid_indices] - f_means[valid_indices])


        mean_p = results_for_cat.get('mean_p', np.nan)
        norm_p = results_for_cat.get('mean_normality_p', np.nan)

        # Format for nice output
        n_str = str(n_valid_pairs)
        mean_diff_str = f"{mean_diff:.4g}" if pd.notna(mean_diff) else "N/A"
        mean_p_str = format_p_value(mean_p)
        norm_p_str = f"{norm_p:.3g}" if pd.notna(norm_p) else ("N/A" if n_valid_pairs < 3 else "Const") # Indicate if const or N<3

        logger.info(f"{cat:<25} {n_str:<15} {mean_diff_str:<18} {mean_p_str:<15} {norm_p_str:<15}")
        summary_data.append({
            'Category': cat,
            'N_Valid_Pairs': n_valid_pairs,
            'Mean_Difference_Middle_Minus_Flanking': mean_diff,
            'Mean_Permutation_p_value': mean_p,
            'Mean_Diff_Normality_p_value': norm_p
         })

    logger.info("-" * 85)

    # --- Calculate and Print Overall Fold Difference ---
    logger.info("\n--- Overall Fold Difference (Mean Pi) ---")
    # Extract all flanking and middle means from the sequences used in the final analysis
    all_flanking_means = np.array([s.get('flanking_mean', np.nan) for s in sequences_for_analysis], dtype=float)
    all_middle_means = np.array([s.get('middle_mean', np.nan) for s in sequences_for_analysis], dtype=float)

    # Calculate the overall aggregate means (ignoring NaNs)
    overall_agg_mean_flank = np.nanmean(all_flanking_means)
    overall_agg_mean_middle = np.nanmean(all_middle_means)

    # Calculate Fold Difference (Middle / Flanking), handling division by zero/NaN
    overall_fc_mean = np.nan
    if pd.notna(overall_agg_mean_flank) and not np.isclose(overall_agg_mean_flank, 0):
        overall_fc_mean = overall_agg_mean_middle / overall_agg_mean_flank
    elif pd.notna(overall_agg_mean_middle) and pd.notna(overall_agg_mean_flank) and np.isclose(overall_agg_mean_middle, 0) and np.isclose(overall_agg_mean_flank, 0):
         overall_fc_mean = np.nan # 0/0 case - Undefined
    elif pd.notna(overall_agg_mean_middle) and not np.isclose(overall_agg_mean_middle, 0) and pd.notna(overall_agg_mean_flank) and np.isclose(overall_agg_mean_flank, 0):
         overall_fc_mean = np.inf # non-zero / zero case - Infinity

    # Format and print the result
    fc_mean_str = f"{overall_fc_mean:.4g}" if pd.notna(overall_fc_mean) and np.isfinite(overall_fc_mean) else ('Inf' if overall_fc_mean == np.inf else 'N/A')
    logger.info(f"Overall Fold Difference (Mean Middle π / Mean Flanking π): {fc_mean_str}")
    logger.info(f"(Based on Overall Mean Flank π: {overall_agg_mean_flank:.4g}, Overall Mean Middle π: {overall_agg_mean_middle:.4g})")
    # ----------------------------------------------------

    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = OUTPUT_DIR / "pi_analysis_mean_summary_filtered_pi_length.csv" #  filename
    try:
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.5g')
        logger.info(f"Analysis summary saved to {summary_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV to {summary_csv_path}: {e}")

    total_elapsed_time = time.time() - total_start_time
    logger.info(f"--- Analysis finished in {total_elapsed_time:.2f} seconds ---")

    # Close plot figure if it was created
    if fig_mean:
        plt.close(fig_mean)

if __name__ == "__main__":
    main()
