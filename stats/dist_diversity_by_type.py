"""
Analyzes nucleotide diversity (pi) in flanking regions vs. middle regions
of genomic segments, categorized by inversion type and haplotype status,
focusing on mean values and ensuring robust statistical testing.

Handles potentially sparse pi data (many zeros).
"""
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
logger = logging.getLogger('pi_flanking_analysis')

# Constants
MIN_LENGTH = 150_000  # Minimum sequence length for analysis
FLANK_SIZE = 50_000  # Size of flanking regions (each end)
PERMUTATIONS = 10000 # Number of permutations for significance testing
BAR_ALPHA = 0.3      # Transparency for plot bars

# File paths (Update these paths if necessary)
PI_DATA_FILE = 'per_site_output.falsta'
INVERSION_FILE = 'inv_info.csv'
OUTPUT_DIR = Path('pi_analysis_results')

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
plt.style.use('seaborn-v0_8-whitegrid')
# Using Tableau 10 color scheme for better distinction
COLOR_PALETTE = plt.cm.tab10.colors
FLANKING_COLOR = COLOR_PALETTE[0] # Blue
MIDDLE_COLOR = COLOR_PALETTE[1]  # Orange
SCATTER_ALPHA = 0.6
SCATTER_SIZE = 25

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
    """
    # Regex to capture chromosome, start, end, and  group
    # Allows 'chr' or 'chr_' prefix for chromosome part
    pattern = re.compile(
        r'>.*?_chr_?([\w\.\-]+)_start_(\d+)_end_(\d+)(?:_group_([01]))?', # Allow more chars in chrom name
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
        logger.warning(f"Failed to extract coordinates using regex from header: {header[:70]}...")
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

    # correct data types, coercing errors to NaT/NaN
    inversion_df['Start'] = pd.to_numeric(inversion_df['Start'], errors='coerce')
    inversion_df['End'] = pd.to_numeric(inversion_df['End'], errors='coerce')
    inversion_df['0_single_1_recur'] = pd.to_numeric(inversion_df['0_single_1_recur'], errors='coerce')
    # Chromosome is string for normalization
    inversion_df['Chromosome'] = inversion_df['Chromosome'].astype(str)


    # Drop rows where essential information is missing
    original_rows = len(inversion_df)
    inversion_df = inversion_df.dropna(subset=['Chromosome', 'Start', 'End', '0_single_1_recur'])
    dropped_rows = original_rows - len(inversion_df)
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows from inversion info due to missing values.")

    for _, row in inversion_df.iterrows():
        chrom = normalize_chromosome(row['Chromosome'])
        if chrom is None:
            continue # Skip if chromosome name is invalid

        start = int(row['Start']) # Already  numeric and not NaN
        end = int(row['End'])
        is_recurrent = int(row['0_single_1_recur']) == 1

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
    """Check if two genomic regions overlap (inclusive coordinates)."""
    # Assumes coordinates are 1-based inclusive, typical in biology
    # Overlap exists if one region starts before the other ends, and vice versa
    return region1_start <= region2_end and region1_end >= region2_start

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
        logger.warning(f"ValueError parsing data line value '{val_str}': {e}. Skipping line: {line[:50]}...")
        return None

def load_pi_data(file_path: str | Path) -> list[dict]:
    """
    Load filtered pi data from a FASTA-like file.
    Assumes header lines start with '>' and data follows on the next line(s).
    Handles multi-line sequences by concatenating.
    Filters sequences based on MIN_LENGTH.
    Extracts coordinates and group info from headers containing 'filtered_pi'.
    """
    logger.info(f"Loading pi data from {file_path}")
    start_time = time.time()

    pi_sequences = []
    sequences_processed = 0
    headers_read = 0
    skipped_short = 0
    skipped_not_filtered = 0
    skipped_coord_error = 0
    skipped_data_error = 0

    current_header = None
    current_sequence_parts = []

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                if line.startswith('>'):
                    headers_read += 1
                    # --- Process the *previous* sequence if one was being read ---
                    if current_header and current_sequence_parts:
                        sequences_processed += 1
                        full_sequence_line = "".join(current_sequence_parts)
                        pi_data = parse_pi_data_line(full_sequence_line)

                        if pi_data is not None:
                            actual_length = len(pi_data)
                            if actual_length >= MIN_LENGTH:
                                coords = extract_coordinates_from_header(current_header)
                                if coords and coords.get('group') is not None: # group info exists
                                    # Determine haplotype status (group 1 = inverted)
                                    is_inverted = coords['group'] == 1

                                    pi_sequences.append({
                                        'header': current_header,
                                        'coords': coords,
                                        'data': pi_data,
                                        'length': actual_length,
                                        'is_inverted': is_inverted
                                    })
                                else:
                                    # Log reason for skipping (coord error or missing group)
                                    reason = "coordinate/group extraction failed" if not coords or coords.get('group') is None else "coordinate extraction failed"
                                    logger.warning(f"Skipping sequence from header '{current_header[:70]}...' ({reason})")
                                    skipped_coord_error += 1
                            else:
                                skipped_short += 1
                        else:
                            skipped_data_error +=1 # Parsing the data string failed
                        # Reset for next sequence
                        current_header = None
                        current_sequence_parts = []
                    # --- Start processing the *new* sequence header ---
                    if 'filtered_pi' in line.lower():
                        current_header = line
                        current_sequence_parts = [] # Reset sequence parts for the new header
                    else:
                        skipped_not_filtered += 1
                        current_header = None # Ignore sequences not marked as filtered_pi

                elif current_header:
                    # Append data line to current sequence parts (handles multi-line)
                    # Only append if we have a valid header we care about
                    current_sequence_parts.append(line)

            # --- Process the very last sequence in the file after loop ends ---
            if current_header and current_sequence_parts:
                sequences_processed += 1
                full_sequence_line = "".join(current_sequence_parts)
                pi_data = parse_pi_data_line(full_sequence_line)

                if pi_data is not None:
                    actual_length = len(pi_data)
                    if actual_length >= MIN_LENGTH:
                        coords = extract_coordinates_from_header(current_header)
                        if coords and coords.get('group') is not None:
                            is_inverted = coords['group'] == 1
                            pi_sequences.append({
                                'header': current_header,
                                'coords': coords,
                                'data': pi_data,
                                'length': actual_length,
                                'is_inverted': is_inverted
                            })
                        else:
                            reason = "coordinate/group extraction failed" if not coords or coords.get('group') is None else "coordinate extraction failed"
                            logger.warning(f"Skipping last sequence from header '{current_header[:70]}...' ({reason})")
                            skipped_coord_error += 1
                    else:
                        skipped_short += 1
                else:
                    skipped_data_error += 1

    except FileNotFoundError:
        logger.error(f"Fatal Error: Pi data file not found at {file_path}")
        return [] # Return empty list if file not found
    except Exception as e: # Catch other potential file reading errors
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}", exc_info=True)
        return []

    elapsed_time = time.time() - start_time
    logger.info(f"Read {headers_read} headers, processed {sequences_processed} potential sequences in {elapsed_time:.2f} seconds.")
    logger.info(f"Loaded {len(pi_sequences)} valid 'filtered_pi' sequences with length >= {MIN_LENGTH} and group info.")
    logger.info(f"Skipped: {skipped_short} (too short), {skipped_not_filtered} (not 'filtered_pi'), "
                f"{skipped_coord_error} (coord/group error), {skipped_data_error} (data parse error).")

    if pi_sequences:
         # Log chromosome distribution
        chrom_counts = {}
        for seq in pi_sequences:
            chrom = seq['coords'].get('chrom', 'Unknown')
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
        logger.info(f"Chromosome distribution of loaded sequences: {chrom_counts}")
    else:
        logger.warning("No valid pi sequences were loaded that met all criteria.")

    return pi_sequences

def calculate_flanking_stats(pi_sequences: list[dict]) -> list[dict]:
    """
    Calculate mean and median pi for beginning, middle, and ending regions.
    Handles NaN values within regions. Returns list of dicts with stats.
    """
    logger.info(f"Calculating flanking and middle statistics for {len(pi_sequences)} sequences...")
    start_time = time.time()
    results = []
    skipped_too_short_for_flanks = 0

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
            'beginning_mean': np.nanmean(beginning_flank),
            'ending_mean': np.nanmean(ending_flank),
            'middle_mean': np.nanmean(middle_region),
            'beginning_median': np.nanmedian(beginning_flank),
            'ending_median': np.nanmedian(ending_flank),
            'middle_median': np.nanmedian(middle_region)
        }

        # Calculate combined flanking stats AFTER individual stats are computed
        # This prevents issues if one flank is all NaN
        b_mean, e_mean = stats['beginning_mean'], stats['ending_mean']
        stats['flanking_mean'] = np.nanmean([b_mean, e_mean]) # nanmean handles if one is NaN

        b_med, e_med = stats['beginning_median'], stats['ending_median']
        stats['flanking_median'] = np.nanmean([b_med, e_med]) # Use nanmean of medians as combined measure

        # Check if any critical calculation resulted in NaN (e.g., all NaNs in middle region)
        # Flanking mean/median might be valid even if one flank is NaN, but middle must be valid.
        if np.isnan(stats['middle_mean']) or np.isnan(stats['middle_median']):
             logger.warning(f"Sequence {i} ({seq_id}...) middle region resulted in NaN stats. Skipping this sequence.")
             # skip sequences where essential stats are NaN
             continue
        # Also check if BOTH flanking regions resulted in NaN
        if np.isnan(stats['flanking_mean']) or np.isnan(stats['flanking_median']):
            logger.warning(f"Sequence {i} ({seq_id}...) both flanking regions resulted in NaN stats. Skipping this sequence.")
            continue


        results.append(stats)

    elapsed_time = time.time() - start_time
    if skipped_too_short_for_flanks > 0:
        logger.warning(f"Skipped {skipped_too_short_for_flanks} sequences too short (< {min_req_len} bp) for flank analysis.")
    logger.info(f"Successfully calculated statistics for {len(results)} sequences in {elapsed_time:.2f} seconds.")
    return results

def categorize_sequences(flanking_stats: list[dict], recurrent_regions: dict, single_event_regions: dict) -> dict:
    """
    Categorize sequences based on inversion type and haplotype status.
    Returns a dictionary where keys are internal category names and values are lists of sequence stat dicts.
    """
    logger.info("Categorizing sequences...")
    start_time = time.time()
    # Initialize categories using the defined order
    categories = {CAT_MAPPING[cat_name]: [] for cat_name in CATEGORY_ORDER}
    ambiguous_count = 0
    unknown_count = 0
    sequences_without_coords = 0 # Should be low if load_pi_data works

    for seq_stats in flanking_stats:
        coords = seq_stats.get('coords')
        is_inverted = seq_stats.get('is_inverted') # Should always exist if it passed calculate_stats

        if coords is None or is_inverted is None:
             sequences_without_coords += 1 # Should not happen often
             continue

        inversion_type = determine_inversion_type(coords, recurrent_regions, single_event_regions)

        category_key = None
        if inversion_type == 'recurrent':
            category_key = 'recurrent_inverted' if is_inverted else 'recurrent_direct'
        elif inversion_type == 'single_event':
            category_key = 'single_event_inverted' if is_inverted else 'single_event_direct'
        elif inversion_type == 'ambiguous':
            ambiguous_count += 1
        else: # 'unknown'
            unknown_count += 1

        if category_key and category_key in categories:
            categories[category_key].append(seq_stats)

    elapsed_time = time.time() - start_time
    logger.info(f"Finished categorization in {elapsed_time:.2f} seconds.")
    logger.info("Category counts:")
    for display_cat in CATEGORY_ORDER:
        internal_cat = CAT_MAPPING[display_cat]
        logger.info(f"  {display_cat}: {len(categories[internal_cat])}")
    logger.info(f"  Skipped (Ambiguous overlap): {ambiguous_count}")
    logger.info(f"  Skipped (Unknown overlap / Not in inv file): {unknown_count}")
    if sequences_without_coords > 0:
        logger.warning(f"  Skipped due to missing coords/group in stats data: {sequences_without_coords}")

    # Check for empty categories
    for display_cat in CATEGORY_ORDER:
        internal_cat = CAT_MAPPING[display_cat]
        if not categories[internal_cat]:
            logger.warning(f"Category '{display_cat}' is empty.")

    return categories

# --- Statistical Testing ---

def perform_statistical_tests(categories: dict, all_sequences_stats: list[dict]) -> dict:
    """
    Performs paired permutation tests and Shapiro-Wilk normality tests
    for mean differences between flanking and middle regions.

    Returns:
        dict: Nested dictionary with p-values
              {'CategoryName': {'mean_p': float, 'mean_normality_p': float}}.
    """
    logger.info("Performing statistical tests (Middle vs Flanking - mean)...")
    test_results = {}

    # Combine specific categories and 'Overall' for iteration
    category_data_map = {REVERSE_CAT_MAPPING[k]: v for k, v in categories.items()}
    category_data_map['Overall'] = all_sequences_stats

    for category_name, seq_list in category_data_map.items():
        logger.info(f"  Testing category: {category_name} ({len(seq_list)} sequences)")
        test_results[category_name] = {'mean_p': np.nan, 'mean_normality_p': np.nan}

        if len(seq_list) < 2: # Need at least 2 sequences
            logger.warning(f"    Skipping tests for {category_name}: < 2 sequences.")
            continue

        # Extract paired mean values, handling potential NaNs from calculations
        flanking_means = np.array([s['flanking_mean'] for s in seq_list], dtype=float)
        middle_means = np.array([s['middle_mean'] for s in seq_list], dtype=float)

        # Perform tests only if there are enough valid pairs
        valid_indices = ~np.isnan(flanking_means) & ~np.isnan(middle_means)
        num_valid_pairs = np.sum(valid_indices)

        if num_valid_pairs >= 2:
             # --- Permutation Test (mean) ---
             # Test: Middle vs Flanking (mean differences)
             mean_perm_p = paired_permutation_test(
                 middle_means[valid_indices], flanking_means[valid_indices], use_median=False
             )
             test_results[category_name]['mean_p'] = mean_perm_p
             logger.info(f"    Permutation test (mean): p = {mean_perm_p:.4g} ({num_valid_pairs} valid pairs)")

             # --- Normality Test (mean Differences) ---
             if num_valid_pairs >= 3: # Shapiro-Wilk requires at least 3 samples
                 differences = middle_means[valid_indices] - flanking_means[valid_indices]
                 try:
                     shapiro_stat, shapiro_p = shapiro(differences)
                     test_results[category_name]['mean_normality_p'] = shapiro_p
                     # Log normality result clearly
                     logger.info(f"    Normality test (mean Diffs): Shapiro-Wilk W={shapiro_stat:.4f}, p={shapiro_p:.4g}")
                 except ValueError as e:
                     # Handle cases like all differences being identical
                     logger.warning(f"    Could not perform normality test for {category_name}: {e}")
             else:
                 logger.info(f"    Skipping normality test for {category_name}: < 3 valid pairs ({num_valid_pairs}).")

        else:
            logger.warning(f"    Skipping tests for {category_name}: < 2 valid pairs ({num_valid_pairs}).")

    return test_results

def diagnose_single_event_discrepancy(
    pi_file_path: str | Path,
    single_event_regions: dict,
    min_length: int,
    inv_info_path: str | Path # Added for context
    ) -> None:
    """
    Diagnoses why counts for single-event direct vs inverted might differ.

    Checks each defined single-event region against the pi data headers
    to see if direct/inverted sequences are found and why they might be
    filtered out *before* categorization (Not Found, Too Short, Not 'filtered_pi').

    Args:
        pi_file_path: Path to the per_site_output.falsta file.
        single_event_regions: Dict mapping {chrom: [(start, end), ...]} for SINGLE events.
        min_length: The minimum sequence length threshold used in load_pi_data.
        inv_info_path: Path to the inversion info file (for logging context).
    """
    logger.info("--- STARTING DIAGNOSIS: Single-Event Discrepancy ---")
    logger.info(f"Analyzing Pi file: {pi_file_path}")
    logger.info(f"Based on Single-Event regions from: {inv_info_path}")
    logger.info(f"Minimum Length Threshold: {min_length}")

    if not single_event_regions:
        logger.warning("DIAGNOSIS: No single-event regions were defined. Skipping diagnosis.")
        logger.info("--- ENDING DIAGNOSIS ---")
        return

    # Initialize tracking for each defined single-event region
    # Key: tuple (chrom, start, end)
    # Value: dict {'direct_status': 'Not Found', 'inverted_status': 'Not Found',
    #             'direct_details': '', 'inverted_details': ''}
    region_status = {}
    defined_region_count = 0
    for chrom, regions in single_event_regions.items():
        for start, end in regions:
            region_key = (chrom, start, end)
            region_status[region_key] = {
                'direct_status': 'Not Found', 'inverted_status': 'Not Found',
                'direct_details': '', 'inverted_details': ''
            }
            defined_region_count += 1
    logger.info(f"Tracking {defined_region_count} defined single-event regions.")

    processed_headers = 0
    potential_matches = 0
    try:
        with open(pi_file_path, 'r') as f:
            current_header = None
            line_num = 0
            while True: # Read header and associated data line
                line = f.readline()
                line_num += 1
                if not line: # End of file
                     break
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    current_header = line
                    processed_headers += 1

                    # Now, attempt to parse this header and see if it *could* match a single-event region
                    coords = extract_coordinates_from_header(current_header)
                    if not coords or coords.get('group') is None:
                        # Cannot evaluate this header for matching regions if coords/group invalid
                        continue

                    header_chrom = coords['chrom']
                    header_start = coords['start']
                    header_end = coords['end']
                    header_group = coords['group'] # 0 or 1

                    # Read the *next* line to determine length (simplification: assumes data on one line)
                    data_line = f.readline()
                    line_num += 1
                    if not data_line: # Unexpected EOF after header
                         logger.warning(f"DIAGNOSIS: Unexpected EOF after header: {current_header[:70]}...")
                         continue
                    data_line = data_line.strip()
                    try:
                        # Mimic part of parse_pi_data_line for length check
                        values = data_line.split(',')
                        # Count actual numeric values, excluding empty strings or 'NA'
                        actual_length = sum(1 for x in values if x.strip() and x.strip().upper() != 'NA')
                    except Exception as e:
                        logger.warning(f"DIAGNOSIS: Error parsing data line for length check below header {current_header[:70]}... Error: {e}")
                        actual_length = 0 # Cannot determine length

                    # Check if this header corresponds to any defined single-event region
                    matched = False
                    if header_chrom in single_event_regions:
                        for region_start, region_end in single_event_regions[header_chrom]:
                            if is_overlapping(header_start, header_end, region_start, region_end):
                                potential_matches += 1
                                matched = True
                                region_key = (header_chrom, region_start, region_end)

                                # Determine the status based on filters applied in load_pi_data
                                status = "Error (Should not happen)"
                                details = f"Header: {current_header[:70]}... Len={actual_length}"
                                if 'filtered_pi' not in current_header.lower():
                                    status = "Filtered (Not 'filtered_pi')"
                                elif actual_length < min_length:
                                    status = f"Filtered (Too Short: {actual_length})"
                                else:
                                    status = "Found & Passed Filters" # Passed primary filters relevant here

                                # Update the status for the matched DEFINED region
                                target_status_key = 'direct_status' if header_group == 0 else 'inverted_status'
                                target_details_key = 'direct_details' if header_group == 0 else 'inverted_details'

                                # Only update if 'Not Found' or if a new reason is found (e.g., found header but filtered)
                                # We prioritize 'Found & Passed Filters' if multiple headers match a region
                                current_region_s = region_status[region_key][target_status_key]
                                if current_region_s == 'Not Found' or status == "Found & Passed Filters":
                                     region_status[region_key][target_status_key] = status
                                     region_status[region_key][target_details_key] = details
                                # Important: If multiple headers overlap the same defined region, this logic
                                # might slightly simplify, but focuses on whether *at least one* passed filters.

                                break # Assume header maps to only one defined region for simplicity

    except FileNotFoundError:
        logger.error(f"DIAGNOSIS: Pi data file not found at {pi_file_path}")
        logger.info("--- ENDING DIAGNOSIS ---")
        return
    except Exception as e:
        logger.error(f"DIAGNOSIS: An error occurred reading {pi_file_path}: {e}", exc_info=True)
        logger.info("--- ENDING DIAGNOSIS ---")
        return

    logger.info(f"Processed {processed_headers} headers from Pi file.")
    logger.info(f"Found {potential_matches} headers potentially overlapping defined single-event regions.")

    # Summarize the findings
    summary_counts = {
        'direct': {'Not Found': 0, "Filtered (Not 'filtered_pi')": 0, 'Filtered (Too Short)': 0, 'Found & Passed Filters': 0, 'Other Filtered': 0},
        'inverted': {'Not Found': 0, "Filtered (Not 'filtered_pi')": 0, 'Filtered (Too Short)': 0, 'Found & Passed Filters': 0, 'Other Filtered': 0}
    }
    detailed_examples = {'direct_filtered_short': [], 'inverted_filtered_short': []}

    for region_key, status_dict in region_status.items():
        # Direct status
        d_stat = status_dict['direct_status']
        d_details = status_dict['direct_details']
        if d_stat == 'Not Found':
            summary_counts['direct']['Not Found'] += 1
        elif "Filtered (Not 'filtered_pi')" in d_stat:
             summary_counts['direct']["Filtered (Not 'filtered_pi')"] += 1
        elif "Filtered (Too Short" in d_stat:
             summary_counts['direct']['Filtered (Too Short)'] += 1
             if len(detailed_examples['direct_filtered_short']) < 5: # Log few examples
                 detailed_examples['direct_filtered_short'].append(f"{region_key}: {d_details}")
        elif d_stat == "Found & Passed Filters":
             summary_counts['direct']['Found & Passed Filters'] += 1
        else: # Catch any other filtered status if logic expands
             summary_counts['direct']['Other Filtered'] += 1

        # Inverted status
        i_stat = status_dict['inverted_status']
        i_details = status_dict['inverted_details']
        if i_stat == 'Not Found':
            summary_counts['inverted']['Not Found'] += 1
        elif "Filtered (Not 'filtered_pi')" in i_stat:
             summary_counts['inverted']["Filtered (Not 'filtered_pi')"] += 1
        elif "Filtered (Too Short" in i_stat:
             summary_counts['inverted']['Filtered (Too Short)'] += 1
             if len(detailed_examples['inverted_filtered_short']) < 5: # Log few examples
                 detailed_examples['inverted_filtered_short'].append(f"{region_key}: {i_details}")
        elif i_stat == "Found & Passed Filters":
             summary_counts['inverted']['Found & Passed Filters'] += 1
        else: # Catch any other filtered status
             summary_counts['inverted']['Other Filtered'] += 1

    logger.info("\n--- DIAGNOSIS SUMMARY ---")
    logger.info(f"Total Defined Single-Event Regions: {defined_region_count}")
    logger.info("\nStatus Counts (Direct Haplotype):")
    for status, count in summary_counts['direct'].items():
        logger.info(f"  {status}: {count}")
    logger.info("\nStatus Counts (Inverted Haplotype):")
    for status, count in summary_counts['inverted'].items():
        logger.info(f"  {status}: {count}")

    logger.info(f"\nExpected 'Single-event Direct' count (Found & Passed Filters): {summary_counts['direct']['Found & Passed Filters']}")
    logger.info(f"Expected 'Single-event Inverted' count (Found & Passed Filters): {summary_counts['inverted']['Found & Passed Filters']}")

    if summary_counts['direct']['Filtered (Too Short)'] > 0 or summary_counts['inverted']['Filtered (Too Short)'] > 0:
        logger.info("\nExamples of sequences Filtered (Too Short):")
        if detailed_examples['direct_filtered_short']:
             logger.info("  Direct:")
             for ex in detailed_examples['direct_filtered_short']: logger.info(f"    {ex}")
        if detailed_examples['inverted_filtered_short']:
             logger.info("  Inverted:")
             for ex in detailed_examples['inverted_filtered_short']: logger.info(f"    {ex}")

    if summary_counts['direct']['Found & Passed Filters'] != summary_counts['inverted']['Found & Passed Filters']:
        logger.warning("DIAGNOSIS: Counts for 'Found & Passed Filters' DIFFER between Direct and Inverted.")
        logger.warning("DIAGNOSIS: Primary reasons likely: 'Not Found' in Pi file OR 'Filtered (Too Short)' OR 'Filtered (Not 'filtered_pi')'. Check counts above.")
    else:
        logger.info("DIAGNOSIS: Counts for 'Found & Passed Filters' MATCH. If final category counts differ, the issue might be later (e.g., NaN stats - check main logs) or in categorization logic itself.")

    logger.info("--- ENDING DIAGNOSIS ---")

logger = logging.getLogger('pi_flanking_analysis')
OUTPUT_DIR = Path('pi_analysis_results')
plt.style.use('seaborn-v0_8-ticks')
COLOR_PALETTE = plt.cm.tab10.colors
FLANKING_COLOR = COLOR_PALETTE[0]
MIDDLE_COLOR = COLOR_PALETTE[1]
SCATTER_ALPHA = 0.35
SCATTER_SIZE = 3.0
LINE_ALPHA = 0.15
LINE_WIDTH = 0.9
VIOLIN_ALPHA = 0.2
MEDIAN_LINE_COLOR = 'k'
MEDIAN_LINE_WIDTH = 1.5
DEFAULT_LINE_COLOR = 'grey'
PLOT_COLORMAP = cm.coolwarm

def create_kde_plot(all_sequences_stats: List[Dict], test_results: Dict) -> Optional[plt.Figure]:
    stat_type = "mean"
    logger.info(f"Creating Overall Paired Violin Plot for {stat_type.capitalize()} Pi...")
    start_time = time.time()

    flanking_field = f"flanking_{stat_type}"
    middle_field = f"middle_{stat_type}"

    plot_data = []
    paired_list = []

    for i, s in enumerate(all_sequences_stats):
        f_val = s.get(flanking_field)
        m_val = s.get(middle_field)
        pair_id = f'pair_{i}'

        if pd.notna(f_val) and pd.notna(m_val):
            plot_data.append({'pair_id': pair_id, 'region_type': 'Flanking', 'pi_value': f_val})
            plot_data.append({'pair_id': pair_id, 'region_type': 'Middle', 'pi_value': m_val})
            paired_list.append({'pair_id': pair_id, 'Flanking': f_val, 'Middle': m_val})

    n_valid_pairs = len(paired_list)
    overall_results = test_results.get('Overall', {})
    overall_p_value = overall_results.get('mean_p', np.nan)
    n_reported = overall_results.get('n_valid_pairs', n_valid_pairs)

    if n_valid_pairs < 2:
        logger.warning(f"Insufficient valid pairs ({n_valid_pairs}) for Overall comparison. Skipping plot.")
        return None

    df_long = pd.DataFrame(plot_data)
    df_paired = pd.DataFrame(paired_list)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = df_paired['Middle'] / df_paired['Flanking']
        df_paired['L2FC'] = np.where(ratio > 0, np.log2(ratio), np.nan)

    mean_diff = (df_paired['Middle'] - df_paired['Flanking']).mean()

    l2fc_values = df_paired['L2FC'].dropna()
    can_draw_colorbar = not l2fc_values.empty

    if can_draw_colorbar:
        vmin, vmax = l2fc_values.min(), l2fc_values.max()
        if vmin < 0 < vmax:
            max_abs = max(abs(vmin), abs(vmax), 1e-9)
            norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=PLOT_COLORMAP)
    else:
        logger.warning("No valid L2FC values to create color mapping. Lines will use default color.")
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=PLOT_COLORMAP)

    fig, ax = plt.subplots(figsize=(7, 7))
    region_palette = {'Flanking': FLANKING_COLOR, 'Middle': MIDDLE_COLOR}
    region_order = ['Flanking', 'Middle']
    x_coords_cat = {'Flanking': 0, 'Middle': 1}
    violin_width = 0.8
    median_line_width_on_plot = 0.15

    sns.stripplot(data=df_long, x='region_type', y='pi_value', order=region_order,
                  palette=region_palette,
                  size=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                  jitter=0.15, legend=False, ax=ax, zorder=5)

    sns.violinplot(data=df_long, x='region_type', y='pi_value', order=region_order,
                   palette=region_palette,
                   inner=None, linewidth=1.2, width=violin_width,
                   cut=0, scale='width', alpha=VIOLIN_ALPHA,
                   ax=ax, zorder=10)

    median_values = df_long.groupby('region_type', observed=False)['pi_value'].median()
    for region, median_val in median_values.items():
        x_center = x_coords_cat[region]
        xmin = x_center - median_line_width_on_plot / 2
        xmax = x_center + median_line_width_on_plot / 2
        ax.hlines(y=median_val, xmin=xmin, xmax=xmax,
                  color=MEDIAN_LINE_COLOR, linestyle='-', linewidth=MEDIAN_LINE_WIDTH,
                  zorder=12, alpha=0.8)

    for _, row in df_paired.iterrows():
        l2fc_val = row['L2FC']
        x_flank = x_coords_cat['Flanking']
        x_middle = x_coords_cat['Middle']
        y_flank = row['Flanking']
        y_middle = row['Middle']

        if pd.notna(l2fc_val) and can_draw_colorbar:
             line_color = scalar_mappable.to_rgba(l2fc_val)
        else:
             line_color = DEFAULT_LINE_COLOR

        ax.plot([x_flank, x_middle], [y_flank, y_middle],
                color=line_color,
                alpha=LINE_ALPHA,
                lw=LINE_WIDTH,
                zorder=15)

    if can_draw_colorbar:
        cbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02, aspect=25, shrink=0.6)
        cbar.set_label('Log2 (π Middle / π Flanking)', rotation=270, labelpad=18, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)
        colorbar_width_adjustment = 0.90
    else:
        colorbar_width_adjustment = 0.97

    p_text = format_p_value(overall_p_value)
    diff_text = f"Mean Diff (Middle - Flank): {mean_diff:.4g}"
    n_text = f"N = {n_reported} pairs"
    annotation_text = f"{n_text}\n{diff_text}\n{p_text} (Permutation Test)"

    ax.text(0.03, 0.97, annotation_text,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=10, color='black',
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

    min_val = df_long['pi_value'].min()
    max_val = df_long['pi_value'].max()
    y_range = max_val - min_val
    y_buffer = y_range * 0.05 if y_range > 0 else 0.1
    ax.set_ylim(bottom=max(0, min_val - y_buffer), top=max_val + y_buffer)

    try:
        fig.tight_layout(rect=[0.03, 0.03, colorbar_width_adjustment, 0.93])
    except ValueError:
        logger.warning("Could not apply tight_layout. Manual adjustment might be needed.")

    output_filename = OUTPUT_DIR / f"pi_overall_{stat_type}_violin_paired_styled.png"
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved overall styled paired violin plot to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save styled violin plot to {output_filename}: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"Created and saved styled violin plot in {elapsed_time:.2f} seconds.")

    return fig


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "p = N/A"
    elif p_value < 0.001:
        return "p < 0.001"
    elif p_value < 1e-6:
         return f"p = {p_value:.2e}"
    else:
        return f"p = {p_value:.3g}"

# --- Main Execution ---

def main():
    """Main function to run the analysis pipeline."""
    total_start_time = time.time()
    logger.info("--- Starting Pi Flanking Regions Analysis (mean Focus) ---")

    # --- 1. Load Inversion Data ---
    # Specifies the path to the file containing information about inversions.
    inv_file_path = Path(INVERSION_FILE)
    # Checks if the inversion file actually exists before proceeding.
    if not inv_file_path.is_file():
        logger.error(f"Inversion info file not found: {inv_file_path}. Cannot proceed.")
        return
    logger.info(f"Loading inversion info from {inv_file_path}")
    try:
        # Reads the inversion information from the CSV file into a pandas DataFrame.
        inversion_df = pd.read_csv(inv_file_path)
        logger.info(f"Loaded {inversion_df.shape[0]} rows from inversion file.")
        # Processes the DataFrame to create dictionaries mapping chromosomes to region coordinates
        # for both recurrent and single-event inversions.
        recurrent_regions, single_event_regions = map_regions_to_inversions(inversion_df)
    except Exception as e:
        # Catches potential errors during file reading or processing.
        logger.error(f"Failed to load or process inversion file {inv_file_path}: {e}", exc_info=True)
        return

    # Define pi_file_path early for use in diagnostics and main loading.
    # Specifies the path to the file containing per-site nucleotide diversity (pi) data.
    pi_file_path = Path(PI_DATA_FILE)
    # Checks if the pi data file actually exists.
    if not pi_file_path.is_file():
         logger.error(f"Pi data file not found: {pi_file_path}. Cannot run diagnostics or analysis.")
         return

    # --- Quick Raw File Header Scan ---
    # This scan checks the input pi file *before* applying the main script's filters
    # (like 'filtered_pi' tag or MIN_LENGTH) to get a baseline count of headers
    # that geographically overlap defined single-event regions for each haplotype.
    # This helps determine if an imbalance in final counts originates from the raw input.
    logger.info("--- STARTING Quick Raw File Header Scan (Single-Event) ---")
    raw_se_direct_headers = 0     # Counter for headers matching direct single-event regions.
    raw_se_inverted_headers = 0   # Counter for headers matching inverted single-event regions.
    headers_checked_raw = 0       # Counter for total headers processed in this scan.
    try:
        # Only perform the scan if single-event regions were successfully loaded.
        if single_event_regions:
            # Opens the pi data file for reading.
            with open(pi_file_path, 'r') as f_raw:
                # Iterates through each line in the pi data file.
                for line in f_raw:
                    # Process only header lines (starting with '>').
                    if line.startswith('>'):
                        headers_checked_raw += 1
                        # Attempt to extract coordinates and group info using the existing helper function.
                        coords = extract_coordinates_from_header(line)
                        # Proceed only if coordinates and group were successfully extracted.
                        if coords and coords.get('group') is not None:
                            header_chrom = coords['chrom']
                            header_start = coords['start']
                            header_end = coords['end']
                            header_group = coords['group'] # Expected to be 0 (direct) or 1 (inverted).

                            # Check if the header's chromosome exists in the single-event regions map.
                            if header_chrom in single_event_regions:
                                # Iterate through defined single-event regions on that chromosome.
                                for r_start, r_end in single_event_regions[header_chrom]:
                                    # Use the existing helper function to check for coordinate overlap.
                                    if is_overlapping(header_start, header_end, r_start, r_end):
                                        # Increment the appropriate counter based on the haplotype group.
                                        if header_group == 0:
                                            raw_se_direct_headers += 1
                                        elif header_group == 1:
                                            raw_se_inverted_headers += 1
                                        # Important: Stop checking regions for this header once an overlap is found.
                                        # This prevents counting the same header multiple times if it spans
                                        # multiple defined single-event regions.
                                        break
        else:
            # Log a warning if no single-event regions were loaded from the inversion file.
            logger.warning("RAW SCAN: No single-event regions defined, skipping raw header count.")

    except FileNotFoundError:
        # Handle the error if the pi data file cannot be found during the scan.
        logger.error(f"RAW SCAN ERROR: Pi file not found at {pi_file_path}")
    except Exception as e:
        # Catch any other potential errors during the raw file scan.
        logger.error(f"RAW SCAN ERROR: Error reading {pi_file_path}: {e}")

    # Log the results derived from the raw file scan.
    logger.info(f"Raw Scan Results ({headers_checked_raw} headers checked):")
    logger.info(f"  Headers overlapping ANY defined Single-Event region (Direct, group=0): {raw_se_direct_headers}")
    logger.info(f"  Headers overlapping ANY defined Single-Event region (Inverted, group=1): {raw_se_inverted_headers}")
    # Issue a warning if the raw counts for direct and inverted headers differ.
    if raw_se_direct_headers != raw_se_inverted_headers:
        logger.warning("RAW SCAN: Imbalance detected in the number of headers present for single-event regions.")
    else:
        logger.info("RAW SCAN: Header counts for single-event regions appear balanced in the raw file.")
    logger.info("--- ENDING Quick Raw File Header Scan ---")
    # --- End of Raw File Scan Block ---


    # --- Run Original Post-Filter Diagnosis ---
    # This diagnosis remains useful. It checks how many *defined* single-event regions
    # have corresponding headers in the pi file that meet the specific criteria
    # applied later by the `load_pi_data` function (e.g., must contain 'filtered_pi',
    # must meet MIN_LENGTH). This helps differentiate between raw file absence and
    # filtering effects within this script.
    if single_event_regions:
         # Calls the original diagnostic function.
         diagnose_single_event_discrepancy(
             pi_file_path=pi_file_path, # Use the already defined pi_file_path
             single_event_regions=single_event_regions,
             min_length=MIN_LENGTH,
             inv_info_path=inv_file_path
         )
    else:
         # Skips this diagnosis if no single-event regions were loaded.
         logger.warning("Skipping single-event discrepancy diagnosis as no single-event regions were loaded/mapped.")


    # --- 2. Load Pi Data ---
    # This step loads the actual pi data sequences, applying filters.
    # It specifically looks for headers containing 'filtered_pi' and sequences
    # meeting the MIN_LENGTH requirement. It uses the pi_file_path defined earlier.
    # The function includes its own internal error handling and logging.
    pi_sequences = load_pi_data(pi_file_path)
    # Checks if any sequences actually passed the loading and filtering steps.
    if not pi_sequences:
        logger.error("No valid pi sequences loaded after filtering by `load_pi_data`. Exiting.")
        return

    # --- 3. Calculate Statistics ---
    # Calculates statistics (mean/median pi) for flanking and middle regions
    # for each loaded sequence.
    # This function also includes internal error handling, potentially skipping
    # sequences that are too short for flanking analysis or have NaN stats.
    flanking_stats = calculate_flanking_stats(pi_sequences)
    # Checks if any sequences remained after calculating statistics.
    if not flanking_stats:
        logger.error("No sequences remained after calculating flanking statistics (check logs for NaN/length issues). Exiting.")
        return

    # --- 4. Categorize Sequences ---
    # Assigns each sequence (with its calculated stats) into categories based on
    # overlap with defined inversion regions (recurrent/single-event) and
    # haplotype status (direct/inverted).
    # This function includes internal error handling.
    categories = categorize_sequences(flanking_stats, recurrent_regions, single_event_regions)

    # --- 5. Perform Statistical Tests (mean focus + Normality logging) ---
    # Performs statistical tests (paired permutation test for mean differences,
    # Shapiro-Wilk normality test on differences) comparing middle vs. flanking regions
    # for each category and overall.
    # This function includes internal error handling.
    test_results = perform_statistical_tests(categories, flanking_stats)

    # --- 6. Create mean Plot ---
    # Generates a bar plot visualizing the mean pi values for middle vs. flanking regions
    # across the different categories, overlaying individual data points and p-values.
    # This function includes internal error handling.
    fig_mean = create_kde_plot(flanking_stats, test_results)
    # Note: No median plot is created in this version.

    # --- 7. Summary Report (mean Focus) ---
    # Logs a formatted summary table of the analysis results to the console.
    logger.info("\n--- Analysis Summary (mean Focus) ---")
    logger.info(f"Input Pi File: {PI_DATA_FILE}")
    logger.info(f"Input Inversion File: {INVERSION_FILE}")
    logger.info(f"Min Sequence Length Filter (load_pi_data): {MIN_LENGTH}, Flank Size: {FLANK_SIZE}")
    logger.info(f"Total Sequences Used in Final Analysis (after all filters & stat calcs): {len(flanking_stats)}")

    logger.info("\nPaired Test Results (Middle vs Flanking - mean):")
    logger.info("-" * 80)
    logger.info(f"{'Category':<25} {'N Valid Pairs':<15} {'mean Diff (M-F)':<18} {'Permutation p':<15} {'Normality p':<15}")
    logger.info("-" * 80)

    # Prepare data for saving to a CSV file.
    summary_data = []

    # Iterate through the defined categories plus 'Overall'.
    for cat in CATEGORY_ORDER_WITH_OVERALL:
        # Get the list of sequences for the current category.
        if cat == 'Overall':
            seq_list = flanking_stats
        else:
            internal_cat = CAT_MAPPING[cat]
            seq_list = categories.get(internal_cat, [])

        # Recalculate N valid pairs and mean difference for summary accuracy,
        # ensuring consistency with the values used in tests.
        n_valid_pairs = 0
        mean_diff = np.nan
        if seq_list: # Ensure the list is not empty
            # Extract mean values for middle and flanking regions.
            m_means = np.array([s['middle_mean'] for s in seq_list])
            f_means = np.array([s['flanking_mean'] for s in seq_list])
            # Identify pairs where both middle and flanking means are valid (not NaN).
            valid_indices = ~np.isnan(m_means) & ~np.isnan(f_means)
            n_valid_pairs = int(np.sum(valid_indices)) # Get the count of valid pairs.
            # Calculate the mean difference only if valid pairs exist.
            if n_valid_pairs > 0:
                mean_diff = np.mean(m_means[valid_indices] - f_means[valid_indices])

        # Retrieve pre-calculated p-values from the test results dictionary.
        mean_p = test_results.get(cat, {}).get('mean_p', np.nan)
        norm_p = test_results.get(cat, {}).get('mean_normality_p', np.nan) # Normality p-value for mean differences.

        # Format the results for display.
        n_str = str(n_valid_pairs)
        mean_diff_str = f"{mean_diff:.4g}" if not np.isnan(mean_diff) else "N/A"
        mean_p_str = format_p_value(mean_p) # Use helper function for p-value formatting.
        norm_p_str = f"{norm_p:.3g}" if not np.isnan(norm_p) else "N/A"

        # Log the formatted results for the current category.
        logger.info(f"{cat:<25} {n_str:<15} {mean_diff_str:<18} {mean_p_str:<15} {norm_p_str:<15}")
        # Append the results to the list for CSV export.
        summary_data.append({
            'Category': cat,
            'N_Valid_Pairs': n_valid_pairs, # Number of pairs used in the test for this category.
            'mean_Difference_Middle_Minus_Flanking': mean_diff,
            'mean_Permutation_p_value': mean_p,
            'mean_Diff_Normality_p_value': norm_p
         })

    logger.info("-" * 80)

    # Save the summary data to a CSV file in the specified output directory.
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = OUTPUT_DIR / "pi_analysis_mean_summary.csv"
    try:
        # Writes the DataFrame to CSV, formatting floats and omitting the index.
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.5g')
        logger.info(f"Analysis summary saved to {summary_csv_path}")
    except Exception as e:
        # Catches potential errors during CSV writing.
        logger.error(f"Failed to save summary CSV to {summary_csv_path}: {e}")

    # Calculate and log the total execution time of the script.
    total_elapsed_time = time.time() - total_start_time
    logger.info(f"--- Analysis finished in {total_elapsed_time:.2f} seconds ---")

    # Close the matplotlib plot figure if it was generated to free resources.
    if fig_mean:
        plt.close(fig_mean)

if __name__ == "__main__":
    main()
