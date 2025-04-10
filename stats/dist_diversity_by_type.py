# -*- coding: utf-8 -*-
"""
Analyzes nucleotide diversity (pi) in flanking regions vs. middle regions
of genomic segments, categorized by inversion type and haplotype status,
focusing on mean values and ensuring robust statistical testing.

Handles potentially sparse pi data (many zeros).
"""

import logging
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro # Re-added for logging normality results

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

# --- Plotting ---

def format_p_value(p_value: float) -> str:
    """Formats p-value for display on the plot."""
    if np.isnan(p_value):
        return "p = N/A"
    elif p_value < 0.001:
        return "p < 0.001"
    else:
        # Use 2 or 3 significant figures, avoiding trailing zeros where possible
        return f"p = {p_value:.2g}" if p_value >= 0.01 else f"p = {p_value:.3f}"


def create_bar_plot(categories: dict, all_sequences_stats: list[dict], test_results: dict) -> plt.Figure | None:
    """
    Create bar plot comparing mean pi values by category.
    Includes scatter plot overlay of individual data points and permutation test p-values.
    Uses highly transparent bars. Only produces mean plot.

    Args:
        categories: Dictionary with categorized sequence statistics.
        all_sequences_stats: List of all sequence statistics (for 'Overall').
        test_results: Dictionary containing pre-calculated p-values for the mean.

    Returns:
        matplotlib Figure object or None if no data to plot.
    """
    stat_type = "mean" # Hardcoded to mean
    logger.info(f"Creating bar plot for {stat_type.capitalize()}...")
    start_time = time.time()

    # Determine which fields to use based on stat_type="mean"
    flanking_field = f"flanking_{stat_type}"
    middle_field = f"middle_{stat_type}"
    p_value_field = f"{stat_type}_p" # Key to fetch p-value from test_results

    # Prepare data for plotting
    plot_data = {}
    max_stat_value = 0 # Keep track of max value for y-axis limits and annotation placement

    # Collect stats for each category in the defined order + Overall
    for display_cat in CATEGORY_ORDER_WITH_OVERALL:
        if display_cat == 'Overall':
            seq_list = all_sequences_stats
        else:
            internal_cat = CAT_MAPPING[display_cat]
            seq_list = categories.get(internal_cat, [])
        count = len(seq_list)

        # Calculate aggregated mean stats for the bars
        valid_flanking = [s[flanking_field] for s in seq_list if not np.isnan(s[flanking_field])]
        valid_middle = [s[middle_field] for s in seq_list if not np.isnan(s[middle_field])]

        # Use mean for bar height
        agg_flanking_stat = np.mean(valid_flanking) if valid_flanking else np.nan
        agg_middle_stat = np.mean(valid_middle) if valid_middle else np.nan

        # Extract individual paired points for scatter plot, removing pairs with NaNs
        paired_flanking = []
        paired_middle = []
        for s in seq_list:
            f_val = s[flanking_field]
            m_val = s[middle_field]
            if not np.isnan(f_val) and not np.isnan(m_val):
                paired_flanking.append(f_val)
                paired_middle.append(m_val)
                # Update max for y-limit based on individual points
                max_stat_value = max(max_stat_value, f_val, m_val)

        plot_data[display_cat] = {
            'flanking_agg': agg_flanking_stat,
            'middle_agg': agg_middle_stat,
            'flanking_points': paired_flanking,
            'middle_points': paired_middle,
            'count': count,
            'p_value': test_results.get(display_cat, {}).get(p_value_field, np.nan)
        }
        # Update max for y-limit based on aggregated bar heights
        max_stat_value = max(max_stat_value, agg_flanking_stat if not np.isnan(agg_flanking_stat) else 0, agg_middle_stat if not np.isnan(agg_middle_stat) else 0)


    # Check if there's anything to plot
    all_points_count = sum(len(d['flanking_points']) for d in plot_data.values())
    if all_points_count == 0:
         logger.warning(f"No valid data points found to plot for {stat_type}. Skipping plot generation.")
         return None

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(16, 9)) # Wider format
    x = np.arange(len(CATEGORY_ORDER_WITH_OVERALL)) # x-coordinates for categories
    width = 0.35 # Width of the bars

    # Extract aggregated values for bars
    flanking_bar_values = [plot_data[cat]['flanking_agg'] for cat in CATEGORY_ORDER_WITH_OVERALL]
    middle_bar_values = [plot_data[cat]['middle_agg'] for cat in CATEGORY_ORDER_WITH_OVERALL]

    # Create bars with high transparency
    rects1 = ax.bar(x - width/2, flanking_bar_values, width, label='Flanking Regions', color=FLANKING_COLOR, alpha=BAR_ALPHA)
    rects2 = ax.bar(x + width/2, middle_bar_values, width, label='Middle Region', color=MIDDLE_COLOR, alpha=BAR_ALPHA)

    # Add scatter plot overlay and p-value annotations
    # Add a small buffer to the max value for limits
    y_lim_max = max_stat_value * 1.20 if max_stat_value > 0 else 1.0 # Avoid 0 limit if max is 0
    annotation_y_pos = max_stat_value * 1.05 if max_stat_value > 0 else 0.1 # Position annotations

    for i, cat in enumerate(CATEGORY_ORDER_WITH_OVERALL):
        cat_data = plot_data[cat]
        n_points = len(cat_data['flanking_points']) # Should be same as middle_points

        if n_points > 0:
            # Add jitter for scatter points
            jitter_flank = np.random.normal(0, 0.04, size=n_points)
            jitter_middle = np.random.normal(0, 0.04, size=n_points)

            # Plot scatter points (they are drawn on top of bars)
            ax.scatter(x[i] - width/2 + jitter_flank, cat_data['flanking_points'],
                       color=FLANKING_COLOR, alpha=SCATTER_ALPHA, s=SCATTER_SIZE, edgecolors='grey', linewidth=0.5, zorder=3)
            ax.scatter(x[i] + width/2 + jitter_middle, cat_data['middle_points'],
                       color=MIDDLE_COLOR, alpha=SCATTER_ALPHA, s=SCATTER_SIZE, edgecolors='grey', linewidth=0.5, zorder=3)

        # Add p-value annotation (using the pre-calculated mean p-value)
        p_value = cat_data['p_value']
        p_text = format_p_value(p_value)
        # Position annotation slightly above the highest point/bar in that category
        cat_max_y = 0
        if n_points > 0:
             cat_max_y = max(max(cat_data['flanking_points']), max(cat_data['middle_points']))
        cat_max_y = max(cat_max_y, cat_data['flanking_agg'] if not np.isnan(cat_data['flanking_agg']) else 0, cat_data['middle_agg'] if not np.isnan(cat_data['middle_agg']) else 0)
        current_annotation_y_pos = cat_max_y * 1.05 if cat_max_y > 0 else annotation_y_pos # Dynamic y-pos per category


        ax.text(x[i], current_annotation_y_pos, p_text, # Position above the pair of bars
                ha='center', va='bottom', fontsize=10, color='black')

    # --- Final Plot Styling ---
    ax.set_ylabel(f'Nucleotide Diversity (π) - {stat_type.capitalize()}', fontsize=12)
    ax.set_xticks(x)
    # Use display names and add counts to labels
    x_labels = [f"{cat}\n(n={plot_data[cat]['count']})" for cat in CATEGORY_ORDER_WITH_OVERALL]
    ax.set_xticklabels(x_labels, fontsize=10, rotation=0, ha='center')
    ax.set_title(f'Comparison of π ({stat_type.capitalize()}) between Flanking and Middle Regions', fontsize=16, pad=20)
    ax.legend(title="Region Type", loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust y-axis limit
    ax.set_ylim(bottom=0, top=y_lim_max) # bottom is 0

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save plot
    output_filename = OUTPUT_DIR / f"pi_flanking_regions_{stat_type}_bar_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {stat_type} plot to {output_filename}")

    elapsed_time = time.time() - start_time
    logger.info(f"Created and saved plot in {elapsed_time:.2f} seconds.")

    return fig # Return figure object if needed elsewhere


# --- Main Execution ---

def main():
    """Main function to run the analysis pipeline."""
    total_start_time = time.time()
    logger.info("--- Starting Pi Flanking Regions Analysis (mean Focus) ---")

    # --- 1. Load Inversion Data ---
    inv_file_path = Path(INVERSION_FILE)
    if not inv_file_path.is_file():
        logger.error(f"Inversion info file not found: {inv_file_path}. Cannot proceed.")
        return
    logger.info(f"Loading inversion info from {inv_file_path}")
    try:
        inversion_df = pd.read_csv(inv_file_path)
        logger.info(f"Loaded {inversion_df.shape[0]} rows from inversion file.")
        recurrent_regions, single_event_regions = map_regions_to_inversions(inversion_df)
    except Exception as e:
        logger.error(f"Failed to load or process inversion file {inv_file_path}: {e}", exc_info=True)
        return

    # --- 2. Load Pi Data ---
    pi_file_path = Path(PI_DATA_FILE)
    if not pi_file_path.is_file():
        logger.error(f"Pi data file not found: {pi_file_path}. Cannot proceed.")
        return
    # load_pi_data includes internal error handling and logging
    pi_sequences = load_pi_data(pi_file_path)
    if not pi_sequences:
        logger.error("No valid pi sequences loaded after filtering. Exiting.")
        return

    # --- 3. Calculate Statistics ---
    # calculate_flanking_stats includes internal error handling
    flanking_stats = calculate_flanking_stats(pi_sequences)
    if not flanking_stats:
        logger.error("No sequences remained after calculating flanking statistics (check min length/NaNs). Exiting.")
        return

    # --- 4. Categorize Sequences ---
    # categorize_sequences includes internal error handling
    categories = categorize_sequences(flanking_stats, recurrent_regions, single_event_regions)

    # --- 5. Perform Statistical Tests (mean focus + Normality logging) ---
    # perform_statistical_tests includes internal error handling
    test_results = perform_statistical_tests(categories, flanking_stats)

    # --- 6. Create mean Plot ---
    # create_bar_plot includes internal error handling
    fig_mean = create_bar_plot(categories, flanking_stats, test_results)
    # No median plot is created

    # --- 7. Summary Report (mean Focus) ---
    logger.info("\n--- Analysis Summary (mean Focus) ---")
    logger.info(f"Input Pi File: {PI_DATA_FILE}")
    logger.info(f"Input Inversion File: {INVERSION_FILE}")
    logger.info(f"Min Sequence Length: {MIN_LENGTH}, Flank Size: {FLANK_SIZE}")
    logger.info(f"Total Sequences Analyzed (Passed All Filters): {len(flanking_stats)}")

    logger.info("\nPaired Test Results (Middle vs Flanking - mean):")
    logger.info("-" * 80)
    logger.info(f"{'Category':<25} {'N Valid Pairs':<15} {'mean Diff (M-F)':<18} {'Permutation p':<15} {'Normality p':<15}")
    logger.info("-" * 80)

    summary_data = [] # To potentially save later

    for cat in CATEGORY_ORDER_WITH_OVERALL:
        if cat == 'Overall':
            seq_list = flanking_stats
        else:
            internal_cat = CAT_MAPPING[cat]
            seq_list = categories.get(internal_cat, [])

        # Calculate N valid pairs and mean difference again for summary accuracy
        n_valid_pairs = 0
        mean_diff = np.nan
        if seq_list:
            m_means = np.array([s['middle_mean'] for s in seq_list])
            f_means = np.array([s['flanking_mean'] for s in seq_list])
            valid_indices = ~np.isnan(m_means) & ~np.isnan(f_means)
            n_valid_pairs = int(np.sum(valid_indices)) # Cast to int for display
            if n_valid_pairs > 0:
                mean_diff = np.mean(m_means[valid_indices] - f_means[valid_indices])

        # Get pre-calculated p-values
        mean_p = test_results.get(cat, {}).get('mean_p', np.nan)
        norm_p = test_results.get(cat, {}).get('mean_normality_p', np.nan) # Normality p-value

        # Format for printing
        n_str = str(n_valid_pairs)
        mean_diff_str = f"{mean_diff:.4g}" if not np.isnan(mean_diff) else "N/A"
        mean_p_str = format_p_value(mean_p) # Uses helper
        norm_p_str = f"{norm_p:.3g}" if not np.isnan(norm_p) else "N/A"

        logger.info(f"{cat:<25} {n_str:<15} {mean_diff_str:<18} {mean_p_str:<15} {norm_p_str:<15}")
        summary_data.append({
            'Category': cat,
            'N_Valid_Pairs': n_valid_pairs,
            'mean_Difference_Middle_Minus_Flanking': mean_diff,
            'mean_Permutation_p_value': mean_p,
            'mean_Diff_Normality_p_value': norm_p
         })

    logger.info("-" * 80)

    # : Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = OUTPUT_DIR / "pi_analysis_mean_summary.csv"
    try:
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.5g')
        logger.info(f"Analysis summary saved to {summary_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV to {summary_csv_path}: {e}")


    total_elapsed_time = time.time() - total_start_time
    logger.info(f"--- Analysis finished in {total_elapsed_time:.2f} seconds ---")

    # Close the plot figure if it was generated
    if fig_mean:
        plt.close(fig_mean)


if __name__ == "__main__":
    main()
