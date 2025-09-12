import pandas as pd
import numpy as np
import re
import logging
import sys
from scipy import stats # Keep for mannwhitneyu, fisher_exact
from collections import defaultdict
import warnings

# Ignore specific warnings if needed
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

# --- Configuration ---
# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', # More detail
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('conservation_analysis_detailed')

# File paths
PAIRWISE_FILE = 'all_pairwise_results.csv'
INVERSION_FILE = 'inv_info.tsv'
OUTPUT_RESULTS_LOO_GENE_PROPORTION = 'leave_one_out_gene_proportion_results.csv' # Output name

# --- Helper Functions ---

def extract_coordinates_from_cds(cds_string):
    """Extract genomic coordinates from CDS string. (Unchanged)"""
    if not isinstance(cds_string, str):
        return None
    pattern = r'(chr\w+)_start(\d+)_end(\d+)'
    match = re.search(pattern, cds_string)
    return {
        'chrom': match.group(1),
        'start': int(match.group(2)),
        'end': int(match.group(3))
    } if match else None

def map_cds_to_inversions(pairwise_df, inversion_df):
    """
    Map CDS strings to specific inversions, track types, and log filtering extensively.
    """
    logger.info("--- Function: map_cds_to_inversions START ---")
    func_name = "map_cds_to_inversions" # For logging context

    # --- Step 1: Prepare Pairwise Data (Extract CDS Coordinates) ---
    logger.info(f"[{func_name}] Preparing pairwise data...")
    if 'CDS' not in pairwise_df.columns:
        logger.error(f"[{func_name}] Input pairwise DataFrame missing 'CDS' column. Cannot map.")
        return {}, {}, {}
    initial_pairwise_rows = len(pairwise_df)
    logger.info(f"[{func_name}] Initial pairwise rows: {initial_pairwise_rows:,}")

    unique_cds_series = pairwise_df['CDS'].dropna().unique()
    initial_unique_cds_count = len(unique_cds_series)
    logger.info(f"[{func_name}] Found {initial_unique_cds_count:,} unique non-NA CDS identifiers in pairwise data.")
    if initial_unique_cds_count == 0:
        logger.warning(f"[{func_name}] No unique non-NA CDS found. Cannot map.")
        return {}, {}, {}

    cds_coords = {}
    parsing_failures = 0
    for cds in unique_cds_series:
        coords = extract_coordinates_from_cds(cds)
        if coords:
            cds_coords[cds] = coords
        else:
            parsing_failures += 1
            # Log only a few examples to avoid flooding
            if parsing_failures <= 5:
                logger.warning(f"[{func_name}] Could not parse coordinates from CDS string: {cds}")
            elif parsing_failures == 6:
                 logger.warning(f"[{func_name}] (Further coordinate parsing warnings suppressed)")

    parsed_cds_count = len(cds_coords)
    logger.info(f"[{func_name}] Successfully parsed coordinates for {parsed_cds_count:,} unique CDS.")
    if parsing_failures > 0:
        logger.warning(f"[{func_name}] Failed to parse coordinates for {parsing_failures:,} unique CDS strings.")
    if parsed_cds_count == 0:
        logger.warning(f"[{func_name}] No valid CDS coordinates could be extracted. Cannot map.")
        return {}, {}, {}

    # --- Step 2: Prepare Inversion Data ---
    logger.info(f"[{func_name}] Preparing inversion data...")
    initial_inversion_count = len(inversion_df)
    logger.info(f"[{func_name}] Initial inversion count: {initial_inversion_count:,}")
    required_inv_cols = ['Start', 'End', '0_single_1_recur', 'Chromosome']
    if not all(col in inversion_df.columns for col in required_inv_cols):
        logger.error(f"[{func_name}] Inversion info DataFrame missing required columns: {required_inv_cols}. Cannot map.")
        return {}, {}, {}

    # Make a copy to avoid SettingWithCopyWarning if inversion_df is used elsewhere
    inversion_df_proc = inversion_df.copy()

    # Convert necessary columns to numeric, coercing errors
    for col in ['Start', 'End', '0_single_1_recur']:
        inversion_df_proc[col] = pd.to_numeric(inversion_df_proc[col], errors='coerce')

    # Drop rows with NaNs in essential columns AFTER coercion
    rows_before_dropna = len(inversion_df_proc)
    inversion_df_proc.dropna(subset=required_inv_cols, inplace=True)
    rows_after_dropna = len(inversion_df_proc)
    dropped_inversions = rows_before_dropna - rows_after_dropna
    if dropped_inversions > 0:
        logger.warning(f"[{func_name}] Removed {dropped_inversions:,} inversions due to missing/invalid data in columns: {required_inv_cols}.")
    logger.info(f"[{func_name}] Inversions remaining after cleaning: {rows_after_dropna:,}")
    if rows_after_dropna == 0:
        logger.error(f"[{func_name}] No valid inversions remain after cleaning. Cannot map.")
        return {}, {}, {}

    # Ensure correct types
    inversion_df_proc['Start'] = inversion_df_proc['Start'].astype(int)
    inversion_df_proc['End'] = inversion_df_proc['End'].astype(int)
    inversion_df_proc['0_single_1_recur'] = inversion_df_proc['0_single_1_recur'].astype(int)

    # Separate inversion types
    recurrent_inv = inversion_df_proc[inversion_df_proc['0_single_1_recur'] == 1]
    single_event_inv = inversion_df_proc[inversion_df_proc['0_single_1_recur'] == 0]
    logger.info(f"[{func_name}] Input inversion counts by type: Recurrent={len(recurrent_inv):,}, Single-Event={len(single_event_inv):,}")
    logger.info(f"   (Total potential inversions to match against: {len(recurrent_inv) + len(single_event_inv):,})")

    # --- Step 3: Match CDS to Inversions ---
    logger.info(f"[{func_name}] Matching {parsed_cds_count:,} parsed CDS to {len(inversion_df_proc):,} cleaned inversions...")
    cds_to_type = {}
    cds_to_inversion_id = {}
    inversion_to_cds = defaultdict(list)
    inversions_with_matches = set() # Track which inversions actually get matched

    processed_cds_count = 0
    cds_matched_to_rec = 0
    cds_matched_to_single = 0
    cds_matched_to_ambiguous = 0
    cds_not_matched = 0

    for cds, coords in cds_coords.items():
        # This loop iterates through CDS that *could* be parsed
        processed_cds_count += 1
        if coords is None: continue # Should not happen due to previous check, but safe
        chrom, start, end = coords['chrom'], coords['start'], coords['end']

        # Find overlapping inversions
        rec_matches = recurrent_inv[
            (recurrent_inv['Chromosome'] == chrom) &
            (start <= recurrent_inv['End']) & # Gene starts before inversion ends
            (end >= recurrent_inv['Start'])   # Gene ends after inversion starts
        ]
        single_matches = single_event_inv[
            (single_event_inv['Chromosome'] == chrom) &
            (start <= single_event_inv['End']) &
            (end >= single_event_inv['Start'])
        ]

        is_recurrent = len(rec_matches) > 0
        is_single = len(single_matches) > 0
        inv_type = 'unknown' # Default if no match
        matched_inv_ids = []

        if is_recurrent and not is_single:
            inv_type = 'recurrent'
            matched_inv_ids = rec_matches.index.tolist()
            cds_matched_to_rec += 1
        elif is_single and not is_recurrent:
            inv_type = 'single_event'
            matched_inv_ids = single_matches.index.tolist()
            cds_matched_to_single += 1
        elif is_recurrent and is_single:
            inv_type = 'ambiguous' # Overlaps both types
            matched_inv_ids = rec_matches.index.tolist() + single_matches.index.tolist()
            cds_matched_to_ambiguous += 1
        else:
            # No overlap with *any* cleaned recurrent or single inversion
            cds_not_matched += 1
            inv_type = 'unknown' # Explicitly mark as not overlapping known types

        cds_to_type[cds] = inv_type
        cds_to_inversion_id[cds] = matched_inv_ids
        for inv_id in matched_inv_ids:
            inversion_to_cds[inv_id].append(cds)
            inversions_with_matches.add(inv_id) # Add inversion index

        # Log progress periodically
        if processed_cds_count % 500 == 0 or processed_cds_count == parsed_cds_count:
             logger.info(f"[{func_name}]   Processed {processed_cds_count:,}/{parsed_cds_count:,} parsed CDS for mapping...")

    logger.info(f"[{func_name}] Finished matching CDS to inversions.")
    logger.info(f"[{func_name}]   CDS Mapping Summary (based on {parsed_cds_count:,} parsed CDS):")
    logger.info(f"[{func_name}]     - Mapped to Recurrent ONLY: {cds_matched_to_rec:,}")
    logger.info(f"[{func_name}]     - Mapped to Single-Event ONLY: {cds_matched_to_single:,}")
    logger.info(f"[{func_name}]     - Mapped to Ambiguous (Both Rec & Single): {cds_matched_to_ambiguous:,}")
    logger.info(f"[{func_name}]     - Not overlapping any Rec/Single inversion: {cds_not_matched:,}")
    logger.info(f"[{func_name}]     (Total CDS mapped to *any* type: {parsed_cds_count - cds_not_matched:,})")

    # Add CDS that failed parsing back as 'unknown' for completeness in the map
    unparsed_cds = set(unique_cds_series) - set(cds_coords.keys())
    for cds in unparsed_cds:
        cds_to_type[cds] = 'unknown'
        cds_to_inversion_id[cds] = []
    logger.info(f"[{func_name}] Added {len(unparsed_cds):,} CDS that failed coordinate parsing to maps as 'unknown'.")
    logger.info(f"[{func_name}] Final `cds_to_type` map size: {len(cds_to_type):,} (covers all initial unique non-NA CDS)")

    final_type_counts = pd.Series(cds_to_type).value_counts().to_dict()
    logger.info(f"[{func_name}] Final CDS counts by assigned type: {final_type_counts}")

    inversion_to_cds_dict = dict(inversion_to_cds) # Convert defaultdict to dict
    logger.info(f"[{func_name}] Number of unique inversions with at least one matched CDS: {len(inversions_with_matches):,}")
    logger.info(f"[{func_name}]   (Compare this to the {len(recurrent_inv) + len(single_event_inv):,} potential inversions)")
    logger.info(f"[{func_name}] Final `inversion_to_cds` map size (number of keys = inversions with genes): {len(inversion_to_cds_dict):,}")


    logger.info("--- Function: map_cds_to_inversions END ---")
    return cds_to_type, cds_to_inversion_id, inversion_to_cds_dict

# --- Analysis 1: RAW Pairwise Omega Values (Identical vs Non-Identical) ---

def analyze_raw_pair_proportions(pairwise_df, cds_to_type):
    """
    Compares proportion of identical pairs (omega == -1) using RAW pairwise data
    between recurrent and single-event inversions. Logs filtering steps.
    """
    logger.info("\n--- Function: analyze_raw_pair_proportions START ---")
    func_name = "analyze_raw_pair_proportions"

    # --- Step 1: Map inversion types ---
    initial_rows = len(pairwise_df)
    logger.info(f"[{func_name}] Starting with {initial_rows:,} raw pairwise rows.")
    if 'CDS' not in pairwise_df.columns:
        logger.error(f"[{func_name}] Pairwise DataFrame missing 'CDS' column. Cannot proceed.")
        return
    pairwise_df['inversion_type'] = pairwise_df['CDS'].map(cds_to_type) # Get type for each pair via its CDS
    logger.info(f"[{func_name}] Mapped inversion types to pairwise rows using `cds_to_type` map.")
    # Count how many pairs are now associated with each type
    type_counts_in_pairs = pairwise_df['inversion_type'].value_counts(dropna=False).to_dict()
    logger.info(f"[{func_name}] Pairwise row counts by mapped CDS type: {type_counts_in_pairs}")

    # --- Step 2: Filter for relevant pairs ---
    logger.info(f"[{func_name}] Filtering pairwise data for analysis...")
    required_cols = ['inversion_type', 'Group1', 'Group2', 'omega', 'CDS']
    if not all(col in pairwise_df.columns for col in required_cols):
        logger.error(f"[{func_name}] Pairwise DataFrame missing required columns for raw analysis: {required_cols}")
        return

    # Filter 1: Keep only pairs associated with 'recurrent' or 'single_event' CDS
    rows_before_type_filter = len(pairwise_df)
    analysis_df = pairwise_df[pairwise_df['inversion_type'].isin(['recurrent', 'single_event'])].copy()
    rows_after_type_filter = len(analysis_df)
    logger.info(f"[{func_name}] Filter 1: Kept pairs with inversion_type 'recurrent' or 'single_event'. Rows remaining: {rows_after_type_filter:,} (removed {rows_before_type_filter - rows_after_type_filter:,})")

    # Filter 2: Keep only pairs where both groups are Group 1 (within-inversion comparisons)
    rows_before_group_filter = len(analysis_df)
    analysis_df = analysis_df[(analysis_df['Group1'] == 1) & (analysis_df['Group2'] == 1)]
    rows_after_group_filter = len(analysis_df)
    logger.info(f"[{func_name}] Filter 2: Kept pairs with Group1 == 1 and Group2 == 1. Rows remaining: {rows_after_group_filter:,} (removed {rows_before_group_filter - rows_after_group_filter:,})")

    # Filter 3: Keep only pairs with valid omega values (-1 or >= 0)
    rows_before_omega_filter = len(analysis_df)
    analysis_df['omega'] = pd.to_numeric(analysis_df['omega'], errors='coerce') # Ensure numeric
    analysis_df = analysis_df[pd.notna(analysis_df['omega']) & (analysis_df['omega'] != 99)]
    rows_after_omega_filter = len(analysis_df)
    logger.info(f"[{func_name}] Filter 3: Kept pairs with non-NA and non-99 omega. Rows remaining: {rows_after_omega_filter:,} (removed {rows_before_omega_filter - rows_after_omega_filter:,})")

    relevant_pairs_df = analysis_df # Final dataframe for this analysis
    final_relevant_count = len(relevant_pairs_df)
    logger.info(f"[{func_name}] Total relevant pairs for raw proportion analysis: {final_relevant_count:,}")

    if relevant_pairs_df.empty:
        logger.warning(f"[{func_name}] No relevant pairs found for raw pairwise analysis after filtering.")
        print("\n--- RAW PAIR PROPORTION ANALYSIS ---")
        print("  No data available for comparison.")
        print("------------------------------------")
        logger.info(f"--- Function: analyze_raw_pair_proportions END (No data) ---")
        return

    # --- Step 3: Calculate Proportions and Test ---
    rec_pairs = relevant_pairs_df[relevant_pairs_df['inversion_type'] == 'recurrent']
    single_pairs = relevant_pairs_df[relevant_pairs_df['inversion_type'] == 'single_event']
    logger.info(f"[{func_name}] Counts for Fisher's Exact Test: Recurrent pairs = {len(rec_pairs):,}, Single-Event pairs = {len(single_pairs):,}")

    rec_total_raw = len(rec_pairs)
    rec_identical_raw = len(rec_pairs[rec_pairs['omega'] == -1])
    rec_non_identical_raw = rec_total_raw - rec_identical_raw

    single_total_raw = len(single_pairs)
    single_identical_raw = len(single_pairs[single_pairs['omega'] == -1])
    single_non_identical_raw = single_total_raw - single_identical_raw

    if rec_total_raw == 0 or single_total_raw == 0:
         logger.warning(f"[{func_name}] One or both groups (recurrent/single-event) have zero pairs after filtering. Cannot perform Fisher's test.")
         print("\n--- RAW PAIR PROPORTION ANALYSIS (Fisher's Exact Test) ---")
         print("  Skipping test: Not enough data in one or both groups.")
         print(f"    Recurrent Pair Count: {rec_total_raw}")
         print(f"    Single-Event Pair Count: {single_total_raw}")
         print("----------------------------------------------------------")
         logger.info(f"--- Function: analyze_raw_pair_proportions END (Insufficient data for test) ---")
         return

    # --- Step 4: Perform Fisher's Exact Test and Report ---
    table_raw = [[rec_identical_raw, rec_non_identical_raw],
                 [single_identical_raw, single_non_identical_raw]]
    logger.info(f"[{func_name}] Contingency table for Fisher's test: {table_raw}")

    print("\n--- RAW PAIR PROPORTION ANALYSIS (Fisher's Exact Test) ---")
    print(f"Comparing proportion of identical pairs (omega == -1) using RAW pairwise data:")

    odds_ratio_raw, p_value_raw = np.nan, 1.0 # Initialize
    # Calculate percentages carefully to avoid division by zero
    rec_pct_identical_raw = (rec_identical_raw / rec_total_raw * 100) if rec_total_raw > 0 else 0
    single_pct_identical_raw = (single_identical_raw / single_total_raw * 100) if single_total_raw > 0 else 0
    rec_pct_non_identical_raw = 100.0 - rec_pct_identical_raw
    single_pct_non_identical_raw = 100.0 - single_pct_identical_raw

    print(f"\n  Overall Counts & Proportions:")
    print(f"    Recurrent Pairs:    {rec_identical_raw:,} / {rec_total_raw:,} ({rec_pct_identical_raw:.2f}%) identical")
    print(f"    Single-Event Pairs: {single_identical_raw:,} / {single_total_raw:,} ({single_pct_identical_raw:.2f}%) identical")
    print(f"    (Non-Identical: Rec={rec_pct_non_identical_raw:.2f}%, Single={single_pct_non_identical_raw:.2f}%)")

    try:
        # Check for cases where Fisher's test might be trivial or invalid
        if (rec_total_raw == rec_identical_raw and single_total_raw == single_identical_raw) or \
           (rec_identical_raw == 0 and single_identical_raw == 0) or \
           (rec_total_raw == rec_non_identical_raw and single_total_raw == single_non_identical_raw) or \
           (rec_non_identical_raw == 0 and single_non_identical_raw == 0):
             logger.warning(f"[{func_name}] Fisher's test skipped for raw pairs: No variation in identity across groups or within a group.")
             odds_ratio_raw, p_value_raw = np.nan, 1.0 # Indicate skipped/trivial
        else:
             # Ensure table contains integers >= 0
             table_raw_int = [[int(max(0, c)) for c in row] for row in table_raw]
             odds_ratio_raw, p_value_raw = stats.fisher_exact(table_raw_int)
             logger.info(f"[{func_name}] Fisher's exact test successful: OR={odds_ratio_raw:.4f}, p={p_value_raw:.4e}")

        print(f"\n  Fisher's Exact Test Results:")
        print(f"    Odds Ratio (Identical, Single vs Recurrent): {odds_ratio_raw:.4f}" if not np.isnan(odds_ratio_raw) else "    Odds Ratio: N/A (Trivial case or error)")
        print(f"    P-value: {p_value_raw:.4e}" if not np.isnan(p_value_raw) else "    P-value: N/A (Trivial case or error)")

    except ValueError as e:
         logger.error(f"[{func_name}] Fisher's exact test failed for raw pairs: {e}")
         print(f"\n  Fisher's Exact Test Results:")
         print(f"    Error during calculation: {e}. Table: {table_raw}")

    # Fold Change Calculation (Optional but informative)
    print(f"\n  Fold Change Calculations:")
    if single_pct_non_identical_raw > 1e-9: # Avoid division by zero
        fold_change_non_identical_rec_vs_single = rec_pct_non_identical_raw / single_pct_non_identical_raw
        print(f"    Fold Change (Non-Identical %, Recurrent / Single): {fold_change_non_identical_rec_vs_single:.1f}x")
    elif rec_pct_non_identical_raw > 1e-9: # Rec has non-identical, Single does not
         print(f"    Fold Change (Non-Identical %, Recurrent / Single): Infinite (Single % non-identical is zero)")
    else: # Neither has non-identical pairs
         print(f"    Fold Change (Non-Identical %, Recurrent / Single): Undefined (Both groups have 0% non-identical)")

    alpha = 0.05
    print(f"\n  Conclusion (alpha={alpha}):")
    if not np.isnan(p_value_raw) and p_value_raw != 1.0: # Check if test was performed and wasn't trivial=1.0
        if p_value_raw < alpha:
            print(f"    Significant difference detected between groups based on raw pair proportions.")
            direction = "higher" if single_pct_identical_raw > rec_pct_identical_raw else "lower"
            print(f"    Single-Event pairs have a significantly {direction} proportion of identical sequences.")
        else:
            print(f"    NO significant difference detected between groups based on raw pair proportions.")
    else:
        print(f"    Significance could not be determined (test skipped, failed, or trivial).")

    print("----------------------------------------------------------")
    logger.info("--- Function: analyze_raw_pair_proportions END ---")


# --- Analysis 2 Helper: Calculate Proportion Non-Identical Pairs per GENE ---

def calculate_gene_proportion_non_identical(pairwise_df):
    """
    Calculate the proportion of non-identical pairs within each GENE (CDS). Logs filtering.
    """
    logger.info("\n--- Function: calculate_gene_proportion_non_identical START ---")
    func_name = "calculate_gene_proportion_non_identical"

    # --- Step 1: Filter pairwise data ---
    initial_rows = len(pairwise_df)
    initial_unique_cds = pairwise_df['CDS'].nunique()
    logger.info(f"[{func_name}] Starting with {initial_rows:,} raw pairwise rows, representing {initial_unique_cds:,} unique CDS.")
    required_cols = ['CDS', 'Group1', 'Group2', 'omega']
    if not all(col in pairwise_df.columns for col in required_cols):
        logger.error(f"[{func_name}] Pairwise DataFrame missing required columns for gene proportion calculation: {required_cols}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['CDS', 'total_pairs', 'non_identical_pairs', 'proportion_non_identical'])

    logger.info(f"[{func_name}] Filtering pairwise data for gene proportion calculation...")

    # Filter 1: Keep only pairs where both groups are Group 1
    rows_before_group_filter = len(pairwise_df)
    analysis_df = pairwise_df[(pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1)].copy()
    rows_after_group_filter = len(analysis_df)
    logger.info(f"[{func_name}] Filter 1: Kept pairs with Group1 == 1 and Group2 == 1. Rows remaining: {rows_after_group_filter:,} (removed {rows_before_group_filter - rows_after_group_filter:,})")

    # Filter 2: Keep only pairs with valid omega values (-1 or >= 0)
    rows_before_omega_filter = len(analysis_df)
    analysis_df['omega'] = pd.to_numeric(analysis_df['omega'], errors='coerce') # Ensure numeric
    analysis_df = analysis_df[pd.notna(analysis_df['omega']) & (analysis_df['omega'] != 99)]
    rows_after_omega_filter = len(analysis_df)
    logger.info(f"[{func_name}] Filter 2: Kept pairs with non-NA and non-99 omega. Rows remaining: {rows_after_omega_filter:,} (removed {rows_before_omega_filter - rows_after_omega_filter:,})")

    # Filter 3: Ensure CDS identifier is not NA
    rows_before_cds_filter = len(analysis_df)
    analysis_df = analysis_df[analysis_df['CDS'].notna()]
    rows_after_cds_filter = len(analysis_df)
    logger.info(f"[{func_name}] Filter 3: Kept pairs with non-NA CDS identifier. Rows remaining: {rows_after_cds_filter:,} (removed {rows_before_cds_filter - rows_after_cds_filter:,})")

    relevant_pairs_df = analysis_df
    final_relevant_count = len(relevant_pairs_df)
    final_unique_cds_count = relevant_pairs_df['CDS'].nunique()
    logger.info(f"[{func_name}] Total relevant pairs for gene proportion calculation: {final_relevant_count:,}")
    logger.info(f"[{func_name}] These pairs represent {final_unique_cds_count:,} unique CDS identifiers.")

    if relevant_pairs_df.empty:
        logger.warning(f"[{func_name}] No valid pairs found for gene proportion calculation after filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=['CDS', 'total_pairs', 'non_identical_pairs', 'proportion_non_identical'])

    # --- Step 2: Calculate proportions per gene ---
    def calculate_proportions(group):
        """Helper to calculate proportions for a group (gene)."""
        total_pairs = len(group)
        non_identical_pairs = len(group[group['omega'] != -1]) # Count pairs where omega is not -1
        # Note: NaN or 99 omegas were already filtered out
        if total_pairs > 0:
            proportion = non_identical_pairs / total_pairs
        else:
            proportion = np.nan # Should not happen if group exists, but safe
        return pd.Series({
            'total_pairs': total_pairs,
            'non_identical_pairs': non_identical_pairs,
            'proportion_non_identical': proportion
        })

    logger.info(f"[{func_name}] Grouping by 'CDS' and calculating proportions for {final_unique_cds_count:,} genes...")
    try:
        # Group by CDS and apply the calculation
        gene_proportion_df = relevant_pairs_df.groupby('CDS', observed=True).apply(calculate_proportions, include_groups=False).reset_index()
        # `observed=True` is generally good practice with categorical/string keys
        # `include_groups=False` avoids a potential future warning/change in pandas behavior.

        calculated_gene_count = len(gene_proportion_df)
        logger.info(f"[{func_name}] Successfully calculated proportions for {calculated_gene_count:,} unique CDS identifiers.")

        # --- Step 3: Final check and cleanup ---
        # Although the calculation logic avoids NaN proportions if total_pairs > 0,
        # it's good practice to check, especially if the input data could be strange.
        initial_count = len(gene_proportion_df)
        gene_proportion_df.dropna(subset=['proportion_non_identical'], inplace=True)
        final_gene_count = len(gene_proportion_df)
        genes_removed_post_calc = initial_count - final_gene_count
        if genes_removed_post_calc > 0:
             logger.warning(f"[{func_name}] Removed {genes_removed_post_calc:,} genes post-calculation due to NaN proportion (unexpected).")
        else:
             logger.info(f"[{func_name}] All {initial_count:,} genes had valid proportions calculated.")

    except Exception as e:
        logger.error(f"[{func_name}] Error during gene proportion calculation: {e}")
        logger.info("--- Function: calculate_gene_proportion_non_identical END (Error) ---")
        return pd.DataFrame(columns=['CDS', 'total_pairs', 'non_identical_pairs', 'proportion_non_identical'])

    if gene_proportion_df.empty:
         logger.warning(f"[{func_name}] No gene proportions could be calculated or survived filtering. Returning empty DataFrame.")
    else:
         logger.info(f"[{func_name}] Final gene proportion DataFrame contains {final_gene_count:,} genes.")

    print("-------------------------------------------------------------------") # Visual separator in output
    logger.info("--- Function: calculate_gene_proportion_non_identical END ---")
    return gene_proportion_df


# --- Analysis 3: LOO Analysis based on GENE PROPORTION Non-Identical ---

def compare_gene_proportions(df):
    """
    Compare distributions of 'proportion_non_identical' between types using Mann-Whitney U.
    (Helper function - logging within is minimal as it's called repeatedly)
    """
    # Logging context will come from the caller (conduct_leave_one_out...)
    if df.empty or not all(col in df.columns for col in ['inversion_type', 'proportion_non_identical']):
        # logger.warning("Input DataFrame for compare_gene_proportions is empty or missing columns.") # Too noisy for LOO
        return {'p_value': np.nan, 'statistic': np.nan, 'median_diff': np.nan, 'rec_median': np.nan, 'single_median': np.nan}

    rec_proportions = df.loc[df['inversion_type'] == 'recurrent', 'proportion_non_identical'].dropna()
    single_proportions = df.loc[df['inversion_type'] == 'single_event', 'proportion_non_identical'].dropna()

    rec_n = len(rec_proportions)
    single_n = len(single_proportions)

    if rec_n < 1 or single_n < 1:
        # Not enough data in one or both groups for the test
        rec_median = rec_proportions.median() if rec_n > 0 else np.nan
        single_median = single_proportions.median() if single_n > 0 else np.nan
        median_diff = single_median - rec_median if rec_n > 0 and single_n > 0 and not (np.isnan(rec_median) or np.isnan(single_median)) else np.nan
        # Return NaN for p-value and statistic to indicate test wasn't run
        return {'p_value': np.nan, 'statistic': np.nan, 'median_diff': median_diff, 'rec_median': rec_median, 'single_median': single_median, 'rec_n': rec_n, 'single_n': single_n}

    try:
        # Using 'greater' means testing if single_event proportions are GREATER than recurrent
        # In the context of "proportion NON-IDENTICAL", this tests if single events are MORE DIVERGENT
        # If you expect RECURRENT to be more divergent (higher proportion non-identical), use alternative='less'
        # Sticking with 'greater' as in the original script.
        u_statistic, p_value = stats.mannwhitneyu(
            rec_proportions,      # Group 1
            single_proportions,   # Group 2
            alternative='greater', # Test H1: single_proportions > rec_proportions
            use_continuity=True
        )
        rec_median = rec_proportions.median()
        single_median = single_proportions.median()
        median_diff = single_median - rec_median

        return {'p_value': p_value, 'statistic': u_statistic, 'median_diff': median_diff, 'rec_median': rec_median, 'single_median': single_median, 'rec_n': rec_n, 'single_n': single_n}

    except ValueError as e:
        # Example: All values might be identical, leading to ConstantInputWarning (ignored) or other issues
        logger.warning(f"Mann-Whitney U test failed: {e}. Groups (Rec N={rec_n}, Single N={single_n}). May indicate identical values.")
        rec_median = rec_proportions.median()
        single_median = single_proportions.median()
        median_diff = single_median - rec_median if not (np.isnan(rec_median) or np.isnan(single_median)) else np.nan
        # Return NaN p-value on error
        return {'p_value': np.nan, 'statistic': np.nan, 'median_diff': median_diff, 'rec_median': rec_median, 'single_median': single_median, 'rec_n': rec_n, 'single_n': single_n}


def conduct_leave_one_out_gene_proportion_analysis(gene_proportion_df, cds_to_type, inversion_to_cds_map):
    """
    Perform leave-one-out analysis based on proportion non-identical per GENE.
    Iterates through ALL inversions that have associated genes REMAINING at this stage.
    Logs filtering steps extensively.
    """
    logger.info("\n--- Function: conduct_leave_one_out_gene_proportion_analysis START ---")
    func_name = "conduct_leave_one_out_gene_proportion_analysis"

    # --- Step 1: Prepare input gene proportion data ---
    logger.info(f"[{func_name}] Preparing gene proportion data for LOO analysis...")
    if gene_proportion_df is None or gene_proportion_df.empty:
        logger.error(f"[{func_name}] Input gene proportion dataframe is empty. Cannot perform LOO analysis.")
        print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Gene Proportion Non-Identical)")
        print("---------------------------------------------------------------")
        print("Skipped: No gene proportion data available.")
        logger.info(f"--- Function: conduct_leave_one_out_gene_proportion_analysis END (No input data) ---")
        return None, np.nan
    initial_gene_count = len(gene_proportion_df)
    logger.info(f"[{func_name}] Received {initial_gene_count:,} genes from `calculate_gene_proportion_non_identical` function.")

    # Filter 1: Map inversion type to genes using the comprehensive `cds_to_type` map
    if 'CDS' not in gene_proportion_df.columns:
         logger.error(f"[{func_name}] Gene proportion DataFrame missing 'CDS' column. Cannot map types.")
         return None, np.nan
    gene_proportion_df['inversion_type'] = gene_proportion_df['CDS'].map(cds_to_type)
    logger.info(f"[{func_name}] Filter 1: Mapped inversion type to {initial_gene_count:,} genes.")
    type_counts_in_genes = gene_proportion_df['inversion_type'].value_counts(dropna=False).to_dict()
    logger.info(f"[{func_name}] Gene counts by mapped type: {type_counts_in_genes}")

    # Filter 2: Keep only genes mapped to 'recurrent' or 'single_event'
    rows_before_type_filter = len(gene_proportion_df)
    valid_df = gene_proportion_df[gene_proportion_df['inversion_type'].isin(['recurrent', 'single_event'])].copy()
    rows_after_type_filter = len(valid_df)
    logger.info(f"[{func_name}] Filter 2: Kept genes with inversion_type 'recurrent' or 'single_event'. Genes remaining: {rows_after_type_filter:,} (removed {rows_before_type_filter - rows_after_type_filter:,})")

    # Filter 3: Ensure the calculated proportion is not NaN (already done in calculation, but double-check)
    rows_before_prop_filter = len(valid_df)
    valid_df = valid_df[pd.notna(valid_df['proportion_non_identical'])]
    rows_after_prop_filter = len(valid_df)
    removed_prop_filter = rows_before_prop_filter - rows_after_prop_filter
    if removed_prop_filter > 0:
        logger.warning(f"[{func_name}] Filter 3: Removed {removed_prop_filter:,} additional genes with NA proportion_non_identical (unexpected). Genes remaining: {rows_after_prop_filter:,}")
    else:
        logger.info(f"[{func_name}] Filter 3: All {rows_after_prop_filter:,} genes have valid proportion_non_identical.")

    final_valid_gene_count = len(valid_df)
    if final_valid_gene_count == 0:
        logger.error(f"[{func_name}] No valid genes found after filtering (type and valid proportion) for LOO analysis. Cannot proceed.")
        print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Gene Proportion Non-Identical)")
        print("---------------------------------------------------------------")
        print("Skipped: No valid gene data for recurrent/single types with calculated proportions.")
        logger.info(f"--- Function: conduct_leave_one_out_gene_proportion_analysis END (No valid genes) ---")
        return None, np.nan

    final_rec_genes = len(valid_df[valid_df['inversion_type'] == 'recurrent'])
    final_single_genes = len(valid_df[valid_df['inversion_type'] == 'single_event'])
    logger.info(f"[{func_name}] Final dataset for LOO analysis (`valid_df`) contains {final_valid_gene_count:,} genes.")
    logger.info(f"[{func_name}]   Breakdown: Recurrent = {final_rec_genes:,}, Single-Event = {final_single_genes:,}")

    # --- Step 2: Baseline analysis (using all valid genes) ---
    logger.info(f"[{func_name}] Performing baseline analysis using all {final_valid_gene_count:,} valid genes...")
    baseline_stats = compare_gene_proportions(valid_df)
    baseline_p = baseline_stats['p_value']
    baseline_effect = baseline_stats['median_diff']
    baseline_rec_n = baseline_stats['rec_n']
    baseline_single_n = baseline_stats['single_n']

    if np.isnan(baseline_p):
         logger.error(f"[{func_name}] Baseline Mann-Whitney U test failed or had insufficient data (Rec N={baseline_rec_n}, Single N={baseline_single_n}). Aborting LOO analysis.")
         print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Gene Proportion Non-Identical)")
         print("---------------------------------------------------------------")
         print("Skipped: Baseline analysis failed or had insufficient data.")
         logger.info(f"--- Function: conduct_leave_one_out_gene_proportion_analysis END (Baseline failed) ---")
         return None, np.nan

    logger.info(f"[{func_name}]   Baseline Results: P-value={baseline_p:.4e}, Effect (Median Diff)={baseline_effect:.4f}, Rec N={baseline_rec_n:,}, Single N={baseline_single_n:,}")
    logger.info(f"[{func_name}]   Baseline Medians: Recurrent={baseline_stats['rec_median']:.4f}, Single={baseline_stats['single_median']:.4f}")

    # --- Step 3: Identify **RELEVANT** inversions for LOO ---
    # An inversion is relevant *only if* it has associated genes THAT ARE PRESENT in `valid_df`.
    logger.info(f"[{func_name}] Identifying inversions relevant for LOO (i.e., inversions whose associated genes are in the `valid_df`)...")
    logger.info(f"[{func_name}]   Starting with {len(inversion_to_cds_map):,} inversions that had *any* gene mapped in `map_cds_to_inversions`.")

    relevant_inversions_for_loo = []
    valid_cds_set = set(valid_df['CDS'].unique()) # Set for faster lookup
    logger.info(f"[{func_name}]   Checking against {len(valid_cds_set):,} unique CDS identifiers present in the `valid_df`.")

    total_inversions_checked = 0
    inversions_kept_for_loo = 0
    inversions_discarded_no_valid_genes = 0

    for inv_id, cds_list in inversion_to_cds_map.items():
        total_inversions_checked += 1
        # Find which genes associated with this inversion are *actually* in our analysis set (`valid_df`)
        genes_for_this_inv_in_valid_df = [cds for cds in cds_list if cds in valid_cds_set]

        if genes_for_this_inv_in_valid_df:
            # This inversion has genes relevant to the current analysis.
            # Determine the type of this inversion based on its VALID genes.
            inv_genes_df = valid_df[valid_df['CDS'].isin(genes_for_this_inv_in_valid_df)]
            if not inv_genes_df.empty:
                 # Use mode() in case an inversion somehow got linked to genes of different types in valid_df (unlikely but safe)
                 inv_type_mode = inv_genes_df['inversion_type'].mode()
                 inv_type = inv_type_mode[0] if len(inv_type_mode) > 0 else 'unknown_mix' # Assign a type

                 if inv_type in ['recurrent', 'single_event']:
                     # This inversion is relevant and has a clear type based on the genes used in the analysis
                     relevant_inversions_for_loo.append({
                         'inversion_id': inv_id,
                         'gene_count': len(genes_for_this_inv_in_valid_df), # Count of *valid* genes
                         'inv_type': inv_type
                     })
                     inversions_kept_for_loo += 1
                     # Log first few keeps for detail
                     if inversions_kept_for_loo <= 5:
                         logger.info(f"[{func_name}]     -> Keeping Inv {inv_id} (type: {inv_type}) for LOO. Has {len(genes_for_this_inv_in_valid_df)} valid genes associated.")
                     elif inversions_kept_for_loo == 6:
                         logger.info(f"[{func_name}]     -> (Further 'keeping' logs suppressed)")

                 else:
                     # This inversion's valid genes are of mixed or unknown type (shouldn't happen with current filters)
                     logger.warning(f"[{func_name}]     -> Discarding Inv {inv_id}. Its valid genes have unclear type: {inv_type}. Valid genes: {len(genes_for_this_inv_in_valid_df)}")
                     inversions_discarded_no_valid_genes += 1 # Count as discarded for this reason
            else:
                 # This case (genes_for_this_inv_in_valid_df is not empty, but inv_genes_df is) seems impossible. Log if it occurs.
                 logger.error(f"[{func_name}]     -> LOGIC ERROR: Inv {inv_id} had valid genes listed but subset was empty.")
                 inversions_discarded_no_valid_genes += 1
        else:
            # This inversion has NO associated genes remaining in `valid_df`. It's irrelevant for LOO.
            # Log first few discards for detail
            if inversions_discarded_no_valid_genes < 5:
                logger.info(f"[{func_name}]     -> Discarding Inv {inv_id} from LOO. Associated genes ({len(cds_list)}) are not in the final valid gene set (`valid_df`).")
            elif inversions_discarded_no_valid_genes == 5:
                 logger.info(f"[{func_name}]     -> (Further 'discarding' logs suppressed)")
            inversions_discarded_no_valid_genes += 1

    logger.info(f"[{func_name}] Finished checking {total_inversions_checked:,} inversions from the map.")
    logger.info(f"[{func_name}] *** IDENTIFIED {inversions_kept_for_loo:,} RELEVANT INVERSIONS FOR LOO ANALYSIS ***")
    logger.info(f"[{func_name}]   (Discarded {inversions_discarded_no_valid_genes:,} inversions because their associated genes were filtered out before LOO)")
    logger.info(f"[{func_name}] --- THIS EXPLAINS WHY THE NUMBER OF LOO RUNS IS {inversions_kept_for_loo} ---")


    if not relevant_inversions_for_loo:
        logger.warning(f"[{func_name}] No inversions found to be relevant for LOO analysis (none had genes remaining in the valid set). Skipping LOO iterations.")
        # Still save the baseline result if available
        results_df = pd.DataFrame([
            {
                'inversion_excluded': 'None', 'inv_type': 'NA', 'excluded_gene_count': 0,
                'rec_median_prop': baseline_stats['rec_median'], 'single_median_prop': baseline_stats['single_median'],
                'p_value': baseline_p, 'statistic': baseline_stats['statistic'], 'effect_size': baseline_effect,
                'rec_n_remain': baseline_rec_n, 'single_n_remain': baseline_single_n
             }
        ])
        try:
            results_df.to_csv(OUTPUT_RESULTS_LOO_GENE_PROPORTION, index=False)
            logger.info(f"[{func_name}] Baseline-only results saved to {OUTPUT_RESULTS_LOO_GENE_PROPORTION}")
        except Exception as e:
            logger.error(f"[{func_name}] Failed to save baseline-only LOO results: {e}")
        logger.info(f"--- Function: conduct_leave_one_out_gene_proportion_analysis END (No relevant inversions for LOO) ---")
        return results_df, baseline_p # Return baseline p-value

    # Create DataFrame of relevant inversions and sort (optional, but nice for consistency)
    all_relevant_inversions_df = pd.DataFrame(relevant_inversions_for_loo)
    all_relevant_inversions_df = all_relevant_inversions_df.sort_values('gene_count', ascending=False)
    logger.info(f"[{func_name}] Top 5 relevant inversions by associated valid gene count:")
    for _, row in all_relevant_inversions_df.head(5).iterrows():
        logger.info(f"[{func_name}]     - Inv {row['inversion_id']} ({row['inv_type']}), Valid Genes: {row['gene_count']}")


    # --- Step 4: Perform LOO iterations ---
    logger.info(f"[{func_name}] Starting {len(all_relevant_inversions_df):,} LOO iterations...")
    results = []
    all_p_values = []
    all_effect_sizes = []

    # Add baseline result first
    baseline_result_row = {
        'inversion_excluded': 'None',
        'inv_type': 'NA',
        'excluded_gene_count': 0,
        'rec_median_prop': baseline_stats['rec_median'],
        'single_median_prop': baseline_stats['single_median'],
        'p_value': baseline_p,
        'statistic': baseline_stats['statistic'],
        'effect_size': baseline_effect,
        'rec_n_remain': baseline_rec_n,
        'single_n_remain': baseline_single_n
    }
    results.append(baseline_result_row)
    all_p_values.append(baseline_p)
    if not np.isnan(baseline_effect):
        all_effect_sizes.append(baseline_effect)

    loo_iterations_run = 0
    loo_iterations_skipped = 0

    for _, row in all_relevant_inversions_df.iterrows():
        inv_id = row['inversion_id']
        inv_type_of_excluded = row['inv_type']
        # Get the FULL list of genes associated with this inversion from the original map
        # We need this to ensure we exclude ALL genes linked to it, even if some weren't in valid_df initially
        # Although, we previously established it MUST have *some* genes in valid_df to be in this loop.
        cds_list_to_exclude_potentially = inversion_to_cds_map.get(inv_id, [])
        if not cds_list_to_exclude_potentially:
             logger.warning(f"[{func_name}] LOO Iteration: Inv {inv_id} is relevant, but no CDS found in `inversion_to_cds_map`. Skipping.")
             loo_iterations_skipped += 1
             continue

        # Exclude these specific CDS from our analysis set (`valid_df`)
        filtered_df = valid_df[~valid_df['CDS'].isin(cds_list_to_exclude_potentially)].copy()
        excluded_genes_count_actual = final_valid_gene_count - len(filtered_df) # How many were actually removed from valid_df

        # Check if enough data remains in BOTH groups
        filtered_stats = compare_gene_proportions(filtered_df) # This calculates Ns too
        rec_remaining = filtered_stats['rec_n']
        single_remaining = filtered_stats['single_n']

        if filtered_df.empty or rec_remaining < 1 or single_remaining < 1:
            logger.warning(f"[{func_name}] LOO Iteration: Excluding Inv {inv_id} ({inv_type_of_excluded}, {excluded_genes_count_actual} valid genes removed) left insufficient data (Rec={rec_remaining}, Single={single_remaining}). Skipping this iteration.")
            loo_iterations_skipped += 1
            continue

        # Recalculate stats on the filtered data
        current_p = filtered_stats['p_value']
        current_effect = filtered_stats['median_diff']

        if np.isnan(current_p):
            logger.warning(f"[{func_name}] LOO Iteration: Mann-Whitney U failed after excluding Inv {inv_id}. Skipping result for this inversion.")
            loo_iterations_skipped += 1
            continue

        # Store results
        result = {
            'inversion_excluded': inv_id,
            'inv_type': inv_type_of_excluded,
            'excluded_gene_count': excluded_genes_count_actual,
            'rec_median_prop': filtered_stats['rec_median'],
            'single_median_prop': filtered_stats['single_median'],
            'p_value': current_p,
            'statistic': filtered_stats['statistic'],
            'effect_size': current_effect,
            'rec_n_remain': rec_remaining,
            'single_n_remain': single_remaining
        }
        results.append(result)
        all_p_values.append(current_p)
        if not np.isnan(current_effect):
            all_effect_sizes.append(current_effect)
        loo_iterations_run += 1
        # Log progress occasionally
        if loo_iterations_run % 10 == 0:
             logger.info(f"[{func_name}]   Completed {loo_iterations_run}/{len(all_relevant_inversions_df)} LOO iterations...")


    logger.info(f"[{func_name}] Completed LOO iterations.")
    logger.info(f"[{func_name}]   Successful iterations run: {loo_iterations_run:,}")
    logger.info(f"[{func_name}]   Iterations skipped (insufficient data/test failure): {loo_iterations_skipped:,}")
    total_attempted_loo = loo_iterations_run + loo_iterations_skipped
    logger.info(f"[{func_name}]   (Total relevant inversions identified earlier: {len(all_relevant_inversions_df):,}. Attempted: {total_attempted_loo:,})") # Should match

    # --- Step 5: Compile, Save, and Report Results ---
    logger.info(f"[{func_name}] Compiling and reporting LOO results...")
    if not results:
        logger.error(f"[{func_name}] No results (not even baseline) were generated. Cannot report.")
        print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Gene Proportion Non-Identical)")
        print("---------------------------------------------------------------")
        print("Skipped: No results generated (baseline or LOO).")
        logger.info(f"--- Function: conduct_leave_one_out_gene_proportion_analysis END (No results) ---")
        return None, np.nan

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value') # Sort by p-value ascending

    try:
        results_df.to_csv(OUTPUT_RESULTS_LOO_GENE_PROPORTION, index=False)
        logger.info(f"[{func_name}] Leave-one-out results saved to {OUTPUT_RESULTS_LOO_GENE_PROPORTION}")
    except Exception as e:
        logger.error(f"[{func_name}] Failed to save leave-one-out gene proportion results: {e}")

    # --- Reporting to Console ---
    print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Proportion Non-Identical per GENE - All Inversions)")
    print("----------------------------------------------------------------------------------")
    print(f"Comparison method: Mann-Whitney U test (Alternative='greater')")
    print(f"  (Testing if Single-Event median proportion non-identical > Recurrent median proportion non-identical)")
    print(f"Metric: Proportion of non-identical (omega != -1) pairs within each gene.")
    print(f"Total unique genes in baseline analysis: {final_valid_gene_count:,} (Rec: {final_rec_genes:,}, Single: {final_single_genes:,})")
    print(f"Total relevant inversions identified for LOO: {len(all_relevant_inversions_df):,}")
    print(f"Successful LOO iterations performed: {loo_iterations_run:,}")
    if loo_iterations_skipped > 0:
        print(f"LOO iterations skipped due to insufficient data/test failure: {loo_iterations_skipped:,}")

    if not all_p_values: # Should only happen if baseline failed AND no LOO runs succeeded
        print("\nNo valid LOO P-values generated to summarize.")
        logger.info(f"--- Function: conduct_leave_one_out_gene_proportion_analysis END (No P-values) ---")
        return results_df, np.nan # Return DF (maybe baseline only) but NaN P-value

    # Calculate robust stats from all successful runs (including baseline)
    min_p_value = np.nanmin(all_p_values) # Use nanmin/nanmedian
    median_p_value = np.nanmedian(all_p_values)
    max_p_value = np.nanmax(all_p_values)
    # Calculate median effect size only from non-nan effects
    valid_effect_sizes = [e for e in all_effect_sizes if not np.isnan(e)]

    print(f"\nBaseline Result (All {final_valid_gene_count} Valid Genes):")
    print(f"  Mann-Whitney U P-value: {baseline_p:.4e}")
    print(f"  Effect size (Median Prop Non-Identical: Single - Recurrent): {baseline_effect:.4f}")
    print(f"  Median Proportion Non-Identical: Recurrent={baseline_result_row['rec_median_prop']:.4f}, Single={baseline_result_row['single_median_prop']:.4f}")
    print(f"  Group Sizes: Recurrent N={baseline_rec_n}, Single N={baseline_single_n}")


    print(f"\nRobust Overall Results (Stats from Baseline + {loo_iterations_run} successful LOO iterations):")
    print(f"  MINIMUM P-VALUE OBSERVED: {min_p_value:.4e}")
    print(f"  ROBUST MEDIAN P-VALUE: {median_p_value:.4e}")
    print(f"  MAXIMUM P-VALUE OBSERVED (Most Conservative): {max_p_value:.4e}")

    # Display top results from the sorted DataFrame
    print("\nTop 5 results by p-value (including Baseline):")
    top5 = results_df.head(5)
    for i, row in top5.iterrows():
        excluded = row['inversion_excluded']
        p_val = row['p_value']
        effect = row['effect_size']
        rec_med = row['rec_median_prop']
        sing_med = row['single_median_prop']
        excluded_count_disp = int(row['excluded_gene_count'])
        rec_n_rem = int(row['rec_n_remain'])
        sing_n_rem = int(row['single_n_remain'])

        if excluded == 'None':
            print(f"  BASELINE: p={p_val:.4e}, RecMed={rec_med:.4f}, SingMed={sing_med:.4f}, Effect={effect:.4f} (N={rec_n_rem}+{sing_n_rem} genes)")
        else:
            inv_type = row['inv_type']
            print(f"  Excl. {inv_type} Inv {excluded} ({excluded_count_disp:,} gene): p={p_val:.4e}, RecMed={rec_med:.4f}, SingMed={sing_med:.4f}, Effect={effect:.4f} (N={rec_n_rem}+{sing_n_rem})")

    # Display most influential inversions
    print("\nMost influential inversions (largest absolute p-value change from baseline):")
    if not np.isnan(baseline_p):
        # Ensure p_value is numeric, coerce errors just in case
        results_df['p_value'] = pd.to_numeric(results_df['p_value'], errors='coerce')
        results_df_valid_p = results_df.dropna(subset=['p_value']).copy() # Work on copy with valid p-values

        if not results_df_valid_p.empty and len(results_df_valid_p) > 1: # Need >1 row to calculate change
             # Calculate absolute change from baseline p-value
             results_df_valid_p['p_change'] = results_df_valid_p['p_value'] - baseline_p
             results_df_valid_p['abs_p_change'] = results_df_valid_p['p_change'].abs()

             # Get rows excluding baseline, sort by absolute change descending, take top 5
             top_influence = results_df_valid_p[results_df_valid_p['inversion_excluded'] != 'None'].sort_values('abs_p_change', ascending=False).head(5)

             if not top_influence.empty:
                  print(f"  (Showing top {len(top_influence)} influencers with largest absolute p-value change)")
                  for i, row in top_influence.iterrows():
                    excluded = row['inversion_excluded']
                    inv_type = row['inv_type']
                    p_val = row['p_value']
                    p_change = row['p_change']
                    count = int(row['excluded_gene_count'])
                    effect = row['effect_size']
                    rec_n_rem = int(row['rec_n_remain'])
                    sing_n_rem = int(row['single_n_remain'])
                    change_direction = "increased" if p_change > 0 else "decreased"
                    print(f"    Inv {excluded} ({inv_type}, {count:,} gene): p-value {change_direction} by {abs(p_change):.4e} to {p_val:.4e}. Effect={effect:.4f} (N={rec_n_rem}+{sing_n_rem})")
             else:
                  print("  No influential inversions found (excluding baseline).")
        else:
             print("  Could not calculate influence (only baseline result or no valid p-values available).")
    else:
        print("  Cannot calculate influence as baseline p-value is invalid.")

    # Final conclusion based on robust median p-value
    alpha = 0.05
    significance = "SIGNIFICANT" if not np.isnan(median_p_value) and median_p_value < alpha else "NO significant"
    direction_conclusion = ""

    print(f"\nFINAL CONCLUSION (Proportion Non-Identical per GENE LOO - All Inversions):")
    print(f"  Using Mann-Whitney U (alternative='greater'), there is {significance} evidence")
    print(f"  that the distribution of the proportion of non-identical pairs per gene differs")
    print(f"  between recurrent and single-event associated groups (Robust Median p={median_p_value:.4e}, α={alpha}).")
    print("----------------------------------------------------------------------------------")
    logger.info("--- Function: conduct_leave_one_out_gene_proportion_analysis END ---")

    return results_df, median_p_value


# --- Main Execution ---

def main():
    """Main execution function."""
    logger.info("======== Starting Conservation Analysis Script ========")

    # --- Load Data ---
    try:
        logger.info(f"Loading pairwise data from: {PAIRWISE_FILE}")
        pairwise_df = pd.read_csv(PAIRWISE_FILE)
        logger.info(f"  Loaded {len(pairwise_df):,} rows from {PAIRWISE_FILE}.")

        logger.info(f"Loading inversion info from: {INVERSION_FILE}")
        inversion_df = pd.read_csv(INVERSION_FILE, sep='\t')
        logger.info(f"  Loaded {len(inversion_df):,} rows (inversions) from {INVERSION_FILE}.")
        # Display head of inversion data for context
        logger.info(f"  Inversion data columns: {inversion_df.columns.tolist()}")
        logger.info(f"  First 5 rows of inversion data:\n{inversion_df.head().to_string()}")
        # Count initial potential types from inversion file
        if '0_single_1_recur' in inversion_df.columns:
            raw_inv_type_counts = inversion_df['0_single_1_recur'].value_counts(dropna=False).to_dict()
            logger.info(f"  Initial type counts in {INVERSION_FILE} (0=Single, 1=Recurrent, NaN=Unknown): {raw_inv_type_counts}")
        else:
            logger.warning(f" Column '0_single_1_recur' not found in {INVERSION_FILE} for initial type count.")


    except FileNotFoundError as e:
        logger.error(f"FATAL: Error loading input file: {e}. Please check file paths.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: An unexpected error occurred loading input files: {e}")
        sys.exit(1)

    # --- Map CDS to Inversions ---
    # This function now contains detailed logging about filtering
    cds_to_type, cds_to_inversion_id, inversion_to_cds_map = map_cds_to_inversions(pairwise_df, inversion_df)

    if not cds_to_type or not inversion_to_cds_map:
         logger.error("FATAL: CDS to type mapping failed or resulted in no mappings/inversions with genes. Exiting.")
         sys.exit(1)
    logger.info(f"Mapping complete. cds_to_type map size: {len(cds_to_type)}, inversion_to_cds_map size: {len(inversion_to_cds_map)}")

    # --- Analysis 1: Raw Pairwise Proportions ---
    # This function now contains detailed logging about filtering
    analyze_raw_pair_proportions(pairwise_df.copy(), cds_to_type)

    # --- Analysis 2: Calculate Gene Proportions ---
    # This function now contains detailed logging about filtering
    gene_proportion_df = calculate_gene_proportion_non_identical(pairwise_df.copy()) # Pass copy

    # --- Analysis 3: Leave-One-Out Gene Proportions ---
    final_gene_prop_loo_p_value = np.nan # Initialize
    loo_gene_prop_results_df = None # Initialize
    if gene_proportion_df is not None and not gene_proportion_df.empty:
        # This function now explains how the relevant inversions for LOO are selected
        loo_gene_prop_results_df, final_gene_prop_loo_p_value = conduct_leave_one_out_gene_proportion_analysis(
            gene_proportion_df.copy(), cds_to_type, inversion_to_cds_map
        )
    else:
        logger.warning("Skipping Leave-One-Out (Gene Proportion) analysis because the input gene proportion DataFrame is empty or None.")

    # --- Summary ---
    logger.info("\n======== Analysis Summary ========")
    logger.info(f"1. Raw Pairwise Proportion Analysis (Fisher's Exact):")
    logger.info(f"   - Input: {len(pairwise_df):,} pairwise rows.")

    logger.info(f"\n2. Proportion Non-Identical Pairs per Gene Calculation:")
    if gene_proportion_df is not None and not gene_proportion_df.empty:
        logger.info(f"   - Input: {len(pairwise_df):,} pairwise rows.")
        logger.info(f"   - Filtering steps detailed in logs above.")
        logger.info(f"   - Output: DataFrame `gene_proportion_df` with {len(gene_proportion_df):,} genes.")
    else:
        logger.info(f"   - Failed or resulted in an empty DataFrame.")

    logger.info(f"\n3. Leave-One-Out Analysis (GENE Proportion Non-Identical - Mann-Whitney U):")
    if loo_gene_prop_results_df is not None:
        logger.info(f"   - Input: `gene_proportion_df` ({len(gene_proportion_df) if gene_proportion_df is not None else 'N/A'} genes), `cds_to_type`, `inversion_to_cds_map`.")
        logger.info(f"   - Results saved to: {OUTPUT_RESULTS_LOO_GENE_PROPORTION}")
        logger.info(f"   - Robust Median P-value: {final_gene_prop_loo_p_value:.4e}" if not np.isnan(final_gene_prop_loo_p_value) else "N/A")
    else:
        logger.info(f"   - Skipped or failed (Input `gene_proportion_df` was likely empty).")

    logger.info("\n======== Script Finished ========")

if __name__ == "__main__":
    main()
