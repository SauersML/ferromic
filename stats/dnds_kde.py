import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import re
import logging
import sys
from pathlib import Path # Use pathlib for paths

# --- Configuration ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('median_omega_analysis_standalone')

# File paths
PAIRWISE_FILE = Path('all_pairwise_results.csv')
INVERSION_FILE = Path('inv_info.csv')
OUTPUT_PLOT_PATH = Path('median_omega_distribution_standalone.png')

# Plotting Style Adjustments
plt.rcParams.update({'font.size': 14}) # base font size for better readability

RECURRENT_COLOR = 'salmon'
SINGLE_EVENT_COLOR = 'skyblue'
RECURRENT_HATCH = '//'
SINGLE_EVENT_HATCH = '..'

# --- Helper Functions (Mostly Unchanged) ---

def extract_coordinates_from_cds(cds_string):
    """Extract genomic coordinates from CDS string."""
    if not isinstance(cds_string, str): # Handle potential non-string input
        logger.warning(f"Invalid CDS input type: {type(cds_string)}. Skipping.")
        return None
    pattern = r'(chr[\w\.]+)_start(\d+)_end(\d+)' # Adjusted pattern for chr names like chr1.1
    match = re.search(pattern, cds_string, re.IGNORECASE) # Ignore case for 'chr'
    if match:
        return {
            'chrom': match.group(1).lower(), # Normalize chromosome name to lower case
            'start': int(match.group(2)),
            'end': int(match.group(3))
        }
    # Add more specific logging if needed
    # logger.warning(f"Failed to extract coordinates from CDS string: {cds_string}")
    return None

def load_input_files():
    """Load input files and perform basic validation."""
    try:
        if not PAIRWISE_FILE.is_file():
            logger.error(f"Pairwise results file not found: {PAIRWISE_FILE}")
            return None, None
        logger.info(f"Loading pairwise results from {PAIRWISE_FILE}")
        pairwise_df = pd.read_csv(PAIRWISE_FILE)

        if not INVERSION_FILE.is_file():
            logger.error(f"Inversion info file not found: {INVERSION_FILE}")
            return None, None
        logger.info(f"Loading inversion info from {INVERSION_FILE}")
        inversion_df = pd.read_csv(INVERSION_FILE)

        logger.info(f"Pairwise results: {pairwise_df.shape[0]} rows, {pairwise_df.shape[1]} columns")
        logger.info(f"Inversion info: {inversion_df.shape[0]} rows, {inversion_df.shape[1]} columns")

        # --- Data Validation and Cleaning ---
        # Ensure required columns exist
        required_pairwise_cols = ['CDS', 'Seq1', 'Seq2', 'omega', 'Group1', 'Group2']
        if not all(col in pairwise_df.columns for col in required_pairwise_cols):
            logger.error(f"Pairwise DF missing required columns. Need: {required_pairwise_cols}")
            return None, None

        required_inversion_cols = ['Chromosome', 'Start', 'End', '0_single_1_recur']
        if not all(col in inversion_df.columns for col in required_inversion_cols):
            logger.error(f"Inversion DF missing required columns. Need: {required_inversion_cols}")
            return None, None

        # Convert inversion columns to correct types
        inversion_df['Chromosome'] = inversion_df['Chromosome'].astype(str).str.lower() # Normalize chromosome names
        inversion_df['Start'] = pd.to_numeric(inversion_df['Start'], errors='coerce')
        inversion_df['End'] = pd.to_numeric(inversion_df['End'], errors='coerce')
        inversion_df['0_single_1_recur'] = pd.to_numeric(inversion_df['0_single_1_recur'], errors='coerce')

        # Drop rows with invalid numeric data in inversions
        inversion_df = inversion_df.dropna(subset=['Start', 'End', '0_single_1_recur'])
        inversion_df['Start'] = inversion_df['Start'].astype(int)
        inversion_df['End'] = inversion_df['End'].astype(int)
        inversion_df['0_single_1_recur'] = inversion_df['0_single_1_recur'].astype(int)

        # Convert omega to numeric, coercing errors (like non-numeric values) to NaN
        pairwise_df['omega'] = pd.to_numeric(pairwise_df['omega'], errors='coerce')

        # <<<--- Explicitly Drop Rows with NaN omega BEFORE filtering 99 --->>>
        initial_rows = len(pairwise_df)
        pairwise_df = pairwise_df.dropna(subset=['omega'])
        dropped_nan = initial_rows - len(pairwise_df)
        if dropped_nan > 0:
            logger.info(f"Dropped {dropped_nan} rows with non-numeric or NaN omega values.")

        # <<<--- Filter out omega == 99 EARLIER --->>>
        initial_rows = len(pairwise_df)
        pairwise_df = pairwise_df[pairwise_df['omega'] != 99].copy()
        dropped_99 = initial_rows - len(pairwise_df)
        if dropped_99 > 0:
             logger.info(f"Filtered out {dropped_99} rows where omega == 99.")

        return pairwise_df, inversion_df

    except Exception as e:
        logger.error(f"Error loading or validating input files: {e}", exc_info=True)
        return None, None

def map_cds_to_inversions_excluding_ambiguous(pairwise_df, inversion_df):
    """
    Map CDS strings to inversion types (0 or 1), excluding any CDS that map to both.
    Uses coordinate overlap.
    """
    logger.info("Mapping CDS regions to inversion types (excluding ambiguous overlaps)...")

    # Extract unique CDS strings and attempt coordinate extraction
    unique_cds = pairwise_df['CDS'].unique()
    logger.info(f"Found {len(unique_cds)} unique CDS strings to map.")

    cds_coords = {}
    failed_coord_extraction = 0
    for cds in unique_cds:
        coords = extract_coordinates_from_cds(cds)
        if coords:
            cds_coords[cds] = coords
        else:
            failed_coord_extraction += 1

    if failed_coord_extraction > 0:
         logger.warning(f"Failed to extract coordinates for {failed_coord_extraction} unique CDS strings. These will not be mapped.")
    logger.info(f"Successfully extracted coordinates for {len(cds_coords)} unique CDS strings.")
    if not cds_coords:
         logger.error("No valid CDS coordinates extracted. Cannot proceed with mapping.")
         return {}

    # Prepare inversion dataframes (already cleaned in load_input_files)
    recurrent_inv = inversion_df[inversion_df['0_single_1_recur'] == 1].copy()
    single_event_inv = inversion_df[inversion_df['0_single_1_recur'] == 0].copy()
    logger.info(f"Using {len(recurrent_inv)} recurrent and {len(single_event_inv)} single-event inversion regions for mapping.")

    # Map CDS to inversion types
    recurrent_cds = set()
    single_event_cds = set()
    ambiguous_cds = set()
    unmapped_cds_no_overlap = 0

    for cds, coords in cds_coords.items():
        chrom = coords['chrom']
        start = coords['start']
        end = coords['end']

        # Optimized overlap check
        rec_matches = recurrent_inv[
            (recurrent_inv['Chromosome'] == chrom) &
            (recurrent_inv['Start'] < end) & # Inversion starts before CDS ends
            (recurrent_inv['End'] > start)    # Inversion ends after CDS starts
        ]

        single_matches = single_event_inv[
            (single_event_inv['Chromosome'] == chrom) &
            (single_event_inv['Start'] < end) &
            (single_event_inv['End'] > start)
        ]

        is_recurrent = len(rec_matches) > 0
        is_single = len(single_matches) > 0

        if is_recurrent and not is_single:
            recurrent_cds.add(cds)
        elif is_single and not is_recurrent:
            single_event_cds.add(cds)
        elif is_recurrent and is_single:
            ambiguous_cds.add(cds)
        else:
            unmapped_cds_no_overlap += 1 # CDS did not overlap any inversion region

    # Report mapping results
    logger.info(f"CDS mapping results: {len(recurrent_cds)} mapped to recurrent, {len(single_event_cds)} mapped to single-event.")
    logger.info(f"Excluded: {len(ambiguous_cds)} ambiguous overlaps, {unmapped_cds_no_overlap} with no overlap.")
    total_mapped = len(recurrent_cds) + len(single_event_cds)
    if total_mapped == 0:
        logger.warning("No CDS were successfully mapped to either inversion type. Check coordinate formats and overlap logic.")

    # Create mapping dictionary (string keys for clarity in merge/map)
    cds_to_type = {}
    for cds in recurrent_cds:
        cds_to_type[cds] = 'Recurrent' # Use descriptive strings
    for cds in single_event_cds:
        cds_to_type[cds] = 'Single-event' # Use descriptive strings

    return cds_to_type

# <<<--- Modified Function to Filter 99 during Calculation --->>>
def calculate_sequence_median_omega(pairwise_df):
    """
    Calculate median omega value for each unique sequence in each CDS.
    Only uses GROUP 1 pairs. Filters out omega == 99 during aggregation.
    """
    logger.info("Calculating median omega for each sequence (using Group 1 pairs only)...")

    # Apply GROUP 1 filtering
    group1_df = pairwise_df[(pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1)].copy()
    if group1_df.empty:
        logger.warning("No rows found where Group1=1 and Group2=1. Cannot calculate median omega.")
        return pd.DataFrame(columns=['CDS', 'Sequence', 'median_omega']) # Return empty df
    logger.info(f"Processing {group1_df.shape[0]} Group 1 pairwise comparisons.")

    # Use defaultdict for cleaner aggregation
    from collections import defaultdict
    sequence_omega_values = defaultdict(list) # {(cds, sequence): [omega values]}

    # Process each row in the GROUP 1 dataframe
    processed_rows = 0
    for _, row in group1_df.iterrows():
        # Skip rows with NaN omega values (already done in load, but double-check)
        omega = row['omega']
        if pd.isna(omega):
            continue
        # Skip omega = 99 (already done in load, but good practice)
        # if omega == 99:
        #     continue

        # Add omega value for each sequence in this pair
        cds = row['CDS']
        seq1 = row['Seq1']
        seq2 = row['Seq2']

        sequence_omega_values[(cds, seq1)].append(omega)
        sequence_omega_values[(cds, seq2)].append(omega)
        processed_rows += 1

    logger.info(f"Aggregated omega values from {processed_rows} valid Group 1 comparisons.")

    # Calculate median for each sequence
    sequence_median_omega = {}
    calculated_medians = 0
    for (cds, seq), omega_list in sequence_omega_values.items():
        if omega_list: # Ensure list is not empty
            sequence_median_omega[(cds, seq)] = np.median(omega_list)
            calculated_medians += 1

    logger.info(f"Calculated median omega for {calculated_medians} unique CDS-sequence pairs.")

    # Create a dataframe for analysis
    if not sequence_median_omega:
        logger.warning("No median omega values could be calculated.")
        return pd.DataFrame(columns=['CDS', 'Sequence', 'median_omega'])

    median_data = []
    for (cds, seq), median_omega in sequence_median_omega.items():
        median_data.append({
            'CDS': cds,
            'Sequence': seq,
            'median_omega': median_omega
        })

    median_df = pd.DataFrame(median_data)
    logger.info(f"Created median omega dataframe with {len(median_df)} rows.")

    return median_df

# <<<--- Modified Function to reflect pre-filtering of 99 --->>>
def categorize_median_omega_values(median_df, cds_to_type):
    """
    Categorize median omega values into three categories based on mapped inversion type.
    Expects omega=99 to be already filtered out.
    """
    if median_df.empty:
        logger.warning("Median omega dataframe is empty. Cannot categorize.")
        return {'Recurrent': {}, 'Single-event': {}}

    logger.info("Categorizing median omega values by inversion type...")

    # Add inversion type column using the mapping
    median_df['inversion_type'] = median_df['CDS'].map(cds_to_type)

    # Drop rows where CDS couldn't be mapped to an inversion type
    initial_rows = len(median_df)
    median_df_mapped = median_df.dropna(subset=['inversion_type']).copy()
    dropped_unmapped = initial_rows - len(median_df_mapped)
    if dropped_unmapped > 0:
        logger.info(f"Excluded {dropped_unmapped} median omega rows corresponding to unmapped/ambiguous CDS.")

    if median_df_mapped.empty:
        logger.warning("No median omega values remain after filtering for mapped inversion types.")
        return {'Recurrent': {}, 'Single-event': {}}

    # Get rows for each inversion type
    recurrent_df = median_df_mapped[median_df_mapped['inversion_type'] == 'Recurrent']
    single_event_df = median_df_mapped[median_df_mapped['inversion_type'] == 'Single-event']

    logger.info(f"Median omega rows available for categorization: {len(recurrent_df)} Recurrent, {len(single_event_df)} Single-event.")

    # Define function to count categories
    def count_omega_categories(df, type_name):
        """Count median omega values in each category."""
        if df.empty:
            logger.info(f"No data for {type_name} type to categorize.")
            return {"Exactly -1": 0, "0 to 1": 0, "Above 1": 0}

        # Ensure median_omega is numeric, handling potential NaNs introduced during merge/map
        df = df.dropna(subset=['median_omega'])
        if df.empty:
             logger.info(f"No non-NaN median_omega data for {type_name} type.")
             return {"Exactly -1": 0, "0 to 1": 0, "Above 1": 0}

        minus_one = (df['median_omega'] == -1).sum()
        # Note: >= 0 automatically excludes -1
        zero_to_one = ((df['median_omega'] >= 0) & (df['median_omega'] <= 1)).sum()
        above_one = (df['median_omega'] > 1).sum()

        # Verify all rows are accounted for (should be, as 99 is pre-filtered)
        total_counted = minus_one + zero_to_one + above_one
        if total_counted != len(df):
            logger.warning(f"Row count mismatch for {type_name}: {total_counted} categorized vs {len(df)} total rows. Check for unexpected omega values.")

        return {
            "Exactly -1": minus_one,
            "0 to 1": zero_to_one,
            "Above 1": above_one
        }

    # Count categories for each type
    recurrent_counts = count_omega_categories(recurrent_df, 'Recurrent')
    single_event_counts = count_omega_categories(single_event_df, 'Single-event')

    # Report counts
    logger.info("Recurrent median omega category counts:")
    for category, count in recurrent_counts.items():
        logger.info(f"  {category}: {count}")

    logger.info("Single-event median omega category counts:")
    for category, count in single_event_counts.items():
        logger.info(f"  {category}: {count}")

    return {
        'Recurrent': recurrent_counts,
        'Single-event': single_event_counts
    }


# <<<--- New Standalone Plotting Function --->>>
def plot_median_omega_distribution(median_count_data):
    """
    Create a standalone bar plot showing the percentage distribution of
    median omega values for Recurrent vs. Single-event inversion types.
    """
    logger.info("Creating standalone plot for median omega distribution...")

    # Define categories for display and internal lookup
    display_categories = ["Identical Sequences", "0 ≤ ω ≤ 1", "ω > 1"]
    internal_categories = ["Exactly -1", "0 to 1", "Above 1"]
    n_categories = len(display_categories)

    # Extract counts, ensuring keys exist
    rec_counts = median_count_data.get('Recurrent', {})
    single_counts = median_count_data.get('Single-event', {})

    rec_values = [rec_counts.get(cat, 0) for cat in internal_categories]
    single_values = [single_counts.get(cat, 0) for cat in internal_categories]

    # Calculate totals and percentages
    rec_total = sum(rec_values)
    single_total = sum(single_values)

    if rec_total == 0 and single_total == 0:
        logger.error("No data available for plotting in either category. Skipping plot generation.")
        return None

    rec_percentages = [(v / rec_total * 100) if rec_total > 0 else 0 for v in rec_values]
    single_percentages = [(v / single_total * 100) if single_total > 0 else 0 for v in single_values]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7)) # Adjusted figure size for better spacing

    x = np.arange(n_categories) # the label locations
    width = 0.35                # the width of the bars

    # Plot bars
    rects1 = ax.bar(x - width/2, rec_percentages, width,
                  label=f'Recurrent (N={rec_total})', # N in legend
                  color=RECURRENT_COLOR, hatch=RECURRENT_HATCH, alpha=0.85, edgecolor='grey')
    rects2 = ax.bar(x + width/2, single_percentages, width,
                  label=f'Single-event (N={single_total})', # N in legend
                  color=SINGLE_EVENT_COLOR, hatch=SINGLE_EVENT_HATCH, alpha=0.85, edgecolor='grey')

    # Add percentage labels above bars
    # <<<--- Using ax.bar_label for cleaner code --->>>
    ax.bar_label(rects1, fmt='%.1f%%', padding=3, fontsize=11) # Add '%' sign
    ax.bar_label(rects2, fmt='%.1f%%', padding=3, fontsize=11) # Add '%' sign

    # Add labels, title, and ticks
    ax.set_ylabel('Percentage of Sequences (%)', fontsize=14) # Slightly larger axis label
    ax.set_title('Distribution of Median Sequence ω by Inversion Type', fontsize=16, pad=15) # Larger title
    ax.set_xticks(x)
    ax.set_xticklabels(display_categories, fontsize=12) # Larger tick labels
    ax.legend(fontsize=12, title='Inversion Type', title_fontsize=13) # Larger legend, add title

    # Customize appearance
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, max(rec_percentages + single_percentages) * 1.15) # Adjust ylim slightly for labels
    sns.despine(ax=ax) # Remove top and right spines

    plt.tight_layout() # Adjust layout

    # Save the figure
    try:
        plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
        logger.info(f"Saved standalone median omega plot to {OUTPUT_PLOT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")

    plt.show() # Display the plot
    return fig # Return figure object if needed elsewhere

# --- Main Execution ---

def main():
    """Main execution function."""
    logger.info("--- Starting Median Omega Analysis (Standalone Plot) ---")

    # Load input files (includes validation and filtering of NaN/99 omega)
    pairwise_df, inversion_df = load_input_files()
    if pairwise_df is None or inversion_df is None:
        logger.error("Failed to load input files. Exiting.")
        return

    # Map CDS to inversion types, excluding ambiguous cases
    cds_to_type = map_cds_to_inversions_excluding_ambiguous(pairwise_df, inversion_df)
    if not cds_to_type:
        # Warning already logged in function if no CDS mapped
        logger.error("Failed to map CDS to inversion types. Exiting.")
        return

    # Calculate median omega value for each sequence (using Group 1, filters 99)
    median_df = calculate_sequence_median_omega(pairwise_df)
    if median_df.empty:
        logger.error("Median omega DataFrame is empty after calculation. Exiting.")
        return

    # Categorize median omega values based on inversion type
    median_count_data = categorize_median_omega_values(median_df, cds_to_type)
    if not median_count_data.get('Recurrent') and not median_count_data.get('Single-event'):
         logger.error("No data found for either Recurrent or Single-event categories after mapping. Cannot plot. Exiting.")
         return

    # Create the standalone plot
    fig = plot_median_omega_distribution(median_count_data)
    if fig is None:
        logger.error("Plot generation failed.")
    else:
        logger.info("Plot generation successful.")

    logger.info("--- Analysis completed ---")

if __name__ == "__main__":
    main()
