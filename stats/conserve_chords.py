# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import logging
import sys
from collections import defaultdict
import holoviews as hv
from holoviews import opts
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb # For color interpolation
import os
import warnings
import math
import time

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for production, DEBUG if needed
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[ logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger('chord_plot_proportion_color') # New logger name reflecting the change
logger.info("--- Starting New Script Run (Proportion Non-Identical Color Mapping) ---")

# File paths
PAIRWISE_FILE = 'all_pairwise_results.csv'
INVERSION_FILE = 'inv_info.csv'
timestamp = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f'chord_plots_proportion_color_{timestamp}' # Updated output dir
RECURRENT_CHORD_PLOT_FILE = os.path.join(OUTPUT_DIR, 'recurrent_chord_proportion.html') # Updated filename
SINGLE_EVENT_CHORD_PLOT_FILE = os.path.join(OUTPUT_DIR, 'single_event_chord_proportion.html') # Updated filename

# Plotting Parameters
CONSTANT_EDGE_WIDTH = 0.5  # All edges will have this width
NODE_COLOR_MAP = 'Category10' # Colors for the haplotype nodes
GLOBAL_EDGE_ALPHA = 0.6 # Transparency of edges (adjust as needed)

# Color Scale for Proportion Non-Identical
COLOR_SCALE_START = '#D3D3D3' # Light Grey (Proportion = 0, all identical)
COLOR_SCALE_END = '#FF0000'   # Bright Red (Proportion = 1, none identical)
# Create a colormap object for easy mapping
PROPORTION_CMAP = LinearSegmentedColormap.from_list(
    "grey_to_red", [COLOR_SCALE_START, COLOR_SCALE_END]
)
PROPORTION_NAN_COLOR = '#808080' # Color for edges where proportion couldn't be calculated (e.g., 0 total comparisons)

# --- Helper Functions ---
def extract_coordinates_from_cds(cds_string):
    """Extract genomic coordinates from CDS string."""
    if not isinstance(cds_string, str): return None
    pattern = r'(chr\w+)_start(\d+)_end(\d+)'; match = re.search(pattern, cds_string)
    return {'chrom': match.group(1), 'start': int(match.group(2)), 'end': int(match.group(3))} if match else None

def map_cds_to_inversions(pairwise_df, inversion_df):
    """Map CDS strings to specific inversions and track inversion types. (Unchanged from previous version)"""
    logger.info("Mapping CDS to inversion types...");
    if 'CDS' not in pairwise_df.columns or pairwise_df['CDS'].isnull().all(): logger.error("Missing 'CDS' column or all CDS values are null."); return {}, {}, {}
    unique_cds = pairwise_df['CDS'].dropna().unique();
    if len(unique_cds) == 0: logger.warning("No unique non-null CDS found."); return {}, {}, {}

    cds_coords = {cds: coords for cds in unique_cds if (coords := extract_coordinates_from_cds(cds))}
    if not cds_coords: logger.warning("Could not extract coordinates from any CDS string."); # Allow proceeding even if some fail

    required_inv_cols = ['Start', 'End', '0_single_1_recur', 'Chromosome']
    if not all(col in inversion_df.columns for col in required_inv_cols):
        logger.error(f"Missing required columns in inversion info file: {required_inv_cols}. Cannot map types."); return {}, {}, {}

    # Clean inversion data
    for col in ['Start', 'End', '0_single_1_recur']:
         try: inversion_df[col] = pd.to_numeric(inversion_df[col], errors='coerce')
         except Exception as e: logger.error(f"Error converting inversion column {col} to numeric: {e}"); return {}, {}, {}
    inversion_df.dropna(subset=['Start', 'End', '0_single_1_recur', 'Chromosome'], inplace=True)
    if inversion_df.empty: logger.error("Inversion df empty after cleaning required columns."); return {}, {}, {}
    inversion_df['Start'] = inversion_df['Start'].astype(int); inversion_df['End'] = inversion_df['End'].astype(int); inversion_df['0_single_1_recur'] = inversion_df['0_single_1_recur'].astype(int)

    recurrent_inv = inversion_df[inversion_df['0_single_1_recur'] == 1];
    single_event_inv = inversion_df[inversion_df['0_single_1_recur'] == 0]

    cds_to_type = {}; cds_to_inversion_id = {}; inversion_to_cds = defaultdict(list)
    processed_cds_count = 0
    mapped_cds_count = 0

    logger.info(f"Attempting to map {len(cds_coords)} unique CDS with valid coordinates...")
    for cds, coords in cds_coords.items():
        if not coords: continue # Skip if coords extraction failed for this CDS
        processed_cds_count += 1
        chrom, start, end = coords['chrom'], coords['start'], coords['end']

        rec_matches = recurrent_inv[ (recurrent_inv['Chromosome'] == chrom) & (start < recurrent_inv['End']) & (end > recurrent_inv['Start']) ]
        single_matches = single_event_inv[ (single_event_inv['Chromosome'] == chrom) & (start < single_event_inv['End']) & (end > single_event_inv['Start']) ]

        is_recurrent = len(rec_matches) > 0; is_single = len(single_matches) > 0

        type_assigned = 'unknown'
        if is_recurrent and not is_single: type_assigned = 'recurrent'
        elif is_single and not is_recurrent: type_assigned = 'single_event'
        elif is_recurrent and is_single: type_assigned = 'ambiguous'
        else: type_assigned = 'unknown'

        cds_to_type[cds] = type_assigned
        mapped_cds_count += 1

        if processed_cds_count % 1000 == 0: # Log progress less frequently
            logger.debug(f"  Mapped {mapped_cds_count}/{processed_cds_count} CDS considered...")

    logger.info(f"Finished mapping {mapped_cds_count} CDS to types out of {processed_cds_count} processed.")
    type_counts = pd.Series(cds_to_type).value_counts();
    logger.info(f"  Type counts: {type_counts.to_dict()}")
    if 'ambiguous' in type_counts: logger.warning(f"Found {type_counts.get('ambiguous', 0)} CDS mapping to both recurrent and single event inversions.")
    if 'unknown' in type_counts: logger.warning(f"Found {type_counts.get('unknown', 0)} CDS that did not map to any inversion.")

    return cds_to_type, cds_to_inversion_id, dict(inversion_to_cds)


def map_proportion_to_color(proportion):
    """Maps a proportion (0-1) to a hex color using the PROPORTION_CMAP."""
    if pd.isna(proportion):
        # logger.warning(f"Received NaN proportion, returning NaN color: {PROPORTION_NAN_COLOR}") # Log only if needed
        return PROPORTION_NAN_COLOR
    # Clamp proportion to [0, 1] just in case
    clamped_proportion = max(0.0, min(1.0, proportion))
    try:
        # PROPORTION_CMAP returns RGBA, convert to hex RGB
        rgba_color = PROPORTION_CMAP(clamped_proportion)
        hex_color = to_hex(rgba_color[:3]) # Get RGB part and convert to hex
        return hex_color
    except Exception as e:
        logger.error(f"Error mapping proportion {proportion} to color: {e}")
        return PROPORTION_NAN_COLOR # Return fallback color on error


# --- Chord Plot Specific Functions ---

def aggregate_pairwise_data_proportion_color(df, type_name):
    """
    Aggregates pairwise data for chord plot where color represents the
    proportion of non-identical (omega != -1) comparisons, and width is constant.
    Ensures exactly one edge per haplotype pair.
    """
    func_name = f"aggregate_pairwise_data_proportion_color ({type_name})"
    logger.info(f"[{func_name}] Starting aggregation...")
    # Define expected columns in the final aggregated output
    agg_cols = ['source', 'target', 'total_comparisons', 'non_identical_comparisons',
                'proportion_non_identical', 'median_omega', # Keep median for context if needed later
                'edge_width', 'edge_color']
    empty_agg_df = pd.DataFrame(columns=agg_cols)

    if df.empty: logger.warning(f"[{func_name}] Input df empty."); return empty_agg_df, set()

    logger.debug(f"[{func_name}] Input df shape: {df.shape}")
    logger.debug(f"[{func_name}] Input df 'omega' describe before aggregation:\n{df['omega'].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]).to_string()}")

    # --- Grouping and Basic Aggregation ---
    df['Seq1'] = df['Seq1'].astype(str); df['Seq2'] = df['Seq2'].astype(str)
    # Create a unique, order-independent identifier for each pair
    df['haplotype_pair'] = df.apply(lambda row: tuple(sorted((row['Seq1'], row['Seq2']))), axis=1)

    def count_non_identical(x):
        # Ensure numeric conversion before comparison, treat NaN as non-identical? NO, only count != -1
        numeric_x = pd.to_numeric(x, errors='coerce')
        # Count where omega is NOT NA and NOT -1. Use np.isclose for float comparison if needed.
        # Assuming -1 is stored accurately, direct comparison is fine.
        return (numeric_x.notna() & (numeric_x != -1)).sum()

    # Define aggregation functions
    agg_funcs = {
        'omega': ['median', count_non_identical, 'size'] # size gives total comparisons
    }

    logger.debug(f"[{func_name}] Starting groupby aggregation on 'haplotype_pair'...")
    with warnings.catch_warnings():
        # Median might raise RuntimeWarning if all values in a group are NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        connection_stats = df.groupby('haplotype_pair').agg(agg_funcs).reset_index()

    # Rename columns for clarity
    connection_stats.columns = ['haplotype_pair', 'median_omega', 'non_identical_comparisons', 'total_comparisons']
    logger.info(f"[{func_name}] Aggregated into {len(connection_stats)} unique haplotype pairs.")
    if connection_stats.empty: logger.warning(f"[{func_name}] No aggregated stats found."); return empty_agg_df, set()

    # Extract source and target haplotypes from the pair tuple
    connection_stats['source'] = connection_stats['haplotype_pair'].apply(lambda x: x[0])
    connection_stats['target'] = connection_stats['haplotype_pair'].apply(lambda x: x[1])

    # --- Calculate Proportion Non-Identical ---
    logger.debug(f"[{func_name}] Calculating proportion non-identical...")
    # Avoid division by zero: replace 0 totals with NaN temporarily
    connection_stats['total_comparisons_safe'] = connection_stats['total_comparisons'].replace(0, np.nan)
    connection_stats['proportion_non_identical'] = (
        connection_stats['non_identical_comparisons'] / connection_stats['total_comparisons_safe']
    )
    # Fill NaN proportions (from 0 total comparisons) with 0.0 - these pairs had no valid comparisons.
    # Alternatively, could assign NaN and use PROPORTION_NAN_COLOR later. Let's fill with 0.
    connection_stats['proportion_non_identical'].fillna(0.0, inplace=True)
    connection_stats.drop(columns=['total_comparisons_safe'], inplace=True)
    connection_stats['total_comparisons'] = connection_stats['total_comparisons'].fillna(0).astype(int) # Ensure integer

    # --- Assign Constant Edge Width ---
    logger.debug(f"[{func_name}] Assigning constant edge width: {CONSTANT_EDGE_WIDTH}")
    connection_stats['edge_width'] = CONSTANT_EDGE_WIDTH

    # --- Assign Edge Color based on Proportion ---
    logger.debug(f"[{func_name}] Assigning edge colors based on proportion_non_identical (Grey to Red)...")
    connection_stats['edge_color'] = connection_stats['proportion_non_identical'].apply(map_proportion_to_color)

    # --- Final Selection and Node Identification ---
    # Ensure only required columns are kept
    aggregated_df = connection_stats[agg_cols].copy()
    nodes = set(aggregated_df['source']).union(set(aggregated_df['target']))

    logger.info(f"[{func_name}] Aggregation finished. Final aggregated data shape: {aggregated_df.shape}")
    logger.info(f"[{func_name}] Identified {len(nodes)} unique haplotypes involved.")
    logger.debug(f"[{func_name}] Aggregated Data Head:\n{aggregated_df.head().to_string()}")

    # Sanity check colors
    unique_colors = aggregated_df['edge_color'].unique()
    n_unique_colors = len(unique_colors)
    n_start_color = (aggregated_df['edge_color'] == COLOR_SCALE_START).sum()
    n_end_color = (aggregated_df['edge_color'] == COLOR_SCALE_END).sum()
    n_nan_color = (aggregated_df['edge_color'] == PROPORTION_NAN_COLOR).sum()
    n_intermediate = len(aggregated_df) - n_start_color - n_end_color - n_nan_color

    logger.debug(f"[{func_name}] Final edge color summary: {n_unique_colors} unique colors total.")
    logger.debug(f"  - Start color ({COLOR_SCALE_START}): {n_start_color} edges")
    logger.debug(f"  - End color ({COLOR_SCALE_END}): {n_end_color} edges")
    logger.debug(f"  - Intermediate colors: {n_intermediate} edges")
    if n_nan_color > 0:
        logger.warning(f"  - NAN color ({PROPORTION_NAN_COLOR}): {n_nan_color} edges")

    return aggregated_df, nodes


# --- Plotting Function ---
def plot_proportion_color_chord_diagram(aggregated_data, nodes, title, filename):
    """
    Generates static chord diagram.
    Color: Grey-to-Red based on proportion non-identical (omega != -1).
    Width: Constant thin width.
    No interactivity (hover tools disabled).
    """
    func_name = f"plot_proportion_color_chord_diagram ({os.path.basename(filename)})"
    logger.info(f"[{func_name}] Starting plot generation...")

    if aggregated_data is None or aggregated_data.empty:
        logger.warning(f"[{func_name}] No aggregated data provided. Skipping plot creation for '{filename}'.")
        return
    if not nodes:
         logger.warning(f"[{func_name}] No nodes provided. Skipping plot creation for '{filename}'.")
         return

    logger.info(f"[{func_name}] Generating plot: {title} (Nodes: {len(nodes)})")
    if len(nodes) > 200: logger.warning(f"[{func_name}] Plotting large number of nodes ({len(nodes)}). Rendering might be slow/cluttered.")

    # --- Prepare Data and Nodes Dataset ---
    plot_data = aggregated_data.copy()
    # Ensure required columns have correct types
    numeric_cols = ['edge_width', 'median_omega', 'proportion_non_identical',
                    'total_comparisons', 'non_identical_comparisons']
    for col in numeric_cols:
        if col in plot_data.columns: plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
        else: logger.warning(f"[{func_name}] Expected numeric column '{col}' not found."); plot_data[col] = np.nan

    string_cols = ['source', 'target', 'edge_color']
    for col in string_cols:
         if col in plot_data.columns: plot_data[col] = plot_data[col].astype(str)
         else: logger.error(f"[{func_name}] Required string column '{col}' not found."); return None

    # Handle potential NaNs introduced by coercion or calculation issues
    plot_data['edge_width'].fillna(0.0, inplace=True) # Make edges with no width invisible
    plot_data['edge_color'].fillna(PROPORTION_NAN_COLOR, inplace=True) # Use designated NaN color if needed

    # Create nodes dataset for HoloViews
    nodes_df = pd.DataFrame({'haplotype': sorted(list(nodes))}) # Sort nodes alphabetically
    nodes_dataset = hv.Dataset(nodes_df, 'haplotype')

    # --- Define Value Dimensions (vdims) ---
    # These are columns available for plot options (like edge_color, edge_width)
    # Include potentially useful context columns even if not directly mapped to aesthetics now
    required_hv_cols = ['source', 'target', # kdims
                        'edge_width', 'edge_color', # Mapped aesthetics
                        'proportion_non_identical', 'total_comparisons', # Context
                        'non_identical_comparisons', 'median_omega'] # Context
    vdims = [col for col in required_hv_cols if col not in ['source', 'target']]

    missing_hv_cols = [col for col in required_hv_cols if col not in plot_data.columns]
    if missing_hv_cols:
        logger.error(f"[{func_name}] Aggregated data missing required columns for vdims: {missing_hv_cols}. Cannot plot.")
        return None

    # --- Create Chord object ---
    logger.debug(f"[{func_name}] Creating hv.Chord object...")
    try:
        chord_element = hv.Chord((plot_data, nodes_dataset), vdims=vdims)
        logger.debug(f"[{func_name}] hv.Chord object created successfully.")
    except Exception as e:
        logger.error(f"[{func_name}] Error creating hv.Chord object: {e}", exc_info=True)
        logger.error(f"Data dtypes:\n{plot_data.dtypes}")
        logger.error(f"Plot data head:\n{plot_data.head()}")
        return None

    # --- Apply Options (Static Plot) ---
    logger.debug(f"[{func_name}] Applying HoloViews options for static plot...")
    try:
        final_plot = chord_element.opts(
            opts.Chord(
                title=title,
                labels='haplotype', node_color='haplotype', node_cmap=NODE_COLOR_MAP, node_size=9,
                edge_color='edge_color',      # Use the pre-calculated hex colors directly
                edge_line_width='edge_width', # Use the constant edge width column
                edge_alpha=GLOBAL_EDGE_ALPHA, # Apply uniform transparency
                tools=[], # Explicitly disable tools like hover, tap
                width=850, height=850, toolbar=None, # Disable toolbar
                xaxis=None, yaxis=None # Remove axes if they appear
                # label_text_font_size='8pt' # Uncomment if labels overlap significantly
            )
        )
        logger.debug(f"[{func_name}] HoloViews options applied.")
    except Exception as e:
        logger.error(f"[{func_name}] Error applying HoloViews options: {e}", exc_info=True)
        return None

    # --- Save Plot (No Opening) ---
    try:
        logger.debug(f"[{func_name}] Ensuring output directory exists: {OUTPUT_DIR}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.info(f"[{func_name}] Saving static plot to: {filename}")
        # Use Bokeh backend for HTML output
        hv.save(final_plot, filename, backend='bokeh')
        logger.info(f"[{func_name}] Plot saved successfully.")
        # Removed webbrowser opening
    except Exception as e_save:
        logger.error(f"[{func_name}] Failed to save plot '{filename}': {e_save}", exc_info=True)

    return final_plot


# --- Main Execution ---
def main():
    """Main execution function."""
    logger.info("--- Starting Aggregated Chord Plot Generation Script (Proportion Color, Static) ---")
    hv.extension('bokeh') # Still needed for rendering engine

    # --- Create Output Directory ---
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory set to: {OUTPUT_DIR}")
    except OSError as e: logger.error(f"Failed to create output directory '{OUTPUT_DIR}': {e}"); return

    # --- Load input files ---
    try:
        logger.info(f"Loading pairwise data from: {PAIRWISE_FILE}")
        pairwise_df = pd.read_csv(PAIRWISE_FILE)
        logger.info(f"Loaded {len(pairwise_df):,} rows from pairwise data.")
        logger.info(f"Loading inversion info from: {INVERSION_FILE}")
        inversion_df = pd.read_csv(INVERSION_FILE)
        logger.info(f"Loaded {len(inversion_df):,} rows from inversion info.")
    except FileNotFoundError as e: logger.error(f"Error loading input file: {e}."); return
    except Exception as e: logger.error(f"An unexpected error occurred loading input files: {e}", exc_info=True); return

    # --- Step 1: Map CDS to Inversion Types ---
    logger.info("--- Step 1: Mapping CDS to Inversion Types ---")
    cds_to_type, _, _ = map_cds_to_inversions(pairwise_df, inversion_df)
    if not cds_to_type: logger.warning("CDS mapping resulted in an empty map. Subsequent steps might yield empty plots.")
    pairwise_df['inversion_type'] = pairwise_df['CDS'].map(cds_to_type).fillna('unknown')
    logger.info("Added 'inversion_type' column to pairwise data.")
    logger.debug(f"Value counts for 'inversion_type':\n{pairwise_df['inversion_type'].value_counts().to_string()}")

    # --- Step 2: Prepare Base Pairwise Data for Aggregation ---
    logger.info("--- Step 2: Filtering Pairwise Data ---")
    required_cols = ['inversion_type', 'Group1', 'Group2', 'omega', 'Seq1', 'Seq2']
    missing_base_cols = [col for col in required_cols if col not in pairwise_df.columns]
    if missing_base_cols: logger.error(f"Pairwise data missing required columns: {missing_base_cols}. Cannot proceed."); return

    logger.debug("Converting key columns ('Group1', 'Group2', 'omega') to numeric...")
    try:
        # Coerce errors to NaN for filtering
        for col in ['Group1', 'Group2', 'omega']: pairwise_df[col] = pd.to_numeric(pairwise_df[col], errors='coerce')
    except Exception as e: logger.error(f"Error converting columns to numeric: {e}", exc_info=True); return
    logger.debug("Numeric conversion complete.")

    # Define filter mask based on requirements
    filter_mask = (
        (pairwise_df['inversion_type'].isin(['recurrent', 'single_event'])) &
        (pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1) &
        pairwise_df['omega'].notna() & (pairwise_df['omega'] != 99) & # Ensure omega is valid number, exclude 99
        pairwise_df['Seq1'].notna() & pairwise_df['Seq2'].notna() & # Ensure sequences are present
        (pairwise_df['Seq1'] != pairwise_df['Seq2']) # Exclude self-comparisons
    )
    filtered_df = pairwise_df[filter_mask].copy()
    num_filtered = len(filtered_df)
    logger.info(f"Applied filters (Type=rec/single, G1=1, G2=1, valid omega!=99, valid/diff seqs). Found {num_filtered:,} relevant pairs.")
    if filtered_df.empty: logger.error("No pairs found matching filters. Cannot generate plots."); return

    # --- Step 3: Aggregate Data for Recurrent Plot ---
    logger.info("--- Step 3: Aggregating Data for Recurrent Inversions ---")
    recurrent_pairs_df = filtered_df[filtered_df['inversion_type'] == 'recurrent'].copy()
    logger.info(f"Processing {len(recurrent_pairs_df):,} pairs for Recurrent plot.")
    if not recurrent_pairs_df.empty:
        rec_agg_data, rec_nodes = aggregate_pairwise_data_proportion_color(recurrent_pairs_df, "recurrent")
    else:
        logger.warning("No recurrent pairs found after filtering.")
        # Define columns based on the aggregation function's output spec
        agg_cols_spec = ['source', 'target', 'total_comparisons', 'non_identical_comparisons',
                         'proportion_non_identical', 'median_omega', 'edge_width', 'edge_color']
        rec_agg_data, rec_nodes = pd.DataFrame(columns=agg_cols_spec), set()


    # --- Step 4: Aggregate Data for Single-Event Plot ---
    logger.info("--- Step 4: Aggregating Data for Single-Event Inversions ---")
    single_event_pairs_df = filtered_df[filtered_df['inversion_type'] == 'single_event'].copy()
    logger.info(f"Processing {len(single_event_pairs_df):,} pairs for Single-Event plot.")
    if not single_event_pairs_df.empty:
        single_agg_data, single_nodes = aggregate_pairwise_data_proportion_color(single_event_pairs_df, "single_event")
    else:
        logger.warning("No single-event pairs found after filtering.")
        agg_cols_spec = ['source', 'target', 'total_comparisons', 'non_identical_comparisons',
                         'proportion_non_identical', 'median_omega', 'edge_width', 'edge_color']
        single_agg_data, single_nodes = pd.DataFrame(columns=agg_cols_spec), set()

    # --- Step 5: Plotting (Static) ---
    logger.info("--- Step 5: Generating Static Chord Diagrams ---")

    # Plot Recurrent
    plot_proportion_color_chord_diagram(
        rec_agg_data,
        rec_nodes,
        (f"Aggregated Comparisons (Recurrent Inversions)\n"
         f"Width=Constant ({CONSTANT_EDGE_WIDTH}), Alpha={GLOBAL_EDGE_ALPHA}, "
         f"Color=Proportion(ω≠-1) [{COLOR_SCALE_START} to {COLOR_SCALE_END}]"), # Updated title
        RECURRENT_CHORD_PLOT_FILE
    )

    # Plot Single-Event
    plot_proportion_color_chord_diagram(
        single_agg_data,
        single_nodes,
         (f"Aggregated Comparisons (Single-Event Inversions)\n"
         f"Width=Constant ({CONSTANT_EDGE_WIDTH}), Alpha={GLOBAL_EDGE_ALPHA}, "
         f"Color=Proportion(ω≠-1) [{COLOR_SCALE_START} to {COLOR_SCALE_END}]"), # Updated title
        SINGLE_EVENT_CHORD_PLOT_FILE
    )

    logger.info("--- Chord plot generation script finished ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as main_e:
        # Use critical level for unhandled exceptions in main flow
        logger.critical(f"An unhandled exception occurred in the main script execution: {main_e}", exc_info=True)
