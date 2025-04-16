# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import logging
import sys
from collections import defaultdict
import holoviews as hv
from holoviews import opts
from bokeh.plotting import show, output_file
from bokeh.models import HoverTool
from bokeh.palettes import plasma # Using plasma
import os
import warnings
import math
import webbrowser
import time

# --- Configuration ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[ logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger('chord_plot_refined_color') # New logger name
logger.info("--- Starting New Script Run (Refined Color Mapping) ---")

# File paths
PAIRWISE_FILE = 'all_pairwise_results.csv'
INVERSION_FILE = 'inv_info.csv'
timestamp = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f'chord_plots_agg_refined_color_{timestamp}' # Updated output dir
RECURRENT_CHORD_PLOT_FILE = os.path.join(OUTPUT_DIR, 'recurrent_chord_refined.html') # Updated filename
SINGLE_EVENT_CHORD_PLOT_FILE = os.path.join(OUTPUT_DIR, 'single_event_chord_refined.html') # Updated filename

# Plotting Parameters
EDGE_WIDTH_SCALE_FACTOR = 15
MIN_EDGE_WIDTH = 0.3
NODE_COLOR_MAP = 'Category10'
EDGE_COLOR_MAP_POS_NAME = 'plasma'
EDGE_COLOR_MAP_POS_LIST = plasma(256) # Palette instance
EDGE_COLOR_NEG_ONE = '#AAAAAA'  # Grey ONLY for median_omega == -1
GLOBAL_EDGE_ALPHA = 0.2

# Omega Range for Initial Clipping (Applied *before* log transform to values >= 0)
OMEGA_CLIP_MIN = 0.001 # Must be > 0. Values exactly 0 will map to this.
OMEGA_CLIP_MAX = 100.0  # Adjusted max, tune as needed

# --- Helper Functions ---
def extract_coordinates_from_cds(cds_string):
    """Extract genomic coordinates from CDS string."""
    if not isinstance(cds_string, str): return None
    pattern = r'(chr\w+)_start(\d+)_end(\d+)'; match = re.search(pattern, cds_string)
    return {'chrom': match.group(1), 'start': int(match.group(2)), 'end': int(match.group(3))} if match else None

def map_cds_to_inversions(pairwise_df, inversion_df):
    """Map CDS strings to specific inversions and track inversion types."""
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

        if processed_cds_count % 500 == 0: # Log progress less frequently
            logger.debug(f"  Mapped {mapped_cds_count}/{processed_cds_count} CDS considered...")

    logger.info(f"Finished mapping {mapped_cds_count} CDS to types out of {processed_cds_count} processed.")
    type_counts = pd.Series(cds_to_type).value_counts();
    logger.info(f"  Type counts: {type_counts.to_dict()}")
    if 'ambiguous' in type_counts: logger.warning(f"Found {type_counts.get('ambiguous', 0)} CDS mapping to both recurrent and single event inversions.")
    if 'unknown' in type_counts: logger.warning(f"Found {type_counts.get('unknown', 0)} CDS that did not map to any inversion.")

    return cds_to_type, cds_to_inversion_id, dict(inversion_to_cds)

# --- Chord Plot Specific Functions ---

def map_value_to_hex_color(value, palette, cmin, cmax, nan_color="#FF00FF"): # Default error color changed
    """Maps a numeric value to a hex color using provided cmin/cmax."""
    if pd.isna(value):
        logger.warning(f"map_value_to_hex_color received NaN value, returning error color {nan_color}")
        return nan_color

    if cmax <= cmin:
        # If range is zero or negative map to middle of palette
        normalized = 0.5
        logger.log(logging.DEBUG if np.isclose(cmin, cmax) else logging.WARNING,
                   f"Color range invalid or zero (cmin={cmin:.4f}, cmax={cmax:.4f}). Mapping value {value:.4f} to midpoint (0.5).")
    else:
        normalized = (value - cmin) / (cmax - cmin)

    # Clamp normalized value to [0, 1]
    normalized = max(0.0, min(1.0, normalized))

    palette_size = len(palette)
    index = min(max(int(normalized * (palette_size - 1)), 0), palette_size - 1)

    try:
        return palette[index]
    except IndexError:
        logger.error(f"IndexError in map_value_to_hex_color: index={index}, palette_size={palette_size}, value={value:.4f}, cmin={cmin:.4f}, cmax={cmax:.4f}, normalized={normalized:.4f}. Returning error color.")
        return nan_color

def aggregate_pairwise_data_refined_color(df, type_name):
    """
    Aggregates pairwise data.
    Thickness = proportion of non -1 omega values.
    Color: Grey if median_omega == -1, otherwise map log10(median_omega >= 0)
           to plasma palette using Min/Max range of these log10 values.
    Ensures exactly one edge per haplotype pair.
    """
    func_name = f"aggregate_pairwise_data_refined_color ({type_name})"
    logger.info(f"[{func_name}] Starting aggregation...")
    agg_cols = ['source', 'target', 'total_comparisons', 'non_identical_comparisons',
                'proportion_non_identical', 'median_omega',
                'edge_width', 'color_value_input', # Log10 value or NaN
                'final_edge_color'] # Hex color string
    empty_agg_df = pd.DataFrame(columns=agg_cols)

    if df.empty: logger.warning(f"[{func_name}] Input df empty."); return empty_agg_df, set()

    logger.debug(f"[{func_name}] Input df shape: {df.shape}")
    logger.debug(f"[{func_name}] Input df 'omega' describe before aggregation:\n{df['omega'].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]).to_string()}")

    # --- Grouping and Basic Aggregation ---
    df['Seq1'] = df['Seq1'].astype(str); df['Seq2'] = df['Seq2'].astype(str)
    df['haplotype_pair'] = df.apply(lambda row: tuple(sorted((row['Seq1'], row['Seq2']))), axis=1) # Ensures unique pair ID

    def count_non_identical(x):
        # Ensure numeric conversion before comparison
        numeric_x = pd.to_numeric(x, errors='coerce')
        return (numeric_x.notna() & (numeric_x != -1)).sum()

    agg_funcs = {'omega': ['median', count_non_identical, 'size']}

    logger.debug(f"[{func_name}] Starting groupby aggregation on 'haplotype_pair'...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        connection_stats = df.groupby('haplotype_pair').agg(agg_funcs).reset_index()
    connection_stats.columns = ['haplotype_pair', 'median_omega', 'non_identical_comparisons', 'total_comparisons']
    logger.info(f"[{func_name}] Aggregated into {len(connection_stats)} unique haplotype pairs.")
    if connection_stats.empty: logger.warning(f"[{func_name}] No aggregated stats found."); return empty_agg_df, set()

    connection_stats['source'] = connection_stats['haplotype_pair'].apply(lambda x: x[0])
    connection_stats['target'] = connection_stats['haplotype_pair'].apply(lambda x: x[1])

    # --- Calculate Proportion and Edge Width (Thickness) ---
    logger.debug(f"[{func_name}] Calculating proportion non-identical and edge width...")
    connection_stats['total_comparisons_safe'] = connection_stats['total_comparisons'].replace(0, np.nan)
    connection_stats['proportion_non_identical'] = (
        connection_stats['non_identical_comparisons'] / connection_stats['total_comparisons_safe']
    ).fillna(0.0)
    connection_stats.drop(columns=['total_comparisons_safe'], inplace=True)
    connection_stats['total_comparisons'] = connection_stats['total_comparisons'].fillna(0).astype(int)

    connection_stats['edge_width'] = connection_stats['proportion_non_identical'] * EDGE_WIDTH_SCALE_FACTOR
    always_identical_mask = (connection_stats['proportion_non_identical'] == 0) & (connection_stats['total_comparisons'] > 0)
    num_always_identical = always_identical_mask.sum()
    if num_always_identical > 0:
        logger.info(f"[{func_name}] Found {num_always_identical} pairs always identical (prop=0). Applying min width {MIN_EDGE_WIDTH}.")
        connection_stats.loc[always_identical_mask, 'edge_width'] = MIN_EDGE_WIDTH
    connection_stats['edge_width'] = connection_stats['edge_width'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    connection_stats['edge_width'] = np.maximum(0, connection_stats['edge_width'])
    logger.debug(f"[{func_name}] Edge width calculated. Range: {connection_stats['edge_width'].min():.3f} - {connection_stats['edge_width'].max():.3f}")

    # --- Refined Color Calculation ---
    logger.debug(f"[{func_name}] Assigning edge colors (Grey for -1, Log10 scale for >= 0)...")
    connection_stats['median_omega'] = pd.to_numeric(connection_stats['median_omega'], errors='coerce')

    # Initialize columns
    connection_stats['color_value_input'] = np.nan # Will store log10(omega >= 0)
    connection_stats['final_edge_color'] = EDGE_COLOR_NEG_ONE # Default to grey

    # --- Define masks based on median_omega ---
    neg_one_mask = np.isclose(connection_stats['median_omega'], -1) # Use isclose for float comparison
    colorable_mask = (connection_stats['median_omega'].notna()) & (connection_stats['median_omega'] >= 0)
    # Rows that are neither -1 nor >= 0 (e.g., other negatives, NaNs not caught earlier) will remain grey
    other_mask = ~(neg_one_mask | colorable_mask)
    if other_mask.any():
         logger.warning(f"[{func_name}] Found {other_mask.sum()} rows with median_omega not equal to -1 and not >= 0. They will remain grey ('{EDGE_COLOR_NEG_ONE}').")

    # --- Process colorable values (median_omega >= 0) ---
    log_transformed_colorable_values = pd.Series(dtype=float)
    if colorable_mask.any():
        n_colorable = colorable_mask.sum()
        logger.debug(f"[{func_name}] Found {n_colorable} pairs with median omega >= 0 to be colored.")

        # 1. Get omega values >= 0
        omega_to_transform = connection_stats.loc[colorable_mask, 'median_omega']

        # 2. Clip these values. IMPORTANT: Map 0 to OMEGA_CLIP_MIN *before* log transform.
        #    This ensures log10(0) doesn't cause issues and maps 0 to the bottom of the color scale.
        omega_clipped = np.maximum(omega_to_transform, OMEGA_CLIP_MIN) # Treat 0 as the minimum clip value
        omega_clipped = np.clip(omega_clipped, OMEGA_CLIP_MIN, OMEGA_CLIP_MAX) # Apply upper clip

        # 3. Apply log10 transformation
        with np.errstate(divide='ignore', invalid='ignore'):
            log_transformed_colorable = np.log10(omega_clipped)

        # 4. Store the log10 value in 'color_value_input' for these rows
        connection_stats.loc[colorable_mask, 'color_value_input'] = log_transformed_colorable

        # 5. Get the series of valid, finite log10 values for range calculation
        log_transformed_colorable_values = connection_stats.loc[colorable_mask, 'color_value_input'].copy()
        log_transformed_colorable_values = log_transformed_colorable_values.replace([np.inf, -np.inf], np.nan).dropna()

        if not log_transformed_colorable_values.empty:
             logger.debug(f"[{func_name}] Finite Log10 transformed values (from median_omega >= 0) stored. Count: {len(log_transformed_colorable_values)}, Range: {log_transformed_colorable_values.min():.4f} to {log_transformed_colorable_values.max():.4f}")
        else:
             # This could happen if all >= 0 values were NaN or became non-finite after log
             logger.warning(f"[{func_name}] No finite log10 transformed values found for median_omega >= 0.")

    else:
        logger.info(f"[{func_name}] No pairs with median omega >= 0 found. All non-grey edges will use fallback range (if any).")


    # --- Determine Dynamic Color Range using Min/Max (based *only* on >= 0 values) ---
    color_low, color_high = np.log10(OMEGA_CLIP_MIN), np.log10(OMEGA_CLIP_MAX) # Default based on clip limits
    range_calculated = False
    if not log_transformed_colorable_values.empty:
        cmin_calc = log_transformed_colorable_values.min()
        cmax_calc = log_transformed_colorable_values.max()

        if np.isclose(cmin_calc, cmax_calc):
            # Handle case where all colorable values map to the same log value
            buffer = 0.1
            color_low = cmin_calc - buffer
            color_high = cmax_calc + buffer
            single_val = (color_low + color_high) / 2
            logger.warning(f"[{func_name}] All finite log omega values (>=0) are identical or very close ({single_val:.4f}). Setting color range artificially to [{color_low:.4f}, {color_high:.4f}].")
            range_calculated = True
        else:
             # Use the actual min/max
             color_low = cmin_calc
             color_high = cmax_calc
             logger.info(f"[{func_name}] Dynamic Color Range for Omega >= 0 (Log10 Min/Max): Low={color_low:.4f}, High={color_high:.4f}")
             range_calculated = True

    if not range_calculated:
        # Ensure default low < high
        if color_high <= color_low: color_high = color_low + 1.0
        logger.info(f"[{func_name}] No valid range calculated from data (median_omega >= 0). Using fallback color range [{color_low:.4f}, {color_high:.4f}] based on clip limits.")


    # --- Apply Manual Color Mapping (only to colorable rows) ---
    if colorable_mask.any() and range_calculated:
        logger.debug(f"[{func_name}] Applying manual color map to {colorable_mask.sum()} edges (median_omega >= 0) using range [{color_low:.4f}, {color_high:.4f}]")

        # Get the indices of rows to color
        color_indices = connection_stats[colorable_mask].index

        # Apply the function row by row (can be slow for large data, but safer)
        # Alternatively, use .apply() if confident NaNs in color_value_input are handled
        for idx in color_indices:
            value_to_map = connection_stats.loc[idx, 'color_value_input']
            if pd.notna(value_to_map): # Ensure we don't map NaN
                 connection_stats.loc[idx, 'final_edge_color'] = map_value_to_hex_color(
                    value_to_map,
                    EDGE_COLOR_MAP_POS_LIST,
                    color_low,
                    color_high
                 )
            else:
                 # Should not happen if log_transformed_colorable_values logic is correct, but safety check
                 logger.warning(f"[{func_name}] Found NaN in 'color_value_input' for colorable row index {idx}. Leaving color as grey.")
                 connection_stats.loc[idx, 'final_edge_color'] = EDGE_COLOR_NEG_ONE # Fallback to grey if unexpected NaN

    elif colorable_mask.any():
         logger.warning(f"[{func_name}] Cannot apply color map: Although colorable rows exist, a valid min/max range was not determined (range_calculated=False). These rows will remain grey.")
    else:
        logger.info(f"[{func_name}] No colorable rows (median_omega >= 0) found to apply color mapping.")


    # Final check on assigned colors
    logger.debug(f"[{func_name}] Final hex edge colors assigned.")
    logger.debug(f"[{func_name}] Value counts for 'final_edge_color':\n{connection_stats['final_edge_color'].value_counts().to_string()}")

    # --- Final Selection and Node Identification ---
    aggregated_df = connection_stats[agg_cols].copy()
    nodes = set(aggregated_df['source']).union(set(aggregated_df['target']))

    logger.info(f"[{func_name}] Aggregation finished. Final aggregated data shape: {aggregated_df.shape}")
    logger.info(f"[{func_name}] Identified {len(nodes)} unique haplotypes involved.")
    logger.debug(f"[{func_name}] Aggregated Data Head:\n{aggregated_df.head().to_string()}")

    # Sanity check colors again
    unique_colors = aggregated_df['final_edge_color'].unique()
    n_unique_colors = len(unique_colors)
    n_grey = (aggregated_df['final_edge_color'] == EDGE_COLOR_NEG_ONE).sum()
    n_non_grey = len(aggregated_df) - n_grey
    logger.debug(f"[{func_name}] Final edge color summary: {n_unique_colors} unique colors total. {n_grey} grey edges, {n_non_grey} potentially colored edges.")
    if n_non_grey > 0:
        non_grey_colors = aggregated_df[aggregated_df['final_edge_color'] != EDGE_COLOR_NEG_ONE]['final_edge_color'].unique()
        logger.debug(f"[{func_name}] Found {len(non_grey_colors)} unique non-grey colors.")
        # Check if all non-grey colors are the same (e.g., all map to blue)
        if len(non_grey_colors) == 1:
             logger.warning(f"[{func_name}] All {n_non_grey} non-grey edges have the *same* color: {non_grey_colors[0]}. This might indicate data skew towards one end of the range.")
        elif len(non_grey_colors) > 1:
             logger.info(f"[{func_name}] Non-grey edges span multiple colors, suggesting variation was mapped.")


    return aggregated_df, nodes


# --- Plotting Function ---
# plot_chord_diagram remains largely the same, but rename to reflect intent
def plot_refined_color_chord_diagram(aggregated_data, nodes, title, filename):
    """
    Generates chord diagram using pre-calculated hex colors.
    Grey for -1, plasma scale for >= 0 based on Min/Max log10 range.
    Thickness is proportion non-identical. Uniform alpha.
    """
    func_name = f"plot_refined_color_chord_diagram ({os.path.basename(filename)})" # Renamed function
    logger.info(f"[{func_name}] Starting plot generation...")

    if aggregated_data is None or aggregated_data.empty:
        logger.warning(f"[{func_name}] No aggregated data provided. Skipping plot creation for '{filename}'.")
        return
    if not nodes:
         logger.warning(f"[{func_name}] No nodes provided. Skipping plot creation for '{filename}'.")
         return

    logger.info(f"[{func_name}] Generating plot: {title} (Nodes: {len(nodes)})")
    if len(nodes) > 150: logger.warning(f"[{func_name}] Plotting large number of nodes ({len(nodes)}).")

    # --- Prepare Data and Nodes Dataset ---
    plot_data = aggregated_data.copy()
    numeric_cols = ['edge_width', 'median_omega', 'proportion_non_identical', 'color_value_input',
                    'total_comparisons', 'non_identical_comparisons']
    for col in numeric_cols:
        if col in plot_data.columns: plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
        else: logger.warning(f"[{func_name}] Expected numeric column '{col}' not found."); plot_data[col] = np.nan

    for col in ['source', 'target', 'final_edge_color']:
         if col in plot_data.columns: plot_data[col] = plot_data[col].astype(str)
         else: logger.error(f"[{func_name}] Required string column '{col}' not found."); return None

    # Handle potential NaNs in critical columns
    plot_data['edge_width'].fillna(0.0, inplace=True) # Make edges with no width invisible
    plot_data['final_edge_color'].fillna(EDGE_COLOR_NEG_ONE, inplace=True) # Use grey if color somehow became NaN

    nodes_df = pd.DataFrame({'haplotype': sorted(list(nodes))}) # Sort nodes
    nodes_dataset = hv.Dataset(nodes_df, 'haplotype')

    # --- Define Hover Tool ---
    hover = HoverTool(tooltips=[
        ('From', '@source'), ('To', '@target'),
        ('Median ω', '@median_omega{0.000a}'),
        ('Prop. Non-Identical (ω≠-1)', '@proportion_non_identical{0.0%}'), # Thickness basis
        ('# Non-Identical', '@non_identical_comparisons{0,0}'),
        ('# Total Comparisons', '@total_comparisons{0,0}'),
        ('Color Value (Log10[ω>=0])', '@color_value_input{0.000}'), # Value used for color scale
        ('Edge Color Code', '@final_edge_color') # The resulting hex color
    ])

    # --- Create Chord object ---
    required_hv_cols = ['source', 'target', 'edge_width', 'final_edge_color', # Core Chord needs
                       'median_omega', 'proportion_non_identical',           # Needed for vdims/hover
                       'non_identical_comparisons', 'total_comparisons', 'color_value_input']
    vdims = [col for col in required_hv_cols if col not in ['source', 'target']]

    missing_hv_cols = [col for col in required_hv_cols if col not in plot_data.columns]
    if missing_hv_cols:
        logger.error(f"[{func_name}] Aggregated data missing required columns: {missing_hv_cols}. Cannot plot.")
        return None

    logger.debug(f"[{func_name}] Creating hv.Chord object...")
    try:
        chord_element = hv.Chord((plot_data, nodes_dataset), vdims=vdims)
        logger.debug(f"[{func_name}] hv.Chord object created successfully.")
    except Exception as e:
        logger.error(f"[{func_name}] Error creating hv.Chord object: {e}", exc_info=True)
        logger.error(f"Data dtypes:\n{plot_data.dtypes}")
        return None

    # --- Apply Options ---
    logger.debug(f"[{func_name}] Applying HoloViews options...")
    try:
        final_plot = chord_element.opts(
            opts.Chord(
                title=title,
                labels='haplotype', node_color='haplotype', node_cmap=NODE_COLOR_MAP, node_size=9,
                edge_color='final_edge_color', # Use the pre-calculated hex colors
                edge_line_width='edge_width',  # Use pre-calculated edge width for thickness
                edge_alpha=GLOBAL_EDGE_ALPHA,  # Uniform alpha
                tools=[hover, 'tap'], width=850, height=850, toolbar='above',
                # label_text_font_size='8pt' # Uncomment if labels overlap
            )
        )
        logger.debug(f"[{func_name}] HoloViews options applied.")
    except Exception as e:
        logger.error(f"[{func_name}] Error applying HoloViews options: {e}", exc_info=True)
        return None

    # --- Save and Open ---
    # (Save/Open logic remains the same as previous version)
    try:
        logger.debug(f"[{func_name}] Ensuring output directory exists: {OUTPUT_DIR}")
        os.makedirs(os.path.dirname(filename), exist_ok=True) # Use os.path.dirname
        logger.info(f"[{func_name}] Saving plot to: {filename}")
        hv.save(final_plot, filename, backend='bokeh')
        logger.info(f"[{func_name}] Plot saved successfully.")
        try:
            logger.info(f"[{func_name}] Attempting to open plot in browser...")
            webbrowser.open(f'file://{os.path.realpath(filename)}')
            logger.info(f"[{func_name}] Browser launch command issued.")
        except Exception as e_open: logger.warning(f"[{func_name}] Could not automatically open the plot: {e_open}")
    except Exception as e_save: logger.error(f"[{func_name}] Failed to save plot '{filename}': {e_save}", exc_info=True)

    return final_plot


# --- Main Execution ---
def main():
    """Main execution function."""
    logger.info("--- Starting Aggregated Chord Plot Generation Script (Refined Color) ---")
    hv.extension('bokeh')

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
    if not cds_to_type: logger.warning("CDS mapping resulted in an empty map.")
    pairwise_df['inversion_type'] = pairwise_df['CDS'].map(cds_to_type).fillna('unknown')
    logger.info("Added 'inversion_type' column to pairwise data.")
    logger.debug(f"Value counts for 'inversion_type':\n{pairwise_df['inversion_type'].value_counts().to_string()}")

    # --- Step 2: Prepare Base Pairwise Data for Aggregation ---
    logger.info("--- Step 2: Filtering Pairwise Data ---")
    required_cols = ['inversion_type', 'Group1', 'Group2', 'omega', 'Seq1', 'Seq2']
    missing_base_cols = [col for col in required_cols if col not in pairwise_df.columns]
    if missing_base_cols: logger.error(f"Pairwise data missing required columns: {missing_base_cols}."); return

    logger.debug("Converting key columns ('Group1', 'Group2', 'omega') to numeric...")
    try:
        for col in ['Group1', 'Group2', 'omega']: pairwise_df[col] = pd.to_numeric(pairwise_df[col], errors='coerce')
    except Exception as e: logger.error(f"Error converting columns to numeric: {e}", exc_info=True); return
    logger.debug("Numeric conversion complete.")

    filter_mask = (
        (pairwise_df['inversion_type'].isin(['recurrent', 'single_event'])) &
        (pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1) &
        pairwise_df['omega'].notna() & (pairwise_df['omega'] != 99) &
        pairwise_df['Seq1'].notna() & pairwise_df['Seq2'].notna() &
        (pairwise_df['Seq1'] != pairwise_df['Seq2'])
    )
    filtered_df = pairwise_df[filter_mask].copy()
    num_filtered = len(filtered_df)
    logger.info(f"Applied filters. Found {num_filtered:,} relevant pairs.")
    if filtered_df.empty: logger.error("No pairs found matching filters."); return

    # --- Step 3: Aggregate Data for Recurrent Plot ---
    logger.info("--- Step 3: Aggregating Data for Recurrent Inversions ---")
    recurrent_pairs_df = filtered_df[filtered_df['inversion_type'] == 'recurrent'].copy()
    logger.info(f"Processing {len(recurrent_pairs_df):,} pairs for Recurrent plot.")
    if not recurrent_pairs_df.empty:
        rec_agg_data, rec_nodes = aggregate_pairwise_data_refined_color(recurrent_pairs_df, "recurrent") # Use refined func
    else:
        logger.warning("No recurrent pairs found after filtering.")
        rec_agg_data, rec_nodes = pd.DataFrame(columns=agg_cols), set()

    # --- Step 4: Aggregate Data for Single-Event Plot ---
    logger.info("--- Step 4: Aggregating Data for Single-Event Inversions ---")
    single_event_pairs_df = filtered_df[filtered_df['inversion_type'] == 'single_event'].copy()
    logger.info(f"Processing {len(single_event_pairs_df):,} pairs for Single-Event plot.")
    if not single_event_pairs_df.empty:
        single_agg_data, single_nodes = aggregate_pairwise_data_refined_color(single_event_pairs_df, "single_event") # Use refined func
    else:
        logger.warning("No single-event pairs found after filtering.")
        single_agg_data, single_nodes = pd.DataFrame(columns=agg_cols), set()

    # --- Step 5: Plotting ---
    logger.info("--- Step 5: Generating Chord Diagrams ---")

    # Plot Recurrent
    plot_refined_color_chord_diagram( # Use refined plot func
        rec_agg_data,
        rec_nodes,
        (f"Aggregated Comparisons (Recurrent Inversions)\n"
         f"Thickness=Prop(ω≠-1), Alpha={GLOBAL_EDGE_ALPHA}, "
         f"Color=Median ω (-1=Grey, ≥0=Log10[Min-Max] {EDGE_COLOR_MAP_POS_NAME.capitalize()})"), # Refined title
        RECURRENT_CHORD_PLOT_FILE
    )

    # Plot Single-Event
    plot_refined_color_chord_diagram( # Use refined plot func
        single_agg_data,
        single_nodes,
         (f"Aggregated Comparisons (Single-Event Inversions)\n"
         f"Thickness=Prop(ω≠-1), Alpha={GLOBAL_EDGE_ALPHA}, "
         f"Color=Median ω (-1=Grey, ≥0=Log10[Min-Max] {EDGE_COLOR_MAP_POS_NAME.capitalize()})"), # Refined title
        SINGLE_EVENT_CHORD_PLOT_FILE
    )

    logger.info("--- Chord plot generation script finished ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as main_e:
        logger.critical(f"An unhandled exception occurred in the main script execution: {main_e}", exc_info=True)
