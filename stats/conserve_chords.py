# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import logging
import sys
from collections import defaultdict
import holoviews as hv
from holoviews import opts
import bokeh.plotting as bk_plt
from bokeh.model import Model as bk_Model # Import the base Model class
from matplotlib.colors import LinearSegmentedColormap, to_hex, FuncNorm # For color interpolation and non-linear norm
import matplotlib.pyplot as plt # For creating the colorbar figure
import os
import warnings
import math
import time

"""
Fix edge coloring for constant-value data (e.g., all proportions 0.0): Instead of relying on potentially ambiguous 'edge_color' or applying a colormap to zero-range data (which can cause default colors like black), explicitly set 'edge_line_color' to the desired fixed color (e.g., '#0000FF') and 'edge_line_alpha'. This directly controls the Bokeh MultiLine glyph's line appearance. Removed 'edge_fill_alpha' as it's not a valid property for MultiLine and caused crashes.
"""

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[ logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger('chord_plot_node_avg_color')
logger.info("--- Starting New Script Run (Node Avg Color, Blue-Red CMap 0-1, Separate Key, Explicit Line Color, No Fill Alpha) ---") # Updated description

# File paths
PAIRWISE_FILE = 'all_pairwise_results.csv'
INVERSION_FILE = 'inv_info.tsv'
timestamp = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f'chord_plots_node_avg_color_{timestamp}'
RECURRENT_CHORD_PLOT_FILE = os.path.join(OUTPUT_DIR, 'recurrent_chord_node_avg.html')
SINGLE_EVENT_CHORD_PLOT_FILE = os.path.join(OUTPUT_DIR, 'single_event_chord_node_avg.html')
COLORBAR_LEGEND_FILE = os.path.join(OUTPUT_DIR, 'edge_colorbar_legend.png')

# Plotting Parameters
CONSTANT_EDGE_WIDTH = 0.5
GLOBAL_EDGE_ALPHA = 0.6 # Alpha for lines/edges

# Color Scale: Blue (0.0) to Red (1.0)
COLOR_SCALE_START = '#0000FF' # BLUE for Proportion = 0
COLOR_SCALE_END = '#FF0000'   # RED for Proportion = 1
PROPORTION_CMAP_MPL = LinearSegmentedColormap.from_list(
    "blue_to_red", [COLOR_SCALE_START, COLOR_SCALE_END]
)
logger.info(f"Defined colormap '{PROPORTION_CMAP_MPL.name}' mapping 0->{COLOR_SCALE_START} and 1->{COLOR_SCALE_END}")
PROPORTION_NAN_COLOR = '#000000' # BLACK for NaNs (or default color if mapping fails)

# Non-Linear Color Scaling (Applied to Nodes and fixed-color edges)
COLOR_TRANSITION_EXPONENT = 0.5

# Randomization Seed
SEED = 2025

# --- Helper Functions ---
def extract_coordinates_from_cds(cds_string):
    """Extract genomic coordinates from CDS string."""
    if not isinstance(cds_string, str): return None
    pattern = r'(chr\w+)_start(\d+)_end(\d+)'; match = re.search(pattern, cds_string)
    return {'chrom': match.group(1), 'start': int(match.group(2)), 'end': int(match.group(3))} if match else None

def map_cds_to_inversions(pairwise_df, inversion_df):
    """Map CDS strings to specific inversions and track inversion types."""
    logger.info("Mapping CDS to inversion types...");
    if 'CDS' not in pairwise_df.columns or pairwise_df['CDS'].isnull().all():
        logger.error("Missing 'CDS' column or all CDS values are null."); return {}, {}, {}
    unique_cds = pairwise_df['CDS'].dropna().unique();
    if len(unique_cds) == 0:
        logger.warning("No unique non-null CDS found."); return {}, {}, {}

    cds_coords = {cds: coords for cds in unique_cds if (coords := extract_coordinates_from_cds(cds))}
    if not cds_coords:
        logger.warning("Could not extract coordinates from any CDS string."); return {}, {}, {} # Still return empty dicts

    required_inv_cols = ['Start', 'End', '0_single_1_recur', 'Chromosome']
    if not all(col in inversion_df.columns for col in required_inv_cols):
        logger.error(f"Missing required columns in inversion info file: {required_inv_cols}. Cannot map types."); return {}, {}, {}

    logger.info("Cleaning inversion data...")
    # Create a copy to avoid SettingWithCopyWarning
    inversion_df_cleaned = inversion_df.copy()
    for col in ['Start', 'End', '0_single_1_recur']:
        try:
            inversion_df_cleaned[col] = pd.to_numeric(inversion_df_cleaned[col], errors='coerce')
        except Exception as e:
             logger.error(f"Error converting inversion column {col} to numeric: {e}"); return {}, {}, {}
    inversion_df_cleaned.dropna(subset=['Start', 'End', '0_single_1_recur', 'Chromosome'], inplace=True)
    if inversion_df_cleaned.empty:
        logger.error("Inversion df empty after cleaning required columns."); return {}, {}, {}
    inversion_df_cleaned['Start'] = inversion_df_cleaned['Start'].astype(int)
    inversion_df_cleaned['End'] = inversion_df_cleaned['End'].astype(int)
    inversion_df_cleaned['0_single_1_recur'] = inversion_df_cleaned['0_single_1_recur'].astype(int)

    recurrent_inv = inversion_df_cleaned[inversion_df_cleaned['0_single_1_recur'] == 1];
    single_event_inv = inversion_df_cleaned[inversion_df_cleaned['0_single_1_recur'] == 0]

    cds_to_type = {}; cds_to_inversion_id = {}; inversion_to_cds = defaultdict(list) # inversion_id/cds mapping not fully used here
    processed_cds_count = 0; mapped_cds_count = 0; ambiguous_map_count = 0; unknown_map_count = 0

    logger.info(f"Attempting to map {len(cds_coords)} unique CDS with valid coordinates...")
    for cds, coords in cds_coords.items():
        if not coords: continue
        processed_cds_count += 1
        chrom, start, end = coords['chrom'], coords['start'], coords['end']
        # Overlap check: CDS overlaps if CDS_start < Inv_end AND CDS_end > Inv_start
        rec_matches = recurrent_inv.loc[
            (recurrent_inv['Chromosome'] == chrom) &
            (start < recurrent_inv['End']) &
            (end > recurrent_inv['Start'])
        ]
        single_matches = single_event_inv.loc[
            (single_event_inv['Chromosome'] == chrom) &
            (start < single_event_inv['End']) &
            (end > single_event_inv['Start'])
        ]

        is_recurrent = len(rec_matches) > 0; is_single = len(single_matches) > 0

        type_assigned = 'unknown'
        if is_recurrent and not is_single: type_assigned = 'recurrent'
        elif is_single and not is_recurrent: type_assigned = 'single_event'
        elif is_recurrent and is_single: type_assigned = 'ambiguous'; ambiguous_map_count += 1
        else: type_assigned = 'unknown'; unknown_map_count +=1 # No overlap found

        cds_to_type[cds] = type_assigned
        mapped_cds_count += 1 # Count every processed CDS here, even if unknown

    logger.info(f"Finished mapping {mapped_cds_count} CDS to types out of {processed_cds_count} processed.")
    type_counts = pd.Series(cds_to_type).value_counts();
    logger.info(f"  Type counts: {type_counts.to_dict()}")
    if ambiguous_map_count > 0:
        logger.warning(f"Found {ambiguous_map_count} CDS mapping to both recurrent and single event inversions (marked 'ambiguous').")
    # Use the count from value_counts for unknown
    unknown_count_final = type_counts.get('unknown', 0)
    if unknown_count_final > 0:
         logger.warning(f"Found {unknown_count_final} CDS that did not map to any inversion (marked 'unknown').")
    return cds_to_type, cds_to_inversion_id, dict(inversion_to_cds)

def map_value_to_color_nonlinear(value, exponent, cmap):
    """
    Maps a value (0-1) to a hex color using a non-linear scale.
    Mainly used for NODE colors now, and for fixed-color EDGES.
    """
    if pd.isna(value):
        return PROPORTION_NAN_COLOR
    # Handle exact 0.0 case explicitly
    if np.isclose(value, 0.0):
         return COLOR_SCALE_START # Blue
    # Handle exact 1.0 case explicitly
    if np.isclose(value, 1.0):
        return COLOR_SCALE_END # Red

    # Clip value to be strictly within [0, 1] after checking endpoints
    normalized = max(0.0, min(1.0, value))

    # Apply non-linear transformation
    if exponent == 1.0:
        transformed = normalized
    else:
        # Ensure exponentiation is safe
        try:
            transformed = normalized ** exponent
        except ValueError: # Handle potential issues like 0**negative_exponent
             logger.warning(f"map_value_to_color_nonlinear: ValueError during exponentiation for value {value}, exponent {exponent}. Defaulting to NaN color.")
             return PROPORTION_NAN_COLOR

    # Clip again after transformation
    transformed = max(0.0, min(1.0, transformed))

    try:
        # Get RGBA color from the colormap
        rgba_color = cmap(transformed)
        # Convert RGB part to hex (ignore alpha)
        hex_color = to_hex(rgba_color[:3])
        return hex_color
    except Exception as e:
        logger.error(f"map_value_to_color_nonlinear: Error mapping value {value} (transformed {transformed:.4f}) to color: {e}")
        return PROPORTION_NAN_COLOR

# --- Chord Plot Specific Functions ---
def aggregate_pairwise_data_and_calc_node_colors(df, type_name, exponent, cmap):
    """
    Aggregates pairwise data, calculates node colors (Blue-Red 0-1 non-linear scale),
    and prepares edge data for colormapping based on proportion.
    Returns:
        - aggregated_df: DataFrame with edge info including 'proportion_non_identical'.
        - sorted_nodes_list: Alphabetically sorted list of unique node names.
        - node_to_color_map: Dictionary mapping node names to their calculated hex colors.
    """
    func_name = f"aggregate_and_calc_node_colors ({type_name})"
    logger.info(f"[{func_name}] Starting aggregation and node color calculation...")

    agg_cols = ['source', 'target', 'total_comparisons', 'non_identical_comparisons',
                'proportion_non_identical', 'median_omega', 'edge_width']
    empty_agg_df = pd.DataFrame(columns=agg_cols)
    nodes_set = set()
    node_to_color_map = {}

    if df.empty:
        logger.warning(f"[{func_name}] Input df empty. Returning empty results.");
        return empty_agg_df, sorted(list(nodes_set)), node_to_color_map

    logger.info(f"[{func_name}] Input df shape: {df.shape}.")

    # --- Grouping and Basic Aggregation ---
    df['Seq1'] = df['Seq1'].astype(str); df['Seq2'] = df['Seq2'].astype(str)
    df['haplotype_pair'] = df.apply(lambda row: tuple(sorted((row['Seq1'], row['Seq2']))), axis=1)

    # Define aggregation function for omega
    def count_non_identical(x):
        # Count where omega is numeric, not NA, and not -1
        numeric_x = pd.to_numeric(x, errors='coerce')
        return (numeric_x.notna() & (numeric_x != -1)).sum()

    agg_funcs = {'omega': ['median', count_non_identical, 'size']} # size gives total comparisons per pair

    logger.info(f"[{func_name}] Grouping by 'haplotype_pair' and aggregating...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore potential median of empty slice
        connection_stats = df.groupby('haplotype_pair').agg(agg_funcs).reset_index()

    connection_stats.columns = ['haplotype_pair', 'median_omega', 'non_identical_comparisons', 'total_comparisons']
    logger.info(f"[{func_name}] Aggregated into {len(connection_stats)} unique haplotype pairs.")

    if connection_stats.empty:
        logger.warning(f"[{func_name}] No aggregated stats found. Returning empty results.");
        return empty_agg_df, sorted(list(nodes_set)), node_to_color_map

    connection_stats['source'] = connection_stats['haplotype_pair'].apply(lambda x: x[0])
    connection_stats['target'] = connection_stats['haplotype_pair'].apply(lambda x: x[1])

    # --- Calculate Proportion Non-Identical for Edges ---
    logger.info(f"[{func_name}] Calculating edge proportion_non_identical...")
    # Use .loc to avoid potential SettingWithCopyWarning if connection_stats is a slice
    connection_stats.loc[:, 'total_comparisons_safe'] = connection_stats['total_comparisons'].replace(0, np.nan)
    connection_stats.loc[:, 'proportion_non_identical'] = (
        connection_stats['non_identical_comparisons'] / connection_stats['total_comparisons_safe']
    )
    initial_nan_count = connection_stats['proportion_non_identical'].isna().sum()
    # Fill NaNs resulting from division by zero (or NaN) with 0.0
    connection_stats.loc[:, 'proportion_non_identical'] = connection_stats['proportion_non_identical'].fillna(0.0)
    if initial_nan_count > 0:
        logger.info(f"[{func_name}] Filled {initial_nan_count} NaN proportions (likely from 0 total comparisons) with 0.0.")

    # Log the range of proportions, crucial for colormapping
    if connection_stats['proportion_non_identical'].notna().any():
        min_prop = connection_stats['proportion_non_identical'].min()
        max_prop = connection_stats['proportion_non_identical'].max()
        logger.info(f"[{func_name}] Calculated 'proportion_non_identical'. Min={min_prop:.4f}, Max={max_prop:.4f}")
    else:
        logger.warning(f"[{func_name}] 'proportion_non_identical' contains only NaNs after calculation and fillna.")

    connection_stats.drop(columns=['total_comparisons_safe'], inplace=True)
    # Ensure total_comparisons is int after potential NaN introduced by 'safe' version
    connection_stats['total_comparisons'] = connection_stats['total_comparisons'].fillna(0).astype(int)

    # --- Assign Constant Edge Width ---
    connection_stats['edge_width'] = CONSTANT_EDGE_WIDTH

    # --- Identify Nodes ---
    # Select only the necessary columns for the final aggregated_df
    aggregated_df = connection_stats[agg_cols].copy()
    nodes_set = set(aggregated_df['source']).union(set(aggregated_df['target']))
    sorted_nodes_list = sorted(list(nodes_set))
    logger.info(f"[{func_name}] Identified {len(sorted_nodes_list)} unique nodes.")

    # --- Calculate Average Proportion and Color for Each NODE ---
    logger.info(f"[{func_name}] Calculating average proportion and color for {len(sorted_nodes_list)} nodes...")
    node_avg_props = {}
    for node in sorted_nodes_list:
        # Find edges connected to this node
        connected_edges_mask = (aggregated_df['source'] == node) | (aggregated_df['target'] == node)
        connected_props = aggregated_df.loc[connected_edges_mask, 'proportion_non_identical']

        if connected_props.empty:
            # Node might exist but have no connections in the filtered/aggregated data
            avg_prop = 0.0 # Assign default color (blue)
            logger.warning(f"  Node '{node}': Found no connected edges in aggregated data. Assigning avg_prop=0.0.")
        else:
            avg_prop = connected_props.mean(skipna=True)
            # Check if mean resulted in NaN (e.g., all connected props were NaN, though unlikely now)
            if pd.isna(avg_prop):
                logger.warning(f"  Node '{node}': Avg prop calculated as NaN. Setting to 0.0.");
                avg_prop = 0.0

        node_avg_props[node] = avg_prop
        # Use the non-linear mapping function for node colors
        node_color = map_value_to_color_nonlinear(avg_prop, exponent, cmap)
        node_to_color_map[node] = node_color

    logger.info(f"[{func_name}] Node color calculation finished.")
    logger.info(f"[{func_name}] Aggregation finished. Final aggregated data shape: {aggregated_df.shape}")

    # Check proportion values again, especially for 0.0
    zero_prop_count = (aggregated_df['proportion_non_identical'] == 0.0).sum()
    logger.info(f"[{func_name}] Final aggregated data has {zero_prop_count} rows with proportion_non_identical == 0.0.")

    return aggregated_df, sorted_nodes_list, node_to_color_map


# --- Plotting Function (Corrected: Using valid MultiLine properties) ---
def plot_chord_diagram_node_avg_color(aggregated_data, sorted_nodes_list, node_to_color_map, filename):
    """
    Generates static chord diagram WITHOUT a colorbar key. Uses Blue-Red Colormap for Edges
    conditionally, focusing on edge_line_color.
    Nodes: Colored directly by average proportion, labeled with 1-based index.
    Edges: If proportion varies, line color mapped from 'proportion_non_identical'.
           If proportion is constant, line color set to the corresponding single color.
           Width is constant thin width. Randomized draw order.
    No interactivity. No plot title.
    """
    func_name = f"plot_chord_diagram_node_avg_color ({os.path.basename(filename)})"
    logger.info(f"[{func_name}] Starting plot generation...")

    # Ensure output file is set for Bokeh saving
    bk_plt.output_file(filename=filename, title="") # Use empty title for the HTML file itself

    # --- Handle Empty Data Cases ---
    if aggregated_data is None or aggregated_data.empty:
        logger.warning(f"[{func_name}] No aggregated data provided. Saving placeholder plot.")
        p = bk_plt.figure(width=850, height=850, title="No Data") # Add title to placeholder
        p.text(x=100, y=400, text=["No aggregated edge data available for this plot."], text_align="center", text_font_size="12pt")
        bk_plt.save(p)
        return
    if not sorted_nodes_list:
         logger.warning(f"[{func_name}] No nodes provided. Saving placeholder plot.")
         p = bk_plt.figure(width=850, height=850, title="No Nodes") # Add title to placeholder
         p.text(x=100, y=400, text=["No nodes found for this plot."], text_align="center", text_font_size="12pt")
         bk_plt.save(p)
         return

    logger.info(f"[{func_name}] Preparing data for plot (Nodes: {len(sorted_nodes_list)}, Edges: {len(aggregated_data)})")
    if len(sorted_nodes_list) > 200:
        logger.warning(f"[{func_name}] Plotting large number of nodes ({len(sorted_nodes_list)}). Performance might be affected.")

    # --- Prepare Edge Data (Make Copy!) ---
    plot_data = aggregated_data.copy()

    # --- Explicit Type Casting and Validation ---
    numeric_cols = ['edge_width', 'median_omega', 'proportion_non_identical',
                    'total_comparisons', 'non_identical_comparisons']
    for col in numeric_cols:
        if col in plot_data.columns:
            # Convert to numeric, coercing errors to NaN
            plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
            if plot_data[col].isna().any():
                logger.warning(f"  Column '{col}' contains NaNs after coercing to numeric.")
                # Apply specific filling strategies if needed
                if col == 'proportion_non_identical':
                    fill_val = 0.0
                    plot_data[col] = plot_data[col].fillna(fill_val)
                    logger.info(f"  Filled NaNs in '{col}' with {fill_val} for plotting.")
                elif col == 'edge_width':
                     fill_val = CONSTANT_EDGE_WIDTH
                     plot_data[col] = plot_data[col].fillna(fill_val)
                     logger.info(f"  Filled NaNs in '{col}' with {fill_val} for plotting.")
                # Add more specific NaN handling for other cols if necessary
        else:
             logger.warning(f"[{func_name}] Expected numeric column '{col}' not found in aggregated data.")
             # Add the column with a default if essential for plotting
             if col == 'edge_width':
                 plot_data[col] = CONSTANT_EDGE_WIDTH
                 logger.info(f" Added missing '{col}' column with default value {CONSTANT_EDGE_WIDTH}")
             elif col == 'proportion_non_identical':
                 plot_data[col] = 0.0 # Assume 0 if missing entirely
                 logger.warning(f" Added missing '{col}' column with default value 0.0")
             else:
                 plot_data[col] = np.nan # Add as NaN otherwise

    string_cols = ['source', 'target']
    for col in string_cols:
         if col in plot_data.columns:
             plot_data[col] = plot_data[col].astype(str)
         else:
             logger.error(f"[{func_name}] Required string column '{col}' not found. Cannot create plot.");
             # Save placeholder for error
             p = bk_plt.figure(width=850, height=850, title="Data Error")
             p.text(x=100, y=400, text=[f"Missing required column: {col}"], text_align="center", text_font_size="12pt")
             bk_plt.save(p)
             return

    # Ensure width exists after potential fillna
    plot_data['edge_width'] = plot_data['edge_width'].fillna(CONSTANT_EDGE_WIDTH)

    # --- Create Node Mapping and Dataset with Color ---
    logger.info(f"[{func_name}] Creating node mapping and HoloViews Nodes Dataset...")
    hap_to_index_map = {name: i + 1 for i, name in enumerate(sorted_nodes_list)} # 1-based index for labels
    nodes_df = pd.DataFrame({'haplotype': sorted_nodes_list})
    nodes_df['node_index'] = nodes_df['haplotype'].map(hap_to_index_map)
    # Assign node colors directly using the pre-calculated map, handle missing nodes
    nodes_df['node_color'] = nodes_df['haplotype'].map(node_to_color_map).fillna(PROPORTION_NAN_COLOR)

    # Log node color distribution summary
    node_color_counts = nodes_df['node_color'].value_counts()
    logger.info(f"[{func_name}] Node color value counts (Top 5):\n{node_color_counts.head().to_string()}")
    if node_color_counts.get(PROPORTION_NAN_COLOR, 0) > 0:
        logger.warning(f"[{func_name}] {node_color_counts.get(PROPORTION_NAN_COLOR, 0)} nodes have NaN/fallback color {PROPORTION_NAN_COLOR}.")
    # Check if all nodes are a specific color (useful for debugging single-value cases)
    if len(node_color_counts) == 1:
         single_color = node_color_counts.index[0]
         logger.info(f"[{func_name}] Confirmed: All {len(nodes_df)} nodes assigned the same color: {single_color}.")


    # Create the HoloViews Nodes dataset
    try:
        nodes_dataset = hv.Dataset(nodes_df, kdims='haplotype', vdims=['node_index', 'node_color'])
    except Exception as e_node_ds:
         logger.error(f"[{func_name}] Error creating HoloViews Nodes Dataset: {e_node_ds}", exc_info=True)
         # Save placeholder for error
         p = bk_plt.figure(width=850, height=850, title="Node Data Error")
         p.text(x=100, y=400, text=["Failed to create node dataset."], text_align="center", text_font_size="12pt")
         bk_plt.save(p)
         return


    # --- Define Value Dimensions (vdims) for Edges ---
    # Base vdims needed regardless of coloring method
    base_vdims = ['edge_width', 'total_comparisons', 'non_identical_comparisons', 'median_omega']
    # Check if proportion column exists and is needed for colormapping or fixed color calculation
    vdims = base_vdims # Start with base
    if 'proportion_non_identical' in plot_data.columns:
        vdims = vdims + ['proportion_non_identical'] # Add proportion if present
    else:
        logger.warning(f"[{func_name}] 'proportion_non_identical' column not found in plot_data for vdims.")

    # Final check for required columns in the edge data before creating hv.Chord
    required_hv_cols = ['source', 'target'] + vdims
    missing_hv_cols = [col for col in required_hv_cols if col not in plot_data.columns]
    if missing_hv_cols:
        logger.error(f"[{func_name}] Plotting data missing required columns for HoloViews Edges: {missing_hv_cols}.")
        # Save placeholder for error
        p = bk_plt.figure(width=850, height=850, title="Edge Data Error")
        p.text(x=100, y=400, text=[f"Missing required edge columns: {missing_hv_cols}"], text_align="center", text_font_size="12pt")
        bk_plt.save(p)
        return

    # --- *** RANDOMIZE EDGE ORDER *** ---
    logger.info(f"[{func_name}] Shuffling edge data using SEED={SEED} before passing to HoloViews...")
    plot_data_shuffled = plot_data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # --- *** Check Proportion Range for Conditional Coloring *** ---
    min_plot_prop = np.nan
    max_plot_prop = np.nan
    use_edge_colormap = False # Default to not using colormap
    fixed_edge_color = PROPORTION_NAN_COLOR # Default fixed color if proportion is missing or all NaN

    if 'proportion_non_identical' in plot_data_shuffled.columns:
        # Drop NaNs *before* calculating min/max for range check
        proportions_for_range = plot_data_shuffled['proportion_non_identical'].dropna()
        if not proportions_for_range.empty:
            min_plot_prop = proportions_for_range.min()
            max_plot_prop = proportions_for_range.max()
            logger.info(f"[{func_name}] Shuffled data 'proportion_non_identical' range (non-NaN): Min={min_plot_prop:.4f}, Max={max_plot_prop:.4f}")

            # Check if the range is non-zero (use tolerance for floating point)
            if not np.isclose(min_plot_prop, max_plot_prop):
                use_edge_colormap = True
                logger.info(f"[{func_name}] Proportion range is non-zero. EDGE LINE COLORING: Using colormap based on 'proportion_non_identical'.")
            else:
                # Range is zero, use a fixed color
                logger.info(f"[{func_name}] Proportion range is zero (all values={min_plot_prop:.4f}). EDGE LINE COLORING: Using fixed color.")
                # Calculate the single color based on the constant proportion value
                fixed_edge_color = map_value_to_color_nonlinear(min_plot_prop, COLOR_TRANSITION_EXPONENT, PROPORTION_CMAP_MPL)
                logger.info(f"[{func_name}] Fixed edge line color calculated: {fixed_edge_color}")
        else:
             logger.warning(f"[{func_name}] 'proportion_non_identical' column contains only NaNs. EDGE LINE COLORING: Using fallback fixed color {fixed_edge_color}.")
             # fixed_edge_color is already set to PROPORTION_NAN_COLOR
    else:
        logger.warning(f"[{func_name}] 'proportion_non_identical' column missing. EDGE LINE COLORING: Using fallback fixed color {fixed_edge_color}.")
        # fixed_edge_color is already set to PROPORTION_NAN_COLOR


    # --- Create Chord object ---
    logger.info(f"[{func_name}] Creating hv.Chord object...")
    try:
        # Pass only the necessary columns to the Chord constructor
        # Ensure the order matches ['source', 'target'] + vdims
        chord_data_subset = plot_data_shuffled[['source', 'target'] + vdims]
        chord_element = hv.Chord((chord_data_subset, nodes_dataset), vdims=vdims)
    except Exception as e:
        logger.error(f"[{func_name}] Error creating hv.Chord object: {e}", exc_info=True)
        logger.error(f"Columns present in chord_data_subset: {list(chord_data_subset.columns)}")
        logger.error(f"Nodes dataset info: {nodes_dataset}")
        # Save placeholder for error
        p = bk_plt.figure(width=850, height=850, title="Chord Creation Error")
        p.text(x=100, y=400, text=["Failed to create HoloViews Chord object."], text_align="center", text_font_size="12pt")
        bk_plt.save(p)
        return

    # --- Apply HoloViews Options (Corrected: Using valid MultiLine properties) ---
    logger.info(f"[{func_name}] Applying HoloViews options...")
    # Base options dictionary - REMOVED edge_fill_alpha
    chord_opts_dict = {
        'labels': 'node_index',             # Use the 1-based index for labels
        'node_color': 'node_color',         # Use the pre-assigned color from nodes_dataset
        'node_size': 9,
        'edge_line_width': 'edge_width',    # Use the calculated edge width
        # Styling and Interactivity
        'tools': [],                        # Disable default Bokeh tools initially
        'width': 850, 'height': 850,        # Plot dimensions
        'toolbar': None,                    # Disable the Bokeh toolbar
        'xaxis': None, 'yaxis': None,       # Hide axes
        'label_text_font_size': '8pt'       # Font size for node labels
    }

    # *** Set Edge LINE Color and Alpha Options Based on Range ***
    if use_edge_colormap:
        # Apply colormap based on the proportion column for the LINE color
        logger.info(f"[{func_name}] Setting edge options for colormapping LINE via 'proportion_non_identical'.")
        chord_opts_dict['edge_cmap'] = PROPORTION_CMAP_MPL
        chord_opts_dict['edge_line_color'] = 'proportion_non_identical' # Map LINE color
        chord_opts_dict['edge_line_alpha'] = GLOBAL_EDGE_ALPHA       # Apply alpha to line when using cmap

    else:
        # Use the pre-calculated fixed edge color for the LINE
        logger.info(f"[{func_name}] Setting edge options for fixed LINE color: {fixed_edge_color}")
        # Don't set edge_cmap if using a fixed color string

        # Explicitly set the edge LINE color to the fixed color.
        chord_opts_dict['edge_line_color'] = fixed_edge_color
        # Set line alpha for fixed color lines
        chord_opts_dict['edge_line_alpha'] = GLOBAL_EDGE_ALPHA
        # DO NOT SET edge_color or edge_fill_alpha as they are invalid for MultiLine or cause issues

    logger.info(f"[{func_name}] Final Chord options being applied: {chord_opts_dict}")
    try:
        # Apply the options dictionary to the Chord element
        final_hv_plot = chord_element.opts(opts.Chord(**chord_opts_dict))
    except Exception as e:
        logger.error(f"[{func_name}] Error applying HoloViews options: {e}", exc_info=True)
        # Save placeholder for error
        p = bk_plt.figure(width=850, height=850, title="Options Error")
        p.text(x=100, y=400, text=["Failed to apply HoloViews options."], text_align="center", text_font_size="12pt")
        bk_plt.save(p)
        return

    # --- Render to Bokeh Object ---
    logger.info(f"[{func_name}] Rendering HoloViews plot to Bokeh object...")
    try:
        # Convert the HoloViews object to a Bokeh plot object
        bokeh_plot = hv.render(final_hv_plot, backend='bokeh')

        # Validate that rendering produced a Bokeh Model
        if not isinstance(bokeh_plot, bk_Model):
             logger.error(f"[{func_name}] Rendering did not return a Bokeh Model. Type: {type(bokeh_plot)}")
             # Save placeholder for error
             p = bk_plt.figure(width=850, height=850, title="Rendering Error")
             p.text(x=100, y=400, text=[f"Failed to render plot: Invalid type {type(bokeh_plot)}."], text_align="center", text_font_size="12pt")
             bk_plt.save(p) # Overwrite with error message
             return
        logger.info(f"[{func_name}] Rendering successful. Bokeh plot type: {type(bokeh_plot)}")
    except Exception as e:
        logger.error(f"[{func_name}] Error rendering HoloViews plot to Bokeh: {e}", exc_info=True)
        # Save placeholder for error - CATCHING THE ERROR HERE NOW
        p = bk_plt.figure(width=850, height=850, title="Rendering Error")
        # Display the specific error message in the placeholder
        error_msg = f"Failed to render plot (Exception):\n{type(e).__name__}: {e}"
        p.text(x=100, y=400, text=[error_msg], text_align="center", text_font_size="10pt")
        bk_plt.save(p) # Overwrite with error message
        return

    # --- Save Final Bokeh Plot ---
    try:
        logger.info(f"[{func_name}] Saving final Bokeh plot to: {filename}")
        bk_plt.save(bokeh_plot) # Saves the plot to the file specified by bk_plt.output_file()
        logger.info(f"[{func_name}] Plot saved successfully.")
    except Exception as e_save:
        logger.error(f"[{func_name}] Failed to save final Bokeh plot '{filename}': {e_save}", exc_info=True)


# --- Colorbar Generation Function ---
def save_colorbar_legend(filename, cmap, exponent, label):
    """Saves a colorbar legend image based on the colormap and exponent."""
    logger.info(f"Generating colorbar legend: {filename}")
    fig, ax = plt.subplots(figsize=(1.5, 6)) # Adjust figure size as needed
    fig.subplots_adjust(left=0.1, right=0.5) # Adjust spacing

    # Define the forward and inverse functions for the non-linear norm
    # Clip input to [0, 1] before applying functions
    def forward(x): return np.power(np.clip(x, 0, 1), exponent)
    def inverse(x): return np.power(np.clip(x, 0, 1), 1.0 / exponent if exponent != 0 else 1.0) # Avoid division by zero

    try:
        norm = FuncNorm((forward, inverse), vmin=0, vmax=1)
    except Exception as e_norm:
        logger.error(f"Failed to create FuncNorm for colorbar: {e_norm}", exc_info=True)
        plt.close(fig)
        return

    try:
        # Create a ScalarMappable with the norm and colormap
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        # Create the colorbar associated with the ScalarMappable
        cb = plt.colorbar(mappable, cax=ax, orientation='vertical', ticks=np.linspace(0, 1, 6)) # Example: 6 ticks
        cb.set_label(label, size=10)
        cb.ax.tick_params(labelsize=8)
    except Exception as e_cb:
        logger.error(f"Failed to create colorbar: {e_cb}", exc_info=True)
        plt.close(fig)
        return

    try:
        # Save the figure
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Colorbar legend saved successfully to {filename}")
    except Exception as e_save:
        logger.error(f"Failed to save colorbar legend '{filename}': {e_save}", exc_info=True)
    finally:
        # Ensure the plot is closed to free memory
        plt.close(fig)


# --- Main Execution ---
def main():
    """Main execution function."""
    logger.info("--- Starting Chord Plot Script (Node Avg Color, Blue-Red CMap 0-1, Separate Key, Explicit Line Color, No Fill Alpha) ---")
    hv.extension('bokeh', logo=False)
    logger.info("HoloViews Bokeh extension activated.")

    # Ensure output directory exists
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory ensured: {OUTPUT_DIR}")
    except OSError as e:
        logger.error(f"Failed to create output directory '{OUTPUT_DIR}': {e}"); return

    # Generate and save the colorbar legend image
    save_colorbar_legend(
        COLORBAR_LEGEND_FILE, PROPORTION_CMAP_MPL, COLOR_TRANSITION_EXPONENT,
        f"Prop. Non-Identical (ω ≠ -1)\n(Blue=0, Red=1, Exp={COLOR_TRANSITION_EXPONENT})"
    )

    # --- Load Data ---
    try:
        logger.info(f"Loading pairwise data from: {PAIRWISE_FILE}")
        pairwise_df = pd.read_csv(PAIRWISE_FILE); logger.info(f"Loaded {len(pairwise_df):,} rows.")
        logger.info(f"Loading inversion info from: {INVERSION_FILE}")
        inversion_df = pd.read_csv(INVERSION_FILE, sep='\t'); logger.info(f"Loaded {len(inversion_df):,} rows.")
    except FileNotFoundError as e:
        logger.error(f"Error loading input file: {e}. ABORTING."); return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading input files: {e}. ABORTING.", exc_info=True); return

    # --- Step 1: Map CDS to Inversion Types ---
    logger.info("--- Step 1: Mapping CDS to Inversion Types ---")
    cds_to_type, _, _ = map_cds_to_inversions(pairwise_df, inversion_df)
    if not cds_to_type:
        logger.warning("CDS mapping resulted in an empty map. Proceeding, but 'inversion_type' will be mostly 'unknown'.")
    # Use .map() for mapping, fill missing CDS entries with 'unknown'
    pairwise_df['inversion_type'] = pairwise_df['CDS'].map(cds_to_type).fillna('unknown')
    logger.info("Added 'inversion_type' column to pairwise data.")

    # --- Step 2: Filtering Pairwise Data ---
    logger.info("--- Step 2: Filtering Pairwise Data ---")
    required_cols = ['inversion_type', 'Group1', 'Group2', 'omega', 'Seq1', 'Seq2']
    missing_base_cols = [col for col in required_cols if col not in pairwise_df.columns]
    if missing_base_cols:
        logger.error(f"Pairwise data missing required columns: {missing_base_cols}. ABORTING."); return

    logger.info("Converting key columns (Group1, Group2, omega) to numeric...")
    try:
        # Use .loc to modify DataFrame in place safely
        pairwise_df.loc[:, 'Group1'] = pd.to_numeric(pairwise_df['Group1'], errors='coerce')
        pairwise_df.loc[:, 'Group2'] = pd.to_numeric(pairwise_df['Group2'], errors='coerce')
        pairwise_df.loc[:, 'omega'] = pd.to_numeric(pairwise_df['omega'], errors='coerce')
    except Exception as e:
        logger.error(f"Error converting columns to numeric: {e}. ABORTING.", exc_info=True); return

    logger.info("Applying filters to pairwise data...")
    filter_mask = (
        (pairwise_df['inversion_type'].isin(['recurrent', 'single_event'])) & # Only these types
        (pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1) &          # Both groups are 1
        pairwise_df['omega'].notna() & (pairwise_df['omega'] != 99) &          # Omega is valid
        pairwise_df['Seq1'].notna() & pairwise_df['Seq2'].notna() &            # Sequences are not null
        (pairwise_df['Seq1'] != pairwise_df['Seq2'])                           # Exclude self-comparisons
    )
    filtered_df = pairwise_df[filter_mask].copy() # Create a copy to avoid chained assignment issues
    num_original = len(pairwise_df); num_filtered = len(filtered_df)
    logger.info(f"Applied filters. Kept {num_filtered:,} rows out of {num_original:,} original rows.")
    if filtered_df.empty:
        logger.error("No pairs found matching all filter criteria. Cannot generate plots. ABORTING."); return
    logger.info(f"Value counts for 'inversion_type' in filtered data:\n{filtered_df['inversion_type'].value_counts().to_string()}")

    # --- Step 3: Processing Data for Recurrent Inversions ---
    logger.info("--- Step 3: Processing Data for Recurrent Inversions ---")
    recurrent_pairs_df = filtered_df[filtered_df['inversion_type'] == 'recurrent'].copy()
    logger.info(f"Processing {len(recurrent_pairs_df):,} pairs for Recurrent plot.")
    rec_agg_data, rec_sorted_nodes, rec_node_to_color_map = aggregate_pairwise_data_and_calc_node_colors(
        recurrent_pairs_df, "recurrent", COLOR_TRANSITION_EXPONENT, PROPORTION_CMAP_MPL
    )
    if rec_agg_data.empty or not rec_sorted_nodes:
         logger.warning("Aggregation for recurrent pairs resulted in empty data or no nodes. Plot may be empty or placeholder.")
    else:
         logger.info(f"Recurrent aggregation successful. Shape: {rec_agg_data.shape}, Nodes: {len(rec_sorted_nodes)}")

    # --- Step 4: Processing Data for Single-Event Inversions ---
    logger.info("--- Step 4: Processing Data for Single-Event Inversions ---")
    single_event_pairs_df = filtered_df[filtered_df['inversion_type'] == 'single_event'].copy()
    logger.info(f"Processing {len(single_event_pairs_df):,} pairs for Single-Event plot.")
    single_agg_data, single_sorted_nodes, single_node_to_color_map = aggregate_pairwise_data_and_calc_node_colors(
        single_event_pairs_df, "single_event", COLOR_TRANSITION_EXPONENT, PROPORTION_CMAP_MPL
    )
    if single_agg_data.empty or not single_sorted_nodes:
        logger.warning("Aggregation for single-event pairs resulted in empty data or no nodes. Plot may be empty or placeholder.")
    else:
        logger.info(f"Single-Event aggregation successful. Shape: {single_agg_data.shape}, Nodes: {len(single_sorted_nodes)}")

    # --- Step 5: Generating Static Chord Diagrams ---
    logger.info("--- Step 5: Generating Static Chord Diagrams ---")

    logger.info("Generating Recurrent Chord Diagram...")
    plot_chord_diagram_node_avg_color(
        rec_agg_data, rec_sorted_nodes, rec_node_to_color_map, RECURRENT_CHORD_PLOT_FILE
    )

    logger.info("Generating Single-Event Chord Diagram...")
    plot_chord_diagram_node_avg_color(
        single_agg_data, single_sorted_nodes, single_node_to_color_map, SINGLE_EVENT_CHORD_PLOT_FILE
    )

    logger.info("--- Chord plot generation script finished ---")

# --- Script Entry Point ---
if __name__ == "__main__":
    logger.info(f"Executing script: {__file__}")
    try:
        main()
    except Exception as main_e:
        # Catch any unexpected errors during main execution
        logger.critical(f"An unhandled exception occurred in the main script execution: {main_e}", exc_info=True)
    finally:
        # Log script completion regardless of success or failure
        logger.info("--- Script execution ended ---")
