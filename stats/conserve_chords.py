import pandas as pd
import numpy as np
import re
import logging
import sys
from collections import defaultdict
import holoviews as hv
from holoviews import opts
import bokeh.plotting as bk_plt
from bokeh.model import Model as bk_Model
from matplotlib.colors import LinearSegmentedColormap, to_hex, FuncNorm
import matplotlib.pyplot as plt
import os
import warnings
import math
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[ logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger('chord_plot_node_avg_color')
logger.info("--- Starting New Script Run (Node Avg Color, 0-1 Scale, Separate Key) ---")


PAIRWISE_FILE = 'all_pairwise_results.csv'
INVERSION_FILE = 'inv_info.csv'
timestamp = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f'chord_plots_node_avg_color_{timestamp}'
RECURRENT_CHORD_PLOT_FILE = os.path.join(OUTPUT_DIR, 'recurrent_chord_node_avg.html')
SINGLE_EVENT_CHORD_PLOT_FILE = os.path.join(OUTPUT_DIR, 'single_event_chord_node_avg.html')
COLORBAR_LEGEND_FILE = os.path.join(OUTPUT_DIR, 'edge_colorbar_legend.png')


CONSTANT_EDGE_WIDTH = 0.5
GLOBAL_EDGE_ALPHA = 0.6
COLOR_SCALE_START = '#D3D3D3'
COLOR_SCALE_END = '#FF0000'
PROPORTION_CMAP_MPL = LinearSegmentedColormap.from_list(
    "grey_to_red", [COLOR_SCALE_START, COLOR_SCALE_END]
)
PROPORTION_NAN_COLOR = '#808080'
COLOR_TRANSITION_EXPONENT = 0.5
SEED = 2025


def extract_coordinates_from_cds(cds_string):
    if not isinstance(cds_string, str): return None
    pattern = r'(chr\w+)_start(\d+)_end(\d+)'; match = re.search(pattern, cds_string)
    return {'chrom': match.group(1), 'start': int(match.group(2)), 'end': int(match.group(3))} if match else None

def map_cds_to_inversions(pairwise_df, inversion_df):
    logger.info("Mapping CDS to inversion types...");
    if 'CDS' not in pairwise_df.columns or pairwise_df['CDS'].isnull().all(): logger.error("Missing 'CDS' column or all CDS values are null."); return {}, {}, {}
    unique_cds = pairwise_df['CDS'].dropna().unique();
    if len(unique_cds) == 0: logger.warning("No unique non-null CDS found."); return {}, {}, {}

    cds_coords = {cds: coords for cds in unique_cds if (coords := extract_coordinates_from_cds(cds))}
    if not cds_coords: logger.warning("Could not extract coordinates from any CDS string.");

    required_inv_cols = ['Start', 'End', '0_single_1_recur', 'Chromosome']
    if not all(col in inversion_df.columns for col in required_inv_cols):
        logger.error(f"Missing required columns in inversion info file: {required_inv_cols}. Cannot map types."); return {}, {}, {}

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
        if not coords: continue
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

        if processed_cds_count % 1000 == 0:
            logger.debug(f"  Mapped {mapped_cds_count}/{processed_cds_count} CDS considered...")

    logger.info(f"Finished mapping {mapped_cds_count} CDS to types out of {processed_cds_count} processed.")
    type_counts = pd.Series(cds_to_type).value_counts();
    logger.info(f"  Type counts: {type_counts.to_dict()}")
    if 'ambiguous' in type_counts: logger.warning(f"Found {type_counts.get('ambiguous', 0)} CDS mapping to both recurrent and single event inversions.")
    if 'unknown' in type_counts: logger.warning(f"Found {type_counts.get('unknown', 0)} CDS that did not map to any inversion.")
    return cds_to_type, cds_to_inversion_id, dict(inversion_to_cds)


def map_value_to_color_nonlinear(value, exponent, cmap):
    if pd.isna(value):
        return PROPORTION_NAN_COLOR

    normalized = max(0.0, min(1.0, value))

    if exponent == 1.0:
        transformed = normalized
    else:
        transformed = normalized ** exponent
        transformed = max(0.0, min(1.0, transformed))

    try:
        rgba_color = cmap(transformed)
        hex_color = to_hex(rgba_color[:3])
        return hex_color
    except Exception as e:
        logger.error(f"Error mapping value {value} (transformed {transformed:.3f}) to color: {e}")
        return PROPORTION_NAN_COLOR


def aggregate_pairwise_data_and_calc_node_colors(df, type_name, exponent, cmap):
    func_name = f"aggregate_and_calc_node_colors ({type_name})"
    logger.info(f"[{func_name}] Starting aggregation and node color calculation...")
    agg_cols = ['source', 'target', 'total_comparisons', 'non_identical_comparisons',
                'proportion_non_identical', 'median_omega',
                'edge_width', 'edge_color']
    empty_agg_df = pd.DataFrame(columns=agg_cols)
    nodes_set = set()
    node_to_color_map = {}
    if df.empty:
        logger.warning(f"[{func_name}] Input df empty.");
        return empty_agg_df, sorted(list(nodes_set)), node_to_color_map
    logger.debug(f"[{func_name}] Input df shape: {df.shape}")

    df['Seq1'] = df['Seq1'].astype(str); df['Seq2'] = df['Seq2'].astype(str)
    df['haplotype_pair'] = df.apply(lambda row: tuple(sorted((row['Seq1'], row['Seq2']))), axis=1)
    def count_non_identical(x):
        numeric_x = pd.to_numeric(x, errors='coerce')
        return (numeric_x.notna() & (numeric_x != -1)).sum()

    agg_funcs = {'omega': ['median', count_non_identical, 'size']}

    logger.debug(f"[{func_name}] Grouping by 'haplotype_pair'...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        connection_stats = df.groupby('haplotype_pair').agg(agg_funcs).reset_index()

    connection_stats.columns = ['haplotype_pair', 'median_omega', 'non_identical_comparisons', 'total_comparisons']
    logger.info(f"[{func_name}] Aggregated into {len(connection_stats)} unique haplotype pairs.")
    if connection_stats.empty:
        logger.warning(f"[{func_name}] No aggregated stats found.");
        return empty_agg_df, sorted(list(nodes_set)), node_to_color_map

    connection_stats['source'] = connection_stats['haplotype_pair'].apply(lambda x: x[0])
    connection_stats['target'] = connection_stats['haplotype_pair'].apply(lambda x: x[1])

    logger.debug(f"[{func_name}] Calculating edge proportion_non_identical...")
    connection_stats['total_comparisons_safe'] = connection_stats['total_comparisons'].replace(0, np.nan)
    connection_stats['proportion_non_identical'] = (
        connection_stats['non_identical_comparisons'] / connection_stats['total_comparisons_safe']
    )
    connection_stats['proportion_non_identical'] = connection_stats['proportion_non_identical'].fillna(0.0)
    connection_stats.drop(columns=['total_comparisons_safe'], inplace=True)
    connection_stats['total_comparisons'] = connection_stats['total_comparisons'].fillna(0).astype(int)

    connection_stats['edge_width'] = CONSTANT_EDGE_WIDTH

    logger.debug(f"[{func_name}] Assigning edge colors non-linearly (Exponent={exponent}, Scale=0-1)...")
    connection_stats['edge_color'] = connection_stats['proportion_non_identical'].apply(
        map_value_to_color_nonlinear, args=(exponent, cmap)
    )

    aggregated_df = connection_stats[agg_cols].copy()
    nodes_set = set(aggregated_df['source']).union(set(aggregated_df['target']))
    sorted_nodes_list = sorted(list(nodes_set))

    logger.info(f"[{func_name}] Calculating average proportion and color for {len(sorted_nodes_list)} nodes...")
    node_avg_props = {}
    for node in sorted_nodes_list:
        connected_edges_mask = (aggregated_df['source'] == node) | (aggregated_df['target'] == node)
        connected_props = aggregated_df.loc[connected_edges_mask, 'proportion_non_identical']

        if connected_props.empty:
            avg_prop = 0.0
        else:
            avg_prop = connected_props.mean(skipna=True)
            if pd.isna(avg_prop):
                avg_prop = 0.0

        node_avg_props[node] = avg_prop
        node_color = map_value_to_color_nonlinear(avg_prop, exponent, cmap)
        node_to_color_map[node] = node_color

    logger.info(f"[{func_name}] Node color calculation finished.")
    logger.info(f"[{func_name}] Aggregation finished. Final aggregated data shape: {aggregated_df.shape}")

    return aggregated_df, sorted_nodes_list, node_to_color_map


def plot_chord_diagram_node_avg_color(aggregated_data, sorted_nodes_list, node_to_color_map, filename):
    func_name = f"plot_chord_diagram_node_avg_color ({os.path.basename(filename)})"
    logger.info(f"[{func_name}] Starting plot generation...")

    bk_plt.output_file(filename=filename, title="")

    if aggregated_data is None or aggregated_data.empty:
        logger.warning(f"[{func_name}] No aggregated data provided. Skipping plot creation for '{filename}'.")
        bk_plt.save(bk_plt.figure(width=850, height=850, title=None))
        return
    if not sorted_nodes_list:
         logger.warning(f"[{func_name}] No nodes provided. Skipping plot creation for '{filename}'.")
         bk_plt.save(bk_plt.figure(width=850, height=850, title=None))
         return

    logger.info(f"[{func_name}] Generating plot (Nodes: {len(sorted_nodes_list)})")
    if len(sorted_nodes_list) > 200: logger.warning(f"[{func_name}] Plotting large number of nodes ({len(sorted_nodes_list)}).")

    plot_data = aggregated_data.copy()
    numeric_cols = ['edge_width', 'median_omega', 'proportion_non_identical',
                    'total_comparisons', 'non_identical_comparisons']
    for col in numeric_cols:
        if col in plot_data.columns: plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
        else: logger.warning(f"[{func_name}] Expected numeric column '{col}' not found."); plot_data[col] = np.nan
    string_cols = ['source', 'target', 'edge_color']
    for col in string_cols:
         if col in plot_data.columns: plot_data[col] = plot_data[col].astype(str)
         else: logger.error(f"[{func_name}] Required string column '{col}' not found."); return None
    plot_data['edge_width'] = plot_data['edge_width'].fillna(0.0)
    plot_data['edge_color'] = plot_data['edge_color'].fillna(PROPORTION_NAN_COLOR)

    hap_to_index_map = {name: i + 1 for i, name in enumerate(sorted_nodes_list)}
    nodes_df = pd.DataFrame({'haplotype': sorted_nodes_list})
    nodes_df['node_index'] = nodes_df['haplotype'].map(hap_to_index_map)
    nodes_df['node_color'] = nodes_df['haplotype'].map(node_to_color_map).fillna(PROPORTION_NAN_COLOR)

    nodes_dataset = hv.Dataset(nodes_df, kdims='haplotype', vdims=['node_index', 'node_color'])
    logger.debug(f"Node dataset created with index and color columns.")

    required_hv_cols = ['source', 'target',
                        'edge_width', 'edge_color',
                        'proportion_non_identical', 'total_comparisons',
                        'non_identical_comparisons', 'median_omega']
    vdims = [col for col in required_hv_cols if col not in ['source', 'target']]
    missing_hv_cols = [col for col in required_hv_cols if col not in plot_data.columns]
    if missing_hv_cols:
        logger.error(f"[{func_name}] Aggregated data missing required columns for vdims: {missing_hv_cols}. Cannot plot.")
        return

    logger.debug(f"[{func_name}] Shuffling edge data using seed {SEED}.")
    plot_data_shuffled = plot_data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    logger.debug(f"[{func_name}] Creating hv.Chord object...")
    try:
        chord_element = hv.Chord((plot_data_shuffled, nodes_dataset), vdims=vdims)
        logger.debug(f"[{func_name}] hv.Chord object created successfully.")
    except Exception as e:
        logger.error(f"[{func_name}] Error creating hv.Chord object: {e}", exc_info=True)
        return

    logger.debug(f"[{func_name}] Applying HoloViews options...")
    try:
        final_hv_plot = chord_element.opts(
            opts.Chord(
                labels='node_index',
                node_color='node_color',
                node_size=9,
                edge_color='edge_color',
                edge_line_width='edge_width',
                edge_alpha=GLOBAL_EDGE_ALPHA,
                tools=[], width=850, height=850, toolbar=None,
                xaxis=None, yaxis=None,
                label_text_font_size='8pt'
            )
        )
        logger.debug(f"[{func_name}] HoloViews options applied.")
    except Exception as e:
        logger.error(f"[{func_name}] Error applying HoloViews options: {e}", exc_info=True)
        return

    logger.debug(f"[{func_name}] Rendering HoloViews plot to Bokeh object...")
    try:
        bokeh_plot = hv.render(final_hv_plot, backend='bokeh')
        if not isinstance(bokeh_plot, bk_Model):
             logger.error(f"[{func_name}] Rendering did not return a Bokeh Model. Type: {type(bokeh_plot)}")
             return
    except Exception as e:
        logger.error(f"[{func_name}] Error rendering HoloViews plot to Bokeh: {e}", exc_info=True)
        return

    try:
        logger.info(f"[{func_name}] Saving final Bokeh plot to: {filename}")
        bk_plt.save(bokeh_plot)
        logger.info(f"[{func_name}] Plot saved successfully.")
    except Exception as e_save:
        logger.error(f"[{func_name}] Failed to save final Bokeh plot '{filename}': {e_save}", exc_info=True)


def save_colorbar_legend(filename, cmap, exponent, label):
    logger.info(f"Generating colorbar legend: {filename}")
    fig, ax = plt.subplots(figsize=(1, 6))
    fig.subplots_adjust(right=0.5)

    def forward(x):
        return np.power(np.clip(x, 0, 1), exponent)
    def inverse(x):
        return np.power(np.clip(x, 0, 1), 1.0/exponent)

    norm = FuncNorm((forward, inverse), vmin=0, vmax=1)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical',
        ticks=np.linspace(0, 1, 6)
    )
    cb.set_label(label, size=10)
    cb.ax.tick_params(labelsize=8)

    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"Colorbar legend saved successfully to {filename}")
    except Exception as e:
        logger.error(f"Failed to save colorbar legend '{filename}': {e}", exc_info=True)
    finally:
        plt.close(fig)


def main():
    logger.info("--- Starting Chord Plot Script (Node Avg Color, 0-1 Scale, Separate Key) ---")
    hv.extension('bokeh', logo=False)

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory set to: {OUTPUT_DIR}")
    except OSError as e: logger.error(f"Failed to create output directory '{OUTPUT_DIR}': {e}"); return

    save_colorbar_legend(
        COLORBAR_LEGEND_FILE,
        PROPORTION_CMAP_MPL,
        COLOR_TRANSITION_EXPONENT,
        f"Prop. Non-Identical (ω ≠ -1)\n(Non-linear Scale, Exp={COLOR_TRANSITION_EXPONENT})"
    )

    try:
        logger.info(f"Loading pairwise data from: {PAIRWISE_FILE}")
        pairwise_df = pd.read_csv(PAIRWISE_FILE)
        logger.info(f"Loaded {len(pairwise_df):,} rows from pairwise data.")
        logger.info(f"Loading inversion info from: {INVERSION_FILE}")
        inversion_df = pd.read_csv(INVERSION_FILE)
        logger.info(f"Loaded {len(inversion_df):,} rows from inversion info.")
    except FileNotFoundError as e: logger.error(f"Error loading input file: {e}."); return
    except Exception as e: logger.error(f"An unexpected error occurred loading input files: {e}", exc_info=True); return

    logger.info("--- Step 1: Mapping CDS to Inversion Types ---")
    cds_to_type, _, _ = map_cds_to_inversions(pairwise_df, inversion_df)
    if not cds_to_type: logger.warning("CDS mapping resulted in an empty map.")
    pairwise_df['inversion_type'] = pairwise_df['CDS'].map(cds_to_type).fillna('unknown')
    logger.info("Added 'inversion_type' column to pairwise data.")

    logger.info("--- Step 2: Filtering Pairwise Data ---")
    required_cols = ['inversion_type', 'Group1', 'Group2', 'omega', 'Seq1', 'Seq2']
    missing_base_cols = [col for col in required_cols if col not in pairwise_df.columns]
    if missing_base_cols: logger.error(f"Pairwise data missing required columns: {missing_base_cols}. Cannot proceed."); return

    logger.debug("Converting key columns to numeric...")
    try:
        for col in ['Group1', 'Group2', 'omega']: pairwise_df[col] = pd.to_numeric(pairwise_df[col], errors='coerce')
    except Exception as e: logger.error(f"Error converting columns to numeric: {e}", exc_info=True); return

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
    if filtered_df.empty: logger.error("No pairs found matching filters. Cannot generate plots."); return

    logger.info("--- Step 3: Processing Data for Recurrent Inversions ---")
    recurrent_pairs_df = filtered_df[filtered_df['inversion_type'] == 'recurrent'].copy()
    logger.info(f"Processing {len(recurrent_pairs_df):,} pairs for Recurrent plot.")
    rec_agg_data, rec_sorted_nodes, rec_node_to_color_map = aggregate_pairwise_data_and_calc_node_colors(
        recurrent_pairs_df, "recurrent", COLOR_TRANSITION_EXPONENT, PROPORTION_CMAP_MPL
    )
    if rec_agg_data.empty: logger.warning("Aggregation for recurrent pairs resulted in empty data.")

    logger.info("--- Step 4: Processing Data for Single-Event Inversions ---")
    single_event_pairs_df = filtered_df[filtered_df['inversion_type'] == 'single_event'].copy()
    logger.info(f"Processing {len(single_event_pairs_df):,} pairs for Single-Event plot.")
    single_agg_data, single_sorted_nodes, single_node_to_color_map = aggregate_pairwise_data_and_calc_node_colors(
        single_event_pairs_df, "single_event", COLOR_TRANSITION_EXPONENT, PROPORTION_CMAP_MPL
    )
    if single_agg_data.empty: logger.warning("Aggregation for single-event pairs resulted in empty data.")


    logger.info("--- Step 5: Generating Static Chord Diagrams ---")

    plot_chord_diagram_node_avg_color(
        rec_agg_data,
        rec_sorted_nodes,
        rec_node_to_color_map,
        RECURRENT_CHORD_PLOT_FILE
    )

    plot_chord_diagram_node_avg_color(
        single_agg_data,
        single_sorted_nodes,
        single_node_to_color_map,
        SINGLE_EVENT_CHORD_PLOT_FILE
    )

    logger.info("--- Chord plot generation script finished ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as main_e:
        logger.critical(f"An unhandled exception occurred in the main script execution: {main_e}", exc_info=True)
