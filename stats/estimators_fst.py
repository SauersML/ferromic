import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import logging
import sys
import time
import seaborn as sns

# --- Configuration ---

# Input Files
SUMMARY_STATS_FILE = 'output.csv'
INVERSION_FILE = 'inv_info.tsv'

# Output Files
FST_SCATTER_PLOT_FILENAME = 'fst_wc_vs_hudson_colored_by_inversion_type.png'
VARIANCE_VS_DXY_PLOT_FILENAME = 'variance_wc_vs_dxy_hudson_log_scale_colored.png' # Updated name

# Column Names
FST_WC_COL = 'haplotype_overall_fst_wc'
FST_HUDSON_COL = 'hudson_fst_hap_group_0v1'
N_HAP_0_COL = '0_num_hap_filter'
N_HAP_1_COL = '1_num_hap_filter'

HAP_BETWEEN_POP_VARIANCE_WC_COL = 'haplotype_between_pop_variance_wc'
HUDSON_DXY_COL = 'hudson_dxy_hap_group_0v1'

SUMMARY_STATS_COORDINATE_COLUMNS = {'chr': 'chr', 'start': 'region_start', 'end': 'region_end'}
INVERSION_FILE_COLUMNS = ['Chromosome', 'Start', 'End', '0_single_1_recur']
INV_CATEGORY_COL_ORIGINAL = '0_single_1_recur'
INVERSION_TYPE_COL = 'inversion_type'

INVERSION_CATEGORY_MAPPING = {
    'Recurrent': 'recurrent',
    'Single-event': 'single_event'
}
COLOR_PALETTE = sns.color_palette("Set2", n_colors=max(2, len(INVERSION_CATEGORY_MAPPING)))
SCATTER_COLOR_MAP = {
    INVERSION_CATEGORY_MAPPING['Recurrent']: COLOR_PALETTE[0],
    INVERSION_CATEGORY_MAPPING['Single-event']: COLOR_PALETTE[1],
    'no_match': 'grey',
    'ambiguous_match': 'purple',
    'coordinate_error': 'black'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def normalize_chromosome_name(chromosome_id):
    chromosome_id_str = str(chromosome_id).strip().lower()
    if chromosome_id_str.startswith('chr_'):
        chromosome_id_str = chromosome_id_str[4:]
    elif chromosome_id_str.startswith('chr'):
        chromosome_id_str = chromosome_id_str[3:]
    if not chromosome_id_str.startswith('chr'):
         chromosome_id_str = f"chr{chromosome_id_str}"
    return chromosome_id_str

def map_coordinates_to_inversion_types(inversion_info_df):
    recurrent_regions = {}
    single_event_regions = {}
    if not all(col in inversion_info_df.columns for col in INVERSION_FILE_COLUMNS):
        missing = [c for c in INVERSION_FILE_COLUMNS if c not in inversion_info_df.columns]
        raise ValueError(f"Inversion data missing required columns: {missing}")
    logger.info(f"Mapping inversion types from {len(inversion_info_df)} entries...")
    parsed_count = 0; skipped_count = 0
    for index, row in inversion_info_df.iterrows():
        try:
            chrom = normalize_chromosome_name(row['Chromosome'])
            if any(pd.isna(row[col]) for col in ['Start', 'End', INV_CATEGORY_COL_ORIGINAL]):
                nan_cols = [col for col in ['Start', 'End', INV_CATEGORY_COL_ORIGINAL] if pd.isna(row[col])]
                raise ValueError(f"Essential data NaN in: {', '.join(nan_cols)}")
            start, end = int(row['Start']), int(row['End'])
            category_code = int(row[INV_CATEGORY_COL_ORIGINAL])
            if start > end: raise ValueError("Start > End")
            parsed_count += 1
            if category_code == 1: recurrent_regions.setdefault(chrom, []).append((start, end))
            elif category_code == 0: single_event_regions.setdefault(chrom, []).append((start, end))
            else: raise ValueError(f"Unrecognized category code '{category_code}'")
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping row {index + 2} in inversion data: {e}. Row: {row.to_dict()}")
            skipped_count +=1
    logger.info(f"Mapped {parsed_count} inversion entries. Skipped {skipped_count}.")
    if parsed_count == 0 and len(inversion_info_df) > 0: logger.warning("No valid inversion entries mapped.")
    return recurrent_regions, single_event_regions

def check_coordinate_overlap(summary_coords, inv_coords, tolerance=1):
    _, s_start, s_end = summary_coords
    i_start, i_end = inv_coords
    return abs(s_start - i_start) <= tolerance and abs(s_end - i_end) <= tolerance

def determine_region_inversion_type(chrom, start, end, recurrent_map, single_map):
    curr_coords = (chrom, start, end)
    is_rec = any(check_coordinate_overlap(curr_coords, r_coords) for r_coords in recurrent_map.get(chrom, []))
    is_sing = any(check_coordinate_overlap(curr_coords, s_coords) for s_coords in single_map.get(chrom, []))
    if is_rec and is_sing: return 'ambiguous_match'
    if is_rec: return INVERSION_CATEGORY_MAPPING['Recurrent']
    if is_sing: return INVERSION_CATEGORY_MAPPING['Single-event']
    return 'no_match'

def assign_inversion_type_to_summary_row(row, rec_map, sing_map, coord_conf):
    try:
        chrom = normalize_chromosome_name(row[coord_conf['chr']])
        if any(pd.isna(row[coord_conf[c]]) for c in ['start', 'end']): return 'coordinate_error'
        start, end = int(row[coord_conf['start']]), int(row[coord_conf['end']])
        if start > end: return 'coordinate_error'
        return determine_region_inversion_type(chrom, start, end, rec_map, sing_map)
    except (ValueError, TypeError, KeyError) : return 'coordinate_error'

def create_scatter_plot(df, x_col, y_col, type_col, output_filename,
                        x_is_non_negative=False, y_is_non_negative=False,
                        log_scale_x=False, log_scale_y=False):
    logger.info(f"Generating scatter plot: {x_col} vs {y_col} "
                f"(Log X: {log_scale_x}, Log Y: {log_scale_y})")
    plt.figure(figsize=(10, 8))
    
    df_plot = df.copy()
    df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors='coerce')
    df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
    df_plot.dropna(subset=[x_col, y_col], inplace=True)

    if log_scale_x:
        original_count = len(df_plot)
        df_plot = df_plot[df_plot[x_col] > 0]
        removed_count = original_count - len(df_plot)
        if removed_count > 0:
            logger.warning(f"For log scale on X-axis ({x_col}), removed {removed_count} non-positive value(s).")
    if log_scale_y:
        original_count = len(df_plot)
        df_plot = df_plot[df_plot[y_col] > 0]
        removed_count = original_count - len(df_plot)
        if removed_count > 0:
            logger.warning(f"For log scale on Y-axis ({y_col}), removed {removed_count} non-positive value(s).")

    if df_plot.empty:
        log_msg_suffix = " with requested log scale(s)" if (log_scale_x or log_scale_y) else ""
        logger.warning(f"No data available for scatter plot ({x_col} vs {y_col}) after removing NaNs{log_msg_suffix}.")
        plt.text(0.5, 0.5, f"No data to plot for\n{x_col} vs {y_col}\n(NaN/non-positive removed)", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12, color='red',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
        plt.title(f"{x_col.replace('_', ' ').title()} vs. {y_col.replace('_', ' ').title()}", fontsize=16)
        plt.xlabel(x_col.replace('_', ' ').title() + (" (log scale)" if log_scale_x else ""), fontsize=14)
        plt.ylabel(y_col.replace('_', ' ').title() + (" (log scale)" if log_scale_y else ""), fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        try: plt.savefig(output_filename, dpi=300, bbox_inches='tight'); logger.info(f"Saved empty plot placeholder to '{output_filename}'")
        except Exception as e: logger.error(f"Failed to save empty scatter plot: {e}")
        plt.close()
        return

    internal_to_display_legend_map = {
        **{v: k for k, v in INVERSION_CATEGORY_MAPPING.items()}, 
        'no_match': 'No Match', 'ambiguous_match': 'Ambiguous Match', 'coordinate_error': 'Coordinate Error'
    }
    for key in SCATTER_COLOR_MAP.keys():
        if key not in internal_to_display_legend_map:
            internal_to_display_legend_map[key] = key.replace('_', ' ').title()

    present_types_in_data = df_plot[type_col].unique()
    preferred_legend_order_internal_keys = [
        INVERSION_CATEGORY_MAPPING['Recurrent'], INVERSION_CATEGORY_MAPPING['Single-event'],
        'no_match', 'ambiguous_match', 'coordinate_error'
    ]
    actual_preferred_keys = [k for k in preferred_legend_order_internal_keys if k in present_types_in_data and k in SCATTER_COLOR_MAP]
    additional_types_internal = [p for p in present_types_in_data if p in SCATTER_COLOR_MAP and p not in actual_preferred_keys]
    final_internal_keys_for_plotting = actual_preferred_keys + additional_types_internal

    if not final_internal_keys_for_plotting: logger.warning(f"No categories with defined colors in plotted data for {x_col} vs {y_col}.")

    for internal_key in final_internal_keys_for_plotting:
        subset = df_plot[df_plot[type_col] == internal_key]
        if not subset.empty:
            display_name = internal_to_display_legend_map.get(internal_key, internal_key.replace('_',' ').title())
            color_for_plot = SCATTER_COLOR_MAP[internal_key] 
            plt.scatter(subset[x_col], subset[y_col], label=f"{display_name} (N={len(subset)})", 
                        color=color_for_plot, alpha=0.7, s=50, edgecolor='k', linewidth=0.5)

    plt.title(f"{x_col.replace('_', ' ').title()} vs. {y_col.replace('_', ' ').title()}", fontsize=18)
    plt.xlabel(x_col.replace('_', ' ').title() + (" (log scale)" if log_scale_x else ""), fontsize=16)
    plt.ylabel(y_col.replace('_', ' ').title() + (" (log scale)" if log_scale_y else ""), fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    if final_internal_keys_for_plotting:
        plt.legend(title="Inversion Type", loc='best', frameon=True, fancybox=True, shadow=True, borderpad=1,
                   fontsize=12, title_fontsize=13)

    all_x_vals = df_plot[x_col].dropna() # Already filtered for positive if log_scale_x
    all_y_vals = df_plot[y_col].dropna() # Already filtered for positive if log_scale_y

    if log_scale_x and not all_x_vals.empty: plt.xscale('log')
    if log_scale_y and not all_y_vals.empty: plt.yscale('log')

    if not all_x_vals.empty:
        min_x, max_x = all_x_vals.min(), all_x_vals.max()
        if log_scale_x:
            xlim_lower = min_x * 0.5 if min_x > 0 else 1e-9 # Default small value if min_x is not usable
            xlim_upper = max_x * 2.0 if max_x > 0 else 1.0
            if min_x == max_x: # Single point
                xlim_lower = min_x * 0.1 if min_x > 0 else 1e-9
                xlim_upper = max_x * 10.0 if max_x > 0 else 1.0
        else:
            buffer_x = 0.05 * (max_x - min_x) if (max_x > min_x) else (abs(min_x * 0.1) if min_x != 0 else 0.1)
            if buffer_x == 0 and max_x == min_x : buffer_x = 0.1 
            xlim_lower = min_x - buffer_x
            if x_is_non_negative: xlim_lower = max(0, xlim_lower)
            xlim_upper = max_x + buffer_x
        if not (pd.isna(xlim_lower) or pd.isna(xlim_upper) or xlim_lower >= xlim_upper): plt.xlim(xlim_lower, xlim_upper)
        elif not log_scale_x and x_is_non_negative : plt.axvline(0, color='grey', lw=0.5, linestyle='--')

    if not all_y_vals.empty:
        min_y, max_y = all_y_vals.min(), all_y_vals.max()
        if log_scale_y:
            ylim_lower = min_y * 0.5 if min_y > 0 else 1e-9
            ylim_upper = max_y * 2.0 if max_y > 0 else 1.0
            if min_y == max_y: # Single point
                ylim_lower = min_y * 0.1 if min_y > 0 else 1e-9
                ylim_upper = max_y * 10.0 if max_y > 0 else 1.0
        else:
            buffer_y = 0.05 * (max_y - min_y) if (max_y > min_y) else (abs(min_y * 0.1) if min_y != 0 else 0.1)
            if buffer_y == 0 and max_y == min_y : buffer_y = 0.1
            ylim_lower = min_y - buffer_y
            if y_is_non_negative: ylim_lower = max(0, ylim_lower)
            ylim_upper = max_y + buffer_y
        if not (pd.isna(ylim_lower) or pd.isna(ylim_upper) or ylim_lower >= ylim_upper): plt.ylim(ylim_lower, ylim_upper)
        elif not log_scale_y and y_is_non_negative : plt.axhline(0, color='grey', lw=0.5, linestyle='--')
    
    if all_x_vals.empty: plt.xlim( (1e-3 if log_scale_x else (0 if x_is_non_negative else -0.1)), 1)
    if all_y_vals.empty: plt.ylim( (1e-3 if log_scale_y else (0 if y_is_non_negative else -0.1)), 1)

    plt.tight_layout()
    try: plt.savefig(output_filename, dpi=300, bbox_inches='tight'); logger.info(f"Scatter plot saved to '{output_filename}'")
    except Exception as e: logger.error(f"Failed to save scatter plot '{output_filename}': {e}", exc_info=True)
    plt.close()

def calculate_and_print_proportions(df, type_col, n_hap_0_col, n_hap_1_col, fst_wc_col, fst_hudson_col):
    logger.info("\n--- Calculating Proportions ---")
    df_calc = df.copy()
    df_calc[n_hap_0_col] = pd.to_numeric(df_calc[n_hap_0_col], errors='coerce')
    df_calc[n_hap_1_col] = pd.to_numeric(df_calc[n_hap_1_col], errors='coerce')
    valid_hap_counts_mask = df_calc[n_hap_0_col].notna() & df_calc[n_hap_1_col].notna()
    df_calc['min_hap_count'] = np.nan 
    df_calc.loc[valid_hap_counts_mask, 'min_hap_count'] = df_calc.loc[valid_hap_counts_mask, [n_hap_0_col, n_hap_1_col]].min(axis=1)

    for inv_map_key_display, inv_map_key_internal in INVERSION_CATEGORY_MAPPING.items():
        subset_df = df_calc[df_calc[type_col] == inv_map_key_internal]
        if not subset_df.empty:
            valid_hap_loci = subset_df[subset_df['min_hap_count'].notna()]
            if not valid_hap_loci.empty:
                num_min_1 = (valid_hap_loci['min_hap_count'] == 1).sum()
                proportion = num_min_1 / len(valid_hap_loci)
                logger.info(f"Proportion of {inv_map_key_display.lower()} loci (N={len(valid_hap_loci)}) with min(hap_counts) == 1: {proportion:.4f} ({num_min_1}/{len(valid_hap_loci)})")
            else: logger.info(f"No {inv_map_key_display.lower()} loci (key: {inv_map_key_internal}) with valid haplotype counts for proportion calculation.")
        else: logger.info(f"No {inv_map_key_display.lower()} loci (key: {inv_map_key_internal}) found.")

    df_calc[fst_wc_col] = pd.to_numeric(df_calc[fst_wc_col], errors='coerce')
    df_calc[fst_hudson_col] = pd.to_numeric(df_calc[fst_hudson_col], errors='coerce')

    wc_only_fst_df = df_calc[df_calc[fst_wc_col].notna() & df_calc[fst_hudson_col].isna()]
    if not wc_only_fst_df.empty:
        wc_only_valid_hap = wc_only_fst_df[wc_only_fst_df['min_hap_count'].notna()]
        if not wc_only_valid_hap.empty:
            num_min_1 = (wc_only_valid_hap['min_hap_count'] == 1).sum()
            prop = num_min_1 / len(wc_only_valid_hap)
            logger.info(f"Proportion of loci W&C FST, no Hudson FST (N_valid_hap={len(wc_only_valid_hap)}) & min(hap_counts)==1: {prop:.4f} ({num_min_1}/{len(wc_only_valid_hap)})")
            logger.info(f" (Total loci W&C FST, no Hudson FST, regardless of hap counts: {len(wc_only_fst_df)})")
        else: logger.info(f"No loci W&C FST, no Hudson FST, AND valid hap counts (Total W&C_only_FST: {len(wc_only_fst_df)}).")
    else: logger.info("No loci where W&C FST exists and Hudson FST is missing.")

    both_fst_df = df_calc[df_calc[fst_wc_col].notna() & df_calc[fst_hudson_col].notna()]
    if not both_fst_df.empty:
        both_fst_valid_hap = both_fst_df[both_fst_df['min_hap_count'].notna()]
        if not both_fst_valid_hap.empty:
            num_min_1 = (both_fst_valid_hap['min_hap_count'] == 1).sum()
            prop = num_min_1 / len(both_fst_valid_hap)
            logger.info(f"Proportion of loci with both W&C & Hudson FST (N_valid_hap={len(both_fst_valid_hap)}) & min(hap_counts)==1: {prop:.4f} ({num_min_1}/{len(both_fst_valid_hap)})")
            logger.info(f" (Total loci with both FSTs, regardless of hap counts: {len(both_fst_df)})")
        else: logger.info(f"No loci with both FSTs AND valid hap counts (Total both_FST: {len(both_fst_df)}).")
    else: logger.info("No loci where both W&C FST and Hudson FST exist.")

def main():
    overall_start_time = time.time()
    logger.info(f"--- Starting Analysis ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")

    summary_cols_to_load = list(dict.fromkeys(
        list(SUMMARY_STATS_COORDINATE_COLUMNS.values()) +
        [FST_WC_COL, FST_HUDSON_COL, N_HAP_0_COL, N_HAP_1_COL,
         HAP_BETWEEN_POP_VARIANCE_WC_COL, HUDSON_DXY_COL]
    ))

    try:
        inv_df = pd.read_csv(INVERSION_FILE, sep='\t', usecols=INVERSION_FILE_COLUMNS)
        sum_df_raw = pd.read_csv(SUMMARY_STATS_FILE)
        missing_cols = [col for col in summary_cols_to_load if col not in sum_df_raw.columns]
        if missing_cols:
            logger.critical(f"CRITICAL: Missing required columns in '{SUMMARY_STATS_FILE}': {missing_cols}. Halting.")
            sys.exit(1)
        sum_df = sum_df_raw[summary_cols_to_load].copy()
    except FileNotFoundError as e: logger.critical(f"CRITICAL: Input file not found: {e}. Halting."); sys.exit(1)
    except Exception as e: logger.critical(f"CRITICAL: Failed to load data: {e}", exc_info=True); sys.exit(1)
        
    logger.info(f"Loaded {len(sum_df)} summary rows, {len(inv_df)} inversion rows.")

    try:
        recurrent_map, single_event_map = map_coordinates_to_inversion_types(inv_df)
    except Exception as e: logger.critical(f"CRITICAL: Error processing inversion file: {e}", exc_info=True); sys.exit(1)

    if not recurrent_map and not single_event_map and len(inv_df) > 0 : logger.warning("Inversion maps empty.")

    logger.info(f"Assigning inversion types...")
    sum_df.loc[:, INVERSION_TYPE_COL] = sum_df.apply(
        lambda row: assign_inversion_type_to_summary_row(row, recurrent_map, single_event_map, SUMMARY_STATS_COORDINATE_COLUMNS), axis=1)
    
    type_counts = sum_df[INVERSION_TYPE_COL].value_counts(dropna=False)
    logger.info(f"Counts by assigned inversion type:\n{type_counts.to_string()}")
    if not any(k in type_counts for k in INVERSION_CATEGORY_MAPPING.values()) and len(sum_df) > 0:
        logger.warning("No regions classified as 'Recurrent' or 'Single-event'. Check data/logic.")

    logger.info(f"\n--- Generating FST scatter plot ---")
    create_scatter_plot(sum_df, FST_WC_COL, FST_HUDSON_COL, INVERSION_TYPE_COL, 
                        FST_SCATTER_PLOT_FILENAME,
                        x_is_non_negative=False, y_is_non_negative=False,
                        log_scale_x=False, log_scale_y=False)

    logger.info(f"\n--- Generating Variance vs Dxy scatter plot (log scale) ---")
    create_scatter_plot(sum_df, HAP_BETWEEN_POP_VARIANCE_WC_COL, HUDSON_DXY_COL, 
                        INVERSION_TYPE_COL, VARIANCE_VS_DXY_PLOT_FILENAME,
                        x_is_non_negative=True, y_is_non_negative=True,
                        log_scale_x=True, log_scale_y=True)

    calculate_and_print_proportions(sum_df, INVERSION_TYPE_COL, N_HAP_0_COL, N_HAP_1_COL, FST_WC_COL, FST_HUDSON_COL)

    logger.info(f"--- Analysis Complete. Total time: {time.time() - overall_start_time:.2f}s ---")

if __name__ == "__main__":
    main()
