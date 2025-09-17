import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import logging
import sys
import time
import seaborn as sns
from scipy.stats import mannwhitneyu
import os

# --- Configuration ---

# Input Files
SUMMARY_STATS_FILE = 'output.csv'
INVERSION_FILE = 'inv_info.tsv'
COORDINATE_MAP_FILE = 'map.tsv' # New configuration for the map file

# Output File Templates
VIOLIN_PLOT_TEMPLATE = 'comparison_violin_{column_safe_name}.png'
BOX_PLOT_TEMPLATE = 'comparison_boxplot_{column_safe_name}.png'
SCATTER_PLOT_TEMPLATE = 'scatter_fst_{fst_col_safe}_vs_{attr_col_safe}.png'
FST_OUTPUT_TSV = 'inversion_fst_estimates.tsv' # Output for inversion FST list


# Columns for Analysis (all columns to process for data quality checks)
ANALYSIS_COLUMNS = [
    'haplotype_overall_fst_wc',
    'haplotype_between_pop_variance_wc',
    'haplotype_within_pop_variance_wc',
    'haplotype_num_informative_sites_wc',
    'hudson_fst_hap_group_0v1',
    'hudson_dxy_hap_group_0v1',
    'hudson_pi_hap_group_0',
    'hudson_pi_hap_group_1',
    'hudson_pi_avg_hap_group_0v1',
    'inversion_freq_filter',
    '0_num_hap_filter',
    '1_num_hap_filter'
]

# Columns for which to perform statistical tests (Recurrent vs Single-event)
FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT = [
    'haplotype_overall_fst_wc',
    'hudson_fst_hap_group_0v1'
]

# Additional columns for which to generate box plots
OTHER_COLUMNS_FOR_BOX_PLOT = [
    'haplotype_num_informative_sites_wc',
    'hudson_dxy_hap_group_0v1',
    'hudson_pi_avg_hap_group_0v1',
    'haplotype_within_pop_variance_wc'
]

COLUMNS_FOR_PLOTTING = FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT + OTHER_COLUMNS_FOR_BOX_PLOT

FST_WC_COL = 'haplotype_overall_fst_wc'
FST_HUDSON_COL = 'hudson_fst_hap_group_0v1'
INV_FREQ_COL = 'inversion_freq_filter'
N_HAP_0_COL = '0_num_hap_filter'
N_HAP_1_COL = '1_num_hap_filter'

SCATTER_PLOT_CONFIG = [
    {'fst_col': FST_WC_COL, 'attr_col': INV_FREQ_COL, 'attr_name': 'Inversion Allele Frequency'},
    {'fst_col': FST_HUDSON_COL, 'attr_col': INV_FREQ_COL, 'attr_name': 'Inversion Allele Frequency'},
    {'fst_col': FST_WC_COL, 'attr_col': N_HAP_1_COL, 'attr_name': 'N Inverted Haplotypes'},
    {'fst_col': FST_HUDSON_COL, 'attr_col': N_HAP_1_COL, 'attr_name': 'N Inverted Haplotypes'},
    {'fst_col': FST_WC_COL, 'attr_col': N_HAP_0_COL, 'attr_name': 'N Non-Inverted Haplotypes'},
    {'fst_col': FST_HUDSON_COL, 'attr_col': N_HAP_0_COL, 'attr_name': 'N Non-Inverted Haplotypes'}
]

SUMMARY_STATS_COORDINATE_COLUMNS = {'chr': 'chr', 'start': 'region_start', 'end': 'region_end'}
INVERSION_FILE_COLUMNS = ['Chromosome', 'Start', 'End', '0_single_1_recur_consensus']
MAP_FILE_COLUMNS = ['Original_Chr', 'Original_Start', 'Original_End', 'New_Chr', 'New_Start', 'New_End']


INVERSION_CATEGORY_MAPPING = {
    'Recurrent': 'recurrent',
    'Single-event': 'single_event'
}
COLOR_PALETTE = sns.color_palette("Set2", n_colors=len(INVERSION_CATEGORY_MAPPING))
SCATTER_COLOR_MAP = {
    'recurrent': COLOR_PALETTE[0],
    'single_event': COLOR_PALETTE[1]
}

DATA_QUALITY_DISCREPANCY_THRESHOLD = 0.20
MAX_INDIVIDUAL_WARNINGS_PER_CATEGORY = 10
MAX_EXAMPLES_OF_OUT_OF_SPEC_LOGGING = 5

FST_TEST_SUMMARIES = []
FST_TEST_SUMMARIES_FILTERED = [] # Global list for filtered FST test summaries
SCATTER_PLOT_SUMMARIES = [] # global list for scatterplot summaries

# Output File Template for filtered plots
VIOLIN_PLOT_FILTERED_TEMPLATE = 'comparison_violin_filtered_hap_min_{column_safe_name}.png'

# Suffix for logging and identifying filtered analysis runs
FILTER_SUFFIX = "_filtered_hap_min"

# Specific FST component columns for summary
FST_WC_COMPONENT_COLUMNS = [
    'haplotype_between_pop_variance_wc',
    'haplotype_within_pop_variance_wc'
]
FST_HUDSON_COMPONENT_COLUMNS = [
    'hudson_dxy_hap_group_0v1',
    'hudson_pi_hap_group_0',
    'hudson_pi_hap_group_1',
    'hudson_pi_avg_hap_group_0v1'
]
ALL_FST_COMPONENT_COLUMNS = FST_WC_COMPONENT_COLUMNS + FST_HUDSON_COMPONENT_COLUMNS


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('InversionComparisonAnalysis')

# --- Warning Throttling Class ---
class WarningTracker:
    def __init__(self):
        self.warnings = {}

    def log_warning(self, category_key, message_func, *args, **kwargs):
        if category_key not in self.warnings:
            self.warnings[category_key] = {"count": 0, "suppressed_count": 0, "limit_reached_msg_logged": False}
        cat_tracker = self.warnings[category_key]
        cat_tracker["count"] += 1
        if cat_tracker["count"] <= MAX_INDIVIDUAL_WARNINGS_PER_CATEGORY:
            logger.warning(message_func(*args, **kwargs))
        elif not cat_tracker["limit_reached_msg_logged"]:
            logger.warning(
                f"Warning limit ({MAX_INDIVIDUAL_WARNINGS_PER_CATEGORY}) reached for category '{category_key}'. "
                f"Further warnings of this type will be suppressed. "
                f"Example: {message_func(*args, **kwargs)}"
            )
            cat_tracker["limit_reached_msg_logged"] = True
            cat_tracker["suppressed_count"] += 1
        else:
            cat_tracker["suppressed_count"] += 1
            
    def get_suppressed_summary(self):
        summary = [f"Category '{cat_key}': {data['suppressed_count']} warnings were suppressed."
                   for cat_key, data in self.warnings.items() if data["suppressed_count"] > 0]
        return "\n".join(summary) if summary else ""

global_warning_tracker = WarningTracker()

# --- Helper Functions ---
def normalize_chromosome_name(chromosome_id):
    chromosome_id_str = str(chromosome_id).strip().lower()
    if chromosome_id_str.startswith('chr_'): chromosome_id_str = chromosome_id_str[4:]
    elif chromosome_id_str.startswith('chr'): chromosome_id_str = chromosome_id_str[3:]
    if not chromosome_id_str.startswith('chr') and chromosome_id_str not in ['x', 'y', 'm', 'mt']: # Avoid chrchrX
        chromosome_id_str = f"chr{chromosome_id_str}"
    return chromosome_id_str

def get_column_value_specifications(column_name):
    col_lower = column_name.lower()
    if 'fst' in col_lower: return (-0.1, 1.1, 'numeric', True)
    if 'pi' in col_lower: return (0.0, 1.0, 'numeric', False)
    if 'dxy' in col_lower: return (0.0, float('inf'), 'numeric', False)
    if '_variance_' in col_lower: return (0.0, float('inf'), 'numeric', False)
    if '_num_informative_sites_' in col_lower: return (0, float('inf'), 'integer', False)
    if 'inversion_freq_filter' in col_lower: return (0.0, 1.0, 'numeric', False)
    if 'num_hap_filter' in col_lower: return (0, float('inf'), 'integer', False)
    logger.debug(f"No specific value range for '{column_name}'. Generic numeric range used for flagging.")
    return (-float('inf'), float('inf'), 'numeric', True)

# --- Data Processing Functions ---
def map_coordinates_to_inversion_types(inversion_info_df, map_df, perform_mapping_flag):
    recurrent_regions = {}
    single_event_regions = {}
    warn_key = "inversion_file_row_processing" # Renamed for clarity
    
    required_cols_inv = ['Chromosome', 'Start', 'End', '0_single_1_recur_consensus']
    if not all(col in inversion_info_df.columns for col in required_cols_inv):
        missing_cols = [c for c in required_cols_inv if c not in inversion_info_df.columns]
        logger.critical(f"Inversion data '{INVERSION_FILE}' missing required columns: {missing_cols}. Exiting.")
        sys.exit(1)

    logger.info(f"Processing {len(inversion_info_df)} entries from inversion info file...")
    map_lookup = {}
    if perform_mapping_flag and map_df is not None:
        logger.info(f"Coordinate mapping is ENABLED using '{COORDINATE_MAP_FILE}'.")
        for _, map_row in map_df.iterrows():
            # Ensure all necessary columns are in map_row and handle potential errors if not
            if not all(col in map_row for col in MAP_FILE_COLUMNS):
                logger.error(f"Skipping a row in {COORDINATE_MAP_FILE} due to missing columns: {map_row.to_dict()}")
                continue
            try:
                orig_c = normalize_chromosome_name(map_row['Original_Chr'])
                orig_s = int(map_row['Original_Start'])
                orig_e = int(map_row['Original_End'])
                new_c = normalize_chromosome_name(map_row['New_Chr'])
                new_s = int(map_row['New_Start'])
                new_e = int(map_row['New_End'])
                map_lookup[(orig_c, orig_s, orig_e)] = (new_c, new_s, new_e)
            except ValueError: # Handle cases where Start/End might not be integers
                 logger.warning(f"Could not parse coordinates in {COORDINATE_MAP_FILE} row: {map_row.to_dict()}. Skipping this map entry.")
                 continue
    else:
        logger.info(f"Coordinate mapping is DISABLED. Using raw coordinates from '{INVERSION_FILE}'.")

    parsed_count = 0
    skipped_count = 0

    for index, row in inversion_info_df.iterrows():
        # Check for NaN in essential columns before any processing
        essential_cols_check = ['Chromosome', 'Start', 'End', '0_single_1_recur_consensus']
        if any(pd.isna(row[col]) for col in essential_cols_check):
            nan_cols_found = [col for col in essential_cols_check if pd.isna(row[col])]
            global_warning_tracker.log_warning(warn_key, lambda r, i, nc: f"Skipping row {i+2} in inversion data: NaN in {', '.join(nc)}. Row: {r.to_dict()}", row, index, nan_cols_found)
            skipped_count += 1
            continue

        # Attempt to convert Start, End, 0_single_1_recur_consensus to numeric types safely
        try:
            original_chrom_val = str(row['Chromosome'])
            original_start_val = int(row['Start'])
            original_end_val = int(row['End'])
            cat_code_val = int(row['0_single_1_recur_consensus'])
        except ValueError:
            global_warning_tracker.log_warning(warn_key, lambda r, i: f"Skipping row {i+2} in inversion data: Non-integer values in coordinate/category columns. Row: {r.to_dict()}", row, index)
            skipped_count += 1
            continue
            
        chrom_norm = normalize_chromosome_name(original_chrom_val)
        
        current_chrom, current_start, current_end = chrom_norm, original_start_val, original_end_val
        
        mapped_coords_key = (chrom_norm, original_start_val, original_end_val)
        if perform_mapping_flag and map_lookup and mapped_coords_key in map_lookup:
            current_chrom, current_start, current_end = map_lookup[mapped_coords_key]

        if current_start > current_end:
            global_warning_tracker.log_warning(warn_key, lambda r, i, c, s, e: f"Skipping row {i+2} in inversion data: Start ({s}) > End ({e}) for coordinates ({c}). Row: {r.to_dict()}", row, index, current_chrom, current_start, current_end)
            skipped_count += 1
            continue
        
        if cat_code_val == 1:
            recurrent_regions.setdefault(current_chrom, []).append((current_chrom, current_start, current_end))
        elif cat_code_val == 0:
            single_event_regions.setdefault(current_chrom, []).append((current_chrom, current_start, current_end))
        else:
            global_warning_tracker.log_warning(warn_key, lambda r, i, cc: f"Skipping row {i+2} in inversion data: Unrecognized category code '{cc}'. Row: {r.to_dict()}", row, index, cat_code_val)
            skipped_count +=1
            continue # Don't count as parsed if category code is invalid
        
        parsed_count += 1
            
    logger.info(f"Successfully processed {parsed_count} inversion entries. Skipped {skipped_count} entries.")
    if parsed_count == 0 and not inversion_info_df.empty:
        logger.warning("No valid inversion entries were processed after coordinate mapping and validation.")
    
    suppressed = global_warning_tracker.warnings.get(warn_key, {}).get("suppressed_count",0)
    if suppressed > 0:
        logger.info(f"{suppressed} warnings for '{warn_key}' suppressed.")
        
    return recurrent_regions, single_event_regions


def check_coordinate_overlap(summary_coords_tuple, inv_coords_tuple_in_map):
    # summary_coords_tuple: (chrom_str, start_int, end_int) from output.csv
    # inv_coords_tuple_in_map: (chrom_str, start_int, end_int) from the 'New_*' columns of map.tsv
    # This function is now more general for checking if two coordinate sets overlap closely.
    s_chrom, s_start, s_end = summary_coords_tuple
    i_chrom, i_start, i_end = inv_coords_tuple_in_map
    
    if s_chrom != i_chrom: # Must be on the same chromosome
        return False
    return abs(s_start - i_start) <= 1 and abs(s_end - i_end) <= 1


def determine_region_inversion_type(chrom, start, end, recurrent_map, single_map):
    rec_matches = recurrent_map.get(chrom, [])
    sing_matches = single_map.get(chrom, [])
    curr_coords_tuple = (chrom, start, end) # Use tuple for check_coordinate_overlap
    
    is_rec = any(check_coordinate_overlap(curr_coords_tuple, r_coords) for r_coords in rec_matches)
    is_sing = any(check_coordinate_overlap(curr_coords_tuple, s_coords) for s_coords in sing_matches)
    
    if is_rec and not is_sing: return INVERSION_CATEGORY_MAPPING['Recurrent']
    if is_sing and not is_rec: return INVERSION_CATEGORY_MAPPING['Single-event']
    if is_rec and is_sing:
        logger.debug(f"Ambiguous match for {chrom}:{start}-{end}. Matches both recurrent and single-event.")
        return 'ambiguous_match'
    return 'no_match'

def assign_inversion_type_to_summary_row(row, rec_map, sing_map, coord_conf):
    p_key="summary_coord_parsing_error"; l_key="summary_coord_logic_error"

    # Removed outer try-except block
    # KeyErrors will propagate if coord_conf keys are wrong or row is missing keys.
    
    chrom_val = row.get(coord_conf['chr'])
    start_val = row.get(coord_conf['start'])
    end_val = row.get(coord_conf['end'])

    if chrom_val is None or pd.isna(chrom_val) or \
       start_val is None or pd.isna(start_val) or \
       end_val is None or pd.isna(end_val):
        nan_cols = [k for k,v_name in coord_conf.items() if pd.isna(row.get(v_name))]
        global_warning_tracker.log_warning(p_key, lambda r, nc: f"Essential coordinate data NaN in summary row: {', '.join(nc)}. Row: {r.to_dict()}", row, nan_cols)
        return 'coordinate_error'

    try:
        chrom = normalize_chromosome_name(chrom_val)
        start = int(start_val)
        end = int(end_val)
    except (ValueError, TypeError) as e: # Catch specific conversion errors
        global_warning_tracker.log_warning(p_key, lambda r, err_msg: f"Coord parsing error for summary row: {err_msg}. Row: {r.to_dict()}", row, str(e))
        return 'coordinate_error'

    if start > end:
        global_warning_tracker.log_warning(l_key, lambda r,c,s,e:f"Invalid coords in summary (start>end): {r.get(c['chr'],'N/A')}:{s}-{e}", row,coord_conf,start,end)
        return 'coordinate_error'
        
    return determine_region_inversion_type(chrom, start, end, rec_map, sing_map)


def prepare_data_for_analysis(summary_df_with_types, column_name):
    logger.info(f"--- Data Quality Check for column: '{column_name}' ---")
    if column_name not in summary_df_with_types.columns:
        logger.error(f"Column '{column_name}' not found in summary_df_with_types. Skipping DQ check.")
        return

    cat_stats = {}
    min_exp, max_exp, val_type, _ = get_column_value_specifications(column_name)

    for disp_name, int_key in INVERSION_CATEGORY_MAPPING.items():
        subset_df = summary_df_with_types[summary_df_with_types['inversion_type'] == int_key]
        raw_series = subset_df[column_name] if column_name in subset_df else pd.Series(dtype=float)
        initial_n = len(raw_series)
        stats = cat_stats[int_key] = {'initial':initial_n, 'missing_non_numeric':0, 'numeric_for_analysis':0, 'flagged_oos':0}
        
        if initial_n == 0:
            logger.debug(f"No regions for '{disp_name}', column '{column_name}' in DQ check.")
            continue
        
        num_series_attempts = pd.to_numeric(raw_series, errors='coerce')
        stats['missing_non_numeric'] = num_series_attempts.isna().sum()
        valid_numerics = num_series_attempts.dropna()
        stats['numeric_for_analysis'] = len(valid_numerics)
        
        if not valid_numerics.empty:
            oos_mask = pd.Series(False, index=valid_numerics.index)
            if val_type == 'integer':
                oos_mask |= (valid_numerics != np.floor(valid_numerics)) & (np.abs(valid_numerics - np.round(valid_numerics)) > 1e-9)
            oos_mask |= (valid_numerics < min_exp) | (valid_numerics > max_exp)
            stats['flagged_oos'] = oos_mask.sum()
            if stats['flagged_oos'] > 0:
                logger.info(f"  For '{disp_name}', '{column_name}': {stats['flagged_oos']} numeric values flagged as out-of-spec. Examples:")
                shown = 0
                for idx, problem_flag in oos_mask.items(): 
                    if problem_flag and shown < MAX_EXAMPLES_OF_OUT_OF_SPEC_LOGGING:
                        val = valid_numerics[idx] 
                        reasons = []
                        if val_type=='integer' and ((val!=np.floor(val)) and (np.abs(val-np.round(val))>1e-9)): reasons.append(f"expected int, got {val:.4g}")
                        if val < min_exp: reasons.append(f"below min {min_exp} (is {val:.4g})")
                        if val > max_exp: reasons.append(f"above max {max_exp} (is {val:.4g})")
                        logger.info(f"    - Value {val:.4g} (Index {idx}): {'; '.join(reasons) or 'flagged issue'}")
                        shown += 1
                    elif problem_flag and shown == MAX_EXAMPLES_OF_OUT_OF_SPEC_LOGGING:
                        logger.info("    - (Further out-of-spec examples suppressed)"); shown+=1; break
        
        prop_miss = stats['missing_non_numeric']/initial_n if initial_n > 0 else 0
        prop_flag = stats['flagged_oos']/stats['numeric_for_analysis'] if stats['numeric_for_analysis'] > 0 else 0
        logger.info(f"  DQ Check Col '{column_name}', Cat '{disp_name}': Initial={initial_n}, Missing={stats['missing_non_numeric']} ({prop_miss:.2%}). "
                    f"Numerics={stats['numeric_for_analysis']} (of which {stats['flagged_oos']} or {prop_flag:.2%} flagged OOS).")

    keys = list(INVERSION_CATEGORY_MAPPING.values())
    if len(keys) == 2:
        s1_stats, s2_stats = cat_stats.get(keys[0]), cat_stats.get(keys[1])
        if s1_stats and s2_stats: 
            if s1_stats['initial'] > 0 and s2_stats['initial'] > 0:
                pm1 = s1_stats['missing_non_numeric'] / s1_stats['initial']
                pm2 = s2_stats['missing_non_numeric'] / s2_stats['initial']
                if abs(pm1 - pm2) > DATA_QUALITY_DISCREPANCY_THRESHOLD:
                    logger.warning(f"DISCREPANCY MissingOrNonNumeric for '{column_name}': {keys[0]} {pm1:.2%}, {keys[1]} {pm2:.2%}.")
            
            if s1_stats['numeric_for_analysis'] > 0 and s2_stats['numeric_for_analysis'] > 0 :
                pf1 = s1_stats['flagged_oos'] / s1_stats['numeric_for_analysis']
                pf2 = s2_stats['flagged_oos'] / s2_stats['numeric_for_analysis']
                if abs(pf1 - pf2) > DATA_QUALITY_DISCREPANCY_THRESHOLD:
                    logger.warning(f"DISCREPANCY FlaggedAsOutOfSpec for '{column_name}': {keys[0]} {pf1:.2%}, {keys[1]} {pf2:.2%}.")

def write_inversion_fst_to_tsv(summary_df_with_types, output_filename):
    """
    Filters for classified inversions and writes their coordinates, type, and FST estimates to a TSV file.

    Args:
        summary_df_with_types (pd.DataFrame): The main dataframe after inversion types have been assigned.
        output_filename (str): The path to the output TSV file.
    """
    logger.info(f"\n====== Writing Inversion FST Estimates to TSV ======")

    # Define the inversion types to be included in the output
    valid_inversion_types = list(INVERSION_CATEGORY_MAPPING.values())

    # Filter the DataFrame to include only rows with valid inversion types
    inversion_df = summary_df_with_types[summary_df_with_types['inversion_type'].isin(valid_inversion_types)].copy()

    if inversion_df.empty:
        logger.warning(f"No classified inversions found. The output TSV file '{output_filename}' will not be created.")
        return

    # Define the columns for the output file from existing constants
    coord_cols = list(SUMMARY_STATS_COORDINATE_COLUMNS.values())
    fst_cols = FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT
    output_columns = coord_cols + ['inversion_type'] + fst_cols

    # Check if all desired columns exist in the dataframe
    missing_cols = [col for col in output_columns if col not in inversion_df.columns]
    if missing_cols:
        logger.error(f"Cannot create FST output TSV. The following required columns are missing from the data: {missing_cols}. Skipping TSV generation.")
        return

    # Select the columns for the output file
    output_df = inversion_df[output_columns]

    try:
        output_df.to_csv(output_filename, sep='\t', index=False, float_format='%.6f', na_rep='NA')
        logger.info(f"Successfully wrote {len(output_df)} inversion records to '{output_filename}'")
    except IOError as e:
        logger.error(f"Failed to write FST estimates to TSV file '{output_filename}': {e}")

def _plot_common_elements(ax, plot_data_for_current_col, analysis_column_name, plot_type_specific_func):
    plot_labels = list(INVERSION_CATEGORY_MAPPING.keys())
    plot_data = [plot_data_for_current_col.get(INVERSION_CATEGORY_MAPPING[label], []) for label in plot_labels]
    
    meta = {}
    metric_name = analysis_column_name.replace('_', ' ').title()
    total_pts = 0
    all_vals_for_ylim = []

    for i, label in enumerate(plot_labels):
        vals = plot_data[i] 
        n = len(vals)
        total_pts += n
        meta[label] = {'median': np.median(vals) if n > 0 else np.nan, 'n_points': n}
        if n > 0: all_vals_for_ylim.extend(vals)
    
    logger.info(f"{plot_type_specific_func.__name__.split('_')[-1].capitalize()} for '{analysis_column_name}': " + 
                ", ".join([f"{lbl} N={meta[lbl]['n_points']}" for lbl in plot_labels]))

    positions = np.arange(len(plot_labels))
    err_style = {'ha':'center','va':'center','transform':ax.transAxes,'fontsize':12,'color':'red'}

    if total_pts > 0 and any(len(s) > 0 for s in plot_data):
        valid_series, valid_pos, valid_colors = [], [], []
        cmap_fill = {k: COLOR_PALETTE[i] for i,k in enumerate(INVERSION_CATEGORY_MAPPING.values())}

        for i, series in enumerate(plot_data):
            if series: 
                valid_series.append(series)
                valid_pos.append(positions[i])
                valid_colors.append(cmap_fill[INVERSION_CATEGORY_MAPPING[plot_labels[i]]])
        
        if not valid_series:
            logger.warning(f"No non-empty data series found for plotting {analysis_column_name}, though total_pts was {total_pts}.")
            ax.text(0.5, 0.5, f"No numeric data for categories\n(Column: {analysis_column_name})", **err_style)
        else:
            plot_type_specific_func(ax, valid_series, valid_pos, valid_colors)

            pt_clr, pt_alpha, pt_size = 'dimgray', 0.5, 15
            for i, orig_series in enumerate(plot_data):
                if orig_series:
                    jitters = np.random.normal(0, 0.04, size=len(orig_series))
                    ax.scatter(positions[i] + jitters, orig_series, color=pt_clr, alpha=pt_alpha, s=pt_size, edgecolor='none', zorder=5)
            
            if all_vals_for_ylim:
                min_y, max_y = np.min(all_vals_for_ylim), np.max(all_vals_for_ylim)
                y_range = max_y - min_y if max_y > min_y else 0.1
                if y_range == 0: y_range = np.abs(all_vals_for_ylim[0] * 0.1) if all_vals_for_ylim[0] != 0 else 0.1
                
                for i, label in enumerate(plot_labels):
                    m = meta[label]
                    if m['n_points'] > 0 and not np.isnan(m['median']):
                        med_y, txt_off = m['median'], y_range * 0.02 if y_range > 0 else 0.005
                        ax.text(positions[i] + 0.12, med_y + txt_off, f"{med_y:.3f}", fontsize=9, color='black', ha='left', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'), zorder=15)
    else:
        ax.text(0.5, 0.5, f"No numeric data for categories\n(Column: {analysis_column_name})", **err_style)

    ax.set_ylabel(metric_name, fontsize=16)
    ax.set_title(f'Comparison of {metric_name}', fontsize=18, pad=18)
    xt_lbls_n = [f"{lbl}\n(N={meta[lbl]['n_points']})" for lbl in plot_labels]
    ax.set_xticks(positions); ax.set_xticklabels(xt_lbls_n, fontsize=13)
    ax.set_xlabel(""); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13, length=0)
    if all_vals_for_ylim:
        min_v, max_v = np.min(all_vals_for_ylim), np.max(all_vals_for_ylim); dr = max_v - min_v
        pad = 0.1 if dr == 0 and min_v == 0 else (np.abs(min_v * 0.1) if dr == 0 else max(dr * 0.08, 0.005))
        ax.set_ylim(min_v - pad, max_v + pad * 1.5)
    else: ax.set_ylim(-0.02, 0.1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey'); ax.spines['left'].set_color('grey')

def _draw_violins(ax, valid_series, valid_pos, valid_colors):
    v_parts = ax.violinplot(valid_series, positions=valid_pos, showmedians=True, showextrema=False, widths=0.75)
    for i, b in enumerate(v_parts['bodies']):
        b.set_facecolor(valid_colors[i]); b.set_edgecolor('darkgrey'); b.set_linewidth(0.8); b.set_alpha(0.4)
    v_parts['cmedians'].set_edgecolor('black'); v_parts['cmedians'].set_linewidth(1.5); v_parts['cmedians'].set_zorder(10)

def _draw_boxplots(ax, valid_series, valid_pos, valid_colors):
    bp = ax.boxplot(valid_series, positions=valid_pos, widths=0.6, patch_artist=True,
                    showfliers=False,
                    medianprops={'color':'black', 'linewidth':1.5},
                    boxprops={'edgecolor':'darkgrey'},
                    whiskerprops={'color':'darkgrey'},
                    capprops={'color':'darkgrey'})
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(valid_colors[i]); patch.set_alpha(0.5)

def create_comparison_plot(plot_data_for_current_col, categorized_dfs_for_summary, analysis_column_name, plot_type,
                               output_filename_template_override=None,
                               plot_suffix_for_logging=""):
    fig, ax = plt.subplots(figsize=(7, 7))
    output_filename = ""
    plot_specific_draw_func = None
    returned_summary = None 

    safe_col_name = "".join(c if c.isalnum() else "_" for c in analysis_column_name).lower()
    if plot_type == 'violin':
        plot_specific_draw_func = _draw_violins
        template_to_use = output_filename_template_override if output_filename_template_override else VIOLIN_PLOT_TEMPLATE
        output_filename = template_to_use.format(column_safe_name=safe_col_name)
    elif plot_type == 'box':
        plot_specific_draw_func = _draw_boxplots
        template_to_use = output_filename_template_override if output_filename_template_override else BOX_PLOT_TEMPLATE
        output_filename = template_to_use.format(column_safe_name=safe_col_name)
    else:
        logger.error(f"Unknown plot type '{plot_type}' requested for {analysis_column_name}{plot_suffix_for_logging}. Skipping plot.")
        plt.close(fig)
        return None

    # Removed broad try-except Exception block
    _plot_common_elements(ax, plot_data_for_current_col, analysis_column_name, plot_specific_draw_func)

    if analysis_column_name in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT and plot_type == 'violin':
        p_val_text = "Test N/A (logic error)"
        rec_key = INVERSION_CATEGORY_MAPPING.get('Recurrent')
        sing_key = INVERSION_CATEGORY_MAPPING.get('Single-event')

        data_r, data_s = [], []
        recurrent_n, single_event_n = 0, 0
        recurrent_mean, recurrent_median = np.nan, np.nan
        single_event_mean, single_event_median = np.nan, np.nan
        recurrent_inv_freq_mean, single_event_inv_freq_mean = np.nan, np.nan
        recurrent_n0_hap_mean, single_event_n0_hap_mean = np.nan, np.nan
        recurrent_n1_hap_mean, single_event_n1_hap_mean = np.nan, np.nan

        if rec_key and sing_key:
            data_r = plot_data_for_current_col.get(rec_key, [])
            data_s = plot_data_for_current_col.get(sing_key, [])
            recurrent_n = len(data_r)
            single_event_n = len(data_s)

            data_r_df = categorized_dfs_for_summary.get(rec_key, pd.DataFrame())
            data_s_df = categorized_dfs_for_summary.get(sing_key, pd.DataFrame())

            if recurrent_n > 0:
                recurrent_mean = np.mean(data_r)
                recurrent_median = np.median(data_r)
                if INV_FREQ_COL in data_r_df.columns: recurrent_inv_freq_mean = pd.to_numeric(data_r_df[INV_FREQ_COL], errors='coerce').mean()
                if N_HAP_0_COL in data_r_df.columns: recurrent_n0_hap_mean = pd.to_numeric(data_r_df[N_HAP_0_COL], errors='coerce').mean()
                if N_HAP_1_COL in data_r_df.columns: recurrent_n1_hap_mean = pd.to_numeric(data_r_df[N_HAP_1_COL], errors='coerce').mean()

            if single_event_n > 0:
                single_event_mean = np.mean(data_s)
                single_event_median = np.median(data_s)
                if INV_FREQ_COL in data_s_df.columns: single_event_inv_freq_mean = pd.to_numeric(data_s_df[INV_FREQ_COL], errors='coerce').mean()
                if N_HAP_0_COL in data_s_df.columns: single_event_n0_hap_mean = pd.to_numeric(data_s_df[N_HAP_0_COL], errors='coerce').mean()
                if N_HAP_1_COL in data_s_df.columns: single_event_n1_hap_mean = pd.to_numeric(data_s_df[N_HAP_1_COL], errors='coerce').mean()

            if data_r and data_s: 
                if np.var(data_r) == 0 and np.var(data_s) == 0 and np.mean(data_r) == np.mean(data_s):
                    p_val_text = "p = 1.0 (Identical)"
                else:
                    # Specific try-except for mannwhitneyu as it can raise ValueError for certain data conditions
                    try:
                        stat, p_val = mannwhitneyu(data_r, data_s, alternative='two-sided')
                        p_val_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
                        logger.info(f"Mann-Whitney U for '{analysis_column_name}'{plot_suffix_for_logging}: W={stat:.1f}, {p_val_text} (N={recurrent_n} vs N={single_event_n})")
                    except ValueError as e_mwu: # Catch only MWU specific errors
                        p_val_text = "Test Error (MWU)"
                        logger.warning(f"MWU test failed for '{analysis_column_name}'{plot_suffix_for_logging}: {e_mwu}")
            else: 
                p_val_text = "Test N/A (groups empty)"
        else: 
            p_val_text = "Test N/A (category key missing)"
        
        current_fst_summary = {
            'column_name': analysis_column_name,
            'analysis_type': plot_suffix_for_logging if plot_suffix_for_logging else "unfiltered", 
            'recurrent_N': recurrent_n, 'recurrent_mean': recurrent_mean, 'recurrent_median': recurrent_median,
            'single_event_N': single_event_n, 'single_event_mean': single_event_mean, 'single_event_median': single_event_median,
            'p_value_text': p_val_text,
            'recurrent_inv_freq_mean': recurrent_inv_freq_mean, 'single_event_inv_freq_mean': single_event_inv_freq_mean,
            'recurrent_n0_hap_mean': recurrent_n0_hap_mean, 'single_event_n0_hap_mean': single_event_n0_hap_mean,
            'recurrent_n1_hap_mean': recurrent_n1_hap_mean, 'single_event_n1_hap_mean': single_event_n1_hap_mean,
        }
        returned_summary = current_fst_summary
        
        ax.text(0.04,0.96,f"Mann-Whitney U\n{p_val_text}",transform=ax.transAxes,fontsize=12,va='top',ha='left',
                bbox=dict(boxstyle='round,pad=0.3',fc='ghostwhite',alpha=0.7,ec='lightgrey'))
    
    plt.tight_layout(pad=1.5)
    # Specific try-except for savefig, as file I/O can fail for many reasons
    try:
        plt.savefig(output_filename, dpi=350, bbox_inches='tight')
        logger.info(f"Saved {plot_type} plot{plot_suffix_for_logging} for '{analysis_column_name}' to '{output_filename}'")
    except IOError as e_save:
        logger.error(f"Failed to save {plot_type} plot '{output_filename}': {e_save}")

    if analysis_column_name in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT and plot_type == 'violin':
        logger.info(f"Displaying FST plot for '{analysis_column_name}'. Close plot window to continue...")
        # plt.show() # Disabled for non-interactive run
    
    plt.close(fig) # ensure figure is closed
    return returned_summary

def create_fst_vs_attribute_scatterplot(summary_df_with_types, fst_col, attr_col, attr_name):
    fig, ax = plt.subplots(figsize=(8, 7))
    fst_col_name_pretty = fst_col.replace('_', ' ').title()
    title = f'{fst_col_name_pretty} vs. {attr_name}'
    any_data_plotted = False
    plot_specific_summary = {
        'fst_metric': fst_col,
        'attribute_plotted': attr_col,
        'attribute_name_pretty': attr_name,
    }

    logger.info(f"--- Scatter Plot: {title} ---")
    for inv_type_display_name, inv_type_internal_key in INVERSION_CATEGORY_MAPPING.items():
        subset_df = summary_df_with_types[summary_df_with_types['inversion_type'] == inv_type_internal_key]
        
        current_cat_summary = {'N': 0, 'mean_fst': np.nan, 'mean_attr': np.nan}

        if fst_col not in subset_df.columns or attr_col not in subset_df.columns:
            logger.warning(f"  Category '{inv_type_display_name}': Missing '{fst_col}' or '{attr_col}'. Skipping scatter points.")
            plot_specific_summary[f"{inv_type_internal_key}_N"] = 0
            plot_specific_summary[f"{inv_type_internal_key}_mean_fst"] = np.nan
            plot_specific_summary[f"{inv_type_internal_key}_mean_attr"] = np.nan
            continue

        fst_values = pd.to_numeric(subset_df[fst_col], errors='coerce')
        attr_values = pd.to_numeric(subset_df[attr_col], errors='coerce')
        
        valid_mask = ~fst_values.isna() & ~attr_values.isna()
        fst_values_valid = fst_values[valid_mask]
        attr_values_valid = attr_values[valid_mask]
        
        num_points = len(fst_values_valid)
        current_cat_summary['N'] = num_points
        logger.info(f"  Category '{inv_type_display_name}': Plotting {num_points} valid points for {fst_col} vs {attr_col}.")

        if num_points > 0:
            ax.scatter(attr_values_valid, fst_values_valid, 
                       label=f"{inv_type_display_name} (N={num_points})", 
                       color=SCATTER_COLOR_MAP[inv_type_internal_key], 
                       alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
            any_data_plotted = True
            current_cat_summary['mean_fst'] = fst_values_valid.mean()
            current_cat_summary['mean_attr'] = attr_values_valid.mean()
        else:
            logger.info(f"  Category '{inv_type_display_name}': No valid points to plot.")
        
        plot_specific_summary[f"{inv_type_internal_key}_N"] = current_cat_summary['N']
        plot_specific_summary[f"{inv_type_internal_key}_mean_fst"] = current_cat_summary['mean_fst']
        plot_specific_summary[f"{inv_type_internal_key}_mean_attr"] = current_cat_summary['mean_attr']

    if not any_data_plotted:
        logger.warning(f"No valid data across all categories to plot for '{title}'. Skipping plot generation.")
        plt.close(fig)
        return None

    ax.set_xlabel(attr_name, fontsize=16)
    ax.set_ylabel(fst_col_name_pretty, fontsize=16)
    ax.set_title(title, fontsize=18, pad=18)

    ax.legend(
        title="Inversion Type",
        loc='best',
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1,
        fontsize=12,
        title_fontsize=14,
    )
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=13)
    plt.tight_layout(pad=1.5)
    
    fst_col_safe = "".join(c if c.isalnum() else "_" for c in fst_col).lower()
    attr_col_safe = "".join(c if c.isalnum() else "_" for c in attr_col).lower()
    output_filename = SCATTER_PLOT_TEMPLATE.format(fst_col_safe=fst_col_safe, attr_col_safe=attr_col_safe)
    
    # Specific try-except for savefig
    try:
        plt.savefig(output_filename, dpi=350, bbox_inches='tight')
        logger.info(f"Saved scatter plot to '{output_filename}'")
    except IOError as e_save_scatter:
        logger.error(f"Failed to save scatter plot '{output_filename}': {e_save_scatter}")
        
    plt.close(fig) # ensure figure is closed
    return plot_specific_summary


# --- Main Execution Block ---
def main():
    global SCATTER_PLOT_SUMMARIES 
    overall_start = time.time()
    logger.info(f"--- Starting Inversion Comparison Analysis ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
    logger.info(f"Summary Statistics File: '{SUMMARY_STATS_FILE}'")
    logger.info(f"Inversion Information File: '{INVERSION_FILE}'")
    logger.info(f"Coordinate Map File: '{COORDINATE_MAP_FILE}' (if present, will be used)")

    all_needed_summary_cols = list(SUMMARY_STATS_COORDINATE_COLUMNS.values()) + \
                              [c for c in ANALYSIS_COLUMNS if c not in SUMMARY_STATS_COORDINATE_COLUMNS.values()]
    sum_cols_load = list(dict.fromkeys(all_needed_summary_cols)) # Ensure unique columns

    # Removed try-except for file loading. Script will exit if files not found or critical errors occur.
    if not os.path.exists(INVERSION_FILE):
        logger.critical(f"CRITICAL: Inversion file '{INVERSION_FILE}' not found. Exiting.")
        sys.exit(1)
    if not os.path.exists(SUMMARY_STATS_FILE):
        logger.critical(f"CRITICAL: Summary stats file '{SUMMARY_STATS_FILE}' not found. Exiting.")
        sys.exit(1)

    inv_df = pd.read_csv(INVERSION_FILE, sep='\t', usecols=lambda c: c in INVERSION_FILE_COLUMNS)
    sum_df = pd.read_csv(SUMMARY_STATS_FILE, usecols=lambda c: c in sum_cols_load)
    
    logger.info(f"Loaded data. Summary ('{SUMMARY_STATS_FILE}'): {len(sum_df)} rows. Inversion ('{INVERSION_FILE}'): {len(inv_df)} rows.")

    miss_sum_coords = [c for c in SUMMARY_STATS_COORDINATE_COLUMNS.values() if c not in sum_df.columns]
    if miss_sum_coords: 
        logger.critical(f"CRITICAL: Summary stats file '{SUMMARY_STATS_FILE}' missing coordinate columns: {miss_sum_coords}. Exiting.")
        sys.exit(1)

    map_df = None
    perform_coordinate_mapping = False
    if os.path.exists(COORDINATE_MAP_FILE):
        logger.info(f"'{COORDINATE_MAP_FILE}' found. Attempting to process for coordinate mapping.")
        map_df_temp = pd.read_csv(COORDINATE_MAP_FILE, sep='\t')
        # Normalize chromosome prefixes in mapping file
        map_df_temp['Original_Chr'] = map_df_temp['Original_Chr'].apply(normalize_chromosome_name)
        map_df_temp['New_Chr'] = map_df_temp['New_Chr'].apply(normalize_chromosome_name)
        # Omit mappings for chromosome Y
        map_df_temp = map_df_temp[~map_df_temp['Original_Chr'].eq('y') & ~map_df_temp['New_Chr'].eq('y')]
        
        # Validate map_df_temp columns
        if not all(col in map_df_temp.columns for col in MAP_FILE_COLUMNS):
            logger.warning(f"'{COORDINATE_MAP_FILE}' is missing one or more required columns: {MAP_FILE_COLUMNS}. Coordinate mapping will be disabled.")
            map_df = None
        else:
            map_df = map_df_temp # Assign if columns are okay
            logger.info(f"Loaded '{COORDINATE_MAP_FILE}' with {len(map_df)} mapping entries.")
            
            # Prepare output.csv coordinates for validation lookup
            output_coords_set = set()
            for _, sum_row in sum_df.iterrows():
                try: # try to parse coordinates from sum_df, skip row if error
                    s_chr = normalize_chromosome_name(sum_row[SUMMARY_STATS_COORDINATE_COLUMNS['chr']])
                    s_start = int(sum_row[SUMMARY_STATS_COORDINATE_COLUMNS['start']])
                    s_end = int(sum_row[SUMMARY_STATS_COORDINATE_COLUMNS['end']])
                    output_coords_set.add((s_chr, s_start, s_end))
                except (ValueError, TypeError):
                    logger.debug(f"Skipping row in {SUMMARY_STATS_FILE} during output coordinate set creation due to parsing error: {sum_row.to_dict()}")
                    continue
            
            if not output_coords_set:
                 logger.warning(f"No valid coordinates could be extracted from '{SUMMARY_STATS_FILE}' for map validation. Mapping disabled.")
                 perform_coordinate_mapping = False
                 map_df = None # Disable mapping if output coords are unusable
            elif map_df is not None : # Proceed with validation only if map_df is valid so far
                all_mappable_inv_entries_valid = True
                num_entries_to_map_in_inv_df = 0
                
                # Create a temporary lookup from map_df for efficient validation
                temp_map_lookup = {}
                for _, map_row_val in map_df.iterrows():
                    try:
                        orig_c_val = normalize_chromosome_name(map_row_val['Original_Chr'])
                        orig_s_val = int(map_row_val['Original_Start'])
                        orig_e_val = int(map_row_val['Original_End'])
                        new_c_val = normalize_chromosome_name(map_row_val['New_Chr'])
                        new_s_val = int(map_row_val['New_Start'])
                        new_e_val = int(map_row_val['New_End'])
                        temp_map_lookup[(orig_c_val, orig_s_val, orig_e_val)] = (new_c_val, new_s_val, new_e_val)
                    except ValueError: # Skip malformed map entries during this prep phase
                        logger.debug(f"Skipping malformed row in {COORDINATE_MAP_FILE} during map lookup creation: {map_row_val.to_dict()}")
                        continue

                for _, inv_row in inv_df.iterrows():
                    try: # try to parse inv_df row for validation
                        inv_chr_orig = normalize_chromosome_name(inv_row['Chromosome'])
                        inv_start_orig = int(inv_row['Start'])
                        inv_end_orig = int(inv_row['End'])
                    except (ValueError, TypeError):
                        logger.debug(f"Skipping row in {INVERSION_FILE} during map validation pre-check due to parsing error: {inv_row.to_dict()}")
                        continue

                    inv_coords_tuple_orig = (inv_chr_orig, inv_start_orig, inv_end_orig)
                    
                    if inv_coords_tuple_orig in temp_map_lookup:
                        num_entries_to_map_in_inv_df += 1
                        new_coords_from_map = temp_map_lookup[inv_coords_tuple_orig] # (new_chr, new_start, new_end)
                        
                        # Check if new_coords_from_map "make sense" with output_coords_set
                        found_in_output = any(check_coordinate_overlap(out_csv_coord, new_coords_from_map) for out_csv_coord in output_coords_set)
                        
                        if not found_in_output:
                            logger.warning(f"Validation FAILED for coordinate mapping: Mapped entry {inv_coords_tuple_orig} -> {new_coords_from_map} from '{COORDINATE_MAP_FILE}' does not correspond to any entry in '{SUMMARY_STATS_FILE}'.")
                            continue

                
                if num_entries_to_map_in_inv_df == 0 and not temp_map_lookup.empty:
                    logger.info(f"No coordinates in '{INVERSION_FILE}' match 'Original' coordinates in '{COORDINATE_MAP_FILE}'. Mapping will not be applied.")
                    perform_coordinate_mapping = False
                elif all_mappable_inv_entries_valid and num_entries_to_map_in_inv_df > 0 :
                    logger.info(f"All {num_entries_to_map_in_inv_df} mappable entries from '{INVERSION_FILE}' successfully validated against '{SUMMARY_STATS_FILE}'. Coordinate mapping will be applied.")
                    perform_coordinate_mapping = True
                elif not all_mappable_inv_entries_valid: # This case means validation failed
                    logger.warning(f"Coordinate mapping disabled due to validation failure against '{SUMMARY_STATS_FILE}'. Raw coordinates from '{INVERSION_FILE}' will be used.")
                    perform_coordinate_mapping = False
                    map_df = None # Critical: ensure map_df is None if validation fails
                else: # No mappable entries found, or other edge case
                     logger.info(f"No applicable mappings found or validation criteria not met. Coordinate mapping disabled.")
                     perform_coordinate_mapping = False
                     map_df = None


    else:
        logger.info(f"'{COORDINATE_MAP_FILE}' not found. Using raw coordinates from '{INVERSION_FILE}'.")

    # Call map_coordinates_to_inversion_types with the mapping decision
    # This function is now expected to handle errors internally or log warnings for skippable rows.
    rec_map, single_map = map_coordinates_to_inversion_types(inv_df, map_df, perform_coordinate_mapping)
    
    if not rec_map and not single_map and not inv_df.empty : 
        logger.warning(f"Both recurrent and single-event region maps are empty after processing '{INVERSION_FILE}'. Check input data and mapping logic if '{COORDINATE_MAP_FILE}' was used.")

    logger.info(f"Assigning inversion types to {len(sum_df)} summary regions from '{SUMMARY_STATS_FILE}'...")
    type_assign_start_time = time.time()
    sum_df['inversion_type'] = sum_df.apply(
        lambda r: assign_inversion_type_to_summary_row(r,rec_map,single_map,SUMMARY_STATS_COORDINATE_COLUMNS),axis=1)
    logger.info(f"Completed inversion typing in {time.time()-type_assign_start_time:.2f}s.")
    
    for k_warn in ["summary_coord_parsing_error","summary_coord_logic_error"]:
        sup_c = global_warning_tracker.warnings.get(k_warn,{}).get("suppressed_count",0)
        if sup_c > 0: logger.info(f"{sup_c} warnings for '{k_warn}' suppressed during summary typing.")

    type_cts = sum_df['inversion_type'].value_counts()
    logger.info(f"Counts of regions by assigned inversion type:\n{type_cts.to_string()}")
    if 'coordinate_error' in type_cts: logger.warning(f"{type_cts['coordinate_error']} regions in '{SUMMARY_STATS_FILE}' had coordinate errors during type assignment.")
    if not any(c_type in type_cts for c_type in INVERSION_CATEGORY_MAPPING.values()): 
        logger.warning(f"No regions were classified into known inversion types ('{list(INVERSION_CATEGORY_MAPPING.values())}'). Check inputs and mapping results.")

    # Generate the TSV output file with inversion FST estimates
    write_inversion_fst_to_tsv(sum_df, FST_OUTPUT_TSV)

    categorized_dfs = {}
    for inv_type_display_name, inv_type_internal_key in INVERSION_CATEGORY_MAPPING.items():
        categorized_dfs[inv_type_internal_key] = sum_df[sum_df['inversion_type'] == inv_type_internal_key].copy()

    for current_col in ANALYSIS_COLUMNS:
        # col_start_time = time.time() # Keep if detailed timing per column is needed
        logger.info(f"===== Processing Column: '{current_col}' =====")
        
        prepare_data_for_analysis(sum_df, current_col) 
        
        plot_data_for_current_col = {}
        all_categories_empty_for_plot = True
        for inv_type_key, df_subset in categorized_dfs.items():
            if current_col in df_subset.columns:
                numeric_series = pd.to_numeric(df_subset[current_col], errors='coerce').dropna()
                plot_data_for_current_col[inv_type_key] = numeric_series.tolist()
                if not numeric_series.empty:
                    all_categories_empty_for_plot = False
            else:
                plot_data_for_current_col[inv_type_key] = []
        
        if current_col in COLUMNS_FOR_PLOTTING:
            plot_type_to_use = 'violin' if current_col in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT else 'box'
            logger.info(f"--- Generating {plot_type_to_use.upper()} Plot for: '{current_col}' (Unfiltered) ---")
            
            if all_categories_empty_for_plot:
                logger.warning(f"No numeric data for plotting for '{current_col}' (Unfiltered) across all categories. Skipping plot.")
            else:
                plot_summary = create_comparison_plot(plot_data_for_current_col, categorized_dfs, current_col, plot_type_to_use, plot_suffix_for_logging="")
                if plot_summary: 
                    FST_TEST_SUMMARIES.append(plot_summary)
        else:
            logger.info(f"Skipping violin/box plot generation for '{current_col}' (not in designated plot list).")

    # --- Data Filtering for Haplotype Counts and Second Pass for FST Plots ---
    logger.info(f"\n====== Preparing Data for FILTERED FST Analysis ({FILTER_SUFFIX}) ======")
    logger.info(f"Filter condition: Minimum of '{N_HAP_0_COL}' and '{N_HAP_1_COL}' must be >= 5.")

    sum_df_filtered = pd.DataFrame(columns=sum_df.columns) # Initialize as empty
    if N_HAP_0_COL in sum_df.columns and N_HAP_1_COL in sum_df.columns:
        sum_df[N_HAP_0_COL] = pd.to_numeric(sum_df[N_HAP_0_COL], errors='coerce')
        sum_df[N_HAP_1_COL] = pd.to_numeric(sum_df[N_HAP_1_COL], errors='coerce')
        filter_condition = (sum_df[N_HAP_0_COL] >= 5) & (sum_df[N_HAP_1_COL] >= 5)
        sum_df_filtered = sum_df[filter_condition].copy()
        logger.info(f"Haplotype filter applied: Original rows = {len(sum_df)}, Filtered rows ({FILTER_SUFFIX}) = {len(sum_df_filtered)}")
    else:
        missing_filter_cols = [col for col in [N_HAP_0_COL, N_HAP_1_COL] if col not in sum_df.columns]
        logger.warning(f"One or both filter columns ({', '.join(missing_filter_cols)}) missing from '{SUMMARY_STATS_FILE}'. Proceeding with an empty filtered DataFrame.")
        
    categorized_dfs_filtered = {}
    for inv_type_display_name, inv_type_internal_key in INVERSION_CATEGORY_MAPPING.items():
        if not sum_df_filtered.empty:
            categorized_dfs_filtered[inv_type_internal_key] = sum_df_filtered[sum_df_filtered['inversion_type'] == inv_type_internal_key].copy()
        else:
            categorized_dfs_filtered[inv_type_internal_key] = pd.DataFrame(columns=sum_df_filtered.columns)

    rec_count_filtered = len(categorized_dfs_filtered.get(INVERSION_CATEGORY_MAPPING['Recurrent'], pd.DataFrame()))
    se_count_filtered = len(categorized_dfs_filtered.get(INVERSION_CATEGORY_MAPPING['Single-event'], pd.DataFrame()))
    logger.info(f"Filtered data counts ({FILTER_SUFFIX}): Recurrent regions = {rec_count_filtered}, Single-event regions = {se_count_filtered}")

    logger.info(f"\n====== Starting FILTERED FST Violin Plot Analysis ({FILTER_SUFFIX}) ======")
    if sum_df_filtered.empty:
        logger.warning(f"Skipping filtered FST analysis as the filtered DataFrame ({FILTER_SUFFIX}) is empty.")
    else:
        for current_col_filtered in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT:
            logger.info(f"===== Processing Column (FILTERED {FILTER_SUFFIX}): '{current_col_filtered}' =====")
            
            plot_data_for_current_col_filtered = {}
            all_categories_empty_for_plot_filtered = True
            for inv_type_key, df_subset_filtered in categorized_dfs_filtered.items():
                if current_col_filtered in df_subset_filtered.columns:
                    numeric_series_filtered = pd.to_numeric(df_subset_filtered[current_col_filtered], errors='coerce').dropna()
                    plot_data_for_current_col_filtered[inv_type_key] = numeric_series_filtered.tolist()
                    if not numeric_series_filtered.empty:
                        all_categories_empty_for_plot_filtered = False
                else:
                    plot_data_for_current_col_filtered[inv_type_key] = []
            
            if current_col_filtered not in sum_df_filtered.columns: # Check if col exists in filtered df
                logger.warning(f"Column '{current_col_filtered}' not found in sum_df_filtered. Skipping filtered plot and test.")
                continue

            if all_categories_empty_for_plot_filtered:
                logger.warning(f"No numeric data for plotting (FILTERED {FILTER_SUFFIX}) for '{current_col_filtered}' across all categories. Skipping plot.")
            else:
                logger.info(f"--- Generating VIOLIN Plot (FILTERED {FILTER_SUFFIX}) for: '{current_col_filtered}' ---")
                plot_summary_filtered = create_comparison_plot(
                    plot_data_for_current_col_filtered,
                    categorized_dfs_filtered, 
                    current_col_filtered,
                    'violin',
                    output_filename_template_override=VIOLIN_PLOT_FILTERED_TEMPLATE,
                    plot_suffix_for_logging=FILTER_SUFFIX
                )
                if plot_summary_filtered:
                    FST_TEST_SUMMARIES_FILTERED.append(plot_summary_filtered)

    # --- Calculate and Store FST Component Summaries (using original unfiltered data) ---
    logger.info("\n====== Calculating FST Component Summaries (based on original unfiltered data) ======")
    for component_col_name in ALL_FST_COMPONENT_COLUMNS:
        if component_col_name not in sum_df.columns:
            logger.warning(f"FST Component column '{component_col_name}' not found in summary_df. Skipping its component summary calculation.")
            continue

        component_summary_data = {'component_column': component_col_name, 'column_name': component_col_name}
        logger.info(f"  Calculating summary for component: {component_col_name}")

        for inv_type_display_name, inv_type_internal_key in INVERSION_CATEGORY_MAPPING.items():
            df_subset = categorized_dfs.get(inv_type_internal_key) # Use original categorized_dfs
            mean_val, median_val, n_val = np.nan, np.nan, 0

            if df_subset is not None and not df_subset.empty and component_col_name in df_subset.columns:
                numeric_series = pd.to_numeric(df_subset[component_col_name], errors='coerce').dropna()
                if not numeric_series.empty:
                    mean_val = numeric_series.mean()
                    median_val = numeric_series.median()
                    n_val = len(numeric_series)
                else:
                    logger.debug(f"    No valid numeric data for component '{component_col_name}' in category '{inv_type_display_name}' after NaN drop.")
            
            component_summary_data[f"{inv_type_internal_key}_N"] = n_val
            component_summary_data[f"{inv_type_internal_key}_mean"] = mean_val
            component_summary_data[f"{inv_type_internal_key}_median"] = median_val

        FST_TEST_SUMMARIES.append(component_summary_data)
    logger.info("====== Finished Calculating FST Component Summaries ======")

    logger.info("\n====== Generating Investigative Scatter Plots ======")
    for config in SCATTER_PLOT_CONFIG:
        fst_col = config['fst_col']
        attr_col = config['attr_col']
        attr_name = config['attr_name']
        
        if fst_col not in sum_df.columns or attr_col not in sum_df.columns:
            logger.warning(f"Skipping scatterplot: {fst_col} vs {attr_col}. One or both columns missing from summary data in '{SUMMARY_STATS_FILE}'.")
            continue
        
        plot_summary_scatter = create_fst_vs_attribute_scatterplot(sum_df, fst_col, attr_col, attr_name) # Use original sum_df
        if plot_summary_scatter: 
            SCATTER_PLOT_SUMMARIES.append(plot_summary_scatter)


    final_sup_summary = global_warning_tracker.get_suppressed_summary()
    if final_sup_summary: logger.info(f"\n--- Summary of Suppressed Warnings ---\n{final_sup_summary}")

    # --- Print Unfiltered FST Test Summaries ---
    if FST_TEST_SUMMARIES:
        fst_test_results_to_print = [s for s in FST_TEST_SUMMARIES if 'p_value_text' in s]
        if fst_test_results_to_print:
            logger.info("\n====== Summary Statistics for FST Tests (Recurrent vs Single-event) - Unfiltered Data ======")
            for summary in fst_test_results_to_print:
                logger.info(f"  --- FST Metric: {summary['column_name']} ({summary.get('analysis_type', 'unfiltered')}) ---")
                logger.info(f"    Recurrent:     N={summary['recurrent_N']}, Mean_FST={summary['recurrent_mean']:.4f}, Median_FST={summary['recurrent_median']:.4f}, "
                            f"InvFreq_Mean={summary.get('recurrent_inv_freq_mean', np.nan):.4f}, N_StdHaps_Mean={summary.get('recurrent_n0_hap_mean', np.nan):.2f}, N_InvHaps_Mean={summary.get('recurrent_n1_hap_mean', np.nan):.2f}")
                logger.info(f"    Single-event:  N={summary['single_event_N']}, Mean_FST={summary['single_event_mean']:.4f}, Median_FST={summary['single_event_median']:.4f}, "
                            f"InvFreq_Mean={summary.get('single_event_inv_freq_mean', np.nan):.4f}, N_StdHaps_Mean={summary.get('single_event_n0_hap_mean', np.nan):.2f}, N_InvHaps_Mean={summary.get('single_event_n1_hap_mean', np.nan):.2f}")
                logger.info(f"    Mann-Whitney U: {summary.get('p_value_text', 'N/A')}")
        else:
            logger.info("\n====== No Unfiltered FST Test Summary Statistics (with p-values) to display ======")
            
    # --- Print Scatter Plot Summaries ---
    if SCATTER_PLOT_SUMMARIES:
        logger.info("\n====== Summary Statistics for Scatter Plots (Recurrent vs Single-event) - Unfiltered Data ======")
        rec_key_internal = INVERSION_CATEGORY_MAPPING['Recurrent']
        se_key_internal = INVERSION_CATEGORY_MAPPING['Single-event']
        for summary in SCATTER_PLOT_SUMMARIES:
            logger.info(f"  --- Scatter Plot: {summary['fst_metric']} vs. {summary['attribute_name_pretty']} (Attr: {summary['attribute_plotted']}) ---")
            logger.info(f"    Recurrent:     N={summary.get(f'{rec_key_internal}_N', 0)}, Mean FST={summary.get(f'{rec_key_internal}_mean_fst', np.nan):.4f}, Mean Attribute={summary.get(f'{rec_key_internal}_mean_attr', np.nan):.4f}")
            logger.info(f"    Single-event:  N={summary.get(f'{se_key_internal}_N', 0)}, Mean FST={summary.get(f'{se_key_internal}_mean_fst', np.nan):.4f}, Mean Attribute={summary.get(f'{se_key_internal}_mean_attr', np.nan):.4f}")
    else:
        logger.info("\n====== No Scatter Plot Summary Statistics to display ======")


    # --- Print FST Component Summaries ---
    if FST_TEST_SUMMARIES:
        logger.info("\n====== Summary Statistics for FST Components (Recurrent vs Single-event) - Unfiltered Data ======")
        rec_key_internal = INVERSION_CATEGORY_MAPPING['Recurrent']
        se_key_internal = INVERSION_CATEGORY_MAPPING['Single-event']

        # Weir & Cockerham FST Components
        logger.info("  --- Weir & Cockerham FST Components ---")
        wc_components_found = False
        for summary in FST_TEST_SUMMARIES:
            if 'component_column' in summary and summary['component_column'] in FST_WC_COMPONENT_COLUMNS:
                wc_components_found = True
                col_name = summary['component_column']
                rec_n = summary.get(f'{rec_key_internal}_N', 0)
                rec_mean_str = f"{summary.get(f'{rec_key_internal}_mean', np.nan):.4f}" if rec_n > 0 and not np.isnan(summary.get(f'{rec_key_internal}_mean', np.nan)) else "N/A"
                rec_median_str = f"{summary.get(f'{rec_key_internal}_median', np.nan):.4f}" if rec_n > 0 and not np.isnan(summary.get(f'{rec_key_internal}_median', np.nan)) else "N/A"
                se_n = summary.get(f'{se_key_internal}_N', 0)
                se_mean_str = f"{summary.get(f'{se_key_internal}_mean', np.nan):.4f}" if se_n > 0 and not np.isnan(summary.get(f'{se_key_internal}_mean', np.nan)) else "N/A"
                se_median_str = f"{summary.get(f'{se_key_internal}_median', np.nan):.4f}" if se_n > 0 and not np.isnan(summary.get(f'{se_key_internal}_median', np.nan)) else "N/A"
                logger.info(f"    Component: {col_name}")
                logger.info(f"      Recurrent:     N={rec_n}, Mean={rec_mean_str}, Median={rec_median_str}")
                logger.info(f"      Single-event:  N={se_n}, Mean={se_mean_str}, Median={se_median_str}")
        if not wc_components_found: logger.info("    No Weir & Cockerham FST component data to display.")

        # Hudson FST Components
        logger.info("  --- Hudson FST Components ---")
        hudson_components_found = False
        for summary in FST_TEST_SUMMARIES:
            if 'component_column' in summary and summary['component_column'] in FST_HUDSON_COMPONENT_COLUMNS:
                hudson_components_found = True
                col_name = summary['component_column']
                rec_n = summary.get(f'{rec_key_internal}_N', 0)
                rec_mean_str = f"{summary.get(f'{rec_key_internal}_mean', np.nan):.4f}" if rec_n > 0 and not np.isnan(summary.get(f'{rec_key_internal}_mean', np.nan)) else "N/A"
                rec_median_str = f"{summary.get(f'{rec_key_internal}_median', np.nan):.4f}" if rec_n > 0 and not np.isnan(summary.get(f'{rec_key_internal}_median', np.nan)) else "N/A"
                se_n = summary.get(f'{se_key_internal}_N', 0)
                se_mean_str = f"{summary.get(f'{se_key_internal}_mean', np.nan):.4f}" if se_n > 0 and not np.isnan(summary.get(f'{se_key_internal}_mean', np.nan)) else "N/A"
                se_median_str = f"{summary.get(f'{se_key_internal}_median', np.nan):.4f}" if se_n > 0 and not np.isnan(summary.get(f'{se_key_internal}_median', np.nan)) else "N/A"
                logger.info(f"    Component: {col_name}")
                logger.info(f"      Recurrent:     N={rec_n}, Mean={rec_mean_str}, Median={rec_median_str}")
                logger.info(f"      Single-event:  N={se_n}, Mean={se_mean_str}, Median={se_median_str}")
        if not hudson_components_found: logger.info("    No Hudson FST component data to display.")
    else:
        logger.info("\n====== No FST Component Summary Statistics to display (unfiltered list is empty) ======")
    
    # --- Print Filtered FST Test Summaries ---
    if FST_TEST_SUMMARIES_FILTERED:
        logger.info(f"\n====== Summary Statistics for FST Tests (Recurrent vs Single-event) - FILTERED ({FILTER_SUFFIX}) ======")
        for summary in FST_TEST_SUMMARIES_FILTERED:
            logger.info(f"  --- FST Metric: {summary['column_name']} ({summary.get('analysis_type', FILTER_SUFFIX)}) ---")
            logger.info(f"    Recurrent:     N={summary['recurrent_N']}, Mean_FST={summary.get('recurrent_mean', np.nan):.4f}, Median_FST={summary.get('recurrent_median', np.nan):.4f}, "
                        f"InvFreq_Mean={summary.get('recurrent_inv_freq_mean', np.nan):.4f}, N_StdHaps_Mean={summary.get('recurrent_n0_hap_mean', np.nan):.2f}, N_InvHaps_Mean={summary.get('recurrent_n1_hap_mean', np.nan):.2f}")
            logger.info(f"    Single-event:  N={summary['single_event_N']}, Mean_FST={summary.get('single_event_mean', np.nan):.4f}, Median_FST={summary.get('single_event_median', np.nan):.4f}, "
                        f"InvFreq_Mean={summary.get('single_event_inv_freq_mean', np.nan):.4f}, N_StdHaps_Mean={summary.get('single_event_n0_hap_mean', np.nan):.2f}, N_InvHaps_Mean={summary.get('single_event_n1_hap_mean', np.nan):.2f}")
            logger.info(f"    Mann-Whitney U: {summary.get('p_value_text', 'N/A')}")
    else:
        logger.info(f"\n====== No FILTERED FST Test Summary Statistics ({FILTER_SUFFIX}) to display (list is empty or no data passed filter) ======")

    logger.info(f"====== Full Analysis Script completed in {time.time()-overall_start:.2f}s ======")

if __name__ == "__main__":
    main()
