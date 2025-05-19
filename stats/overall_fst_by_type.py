import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import logging
import sys
import time
import seaborn as sns
from scipy.stats import mannwhitneyu

# --- Configuration ---

# Input Files
SUMMARY_STATS_FILE = 'output.csv'
INVERSION_FILE = 'inv_info.csv'

# Output File Templates
VIOLIN_PLOT_TEMPLATE = 'comparison_violin_{column_safe_name}.png'
BOX_PLOT_TEMPLATE = 'comparison_boxplot_{column_safe_name}.png'
SCATTER_PLOT_TEMPLATE = 'scatter_fst_{fst_col_safe}_vs_{attr_col_safe}.png'


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
INVERSION_FILE_COLUMNS = ['Chromosome', 'Start', 'End', '0_single_1_recur']

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
    if not chromosome_id_str.startswith('chr'): chromosome_id_str = f"chr{chromosome_id_str}"
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
def map_coordinates_to_inversion_types(inversion_info_df):
    recurrent_regions = {}
    single_event_regions = {}
    warn_key = "inversion_file_row_skip"
    if not all(col in inversion_info_df.columns for col in INVERSION_FILE_COLUMNS):
        raise ValueError(f"Inversion data missing required columns: {[c for c in INVERSION_FILE_COLUMNS if c not in inversion_info_df.columns]}")
    logger.info(f"Mapping inversion types from {len(inversion_info_df)} entries...")
    parsed = 0; skipped = 0
    for index, row in inversion_info_df.iterrows():
        try:
            chrom = normalize_chromosome_name(row['Chromosome'])
            if any(pd.isna(row[col]) for col in ['Start', 'End', '0_single_1_recur']):
                nan_cols = [col for col in ['Start', 'End', '0_single_1_recur'] if pd.isna(row[col])]
                raise ValueError(f"Essential data NaN in: {', '.join(nan_cols)}")
            start, end, cat_code = int(row['Start']), int(row['End']), int(row['0_single_1_recur'])
            if start > end: raise ValueError("Start > End")
            parsed += 1
            if cat_code == 1: recurrent_regions.setdefault(chrom, []).append((start, end))
            elif cat_code == 0: single_event_regions.setdefault(chrom, []).append((start, end))
            else: raise ValueError(f"Unrecognized category code '{cat_code}'")
        except (ValueError, TypeError) as e:
            global_warning_tracker.log_warning(warn_key, lambda r, i, err: f"Skipping row {i+2} in inversion data: {err}. Row: {r.to_dict()}", row, index, e)
            skipped += 1
    logger.info(f"Mapped {parsed} inversion entries. Skipped {skipped} invalid entries.")
    if parsed == 0: logger.warning("No valid inversion entries mapped.")
    suppressed = global_warning_tracker.warnings.get(warn_key, {}).get("suppressed_count",0)
    if suppressed > 0: logger.info(f"{suppressed} warnings for '{warn_key}' suppressed.")
    return recurrent_regions, single_event_regions

def check_coordinate_overlap(summary_coords, inv_coords):
    _, s_start, s_end = summary_coords
    i_start, i_end = inv_coords
    return abs(s_start - i_start) <= 1 and abs(s_end - i_end) <= 1

def determine_region_inversion_type(chrom, start, end, recurrent_map, single_map):
    rec_matches = recurrent_map.get(chrom, [])
    sing_matches = single_map.get(chrom, [])
    curr_coords = (chrom, start, end)
    is_rec = any(check_coordinate_overlap(curr_coords, r_coords) for r_coords in rec_matches)
    is_sing = any(check_coordinate_overlap(curr_coords, s_coords) for s_coords in sing_matches)
    if is_rec and not is_sing: return INVERSION_CATEGORY_MAPPING['Recurrent']
    if is_sing and not is_rec: return INVERSION_CATEGORY_MAPPING['Single-event']
    if is_rec and is_sing:
        logger.debug(f"Ambiguous match for {chrom}:{start}-{end}. Matches both.")
        return 'ambiguous_match'
    return 'no_match'

def assign_inversion_type_to_summary_row(row, rec_map, sing_map, coord_conf):
    p_key="summary_coord_parsing_error"; l_key="summary_coord_logic_error"
    try:
        chrom = normalize_chromosome_name(row[coord_conf['chr']])
        if any(pd.isna(row[coord_conf[c]]) for c in ['start', 'end']):
            nan_cols = [k for k,v in coord_conf.items() if k!='chr' and pd.isna(row[v])]
            raise ValueError(f"Essential coord NaN in: {', '.join(nan_cols)}")
        start, end = int(row[coord_conf['start']]), int(row[coord_conf['end']])
        if start > end:
            global_warning_tracker.log_warning(l_key, lambda r,c,s,e:f"Invalid coords (start>end): {r.get(c['chr'],'N/A')}:{s}-{e}", row,coord_conf,start,end)
            return 'coordinate_error'
        return determine_region_inversion_type(chrom, start, end, rec_map, sing_map)
    except (ValueError, TypeError) as e:
        global_warning_tracker.log_warning(p_key, lambda r,err:f"Coord parsing error for summary row: {err}. Row: {r.to_dict()}", row,e)
        return 'coordinate_error'
    except KeyError as e:
        logger.warning(f"Missing coord col in summary row: {e}. Row: {row.to_dict()}. Marking 'coordinate_error'.")
        return 'coordinate_error'

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
            # This is normal if a category has no data, so not a warning.
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
                # Iterate using .items() for Series to get index and value if needed
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
        if s1_stats and s2_stats: # Both categories must exist
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

    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_title(f'Comparison of {metric_name}', fontsize=15, pad=18)
    xt_lbls_n = [f"{lbl}\n(N={meta[lbl]['n_points']})" for lbl in plot_labels]
    ax.set_xticks(positions); ax.set_xticklabels(xt_lbls_n, fontsize=11)
    ax.set_xlabel(""); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=10); ax.tick_params(axis='x', length=0)
    if all_vals_for_ylim:
        min_v, max_v = np.min(all_vals_for_ylim), np.max(all_vals_for_ylim); dr = max_v - min_v
        pad = 0.1 if dr == 0 and min_v == 0 else (np.abs(min_v * 0.1) if dr == 0 else max(dr * 0.08, 0.005))
        ax.set_ylim(min_v - pad, max_v + pad * 1.5)
    else: ax.set_ylim(-0.02, 0.1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey'); ax.spines['left'].set_color('grey')
    ax.yaxis.grid(True, linestyle=':', which='major', color='lightgrey', alpha=0.7); ax.set_axisbelow(True)

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
    returned_summary = None # Initialize to None, will be populated if an FST summary is generated

    # Determine filename template based on overrides or defaults
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
        return None # Return None as no summary can be generated

    try:
        _plot_common_elements(ax, plot_data_for_current_col, analysis_column_name, plot_specific_draw_func)

        # Perform statistical test and prepare summary ONLY for FST violin plots
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

                # Use the provided categorized_dfs_for_summary for detailed stats
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

                if data_r and data_s: # Both groups must have data for the test
                    if np.var(data_r) == 0 and np.var(data_s) == 0 and np.mean(data_r) == np.mean(data_s):
                        p_val_text = "p = 1.0 (Identical)"
                    else:
                        try:
                            stat, p_val = mannwhitneyu(data_r, data_s, alternative='two-sided')
                            p_val_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
                            logger.info(f"Mann-Whitney U for '{analysis_column_name}'{plot_suffix_for_logging}: W={stat:.1f}, {p_val_text} (N={recurrent_n} vs N={single_event_n})")
                        except ValueError as e:
                            p_val_text = "Test Error"
                            logger.warning(f"MWU test failed for '{analysis_column_name}'{plot_suffix_for_logging}: {e}")
                else: # One or both groups are empty
                    p_val_text = "Test N/A (groups empty)"
            else: # Recurrent or Single-event keys not found in mapping (should not happen with current setup)
                p_val_text = "Test N/A (category key missing)"
            
            current_fst_summary = {
                'column_name': analysis_column_name,
                'analysis_type': plot_suffix_for_logging if plot_suffix_for_logging else "unfiltered", # Add analysis type
                'recurrent_N': recurrent_n, 'recurrent_mean': recurrent_mean, 'recurrent_median': recurrent_median,
                'single_event_N': single_event_n, 'single_event_mean': single_event_mean, 'single_event_median': single_event_median,
                'p_value_text': p_val_text,
                'recurrent_inv_freq_mean': recurrent_inv_freq_mean, 'single_event_inv_freq_mean': single_event_inv_freq_mean,
                'recurrent_n0_hap_mean': recurrent_n0_hap_mean, 'single_event_n0_hap_mean': single_event_n0_hap_mean,
                'recurrent_n1_hap_mean': recurrent_n1_hap_mean, 'single_event_n1_hap_mean': single_event_n1_hap_mean,
            }
            returned_summary = current_fst_summary # Set the summary to be returned
            
            ax.text(0.04,0.96,f"Mann-Whitney U\n{p_val_text}",transform=ax.transAxes,fontsize=10,va='top',ha='left',
                    bbox=dict(boxstyle='round,pad=0.3',fc='ghostwhite',alpha=0.7,ec='lightgrey'))
        
        plt.tight_layout(pad=1.5)
        plt.savefig(output_filename, dpi=350, bbox_inches='tight')
        logger.info(f"Saved {plot_type} plot{plot_suffix_for_logging} for '{analysis_column_name}' to '{output_filename}'")
        
        # Display FST plots if they are part of the specific list (original behavior, no suffix needed for this specific log message)
        if analysis_column_name in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT and plot_type == 'violin':
            logger.info(f"Displaying FST plot for '{analysis_column_name}'. Close plot window to continue...")
            # plt.show() # Disabled for non-interactive run
    
    except Exception as e:
        logger.error(f"Failed to create/save {plot_type} plot{plot_suffix_for_logging} '{output_filename}' for '{analysis_column_name}': {e}", exc_info=True)
    finally:
        plt.close(fig)
    
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
        return None # Return None if no plot generated

    ax.set_xlabel(attr_name, fontsize=13)
    ax.set_ylabel(fst_col_name_pretty, fontsize=13)
    ax.set_title(title, fontsize=15, pad=18)
    
    ax.legend(title="Inversion Type", loc='best', frameon=True, fancybox=True, shadow=True, borderpad=1)
    ax.grid(True, linestyle=':', which='major', color='lightgrey', alpha=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout(pad=1.5)
    
    fst_col_safe = "".join(c if c.isalnum() else "_" for c in fst_col).lower()
    attr_col_safe = "".join(c if c.isalnum() else "_" for c in attr_col).lower()
    output_filename = SCATTER_PLOT_TEMPLATE.format(fst_col_safe=fst_col_safe, attr_col_safe=attr_col_safe)
    
    try:
        plt.savefig(output_filename, dpi=350, bbox_inches='tight')
        logger.info(f"Saved scatter plot to '{output_filename}'")
    except Exception as e:
        logger.error(f"Failed to save scatter plot '{output_filename}': {e}", exc_info=True)
    finally:
        plt.close(fig)
    
    return plot_specific_summary


# --- Main Execution Block ---
def main():
    global SCATTER_PLOT_SUMMARIES # Declare we are modifying the global list
    overall_start = time.time()
    logger.info(f"--- Starting Inversion Comparison Analysis ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
    logger.info(f"Summary Statistics File: '{SUMMARY_STATS_FILE}'")
    logger.info(f"Inversion Information File: '{INVERSION_FILE}'")
    logger.info(f"All columns processed for data quality: {ANALYSIS_COLUMNS}")
    logger.info(f"Violin/Box plots will be generated ONLY for: {COLUMNS_FOR_PLOTTING}")
    logger.info(f"  - Violin plots (with test & display) for FST measures: {FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT}")
    logger.info(f"  - Box plots for other selected measures: {OTHER_COLUMNS_FOR_BOX_PLOT}")
    logger.info(f"  - Scatter plots for FST vs attributes: Defined in SCATTER_PLOT_CONFIG")
    logger.info("NOTE: All numeric data is used for plots/tests; out-of-spec checks are for logging/awareness.")

    all_needed_summary_cols = list(SUMMARY_STATS_COORDINATE_COLUMNS.values()) + \
                              [c for c in ANALYSIS_COLUMNS if c not in SUMMARY_STATS_COORDINATE_COLUMNS.values()]
    sum_cols_load = list(dict.fromkeys(all_needed_summary_cols))

    try:
        inv_df = pd.read_csv(INVERSION_FILE, usecols=INVERSION_FILE_COLUMNS)
        sum_df = pd.read_csv(SUMMARY_STATS_FILE, usecols=sum_cols_load)
    except FileNotFoundError as e: logger.critical(f"CRITICAL: Input file not found. {e}"); sys.exit(1)
    except ValueError as e: logger.critical(f"CRITICAL: Column error in input. {e}"); sys.exit(1)
    except Exception as e: logger.critical(f"CRITICAL reading inputs: {e}", exc_info=True); sys.exit(1)
    logger.info(f"Loaded data. Summary: {len(sum_df)} rows. Inversion: {len(inv_df)} rows.")

    miss_sum_coords = [c for c in SUMMARY_STATS_COORDINATE_COLUMNS.values() if c not in sum_df.columns]
    if miss_sum_coords: logger.critical(f"CRITICAL: Summary stats missing coords: {miss_sum_coords}"); sys.exit(1)

    try: rec_map, single_map = map_coordinates_to_inversion_types(inv_df)
    except Exception as e: logger.critical(f"CRITICAL processing inversion data: {e}", exc_info=True); sys.exit(1)
    if not rec_map and not single_map: logger.warning("Both recurrent and single-event region maps are empty.")

    logger.info(f"Assigning inversion types to {len(sum_df)} summary regions...")
    type_start = time.time()
    sum_df['inversion_type'] = sum_df.apply(
        lambda r: assign_inversion_type_to_summary_row(r,rec_map,single_map,SUMMARY_STATS_COORDINATE_COLUMNS),axis=1)
    logger.info(f"Completed inversion typing in {time.time()-type_start:.2f}s.")
    
    for k in ["summary_coord_parsing_error","summary_coord_logic_error"]:
        sup_c = global_warning_tracker.warnings.get(k,{}).get("suppressed_count",0)
        if sup_c > 0: logger.info(f"{sup_c} warnings for '{k}' suppressed during summary typing.")

    type_cts = sum_df['inversion_type'].value_counts()
    logger.info(f"Counts of regions by assigned inversion type:\n{type_cts.to_string()}")
    if 'coordinate_error' in type_cts: logger.warning(f"{type_cts['coordinate_error']} regions had coord errors.")
    if not any(c in type_cts for c in INVERSION_CATEGORY_MAPPING.values()): logger.warning("No regions classified. Check inputs.")

    categorized_dfs = {}
    for inv_type_display_name, inv_type_internal_key in INVERSION_CATEGORY_MAPPING.items():
         categorized_dfs[inv_type_internal_key] = sum_df[sum_df['inversion_type'] == inv_type_internal_key].copy()

    for current_col in ANALYSIS_COLUMNS:
        col_start_time = time.time()
        logger.info(f"===== Processing Column: '{current_col}' =====")
        
        prepare_data_for_analysis(sum_df, current_col) # For DQ logging
        
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
                # Call create_comparison_plot and capture the summary if one is returned
                # The plot_suffix_for_logging defaults to "" for unfiltered plots, or can be explicitly ""
                plot_summary = create_comparison_plot(plot_data_for_current_col, categorized_dfs, current_col, plot_type_to_use, plot_suffix_for_logging="")
                if plot_summary: # If a summary was returned (i.e., it was an FST violin plot)
                    FST_TEST_SUMMARIES.append(plot_summary)
        else:
            logger.info(f"Skipping violin/box plot generation for '{current_col}' (not in designated plot list).")

    # --- Data Filtering for Haplotype Counts and Second Pass for FST Plots ---
    logger.info(f"\n====== Preparing Data for FILTERED FST Analysis ({FILTER_SUFFIX}) ======")
    logger.info(f"Filter condition: Minimum of '{N_HAP_0_COL}' and '{N_HAP_1_COL}' must be > 4 (i.e., >= 5).")

    # filter columns are numeric in the original sum_df before filtering
    # This step is crucial if these columns haven't been robustly converted to numeric yet.
    if N_HAP_0_COL in sum_df.columns:
        sum_df[N_HAP_0_COL] = pd.to_numeric(sum_df[N_HAP_0_COL], errors='coerce')
    else:
        logger.error(f"Filter column '{N_HAP_0_COL}' not found in summary DataFrame. Cannot apply filter.")
        # Depending on desired behavior, could exit or proceed without filtering. Here, we'll log and proceed, filtered_df might be empty.
    if N_HAP_1_COL in sum_df.columns:
        sum_df[N_HAP_1_COL] = pd.to_numeric(sum_df[N_HAP_1_COL], errors='coerce')
    else:
        logger.error(f"Filter column '{N_HAP_1_COL}' not found in summary DataFrame. Cannot apply filter.")

    # Apply filter if both columns exist
    if N_HAP_0_COL in sum_df.columns and N_HAP_1_COL in sum_df.columns:
        filter_condition = (sum_df[N_HAP_0_COL] >= 5) & (sum_df[N_HAP_1_COL] >= 5)
        sum_df_filtered = sum_df[filter_condition].copy()
        logger.info(f"Haplotype filter applied: Original rows = {len(sum_df)}, Filtered rows ({FILTER_SUFFIX}) = {len(sum_df_filtered)}")
    else:
        logger.warning(f"One or both filter columns ('{N_HAP_0_COL}', '{N_HAP_1_COL}') missing. Proceeding with an empty filtered DataFrame.")
        sum_df_filtered = pd.DataFrame(columns=sum_df.columns) # Create empty df with same columns to avoid downstream errors

    categorized_dfs_filtered = {}
    for inv_type_display_name, inv_type_internal_key in INVERSION_CATEGORY_MAPPING.items():
        if not sum_df_filtered.empty:
            categorized_dfs_filtered[inv_type_internal_key] = sum_df_filtered[sum_df_filtered['inversion_type'] == inv_type_internal_key].copy()
        else:
            categorized_dfs_filtered[inv_type_internal_key] = pd.DataFrame(columns=sum_df_filtered.columns) # Empty df with correct columns

    rec_count_filtered = len(categorized_dfs_filtered.get(INVERSION_CATEGORY_MAPPING['Recurrent'], pd.DataFrame()))
    se_count_filtered = len(categorized_dfs_filtered.get(INVERSION_CATEGORY_MAPPING['Single-event'], pd.DataFrame()))
    logger.info(f"Filtered data counts ({FILTER_SUFFIX}): Recurrent regions = {rec_count_filtered}, Single-event regions = {se_count_filtered}")

    logger.info(f"\n====== Starting FILTERED FST Violin Plot Analysis ({FILTER_SUFFIX}) ======")
    if sum_df_filtered.empty:
        logger.warning(f"Skipping filtered FST analysis as the filtered DataFrame ({FILTER_SUFFIX}) is empty.")
    else:
        for current_col_filtered in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT:
            logger.info(f"===== Processing Column (FILTERED {FILTER_SUFFIX}): '{current_col_filtered}' =====")
            
            # Optional: Data Quality check on the FILTERED data for this column
            # prepare_data_for_analysis(sum_df_filtered, current_col_filtered) # Logs DQ info

            plot_data_for_current_col_filtered = {}
            all_categories_empty_for_plot_filtered = True
            for inv_type_key, df_subset_filtered in categorized_dfs_filtered.items():
                if current_col_filtered in df_subset_filtered.columns:
                    #  data is numeric and drop NaNs for plotting
                    numeric_series_filtered = pd.to_numeric(df_subset_filtered[current_col_filtered], errors='coerce').dropna()
                    plot_data_for_current_col_filtered[inv_type_key] = numeric_series_filtered.tolist()
                    if not numeric_series_filtered.empty:
                        all_categories_empty_for_plot_filtered = False
                else:
                    plot_data_for_current_col_filtered[inv_type_key] = []
            
            if current_col_filtered not in sum_df_filtered.columns:
                logger.warning(f"Column '{current_col_filtered}' not found in sum_df_filtered. Skipping filtered plot and test.")
                continue

            if all_categories_empty_for_plot_filtered:
                logger.warning(f"No numeric data for plotting (FILTERED {FILTER_SUFFIX}) for '{current_col_filtered}' across all categories. Skipping plot.")
            else:
                logger.info(f"--- Generating VIOLIN Plot (FILTERED {FILTER_SUFFIX}) for: '{current_col_filtered}' ---")
                plot_summary_filtered = create_comparison_plot(
                    plot_data_for_current_col_filtered,
                    categorized_dfs_filtered, # Pass the full categorized DFs for aux data like N_hap in summary
                    current_col_filtered,
                    'violin',
                    output_filename_template_override=VIOLIN_PLOT_FILTERED_TEMPLATE,
                    plot_suffix_for_logging=FILTER_SUFFIX
                )
                if plot_summary_filtered:
                    FST_TEST_SUMMARIES_FILTERED.append(plot_summary_filtered)

    # --- Calculate and Store FST Component Summaries (using original unfiltered data) ---
    logger.info("\n====== Calculating FST Component Summaries (based on original unfiltered data) ======")
    # The global FST_TEST_SUMMARIES is used here as these components are for the main FST measures from original data.
    # If components for filtered data were desired, a separate loop and list would be needed.
    for component_col_name in ALL_FST_COMPONENT_COLUMNS:
        # Check if the component column exists in the main summary dataframe to avoid errors if it's missing
        if component_col_name not in sum_df.columns:
            logger.warning(f"FST Component column '{component_col_name}' not found in summary_df. Skipping its component summary calculation.")
            continue

        component_summary_data = {'component_column': component_col_name, 'column_name': component_col_name}
        logger.info(f"  Calculating summary for component: {component_col_name}")

        for inv_type_display_name, inv_type_internal_key in INVERSION_CATEGORY_MAPPING.items():
            df_subset = categorized_dfs.get(inv_type_internal_key)
            mean_val, median_val, n_val = np.nan, np.nan, 0

            if df_subset is not None and not df_subset.empty and component_col_name in df_subset.columns:
                # Convert to numeric, coercing errors, and drop NaN values for calculation
                numeric_series = pd.to_numeric(df_subset[component_col_name], errors='coerce').dropna()
                if not numeric_series.empty:
                    mean_val = numeric_series.mean()
                    median_val = numeric_series.median()
                    n_val = len(numeric_series)
                else:
                    logger.debug(f"    No valid numeric data for component '{component_col_name}' in category '{inv_type_display_name}' after NaN drop.")
            elif df_subset is None or df_subset.empty:
                 logger.debug(f"    DataFrame subset for category '{inv_type_display_name}' is None or empty for component '{component_col_name}'.")
            elif component_col_name not in df_subset.columns:
                 logger.debug(f"    Component column '{component_col_name}' not found in DataFrame subset for category '{inv_type_display_name}'.")


            # Store results using the internal key (e.g., 'recurrent', 'single_event')
            component_summary_data[f"{inv_type_internal_key}_N"] = n_val
            component_summary_data[f"{inv_type_internal_key}_mean"] = mean_val
            component_summary_data[f"{inv_type_internal_key}_median"] = median_val
            mean_debug_str = f"{mean_val:.4f}" if not np.isnan(mean_val) else "NaN"
            median_debug_str = f"{median_val:.4f}" if not np.isnan(median_val) else "NaN"
            logger.debug(f"    Stats for '{inv_type_display_name}': N={n_val}, Mean={mean_debug_str}, Median={median_debug_str}")

        FST_TEST_SUMMARIES.append(component_summary_data)
    logger.info("====== Finished Calculating FST Component Summaries ======")

    logger.info("\n====== Generating Investigative Scatter Plots ======")
    for config in SCATTER_PLOT_CONFIG:
        fst_col = config['fst_col']
        attr_col = config['attr_col']
        attr_name = config['attr_name']
        
        if fst_col not in sum_df.columns or attr_col not in sum_df.columns:
            logger.warning(f"Skipping scatterplot: {fst_col} vs {attr_col}. One or both columns missing from summary data.")
            continue
        
        # create_fst_vs_attribute_scatterplot now returns summary data for the plot
        plot_summary = create_fst_vs_attribute_scatterplot(sum_df, fst_col, attr_col, attr_name)
        if plot_summary: # Only append if plot was generated (i.e., data was present)
            SCATTER_PLOT_SUMMARIES.append(plot_summary)


    final_sup_summary = global_warning_tracker.get_suppressed_summary()
    if final_sup_summary: logger.info(f"\n--- Summary of Suppressed Warnings ---\n{final_sup_summary}")

    if FST_TEST_SUMMARIES:
            fst_test_results_to_print = [s for s in FST_TEST_SUMMARIES if 'p_value_text' in s]
            if fst_test_results_to_print:
                logger.info("\n====== Summary Statistics for FST Tests (Recurrent vs Single-event) ======")
                for summary in fst_test_results_to_print:
                    logger.info(f"  --- FST Metric: {summary['column_name']} ---")
                    logger.info(f"    Recurrent:     N={summary['recurrent_N']}, Mean_FST={summary['recurrent_mean']:.4f}, Median_FST={summary['recurrent_median']:.4f}, "
                                f"InvFreq_Mean={summary.get('recurrent_inv_freq_mean', np.nan):.4f}, N_StdHaps_Mean={summary.get('recurrent_n0_hap_mean', np.nan):.2f}, N_InvHaps_Mean={summary.get('recurrent_n1_hap_mean', np.nan):.2f}")
                    logger.info(f"    Single-event:  N={summary['single_event_N']}, Mean_FST={summary['single_event_mean']:.4f}, Median_FST={summary['single_event_median']:.4f}, "
                                f"InvFreq_Mean={summary.get('single_event_inv_freq_mean', np.nan):.4f}, N_StdHaps_Mean={summary.get('single_event_n0_hap_mean', np.nan):.2f}, N_InvHaps_Mean={summary.get('single_event_n1_hap_mean', np.nan):.2f}")
                    logger.info(f"    Mann-Whitney U: {summary.get('p_value_text', 'N/A (Not applicable for this entry)')}")
            else:
                logger.info("\n====== No FST Test Summary Statistics (with p-values) to display ======")

    logger.info("\n====== Summary Statistics for Scatter Plots (Recurrent vs Single-event) ======")
    for summary in SCATTER_PLOT_SUMMARIES:
        rec_key = INVERSION_CATEGORY_MAPPING['Recurrent']
        se_key = INVERSION_CATEGORY_MAPPING['Single-event']
        logger.info(f"  --- Scatter Plot: {summary['fst_metric']} vs. {summary['attribute_name_pretty']} (Attr: {summary['attribute_plotted']}) ---")
        logger.info(f"    Recurrent:     N={summary.get(f'{rec_key}_N', 0)}, Mean FST={summary.get(f'{rec_key}_mean_fst', np.nan):.4f}, Mean Attribute={summary.get(f'{rec_key}_mean_attr', np.nan):.4f}")
        logger.info(f"    Single-event:  N={summary.get(f'{se_key}_N', 0)}, Mean FST={summary.get(f'{se_key}_mean_fst', np.nan):.4f}, Mean Attribute={summary.get(f'{se_key}_mean_attr', np.nan):.4f}")

    # --- Print FST Component Summaries ---
    if FST_TEST_SUMMARIES:
        logger.info("\n====== Summary Statistics for FST Components (Recurrent vs Single-event) ======")
        rec_key_internal = INVERSION_CATEGORY_MAPPING['Recurrent']
        se_key_internal = INVERSION_CATEGORY_MAPPING['Single-event']

        # --- Weir & Cockerham FST Components ---
        logger.info("  --- Weir & Cockerham FST Components ---")
        wc_components_found = False
        for summary in FST_TEST_SUMMARIES:
            # Process only if it's a component summary and relevant to Weir & Cockerham
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
                logger.info(f"      Recurrent:    N={rec_n}, Mean={rec_mean_str}, Median={rec_median_str}")
                logger.info(f"      Single-event: N={se_n}, Mean={se_mean_str}, Median={se_median_str}")
        if not wc_components_found:
            logger.info("    No Weir & Cockerham FST component data to display.")

        # --- Hudson FST Components ---
        logger.info("  --- Hudson FST Components ---")
        hudson_components_found = False
        for summary in FST_TEST_SUMMARIES:
            # Process only if it's a component summary and relevant to Hudson
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
                logger.info(f"      Recurrent:    N={rec_n}, Mean={rec_mean_str}, Median={rec_median_str}")
                logger.info(f"      Single-event: N={se_n}, Mean={se_mean_str}, Median={se_median_str}")
        if not hudson_components_found:
            logger.info("    No Hudson FST component data to display.")
    else:
        logger.info("\n====== No FST Component Summary Statistics to display (list is empty or calculation was skipped) ======")
    
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
