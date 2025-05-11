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
    'hudson_pi_avg_hap_group_0v1'
]

# Columns for which to perform statistical tests (Recurrent vs Single-event)
# These will also be the FST violin plots that are displayed.
FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT = [
    'haplotype_overall_fst_wc',
    'hudson_fst_hap_group_0v1'
]

# Additional columns for which to generate box plots (no statistical test annotation)
OTHER_COLUMNS_FOR_BOX_PLOT = [
    'haplotype_num_informative_sites_wc', # Basic count/diversity indicator
    'hudson_dxy_hap_group_0v1',         # Absolute divergence
    'hudson_pi_avg_hap_group_0v1',      # Average internal diversity
    'haplotype_within_pop_variance_wc'  # Variance component
]

# Combine for the master list of columns that will have some plot generated
COLUMNS_FOR_PLOTTING = FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT + OTHER_COLUMNS_FOR_BOX_PLOT


SUMMARY_STATS_COORDINATE_COLUMNS = {'chr': 'chr', 'start': 'region_start', 'end': 'region_end'}
INVERSION_FILE_COLUMNS = ['Chromosome', 'Start', 'End', '0_single_1_recur']

INVERSION_CATEGORY_MAPPING = {
    'Recurrent': 'recurrent',
    'Single-event': 'single_event'
}
COLOR_PALETTE = sns.color_palette("Set2", n_colors=len(INVERSION_CATEGORY_MAPPING))

DATA_QUALITY_DISCREPANCY_THRESHOLD = 0.20
MAX_INDIVIDUAL_WARNINGS_PER_CATEGORY = 10 
MAX_EXAMPLES_OF_OUT_OF_SPEC_LOGGING = 5 

FST_TEST_SUMMARIES = []

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
    logger.info(f"--- Preparing data for column: '{column_name}' ---")
    if column_name not in summary_df_with_types.columns:
        logger.error(f"Column '{column_name}' not found. Skipping.")
        return {k: [] for k in INVERSION_CATEGORY_MAPPING.values()}
    
    cat_stats = {}; cat_plot_data = {k: [] for k in INVERSION_CATEGORY_MAPPING.values()}
    min_exp, max_exp, val_type, _ = get_column_value_specifications(column_name)

    for disp_name, int_key in INVERSION_CATEGORY_MAPPING.items():
        subset_df = summary_df_with_types[summary_df_with_types['inversion_type'] == int_key]
        raw_series = subset_df[column_name]; initial_n = len(raw_series)
        stats = cat_stats[int_key] = {'initial':initial_n, 'missing_non_numeric':0, 'numeric_for_analysis':0, 'flagged_oos':0}
        if initial_n == 0:
            logger.info(f"No regions for '{disp_name}', column '{column_name}'.")
            continue
        
        num_series_attempts = pd.to_numeric(raw_series, errors='coerce')
        stats['missing_non_numeric'] = num_series_attempts.isna().sum()
        valid_numerics = num_series_attempts.dropna()
        stats['numeric_for_analysis'] = len(valid_numerics)
        cat_plot_data[int_key] = valid_numerics.tolist()
        
        if not valid_numerics.empty:
            oos_mask = pd.Series(False, index=valid_numerics.index)
            if val_type == 'integer':
                oos_mask |= (valid_numerics != np.floor(valid_numerics)) & (np.abs(valid_numerics - np.round(valid_numerics)) > 1e-9)
            oos_mask |= (valid_numerics < min_exp) | (valid_numerics > max_exp)
            stats['flagged_oos'] = oos_mask.sum()
            if stats['flagged_oos'] > 0:
                logger.info(f"  For '{disp_name}', '{column_name}': {stats['flagged_oos']} numeric values flagged as out-of-spec (USED IN PLOTS/TESTS). Examples:")
                shown = 0
                for idx, problem in oos_mask.items():
                    if problem and shown < MAX_EXAMPLES_OF_OUT_OF_SPEC_LOGGING:
                        val = valid_numerics[idx]; reasons = []
                        if val_type=='integer' and ((val!=np.floor(val)) and (np.abs(val-np.round(val))>1e-9)): reasons.append(f"expected int, got {val:.4g}")
                        if val < min_exp: reasons.append(f"below min {min_exp} (is {val:.4g})")
                        if val > max_exp: reasons.append(f"above max {max_exp} (is {val:.4g})")
                        logger.info(f"    - Value {val:.4g}: {'; '.join(reasons) or 'flagged issue'}")
                        shown += 1
                    elif problem and shown == MAX_EXAMPLES_OF_OUT_OF_SPEC_LOGGING:
                        logger.info("    - (Further out-of-spec examples suppressed)"); shown+=1; break
        
        prop_miss = stats['missing_non_numeric']/initial_n if initial_n else 0
        prop_flag = stats['flagged_oos']/stats['numeric_for_analysis'] if stats['numeric_for_analysis'] else 0
        logger.info(f"  Col '{column_name}', Cat '{disp_name}': Initial={initial_n}, Missing={stats['missing_non_numeric']} ({prop_miss:.2%}). "
                    f"Using {stats['numeric_for_analysis']} numerics (of which {stats['flagged_oos']} or {prop_flag:.2%} flagged as OOS).")

    keys = list(INVERSION_CATEGORY_MAPPING.values())
    if len(keys) == 2:
        s1,s2 = cat_stats.get(keys[0]), cat_stats.get(keys[1])
        if s1 and s2:
            if s1['initial']>0 and s2['initial']>0:
                pm1,pm2 = s1['missing_non_numeric']/s1['initial'], s2['missing_non_numeric']/s2['initial']
                if abs(pm1-pm2)>DATA_QUALITY_DISCREPANCY_THRESHOLD: logger.warning(f"DISCREPANCY MissingOrNonNumeric for '{column_name}': {keys[0]} {pm1:.2%}, {keys[1]} {pm2:.2%}.")
            if s1['numeric_for_analysis']>0 and s2['numeric_for_analysis']>0:
                pf1,pf2 = s1['flagged_oos']/s1['numeric_for_analysis'], s2['flagged_oos']/s2['numeric_for_analysis']
                if abs(pf1-pf2)>DATA_QUALITY_DISCREPANCY_THRESHOLD: logger.warning(f"DISCREPANCY FlaggedAsOutOfSpec for '{column_name}': {keys[0]} {pf1:.2%}, {keys[1]} {pf2:.2%}.")
    return cat_plot_data

def _plot_common_elements(ax, categorized_data, analysis_column_name, plot_type_specific_func):
    """Helper for common plot elements: data prep, titles, labels, scatter points."""
    plot_labels = list(INVERSION_CATEGORY_MAPPING.keys())
    plot_data = [categorized_data[INVERSION_CATEGORY_MAPPING[label]] for label in plot_labels]
    
    meta = {}
    metric_name = analysis_column_name.replace('_', ' ').title()
    total_pts = 0
    all_vals = []

    for i, label in enumerate(plot_labels):
        vals = plot_data[i]
        n = len(vals)
        total_pts += n
        meta[label] = {'median': np.median(vals) if n > 0 else np.nan, 'n_points': n}
        if n > 0: all_vals.extend(vals)
    
    logger.info(f"{plot_type_specific_func.__name__.split('_')[-1].capitalize()} for '{analysis_column_name}': " + 
                ", ".join([f"{lbl} N={meta[lbl]['n_points']}" for lbl in plot_labels]))

    positions = np.arange(len(plot_labels))
    err_style = {'ha':'center','va':'center','transform':ax.transAxes,'fontsize':12,'color':'red'}

    if total_pts > 0:
        valid_series, valid_pos, valid_colors = [], [], []
        cmap = {k: COLOR_PALETTE[i] for i,k in enumerate(INVERSION_CATEGORY_MAPPING.values())}
        for i, series in enumerate(plot_data):
            if series: # Only include if list is not empty
                valid_series.append(series)
                valid_pos.append(positions[i])
                valid_colors.append(cmap[INVERSION_CATEGORY_MAPPING[plot_labels[i]]])
        
        if not valid_series: # Should not happen if total_pts > 0, but as a safeguard
            logger.warning(f"No non-empty data series found for plotting {analysis_column_name}, though total_pts was {total_pts}.")
            ax.text(0.5, 0.5, f"No numeric data for categories\n(Column: {analysis_column_name})", **err_style)
            # Fall through to common styling, but plot elements won't be drawn
        else:
            plot_type_specific_func(ax, valid_series, valid_pos, valid_colors) # Call violin or boxplot drawing

            # Scatter points (common for both plot types)
            pt_clr, pt_alpha, pt_size = 'dimgray', 0.5, 15
            for i, orig_series in enumerate(plot_data): # Iterate original data to match positions
                if orig_series: # If there was data for this category
                    jitters = np.random.normal(0, 0.04, size=len(orig_series))
                    ax.scatter(positions[i] + jitters, orig_series, color=pt_clr, alpha=pt_alpha, s=pt_size, edgecolor='none', zorder=5)
            
            # Median annotations (common to both violin and box if medians are shown/relevant)
            if all_vals: # all_vals is not empty
                min_y, max_y = np.min(all_vals), np.max(all_vals)
                y_range = max_y - min_y if max_y > min_y else 0.1
                if y_range == 0: y_range = np.abs(all_vals[0] * 0.1) if all_vals[0] != 0 else 0.1
                
                for i, label in enumerate(plot_labels): # Iterate all original labels for positioning
                    m = meta[label] # Medians calculated on original (potentially empty) series
                    if m['n_points'] > 0 and not np.isnan(m['median']): # Check if median is valid
                        med_y, txt_off = m['median'], y_range * 0.02 if y_range > 0 else 0.005
                        ax.text(positions[i] + 0.12, med_y + txt_off, f"{med_y:.3f}", fontsize=9, color='black', ha='left', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'), zorder=15)
    else: # If total_pts is 0
        ax.text(0.5, 0.5, f"No numeric data for categories\n(Column: {analysis_column_name})", **err_style)

    # Axes, Title, Grid (common)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_title(f'Comparison of {metric_name}', fontsize=15, pad=18)
    xt_lbls_n = [f"{lbl}\n(N={meta[lbl]['n_points']})" for lbl in plot_labels]
    ax.set_xticks(positions); ax.set_xticklabels(xt_lbls_n, fontsize=11)
    ax.set_xlabel(""); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=10); ax.tick_params(axis='x', length=0)
    if all_vals: # all_vals is not empty before trying to use min/max
        min_v, max_v = np.min(all_vals), np.max(all_vals); dr = max_v - min_v
        pad = 0.1 if dr == 0 and min_v == 0 else (np.abs(min_v * 0.1) if dr == 0 else max(dr * 0.08, 0.005))
        ax.set_ylim(min_v - pad, max_v + pad * 1.5)
    else: ax.set_ylim(-0.02, 0.1) # Default if no data
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey'); ax.spines['left'].set_color('grey')
    ax.yaxis.grid(True, linestyle=':', which='major', color='lightgrey', alpha=0.7); ax.set_axisbelow(True)
    # fig is returned by the caller, _plot_common_elements just modifies ax

def _draw_violins(ax, valid_series, valid_pos, valid_colors):
    """Specific drawing function for violin plots."""
    v_parts = ax.violinplot(valid_series, positions=valid_pos, showmedians=True, showextrema=False, widths=0.75)
    for i, b in enumerate(v_parts['bodies']):
        b.set_facecolor(valid_colors[i]); b.set_edgecolor('darkgrey'); b.set_linewidth(0.8); b.set_alpha(0.4)
    v_parts['cmedians'].set_edgecolor('black'); v_parts['cmedians'].set_linewidth(1.5); v_parts['cmedians'].set_zorder(10)

def _draw_boxplots(ax, valid_series, valid_pos, valid_colors):
    """Specific drawing function for box plots."""
    bp = ax.boxplot(valid_series, positions=valid_pos, widths=0.6, patch_artist=True,
                    showfliers=False, # Outliers will be shown by scatter points
                    medianprops={'color':'black', 'linewidth':1.5},
                    boxprops={'edgecolor':'darkgrey'},
                    whiskerprops={'color':'darkgrey'},
                    capprops={'color':'darkgrey'})
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(valid_colors[i]); patch.set_alpha(0.5)


def create_comparison_plot(categorized_data, analysis_column_name, plot_type):
    global FST_TEST_SUMMARIES
    fig, ax = plt.subplots(figsize=(7, 7))

    output_filename = ""
    plot_specific_draw_func = None
    
    output_filename = ""
    plot_specific_draw_func = None

    if plot_type == 'violin':
        plot_specific_draw_func = _draw_violins
        output_filename = VIOLIN_PLOT_TEMPLATE.format(column_safe_name="".join(c if c.isalnum() else "_" for c in analysis_column_name).lower())
    elif plot_type == 'box':
        plot_specific_draw_func = _draw_boxplots
        output_filename = BOX_PLOT_TEMPLATE.format(column_safe_name="".join(c if c.isalnum() else "_" for c in analysis_column_name).lower())
    else:
        logger.error(f"Unknown plot type '{plot_type}' requested for {analysis_column_name}. Skipping plot.")
        plt.close(fig) # Close the unused figure
        return

    try:
        _plot_common_elements(ax, categorized_data, analysis_column_name, plot_specific_draw_func)

        # Statistical Test (Mann-Whitney U) - only for specified FST columns AND if it's a violin plot
        if analysis_column_name in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT and plot_type == 'violin':
            p_val_text = "Test N/A (logic error)" # Initial default for p-value text
            rec_key = INVERSION_CATEGORY_MAPPING.get('Recurrent')
            sing_key = INVERSION_CATEGORY_MAPPING.get('Single-event')

            # Initialize data and stats for summary storage
            data_r, data_s = [], [] # Default to empty lists
            recurrent_n, single_event_n = 0, 0
            recurrent_mean, recurrent_median = np.nan, np.nan
            single_event_mean, single_event_median = np.nan, np.nan

            if rec_key and sing_key:
                # Attempt to get data for recurrent and single-event categories
                data_r, data_s = categorized_data.get(rec_key, []), categorized_data.get(sing_key, [])
                recurrent_n = len(data_r)
                single_event_n = len(data_s)

                # Calculate mean and median if data is available
                if recurrent_n > 0:
                    recurrent_mean = np.mean(data_r)
                    recurrent_median = np.median(data_r)
                if single_event_n > 0:
                    single_event_mean = np.mean(data_s)
                    single_event_median = np.median(data_s)

                # Perform Mann-Whitney U test if both groups have data
                if data_r and data_s: # Check if lists are non-empty for MWU test
                    if np.var(data_r) == 0 and np.var(data_s) == 0 and np.mean(data_r) == np.mean(data_s): 
                        p_val_text = "p = 1.0 (Identical)"
                    else:
                        try:
                            stat, p_val = mannwhitneyu(data_r, data_s, alternative='two-sided')
                            p_val_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
                            logger.info(f"Mann-Whitney U for '{analysis_column_name}': W={stat:.1f}, {p_val_text} (N={recurrent_n} vs N={single_event_n})")
                        except ValueError as e: 
                            p_val_text = "Test Error"
                            logger.warning(f"MWU test failed for '{analysis_column_name}': {e}")
                else: # One or both groups empty, but category keys were valid
                    p_val_text = "Test N/A (groups empty)"
            else: # rec_key or sing_key (or both) were invalid
                p_val_text = "Test N/A (category key missing)"
            
            # Store summary statistics for this FST test in the global list
            current_fst_summary = {
                'column_name': analysis_column_name,
                'recurrent_N': recurrent_n,
                'recurrent_mean': recurrent_mean,
                'recurrent_median': recurrent_median,
                'single_event_N': single_event_n,
                'single_event_mean': single_event_mean,
                'single_event_median': single_event_median,
                'p_value_text': p_val_text # This captures the outcome of p-value calculation and any error/NA states.
            }
            FST_TEST_SUMMARIES.append(current_fst_summary)
            
            # Display p-value on the plot
            ax.text(0.04,0.96,f"Mann-Whitney U\n{p_val_text}",transform=ax.transAxes,fontsize=10,va='top',ha='left',
                    bbox=dict(boxstyle='round,pad=0.3',fc='ghostwhite',alpha=0.7,ec='lightgrey'))
        
        plt.tight_layout(pad=1.5)
        plt.savefig(output_filename, dpi=350, bbox_inches='tight')
        logger.info(f"Saved {plot_type} plot for '{analysis_column_name}' to '{output_filename}'")
        
        # Open (show) the plot only if it's one of the FST measures (which are violin plots)
        if analysis_column_name in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT and plot_type == 'violin':
            logger.info(f"Displaying FST plot for '{analysis_column_name}'. Close plot window to continue...")
            plt.show() # This is blocking. Script execution pauses here until the plot window is closed.
    
    except Exception as e:
        logger.error(f"Failed to create/save {plot_type} plot '{output_filename}' for '{analysis_column_name}': {e}", exc_info=True)

# --- Main Execution Block ---
def main():
    overall_start = time.time()
    logger.info(f"--- Starting Inversion Comparison Analysis ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
    logger.info(f"Summary Statistics File: '{SUMMARY_STATS_FILE}'")
    logger.info(f"Inversion Information File: '{INVERSION_FILE}'")
    logger.info(f"All columns processed for data quality: {ANALYSIS_COLUMNS}")
    logger.info(f"Plots will be generated ONLY for: {COLUMNS_FOR_PLOTTING}")
    logger.info(f"  - Violin plots (with test & display) for FST measures: {FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT}")
    logger.info(f"  - Box plots for other selected measures: {OTHER_COLUMNS_FOR_BOX_PLOT}")
    logger.info("NOTE: All numeric data is used for plots/tests; out-of-spec checks are for logging/awareness.")

    sum_cols_load = list(SUMMARY_STATS_COORDINATE_COLUMNS.values()) + \
                    [c for c in ANALYSIS_COLUMNS if c not in SUMMARY_STATS_COORDINATE_COLUMNS.values()]
    try:
        inv_df = pd.read_csv(INVERSION_FILE, usecols=INVERSION_FILE_COLUMNS)
        sum_df = pd.read_csv(SUMMARY_STATS_FILE, usecols=sum_cols_load)
    except FileNotFoundError as e: logger.critical(f"CRITICAL: Input file not found. {e}"); sys.exit(1)
    except ValueError as e: logger.critical(f"CRITICAL: Column not found in input. {e}"); sys.exit(1)
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

    for current_col in ANALYSIS_COLUMNS:
        col_start_time = time.time()
        logger.info(f"===== Assessing Data Quality for Column: '{current_col}' =====")
        
        try:
            categorized_data = prepare_data_for_analysis(sum_df, current_col)
        except Exception as e: 
            logger.error(f"SKIPPING Col '{current_col}': Error during data prep: {e}", exc_info=True)
            continue 

        if current_col in COLUMNS_FOR_PLOTTING:
            plot_type_to_use = 'violin' if current_col in FST_COLUMNS_FOR_TEST_AND_VIOLIN_PLOT else 'box'
            logger.info(f"--- Generating {plot_type_to_use.upper()} Plot for: '{current_col}' ---")
            
            if not any(categorized_data[cat_key] for cat_key in INVERSION_CATEGORY_MAPPING.values()):
                logger.warning(f"No numeric data for plotting for '{current_col}'. Skipping plot.")
            else:
                create_comparison_plot(categorized_data, current_col, plot_type_to_use)
        else:
            logger.info(f"Skipping plot generation for '{current_col}' (not in designated plot list).")

        logger.info(f"--- Completed assessment for '{current_col}' in {time.time()-col_start_time:.2f}s ---")

    final_sup_summary = global_warning_tracker.get_suppressed_summary()
    if final_sup_summary: logger.info(f"\n--- Summary of Suppressed Warnings ---\n{final_sup_summary}")

    # Print summary of FST test statistics collected during the analysis
    if FST_TEST_SUMMARIES:
        logger.info("\n====== Summary Statistics for FST Tests (Recurrent vs Single-event) ======")
        for summary in FST_TEST_SUMMARIES:
            logger.info(f"  --- FST Metric: {summary['column_name']} ---")
            # Format NaN values as 'nan' which is standard; use .4f for float precision.
            logger.info(f"    Recurrent:     N={summary['recurrent_N']}, Mean={summary['recurrent_mean']:.4f}, Median={summary['recurrent_median']:.4f}")
            logger.info(f"    Single-event:  N={summary['single_event_N']}, Mean={summary['single_event_mean']:.4f}, Median={summary['single_event_median']:.4f}")
            logger.info(f"    Mann-Whitney U: {summary['p_value_text']}") # p_value_text already contains formatted string or error message
    else:
        logger.info("\n====== No FST Test Summary Statistics to display (no FST tests were run or no data was available for them) ======")
    
    logger.info(f"====== Full Analysis Script completed in {time.time()-overall_start:.2f}s ======")

if __name__ == "__main__":
    main()
