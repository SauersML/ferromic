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
SUMMARY_FST_FILE = 'output.csv'
INVERSION_FILE = 'inv_info.csv'

# Output File Template
OUTPUT_PLOT_TEMPLATE = 'fst_recurrent_vs_single_event_violin_{fst_col_safe_name}.png'

# FST Column Selection: List of FST columns to analyze
FST_COLUMN_NAMES = ['haplotype_overall_fst_wc', 'hudson_fst_hap_group_0v1']

# Plotting Categories and Colors
CAT_MAPPING = {
    'Recurrent': 'recurrent',
    'Single-event': 'single_event'
}
# Using a visually distinct and accessible color palette
COLOR_PALETTE = sns.color_palette("Set2", n_colors=len(CAT_MAPPING))

# Text for Y-axis Description Detail (common part of the Y-axis label)
Y_AXIS_DETAIL_DESCRIPTION = "Between Haplotype Orientation Groups"


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        # logging.FileHandler("analysis.log"), # Optional: Log to file
        logging.StreamHandler(sys.stdout) # Log to console
    ]
)
logger = logging.getLogger('fst_analysis')

# --- Data Processing Functions ---

def normalize_chromosome(chrom):
    """Converts chromosome names to a consistent 'chr...' format."""
    chrom = str(chrom).strip().lower()
    if chrom.startswith('chr_'):
        chrom = chrom[4:]
    elif chrom.startswith('chr'):
        chrom = chrom[3:]
    # Handle cases like 'x' or 'y'
    if not chrom.startswith('chr'):
        chrom = f"chr{chrom}"
    return chrom

def map_regions_to_inversions(inversion_df):
    """Parses the inversion info file to map coordinates to inversion types."""
    recurrent_regions = {}
    single_event_regions = {}
    required_cols = ['Chromosome', 'Start', 'End', '0_single_1_recur']

    # Validate input DataFrame structure
    if not all(col in inversion_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in inversion_df.columns]
        raise ValueError(f"Inversion file '{INVERSION_FILE}' is missing required columns: {missing}")

    logger.info(f"Mapping inversion types from {len(inversion_df)} entries in '{INVERSION_FILE}'...")
    parsed_count = 0
    skipped_count = 0
    for index, row in inversion_df.iterrows():
        chrom = normalize_chromosome(row['Chromosome'])
        try:
            # Convert coordinates and category to integers
            start = int(row['Start'])
            end = int(row['End'])
            category = int(row['0_single_1_recur'])
            if start > end: # Basic coordinate sanity check
                raise ValueError("Start coordinate cannot be greater than End coordinate")
            parsed_count += 1
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping row {index+2} in '{INVERSION_FILE}' due to invalid data: {e} | Row data: {row.to_dict()}")
            skipped_count += 1
            continue

        # Assign coordinates to the appropriate dictionary based on category
        if category == 1:
            recurrent_regions.setdefault(chrom, []).append((start, end))
        elif category == 0:
            single_event_regions.setdefault(chrom, []).append((start, end))
        else:
            logger.warning(f"Skipping row {index+2} in '{INVERSION_FILE}' due to unrecognized category value '{category}'. Expected 0 or 1.")
            skipped_count += 1

    logger.info(f"Successfully mapped {parsed_count} inversion entries. Skipped {skipped_count} invalid entries.")
    if parsed_count == 0:
        logger.warning(f"No valid inversion entries were mapped from '{INVERSION_FILE}'.")
    return recurrent_regions, single_event_regions

def is_overlapping(region1_coords, region2_inv_coords):
    """Checks if an FST region's coordinates closely match an inversion's coordinates (allowing 1bp tolerance)."""
    _, start1, end1 = region1_coords
    start2, end2 = region2_inv_coords
    # Check if start and end coordinates are within 1bp of each other
    start_match = abs(start1 - start2) <= 1
    end_match = abs(end1 - end2) <= 1
    return start_match and end_match

def determine_inversion_type(fst_chrom, fst_start, fst_end, recurrent_regions, single_event_regions):
    """Assigns an inversion type ('recurrent', 'single_event', 'ambiguous', 'unknown') to an FST region based on coordinate matching."""
    rec_map = recurrent_regions.get(fst_chrom, [])
    sing_map = single_event_regions.get(fst_chrom, [])
    fst_coords_tuple = (fst_chrom, fst_start, fst_end)

    # Check for near-exact matches in both recurrent and single-event lists
    matches_recurrent = any(is_overlapping(fst_coords_tuple, r_coords) for r_coords in rec_map)
    matches_single = any(is_overlapping(fst_coords_tuple, s_coords) for s_coords in sing_map)

    # Determine category based on matches
    if matches_recurrent and not matches_single:
        return 'recurrent'
    elif matches_single and not matches_recurrent:
        return 'single_event'
    elif matches_recurrent and matches_single:
        logger.debug(f"Ambiguous coordinate match for {fst_chrom}:{fst_start}-{fst_end}. Matches both types.")
        return 'ambiguous'
    else:
        return 'unknown'

def categorize_fst_from_summary(df_fst, fst_col_name, recurrent_regions, single_event_regions):
    """Categorizes FST regions based on inversion type and collects their FST values for a specific FST column."""
    categories_data = {cat_key: [] for cat_key in CAT_MAPPING.values()}
    required_coords = ['chr', 'region_start', 'region_end']

    # --- Input Validation ---
    if not all(col in df_fst.columns for col in required_coords):
        missing = [col for col in required_coords if col not in df_fst.columns]
        raise ValueError(f"FST summary file '{SUMMARY_FST_FILE}' is missing coordinate columns: {missing}")
    if fst_col_name not in df_fst.columns:
        raise ValueError(f"Selected FST column '{fst_col_name}' not found in '{SUMMARY_FST_FILE}'. Available columns: {df_fst.columns.tolist()}")

    # --- Data Cleaning ---
    df_work = df_fst.copy()
    df_work[fst_col_name] = pd.to_numeric(df_work[fst_col_name], errors='coerce')
    initial_rows = len(df_work)
    df_work.dropna(subset=[fst_col_name], inplace=True)
    valid_fst_count = len(df_work)
    logger.info(f"Using {valid_fst_count} rows with valid numeric FST in column '{fst_col_name}' (out of {initial_rows} total rows).")
    if valid_fst_count == 0:
        logger.warning(f"No valid data to categorize in column '{fst_col_name}'. Plot may be empty.")
        return categories_data

    # --- Categorization Loop ---
    processed_count = 0
    matched_counts = {'recurrent': 0, 'single_event': 0, 'ambiguous': 0, 'unknown': 0}
    skipped_coord_errors = 0

    for _, row in df_work.iterrows():
        try:
            chrom = normalize_chromosome(row['chr'])
            start = int(row['region_start'])
            end = int(row['region_end'])
            fst_value = row[fst_col_name]
            if start > end:
                raise ValueError("Start coordinate cannot be greater than End coordinate")
            processed_count += 1
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping row due to invalid coordinate for FST column '{fst_col_name}': {e} | Row: {row.to_dict()}")
            skipped_coord_errors += 1
            continue

        inv_type = determine_inversion_type(chrom, start, end, recurrent_regions, single_event_regions)
        matched_counts[inv_type] += 1

        if inv_type in categories_data:
            categories_data[inv_type].append(fst_value)

    # --- Logging Summary ---
    logger.info(f"For FST column '{fst_col_name}': Processed {processed_count} regions with valid FST and coordinates.")
    if skipped_coord_errors > 0:
        logger.warning(f"For FST column '{fst_col_name}': Skipped {skipped_coord_errors} rows due to invalid coordinate formats.")
    logger.info(f"For FST column '{fst_col_name}': Regions matched to inversion types: Recurrent={matched_counts['recurrent']}, Single-event={matched_counts['single_event']}, Ambiguous={matched_counts['ambiguous']}, Unmatched={matched_counts['unknown']}")
    if matched_counts['recurrent'] == 0 and matched_counts['single_event'] == 0:
        logger.warning(f"For FST column '{fst_col_name}': No FST regions exactly matched the coordinates of 'Recurrent' or 'Single-event' inversions. Check coordinate formats and the `is_overlapping` function.")

    return categories_data

# --- Plotting Function ---

def create_violin_plot_final(categories_data, current_fst_col_name, y_axis_detail, output_filename):
    """Creates a publication-ready violin plot for a given FST column."""

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_labels = list(CAT_MAPPING.keys())
    plot_data = [categories_data[CAT_MAPPING[label]] for label in plot_labels]
    category_stats = {}
    formatted_metric_name = current_fst_col_name.replace('_', ' ').title()

    total_valid_points = 0
    all_plot_values = []
    for i, label in enumerate(plot_labels):
        values = plot_data[i]
        n_cat = len(values)
        total_valid_points += n_cat
        if n_cat > 0:
            median_val = np.median(values)
            category_stats[label] = {'median': median_val, 'n': n_cat}
            all_plot_values.extend(values)
            logger.info(f"Plotting for '{current_fst_col_name}' - {label}: N = {n_cat}, Median FST = {median_val:.4f}")
        else:
            category_stats[label] = {'median': np.nan, 'n': 0}
            logger.info(f"Plotting for '{current_fst_col_name}' - {label}: N = 0")

    plot_positions = np.arange(len(plot_labels))
    error_text_style = {'horizontalalignment': 'center', 'verticalalignment': 'center', 'transform': ax.transAxes, 'fontsize': 12, 'color': 'red'}

    if total_valid_points > 0:
        try:
            violin_parts = ax.violinplot(
                plot_data,
                positions=plot_positions,
                showmedians=True,
                showextrema=False,
                widths=0.75
            )
            for i, body in enumerate(violin_parts['bodies']):
                body.set_facecolor(COLOR_PALETTE[i])
                body.set_edgecolor('darkgrey')
                body.set_linewidth(0.8)
                body.set_alpha(0.25)
            violin_parts['cmedians'].set_edgecolor('black')
            violin_parts['cmedians'].set_linewidth(1.5)
            violin_parts['cmedians'].set_zorder(10)
        except Exception as e:
            logger.error(f"Error during violin plot creation for '{current_fst_col_name}': {e}")
            ax.text(0.5, 0.6, "Error creating violins", **error_text_style)

        point_color = 'dimgray'
        point_alpha = 0.5
        point_size = 15
        for i, values in enumerate(plot_data):
            if values:
                jitter = np.random.normal(0, 0.04, size=len(values))
                ax.scatter(plot_positions[i] + jitter, values,
                           color=point_color, alpha=point_alpha, s=point_size,
                           edgecolor='none', zorder=5)

        if 'violin_parts' in locals():
            min_y_plot, max_y_plot = ax.get_ylim() if ax.get_ylim() and ax.get_ylim()[1] > ax.get_ylim()[0] else (np.min(all_plot_values) if all_plot_values else 0, np.max(all_plot_values) if all_plot_values else 0.1)
            y_range_plot = max_y_plot - min_y_plot if max_y_plot > min_y_plot else 0.1

            for i, label in enumerate(plot_labels):
                stats = category_stats[label]
                if stats['n'] > 0 and not np.isnan(stats['median']):
                    median_y = stats['median']
                    text_y_offset = y_range_plot * 0.015
                    ax.text(plot_positions[i] + 0.1, median_y + text_y_offset,
                            f"{median_y:.3f}", fontsize=9, color='black',
                            ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'),
                            zorder=15)
    else:
        ax.text(0.5, 0.5, f"No valid FST data found for categories\n(Column: {current_fst_col_name})", **error_text_style)

    recurrent_fst = categories_data[CAT_MAPPING['Recurrent']]
    single_event_fst = categories_data[CAT_MAPPING['Single-event']]
    p_value_text = "Test N/A"

    if recurrent_fst and single_event_fst:
        try:
            stat, p_value = mannwhitneyu(recurrent_fst, single_event_fst, alternative='two-sided')
            if p_value < 0.001:
                p_value_text = f"p < 0.001"
            else:
                p_value_text = f"p = {p_value:.3f}"
            logger.info(f"Mann-Whitney U test for '{current_fst_col_name}': W={stat:.1f}, {p_value_text}")
        except ValueError as e:
            p_value_text = "Test Error"
            logger.warning(f"Mann-Whitney U test failed for '{current_fst_col_name}': {e}")
    else:
        logger.info(f"Skipping Mann-Whitney U test for '{current_fst_col_name}': one or both categories have no data.")

    ax.text(0.05, 0.95, f"Mann-Whitney U\n{p_value_text}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='ghostwhite', alpha=0.7, ec='lightgrey'))

    ax.set_ylabel(f"{formatted_metric_name} ({y_axis_detail})", fontsize=13)
    ax.set_title(f'Comparison of {formatted_metric_name} Values', fontsize=15, pad=18)

    x_labels_with_n = [f"{label}\n(N = {category_stats[label]['n']})" for label in plot_labels]
    ax.set_xticks(plot_positions)
    ax.set_xticklabels(x_labels_with_n, fontsize=11)
    ax.set_xlabel("")

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', length=0)

    if all_plot_values:
        min_val = np.min(all_plot_values)
        max_val = np.max(all_plot_values)
        data_range = max_val - min_val
        padding = max(data_range * 0.08, 0.005 if data_range > 0 else 0.01)
        lower_limit = min_val - padding
        upper_limit = max_val + padding * 2.0
        ax.set_ylim(lower_limit, upper_limit)
    else:
        ax.set_ylim(-0.02, 0.1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.yaxis.grid(True, linestyle=':', which='major', color='lightgrey', alpha=0.7)
    ax.set_axisbelow(True)

    try:
        plt.tight_layout(pad=1.5)
        plt.savefig(output_filename, dpi=350, bbox_inches='tight')
        logger.info(f"Saved final violin plot for '{current_fst_col_name}' to '{output_filename}'")
    except Exception as e:
        logger.error(f"Failed to save plot for '{current_fst_col_name}' ('{output_filename}'): {e}")
    plt.close(fig) # Close the figure to free memory

# --- Main Execution Block ---

def main():
    """Loads data, performs categorization, and generates plots for specified FST columns."""
    overall_start_time = time.time()
    logger.info(f"--- Starting FST Analysis Script ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
    logger.info(f"Using FST summary: '{SUMMARY_FST_FILE}'")
    logger.info(f"Using Inversion info: '{INVERSION_FILE}'")
    logger.info(f"Target FST columns for analysis: {FST_COLUMN_NAMES}")

    # --- Load Data (once) ---
    try:
        inversion_df = pd.read_csv(INVERSION_FILE)
        df_fst_summary_full = pd.read_csv(SUMMARY_FST_FILE)
    except FileNotFoundError as e:
        logger.critical(f"ERROR: Input file not found. {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"ERROR reading input files: {e}")
        sys.exit(1)

    # --- Process Inversion Data (once) ---
    try:
        recurrent_regions, single_event_regions = map_regions_to_inversions(inversion_df)
    except ValueError as e:
        logger.critical(f"ERROR during inversion data processing: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"ERROR: An unexpected error occurred during inversion processing: {e}", exc_info=True)
        sys.exit(1)

    # --- Loop through each FST column for analysis and plotting ---
    for fst_column_name in FST_COLUMN_NAMES:
        loop_start_time = time.time()
        logger.info(f"--- Processing FST Column: '{fst_column_name}' ---")

        safe_fst_col_for_filename = "".join(c if c.isalnum() else "_" for c in fst_column_name).lower()
        current_output_plot_filename = OUTPUT_PLOT_TEMPLATE.format(fst_col_safe_name=safe_fst_col_for_filename)

        # --- Categorize FST data for the current column ---
        try:
            categories_data = categorize_fst_from_summary(
                df_fst_summary_full,
                fst_column_name,
                recurrent_regions,
                single_event_regions
            )
        except ValueError as e:
            logger.error(f"SKIPPING FST Column '{fst_column_name}': Error during data categorization: {e}")
            continue
        except Exception as e:
            logger.error(f"SKIPPING FST Column '{fst_column_name}': An unexpected error during categorization: {e}", exc_info=True)
            continue

        # --- Log Overall Statistics for the current FST column ---
        all_categorized_values = [val for cat_vals in categories_data.values() for val in cat_vals]
        if all_categorized_values:
            mean_all = np.mean(all_categorized_values)
            median_all = np.median(all_categorized_values)
            logger.info(f"Overall for '{fst_column_name}' (matched/categorized regions): N = {len(all_categorized_values)}, Mean FST = {mean_all:.4f}, Median FST = {median_all:.4f}")
        else:
            logger.warning(f"For FST column '{fst_column_name}', no regions were successfully categorized as 'Recurrent' or 'Single-event'. Cannot calculate overall stats for this column.")

        # --- Generate Plot for the current FST column ---
        try:
            create_violin_plot_final(
                categories_data,
                fst_column_name,
                Y_AXIS_DETAIL_DESCRIPTION,
                current_output_plot_filename
            )
        except Exception as e:
            logger.error(f"ERROR creating plot for FST column '{fst_column_name}': {e}", exc_info=True)
            # Continue to the next FST column even if plotting fails for one

        loop_elapsed = time.time() - loop_start_time
        logger.info(f"--- Completed processing for FST column '{fst_column_name}' in {loop_elapsed:.2f} seconds ---")

    overall_elapsed = time.time() - overall_start_time
    logger.info(f"--- Full Analysis Script completed in {overall_elapsed:.2f} seconds ---")

if __name__ == "__main__":
    main()
