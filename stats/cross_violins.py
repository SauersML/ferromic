import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging
import sys
from pathlib import Path

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('inversion_omega_analysis')

# --- File Paths & Constants ---
PAIRWISE_FILE = Path('all_pairwise_results.csv')
INVERSION_FILE = Path('inv_info.tsv')
OUTPUT_PLOT_PATH = Path('inversion_omega_analysis_plot_median_only.png')

# --- Plotting Style ---
EVENT_TYPE_PALETTE = {
    "Single-Event": "skyblue",
    "Recurrent": "salmon"
}
CATEGORY_ORDER = ["Direct", "Inverted", "Cross-Group"]


def load_and_prepare_data():
    """
    Loads pairwise and inversion data, then processes it for both single-event
    and recurrent inversion types.

    This function performs a stratified analysis:
    1.  Loads all data and assigns a base 'ComparisonGroup' to every pair.
    2.  Iterates through inversion types (Single-Event, Recurrent).
    3.  For each type, it identifies the relevant genes within those regions.
    4.  Filters the main dataset for those genes, preserving their comparison group.
    5.  Applies a strict filter for valid omega values (must be finite and positive).
    6.  Combines the processed data from all event types into a single DataFrame.

    Returns:
        A pandas DataFrame ready for plotting, or None if critical data is missing.
    """
    logger.info("--- Starting Data Loading and Preparation ---")

    try:
        pairwise_df = pd.read_csv(PAIRWISE_FILE)
        inversion_df = pd.read_csv(INVERSION_FILE, sep='\t')
        logger.info(f"Loaded {len(pairwise_df)} pairwise results and {len(inversion_df)} inversion records.")
    except FileNotFoundError:
        logger.error(f"Input file not found. Ensure '{PAIRWISE_FILE}' and '{INVERSION_FILE}' exist.")
        return None

    inversion_df.rename(columns={'0_single_1_recur': 'InversionClass'}, inplace=True)
    inversion_df['Chromosome'] = inversion_df['Chromosome'].astype(str).str.lower()
    for col in ['Start', 'End', 'InversionClass']:
        inversion_df[col] = pd.to_numeric(inversion_df[col], errors='coerce')
    inversion_df.dropna(subset=['Start', 'End', 'InversionClass', 'Chromosome'], inplace=True)

    def categorize_comparison(row):
        g1, g2 = row['Group1'], row['Group2']
        if g1 == 0 and g2 == 0: return "Direct"
        if g1 == 1 and g2 == 1: return "Inverted"
        if g1 != g2: return "Cross-Group"
        return None

    pairwise_df['ComparisonGroup'] = pairwise_df.apply(categorize_comparison, axis=1)
    pairwise_df.dropna(subset=['ComparisonGroup'], inplace=True)

    processed_dfs = []
    event_types = {'Single-Event': 0, 'Recurrent': 1}

    for event_name, event_class in event_types.items():
        logger.info(f"--- Processing: {event_name} Inversions (Class {event_class}) ---")

        event_inv_df = inversion_df[inversion_df['InversionClass'] == event_class]
        if event_inv_df.empty:
            logger.warning(f"No regions found for event type '{event_name}'. Skipping.")
            continue

        cds_in_region = set()
        for _, inv_row in event_inv_df.iterrows():
            chrom, start, end = inv_row['Chromosome'], inv_row['Start'], inv_row['End']
            for cds_string in pairwise_df['CDS'].unique():
                match = re.search(r'(chr[\w\.]+)_start(\d+)_end(\d+)', str(cds_string), re.I)
                if match:
                    cds_chrom, cds_start, cds_end = match.group(1).lower(), int(match.group(2)), int(match.group(3))
                    if cds_chrom == chrom and (start - 1 < cds_end) and (end + 1 > cds_start):
                        cds_in_region.add(cds_string)

        if not cds_in_region:
            logger.warning(f"No gene CDS mapped to '{event_name}' regions. Skipping.")
            continue
        logger.info(f"Mapped {len(cds_in_region)} unique CDS to {event_name} regions.")

        analysis_df = pairwise_df[pairwise_df['CDS'].isin(cds_in_region)].copy()
        analysis_df['EventType'] = event_name

        analysis_df['omega'] = pd.to_numeric(analysis_df['omega'], errors='coerce')
        analysis_df.dropna(subset=['omega'], inplace=True)
        valid_omega_df = analysis_df[(analysis_df['omega'] > 0) & (analysis_df['omega'] != 99)].copy()

        if 'Inverted' not in valid_omega_df['ComparisonGroup'].unique():
            initial_inverted_count = len(analysis_df[analysis_df['ComparisonGroup'] == 'Inverted'])
            if initial_inverted_count > 0:
                logger.warning(f"For '{event_name}', all {initial_inverted_count} 'Inverted' pairs were filtered out due to invalid omega values.")

        # --- Analysis of CDS with median dN/dS > 1 ---
        logger.info(f"--- Analysis of CDS with Median dN/dS > 1 for {event_name} ---")
        if not valid_omega_df.empty:
            # For each unique CDS within each comparison group, calculate the median omega across its pairwise comparisons.
            median_omega_per_cds = valid_omega_df.groupby(['CDS', 'ComparisonGroup'])['omega'].median()

            # Filter for CDS-group combinations where the median omega is > 1, indicating positive selection pressure.
            positively_selected_cds_groups = median_omega_per_cds[median_omega_per_cds > 1]

            if not positively_selected_cds_groups.empty:
                unique_selected_cds = positively_selected_cds_groups.index.get_level_values('CDS').unique()
                logger.info(f"Found {len(unique_selected_cds)} total unique CDS with median dN/dS > 1.")

                # Provide a detailed breakdown by comparison group, listing the actual CDS names.
                logger.info("Detailed breakdown by comparison group, with CDS names:")
                
                # Convert the result Series to a DataFrame for easier iteration.
                detailed_selection_df = positively_selected_cds_groups.reset_index()

                for group_name, group_df in detailed_selection_df.groupby('ComparisonGroup'):
                    # For each group, list the CDS names that met the criterion.
                    cds_list = group_df['CDS'].tolist()
                    logger.info(f"  - {group_name} ({len(cds_list)} CDS): {', '.join(cds_list)}")
            else:
                logger.info("No CDS found with median dN/dS > 1 for any category.")
        else:
            logger.info("Skipping analysis as no valid omega data is available.")

        logger.info(f"Final valid pairwise counts for {event_name} plotting: \n{valid_omega_df['ComparisonGroup'].value_counts().to_string()}")
        processed_dfs.append(valid_omega_df)

    if not processed_dfs:
        logger.error("No data available for plotting after processing all event types.")
        return None

    return pd.concat(processed_dfs, ignore_index=True)


def create_and_save_plot(data_df: pd.DataFrame, output_path: Path):
    """
    Generates and saves a grouped violin plot showing only the median line.
    """
    logger.info("--- Generating Final Plot ---")
    if data_df.empty:
        logger.warning("Input DataFrame for plotting is empty. Skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.violinplot(
        x='ComparisonGroup',
        y='omega',
        hue='EventType',
        data=data_df,
        order=CATEGORY_ORDER,
        palette=EVENT_TYPE_PALETTE,
        split=True,
        ax=ax,
        scale='width',
        inner='quartile'  # Draw quartiles initially, we will modify them next.
    )

    # --- Post-processing to show only the median line ---
    # We iterate through the lines drawn by the `inner='quartile'` option.
    # They are drawn in sets of 3 (Q1, Median, Q3) for each violin half.
    for i, line in enumerate(ax.lines):
        # The line with index 1 in each set of 3 is the median.
        if i % 3 == 1:
            line.set_linestyle('--')
            line.set_color('black')
            line.set_linewidth(1.5)
            line.set_alpha(0.8)  # Make it prominent but not overpowering
        # The other lines (Q1 and Q3) are made invisible.
        else:
            line.set_visible(False)

    # --- Plot Aesthetics and Labels ---
    if not data_df.empty:
        upper_limit = data_df['omega'].quantile(0.99)
        if upper_limit < 5: upper_limit = 5
        ax.set_ylim(0, upper_limit)
        logger.info(f"Y-axis capped at {upper_limit:.2f} (99th percentile) for visual clarity.")

    ax.set_title('dN/dS Distribution by Inversion Event Type', fontsize=18, pad=20)
    ax.set_xlabel('Comparison Type', fontsize=14, labelpad=15)
    ax.set_ylabel('Omega (dN/dS)', fontsize=14, labelpad=15)
    ax.tick_params(axis='y', labelsize=13)

    counts = data_df.groupby(['ComparisonGroup', 'EventType']).size().unstack(fill_value=0)
    new_xticklabels = []
    for cat in CATEGORY_ORDER:
        label = f"{cat}\n"
        s_count = counts.at[cat, 'Single-Event'] if cat in counts.index and 'Single-Event' in counts.columns else 0
        r_count = counts.at[cat, 'Recurrent'] if cat in counts.index and 'Recurrent' in counts.columns else 0
        label += f"Single: {s_count}\nRecurrent: {r_count}"
        new_xticklabels.append(label)
    ax.set_xticklabels(new_xticklabels, fontsize=13)

    ax.legend(title='Inversion Type', loc='upper right', fontsize=12, title_fontsize=14)
    sns.despine(ax=ax)
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Successfully saved the plot to '{output_path}'.")
    except Exception as e:
        logger.error(f"Failed to save the plot: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    plot_data = load_and_prepare_data()

    if plot_data is not None and not plot_data.empty:
        create_and_save_plot(plot_data, OUTPUT_PLOT_PATH)
    else:
        logger.error("Analysis concluded without any data to plot.")

    logger.info("--- Analysis Finished ---")
