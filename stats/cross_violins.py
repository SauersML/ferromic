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
INVERSION_FILE = Path('inv_info.csv')
OUTPUT_PLOT_PATH = Path('inversion_omega_analysis_plot.png')

# --- Plotting Style ---
plt.rcParams.update({'font.size': 14})
# Palette now distinguishes between the event types
EVENT_TYPE_PALETTE = {
    "Single-Event": "skyblue",
    "Recurrent": "salmon"
}
# Define the order for the plot categories
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
        A pandas DataFrame ready for plotting, or None if  data is missing.
    """
    logger.info("--- Starting Data Loading and Preparation ---")

    # --- 1. Load and Basic Cleaning ---
    try:
        pairwise_df = pd.read_csv(PAIRWISE_FILE)
        inversion_df = pd.read_csv(INVERSION_FILE)
        logger.info(f"Loaded {len(pairwise_df)} pairwise results and {len(inversion_df)} inversion records.")
    except FileNotFoundError:
        logger.error(f"Input file not found. Ensure '{PAIRWISE_FILE}' and '{INVERSION_FILE}' exist.")
        return None

    # Clean inversion data
    inversion_df.rename(columns={'0_single_1_recur': 'InversionClass'}, inplace=True)
    inversion_df['Chromosome'] = inversion_df['Chromosome'].astype(str).str.lower()
    for col in ['Start', 'End', 'InversionClass']:
        inversion_df[col] = pd.to_numeric(inversion_df[col], errors='coerce')
    inversion_df.dropna(subset=['Start', 'End', 'InversionClass', 'Chromosome'], inplace=True)

    # --- 2. Pre-categorize all pairs ---
    def categorize_comparison(row):
        g1, g2 = row['Group1'], row['Group2']
        if g1 == 0 and g2 == 0: return "Direct"
        if g1 == 1 and g2 == 1: return "Inverted"
        if g1 != g2: return "Cross-Group"
        return None

    pairwise_df['ComparisonGroup'] = pairwise_df.apply(categorize_comparison, axis=1)
    pairwise_df.dropna(subset=['ComparisonGroup'], inplace=True)

    # --- 3. Process each event type separately ---
    processed_dfs = []
    event_types = {'Single-Event': 0, 'Recurrent': 1}

    for event_name, event_class in event_types.items():
        logger.info(f"--- Processing: {event_name} Inversions (Class {event_class}) ---")

        # Isolate inversion regions for the current type
        event_inv_df = inversion_df[inversion_df['InversionClass'] == event_class]
        if event_inv_df.empty:
            logger.warning(f"No regions found for event type '{event_name}'. Skipping.")
            continue

        # Find all unique CDS strings within these regions with a 1bp tolerance
        cds_in_region = set()
        for _, inv_row in event_inv_df.iterrows():
            chrom, start, end = inv_row['Chromosome'], inv_row['Start'], inv_row['End']
            # This is slow, but robust. A better way would be an interval tree for massive datasets.
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

        # Filter the main dataframe for these genes
        analysis_df = pairwise_df[pairwise_df['CDS'].isin(cds_in_region)].copy()
        analysis_df['EventType'] = event_name

        logger.info(f"Initial counts for {event_name} region genes: \n{analysis_df['ComparisonGroup'].value_counts().to_string()}")

        # Filter for valid, plottable omega values
        analysis_df['omega'] = pd.to_numeric(analysis_df['omega'], errors='coerce')
        analysis_df.dropna(subset=['omega'], inplace=True)
        valid_omega_df = analysis_df[(analysis_df['omega'] > 0) & (analysis_df['omega'] != 99)].copy()

        # Report on the outcome, especially for the 'Inverted' group
        if 'Inverted' not in valid_omega_df['ComparisonGroup'].unique():
            initial_inverted_count = len(analysis_df[analysis_df['ComparisonGroup'] == 'Inverted'])
            if initial_inverted_count > 0:
                logger.warning(f"For '{event_name}', all {initial_inverted_count} 'Inverted' pairs were filtered out due to invalid omega values (e.g., -1, 99, NaN).")
            else:
                 logger.info(f"For '{event_name}', no 'Inverted' pairs were found in the regions to begin with.")

        logger.info(f"Final valid counts for {event_name} plotting: \n{valid_omega_df['ComparisonGroup'].value_counts().to_string()}")
        processed_dfs.append(valid_omega_df)

    if not processed_dfs:
        logger.error("No data available for plotting after processing all event types.")
        return None

    return pd.concat(processed_dfs, ignore_index=True)


def create_and_save_plot(data_df: pd.DataFrame, output_path: Path):
    """
    Generates and saves a grouped violin plot of omega distributions.
    """
    logger.info("--- Generating Final Plot ---")
    if data_df.empty:
        logger.warning("Input DataFrame for plotting is empty. Skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 1: Draw the split violins with NO inner elements.
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
        inner=None  # This removes all default inner lines
    )

    # Step 2: Manually overlay a plot that ONLY shows the median.
    # We use a pointplot configured to calculate the median and display it as a line.
    sns.pointplot(
        x='ComparisonGroup',
        y='omega',
        hue='EventType',
        data=data_df,
        order=CATEGORY_ORDER,
        estimator=np.median,   # : Sets the calculation to median
        errorbar=None,         # : Removes the confidence interval lines
        marker='_',            # Use a short horizontal line for the median marker
        markersize=15,         # Make the median line wide enough to be visible
        linestyle='none',      # Do not draw lines connecting the medians
        dodge=0.53,            # Align the median with the center of each split violin
        ax=ax,
        palette=['black']      # Make the median line black for both hues
    )
    # --- Plot Aesthetics and Labels ---
    # Cap the y-axis to prevent extreme outliers from squishing the plot
    if not data_df.empty:
        # Calculate a reasonable upper limit, e.g., the 99th percentile
        upper_limit = data_df['omega'].quantile(0.99)
        if upper_limit < 5: upper_limit = 5 # Set a minimum reasonable cap
        ax.set_ylim(0, upper_limit)
        logger.info(f"Y-axis capped at {upper_limit:.2f} (99th percentile) for visual clarity.")

    ax.set_title('dN/dS Distribution by Inversion Event Type', fontsize=18, pad=20)
    ax.set_xlabel('Comparison Type', fontsize=14, labelpad=15)
    ax.set_ylabel('Omega (dN/dS)', fontsize=14, labelpad=15)

    # Add detailed N-counts to x-tick labels
    counts = data_df.groupby(['ComparisonGroup', 'EventType']).size().unstack(fill_value=0)
    new_xticklabels = []
    for cat in CATEGORY_ORDER:
        if cat in counts.index:
            s_count = counts.at[cat, 'Single-Event'] if 'Single-Event' in counts.columns else 0
            r_count = counts.at[cat, 'Recurrent'] if 'Recurrent' in counts.columns else 0
            label = f"{cat}\nSingle: {s_count}\nRecurrent: {r_count}"
        else:
            label = f"{cat}\n(N=0)"
        new_xticklabels.append(label)
    ax.set_xticklabels(new_xticklabels)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Inversion Type', loc='upper right')
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
