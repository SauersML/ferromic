"""
Performs an analysis of dN/dS (omega) ratios for genes within
chromosomal inversions, comparing different orientations and recurrence types.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
from scipy import stats
from tqdm import tqdm
import warnings
import math
from typing import Dict, List, Optional, Tuple

# --- Suppress specific warnings for cleaner output ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- Configuration Constants ---
# File Paths
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
RAW_DATA_FILE = 'all_pairwise_results.csv'
INV_INFO_FILE = 'inv_info.tsv'

# Analysis Parameters
MIN_SAMPLES_FOR_TEST = 5  # Minimum data points per group for statistical tests
COORDINATE_TOLERANCE = 1   # Bp tolerance for matching CDS to inversions

# Plotting Parameters
N_PLOT_BINS = 20           # Number of bins for the percentile plot
PERCENTILE_START = 90      # Start percentile for the zoomed-in plot
PERCENTILE_END = 100       # End percentile for the zoomed-in plot

# --- Helper Functions ---

def parse_coords_from_cds(cds_str: str) -> Optional[Tuple[str, int, int]]:
    """Extracts chromosome, start, and end from a CDS identifier string."""
    if pd.isna(cds_str):
        return None
    match = re.search(r'chr(\w+)_start(\d+)_end(\d+)', cds_str)
    if match:
        try:
            return f'chr{match.group(1)}', int(match.group(2)), int(match.group(3))
        except (ValueError, IndexError):
            return None
    return None

def assign_recurrence_status(row: pd.Series, inv_lookup: Dict[str, List]) -> Optional[int]:
    """Assigns recurrence status based on genomic overlap."""
    coords = row.get('coords')
    if not coords:
        return None

    chrom, start, end = coords
    for inv_start, inv_end, recurrence_flag in inv_lookup.get(chrom, []):
        if max(start, inv_start - COORDINATE_TOLERANCE) <= min(end, inv_end + COORDINATE_TOLERANCE):
            return recurrence_flag
    return None

def run_mannwhitneyu(group1_data: pd.Series, group2_data: pd.Series, g1_name: str, g2_name: str, test_desc: str):
    """Runs and prints a formatted Mann-Whitney U test."""
    group1_data, group2_data = group1_data.dropna(), group2_data.dropna()
    n1, n2 = len(group1_data), len(group2_data)

    print(f"\n--- Test: {test_desc} ---")
    print(f"  Group 1: {g1_name} (n={n1}), Median={group1_data.median():.6f}" if n1 > 0 else f"  Group 1: {g1_name} (n=0)")
    print(f"  Group 2: {g2_name} (n={n2}), Median={group2_data.median():.6f}" if n2 > 0 else f"  Group 2: {g2_name} (n=0)")

    if n1 < MIN_SAMPLES_FOR_TEST or n2 < MIN_SAMPLES_FOR_TEST:
        print("\n  Result: SKIPPED (insufficient samples)")
        return

    try:
        stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        print(f"\n  Mann-Whitney U Test:")
        print(f"    U Statistic = {stat:.4f}, P-value = {p_value:.4g}")
        if p_value < 0.05:
            print("    Result: Statistically significant (p < 0.05)")
            cles = stat / (n1 * n2)
            direction = f"{g1_name} > {g2_name}" if cles > 0.5 else f"{g2_name} > {g1_name}"
            print(f"    Direction: {direction} (CLES = {cles:.4f})")
        else:
            print("    Result: Not statistically significant (p >= 0.05)")
    except ValueError as e:
        print(f"\n  Result: SKIPPED (ValueError: {e})")

def calculate_manual_percentile_bins(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Calculates stats for N bins within the top percentile range for plotting."""
    data_sorted = data.sort_values(by=value_col, ascending=True).copy()
    n_total = len(data_sorted)
    if n_total < N_PLOT_BINS:
        return pd.DataFrame()

    start_index = math.floor(n_total * (PERCENTILE_START / 100.0))
    top_data = data_sorted.iloc[start_index:]
    n_subset = len(top_data)
    if n_subset < N_PLOT_BINS:
        return pd.DataFrame()

    chunk_size, remainder = divmod(n_subset, N_PLOT_BINS)
    binned_results = []
    current_idx = 0
    percentile_step = (PERCENTILE_END - PERCENTILE_START) / N_PLOT_BINS

    for i in range(N_PLOT_BINS):
        size = chunk_size + 1 if i < remainder else chunk_size
        chunk = top_data.iloc[current_idx : current_idx + size]
        current_idx += size
        if not chunk.empty:
            binned_results.append({
                'percentile_midpoint': PERCENTILE_START + (i + 0.5) * percentile_step,
                'mean_omega': chunk[value_col].mean(),
                'sem_omega': stats.sem(chunk[value_col], nan_policy='omit'),
            })
    return pd.DataFrame(binned_results)


# --- Main Logic Functions ---

def load_and_prepare_data() -> Optional[pd.DataFrame]:
    """Loads and preprocesses both inversion and raw dN/dS data."""
    print("--- 1. Loading and Preparing Data ---")
    try:
        # Load and process inversion info
        inv_df = pd.read_csv(INV_INFO_FILE, sep='\t')
        inv_df.rename(columns={'Chromosome': 'chr', 'Start': 'inv_start', 'End': 'inv_end'}, inplace=True)
        inv_df['chr'] = inv_df['chr'].astype(str).apply(lambda x: x if x.startswith('chr') else 'chr' + x)
        inv_df.dropna(subset=['chr', 'inv_start', 'inv_end'], inplace=True)
        inv_lookup = defaultdict(list)
        for _, row in inv_df.iterrows():
            inv_lookup[row['chr']].append((row['inv_start'], row['inv_end'], row['0_single_1_recur']))
        print(f"  Loaded and processed {len(inv_df)} inversion entries.")

        # Load and process raw pairwise data
        raw_df = pd.read_csv(RAW_DATA_FILE)
        raw_df['omega'] = pd.to_numeric(raw_df['omega'], errors='coerce')
        raw_df.dropna(subset=['omega', 'CDS'], inplace=True)
        raw_df = raw_df[raw_df['omega'] != 99].copy()
        raw_df = raw_df[raw_df['Group1'] == raw_df['Group2']].copy() # Keep only within-group comparisons
        raw_df['Orientation'] = raw_df['Group1'].map({0: 'Direct', 1: 'Inverted'})
        raw_df['coords'] = raw_df['CDS'].apply(parse_coords_from_cds)

        # Assign recurrence status
        raw_df['recurrence_flag'] = raw_df.apply(lambda r: assign_recurrence_status(r, inv_lookup), axis=1)
        raw_df.dropna(subset=['recurrence_flag'], inplace=True)
        raw_df['Recurrence Type'] = raw_df['recurrence_flag'].map({0.0: 'Single-Event', 1.0: 'Recurrent'})

        print(f"  Loaded and processed {len(raw_df)} valid pairwise comparisons.")
        return raw_df
    except FileNotFoundError as e:
        print(f"  Error: Data file not found - {e}. Aborting.")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during data loading: {e}")
        return None

def perform_dnds_analysis(raw_df: pd.DataFrame, exclude_identical: bool) -> Optional[pd.DataFrame]:
    """
    Performs the full dN/dS aggregation and statistical analysis.
    - If exclude_identical is False, treats omega=-1 as 0.
    - If exclude_identical is True, filters out omega=-1 pairs.
    """
    analysis_title = "Excluding Identical Pairs" if exclude_identical else "Including Identical Pairs (as dN/dS=0)"
    print("\n" + "="*60)
    print(f"--- 2. Performing Analysis: {analysis_title} ---")
    print("="*60)

    aggregated_results = []
    grouped = raw_df.groupby(['Recurrence Type', 'Orientation', 'CDS'])

    print("  Aggregating per-sequence median dN/dS...")
    for name, group_df in tqdm(grouped, desc=f"Aggregating ({'Excl.' if exclude_identical else 'Incl.'} Identicals)"):
        rec_type, orient, cds = name
        unique_seqs = pd.unique(pd.concat([group_df['Seq1'], group_df['Seq2']]))
        for seq_id in unique_seqs:
            comparisons = group_df[(group_df['Seq1'] == seq_id) | (group_df['Seq2'] == seq_id)]
            omega_values = comparisons['omega']

            if exclude_identical:
                values_to_process = omega_values[omega_values != -1]
            else:
                values_to_process = omega_values.replace(-1, 0.0)

            if not values_to_process.empty:
                aggregated_results.append({
                    'Recurrence Type': rec_type,
                    'Orientation': orient,
                    'median_omega_per_sequence': values_to_process.median()
                })

    if not aggregated_results:
        print("  No data to analyze after aggregation.")
        return None

    agg_df = pd.DataFrame(aggregated_results)
    print(f"\n  Generated {len(agg_df)} aggregated per-sequence data points.")

    # --- Print Summary Statistics ---
    summary = agg_df.groupby(['Recurrence Type', 'Orientation'])['median_omega_per_sequence'].agg(['mean', 'median', 'count'])
    print("\n--- Summary Statistics (Per-Sequence Medians) ---")
    print(summary.unstack().to_string(float_format="%.6f"))

    # --- Perform Statistical Tests ---
    groups = {
        (rec_type, orient): group['median_omega_per_sequence']
        for (rec_type, orient), group in agg_df.groupby(['Recurrence Type', 'Orientation'])
    }
    rec_inv = groups.get(('Recurrent', 'Inverted'), pd.Series(dtype=float))
    rec_dir = groups.get(('Recurrent', 'Direct'), pd.Series(dtype=float))
    sngl_inv = groups.get(('Single-Event', 'Inverted'), pd.Series(dtype=float))
    sngl_dir = groups.get(('Single-Event', 'Direct'), pd.Series(dtype=float))

    run_mannwhitneyu(rec_inv, rec_dir, "Rec Inv", "Rec Dir", "Inv vs Dir within Recurrent")
    run_mannwhitneyu(sngl_inv, sngl_dir, "Sngl Inv", "Sngl Dir", "Inv vs Dir within Single-Event")
    run_mannwhitneyu(rec_dir, sngl_dir, "Rec Dir", "Sngl Dir", "Rec vs Sngl within Direct")
    run_mannwhitneyu(sngl_inv, sngl_inv, "Rec Inv", "Sngl Inv", "Rec vs Sngl within Inverted")

    return agg_df

def generate_percentile_plot(agg_df: pd.DataFrame):
    """Generates and saves the percentile distribution plot."""
    print("\n" + "="*60)
    print("--- 3. Generating Percentile Plot (from analysis including identicals) ---")
    print("="*60)

    if agg_df is None or agg_df.empty:
        print("  Skipping plot generation: No data available.")
        return

    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(14, 9))
    group_styles = {
        ('Recurrent', 'Direct'): {'color': 'royalblue', 'linestyle': '-', 'label': 'Recurrent Direct'},
        ('Recurrent', 'Inverted'): {'color': 'skyblue', 'linestyle': '--', 'label': 'Recurrent Inverted'},
        ('Single-Event', 'Direct'): {'color': 'firebrick', 'linestyle': '-', 'label': 'Single-Event Direct'},
        ('Single-Event', 'Inverted'): {'color': 'salmon', 'linestyle': '--', 'label': 'Single-Event Inverted'}
    }

    for (rec_type, orient), style in group_styles.items():
        group_data = agg_df[(agg_df['Recurrence Type'] == rec_type) & (agg_df['Orientation'] == orient)]
        print(f"  Processing for plot: {style['label']} (n={len(group_data)} sequences)")
        if not group_data.empty:
            binned_stats = calculate_manual_percentile_bins(group_data, 'median_omega_per_sequence')
            if not binned_stats.empty:
                ax.plot(
                    binned_stats['percentile_midpoint'], binned_stats['mean_omega'],
                    color=style['color'], linestyle=style['linestyle'],
                    label=style['label'], marker='o', markersize=4, alpha=0.7
                )
                ax.fill_between(
                    binned_stats['percentile_midpoint'],
                    binned_stats['mean_omega'] - binned_stats['sem_omega'],
                    binned_stats['mean_omega'] + binned_stats['sem_omega'],
                    color=style['color'], alpha=0.15
                )

    ax.set_xlabel(r"Percentile of Per-Sequence Median $\omega$ (within group)", fontsize=16)
    ax.set_ylabel(r"Mean of Per-Sequence Median $\omega$ (within percentile bin)", fontsize=16)
    ax.set_title(r"Distribution of Per-Sequence Median $\omega$ at High Percentiles", fontsize=18)
    ax.set_xlim(PERCENTILE_START, PERCENTILE_END)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=13, loc='upper left')

    plot_path = PLOTS_DIR / "omega_percentile_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"\n  Plot saved to: {plot_path}")

def main():
    """Main execution workflow."""
    print("--- Starting Full dN/dS Inversion Analysis ---")
    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    raw_df = load_and_prepare_data()
    if raw_df is None:
        return

    # Run the first analysis (original method, including identicals)
    agg_data_with_identicals = perform_dnds_analysis(raw_df, exclude_identical=False)

    # Run the second analysis (new method, excluding identicals)
    perform_dnds_analysis(raw_df, exclude_identical=True)

    # Generate the plot based on the primary (first) analysis
    generate_percentile_plot(agg_data_with_identicals)

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
