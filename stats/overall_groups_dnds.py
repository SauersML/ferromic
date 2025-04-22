import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator, MultipleLocator
from pathlib import Path
import os
import math
import re
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from collections import defaultdict
from scipy import stats # Import stats for testing
from tqdm import tqdm # Import tqdm for progress bar
import warnings # Import warnings module

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Configuration ---
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots") # For saving the plot
RAW_DATA_FILE = 'all_pairwise_results.csv'
INV_INFO_FILE = 'inv_info.csv'
MIN_SAMPLES_FOR_TEST = 5 # Minimum number of data points (sequences) per group for tests
N_PLOT_BINS = 20 # Number of bins to plot
PERCENTILE_START = 90 # Starting percentile
PERCENTILE_END = 100 # Ending percentile
# --- Constants ---
RECURRENCE_COL_RAW = '0_single_1_recur'
RECURRENCE_COL_INTERNAL = 'recurrence_flag'
COORDINATE_TOLERANCE = 1 # Allow off-by-one matching

# --- Setup ---
print("--- Script Start ---")
for directory in [RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)
print(f"Ensured directories exist: {RESULTS_DIR}, {PLOTS_DIR}")

# --- Helper Functions ---
def parse_transcript_from_cds(cds_str):
    """Extracts ENST ID from CDS string."""
    if pd.isna(cds_str):
        return None
    match = re.search(r'(ENST\d+\.\d+)', cds_str)
    if match:
        return match.group(1)
    else:
        match_simple = re.match(r'(ENST\d+\.\d+)', cds_str)
        if match_simple:
             return match_simple.group(1)
        return None

def parse_coords_raw(cds_str):
    """Parses 'chrN_startN_endN...' from the RAW CDS column."""
    if pd.isna(cds_str):
        return None, np.nan, np.nan
    match = re.search(r'chr(\w+)_start(\d+)_end(\d+)', cds_str)
    if match:
        try:
            chrom = 'chr' + match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            return chrom, start, end
        except ValueError:
            return None, np.nan, np.nan
    else:
        return None, np.nan, np.nan

def assign_recurrence_status(row, inv_lookup, tolerance=COORDINATE_TOLERANCE):
    """Assigns recurrence status based on overlap with inversions, allowing tolerance."""
    chrom = row['chrom']
    start = row['start']
    end = row['end']

    if pd.isna(chrom) or pd.isna(start) or pd.isna(end):
        return np.nan

    relevant_inversions = inv_lookup.get(chrom, [])
    if not relevant_inversions:
        return np.nan

    for inv_start, inv_end, recurrence_flag in relevant_inversions:
        if max(start, inv_start - tolerance) <= min(end, inv_end + tolerance):
            return recurrence_flag

    return np.nan # Return NaN if no overlap found

def run_mannwhitneyu(group1_data, group2_data, group1_name, group2_name, test_description):
    """Runs Mann-Whitney U test and prints results (without summary stats)."""
    group1_data = group1_data.dropna()
    group2_data = group2_data.dropna()
    n1 = len(group1_data)
    n2 = len(group2_data)

    print(f"\n  --- Test: {test_description} ---")
    print(f"    Group 1: {group1_name} (n={n1})")
    print(f"    Group 2: {group2_name} (n={n2})")

    if n1 < MIN_SAMPLES_FOR_TEST or n2 < MIN_SAMPLES_FOR_TEST:
        print(f"\n    Skipping statistical test: Insufficient samples (min required={MIN_SAMPLES_FOR_TEST}).")
        return None, None
    try:
        stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        print(f"\n    Mann-Whitney U Test Result:")
        print(f"      Statistic = {stat:.4f}, P-value = {p_value:.4g}")
        if p_value < 0.05:
            print("      Result: Statistically significant (p < 0.05)")
        else:
            print("      Result: Not statistically significant (p >= 0.05)")
        return stat, p_value
    except ValueError as e:
         print(f"\n    Skipping statistical test due to error: {e}")
         return None, None
    except Exception as e:
        print(f"\n    An unexpected error occurred during the test: {e}")
        return None, None

def calculate_manual_percentile_bins(data, value_col, start_perc=PERCENTILE_START, n_bins=N_PLOT_BINS):
    """Manually calculates stats within n_bins covering the top (100-start_perc)% range."""

    if data.empty or value_col not in data.columns or data[value_col].isna().all():
        print(f"Warning: Input data empty or value column '{value_col}' missing/all NaN. Skipping binning.")
        return pd.DataFrame()

    # Sort data by the value column to easily select top percentiles
    data_sorted = data.sort_values(by=value_col, ascending=True).copy()
    n_total = len(data_sorted)

    if n_total < n_bins:
        print(f"Warning: Only {n_total} data points available, less than target bins ({n_bins}). Skipping binning.")
        return pd.DataFrame()

    # Determine the index cutoff for the starting percentile
    start_index = math.floor(n_total * (start_perc / 100.0))
    start_index = min(start_index, n_total - 1)

    # Select the data from the starting percentile onwards
    top_data = data_sorted.iloc[start_index:].copy()
    n_subset = len(top_data)

    if n_subset < n_bins :
         print(f"Warning: Only {n_subset} points >= {start_perc}th percentile, less than target bins ({n_bins}). Skipping binning.")
         return pd.DataFrame()

    chunk_size = n_subset // n_bins
    remainder = n_subset % n_bins
    print(f"    Manual Binning: N in Top {100-start_perc}% = {n_subset}, Target Bins={n_bins}, Base Chunk Size={chunk_size}, Remainder={remainder}")

    binned_results = []
    current_idx = 0
    percentile_step = (100.0 - start_perc) / n_bins

    for i in range(n_bins):
        current_chunk_size = chunk_size + 1 if i < remainder else chunk_size
        chunk_start_idx = current_idx
        chunk_end_idx = current_idx + current_chunk_size
        current_chunk = top_data.iloc[chunk_start_idx:chunk_end_idx]
        current_idx = chunk_end_idx

        if current_chunk.empty:
            print(f"      Bin {i}: Empty chunk unexpected, skipping.")
            continue

        chunk_values = current_chunk[value_col].dropna()
        if chunk_values.empty:
             continue

        mean_omega = chunk_values.mean()
        n_obs = len(chunk_values)
        sem_omega = 0 if n_obs < 2 else chunk_values.std(ddof=1) / np.sqrt(n_obs)
        # Removed all_zero check
        percentile_midpoint = start_perc + (i + 0.5) * percentile_step

        binned_results.append({
            'manual_bin_index': i,
            'percentile_midpoint': percentile_midpoint, # X-value for plotting
            'mean_omega': mean_omega,
            'sem_omega': sem_omega,
            'n_obs': n_obs,
            # 'all_zero': all_zero # Removed
        })

    binned_stats_df = pd.DataFrame(binned_results)
    if len(binned_stats_df) != n_bins:
         print(f"Warning: Generated {len(binned_stats_df)} bins, but expected exactly {n_bins}. Review binning logic.")
    else:
         print(f"    Successfully generated {len(binned_stats_df)} bins.")

    return binned_stats_df


# --- Main Function ---
def summarize_and_test_dnds_effects():
    """Loads data, calculates summaries, performs tests, generates plot and prints results."""
    print("--- Starting Summary and Statistical Testing ---")

    # --- 1. Load and Process Inversion Info ---
    print(f"\n[1] Loading and processing inversion data from: {INV_INFO_FILE}")
    inv_raw_cols = ['Chromosome', 'Start', 'End']
    inv_expected_cols_internal = ['chr', 'inv_start', 'inv_end', RECURRENCE_COL_INTERNAL]
    inv_df = pd.DataFrame(columns=inv_expected_cols_internal)
    try:
        inv_df_raw = pd.read_csv(INV_INFO_FILE)
        print(f"  Read {INV_INFO_FILE}, shape: {inv_df_raw.shape}")

        if all(col in inv_df_raw.columns for col in inv_raw_cols):
            print(f"  Found expected columns: {inv_raw_cols}")
            inv_df = inv_df_raw.rename(columns={
                'Chromosome': 'chr',
                'Start': 'inv_start',
                'End': 'inv_end'
            }).copy()

            if RECURRENCE_COL_RAW not in inv_df.columns:
                print(f"  Warning: Recurrence column '{RECURRENCE_COL_RAW}' not found. Assigning 0.")
                inv_df[RECURRENCE_COL_INTERNAL] = 0
            else:
                inv_df[RECURRENCE_COL_INTERNAL] = pd.to_numeric(inv_df[RECURRENCE_COL_RAW], errors='coerce').fillna(0).astype(int)

            inv_df.dropna(subset=['chr', 'inv_start', 'inv_end'], inplace=True)
            print(f"  Shape after dropping NaNs in key columns: {inv_df.shape}")

            if not inv_df.empty:
                inv_df['chr'] = inv_df['chr'].astype(str).apply(lambda x: x if x.startswith('chr') else 'chr' + x)
                inv_df['inv_start'] = pd.to_numeric(inv_df['inv_start'], errors='coerce')
                inv_df['inv_end'] = pd.to_numeric(inv_df['inv_end'], errors='coerce')
                inv_df.dropna(subset=['inv_start', 'inv_end'], inplace=True)
                print(f"  Shape after coord conversion/drop: {inv_df.shape}")

                if not inv_df.empty:
                    inv_df['inv_start'] = inv_df['inv_start'].astype(int)
                    inv_df['inv_end'] = inv_df['inv_end'].astype(int)
                    inv_df = inv_df[['chr', 'inv_start', 'inv_end', RECURRENCE_COL_INTERNAL]]
                    print(f"  Processed {len(inv_df)} valid inversion entries.")
                else:
                    print("  Inversion data empty after coordinate processing.")
                    inv_df = pd.DataFrame(columns=inv_expected_cols_internal)
            else:
                 print("  Inversion data empty after initial NaN drop.")
                 inv_df = pd.DataFrame(columns=inv_expected_cols_internal)
        else:
            missing_cols = [col for col in inv_raw_cols if col not in inv_df_raw.columns]
            print(f"  Error: Missing expected columns: {missing_cols}. Cannot process inversions.")
            inv_df = pd.DataFrame(columns=inv_expected_cols_internal)

    except FileNotFoundError:
        print(f"  Error: Inversion file not found at {INV_INFO_FILE}. Cannot assign recurrence.")
        inv_df = pd.DataFrame(columns=inv_expected_cols_internal)
    except Exception as e:
        print(f"  Error processing inversion file: {e}")
        inv_df = pd.DataFrame(columns=inv_expected_cols_internal)

    inv_lookup = defaultdict(list)
    for _, row in inv_df.iterrows():
        inv_lookup[row['chr']].append((row['inv_start'], row['inv_end'], row[RECURRENCE_COL_INTERNAL]))
    print(f"  Created inversion lookup for {len(inv_lookup)} chromosomes.")

    # --- 2. Load and Process Raw Pairwise Data ---
    print(f"\n[2] Loading and processing raw pairwise data from: {RAW_DATA_FILE}")
    raw_df_processed = pd.DataFrame() # Initialize empty dataframe
    try:
        raw_df = pd.read_csv(RAW_DATA_FILE)
        print(f"  Read {RAW_DATA_FILE}, shape: {raw_df.shape}")

        print("  Parsing transcript IDs and coordinates from raw data...")
        raw_df['transcript_id'] = raw_df['CDS'].apply(parse_transcript_from_cds)
        coords_parsed_raw = raw_df['CDS'].apply(parse_coords_raw)
        raw_df['chrom'] = coords_parsed_raw.apply(lambda x: x[0])
        raw_df['start'] = coords_parsed_raw.apply(lambda x: x[1])
        raw_df['end'] = coords_parsed_raw.apply(lambda x: x[2])

        essential_cols = ['transcript_id', 'omega', 'Group1', 'Group2', 'chrom', 'start', 'end', 'Seq1', 'Seq2']
        raw_df.dropna(subset=essential_cols, inplace=True)
        print(f"  Shape after parsing and dropping essential NaNs: {raw_df.shape}")

        raw_df = raw_df[raw_df['Group1'] == raw_df['Group2']].copy()
        print(f"  Shape after keeping only within-group comparisons: {raw_df.shape}")

        raw_df['Orientation'] = raw_df['Group1'].map({0: 'Direct', 1: 'Inverted'})
        print(f"  Assigned Orientations: {raw_df['Orientation'].value_counts().to_dict()}")

        raw_df['omega'] = pd.to_numeric(raw_df['omega'], errors='coerce')
        original_len = len(raw_df)
        raw_df = raw_df[raw_df['omega'] != 99].copy() # Filter 99
        filtered_count = original_len - len(raw_df)
        print(f"  Filtered out {filtered_count} rows with omega=99.")
        raw_df.dropna(subset=['omega'], inplace=True)
        print(f"  Shape after cleaning omega: {raw_df.shape}")

        if not inv_df.empty:
            print(f"  Assigning recurrence status to raw comparisons based on overlap (tolerance={COORDINATE_TOLERANCE})...")
            raw_df[RECURRENCE_COL_INTERNAL] = raw_df.apply(
                lambda row: assign_recurrence_status(row, inv_lookup, tolerance=COORDINATE_TOLERANCE), axis=1
            )
            assigned_count = raw_df[RECURRENCE_COL_INTERNAL].notna().sum()
            print(f"  Assigned recurrence status to {assigned_count} / {len(raw_df)} raw comparisons.")
            raw_df['Recurrence Type'] = raw_df[RECURRENCE_COL_INTERNAL].map({0.0: 'Single-Event', 1.0: 'Recurrent'}).fillna('Non-Overlapping')
        else:
            print("  Skipping recurrence assignment to raw data due to issues with inversion data.")
            raw_df[RECURRENCE_COL_INTERNAL] = np.nan
            raw_df['Recurrence Type'] = 'Unknown'

        raw_df_processed = raw_df[raw_df['Recurrence Type'].isin(['Single-Event', 'Recurrent'])].copy()
        print(f"  Filtered raw comparisons for Recurrent/Single-Event types: {len(raw_df_processed)} rows.")

    except FileNotFoundError:
        print(f"  Error: Raw data file not found at {RAW_DATA_FILE}. Cannot calculate raw dN/dS stats.")
    except Exception as e:
        print(f"  Error reading raw data file: {e}")

    # --- 3. Aggregate Raw Omega per Sequence within Transcript/Category ---
    print("\n[3] Aggregating raw omega values per sequence...")
    agg_omega_per_seq = pd.DataFrame() # Initialize empty
    if not raw_df_processed.empty:
        aggregated_results_list = []
        grouping_cols_agg = ['transcript_id', 'Recurrence Type', 'Orientation']
        if raw_df_processed[grouping_cols_agg].isnull().any().any():
            print(f"  Warning: Found NaNs in aggregation grouping columns: {raw_df_processed[grouping_cols_agg].isnull().sum().to_dict()}")
            raw_df_processed.dropna(subset=grouping_cols_agg, inplace=True)
            print(f"  Shape after dropping NaN grouping rows: {raw_df_processed.shape}")

        grouped = raw_df_processed.groupby(grouping_cols_agg)
        print(f"  Aggregating across {len(grouped)} transcript/category groups...")

        for name, group_df in tqdm(grouped, desc="Aggregating per sequence"):
            trans_id, rec_type, orient = name
            unique_seqs = pd.unique(pd.concat([group_df['Seq1'], group_df['Seq2']]))

            for seq_id in unique_seqs:
                seq_comparisons = group_df[(group_df['Seq1'] == seq_id) | (group_df['Seq2'] == seq_id)]
                if not seq_comparisons.empty:
                    omega_values_mapped = seq_comparisons['omega'].replace(-1, 0.0)
                    median_omega_mapped = omega_values_mapped.median()
                    aggregated_results_list.append({
                        'transcript_id': trans_id,
                        'Recurrence Type': rec_type,
                        'Orientation': orient,
                        'Sequence ID': seq_id,
                        'median_omega_per_sequence': median_omega_mapped
                    })

        if aggregated_results_list:
            agg_omega_per_seq = pd.DataFrame(aggregated_results_list)
            print(f"  Aggregated raw omega into {len(agg_omega_per_seq)} median values (one per sequence per transcript/category).")
            agg_omega_per_seq.dropna(subset=['median_omega_per_sequence'], inplace=True)
            print(f"  Shape after dropping potential NaN medians: {agg_omega_per_seq.shape}")
        else:
            print("  No results after aggregation loop.")
    else:
        print("  Skipping aggregation as processed raw data is empty.")


    # --- 4. Remove Model Loading Section ---
    print("\n[4] Skipping model results loading.")

    # --- 5. Calculate and Print Summaries (Aggregated Raw Omega Only) ---
    print("\n" + "="*40)
    print("      Summary Statistics")
    print("="*40)

    # REMOVED Model Effect Size Summary

    print("\n--- Aggregated Raw dN/dS (Median of Per-Sequence-within-Transcript Medians) ---")
    if not agg_omega_per_seq.empty:
        summary_group_cols = ['Recurrence Type', 'Orientation']
        if agg_omega_per_seq[summary_group_cols].isnull().any().any():
            print(f"  Warning: Found NaNs in summary grouping columns: {agg_omega_per_seq[summary_group_cols].isnull().sum().to_dict()}")
            agg_omega_per_seq.dropna(subset=summary_group_cols, inplace=True)
            print(f"  Shape after dropping NaN summary grouping columns: {agg_omega_per_seq.shape}")

        if not agg_omega_per_seq.empty:
            raw_summary = agg_omega_per_seq.groupby(summary_group_cols)['median_omega_per_sequence'].agg(['median'])
            raw_summary = raw_summary.rename(columns={'median': 'Median of Per-Sequence Medians (Omega, -1 mapped to 0)'})
            print(raw_summary.unstack().to_string(float_format="%.6f"))
            raw_summary_available_for_test = True
        else:
             print("  Aggregated data empty after dropping NaN grouping columns.")
             raw_summary_available_for_test = False
    else:
        print("  No aggregated per-sequence omega data available to summarize.")
        raw_summary_available_for_test = False


    # --- 6. Perform and Print Statistical Tests (on Aggregated Per-Sequence Omega) ---
    print("\n" + "="*40)
    print("     Statistical Tests (on Per-Sequence-within-Transcript Median Omega)")
    print("="*40)

    groups_for_test = {} # To store data for testing
    if raw_summary_available_for_test and not agg_omega_per_seq.empty:
        for name, group in agg_omega_per_seq.groupby(['Recurrence Type', 'Orientation']):
            groups_for_test[name] = group['median_omega_per_sequence'] # Data points for test

        rec_direct = groups_for_test.get(('Recurrent', 'Direct'), pd.Series(dtype=float))
        rec_invert = groups_for_test.get(('Recurrent', 'Inverted'), pd.Series(dtype=float))
        sngl_direct = groups_for_test.get(('Single-Event', 'Direct'), pd.Series(dtype=float))
        sngl_invert = groups_for_test.get(('Single-Event', 'Inverted'), pd.Series(dtype=float))

        all_recurrent = pd.concat([rec_direct, rec_invert]).dropna()
        all_single_event = pd.concat([sngl_direct, sngl_invert]).dropna()

        # --- Run Tests ---
        run_mannwhitneyu(rec_invert, rec_direct, "Rec Inv", "Rec Dir", "Inv vs Dir within Recurrent")
        run_mannwhitneyu(sngl_invert, sngl_direct, "Sngl Inv", "Sngl Dir", "Inv vs Dir within Single-Event")
        run_mannwhitneyu(rec_direct, sngl_direct, "Rec Dir", "Sngl Dir", "Rec vs Sngl within Direct")
        run_mannwhitneyu(rec_invert, sngl_invert, "Rec Inv", "Sngl Inv", "Rec vs Sngl within Inverted")
        run_mannwhitneyu(all_recurrent, all_single_event, "All Recurrent", "All Single-Event", "Overall Recurrent vs Single-Event")

    else:
        print("  Skipping tests: Insufficient aggregated raw dN/dS data.")

    # --- 7. Generate Percentile Plot ---
    print("\n" + "="*40)
    print("     Generating Percentile Plot")
    print("="*40)

    if raw_summary_available_for_test and not agg_omega_per_seq.empty: # Removed STATSMODELS_AVAILABLE check
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 9))

        group_styles = {
            ('Recurrent', 'Direct'): {'color': 'royalblue', 'linestyle': '-', 'label': 'Recurrent Direct'},
            ('Recurrent', 'Inverted'): {'color': 'skyblue', 'linestyle': '--', 'label': 'Recurrent Inverted'},
            ('Single-Event', 'Direct'): {'color': 'firebrick', 'linestyle': '-', 'label': 'Single-Event Direct'},
            ('Single-Event', 'Inverted'): {'color': 'salmon', 'linestyle': '--', 'label': 'Single-Event Inverted'}
        }

        plot_data_generated = False
        all_binned_stats_dfs = [] # Collect binned stats for each group

        for name, style in group_styles.items():
            rec_type, orient = name
            group_data = agg_omega_per_seq[
                (agg_omega_per_seq['Recurrence Type'] == rec_type) &
                (agg_omega_per_seq['Orientation'] == orient)
            ].copy()

            print(f"\n  Processing group for plot: {style['label']} (n={len(group_data)} sequences)")
            if group_data.empty or group_data['median_omega_per_sequence'].isna().all():
                 print("    Skipping group due to no data.")
                 continue

            # Calculate manually chunked bins for the top percentiles
            binned_stats = calculate_manual_percentile_bins(
                group_data, 'median_omega_per_sequence',
                start_perc=PERCENTILE_START, n_bins=N_PLOT_BINS
            )

            if not binned_stats.empty and 'percentile_midpoint' in binned_stats.columns:
                 if len(binned_stats) < N_PLOT_BINS: # Check if exactly N_PLOT_BINS were generated
                     print(f"    Warning: Manual binning generated {len(binned_stats)} bins, expected {N_PLOT_BINS}. Plotting available bins.")
                 else:
                     print(f"    Generated {len(binned_stats)} manual bins for plotting.")
                 all_binned_stats_dfs.append({'name': name, 'style': style, 'bins': binned_stats}) # Store for plotting
                 plot_data_generated = True # Mark that we have data to plot
            else:
                 print(f"    Skipping plotting for {style['label']} due to empty or invalid binned stats.")

        # Now plot the collected binned stats
        if plot_data_generated:
            for group_plot_data in all_binned_stats_dfs:
                name = group_plot_data['name']
                style = group_plot_data['style']
                plot_df = group_plot_data['bins'] # This is the binned data for the group

                if plot_df.empty: continue

                # Plot simple line connecting the mean points
                # Use percentile_midpoint for x-axis
                ax.plot(plot_df['percentile_midpoint'], plot_df['mean_omega'],
                        color=style['color'],
                        linestyle=style['linestyle'],
                        linewidth=2.0, # Make lines slightly thinner
                        alpha=0.6, # Add transparency
                        label=style['label'],
                        marker='o', # Add markers to see the bin points
                        markersize=4, # Smaller markers
                        zorder=5) # Keep lines above shading
                print(f"      Plotted line for {style['label']} with {len(plot_df)} points.")

                # Plot SEM shading
                valid_sem_points = plot_df.dropna(subset=['sem_omega'])
                if len(valid_sem_points) > 1:
                    # Sort points by x-axis for fill_between
                    valid_sem_points = valid_sem_points.sort_values('percentile_midpoint')
                    x_coords = valid_sem_points['percentile_midpoint']
                    y_means = valid_sem_points['mean_omega']
                    y_sem = valid_sem_points['sem_omega']
                    ax.fill_between(x_coords, y_means - y_sem, y_means + y_sem,
                                    color=style['color'], alpha=0.10, zorder=0)
                    print(f"      Plotted SEM shading for {style['label']}.")
                else:
                    print(f"      Skipping SEM shading for {style['label']} due to insufficient points with valid SEM.")

            # --- Plot Customization ---
            ax.set_xlabel(r"Percentile of Per-Sequence Median $\omega$ (within group)", fontsize=14)
            ax.set_ylabel(r"Per-Sequence Median $\omega$ (averaged within each percentile)", fontsize=14)
            ax.set_title(r"Distribution of Per-Sequence Median $\omega$ (Top Percentiles)", fontsize=16)
            
            ax.set_xlim(PERCENTILE_START, PERCENTILE_END)
            # Set integer ticks every 1 percentile
            tick_values = np.arange(PERCENTILE_START, PERCENTILE_END + 1, 1)
            ax.set_xticks(tick_values)
            # Optionally rotate x-tick labels if they overlap
            ax.tick_params(axis='x', labelsize=10, rotation=45)
            ax.tick_params(axis='y', labelsize=12)
            
            # Add vertical grid lines at each percentile tick
            for tick_val in tick_values:
                ax.axvline(tick_val, color='grey', linestyle=':', linewidth=0.5, alpha=0.3, zorder=-1)
            print(f"   Added vertical grid lines at percentiles: {tick_values}")
            
            # Place legend inside the plot area, top left
            ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor='white', framealpha=0.7)
            ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6) # Keep horizontal grid
            ax.grid(False, axis='x') # Turn off default x grid
            ax.set_ylim(bottom=-0.01)
            
            plot_out_fname = PLOTS_DIR / "omega_percentile_distribution.png"
            print(f"\n  Saving percentile plot to {plot_out_fname}")
            try:
                plt.tight_layout()
                plt.savefig(plot_out_fname, dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"  Error saving plot: {e}")
            plt.close(fig)
        else:
            print("\n  Skipping plot generation as no data was prepared or plotted.")

    else:
        print("\n  Skipping plot generation: Insufficient aggregated raw dN/dS data.")

    print("\n" + "="*40)
    print("--- Summary, Testing, and Plotting Complete ---")

# --- Main Execution ---
if __name__ == "__main__":
    summarize_and_test_dnds_effects()
