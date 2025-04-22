import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator
from pathlib import Path
import os
import math
import re
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable

try:
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    STATSMODELS_AVAILABLE = True
    print("Successfully imported statsmodels.")
except ImportError:
    print("Warning: statsmodels not found. LOESS smoothing will be skipped.")
    STATSMODELS_AVAILABLE = False
    lowess = None

from collections import defaultdict
from scipy import stats # Import stats for testing
from tqdm import tqdm # Import tqdm for progress bar
import warnings # Import warnings module

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- Configuration ---
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots") # For saving the plot
RAW_DATA_FILE = 'all_pairwise_results.csv'
INV_INFO_FILE = 'inv_info.csv'
MIN_SAMPLES_FOR_TEST = 5 # Minimum number of data points (sequences) per group for tests
N_PERCENTILE_BINS = 100 # Use 100 bins for 100 percentiles
LOESS_FRAC = 0.4 # Smoothing fraction for LOESS (adjust as needed)
PLOT_X_MIN_PERCENTILE = 80 # Start plotting from this percentile
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

def calculate_percentile_bins(data, value_col, n_bins=N_PERCENTILE_BINS):
    """Calculates mean, SEM, and checks if all values are zero within exact percentile bins."""
    if data.empty or data[value_col].isna().all():
        return pd.DataFrame(columns=['percentile_int', 'mean_omega', 'sem_omega', 'n_obs', 'all_zero'])

    valid_data = data[[value_col]].dropna()
    if len(valid_data) < 2: # Need at least 2 points to calculate percentiles reasonably
        print(f"Warning: Not enough valid data points ({len(valid_data)}) for percentile binning. Skipping.")
        return pd.DataFrame(columns=['percentile_int', 'mean_omega', 'sem_omega', 'n_obs', 'all_zero'])

    # Calculate exact percentiles for each data point
    data['percentile_exact'] = data[value_col].rank(pct=True) * 100
    # Assign to integer percentile bin (e.g., 0.5% -> bin 0, 99.8% -> bin 99)
    # Floor the percentile rank to get the bin index (0-99)
    data['percentile_int'] = np.floor(data['percentile_exact']).astype(int)
    # Ensure values exactly at 100 go into the last bin (99)
    data.loc[data['percentile_int'] == 100, 'percentile_int'] = n_bins - 1

    # Define aggregation functions
    def sem(x):
        x = x.dropna()
        if len(x) < 2: return 0
        return x.std() / np.sqrt(len(x))

    def check_all_zero(x):
        x = x.dropna()
        if x.empty: return False
        return (x == 0.0).all()

    # Aggregate per integer percentile bin
    binned_stats = data.groupby('percentile_int', observed=False)[value_col].agg(
        mean_omega='mean',
        sem_omega=sem,
        n_obs='count',
        all_zero=check_all_zero
    ).reset_index()

    # percentile_int now directly represents the percentile (0 to 99)
    return binned_stats


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

        # Combine orientations for Recurrent vs Single-Event test
        all_recurrent = pd.concat([rec_direct, rec_invert]).dropna()
        all_single_event = pd.concat([sngl_direct, sngl_invert]).dropna()

        # --- Run Tests ---
        run_mannwhitneyu(rec_invert, rec_direct, "Rec Inv", "Rec Dir", "Inv vs Dir within Recurrent")
        run_mannwhitneyu(sngl_invert, sngl_direct, "Sngl Inv", "Sngl Dir", "Inv vs Dir within Single-Event")
        run_mannwhitneyu(rec_direct, sngl_direct, "Rec Dir", "Sngl Dir", "Rec vs Sngl within Direct")
        run_mannwhitneyu(rec_invert, sngl_invert, "Rec Inv", "Sngl Inv", "Rec vs Sngl within Inverted")
        # --- New Test ---
        run_mannwhitneyu(all_recurrent, all_single_event, "All Recurrent", "All Single-Event", "Overall Recurrent vs Single-Event")

    else:
        print("  Skipping tests: Insufficient aggregated raw dN/dS data.")

    # --- 7. Generate Percentile Plot ---
    print("\n" + "="*40)
    print("     Generating Percentile Plot")
    print("="*40)

    if raw_summary_available_for_test and not agg_omega_per_seq.empty and STATSMODELS_AVAILABLE:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 9))

        group_styles = {
            ('Recurrent', 'Direct'): {'color': 'royalblue', 'linestyle': '-', 'label': 'Recurrent Direct'},
            ('Recurrent', 'Inverted'): {'color': 'skyblue', 'linestyle': '--', 'label': 'Recurrent Inverted'},
            ('Single-Event', 'Direct'): {'color': 'firebrick', 'linestyle': '-', 'label': 'Single-Event Direct'},
            ('Single-Event', 'Inverted'): {'color': 'salmon', 'linestyle': '--', 'label': 'Single-Event Inverted'}
        }

        plot_data_generated = False
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

            # Calculate binned statistics using integer percentiles
            binned_stats = calculate_percentile_bins(group_data, 'median_omega_per_sequence', n_bins=N_PERCENTILE_BINS)

            if not binned_stats.empty and 'percentile_int' in binned_stats.columns:
                print(f"    Generated {len(binned_stats)} bins for plotting.")

                # Filter bins for plotting based on percentile range
                plot_df = binned_stats[binned_stats['percentile_int'] >= PLOT_X_MIN_PERCENTILE].copy()
                print(f"    Keeping {len(plot_df)} bins >= {PLOT_X_MIN_PERCENTILE}th percentile.")

                if not plot_df.empty:
                    # Separate points that are all zero vs. not
                    all_zero_points = plot_df[plot_df['all_zero'] == True]
                    non_zero_points = plot_df[plot_df['all_zero'] == False]

                    # Plot actual binned mean points (non-zero)
                    if not non_zero_points.empty:
                        ax.scatter(non_zero_points['percentile_int'], non_zero_points['mean_omega'],
                                   color=style['color'], alpha=0.6, s=30, label=f"_{style['label']} (bins)") # Underscore hides from legend
                        print(f"      Plotted {len(non_zero_points)} non-zero mean bin points.")

                    # Plot actual binned mean points (all-zero)
                    if not all_zero_points.empty:
                        ax.scatter(all_zero_points['percentile_int'], all_zero_points['mean_omega'],
                                   color='grey', alpha=0.4, s=30, marker='x', label=f"_{style['label']} (all zero bins)") # Underscore hides from legend
                        print(f"      Plotted {len(all_zero_points)} 'all zero' mean bin points.")

                    # Fit and plot LOESS on non-zero mean points (if enough data)
                    # Use 'percentile_int' as x for LOESS fitting
                    loess_plot_df = non_zero_points.dropna(subset=['percentile_int', 'mean_omega'])
                    if len(loess_plot_df) >= 5:
                        try:
                            # Sort by x value before LOESS for reliable results
                            loess_plot_df = loess_plot_df.sort_values('percentile_int')
                            loess_result = lowess(loess_plot_df['mean_omega'], loess_plot_df['percentile_int'], frac=LOESS_FRAC, it=1)
                            loess_x = loess_result[:, 0]
                            loess_y = loess_result[:, 1]

                            # Plot LOESS curve
                            ax.plot(loess_x, loess_y, color=style['color'], linestyle=style['linestyle'], label=style['label'], linewidth=3.0) # Keep label for LOESS line

                            # Interpolate SEM to match LOESS x-points for smooth shading
                            valid_sem_points = loess_plot_df.dropna(subset=['sem_omega'])
                            if len(valid_sem_points) > 1:
                                interp_sem = np.interp(loess_x, valid_sem_points['percentile_int'], valid_sem_points['sem_omega'])
                                ax.fill_between(loess_x, loess_y - interp_sem, loess_y + interp_sem, color=style['color'], alpha=0.10)
                                print(f"      Plotted LOESS and SEM shading for {style['label']}.")
                            else:
                                print(f"      Skipping SEM shading for {style['label']} due to insufficient points with valid SEM.")

                            plot_data_generated = True

                        except Exception as e:
                            print(f"      Error during LOESS fitting/plotting for {style['label']}: {e}")
                    else:
                         print(f"      Skipping LOESS for {style['label']} due to insufficient non-zero points ({len(loess_plot_df)}).")
                else:
                     print(f"    No bins remaining after percentile filter for {style['label']}.")
            else:
                 print(f"    Skipping plotting for {style['label']} due to empty binned stats or missing columns.")


        if plot_data_generated:
            # Update axis labels
            ax.set_xlabel(f"Percentile of Per-Sequence Median $\omega$ (within group)", fontsize=14)
            ax.set_ylabel("Median Per-Sequence $\omega$ (averaged within percentile bin)", fontsize=14)
            ax.set_title("Distribution of Per-Sequence Median $\omega$ (Top Percentiles)", fontsize=16)

            # Set x-axis limits and ticks
            ax.set_xlim(PLOT_X_MIN_PERCENTILE - 0.5, N_PERCENTILE_BINS - 0.5) # Show from specified percentile
            # Set integer ticks, potentially skipping some
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10)) # Adjust nbins for desired tick density

            ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_ylim(bottom=-0.01)

            plot_out_fname = PLOTS_DIR / "omega_percentile_distribution.png"
            print(f"\n  Saving percentile plot to {plot_out_fname}")
            try:
                plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust right margin
                plt.savefig(plot_out_fname, dpi=300)
            except Exception as e:
                print(f"  Error saving plot: {e}")
            plt.close(fig)
        else:
            print("\n  Skipping plot generation as no data was prepared or plotted.")
    elif not STATSMODELS_AVAILABLE:
        print("\n Skipping plot generation because 'statsmodels' package is not installed.")
    else:
        print("\n  Skipping plot generation: Insufficient aggregated raw dN/dS data.")

    print("\n" + "="*40)
    print("--- Summary, Testing, and Plotting Complete ---")


# --- Main Execution ---
if __name__ == "__main__":
    summarize_and_test_dnds_effects()
