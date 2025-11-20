import pandas as pd
import numpy as np
import re
import os
from scipy import stats
from tqdm.auto import tqdm
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
# No ThreadPoolExecutor needed

# Suppress potential warnings
warnings.filterwarnings('ignore')
# Apply tqdm integration with pandas
tqdm.pandas()

# =====================================================================
# CONFIGURATION PARAMETERS
# =====================================================================
INPUT_CSV = "all_pairwise_results.csv"
OUTPUT_DIR = "results"
PLOT_DIR = "plots"
FULL_RESULTS_FILENAME = "proportion_identity_transcript_results.csv"    
SIGNIFICANT_TABLE_FILENAME = "significant_proportion_transcript_summary.csv" 
PROPORTION_PLOT_FILENAME = "gene_identity_proportions_transcript_plot.png" 
FDR_THRESHOLD = 0.05
MIN_VALID_PAIRS_PER_GROUP = 10 # Applies to the AGGREGATED counts per transcript ID
OMEGA_IDENTICAL_VALUE = -1.0
OMEGA_IGNORE_VALUE = 99.0
PSEUDOCOUNT = 1e-6
MAX_GENES_TO_LABEL_PLOT = 50

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def extract_transcript_id(cds_identifier):
    """Extracts the Ensembl Transcript ID (ENST...) from the CDS string."""
    if pd.isna(cds_identifier): return None
    # Regex to find ENST followed by digits, optionally with a version (.digit)
    match = re.search(r'(ENST\d+(\.\d+)?)', str(cds_identifier))
    # Return the full match (e.g., ENST00000396410.9) if found
    return match.group(1) if match else None

def extract_gene_name_from_cds(cds_identifier):
    """Extracts the gene name (part before the first underscore) from the CDS string."""
    if pd.isna(cds_identifier): return "UnknownGene"
    cds_str = str(cds_identifier)
    return cds_str.split('_')[0] if '_' in cds_str else cds_str # Return full string if no underscore

def apply_fdr_correction(df, p_value_col='p_value', corrected_col='corrected_p_value'):
    """Applies Benjamini-Hochberg FDR correction."""
    df_copy = df.copy()
    df_copy[corrected_col] = np.nan
    # Filter out NaN p-values *before* correction
    valid_results = df_copy[df_copy[p_value_col].notna() & (df_copy[p_value_col] >= 0)].copy() # Allow p=0 edge case if needed, though Fisher usually > 0
    num_valid_tests = len(valid_results)
    if num_valid_tests > 0:
        valid_results = valid_results.sort_values(p_value_col)
        valid_results['rank'] = np.arange(1, num_valid_tests + 1)
        # Calculate corrected p-value
        corrected_p = valid_results[p_value_col] * num_valid_tests / valid_results['rank']
        # Enforce monotonicity (step-up procedure)
        corrected_p = corrected_p.iloc[::-1].cummin().iloc[::-1]
        # Clip at 1.0
        corrected_p = corrected_p.clip(upper=1.0)
        # Assign back using index
        df_copy.loc[valid_results.index, corrected_col] = corrected_p
    return df_copy


def plot_proportions(results_df, output_path):
    """Creates an improved grouped bar plot based on Transcript ID results, labeled by Gene Name."""
    print(f"\nGenerating plot...")
    # Filter for transcripts where the test was actually performed and successful
    plot_df = results_df[results_df['test_status'] == 'Success'].copy()
    if plot_df.empty:
        print("  No data available for plotting (no tests were successful).")
        return

    # Sort for plotting order (most significant transcript first)
    plot_df['neg_log_p'] = -np.log10(plot_df['corrected_p_value'].clip(lower=1e-300)) # Avoid log(0)
    # Sort primarily by significance, secondarily by gene name for grouping duplicates
    plot_df = plot_df.sort_values(by=['neg_log_p', 'gene_name'], ascending=[False, True])

    n_transcripts_plot = len(plot_df)
    print(f"  Plotting results for {n_transcripts_plot} tested transcripts.")

    # --- Plotting Setup ---
    fig_height = max(8, n_transcripts_plot * 0.28) # Dynamic height
    fig_width = 14 # Wider for labels
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.style.use('seaborn-v0_8-white')

    bar_height = 0.38
    index = np.arange(n_transcripts_plot) # One position per transcript

    prop_direct = plot_df['prop_identical_direct']
    prop_inverted = plot_df['prop_identical_inverted']
    is_significant = plot_df['corrected_p_value'] < FDR_THRESHOLD

    # Colors - Distinct, colorblind-friendly potentially
    color_direct_nonsig = '#88CCEE' # Light Blue
    color_inverted_nonsig = '#FDB462' # Light Orange/Yellow
    color_direct_sig = '#332288' # Indigo
    color_inverted_sig = '#CC6677' # Rose

    # --- Create Bars ---
    ax.barh(index + bar_height/2, prop_inverted, bar_height,
            color=[color_inverted_sig if sig else color_inverted_nonsig for sig in is_significant],
            edgecolor='black', linewidth=0.6, label='Inverted')
    ax.barh(index - bar_height/2, prop_direct, bar_height,
            color=[color_direct_sig if sig else color_direct_nonsig for sig in is_significant],
            edgecolor='black', linewidth=0.6, label='Direct')

    # --- Labels and Title ---
    ax.set_xlabel('Proportion of Identical Pairs (ω = -1)', fontsize=16, labelpad=10)
    ax.set_ylabel('Gene Name', fontsize=16, labelpad=10) # Label axis conceptually as Gene Name
    ax.set_title(f'Proportion of Identical Sequence Pairs within Groups\n(Analysis per Transcript, N = {n_transcripts_plot})', fontsize=18, fontweight='bold', pad=20)

    # --- Y-axis Ticks and Labels ---
    # Labels are the extracted gene names. Duplicate labels are expected.
    labels = plot_df['gene_name'].tolist()

    if n_transcripts_plot > MAX_GENES_TO_LABEL_PLOT:
        step = math.ceil(n_transcripts_plot / MAX_GENES_TO_LABEL_PLOT)
        tick_indices = list(range(0, n_transcripts_plot, step))
        if n_transcripts_plot - 1 not in tick_indices: tick_indices.append(n_transcripts_plot - 1)
        tick_labels = [labels[i] if i in tick_indices else "" for i in index]
        print(f"  Labeling approximately {len(tick_indices)} of {n_transcripts_plot} transcripts on y-axis.")
    else:
        tick_labels = labels
    ax.set_yticks(index)
    ax.set_yticklabels(tick_labels, fontsize=13) # Slightly larger font for names
    ax.invert_yaxis() # Most significant transcript at top

    # --- Legend ---
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc=color_direct_nonsig, ec='black', lw=0.6, label='Direct (Non-sig.)'),
        plt.Rectangle((0,0),1,1, fc=color_inverted_nonsig, ec='black', lw=0.6, label='Inverted (Non-sig.)'),
        plt.Rectangle((0,0),1,1, fc=color_direct_sig, ec='black', lw=0.6, label=f'Direct (FDR<{FDR_THRESHOLD})'),
        plt.Rectangle((0,0),1,1, fc=color_inverted_sig, ec='black', lw=0.6, label=f'Inverted (FDR<{FDR_THRESHOLD})')
        ]
    ax.legend(handles=legend_elements, title="Group & Significance",
              loc='center left', bbox_to_anchor=(1.02, 0.5), # Place legend outside plot area
              fontsize=13, title_fontsize=14, frameon=True, facecolor='white', framealpha=0.9)

    # --- Styling and Saving ---
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0)) # Format x-axis as percentage
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_axisbelow(True)
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0) # Remove y-axis ticks

    plt.subplots_adjust(right=0.85) # Make space for the external legend

    try:
        plt.savefig(output_path, dpi=250, bbox_inches='tight') # Higher DPI
        print(f"  Plot saved successfully to {output_path}")
    except Exception as e: print(f"  ERROR: Failed to save plot: {e}")
    plt.close(fig)

# =====================================================================
# MAIN ANALYSIS FUNCTION
# =====================================================================

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"Analysis started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # Print config... (omitted for brevity)

    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_DIR, FULL_RESULTS_FILENAME)
    sig_table_path = os.path.join(OUTPUT_DIR, SIGNIFICANT_TABLE_FILENAME)
    plot_output_path = os.path.join(PLOT_DIR, PROPORTION_PLOT_FILENAME)

    # --- Load and Preprocess ---
    print(f"\nLoading and filtering data from {INPUT_CSV}...")
    try: df = pd.read_csv(INPUT_CSV)
    except Exception as e: print(f"ERROR loading CSV: {e}"); return

    df['omega'] = pd.to_numeric(df['omega'], errors='coerce')
    initial_rows = len(df)
    df_filtered = df[df['Group1'] == df['Group2']].copy()
    df_filtered = df_filtered[df_filtered['omega'] != OMEGA_IGNORE_VALUE]
    df_filtered = df_filtered.dropna(subset=['omega'])
    print(f"  Data filtered from {initial_rows} to {len(df_filtered)} rows.")
    if df_filtered.empty: print("ERROR: No valid data remains."); return

    print("  Extracting identifiers and identity flag...")
    df_filtered['transcript_id'] = df_filtered['CDS'].progress_apply(extract_transcript_id)
    # Drop rows where transcript ID could not be extracted
    df_filtered = df_filtered.dropna(subset=['transcript_id'])
    df_filtered['gene_name'] = df_filtered['CDS'].progress_apply(extract_gene_name_from_cds)
    df_filtered['is_identical'] = (df_filtered['omega'] == OMEGA_IDENTICAL_VALUE)
    unique_transcripts = df_filtered['transcript_id'].nunique()
    print(f"  Found {unique_transcripts} unique transcript IDs.")

    # --- Aggregate Counts by Transcript ID ---
    print(f"\nAggregating pair counts per Transcript ID...")
    agg_data = []
    # Use pandas groupby for potentially faster aggregation
    grouped_by_transcript = df_filtered.groupby('transcript_id')

    for transcript_id, transcript_df in tqdm(grouped_by_transcript, desc="Aggregating Counts", total=unique_transcripts):
        # Get the gene name (should be consistent within the transcript group)
        gene_name = transcript_df['gene_name'].iloc[0] if not transcript_df.empty else "UnknownGene"

        direct_pairs = transcript_df[transcript_df['Group1'] == 0]
        inverted_pairs = transcript_df[transcript_df['Group1'] == 1]

        n_total_0 = len(direct_pairs)
        n_id_0 = direct_pairs['is_identical'].sum()
        n_total_1 = len(inverted_pairs)
        n_id_1 = inverted_pairs['is_identical'].sum()

        agg_data.append({
            'transcript_id': transcript_id,
            'gene_name': gene_name,
            'n_identical_direct': n_id_0, 'n_total_direct': n_total_0,
            'n_identical_inverted': n_id_1, 'n_total_inverted': n_total_1
        })

    if not agg_data: print("ERROR: No data after aggregation."); return
    agg_df = pd.DataFrame(agg_data)
    print(f"  Aggregated data for {len(agg_df)} transcripts.")

    # --- Perform Fisher Test on Aggregated Data ---
    print("\nPerforming Fisher's Exact Test per transcript...")
    results_list = []
    for _, row in tqdm(agg_df.iterrows(), total=len(agg_df), desc="Testing Transcripts"):
        n_id_0, n_total_0 = row['n_identical_direct'], row['n_total_direct']
        n_id_1, n_total_1 = row['n_identical_inverted'], row['n_total_inverted']

        prop_id_0 = (n_id_0 / n_total_0) if n_total_0 > 0 else 0.0
        prop_id_1 = (n_id_1 / n_total_1) if n_total_1 > 0 else 0.0

        p_value, test_status, failure_reason = np.nan, "Not Tested", None
        if n_total_0 >= MIN_VALID_PAIRS_PER_GROUP and n_total_1 >= MIN_VALID_PAIRS_PER_GROUP:
            # Ensure counts are integers for Fisher's test
            table = [[int(n_id_0), int(n_total_0 - n_id_0)],
                     [int(n_id_1), int(n_total_1 - n_id_1)]]
            try:
                if any(x < 0 for sublist in table for x in sublist): raise ValueError("Counts cannot be negative.")
                _, p_value = stats.fisher_exact(table, alternative='two-sided') # Ensure two-sided test
                test_status = "Success"
            except ValueError as e: test_status, failure_reason = "Fisher Error", f"Fisher test error: {e}"
        else:
            test_status = "Skipped"
            failure_reason = f"Insufficient pairs: Dir({n_total_0}) Inv({n_total_1}) < {MIN_VALID_PAIRS_PER_GROUP}"

        results_list.append({
            'transcript_id': row['transcript_id'],
            'gene_name': row['gene_name'],
            'n_identical_direct': n_id_0, 'n_total_direct': n_total_0, 'prop_identical_direct': prop_id_0,
            'n_identical_inverted': n_id_1, 'n_total_inverted': n_total_1, 'prop_identical_inverted': prop_id_1,
            'p_value': p_value, 'test_status': test_status, 'failure_reason': failure_reason
        })

    results_df = pd.DataFrame(results_list)

    # --- Post-Processing ---
    print("\nFinalizing results...")
    print("  Calculating fold difference...")
    results_df['fold_difference'] = (results_df['prop_identical_inverted'] + PSEUDOCOUNT) / \
                                    (results_df['prop_identical_direct'] + PSEUDOCOUNT)
    print("  Applying FDR correction...")
    results_df = apply_fdr_correction(results_df, 'p_value', 'corrected_p_value')

    # --- Save Full Results ---
    try:
        results_df.to_csv(full_output_path, index=False, float_format='%.6g')
        print(f"  Full results saved to {full_output_path}")
    except Exception as e: print(f"  ERROR saving full results: {e}")

    # --- Create and Save Significant Summary Table ---
    print("  Creating significant summary table...")
    significant_df = results_df[results_df['corrected_p_value'] < FDR_THRESHOLD].copy()
    summary_table = significant_df.sort_values(by='corrected_p_value', ascending=True)
    summary_cols = ['transcript_id', 'gene_name', 'prop_identical_direct', 'prop_identical_inverted',
                    'fold_difference', 'corrected_p_value'] # Rearranged slightly
    summary_cols = [col for col in summary_cols if col in summary_table.columns]
    summary_table = summary_table[summary_cols]
    if not summary_table.empty:
        try:
            summary_table.to_csv(sig_table_path, index=False, float_format='%.4g')
            print(f"  Significant summary table ({len(summary_table)} transcripts) saved to {sig_table_path}")
        except Exception as e: print(f"  ERROR saving summary table: {e}")
    else: print("  No significant results found for summary table.")

    # --- Generate Plot ---
    plot_proportions(results_df, plot_output_path)

    # --- Final Summary ---
    print("\n--- Analysis Summary ---")
    total_transcripts_processed = len(results_df)
    tested_transcripts = results_df[results_df['test_status'] == 'Success'].shape[0]
    skipped_transcripts = results_df[results_df['test_status'] == 'Skipped'].shape[0]
    error_transcripts = results_df[results_df['test_status'] == 'Fisher Error'].shape[0]
    significant_transcripts_count = len(summary_table)

    print(f"Total unique transcript IDs processed: {total_transcripts_processed}")
    print(f"  Transcripts tested (sufficient pairs): {tested_transcripts}")
    print(f"  Transcripts skipped (insufficient pairs): {skipped_transcripts}")
    if error_transcripts > 0: print(f"  Transcripts with Fisher test errors: {error_transcripts}")
    print(f"Significant differences found (FDR < {FDR_THRESHOLD}): {significant_transcripts_count}")

    end_time = datetime.now()
    print("-" * 30); print(f"Analysis finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}"); print(f"Total runtime: {end_time - start_time}")

# =====================================================================
if __name__ == "__main__":
    main()
