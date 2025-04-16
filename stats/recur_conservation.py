import pandas as pd
import numpy as np
import re
import logging
import sys
from scipy import stats
from collections import defaultdict

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', # timestamp
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('conservation_analysis')

# File paths
PAIRWISE_FILE = 'all_pairwise_results.csv'
INVERSION_FILE = 'inv_info.csv'
OUTPUT_RESULTS_LOO_MEDIAN = 'leave_one_out_median_results.csv' # Specific output name

# --- Helper Functions ---

def extract_coordinates_from_cds(cds_string):
    """Extract genomic coordinates from CDS string."""
    if not isinstance(cds_string, str):
        return None
    pattern = r'(chr\w+)_start(\d+)_end(\d+)'
    match = re.search(pattern, cds_string)
    return {
        'chrom': match.group(1),
        'start': int(match.group(2)),
        'end': int(match.group(3))
    } if match else None

def map_cds_to_inversions(pairwise_df, inversion_df):
    """Map CDS strings to specific inversions and track inversion types."""
    logger.info("Mapping CDS to inversion types...")
    unique_cds = pairwise_df['CDS'].dropna().unique()

    cds_coords = {}
    for cds in unique_cds:
         coords = extract_coordinates_from_cds(cds)
         if coords:
             cds_coords[cds] = coords
         else:
             logger.warning(f"Could not parse coordinates from CDS string: {cds}")

    #  inversion coordinates are numeric and handle potential errors
    for col in ['Start', 'End', '0_single_1_recur']:
        inversion_df[col] = pd.to_numeric(inversion_df[col], errors='coerce')
    inversion_df.dropna(subset=['Start', 'End', '0_single_1_recur', 'Chromosome'], inplace=True)
    inversion_df['Start'] = inversion_df['Start'].astype(int)
    inversion_df['End'] = inversion_df['End'].astype(int)
    inversion_df['0_single_1_recur'] = inversion_df['0_single_1_recur'].astype(int)

    recurrent_inv = inversion_df[inversion_df['0_single_1_recur'] == 1]
    single_event_inv = inversion_df[inversion_df['0_single_1_recur'] == 0]

    cds_to_type = {}
    cds_to_inversion_id = {}
    inversion_to_cds = defaultdict(list)

    processed_cds_count = 0
    for cds, coords in cds_coords.items():
        chrom, start, end = coords['chrom'], coords['start'], coords['end']

        rec_matches = recurrent_inv[
            (recurrent_inv['Chromosome'] == chrom) &
            (start <= recurrent_inv['End']) &
            (end >= recurrent_inv['Start'])
        ]

        single_matches = single_event_inv[
            (single_event_inv['Chromosome'] == chrom) &
            (start <= single_event_inv['End']) &
            (end >= single_event_inv['Start'])
        ]

        is_recurrent = len(rec_matches) > 0
        is_single = len(single_matches) > 0

        if is_recurrent and not is_single:
            cds_to_type[cds] = 'recurrent'
            inv_ids = rec_matches.index.tolist()
            cds_to_inversion_id[cds] = inv_ids
            for inv_id in inv_ids:
                inversion_to_cds[inv_id].append(cds)
        elif is_single and not is_recurrent:
            cds_to_type[cds] = 'single_event'
            inv_ids = single_matches.index.tolist()
            cds_to_inversion_id[cds] = inv_ids
            for inv_id in inv_ids:
                inversion_to_cds[inv_id].append(cds)
        elif is_recurrent and is_single:
            cds_to_type[cds] = 'ambiguous'
            inv_ids = rec_matches.index.tolist() + single_matches.index.tolist()
            cds_to_inversion_id[cds] = inv_ids
            for inv_id in inv_ids:
                 inversion_to_cds[inv_id].append(cds)
        else:
            cds_to_type[cds] = 'unknown'
            cds_to_inversion_id[cds] = []

        processed_cds_count += 1
        if processed_cds_count % 500 == 0:
             logger.info(f"  Processed {processed_cds_count}/{len(cds_coords)} CDS for mapping...")

    logger.info(f"Finished mapping {len(cds_to_type)} CDS to types.")
    type_counts = pd.Series(cds_to_type).value_counts()
    logger.info(f"  Type counts: {type_counts.to_dict()}")

    # Convert defaultdict back to dict for consistent return type
    inversion_to_cds = dict(inversion_to_cds)
    return cds_to_type, cds_to_inversion_id, inversion_to_cds

# --- Analysis 1: RAW Pairwise Omega Values (Identical vs Non-Identical) ---

def analyze_raw_pair_proportions(pairwise_df, cds_to_type):
    """
    Compares proportion of identical pairs (omega == -1) using RAW pairwise data
    between recurrent and single-event inversions.
    """
    logger.info("\n--- Starting Raw Pairwise Proportion Analysis ---")

    # 1. Prepare Data
    pairwise_df['inversion_type'] = pairwise_df['CDS'].map(cds_to_type)

    relevant_pairs_df = pairwise_df[
        pairwise_df['inversion_type'].isin(['recurrent', 'single_event']) &
        (pairwise_df['Group1'] == 1) &
        (pairwise_df['Group2'] == 1) &
        pd.notna(pairwise_df['omega']) &
        (pairwise_df['omega'] != 99)
    ].copy()

    if relevant_pairs_df.empty:
        logger.warning("No relevant pairs found for raw pairwise analysis after filtering.")
        print("\n--- RAW PAIR PROPORTION ANALYSIS ---")
        print("  No data available for comparison.")
        print("------------------------------------")
        return

    logger.info(f"Found {len(relevant_pairs_df):,} relevant pairs for raw analysis.")

    # 2. Get Counts for the 2x2 Table
    rec_pairs = relevant_pairs_df[relevant_pairs_df['inversion_type'] == 'recurrent']
    single_pairs = relevant_pairs_df[relevant_pairs_df['inversion_type'] == 'single_event']

    rec_total_raw = len(rec_pairs)
    rec_identical_raw = len(rec_pairs[rec_pairs['omega'] == -1])
    rec_non_identical_raw = rec_total_raw - rec_identical_raw

    single_total_raw = len(single_pairs)
    single_identical_raw = len(single_pairs[single_pairs['omega'] == -1])
    single_non_identical_raw = single_total_raw - single_identical_raw

    table_raw = [
        [rec_identical_raw, rec_non_identical_raw],
        [single_identical_raw, single_non_identical_raw]
    ]

    # 3. Perform Statistical Test & Report Results
    print("\n--- RAW PAIR PROPORTION ANALYSIS (Fisher's Exact Test) ---")
    print(f"Comparing proportion of identical pairs (omega == -1) using RAW pairwise data:")

    odds_ratio_raw, p_value_raw = np.nan, 1.0

    if rec_total_raw > 0 and single_total_raw > 0:
        rec_pct_identical_raw = (rec_identical_raw / rec_total_raw) * 100
        single_pct_identical_raw = (single_identical_raw / single_total_raw) * 100
        rec_pct_non_identical_raw = 100.0 - rec_pct_identical_raw
        single_pct_non_identical_raw = 100.0 - single_pct_identical_raw

        print(f"\n  Overall Counts & Proportions:")
        print(f"    Recurrent Pairs:    {rec_identical_raw:,} / {rec_total_raw:,} ({rec_pct_identical_raw:.2f}%) identical")
        print(f"    Single-Event Pairs: {single_identical_raw:,} / {single_total_raw:,} ({single_pct_identical_raw:.2f}%) identical")
        print(f"    (Non-Identical: Rec={rec_pct_non_identical_raw:.2f}%, Single={single_pct_non_identical_raw:.2f}%)")

        try:
            if (rec_total_raw == rec_identical_raw and single_total_raw == single_identical_raw) or \
               (rec_identical_raw == 0 and single_identical_raw == 0):
                 logger.warning("Fisher's test skipped for raw pairs: No variation in identity across groups.")
                 odds_ratio_raw, p_value_raw = np.nan, 1.0
            else:
                odds_ratio_raw, p_value_raw = stats.fisher_exact(table_raw)

            print(f"\n  Fisher's Exact Test Results:")
            print(f"    Odds Ratio (Identical, Single vs Recurrent): {odds_ratio_raw:.4f}" if not np.isnan(odds_ratio_raw) else "    Odds Ratio: N/A")
            print(f"    P-value: {p_value_raw:.4e}" if not np.isnan(p_value_raw) else "    P-value: N/A")

        except ValueError as e:
             logger.error(f"Fisher's exact test failed for raw pairs: {e}")
             print(f"\n  Fisher's Exact Test Results:")
             print(f"    Error during calculation: {e}")
             print(f"    Table causing error: {table_raw}")

        print(f"\n  Fold Change Calculations:")
        # Fold Change for NON-IDENTICAL pairs (Recurrent / Single)
        if single_pct_non_identical_raw > 1e-9: # Denominator check
            fold_change_non_identical_rec_vs_single = rec_pct_non_identical_raw / single_pct_non_identical_raw
            print(f"    Fold Change (Non-Identical %, Recurrent / Single): {fold_change_non_identical_rec_vs_single:.1f}x")
        elif rec_pct_non_identical_raw > 1e-9:
             print(f"    Fold Change (Non-Identical %, Recurrent / Single): Infinite (Single % non-identical is zero)")
        else:
             print(f"    Fold Change (Non-Identical %, Recurrent / Single): Undefined (Both groups have 0% non-identical)")


        alpha = 0.05
        print(f"\n  Conclusion (alpha={alpha}):")
        if not np.isnan(p_value_raw):
            if p_value_raw < alpha:
                print(f"    Significant difference detected between groups based on raw pair proportions.")
                direction = "higher" if single_pct_identical_raw > rec_pct_identical_raw else "lower"
                print(f"    Single-Event pairs have a significantly {direction} proportion of identical sequences.")
            else:
                print(f"    NO significant difference detected between groups based on raw pair proportions.")
        else:
            print(f"    Significance could not be determined due to test error or lack of variation.")

    else:
        print("\n  Skipping raw pair proportion analysis: Not enough data in one or both groups after filtering.")
        print(f"    Recurrent Pairs Count: {rec_total_raw:,}")
        print(f"    Single-Event Pairs Count: {single_total_raw:,}")

    print("----------------------------------------------------------")
    logger.info("--- Finished Raw Pairwise Proportion Analysis ---")

# --- Analysis 2: MEDIAN Omega per Sequence (Leave-One-Out) ---

def calculate_sequence_median_omega(pairwise_df):
    """Calculate median omega value for each unique sequence in each CDS."""
    logger.info("Calculating median omega per sequence per CDS...")
    group1_df = pairwise_df[
        (pairwise_df['Group1'] == 1) &
        (pairwise_df['Group2'] == 1) &
        pd.notna(pairwise_df['omega']) &
        (pairwise_df['omega'] != 99)
    ].copy()

    if group1_df.empty:
        logger.warning("No valid Group1 pairs found for median omega calculation.")
        return pd.DataFrame(columns=['CDS', 'Sequence', 'median_omega'])

    seq1_df = group1_df.dropna(subset=['Seq1'])[['CDS', 'Seq1', 'omega']].rename(columns={'Seq1': 'Sequence'})
    seq2_df = group1_df.dropna(subset=['Seq2'])[['CDS', 'Seq2', 'omega']].rename(columns={'Seq2': 'Sequence'})

    all_seqs_df = pd.concat([seq1_df, seq2_df], ignore_index=True)
    logger.info(f"  Total sequence entries for median calculation: {len(all_seqs_df):,}")

    if all_seqs_df.empty:
         logger.warning("No sequences available after combining Seq1 and Seq2.")
         return pd.DataFrame(columns=['CDS', 'Sequence', 'median_omega'])

    try:
        median_df = all_seqs_df.groupby(['CDS', 'Sequence'])['omega'].median().reset_index()
        median_df.rename(columns={'omega': 'median_omega'}, inplace=True)
        logger.info(f"  Calculated medians for {len(median_df):,} unique sequence-CDS combinations.")
    except Exception as e:
        logger.error(f"Error during median calculation groupby: {e}")
        return pd.DataFrame(columns=['CDS', 'Sequence', 'median_omega'])

    return median_df

def run_statistical_test_median_based(table):
    """Run Fisher's exact test specifically for the median-based analysis."""
    try:
        #  table elements are integers
        int_table = [[int(c) for c in row] for row in table]
        odds_ratio, p_value = stats.fisher_exact(int_table)
        return {'p_value': p_value, 'odds_ratio': odds_ratio}
    except ValueError as e:
         logger.warning(f"Fisher's exact test failed for median-based table: {e}. Table: {table}")
         return {'p_value': 1.0, 'odds_ratio': np.nan} # Return non-significant, NaN OR

def analyze_proportions_median_based(df, inversion_to_cds_map):
    """
    Calculate conservation statistics (based on median omega) and run statistical tests.
    """
    stats_agg = df.groupby('inversion_type').agg(
        total_sequences=('is_identical', 'count'),
        identical_sequences=('is_identical', 'sum')
    )

    results = { # Initialize with defaults
            'recurrent_total': 0, 'recurrent_identical': 0, 'recurrent_pct': 0,
            'single_total': 0, 'single_identical': 0, 'single_pct': 0,
            'test_results': {'p_value': 1.0, 'odds_ratio': np.nan},
            'recurrent_inversions': set(), 'single_event_inversions': set()
    }

    if 'recurrent' in stats_agg.index:
        results['recurrent_total'] = stats_agg.loc['recurrent', 'total_sequences']
        results['recurrent_identical'] = stats_agg.loc['recurrent', 'identical_sequences']
        if results['recurrent_total'] > 0:
             results['recurrent_pct'] = (results['recurrent_identical'] / results['recurrent_total']) * 100

    if 'single_event' in stats_agg.index:
        results['single_total'] = stats_agg.loc['single_event', 'total_sequences']
        results['single_identical'] = stats_agg.loc['single_event', 'identical_sequences']
        if results['single_total'] > 0:
             results['single_pct'] = (results['single_identical'] / results['single_total']) * 100

    # Proceed with test only if both groups have data
    if results['recurrent_total'] > 0 and results['single_total'] > 0:
        table = [
            [results['recurrent_identical'], results['recurrent_total'] - results['recurrent_identical']],
            [results['single_identical'], results['single_total'] - results['single_identical']]
        ]
        results['test_results'] = run_statistical_test_median_based(table)

        # Identify contributing inversions (approximation based on CDS in current df subset)
        contributing_inversions = defaultdict(set)
        cds_in_df = df['CDS'].unique()
        for inv_id, cds_list in inversion_to_cds_map.items():
            if any(cds in cds_in_df for cds in cds_list):
                inv_cds_in_df = df[df['CDS'].isin(cds_list)]
                if not inv_cds_in_df.empty:
                     inv_type_mode = inv_cds_in_df['inversion_type'].mode()
                     if len(inv_type_mode) > 0:
                          dominant_type = inv_type_mode[0]
                          if dominant_type in ['recurrent', 'single_event']:
                              contributing_inversions[dominant_type].add(inv_id)
        results['recurrent_inversions'] = contributing_inversions.get('recurrent', set())
        results['single_event_inversions'] = contributing_inversions.get('single_event', set())
    else:
         logger.warning("Median-based analysis missing data for one or both groups.")

    return results

def conduct_leave_one_out_analysis(median_df, cds_to_type, inversion_to_cds_map):
    """Perform leave-one-out analysis based on median omega per sequence."""
    logger.info("\n--- Starting Leave-One-Out Analysis (Median Omega per Sequence) ---")

    if median_df.empty:
        logger.error("Median omega dataframe is empty. Cannot perform Leave-One-Out analysis.")
        print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Median Omega)")
        print("---------------------------------------------")
        print("Skipped: No median omega data available.")
        return None, np.nan

    median_df['inversion_type'] = median_df['CDS'].map(cds_to_type)

    valid_df = median_df[
        median_df['inversion_type'].isin(['recurrent', 'single_event']) &
        pd.notna(median_df['median_omega'])
    ].copy()

    if valid_df.empty:
        logger.error("No valid sequences found after filtering for median-based LOO analysis.")
        print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Median Omega)")
        print("---------------------------------------------")
        print("Skipped: No valid sequence data for recurrent/single types.")
        return None, np.nan

    valid_df['is_identical'] = (valid_df['median_omega'] == -1).astypupdatee(int)
    logger.info(f"Total sequences for LOO analysis: {len(valid_df):,}")

    baseline_stats = analyze_proportions_median_based(valid_df, inversion_to_cds_map)
    if baseline_stats['recurrent_total'] == 0 or baseline_stats['single_total'] == 0:
         logger.error("Baseline analysis lacks data for both groups. Aborting LOO.")
         print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Median Omega)")
         print("---------------------------------------------")
         print("Skipped: Baseline analysis missing data for one or both groups.")
         return None, np.nan

    inv_counts = []
    valid_cds_list = valid_df['CDS'].unique()
    for inv_id, cds_list in inversion_to_cds_map.items():
        cds_for_this_inv_in_valid_df = [cds for cds in cds_list if cds in valid_cds_list]
        if cds_for_this_inv_in_valid_df:
            inv_sequences = valid_df[valid_df['CDS'].isin(cds_for_this_inv_in_valid_df)]
            if not inv_sequences.empty:
                 inv_type_mode = inv_sequences['inversion_type'].mode()
                 inv_type = inv_type_mode[0] if len(inv_type_mode) > 0 else 'unknown'
                 inv_counts.append({
                     'inversion_id': inv_id,
                     'count': len(inv_sequences),
                     'inv_type': inv_type
                 })

    inv_counts_df = pd.DataFrame(inv_counts)
    large_inversions = inv_counts_df[
        (inv_counts_df['count'] >= 10) &
        (inv_counts_df['inv_type'].isin(['recurrent', 'single_event']))
    ].sort_values('count', ascending=False)

    logger.info(f"Identified {len(large_inversions)} large inversions (>=10 sequences) for LOO.")

    results = []
    all_p_values = []
    all_effect_sizes = []

    baseline_result = {
        'inversion_excluded': 'None',
        'inv_type': 'NA',
        'excluded_count': 0,
        'recurrent_total': baseline_stats['recurrent_total'],
        'recurrent_identical': baseline_stats['recurrent_identical'],
        'recurrent_pct': baseline_stats['recurrent_pct'],
        'single_total': baseline_stats['single_total'],
        'single_identical': baseline_stats['single_identical'],
        'single_pct': baseline_stats['single_pct'],
        'p_value': baseline_stats['test_results']['p_value'],
        'odds_ratio': baseline_stats['test_results']['odds_ratio'],
        'effect_size': baseline_stats['single_pct'] - baseline_stats['recurrent_pct'] # Diff in identical %
    }

    results.append(baseline_result)
    if not np.isnan(baseline_result['p_value']):
        all_p_values.append(baseline_result['p_value'])
    if not np.isnan(baseline_result['effect_size']):
         all_effect_sizes.append(baseline_result['effect_size'])

    loo_count = 0
    for _, row in large_inversions.iterrows():
        inv_id = row['inversion_id']
        inv_type_of_excluded = row['inv_type']
        cds_list_to_exclude = inversion_to_cds_map.get(inv_id, [])

        if not cds_list_to_exclude:
             logger.warning(f"No CDS found for inversion {inv_id} in map, skipping LOO iteration.")
             continue

        filtered_df = valid_df[~valid_df['CDS'].isin(cds_list_to_exclude)].copy()
        excluded_sequences = valid_df[valid_df['CDS'].isin(cds_list_to_exclude)]
        excluded_count = len(excluded_sequences)

        remaining_types = filtered_df['inversion_type'].nunique()
        if remaining_types < 2 or filtered_df.empty:
            logger.warning(f"Excluding inversion {inv_id} left insufficient data or types, skipping LOO iteration.")
            continue

        filtered_stats = analyze_proportions_median_based(filtered_df, inversion_to_cds_map)

        if filtered_stats['recurrent_total'] == 0 or filtered_stats['single_total'] == 0 or np.isnan(filtered_stats['test_results']['p_value']):
            logger.warning(f"Analysis after excluding inversion {inv_id} resulted in insufficient data or test failure, skipping.")
            continue

        effect_size = filtered_stats['single_pct'] - filtered_stats['recurrent_pct']

        result = {
            'inversion_excluded': inv_id,
            'inv_type': inv_type_of_excluded,
            'excluded_count': excluded_count,
            'recurrent_total': filtered_stats['recurrent_total'],
            'recurrent_identical': filtered_stats['recurrent_identical'],
            'recurrent_pct': filtered_stats['recurrent_pct'],
            'single_total': filtered_stats['single_total'],
            'single_identical': filtered_stats['single_identical'],
            'single_pct': filtered_stats['single_pct'],
            'p_value': filtered_stats['test_results']['p_value'],
            'odds_ratio': filtered_stats['test_results']['odds_ratio'],
            'effect_size': effect_size
        }

        results.append(result)
        all_p_values.append(result['p_value'])
        all_effect_sizes.append(result['effect_size'])
        loo_count += 1

    logger.info(f"Completed {loo_count} successful LOO iterations.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')

    try:
        results_df.to_csv(OUTPUT_RESULTS_LOO_MEDIAN, index=False)
        logger.info(f"Leave-one-out (median-based) results saved to {OUTPUT_RESULTS_LOO_MEDIAN}")
    except Exception as e:
        logger.error(f"Failed to save leave-one-out results: {e}")

    print("\nLEAVE-ONE-OUT ANALYSIS RESULTS (Median Omega per Sequence)")
    print("---------------------------------------------------------")
    print(f"Total large inversions considered for LOO: {len(large_inversions)}")
    print(f"Successful LOO iterations: {loo_count}")

    if not all_p_values:
        print("\nNo valid LOO results generated to summarize.")
        return results_df, np.nan

    median_p_value = np.median(all_p_values)
    max_p_value = np.max(all_p_values)
    median_effect_size = np.median(all_effect_sizes)

    print(f"\nBaseline Result (All data):")
    print(f"  P-value: {baseline_result['p_value']:.4e}")
    print(f"  Effect size (Identical % Diff, Single-Rec): {baseline_result['effect_size']:.2f}%")
    print(f"  Recurrent: {int(baseline_result['recurrent_identical']):,}/{int(baseline_result['recurrent_total']):,} ({baseline_result['recurrent_pct']:.1f}%) identical")
    print(f"  Single:    {int(baseline_result['single_identical']):,}/{int(baseline_result['single_total']):,} ({baseline_result['single_pct']:.1f}%) identical")

    print(f"\nRobust Overall Results (Median of Baseline + LOO iterations):")
    print(f"  ROBUST MEDIAN P-VALUE: {median_p_value:.4e}")
    print(f"  ROBUST MEDIAN EFFECT SIZE (Identical % Diff): {median_effect_size:.2f}%")
    print(f"  Most conservative p-value (max): {max_p_value:.4e}")

    print("\nTop 5 results by p-value (including Baseline):")
    top5 = results_df.head(5)
    for i, row in top5.iterrows():
        excluded = row['inversion_excluded']
        p_val = row['p_value']
        effect = row['effect_size']
        rec_pct = row['recurrent_pct']
        single_pct = row['single_pct']
        excluded_count_disp = int(row['excluded_count'])

        if excluded == 'None':
            print(f"  BASELINE: p={p_val:.4e}, Rec={rec_pct:.1f}%, Single={single_pct:.1f}%, Effect={effect:.2f}%")
        else:
            inv_type = row['inv_type']
            print(f"  Excl. {inv_type} Inv {excluded} ({excluded_count_disp:,} seq): p={p_val:.4e}, Rec={rec_pct:.1f}%, Single={single_pct:.1f}%, Effect={effect:.2f}%")

    print("\nMost influential inversions (largest absolute p-value change from baseline):")
    if not np.isnan(baseline_result['p_value']):
        #  p_value is numeric before subtraction
        results_df['p_value'] = pd.to_numeric(results_df['p_value'], errors='coerce')
        results_df.dropna(subset=['p_value'], inplace=True) # Drop rows where p_value became NaN

        results_df['p_change'] = results_df['p_value'] - baseline_result['p_value']
        top_influence = results_df[results_df['inversion_excluded'] != 'None'].reindex(
                results_df['p_change'].abs().sort_values(ascending=False).index
            ).head(5)

        for i, row in top_influence.iterrows():
            excluded = row['inversion_excluded']
            inv_type = row['inv_type']
            p_val = row['p_value']
            p_change = row['p_change']
            count = int(row['excluded_count'])
            effect = row['effect_size']

            change_direction = "increased" if p_change > 0 else "decreased"
            print(f"  Inv {excluded} ({inv_type}, {count:,} seq): p-value {change_direction} by {abs(p_change):.4e} to {p_val:.4e}, Effect={effect:.2f}%")
    else:
        print("  Cannot calculate influence as baseline p-value is invalid.")

    alpha = 0.05
    significance = "SIGNIFICANT difference" if median_p_value < alpha else "NO significant difference"
    more_conserved = "SINGLE-EVENT sequences show higher identical proportion" if median_effect_size > 0 else "RECURRENT sequences show higher identical proportion"
    if abs(median_effect_size) < 1e-6:
        more_conserved = "NO substantial difference in identical proportions"

    print(f"\nFINAL CONCLUSION (Median Omega LOO):")
    print(f"  There is a {significance} in the proportion of identical sequences (median omega = -1)")
    print(f"  between recurrent and single-event associated sequences (Robust Median p={median_p_value:.4e}, α={alpha}).")
    print(f"  DIRECTION: {more_conserved} (Robust Median difference of {abs(median_effect_size):.2f}%).")
    print("---------------------------------------------------------")
    logger.info("--- Finished Leave-One-Out Analysis ---")

    return results_df, median_p_value

# --- Main Execution ---

def main():
    """Main execution function."""
    logger.info("Starting Conservation Analysis Script...")

    # Load input files
    try:
        logger.info(f"Loading pairwise data from: {PAIRWISE_FILE}")
        pairwise_df = pd.read_csv(PAIRWISE_FILE)
        logger.info(f"  Loaded {len(pairwise_df):,} rows.")

        logger.info(f"Loading inversion info from: {INVERSION_FILE}")
        inversion_df = pd.read_csv(INVERSION_FILE) # Add index_col=0 if needed
        logger.info(f"  Loaded {len(inversion_df):,} inversions.")

    except FileNotFoundError as e:
        logger.error(f"Error loading input file: {e}. Please check file paths.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading input files: {e}")
        return

    # Step 1: Map CDS to Inversion Types
    # Global is used here to match previous structure, but passing map is safer
    global inversion_to_cds_map # Declare intent to use/modify global
    cds_to_type, cds_to_inversion_id, inversion_to_cds_map = map_cds_to_inversions(pairwise_df, inversion_df)

    if not cds_to_type:
         logger.error("CDS to type mapping failed or resulted in no mappings. Exiting.")
         return

    # Step 2: Analyze Raw Pairwise Proportions
    analyze_raw_pair_proportions(pairwise_df.copy(), cds_to_type)

    # Step 3: Calculate Median Omega per Sequence
    median_df = calculate_sequence_median_omega(pairwise_df)

    # Step 4: Conduct Leave-One-Out Analysis (Based on Median Omega)
    loo_results_df, final_median_loo_p_value = conduct_leave_one_out_analysis(
        median_df.copy(), cds_to_type, inversion_to_cds_map # Pass map explicitly
    )

    logger.info("\n=== Analysis Summary ===")
    logger.info("1. Raw Pairwise Proportion Analysis: Results printed above.")
    if loo_results_df is not None:
        logger.info(f"2. Leave-One-Out Analysis (Median Omega): Results saved to {OUTPUT_RESULTS_LOO_MEDIAN}")
        logger.info(f"   Robust Median P-value (LOO Median): {final_median_loo_p_value:.4e}")
    else:
        logger.info("2. Leave-One-Out Analysis (Median Omega): Skipped or failed.")

    logger.info("Script finished.")


inversion_to_cds_map = {}

if __name__ == "__main__":
    main()
