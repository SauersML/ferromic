import pandas as pd
import numpy as np
import re
import logging
import sys
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import Table2x2

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('conservation_analysis')

# File paths
PAIRWISE_FILE = 'all_pairwise_results.csv'
INVERSION_FILE = 'inv_info.csv'
OUTPUT_RESULTS = 'leave_one_out_results.csv'

def extract_coordinates_from_cds(cds_string):
    """Extract genomic coordinates from CDS string."""
    pattern = r'(chr\w+)_start(\d+)_end(\d+)'
    match = re.search(pattern, cds_string)
    return {
        'chrom': match.group(1),
        'start': int(match.group(2)),
        'end': int(match.group(3))
    } if match else None

def map_cds_to_inversions(pairwise_df, inversion_df):
    """Map CDS strings to specific inversions and track which inversion each belongs to."""
    # Get unique CDS strings and extract coordinates
    unique_cds = pairwise_df['CDS'].unique()
    
    cds_coords = {cds: coords for cds in unique_cds 
                 if (coords := extract_coordinates_from_cds(cds))}
    
    # Split inversions by type
    recurrent_inv = inversion_df[inversion_df['0_single_1_recur'] == 1]
    single_event_inv = inversion_df[inversion_df['0_single_1_recur'] == 0]
    
    # Map CDS to inversion types and track specific inversions
    cds_to_type = {}
    cds_to_inversion_id = {}
    inversion_to_cds = {inv_id: [] for inv_id in inversion_df.index}
    
    for cds, coords in cds_coords.items():
        chrom, start, end = coords['chrom'], coords['start'], coords['end']
        
        # Find matching inversions
        rec_matches = recurrent_inv[
            (recurrent_inv['Chromosome'] == chrom) &
            ((start <= recurrent_inv['End']) & (end >= recurrent_inv['Start']))
        ]
        
        single_matches = single_event_inv[
            (single_event_inv['Chromosome'] == chrom) &
            ((start <= single_event_inv['End']) & (end >= single_event_inv['Start']))
        ]
        
        # Classify CDS
        if len(rec_matches) > 0 and len(single_matches) == 0:
            cds_to_type[cds] = 'recurrent'
            cds_to_inversion_id[cds] = rec_matches.index.tolist()
            for inv_id in rec_matches.index:
                inversion_to_cds[inv_id].append(cds)
        elif len(single_matches) > 0 and len(rec_matches) == 0:
            cds_to_type[cds] = 'single_event'
            cds_to_inversion_id[cds] = single_matches.index.tolist()
            for inv_id in single_matches.index:
                inversion_to_cds[inv_id].append(cds)
        elif len(rec_matches) > 0 and len(single_matches) > 0:
            cds_to_type[cds] = 'ambiguous'
            cds_to_inversion_id[cds] = rec_matches.index.tolist() + single_matches.index.tolist()
            for inv_id in list(rec_matches.index) + list(single_matches.index):
                inversion_to_cds[inv_id].append(cds)
        else:
            cds_to_type[cds] = 'unknown'
            cds_to_inversion_id[cds] = []
    
    # Clean up empty inversions
    inversion_to_cds = {k: v for k, v in inversion_to_cds.items() if v}
    
    return cds_to_type, cds_to_inversion_id, inversion_to_cds

def calculate_sequence_median_omega(pairwise_df):
    """Calculate median omega value for each unique sequence in each CDS."""
    # Filter for Group 1
    group1_df = pairwise_df[(pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1)]
    
    # Create sequence-omega mappings
    # First, create a dataframe with all Seq1 occurrences
    seq1_df = group1_df[group1_df['omega'] != 99][['CDS', 'Seq1', 'omega']].rename(columns={'Seq1': 'Sequence'})
    
    # Then create a dataframe with all Seq2 occurrences
    seq2_df = group1_df[group1_df['omega'] != 99][['CDS', 'Seq2', 'omega']].rename(columns={'Seq2': 'Sequence'})
    
    # Combine both dataframes
    all_seqs_df = pd.concat([seq1_df, seq2_df])
    
    # Group by CDS and Sequence, then calculate median omega
    median_df = all_seqs_df.groupby(['CDS', 'Sequence'])['omega'].median().reset_index()
    median_df.rename(columns={'omega': 'median_omega'}, inplace=True)
    
    return median_df

def run_statistical_test(table):
    """Run Fisher's exact test"""
    odds_ratio, p_value = stats.fisher_exact(table)
    
    return {
        'p_value': p_value,
        'odds_ratio': odds_ratio
    }

def conduct_leave_one_out_analysis(median_df, cds_to_type, inversion_to_cds):
    """Perform leave-one-out analysis for each inversion."""
    # Add inversion type to median df
    median_df['inversion_type'] = median_df['CDS'].map(cds_to_type)
    
    # Filter for valid types
    valid_df = median_df[median_df['inversion_type'].isin(['recurrent', 'single_event'])].copy()
    
    # Create binary conservation indicator
    valid_df['is_identical'] = (valid_df['median_omega'] == -1).astype(int)
    
    # Baseline analysis with all inversions
    baseline_stats = analyze_proportions(valid_df)
    
    # Find inversions with enough sequences for meaningful analysis
    inv_counts = []
    for inv_id, cds_list in inversion_to_cds.items():
        inv_sequences = valid_df[valid_df['CDS'].isin(cds_list)]
        inv_counts.append({
            'inversion_id': inv_id, 
            'count': len(inv_sequences),
            'inv_type': 'recurrent' if inv_id in baseline_stats['recurrent_inversions'] else 'single_event'
        })
    
    inv_counts_df = pd.DataFrame(inv_counts)
    large_inversions = inv_counts_df[inv_counts_df['count'] >= 10].sort_values('count', ascending=False)
    
    # Perform leave-one-out analysis
    results = []
    all_p_values = []
    all_effect_sizes = []
    
    # Baseline result
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
        'effect_size': baseline_stats['single_pct'] - baseline_stats['recurrent_pct']
    }
    
    results.append(baseline_result)
    all_p_values.append(baseline_stats['test_results']['p_value'])
    all_effect_sizes.append(baseline_result['effect_size'])
    
    # Leave-one-out for each large inversion
    for _, row in large_inversions.iterrows():
        inv_id = row['inversion_id']
        inv_type = row['inv_type']
        
        # Get CDS list for this inversion
        cds_list = inversion_to_cds[inv_id]
        
        # Create filtered dataframe excluding this inversion
        filtered_df = valid_df[~valid_df['CDS'].isin(cds_list)].copy()
        
        # Get stats for filtered data
        filtered_stats = analyze_proportions(filtered_df)
        
        # Count excluded sequences
        excluded_count = valid_df[valid_df['CDS'].isin(cds_list)].shape[0]
        
        # Calculate effect size (difference in percentages)
        effect_size = filtered_stats['single_pct'] - filtered_stats['recurrent_pct']
        
        # Add result
        result = {
            'inversion_excluded': inv_id,
            'inv_type': inv_type,
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
        all_p_values.append(filtered_stats['test_results']['p_value'])
        all_effect_sizes.append(effect_size)
    
    # Create results dataframe and sort by p-value
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    
    # Calculate robust overall p-value (median of all leave-one-out tests)
    # This is more conservative than taking the baseline p-value alone
    median_p_value = np.median(all_p_values)
    max_p_value = np.max(all_p_values)
    median_effect_size = np.median(all_effect_sizes)
    
    # Save results
    results_df.to_csv(OUTPUT_RESULTS, index=False)
    
    # Print summary of results
    print("\nLEAVE-ONE-OUT ANALYSIS RESULTS")
    print("-------------------------------")
    print(f"Total inversions analyzed: {len(large_inversions)}")
    print(f"Baseline p-value: {baseline_result['p_value']:.8e}")
    print(f"Baseline effect size: {baseline_result['effect_size']:.2f}% (Single-event minus Recurrent)")
    print(f"ROBUST FINAL P-VALUE (median of all leave-one-out tests): {median_p_value:.8e}")
    print(f"ROBUST EFFECT SIZE: {median_effect_size:.2f}% (Single-event minus Recurrent)")
    print(f"Most conservative p-value (maximum of all tests): {max_p_value:.8e}")
    
    print("\nTop 5 results by p-value:")
    
    top5 = results_df.head(5)
    for i, row in top5.iterrows():
        excluded = row['inversion_excluded']
        p_value = row['p_value']
        rec_pct = row['recurrent_pct']
        single_pct = row['single_pct']
        effect = row['effect_size']
        
        if excluded == 'None':
            print(f"BASELINE: p={p_value:.8e}, Recurrent={rec_pct:.1f}%, Single={single_pct:.1f}%, Effect={effect:.2f}%")
        else:
            inv_type = row['inv_type']
            count = row['excluded_count']
            print(f"Excluding {inv_type} inversion {excluded} ({count} sequences): p={p_value:.8e}, Recurrent={rec_pct:.1f}%, Single={single_pct:.1f}%, Effect={effect:.2f}%")
    
    print("\nMost influential inversions (largest p-value change):")
    results_df['p_change'] = results_df['p_value'] - baseline_result['p_value']
    top_influence = results_df[results_df['inversion_excluded'] != 'None'].sort_values('p_change', key=abs, ascending=False).head(5)
    
    for i, row in top_influence.iterrows():
        excluded = row['inversion_excluded']
        inv_type = row['inv_type']
        p_value = row['p_value']
        p_change = row['p_change']
        count = row['excluded_count']
        effect = row['effect_size']
        
        change_direction = "increased" if p_change > 0 else "decreased"
        print(f"Inversion {excluded} ({inv_type}, {count} sequences): p-value {change_direction} by {abs(p_change):.8e} to {p_value:.8e}, Effect size={effect:.2f}%")
    
    # Determine final conclusion based on median p-value and effect direction
    alpha = 0.05
    significance = "SIGNIFICANT difference" if median_p_value < alpha else "NO significant difference"
    more_conserved = "SINGLE-EVENT inversions are more conserved than RECURRENT inversions" if median_effect_size > 0 else "RECURRENT inversions are more conserved than SINGLE-EVENT inversions"
    
    print(f"\nFINAL CONCLUSION: There is a {significance} in conservation between recurrent and single-event inversions (p={median_p_value:.8e}, α={alpha})")
    print(f"DIRECTION: {more_conserved} (difference of {abs(median_effect_size):.2f}% in identical sequences)")
    
    return results_df, median_p_value

def analyze_proportions(df):
    """Calculate conservation statistics and run statistical tests."""
    # Group by inversion type and calculate proportions
    stats = df.groupby('inversion_type').agg(
        total=('is_identical', 'count'),
        identical=('is_identical', 'sum')
    )
    
    # Get contingency table for statistical tests
    if 'recurrent' in stats.index and 'single_event' in stats.index:
        rec_total = stats.loc['recurrent', 'total']
        rec_identical = stats.loc['recurrent', 'identical']
        single_total = stats.loc['single_event', 'total']
        single_identical = stats.loc['single_event', 'identical']
        
        # Contingency table: [identical, non-identical] x [recurrent, single-event]
        table = [
            [rec_identical, rec_total - rec_identical],
            [single_identical, single_total - single_identical]
        ]
        
        # Run statistical test
        test_results = run_statistical_test(table)
        
        # Identify which inversions are included
        recurrent_inversions = set()
        single_event_inversions = set()
        
        for _, row in df.iterrows():
            if row['inversion_type'] == 'recurrent':
                # This is an approximation since we don't have direct inversion mapping in this function
                for inv_id, cds_list in inversion_to_cds.items():
                    if row['CDS'] in cds_list and inv_id not in single_event_inversions:
                        recurrent_inversions.add(inv_id)
            elif row['inversion_type'] == 'single_event':
                for inv_id, cds_list in inversion_to_cds.items():
                    if row['CDS'] in cds_list and inv_id not in recurrent_inversions:
                        single_event_inversions.add(inv_id)
        
        # Calculate percentages
        rec_pct = (rec_identical / rec_total) * 100 if rec_total > 0 else 0
        single_pct = (single_identical / single_total) * 100 if single_total > 0 else 0
        
        return {
            'recurrent_total': rec_total,
            'recurrent_identical': rec_identical,
            'recurrent_pct': rec_pct,
            'single_total': single_total,
            'single_identical': single_identical,
            'single_pct': single_pct,
            'test_results': test_results,
            'recurrent_inversions': recurrent_inversions,
            'single_event_inversions': single_event_inversions
        }
    else:
        # Return empty results if we don't have both types
        return {
            'recurrent_total': 0,
            'recurrent_identical': 0,
            'recurrent_pct': 0,
            'single_total': 0,
            'single_identical': 0,
            'single_pct': 0,
            'test_results': {'p_value': 1.0, 'odds_ratio': 1.0},
            'recurrent_inversions': set(),
            'single_event_inversions': set()
        }

def main():
    """Main execution function with minimal output."""
    print("Running conservation leave-one-out analysis...")
    
    # Load input files
    try:
        pairwise_df = pd.read_csv(PAIRWISE_FILE)
        inversion_df = pd.read_csv(INVERSION_FILE)
        
        # Ensure proper numeric types
        for col in ['Start', 'End', '0_single_1_recur']:
            inversion_df[col] = pd.to_numeric(inversion_df[col], errors='coerce')
    except Exception as e:
        print(f"Error loading input files: {e}")
        return
    
    # Map CDS to inversions
    global inversion_to_cds  # Make sure this is available to all functions
    cds_to_type, cds_to_inversion_id, inversion_to_cds = map_cds_to_inversions(pairwise_df, inversion_df)
    
    # Calculate median omega
    median_df = calculate_sequence_median_omega(pairwise_df)
    
    # Conduct leave-one-out analysis
    _, final_p_value = conduct_leave_one_out_analysis(median_df, cds_to_type, inversion_to_cds)
    
    print(f"\nAnalysis complete. Full results saved to {OUTPUT_RESULTS}")
    print(f"FINAL P-VALUE: {final_p_value:.8e}")

# Global variable for inversion_to_cds mapping (needed across functions)
inversion_to_cds = {}

if __name__ == "__main__":
    main()
