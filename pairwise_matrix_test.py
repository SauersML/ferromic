"""
This script implements a permutation test to analyze differences between two groups 
of pairwise comparisons while accounting for their non-independence. The key challenge
is that pairwise comparisons within a group are usually not independent.

DATA STRUCTURE:
- Input data contains pairwise omega values between sequences (but this method can be applied generally to other types of data)
- Sequences belong to two groups (marked by *_0 or *_1 suffix)

METHOD:
1. Data Organization
  - For each group, create a matrix of all pairwise omega values
  - Matrix[i,j] = omega value between sequence i and j in that group
  - Matrices are symmetric (Matrix[i,j] = Matrix[j,i])
  - Diagonal is NaN (no self-comparisons)

2. Test Statistic
  - Calculate mean omega value in each group's matrix (using upper triangle only)
  - Take difference: mean(Group 1) - mean(Group 0) (or absolute mean, or median, etc.)
  - This captures if one group tends to have higher/lower omega values

3. Permutation Test
  - Null hypothesis: Group labels are random with respect to omega values
  - Take all sequences from both groups
  - Randomly shuffle sequences into two new groups (maintaining original group sizes)
  - Reconstruct matrices using original omega values but new group assignments
  - Calculate test statistic for this permutation
  - Repeat many times (default 1000)

VALIDITY:
The test is valid because:
1. Maintains dependency structure
  - Never creates new omega values
  - Only redistributes existing pairwise comparisons
  - Dependencies between comparisons sharing sequences are preserved

2. Handles missing data appropriately
  - Permuted matrices have more NaNs than original (when mixing groups)
  - This is correct because we only had within-group comparisons
  - NaNs are properly excluded from mean calculations

3. Tests proper null hypothesis
  - If groups have no real difference, omega values should be exchangeable
  - Permutation distribution represents null of "no group effect"
  - P-value = proportion of permuted differences more extreme than observed

INTERPRETATION:
- Low p-value means the observed difference between groups is unlikely by chance
- Suggests systematic differences in omega values between groups
- Original group structure (0 vs 1) captures meaningful variation

OUTPUT:
- P-value for each CDS
- Observed difference in means
- Summary statistics across all CDS regions
"""


import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm.auto import tqdm
import warnings
import os
import json
from datetime import datetime
from pathlib import Path
from functools import partial
import pickle
from scipy import stats
warnings.filterwarnings('ignore')

# Constants
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
CACHE_INTERVAL = 50  # Save results every N CDSs
N_PERMUTATIONS = 1000

# Create necessary directories
for directory in [CACHE_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

def get_cache_path(cds):
    """Generate cache file path for a CDS."""
    return CACHE_DIR / f"{cds.replace('/', '_')}.pkl"

def load_cached_result(cds):
    """Load cached result for a CDS if it exists."""
    cache_path = get_cache_path(cds)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_cached_result(cds, result):
    """Save result for a CDS to cache."""
    cache_path = get_cache_path(cds)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

def read_and_preprocess_data(file_path):
    """Read and preprocess the CSV file with parallel processing."""
    print("Reading data...")
    df = pd.read_csv(file_path)
    
    # Parallel filtering using dask if the dataset is large
    if len(df) > 1_000_000:
        import dask.dataframe as dd
        ddf = dd.from_pandas(df, npartitions=cpu_count())
        df = ddf[
            (ddf['omega'] != -1) & 
            (ddf['omega'] != 99) & 
            ddf['omega'].notnull()
        ].compute()
    else:
        df = df[
            (df['omega'] != -1) & 
            (df['omega'] != 99)
        ].dropna(subset=['omega'])
    
    print(f"Total valid comparisons: {len(df):,}")
    print(f"Unique CDSs found: {df['CDS'].nunique():,}")
    return df

def calculate_test_statistics(matrix_0, matrix_1):
    """Calculate both mean and median differences."""
    if matrix_0 is None or matrix_1 is None:
        return np.nan, np.nan
        
    # Get upper triangle entries excluding diagonal
    n0 = matrix_0.shape[0]
    n1 = matrix_1.shape[0]
    rows0, cols0 = np.triu_indices(n0, k=1)
    rows1, cols1 = np.triu_indices(n1, k=1)
    
    # Extract values, filtering out NaN and zeros used for missing data
    values_0 = matrix_0[rows0, cols0]
    values_1 = matrix_1[rows1, cols1]
    values_0 = values_0[(~np.isnan(values_0))]
    values_1 = values_1[(~np.isnan(values_1))]
    
    if len(values_0) == 0 or len(values_1) == 0:
        return np.nan, np.nan
        
    mean_diff = abs(np.mean(values_1) - np.mean(values_0))
    median_diff = abs(np.median(values_1) - np.median(values_0))
    #print("\nCalculate_test_statistics:")
    #print(f"  Group 0 values: {values_0}")
    #print(f"  Group 1 values: {values_1}") 
    #print(f"  Group 0 median: {np.median(values_0):.4f}")
    #print(f"  Group 1 median: {np.median(values_1):.4f}")
    #print(f"  Median diff: {median_diff:.4f}")
    #print(f"  Mean diff: {mean_diff:.4f}")
    
    return mean_diff, median_diff


def get_pairwise_value(seq1, seq2, pairwise_dict):
    """Get omega value for a pair of sequences."""
    key1 = (seq1, seq2)
    key2 = (seq2, seq1)
    val1 = pairwise_dict.get(key1)
    val2 = pairwise_dict.get(key2)
    if val1 is not None:
        return val1
    return val2

def create_matrices(sequences_0, sequences_1, pairwise_dict):
    """Create matrices for two groups based on sequence assignments."""
    if len(sequences_0) == 0 or len(sequences_1) == 0:
        return None, None
        
    n0, n1 = len(sequences_0), len(sequences_1)
    matrix_0 = np.full((n0, n0), np.nan)
    matrix_1 = np.full((n1, n1), np.nan)
    
    # Fill matrix 0
    for i in range(n0):
        for j in range(i+1, n0):
            val = get_pairwise_value(sequences_0[i], sequences_0[j], pairwise_dict)
            if val is not None:
                matrix_0[i,j] = matrix_0[j,i] = val
                
    # Fill matrix 1
    for i in range(n1):
        for j in range(i+1, n1):
            val = get_pairwise_value(sequences_1[i], sequences_1[j], pairwise_dict)
            if val is not None:
                matrix_1[i,j] = matrix_1[j,i] = val
    
    return matrix_0, matrix_1

def permutation_test_worker(args):
    """Worker function for permutation testing optimized for speed."""
    all_sequences, n0, pairwise_df, sequences_0, sequences_1 = args

    # Create DataFrames for original groups
    group_assignments = pd.Series(index=all_sequences, data=0)
    group_assignments.loc[sequences_1] = 1

    # Merge omega values with group assignments
    pairwise_df['Group1'] = pairwise_df['Seq1'].map(group_assignments)
    pairwise_df['Group2'] = pairwise_df['Seq2'].map(group_assignments)

    # Original test statistic
    orig_values_0 = pairwise_df[
        (pairwise_df['Group1'] == 0) & (pairwise_df['Group2'] == 0)
    ]['omega'].values

    orig_values_1 = pairwise_df[
        (pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1)
    ]['omega'].values

    if len(orig_values_0) == 0 or len(orig_values_1) == 0:
        return {
            'observed_mean': np.nan,
            'observed_median': np.nan,
            'mean_pvalue': np.nan,
            'median_pvalue': np.nan,
            'n0': len(sequences_0),
            'n1': len(sequences_1),
            'effect_size_mean': np.nan,
            'effect_size_median': np.nan
        }

    orig_mean_diff = abs(np.mean(orig_values_1) - np.mean(orig_values_0))
    orig_median_diff = abs(np.median(orig_values_1) - np.median(orig_values_0))

    # Store original omega values for permutations
    omega_values = pairwise_df[['Seq1', 'Seq2', 'omega']].values

    # Run permutations
    permuted_mean_diffs = []
    permuted_median_diffs = []
    for _ in range(N_PERMUTATIONS):
        # Shuffle group assignments
        np.random.shuffle(all_sequences)
        perm_groups = pd.Series(index=all_sequences, data=0)
        perm_groups.iloc[n0:] = 1

        # Map new group assignments
        group_map = perm_groups.to_dict()
        pairwise_df['PermGroup1'] = pairwise_df['Seq1'].map(group_map)
        pairwise_df['PermGroup2'] = pairwise_df['Seq2'].map(group_map)

        # Select within-group omega values based on permuted groups
        perm_values_0 = pairwise_df[
            (pairwise_df['PermGroup1'] == 0) & (pairwise_df['PermGroup2'] == 0)
        ]['omega'].values

        perm_values_1 = pairwise_df[
            (pairwise_df['PermGroup1'] == 1) & (pairwise_df['PermGroup2'] == 1)
        ]['omega'].values

        if len(perm_values_0) == 0 or len(perm_values_1) == 0:
            continue

        mean_diff = abs(np.mean(perm_values_1) - np.mean(perm_values_0))
        median_diff = abs(np.median(perm_values_1) - np.median(perm_values_0))

        permuted_mean_diffs.append(mean_diff)
        permuted_median_diffs.append(median_diff)

    # Calculate p-values
    permuted_mean_diffs = np.array(permuted_mean_diffs)
    permuted_median_diffs = np.array(permuted_median_diffs)

    mean_pval = (np.sum(permuted_mean_diffs >= orig_mean_diff) + 1) / (len(permuted_mean_diffs) + 1)
    median_pval = (np.sum(permuted_median_diffs >= orig_median_diff) + 1) / (len(permuted_median_diffs) + 1)

    return {
        'observed_mean': orig_mean_diff,
        'observed_median': orig_median_diff,
        'mean_pvalue': mean_pval,
        'median_pvalue': median_pval,
        'n0': len(sequences_0),
        'n1': len(sequences_1),
        'effect_size_mean': orig_mean_diff / permuted_mean_diffs.std(ddof=1),
        'effect_size_median': orig_median_diff / permuted_median_diffs.std(ddof=1)
    }

def create_visualization(matrix_0, matrix_1, cds, result):
    if matrix_0 is None or matrix_1 is None:
        return
        
    # Create figure without any specific style
    fig = plt.figure(figsize=(20, 10))
    
    # Main title
    plt.suptitle(f'Pairwise Comparison Analysis: {cds}', 
                fontsize=16, fontweight='bold', y=1.02)
    
    gs = plt.GridSpec(2, 3, figure=fig)
  
    # Heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
  
    # Custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Plot heatmaps
    sns.heatmap(matrix_0, cmap=cmap, center=1, ax=ax1, 
                square=True, cbar_kws={'label': 'Omega Value'})
    sns.heatmap(matrix_1, cmap=cmap, center=1, ax=ax2, 
                square=True, cbar_kws={'label': 'Omega Value'})
    
    ax1.set_title(f'Group 0 Matrix (n={result["n0"]})', fontsize=12, pad=10)
    ax2.set_title(f'Group 1 Matrix (n={result["n1"]})', fontsize=12, pad=10)
    
    # Distribution comparison
    ax3 = fig.add_subplot(gs[0, 2])
    values_0 = matrix_0[np.triu_indices_from(matrix_0, k=1)]
    values_1 = matrix_1[np.triu_indices_from(matrix_1, k=1)]
    
    sns.kdeplot(data=values_0[~np.isnan(values_0)], ax=ax3, label='Group 0', 
                fill=True, alpha=0.5)
    sns.kdeplot(data=values_1[~np.isnan(values_1)], ax=ax3, label='Group 1', 
                fill=True, alpha=0.5)
    ax3.set_title('Distribution of Omega Values', fontsize=12)
    ax3.legend()
    
    # Results table
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Mean', 'Median'],
        ['Observed Difference', f"{result['observed_mean']:.4f}", f"{result['observed_median']:.4f}"],
        ['P-value', f"{result['mean_pvalue']:.4f}", f"{result['median_pvalue']:.4f}"],
        ['Effect Size', f"{result['effect_size_mean']:.4f}", f"{result['effect_size_median']:.4f}"]
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 2)
    
    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E6E6E6')
        if col == 0:
            cell.set_text_props(weight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'analysis_{cds.replace("/", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_cds_parallel(args):
    """Analyze a single CDS with parallel permutations."""
    df_cds, cds = args
    
    # Check cache first
    cached_result = load_cached_result(cds)
    if cached_result is not None:
        return cds, cached_result
    
    # Create pairwise dictionary
    pairwise_dict = {(row['Seq1'], row['Seq2']): row['omega'] 
                     for _, row in df_cds.iterrows()}
    
    # Get sequences for each group
    sequences_0 = np.array([seq for seq in df_cds['Seq1'].unique() 
                           if not seq.endswith('1')])
    sequences_1 = np.array([seq for seq in df_cds['Seq1'].unique() 
                           if seq.endswith('1')])
    
    if len(sequences_0) < 2 or len(sequences_1) < 2:
        result = {
            'observed_mean': np.nan,
            'observed_median': np.nan,
            'mean_pvalue': np.nan,
            'median_pvalue': np.nan,
            'matrix_0': None,
            'matrix_1': None,
            'n0': len(sequences_0),
            'n1': len(sequences_1),
            'effect_size_mean': np.nan,
            'effect_size_median': np.nan
        }
    else:
        all_sequences = np.concatenate([sequences_0, sequences_1])
        result = permutation_test_worker((
            all_sequences, len(sequences_0), pairwise_dict, 
            sequences_0, sequences_1
        ))
    
    # Cache the result
    save_cached_result(cds, result)
    return cds, result

def parse_cds_coordinates(cds_name):
    """Extract chromosome and coordinates from CDS name."""
    try:
        # Debug print to see actual CDS name format
        #print(f"Parsing CDS name: {cds_name}")
        
        # Try different possible formats
        if '/' in cds_name:  # If it's a path, take last part
            cds_name = cds_name.split('/')[-1]
            
        if '_' in cds_name: # This is what we expect
          try:
              # Format is chr{X}_start{Y}_end{Z}
              parts = cds_name.split('_')
              if len(parts) == 3 and parts[1].startswith('start') and parts[2].startswith('end'):
                  chrom = parts[0]
                  start = int(parts[1].replace('start', ''))
                  end = int(parts[2].replace('end', ''))
                  return chrom, start, end
          except Exception as e:
              print(f"Error parsing {cds_name}: {str(e)}")
              return None, None, None
        elif ':' in cds_name:
            chrom, coords = cds_name.split(':')
            if '-' in coords:
                start, end = coords.split('-')
            else:
                start, end = coords.split('..')
            return chrom, int(start), int(end)
            
        # Print debug info if parsing fails
        print(f"Failed to parse: {cds_name}")
        return None, None, None
    except Exception as e:
        print(f"Error parsing {cds_name}: {str(e)}")
        return None, None, None

def build_overlap_clusters(results_df):
    """Build clusters of overlapping CDS regions."""
    # Debug info
    print(f"\nAnalyzing {len(results_df)} CDS entries")
    
    # Initialize clusters
    clusters = {}
    cluster_id = 0
    cds_to_cluster = {}
    
    # Sort CDSs by chromosome and start position
    cds_coords = []
    for cds in results_df['CDS']:
        chrom, start, end = parse_cds_coordinates(cds)
        if None not in (chrom, start, end):
            cds_coords.append((chrom, start, end, cds))

    print(f"\nSuccessfully parsed {len(cds_coords)} CDS coordinates")
    if len(cds_coords) == 0:
        print("No CDS coordinates could be parsed! Check CDS name format.")
        # Print a few example CDS names
        print("\nExample CDS names:")
        for cds in results_df['CDS'].head():
            print(cds)
    
    cds_coords.sort()  # Sort by chromosome, then start
    
    # Build clusters
    active_regions = []  # (chrom, end, cluster_id)
    
    for chrom, start, end, cds in cds_coords:
        # Remove finished active regions
        active_regions = [(c, e, cid) for c, e, cid in active_regions 
                         if c == chrom and e >= start]
        
        # Find overlapping clusters
        overlapping_clusters = set(cid for c, e, cid in active_regions)
        
        if not overlapping_clusters:
            # Create new cluster
            clusters[cluster_id] = {cds}
            cds_to_cluster[cds] = cluster_id
            active_regions.append((chrom, end, cluster_id))
            cluster_id += 1
        else:
            # Add to first overlapping cluster
            target_cluster = min(overlapping_clusters)
            clusters[target_cluster].add(cds)
            cds_to_cluster[cds] = target_cluster
            
            # Merge other overlapping clusters if any
            for cid in overlapping_clusters:
                if cid != target_cluster:
                    clusters[target_cluster].update(clusters[cid])
                    del clusters[cid]
            
            # Update active regions
            active_regions = [(c, e, target_cluster) if cid in overlapping_clusters
                            else (c, e, cid) 
                            for c, e, cid in active_regions]
            active_regions.append((chrom, end, target_cluster))
    
    return clusters

def combine_cluster_evidence(cluster_cdss, results_df):
    """Combine statistics for a cluster of overlapping CDSs."""
    cluster_data = results_df[results_df['CDS'].isin(cluster_cdss)]
    
    # Get weights based on CDS length
    weights = {}
    total_length = 0
    for cds in cluster_cdss:
        _, start, end = parse_cds_coordinates(cds)
        if None not in (start, end):
            length = end - start
            weights[cds] = length
            total_length += length
    
    # Normalize weights
    for cds in weights:
        weights[cds] /= total_length
    
    # Compute weighted statistics
    weighted_mean_diff = 0
    weighted_effect_size = 0
    total_comparisons = 0
    valid_cdss = 0
    
    for cds in cluster_cdss:
        row = cluster_data[cluster_data['CDS'] == cds].iloc[0]
        if not np.isnan(row['observed_mean']) and not np.isnan(row['effect_size_mean']):
            weight = weights.get(cds, 1/len(cluster_cdss))
            weighted_mean_diff += row['observed_mean'] * weight
            weighted_effect_size += row['effect_size_mean'] * weight
            total_comparisons += row['n0'] * (row['n0'] - 1) // 2  # comparisons in group 0
            total_comparisons += row['n1'] * (row['n1'] - 1) // 2  # comparisons in group 1
            valid_cdss += 1
    
    # Combine p-values if we have valid data
    if valid_cdss > 0:
        # Use Fisher's method within cluster
        valid_pvals = cluster_data['mean_pvalue'].dropna()
        if len(valid_pvals) > 0:
            fisher_stat = -2 * np.sum(np.log(valid_pvals))
            combined_p = stats.chi2.sf(fisher_stat, df=2*len(valid_pvals))
        else:
            combined_p = np.nan
    else:
        combined_p = np.nan
        weighted_mean_diff = np.nan
        weighted_effect_size = np.nan
        total_comparisons = 0
    
    return {
        'combined_pvalue': combined_p,
        'weighted_mean_diff': weighted_mean_diff,
        'weighted_effect_size': weighted_effect_size,
        'n_comparisons': total_comparisons,
        'n_valid_cds': valid_cdss
    }







"""
OVERALL TEST METHODOLOGY

The overall test combines evidence across CDS regions.

1. DEPENDENCY STRUCTURE
- CDSs that share genomic coordinates are not independent
- They may be using the same underlying sequence data
- We handle this by first grouping overlapping CDSs into clusters
- Each cluster is assumed to represent one independent piece of evidence

2. WITHIN-CLUSTER COMBINATION
For each cluster of overlapping CDSs:
- Weight each CDS by its genomic length
- Combine their test statistics using weighted averages
- Use Fisher's method to combine p-values within cluster
- Track number of actual pairwise comparisons made

3. BETWEEN-CLUSTER COMBINATION 
Once we have cluster-level statistics:
- We assume clusters are truly independent (no sequence overlap)
- Use Fisher's method to combine cluster p-values
- Weight effect sizes by number of comparisons
"""


def compute_overall_significance(cluster_results):
    """Compute overall significance from independent clusters."""
    valid_clusters = [c for c in cluster_results.values() 
                     if not np.isnan(c['combined_pvalue'])]
    
    if not valid_clusters:
        return {
            'overall_pvalue': np.nan,
            'overall_effect': np.nan,
            'n_valid_clusters': 0,
            'total_comparisons': 0
        }
    
    # Combine p-values using Fisher's method
    cluster_pvals = [c['combined_pvalue'] for c in valid_clusters]
    fisher_stat = -2 * np.sum(np.log(cluster_pvals))
    overall_pvalue = stats.chi2.sf(fisher_stat, df=2*len(cluster_pvals))
    
    # Compute weighted effect size
    total_comparisons = sum(c['n_comparisons'] for c in valid_clusters)
    weighted_effect = np.average(
        [c['weighted_effect_size'] for c in valid_clusters],
        weights=[c['n_comparisons'] for c in valid_clusters]
    )
    
    return {
        'overall_pvalue': overall_pvalue,
        'overall_effect': weighted_effect,
        'n_valid_clusters': len(valid_clusters),
        'total_comparisons': total_comparisons
    }




def main():
    start_time = datetime.now()
    print(f"Analysis started at {start_time}")
    
    # Read data
    df = read_and_preprocess_data('all_pairwise_results.csv')
    
    # Prepare arguments for parallel processing
    cds_args = [(df[df['CDS'] == cds], cds) for cds in df['CDS'].unique()]
    
    # Process CDSs in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for cds, result in tqdm(
            executor.map(analyze_cds_parallel, cds_args),
            total=len(cds_args),
            desc="Processing CDSs"
        ):
            results[cds] = result
    
    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            'CDS': cds,
            **{k: v for k, v in result.items() 
               if k not in ['matrix_0', 'matrix_1']}
        }
        for cds, result in results.items()
    ])
    
    # Save final results
    results_df.to_csv(RESULTS_DIR / 'final_results.csv', index=False)
    
    # Overall analysis
    print("\nComputing overall significance...")
    clusters = build_overlap_clusters(results_df)
    cluster_stats = {}
    for cluster_id, cluster_cdss in clusters.items():
        cluster_stats[cluster_id] = combine_cluster_evidence(cluster_cdss, results_df)
    
    overall_results = compute_overall_significance(cluster_stats)
    
    # Convert numpy values to native Python types for JSON serialization
    json_safe_results = {
        'overall_pvalue': float(overall_results['overall_pvalue']) if not np.isnan(overall_results['overall_pvalue']) else None,
        'overall_effect': float(overall_results['overall_effect']) if not np.isnan(overall_results['overall_effect']) else None,
        'n_valid_clusters': int(overall_results['n_valid_clusters']),
        'total_comparisons': int(overall_results['total_comparisons'])
    }
    
    # Save overall results
    with open(RESULTS_DIR / 'overall_results.json', 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    
    # Print overall results
    print("\nOverall Analysis Results:")
    print(f"Number of independent clusters: {overall_results['n_valid_clusters']}")
    print(f"Total pairwise comparisons: {overall_results['total_comparisons']:,}")
    print(f"Overall p-value: {overall_results['overall_pvalue']:.4e}")
    print(f"Overall effect size: {overall_results['overall_effect']:.4f}")
    
    significant_results = results_df.sort_values('mean_pvalue')
    for _, row in significant_results.head().iterrows():
        cds = row['CDS']
        result = results[cds]
        create_visualization(
            result['matrix_0'], 
            result['matrix_1'], 
            cds, 
            result
        )
  
    # Print summary statistics
    valid_results = results_df[~results_df['mean_pvalue'].isna()]
    print("\nPer-CDS Analysis Summary:")
    print(f"Total CDSs analyzed: {len(results_df):,}")
    print(f"Valid analyses: {len(valid_results):,}")
    print(f"Significant CDSs (p < 0.05):")
    print(f"  By mean: {(valid_results['mean_pvalue'] < 0.05).sum():,}")
    print(f"  By median: {(valid_results['median_pvalue'] < 0.05).sum():,}")
    
    end_time = datetime.now()
    print(f"\nAnalysis completed at {end_time}")
    print(f"Total runtime: {end_time - start_time}")

if __name__ == "__main__":
    main()
