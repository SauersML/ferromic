import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import warnings
import os
import json
from datetime import datetime
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import combine_pvalues

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
N_PERMUTATIONS = 10000  # Number of permutations for the permutation test

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
                cached_result = pickle.load(f)
            return cached_result
        except:
            return None
    return None

def save_cached_result(cds, result):
    """Save result for a CDS to cache."""
    cache_path = get_cache_path(cds)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

def read_and_preprocess_data(file_path):
    """Read and preprocess the CSV file."""
    print("Reading data...")
    df = pd.read_csv(file_path)

    # Filtering valid omega values
    df = df[
        (df['omega'] != -1) &
        (df['omega'] != 99)
    ].dropna(subset=['omega'])

    print(f"Total valid comparisons: {len(df):,}")
    print(f"Unique CDSs found: {df['CDS'].nunique():,}")
    return df

def get_pairwise_value(seq1, seq2, pairwise_dict):
    """Get omega value for a pair of sequences."""
    key = (seq1, seq2) if (seq1, seq2) in pairwise_dict else (seq2, seq1)
    return pairwise_dict.get(key)

def create_matrices(sequences_0, sequences_1, pairwise_dict):
    """Create matrices for two groups based on sequence assignments."""
    if len(sequences_0) == 0 or len(sequences_1) == 0:
        return None, None

    n0, n1 = len(sequences_0), len(sequences_1)
    matrix_0 = np.full((n0, n0), np.nan)
    matrix_1 = np.full((n1, n1), np.nan)

    # Fill matrix 0
    for i in range(n0):
        for j in range(i + 1, n0):
            val = get_pairwise_value(sequences_0[i], sequences_0[j], pairwise_dict)
            if val is not None:
                matrix_0[i, j] = matrix_0[j, i] = val

    # Fill matrix 1
    for i in range(n1):
        for j in range(i + 1, n1):
            val = get_pairwise_value(sequences_1[i], sequences_1[j], pairwise_dict)
            if val is not None:
                matrix_1[i, j] = matrix_1[j, i] = val

    return matrix_0, matrix_1

def analysis_worker(args):
    """Mixed effects analysis for a single CDS with crossed random effects."""
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    import numpy as np

    all_sequences, n0, pairwise_dict, sequences_0, sequences_1 = args

    # Prepare data
    data = []
    for (seq1, seq2), omega in pairwise_dict.items():
        # Determine group of the pair
        if seq1 in sequences_0 and seq2 in sequences_0:
            group = 0
        elif seq1 in sequences_1 and seq2 in sequences_1:
            group = 1
        else:
            continue  # Skip pairs not within the same group

        data.append({
            'omega_value': omega,
            'group': group,
            'seq1': seq1,
            'seq2': seq2
        })

    df = pd.DataFrame(data)

    # Initialize variables
    effect_size = np.nan
    p_value = np.nan
    std_err = np.nan

    # Check if DataFrame has sufficient data
    if df.empty or df['group'].nunique() < 2 or df['omega_value'].nunique() < 2:
        return {
            'observed_effect_size': effect_size,
            'p_value': p_value,
            'n0': n0,
            'n1': len(all_sequences) - n0,
            'num_comp_group_0': (df['group'] == 0).sum() if not df.empty else 0,
            'num_comp_group_1': (df['group'] == 1).sum() if not df.empty else 0,
            'std_err': std_err
        }

    # Convert sequences to categorical codes
    df['seq1_code'] = pd.Categorical(df['seq1']).codes
    df['seq2_code'] = pd.Categorical(df['seq2']).codes

    # Print diagnostic info
    print(f"\nAnalyzing CDS with:")
    print(f"Data shape: {df.shape}")
    print(f"Unique seq1_codes: {len(df['seq1_code'].unique())}")
    print(f"Unique seq2_codes: {len(df['seq2_code'].unique())}")
    print(f"Group counts:\n{df['group'].value_counts()}")

    try:
        # Prepare data for MixedLM
        # Introduce a dummy 'groups' variable since we'll specify random effects via 'vc_formula'
        df['groups'] = 1  # All data belongs to the same group for variance components

        # Define variance components for crossed random effects
        vc = {
            'seq1': '0 + C(seq1_code)',
            'seq2': '0 + C(seq2_code)'
        }

        # Fit the mixed model using MixedLM.from_formula
        model = MixedLM.from_formula(
            'omega_value ~ group',
            groups='groups',
            vc_formula=vc,
            re_formula='0',  # No random intercept for 'groups' since it's a dummy
            data=df
        )
        result = model.fit(reml=False)  # Use ML estimation

        # Extract results
        effect_size = result.fe_params['group']
        p_value = result.pvalues['group']
        std_err = result.bse['group']

        print(f"Successfully fit model:")
        print(f"Effect size: {effect_size:.4f}")
        print(f"P-value: {p_value:.4e}")
        print(f"Std error: {std_err:.4f}")

    except Exception as e:
        print(f"Model fitting failed with error: {str(e)}")
        # Variables remain as initialized (np.nan)

    return {
        'observed_effect_size': effect_size,
        'p_value': p_value,
        'n0': n0,
        'n1': len(all_sequences) - n0,
        'std_err': std_err,
        'num_comp_group_0': (df['group'] == 0).sum(),
        'num_comp_group_1': (df['group'] == 1).sum(),
    }

def compute_cliffs_delta(x, y):
    """Compute Cliff's Delta effect size."""
    n_x = len(x)
    n_y = len(y)
    n_pairs = n_x * n_y

    # Efficient computation using broadcasting
    x = np.array(x)
    y = np.array(y)

    difference_matrix = x[:, np.newaxis] - y
    num_greater = np.sum(difference_matrix > 0)
    num_less = np.sum(difference_matrix < 0)

    cliffs_delta = (num_greater - num_less) / n_pairs

    return cliffs_delta



def create_visualization(matrix_0, matrix_1, cds, result):
    """Create enhanced visualizations for a CDS analysis."""
    if matrix_0 is None or matrix_1 is None:
        print(f"No data available for CDS: {cds}")
        return

    # Reset any custom font settings to defaults
    plt.rcParams.update(plt.rcParamsDefault)

    # Create a figure with specified size and layout
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 3, height_ratios=[4, 1], hspace=0.4, wspace=0.4)

    # Main title
    fig.suptitle(f'Pairwise Comparison Analysis: {cds}', fontsize=18, fontweight='bold', y=0.95)

    # Custom colormap excluding white
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Function to plot the lower triangle of a matrix without masking
    def plot_lower_triangle(ax, matrix, title, tick_labels):
        n = matrix.shape[0]
        # Extract the indices for the lower triangle
        x_coords, y_coords = np.meshgrid(np.arange(n), np.arange(n))
        lower_tri_indices = np.tril_indices(n, k=-1)
        # Extract the values for the lower triangle
        values = matrix[lower_tri_indices]

        # Create a scatter plot to represent the lower triangle
        sc = ax.scatter(x_coords[lower_tri_indices], y_coords[lower_tri_indices],
                        c=values, cmap=cmap, marker='s', s=100)
        # Adjust axis labels and ticks
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel('Sequence Index', fontsize=12)
        ax.set_ylabel('Sequence Index', fontsize=12)
        # Invert y-axis to have (1,1) at bottom-left
        ax.invert_yaxis()
        # Ensure x-axis labels are displayed right side up
        ax.tick_params(axis='x', rotation=0)
        ax.set_title(title, fontsize=14, pad=12)
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Omega Value', rotation=270, labelpad=15)
        # Set axis limits
        ax.set_xlim(-0.5, n - 1 + 0.5)
        ax.set_ylim(n - 1 + 0.5, -0.5)
        # Adjust aspect ratio
        ax.set_aspect('equal')

    # Prepare tick labels starting from 1
    n0 = result['n0']
    n1 = result['n1']
    tick_labels_0 = [str(i + 1) for i in range(n0)]
    tick_labels_1 = [str(i + 1) for i in range(n1)]

    # Plot for Group 0
    ax1 = fig.add_subplot(gs[0, 0])
    plot_lower_triangle(ax1, matrix_0, f'Group 0 Matrix (n={n0})', tick_labels_0)

    # Plot for Group 1
    ax2 = fig.add_subplot(gs[0, 1])
    plot_lower_triangle(ax2, matrix_1, f'Group 1 Matrix (n={n1})', tick_labels_1)

    # Distribution comparison between groups
    ax3 = fig.add_subplot(gs[0, 2])
    # Extract the lower triangle values excluding the diagonal
    values_0 = matrix_0[np.tril_indices_from(matrix_0, k=-1)]
    values_1 = matrix_1[np.tril_indices_from(matrix_1, k=-1)]
    sns.kdeplot(values_0[~np.isnan(values_0)], ax=ax3, label='Group 0', fill=True,
                common_norm=False, color='#1f77b4', alpha=0.6)
    sns.kdeplot(values_1[~np.isnan(values_1)], ax=ax3, label='Group 1', fill=True,
                common_norm=False, color='#ff7f0e', alpha=0.6)
    ax3.set_title('Distribution of Omega Values', fontsize=14, pad=12)
    ax3.set_xlabel('Omega Value', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.legend(title='Groups', title_fontsize=12, fontsize=10)

    # Results table
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')

    # Prepare table data with clear indication of missing results
    effect_size = f"{result['observed_effect_size']:.4f}" if not np.isnan(result['observed_effect_size']) else 'N/A'
    p_value = f"{result['p_value']:.4e}" if not np.isnan(result['p_value']) else 'N/A'
    std_err = f"{result['std_err']:.4f}" if not np.isnan(result['std_err']) else 'N/A'

    table_data = [
        ['Metric', 'Value'],
        ['Observed Effect Size (from Mixed Model)', effect_size],
        ['Standard Error', std_err],
        ['P-value', p_value],
        ['Number of Sequences in Group 0', str(n0)],
        ['Number of Sequences in Group 1', str(n1)],
        ['Comparisons in Group 0', str(result['num_comp_group_0'])],
        ['Comparisons in Group 1', str(result['num_comp_group_1'])]
    ]

    # Create table
    table = ax4.table(cellText=table_data, loc='center', cellLoc='left',
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', ha='center')
            cell.set_facecolor('#E6E6E6')
        elif col == 0:
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('gray')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(PLOTS_DIR / f'analysis_{cds.replace('/', '_')}.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def analyze_cds_parallel(args):
    """Analyze a single CDS"""
    df_cds, cds = args

    # Check cache first
    cached_result = load_cached_result(cds)
    if cached_result is not None:
        return cds, cached_result

    # Create pairwise dictionary
    pairwise_dict = {(row['Seq1'], row['Seq2']): row['omega']
                     for _, row in df_cds.iterrows()}

    # Collect all unique sequences from both 'Seq1' and 'Seq2' columns
    all_seqs = pd.concat([df_cds['Seq1'], df_cds['Seq2']]).unique()
    sequences_0 = np.array([seq for seq in all_seqs if not seq.endswith('1')])
    sequences_1 = np.array([seq for seq in all_seqs if seq.endswith('1')])
    all_sequences = np.concatenate([sequences_0, sequences_1])

    n0 = len(sequences_0)
    n1 = len(sequences_1)

    # Generate matrices for visualization (do this first)
    matrix_0, matrix_1 = create_matrices(sequences_0, sequences_1, pairwise_dict)

    # Set minimum required sequences per group
    min_sequences_per_group = 5

    # Initialize base result dictionary with matrices
    result = {
        'matrix_0': matrix_0,
        'matrix_1': matrix_1,
        'pairwise_comparisons': set(pairwise_dict.keys())
    }

    if n0 < min_sequences_per_group or n1 < min_sequences_per_group:
        # Not enough sequences in one of the groups
        result.update({
            'observed_effect_size': np.nan,
            'p_value': np.nan,
            'n0': n0,
            'n1': n1,
            'num_comp_group_0': 0,
            'num_comp_group_1': 0,
            'std_err': np.nan
        })
    else:
        # Call the analysis worker and update result
        worker_result = analysis_worker((
            all_sequences, n0, pairwise_dict,
            sequences_0, sequences_1
        ))
        result.update(worker_result)

    # Cache the result
    save_cached_result(cds, result)
    return cds, result


def parse_cds_coordinates(cds_name):
    """Extract chromosome and coordinates from CDS name."""
    try:
        # Try different possible formats
        if '/' in cds_name:  # If it's a path, take the last part
            cds_name = cds_name.split('/')[-1]

        if '_' in cds_name:  # Expected format
            parts = cds_name.split('_')
            if len(parts) == 3 and parts[1].startswith('start') and parts[2].startswith('end'):
                chrom = parts[0]
                start = int(parts[1].replace('start', ''))
                end = int(parts[2].replace('end', ''))
                return chrom, start, end
        elif ':' in cds_name:
            chrom, coords = cds_name.split(':')
            start, end = map(int, coords.replace('-', '..').split('..'))
            return chrom, start, end

        # If parsing fails
        print(f"Failed to parse: {cds_name}")
        return None, None, None
    except Exception as e:
        print(f"Error parsing {cds_name}: {str(e)}")
        return None, None, None

def build_overlap_clusters(results_df):
    """Build clusters of overlapping CDS regions."""
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

    # Sort by chromosome and start position
    cds_coords.sort()

    # Build clusters
    active_regions = []  # (chrom, end, cluster_id)

    for chrom, start, end, cds in cds_coords:
        # Remove finished active regions
        active_regions = [(c, e, cid) for c, e, cid in active_regions
                          if c != chrom or e >= start]

        # Find overlapping clusters
        overlapping_clusters = set(cid for c, e, cid in active_regions
                                   if c == chrom and e >= start)

        if not overlapping_clusters:
            # Create new cluster
            clusters[cluster_id] = {cds}
            cds_to_cluster[cds] = cluster_id
            active_regions.append((chrom, end, cluster_id))
            cluster_id += 1
        else:
            # Merge overlapping clusters
            target_cluster = min(overlapping_clusters)
            clusters[target_cluster].add(cds)
            cds_to_cluster[cds] = target_cluster

            # Merge other overlapping clusters
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

def combine_cluster_evidence(cluster_cdss, results_df, results):
    """Combine statistics for a cluster of overlapping CDSs using the smallest p-value."""
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

    # Initialize statistics
    weighted_effect_size = 0.0
    valid_cdss = 0
    valid_indices = []

    # Initialize a set to collect unique pairwise comparisons for the cluster
    cluster_pairs = set()

    for idx, row in cluster_data.iterrows():
        cds = row['CDS']
        effect_size = row['observed_effect_size']

        if np.isnan(effect_size):
            continue  # Skip invalid entries

        weight = weights.get(cds, 1 / len(cluster_cdss))
        weighted_effect_size += effect_size * weight

        # Accumulate unique pairwise comparisons from the results dictionary
        cds_pairs = results[cds]['pairwise_comparisons']
        cluster_pairs.update(cds_pairs)

        valid_cdss += 1
        valid_indices.append(idx)

    # After the loop, set total_comparisons to the number of unique pairs
    total_comparisons = len(cluster_pairs)

    if valid_cdss > 0:
        # Collect valid p-values
        valid_pvals = cluster_data.loc[valid_indices]['p_value'].values
    
        # Filter out invalid p-values
        valid_pvals = valid_pvals[~np.isnan(valid_pvals)]
        valid_pvals = valid_pvals[~np.isinf(valid_pvals)]
    
        if len(valid_pvals) > 0:
            # Normalized weights based on CDS lengths
            valid_weights = []
            for idx in valid_indices:
                cds = cluster_data.loc[idx, 'CDS']
                weight = weights.get(cds, 1 / len(cluster_cdss))
                valid_weights.append(weight)
            valid_weights = np.array(valid_weights)
            # Normalize weights so that they sum to 1
            valid_weights /= valid_weights.sum()
    
            # Combine p-values using Stouffer's method with weights
            z_stat, combined_p = combine_pvalues(valid_pvals, method='stouffer', weights=valid_weights)
        else:
            combined_p = np.nan
    else:
        combined_p = np.nan
        weighted_effect_size = np.nan
        total_comparisons = 0

    return {
        'combined_pvalue': combined_p,
        'weighted_effect_size': weighted_effect_size,
        'n_comparisons': total_comparisons,
        'n_valid_cds': valid_cdss,
        'cluster_pairs': cluster_pairs
    }

def compute_overall_significance(cluster_results):
    """Compute overall significance from independent clusters using scipy's combine_pvalues."""
    import numpy as np
    from scipy import stats

    # Initialize default return values
    overall_pvalue_combined = np.nan
    overall_effect = np.nan
    n_valid_clusters = 0
    total_comparisons = 0

    # Filter out clusters with valid combined_pvalue and weighted_effect_size
    valid_clusters = [
        c for c in cluster_results.values()
        if not np.isnan(c['combined_pvalue']) and not np.isnan(c['weighted_effect_size'])
    ]

    if valid_clusters:
        cluster_pvals = np.array([c['combined_pvalue'] for c in valid_clusters])

        # Use scipy.stats.combine_pvalues to combine p-values
        # Choose method: 'fisher', 'stouffer', or others as appropriate etc.
        statistic, overall_pvalue_combined = stats.combine_pvalues(cluster_pvals, method='fisher')

        print(f"\nCombined p-value using Fisher's method: {overall_pvalue_combined:.4e}")
        print(f"Fisher's statistic: {statistic:.4f}")

        # Stouffer's method
        weights = np.array([c['n_comparisons'] for c in valid_clusters], dtype=float)

        # Check for zero weights
        if np.all(weights == 0) or np.isnan(weights).any():
            weights = None
            print("Note: Weights not used in Stouffer's method due to zero or NaN values.")

        # Use Stouffer's method with weights
        statistic_stouffer, pvalue_stouffer = stats.combine_pvalues(cluster_pvals, method='stouffer', weights=weights)
        print(f"Combined p-value using Stouffer's method: {pvalue_stouffer:.4e}")
        print(f"Stouffer's Z-score statistic: {statistic_stouffer:.4f}")

        # Compute weighted effect size
        effect_sizes = np.array([c['weighted_effect_size'] for c in valid_clusters])

        if weights is not None:
            normalized_weights = weights / np.sum(weights)
        else:
            normalized_weights = np.ones_like(effect_sizes) / len(effect_sizes)

        overall_effect = np.average(effect_sizes, weights=normalized_weights)

        # Count comparisons
        all_unique_pairs = set()
        for c in valid_clusters:
            all_unique_pairs.update(c['cluster_pairs'])
        total_comparisons = len(all_unique_pairs)
        n_valid_clusters = len(valid_clusters)

        overall_pvalue = overall_pvalue_combined

    else:
        print("No valid clusters available for significance computation.")
        overall_pvalue = np.nan
        overall_pvalue_combined = np.nan
        overall_effect = np.nan
        pvalue_stouffer = np.nan

    return {
        'overall_pvalue': overall_pvalue,
        'overall_pvalue_fisher': overall_pvalue_combined,
        'overall_pvalue_stouffer': pvalue_stouffer,
        'overall_effect': overall_effect,
        'n_valid_clusters': n_valid_clusters,
        'total_comparisons': total_comparisons
    }

def main():
    start_time = datetime.now()
    print(f"Analysis started at {start_time}")

    # Read data
    df = read_and_preprocess_data('all_pairwise_results.csv')

    # Prepare arguments for parallel processing
    cds_list = df['CDS'].unique()
    cds_args = [(df[df['CDS'] == cds], cds) for cds in cds_list]

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
               if k not in ['matrix_0', 'matrix_1', 'pairwise_comparisons']}
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
        cluster_stats[cluster_id] = combine_cluster_evidence(cluster_cdss, results_df, results)


    # Compute overall significance
    overall_results = compute_overall_significance(cluster_stats)

    # Convert numpy values to native Python types for JSON serialization
    json_safe_results = {
        'overall_pvalue': float(overall_results['overall_pvalue']) if not np.isnan(overall_results['overall_pvalue']) else None,
        'overall_pvalue_fisher': float(overall_results['overall_pvalue_fisher']) if not np.isnan(overall_results['overall_pvalue_fisher']) else None,
        'overall_effect': float(overall_results['overall_effect']) if not np.isnan(overall_results['overall_effect']) else None,
        'n_valid_clusters': int(overall_results['n_valid_clusters']) if not np.isnan(overall_results['n_valid_clusters']) else None,
        'total_comparisons': int(overall_results['total_comparisons']) if not np.isnan(overall_results['total_comparisons']) else None
    }

    # Save overall results
    with open(RESULTS_DIR / 'overall_results.json', 'w') as f:
        json.dump(json_safe_results, f, indent=2)

    # Print overall results
    print("\nOverall Analysis Results:")
    print(f"Number of independent clusters: {overall_results['n_valid_clusters']}")
    print(f"Total unique CDS pairs: {overall_results['total_comparisons']:,}")
    print(f"Overall p-value: {overall_results['overall_pvalue']:.4e}")
    print(f"Overall effect size (Cliff's Delta): {overall_results['overall_effect']:.4f}")

    # Sort results by p-value
    significant_results = results_df.sort_values('p_value')

    # Create visualizations for top significant results
    for _, row in significant_results.head().iterrows():
        cds = row['CDS']
        viz_result = results[cds]  # Get full result with matrices
        create_visualization(
            viz_result['matrix_0'],
            viz_result['matrix_1'],
            cds,
            row  # Use row for stats since they're the same
        )

    # Print summary statistics
    valid_results = results_df[~results_df['p_value'].isna()]
    print("\nPer-CDS Analysis Summary:")
    print(f"Total CDSs analyzed: {len(results_df):,}")
    print(f"Valid analyses: {len(valid_results):,}")
    print(f"Significant CDSs (p < 0.05): {(valid_results['p_value'] < 0.05).sum():,}")

    end_time = datetime.now()
    print(f"\nAnalysis completed at {end_time}")
    print(f"Total runtime: {end_time - start_time}")

if __name__ == "__main__":
    main()
