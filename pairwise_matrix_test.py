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
  - Take difference: mean(Group 1) - mean(Group 0)
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def read_and_preprocess_data(file_path):
    """Read and preprocess the CSV file."""
    print("Reading data...")
    df = pd.read_csv(file_path)
    
    # Remove invalid omega values
    df = df[df['omega'] != -1]
    df = df[df['omega'] != 99]
    df = df.dropna(subset=['omega'])  # Remove NaN omega values
    
    print(f"Total valid comparisons: {len(df)}")
    print(f"Unique CDSs found: {df['CDS'].nunique()}")
    return df

def get_pairwise_value(seq1, seq2, pairwise_dict):
    """Get omega value for a pair of sequences."""
    key1 = (seq1, seq2)
    key2 = (seq2, seq1)
    return pairwise_dict.get(key1) or pairwise_dict.get(key2)

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

def calculate_test_statistic(matrix_0, matrix_1):
    """Calculate difference in means between upper triangles."""
    if matrix_0 is None or matrix_1 is None:
        return np.nan
        
    upper_0 = np.triu(matrix_0, k=1)
    upper_1 = np.triu(matrix_1, k=1)
    values_0 = upper_0[~np.isnan(upper_0)]
    values_1 = upper_1[~np.isnan(upper_1)]
    
    # Check if we have enough values
    if len(values_0) == 0 or len(values_1) == 0:
        return np.nan
        
    return np.mean(values_1) - np.mean(values_0)

def permutation_test_single(args):
    """Single permutation iteration."""
    all_sequences, n0, pairwise_dict = args
    
    # Randomly assign sequences to groups, maintaining original group sizes
    np.random.shuffle(all_sequences)
    sequences_0 = all_sequences[:n0]
    sequences_1 = all_sequences[n0:]
    
    # Create new matrices based on this assignment
    matrix_0, matrix_1 = create_matrices(sequences_0, sequences_1, pairwise_dict)
    
    return calculate_test_statistic(matrix_0, matrix_1)

def analyze_cds(df_cds, n_permutations=1000):
    """Analyze a single CDS."""
    # Create dictionary of pairwise values
    pairwise_dict = {(row['Seq1'], row['Seq2']): row['omega'] 
                     for _, row in df_cds.iterrows()}
    
    # Get original group assignments
    sequences_0 = np.array([seq for seq in df_cds['Seq1'].unique() 
                           if not seq.endswith('1')])
    sequences_1 = np.array([seq for seq in df_cds['Seq1'].unique() 
                           if seq.endswith('1')])
    
    print(f"Group 0 size: {len(sequences_0)}, Group 1 size: {len(sequences_1)}")
    
    # Check if we have enough sequences in both groups
    if len(sequences_0) < 2 or len(sequences_1) < 2:
        return {
            'observed_stat': np.nan,
            'p_value': np.nan,
            'matrix_0': None,
            'matrix_1': None,
            'n0': len(sequences_0),
            'n1': len(sequences_1)
        }
    
    n0 = len(sequences_0)
    
    # Calculate observed statistic
    matrix_0, matrix_1 = create_matrices(sequences_0, sequences_1, pairwise_dict)
    observed_stat = calculate_test_statistic(matrix_0, matrix_1)
    
    if np.isnan(observed_stat):
        return {
            'observed_stat': np.nan,
            'p_value': np.nan,
            'matrix_0': matrix_0,
            'matrix_1': matrix_1,
            'n0': n0,
            'n1': len(sequences_1)
        }
    
    # Run permutations
    all_sequences = np.concatenate([sequences_0, sequences_1])
    args = [(all_sequences.copy(), n0, pairwise_dict) for _ in range(n_permutations)]
    
    with Pool(cpu_count()) as pool:
        permuted_stats = list(tqdm(
            pool.imap(permutation_test_single, args),
            total=n_permutations,
            desc="Running permutations"
        ))
    
    # Remove any NaN values from permuted stats
    permuted_stats = np.array([x for x in permuted_stats if not np.isnan(x)])
    
    # Calculate p-value if we have enough permutations
    if len(permuted_stats) > 0:
        p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    else:
        p_value = np.nan
    
    return {
        'observed_stat': observed_stat,
        'p_value': p_value,
        'matrix_0': matrix_0,
        'matrix_1': matrix_1,
        'n0': n0,
        'n1': len(sequences_1)
    }

def visualize_matrices(matrix_0, matrix_1, cds, result):
    """Create and save visualization of the comparison matrices."""
    if matrix_0 is None or matrix_1 is None:
        return
        
    plt.figure(figsize=(15, 6))
    
    plt.subplot(121)
    sns.heatmap(matrix_0, cmap='viridis', center=1)
    plt.title(f'Group 0 Matrix (n={result["n0"]})')
    
    plt.subplot(122)
    sns.heatmap(matrix_1, cmap='viridis', center=1)
    plt.title(f'Group 1 Matrix (n={result["n1"]})')
    
    plt.suptitle(f'CDS: {cds}\nObs diff: {result["observed_stat"]:.4f}, p-value: {result["p_value"]:.4f}')
    plt.tight_layout()
    plt.savefig(f'matrices_{cds.replace("/", "_")}.png')
    plt.close()

def main():
    # Read data
    df = read_and_preprocess_data('all_pairwise_results.csv')
    results = []
    
    # Process each CDS
    for cds in tqdm(df['CDS'].unique(), desc="Processing CDSs"):
        print(f"\nAnalyzing CDS: {cds}")
        df_cds = df[df['CDS'] == cds]
        
        result = analyze_cds(df_cds)
        result['CDS'] = cds
        results.append(result)
        
        if not np.isnan(result['observed_stat']):
            print(f"Observed difference: {result['observed_stat']:.4f}")
            print(f"P-value: {result['p_value']:.4f}")
        else:
            print("Not enough data for analysis")
        
        # Visualize first few valid results
        if len([r for r in results if not np.isnan(r['observed_stat'])]) <= 5:
            visualize_matrices(result['matrix_0'], result['matrix_1'], cds, result)
    
    # Save results
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['matrix_0', 'matrix_1']}
        for r in results
    ])
    results_df.to_csv('permutation_test_results.csv', index=False)
    
    # Calculate summary statistics for valid results
    valid_results = results_df[~results_df['p_value'].isna()]
    
    print("\nAnalysis Complete!")
    print(f"Total CDSs analyzed: {len(results_df)}")
    print(f"Valid analyses: {len(valid_results)}")
    print(f"Significant CDSs (p < 0.05): {(valid_results['p_value'] < 0.05).sum()}")

if __name__ == "__main__":
    main()
