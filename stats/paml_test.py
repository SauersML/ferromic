import pandas as pd
import numpy as np
from scipy.stats import chi2, kstest, uniform
import sys
import os
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_FILE = "GRAND_PAML_RESULTS.tsv"
OUTPUT_FILE = "GLOBAL_TEST_RESULTS.txt"
BOOTSTRAP_REPLICATES = 1000000
SEED = 42

def load_and_prep_data(filepath):
    print(f"Loading {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found in current directory.")
        sys.exit(1)

    df = pd.read_csv(filepath, sep='\t')
    
    # Ensure we have necessary columns
    req_cols = ['region', 'gene', 'overall_h0_lnl', 'overall_h1_lnl']
    if not all(col in df.columns for col in req_cols):
        print(f"Error: Input file missing one of {req_cols}")
        sys.exit(1)

    # Filter invalid runs (NaN likelihoods)
    initial_len = len(df)
    df = df.dropna(subset=['overall_h0_lnl', 'overall_h1_lnl'])
    print(f"Data loaded. Dropped {initial_len - len(df)} rows with missing likelihoods. Total genes: {len(df)}")

    # 1. Recalculate LRT Statistic
    # We recalculate to ensure we are using raw likelihoods and handling negative noise correctly
    df['lrt_recalc'] = 2 * (df['overall_h1_lnl'] - df['overall_h0_lnl'])
    
    # Clip negative values (optimization noise) to 0
    df.loc[df['lrt_recalc'] < 0, 'lrt_recalc'] = 0

    # 2. Calculate P-values using Chi-squared DF=1 (NOT Mixture)
    # Survival function (sf) is 1 - cdf
    df['pval_recalc'] = chi2.sf(df['lrt_recalc'], df=1)

    return df

def calculate_ks_statistic(p_values):
    """
    Calculates the Kolmogorov-Smirnov D statistic comparing 
    observed p-values to a Uniform(0,1) distribution.
    High D indicates deviation from the Null (Neutrality).
    """
    # We test against the uniform CDF
    res = kstest(p_values, 'uniform')
    return res.statistic

def simulate_theoretical_null_mean(n_genes, replicates=1000):
    """
    Simulates purely random unlinked data to find where the KS statistic 
    should be centered if there were NO signal and NO linkage.
    """
    print(f"Simulating theoretical null baseline (N={n_genes})...")
    null_stats = []
    for _ in range(replicates):
        # Generate N random p-values from Uniform(0,1)
        random_p = np.random.uniform(0, 1, n_genes)
        stat = calculate_ks_statistic(random_p)
        null_stats.append(stat)
    return np.mean(null_stats)

def block_bootstrap_test(df, n_replicates):
    """
    Performs Block Bootstrap resampling Regions to account for Linkage.
    """
    
    # 1. Calculate Observed Global Statistic
    obs_ks = calculate_ks_statistic(df['pval_recalc'])
    print(f"\nObserved Global KS Statistic: {obs_ks:.6f}")

    # 2. Prepare for Bootstrap
    regions = df['region'].unique()
    n_regions = len(regions)
    
    # Group data by region for fast access
    region_groups = {r: df[df['region'] == r]['pval_recalc'].values for r in regions}
    
    print(f"Starting Block Bootstrap ({n_replicates} replicates) resampling {n_regions} regions...")
    
    bootstrap_stats = []
    
    # Pre-generate random indices for speed
    rng = np.random.default_rng(SEED)
    
    for _ in tqdm(range(n_replicates)):
        # Sample region IDs with replacement
        resampled_regions = rng.choice(regions, size=n_regions, replace=True)
        
        # Collect p-values from selected regions (preserve block structure)
        # Using list comprehension for speed
        resampled_pvals = np.concatenate([region_groups[r] for r in resampled_regions])
        
        # Calculate KS for this bootstrap sample
        stat = calculate_ks_statistic(resampled_pvals)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)
    
    # 3. Centering / Shifting
    # The bootstrap distribution is centered on the Observed Statistic (because we sampled observed data).
    # To get a Null Distribution, we shift it so its mean aligns with the Theoretical Null Mean.
    # This preserves the Variance (spread) caused by Linkage, but moves the center to 0 signal.
    
    boot_mean = np.mean(bootstrap_stats)
    theoretical_null_mean = simulate_theoretical_null_mean(len(df))
    
    shift = boot_mean - theoretical_null_mean
    null_distribution = bootstrap_stats - shift
    
    print(f"\nBootstrap Mean: {boot_mean:.6f}")
    print(f"Theoretical Null Mean (No Signal, No Linkage): {theoretical_null_mean:.6f}")
    print(f"Shift applied to create Null Distribution: {shift:.6f}")

    # 4. Calculate P-value
    # How many times did the Null Distribution produce a value >= Observed?
    # (One-sided test: we care if observed deviation is larger than expected noise)
    n_extreme = np.sum(null_distribution >= obs_ks)
    p_value = (n_extreme + 1) / (n_replicates + 1)
    
    return obs_ks, p_value, n_extreme

def main():
    # Load
    df = load_and_prep_data(INPUT_FILE)
    
    # Run Test
    obs_stat, p_val, n_ext = block_bootstrap_test(df, BOOTSTRAP_REPLICATES)
    
    # Report
    print("\n" + "="*50)
    print("GLOBAL OMNIBUS TEST FOR SELECTION")
    print("="*50)
    print(f"Input File:       {INPUT_FILE}")
    print(f"Total Genes:      {len(df)}")
    print(f"Unique Regions:   {df['region'].nunique()}")
    print(f"Metric:           Kolmogorov-Smirnov (KS) vs Uniform(0,1)")
    print(f"Null Model:       Chi-squared (df=1)")
    print(f"Correction:       Block Bootstrap (Regions) for Linkage")
    print("-" * 50)
    print(f"Observed KS Stat: {obs_stat:.6f}")
    print(f"Global P-Value:   {p_val:.6g}")
    print("-" * 50)
    
    result_msg = ""
    if p_val < 0.05:
        result_msg = "RESULT: SIGNIFICANT global signal of selection."
    else:
        result_msg = "RESULT: NO significant global signal detected."
    
    print(result_msg)
    print("="*50)

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write("Metric\tValue\n")
        f.write(f"Genes\t{len(df)}\n")
        f.write(f"Regions\t{df['region'].nunique()}\n")
        f.write(f"Observed_KS_Stat\t{obs_stat}\n")
        f.write(f"Global_P_Value\t{p_val}\n")
        f.write(f"Bootstrap_Replicates\t{BOOTSTRAP_REPLICATES}\n")
        f.write(f"Conclusion\t{result_msg}\n")
    
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
