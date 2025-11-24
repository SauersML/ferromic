import os
import glob
import sys
import re
import pandas as pd
import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import fdrcorrection

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_PATTERN = "full_paml_results*.tsv"
OUTPUT_FILENAME = "GRAND_PAML_RESULTS.tsv"

# Columns to explicitly remove from the inputs before merging to ensure
# we rely purely on the recalculated values from the Grand Tournament.
BLACKLIST_COLUMNS = {
    # Old Branch Model stats
    'bm_p_value', 'bm_q_value', 'bm_lrt_stat', 
    'bm_omega_inverted', 'bm_omega_direct', 'bm_omega_background', 
    'bm_kappa', 'bm_lnl_h1', 'bm_lnl_h0',
    # Old Clade Model stats
    'cmc_p_value', 'cmc_q_value', 'cmc_lrt_stat',
    'cmc_p0', 'cmc_p1', 'cmc_p2', 
    'cmc_omega0', 'cmc_omega2_direct', 'cmc_omega2_inverted', 
    'cmc_kappa', 'cmc_lnl_h1', 'cmc_lnl_h0',
    # Old winner seeds
    'h0_winner_seed', 'h1_winner_seed',
    'cmc_h0_key', 'cmc_h1_key'
}

# Mapping of {New_Column_Name: Replacement_Suffix}
# This maps the metric name found in the column headers to the final output name.
# Based on pipeline_lib.py, raw columns look like: "h1_s1_def_cmc_p0"
# We find the winner via "h1_s1_def_lnl" and swap "_lnl" for the suffixes below.
H1_PARAM_MAP = {
    'winner_p0': '_cmc_p0',
    'winner_p1': '_cmc_p1',
    'winner_p2': '_cmc_p2',
    'winner_omega0': '_cmc_omega0',
    'winner_omega2_direct': '_cmc_omega2_direct',
    'winner_omega2_inverted': '_cmc_omega2_inverted',
    'winner_kappa': '_cmc_kappa'
}

# ==============================================================================
# CORE LOGIC
# ==============================================================================

def load_and_merge_files():
    """
    Loads all matching TSV files, strips old summary columns, prefixes remaining
    columns with unique run identifiers, and performs an outer join on Region+Gene.
    """
    files = glob.glob(INPUT_PATTERN)
    if not files:
        print(f"Error: No files matching '{INPUT_PATTERN}' found.")
        sys.exit(1)
    
    print(f"Found {len(files)} input files: {files}")
    
    dfs = []
    for i, fpath in enumerate(sorted(files), 1):
        print(f"Loading Run {i}: {fpath}...")
        try:
            df = pd.read_csv(fpath, sep='\t')
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue
        
        # Validate identifiers
        if 'region' not in df.columns or 'gene' not in df.columns:
            print(f"Skipping {fpath}: Missing 'region' or 'gene' identity columns.")
            continue

        # Drop old summary columns
        cols_to_drop = [c for c in df.columns if c in BLACKLIST_COLUMNS]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # Set index for alignment
        # We drop duplicates just in case a single file has the same gene twice (shouldn't happen)
        df = df.drop_duplicates(subset=['region', 'gene'])
        df.set_index(['region', 'gene'], inplace=True)
        
        # Rename data columns with run suffix
        suffix = f"_run_{i}"
        df.columns = [f"{col}{suffix}" for col in df.columns]
        
        dfs.append(df)
    
    if not dfs:
        print("No valid dataframes to merge.")
        sys.exit(1)

    print("Merging runs into master dataframe...")
    # Outer join ensures we keep a gene even if it only succeeded in one run
    combined = pd.concat(dfs, axis=1, join='outer')
    return combined

def find_best_lnl_and_params(row, hypothesis_prefix):
    """
    Scans a row for all columns belonging to a hypothesis (h0 or h1).
    Finds the Max Likelihood value and the column that produced it.
    Returns: (max_lnl, winner_seed_name, winning_col_name)
    """
    # Regex: Start with h0_ or h1_, contain _lnl, end with _run_X
    # Example: h1_s1_def_lnl_run_1
    pat = re.compile(rf"^{hypothesis_prefix}_.*_lnl_run_\d+$")
    lnl_cols = [c for c in row.index if pat.match(c)]
    
    if not lnl_cols:
        return np.nan, "no_data", None

    # Extract float values
    values = pd.to_numeric(row[lnl_cols], errors='coerce')
    
    # If all are NaN (failed runs), return NaN
    if values.isna().all():
        return np.nan, "all_fail", None
        
    # Find winner
    max_val = values.max()
    best_col = values.idxmax()
    
    # Extract seed name for reference (e.g. "s1_def_run_1")
    # Remove hypothesis prefix ("h1_") and "_lnl" part
    # "h1_s1_def_lnl_run_1" -> "s1_def_run_1"
    # We replace specifically "_lnl" to preserve the run suffix
    winner_seed = best_col[len(hypothesis_prefix)+1:].replace('_lnl', '')
    
    return max_val, winner_seed, best_col

def main():
    # 1. Load Data
    df = load_and_merge_files()
    print(f"Combined Data Dimensions: {df.shape[0]} genes, {df.shape[1]} raw columns")

    # 2. Prepare new result vectors
    h0_lnls, h0_seeds = [], []
    h1_lnls, h1_seeds = [], []
    
    # Dictionary to hold lists for each parameter we want to extract
    param_data = {k: [] for k in H1_PARAM_MAP.keys()}

    print("Conducting Grand Tournament across all seeds and runs...")
    
    # 3. Iterate and Recalculate
    for idx, row in df.iterrows():
        # --- H0 ---
        h0_val, h0_seed, _ = find_best_lnl_and_params(row, 'h0')
        h0_lnls.append(h0_val)
        h0_seeds.append(h0_seed)
        
        # --- H1 ---
        h1_val, h1_seed, h1_best_col = find_best_lnl_and_params(row, 'h1')
        h1_lnls.append(h1_val)
        h1_seeds.append(h1_seed)
        
        # --- Parameter Extraction ---
        # If we have a winning H1 column, we can find its associated parameters
        # by swapping the column suffix (e.g. _lnl -> _cmc_p0)
        if h1_best_col and pd.notna(h1_val):
            for target_key, search_suffix in H1_PARAM_MAP.items():
                # Strict replacement of '_lnl' with the parameter suffix
                # e.g. "h1_s1_def_lnl_run_1" -> "h1_s1_def_cmc_p0_run_1"
                param_col = h1_best_col.replace('_lnl', search_suffix)
                
                if param_col in df.columns:
                    param_data[target_key].append(row[param_col])
                else:
                    param_data[target_key].append(np.nan)
        else:
            # No valid H1 winner
            for k in param_data:
                param_data[k].append(np.nan)

    # 4. Assign Results to DataFrame
    df['overall_h0_lnl'] = h0_lnls
    df['h0_winner_seed'] = h0_seeds
    df['overall_h1_lnl'] = h1_lnls
    df['h1_winner_seed'] = h1_seeds
    
    for k, v in param_data.items():
        df[k] = v

    # 5. Statistics
    print("Calculating LRT, P-values, and FDR Q-values...")
    
    # Calculate differences to check for optimization consistency
    diffs = df['overall_h1_lnl'] - df['overall_h0_lnl']
    
    # Identify optimization failures: H1 significantly less than H0 (e.g. < -1e-6)
    # We treat these as invalid tests (NaN p-value), not valid null results.
    epsilon = 1e-6
    optim_fail_mask = diffs < -epsilon

    # Calculate LRT statistic
    lrt_stats = 2 * diffs
    # Mask failures as NaN
    lrt_stats[optim_fail_mask] = np.nan
    # Clip valid small negatives (noise) to 0
    lrt_stats = lrt_stats.clip(lower=0)
    
    df['overall_lrt_stat'] = lrt_stats
    
    # P-Value (Chi-squared, df=1)
    # Only where LRT is valid (not NaN)
    mask_valid = df['overall_lrt_stat'].notna()
    df.loc[mask_valid, 'overall_p_value'] = chi2.sf(df.loc[mask_valid, 'overall_lrt_stat'], df=1)
    
    # Q-Value (Benjamini-Hochberg FDR 0.05)
    df['overall_q_value'] = np.nan
    
    # We only FDR correct genes that have a valid numeric p-value
    valid_p_idx = df['overall_p_value'].dropna().index
    if len(valid_p_idx) > 0:
        pvals = df.loc[valid_p_idx, 'overall_p_value']
        rejected, qvals = fdrcorrection(pvals, alpha=0.05, method='indep')
        df.loc[valid_p_idx, 'overall_q_value'] = qvals
        print(f"FDR correction applied to {len(valid_p_idx)} genes.")
        print(f"Significant genes (q < 0.05): {sum(qvals < 0.05)}")
    else:
        print("No valid P-values found; skipping FDR.")

    # 6. Final Cleanup and Save
    # Reset index to make region/gene normal columns
    df.reset_index(inplace=True)
    
    # Define output order
    meta_cols = ['region', 'gene']
    stats_cols = [
        'overall_p_value', 'overall_q_value', 'overall_lrt_stat',
        'overall_h1_lnl', 'overall_h0_lnl',
        'h1_winner_seed', 'h0_winner_seed'
    ]
    param_cols = list(H1_PARAM_MAP.keys())
    
    # Separate remaining raw columns (the raw run data)
    # We exclude the columns we just created to avoid duplication in the sort logic
    new_cols = set(meta_cols + stats_cols + param_cols)
    raw_cols = [c for c in df.columns if c not in new_cols]
    raw_cols.sort() # Alphabetical sort for raw data
    
    final_order = meta_cols + stats_cols + param_cols + raw_cols
    
    # Create final dataframe
    df_final = df[final_order]
    
    print(f"Saving results to {OUTPUT_FILENAME}...")
    df_final.to_csv(OUTPUT_FILENAME, sep='\t', index=False, float_format='%.6g')
    print("Success.")

if __name__ == "__main__":
    main()
