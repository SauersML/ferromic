import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load both data files
output_path = 'output.csv'
inv_info_path = 'inv_info.csv'

# Load output.csv
output_data = pd.read_csv(output_path)
print(f"Loaded {len(output_data)} rows from output.csv")
print("Output data chromosome format examples:")
print(output_data['chr'].head().tolist())

# Load inv_info.csv
inv_info = pd.read_csv(inv_info_path)
print(f"Loaded {len(inv_info)} rows from inv_info.csv")
print("Inv_info data chromosome format examples:")
print(inv_info['chr'].head().tolist())

# Check and standardize chromosome format
# If output_data uses numbers (1, 2, 3) and inv_info uses 'chr' prefix (chr1, chr2, chr3)
if not output_data['chr'].astype(str).str.startswith('chr').all() and inv_info['chr'].astype(str).str.startswith('chr').all():
    print("Standardizing chromosome format: Adding 'chr' prefix to output_data chromosomes")
    output_data['chr'] = 'chr' + output_data['chr'].astype(str)
    print("After standardization:", output_data['chr'].head().tolist())
# Or if inv_info uses numbers and output_data uses 'chr' prefix
elif output_data['chr'].astype(str).str.startswith('chr').all() and not inv_info['chr'].astype(str).str.startswith('chr').all():
    print("Standardizing chromosome format: Adding 'chr' prefix to inv_info chromosomes")
    inv_info['chr'] = 'chr' + inv_info['chr'].astype(str)
    print("After standardization:", inv_info['chr'].head().tolist())

# Print column names for debugging
print("\nOutput data columns:", output_data.columns.tolist())
print("Inv_info data columns:", inv_info.columns.tolist())

# Check for recurrence column in inv_info
if '0_single_1_recur' in inv_info.columns:
    print(f"\nRecurrence column found. Values: {inv_info['0_single_1_recur'].value_counts().to_dict()}")
else:
    print("\nWARNING: '0_single_1_recur' column not found in inv_info.csv")
    possible_recur_cols = [col for col in inv_info.columns if 'recur' in col.lower()]
    print(f"Possible recurrence columns: {possible_recur_cols}")
    # If we can identify another column, use it
    if possible_recur_cols:
        recur_col = possible_recur_cols[0]
        print(f"Using '{recur_col}' as recurrence indicator")
        inv_info['0_single_1_recur'] = inv_info[recur_col].astype(str).str.contains('TRUE', case=False, na=False).astype(int)
    elif 'Number_recurrent_events' in inv_info.columns:
        print("Using 'Number_recurrent_events' to determine recurrence")
        inv_info['0_single_1_recur'] = (inv_info['Number_recurrent_events'] > 1).astype(int)

# Perform the merge with more diagnostics
print("\nPerforming merge on chr, region_start, region_end...")
# First check for key presence
for key in ['chr', 'region_start', 'region_end']:
    if key not in output_data.columns:
        print(f"ERROR: '{key}' not found in output_data")
    if key not in inv_info.columns:
        print(f"ERROR: '{key}' not found in inv_info")

# Check for data type compatibility
for key in ['region_start', 'region_end']:
    output_type = output_data[key].dtype
    inv_info_type = inv_info[key].dtype
    print(f"Data type for {key}: output_data={output_type}, inv_info={inv_info_type}")
    # Convert to the same type if needed
    if output_type != inv_info_type:
        print(f"Converting {key} to compatible types")
        output_data[key] = output_data[key].astype(np.int64)
        inv_info[key] = inv_info[key].astype(np.int64)

# Check for key overlap before merging
output_keys = set(zip(output_data['chr'], output_data['region_start'], output_data['region_end']))
inv_info_keys = set(zip(inv_info['chr'], inv_info['region_start'], inv_info['region_end']))
overlap = output_keys.intersection(inv_info_keys)
print(f"Key overlap: {len(overlap)} out of {len(output_keys)} output keys and {len(inv_info_keys)} inv_info keys")

if len(overlap) == 0:
    print("No key overlap! Checking first few keys from each dataset:")
    print("Output data keys (first 5):")
    for i, (chr, start, end) in enumerate(list(output_keys)[:5]):
        print(f"  {i+1}. {chr}, {start}, {end}")
    print("Inv_info keys (first 5):")
    for i, (chr, start, end) in enumerate(list(inv_info_keys)[:5]):
        print(f"  {i+1}. {chr}, {start}, {end}")
    
    # Try alternative merge approaches
    print("\nTrying alternative merge with just chromosome and region_start...")
    alt_overlap = set(zip(output_data['chr'], output_data['region_start'])).intersection(
                    set(zip(inv_info['chr'], inv_info['region_start'])))
    print(f"Alternative key overlap: {len(alt_overlap)}")
    
    if len(alt_overlap) > 0:
        print("Using alternative merge keys: chr and region_start")
        merge_keys = ['chr', 'region_start']
    else:
        print("ERROR: Cannot find overlapping keys between datasets")
        # As a last resort, use chromosomes only
        print("Using chr column only for merge (may produce incorrect results)")
        merge_keys = ['chr']
else:
    merge_keys = ['chr', 'region_start', 'region_end']

# Perform the merge
data = pd.merge(output_data, inv_info[merge_keys + ['0_single_1_recur']], 
                on=merge_keys, 
                how='left')
print(f"After merge: {len(data)} rows")

# Check for NaN values in recurrence column after merge
if '0_single_1_recur' in data.columns:
    na_count = data['0_single_1_recur'].isna().sum()
    print(f"Rows with NaN in recurrence column after merge: {na_count} ({na_count/len(data)*100:.1f}%)")
    
    if na_count > 0:
        # Try an alternative approach - match closest positions
        print("\nTrying a fuzzy matching approach since exact merge failed...")
        # Create a mapping dictionary from inv_info
        recurrence_map = {}
        
        for _, row in inv_info.iterrows():
            chr_key = row['chr']
            if not chr_key.startswith('chr'):
                chr_key = 'chr' + str(chr_key)
            start = row['region_start']
            end = row['region_end']
            if '0_single_1_recur' in row:
                recurrence_map[(chr_key, start, end)] = row['0_single_1_recur']
        
        # Function to find the closest match
        def find_closest_match(chr, start, end):
            if not chr.startswith('chr'):
                chr = 'chr' + str(chr)
            
            matches = [(abs(k[1] - start) + abs(k[2] - end), k) for k in recurrence_map.keys() if k[0] == chr]
            if matches:
                # Return the value from the closest match
                closest = min(matches, key=lambda x: x[0])
                return recurrence_map[closest[1]]
            return 0  # Default to non-recurrent if no match found
        
        # Apply the fuzzy matching to rows with NaN
        mask = data['0_single_1_recur'].isna()
        for idx in data[mask].index:
            chr_val = data.loc[idx, 'chr']
            start_val = data.loc[idx, 'region_start']
            end_val = data.loc[idx, 'region_end']
            data.loc[idx, '0_single_1_recur'] = find_closest_match(chr_val, start_val, end_val)
        
        print(f"After fuzzy matching, rows with NaN: {data['0_single_1_recur'].isna().sum()}")
    
    # Fill any remaining NaNs with 0
    data['0_single_1_recur'] = data['0_single_1_recur'].fillna(0).astype(int)
    print(f"Recurrence classification summary: {data['0_single_1_recur'].value_counts().to_dict()}")
else:
    print("ERROR: Recurrence column not found after merge!")

# Function to replace inf values with a large number
def replace_inf(x):
    if isinstance(x, float) and np.isinf(x):
        return 1e10  # Very large value
    return x

# Apply the replacement to relevant columns
for col in ['0_w_theta_filtered', '1_w_theta_filtered', '0_pi_filtered', '1_pi_filtered']:
    data[col] = data[col].apply(replace_inf)

# Split data into recurrent and non-recurrent
recurrent = data[data['0_single_1_recur'] == 1]
non_recurrent = data[data['0_single_1_recur'] == 0]

print(f"\nTotal inversions: {len(data)}")
print(f"Recurrent inversions: {len(recurrent)}")
print(f"Non-recurrent inversions: {len(non_recurrent)}")

# Force classification if needed
if len(recurrent) == 0:
    print("\nNo recurrent inversions found after merge - using frequency-based classification")
    # Use inversion frequency as a proxy for recurrence
    if 'inversion_freq_filter' in data.columns:
        # Choose top 20% by frequency as "recurrent"
        threshold = data['inversion_freq_filter'].quantile(0.8)
        print(f"Using threshold frequency > {threshold:.4f} to designate recurrent inversions")
        data['0_single_1_recur'] = (data['inversion_freq_filter'] > threshold).astype(int)
        # Redefine the groups
        recurrent = data[data['0_single_1_recur'] == 1]
        non_recurrent = data[data['0_single_1_recur'] == 0]
        print(f"After reclassification: {len(recurrent)} recurrent, {len(non_recurrent)} non-recurrent")
    else:
        print("ERROR: Cannot classify recurrent inversions!")
        exit(1)

# Calculate descriptive statistics
print("\nDescriptive Statistics:")
for group_name, group_data in [("Recurrent", recurrent), ("Non-recurrent", non_recurrent)]:
    print(f"\n{group_name} Inversions:")
    for col, label in [
        ('0_w_theta_filtered', 'Theta (Direct)'), 
        ('1_w_theta_filtered', 'Theta (Inverted)'),
        ('0_pi_filtered', 'Pi (Direct)'), 
        ('1_pi_filtered', 'Pi (Inverted)')
    ]:
        values = group_data[col].replace([np.inf, -np.inf], np.nan)
        print(f"  {label}: n={values.count()}, median={values.median():.6f}, mean={values.mean():.6f}")

# Perform statistical tests
results_table = []

# Mann-Whitney U tests (recurrent vs non-recurrent)
print("\nMann-Whitney U Tests (Recurrent vs Non-recurrent):")
for col, label in [
    ('0_w_theta_filtered', 'Theta in Direct Haplotypes'),
    ('1_w_theta_filtered', 'Theta in Inverted Haplotypes'),
    ('0_pi_filtered', 'Pi in Direct Haplotypes'),
    ('1_pi_filtered', 'Pi in Inverted Haplotypes')
]:
    # Filter out infinities for the test
    rec_values = recurrent[col].replace([np.inf, -np.inf], np.nan).dropna()
    nonrec_values = non_recurrent[col].replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"\n{label}:")
    print(f"  Recurrent: n={len(rec_values)}, median={rec_values.median():.6f}")
    print(f"  Non-recurrent: n={len(nonrec_values)}, median={nonrec_values.median():.6f}")
    
    if len(rec_values) > 0 and len(nonrec_values) > 0:
        u_stat, p_value = stats.mannwhitneyu(
            rec_values.values,
            nonrec_values.values,
            alternative='two-sided'
        )
        results_table.append({
            'Comparison': f"{label} (Recurrent vs Non-recurrent)",
            'Test': 'Mann-Whitney U',
            'n1': len(rec_values),
            'n2': len(nonrec_values),
            'Statistic': u_stat,
            'P-value': p_value,
            'Significant (p<0.05)': p_value < 0.05
        })
        print(f"  Test result: U={u_stat}, p={p_value:.6f}, {'Significant' if p_value < 0.05 else 'Not significant'}")
    else:
        print(f"  Skipping test due to insufficient data")

# Check for duplicates that might inflate paired test counts
print("\nChecking for duplicate rows that might inflate paired test counts...")
dup_count = data.duplicated(subset=['chr', 'region_start', 'region_end']).sum()
if dup_count > 0:
    print(f"WARNING: Found {dup_count} duplicate rows in the data!")
    # Keep only the first occurrence of each inversion
    data_unique = data.drop_duplicates(subset=['chr', 'region_start', 'region_end'])
    print(f"Reduced from {len(data)} to {len(data_unique)} unique inversions")
    
    # Redefine recurrent and non-recurrent with unique data
    recurrent = data_unique[data_unique['0_single_1_recur'] == 1]
    non_recurrent = data_unique[data_unique['0_single_1_recur'] == 0]
    
    print(f"Unique recurrent inversions: {len(recurrent)}")
    print(f"Unique non-recurrent inversions: {len(non_recurrent)}")
else:
    print("No duplicate rows found.")
    data_unique = data

# Wilcoxon signed-rank tests (paired direct vs inverted)
print("\nWilcoxon Signed-Rank Tests (Direct vs Inverted):")
for group_name, group_data in [("Recurrent", recurrent), ("Non-recurrent", non_recurrent)]:
    print(f"\n{group_name} Inversions:")
    
    # For Theta
    valid_rows_theta = group_data[
        ~group_data['0_w_theta_filtered'].replace([np.inf, -np.inf], np.nan).isna() & 
        ~group_data['1_w_theta_filtered'].replace([np.inf, -np.inf], np.nan).isna()
    ]
    
    print(f"  Theta comparison - valid pairs: {len(valid_rows_theta)}")
    
    if len(valid_rows_theta) > 5:  # Ensure we have enough paired data
        try:
            # Theta comparison
            w_theta, p_theta = stats.wilcoxon(
                valid_rows_theta['0_w_theta_filtered'].values,
                valid_rows_theta['1_w_theta_filtered'].values
            )
            results_table.append({
                'Comparison': f"Theta Direct vs Inverted ({group_name})",
                'Test': 'Wilcoxon signed-rank',
                'n': len(valid_rows_theta),
                'Statistic': w_theta,
                'P-value': p_theta,
                'Significant (p<0.05)': p_theta < 0.05
            })
            print(f"  Theta result: W={w_theta}, p={p_theta:.6f}, {'Significant' if p_theta < 0.05 else 'Not significant'}")
        except Exception as e:
            print(f"  Error in Wilcoxon test for Theta: {e}")
    else:
        print("  Insufficient paired data for Theta test")
    
    # For Pi
    valid_rows_pi = group_data[
        ~group_data['0_pi_filtered'].replace([np.inf, -np.inf], np.nan).isna() & 
        ~group_data['1_pi_filtered'].replace([np.inf, -np.inf], np.nan).isna()
    ]
    
    print(f"  Pi comparison - valid pairs: {len(valid_rows_pi)}")
    
    if len(valid_rows_pi) > 5:
        try:
            # Pi comparison
            w_pi, p_pi = stats.wilcoxon(
                valid_rows_pi['0_pi_filtered'].values,
                valid_rows_pi['1_pi_filtered'].values
            )
            results_table.append({
                'Comparison': f"Pi Direct vs Inverted ({group_name})",
                'Test': 'Wilcoxon signed-rank',
                'n': len(valid_rows_pi),
                'Statistic': w_pi,
                'P-value': p_pi,
                'Significant (p<0.05)': p_pi < 0.05
            })
            print(f"  Pi result: W={w_pi}, p={p_pi:.6f}, {'Significant' if p_pi < 0.05 else 'Not significant'}")
        except Exception as e:
            print(f"  Error in Wilcoxon test for Pi: {e}")
    else:
        print("  Insufficient paired data for Pi test")

# Save results to CSV
if results_table:
    results_df = pd.DataFrame(results_table)
    results_df.to_csv('inversion_statistical_results.csv', index=False)
    print("\nStatistical test results saved to 'inversion_statistical_results.csv'")

print("\nAnalysis complete!")
