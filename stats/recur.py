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
inv_info['orig_inv_index'] = inv_info.index

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

# Add an original index to output_data so we can verify every row gets matched
output_data['orig_index'] = np.arange(len(output_data))

# First, merge on 'chr' only
merged_temp = pd.merge(
    output_data,
    inv_info[['orig_inv_index', 'chr', 'region_start', 'region_end', '0_single_1_recur']],
    on='chr',
    how='inner',
    suffixes=('_out', '_inv')
)

print(f"Preliminary merge on 'chr' only: {len(merged_temp)} rows")

# Filter for rows where region_start and region_end differ by at most one
mask = (
    (abs(merged_temp['region_start_out'] - merged_temp['region_start_inv']) <= 1) &
    (abs(merged_temp['region_end_out'] - merged_temp['region_end_inv']) <= 1)
)
merged = merged_temp[mask].copy()
print(f"After filtering for one-off differences: {len(merged)} matching rows found")

if len(merged) == 0:
    raise ValueError("ERROR: No key overlap found allowing a one-off difference for region_start and region_end between datasets.")

# Identify and print inv_info rows that were dropped from the merge with a non-null recurrence value
matched_inv_indices = merged['orig_inv_index'].unique()
dropped_inv_info = inv_info[~inv_info['orig_inv_index'].isin(matched_inv_indices)]
dropped_with_recur = dropped_inv_info[dropped_inv_info['0_single_1_recur'].notna()]
print("\nRows from inv_info that were dropped from the merge but have a 0_single_1_recur value:")
print(dropped_with_recur)

# Verify that every row in output_data is matched at least once
matched_indices = merged['orig_index'].unique()
if len(matched_indices) != len(output_data):
    missing = set(range(len(output_data))) - set(matched_indices)
    raise ValueError(f"ERROR: The following output.csv row indices were not matched: {missing}")

# Use the output_data's region_start and region_end as canonical keys
merged['region_start'] = merged['region_start_out']
merged['region_end'] = merged['region_end_out']

# drop the redundant columns from inv_info
merged.drop(columns=['region_start_inv', 'region_end_inv'], inplace=True)

data = merged

print(f"After merge: {len(data)} rows")

# Check for NaN values in recurrence column after merge
if '0_single_1_recur' in data.columns:
    na_count = data['0_single_1_recur'].isna().sum()
    print(f"Rows with NaN in recurrence column after merge: {na_count} ({na_count/len(data)*100:.1f}%)")
    print(f"Recurrence classification summary: {data['0_single_1_recur'].value_counts(dropna=False).to_dict()}")
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

if len(recurrent) == 0:
    raise ValueError("ERROR: No recurrent inversions found after merge.")

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

# Save results to CSV
if results_table:
    results_df = pd.DataFrame(results_table)
    results_df.to_csv('inversion_statistical_results.csv', index=False)
    print("\nStatistical test results saved to 'inversion_statistical_results.csv'")

print("\nAnalysis complete!")
