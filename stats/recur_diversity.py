import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load both data files
output_path = 'output.csv'
inv_info_path = 'inv_info.tsv'

# Load output.csv
output_data = pd.read_csv(output_path)
print(f"Loaded {len(output_data)} rows from output.csv")
print("Output data chromosome format examples:")
print(output_data['chr'].head().tolist())

# Load inv_info.tsv
inv_info = pd.read_csv(inv_info_path, sep='\t')
print(f"Loaded {len(inv_info)} rows from inv_info.tsv")

# Rename columns to match expected format
inv_info = inv_info.rename(columns={
    'Chromosome': 'chr',
    'Start': 'region_start',
    'End': 'region_end'
})

print("Inv_info data chromosome format examples:")
print(inv_info['chr'].head().tolist())
inv_info['orig_inv_index'] = inv_info.index

# Check and standardize chromosome format
if not output_data['chr'].astype(str).str.startswith('chr').all() and inv_info['chr'].astype(str).str.startswith('chr').all():
    print("Standardizing chromosome format: Adding 'chr' prefix to output_data chromosomes")
    output_data['chr'] = 'chr' + output_data['chr'].astype(str)
elif output_data['chr'].astype(str).str.startswith('chr').all() and not inv_info['chr'].astype(str).str.startswith('chr').all():
    print("Standardizing chromosome format: Adding 'chr' prefix to inv_info chromosomes")
    inv_info['chr'] = 'chr' + inv_info['chr'].astype(str)

# Print column names for debugging
print("\nOutput data columns:", output_data.columns.tolist())
print("Inv_info data columns:", inv_info.columns.tolist())

# Check for recurrence column in inv_info
if '0_single_1_recur' in inv_info.columns:
    print(f"\nRecurrence column found. Values: {inv_info['0_single_1_recur'].value_counts().to_dict()}")
else:
    print("\nNo recurrence column ('0_single_1_recur') found in inv_info!")

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

# Check for ambiguous matches (one output entry matching multiple inv_info entries)
duplicate_matches = merged.duplicated(subset=['orig_index'], keep=False)
if duplicate_matches.any():
    print(f"WARNING: {duplicate_matches.sum()} rows have ambiguous matches!")
    print("The following output entries match multiple inv_info entries:")
    ambiguous_indices = merged[duplicate_matches]['orig_index'].unique()
    for idx in ambiguous_indices:
        matches = merged[merged['orig_index'] == idx]
        print(f"  Output index {idx} matches {len(matches)} inv_info entries:")
        for _, match in matches.iterrows():
            print(f"    inv_info index: {match['orig_inv_index']}, chr: {match['chr']}, " +
                  f"region: {match['region_start_inv']}-{match['region_end_inv']}, " +
                  f"recurrence: {match['0_single_1_recur']}")

if len(merged) == 0:
    raise ValueError("ERROR: No key overlap found allowing a one-off difference for region_start and region_end between datasets.")

# Use the output_data's region_start and region_end as canonical keys
merged['region_start'] = merged['region_start_out']
merged['region_end'] = merged['region_end_out']

# drop the redundant columns from inv_info
merged.drop(columns=['region_start_inv', 'region_end_inv'], inplace=True)

data = merged

# Check for NaN values in recurrence column after merge
if '0_single_1_recur' in data.columns:
    na_count = data['0_single_1_recur'].isna().sum()
    print(f"Rows with NaN in recurrence column after merge: {na_count} ({na_count/len(data)*100:.1f}%)")
else:
    print("ERROR: Recurrence column not found after merge!")

# Function to replace inf values with a large number
def replace_inf(x):
    if isinstance(x, float) and np.isinf(x):
        print("INF detected")
        return 1e10  # Very large value
    return x

# Apply the replacement to relevant columns
for col in ['0_pi_filtered', '1_pi_filtered']:
    data[col] = data[col].apply(replace_inf)

# Split data into recurrent and non-recurrent
recurrent = data[data['0_single_1_recur'] == 1]
non_recurrent = data[data['0_single_1_recur'] == 0]

# Check if any entries appear in both categories
recurrent_indices = set(recurrent['orig_index'])
non_recurrent_indices = set(non_recurrent['orig_index'])
overlap_indices = recurrent_indices.intersection(non_recurrent_indices)

if overlap_indices:
    print(f"WARNING: {len(overlap_indices)} entries classified as both recurrent AND non-recurrent!")
    print("The following entries appear in both categories:")
    for idx in list(overlap_indices)[:10]:  # Show first 10 examples if many
        rec_match = recurrent[recurrent['orig_index'] == idx].iloc[0]
        nonrec_match = non_recurrent[non_recurrent['orig_index'] == idx].iloc[0]
        print(f"  Output index {idx}, chr: {rec_match['chr']}, region: {rec_match['region_start']}-{rec_match['region_end']}")
        print(f"    Matched to RECURRENT entry (inv_info idx: {rec_match['orig_inv_index']})")
        print(f"    Matched to NON-RECURRENT entry (inv_info idx: {nonrec_match['orig_inv_index']})")
    if len(overlap_indices) > 10:
        print(f"    ... and {len(overlap_indices) - 10} more")

# Descriptive Statistics
print("\nDescriptive Statistics:")
for group_name, group_data in [("Recurrent", recurrent), ("Non-recurrent", non_recurrent)]:
    print(f"\n{group_name} Inversions:")
    for col, label in [
        ('0_pi_filtered', 'Pi (Direct)'), 
        ('1_pi_filtered', 'Pi (Inverted)')
    ]:
        values = group_data[col].replace([np.inf, -np.inf], np.nan)
        print(f"  {label}: n={values.count()}, median={values.median():.6f}, mean={values.mean():.6f}")

# Check for Similar Shape of Distributions (Using Kolmogorov-Smirnov Test)
print("\nKolmogorov-Smirnov Test for Similarity of Distributions:")
for col, label in [
    ('0_pi_filtered', 'Pi in Direct Haplotypes'),
    ('1_pi_filtered', 'Pi in Inverted Haplotypes')
]:
    print(f"\n{label}:")
    
    rec_values = recurrent[col].replace([np.inf, -np.inf], np.nan).dropna()
    nonrec_values = non_recurrent[col].replace([np.inf, -np.inf], np.nan).dropna()

    if len(rec_values) > 1 and len(nonrec_values) > 1:
        # Perform Kolmogorov-Smirnov test for comparing distributions
        ks_stat, ks_p_value = stats.ks_2samp(rec_values, nonrec_values)
        print(f"  Kolmogorov-Smirnov test p-value={ks_p_value:.6f}")
        if ks_p_value < 0.05:
            print(f"Note: distributions are significantly different (p < 0.05)")
    else:
        print(f"  Skipping Kolmogorov-Smirnov test due to insufficient data")

# Mann-Whitney U Tests (Recurrent vs Non-recurrent)
results_table = []
print("\nMann-Whitney U Tests (Recurrent vs Non-recurrent):")
for col, label in [
    ('0_pi_filtered', 'Pi in Direct Haplotypes'),
    ('1_pi_filtered', 'Pi in Inverted Haplotypes')
]:
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

# Save results to CSV
if results_table:
    results_df = pd.DataFrame(results_table)
    results_df.to_csv('inversion_statistical_results.csv', index=False)
    print("\nStatistical test results saved to 'inversion_statistical_results.csv'")

# Add violin plots comparing recurrent vs. single-event inversions
plt.figure(figsize=(12, 6))

# Subplot for direct haplotypes (0_pi_filtered)

# Subplot for direct haplotypes (0_pi_filtered)
plt.subplot(1, 2, 1)
direct_data = [
    non_recurrent['0_pi_filtered'].replace([np.inf, -np.inf], np.nan).dropna().values,
    recurrent['0_pi_filtered'].replace([np.inf, -np.inf], np.nan).dropna().values
]
violin_parts = plt.violinplot(direct_data, showmeans=False, showmedians=False, showextrema=False)

# Add styling to violin plot
for pc in violin_parts['bodies']:
    pc.set_facecolor('#2196F3')
    pc.set_edgecolor('black')
    pc.set_alpha(0.6)  # Slightly more transparent to see points

# Add jittered data points
for i, dataset in enumerate(direct_data):
    # Create jitter
    x = np.random.normal(i+1, 0.05, size=len(dataset))
    plt.scatter(x, dataset, s=20, alpha=0.6, c='#0D47A1', edgecolor='white', linewidth=0.5)

# Add medians as horizontal lines
medians = []
for dataset in direct_data:
    if len(dataset) > 0:
        medians.append(np.median(dataset))
    else:
        medians.append(np.nan)

plt.hlines(medians, [0.8, 1.8], [1.2, 2.2], colors='black', linestyles='solid', lw=2)

plt.xticks([1, 2], ['Single-event', 'Recurrent'])
plt.title('Direct Haplotypes (Pi)')
plt.ylabel('Pi Value')

# Subplot for inverted haplotypes (1_pi_filtered)
plt.subplot(1, 2, 2)
inverted_data = [
    non_recurrent['1_pi_filtered'].replace([np.inf, -np.inf], np.nan).dropna().values,
    recurrent['1_pi_filtered'].replace([np.inf, -np.inf], np.nan).dropna().values
]
violin_parts = plt.violinplot(inverted_data, showmeans=False, showmedians=False, showextrema=False)

# Add styling to violin plot
for pc in violin_parts['bodies']:
    pc.set_facecolor('#F44336')
    pc.set_edgecolor('black')
    pc.set_alpha(0.6)  # Slightly more transparent to see points

# Add jittered data points
for i, dataset in enumerate(inverted_data):
    # Create jitter
    x = np.random.normal(i+1, 0.05, size=len(dataset))
    plt.scatter(x, dataset, s=20, alpha=0.6, c='#B71C1C', edgecolor='white', linewidth=0.5)

# Add medians as horizontal lines
medians = []
for dataset in inverted_data:
    if len(dataset) > 0:
        medians.append(np.median(dataset))
    else:
        medians.append(np.nan)

plt.hlines(medians, [0.8, 1.8], [1.2, 2.2], colors='black', linestyles='solid', lw=2)

plt.xticks([1, 2], ['Single-event', 'Recurrent'])
plt.title('Inverted Haplotypes (Pi)')
plt.ylabel('Pi Value')

# Subplot for inverted haplotypes (1_pi_filtered)
plt.subplot(1, 2, 2)
inverted_data = [
    non_recurrent['1_pi_filtered'].replace([np.inf, -np.inf], np.nan).dropna().values,
    recurrent['1_pi_filtered'].replace([np.inf, -np.inf], np.nan).dropna().values
]
violin_parts = plt.violinplot(inverted_data, showmeans=False, showmedians=False, showextrema=False)

# Add styling to violin plot
for pc in violin_parts['bodies']:
    pc.set_facecolor('#F44336')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

# Add medians as horizontal lines
medians = []
for dataset in inverted_data:
    if len(dataset) > 0:
        medians.append(np.median(dataset))
    else:
        medians.append(np.nan)

plt.hlines(medians, [0.8, 1.8], [1.2, 2.2], colors='black', linestyles='solid', lw=2)

plt.xticks([1, 2], ['Single-event', 'Recurrent'])
plt.title('Inverted Haplotypes (Pi)')
plt.ylabel('Pi Value')

plt.tight_layout()
plt.savefig('inversion_pi_violins.png', dpi=300)

# Ensure each inversion maps to a single recurrence label
uniq_recur_counts = data.groupby('orig_index')['0_single_1_recur'].nunique()
valid_indices = uniq_recur_counts[uniq_recur_counts == 1].index
inter_df = data[data['orig_index'].isin(valid_indices)].copy()

# Prepare numeric pi values and clean non-finite and sentinel values that indicate prior infinity handling
inter_df['pi_direct'] = pd.to_numeric(inter_df['0_pi_filtered'], errors='coerce')
inter_df['pi_inverted'] = pd.to_numeric(inter_df['1_pi_filtered'], errors='coerce')
inter_df.replace([np.inf, -np.inf], np.nan, inplace=True)
inter_df.loc[inter_df['pi_direct'] >= 1000000000, 'pi_direct'] = np.nan
inter_df.loc[inter_df['pi_inverted'] >= 1000000000, 'pi_inverted'] = np.nan
inter_df = inter_df.dropna(subset=['pi_direct', 'pi_inverted', '0_single_1_recur'])

# Log transform for variance stabilization
inter_df['log_pi_direct'] = np.log1p(inter_df['pi_direct'])
inter_df['log_pi_inverted'] = np.log1p(inter_df['pi_inverted'])

# Compute within-inversion paired differences on the log scale
paired = inter_df[['orig_index', '0_single_1_recur', 'log_pi_direct', 'log_pi_inverted']].dropna()
paired['delta_log_pi'] = paired['log_pi_inverted'] - paired['log_pi_direct']

# Split deltas by recurrence group
delta_rec = paired.loc[paired['0_single_1_recur'] == 1, 'delta_log_pi'].dropna()
delta_nonrec = paired.loc[paired['0_single_1_recur'] == 0, 'delta_log_pi'].dropna()

print("\nPaired within-inversion tests on log-transformed pi:")
if len(delta_rec) > 0 and not np.allclose(delta_rec.values, 0.0):
    w_stat_rec, w_p_rec = stats.wilcoxon(delta_rec)
    print(f"  Recurrent: n={len(delta_rec)}, Wilcoxon signed-rank statistic={w_stat_rec}, p={w_p_rec:.6f}")
else:
    w_stat_rec, w_p_rec = np.nan, np.nan
    print("  Recurrent: insufficient non-zero paired differences")

if len(delta_nonrec) > 0 and not np.allclose(delta_nonrec.values, 0.0):
    w_stat_nonrec, w_p_nonrec = stats.wilcoxon(delta_nonrec)
    print(f"  Non-recurrent: n={len(delta_nonrec)}, Wilcoxon signed-rank statistic={w_stat_nonrec}, p={w_p_nonrec:.6f}")
else:
    w_stat_nonrec, w_p_nonrec = np.nan, np.nan
    print("  Non-recurrent: insufficient non-zero paired differences")

print("\nBetween-group test on paired deltas (inverted minus direct):")
if len(delta_rec) > 0 and len(delta_nonrec) > 0:
    u_stat_delta, p_delta = stats.mannwhitneyu(delta_rec.values, delta_nonrec.values, alternative='two-sided')
    print(f"  Mann-Whitney U on deltas: U={u_stat_delta}, p={p_delta:.6f}")
else:
    u_stat_delta, p_delta = np.nan, np.nan
    print("  Insufficient paired data for between-group test")

# Save paired and between-group results
paired_results = pd.DataFrame([
    {'Comparison': 'Paired (Recurrent): inverted - direct (log)', 'Test': 'Wilcoxon signed-rank', 'n': len(delta_rec), 'Statistic': w_stat_rec, 'P-value': w_p_rec},
    {'Comparison': 'Paired (Non-recurrent): inverted - direct (log)', 'Test': 'Wilcoxon signed-rank', 'n': len(delta_nonrec), 'Statistic': w_stat_nonrec, 'P-value': w_p_nonrec},
    {'Comparison': 'Between groups: delta_log_pi (Recurrent vs Non-recurrent)', 'Test': 'Mann-Whitney U', 'n1': len(delta_rec), 'n2': len(delta_nonrec), 'Statistic': u_stat_delta, 'P-value': p_delta}
])
paired_results.to_csv('inversion_within_between_delta_tests.csv', index=False)

plt.close()
print("\nViolin plots saved to 'inversion_pi_violins.png'")
print("\nAnalysis complete!")
