import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Input file paths
output_path = 'output.csv'  # Contains pi values per orientation
inv_info_path = 'inv_info.csv' # Contains recurrence info per region

# Output directory
output_dir = 'analysis_results'
os.makedirs(output_dir, exist_ok=True)

# Output file paths
merged_data_path = os.path.join(output_dir, 'merged_inversion_data.csv')
long_data_path = os.path.join(output_dir, 'long_format_pi_data.csv')
model_summary_path = os.path.join(output_dir, 'mixed_model_summary.txt')
boxplot_path = os.path.join(output_dir, 'pi_boxplot_grouped.png')
interaction_plot_path = os.path.join(output_dir, 'pi_interaction_plot.png')

print("--- Starting Inversion Analysis ---")

# --- Step 1: Load Data ---
print(f"Loading data from {output_path} and {inv_info_path}...")
try:
    output_data = pd.read_csv(output_path)
    inv_info = pd.read_csv(inv_info_path)
    print(f"Loaded {len(output_data)} rows from {output_path}")
    print(f"Loaded {len(inv_info)} rows from {inv_info_path}")
except FileNotFoundError as e:
    print(f"ERROR: Cannot find input file: {e}. Exiting.")
    exit()

# --- Step 2: Initial Data Cleaning and Preparation ---
print("Preparing data for merging...")

inv_info = inv_info.rename(columns={
    'Chromosome': 'chr',
    'Start': 'region_start',
    'End': 'region_end',
    '0_single_1_recur': 'RecurrenceCode'
})

if 'RecurrenceCode' not in inv_info.columns:
    print("ERROR: Recurrence column ('0_single_1_recur' or renamed 'RecurrenceCode') not found in inv_info. Exiting.")
    exit()

inv_info.dropna(subset=['RecurrenceCode'], inplace=True)
inv_info['RecurrenceCode'] = inv_info['RecurrenceCode'].astype(int)

def standardize_chr(df, chr_col='chr'):
    if df[chr_col].astype(str).str.startswith('chr').all():
        return df
    elif not df[chr_col].astype(str).str.contains('chr').any():
        df[chr_col] = 'chr' + df[chr_col].astype(str)
        return df
    else:
         raise ValueError(f"Mixed chromosome formats found in column '{chr_col}'. Please standardize input.")

try:
    output_data = standardize_chr(output_data, 'chr')
    inv_info = standardize_chr(inv_info, 'chr')
except ValueError as e:
    print(f"ERROR: {e}. Exiting.")
    exit()

for col in ['region_start', 'region_end']:
    output_data[col] = pd.to_numeric(output_data[col], errors='coerce')
    inv_info[col] = pd.to_numeric(inv_info[col], errors='coerce')
output_data.dropna(subset=['region_start', 'region_end'], inplace=True)
inv_info.dropna(subset=['region_start', 'region_end'], inplace=True)
for col in ['region_start', 'region_end']:
    output_data[col] = output_data[col].astype(np.int64)
    inv_info[col] = inv_info[col].astype(np.int64)


# --- Step 3: Merge Dataframes ---
print("Merging dataframes...")
output_data['orig_output_index'] = output_data.index
inv_info['orig_inv_index'] = inv_info.index

merged_temp = pd.merge(
    output_data,
    inv_info[['orig_inv_index', 'chr', 'region_start', 'region_end', 'RecurrenceCode']],
    on='chr',
    how='inner',
    suffixes=('_out', '_inv')
)

coordinate_match_mask = (
    (abs(merged_temp['region_start_out'] - merged_temp['region_start_inv']) <= 1) &
    (abs(merged_temp['region_end_out'] - merged_temp['region_end_inv']) <= 1)
)
merged_filtered = merged_temp[coordinate_match_mask].copy()

merged_filtered['match_count'] = merged_filtered.groupby('orig_output_index')['orig_output_index'].transform('count')
ambiguous_matches = merged_filtered[merged_filtered['match_count'] > 1]
if not ambiguous_matches.empty:
    print(f"WARNING: Found {len(ambiguous_matches)} rows corresponding to {ambiguous_matches['orig_output_index'].nunique()} output entries that ambiguously matched multiple inv_info entries. Removing ambiguous matches.")
    merged_data = merged_filtered[merged_filtered['match_count'] == 1].copy()
else:
    merged_data = merged_filtered.copy()

if len(merged_data) == 0:
    print("ERROR: No matching regions found after merging and filtering. Check input files. Exiting.")
    exit()
print(f"Found {len(merged_data)} unambiguous matching inversion regions.")

merged_data['chr'] = merged_data['chr']
merged_data['region_start'] = merged_data['region_start_out']
merged_data['region_end'] = merged_data['region_end_out']
merged_data = merged_data[['orig_output_index', 'orig_inv_index', 'chr', 'region_start', 'region_end',
                           '0_pi_filtered', '1_pi_filtered', 'RecurrenceCode']].copy()

merged_data.to_csv(merged_data_path, index=False)
print(f"Merged wide-format data saved to {merged_data_path}")

# --- Step 4: Data Reshaping to Long Format ---
print("Reshaping data to long format...")
merged_data['InversionRegionID'] = merged_data['chr'] + ':' + \
                                  merged_data['region_start'].astype(str) + '-' + \
                                  merged_data['region_end'].astype(str)

direct_df = merged_data[['InversionRegionID', 'RecurrenceCode', '0_pi_filtered']].copy()
direct_df['Orientation'] = 'Direct'
direct_df = direct_df.rename(columns={'0_pi_filtered': 'PiValue'})

inverted_df = merged_data[['InversionRegionID', 'RecurrenceCode', '1_pi_filtered']].copy()
inverted_df['Orientation'] = 'Inverted'
inverted_df = inverted_df.rename(columns={'1_pi_filtered': 'PiValue'})

data_long = pd.concat([direct_df, inverted_df], ignore_index=True)

data_long['Recurrence'] = data_long['RecurrenceCode'].map({0: 'Single-event', 1: 'Recurrent'})
data_long.drop(columns=['RecurrenceCode'], inplace=True)

# Define as categorical with specific reference levels for the model
data_long['Orientation'] = pd.Categorical(data_long['Orientation'], categories=['Direct', 'Inverted'], ordered=False)
data_long['Recurrence'] = pd.Categorical(data_long['Recurrence'], categories=['Single-event', 'Recurrent'], ordered=False)

# --- Step 5: Handle Pi Values ---
print("Cleaning PiValue column...")
data_long['PiValue'] = data_long['PiValue'].replace([np.inf, -np.inf], np.nan)
initial_rows = len(data_long)
data_long.dropna(subset=['PiValue'], inplace=True)
final_rows = len(data_long)
print(f"Removed {initial_rows - final_rows} rows with missing/infinite PiValue.")
print(f"Proceeding with {final_rows} observations ({data_long['InversionRegionID'].nunique()} unique regions) for modeling.")

data_long.to_csv(long_data_path, index=False)
print(f"Long-format data saved to {long_data_path}")

# --- Step 6: Fit Linear Mixed-Effects Model ---
print("Fitting Linear Mixed-Effects Model...")
group_counts = data_long.groupby(['Orientation', 'Recurrence']).size()
if (group_counts < 2).any():
    print("\nWARNING: Some groups have less than 2 data points. Model estimation might be unstable or fail.")

try:
    model_formula = "PiValue ~ C(Orientation, Treatment('Direct')) * C(Recurrence, Treatment('Single-event'))"
    mixed_model = smf.mixedlm(model_formula, data_long, groups=data_long["InversionRegionID"])
    result = mixed_model.fit()
    print("Model fitting successful.")
except Exception as e:
    print(f"\nERROR: Model fitting failed: {e}")
    exit()

# --- Step 7: Display and Save Model Results ---
print("\n--- Mixed Effects Model Results ---")
print(result.summary())

with open(model_summary_path, 'w') as f:
    f.write("Mixed Linear Model Regression Results\n")
    f.write("=========================================\n")
    f.write(f"Model Formula: {model_formula}\n")
    f.write(f"Grouping Variable: InversionRegionID\n")
    f.write("Data Counts per Group:\n")
    f.write(group_counts.to_string())
    f.write("\n=========================================\n")
    f.write(result.summary().as_text())
print(f"Model summary saved to {model_summary_path}")

# --- Step 8: Visualization ---
print("Generating visualizations...")

# Set seaborn style and color palette
sns.set_theme(style="whitegrid", palette="muted")
color_palette = sns.color_palette("Set2", n_colors=2) # Choose a nice palette for Orientation

# Enhanced Boxplot with individual data points
plt.figure(figsize=(9, 7))
ax_box = sns.boxplot(x='Recurrence', y='PiValue', hue='Orientation', data=data_long,
                 palette=color_palette, showfliers=False,
                 hue_order=['Direct', 'Inverted'])

# Overlay swarmplot/stripplot to show all data points with jitter
sns.stripplot(x='Recurrence', y='PiValue', hue='Orientation', data=data_long,
              palette=color_palette, dodge=True, # Dodge points based on hue
              size=4, alpha=0.6, jitter=True, legend=False, # Disable legend for stripplot
              hue_order=['Direct', 'Inverted']) # consistent order

ax_box.set_title('Nucleotide Diversity (π) by Inversion Type and Haplotype Orientation', fontsize=16, pad=20)
ax_box.set_xlabel('Inversion Region Type', fontsize=12)
ax_box.set_ylabel('Nucleotide Diversity (π)', fontsize=12)
ax_box.tick_params(axis='both', which='major', labelsize=10)
ax_box.legend(title='Orientation', title_fontsize='11', fontsize='10', loc='upper right')
plt.tight_layout()
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
print(f"boxplot saved to {boxplot_path}")
plt.close()


# Interaction Plot
plt.figure(figsize=(8, 6))
# Using pointplot which automatically calculates and plots means and 95% CIs
ax_int = sns.pointplot(x='Orientation', y='PiValue', hue='Recurrence', data=data_long,
                       palette="viridis", # Use a different palette for distinction
                       markers=["o", "^"], linestyles=["-", "--"], dodge=0.1, # Dodge slightly for clarity
                       errorbar=('ci', 95), capsize=.1, # Show 95% CI with caps
                       hue_order=['Single-event', 'Recurrent']) # consistent order

ax_int.set_title('Interaction Plot: Mean π by Orientation and Recurrence', fontsize=16, pad=20)
ax_int.set_xlabel('Haplotype Orientation', fontsize=12)
ax_int.set_ylabel('Estimated Mean Nucleotide Diversity (π)', fontsize=12)
ax_int.tick_params(axis='both', which='major', labelsize=10)
ax_int.legend(title='Recurrence Type', title_fontsize='11', fontsize='10', loc='best')
ax_int.grid(True, axis='y', linestyle=':', alpha=0.6) # Lighter grid
plt.tight_layout()
plt.savefig(interaction_plot_path, dpi=300, bbox_inches='tight')
print(f"interaction plot saved to {interaction_plot_path}")
plt.close()

print("\n--- Analysis Complete ---")
print(f"Results saved in directory: {output_dir}")
