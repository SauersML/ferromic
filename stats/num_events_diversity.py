import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import re
import os
import warnings

"""
Each 'InversionRegionID' group contains only a single observation (N=1). This
data structure prevents the LMM from properly estimating the random effect variance.
Consequently, while fixed effect coefficients might be estimated, the calculation
of their standard errors, p-values, and confidence intervals within the 'mixedlm'
framework under these specific N=1 conditions can be hard to interpret.
"""

# --- Configuration ---
PI_DATA_PATH = 'output.csv'        # Pi values per orientation (0_pi_filtered, 1_pi_filtered)
INV_INFO_PATH = 'inv_info.tsv'     # Inversion info (Recurrence, Num events string, coords)
OUTPUT_DIR = 'recurrent_events_analysis_separate_v2' # Directory for analysis results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
MERGED_DATA_WIDE_PATH = os.path.join(OUTPUT_DIR, 'merged_recurrent_wide.csv')
LONG_DATA_PATH = os.path.join(OUTPUT_DIR, 'model_ready_long_data.csv')
MODEL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'lmm_recurrent_events_separate_summary.txt')
SCATTER_PLOT_PATH = os.path.join(OUTPUT_DIR, 'pi_vs_recurrent_events_separate_lmm_plot.png')

# --- Helper Functions ---

def extract_numeric_value(s):
    """Extract the leading numeric value from strings like '13 [7.00 ,13.75]'."""
    if pd.isna(s) or s == "NA":
        return np.nan
    # Attempt to find the first number (integer or float, potentially scientific notation)
    match = re.search(r'^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', str(s))
    if match:
        value_str = match.group(1)
        try:
            num = float(value_str)
            # Return as int if it's effectively an integer
            return int(num) if num.is_integer() else num
        except ValueError:
            return np.nan # Handle cases where conversion fails despite regex match
    return np.nan # No numeric pattern found at the start

def standardize_chr(df, chr_col):
    """Ensure chromosome column values start with 'chr'."""
    df[chr_col] = df[chr_col].astype(str)
    df[chr_col] = df[chr_col].apply(lambda x: x if x.startswith('chr') else 'chr' + x)
    return df

# --- Main Analysis Script ---

def main():
    print("--- Starting Recurrent Events Analysis (Separate Models) ---")

    # --- Load Data ---
    print(f"Loading data from {PI_DATA_PATH} and {INV_INFO_PATH}...")
    try:
        pi_data = pd.read_csv(PI_DATA_PATH)
        inv_info = pd.read_csv(INV_INFO_PATH, sep='\t')
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found: {e}. Exiting.")
        exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}. Exiting.")
        exit(1)

    # --- Prepare Data ---
    print("Preparing and cleaning data...")
    # Rename columns for clarity and consistency
    inv_info = inv_info.rename(columns={
        'Chromosome': 'chr', 'Start': 'region_start_info', 'End': 'region_end_info',
        '0_single_1_recur': 'RecurrenceCode',
        'Number_recurrent_events_.95..C.I..': 'NumRecurrentEventsStr'
    })

    # Check for required columns
    required_pi_cols = ['chr', 'region_start', 'region_end', '0_pi_filtered', '1_pi_filtered']
    required_info_cols = ['chr', 'region_start_info', 'region_end_info', 'RecurrenceCode', 'NumRecurrentEventsStr']
    if not all(col in pi_data.columns for col in required_pi_cols):
        print(f"ERROR: Missing required columns in pi data: {set(required_pi_cols) - set(pi_data.columns)}")
        exit(1)
    if not all(col in inv_info.columns for col in required_info_cols):
        print(f"ERROR: Missing required columns in inversion info: {set(required_info_cols) - set(inv_info.columns)}")
        exit(1)

    # Standardize chromosome format and ensure coordinate columns are numeric
    pi_data = standardize_chr(pi_data.copy(), 'chr')
    inv_info = standardize_chr(inv_info.copy(), 'chr')
    for df, cols in [(pi_data, ['region_start', 'region_end']), (inv_info, ['region_start_info', 'region_end_info'])]:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=cols, inplace=True) # Drop rows where coordinates couldn't be converted
        for col in cols:
             df[col] = df[col].astype(np.int64) # Convert valid coordinates to integer

    # Extract numeric recurrent event count and ensure RecurrenceCode is numeric
    inv_info['NumRecurrentEvents'] = inv_info['NumRecurrentEventsStr'].apply(extract_numeric_value)
    inv_info['RecurrenceCode'] = pd.to_numeric(inv_info['RecurrenceCode'], errors='coerce')

    # --- Merge Data ---
    print("Merging pi data with inversion info based on chromosome and coordinates...")
    # Use an identifier for pi_data rows before merge to handle potential duplicates after coordinate matching
    pi_data['pi_data_id'] = range(len(pi_data))
    merged_temp = pd.merge(
        pi_data,
        inv_info[['chr', 'region_start_info', 'region_end_info', 'RecurrenceCode', 'NumRecurrentEvents']],
        on='chr', how='inner' # Merge only matching chromosomes
    )

    # Filter merge results based on coordinate proximity (allow 1bp tolerance)
    coordinate_match_mask = (
        (abs(merged_temp['region_start'] - merged_temp['region_start_info']) <= 1) &
        (abs(merged_temp['region_end'] - merged_temp['region_end_info']) <= 1)
    )
    merged_filtered = merged_temp[coordinate_match_mask].copy()

    # Keep only unambiguous matches (where one pi region matches exactly one info region)
    merged_filtered['match_count'] = merged_filtered.groupby('pi_data_id')['pi_data_id'].transform('count')
    merged_data_wide = merged_filtered[merged_filtered['match_count'] == 1].copy()
    merged_data_wide = merged_data_wide.drop(columns=['region_start_info', 'region_end_info', 'pi_data_id', 'match_count'])

    if merged_data_wide.empty:
        print("ERROR: No matching regions found after coordinate filtering. Exiting.")
        exit(1)
    print(f"Found {len(merged_data_wide)} unique matching inversion regions.")

    # --- Filter for Recurrent Inversions ---
    print("Filtering for recurrent inversions (RecurrenceCode == 1)...")
    initial_count = len(merged_data_wide)
    merged_data_wide = merged_data_wide[merged_data_wide['RecurrenceCode'] == 1.0].copy()
    recurrent_count = len(merged_data_wide)
    print(f"Retained {recurrent_count} recurrent inversion regions (filtered from {initial_count}).")

    # Drop rows where the numeric recurrent event count could not be determined
    merged_data_wide['NumRecurrentEvents'] = pd.to_numeric(merged_data_wide['NumRecurrentEvents'], errors='coerce')
    merged_data_wide.dropna(subset=['NumRecurrentEvents'], inplace=True)
    recurrent_count_final = len(merged_data_wide)
    if recurrent_count_final < recurrent_count:
         print(f"Dropped {recurrent_count - recurrent_count_final} rows due to missing NumRecurrentEvents values.")

    if recurrent_count_final == 0:
        print("ERROR: No recurrent inversions with valid NumRecurrentEvents found. Cannot proceed. Exiting.")
        exit(1)

    merged_data_wide.to_csv(MERGED_DATA_WIDE_PATH, index=False)
    print(f"Filtered wide-format data saved to {MERGED_DATA_WIDE_PATH}")

    # --- Reshape to Long Format ---
    print("Reshaping data to long format for modeling...")
    # Create a unique ID for each inversion region to use as grouping factor in LMM
    merged_data_wide['InversionRegionID'] = merged_data_wide['chr'] + ':' + \
                                             merged_data_wide['region_start'].astype(str) + '-' + \
                                             merged_data_wide['region_end'].astype(str)

    # Melt the dataframe
    data_long = pd.melt(
        merged_data_wide,
        id_vars=['InversionRegionID', 'NumRecurrentEvents'],
        value_vars=['0_pi_filtered', '1_pi_filtered'],
        var_name='OrientationSource',
        value_name='PiValue'
    )
    data_long['Orientation'] = data_long['OrientationSource'].map({'0_pi_filtered': 'Direct', '1_pi_filtered': 'Inverted'})
    data_long = data_long.drop(columns=['OrientationSource'])

    # --- Final Data Cleaning for Modeling ---
    print("Final cleaning before modeling...")
    cols_for_model = ['PiValue', 'NumRecurrentEvents', 'Orientation', 'InversionRegionID']
    data_long = data_long[cols_for_model]
    initial_rows_long = len(data_long)
    data_long.dropna(subset=['PiValue', 'NumRecurrentEvents'], inplace=True) # Drop rows missing essential values
    final_rows_long = len(data_long)
    unique_regions = data_long['InversionRegionID'].nunique()
    print(f"Removed {initial_rows_long - final_rows_long} rows with missing PiValue or NumRecurrentEvents.")
    print(f"Proceeding with {final_rows_long} observations across {unique_regions} unique regions.")

    if final_rows_long < 10 or unique_regions < 3 : # Check for minimum data points for modeling
         print("ERROR: Insufficient data remaining after cleaning to fit reliable models. Exiting.")
         exit(1)

    # Create separate dataframes for each orientation model
    data_direct = data_long[data_long['Orientation'] == 'Direct'].copy()
    data_inverted = data_long[data_long['Orientation'] == 'Inverted'].copy()
    print(f"Direct orientation dataset size: {len(data_direct)}")
    print(f"Inverted orientation dataset size: {len(data_inverted)}")

    data_long.to_csv(LONG_DATA_PATH, index=False)
    print(f"Model-ready long-format data saved to {LONG_DATA_PATH}")

    # --- Fit Separate Linear Mixed-Effects Models ---
    print("\nFitting separate LMMs for each orientation...")
    # Model: PiValue depends on NumRecurrentEvents, accounting for baseline differences between regions
    model_formula = "PiValue ~ NumRecurrentEvents"
    results_dict = {}
    fit_successful = True

    for orientation, data_subset in [('Direct', data_direct), ('Inverted', data_inverted)]:
        print(f"\n--- Fitting Model for {orientation} Orientation ---")
        # Check if subset has enough data and groups for LMM
        if len(data_subset) < 5 or data_subset['InversionRegionID'].nunique() < 2:
             print(f"Skipping {orientation} model: Insufficient data (Obs={len(data_subset)}, Groups={data_subset['InversionRegionID'].nunique()})")
             results_dict[orientation] = None
             continue

        try:
            # Fit LMM with InversionRegionID as the grouping factor (random intercepts)
            mixed_model = smf.mixedlm(model_formula, data_subset,
                                      groups=data_subset["InversionRegionID"])
            result = mixed_model.fit(method=["lbfgs"]) # Try L-BFGS optimizer
            print(f"{orientation} model fitting successful.")
            results_dict[orientation] = result
            print(result.summary())
        except np.linalg.LinAlgError as e:
             print(f"ERROR: Linear algebra error during {orientation} model fitting: {e}")
             print("This might indicate issues like perfect multicollinearity or low variance.")
             print(f"Variance of NumRecurrentEvents in {orientation} data: {data_subset['NumRecurrentEvents'].var()}")
             results_dict[orientation] = None
             fit_successful = False
        except Exception as e:
            print(f"ERROR: Model fitting failed unexpectedly for {orientation}: {e}")
            results_dict[orientation] = None
            fit_successful = False

    # --- Save Model Summaries ---
    print("\n--- Saving Model Summaries ---")
    with open(MODEL_SUMMARY_PATH, 'w') as f:
        f.write("Separate Mixed Linear Model Regression Results: Pi ~ Number of Recurrent Events\n")
        f.write("==================================================================================\n")
        f.write(f"Data: {final_rows_long} observations from {unique_regions} unique recurrent regions.\n")
        f.write(f"Model Formula (for each orientation): {model_formula}\n")
        f.write(f"Grouping Variable (Random Intercept): InversionRegionID\n")
        f.write("==================================================================================\n\n")

        for orientation, result in results_dict.items():
            f.write(f"--- {orientation} Orientation Model Results ---\n")
            if result:
                try:
                    f.write(result.summary().as_text())
                except Exception as e:
                     f.write(f"Summary generation failed: {e}\n")
            else:
                f.write("Model fitting failed or was skipped due to insufficient data.\n")
            f.write("\n\n")
    print(f"Model summaries saved to {MODEL_SUMMARY_PATH}")

    if not fit_successful:
        print("\nSkipping visualization due to model fitting errors.")
        print("\n--- Analysis Complete (with errors) ---")
        print(f"Results saved in directory: {OUTPUT_DIR}")
        exit(1)

    # --- Visualization ---
    print("\nGenerating visualization...")

    # Use a visually appealing seaborn style
    sns.set_theme(style="white", palette="muted")
    plot_colors = {'Direct': sns.color_palette("colorblind")[0], 'Inverted': sns.color_palette("colorblind")[3]} # Blue and Orange/Red

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True) # 1 row, 2 cols, shared Y axis
    fig.suptitle('Nucleotide Diversity (π) vs. Number of Recurrent Events by Orientation', fontsize=20, y=1.03)

    max_pi_value = data_long['PiValue'].max()

    for i, (orientation, result) in enumerate(results_dict.items()):
        ax = axs[i]
        data_subset = data_direct if orientation == 'Direct' else data_inverted
        color = plot_colors[orientation]

        if result is None or data_subset.empty:
            ax.text(0.5, 0.5, f'No model or data\nfor {orientation}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14, color='grey')
            ax.set_title(f'{orientation} Orientation', fontsize=16)
            ax.set_xlabel('Number of Recurrent Events', fontsize=15)
            if i == 0: ax.set_ylabel('Nucleotide Diversity (π)', fontsize=15)
            continue

        # Scatter plot of the actual data points
        sns.scatterplot(
            data=data_subset,
            x='NumRecurrentEvents',
            y='PiValue',
            color=color,
            alpha=0.7,
            s=60,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
        )

        # Generate predictions for the fixed effects line (population average)
        x_pred = np.linspace(data_subset['NumRecurrentEvents'].min(),
                             data_subset['NumRecurrentEvents'].max(), 100)
        pred_df = pd.DataFrame({'NumRecurrentEvents': x_pred})
        # Need to add Intercept for exog prediction if model wasn't formula-based
        # Since we used formula, statsmodels handles the intercept automatically
        try:
            pred_df['Predicted_PiValue'] = result.predict(exog=pred_df)

            # Plot the LMM fixed effects regression line
            sns.lineplot(
                data=pred_df,
                x='NumRecurrentEvents',
                y='Predicted_PiValue',
                color=color,
                linewidth=3, # Thicker line
                alpha=0.9,
                ax=ax,
                label='LMM Fit' # Add label for potential legend
            )
        except Exception as e:
            print(f"Warning: Could not generate/plot prediction line for {orientation}: {e}")

        # Add model results annotation
        try:
            intercept = result.params['Intercept']
            slope = result.params['NumRecurrentEvents']
            pval_slope = result.pvalues['NumRecurrentEvents']
            # Use scientific notation for p-value if small, otherwise standard format
            pval_text = f"{pval_slope:.2g}" if pval_slope < 0.01 else f"{pval_slope:.3f}"
            annotation = (f"Intercept = {intercept:.3f}\n"
                          f"Slope = {slope:.2e}\n"
                          f"P-value (Slope) = {pval_text}")
            # Place annotation box in upper corner
            ax.text(0.95, 0.95, annotation, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8, ec='grey'))
        except KeyError as e:
             print(f"Warning: Could not retrieve coefficient/p-value for annotation ({orientation}): {e}")
        except Exception as e:
             print(f"Warning: Error during annotation for {orientation}: {e}")


        ax.set_title(f'{orientation} Orientation', fontsize=18, fontweight='bold')
        ax.set_xlabel('Number of Recurrent Events', fontsize=15)
        if i == 0:
            ax.set_ylabel('Nucleotide Diversity (π)', fontsize=15)
        else:
             ax.set_ylabel('') # Avoid repeating Y label

        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True)) # Prefer integers for event counts
        ax.tick_params(axis='both', which='major', labelsize=13)

        # ax.set_ylim(bottom=0, top=max_pi_value * 1.1)

    # Adjust layout to prevent overlap and make space for title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(SCATTER_PLOT_PATH, dpi=300, bbox_inches='tight') # Use bbox_inches for tight saving
    print(f"Scatter plot saved to {SCATTER_PLOT_PATH}")
    plt.close(fig) # Close the figure to free memory

    print("\n--- Analysis Complete ---")
    print(f"Results saved in directory: {OUTPUT_DIR}")

# --- Run Main Analysis ---
if __name__ == "__main__":
    # Suppress common warnings for cleaner output
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")

    main()
