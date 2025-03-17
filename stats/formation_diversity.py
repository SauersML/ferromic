import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.ticker as mticker

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in.*")

def extract_numeric_value(s):
    """Extract the main numeric value from formatted strings.
    
    Examples:
    - "1.02e-04 [2.72e-05 ,1.21e-04]" -> 1.02e-04
    - "13 [7.00 ,13.75]" -> 13
    """
    if pd.isna(s) or s == "NA":
        return np.nan
    
    # Try to extract the main number before the square bracket
    match = re.search(r'([0-9.]+(?:e[+-]?[0-9]+)?)', str(s))
    if match:
        value_str = match.group(1)
        try:
            return float(value_str)
        except:
            return np.nan
    return np.nan

def check_invalid_values(df, columns_to_check):
    """Check for suspicious values that are not None, NaN, empty, or inf."""
    suspicious_values = {}
    
    for col in columns_to_check:
        if col not in df.columns:
            continue
            
        # Get only values that exist but might be problematic
        mask = ~df[col].isna() & ~np.isinf(df[col]) & (df[col] != '')
        values = df.loc[mask, col]
        
        # For numeric columns, all values should be convertible to float
        try:
            values.astype(float)
        except (ValueError, TypeError):
            # Found some suspicious values
            suspicious = [v for v in values if not isinstance(v, (int, float)) or (isinstance(v, str) and not v.strip().replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).replace('+', '', 1).isdigit())]
            if suspicious:
                suspicious_values[col] = suspicious
    
    return suspicious_values

def correlation_analysis(x, y, x_name, y_name):
    """Perform Spearman correlation analysis between two variables."""
    # Drop NaN values (ensure both x and y have no NaNs)
    valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[valid]
    y_clean = y[valid]
    
    if len(x_clean) < 3:  # Need at least 3 points for meaningful statistics
        print(f"WARNING: Insufficient data for {x_name} vs {y_name} correlation (only {len(x_clean)} valid points)")
        return {
            'spearman_corr': np.nan,
            'spearman_p': np.nan,
            'x_clean': x_clean,
            'y_clean': y_clean,
            'valid_data': False,
            'n': len(x_clean),
            # Keep these for regression line plotting
            'slope': np.nan,
            'intercept': np.nan
        }
    
    # Spearman correlation (non-parametric, rank-based)
    spearman_corr, spearman_p = stats.spearmanr(x_clean, y_clean)
    
    # Linear regression (still needed for trend line)
    model = LinearRegression()
    x_reshape = x_clean.values.reshape(-1, 1)
    model.fit(x_reshape, y_clean)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return {
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'x_clean': x_clean,
        'y_clean': y_clean,
        'valid_data': True,
        'n': len(x_clean),
        'slope': slope,
        'intercept': intercept
    }

def format_p_value(p):
    """Format p-values as raw decimal values."""
    if pd.isna(p):
        return "N/A"
    else:
        return f"{p:.6f}"

def create_scatterplot(ax, x, y, color, title, xlabel, ylabel, results):
    """Create a single scatterplot with correlation information."""
    if not results['valid_data']:
        ax.text(0.5, 0.5, 'Insufficient data for analysis', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return
    
    # Plot scatter with regression
    sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
    
    # Add regression line
    x_range = np.linspace(min(x), max(x), 100)
    y_pred = results['slope'] * x_range + results['intercept']
    ax.plot(x_range, y_pred, 'r-', linewidth=2)
    
    # Only add Spearman correlation with more decimal places
    spearman_text = f"Spearman ρ = {results['spearman_corr']:.6f}"
    p_text = f"p = {results['spearman_p']:.9f}"
    n_text = f"n = {results['n']}"
    
    # Place text in upper right corner with a semi-transparent background
    textstr = f"{spearman_text}\n{p_text}\n{n_text}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Customize axes
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set scientific notation for axes if needed
    if max(x) < 0.001 or max(x) > 10000:
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    if max(y) < 0.001 or max(y) > 10000:
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

def main():
    print("Loading data files...")
    # Load the data
    try:
        output_df = pd.read_csv('output.csv')
        inv_info_df = pd.read_csv('inv_info.csv')
    except Exception as e:
        print(f"Error loading data files: {e}")
        return
    
    print(f"Loaded {len(output_df)} rows from output.csv")
    print(f"Loaded {len(inv_info_df)} rows from inv_info.csv")
    
    # Format chromosome keys for matching
    output_df['chr_key'] = output_df['chr'].astype(str).str.replace('chr', '')
    inv_info_df['chr_key'] = inv_info_df['Chromosome'].astype(str).str.replace('chr', '')
    
    # Extract numeric values from formation rate and number of recurrent events
    print("Extracting numeric values from formatted strings...")
    inv_info_df['Formation_rate'] = inv_info_df['Formation_rate_per_generation_.95..C.I..'].apply(extract_numeric_value)
    inv_info_df['Num_recurrent_events'] = inv_info_df['Number_recurrent_events_.95..C.I..'].apply(extract_numeric_value)
    
    # Check for suspicious values
    print("Checking for invalid values...")
    suspicious_values = check_invalid_values(inv_info_df, ['Formation_rate', 'Num_recurrent_events'])
    if suspicious_values:
        print("Found suspicious values:")
        for col, values in suspicious_values.items():
            print(f"  {col}: {values}")
    
    # Perform exact matching with a tolerance of exactly 1 for start and end coordinates
    print("Performing exact matching with tolerance of 1 for coordinates...")
    merged_rows = []
    
    for _, output_row in output_df.iterrows():
        chr_key = output_row['chr_key']
        start = output_row['region_start']
        end = output_row['region_end']
        
        # Filter inv_info_df for matching chromosome
        chr_matches = inv_info_df[inv_info_df['chr_key'] == chr_key]
        
        for _, inv_row in chr_matches.iterrows():
            inv_start = inv_row['Start']
            inv_end = inv_row['End']
            
            # Check if coordinates match exactly or are off by exactly 1
            start_diff = abs(start - inv_start)
            end_diff = abs(end - inv_end)
            
            if (start_diff <= 1) and (end_diff <= 1):
                # Create a dictionary with combined values
                combined_row = {
                    'chr': output_row['chr'],
                    'region_start': output_row['region_start'],
                    'region_end': output_row['region_end'],
                    '0_pi_filtered': output_row['0_pi_filtered'],
                    '1_pi_filtered': output_row['1_pi_filtered'],
                    'Formation_rate': inv_row['Formation_rate'],
                    'Num_recurrent_events': inv_row['Num_recurrent_events'],
                    'Chromosome': inv_row['Chromosome'],
                    'Start': inv_row['Start'],
                    'End': inv_row['End']
                }
                merged_rows.append(combined_row)
    
    # Create merged dataframe
    if not merged_rows:
        print("ERROR: No matching regions found between the files!")
        return
    
    merged_df = pd.DataFrame(merged_rows)
    print(f"Found {len(merged_df)} matching regions (with tolerance of 1 for coordinates).")
    
    # Print the matched regions
    print("\nMatched regions:")
    for _, row in merged_df.iterrows():
        print(f"  {row['chr']}:{row['region_start']}-{row['region_end']} matched with {row['Chromosome']}:{row['Start']}-{row['End']}")
    
    # Perform correlation analysis for all four scenarios
    print("\nPerforming correlation analysis...")
    results = {
        'form_pi_inv': correlation_analysis(
            merged_df['Formation_rate'], 
            merged_df['0_pi_filtered'], 
            'Formation Rate', 
            'Pi (Inverted)'
        ),
        'form_pi_dir': correlation_analysis(
            merged_df['Formation_rate'], 
            merged_df['1_pi_filtered'], 
            'Formation Rate', 
            'Pi (Direct)'
        ),
        'recur_pi_inv': correlation_analysis(
            merged_df['Num_recurrent_events'], 
            merged_df['0_pi_filtered'], 
            'Number of Recurrent Events', 
            'Pi (Inverted)'
        ),
        'recur_pi_dir': correlation_analysis(
            merged_df['Num_recurrent_events'], 
            merged_df['1_pi_filtered'], 
            'Number of Recurrent Events', 
            'Pi (Direct)'
        ),
        'recur_form_rate': correlation_analysis(
            merged_df['Num_recurrent_events'], 
            merged_df['Formation_rate'], 
            'Number of Recurrent Events', 
            'Formation Rate'
        )
    }
    
    # Create a meta-plot with four scatterplots
    print("Creating visualization...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # Create the four plots
    create_scatterplot(
        axs[0, 0],
        results['form_pi_inv']['x_clean'], 
        results['form_pi_inv']['y_clean'],
        'blue',
        'Formation Rate vs Pi (Inverted)',
        'Formation Rate (per generation)',
        'Pi (Inverted)',
        results['form_pi_inv']
    )
    
    create_scatterplot(
        axs[0, 1],
        results['form_pi_dir']['x_clean'], 
        results['form_pi_dir']['y_clean'],
        'green',
        'Formation Rate vs Pi (Direct)',
        'Formation Rate (per generation)',
        'Pi (Direct)',
        results['form_pi_dir']
    )
    
    create_scatterplot(
        axs[1, 0],
        results['recur_pi_inv']['x_clean'], 
        results['recur_pi_inv']['y_clean'],
        'purple',
        'Number of Recurrent Events vs Pi (Inverted)',
        'Number of Recurrent Events',
        'Pi (Inverted)',
        results['recur_pi_inv']
    )
    
    create_scatterplot(
        axs[1, 1],
        results['recur_pi_dir']['x_clean'], 
        results['recur_pi_dir']['y_clean'],
        'orange',
        'Number of Recurrent Events vs Pi (Direct)',
        'Number of Recurrent Events',
        'Pi (Direct)',
        results['recur_pi_dir']
    )
    
    # Add a main title to the figure
    fig.suptitle('Correlation Analysis: Formation Rate and Recurrent Events vs Pi', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print("Saving figure to correlation_analysis.png...")
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create a separate plot for Formation Rate vs Number of Recurrent Events
    print("Creating Formation Rate vs Number of Recurrent Events plot...")
    fig_recur, ax_recur = plt.subplots(figsize=(10, 8), dpi=150)
    
    create_scatterplot(
        ax_recur,
        results['recur_form_rate']['x_clean'], 
        results['recur_form_rate']['y_clean'],
        'red',
        'Number of Recurrent Events vs Formation Rate',
        'Number of Recurrent Events',
        'Formation Rate (per generation)',
        results['recur_form_rate']
    )
    
    plt.tight_layout()
    
    print("Saving figure to recurrent_vs_formation.png...")
    plt.savefig('recurrent_vs_formation.png', dpi=300, bbox_inches='tight')
    
    # Print detailed summary of correlation results
    print("\n===== CORRELATION ANALYSIS SUMMARY =====")
    
    for analysis_name, analysis_results in results.items():
        if analysis_name == 'form_pi_inv':
            print("\nFormation Rate vs Pi (Inverted):")
        elif analysis_name == 'form_pi_dir':
            print("\nFormation Rate vs Pi (Direct):")
        elif analysis_name == 'recur_pi_inv':
            print("\nNumber of Recurrent Events vs Pi (Inverted):")
        elif analysis_name == 'recur_pi_dir':
            print("\nNumber of Recurrent Events vs Pi (Direct):")
        elif analysis_name == 'recur_form_rate':
            print("\nNumber of Recurrent Events vs Formation Rate:")
  
        if analysis_results['valid_data']:
            print(f"  Spearman correlation: ρ = {analysis_results['spearman_corr']:.6f}, p-value = {analysis_results['spearman_p']:.9f}")
            print(f"  Number of valid data points: {analysis_results['n']}")
        else:
            print("  Insufficient data for statistical analysis.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
