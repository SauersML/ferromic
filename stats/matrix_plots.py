import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import warnings
import os
from pathlib import Path
import pickle
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# Create necessary directories
for directory in [CACHE_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)
    print(f"Directory {directory} exists: {directory.exists()}")

def read_significant_results(csv_path):
    """
    Read the significant results from CSV file.
    Returns DataFrame of significant results (corrected_p_value < 0.05).
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Read {len(df)} entries from {csv_path}")
        
        # Filter for significant results
        significant = df[df['corrected_p_value'] < 0.05].sort_values('p_value')
        print(f"Found {len(significant)} significant results (corrected_p_value < 0.05)")
        
        return significant
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

def load_pickle_data(pickle_path):
    """
    Load CDS results from pickle file.
    Returns dictionary of CDS results.
    """
    try:
        with open(pickle_path, 'rb') as f:
            cds_results = pickle.load(f)
        print(f"Loaded {len(cds_results)} entries from {pickle_path}")
        return cds_results
    except Exception as e:
        print(f"Error loading pickle file {pickle_path}: {e}")
        return {}  # Return empty dictionary on error

def create_visualization(transcript_id, coordinates, stats, pickle_data, output_path=None):
    """
    Create visualization for a transcript using data from both CSV and pickle sources.
    
    Parameters:
    -----------
    transcript_id : str
        Transcript identifier
    coordinates : str
        Genomic coordinates string
    stats : dict
        Statistics from CSV, including p-values, effect sizes, gene info
    pickle_data : dict
        Dictionary containing matrix data from pickle file
    output_path : Path, optional
        Path to save the visualization
    
    Returns:
    --------
    bool
        True if visualization was successfully created, False otherwise
    """
    # Create a suitable output path if not provided
    if output_path is None:
        if stats.get('gene_symbol'):
            base_name = f"{stats['gene_symbol']}_{transcript_id}"
        else:
            base_name = f"transcript_{transcript_id}"
        
        safe_name = base_name.replace('/', '_').replace(':', '_').replace('-', '_')
        output_path = PLOTS_DIR / f"{safe_name}.png"
    
    print(f"\nProcessing transcript: {transcript_id}")
    print(f"Coordinates: {coordinates}")
    print(f"Will save plot to: {output_path}")
    
    # Get gene information
    gene_symbol = stats.get('gene_symbol')
    gene_name = stats.get('gene_name')
    if gene_symbol:
        print(f"Gene: {gene_symbol}: {gene_name}")
    
    # Check if the coordinates exist in pickle data
    if coordinates not in pickle_data:
        print(f"ERROR: Coordinates '{coordinates}' not found in pickle data")
        return False
    
    # Get matrix data from pickle
    cds_data = pickle_data[coordinates]
    
    # Check if matrices are available
    matrix_0 = cds_data.get('matrix_0')
    matrix_1 = cds_data.get('matrix_1')
    
    if matrix_0 is None or matrix_1 is None:
        print(f"ERROR: Matrix data not available for {coordinates}")
        return False

    is_valid_0, counts_0 = validate_matrix(matrix_0, "Direct")
    is_valid_1, counts_1 = validate_matrix(matrix_1, "Inverted")

    # Print detailed validation information
    print("\nMatrix validation results:")
    for counts in [counts_0, counts_1]:
        if isinstance(counts, dict) and "matrix_name" in counts:
            name = counts["matrix_name"]
            print(f"  {name} matrix: {counts['normal_values']} normal values ({counts['pct_normal']:.1f}%)")
            print(f"    minus_one values: {counts['minus_one_values']} ({counts['pct_minus_one']:.1f}%)")
            print(f"    ninety_nine values: {counts['ninety_nine_values']} ({counts['pct_ninety_nine']:.1f}%)")
            print(f"    non-diagonal NaN: {counts['nan_values_non_diag']} ({counts['pct_nan']:.1f}%)")
    
    # Get sequence counts (estimated from matrix shapes)
    n0 = matrix_0.shape[0]
    n1 = matrix_1.shape[0]
    sequences_0 = [f"seq0_{i}" for i in range(n0)]
    sequences_1 = [f"seq1_{i}" for i in range(n1)]
    
    print(f"Matrix shapes: matrix_0={matrix_0.shape}, matrix_1={matrix_1.shape}")
    
    # Define colors for special values
    color_minus_one = (242/255, 235/255, 250/255)  # lavender for identical sequences (-1)
    color_ninety_nine = (1, 192/255, 192/255)      # light red for no non-syn variation (99)
    special_patches = [
        mpatches.Patch(color=color_minus_one, label='Identical sequences'),
        mpatches.Patch(color=color_ninety_nine, label='No non-synonymous variation')
    ]

    cmap_normal = sns.color_palette("viridis", as_cmap=True)

    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # GridSpec: 
    # Top row: 3 columns (matrix_0, matrix_1, colorbar)
    # Bottom row: 1 column spanning all three top columns for the histogram
    gs = plt.GridSpec(
        2, 3,
        width_ratios=[1, 1, 0.05],
        height_ratios=[1, 0.4],
        hspace=0.3, wspace=0.3
    )

    # Main Title
    if gene_symbol and gene_name:
        title_str = f'{gene_symbol}: {gene_name}\n{coordinates}'
    else:
        title_str = f'Transcript: {transcript_id}\n{coordinates}'
    
    fig.suptitle(title_str, fontsize=26, fontweight='bold', y=0.98)

    def plot_matrices(ax, matrix, title_str):
        """
        Plot the given matrix:
        - Upper triangle: normal omega values (log scale colormap)
        - Lower triangle: special values (-1 and 99) with distinct colors
        """
        n = matrix.shape[0]
        upper_triangle = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        lower_triangle = np.tril(np.ones_like(matrix, dtype=bool), k=-1)

        special_minus_one = (matrix == -1)
        special_ninety_nine = (matrix == 99)
        normal_mask = (~np.isnan(matrix)) & (~special_minus_one) & (~special_ninety_nine)

        # NORMAL VALUES (upper triangle)
        normal_data = matrix.copy()
        normal_data[~(normal_mask & upper_triangle)] = np.nan
        normal_data = np.where(normal_data < 0.01, 0.01, normal_data)
        normal_data = np.where(normal_data > 50, 50, normal_data)
        normal_data_inv = normal_data[::-1, :]

        sns.heatmap(
            normal_data_inv, cmap=cmap_normal, norm=LogNorm(vmin=0.01, vmax=50),
            ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False,
            mask=np.isnan(normal_data_inv)
        )

        # SPECIAL VALUES (lower triangle)
        special_cmap = ListedColormap([color_minus_one, color_ninety_nine])
        special_data = np.full_like(matrix, np.nan, dtype=float)
        special_data[special_minus_one] = 0
        special_data[special_ninety_nine] = 1
        special_data[~lower_triangle] = np.nan
        special_data_inv = special_data[::-1, :]

        sns.heatmap(
            special_data_inv, cmap=special_cmap, ax=ax, cbar=False, square=True,
            xticklabels=False, yticklabels=False, mask=np.isnan(special_data_inv)
        )

        ax.set_title(title_str, fontsize=20, pad=15)
        ax.tick_params(axis='both', which='both', length=0)

    # Plot Direct matrix (top row, first column)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_matrices(ax1, matrix_0, f'Direct Sequences (n={n0})')

    # Plot Inverted matrix (top row, second column)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_matrices(ax2, matrix_1, f'Inverted Sequences (n={n1})')

    # Add the colorbar in the top row, third column
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # Create a ScalarMappable with a logarithmic scale color normalization
    sm = ScalarMappable(norm=LogNorm(vmin=0.01, vmax=50), cmap=cmap_normal)
    sm.set_array([])
    
    # Generate the colorbar without predefined ticks and labels
    cbar = plt.colorbar(sm, cax=cbar_ax)
    
    # Specify the exact ticks and corresponding labels
    desired_ticks = [0.01, 1, 3, 10, 50]
    desired_labels = ['0', '1', '3', '10', '50']

    # Apply the fixed ticks and labels to the colorbar
    cbar.ax.yaxis.set_major_locator(FixedLocator(desired_ticks))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(desired_labels))
    
    # Set the colorbar label and tick label size
    cbar.set_label('Omega Value', fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    legend = ax1.legend(
        handles=special_patches,
        title='Special Values',
        loc='upper left',
        bbox_to_anchor=(0.0, -0.05),
        ncol=1,
        frameon=True,
        fontsize=13,
    )
    legend.get_title().set_fontsize(13)

    # Extract normal omega values for distribution plot
    values_direct = matrix_0[np.tril_indices_from(matrix_0, k=-1)]
    values_inverted = matrix_1[np.tril_indices_from(matrix_1, k=-1)]
    values_direct = values_direct[~np.isnan(values_direct)]
    values_direct = values_direct[(values_direct != -1) & (values_direct != 99)]
    values_inverted = values_inverted[~np.isnan(values_inverted)]
    values_inverted = values_inverted[(values_inverted != -1) & (values_inverted != 99)]

    # Histogram (actually KDE plot) on the second row, spanning all columns
    ax3 = fig.add_subplot(gs[1, :])
    
    try:
        if len(values_direct) > 0:
            sns.kdeplot(values_direct, ax=ax3, label='Direct', fill=True, common_norm=False,
                        color='#1f77b4', alpha=0.6)
        else:
            print("Warning: No valid direct values for KDE plot")
            
        if len(values_inverted) > 0:
            sns.kdeplot(values_inverted, ax=ax3, label='Inverted', fill=True, common_norm=False,
                        color='#ff7f0e', alpha=0.6)
        else:
            print("Warning: No valid inverted values for KDE plot")
    except Exception as e:
        print(f"Error creating KDE plot: {e}")
        # Fall back to histograms if KDE fails
        if len(values_direct) > 0:
            ax3.hist(values_direct, alpha=0.6, label='Direct', color='#1f77b4', bins=20)
        if len(values_inverted) > 0:
            ax3.hist(values_inverted, alpha=0.6, label='Inverted', color='#ff7f0e', bins=20)
    
    ax3.set_title('Distribution of Omega Values', fontsize=20, pad=15)
    ax3.set_xlabel('Omega Value', fontsize=16)
    ax3.set_ylabel('Density', fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.legend(title='Groups', title_fontsize=16, fontsize=14)

    # Add P-value text inside the histogram subplot (top-right corner)
    p_value = stats.get('p_value', np.nan)
    bonf_p_value = stats.get('corrected_p_value', np.nan)
    effect_size = stats.get('effect_size', np.nan)
    
    # Determine effect direction
    effect_dir = "Higher in Inverted" if effect_size > 0 else "Higher in Direct"
    effect_str = f"Effect size: {effect_size:.4f} ({effect_dir})" if not np.isnan(effect_size) else "Effect size: N/A"
    bonf_p_value_str = f"Corrected p-value: {bonf_p_value:.4e}" if not np.isnan(bonf_p_value) else "P-value: N/A"
    
    ax3.text(
        0.97, 0.97, 
        f"{bonf_p_value_str}\n{effect_str}",
        transform=ax3.transAxes,
        ha='right', va='top', fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Adjust layout to reduce clutter
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure - with error handling
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Successfully saved figure to {output_path}")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error saving figure to {output_path}: {e}")
        plt.close(fig)
        return False


def validate_matrix(matrix, matrix_name):
    """
    Validate a matrix to ensure it's not dominated by special values.
    Returns a tuple (is_valid, detailed_counts) where:
    - is_valid: Boolean indicating if the matrix has sufficient normal values
    - detailed_counts: Dictionary with counts of different value types
    """
    if matrix is None:
        return False, {"error": "Matrix is None"}
    
    total_cells = matrix.size
    diag_cells = matrix.shape[0]  # Number of diagonal cells (usually NaN)
    non_diag_cells = total_cells - diag_cells
    
    # Count different types of values
    nan_values = np.isnan(matrix).sum()
    minus_one_values = np.sum(matrix == -1)
    ninety_nine_values = np.sum(matrix == 99)
    normal_values = total_cells - nan_values - minus_one_values - ninety_nine_values
    
    # Calculate percentages (of non-diagonal cells)
    pct_normal = normal_values / non_diag_cells * 100 if non_diag_cells > 0 else 0
    pct_minus_one = minus_one_values / non_diag_cells * 100 if non_diag_cells > 0 else 0
    pct_ninety_nine = ninety_nine_values / non_diag_cells * 100 if non_diag_cells > 0 else 0
    pct_nan = (nan_values - diag_cells) / non_diag_cells * 100 if non_diag_cells > 0 else 0
    
    counts = {
        "matrix_name": matrix_name,
        "total_cells": total_cells,
        "diagonal_cells": diag_cells,
        "non_diagonal_cells": non_diag_cells,
        "normal_values": normal_values,
        "pct_normal": pct_normal,
        "minus_one_values": minus_one_values,
        "pct_minus_one": pct_minus_one,
        "ninety_nine_values": ninety_nine_values,
        "pct_ninety_nine": pct_ninety_nine,
        "nan_values_non_diag": nan_values - diag_cells,
        "pct_nan": pct_nan
    }
    
    # Consider a matrix invalid if >95% of non-diagonal cells are special values or NaN
    is_valid = pct_normal >= 5
    
    return is_valid, counts


def main():
    """Main function to process data and create visualizations."""
    start_time = datetime.now()
    print(f"Matrix visualization started at {start_time}")
    
    # Read significant results from CSV
    csv_path = RESULTS_DIR / "significant_by_effect.csv"
    significant_df = read_significant_results(csv_path)
    
    if significant_df.empty:
        print("No significant results found. Exiting.")
        return
    
    # Load CDS results from pickle
    pickle_path = CACHE_DIR / "all_cds_results.pkl"
    cds_results = load_pickle_data(pickle_path)
    
    if not cds_results:
        print("No CDS results found in pickle file. Exiting.")
        return
    
    # Process each significant result
    successful_plots = 0
    failed_plots = 0
    
    for index, row in significant_df.iterrows():
        # Get transcript and coordinates
        transcript_id = row['transcript_id']
        coordinates = row['coordinates']
        
        # Create output path
        gene_symbol = row.get('gene_symbol')
        if gene_symbol:
            output_path = PLOTS_DIR / f"{gene_symbol}_{transcript_id.replace('/', '_').replace('.', '_')}.png"
        else:
            output_path = PLOTS_DIR / f"transcript_{transcript_id.replace('/', '_').replace('.', '_')}.png"
        
        # Create visualization
        success = create_visualization(
            transcript_id=transcript_id,
            coordinates=coordinates,
            stats=row.to_dict(),
            pickle_data=cds_results,
            output_path=output_path
        )
        
        if success:
            successful_plots += 1
            print(f"Visualization created for transcript {transcript_id}")
        else:
            failed_plots += 1
            print(f"Failed to create visualization for transcript {transcript_id}")
    
    # Print summary
    total_plots = successful_plots + failed_plots
    print(f"\nVisualization Summary:")
    print(f"Total significant results: {total_plots}")
    print(f"Successful visualizations: {successful_plots}")
    print(f"Failed visualizations: {failed_plots}")
    print(f"Success rate: {successful_plots/total_plots*100:.1f}%")
    print(f"Plots are saved in the '{PLOTS_DIR}' directory")
    
    end_time = datetime.now()
    print(f"\nVisualization completed at {end_time}")
    print(f"Total runtime: {end_time - start_time}")

if __name__ == "__main__":
    main()
