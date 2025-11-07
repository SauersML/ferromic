#!/usr/bin/env python3
"""
Chr17 Inversion vs Tag SNP Correlation Plot

Correlates effect sizes (OR) between chr17 inversion PheWAS results
and tag SNP PheWAS results for significant phenotypes (Q_GLOBAL < 0.05).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Set font to Helvetica with Arial fallback
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'

# ---------- Config ----------
PHEWAS_FILE = "data/phewas_results.tsv"
TAG_FILE = "data/all_pop_phewas_tag.tsv"
OUTDIR = "phewas_plots"
OUTPUT_BASE = "chr17_inversion_tag_correlation"

# Chr17 inversion identifier
CHR17_INVERSION = "chr17-45585160-INV-706887"

# Significance threshold
Q_THRESHOLD = 0.05

# Plot styling
FIGSIZE = (8, 8)
DPI = 300
POINT_SIZE = 60
POINT_COLOR = '#2E86AB'
POINT_ALPHA = 0.7
LINE_COLOR = '#E63946'
LINE_WIDTH = 2
LINE_STYLE = '--'


def load_and_merge_data():
    """
    Load PheWAS results and tag SNP data, filter for significant chr17 results,
    and merge on Phenotype.
    """
    # Load datasets
    print(f"Loading {PHEWAS_FILE}...")
    phewas = pd.read_csv(PHEWAS_FILE, sep='\t')

    print(f"Loading {TAG_FILE}...")
    tag = pd.read_csv(TAG_FILE, sep='\t')

    # Filter for chr17 inversion with Q_GLOBAL < threshold
    print(f"\nFiltering chr17 inversion ({CHR17_INVERSION}) with Q_GLOBAL < {Q_THRESHOLD}...")
    chr17_sig = phewas[
        (phewas['Inversion'] == CHR17_INVERSION) &
        (phewas['Q_GLOBAL'] < Q_THRESHOLD)
    ].copy()

    print(f"Found {len(chr17_sig)} significant phenotypes for chr17 inversion")

    # Merge on Phenotype
    print(f"\nMerging with tag SNP data...")
    merged = chr17_sig.merge(tag, on='Phenotype', suffixes=('_inv', '_tag'))

    print(f"Successfully merged {len(merged)} phenotypes")

    if len(merged) == 0:
        raise ValueError("No overlapping phenotypes found between datasets")

    return merged


def load_and_merge_data_all():
    """
    Load PheWAS results and tag SNP data for ALL chr17 phenotypes,
    and merge on Phenotype.
    """
    # Load datasets
    print(f"Loading {PHEWAS_FILE}...")
    phewas = pd.read_csv(PHEWAS_FILE, sep='\t')

    print(f"Loading {TAG_FILE}...")
    tag = pd.read_csv(TAG_FILE, sep='\t')

    # Filter for chr17 inversion (all phenotypes, not just significant)
    print(f"\nFiltering chr17 inversion ({CHR17_INVERSION}) - ALL phenotypes...")
    chr17_all = phewas[phewas['Inversion'] == CHR17_INVERSION].copy()

    print(f"Found {len(chr17_all)} total phenotypes for chr17 inversion")

    # Merge on Phenotype
    print(f"\nMerging with tag SNP data...")
    merged = chr17_all.merge(tag, on='Phenotype', suffixes=('_inv', '_tag'))

    print(f"Successfully merged {len(merged)} phenotypes")

    if len(merged) == 0:
        raise ValueError("No overlapping phenotypes found between datasets")

    return merged


def create_correlation_plot(data, x_col, y_col, output_path):
    """
    Create a correlation scatter plot with regression line and statistics.

    Parameters:
    -----------
    data : DataFrame
        Merged data with both inversion and tag results
    x_col : str
        Column name for x-axis (tag SNP)
    y_col : str
        Column name for y-axis (inversion)
    output_path : str
        Base path for output files (without extension)
    """
    # Extract values
    x = data[x_col].values
    y = data[y_col].values

    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError(f"Insufficient data points for correlation (n={len(x)})")

    # Calculate correlation
    r, p_value = stats.pearsonr(x, y)

    # Calculate regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Scatter plot
    ax.scatter(x, y, s=POINT_SIZE, alpha=POINT_ALPHA,
              color=POINT_COLOR, edgecolors='white', linewidth=0.5)

    # Labels and title
    ax.set_xlabel('Tag SNP Odds Ratio', fontsize=12)
    ax.set_ylabel('Inversion Odds Ratio', fontsize=12)
    ax.set_title('Chr17 Inversion vs Tag SNP Effect Sizes', fontsize=14, pad=20)

    # Add statistics text box
    stats_text = f'r = {r:.3f}\np = {p_value:.2e}\nn = {len(x)}'
    ax.text(0.05, 0.95, stats_text,
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Force equal axis limits based on combined data range
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    padding = 0.02
    data_range = max_val - min_val
    axis_min = min_val - padding * data_range
    axis_max = max_val + padding * data_range

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    # Force square aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save outputs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    png_path = f"{output_path}.png"
    pdf_path = f"{output_path}.pdf"

    plt.savefig(png_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved plot to {png_path}")

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {pdf_path}")

    plt.close()

    # Print statistics
    print(f"\nCorrelation Statistics:")
    print(f"  Pearson r: {r:.4f}")
    print(f"  P-value: {p_value:.4e}")
    print(f"  N: {len(x)}")
    print(f"  Slope: {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")


def create_pvalue_correlation_plot(data, x_pval_col, y_pval_col, output_path):
    """
    Create a correlation scatter plot for -log10(p-values) with regression line and statistics.

    Parameters:
    -----------
    data : DataFrame
        Merged data with both inversion and tag results
    x_pval_col : str
        Column name for x-axis p-values (tag SNP)
    y_pval_col : str
        Column name for y-axis p-values (inversion)
    output_path : str
        Base path for output files (without extension)
    """
    # Extract p-values
    x_pval = data[x_pval_col].values
    y_pval = data[y_pval_col].values

    # Remove any NaN or non-positive values
    mask = ~(np.isnan(x_pval) | np.isnan(y_pval) | (x_pval <= 0) | (y_pval <= 0))
    x_pval = x_pval[mask]
    y_pval = y_pval[mask]

    if len(x_pval) < 3:
        raise ValueError(f"Insufficient data points for correlation (n={len(x_pval)})")

    # Transform to -log10
    x = -np.log10(x_pval)
    y = -np.log10(y_pval)

    # Calculate correlation
    r, p_value = stats.pearsonr(x, y)

    # Print actual p-value for confirmation
    print(f"  Raw correlation p-value: {p_value}")

    # Calculate regression line
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Scatter plot
    ax.scatter(x, y, s=POINT_SIZE, alpha=POINT_ALPHA,
              color=POINT_COLOR, edgecolors='white', linewidth=0.5)

    # Labels and title
    ax.set_xlabel('Tag SNP -log₁₀(P)', fontsize=12)
    ax.set_ylabel('Inversion -log₁₀(P)', fontsize=12)
    ax.set_title('Chr17 Inversion vs Tag SNP P-values', fontsize=14, pad=20)

    # Format p-value for display (handle very small values)
    if p_value < 1e-20:
        p_text = 'p < 1e-20'
    else:
        p_text = f'p = {p_value:.2e}'

    # Add statistics text box
    stats_text = f'r = {r:.3f}\n{p_text}\nn = {len(x)}'
    ax.text(0.05, 0.95, stats_text,
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Force equal axis limits based on combined data range
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    padding = 0.02
    data_range = max_val - min_val
    axis_min = min_val - padding * data_range
    axis_max = max_val + padding * data_range

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    # Force square aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save outputs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    png_path = f"{output_path}.png"
    pdf_path = f"{output_path}.pdf"

    plt.savefig(png_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved plot to {png_path}")

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {pdf_path}")

    plt.close()

    # Print statistics
    print(f"\nCorrelation Statistics:")
    print(f"  Pearson r: {r:.4f}")
    print(f"  P-value: {p_value:.4e}")
    print(f"  N: {len(x)}")
    print(f"  Slope: {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")


def main():
    """Main function to generate correlation plot."""
    print("=" * 60)
    print("Chr17 Inversion vs Tag SNP Correlation Analysis")
    print("=" * 60)

    # Load and merge data for significant phenotypes
    merged_data = load_and_merge_data()

    # Create correlation plot for OR values
    print(f"\nGenerating OR correlation plot (significant phenotypes)...")
    output_path = os.path.join(OUTDIR, OUTPUT_BASE)

    create_correlation_plot(
        merged_data,
        x_col='OR_tag',
        y_col='OR_inv',
        output_path=output_path
    )

    # Load and merge data for ALL phenotypes
    print("\n" + "=" * 60)
    merged_data_all = load_and_merge_data_all()

    # Create correlation plot for p-values (all phenotypes)
    print(f"\nGenerating p-value correlation plot (all phenotypes)...")
    output_path_pval = os.path.join(OUTDIR, f"{OUTPUT_BASE}_pvalue")

    create_pvalue_correlation_plot(
        merged_data_all,
        x_pval_col='P_LRT_Overall_tag',
        y_pval_col='P_LRT_Overall_inv',
        output_path=output_path_pval
    )

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
