#!/usr/bin/env python3
"""
QQ Plot for PheWAS Results

Generates quantile-quantile plots to assess the distribution of p-values
from PheWAS analysis against the expected uniform distribution under the null hypothesis.
"""

import os
import sys
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
INFILE = "data/phewas_results.tsv"
OUTDIR = "phewas_plots"
P_COL = "P_Value_x"  # Primary p-value column
INV_COL = "Inversion"

# Plot styling
FIGSIZE = (8, 8)
DPI = 300
POINT_SIZE = 20
POINT_ALPHA = 0.6
LINE_COLOR = 'red'
LINE_WIDTH = 2
LINE_STYLE = '--'

# Significance thresholds
BONFERRONI_ALPHA = 0.05
SUGGESTIVE_ALPHA = 1e-5


def load_phewas_data(filepath):
    """Load PheWAS results from TSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    df = pd.read_csv(filepath, sep='\t')
    print(f"Loaded {len(df)} PheWAS results from {filepath}")
    return df


def calculate_lambda_gc(pvalues):
    """
    Calculate genomic inflation factor (lambda_GC).
    
    Lambda is the ratio of the median observed chi-squared statistic
    to the expected median under the null hypothesis (0.456).
    """
    pvalues = np.array(pvalues)
    pvalues = pvalues[~np.isnan(pvalues)]
    pvalues = pvalues[(pvalues > 0) & (pvalues <= 1)]
    
    if len(pvalues) == 0:
        return np.nan
    
    chisq = stats.chi2.ppf(1 - pvalues, df=1)
    lambda_gc = np.median(chisq) / stats.chi2.ppf(0.5, df=1)
    return lambda_gc


def sanitize_filename(name):
    """Sanitize string for use in filename."""
    import re
    # Replace any non-alphanumeric, non-dash, non-underscore, non-dot with underscore
    return re.sub(r'[^A-Za-z0-9._-]', '_', str(name))


def qq_plot(pvalues, title="QQ Plot", output_path=None, lambda_gc=None):
    """
    Generate a QQ plot for p-values.
    
    Parameters:
    -----------
    pvalues : array-like
        Array of p-values to plot
    title : str
        Plot title
    output_path : str
        Path to save the plot
    lambda_gc : float
        Genomic inflation factor (calculated if not provided)
    """
    # Remove NaN and invalid p-values
    pvalues = np.array(pvalues, dtype=float)
    pvalues = pvalues[~np.isnan(pvalues)]
    pvalues = pvalues[(pvalues > 0) & (pvalues <= 1)]
    
    if len(pvalues) == 0:
        print(f"Warning: No valid p-values for {title}")
        return
    
    # Check for degenerate case (all p-values identical)
    if np.std(pvalues) == 0:
        print(f"Warning: All p-values identical for {title}, skipping QQ plot")
        return
    
    # Sort p-values
    pvalues_sorted = np.sort(pvalues)
    n = len(pvalues_sorted)
    
    # Expected p-values under null hypothesis
    expected = np.arange(1, n + 1) / (n + 1)
    
    # Convert to -log10 scale
    observed_log = -np.log10(pvalues_sorted)
    expected_log = -np.log10(expected)
    
    # Calculate lambda if not provided
    if lambda_gc is None:
        lambda_gc = calculate_lambda_gc(pvalues)
    
    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Plot points
    ax.scatter(expected_log, observed_log, 
              s=POINT_SIZE, alpha=POINT_ALPHA, 
              edgecolors='none', c='#2E86AB')
    
    # Plot diagonal line (expected under null)
    # Use expected max for diagonal line (actual data range)
    max_expected = expected_log.max()
    max_observed = observed_log.max()
    diagonal_max = max(max_expected, max_observed)
    ax.plot([0, diagonal_max], [0, diagonal_max], 
           color=LINE_COLOR, linewidth=LINE_WIDTH, 
           linestyle=LINE_STYLE, label='Expected')
    
    # Add confidence interval (95%)
    # Using beta distribution for confidence bands
    # Note: beta.ppf can produce 0 or 1 at extremes, leading to inf in -log10
    # Clamp to avoid numerical issues
    alpha = 0.05
    ranks = np.arange(1, n+1)
    
    # Compute beta quantiles with numerical safeguards
    p_upper = stats.beta.ppf(1 - alpha/2, ranks, n - ranks + 1)
    p_lower = stats.beta.ppf(alpha/2, ranks, n - ranks + 1)
    
    # Clamp to avoid log10(0) = -inf
    p_upper = np.clip(p_upper, 1e-300, 1 - 1e-16)
    p_lower = np.clip(p_lower, 1e-300, 1 - 1e-16)
    
    # Note: variable naming is inverted due to -log10 transform
    # p_lower (smaller p) -> larger -log10(p) -> upper curve
    # p_upper (larger p) -> smaller -log10(p) -> lower curve
    upper_ci = -np.log10(p_lower)  # Upper curve (more extreme)
    lower_ci = -np.log10(p_upper)  # Lower curve (less extreme)
    
    # Clip extreme CI values for visual sanity
    max_ci = max_observed * 1.5
    upper_ci = np.clip(upper_ci, 0, max_ci)
    lower_ci = np.clip(lower_ci, 0, max_ci)
    
    ax.fill_between(expected_log, lower_ci, upper_ci, 
                    alpha=0.2, color='gray', label='95% CI')
    
    # Labels and title
    ax.set_xlabel('Expected -log₁₀(p)', fontsize=12)
    ax.set_ylabel('Observed -log₁₀(p)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add lambda value to plot
    lambda_text = f'λ = {lambda_gc:.3f}'
    ax.text(0.05, 0.95, lambda_text, 
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set axis limits based on data range with small padding
    # Guard against degenerate ranges
    padding = 0.05
    x_range = max_expected
    y_range = max_observed
    
    # Ensure minimum range to avoid singular axes
    min_range = 0.1
    if x_range < min_range:
        x_range = min_range
    if y_range < min_range:
        y_range = min_range
    
    ax.set_xlim(-padding * x_range, x_range * (1 + padding))
    ax.set_ylim(-padding * y_range, y_range * (1 + padding))
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=10)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save PNG
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Saved QQ plot to {output_path}")
        
        # Save PDF
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Saved QQ plot to {pdf_path}")
    
    plt.close()


def main():
    """Main function to generate QQ plots."""
    # Load data
    df = load_phewas_data(INFILE)
    
    # Check for required columns
    if P_COL not in df.columns:
        raise ValueError(f"Column '{P_COL}' not found in input file")
    
    # Create output directory
    os.makedirs(OUTDIR, exist_ok=True)
    
    # Overall QQ plot
    print("\nGenerating overall QQ plot...")
    # Coerce to numeric, handling any non-numeric values
    pvalues = pd.to_numeric(df[P_COL], errors='coerce').values
    lambda_gc = calculate_lambda_gc(pvalues)
    print(f"Overall λ_GC = {lambda_gc:.3f}")
    
    qq_plot(pvalues, 
           title="QQ Plot - All PheWAS Results",
           output_path=os.path.join(OUTDIR, "qq_plot_overall.png"),
           lambda_gc=lambda_gc)
    
    # QQ plots by inversion (if column exists)
    if INV_COL in df.columns:
        print("\nGenerating QQ plots by inversion...")
        inversions = df[INV_COL].unique()
        
        for inv in inversions:
            if pd.isna(inv):
                continue
            
            inv_df = df[df[INV_COL] == inv]
            pvalues_inv = pd.to_numeric(inv_df[P_COL], errors='coerce').values
            lambda_gc_inv = calculate_lambda_gc(pvalues_inv)
            
            # Sanitize inversion name for filename
            inv_clean = sanitize_filename(inv)
            
            print(f"  {inv}: n={len(pvalues_inv)}, λ_GC={lambda_gc_inv:.3f}")
            
            qq_plot(pvalues_inv,
                   title=f"QQ Plot - {inv}",
                   output_path=os.path.join(OUTDIR, f"qq_plot_{inv_clean}.png"),
                   lambda_gc=lambda_gc_inv)
    
    print("\n✅ QQ plot generation complete!")
    print(f"Output directory: {OUTDIR}")


if __name__ == "__main__":
    main()
