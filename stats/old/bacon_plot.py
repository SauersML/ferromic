#!/usr/bin/env python3
"""
Generate BACON correction QQ plots and apply FDR correction

Reads bacon_corrected_results.tsv and:
1. Applies Benjamini-Hochberg FDR correction globally and per-inversion
2. Creates QQ plots comparing unadjusted vs BACON-corrected p-values

Color scheme:
  Orange: Unadjusted p-values
  Blue: BACON-corrected p-values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.multitest import multipletests

# Read data
results = pd.read_csv("stats/bacon_corrected_results.tsv", sep="\t")
params = pd.read_csv("stats/bacon_parameters.tsv", sep="\t")

# Apply FDR correction (Benjamini-Hochberg) across ALL tests
_, results['P_adjusted_BH'], _, _ = multipletests(
    results['P_corrected'], 
    method='fdr_bh'
)

# Save updated results
results.to_csv("stats/bacon_corrected_results.tsv", sep="\t", index=False)
print("Applied FDR correction:")
print("  - P_adjusted_BH: BH-FDR adjusted across all inversions and phenotypes")
print()

# Get unique inversions
inversions = results['Inversion'].unique()

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

for idx, inv in enumerate(inversions):
    ax = axes[idx]
    
    # Subset data
    inv_data = results[results['Inversion'] == inv].copy()
    
    # Sort p-values
    p_raw_sorted = np.sort(inv_data['P_raw'].values)
    p_corr_sorted = np.sort(inv_data['P_corrected'].values)
    n = len(p_raw_sorted)
    
    # Expected uniform distribution
    expected = np.arange(1, n + 1) / (n + 1)
    
    # Convert to -log10
    obs_raw = -np.log10(p_raw_sorted)
    obs_corr = -np.log10(p_corr_sorted)
    exp_vals = -np.log10(expected)
    
    # Plot unadjusted
    ax.scatter(exp_vals, obs_raw, 
              c='#CC6600', alpha=0.4, s=20, edgecolors='none',
              label='Unadjusted')
    
    # Plot BACON-corrected
    ax.scatter(exp_vals, obs_corr, 
              c='#0066CC', alpha=0.4, s=20, edgecolors='none',
              label='BACON')
    
    # Identity line
    max_val = max(exp_vals.max(), obs_raw.max(), obs_corr.max())
    ax.plot([0, max_val], [0, max_val], 'k--', lw=1.5, alpha=0.7)
    
    # Labels
    inv_short = inv.replace('chr', '').replace('-INV-', ':')
    ax.set_title(inv_short, fontsize=10)
    ax.set_xlabel("Expected -log10(P)", fontsize=9)
    ax.set_ylabel("Observed -log10(P)", fontsize=9)
    
    # Add parameters
    inv_params = params[params['Inversion'] == inv].iloc[0]
    lambda_gc = inv_params['sigma_0'] ** 2
    ax.text(0.05, 0.95, 
            f"μ₀={inv_params['mu_0']:.3f}\nσ₀={inv_params['sigma_0']:.3f}\nλ={lambda_gc:.3f}",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend (only on first plot)
    if idx == 0:
        ax.legend(loc='upper left', fontsize=7, framealpha=0.9)

plt.tight_layout()
plt.savefig("stats/bacon_qq_plots_python.png", dpi=150, bbox_inches='tight')
print("Saved: stats/bacon_qq_plots_python.png")
plt.close()

# Print summary statistics
print("\n" + "="*100)
print("Significant Associations (FDR < 0.05)")
print("="*100)
print()

sig_total = (results['P_adjusted_BH'] < 0.05).sum()
print(f"Total significant at FDR < 0.05: {sig_total}")
print()

for inv in results['Inversion'].unique():
    inv_data = results[results['Inversion'] == inv]
    sig_count = (inv_data['P_adjusted_BH'] < 0.05).sum()
    
    inv_short = inv.replace('chr', '').replace('-INV-', ':')
    print(f"{inv_short}: {sig_count} significant")

print()
print("="*100)
print("\nBACON correction and FDR adjustment complete!")
