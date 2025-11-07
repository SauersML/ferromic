#!/usr/bin/env python3
"""
Generate volcano plot for PGS control analysis results.
Visualizes effect sizes vs significance with and without custom PGS controls.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Configuration
INPUT_FILE = "data/PGS_controls.tsv"
OUT_PDF = "PGS_control_volcano.pdf"
OUT_PNG = "PGS_control_volcano.png"
OUT_COMPARISON_PDF = "PGS_control_comparison.pdf"
OUT_COMPARISON_PNG = "PGS_control_comparison.png"

# Styling
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colors - modern palette
COLOR_WITH_CONTROLS = "#2E86AB"      # deep blue
COLOR_WITHOUT_CONTROLS = "#E63946"   # vibrant red
COLOR_NEUTRAL = "#CCCCCC"            # gray for non-significant
ALPHA_POINT = 0.75
EDGE_COLOR = "#333333"
EDGE_WIDTH = 0.8


def load_data(path: str) -> pd.DataFrame:
    """Load and validate PGS controls data."""
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: '{path}' not found.")
    
    df = pd.read_csv(path, sep="\t")
    
    required = ["Phenotype", "Category", "OR", "OR_95CI_Lower", "OR_95CI_Upper",
                "OR_NoCustomControls", "OR_NoCustomControls_95CI_Lower", 
                "OR_NoCustomControls_95CI_Upper", "P_Value", "P_Value_NoCustomControls", "BH_FDR_Q"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: missing required column(s): {', '.join(missing)}")
    
    # Convert to numeric
    for col in ["OR", "OR_95CI_Lower", "OR_95CI_Upper", 
                "OR_NoCustomControls", "OR_NoCustomControls_95CI_Lower",
                "OR_NoCustomControls_95CI_Upper", "P_Value", "P_Value_NoCustomControls", "BH_FDR_Q"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Clean phenotype names
    df["Phenotype_Clean"] = df["Phenotype"].str.replace("_", " ")
    
    # Sort by FDR q-value (most significant first)
    df = df.sort_values("BH_FDR_Q", ascending=True).reset_index(drop=True)
    
    return df


def plot_volcano(df: pd.DataFrame, out_pdf: str, out_png: str):
    """Create volcano plot comparing effect sizes vs significance."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    df_plot = df.copy()
    
    # Unadjusted (without controls) - use OR directly
    df_plot["or_unadj"] = df_plot["OR_NoCustomControls"]
    df_plot["log10_p_unadj"] = -np.log10(df_plot["P_Value_NoCustomControls"])
    
    # Adjusted (with controls) - use OR directly
    df_plot["or_adj"] = df_plot["OR"]
    df_plot["log10_p_adj"] = -np.log10(df_plot["P_Value"])
    
    # Draw arrows from unadjusted to adjusted
    for _, row in df_plot.iterrows():
        ax.annotate('', 
                   xy=(row["or_adj"], row["log10_p_adj"]),
                   xytext=(row["or_unadj"], row["log10_p_unadj"]),
                   arrowprops=dict(arrowstyle='->', color='#888888', 
                                 lw=1.2, alpha=0.5, shrinkA=5, shrinkB=5))
    
    # Plot unadjusted points (without controls)
    ax.scatter(df_plot["or_unadj"], df_plot["log10_p_unadj"],
              s=100, c=COLOR_WITHOUT_CONTROLS, alpha=ALPHA_POINT,
              edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, 
              zorder=3, label='Unadjusted')
    
    # Plot adjusted points (with controls)
    ax.scatter(df_plot["or_adj"], df_plot["log10_p_adj"],
              s=100, c=COLOR_WITH_CONTROLS, alpha=ALPHA_POINT,
              edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH,
              zorder=4, label='Adjusted for PGS')
    
    # Null line at OR=1
    ax.axvline(1.0, color='#999999', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    
    # Significance line at p=0.05
    sig_line_y = -np.log10(0.05)
    ax.axhline(sig_line_y, color='#666666', linestyle='--',
              linewidth=1.5, alpha=0.5, zorder=1)
    
    # Labels for adjusted points
    for _, row in df_plot.iterrows():
        ax.annotate(row["Phenotype_Clean"],
                   xy=(row["or_adj"], row["log10_p_adj"]),
                   xytext=(6, 6), textcoords='offset points',
                   fontsize=9, alpha=0.85)
    
    ax.set_xlabel("Odds Ratio", fontsize=14)
    ax.set_ylabel("-log₁₀(P-value)", fontsize=14)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=False, 
             shadow=False, fontsize=11)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    
    print(f"✅ Saved: {out_pdf}")
    print(f"✅ Saved: {out_png}")
    
    plt.close()


def plot_comparison(df: pd.DataFrame, out_pdf: str, out_png: str):
    """Create comparison plot with raw vs PGS adjusted on x-axis, log p-values on y-axis.
    
    Points are scaled by effect size (OR) and connected by lines.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    df_plot = df.copy()
    
    # Calculate log p-values
    df_plot["log10_p_unadj"] = -np.log10(df_plot["P_Value_NoCustomControls"])
    df_plot["log10_p_adj"] = -np.log10(df_plot["P_Value"])
    
    # Calculate effect sizes for scaling (use absolute log OR for size)
    df_plot["effect_size_unadj"] = np.abs(np.log(df_plot["OR_NoCustomControls"]))
    df_plot["effect_size_adj"] = np.abs(np.log(df_plot["OR"]))
    
    # Normalize effect sizes for point scaling (50 to 500 range)
    all_effects = pd.concat([df_plot["effect_size_unadj"], df_plot["effect_size_adj"]])
    min_effect = all_effects.min()
    max_effect = all_effects.max()
    
    def scale_size(effect):
        if max_effect == min_effect:
            return 150
        return 50 + 450 * (effect - min_effect) / (max_effect - min_effect)
    
    df_plot["size_unadj"] = df_plot["effect_size_unadj"].apply(scale_size)
    df_plot["size_adj"] = df_plot["effect_size_adj"].apply(scale_size)
    
    # X-axis positions: 0 for unadjusted, 1 for adjusted
    x_unadj = 0
    x_adj = 1
    
    # Add jitter to x positions
    np.random.seed(42)  # For reproducibility
    jitter_amount = 0.05
    df_plot["x_unadj_jitter"] = x_unadj + np.random.uniform(-jitter_amount, jitter_amount, len(df_plot))
    df_plot["x_adj_jitter"] = x_adj + np.random.uniform(-jitter_amount, jitter_amount, len(df_plot))
    
    # Draw connecting lines between raw and adjusted for each phenotype
    for _, row in df_plot.iterrows():
        ax.plot([row["x_unadj_jitter"], row["x_adj_jitter"]], 
               [row["log10_p_unadj"], row["log10_p_adj"]],
               color='#888888', linewidth=1.5, alpha=0.6, zorder=1)
    
    # Plot unadjusted points
    for _, row in df_plot.iterrows():
        ax.scatter(row["x_unadj_jitter"], row["log10_p_unadj"],
                  s=row["size_unadj"], c=COLOR_WITHOUT_CONTROLS, 
                  alpha=ALPHA_POINT, edgecolors=EDGE_COLOR, 
                  linewidths=EDGE_WIDTH, zorder=3)
    
    # Plot adjusted points
    for _, row in df_plot.iterrows():
        ax.scatter(row["x_adj_jitter"], row["log10_p_adj"],
                  s=row["size_adj"], c=COLOR_WITH_CONTROLS, 
                  alpha=ALPHA_POINT, edgecolors=EDGE_COLOR, 
                  linewidths=EDGE_WIDTH, zorder=3)
    
    # Add phenotype labels at adjusted points, alternating left and right
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        if i % 2 == 0:  # Even counter: label on right
            ax.annotate(row["Phenotype_Clean"],
                       xy=(row["x_adj_jitter"], row["log10_p_adj"]),
                       xytext=(8, 0), textcoords='offset points',
                       fontsize=9, alpha=0.85, va='center', ha='left')
        else:  # Odd counter: label on left
            ax.annotate(row["Phenotype_Clean"],
                       xy=(row["x_adj_jitter"], row["log10_p_adj"]),
                       xytext=(-8, 0), textcoords='offset points',
                       fontsize=9, alpha=0.85, va='center', ha='right')
    
    # Significance line at p=0.05
    sig_line_y = -np.log10(0.05)
    ax.axhline(sig_line_y, color='#666666', linestyle='--',
              linewidth=1.5, alpha=0.5, zorder=1, label='p = 0.05')
    
    # Styling
    ax.set_xticks([x_unadj, x_adj])
    ax.set_xticklabels(['Raw/Unadjusted', 'PGS/Adjusted'], fontsize=12)
    ax.set_ylabel("-log₁₀(P-value)", fontsize=14)
    ax.set_xlim(-0.3, 1.5)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # Create legend for effect size scaling
    # Calculate actual OR values for example sizes
    # effect_size = |log(OR)|, so OR = exp(effect_size) or exp(-effect_size)
    # Show both OR > 1 and OR < 1 versions
    
    # Get actual min/max effect sizes from data
    example_effects = [min_effect, (min_effect + max_effect) / 2, max_effect]
    example_sizes = [scale_size(e) for e in example_effects]
    example_ors_high = [np.exp(e) for e in example_effects]
    example_ors_low = [np.exp(-e) for e in example_effects]
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='#666666', markersize=np.sqrt(example_sizes[0]/10),
               label=f'Small (OR≈{example_ors_high[0]:.2f} or {example_ors_low[0]:.2f})', 
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH),
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='#666666', markersize=np.sqrt(example_sizes[1]/10),
               label=f'Medium (OR≈{example_ors_high[1]:.2f} or {example_ors_low[1]:.2f})', 
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH),
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='#666666', markersize=np.sqrt(example_sizes[2]/10),
               label=f'Large (OR≈{example_ors_high[2]:.2f} or {example_ors_low[2]:.2f})', 
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=False, shadow=False, fontsize=10, title='Effect Size (|log OR|)')
    
    plt.tight_layout()
    
    # Save
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    
    print(f"✅ Saved: {out_pdf}")
    print(f"✅ Saved: {out_png}")
    
    plt.close()


def main():
    """Main execution."""
    print(f"Loading data from {INPUT_FILE}...")
    df = load_data(INPUT_FILE)
    
    print(f"Found {len(df)} phenotypes")
    print(f"Categories: {', '.join(df['Category'].unique())}")
    
    print("\nGenerating volcano plot...")
    plot_volcano(df, OUT_PDF, OUT_PNG)
    
    print("\nGenerating comparison plot...")
    plot_comparison(df, OUT_COMPARISON_PDF, OUT_COMPARISON_PNG)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
