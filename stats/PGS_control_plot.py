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

# Colors - modern palette (matching manhattan_phe.py for risk direction)
COLOR_RISK_INC = "#2B6CB0"    # blue for risk increasing (OR > 1)
COLOR_RISK_DEC = "#C53030"    # red for risk decreasing (OR < 1)
COLOR_NEUTRAL = "#CCCCCC"     # gray for non-significant
ALPHA_POINT = 0.75
EDGE_COLOR = "#333333"
EDGE_WIDTH = 0.8

# Triangle sizing (matching manhattan_phe.py)
TRI_BASE_SIZE = 260.0    # triangle area (pt^2) when OR = 1.0
TRI_OR_MIN = 0.67        # minimum OR for scaling (protective effects)
TRI_OR_MAX = 1.5         # maximum OR for scaling (risk effects)


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


def scale_all_sizes(or_values: pd.Series) -> np.ndarray:
    """Scale marker sizes linearly with the odds ratio (clamped to [0.67, 1.5])."""
    arr = pd.to_numeric(or_values, errors="coerce").to_numpy()
    arr = np.nan_to_num(arr, nan=1.0, posinf=TRI_OR_MAX, neginf=TRI_OR_MIN)
    arr[arr <= 0] = 1.0

    # Clamp to [0.67, 1.5] range first
    arr = np.clip(arr, TRI_OR_MIN, TRI_OR_MAX)

    # Linear scaling: OR=1.0 keeps baseline size, OR<1 shrinks, OR>1 enlarges
    return TRI_BASE_SIZE * arr


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

    Points use triangles (up for risk increasing, down for decreasing) scaled by OR.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # Prepare data
    df_plot = df.copy()

    # Calculate log p-values
    df_plot["log10_p_unadj"] = -np.log10(df_plot["P_Value_NoCustomControls"])
    df_plot["log10_p_adj"] = -np.log10(df_plot["P_Value"])

    # Scale marker sizes by OR (matching manhattan_phe.py approach)
    df_plot["size_unadj"] = scale_all_sizes(df_plot["OR_NoCustomControls"])
    df_plot["size_adj"] = scale_all_sizes(df_plot["OR"])

    # Determine risk direction and colors based on OR
    # OR > 1 = risk increasing (blue, triangle up)
    # OR < 1 = risk decreasing (red, triangle down)
    df_plot["risk_dir_unadj"] = df_plot["OR_NoCustomControls"].apply(lambda x: "inc" if x >= 1.0 else "dec")
    df_plot["risk_dir_adj"] = df_plot["OR"].apply(lambda x: "inc" if x >= 1.0 else "dec")
    df_plot["color_unadj"] = df_plot["risk_dir_unadj"].apply(lambda x: COLOR_RISK_INC if x == "inc" else COLOR_RISK_DEC)
    df_plot["color_adj"] = df_plot["risk_dir_adj"].apply(lambda x: COLOR_RISK_INC if x == "inc" else COLOR_RISK_DEC)
    df_plot["marker_unadj"] = df_plot["risk_dir_unadj"].apply(lambda x: "^" if x == "inc" else "v")
    df_plot["marker_adj"] = df_plot["risk_dir_adj"].apply(lambda x: "^" if x == "inc" else "v")

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
               color='#888888', linewidth=1.5, alpha=0.4, zorder=1)

    # Plot unadjusted points with triangles
    for _, row in df_plot.iterrows():
        ax.scatter(row["x_unadj_jitter"], row["log10_p_unadj"],
                  s=row["size_unadj"], marker=row["marker_unadj"],
                  c=row["color_unadj"], alpha=ALPHA_POINT,
                  edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, zorder=3)

    # Plot adjusted points with triangles
    for _, row in df_plot.iterrows():
        ax.scatter(row["x_adj_jitter"], row["log10_p_adj"],
                  s=row["size_adj"], marker=row["marker_adj"],
                  c=row["color_adj"], alpha=ALPHA_POINT,
                  edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, zorder=3)

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
    ax.set_xticklabels(['PGS-unadjusted', 'PGS-adjusted'], fontsize=12)
    ax.set_ylabel("-log₁₀(P-value)", fontsize=14)
    ax.set_xlim(-0.3, 1.5)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)

    # Create combined legend for direction and OR scaling
    or_levels = [TRI_OR_MIN, 1.0, TRI_OR_MAX]
    or_labels = [f"OR = {val:.2f}" for val in or_levels]
    or_sizes = scale_all_sizes(pd.Series(or_levels))

    legend_elements = [
        # Direction indicators
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor=COLOR_RISK_INC, markersize=10,
               label='Risk increasing (OR > 1)',
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH),
        Line2D([0], [0], marker='v', color='w',
               markerfacecolor=COLOR_RISK_DEC, markersize=10,
               label='Risk decreasing (OR < 1)',
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH),
        # Size legend spacer
        Line2D([0], [0], linestyle='None', label=''),
        # OR scaling examples
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor='#666666', markersize=np.sqrt(or_sizes[0]/10),
               label=or_labels[0],
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH),
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor='#666666', markersize=np.sqrt(or_sizes[1]/10),
               label=or_labels[1],
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH),
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor='#666666', markersize=np.sqrt(or_sizes[2]/10),
               label=or_labels[2],
               markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH),
    ]

    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
             fancybox=False, shadow=False, fontsize=10)

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
