import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FixedFormatter
from adjustText import adjust_text
from pathlib import Path
import os
import json
from matplotlib.patches import ConnectionPatch

# Constants
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# Create necessary directories
for directory in [RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

def create_manhattan_plot(data_file, inv_file='inv_info.csv', top_hits_to_annotate=10):
    # Read the Manhattan plot data from the CSV file
    results_df = pd.read_csv(data_file)
    
    # Filter to valid p-values
    valid_mask = results_df['p_value'].notnull() & (results_df['p_value'] > 0)
    if valid_mask.sum() == 0:
        print("No valid p-values found. Cannot create plot.")
        return

    # Bonferroni correction and compute -log10(p)
    valid_pvals = results_df.loc[valid_mask, 'p_value']
    m = len(valid_pvals)
    results_df['bonferroni_p_value'] = np.nan
    results_df.loc[valid_mask, 'bonferroni_p_value'] = (
        results_df.loc[valid_mask, 'p_value'] * m
    ).clip(upper=1.0)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))
    
    # Read inversion file
    inv_df = pd.read_csv(inv_file).dropna(subset=['chr', 'region_start', 'region_end'])
    
    def overlaps(inv_row, cds_df):
        c = inv_row['chr']
        s = inv_row['region_start']
        e = inv_row['region_end']
        subset = cds_df[
            (cds_df['chrom'] == c)
            & (cds_df['start'] <= e)
            & (cds_df['end'] >= s)
            & (cds_df['p_value'].notnull())
            & (cds_df['p_value'] > 0)
        ]
        return len(subset) > 0

    inv_df = inv_df[inv_df.apply(lambda row: overlaps(row, results_df), axis=1)]
    
    def chr_sort_key(ch):
        base = ch.replace('chr', '')
        try:
            return (0, int(base))
        except:
            mapping = {'X': 23, 'Y': 24, 'M': 25}
            return (1, mapping.get(base, 99))
    
    unique_chroms = sorted(results_df['chrom'].dropna().unique(), key=chr_sort_key)
    chrom_to_index = {c: i for i, c in enumerate(unique_chroms)}
    chrom_ranges = {}
    for c in unique_chroms:
        chr_subset = results_df[results_df['chrom'] == c]
        c_min = chr_subset['start'].min()
        c_max = chr_subset['end'].max()
        chrom_ranges[c] = (c_min, c_max)

    # A piecewise transform to zoom in the combined inversion region [A,B] for each chromosome
    def transform_coordinate(L, A, B):
        # A,B define the region to zoom. We map [0,A]→[0,0.2], [A,B]→[0.2,0.8], [B,1]→[0.8,1].
        if L <= A:
            return (L / A) * 0.2 if A > 0 else 0
        elif L <= B:
            return 0.2 + ((L - A) / (B - A)) * 0.6 if (B - A) > 0 else 0.2
        else:
            return 0.8 + ((L - B) / (1 - B)) * 0.2 if (1 - B) > 0 else 0.8

    # For each chromosome, find the combined minimal and maximal boundaries over all inversions
    chr_trans_info = {}
    for c in unique_chroms:
        c_inv = inv_df[inv_df['chr'] == c]
        c_min, c_max = chrom_ranges[c]
        if not c_inv.empty and c_max > c_min:
            norm_starts = (c_inv['region_start'] - c_min) / (c_max - c_min)
            norm_ends = (c_inv['region_end'] - c_min) / (c_max - c_min)
            A = norm_starts.min()
            B = norm_ends.max()
            A = max(0, min(1, A))
            B = max(0, min(1, B))
            if B < A:
                A, B = B, A
            chr_trans_info[c] = {
                'A': A,
                'B': B,
                'fA': transform_coordinate(A, A, B),
                'fB': transform_coordinate(B, A, B)
            }

    # Build the final x-values for the main plot
    xs = []
    for _, row in results_df.iterrows():
        c = row['chrom']
        if c not in chrom_ranges:
            xs.append(np.nan)
            continue
        c_min, c_max = chrom_ranges[c]
        if c_max > c_min:
            L = (row['start'] - c_min) / (c_max - c_min)
        else:
            L = 0.5
        if c in chr_trans_info:
            fL = transform_coordinate(L, chr_trans_info[c]['A'], chr_trans_info[c]['B'])
        else:
            fL = L
        xs.append(chrom_to_index[c] + fL if c in chrom_to_index else np.nan)

    results_df['plot_x'] = xs
    
    # Z-scored effect sizes for coloring
    eff = results_df['observed_effect_size']
    eff_mean = eff.mean()
    eff_std = eff.std()
    if eff_std == 0 or np.isnan(eff_std):
        eff_std = 1.0
    eff_z = (eff - eff_mean) / eff_std
    eff_z = np.clip(eff_z, -1, 1)
    
    # Define a custom color map
    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['blue', 'gray', 'red'])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
    
    # Prepare the figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[9, 1], hspace=0.05)
    sns.set_style("ticks")
    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharex=ax)
    
    # Colors for inversions
    recurrent_color = 'purple'
    single_color = 'green'
    
    # Highlight inversion regions on the main axis
    for _, inv_row in inv_df.iterrows():
        inv_chr = inv_row['chr']
        if inv_chr not in chrom_to_index:
            continue
        c_idx = chrom_to_index[inv_chr]
        c_min, c_max = chrom_ranges[inv_chr]
        inv_size = inv_row['region_end'] - inv_row['region_start']
        chrom_size = c_max - c_min if c_max > c_min else 1
        if chrom_size <= 0:
            continue

        rel_start = (inv_row['region_start'] - c_min) / chrom_size
        rel_end = (inv_row['region_end'] - c_min) / chrom_size
        rel_start = max(0, min(1, rel_start))
        rel_end = max(0, min(1, rel_end))

        inv_x_start = c_idx + rel_start
        inv_x_end = c_idx + rel_end

        inversion_type = inv_row.get('0_single_1_recur', 0)
        if inversion_type == 1:
            inv_color = recurrent_color
        else:
            inv_color = single_color

        if inv_size > 0.5 * chrom_size:
            # Large inversion
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.1, zorder=0, hatch='//')
        else:
            # Smaller inversion
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.2, zorder=0)
    
    import matplotlib.patches as mpatches
    recurrent_patch = mpatches.Patch(color=recurrent_color, alpha=0.2, label='Recurrent inversion')
    single_patch = mpatches.Patch(color=single_color, alpha=0.2, label='Single-event inversion')
    recurrent_large_patch = mpatches.Patch(
        facecolor=recurrent_color, hatch='//', edgecolor='black', alpha=0.1,
        label='Large recurrent inversion'
    )
    single_large_patch = mpatches.Patch(
        facecolor=single_color, hatch='//', edgecolor='black', alpha=0.1,
        label='Large single-event inversion'
    )
    
    # Temporary legend for the inversion patches
    ax.legend(
        handles=[recurrent_patch, single_patch, recurrent_large_patch, single_large_patch],
        loc='upper left', fontsize=14, frameon=True
    )
    
    # Plot the main scatter of data
    scatter = ax.scatter(
        results_df['plot_x'],
        results_df['neg_log_p'],
        c=eff_z,
        cmap=cmap,
        norm=norm,
        s=50,
        alpha=0.7,
        linewidth=0,
        zorder=2
    )
    
    # Add a colorbar
    cb = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Z-scored Effect Size', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    cb.ax.text(0.0, 1.05, 'Higher dN/dS\nfor inverted', transform=cb.ax.transAxes,
               ha='left', va='bottom', fontsize=10)
    cb.ax.text(0.0, -0.05, 'Higher dN/dS\nfor non-inverted', transform=cb.ax.transAxes,
               ha='left', va='top', fontsize=10)
    
    # Significance thresholds
    sig_threshold = -np.log10(0.05)
    ax.axhline(
        y=sig_threshold, color='red', linestyle='--', linewidth=2, zorder=3,
        label='p=0.05'
    )
    if m > 0:
        bonf_threshold = -np.log10(0.05 / m)
        if np.isfinite(bonf_threshold) and bonf_threshold > 0:
            ax.axhline(
                y=bonf_threshold, color='darkred', linestyle='--', linewidth=2,
                zorder=3, label='p=0.05 (Bonferroni)'
            )
    
    # Adjust the y-limits to ensure significance lines are visible
    current_ylim = ax.get_ylim()
    new_ylim = max(current_ylim[1], sig_threshold + 1)
    if m > 0 and np.isfinite(bonf_threshold):
        new_ylim = max(new_ylim, bonf_threshold + 1)
    ax.set_ylim(0, new_ylim)
    
    # Turn off the bottom x-axis labels in the top plot
    ax.tick_params(labelbottom=False)
    
    # Main axis labeling
    ax.set_ylabel('-log10(p-value)', fontsize=18)
    ax.set_title('Manhattan Plot of CDS Significance', fontsize=24, fontweight='bold', pad=20)
    
    # Combine legends
    handles_auto, labels_auto = ax.get_legend_handles_labels()
    all_handles = [
        recurrent_patch, single_patch,
        recurrent_large_patch, single_large_patch
    ] + handles_auto
    ax.legend(handles=all_handles, fontsize=14, frameon=True, loc='upper right')
    
    # Grid lines
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--', lw=0.5)
    ax.xaxis.grid(False)
    
    # Annotate significant hits
    significant_hits = results_df[valid_mask].sort_values('p_value').head(top_hits_to_annotate)
    text_objects = []
    label_points_x = []
    label_points_y = []
    for _, hit_row in significant_hits.iterrows():
        c_symbol = hit_row.get('gene_symbol', None)
        label_txt = None if (c_symbol is None or c_symbol in [None, 'Unknown']) else c_symbol
        if label_txt:
            txt = ax.text(
                hit_row['plot_x'],
                hit_row['neg_log_p'] + 1.0,
                label_txt,
                fontsize=12,
                ha='center',
                va='bottom',
                color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )
            text_objects.append(txt)
            label_points_x.append(hit_row['plot_x'])
            label_points_y.append(hit_row['neg_log_p'])
    
    adjust_text(
        text_objects,
        x=label_points_x,
        y=label_points_y,
        ax=ax,
        force_text=2.0,
        force_points=2.0,
        expand_points=(2, 2),
        expand_text=(2, 2),
        lim=200
    )
    for txt, (x, y) in zip(text_objects, zip(label_points_x, label_points_y)):
        x_text, y_text = txt.get_position()
        ax.plot([x_text, x], [y_text, y], color='black', lw=0.5, zorder=3, alpha=0.5)
    
    # -------------------------
    # Bottom axis (linear coordinates)
    # -------------------------
    # We'll set the bottom x-limits to span the same range as the top's *chromosome index*,
    # but for clarity, we preserve the chromosome-based position from the top axis.
    # We won't alter it to a global basepair min/max, because we do want them to visually align in x-axis space.
    ax_bar.set_xlim(ax.get_xlim())
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Chromosome (Linear Coordinates)', fontsize=16)
    
    # Draw a horizontal line for each chromosome in the lower axis,
    # exactly from c_idx to c_idx+1, to keep alignment with the top plot,
    # and then add text showing the actual numeric boundaries.
    for c in unique_chroms:
        c_idx = chrom_to_index[c]
        ax_bar.hlines(0.5, c_idx, c_idx + 1, color='black', lw=2)
    
    # We add ticks that show c_min and c_max for each chromosome. We'll place them at x=c_idx and x=c_idx+1
    # The numeric labels will be c_min and c_max in basepairs. 
    tick_positions = []
    tick_labels = []
    for c in unique_chroms:
        c_idx = chrom_to_index[c]
        c_min, c_max = chrom_ranges[c]
        tick_positions.extend([c_idx, c_idx + 1])
        tick_labels.extend([f"{c_min}", f"{c_max}"])
    ax_bar.set_xticks(tick_positions)
    ax_bar.set_xticklabels(tick_labels, fontsize=10, rotation=45)
    
    # Connect top axis edges to the actual linear boundaries on the bottom axis with red dots
    # Because we're using the same x-axis spacing on top and bottom, the left boundary is at x=c_idx,
    # and the right boundary is at x=c_idx+1. We'll place two red dots on top (at y=0) and two red dots
    # on bottom (at y=0.5) for each chromosome, and connect them with lines. 
    for c in unique_chroms:
        c_idx = chrom_to_index[c]
        
        # The top boundary points for chromosome c
        top_left = (c_idx, 0)
        top_right = (c_idx + 1, 0)
        
        # The bottom boundary points for chromosome c
        bottom_left = (c_idx, 0.5)
        bottom_right = (c_idx + 1, 0.5)
        
        # Plot them as red dots
        ax.scatter([top_left[0], top_right[0]], [top_left[1], top_right[1]], color='red', zorder=5)
        ax_bar.scatter([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], color='red', zorder=5)
        
        # Connect them with lines
        con_left = ConnectionPatch(
            xyA=bottom_left,
            xyB=top_left,
            coordsA="data", coordsB="data",
            axesA=ax_bar, axesB=ax,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_left)
        con_right = ConnectionPatch(
            xyA=bottom_right,
            xyB=top_right,
            coordsA="data", coordsB="data",
            axesA=ax_bar, axesB=ax,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_right)
    
    # Final layout
    plt.tight_layout()
    plt.savefig(os.path.join("plots", 'manhattan_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    data_file = RESULTS_DIR / 'manhattan_plot_data.csv'
    inv_file = 'inv_info.csv'
    create_manhattan_plot(data_file, inv_file=inv_file, top_hits_to_annotate=10)

if __name__ == "__main__":
    main()
