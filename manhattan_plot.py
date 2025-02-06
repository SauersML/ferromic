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
    # Read the main data for the Manhattan plot
    results_df = pd.read_csv(data_file)
    
    # Filter to valid p-values
    valid_mask = results_df['p_value'].notnull() & (results_df['p_value'] > 0)
    if valid_mask.sum() == 0:
        print("No valid p-values found. Cannot create plot.")
        return

    # Determine the number of valid p-values and apply Bonferroni
    valid_pvals = results_df.loc[valid_mask, 'p_value']
    m = len(valid_pvals)
    results_df['bonferroni_p_value'] = np.nan
    results_df.loc[valid_mask, 'bonferroni_p_value'] = (
        results_df.loc[valid_mask, 'p_value'] * m
    ).clip(upper=1.0)
    
    # Compute -log10(p)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))
    
    # Read inversion info
    inv_df = pd.read_csv(inv_file).dropna(subset=['chr','region_start','region_end'])
    
    # Keep only inversions that overlap with valid rows in results_df
    def overlaps(inv_row, cds_df):
        c = inv_row['chr']
        s = inv_row['region_start']
        e = inv_row['region_end']
        overlap_subset = cds_df[
            (cds_df['chrom'] == c)
            & (cds_df['start'] <= e)
            & (cds_df['end'] >= s)
            & (cds_df['p_value'].notnull())
            & (cds_df['p_value'] > 0)
        ]
        return len(overlap_subset) > 0

    inv_df = inv_df[inv_df.apply(lambda row: overlaps(row, results_df), axis=1)]
    
    # Sort chromosomes in a natural order
    def chr_sort_key(ch):
        base = ch.replace('chr','')
        try:
            return (0, int(base))
        except:
            mapping = {'X': 23, 'Y': 24, 'M': 25}
            return (1, mapping.get(base, 99))

    unique_chroms = sorted(results_df['chrom'].dropna().unique(), key=chr_sort_key)
    chrom_to_index = {c: i for i, c in enumerate(unique_chroms)}
    
    # For each chromosome, find c_min and c_max
    chrom_ranges = {}
    for c in unique_chroms:
        subset = results_df[results_df['chrom'] == c]
        c_min = subset['start'].min()
        c_max = subset['end'].max()
        chrom_ranges[c] = (c_min, c_max)

    # Build top-axis x-values: each chromosome is allocated [i, i+1]
    # We place x = i + normalized_position_in_chrom
    top_xs = []
    for _, row in results_df.iterrows():
        c = row['chrom']
        if c not in chrom_to_index:
            top_xs.append(np.nan)
            continue
        i = chrom_to_index[c]
        c_min, c_max = chrom_ranges[c]
        if c_max > c_min:
            rel_pos = (row['start'] - c_min) / (c_max - c_min)
        else:
            rel_pos = 0.5
        top_xs.append(i + rel_pos)
    results_df['plot_x'] = top_xs
    
    # Z-score the effect sizes
    eff = results_df['observed_effect_size']
    eff_mean = eff.mean()
    eff_std = eff.std()
    if eff_std == 0 or np.isnan(eff_std):
        eff_std = 1.0
    eff_z = np.clip((eff - eff_mean) / eff_std, -1, 1)
    
    # Define a diverging colormap
    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['blue', 'gray', 'red'])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
    
    # Create figure with top (main) and bottom (real coords) subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[9, 1], hspace=0.05)
    sns.set_style("ticks")
    ax = fig.add_subplot(gs[0])     # top manhattan
    ax_bar = fig.add_subplot(gs[1]) # bottom linear coords
    
    # Define colors for inversions
    recurrent_color = 'purple'
    single_color = 'green'
    
    # Highlight inversion spans on the top axis
    for _, inv_row in inv_df.iterrows():
        c = inv_row['chr']
        if c not in chrom_to_index:
            continue
        i = chrom_to_index[c]
        c_min, c_max = chrom_ranges[c]
        chrom_size = c_max - c_min if c_max > c_min else 1
        inv_size = inv_row['region_end'] - inv_row['region_start']
        
        if chrom_size <= 0:
            continue
        
        rel_start = (inv_row['region_start'] - c_min) / chrom_size
        rel_end = (inv_row['region_end'] - c_min) / chrom_size
        rel_start = max(0, min(1, rel_start))
        rel_end   = max(0, min(1, rel_end))
        
        start_x = i + rel_start
        end_x   = i + rel_end
        
        typ = inv_row.get('0_single_1_recur', 0)
        inv_color = recurrent_color if typ == 1 else single_color
        
        # Distinguish large from smaller inversions
        if inv_size > 0.5 * chrom_size:
            ax.axvspan(start_x, end_x, color=inv_color, alpha=0.1, hatch='//', zorder=0)
        else:
            ax.axvspan(start_x, end_x, color=inv_color, alpha=0.2, zorder=0)

    import matplotlib.patches as mpatches
    recurrent_patch = mpatches.Patch(color=recurrent_color, alpha=0.2, label='Recurrent inversion')
    single_patch = mpatches.Patch(color=single_color, alpha=0.2, label='Single-event inversion')
    recurrent_large_patch = mpatches.Patch(facecolor=recurrent_color, hatch='//', edgecolor='black', alpha=0.1,
                                           label='Large recurrent inversion')
    single_large_patch = mpatches.Patch(facecolor=single_color, hatch='//', edgecolor='black', alpha=0.1,
                                        label='Large single-event inversion')

    # Temporary legend for inversions
    ax.legend(
        handles=[recurrent_patch, single_patch, recurrent_large_patch, single_large_patch],
        loc='upper left', fontsize=14, frameon=True
    )
    
    # Scatter the data on the top axis
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
    
    # Colorbar
    cb = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Z-scored Effect Size', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    cb.ax.text(0.0, 1.05, 'Higher dN/dS\nfor inverted', transform=cb.ax.transAxes,
               ha='left', va='bottom', fontsize=10)
    cb.ax.text(0.0, -0.05, 'Higher dN/dS\nfor non-inverted', transform=cb.ax.transAxes,
               ha='left', va='top', fontsize=10)
    
    # Significance lines
    sig_threshold = -np.log10(0.05)
    ax.axhline(sig_threshold, color='red', linestyle='--', linewidth=2, zorder=3, label='p=0.05')
    if m > 0:
        bonf_threshold = -np.log10(0.05 / m)
        if np.isfinite(bonf_threshold) and bonf_threshold > 0:
            ax.axhline(bonf_threshold, color='darkred', linestyle='--', linewidth=2, zorder=3,
                       label='p=0.05 (Bonferroni)')
    
    # Adjust y-limits
    ylim_current = ax.get_ylim()
    new_ylim = max(ylim_current[1], sig_threshold + 1)
    if m > 0 and np.isfinite(bonf_threshold):
        new_ylim = max(new_ylim, bonf_threshold + 1)
    ax.set_ylim(0, new_ylim)
    
    # Turn off x tick labels on the top axis
    ax.tick_params(labelbottom=False)
    ax.set_ylabel('-log10(p-value)', fontsize=18)
    ax.set_title('Manhattan Plot of CDS Significance', fontsize=24, fontweight='bold', pad=20)
    
    # Combine legend handles
    handles_auto, labels_auto = ax.get_legend_handles_labels()
    my_handles = [recurrent_patch, single_patch, recurrent_large_patch, single_large_patch]
    my_labels = [
        "Recurrent inversion",
        "Single-event inversion",
        "Large recurrent inversion",
        "Large single-event inversion"
    ]
    all_handles = my_handles + handles_auto
    all_labels = my_labels + labels_auto
    ax.legend(all_handles, all_labels, fontsize=14, frameon=True, loc='upper right')
    
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--', lw=0.5)
    ax.xaxis.grid(False)
    
    # Annotate top hits
    from adjustText import adjust_text
    sig_hits = results_df[valid_mask].sort_values('p_value').head(top_hits_to_annotate)
    text_objs = []
    label_xs = []
    label_ys = []
    for _, row in sig_hits.iterrows():
        symbol = row.get('gene_symbol', None)
        if symbol and symbol not in [None, 'Unknown']:
            tx = ax.text(
                row['plot_x'],
                row['neg_log_p'] + 1.0,
                symbol,
                fontsize=12,
                ha='center',
                va='bottom',
                color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )
            text_objs.append(tx)
            label_xs.append(row['plot_x'])
            label_ys.append(row['neg_log_p'])
    
    adjust_text(
        text_objs,
        x=label_xs,
        y=label_ys,
        ax=ax,
        force_text=2.0,
        force_points=2.0,
        expand_points=(2, 2),
        expand_text=(2, 2),
        lim=200
    )
    for tx, (lx, ly) in zip(text_objs, zip(label_xs, label_ys)):
        x_text, y_text = tx.get_position()
        ax.plot([x_text, lx], [y_text, ly], color='black', lw=0.5, zorder=3, alpha=0.5)
    
    # --------------------------
    # Bottom axis: real coords
    # --------------------------
    # We allocate a separate "section" for each chromosome in real base pairs,
    # so that each chromosome is placed in its own range without overlapping 
    # or mixing coords from other chromosomes.
    
    # We'll create a cumulative offset. For chromosome c, we map [c_min, c_max] 
    # to [current_offset, current_offset + (c_max - c_min)] on the bottom axis.
    # Then we place two dots for c_min, c_max at that bottom range, 
    # plus we connect them to the top axis edges (i, i+1).
    
    # Precompute the total length for each chromosome
    # to know how wide each segment is.
    chr_lengths = {}
    for c in unique_chroms:
        c_min, c_max = chrom_ranges[c]
        if c_max > c_min:
            chr_lengths[c] = c_max - c_min
        else:
            chr_lengths[c] = 1
    
    # We'll then define offsets: for chromosome i, offset is sum of all lengths before it.
    chr_offset = {}
    sum_len = 0
    for c in unique_chroms:
        chr_offset[c] = sum_len
        sum_len += chr_lengths[c]
    
    # The bottom axis overall range is [0, sum_len].
    ax_bar.set_xlim(0, sum_len)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Chromosome (Linear Coordinates, segmented by chromosome)', fontsize=16)
    
    # For each chromosome c:
    #   - we draw line from offset_c to offset_c + (c_max - c_min) at y=0.5
    #   - place red dots at each end
    #   - connect to top axis edges at (i,0) & (i+1,0)
    #   - label with c_min and c_max near the bottom dots
    for c in unique_chroms:
        i = chrom_to_index[c]
        c_min, c_max = chrom_ranges[c]
        c_len = chr_lengths[c]
        offset = chr_offset[c]
        
        # We define the bottom left and right in this segmented space
        bottom_left  = offset
        bottom_right = offset + c_len
        
        # Draw a horizontal line for this chromosome from bottom_left to bottom_right at y=0.5
        ax_bar.hlines(0.5, bottom_left, bottom_right, color='black', lw=2)
        
        # Place red dots at these edges
        ax_bar.scatter([bottom_left, bottom_right], [0.5, 0.5], color='red', zorder=5)
        
        # Label them with their real coordinate c_min, c_max
        ax_bar.text(bottom_left, 0.55, f"{c_min}", ha='center', va='bottom', fontsize=9, rotation=45)
        ax_bar.text(bottom_right, 0.55, f"{c_max}", ha='center', va='bottom', fontsize=9, rotation=45)
        
        # The top axis edges: (i,0) and (i+1,0). We'll plot red dots there as well.
        top_left  = (i, 0)
        top_right = (i+1, 0)
        ax.scatter([top_left[0], top_right[0]], [top_left[1], top_right[1]], color='red', zorder=5)
        
        # Connect left edge
        con_left = ConnectionPatch(
            xyA=(bottom_left, 0.5),  # bottom axis
            xyB=(top_left[0], top_left[1]),  # top axis
            coordsA="data", coordsB="data",
            axesA=ax_bar, axesB=ax,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_left)
        
        # Connect right edge
        con_right = ConnectionPatch(
            xyA=(bottom_right, 0.5),
            xyB=(top_right[0], top_right[1]),
            coordsA="data", coordsB="data",
            axesA=ax_bar, axesB=ax,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_right)
    
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "manhattan_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    data_file = RESULTS_DIR / 'manhattan_plot_data.csv'
    inv_file = 'inv_info.csv'
    create_manhattan_plot(data_file, inv_file=inv_file, top_hits_to_annotate=10)

if __name__ == "__main__":
    main()
