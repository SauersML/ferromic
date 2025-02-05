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

    valid_pvals = results_df.loc[valid_mask, 'p_value']
    m = len(valid_pvals)

    # Bonferroni correction and compute -log10(p)
    results_df['bonferroni_p_value'] = np.nan
    results_df.loc[valid_mask, 'bonferroni_p_value'] = results_df.loc[valid_mask, 'p_value'] * m
    results_df['bonferroni_p_value'] = results_df['bonferroni_p_value'].clip(upper=1.0)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))
    
    # Read inversion file
    inv_df = pd.read_csv(inv_file).dropna(subset=['chr', 'region_start', 'region_end'])
    
    def overlaps(inv_row, cds_df):
        c = inv_row['chr']
        s = inv_row['region_start']
        e = inv_row['region_end']
        subset = cds_df[(cds_df['chrom'] == c) & 
                        (cds_df['start'] <= e) & 
                        (cds_df['end'] >= s) & 
                        (cds_df['p_value'].notnull()) & (cds_df['p_value'] > 0)]
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
        chr_df = results_df[results_df['chrom'] == c]
        c_min = chr_df['start'].min()
        c_max = chr_df['end'].max()
        chrom_ranges[c] = (c_min, c_max)
    
    def transform_coordinate(L, A, B):
        if L <= A:
            return (L / A) * 0.2 if A > 0 else 0
        elif L <= B:
            return 0.2 + ((L - A) / (B - A)) * 0.6 if (B - A) > 0 else 0.2
        else:
            return 0.8 + ((L - B) / (1 - B)) * 0.2 if (1 - B) > 0 else 0.8

    # For each chromosome, if an inversion exists, compute its normalized boundaries and the transformed (zoomed) boundaries.
    chr_trans_info = {}
    for c in unique_chroms:
        c_inv = inv_df[inv_df['chr'] == c]
        c_min, c_max = chrom_ranges[c]
        if not c_inv.empty and c_max > c_min:
            inv_row = c_inv.iloc[0]
            A = (inv_row['region_start'] - c_min) / (c_max - c_min)
            B = (inv_row['region_end'] - c_min) / (c_max - c_min)
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
    xs = []
    for _, row in results_df.iterrows():
        c = row['chrom']
        c_min, c_max = chrom_ranges[c]
        if c_max > c_min:
            L = (row['start'] - c_min) / (c_max - c_min)
        else:
            L = 0.5
        if c in chr_trans_info:
            fL = transform_coordinate(L, chr_trans_info[c]['A'], chr_trans_info[c]['B'])
        else:
            fL = L
        xs.append(chrom_to_index[c] + fL)
    results_df['plot_x'] = xs
    
    eff = results_df['observed_effect_size']
    eff_mean = eff.mean()
    eff_std = eff.std()
    if eff_std == 0 or np.isnan(eff_std):
        eff_std = 1.0
    eff_z = (eff - eff_mean) / eff_std
    eff_z = np.clip(eff_z, -1, 1)
    
    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['blue', 'gray', 'red'])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[9, 1], hspace=0.05)
    sns.set_style("ticks")
    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharex=ax)
    
    recurrent_color = 'purple'
    single_color = 'green'
    
    for _, inv in inv_df.iterrows():
        inv_chr = inv['chr']
        if inv_chr not in chrom_to_index:
            continue
        c_idx = chrom_to_index[inv_chr]
        c_min, c_max = chrom_ranges[inv_chr]
        inv_size = inv['region_end'] - inv['region_start']
        chrom_size = c_max - c_min if c_max > c_min else 1
        if c_max > c_min:
            rel_start = (inv['region_start'] - c_min) / (c_max - c_min)
            rel_end = (inv['region_end'] - c_min) / (c_max - c_min)
        else:
            rel_start = 0.4
            rel_end = 0.6
        inv_x_start = c_idx + max(0, min(rel_start, 1))
        inv_x_end = c_idx + min(1, max(rel_end, 0))
        inversion_type = inv.get('0_single_1_recur', 0)
        if inversion_type == 1:
            inv_color = recurrent_color
        else:
            inv_color = single_color
        if inv_size > 0.5 * chrom_size:
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.1, zorder=0, hatch='//')
        else:
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.2, zorder=0)
    
    import matplotlib.patches as mpatches
    recurrent_patch = mpatches.Patch(color=recurrent_color, alpha=0.2, label='Recurrent inversion')
    single_patch = mpatches.Patch(color=single_color, alpha=0.2, label='Single-event inversion')
    recurrent_large_patch = mpatches.Patch(facecolor=recurrent_color, hatch='//', edgecolor='black', alpha=0.1, label='Large recurrent inversion')
    single_large_patch = mpatches.Patch(facecolor=single_color, hatch='//', edgecolor='black', alpha=0.1, label='Large single-event inversion')
    
    ax.legend(
        handles=[recurrent_patch, single_patch, recurrent_large_patch, single_large_patch],
        loc='upper left', fontsize=14, frameon=True
    )
    
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
    
    cb = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Z-scored Effect Size', fontsize=16)
    cb.ax.tick_params(labelsize=14)

    cb.ax.text(0.0, 1.05, 'Higher dN/dS\nfor inverted', transform=cb.ax.transAxes, ha='left', va='bottom', fontsize=10)
    cb.ax.text(0.0, -0.05, 'Higher dN/dS\nfor non-inverted', transform=cb.ax.transAxes, ha='left', va='top', fontsize=10)
    
    sig_threshold = -np.log10(0.05)
    ax.axhline(y=sig_threshold, color='red', linestyle='--', linewidth=2, zorder=3, label='p=0.05')
    if m > 0:
        bonf_threshold = -np.log10(0.05 / m)
        if np.isfinite(bonf_threshold) and bonf_threshold > 0:
            ax.axhline(y=bonf_threshold, color='darkred', linestyle='--', linewidth=2, zorder=3,
                       label='p=0.05 (Bonferroni)')
    
    current_ylim = ax.get_ylim()
    new_ylim = max(current_ylim[1], sig_threshold+1)
    if m > 0 and np.isfinite(bonf_threshold):
        new_ylim = max(new_ylim, bonf_threshold+1)
    ax.set_ylim(0, new_ylim)
    
    ax.tick_params(labelbottom=False)
    
    ax.set_ylabel('-log10(p-value)', fontsize=18)
    ax.set_title('Manhattan Plot of CDS Significance', fontsize=24, fontweight='bold', pad=20)
    
    handles_auto, labels_auto = ax.get_legend_handles_labels()
    all_handles = [recurrent_patch, single_patch, recurrent_large_patch, single_large_patch] + handles_auto
    ax.legend(handles=all_handles, fontsize=14, frameon=True, loc='upper right')
    
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--', lw=0.5)
    ax.xaxis.grid(False)
    
    significant_hits = results_df[valid_mask].sort_values('p_value').head(top_hits_to_annotate)
    
    text_objects = []
    label_points_x = []
    label_points_y = []
    for _, hit_row in significant_hits.iterrows():
        cds = hit_row['CDS']
        gene_symbol = hit_row.get('gene_symbol', None)
        if gene_symbol is None or gene_symbol in [None, 'Unknown']:
            label_txt = None
        else:
            label_txt = f"{gene_symbol}"
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
        expand_points=(2,2),
        expand_text=(2,2),
        lim=200
    )
    
    for txt, (x, y) in zip(text_objects, zip(label_points_x, label_points_y)):
        x_text, y_text = txt.get_position()
        ax.plot([x_text, x], [y_text, y], color='black', lw=0.5, zorder=3, alpha=0.5)
    
    # Create a linear coordinate bar for each chromosome in the lower subplot (ax_bar)
    ax_bar.set_xlim(ax.get_xlim())
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Chromosome (Linear Coordinates)', fontsize=16)
    for c in unique_chroms:
        c_idx = chrom_to_index[c]
        # Draw a horizontal line representing the chromosome from its start to end in linear coords
        ax_bar.hlines(0.5, c_idx, c_idx+1, color='black', lw=2)
    ax_bar.set_xticks([i+0.5 for i in range(len(unique_chroms))])
    ax_bar.set_xticklabels([c.replace("chr", "") for c in unique_chroms], fontsize=14, fontweight='bold')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)

    # For each chromosome with an inversion, draw connection lines from the lower linear axis (true base-pair space)
    # to the main plot's zoomed-in inversion boundaries.
    for c in chr_trans_info:
        c_idx = chrom_to_index[c]
        A = chr_trans_info[c]['A']  # Normalized inversion start (linear coordinate)
        B = chr_trans_info[c]['B']  # Normalized inversion end (linear coordinate)
        fA = chr_trans_info[c]['fA']  # Transformed inversion start (main plot coordinate)
        fB = chr_trans_info[c]['fB']  # Transformed inversion end (main plot coordinate)
        lower_left = c_idx + A
        lower_right = c_idx + B
        main_left = c_idx + fA
        main_right = c_idx + fB
        con_left = ConnectionPatch(xyA=(lower_left, 0.5), xyB=(main_left, 0),
                                    coordsA="data", coordsB="data", axesA=ax_bar, axesB=ax,
                                    color="black", lw=0.5, linestyle="--")
        fig.add_artist(con_left)
        con_right = ConnectionPatch(xyA=(lower_right, 0.5), xyB=(main_right, 0),
                                     coordsA="data", coordsB="data", axesA=ax_bar, axesB=ax,
                                     color="black", lw=0.5, linestyle="--")
        fig.add_artist(con_right)

    
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
