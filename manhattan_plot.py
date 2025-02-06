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

# Hard-coded chromosome lengths (e.g., hg38). We will lay out a single horizontal line at y=0.5 for all chromosomes,
# subdivided by offsets. We do not label the basepair coordinates themselves, but we do reflect them in the positioning
# of the boundary dots and the total segment length for each chromosome.
CHR_LENGTHS = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895
}

def create_manhattan_plot(data_file, inv_file='inv_info.csv', top_hits_to_annotate=10):
    # Read the main data
    results_df = pd.read_csv(data_file)
    
    # Filter valid p-values
    valid_mask = (results_df['p_value'].notnull()) & (results_df['p_value'] > 0)
    if valid_mask.sum() == 0:
        print("No valid p-values found. Cannot create plot.")
        return

    # Bonferroni correction, -log10(p)
    valid_pvals = results_df.loc[valid_mask, 'p_value']
    m = len(valid_pvals)
    results_df['bonferroni_p_value'] = np.nan
    results_df.loc[valid_mask, 'bonferroni_p_value'] = (
        results_df.loc[valid_mask, 'p_value'] * m
    ).clip(upper=1.0)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))

    # Read inversions, filter by overlap
    inv_df = pd.read_csv(inv_file).dropna(subset=['chr','region_start','region_end'])
    
    def overlaps(inv_row, all_df):
        c = inv_row['chr']
        s = inv_row['region_start']
        e = inv_row['region_end']
        subset = all_df[
            (all_df['chrom'] == c)
            & (all_df['start'] <= e)
            & (all_df['end'] >= s)
            & (all_df['p_value'].notnull())
            & (all_df['p_value'] > 0)
        ]
        return len(subset) > 0
    inv_df = inv_df[inv_df.apply(lambda row: overlaps(row, results_df), axis=1)]
    
    # Sort chromosomes in natural order
    def chr_sort_key(ch):
        base = ch.replace('chr','')
        try:
            return (0, int(base))
        except:
            mapping = {'X':23,'Y':24,'M':25}
            return (1, mapping.get(base,99))

    unique_chroms = sorted(results_df['chrom'].dropna().unique(), key=chr_sort_key)
    chrom_to_index = {c: i for i, c in enumerate(unique_chroms)}

    # For each chromosome, find the minimal and maximal coordinate used in the data
    chrom_ranges = {}
    for c in unique_chroms:
        subset = results_df[results_df['chrom'] == c]
        c_min = subset['start'].min()
        c_max = subset['end'].max()
        chrom_ranges[c] = (c_min, c_max)

    # Build the top-axis x-values: each chromosome is [i, i+1]
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
    
    # Z-score effect sizes
    eff = results_df['observed_effect_size']
    eff_mean = eff.mean()
    eff_std = eff.std()
    if eff_std == 0 or np.isnan(eff_std):
        eff_std = 1.0
    eff_z = np.clip((eff - eff_mean) / eff_std, -1, 1)

    # Colormap
    cmap = LinearSegmentedColormap.from_list('custom_diverging',['blue','gray','red'])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20,10))
    gs = fig.add_gridspec(2,1, height_ratios=[9,1], hspace=0.05)
    sns.set_style("ticks")
    ax = fig.add_subplot(gs[0])     # top axis
    ax_bar = fig.add_subplot(gs[1]) # bottom axis

    # Colors for inversions
    recurrent_color = 'purple'
    single_color    = 'green'

    # Shade inversion regions on the top axis
    for _, inv_row in inv_df.iterrows():
        c = inv_row['chr']
        if c not in chrom_to_index:
            continue
        i = chrom_to_index[c]
        c_min, c_max = chrom_ranges[c]
        inv_size = inv_row['region_end'] - inv_row['region_start']
        chrom_size = (c_max - c_min) if c_max>c_min else 1
        if chrom_size <= 0:
            continue
        rel_start = (inv_row['region_start'] - c_min)/chrom_size
        rel_end   = (inv_row['region_end']   - c_min)/chrom_size
        rel_start = max(0, min(1, rel_start))
        rel_end   = max(0, min(1, rel_end))
        left_x  = i + rel_start
        right_x = i + rel_end
        t = inv_row.get('0_single_1_recur', 0)
        inv_color = recurrent_color if t==1 else single_color
        
        # Large vs. smaller inversion
        if inv_size > 0.5*chrom_size:
            ax.axvspan(left_x, right_x, color=inv_color, alpha=0.1, hatch='//', zorder=0)
        else:
            ax.axvspan(left_x, right_x, color=inv_color, alpha=0.2, zorder=0)

    import matplotlib.patches as mpatches
    recurrent_patch = mpatches.Patch(color=recurrent_color, alpha=0.2)
    single_patch    = mpatches.Patch(color=single_color, alpha=0.2)
    recurrent_large_patch = mpatches.Patch(facecolor=recurrent_color, hatch='//', edgecolor='black', alpha=0.1)
    single_large_patch    = mpatches.Patch(facecolor=single_color,  hatch='//', edgecolor='black', alpha=0.1)

    # Temporary legend for inversions
    ax.legend(
        handles=[recurrent_patch, single_patch, recurrent_large_patch, single_large_patch],
        loc='upper left', fontsize=14, frameon=True
    )
    
    # Scatter main data
    sc = ax.scatter(
        results_df['plot_x'],
        results_df['neg_log_p'],
        c=eff_z, cmap=cmap, norm=norm,
        s=50, alpha=0.7, linewidth=0, zorder=2
    )
    
    # Colorbar
    cb = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Z-scored Effect Size', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    cb.ax.text(0.0, 1.05, 'Higher dN/dS\nfor inverted', transform=cb.ax.transAxes,
               ha='left', va='bottom', fontsize=10)
    cb.ax.text(0.0, -0.05, 'Higher dN/dS\nfor non-inverted', transform=cb.ax.transAxes,
               ha='left', va='top', fontsize=10)
    
    # Significance lines
    sig_line = -np.log10(0.05)
    ax.axhline(sig_line, color='red', linestyle='--', linewidth=2, zorder=3, label='p=0.05')
    if m>0:
        bonf_line = -np.log10(0.05 / m)
        if np.isfinite(bonf_line) and bonf_line>0:
            ax.axhline(bonf_line, color='darkred', linestyle='--', linewidth=2, zorder=3,
                       label='p=0.05 (Bonferroni)')
    
    # Adjust y-limits
    ylo, yhi = ax.get_ylim()
    new_yhi = max(yhi, sig_line+1)
    if m>0 and np.isfinite(bonf_line):
        new_yhi = max(new_yhi, bonf_line+1)
    ax.set_ylim(0, new_yhi)

    # Turn off x tick labels on top axis
    ax.tick_params(labelbottom=False)
    ax.set_ylabel('-log10(p-value)', fontsize=18)
    ax.set_title('Manhattan Plot of CDS Significance', fontsize=24, fontweight='bold', pad=20)
    
    # Combine legends
    handles_auto, labels_auto = ax.get_legend_handles_labels()
    my_handles = [recurrent_patch, single_patch, recurrent_large_patch, single_large_patch]
    my_labels  = [
        "Recurrent inversion",
        "Single-event inversion",
        "Large recurrent inversion",
        "Large single-event inversion"
    ]
    all_handles = my_handles + handles_auto
    all_labels  = my_labels  + labels_auto
    ax.legend(all_handles, all_labels, fontsize=14, frameon=True, loc='upper right')
    
    # Grid on y-axis only
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--', lw=0.5)
    ax.xaxis.grid(False)
    
    # Annotate top hits
    top_hits = results_df[valid_mask].sort_values('p_value').head(top_hits_to_annotate)
    text_objs = []
    text_xs   = []
    text_ys   = []
    for _, row in top_hits.iterrows():
        gsym = row.get('gene_symbol', None)
        if gsym and gsym not in [None,'Unknown']:
            txt = ax.text(
                row['plot_x'],
                row['neg_log_p'] + 1.0,
                gsym,
                fontsize=12,
                ha='center',
                va='bottom',
                color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )
            text_objs.append(txt)
            text_xs.append(row['plot_x'])
            text_ys.append(row['neg_log_p'])
    
    adjust_text(
        text_objs,
        x=text_xs, y=text_ys,
        ax=ax,
        force_text=2.0, force_points=2.0,
        expand_points=(2,2),
        expand_text=(2,2),
        lim=200
    )
    for txt,(xx,yy) in zip(text_objs, zip(text_xs,text_ys)):
        xt,yt = txt.get_position()
        ax.plot([xt, xx],[yt, yy], color='black', lw=0.5, zorder=3, alpha=0.5)
    
    # ---------------------------------------
    # BOTTOM AXIS: Single horizontal line at y=0.5
    # subdivided for each chromosome in proportion to their real lengths.
    # We'll place the chromosomes side by side, from left to right, in sorted order,
    # with no stacking. We do not label real base pairs, just the chromosome name.
    # We'll place exactly two lines (four red dots) for each chromosome:
    # top-left -> bottom-left, top-right -> bottom-right
    # bottom-left, bottom-right are at the actual portion of the single horizontal line
    # corresponding to c_min_used.. c_max_used in real base pairs, scaled by total length, etc.
    # But we do not show base pair coords, just the chromosome label in the center of its segment.
    # ---------------------------------------
    
    # Clear spines, y ticks
    for spn in ['top','left','right','bottom']:
        ax_bar.spines[spn].set_visible(False)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Chromosomes (Real lengths, single horizontal line)", fontsize=16)
    
    # We'll compute offsets so that chromosome i starts exactly where chromosome i-1 ended,
    # each scaled by the known CHR_LENGTHS. Then, for c we have offset_c.. offset_c + CHR_LENGTHS[c].
    # We'll store them in a dictionary.
    offsets = {}
    cum_len = 0
    for c in unique_chroms:
        # fallback length if not present
        real_len = CHR_LENGTHS.get(c, 1)
        offsets[c] = cum_len
        cum_len += real_len
    
    # The bottom axis from x=0..x=cum_len
    ax_bar.set_xlim(0, cum_len)
    ax_bar.set_ylim(0, 1)
    
    # We'll place for each chromosome a horizontal line from offset.. offset+chr_len at y=0.5
    # plus a text label for the chromosome near the center. Then, we place 2 boundary dots for the region used
    # and connect them to top axis boundaries.
    for c in unique_chroms:
        real_len = CHR_LENGTHS.get(c, 1)
        off = offsets[c]
        left_x = off
        right_x= off + real_len
        
        # horizontal line for that chromosome's segment
        ax_bar.hlines(0.5, left_x, right_x, color='black', lw=2)
        
        # place the chromosome label in the center
        mid_x = (left_x + right_x)*0.5
        ax_bar.text(mid_x, 0.55, c.replace("chr",""), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # gather the minimal used region from the data c_min_used.. c_max_used
        cdata = results_df[results_df['chrom'] == c]
        if cdata.empty:
            continue
        used_min = cdata['start'].min()
        used_max = cdata['end'].max()
        
        # clamp if beyond real length
        # but we just do offset + used_min for the scaled position, and offset+ used_max
        # actually, we want proportion of used_min, used_max within the entire chromosome length
        # We'll define bottom_left = off + used_min, etc. This is if we treat the entire region as 0.. real_len in base pairs
        # but used_min might be bigger than real_len for some reason, clamp if needed
        actual_len = real_len
        # ensure used_min.. used_max are within [0..actual_len]
        used_min_clamped = max(0, min(actual_len, used_min))
        used_max_clamped = max(0, min(actual_len, used_max))
        
        bottom_left  = off + used_min_clamped
        bottom_right = off + used_max_clamped
        
        # place 2 red dots
        ax_bar.scatter([bottom_left, bottom_right], [0.5, 0.5], color='red', zorder=5)
        
        # top axis boundaries
        i = chrom_to_index[c]
        c_min_data, c_max_data = chrom_ranges[c]
        if c_max_data > c_min_data:
            left_norm  = (used_min_clamped - c_min_data)/(c_max_data - c_min_data)
            right_norm = (used_max_clamped - c_min_data)/(c_max_data - c_min_data)
            left_norm  = max(0, min(1,left_norm))
            right_norm = max(0, min(1,right_norm))
            top_left_x  = i + left_norm
            top_right_x = i + right_norm
        else:
            top_left_x  = i + 0.4
            top_right_x = i + 0.6
        
        # place top boundary red dots
        ax.scatter([top_left_x, top_right_x], [0,0], color='red', zorder=5)
        
        # connect lines
        con_left = ConnectionPatch(
            xyA=(bottom_left, 0.5), xyB=(top_left_x, 0),
            coordsA="data", coordsB="data",
            axesA=ax_bar, axesB=ax,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_left)
        
        con_right = ConnectionPatch(
            xyA=(bottom_right, 0.5), xyB=(top_right_x, 0),
            coordsA="data", coordsB="data",
            axesA=ax_bar, axesB=ax,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_right)
    
    # finalize
    plt.tight_layout()
    plt.savefig(os.path.join("plots","manhattan_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    data_file = RESULTS_DIR / 'manhattan_plot_data.csv'
    inv_file = 'inv_info.csv'
    create_manhattan_plot(data_file, inv_file=inv_file, top_hits_to_annotate=10)

if __name__=="__main__":
    main()
