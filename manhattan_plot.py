import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FixedFormatter
from adjustText import adjust_text
from pathlib import Path
import os
import math
from matplotlib.patches import ConnectionPatch

# Constants
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# Create necessary directories
for directory in [RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

# Hard-coded chromosome lengths
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
    """
    Creates a figure with:
      1) One row of subplots (side by side), one per chromosome. 
         Each top subplot x-range => [0..1], y-range => 0..(global max neg_log_p).
         We place data points, shading for inversions, boundary dots for left/right region. 
      2) A single axis at the bottom (one horizontal bar) representing the real, linear
         base-pair coordinates for *all chromosomes side by side.* 
         Then from each sub-subplot boundary to the corresponding bottom region, 
         we draw lines to connect the "zoomed" top view to the "real" coordinate space.
    """

    # Read main data
    results_df = pd.read_csv(data_file)
    valid_mask = (results_df['p_value'].notnull()) & (results_df['p_value'] > 0)
    if valid_mask.sum() == 0:
        print("No valid p-values found. Cannot create plot.")
        return

    # Bonferroni correction, -log10
    valid_pvals = results_df.loc[valid_mask, 'p_value']
    m = len(valid_pvals)
    results_df['bonferroni_p_value'] = np.nan
    results_df.loc[valid_mask, 'bonferroni_p_value'] = (
        results_df.loc[valid_mask, 'p_value'] * m
    ).clip(upper=1.0)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))

    # Read inversion data, filter by overlap
    inv_df = pd.read_csv(inv_file).dropna(subset=['chr','region_start','region_end'])
    def overlaps(inv_row, all_df):
        c = inv_row['chr']
        s = inv_row['region_start']
        e = inv_row['region_end']
        subset = all_df[
            (all_df['chrom'] == c)
            & (all_df['start'] <= e)
            & (all_df['end']   >= s)
            & (all_df['p_value'].notnull())
            & (all_df['p_value'] > 0)
        ]
        return len(subset) > 0
    inv_df = inv_df[inv_df.apply(lambda row: overlaps(row, results_df), axis=1)]

    # Sort chromosomes
    def chr_sort_key(ch):
        base = ch.replace('chr','')
        try:
            return (0, int(base))
        except:
            mapping = {'X':23,'Y':24,'M':25}
            return (1, mapping.get(base,99))
    unique_chroms = sorted(results_df['chrom'].dropna().unique(), key=chr_sort_key)
    if not unique_chroms:
        print("No valid chromosomes found. Exiting.")
        return

    # Determine c_min, c_max from valid data only
    chrom_ranges = {}
    for c in unique_chroms:
        cdata = results_df[(results_df['chrom'] == c) & valid_mask]
        if cdata.empty:
            chrom_ranges[c] = (0, 1)  # fallback when no valid data is present
        else:
            c_min = cdata['start'].min()
            c_max = cdata['end'].max()
            chrom_ranges[c] = (c_min, c_max)

    # We'll unify the y-limits for all subplots by a global max of neg_log_p
    global_max_neglogp = results_df['neg_log_p'].max()
    if not math.isfinite(global_max_neglogp):
        global_max_neglogp = 1.0
    YLIM_TOP = global_max_neglogp * 1.1

    # Color map for effect sizes
    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['blue','gray','red'])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)

    # We'll layout: 2 rows. 
    #   Row 1: ncols = len(unique_chroms) => subplots for each chromosome
    #   Row 2: single axis spanning entire width => real linear coords bar
    n_chroms = len(unique_chroms)
    fig_h = max(6, 3.0)  # minimum height
    fig_w = max(16, 3*n_chroms)  # width scales with n_chroms
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    # We'll define a top row for the subplots, a bottom row for the single axis
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=2, ncols=n_chroms, height_ratios=[5,1], figure=fig)

    # We'll create a subplot for each chromosome in row=0, col i => the top "zoomed" axis
    ax_subplots = []
    for i, c in enumerate(unique_chroms):
        ax_i = fig.add_subplot(gs[0, i])
        ax_subplots.append(ax_i)

    # The single bottom axis spans row=1, all columns
    ax_bottom = fig.add_subplot(gs[1, :])

    # *** Build the bottom axis for the real, linear coords ***
    # We'll place each chromosome side by side, offset-based
    # Then we can show how each region maps from the top subplot to the bottom bar

    # Offsets for each chromosome
    offsets = {}
    cum_len = 0
    for c in unique_chroms:
        real_len = CHR_LENGTHS.get(c, 1)
        offsets[c] = cum_len
        cum_len += real_len

    # The bottom axis from x=0..cum_len
    ax_bottom.set_xlim(0, cum_len)
    ax_bottom.set_ylim(0, 1)
    # We'll draw each chromosome's segment from offsets[c].. offsets[c]+chr_len at y=0.5
    # plus label
    for spine in ['top','left','right','bottom']:
        ax_bottom.spines[spine].set_visible(False)
    ax_bottom.set_yticks([])
    ax_bottom.set_xlabel("Chromosomes: Real Base-Pair Coordinates", fontsize=13)
    tick_positions = []
    for c in unique_chroms:
        tick_positions.append(offsets[c])
        tick_positions.append(offsets[c] + CHR_LENGTHS.get(c, 1))
    tick_positions = sorted(set(tick_positions))
    ax_bottom.set_xticks(tick_positions)
    ax_bottom.set_xticklabels([])

    # Draw a line & label for each chromosome
    for c in unique_chroms:
        real_len = CHR_LENGTHS.get(c,1)
        off = offsets[c]
        left_bar = off
        right_bar= off+real_len
        ax_bottom.hlines(0.5, left_bar, right_bar, color='black', lw=2)
        # place label near the center
        mid_x = (left_bar + right_bar)*0.5
        ax_bottom.text(mid_x, 0.45, c.replace("chr",""), ha='center', va='top', fontsize=9, fontweight='bold')

    # *** Now fill each chromosome top subplot with data ***
    # Then connect boundary dots to the bottom axis
    from matplotlib import patches as mpatches
    recurrent_color = 'purple'
    single_color    = 'green'

    for i, c in enumerate(unique_chroms):
        ax_top = ax_subplots[i]
        ax_top.set_xlim(-0.05,1.05)
        ax_top.set_ylim(0,YLIM_TOP)
        if i == 0:
            ax_top.set_ylabel("-log10(p)", fontsize=9)
        else:
            ax_top.set_yticks([])
            ax_top.set_ylabel("")
        # remove unneeded spines
        ax_top.yaxis.grid(True, which='major', color='lightgray', linestyle='--', lw=0.5)
        ax_top.xaxis.grid(False)
        for spine in ['top','right','left']:
            ax_top.spines[spine].set_visible(False)
        ax_top.set_xticks([])
        # gather data for c
        cdata = results_df[(results_df['chrom']==c) & valid_mask].copy()
        if cdata.empty:
            ax_top.text(0.5, 0.5*YLIM_TOP, "No data", ha='center', va='center', fontsize=9)
            continue
        c_min_data, c_max_data = chrom_ranges[c]
        # Build the x for each row => normalized
        def normpos(x):
            if c_max_data>c_min_data:
                return max(0, min(1,(x-c_min_data)/(c_max_data-c_min_data)))
            else:
                return 0.5
        cdata['chr_plot_x'] = cdata['start'].apply(normpos)

        # scatter
        eff_this = cdata['observed_effect_size']
        eff_mean = eff_this.mean()
        eff_std  = eff_this.std() if math.isfinite(eff_this.std()) and eff_this.std()>0 else 1
        eff_zchr = np.clip((eff_this - eff_mean)/eff_std, -1,1)

        ax_top.scatter(
            cdata['chr_plot_x'],
            cdata['neg_log_p'],
            c=eff_zchr,
            cmap=cmap,
            norm=norm,
            s=40, alpha=0.7, linewidth=0, zorder=2
        )

        # highlight inversions
        invsub = inv_df[inv_df['chr']==c]
        for _, invrow in invsub.iterrows():
            inv_start = invrow['region_start']
            inv_end   = invrow['region_end']
            inv_size  = inv_end - inv_start
            if inv_size<=0: continue
            if c_max_data>c_min_data:
                left_rel = (inv_start - c_min_data)/(c_max_data - c_min_data)
                right_rel= (inv_end   - c_min_data)/(c_max_data - c_min_data)
            else:
                left_rel,right_rel=0.4,0.6
            left_rel= max(0, min(1,left_rel))
            right_rel=max(0, min(1,right_rel))
            t = invrow.get('0_single_1_recur',0)
            inv_color = recurrent_color if t==1 else single_color
            if inv_size>0.5*(c_max_data-c_min_data):
                ax_top.axvspan(left_rel, right_rel, color=inv_color, alpha=0.1, hatch='//', zorder=0)
            else:
                ax_top.axvspan(left_rel, right_rel, color=inv_color, alpha=0.2, zorder=0)

        # boundary
        used_min = cdata['start'].min()
        used_max = cdata['end'].max()
        if not math.isfinite(used_min) or not math.isfinite(used_max):
            used_min,used_max=0,0
        left_rel = normpos(used_min)
        right_rel= normpos(used_max)
        # top boundary dots
        ax_top.axvspan(left_rel, right_rel, facecolor='white', alpha=0.2, zorder=1)

        # connect lines to the bottom axis
        # bottom coords => offset + used_min.. offset+ used_max
        off = offsets.get(c,0)
        real_len = CHR_LENGTHS.get(c,1)
        used_min_clamp = max(0,min(real_len, used_min))
        used_max_clamp = max(0,min(real_len, used_max))
        bottom_left_x  = off+used_min_clamp
        bottom_right_x = off+used_max_clamp

        # connect
        # Draw the red dashed connection lines.
        con_left = ConnectionPatch(
            xyA=(bottom_left_x, 0.5),
            xyB=(left_rel, 0),
            coordsA="data", coordsB="data",
            axesA=ax_bottom, axesB=ax_top,
            color="red", lw=1, linestyle="--", zorder=10
        )
        con_right = ConnectionPatch(
            xyA=(bottom_right_x, 0.5),
            xyB=(right_rel, 0),
            coordsA="data", coordsB="data",
            axesA=ax_bottom, axesB=ax_top,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_left)
        fig.add_artist(con_right)
        # Draw multiple solid gray connection lines in between the red dashed connection lines.
        n_lines = 20
        for i in range(1, n_lines):
            f = i / n_lines
            top_x = left_rel + f * (right_rel - left_rel)
            bottom_x = bottom_left_x + f * (bottom_right_x - bottom_left_x)
            con_mid = ConnectionPatch(
                xyA=(bottom_x, 0.5),
                xyB=(top_x, 0),
                coordsA="data", coordsB="data",
                axesA=ax_bottom, axesB=ax_top,
                color="gray", lw=3, linestyle="-", zorder=9
            )
            fig.add_artist(con_mid)

    # finalize
    plt.tight_layout()
    out_fname = os.path.join(PLOTS_DIR, "manhattan_plot_subplots.png")
    plt.savefig(out_fname, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    data_file = RESULTS_DIR / 'manhattan_plot_data.csv'
    inv_file = 'inv_info.csv'
    create_manhattan_plot(data_file, inv_file=inv_file, top_hits_to_annotate=10)

if __name__ == "__main__":
    main()
