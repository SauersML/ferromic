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
    # Read the main data
    results_df = pd.read_csv(data_file)
    
    # Filter to valid p-values
    valid_mask = results_df['p_value'].notnull() & (results_df['p_value'] > 0)
    if valid_mask.sum() == 0:
        print("No valid p-values found. Cannot create plot.")
        return

    # Compute number of valid p-values, apply Bonferroni, compute -log10(p)
    valid_pvals = results_df.loc[valid_mask, 'p_value']
    m = len(valid_pvals)
    results_df['bonferroni_p_value'] = np.nan
    results_df.loc[valid_mask, 'bonferroni_p_value'] = (
        results_df.loc[valid_mask, 'p_value'] * m
    ).clip(upper=1.0)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))
    
    # Read inversion file (and filter to rows overlapping valid data)
    inv_df = pd.read_csv(inv_file).dropna(subset=['chr','region_start','region_end'])
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
    
    # Sort chromosomes in natural order
    def chr_sort_key(ch):
        base = ch.replace('chr','')
        try:
            return (0, int(base))
        except:
            mapping = {'X': 23, 'Y': 24, 'M': 25}
            return (1, mapping.get(base, 99))

    unique_chroms = sorted(results_df['chrom'].dropna().unique(), key=chr_sort_key)
    chrom_to_index = {c: i for i,c in enumerate(unique_chroms)}

    # For each chromosome, find min and max start or end
    chrom_ranges = {}
    for c in unique_chroms:
        subset = results_df[results_df['chrom']==c]
        c_min = subset['start'].min()
        c_max = subset['end'].max()
        chrom_ranges[c] = (c_min, c_max)
    
    # The top plot's x-axis: we place each chromosome in [i, i+1].
    # We'll compute x = i + (start-c_min)/(c_max-c_min) if c_max>c_min, else 0.5
    xs = []
    for _,row in results_df.iterrows():
        c = row['chrom']
        if c not in chrom_to_index:
            xs.append(np.nan)
            continue
        c_idx = chrom_to_index[c]
        c_min, c_max = chrom_ranges[c]
        if c_max>c_min:
            rel_pos = (row['start'] - c_min)/(c_max - c_min)
        else:
            rel_pos = 0.5
        xs.append(c_idx + rel_pos)
    results_df['plot_x'] = xs
    
    # Z-scoring effect sizes for coloring
    eff = results_df['observed_effect_size']
    mean_eff = eff.mean()
    std_eff = eff.std()
    if std_eff==0 or np.isnan(std_eff):
        std_eff=1.0
    eff_z = (eff - mean_eff)/std_eff
    eff_z = np.clip(eff_z, -1, 1)

    # A custom diverging colormap
    cmap = LinearSegmentedColormap.from_list('custom_diverging',['blue','gray','red'])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)

    # Set up figure with top (manhattan) and bottom (linear coords) subplots
    fig = plt.figure(figsize=(20,10))
    gs = fig.add_gridspec(2,1,height_ratios=[9,1],hspace=0.05)
    sns.set_style("ticks")
    ax = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    
    # We'll highlight inversion regions on the top axis
    recurrent_color = 'purple'
    single_color = 'green'
    
    for _,inv_row in inv_df.iterrows():
        inv_chr = inv_row['chr']
        if inv_chr not in chrom_to_index:
            continue
        c_idx = chrom_to_index[inv_chr]
        c_min, c_max = chrom_ranges[inv_chr]
        chrom_size = c_max - c_min if c_max>c_min else 1
        inv_size = inv_row['region_end'] - inv_row['region_start']
        if chrom_size<=0:
            continue
        # relative positions
        rel_start = max(0,min(1,(inv_row['region_start']-c_min)/chrom_size))
        rel_end   = max(0,min(1,(inv_row['region_end']  -c_min)/chrom_size))
        inv_x_start = c_idx + rel_start
        inv_x_end   = c_idx + rel_end
        inv_color = recurrent_color if inv_row.get('0_single_1_recur',0)==1 else single_color
        if inv_size > 0.5*chrom_size:
            # large inversion
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.1, hatch='//', zorder=0)
        else:
            # smaller inversion
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.2, zorder=0)

    import matplotlib.patches as mpatches
    recurrent_patch = mpatches.Patch(color=recurrent_color, alpha=0.2, label='Recurrent inversion')
    single_patch = mpatches.Patch(color=single_color, alpha=0.2, label='Single-event inversion')
    recurrent_large_patch = mpatches.Patch(
        facecolor=recurrent_color, hatch='//', edgecolor='black', alpha=0.1, label='Large recurrent inversion'
    )
    single_large_patch = mpatches.Patch(
        facecolor=single_color, hatch='//', edgecolor='black', alpha=0.1, label='Large single-event inversion'
    )
    
    ax.legend(
        handles=[recurrent_patch, single_patch, recurrent_large_patch, single_large_patch],
        loc='upper left', fontsize=14, frameon=True
    )
    
    # Scatter the main data
    scatter = ax.scatter(
        results_df['plot_x'], results_df['neg_log_p'],
        c=eff_z, cmap=cmap, norm=norm,
        s=50, alpha=0.7, linewidth=0, zorder=2
    )
    
    # Add colorbar
    cb = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Z-scored Effect Size', fontsize=16)
    cb.ax.tick_params(labelsize=14)
    cb.ax.text(0.0,1.05,'Higher dN/dS\nfor inverted',transform=cb.ax.transAxes,ha='left',va='bottom',fontsize=10)
    cb.ax.text(0.0,-0.05,'Higher dN/dS\nfor non-inverted',transform=cb.ax.transAxes,ha='left',va='top',fontsize=10)
    
    # p=0.05 lines
    sig_line = -np.log10(0.05)
    ax.axhline(sig_line, color='red', linestyle='--', linewidth=2, zorder=3, label='p=0.05')
    if m>0:
        bonf_line = -np.log10(0.05/m)
        if np.isfinite(bonf_line) and bonf_line>0:
            ax.axhline(bonf_line, color='darkred', linestyle='--', linewidth=2, zorder=3, label='p=0.05 (Bonferroni)')
    
    # adjust y-limits
    cur_ylim = ax.get_ylim()
    new_ylim = max(cur_ylim[1], sig_line+1)
    if m>0 and np.isfinite(bonf_line):
        new_ylim = max(new_ylim, bonf_line+1)
    ax.set_ylim(0, new_ylim)
    
    # turn off top axis x tick labels
    ax.tick_params(labelbottom=False)
    ax.set_ylabel('-log10(p-value)', fontsize=18)
    ax.set_title('Manhattan Plot of CDS Significance', fontsize=24, fontweight='bold', pad=20)

    handles_auto, labels_auto = ax.get_legend_handles_labels()
    m_handles = [recurrent_patch, single_patch, recurrent_large_patch, single_large_patch]
    m_labels = [
        "Recurrent inversion",
        "Single-event inversion",
        "Large recurrent inversion",
        "Large single-event inversion"
    ]
    all_handles = m_handles + handles_auto
    all_labels = m_labels + labels_auto
    ax.legend(all_handles, all_labels, fontsize=14, frameon=True, loc='upper right')

    
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--', lw=0.5)
    ax.xaxis.grid(False)
    
    # Optionally annotate top hits
    sig_hits = results_df[valid_mask].sort_values('p_value').head(top_hits_to_annotate)
    text_objects = []
    label_points_x=[]
    label_points_y=[]
    for _,hit_row in sig_hits.iterrows():
        gsym = hit_row.get('gene_symbol',None)
        if gsym and gsym not in [None,'Unknown']:
            txt = ax.text(
                hit_row['plot_x'], hit_row['neg_log_p']+1.0, gsym,
                fontsize=12,ha='center',va='bottom',color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )
            text_objects.append(txt)
            label_points_x.append(hit_row['plot_x'])
            label_points_y.append(hit_row['neg_log_p'])
    adjust_text(
        text_objects,x=label_points_x,y=label_points_y,ax=ax,
        force_text=2.0,force_points=2.0,
        expand_points=(2,2),expand_text=(2,2),
        lim=200
    )
    for txt, (lx,ly) in zip(text_objects, zip(label_points_x,label_points_y)):
        xx,yy = txt.get_position()
        ax.plot([xx,lx],[yy,ly], color='black', lw=0.5, zorder=3, alpha=0.5)
    
    # ------------------------------------------------------------------
    # BOTTOM AXIS: Each chromosome gets its own space [i, i+1], 
    # but we label them with their actual c_min, c_max in basepairs.
    # We'll also connect the top edges down to these linear coords.
    # ------------------------------------------------------------------
    ax_bar.set_xlim(ax.get_xlim())  # same overall x-limits as top
    ax_bar.set_ylim(0,1)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Chromosome (Linear Coordinates)', fontsize=16)
    
    # For each chromosome c in [i,i+1], we:
    #   (1) draw black line from x=i to x=i+1 at y=0.5
    #   (2) place two text labels for c_min, c_max at x=i and x=i+1
    #   (3) place two red dots at x=i, x=i+1
    #   (4) connect top-left -> bottom-left, top-right->bottom-right
    
    for c in unique_chroms:
        i = chrom_to_index[c]
        c_min, c_max = chrom_ranges[c]
        
        # Draw the horizontal line at y=0.5
        ax_bar.hlines(0.5, i, i+1, color='black', lw=2)
        
        # Place text labels at x=i, i+1 for the real c_min, c_max
        ax_bar.text(i,   0.6, f"{c_min}", ha='left', va='bottom', fontsize=10, rotation=45)
        ax_bar.text(i+1, 0.6, f"{c_max}", ha='right', va='bottom', fontsize=10, rotation=45)
        
        # Red dots at the ends
        ax_bar.scatter([i, i+1],[0.5,0.5], color='red', zorder=5)
        
        # Now connect the top edges
        top_left  = (i,   0)   # in top axis coords
        top_right = (i+1, 0)
        bottom_left  = (i,   0.5) # in bottom axis coords
        bottom_right = (i+1, 0.5)
        
        # Plot red dots on the top axis
        ax.scatter([top_left[0], top_right[0]],[top_left[1], top_right[1]], color='red', zorder=5)
        
        # connect left boundary
        con_left = ConnectionPatch(
            xyA=bottom_left, xyB=top_left,
            coordsA="data", coordsB="data",
            axesA=ax_bar, axesB=ax,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_left)
        
        # connect right boundary
        con_right = ConnectionPatch(
            xyA=bottom_right, xyB=top_right,
            coordsA="data", coordsB="data",
            axesA=ax_bar, axesB=ax,
            color="red", lw=1, linestyle="--", zorder=10
        )
        fig.add_artist(con_right)
    
    # finalize layout
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
