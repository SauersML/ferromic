import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter
from pathlib import Path
import os
import math
import re
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

for directory in [RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

CHR_LENGTHS = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895
}

def parse_coords(coord_str):
    if pd.isna(coord_str):
        return np.nan, np.nan
    first_segment = coord_str.split(';')[0]
    match = re.match(r'chr.*?:(\d+)-(\d+)', first_segment)
    if match:
        try:
            start = int(match.group(1))
            end = int(match.group(2))
            return start, end
        except ValueError:
            return np.nan, np.nan
    else:
        return np.nan, np.nan

def create_manhattan_plot(data_file, inv_file='inv_info.csv'):
    print(f"Reading main data from: {data_file}")
    try:
        results_df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: Input data file not found at {data_file}")
        return
    except Exception as e:
        print(f"Error reading {data_file}: {e}")
        return

    print("Parsing coordinates and renaming columns...")
    coords_parsed = results_df['coordinates'].apply(parse_coords)
    results_df['start'] = coords_parsed.apply(lambda x: x[0])
    results_df['end'] = coords_parsed.apply(lambda x: x[1])

    results_df.rename(columns={'chromosome': 'chrom', 'effect_size': 'observed_effect_size'}, inplace=True)

    original_rows = len(results_df)
    results_df.dropna(subset=['start', 'end', 'chrom'], inplace=True)
    rows_dropped = original_rows - len(results_df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing/unparseable coordinates or chromosome.")

    if results_df.empty:
        print("No valid data rows remain after parsing coordinates. Cannot create plot.")
        return

    results_df['start'] = results_df['start'].astype(int)
    results_df['end'] = results_df['end'].astype(int)
    results_df['chrom'] = results_df['chrom'].astype(str)
    results_df['chrom'] = results_df['chrom'].apply(lambda x: x if x.startswith('chr') else 'chr' + x)

    print("Calculating p-value metrics...")
    valid_mask = (results_df['p_value'].notnull()) & (results_df['p_value'] > 0) & (results_df['start'].notnull()) & (results_df['end'].notnull())
    if valid_mask.sum() == 0:
        print("No valid p-values with associated coordinates found. Cannot create plot.")
        return

    valid_pvals = results_df.loc[valid_mask, 'p_value']
    m = len(valid_pvals)
    results_df['bonferroni_p_value'] = np.nan
    results_df.loc[valid_mask, 'bonferroni_p_value'] = (
        results_df.loc[valid_mask, 'p_value'] * m
    ).clip(upper=1.0)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))

    print("Calculating Z-scores for effect size...")
    effect_sizes = results_df.loc[valid_mask, 'observed_effect_size'].dropna()
    if not effect_sizes.empty:
        mean_effect = effect_sizes.mean()
        std_effect = effect_sizes.std()
        if std_effect > 0:
             results_df['effect_zscore'] = (results_df['observed_effect_size'] - mean_effect) / std_effect
        else:
             results_df['effect_zscore'] = 0.0
             print("Warning: Standard deviation of effect size is zero. Z-scores set to 0.")
    else:
        results_df['effect_zscore'] = np.nan
        print("Warning: No valid effect sizes to calculate Z-scores.")

    print("Filtering chromosomes based on valid data...")
    chroms_with_valid_data = results_df.loc[valid_mask, 'chrom'].unique()
    if len(chroms_with_valid_data) == 0:
        print("No chromosomes have valid data points. Cannot create plot.")
        return

    def chr_sort_key(ch):
        base = ch.replace('chr','')
        try:
            return (0, int(base))
        except ValueError:
            mapping = {'X':23,'Y':24,'M':25}
            return (1, mapping.get(base,99))

    unique_chroms = sorted(chroms_with_valid_data, key=chr_sort_key)
    print(f"Chromosomes to plot (having valid data): {unique_chroms}")

    results_df = results_df[results_df['chrom'].isin(unique_chroms)].copy()
    valid_mask = (results_df['p_value'].notnull()) & (results_df['p_value'] > 0) & (results_df['start'].notnull()) & (results_df['end'].notnull())

    print("Reading and filtering inversion data...")
    inv_raw_cols = ['Chromosome', 'Start', 'End']
    inv_recur_col = '0_single_1_recur'
    inv_internal_cols = ['chr', 'region_start', 'region_end']
    inv_df = pd.DataFrame(columns=inv_internal_cols + [inv_recur_col])

    try:
        inv_df_raw = pd.read_csv(inv_file)

        if all(col in inv_df_raw.columns for col in inv_raw_cols):
            inv_df = inv_df_raw.rename(columns={
                'Chromosome': 'chr',
                'Start': 'region_start',
                'End': 'region_end'
            })
            inv_df.dropna(subset=inv_internal_cols, inplace=True)

            if not inv_df.empty:
                inv_df['chr'] = inv_df['chr'].astype(str).apply(lambda x: x if x.startswith('chr') else 'chr' + x)
                inv_df = inv_df[inv_df['chr'].isin(unique_chroms)].copy()

                if not inv_df.empty:
                    inv_df['region_start'] = pd.to_numeric(inv_df['region_start'], errors='coerce')
                    inv_df['region_end'] = pd.to_numeric(inv_df['region_end'], errors='coerce')
                    inv_df.dropna(subset=['region_start','region_end'], inplace=True)

                    if not inv_df.empty:
                        inv_df['region_start'] = inv_df['region_start'].astype(int)
                        inv_df['region_end'] = inv_df['region_end'].astype(int)
                        # Handle potential missing recurrence column gracefully
                        if inv_recur_col not in inv_df.columns:
                            print(f"Warning: Inversion file missing recurrence column '{inv_recur_col}'. Defaulting to 0 (single).")
                            inv_df[inv_recur_col] = 0
                        else:
                            inv_df[inv_recur_col] = pd.to_numeric(inv_df[inv_recur_col], errors='coerce').fillna(0).astype(int)

                        print(f"Found {len(inv_df)} inversions on plotted chromosomes after coordinate validation.")
                    else:
                        print("Inversion data invalid after coordinate conversion.")
                        inv_df = pd.DataFrame(columns=inv_internal_cols + [inv_recur_col])
                else:
                    print("No inversions found on the chromosomes being plotted.")
                    inv_df = pd.DataFrame(columns=inv_internal_cols + [inv_recur_col])

            else:
                print("Inversion data empty after initial NaN drop.")
                inv_df = pd.DataFrame(columns=inv_internal_cols + [inv_recur_col])

        else:
            missing_cols = [col for col in inv_raw_cols if col not in inv_df_raw.columns]
            print(f"Warning: Inversion info file '{inv_file}' is missing expected columns: {missing_cols}. Proceeding without inversion shading.")
            inv_df = pd.DataFrame(columns=inv_internal_cols + [inv_recur_col])

    except FileNotFoundError:
        print(f"Warning: Inversion info file not found at {inv_file}. Proceeding without inversion shading.")
    except Exception as e:
        print(f"Warning: Error reading inversion file {inv_file}: {e}. Proceeding without inversion shading.")


    chrom_ranges = {}
    for c in unique_chroms:
        cdata = results_df[(results_df['chrom'] == c) & valid_mask]
        if cdata.empty:
             print(f"Error: Chromosome {c} was selected but has no valid data after filtering.")
             continue
        c_min = cdata['start'].min()
        c_max = cdata['end'].max()
        if c_min >= c_max:
             c_max = c_min + 1
        chrom_ranges[c] = (c_min, c_max)

    neg_log_p_values = results_df.loc[valid_mask, 'neg_log_p'].dropna()
    if neg_log_p_values.empty:
        print("Warning: No valid -log10(p) values found. Setting default Y-limit.")
        global_max_neglogp = 1.0
    else:
        global_max_neglogp = neg_log_p_values.max()

    if not np.isfinite(global_max_neglogp) or global_max_neglogp <= 0:
         print(f"Warning: Invalid global max -log10(p) ({global_max_neglogp}). Setting default Y-limit.")
         global_max_neglogp = 1.0
    YLIM_TOP = global_max_neglogp * 1.1

    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['blue','lightgray','red'], N=256)
    z_scores = results_df.loc[valid_mask, 'effect_zscore'].dropna()
    if not z_scores.empty:
        z_min = z_scores.min()
        z_max = z_scores.max()
        # Use a reasonable clip like +/- 3 for normalization range, centered at 0
        norm_vmin = max(z_min, -3)
        norm_vmax = min(z_max, 3)
        # Ensure vmin < vmax even if all z-scores are outside +/-3
        if norm_vmin >= norm_vmax:
             norm_vmin = -3
             norm_vmax = 3
        norm = TwoSlopeNorm(vmin=norm_vmin, vcenter=0.0, vmax=norm_vmax)
    else:
        norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
        print("Warning: No valid Z-scores found for coloring. Using default range [-1, 1].")

    print("Setting up plot layout...")
    n_chroms = len(unique_chroms)
    fig_h = max(8, 4.0)
    fig_w = max(20, 3*n_chroms)
    fig = plt.figure(figsize=(fig_w, fig_h))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=2, ncols=n_chroms+1, width_ratios=[1]*n_chroms + [0.15], height_ratios=[6,1], figure=fig, wspace=0.1, hspace=0.1)

    ax_subplots = []
    for i, c in enumerate(unique_chroms):
        ax_i = fig.add_subplot(gs[0, i])
        ax_subplots.append(ax_i)

    ax_bottom = fig.add_subplot(gs[1, :n_chroms])

    print("Building linear genome axis...")
    offsets = {}
    cum_len = 0
    actual_chrom_lengths_used = {}
    boundary_tick_positions = []
    midpoint_label_positions = []
    midpoint_label_texts = []

    for c in unique_chroms:
        real_len = CHR_LENGTHS.get(c, 1000000)
        if c not in CHR_LENGTHS:
            print(f"Warning: Chromosome {c} not found in CHR_LENGTHS dictionary. Using default length 1,000,000.")
        offsets[c] = cum_len
        actual_chrom_lengths_used[c] = real_len

        start_pos = offsets[c]
        end_pos = start_pos + real_len
        mid_pos = start_pos + real_len / 2

        boundary_tick_positions.extend([start_pos, end_pos])
        midpoint_label_positions.append(mid_pos)
        midpoint_label_texts.append(c.replace("chr",""))

        ax_bottom.hlines(0.5, start_pos, end_pos, color='darkgray', lw=5)

        cum_len += real_len

    ax_bottom.set_xlim(0, cum_len)
    ax_bottom.set_ylim(0, 1)
    for spine in ['top','left','right','bottom']:
        ax_bottom.spines[spine].set_visible(False)
    ax_bottom.set_yticks([])
    ax_bottom.set_xlabel("Chromosome", fontsize=28, labelpad=20)

    ax_bottom.set_xticks(boundary_tick_positions)
    ax_bottom.set_xticklabels([])
    ax_bottom.tick_params(axis='x', which='major', length=6, width=1.5, color='gray', direction='out')

    for pos, txt in zip(midpoint_label_positions, midpoint_label_texts):
         ax_bottom.text(pos, 0.3, txt, ha='center', va='center', fontsize=26, fontweight='bold')


    print("Plotting data for each chromosome...")
    recurrent_color = 'purple'
    single_color    = 'green'

    for i, c in enumerate(unique_chroms):
        ax_top = ax_subplots[i]
        ax_top.set_xlim(-0.08,1.08)
        ax_top.set_ylim(0, YLIM_TOP)

        if i == 0:
            ax_top.set_ylabel("-log10(p)", fontsize=26, labelpad=10)
            ax_top.tick_params(axis='y', labelsize=26)
        else:
            ax_top.set_yticks([])
            ax_top.set_ylabel("")

        ax_top.yaxis.grid(True, which='major', color='lightgray', linestyle='--', lw=0.7)
        ax_top.xaxis.grid(False)
        for spine in ['top','right']:
            ax_top.spines[spine].set_visible(False)
        ax_top.spines['left'].set_linewidth(0.5)
        ax_top.spines['bottom'].set_linewidth(0.5)
        ax_top.set_xticks([])

        cdata = results_df[(results_df['chrom'] == c) & valid_mask].copy()
        c_min_data, c_max_data = chrom_ranges[c]

        def normpos(x):
            if c_max_data > c_min_data:
                return max(0.0, min(1.0, (x - c_min_data) / (c_max_data - c_min_data)))
            else:
                return 0.5
        cdata['chr_plot_x'] = cdata['start'].apply(normpos)

        z_scores_chr = cdata['effect_zscore'].dropna()
        if not z_scores_chr.empty:
             point_colors = cmap(norm(z_scores_chr))
        else:
             point_colors = 'grey'

        ax_top.scatter(
            cdata['chr_plot_x'],
            cdata['neg_log_p'],
            c=point_colors,
            s=150, alpha=0.7, linewidth=0, zorder=2
        )

        used_min = cdata['start'].min()
        used_max = cdata['end'].max()
        if used_min >= used_max: used_max = used_min + 1
        left_rel = normpos(used_min)
        right_rel = normpos(used_max)

        invsub = inv_df[inv_df['chr'] == c]
        for _, invrow in invsub.iterrows():
            inv_start = invrow['region_start']
            inv_end   = invrow['region_end']
            if pd.isna(inv_start) or pd.isna(inv_end) or inv_end <= inv_start: continue

            if c_max_data > c_min_data:
                inv_left_rel = max(0.0, min(1.0, (inv_start - c_min_data) / (c_max_data - c_min_data)))
                inv_right_rel = max(0.0, min(1.0, (inv_end - c_min_data) / (c_max_data - c_min_data)))
            else:
                continue

            t = invrow.get(inv_recur_col, 0)
            inv_color = recurrent_color if t == 1 else single_color
            ax_top.axvspan(inv_left_rel, inv_right_rel, color=inv_color, alpha=0.15, zorder=0, lw=0)


        off = offsets.get(c, 0)
        real_len_chrom = actual_chrom_lengths_used.get(c, 1)

        bottom_min_pos = c_min_data
        bottom_max_pos = c_max_data

        bottom_left_x  = off + max(0, min(real_len_chrom, bottom_min_pos))
        bottom_right_x = off + max(0, min(real_len_chrom, bottom_max_pos))

        con_left = ConnectionPatch(
            xyA=(bottom_left_x, 0.5), xyB=(left_rel, 0),
            coordsA="data", coordsB="data",
            axesA=ax_bottom, axesB=ax_top,
            color="dimgray", lw=1, linestyle="--", zorder=10, alpha=0.6
        )
        con_right = ConnectionPatch(
            xyA=(bottom_right_x, 0.5), xyB=(right_rel, 0),
            coordsA="data", coordsB="data",
            axesA=ax_bottom, axesB=ax_top,
            color="dimgray", lw=1, linestyle="--", zorder=10, alpha=0.6
        )
        fig.add_artist(con_left)
        fig.add_artist(con_right)

        n_lines = 20
        if right_rel > left_rel and bottom_right_x > bottom_left_x:
            for k in range(1, n_lines):
                f = k / n_lines
                top_x = left_rel + f * (right_rel - left_rel)
                bottom_x = bottom_left_x + f * (bottom_right_x - bottom_left_x)
                con_mid = ConnectionPatch(
                    xyA=(bottom_x, 0.5), xyB=(top_x, 0),
                    coordsA="data", coordsB="data",
                    axesA=ax_bottom, axesB=ax_top,
                    color="lightgray", lw=0.5, alpha=0.2, linestyle="-", zorder=9
                )
                fig.add_artist(con_mid)

    print("Adding colorbar and legend...")
    ax_cb = fig.add_subplot(gs[0, n_chroms])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical', aspect=15)
    cb.set_label('Effect Size Z-score', fontsize=20, labelpad=15)
    cb.ax.tick_params(labelsize=20)
    min_val, max_val = norm.vmin, norm.vmax
    cb.ax.text(0.5, 1.05, f'Higher dN/dS\nin Inverted (Z>{max_val:.1f})', transform=cb.ax.transAxes, ha='center', va='bottom', fontsize=16)
    cb.ax.text(0.5, -0.05, f'Higher dN/dS\nin Direct (Z<{min_val:.1f})', transform=cb.ax.transAxes, ha='center', va='top', fontsize=16)
    pos = ax_cb.get_position()
    ax_cb.set_position([pos.x0 + pos.width*0.1, pos.y0 + pos.height*0.1, pos.width*0.8, pos.height*0.8])

    recurrent_patch = mpatches.Patch(color=recurrent_color, alpha=0.2, label='Recurrent inversion')
    single_patch = mpatches.Patch(color=single_color, alpha=0.2, label='Single-event inversion')
    if ax_subplots:
        ax_subplots[-1].legend(handles=[recurrent_patch, single_patch], fontsize=20, frameon=True, loc='upper right', bbox_to_anchor=(1.0, 1.15))
    else:
        print("Warning: No subplots created, cannot add legend.")

    print("Finalizing plot...")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_fname = PLOTS_DIR / "manhattan_plot_subplots.png"
    print(f"Saving plot to {out_fname}")
    plt.savefig(out_fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Plotting complete.")

def main():
    data_file = RESULTS_DIR / 'final_results.csv'
    inv_file = 'inv_info.csv'

    if not data_file.exists():
        print(f"ERROR: Main data file not found: {data_file}")
        print("Please ensure the first script ran successfully and produced this file.")
        return

    create_manhattan_plot(data_file, inv_file=inv_file)

if __name__ == "__main__":
    main()
