import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import mannwhitneyu

SUMMARY_STATS_FILE = 'output.csv'
INVERSION_FILE = 'inv_info.tsv'
COORDINATE_MAP_FILE = 'map.tsv'

SUMMARY_STATS_COORDINATE_COLUMNS = {'chr': 'chr', 'start': 'region_start', 'end': 'region_end'}
INVERSION_FILE_COLUMNS = ['Chromosome', 'Start', 'End', '0_single_1_recur_consensus']
MAP_FILE_COLUMNS = ['Original_Chr', 'Original_Start', 'Original_End', 'New_Chr', 'New_Start', 'New_End']

HUDSON_FST_COL = 'hudson_fst_hap_group_0v1'

INVERSION_CATEGORY_MAPPING = {
    'Recurrent': 'recurrent',
    'Single-event': 'single_event'
}

RECURRENT_COLOR = '#014421'
SINGLE_EVENT_COLOR = '#8B0000'
VIOLIN_ALPHA = 0.5
EDGE_COLOR = '#4a4a4a'
FIGSIZE = (4.6, 9.6)
DPI = 350
OUTPUT_PDF = 'hudson_fst.pdf'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('HudsonFstSinglePlot')

def normalize_chromosome_name(chromosome_id):
    s = str(chromosome_id).strip().lower()
    if s.startswith('chr_'):
        s = s[4:]
    elif s.startswith('chr'):
        s = s[3:]
    if not s.startswith('chr') and s not in ['x', 'y', 'm', 'mt']:
        s = f'chr{s}'
    return s

def check_coordinate_overlap(a, b):
    return a[0] == b[0] and abs(a[1]-b[1]) <= 1 and abs(a[2]-b[2]) <= 1

def load_required_inputs():
    if not os.path.exists(INVERSION_FILE):
        log.critical(f"Required inversion file '{INVERSION_FILE}' not found.")
        sys.exit(1)
    if not os.path.exists(SUMMARY_STATS_FILE):
        log.critical(f"Required summary file '{SUMMARY_STATS_FILE}' not found.")
        sys.exit(1)
    inv_df = pd.read_csv(INVERSION_FILE, sep='\t', usecols=lambda c: c in INVERSION_FILE_COLUMNS)
    sum_cols = list(SUMMARY_STATS_COORDINATE_COLUMNS.values()) + [HUDSON_FST_COL]
    sum_df = pd.read_csv(SUMMARY_STATS_FILE, usecols=lambda c: c in sum_cols)
    miss = [c for c in SUMMARY_STATS_COORDINATE_COLUMNS.values() if c not in sum_df.columns]
    if miss:
        log.critical(f"Summary file missing coordinate columns: {miss}")
        sys.exit(1)
    map_df = None
    if os.path.exists(COORDINATE_MAP_FILE):
        tmp = pd.read_csv(COORDINATE_MAP_FILE, sep='\t')
        if all(col in tmp.columns for col in MAP_FILE_COLUMNS):
            tmp['Original_Chr'] = tmp['Original_Chr'].apply(normalize_chromosome_name)
            tmp['New_Chr'] = tmp['New_Chr'].apply(normalize_chromosome_name)
            tmp = tmp[~tmp['Original_Chr'].eq('y') & ~tmp['New_Chr'].eq('y')]
            map_df = tmp
            log.info(f"Loaded coordinate mapping with {len(map_df)} rows.")
        else:
            log.warning("Map file present but missing required columns; ignoring mapping.")
    else:
        log.info("No coordinate map provided; using raw inversion coordinates.")
    return inv_df, sum_df, map_df

def build_mapping_lookup(map_df):
    if map_df is None:
        return {}
    lookup = {}
    for _, r in map_df.iterrows():
        try:
            oc = normalize_chromosome_name(r['Original_Chr'])
            os_ = int(r['Original_Start']); oe = int(r['Original_End'])
            nc = normalize_chromosome_name(r['New_Chr'])
            ns = int(r['New_Start']); ne = int(r['New_End'])
            lookup[(oc, os_, oe)] = (nc, ns, ne)
        except Exception:
            continue
    return lookup

def partition_inversions(inv_df, map_lookup):
    rec, sing = {}, {}
    for _, r in inv_df.iterrows():
        if pd.isna(r['Chromosome']) or pd.isna(r['Start']) or pd.isna(r['End']) or pd.isna(r['0_single_1_recur_consensus']):
            continue
        try:
            oc = normalize_chromosome_name(r['Chromosome'])
            os_ = int(r['Start']); oe = int(r['End'])
            cat = int(r['0_single_1_recur_consensus'])
        except Exception:
            continue
        key = (oc, os_, oe)
        if key in map_lookup:
            c, s, e = map_lookup[key]
        else:
            c, s, e = oc, os_, oe
        if s > e:
            s, e = e, s
        if cat == 1:
            rec.setdefault(c, []).append((c, s, e))
        elif cat == 0:
            sing.setdefault(c, []).append((c, s, e))
    return rec, sing

def assign_inversion_type(row, rec_map, sing_map):
    c = normalize_chromosome_name(row[SUMMARY_STATS_COORDINATE_COLUMNS['chr']])
    try:
        s = int(row[SUMMARY_STATS_COORDINATE_COLUMNS['start']])
        e = int(row[SUMMARY_STATS_COORDINATE_COLUMNS['end']])
    except Exception:
        return 'coordinate_error'
    if s > e:
        s, e = e, s
    curr = (c, s, e)
    is_rec = any(check_coordinate_overlap(curr, t) for t in rec_map.get(c, []))
    is_sing = any(check_coordinate_overlap(curr, t) for t in sing_map.get(c, []))
    if is_rec and not is_sing:
        return INVERSION_CATEGORY_MAPPING['Recurrent']
    if is_sing and not is_rec:
        return INVERSION_CATEGORY_MAPPING['Single-event']
    if is_rec and is_sing:
        return 'ambiguous_match'
    return 'no_match'

def mann_whitney_fmt(a, b):
    if len(a) == 0 or len(b) == 0:
        return "Test N/A"
    try:
        if np.var(a) == 0 and np.var(b) == 0 and np.mean(a) == np.mean(b):
            return "p = 1.0"
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    except ValueError:
        return "Test error"

def main():
    log.info("Starting single-figure Hudson F_ST analysis (fine overlays, blue points)")
    inv_df, sum_df, map_df = load_required_inputs()
    map_lookup = build_mapping_lookup(map_df)
    rec_map, sing_map = partition_inversions(inv_df, map_lookup)
    sum_df = sum_df.copy()
    sum_df['inversion_type'] = sum_df.apply(assign_inversion_type, axis=1, args=(rec_map, sing_map))
    key_rec = INVERSION_CATEGORY_MAPPING['Recurrent']
    key_sing = INVERSION_CATEGORY_MAPPING['Single-event']
    
    # Extract numeric FST values by category
    rec_series = pd.to_numeric(sum_df.loc[sum_df['inversion_type'] == key_rec, HUDSON_FST_COL], errors='coerce')
    se_series  = pd.to_numeric(sum_df.loc[sum_df['inversion_type'] == key_sing, HUDSON_FST_COL], errors='coerce')
    rec_vals = rec_series.dropna().tolist()
    se_vals  = se_series.dropna().tolist()
    n_rec, n_se = len(rec_vals), len(se_vals)

    log.info("--- Analyzing recurrent inversion matches and drops ---")
    all_rec_invs = sorted(list({inv for sublist in rec_map.values() for inv in sublist}))
    total_recurrent_found = len(all_rec_invs)

    # Pre-process summary stats for efficient lookup
    chr_col = SUMMARY_STATS_COORDINATE_COLUMNS['chr']
    start_col = SUMMARY_STATS_COORDINATE_COLUMNS['start']
    end_col = SUMMARY_STATS_COORDINATE_COLUMNS['end']
    processed_sum_stats = []
    for _, row in sum_df.iterrows():
        try:
            c = normalize_chromosome_name(row[chr_col])
            s = int(row[start_col]); e = int(row[end_col])
            if s > e: s, e = e, s
            fst_val = row[HUDSON_FST_COL]
            # Store original FST value along with numeric version
            numeric_fst = pd.to_numeric(fst_val, errors='coerce')
            processed_sum_stats.append({
                'coord': (c, s, e),
                'original_fst': fst_val,
                'numeric_fst': numeric_fst
            })
        except Exception:
            continue

    # Group stats by chromosome for faster search
    sum_stats_by_chr = {}
    for item in processed_sum_stats:
        chrom = item['coord'][0]
        sum_stats_by_chr.setdefault(chrom, []).append(item)

    # Analyze each recurrent inversion from the inversion file
    unmatched_inversions = []
    dropped_for_fst = []

    for inv_coord in all_rec_invs:
        inv_chr = inv_coord[0]
        matches = []
        if inv_chr in sum_stats_by_chr:
            for stat_item in sum_stats_by_chr[inv_chr]:
                if check_coordinate_overlap(inv_coord, stat_item['coord']):
                    matches.append(stat_item)

        if not matches:
            unmatched_inversions.append(inv_coord)
        else:
            # Check if any of the matches have a valid FST value
            has_valid_fst = any(not np.isnan(m['numeric_fst']) for m in matches)
            if not has_valid_fst:
                # All matches had invalid FST. Record the inversion and the first invalid value found.
                first_invalid_match = matches[0]
                dropped_for_fst.append((inv_coord, first_invalid_match['coord'], first_invalid_match['original_fst']))

    if unmatched_inversions:
        log.warning("The following recurrent inversions from the inversion file had NO coordinate match in the summary file:")
        for c, s, e in unmatched_inversions:
            log.warning(f"  - Unmatched: {c}:{s}-{e}")

    if dropped_for_fst:
        log.warning("The following recurrent inversions were matched but dropped because ALL matches had invalid FST values:")
        for inv_c, match_c, fst in dropped_for_fst:
            log.warning(f"  - Dropped: {inv_c[0]}:{inv_c[1]}-{inv_c[2]} (matched e.g. {match_c[0]}:{match_c[1]}-{match_c[2]} with FST='{fst}')")

    log.info(f"Found {total_recurrent_found} recurrent inversions in the inversion file.")
    log.info(f"Dropped {len(unmatched_inversions)} inversions with no coordinate match and {len(dropped_for_fst)} inversions with only invalid FST matches.")

    log.info(f"Final counts for analysis (based on summary file rows): Recurrent N={n_rec}, Single-event N={n_se}")

    rec_mean = float(np.mean(rec_vals)) if n_rec else float('nan')
    se_mean  = float(np.mean(se_vals)) if n_se else float('nan')
    rec_median = float(np.median(rec_vals)) if n_rec else float('nan')
    se_median  = float(np.median(se_vals)) if n_se else float('nan')

    # Mann–Whitney U test (two-sided)
    p_value = np.nan
    try:
        if n_rec and n_se:
            if (np.var(rec_vals) == 0 and np.var(se_vals) == 0 and np.mean(rec_vals) == np.mean(se_vals)):
                p_value = 1.0
            else:
                _, p_value = mannwhitneyu(rec_vals, se_vals, alternative='two-sided')
    except Exception:
        pass

    # Safe fold helper
    def _safe_fold(num, den):
        if den is None or np.isnan(den) or den == 0:
            return np.inf if (num is not None and not np.isnan(num) and num > 0) else np.nan
        return float(num) / float(den)

    fold_median_rec_over_se = _safe_fold(rec_median, se_median)
    fold_median_se_over_rec = _safe_fold(se_median, rec_median)
    fold_mean_rec_over_se   = _safe_fold(rec_mean, se_mean)
    fold_mean_se_over_rec   = _safe_fold(se_mean, rec_mean)

    log.info("--- Hudson F_ST summary ---")
    log.info(f"Median F_ST: Recurrent = {rec_median:.6g}, Single-event = {se_median:.6g}")
    log.info(f"Mean   F_ST: Recurrent = {rec_mean:.6g},  Single-event = {se_mean:.6g}")
    if not np.isnan(p_value):
        log.info(f"Mann–Whitney U (two-sided): p = {p_value:.6g}")
    else:
        log.info("Mann–Whitney U (two-sided): p = N/A")

    log.info(f"Fold (medians): recurrent/single-event = {fold_median_rec_over_se:.6g}; single-event/recurrent = {fold_median_se_over_rec:.6g}")
    log.info(f"Fold (means)  : recurrent/single-event = {fold_mean_rec_over_se:.6g}; single-event/recurrent = {fold_mean_se_over_rec:.6g}")

    # List all loci with F_ST values (by category)
    chr_col   = SUMMARY_STATS_COORDINATE_COLUMNS['chr']
    start_col = SUMMARY_STATS_COORDINATE_COLUMNS['start']
    end_col   = SUMMARY_STATS_COORDINATE_COLUMNS['end']

    def _fmt_id(row):
        try:
            c = normalize_chromosome_name(row[chr_col])
            s = int(row[start_col]); e = int(row[end_col])
            if s > e: s, e = e, s
            return f"{c}:{s}-{e}"
        except Exception:
            return "unknown"

    rec_rows = sum_df.loc[sum_df['inversion_type'] == key_rec,  [chr_col, start_col, end_col, HUDSON_FST_COL]].copy()
    se_rows  = sum_df.loc[sum_df['inversion_type'] == key_sing, [chr_col, start_col, end_col, HUDSON_FST_COL]].copy()
    rec_rows[HUDSON_FST_COL] = pd.to_numeric(rec_rows[HUDSON_FST_COL], errors='coerce')
    se_rows[HUDSON_FST_COL]  = pd.to_numeric(se_rows[HUDSON_FST_COL],  errors='coerce')

    log.info("-- Per-locus F_ST (recurrent) --")
    for _, r in rec_rows.dropna(subset=[HUDSON_FST_COL]).iterrows():
        log.info(f"recurrent\t{_fmt_id(r)}\tF_ST={float(r[HUDSON_FST_COL]):.6g}")

    log.info("-- Per-locus F_ST (single-event) --")
    for _, r in se_rows.dropna(subset=[HUDSON_FST_COL]).iterrows():
        log.info(f"single-event\t{_fmt_id(r)}\tF_ST={float(r[HUDSON_FST_COL]):.6g}")
    # ---------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor('white')
    positions = [0, 1]
    if n_rec == 0 and n_se == 0:
        ax.text(0.5, 0.5, "No numeric data for Hudson $F_{\\mathrm{ST}}$", ha='center', va='center', fontsize=12, color='crimson', transform=ax.transAxes)
        ax.axis('off')
        plt.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        log.warning("No data available; saved placeholder figure.")
        return
    vp = ax.violinplot([rec_vals, se_vals], positions=positions, widths=0.8, showmedians=True, showextrema=False)
    jitter_width = 0.08  # Controls the horizontal spread of points
    point_size = 8       # Controls the size of the points

    # Jitter points for the 'Recurrent' category (at position 0)
    if n_rec > 0:
        x_rec = positions[0] + np.random.normal(0, jitter_width, n_rec)
        ax.scatter(x_rec, rec_vals, s=point_size, color='black', alpha=0.5, zorder=10)

    # Jitter points for the 'Single-event' category (at position 1)
    if n_se > 0:
        x_se = positions[1] + np.random.normal(0, jitter_width, n_se)
        ax.scatter(x_se, se_vals, s=point_size, color='black', alpha=0.5, zorder=10)
    violin_colors = [RECURRENT_COLOR, SINGLE_EVENT_COLOR]
    for idx, b in enumerate(vp['bodies']):
        color = violin_colors[idx] if idx < len(violin_colors) else '#7f7f7f'
        b.set_facecolor(color)
        b.set_alpha(VIOLIN_ALPHA)
        b.set_edgecolor(EDGE_COLOR)
        b.set_linewidth(0.8)
    vp['cmedians'].set_edgecolor('#1f1f1f')
    vp['cmedians'].set_linewidth(1.2)
    vp['cmedians'].set_zorder(13)
    ax.set_xlabel("")
    ax.set_ylabel(r"Hudson $F_{\mathrm{ST}}$", fontsize=16)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=13, length=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Recurrent\n(N={n_rec})", f"Single-event\n(N={n_se})"], fontsize=13)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set_color('#8a8a8a')
    all_vals = np.array(rec_vals + se_vals) if (n_rec + n_se) > 0 else np.array([0.0])
    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    if np.isfinite(ymin) and np.isfinite(ymax):
        rng_y = ymax - ymin
        pad = 0.08 * rng_y if rng_y > 0 else max(0.01, abs(ymax) * 0.1)
        ax.set_ylim(ymin - pad, ymax + pad * 1.25)
    p_text = mann_whitney_fmt(rec_vals, se_vals)
    ax.text(0.04, 0.97, f"Mann–Whitney U\n{p_text}", transform=ax.transAxes, ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round,pad=0.35', fc='ghostwhite', ec='lightgrey', alpha=0.9))
    plt.tight_layout(pad=1.6)
    plt.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved figure to '{OUTPUT_PDF}'")

if __name__ == "__main__":
    main()
