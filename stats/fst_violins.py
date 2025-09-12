import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import PathPatch
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

MAIN_COLOR = '#6A5ACD'
FILL_ALPHA = 0.10
EDGE_COLOR = '#4a4a4a'
RECURRENT_HATCH = '////////////////////'
SINGLE_EVENT_HATCH = '....................'
RECURRENT_HATCH_COLOR = (0.20, 0.20, 0.20, 0.85)
SINGLE_EVENT_HATCH_COLOR = (0.92, 0.92, 0.95, 1.0)
POINT_COLOR = '#1f77b4'
POINT_SIZE = 14
POINT_ALPHA = 0.6
JITTER_SD = 0.042
FIGSIZE = (4.6, 9.6)
DPI = 350
OUTPUT_PNG = 'hudson_fst.png'

plt.rcParams['hatch.linewidth'] = 0.18

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

def add_hatch_overlay(ax, poly_collection, hatch, hatch_color):
    paths = poly_collection.get_paths()
    for path in paths:
        patch = PathPatch(
            path,
            transform=ax.transData,
            facecolor=(1, 1, 1, 0),
            edgecolor=hatch_color,
            hatch=hatch,
            linewidth=0.0,
            zorder=14
        )
        ax.add_patch(patch)

def main():
    log.info("Starting single-figure Hudson F_ST analysis (fine overlays, blue points)")
    inv_df, sum_df, map_df = load_required_inputs()
    map_lookup = build_mapping_lookup(map_df)
    rec_map, sing_map = partition_inversions(inv_df, map_lookup)
    sum_df = sum_df.copy()
    sum_df['inversion_type'] = sum_df.apply(assign_inversion_type, axis=1, args=(rec_map, sing_map))
    key_rec = INVERSION_CATEGORY_MAPPING['Recurrent']
    key_sing = INVERSION_CATEGORY_MAPPING['Single-event']
    rec_vals = pd.to_numeric(sum_df.loc[sum_df['inversion_type'] == key_rec, HUDSON_FST_COL], errors='coerce').dropna().tolist()
    se_vals  = pd.to_numeric(sum_df.loc[sum_df['inversion_type'] == key_sing, HUDSON_FST_COL], errors='coerce').dropna().tolist()
    n_rec, n_se = len(rec_vals), len(se_vals)
    log.info(f"Hudson F_ST values: Recurrent N={n_rec}, Single-event N={n_se}")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor('white')
    positions = [0, 1]
    if n_rec == 0 and n_se == 0:
        ax.text(0.5, 0.5, "No numeric data for Hudson $F_{\\mathrm{ST}}$", ha='center', va='center', fontsize=12, color='crimson', transform=ax.transAxes)
        ax.axis('off')
        plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        log.warning("No data available; saved placeholder figure.")
        return
    vp = ax.violinplot([rec_vals, se_vals], positions=positions, widths=0.8, showmedians=True, showextrema=False)
    for b in vp['bodies']:
        b.set_facecolor(MAIN_COLOR)
        b.set_alpha(FILL_ALPHA)
        b.set_edgecolor(EDGE_COLOR)
        b.set_linewidth(0.8)
    vp['cmedians'].set_edgecolor('#1f1f1f')
    vp['cmedians'].set_linewidth(1.2)
    vp['cmedians'].set_zorder(13)
    if len(vp['bodies']) >= 1:
        add_hatch_overlay(ax, vp['bodies'][0], RECURRENT_HATCH, RECURRENT_HATCH_COLOR)
    if len(vp['bodies']) >= 2:
        add_hatch_overlay(ax, vp['bodies'][1], SINGLE_EVENT_HATCH, SINGLE_EVENT_HATCH_COLOR)
    rng = np.random.default_rng()
    if n_rec > 0:
        jit = rng.normal(0, JITTER_SD, size=n_rec)
        ax.scatter(np.full(n_rec, positions[0]) + jit, rec_vals, s=POINT_SIZE, c=POINT_COLOR, alpha=POINT_ALPHA, edgecolors='none', zorder=20)
    if n_se > 0:
        jit = rng.normal(0, JITTER_SD, size=n_se)
        ax.scatter(np.full(n_se, positions[1]) + jit, se_vals, s=POINT_SIZE, c=POINT_COLOR, alpha=POINT_ALPHA, edgecolors='none', zorder=20)
    ax.set_xlabel("")
    ax.set_ylabel(r"Hudson $F_{\mathrm{ST}}$", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', length=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Recurrent\n(N={n_rec})", f"Single-event\n(N={n_se})"], fontsize=11)
    ax.yaxis.grid(True, linestyle=':', color='lightgrey', alpha=0.75)
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
    ax.text(0.04, 0.97, f"Mann–Whitney U\n{p_text}", transform=ax.transAxes, ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.35', fc='ghostwhite', ec='lightgrey', alpha=0.9))
    plt.tight_layout(pad=1.6)
    plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved figure to '{OUTPUT_PNG}'")

if __name__ == "__main__":
    main()
