import os
import re
import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde
import seaborn as sns

# =============================================================================
# Global configuration
# =============================================================================

matplotlib.rcParams.update({
    "pdf.fonttype": 42,   
    "ps.fonttype": 42,  
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RANDOM_SEED = 2027
np.random.seed(RANDOM_SEED)

# === Match the other plot's look ===
COLOR_DIRECT   = "#1f3b78"   # dark blue
COLOR_INVERTED = "#8c2d7e"   # reddish purple

# Hatching overlays (by recurrence)
OVERLAY_SINGLE = "#d9d9d9"   # very light dots
OVERLAY_RECUR  = "#4a4a4a"   # dark gray diagonals
ALPHA_VIOLIN   = 0.72        # same translucent fill

# Category order + face colors (orientation decides the fill color)
CATEGORY_ORDER = [
    "Single-event, direct",
    "Single-event, inverted",
    "Recurrent, direct",
    "Recurrent, inverted",
]
CATEGORY_FACE = {
    "Single-event, direct":   COLOR_DIRECT,
    "Single-event, inverted": COLOR_INVERTED,
    "Recurrent, direct":      COLOR_DIRECT,
    "Recurrent, inverted":    COLOR_INVERTED,
}

# Base colors for raw haplotype plots: A,C,G,T (no gap color used)
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
BASE_COLORS = ["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"]  # A,C,G,T (IGV: A=green, C=blue, G=orange, T=red)
BASE_CMAP = ListedColormap(BASE_COLORS)  # vmin=0, vmax=3

# Input filenames (all in current directory)
CDS_SUMMARY_FILE = "cds_identical_proportions.tsv"
GENE_TESTS_FILE  = "gene_inversion_direct_inverted.tsv"
# Output figure filenames
VIOLIN_PLOT_FILE = "cds_proportion_identical_by_category_violin.pdf"
VOLCANO_PLOT_FILE = "cds_conservation_volcano.pdf"
MAPT_HEATMAP_FILE = "mapt_cds_polymorphism_heatmap.pdf"

# Fixed-differences gene to show raw haplotypes (MAPT example)
MAPT_GENE = "MAPT"
# Unanimity threshold per orientation for "fixed difference" columns (strict by default)
FIXED_DIFF_UNANIMITY_THRESHOLD = 1.0

# =============================================================================
# Utilities
# =============================================================================

def safe_read_tsv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    # Coerce numeric-looking columns at the DataFrame level (not via .apply)
    df = _coerce_numeric_cols(df)
    return df

def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce numeric-looking columns to numeric dtype, safely."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        try:
            numeric_mask = pd.to_numeric(out[c], errors="coerce").notna()
            if numeric_mask.mean() >= 0.7:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        except Exception:
            pass
    return out

def build_category(consensus: int, phy_group: int) -> str:
    recurrence = "Single-event" if int(consensus) == 0 else "Recurrent"
    orientation = "direct" if int(phy_group) == 0 else "inverted"
    return f"{recurrence}, {orientation}"

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# =============================================================================
# PHYLIP parsing (strict)
# =============================================================================

NONSTD_LINE_RE = re.compile(r"^(?P<name>.*?_[LR])(?P<seq>[ACGTN\-]+)$")
HEADER_RE      = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")

def read_phy(phy_path: str):
    """
    Reads PHYLIP in either:
    - Non-standard format: one line per sequence: name_L/R immediately followed by sequence.
    - Standard sequential or interleaved with 10-char name field.
    Returns: dict with keys: seq_order (list), seqs (dict name->str), n (int), m (int)
    """
    if not os.path.exists(phy_path):
        raise FileNotFoundError(f"PHYLIP file not found: {phy_path}")

    with open(phy_path, "r") as f:
        lines = [ln.rstrip("\n\r") for ln in f]

    if not lines:
        raise ValueError(f"Empty PHYLIP file: {phy_path}")

    m = HEADER_RE.match(lines[0])
    if not m:
        raise ValueError(f"Malformed PHYLIP header in {phy_path!r}: {lines[0]!r}")
    n_expected = int(m.group(1))
    m_sites    = int(m.group(2))

    if len(lines) < 2:
        raise ValueError(f"No sequence lines in {phy_path!r}")
    first_seq_line = lines[1].strip()

    if NONSTD_LINE_RE.match(first_seq_line):
        # Non-standard: each line is name + sequence glued
        seqs = {}
        seq_order = []
        for ln in lines[1:]:
            ln = ln.strip()
            if not ln:
                continue
            m2 = NONSTD_LINE_RE.match(ln)
            if not m2:
                raise ValueError(f"Non-standard PHYLIP line didn't match regex in {phy_path!r}: {ln!r}")
            name = m2.group("name")
            seq  = m2.group("seq").upper()
            if len(seq) != m_sites:
                raise ValueError(f"Sequence length {len(seq)} != m_sites {m_sites} in {phy_path!r} for {name}")
            if name in seqs:
                raise ValueError(f"Duplicate sequence name {name!r} in {phy_path!r}")
            seqs[name] = seq
            seq_order.append(name)
        if len(seqs) != n_expected:
            raise ValueError(f"Found {len(seqs)} seqs but header says {n_expected} in {phy_path!r}")
        return {"seq_order": seq_order, "seqs": seqs, "n": n_expected, "m": m_sites}

    # Standard PHYLIP (sequential or interleaved)
    seqs = {}
    seq_order = []
    content = lines[1:]
    while content and not content[0].strip():
        content.pop(0)

    def parse_name_seq(line):
        if not line.strip():
            return None, ""
        parts = line.strip().split()
        if len(parts) >= 2 and re.fullmatch(r"[A-Za-z0-9_\-\.]+", parts[0]) and re.fullmatch(r"[ACGTN\-]+", parts[1]):
            return parts[0], parts[1]
        if len(line) > 10:
            name = line[:10].strip()
            seq  = line[10:].strip().replace(" ", "")
            if name and re.fullmatch(r"[ACGTN\-]+", seq):
                return name, seq
        if re.fullmatch(r"[ACGTN\-]+", line.strip()):
            return None, line.strip()
        return None, ""

    idx = 0
    while idx < len(content) and len(seqs) < n_expected:
        line = content[idx]
        name, seq_part = parse_name_seq(line)
        if name is None or not seq_part:
            idx += 1
            continue
        if name in seqs:
            raise ValueError(f"Duplicate sequence name {name!r} in {phy_path!r}")
        seqs[name] = seq_part
        seq_order.append(name)
        idx += 1

    while any(len(seqs[nm]) < m_sites for nm in seq_order) and idx < len(content):
        while idx < len(content) and not content[idx].strip():
            idx += 1
        for nm_i in range(n_expected):
            if idx >= len(content):
                break
            line = content[idx]
            name_guess, seq_part = parse_name_seq(line)
            if not seq_part:
                idx += 1
                continue
            if name_guess is None:
                target = seq_order[nm_i]
                seqs[target] += seq_part
            else:
                if name_guess not in seqs:
                    raise ValueError(f"Unexpected name {name_guess!r} in subsequent block in {phy_path!r}")
                seqs[name_guess] += seq_part
            idx += 1

    for nm in seq_order:
        if len(seqs[nm]) != m_sites:
            raise ValueError(f"Sequence {nm!r} length {len(seqs[nm])} != m_sites {m_sites} in {phy_path!r}")

    return {"seq_order": seq_order, "seqs": seqs, "n": n_expected, "m": m_sites}


# =============================================================================
# Data loading & preparation
# =============================================================================

def load_cds_summary() -> pd.DataFrame:
    """
    Load cds_identical_proportions.tsv with **strict locus normalization**.

    Key fixes:
      - Force `chr` to a canonical **string** WITHOUT decimals or 'chr' prefix (e.g., '7', not '7.0' or 'chr7').
      - Build inv_id AFTER that normalization so it's always '7:123-456' (ASCII hyphen).
      - Coerce only numeric measurement columns to numeric; never `chr`.
      - Emit extensive DEBUG about types, example values, and row drops.

    This prevents '7.0:...' inv_id strings and ensures numeric-locus matches work downstream.
    """

    # ------------------------ load ------------------------
    print("[load_cds_summary] START")
    print(f"[load_cds_summary] reading: {CDS_SUMMARY_FILE}")
    df = safe_read_tsv(CDS_SUMMARY_FILE)

    print(f"[load_cds_summary] raw shape: {df.shape}")
    print("[load_cds_summary] raw columns:", list(df.columns))
    try:
        print("[load_cds_summary] raw dtypes:\n" + df.dtypes.to_string())
    except Exception:
        pass

    # ------------------------ snapshot: chr BEFORE ------------------------
    if "chr" not in df.columns:
        raise ValueError("[load_cds_summary] FATAL: input is missing 'chr' column")

    # Show a few raw 'chr' values (whatever type they are right now)
    try:
        ex_chr = df["chr"].head(12).tolist()
        print("[load_cds_summary] examples of 'chr' BEFORE normalization:", ex_chr)
    except Exception as e:
        print(f"[load_cds_summary] could not preview raw chr values: {e}")

    # ------------------------ strict normalization ------------------------
    def _canon_chr_val(x):
        """
        Canonicalize a single chromosome value to a clean string:
          - strip spaces
          - drop leading 'chr'/'CHR'
          - remove trailing '.0' if present (from floatification)
          - keep 'X','Y','MT' as-is (after prefix strip)
        """
        s = str(x).strip()
        # remove 'chr' prefix if present
        if s.lower().startswith("chr"):
            s = s[3:]
        # common float artifacts like '7.0', '10.000'
        m = re.fullmatch(r"(\d+)(?:\.0+)?", s)
        if m:
            return m.group(1)
        return s  # e.g., 'X', 'Y', 'MT', or already clean numerics

    # Make a **new** normalized chr column then overwrite df['chr']
    chr_norm = df["chr"].map(_canon_chr_val)

    # Show diagnostics around suspicious values
    bad_like_float = chr_norm[chr_norm.str.contains(r"\.", regex=True, na=False)]
    with_chr_prefix = df["chr"].astype(str).str.lower().str.startswith("chr", na=False)

    print(f"[load_cds_summary] chr: dtype_before={df['chr'].dtype} -> dtype_after={chr_norm.dtype}")
    print(f"[load_cds_summary] chr: #with 'chr' prefix BEFORE: {int(with_chr_prefix.sum())}")
    print(f"[load_cds_summary] chr: #with '.' AFTER normalization: {int(bad_like_float.shape[0])}")
    try:
        print("[load_cds_summary] chr unique (up to 15) AFTER:", sorted(chr_norm.dropna().unique().tolist())[:15])
    except Exception:
        pass

    df["chr"] = chr_norm  # overwrite with canonical strings

    # ------------------------ coerce numeric metrics ------------------------
    # Only coerce **metrics/coordinates**; never 'chr', 'gene_name', 'transcript_id', 'filename', 'inv_id'
    numeric_cols = [
        "consensus", "phy_group", "n_sequences", "n_pairs", "n_identical_pairs",
        "prop_identical_pairs", "cds_start", "cds_end", "inv_start", "inv_end"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------ build canonical inv_id ------------------------
    # We MUST build inv_id from the normalized 'chr' and integer start/end
    if not {"inv_start", "inv_end"}.issubset(df.columns):
        missing = {"inv_start", "inv_end"} - set(df.columns)
        raise ValueError(f"[load_cds_summary] FATAL: missing coordinate column(s): {missing}")

    # Keep a pre-build preview of potential floaty starts/ends
    try:
        ex_coords = df[["inv_start", "inv_end"]].head(8).to_dict("records")
        print("[load_cds_summary] examples inv_start/inv_end BEFORE int-cast:", ex_coords)
    except Exception:
        pass

    # Construct canonical inv_id safely, row-wise
    def _mk_inv_id(r):
        try:
            ch = str(r["chr"]).strip()
            st = int(float(r["inv_start"]))  # robust if stored as '73683626.0'
            en = int(float(r["inv_end"]))
            return f"{ch}:{st}-{en}"
        except Exception:
            return np.nan

    df["inv_id"] = df.apply(_mk_inv_id, axis=1)

    # Diagnostics on inv_id just created
    inv_na = int(df["inv_id"].isna().sum())
    inv_bad_chr = int(df["inv_id"].str.contains(r"chr", case=False, na=False).sum())
    inv_has_dot = int(df["inv_id"].str.contains(r"\.", na=False).sum())
    print(f"[load_cds_summary] inv_id built: n={len(df)}, n_missing={inv_na}, n_with_chr_prefix={inv_bad_chr}, n_with_dot={inv_has_dot}")

    try:
        print("[load_cds_summary] inv_id examples (10):", df["inv_id"].dropna().head(10).tolist())
    except Exception:
        pass

    # ------------------------ derived labels ------------------------
    df["category"]    = df.apply(lambda r: build_category(r["consensus"], r["phy_group"]), axis=1)
    df["orientation"] = df["phy_group"].map({0: "D", 1: "I"})
    df["recurrence"]  = df["consensus"].map({0: "SE", 1: "REC"})

    # ------------------------ filters with counts ------------------------
    n_before = len(df)

    print("\n[load_cds_summary] --- Pre-Filter Dropped Data Report (Sequential) ---")

    df_for_reporting = df.copy()

    # --- Step 1: Report on 'inv_exact_match' filter ---
    if "inv_exact_match" in df.columns:
        mask_bad_match = df_for_reporting['inv_exact_match'] != 1
        print(f"\n[Step 1] Analyzing 'inv_exact_match' filter on {len(df_for_reporting)} rows...")
        if mask_bad_match.any():
            print(f"[!] Found {int(mask_bad_match.sum())} CDSs that will be dropped because 'inv_exact_match' is not 1.")
            # We now proceed with only the rows that would pass this filter
            df_for_reporting = df_for_reporting[~mask_bad_match]
        else:
            print("[✔] All CDS entries pass the 'inv_exact_match == 1' check.")
        print("-" * 60)
    
    # --- Step 2: Report on 'n_pairs > 0' filter (on the REMAINING data) ---
    print(f"[Step 2] Analyzing 'n_pairs > 0' filter on the remaining {len(df_for_reporting)} rows...")
    mask_not_enough_haps = ~(df_for_reporting['n_pairs'] > 0)
    if mask_not_enough_haps.any():
        dropped_for_haps = df_for_reporting[mask_not_enough_haps]
        print(f"\n[!] Found {len(dropped_for_haps)} CDSs that will be dropped due to having < 2 haplotypes (n_pairs <= 0).")
        
        # Aggregate by inversion AND orientation
        counts_table = (dropped_for_haps.groupby(['inv_id', 'orientation'])
                                        .size()
                                        .unstack(fill_value=0))
        
        # Ensure both D and I columns exist for clarity
        if 'D' not in counts_table: counts_table['D'] = 0
        if 'I' not in counts_table: counts_table['I'] = 0
        
        # Rename columns for clarity and add a total
        counts_table = counts_table.rename(columns={'D': 'Direct_Dropped', 'I': 'Inverted_Dropped'})
        counts_table['Total_Dropped'] = counts_table['Direct_Dropped'] + counts_table['Inverted_Dropped']
        
        # Sort by the total number of dropped CDSs
        counts_table = counts_table.sort_values('Total_Dropped', ascending=False)
        
        print("[+] Details: Breakdown of CDSs dropped per inversion due to insufficient haplotypes:")
        if not counts_table.empty:
            # Use to_string() to ensure the full table prints
            print(counts_table.to_string())
        else:
            print("   (No specific inversion data to show for this filter)")
    else:
        print("\n[✔] All remaining CDS entries have sufficient haplotypes (>= 2).")

    print("\n[load_cds_summary] --- End of Pre-Filter Report ---\n")
    

    if "inv_exact_match" in df.columns:
        before = len(df)
        df = df[df["inv_exact_match"] == 1]
        print(f"[load_cds_summary] filter inv_exact_match==1: kept {len(df)}/{before} (dropped {before-len(df)})")
    else:
        print("[load_cds_summary] NOTE: no 'inv_exact_match' column; skipping that filter")

    before = len(df)
    df = df[df["n_pairs"] > 0]
    print(f"[load_cds_summary] filter n_pairs>0: kept {len(df)}/{before} (dropped {before-len(df)})")

    # category as ordered categorical
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_ORDER, ordered=True)

    # ------------------------ post-load sanity ------------------------
    print(f"[load_cds_summary] final shape: {df.shape} (started with {n_before})")
    try:
        by_orient = df["phy_group"].value_counts(dropna=False).to_dict()
        print("[load_cds_summary] phy_group counts:", by_orient)
    except Exception:
        pass

    try:
        uniq_inv = df["inv_id"].dropna().unique()
        print(f"[load_cds_summary] unique inv_id: {len(uniq_inv)}")
        # show if any lingering malformed inv_ids exist
        malformed = df["inv_id"].dropna().loc[~df["inv_id"].str.match(r"^[^:]+:\d+-\d+$")].unique()
        if len(malformed):
            print("[load_cds_summary] WARNING: malformed inv_id examples:", malformed[:10])
        else:
            print("[load_cds_summary] inv_id look clean (ASCII, no decimals, no 'chr' prefix).")
    except Exception:
        pass

    # Extra: quick cross-check that common loci exist in this table
    probe_examples = [
        "7:73113989-74799029",
        "17:45585159-46292045",
        "10:79542901-80217413",
        "15:30618103-32153204",
        "8:7301024-12598379",
        "6:167209001-167357782",
    ]
    for pe in probe_examples:
        try:
            n_hit = int((df["inv_id"] == pe).sum())
            print(f"[load_cds_summary] probe inv_id={pe} -> rows={n_hit}")
        except Exception:
            pass

    print("[load_cds_summary] END")
    return df


def load_gene_tests() -> pd.DataFrame:
    df = safe_read_tsv(GENE_TESTS_FILE)
    for col in ["p_direct", "p_inverted", "delta", "se_delta", "z_value", "p_value", "q_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_pairs_and_phy_index(cds_summary: pd.DataFrame) -> pd.DataFrame:
    files = sorted(set(cds_summary["filename"].astype(str).tolist()))
    rows = []
    for fn in files:
        phy_path = os.path.join(".", fn)
        rows.append({"filename": fn, "phy_path": phy_path})
    return pd.DataFrame(rows).set_index("filename")

# =============================================================================
# Violin plot: proportion of identical CDS pairs by inversion class
# =============================================================================

def compute_group_distributions(cds_summary: pd.DataFrame):
    out = {}
    for cat in CATEGORY_ORDER:
        sub = cds_summary[cds_summary["category"] == cat]
        vals = sub["prop_identical_pairs"].dropna().astype(float).values
        if len(vals) == 0:
            out[cat] = dict(values_all=np.array([]), values_core=np.array([]),
                            share_at_1=0.0, n_cds=0, n_pairs_total=0,
                            n_at1=0, box_stats=(np.nan, np.nan, np.nan))
            continue
        at1_mask = (vals == 1.0)
        n_total  = len(vals)
        n_at1    = int(at1_mask.sum())
        core     = vals[~at1_mask]   # for violin
        med      = np.median(vals)
        q1       = np.percentile(vals, 25)
        q3       = np.percentile(vals, 75)
        out[cat] = dict(values_all=vals, values_core=core, share_at_1=n_at1/n_total,
                        n_cds=n_total, n_pairs_total=int(sub["n_pairs"].sum()),
                        n_at1=n_at1, box_stats=(med, q1, q3))
    return out

def draw_half_violin(
    ax, y_vals, center_x, width=0.4, side="left",
    clip=(0.0, 1.0), facecolor="#cccccc", alpha=0.6,
    hatch=None, hatch_edgecolor="#000000", hatch_linewidth=0.0,
    y_grid=None, global_density_max=None, bw_method="scott"
):
    # keep only clipped values
    y = np.asarray(y_vals, dtype=float)
    y = y[(y >= clip[0]) & (y <= clip[1])]
    if y.size < 2:
        return

    # shared y-grid for consistent shapes
    if y_grid is None:
        y_grid = np.linspace(clip[0], clip[1], 400)

    # KDE smoothing (scale to counts so widths reflect absolute number of points)
    try:
        kde  = gaussian_kde(y, bw_method=bw_method)
        dens = kde(y_grid) * y.size  # counts, not PDF
    except Exception:
        # robust fallback if KDE fails
        hist, bin_edges = np.histogram(y, bins=40, range=clip, density=False)
        # simple step function onto y_grid
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dens = np.interp(y_grid, bin_centers, hist, left=0, right=0)

    dens  = np.clip(dens, 0, np.inf)
    scale = float(global_density_max) if (global_density_max is not None and global_density_max > 0) else float(dens.max())
    if scale <= 0:
        return
    xs = (dens / scale) * width

    # x on the "open" side
    xcoords = center_x - xs if side == "left" else center_x + xs

    # 1) colored base fill
    poly = ax.fill_betweenx(
        y_grid,
        center_x, xcoords,
        color=facecolor, alpha=alpha, linewidth=0, zorder=1.5
    )

    # 2) hatched overlay (drawn as a closed polygon with no facecolor)
    if hatch:
        # Build polygon vertices around the filled area
        xv = np.r_[np.full_like(y_grid, center_x), xcoords[::-1]]
        yv = np.r_[y_grid, y_grid[::-1]]
        verts = np.column_stack([xv, yv])
        patch = mpatches.Polygon(
            verts, closed=True,
            facecolor="none",
            edgecolor=hatch_edgecolor,
            hatch=hatch,
            linewidth=hatch_linewidth,
            zorder=2.0
        )
        ax.add_patch(patch)

def plot_proportion_identical_violin(cds_summary: pd.DataFrame, outfile: str):
    dist = compute_group_distributions(cds_summary)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.set_facecolor("white")
    ax.set_ylabel("Proportion of identical CDS pairs")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlim(0.5, len(CATEGORY_ORDER) + 0.5)

    positions = range(1, len(CATEGORY_ORDER) + 1)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(CATEGORY_ORDER)

    # Background shading separates single-event vs recurrent classes
    ax.axvspan(0.5, 2.5, color="#ede7f6", alpha=0.18, zorder=0)
    ax.axvspan(2.5, 4.5, color="#e6f4ea", alpha=0.18, zorder=0)

    # --- Compute a global max of counts (so widths are comparable across violins) ---
    y_grid = np.linspace(0.0, 1.0, 400)
    global_density_max = 0.0
    for cat_tmp in CATEGORY_ORDER:
        vals_tmp = dist[cat_tmp]["values_core"]
        vals_tmp = vals_tmp[(vals_tmp >= 0.0) & (vals_tmp <= 1.0)]
        if vals_tmp.size >= 2:
            try:
                kde_tmp = gaussian_kde(vals_tmp, bw_method="scott")
                dens_tmp = kde_tmp(y_grid) * vals_tmp.size  # convert PDF to expected counts
                global_density_max = max(global_density_max, float(np.max(dens_tmp)))
            except Exception:
                hist_tmp, _ = np.histogram(vals_tmp, bins=40, range=(0.0, 1.0), density=False)  # counts
                if hist_tmp.size:
                    global_density_max = max(global_density_max, float(hist_tmp.max()))

    for i, cat in enumerate(CATEGORY_ORDER, start=1):
        d = dist[cat]
        core = d["values_core"]
        all_vals = d["values_all"]

        # Orientation-based face color, recurrence-based hatch
        face = CATEGORY_FACE[cat]
        if cat.startswith("Single-event"):
            hatch_pat  = "."
            hatch_edge = OVERLAY_SINGLE
        else:
            hatch_pat  = "//"
            hatch_edge = OVERLAY_RECUR

        # Symmetric half-violins with GLOBAL scaling + hatch overlay
        # Half-violin + half-box; jitter on the BOX side (raincloud-style)
        side_violin = "left" if "direct" in cat else "right"
        side_box = "right" if side_violin == "left" else "left"
        offset = 0.08 if side_box == "right" else -0.08
        sign = 1.0 if side_box == "right" else -1.0
        x_center = (i + offset) + sign * 0.03

        # Half-box shifted to the opposite side of the violin
        median, q1, q3 = d["box_stats"]
        if not np.isnan(median):
            cx = i + offset
            ax.plot([cx - 0.09, cx + 0.09], [median, median], color="#333333", lw=0.6, zorder=3)
            ax.plot([cx, cx], [q1, q3], color="#333333", lw=0.6, zorder=3)
            ax.plot([cx - 0.06, cx + 0.06], [q1, q1], color="#333333", lw=0.6, zorder=3)
            ax.plot([cx - 0.06, cx + 0.06], [q3, q3], color="#333333", lw=0.6, zorder=3)

        # Jitter points: NOT over the violin. Nudge just past the box on the box side.
        if all_vals.size > 0:
            mask_at1 = (all_vals == 1.0)
            vals_non1 = all_vals[~mask_at1]

            if vals_non1.size > 0:
                x_jit = x_center + (np.random.rand(vals_non1.size) - 0.5) * 0.04

                ax.scatter(
                    x_jit, vals_non1,
                    s=12, alpha=0.9, color=face,
                    edgecolor="white", linewidths=0.4, zorder=2.4,
                )

            n_at1_pts = int(mask_at1.sum())
            if n_at1_pts > 0:
                rect_w = 0.44
                rect_h = 0.045
                rect_y0 = 1.005

                rect = mpatches.Rectangle(
                    (x_center - rect_w / 2, rect_y0),
                    rect_w, rect_h,
                    facecolor="none",
                    edgecolor=face,
                    linewidth=0.8,
                    zorder=1.8,
                )
                ax.add_patch(rect)

                x_rect = (x_center - rect_w / 2) + np.random.rand(n_at1_pts) * rect_w
                y_rect = rect_y0 + np.random.rand(n_at1_pts) * rect_h
                ax.scatter(
                    x_rect, y_rect,
                    s=12, alpha=0.9, color=face,
                    edgecolor="white", linewidths=0.4, zorder=2.2,
                )

        # Cap at 1.0: stacked dots + explicit counts
        n_at1 = d["n_at1"]
        label = f"N (CDSs) = {d['n_cds']}\n100% identical: {d['share_at_1']*100:.0f}%"
        ax.text(i, 1.065, label, ha="center", va="bottom", fontsize=8, color="#333333")

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis="both", labelsize=9)

    legend_entries = [
        ("Single-event, direct",   COLOR_DIRECT,   ".",  OVERLAY_SINGLE),
        ("Single-event, inverted", COLOR_INVERTED, ".",  OVERLAY_SINGLE),
        ("Recurrent, direct",      COLOR_DIRECT,   "//", OVERLAY_RECUR),
        ("Recurrent, inverted",    COLOR_INVERTED, "//", OVERLAY_RECUR),
    ]
    patches = [
        mpatches.Rectangle(
            (0, 0), 1, 1,
            facecolor=face, edgecolor=edge, hatch=hatch,
            linewidth=0.0, alpha=ALPHA_VIOLIN
        )
        for (_, face, hatch, edge) in legend_entries
    ]
    labels = [lbl for (lbl, *_rest) in legend_entries]

    leg = ax.legend(
        handles=patches,
        labels=labels,
        title="Category",
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        fontsize=8,
    )
    if leg and leg.get_title():
        leg.get_title().set_fontsize(8.5)

    fig.tight_layout()
    ensure_dir(outfile)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Volcano plot: Δ proportion identical (Inverted − Direct)
# =============================================================================



def prepare_volcano(gene_tests: pd.DataFrame, cds_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build volcano DF with:
      - recurrence: assigned STRICTLY from inv_info.tsv via (chr, start, end) locus match only
      - n_pairs_total: optional size, summed from cds_summary via (chr, start, end) locus match only

    No joins on gene_name or transcript_id for recurrence or sizing.
    """

    # ----------------- helpers -----------------
    def _canon_chr(x):
        s = str(x).strip()
        return s[3:] if s.lower().startswith("chr") else s

    def _parse_inv_id(inv_id_str):
        # expected like "7:5989046-6735643" (chr prefix ok; will strip)
        s = str(inv_id_str).strip()
        try:
            chrom, rest = s.split(":")
            start_s, end_s = rest.split("-")
            chrom_c = _canon_chr(chrom)
            start_i = int(float(start_s))
            end_i   = int(float(end_s))
            return chrom_c, start_i, end_i
        except Exception:
            return None, None, None

    def _pick_col(df, must_contain, avoid=None):
        lc = [c for c in df.columns if must_contain in c.lower()]
        if avoid:
            lc = [c for c in lc if avoid not in c.lower()]
        return lc[0] if lc else None

    # ----------------- start -----------------
    print("[prepare_volcano] START")
    print(f"[prepare_volcano] gene_tests rows: {len(gene_tests)}, columns: {list(gene_tests.columns)}")
    print(f"[prepare_volcano] cds_summary rows: {len(cds_summary)}, columns: {list(cds_summary.columns)}")

    # ---- parse loci from gene_tests.inv_id ----
    gt = gene_tests.copy()
    parsed = gt["inv_id"].apply(_parse_inv_id)
    gt["chr_c"]   = parsed.apply(lambda t: t[0])
    gt["start_c"] = parsed.apply(lambda t: t[1])
    gt["end_c"]   = parsed.apply(lambda t: t[2])
    bad_gt = gt[gt[["chr_c","start_c","end_c"]].isna().any(axis=1)]
    print(f"[prepare_volcano] gene_tests locus parse: bad rows={len(bad_gt)}")
    if not bad_gt.empty:
        print(bad_gt[["gene_name","transcript_id","inv_id"]].head(5).to_string(index=False))

    # ---- read inv_info.tsv and normalize columns ----
    inv_info_path = "inv_info.tsv"
    inv_df = safe_read_tsv(inv_info_path)
    print(f"[prepare_volcano] inv_info path: {inv_info_path}")
    print(f"[prepare_volcano] inv_info columns: {list(inv_df.columns)}")

    col_chr  = _pick_col(inv_df, "chrom")
    col_start= _pick_col(inv_df, "start")
    col_end  = _pick_col(inv_df, "end")
    # prefer exact consensus col name, else fall back to anything containing 'consensus'
    if "0_single_1_recur_consensus" in inv_df.columns:
        col_cons = "0_single_1_recur_consensus"
    else:
        cand = [c for c in inv_df.columns if "consensus" in c.lower()]
        col_cons = cand[0] if cand else None

    if not all([col_chr, col_start, col_end, col_cons]):
        raise ValueError(f"[prepare_volcano] inv_info missing required cols. "
                         f"Found chr={col_chr}, start={col_start}, end={col_end}, consensus={col_cons}")

    inv_norm = inv_df[[col_chr, col_start, col_end, col_cons]].copy()
    inv_norm["chr_c"]   = inv_norm[col_chr].apply(_canon_chr)
    inv_norm["start_c"] = pd.to_numeric(inv_norm[col_start], errors="coerce").astype("Int64")
    inv_norm["end_c"]   = pd.to_numeric(inv_norm[col_end],   errors="coerce").astype("Int64")
    inv_norm["consensus_mode"] = pd.to_numeric(inv_norm[col_cons], errors="coerce")
    inv_bad = inv_norm[inv_norm[["chr_c","start_c","end_c","consensus_mode"]].isna().any(axis=1)]
    print(f"[prepare_volcano] inv_info parse: bad rows={len(inv_bad)}")

    # collapse to unique locus -> consensus (mode over any duplicates)
    rec_map = (inv_norm
               .dropna(subset=["chr_c","start_c","end_c","consensus_mode"])
               .groupby(["chr_c","start_c","end_c"])["consensus_mode"]
               .agg(lambda s: pd.to_numeric(s, errors="coerce").dropna().astype(int).mode().iloc[0]
                    if not pd.to_numeric(s, errors="coerce").dropna().empty else np.nan)
               .reset_index())
    rec_map["recurrence"] = rec_map["consensus_mode"].map({0:"SE", 1:"REC"})

    # ---- coverage diagnostics: loci in GT vs inv_info ----
    gt_loci = set(tuple(x) for x in gt[["chr_c","start_c","end_c"]].dropna().values.tolist())
    inv_loci= set(tuple(x) for x in rec_map[["chr_c","start_c","end_c"]].values.tolist())
    print(f"[prepare_volcano] unique loci in gene_tests: {len(gt_loci)}")
    print(f"[prepare_volcano] unique loci in inv_info:   {len(inv_loci)}")
    miss_gt_in_inv = gt_loci - inv_loci
    print(f"[prepare_volcano] loci in gene_tests but NOT in inv_info: {len(miss_gt_in_inv)}")
    if miss_gt_in_inv:
        ex = list(miss_gt_in_inv)[:10]
        print("[prepare_volcano] examples:", ", ".join([f"{c}:{s}-{e}" for (c,s,e) in ex]))

    # ---- merge recurrence to gene_tests by locus ONLY ----
    out = gt.merge(rec_map[["chr_c","start_c","end_c","recurrence"]],
                   on=["chr_c","start_c","end_c"], how="left")

    n_rec = out["recurrence"].notna().sum()
    print(f"[prepare_volcano] recurrence assigned: {n_rec}/{len(out)} rows")
    print(f"[prepare_volcano] recurrence counts:", out["recurrence"].value_counts(dropna=False).to_dict())

    # ---- optional: size from cds_summary by locus ONLY ----
    cs = cds_summary.copy()
    cs["chr_c"]   = cs["chr"].apply(_canon_chr)
    cs["start_c"] = pd.to_numeric(cs["inv_start"], errors="coerce").astype("Int64")
    cs["end_c"]   = pd.to_numeric(cs["inv_end"],   errors="coerce").astype("Int64")

    size_map = (cs.dropna(subset=["chr_c","start_c","end_c","n_pairs"])
                  .groupby(["chr_c","start_c","end_c","phy_group"])["n_pairs"]
                  .sum().reset_index())
    size_piv = (size_map
                .pivot_table(index=["chr_c","start_c","end_c"],
                             columns="phy_group", values="n_pairs", aggfunc="sum")
                .rename(columns={0:"n_pairs_direct", 1:"n_pairs_inverted"})
                .reset_index())
    size_piv["n_pairs_total"] = size_piv[["n_pairs_direct","n_pairs_inverted"]].sum(axis=1, min_count=1)

    out = out.merge(size_piv[["chr_c","start_c","end_c","n_pairs_total"]],
                    on=["chr_c","start_c","end_c"], how="left")

    n_size_nan = out["n_pairs_total"].isna().sum()
    print(f"[prepare_volcano] rows with NaN n_pairs_total AFTER locus-only merge: {n_size_nan}")
    if n_size_nan:
        miss = out[out["n_pairs_total"].isna()][["chr_c","start_c","end_c"]].drop_duplicates()
        ex = miss.head(10).itertuples(index=False, name=None)
        ex_str = ", ".join([f"{c}:{s}-{e}" for (c,s,e) in ex])
        print("[prepare_volcano] examples (size missing):", ex_str if ex_str else "(none)")

    # ---- final quick ranges for sanity ----
    if "q_value" in out.columns:
        qv = pd.to_numeric(out["q_value"], errors="coerce")
        print(f"[prepare_volcano] q_value: n={qv.notna().sum()}, min={qv.min()}, max={qv.max()}")
    if "delta" in out.columns:
        dv = pd.to_numeric(out["delta"], errors="coerce")
        print(f"[prepare_volcano] delta:   n={dv.notna().sum()}, min={dv.min()}, max={dv.max()}")

    print("[prepare_volcano] END")
    return out


def plot_cds_conservation_volcano(df: pd.DataFrame, outfile: str):
    """
    Render the conservation volcano. Detect label overlaps and resolve them by
    moving only ONE label per iteration, alternating:
      - Step 1a (vertical): pick ONE overlapping pair; move the label that is already
        higher in that pair upward just enough to clear the overlap.
      - Step 1b (horizontal): rescan; pick ONE overlapping pair; move the label that
        is already left in that pair leftward just enough to clear the overlap.
    Repeat (1a -> rescan -> 1b -> rescan -> ...) until no overlaps remain.

    Data points and axes are NEVER adjusted by the overlap solver. Only label
    offsets (in offset points) are changed.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    ax.set_xlabel("Δ proportion identical (Inverted − Direct)")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{BH}\;q)$")
    ax.set_facecolor("white")

    x = pd.to_numeric(df["delta"], errors="coerce")
    q = pd.to_numeric(df["q_value"], errors="coerce").clip(1e-22, 1.0)
    with np.errstate(divide="ignore"):
        y = -np.log10(q.to_numpy())

    # Point sizes by total pairs (scaled smoothly)
    sizes_raw = pd.to_numeric(df.get("n_pairs_total", pd.Series(dtype=float)), errors="coerce").fillna(0)
    size_scale_max = float(sizes_raw.max()) if not sizes_raw.empty else 0.0
    size_legend_info = None
    if size_scale_max > 0:
        sizes = 28 + 220 * (sizes_raw / size_scale_max)
    else:
        sizes = np.full(len(df), 60.0)

    # Colors by recurrence (Single-event: deep red, Recurrent: forest green)
    rec = df["recurrence"].astype(str)
    color_map = {
        "SE": "#8B0000",   # Single-event: deep red
        "REC": "#014421",  # Recurrent: deep forest green
    }
    colors = rec.map(color_map).fillna("#7f7f7f")

    def _format_pairs_label(val: float) -> str:
        if val >= 1000:
            return f"{int(round(val)):,}"
        if val >= 10:
            return str(int(round(val)))
        if val >= 1:
            txt = f"{val:.1f}"
        else:
            txt = f"{val:.2f}"
        return txt.rstrip("0").rstrip(".")

    def _size_legend_levels(scale_max: float) -> list[float]:
        if not np.isfinite(scale_max) or scale_max <= 0:
            return []
        fractions = np.array([0.35, 0.65, 1.0])
        levels = (fractions * scale_max).tolist()
        eps = scale_max * 1e-3
        levels = [max(val, eps) for val in levels]
        for idx in range(1, len(levels)):
            if levels[idx] <= levels[idx - 1]:
                levels[idx] = min(scale_max, levels[idx - 1] + eps)
        return levels

    # Scatter (data only)
    ax.scatter(
        x, y, s=sizes, c=colors, alpha=0.5,
        edgecolor="white", linewidths=0.7, zorder=3,
    )

    if size_scale_max > 0:
        legend_levels = _size_legend_levels(size_scale_max)
        size_handles = []
        size_labels = []
        for lvl in legend_levels:
            lvl = float(np.clip(lvl, 1e-9, size_scale_max))
            handle_size = 28 + 220 * (lvl / size_scale_max)
            size_handles.append(
                ax.scatter(
                    [], [],
                    s=handle_size,
                    facecolor="#bdbdd0",
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.5,
                )
            )
            size_labels.append(f"{_format_pairs_label(lvl)} pairs")
        if size_handles and size_labels:
            size_legend_info = (size_handles, size_labels)

    # Axes limits from DATA ONLY (never touched again)
    if x.notna().any():
        x_lim = float(np.nanmax(np.abs(x))) * 1.1 + 0.05
    else:
        x_lim = 1.0
    ax.set_xlim(-x_lim, x_lim)
    y_max = np.nanmax(y) if np.isfinite(np.nanmax(y)) else 1.0
    ax.set_ylim(0, y_max * 1.05 + 0.3)

    left, right = ax.get_xlim()
    # ax.axvspan(left, 0, color=COLOR_DIRECT, alpha=0.06, zorder=0)      # REMOVED
    # ax.axvspan(0, right, color=COLOR_INVERTED, alpha=0.05, zorder=0)   # REMOVED
    ax.axvline(0, color="#595959", linestyle="--", linewidth=1.0, zorder=1)

    # FDR 5% reference line (~1.301)
    thresh_y = -math.log10(0.05)
    ax.axhline(thresh_y, linestyle="--", color="#b0b0b0", linewidth=1.0, zorder=1)
    ax.text(
        left + 0.02 * (right - left),
        thresh_y + 0.05,
        "FDR 5%",
        va="bottom", ha="left",
        fontsize=8, color="#666666",
    )

    # --------------------------
    # Select labels
    # --------------------------
    sig = df.copy()
    sig["delta_val"] = pd.to_numeric(sig["delta"], errors="coerce")
    sig["q_val"] = pd.to_numeric(sig["q_value"], errors="coerce")
    sig = sig[sig["q_val"] <= 0.05].dropna(subset=["delta_val", "q_val"])

    label_rows = []
    if not sig.empty:
        top_by_q    = sig.nsmallest(8, "q_val")
        extreme_pos = sig.nlargest(4, "delta_val")
        extreme_neg = sig.nsmallest(4, "delta_val")
        label_df = pd.concat([top_by_q, extreme_pos, extreme_neg]).drop_duplicates(subset="gene_name")
        # Mild sort for stable placement; exact order doesn't matter for solver
        label_rows = list(label_df.sort_values(["q_val", "delta_val"]).itertuples(index=False))

    # -----------------------------------------
    # Place labels (offset points) then resolve overlaps
    # -----------------------------------------
    annotations = []
    if label_rows:
        for row in label_rows:
            rowd = row._asdict() if hasattr(row, "_asdict") else dict(row)
            x0 = float(rowd["delta_val"])
            y0 = -math.log10(max(float(rowd["q_val"]), 1e-22))
            name = str(rowd["gene_name"])
            rec_mode = str(rowd.get("recurrence", ""))
            txt_color = color_map.get(rec_mode, "#555555")

            dx = 6 if x0 >= 0 else -6
            ha = "left" if x0 >= 0 else "right"

            ann = ax.annotate(
                name,
                xy=(x0, y0), xycoords="data",
                xytext=(dx, 2), textcoords="offset points",
                fontsize=9,               
                ha=ha, 
                va="bottom", 
                color="black",                
                arrowprops=dict(arrowstyle="-", color="black", linewidth=0.8),
                zorder=5,
                annotation_clip=False,
            )
            annotations.append(ann)

        # Helpers -------------------------------------------------------------
        def _draw_and_renderer():
            fig.canvas.draw()
            return fig.canvas.get_renderer()

        def _bboxes(renderer, expand=(1.0, 1.0)):
            return [ann.get_window_extent(renderer=renderer).expanded(expand[0], expand[1]) for ann in annotations]

        def _overlap_pairs(bbs):
            """Return list of (i,j, area_px, w_px, h_px) for overlapping pairs, sorted by area desc."""
            pairs = []
            n = len(bbs)
            for i in range(n):
                bi = bbs[i]
                for j in range(i+1, n):
                    bj = bbs[j]
                    w = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                    h = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                    if w > 0 and h > 0:
                        pairs.append((i, j, w * h, w, h))
            pairs.sort(key=lambda t: t[2], reverse=True)
            return pairs

        # Cap movements to prevent runaway offsets that fling labels off-canvas.
        MAX_OFFSET_PT = 144.0  # total offset cap (±2 inches)
        MAX_STEP_PT   = 24.0   # per-step cap (<= 24 pt)

        def _px_to_pt(px):
            return px * 72.0 / fig.dpi

        def _clamp_offset(ox, oy):
            ox = float(np.clip(ox, -MAX_OFFSET_PT, MAX_OFFSET_PT))
            oy = float(np.clip(oy, -MAX_OFFSET_PT, MAX_OFFSET_PT))
            return ox, oy

        def _move_up_min(j_idx, dy_px):
            """Move ONE label upward by a bounded amount; clamp offsets to avoid runaway."""
            ox, oy = annotations[j_idx].get_position()
            step_pt = min(_px_to_pt(dy_px + 0.5), MAX_STEP_PT)
            newx, newy = ox, oy + step_pt
            newx, newy = _clamp_offset(newx, newy)
            annotations[j_idx].set_position((newx, newy))

        def _move_down_min(j_idx, dy_px):
            """Fallback: move ONE label downward by a bounded amount."""
            ox, oy = annotations[j_idx].get_position()
            step_pt = min(_px_to_pt(dy_px + 0.5), MAX_STEP_PT)
            newx, newy = ox, oy - step_pt
            newx, newy = _clamp_offset(newx, newy)
            annotations[j_idx].set_position((newx, newy))

        def _move_left_min(j_idx, dx_px):
            """Move ONE label left by a bounded amount; clamp offsets to avoid runaway."""
            ox, oy = annotations[j_idx].get_position()
            step_pt = min(_px_to_pt(dx_px + 0.5), MAX_STEP_PT)
            newx, newy = ox - step_pt, oy
            newx, newy = _clamp_offset(newx, newy)
            annotations[j_idx].set_position((newx, newy))

        def _move_right_min(j_idx, dx_px):
            """Fallback: move ONE label right by a bounded amount."""
            ox, oy = annotations[j_idx].get_position()
            step_pt = min(_px_to_pt(dx_px + 0.5), MAX_STEP_PT)
            newx, newy = ox + step_pt, oy
            newx, newy = _clamp_offset(newx, newy)
            annotations[j_idx].set_position((newx, newy))

        # Alternate vertical / horizontal moves, ONE PAIR per step -------------
        do_vertical = True
        max_iters = 5  # hard cap to guarantee termination

        for _ in range(max_iters):
            renderer = _draw_and_renderer()
            bbs = _bboxes(renderer, expand=(1.0, 1.0))
            pairs = _overlap_pairs(bbs)

            if not pairs:
                break  # done

            # Pick ONE pair (largest overlap)
            i, j, area, w_px, h_px = pairs[0]
            bi, bj = bbs[i], bbs[j]

            if do_vertical:
                # Prefer to move the higher label UP, but if it's at the cap, move the other DOWN.
                idx_pref = i if bi.y1 >= bj.y1 else j
                other    = j if idx_pref == i else i
                ox, oy = annotations[idx_pref].get_position()
                step_pt = min(_px_to_pt(h_px + 0.5), MAX_STEP_PT)
                if oy + step_pt > MAX_OFFSET_PT - 1e-6:
                    _move_down_min(other, h_px)
                else:
                    _move_up_min(idx_pref, h_px)
                do_vertical = False
            else:
                # Prefer to move the left label LEFT, but if it's at the cap, move the other RIGHT.
                idx_pref = i if bi.x0 <= bj.x0 else j
                other    = j if idx_pref == i else i
                ox, oy = annotations[idx_pref].get_position()
                step_pt = min(_px_to_pt(w_px + 0.5), MAX_STEP_PT)
                if ox - step_pt < -MAX_OFFSET_PT + 1e-6:
                    _move_right_min(other, w_px)
                else:
                    _move_left_min(idx_pref, w_px)
                do_vertical = True

        _draw_and_renderer()

    proxy_se = mpatches.Patch(facecolor=color_map["SE"], edgecolor="none", alpha=0.5, label="Single-event")
    proxy_rec = mpatches.Patch(facecolor=color_map["REC"], edgecolor="none", alpha=0.5, label="Recurrent")
    rec_legend = ax.legend(
        handles=[proxy_se, proxy_rec],
        frameon=False,
        title="Recurrence",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(rec_legend)
    if rec_legend.get_title():
        rec_legend.get_title().set_fontsize(8.5)

    if size_legend_info:
        size_handles, size_labels = size_legend_info
        size_legend = ax.legend(
            size_handles,
            size_labels,
            title="Total pairs (circle size)",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.62),
            borderaxespad=0.0,
            labelspacing=0.7,
            handletextpad=1.2,
            scatterpoints=1,
            fontsize=8,
            bbox_transform=ax.transAxes,
        )
        ax.add_artist(size_legend)
        if size_legend.get_title():
            size_legend.get_title().set_fontsize(8.5)


    for side in ['bottom', 'left']:
        ax.spines[side].set_color('#8a8a8a')

    fig.tight_layout()
    ensure_dir(outfile)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# MAPT CDS polymorphism heatmap (polymorphic sites only)
# =============================================================================

def detect_fixed_columns(seqs_D: list, seqs_I: list, threshold: float = FIXED_DIFF_UNANIMITY_THRESHOLD):
    if not seqs_D or not seqs_I:
        return []
    m = len(seqs_D[0])
    for s in seqs_D + seqs_I:
        if len(s) != m:
            raise ValueError("All sequences must have equal length for fixed-difference detection.")
    fixed = []
    for j in range(m):
        col_D = [s[j] for s in seqs_D]
        col_I = [s[j] for s in seqs_I]
        base_D, frac_D = _consensus_base_and_frac(col_D)
        base_I, frac_I = _consensus_base_and_frac(col_I)
        if base_D is None or base_I is None:
            continue
        if frac_D >= threshold and frac_I >= threshold and base_D != base_I:
            fixed.append(j)
    return fixed

def _consensus_base_and_frac(col_list: list):
    counts = Counter(col_list)
    if not counts:
        return None, 0.0
    base, cnt = counts.most_common(1)[0]
    frac = cnt / len(col_list)
    return base, frac


def plot_fixed_diff_panel(ax, phyD, phyI, gene_name: str, inv_id: str, threshold: float):
    """
    Polymorphism heatmap panel with fixes:
      • Brackets now face inward and both use the SAME x-position (aligned).
      • X tick labels are placed at integer column centers and align with columns.
      • "fixed" text label is added above each fixed triangle marker (triangle retained).
    """
    import numpy as _np
    from matplotlib.colors import ListedColormap as _ListedColormap
    from matplotlib.transforms import blended_transform_factory as _blend

    # ---------- helpers ----------
    def _consensus_base_and_frac(col_list: list):
        from collections import Counter as _Counter
        counts = _Counter(col_list)
        if not counts:
            return None, 0.0
        base, cnt = counts.most_common(1)[0]
        return base, cnt / len(col_list)

    def _detect_fixed_columns(seqs_D: list, seqs_I: list, thr: float):
        if not seqs_D or not seqs_I:
            return []
        m = len(seqs_D[0])
        for s in seqs_D + seqs_I:
            if len(s) != m:
                raise ValueError("All sequences must have equal length.")
        fixed = []
        for j in range(m):
            col_D = [s[j] for s in seqs_D]
            col_I = [s[j] for s in seqs_I]
            bD, fD = _consensus_base_and_frac(col_D)
            bI, fI = _consensus_base_and_frac(col_I)
            if bD is None or bI is None:
                continue
            if fD >= thr and fI >= thr and bD != bI:
                fixed.append(j)
        return fixed

    def _encode_no_gaps(seq_strs: list, cols_keep: list) -> _np.ndarray:
        arr = _np.zeros((len(seq_strs), len(cols_keep)), dtype=float)
        for i, s in enumerate(seq_strs):
            for k, j in enumerate(cols_keep):
                arr[i, k] = BASE_TO_IDX.get(s[j].upper(), 0)
        return arr

    def _insert_row_gaps(arr: _np.ndarray, nI: int, gap: int = 1, group_gap: int = 3) -> _np.ndarray:
        """Insert NaN spacer rows between all rows; bigger spacer between I and D groups."""
        r, c = arr.shape
        rows = []
        for i in range(r):
            rows.append(arr[i])
            if i < r - 1:
                n_spacers = group_gap if i == (nI - 1) else gap
                for _ in range(n_spacers):
                    rows.append(_np.full(c, _np.nan))
        return _np.vstack(rows)

    def _add_square_bracket(ax, y0, y1, label, *, x_axes=-0.055, tick_len_axes=0.018, lw=1.8):
        """
        Draw a black square/rectangle bracket just LEFT of the y-axis.
        x is in AXES coords (negative puts it outside the plot), y in DATA coords.
        Ticks now point INWARD (toward the plot).
        """
        trans = _blend(ax.transAxes, ax.transData)
        x = x_axes
        # vertical spine of bracket
        ax.plot([x, x], [y0, y1], color="black", lw=lw, transform=trans, clip_on=False, zorder=6)
        # end ticks (pointing inward, into the plot)
        ax.plot([x, x + tick_len_axes], [y0, y0], color="black", lw=lw, transform=trans, clip_on=False, zorder=6)
        ax.plot([x, x + tick_len_axes], [y1, y1], color="black", lw=lw, transform=trans, clip_on=False, zorder=6)
        # vertical label (black)
        ax.text(x - 0.006, (y0 + y1) / 2.0, label,
                transform=trans, va="center", ha="right",
                rotation=90, fontsize=12, fontweight="bold",
                color="black", clip_on=False, zorder=7)

    # ---------- get & alphabetically sort haplotype orders (labels hidden later) ----------
    namesD = sorted(list(phyD["seq_order"]), key=str)
    namesI = sorted(list(phyI["seq_order"]), key=str)
    seqsD = [phyD["seqs"].get(nm, "") for nm in namesD]
    seqsI = [phyI["seqs"].get(nm, "") for nm in namesI]
    if not seqsD or not seqsI:
        ax.text(0.5, 0.5, f"No sequences for {gene_name}", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return {"n_polymorphic": 0, "n_fixed": 0, "n_inverted": len(seqsI), "n_direct": len(seqsD)}

    # ---------- choose polymorphic columns across ALL haplotypes ----------
    all_seqs = seqsI + seqsD
    m_full = len(all_seqs[0])
    if any(len(s) != m_full for s in all_seqs):
        raise ValueError("Seq lengths differ between orientations.")
    cols_keep = [j for j in range(m_full) if len({s[j] for s in all_seqs}) > 1]
    if not cols_keep:
        ax.text(0.5, 0.5, f"{gene_name} — {inv_id}\nNo polymorphic CDS sites",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        return {"n_polymorphic": 0, "n_fixed": 0, "n_inverted": len(seqsI), "n_direct": len(seqsD)}

    # ---------- encode, stack (I over D), and add minimal row spacing ----------
    arr_I = _encode_no_gaps(seqsI, cols_keep)
    arr_D = _encode_no_gaps(seqsD, cols_keep)
    arr   = _np.vstack([arr_I, arr_D])
    nI, nD = len(seqsI), len(seqsD)

    # Thin-line look: no gaps between adjacent haplotypes; single-row gap between groups
    GAP_BETWEEN_ROWS  = 0
    GAP_BETWEEN_GROUP = 1
    arr_plot = _insert_row_gaps(arr, nI=nI, gap=GAP_BETWEEN_ROWS, group_gap=GAP_BETWEEN_GROUP)

    # NaN spacers render as white so the (now thin) group gap is visible
    _cmap = _ListedColormap(BASE_COLORS)
    _cmap.set_bad(color="white")


    ax.imshow(arr_plot, cmap=_cmap, vmin=0, vmax=3, aspect="auto",
              interpolation="nearest", origin="upper", zorder=1)

    # ---------- x-axis (positions), y-axis (hide labels entirely) ----------
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)

    # Label EVERY polymorphic position at integer column centers
    ncols = arr.shape[1]
    ax.set_xticks(_np.arange(ncols, dtype=int))
    ax.set_xticklabels([str(cols_keep[t] + 1) for t in range(ncols)], fontsize=12)
    ax.set_xlabel("Polymorphic CDS positions", fontsize=18)

    ax.set_yticks([])  # hide haplotype labels entirely
    ax.tick_params(axis="both", which="both", length=0)

    # ---------- mark fixed-difference columns ----------
    fixed_full = set(_detect_fixed_columns(seqsD, seqsI, thr=threshold))
    fixed_kept = [k for k, j in enumerate(cols_keep) if j in fixed_full]

    for k in fixed_kept:
        ax.axvline(k - 0.5, color="#111111", linewidth=1.2, zorder=3)
        ax.axvline(k + 0.5, color="#111111", linewidth=1.2, zorder=3)

    if fixed_kept:
        # draw triangles and place the word "fixed" above each triangle
        tri_y = -0.35
        label_y = -0.95  # a bit above the triangle; outside plot; included via bbox_inches="tight"
        ax.scatter(fixed_kept, _np.full(len(fixed_kept), tri_y),
                   marker="v", s=36, color="#111111", edgecolor="none",
                   clip_on=False, zorder=4)
        for k in fixed_kept:
            ax.text(k, label_y, "fixed", ha="center", va="bottom",
                    fontsize=14, color="#111111", clip_on=False, zorder=5)


    # ---------- compute DATA y extents for each bracket in the gapped image ----------
    last_inv_row_idx  = (nI - 1) * (GAP_BETWEEN_ROWS + 1) if nI > 0 else -1
    first_dir_row_idx = last_inv_row_idx + 1 + GAP_BETWEEN_GROUP
    last_dir_row_idx  = first_dir_row_idx + (nD - 1) * (GAP_BETWEEN_ROWS + 1) if nD > 0 else last_inv_row_idx

    # ---------- draw square brackets (same x position; facing inward) ----------
    BRACKET_X = -0.055  # same for both labels so they align horizontally
    _add_square_bracket(ax, y0=-0.5,                  y1=last_inv_row_idx + 0.5, label="Inverted haplotypes",
                        x_axes=BRACKET_X, tick_len_axes=0.018, lw=1.8)
    _add_square_bracket(ax, y0=first_dir_row_idx - 0.5, y1=last_dir_row_idx + 0.5, label="Direct haplotypes",
                        x_axes=BRACKET_X, tick_len_axes=0.018, lw=1.8)

    return {
        "n_polymorphic": int(arr.shape[1]),
        "n_fixed": int(len(fixed_kept)),
        "n_inverted": int(nI),
        "n_direct": int(nD),
    }

def plot_mapt_polymorphism_heatmap(cds_summary: pd.DataFrame, pairs_index: pd.DataFrame, outfile: str):
    sub = cds_summary[(cds_summary["gene_name"] == MAPT_GENE)]
    if sub.empty:
        warnings.warn(f"No CDS summary entries found for fixed-diff gene {MAPT_GENE}")
        return

    candidates = []
    for inv_id, grp in sub.groupby("inv_id"):
        have_D = not grp[grp["phy_group"] == 0].empty
        have_I = not grp[grp["phy_group"] == 1].empty
        if have_D and have_I:
            total_nseq = int(grp["n_sequences"].sum())
            candidates.append((inv_id, total_nseq))
    if not candidates:
        warnings.warn(f"No inversion with both orientations for gene {MAPT_GENE}")
        return

    inv_id, _ = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    locus_rows = sub[sub["inv_id"] == inv_id]
    rowD = locus_rows[locus_rows["phy_group"] == 0].iloc[0]
    rowI = locus_rows[locus_rows["phy_group"] == 1].iloc[0]
    phyD = read_phy(pairs_index.loc[str(rowD["filename"]), "phy_path"])
    phyI = read_phy(pairs_index.loc[str(rowI["filename"]), "phy_path"])

    nrows = len(phyI["seq_order"]) + len(phyD["seq_order"])
    # Smaller per-row height so any spacers render as a thin line
    panel_height = max(2.5, nrows * 0.14)

    # Narrower figure => slimmer columns
    fig, ax = plt.subplots(figsize=(10.0, panel_height + 2.0))

    # Render the panel (now adds "fixed" labels and uses inward/same-x brackets)
    stats_info = plot_fixed_diff_panel(
        ax,
        phyD,
        phyI,
        MAPT_GENE,
        inv_id,
        threshold=FIXED_DIFF_UNANIMITY_THRESHOLD,
    )

    # Move the Base legend OUTSIDE the axes so it does not overlap the plot
    base_handles = [mpatches.Patch(color=BASE_COLORS[i], label=b) for b, i in BASE_TO_IDX.items()]
    fig.legend(
            handles=base_handles,
            loc="upper left",
            bbox_to_anchor=(0.88, 1.0),
            frameon=False,
            fontsize=14,
            title="Base",
            title_fontsize=20,
            borderaxespad=0.0,
        )

    # Leave space on the right for the external legend
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])

    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)



# =============================================================================
# Main
# =============================================================================

def main():
    cds_summary = load_cds_summary()
    gene_tests  = load_gene_tests()
    pairs_index = build_pairs_and_phy_index(cds_summary)

    plot_proportion_identical_violin(cds_summary, VIOLIN_PLOT_FILE)

    volcano_df = prepare_volcano(gene_tests, cds_summary)
    plot_cds_conservation_volcano(volcano_df, VOLCANO_PLOT_FILE)

    plot_mapt_polymorphism_heatmap(cds_summary, pairs_index, MAPT_HEATMAP_FILE)

if __name__ == "__main__":
    main()
