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

# =============================================================================
# Global configuration
# =============================================================================

matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RANDOM_SEED = 2024
np.random.seed(RANDOM_SEED)

# Orientation colors (high contrast)
COLOR_DIRECT   = "#1f77b4"  # blue
COLOR_INVERTED = "#d62728"  # red

# Category palette (Single-event, direct, Single-event, inverted, Recurrent, direct, Recurrent, inverted) — use four highly distinct colors
CATEGORY_ORDER = ["Single-event, direct", "Single-event, inverted", "Recurrent, direct", "Recurrent, inverted"]
CATEGORY_COLORS = {
    "Single-event, direct": "#6a3d9a",  # purple
    "Single-event, inverted": "#ff7f00",  # orange
    "Recurrent, direct": "#1f78b4", # blue
    "Recurrent, inverted": "#33a02c", # green
}

# Base colors for raw haplotype plots: A,C,G,T (no gap color used)
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
BASE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]  # A,C,G,T
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
    recurrence = "SE" if int(consensus) == 0 else "REC"
    orientation = "D" if int(phy_group) == 0 else "I"
    return f"{recurrence}-{orientation}"

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

def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length strings.")
    return sum(1 for x, y in zip(a, b) if x != y)

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

def draw_half_violin(ax, y_vals, center_x, width=0.4, side="left", bins=40, clip=(0.0, 1.0), facecolor="#cccccc", alpha=0.6):
    y_vals = np.asarray(y_vals, dtype=float)
    y_vals = y_vals[(y_vals >= clip[0]) & (y_vals <= clip[1])]
    if y_vals.size == 0:
        return
    hist, edges = np.histogram(y_vals, bins=bins, range=clip, density=True)
    if hist.size >= 3:
        hist = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    if hist.max() > 0:
        hist = hist / hist.max()
    xs   = hist * width
    mids = 0.5 * (edges[:-1] + edges[1:])
    xcoords = center_x - xs if side == "left" else center_x + xs
    ax.fill_betweenx(mids, center_x, xcoords, color=facecolor, alpha=alpha, linewidth=0)
def plot_proportion_identical_violin(cds_summary: pd.DataFrame, outfile: str):
    dist = compute_group_distributions(cds_summary)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.set_facecolor("#f9f9f9")
    ax.set_xlabel("Inversion class")
    ax.set_ylabel("Proportion of identical CDS pairs")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlim(0.5, len(CATEGORY_ORDER) + 0.5)

    positions = range(1, len(CATEGORY_ORDER) + 1)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(CATEGORY_ORDER)

    # Background shading separates single-event vs recurrent classes
    ax.axvspan(0.5, 2.5, color="#ede7f6", alpha=0.18, zorder=0)
    ax.axvspan(2.5, 4.5, color="#e6f4ea", alpha=0.18, zorder=0)
    ax.text(1.5, 1.05, "Single-event", ha="center", va="bottom", fontsize=8, color="#5d3a9b")
    ax.text(3.5, 1.05, "Recurrent", ha="center", va="bottom", fontsize=8, color="#1b8132")

    for i, cat in enumerate(CATEGORY_ORDER, start=1):
        d = dist[cat]
        core = d["values_core"]
        all_vals = d["values_all"]

        # Symmetric half-violins with subtle transparency
        draw_half_violin(ax, core, i, width=0.36, side="left", facecolor=CATEGORY_COLORS[cat], alpha=0.40)
        draw_half_violin(ax, core, i, width=0.36, side="right", facecolor=CATEGORY_COLORS[cat], alpha=0.40)

        # Box (median & IQR)
        median, q1, q3 = d["box_stats"]
        if not np.isnan(median):
            ax.plot([i - 0.18, i + 0.18], [median, median], color="#333333", lw=1.0, zorder=3)
            ax.plot([i, i], [q1, q3], color="#333333", lw=1.0, zorder=3)
            ax.plot([i - 0.12, i + 0.12], [q1, q1], color="#333333", lw=1.0, zorder=3)
            ax.plot([i - 0.12, i + 0.12], [q3, q3], color="#333333", lw=1.0, zorder=3)

        # Jitter points (all values, including 1.0)
        if all_vals.size > 0:
            x_jit = i + (np.random.rand(all_vals.size) - 0.5) * 0.24
            ax.scatter(
                x_jit,
                all_vals,
                s=12,
                alpha=0.7,
                color=CATEGORY_COLORS[cat],
                edgecolor="white",
                linewidths=0.4,
                zorder=2,
            )

        # Cap at 1.0: stacked dots + explicit counts
        n_at1 = d["n_at1"]
    
        label = f"CDS: {d['n_cds']}\n100% identical: {n_at1} ({d['share_at_1']*100:.0f}%)"
        ax.text(i, 1.065, label, ha="center", va="bottom", fontsize=8, color="#333333")

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#cfcfcf", alpha=0.7)
    ax.tick_params(axis="both", labelsize=9)

    patches = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    leg = ax.legend(
        handles=patches,
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
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    ax.set_xlabel("Δ proportion identical (Inverted − Direct)")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{BH}\;q)$")
    ax.set_facecolor("#f9f9f9")

    # Coordinates (cap q to avoid inf on -log10)
    x = pd.to_numeric(df["delta"], errors="coerce")
    q = pd.to_numeric(df["q_value"], errors="coerce").clip(1e-22, 1.0)
    with np.errstate(divide="ignore"):
        y = -np.log10(q.to_numpy())

    # Point sizes by total pairs (scaled smoothly)
    sizes_raw = pd.to_numeric(df.get("n_pairs_total", pd.Series(dtype=float)), errors="coerce").fillna(0)
    if sizes_raw.max() > 0:
        sizes = 28 + 220 * (sizes_raw / sizes_raw.max())
    else:
        sizes = np.full_like(y, 40.0, dtype=float)

    # Colors: trust the recurrence column provided by prepare_volcano
    rec = df["recurrence"].astype(str)
    color_map = {"SE": CATEGORY_COLORS["Single-event, inverted"], "REC": CATEGORY_COLORS["Recurrent, inverted"]}
    colors = rec.map(color_map).fillna("#7f7f7f")

    ax.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        alpha=0.88,
        edgecolor="white",
        linewidths=0.7,
        zorder=3,
    )

    # Symmetric x limits with breathing room for annotations
    if x.notna().any():
        x_lim = float(np.nanmax(np.abs(x))) * 1.1 + 0.05
    else:
        x_lim = 1.0
    ax.set_xlim(-x_lim, x_lim)
    y_max = np.nanmax(y) if np.isfinite(np.nanmax(y)) else 1.0
    ax.set_ylim(0, y_max * 1.05 + 0.3)

    left, right = ax.get_xlim()
    ax.axvspan(left, 0, color=COLOR_DIRECT, alpha=0.06, zorder=0)
    ax.axvspan(0, right, color=COLOR_INVERTED, alpha=0.05, zorder=0)
    ax.axvline(0, color="#595959", linestyle="--", linewidth=1.0, zorder=1)

    # Significance threshold at q=0.05
    thresh_y = -math.log10(0.05)
    ax.axhline(thresh_y, linestyle="--", color="#b0b0b0", linewidth=1.0, zorder=1)
    ax.text(
        left + 0.02 * (right - left),
        thresh_y + 0.05,
        "FDR 5%",
        va="bottom",
        ha="left",
        fontsize=8,
        color="#666666",
    )

    # --------------------------
    # Label selection (as before)
    # --------------------------
    sig = df.copy()
    sig["delta_val"] = pd.to_numeric(sig["delta"], errors="coerce")
    sig["q_val"] = pd.to_numeric(sig["q_value"], errors="coerce")
    sig = sig[sig["q_val"] <= 0.05].dropna(subset=["delta_val", "q_val"])
    label_rows = []
    if not sig.empty:
        top_by_q   = sig.nsmallest(8, "q_val")
        extreme_pos = sig.nlargest(4, "delta_val")
        extreme_neg = sig.nsmallest(4, "delta_val")
        label_df = pd.concat([top_by_q, extreme_pos, extreme_neg]).drop_duplicates(subset="gene_name")
        label_rows = list(label_df.itertuples(index=False))

    # -----------------------------------------
    # Place labels then repel to remove overlap
    # -----------------------------------------
    annotations = []
    if label_rows:
        # Initial placement
        for row in label_rows:
            # Access by attribute if present; otherwise via dict-like
            if hasattr(row, "_asdict"):
                rowd = row._asdict()
            else:
                rowd = dict(row)
            x0 = float(rowd["delta_val"])
            y0 = -math.log10(max(float(rowd["q_val"]), 1e-22))
            name = str(rowd["gene_name"])
            rec_mode = str(rowd.get("recurrence", ""))
            txt_color = color_map.get(rec_mode, "#555555")

            x_offset = 0.35 if x0 >= 0 else -0.35
            align = "left" if x0 >= 0 else "right"

            ann = ax.annotate(
                name,
                xy=(x0, y0),
                xytext=(x0 + x_offset, y0 + 0.25),
                textcoords="data",
                fontsize=8,
                ha=align,
                va="bottom",
                color=txt_color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="#808080", linewidth=0.6),
                zorder=5,
                annotation_clip=False,
            )
            annotations.append(ann)

        # Helper: compute pixels->data conversion for Y
        def _px_to_data_y(px):
            # How many display pixels correspond to 1.0 data unit on Y?
            d0 = ax.transData.transform((0, 0))[1]
            d1 = ax.transData.transform((0, 1))[1]
            per_data = (d1 - d0)
            if per_data == 0:
                return 0.0
            return px / per_data

        # Repulsion loop
        max_iter = 200
        pad_px = 2.0    # extra pixels between boxes
        grew_ylim = False

        for _ in range(max_iter):
            # Must draw to get valid renderer + extents
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # Collect (bbox, center_y_px) for all current annotations
            bboxes = [ann.get_window_extent(renderer=renderer).expanded(1.02, 1.10) for ann in annotations]

            moved_any = False
            n = len(annotations)
            for i in range(n):
                for j in range(i + 1, n):
                    bi, bj = bboxes[i], bboxes[j]
                    if not bi.overlaps(bj):
                        continue

                    # Compute vertical overlap in pixels
                    overlap_y = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                    if overlap_y <= 0:
                        continue

                    # Amount to separate each label (symmetrically)
                    sep_each_px = overlap_y / 2.0 + pad_px

                    # Decide directions based on current data y
                    xi, yi = annotations[i].get_position()
                    xj, yj = annotations[j].get_position()
                    if yi <= yj:
                        dy_i = -_px_to_data_y(sep_each_px)
                        dy_j =  _px_to_data_y(sep_each_px)
                    else:
                        dy_i =  _px_to_data_y(sep_each_px)
                        dy_j = -_px_to_data_y(sep_each_px)

                    annotations[i].set_position((xi, yi + dy_i))
                    annotations[j].set_position((xj, yj + dy_j))
                    moved_any = True

            # If anything moved beyond top/bottom, expand y-limits a bit
            if moved_any:
                # Find the highest/lowest label y in data coords
                ys = [ann.get_position()[1] for ann in annotations]
                ylo, yhi = ax.get_ylim()
                margin = 0.02 * (yhi - ylo) + 0.1
                new_lo = min(ylo, min(ys) - margin)
                new_hi = max(yhi, max(ys) + margin)
                if new_hi > yhi or new_lo < ylo:
                    ax.set_ylim(new_lo, new_hi)
                    grew_ylim = True

            if not moved_any:
                break

        # Final safety pass: ensure no overlaps remain; if any, spread uniformly
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        final_boxes = [ann.get_window_extent(renderer=renderer).expanded(1.01, 1.05) for ann in annotations]
        def _any_overlap(boxes):
            for i in range(len(boxes)):
                for j in range(i+1, len(boxes)):
                    if boxes[i].overlaps(boxes[j]):
                        return True
            return False

        if _any_overlap(final_boxes):
            # Uniformly stack by y-order with minimal spacing (guaranteed separation)
            order = np.argsort([ann.get_position()[1] for ann in annotations])
            min_gap_px = 3.0
            # Convert pixel gap to data
            gap_dy = _px_to_data_y(min_gap_px)
            base_y = annotations[order[0]].get_position()[1]
            for k, idx in enumerate(order):
                xk, _ = annotations[idx].get_position()
                annotations[idx].set_position((xk, base_y + k * gap_dy))
            # Expand ylim accordingly
            ys = [ann.get_position()[1] for ann in annotations]
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(min(ylo, min(ys) - 0.1), max(yhi, max(ys) + 0.1))

    # Legend emphasising recurrence mode
    proxy_se = mpatches.Patch(color=color_map["SE"], label="Single-event")
    proxy_rec = mpatches.Patch(color=color_map["REC"], label="Recurrent")
    ax.legend(
        handles=[proxy_se, proxy_rec],
        frameon=False,
        title="Recurrence",
        loc="upper left",
        fontsize=8,
    )

    ax.grid(axis="both", linestyle=":", linewidth=0.6, color="#cfcfcf", alpha=0.7)
    ax.tick_params(axis="both", labelsize=8.5)

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

def encode_sequence_array_no_gaps(seq_strs: list, cols_keep: list) -> np.ndarray:
    """
    Map A,C,G,T -> 0..3; only keep columns in cols_keep.
    Returns (n_rows x len(cols_keep)) int array.
    """
    n = len(seq_strs)
    m = len(cols_keep)
    arr = np.zeros((n, m), dtype=int)
    for i, s in enumerate(seq_strs):
        for k, j in enumerate(cols_keep):
            arr[i, k] = BASE_TO_IDX.get(s[j].upper(), 0)  # assume A/C/G/T only per user note
    return arr

def plot_fixed_diff_panel(ax, phyD, phyI, gene_name: str, inv_id: str, threshold: float):
    """
    True haplotype heatmap:
      - Rows = haplotypes (Inverted on top, Direct on bottom), names shown
      - Columns = polymorphic sites only (any difference across ALL haplotypes)
      - Bold columns for fixed differences (as defined per-orientation)
      - Slight separators between rows; clear divider between orientations
    """
    namesD = list(phyD["seq_order"])
    namesI = list(phyI["seq_order"])
    seqsD = [phyD["seqs"].get(nm, "") for nm in namesD]
    seqsI = [phyI["seqs"].get(nm, "") for nm in namesI]
    if not seqsD or not seqsI:
        ax.text(0.5, 0.5, f"No sequences for {gene_name}", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return {"n_polymorphic": 0, "n_fixed": 0, "n_inverted": len(seqsI), "n_direct": len(seqsD)}

    # Determine columns to keep: any polymorphism across ALL haplotypes (Direct + Inverted)
    all_seqs = seqsI + seqsD
    m_full = len(all_seqs[0])
    for s in all_seqs:
        if len(s) != m_full:
            raise ValueError("Seq lengths differ between orientations.")
    cols_keep = []
    for j in range(m_full):
        col = [s[j] for s in all_seqs]
        if len(set(col)) > 1:  # polymorphic across ALL samples
            cols_keep.append(j)
    if not cols_keep:
        ax.text(
            0.5,
            0.5,
            f"{gene_name} — {inv_id}\nNo polymorphic CDS sites",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.axis("off")
        return {"n_polymorphic": 0, "n_fixed": 0, "n_inverted": len(seqsI), "n_direct": len(seqsD)}

    # Order rows within each orientation by clustering on kept columns to group duplicates
    def reorder_by_cols(names, seqs):
        if len(seqs) <= 2:
            return list(range(len(seqs)))
        # Build Hamming distance on filtered columns
        n = len(seqs)
        dmat = np.zeros((n, n), dtype=float)
        for i in range(n):
            si = "".join(seqs[i][j] for j in cols_keep)
            for j in range(i + 1, n):
                sj = "".join(seqs[j][j2] for j2 in cols_keep)
                d = hamming(si, sj)
                dmat[i, j] = d
                dmat[j, i] = d
        condensed = squareform(dmat, checks=False)
        Z = linkage(condensed, method="average")
        return list(leaves_list(Z))

    ordI = reorder_by_cols(namesI, seqsI)
    ordD = reorder_by_cols(namesD, seqsD)
    namesI_ord = [namesI[i] for i in ordI]
    seqsI_ord = [seqsI[i] for i in ordI]
    namesD_ord = [namesD[i] for i in ordD]
    seqsD_ord = [seqsD[i] for i in ordD]

    # Encode to int array (A/C/G/T) on filtered columns
    arr_I = encode_sequence_array_no_gaps(seqsI_ord, cols_keep)
    arr_D = encode_sequence_array_no_gaps(seqsD_ord, cols_keep)
    arr = np.vstack([arr_I, arr_D])

    # Show image
    ax.imshow(
        arr,
        cmap=BASE_CMAP,
        vmin=0,
        vmax=3,
        aspect="auto",
        interpolation="nearest",
        zorder=1,
    )

    # Fixed columns (recompute detection on full columns, then map to kept subset)
    fixed_full = set(detect_fixed_columns(seqsD, seqsI, threshold=threshold))
    fixed_kept = [k for k, j in enumerate(cols_keep) if j in fixed_full]
    for k in fixed_kept:
        ax.axvline(k - 0.5, color="#111111", linewidth=1.2, zorder=3)
        ax.axvline(k + 0.5, color="#111111", linewidth=1.2, zorder=3)
    if fixed_kept:
        ax.scatter(
            fixed_kept,
            np.full(len(fixed_kept), -0.35),
            marker="v",
            s=36,
            color="#111111",
            edgecolor="none",
            clip_on=False,
            zorder=4,
        )

    # Row separators (slight)
    nI = len(namesI_ord)
    total_rows = arr.shape[0]
    for y in np.arange(-0.5, total_rows, 1.0):
        ax.axhline(y + 0.5, color="#ffffff", linewidth=0.8, alpha=0.8, zorder=2)
    ax.axhline(nI - 0.5, color="#333333", linewidth=1.2, zorder=3)

    # Axis labels & ticks
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    ax.set_xticks(np.linspace(0, arr.shape[1] - 1, num=min(12, arr.shape[1])))
    xtick_orig = [cols_keep[int(t)] + 1 for t in ax.get_xticks()]
    ax.set_xticklabels([str(v) for v in xtick_orig], rotation=0, fontsize=7)
    ax.set_xlabel("Polymorphic CDS positions")

    y_labels = namesI_ord + namesD_ord
    ax.set_yticks(np.arange(total_rows))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.tick_params(axis="both", which="both", length=0)

    ax.text(
        -0.02,
        (nI - 0.5) / total_rows,
        "Inverted",
        ha="right",
        va="center",
        rotation=90,
        fontsize=8,
        transform=ax.transAxes,
        color=COLOR_INVERTED,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.text(
        -0.02,
        (nI + (total_rows - nI) / 2) / total_rows,
        "Direct",
        ha="right",
        va="center",
        rotation=90,
        fontsize=8,
        transform=ax.transAxes,
        color=COLOR_DIRECT,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    return {
        "n_polymorphic": int(arr.shape[1]),
        "n_fixed": int(len(fixed_kept)),
        "n_inverted": int(nI),
        "n_direct": int(total_rows - nI),
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
    panel_height = max(2.5, nrows * 0.28)

    fig, ax = plt.subplots(figsize=(14.0, panel_height + 2.0))
    stats_info = plot_fixed_diff_panel(
        ax,
        phyD,
        phyI,
        MAPT_GENE,
        inv_id,
        threshold=FIXED_DIFF_UNANIMITY_THRESHOLD,
    )

    fig.suptitle(
        "MAPT CDS polymorphisms align with inversion orientation",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    subtitle = (
        f"{MAPT_GENE} — {inv_id} | {stats_info['n_polymorphic']} polymorphic CDS positions"
        f"; {stats_info['n_fixed']} orientation-fixed"
    )
    ax.set_title(subtitle, loc="left", fontsize=10, color="#333333")

    base_handles = [mpatches.Patch(color=BASE_COLORS[i], label=b) for b, i in BASE_TO_IDX.items()]
    fig.legend(
        handles=base_handles,
        loc="upper right",
        bbox_to_anchor=(0.96, 0.96),
        frameon=False,
        fontsize=9,
        title="Base",
    )

    fig.text(
        0.01,
        0.02,
        "Rows ordered by haplotype similarity. Triangles denote CDS sites fixed between orientations.",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
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
