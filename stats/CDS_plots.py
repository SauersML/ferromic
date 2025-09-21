import os
import re
import math
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator

import networkx as nx
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon, ttest_rel

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

# Category palette (SE-D, SE-I, REC-D, REC-I) — use four highly distinct colors
CATEGORY_ORDER = ["SE-D", "SE-I", "REC-D", "REC-I"]
CATEGORY_COLORS = {
    "SE-D": "#6a3d9a",  # purple
    "SE-I": "#ff7f00",  # orange
    "REC-D": "#1f78b4", # blue
    "REC-I": "#33a02c", # green
}

# Base colors for raw haplotype plots (Fig 6): A,C,G,T (no gap color used)
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
BASE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]  # A,C,G,T
BASE_CMAP = ListedColormap(BASE_COLORS)  # vmin=0, vmax=3

# Input filenames (all in current directory)
CDS_SUMMARY_FILE = "cds_identical_proportions.tsv"
GENE_TESTS_FILE  = "gene_inversion_direct_inverted.tsv"
EMM_ADJ_FILE     = "cds_emm_adjusted.tsv"     # optional (not plotted directly)
PAIRWISE_PREFIX  = "pairs_CDS__"              # + {filename}.tsv

# Output figure filenames
FIG2A_FILE = "fig2A_split_violin_box_jitter.pdf"
FIG2B_FILE = "fig2B_ecdf_by_group.pdf"
FIG3_FILE  = "fig3_slopegraph_per_inversion.pdf"
FIG4_FILE  = "fig4_volcano_cds_contrasts.pdf"
FIG5_FILE  = "fig5_smallmultiples_identity_matrix_and_mst.pdf"
FIG6_FILE  = "fig6_fixed_differences_MAPT_SPPL2C.pdf"

# Fixed-differences genes to show raw haplotypes (Fig 6)
FIXED_DIFF_GENES = ["MAPT", "SPPL2C"]
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

def inv_id_str(chr_val, inv_start, inv_end) -> str:
    chr_clean = str(chr_val)
    if chr_clean.startswith("chr"):
        chr_clean = chr_clean[3:]
    return f"{chr_clean}:{int(inv_start)}-{int(inv_end)}"

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
        pairs_path = os.path.join(".", f"{PAIRWISE_PREFIX}{fn}.tsv")
        phy_path   = os.path.join(".", fn)
        rows.append({"filename": fn, "pairs_path": pairs_path, "phy_path": phy_path})
    return pd.DataFrame(rows).set_index("filename")

# =============================================================================
# Figure 2A: Split-violin + box + jitter (with clear n’s)
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

def plot_fig2A(cds_summary: pd.DataFrame, outfile: str):
    dist = compute_group_distributions(cds_summary)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlabel("Category")
    ax.set_ylabel("Proportion of identical pairs (per CDS)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, len(CATEGORY_ORDER) + 0.5)
    ax.set_xticks(range(1, len(CATEGORY_ORDER)+1))
    ax.set_xticklabels(CATEGORY_ORDER, rotation=0)

    for i, cat in enumerate(CATEGORY_ORDER, start=1):
        d = dist[cat]
        core = d["values_core"]
        all_vals = d["values_all"]

        # symmetric half-violins for readability
        draw_half_violin(ax, core, i, width=0.36, side="left",  facecolor=CATEGORY_COLORS[cat], alpha=0.32)
        draw_half_violin(ax, core, i, width=0.36, side="right", facecolor=CATEGORY_COLORS[cat], alpha=0.32)

        # Box (median & IQR)
        median, q1, q3 = d["box_stats"]
        if not np.isnan(median):
            ax.plot([i-0.18, i+0.18], [median, median], color="black", lw=1.0)
            ax.plot([i, i], [q1, q3], color="black", lw=1.0)
            ax.plot([i-0.12, i+0.12], [q1, q1], color="black", lw=1.0)
            ax.plot([i-0.12, i+0.12], [q3, q3], color="black", lw=1.0)

        # Jitter points (all values, including 1.0)
        if all_vals.size > 0:
            x_jit = i + (np.random.rand(all_vals.size) - 0.5) * 0.28
            ax.scatter(x_jit, all_vals, s=8, alpha=0.65, color=CATEGORY_COLORS[cat], edgecolor="none")

        # Cap at 1.0: stacked dots + explicit counts
        n_at1 = d["n_at1"]
        if n_at1 > 0:
            for k in range(n_at1):
                ax.scatter(i, 1.0 - 0.006 * k, s=10, color=CATEGORY_COLORS[cat], edgecolor="black", linewidths=0.2, zorder=3)
            # Explicit label clarifying counts
            ax.text(i, 1.0 + 0.035,
                    f"n_total={d['n_cds']}\n n_at1={n_at1} ({d['share_at_1']*100:.0f}%)",
                    ha="center", va="bottom", fontsize=7.5)

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Small legend
    patches = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in CATEGORY_ORDER]
    leg = ax.legend(handles=patches, title="Category", frameon=False, ncol=2,
                    loc="lower left", bbox_to_anchor=(0.0, -0.22), fontsize=7)
    if leg and leg.get_title():
        leg.get_title().set_fontsize(8)

    fig.tight_layout()
    ensure_dir(outfile)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Figure 2B: ECDF curves (more distinct colors, TL legend, de-overlap 1.0)
# =============================================================================

def plot_fig2B(cds_summary: pd.DataFrame, outfile: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlabel("Proportion of identical pairs (per CDS)")
    ax.set_ylabel("Empirical cumulative distribution (ECDF)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    # Collect % at 1.0 for a separate annotated list to avoid on-curve overlap
    at1_info = []

    for cat in CATEGORY_ORDER:
        sub = cds_summary[cds_summary["category"] == cat]["prop_identical_pairs"].dropna().astype(float).values
        if sub.size == 0:
            continue
        xs = np.sort(sub)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.step(xs, ys, where="post", color=CATEGORY_COLORS[cat], label=f"{cat} (n={sub.size})", linewidth=2.0, alpha=0.95)

        share1 = np.mean(xs == 1.0)
        at1_info.append((cat, share1))

    # Vertical guides
    for x in [0.8, 0.9, 1.0]:
        ax.axvline(x, linestyle="--", color="#999999", linewidth=0.7, alpha=0.6)

    # Legend top-left
    leg = ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=8)
    if leg and leg.get_title():
        leg.get_title().set_fontsize(8)

    # A tidy annotation list for % at 1.0 in the top-right corner
    # Sort by descending share to make the list meaningful
    at1_info.sort(key=lambda x: x[1], reverse=True)
    x_annot = 0.995
    y_start = 0.94
    y_step  = 0.08
    for i, (cat, share) in enumerate(at1_info):
        y_pos = y_start - i * y_step
        ax.text(x_annot, y_pos,
                f"{cat}: {share*100:.0f}%",
                ha="right", va="center",
                fontsize=8, color=CATEGORY_COLORS[cat],
                transform=ax.transAxes)

    ax.grid(axis="both", linestyle=":", alpha=0.3)

    fig.tight_layout()
    ensure_dir(outfile)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Figure 3: Slopegraph per inversion + paired p-values
# =============================================================================

def aggregate_by_inversion(cds_summary: pd.DataFrame) -> pd.DataFrame:
    agg = cds_summary.groupby(["inv_id", "phy_group"]).agg(
        k_ident_total=("n_identical_pairs", "sum"),
        pairs_total=("n_pairs", "sum"),
        n_cds=("filename", "nunique"),
    ).reset_index()
    agg["p_hat"] = agg["k_ident_total"] / agg["pairs_total"]

    piv  = agg.pivot(index="inv_id", columns="phy_group", values="p_hat").rename(columns={0: "p_D", 1: "p_I"})
    sums = agg.pivot(index="inv_id", columns="phy_group", values="pairs_total").rename(columns={0: "pairs_D", 1: "pairs_I"})
    cds_counts = agg.pivot(index="inv_id", columns="phy_group", values="n_cds").rename(columns={0: "cds_D", 1: "cds_I"})

    out = pd.concat([piv, sums, cds_counts], axis=1).reset_index()

    rec_map = cds_summary.groupby("inv_id")["recurrence"].agg(lambda x: x.mode().iloc[0] if len(x) else np.nan).to_dict()
    out["recurrence"] = out["inv_id"].map(rec_map)
    out = out.dropna(subset=["p_D", "p_I"])
    out["delta"] = out["p_I"] - out["p_D"]
    out["pairs_total"] = out[["pairs_D", "pairs_I"]].sum(axis=1)
    out["cds_total"]   = out[["cds_D", "cds_I"]].sum(axis=1)
    return out

def paired_pvalue(deltas: np.ndarray) -> float:
    """Wilcoxon signed-rank p (two-sided); fallback to paired t-test if needed."""
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size < 2:
        return np.nan
    try:
        # If all zeros, wilcoxon fails
        if np.allclose(deltas, 0.0):
            return 1.0
        stat, p = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(p)
    except Exception:
        try:
            _, p = ttest_rel(deltas, np.zeros_like(deltas))
            return float(p)
        except Exception:
            return np.nan

def plot_fig3_slopegraph(locus_df: pd.DataFrame, outfile: str):
    facets = ["SE", "REC"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.75), sharey=True)
    for ax, rec in zip(axes, facets):
        sub = locus_df[locus_df["recurrence"] == rec].copy()
        ax.set_title(f"{rec} inversions (n={len(sub)})")
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Direct", "Inverted"])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

        # Draw lines with width ~ total pairs
        if not sub.empty:
            max_pairs = sub["pairs_total"].max() if sub["pairs_total"].max() > 0 else 1.0
            for _, r in sub.iterrows():
                lw = 0.7 + 2.8 * (r["pairs_total"] / max_pairs)
                color = COLOR_INVERTED if r["delta"] >= 0 else COLOR_DIRECT
                ax.plot([1, 2], [r["p_D"], r["p_I"]], color=color, alpha=0.75, linewidth=lw)

            # Paired p-value (Wilcoxon; fallback to t-test)
            p = paired_pvalue(sub["delta"].values)
            if not np.isnan(p):
                ax.text(0.52, 0.05, f"Paired test p = {p:.2e}", transform=ax.transAxes,
                        fontsize=9, ha="left", va="bottom")

            # median delta annotation
            med_delta = sub["delta"].median()
            ax.text(0.52, 0.13, f"median Δ = {med_delta:+.3f}", transform=ax.transAxes,
                    fontsize=9, ha="left", va="bottom")

    axes[0].set_ylabel("Proportion of identical pairs (locus-level)")
    fig.tight_layout()
    ensure_dir(outfile)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Figure 4: Volcano plot (de-overlapped labels, cap q at 1e-22, distinct colors)
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




def _nonoverlapping_text(ax, xs, ys, labels, colors, xpad=0.02, ypad=0.04):
    """
    Very simple de-overlap: sort by y, then nudge labels up/down if within ypad.
    """
    items = sorted(zip(xs, ys, labels, colors), key=lambda t: t[1])
    placed = []
    for x, y, lab, col in items:
        y_new = y
        for _ in range(50):  # limited attempts
            if all(abs(y_new - yy) > ypad for _, yy, _ in placed):
                break
            y_new += ypad * 0.6
        ax.text(x + xpad, y_new, lab, fontsize=8, color=col, ha="left", va="center")
        placed.append((x, y_new, lab))

def plot_fig4_volcano(df: pd.DataFrame, outfile: str):
    """
    Volcano plot:
      - Colors use the pre-merged df['recurrence'] from prepare_volcano (SE/REC).
      - No special outline for CLIP2 (or anything else).
      - Legend moved to TOP LEFT.
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.set_xlabel("Δ proportion (Inverted − Direct)")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{BH}\;q)$")

    # Coordinates (cap q to avoid inf on -log10)
    x = pd.to_numeric(df["delta"], errors="coerce").values
    q = pd.to_numeric(df["q_value"], errors="coerce").values
    q = np.clip(q, 1e-22, 1.0)
    with np.errstate(divide="ignore"):
        y = -np.log10(q)

    # Point sizes by total pairs
    sizes = pd.to_numeric(df.get("n_pairs_total", pd.Series(0)), errors="coerce").fillna(0).values
    if np.nanmax(sizes) > 0:
        sizes = 24 + 220 * (sizes / np.nanmax(sizes))
    else:
        sizes = np.full_like(y, 32.0)

    # Colors: trust the recurrence column provided by prepare_volcano
    # Expect exact "SE" / "REC" strings from load_cds_summary -> prepare_volcano merge
    rec = df["recurrence"].astype(str)
    color_map = {"SE": CATEGORY_COLORS["SE-I"], "REC": CATEGORY_COLORS["REC-I"]}
    colors = rec.map(color_map).fillna("#7f7f7f").values  # gray only if truly missing

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.85, edgecolor="white", linewidths=0.6)

    # Significance threshold at q=0.05
    thresh_y = -math.log10(0.05)
    ax.axhline(thresh_y, linestyle="--", color="#999999", linewidth=1.0)
    ax.text(ax.get_xlim()[0], thresh_y + 0.05, "q = 0.05", va="bottom", ha="left", fontsize=8, color="#666666")

    # Label significant points (q<=0.05) with a simple de-overlap
    sig = df[pd.to_numeric(df["q_value"], errors="coerce") <= 0.05].copy()
    if not sig.empty:
        xs_lab = pd.to_numeric(sig["delta"], errors="coerce").values
        ys_lab = -np.log10(np.clip(pd.to_numeric(sig["q_value"], errors="coerce").values, 1e-22, 1.0))
        labs   = sig["gene_name"].astype(str).values
        cols   = sig["recurrence"].map(color_map).fillna("#7f7f7f").values

        # minimal non-overlap
        items = sorted(zip(xs_lab, ys_lab, labs, cols), key=lambda t: t[1])
        placed = []
        for x0, y0, lab, col in items:
            y_new = y0
            for _ in range(50):
                if all(abs(y_new - yy) > 0.05 for _, yy, _ in placed):
                    break
                y_new += 0.03
            ax.text(x0 + 0.01, y_new, lab, fontsize=8, color=col, ha="left", va="center")
            placed.append((x0, y_new, lab))

    # Legend moved to TOP LEFT; no special highlighting for CLIP2
    proxy_se  = mpatches.Patch(color=color_map["SE"],  label="Single-event")
    proxy_rec = mpatches.Patch(color=color_map["REC"], label="Recurrent")
    ax.legend(handles=[proxy_se, proxy_rec], frameon=False, title="Recurrence", loc="upper left", fontsize=8)

    ax.grid(axis="both", linestyle=":", alpha=0.3)
    fig.tight_layout()
    ensure_dir(outfile)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Figure 5: Small-multiples (row per gene: Identity matrix (left) + MST (right))
# =============================================================================

def read_pairs_tsv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pairs TSV not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype={"sample1": str, "sample2": str})
    for col in ["n_sites", "n_diff_sites", "prop_sites_different"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["identical"] = (df["n_diff_sites"] == 0).astype(int)
    return df

def order_names_by_sequence(seqs: dict) -> list:
    names = list(seqs.keys())
    n = len(names)
    if n <= 2:
        return names
    dists = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = hamming(seqs[names[i]], seqs[names[j]])
            dists[i, j] = d
            dists[j, i] = d
    condensed = squareform(dists, checks=False)
    Z = linkage(condensed, method="average")
    order_idx = leaves_list(Z)
    return [names[i] for i in order_idx]

def build_identity_block_from_pairs(df_pairs: pd.DataFrame, names: list) -> np.ndarray:
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    mat = np.eye(n, dtype=int)
    for _, r in df_pairs.iterrows():
        s1 = r["sample1"]; s2 = r["sample2"]
        if s1 in idx and s2 in idx:
            i = idx[s1]; j = idx[s2]
            val = 1 if r["n_diff_sites"] == 0 else 0
            mat[i, j] = val
            mat[j, i] = val
    return mat

def collapse_unique_sequences(seq_dict: dict, orientations: dict):
    seq_to_members = defaultdict(list)
    for name, seq in seq_dict.items():
        seq_to_members[seq].append(name)
    unique_list, members, multiplicity, orient_counts = [], [], [], []
    for seq, mems in seq_to_members.items():
        unique_list.append(seq)
        members.append(mems)
        multiplicity.append(len(mems))
        d_count = sum(1 for nm in mems if orientations.get(nm) == "D")
        i_count = sum(1 for nm in mems if orientations.get(nm) == "I")
        orient_counts.append((d_count, i_count))
    return unique_list, members, multiplicity, orient_counts

def build_mst(unique_seqs: list):
    n = len(unique_seqs)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            d = hamming(unique_seqs[i], unique_seqs[j])
            G.add_edge(i, j, weight=d)
    if n <= 1:
        return G
    T = nx.minimum_spanning_tree(G, weight="weight")
    return T

def draw_node_pie(ax, center, sizes, colors, radius):
    total = sum(sizes)
    if total <= 0:
        circ = plt.Circle(center, radius, facecolor="#dddddd", edgecolor="black", linewidth=0.5)
        ax.add_patch(circ)
        return
    start = 0.0
    for sz, col in zip(sizes, colors):
        if sz <= 0:
            continue
        theta1 = start * 360 / total
        theta2 = (start + sz) * 360 / total
        wedge = mpatches.Wedge(center, radius, theta1, theta2, facecolor=col, edgecolor="black", linewidth=0.5)
        ax.add_patch(wedge)
        start += sz

def plot_identity_matrix(ax, block_mat: np.ndarray, names_top: list, names_bottom: list,
                         title=None, subtitle=None):
    """
    Minimal, clean identity matrix: top-left = Direct×Direct; bottom-right = Inverted×Inverted;
    off-diagonals included (Direct×Inverted). No dense per-row bars to avoid clutter.
    """
    n_top = len(names_top)
    n_bot = len(names_bottom)
    im = ax.imshow(block_mat, interpolation="nearest",
                   cmap=ListedColormap(["#212121", "#f2f2f2"]), vmin=0, vmax=1, aspect="auto")
    # Quadrant boundaries
    if n_top > 0 and n_bot > 0:
        ax.axhline(n_top - 0.5, color="#555555", linewidth=1.0)
        ax.axvline(n_top - 0.5, color="#555555", linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, loc="left", pad=4)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color="#333333")

def build_identity_and_mst_for_gene(gene_row: pd.Series, cds_summary: pd.DataFrame, pairs_index: pd.DataFrame):
    """
    Robust version with deep DEBUG:
      - Canonicalize gene_tests.inv_id to ASCII (strip 'chr', unify dashes, strip unicode spaces)
      - Parse chr/start/end and match cds_summary by numeric locus + gene + transcript
      - Fall back to also reporting the result of raw-string equality (for evidence)
      - Print code points to reveal hidden Unicode
    """
    # -----------------------------
    # Helpers (local to this function)
    # -----------------------------
    def _codepoints(s: str) -> str:
        return " ".join(f"U+{ord(c):04X}" for c in s)

    def _norm_inv_id(raw: str):
        # strip leading/trailing whitespace incl. Unicode
        s = re.sub(r"\s+", "", str(raw))
        # remove any leading "chr" (case-insensitive)
        s = re.sub(r"^chr", "", s, flags=re.IGNORECASE)
        # unify dash-like characters to ASCII hyphen
        s = s.translate({
            0x2010: "-",  # hyphen
            0x2011: "-",  # non-breaking hyphen
            0x2012: "-",  # figure dash
            0x2013: "-",  # en dash
            0x2014: "-",  # em dash
            0x2015: "-",
            0x2212: "-",  # minus
            0xFE58: "-",
            0xFE63: "-",
            0xFF0D: "-",
        })
        # Must be <chr>:<start>-<end>
        m = re.match(r"^([^:]+):(\d+)-(\d+)$", s)
        if not m:
            return None, None, None, s
        chr_norm = m.group(1)
        try:
            start_norm = int(m.group(2))
            end_norm   = int(m.group(3))
        except Exception:
            return None, None, None, s
        # Return canonical ASCII form (no 'chr' prefix)
        canon = f"{chr_norm}:{start_norm}-{end_norm}"
        return chr_norm, start_norm, end_norm, canon

    # -----------------------------
    # Inputs & raw debug
    # -----------------------------
    gene_name     = str(gene_row["gene_name"])
    transcript_id = str(gene_row["transcript_id"])
    inv_raw       = str(gene_row["inv_id"])

    print(f"[Fig5 DEBUG] ----")
    print(f"[Fig5 DEBUG] Request: gene={gene_name} | transcript={transcript_id}")
    print(f"[Fig5 DEBUG] inv_id (raw) = {repr(inv_raw)} | len={len(inv_raw)} | codepoints=<{_codepoints(inv_raw)}>")

    chr_norm, start_norm, end_norm, inv_canon = _norm_inv_id(inv_raw)
    print(f"[Fig5 DEBUG] inv_id (canon) = {repr(inv_canon)} | codepoints=<{_codepoints(inv_canon)}>")
    print(f"[Fig5 DEBUG] Parsed locus: chr={chr_norm} start={start_norm} end={end_norm}")

    # Show what cds_summary contains for this gene+transcript (to compare)
    cs_slice_gt = cds_summary[
        (cds_summary["gene_name"].astype(str) == gene_name) &
        (cds_summary["transcript_id"].astype(str) == transcript_id)
    ]
    uniq_inv_ids = sorted(set(cs_slice_gt["inv_id"].astype(str))) if not cs_slice_gt.empty else []
    if uniq_inv_ids:
        print(f"[Fig5 DEBUG] cds_summary has {len(cs_slice_gt)} rows for this gene+transcript; "
              f"{len(uniq_inv_ids)} unique inv_id strings:")
        for x in uniq_inv_ids[:6]:
            print(f"[Fig5 DEBUG]   inv_id in cs = {repr(x)} | len={len(x)} | codepoints=<{_codepoints(x)}>")
    else:
        print(f"[Fig5 DEBUG] cds_summary has 0 rows for this gene+transcript.")

    # -----------------------------
    # Evidence: does raw-string equality find anything?
    # -----------------------------
    sub_raw = cds_summary[
        (cds_summary["gene_name"].astype(str) == gene_name) &
        (cds_summary["transcript_id"].astype(str) == transcript_id) &
        (cds_summary["inv_id"].astype(str) == inv_raw)
    ]
    print(f"[Fig5 DEBUG] Match by RAW string equality: {len(sub_raw)} rows.")

    # -----------------------------
    # FIX: Match by numeric locus triplet (canonical), not by inv_id string.
    # This avoids any hidden-Unicode issues entirely.
    # -----------------------------
    # Build normalized locus columns locally (no mutation of original df)
    cs_loc = cds_summary.copy()
    cs_loc["chr_norm"]   = cs_loc["chr"].astype(str).str.replace(r"^chr", "", regex=True)
    cs_loc["start_norm"] = pd.to_numeric(cs_loc["inv_start"], errors="coerce").astype("Int64")
    cs_loc["end_norm"]   = pd.to_numeric(cs_loc["inv_end"],   errors="coerce").astype("Int64")

    if chr_norm is None or pd.isna(start_norm) or pd.isna(end_norm):
        print(f"[Fig5 DEBUG] ERROR: Could not parse locus from inv_id. Aborting.")
        raise ValueError(f"Malformed inv_id in gene_tests: {repr(inv_raw)}")

    sub = cs_loc[
        (cs_loc["gene_name"].astype(str) == gene_name) &
        (cs_loc["transcript_id"].astype(str) == transcript_id) &
        (cs_loc["chr_norm"].astype(str) == str(chr_norm)) &
        (cs_loc["start_norm"] == start_norm) &
        (cs_loc["end_norm"]   == end_norm)
    ]

    print(f"[Fig5 DEBUG] Match by NUMERIC locus+gene+transcript: {len(sub)} rows.")
    if sub.empty:
        # If still empty, show what *is* present at this locus for the same gene (transcripts present)
        locus_any = cs_loc[
            (cs_loc["gene_name"].astype(str) == gene_name) &
            (cs_loc["chr_norm"].astype(str) == str(chr_norm)) &
            (cs_loc["start_norm"] == start_norm) &
            (cs_loc["end_norm"]   == end_norm)
        ]
        if not locus_any.empty:
            tset = sorted(set(locus_any["transcript_id"].astype(str)))
            print(f"[Fig5 DEBUG] Locus exists for gene={gene_name}, transcripts present at locus: {tset}")
            print(f"[Fig5 DEBUG] Orientations present per transcript at locus:")
            for tr, grp in locus_any.groupby("transcript_id"):
                ors = sorted(set(grp["phy_group"]))
                print(f"[Fig5 DEBUG]   {tr}: phy_groups={ors}, n={len(grp)}")
            raise ValueError(f"{gene_name} — {inv_canon} | No rows for this exact transcript after locus match.")
        else:
            print(f"[Fig5 DEBUG] No rows at this locus for gene={gene_name} at all.")
            raise ValueError(f"{gene_name} — {inv_canon} | Locus not found in cds_summary.")

    # Require both orientations
    have_D = not sub[sub["phy_group"] == 0].empty
    have_I = not sub[sub["phy_group"] == 1].empty
    print(f"[Fig5 DEBUG] Orientation coverage for this transcript at locus: D={have_D} I={have_I}")
    if not (have_D and have_I):
        # Show what’s missing, but also show if the *other* transcript(s) have both
        print(f"[Fig5 DEBUG] Rows we matched (head):")
        print(sub[["phy_group","filename","n_sequences","n_pairs","prop_identical_pairs"]].head(3).to_string(index=False))
        locus_any = cs_loc[
            (cs_loc["gene_name"].astype(str) == gene_name) &
            (cs_loc["chr_norm"].astype(str) == str(chr_norm)) &
            (cs_loc["start_norm"] == start_norm) &
            (cs_loc["end_norm"]   == end_norm)
        ]
        both_transcripts = []
        for tr, grp in locus_any.groupby("transcript_id"):
            if not grp[grp["phy_group"] == 0].empty and not grp[grp["phy_group"] == 1].empty:
                both_transcripts.append(tr)
        print(f"[Fig5 DEBUG] Transcripts at locus with BOTH orientations: {both_transcripts}")
        raise ValueError(f"{gene_name} — {inv_canon} | This transcript lacks one orientation.")

    # Choose best row per orientation (most sequences, then pairs)
    rowD = sub[sub["phy_group"] == 0].sort_values(["n_sequences", "n_pairs"], ascending=False).iloc[0]
    rowI = sub[sub["phy_group"] == 1].sort_values(["n_sequences", "n_pairs"], ascending=False).iloc[0]
    print(f"[Fig5 DEBUG] Chosen files: D={rowD['filename']} | I={rowI['filename']}")

    # Load pairs + PHYLIP
    fnD = str(rowD["filename"]); fnI = str(rowI["filename"])
    pairsD = read_pairs_tsv(pairs_index.loc[fnD, "pairs_path"])
    pairsI = read_pairs_tsv(pairs_index.loc[fnI, "pairs_path"])
    phyD   = read_phy(pairs_index.loc[fnD, "phy_path"])
    phyI   = read_phy(pairs_index.loc[fnI, "phy_path"])
    print(f"[Fig5 DEBUG] PHYLIP sizes: D n={phyD['n']} m={phyD['m']} | I n={phyI['n']} m={phyI['m']}")

    # Order names within each orientation by sequence clustering
    orderD = order_names_by_sequence(phyD["seqs"])
    orderI = order_names_by_sequence(phyI["seqs"])
    print(f"[Fig5 DEBUG] Ordered names: nD={len(orderD)} nI={len(orderI)}")

    # Build within-orientation identity blocks from the pairs tables
    matD = build_identity_block_from_pairs(pairsD, orderD)
    matI = build_identity_block_from_pairs(pairsI, orderI)

    # Assemble full matrix, compute cross-orientation from sequences
    N_top = len(orderD); N_bot = len(orderI)
    full  = np.zeros((N_top + N_bot, N_top + N_bot), dtype=int)
    full[:N_top, :N_top]     = matD
    full[N_top:, N_top:]     = matI
    for i, nmD in enumerate(orderD):
        seqD = phyD["seqs"][nmD]
        for j, nmI in enumerate(orderI):
            seqI = phyI["seqs"][nmI]
            same = 1 if hamming(seqD, seqI) == 0 else 0
            full[i, N_top + j] = same
            full[N_top + j, i] = same

    # Recurrence label (consistent within locus for this transcript)
    rec_label = rowD.get("recurrence", rowI.get("recurrence", ""))

    # Build MST over unique sequences across both orientations
    all_seqs     = {}
    orientations = {}
    for nm in orderD:
        all_seqs[f"D::{nm}"] = phyD["seqs"][nm]; orientations[f"D::{nm}"] = "D"
    for nm in orderI:
        all_seqs[f"I::{nm}"] = phyI["seqs"][nm]; orientations[f"I::{nm}"] = "I"
    uniq_list, members, multiplicity, orient_counts = collapse_unique_sequences(all_seqs, orientations)
    T = build_mst(uniq_list)

    # Final sanity prints
    print(f"[Fig5 DEBUG] Unique haplotypes: {len(uniq_list)} | multiplicities range: "
          f"{(min(multiplicity) if multiplicity else 0)}–{(max(multiplicity) if multiplicity else 0)}")

    return {
        "full_matrix": full,
        "names_top": orderD,
        "names_bottom": orderI,
        "mst_graph": T,
        "mst_multiplicity": multiplicity,
        "mst_orient_counts": orient_counts,
        "nD": len(orderD),
        "nI": len(orderI),
        "pD": float(rowD["prop_identical_pairs"]),
        "pI": float(rowI["prop_identical_pairs"]),
        "delta": float(gene_row.get("delta", np.nan)),
        "q": float(gene_row.get("q_value", np.nan)),
        "recurrence": str(rec_label),
        "inv_id": inv_canon,  # report canonical
        "gene_name": gene_name,
    }



def plot_fig5_smallmultiples(gene_test_df: pd.DataFrame, cds_summary: pd.DataFrame, pairs_index: pd.DataFrame, outfile: str):

    # ---------- helper (local) ----------
    def _pick_transcript_for_gene_inv(gene_name: str, inv_id: str, requested_tid: str | None):
        """
        Strict: use numeric locus (chr,start,end) from inv_id + exact gene_name + exact requested transcript.
        No fallbacks. Emits detailed DEBUG about each filtering step.
        Returns the requested_tid if and only if BOTH orientations exist for that transcript at that locus.
        Otherwise returns None.
        """
        # ---- parse locus from inv_id (ASCII hyphen, no 'chr' prefix expected) ----
        m = re.match(r"^\s*([^:]+):(\d+)-(\d+)\s*$", str(inv_id))
        if not m:
            print(f"[Fig5 DEBUG] BAD inv_id format in volcano row: {repr(inv_id)}")
            return None
        chr_q = m.group(1)
        start_q = int(m.group(2))
        end_q   = int(m.group(3))
        inv_canon = f"{chr_q}:{start_q}-{end_q}"
        print(f"[Fig5 DEBUG] >>> START pick: gene={gene_name} | inv_id={inv_canon} | requested_tid={requested_tid}")
    
        # ---- quick evidence: what string-equality says (diagnostic only) ----
        sub_by_str = cds_summary[
            (cds_summary["gene_name"].astype(str) == str(gene_name)) &
            (cds_summary["inv_id"].astype(str)   == inv_canon)
        ]
        print(f"[Fig5 DEBUG] string-match (gene+inv_id): n={len(sub_by_str)}")
    
        # ---- robust numeric locus view ----
        cs = cds_summary.copy()
        cs["chr_norm"]   = cs["chr"].astype(str).str.replace(r"^chr", "", regex=True)
        cs["start_norm"] = pd.to_numeric(cs["inv_start"], errors="coerce").astype("Int64")
        cs["end_norm"]   = pd.to_numeric(cs["inv_end"],   errors="coerce").astype("Int64")
    
        # rows at this numeric locus (all genes)
        sub_locus_any = cs[
            (cs["chr_norm"].astype(str) == str(chr_q)) &
            (cs["start_norm"] == start_q) &
            (cs["end_norm"]   == end_q)
        ]
        print(f"[Fig5 DEBUG] numeric locus match (all genes): n={len(sub_locus_any)}")
    
        # rows at this locus for this gene
        sub_gene_locus = sub_locus_any[sub_locus_any["gene_name"].astype(str) == str(gene_name)]
        print(f"[Fig5 DEBUG] numeric locus + gene: n={len(sub_gene_locus)}")
    
        if sub_gene_locus.empty:
            # Show what genes *are* present at this locus
            genes_here = sorted(sub_locus_any["gene_name"].astype(str).unique().tolist())
            print(f"[Fig5 DEBUG] genes present at locus {inv_canon}: {genes_here}")
            print(f"[Fig5 DEBUG] >>> END pick: FAIL (no rows for this gene at locus)")
            return None
    
        # Must have a requested transcript (no fallbacks)
        if requested_tid is None:
            print(f"[Fig5 DEBUG] requested_tid is None; refusing to guess.")
            print(f"[Fig5 DEBUG] transcripts available for gene={gene_name} at locus {inv_canon}:",
                  sorted(sub_gene_locus['transcript_id'].astype(str).unique().tolist()))
            print(f"[Fig5 DEBUG] >>> END pick: FAIL (no transcript specified)")
            return None
    
        sub_tid = sub_gene_locus[sub_gene_locus["transcript_id"].astype(str) == str(requested_tid)]
        print(f"[Fig5 DEBUG] numeric locus + gene + requested_tid: n={len(sub_tid)}")
    
        if sub_tid.empty:
            # Show what transcripts are actually present for this gene at the locus
            tset = sorted(sub_gene_locus["transcript_id"].astype(str).unique().tolist())
            print(f"[Fig5 DEBUG] requested_tid NOT present at locus for this gene. available={tset}")
            print(f"[Fig5 DEBUG] >>> END pick: FAIL (requested transcript absent)")
            return None
    
        # Check orientation coverage for the requested transcript
        have_D = not sub_tid[sub_tid["phy_group"] == 0].empty
        have_I = not sub_tid[sub_tid["phy_group"] == 1].empty
        print(f"[Fig5 DEBUG] requested_tid orientation coverage: D={have_D} I={have_I}")
    
        if have_D and have_I:
            print(f"[Fig5 DEBUG] >>> END pick: OK (using {requested_tid})")
            return str(requested_tid)
    
        # If missing, show per-transcript orientation summary (proof there is or isn't a both-orientation transcript)
        orient_table = (sub_gene_locus.groupby(["transcript_id","phy_group"], dropna=False)
                        .size().unstack(fill_value=0).rename(columns={0:"rows_D",1:"rows_I"}))
        print("[Fig5 DEBUG] orientation summary at locus for this gene (rows_D, rows_I):")
        try:
            print(orient_table.sort_values(["rows_D","rows_I"], ascending=False).to_string())
        except Exception:
            print(orient_table)
    
        print(f"[Fig5 DEBUG] >>> END pick: FAIL (requested transcript lacks one orientation)")
        return None


    # Select significant rows
    sig = gene_test_df.dropna(subset=["q_value"]).sort_values("q_value")
    sig = sig[sig["q_value"] <= 0.05]
    if sig.empty:
        warnings.warn("No significant genes (q<=0.05) found; plotting top 10 by lowest q anyway.")
        sig = gene_test_df.dropna(subset=["q_value"]).sort_values("q_value").head(10)
    elif len(sig) > 10:
        sig = sig.head(10)
    sig = sig.reset_index(drop=True)

    n_rows = len(sig)
    if n_rows == 0:
        warnings.warn("No genes to plot in Fig. 5.")
        return

    # Big canvas: each row gets ample height; wide for identity matrix readability
    fig_w = 14.0
    row_h = 2.8  # per-gene row height
    fig_h = max(6.0, n_rows * row_h)

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Manual layout: for each row: two axes — left (identity, ~75% width), right (MST, ~25% width)
    left_w = 0.73
    right_w = 0.23
    vspace = 0.03
    top_margin = 0.04
    bottom_margin = 0.06

    for i, (_, row) in enumerate(sig.iterrows()):
        gene_name     = str(row.get("gene_name"))
        inv_id        = str(row.get("inv_id"))
        requested_tid = str(row.get("transcript_id")) if pd.notna(row.get("transcript_id")) else None

        # vertical slot for this row
        slot_top = 1.0 - top_margin - i * ((1.0 - top_margin - bottom_margin - (n_rows-1) * vspace) / n_rows + vspace)
        slot_h   = (1.0 - top_margin - bottom_margin - (n_rows-1) * vspace) / n_rows
        slot_bottom = slot_top - slot_h

        # Axes
        ax_identity = fig.add_axes([0.06, slot_bottom, left_w, slot_h])
        ax_mst      = fig.add_axes([0.06 + left_w + 0.02, slot_bottom, right_w, slot_h])

        # Decide which transcript_id to use for this (gene, inv)
        chosen_tid = _pick_transcript_for_gene_inv(gene_name, inv_id, requested_tid)

        if chosen_tid is None:
            msg = (f"Error: {gene_name} — {inv_id}\n"
                   f"No transcript with both orientations.\n"
                   f"(requested={requested_tid})")
            print(f"[Fig5 DEBUG] {msg.replace(chr(10), ' | ')}")
            ax_identity.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9)
            ax_identity.axis("off"); ax_mst.axis("off")
            continue

        # Build a strict row using the chosen transcript_id
        row_strict = row.copy()
        row_strict["transcript_id"] = chosen_tid

        # For more debugging, show the filenames we will use
        sub_chosen = cds_summary[
            (cds_summary["gene_name"].astype(str) == gene_name) &
            (cds_summary["inv_id"].astype(str) == inv_id) &
            (cds_summary["transcript_id"].astype(str) == chosen_tid)
        ]
        try:
            fnD = str(sub_chosen[sub_chosen["phy_group"] == 0].sort_values(["n_sequences", "n_pairs"], ascending=False)["filename"].iloc[0])
            fnI = str(sub_chosen[sub_chosen["phy_group"] == 1].sort_values(["n_sequences", "n_pairs"], ascending=False)["filename"].iloc[0])
            print(f"[Fig5 DEBUG] Using transcript_id={chosen_tid} for {gene_name} {inv_id} "
                  f"-> files: D='{fnD}', I='{fnI}'")
        except Exception as e:
            print(f"[Fig5 DEBUG] Failed to preview filenames for {gene_name} {inv_id} (tid={chosen_tid}): {e}")

        # Build identity + MST
        try:
            data = build_identity_and_mst_for_gene(row_strict, cds_summary, pairs_index)
        except Exception as e:
            print(f"[Fig5 DEBUG] build_identity_and_mst_for_gene FAILED for {gene_name} {inv_id} (tid={chosen_tid}): {e}")
            ax_identity.text(0.5, 0.5, f"Error: {gene_name}\n{str(e)}", ha="center", va="center", fontsize=9)
            ax_identity.axis("off"); ax_mst.axis("off"); continue

        title = f"{data['gene_name']} — {data['inv_id']} ({data['recurrence']})"
        subtitle = (f"nD={data['nD']}, nI={data['nI']}; "
                    f"p̂D={data['pD']:.3f}, p̂I={data['pI']:.3f}; "
                    f"Δ={data['delta']:+.3f}, q={data['q']:.2e} "
                    f"| transcript={chosen_tid}")
        plot_identity_matrix(ax_identity, data["full_matrix"], data["names_top"], data["names_bottom"],
                             title=title, subtitle=subtitle)

        # MST plot
        T = data["mst_graph"]
        multiplicity   = data["mst_multiplicity"]
        orient_counts  = data["mst_orient_counts"]
        ax_mst.set_title("MST (node size = multiplicity)", fontsize=9, loc="left", pad=2)
        if T.number_of_nodes() == 0:
            ax_mst.axis("off")
        else:
            pos = nx.spring_layout(T, seed=RANDOM_SEED, k=0.5)
            nx.draw_networkx_edges(T, pos, ax=ax_mst, width=0.9, edge_color="#9e9e9e", alpha=0.8)
            max_mult = max(multiplicity) if multiplicity else 1
            for node in T.nodes():
                mult = multiplicity[node]
                d_count, i_count = orient_counts[node]
                radius = 0.06 + 0.10 * (math.sqrt(mult) / math.sqrt(max_mult))
                cx, cy = pos[node]
                draw_node_pie(ax_mst, (cx, cy), [d_count, i_count], [COLOR_DIRECT, COLOR_INVERTED], radius)
            ax_mst.set_xticks([]); ax_mst.set_yticks([])
            xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
            pad = 0.15
            ax_mst.set_xlim(min(xs)-pad, max(xs)+pad)
            ax_mst.set_ylim(min(ys)-pad, max(ys)+pad)

    # Global legends — below figure
    proxy_ident = [mpatches.Patch(color="#f2f2f2", label="Identical"),
                   mpatches.Patch(color="#212121", label="Non-identical")]
    proxy_orient = [mpatches.Patch(color=COLOR_DIRECT, label="Direct"),
                    mpatches.Patch(color=COLOR_INVERTED, label="Inverted")]
    fig.legend(handles=proxy_ident + proxy_orient, loc="lower center",
               ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.01))

    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)





# =============================================================================
# Figure 6: Fixed differences — true haplotype heatmaps (MAPT & SPPL2C)
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

def row_separators(ax, y_positions, color="#ffffff"):
    for y in y_positions:
        ax.axhline(y, color=color, linewidth=1.0, alpha=0.8)

def plot_fixed_diff_panel(ax, phyD, phyI, gene_name: str, inv_id: str, threshold: float):
    """
    True haplotype heatmap:
      - Rows = haplotypes (Inverted on top, Direct on bottom), names shown
      - Columns = polymorphic sites only (any difference across ALL haplotypes)
      - Bold columns for fixed differences (as defined per-orientation)
      - Slight separators between rows; clear divider between orientations
    """
    namesD = list(phyD["seq_order"]); seqsD = [phyD["seqs"][nm] for nm in namesD]
    namesI = list(phyI["seq_order"]); seqsI = [phyI["seqs"][nm] for nm in namesI]
    if not seqsD or not seqsI:
        ax.text(0.5, 0.5, f"No sequences for {gene_name}", ha="center", va="center")
        ax.axis("off")
        return

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
        # Degenerate: no polymorphic sites — display a message
        ax.text(0.5, 0.5, f"{gene_name} — {inv_id}\nNo polymorphic sites in CDS", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return

    # Order rows within each orientation by clustering on kept columns to group duplicates
    def reorder_by_cols(names, seqs):
        if len(seqs) <= 2:
            return list(range(len(seqs)))
        # Build Hamming distance on filtered columns
        n = len(seqs)
        dmat = np.zeros((n, n), dtype=float)
        for i in range(n):
            si = ''.join(seqs[i][j] for j in cols_keep)
            for j in range(i+1, n):
                sj = ''.join(seqs[j][j2] for j2 in cols_keep)
                d = hamming(si, sj)
                dmat[i, j] = d; dmat[j, i] = d
        condensed = squareform(dmat, checks=False)
        Z = linkage(condensed, method="average")
        return list(leaves_list(Z))

    ordI = reorder_by_cols(namesI, seqsI)
    ordD = reorder_by_cols(namesD, seqsD)
    namesI_ord = [namesI[i] for i in ordI]; seqsI_ord = [seqsI[i] for i in ordI]
    namesD_ord = [namesD[i] for i in ordD]; seqsD_ord = [seqsD[i] for i in ordD]

    # Encode to int array (A/C/G/T) on filtered columns
    arr_I = encode_sequence_array_no_gaps(seqsI_ord, cols_keep)
    arr_D = encode_sequence_array_no_gaps(seqsD_ord, cols_keep)
    arr   = np.vstack([arr_I, arr_D])

    # Show image
    im = ax.imshow(arr, cmap=BASE_CMAP, vmin=0, vmax=3, aspect="auto", interpolation="nearest")

    # Fixed columns (recompute detection on full columns, then map to kept subset)
    fixed_full = set(detect_fixed_columns(seqsD, seqsI, threshold=threshold))
    fixed_kept = [k for k, j in enumerate(cols_keep) if j in fixed_full]
    for k in fixed_kept:
        ax.axvline(k-0.5, color="black", linewidth=1.2)
        ax.axvline(k+0.5, color="black", linewidth=1.2)

    # Row separators (slight)
    nI = len(namesI_ord)
    total_rows = arr.shape[0]
    # Fine separators between ALL rows
    for y in np.arange(-0.5, total_rows, 1.0):
        ax.axhline(y + 0.5, color="#ffffff", linewidth=0.8, alpha=0.8)
    # Bold separator between orientations
    ax.axhline(nI - 0.5, color="#333333", linewidth=1.2)

    # Axis labels & ticks
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    ax.set_xticks(np.linspace(0, arr.shape[1]-1, num=min(12, arr.shape[1])))
    # Convert back to original CDS positions (1-based) for label clarity
    xtick_orig = [cols_keep[int(t)] + 1 for t in ax.get_xticks()]
    ax.set_xticklabels([str(v) for v in xtick_orig], rotation=0, fontsize=7)
    ax.set_xlabel("Polymorphic CDS positions")

    # Y ticks (haplotype names)
    y_labels = namesI_ord + namesD_ord
    ax.set_yticks(np.arange(total_rows))
    ax.set_yticklabels(y_labels, fontsize=7)
    # Orientation side labels
    ax.text(-0.01, (nI-0.5)/total_rows, "Inverted", ha="right", va="center", rotation=90, fontsize=8, transform=ax.transAxes, color=COLOR_INVERTED)
    ax.text(-0.01, (nI + (total_rows-nI)/2)/total_rows, "Direct", ha="right", va="center", rotation=90, fontsize=8, transform=ax.transAxes, color=COLOR_DIRECT)

    ax.set_title(f"{gene_name} — {inv_id} | polymorphic sites={arr.shape[1]} | fixed columns={len(fixed_kept)}",
                 fontsize=10, loc="left")

def plot_fig6_fixed_differences(cds_summary: pd.DataFrame, pairs_index: pd.DataFrame, outfile: str):
    # Choose, for each gene, the inversion with both orientations and most sequences
    selections = []
    for gene in FIXED_DIFF_GENES:
        sub = cds_summary[(cds_summary["gene_name"] == gene)]
        if sub.empty:
            warnings.warn(f"No CDS summary entries found for fixed-diff gene {gene}")
            continue
        candidates = []
        for inv_id, grp in sub.groupby("inv_id"):
            have_D = not grp[grp["phy_group"] == 0].empty
            have_I = not grp[grp["phy_group"] == 1].empty
            if have_D and have_I:
                total_nseq = int(grp["n_sequences"].sum())
                candidates.append((inv_id, total_nseq))
        if not candidates:
            warnings.warn(f"No inversion with both orientations for gene {gene}")
            continue
        inv_id_best = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
        selections.append((gene, inv_id_best))

    if not selections:
        warnings.warn("No fixed-difference panels to plot.")
        return

    # Dynamic canvas sizing based on #haplotypes and #polymorphic sites
    # We'll parse sequences first to compute sizes
    gene_panels = []
    for gene, inv_id in selections:
        sub = cds_summary[(cds_summary["gene_name"] == gene) & (cds_summary["inv_id"] == inv_id)]
        rowD = sub[sub["phy_group"] == 0].iloc[0]
        rowI = sub[sub["phy_group"] == 1].iloc[0]
        phyD = read_phy(pairs_index.loc[str(rowD["filename"]), "phy_path"])
        phyI = read_phy(pairs_index.loc[str(rowI["filename"]), "phy_path"])
        gene_panels.append((gene, inv_id, phyD, phyI))

    # Estimate size: row height 0.28", width generous (polymorphic columns unknown pre-filter)
    # We'll fix width wide for readability
    fig_w = 16.0
    # Height per haplotype row ~0.28", min panel height ~2.5"
    heights = []
    for _, _, phyD, phyI in gene_panels:
        nrows = len(phyI["seq_order"]) + len(phyD["seq_order"])
        heights.append(max(2.5, nrows * 0.28))
    fig_h = sum(heights) + (len(heights) - 1) * 0.8  # gaps between panels

    fig = plt.figure(figsize=(fig_w, fig_h))
    top = 0.98; bottom = 0.05
    total_h = top - bottom
    gaps = 0.02 * (len(heights) - 1)
    slot_total = total_h - gaps
    slot_fracs = [h / sum(heights) for h in heights]

    current_top = top
    for (gene, inv_id, phyD, phyI), frac in zip(gene_panels, slot_fracs):
        slot_h = slot_total * frac
        ax = fig.add_axes([0.08, current_top - slot_h, 0.88, slot_h])
        plot_fixed_diff_panel(ax, phyD, phyI, gene, inv_id, threshold=FIXED_DIFF_UNANIMITY_THRESHOLD)
        current_top = current_top - slot_h - 0.02  # gap

    # Legend for base colors (A/C/G/T)
    base_handles = [mpatches.Patch(color=BASE_COLORS[i], label=b) for b, i in BASE_TO_IDX.items()]
    fig.legend(handles=base_handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Main
# =============================================================================

def main():
    cds_summary = load_cds_summary()
    gene_tests  = load_gene_tests()
    pairs_index = build_pairs_and_phy_index(cds_summary)

    # FIGURE 2A
    plot_fig2A(cds_summary, FIG2A_FILE)

    # FIGURE 2B
    plot_fig2B(cds_summary, FIG2B_FILE)

    # FIGURE 3 (slopegraph + paired p-values)
    locus_df = aggregate_by_inversion(cds_summary)
    plot_fig3_slopegraph(locus_df, FIG3_FILE)

    # FIGURE 4 (volcano)
    volcano_df = prepare_volcano(gene_tests, cds_summary)
    plot_fig4_volcano(volcano_df, FIG4_FILE)

    # FIGURE 5 (per-gene rows: identity matrix + MST)
    plot_fig5_smallmultiples(volcano_df, cds_summary, pairs_index, FIG5_FILE)

    # FIGURE 6 (raw haplotypes: MAPT & SPPL2C; polymorphic columns only)
    plot_fig6_fixed_differences(cds_summary, pairs_index, FIG6_FILE)

if __name__ == "__main__":
    main()
