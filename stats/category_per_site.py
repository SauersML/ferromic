from __future__ import annotations
import logging, re, sys, time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, sem
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------- CONFIG -------------------------

INV_CSV       = Path("inv_info.csv")  # recurrence mapping input

DIVERSITY_FILE = Path("per_site_diversity_output.falsta")
FST_FILE       = Path("per_site_fst_output.falsta")
OUTDIR         = Path("length_norm_trend_fast")

MIN_LEN_PI     = 100_000
MIN_LEN_FST    = 100_000

NUM_BINS       = 100
LOWESS_FRAC    = 0.4

# Visual
SCATTER_SIZE   = 36
SCATTER_ALPHA  = 0.45
LINE_WIDTH     = 3.0
BAND_ALPHA     = 0.22

# Palette (3 lines → 3 distinct colors; keep prior look & feel)
COLOR_OVERALL  = "#4F46E5"  # indigo-600  (formerly line color)
COLOR_RECUR    = "#EF4444"  # red-500
COLOR_SINGLE   = "#22C55E"  # emerald-500 (formerly dots)
COLOR_BAND     = "#8B5CF6"  # violet-500 (shading, reused)

N_CORES        = max(1, mp.cpu_count() - 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("len_norm_fast_grouped")

# ---------------------- REGEX & PARSING --------------------

_RE_PI = re.compile(
    r">.*?filtered_pi.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)(?:_group_[01])?",
    re.IGNORECASE,
)
_RE_HUD = re.compile(
    r">.*?hudson_pairwise_fst.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)",
    re.IGNORECASE,
)

def _norm_chr(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith("chr_"): s = s[4:]
    elif s.startswith("chr"): s = s[3:]
    return f"chr{s}"

def _parse_values_fast(line: str) -> np.ndarray:
    """Fast parser: replace 'NA' with 'nan' and use np.fromstring with sep=','."""
    return np.fromstring(line.strip().replace("NA", "nan"), sep=",", dtype=np.float32)

# -------------------- INVERSION MAPPING --------------

def _load_inv_mapping(inv_csv: Path) -> pd.DataFrame:
    """
    Load inv_info.csv robustly; pull Chromosome/Start/End and recurrence flag.

    Recurrence logic:
      - If column '0_single_1_recur_consensus' exists and equals 1 → recurrent; 0 → single-event
      - If missing or NA → uncategorized
    """
    if not inv_csv.is_file():
        log.warning(f"INV CSV not found: {inv_csv} → all sequences will be uncategorized.")
        return pd.DataFrame(columns=["chrom", "start", "end", "group"])

    df = pd.read_csv(inv_csv, engine="python")
    # Normalize column names (tolerate weird headers)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Find the recurrence column
    recur_col = None
    for candidate in ["0_single_1_recur_consensus"]:
        if candidate in df.columns:
            recur_col = candidate
            break

    # Core fields
    if "Chromosome" not in df.columns or "Start" not in df.columns or "End" not in df.columns:
        raise ValueError("inv_info.csv must contain 'Chromosome', 'Start', 'End' columns.")

    df["_chrom"] = df["Chromosome"].map(_norm_chr)
    df["_start"] = pd.to_numeric(df["Start"], errors="coerce").astype("Int64")
    df["_end"]   = pd.to_numeric(df["End"],   errors="coerce").astype("Int64")

    # Determine group and align it to filtered rows using indexes
    if recur_col is not None:
        rc = pd.to_numeric(df[recur_col], errors="coerce")
        group = pd.Series(
            np.where(rc == 1, "recurrent", np.where(rc == 0, "single-event", "uncategorized")),
            index=df.index,
        )
    else:
        group = pd.Series("uncategorized", index=df.index)

    # Build filtered output and attach the aligned group labels
    mask = df["_chrom"].notna() & df["_start"].notna() & df["_end"].notna()
    out = df.loc[mask, ["_chrom", "_start", "_end"]].copy()
    out.rename(columns={"_chrom": "chrom", "_start": "start", "_end": "end"}, inplace=True)
    out["group"] = group.loc[out.index].values
    out["start"] = out["start"].astype(int)
    out["end"]   = out["end"].astype(int)

    n_groups = Counter(out["group"])
    log.info(f"Loaded inversion mapping: recurrent={n_groups.get('recurrent',0)}, "
             f"single-event={n_groups.get('single-event',0)}, "
             f"uncategorized(by CSV)={n_groups.get('uncategorized',0)}")

    return out

def _build_fuzzy_lookup(inv_df: pd.DataFrame) -> Dict[Tuple[str,int,int], str]:
    """
    Build a fuzzy (±1 bp) lookup: for each (chrom, start, end),
    create keys for all combinations of start±{0,1} and end±{0,1}.

    If multiple CSV rows collide on a fuzzy key with different groups,
    priority = recurrent > single-event > uncategorized.
    """
    prio = {"recurrent": 2, "single-event": 1, "uncategorized": 0}
    lut: Dict[Tuple[str,int,int], Tuple[str,int]] = {}  # key → (group, priority)

    for chrom, s, e, g in inv_df[["chrom","start","end","group"]].itertuples(index=False):
        for ds in (-1, 0, 1):
            for de in (-1, 0, 1):
                key = (chrom, s + ds, e + de)
                if key in lut:
                    if prio[g] > lut[key][1]:
                        lut[key] = (g, prio[g])
                else:
                    lut[key] = (g, prio[g])

    # Strip priorities
    return {k: v[0] for k, v in lut.items()}

# -------------------- FALSTA ITERATION ----------------------

def _iter_falsta(file_path: Path, which: str, min_len: int):
    """
    Yields dicts: {header, coords:{chrom,start,end}, data:np.ndarray, length:int}
    which ∈ {'pi','hudson'}
    """
    if which not in ("pi","hudson"):
        raise ValueError("which must be 'pi' or 'hudson'")
    if not file_path.is_file():
        log.error(f"File not found: {file_path}"); return

    rx = _RE_PI if which=="pi" else _RE_HUD
    total, loaded, skip_len, skip_mismatch = 0,0,0,0

    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if not line: continue
            if line[0] == ">":
                header = line
                total += 1
                continue
            # values line
            if header is None: continue
            m = rx.search(header)
            if not m:
                header = None
                continue
            chrom, s, e = _norm_chr(m.group(1)), int(m.group(2)), int(m.group(3))
            data = _parse_values_fast(line)
            # FALSTA sanity: length matches header bounds
            exp_len = e - s + 1
            if data.size != exp_len:
                skip_mismatch += 1
                header = None
                continue
            if data.size < min_len or np.all(np.isnan(data)):
                skip_len += 1
                header = None
                continue
            yield {
                "header": header,
                "coords": {"chrom": chrom, "start": s, "end": e},
                "data": data,
                "length": int(data.size)
            }
            loaded += 1
            header = None

    log.info(f"[{which}] headers={total}, loaded={loaded}, skipped_len={skip_len}, len_mismatch={skip_mismatch}")

# --------------- BIN EDGES (shared in workers) --------------

_BIN_EDGES = None
_NUM_BINS  = None

def _pool_init(num_bins: int):
    """Initializer to create global bin edges once per worker."""
    global _BIN_EDGES, _NUM_BINS
    _NUM_BINS  = int(num_bins)
    _BIN_EDGES = np.linspace(0.0, 1.0, _NUM_BINS + 1, dtype=np.float64)
    _BIN_EDGES[-1] = _BIN_EDGES[-1] + 1e-9

def _bin_one_sequence(seq: np.ndarray) -> Optional[np.ndarray]:
    """
    Map one sequence to normalized distance (center=0 → edge=1), then bin.
    Returns per-bin MEANS (length _NUM_BINS_, NaN where empty).
    """
    global _BIN_EDGES, _NUM_BINS
    L = int(seq.shape[0])
    if L < 2: return None

    # 0=center → 1=edge
    idx = np.arange(L, dtype=np.float64)
    dc  = np.minimum(1.0, np.abs(idx - (L-1)/2.0) / (L/2.0))

    valid = ~np.isnan(seq)
    if not np.any(valid): return None

    dc = dc[valid]
    vv = seq[valid].astype(np.float64)

    bi = np.digitize(dc, _BIN_EDGES[1:], right=False)

    sums   = np.bincount(bi, weights=vv, minlength=_NUM_BINS).astype(np.float64)
    counts = np.bincount(bi, minlength=_NUM_BINS).astype(np.int32)

    means = np.full(_NUM_BINS, np.nan, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    return means

# --------------------- AGGREGATION --------------------------

def _aggregate_unweighted(per_seq_means: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Script-5-style: mean across sequences per bin; SEM across sequences; n_seq per bin.
    """
    M = np.vstack(per_seq_means)  # [n_seq, num_bins]
    n_seq_per = np.sum(~np.isnan(M), axis=0)
    with np.errstate(invalid="ignore"):
        mean_per = np.nanmean(M, axis=0)
        se_per   = np.full(M.shape[1], np.nan, dtype=np.float64)
        mask = n_seq_per > 1
        if np.any(mask):
            se_per[mask] = sem(M[:, mask], axis=0, nan_policy="omit")
    return mean_per, se_per, n_seq_per

def _spearman(dist_edge: np.ndarray, mean_y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    ok = ~np.isnan(dist_edge) & ~np.isnan(mean_y)
    x, y = dist_edge[ok], mean_y[ok]
    if x.size < 5: return (None, None)
    rho, p = spearmanr(x, y)
    if np.isnan(rho) or np.isnan(p): return (None, None)
    return float(rho), float(p)

# ----------------------- PLOTTING ---------------------------

def _plot_multi(dist_edge: np.ndarray,
                group_stats: Dict[str, dict],
                y_label: str,
                title: str,
                out_png: Path):
    """
    Plot multiple groups on the same axes. group_stats[group] contains:
       { 'mean': np.ndarray, 'se': np.ndarray, 'n_per_bin': np.ndarray,
         'N_total': int, 'rho': float|None, 'p': float|None, 'color': str }
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # Order for consistent legend
    draw_order = ["recurrent", "single-event", "overall"]
    for grp in draw_order:
        if grp not in group_stats: continue
        st  = group_stats[grp]
        col = st["color"]
        mean_y = st["mean"]
        se_y   = st["se"]

        ok = ~np.isnan(dist_edge) & ~np.isnan(mean_y)
        if ok.sum() < 5:
            log.warning(f"[plot] Not enough bins with data to plot for group '{grp}': {ok.sum()}")
            continue
        x = dist_edge[ok]; y = mean_y[ok]; e = se_y[ok] if se_y is not None else np.full_like(y, np.nan)

        # LOWESS
        sm = lowess(y, x, frac=LOWESS_FRAC, it=1, return_sorted=True)
        xs, ys = sm[:, 0], sm[:, 1]

        # Interpolate SEM onto smooth x
        try:
            mask_e = ~np.isnan(e)
            if mask_e.sum() >= 2:
                order = np.argsort(x[mask_e])
                xi, ei = x[mask_e][order], e[mask_e][order]
                es = np.interp(xs, xi, ei, left=np.nan, right=np.nan)
            else:
                es = np.full_like(xs, np.nan)
        except Exception:
            es = np.full_like(xs, np.nan)

        label = f"{grp} (N={st['N_total']}, ρ={st['rho']:.3f} p={'<0.001' if (st['p'] is not None and st['p']<1e-3) else (f'{st['p']:.3g}' if st['p'] is not None else 'N/A')})"
        # Scatter of binned means (light alpha)
        ax.scatter(x, y, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, color=col, edgecolors="none", label=None)
        # Smooth line
        ax.plot(xs, ys, lw=LINE_WIDTH, color=col, label=label)
        # Shaded band
        if np.any(~np.isnan(es)):
            m = ~np.isnan(es)
            ax.fill_between(xs[m], ys[m]-es[m], ys[m]+es[m], color=col, alpha=BAND_ALPHA, edgecolor="none")

    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Normalized distance from segment edge (0 = edge, 1 = center)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    log.info(f"Saved plot → {out_png}")

# --------------------- END-TO-END RUN -----------------------

def _collect_grouped_means(which: str,
                           falsta: Path,
                           min_len: int,
                           fuzzy_map: Dict[Tuple[str,int,int], str]) -> Tuple[Dict[str, List[np.ndarray]], Dict[str,int]]:
    """
    Iterate falsta, assign each record to a group using fuzzy_map (±1 bp),
    and compute per-sequence binned means. Returns:
       per_group_means: group -> list of per-seq means
       per_group_counts: group -> number of sequences contributing
    """
    per_group_means = defaultdict(list)
    per_group_counts = Counter()

    log.info(f"[{which}] scanning sequences and assigning groups...")
    # Pre-warm pool once; we'll reuse it for binning
    with mp.Pool(processes=N_CORES, initializer=_pool_init, initargs=(NUM_BINS,)) as pool:
        # Batch up sequences then map
        seqs_with_group = []
        for rec in _iter_falsta(falsta, which=which, min_len=min_len):
            c = rec["coords"]
            key = (c["chrom"], c["start"], c["end"])
            grp = fuzzy_map.get(key, "uncategorized")
            # Always also contribute to 'overall' later (handled after binning)
            seqs_with_group.append((grp, rec["data"]))

        # Bin all sequences (fast, in parallel)
        per_means = pool.map(_bin_one_sequence, [x[1] for x in seqs_with_group],
                             chunksize=max(1, len(seqs_with_group)//(N_CORES*4) if seqs_with_group else 1))

    # Collect by group (+ overall)
    for (grp, _), m in zip(seqs_with_group, per_means):
        if m is None: continue
        per_group_means[grp].append(m)
        per_group_counts[grp] += 1
        per_group_means["overall"].append(m)
        per_group_counts["overall"] += 1

    # Log counts
    for g in ["recurrent","single-event","uncategorized","overall"]:
        if per_group_counts.get(g, 0):
            log.info(f"[{which}] N {g:>13} = {per_group_counts[g]}")

    return per_group_means, per_group_counts

def _assemble_outputs(per_group_means: Dict[str, List[np.ndarray]],
                      per_group_counts: Dict[str,int],
                      y_label: str,
                      title: str,
                      out_png: Path,
                      out_csv: Path):
    # Centers (dist_center) are fixed for equal-width bins
    edges = np.linspace(0.0, 1.0, NUM_BINS + 1, dtype=np.float64)
    centers_dc = (edges[:-1] + edges[1:]) / 2.0
    dist_edge   = 1.0 - centers_dc

    # Aggregate per group
    color_map = {"recurrent": COLOR_RECUR, "single-event": COLOR_SINGLE, "overall": COLOR_OVERALL}
    group_stats = {}
    all_rows = []

    for grp in ["recurrent","single-event","overall"]:
        seqs = per_group_means.get(grp, [])
        if not seqs:
            continue
        mean_per, se_per, nseq_per = _aggregate_unweighted(seqs)
        rho, p = _spearman(dist_edge, mean_per)
        group_stats[grp] = {
            "mean": mean_per,
            "se": se_per,
            "n_per_bin": nseq_per.astype(int),
            "N_total": per_group_counts.get(grp, 0),
            "rho": (np.nan if rho is None else rho),
            "p": (np.nan if p is None else p),
            "color": color_map[grp],
        }
        # Save table rows
        for bi in range(NUM_BINS):
            all_rows.append({
                "group": grp,
                "bin_index": bi,
                "dist_edge": dist_edge[bi],
                "dist_center": centers_dc[bi],
                "mean_value": mean_per[bi],
                "stderr_value": se_per[bi],
                "n_sequences_in_bin": int(nseq_per[bi]),
                "N_total_sequences": per_group_counts.get(grp, 0),
                "spearman_rho": group_stats[grp]["rho"],
                "spearman_p": group_stats[grp]["p"],
            })

    # Save CSV (combined)
    df = pd.DataFrame(all_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6g")
    log.info(f"Saved CSV → {out_csv}")

    # Plot
    _plot_multi(dist_edge, group_stats, y_label, title, out_png)

def run_metric(which: str, falsta: Path, min_len: int, fuzzy_map: Dict[Tuple[str,int,int], str],
               y_label: str, out_png: Path, out_csv: Path):
    t0 = time.time()

    per_group_means, per_group_counts = _collect_grouped_means(which, falsta, min_len, fuzzy_map)
    # If absolutely nothing loaded, bail
    total_loaded = sum(per_group_counts.values())
    if total_loaded == 0:
        log.error(f"[{which}] No sequences loaded from {falsta}."); return

    title = "π vs. normalized distance from edge (grouped)" if which=="pi" \
            else "Hudson FST vs. normalized distance from edge (grouped)"

    _assemble_outputs(per_group_means, per_group_counts, y_label, title, out_png, out_csv)

    log.info(f"[{which}] done in {time.time() - t0:.2f}s\n")

# --------------------------- MAIN --------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load inversion mapping and build fuzzy (±1bp) lookup
    inv_df = _load_inv_mapping(INV_CSV)
    fuzzy_map = _build_fuzzy_lookup(inv_df) if not inv_df.empty else {}

    # π
    run_metric(
        which="pi",
        falsta=DIVERSITY_FILE,
        min_len=MIN_LEN_PI,
        fuzzy_map=fuzzy_map,
        y_label="Mean nucleotide diversity (π per site)",
        out_png=OUTDIR / f"pi_overall_vs_dist_edge_{NUM_BINS}bins.png",     # keep filenames
        out_csv=OUTDIR / f"pi_overall_vs_dist_edge_{NUM_BINS}bins.csv",
    )

    # Hudson FST
    run_metric(
        which="hudson",
        falsta=FST_FILE,
        min_len=MIN_LEN_FST,
        fuzzy_map=fuzzy_map,
        y_label="Mean Hudson FST (per site)",
        out_png=OUTDIR / f"hudson_fst_overall_vs_dist_edge_{NUM_BINS}bins.png",
        out_csv=OUTDIR / f"hudson_fst_overall_vs_dist_edge_{NUM_BINS}bins.csv",
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
