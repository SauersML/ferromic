from __future__ import annotations
import logging, re, sys, time
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, sem
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------- CONFIG -------------------------

DIVERSITY_FILE = Path("per_site_diversity_output.falsta")
FST_FILE       = Path("per_site_fst_output.falsta")
OUTDIR         = Path("length_norm_trend_fast")

MIN_LEN_PI     = 150_000
MIN_LEN_FST    = 150_000

NUM_BINS       = 200
LOWESS_FRAC    = 0.4

# Visual
SCATTER_SIZE   = 36 
SCATTER_ALPHA  = 0.45
LINE_WIDTH     = 3.0
BAND_ALPHA     = 0.22

COLOR_LINE     = "#4F46E5"  # indigo-600
COLOR_DOTS     = "#22C55E"  # emerald-500
COLOR_BAND     = "#8B5CF6"  # violet-500

N_CORES        = max(1, mp.cpu_count() - 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("len_norm_fast")

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
    """
    Fast parser: replace 'NA' with 'nan' and use np.fromstring with sep=','.
    Keeps length intact and is much faster than per-token loops.
    """
    arr = np.fromstring(line.strip().replace("NA", "nan"), sep=",", dtype=np.float32)
    return arr

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

# -------------------- TESTS & PLOTTING ----------------------

def _spearman(dist_edge: np.ndarray, mean_y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    ok = ~np.isnan(dist_edge) & ~np.isnan(mean_y)
    x, y = dist_edge[ok], mean_y[ok]
    if x.size < 5: return (None, None)
    rho, p = spearmanr(x, y)
    if np.isnan(rho) or np.isnan(p): return (None, None)
    return float(rho), float(p)

def _plot(dist_edge: np.ndarray, mean_y: np.ndarray, se_y: np.ndarray,
          title: str, y_label: str, bins: int, out_png: Path):
    ok = ~np.isnan(dist_edge) & ~np.isnan(mean_y)
    if ok.sum() < 5:
        log.warning(f"Not enough bins with data to plot: {ok.sum()}")
        return
    x = dist_edge[ok]; y = mean_y[ok]; e = se_y[ok] if se_y is not None else np.full_like(y, np.nan)

    # LOWESS (more smoothing than before)
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

    rho, p = _spearman(dist_edge, mean_y)
    p_txt = "< 0.001" if (p is not None and p < 1e-3) else (f"{p:.3g}" if p is not None else "N/A")
    rho_txt = f"{rho:.3f}" if rho is not None else "N/A"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    ax.scatter(x, y, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, color=COLOR_DOTS,
               edgecolors="none", label=f"Binned mean ({bins} bins)")
    ax.plot(xs, ys, lw=LINE_WIDTH, color=COLOR_LINE, label=f"LOWESS (f={LOWESS_FRAC})")

    if np.any(~np.isnan(es)):
        m = ~np.isnan(es)
        ax.fill_between(xs[m], ys[m]-es[m], ys[m]+es[m], color=COLOR_BAND, alpha=BAND_ALPHA,
                        edgecolor="none", label="±1 SEM")

    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Normalized distance from segment edge (0 = edge, 1 = center)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    ax.text(0.98, 0.98, f"Spearman ρ = {rho_txt}\np-value = {p_txt}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CBD5E1", alpha=0.9))

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    log.info(f"Saved plot → {out_png}")

# --------------------- END-TO-END RUN -----------------------

def run_metric(which: str, falsta: Path, min_len: int,
               y_label: str, out_png: Path, out_csv: Path):
    t0 = time.time()
    # Load sequences
    seqs = [rec["data"] for rec in _iter_falsta(falsta, which=which, min_len=min_len)]
    if not seqs:
        log.error(f"[{which}] No sequences loaded from {falsta}."); return
    log.info(f"[{which}] sequences loaded: {len(seqs)} (min_len={min_len})")

    # Bin per sequence (parallel), edges computed once per worker
    log.info(f"[{which}] binning {len(seqs)} sequences into {NUM_BINS} bins using {N_CORES} cores...")
    with mp.Pool(processes=N_CORES, initializer=_pool_init, initargs=(NUM_BINS,)) as pool:
        per_means = pool.map(_bin_one_sequence, seqs, chunksize=max(1, len(seqs)//(N_CORES*4)))
    per_means = [m for m in per_means if m is not None]
    if not per_means:
        log.error(f"[{which}] Binning failed for all sequences."); return

    # Centers (dist_center) are fixed for equal-width bins
    edges = np.linspace(0.0, 1.0, NUM_BINS + 1, dtype=np.float64)
    centers_dc = (edges[:-1] + edges[1:]) / 2.0
    dist_edge   = 1.0 - centers_dc

    # Aggregate (unweighted) to match Script 5
    mean_per, se_per, nseq_per = _aggregate_unweighted(per_means)

    # Save CSV
    df = pd.DataFrame({
        "bin_index": np.arange(NUM_BINS),
        "dist_edge": dist_edge,
        "dist_center": centers_dc,
        "mean_value": mean_per,
        "stderr_value": se_per,
        "n_sequences": nseq_per.astype(int)
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6g")
    log.info(f"[{which}] saved CSV → {out_csv}")

    # Plot
    title = "Overall π vs. normalized distance from edge" if which=="pi" \
            else "Overall Hudson FST vs. normalized distance from edge"
    _plot(dist_edge, mean_per, se_per, title, y_label, NUM_BINS, out_png)

    log.info(f"[{which}] done in {time.time() - t0:.2f}s\n")

# --------------------------- MAIN --------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # π
    run_metric(
        which="pi",
        falsta=DIVERSITY_FILE,
        min_len=MIN_LEN_PI,
        y_label="Mean nucleotide diversity (π per site)",
        out_png=OUTDIR / f"pi_overall_vs_dist_edge_{NUM_BINS}bins.png",
        out_csv=OUTDIR / f"pi_overall_vs_dist_edge_{NUM_BINS}bins.csv",
    )

    # Hudson FST
    run_metric(
        which="hudson",
        falsta=FST_FILE,
        min_len=MIN_LEN_FST,
        y_label="Mean Hudson FST (per site)",
        out_png=OUTDIR / f"hudson_fst_overall_vs_dist_edge_{NUM_BINS}bins.png",
        out_csv=OUTDIR / f"hudson_fst_overall_vs_dist_edge_{NUM_BINS}bins.csv",
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
