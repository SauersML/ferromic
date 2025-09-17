from __future__ import annotations
import logging, re, sys, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

# ------------------------- CONFIG -------------------------

DIVERSITY_FILE = Path("per_site_diversity_output.falsta")
FST_FILE       = Path("per_site_fst_output.falsta")
OUTDIR         = Path("per_inversion_trends")

MIN_LEN_PI     = 150_000
MIN_LEN_FST    = 150_000

NUM_BINS       = 200
LOWESS_FRAC    = 0.4
MIN_VALID_BINS = 5        # minimum informative bins for Spearman

# Visual
SCATTER_SIZE   = 34 
SCATTER_ALPHA  = 0.50
LINE_WIDTH     = 3.0

COLOR_LINE     = "#4F46E5"  # indigo-600
COLOR_DOTS     = "#22C55E"  # emerald-500

N_CORES        = max(1, mp.cpu_count() - 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("per_inversion_trends")

# ---------------------- REGEX & PARSING --------------------

# Capture optional group for pi
_RE_PI = re.compile(
    r">.*?filtered_pi.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)(?:_group_([0-9]+))?",
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

@dataclass
class Record:
    which: str  # "pi" or "hudson"
    header: str
    chrom: str
    start: int
    end: int
    group: Optional[str]  # only for pi
    data: np.ndarray

    @property
    def length(self) -> int:
        return int(self.data.size)

    @property
    def inv_key(self) -> str:
        """Unique identity for outputs (safe for filenames)."""
        base = f"{self.chrom}_{self.start}_{self.end}"
        return f"{base}_group_{self.group}" if (self.group is not None) else base

def _iter_falsta(file_path: Path, which: str, min_len: int):
    """
    Yields Record objects for 'pi' or 'hudson'.
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
            if not line: 
                continue
            if line[0] == ">":
                header = line
                total += 1
                continue
            if header is None:
                continue

            m = rx.search(header)
            if not m:
                header = None
                continue

            chrom  = _norm_chr(m.group(1))
            s      = int(m.group(2))
            e      = int(m.group(3))
            group  = m.group(4) if (which=="pi" and m.lastindex and m.lastindex >= 4) else None

            data = _parse_values_fast(line)
            exp_len = e - s + 1
            if data.size != exp_len:
                skip_mismatch += 1
                header = None
                continue
            if data.size < min_len or np.all(np.isnan(data)):
                skip_len += 1
                header = None
                continue

            yield Record(which=which, header=header, chrom=chrom, start=s, end=e, group=group, data=data)
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
    _BIN_EDGES[-1] = _BIN_EDGES[-1] + 1e-9  # keep rightmost inclusive

def _bin_one_sequence(seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map one sequence to normalized distance (center=0 → edge=1), then bin.
    Returns (means, counts) per bin (length _NUM_BINS_, NaN where empty).
    """
    global _BIN_EDGES, _NUM_BINS
    L = int(seq.shape[0])
    if L < 2:
        return (np.full(_NUM_BINS, np.nan, dtype=np.float64),
                np.zeros(_NUM_BINS, dtype=np.int32))

    # 0=center → 1=edge
    idx = np.arange(L, dtype=np.float64)
    dc  = np.minimum(1.0, np.abs(idx - (L-1)/2.0) / (L/2.0))

    valid = ~np.isnan(seq)
    if not np.any(valid):
        return (np.full(_NUM_BINS, np.nan, dtype=np.float64),
                np.zeros(_NUM_BINS, dtype=np.int32))

    dc = dc[valid]
    vv = seq[valid].astype(np.float64)

    bi = np.digitize(dc, _BIN_EDGES[1:], right=False)  # 0..NUM_BINS-1

    sums   = np.bincount(bi, weights=vv, minlength=_NUM_BINS).astype(np.float64)
    counts = np.bincount(bi, minlength=_NUM_BINS).astype(np.int32)

    means = np.full(_NUM_BINS, np.nan, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    return means, counts

# --------------------- PER-INVERSION TASK -------------------

def _spearman_on_binned(dist_edge: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float], int]:
    ok = ~np.isnan(dist_edge) & ~np.isnan(y)
    if ok.sum() < MIN_VALID_BINS:
        return (None, None, int(ok.sum()))
    rho, p = spearmanr(dist_edge[ok], y[ok])
    if np.isnan(rho) or np.isnan(p):
        return (None, None, int(ok.sum()))
    return float(rho), float(p), int(ok.sum())

@dataclass
class InvResult:
    which: str
    inv_key: str
    chrom: str
    start: int
    end: int
    group: Optional[str]
    length: int
    means: np.ndarray       # per-bin means
    counts: np.ndarray      # per-bin counts
    rho: Optional[float]
    p: Optional[float]
    n_valid_bins: int

def _process_record(rec: Record, dist_edge: np.ndarray) -> InvResult:
    means, counts = _bin_one_sequence(rec.data)
    rho, p, n_ok = _spearman_on_binned(dist_edge, means)
    return InvResult(
        which=rec.which,
        inv_key=rec.inv_key,
        chrom=rec.chrom,
        start=rec.start,
        end=rec.end,
        group=rec.group,
        length=rec.length,
        means=means,
        counts=counts,
        rho=rho,
        p=p,
        n_valid_bins=n_ok,
    )

# -------------------- PLOTTING (per inversion) --------------

def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p))):
        return "N/A"
    if p < 1e-3:
        return "< 0.001"
    return f"{p:.3g}"

def _fmt_rho(rho: Optional[float]) -> str:
    return "N/A" if (rho is None or (isinstance(rho, float) and math.isnan(rho))) else f"{rho:.3f}"

def _fmt_q(q: Optional[float]) -> str:
    if q is None or (isinstance(q, float) and math.isnan(q)):
        return "N/A"
    if q < 1e-3:
        return "< 0.001"
    return f"{q:.3g}"

def _plot_one_inv(dist_edge: np.ndarray,
                  means: np.ndarray,
                  title: str,
                  y_label: str,
                  bins: int,
                  rho: Optional[float],
                  p_raw: Optional[float],
                  q_fdr: Optional[float],
                  out_png: Path):
    ok = ~np.isnan(dist_edge) & ~np.isnan(means)
    if ok.sum() < MIN_VALID_BINS:
        log.warning(f"Not enough informative bins to plot ({ok.sum()} < {MIN_VALID_BINS}): {out_png.name}")
        return

    x = dist_edge[ok]; y = means[ok]

    # LOWESS
    sm = lowess(y, x, frac=LOWESS_FRAC, it=1, return_sorted=True)
    xs, ys = sm[:, 0], sm[:, 1]

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(10.4, 6.4))

    ax.scatter(x, y, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, color=COLOR_DOTS,
               edgecolors="none", label=f"Binned mean ({bins} bins)")
    ax.plot(xs, ys, lw=LINE_WIDTH, color=COLOR_LINE, label=f"LOWESS (f={LOWESS_FRAC})")

    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Normalized distance from segment edge (0 = edge, 1 = center)", fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.set_title(title, fontsize=17)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=12)

    ax.text(0.98, 0.98,
            f"Spearman ρ = {_fmt_rho(rho)}\n"
            f"p-value = {_fmt_p(p_raw)}\n"
            f"FDR q = {_fmt_q(q_fdr)}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CBD5E1", alpha=0.9))

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    log.info(f"Saved plot → {out_png}")

# --------------------- END-TO-END (per metric) --------------

def _dist_edge_vector(num_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (edges, centers_dc, dist_edge) consistent with binning."""
    edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    centers_dc = (edges[:-1] + edges[1:]) / 2.0  # center=0 -> edge=1
    dist_edge = 1.0 - centers_dc                 # edge=0 -> center=1
    return edges, centers_dc, dist_edge

def run_metric_per_inversion(which: str,
                             falsta: Path,
                             min_len: int,
                             y_label: str,
                             outdir_metric: Path):
    t0 = time.time()
    outdir_metric.mkdir(parents=True, exist_ok=True)

    # Load records
    records = list(_iter_falsta(falsta, which=which, min_len=min_len))
    if not records:
        log.error(f"[{which}] No records loaded from {falsta}.")
        return
    log.info(f"[{which}] records loaded: {len(records)} (min_len={min_len})")

    # Fixed distance vector (shared across all inversions)
    _, _, dist_edge = _dist_edge_vector(NUM_BINS)

    # Parallel compute: per-inversion binning + Spearman
    log.info(f"[{which}] processing {len(records)} inversions into {NUM_BINS} bins using {N_CORES} cores...")
    with mp.Pool(processes=N_CORES, initializer=_pool_init, initargs=(NUM_BINS,)) as pool:
        results: List[InvResult] = pool.starmap(
            _process_record,
            [(rec, dist_edge) for rec in records],
            chunksize=max(1, len(records)//(N_CORES*4))
        )

    # FDR correction (Benjamini–Hochberg) for valid p-values only
    pvals = np.array([r.p for r in results], dtype=float)
    valid_mask = np.isfinite(pvals)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    if valid_mask.sum() > 0:
        _, p_corr, _, _ = multipletests(pvals[valid_mask], alpha=0.05, method="fdr_bh")
        qvals[valid_mask] = p_corr

    # Summary CSV
    summary_rows = []
    for r, q in zip(results, qvals):
        summary_rows.append({
            "inv_key": r.inv_key,
            "chrom": r.chrom,
            "start": r.start,
            "end": r.end,
            "group": r.group if r.group is not None else "",
            "length": r.length,
            "n_valid_bins": r.n_valid_bins,
            "rho": np.nan if r.rho is None else r.rho,
            "p_value": np.nan if r.p   is None else r.p,
            "q_fdr": q,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = outdir_metric / f"{which}_per_inversion_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    log.info(f"[{which}] wrote summary → {summary_csv}")

    # Per-inversion CSVs + plots (sequential to keep matplotlib happy)
    for r, q in zip(results, qvals):
        inv_dir = outdir_metric / "inversions"
        inv_dir.mkdir(parents=True, exist_ok=True)

        # Per-inversion CSV
        per_csv = inv_dir / f"{which}_{r.inv_key}_binned_{NUM_BINS}bins.csv"
        edges, centers_dc, dist_edge = _dist_edge_vector(NUM_BINS)
        df = pd.DataFrame({
            "bin_index": np.arange(NUM_BINS),
            "dist_edge": dist_edge,
            "dist_center": centers_dc,
            "mean_value": r.means,
            "n_sites_in_bin": r.counts.astype(int),
        })
        df.to_csv(per_csv, index=False, float_format="%.6g")

        # Plot
        title_bits = [
            "π vs. dist. from edge" if (which=="pi") else "Hudson FST vs. dist. from edge",
            f"{r.chrom}:{r.start:,}-{r.end:,}",
        ]
        if r.group is not None:
            title_bits.append(f"group {r.group}")
        title_bits.append(f"L={r.length:,}")
        title = " • ".join(title_bits)

        out_png = inv_dir / f"{which}_{r.inv_key}_vs_dist_edge_{NUM_BINS}bins.png"
        _plot_one_inv(
            dist_edge=dist_edge,
            means=r.means,
            title=title,
            y_label=y_label,
            bins=NUM_BINS,
            rho=r.rho,
            p_raw=r.p,
            q_fdr=float(q) if np.isfinite(q) else None,
            out_png=out_png
        )

    log.info(f"[{which}] per-inversion run done in {time.time() - t0:.2f}s\n")

# --------------------------- MAIN --------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # π (per inversion)
    run_metric_per_inversion(
        which="pi",
        falsta=DIVERSITY_FILE,
        min_len=MIN_LEN_PI,
        y_label="Nucleotide diversity π (per site)",
        outdir_metric=OUTDIR / "pi",
    )

    # Hudson FST (per inversion)
    run_metric_per_inversion(
        which="hudson",
        falsta=FST_FILE,
        min_len=MIN_LEN_FST,
        y_label="Hudson FST (per site)",
        outdir_metric=OUTDIR / "hudson",
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
