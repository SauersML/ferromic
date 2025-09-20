from __future__ import annotations
import logging, re, sys, time, subprocess, shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from scipy.stats import spearmanr, sem

# ------------------------- CONFIG -------------------------

INV_TSV        = Path("inv_info.tsv")  # recurrence mapping input

DIVERSITY_FILE = Path("per_site_diversity_output.falsta")
FST_FILE       = Path("per_site_fst_output.falsta")

OUTDIR         = Path("length_norm_trend_fast")

MIN_LEN_PI     = 100_000
MIN_LEN_FST    = 100_000

MAX_BP         = 100_000          # cap distance from inversion edge

# Proportion mode
NUM_BINS_PROP  = 50

# Base-pair mode 
NUM_BINS_BP    = 50               # number of bins between 0..MAX_BP

# Plotting/analysis rules
LOWESS_FRAC     = 0.4
MIN_INV_PER_BIN = 5               # if <5 inversions in a bin → don't plot that bin

# Visual
SCATTER_SIZE   = 34
SCATTER_ALPHA  = 0.10   # transparent dots
LINE_WIDTH     = 2.8
BAND_ALPHA     = 0.1

# Y-axis padding as a fraction of the smoothed-line span
Y_PAD_FRAC     = 0.1

# ----------- Color scheme & formatting (match example) ------------
# Orientation colors
COLOR_DIRECT   = "#1f3b78"   # dark blue
COLOR_INVERTED = "#8c2d7e"   # reddish purple

# Overlays for recurrence coding
OVERLAY_SINGLE = "#d9d9d9"   # very light gray (circles)
OVERLAY_RECUR  = "#4a4a4a"   # dark gray (small dashes)

# Overall aggregate color (black per instruction)
COLOR_OVERALL  = "#000000"

AX_TEXT        = "#333333"   # labels/ticks

# Matplotlib rcParams for a professional look
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Liberation Sans", "Ubuntu", "Arial"],
    "font.size": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": AX_TEXT,
    "xtick.color": AX_TEXT,
    "ytick.color": AX_TEXT,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.borderaxespad": 0.3,
})

N_CORES        = max(1, mp.cpu_count() - 1)
OPEN_PLOTS_ON_LINUX = True  # auto open generated PDFs using `xdg-open` if available

# Hudson specifics
EPS_DENOM        = 1e-12         # treat denom <= EPS as uninformative
BOOTSTRAP_REPS   = 300           # inversion bootstrap reps for pooled curve SE

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("len_norm_fast_grouped")

# ---------------------- REGEX & PARSING --------------------

# IMPORTANT: use only FILTERED π and capture orientation group (_0 = direct, _1 = inverted)
_RE_PI = re.compile(
    r">.*?filtered_pi.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)(?:_group_([01]))?",
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

def _load_inv_mapping(INV_TSV: Path) -> pd.DataFrame:
    """
    Load inv_info.tsv robustly; pull Chromosome/Start/End and recurrence flag.

    Recurrence logic:
      - If column '0_single_1_recur_consensus' exists and equals 1 → recurrent; 0 → single-event
      - If missing or NA → uncategorized
    """
    if not INV_TSV.is_file():
        log.warning(f"INV tsv not found: {INV_TSV} → all sequences will be uncategorized.")
        return pd.DataFrame(columns=["chrom", "start", "end", "group"])

    df = pd.read_csv(INV_TSV, sep="\t")
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    recur_col = None
    for candidate in ["0_single_1_recur_consensus"]:
        if candidate in df.columns:
            recur_col = candidate
            break

    if "Chromosome" not in df.columns or "Start" not in df.columns or "End" not in df.columns:
        raise ValueError("inv_info.tsv must contain 'Chromosome', 'Start', 'End' columns.")

    df["_chrom"] = df["Chromosome"].map(_norm_chr)
    df["_start"] = pd.to_numeric(df["Start"], errors="coerce").astype("Int64")
    df["_end"]   = pd.to_numeric(df["End"],   errors="coerce").astype("Int64")

    if recur_col is not None:
        rc = pd.to_numeric(df[recur_col], errors="coerce")
        group = pd.Series(
            np.where(rc == 1, "recurrent", np.where(rc == 0, "single-event", "uncategorized")),
            index=df.index,
        )
    else:
        group = pd.Series("uncategorized", index=df.index)

    mask = df["_chrom"].notna() & df["_start"].notna() & df["_end"].notna()
    out = df.loc[mask, ["_chrom", "_start", "_end"]].copy()
    out.rename(columns={"_chrom": "chrom", "_start": "start", "_end": "end"}, inplace=True)
    out["group"] = group.loc[out.index].values
    out["start"] = out["start"].astype(int)
    out["end"]   = out["end"].astype(int)

    n_groups = Counter(out["group"])
    log.info(f"Loaded inversion mapping: recurrent={n_groups.get('recurrent',0)}, "
             f"single-event={n_groups.get('single-event',0)}, "
             f"uncategorized(by tsv)={n_groups.get('uncategorized',0)}")

    return out

def _build_fuzzy_lookup(inv_df: pd.DataFrame) -> Dict[Tuple[str,int,int], str]:
    """
    Build a fuzzy (±1 bp) lookup: for each (chrom, start, end),
    create keys for all combinations of start±{0,1} and end±{0,1}.
    Resolve collisions by priority: recurrent > single-event > uncategorized.
    """
    prio = {"recurrent": 2, "single-event": 1, "uncategorized": 0}
    lut: Dict[Tuple[str,int,int], Tuple[str,int]] = {}

    for chrom, s, e, g in inv_df[["chrom","start","end","group"]].itertuples(index=False):
        for ds in (-1, 0, 1):
            for de in (-1, 0, 1):
                key = (chrom, s + ds, e + de)
                if key in lut:
                    if prio[g] > lut[key][1]:
                        lut[key] = (g, prio[g])
                else:
                    lut[key] = (g, prio[g])

    return {k: v[0] for k, v in lut.items()}

# -------------------- FALSTA ITERATION ----------------------

def _iter_falsta(file_path: Path, which: str, min_len: int):
    """
    Yields dicts:
        {
          "header": str,
          "coords": {"chrom": str, "start": int, "end": int},
          "data": np.ndarray,
          "length": int,
          "orient": 'direct'|'inverted'|None
        }
    which ∈ {'pi','hudson'}
    """
    if which not in ("pi","hudson"):
        raise ValueError("which must be 'pi' or 'hudson'")
    if not file_path.is_file():
        log.error(f"File not found: {file_path}")
        return

    rx = _RE_PI if which=="pi" else _RE_HUD
    total, loaded, skip_len = 0, 0, 0

    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if not line: continue
            if line[0] == ">":
                header = line
                total += 1
                continue
            if header is None: continue
            m = rx.search(header)
            if not m:
                header = None
                continue

            chrom, s, e = _norm_chr(m.group(1)), int(m.group(2)), int(m.group(3))
            orient = None
            if which == "pi":
                gid = m.group(4)
                if gid is not None:
                    orient = "direct" if gid == "0" else "inverted"

            data = _parse_values_fast(line)
            exp_len = e - s + 1
            if data.size != exp_len:
                raise RuntimeError(
                    f"Parsed values length {data.size} does not match header bounds {exp_len} "
                    f"for metric '{which}' in {file_path} with header: {header}"
                )
            if data.size < min_len or np.all(np.isnan(data)):
                skip_len += 1
                header = None
                continue

            yield {
                "header": header,
                "coords": {"chrom": chrom, "start": s, "end": e},
                "data": data,
                "length": int(data.size),
                "orient": orient
            }
            loaded += 1
            header = None

    log.info(f"[{which}] headers={total}, loaded={loaded}, skipped_len={skip_len}")

# --- Gaussian KDE-style smoothing across bins (kernel regression) ---
def _kernel_regress_1d(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: Optional[np.ndarray] = None,
    frac: Optional[float] = None,          # SINGLE knob (0..1-ish). If None, uses 0.4.
    bw_bins: Optional[float] = None        # ignored (kept only for call-compat)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian kernel regression with a *single* smoothness knob `frac`.
    - Local bandwidth widens near the edges to avoid boundary bias (no KNN).
    - `bw_bins` is ignored on purpose (kept to avoid changing callers).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        xe = np.asarray(x_eval, float) if x_eval is not None else x[ok]
        if xe is None or xe.size == ok.sum():
            return x[ok], y[ok]
        out = np.interp(xe, x[ok], y[ok], left=np.nan, right=np.nan)
        return xe, out

    xs = x[ok]
    ys = y[ok]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    xe = xs if x_eval is None else np.asarray(x_eval, dtype=float)

    # median spacing → convert “bin units” to x-scale
    if xs.size > 1:
        dx = float(np.median(np.diff(np.unique(xs))))
    else:
        dx = 1.0

    # ----- single knob → base sigma in *bin units* -----
    f = 1.0 if (frac is None or not np.isfinite(float(frac))) else float(frac)
    # map 'frac' to σ so ~95% mass spans ≈ frac * n_points bins
    base_sigma_bins = max(0.5, (f * max(xs.size, 1.0)) / 4.0)

    # ----- local edge-aware boost (no extra knobs) -----
    # normalize eval positions to [0,1]
    xmin, xmax = float(xs.min()), float(xs.max())
    span = max(xmax - xmin, 1e-12)
    r = (xe - xmin) / span
    # parabola: 4r(1-r) peaks at center=1, 0 at edges → invert to boost edges
    edge_boost = 1.0 + (1.0 - 4.0 * r * (1.0 - r))   # 1.0 (center) → 2.0 (edges)

    sigma_bins = np.maximum(0.5, base_sigma_bins * edge_boost)  # floor to avoid degeneracy
    h = np.maximum(1e-12, sigma_bins * dx)                      # convert to x-scale

    # Gaussian weights with per-eval bandwidth
    d = (xe[:, None] - xs[None, :]) / h[:, None]
    W = np.exp(-0.5 * d * d)

    num = W @ ys
    den = W.sum(axis=1)

    ye = np.full_like(xe, np.nan, dtype=float)
    nz = den > 0
    ye[nz] = num[nz] / den[nz]
    return xe, ye


def _smooth_series(x: np.ndarray, y: np.ndarray, frac: float = LOWESS_FRAC) -> Tuple[np.ndarray, np.ndarray]:
    """
    KDE-style smoothing across bins (Gaussian kernel regression).
    Returns xs (sorted finite x) and smoothed y on that same grid.
    """
    return _kernel_regress_1d(x, y, x_eval=None, frac=frac)


def _smooth_sem_to(xs_target: np.ndarray,
                   x_raw: np.ndarray,
                   e_raw: np.ndarray,
                   frac: float = LOWESS_FRAC * 0.8) -> np.ndarray:
    """
    Smooth SEM across bins with the same Gaussian kernel approach,
    then evaluate on xs_target. Clips negatives to 0.
    """
    _, es = _kernel_regress_1d(x_raw, e_raw, x_eval=xs_target, frac=frac)
    if es.size:
        es = np.where(es < 0, 0, es)
    return es


# --------------- BIN EDGES (shared in workers) --------------

_BIN_EDGES: Optional[np.ndarray] = None
_NUM_BINS: Optional[int] = None
_MODE: Optional[str] = None   # 'proportion' or 'bp'
_MAX_BP: Optional[int] = None
_BIN_RETURN_SUMS: Optional[bool] = None

# Hudson globals for pooled & median variants
_HUDSON_SUMS: Optional[dict] = None   # {"num": defaultdict(np.array), "den": defaultdict(np.array)}
_HUDSON_PERINV: Optional[dict] = None # {"num": defaultdict(list[np.array]), "den": defaultdict(list[np.array]), "cnt": defaultdict(list[np.array])}

def _pool_init(mode: str, num_bins: int, max_bp: Optional[int], return_sums: bool = False):
    """
    Initializer for workers: set global bin edges and mode.
    - proportion: bins across [0, 1]
    - bp:         bins across [0, MAX_BP]
    When return_sums is True, the binning kernel returns per-bin sums instead of means.
    """
    global _BIN_EDGES, _NUM_BINS, _MODE, _MAX_BP, _BIN_RETURN_SUMS
    _MODE = mode
    _NUM_BINS = int(num_bins)
    _MAX_BP = int(max_bp) if max_bp is not None else None
    _BIN_RETURN_SUMS = bool(return_sums)

    if mode == "proportion":
        _BIN_EDGES = np.linspace(0.0, 1.0, _NUM_BINS + 1, dtype=np.float64)
        _BIN_EDGES[-1] = _BIN_EDGES[-1] + 1e-9
    elif mode == "bp":
        if _MAX_BP is None or _MAX_BP <= 0:
            raise ValueError("MAX_BP must be positive for bp mode.")
        _BIN_EDGES = np.linspace(0.0, float(_MAX_BP), _NUM_BINS + 1, dtype=np.float64)
        _BIN_EDGES[-1] = _BIN_EDGES[-1] + 1e-9
    else:
        raise ValueError("mode must be 'proportion' or 'bp'")

def _bin_one_sequence(seq: np.ndarray) -> Optional[np.ndarray]:
    """
    Map one sequence to distance-from-inversion-edge based on global _MODE, then bin.
    When _BIN_RETURN_SUMS is True, returns per-bin sums of values. Otherwise returns per-bin means.
    Length is _NUM_BINS_. Bins with no data are NaN for means, and 0.0 for sums.
    """
    global _BIN_EDGES, _NUM_BINS, _MODE, _MAX_BP, _BIN_RETURN_SUMS
    if _BIN_EDGES is None or _NUM_BINS is None or _MODE is None:
        raise RuntimeError("Worker not initialized with _pool_init.")
    L = int(seq.shape[0])
    if L < 2:
        return None

    idx = np.arange(L, dtype=np.float64)
    valid = ~np.isnan(seq)
    if not np.any(valid):
        return None

    dist_bp_full = np.minimum(idx, (L - 1) - idx)

    if _MODE == "proportion":
        halfspan = (L - 1) / 2.0
        dc_center = np.abs(idx - halfspan) / max(halfspan, 1e-9)
        dc_center = np.clip(dc_center, 0.0, 1.0)

        use = valid
        if _MAX_BP is not None:
            use = valid & (dist_bp_full <= float(_MAX_BP))
        if not np.any(use):
            return None

        xvals = dc_center[use]
        vv    = seq[use].astype(np.float64)

    elif _MODE == "bp":
        use = valid
        if _MAX_BP is not None:
            use = valid & (dist_bp_full <= float(_MAX_BP))
        if not np.any(use):
            return None

        xvals = dist_bp_full[use]
        vv    = seq[use].astype(np.float64)
    else:
        raise RuntimeError("Unknown mode in _bin_one_sequence")

    bi = np.digitize(xvals, _BIN_EDGES[1:], right=False)

    sums   = np.bincount(bi, weights=vv, minlength=_NUM_BINS).astype(np.float64)
    counts = np.bincount(bi, minlength=_NUM_BINS).astype(np.int32)

    if _BIN_RETURN_SUMS:
        return sums

    means = np.full(_NUM_BINS, np.nan, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    return means


# --------------------- AGGREGATION --------------------------

def _aggregate_unweighted_mean(per_seq_means: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean across sequences per bin; SEM across sequences; n_seq per bin.
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

def _aggregate_unweighted_median(per_seq_means: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Median across sequences per bin (median of per-sequence means); SEM computed as in mean case.
    """
    M = np.vstack(per_seq_means)  # [n_seq, num_bins]
    n_seq_per = np.sum(~np.isnan(M), axis=0)
    with np.errstate(invalid="ignore"):
        median_per = np.nanmedian(M, axis=0)
        se_per     = np.full(M.shape[1], np.nan, dtype=np.float64)
        mask = n_seq_per > 1
        if np.any(mask):
            se_per[mask] = sem(M[:, mask], axis=0, nan_policy="omit")
    return median_per, se_per, n_seq_per

def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    ok = ~np.isnan(x) & ~np.isnan(y)
    xx, yy = x[ok], y[ok]
    if xx.size < 5:
        return (None, None)
    rho, p = spearmanr(xx, yy)
    if np.isnan(rho) or np.isnan(p):
        return (None, None)
    return float(rho), float(p)

# -------------------- UTILS --------------------

def _maybe_open_path(path: Path):
    if not OPEN_PLOTS_ON_LINUX:
        return
    try:
        if sys.platform.startswith("linux") and shutil.which("xdg-open"):
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception as e:
        log.warning(f"Could not auto-open {path}: {e}")

# ----------------------- LEGEND PROXIES ---------------------

class PatternProxy:
    """Proxy for a single-color patterned line with a small gap and a marker in the gap."""
    def __init__(self, color: str, overlay: Optional[str] = None):
        self.color = color
        self.overlay = overlay  # 'single' or 'recurrent' or None

class AltPatternProxy:
    """Proxy for an alternating-color patterned line (blue/purple) with a marker in the gap."""
    def __init__(self, colors: Tuple[str, str], overlay: Optional[str] = None):
        self.colors = colors
        self.overlay = overlay

class PatternHandler(HandlerBase):
    """Draw: line — small gap — tiny marker — small gap — line (scaled to legend box)."""
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        y = y0 + height / 2.0
        lw = max(1.5, LINE_WIDTH * 0.65)
        gap = width * 0.10
        seg = (width - 2*gap) / 2.0
        artists = []
        # left segment
        artists.append(Line2D([x0, x0 + seg], [y, y],
                              color=getattr(orig_handle, "color", COLOR_OVERALL),
                              lw=lw, solid_capstyle="round", transform=trans))
        # marker in the central gap
        xm = x0 + seg + gap/2.0
        overlay = getattr(orig_handle, "overlay", None)
        if overlay == "single":
            artists.append(Line2D([xm], [y], linestyle="None", marker="o",
                                  markersize=3.5, markerfacecolor=OVERLAY_SINGLE,
                                  markeredgecolor=getattr(orig_handle, "color", COLOR_OVERALL), markeredgewidth=0.9, transform=trans))

        elif overlay == "recurrent":
            artists.append(Line2D([xm], [y], linestyle="None", marker="_",
                                  markersize=6.0, color=OVERLAY_RECUR, transform=trans))
        # right segment
        x1 = x0 + seg + gap
        artists.append(Line2D([x1, x1 + seg], [y, y],
                              color=getattr(orig_handle, "color", COLOR_OVERALL),
                              lw=lw, solid_capstyle="round", transform=trans))
        return artists

class AltPatternHandler(HandlerBase):
    """Like PatternHandler but left/right segments alternate blue/purple."""
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        y = y0 + height / 2.0
        lw = max(1.5, LINE_WIDTH * 0.65)
        gap = width * 0.10
        seg = (width - 2*gap) / 2.0
        artists = []
        c0, c1 = getattr(orig_handle, "colors", (COLOR_DIRECT, COLOR_INVERTED))
        artists.append(Line2D([x0, x0 + seg], [y, y], color=c0, lw=lw, solid_capstyle="round", transform=trans))
        xm = x0 + seg + gap/2.0
        overlay = getattr(orig_handle, "overlay", None)
        if overlay == "single":
            artists.append(Line2D([xm], [y], linestyle="None", marker="o",
                                  markersize=3.5, markerfacecolor=OVERLAY_SINGLE,
                                  markeredgecolor=c0, markeredgewidth=0.9, transform=trans))

        elif overlay == "recurrent":
            artists.append(Line2D([xm], [y], linestyle="None", marker="_",
                                  markersize=6.0, color=OVERLAY_RECUR, transform=trans))
        x1 = x0 + seg + gap
        artists.append(Line2D([x1, x1 + seg], [y, y], color=c1, lw=lw, solid_capstyle="round", transform=trans))
        return artists

def _legend_label_for(group_key: str, N: int, rho: Optional[float], p: Optional[float]) -> str:
    def ptxt(p_):
        if p_ is None or np.isnan(p_): return "N/A"
        return "<0.001" if p_ < 1e-3 else f"{p_:.3g}"
    def rtxt(r_):
        if r_ is None or np.isnan(r_): return "N/A"
        return f"{r_:.3f}"
    name_map = {
        "direct-single-event":   "Direct — single-event",
        "direct-recurrent":      "Direct — recurrent",
        "inverted-single-event": "Inverted — single-event",
        "inverted-recurrent":    "Inverted — recurrent",
        "single-event":          "FST — single-event",
        "recurrent":             "FST — recurrent",
        "overall":               "Overall",
    }
    base = name_map.get(group_key, group_key)
    return f"{base} (N={N}, ρ={rtxt(rho)} p={ptxt(p)})"

# ----------------------- PATTERNED LINES --------------------

def _draw_patterned_line(ax,
                         xs: np.ndarray,
                         ys: np.ndarray,
                         base_color: Optional[str] = None,
                         alt_colors: Optional[Tuple[str, str]] = None,
                         overlay: Optional[str] = None):
    """
    Draw a TRUE patterned line with real gaps:
      line — small gap (with tiny overlay marker) — line — ...
    - If alt_colors is provided, segments alternate between those two colors.
    - overlay: 'single' → light gray circle in gap; 'recurrent' → dark gray small dash in gap.
    """
    n = xs.size
    if n < 2:
        return

    # Choose number of pattern cycles based on resolution (aim ~20 markers)
    n_cycles = max(6, min(24, n // 20))
    # Points per cycle
    pts_per = max(4, n // n_cycles)
    on_len  = max(2, int(round(pts_per * 0.70)))   # visible line portion
    gap_len = max(1, pts_per - on_len)             # true gap

    idx = 0
    seg_idx = 0
    while idx < n - 1:
        j = min(n - 1, idx + on_len)
        if j - idx >= 1:
            xseg = xs[idx:j+1]
            yseg = ys[idx:j+1]
            color = (alt_colors[seg_idx % 2] if alt_colors is not None else base_color)
            ax.plot(xseg, yseg, color=color, lw=LINE_WIDTH, solid_capstyle="round", zorder=5)
            seg_idx += 1
        # gap with marker
        g0 = j + 1
        g1 = min(n - 1, g0 + gap_len - 1)
        if g0 <= g1:
            mid = (g0 + g1) // 2
            if overlay == "single":
                ax.plot(xs[mid:mid+1], ys[mid:mid+1], linestyle="None", marker="o",
                        markersize=3.6, markerfacecolor=OVERLAY_SINGLE, markeredgecolor=(base_color if base_color is not None else (alt_colors[0] if alt_colors is not None else COLOR_OVERALL)), markeredgewidth=0.9, zorder=6)

            elif overlay == "recurrent":
                ax.plot(xs[mid:mid+1], ys[mid:mid+1], linestyle="None", marker="_",
                        markersize=6.2, color=OVERLAY_RECUR, zorder=6)
        idx = g1 + 1

# ----------------------- PLOTTING CORE ----------------------

def _plot_multi(x_centers: np.ndarray,
                group_stats: Dict[str, dict],
                y_label: str,
                out_path: Path,
                x_label: str,
                metric: str):
    """
    Plot multiple groups on the same axes. group_stats[group] contains:
       { 'mean': np.ndarray, 'se': np.ndarray, 'n_per_bin': np.ndarray,
         'N_total': int, 'rho': float|None, 'p': float|None,
         'color': str, 'plot_mask': np.ndarray[bool] }
    """
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    fig.subplots_adjust(right=0.76)  # reserve space for legend so it never overlaps

    # Axis formatting
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Drawing order ensures overlays sit nicely
    if metric == "pi":
        draw_order = ["direct-recurrent", "direct-single-event",
                      "inverted-recurrent", "inverted-single-event", "overall"]
    else:
        draw_order = ["recurrent", "single-event", "overall"]

    legend_handles: List[object] = []
    legend_labels: List[str] = []
    handler_map: Dict[object, HandlerBase] = {}

    # Track y extents of *smoothed* lines only
    y_min, y_max = np.inf, -np.inf

    for grp in draw_order:
        if grp not in group_stats:
            continue
        st  = group_stats[grp]
        col = st["color"]
        mean_y = st["mean"].copy()
        se_y   = st["se"].copy()
        mask_allowed = st["plot_mask"].astype(bool)

        mean_y[~mask_allowed] = np.nan
        se_y[~mask_allowed]   = np.nan

        ok = ~np.isnan(x_centers) & ~np.isnan(mean_y)
        if ok.sum() < 5:
            log.warning(f"[plot] Not enough bins with data to plot for group '{grp}': {ok.sum()}")
            continue

        x = x_centers[ok]
        y = mean_y[ok]
        e = se_y[ok]

        # Smooth mean (LOWESS)
        xs, ys = _smooth_series(x, y, frac=LOWESS_FRAC)

        # Track y-range from smoothed lines (ignore scatter/bands)
        if ys.size:
            y_min = min(y_min, np.nanmin(ys))
            y_max = max(y_max, np.nanmax(ys))

        # Smooth SEM (LOWESS over e vs x), clamp to ≥0 and interpolate to xs
        es = _smooth_sem_to(xs, x, e, frac=LOWESS_FRAC*0.8)

        # Scatter of binned means (light alpha, tinted by base color)
        ax.scatter(x, y, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, color=col, edgecolors="none", zorder=3)

        # Patterned line drawing (REAL gaps + tiny markers)
        if metric == "hudson" and grp in ("single-event", "recurrent"):
            _draw_patterned_line(ax, xs, ys,
                                 base_color=None,
                                 alt_colors=(COLOR_DIRECT, COLOR_INVERTED),
                                 overlay=("single" if grp == "single-event" else "recurrent"))
            # Legend proxy (alternating colors)
            proxy = AltPatternProxy(colors=(COLOR_DIRECT, COLOR_INVERTED),
                                    overlay=("single" if grp == "single-event" else "recurrent"))
            handler = AltPatternHandler()
        elif grp == "overall":
            # Overall: solid black
            ax.plot(xs, ys, lw=LINE_WIDTH, color=COLOR_OVERALL, zorder=5)
            proxy  = PatternProxy(color=COLOR_OVERALL, overlay=None)
            handler = PatternHandler()
        else:
            # π by orientation (blue or purple) with recurrence-coded gap markers
            overlay = "single" if grp.endswith("single-event") else "recurrent"
            base_c  = col  # orientation color
            _draw_patterned_line(ax, xs, ys, base_color=base_c, alt_colors=None, overlay=overlay)
            proxy  = PatternProxy(color=base_c, overlay=overlay)
            handler = PatternHandler()

        # Smoothed uncertainty band (tinted by col)
        if np.any(~np.isnan(es)):
            m = ~np.isnan(es)
            ax.fill_between(xs[m], ys[m]-es[m], ys[m]+es[m],
                            color=col, alpha=BAND_ALPHA, edgecolor="none", zorder=2)

        legend_handles.append(proxy)
        legend_labels.append(_legend_label_for(grp, st['N_total'], st.get('rho'), st.get('p')))
        handler_map[proxy] = handler

    # Lock y-limits to smoothed lines (+ margin), ignoring scatter/bands
    if np.isfinite(y_min) and np.isfinite(y_max):
        span = y_max - y_min
        if span <= 0:  # flat-line edge case
            base = abs(y_max) if y_max != 0 else 1.0
            pad = Y_PAD_FRAC * base
        else:
            pad = Y_PAD_FRAC * span
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Legend OUTSIDE the axes (never overlaps data), with tiny markers
    if legend_handles:
        ax.legend(legend_handles, legend_labels,
                  handler_map=handler_map,
                  loc="upper left", bbox_to_anchor=(1.005, 1.0),
                  frameon=True, framealpha=0.92,
                  borderpad=0.5, labelspacing=0.8, handlelength=2.8, handletextpad=0.8)

    fig.tight_layout()
    out_pdf = out_path.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot → {out_pdf}")
    _maybe_open_path(out_pdf)

# --------------------- END-TO-END RUN -----------------------

def _collect_grouped_means(which: str,
                           falsta: Path,
                           min_len: int,
                           fuzzy_map: Dict[Tuple[str,int,int], str],
                           mode: str,
                           num_bins: int,
                           max_bp: Optional[int]) -> Tuple[Dict[str, List[np.ndarray]], Dict[str,int]]:
    """
    Iterate falsta, assign each record to a group using fuzzy_map (±1 bp),
    and compute per-sequence binned means for the requested mode ('proportion' or 'bp').
    For Hudson FST, uses numerator/denominator records to compute ratio-of-sums per bin.
    Returns:
       per_group_means: group -> list of per-seq values (means for π; per-seq ratios for FST)
       per_group_counts: group -> number of sequences contributing
    """
    global _HUDSON_SUMS, _HUDSON_PERINV
    per_group_means = defaultdict(list)
    per_group_counts = Counter()

    log.info(f"[{which}/{mode}] scanning sequences and assigning groups...")

    if which == "hudson":
        by_coords: Dict[Tuple[str,int,int], dict] = {}
        for rec in _iter_falsta(falsta, which=which, min_len=min_len):
            c = rec["coords"]
            key = (c["chrom"], c["start"], c["end"])
            recur = fuzzy_map.get(key, "uncategorized")
            d = by_coords.setdefault(key, {"recur": recur, "num": None, "den": None, "header": rec.get("header", "")})
            h = rec.get("header", "")
            hl = h.lower()
            if "numerator" in hl:
                d["num"] = rec["data"]
            elif "denominator" in hl:
                d["den"] = rec["data"]

        pairs = [(k, v) for k, v in by_coords.items() if v["num"] is not None and v["den"] is not None]
        if not pairs:
            log.warning(f"[{which}/{mode}] No numerator/denominator pairs found.")
            _HUDSON_SUMS = None
            _HUDSON_PERINV = None
            return per_group_means, per_group_counts

        groups = []
        nums = []
        dens = []
        for (chrom, s, e), v in pairs:
            gkey = v["recur"] if v["recur"] in ("single-event", "recurrent") else "uncategorized"
            groups.append(gkey)
            nums.append(v["num"])
            dens.append(v["den"])

        # Bin numerator and denominator sums across sites
        with mp.Pool(processes=N_CORES, initializer=_pool_init, initargs=(mode, num_bins, max_bp, True)) as pool:
            binned_num_sums = pool.map(_bin_one_sequence, nums, chunksize=max(1, len(nums)//(N_CORES*4) if nums else 1))
        with mp.Pool(processes=N_CORES, initializer=_pool_init, initargs=(mode, num_bins, max_bp, True)) as pool:
            binned_den_sums = pool.map(_bin_one_sequence, dens, chunksize=max(1, len(dens)//(N_CORES*4) if dens else 1))

        # Bin counts of informative sites (where per-site denom > EPS_DENOM)
        cnt_inputs = [np.where(d > EPS_DENOM, 1.0, np.nan).astype(np.float32) for d in dens]
        with mp.Pool(processes=N_CORES, initializer=_pool_init, initargs=(mode, num_bins, max_bp, True)) as pool:
            binned_cnt_sums = pool.map(_bin_one_sequence, cnt_inputs, chunksize=max(1, len(cnt_inputs)//(N_CORES*4) if cnt_inputs else 1))

        # Totals for pooled; per-inversion lists for median & bootstrap
        hud_num_tot = defaultdict(lambda: np.zeros(num_bins, dtype=float))
        hud_den_tot = defaultdict(lambda: np.zeros(num_bins, dtype=float))
        hud_perinv_num = defaultdict(list)
        hud_perinv_den = defaultdict(list)
        hud_perinv_cnt = defaultdict(list)

        for gkey, nsum, dsum, csum in zip(groups, binned_num_sums, binned_den_sums, binned_cnt_sums):
            if nsum is None or dsum is None or csum is None:
                continue
            # Per-inversion ratio-of-sums per bin (for MEDIAN view)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(dsum > EPS_DENOM, nsum / dsum, np.nan)
            per_group_means[gkey].append(ratio)
            per_group_counts[gkey] += 1
            if gkey != "uncategorized":
                per_group_means["overall"].append(ratio)
                per_group_counts["overall"] += 1

            # Store per-inversion sums for pooled & bootstrap
            hud_perinv_num[gkey].append(nsum)
            hud_perinv_den[gkey].append(dsum)
            hud_perinv_cnt[gkey].append(csum)
            if gkey != "uncategorized":
                hud_perinv_num["overall"].append(nsum)
                hud_perinv_den["overall"].append(dsum)
                hud_perinv_cnt["overall"].append(csum)

            # Update pooled totals
            hud_num_tot[gkey] += np.nan_to_num(nsum, nan=0.0)
            hud_den_tot[gkey] += np.nan_to_num(dsum, nan=0.0)
            if gkey != "uncategorized":
                hud_num_tot["overall"] += np.nan_to_num(nsum, nan=0.0)
                hud_den_tot["overall"] += np.nan_to_num(dsum, nan=0.0)

        _HUDSON_SUMS = {"num": hud_num_tot, "den": hud_den_tot}
        _HUDSON_PERINV = {"num": hud_perinv_num, "den": hud_perinv_den, "cnt": hud_perinv_cnt}

        for g in ["recurrent", "single-event", "uncategorized", "overall"]:
            if per_group_counts.get(g, 0):
                log.info(f"[{which}/{mode}] N {g:>22} = {per_group_counts[g]}")

        return per_group_means, per_group_counts

    # π pathway (unchanged): bin per-sequence means
    seqs_with_meta: List[Tuple[str, Optional[str], np.ndarray]] = []
    for rec in _iter_falsta(falsta, which=which, min_len=min_len):
        c = rec["coords"]
        key = (c["chrom"], c["start"], c["end"])
        recur = fuzzy_map.get(key, "uncategorized")
        orient = rec.get("orient", None)
        seqs_with_meta.append((recur, orient, rec["data"]))

    if not seqs_with_meta:
        log.warning(f"[{which}/{mode}] No sequences to bin.")
        return per_group_means, per_group_counts

    with mp.Pool(processes=N_CORES, initializer=_pool_init, initargs=(mode, num_bins, max_bp, False)) as pool:
        per_means = pool.map(
            _bin_one_sequence,
            [x[2] for x in seqs_with_meta],
            chunksize=max(1, len(seqs_with_meta)//(N_CORES*4) if seqs_with_meta else 1)
        )

    for (recur, orient, _), m in zip(seqs_with_meta, per_means):
        if m is None:
            continue

        if which == "pi":
            if orient in ("direct", "inverted") and recur in ("single-event", "recurrent"):
                gkey = f"{orient}-{recur}"
            else:
                gkey = "uncategorized"
        else:
            gkey = recur if recur in ("single-event", "recurrent") else "uncategorized"

        per_group_means[gkey].append(m)
        per_group_counts[gkey] += 1
        if gkey != "uncategorized":
            per_group_means["overall"].append(m)
            per_group_counts["overall"] += 1

    log_groups = (["direct-recurrent", "direct-single-event",
                   "inverted-recurrent", "inverted-single-event"]
                  if which == "pi" else ["recurrent", "single-event"])
    log_groups += ["uncategorized", "overall"]
    for g in log_groups:
        if per_group_counts.get(g, 0):
            log.info(f"[{which}/{mode}] N {g:>22} = {per_group_counts[g]}")

    return per_group_means, per_group_counts


def _bootstrap_pooled_se(perinv_num: List[np.ndarray],
                         perinv_den: List[np.ndarray],
                         reps: int = BOOTSTRAP_REPS) -> np.ndarray:
    """
    Bootstrap SE for pooled ratio-of-sums curve (resample inversions).
    Returns per-bin standard deviation of bootstrap estimates.
    """
    if not perinv_num or not perinv_den:
        return np.array([])
    N = len(perinv_num)
    num_mat = np.vstack(perinv_num)  # [N, B]
    den_mat = np.vstack(perinv_den)  # [N, B]
    B = num_mat.shape[1]
    boot = np.full((reps, B), np.nan, dtype=float)
    rng = np.random.default_rng(1337)
    for r in range(reps):
        idx = rng.integers(0, N, size=N)
        Ns = np.nansum(num_mat[idx, :], axis=0)
        Ds = np.nansum(den_mat[idx, :], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot[r, :] = np.where(Ds > EPS_DENOM, Ns / Ds, np.nan)
    return np.nanstd(boot, axis=0, ddof=1)


def _assemble_outputs(per_group_means: Dict[str, List[np.ndarray]],
                      per_group_counts: Dict[str,int],
                      which: str,
                      mode: str,
                      num_bins: int,
                      max_bp: Optional[int],
                      y_label: str,
                      out_path: Path,
                      out_tsv: Path,
                      agg_kind: str):
    """
    Build tables, compute stats, and plot for given mode.
    For Hudson FST:
      - agg_kind == 'median': median across inversions of per-inversion ratios (each inversion’s ratio is ratio-of-sums across sites per bin).
      - agg_kind == 'pooled': pooled ratio-of-sums across inversions (sum numerators/denominators first, then divide).
    For π: behavior unchanged (agg_kind ∈ {'mean','median'}).
    """
    global _HUDSON_SUMS, _HUDSON_PERINV

    if mode == "proportion":
        edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
        centers_dc = (edges[:-1] + edges[1:]) / 2.0
        dist_edge = 1.0 - centers_dc
        x_centers = dist_edge
        x_label   = "Normalized distance from inversion edge (0 = edge, 1 = center)"
    elif mode == "bp":
        assert max_bp is not None and max_bp > 0
        edges = np.linspace(0.0, float(max_bp), num_bins + 1, dtype=np.float64)
        x_centers = (edges[:-1] + edges[1:]) / 2.0
        x_label   = f"Distance from inversion edge (bp; capped at {max_bp:,})"
    else:
        raise ValueError("mode must be 'proportion' or 'bp'")

    if which == "pi":
        color_map = {
            "direct-recurrent":      COLOR_DIRECT,
            "direct-single-event":   COLOR_DIRECT,
            "inverted-recurrent":    COLOR_INVERTED,
            "inverted-single-event": COLOR_INVERTED,
            "overall":               COLOR_OVERALL,
        }
        group_iter = ["direct-recurrent","direct-single-event",
                      "inverted-recurrent","inverted-single-event","overall"]
    else:
        color_map = {
            "recurrent": COLOR_DIRECT,
            "single-event": COLOR_INVERTED,
            "overall": COLOR_OVERALL,
        }
        group_iter = ["recurrent","single-event","overall"]

    group_stats: Dict[str, dict] = {}
    all_rows = []

    agg_func = None
    if agg_kind == "mean":
        agg_func = _aggregate_unweighted_mean
    elif agg_kind == "median":
        agg_func = _aggregate_unweighted_median
    elif agg_kind == "pooled":
        agg_func = None
    else:
        raise ValueError("agg_kind must be one of {'mean','median','pooled'} for Hudson; {'mean','median'} for π.")

    for grp in group_iter:
        seqs = per_group_means.get(grp, [])
        if which == "pi":
            if not seqs:
                continue
            mean_per, se_per, nseq_per = agg_func(seqs)  # type: ignore[misc]
        else:
            # Hudson:
            if agg_kind == "median":
                if not seqs:
                    continue
                mean_per, se_per, nseq_per = _aggregate_unweighted_median(seqs)
            elif agg_kind == "pooled":
                # Build pooled curve from totals; SE from inversion bootstrap
                if _HUDSON_SUMS is None or _HUDSON_PERINV is None:
                    log.warning("[hudson/pooled] Missing pooled components.")
                    continue
                num_tot = _HUDSON_SUMS["num"].get(grp)
                den_tot = _HUDSON_SUMS["den"].get(grp)
                if num_tot is None or den_tot is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    pooled_curve = np.where(den_tot > EPS_DENOM, num_tot / den_tot, np.nan)

                perinv_num = _HUDSON_PERINV["num"].get(grp, [])
                perinv_den = _HUDSON_PERINV["den"].get(grp, [])
                # n inversions contributing per bin = count of inversions with positive denom in that bin
                if perinv_den:
                    den_mat = np.vstack(perinv_den)
                    nseq_per = np.sum(den_mat > EPS_DENOM, axis=0).astype(int)
                else:
                    nseq_per = np.zeros_like(pooled_curve, dtype=int)
                # Bootstrap SE (over inversions); band will render if SE is finite
                se_per = _bootstrap_pooled_se(perinv_num, perinv_den, reps=BOOTSTRAP_REPS)
                mean_per = pooled_curve
            else:
                # (Unweighted mean, not requested; keep for completeness if ever used)
                if not seqs:
                    continue
                mean_per, se_per, nseq_per = _aggregate_unweighted_mean(seqs)

        # Plot mask: require enough inversions in the bin
        plot_mask = (nseq_per >= MIN_INV_PER_BIN)

        # Correlation uses only allowed bins
        mean_for_corr = mean_per.copy()
        mean_for_corr[~plot_mask] = np.nan
        x_for_corr = x_centers.copy()
        x_for_corr[~plot_mask] = np.nan
        rho, p = _spearman(x_for_corr, mean_for_corr)

        group_stats[grp] = {
            "mean": mean_per,
            "se": se_per,
            "n_per_bin": nseq_per.astype(int),
            "N_total": per_group_counts.get(grp, 0),
            "rho": (np.nan if rho is None else rho),
            "p": (np.nan if p is None else p),
            "color": color_map[grp],
            "plot_mask": plot_mask,
        }

        for bi in range(num_bins):
            all_rows.append({
                "group": grp,
                "bin_index": bi,
                "x_center": x_centers[bi],
                "mean_value": mean_per[bi],
                "stderr_value": se_per[bi],
                "n_sequences_in_bin": int(nseq_per[bi]),
                "plotting_allowed": bool(plot_mask[bi]),
                "N_total_sequences_in_group": per_group_counts.get(grp, 0),
                "spearman_rho_over_allowed_bins": group_stats[grp]["rho"],
                "spearman_p_over_allowed_bins": group_stats[grp]["p"],
                "mode": mode,
                "metric": which,
                "aggregate": agg_kind,
            })

    # Save table (combined)
    df = pd.DataFrame(all_rows)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False, float_format="%.6g")
    log.info(f"Saved table → {out_tsv}")

    # Plot (grouped)
    _plot_multi(x_centers, group_stats, y_label, out_path, x_label, metric=which)

    # Plot (overall only)
    overall_only_path = out_path.with_name(f"{out_path.stem}_overall_only.pdf")
    overall_stats = {"overall": group_stats["overall"]} if "overall" in group_stats else {}
    _plot_multi(x_centers, overall_stats, y_label, overall_only_path, x_label, metric=which)

def run_metric(which: str,
               falsta: Path,
               min_len: int,
               fuzzy_map: Dict[Tuple[str,int,int], str],
               y_label: str,
               # proportion mode outputs
               out_plot_prop: Path,
               out_tsv_prop: Path,
               # bp mode outputs
               out_plot_bp: Path,
               out_tsv_bp: Path,
               agg_kind: str):
    """
    Run a metric end-to-end for both proportion and bp modes.
    π: agg_kind ∈ {'mean','median'} on per-sequence bin means.
    Hudson FST:
      - 'median' → median across inversions of per-inversion (ratio-of-sums) bin values.
      - 'pooled' → pooled ratio-of-sums across inversions.
    """
    t0 = time.time()

    # ---------- PROPORTION MODE (apply base-pair cap here too) ----------
    per_group_means_prop, per_group_counts_prop = _collect_grouped_means(
        which=which,
        falsta=falsta,
        min_len=min_len,
        fuzzy_map=fuzzy_map,
        mode="proportion",
        num_bins=NUM_BINS_PROP,
        max_bp=MAX_BP,     # cap for proportion mode too
    )
    total_loaded_prop = sum(per_group_counts_prop.values())
    if total_loaded_prop == 0:
        log.error(f"[{which}/proportion/{agg_kind}] No sequences loaded from {falsta}.")
    else:
        _assemble_outputs(
            per_group_means_prop, per_group_counts_prop,
            which=which, mode="proportion", num_bins=NUM_BINS_PROP, max_bp=MAX_BP,
            y_label=y_label,
            out_path=out_plot_prop,
            out_tsv=out_tsv_prop,
            agg_kind=agg_kind,
        )

    # ---------- BASE-PAIR MODE  ----------
    per_group_means_bp, per_group_counts_bp = _collect_grouped_means(
        which=which,
        falsta=falsta,
        min_len=min_len,
        fuzzy_map=fuzzy_map,
        mode="bp",
        num_bins=NUM_BINS_BP,
        max_bp=MAX_BP,
    )
    total_loaded_bp = sum(per_group_counts_bp.values())
    if total_loaded_bp == 0:
        log.error(f"[{which}/bp/{agg_kind}] No sequences loaded from {falsta}.")
    else:
        _assemble_outputs(
            per_group_means_bp, per_group_counts_bp,
            which=which, mode="bp", num_bins=NUM_BINS_BP, max_bp=MAX_BP,
            y_label=y_label,
            out_path=out_plot_bp,
            out_tsv=out_tsv_bp,
            agg_kind=agg_kind,
        )

    log.info(f"[{which}/{agg_kind}] done in {time.time() - t0:.2f}s\n")

# --------------------------- MAIN --------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load inversion mapping and build fuzzy (±1bp) lookup
    inv_df = _load_inv_mapping(INV_TSV)
    fuzzy_map = _build_fuzzy_lookup(inv_df) if not inv_df.empty else {}

    # π (diversity): produce MEAN-suffixed originals and MEDIAN-suffixed additions
    # --- MEAN ---
    run_metric(
        which="pi",
        falsta=DIVERSITY_FILE,
        min_len=MIN_LEN_PI,
        fuzzy_map=fuzzy_map,
        y_label="Mean nucleotide diversity (π per site)",
        # proportion mode outputs (now capped by MAX_BP too)
        out_plot_prop=OUTDIR / "pi_vs_inversion_edge_proportion_grouped_mean.pdf",
        out_tsv_prop=OUTDIR / "pi_vs_inversion_edge_proportion_grouped_mean.tsv",
        # bp mode outputs
        out_plot_bp=OUTDIR / f"pi_vs_inversion_edge_bp_cap{MAX_BP//1000}kb_grouped_mean.pdf",
        out_tsv_bp=OUTDIR / f"pi_vs_inversion_edge_bp_cap{MAX_BP//1000}kb_grouped_mean.tsv",
        agg_kind="mean",
    )
    # --- MEDIAN ---
    run_metric(
        which="pi",
        falsta=DIVERSITY_FILE,
        min_len=MIN_LEN_PI,
        fuzzy_map=fuzzy_map,
        y_label="Median nucleotide diversity (π per site)",
        # proportion mode outputs
        out_plot_prop=OUTDIR / "pi_vs_inversion_edge_proportion_grouped_median.pdf",
        out_tsv_prop=OUTDIR / "pi_vs_inversion_edge_proportion_grouped_median.tsv",
        # bp mode outputs
        out_plot_bp=OUTDIR / f"pi_vs_inversion_edge_bp_cap{MAX_BP//1000}kb_grouped_median.pdf",
        out_tsv_bp=OUTDIR / f"pi_vs_inversion_edge_bp_cap{MAX_BP//1000}kb_grouped_median.tsv",
        agg_kind="median",
    )

    # Hudson FST — produce POOLED and MEDIAN versions
    # --- POOLED (ratio-of-sums across inversions) ---
    run_metric(
        which="hudson",
        falsta=FST_FILE,
        min_len=MIN_LEN_FST,
        fuzzy_map=fuzzy_map,
        y_label="Hudson FST (pooled ratio-of-sums)",
        # proportion mode outputs
        out_plot_prop=OUTDIR / "fst_vs_inversion_edge_proportion_grouped_pooled.pdf",
        out_tsv_prop=OUTDIR / "fst_vs_inversion_edge_proportion_grouped_pooled.tsv",
        # bp mode outputs
        out_plot_bp=OUTDIR / f"fst_vs_inversion_edge_bp_cap{MAX_BP//1000}kb_grouped_pooled.pdf",
        out_tsv_bp=OUTDIR / f"fst_vs_inversion_edge_bp_cap{MAX_BP//1000}kb_grouped_pooled.tsv",
        agg_kind="pooled",
    )
    # --- MEDIAN (across inversions; per-inversion is ratio-of-sums across sites) ---
    run_metric(
        which="hudson",
        falsta=FST_FILE,
        min_len=MIN_LEN_FST,
        fuzzy_map=fuzzy_map,
        y_label="Hudson FST (median across inversions)",
        # proportion mode outputs
        out_plot_prop=OUTDIR / "fst_vs_inversion_edge_proportion_grouped_median.pdf",
        out_tsv_prop=OUTDIR / "fst_vs_inversion_edge_proportion_grouped_median.tsv",
        # bp mode outputs
        out_plot_bp=OUTDIR / f"fst_vs_inversion_edge_bp_cap{MAX_BP//1000}kb_grouped_median.pdf",
        out_tsv_bp=OUTDIR / f"fst_vs_inversion_edge_bp_cap{MAX_BP//1000}kb_grouped_median.tsv",
        agg_kind="median",
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
