from __future__ import annotations
import logging, re, sys, time, subprocess, shutil
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

INV_CSV        = Path("inv_info.csv")  # recurrence mapping input

DIVERSITY_FILE = Path("per_site_diversity_output.falsta")
FST_FILE       = Path("per_site_fst_output.falsta")

OUTDIR         = Path("length_norm_trend_fast")

MIN_LEN_PI     = 150_000
MIN_LEN_FST    = 150_000

# Proportion mode
NUM_BINS_PROP  = 250

# Base-pair mode 
MAX_BP         = 2_000_000          # cap distance from edge at 2 Mbp
NUM_BINS_BP    = 250                # number of bins between 0..MAX_BP

# Plotting/analysis rules
LOWESS_FRAC    = 0.4
MIN_INV_PER_BIN = 5                 # if <5 inversions in a bin → don't plot that bin

# Visual
SCATTER_SIZE   = 36
SCATTER_ALPHA  = 0.45
LINE_WIDTH     = 3.0
BAND_ALPHA     = 0.22

# Palette (3 lines → 3 distinct colors; keep prior look & feel)
COLOR_OVERALL  = "#4F46E5"  # indigo-600
COLOR_RECUR    = "#EF4444"  # red-500
COLOR_SINGLE   = "#22C55E"  # emerald-500

N_CORES        = max(1, mp.cpu_count() - 1)

OPEN_PLOTS_ON_LINUX = True  # auto open PNGs using `xdg-open` if available

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
    # Normalize column names
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

    # Determine group
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
    Yields dicts: {header, coords:{chrom,start,end}, data:np.ndarray, length:int}
    which ∈ {'pi','hudson'}
    """
    if which not in ("pi","hudson"):
        raise ValueError("which must be 'pi' or 'hudson'")
    if not file_path.is_file():
        log.error(f"File not found: {file_path}")
        return

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
                # Header does not correspond to the requested metric; skip this record without treating it as a formatting error.
                header = None
                continue
            chrom, s, e = _norm_chr(m.group(1)), int(m.group(2)), int(m.group(3))

            data = _parse_values_fast(line)
            # FALSTA sanity: the parsed vector length must exactly match the bounds from the header.
            exp_len = e - s + 1
            if data.size != exp_len:
                # Treat any formatting or parsing inconsistency as a fatal error to prevent silent data loss.
                raise RuntimeError(f"Parsed values length {data.size} does not match header bounds {exp_len} for metric '{which}' in {file_path} with header: {header}")
            if data.size < min_len or np.all(np.isnan(data)):
                # Data-quality filter: too short or entirely NaN is not a formatting/parsing failure.
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

# Globals set in pool initializer
_BIN_EDGES: Optional[np.ndarray] = None
_NUM_BINS: Optional[int] = None
_MODE: Optional[str] = None   # 'proportion' or 'bp'
_MAX_BP: Optional[int] = None

def _pool_init(mode: str, num_bins: int, max_bp: Optional[int]):
    """
    Initializer for workers: set global bin edges and mode.
    - proportion: bins across [0, 1]
    - bp:         bins across [0, MAX_BP]
    """
    global _BIN_EDGES, _NUM_BINS, _MODE, _MAX_BP
    _MODE = mode
    _NUM_BINS = int(num_bins)
    _MAX_BP = int(max_bp) if max_bp is not None else None

    if mode == "proportion":
        _BIN_EDGES = np.linspace(0.0, 1.0, _NUM_BINS + 1, dtype=np.float64)
        _BIN_EDGES[-1] = _BIN_EDGES[-1] + 1e-9  # to include right edge
    elif mode == "bp":
        if _MAX_BP is None or _MAX_BP <= 0:
            raise ValueError("MAX_BP must be positive for bp mode.")
        _BIN_EDGES = np.linspace(0.0, float(_MAX_BP), _NUM_BINS + 1, dtype=np.float64)
        _BIN_EDGES[-1] = _BIN_EDGES[-1] + 1e-9
    else:
        raise ValueError("mode must be 'proportion' or 'bp'")

def _bin_one_sequence(seq: np.ndarray) -> Optional[np.ndarray]:
    """
    Map one sequence to distance-from-edge based on global _MODE, then bin.
    Returns per-bin MEANS (length _NUM_BINS_, NaN where empty).
    - proportion mode: uses normalized distance-from-center (0=center→1=edge internally),
                       which is converted to distance-from-edge via (1 - center) later.
      Here we directly bin by 'center distance' in [0..1], because the down-stream code
      converts to 'distance from edge'.
    - bp mode: uses base-pair distance from nearest edge, capped at _MAX_BP.
               Only positions with distance<=_MAX_BP are binned.
    """
    global _BIN_EDGES, _NUM_BINS, _MODE, _MAX_BP
    if _BIN_EDGES is None or _NUM_BINS is None or _MODE is None:
        raise RuntimeError("Worker not initialized with _pool_init.")
    L = int(seq.shape[0])
    if L < 2:
        return None

    idx = np.arange(L, dtype=np.float64)
    valid = ~np.isnan(seq)
    if not np.any(valid):
        return None

    vv = seq[valid].astype(np.float64)

    if _MODE == "proportion":
        # 0=center → 1=edge
        dc_center = np.minimum(1.0, np.abs(idx - (L-1)/2.0) / (L/2.0))
        xvals = dc_center[valid]
    elif _MODE == "bp":
        # distance in *bp* to nearest edge, cap at _MAX_BP and keep only ≤ cap
        dist_bp = np.minimum(idx, (L - 1) - idx)  # 0 at edges, up to ~L/2 at center
        dist_bp = dist_bp[valid]
        keep = dist_bp <= float(_MAX_BP)
        if not np.any(keep):
            return None
        xvals = dist_bp[keep]
        vv = vv[keep]
    else:
        raise RuntimeError("Unknown mode in _bin_one_sequence")

    bi = np.digitize(xvals, _BIN_EDGES[1:], right=False)

    sums   = np.bincount(bi, weights=vv, minlength=_NUM_BINS).astype(np.float64)
    counts = np.bincount(bi, minlength=_NUM_BINS).astype(np.int32)

    means = np.full(_NUM_BINS, np.nan, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    return means

# --------------------- AGGREGATION --------------------------

def _aggregate_unweighted(per_seq_means: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def _maybe_open_png(path: Path):
    if not OPEN_PLOTS_ON_LINUX:
        return
    try:
        if sys.platform.startswith("linux") and shutil.which("xdg-open"):
            # Non-blocking open
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception as e:
        log.warning(f"Could not auto-open {path}: {e}")

# ----------------------- PLOTTING ---------------------------

def _plot_multi(x_centers: np.ndarray,
                group_stats: Dict[str, dict],
                y_label: str,
                title: str,
                out_png: Path,
                x_label: str):
    """
    Plot multiple groups on the same axes. group_stats[group] contains:
       { 'mean': np.ndarray, 'se': np.ndarray, 'n_per_bin': np.ndarray,
         'N_total': int, 'rho': float|None, 'p': float|None,
         'color': str, 'plot_mask': np.ndarray[bool] }
    x_centers: x coordinate for each bin center (same length as mean/se)
    x_label:   label string for x-axis
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    draw_order = ["recurrent", "single-event", "overall"]
    for grp in draw_order:
        if grp not in group_stats:
            continue
        st  = group_stats[grp]
        col = st["color"]
        mean_y = st["mean"].copy()
        se_y   = st["se"].copy()
        mask_allowed = st["plot_mask"].astype(bool)

        # Mask out bins with insufficient inversions
        mean_y[~mask_allowed] = np.nan
        se_y[~mask_allowed]   = np.nan

        ok = ~np.isnan(x_centers) & ~np.isnan(mean_y)
        if ok.sum() < 5:
            log.warning(f"[plot] Not enough bins with data to plot for group '{grp}': {ok.sum()}")
            continue

        x = x_centers[ok]; y = mean_y[ok]; e = se_y[ok]

        # LOWESS on allowed bins
        sm = lowess(y, x, frac=LOWESS_FRAC, it=1, return_sorted=True)
        xs, ys = sm[:, 0], sm[:, 1]

        # Interpolate SEM onto smooth x (only where available)
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

        # Label with Spearman
        rho = st.get("rho", np.nan)
        p   = st.get("p",   np.nan)
        if p is not None and not np.isnan(p):
            ptext = "<0.001" if p < 1e-3 else f"{p:.3g}"
        else:
            ptext = "N/A"
        label = f"{grp} (N={st['N_total']}, ρ={rho:.3f} p={ptext})"

        # Scatter of binned means (light alpha)
        ax.scatter(x, y, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, color=col, edgecolors="none", label=None)
        # Smooth line
        ax.plot(xs, ys, lw=LINE_WIDTH, color=col, label=label)
        # Shaded band
        if np.any(~np.isnan(es)):
            m = ~np.isnan(es)
            ax.fill_between(xs[m], ys[m]-es[m], ys[m]+es[m], color=col, alpha=BAND_ALPHA, edgecolor="none")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    log.info(f"Saved plot → {out_png}")
    _maybe_open_png(out_png)

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
    Returns:
       per_group_means: group -> list of per-seq means
       per_group_counts: group -> number of sequences contributing
    """
    per_group_means = defaultdict(list)
    per_group_counts = Counter()

    log.info(f"[{which}/{mode}] scanning sequences and assigning groups...")
    seqs_with_group = []

    for rec in _iter_falsta(falsta, which=which, min_len=min_len):
        c = rec["coords"]
        key = (c["chrom"], c["start"], c["end"])
        grp = fuzzy_map.get(key, "uncategorized")
        seqs_with_group.append((grp, rec["data"]))

    if not seqs_with_group:
        log.warning(f"[{which}/{mode}] No sequences to bin.")
        return per_group_means, per_group_counts

    # Bin all sequences (fast, in parallel)
    with mp.Pool(processes=N_CORES, initializer=_pool_init, initargs=(mode, num_bins, max_bp)) as pool:
        per_means = pool.map(
            _bin_one_sequence,
            [x[1] for x in seqs_with_group],
            chunksize=max(1, len(seqs_with_group)//(N_CORES*4) if seqs_with_group else 1)
        )

    # Collect by group (+ overall)
    for (grp, _), m in zip(seqs_with_group, per_means):
        if m is None:
            continue
        per_group_means[grp].append(m)
        per_group_counts[grp] += 1
        per_group_means["overall"].append(m)
        per_group_counts["overall"] += 1

    # Log counts
    for g in ["recurrent","single-event","uncategorized","overall"]:
        if per_group_counts.get(g, 0):
            log.info(f"[{which}/{mode}] N {g:>13} = {per_group_counts[g]}")

    return per_group_means, per_group_counts

def _assemble_outputs(per_group_means: Dict[str, List[np.ndarray]],
                      per_group_counts: Dict[str,int],
                      which: str,
                      mode: str,
                      num_bins: int,
                      max_bp: Optional[int],
                      y_label: str,
                      out_png: Path,
                      out_csv: Path):
    """
    Build tables, compute stats, and plot for given mode.
    """
    # Build x-axis centers for this mode
    if mode == "proportion":
        edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
        centers_dc = (edges[:-1] + edges[1:]) / 2.0  # distance-from-center (0=center,1=edge)
        dist_edge = 1.0 - centers_dc                  # 0=edge → 1=center (as label)
        x_centers = dist_edge
        x_label   = "Normalized distance from segment edge (0 = edge, 1 = center)"
    elif mode == "bp":
        assert max_bp is not None and max_bp > 0
        edges = np.linspace(0.0, float(max_bp), num_bins + 1, dtype=np.float64)
        x_centers = (edges[:-1] + edges[1:]) / 2.0  # bp from edge
        x_label   = f"Distance from segment edge (bp; capped at {max_bp:,})"
    else:
        raise ValueError("mode must be 'proportion' or 'bp'")

    # Aggregate per group
    color_map = {"recurrent": COLOR_RECUR, "single-event": COLOR_SINGLE, "overall": COLOR_OVERALL}
    group_stats = {}
    all_rows = []

    plot_title_core = {
        "pi": "π vs. distance from edge (grouped)",
        "hudson": "Hudson FST vs. distance from edge (grouped)"
    }[which]
    title = f"{plot_title_core} — {mode}"

    for grp in ["recurrent","single-event","overall"]:
        seqs = per_group_means.get(grp, [])
        if not seqs:
            continue
        mean_per, se_per, nseq_per = _aggregate_unweighted(seqs)

        # Apply plotting rule via mask (bins must have ≥ MIN_INV_PER_BIN inversions)
        plot_mask = (nseq_per >= MIN_INV_PER_BIN)

        # Spearman is computed ONLY on bins that pass the plotting rule
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

        # Save table rows
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
            })

    # Save CSV (combined)
    df = pd.DataFrame(all_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6g")
    log.info(f"Saved CSV → {out_csv}")

    # Plot
    # Render grouped plot including all available categories along with the overall aggregate.
    _plot_multi(x_centers, group_stats, y_label, title, out_png, x_label)

    # Render an additional plot that contains only the overall aggregate.
    # File name is derived deterministically from the grouped figure by appending "_overall_only" before the extension.
    overall_only_png = out_png.with_name(f"{out_png.stem}_overall_only.png")
    overall_only_title = f"{title} — overall only"
    overall_stats = {"overall": group_stats["overall"]} if "overall" in group_stats else {}
    _plot_multi(x_centers, overall_stats, y_label, overall_only_title, overall_only_png, x_label)

def run_metric(which: str,
               falsta: Path,
               min_len: int,
               fuzzy_map: Dict[Tuple[str,int,int], str],
               y_label: str,
               # proportion mode outputs
               out_png_prop: Path,
               out_csv_prop: Path,
               # bp mode outputs
               out_png_bp: Path,
               out_csv_bp: Path):
    t0 = time.time()

    # ---------- PROPORTION MODE ----------
    per_group_means_prop, per_group_counts_prop = _collect_grouped_means(
        which=which,
        falsta=falsta,
        min_len=min_len,
        fuzzy_map=fuzzy_map,
        mode="proportion",
        num_bins=NUM_BINS_PROP,
        max_bp=None,
    )
    total_loaded_prop = sum(per_group_counts_prop.values())
    if total_loaded_prop == 0:
        log.error(f"[{which}/proportion] No sequences loaded from {falsta}.")
    else:
        _assemble_outputs(
            per_group_means_prop, per_group_counts_prop,
            which=which, mode="proportion", num_bins=NUM_BINS_PROP, max_bp=None,
            y_label=y_label,
            out_png=out_png_prop,
            out_csv=out_csv_prop,
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
        log.error(f"[{which}/bp] No sequences loaded from {falsta}.")
    else:
        _assemble_outputs(
            per_group_means_bp, per_group_counts_bp,
            which=which, mode="bp", num_bins=NUM_BINS_BP, max_bp=MAX_BP,
            y_label=y_label,
            out_png=out_png_bp,
            out_csv=out_csv_bp,
        )

    log.info(f"[{which}] done in {time.time() - t0:.2f}s\n")

# --------------------------- MAIN --------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load inversion mapping and build fuzzy (±1bp) lookup
    inv_df = _load_inv_mapping(INV_CSV)
    fuzzy_map = _build_fuzzy_lookup(inv_df) if not inv_df.empty else {}

    # π (diversity)
    run_metric(
        which="pi",
        falsta=DIVERSITY_FILE,
        min_len=MIN_LEN_PI,
        fuzzy_map=fuzzy_map,
        y_label="Mean nucleotide diversity (π per site)",
        # proportion mode outputs (keep original filenames)
        out_png_prop=OUTDIR / f"pi_overall_vs_dist_edge_{NUM_BINS_PROP}bins.png",
        out_csv_prop=OUTDIR / f"pi_overall_vs_dist_edge_{NUM_BINS_PROP}bins.csv",
        # bp mode outputs 
        out_png_bp=OUTDIR / f"pi_overall_vs_dist_bp_cap{MAX_BP//1000}kb_{NUM_BINS_BP}bins.png",
        out_csv_bp=OUTDIR / f"pi_overall_vs_dist_bp_cap{MAX_BP//1000}kb_{NUM_BINS_BP}bins.csv",
    )

    # Hudson FST
    run_metric(
        which="hudson",
        falsta=FST_FILE,
        min_len=MIN_LEN_FST,
        fuzzy_map=fuzzy_map,
        y_label="Mean Hudson FST (per site)",
        # proportion mode outputs (keep original filenames)
        out_png_prop=OUTDIR / f"hudson_fst_overall_vs_dist_edge_{NUM_BINS_PROP}bins.png",
        out_csv_prop=OUTDIR / f"hudson_fst_overall_vs_dist_edge_{NUM_BINS_PROP}bins.csv",
        # bp mode outputs
        out_png_bp=OUTDIR / f"hudson_fst_overall_vs_dist_bp_cap{MAX_BP//1000}kb_{NUM_BINS_BP}bins.png",
        out_csv_bp=OUTDIR / f"hudson_fst_overall_vs_dist_bp_cap{MAX_BP//1000}kb_{NUM_BINS_BP}bins.csv",
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
