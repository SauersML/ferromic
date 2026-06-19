"""Breakpoint-vs-middle (edge decay) analysis for da and dxy, not FST.

The paper shows that Hudson FST is enriched near inversion breakpoints and
decays toward the centre (stats/fst_edge_decay.py, stats/middle_vs_flank_fst.py).
Reviewer 1's concern is that FST is mechanically tied to within-group diversity,
so this edge enrichment could be a diversity artefact. We therefore repeat the
analysis on the divergence scale.

The per-site Hudson FST tracks in data/per_site_fst_output.falsta(.gz) are stored
as numerator/denominator components. By construction in the Rust pipeline
(src/stats.rs, calculate_hudson_fst_per_site):

    numerator_i   = D_xy,i - 0.5 * (pi_1,i + pi_2,i)   = net divergence da per site
    denominator_i = D_xy,i                              = absolute divergence dxy per site

so the existing per-site files already contain per-site da and dxy directly; no
new data are introduced. We summarise each window as a mean over informative
sites (sum of the component over sites with a finite, positive dxy denominator).

Two complementary tests, mirroring the FST analyses:

1. Folded edge decay (cf. fst_edge_decay.py): two-sided Spearman correlation of
   per-site value vs distance from the nearest breakpoint, folded across the two
   flanks, per inversion; BH-FDR across inversions; reported by recurrence class.
2. Middle vs flank (cf. middle_vs_flank_fst.py): per inversion, mean da and dxy
   in the flank windows vs the middle window (40 kb total: 10 kb each flank,
   20 kb middle); paired Wilcoxon across inversions and Mann-Whitney U of the
   flank-minus-middle difference between recurrent and single-event inversions.

Outputs (under data/):
    divergence_edge_decay_spearman.tsv
    divergence_middle_vs_flank.tsv
    divergence_middle_vs_flank_stats.tsv
    divergence_middle_vs_flank.pdf
"""
from __future__ import annotations

import gzip
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger("divergence_edge_decay")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INV_FILE = DATA_DIR / "inv_properties.tsv"

EPS_DENOM = 1e-12
# Window geometry matched to stats/middle_vs_flank_fst.py (40 kb total).
TOTAL_WINDOW = 40_000
FLANK_SIZE = TOTAL_WINDOW // 4      # 10 kb each flank
MIDDLE_SIZE = TOTAL_WINDOW // 2     # 20 kb middle
BIN_SIZE = 2_000                    # matched to fst_edge_decay.py
MAX_DECAY_SPAN = 100_000

COLOR_SINGLE = "#1f3b78"
COLOR_RECUR = "#8c2d7e"

# numerator => da, denominator => dxy
RE_NUM = re.compile(r">.*?hudson_pairwise_fst.*?_numerator_chr_?(?P<chrom>[\w.\-]+)_start_(?P<start>\d+)_end_(?P<end>\d+)", re.IGNORECASE)
RE_DEN = re.compile(r">.*?hudson_pairwise_fst.*?_denominator_chr_?(?P<chrom>[\w.\-]+)_start_(?P<start>\d+)_end_(?P<end>\d+)", re.IGNORECASE)


def normalize_chrom(chrom: str) -> str:
    c = str(chrom).strip().lower()
    if c.startswith("chr_"):
        c = c[4:]
    elif c.startswith("chr"):
        c = c[3:]
    return c


def load_recurrence_map() -> dict[tuple[str, int, int], tuple[int, str]]:
    inv = pd.read_csv(INV_FILE, sep="\t", low_memory=False)
    inv["Start"] = pd.to_numeric(inv["Start"], errors="coerce")
    inv["End"] = pd.to_numeric(inv["End"], errors="coerce")
    inv["rec"] = pd.to_numeric(inv["0_single_1_recur_consensus"], errors="coerce")
    inv = inv.dropna(subset=["Start", "End", "rec"])
    inv = inv[inv["rec"].isin([0, 1])]
    label = {0: "Single-event", 1: "Recurrent"}
    return {
        (normalize_chrom(c), int(s), int(e)): (int(r), label[int(r)])
        for c, s, e, r in zip(inv["Chromosome"], inv["Start"], inv["End"], inv["rec"])
    }


def parse_values(lines: list[str]) -> np.ndarray:
    seq = "".join(s.strip() for s in lines if s.strip())
    if not seq:
        return np.array([], dtype=np.float64)
    return np.fromstring(seq.replace("NA", "nan"), sep=",", dtype=np.float64)


def load_per_site_components() -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    """Return {(chrom,start,end) -> {'num': da_array, 'den': dxy_array}}."""
    candidates = [DATA_DIR / "per_site_fst_output.falsta", DATA_DIR / "per_site_fst_output.falsta.gz"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Missing per_site_fst_output.falsta(.gz)")

    store: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}

    def handle(header, lines):
        if not header:
            return
        m = RE_NUM.search(header)
        comp = "num"
        if not m:
            m = RE_DEN.search(header)
            comp = "den"
        if not m:
            return
        key = (normalize_chrom(m.group("chrom")), int(m.group("start")), int(m.group("end")))
        vals = parse_values(lines)
        if vals.size:
            store.setdefault(key, {})[comp] = vals

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        header = None
        lines: list[str] = []
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                handle(header, lines)
                header = line
                lines = []
            else:
                lines.append(line)
        handle(header, lines)

    # keep only loci with both da (num) and dxy (den) of equal length
    out = {}
    for key, comp in store.items():
        if "num" in comp and "den" in comp and comp["num"].size == comp["den"].size:
            out[key] = comp
    return out


def _window_mean(values: np.ndarray, valid: np.ndarray) -> float:
    """Mean of per-site value over informative sites (finite dxy denominator)."""
    m = valid & np.isfinite(values)
    if not m.any():
        return np.nan
    return float(np.mean(values[m]))


def middle_vs_flank(components, recmap) -> pd.DataFrame:
    rows = []
    for key, comp in components.items():
        if key not in recmap:
            continue
        da = comp["num"]
        dxy = comp["den"]
        L = da.size
        if L < TOTAL_WINDOW:
            continue
        # informative = finite, positive dxy denominator (matches ratio-of-sums logic)
        valid = np.isfinite(dxy) & (dxy > EPS_DENOM)

        start_mid = (L - MIDDLE_SIZE) // 2
        end_mid = start_mid + MIDDLE_SIZE
        if not (FLANK_SIZE <= start_mid and end_mid <= L - FLANK_SIZE):
            continue

        flank_idx = np.concatenate([np.arange(FLANK_SIZE), np.arange(L - FLANK_SIZE, L)])
        mid_idx = np.arange(start_mid, end_mid)

        flag, label = recmap[key]
        rows.append({
            "chrom": key[0], "start": key[1], "end": key[2],
            "recurrence_flag": flag, "recurrence_label": label,
            "da_flank": _window_mean(da[flank_idx], valid[flank_idx]),
            "da_middle": _window_mean(da[mid_idx], valid[mid_idx]),
            "dxy_flank": _window_mean(dxy[flank_idx], valid[flank_idx]),
            "dxy_middle": _window_mean(dxy[mid_idx], valid[mid_idx]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["da_flank_minus_middle"] = df["da_flank"] - df["da_middle"]
        df["dxy_flank_minus_middle"] = df["dxy_flank"] - df["dxy_middle"]
    return df


def folded_spearman(values: np.ndarray, valid: np.ndarray) -> tuple[float | None, float | None, int]:
    """Two-sided Spearman of per-site value vs distance from nearest edge (folded)."""
    length = values.size
    if length <= MAX_DECAY_SPAN:
        return None, None, 0
    max_len = min(MAX_DECAY_SPAN, length // 2)
    usable = (max_len // BIN_SIZE) * BIN_SIZE
    if usable < 2 * BIN_SIZE:
        return None, None, 0

    vals = values.copy()
    vals[~valid] = np.nan
    left = vals[:usable]
    right = vals[-usable:][::-1]
    try:
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                left_bins = np.nanmean(left.reshape(-1, BIN_SIZE), axis=1)
                right_bins = np.nanmean(right.reshape(-1, BIN_SIZE), axis=1)
                folded = np.nanmean(np.vstack([left_bins, right_bins]), axis=0)
    except ValueError:
        return None, None, 0
    centers = np.arange(folded.size, dtype=float) * BIN_SIZE + BIN_SIZE / 2
    mask = np.isfinite(folded)
    bins_used = int(mask.sum())
    if bins_used < 5:
        return None, None, bins_used
    rho, p = stats.spearmanr(centers[mask], folded[mask])
    rho = float(rho) if np.isfinite(rho) else None
    p = float(p) if np.isfinite(p) else None
    return rho, p, bins_used


def edge_decay_spearman(components, recmap) -> pd.DataFrame:
    rows = []
    for key, comp in components.items():
        if key not in recmap:
            continue
        da = comp["num"]
        dxy = comp["den"]
        flag, label = recmap[key]
        rec = {"chrom": key[0], "start": key[1], "end": key[2],
               "recurrence_flag": flag, "recurrence_label": label}
        for metric, arr in [("da", da), ("dxy", dxy)]:
            # Fold over all sites where the metric is defined (finite), matching the
            # FST edge-decay treatment of the summary track (no extra masking).
            rho, p, bins_used = folded_spearman(arr, np.isfinite(arr))
            rec[f"{metric}_rho"] = rho
            rec[f"{metric}_p"] = p
            rec[f"{metric}_bins_used"] = bins_used
        rows.append(rec)
    df = pd.DataFrame(rows)
    for metric in ["da", "dxy"]:
        col = f"{metric}_p"
        if col in df:
            mask = df[col].notna()
            df[f"{metric}_q"] = np.nan
            if mask.any():
                _, q, _, _ = multipletests(df.loc[mask, col].to_numpy(), method="fdr_bh")
                df.loc[mask, f"{metric}_q"] = q
    return df


def paired_and_group_tests(mf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["da", "dxy"]:
        fcol, mcol, dcol = f"{metric}_flank", f"{metric}_middle", f"{metric}_flank_minus_middle"
        sub = mf.dropna(subset=[fcol, mcol])
        # Overall paired Wilcoxon (flank vs middle)
        for group, label in [(sub, "Overall"),
                             (sub[sub.recurrence_label == "Single-event"], "Single-event"),
                             (sub[sub.recurrence_label == "Recurrent"], "Recurrent")]:
            n = len(group)
            w_stat, w_p = (np.nan, np.nan)
            diffs = group[dcol].to_numpy()
            if n >= 2 and np.any(diffs != 0):
                try:
                    w_stat, w_p = wilcoxon(group[fcol].to_numpy(), group[mcol].to_numpy())
                    w_stat, w_p = float(w_stat), float(w_p)
                except ValueError:
                    pass
            rows.append({
                "metric": metric, "group": label, "n": n,
                "median_flank": float(np.median(group[fcol])) if n else np.nan,
                "median_middle": float(np.median(group[mcol])) if n else np.nan,
                "median_flank_minus_middle": float(np.median(diffs)) if n else np.nan,
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            })
        # Recurrent vs single-event MWU on flank-minus-middle
        rec = sub.loc[sub.recurrence_label == "Recurrent", dcol].to_numpy()
        sing = sub.loc[sub.recurrence_label == "Single-event", dcol].to_numpy()
        u, p = (np.nan, np.nan)
        if rec.size and sing.size:
            u, p = mannwhitneyu(rec, sing, alternative="two-sided")
            u, p = float(u), float(p)
        rows.append({
            "metric": metric, "group": "Recurrent_vs_Single_diff", "n": rec.size + sing.size,
            "median_flank": np.nan, "median_middle": np.nan,
            "median_flank_minus_middle": np.nan,
            "wilcoxon_stat": u, "wilcoxon_p": p,
        })
    return pd.DataFrame(rows)


def make_plot(mf: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, metric, title in zip(axes, ["dxy", "da"],
                                 ["Absolute divergence d_xy", "Net divergence d_a"]):
        fcol, mcol = f"{metric}_flank", f"{metric}_middle"
        sub = mf.dropna(subset=[fcol, mcol])
        for cat, color in [("Single-event", COLOR_SINGLE), ("Recurrent", COLOR_RECUR)]:
            g = sub[sub.recurrence_label == cat]
            for _, r in g.iterrows():
                ax.plot([0, 1], [r[fcol], r[mcol]], color=color, alpha=0.35, lw=0.9, zorder=2)
            ax.scatter(np.zeros(len(g)), g[fcol], color=color, s=22, alpha=0.7, edgecolor="k", lw=0.3, zorder=4)
            ax.scatter(np.ones(len(g)), g[mcol], color=color, s=22, alpha=0.7, edgecolor="k", lw=0.3,
                       zorder=4, label=cat)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Flank\n(breakpoints)", "Middle"], fontsize=11)
        ax.set_ylabel(title, fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(-0.3, 1.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(fontsize=10, frameon=False)
    fig.suptitle("Breakpoint (flank) vs middle divergence per inversion", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    recmap = load_recurrence_map()
    components = load_per_site_components()
    logger.info("Loaded per-site da/dxy for %d loci; %d classified by recurrence",
                len(components), sum(1 for k in components if k in recmap))

    mf = middle_vs_flank(components, recmap)
    mf.to_csv(DATA_DIR / "divergence_middle_vs_flank.tsv", sep="\t", index=False, na_rep="NA")
    logger.info("Middle-vs-flank: %d inversions (rec=%d, single=%d)",
                len(mf), int((mf.recurrence_label == "Recurrent").sum()),
                int((mf.recurrence_label == "Single-event").sum()))

    tests = paired_and_group_tests(mf)
    tests.to_csv(DATA_DIR / "divergence_middle_vs_flank_stats.tsv", sep="\t", index=False)
    for _, r in tests.iterrows():
        logger.info("[%s | %s] n=%d med_flank=%.3e med_mid=%.3e diff=%.3e stat=%.3g p=%.4g",
                    r.metric, r.group, r.n, r.median_flank, r.median_middle,
                    r.median_flank_minus_middle, r.wilcoxon_stat, r.wilcoxon_p)

    decay = edge_decay_spearman(components, recmap)
    decay.to_csv(DATA_DIR / "divergence_edge_decay_spearman.tsv", sep="\t", index=False, na_rep="NA")
    for metric in ["da", "dxy"]:
        d = decay.dropna(subset=[f"{metric}_rho"])
        if len(d):
            logger.info("[edge-decay %s] %d inversions with valid Spearman; median rho=%.3f; "
                        "n with q<0.05 = %d",
                        metric, len(d), float(d[f"{metric}_rho"].median()),
                        int((d[f"{metric}_q"] < 0.05).sum()))

    make_plot(mf, DATA_DIR / "divergence_middle_vs_flank.pdf")


if __name__ == "__main__":
    main()
