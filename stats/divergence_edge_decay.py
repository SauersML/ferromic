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
new data are introduced.

PER-CALLABLE-BASE vs SNP-CONDITIONED (audit BUG #11)
----------------------------------------------------
The Rust per-site writer initialises every base in the region to NA and fills
ONLY variant positions (src/process.rs::append_fst_falsta iterates the Hudson
variant sites). Invariant callable bases are therefore indistinguishable from
uncallable bases in the FST falsta alone: both are NA. Averaging da/dxy over the
sites with a finite, positive dxy denominator (the previous behaviour) yields
divergence among VARIANT sites — a SNP-density-sensitive quantity — not
divergence per callable base. Reviewer 1 asked for divergence corrected for
diversity, i.e. per callable base, where invariant callable bases contribute 0.

We recover the callable mask from the per-base diversity tracks
(data/per_site_diversity_output.falsta(.gz)): a base is callable in group g iff
its filtered_pi value is finite (NA = uncallable). For the between-group da/dxy a
base is callable iff BOTH groups are callable there. At a callable base:
  - if the FST track has a finite value (a variant) we use it;
  - if the FST track is NA but the base is callable, it is an invariant callable
    base and contributes da = dxy = 0.
The PER-BASE window statistic is then sum(component over callable bases) /
(number of callable bases). This is the primary metric reported here.

For transparency we also report the previous SNP-conditioned statistic (mean over
variant sites with positive dxy), explicitly suffixed `_snpcond`, so the two can
be compared. When the diversity track is unavailable for a locus we cannot build
the callable mask and fall back to the SNP-conditioned statistic for that locus
(flagged in the `callable_basis` column).

Two complementary tests, mirroring the FST analyses:

1. Folded edge decay (cf. fst_edge_decay.py): two-sided Spearman correlation of
   per-base value vs distance from the nearest breakpoint, folded across the two
   flanks, per inversion; BH-FDR across inversions; reported by recurrence class.
   Per-base means within each 2 kb bin set non-variant callable bases to 0.
2. Middle vs flank (cf. middle_vs_flank_fst.py): per inversion, per-base mean da
   and dxy in the flank windows vs the middle window (40 kb total: 10 kb each
   flank, 20 kb middle); paired Wilcoxon across inversions and Mann-Whitney U of
   the flank-minus-middle difference between recurrent and single-event
   inversions.

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


def _resolve_input(name: str) -> Path:
    """Prefer a fresh copy in the CWD (CI working dir), else fall back to data/."""
    for base in (Path.cwd(), DATA_DIR):
        p = base / name
        if p.exists():
            return p
    return DATA_DIR / name


INV_FILE = _resolve_input("inv_properties.tsv")

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
    candidates = [base / fn
                  for base in (Path.cwd(), DATA_DIR)
                  for fn in ("per_site_fst_output.falsta", "per_site_fst_output.falsta.gz")]
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


# filtered_pi_chr_<chrom>_start_<s>_end_<e>_group_<0|1>
RE_FILTERED_PI = re.compile(
    r">filtered_pi_chr_?(?P<chrom>[\w.\-]+)_start_(?P<start>\d+)_end_(?P<end>\d+)_group_(?P<grp>[01])",
    re.IGNORECASE,
)


def load_callable_masks() -> dict[tuple[str, int, int], np.ndarray]:
    """Return {(chrom,start,end) -> per-base callable mask} for da/dxy.

    A base is callable for the between-group divergence iff BOTH orientation
    groups are callable there. The per-base filtered diversity track is NA at
    uncallable bases and finite (0 or >0) at callable bases, so callable = both
    groups finite. Returns {} if the diversity falsta is absent (callers then
    fall back to the SNP-conditioned statistic)."""
    candidates = [base / fn
                  for base in (Path.cwd(), DATA_DIR)
                  for fn in ("per_site_diversity_output.falsta",
                             "per_site_diversity_output.falsta.gz")]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        logger.warning("per_site_diversity_output.falsta(.gz) not found; da/dxy "
                       "will use the SNP-conditioned statistic (no callable mask).")
        return {}

    groups: dict[tuple[str, int, int], dict[int, np.ndarray]] = {}

    def handle(header, lines):
        if not header:
            return
        m = RE_FILTERED_PI.search(header)
        if not m:
            return
        key = (normalize_chrom(m.group("chrom")), int(m.group("start")), int(m.group("end")))
        groups.setdefault(key, {})[int(m.group("grp"))] = parse_values(lines)

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

    masks: dict[tuple[str, int, int], np.ndarray] = {}
    for key, comp in groups.items():
        g0 = comp.get(0)
        g1 = comp.get(1)
        if g0 is None or g1 is None or g0.size != g1.size or g0.size == 0:
            continue
        masks[key] = np.isfinite(g0) & np.isfinite(g1)
    return masks


def _window_mean_snpcond(values: np.ndarray, valid: np.ndarray) -> float:
    """SNP-CONDITIONED mean: average of the per-site component over informative
    (variant) sites only — the sites with a finite, positive dxy denominator.
    This is SNP-density-sensitive (audit BUG #11) and is retained for comparison,
    not as the primary metric."""
    m = valid & np.isfinite(values)
    if not m.any():
        return np.nan
    return float(np.mean(values[m]))


def _window_mean_per_base(values: np.ndarray, callable_mask: np.ndarray) -> float:
    """PER-CALLABLE-BASE mean (the metric Reviewer 1 requested; audit BUG #11).

    Divergence is summed over every callable base and divided by the number of
    callable bases. Callable bases where the FST track is NA are invariant
    callable bases and contribute 0 (no between-group difference); uncallable
    bases (mask False) are excluded from both numerator and denominator."""
    if callable_mask is None or not callable_mask.any():
        return np.nan
    vals = np.where(np.isfinite(values), values, 0.0)
    n_callable = int(np.count_nonzero(callable_mask))
    return float(np.sum(vals[callable_mask]) / n_callable)


def middle_vs_flank(components, recmap, callable_masks=None) -> pd.DataFrame:
    rows = []
    for key, comp in components.items():
        if key not in recmap:
            continue
        da = comp["num"]
        dxy = comp["den"]
        L = da.size
        if L < TOTAL_WINDOW:
            continue
        # SNP-conditioned informative sites: finite, positive dxy denominator.
        valid = np.isfinite(dxy) & (dxy > EPS_DENOM)

        # Per-base callable mask (audit BUG #11). When absent for this locus we
        # fall back to the SNP-conditioned statistic and flag the basis.
        cmask = callable_masks.get(key) if callable_masks else None
        if cmask is not None and cmask.size == L:
            basis = "per_base"
        else:
            cmask = None
            basis = "snp_conditioned_fallback"

        start_mid = (L - MIDDLE_SIZE) // 2
        end_mid = start_mid + MIDDLE_SIZE
        if not (FLANK_SIZE <= start_mid and end_mid <= L - FLANK_SIZE):
            continue

        flank_idx = np.concatenate([np.arange(FLANK_SIZE), np.arange(L - FLANK_SIZE, L)])
        mid_idx = np.arange(start_mid, end_mid)

        def per_base(arr, idx):
            if cmask is None:
                return _window_mean_snpcond(arr[idx], valid[idx])
            return _window_mean_per_base(arr[idx], cmask[idx])

        flag, label = recmap[key]
        rows.append({
            "chrom": key[0], "start": key[1], "end": key[2],
            "recurrence_flag": flag, "recurrence_label": label,
            "callable_basis": basis,
            # Primary metric: divergence per callable base (invariant callable
            # bases contribute 0).
            "da_flank": per_base(da, flank_idx),
            "da_middle": per_base(da, mid_idx),
            "dxy_flank": per_base(dxy, flank_idx),
            "dxy_middle": per_base(dxy, mid_idx),
            # SNP-conditioned comparison (mean over variant sites only).
            "da_flank_snpcond": _window_mean_snpcond(da[flank_idx], valid[flank_idx]),
            "da_middle_snpcond": _window_mean_snpcond(da[mid_idx], valid[mid_idx]),
            "dxy_flank_snpcond": _window_mean_snpcond(dxy[flank_idx], valid[flank_idx]),
            "dxy_middle_snpcond": _window_mean_snpcond(dxy[mid_idx], valid[mid_idx]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["da_flank_minus_middle"] = df["da_flank"] - df["da_middle"]
        df["dxy_flank_minus_middle"] = df["dxy_flank"] - df["dxy_middle"]
        df["da_flank_minus_middle_snpcond"] = df["da_flank_snpcond"] - df["da_middle_snpcond"]
        df["dxy_flank_minus_middle_snpcond"] = df["dxy_flank_snpcond"] - df["dxy_middle_snpcond"]
    return df


def folded_spearman(values: np.ndarray, valid: np.ndarray,
                    callable_mask: np.ndarray | None = None) -> tuple[float | None, float | None, int]:
    """Two-sided Spearman of per-base value vs distance from nearest edge (folded).

    When `callable_mask` is given the bin means are PER CALLABLE BASE (audit
    BUG #11): callable bases keep their value, callable-but-NA bases (invariant)
    are set to 0, and uncallable bases are excluded (set NaN so nanmean over the
    bin divides by the callable count). When it is None the previous
    SNP-conditioned behaviour is used: only `valid` (variant) sites contribute."""
    length = values.size
    if length <= MAX_DECAY_SPAN:
        return None, None, 0
    max_len = min(MAX_DECAY_SPAN, length // 2)
    usable = (max_len // BIN_SIZE) * BIN_SIZE
    if usable < 2 * BIN_SIZE:
        return None, None, 0

    vals = values.copy()
    if callable_mask is not None and callable_mask.size == length:
        # Per callable base: invariant callable bases -> 0, uncallable -> NaN.
        vals = np.where(callable_mask, np.where(np.isfinite(vals), vals, 0.0), np.nan)
    else:
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


def edge_decay_spearman(components, recmap, callable_masks=None) -> pd.DataFrame:
    rows = []
    for key, comp in components.items():
        if key not in recmap:
            continue
        da = comp["num"]
        dxy = comp["den"]
        cmask = callable_masks.get(key) if callable_masks else None
        if cmask is not None and cmask.size != da.size:
            cmask = None
        basis = "per_base" if cmask is not None else "snp_conditioned_fallback"
        flag, label = recmap[key]
        rec = {"chrom": key[0], "start": key[1], "end": key[2],
               "recurrence_flag": flag, "recurrence_label": label,
               "callable_basis": basis}
        for metric, arr in [("da", da), ("dxy", dxy)]:
            # Per callable base when the mask is available (invariant callable
            # bases contribute 0), else the SNP-conditioned fallback over finite
            # (variant) sites only (audit BUG #11).
            rho, p, bins_used = folded_spearman(arr, np.isfinite(arr), cmask)
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
    callable_masks = load_callable_masks()
    n_masked = sum(1 for k in components if k in callable_masks)
    logger.info("Loaded per-site da/dxy for %d loci; %d classified by recurrence; "
                "callable mask available for %d loci (per-base metric)",
                len(components), sum(1 for k in components if k in recmap), n_masked)

    mf = middle_vs_flank(components, recmap, callable_masks)
    mf.to_csv(DATA_DIR / "divergence_middle_vs_flank.tsv", sep="\t", index=False, na_rep="NA")
    n_perbase = int((mf.get("callable_basis") == "per_base").sum()) if not mf.empty else 0
    logger.info("Middle-vs-flank: %d inversions (rec=%d, single=%d); %d per-base, %d snp-cond fallback",
                len(mf), int((mf.recurrence_label == "Recurrent").sum()),
                int((mf.recurrence_label == "Single-event").sum()),
                n_perbase, len(mf) - n_perbase)

    tests = paired_and_group_tests(mf)
    tests.to_csv(DATA_DIR / "divergence_middle_vs_flank_stats.tsv", sep="\t", index=False)
    for _, r in tests.iterrows():
        logger.info("[%s | %s] n=%d med_flank=%.3e med_mid=%.3e diff=%.3e stat=%.3g p=%.4g",
                    r.metric, r.group, r.n, r.median_flank, r.median_middle,
                    r.median_flank_minus_middle, r.wilcoxon_stat, r.wilcoxon_p)

    decay = edge_decay_spearman(components, recmap, callable_masks)
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
