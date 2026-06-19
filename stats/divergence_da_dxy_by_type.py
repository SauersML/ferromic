"""Recurrent vs single-event comparison of absolute (dxy) and net (da) divergence.

Reviewer 1 noted that FST and pi are inversely related by construction, so a
difference in FST between recurrent and single-event inversions could reflect a
difference in within-group diversity rather than in between-group divergence.
Following Charlesworth (1998, MBE 15:538-543) and Cruickshank & Hahn (2014,
Mol Ecol 23:3133-3157), we therefore repeat the recurrent-vs-single comparison
the paper performs for FST (see stats/overall_fst_by_type.py) using

    dxy = absolute divergence between the direct and inverted haplotype groups
    da  = dxy - 0.5 * (pi_direct + pi_inverted)   (net divergence)

Both quantities come directly from the per-inversion Hudson columns already in
``data/output.csv``; no new data are introduced. da is identical to
FST_Hudson * dxy, but is reported on the divergence scale so that it is not
mechanically anti-correlated with within-group pi.

Outputs (under data/):
    divergence_da_dxy_by_type.tsv        per-locus table
    divergence_da_dxy_by_type_stats.tsv  test summary (medians, p, n)
    divergence_da_dxy_by_type.pdf        violin + strip plot
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger("divergence_da_dxy_by_type")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "output.csv"
INV_FILE = DATA_DIR / "inv_properties.tsv"

DXY_COL = "hudson_dxy_hap_group_0v1"
PI0_COL = "hudson_pi_hap_group_0"
PI1_COL = "hudson_pi_hap_group_1"

# Match the paper's category palette (Single-event / Recurrent).
COLOR_SINGLE = "#1f3b78"
COLOR_RECUR = "#8c2d7e"
CATEGORY_ORDER = ["Single-event", "Recurrent"]
CATEGORY_COLOR = {"Single-event": COLOR_SINGLE, "Recurrent": COLOR_RECUR}


def normalize_chrom(chrom) -> str:
    c = str(chrom).strip().lower()
    if c.startswith("chr_"):
        c = c[4:]
    elif c.startswith("chr"):
        c = c[3:]
    return c


def load_recurrence_map() -> dict[tuple[str, int, int], int]:
    """Return {(chrom, start, end) -> recurrence_flag} from inv_properties.tsv."""
    inv = pd.read_csv(INV_FILE, sep="\t", low_memory=False)
    inv["Start"] = pd.to_numeric(inv["Start"], errors="coerce")
    inv["End"] = pd.to_numeric(inv["End"], errors="coerce")
    inv["rec"] = pd.to_numeric(inv["0_single_1_recur_consensus"], errors="coerce")
    inv = inv.dropna(subset=["Start", "End", "rec"])
    inv = inv[inv["rec"].isin([0, 1])]
    return {
        (normalize_chrom(c), int(s), int(e)): int(r)
        for c, s, e, r in zip(inv["Chromosome"], inv["Start"], inv["End"], inv["rec"])
    }


def assign_recurrence(df: pd.DataFrame, recmap: dict[tuple[str, int, int], int]) -> pd.DataFrame:
    """Annotate each output.csv row with its recurrence flag via 1bp coordinate match."""
    df = df.copy()
    df["_chr"] = df["chr"].map(normalize_chrom)

    def match(row):
        c = row["_chr"]
        s = int(row["region_start"])
        e = int(row["region_end"])
        for (kc, ks, ke), flag in recmap.items():
            if kc == c and abs(ks - s) <= 1 and abs(ke - e) <= 1:
                return flag
        return np.nan

    df["recurrence_flag"] = df.apply(match, axis=1)
    df["category"] = df["recurrence_flag"].map({0: "Single-event", 1: "Recurrent"})
    return df


def mwu(rec_vals: np.ndarray, single_vals: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U; returns (U, p)."""
    if rec_vals.size == 0 or single_vals.size == 0:
        return np.nan, np.nan
    u, p = mannwhitneyu(rec_vals, single_vals, alternative="two-sided")
    return float(u), float(p)


def make_plot(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path, p_value: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 6.5))
    positions = np.arange(len(CATEGORY_ORDER))
    series = [df.loc[df["category"] == cat, metric].dropna().to_numpy() for cat in CATEGORY_ORDER]

    parts = ax.violinplot([s for s in series if s.size], positions=[p for p, s in zip(positions, series) if s.size],
                          widths=0.75, showmedians=True, showextrema=False)
    valid_cats = [cat for cat, s in zip(CATEGORY_ORDER, series) if s.size]
    for body, cat in zip(parts["bodies"], valid_cats):
        body.set_facecolor(CATEGORY_COLOR[cat])
        body.set_edgecolor("darkgrey")
        body.set_alpha(0.4)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

    rng = np.random.default_rng(0)
    for pos, s, cat in zip(positions, series, CATEGORY_ORDER):
        if not s.size:
            continue
        jitter = rng.normal(0, 0.04, size=s.size)
        ax.scatter(pos + jitter, s, color="dimgray", alpha=0.55, s=18, edgecolor="none", zorder=5)
        med = float(np.median(s))
        ax.text(pos + 0.14, med, f"{med:.2e}", fontsize=9, va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{cat}\n(n={s.size})" for cat, s in zip(CATEGORY_ORDER, series)], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    p_txt = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.set_title(f"{ylabel} by inversion type\nMann-Whitney U (two-sided): {p_txt}", fontsize=13)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    recmap = load_recurrence_map()
    logger.info("Classified inversions: %d (recurrent=%d, single=%d)",
                len(recmap), sum(recmap.values()), sum(1 for v in recmap.values() if v == 0))

    df = pd.read_csv(OUTPUT_CSV, usecols=lambda c: c in
                     {"chr", "region_start", "region_end", DXY_COL, PI0_COL, PI1_COL,
                      "hudson_fst_hap_group_0v1"})
    df = assign_recurrence(df, recmap)

    df["dxy"] = pd.to_numeric(df[DXY_COL], errors="coerce")
    df["da"] = df["dxy"] - 0.5 * (pd.to_numeric(df[PI0_COL], errors="coerce")
                                  + pd.to_numeric(df[PI1_COL], errors="coerce"))

    classified = df[df["category"].notna()].copy()

    out_cols = ["chr", "region_start", "region_end", "category",
                "hudson_fst_hap_group_0v1", DXY_COL, PI0_COL, PI1_COL, "da", "dxy"]
    table = classified[out_cols].sort_values(["category", "chr", "region_start"])
    table.to_csv(DATA_DIR / "divergence_da_dxy_by_type.tsv", sep="\t", index=False, na_rep="NA")
    logger.info("Wrote per-locus table (%d rows)", len(table))

    summary_rows = []
    for metric, ylabel, fname in [
        ("dxy", "Absolute divergence d$_{xy}$", "divergence_dxy_by_type.pdf"),
        ("da", "Net divergence d$_a$", "divergence_da_by_type.pdf"),
    ]:
        rec = classified.loc[classified["category"] == "Recurrent", metric].dropna().to_numpy()
        single = classified.loc[classified["category"] == "Single-event", metric].dropna().to_numpy()
        u, p = mwu(rec, single)
        make_plot(classified, metric, ylabel.replace("$_{xy}$", "_xy").replace("$_a$", "_a"),
                  DATA_DIR / fname, p)
        row = {
            "metric": metric,
            "n_recurrent": rec.size,
            "n_single_event": single.size,
            "median_recurrent": float(np.median(rec)) if rec.size else np.nan,
            "median_single_event": float(np.median(single)) if single.size else np.nan,
            "mean_recurrent": float(np.mean(rec)) if rec.size else np.nan,
            "mean_single_event": float(np.mean(single)) if single.size else np.nan,
            "mannwhitney_u": u,
            "p_two_sided": p,
        }
        summary_rows.append(row)
        logger.info("[%s] median rec=%.3e (n=%d) vs single=%.3e (n=%d); MWU U=%.1f p=%.4g",
                    metric, row["median_recurrent"], rec.size, row["median_single_event"],
                    single.size, u, p)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_DIR / "divergence_da_dxy_by_type_stats.tsv", sep="\t", index=False)
    logger.info("Wrote test summary")


if __name__ == "__main__":
    main()
