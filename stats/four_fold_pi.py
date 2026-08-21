"""Nucleotide diversity (pi) at 4-fold degenerate (synonymous) sites.

Reviewer 1 requested pi at fourfold-degenerate third-codon positions as the most
reliable, selection-neutral estimate of nucleotide diversity, to be compared
against the whole-locus pi reported in the manuscript.

This script:
  1. Reads the per-CDS combined PHYLIP alignments (group0 = direct haplotypes,
     group1 = inverted haplotypes) produced by the CDS pipeline. Each alignment
     is in frame (length divisible by 3, starting at the start codon), per
     cds/axt_to_phy.py / cds/combine_phy.py conventions.
  2. Walks codons in frame and identifies the eight fourfold-degenerate codon
     families (Leu CTN, Val GTN, Ser TCN, Pro CCN, Thr ACN, Ala GCN, Arg CGN,
     Gly GGN). The third position of such codons is a fourfold site.
  3. Computes per-site pi using the SAME estimator as the Rust pipeline
     (src/stats.rs::dense_pi_from_counts):
         per-site pi = n/(n-1) * (1 - sum_i p_i^2)
     where n is the number of called (A/C/G/T) haplotypes at the site and p_i
     the per-allele frequencies. Locus pi is the mean of per-site pi over
     callable sites (>= 2 called haplotypes), matching
     calculate_pi_from_summary (sum of per-site pi / effective length).
     This is done separately for the inverted and direct haplotype groups.
  4. Restricts the analysis to loci with a consensus recurrence classification
     in data/inv_properties.tsv (0_single_1_recur_consensus), then compares
     fourfold pi to whole-locus pi (output.csv 0_pi_filtered / 1_pi_filtered) by
     orientation and recurrence using the paper's paired Wilcoxon signed-rank
     and Mann-Whitney U tests (cf. stats/recur_diversity.py,
     stats/inv_dir_recur_model.py).

Outputs (written to the working directory, i.e. data/ when run from there):
  - four_fold_pi_by_inversion.tsv : per-inversion fourfold and whole-locus pi
  - four_fold_pi_tests.tsv        : test statistics by orientation/recurrence
  - four_fold_pi.pdf              : supplementary figure

Run from the data/ directory:
    cd data && python ../stats/four_fold_pi.py
"""

import os
import re
import gzip
import glob
import math
import shutil
import zipfile
import tempfile
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

try:  # plotting is optional so the pass can run on cluster nodes without mpl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Shared figure style so panels across the paper read as one system.
try:
    from stats._figstyle import apply as _apply_figstyle
    _apply_figstyle()
except Exception:  # style is cosmetic; never let it break a run
    pass

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Single source of truth for codon-aware diversity (shared with pin_pis.py).
from _codon_diversity import (
    VALID, FOURFOLD_PREFIXES, read_phy, site_pi, locus_pi,
    fourfold_codon_starts as fourfold_columns, class_aware_locus_pi,
    class_aware_site_pi_values,
)

warnings.filterwarnings("ignore")

# ------------------------- FILE PATHS -------------------------

# Directory holding the per-CDS group0_*/group1_* .phy.gz alignments. If unset,
# resolve_phy_dir() recovers them from data/phy_outputs.zip (or, when that file
# has been pruned from the tree, from its git-LFS object) into a temp dir.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _resolve_input(name):
    """Prefer a fresh copy in the CWD (CI working dir), else fall back to data/."""
    for base in (os.getcwd(), _DATA_DIR):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return name


PHY_DIR = os.environ.get("FOURFOLD_PHY_DIR")
PHY_OUTPUTS_ZIP_NAME = "phy_outputs.zip"          # zip member / basename
PHY_OUTPUTS_ZIP = _resolve_input(PHY_OUTPUTS_ZIP_NAME)  # resolved filesystem path
LFS_OID = "03f9b4d8167a0f2b3e715c6c978eddb9b03340a4334aa9ec50c07a3a8b7abf7d"
OUTPUT_CSV = _resolve_input("output.csv")
INVINFO_TSV = _resolve_input("inv_properties.tsv")

OUT_TABLE = os.path.join(_DATA_DIR, "four_fold_pi_by_inversion.tsv")
OUT_TESTS = os.path.join(_DATA_DIR, "four_fold_pi_tests.tsv")
OUT_FIG = os.path.join(_DATA_DIR, "four_fold_pi.pdf")

# VALID and FOURFOLD_PREFIXES are imported from _codon_diversity (single source of
# truth) so four_fold_pi and pin_pis cannot drift apart.

# ------------------------- PHYLIP I/O ------------------------

FNAME_RE = re.compile(
    r"^group(?P<grp>[01])_(?P<gene>.+?)_(?P<ensg>ENSG[0-9.]+)_(?P<enst>ENST[0-9.]+)_"
    r"(?P<chrom>chr[^_]+)_cds_start(?P<cs>\d+)_cds_end(?P<ce>\d+)_"
    r"inv_start(?P<is>\d+)_inv_end(?P<ie>\d+)\.phy\.gz$"
)


def _find_lfs_object():
    """Locate the local git-LFS object for phy_outputs.zip, if present."""
    candidates = [
        os.path.join(".git", "lfs", "objects", LFS_OID[:2], LFS_OID[2:4], LFS_OID),
        os.path.join("..", ".git", "lfs", "objects", LFS_OID[:2], LFS_OID[2:4], LFS_OID),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def resolve_phy_dir():
    """Return a directory containing group0_*/group1_* CDS .phy.gz alignments.

    Resolution order:
      1. FOURFOLD_PHY_DIR env var (a pre-extracted directory), if set.
      2. data/phy_outputs.zip in the working dir.
      3. The git-LFS object for phy_outputs.zip (it was pruned from the tree in
         commit 970b17ec to shrink clone size; the object may still be cached
         under .git/lfs/objects).
    Cases 2-3 extract the gene-level alignments into a temp dir, which the
    caller is responsible for cleaning up (returned as the second element)."""
    if PHY_DIR and os.path.isdir(PHY_DIR):
        return PHY_DIR, None

    src = PHY_OUTPUTS_ZIP if os.path.exists(PHY_OUTPUTS_ZIP) else _find_lfs_object()
    if not src:
        raise SystemExit(
            "Could not locate per-CDS alignments. Set FOURFOLD_PHY_DIR to a "
            "directory of group0_*/group1_* .phy.gz files, or make "
            "data/phy_outputs.zip available."
        )

    tmp = tempfile.mkdtemp(prefix="fourfold_phy_")
    print(f"Recovering CDS alignments from {src} -> {tmp}")
    # The archive is a zip whose single member is phy_outputs.zip (an inner zip
    # of the per-group .phy.gz files). Handle both the wrapped and direct cases.
    with zipfile.ZipFile(src) as outer:
        names = outer.namelist()
        if names == [PHY_OUTPUTS_ZIP_NAME] or PHY_OUTPUTS_ZIP_NAME in names:
            inner_path = os.path.join(tmp, PHY_OUTPUTS_ZIP_NAME)
            with outer.open(PHY_OUTPUTS_ZIP_NAME) as f_in, open(inner_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            archive = inner_path
        else:
            archive = src

    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if (base.startswith("group0_") or base.startswith("group1_")) and "ENST" in base and base.endswith(".phy.gz"):
                with z.open(name) as f_in, open(os.path.join(tmp, base), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
    return tmp, tmp


# read_phy, site_pi, locus_pi, fourfold_columns (= fourfold_codon_starts) are
# imported from _codon_diversity. fourfold_locus_pi is the codon-aware 4-fold third-
# position estimator, expressed via the shared class_aware_locus_pi so pin_pis uses
# the identical rule (a haplotype contributes its third base only when its own first
# two codon positions establish a 4-fold family; N/gap there excludes it).


def fourfold_locus_pi(seqs, codon_starts):
    """Mean per-site pi at fourfold third positions (codon-aware; see module note)."""
    return class_aware_locus_pi(seqs, [(cs, 2, "4") for cs in codon_starts])


def fourfold_site_pis(seqs, codon_starts):
    """Per-site pi values at fourfold third positions; their mean is
    fourfold_locus_pi and the per-inversion estimate is the pooled mean, so
    resampling these values resamples the per-inversion estimator exactly."""
    return class_aware_site_pi_values(seqs, [(cs, 2, "4") for cs in codon_starts])


def whole_cds_pi(seqs, L):
    return locus_pi(seqs, range(L))


# ------------------------- DATA LOADING ----------------------


def load_recurrence():
    """Return dict (chrom, start, end) -> recurrence consensus (0/1/NaN)."""
    inv = pd.read_csv(INVINFO_TSV, sep="\t")
    rec = {}
    for _, r in inv.iterrows():
        key = (str(r["Chromosome"]), int(r["Start"]), int(r["End"]))
        rec[key] = r["0_single_1_recur_consensus"]
    return rec


def collect_fourfold_pi(phy_dir):
    """Walk all group0/group1 CDS alignments; aggregate fourfold and whole-CDS
    pi to the inversion level (pi summed over CDS, then per-inversion mean of
    per-CDS pi weighted by callable sites)."""
    g0_files = sorted(glob.glob(os.path.join(phy_dir, "group0_*ENST*.phy.gz")))

    # Per-inversion accumulators: weighted sums of per-CDS pi over callable sites.
    acc = defaultdict(
        lambda: {
            "ff0_num": 0.0, "ff0_den": 0,
            "ff1_num": 0.0, "ff1_den": 0,
            "wc0_num": 0.0, "wc0_den": 0,
            "wc1_num": 0.0, "wc1_den": 0,
            "n_cds": 0, "n_cds_ff": 0,
            "ff0_sites": [], "ff1_sites": [],
        }
    )

    n_proc = 0
    for g0 in g0_files:
        base = os.path.basename(g0)
        m = FNAME_RE.match(base)
        if not m:
            continue
        g1 = g0.replace("group0_", "group1_")
        if not os.path.exists(g1):
            continue

        key = (m.group("chrom"), int(m.group("is")), int(m.group("ie")))

        s0, L0 = read_phy(g0)
        s1, L1 = read_phy(g1)
        if not s0 or not s1 or L0 != L1 or L0 % 3 != 0:
            continue
        L = L0
        n_proc += 1

        # Fourfold sites: a codon is a fourfold site only if BOTH haplotype
        # groups agree it is fourfold (prefix is fourfold-degenerate for every
        # called haplotype across both groups). This uses the combined sample to
        # define the site set, then measures pi within each group at those sites.
        combined = s0 + s1
        ff_cols = list(fourfold_columns(combined, L))

        a = acc[key]
        a["n_cds"] += 1

        # whole-CDS pi per group (mean per-site pi over callable sites)
        wc0, wc0n = whole_cds_pi(s0, L)
        wc1, wc1n = whole_cds_pi(s1, L)
        if wc0n:
            a["wc0_num"] += wc0 * wc0n
            a["wc0_den"] += wc0n
        if wc1n:
            a["wc1_num"] += wc1 * wc1n
            a["wc1_den"] += wc1n

        if ff_cols:
            v0 = fourfold_site_pis(s0, ff_cols)
            v1 = fourfold_site_pis(s1, ff_cols)
            ff0, ff0n = (float(np.mean(v0)), len(v0)) if v0 else (np.nan, 0)
            ff1, ff1n = (float(np.mean(v1)), len(v1)) if v1 else (np.nan, 0)
            if ff0n or ff1n:
                a["n_cds_ff"] += 1
            if ff0n:
                a["ff0_num"] += ff0 * ff0n
                a["ff0_den"] += ff0n
                a["ff0_sites"].extend(v0)
            if ff1n:
                a["ff1_num"] += ff1 * ff1n
                a["ff1_den"] += ff1n
                a["ff1_sites"].extend(v1)

    print(f"Processed {n_proc} CDS group pairs across {len(acc)} inversion loci.")

    rec = load_recurrence()
    rows = []
    for key, a in acc.items():
        chrom, istart, iend = key
        def ratio(num, den):
            return (num / den) if den > 0 else np.nan
        rows.append(
            {
                "chr": chrom,
                "region_start": istart,
                "region_end": iend,
                "recurrence": rec.get(key, np.nan),
                "n_cds": a["n_cds"],
                "n_cds_with_fourfold": a["n_cds_ff"],
                "fourfold_sites_direct": a["ff0_den"],
                "fourfold_sites_inverted": a["ff1_den"],
                "pi_fourfold_direct": ratio(a["ff0_num"], a["ff0_den"]),
                "pi_fourfold_inverted": ratio(a["ff1_num"], a["ff1_den"]),
                "pi_wholeCDS_direct": ratio(a["wc0_num"], a["wc0_den"]),
                "pi_wholeCDS_inverted": ratio(a["wc1_num"], a["wc1_den"]),
            }
        )
    site_map = {key: (np.asarray(a["ff0_sites"], float),
                      np.asarray(a["ff1_sites"], float))
                for key, a in acc.items()}
    return pd.DataFrame(rows), site_map


def attach_whole_locus_pi(df):
    """Add whole-locus pi (output.csv 0_pi_filtered/1_pi_filtered) by +-1 bp match."""
    out = pd.read_csv(OUTPUT_CSV)
    out["chr"] = out["chr"].astype(str)
    if not out["chr"].str.startswith("chr").all():
        out["chr"] = "chr" + out["chr"].astype(str).str.replace("chr", "", regex=False)

    pi_dir = []
    pi_inv = []
    for _, r in df.iterrows():
        cand = out[
            (out["chr"] == r["chr"])
            & ((out["region_start"] - r["region_start"]).abs() <= 1)
            & ((out["region_end"] - r["region_end"]).abs() <= 1)
        ]
        if len(cand):
            pi_dir.append(pd.to_numeric(cand["0_pi_filtered"], errors="coerce").iloc[0])
            pi_inv.append(pd.to_numeric(cand["1_pi_filtered"], errors="coerce").iloc[0])
        else:
            pi_dir.append(np.nan)
            pi_inv.append(np.nan)
    df["pi_wholeLocus_direct"] = pi_dir
    df["pi_wholeLocus_inverted"] = pi_inv
    return df


# ------------------------- STATISTICS ------------------------


def paired_wilcoxon(delta):
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0 or np.allclose(delta, 0.0):
        return np.nan, np.nan, len(delta)
    w, p = stats.wilcoxon(delta)
    return w, p, len(delta)


def run_tests(df):
    """Replicate the paper's paired/MWU tests on fourfold and whole-locus pi.

    Paired within-inversion: delta = log1p(pi_inverted) - log1p(pi_direct),
    Wilcoxon signed-rank, split by recurrence (single = 0, recurrent = 1).
    Between-group: Mann-Whitney U on the deltas (single vs recurrent).
    Same transform/tests as stats/recur_diversity.py."""
    results = []

    for metric, cdir, cinv in [
        ("fourfold", "pi_fourfold_direct", "pi_fourfold_inverted"),
        ("wholeCDS", "pi_wholeCDS_direct", "pi_wholeCDS_inverted"),
        ("wholeLocus", "pi_wholeLocus_direct", "pi_wholeLocus_inverted"),
    ]:
        sub = df.dropna(subset=[cdir, cinv, "recurrence"]).copy()
        sub["delta"] = np.log1p(sub[cinv]) - np.log1p(sub[cdir])

        single = sub[sub["recurrence"] == 0]
        recur = sub[sub["recurrence"] == 1]

        # Paired inverted-vs-direct within each recurrence category
        for label, grp in [("single", single), ("recurrent", recur)]:
            w, p, n = paired_wilcoxon(grp["delta"].values)
            results.append(
                {
                    "metric": metric,
                    "test": "paired Wilcoxon (inverted vs direct, log1p)",
                    "category": label,
                    "n": n,
                    "median_direct": grp[cdir].median(),
                    "median_inverted": grp[cinv].median(),
                    "statistic": w,
                    "p_value": p,
                }
            )

        # Between-group MWU on deltas (single vs recurrent)
        d_single = single["delta"].replace([np.inf, -np.inf], np.nan).dropna().values
        d_recur = recur["delta"].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(d_single) and len(d_recur):
            u, p = stats.mannwhitneyu(d_single, d_recur, alternative="two-sided")
        else:
            u, p = np.nan, np.nan
        results.append(
            {
                "metric": metric,
                "test": "Mann-Whitney U (delta: single vs recurrent)",
                "category": "single_vs_recurrent",
                "n": len(d_single) + len(d_recur),
                "median_direct": np.nan,
                "median_inverted": np.nan,
                "statistic": u,
                "p_value": p,
            }
        )

        # MWU recurrent vs single within each orientation (as in recur_diversity.py)
        for orient, col in [("direct", cdir), ("inverted", cinv)]:
            a = single[col].replace([np.inf, -np.inf], np.nan).dropna().values
            b = recur[col].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(a) and len(b):
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            else:
                u, p = np.nan, np.nan
            results.append(
                {
                    "metric": metric,
                    "test": f"Mann-Whitney U ({orient}: single vs recurrent)",
                    "category": "single_vs_recurrent",
                    "n": len(a) + len(b),
                    "median_direct": np.median(a) if orient == "direct" and len(a) else np.nan,
                    "median_inverted": np.median(b) if orient == "inverted" and len(b) else np.nan,
                    "statistic": u,
                    "p_value": p,
                }
            )

    return pd.DataFrame(results)


# ------------------------- FIGURE ----------------------------


def _sci(value, sig=2):
    """Mathtext 1.3x10^-4; mathtext glyphs render regardless of the text font."""
    if not (value == value) or value is None:
        return "NA"
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / (10 ** exp)
    return f"${mant:.1f}\\times10^{{{exp}}}$"


def _p_label(value):
    if not (value == value):  # NaN
        return "P = NA"
    return f"P = {_sci(value)}" if value < 0.001 else f"P = {value:.3f}"


def make_figure(df):
    """Four-panel figure for consensus-classified loci only.

    The top row compares direct and inverted diversity within each recurrence
    class. The bottom row compares per-locus orientation differences among the
    same loci. Unclassified loci are excluded from every panel and statistic.
    """
    if plt is None:
        print("matplotlib unavailable; skipping figure")
        return
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = {"Direct": "#1f3b78", "Inverted": "#8c2d7e"}
    positions = {
        ("Single-event", "Direct"): 0.0,
        ("Single-event", "Inverted"): 1.0,
        ("Recurrent", "Direct"): 3.0,
        ("Recurrent", "Inverted"): 4.0,
    }
    groups = list(positions)
    panels = [
        ("A", "Whole inversion locus", "pi_wholeLocus_direct", "pi_wholeLocus_inverted"),
        ("B", "4-fold-degenerate sites", "pi_fourfold_direct", "pi_fourfold_inverted"),
    ]

    # One shared color scale for the paired log2(pi_direct/pi_inverted) lines.
    ratios = []
    for _, _, cdir, cinv in panels:
        r = np.log2((df[cdir].to_numpy(float) + 1e-12)
                    / (df[cinv].to_numpy(float) + 1e-12))
        ratios.append(r[np.isfinite(r)])
    max_abs = max(float(np.percentile(np.abs(np.concatenate(ratios)), 98)), 1e-12)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 12.0))
    for ax, (letter, title, cdir, cinv) in zip(axes[0], panels):
        rng = np.random.default_rng(2025)
        sub = df.dropna(subset=[cdir, cinv, "recurrence"]).copy()
        sub = sub[np.isin(sub["recurrence"], [0, 1])]
        rec_name = {0: "Single-event", 1: "Recurrent"}

        values = [
            sub.loc[sub["recurrence"] == code, col].dropna().to_numpy(float)
            for (grp, ori), code, col in [
                (g, 0 if g[0] == "Single-event" else 1,
                 cdir if g[1] == "Direct" else cinv) for g in groups]
        ]

        violin = ax.violinplot(
            values,
            positions=[positions[g] for g in groups],
            widths=0.9,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, (_, orientation) in zip(violin["bodies"], groups):
            body.set_facecolor(colors[orientation])
            body.set_edgecolor("none")
            body.set_alpha(0.55)

        for vals_box, g in zip(values, groups):
            if not len(vals_box):
                continue
            ax.boxplot(
                [vals_box],
                positions=[positions[g]],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": "white", "edgecolor": "#111111"},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "#111111"},
                capprops={"color": "#111111"},
            )

        for _, row in sub.iterrows():
            grp = rec_name[int(row["recurrence"])]
            d, v = float(row[cdir]), float(row[cinv])
            if not (np.isfinite(d) and np.isfinite(v)):
                continue
            fold = np.log2((d + 1e-12) / (v + 1e-12))
            j = float(rng.uniform(0.06, 0.20))
            x_d = positions[(grp, "Direct")] + j
            x_v = positions[(grp, "Inverted")] - j
            ax.plot([x_d, x_v], [d, v], color=cmap(norm(fold)),
                    linewidth=1.25, alpha=0.85, zorder=2)
            ax.scatter([x_d], [d], c=[colors["Direct"]], s=25,
                       edgecolors="black", linewidths=0.4, alpha=0.72, zorder=3)
            ax.scatter([x_v], [v], c=[colors["Inverted"]], s=25,
                       edgecolors="black", linewidths=0.4, alpha=0.72, zorder=3)

        # Same paired test as run_tests(), so brackets and table cannot drift.
        p_by_code = {}
        for code in (0, 1):
            g = sub[sub["recurrence"] == code]
            delta = np.log1p(g[cinv].to_numpy(float)) - np.log1p(g[cdir].to_numpy(float))
            _, p, _ = paired_wilcoxon(delta)
            p_by_code[code] = p

        ymax = max(float(np.nanmax(np.concatenate([v for v in values if len(v)]))), 1e-8)
        ax.set_ylim(0, ymax * 1.23)
        for left, right, p_value in ((0, 1, p_by_code[0]), (3, 4, p_by_code[1])):
            y = ymax * 1.08
            h = ymax * 0.025
            ax.plot([left, left, right, right], [y, y + h, y + h, y], color="#222222")
            ax.text((left + right) / 2, y + h * 1.25, _p_label(p_value), ha="center")

        ax.axvline(2, color="#dddddd", linewidth=1)
        ax.set_xticks([0, 1, 3, 4])
        ax.set_xticklabels(["Direct", "Inverted", "Direct", "Inverted"])
        n_single = int((sub["recurrence"] == 0).sum())
        n_recur = int((sub["recurrence"] == 1).sum())
        ax.text(0.5, -0.11, f"Single-event\n(n = {n_single})",
                transform=ax.get_xaxis_transform(), ha="center", fontweight="bold")
        ax.text(3.5, -0.11, f"Recurrent\n(n = {n_recur})",
                transform=ax.get_xaxis_transform(), ha="center", fontweight="bold")
        ax.set_ylabel(r"Nucleotide diversity ($\pi$, $\times10^{-3}$)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
        ax.set_title(title, fontsize=12)
        ax.text(-0.08, 1.04, letter, transform=ax.transAxes,
                fontsize=15, fontweight="bold")

    # ---- bottom row: the correlations the response letter quotes ----------
    # Same "usable" rule as four_fold_pi_correlations.py: a locus carries 4-fold
    # information only when both orientations actually have 4-fold sites.
    from scipy import stats as _st
    usable = df[(df["fourfold_sites_direct"].fillna(0) > 0)
                & (df["fourfold_sites_inverted"].fillna(0) > 0)].copy()
    for col in ("wholeLocus", "fourfold", "wholeCDS"):
        usable[f"d_{col}"] = (usable[f"pi_{col}_inverted"].astype(float)
                              - usable[f"pi_{col}_direct"].astype(float))

    scatter_panels = [
        ("C", "d_wholeLocus", "d_fourfold",
         "\u0394\u03c0 (inverted \u2212 direct), whole inversion locus",
         "\u0394\u03c0 (inverted \u2212 direct), 4-fold sites"),
        ("D", "d_wholeCDS", "d_fourfold",
         "\u0394\u03c0 (inverted \u2212 direct), whole CDS",
         "\u0394\u03c0 (inverted \u2212 direct), 4-fold sites"),
    ]
    for ax, (letter, xcol, ycol, xlabel, ylabel) in zip(axes[1], scatter_panels):
        sub = usable.dropna(subset=[xcol, ycol])
        x = sub[xcol].to_numpy(float)
        y = sub[ycol].to_numpy(float)
        rec = pd.to_numeric(sub["recurrence"], errors="raise").to_numpy()
        if not np.isin(rec, [0, 1]).all():
            raise ValueError("four-fold figure requires consensus-classified loci")

        # Color encodes orientation in this figure, so recurrence class is
        # carried by marker shape instead.
        is_single = rec == 0
        is_recur = rec == 1
        # Class colors match the other recurrence figures
        # (stats/divergence_da_dxy_by_type.py); marker shape is a redundant cue.
        ax.scatter(x[is_single], y[is_single], s=42, marker="o",
                   color="#1f3b78", edgecolors="black", linewidths=0.4,
                   label=f"single-event (n = {int(is_single.sum())})")
        ax.scatter(x[is_recur], y[is_recur], s=52, marker="^",
                   color="#8c2d7e", edgecolors="black", linewidths=0.4,
                   label=f"recurrent (n = {int(is_recur.sum())})")

        rho, p = _st.spearmanr(x, y)
        ax.set_title(
            f"\u03c1 = {rho:.3f}, {_p_label(p)} (n = {len(x)})",
            fontsize=11)

        xlim = np.nanmax(np.abs(x)) * 1.15
        ylim = np.nanmax(np.abs(y)) * 1.15
        lim = max(xlim, ylim)
        ax.plot([-lim, lim], [-lim, lim], color="#bbbbbb", ls="--", lw=1.0,
                zorder=1)
        ax.axhline(0, color="#e3e3e3", lw=0.8, zorder=0)
        ax.axvline(0, color="#e3e3e3", lw=0.8, zorder=0)
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel(xlabel + r" ($\times10^{-3}$)")
        ax.set_ylabel(ylabel + r" ($\times10^{-3}$)")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.text(-0.08, 1.10, letter, transform=ax.transAxes,
                fontsize=15, fontweight="bold")

    # Adjust the grid BEFORE the colorbar: fig.colorbar() carves its space out of
    # the current axes positions, and a later subplots_adjust would undo that and
    # slide the right panel underneath the bar.
    fig.subplots_adjust(bottom=0.07, hspace=0.42, wspace=0.28)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=list(axes[0]), pad=0.02, fraction=0.04,
                            shrink=0.85)
    colorbar.set_label(r"$\log_{2}(\pi_{\mathrm{direct}}/\pi_{\mathrm{inverted}})$")
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(os.path.splitext(OUT_FIG)[0] + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {OUT_FIG}")


OUT_SPLIT = os.path.join(_DATA_DIR, "four_fold_split_half.tsv")


def split_half_analysis(df, site_map, n_rep=2000, seed=2026):
    """How well does the 4-fold estimate agree with itself?

    The per-inversion 4-fold pi is the mean of per-site pi over the pooled
    callable 4-fold sites, so randomly splitting those sites in two and
    recomputing the orientation difference from each half gives two independent
    same-length measurements of the same quantity. Their Spearman correlation
    across loci (r_hh) is the measure's self-agreement; no noise model, no
    demographic assumption. Spearman-Brown then gives the reliability of the
    full-length measure, r_full = 2 r_hh / (1 + r_hh), and the correlation
    attainable between the full 4-fold measure and ANY perfectly-agreeing
    noiseless quantity is bounded by sqrt(r_full). Comparing the observed
    whole-locus correlation to that bound asks whether the 4-fold data agree
    with the whole-locus data as well as they agree with themselves.
    """
    from scipy import stats as st

    rng = np.random.default_rng(seed)
    d = df.copy()
    d["d_whole"] = d["pi_wholeLocus_inverted"] - d["pi_wholeLocus_direct"]
    keys, vecs = [], []
    for _, r in d.iterrows():
        key = (r["chr"], int(r["region_start"]), int(r["region_end"]))
        v0, v1 = site_map.get(key, (np.array([]), np.array([])))
        # need >= 2 sites per orientation to split, and a whole-locus delta
        if len(v0) >= 2 and len(v1) >= 2 and np.isfinite(r["d_whole"]):
            keys.append((key, float(r["d_whole"]),
                         pd.to_numeric(r["recurrence"], errors="coerce")))
            vecs.append((v0, v1))
    n = len(keys)
    classified = np.array([k[2] in (0.0, 1.0) for k in keys])
    if not classified.all():
        raise ValueError("split-half analysis requires consensus-classified loci")
    d_whole = np.array([k[1] for k in keys])
    print(f"\nsplit-half: {n} consensus-classified loci with >=2 callable "
          "4-fold sites in both orientations")

    def one_rep():
        dA = np.empty(n)
        dB = np.empty(n)
        for i, (v0, v1) in enumerate(vecs):
            halves = []
            for v in (v0, v1):
                idx = rng.permutation(len(v))
                h = len(v) // 2
                halves.append((v[idx[:h]].mean(), v[idx[h:]].mean()))
            dA[i] = halves[1][0] - halves[0][0]
            dB[i] = halves[1][1] - halves[0][1]
        return dA, dB

    subsets = [("recurrence_classified", classified)]
    stats_acc = {name: {"r_hh": [], "ceiling": [], "r_half_whole": []}
                 for name, _ in subsets}
    for _ in range(n_rep):
        dA, dB = one_rep()
        for name, mask in subsets:
            r_hh = st.spearmanr(dA[mask], dB[mask]).statistic
            r_full = 2 * r_hh / (1 + r_hh) if r_hh > -1 else np.nan
            stats_acc[name]["r_hh"].append(r_hh)
            stats_acc[name]["ceiling"].append(
                np.sqrt(r_full) if (r_full == r_full and r_full > 0) else 0.0)
            stats_acc[name]["r_half_whole"].append(
                st.spearmanr(dA[mask], d_whole[mask]).statistic)

    rows = []
    label = {
        "r_hh": "split-half self-correlation of the 4-fold orientation difference",
        "ceiling": "attainable correlation with a perfectly-agreeing measure "
                   "(Spearman-Brown sqrt of full-length reliability)",
        "r_half_whole": "half-length 4-fold orientation difference vs whole-locus",
    }
    for name, mask in subsets:
        for stat, valslist in stats_acc[name].items():
            vals = np.asarray(valslist, float)
            vals = vals[np.isfinite(vals)]
            lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
            rows.append({
                "subset": name, "n_loci": int(mask.sum()), "statistic": stat,
                "median": f"{med:.6f}", "ci_lo": f"{lo:.6f}", "ci_hi": f"{hi:.6f}",
                "description": label[stat], "n_replicates": n_rep,
            })
            print(f"  {name:22s} {stat:14s} median={med:.3f} [{lo:.3f}, {hi:.3f}]")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SPLIT, sep="\t", index=False)
    print(f"Saved split-half analysis -> {OUT_SPLIT}")
    return out


# ------------------------- MAIN ------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--from-table", action="store_true",
                    help="skip the alignment pass and recompute the downstream tests "
                         "and figure from the committed per-inversion table. The tests "
                         "are a deterministic function of that table, so this "
                         "regenerates them exactly without needing phy_outputs.zip.")
    args = ap.parse_args()

    if args.from_table:
        if not os.path.exists(OUT_TABLE):
            raise SystemExit(f"{OUT_TABLE} not found; run without --from-table first")
        df = pd.read_csv(OUT_TABLE, sep="\t")
        print(f"Loaded per-inversion table <- {OUT_TABLE} ({len(df)} loci)")
    else:
        phy_dir, tmp_dir = resolve_phy_dir()
        try:
            df, site_map = collect_fourfold_pi(phy_dir)
        finally:
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        df = attach_whole_locus_pi(df)
        df = df.sort_values(["chr", "region_start"]).reset_index(drop=True)
        df.to_csv(OUT_TABLE, sep="\t", index=False)
        print(f"Saved per-inversion table -> {OUT_TABLE} ({len(df)} loci)")
    recurrence = pd.to_numeric(df["recurrence"], errors="coerce")
    analysis_df = df[recurrence.isin([0, 1])].copy()
    analysis_df["recurrence"] = pd.to_numeric(
        analysis_df["recurrence"], errors="raise"
    ).astype(int)
    print(f"Restricted 4-fold analysis to {len(analysis_df)} loci with a "
          "consensus recurrence classification")
    if not args.from_table:
        split_half_analysis(analysis_df, site_map)

    tests = run_tests(analysis_df)
    tests.to_csv(OUT_TESTS, sep="\t", index=False)
    print(f"Saved tests -> {OUT_TESTS}")

    make_figure(analysis_df)

    # Console summary by recurrence category
    print("\n=== Median pi by orientation x recurrence ===")
    for code, name in [(0, "Single-event"), (1, "Recurrent")]:
        sub = analysis_df[analysis_df["recurrence"] == code]
        print(f"\n{name} (n loci = {len(sub)}):")
        for label, col in [
            ("4-fold  direct  ", "pi_fourfold_direct"),
            ("4-fold  inverted", "pi_fourfold_inverted"),
            ("whole   direct  ", "pi_wholeLocus_direct"),
            ("whole   inverted", "pi_wholeLocus_inverted"),
        ]:
            v = sub[col].dropna()
            med = v.median() if len(v) else float("nan")
            print(f"  {label}: n={len(v):2d} median={med:.6e}")

    print("\n=== Key tests ===")
    for _, r in tests.iterrows():
        if r["test"].startswith("paired"):
            print(
                f"  [{r['metric']:>10}] {r['category']:>10}: "
                f"med_dir={r['median_direct']:.3e} med_inv={r['median_inverted']:.3e} "
                f"n={int(r['n']) if not math.isnan(r['n']) else 0} p={r['p_value']}"
            )


if __name__ == "__main__":
    main()
