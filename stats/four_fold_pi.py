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
  4. Aggregates by recurrence category from data/inv_properties.tsv
     (0_single_1_recur_consensus) and compares fourfold pi to whole-locus pi
     (output.csv 0_pi_filtered / 1_pi_filtered) by orientation and recurrence,
     using the paper's paired Wilcoxon signed-rank and Mann-Whitney U tests
     (cf. stats/recur_diversity.py, stats/inv_dir_recur_model.py).

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _inv_common import is_flipped  # chimp polarization: group0/1 -> ancestral/derived

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

VALID = set("ACGT")

# Eight fourfold-degenerate codon families: the first two positions fully
# determine the amino acid regardless of the third position. The third position
# is therefore a fourfold-degenerate (synonymous) site.
FOURFOLD_PREFIXES = {"CT", "GT", "TC", "CC", "AC", "GC", "CG", "GG"}

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


def read_phy(path):
    """Return list of (uppercased) sequences from a PHYLIP .phy.gz alignment.

    Mirrors the sequence extraction in cds/axt_to_phy.py: a header line of two
    integers, then one sequence per line with the sequence as the final
    whitespace-delimited token (names may contain '_L'/'_R')."""
    seqs = []
    with gzip.open(path, "rt") as fh:
        first = fh.readline().split()
        if len(first) != 2:
            return [], 0
        try:
            exp_len = int(first[1])
        except ValueError:
            return [], 0
        for line in fh:
            if not line.strip():
                continue
            m = re.search(r"([ACGTNacgtn-]+)\s*$", line)
            if m:
                seqs.append(m.group(1).upper())
    if not seqs:
        return [], exp_len
    L = len(seqs[0])
    if any(len(s) != L for s in seqs) or L != exp_len:
        return [], exp_len
    return seqs, L


# ------------------------- PI ESTIMATOR ----------------------


def site_pi(column):
    """Per-site pi with the Rust pipeline estimator (src/stats.rs).

    column: iterable of single-character bases for one alignment column.
    Returns None for uncallable sites (< 2 called A/C/G/T haplotypes)."""
    counts = Counter(b for b in column if b in VALID)
    n = sum(counts.values())
    if n < 2:
        return None
    sum_sq = sum(c * c for c in counts.values())
    return n / (n - 1.0) * (1.0 - sum_sq / (n * n))


def fourfold_columns(seqs, L):
    """Yield the codon-start index of each fourfold-degenerate codon.

    A codon is included only when every *called* haplotype (ACGT at positions 1-2) carries a
    fourfold-family prefix; if any called haplotype has a non-fourfold prefix the codon is
    skipped. Haplotypes with a gap/N at positions 1-2 are ignored *here* (their family is
    unknown) -- and, crucially, they are also excluded from this codon's pi by
    ``fourfold_locus_pi``, which re-checks each haplotype's own prefix before counting its
    third base. Yields the codon start (the third position is ``codon_start + 2``)."""
    for codon_start in range(0, L - 2, 3):
        ok = True
        seen_called = False
        for s in seqs:
            p1, p2 = s[codon_start], s[codon_start + 1]
            if p1 in VALID and p2 in VALID:
                seen_called = True
                if (p1 + p2) not in FOURFOLD_PREFIXES:
                    ok = False
                    break
        if ok and seen_called:
            yield codon_start


def locus_pi(seqs, columns):
    """Mean per-site pi over the given columns (callable sites only)."""
    vals = []
    for col in columns:
        p = site_pi(s[col] for s in seqs)
        if p is not None:
            vals.append(p)
    if not vals:
        return np.nan, 0
    return float(np.mean(vals)), len(vals)


def fourfold_locus_pi(seqs, codon_starts):
    """Mean per-site pi at fourfold third positions.

    Each haplotype contributes its third base at a codon ONLY when its own first two bases
    establish a fourfold-degenerate family. Haplotypes with a gap/N (or any non-fourfold
    prefix) at positions 1-2 are excluded from that codon's pi, because their third base
    cannot be assumed synonymous. This fixes the prior behaviour where such a haplotype was
    ignored when classifying the column as fourfold yet still counted in pi (e.g.
    ['GCA','NNG'] wrongly yielded pi = 1.0 at the third position)."""
    vals = []
    for cs in codon_starts:
        col = cs + 2
        bases = [
            s[col] for s in seqs
            if s[cs] in VALID and s[cs + 1] in VALID and (s[cs] + s[cs + 1]) in FOURFOLD_PREFIXES
        ]
        p = site_pi(bases)
        if p is not None:
            vals.append(p)
    if not vals:
        return np.nan, 0
    return float(np.mean(vals)), len(vals)


def whole_cds_pi(seqs, L):
    cols = range(L)
    return locus_pi(seqs, cols)


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
        # Chimp polarization: group0 is the hg38-reference arrangement. Where the
        # reference orientation is itself DERIVED (flip bit set), swap so that the
        # "direct" set always holds the ANCESTRAL haplotypes and "inverted" the
        # DERIVED ones (inverted == derived w.r.t. chimp).
        if is_flipped(key[0], key[1], key[2]):
            s0, s1 = s1, s0
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
            ff0, ff0n = fourfold_locus_pi(s0, ff_cols)
            ff1, ff1n = fourfold_locus_pi(s1, ff_cols)
            if ff0n or ff1n:
                a["n_cds_ff"] += 1
            if ff0n:
                a["ff0_num"] += ff0 * ff0n
                a["ff0_den"] += ff0n
            if ff1n:
                a["ff1_num"] += ff1 * ff1n
                a["ff1_den"] += ff1n

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
    return pd.DataFrame(rows)


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
            # output.csv is stored chimp-POLARIZED on disk (0_*=ancestral/direct,
            # 1_*=derived/inverted), so its columns are read directly -- no swap.
            # (The phy-derived fourfold/wholeCDS columns ARE swapped via is_flipped
            # because phy files remain in the raw hg38 group encoding.)
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


def make_figure(df):
    """Supplementary figure: fourfold pi by orientation x recurrence, plus a
    fourfold-vs-whole-locus scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cats = [(0, "Single-event"), (1, "Recurrent")]
    colors = {"direct": "#2196F3", "inverted": "#F44336"}

    # Panels 1-2: violins of fourfold pi by orientation, split by recurrence
    for ax, (code, name) in zip(axes[:2], cats):
        sub = df[df["recurrence"] == code]
        data = [
            sub["pi_fourfold_direct"].dropna().values,
            sub["pi_fourfold_inverted"].dropna().values,
        ]
        data = [d for d in data if len(d)]
        if not data:
            ax.set_title(f"{name}\n(no data)")
            continue
        parts = ax.violinplot(data, showextrema=False)
        for pc, c in zip(parts["bodies"], [colors["direct"], colors["inverted"]]):
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
            pc.set_edgecolor("black")
        for i, d in enumerate(data, 1):
            x = np.random.normal(i, 0.05, size=len(d))
            ax.scatter(x, d, s=18, alpha=0.7, color="black", zorder=3)
            ax.hlines(np.median(d), i - 0.2, i + 0.2, color="black", lw=2, zorder=4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Direct", "Inverted"])
        ax.set_ylabel("pi at 4-fold sites")
        ax.set_title(f"{name} (n={len(sub.dropna(subset=['pi_fourfold_direct']))})")

    # Panel 3: fourfold pi vs whole-locus pi
    ax = axes[2]
    for orient, col_ff, col_wl, c in [
        ("direct", "pi_fourfold_direct", "pi_wholeLocus_direct", colors["direct"]),
        ("inverted", "pi_fourfold_inverted", "pi_wholeLocus_inverted", colors["inverted"]),
    ]:
        s = df.dropna(subset=[col_ff, col_wl])
        ax.scatter(s[col_wl], s[col_ff], s=28, alpha=0.7, color=c, label=orient, edgecolor="white")
    lim = max(
        df[["pi_fourfold_direct", "pi_fourfold_inverted", "pi_wholeLocus_direct", "pi_wholeLocus_inverted"]]
        .max(numeric_only=True)
        .max(),
        1e-6,
    )
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("pi whole-locus (output.csv)")
    ax.set_ylabel("pi at 4-fold sites")
    ax.set_title("4-fold vs whole-locus pi")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {OUT_FIG}")


# ------------------------- MAIN ------------------------------


def main():
    phy_dir, tmp_dir = resolve_phy_dir()
    try:
        df = collect_fourfold_pi(phy_dir)
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
    df = attach_whole_locus_pi(df)
    df = df.sort_values(["chr", "region_start"]).reset_index(drop=True)
    df.to_csv(OUT_TABLE, sep="\t", index=False)
    print(f"Saved per-inversion table -> {OUT_TABLE} ({len(df)} loci)")

    tests = run_tests(df)
    tests.to_csv(OUT_TESTS, sep="\t", index=False)
    print(f"Saved tests -> {OUT_TESTS}")

    make_figure(df)

    # Console summary by recurrence category
    print("\n=== Median pi by orientation x recurrence ===")
    for code, name in [(0, "Single-event"), (1, "Recurrent")]:
        sub = df[df["recurrence"] == code]
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
