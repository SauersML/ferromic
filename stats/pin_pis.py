"""Nonsynonymous vs synonymous nucleotide diversity (piN/piS) per orientation.

Reviewer 1 (comment 4) asked us to adopt the population-genomic standard of
comparing diversity (and divergence) at synonymous vs nonsynonymous sites,
following Charlesworth (2024, Genetics 226:iyad218, "Fitness consequences of
genetic divergence between polymorphic gene arrangements"). We already report
4-fold synonymous-site pi (stats/four_fold_pi.py) and PAML dN/dS separately,
but not the combined piN/piS constraint framework. This script supplies it.

For each inversion and each orientation (group0 = direct, group1 = inverted) it
computes:
  - piS : nucleotide diversity at 4-fold degenerate (synonymous) sites,
  - piN : nucleotide diversity at 0-fold degenerate (nonsynonymous) sites,
  - piN/piS : the ratio, a within-orientation index of purifying-selection
    constraint on the protein-coding sequence of the inversion.

Site classification uses the standard genetic code (NCBI table 1). A codon
position is N-fold degenerate when N of the four nucleotides at that position
encode the same amino acid (holding the other two positions fixed). The first
and second positions of almost all codons are 0-fold; the third position is
0/2/3/4-fold depending on the family. We classify a codon position as 0-fold
only when EVERY called haplotype across both orientations carries a codon whose
position is 0-fold (likewise 4-fold for the synonymous set), so the site class
is unambiguous for the sampled sequences. This mirrors how four_fold_pi.py
defines its 4-fold set from the combined sample.

Per-site pi uses the SAME estimator as the Rust pipeline
(src/stats.rs::dense_pi_from_counts):
    per-site pi = n/(n-1) * (1 - sum_i p_i^2)
and locus pi is the mean of per-site pi over callable sites (>= 2 called
A/C/G/T haplotypes), matching calculate_pi_from_summary. piN and piS are the
mean per-site pi over the 0-fold and 4-fold site sets respectively (i.e. pi per
site, directly comparable; the ratio is dimensionless and independent of the
relative number of 0-fold vs 4-fold sites).

The per-CDS alignment recovery (group0_*/group1_* .phy.gz from phy_outputs.zip
or its git-LFS object) is shared verbatim with four_fold_pi.py.

Outputs (written to data/):
  - pin_pis_by_inversion.tsv : per-inversion piN, piS, piN/piS by orientation
  - pin_pis_tests.tsv        : test statistics by orientation/recurrence
  - pin_pis.pdf              : supplementary figure

Run from the data/ directory:
    cd data && python ../stats/pin_pis.py
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
import itertools
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

OUT_TABLE = os.path.join(_DATA_DIR, "pin_pis_by_inversion.tsv")
OUT_TESTS = os.path.join(_DATA_DIR, "pin_pis_tests.tsv")
OUT_FIG = os.path.join(_DATA_DIR, "pin_pis.pdf")

VALID = set("ACGT")

# ------------------------- GENETIC CODE / DEGENERACY ----------

# Standard genetic code (NCBI translation table 1).
BASES = "TCAG"
_AA = (
    "FFLLSSSSYY**CC*W"
    "LLLLPPPPHHQQRRRR"
    "IIIMTTTTNNKKSSRR"
    "VVVVAAAADDEEGGGG"
)
CODON_TABLE = {
    a + b + c: _AA[i]
    for i, (a, b, c) in enumerate(itertools.product(BASES, repeat=3))
}


def _degeneracy(codon, pos):
    """N-fold degeneracy of `pos` (0,1,2) in `codon` under the standard code.

    Counts how many of the four nucleotides at `pos` yield the same amino acid
    as `codon` (the codon itself counts). Returns 1 for an undefined codon (stop
    or non-ACGT), so such codons contribute to neither the 0-fold nor 4-fold
    set."""
    aa = CODON_TABLE.get(codon)
    if aa is None or aa == "*":
        return None
    n_same = 0
    for b in BASES:
        alt = codon[:pos] + b + codon[pos + 1:]
        if CODON_TABLE.get(alt) == aa:
            n_same += 1
    return n_same


# Precompute, for every sense codon and position, whether it is 0-fold or 4-fold.
ZEROFOLD = {}   # (codon, pos) -> True/False
FOURFOLD = {}
for _cod in CODON_TABLE:
    if CODON_TABLE[_cod] == "*":
        continue
    for _p in range(3):
        d = _degeneracy(_cod, _p)
        ZEROFOLD[(_cod, _p)] = (d == 1)
        FOURFOLD[(_cod, _p)] = (d == 4)

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

    Resolution order matches four_fold_pi.py:
      1. FOURFOLD_PHY_DIR env var (a pre-extracted directory), if set.
      2. phy_outputs.zip in the working dir / data/.
      3. The git-LFS object for phy_outputs.zip.
    Cases 2-3 extract into a temp dir, returned as the second element for the
    caller to clean up."""
    if PHY_DIR and os.path.isdir(PHY_DIR):
        return PHY_DIR, None

    src = PHY_OUTPUTS_ZIP if os.path.exists(PHY_OUTPUTS_ZIP) else _find_lfs_object()
    if not src:
        raise SystemExit(
            "Could not locate per-CDS alignments. Set FOURFOLD_PHY_DIR to a "
            "directory of group0_*/group1_* .phy.gz files, or make "
            "phy_outputs.zip available."
        )

    tmp = tempfile.mkdtemp(prefix="pin_pis_phy_")
    print(f"Recovering CDS alignments from {src} -> {tmp}")
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

    Mirrors the sequence extraction in cds/axt_to_phy.py and four_fold_pi.py."""
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

    Returns None for uncallable sites (< 2 called A/C/G/T haplotypes)."""
    counts = Counter(b for b in column if b in VALID)
    n = sum(counts.values())
    if n < 2:
        return None
    sum_sq = sum(c * c for c in counts.values())
    return n / (n - 1.0) * (1.0 - sum_sq / (n * n))


def degenerate_columns(seqs, L):
    """Classify each codon position as 0-fold / 4-fold across the sample.

    Yields (column_index, "0" | "4"). A position is reported as 0-fold only when
    every called haplotype carries a codon for which that position is 0-fold
    under the standard code (likewise 4-fold). Codons with gaps/N or stops in a
    haplotype are ignored for that haplotype; if any called haplotype disagrees
    with the class the position is skipped. Codons must be fully called (ACGT) at
    all three positions to vote, so the classification is well defined."""
    for codon_start in range(0, L - 2, 3):
        for pos in range(3):
            col = codon_start + pos
            is_zero = True
            is_four = True
            seen = False
            for s in seqs:
                codon = s[codon_start:codon_start + 3]
                if any(b not in VALID for b in codon):
                    continue
                key = (codon, pos)
                if key not in ZEROFOLD:   # stop codon
                    is_zero = is_four = False
                    break
                seen = True
                if not ZEROFOLD[key]:
                    is_zero = False
                if not FOURFOLD[key]:
                    is_four = False
                if not is_zero and not is_four:
                    break
            if not seen:
                continue
            if is_zero:
                yield col, "0"
            elif is_four:
                yield col, "4"


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


# ------------------------- DATA LOADING ----------------------


def load_recurrence():
    """Return dict (chrom, start, end) -> recurrence consensus (0/1/NaN)."""
    inv = pd.read_csv(INVINFO_TSV, sep="\t")
    rec = {}
    for _, r in inv.iterrows():
        key = (str(r["Chromosome"]), int(r["Start"]), int(r["End"]))
        rec[key] = r["0_single_1_recur_consensus"]
    return rec


def collect_pin_pis(phy_dir):
    """Walk all group0/group1 CDS alignments; aggregate piN (0-fold) and piS
    (4-fold) to the inversion level (per-CDS pi weighted by callable sites)."""
    g0_files = sorted(glob.glob(os.path.join(phy_dir, "group0_*ENST*.phy.gz")))

    acc = defaultdict(
        lambda: {
            "piN0_num": 0.0, "piN0_den": 0,   # 0-fold, direct
            "piN1_num": 0.0, "piN1_den": 0,   # 0-fold, inverted
            "piS0_num": 0.0, "piS0_den": 0,   # 4-fold, direct
            "piS1_num": 0.0, "piS1_den": 0,   # 4-fold, inverted
            "n_cds": 0, "n_cds_used": 0,
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
        # Chimp polarization: swap so "direct" (0) holds the ANCESTRAL arrangement
        # and "inverted" (1) the DERIVED one (inverted == derived w.r.t. chimp).
        if is_flipped(key[0], key[1], key[2]):
            s0, s1 = s1, s0
        L = L0
        n_proc += 1

        # Define the 0-fold and 4-fold site sets from the COMBINED sample so the
        # classification is consistent across orientations, then measure pi
        # within each orientation at those sites.
        combined = s0 + s1
        zero_cols, four_cols = [], []
        for col, cls in degenerate_columns(combined, L):
            (zero_cols if cls == "0" else four_cols).append(col)

        a = acc[key]
        a["n_cds"] += 1
        used = False

        if zero_cols:
            n0, n0n = locus_pi(s0, zero_cols)
            n1, n1n = locus_pi(s1, zero_cols)
            if n0n:
                a["piN0_num"] += n0 * n0n
                a["piN0_den"] += n0n
                used = True
            if n1n:
                a["piN1_num"] += n1 * n1n
                a["piN1_den"] += n1n
                used = True
        if four_cols:
            s0v, s0n = locus_pi(s0, four_cols)
            s1v, s1n = locus_pi(s1, four_cols)
            if s0n:
                a["piS0_num"] += s0v * s0n
                a["piS0_den"] += s0n
                used = True
            if s1n:
                a["piS1_num"] += s1v * s1n
                a["piS1_den"] += s1n
                used = True
        if used:
            a["n_cds_used"] += 1

    print(f"Processed {n_proc} CDS group pairs across {len(acc)} inversion loci.")

    rec = load_recurrence()
    rows = []
    for key, a in acc.items():
        chrom, istart, iend = key

        def ratio(num, den):
            return (num / den) if den > 0 else np.nan

        piN_dir = ratio(a["piN0_num"], a["piN0_den"])
        piN_inv = ratio(a["piN1_num"], a["piN1_den"])
        piS_dir = ratio(a["piS0_num"], a["piS0_den"])
        piS_inv = ratio(a["piS1_num"], a["piS1_den"])

        def pn_ps(pn, ps):
            return (pn / ps) if (np.isfinite(pn) and np.isfinite(ps) and ps > 0) else np.nan

        rows.append(
            {
                "chr": chrom,
                "region_start": istart,
                "region_end": iend,
                "recurrence": rec.get(key, np.nan),
                "n_cds": a["n_cds"],
                "n_cds_used": a["n_cds_used"],
                "zerofold_sites_direct": a["piN0_den"],
                "zerofold_sites_inverted": a["piN1_den"],
                "fourfold_sites_direct": a["piS0_den"],
                "fourfold_sites_inverted": a["piS1_den"],
                "piN_direct": piN_dir,
                "piN_inverted": piN_inv,
                "piS_direct": piS_dir,
                "piS_inverted": piS_inv,
                "piN_piS_direct": pn_ps(piN_dir, piS_dir),
                "piN_piS_inverted": pn_ps(piN_inv, piS_inv),
            }
        )
    return pd.DataFrame(rows)


# ------------------------- STATISTICS ------------------------


def paired_wilcoxon(delta):
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0 or np.allclose(delta, 0.0):
        return np.nan, np.nan, len(delta)
    w, p = stats.wilcoxon(delta)
    return w, p, len(delta)


def run_tests(df):
    """Tests mirroring four_fold_pi.py / recur_diversity.py.

    For piN, piS, and piN/piS:
      - paired Wilcoxon (inverted vs direct) within each recurrence category,
      - Mann-Whitney U on the inverted-vs-direct delta (single vs recurrent),
      - Mann-Whitney U recurrent vs single within each orientation.
    piN and piS deltas use log1p (as four_fold_pi.py does for pi); the piN/piS
    ratio is already dimensionless so its delta is taken on the raw ratio."""
    results = []

    specs = [
        ("piN", "piN_direct", "piN_inverted", True),
        ("piS", "piS_direct", "piS_inverted", True),
        ("piN/piS", "piN_piS_direct", "piN_piS_inverted", False),
    ]

    for metric, cdir, cinv, use_log1p in specs:
        sub = df.dropna(subset=[cdir, cinv, "recurrence"]).copy()
        if use_log1p:
            sub["delta"] = np.log1p(sub[cinv]) - np.log1p(sub[cdir])
        else:
            sub["delta"] = sub[cinv] - sub[cdir]

        single = sub[sub["recurrence"] == 0]
        recur = sub[sub["recurrence"] == 1]

        for label, grp in [("single", single), ("recurrent", recur)]:
            w, p, n = paired_wilcoxon(grp["delta"].values)
            results.append(
                {
                    "metric": metric,
                    "test": "paired Wilcoxon (inverted vs direct)",
                    "category": label,
                    "n": n,
                    "median_direct": grp[cdir].median(),
                    "median_inverted": grp[cinv].median(),
                    "statistic": w,
                    "p_value": p,
                }
            )

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
    """Supplementary figure: piN/piS by orientation x recurrence, and a piN-vs-piS
    scatter showing the constraint (most points below the y=x diagonal)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cats = [(0, "Single-event"), (1, "Recurrent")]
    colors = {"direct": "#2196F3", "inverted": "#F44336"}

    # Panels 1-2: violins of piN/piS by orientation, split by recurrence
    for ax, (code, name) in zip(axes[:2], cats):
        sub = df[df["recurrence"] == code]
        data = [
            sub["piN_piS_direct"].replace([np.inf, -np.inf], np.nan).dropna().values,
            sub["piN_piS_inverted"].replace([np.inf, -np.inf], np.nan).dropna().values,
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
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Direct", "Inverted"])
        ax.set_ylabel("piN / piS")
        ax.set_title(f"{name} (n={len(sub.dropna(subset=['piN_piS_direct']))})")

    # Panel 3: piN vs piS (both orientations); points below y=x indicate
    # purifying selection (piN < piS).
    ax = axes[2]
    for orient, cn, cs, c in [
        ("direct", "piN_direct", "piS_direct", colors["direct"]),
        ("inverted", "piN_inverted", "piS_inverted", colors["inverted"]),
    ]:
        s = df.dropna(subset=[cn, cs])
        ax.scatter(s[cs], s[cn], s=28, alpha=0.7, color=c, label=orient, edgecolor="white")
    lim = max(
        df[["piN_direct", "piN_inverted", "piS_direct", "piS_inverted"]]
        .max(numeric_only=True)
        .max(),
        1e-6,
    )
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("piS (4-fold synonymous)")
    ax.set_ylabel("piN (0-fold nonsynonymous)")
    ax.set_title("piN vs piS (below diagonal = constraint)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {OUT_FIG}")


# ------------------------- MAIN ------------------------------


def main():
    phy_dir, tmp_dir = resolve_phy_dir()
    try:
        df = collect_pin_pis(phy_dir)
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
    df = df.sort_values(["chr", "region_start"]).reset_index(drop=True)
    df.to_csv(OUT_TABLE, sep="\t", index=False)
    print(f"Saved per-inversion table -> {OUT_TABLE} ({len(df)} loci)")

    tests = run_tests(df)
    tests.to_csv(OUT_TESTS, sep="\t", index=False)
    print(f"Saved tests -> {OUT_TESTS}")

    make_figure(df)

    # Console summary by recurrence category
    print("\n=== Median piN, piS, piN/piS by orientation x recurrence ===")
    for code, name in [(0, "Single-event"), (1, "Recurrent")]:
        sub = df[df["recurrence"] == code]
        print(f"\n{name} (n loci = {len(sub)}):")
        for label, col in [
            ("piN     direct  ", "piN_direct"),
            ("piN     inverted", "piN_inverted"),
            ("piS     direct  ", "piS_direct"),
            ("piS     inverted", "piS_inverted"),
            ("piN/piS direct  ", "piN_piS_direct"),
            ("piN/piS inverted", "piN_piS_inverted"),
        ]:
            v = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
            med = v.median() if len(v) else float("nan")
            print(f"  {label}: n={len(v):2d} median={med:.6e}")

    # Overall (pooled) piN/piS for the abstract / response letter
    print("\n=== Pooled piN/piS (all loci) ===")
    for orient, col in [("direct", "piN_piS_direct"), ("inverted", "piN_piS_inverted")]:
        v = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v):
            print(f"  {orient:>8}: n={len(v):2d} median={v.median():.4f} mean={v.mean():.4f}")


if __name__ == "__main__":
    main()
