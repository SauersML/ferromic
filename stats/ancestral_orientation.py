#!/usr/bin/env python3
"""Polarize inversion orientation (ancestral vs derived) using the chimp outgroup.

Reviewer 1 comment 4 / Reviewer 2 comment 7: the less-frequent arrangement is
"not necessarily the derived arrangement"; reviewers ask that the ancestral vs
derived orientation be determined (e.g. by chimp) so that pi/divergence
asymmetries between the direct and inverted orientations can be interpreted --
in particular, some inversions with lower pi in the DIRECT orientation may
actually be derived in that orientation (Reviewer 2, Fig 2A point).

Method: structural breakpoint/flanking orientation from chimp
------------------------------------------------------------
The repository's coordinate convention (Porubsky et al. 2022 / the Rust
pipeline) is that group0 = "direct" = the hg38 REFERENCE-genome orientation and
group1 = "inverted" = the opposite orientation. The ancestral arrangement is
therefore whichever orientation the chimp (PanTro6) reference matches.

We read that directly from the UCSC hg38-vs-panTro6 *net* AXT alignment -- the
same alignment cds/axt_to_phy.py uses. In a net AXT the human (target) side is
always on the + strand; the strand field gives the orientation of the chimp
(query) alignment. Across an inversion interval:

  * chimp aligns mostly on the '+' strand  -> chimp is COLLINEAR with the hg38
    reference orientation -> the DIRECT (reference) orientation is ANCESTRAL,
    the inverted orientation is derived.
  * chimp aligns mostly on the '-' strand  -> chimp is INVERTED relative to the
    hg38 reference -> the reference orientation is itself DERIVED, so the
    INVERTED (group1) orientation is ANCESTRAL.

For each inversion interval we sum aligned chimp base-pairs on each strand
(restricted to blocks whose chimp contig is a real chromosome, so spurious
unplaced-contig hits do not dominate) and call the ancestral orientation from
the majority strand. The fraction of chimp-aligned bp on the majority strand is
a continuous confidence score.

This is a STRUCTURAL read-out of the ancestral arrangement and does not depend
on the orientation groups carrying distinct SNP backgrounds (they largely do
not -- these are young, often recurrent inversions that share polymorphism), so
it works for loci where one orientation is represented by a single haplotype.

Caveats reported honestly per locus:
  * loci with little/no chimp net alignment over the interval -> "ambiguous".
  * loci where both strands carry substantial chimp alignment (mixed/segmental
    duplication / inverted repeats at the breakpoints) -> lower confidence.
  * For recurrent inversions the structural ancestral state is the ancestral
    *arrangement*; individual present-day haplotypes of a given orientation may
    have arisen multiple times. The recurrence class is carried in the table so
    these can be interpreted with appropriate caution.

Output: data/ancestral_orientation.tsv with, per locus:
  inv_id, chrom, start, end, recurrence_class,
  n_direct_hap, n_inverted_hap,
  chimp_bp_plus, chimp_bp_minus, chimp_bp_total, strand_fraction,
  ancestral_orientation, ancestral_call_confidence,
  derived_orientation, minor_orientation, minor_is_derived,
  pi_direct, pi_inverted, direct_pi_lt_inverted, direct_derived_low_pi_flag
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import re
import zipfile
from collections import defaultdict

# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

PHY_OUTPUTS_ZIP_NAME = "phy_outputs.zip"
LFS_OID = "03f9b4d8167a0f2b3e715c6c978eddb9b03340a4334aa9ec50c07a3a8b7abf7d"

AXT_URL = "http://hgdownload.soe.ucsc.edu/goldenpath/hg38/vsPanTro6/hg38.panTro6.net.axt.gz"
AXT_GZ_NAME = "hg38.panTro6.net.axt.gz"

OUT_TABLE = os.path.join(DATA_DIR, "ancestral_orientation.tsv")

VALID = set("ACGT")

# Calling thresholds (on the fraction of chimp-aligned bp on the majority strand).
MIN_CHIMP_BP = 200          # need this much chimp net alignment to call at all
FRACTION_CONFIDENT = 0.90   # majority-strand fraction >= this -> "high"
FRACTION_MODERATE = 0.70    # >= this -> "moderate"; below -> "ambiguous"

REGION_RE = re.compile(
    r"^inversion_group(?P<g>[01])_(?P<chrom>[^_]+)_start(?P<s>\d+)_end(?P<e>\d+)\.phy\.gz$"
)
SEQ_RE = re.compile(r"([ACGTNacgtn-]+)\s*$")
AXT_HEADER_RE = re.compile(
    r"^(-?\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+([+-])\s+(\d+)$"
)
# A "real" chimp chromosome (panTro6): chr1..chr22, chrX, chrY (not chrUn_* /
# *_random / scaffolds). Restricting to these avoids unplaced-contig noise.
REAL_CHIMP_CHR_RE = re.compile(r"^chr([0-9]+|[XY])$", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Recover region alignments (for haplotype counts) from phy_outputs.zip / LFS
# ----------------------------------------------------------------------------

def _resolve_input(name: str) -> str:
    for base in (os.getcwd(), DATA_DIR):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return name


def _find_lfs_object() -> str | None:
    rel = os.path.join(".git", "lfs", "objects", LFS_OID[:2], LFS_OID[2:4], LFS_OID)
    for base in (REPO_ROOT, os.getcwd(), os.path.join(REPO_ROOT, "..")):
        c = os.path.join(base, rel)
        if os.path.exists(c):
            return c
    return None


def open_region_archive():
    """Return (open ZipFile of the inner region archive, cleanup callable)."""
    src = _resolve_input(PHY_OUTPUTS_ZIP_NAME)
    if not os.path.exists(src):
        src = _find_lfs_object()
    if not src or not os.path.exists(src):
        raise SystemExit(
            "Could not locate phy_outputs.zip (region alignments). Provide "
            "data/phy_outputs.zip or its git-LFS object."
        )
    log(f"Reading region alignments from {src}")
    outer = zipfile.ZipFile(src)
    if PHY_OUTPUTS_ZIP_NAME in outer.namelist():
        inner_bytes = outer.read(PHY_OUTPUTS_ZIP_NAME)
        outer.close()
        return zipfile.ZipFile(io.BytesIO(inner_bytes)), lambda: None
    return outer, outer.close


def count_haps(raw: bytes) -> int:
    """Number of haplotype sequences in a gzipped PHYLIP region alignment."""
    text = gzip.decompress(raw).decode()
    lines = text.splitlines()
    if not lines:
        return 0
    hdr = lines[0].split()
    if len(hdr) == 2 and hdr[0].isdigit():
        return int(hdr[0])
    return sum(1 for line in lines[1:] if SEQ_RE.search(line))


# ----------------------------------------------------------------------------
# Chimp strand coverage per inversion interval from UCSC net AXT
# ----------------------------------------------------------------------------

def norm_chrom(raw: str) -> str | None:
    if raw is None:
        return None
    c = str(raw).strip()
    while c.lower().startswith("chr"):
        c = c[3:]
    if not c:
        return None
    cl = c.lower()
    if cl == "x":
        core = "X"
    elif cl == "y":
        core = "Y"
    elif cl in {"m", "mt"}:
        core = "M"
    elif cl.isdigit():
        core = str(int(cl))
    else:
        core = c.upper()
    return "chr" + core


def chimp_strand_coverage(axt_path: str, regions):
    """regions: dict (chrom_norm,start,end) -> length (1-based inclusive human).

    Stream the net AXT once. For each block, intersect its human target span
    with every overlapping region and add the overlap length to that region's
    per-strand chimp coverage (only counting blocks whose chimp query is a real
    chromosome). Returns dict key -> {'+': bp, '-': bp}."""
    BIN = 100000
    index = defaultdict(lambda: defaultdict(list))  # chrom -> bin -> [(key,s,e)]
    cov = {k: {"+": 0, "-": 0} for k in regions}
    for (chrom, s, e) in regions:
        b0, b1 = (s - 1) // BIN, (e - 1) // BIN
        for b in range(b0, b1 + 1):
            index[chrom][b].append(((chrom, s, e), s, e))

    opener = gzip.open if axt_path.endswith(".gz") else open
    blocks = 0
    with opener(axt_path, "rt", encoding="latin-1") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            header = header.strip()
            m = AXT_HEADER_RE.match(header)
            if not m:
                # not a header line (blank / continuation); skip
                if header:
                    continue
                continue
            fh.readline()  # human seq line
            fh.readline()  # chimp seq line
            chrom = norm_chrom(m.group(2))
            ts, te = int(m.group(3)), int(m.group(4))  # 1-based inclusive human
            q_name, strand = m.group(5), m.group(8)
            blocks += 1
            if blocks % 500000 == 0:
                log(f"  ...parsed {blocks} AXT blocks")
            if chrom not in index:
                continue
            if not REAL_CHIMP_CHR_RE.match(q_name):
                continue
            chrom_bins = index[chrom]
            seen = set()
            for b in range((ts - 1) // BIN, (te - 1) // BIN + 1):
                for rec in chrom_bins.get(b, ()):
                    key, s, e = rec
                    if key in seen:
                        continue
                    ov = min(te, e) - max(ts, s) + 1
                    if ov > 0:
                        cov[key][strand] += ov
                        seen.add(key)
    log(f"Parsed {blocks} AXT blocks total.")
    return cov


def ensure_axt(local_axt: str | None) -> str:
    if local_axt and os.path.exists(local_axt):
        return local_axt
    for cand in (AXT_GZ_NAME, AXT_GZ_NAME[:-3], os.path.join(DATA_DIR, AXT_GZ_NAME)):
        if os.path.exists(cand):
            return cand
    import requests  # only needed when downloading

    log(f"Downloading {AXT_URL} ...")
    with requests.get(AXT_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(AXT_GZ_NAME, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                if chunk:
                    f.write(chunk)
    log(f"Downloaded {AXT_GZ_NAME} ({os.path.getsize(AXT_GZ_NAME)/1e6:.0f} MB).")
    return AXT_GZ_NAME


# ----------------------------------------------------------------------------
# Cross-reference tables
# ----------------------------------------------------------------------------

def load_recurrence():
    path = _resolve_input("inv_properties.tsv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        rdr = csv.reader(fh, delimiter="\t")
        header = next(rdr)
        idx = {h.strip(): i for i, h in enumerate(header)}
        ci, si, ei = idx.get("Chromosome"), idx.get("Start"), idx.get("End")
        rci = idx.get("0_single_1_recur_consensus")
        for row in rdr:
            try:
                key = (norm_chrom(row[ci]), int(row[si]), int(row[ei]))
            except (ValueError, IndexError, TypeError):
                continue
            label = row[rci].strip() if rci is not None and rci < len(row) else ""
            out[key] = {"1": "recurrent", "0": "single-event"}.get(label, "unknown")
    return out


def load_pi():
    path = _resolve_input("output.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                key = (norm_chrom(row["chr"]), int(row["region_start"]), int(row["region_end"]))
            except (ValueError, KeyError, TypeError):
                continue

            def fv(x):
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return None

            out[key] = (fv(row.get("0_pi")), fv(row.get("1_pi")))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Chimp polarization of inversion orientation.")
    ap.add_argument("--axt", help="Local hg38.panTro6 net AXT (.axt[.gz]); else "
                                   "looked up locally then downloaded.")
    ap.add_argument("--limit", type=int, default=0, help="Process only N loci (debug).")
    ap.add_argument("--out", default=OUT_TABLE)
    args = ap.parse_args()

    # 1. Enumerate loci and haplotype counts from the region alignments.
    arc, cleanup = open_region_archive()
    loci = defaultdict(dict)
    for name in arc.namelist():
        m = REGION_RE.match(os.path.basename(name))
        if not m:
            continue
        key = (norm_chrom(m.group("chrom")), int(m.group("s")), int(m.group("e")))
        loci[key][m.group("g")] = name
    keys = sorted(loci)
    if args.limit:
        keys = keys[: args.limit]

    hap_counts = {}
    region_len = {}
    for key in keys:
        d = loci[key]
        n0 = count_haps(arc.read(d["0"])) if "0" in d else 0
        n1 = count_haps(arc.read(d["1"])) if "1" in d else 0
        hap_counts[key] = (n0, n1)
        region_len[key] = key[2] - key[1] + 1
    cleanup()
    log(f"Enumerated {len(keys)} inversion loci.")

    # 2. Chimp per-strand coverage over each interval from the net AXT.
    axt_path = ensure_axt(args.axt)
    cov = chimp_strand_coverage(axt_path, region_len)

    # 3. Cross-reference tables.
    recur = load_recurrence()
    pis = load_pi()

    # 4. Call ancestral orientation per locus.
    rows = []
    for key in keys:
        chrom, s, e = key
        n_direct, n_inverted = hap_counts[key]
        bp_plus = cov.get(key, {}).get("+", 0)
        bp_minus = cov.get(key, {}).get("-", 0)
        bp_total = bp_plus + bp_minus

        if bp_total < MIN_CHIMP_BP:
            ori, conf, frac = "ambiguous", "no_chimp_alignment", None
        else:
            maj = max(bp_plus, bp_minus)
            frac = maj / bp_total
            # '+' dominant => reference (direct) orientation is ancestral.
            anc = "direct" if bp_plus >= bp_minus else "inverted"
            if frac >= FRACTION_CONFIDENT:
                ori, conf = anc, "high"
            elif frac >= FRACTION_MODERATE:
                ori, conf = anc, "moderate"
            else:
                ori, conf = "ambiguous", "mixed_strand"

        if ori in ("direct", "inverted"):
            derived = "inverted" if ori == "direct" else "direct"
        else:
            derived = "NA"

        # Minor (less-frequent) arrangement = orientation with fewer haplotypes.
        # Ties / missing counts -> use inverted as the nominal minor only if it
        # is strictly fewer; otherwise mark NA.
        if n_direct == 0 or n_inverted == 0:
            minor = "NA"
        elif n_inverted < n_direct:
            minor = "inverted"
        elif n_direct < n_inverted:
            minor = "direct"
        else:
            minor = "tie"

        if derived != "NA" and minor in ("direct", "inverted"):
            minor_is_derived = (minor == derived)
        else:
            minor_is_derived = "NA"

        pi_d, pi_i = pis.get(key, (None, None))
        direct_pi_lt_inverted = (pi_d is not None and pi_i is not None and pi_d < pi_i)
        direct_derived_low_pi = (direct_pi_lt_inverted and derived == "direct")

        rows.append({
            "inv_id": f"{chrom}:{s}-{e}",
            "chrom": chrom, "start": s, "end": e,
            "recurrence_class": recur.get(key, "unknown"),
            "n_direct_hap": n_direct, "n_inverted_hap": n_inverted,
            "chimp_bp_plus": bp_plus, "chimp_bp_minus": bp_minus,
            "chimp_bp_total": bp_total,
            "strand_fraction": "NA" if frac is None else round(frac, 4),
            "ancestral_orientation": ori,
            "ancestral_call_confidence": conf,
            "derived_orientation": derived,
            "minor_orientation": minor,
            "minor_is_derived": minor_is_derived,
            "pi_direct": "NA" if pi_d is None else pi_d,
            "pi_inverted": "NA" if pi_i is None else pi_i,
            "direct_pi_lt_inverted": direct_pi_lt_inverted,
            "direct_derived_low_pi_flag": direct_derived_low_pi,
        })

    # 5. Write the table.
    cols = [
        "inv_id", "chrom", "start", "end", "recurrence_class",
        "n_direct_hap", "n_inverted_hap",
        "chimp_bp_plus", "chimp_bp_minus", "chimp_bp_total", "strand_fraction",
        "ancestral_orientation", "ancestral_call_confidence",
        "derived_orientation", "minor_orientation", "minor_is_derived",
        "pi_direct", "pi_inverted", "direct_pi_lt_inverted", "direct_derived_low_pi_flag",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["chrom"], r["start"])):
            w.writerow(r)

    # 6. Summary.
    n = len(rows)
    called = [r for r in rows if r["ancestral_orientation"] in ("direct", "inverted")]
    high = [r for r in called if r["ancestral_call_confidence"] == "high"]
    moderate = [r for r in called if r["ancestral_call_confidence"] == "moderate"]
    direct_anc = sum(1 for r in called if r["ancestral_orientation"] == "direct")
    inverted_anc = sum(1 for r in called if r["ancestral_orientation"] == "inverted")
    minor_called = [r for r in called if r["minor_is_derived"] in (True, False)]
    minor_derived = sum(1 for r in minor_called if r["minor_is_derived"] is True)
    flagged = [r for r in rows if r["direct_derived_low_pi_flag"] is True]
    log("")
    log("=== Ancestral-orientation polarization summary ===")
    log(f"Loci in table:                       {n}")
    log(f"Confident calls (direct/inverted):   {len(called)}  "
        f"(high={len(high)}, moderate={len(moderate)})")
    log(f"  ancestral = direct (reference):    {direct_anc}")
    log(f"  ancestral = inverted:              {inverted_anc}")
    log(f"Loci where minor arrangement orientation could be tested: {len(minor_called)}")
    log(f"  minor arrangement IS derived (the common assumption):   "
        f"{minor_derived}/{len(minor_called)}")
    log(f"  minor arrangement is NOT derived (reviewer's concern):  "
        f"{len(minor_called) - minor_derived}/{len(minor_called)}")
    log(f"Direct-orientation-derived yet lower-pi loci (Reviewer 2 Fig 2A): {len(flagged)}")
    for r in flagged[:25]:
        log(f"    {r['inv_id']}  pi_direct={r['pi_direct']} < pi_inverted={r['pi_inverted']}"
            f"  conf={r['ancestral_call_confidence']} recurrence={r['recurrence_class']}")
    log(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
