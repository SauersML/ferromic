#!/usr/bin/env python3
"""Apply chimp-polarization (inverted == derived) across committed data artifacts.

This migrates every orientation-bearing data file from the historical
hg38-REFERENCE encoding (0/group0/"direct" = reference arrangement) to the
chimp-POLARIZED encoding (0/"direct" = ANCESTRAL, 1/"inverted" = DERIVED), using
``data/inversion_polarity.tsv`` as the single source of truth. For each inversion
whose ``flip_ref_polarity`` bit is set (the hg38 reference orientation is itself
derived), the per-orientation quantities are swapped so that the "inverted"
columns always describe the DERIVED arrangement.

The transform is a deterministic, label-permutation operation and is provably
equivalent to re-running the pipeline from a polarity-flipped genotype callset
for every statistic that is computed per orientation group and merely labelled
0/1 (pi, theta, segregating sites, haplotype counts, piN/piS, four-fold pi, PAML
omega_direct/inverted, fixed-difference counts, tagging-SNP frequencies).
Symmetric quantities (dxy, Fst, pi_avg, da, |r|) are unchanged; sign-bearing
quantities (correlation r, signed inverted-minus-direct deltas) are negated;
boolean comparison flags are recomputed.

Aggregate / across-locus summary tables (``*_tests.tsv``, GRAND_PAML overall
rows) are NOT swap-migratable because they pool flipped and unflipped loci;
regenerate those by re-running their producing scripts on the migrated
per-inversion inputs (see --list-regenerate).

Idempotent: refuses to run twice against the same polarity table unless --force.
Originals are recoverable from git history.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")

POLARITY = os.path.join(DATA, "inversion_polarity.tsv")
MARKER = os.path.join(DATA, ".polarity_applied")

_REGION_RE = re.compile(r"(?:chr)?(\w+?)[\s:_-]+(\d+)[\s:_-]+(\d+)")


# Stdlib-only polarity loader (mirrors stats/_inv_common.load_polarity so this
# migration tool has no third-party dependency).
def _norm_chrom(raw):
    if raw is None:
        return None
    c = str(raw).strip()
    while c.lower().startswith("chr"):
        c = c[3:]
    if not c:
        return None
    cl = c.lower()
    core = {"x": "X", "y": "Y", "m": "M", "mt": "M"}.get(cl,
            str(int(cl)) if cl.isdigit() else c.upper())
    return "chr" + core


def _to_int(x):
    try:
        return int(float(str(x).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


class Polarity:
    def __init__(self, path):
        self.by_coord = {}
        self.by_orig = {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                chrom = _norm_chrom(row.get("chrom"))
                s, e = _to_int(row.get("start")), _to_int(row.get("end"))
                if chrom is None or s is None or e is None:
                    continue
                rec = {"flip": str(row.get("flip_ref_polarity", "0")).strip() == "1",
                       "confidence": (row.get("confidence") or "").strip()}
                self.by_coord[(chrom, s, e)] = rec
                for orig in (row.get("orig_id") or "").split(";"):
                    orig = orig.strip()
                    if orig:
                        self.by_orig[orig] = rec

    def record(self, chrom=None, start=None, end=None, orig=None):
        nc, s, e = _norm_chrom(chrom), _to_int(start), _to_int(end)
        if nc is not None and s is not None and e is not None:
            for ds in (0, -1, 1):
                for de in (0, -1, 1):
                    r = self.by_coord.get((nc, s + ds, e + de))
                    if r is not None:
                        return r
        if orig:
            return self.by_orig.get(str(orig).strip())
        return None


_POL = None


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(orig, val):
    """Format a numeric back to a string, preserving int-ness where the source
    looked like an int."""
    if val is None:
        return orig
    s = str(orig).strip()
    if s and re.fullmatch(r"[+-]?\d+", s):
        return str(int(round(val)))
    return repr(val) if isinstance(val, float) else str(val)


def parse_coords(row, key):
    """Return (chrom,start,end,orig_id) from a row given a key spec."""
    if key["type"] == "cols":
        return (row.get(key["chrom"]), row.get(key["start"]),
                row.get(key["end"]), row.get(key.get("orig")) if key.get("orig") else None)
    if key["type"] == "region_str":
        m = _REGION_RE.search(str(row.get(key["col"], "")))
        if m:
            return (m.group(1), m.group(2), m.group(3),
                    row.get(key.get("orig")) if key.get("orig") else None)
    return (None, None, None, None)


# ---- per-file specifications --------------------------------------------------
# swap: column pairs swapped on flip. complement: x->1-x. negate: x->-x.
# recompute_lt: (out, a, b) sets out := str(a<b) after swap. genotype_swap: list
# of sample columns whose 0/1 alleles are swapped; complement applies to inv_AF.

SPECS = [
    {
        "path": "callset.tsv", "sep": "\t",
        "key": {"type": "cols", "chrom": "seqnames", "start": "start", "end": "end",
                "orig": "inv_id"},
        "genotype_swap": "samples",  # columns after the fixed metadata block
        "complement": ["inv_AF"],
    },
    {
        "path": "output.csv", "sep": ",",
        "key": {"type": "cols", "chrom": "chr", "start": "region_start", "end": "region_end"},
        "swap": [
            ("0_sequence_length", "1_sequence_length"),
            ("0_sequence_length_adjusted", "1_sequence_length_adjusted"),
            ("0_segregating_sites", "1_segregating_sites"),
            ("0_w_theta", "1_w_theta"),
            ("0_pi", "1_pi"),
            ("0_segregating_sites_filtered", "1_segregating_sites_filtered"),
            ("0_w_theta_filtered", "1_w_theta_filtered"),
            ("0_pi_filtered", "1_pi_filtered"),
            ("0_num_hap_no_filter", "1_num_hap_no_filter"),
            ("0_num_hap_filter", "1_num_hap_filter"),
            ("hudson_pi_hap_group_0", "hudson_pi_hap_group_1"),
        ],
        "complement": ["inversion_freq_no_filter", "inversion_freq_filter"],
    },
    {
        "path": "inv_properties.tsv", "sep": "\t",
        "key": {"type": "cols", "chrom": "Chromosome", "start": "Start", "end": "End",
                "orig": "OrigID"},
        "complement": ["Inverted_AF"],  # now the DERIVED-orientation frequency
    },
    {
        "path": "balanced_recurrence_results.tsv", "sep": "\t",
        "key": {"type": "cols", "chrom": "Chromosome", "start": "Start", "end": "End",
                "orig": "Inversion_ID"},
        "complement": ["Inverted_AF"],
    },
    {
        "path": "recurrence_controls_covariates.tsv", "sep": "\t",
        "key": {"type": "cols", "chrom": "chr_std", "start": "region_start", "end": "region_end",
                "orig": "region_id"},
        "swap": [("pi_direct", "pi_inverted")],
        "complement": ["inv_af"],
    },
    {
        "path": "divergence_da_dxy_by_type.tsv", "sep": "\t",
        "key": {"type": "cols", "chrom": "chr", "start": "region_start", "end": "region_end"},
        "swap": [("hudson_pi_hap_group_0", "hudson_pi_hap_group_1")],
    },
    {
        # Regenerated directly by the polarity-aware stats/four_fold_pi.py
        # (which swaps group0/group1 by is_flipped). Do NOT also swap here, or the
        # flip would be applied twice. Kept for column documentation / verification.
        "path": "four_fold_pi_by_inversion.tsv", "sep": "\t", "regenerated": True,
        "key": {"type": "cols", "chrom": "chr", "start": "region_start", "end": "region_end"},
        "swap": [
            ("fourfold_sites_direct", "fourfold_sites_inverted"),
            ("pi_fourfold_direct", "pi_fourfold_inverted"),
            ("pi_wholeCDS_direct", "pi_wholeCDS_inverted"),
            ("pi_wholeLocus_direct", "pi_wholeLocus_inverted"),
        ],
    },
    {
        # Regenerated directly by the polarity-aware stats/pin_pis.py.
        "path": "pin_pis_by_inversion.tsv", "sep": "\t", "regenerated": True,
        "key": {"type": "cols", "chrom": "chr", "start": "region_start", "end": "region_end"},
        "swap": [
            ("zerofold_sites_direct", "zerofold_sites_inverted"),
            ("fourfold_sites_direct", "fourfold_sites_inverted"),
            ("piN_direct", "piN_inverted"),
            ("piS_direct", "piS_inverted"),
            ("piN_piS_direct", "piN_piS_inverted"),
        ],
    },
    {
        "path": "fixed_diff_summary.tsv", "sep": "\t",
        "key": {"type": "region_str", "col": "inv_id"},
        "swap": [("n_direct", "n_inverted")],
    },
    {
        "path": "gene_inversion_fixed_differences.tsv", "sep": "\t",
        "key": {"type": "region_str", "col": "inv_id"},
        "swap": [("n_direct", "n_inverted"), ("direct_allele", "inverted_allele")],
    },
    {
        "path": "gene_inversion_direct_inverted.tsv", "sep": "\t",
        "key": {"type": "region_str", "col": "inv_id"},
        "swap": [("p_direct", "p_inverted")],
        "negate": ["delta", "z_value"],  # delta defined as inverted - direct
    },
    {
        "path": "cds_conservation_table.tsv", "sep": "\t",
        "key": {"type": "region_str", "col": "Inversion locus"},
        "swap": [("Pairs direct", "Pairs inverted")],
        "negate": ["Δ proportion identical (inverted − direct)"],
    },
    {
        # Clade Model C omega2 is the only branch-class-specific quantity; site-class
        # proportions and omega0 are shared, and the H1-vs-H0 LRT/p/q are label-
        # symmetric. So polarizing only requires swapping the direct/inverted omega2
        # estimates (winner + each per-seed run) for flipped loci.
        "path": "GRAND_PAML_RESULTS.tsv", "sep": "\t",
        "key": {"type": "region_str", "col": "region"},
        "swap": [
            ("winner_omega2_direct", "winner_omega2_inverted"),
            ("h1_s1_def_cmc_omega2_direct_run_2", "h1_s1_def_cmc_omega2_inverted_run_2"),
            ("h1_s2_pur_cmc_omega2_direct_run_2", "h1_s2_pur_cmc_omega2_inverted_run_2"),
            ("h1_s3_pos_cmc_omega2_direct_run_2", "h1_s3_pos_cmc_omega2_inverted_run_2"),
            ("h1_s4_mix_cmc_omega2_direct_run_2", "h1_s4_mix_cmc_omega2_inverted_run_2"),
        ],
    },
    {
        # Produced by find_tagging_snps.py, which is itself is_flipped-aware (it
        # swaps group0/group1 at read time), so the committed table is ALREADY
        # polarized. Do NOT swap/negate here or flipped loci would be double-flipped
        # back to raw. Kept for column documentation / verification only.
        "path": "best_tagging_snps_qvalues.tsv", "sep": "\t", "regenerated": True,
        "key": {"type": "region_str", "col": "region"},
        "swap": [("REF_freq_direct", "REF_freq_inverted"),
                 ("ALT_freq_direct", "ALT_freq_inverted")],
        "negate": ["correlation_r"],
    },
]

# Tables that must be regenerated (pool loci) rather than swap-migrated.
REGENERATE = [
    ("four_fold_pi_tests.tsv", "stats/four_fold_pi.py"),
    ("pin_pis_tests.tsv", "stats/pin_pis.py (or producing script)"),
    ("ancestral_orientation.tsv", "superseded by inversion_polarity.tsv"),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


# A biallelic genotype call: two alleles in {0,1} joined by '|' (phased) or '/'
# (unphased), optionally followed by a suffix such as '_lowconf'. This deliberately
# excludes ArbiGent multi-allelic / copy-number codes ('0021', '2100' — no separator
# at position 1) and text codes ('noreads', 'idup_hom').
_BIALLELIC_GT_RE = re.compile(r"[01][|/][01]")


def swap_genotype_alleles(cell):
    """Swap 0<->1 in a *biallelic* genotype cell ('1|0'->'0|1', '0/0'->'1/1',
    '0|1_lowconf'->'1|0_lowconf').

    Only the leading 'A<sep>B' allele pair is swapped; any suffix is preserved.
    Non-biallelic cells are returned UNCHANGED: ArbiGent multi-allelic / copy-number
    codes ('0021', '2100', ...) and text codes ('noreads', 'idup_hom', ...) are not
    0/1 orientation calls, and the old blind ``i < 3`` character swap corrupted them
    (e.g. '0021' -> '1121')."""
    s = str(cell)
    if not _BIALLELIC_GT_RE.match(s):
        return s
    out = []
    for i, ch in enumerate(s):
        if i < 3 and ch == "0":
            out.append("1")
        elif i < 3 and ch == "1":
            out.append("0")
        else:
            out.append(ch)
    return "".join(out)


def migrate_file(spec, dry):
    path = os.path.join(DATA, spec["path"])
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        rdr = csv.reader(fh, delimiter=spec["sep"])
        rows = list(rdr)
    if not rows:
        return None
    header = rows[0]
    col = {h: i for i, h in enumerate(header)}
    key = spec["key"]
    # resolve sample columns for genotype swap
    if spec.get("genotype_swap") == "samples":
        # everything after the last fixed metadata column 'inv_AF' (callset.tsv)
        start_i = col.get("inv_AF")
        sample_cols = list(range(start_i + 1, len(header))) if start_i is not None else []
    else:
        sample_cols = []

    keycols = [v for k, v in key.items() if k in ("chrom", "start", "end", "col", "orig")]
    n_match = n_flip = n_unmatched = 0
    examples = []
    out_rows = [header]
    for r in rows[1:]:
        # Drop genuinely empty rows (e.g. a trailing blank line) instead of
        # padding them into an all-blank row, which downstream readers parse as a
        # spurious NaN record.
        if not any(str(x).strip() for x in r):
            continue
        if len(r) < len(header):
            r += [""] * (len(header) - len(r))
        out_rows.append(r)
        rowd = {h: r[col[h]] for h in header}
        chrom, start, end, orig = parse_coords(rowd, key)
        rec = _POL.record(chrom, start, end, orig)
        if rec is None:
            n_unmatched += 1
            continue
        n_match += 1
        if not rec.get("flip"):
            continue
        n_flip += 1
        before = list(r)
        # swaps
        for a, b in spec.get("swap", []):
            if a in col and b in col:
                r[col[a]], r[col[b]] = r[col[b]], r[col[a]]
        # complements x -> 1-x
        for c in spec.get("complement", []):
            if c in col:
                v = _num(r[col[c]])
                if v is not None:
                    r[col[c]] = _fmt(r[col[c]], 1.0 - v)
        # negations
        for c in spec.get("negate", []):
            if c in col:
                v = _num(r[col[c]])
                if v is not None:
                    r[col[c]] = _fmt(r[col[c]], -v)
        # genotype swaps
        for ci in sample_cols:
            r[ci] = swap_genotype_alleles(r[ci])
        if len(examples) < 3:
            examples.append((chrom, start, end))
    if not dry:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter=spec["sep"])
            w.writerows(out_rows)
    return {"path": spec["path"], "matched": n_match, "flipped": n_flip,
            "unmatched": n_unmatched, "examples": examples}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="re-apply even if marker present")
    ap.add_argument("--only", nargs="+", metavar="FILE",
                    help="migrate only these data file basenames (e.g. output.csv). "
                         "Use in CI to polarize a freshly-produced RAW artifact without "
                         "touching already-polarized committed tables. Skips the global "
                         "marker check (the named files are assumed RAW).")
    ap.add_argument("--list-regenerate", action="store_true")
    ap.add_argument("--data-dir", metavar="DIR",
                    help="override the directory the named tables live in (default: "
                         "repo data/). Use to polarize a freshly-produced RAW artifact "
                         "staged outside data/ (e.g. analysis_downloads/public_internet/"
                         "output.csv before figure generation).")
    args = ap.parse_args()

    if args.data_dir:
        global DATA
        DATA = os.path.abspath(args.data_dir)

    if args.list_regenerate:
        print("Tables to REGENERATE (pool loci; not swap-migratable):")
        for f, how in REGENERATE:
            print(f"  data/{f:40s} <- {how}")
        return

    if not os.path.exists(POLARITY):
        sys.exit(f"Missing {POLARITY}; run stats/polarize_orientation.py first.")
    global _POL
    _POL = Polarity(POLARITY)
    pol_hash = sha(POLARITY)
    only = set(args.only) if args.only else None
    # The global marker guards against double-flipping the full committed dataset.
    # With --only we are polarizing a specific freshly-produced RAW file (e.g. the
    # Rust output.csv in CI), so the dataset-wide marker does not apply.
    if args.apply and only is None and os.path.exists(MARKER) and not args.force:
        with open(MARKER) as fh:
            prev = fh.read().strip()
        if prev == pol_hash:
            sys.exit("Polarity already applied for this table (marker matches). "
                     "Use --force to re-apply (DANGER: double-flips).")

    dry = not args.apply
    print(f"{'DRY-RUN' if dry else 'APPLYING'} polarity from {POLARITY}")
    if only is not None:
        print(f"Restricting to: {', '.join(sorted(only))}")
    print(f"Flipped loci in table: "
          f"{sum(1 for r in _POL.by_coord.values() if r['flip'])}\n")
    total_flip = 0
    for spec in SPECS:
        if only is not None and spec["path"] not in only:
            continue
        if spec.get("regenerated"):
            print(f"  - {spec['path']:42s} (skipped: regenerated by polarity-aware generator)")
            continue
        res = migrate_file(spec, dry)
        if res is None:
            print(f"  - {spec['path']:42s} (absent)")
            continue
        total_flip += res["flipped"]
        print(f"  - {res['path']:42s} matched={res['matched']:4d} "
              f"flipped={res['flipped']:4d} unmatched={res['unmatched']:4d}")
    print(f"\nTotal per-file flips applied: {total_flip}")
    print("\nRegenerate (pool loci):")
    for f, how in REGENERATE:
        print(f"  data/{f:40s} <- {how}")
    if not dry and only is None:
        # Only the full-dataset migration drops the marker; --only is a scoped,
        # idempotent-by-construction polarization of a freshly-produced raw file.
        with open(MARKER, "w") as fh:
            fh.write(pol_hash + "\n")
        print(f"\nWrote marker {MARKER}")
    elif dry:
        print("\n(dry-run; re-run with --apply to write)")


if __name__ == "__main__":
    main()
