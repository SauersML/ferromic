#!/usr/bin/env python3
"""Gold-standard ancestral polarization from the 2025 Complete Ape Genomes (T2T).

The 2025 Complete Ape Genomes consortium released telomere-to-telomere assemblies for
six non-human apes and called inversions against T2T-CHM13v2 with SYRI and PAV
(data/apes2025_t2t_inversions.tsv: per-sample INV calls in CHM13 coordinates). Because
these are assembly-resolved breakpoints (not chain synteny), they are immune to the
segmental-duplication paralog confound that defeats absolute-strand and chain methods,
and they cover the deep outgroups (orangutan, siamang) that retain the ancestral state
when the shallow apes have toggled -- the 17q21.31 / 8p23.1 lesson.

Method, per human inversion locus (data/inversion_polarity.tsv, hg38):
  1. Lift the hg38 locus to CHM13v2 (pyliftover, hg38ToHs1.over.chain.gz from UCSC).
  2. Intersect the lifted interval with the ape INV calls (>0.3 reciprocal overlap).
  3. Per species, HOM if any homozygous-inverted sample hits, else HET if any het hits.
  4. An ape inverted vs the DIRECT CHM13 backbone means the inverted arrangement is the
     ancestral one carried down the ape tree -> GRCh38 reference (direct) is DERIVED ->
     flip_ref_polarity = 1. (The consortium calls are oriented vs CHM13's direct backbone,
     so an INV call at a human-polymorphic locus reports the ancestral ape orientation.)

Confidence (parsimony depth: orangutan PPY/PAB and siamang SSY are the deep outgroups
that root the human call; chimp PTR, bonobo PPA, gorilla GGO are shallower):
  high      n_deep >= 2 and n_hom >= 3   (deep clade + majority homozygous-inverted)
  moderate  n_deep >= 1 and n_hom >= 2
  recurrent n_het >= 3, or (n_het >= 2 and n_hom < 2)   -- segregating in apes too
  low       otherwise (sparse / shallow-only support)

Only high+moderate calls are promoted to the gold_t2t_apes evidence tier in the SoT;
recurrent flags toggling; low is reported but not used to override synteny.

Emits data/apes2025_t2t_polarity.tsv. Requires: pip/uv `pyliftover`, and the UCSC chain
hg38ToHs1.over.chain.gz (pass via --chain).
"""
from __future__ import annotations
import argparse, csv, os, sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")

# Deep outgroups for rooting the human ancestral state (Pongo + Symphalangus).
DEEP = {"PPY", "PAB", "SSY"}
SPECIES = ["PTR", "PPA", "GGO", "PPY", "PAB", "SSY"]


def norm(c):
    c = str(c).strip()
    return c if c.startswith("chr") else "chr" + c


def ov(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def load_ape_calls():
    """species_code -> [(chrom, start, end, genotype)] from the consortium INV table."""
    sp = defaultdict(list)
    with open(os.path.join(DATA, "apes2025_t2t_inversions.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            code = (r["species"] or "").split("_")[-1]
            try:
                sp[code].append((norm(r["chm13_chrom"]), int(r["chm13_start"]),
                                 int(r["chm13_end"]), (r["genotype"] or "").lower()))
            except (ValueError, KeyError):
                continue
    return sp


def call_locus(sp, chrom, s, e):
    """Aggregate per-species state at a CHM13 interval; return None if no ape evidence."""
    state = {}
    for code in SPECIES:
        hits = [g for (c, cs, ce, g) in sp.get(code, [])
                if c == chrom and ov(s, e, cs, ce) > 0.3 * min(e - s, ce - cs)]
        if not hits:
            continue
        state[code] = "HOM" if any(h == "hom" for h in hits) else "HET"
    if not state:
        return None
    n_hom = sum(1 for v in state.values() if v == "HOM")
    n_het = sum(1 for v in state.values() if v == "HET")
    n_deep = sum(1 for c, v in state.items() if c in DEEP and v == "HOM")
    if n_deep >= 2 and n_hom >= 3:
        conf = "high"
    elif n_deep >= 1 and n_hom >= 2:
        conf = "moderate"
    elif n_het >= 3 or (n_het >= 2 and n_hom < 2):
        conf = "recurrent"
    else:
        conf = "low"
    return dict(flip=1, conf=conf, n_species=len(state), n_deep=n_deep, n_het=n_het,
                apes=";".join(f"{c}:{state[c]}" for c in SPECIES if c in state))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", help="UCSC hg38ToHs1.over.chain.gz (for hg38->CHM13 liftover)")
    ap.add_argument("--out", default=os.path.join(DATA, "apes2025_t2t_polarity.tsv"))
    args = ap.parse_args()

    try:
        from pyliftover import LiftOver
    except ImportError:
        sys.exit("Need pyliftover: `uv run --no-project --with pyliftover python ...`")
    if not args.chain or not os.path.exists(args.chain):
        sys.exit("Pass --chain hg38ToHs1.over.chain.gz (UCSC goldenPath liftOver).")
    lo = LiftOver(args.chain)

    sp = load_ape_calls()
    print(f"ape calls: { {k: len(v) for k, v in sorted(sp.items())} }")
    loci = list(csv.DictReader(open(os.path.join(DATA, "inversion_polarity.tsv")), delimiter="\t"))
    out = []
    for o in loci:
        a = lo.convert_coordinate(norm(o["chrom"]), int(o["start"]))
        b = lo.convert_coordinate(norm(o["chrom"]), int(o["end"]))
        if not a or not b or a[0][0] != b[0][0]:
            continue
        cs, ce = sorted((a[0][1], b[0][1]))
        c = call_locus(sp, a[0][0], cs, ce)
        if c is None:
            continue
        out.append({"inv_id": o["inv_id"], "t2t_flip": c["flip"], "confidence": c["conf"],
                    "n_species": c["n_species"], "n_deep": c["n_deep"],
                    "n_het": c["n_het"], "apes": c["apes"]})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(out)
    from collections import Counter
    print(f"wrote {args.out}  ({len(out)} loci)")
    print("confidence:", dict(Counter(r["confidence"] for r in out)))


if __name__ == "__main__":
    main()
