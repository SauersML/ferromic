#!/usr/bin/env python3
"""Diploid assembly-resolved polarization from the UCSC 16-way T2T-ape Cactus alignment.

The 2025 Complete Ape Genomes SYRI/PAV inversion calls (data/apes2025_t2t_polarity.tsv)
cover only loci where a caller emitted an INV record, leaving ~212 human inversions with
no ape assembly evidence. The UCSC CGL 16-way diploid Cactus alignment
(16-t2t-apes-2023v2) instead provides GENOME-WIDE hg38-referenced reciprocal chains for
all 12 ape haplotypes (two per species: chimp, bonobo, gorilla, Sumatran + Bornean
orangutan, siamang), so every locus with orthologous sequence in any ape gets a call.

Method: the flank-relative BACKBONE-BRIDGING test (stats/polarize_v2_experiment.py) per
haplotype -- a locus is inverted in hg38 relative to an ape iff the syntenic chain that
spans both flanks breaks at the inversion and a reverse-strand chain fills the interior
(assembly-orientation-invariant). Because the alignment is DIPLOID we read each species'
two haplotypes independently:
  both inverted    -> species HOM inverted
  both collinear   -> species HOM direct
  one of each      -> species HET  (within-species polymorphism / recurrent toggling)

An ape inverted relative to hg38 means the ancestral (ape) orientation is the opposite of
the hg38 reference -> reference is DERIVED -> flip_ref_polarity = 1. Calls are rooted by
depth-weighted parsimony (siamang deepest), so the deep outgroups that retain the
ancestral state when shallow apes have toggled (17q21.31 / 8p23.1) win. n_het>=2 species
=> recurrent.

Emits data/cactus_ape_polarity.tsv. Chains: hg38_vs_<acc>.chain.gz from
https://cgl.gi.ucsc.edu/data/cactus/t2t-apes/16-t2t-apes-2023v2/chains/ (pass --chain-dir).

VALIDATION / LIMITATION (2026-06-23): run over all 12 haplotypes, this caller covers 336
loci -- but its calls agree with the Strand-seq gold standard only ~50% (even for
high-confidence, all-haplotype-unanimous calls), i.e. no better than chance. The cause is
the documented failure mode of PRECOMPUTED chains in segmental duplications: a chain can be
forced onto one paralogous copy across an inversion junction, fabricating or erasing the
orientation flip exactly at the SD-flanked loci where human inversions live. The output is
therefore NOT integrated into the source of truth. To make this method gold, replace the
precomputed chains with RAW local realignment of breakpoint-flanking windows
(`minimap2 -x asm20 -c --cs --secondary=yes -N 50`, keeping secondary alignments so SD
copies are not collapsed) against the ape assembly FASTAs -- which needs the ~32 GB
assemblies + an aligner on real compute (MSI/acn116 or a per-assembly CI job). The
published consortium SV callers (SYRI/PAV; data/apes2025_t2t_polarity.tsv, 88% concordant
with Strand-seq) already do synteny-block-aware calling and are the reliable assembly tier.
"""
from __future__ import annotations
import argparse, csv, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from polarize_v2_experiment import (  # noqa: E402
    build_intervals, dominant_by_interval, call_relative, load_catalog)

REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")

# 12 ape haplotypes in the 16-way diploid HAL. depth = phylogenetic distance from human
# (deeper roots the call better); two accessions per species = diploid genotype.
HAPS = [
    ("PTR", "chimpanzee", 2, "GCA_028858775.2"), ("PTR", "chimpanzee", 2, "GCA_028858805.2"),
    ("PPA", "bonobo",     2, "GCA_028858825.2"), ("PPA", "bonobo",     2, "GCA_028858845.2"),
    ("GGO", "gorilla",    3, "GCA_028885475.2"), ("GGO", "gorilla",    3, "GCA_028885495.2"),
    ("PAB", "orangutanS", 4, "GCA_028885655.2"), ("PAB", "orangutanS", 4, "GCA_028885685.2"),
    ("PPY", "orangutanB", 4, "GCA_028885625.2"), ("PPY", "orangutanB", 4, "GCA_028885525.2"),
    ("SSY", "siamang",    5, "GCA_028878055.2"), ("SSY", "siamang",    5, "GCA_028878085.2"),
]
DEEP = {"PAB", "PPY", "SSY"}   # deep outgroups (orangutans + siamang)


def per_hap_orient(chain_path, intervals):
    """slot dict -> per-locus orientation (collinear/inverted/None) for one haplotype."""
    raw = dominant_by_interval(chain_path, intervals)
    by_locus = defaultdict(dict)
    for (key, tag), chains in raw.items():
        by_locus[key][tag] = chains
    out = {}
    for key, slots in by_locus.items():
        orient, conf, _ = call_relative(slots.get("interior"),
                                        slots.get("lflank"), slots.get("rflank"))
        out[key] = (orient, conf)
    return out


def species_state(hap_orients):
    """Two haplotype (orient,conf) tuples -> 'HOM_inv' / 'HOM_dir' / 'HET' / None."""
    calls = [o for (o, c) in hap_orients if o in ("collinear", "inverted")]
    if not calls:
        return None
    inv = sum(o == "inverted" for o in calls)
    dirr = sum(o == "collinear" for o in calls)
    if inv and dirr:
        return "HET"
    return "HOM_inv" if inv else "HOM_dir"


def consensus(states):
    """species_code -> state. Depth-weighted parsimony over the ape tree."""
    depth = {c: d for (c, _, d, _) in HAPS}
    votes = {c: s for c, s in states.items() if s}
    if not votes:
        return dict(flip="", conf="none", n_sp=0, n_inv=0, n_dir=0, n_het=0,
                    recur=False, detail="no_ape_ortholog")
    n_het = sum(v == "HET" for v in votes.values())
    n_inv = sum(v == "HOM_inv" for v in votes.values())
    n_dir = sum(v == "HOM_dir" for v in votes.values())
    # recurrent: polymorphic within >=2 ape species
    recur = n_het >= 2
    w_inv = sum(depth[c] for c, v in votes.items() if v == "HOM_inv")
    w_dir = sum(depth[c] for c, v in votes.items() if v == "HOM_dir")
    deep_inv = sum(1 for c, v in votes.items() if c in DEEP and v == "HOM_inv")
    deep_dir = sum(1 for c, v in votes.items() if c in DEEP and v == "HOM_dir")
    if w_inv == w_dir:
        # tie -> defer to the deepest informative species
        deepest = max((c for c, v in votes.items() if v in ("HOM_inv", "HOM_dir")),
                      key=lambda c: depth[c], default=None)
        flip = 1 if deepest and votes[deepest] == "HOM_inv" else 0
    else:
        flip = 1 if w_inv > w_dir else 0
    n_hom = n_inv + n_dir
    deep_concord = deep_inv if flip else deep_dir
    if n_hom >= 3 and deep_concord >= 2 and n_het == 0:
        conf = "high"
    elif n_hom >= 2 and deep_concord >= 1:
        conf = "moderate"
    elif n_hom >= 1:
        conf = "low"
    else:
        conf = "recurrent"
    detail = ";".join(f"{c}:{v}" for c, v in sorted(votes.items()))
    return dict(flip=flip, conf=("recurrent" if recur else conf), n_sp=len(votes),
                n_inv=n_inv, n_dir=n_dir, n_het=n_het, recur=recur, detail=detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", required=True, help="dir of hg38_vs_<acc>.chain.gz")
    ap.add_argument("--margin", type=int, default=10000)
    ap.add_argument("--flank", type=int, default=100000)
    ap.add_argument("--out", default=os.path.join(DATA, "cactus_ape_polarity.tsv"))
    args = ap.parse_args()

    cat = load_catalog()
    intervals = build_intervals(cat, args.margin, args.flank, 0.10, 20000)
    print(f"loci: {len(cat)}; scanning 12 ape haplotypes ...")

    # per species -> {locus -> [hap1 (orient,conf), hap2 (orient,conf)]}
    sp_hap = defaultdict(lambda: defaultdict(list))
    for (code, name, depth, acc) in HAPS:
        path = os.path.join(args.chain_dir, f"hg38_vs_{acc}.chain.gz")
        if not os.path.exists(path):
            print(f"  MISSING {path}", file=sys.stderr); continue
        orient = per_hap_orient(path, intervals)
        for key, oc in orient.items():
            sp_hap[code][key].append(oc)
        ncall = sum(1 for oc in orient.values() if oc[0] in ("collinear", "inverted"))
        print(f"  {acc} ({code}): {ncall} loci called")

    rows = []
    for key, rec in cat.items():
        states = {code: species_state(sp_hap[code].get(key, [])) for code in {h[0] for h in HAPS}}
        cs = consensus(states)
        rows.append({"inv_id": key, "chrom": rec["chrom"], "start": rec["start"],
                     "end": rec["end"], "cactus_flip": cs["flip"], "confidence": cs["conf"],
                     "n_species": cs["n_sp"], "n_inv": cs["n_inv"], "n_dir": cs["n_dir"],
                     "n_het": cs["n_het"], "recurrent": int(cs["recur"]), "apes": cs["detail"]})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    called = [r for r in rows if r["confidence"] not in ("none",)]
    print(f"wrote {args.out} ({len(rows)} loci; {len(called)} with ape orthology)")
    print("confidence:", dict(Counter(r["confidence"] for r in rows)))
    print("flip=1:", sum(1 for r in rows if r["cactus_flip"] == 1))


if __name__ == "__main__":
    main()
