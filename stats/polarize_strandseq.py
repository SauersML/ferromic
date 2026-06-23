#!/usr/bin/env python3
"""Gold-standard ancestral polarization from Strand-seq great-ape inversion genotypes.

Strand-seq directly measures inversion ORIENTATION per great-ape genome relative to
GRCh38 (Porubsky et al. 2020, Nat Genet), so it is immune to the segmental-duplication
paralog confound that defeats chain synteny. We use the published per-species genotypes
(data/strandseq_ape_inversions.tsv) and the deep-outgroup (orangutan/macaque/mouse)
ancestral classifications (data/strandseq_deepoutgroup_ancestral.tsv) to infer, for each
human inversion, whether the GRCh38 reference orientation is ancestral or derived, by
parsimony over the ape tree.

Convention: an ape "inv" call = that ape is inverted relative to GRCh38. If the deeper
outgroups carry the inverted state, GRCh38 (direct) is DERIVED -> flip_ref_polarity=1.
A species that is HET (polymorphic) is evidence of recurrence/toggling at that locus.
"""
from __future__ import annotations
import csv, os, sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
# phylogenetic depth from human (chimp/bonobo are equidistant; deeper = better root)
DEPTH = {"chimpanzee": 2, "bonobo": 2, "gorilla": 3, "orangutan": 4}


def norm(c):
    c = str(c).strip()
    return c if c.startswith("chr") else "chr" + c


def ov(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def load_strandseq():
    sp = defaultdict(list)   # species -> [(chrom,s,e,gen)]
    with open(os.path.join(DATA, "strandseq_ape_inversions.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if "INV" not in (r["svclass"] or "").upper():
                continue
            try:
                sp[r["species"]].append((norm(r["chrom"]), int(r["start"]), int(r["end"]), r["gen"]))
            except ValueError:
                continue
    deep = []
    p = os.path.join(DATA, "strandseq_deepoutgroup_ancestral.tsv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p), delimiter="\t"):
            try:
                deep.append((norm(r["chrom"]), int(r["start"]), int(r["end"]),
                             r["classification"], r["consistent"]))
            except ValueError:
                continue
    return sp, deep


def deep_call(deep, chrom, s, e):
    """Return (flip, source) from the deep-outgroup (Or/Ma/Mo) table if it covers this
    locus consistently, else None."""
    for (c, ds, de, cls, cons) in deep:
        if c != chrom:
            continue
        if ov(s, e, ds, de) <= 0.3 * min(e - s, de - ds):
            continue
        if str(cons) != "True":
            return ("ambiguous", f"deepOG[{cls}]")
        if "Rev" in cls and "Dir" not in cls:
            return (1, f"deepOG[{cls}]")          # inverted ancestral -> ref derived
        if "Dir" in cls and "Rev" not in cls:
            return (0, f"deepOG[{cls}]")           # direct ancestral
        return ("ambiguous", f"deepOG[{cls}]")
    return None


def call_locus(sp, deep, chrom, s, e):
    # 1) deep-outgroup (orangutan+macaque+mouse) is the strongest signal
    d = deep_call(deep, chrom, s, e)
    # 2) per-ape Strand-seq state
    state = {}
    for species, calls in sp.items():
        hit = [g for (c, cs, ce, g) in calls
               if c == chrom and ov(s, e, cs, ce) > 0.3 * min(e - s, ce - cs)]
        if hit:
            state[species] = "HET" if any(h == "HET" for h in hit) else "HOM"
        else:
            state[species] = "dir"   # tested-but-not-inverted (matches GRCh38) — see caveat
    n_inv = sum(1 for v in state.values() if v in ("HOM", "HET"))
    n_het = sum(1 for v in state.values() if v == "HET")

    if d and d[0] in (0, 1):
        flip, src, conf = d[0], d[1], "high"
        recur = "recurrent" if n_het >= 2 else ""
        return dict(flip=flip, conf=conf, source=src, n_inv=n_inv, n_het=n_het,
                    recur=recur, state=state)
    if d and d[0] == "ambiguous":
        return dict(flip="", conf="recurrent", source=d[1], n_inv=n_inv, n_het=n_het,
                    recur="recurrent", state=state)

    if n_inv == 0:
        return None   # no ape evidence at this locus

    # 3) depth-weighted parsimony over the great apes
    w_inv = sum(DEPTH[s_] * (1.0 if v == "HOM" else 0.5) for s_, v in state.items() if v in ("HOM", "HET"))
    w_dir = sum(DEPTH[s_] for s_, v in state.items() if v == "dir")
    # HET in >=2 lineages => the inversion is segregating in apes too => recurrent/toggling
    if n_het >= 2:
        return dict(flip="", conf="recurrent", source="strandseq_parsimony",
                    n_inv=n_inv, n_het=n_het, recur="recurrent", state=state)
    flip = 1 if w_inv > w_dir else 0
    # confidence: high if the deepest ape (orangutan) is HOM and agrees, and margin is clear
    orang = state.get("orangutan")
    margin = abs(w_inv - w_dir) / (w_inv + w_dir) if (w_inv + w_dir) else 0
    if orang in ("HOM",) and flip == 1 and margin >= 0.5:
        conf = "high"
    elif margin >= 0.5:
        conf = "moderate"
    else:
        conf = "low"
    return dict(flip=flip, conf=conf, source="strandseq_parsimony",
                n_inv=n_inv, n_het=n_het, recur="", state=state)


def main():
    sp, deep = load_strandseq()
    print(f"Strand-seq species: { {k: len(v) for k, v in sp.items()} }; deep-OG regions: {len(deep)}")
    ours = list(csv.DictReader(open(os.path.join(DATA, "inversion_polarity.tsv")), delimiter="\t"))
    out = []
    for o in ours:
        chrom, s, e = norm(o["chrom"]), int(o["start"]), int(o["end"])
        c = call_locus(sp, deep, chrom, s, e)
        if c is None:
            continue
        out.append({"inv_id": o["inv_id"], "chrom": chrom, "start": s, "end": e,
                    "strandseq_flip": c["flip"], "confidence": c["conf"],
                    "source": c["source"], "n_apes_inverted": c["n_inv"],
                    "n_apes_het": c["n_het"], "recurrence": c["recur"],
                    "apes": ";".join(f"{k}:{v}" for k, v in sorted(c["state"].items()) if v != "dir"),
                    "v2_flip": o["flip_ref_polarity"], "v2_conf": o["confidence"]})
    with open(os.path.join(DATA, "strandseq_polarity.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(out)
    from collections import Counter
    print(f"Strand-seq covers {len(out)}/{len(ours)} loci")
    print("confidence:", dict(Counter(r["confidence"] for r in out)))
    # agreement with v2 where both give a binary flip
    bothbin = [r for r in out if r["strandseq_flip"] in (0, 1) and r["v2_flip"] in ("0", "1")]
    ag = sum(1 for r in bothbin if str(r["strandseq_flip"]) == r["v2_flip"])
    print(f"v2 vs Strand-seq (gold) agreement: {ag}/{len(bothbin)} = {ag/len(bothbin)*100:.0f}%")
    hi = [r for r in bothbin if r["v2_conf"] == "high"]
    aghi=sum(1 for r in hi if str(r["strandseq_flip"])==r["v2_flip"])
    print(f"  among v2 HIGH-confidence: {aghi}/{len(hi)} = {aghi/len(hi)*100:.0f}% agree with gold")
    print("wrote data/strandseq_polarity.tsv")


if __name__ == "__main__":
    main()
