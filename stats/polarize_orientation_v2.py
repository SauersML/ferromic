#!/usr/bin/env python3
"""Production v2 derived-orientation caller: flank-relative, assembly-invariant.

Replaces the absolute-strand caller (root cause of ~65% spurious cross-outgroup
discordance: it read each outgroup's assembly orientation, not the inversion).

Method (per outgroup, per locus):
  Compare the inversion interior's orthologous chain orientation to the flanking
  syntenic BACKBONE chain (the chain spanning both flanks). The reference is in the
  derived (flipped) state w.r.t. that outgroup iff the backbone breaks at the
  inversion and a reverse-strand chain fills the interior. This is invariant to the
  outgroup's assembly orientation. Uses reciprocal-best (rbest) chains primarily
  (paralog/SD-filtered) with all-chain fallback for coverage.

Consensus: depth-weighted parsimony across chimp<gorilla<orangutan<macaque, so the
deeper outgroups (which retain the ancestral state when shallow apes have toggled --
the 17q21.31 / 8p23.1 lesson) root the call. EVERY locus with any orthology gets a
call; confidence is reported (never used to drop a locus).

Emits data/inversion_polarity.tsv (drop-in: identical columns to the previous table).
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from polarize_v2_experiment import call_relative, DEPTH, OUTGROUPS  # noqa: E402

REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
SP = [o["name"] for o in OUTGROUPS]

# Curated gold-standard ancestral calls from the literature, used ONLY for loci
# where the ancestral state is firmly established by independent evidence (SNP-
# haplotype divergence to outgroups, ancestral-state reconstruction) and where
# synteny cannot resolve it because the entire catarrhine clade has toggled. Keyed
# by (chrom, approx_start, approx_end) with +/- overlap matching; takes precedence
# over the synteny consensus and is marked high confidence with the source.
LITERATURE_OVERRIDES = [
    # 17q21.31 MAPT inversion: H2 (inverted) is ANCESTRAL; H1/reference is derived.
    # Zody et al. 2008 Nat Genet (evolutionary toggling); Steinberg et al. 2012 Nat Genet.
    {"chrom": "chr17", "start": 45_585_159, "end": 46_292_045, "ancestral": "inverted",
     "source": "Zody2008/Steinberg2012:H2_ancestral"},
    # 8p23.1 inversion: inverted orientation is ANCESTRAL (orangutan BAC + phylogeny).
    # Salm et al. 2012 Genome Res. (synteny already recovers this; override pins confidence.)
    {"chrom": "chr8", "start": 8_000_000, "end": 12_000_000, "ancestral": "inverted",
     "source": "Salm2012:inverted_ancestral"},
]


def literature_override(chrom, start, end):
    for o in LITERATURE_OVERRIDES:
        if o["chrom"] == chrom and end > o["start"] and start < o["end"]:
            # require substantial reciprocal overlap to avoid catching sub-loci
            ov = min(end, o["end"]) - max(start, o["start"])
            if ov > 0.5 * min(end - start, o["end"] - o["start"]):
                return o
    return None


def load_raw(tag, margin, flank):
    p = os.path.join(DATA, f"_polarity_raw_{tag}_m{margin}_f{flank}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def per_outgroup_call(rbest, allch, og, key):
    """rbest-primary, all-chain fallback bridging call for one outgroup at one locus."""
    for src in (rbest, allch):
        slots = src.get(og, {}).get(key, {})
        if not slots:
            continue
        orient, conf, det = call_relative(slots.get("interior"),
                                          slots.get("lflank"), slots.get("rflank"))
        if orient in ("collinear", "inverted"):
            return orient, conf, det
    return None, None, {}


def consensus(per):
    """per: og -> (orient, strength). Depth-weighted parsimony.
    Returns dict with ancestral/derived/flip/confidence/counts/evidence."""
    votes = [(DEPTH[og], o == "collinear", s == "strong")
             for og, (o, s) in per.items() if o in ("collinear", "inverted")]
    n_anc = sum(1 for d, a, s in votes if a)
    n_der = sum(1 for d, a, s in votes if not a)
    discord = n_anc > 0 and n_der > 0
    if not votes:
        return dict(anc="direct", der="inverted", flip=0, conf="assumed",
                    n_inf=0, n_anc=0, n_der=0, discord=0, evidence="no_ape_orthology")
    w = lambda strong: 1.0 if strong else 0.5
    w_anc = sum(d * w(s) for d, a, s in votes if a)
    w_der = sum(d * w(s) for d, a, s in votes if not a)
    ref_anc = (max(votes, key=lambda v: v[0])[1] if w_anc == w_der else w_anc > w_der)
    flip = not ref_anc
    n_inf = len(votes); n_agree = sum(1 for d, a, s in votes if a == ref_anc)
    frac = n_agree / n_inf
    strong_agree = any(s for d, a, s in votes if a == ref_anc)
    deepest_agrees = (max(votes, key=lambda v: v[0])[1] == ref_anc)
    if frac == 1.0 and n_inf >= 2 and strong_agree:
        conf = "high"
    elif frac >= 0.75 and strong_agree and deepest_agrees:
        conf = "high"
    elif frac >= 0.75 and deepest_agrees:
        conf = "moderate"
    elif strong_agree and deepest_agrees:
        conf = "moderate"
    else:
        conf = "low"
    tag = "congruent" if not discord else "discordant"
    ev = f"{tag}[{frac:.2f}]:" + ",".join(
        f"{og}{'+' if per[og][0]=='collinear' else '-'}" for og in SP
        if og in per and per[og][0] in ("collinear", "inverted"))
    return dict(anc="direct" if ref_anc else "inverted",
                der="inverted" if ref_anc else "direct",
                flip=int(flip), conf=conf, n_inf=n_inf, n_anc=n_anc, n_der=n_der,
                discord=int(discord), evidence=ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=int, default=10000)
    ap.add_argument("--flank", type=int, default=100000)
    ap.add_argument("--out", default=os.path.join(DATA, "inversion_polarity_v2.tsv"))
    args = ap.parse_args()

    rbest = load_raw("aln_rbest", args.margin, args.flank)
    allch = load_raw("aln", args.margin, args.flank)
    if not rbest and not allch:
        sys.exit("No raw caches found; run the scan first.")

    # metadata (coords, orig_id, recurrence, sources, inv_af_ref) from existing table
    meta = {r["inv_id"]: r for r in csv.DictReader(
        open(os.path.join(DATA, "inversion_polarity.tsv")), delimiter="\t")}

    orient_cols = [f"orient_{og}" for og in SP]
    cols = ["inv_id", "chrom", "start", "end", "orig_id", "recurrence_class", "sources",
            "ancestral_orientation", "derived_orientation", "flip_ref_polarity",
            "confidence", "n_outgroups_informative", "n_support_ref_ancestral",
            "n_support_ref_derived", "outgroup_discordant"] + orient_cols + [
            "evidence", "inv_af_ref", "derived_af", "minor_is_derived"]
    rows = []
    for key, m in meta.items():
        per = {}
        for og in SP:
            o, c, _ = per_outgroup_call(rbest, allch, og, key)
            per[og] = (o, c)
        cs = consensus(per)
        # Literature override for firmly-established, synteny-unresolvable loci.
        lo = literature_override(m["chrom"], int(m["start"]), int(m["end"]))
        if lo:
            anc = lo["ancestral"]; der = "inverted" if anc == "direct" else "direct"
            cs = {**cs, "anc": anc, "der": der, "flip": int(anc == "inverted"),
                  "conf": "high", "evidence": "literature:" + lo["source"]}
        af = m.get("inv_af_ref", "")
        try:
            afv = float(af); der_af = round((1 - afv) if cs["flip"] else afv, 6)
            minor = "inverted" if afv < 0.5 else ("direct" if afv > 0.5 else "tie")
            minor_is_der = (minor == cs["der"]) if minor in ("direct", "inverted") else "NA"
        except (ValueError, TypeError):
            der_af = "NA"; minor_is_der = "NA"
        rows.append({
            "inv_id": key, "chrom": m["chrom"], "start": m["start"], "end": m["end"],
            "orig_id": m.get("orig_id", ""), "recurrence_class": m.get("recurrence_class", "unknown"),
            "sources": m.get("sources", ""),
            "ancestral_orientation": cs["anc"], "derived_orientation": cs["der"],
            "flip_ref_polarity": cs["flip"], "confidence": cs["conf"],
            "n_outgroups_informative": cs["n_inf"], "n_support_ref_ancestral": cs["n_anc"],
            "n_support_ref_derived": cs["n_der"], "outgroup_discordant": cs["discord"],
            **{f"orient_{og}": (per[og][0] or "NA") for og in SP},
            "evidence": cs["evidence"], "inv_af_ref": af,
            "derived_af": der_af, "minor_is_derived": minor_is_der})

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"wrote {args.out}  ({len(rows)} loci)")
    print("confidence:", dict(Counter(r["confidence"] for r in rows)))
    print("flip=1:", sum(r["flip_ref_polarity"] for r in rows))


if __name__ == "__main__":
    main()
