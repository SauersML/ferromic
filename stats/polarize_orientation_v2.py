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
from polarize_v2_experiment import (  # noqa: E402
    call_relative, DEPTH, OUTGROUPS, build_intervals, dominant_by_interval, load_catalog)

REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
SP = [o["name"] for o in OUTGROUPS]

# NOTE: no per-locus literature overrides. Every call is produced by the method.
# Deep-toggling inversions whose entire catarrhine clade carries the derived
# arrangement (canonically 17q21.31) are reported by their synteny evidence; the
# ancestral state at those loci is only recoverable by an orthogonal SNP-divergence /
# ancestral-allele analysis (the documented next-step method), not by synteny.


def load_raw(tag, margin, flank):
    p = os.path.join(DATA, f"_polarity_raw_{tag}_m{margin}_f{flank}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def scan_chain_dir(aln_dir, margin, flank, trim_frac=0.10, trim_max=20000):
    """Self-contained scan: parse the chain files in aln_dir and return
    {outgroup -> {locus_key -> {interior/lflank/rflank: {chain_id:[strand,bp]}}}}.
    Used so CI can produce the table directly from downloaded chains (no cache)."""
    cat = load_catalog()
    intervals = build_intervals(cat, margin, flank, trim_frac, trim_max)
    raw = {}
    for og in OUTGROUPS:
        path = os.path.join(aln_dir, og["chain"])
        if not os.path.exists(path):
            continue
        dom = dominant_by_interval(path, intervals)
        d = raw.setdefault(og["name"], {})
        for (key, tagn), chains in dom.items():
            d.setdefault(key, {})[tagn] = chains
    return raw


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
    ap.add_argument("--aln-rbest", help="dir of reciprocal-best chains (scan directly)")
    ap.add_argument("--aln-all", help="dir of all-chains (scan directly, fallback coverage)")
    ap.add_argument("--out", default=os.path.join(DATA, "inversion_polarity_v2.tsv"))
    args = ap.parse_args()

    if args.aln_rbest or args.aln_all:
        rbest = scan_chain_dir(args.aln_rbest, args.margin, args.flank) if args.aln_rbest else {}
        allch = scan_chain_dir(args.aln_all, args.margin, args.flank) if args.aln_all else {}
    else:
        rbest = load_raw("aln_rbest", args.margin, args.flank)
        allch = load_raw("aln", args.margin, args.flank)
    if not rbest and not allch:
        sys.exit("No chains: pass --aln-rbest/--aln-all or provide cached raw scans.")

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
