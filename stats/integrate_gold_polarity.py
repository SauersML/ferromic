#!/usr/bin/env python3
"""Overlay gold-standard ancestral-orientation evidence onto the synteny base table.

The v2 synteny caller (stats/polarize_orientation_v2.py) produces a chain-synteny call
for every locus, but synteny is only ~53% accurate vs the Strand-seq gold standard and
cannot resolve deep-toggling loci (17q21.31, 8p23.1). This script layers the
assembly/Strand-seq gold evidence on top, by a fixed precedence, to produce the final
data/inversion_polarity.tsv source of truth.

Gold inputs (committed reference data, paralog-immune, NOT chain synteny):
  - data/apes2025_t2t_polarity.tsv   2025 Complete Ape Genomes T2T (6 species, SYRI/PAV)
  - data/strandseq_polarity.tsv      Porubsky 2020 Strand-seq + deep-outgroup (Or/Ma/Mo)

Precedence per locus (highest wins; lower tiers never override a higher call):
  1. gold_t2t_apes      t2t confidence in {high, moderate}      -> flip/conf := T2T
  2. gold_deepOG        strandseq source deepOG  & conf high     -> flip/conf := Strand-seq
  3. gold_strandseq     strandseq parsimony & conf in {high,mod} -> flip/conf := Strand-seq
  4. recurrent_strandseq strandseq confidence == recurrent       -> keep synteny, flag
  5. recurrent_t2t_apes  t2t confidence == recurrent             -> keep synteny, flag
  6. synteny            none of the above                        -> keep synteny base

IDEMPOTENT: the gold tiers (1-3) deterministically overwrite flip/confidence, and the
recurrent/synteny tiers preserve the base, so re-running on an already-integrated table
reproduces it byte-for-byte. CI (ancestral_orientation.yml) runs the v2 caller then this
overlay, so regeneration never reverts the gold integration.
"""
from __future__ import annotations
import argparse, csv, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")


def load(p, key="inv_id"):
    if not os.path.exists(p):
        return {}
    return {r[key]: r for r in csv.DictReader(open(p), delimiter="\t")}


def tier_for(t2t, ss):
    """Return (tier, flip_override_or_None, conf_override_or_None)."""
    tconf = (t2t or {}).get("confidence", "")
    sconf = (ss or {}).get("confidence", "")
    ssrc = (ss or {}).get("source", "")
    if tconf in ("high", "moderate"):
        return "gold_t2t_apes", t2t.get("t2t_flip"), tconf
    if ss and sconf == "high" and ssrc.startswith("deepOG"):
        return "gold_deepOG", ss.get("strandseq_flip"), "high"
    if ss and sconf in ("high", "moderate") and not ssrc.startswith("deepOG"):
        return "gold_strandseq", ss.get("strandseq_flip"), sconf
    if ss and sconf == "recurrent":
        return "recurrent_strandseq", None, None
    if tconf == "recurrent":
        return "recurrent_t2t_apes", None, None
    return "synteny", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(DATA, "inversion_polarity.tsv"),
                    help="v2 synteny base table (modified in place by default)")
    ap.add_argument("--out", default=os.path.join(DATA, "inversion_polarity.tsv"))
    args = ap.parse_args()

    t2t = load(os.path.join(DATA, "apes2025_t2t_polarity.tsv"))
    ss = load(os.path.join(DATA, "strandseq_polarity.tsv"))
    rows = list(csv.DictReader(open(args.base), delimiter="\t"))
    cols = ["inv_id", "chrom", "start", "end", "orig_id", "recurrence_class", "sources",
            "ancestral_orientation", "derived_orientation", "flip_ref_polarity",
            "confidence", "n_outgroups_informative", "n_support_ref_ancestral",
            "n_support_ref_derived", "outgroup_discordant", "orient_chimp",
            "orient_gorilla", "orient_orangutan", "orient_macaque", "evidence",
            "inv_af_ref", "derived_af", "minor_is_derived", "strandseq_flip",
            "strandseq_confidence", "strandseq_source", "evidence_tier",
            "t2t_ape_flip", "t2t_ape_confidence"]

    from collections import Counter
    tiers = Counter()
    for r in rows:
        iid = r["inv_id"]
        t, s = t2t.get(iid), ss.get(iid)
        tier, flip_ov, conf_ov = tier_for(t, s)
        tiers[tier] += 1
        # gold provenance columns (always reflect the underlying gold tables)
        r["strandseq_flip"] = s.get("strandseq_flip", "") if s else ""
        r["strandseq_confidence"] = s.get("confidence", "") if s else ""
        r["strandseq_source"] = s.get("source", "") if s else ""
        r["t2t_ape_flip"] = t.get("t2t_flip", "") if t else ""
        r["t2t_ape_confidence"] = t.get("confidence", "") if t else ""
        r["evidence_tier"] = tier
        if flip_ov is not None and str(flip_ov) != "":
            r["flip_ref_polarity"] = str(flip_ov)
            r["confidence"] = conf_ov
        # derive orientation + derived_af from the final flip
        flip = r["flip_ref_polarity"] == "1"
        r["ancestral_orientation"] = "inverted" if flip else "direct"
        r["derived_orientation"] = "direct" if flip else "inverted"
        af = r.get("inv_af_ref", "")
        try:
            afv = float(af)
            r["derived_af"] = repr(round((1 - afv) if flip else afv, 6))
            minor = "inverted" if afv < 0.5 else ("direct" if afv > 0.5 else "tie")
            r["minor_is_derived"] = (str(minor == r["derived_orientation"])
                                     if minor in ("direct", "inverted") else "NA")
        except (ValueError, TypeError):
            pass

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} loci)")
    print("evidence_tier:", dict(tiers))
    print("flip=1:", sum(r["flip_ref_polarity"] == "1" for r in rows))


if __name__ == "__main__":
    main()
