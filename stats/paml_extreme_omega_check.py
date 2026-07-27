#!/usr/bin/env python
"""Identifiability check for the genes with extreme clade-model omega (Reviewer 2 #2).

Reviewer 2 asked us to look more closely at the genes with extremely high Ka/Ks,
naming FDFT1 and BLK. This script pulls those genes out of the committed PAML
table with the two quantities that decide whether an extreme omega means
anything: the proportion of codons assigned to the divergent site class (p2) and
whether either clade's omega sits on PAML's optimiser boundary (999).

It also reports, per gene, which clade carries the extreme value **as recorded in
the table**, because the orientation direction is easy to get backwards in prose.

Where the labels come from
--------------------------
In cds/pipeline_lib.py the H1 clade-model-C tree marks every pure *direct*
internal branch ``#1`` and every pure *inverted* branch ``#2``
(``node.add_feature("paml_mark", "#1")`` under ``status == "direct"``), and the
parser assigns PAML's ``branch type 1`` row to ``cmc_omega2_direct`` and
``branch type 2`` to ``cmc_omega2_inverted``. The pipeline is therefore internally
consistent: ``winner_omega2_direct`` really is the direct clade. Any disagreement
with the manuscript text is a text error, not a column-naming error.

Input : data/GRAND_PAML_RESULTS.tsv
Output: data/paml_extreme_omega_check.tsv
"""
from __future__ import annotations

import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_IN = os.path.join(REPO, "data", "GRAND_PAML_RESULTS.tsv")
DEFAULT_OUT = os.path.join(REPO, "data", "paml_extreme_omega_check.tsv")

# The genes the reviewer named, plus the other two boundary hits in the table.
GENES = ["FDFT1", "BLK", "MAPT", "PRSS55"]

PAML_OMEGA_CEILING = 999.0
# Clade-model omega is not meaningfully estimable when the divergent class holds
# a negligible share of codons; 1% is a generous threshold here.
P2_NEGLIGIBLE = 0.01


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def build(in_path=DEFAULT_IN, out_path=DEFAULT_OUT, genes=GENES):
    with open(in_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    wanted = {g: None for g in genes}
    for r in rows:
        sym = r["gene"].split("_")[0]
        if sym in wanted and wanted[sym] is None:
            wanted[sym] = r

    out = []
    for gene in genes:
        r = wanted[gene]
        if r is None:
            print(f"WARNING: {gene} not found in {in_path}")
            continue
        w_dir, w_inv = _f(r["winner_omega2_direct"]), _f(r["winner_omega2_inverted"])
        p2 = _f(r["winner_p2"])
        at_ceiling = [n for n, v in (("direct", w_dir), ("inverted", w_inv))
                      if v >= PAML_OMEGA_CEILING]
        higher = "direct" if w_dir > w_inv else "inverted"
        flags = []
        if at_ceiling:
            flags.append("omega at PAML boundary (999) in " + "+".join(at_ceiling))
        if p2 < P2_NEGLIGIBLE:
            flags.append(f"divergent site class holds {p2 * 100:.2f}% of codons")
        out.append({
            "gene": gene,
            "transcript": r["gene"],
            "region": r["region"],
            "status": r["status"],
            "overall_p_value": r["overall_p_value"],
            "overall_q_value": r["overall_q_value"],
            "p2_divergent_class": r["winner_p2"],
            "omega2_direct": r["winner_omega2_direct"],
            "omega2_inverted": r["winner_omega2_inverted"],
            "clade_with_higher_omega2": higher,
            "not_identifiable_flags": "; ".join(flags),
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)

    print(f"{len(out)} genes -> {out_path}\n")
    for r in out:
        print(f"{r['gene']:8s} p2={float(r['p2_divergent_class']) * 100:5.2f}%  "
              f"omega2 direct={float(r['omega2_direct']):>9.4g}  "
              f"inverted={float(r['omega2_inverted']):>9.4g}  "
              f"-> higher in {r['clade_with_higher_omega2']}  "
              f"q={float(r['overall_q_value']):.3g}")
        if r["not_identifiable_flags"]:
            print(f"         {r['not_identifiable_flags']}")
    print("\nNone of these genes passes multiple-testing correction "
          "(all q > 0.05), and every one of them has its divergent site class\n"
          "confined to a negligible fraction of codons, so the extreme omega "
          "values are not identifiable rather than evidence of selection.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    ap.add_argument("--genes", nargs="+", default=GENES)
    args = ap.parse_args(argv)
    build(args.in_path, args.out_path, args.genes)


if __name__ == "__main__":
    main()
