#!/usr/bin/env python
"""Per-gene haplotype counts by orientation (k_dir / k_inv) for the CDS analyses.

Every per-gene coding result -- CDS conservation, the clade-model omegas -- rests
on how many haplotypes each orientation actually contributes at that locus. That
count is the first thing to check before reading anything into a per-gene value,
and it is not currently written down anywhere: it has to be recovered from
``n_sequences`` in ``cds_identical_proportions.tsv``.

This emits it directly, per gene and per inversion, with the recurrence class, so
k_inv is checkable for any gene in the coding tables (e.g. the 17q21.31 inversion
behind MAPT contributes k_inv = 10, whereas most single-event loci with an
inverted haplotype at all contribute only 2 or 3).

Input : data/cds_identical_proportions.tsv
Output: data/cds_haplotype_counts.tsv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_IN = os.path.join(REPO, "data", "cds_identical_proportions.tsv")
DEFAULT_OUT = os.path.join(REPO, "data", "cds_haplotype_counts.tsv")

CLASS = {"0": "single", "1": "recurrent"}


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def build(in_path=DEFAULT_IN, out_path=DEFAULT_OUT):
    with open(in_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    by_gene = defaultdict(dict)
    for r in rows:
        key = (r["gene_name"], r["transcript_id"], r["chr"],
               r["inv_start"], r["inv_end"], r["consensus"])
        by_gene[key][r["phy_group"]] = r

    out = []
    for (gene, tx, chrom, s, e, cons), groups in by_gene.items():
        dir_row, inv_row = groups.get("0"), groups.get("1")
        k_dir, k_inv = _i(dir_row and dir_row["n_sequences"]), _i(inv_row and inv_row["n_sequences"])
        out.append({
            "gene_name": gene,
            "transcript_id": tx,
            "inversion": f"{chrom}:{s}-{e}",
            "recurrence": CLASS.get(str(cons).strip(), "NA"),
            "k_dir": k_dir,
            "k_inv": k_inv,
            "both_orientations": "yes" if (k_dir >= 1 and k_inv >= 1) else "no",
            "inv_underpowered_lt4": "yes" if k_inv < 4 else "no",
        })

    out.sort(key=lambda r: (r["recurrence"], r["k_inv"], r["inversion"], r["gene_name"]))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)

    print(f"{len(out)} gene x inversion entries -> {out_path}\n")
    for cls in ("single", "recurrent", "NA"):
        sel = [r for r in out if r["recurrence"] == cls]
        if not sel:
            continue
        both = [r for r in sel if r["both_orientations"] == "yes"]
        weak = [r for r in both if r["inv_underpowered_lt4"] == "yes"]
        print(f"{cls:10s} {len(sel):4d} entries; {len(both):4d} with both "
              f"orientations; {len(weak):4d} of those have k_inv < 4")
        seen = {}
        for r in both:
            seen.setdefault(r["inversion"], r["k_inv"])
        for invn, k in sorted(seen.items(), key=lambda kv: kv[1]):
            n = sum(1 for r in both if r["inversion"] == invn)
            print(f"           {invn:28s} k_inv={k:3d}  ({n} genes)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    build(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
