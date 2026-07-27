#!/usr/bin/env python
"""Reconcile the two imputation-quality counts that appear in the write-up.

`replicate_manuscript_statistics.txt` reports "158 models evaluated, 21 with
r2 > 0.3 and BH p < 0.05". The manuscript reports "12 out of the 93 inversions had
reliable imputation performance (r2 > 0.5 and BH p < 0.05)". Those look
contradictory but are not: they use different thresholds *over different
denominators*, and both reproduce.

  * 158 is every model that was fit.
  * 93 is the consensus-classified analysis set, i.e. the subset of models whose
    locus survives the two-method recurrence agreement filter. Only 75 of the 158
    models correspond to a locus in that set at all.

This emits the full threshold x denominator grid so whichever pair is quoted can
be quoted with its denominator attached.

One discrepancy it surfaces: within the 93-locus set, 12 loci have r2 > 0.5, but
only 11 of those also have BH p < 0.05. The manuscript's sentence attaches both
conditions to the count of 12, which is the count for the r2 threshold alone.

Output: data/imputation_threshold_summary.tsv
"""
from __future__ import annotations

import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMPUTATION = os.path.join(REPO, "data", "imputation_results.tsv")
INV_PROPS = os.path.join(REPO, "data", "inv_properties.tsv")
OUT = os.path.join(REPO, "data", "imputation_threshold_summary.tsv")

R2_THRESHOLDS = (0.3, 0.5, 0.7)
BH_ALPHA = 0.05


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def consensus_keys():
    keys = set()
    with open(INV_PROPS, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if str(r.get("0_single_1_recur_consensus", "")).strip() in ("0", "1"):
                try:
                    keys.add((str(r["Chromosome"]).replace("chr", ""),
                              int(float(r["Start"])), int(float(r["End"]))))
                except (KeyError, TypeError, ValueError):
                    continue
    return keys


def build(out_path=OUT):
    with open(IMPUTATION, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    keys = consensus_keys()

    def in_consensus(r):
        try:
            return (str(r["Chromosome"]).replace("chr", ""),
                    int(float(r["Start"])), int(float(r["End"]))) in keys
        except (KeyError, TypeError, ValueError):
            return False

    subsets = [("all_models_fit", rows),
               ("consensus_93_locus_set", [r for r in rows if in_consensus(r)])]

    out = []
    for name, sel in subsets:
        for t in R2_THRESHOLDS:
            n_r2 = sum(1 for r in sel if (_f(r["unbiased_pearson_r2"]) or -1) > t)
            n_both = sum(1 for r in sel
                         if (_f(r["unbiased_pearson_r2"]) or -1) > t
                         and (_f(r.get("p_fdr_bh")) or 1.0) < BH_ALPHA)
            out.append({"subset": name, "n_models": len(sel),
                        "r2_threshold": t,
                        "n_passing_r2": n_r2,
                        "n_passing_r2_and_bh": n_both})

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)

    print(f"{len(rows)} models fit; {len(subsets[1][1])} of them at a "
          f"consensus-classified locus\n")
    print(f"{'subset':24s} {'n':>5s} {'r2>':>5s} {'pass r2':>8s} {'+ BH<0.05':>10s}")
    for r in out:
        print(f"{r['subset']:24s} {r['n_models']:5d} {r['r2_threshold']:5.1f} "
              f"{r['n_passing_r2']:8d} {r['n_passing_r2_and_bh']:10d}")
    print(f"\nwrote {out_path}\n")
    print("Reconciliation:")
    print("  replication log  '158 models, 21 with r2 > 0.3 and BH p < 0.05'"
          "  -> reproduces over all models fit")
    print("  manuscript       '12 of the 93 with r2 > 0.5 and BH p < 0.05'"
          "   -> 12 is the count over the")
    print("                    93-locus set at r2 > 0.5 alone; adding BH p < 0.05 "
          "gives 11.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    build(ap.parse_args(argv).out)


if __name__ == "__main__":
    main()
