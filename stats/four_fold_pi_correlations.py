#!/usr/bin/env python
"""Concordance between 4-fold-site pi and the other pi measures (Reviewer 1).

Reviewer 1 asked for diversity at 4-fold degenerate sites as the most reliable
neutral estimate. The response letter quotes two Spearman correlations between
orientation *differences* -- whole-locus vs 4-fold, and 4-fold vs whole-CDS --
which ``four_fold_pi.py`` computes the inputs for but never records. This script
computes them from the committed per-inversion table so both numbers are
reproducible from committed data alone, with no alignment inputs required.

Orientation difference for a measure m is ``m_inverted - m_direct`` per locus.
A locus contributes only when both orientations actually have 4-fold sites
(``pi_fourfold_*`` is 0, not blank, when they do not). Two subsets are reported:
``all_with_fourfold`` (46 loci) and ``recurrence_classified`` (26 loci -- the
further restriction to a consensus recurrence call that the paired tests in
``four_fold_pi.py`` apply, and the set the response letter quotes).

Input : data/four_fold_pi_by_inversion.tsv
Output: data/four_fold_pi_correlations.tsv
"""
from __future__ import annotations

import argparse
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_IN = os.path.join(REPO, "data", "four_fold_pi_by_inversion.tsv")
DEFAULT_OUT = os.path.join(REPO, "data", "four_fold_pi_correlations.tsv")

PAIRS = [
    ("wholeLocus", "fourfold",
     "whole-locus vs 4-fold orientation difference"),
    ("fourfold", "wholeCDS",
     "4-fold vs whole-CDS orientation difference"),
    ("wholeLocus", "wholeCDS",
     "whole-locus vs whole-CDS orientation difference"),
]


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _delta(row, measure):
    inv = _f(row.get(f"pi_{measure}_inverted"))
    dir_ = _f(row.get(f"pi_{measure}_direct"))
    if inv is None or dir_ is None:
        return None
    return inv - dir_


def spearman(x, y):
    """Spearman rho with average ranks, and a two-sided t-approximation p-value."""
    from scipy import stats

    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def build(in_path=DEFAULT_IN, out_path=DEFAULT_OUT):
    with open(in_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    # A locus only carries 4-fold information when both orientations actually
    # have 4-fold sites; pi_fourfold_* is 0 rather than blank when they do not.
    usable = [r for r in rows
              if (_f(r.get("fourfold_sites_direct")) or 0) > 0
              and (_f(r.get("fourfold_sites_inverted")) or 0) > 0]
    # The paired tests in four_fold_pi.py are further restricted to loci with a
    # consensus recurrence call, which is the set the response letter quotes.
    classified = [r for r in usable if str(r.get("recurrence", "")).strip() != ""]

    subsets = [("all_with_fourfold", usable),
               ("recurrence_classified", classified)]

    out = []
    for subset_name, subset in subsets:
        for a, b, label in PAIRS:
            pairs = [(_delta(r, a), _delta(r, b)) for r in subset]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 3:
                print(f"skipping {subset_name}/{label}: {len(pairs)} usable loci")
                continue
            rho, p = spearman([q[0] for q in pairs], [q[1] for q in pairs])
            out.append({
                "subset": subset_name,
                "measure_x": a, "measure_y": b, "comparison": label,
                "n_loci": len(pairs),
                "spearman_rho": f"{rho:.6f}", "p_value": f"{p:.6g}",
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)

    print(f"{len(rows)} loci in table -> {out_path}\n")
    for r in out:
        print(f"{r['subset']:22s} {r['comparison']:48s} n={r['n_loci']:3d}  "
              f"rho={float(r['spearman_rho']):.3f}  p={float(r['p_value']):.3g}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    build(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
