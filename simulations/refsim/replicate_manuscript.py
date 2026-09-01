#!/usr/bin/env python
"""Check the manuscript's Fig. 1 simulation claims against a replication run.

Reads the per-locus CSVs from ``run_grid.py --task replicate`` and reports each
claim beside what the simulation produced, rather than leaving the comparison to
the reader. The three claims, quoted from the Results:

1. "Across all single-event models considered ... low false positive rates
   (<5%), with the highest rate (4%, 95% C.I.: 0-8%) observed in models of very
   recent inversion events (50 kya) and when recombination rate is high
   (1.0x10-6 per base per generation), regardless of the sampling frequency."
2. "With moderate recombination rate (1.0x10-8), the power ranges from 66-92%
   when the frequency of inversions is above 5%; however, it decreases to as low
   as 28% when the inversion frequency is below 5%."
3. "it tends to overestimate the number of inversion events ... especially for
   younger events."

The single-origin arm uses the public two-population ``singleINV_m1.py``
topology with upstream's N_a/100 inverted deme. The recurrent arm uses the public nine-population
``recurrentINV_m1.2pop.py`` model and its two independent random sampling
mixtures.

    python replicate_manuscript.py 'out/replicate_shard*.csv'
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
from collections import defaultdict

# Deepest first, so tables read the way the Methods list the depths.
DEPTHS = ["old", "young", "recent", "very_recent"]
DEPTH_KYA = {"old": 500, "young": 250, "recent": 100, "very_recent": 50}
RHOS = [0.0, 1e-8, 1e-6]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("error"):
                        continue
                    rows.append(dict(
                        scenario=r["scenario"], depth=r["depth"],
                        rho=float(r["rho"]), freq=float(r["inv_freq"]),
                        events=int(float(r["tree_n_events"])),
                        call=int(r["call_recurrent"]),
                        fi=float(r["frac_admix_i"]),
                    ))
    return rows


def rate(rs):
    k, n = sum(r["call"] for r in rs), len(rs)
    lo, hi = wilson(k, n)
    return k, n, (k / n if n else float("nan")), lo, hi


def claim_one(rows):
    print("\n" + "=" * 78)
    print("CLAIM 1  single-event false-positive rate < 5%, highest 4% (CI 0-8%)")
    print("         at 50 kya with rho = 1e-6, regardless of frequency")
    print("=" * 78)
    sing = [r for r in rows if r["scenario"] == "single"]
    if not sing:
        print("  no single-event rows found")
        return
    print("\nBy depth and recombination rate, pooled over frequency:\n")
    print("%-14s %-22s %-22s %-22s" % ("depth", "rho = 0", "rho = 1e-8", "rho = 1e-6"))
    worst = (None, -1.0)
    for d in DEPTHS:
        cells = []
        for rho in RHOS:
            rs = [r for r in sing if r["depth"] == d and r["rho"] == rho]
            if not rs:
                cells.append("%-22s" % "-")
                continue
            k, n, p, lo, hi = rate(rs)
            cells.append("%-22s" % ("%.3f (%.2f-%.2f) n=%d" % (p, lo, hi, n)))
            if p > worst[1]:
                worst = ((d, rho), p)
        print("%-14s %s" % ("%s (%d kya)" % (d, DEPTH_KYA[d]), " ".join(cells)))

    k, n, p, lo, hi = rate(sing)
    print("\n  overall            : %.4f (%.3f-%.3f), n = %d" % (p, lo, hi, n))
    print("  worst cell         : %s at rho = %g, rate %.3f"
          % (worst[0][0], worst[0][1], worst[1]))
    print("  manuscript says    : < 0.05 everywhere, max 0.04 at 50 kya / rho 1e-6")
    over = []
    for d in DEPTHS:
        for rho in RHOS:
            rs = [r for r in sing if r["depth"] == d and r["rho"] == rho]
            if rs:
                kk, nn, pp, ll, hh = rate(rs)
                if ll > 0.05:
                    over.append("%s/rho=%g (%.3f, CI %.2f-%.2f)" % (d, rho, pp, ll, hh))
    print("  cells whose 95%% CI excludes 5%%: %s"
          % (", ".join(over) if over else "none"))

    print("\n  Is the maximum at rho = 1e-6, as claimed?")
    for d in DEPTHS:
        line = []
        for rho in RHOS:
            rs = [r for r in sing if r["depth"] == d and r["rho"] == rho]
            line.append("%.3f" % rate(rs)[2] if rs else "-")
        print("    %-22s %s" % ("%s (%d kya)" % (d, DEPTH_KYA[d]),
                                "  ".join("rho=%-6g %s" % (r, v)
                                          for r, v in zip(RHOS, line))))

    print("\n  Does frequency matter, as the claim's 'regardless' implies?")
    for f in sorted({r["freq"] for r in sing}):
        rs = [r for r in sing if r["freq"] == f]
        k, n, p, lo, hi = rate(rs)
        print("    freq %5.2f : %.3f (%.2f-%.2f) n=%d" % (f, p, lo, hi, n))


def claim_two(rows):
    print("\n" + "=" * 78)
    print("CLAIM 2  at rho = 1e-8, power 66-92% above 5% frequency,")
    print("         falling to 28% below 5%")
    print("=" * 78)
    up = [r for r in rows if r["scenario"] == "recurrent" and r["rho"] == 1e-8]
    if not up:
        print("  no recurrent rows at rho = 1e-8")
        return
    print("\nPower by frequency and depth (rho = 1e-8):\n")
    freqs = sorted({r["freq"] for r in up})
    print("%-8s %s" % ("freq", " ".join("%-18s" % ("%s (%dk)" % (d, DEPTH_KYA[d]))
                                        for d in DEPTHS)))
    for f in freqs:
        cells = []
        for d in DEPTHS:
            rs = [r for r in up if r["freq"] == f and r["depth"] == d]
            cells.append("%-18s" % ("%.2f (%.2f-%.2f)" % rate(rs)[2:5] if rs else "-"))
        print("%-8.2f %s" % (f, " ".join(cells)))

    above = [r for r in up if r["freq"] > 0.05]
    below = [r for r in up if r["freq"] < 0.05]
    for label, rs in (("freq > 5%", above), ("freq < 5%", below)):
        if rs:
            k, n, p, lo, hi = rate(rs)
            print("  %-10s pooled power %.3f (%.3f-%.3f), n = %d"
                  % (label, p, lo, hi, n))
    per = []
    for f in freqs:
        if f > 0.05:
            for d in DEPTHS:
                rs = [r for r in up if r["freq"] == f and r["depth"] == d]
                if rs:
                    per.append(rate(rs)[2])
    if per:
        print("  range across cells above 5%%: %.2f to %.2f  (manuscript 0.66-0.92)"
              % (min(per), max(per)))


def claim_three(rows):
    print("\n" + "=" * 78)
    print("CLAIM 3  the approach overestimates the number of inversion events,")
    print("         especially for younger events")
    print("=" * 78)
    print("\nInferred events (truth: 1 for single-event, 3 for recurrent):\n")
    print("%-14s %-26s %-26s" % ("depth", "single: mean (median)",
                                 "recurrent: mean (median)"))
    for d in DEPTHS:
        cells = []
        for sc in ("single", "recurrent"):
            rs = [r for r in rows if r["scenario"] == sc and r["depth"] == d]
            if not rs:
                cells.append("%-26s" % "-")
                continue
            ev = [r["events"] for r in rs]
            cells.append("%-26s" % ("%.2f (%.0f)  n=%d"
                                    % (statistics.fmean(ev),
                                       statistics.median(ev), len(ev))))
        print("%-14s %s" % ("%s (%d kya)" % (d, DEPTH_KYA[d]), " ".join(cells)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="*", default=["out/replicate_shard*.csv"])
    args = ap.parse_args(argv)
    rows = load(args.inputs or ["out/replicate_shard*.csv"])
    if not rows:
        raise SystemExit("no rows found")
    print("replication run: %d loci, scenarios %s"
          % (len(rows), sorted({r["scenario"] for r in rows})))
    claim_one(rows)
    claim_two(rows)
    claim_three(rows)


if __name__ == "__main__":
    main()
