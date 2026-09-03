#!/usr/bin/env python
"""Assert that every derived artifact for one grid prefix came from one run.

``make_report.py`` writes ``<prefix>_results.csv``, ``<prefix>_summary.tsv``,
``<prefix>_caption.txt`` and ``sweep_<prefix>.json`` from a single aggregation,
so they agree by construction. They stop agreeing when one of them is rebuilt on
its own: commit b10d8ed9 restored the N_a/100 single-event deme and regenerated
the CSV, the summary and the caption, but left ``sweep_gene_flux.json`` holding
the superseded N_a/10 counts. Nothing caught it, and the stale file reached
``origin/main``.

This runs after ``make_report.py`` and fails the job if the artifacts disagree.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def cell_key(row):
    return (row["scenario"], row["depth"], float(row["rho"]), float(row["m_flux"]))


def load_csv(path):
    with open(path, newline="") as handle:
        return {cell_key(r): (int(r["n_called"]), int(r["reps"]))
                for r in csv.DictReader(handle)}


def load_sweep(path):
    with open(path) as handle:
        return {cell_key(r): (int(r["n_called"]), int(r["reps"]))
                for r in json.load(handle)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="gene_flux")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--expect-single-fraction", type=float, default=1 / 100,
                    help="single-event inverted deme as a fraction of N_a")
    args = ap.parse_args()

    p = lambda name: os.path.join(args.dir, name)
    csv_path = p(f"{args.prefix}_results.csv")
    sweep_path = p(f"sweep_{args.prefix}.json")
    caption_path = p(f"{args.prefix}_caption.txt")
    summary_path = p(f"{args.prefix}_summary.tsv")

    problems = []

    # 1. the JSON and the CSV are the same aggregation serialized twice
    table = load_csv(csv_path)
    sweep = load_sweep(sweep_path)
    if set(table) != set(sweep):
        problems.append(
            f"{os.path.basename(sweep_path)} and {os.path.basename(csv_path)} "
            f"cover different cells: only-csv={len(set(table) - set(sweep))}, "
            f"only-json={len(set(sweep) - set(table))}"
        )
    else:
        bad = [k for k in table if table[k] != sweep[k]]
        if bad:
            k0 = bad[0]
            problems.append(
                f"{len(bad)} of {len(table)} cells disagree between "
                f"{os.path.basename(csv_path)} and {os.path.basename(sweep_path)}; "
                f"first {k0}: csv={table[k0]} json={sweep[k0]}"
            )

    # 2. the summary totals are the CSV totals
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as handle:
            for r in csv.DictReader(handle, delimiter="\t"):
                key = (r["scenario"], float(r["m_flux"]))
                k = sum(v[0] for c, v in table.items()
                        if c[3] == key[1] and c[0].startswith(key[0].split("-")[0]))
                n = sum(v[1] for c, v in table.items()
                        if c[3] == key[1] and c[0].startswith(key[0].split("-")[0]))
                if (int(r["n_called"]), int(r["reps"])) != (k, n):
                    problems.append(
                        f"summary row {key} says {r['n_called']}/{r['reps']}, "
                        f"csv gives {k}/{n}"
                    )
                    break

    # 3. the caption's headline rate is the CSV's highest-flux rate
    if os.path.exists(caption_path):
        caption = open(caption_path, encoding="utf-8").read()
        singles = [c for c in table if c[0].startswith("single")]
        if singles:
            top = max(c[3] for c in singles)
            k = sum(table[c][0] for c in singles if c[3] == top)
            n = sum(table[c][1] for c in singles if c[3] == top)
            lo, hi = wilson_ci(k, n)
            want = (f"false-positive rate is {100 * k / n:.1f}% "
                    f"(95% C.I.: {100 * lo:.1f}–{100 * hi:.1f}%)")
            if want not in caption:
                problems.append(
                    f"caption does not state the csv's highest-flux rate; "
                    f"expected \"{want}\""
                )
            # make_report tests the rates pooled by flux level, not per cell
            pooled = {}
            for c in singles:
                k_f, n_f = pooled.get(c[3], (0, 0))
                pooled[c[3]] = (k_f + table[c][0], n_f + table[c][1])
            worst = max(k_f / n_f for k_f, n_f in pooled.values())
            claims_below = "but remains below 5%" in caption
            if (worst < 0.05) != claims_below:
                problems.append(
                    f"caption says {'below' if claims_below else 'not below'} 5% "
                    f"but the csv's highest pooled single-event rate is "
                    f"{100 * worst:.1f}%"
                )

    # 4. the single-event deme size has not drifted back off N_a/100
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import refsim
    except Exception as exc:                       # pragma: no cover
        problems.append(f"could not import refsim to check the deme size: {exc}")
    else:
        if refsim.SINGLE_INV_FRACTION != args.expect_single_fraction:
            problems.append(
                f"single-event inverted deme is N_a x {refsim.SINGLE_INV_FRACTION!r}, "
                f"expected N_a x {args.expect_single_fraction!r} (N_a/100)"
            )

    if problems:
        for line in problems:
            print(f"inconsistent: {line}", file=sys.stderr)
        raise SystemExit(f"{len(problems)} artifact inconsistencies for '{args.prefix}'")
    print(f"{args.prefix}: results.csv, sweep json, summary and caption agree "
          f"across {len(table)} cells; single-event deme is N_a/100")


if __name__ == "__main__":
    main()
