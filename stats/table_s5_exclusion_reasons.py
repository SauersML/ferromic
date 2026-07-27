#!/usr/bin/env python
"""Per-locus exclusion reason for the balanced inversion set (Table S5).

Reviewer 2 (comment 5) counted 292 rows in Table S5 and asked why only 93 are
analysed. The answer is per locus and is fully determined by the two Porubsky
et al. (2022) recurrence verdicts already carried in ``inv_properties.tsv``:
a locus enters the analysis set only when both methods returned a call and the
two calls agree.

This script emits that reason as a column so the table answers the question on
its face, and prints the counts that back the sentence in the response letter.

Input : data/inv_properties.tsv   (columns verdictRecurrence_hufsah,
        verdictRecurrence_benson, 0_single_1_recur_consensus)
Output: data/table_s5_exclusion_reasons.tsv

The two verdict columns are reported under their source column names rather than
under method labels: the mapping from these columns to "tagging SNP" versus
"haplotype-based coalescent" is not recorded in the input table, and asserting it
here would be a guess.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_IN = os.path.join(REPO, "data", "inv_properties.tsv")
DEFAULT_OUT = os.path.join(REPO, "data", "table_s5_exclusion_reasons.tsv")

A = "verdictRecurrence_hufsah"
B = "verdictRecurrence_benson"
CONSENSUS = "0_single_1_recur_consensus"


def reason(row):
    """(analysed?, reason) for one locus."""
    a, b = row.get(A, "NA").strip(), row.get(B, "NA").strip()
    a_called, b_called = a in ("TRUE", "FALSE"), b in ("TRUE", "FALSE")
    if not a_called and not b_called:
        return False, "no recurrence call from either method"
    if not b_called:
        return False, f"no call from {B} (only {A} = {a})"
    if not a_called:
        return False, f"no call from {A} (only {B} = {b})"
    if a != b:
        return False, f"methods disagree ({A} = {a}, {B} = {b})"
    return True, ""


def build(in_path=DEFAULT_IN, out_path=DEFAULT_OUT):
    with open(in_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    counts = Counter()
    out_rows = []
    for r in rows:
        analysed, why = reason(r)
        counts[why or "ANALYSED"] += 1
        consensus = (r.get(CONSENSUS) or "NA").strip()
        # Cross-check: the recorded consensus must agree with the derived status.
        derived = consensus in ("0", "1")
        if derived != analysed:
            raise SystemExit(
                f"consensus/verdict mismatch at {r.get('OrigID')}: "
                f"consensus={consensus!r} but derived analysed={analysed}")
        out_rows.append({
            "OrigID": r.get("OrigID", ""),
            "Chromosome": r.get("Chromosome", ""),
            "Start": r.get("Start", ""),
            "End": r.get("End", ""),
            A: r.get(A, ""),
            B: r.get(B, ""),
            "consensus_recurrence": {"0": "single", "1": "recurrent"}.get(
                consensus, "NA"),
            "analysed": "yes" if analysed else "no",
            "exclusion_reason": why,
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(rows)} loci -> {out_path}")
    for why, n in counts.most_common():
        print(f"  {n:4d}  {why}")
    n_used = counts["ANALYSED"]
    n_rec = sum(1 for r in out_rows if r["consensus_recurrence"] == "recurrent")
    n_sin = sum(1 for r in out_rows if r["consensus_recurrence"] == "single")
    print(f"\nanalysed {n_used} of {len(rows)} "
          f"({n_sin} single-event, {n_rec} recurrent); "
          f"{len(rows) - n_used} excluded")
    return out_rows, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    build(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
