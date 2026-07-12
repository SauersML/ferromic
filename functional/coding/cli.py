"""Command-line entry point for the coding functional-call consolidation.

    python -m functional.coding.cli consolidate \
        --variants  arm1_coding_variants.tsv \
        --scores    arm1_final_3method.tsv \
        --out       results/coding/arm1_coding_calls.tsv

``consolidate`` joins the consequence annotation (all CDS sites) to the 3-method scores and
writes the coding-call table + summary. The per-method scoring stages (AlphaMissense / ESM C
/ Evo 2) run upstream and produce ``arm1_final_3method.tsv``; see the module docstrings and
``functional/coding/README.md`` for those (they require reference data + GPU).

``recombine`` recomputes the per-method damaging flags and ``n_methods_flag`` from raw
per-method scores using the documented thresholds — useful to regenerate the concordance
count without re-running the models.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import combine


def _read_tsv(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_tsv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def cmd_consolidate(args: argparse.Namespace) -> None:
    variants = _read_tsv(args.variants)
    scores = _read_tsv(args.scores)
    rows = combine.consolidate(variants, scores)
    _write_tsv(rows, args.out)
    summ = combine.summarize(rows)
    with open(os.path.splitext(args.out)[0].replace("_calls", "") + "_summary.json", "w") as fh:
        json.dump(summ, fh, indent=2)
    print(json.dumps({k: summ[k] for k in ("n_cds_sites", "n_missense", "n_functional_3of3",
                                           "n_likely_2of3", "n_seqmodel_only")}, indent=2))


def cmd_recombine(args: argparse.Namespace) -> None:
    rows = combine.combine_methods(_read_tsv(args.scores))
    _write_tsv(rows, args.out)
    n = sum(1 for r in rows if r["n_methods_flag"] >= 2)
    print(f"recomputed flags for {len(rows)} variants; {n} flagged by >=2 methods -> {args.out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("consolidate", help="join consequence annotation + 3-method scores -> coding calls")
    p.add_argument("--variants", required=True, help="per-site consequence annotation (arm1_coding_variants.tsv)")
    p.add_argument("--scores", required=True, help="3-method per-variant scores (arm1_final_3method.tsv)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_consolidate)

    p = sub.add_parser("recombine", help="recompute per-method flags + n_methods_flag from raw scores")
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_recombine)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
