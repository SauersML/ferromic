"""Command-line entry point for the gene-localised AlphaGenome splice analysis.

    # Recompute the validated per-inversion splice formulation from cached AlphaGenome scores
    python -m functional.splice.cli formulate \
        --npz-dir <agscore_dir> --out results/splice/per_inversion_splice.tsv

    # (upstream) score inversions via the AlphaGenome API -> per-inversion .npz
    python -m functional.splice.cli score --inversions data/inversions.tsv --out-dir <agscore_dir>

``formulate`` reproduces ``ag_top_splice_gene`` / ``ag_max_splice`` per inversion without any
API call. ``score`` requires ALPHAGENOME_API_KEY and network access.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import formulations as F


def _write_tsv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def cmd_formulate(a):
    ag = F.load_all(a.npz_dir)
    rows = []
    for locus, genes in ag.items():
        gid, name, ag_max = F.top_splice_gene(genes)
        rows.append({"locus": locus, "n_ag_genes": len(genes),
                     "ag_top_splice_gene": name,
                     "ag_max_splice": round(ag_max, 3) if ag_max == ag_max else None})
    rows.sort(key=lambda r: -(r["ag_max_splice"] or 0))
    _write_tsv(rows, a.out)
    print(f"formulated splice scores for {len(rows)} inversions -> {a.out}")


def cmd_score(a):
    from .score_alphagenome import make_client, save_region, score_breakpoint

    client, scorers = make_client()
    loci = [r for r in csv.DictReader(open(a.inversions), delimiter="\t") if r.get("tag_status") == "ok"]
    n = 0
    for r in loci:
        # score each breakpoint of the inversion (start + end) in a centred window
        for bp in ("start", "end"):
            chrom = f"chr{r['chrom_no']}" if not str(r.get("chrom", "")).startswith("chr") else r["chrom"]
            pos = int(float(r[f"{bp}19"])) if a.build == "hg19" else int(float(r.get(bp, r[f"{bp}19"])))
            # ref base must be supplied by the caller's reference; here we require it in the row
            ref = r.get(f"{bp}_ref", "N")
            scored = score_breakpoint(client, scorers, chrom, pos, ref, a.window)
            save_region(a.out_dir, f"{r['locus']}__{bp}", scored)
            n += 1
    print(f"scored {n} breakpoints -> {a.out_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("formulate", help="recompute validated splice formulation from cached .npz")
    p.add_argument("--npz-dir", required=True, dest="npz_dir")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_formulate)

    p = sub.add_parser("score", help="score inversions via the AlphaGenome API (needs API key)")
    p.add_argument("--inversions", required=True)
    p.add_argument("--out-dir", required=True, dest="out_dir")
    p.add_argument("--window", type=int, default=1_048_576)
    p.add_argument("--build", default="hg38", choices=["hg38", "hg19"])
    p.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
