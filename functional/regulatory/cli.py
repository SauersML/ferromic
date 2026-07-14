"""Command-line entry point for the measured regulatory-QTL analyses.

    # Geuvadis cis-eQTL (deterministic, seed 42)
    python -m functional.regulatory.cli eqtl \
        --inversions data/inversions.tsv --out results/regulatory/arm_eqtl.tsv

    # Geuvadis cis splicing-QTL for one phenotype
    python -m functional.regulatory.cli sqtl-geuvadis --pheno junction \
        --inversions data/inversions.tsv --out results/regulatory/armA_junction.tsv

    # Integrate measured + predicted regulatory consequences per locus
    python -m functional.regulatory.cli integrate \
        --eqtl results/regulatory/arm_eqtl.tsv --out results/regulatory/regulatory_per_locus.tsv

Large inputs (Geuvadis genotypes / expression / splicing matrices, GTEx caches) resolve via
``functional.paths`` (``--<name>`` overrides, ``FUNCTIONAL_*`` env vars, or FUNCTIONAL_DATA_ROOT).
See ``functional/regulatory/README.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from .. import paths
from . import eqtl as eqtl_mod
from . import integrate as integrate_mod
from . import sqtl_geuvadis


def _write_tsv(rows, path, fields=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _progress(i, n):
    if i == n or i % 10 == 0:
        print(f"  [{i}/{n}] loci", flush=True)


def cmd_eqtl(a):
    rows = eqtl_mod.run_eqtl(
        a.inversions,
        paths.resolve("geuvadis_gene_rpkm", a.gene_rpkm),
        paths.resolve("geuvadis_pgen", a.pgen),
        _panel(a),
        loci_subset=a.loci.split(",") if a.loci else None,
        n_perm=a.n_perm, seed=a.seed, progress=_progress,
    )
    _write_tsv(rows, a.out)
    resolved = {"geuvadis_gene_rpkm": paths.resolve("geuvadis_gene_rpkm", a.gene_rpkm),
                "geuvadis_pgen": paths.resolve("geuvadis_pgen", a.pgen)}
    paths.write_provenance(os.path.splitext(a.out)[0] + "_provenance.json", resolved,
                           extra={"n_tests": len(rows), "seed": a.seed, "n_perm": a.n_perm})
    n_sig = sum(1 for r in rows if r.get("bh_q", 1) < 0.05)
    print(f"eqtl: {len(rows)} gene x locus tests, {n_sig} BH-significant -> {a.out}")


def cmd_sqtl_geuvadis(a):
    matrix_key = {"junction": "geuvadis_junction", "exon": "geuvadis_exon",
                  "transcript": "geuvadis_transcript"}[a.pheno]
    rows, summary = sqtl_geuvadis.run_sqtl(
        a.pheno, paths.resolve(matrix_key, a.matrix), a.inversions,
        paths.resolve("geuvadis_pgen", a.pgen),
        paths.resolve("geuvadis_gene_rpkm", a.gene_rpkm), _panel(a),
        n_perm=a.n_perm, seed=a.seed,
    )
    _write_tsv(rows, a.out, fields=["locus", "gene", "targetid", "chrom", "fs", "fe",
                                    "cluster_size", "beta", "se", "t", "p", "bh_q", "n",
                                    "mean_ratio", "direction"])
    with open(os.path.splitext(a.out)[0] + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def cmd_integrate(a):
    rows = integrate_mod.integrate(
        a.eqtl, a.sqtl_master, a.gtex_eqtl, a.ag_splice, a.inversions, a.ensg_symbol)
    integrate_mod.write_outputs(rows, a.out)
    print(json.dumps(integrate_mod.summarize(rows), indent=2))


def _panel(a):
    # the 1000G panel ships with the package data/ dir by default
    if getattr(a, "panel", None):
        return a.panel
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "1kg_panel.tsv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eqtl", help="Geuvadis cis-eQTL by inversion-tag dosage")
    p.add_argument("--inversions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gene-rpkm", dest="gene_rpkm")
    p.add_argument("--pgen")
    p.add_argument("--panel")
    p.add_argument("--loci", help="comma-separated locus subset (e.g. a smoke run)")
    p.add_argument("--n-perm", type=int, default=eqtl_mod.N_PERM)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_eqtl)

    p = sub.add_parser("sqtl-geuvadis", help="Geuvadis cis splicing-QTL")
    p.add_argument("--pheno", choices=["junction", "exon", "transcript"], required=True)
    p.add_argument("--inversions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--matrix")
    p.add_argument("--pgen")
    p.add_argument("--gene-rpkm", dest="gene_rpkm")
    p.add_argument("--panel")
    p.add_argument("--n-perm", type=int, default=sqtl_geuvadis.N_PERM)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_sqtl_geuvadis)

    p = sub.add_parser("integrate", help="per-locus integration of measured + predicted consequences")
    p.add_argument("--eqtl", required=True)
    p.add_argument("--sqtl-master", required=True, dest="sqtl_master")
    p.add_argument("--gtex-eqtl", required=True, dest="gtex_eqtl")
    p.add_argument("--ag-splice", required=True, dest="ag_splice")
    p.add_argument("--inversions", required=True)
    p.add_argument("--ensg-symbol", required=True, dest="ensg_symbol")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_integrate)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
