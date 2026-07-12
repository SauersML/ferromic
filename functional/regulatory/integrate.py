"""Integrate measured + predicted regulatory consequences into one per-locus table.

Inputs (all measured except the AlphaGenome splice prediction):
  * ``eqtl_tsv``          measured Geuvadis cis-eQTL (this package, :mod:`functional.regulatory.eqtl`)
  * ``sqtl_master_tsv``   measured GTEx sQTL + Geuvadis splicing per inversion (upstream #8 arm)
  * ``gtex_eqtl_tsv``     measured GTEx cis-eQTL at tag SNPs (multi-tissue)
  * ``ag_splice_tsv``     AlphaGenome predicted top-splice gene per inversion (functional.splice)
  * ``inversions_tsv``    inversions with an OK tag SNP
  * ``ensg_symbol_tsv``   ENSG -> gene symbol map

Output: one row per inversion haplotype with measured eQTL (n significant genes, top gene +
direction), measured sQTL/splicing, measured GTEx eQTL, the AlphaGenome predicted top-splice
gene, and a ``measured_molecular_any`` flag.

Framing note (carried from the analysis): cis-eQTLs are ubiquitous for common variants, so
NO genome-wide "eQTL enrichment vs background" claim is made — that would be trivially true.
The eQTL results are used mechanistically (which gene, which direction). All associational,
haplotype-level.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

from .. import featured_loci as FL


def _read_tsv(path: str, delim: str = "\t") -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter=delim))


def load_symbols(ensg_symbol_tsv: str) -> dict:
    import gzip
    opener = gzip.open if str(ensg_symbol_tsv).endswith(".gz") else open
    d = {}
    with opener(ensg_symbol_tsv, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) == 2:
                d[p[0]] = p[1]
    return d


def _sym(ensg, symbols: dict) -> str:
    return symbols.get(str(ensg).split(".")[0], ensg)


def integrate(eqtl_tsv: str, sqtl_master_tsv: str, gtex_eqtl_tsv: str, ag_splice_tsv: str,
              inversions_tsv: str, ensg_symbol_tsv: str) -> list[dict]:
    symbols = load_symbols(ensg_symbol_tsv)

    eqtl_by_locus: dict = defaultdict(list)
    for r in _read_tsv(eqtl_tsv):
        try:
            r["_q"] = float(r["bh_q"]) if r["bh_q"] != "" else 1.0
        except (ValueError, KeyError):
            r["_q"] = 1.0
        eqtl_by_locus[r["locus"]].append(r)

    master8 = {r["locus"]: r for r in _read_tsv(sqtl_master_tsv)}
    ag6 = {r["locus"]: r for r in _read_tsv(ag_splice_tsv)}
    gtex_eq: dict = defaultdict(list)
    for r in _read_tsv(gtex_eqtl_tsv):
        try:
            r["_nes"] = abs(float(r["nes"]))
        except (ValueError, KeyError):
            r["_nes"] = 0.0
        gtex_eq[r["locus"]].append(r)

    out = []
    for L in (r for r in _read_tsv(inversions_tsv) if r.get("tag_status") == "ok"):
        locus = L["locus"]
        inv = FL.norm(locus)
        m = master8.get(locus, {})
        ag = ag6.get(locus, {})
        eq_sig = sorted((r for r in eqtl_by_locus.get(locus, []) if r["_q"] < 0.05), key=lambda r: r["_q"])
        top_eq = eq_sig[0] if eq_sig else None

        gtex_n_sqtl = int(m.get("gtex_n_sqtl_genes", 0) or 0)
        measured_sqtl_any = int((m.get("measured_any", "0") or "0") == "1")
        geq = gtex_eq.get(locus, [])
        gtex_eq_genes = sorted({g["geneSymbol"] for g in geq})
        gtex_eq_top = max(geq, key=lambda g: g["_nes"]) if geq else None
        measured_any = int(len(eq_sig) > 0 or measured_sqtl_any or len(gtex_eq_genes) > 0)

        out.append(dict(
            locus=locus, inv_id=inv, band=(FL.meta(inv) or {}).get("band", ""),
            is_featured=FL.is_featured(inv), tags=";".join((FL.meta(inv) or {}).get("tags", [])),
            size_bp=L.get("size", ""), tag_snp=L.get("tag_snpId", ""),
            eqtl_n_sig=len(eq_sig),
            eqtl_n_up=sum(1 for r in eq_sig if r["direction"] == "up"),
            eqtl_n_down=sum(1 for r in eq_sig if r["direction"] == "down"),
            eqtl_top_gene=_sym(top_eq["gene_id"], symbols) if top_eq else "",
            eqtl_top_beta=top_eq["beta_log2fc_per_alt"] if top_eq else "",
            eqtl_top_dir=top_eq["direction"] if top_eq else "",
            eqtl_top_q=f"{top_eq['_q']:.3g}" if top_eq else "",
            gtex_n_sqtl_genes=gtex_n_sqtl,
            gtex_sqtl_top_gene=_sym(m.get("gtex_top_gene", ""), symbols),
            geuv_splice_n_sig=int(m.get("geuv_n_sig", 0) or 0),
            geuv_splice_top_gene=_sym(m.get("geuv_top_gene", ""), symbols),
            measured_sqtl_any=measured_sqtl_any,
            gtex_eqtl_n_genes=len(gtex_eq_genes),
            gtex_eqtl_top_gene=(gtex_eq_top["geneSymbol"] if gtex_eq_top else ""),
            gtex_eqtl_top_nes=(round(float(gtex_eq_top["nes"]), 3) if gtex_eq_top else ""),
            gtex_eqtl_top_tissue=(gtex_eq_top["tissue"] if gtex_eq_top else ""),
            ag_top_splice_gene=ag.get("ag_top_splice_gene", ""),
            ag_max_splice=ag.get("ag_max_splice", ""),
            ag_top_is_measured_sgene=ag.get("ag_top_splice_is_measured_sgene", ""),
            measured_molecular_any=measured_any,
        ))
    out.sort(key=lambda r: (not r["is_featured"], -r["eqtl_n_sig"]))
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    feat = [r for r in rows if r["is_featured"]]
    return dict(
        n_loci=n,
        n_measured_molecular_any=sum(1 for r in rows if r["measured_molecular_any"]),
        frac_measured_molecular=round(sum(1 for r in rows if r["measured_molecular_any"]) / n, 3) if n else 0.0,
        n_with_geuvadis_eqtl=sum(1 for r in rows if r["eqtl_n_sig"] > 0),
        n_with_gtex_eqtl=sum(1 for r in rows if r["gtex_eqtl_n_genes"] > 0),
        n_with_sqtl=sum(1 for r in rows if r["measured_sqtl_any"]),
        n_featured=len(feat),
        n_featured_measured=sum(1 for r in feat if r["measured_molecular_any"]),
        note=("cis-eQTL is ubiquitous for common variants; no class-enrichment claim is made "
              "for eQTL. The molecular-consequence fraction establishes these haplotypes are "
              "molecularly active, NOT selection or causality. Haplotype-level throughout."),
    )


def write_outputs(rows: list[dict], out_tsv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_tsv)), exist_ok=True)
    with open(out_tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(os.path.dirname(out_tsv), "regulatory_summary.json"), "w") as fh:
        json.dump(summarize(rows), fh, indent=2)
