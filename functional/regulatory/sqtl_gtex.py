"""Measured multi-tissue splicing at inversions via GTEx v10 sQTLs (Arm B), with a
MAF- and gene-proximity-matched enrichment test.

For each inversion tag SNP, query the GTEx portal ``singleTissueSqtl`` endpoint (measured
LeafCutter intron-excision sQTLs across ~49 tissues) to build a per-inversion x gene x tissue
map with NES and direction. Enrichment asks whether an inversion tag SNP is more likely to be
a measured sVariant than a MAF/location-matched background of random common variants (one-
sided Fisher exact). Matching on MAF *and* gene proximity is essential because inversion tag
SNPs are common and sit in gene-dense regions; the enrichment is modest and goes null under a
looser match, so it is reported as borderline, not robust.

Tag SNPs are hg38 (GTEx build); background variants are drawn from the Geuvadis pvar (hg19)
and lifted hg19->hg38. Requires network access to the GTEx portal; results are cached to disk.
"""
from __future__ import annotations

import csv
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

from . import common as C

BASE = "https://gtexportal.org/api/v2"


def _ensg(x) -> str:
    return str(x).split(".")[0]


def _get(path: str, params: dict, retries: int = 4):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params, doseq=True)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def _paged(path: str, params: dict) -> list[dict]:
    out, page = [], 0
    while True:
        d = _get(path, dict(params, itemsPerPage=1000, page=page))
        rows = d.get("data", [])
        out.extend(rows)
        pg = d.get("paging_info", {})
        if page >= pg.get("numberOfPages", 1) - 1 or not rows:
            break
        page += 1
    return out


def sqtl_for_variant(vid: str, cache: dict) -> list[dict]:
    """Cached GTEx v10 single-tissue sQTLs for one variant id (slim rows)."""
    if vid in cache:
        return cache[vid]
    try:
        rows = _paged("association/singleTissueSqtl", {"variantId": vid, "datasetId": "gtex_v10"})
    except Exception:
        rows = []
    slim = [{"gene": _ensg(r.get("gencodeId", "")), "geneSymbol": r.get("geneSymbol", ""),
             "tissue": r.get("tissueSiteDetailId", ""), "nes": r.get("nes"),
             "p": r.get("pValue"), "phenotypeId": r.get("phenotypeId", "")}
            for r in rows]
    cache[vid] = slim
    return slim


def gene_tss_by_chrom(gene_rpkm_path: str) -> dict:
    """hg19 gene start sites per chromosome from the Geuvadis gene RPKM matrix (Coord col)."""
    import gzip
    from collections import defaultdict
    by = defaultdict(list)
    with gzip.open(gene_rpkm_path, "rt") as fh:
        fh.readline()
        for line in fh:
            p = line.split("\t", 4)
            try:
                by[p[2].replace("chr", "")].append(int(float(p[3])))
            except (ValueError, IndexError):
                continue
    return {c: np.array(sorted(v)) for c, v in by.items()}


def _maf_from_dosage(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    af = d.mean() / 2.0
    return min(af, 1 - af)


def draw_background(pgen_prefix: str, gene_rpkm_path: str, n_bg: int, tag_mafs,
                    seed: int = 42, prox: int = 50_000, pool_mult: int = 20):
    """MAF- and gene-proximity-matched background of common biallelic SNVs (MAF>=0.01, within
    ``prox`` bp of a gene TSS), nearest-MAF-matched to the inversion tag SNPs. Returns
    ``(gtex_variant_ids_hg38, background_mafs)``."""
    import pgenlib
    from pyliftover import LiftOver

    lo = LiftOver("hg19", "hg38")
    rng = np.random.default_rng(seed)
    tss = gene_tss_by_chrom(gene_rpkm_path)
    tag_mafs = np.array([m for m in tag_mafs if np.isfinite(m)])

    def near_gene(chrom_no, pos):
        arr = tss.get(chrom_no)
        if arr is None or arr.size == 0:
            return False
        i = np.searchsorted(arr, pos)
        return any(0 <= j < arr.size and abs(int(arr[j]) - pos) <= prox for j in (i - 1, i))

    pool_target = n_bg * pool_mult
    picks, k = [], 0
    pvar = pgen_prefix.replace(".pgen", ".pvar")
    with open(pvar) as fh:
        idx = -1
        for line in fh:
            if line.startswith("#"):
                continue
            idx += 1
            tab = line.split("\t", 6)
            ref, alt = tab[3], tab[4]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            chrom_no, pos = tab[0], int(tab[1])
            if not near_gene(chrom_no, pos):
                continue
            k += 1
            rec = (idx, chrom_no, pos, ref, alt)
            if len(picks) < pool_target:
                picks.append(rec)
            else:
                j = rng.integers(0, k)
                if j < pool_target:
                    picks[j] = rec

    psam_n = len(open(pgen_prefix.replace(".pgen", ".psam")).read().splitlines()) - 1
    pr = pgenlib.PgenReader(bytes(pgen_prefix, "utf8"), raw_sample_ct=psam_n)
    buf = np.empty(psam_n, dtype=np.int8)
    cand = []
    for idx, chrom_no, pos, ref, alt in sorted(picks):
        pr.read(idx, buf)
        d = buf.astype(np.float64)
        d[d < 0] = np.nan
        maf = _maf_from_dosage(d)
        if np.isfinite(maf) and maf >= 0.01:
            cand.append((maf, chrom_no, pos, ref, alt))
    pr.close()
    if not cand or tag_mafs.size == 0:
        return [], []

    cand_maf = np.array([c[0] for c in cand])
    used: set = set()
    out, out_mafs = [], []
    for t in (tag_mafs[i % len(tag_mafs)] for i in range(n_bg * 3)):
        for j in np.argsort(np.abs(cand_maf - t)):
            j = int(j)
            if j not in used:
                used.add(j)
                mafj, chrom_no, pos, ref, alt = cand[j]
                lifted = lo.convert_coordinate(f"chr{chrom_no}", pos - 1)
                if lifted:
                    out.append(f"chr{chrom_no}_{lifted[0][1] + 1}_{ref}_{alt}_b38")
                    out_mafs.append(float(mafj))
                break
        if len(out) >= n_bg:
            break
    return out, out_mafs


def enrichment_fisher(n_tag_sqtl: int, n_tag: int, n_bg_sqtl: int, n_bg: int) -> float:
    """One-sided (greater) Fisher exact p for tag-SNP sQTL rate vs background."""
    from scipy.stats import fisher_exact
    _, p = fisher_exact([[n_tag_sqtl, n_tag - n_tag_sqtl],
                         [n_bg_sqtl, n_bg - n_bg_sqtl]], alternative="greater")
    return float(p)


def per_inversion(loci: list[dict], cache: dict) -> tuple[list[dict], list[dict], int]:
    """Build the sQTL map + per-inversion summary rows. Returns (sqtl_map, per_inv, n_tag_sqtl)."""
    sqtl_map, per_inv, n_tag_sqtl = [], [], 0
    for r in loci:
        vid = r["tag_variantId"]
        genes: dict = {}
        for row in sqtl_for_variant(vid, cache):
            if row["nes"] is None:
                continue
            genes.setdefault(row["gene"], []).append(row)
            sqtl_map.append({"locus": r["locus"], "tag_variantId": vid, "gene": row["gene"],
                             "geneSymbol": row["geneSymbol"], "tissue": row["tissue"],
                             "nes": row["nes"], "p": row["p"],
                             "direction": "higher_with_alt" if float(row["nes"]) > 0 else "lower_with_alt"})
        n_tag_sqtl += 1 if genes else 0
        if genes:
            best = max(genes, key=lambda g: max(abs(float(x["nes"])) for x in genes[g]))
            per_inv.append({"locus": r["locus"], "n_sqtl_genes": len(genes),
                            "n_sqtl_rows": sum(len(v) for v in genes.values()),
                            "n_tissues": len({x["tissue"] for v in genes.values() for x in v}),
                            "top_gene": best, "top_geneSymbol": genes[best][0]["geneSymbol"],
                            "max_absnes": max(abs(float(x["nes"])) for v in genes.values() for x in v),
                            "has_sqtl": 1})
        else:
            per_inv.append({"locus": r["locus"], "n_sqtl_genes": 0, "n_sqtl_rows": 0,
                            "n_tissues": 0, "top_gene": "", "top_geneSymbol": "",
                            "max_absnes": 0.0, "has_sqtl": 0})
    return sqtl_map, per_inv, n_tag_sqtl
