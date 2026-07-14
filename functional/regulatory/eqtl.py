"""Geuvadis cis-eQTL by inversion-tag dosage.

For every gene whose hg19 position falls within ``window`` bp of an inversion, regress
``log2(RPKM+1)`` on inversion-tag ALT dosage plus covariates (sex, population, genome-wide
expression PCs). Reports the dosage effect (``beta`` = per-ALT-allele log2 fold change),
direction, analytic p, a per-locus permuted-dosage empirical null, and a genome-wide
BH-FDR q across all gene x locus tests.

Associational and haplotype-level: the tag-SNP dosage indexes the inversion haplotype, not
the inversion per se. Deterministic given the inputs and ``seed`` (default 42).
"""
from __future__ import annotations

import csv
import gzip

import numpy as np

from . import common as C

WINDOW = 1_000_000   # cis window (bp) around the inversion interval (hg19)
MIN_SAMPLES = 80     # min samples with defined expression + dosage
MIN_MEAN_RPKM = 0.1  # drop genes essentially unexpressed in LCL
N_PERM = 1000        # permutations for the per-locus empirical null


def load_loci(inversions_tsv: str, window: int = WINDOW) -> list[dict]:
    """Inversions with an OK tag SNP and hg19 coordinates, annotated with cis-window bounds."""
    rows = [r for r in csv.DictReader(open(inversions_tsv), delimiter="\t")
            if r.get("tag_pos19") and r.get("start19") and r.get("tag_variantId")
            and r.get("tag_status") == "ok"]
    for r in rows:
        r["s19"], r["e19"] = int(float(r["start19"])), int(float(r["end19"]))
        r["lo"] = min(r["s19"], r["e19"]) - window
        r["hi"] = max(r["s19"], r["e19"]) + window
    return rows


def collect_cis(gene_rpkm_path: str, loci: list[dict]) -> tuple[list[str], list[dict]]:
    """Stream the gene RPKM matrix once; keep genes whose hg19 coord overlaps any cis window."""
    by_chrom: dict = {}
    for r in loci:
        by_chrom.setdefault(str(r["chrom_no"]), []).append(r)
    kept = []
    with gzip.open(gene_rpkm_path, "rt") as fh:
        samples = fh.readline().rstrip("\n").split("\t")[4:]
        for line in fh:
            p = line.rstrip("\n").split("\t")
            cand = by_chrom.get(p[2])
            if not cand:
                continue
            try:
                coord = int(float(p[3]))
            except ValueError:
                continue
            hit = [rr["locus"] for rr in cand if rr["lo"] <= coord <= rr["hi"]]
            if not hit:
                continue
            try:
                vals = np.array(p[4:], dtype=np.float64)
            except ValueError:
                continue
            kept.append({"gene_id": p[0], "gene_sym": p[1], "chrom": p[2],
                         "coord": coord, "vals": vals, "loci": hit})
    return samples, kept


def run_eqtl(inversions_tsv: str, gene_rpkm_path: str, pgen_prefix: str, panel_tsv: str,
             loci_subset: list[str] | None = None, window: int = WINDOW, n_perm: int = N_PERM,
             n_pc: int = 10, seed: int = 42, progress=None) -> list[dict]:
    """Run the cis-eQTL scan and return one row per gene x locus test, sorted by analytic p
    with a genome-wide BH q attached. Deterministic given the inputs and ``seed``."""
    loci = load_loci(inversions_tsv, window)
    if loci_subset is not None:
        loci = [r for r in loci if r["locus"] in set(loci_subset)]

    esamp, kept = collect_cis(gene_rpkm_path, loci)
    psam, dos, meta = C.load_tag_dosages(pgen_prefix, loci)
    pcol = {s: i for i, s in enumerate(psam)}
    keep_samp = [s for s in esamp if s in pcol]
    e_idx = np.array([esamp.index(s) for s in keep_samp])
    p_idx = np.array([pcol[s] for s in keep_samp])
    Cmat, _, _ = C.build_covariates(gene_rpkm_path, panel_tsv, keep_samp, n_pc=n_pc)

    rng = np.random.default_rng(seed)
    rows = []
    for li, loc in enumerate(loci):
        locus = loc["locus"]
        d_full = dos.get(locus)
        if d_full is None:
            continue
        d = d_full[p_idx]
        for g in (g for g in kept if locus in g["loci"]):
            y_full = g["vals"][e_idx]
            ok = np.isfinite(y_full) & np.isfinite(d)
            if ok.sum() < MIN_SAMPLES:
                continue
            y = np.log2(np.clip(y_full[ok], 0, None) + 1.0)
            if y.mean() < np.log2(1 + MIN_MEAN_RPKM) or y.std() < 1e-6:
                continue
            dd = d[ok]
            if dd.std() < 1e-6:
                continue
            beta, se, t, p, n = C.ols_beta_p(y, np.column_stack([dd, Cmat[ok]]), 0)
            if not np.isfinite(p):
                continue
            perm_ge = 0
            for _ in range(n_perm):
                _, _, _, pp, _ = C.ols_beta_p(y, np.column_stack([rng.permutation(dd), Cmat[ok]]), 0)
                if np.isfinite(pp) and pp <= p:
                    perm_ge += 1
            rows.append(dict(
                locus=locus, band=(meta.get(locus, {}) or {}).get("band", ""),
                gene_id=g["gene_id"], gene_sym=g["gene_sym"], chrom=g["chrom"],
                coord_hg19=g["coord"], n=n,
                beta_log2fc_per_alt=round(beta, 5),
                se=round(se, 5) if np.isfinite(se) else "",
                t=round(t, 4) if np.isfinite(t) else "",
                p=p, perm_p=(perm_ge + 1) / (n_perm + 1),
                direction=("up" if beta > 0 else "down"),
                mean_log2rpkm=round(float(y.mean()), 4),
            ))
        if progress:
            progress(li + 1, len(loci))

    if rows:
        q = C.bh_fdr([r["p"] for r in rows])
        for r, qq in zip(rows, q):
            r["bh_q"] = float(qq)
        rows.sort(key=lambda r: r["p"])
    return rows
