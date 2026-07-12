"""Measured cis differential splicing at inversions in Geuvadis LCLs (Arm A of the measured
regulatory analysis).

Phenotypes:
  ``junction``   LeafCutter-style intron-excision ratio = junction / cluster-sum, where a
                 cluster = junctions in a gene sharing a splice site (donor/acceptor). PRIMARY.
  ``exon``       exon-inclusion ratio = exon / sum(exons of gene) (PSI).
  ``transcript`` isoform-usage fraction = transcript RPKM / sum(gene transcript RPKM).

For each cis feature its per-sample splicing ratio is regressed on inversion-tag ALT dosage
plus covariates (sex, population, expression PCs); reports the dosage effect, direction,
analytic p, genome-wide BH-q, and a permuted-dosage empirical null. Deterministic given the
inputs and ``seed`` (default 42).
"""
from __future__ import annotations

import csv
import gzip

import numpy as np

from . import common as C

WINDOW = 1_000_000
MIN_SAMPLES = 80
MIN_CLUSTER_READS = 10
MIN_MEAN_RATIO = 0.02
MIN_RATIO_SD = 0.005
N_PERM = 200


def load_loci(inversions_tsv: str, window: int = WINDOW) -> list[dict]:
    rows = [r for r in csv.DictReader(open(inversions_tsv), delimiter="\t")
            if r.get("tag_pos19") and r.get("start19") and r.get("tag_status") == "ok"
            and r.get("tag_variantId")]
    for r in rows:
        r["s19"], r["e19"] = int(float(r["start19"])), int(float(r["end19"]))
        r["lo"] = min(r["s19"], r["e19"]) - window
        r["hi"] = max(r["s19"], r["e19"]) + window
    return rows


def feature_span(pheno: str, targetid: str, coord) -> tuple[int, int]:
    """hg19 (start, end) span of a matrix feature from its TargetID encoding."""
    if pheno == "junction":
        p = targetid.split("_")
        return int(p[1]), int(p[2])
    if pheno == "exon":
        p = targetid.rsplit("_", 2)
        return int(p[1]), int(p[2])
    c = int(float(coord))
    return c, c


def collect_cis(pheno: str, path: str, loci: list[dict]) -> tuple[list[str], list[dict]]:
    """Stream a splicing matrix once; keep features overlapping any cis window."""
    by_chrom: dict = {}
    for r in loci:
        by_chrom.setdefault(str(r["chrom_no"]), []).append(r)
    kept = []
    with gzip.open(path, "rt") as fh:
        samples = fh.readline().rstrip("\n").split("\t")[4:]
        for line in fh:
            p = line.rstrip("\n").split("\t")
            cand = by_chrom.get(p[2])
            if not cand:
                continue
            fs, fe = feature_span(pheno, p[0], p[3])
            hit = [rr["locus"] for rr in cand if not (fe < rr["lo"] or fs > rr["hi"])]
            if not hit:
                continue
            try:
                vals = np.array(p[4:], dtype=np.float64)
            except ValueError:
                continue
            kept.append({"targetid": p[0], "gene": p[1].split(".")[0], "chrom": p[2],
                         "fs": fs, "fe": fe, "vals": vals, "loci": hit})
    return samples, kept


def union_find_clusters(feats: list[dict]) -> dict:
    """Junctions in the same gene sharing a splice site (donor or acceptor) -> cluster id."""
    parent = list(range(len(feats)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_start: dict = {}
    by_end: dict = {}
    for i, f in enumerate(feats):
        by_start.setdefault((f["gene"], f["fs"]), []).append(i)
        by_end.setdefault((f["gene"], f["fe"]), []).append(i)
    for grp in list(by_start.values()) + list(by_end.values()):
        for k in range(1, len(grp)):
            union(grp[0], grp[k])
    clusters: dict = {}
    for i in range(len(feats)):
        clusters.setdefault(find(i), []).append(i)
    return clusters


def build_ratios(pheno: str, kept: list[dict]) -> list[dict]:
    """Per-sample splicing ratio vectors: junction clusters share a splice site; exon/
    transcript cluster by gene. Only multi-member clusters are testable."""
    recs = []
    keys = ("targetid", "gene", "chrom", "fs", "fe", "loci")
    if pheno == "junction":
        groups = list(union_find_clusters(kept).items())
    else:
        by_gene: dict = {}
        for i, f in enumerate(kept):
            by_gene.setdefault(f["gene"], []).append(i)
        groups = [(hash(g) & 0xffffffff, idxs) for g, idxs in by_gene.items()]
    for cid, idxs in groups:
        if len(idxs) < 2:
            continue
        mat = np.vstack([kept[i]["vals"] for i in idxs])
        csum = mat.sum(axis=0)
        for row, i in enumerate(idxs):
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(csum >= MIN_CLUSTER_READS, mat[row] / csum, np.nan)
            recs.append({**{k: kept[i][k] for k in keys},
                         "cluster": int(cid), "cluster_size": len(idxs), "ratio": ratio})
    return recs


def run_sqtl(pheno: str, matrix_path: str, inversions_tsv: str, pgen_prefix: str,
             gene_rpkm_path: str, panel_tsv: str, window: int = WINDOW, n_perm: int = N_PERM,
             n_pc: int = 10, seed: int = 42) -> tuple[list[dict], dict]:
    """Run the splicing-QTL scan for one phenotype. Returns (per-test rows sorted by p with
    BH-q attached, summary dict). Deterministic given inputs and ``seed``."""
    loci = load_loci(inversions_tsv, window)
    samples, kept = collect_cis(pheno, matrix_path, loci)
    psam, dos, _ = C.load_tag_dosages(pgen_prefix, loci)
    pos = {s: i for i, s in enumerate(psam)}
    keep_s = [j for j, s in enumerate(samples) if s in pos]
    samp2 = [samples[j] for j in keep_s]
    gidx = np.array([pos[s] for s in samp2])
    Cov, _, _ = C.build_covariates(gene_rpkm_path, panel_tsv, samp2, n_pc=n_pc)
    dosages = {loc: dos[loc][gidx] for loc in dos}

    recs = build_ratios(pheno, kept)
    rng = np.random.default_rng(seed)
    results = []
    for rec in recs:
        ratio_all = rec["ratio"][keep_s]
        for locus in rec["loci"]:
            d = dosages.get(locus)
            if d is None:
                continue
            m = np.isfinite(ratio_all) & np.isfinite(d)
            y = ratio_all[m]
            if m.sum() < MIN_SAMPLES or np.unique(d[m]).size < 2:
                continue
            if y.mean() < MIN_MEAN_RATIO or y.std() < MIN_RATIO_SD:
                continue
            beta, se, t, p, n = C.ols_beta_p(y, np.column_stack([d[m], Cov[m]]), 0)
            if not np.isfinite(p):
                continue
            results.append({"locus": locus, "gene": rec["gene"], "targetid": rec["targetid"],
                            "chrom": rec["chrom"], "fs": rec["fs"], "fe": rec["fe"],
                            "cluster_size": rec["cluster_size"], "beta": beta, "se": se, "t": t,
                            "p": p, "n": n, "mean_ratio": float(y.mean()),
                            "direction": "higher_with_alt" if beta > 0 else "lower_with_alt"})

    perm_p = []
    if results:
        for r, qq in zip(results, C.bh_fdr([r["p"] for r in results])):
            r["bh_q"] = float(qq)
        ratio_lookup = {(rec["targetid"], loc): rec["ratio"][keep_s]
                        for rec in recs for loc in rec["loci"]}
        for _ in range(n_perm):
            r = results[rng.integers(0, len(results))]
            ratio_all = ratio_lookup.get((r["targetid"], r["locus"]))
            d = dosages.get(r["locus"])
            if ratio_all is None or d is None:
                continue
            m = np.isfinite(ratio_all) & np.isfinite(d)
            dperm = rng.permutation(d[m])
            if np.unique(dperm).size < 2:
                continue
            _, _, _, pp, _ = C.ols_beta_p(ratio_all[m], np.column_stack([dperm, Cov[m]]), 0)
            if np.isfinite(pp):
                perm_p.append(pp)

    results.sort(key=lambda r: r["p"])
    n_sig = sum(1 for r in results if r.get("bh_q", 1) < 0.05)
    summary = {
        "pheno": pheno, "n_tests": len(results), "n_features": len(recs), "n_samples": len(samp2),
        "n_sig_bh05": n_sig,
        "n_loci_sig_bh05": len({r["locus"] for r in results if r.get("bh_q", 1) < 0.05}),
        "obs_frac_p05": float(np.mean([r["p"] < 0.05 for r in results])) if results else None,
        "perm_frac_p05": float(np.mean(np.array(perm_p) < 0.05)) if perm_p else None,
        "n_perm": len(perm_p),
    }
    return results, summary
