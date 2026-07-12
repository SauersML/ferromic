"""Shared machinery for the measured regulatory-QTL analyses (Geuvadis LCL RNA-seq).

Inversion dosage proxy = best tag-SNP ALT dosage (0/1/2) in the 462 Geuvadis LCL samples.
Covariates = intercept + sex + population one-hots + top-K genome-wide expression PCs. The
Geuvadis expression/splicing matrices are library-depth-normalised but NOT PEER-corrected, so
the expression PCs residualise hidden structure (a PEER proxy).

All model fits use ordinary least squares with a two-sided t-test on the dosage coefficient.
"""
from __future__ import annotations

import csv
import gzip

import numpy as np


def matrix_samples(path: str) -> list[str]:
    """Sample ids from a Geuvadis matrix header (columns 5+ are samples)."""
    with gzip.open(path, "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")[4:]


def load_panel(panel_tsv: str) -> dict:
    """1000G sample -> (pop, super_pop, sex)."""
    d = {}
    with open(panel_tsv) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            d[r["sample"]] = (r["pop"], r["super_pop"], r.get("gender", ""))
    return d


def expression_pcs(gene_rpkm_path: str, samples: list[str], n_pc: int) -> np.ndarray:
    """Top-``n_pc`` PCs of log2 gene RPKM (top-3000 variable genes), aligned to ``samples``."""
    esamp = matrix_samples(gene_rpkm_path)
    col = {s: i for i, s in enumerate(esamp)}
    rows = []
    with gzip.open(gene_rpkm_path, "rt") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            try:
                rows.append(np.array(p[4:], dtype=np.float64))
            except ValueError:
                continue
    X = np.vstack(rows)[:, np.array([col[s] for s in samples])]
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    X = np.log2(X + 1.0)
    keep = np.argsort(X.var(axis=1))[-3000:]
    X = X[keep]
    X = X - X.mean(axis=1, keepdims=True)
    # PCA over samples via symmetric eigendecomposition (robust vs SVD)
    w, V = np.linalg.eigh(X.T @ X)
    return V[:, np.argsort(w)[::-1][:n_pc]]


def build_covariates(gene_rpkm_path: str, panel_tsv: str, samples: list[str], n_pc: int = 10):
    """Design matrix ``[n_samples, n_cov]``: intercept + sex + population one-hots (drop one as
    reference) + top-``n_pc`` expression PCs. Returns ``(C, names, pops)``."""
    panel = load_panel(panel_tsv)
    pops = [panel.get(s, ("NA", "NA", ""))[0] for s in samples]
    sexes = [panel.get(s, ("NA", "NA", ""))[2] for s in samples]
    uniq_pops = sorted(p for p in set(pops) if p != "NA")
    cols = [np.ones(len(samples))]
    names = ["intercept"]
    cols.append(np.array([1.0 if x == "male" else 0.0 for x in sexes]))
    names.append("sex_male")
    for p in uniq_pops[1:]:  # drop the first population as reference
        cols.append(np.array([1.0 if pp == p else 0.0 for pp in pops]))
        names.append(f"pop_{p}")
    pcs = expression_pcs(gene_rpkm_path, samples, n_pc)
    for k in range(pcs.shape[1]):
        cols.append(pcs[:, k])
        names.append(f"exprPC{k + 1}")
    return np.column_stack(cols), names, np.array(pops)


def load_tag_dosages(pgen_prefix: str, loci_rows: list[dict]) -> tuple[list[str], dict, dict]:
    """For inversion rows (with ``tag_pos19`` and ``chrom_no``), return the psam sample order,
    ``{locus: dosage_vec}`` (ALT count 0/1/2, missing -> nan), and per-locus variant metadata.

    ``pgen_prefix`` is the path to ``<prefix>.pgen`` with ``.pvar`` / ``.psam`` alongside.
    """
    import pgenlib

    psam = [ln.split("\t")[0] for ln in open(pgen_prefix.replace(".pgen", ".psam")).read().splitlines()[1:]]
    n_samp = len(psam)
    want = {}
    for r in loci_rows:
        if not r.get("tag_pos19"):
            continue
        want[(str(r["chrom_no"]), int(float(r["tag_pos19"])))] = r["locus"]
    found = {}
    with open(pgen_prefix.replace(".pgen", ".pvar")) as fh:
        idx = -1
        for line in fh:
            if line.startswith("#"):
                continue
            idx += 1
            tab = line.split("\t", 5)
            key = (tab[0], int(tab[1]))
            if key in want:
                found[key] = (idx, tab[3], tab[4], want[key])
                if len(found) == len(want):
                    break
    pr = pgenlib.PgenReader(bytes(pgen_prefix, "utf8"), raw_sample_ct=n_samp)
    buf = np.empty(n_samp, dtype=np.int8)
    dos, meta = {}, {}
    for key, (vidx, ref, alt, locus) in found.items():
        pr.read(vidx, buf)
        d = buf.astype(np.float64).copy()
        d[d < 0] = np.nan
        dos[locus] = d
        meta[locus] = {"ref": ref, "alt": alt, "pos19": key[1], "chrom_no": key[0]}
    pr.close()
    return psam, dos, meta


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def ols_beta_p(y, X, col_test: int):
    """OLS of ``y`` on ``X``; return ``(beta, se, t, p, n)`` for ``col_test`` (two-sided t)."""
    from scipy import stats

    n = len(y)
    XtX = X.T @ X
    try:
        XtXi = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    dof = n - X.shape[1]
    if dof <= 0:
        return np.nan, np.nan, np.nan, np.nan, n
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(sigma2 * XtXi[col_test, col_test])
    if se == 0 or not np.isfinite(se):
        return float(beta[col_test]), np.nan, np.nan, np.nan, n
    t = beta[col_test] / se
    return float(beta[col_test]), float(se), float(t), float(2 * stats.t.sf(abs(t), dof)), n
