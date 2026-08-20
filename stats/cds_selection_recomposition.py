"""Calibration of the CDS-vs-intron selection test under the
composition-equal null, on the real data.

The permutation in stats/cds_selection_intron_control.py tests exchangeability
of orientation labels. The scientific null is weaker: orientation may affect
the AMOUNT of variation (demography) but not its CDS/intron COMPOSITION. A
permutation test of a ratio statistic under a non-exchangeable null is not
automatically calibrated, so this script measures its realized type-I error
directly on the observed data structure:

  For each testable gene, take the real haplotypes and their real segregating
  sites (CDS and intron pooled). Keep every site's carrier pattern -- all
  haplotype structure, LD, group sizes, missingness stay exactly as observed.
  Re-deal only the compartment LABEL of each segregating site: CDS with
  probability w_hat = (total CDS diffs)/(total diffs), intron otherwise.
  Under this recomposition the orientation groups differ in amount exactly as
  the real data do, but composition is orientation-free by construction: the
  null hypothesis of the selection test is TRUE.

  For each of B recomposed datasets, run the same statistic and the same
  locus-level label permutation, and record the one-sided p. The fraction of
  recompositions with p < alpha is the realized size of the test at alpha,
  gene by gene, with every dependency of the real data intact:
    - haplotype sharing between pairs (pairs are quadratic in haplotypes)
    - LD between sites (carrier patterns are never altered)
    - group-size asymmetry and missing haplotypes
    - discreteness of small counts

Run after cds_selection_intron_control.py in the same workdir.
Output: results/cds_selection_recomposition.tsv
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_selection_intron_control import (  # noqa: E402
    CDS_RE, INV_RE, encode, read_phy, load_gtf, intron_sites, quad_form_sums)

B_RECOMP = 300
N_PERM = 2000
ALPHAS = (0.01, 0.05, 0.10)
RNG_SEED = 77


def seg_pair_vectors(mat):
    """mat: (n_hap, n_sites) uint8. Returns (n_seg, n_pairs) uint8 mismatch
    indicators over segregating sites, plus upper-triangle pair index."""
    n = mat.shape[0]
    iu = np.triu_indices(n, 1)
    ok = mat != 255
    called = ok.all(axis=0)
    # segregating among called-or-not: any two called haplotypes differ
    vecs = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        okj = col != 255
        vals = np.unique(col[okj])
        if len(vals) < 2:
            continue
        mism = (col[iu[0]] != col[iu[1]]) & okj[iu[0]] & okj[iu[1]]
        if mism.any():
            vecs.append(mism.astype(np.uint8))
    if not vecs:
        return np.zeros((0, len(iu[0])), dtype=np.uint8), iu
    return np.vstack(vecs), iu


def pairs_to_matrix(vec, iu, n):
    D = np.zeros((n, n))
    D[iu] = vec
    return D + D.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    os.chdir(a.workdir)

    res = pd.read_csv("results/cds_selection_intron_control.tsv", sep="\t")
    testable = res[res["status"] == "OK"]
    tests = pd.read_csv("repo/data/gene_inversion_direct_inverted.tsv", sep="\t")
    tests = tests.merge(testable[["gene_name", "inv_id"]], on=["gene_name", "inv_id"])

    phy_dir = "phy_outputs"
    cds_files, region_files = {}, {}
    for fn in os.listdir(phy_dir):
        m = CDS_RE.match(fn)
        if m:
            cds_files[(m["grp"], m["gene"], m["enst"], m["chrom"],
                       int(m["is"]), int(m["ie"]))] = os.path.join(phy_dir, fn)
            continue
        m = INV_RE.match(fn)
        if m:
            region_files.setdefault(
                (m["chrom"], int(m["s"]), int(m["e"])), {})[m["grp"]] = \
                os.path.join(phy_dir, fn)

    chroms = {f"chr{t.split(':')[0]}" for t in tests["inv_id"]}
    all_exons, tx_exons = load_gtf("gencode.v47.basic.annotation.gtf.gz", chroms)

    # offsets were validated by the main run; recompute cheaply per locus by
    # trusting inv_start (the main run's calibration accepted it everywhere --
    # check its log if in doubt).
    by_locus = defaultdict(list)
    for _, r in tests.iterrows():
        by_locus[r["inv_id"]].append(r)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for inv_id, locus_tests in sorted(by_locus.items()):
        chrom_num, coords = inv_id.split(":")
        inv_s, inv_e = (int(x) for x in coords.split("-"))
        chrom = f"chr{chrom_num}"
        reg = region_files.get((chrom_num, inv_s, inv_e)) or \
            region_files.get((chrom, inv_s, inv_e))
        seq_dir = read_phy(reg["0"])
        seq_inv = read_phy(reg["1"])
        names = sorted(seq_dir) + sorted(seq_inv)
        is_inv_full = np.array([n in seq_inv for n in names])
        L = len(next(iter(seq_dir.values())))
        region = np.vstack([encode((seq_dir | seq_inv)[n]) for n in names])
        print(f"[{inv_id}] {len(locus_tests)} genes", flush=True)

        for r in locus_tests:
            key = (r["gene_name"], r["transcript_id"], chrom, inv_s, inv_e)
            f0, f1 = cds_files.get(("0", *key)), cds_files.get(("1", *key))
            cseq = read_phy(f0) | read_phy(f1)
            present = [n for n in names if n in cseq]
            idx = np.array([names.index(n) for n in present])
            inv_mask = is_inv_full[idx]
            n = len(present)
            k_inv = int(inv_mask.sum())
            cmat = np.vstack([encode(cseq[nm]) for nm in present])

            isites = intron_sites(r["transcript_id"], chrom, all_exons, tx_exons)
            cols = isites - (inv_s - 1)
            cols = cols[(cols >= 0) & (cols < L)]
            imat = region[np.ix_(idx, cols)]

            Vc, iu = seg_pair_vectors(cmat)
            Vi, _ = seg_pair_vectors(imat)
            allV = np.vstack([Vc, Vi]) if len(Vc) or len(Vi) else Vc
            n_cds_seg, n_all_seg = len(Vc), len(allV)
            if n_all_seg == 0:
                continue

            # pooled compartment weight from total mismatch mass
            w_hat = Vc.sum() / allV.sum() if len(Vc) else 0.0

            # locus deals for the permutation inside each recomposition
            P = np.zeros((N_PERM, n), dtype=np.float32)
            for d in range(N_PERM):
                P[d, rng.choice(n, k_inv, replace=False)] = 1.0

            v = inv_mask.astype(float)
            u = 1.0 - v
            p_vals = np.empty(B_RECOMP)
            S_vals = np.empty(B_RECOMP)
            for b in range(B_RECOMP):
                lab = rng.random(n_all_seg) < w_hat
                xa = allV[lab].sum(axis=0) if lab.any() else np.zeros(allV.shape[1])
                ma = allV[~lab].sum(axis=0) if (~lab).any() else np.zeros(allV.shape[1])
                Dc = pairs_to_matrix(xa, iu, n)
                Di = pairs_to_matrix(ma, iu, n)
                x_inv = v @ Dc @ v / 2; m_inv = v @ Di @ v / 2
                x_dir = u @ Dc @ u / 2; m_dir = u @ Di @ u / 2
                if (x_inv + m_inv) == 0 or (x_dir + m_dir) == 0:
                    p_vals[b] = np.nan; S_vals[b] = np.nan
                    continue
                S_obs = x_dir / (x_dir + m_dir) - x_inv / (x_inv + m_inv)
                x1, x0 = quad_form_sums(P, Dc)[:2]
                m1, m0 = quad_form_sums(P, Di)[:2]
                with np.errstate(invalid="ignore", divide="ignore"):
                    S_null = x0 / (x0 + m0) - x1 / (x1 + m1)
                valid = np.isfinite(S_null)
                p_vals[b] = ((S_null[valid] >= S_obs - 1e-12).sum() + 1) / \
                    (valid.sum() + 1)
                S_vals[b] = S_obs

            good = np.isfinite(p_vals)
            row = {"gene_name": r["gene_name"], "inv_id": inv_id,
                   "n_hap": n, "k_inv": k_inv,
                   "n_seg_sites": n_all_seg, "n_cds_seg": n_cds_seg,
                   "w_hat": w_hat, "B": int(good.sum())}
            for al in ALPHAS:
                row[f"size_at_{al}"] = float((p_vals[good] < al).mean())
            rows.append(row)
            print(f"  {r['gene_name']}: w={w_hat:.4f} seg={n_all_seg} "
                  f"size@0.05={row['size_at_0.05']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("results/cds_selection_recomposition.tsv", sep="\t", index=False)
    print("\n=========== CALIBRATION SUMMARY ===========")
    print(f"genes: {len(out)}")
    for al in ALPHAS:
        c = out[f"size_at_{al}"]
        print(f"alpha={al}: mean realized size {c.mean():.4f}  "
              f"(min {c.min():.3f}, max {c.max():.3f}, "
              f"genes over 2x nominal: {(c > 2 * al).sum()})")
    print("\nWrote results/cds_selection_recomposition.tsv")


if __name__ == "__main__":
    main()
