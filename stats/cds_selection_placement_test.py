"""Placement-based selection test: corrected inference for the CDS-vs-intron
contrast.

The recomposition calibration (stats/cds_selection_recomposition.py) showed
the label-permutation p-values of stats/cds_selection_intron_control.py are
anti-conservative (mean realized size ~0.16 at alpha 0.05): permuting
orientation labels tests exchangeability, but under the demography-only null
the groups differ in variation AMOUNT, and the share statistic's null spread
is wider than label permutation implies.

This script replaces the reference distribution with one that holds the
orientation labels FIXED (amounts stay exactly as observed) and instead asks:
is the real CDS footprint special, compared to a same-shaped random footprint?

  For each gene: keep the real haplotypes, labels, and every segregating
  site. Draw B random placements of pseudo-exons -- the real exon fragment
  lengths, placed without overlap at uniform positions along the gene's
  analyzable footprint (own CDS + intronic sites, concatenated coordinate).
  Each placement defines a pseudo-CDS compartment of identical bp size and
  block structure; the rest is pseudo-intron. Recompute S for each placement.

  p_place = (1 + #{S_placement >= S_obs}) / (B + 1)

This preserves between-group amount differences (labels never move), site
contiguity and LD (blocks move as blocks), pair dependence (sites carry their
full carrier patterns), and compartment footprint size. Under the null that
coding position is not special, the real exon placement is one draw among B.

Output: results/cds_selection_placement.tsv
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_selection_intron_control import (  # noqa: E402
    CDS_RE, INV_RE, encode, read_phy, load_gtf, intron_sites)

B_PLACE = 2000
RNG_SEED = 99


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

            # analyzable footprint, concatenated: own-CDS exon fragments
            # (from GTF, as in the main run's offset calibration) + introns
            ex = [x for x in tx_exons.get(r["transcript_id"].split(".")[0], [])
                  if x[0] == chrom]
            cs = ce = None
            mm = CDS_RE.match(os.path.basename(f0))
            cs, ce = int(mm["cs"]), int(mm["ce"])
            frag_bounds = []      # (genome_start, genome_end) CDS fragments
            for _, s, e in sorted(ex, key=lambda t: t[1]):
                s2, e2 = max(s, cs), min(e, ce)
                if s2 <= e2:
                    frag_bounds.append((s2, e2))
            frag_lens = [e - s + 1 for s, e in frag_bounds]
            cds_pos = np.concatenate(
                [np.arange(s - 1, e) for s, e in frag_bounds])  # 0-based
            ipos = intron_sites(r["transcript_id"], chrom, all_exons, tx_exons)
            universe = np.concatenate([cds_pos, ipos])
            universe.sort()
            cols = universe - (inv_s - 1)
            keep = (cols >= 0) & (cols < L)
            universe = universe[keep]
            cols = cols[keep]
            U = len(universe)
            total_cds_len = sum(frag_lens)
            if U <= total_cds_len + 10:
                continue

            mat = region[np.ix_(idx, cols)]
            # per-site mismatch mass within each orientation group
            v = inv_mask
            def group_site_diffs(mask):
                sub = mat[mask]
                k = sub.shape[0]
                ok = sub != 255
                diffs = np.zeros(mat.shape[1])
                # pair-sum per site via allele counting
                for allele in range(4):
                    c = ((sub == allele) & ok).sum(axis=0)
                    diffs += c * (ok.sum(axis=0) - c)
                return diffs / 2.0
            d_inv = group_site_diffs(v)
            d_dir = group_site_diffs(~v)

            # observed compartment: first len(cds_pos∩kept) coordinates of
            # universe that belong to CDS fragments
            cds_set = np.isin(universe, cds_pos)
            x_inv, m_inv = d_inv[cds_set].sum(), d_inv[~cds_set].sum()
            x_dir, m_dir = d_dir[cds_set].sum(), d_dir[~cds_set].sum()
            S_obs = x_dir / (x_dir + m_dir) - x_inv / (x_inv + m_inv)

            # placements: real fragment lengths at random non-overlapping
            # offsets along the concatenated universe coordinate
            tot_inv = d_inv.sum(); tot_dir = d_dir.sum()
            S_place = np.empty(B_PLACE)
            gap_total = U - total_cds_len
            nf = len(frag_lens)
            for b in range(B_PLACE):
                cuts = np.sort(rng.choice(gap_total + nf, nf, replace=False))
                gaps = np.diff(np.concatenate([[0], cuts])) - \
                    np.arange(nf) * 0  # spacing via stars-and-bars
                # stars-and-bars: positions of fragments in concatenated line
                starts = cuts - np.arange(nf)  # gap mass before each fragment
                pos0 = starts + np.concatenate(
                    [[0], np.cumsum(frag_lens[:-1])]).astype(np.int64)
                sel = np.zeros(U, dtype=bool)
                for st, ln in zip(pos0, frag_lens):
                    sel[int(st):int(st) + int(ln)] = True
                xb_i = d_inv[sel].sum(); xb_d = d_dir[sel].sum()
                with np.errstate(invalid="ignore", divide="ignore"):
                    S_place[b] = (xb_d / tot_dir) - (xb_i / tot_inv) \
                        if tot_dir > 0 and tot_inv > 0 else np.nan
            good = np.isfinite(S_place)
            p_place = ((S_place[good] >= S_obs - 1e-12).sum() + 1) / \
                (good.sum() + 1)
            p_place_two = ((np.abs(S_place[good]) >= abs(S_obs) - 1e-12).sum()
                           + 1) / (good.sum() + 1)
            rows.append({
                "gene_name": r["gene_name"], "inv_id": inv_id,
                "recurrence": r.get("recurrence", ""),
                "n_hap": n, "k_inv": int(inv_mask.sum()),
                "S_obs": S_obs, "p_place_onesided": p_place,
                "p_place_twosided": p_place_two,
                "x_inv": x_inv, "m_inv": m_inv, "x_dir": x_dir, "m_dir": m_dir,
                "B": int(good.sum()),
            })
            print(f"  {r['gene_name']}: S={S_obs:+.4f} p_place={p_place:.4g}",
                  flush=True)

    out = pd.DataFrame(rows)

    def by_q(p):
        p = np.asarray(p, float)
        nn = len(p)
        c = sum(1.0 / k for k in range(1, nn + 1))
        order = np.argsort(p)
        q = np.empty(nn)
        running = 1.0
        for j, i in enumerate(order[::-1]):
            running = min(running, p[i] * nn * c / (nn - j))
            q[i] = running
        return q

    out["q_by_place"] = by_q(out["p_place_onesided"].to_numpy())
    out["q_by_place_two"] = by_q(out["p_place_twosided"].to_numpy())
    out = out.sort_values("p_place_onesided")
    out.to_csv("results/cds_selection_placement.tsv", sep="\t", index=False)
    print("\n========== PLACEMENT TEST SUMMARY ==========")
    print(f"genes: {len(out)}")
    print(f"one-sided p<0.05: {(out.p_place_onesided < 0.05).sum()}; "
          f"BY q<0.05: {(out.q_by_place < 0.05).sum()}")
    print(f"two-sided BY q<0.05: {(out.q_by_place_two < 0.05).sum()}")
    print("\ntop 12:")
    print(out[["gene_name", "inv_id", "S_obs", "p_place_onesided",
               "q_by_place"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
