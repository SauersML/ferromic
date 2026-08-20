"""Selection test: is inverted-haplotype CDS conservation in excess of the
same haplotypes' non-coding background?

The permutation re-analysis (stats/per_gene_cds_permutation.py) established
that for some genes the inverted copies are more often identical than the
direct copies. That alone does not demonstrate selection: a young or frozen
inverted haplotype is more identical everywhere, coding or not. Selection on
coding sequence predicts something sharper: on inverted backgrounds, the
*share* of sequence differences falling in CDS should be depleted relative to
direct backgrounds, because constrained sites resist the mutations that
neutral intronic sites accumulate freely.

Statistic, per gene x inversion test:
  For every within-orientation haplotype pair, count nucleotide differences in
  the gene's CDS (x) and in its intronic sites (m) -- intronic = the tested
  transcript's span minus every GENCODE-basic exon on that chromosome. Pool
  over pairs within each orientation:
      share_g = sum(x) / sum(x + m)
      S = share_direct - share_inverted
  S > 0 means coding variation is depleted on inverted backgrounds relative to
  their own intronic background: purifying selection visible on the inverted
  haplotypes, with demography cancelled by construction (a uniformly frozen
  haplotype shifts x and m together, leaving the share alone).

Null: orientation labels are re-dealt across the locus's haplotypes (one deal
per locus per draw, shared by all genes at the locus, exactly as in
stats/cds_permutation_joint_control.py), and S is recomputed from the same
pair matrices. Pairs are never recomputed from sequence: within-group sums are
quadratic forms in the 0/1 assignment vector, so the whole null is matrix
algebra.

Genes where either orientation group carries zero differences over CDS and
introns combined are untestable (share undefined): a fully frozen haplotype
set contains no information to separate selection from youth, and the result
table says so rather than pretending otherwise.

Validation baked in: for each locus the CDS pair-difference matrix is
recomputed from the whole-region alignment's CDS columns and must equal the
matrix from the per-CDS alignment (pair hamming distance is strand-proof).
This pins the genome-position -> alignment-column offset before any intron
column is trusted.

Inputs (paths relative to --workdir):
  phy_outputs/            unpacked per-CDS and inversion_* whole-region .phy.gz
  gencode.v47.basic.annotation.gtf.gz
  repo/data/gene_inversion_direct_inverted.tsv
  repo/data/cds_identical_proportions.tsv
Output:
  results/cds_selection_intron_control.tsv and a printed summary.
"""

import argparse
import gzip
import itertools
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

N_DRAWS = 100_000
RNG_SEED = 2026

CDS_RE = re.compile(
    r"^group(?P<grp>[01])_(?P<gene>.+?)_(?P<ensg>ENSG[0-9.]+)_(?P<enst>ENST[0-9.]+)_"
    r"(?P<chrom>chr[^_]+)_cds_start(?P<cs>\d+)_cds_end(?P<ce>\d+)_"
    r"inv_start(?P<is>\d+)_inv_end(?P<ie>\d+)\.phy\.gz$"
)
INV_RE = re.compile(
    r"^inversion_group(?P<grp>[01])_(?P<chrom>[^_]+)_start(?P<s>\d+)_end(?P<e>\d+)\.phy\.gz$"
)

ENC = np.full(256, 255, dtype=np.uint8)
for i, b in enumerate(b"ACGT"):
    ENC[b] = i
    ENC[ord(chr(b).lower())] = i


def read_phy(path):
    seqs = {}
    with gzip.open(path, "rt") as fh:
        n_declared = int(fh.readline().split()[0])
        for line in fh:
            if not line.strip():
                continue
            name, seq = line.rstrip("\n").split(None, 1)
            seqs[name] = seq.replace(" ", "")
    if len(seqs) != n_declared:
        raise ValueError(f"{path}: header {n_declared} taxa, parsed {len(seqs)}")
    return seqs


def encode(seq):
    return ENC[np.frombuffer(seq.encode(), dtype=np.uint8)]


def pair_diff_matrix(mat):
    """mat: (n_hap, n_site) uint8, 255 = unusable. Returns (n,n) int64 counts of
    sites where both haplotypes are called and differ."""
    n = mat.shape[0]
    out = np.zeros((n, n), dtype=np.int64)
    ok = mat != 255
    for i in range(n):
        both = ok[i] & ok[i + 1:]
        out[i, i + 1:] = ((mat[i] != mat[i + 1:]) & both).sum(axis=1)
    return out + out.T


def merge_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def load_gtf(path, chroms):
    """Per chromosome: merged exon intervals (1-based inclusive) for ALL
    GENCODE-basic transcripts, plus per-transcript-base-ID exon lists."""
    exons_by_chrom = defaultdict(list)
    tx_exons = defaultdict(list)
    enst_re = re.compile(r'transcript_id "([^".]+)')
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if f[2] != "exon" or f[0] not in chroms:
                continue
            s, e = int(f[3]), int(f[4])
            exons_by_chrom[f[0]].append((s, e))
            m = enst_re.search(f[8])
            if m:
                tx_exons[m.group(1)].append((f[0], s, e))
    return ({c: merge_intervals(v) for c, v in exons_by_chrom.items()}, tx_exons)


def intron_sites(tx_id, chrom, all_exons, tx_exons):
    """0-based genome positions: transcript span minus every basic exon."""
    ex = [x for x in tx_exons.get(tx_id.split(".")[0], []) if x[0] == chrom]
    if not ex:
        return None
    span_s = min(s for _, s, _ in ex)
    span_e = max(e for _, _, e in ex)
    keep = np.ones(span_e - span_s + 1, dtype=bool)
    for s, e in all_exons.get(chrom, []):
        if e < span_s or s > span_e:
            continue
        keep[max(s, span_s) - span_s: min(e, span_e) - span_s + 1] = False
    return np.nonzero(keep)[0] + (span_s - 1)  # to 0-based


def quad_form_sums(V, D):
    """V: (draws, n) 0/1. Within-group-1 and within-group-0 pair sums of D."""
    tot = D.sum() / 2.0
    VD = V @ D
    in1 = (VD * V).sum(axis=1) / 2.0
    rowsum = D.sum(axis=1)
    cross = (VD * (1 - V)).sum(axis=1)
    in0 = tot - in1 - cross
    del VD
    return in1, in0, rowsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    os.chdir(args.workdir)
    os.makedirs("results", exist_ok=True)

    tests = pd.read_csv("repo/data/gene_inversion_direct_inverted.tsv", sep="\t")
    cip = pd.read_csv("repo/data/cds_identical_proportions.tsv", sep="\t")
    cip["inv_id"] = (cip["chr"].astype(str).str.replace("chr", "", regex=False)
                     + ":" + cip["inv_start"].astype(int).astype(str)
                     + "-" + cip["inv_end"].astype(int).astype(str))
    recurrence = (cip.groupby("inv_id")["consensus"].first()
                  .map({0: "single-event", 1: "recurrent"}))

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
    print(f"CDS files: {len(cds_files)}; whole-region loci: {len(region_files)}")

    chroms = {f"chr{t.split(':')[0]}" for t in tests["inv_id"]}
    all_exons, tx_exons = load_gtf("gencode.v47.basic.annotation.gtf.gz", chroms)
    print(f"GTF loaded: exon maps for {len(all_exons)} chroms, "
          f"{len(tx_exons)} transcripts")

    by_locus = defaultdict(list)
    for _, r in tests.iterrows():
        by_locus[r["inv_id"]].append(r)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    locus_null_shares = {}   # inv_id -> (draws,) array of per-draw S for FWER rows

    for inv_id, locus_tests in sorted(by_locus.items()):
        chrom_num, coords = inv_id.split(":")
        inv_s, inv_e = (int(x) for x in coords.split("-"))
        chrom = f"chr{chrom_num}"
        reg = region_files.get((chrom_num, inv_s, inv_e))
        if reg is None:
            reg = region_files.get((chrom, inv_s, inv_e))
        if not reg or set(reg) != {"0", "1"}:
            for r in locus_tests:
                rows.append({"gene_name": r["gene_name"], "inv_id": inv_id,
                             "status": "MISSING_REGION_PHY"})
            continue

        seq_dir = read_phy(reg["0"])
        seq_inv = read_phy(reg["1"])
        names = sorted(seq_dir) + sorted(seq_inv)
        is_inv_full = np.array([n in seq_inv for n in names])
        L = len(next(iter(seq_dir.values())))
        assert all(len(s) == L for s in
                   itertools.chain(seq_dir.values(), seq_inv.values()))
        span_matches = (L == inv_e - inv_s + 1) or (L == inv_e - inv_s)
        region = np.vstack([encode((seq_dir | seq_inv)[n]) for n in names])
        print(f"\n[{inv_id}] region {L} cols, {len(names)} haplotypes, "
              f"span_delta={L - (inv_e - inv_s + 1)}", flush=True)

        # -- calibrate genome->column offset with a plus-strand CDS check ----
        offset = None
        for r in locus_tests:
            key = (r["gene_name"], r["transcript_id"], chrom, inv_s, inv_e)
            f0, f1 = cds_files.get(("0", *key)), cds_files.get(("1", *key))
            if not (f0 and f1):
                continue
            cseq = read_phy(f0) | read_phy(f1)
            common = [n for n in names if n in cseq]
            if len(common) < 3:
                continue
            cmat = np.vstack([encode(cseq[n]) for n in common])
            cds_D_true = pair_diff_matrix(cmat)
            ex = [x for x in tx_exons.get(r["transcript_id"].split(".")[0], [])
                  if x[0] == chrom]
            # candidate CDS columns from GTF CDS records are not loaded;
            # use the filename CDS span intersected with transcript exons.
            cs, ce = None, None
            for k, path in cds_files.items():
                if k[:1] == ("0",) and k[1] == r["gene_name"] and k[2] == r["transcript_id"]:
                    mm = CDS_RE.match(os.path.basename(path))
                    cs, ce = int(mm["cs"]), int(mm["ce"])
            if cs is None or not ex:
                continue
            cds_cols_1based = []
            for _, s, e in sorted(ex, key=lambda t: t[1]):
                s2, e2 = max(s, cs), min(e, ce)
                if s2 <= e2:
                    cds_cols_1based.extend(range(s2, e2 + 1))
            cds_cols_1based = np.asarray(cds_cols_1based)
            if len(cds_cols_1based) != cmat.shape[1]:
                continue  # exon structure disagrees with alignment; try next gene
            idx_common = [names.index(n) for n in common]
            for cand in (inv_s, inv_s + 1, inv_s - 1):
                cols = cds_cols_1based - 1 - (cand - 1)  # 0-based columns
                if cols.min() < 0 or cols.max() >= L:
                    continue
                D_try = pair_diff_matrix(region[np.ix_(idx_common, cols)])
                if np.array_equal(D_try, cds_D_true):
                    offset = cand
                    break
            if offset is not None:
                print(f"  offset calibrated on {r['gene_name']}: "
                      f"region col 0 = genome pos {offset} (1-based)")
                break
        if offset is None:
            print(f"  !! could not calibrate offset; skipping locus")
            for r in locus_tests:
                rows.append({"gene_name": r["gene_name"], "inv_id": inv_id,
                             "status": "OFFSET_UNCALIBRATED"})
            continue

        # -- shared null deals for the locus ---------------------------------
        n_hap = len(names)
        k_inv_locus = int(is_inv_full.sum())
        V = np.zeros((N_DRAWS, n_hap), dtype=np.float32)
        for d in range(N_DRAWS):
            V[d, rng.choice(n_hap, k_inv_locus, replace=False)] = 1.0

        locus_S_null = []

        for r in locus_tests:
            key = (r["gene_name"], r["transcript_id"], chrom, inv_s, inv_e)
            f0, f1 = cds_files.get(("0", *key)), cds_files.get(("1", *key))
            if not (f0 and f1):
                rows.append({"gene_name": r["gene_name"], "inv_id": inv_id,
                             "status": "MISSING_PHY"})
                continue
            cseq = read_phy(f0) | read_phy(f1)
            present = [n for n in names if n in cseq]
            idx = np.array([names.index(n) for n in present])
            inv_mask = is_inv_full[idx]
            cmat = np.vstack([encode(cseq[n]) for n in present])
            Dc = pair_diff_matrix(cmat).astype(np.float64)

            isites = intron_sites(r["transcript_id"], chrom, all_exons, tx_exons)
            if isites is None:
                rows.append({"gene_name": r["gene_name"], "inv_id": inv_id,
                             "status": "NO_GTF_TRANSCRIPT"})
                continue
            cols = isites - (offset - 1)
            cols = cols[(cols >= 0) & (cols < L)]
            Di = pair_diff_matrix(region[np.ix_(idx, cols)]).astype(np.float64)

            v_obs = inv_mask.astype(np.float64)
            x_inv = float(v_obs @ Dc @ v_obs) / 2.0
            m_inv = float(v_obs @ Di @ v_obs) / 2.0
            u = 1.0 - v_obs
            x_dir = float(u @ Dc @ u) / 2.0
            m_dir = float(u @ Di @ u) / 2.0

            base = {
                "gene_name": r["gene_name"], "transcript_id": r["transcript_id"],
                "inv_id": inv_id, "recurrence": recurrence.get(inv_id, "unknown"),
                "k_direct": int((~inv_mask).sum()), "k_inverted": int(inv_mask.sum()),
                "n_intron_sites": int(len(cols)), "n_cds_sites": int(cmat.shape[1]),
                "cds_diffs_dir": x_dir, "intron_diffs_dir": m_dir,
                "cds_diffs_inv": x_inv, "intron_diffs_inv": m_inv,
            }
            if (x_inv + m_inv) == 0 or (x_dir + m_dir) == 0:
                rows.append(base | {"status": "UNTESTABLE_NO_VARIATION"})
                continue

            share_inv = x_inv / (x_inv + m_inv)
            share_dir = x_dir / (x_dir + m_dir)
            S_obs = share_dir - share_inv

            Vg = V[:, idx]
            x1, x0, _ = quad_form_sums(Vg, Dc)
            m1, m0, _ = quad_form_sums(Vg, Di)
            with np.errstate(invalid="ignore", divide="ignore"):
                S_null = (x0 / (x0 + m0)) - (x1 / (x1 + m1))
            valid = np.isfinite(S_null)
            nv = int(valid.sum())
            p_two = (np.abs(S_null[valid]) >= abs(S_obs) - 1e-12).sum() + 1
            p_two = p_two / (nv + 1)
            p_sel = ((S_null[valid] >= S_obs - 1e-12).sum() + 1) / (nv + 1)
            locus_S_null.append((r["gene_name"], S_null))
            rows.append(base | {
                "share_dir": share_dir, "share_inv": share_inv, "S": S_obs,
                "p_two_sided": p_two, "p_selection_onesided": p_sel,
                "n_valid_draws": nv, "status": "OK",
            })
            print(f"  {r['gene_name']}: x/m dir {x_dir:.0f}/{m_dir:.0f} "
                  f"inv {x_inv:.0f}/{m_inv:.0f}  S={S_obs:+.4f} "
                  f"p2={p_two:.4g} p_sel={p_sel:.4g}", flush=True)

        if locus_S_null:
            locus_null_shares[inv_id] = locus_S_null
        del V, region

    out = pd.DataFrame(rows)
    ok = out["status"].eq("OK")

    # BY across testable genes (arbitrary dependence)
    def by_q(p):
        p = np.asarray(p, float)
        n = len(p)
        c = sum(1.0 / k for k in range(1, n + 1))
        order = np.argsort(p)
        q = np.empty(n)
        running = 1.0
        for j, i in enumerate(order[::-1]):
            running = min(running, p[i] * n * c / (n - j))
            q[i] = running
        return q

    if ok.any():
        out.loc[ok, "q_by_two_sided"] = by_q(out.loc[ok, "p_two_sided"].to_numpy())
        out.loc[ok, "q_by_selection"] = by_q(
            out.loc[ok, "p_selection_onesided"].to_numpy())
    out = out.sort_values(["status", "p_selection_onesided"], na_position="last")
    out.to_csv("results/cds_selection_intron_control.tsv", sep="\t", index=False)

    d = out[ok]
    print("\n================ SUMMARY ================")
    print(f"tests: {len(out)}; testable: {len(d)}; "
          f"untestable(frozen): {int(out.status.eq('UNTESTABLE_NO_VARIATION').sum())}; "
          f"other skips: {int((~ok).sum() - out.status.eq('UNTESTABLE_NO_VARIATION').sum())}")
    if len(d):
        print(f"S>0 (CDS-depleted on inverted background): {int((d.S > 0).sum())}/{len(d)}")
        print(f"selection one-sided p<0.05: {int((d.p_selection_onesided < 0.05).sum())}")
        print(f"selection BY q<0.05: {int((d.q_by_selection < 0.05).sum())}")
        print(f"two-sided BY q<0.05: {int((d.q_by_two_sided < 0.05).sum())}")
        print("\nby recurrence:")
        print(d.groupby("recurrence")
              .agg(n=("S", "size"), S_med=("S", "median"),
                   p_lt05=("p_selection_onesided", lambda p: int((p < .05).sum())))
              .to_string())
        print("\ntop 15 by one-sided selection p:")
        cols = ["gene_name", "inv_id", "recurrence", "k_inverted",
                "cds_diffs_inv", "intron_diffs_inv", "share_dir", "share_inv",
                "S", "p_selection_onesided", "q_by_selection"]
        print(d[cols].head(15).to_string(index=False))
    print("\nWrote results/cds_selection_intron_control.tsv")


if __name__ == "__main__":
    main()
