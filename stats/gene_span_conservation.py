"""Whole-gene refit of the CDS-conservation comparison.

Reviewer/editor question behind this script: the CDS conservation contrast
(single-event inverted haplotypes are more often identical across a coding
sequence than the other three orientation-by-recurrence groups) lost its
significance once orientation labels were permuted at the inversion level
(permutation p = 0.17). Is that a power problem caused by coding sequences
being short and conserved -- would refitting the same model on the ENTIRE GENE
sequence (exons plus introns plus UTRs, i.e. the transcript span) recover the
signal?

This script answers it by rebuilding the same statistic on the transcript span
instead of the CDS, using the whole-locus haplotype alignments:

  For every tested gene x inversion:
    - gene-span columns  = min(exon start) .. max(exon end) of the tested
      transcript, mapped into the whole-region alignment through the offset
      that is calibrated (and verified) against the per-CDS alignment;
    - within each orientation group, count haplotype pairs that are identical
      across all callable gene-span sites -> prop_identical_gene, the exact
      analogue of prop_identical_pairs in data/cds_identical_proportions.tsv;
    - also record mean pairwise differences per callable site (pi_gene), which
      unlike the identity proportion does not saturate once the window is long.

Both quantities are reported per orientation so that stage 2
(stats/gene_span_conservation_model.py) can refit the orientation-by-recurrence
model on the gene span and permute orientation labels at the inversion level,
exactly as was done for the CDS.

The genome-position -> alignment-column offset is calibrated per locus by
requiring the CDS pair-difference matrix recomputed from the region alignment
to equal the matrix from the standalone per-CDS alignment; a locus whose offset
cannot be verified is reported as such rather than silently trusted. This is
the same guard used by stats/cds_selection_intron_control.py.

Inputs (paths relative to --workdir):
  phy_outputs/                       per-CDS and inversion_* whole-region .phy.gz
  gencode.v47.basic.annotation.gtf.gz
  repo/data/gene_inversion_direct_inverted.tsv
  repo/data/cds_identical_proportions.tsv
Output:
  results/gene_span_conservation.tsv
"""

import argparse
import gzip
import itertools
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

CDS_RE = re.compile(
    r"^group(?P<grp>[01])_(?P<gene>.+?)_(?P<ensg>ENSG[0-9.]+)_(?P<enst>ENST[0-9.]+)_"
    r"(?P<chrom>chr[^_]+)_cds_start(?P<cs>\d+)_cds_end(?P<ce>\d+)_"
    r"inv_start(?P<is>\d+)_inv_end(?P<ie>\d+)\.phy\.gz$"
)
INV_RE = re.compile(
    r"^inversion_group(?P<grp>[01])_(?P<chrom>[^_]+)_start(?P<s>\d+)_end(?P<e>\d+)\.phy\.gz$"
)

ENC = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    ENC[_b] = _i
    ENC[ord(chr(_b).lower())] = _i


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


def pair_stats(mat):
    """mat: (n_hap, n_site) uint8, 255 = unusable.

    Returns (diffs, callable) each (n,n) int64: number of sites where both
    haplotypes are called and differ, and number of sites where both are called.
    """
    n = mat.shape[0]
    d = np.zeros((n, n), dtype=np.int64)
    c = np.zeros((n, n), dtype=np.int64)
    ok = mat != 255
    for i in range(n):
        both = ok[i] & ok[i + 1:]
        c[i, i + 1:] = both.sum(axis=1)
        d[i, i + 1:] = ((mat[i] != mat[i + 1:]) & both).sum(axis=1)
    return d + d.T, c + c.T


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
    """Per chromosome merged exon intervals plus per-transcript exon lists."""
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


def gene_span(tx_id, chrom, tx_exons):
    """1-based inclusive transcript span (first exon start .. last exon end)."""
    ex = [x for x in tx_exons.get(tx_id.split(".")[0], []) if x[0] == chrom]
    if not ex:
        return None
    return min(s for _, s, _ in ex), max(e for _, _, e in ex)


def _cds_block_lengths(tx_id, chrom, tx_exons, cds_files, gene_name):
    """Lengths of the coding blocks of the tested transcript, in bp."""
    ex = [x for x in tx_exons.get(tx_id.split(".")[0], []) if x[0] == chrom]
    if not ex:
        return None
    cs = ce = None
    for k, path in cds_files.items():
        if k[0] == "0" and k[1] == gene_name and k[2] == tx_id:
            mm = CDS_RE.match(os.path.basename(path))
            cs, ce = int(mm["cs"]), int(mm["ce"])
            break
    if cs is None:
        return None
    lens = []
    for _, s, e in sorted(ex, key=lambda t: t[1]):
        s2, e2 = max(s, cs), min(e, ce)
        if s2 <= e2:
            lens.append(e2 - s2 + 1)
    return lens or None


def placebo_columns(span_cols, exon_mask, block_lengths, rng, tries=40):
    """Relocate the CDS block structure to random non-exonic positions.

    `span_cols` are the gene-span columns of the region alignment, `exon_mask`
    marks which of them fall in a GENCODE-basic exon, and `block_lengths` are
    the lengths of the real CDS exon blocks. Each placebo draw lays the same
    blocks down at random offsets that avoid real exons and each other, so the
    only thing that changes relative to the real test is WHERE in the gene the
    window sits: same number of sites, same block structure, same haplotypes.
    Returns None when the gene has too little non-exonic room.
    """
    n = len(span_cols)
    free = ~exon_mask
    chosen = np.zeros(n, dtype=bool)
    for L in sorted(block_lengths, reverse=True):
        if L <= 0 or L > n:
            return None
        placed = False
        for _ in range(tries):
            off = int(rng.integers(0, n - L + 1))
            sl = slice(off, off + L)
            if free[sl].all() and not chosen[sl].any():
                chosen[sl] = True
                placed = True
                break
        if not placed:
            return None
    return span_cols[chosen]


def group_summary(D, C, mask):
    """Identity/divergence summary for the sub-block selected by `mask`."""
    idx = np.nonzero(mask)[0]
    k = len(idx)
    if k < 2:
        return dict(k=k, n_pairs=0, n_identical=0, prop_identical=np.nan,
                    mean_diffs=np.nan, mean_callable=np.nan, pi=np.nan)
    sub_d = D[np.ix_(idx, idx)]
    sub_c = C[np.ix_(idx, idx)]
    iu = np.triu_indices(k, 1)
    d = sub_d[iu].astype(float)
    c = sub_c[iu].astype(float)
    usable = c > 0
    n_pairs = int(usable.sum())
    if n_pairs == 0:
        return dict(k=k, n_pairs=0, n_identical=0, prop_identical=np.nan,
                    mean_diffs=np.nan, mean_callable=np.nan, pi=np.nan)
    n_ident = int((d[usable] == 0).sum())
    return dict(
        k=k,
        n_pairs=n_pairs,
        n_identical=n_ident,
        prop_identical=n_ident / n_pairs,
        mean_diffs=float(d[usable].mean()),
        mean_callable=float(c[usable].mean()),
        pi=float(d[usable].sum() / c[usable].sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", default="results/gene_span_conservation.tsv")
    ap.add_argument("--placebo", type=int, default=0,
                    help="number of pseudo-CDS placebo draws per gene; each "
                         "draw relocates the CDS block structure to random "
                         "intronic positions inside the same gene span")
    ap.add_argument("--placebo-out",
                    default="results/gene_span_conservation_placebo.tsv")
    args = ap.parse_args()
    os.chdir(args.workdir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cip = pd.read_csv("repo/data/cds_identical_proportions.tsv", sep="\t")
    cip["inv_id"] = (cip["chr"].astype(str).str.replace("chr", "", regex=False)
                     + ":" + cip["inv_start"].astype(int).astype(str)
                     + "-" + cip["inv_end"].astype(int).astype(str))
    cip = cip[cip["consensus"].isin([0, 1])]
    recurrence = (cip.groupby("inv_id")["consensus"].first()
                  .map({0: "single-event", 1: "recurrent"}))

    # Test list = every gene x inversion stratum for which BOTH orientations are
    # present, i.e. exactly the strata the published CDS model is fit on. Using
    # the model's own input (rather than the smaller per-gene permutation list)
    # keeps the whole-gene refit like-for-like with the published fit.
    paired = (cip.groupby(["gene_name", "transcript_id", "inv_id"])["phy_group"]
              .nunique())
    tests = (paired[paired == 2].reset_index()[
        ["gene_name", "transcript_id", "inv_id"]])
    print(f"paired gene x inversion strata: {len(tests)} across "
          f"{tests['inv_id'].nunique()} inversions", flush=True)

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
    print(f"CDS files: {len(cds_files)}; whole-region loci: {len(region_files)}",
          flush=True)

    chroms = {f"chr{t.split(':')[0]}" for t in tests["inv_id"]}
    all_exons, tx_exons = load_gtf("gencode.v47.basic.annotation.gtf.gz", chroms)
    print(f"GTF: {len(all_exons)} chroms, {len(tx_exons)} transcripts", flush=True)

    by_locus = defaultdict(list)
    for _, r in tests.iterrows():
        by_locus[r["inv_id"]].append(r)

    rows = []
    placebo_rows = []
    placebo_rng = np.random.default_rng(2026)
    for inv_id, locus_tests in sorted(by_locus.items()):
        chrom_num, coords = inv_id.split(":")
        inv_s, inv_e = (int(x) for x in coords.split("-"))
        chrom = f"chr{chrom_num}"
        reg = region_files.get((chrom_num, inv_s, inv_e)) or \
            region_files.get((chrom, inv_s, inv_e))
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
        region = np.vstack([encode((seq_dir | seq_inv)[n]) for n in names])
        print(f"\n[{inv_id}] region {L} cols, {len(names)} haplotypes "
              f"({int(is_inv_full.sum())} inverted)", flush=True)

        # ---- calibrate the genome -> alignment-column offset on a CDS -------
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
            cds_D_true, _ = pair_stats(cmat)
            ex = [x for x in tx_exons.get(r["transcript_id"].split(".")[0], [])
                  if x[0] == chrom]
            cs = ce = None
            for k, path in cds_files.items():
                if k[0] == "0" and k[1] == r["gene_name"] and k[2] == r["transcript_id"]:
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
                continue
            idx_common = [names.index(n) for n in common]
            for cand in (inv_s, inv_s + 1, inv_s - 1):
                cols = cds_cols_1based - 1 - (cand - 1)
                if cols.min() < 0 or cols.max() >= L:
                    continue
                D_try, _ = pair_stats(region[np.ix_(idx_common, cols)])
                if np.array_equal(D_try, cds_D_true):
                    offset = cand
                    break
            if offset is not None:
                print(f"  offset calibrated on {r['gene_name']}: "
                      f"region col 0 = genome pos {offset}", flush=True)
                break
        if offset is None:
            print("  !! offset uncalibrated; skipping locus", flush=True)
            for r in locus_tests:
                rows.append({"gene_name": r["gene_name"], "inv_id": inv_id,
                             "status": "OFFSET_UNCALIBRATED"})
            continue

        # ---- per-gene whole-span statistics ---------------------------------
        for r in locus_tests:
            span = gene_span(r["transcript_id"], chrom, tx_exons)
            if span is None:
                rows.append({"gene_name": r["gene_name"], "inv_id": inv_id,
                             "status": "NO_GTF_TRANSCRIPT"})
                continue
            span_s, span_e = span

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

            cols = np.arange(span_s, span_e + 1) - offset
            in_region = (cols >= 0) & (cols < L)
            cols = cols[in_region]
            if len(cols) == 0:
                rows.append({"gene_name": r["gene_name"], "inv_id": inv_id,
                             "status": "SPAN_OUTSIDE_REGION"})
                continue

            span_block = region[np.ix_(idx, cols)]
            callable_frac = float((span_block != 255).mean())
            Dg, Cg = pair_stats(span_block)
            cmat = np.vstack([encode(cseq[n]) for n in present])
            Dc, Cc = pair_stats(cmat)

            g_dir = group_summary(Dg, Cg, ~inv_mask)
            g_inv = group_summary(Dg, Cg, inv_mask)
            c_dir = group_summary(Dc, Cc, ~inv_mask)
            c_inv = group_summary(Dc, Cc, inv_mask)

            # ---- placebo: the same window size and block structure, moved ----
            if args.placebo:
                exon_mask = np.zeros(len(cols), dtype=bool)
                genome_pos = cols + offset          # 1-based genome coordinate
                for es, ee in all_exons.get(chrom, []):
                    if ee < span_s or es > span_e:
                        continue
                    exon_mask |= (genome_pos >= es) & (genome_pos <= ee)
                cds_blocks = _cds_block_lengths(
                    r["transcript_id"], chrom, tx_exons,
                    cds_files, r["gene_name"])
                if cds_blocks:
                    for draw in range(args.placebo):
                        pc = placebo_columns(cols, exon_mask, cds_blocks,
                                             placebo_rng)
                        if pc is None:
                            break
                        Dp, Cp = pair_stats(region[np.ix_(idx, pc)])
                        p_dir = group_summary(Dp, Cp, ~inv_mask)
                        p_inv = group_summary(Dp, Cp, inv_mask)
                        placebo_rows.append({
                            "gene_name": r["gene_name"],
                            "transcript_id": r["transcript_id"],
                            "inv_id": inv_id,
                            "recurrence": recurrence.get(inv_id, "unknown"),
                            "draw": draw,
                            "n_sites": int(len(pc)),
                            "k_direct": int((~inv_mask).sum()),
                            "k_inverted": int(inv_mask.sum()),
                            "inv_bp": inv_e - inv_s + 1,
                            "span_bp": span_e - span_s + 1,
                            "pairs_direct": p_dir["n_pairs"],
                            "pairs_inverted": p_inv["n_pairs"],
                            "prop_identical_direct": p_dir["prop_identical"],
                            "prop_identical_inverted": p_inv["prop_identical"],
                        })

            rows.append({
                "gene_name": r["gene_name"],
                "transcript_id": r["transcript_id"],
                "inv_id": inv_id,
                "recurrence": recurrence.get(inv_id, "unknown"),
                "chrom": chrom,
                "span_start": span_s, "span_end": span_e,
                "span_bp": span_e - span_s + 1,
                "span_cols_in_region": int(len(cols)),
                "span_coverage": float(len(cols) / (span_e - span_s + 1)),
                # fraction of span bases that survived the repeat / segdup mask
                # in the whole-region alignment; 0 means the gene sits entirely
                # inside masked sequence and has no usable non-coding window
                "span_callable_frac": callable_frac,
                "inv_bp": inv_e - inv_s + 1,
                "cds_bp": int(cmat.shape[1]),
                "k_direct": int((~inv_mask).sum()),
                "k_inverted": int(inv_mask.sum()),
                # ---- whole gene span
                "gene_pairs_direct": g_dir["n_pairs"],
                "gene_pairs_inverted": g_inv["n_pairs"],
                "gene_identical_direct": g_dir["n_identical"],
                "gene_identical_inverted": g_inv["n_identical"],
                "gene_prop_identical_direct": g_dir["prop_identical"],
                "gene_prop_identical_inverted": g_inv["prop_identical"],
                "gene_mean_diffs_direct": g_dir["mean_diffs"],
                "gene_mean_diffs_inverted": g_inv["mean_diffs"],
                "gene_mean_callable_direct": g_dir["mean_callable"],
                "gene_mean_callable_inverted": g_inv["mean_callable"],
                "gene_pi_direct": g_dir["pi"],
                "gene_pi_inverted": g_inv["pi"],
                # ---- CDS only, recomputed here for a like-for-like contrast
                "cds_pairs_direct": c_dir["n_pairs"],
                "cds_pairs_inverted": c_inv["n_pairs"],
                "cds_prop_identical_direct": c_dir["prop_identical"],
                "cds_prop_identical_inverted": c_inv["prop_identical"],
                "cds_pi_direct": c_dir["pi"],
                "cds_pi_inverted": c_inv["pi"],
                "status": "OK",
            })
            print(f"  {r['gene_name']}: span {span_e - span_s + 1} bp "
                  f"| gene propID dir {g_dir['prop_identical']:.3f} "
                  f"inv {g_inv['prop_identical']:.3f} "
                  f"| gene pi dir {g_dir['pi']:.2e} inv {g_inv['pi']:.2e} "
                  f"| CDS propID dir {c_dir['prop_identical']:.3f} "
                  f"inv {c_inv['prop_identical']:.3f}", flush=True)

        del region

    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep="\t", index=False)
    if placebo_rows:
        pl = pd.DataFrame(placebo_rows)
        pl.to_csv(args.placebo_out, sep="\t", index=False)
        print(f"\nWrote {args.placebo_out}: {len(pl)} placebo rows "
              f"({pl['gene_name'].nunique()} genes x up to {args.placebo} draws)")
    ok = out["status"].eq("OK")
    print("\n================ SUMMARY ================")
    print(f"tests: {len(out)}; OK: {int(ok.sum())}")
    if "status" in out:
        print(out["status"].value_counts().to_string())
    d = out[ok]
    if len(d):
        print("\nmean proportion identical, by recurrence:")
        print(d.groupby("recurrence")[[
            "cds_prop_identical_direct", "cds_prop_identical_inverted",
            "gene_prop_identical_direct", "gene_prop_identical_inverted"
        ]].mean().to_string())
        print("\nmedian pi, by recurrence:")
        print(d.groupby("recurrence")[[
            "cds_pi_direct", "cds_pi_inverted",
            "gene_pi_direct", "gene_pi_inverted"
        ]].median().to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
