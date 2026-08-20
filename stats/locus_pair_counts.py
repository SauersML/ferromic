"""Exact pairwise difference and callable-site counts per locus and orientation.

The manuscript's central comparison is fitted to log(pi + epsilon), where epsilon
is a detection floor chosen from the data. That transform exists only because many
inverted haplotype groups have pi exactly zero, and it has consequences: the
estimate depends on the arbitrary floor, a locus with zero differences over ten
callable sites is treated the same as one with zero over a million, and the
sampling variance of pi is discarded.

None of that is necessary, because pi is a ratio of two integers that the
alignments hold exactly. This script emits them: for each locus and each
orientation group, the total number of nucleotide differences summed over all
within-group haplotype pairs, and the total number of sites at which both
haplotypes of a pair are called. pi is the first divided by the second, and the
pair (D, C) supports a likelihood model that needs no floor.

stats/diversity_count_model.py consumes the output.

Inputs (relative to --workdir):
  phy_outputs/inversion_group{0,1}_<chrom>_start<s>_end<e>.phy.gz
Output:
  results/locus_pair_counts.tsv
"""

import argparse
import gzip
import os
import re

import numpy as np
import pandas as pd

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
        raise ValueError(f"{path}: header {n_declared}, parsed {len(seqs)}")
    return seqs


def pair_totals(mat, chunk=200_000):
    """Total pairwise differences and total callable pair-sites for a group.

    Chunked over columns so a two-megabase locus with a hundred haplotypes does
    not need a hundred-by-two-million boolean array in memory at once.
    """
    n = mat.shape[0]
    if n < 2:
        return 0, 0, n, 0
    diffs = 0
    callable_ = 0
    n_pairs = n * (n - 1) // 2
    for start in range(0, mat.shape[1], chunk):
        block = mat[:, start:start + chunk]
        ok = block != 255
        for i in range(n - 1):
            both = ok[i] & ok[i + 1:]
            callable_ += int(both.sum())
            diffs += int(((block[i] != block[i + 1:]) & both).sum())
    return diffs, callable_, n, n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", default="results/locus_pair_counts.tsv")
    args = ap.parse_args()
    os.chdir(args.workdir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    phy_dir = "phy_outputs"
    region_files = {}
    for fn in os.listdir(phy_dir):
        m = INV_RE.match(fn)
        if m:
            region_files.setdefault(
                (m["chrom"], int(m["s"]), int(m["e"])), {})[m["grp"]] = \
                os.path.join(phy_dir, fn)
    both = {k: v for k, v in region_files.items() if set(v) == {"0", "1"}}
    print(f"{len(region_files)} loci with a region alignment, "
          f"{len(both)} with both orientations", flush=True)

    rows = []
    for i, (key, paths) in enumerate(sorted(both.items()), 1):
        chrom, s, e = key
        rec = {"chrom": chrom, "start": s, "end": e, "span_bp": e - s + 1}
        for grp, label in (("0", "direct"), ("1", "inverted")):
            seqs = read_phy(paths[grp])
            if not seqs:
                rec[f"n_hap_{label}"] = 0
                continue
            mat = np.vstack([ENC[np.frombuffer(v.encode(), dtype=np.uint8)]
                             for v in seqs.values()])
            d, c, n, npair = pair_totals(mat)
            rec[f"n_hap_{label}"] = n
            rec[f"n_pairs_{label}"] = npair
            rec[f"diffs_{label}"] = d
            rec[f"callable_{label}"] = c
            rec[f"pi_{label}"] = (d / c) if c else np.nan
            del mat
        rows.append(rec)
        if i % 20 == 0:
            print(f"  {i}/{len(both)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"\nWrote {args.out} ({len(out)} loci)")
    ok = out["callable_direct"].fillna(0).gt(0) & out["callable_inverted"].fillna(0).gt(0)
    print(f"loci with callable sites in both orientations: {int(ok.sum())}")
    print(f"loci with zero inverted differences: "
          f"{int((out['diffs_inverted'].fillna(-1) == 0).sum())}")
    print(out.loc[ok, ["pi_direct", "pi_inverted"]].describe().to_string())


if __name__ == "__main__":
    main()
