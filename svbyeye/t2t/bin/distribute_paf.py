#!/usr/bin/env python3
"""Stream one query's ->hg38 PAF and write, for each inversion whose window
overlaps a target alignment, that record into stageb/<inv_id>/<sample>.paf.

Faithful copy of ferromic/svbyeye/bin/distribute_paf.py. Here `sample` is a chimp
reference id (e.g. chimpT2T or panTro6).
"""
import sys, os, gzip, bisect
from collections import defaultdict

inv_tsv, paf_path, sample, outroot = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

by_chrom = defaultdict(list)
with open(inv_tsv) as fh:
    header = fh.readline().rstrip("\n").split("\t")
    idx = {c: i for i, c in enumerate(header)}
    for line in fh:
        p = line.rstrip("\n").split("\t")
        chrom = p[idx["chrom"]]
        ws = int(p[idx["win_start"]]); we = int(p[idx["win_end"]]); iid = p[idx["inv_id"]]
        by_chrom[chrom].append((ws, we, iid))
for c in by_chrom:
    by_chrom[c].sort()
starts = {c: [x[0] for x in v] for c, v in by_chrom.items()}

opened = {}
def get_fh(iid):
    if iid not in opened:
        d = os.path.join(outroot, iid)
        os.makedirs(d, exist_ok=True)
        opened[iid] = open(os.path.join(d, sample + ".paf"), "w")
    return opened[iid]

opener = gzip.open if paf_path.endswith(".gz") else open
nrec = 0; nwrite = 0
with opener(paf_path, "rt") as fh:
    for line in fh:
        if not line or line[0] == "@":
            continue
        f = line.split("\t")
        if len(f) < 12:
            continue
        tname = f[5]
        if tname not in by_chrom:
            continue
        try:
            tstart = int(f[7]); tend = int(f[8])
        except ValueError:
            continue
        nrec += 1
        ivs = by_chrom[tname]; st = starts[tname]
        hi = bisect.bisect_right(st, tend)
        for j in range(hi - 1, -1, -1):
            ws, we, iid = ivs[j]
            if we < tstart:
                continue
            if ws <= tend and we >= tstart:
                get_fh(iid).write(line)
                nwrite += 1
for fh in opened.values():
    fh.close()
sys.stderr.write(f"{sample}: scanned {nrec} on-target records, wrote {nwrite} into {len(opened)} inversion dirs\n")
