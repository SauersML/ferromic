#!/usr/bin/env python3
"""Stream one sample's assembly->hg38 PAF and write, for each inversion whose
window overlaps a target alignment, that record into stageb/<inv_id>/<sample>.paf.
Race-free: each sample writes only its own files. One pass per sample.
"""
import sys, os, gzip, bisect
from collections import defaultdict

inv_tsv, paf_path, sample, outroot = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# load inversions grouped by chrom, sorted by win_start for bisect
by_chrom = defaultdict(list)  # chrom -> list of (win_start, win_end, inv_id)
with open(inv_tsv) as fh:
    header = fh.readline().rstrip("\n").split("\t")
    idx = {c:i for i,c in enumerate(header)}
    for line in fh:
        p = line.rstrip("\n").split("\t")
        chrom = p[idx["chrom"]]
        ws = int(p[idx["win_start"]]); we = int(p[idx["win_end"]]); iid = p[idx["inv_id"]]
        by_chrom[chrom].append((ws, we, iid))
for c in by_chrom:
    by_chrom[c].sort()
starts = {c:[x[0] for x in v] for c,v in by_chrom.items()}

opened = {}  # inv_id -> file handle
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
        # candidate inversions: those with win_start <= tend; scan back while win_end >= tstart
        hi = bisect.bisect_right(st, tend)
        for j in range(hi-1, -1, -1):
            ws, we, iid = ivs[j]
            if we < tstart:
                # since sorted by ws, earlier ones have smaller ws but we not monotonic;
                # windows are wide/overlapping so keep scanning a bounded number
                # break only if ws far below tstart-maxwin; simple: continue
                continue
            if ws <= tend and we >= tstart:
                get_fh(iid).write(line)
                nwrite += 1
for fh in opened.values():
    fh.close()
sys.stderr.write(f"{sample}: scanned {nrec} on-target records, wrote {nwrite} into {len(opened)} inversion dirs\n")
