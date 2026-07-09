#!/usr/bin/env python3
"""Build a clean, QC'd inversion table + plotting windows from inv_properties.tsv.

Output columns: inv_id chrom start end size_bp inverted_af recurrent disease_genes win_start win_end flank
Validates each locus against the reference .fai (chrom present, 0<=start<end<=len).
"""
import sys, csv, re

inv_props, fai_path, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3]

# chrom -> length
chromlen = {}
with open(fai_path) as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t")
        chromlen[p[0]] = int(p[1])

def clean(x):
    return "" if x is None else x.strip()

rows = []
dropped = []
with open(inv_props) as fh:
    reader = csv.reader(fh, delimiter="\t")
    header = next(reader)
    for r in reader:
        # pad
        if len(r) < 22:
            r = r + [""]*(22-len(r))
        chrom = clean(r[0]); start = clean(r[1]); end = clean(r[2])
        origid = clean(r[7]); size_kbp = clean(r[8]); af = clean(r[10])
        disease = clean(r[14]); recur = clean(r[21])  # verdictRecurrence_benson
        if not chrom or not start or not end:
            continue
        try:
            s = int(float(start)); e = int(float(end))
        except ValueError:
            dropped.append((origid or f"{chrom}:{start}-{end}", "bad_coords")); continue
        if e < s:
            s, e = e, s
        if chrom not in chromlen:
            dropped.append((origid, f"chrom_absent:{chrom}")); continue
        L = chromlen[chrom]
        if s < 0 or e > L:
            dropped.append((origid, f"out_of_bounds:{s}-{e}/{L}")); continue
        inv_id = origid if origid and origid != "NA" else f"{chrom}-{s}-INV-{e-s}"
        inv_id = re.sub(r"[^A-Za-z0-9_.-]", "_", inv_id)
        size_bp = e - s
        flank = min(max(size_bp, 50000), 1000000)
        ws = max(0, s - flank); we = min(L, e + flank)
        rows.append([inv_id, chrom, s, e, size_bp, af or "NA",
                     recur or "NA", (disease or "NA"), ws, we, flank])

# dedup inv_id (append suffix if collision)
seen = {}
for row in rows:
    iid = row[0]
    if iid in seen:
        seen[iid]+=1; row[0]=f"{iid}__{seen[iid]}"
    else:
        seen[iid]=0

with open(out_tsv, "w", newline="") as out:
    w = csv.writer(out, delimiter="\t")
    w.writerow(["inv_id","chrom","start","end","size_bp","inverted_af","recurrent","disease_genes","win_start","win_end","flank"])
    w.writerows(rows)

sys.stderr.write(f"kept {len(rows)} inversions, dropped {len(dropped)}\n")
for d in dropped[:50]:
    sys.stderr.write(f"  DROP {d[0]}: {d[1]}\n")
