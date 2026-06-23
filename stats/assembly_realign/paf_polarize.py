#!/usr/bin/env python3
"""Call ancestral orientation from per-assembly minimap2 PAFs (ape contigs vs hg38 locus
regions). Each PAF = one ape haplotype (filename TAG = SPECIES_hN). Inversion signature:
locus interior aligns to the ape on the OPPOSITE strand vs its flanking backbone, on the
SAME ape contig (paralog jumps to other contigs rejected). Diploid HOM/HET per species,
depth-weighted parsimony (siamang deepest) -> flip. Validates vs Strand-seq, writes calls.

Usage: paf_polarize2.py <manifest.tsv> <outdir> <paf1> <paf2> ...
"""
import csv, os, re, sys
from collections import defaultdict

MIN_MAPQ = 20
MIN_BLK = 1000
DEPTH = {"PTR": 2, "PPA": 2, "GGO": 3, "PAB": 4, "PPY": 4, "SSY": 5}
DEEP = {"PAB", "PPY", "SSY"}


def load_manifest(p):
    return {r["region_name"]: r for r in csv.DictReader(open(p), delimiter="\t")}


def parse_paf(paf):
    """region_name -> ape_contig -> [(rloc0,rloc1,strand,mapq,blk)].
    PAF is ape-assembly(ref) vs hg38-regions(query): query=f[0]=region_name,
    region-local coords = query start/end f[2],f[3]; target=f[5]=ape contig."""
    d = defaultdict(lambda: defaultdict(list))
    with open(paf) as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 12: continue
            mapq, blk = int(f[11]), int(f[10])
            if mapq < MIN_MAPQ or blk < MIN_BLK: continue
            d[f[0]][f[5]].append((int(f[2]), int(f[3]), f[4], mapq, blk))
    return d


def orient(man_row, contig_blocks):
    rs, re_ = int(man_row["reg_start"]), int(man_row["reg_end"])
    ls, le = int(man_row["loc_start"]), int(man_row["loc_end"])
    iL, iR, end = ls - rs, le - rs, re_ - rs
    def seg(blks, a, b):
        return sum(min(t1, b) - max(t0, a) for (t0, t1, *_ ) in blks if min(t1, b) > max(t0, a))
    best = None
    for contig, blks in contig_blocks.items():
        lf, rf = seg(blks, 0, iL), seg(blks, iR, end)
        if lf > 0 and rf > 0 and (best is None or lf + rf > best[1]):
            best = (contig, lf + rf)
    if not best: return None
    blks = contig_blocks[best[0]]
    def dom(a, b):
        w = defaultdict(int)
        for (t0, t1, st, mq, bl) in blks:
            ov = min(t1, b) - max(t0, a)
            if ov > 0: w[st] += ov
        return max(w, key=w.get) if w else None
    fL, fR, fi = dom(0, iL), dom(iR, end), dom(iL, iR)
    flank = fL if fL == fR else (fL or fR)
    if not flank or not fi: return None
    return "inverted" if fi != flank else "collinear"


def main():
    man = load_manifest(sys.argv[1]); outdir = sys.argv[2]
    # region -> species -> hap -> orient
    data = defaultdict(lambda: defaultdict(dict))
    for paf in sys.argv[3:]:
        tag = os.path.basename(paf).split(".")[0]   # e.g. PTR_h1
        sp, hap = tag.split("_")[0], tag.split("_")[-1]
        blocks = parse_paf(paf)
        for rn, row in man.items():
            cb = blocks.get(rn)
            if not cb: continue
            o = orient(row, cb)
            if o: data[rn][sp][hap] = o

    rows = []
    for rn, row in man.items():
        sp_state = {}
        for sp, haps in data.get(rn, {}).items():
            vals = list(haps.values())
            if not vals: continue
            ninv = vals.count("inverted"); ncol = vals.count("collinear")
            sp_state[sp] = "HET" if (ninv and ncol) else ("inv" if ninv else "col")
        if not sp_state:
            rows.append((row, None, {})); continue
        w_inv = sum(DEPTH[s] for s, v in sp_state.items() if v == "inv")
        w_col = sum(DEPTH[s] for s, v in sp_state.items() if v == "col")
        n_het = sum(v == "HET" for v in sp_state.values())
        n_hom = sum(v in ("inv", "col") for v in sp_state.values())
        deep_inv = sum(1 for s, v in sp_state.items() if s in DEEP and v == "inv")
        deep_col = sum(1 for s, v in sp_state.items() if s in DEEP and v == "col")
        if w_inv == w_col:
            flip = None
        else:
            flip = 1 if w_inv > w_col else 0
        rows.append((row, flip, sp_state, n_het, n_hom, deep_inv, deep_col))

    # validation
    ag = tot = 0
    for r in rows:
        row, flip = r[0], r[1]
        if flip is None: continue
        if row["is_valid"] == "1" and row["ss_flip"] in ("0", "1"):
            tot += 1; ok = flip == int(row["ss_flip"]); ag += ok
    acc = f"{ag}/{tot} = {100*ag/tot:.0f}%" if tot else "n/a"
    print(f"REALIGN(MSI full-assembly) vs Strand-seq gold: {acc}")

    with open(os.path.join(outdir, "realign_calls.tsv"), "w") as fh:
        fh.write("inv_id\tflip\tn_species\tn_het\tn_hom\tdeep_inv\tdeep_col\tconfidence\tstates\tis_valid\tss_flip\tmatch\n")
        for r in rows:
            row, flip = r[0], r[1]
            if len(r) < 7:
                fh.write(f"{row['inv_id']}\t\t0\t\t\t\t\tnone\t\t{row['is_valid']}\t{row['ss_flip']}\t\n"); continue
            _, _, st, n_het, n_hom, di, dc = r
            deep_conc = di if flip == 1 else dc
            if n_hom >= 3 and deep_conc >= 2 and n_het == 0: conf = "high"
            elif n_hom >= 2 and deep_conc >= 1: conf = "moderate"
            elif n_het >= 2: conf = "recurrent"
            elif n_hom >= 1: conf = "low"
            else: conf = "none"
            m = "" if row["ss_flip"] not in ("0","1") else ("OK" if str(flip)==row["ss_flip"] else "XX")
            fh.write(f"{row['inv_id']}\t{'' if flip is None else flip}\t{len(st)}\t{n_het}\t{n_hom}\t{di}\t{dc}\t{conf}\t{';'.join(f'{k}:{v}' for k,v in sorted(st.items()))}\t{row['is_valid']}\t{row['ss_flip']}\t{m}\n")
    print("wrote realign_calls.tsv")


if __name__ == "__main__":
    main()
