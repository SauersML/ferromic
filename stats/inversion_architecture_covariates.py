"""Per-inversion recombination rate and genomic compartment.

Reviewer 3 asks whether the recurrent-vs-single-event contrasts are confounded
by "inversion length, genomic compartment, recombination landscape, local SNP
density and allele frequency spectrum". Length, SNP density, CDS density and
allele frequency are already covariates in
stats/recurrence_architecture_controls.py; the two missing ones need external
reference data, which this script turns into a small per-locus covariate table:

  recomb_cM_per_Mb    sex-averaged recombination rate across the inversion,
                      from the HapMap genetic map lifted to GRCh38 (the PLINK
                      map distributed with Beagle). Computed as the genetic
                      length spanned by the inversion divided by its physical
                      length, with linear interpolation between map anchors.
  recomb_cM_per_Mb_flank
                      the same over the 1 Mb flanking each breakpoint, which is
                      what actually governs exchange in heterokaryotypes.
  dist_to_centromere  base pairs from the nearest inversion edge to the nearest
                      centromere boundary (0 if the locus overlaps it), from the
                      UCSC hg38 cytoband acen records.
  rel_arm_position    |midpoint - centromere| / arm length, so 0 is
                      pericentromeric and 1 is telomeric: the "genomic
                      compartment" axis.
  arm                 p or q.

Run where the reference files live (MSI), then commit the small output table.

Inputs (in --refdir):
  plink.GRCh38.map.zip     genetic map, one plink .map per chromosome
  cytoBand.txt.gz          UCSC hg38 cytobands
Input (in --data): inv_properties.tsv
Output: data/inversion_architecture_covariates.tsv
"""

import argparse
import gzip
import io
import os
import zipfile

import numpy as np
import pandas as pd


def load_genetic_map(zip_path):
    """chrom -> (positions, cM) sorted arrays, from the PLINK-format map."""
    maps = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".map"):
                continue
            with z.open(name) as fh:
                txt = io.TextIOWrapper(fh, encoding="utf-8")
                chrom = None
                pos, cm = [], []
                for line in txt:
                    f = line.split()
                    if len(f) < 4:
                        continue
                    chrom = f[0]
                    cm.append(float(f[2]))
                    pos.append(int(f[3]))
            if chrom is None or not pos:
                continue
            key = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
            p = np.asarray(pos, dtype=np.int64)
            c = np.asarray(cm, dtype=float)
            order = np.argsort(p)
            maps[key] = (p[order], c[order])
    return maps


def load_centromeres(path):
    """chrom -> (acen_start, acen_end, chrom_end) in 0-based genome coords."""
    spans = {}
    ends = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            chrom, s, e, _band, stain = f[0], int(f[1]), int(f[2]), f[3], f[4]
            ends[chrom] = max(ends.get(chrom, 0), e)
            if stain == "acen":
                cur = spans.get(chrom)
                spans[chrom] = (min(cur[0], s) if cur else s,
                                max(cur[1], e) if cur else e)
    return {c: (spans[c][0], spans[c][1], ends.get(c, spans[c][1]))
            for c in spans}


def cm_at(pos_arr, cm_arr, x):
    """Interpolated genetic position; NaN outside the mapped range."""
    if len(pos_arr) < 2 or x < pos_arr[0] or x > pos_arr[-1]:
        return np.nan
    return float(np.interp(x, pos_arr, cm_arr))


def rate_over(pos_arr, cm_arr, s, e):
    """cM/Mb over [s, e]; NaN if either end is unmapped or the span is empty."""
    if e <= s:
        return np.nan
    a, b = cm_at(pos_arr, cm_arr, s), cm_at(pos_arr, cm_arr, e)
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return abs(b - a) / ((e - s) / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refdir", required=True,
                    help="directory holding plink.GRCh38.map.zip and cytoBand.txt.gz")
    ap.add_argument("--inv", required=True, help="inv_properties.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flank-bp", type=int, default=1_000_000)
    args = ap.parse_args()

    maps = load_genetic_map(os.path.join(args.refdir, "plink.GRCh38.map.zip"))
    cent = load_centromeres(os.path.join(args.refdir, "cytoBand.txt.gz"))
    print(f"genetic map: {len(maps)} chromosomes; centromeres: {len(cent)}")

    inv = pd.read_csv(args.inv, sep="\t")
    rows = []
    for _, r in inv.iterrows():
        chrom = str(r["Chromosome"])
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        s, e = int(r["Start"]), int(r["End"])
        mid = (s + e) // 2
        rec = rec_fl = np.nan
        if chrom in maps:
            p, c = maps[chrom]
            rec = rate_over(p, c, s, e)
            left = rate_over(p, c, max(0, s - args.flank_bp), s)
            right = rate_over(p, c, e, e + args.flank_bp)
            vals = [v for v in (left, right) if np.isfinite(v)]
            rec_fl = float(np.mean(vals)) if vals else np.nan

        dist = rel = np.nan
        arm = ""
        if chrom in cent:
            cs, ce, cend = cent[chrom]
            if e < cs:
                dist = cs - e
                arm = "p"
                arm_len = cs
                rel = (cs - mid) / arm_len if arm_len > 0 else np.nan
            elif s > ce:
                dist = s - ce
                arm = "q"
                arm_len = cend - ce
                rel = (mid - ce) / arm_len if arm_len > 0 else np.nan
            else:
                dist = 0
                arm = "acen"
                rel = 0.0

        rows.append(dict(
            Chromosome=r["Chromosome"], Start=s, End=e,
            OrigID=r.get("OrigID"),
            recomb_cM_per_Mb=rec,
            recomb_cM_per_Mb_flank=rec_fl,
            dist_to_centromere=dist,
            rel_arm_position=rel,
            arm=arm,
        ))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep="\t", index=False)
    n_ok = int(np.isfinite(out["recomb_cM_per_Mb"]).sum())
    print(f"{len(out)} loci; recombination rate resolved for {n_ok}")
    print(out[["recomb_cM_per_Mb", "recomb_cM_per_Mb_flank",
               "dist_to_centromere", "rel_arm_position"]].describe().to_string())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
