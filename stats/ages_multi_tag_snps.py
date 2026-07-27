#!/usr/bin/env python
"""AGES selection statistics for multiple tagging SNPs per inversion (Reviewer 2 #9).

Reviewer 2 notes that the 8p23.1 inversion's best tagging SNP has r = 0.76 with
the inversion, and asks whether the ancient-DNA selection signal is specific to
that SNP or holds across the linked haplotype. ``batch_best_tagging_snps.py``
reports one SNP per locus, which cannot answer that.

This walks each inversion in segments across its full span, takes the best
tagging SNP within each segment, and looks up that SNP's AGES selection
statistics. A signal carried by the haplotype should appear at several SNPs
spanning the LD range; a signal specific to one variant should not.

Note what this can and cannot establish: multiple tagging SNPs in strong LD with
the inversion are not independent tests, so concordance across them does not
raise confidence in the same way replication would. It distinguishes "one SNP is
an outlier" from "the whole linked region carries the signal" -- it cannot
distinguish an effect of the inversion from an effect of a linked variant, and
nothing in the AGES data can.

Inputs
  data/tagging_snps.tsv                          per-locus tagging SNP table
  data/selection_data/Selection_Summary_Statistics_01OCT2025.tsv
                                                 AGES summary statistics (hg19)
The selection file is ~1.8 GB; it is streamed and pre-filtered to the
chromosomes actually needed before anything is loaded into memory.

Output
  data/ages_multi_tag_snps.tsv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.extract_best_tagging_snp as ebts  # noqa: E402
from scripts.extract_best_tagging_snp import (parse_region,  # noqa: E402
                                              select_segment_bests,
                                              select_top_tags)

ebts.OUTPUT_DIR = Path("data")
ebts.SELECTION_DIR = ebts.OUTPUT_DIR / "selection_data"
ebts.SELECTION_TSV_PATH = ebts.SELECTION_DIR / ebts.SELECTION_TSV_NAME

TAGGING_TSV = Path("data/tagging_snps.tsv")
OUT_TSV = Path("data/ages_multi_tag_snps.tsv")

# The four loci whose best tagging SNP reaches significance in
# data/best_tagging_snps_qvalues.tsv, i.e. the ones the claim rests on.
DEFAULT_REGIONS = [
    "chr8:7301024-12598379",      # 8p23.1  -- the locus Reviewer 2 asks about
    "chr10:79542901-80217413",    # 10q22.3
    "chr12:46896694-46915975",    # 12q13.11
    "chr7:54234014-54308393",     # 7p11.2
]

SEL_COLS = ["CHROM", "POS", "REF", "ALT", "ANC", "RSID", "AF", "S", "SE",
            "X", "P_X", "POSTERIOR", "FDR", "FILTER"]


def load_selection_subset(chroms, path=None):
    """Stream the AGES table, keeping only the chromosomes we need."""
    path = Path(path or ebts.SELECTION_TSV_PATH)
    if not path.exists():
        raise SystemExit(
            f"AGES selection statistics not found at {path}. Run "
            "scripts/extract_best_tagging_snp.py once to download them.")
    wanted = {str(c).removeprefix("chr") for c in chroms}
    pattern = "|".join(sorted(wanted))
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name
    # Header line is the first non-'##' line; keep it plus matching CHROM rows.
    awk = (r'BEGIN{FS=OFS="\t"} /^##/ {next} '
           r'!hdr {print; hdr=1; next} '
           r'{c=$1; sub(/^chr/,"",c); if (c ~ /^(' + pattern + r')$/) print}')
    with open(tmp_path, "w") as out:
        subprocess.run(["awk", awk, str(path)], stdout=out, check=True)
    df = pd.read_csv(tmp_path, sep="\t", low_memory=False)
    os.unlink(tmp_path)
    df["CHROM_norm"] = df["CHROM"].astype(str).str.removeprefix("chr")
    return df


def _lookup(sel, chrom, pos):
    hit = sel[(sel["CHROM_norm"] == str(chrom).removeprefix("chr"))
              & (sel["POS"] == int(pos))]
    return None if hit.empty else hit.iloc[0]


def collect(regions, segments, top_n, tagging_tsv=TAGGING_TSV):
    if not tagging_tsv.exists():
        raise SystemExit(f"tagging SNP table not found at {tagging_tsv}")
    tags = ebts.load_tagging_snps(tagging_tsv)

    chroms = {parse_region(r)[0] for r in regions}
    print(f"Loading AGES statistics for {sorted(chroms)} ...", flush=True)
    sel = load_selection_subset(chroms)
    print(f"  {len(sel):,} AGES rows on those chromosomes", flush=True)

    rows = []
    for region in regions:
        region_df = ebts._prepare_region_df(region, tags)
        picks = []
        for res in select_top_tags(region, tags, top_n=top_n)[0]:
            picks.append(("top_overall", res))
        for (seg_start, seg_end), res in select_segment_bests(
                region, region_df, segments=segments):
            if res is not None:
                res.context = f"segment {seg_start}-{seg_end}"
                picks.append(("segment_best", res))

        seen = set()
        for kind, res in picks:
            key = (res.chromosome_hg37, res.position_hg37)
            if key in seen:
                continue
            seen.add(key)
            srow = _lookup(sel, res.chromosome_hg37, res.position_hg37)
            rows.append({
                "region": region,
                "selection_kind": kind,
                "context": res.context or "",
                "chrom_hg19": res.chromosome_hg37,
                "pos_hg19": res.position_hg37,
                "chrom_hg38": res.chromosome_hg38,
                "pos_hg38": res.position_hg38,
                "rsid": "" if srow is None else srow.get("RSID", ""),
                "r_with_inversion": round(float(res.correlation), 6),
                "abs_r": round(abs(float(res.correlation)), 6),
                "ages_S": "" if srow is None else srow.get("S", ""),
                "ages_SE": "" if srow is None else srow.get("SE", ""),
                "ages_P_X": "" if srow is None else srow.get("P_X", ""),
                "ages_FDR": "" if srow is None else srow.get("FDR", ""),
                "ages_FILTER": "" if srow is None else srow.get("FILTER", ""),
                "in_ages": "no" if srow is None else "yes",
            })
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    ap.add_argument("--segments", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUT_TSV)
    args = ap.parse_args(argv)

    df = collect(args.regions, args.segments, args.top_n)
    df = df.sort_values(["region", "pos_hg19"], kind="mergesort")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"\n{len(df)} tagging SNPs -> {args.out}\n")

    for region, g in df.groupby("region", sort=False):
        found = g[g["in_ages"] == "yes"].copy()
        print(f"{region}   {len(g)} SNPs spanning the locus, "
              f"{len(found)} present in AGES")
        if found.empty:
            continue
        found["p"] = pd.to_numeric(found["ages_P_X"], errors="coerce")
        found["s"] = pd.to_numeric(found["ages_S"], errors="coerce")
        n_sig = int((found["p"] < 0.05).sum())
        print(f"    |r| with inversion: {found['abs_r'].min():.2f}-{found['abs_r'].max():.2f}")
        print(f"    AGES P_X < 0.05 at {n_sig}/{len(found)} of them; "
              f"min P_X = {found['p'].min():.3g}")
        same_sign = found["s"].dropna()
        if len(same_sign):
            pos = int((same_sign > 0).sum())
            print(f"    selection coefficient S sign: {pos} positive, "
                  f"{len(same_sign) - pos} negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
