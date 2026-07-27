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

Reading the direction. AGES reports a selection coefficient ``S`` for its ALT
allele, and which allele that is has nothing to do with orientation -- so raw
``S`` values look contradictory across SNPs that in fact carry the same signal.
``ages_S_inverted_allele`` re-signs ``S`` so it always refers to the
inversion-associated allele: the ALT allele's frequency is read off the tagging
table's per-base counts in the inverted and direct haplotype groups, and ``S`` is
negated when ALT is the *direct*-enriched allele. Compare that column, not ``S``.
Alignment is done on the actual allele rather than on the sign of the
correlation, so it does not depend on which allele the tagging table happened to
count.

Note what this can and cannot establish: multiple tagging SNPs in strong LD with
the inversion are not independent tests, so concordance across them does not
raise confidence in the same way replication would. It distinguishes "one SNP is
an outlier" from "the whole linked region carries the signal" -- it cannot
distinguish an effect of the inversion from an effect of a linked variant, and
nothing in the AGES data can.

On ``ages_FDR``: that column is AGES's own genome-wide correction across every
SNP in their scan, so it is large even for small ``P_X`` (rs4268452 at 10q22.3:
P_X = 7.2e-05, AGES FDR = 0.37). The manuscript's q-values are Benjamini-Hochberg
across the inversion candidate set only, which is a different and much smaller
family. Both are legitimate; whichever is quoted must be named explicitly.

Inputs
  data/tagging_snps.tsv                          per-locus tagging SNP table
  data/selection_data/Selection_Summary_Statistics_01OCT2025.tsv
                                                 AGES summary statistics (hg19)
The 1.8 GB selection file is read through ``batch_best_tagging_snps.load_selection_subset``,
the same chunked streaming loader the single-best-SNP script uses, so only the
requested positions are ever materialised.

Output
  data/ages_multi_tag_snps.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.extract_best_tagging_snp as ebts  # noqa: E402
from scripts.extract_best_tagging_snp import (ensure_selection_data,  # noqa: E402
                                              select_segment_bests,
                                              select_top_tags)
# The chunked streaming loader already used by the single-best-SNP batch script.
# Reused rather than reimplemented: it filters to the exact (chrom, pos) keys
# requested, so the 1.8 GB AGES table is never held in memory.
from stats.batch_best_tagging_snps import load_selection_subset  # noqa: E402

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


BASES = ("A", "C", "G", "T")


def _alt_orientation_sign(row, alt):
    """+1 if the AGES ALT allele is enriched on inverted haplotypes, -1 if on direct.

    Uses the tagging table's per-base frequencies in each orientation group, so the
    alignment is keyed on the actual allele AGES reports S for rather than on the
    sign of a correlation computed against some other allele coding.
    """
    alt = str(alt).strip().upper()
    if alt not in BASES:
        return 0
    try:
        f_inv = float(row[f"{alt}_inv_freq"])
        f_dir = float(row[f"{alt}_dir_freq"])
    except (KeyError, TypeError, ValueError):
        return 0
    if pd.isna(f_inv) or pd.isna(f_dir) or f_inv == f_dir:
        return 0
    return 1 if f_inv > f_dir else -1


def _pick_snps(regions, segments, top_n, tags):
    """Best tagging SNP per segment across each locus, plus the top few overall."""
    picks = []
    for region in regions:
        region_df = ebts._prepare_region_df(region, tags)
        chosen = [("top_overall", r)
                  for r in select_top_tags(region, tags, top_n=top_n)[0]]
        for (seg_start, seg_end), res in select_segment_bests(
                region, region_df, segments=segments):
            if res is not None:
                res.context = f"segment {seg_start}-{seg_end}"
                chosen.append(("segment_best", res))
        seen = set()
        for kind, res in chosen:
            key = (str(res.chromosome_hg37).lstrip("chr"), int(res.position_hg37))
            if key in seen:
                continue
            seen.add(key)
            picks.append((region, kind, res, key))
    return picks


def collect(regions, segments, top_n, tagging_tsv=TAGGING_TSV):
    if not tagging_tsv.exists():
        raise SystemExit(f"tagging SNP table not found at {tagging_tsv}")
    tags = ebts.load_tagging_snps(tagging_tsv)
    picks = _pick_snps(regions, segments, top_n, tags)

    keys = pd.DataFrame([{"chrom_norm": k[0], "position_hg37": k[1]}
                         for *_rest, k in picks]).drop_duplicates()
    subset = load_selection_subset(keys, ensure_selection_data())
    lookup = {(row["CHROM_norm"], int(row["POS"])): row
              for _, row in subset.iterrows()}

    rows = []
    for region, kind, res, key in picks:
        srow = lookup.get(key)
        sign = 0 if srow is None else _alt_orientation_sign(res.row, srow.get("ALT"))
        s_raw = None
        if srow is not None:
            try:
                s_raw = float(srow.get("S"))
            except (TypeError, ValueError):
                s_raw = None
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
            "ages_ref": "" if srow is None else srow.get("REF", ""),
            "ages_alt": "" if srow is None else srow.get("ALT", ""),
            "alt_enriched_on": {1: "inverted", -1: "direct"}.get(sign, ""),
            "ages_S": "" if srow is None else srow.get("S", ""),
            "ages_S_inverted_allele": ("" if (s_raw is None or sign == 0)
                                       else round(s_raw * sign, 8)),
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
        found["s_inv"] = pd.to_numeric(found["ages_S_inverted_allele"], errors="coerce")
        n_sig = int((found["p"] < 0.05).sum())
        print(f"    |r| with inversion: {found['abs_r'].min():.2f}-{found['abs_r'].max():.2f}")
        print(f"    AGES P_X < 0.05 at {n_sig}/{len(found)} of them; "
              f"min P_X = {found['p'].min():.3g}")
        aligned = found["s_inv"].dropna()
        if len(aligned):
            pos = int((aligned > 0).sum())
            print(f"    S on the inverted allele: {pos} positive, "
                  f"{len(aligned) - pos} negative "
                  f"(range {aligned.min():+.5f} to {aligned.max():+.5f})")
            strong = found[found["abs_r"] >= 0.5]["s_inv"].dropna()
            if len(strong):
                sp = int((strong > 0).sum())
                print(f"    restricted to |r| >= 0.5: {sp} positive, "
                      f"{len(strong) - sp} negative (n = {len(strong)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
