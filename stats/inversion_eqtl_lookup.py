#!/usr/bin/env python
"""GTEx eQTL evidence for the inversion tagging SNPs.

The draft carries a gene-expression claim (PRAG1) with no code behind it. This
supplies the reproducible basis for any such claim: for each inversion's tagging
SNPs, every single-tissue eQTL GTEx v8 reports, with the effect re-signed onto the
inversion-associated allele and each eQTL gene flagged for whether it lies inside
the inversion.

What this can and cannot support
--------------------------------
An eQTL at a tagging SNP says the linked haplotype associates with expression. It
does **not** localise the effect to the inversion itself: everything on the
haplotype is equally implicated, and with recombination suppressed across the
locus that is a large block. Genes *inside* the inversion are flagged because a
cis-eQTL there is at least consistent with the rearrangement acting locally, but
this is annotation, not colocalisation, and the script does not claim otherwise.

Direction is re-signed onto the inversion-associated allele (``nes_inverted_allele``)
using the same per-base orientation frequencies as ``ages_multi_tag_snps.py``, so
the sign means "expression change carried by the inverted haplotype" rather than
"by GTEx's ALT allele".

Multiple testing: GTEx's own per-tissue q-values are reported as returned. No
correction is applied across tagging SNPs, because they tag one haplotype and are
not independent tests -- the same reasoning as in ``ages_multi_tag_snps.py``.

Output: data/inversion_eqtl.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.extract_best_tagging_snp as ebts  # noqa: E402
from scripts.extract_best_tagging_snp import parse_region, select_top_tags  # noqa: E402
from stats.ages_multi_tag_snps import _alt_orientation_sign  # noqa: E402

ebts.OUTPUT_DIR = Path("data")

TAGGING_TSV = Path("data/tagging_snps.tsv")
OUT_TSV = Path("data/inversion_eqtl.tsv")
GTEX = "https://gtexportal.org/api/v2/association/singleTissueEqtl"

REGIONS = [
    "chr8:7301024-12598379",       # 8p23.1
    "chr17:45585159-46292045",     # 17q21.31
    "chr12:46896694-46915975",     # 12q13.11
    "chr10:79542901-80217413",     # 10q22.3
    "chr7:54234014-54308393",      # 7p11.2
    "chr6:141866310-141898728",    # 6q24.1
]


def _get(url, retries=3, pause=2.0):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ferromic-eqtl"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404):
                return None
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(pause * (attempt + 1))
    return None


def gtex_eqtls(chrom, pos, a, b, page_size=250):
    """Single-tissue eQTLs for one variant, trying both ref/alt orderings."""
    c = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    for ref, alt in ((a, b), (b, a)):
        vid = f"{c}_{int(pos)}_{ref}_{alt}_b38"
        q = urllib.parse.urlencode({"variantId": vid, "datasetId": "gtex_v8",
                                    "itemsPerPage": page_size})
        data = _get(f"{GTEX}?{q}")
        if data and data.get("data"):
            return data["data"], ref, alt
    return [], None, None


def collect(regions, top_n=3):
    if not TAGGING_TSV.exists():
        raise SystemExit(f"tagging SNP table not found at {TAGGING_TSV}")
    tags = ebts.load_tagging_snps(TAGGING_TSV)

    rows = []
    for region in regions:
        _c, r_start, r_end = parse_region(region)
        try:
            picks = select_top_tags(region, tags, top_n=top_n)[0]
        except ValueError as exc:
            print(f"  {region}: {exc}")
            continue
        for res in picks:
            row = res.row
            bases = [x for x in ("A", "C", "G", "T")
                     if float(row.get(f"{x}_inv_freq", 0) or 0) > 0
                     or float(row.get(f"{x}_dir_freq", 0) or 0) > 0]
            if len(bases) < 2:
                continue
            hits, ref, alt = gtex_eqtls(res.chromosome_hg38, res.position_hg38,
                                        bases[0], bases[1])
            if not hits:
                print(f"  {region} {res.chromosome_hg38}:{res.position_hg38} "
                      f"-> no GTEx eQTL")
                continue
            sign = _alt_orientation_sign(row, alt)
            for h in hits:
                nes = h.get("nes")
                try:
                    nes = float(nes)
                except (TypeError, ValueError):
                    nes = None
                gene_pos = h.get("pos")
                inside = ""
                try:
                    inside = "yes" if r_start <= int(gene_pos) <= r_end else "no"
                except (TypeError, ValueError):
                    inside = ""
                rows.append({
                    "inversion": region,
                    "rsid": h.get("snpId", ""),
                    "variant_id": h.get("variantId", ""),
                    "ref": ref, "alt": alt,
                    "alt_enriched_on": {1: "inverted", -1: "direct"}.get(sign, "unknown"),
                    "r_with_inversion": round(float(res.correlation), 6),
                    "gene": h.get("geneSymbol", ""),
                    "gencode_id": h.get("gencodeId", ""),
                    "tissue": h.get("tissueSiteDetailId", ""),
                    "variant_inside_inversion": inside,
                    "nes_alt_allele": nes,
                    "nes_inverted_allele": (None if (nes is None or sign == 0)
                                            else round(nes * sign, 6)),
                    "p_value": h.get("pValue", ""),
                    "direction_usable": "yes" if (sign != 0 and nes is not None) else "no",
                })
            print(f"  {region} {res.chromosome_hg38}:{res.position_hg38} "
                  f"-> {len(hits)} eQTL records")
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT_TSV)
    args = ap.parse_args(argv)

    print(">>> GTEx v8 eQTL evidence at the inversion tagging SNPs\n")
    df = collect(args.regions, args.top_n)
    if df.empty:
        raise SystemExit("no eQTL records retrieved")
    df = df.sort_values(["inversion", "p_value"], kind="mergesort")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"\n{len(df)} (variant x gene x tissue) records -> {args.out}\n")

    for region, g in df.groupby("inversion", sort=False):
        genes = g["gene"].nunique()
        usable = g[g["direction_usable"] == "yes"]
        print(f"{region}   {len(g)} records, {genes} distinct genes, "
              f"{g['tissue'].nunique()} tissues")
        top = (usable.sort_values("p_value")
                     .drop_duplicates("gene")
                     .head(5))
        for _, r in top.iterrows():
            print(f"    {str(r['gene'])[:22]:24s} {str(r['tissue'])[:26]:28s} "
                  f"NES(inv)={float(r['nes_inverted_allele']):+.3f}  p={float(r['p_value']):.3g}")
    print("\nAn eQTL at a tagging SNP implicates the whole linked haplotype, not the "
          "inversion\nitself. Genes inside the inversion are flagged, but this is "
          "annotation, not colocalisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
