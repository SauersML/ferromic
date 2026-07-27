#!/usr/bin/env python
"""FinnGen replication of the inversion PheWAS signals (Reviewer 2 #3).

The response letter says the tag SNPs were checked for replication in FinnGen.
Nothing in the repository did that, so this does it, from FinnGen's public
PheWeb API -- no credentials, no bulk download.

Approach
--------
FinnGen's browser exposes a per-variant endpoint that returns every endpoint
association for one variant. Querying the inversion's best tagging SNP therefore
gives a FinnGen PheWAS for that inversion in one request, which is exactly the
replication asked for: the All of Us signals were found on imputed inversion
dosage, and the tagging SNP is the transportable proxy.

Direction is the thing to get right. FinnGen reports beta for its own ALT allele,
which has nothing to do with orientation, so the raw sign is not comparable to an
All of Us odds ratio on inversion dosage. Every effect is therefore re-signed onto
the inversion-associated allele using the same per-base orientation frequencies
that ``ages_multi_tag_snps.py`` uses, and reported as ``beta_inverted_allele``.
Rows where the allele cannot be matched are kept but flagged rather than dropped.

Multiple testing: FinnGen's own per-endpoint p-values are reported as given. This
is a *replication* of pre-specified signals, not a discovery scan, so the relevant
comparison is direction and magnitude at the named endpoints, and no new
correction across FinnGen's endpoint list is applied or implied.

Inputs
  data/tagging_snps.tsv         per-locus tagging SNPs with per-orientation freqs
  data/inv_properties.tsv       locus catalog
Output
  data/finngen_replication.tsv  one row per (inversion, FinnGen endpoint)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.extract_best_tagging_snp as ebts  # noqa: E402
from scripts.extract_best_tagging_snp import select_top_tags  # noqa: E402
from stats.ages_multi_tag_snps import _alt_orientation_sign  # noqa: E402

ebts.OUTPUT_DIR = Path("data")

TAGGING_TSV = Path("data/tagging_snps.tsv")
OUT_TSV = Path("data/finngen_replication.tsv")

# FinnGen public PheWeb. The variant endpoint takes GRCh38 chr-pos-ref-alt.
FINNGEN_RELEASE = os.environ.get("FINNGEN_RELEASE", "r12")
VARIANT_URL = "https://{rel}.finngen.fi/api/variant/{chrom}-{pos}-{ref}-{alt}"

# The loci with All of Us signals worth attempting to replicate, and the phenotype
# words to surface in the summary. Everything FinnGen returns is written to the
# table; these only drive the printed digest.
TARGETS = {
    "chr8:7301024-12598379":   ["goiter", "thyroid", "alopecia", "hypothyroid"],
    "chr17:45585159-46292045": ["obesity", "breast", "dementia", "cognitive",
                                "heart failure", "nevi"],
    "chr12:46896694-46915975": ["conjunctivitis", "acne", "migraine"],
    "chr10:79542901-80217413": ["papillomavirus", "cervical"],
    "chr6:141866310-141898728": ["laryngitis", "tracheitis"],
}


def _fetch(url, retries=3, pause=2.0):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ferromic-replication"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(pause * (attempt + 1))
    return None


def variant_phewas(chrom, pos, ref, alt):
    """Every FinnGen endpoint association for one variant, or None if absent."""
    c = str(chrom).replace("chr", "")
    for a, b in ((ref, alt), (alt, ref)):     # FinnGen's ref/alt may be flipped
        url = VARIANT_URL.format(rel=FINNGEN_RELEASE, chrom=c, pos=int(pos),
                                 ref=a, alt=b)
        data = _fetch(url)
        if data:
            return data, (a, b)
    return None, (None, None)


def collect(regions, top_n=3, tagging_tsv=TAGGING_TSV):
    if not tagging_tsv.exists():
        raise SystemExit(f"tagging SNP table not found at {tagging_tsv}")
    tags = ebts.load_tagging_snps(tagging_tsv)

    rows = []
    for region in regions:
        try:
            picks = select_top_tags(region, tags, top_n=top_n)[0]
        except ValueError as exc:
            print(f"  {region}: {exc}")
            continue
        for res in picks:
            chrom38 = str(res.chromosome_hg38)
            pos38 = int(res.position_hg38)
            row = res.row
            # Alleles present in the haplotype panel at this site.
            bases = [b for b in ("A", "C", "G", "T")
                     if float(row.get(f"{b}_inv_freq", 0) or 0) > 0
                     or float(row.get(f"{b}_dir_freq", 0) or 0) > 0]
            if len(bases) < 2:
                continue
            data, (ref, alt) = variant_phewas(chrom38, pos38, bases[0], bases[1])
            if not data:
                print(f"  {region} {chrom38}:{pos38} not found in FinnGen {FINNGEN_RELEASE}")
                continue
            sign = _alt_orientation_sign(row, alt)
            phenos = data.get("phenos") or data.get("results") or []
            for ph in phenos:
                beta = ph.get("beta")
                try:
                    beta = float(beta)
                except (TypeError, ValueError):
                    beta = None
                rows.append({
                    "inversion": region,
                    "rsid": data.get("rsids") or ph.get("rsids") or "",
                    "chrom_hg38": chrom38,
                    "pos_hg38": pos38,
                    "ref": ref, "alt": alt,
                    "alt_enriched_on": {1: "inverted", -1: "direct"}.get(sign, "unknown"),
                    "r_with_inversion": round(float(res.correlation), 6),
                    "finngen_endpoint": ph.get("phenocode") or ph.get("pheno") or "",
                    "finngen_phenotype": ph.get("phenostring") or ph.get("phenoname") or "",
                    "n_case": ph.get("num_cases", ""), "n_control": ph.get("num_controls", ""),
                    "beta_alt_allele": beta,
                    "beta_inverted_allele": (None if (beta is None or sign == 0)
                                             else round(beta * sign, 6)),
                    "sebeta": ph.get("sebeta", ""),
                    "p_value": ph.get("pval", ""),
                    "direction_usable": "yes" if sign != 0 and beta is not None else "no",
                })
            print(f"  {region} {chrom38}:{pos38} -> {len(phenos)} FinnGen endpoints")
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regions", nargs="+", default=sorted(TARGETS))
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT_TSV)
    args = ap.parse_args(argv)

    print(f">>> FinnGen {FINNGEN_RELEASE} replication of inversion PheWAS signals\n")
    df = collect(args.regions, args.top_n)
    if df.empty:
        raise SystemExit("no FinnGen associations retrieved")
    df = df.sort_values(["inversion", "p_value"], kind="mergesort")
    # FinnGen's phenotype strings contain non-breaking spaces (U+00A0). Writing
    # them through would leave a committed table that the repository's NBSP fixer
    # then rewrites on every push, so normalise here instead of letting a bot
    # edit committed data after the fact.
    df = df.map(lambda v: v.replace("\u00a0", " ") if isinstance(v, str) else v)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"\n{len(df)} (variant x endpoint) rows -> {args.out}\n")

    for region, g in df.groupby("inversion", sort=False):
        words = TARGETS.get(region, [])
        usable = g[g["direction_usable"] == "yes"].copy()
        n_unusable = len(g) - len(usable)
        hits = usable[usable["finngen_phenotype"].str.lower().str.contains(
            "|".join(words), na=False)] if words else usable.head(0)
        # One row per phenotype: keep the strongest tagging SNP's result.
        hits = hits.sort_values("p_value").drop_duplicates("finngen_phenotype")
        note = f" ({n_unusable} rows had no resolvable allele direction)" if n_unusable else ""
        print(f"{region}   {len(g)} endpoints; {len(hits)} distinct phenotypes matching "
              f"the All of Us signal words{note}")
        if hits.empty:
            continue
        for _, r in hits.nsmallest(6, "p_value").iterrows():
            print(f"    {str(r['finngen_phenotype'])[:52]:54s} "
                  f"beta(inv)={float(r['beta_inverted_allele']):+.4f}  "
                  f"p={float(r['p_value']):.3g}")
    print("\nDirections are on the inversion-associated allele. These are replications "
          "of pre-specified\nsignals, not a discovery scan, so no correction across "
          "FinnGen's endpoint list is applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
