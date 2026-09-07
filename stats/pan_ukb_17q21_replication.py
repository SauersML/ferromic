#!/usr/bin/env python3
"""Fetch the Pan-UK Biobank results quoted for the 17q21.31 replication.

The Results quote four European-ancestry associations for the 17q21.31 tagging
SNP chr17:45996523 (GRCh38; GRCh37 chr17:44073889, A>G) from the Pan-UK Biobank
summary statistics (Karczewski et al. 2025): malignant neoplasm of breast
(ICD-10 C50), obesity (phecode 278.1), mild cognitive impairment (phecode
292.2), and heart failure (ICD-10 I50). This script queries the public
tabix-indexed flat files for that variant and writes the odds ratio, 95%
confidence interval, and p-value for the European-ancestry and meta-analysis
columns to data/pan_ukb_17q21_replication.tsv.

Requires ``tabix`` (htslib) on the PATH and network access to
https://pan-ukb-us-east-1.s3.amazonaws.com/.
"""
from __future__ import annotations

import argparse
import gzip
import io
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files"
INDEX = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_flat_files_tabix"
CHROM, POS, REF, ALT = "17", 44073889, "A", "G"   # GRCh37; GRCh38 chr17:45996523
PHENOTYPES = [
    ("Malignant neoplasm of breast", "icd10-C50"),
    ("Obesity", "phecode-278.1"),
    ("Mild cognitive impairment", "phecode-292.2"),
    ("Heart failure", "icd10-I50"),
]
Z = 1.959963984540054


def header(phenotype_code: str) -> list[str]:
    url = f"{BASE}/{phenotype_code}-both_sexes.tsv.bgz"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-200000"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        chunk = resp.read()
    with gzip.GzipFile(fileobj=io.BytesIO(chunk)) as gz:
        try:
            line = gz.readline()
        except EOFError:
            line = b""
    return line.decode().rstrip("\n").split("\t")


def row(phenotype_code: str) -> list[str]:
    target = f"{BASE}/{phenotype_code}-both_sexes.tsv.bgz##idx##{INDEX}/{phenotype_code}-both_sexes.tsv.bgz.tbi"
    out = subprocess.run(["tabix", target, f"{CHROM}:{POS}-{POS}"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        fields = line.split("\t")
        if fields[2] == REF and fields[3] == ALT:
            return fields
    raise SystemExit(f"{phenotype_code}: no {REF}>{ALT} record at {CHROM}:{POS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pan_ukb_17q21_replication.tsv"))
    args = parser.parse_args()

    lines = ["phenotype\tpan_ukb_code\tpopulation\tgrch37_position\tref\talt\tbeta\tse\tneglog10_p\todds_ratio\tci_lo\tci_hi\tp"]
    for name, code in PHENOTYPES:
        cols = header(code)
        values = dict(zip(cols, row(code)))
        for pop in ("EUR", "meta"):
            if f"beta_{pop}" not in values:
                continue
            beta, se, nlp = (float(values[f"{k}_{pop}"]) for k in ("beta", "se", "neglog10_pval"))
            lines.append(
                "\t".join([
                    name, code, pop, f"chr{CHROM}:{POS}", REF, ALT, f"{beta:.6g}", f"{se:.6g}", f"{nlp:.6g}",
                    f"{math.exp(beta):.4f}", f"{math.exp(beta - Z * se):.4f}", f"{math.exp(beta + Z * se):.4f}",
                    f"{10 ** (-nlp):.3g}",
                ])
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
