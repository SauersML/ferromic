#!/usr/bin/env python3
"""
Download the ferromic tagging_snps.tsv.zip file, find the top 100 tagging SNPs
for four inversion regions, and for each SNP:

1. Determine the two alleles (from inversion vs direct groups).
2. Query azphewas.com for BOTH allele orders using a real browser session
   via Selenium:

   https://azphewas.com/variantView/{DATASET_ID}/CHR-POS-A1-A2/vlr/binary
   https://azphewas.com/variantView/{DATASET_ID}/CHR-POS-A2-A1/vlr/binary

3. For each URL, check whether the rendered HTML page_source:
     - CONTAINS / NOT_CONTAINS "are not available for this dataset"
     - CONTAINS / NOT_CONTAINS "Consequence type"

4. Print the results as a tab-separated table.

Before processing SNPs, this script performs two pre-flight checks:

1) 17-45996523-A-G:
     - HTML must NOT contain "are not available for this dataset"
     - HTML MUST contain "Consequence type"

2) 17-45996523-G-A  (interpreting your second example as reversed alleles):
     - HTML MUST contain "are not available for this dataset"
     - HTML must NOT contain "Consequence type"

For each pre-flight check it:
  - prints the query URL
  - prints the raw HTML response
  - raises RuntimeError if expectations are not met.

Header assumptions for tagging_snps.tsv (same as your local file):

    inversion_region, chromosome, region_start, region_end,
    site_index, position, position_hg38, chromosome_hg38,
    direct_group_size, inverted_group_size,
    allele_freq_direct, allele_freq_inverted, allele_freq_difference,
    correlation,
    A_inv_freq, C_inv_freq, G_inv_freq, T_inv_freq,
    A_dir_freq, C_dir_freq, G_dir_freq, T_dir_freq,
    chromosome_hg37, position_hg37
"""

import csv
import io
import math
import sys
import time
import zipfile
from dataclasses import dataclass
from urllib.request import urlopen

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# ----------------------------------------------------------------------
# Constants / configuration
# ----------------------------------------------------------------------

TSV_ZIP_URL = (
    "https://github.com/SauersML/ferromic/raw/refs/heads/main/data/tagging_snps.tsv.zip"
)

# Dataset ID from your example
DATASET_ID = "6319c068-fd59-46d8-85ee-82d82482eb14"

AZPHEWAS_URL_TEMPLATE = (
    "https://azphewas.com/variantView/{dataset_id}/{chrom}-{pos}-{a1}-{a2}/vlr/binary"
)


@dataclass
class Region:
    name: str       # human-readable label, e.g. "chr10:79,542,901–80,217,413"
    chrom_hg38: str # e.g. "10"
    start: int      # hg38 start (inclusive)
    end: int        # hg38 end (inclusive)


REGIONS = [
    Region("chr10:79,542,901–80,217,413",   "10",  79_542_901,  80_217_413),
    Region("chr8:7,301,024–12,598,379",     "8",    7_301_024,  12_598_379),
    Region("chr12:46,896,694–46,915,975",   "12",  46_896_694,  46_915_975),
    Region("chr6:141,866,310–141,898,728",  "6",  141_866_310, 141_898_728),
]


# ----------------------------------------------------------------------
# Selenium / browser helpers
# ----------------------------------------------------------------------


def make_driver() -> webdriver.Chrome:
    """
    Create a headless Chrome WebDriver.

    Requires:
      - selenium installed (`pip install selenium`)
      - chromedriver available on PATH
    """
    options = Options()
    # Headless mode
    options.add_argument("--headless=new")  # or "--headless" for older chromes
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/122.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(options=options)
    return driver


def make_azphewas_url(chrom: str, pos: int, a1: str, a2: str) -> str:
    """Format the azphewas variantView URL."""
    return AZPHEWAS_URL_TEMPLATE.format(
        dataset_id=DATASET_ID,
        chrom=chrom,
        pos=pos,
        a1=a1,
        a2=a2,
    )


def fetch_variant_html_with_browser(
    driver: webdriver.Chrome,
    chrom: str,
    pos: int,
    a1: str,
    a2: str,
    delay_seconds: float = 3.0,
) -> tuple[str, str]:
    """
    Use a real browser (Selenium) to fetch the rendered HTML.

    1. driver.get(url)
    2. sleep a bit to give the React app time to load and make API calls
    3. return the final driver.page_source

    Returns:
        (url, html_text)
    """
    url = make_azphewas_url(chrom, pos, a1, a2)
    print(f"# QUERY URL: {url}", file=sys.stderr, flush=True)

    try:
        driver.get(url)
        # Wait for the JS app to execute; tweak if needed
        time.sleep(delay_seconds)
        html = driver.page_source
    except WebDriverException as e:
        print(f"# ERROR fetching {url} via browser: {e!r}", file=sys.stderr, flush=True)
        html = ""

    return url, html


def contains_token_flag(text: str, token: str) -> str:
    """Return 'CONTAINS' or 'NOT_CONTAINS' depending on whether token is in text."""
    return "CONTAINS" if token in text else "NOT_CONTAINS"


# ----------------------------------------------------------------------
# Allele utilities
# ----------------------------------------------------------------------


def determine_alleles(row: dict) -> tuple[str, str, str]:
    """
    Given a typed row dict with *_inv_freq and *_dir_freq fields,
    determine:
        - alleles: a canonical "X/Y" string (sorted)
        - inversion_allele: allele most common in the inverted group
        - direct_allele: allele most common in the direct group
    """
    bases = ("A", "C", "G", "T")
    inv_freqs = [
        row["A_inv_freq"],
        row["C_inv_freq"],
        row["G_inv_freq"],
        row["T_inv_freq"],
    ]
    dir_freqs = [
        row["A_dir_freq"],
        row["C_dir_freq"],
        row["G_dir_freq"],
        row["T_dir_freq"],
    ]

    inv_idx = max(range(4), key=lambda i: inv_freqs[i])
    dir_idx = max(range(4), key=lambda i: dir_freqs[i])

    inversion_allele = bases[inv_idx]
    direct_allele = bases[dir_idx]

    # Canonical allele pair string (order not important here)
    alleles = "/".join(sorted({inversion_allele, direct_allele}))

    return alleles, inversion_allele, direct_allele


# ----------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------


def run_preflight_checks(driver: webdriver.Chrome) -> None:
    """
    Pre-flight sanity checks on azphewas behavior before processing all SNPs.

    1) 17-45996523-A-G:
         - must NOT contain "are not available for this dataset"
         - MUST contain "Consequence type"

    2) 17-45996523-G-A:
         - MUST contain "are not available for this dataset"
         - must NOT contain "Consequence type"

    For each:
        - print raw query URL
        - print raw HTML response
        - crash (RuntimeError) if expectations are not met
    """
    token_na = "are not available for this dataset"
    token_ct = "Consequence type"

    # ---- Check 1: 17-45996523-A-G ----
    print("# Pre-flight check 1: 17-45996523-A-G", file=sys.stderr, flush=True)
    url1, html1 = fetch_variant_html_with_browser(driver, "17", 45996523, "A", "G")
    print("# Pre-flight check 1 URL:", url1, file=sys.stderr)
    print("# Pre-flight check 1 RAW HTML BEGIN", file=sys.stderr)
    print(html1, file=sys.stderr)
    print("# Pre-flight check 1 RAW HTML END", file=sys.stderr)

    cond1_na = token_na not in html1
    cond1_ct = token_ct in html1
    if not (cond1_na and cond1_ct):
        raise RuntimeError(
            "Pre-flight check 1 failed: expected HTML to NOT contain "
            f"'{token_na}' and to CONTAIN '{token_ct}'."
        )

    # ---- Check 2: 17-45996523-G-A ----
    print("# Pre-flight check 2: 17-45996523-G-A", file=sys.stderr, flush=True)
    url2, html2 = fetch_variant_html_with_browser(driver, "17", 45996523, "G", "A")
    print("# Pre-flight check 2 URL:", url2, file=sys.stderr)
    print("# Pre-flight check 2 RAW HTML BEGIN", file=sys.stderr)
    print(html2, file=sys.stderr)
    print("# Pre-flight check 2 RAW HTML END", file=sys.stderr)

    cond2_na = token_na in html2
    cond2_ct = token_ct not in html2
    if not (cond2_na and cond2_ct):
        raise RuntimeError(
            "Pre-flight check 2 failed: expected HTML to CONTAIN "
            f"'{token_na}' and to NOT CONTAIN '{token_ct}'."
        )


# ----------------------------------------------------------------------
# Tagging SNP parsing
# ----------------------------------------------------------------------


def download_zip_bytes(url: str) -> bytes:
    """Download the zip file as raw bytes (for tagging_snps.tsv.zip)."""
    with urlopen(url) as resp:
        return resp.read()


def load_region_hits() -> dict[str, list[dict]]:
    """
    Download tagging_snps.tsv.zip and collect SNPs that fall within
    each of the inversion regions defined in REGIONS.

    Returns:
        region_hits: dict mapping region.name -> list of typed row dicts
    """
    region_hits: dict[str, list[dict]] = {reg.name: [] for reg in REGIONS}
    zip_bytes = download_zip_bytes(TSV_ZIP_URL)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tsv_names = [name for name in zf.namelist() if name.endswith(".tsv")]
        if not tsv_names:
            raise RuntimeError("No .tsv file found inside tagging_snps.tsv.zip")
        tsv_name = tsv_names[0]

        with zf.open(tsv_name) as raw:
            text_file = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text_file, delimiter="\t")

            for row in reader:
                try:
                    chrom_hg38 = str(row["chromosome_hg38"]).strip()
                    pos_hg38 = int(row["position_hg38"])

                    correlation = float(row["correlation"])
                    allele_freq_difference = float(row["allele_freq_difference"])

                    direct_group_size = int(row["direct_group_size"])
                    inverted_group_size = int(row["inverted_group_size"])

                    allele_freq_direct = float(row["allele_freq_direct"])
                    allele_freq_inverted = float(row["allele_freq_inverted"])

                    A_inv = float(row["A_inv_freq"])
                    C_inv = float(row["C_inv_freq"])
                    G_inv = float(row["G_inv_freq"])
                    T_inv = float(row["T_inv_freq"])

                    A_dir = float(row["A_dir_freq"])
                    C_dir = float(row["C_dir_freq"])
                    G_dir = float(row["G_dir_freq"])
                    T_dir = float(row["T_dir_freq"])
                except (KeyError, ValueError, TypeError):
                    # Skip malformed rows
                    continue

                typed_row = {
                    "inversion_region": row.get("inversion_region", ""),
                    "chromosome_hg38": chrom_hg38,
                    "position_hg38": pos_hg38,
                    "correlation": correlation,
                    "allele_freq_difference": allele_freq_difference,
                    "direct_group_size": direct_group_size,
                    "inverted_group_size": inverted_group_size,
                    "allele_freq_direct": allele_freq_direct,
                    "allele_freq_inverted": allele_freq_inverted,
                    "A_inv_freq": A_inv,
                    "C_inv_freq": C_inv,
                    "G_inv_freq": G_inv,
                    "T_inv_freq": T_inv,
                    "A_dir_freq": A_dir,
                    "C_dir_freq": C_dir,
                    "G_dir_freq": G_dir,
                    "T_dir_freq": T_dir,
                }

                # Assign this SNP to any matching hg38 inversion region
                for reg in REGIONS:
                    if (
                        chrom_hg38 == reg.chrom_hg38
                        and reg.start <= pos_hg38 <= reg.end
                    ):
                        region_hits[reg.name].append(typed_row)

    return region_hits


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------


def main() -> None:
    driver = make_driver()
    try:
        # 1. Pre-flight checks with real browser rendering
        run_preflight_checks(driver)

        # 2. Load tagging SNPs for the inversion regions
        region_hits = load_region_hits()

        # 3. Print header for final output (TSV)
        header = [
            "region",
            "rank",
            "chromosome_hg38",
            "position_hg38",
            "chrpos_hg38",
            "alleles",
            "inversion_allele",
            "direct_allele",
            "allele1",
            "allele2",
            "url",
            "are_not_available_for_this_dataset",
            "Consequence_type",
            "correlation",
            "r2",
            "allele_freq_difference",
            "allele_freq_direct",
            "allele_freq_inverted",
            "direct_group_size",
            "inverted_group_size",
        ]
        print("\t".join(header))

        token_na = "are not available for this dataset"
        token_ct = "Consequence type"

        # 4. For each region, pick top 100 SNPs and query azphewas
        for reg in REGIONS:
            hits = region_hits.get(reg.name, [])
            if not hits:
                continue

            # Sort by |correlation|, then by allele_freq_difference (both descending)
            hits_sorted = sorted(
                hits,
                key=lambda r: (abs(r["correlation"]), r["allele_freq_difference"]),
                reverse=True,
            )
            top_hits = hits_sorted[:100]

            for rank, row in enumerate(top_hits, start=1):
                chrom = row["chromosome_hg38"]
                pos = row["position_hg38"]

                alleles, inversion_allele, direct_allele = determine_alleles(row)
                r = row["correlation"]
                r2 = r * r
                chrpos = f"chr{chrom}:{pos}"

                # Two allele orders: inversion/direct and direct/inversion
                for a1, a2 in (
                    (inversion_allele, direct_allele),
                    (direct_allele, inversion_allele),
                ):
                    url, html = fetch_variant_html_with_browser(driver, chrom, pos, a1, a2)

                    flag_na = contains_token_flag(html, token_na)
                    flag_ct = contains_token_flag(html, token_ct)

                    out_row = [
                        reg.name,
                        str(rank),
                        chrom,
                        str(pos),
                        chrpos,
                        alleles,
                        inversion_allele,
                        direct_allele,
                        a1,
                        a2,
                        url,
                        flag_na,
                        flag_ct,
                        f"{r:.6f}",
                        f"{r2:.6f}",
                        f"{row['allele_freq_difference']:.6f}",
                        f"{row['allele_freq_direct']:.6f}",
                        f"{row['allele_freq_inverted']:.6f}",
                        str(row["direct_group_size"]),
                        str(row["inverted_group_size"]),
                    ]
                    print("\t".join(out_row))

    finally:
        # Clean up browser
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
