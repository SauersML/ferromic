import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact, pearsonr, spearmanr


# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------

DOI = "doi:10.7910/DVN/7RVV9N"
BASE_URL = "https://dataverse.harvard.edu"

OUTPUT_DIR = Path("stats")
OUTPUT_DIR.mkdir(exist_ok=True)

INV_PROPERTIES_PATH = Path("data/inv_properties.tsv")

SELECTION_GZ_NAME = "Selection_Summary_Statistics_01OCT2025.tsv.gz"
SELECTION_TSV_NAME = "Selection_Summary_Statistics_01OCT2025.tsv"
SELECTION_TSV_PATH = OUTPUT_DIR / SELECTION_TSV_NAME

PHY_ZIP_URL = "https://sharedspace.s3.msi.umn.edu/public_internet/all_phy/phy_files.zip"
PHY_ZIP_LOCAL = Path("phy_files.zip")
PHY_DIR = Path("phy_files")

FST_DATA_PATH = Path("data/FST_data.tsv")

MIN_HAP_PER_GROUP = 10  # haploid observations required per group at a SNP


# ----------------------------------------------------------------------
# Generic utilities
# ----------------------------------------------------------------------

def get_dataset_metadata():
    api_url = f"{BASE_URL}/api/datasets/:persistentId/?persistentId={DOI}"
    print("Fetching dataset metadata...")
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))

    status = data.get("status")
    if status != "OK":
        raise ValueError(f"API returned non-OK status: {status}")

    return data["data"]


def calculate_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def format_size(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(n_bytes)
    for u in units:
        if size < 1024.0:
            return f"{size:.1f} {u}"
        size /= 1024.0
    return f"{size:.1f} TB"


def unzip_file(gz_path: Path) -> Path:
    if gz_path.suffix == ".gz":
        out_path = gz_path.with_suffix("")
    else:
        out_path = Path(str(gz_path).replace(".gz", ""))

    if out_path.exists():
        print(f"✓ Unzipped file {out_path.name} already exists")
        return out_path

    print(f"Unzipping {gz_path.name}...")

    total = gz_path.stat().st_size
    read_bytes = 0
    chunk = 8192 * 1024

    with gzip.open(gz_path, "rb") as fin, out_path.open("wb") as fout:
        while True:
            buf = fin.read(chunk)
            if not buf:
                break
            fout.write(buf)
            read_bytes += len(buf)
            if total > 0:
                pct = read_bytes / total * 100.0
                print(
                    f"\rUnzipped: {format_size(read_bytes)} "
                    f"({pct:.1f}% of {format_size(total)})",
                    end="",
                )
            else:
                print(f"\rUnzipped: {format_size(read_bytes)}", end="")

    print()
    print(f"✓ Unzipped to {out_path.name} ({format_size(out_path.stat().st_size)})")
    return out_path


def download_file(file_id: int, filename: str, expected_md5: str | None) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / filename

    if out_path.exists():
        print(f"✓ File {filename} already exists")
        if expected_md5:
            print("  Verifying MD5...")
            actual = calculate_md5(out_path)
            if actual == expected_md5:
                print("  ✓ MD5 verified")
                return out_path
            print(f"  ✗ MD5 mismatch (expected {expected_md5}, got {actual}); re-downloading...")
        else:
            return out_path

    url = f"{BASE_URL}/api/access/datafile/{file_id}"
    print(f"Downloading {filename}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, out_path.open("wb") as f:
        clen = r.headers.get("Content-Length")
        total = int(clen) if clen is not None else None
        if total:
            print(f"  File size: {format_size(total)}")

        read_bytes = 0
        chunk = 8192 * 1024
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            read_bytes += len(buf)
            mb = read_bytes / (1024 * 1024)
            if total:
                pct = read_bytes / total * 100.0
                print(f"\r  Progress: {pct:.1f}% ({mb:.1f} MB)", end="")
            else:
                print(f"\r  Downloaded: {mb:.1f} MB", end="")

    print()
    print("✓ Download complete")

    if expected_md5:
        print("  Verifying MD5...")
        actual = calculate_md5(out_path)
        if actual != expected_md5:
            out_path.unlink(missing_ok=True)
            raise ValueError(
                f"MD5 mismatch after download (expected {expected_md5}, got {actual})"
            )
        print("  ✓ MD5 verified")

    return out_path


# ----------------------------------------------------------------------
# liftOver installation and batch liftover
# ----------------------------------------------------------------------

def install_liftover():
    res = subprocess.run(
        ["which", "liftOver"],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        print("✓ liftOver already available")
        return

    print("liftOver not found; installing...")

    import platform

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine == "x86_64":
        url = "http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
    elif system == "darwin" and machine == "x86_64":
        url = "http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/liftOver"
    elif system == "darwin" and machine == "arm64":
        url = "http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.arm64/liftOver"
    else:
        raise RuntimeError(
            f"Unsupported platform for automatic liftOver install: {system} {machine}"
        )

    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    liftover_path = local_bin / "liftOver"

    print(f"  Downloading liftOver from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, liftover_path.open("wb") as f:
        f.write(r.read())

    liftover_path.chmod(0o755)

    current = os.environ.get("PATH", "")
    if str(local_bin) not in current.split(":"):
        os.environ["PATH"] = f"{local_bin}:{current}"

    print(f"✓ liftOver installed to {liftover_path}")


def liftover_coordinates_batch(regions, from_build: str, to_build: str):
    """
    Liftover regions with UCSC liftOver.

    regions:
        iterable of dicts with at least 'chrom', 'start', 'end'.
        Any extra keys are preserved in lifted outputs.

    Returns (lifted_regions, stats_dict)
    """
    if not regions:
        raise RuntimeError("No regions provided for liftover")

    install_liftover()

    tmpdir = Path(tempfile.gettempdir())
    pid = os.getpid()

    bed_in = tmpdir / f"liftover_in_{pid}.bed"
    bed_out = tmpdir / f"liftover_out_{pid}.bed"
    bed_unmapped = tmpdir / f"liftover_unmapped_{pid}.bed"

    prefix = "r_"
    region_map = {}

    with bed_in.open("w") as f:
        for i, r in enumerate(regions):
            chrom = str(r["chrom"])
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            start = int(r["start"])
            end = int(r["end"])
            rid = f"{prefix}{i}"
            f.write(f"{chrom}\t{start}\t{end}\t{rid}\n")
            region_map[rid] = dict(r)

    chain_name = f"{from_build}To{to_build.capitalize()}.over.chain.gz"
    chain_path = OUTPUT_DIR / chain_name
    if not chain_path.exists():
        url = (
            f"https://hgdownload.soe.ucsc.edu/goldenPath/"
            f"{from_build}/liftOver/{chain_name}"
        )
        print(f"Downloading chain file {chain_name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, chain_path.open("wb") as f:
            f.write(r.read())
        print("✓ Chain file downloaded")

    cmd = [
        "liftOver",
        str(bed_in),
        str(chain_path),
        str(bed_out),
        str(bed_unmapped),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if res.returncode != 0:
        bed_in.unlink(missing_ok=True)
        bed_out.unlink(missing_ok=True)
        bed_unmapped.unlink(missing_ok=True)
        raise RuntimeError(
            f"liftOver failed with code {res.returncode}: {res.stderr}"
        )

    lifted = []
    lifted_ids = set()

    with bed_out.open() as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            chrom_out, start_out, end_out, rid = fields[:4]
            base = region_map.get(rid)
            if base is None:
                continue
            lifted_ids.add(rid)
            new = dict(base)
            chrom_no_chr = chrom_out[3:] if chrom_out.startswith("chr") else chrom_out
            new["chrom"] = chrom_no_chr
            new["start"] = int(start_out)
            new["end"] = int(end_out)
            lifted.append(new)

    unmapped = []
    if bed_unmapped.exists():
        with bed_unmapped.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 4:
                    rid = parts[3]
                    original = region_map.get(rid)
                    if original is not None:
                        unmapped.append(original)

    total = len(regions)
    lifted_count = len(lifted)
    failed = total - lifted_count

    bed_in.unlink(missing_ok=True)
    bed_out.unlink(missing_ok=True)
    bed_unmapped.unlink(missing_ok=True)

    stats = {
        "total": total,
        "lifted": lifted_count,
        "failed": failed,
        "unmapped": unmapped,
    }

    if lifted_count == 0:
        raise RuntimeError("All regions failed to liftover")

    return lifted, stats


# ----------------------------------------------------------------------
# PHY management
# ----------------------------------------------------------------------

def download_and_unzip_phy():
    import zipfile

    if PHY_ZIP_LOCAL.exists():
        print(f"✓ PHY archive {PHY_ZIP_LOCAL.name} already exists")
    else:
        print(f"Downloading PHY archive from {PHY_ZIP_URL}...")
        req = urllib.request.Request(PHY_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, PHY_ZIP_LOCAL.open("wb") as f:
            clen = r.headers.get("Content-Length")
            total = int(clen) if clen is not None else None
            if total:
                print(f"  File size: {format_size(total)}")
            read_bytes = 0
            chunk = 8192 * 1024
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                read_bytes += len(buf)
                mb = read_bytes / (1024 * 1024)
                if total:
                    pct = read_bytes / total * 100.0
                    print(f"\r  Progress: {pct:.1f}% ({mb:.1f} MB)", end="")
                else:
                    print(f"\r  Downloaded: {mb:.1f} MB", end="")
        print()
        print("✓ PHY archive downloaded")

    if PHY_DIR.exists() and any(PHY_DIR.iterdir()):
        print(f"✓ PHY files already extracted in {PHY_DIR}")
        return

    print(f"Extracting {PHY_ZIP_LOCAL.name}...")
    PHY_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(PHY_ZIP_LOCAL, "r") as z:
        members = z.namelist()
        total = len(members)
        print(f"  Extracting {total} files...")
        for i, name in enumerate(members, 1):
            z.extract(name, PHY_DIR)
            if i % 100 == 0 or i == total:
                print(f"\r  Extracted {i}/{total}", end="")
    print()
    print(f"✓ Extracted PHY files to {PHY_DIR}")


def load_inversion_phy_file(path: Path):
    """
    Load an inversion PHY file of the form:

        inversion_group[0|1]_... .phy

    with nonstandard labels: SampleName_L / SampleName_R.

    Returns dict:
        sample -> {'L': seq or None, 'R': seq or None}
    """
    seqs = {}

    with path.open() as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise RuntimeError(f"Unexpected PHY header in {path.name}: {' '.join(header)}")
        # n_seqs = int(header[0])  # not strictly needed
        # seq_len = int(header[1])

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            label = parts[0]
            seq = "".join(parts[1:]).strip()

            if label.endswith("_L"):
                sample = label[:-2]
                hap = "L"
            elif label.endswith("_R"):
                sample = label[:-2]
                hap = "R"
            else:
                continue

            if sample not in seqs:
                seqs[sample] = {"L": None, "R": None}

            seqs[sample][hap] = seq

    return seqs


def calculate_allele_frequencies(seqs, pos: int):
    counts = defaultdict(int)
    total = 0

    for sample, haps in seqs.items():
        for tag in ("L", "R"):
            s = haps.get(tag)
            if not s:
                continue
            if pos >= len(s):
                continue
            a = s[pos]
            if a in ("A", "C", "G", "T"):
                counts[a] += 1
                total += 1

    if total == 0:
        return {"counts": {}, "freqs": {}, "total": 0}

    freqs = {a: c / total for a, c in counts.items()}
    return {"counts": dict(counts), "freqs": freqs, "total": total}


def compute_delta_p(group0_seqs, group1_seqs, pos: int):
    """
    Compute absolute allele-frequency difference Δp between group0 and group1
    at 0-based position pos.

    Returns dict with:
        delta_p, p_value, n_group0, n_group1
    or None if not computable or insufficient data.
    """
    af0 = calculate_allele_frequencies(group0_seqs, pos)
    af1 = calculate_allele_frequencies(group1_seqs, pos)

    if af0["total"] < MIN_HAP_PER_GROUP or af1["total"] < MIN_HAP_PER_GROUP:
        return None

    alleles = set(af0["counts"]) | set(af1["counts"])
    if len(alleles) < 2:
        return None

    # Use the most common allele across both groups as reference
    best = None
    for a in alleles:
        c0 = af0["counts"].get(a, 0)
        c1 = af1["counts"].get(a, 0)
        comb = c0 + c1
        if best is None or comb > best[1]:
            best = (a, comb)
    ref = best[0]

    ref0 = af0["counts"].get(ref, 0)
    ref1 = af1["counts"].get(ref, 0)
    alt0 = af0["total"] - ref0
    alt1 = af1["total"] - ref1

    p0 = ref0 / af0["total"]
    p1 = ref1 / af1["total"]
    delta_p = abs(p0 - p1)

    # Fisher's exact as descriptive; no gating
    table = [[ref0, alt0], [ref1, alt1]]
    p_val = fisher_exact(table)[1]

    return {
        "delta_p": delta_p,
        "p_value": p_val,
        "n_group0": af0["total"],
        "n_group1": af1["total"],
    }


# ----------------------------------------------------------------------
# Selection data
# ----------------------------------------------------------------------

def ensure_selection_data():
    if SELECTION_TSV_PATH.exists():
        print(f"✓ Selection TSV present at {SELECTION_TSV_PATH}")
        return

    print("Selection TSV missing; downloading via Dataverse metadata...")
    meta = get_dataset_metadata()
    files = meta["latestVersion"]["files"]

    target = SELECTION_GZ_NAME
    for fmeta in files:
        df = fmeta.get("dataFile", {})
        name = df.get("filename")
        if name != target:
            continue
        file_id = df["id"]
        checksum = df.get("checksum", {})
        expected = checksum.get("value") if checksum.get("type") == "MD5" else None

        gz_path = download_file(file_id, name, expected)
        out = unzip_file(gz_path)
        if out.name != SELECTION_TSV_NAME:
            out.rename(SELECTION_TSV_PATH)
        print(f"✓ Selection TSV available at {SELECTION_TSV_PATH}")
        return

    raise RuntimeError(f"Selection file {target} not found in dataset metadata")


def load_selection_hg19():
    """
    Load selection summary statistics as provided.

    Assumes:
        CHROM: chromosome (no 'chr')
        POS: 1-based position in hg19
        S: selection coefficient
        FILTER: 'PASS' filter column

    Returns DataFrame with:
        chrom, pos_hg19_1based, pos_hg19_0based, s
    """
    ensure_selection_data()

    df = pd.read_csv(
        SELECTION_TSV_PATH,
        sep="\t",
        comment="#",
        dtype={"CHROM": str},
    )

    required = {"CHROM", "POS", "S", "FILTER"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Selection TSV missing required columns: {missing}")

    df = df[df["FILTER"] == "PASS"].copy()

    df.rename(columns={"CHROM": "chrom", "POS": "pos_hg19_1based", "S": "s"}, inplace=True)
    df["pos_hg19_0based"] = df["pos_hg19_1based"] - 1
    df["chrom"] = df["chrom"].astype(str)

    print(f"✓ Loaded {len(df)} PASS selection records (hg19 coordinates)")
    return df


def lift_selection_to_hg38(sel_hg19: pd.DataFrame) -> pd.DataFrame:
    """
    Liftover per-SNP selection coordinates from hg19 to hg38.

    Returns DataFrame with:
        chrom, pos_hg38_1based, pos_hg38_0based, s
    """
    print("\nLifting selection coordinates from hg19 to hg38 for phy/FST analyses...")
    regions = []
    for idx, row in sel_hg19.iterrows():
        regions.append(
            {
                "chrom": row["chrom"],
                "start": int(row["pos_hg19_0based"]),
                "end": int(row["pos_hg19_0based"]) + 1,
                "idx": int(idx),
            }
        )

    lifted, stats = liftover_coordinates_batch(regions, "hg19", "hg38")
    print(
        f"  Liftover: total={stats['total']}, "
        f"lifted={stats['lifted']}, failed={stats['failed']}"
    )

    rows = []
    for r in lifted:
        i = r["idx"]
        s_val = float(sel_hg19.at[i, "s"])
        start = int(r["start"])
        chrom = str(r["chrom"])
        rows.append(
            {
                "chrom": chrom,
                "pos_hg38_0based": start,
                "pos_hg38_1based": start + 1,
                "s": s_val,
            }
        )

    df = pd.DataFrame(rows)
    print(f"✓ {len(df)} selection records mapped to hg38")
    return df


# ----------------------------------------------------------------------
# Inversion metadata and helpers
# ----------------------------------------------------------------------

def load_inversion_properties():
    """
    Load inversion properties (hg38) from INV_PROPERTIES_PATH.
    """
    if not INV_PROPERTIES_PATH.exists():
        raise RuntimeError(f"Missing inversion properties file: {INV_PROPERTIES_PATH}")

    invs = []
    with INV_PROPERTIES_PATH.open() as f:
        header = f.readline().strip().split("\t")
        col_idx = {name: i for i, name in enumerate(header)}

        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            chrom = fields[col_idx["Chromosome"]].replace("chr", "")
            start = int(fields[col_idx["Start"]])
            end = int(fields[col_idx["End"]])
            size = end - start

            num_rec = None
            if "Number_recurrent_events" in col_idx:
                v = fields[col_idx["Number_recurrent_events"]]
                if v not in ("", "NA", "na"):
                    try:
                        num_rec = int(v)
                    except ValueError:
                        num_rec = None

            inv_af = None
            if "Inverted_AF" in col_idx:
                v = fields[col_idx["Inverted_AF"]]
                if v not in ("", "NA", "na"):
                    try:
                        inv_af = float(v)
                    except ValueError:
                        inv_af = None

            invs.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "size": size,
                    "num_recurrent": num_rec,
                    "inverted_af": inv_af,
                }
            )

    print(f"✓ Loaded {len(invs)} inversions from {INV_PROPERTIES_PATH}")
    return invs


def load_inversions_with_phy():
    """
    Return inversions (hg38) that have both inversion_group0 and inversion_group1 PHY files.
    """
    inversions = load_inversion_properties()
    usable = []

    for inv in inversions:
        chrom = inv["chrom"]
        start = inv["start"]
        end = inv["end"]
        g0 = PHY_DIR / f"inversion_group0_{chrom}_start{start}_end{end}.phy"
        g1 = PHY_DIR / f"inversion_group1_{chrom}_start{start}_end{end}.phy"
        if g0.exists() and g1.exists():
            item = dict(inv)
            item["group0_phy"] = g0
            item["group1_phy"] = g1
            usable.append(item)

    print(f"✓ {len(usable)} inversions have inversion_group0/1 PHY pairs")
    return usable


# ----------------------------------------------------------------------
# Analysis 1: Mean S vs inversion properties (using lifted selection)
# ----------------------------------------------------------------------

def compare_selection_coefficients(selection_hg38: pd.DataFrame):
    """
    Descriptive summaries of mean selection coefficient per inversion
    and simple associations with inversion properties.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS: MEAN SELECTION COEFFICIENT PER INVERSION")
    print("=" * 80 + "\n")

    inversions = load_inversion_properties()

    if not inversions:
        print("✗ No inversions loaded")
        return

    # Use inversions as-is in hg38; selection_hg38 already in hg38
    # De-overlap inversions (>10% reciprocal) to avoid redundancy
    by_chrom = defaultdict(list)
    for idx, inv in enumerate(inversions):
        by_chrom[inv["chrom"]].append((idx, inv))

    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda x: x[1]["start"])

    excluded = set()
    for chrom, lst in by_chrom.items():
        n = len(lst)
        for i in range(n):
            idx_i, inv_i = lst[i]
            if idx_i in excluded:
                continue
            for j in range(i + 1, n):
                idx_j, inv_j = lst[j]
                if idx_j in excluded:
                    continue
                if inv_j["start"] >= inv_i["end"]:
                    break
                ov_start = max(inv_i["start"], inv_j["start"])
                ov_end = min(inv_i["end"], inv_j["end"])
                if ov_start < ov_end:
                    ov = ov_end - ov_start
                    if (
                        ov / (inv_i["end"] - inv_i["start"]) > 0.10
                        or ov / (inv_j["end"] - inv_j["start"]) > 0.10
                    ):
                        # Drop the smaller inversion
                        if inv_i["end"] - inv_i["start"] >= inv_j["end"] - inv_j["start"]:
                            excluded.add(idx_j)
                        else:
                            excluded.add(idx_i)
                        break

    non_overlap = [
        inv for idx, inv in enumerate(inversions) if idx not in excluded
    ]
    print(f"  Non-overlapping inversions retained: {len(non_overlap)}")

    if not non_overlap:
        print("✗ No inversions remain after overlap filtering")
        return

    # Index selection by chrom for fast lookup
    sel_by_chrom = defaultdict(list)
    for _, row in selection_hg38.iterrows():
        sel_by_chrom[row["chrom"]].append((int(row["pos_hg38_1based"]), float(row["s"])))

    for chrom in sel_by_chrom:
        sel_by_chrom[chrom].sort(key=lambda x: x[0])

    summaries = []

    for i, inv in enumerate(non_overlap):
        chrom = inv["chrom"]
        start = inv["start"]
        end = inv["end"]
        pts = sel_by_chrom.get(chrom, [])
        if not pts:
            continue

        s_vals = []
        # brute force scan; number of points per chrom is manageable
        for pos, sval in pts:
            if start <= pos <= end:
                s_vals.append(sval)
        if not s_vals:
            continue

        mean_s = float(np.mean(s_vals))
        summaries.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "size_bp": inv["size"],
                "num_recurrent": inv["num_recurrent"],
                "inverted_af": inv["inverted_af"],
                "n_snps": len(s_vals),
                "mean_s": mean_s,
            }
        )

    if not summaries:
        print("✗ No inversions had mapped selection coefficients")
        return

    df = pd.DataFrame(summaries)
    df["log_size"] = np.log(df["size_bp"])
    df["log_n_snps"] = np.log(df["n_snps"])

    out_path = OUTPUT_DIR / "selection_comparison_results.txt"
    with out_path.open("w") as out:
        out.write("Analysis: Mean Selection Coefficient vs Inversion Properties\n")
        out.write("=" * 80 + "\n\n")
        out.write(
            f"Non-overlapping inversions with ≥1 SNP: {len(df)}\n\n"
        )
        out.write("Mean S across inversions:\n")
        out.write(f"  Mean:   {df['mean_s'].mean():.6f}\n")
        out.write(f"  Median: {df['mean_s'].median():.6f}\n")
        out.write(f"  Min:    {df['mean_s'].min():.6f}\n")
        out.write(f"  Max:    {df['mean_s'].max():.6f}\n\n")

        numeric = [
            "mean_s",
            "num_recurrent",
            "inverted_af",
            "size_bp",
            "n_snps",
            "log_size",
            "log_n_snps",
        ]
        cols = [c for c in numeric if c in df.columns]
        if len(cols) > 1:
            corr = df[cols].corr()
            out.write("Pairwise Pearson correlations (descriptive only):\n")
            out.write(corr.to_string(float_format=lambda v: f"{v: .3f}"))
            out.write("\n")

    print(f"✓ Descriptive inversion summaries written to {out_path}")

    # Simple visualization: recurrence vs mean_s when available
    if "num_recurrent" in df.columns and df["num_recurrent"].notna().any():
        valid = df["num_recurrent"].notna()
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = 20 + 200 * (df.loc[valid, "n_snps"] / df.loc[valid, "n_snps"].max())
        ax.scatter(
            df.loc[valid, "num_recurrent"],
            df.loc[valid, "mean_s"],
            s=sizes,
            alpha=0.4,
            edgecolors="black",
            linewidth=0.5,
            label="Inversions (point size ∝ #SNPs)",
        )
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Number of recurrent events")
        ax.set_ylabel("Mean selection coefficient (S)")
        ax.set_title("Mean S vs recurrence count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "recurrence_selection_scatter.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✓ Plot written to {plot_path}")

    print("\n" + "=" * 80)


# ----------------------------------------------------------------------
# Analysis 2: Per-inversion Δp from inversion PHY vs S
# ----------------------------------------------------------------------

def analyze_inversion_differentiation(selection_hg38: pd.DataFrame):
    """
    For each inversion with an inversion_group0/1 PHY pair:

        - Collect SNPs with S in that inversion (hg38).
        - At each SNP, compute Δp between group0 and group1 from PHY.
        - Require ≥ MIN_HAP_PER_GROUP alleles in each group.
        - For each inversion, pick SNP(s) with maximal Δp.
        - If multiple SNPs tie, average S across them.
        - Summarize per-inversion metrics and per-SNP association.

    Produces:
        stats/inversion_differentiation_results.tsv
        stats/inversion_differentiation_scatter.png
        stats/inversion_differentiation_snp.tsv
    """
    print("\n" + "=" * 80)
    print("ANALYSIS: INVERSION Δp (PHY) vs SELECTION COEFFICIENTS")
    print("=" * 80 + "\n")

    inversions = load_inversions_with_phy()
    if not inversions:
        print("✗ No inversions with inversion_group0/1 PHY files")
        return

    # Index selection SNPs by chrom
    sel_by_chrom = defaultdict(list)
    for _, row in selection_hg38.iterrows():
        sel_by_chrom[row["chrom"]].append(
            (int(row["pos_hg38_0based"]), float(row["s"]))
        )
    for chrom in sel_by_chrom:
        sel_by_chrom[chrom].sort(key=lambda x: x[0])

    per_inv = []
    all_snp = []

    for inv in inversions:
        chrom = inv["chrom"]
        start = inv["start"]
        end = inv["end"]
        g0_path = inv["group0_phy"]
        g1_path = inv["group1_phy"]

        print(f"  Inversion chr{chrom}:{start}-{end}")

        snps = sel_by_chrom.get(chrom, [])
        if not snps:
            print("    No selection SNPs on this chromosome")
            continue

        # SNPs in region (0-based positions in [start-1, end))
        region_snps = [
            (pos0, s_val)
            for pos0, s_val in snps
            if start - 1 <= pos0 < end
        ]
        if not region_snps:
            print("    No selection SNPs inside inversion")
            continue

        g0 = load_inversion_phy_file(g0_path)
        g1 = load_inversion_phy_file(g1_path)

        best_delta = None
        best_entries = []

        for pos0, s_val in region_snps:
            phy_pos = pos0 - (start - 1)
            if phy_pos < 0:
                continue

            diff = compute_delta_p(g0, g1, phy_pos)
            if diff is None:
                continue

            dp = diff["delta_p"]

            all_snp.append(
                {
                    "chrom": chrom,
                    "inv_start": start,
                    "inv_end": end,
                    "pos_hg38_0based": pos0,
                    "pos_hg38_1based": pos0 + 1,
                    "S": s_val,
                    "abs_S": abs(s_val),
                    "delta_p": dp,
                    "p_value": diff["p_value"],
                    "n_group0": diff["n_group0"],
                    "n_group1": diff["n_group1"],
                }
            )

            if best_delta is None or dp > best_delta:
                best_delta = dp
                best_entries = [(dp, s_val)]
            elif dp == best_delta:
                best_entries.append((dp, s_val))

        if best_delta is None:
            print("    No SNPs with valid Δp")
            continue

        abs_s_vals = [abs(s) for (_, s) in best_entries]
        mean_abs_s = float(np.mean(abs_s_vals))

        per_inv.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "max_delta_p": best_delta,
                "n_snp_max_delta": len(best_entries),
                "mean_abs_S_at_max_delta": mean_abs_s,
            }
        )

        print(
            f"    Max Δp = {best_delta:.4f} "
            f"(n SNPs at max: {len(best_entries)}, mean |S|={mean_abs_s:.6g})"
        )

    if not per_inv:
        print("\n✗ No inversions with valid Δp and S")
        return

    inv_df = pd.DataFrame(per_inv)
    inv_out = OUTPUT_DIR / "inversion_differentiation_results.tsv"
    inv_df.to_csv(inv_out, sep="\t", index=False)
    print(f"\n✓ Per-inversion differentiation summary written to {inv_out}")

    if all_snp:
        snp_df = pd.DataFrame(all_snp)
        snp_out = OUTPUT_DIR / "inversion_differentiation_snp.tsv"
        snp_df.to_csv(snp_out, sep="\t", index=False)
        print(f"✓ Per-SNP differentiation data written to {snp_out}")
    else:
        snp_df = None

    # Correlation at inversion level
    if len(inv_df) >= 3:
        r_p, p_p = pearsonr(inv_df["max_delta_p"], inv_df["mean_abs_S_at_max_delta"])
        r_s, p_s = spearmanr(inv_df["max_delta_p"], inv_df["mean_abs_S_at_max_delta"])
        print("\nPer-inversion correlation (max Δp vs mean |S| at max Δp SNP):")
        print(f"  Pearson r = {r_p:.4f}, p = {p_p:.3e}")
        print(f"  Spearman ρ = {r_s:.4f}, p = {p_s:.3e}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            inv_df["max_delta_p"],
            inv_df["mean_abs_S_at_max_delta"],
            alpha=0.7,
        )
        ax.set_xlabel("Max Δp within inversion")
        ax.set_ylabel("Mean |S| at SNPs with max Δp")
        ax.set_title("Per-inversion: Δp vs |S|")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "inversion_differentiation_scatter.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✓ Per-inversion scatter plot written to {plot_path}")

    # Optional per-SNP correlation (global)
    if snp_df is not None and len(snp_df) >= 3:
        r_snp, p_snp = spearmanr(snp_df["delta_p"], snp_df["abs_S"])
        print("\nPer-SNP association (Δp vs |S| across all inversion SNPs):")
        print(f"  Spearman ρ = {r_snp:.4f}, p = {p_snp:.3e}")

    print("\n" + "=" * 80)


# ----------------------------------------------------------------------
# Analysis 3: FST vs |S| at most differentiated SNP per inversion
# ----------------------------------------------------------------------

def analyze_fst_vs_max_differentiation(selection_hg38: pd.DataFrame):
    """
    For each inversion locus defined in FST_data.tsv:

        - Identify its inversion_group0/1 PHY pair.
        - Within that locus, consider all SNPs with S (hg38).
        - For each SNP, compute Δp between group0 and group1.
        - Choose SNP with maximal Δp; record |S| at that SNP.
        - Correlate this |S| with FST.

    One point per inversion.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS: FST vs |S| at most differentiated SNP (per inversion)")
    print("=" * 80 + "\n")

    if not FST_DATA_PATH.exists():
        print(f"✗ FST data file not found: {FST_DATA_PATH}")
        return

    fst_df = pd.read_csv(FST_DATA_PATH, sep="\t", dtype={"chr": str})
    if fst_df.empty:
        print("✗ Empty FST data")
        return

    results = []

    # Index selection by chrom for fast region queries
    sel_by_chrom = defaultdict(list)
    for _, row in selection_hg38.iterrows():
        sel_by_chrom[row["chrom"]].append(
            (int(row["pos_hg38_0based"]), float(row["s"]))
        )
    for chrom in sel_by_chrom:
        sel_by_chrom[chrom].sort(key=lambda x: x[0])

    for i, row in fst_df.iterrows():
        chrom = str(row["chr"])
        start0 = int(row["region_start_0based"])
        end0 = int(row["region_end_0based"])
        fst_val = float(row["FST"])

        # Convert FST interval (0-based, half-open) to inversion PHY naming (1-based inclusive)
        start1 = start0 + 1
        end1 = end0

        g0_path = PHY_DIR / f"inversion_group0_{chrom}_start{start1}_end{end1}.phy"
        g1_path = PHY_DIR / f"inversion_group1_{chrom}_start{start1}_end{end1}.phy"

        print(
            f"  [{i+1}/{len(fst_df)}] chr{chrom}:{start0}-{end0} "
            f"(FST={fst_val:.4f})"
        )

        if not (g0_path.exists() and g1_path.exists()):
            print("    Skipping: missing inversion_group0/1 PHY pair")
            continue

        snps = sel_by_chrom.get(chrom, [])
        if not snps:
            print("    No selection SNPs on this chromosome")
            continue

        # SNPs with hg38 0-based positions inside [start0, end0)
        region_snps = [
            (pos0, s_val)
            for pos0, s_val in snps
            if start0 <= pos0 < end0
        ]
        if not region_snps:
            print("    No selection SNPs in FST interval")
            continue

        g0 = load_inversion_phy_file(g0_path)
        g1 = load_inversion_phy_file(g1_path)

        best_delta = None
        best_s_vals = []
        best_pos = None
        best_meta = None

        for pos0, s_val in region_snps:
            phy_pos = pos0 - start0
            if phy_pos < 0:
                continue

            diff = compute_delta_p(g0, g1, phy_pos)
            if diff is None:
                continue

            dp = diff["delta_p"]
            if best_delta is None or dp > best_delta:
                best_delta = dp
                best_s_vals = [s_val]
                best_pos = pos0
                best_meta = diff
            elif dp == best_delta:
                best_s_vals.append(s_val)

        if best_delta is None:
            print("    No SNPs with valid Δp")
            continue

        abs_s = float(np.mean([abs(s) for s in best_s_vals]))

        print(
            f"    Max Δp = {best_delta:.4f} at pos≈{best_pos} "
            f"(mean |S| at max Δp SNPs = {abs_s:.6g})"
        )

        results.append(
            {
                "chr": chrom,
                "region_start_0based": start0,
                "region_end_0based": end0,
                "FST": fst_val,
                "max_delta_p": best_delta,
                "mean_abs_S_at_max_delta": abs_s,
                "n_snp_max_delta": len(best_s_vals),
                "n_group0_at_max_delta": best_meta["n_group0"],
                "n_group1_at_max_delta": best_meta["n_group1"],
            }
        )

    if not results:
        print("\n✗ No FST inversions with usable PHY+S data")
        return

    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "fst_vs_max_diff_selection.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n✓ FST vs max-Δp/|S| summary written to {out_path}")

    clean = df.dropna(subset=["FST", "mean_abs_S_at_max_delta"])
    if len(clean) < 3:
        print(f"✗ Not enough data points for correlation (n={len(clean)})")
        return

    r_p, p_p = pearsonr(clean["FST"], clean["mean_abs_S_at_max_delta"])
    r_s, p_s = spearmanr(clean["FST"], clean["mean_abs_S_at_max_delta"])

    print("\nCorrelation: FST vs |S| at most differentiated SNP (per inversion)")
    print(f"  Pearson r = {r_p:.4f}, p = {p_p:.3e}")
    print(f"  Spearman ρ = {r_s:.4f}, p = {p_s:.3e}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(clean["FST"], clean["mean_abs_S_at_max_delta"], alpha=0.7)
    z = np.polyfit(clean["FST"], clean["mean_abs_S_at_max_delta"], 1)
    x_line = np.linspace(clean["FST"].min(), clean["FST"].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), linestyle="--")
    ax.set_xlabel("FST")
    ax.set_ylabel("|S| at most differentiated SNP")
    ax.set_title("Per-inversion: FST vs |S| at max Δp SNP")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "fst_vs_max_diff_selection.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ FST correlation scatter plot written to {plot_path}")

    print("\n" + "=" * 80)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    if not INV_PROPERTIES_PATH.exists():
        print(f"✗ Required file missing: {INV_PROPERTIES_PATH}")
        sys.exit(1)

    download_and_unzip_phy()
    ensure_selection_data()

    # Selection is provided in hg19
    sel_hg19 = load_selection_hg19()
    selection_hg38 = lift_selection_to_hg38(sel_hg19)

    compare_selection_coefficients(selection_hg38)
    analyze_inversion_differentiation(selection_hg38)
    analyze_fst_vs_max_differentiation(selection_hg38)


if __name__ == "__main__":
    main()
