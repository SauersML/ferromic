"""Locate the strongest tagging SNP for a given inversion region and report selection stats.

This script downloads the latest ``tagging-snps-report`` artifact produced by the
``run_find_tagging_snps.yml`` workflow, filters the table to the requested
``chrom:start-end`` region, and selects the variant with the highest absolute
correlation (``|r|``). It also downloads the selection summary statistics from
Dataverse (doi:10.7910/DVN/7RVV9N) and attempts to annotate the tagging SNP using
GRCh37/hg19 coordinates.

Outputs are written to ``outputs/<region>_best_tagging_snp.txt`` and also
printed to stdout for GitHub Actions log visibility.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# Constants for locating artifacts and selection statistics
ARTIFACT_NAME = "tagging-snps-report"
ARTIFACT_WORKFLOW_PATH = ".github/workflows/run_find_tagging_snps.yml"
DATAVERSE_DOI = "doi:10.7910/DVN/7RVV9N"
DATAVERSE_BASE = "https://dataverse.harvard.edu"
SELECTION_GZ_NAME = "Selection_Summary_Statistics_01OCT2025.tsv.gz"
SELECTION_TSV_NAME = "Selection_Summary_Statistics_01OCT2025.tsv"
OUTPUT_DIR = Path("outputs")
SELECTION_DIR = OUTPUT_DIR / "selection_data"
SELECTION_TSV_PATH = SELECTION_DIR / SELECTION_TSV_NAME
REPO_ENV = "GITHUB_REPOSITORY"
TOKEN_ENVS = ("GITHUB_TOKEN", "GH_TOKEN")


class ArtifactError(RuntimeError):
    """Raised when artifact discovery or download fails."""


def parse_region(region: str) -> tuple[str, int, int]:
    """Parse a region of the form ``chr12:12345-45678``.

    Returns a tuple of (chromosome_without_chr_prefix, start, end).
    """

    match = re.fullmatch(r"chr?([^:]+):(\d+)-(\d+)", region)
    if not match:
        raise ValueError(f"Invalid region string: {region!r}; expected chrN:start-end")

    chrom, start, end = match.groups()
    start_i, end_i = int(start), int(end)
    if start_i > end_i:
        raise ValueError(f"Region start {start_i} is greater than end {end_i}")

    return chrom, start_i, end_i


def github_json(url: str) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = next((os.getenv(env) for env in TOKEN_ENVS if os.getenv(env)), None)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def find_latest_artifact(repo: str, name: str, workflow_path: str | None = None) -> dict:
    """Return metadata for the newest non-expired artifact with the given name.

    If ``workflow_path`` is provided, only artifacts created by that workflow file
    (e.g., ``.github/workflows/run_find_tagging_snps.yml``) are considered.
    """

    page = 1
    latest: Optional[dict] = None
    while True:
        url = f"https://api.github.com/repos/{repo}/actions/artifacts?per_page=100&page={page}"
        data = github_json(url)
        artifacts = data.get("artifacts", [])
        if not artifacts:
            break

        for artifact in artifacts:
            if artifact.get("name") != name or artifact.get("expired"):
                continue

            if workflow_path:
                run_meta = artifact.get("workflow_run") or {}
                if run_meta.get("path") != workflow_path:
                    continue

            if latest is None or artifact.get("created_at") > latest.get("created_at"):
                latest = artifact

        page += 1

    if latest is None:
        raise ArtifactError(f"No non-expired artifact named {name!r} found in {repo}")

    return latest


def download_artifact(artifact: dict, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = artifact.get("archive_download_url")
    if not url:
        raise ArtifactError("Artifact download URL missing")

    token = next((os.getenv(env) for env in TOKEN_ENVS if os.getenv(env)), None)
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    out_path = dest_dir / f"{artifact['name']}.zip"
    print(f"Downloading artifact {artifact['name']} ({artifact['size_in_bytes']/1_048_576:.1f} MB)...")
    with urllib.request.urlopen(req) as resp, out_path.open("wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"✓ Downloaded to {out_path}")
    return out_path


def extract_tagging_snps(archive: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.namelist()
        target = "tagging_snps.tsv"
        if target not in members:
            raise ArtifactError(f"{target} not found in {archive}")
        zf.extract(target, path=dest_dir)

    tsv_path = dest_dir / target
    print(f"✓ Extracted tagging SNPs table to {tsv_path}")
    return tsv_path


def format_size(num: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num)
    for unit in units:
        if val < 1024.0 or unit == units[-1]:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{num} B"


def calculate_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def unzip_file(gz_path: Path) -> Path:
    target = gz_path.with_suffix("")
    with gzip.open(gz_path, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def get_dataset_metadata() -> dict:
    url = f"{DATAVERSE_BASE}/api/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    return data["data"]


def download_file(file_id: int, filename: str, expected_md5: str | None) -> Path:
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SELECTION_DIR / filename

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

    url = f"{DATAVERSE_BASE}/api/access/datafile/{file_id}"
    print(f"Downloading {filename} from Dataverse...")
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


def ensure_selection_data() -> Path:
    if SELECTION_TSV_PATH.exists():
        print(f"✓ Selection TSV present at {SELECTION_TSV_PATH}")
        return SELECTION_TSV_PATH

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
        return SELECTION_TSV_PATH

    raise RuntimeError(f"Selection file {target} not found in dataset metadata")


def sanitize_region(region: str) -> str:
    return region.replace(":", "_").replace("-", "_").replace("/", "_")


@dataclass
class TaggingSNPResult:
    region: str
    inversion_region: str
    correlation: float
    chromosome_hg37: str
    position_hg37: int
    row: pd.Series

    @property
    def abs_correlation(self) -> float:
        return abs(self.correlation)


def load_tagging_snps(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    df["chrom_norm"] = df["chromosome"].astype(str).str.removeprefix("chr")
    return df


def select_best_tag(region: str, df: pd.DataFrame) -> TaggingSNPResult:
    chrom, start, end = parse_region(region)
    target_chrom = chrom.lstrip("chr")
    region_df = df[(df["chrom_norm"] == target_chrom) & (df["region_start"] == start) & (df["region_end"] == end)]

    if region_df.empty:
        raise ValueError(
            f"No tagging SNP rows found for region {region}. "
            "Check that chromosome and coordinates match tagging_snps.tsv."
        )

    region_df = region_df.assign(abs_corr=region_df["correlation"].abs())
    best_idx = region_df["abs_corr"].idxmax()
    best_row = region_df.loc[best_idx]

    return TaggingSNPResult(
        region=region,
        inversion_region=str(best_row["inversion_region"]),
        correlation=float(best_row["correlation"]),
        chromosome_hg37=str(best_row["chromosome_hg37"]),
        position_hg37=int(best_row["position_hg37"]),
        row=best_row,
    )


def load_selection_table() -> pd.DataFrame:
    path = ensure_selection_data()
    df = pd.read_csv(path, sep="\t", comment="#")
    df["CHROM_norm"] = df["CHROM"].astype(str).str.removeprefix("chr")
    return df


def find_selection_row(result: TaggingSNPResult, selection_df: pd.DataFrame) -> Optional[pd.Series]:
    chrom = str(result.chromosome_hg37).lstrip("chr")
    pos = result.position_hg37
    matches = selection_df[(selection_df["CHROM_norm"] == chrom) & (selection_df["POS"] == pos)]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_output(result: TaggingSNPResult, selection_row: Optional[pd.Series]) -> str:
    lines = [
        f"Region: {result.region}",
        f"Inversion region label: {result.inversion_region}",
        f"Best tagging SNP |r|: {result.abs_correlation:.4f} (correlation={result.correlation:+.4f})",
        f"Tagging SNP position (hg37): chr{result.chromosome_hg37}:{result.position_hg37}",
    ]

    if selection_row is None:
        lines.append("Selection summary: not found in selection statistics table")
        return "\n".join(lines)

    selection_columns = [
        "CHROM",
        "POS",
        "REF",
        "ALT",
        "ANC",
        "ID",
        "RSID",
        "AF",
        "S",
        "SE",
        "X",
        "P_X",
        "POSTERIOR",
        "FDR",
        "CHI2_BE",
        "FILTER",
    ]

    lines.append("Selection summary (hg19/GRCh37):")
    for col in selection_columns:
        val = selection_row.get(col, "")
        lines.append(f"  {col}: {val}")

    return "\n".join(lines)


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, help="Inversion region as chrN:start-end")
    parser.add_argument(
        "--repo",
        default=os.getenv(REPO_ENV),
        help="GitHub repository (owner/name). Defaults to GITHUB_REPOSITORY env.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=OUTPUT_DIR,
        help="Working/output directory for artifacts and reports.",
    )
    args = parser.parse_args(list(argv))

    if not args.repo:
        raise ArtifactError("Repository not specified; set --repo or GITHUB_REPOSITORY")

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    artifact = find_latest_artifact(args.repo, ARTIFACT_NAME, ARTIFACT_WORKFLOW_PATH)
    archive = download_artifact(artifact, workdir)
    tagging_tsv = extract_tagging_snps(archive, workdir)
    tag_df = load_tagging_snps(tagging_tsv)
    result = select_best_tag(args.region, tag_df)

    selection_df = load_selection_table()
    selection_row = find_selection_row(result, selection_df)
    output_text = render_output(result, selection_row)

    outfile = workdir / f"{sanitize_region(args.region)}_best_tagging_snp.txt"
    outfile.write_text(output_text)
    print(output_text)
    print(f"\n✓ Saved report to {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
