"""Locate the strongest tagging SNP for a given inversion region and report selection stats.

This script downloads the latest ``tagging-snps-report`` artifact produced by the
``run_find_tagging_snps.yml`` workflow, filters the table to the requested
``chrom:start-end`` region, and selects the variant with the highest absolute
correlation (``|r|``). It also downloads the selection summary statistics from
Dataverse (doi:10.7910/DVN/7RVV9N) and attempts to annotate the tagging SNP using
GRCh37/hg19 coordinates.

The script now reports the top three tagging SNPs by absolute correlation
(``|r|``) for the full inversion region, plus the best tagging SNP within each
of ten equally sized subregions. Outputs are written to
``outputs/<region>_best_tagging_snp.txt`` and also printed to stdout for GitHub
Actions log visibility.
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
ARTIFACT_WORKFLOW_PATH = "run_find_tagging_snps.yml"
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

    If ``workflow_path`` is provided, the latest *successful* run of that workflow
    is located first and its artifacts are inspected. This mirrors the pattern
    used in other workflows (e.g., ``upload_latest_manual_run_artifacts.yml``)
    and avoids relying on the global artifacts listing (which may omit
    ``workflow_run`` metadata).
    """

    if workflow_path:
        workflow_id = Path(workflow_path).name
        runs_url = (
            f"https://api.github.com/repos/{repo}/actions/workflows/"
            f"{workflow_id}/runs?status=success&per_page=1"
        )
        runs = github_json(runs_url).get("workflow_runs", [])
        if not runs:
            raise ArtifactError(
                f"No successful runs of workflow {workflow_path!r} found in {repo}"
            )

        run_id = runs[0].get("id")
        artifacts_url = (
            f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
            "?per_page=100"
        )
        artifacts = github_json(artifacts_url).get("artifacts", [])
        for artifact in artifacts:
            if artifact.get("name") == name and not artifact.get("expired"):
                return artifact

        raise ArtifactError(
            f"No non-expired artifact named {name!r} found in last successful run {run_id}"
        )

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
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    req = urllib.request.Request(url, headers=headers)
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
    try:
        meta = get_dataset_metadata()
    except Exception as exc:  # pragma: no cover - network failure surfaces as runtime error
        raise RuntimeError("Unable to retrieve Dataverse metadata for selection data") from exc

    files = meta.get("latestVersion", {}).get("files", [])
    target = SELECTION_GZ_NAME
    for fmeta in files:
        df = fmeta.get("dataFile", {})
        name = df.get("filename")
        if name != target:
            continue
        file_id = df["id"]
        checksum = df.get("checksum", {})
        expected = checksum.get("value") if checksum.get("type") == "MD5" else None

        try:
            gz_path = download_file(file_id, name, expected)
        except Exception as exc:  # pragma: no cover - network failure surfaces as runtime error
            raise RuntimeError("Unable to download selection data") from exc

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
    chromosome_hg38: str
    position_hg38: int
    row: pd.Series
    rank: Optional[int] = None
    context: Optional[str] = None

    @property
    def abs_correlation(self) -> float:
        return abs(self.correlation)


def load_tagging_snps(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    df["chrom_norm"] = df["chromosome"].astype(str).str.removeprefix("chr")
    return df


def _prepare_region_df(region: str, df: pd.DataFrame) -> pd.DataFrame:
    _chrom, start, end = parse_region(region)
    target_chrom = _chrom.lstrip("chr")
    region_df = df[
        (df["chrom_norm"] == target_chrom)
        & (df["region_start"] == start)
        & (df["region_end"] == end)
    ]

    if region_df.empty:
        raise ValueError(
            f"No tagging SNP rows found for region {region}. "
            "Check that chromosome and coordinates match tagging_snps.tsv."
        )

    return region_df.assign(abs_corr=region_df["correlation"].abs())


def select_top_tags(region: str, df: pd.DataFrame, *, top_n: int = 3) -> tuple[list[TaggingSNPResult], pd.DataFrame]:
    region_df = _prepare_region_df(region, df)
    sorted_df = region_df.sort_values("abs_corr", ascending=False)

    results: list[TaggingSNPResult] = []
    for rank, (_, row) in enumerate(sorted_df.head(top_n).iterrows(), start=1):
        results.append(
            TaggingSNPResult(
                region=region,
                inversion_region=str(row["inversion_region"]),
                correlation=float(row["correlation"]),
                chromosome_hg37=str(row["chromosome_hg37"]),
                position_hg37=int(row["position_hg37"]),
                chromosome_hg38=str(row["chromosome_hg38"]),
                position_hg38=int(row["position_hg38"]),
                row=row,
                rank=rank,
                context="Top overall",
            )
        )

    return results, sorted_df


def select_segment_bests(
    region: str, region_df: pd.DataFrame, *, segments: int = 10
) -> list[tuple[tuple[int, int], Optional[TaggingSNPResult]]]:
    _chrom, start, end = parse_region(region)
    if segments < 1:
        raise ValueError("segments must be at least 1")

    length = end - start + 1
    base_size = length // segments
    remainder = length % segments

    results: list[tuple[tuple[int, int], Optional[TaggingSNPResult]]] = []
    for idx in range(segments):
        seg_start = start + idx * base_size + min(idx, remainder)
        seg_len = base_size + (1 if idx < remainder else 0)
        seg_end = seg_start + seg_len - 1

        seg_df = region_df[(region_df["position"] >= seg_start) & (region_df["position"] <= seg_end)]
        if seg_df.empty:
            results.append(((seg_start, seg_end), None))
            continue

        best_idx = seg_df["abs_corr"].idxmax()
        best_row = seg_df.loc[best_idx]
        results.append(
            (
                (seg_start, seg_end),
                TaggingSNPResult(
                    region=region,
                    inversion_region=str(best_row["inversion_region"]),
                    correlation=float(best_row["correlation"]),
                    chromosome_hg37=str(best_row["chromosome_hg37"]),
                    position_hg37=int(best_row["position_hg37"]),
                    chromosome_hg38=str(best_row["chromosome_hg38"]),
                    position_hg38=int(best_row["position_hg38"]),
                    row=best_row,
                    context=f"Segment {idx + 1}",
                ),
            )
        )

    return results


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


def _format_float(val: float | int | str | None) -> str:
    if pd.isna(val):
        return "NA"
    try:
        return f"{float(val):.6f}"
    except Exception:
        return str(val)


def _format_tagging_result(
    header: str,
    result: TaggingSNPResult,
    selection_row: Optional[pd.Series],
    indent: str = "  ",
) -> list[str]:
    row = result.row
    inner = indent + "  "

    lines = [header]
    lines.append(f"{indent}Inversion region label: {result.inversion_region}")
    lines.append(
        f"{indent}Tagging SNP position (hg38): chr{row['chromosome']}:{int(row['position'])}"
    )
    lines.append(
        f"{indent}Tagging SNP position (hg37): chr{result.chromosome_hg37}:{result.position_hg37}"
    )
    lines.append(f"{indent}Correlation r: {result.correlation:+.6f}")
    lines.append(f"{indent}|r|: {result.abs_correlation:.6f}")
    lines.append(f"{indent}Direct group size: {_format_float(row['direct_group_size'])}")
    lines.append(f"{indent}Inverted group size: {_format_float(row['inverted_group_size'])}")
    lines.append(f"{indent}Allele frequency (direct group): {_format_float(row['allele_freq_direct'])}")
    lines.append(
        f"{indent}Allele frequency (inverted group): {_format_float(row['allele_freq_inverted'])}"
    )
    lines.append(
        f"{indent}Allele frequency difference: {_format_float(row['allele_freq_difference'])}"
    )

    dir_freqs = ", ".join(
        f"{base}={_format_float(row[f'{base}_dir_freq'])}" for base in ["A", "C", "G", "T"]
    )
    inv_freqs = ", ".join(
        f"{base}={_format_float(row[f'{base}_inv_freq'])}" for base in ["A", "C", "G", "T"]
    )
    lines.append(f"{indent}Allele frequencies by base (direct): {dir_freqs}")
    lines.append(f"{indent}Allele frequencies by base (inverted): {inv_freqs}")

    if selection_row is None:
        lines.append(f"{indent}Selection summary: not found in selection statistics table")
        return lines

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

    lines.append(f"{indent}Selection summary (hg19/GRCh37):")
    for col in selection_columns:
        val = selection_row.get(col, "")
        lines.append(f"{inner}{col}: {val}")

    return lines


def _best_allele(row: pd.Series, group: str) -> tuple[str, float | str]:
    best_base = "NA"
    best_freq: float | str = "NA"

    for base in ["A", "C", "G", "T"]:
        freq = row.get(f"{base}_{group}_freq", float("nan"))
        if pd.isna(freq):
            continue
        if best_base == "NA" or float(freq) > float(best_freq):
            best_base = base
            best_freq = float(freq)

    return best_base, best_freq


def _format_segment_summary(
    header: str,
    result: TaggingSNPResult,
    selection_row: Optional[pd.Series],
    indent: str = "  ",
) -> list[str]:
    row = result.row
    lines = [header]

    lines.append(f"{indent}Correlation r: {result.correlation:+.6f}")

    if selection_row is None:
        lines.append(f"{indent}S: not found in selection statistics table")
        lines.append(f"{indent}P_X: not found in selection statistics table")
    else:
        lines.append(f"{indent}S: {_format_float(selection_row.get('S'))}")
        lines.append(f"{indent}P_X: {_format_float(selection_row.get('P_X'))}")

    dir_allele, dir_freq = _best_allele(row, "dir")
    inv_allele, inv_freq = _best_allele(row, "inv")

    lines.append(
        f"{indent}Best allele (direct group): {dir_allele} ({_format_float(dir_freq)})"
    )
    lines.append(
        f"{indent}Best allele (inverted group): {inv_allele} ({_format_float(inv_freq)})"
    )

    return lines


def render_output(
    region: str,
    top_results: list[TaggingSNPResult],
    segment_results: list[tuple[tuple[int, int], Optional[TaggingSNPResult]]],
    selection_df: pd.DataFrame,
) -> str:
    lines: list[str] = [f"Region: {region}"]
    if top_results:
        lines.append(f"Inversion region label: {top_results[0].inversion_region}")

    lines.append("Top tagging SNPs by |r| (overall inversion region):")
    for result in top_results:
        title = f"  #{result.rank} ({result.context})"
        selection_row = find_selection_row(result, selection_df)
        lines.extend(_format_tagging_result(title, result, selection_row, indent="    "))

    lines.append("Best tagging SNP per decile (ten equally sized subregions):")
    for seg_idx, ((seg_start, seg_end), result) in enumerate(segment_results, start=1):
        segment_title = f"  Segment {seg_idx} ({seg_start}-{seg_end})"
        if result is None:
            lines.append(f"{segment_title}: no SNPs found in this segment")
            continue
        selection_row = find_selection_row(result, selection_df)
        lines.extend(
            _format_segment_summary(segment_title, result, selection_row, indent="    ")
        )

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

    predownloaded = next((path for path in [workdir / "tagging_snps.tsv", workdir / ARTIFACT_NAME / "tagging_snps.tsv"] if path.exists()), None)
    if predownloaded is None:
        predownloaded = next(workdir.glob("**/tagging_snps.tsv"), None)

    if predownloaded is not None:
        print(f"✓ Found pre-downloaded tagging SNPs at {predownloaded}")
        tagging_tsv = predownloaded
    else:
        artifact = find_latest_artifact(args.repo, ARTIFACT_NAME, ARTIFACT_WORKFLOW_PATH)
        archive = download_artifact(artifact, workdir)
        tagging_tsv = extract_tagging_snps(archive, workdir)
    tag_df = load_tagging_snps(tagging_tsv)
    top_results, region_df = select_top_tags(args.region, tag_df, top_n=3)
    segment_results = select_segment_bests(args.region, region_df, segments=10)

    selection_df = load_selection_table()
    output_text = render_output(args.region, top_results, segment_results, selection_df)

    outfile = workdir / f"{sanitize_region(args.region)}_best_tagging_snp.txt"
    outfile.write_text(output_text)
    print(output_text)
    print(f"\n✓ Saved report to {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
