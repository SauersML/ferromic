#!/usr/bin/env python3
"""Extract the PHYLIP alignments required for Ferromic figure replication.

Instead of materialising every alignment contained in ``phy_files.zip``, this
script analyses the local metadata tables and extracts only the ``.phy`` files
that are necessary for the downstream statistics pipelines.  Any previously
unpacked ``.phy`` files that are no longer required are deleted to keep disk
usage minimal.
"""

import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple
import csv


PHY_ZIP_URL = "https://sharedspace.s3.msi.umn.edu/public_internet/all_phy/phy_files.zip"
PHY_ZIP_LOCAL = Path("phy_files.zip")

# Metadata artefacts that list the PHYLIP filenames needed for the published
# analyses.  Each tuple records ``(filename, column_with_phy_name)``.
METADATA_REQUIREMENTS: Sequence[Tuple[str, str]] = (
    ("cds_identical_proportions.tsv", "filename"),
    ("region_identical_proportions.tsv", "filename"),
    ("phy_metadata.tsv", "phy_filename"),
)

# Candidate locations to search for the metadata artefacts.  ``analysis_downloads``
# is used by ``scripts/replicate_figures.py`` while ``data/`` contains files that
# ship with the public repository.
METADATA_SEARCH_DIRS: Sequence[Path] = (
    Path("."),
    Path("analysis_downloads"),
    Path("analysis_downloads/public_internet"),
    Path("data"),
)


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination path."""
    import urllib.request
    import urllib.error
    
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            if response.status == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 8192
                with dest.open('wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"  Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
                print()  # New line after progress
                return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  ⚠️  Failed to download {url}: {e}", file=sys.stderr)
    return False


def find_metadata_file(name: str) -> Optional[Path]:
    """Locate ``name`` within the known metadata directories."""

    for base in METADATA_SEARCH_DIRS:
        if not base.exists():
            continue
        candidate = base / name
        if candidate.exists():
            return candidate
        # Fall back to a recursive search to support nested mirrors of the
        # download artefacts (e.g. ``analysis_downloads/public_internet``).
        for path in base.rglob(name):
            if path.is_file():
                return path
    return None


def iter_required_phy_names() -> Set[str]:
    """Return the set of required ``.phy`` basenames from metadata files."""

    required: Set[str] = set()

    for metadata_name, column in METADATA_REQUIREMENTS:
        path = find_metadata_file(metadata_name)
        if path is None:
            print(f"  ⚠️  Metadata file '{metadata_name}' not found; skipping.")
            continue

        try:
            with path.open("r", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if column not in (reader.fieldnames or []):
                    print(
                        f"  ⚠️  Column '{column}' missing from {path}. "
                        "Unable to collect .phy requirements from this file."
                    )
                    continue
                for row in reader:
                    value = (row.get(column) or "").strip()
                    if not value:
                        continue
                    name = Path(value).name
                    required.add(name)
        except OSError as exc:
            print(f"  ⚠️  Failed to read metadata file {path}: {exc}")

    return required


def prune_unneeded_phy(required: Set[str]) -> int:
    """Delete previously extracted ``.phy`` files that are not required."""

    removed = 0
    for path in Path(".").glob("*.phy"):
        if path.name in required:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            print(f"  ⚠️  Unable to delete {path}: {exc}")
    if removed:
        print(f"  Removed {removed} obsolete .phy files from the working directory.")
    return removed


def extract_phy_files(zip_path: Path, required: Set[str]) -> int:
    """Extract only the required ``.phy`` files from ``zip_path``."""

    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found.", file=sys.stderr)
        return 1

    print(f"\nExtracting required .phy files from {zip_path} ...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            archive_members = [name for name in zf.namelist() if name.endswith('.phy')]
            if not archive_members:
                print("ERROR: No .phy files found in the archive.", file=sys.stderr)
                return 1

            available: Dict[str, str] = {Path(name).name: name for name in archive_members}

            missing_from_archive = sorted(
                name for name in required if name not in available and not Path(name).exists()
            )
            if missing_from_archive:
                print("  ⚠️  The archive is missing some required .phy files:")
                for name in missing_from_archive[:10]:
                    print(f"      - {name}")
                if len(missing_from_archive) > 10:
                    print(f"      ... and {len(missing_from_archive) - 10} more")

            already_present = sum(1 for name in required if Path(name).exists())
            to_extract: Sequence[str] = [
                available[name]
                for name in sorted(required)
                if name in available and not Path(name).exists()
            ]

            print(f"  Required .phy files (unique): {len(required)}")
            print(f"  Already present: {already_present}")
            print(f"  To extract: {len(to_extract)}")

            if not to_extract:
                print("\n✅ All .phy files are already present.")
                return 0

            # Extract files
            extracted = 0
            for i, name in enumerate(to_extract, 1):
                filename = Path(name).name

                # Extract to current directory
                with zf.open(name) as source:
                    Path(filename).write_bytes(source.read())

                extracted += 1
                if i % 50 == 0 or i == len(to_extract):
                    print(f"  Progress: {i}/{len(to_extract)} files extracted...")

            print(f"\n✅ Successfully extracted {extracted} .phy files.")
            return 0

    except zipfile.BadZipFile:
        print(f"ERROR: {zip_path} is not a valid zip file.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Failed to extract files: {e}", file=sys.stderr)
        return 1


def main():
    """Download and extract .phy files."""

    print("=" * 70)
    print("CDS PHYLIP Files Extraction")
    print("=" * 70)

    print("\nDetermining required .phy files from metadata ...")
    required = iter_required_phy_names()
    if not required:
        print(
            "ERROR: No metadata sources were available to determine the required .phy files.",
            file=sys.stderr,
        )
        print(
            "Please ensure cds_identical_proportions.tsv and region_identical_proportions.tsv "
            "are present (see scripts/replicate_figures.py downloads).",
            file=sys.stderr,
        )
        return 1

    print(f"  Identified {len(required)} required .phy files from metadata.")

    # Download zip if not present
    if not PHY_ZIP_LOCAL.exists():
        print(f"\n{PHY_ZIP_LOCAL} not found locally.")
        if not download_file(PHY_ZIP_URL, PHY_ZIP_LOCAL):
            print(f"\nERROR: Failed to download {PHY_ZIP_URL}", file=sys.stderr)
            print("Please download phy_files.zip manually and place it in the current directory.", file=sys.stderr)
            return 1
    else:
        print(f"\n✅ Found {PHY_ZIP_LOCAL}")

    prune_unneeded_phy(required)

    # Extract files
    return extract_phy_files(PHY_ZIP_LOCAL, required)


if __name__ == "__main__":
    sys.exit(main())
