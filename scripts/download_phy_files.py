#!/usr/bin/env python3
"""Extract CDS PHYLIP files from phy_files.zip.

This script extracts all .phy files from the phy_files.zip archive into the
current directory for use by the CDS conservation analysis pipeline.
"""

import sys
import zipfile
from pathlib import Path


PHY_ZIP_URL = "https://sharedspace.s3.msi.umn.edu/public_internet/phy_files.zip"
PHY_ZIP_LOCAL = Path("phy_files.zip")


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


def extract_phy_files(zip_path: Path) -> int:
    """Extract all .phy files from the zip archive."""
    
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found.", file=sys.stderr)
        return 1
    
    print(f"\nExtracting .phy files from {zip_path} ...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get list of .phy files in the archive
            phy_files = [name for name in zf.namelist() if name.endswith('.phy')]
            
            if not phy_files:
                print("ERROR: No .phy files found in the archive.", file=sys.stderr)
                return 1
            
            print(f"Found {len(phy_files)} .phy files in archive.")
            
            # Check which already exist
            existing = 0
            to_extract = []
            for name in phy_files:
                # Get just the filename (in case zip has directory structure)
                filename = Path(name).name
                if Path(filename).exists():
                    existing += 1
                else:
                    to_extract.append(name)
            
            print(f"  Already present: {existing}")
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
    
    # Download zip if not present
    if not PHY_ZIP_LOCAL.exists():
        print(f"\n{PHY_ZIP_LOCAL} not found locally.")
        if not download_file(PHY_ZIP_URL, PHY_ZIP_LOCAL):
            print(f"\nERROR: Failed to download {PHY_ZIP_URL}", file=sys.stderr)
            print("Please download phy_files.zip manually and place it in the current directory.", file=sys.stderr)
            return 1
    else:
        print(f"\n✅ Found {PHY_ZIP_LOCAL}")
    
    # Extract files
    return extract_phy_files(PHY_ZIP_LOCAL)


if __name__ == "__main__":
    sys.exit(main())
