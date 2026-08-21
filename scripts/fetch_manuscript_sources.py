#!/usr/bin/env python3
"""Fetch and verify immutable sources used by manuscript analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "reproducibility" / "manuscript_sources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported source-manifest schema in {path}")
    rows = payload.get("sources")
    if not isinstance(rows, list):
        raise ValueError(f"No sources list in {path}")
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate source IDs in {path}")
    return by_id


def verify(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )


def download(source: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        verify(destination, str(source["sha256"]))
        print(f"verified existing {source['id']}: {destination}")
        return

    last_error: Exception | None = None
    for attempt in range(1, 4):
        tmp_name = None
        try:
            request = urllib.request.Request(
                str(source["url"]),
                headers={"User-Agent": "ferromic-manuscript-reproduction"},
            )
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as temporary:
                tmp_name = temporary.name
                with urllib.request.urlopen(request, timeout=300) as response:
                    while block := response.read(1024 * 1024):
                        temporary.write(block)
            temporary_path = Path(tmp_name)
            verify(temporary_path, str(source["sha256"]))
            os.replace(temporary_path, destination)
            print(f"downloaded {source['id']}: {destination}")
            return
        except Exception as exc:
            last_error = exc
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"Failed to fetch {source['id']}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=REPO / "source_data")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    sources = load_manifest(args.manifest)
    if args.all and args.source:
        parser.error("Use either --all or one or more --source values")
    selected = list(sources) if args.all else args.source
    if not selected:
        parser.error("Select sources with --all or --source")
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        parser.error(f"Unknown source IDs: {', '.join(unknown)}")

    for source_id in selected:
        source = sources[source_id]
        kind = source.get("kind")
        if kind == "repository":
            path = REPO / str(source["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            verify(path, str(source["sha256"]))
            print(f"verified repository source {source_id}: {path.relative_to(REPO)}")
        elif kind == "url":
            download(source, args.output_dir / str(source["filename"]))
        else:
            raise ValueError(f"Unsupported source kind for {source_id}: {kind!r}")


if __name__ == "__main__":
    main()
