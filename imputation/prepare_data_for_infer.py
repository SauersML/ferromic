"""Extract PheWAS imputation predictors from chromosome-sharded PLINK1 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from bed_reader import open_bed
from numpy.lib.format import open_memmap

from imputation.targets import PHEWAS_TARGET_INVERSIONS


MODEL_MANIFEST_URL = (
    "https://github.com/SauersML/ferromic/releases/download/"
    "imputation-models-v1/models.manifest.txt"
)
MISSING_INT8 = np.int8(-127)
_TARGET_RE = re.compile(r"^chr([0-9XY]+)-\d+-INV-\d+$")


@dataclass(frozen=True)
class Predictor:
    chrom: str
    position: int
    effect_allele: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    predictors: tuple[Predictor, ...]


def _normalize_chrom(value: str) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return f"chr{text.upper()}"


def _fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ferromic-imputation/2"})
    with urlopen(request, timeout=300) as response:
        return response.read()


def _manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in _fetch_bytes(MODEL_MANIFEST_URL).decode("utf-8").splitlines():
        url = line.strip()
        if url:
            entries[Path(url).name] = url
    return entries


def _load_specs(targets: Sequence[str]) -> list[ModelSpec]:
    manifest = _manifest()
    specs: list[ModelSpec] = []
    for target in targets:
        filename = f"{target}.snps.json"
        url = manifest.get(filename)
        if url is None:
            raise RuntimeError(f"Model manifest has no predictor specification for {target}.")
        rows = json.loads(_fetch_bytes(url))
        predictors: list[Predictor] = []
        for index, row in enumerate(rows):
            try:
                locus = str(row["id"]).strip()
                chrom, position = locus.rsplit(":", 1)
                effect = str(row["effect_allele"]).strip().upper()
                predictors.append(
                    Predictor(_normalize_chrom(chrom), int(position), effect)
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Malformed predictor {index} for {target}: {row!r}."
                ) from error
        if not predictors:
            raise RuntimeError(f"Predictor specification for {target} is empty.")
        specs.append(ModelSpec(target, tuple(predictors)))
    return specs


def _target_chrom(target: str) -> str:
    match = _TARGET_RE.match(target)
    if match is None:
        raise ValueError(f"Malformed inversion id: {target!r}.")
    return _normalize_chrom(match.group(1))


def _read_sample_ids(fam_path: Path) -> list[str]:
    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        usecols=[1],
        dtype=str,
    )
    sample_ids = fam.iloc[:, 0].astype(str).tolist()
    if not sample_ids:
        raise RuntimeError(f"No samples in {fam_path}.")
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"Duplicate participant ids in {fam_path}.")
    return sample_ids


def _sample_digest(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_bim(prefix: Path) -> pd.DataFrame:
    bim = pd.read_csv(
        prefix.with_suffix(".bim"),
        sep=r"\s+",
        header=None,
        names=["chrom", "id", "cm", "position", "a1", "a2"],
        dtype={
            "chrom": str,
            "id": str,
            "position": np.int64,
            "a1": str,
            "a2": str,
        },
    )
    bim["chrom"] = bim["chrom"].map(_normalize_chrom)
    bim["a1"] = bim["a1"].str.upper()
    bim["a2"] = bim["a2"].str.upper()
    return bim


def _build_routes(
    bim: pd.DataFrame,
    specs: Sequence[ModelSpec],
    model_indices: Sequence[int],
) -> tuple[dict[int, list[tuple[int, int, bool]]], dict[str, int]]:
    by_position: dict[tuple[str, int], list[int]] = defaultdict(list)
    for source_index, row in enumerate(bim.itertuples(index=False)):
        by_position[(row.chrom, int(row.position))].append(source_index)

    routes: dict[int, list[tuple[int, int, bool]]] = defaultdict(list)
    missing: dict[str, int] = {}
    for model_index in model_indices:
        spec = specs[model_index]
        unresolved = 0
        for destination, predictor in enumerate(spec.predictors):
            candidates = by_position.get((predictor.chrom, predictor.position), [])
            matches: list[tuple[int, bool]] = []
            for source_index in candidates:
                row = bim.iloc[source_index]
                if row.a1 == predictor.effect_allele:
                    matches.append((source_index, False))
                if row.a2 == predictor.effect_allele:
                    matches.append((source_index, True))
            if len(matches) != 1:
                unresolved += 1
                continue
            source_index, flip = matches[0]
            routes[source_index].append((model_index, destination, flip))
        missing[spec.name] = unresolved
    return routes, missing


def _flip(column: np.ndarray) -> np.ndarray:
    flipped = column.copy()
    called = flipped >= 0
    flipped[called] = 2 - flipped[called]
    return flipped


def prepare(
    plink_dir: Path,
    output_dir: Path,
    targets: Sequence[str],
    threads: int,
    chunk_variants: int,
) -> None:
    specs = _load_specs(targets)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    chromosomes = sorted({_target_chrom(spec.name) for spec in specs})
    by_chrom = {
        chrom: [i for i, spec in enumerate(specs) if _target_chrom(spec.name) == chrom]
        for chrom in chromosomes
    }

    first_prefix = plink_dir / chromosomes[0]
    sample_ids = _read_sample_ids(first_prefix.with_suffix(".fam"))
    sample_hash = _sample_digest(sample_ids)
    n_samples = len(sample_ids)
    matrix_bytes = sum(n_samples * len(spec.predictors) for spec in specs)
    required_bytes = int(matrix_bytes * 1.10) + 512 * 1024**2
    free_bytes = shutil.disk_usage(output_dir.parent).free
    print(f"[prepare] {n_samples:,} samples; {len(specs)} models")
    print(f"[prepare] predictor matrices: {matrix_bytes / 1024**3:.2f} GiB")
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Need {required_bytes / 1024**3:.2f} GiB free in {output_dir.parent}; "
            f"only {free_bytes / 1024**3:.2f} GiB is available."
        )
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; move it aside before preparing a new cohort."
        )

    staging = output_dir.with_name(f".{output_dir.name}.incomplete.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    matrices = [
        open_memmap(
            staging / f"{spec.name}.genotypes.npy",
            mode="w+",
            dtype=np.int8,
            shape=(n_samples, len(spec.predictors)),
            fortran_order=True,
        )
        for spec in specs
    ]
    for matrix in matrices:
        matrix[:] = MISSING_INT8

    unresolved: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    try:
        for chrom in chromosomes:
            prefix = plink_dir / chrom
            for suffix in (".bed", ".bim", ".fam"):
                path = prefix.with_suffix(suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
            shard_ids = _read_sample_ids(prefix.with_suffix(".fam"))
            if _sample_digest(shard_ids) != sample_hash:
                raise RuntimeError(f"Sample order differs in {prefix.with_suffix('.fam')}.")

            bim = _read_bim(prefix)
            routes, shard_missing = _build_routes(bim, specs, by_chrom[chrom])
            unresolved.update(shard_missing)
            selected = sorted(routes)
            source_counts[chrom] = len(selected)
            print(
                f"[prepare] {chrom}: reading {len(selected):,} required variants "
                f"from {len(bim):,} available"
            )
            bed = open_bed(str(prefix.with_suffix(".bed")))
            for start in range(0, len(selected), chunk_variants):
                source_indices = selected[start : start + chunk_variants]
                block = bed.read(
                    index=np.s_[:, source_indices],
                    dtype="int8",
                    order="F",
                    num_threads=threads,
                )
                for local_index, source_index in enumerate(source_indices):
                    column = block[:, local_index]
                    flipped: np.ndarray | None = None
                    for model_index, destination, needs_flip in routes[source_index]:
                        if needs_flip:
                            if flipped is None:
                                flipped = _flip(column)
                            matrices[model_index][:, destination] = flipped
                        else:
                            matrices[model_index][:, destination] = column

        for matrix in matrices:
            matrix.flush()
        pd.DataFrame({"sample_id": sample_ids}).to_csv(
            staging / "sample_ids.tsv",
            sep="\t",
            index=False,
        )
        report = {
            "sample_count": n_samples,
            "sample_sha256": sample_hash,
            "targets": list(targets),
            "predictor_columns": {
                spec.name: len(spec.predictors) for spec in specs
            },
            "unresolved_predictors": unresolved,
            "source_variants_read": source_counts,
            "plink_directory": str(plink_dir.resolve()),
        }
        with open(staging / "preparation.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        staging.rename(output_dir)
    except Exception:
        print(f"[prepare] incomplete files retained at {staging}")
        raise

    for spec in specs:
        total = len(spec.predictors)
        missing = unresolved[spec.name]
        print(f"[prepare] {spec.name}: {total - missing:,}/{total:,} predictors matched")
    print(f"[prepare] wrote {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plink-dir",
        required=True,
        type=Path,
        help="Directory containing chromosome-sharded PLINK1 files.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Inversion id to prepare; repeat for multiple models.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-variants", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    targets = tuple(args.targets or PHEWAS_TARGET_INVERSIONS)
    if len(targets) != len(set(targets)):
        raise SystemExit("Target inversion ids must be unique.")
    if args.threads < 1 or args.chunk_variants < 1:
        raise SystemExit("--threads and --chunk-variants must be positive.")
    prepare(
        args.plink_dir.resolve(),
        args.output_dir.resolve(),
        targets,
        args.threads,
        args.chunk_variants,
    )


if __name__ == "__main__":
    main()
