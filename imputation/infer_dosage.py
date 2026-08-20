"""Infer the seven AoU PheWAS inversion dosages from prepared SNP matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from imputation import pls_patch
from imputation.targets import PHEWAS_TARGET_INVERSIONS


# Published models were serialized when pls_patch.py was imported as a top-level module.
# Register the repository module under that exact name before joblib unpickles them.
sys.modules["pls_patch"] = pls_patch

MODEL_MANIFEST_URL = (
    "https://github.com/SauersML/ferromic/releases/download/"
    "imputation-models-v1/models.manifest.txt"
)
MISSING_INT8 = np.int8(-127)
MIN_PREDICTOR_CALL_RATE = 0.01
DOSAGE_MIN = 0.0
DOSAGE_MAX = 2.0
ANCESTRY_CODES = {
    "eur": 0,
    "afr": 1,
    "amr": 2,
    "eas": 3,
    "sas": 4,
    "mid": 5,
    "oth": 6,
}
UNKNOWN_ANCESTRY = len(ANCESTRY_CODES)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_models(targets: Sequence[str], model_dir: Path) -> dict[str, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    paths: dict[str, Path] = {}
    for target in targets:
        filename = f"{target}.model.joblib"
        url = manifest.get(filename)
        if url is None:
            raise RuntimeError(f"Model manifest has no model for {target}.")
        path = model_dir / filename
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            with open(temporary, "wb") as handle:
                handle.write(_fetch_bytes(url))
            temporary.replace(path)
        paths[target] = path
    return paths


def _load_sample_ids(path: Path) -> list[str]:
    table = pd.read_csv(path, sep="\t", dtype=str)
    if list(table.columns) != ["sample_id"]:
        raise ValueError(f"{path} must contain exactly one column named sample_id.")
    sample_ids = table["sample_id"].astype(str).tolist()
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} has no samples or contains duplicate sample ids.")
    return sample_ids


def _load_ancestry(path: Path, sample_ids: Sequence[str]) -> np.ndarray:
    ancestry = pd.read_csv(
        path,
        sep="\t",
        usecols=["research_id", "ancestry_pred"],
        dtype=str,
    )
    if ancestry["research_id"].duplicated().any():
        raise ValueError(f"Duplicate research_id values in {path}.")
    labels = ancestry.set_index("research_id")["ancestry_pred"]
    codes = labels.str.lower().str.strip().map(ANCESTRY_CODES)
    aligned = codes.reindex(pd.Index(sample_ids)).fillna(UNKNOWN_ANCESTRY)
    missing = int((aligned == UNKNOWN_ANCESTRY).sum())
    if missing:
        print(f"[infer] {missing:,} samples lack a recognized ancestry; using global means")
    return aligned.to_numpy(dtype=np.int8)


def _matrix_statistics(
    matrix: np.ndarray,
    codes: np.ndarray,
    row_chunk: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    n_snps = matrix.shape[1]
    sums = np.zeros((UNKNOWN_ANCESTRY + 1, n_snps), dtype=np.float64)
    counts = np.zeros((UNKNOWN_ANCESTRY + 1, n_snps), dtype=np.int64)
    for start in range(0, matrix.shape[0], row_chunk):
        end = min(start + row_chunk, matrix.shape[0])
        block = np.asarray(matrix[start:end])
        called = block != MISSING_INT8
        safe = np.where(called, block, 0)
        sums[UNKNOWN_ANCESTRY] += safe.sum(axis=0, dtype=np.float64)
        counts[UNKNOWN_ANCESTRY] += called.sum(axis=0, dtype=np.int64)
        block_codes = codes[start:end]
        for code in ANCESTRY_CODES.values():
            rows = block_codes == code
            if rows.any():
                sums[code] += safe[rows].sum(axis=0, dtype=np.float64)
                counts[code] += called[rows].sum(axis=0, dtype=np.int64)

    means = np.empty_like(sums, dtype=np.float32)
    global_mean = np.divide(
        sums[UNKNOWN_ANCESTRY],
        counts[UNKNOWN_ANCESTRY],
        out=np.zeros(n_snps, dtype=np.float64),
        where=counts[UNKNOWN_ANCESTRY] > 0,
    ).astype(np.float32)
    means[UNKNOWN_ANCESTRY] = global_mean
    for code in ANCESTRY_CODES.values():
        means[code] = np.divide(
            sums[code],
            counts[code],
            out=global_mean.astype(np.float64, copy=True),
            where=counts[code] > 0,
        )
    call_rates = counts[UNKNOWN_ANCESTRY] / float(matrix.shape[0])
    return call_rates, means


def _training_feature_means(model: object, n_features: int) -> np.ndarray:
    """Return the PLS centering means used when the model was fitted."""
    try:
        estimator = model.named_steps["pls"]
    except (AttributeError, KeyError, TypeError) as error:
        raise RuntimeError(
            "The inversion model must be a fitted pipeline with a 'pls' step."
        ) from error
    means = np.asarray(getattr(estimator, "_x_mean", None), dtype=np.float32)
    if means.shape != (n_features,) or not np.isfinite(means).all():
        raise RuntimeError(
            "The inversion model has missing or malformed PLS training means."
        )
    return means


def _predict(
    target: str,
    matrix_path: Path,
    model_path: Path,
    ancestry_codes: np.ndarray,
    batch_size: int,
    threads: int,
) -> tuple[np.ndarray, dict[str, object]]:
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[0] != len(ancestry_codes):
        raise ValueError(f"Unexpected matrix shape for {target}: {matrix.shape}.")
    model = joblib.load(model_path)
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is None or int(expected_features) != matrix.shape[1]:
        raise RuntimeError(
            f"{target} model expects {expected_features} predictors but the matrix has "
            f"{matrix.shape[1]}."
        )
    training_means = _training_feature_means(model, matrix.shape[1])
    call_rates, means = _matrix_statistics(matrix, ancestry_codes)
    covered = call_rates >= MIN_PREDICTOR_CALL_RATE
    covered_count = int(covered.sum())
    if covered_count == 0:
        raise RuntimeError(
            f"{target} has no predictors with at least "
            f"{MIN_PREDICTOR_CALL_RATE:.0%} call rate."
        )
    absent = call_rates == 0
    means[:, absent] = training_means[absent]
    predictions = np.empty(matrix.shape[0], dtype=np.float32)
    clamped_count = 0
    with threadpool_limits(limits=threads):
        for start in range(0, matrix.shape[0], batch_size):
            end = min(start + batch_size, matrix.shape[0])
            batch = np.asarray(matrix[start:end], dtype=np.float32).copy()
            missing = batch == MISSING_INT8
            if missing.any():
                fill = means[ancestry_codes[start:end]]
                batch[missing] = fill[missing]
            raw = np.asarray(model.predict(batch), dtype=np.float32).reshape(-1)
            if raw.shape[0] != end - start or not np.isfinite(raw).all():
                raise RuntimeError(f"{target} produced malformed or non-finite predictions.")
            clamped = np.clip(raw, DOSAGE_MIN, DOSAGE_MAX)
            clamped_count += int(np.count_nonzero(raw != clamped))
            predictions[start:end] = clamped
    report = {
        "predictor_count": int(matrix.shape[1]),
        "covered_predictors": covered_count,
        "training_mean_predictors": int(absent.sum()),
        "mean_call_rate": float(call_rates.mean()),
        "clamped_predictions": clamped_count,
        "model_sha256": _sha256(model_path),
    }
    return predictions, report


def infer(
    genotype_dir: Path,
    ancestry_path: Path,
    model_dir: Path,
    output_path: Path,
    targets: Sequence[str],
    batch_size: int,
    threads: int,
) -> None:
    sample_ids = _load_sample_ids(genotype_dir / "sample_ids.tsv")
    ancestry_codes = _load_ancestry(ancestry_path, sample_ids)
    model_paths = _ensure_models(targets, model_dir)
    output = pd.DataFrame(index=pd.Index(sample_ids, name="SampleID"))
    reports: dict[str, object] = {}
    for target in targets:
        matrix_path = genotype_dir / f"{target}.genotypes.npy"
        if not matrix_path.is_file():
            raise FileNotFoundError(matrix_path)
        print(f"[infer] {target}")
        predictions, report = _predict(
            target,
            matrix_path,
            model_paths[target],
            ancestry_codes,
            batch_size,
            threads,
        )
        output[target] = predictions
        reports[target] = report
        print(
            f"[infer] coverage={report['covered_predictors']:,}/"
            f"{report['predictor_count']:,}; "
            f"training-mean={report['training_mean_predictors']:,}; "
            f"clamped={report['clamped_predictions']:,}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    output.to_csv(temporary, sep="\t", float_format="%.6f")
    temporary.replace(output_path)
    provenance = {
        "sample_count": len(sample_ids),
        "targets": list(targets),
        "ancestry_path": str(ancestry_path),
        "genotype_directory": str(genotype_dir),
        "models": reports,
    }
    provenance_path = output_path.with_suffix(output_path.suffix + ".json")
    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
    print(f"[infer] wrote {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genotype-dir", required=True, type=Path)
    parser.add_argument("--ancestry", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Inversion id to infer; repeat for multiple models.",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    targets = tuple(args.targets or PHEWAS_TARGET_INVERSIONS)
    if len(targets) != len(set(targets)):
        raise SystemExit("Target inversion ids must be unique.")
    if args.batch_size < 1 or args.threads < 1:
        raise SystemExit("--batch-size and --threads must be positive.")
    infer(
        args.genotype_dir.resolve(),
        args.ancestry.resolve(),
        args.model_dir.resolve(),
        args.output.resolve(),
        targets,
        args.batch_size,
        args.threads,
    )


if __name__ == "__main__":
    main()
