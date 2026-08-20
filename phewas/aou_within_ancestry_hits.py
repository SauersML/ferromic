"""Run the AoU v8 within-ancestry-PC analysis for the 37 existing PheWAS hits."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from imputation.targets import PHEWAS_TARGET_INVERSIONS


THREADS = 16
MEMORY_MB = 64_000
MIN_CPUS = 16
MIN_MEMORY_BYTES = 100 * 1024**3
MIN_FREE_BYTES = 30 * 1024**3
MARKERS = 100_000
COMPONENTS = 16
POPULATIONS = ("eur", "afr", "eas", "amr", "sas", "mid")


@dataclass(frozen=True)
class Paths:
    repo: Path = Path("/home/jupyter/repos/ferromic")
    local: Path = Path("/home/jupyter/aou-phewas")
    v8: Path = Path("/home/jupyter/workspace/vwb-aou-datasets-controlled/v8")

    @property
    def acaf(self) -> Path:
        return self.v8 / "wgs/short_read/snpindel/acaf_threshold/plink_bed"

    @property
    def auxiliary(self) -> Path:
        return self.v8 / "wgs/short_read/snpindel/aux"

    @property
    def ancestry(self) -> Path:
        return self.auxiliary / "ancestry/echo_v4_r2.ancestry_preds.tsv"

    @property
    def related(self) -> Path:
        return self.auxiliary / "relatedness/samples_relatedness_flagged_samples.tsv"

    @property
    def arrays(self) -> Path:
        return self.v8 / "microarray/plink/arrays"

    @property
    def python(self) -> Path:
        return self.local / "venv/bin/python"

    @property
    def genotype_matrices(self) -> Path:
        return self.local / "genotype_matrices"

    @property
    def models(self) -> Path:
        return self.local / "models"

    @property
    def dosages(self) -> Path:
        return self.repo / "imputed_inversion_dosages.tsv"

    @property
    def sites(self) -> Path:
        return self.local / "sites/include_sites.tsv"

    @property
    def pca_genotypes(self) -> Path:
        return self.local / "pca/arrays_pca"

    @property
    def pca_output(self) -> Path:
        return self.repo / "within_ancestry_pcs"

    @property
    def results(self) -> Path:
        return self.local / "results/within_ancestry_hits"

    @property
    def logs(self) -> Path:
        return self.local / "logs/within_ancestry_hits"

    @property
    def phenotypes(self) -> Path:
        return self.repo / "phewas/data/significant_phenotypes.txt"


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    log_path: Path | None = None,
) -> None:
    rendered = [str(part) for part in command]
    print("[run] " + " ".join(rendered), flush=True)
    if log_path is None:
        subprocess.run(rendered, cwd=cwd, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            rendered,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Subprocess output pipe was not created.")
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, rendered)


def _memory_bytes() -> int:
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal is absent from /proc/meminfo.")


def _cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    count = os.cpu_count()
    if count is None:
        raise RuntimeError("The available CPU count cannot be determined.")
    return count


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are unavailable: " + ", ".join(missing))


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"))


def _data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def validate_shortlist(path: Path) -> None:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(names) != 37 or len(names) != len(set(names)):
        raise ValueError(f"Expected 37 unique hit phenotypes in {path}; found {len(names)}.")


def validate_predictor_checkpoint(directory: Path) -> int:
    report_path = directory / "preparation.json"
    sample_path = directory / "sample_ids.tsv"
    _require_files([report_path, sample_path])
    report = _read_json(report_path)
    if tuple(report.get("targets", ())) != PHEWAS_TARGET_INVERSIONS:
        raise ValueError("Prepared predictor targets do not match the PheWAS targets.")
    _require_files(
        [directory / f"{target}.genotypes.npy" for target in PHEWAS_TARGET_INVERSIONS]
    )
    return int(report["sample_count"])


def validate_dosages(path: Path) -> int:
    sidecar = path.with_suffix(path.suffix + ".json")
    _require_files([path, sidecar])
    missing = [target for target in PHEWAS_TARGET_INVERSIONS if target not in _header(path)]
    if missing:
        raise ValueError("Dosage TSV is missing targets: " + ", ".join(missing))
    provenance = _read_json(sidecar)
    if tuple(provenance.get("targets", ())) != PHEWAS_TARGET_INVERSIONS:
        raise ValueError("Dosage provenance targets do not match the PheWAS targets.")
    sample_count = int(provenance["sample_count"])
    if _data_rows(path) != sample_count:
        raise ValueError("Dosage TSV row count disagrees with its provenance.")
    return sample_count


def validate_staged_genotypes(prefix: Path) -> None:
    paths = [
        Path(f"{prefix}.{suffix}")
        for suffix in ("bed", "bim", "fam", "stage.json")
    ]
    _require_files(paths)
    report = _read_json(Path(f"{prefix}.stage.json"))
    if report.get("staged_markers") != MARKERS:
        raise ValueError(f"The staged PCA dataset does not contain {MARKERS:,} markers.")
    if _line_count(Path(f"{prefix}.bim")) != MARKERS:
        raise ValueError(f"The staged BIM does not contain {MARKERS:,} markers.")


def validate_pcs(table_path: Path, sidecar_path: Path, population: str) -> None:
    _require_files([table_path, sidecar_path])
    expected = ["person_id", *[f"WPC{i}" for i in range(1, COMPONENTS + 1)]]
    if _header(table_path) != expected:
        raise ValueError(f"Malformed PCA table for {population}.")
    if _data_rows(table_path) < COMPONENTS + 1:
        raise ValueError(f"Too few PCA rows for {population}.")
    sidecar = _read_json(sidecar_path)
    if sidecar.get("population") != population:
        raise ValueError(f"PCA population mismatch for {population}.")
    if sidecar.get("components") != COMPONENTS:
        raise ValueError(f"PCA component-count mismatch for {population}.")
    if sidecar.get("strict_convergence") is not True:
        raise ValueError(f"PCA convergence was not strict for {population}.")
    summary = sidecar.get("hwe_summary")
    converged = isinstance(summary, dict) and str(summary.get("converged", "")).lower() == "true"
    if not converged:
        raise ValueError(f"Gnomon did not certify convergence for {population}.")


def validate_phewas_result(path: Path) -> int:
    _require_files([path])
    required = {"Phenotype", "Inversion", "Beta", "OR", "P_Value"}
    missing = sorted(required - set(_header(path)))
    rows = _data_rows(path)
    if missing or rows == 0:
        raise ValueError(f"Malformed PheWAS result {path}; missing={missing}, rows={rows}.")
    return rows


def preflight(paths: Paths) -> None:
    if paths.repo.resolve() != Path.cwd().resolve():
        raise RuntimeError(f"Run from {paths.repo}.")
    for name in ("GOOGLE_PROJECT", "WORKSPACE_CDR"):
        if not os.environ.get(name):
            raise RuntimeError(f"{name} is not set by the Workbench.")
    if _cpu_count() < MIN_CPUS:
        raise RuntimeError(f"At least {MIN_CPUS} CPUs are required.")
    if _memory_bytes() < MIN_MEMORY_BYTES:
        raise RuntimeError("At least 100 GiB RAM is required.")
    if shutil.disk_usage(paths.local).free < MIN_FREE_BYTES:
        raise RuntimeError("At least 30 GiB free local disk is required.")
    if shutil.which("plink2") is None or shutil.which("gnomon") is None:
        raise RuntimeError("Both plink2 and gnomon must be installed.")

    inputs = [paths.ancestry, paths.related, paths.phenotypes]
    inputs.extend(
        Path(f"{paths.arrays}.{suffix}") for suffix in ("bed", "bim", "fam")
    )
    for chromosome in ("chr4", "chr6", "chr8", "chr10", "chr12", "chr17"):
        inputs.extend(
            paths.acaf / f"{chromosome}.{suffix}"
            for suffix in ("bed", "bim", "fam")
        )
    _require_files(inputs)
    validate_shortlist(paths.phenotypes)

    print(f"[preflight] CPUs={_cpu_count()} threads={THREADS}")
    print(f"[preflight] RAM={_memory_bytes() / 1024**3:.1f} GiB")
    free_gib = shutil.disk_usage(paths.local).free / 1024**3
    print(f"[preflight] free disk={free_gib:.1f} GiB")
    _run(["gnomon", "version"], cwd=paths.repo)


def prepare_dosages(paths: Paths) -> None:
    if not paths.genotype_matrices.exists():
        _run(
            [
                paths.python,
                "-m",
                "imputation.prepare_data_for_infer",
                "--plink-dir",
                paths.acaf,
                "--output-dir",
                paths.genotype_matrices,
                "--threads",
                str(THREADS),
            ],
            cwd=paths.repo,
        )
    samples = validate_predictor_checkpoint(paths.genotype_matrices)
    print(f"[dosage] predictor checkpoint: {samples:,} samples")

    if not paths.dosages.exists():
        _run(
            [
                paths.python,
                "-m",
                "imputation.infer_dosage",
                "--genotype-dir",
                paths.genotype_matrices,
                "--ancestry",
                paths.ancestry,
                "--model-dir",
                paths.models,
                "--output",
                paths.dosages,
                "--batch-size",
                "20000",
                "--threads",
                str(THREADS),
            ],
            cwd=paths.repo,
        )
    samples = validate_dosages(paths.dosages)
    print(f"[dosage] dosage checkpoint: {samples:,} samples")


def stage_pca_genotypes(paths: Paths) -> None:
    paths.sites.parent.mkdir(parents=True, exist_ok=True)
    paths.pca_genotypes.parent.mkdir(parents=True, exist_ok=True)
    paths.pca_output.mkdir(parents=True, exist_ok=True)
    _run(
        [
            paths.python,
            "-m",
            "phewas.extra.within_ancestry_pca",
            "sites",
            "--bim",
            Path(f"{paths.arrays}.bim"),
            "--out",
            paths.sites,
        ],
        cwd=paths.repo,
    )

    staged = [
        Path(f"{paths.pca_genotypes}.{suffix}")
        for suffix in ("bed", "bim", "fam", "stage.json")
    ]
    present = sum(path.is_file() for path in staged)
    if present == 0:
        _run(
            [
                paths.python,
                "-m",
                "phewas.extra.within_ancestry_pca",
                "stage",
                "--genotypes",
                paths.arrays,
                "--sites",
                paths.sites,
                "--out",
                paths.pca_genotypes,
                "--markers",
                str(MARKERS),
                "--threads",
                str(THREADS),
                "--memory-mb",
                str(MEMORY_MB),
            ],
            cwd=paths.repo,
        )
    elif present != len(staged):
        raise RuntimeError(f"Incomplete PCA staging checkpoint at {paths.pca_genotypes}.")
    validate_staged_genotypes(paths.pca_genotypes)
    print("[pca] staged genotype checkpoint validated")


def run_population(paths: Paths, population: str) -> None:
    table = paths.pca_output / f"within_ancestry_pcs_{population}.tsv"
    sidecar = paths.pca_output / f"within_ancestry_pcs_{population}.json"
    result = paths.results / f"phewas_{population}_within_ancestry_pcs.tsv"

    present = int(table.is_file()) + int(sidecar.is_file())
    if present == 0:
        _run(
            [
                paths.python,
                "-m",
                "phewas.extra.within_ancestry_pca",
                "fit",
                "--genotypes",
                paths.pca_genotypes,
                "--sites",
                paths.sites,
                "--ancestry",
                paths.ancestry,
                "--cohort",
                paths.dosages,
                "--related",
                paths.related,
                "--group",
                population,
                "--out-dir",
                paths.pca_output,
                "--gnomon",
                "gnomon",
                "--components",
                str(COMPONENTS),
                "--threads",
                str(THREADS),
                "--dosages",
                paths.dosages,
            ],
            cwd=paths.repo,
            log_path=paths.logs / f"pca_{population}.log",
        )
    elif present != 2:
        raise RuntimeError(f"Incomplete PCA checkpoint for {population}.")
    validate_pcs(table, sidecar, population)
    print(f"[pca:{population}] checkpoint validated")

    if not result.exists():
        _run(
            [
                paths.python,
                "-m",
                "phewas.cli",
                "--pop-label",
                population,
                "--pc-source",
                "within-ancestry",
                "--pheno-file",
                paths.phenotypes,
                "--min-cases-controls",
                "100",
                "--output",
                result,
            ],
            cwd=paths.repo,
            log_path=paths.logs / f"phewas_{population}.log",
        )
    rows = validate_phewas_result(result)
    print(f"[phewas:{population}] checkpoint validated: {rows:,} rows")


def main() -> None:
    paths = Paths()
    paths.local.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    preflight(paths)
    prepare_dosages(paths)
    stage_pca_genotypes(paths)
    for population in POPULATIONS:
        run_population(paths, population)
    print(f"[done] results: {paths.results}")


if __name__ == "__main__":
    main()
