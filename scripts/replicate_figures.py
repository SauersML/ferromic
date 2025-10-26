#!/usr/bin/env python3
"""Replicate all published figures for the Ferromic analysis manuscript.

This script downloads the public analysis artefacts referenced by the
``run_analysis`` GitHub Actions workflow and then executes each of the
plotting/statistics scripts that produce manuscript figures.  It reports
success/failure for both the downloads and the downstream figure
reproduction tasks, copying the generated images/PDFs into the repository
root.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Infer the repository root so the script can be relocated without breaking
# path resolution.  We look for common project markers as we ascend from the
# script's directory.
def _detect_repo_root(start: Path) -> Path:
    markers = ("pyproject.toml", "Cargo.toml", ".git")
    for candidate in [start, *start.parents]:
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return start


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _detect_repo_root(SCRIPT_DIR)
DOWNLOAD_ROOT = REPO_ROOT / "analysis_downloads"
BASE_URL = "https://sharedspace.s3.msi.umn.edu/"

# Files mirrored from the run_analysis GitHub Actions workflow plus a few
# additional artefacts that the figure scripts expect.
REMOTE_PATHS: Sequence[str] = [
    "public_internet/variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_"
    "PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv",
    "public_internet/pairwise_glm_contrasts_fdr.tsv",
    "public_internet/pairwise_results_fdr.tsv",
    "public_internet/paml_results.checkpoint.tsv",
    "public_internet/perm_identical_pairs.tsv",
    "public_internet/perm_pairwise_identity.tsv",
    "public_internet/phy_metadata.tsv",
    "public_internet/region_identical_proportions.tsv",
    "public_internet/category_means_at_mean_covariates.tsv",
    "public_internet/category_summary.tsv",
    "public_internet/cds_emm.tsv",
    "public_internet/cds_emm_adjusted.tsv",
    "public_internet/cds_emm_nocov.tsv",
    "public_internet/cds_identical_proportions.tsv",
    "public_internet/cds_pairwise.tsv",
    "public_internet/cds_pairwise_adjusted.tsv",
    "public_internet/cds_pairwise_nocov.tsv",
    "public_internet/fixed_diff_summary.tsv",
    "public_internet/gene_direct_inverted.tsv",
    "public_internet/gene_inversion_direct_inverted.tsv",
    "public_internet/gene_inversion_fixed_differences.tsv",
    "public_internet/gene_inversion_permutation.tsv",
    "public_internet/glm_category_coefs.tsv",
    "public_internet/hudson_fst_results.tsv",
    "public_internet/inv_info.tsv",
    "public_internet/inversion_fst_estimates.tsv",
    "public_internet/inversion_level_counts.tsv",
    "public_internet/inversion_level_medians.tsv",
    "public_internet/kruskal_result.tsv",
    "public_internet/fst_tests_summary.csv",
    "public_internet/inversion_statistical_results.csv",
    "public_internet/inv_info.csv",
    "public_internet/output.csv",
    # Additional large derived artefacts referenced by multiple figure scripts.
    "public_internet/all_pairwise_results.csv",
    "public_internet/per_site_diversity_output.falsta",
    "public_internet/per_site_fst_output.falsta",
]


@dataclass
class DownloadResult:
    """Represents the outcome of a single remote download."""

    url: str
    destination: Path
    ok: bool
    message: str = ""


@dataclass
class FigureTask:
    """Metadata describing a figure replication script."""

    name: str
    script: Path
    outputs: Sequence[Path]
    dependencies: Sequence[str]
    optional_dependencies: Sequence[str] = field(default_factory=tuple)
    required: bool = True
    note: str = ""


FIGURE_TASKS: Sequence[FigureTask] = (
    FigureTask(
        name="Inversion nucleotide diversity violins",
        script=Path("stats/recur_diversity.py"),
        outputs=(Path("inversion_pi_violins.png"),),
        dependencies=("output.csv", "inv_info.tsv"),
    ),
    FigureTask(
        name="Hudson FST violin plot",
        script=Path("stats/fst_violins.py"),
        outputs=(Path("hudson_fst.pdf"),),
        dependencies=("output.csv", "inv_info.tsv"),
        optional_dependencies=("map.tsv",),
    ),
    FigureTask(
        name="Weir vs Hudson FST scatterplots",
        script=Path("stats/estimators_fst.py"),
        outputs=(
            Path("fst_wc_vs_hudson_colored_by_inversion_type.png"),
            Path("variance_wc_vs_dxy_hudson_log_scale_colored.png"),
        ),
        dependencies=("output.csv", "inv_info.tsv"),
    ),
    FigureTask(
        name="Inversion allele frequency vs nucleotide diversity",
        script=Path("stats/af_pi.py"),
        outputs=(Path("scatter_af_vs_pi_combined.png"),),
        dependencies=("output.csv", "inv_info.tsv"),
    ),
    FigureTask(
        name="Recurrent event diversity mixed models",
        script=Path("stats/num_events_diversity.py"),
        outputs=(
            Path("recurrent_events_analysis_separate_v2/pi_vs_recurrent_events_separate_lmm_plot.png"),
        ),
        dependencies=("output.csv", "inv_info.tsv"),
    ),
    FigureTask(
        name="PheWAS forest plot",
        script=Path("stats/forest.py"),
        outputs=(Path("phewas_forest.png"), Path("phewas_forest.pdf")),
        dependencies=("phewas_results.tsv",),
        required=False,
        note="Requires exported phewas_results.tsv from the BigQuery-backed pipeline.",
    ),
    FigureTask(
        name="dN/dS violin by inversion event type",
        script=Path("stats/cross_violins.py"),
        outputs=(Path("inversion_omega_analysis_plot_median_only.png"),),
        dependencies=("all_pairwise_results.csv", "inv_info.tsv"),
        required=False,
        note="Needs all_pairwise_results.csv generated by the CODEML post-processing scripts.",
    ),
    FigureTask(
        name="Median omega distribution",
        script=Path("stats/dnds_kde.py"),
        outputs=(Path("median_omega_distribution_standalone.png"),),
        dependencies=("all_pairwise_results.csv", "inv_info.tsv"),
        required=False,
        note="Needs all_pairwise_results.csv generated by the CODEML post-processing scripts.",
    ),
    FigureTask(
        name="Omega percentile distribution",
        script=Path("stats/overall_groups_dnds.py"),
        outputs=(Path("omega_percentile_distribution.png"),),
        dependencies=("all_pairwise_results.csv", "inv_info.tsv"),
        required=False,
        note="Needs all_pairwise_results.csv generated by the CODEML post-processing scripts.",
    ),
    FigureTask(
        name="Hudson FST flanking region bar plot",
        script=Path("stats/dist_fst_by_type.py"),
        outputs=(Path("fst_flanking_regions_bar_plot.png"),),
        dependencies=("per_site_fst_output.falsta", "inv_info.tsv"),
        required=False,
        note="Requires per_site_fst_output.falsta emitted by the ferromic CLI run.",
    ),
    FigureTask(
        name="FST violin and scatter summary",
        script=Path("stats/overall_fst_by_type.py"),
        outputs=(Path("comparison_violin_haplotype_overall_fst_wc.png"),),
        dependencies=("output.csv", "inv_info.tsv"),
        optional_dependencies=("map.tsv",),
        required=False,
        note="Generates a suite of plots when Weir & Cockerham summaries are available in output.csv.",
    ),
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def log_boxed(message: str) -> None:
    """Pretty-print a message inside a simple text banner."""

    border = "=" * len(message)
    print(f"\n{border}\n{message}\n{border}")


def build_download_plan(paths: Sequence[str]) -> Dict[str, str]:
    """Create a mapping from relative path (under PUBLIC_PREFIX) to URL."""

    plan: Dict[str, str] = {}
    for rel in paths:
        url = urljoin(BASE_URL, rel)
        parsed = urlparse(url)
        # Use the path (without leading slash) as our dictionary key so we can
        # later reconstruct nested directories under DOWNLOAD_ROOT.
        key = parsed.path.lstrip("/")
        plan[key] = url
    return plan


def download_file(url: str, dest: Path) -> DownloadResult:
    """Download ``url`` to ``dest`` using urllib with streaming."""

    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request) as response, dest.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        return DownloadResult(url=url, destination=dest, ok=True, message="downloaded")
    except urllib.error.HTTPError as exc:  # noqa: PERF203 - explicit status handling
        return DownloadResult(url=url, destination=dest, ok=False, message=f"HTTP {exc.code}: {exc.reason}")
    except urllib.error.URLError as exc:
        return DownloadResult(url=url, destination=dest, ok=False, message=f"URL error: {exc.reason}")
    except Exception as exc:  # pragma: no cover - defensive programming
        return DownloadResult(url=url, destination=dest, ok=False, message=str(exc))


def is_valid_data_file(path: Path) -> bool:
    """Return ``True`` if ``path`` appears to contain meaningful data."""

    if not path.exists():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size == 0:
        return False
    try:
        with path.open("rb") as fh:
            sample = fh.read(512)
    except OSError:
        return False
    if not sample.strip():
        return False
    lowered = sample.lower()
    if lowered.startswith(b"<?xml") and b"<error" in lowered:
        return False
    if b"accessdenied" in lowered or b"forbidden" in lowered:
        return False
    return True


def ensure_local_copy(name: str, index: Dict[str, List[Path]]) -> Optional[Path]:
    """Ensure that ``name`` exists in the repository root.

    Returns the resolved Path to the local copy, creating a symlink in the
    repository root if necessary.  ``index`` maps file basenames to candidate
    paths.
    """

    target = REPO_ROOT / name
    if target.exists() and is_valid_data_file(target):
        return target
    if target.is_symlink() and not target.exists():
        target.unlink()

    for candidate in index.get(name, []):
        if not candidate.exists() or not is_valid_data_file(candidate):
            continue
        try:
            if candidate.resolve() == target.resolve():
                continue
        except FileNotFoundError:
            continue
        try:
            target.symlink_to(candidate.resolve())
            return target
        except OSError:
            # Fall back to copying if symlinks are unsupported.
            shutil.copy2(candidate, target)
            return target
    return None


def build_file_index(plan: Dict[str, str]) -> Dict[str, List[Path]]:
    """Index downloaded and repository-local data files by basename."""

    index: Dict[str, List[Path]] = {}

    # Index downloaded artefacts.
    if DOWNLOAD_ROOT.exists():
        for path in DOWNLOAD_ROOT.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, []).append(path)

    # Index known local data directories that ship with the repository.
    for local_dir in (REPO_ROOT / "data", REPO_ROOT / "phewas", REPO_ROOT / "cds"):
        if local_dir.exists():
            for path in local_dir.rglob("*"):
                if path.is_file():
                    index.setdefault(path.name, []).append(path)

    # Index files already in the repository root that may have been provided by
    # earlier runs or manual placement.
    for path in REPO_ROOT.glob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)

    # If the download plan references nested paths, add the resolved target.
    for rel_path in plan.keys():
        path = DOWNLOAD_ROOT / Path(rel_path)
        if path.is_file():
            index.setdefault(path.name, []).append(path)

    return index


def run_task(task: FigureTask, env: Dict[str, str]) -> tuple[str, str]:
    """Execute a figure task and return (status, message)."""

    missing: List[str] = []
    invalid: List[str] = []
    for dep in task.dependencies:
        dep_path = REPO_ROOT / dep
        if not dep_path.exists():
            missing.append(dep)
        elif not is_valid_data_file(dep_path):
            invalid.append(dep)
    if missing or invalid:
        parts: List[str] = []
        if missing:
            parts.append("missing required inputs: " + ", ".join(missing))
        if invalid:
            parts.append("invalid inputs: " + ", ".join(invalid))
        return "missing_inputs", "; ".join(parts)

    optional_missing = [
        dep
        for dep in task.optional_dependencies
        if not (REPO_ROOT / dep).exists() or not is_valid_data_file(REPO_ROOT / dep)
    ]
    if optional_missing:
        print(f"[INFO] Optional dependencies unavailable for {task.name}: {', '.join(optional_missing)}")

    script_path = REPO_ROOT / task.script
    if not script_path.exists():
        return "failed", "script not found"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        if result.stdout:
            print(textwrap.indent(result.stdout.rstrip(), prefix="    "))
        if result.stderr:
            print(textwrap.indent(result.stderr.rstrip(), prefix="    "))
    except subprocess.CalledProcessError as exc:
        combined = "\n".join(filter(None, [exc.stdout, exc.stderr]))
        message = "script failed"
        if combined:
            message = f"script failed:\n{textwrap.indent(combined, '    ')}"
        return "failed", message

    missing_outputs = [
        str(path)
        for path in task.outputs
        if not is_valid_data_file(REPO_ROOT / path)
    ]
    if missing_outputs:
        return "missing_outputs", "expected outputs not found: " + ", ".join(missing_outputs)

    return "success", "ok"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download analysis artefacts and replicate Ferromic figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip downloading remote artefacts (assume they are already present).",
    )
    parser.add_argument(
        "--only",
        metavar="NAME",
        help="Run only the task whose name contains the provided substring (case insensitive).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    plan = build_download_plan(REMOTE_PATHS)
    download_results: List[DownloadResult] = []

    if not args.skip_downloads:
        log_boxed("Downloading analysis artefacts")
        for rel_path, url in plan.items():
            target = DOWNLOAD_ROOT / Path(rel_path)
            if target.exists():
                download_results.append(DownloadResult(url=url, destination=target, ok=True, message="cached"))
                continue
            result = download_file(url, target)
            download_results.append(result)
            status = "OK" if result.ok else "FAIL"
            print(f"[{status}] {url} -> {target.relative_to(REPO_ROOT)} ({result.message})")
    else:
        print("Skipping downloads as requested; assuming artefacts are already present.")

    # Build an index of available files and create symlinks for dependencies.
    index = build_file_index(plan)
    for task in FIGURE_TASKS:
        for dep in task.dependencies + tuple(task.optional_dependencies):
            local = ensure_local_copy(dep, index)
            if local is None:
                index.setdefault(dep, [])  # Ensure missing is tracked for later messaging.
            else:
                index.setdefault(dep, []).append(local)

    log_boxed("Running figure replication tasks")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    selected_tasks = FIGURE_TASKS
    if args.only:
        needle = args.only.lower()
        selected_tasks = tuple(task for task in FIGURE_TASKS if needle in task.name.lower())
        if not selected_tasks:
            print(f"No tasks match substring '{args.only}'.")
            return 1

    summary: List[tuple[FigureTask, str, str]] = []
    for task in selected_tasks:
        print(f"\n--- {task.name} ---")
        for dep in task.dependencies:
            dep_path = REPO_ROOT / dep
            if not dep_path.exists():
                dep_status = "missing"
            elif not is_valid_data_file(dep_path):
                dep_status = "invalid"
            else:
                dep_status = "found"
            print(f"  dependency: {dep} [{dep_status}]")
        status, message = run_task(task, env)
        summary.append((task, status, message))
        if status == "success":
            label = "SUCCESS"
        elif status in {"missing_inputs", "missing_outputs"} and not task.required:
            label = "SKIPPED"
        else:
            label = "FAILED"
        print(f"  => {label}: {message}")

    log_boxed("Summary")
    for task, status, message in summary:
        if status == "success":
            state = "✅"
        elif status in {"missing_inputs", "missing_outputs"} and not task.required:
            state = "⚠️"
        else:
            state = "❌"
        print(f"{state} {task.name}: {message}")
        if state == "⚠️" and task.note:
            print(f"    {task.note}")

    failed_required = [
        (task, status, message)
        for task, status, message in summary
        if task.required and status != "success"
    ]
    failed_optional = [
        (task, status, message)
        for task, status, message in summary
        if not task.required and status == "failed"
    ]
    skipped_optional = [
        (task, status, message)
        for task, status, message in summary
        if not task.required and status in {"missing_inputs", "missing_outputs"}
    ]

    failed_downloads = [res for res in download_results if not res.ok]

    if failed_downloads:
        print("\nThe following downloads failed:")
        for res in failed_downloads:
            print(f"  - {res.url}: {res.message}")

    if failed_required or failed_optional:
        print("\nSome figure scripts failed. Review the messages above, ensure all dependencies are available, "
              "and re-run this script once the issues are resolved.")
        return 1

    if skipped_optional:
        print("\nAll required figure tasks completed. Optional plots were skipped; see notes above for staging the additional inputs.")
        return 0

    print("\nAll requested figures were generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
