#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import fnmatch
import importlib
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

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

# Repository-local directories that may already contain required data files.
LOCAL_DATA_DIRECTORIES: Sequence[Path] = (
    REPO_ROOT / "data",
    REPO_ROOT / "phewas",
    REPO_ROOT / "cds",
)

DEPENDENCY_ALIASES: Dict[str, Sequence[str]] = {
    "imputation_results.tsv": ("phewas_results.tsv", "data/phewas_results.tsv"),
    "inv_properties.tsv": ("inv_properties.tsv", "data/inv_properties.tsv"),
    "cds_identical_proportions.tsv": ("data/cds_identical_proportions.tsv",),
    "gene_inversion_direct_inverted.tsv": ("data/gene_inversion_direct_inverted.tsv",),
    "region_identical_proportions.tsv": ("data/region_identical_proportions.tsv",),
}

# Files mirrored from the run_analysis GitHub Actions workflow plus a few
# additional artefacts that the figure scripts expect.
HOME = Path.home()


RemoteResource = Union[str, Tuple[str, str]]


REMOTE_PATHS: Sequence[RemoteResource] = [
    "public_internet/paml_results.checkpoint.tsv",
    "public_internet/hudson_fst_results.tsv.gz",
    "data/imputation_results.tsv",
    "public_internet/inversion_fst_estimates.tsv",
    "public_internet/inv_info.csv",
    "public_internet/output.csv",
    "phecodeX.csv",
    # Additional large derived artefacts referenced by multiple figure scripts.
    "public_internet/per_site_diversity_output.falsta",
    "public_internet/per_site_fst_output.falsta",
    # MAPT PHYLIP files for CDS polymorphism heatmap
    "public_internet/group0_MAPT_ENSG00000186868.18_ENST00000262410.10_chr17_cds_start45962338_cds_end46024168_inv_start45585159_inv_end46292045.phy",
    "public_internet/group1_MAPT_ENSG00000186868.18_ENST00000262410.10_chr17_cds_start45962338_cds_end46024168_inv_start45585159_inv_end46292045.phy",
    # FRF data
    "data/per_inversion_frf_effects.tsv",
]


TASK_GROUPS: Sequence[str] = (
    "CDS",
    "Diversity",
    "FST",
    "Associations",
)


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
    outputs: Sequence[Union[Path, str]]
    dependencies: Sequence[str]
    optional_dependencies: Sequence[str] = field(default_factory=tuple)
    python_dependencies: Sequence[str] = field(default_factory=tuple)
    required: bool = True
    note: str = ""
    long_running: bool = False
    group: str = ""


FIGURE_TASKS: Sequence[FigureTask] = (
    FigureTask(
        name="CDS pairwise identity computation",
        script=Path("stats/cds_differences.py"),
        outputs=("cds_identical_proportions.tsv", "pairs_CDS__*.tsv"),
        dependencies=("inv_properties.tsv", "*.phy"),
        note="Computes pairwise identity for all CDS alignments. Requires .phy files.",
        long_running=True,
        group="CDS",
    ),
    FigureTask(
        name="CDS conservation GLM analysis",
        script=Path("stats/CDS_identical_model.py"),
        outputs=(
            "cds_emm_adjusted.tsv",
            "cds_pairwise_adjusted.tsv",
            "cds_emm_nocov.tsv",
            "cds_pairwise_nocov.tsv",
        ),
        dependencies=("cds_identical_proportions.tsv", "pairs_CDS__*.tsv"),
        note="Runs binomial GLM with cluster-robust standard errors for CDS conservation test.",
        group="CDS",
    ),
    FigureTask(
        name="Per-gene CDS conservation tests",
        script=Path("stats/per_gene_cds_differences_jackknife.py"),
        outputs=("gene_inversion_direct_inverted.tsv",),
        dependencies=("cds_identical_proportions.tsv", "pairs_CDS__*.tsv"),
        note="Performs jackknife-based per-gene conservation tests.",
        long_running=True,
        group="CDS",
    ),
    FigureTask(
        name="Inversion nucleotide diversity violins",
        script=Path("stats/recur_diversity.py"),
        outputs=(Path("inversion_pi_violins.png"),),
        dependencies=("output.csv", "inv_properties.tsv"),
        group="Diversity",
    ),
    FigureTask(
        name="Hudson FST violin plot",
        script=Path("stats/fst_violins.py"),
        outputs=(Path("hudson_fst.pdf"),),
        dependencies=("output.csv", "inv_properties.tsv"),
        optional_dependencies=("map.tsv",),
        group="FST",
    ),
    FigureTask(
        name="Spearman Decay Raincloud",
        script=Path("stats/replicate_spearman_raincloud.py"),
        outputs=(Path("special/spearman_decay_raincloud.pdf"),),
        dependencies=("data/spearman_decay_points.tsv",),
        note="Generates raincloud plot for Spearman decay correlations.",
        group="Diversity",
    ),
    FigureTask(
        name="FRF Volcano Plot",
        script=Path("stats/frf_volcano.py"),
        outputs=(Path("frf/frf_volcano.pdf"),),
        dependencies=("data/per_inversion_frf_effects.tsv",),
        note="Generates volcano plot for FRF effects.",
        group="Diversity",
    ),
    FigureTask(
        name="FRF Edge vs Middle Violins",
        script=Path("stats/frf_violin.py"),
        outputs=(Path("frf/frf_violin_from_url.pdf"),),
        dependencies=("data/per_inversion_frf_effects.tsv",),
        note="Generates violin plots for FRF edge vs middle regions.",
        group="Diversity",
    ),
    FigureTask(
        name="FRF breakpoint enrichment strip-box plot",
        script=Path("stats/old/frf_delta_strip_box.py"),
        outputs=(Path("frf/frf_delta_strip_box.pdf"),),
        dependencies=("data/per_inversion_frf_effects.tsv",),
        note="Precision-weighted strip-plus-box view of frf_delta_centered by inversion type.",
        group="Diversity",
    ),
    FigureTask(
        name="Inversion imputation performance",
        script=Path("stats/imputation_plot.py"),
        outputs=(Path("inversion_r_plot.pdf"),),
        dependencies=("imputation_results.tsv", "inv_properties.tsv"),
        group="Associations",
    ),
    FigureTask(
        name="Weir vs Hudson FST scatterplots",
        script=Path("stats/estimators_fst.py"),
        outputs=(
            Path("fst_wc_vs_hudson_colored_by_inversion_type.png"),
            Path("variance_wc_vs_dxy_hudson_log_scale_colored.png"),
        ),
        dependencies=("output.csv", "inv_properties.tsv"),
        group="FST",
    ),
    FigureTask(
        name="Inversion allele frequency vs nucleotide diversity",
        script=Path("stats/af_pi.py"),
        outputs=(Path("scatter_af_vs_pi_combined.png"),),
        dependencies=("output.csv", "inv_properties.tsv"),
        group="Diversity",
    ),
    FigureTask(
        name="Recurrent event diversity mixed models",
        script=Path("stats/num_events_diversity.py"),
        outputs=(
            Path("recurrent_events_analysis_separate_v2/pi_vs_recurrent_events_separate_lmm_plot.png"),
        ),
        dependencies=("output.csv", "inv_properties.tsv"),
        group="Diversity",
    ),
    FigureTask(
        name="PheWAS forest plot",
        script=Path("stats/forest.py"),
        outputs=(Path("phewas_forest.png"), Path("phewas_forest.pdf")),
        dependencies=("phewas_results.tsv",),
        required=False,
        note="Requires exported phewas_results.tsv from the BigQuery-backed pipeline.",
        group="Associations",
    ),
    FigureTask(
        name="PheWAS QQ plot",
        script=Path("stats/qq_plot.py"),
        outputs=(
            Path("phewas_plots/qq_plot_overall.png"),
            Path("phewas_plots/qq_plot_overall.pdf"),
        ),
        dependencies=("data/phewas_results.tsv",),
        required=False,
        note="Generates QQ plots for PheWAS results with genomic inflation factor.",
        group="Associations",
    ),
    FigureTask(
        name="Chr17 inversion vs tag SNP correlation",
        script=Path("stats/chr17_inversion_tag_correlation.py"),
        outputs=(
            Path("phewas_plots/chr17_inversion_tag_correlation.png"),
            Path("phewas_plots/chr17_inversion_tag_correlation.pdf"),
        ),
        dependencies=("data/phewas_results.tsv", "data/all_pop_phewas_tag.tsv"),
        required=False,
        note="Correlates effect sizes between chr17 inversion and tag SNP for significant phenotypes.",
        group="Associations",
    ),
    FigureTask(
        name="CDS identity and conservation panels",
        script=Path("stats/CDS_plots.py"),
        outputs=(
            Path("cds_proportion_identical_by_category_violin.pdf"),
            Path("cds_conservation_volcano.pdf"),
            Path("mapt_cds_polymorphism_heatmap.pdf"),
            Path("cds_conservation_table.tsv"),
        ),
        dependencies=(
            "cds_identical_proportions.tsv",
            "gene_inversion_direct_inverted.tsv",
            "inv_properties.tsv",
        ),
        long_running=True,
        group="CDS",
    ),
    FigureTask(
        name="FST violin and scatter summary",
        script=Path("stats/overall_fst_by_type.py"),
        outputs=(Path("comparison_violin_haplotype_overall_fst_wc.png"),),
        dependencies=("output.csv", "inv_properties.tsv"),
        optional_dependencies=("map.tsv",),
        required=False,
        note="Generates a suite of plots when Weir & Cockerham summaries are available in output.csv.",
        group="FST",
    ),
    FigureTask(
        name="Per-site diversity/FST trends by category",
        script=Path("stats/category_per_site.py"),
        outputs=(
            Path("length_norm_trend_fast/pi_vs_inversion_edge_proportion_grouped_mean.pdf"),
            Path("length_norm_trend_fast/pi_vs_inversion_edge_proportion_grouped_mean_overall_only.pdf"),
            Path("length_norm_trend_fast/pi_vs_inversion_edge_bp_cap100kb_grouped_mean.pdf"),
            Path("length_norm_trend_fast/pi_vs_inversion_edge_bp_cap100kb_grouped_mean_overall_only.pdf"),
            Path("length_norm_trend_fast/pi_vs_inversion_edge_proportion_grouped_median.pdf"),
            Path("length_norm_trend_fast/pi_vs_inversion_edge_proportion_grouped_median_overall_only.pdf"),
            Path("length_norm_trend_fast/pi_vs_inversion_edge_bp_cap100kb_grouped_median.pdf"),
            Path("length_norm_trend_fast/pi_vs_inversion_edge_bp_cap100kb_grouped_median_overall_only.pdf"),
            Path("length_norm_trend_fast/fst_vs_inversion_edge_proportion_grouped_pooled.pdf"),
            Path("length_norm_trend_fast/fst_vs_inversion_edge_proportion_grouped_median.pdf"),
            Path("length_norm_trend_fast/fst_vs_inversion_edge_bp_cap100kb_grouped_pooled.pdf"),
            Path("length_norm_trend_fast/fst_vs_inversion_edge_bp_cap40kb_grouped_pooled.pdf"),
            Path("length_norm_trend_fast/fst_vs_inversion_edge_bp_cap100kb_grouped_median.pdf"),
        ),
        dependencies=(
            "per_site_diversity_output.falsta",
            "per_site_fst_output.falsta",
            "inv_properties.tsv",
        ),
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Per-site diversity scatterplot",
        script=Path("stats/diversity_scatterplot.py"),
        outputs=(HOME / "distance_plots_10K_beautiful.png",),
        dependencies=("per_site_diversity_output.falsta",),
        python_dependencies=("tqdm",),
        required=False,
        note="Requires the tqdm Python package; install it to generate this figure.",
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Per-site diversity top-N sequences",
        script=Path("stats/top_n_pi.py"),
        outputs=(HOME / "top_filtered_pi_smoothed.png",),
        dependencies=("per_site_diversity_output.falsta",),
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Per-site diversity vs distance",
        script=Path("stats/distance_diversity.py"),
        outputs=(
            HOME / "distance_plot_theta_some number.png",
            HOME / "distance_plot_pi_some number.png",
        ),
        dependencies=("per_site_diversity_output.falsta",),
        python_dependencies=("numba", "tqdm"),
        required=False,
        note="Requires optional Python packages numba and tqdm to accelerate processing.",
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Normalized per-site diversity/FST trends",
        script=Path("stats/category_per_site_normed.py"),
        outputs=(
            Path("length_norm_trend_fast_normed/pi_vs_inversion_edge_proportion_grouped_mean.pdf"),
            Path("length_norm_trend_fast_normed/pi_vs_inversion_edge_bp_cap100kb_grouped_mean.pdf"),
            Path("length_norm_trend_fast_normed/pi_vs_inversion_edge_proportion_grouped_median.pdf"),
            Path("length_norm_trend_fast_normed/pi_vs_inversion_edge_bp_cap100kb_grouped_median.pdf"),
            Path("length_norm_trend_fast_normed/fst_vs_inversion_edge_proportion_grouped_mean.pdf"),
            Path("length_norm_trend_fast_normed/fst_vs_inversion_edge_proportion_grouped_median.pdf"),
            Path("length_norm_trend_fast_normed/fst_vs_inversion_edge_bp_cap100kb_grouped_mean.pdf"),
            Path("length_norm_trend_fast_normed/fst_vs_inversion_edge_bp_cap100kb_grouped_median.pdf"),
        ),
        dependencies=(
            "per_site_diversity_output.falsta",
            "per_site_fst_output.falsta",
            "inv_properties.tsv",
        ),
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Long-region per-site π overview",
        script=Path("stats/regions_plot.py"),
        outputs=(Path("filtered_pi_beginning_middle_end.png"),),
        dependencies=("per_site_diversity_output.falsta",),
        required=False,
        note="Requires per_site_diversity_output.falsta, which is not included in the public archive.",
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Per-inversion distance trends",
        script=Path("stats/each_per_site.py"),
        outputs=("per_inversion_trends/**/*.png",),
        dependencies=(
            "per_site_diversity_output.falsta",
            "per_site_fst_output.falsta",
        ),
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Middle vs flank π quadrant violins",
        script=Path("stats/middle_vs_flank_pi.py"),
        outputs=("pi_analysis_results_exact_mf_quadrants/total_*/pi_mf_quadrant_violins_total_*.pdf",),
        dependencies=("per_site_diversity_output.falsta", "inv_properties.tsv"),
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Middle vs flank π recurrence violins",
        script=Path("stats/middle_vs_flank_pi_recurrence.py"),
        outputs=(
            "pi_analysis_results_exact_mf_quadrants/total_*/pi_mf_recurrence_violins_total_*.pdf",
            "pi_analysis_results_exact_mf_quadrants/total_*/pi_mf_overall_violins_total_*.pdf",
        ),
        dependencies=("per_site_diversity_output.falsta", "inv_properties.tsv"),
        long_running=True,
        group="Diversity",
    ),
    FigureTask(
        name="Middle vs flank FST quadrant violins",
        script=Path("stats/middle_vs_flank_fst.py"),
        outputs=("fst_analysis_results_exact_mf_quadrants/total_*/fst_mf_quadrant_violins_total_*.pdf",),
        dependencies=("per_site_fst_output.falsta", "inv_properties.tsv"),
        long_running=True,
        group="FST",
    ),
    FigureTask(
        name="Direct vs inverted recurrence violins",
        script=Path("stats/inv_dir_recur_violins.py"),
        outputs=(Path("pi_comparison_violins.pdf"),),
        dependencies=("output.csv", "inv_properties.tsv"),
        group="FST",
    ),
    FigureTask(
        name="Inversion event rate vs diversity",
        script=Path("stats/events_rate_diversity.py"),
        outputs=(
            Path("logfc_vs_formation_rate.pdf"),
            Path("logfc_vs_nrecur.pdf"),
            Path("fst_vs_formation_rate.pdf"),
            Path("fst_vs_nrecur.pdf"),
        ),
        dependencies=("output.csv", "inv_properties.tsv"),
        group="Associations",
    ),
    FigureTask(
        name="PheWAS volcano plot",
        script=Path("stats/volcano.py"),
        outputs=(Path("phewas_volcano.pdf"),),
        dependencies=("phewas_results.tsv", "inv_properties.tsv"),
        required=False,
        note="Requires phewas_results.tsv, which is produced by the BigQuery-backed PheWAS pipeline.",
        group="Associations",
    ),
    FigureTask(
        name="PheWAS category summary heatmap",
        script=Path("stats/category_figure.py"),
        outputs=(Path("phewas_category_heatmap.pdf"),),
        dependencies=tuple(),
        required=False,
        note=(
            "Downloads supplementary category summary files from GitHub. "
            "Ensure internet access is available when running this task."
        ),
        group="Associations",
    ),
    FigureTask(
        name="PheWAS ranged volcano plot",
        script=Path("stats/ranged_volcano.py"),
        outputs=(Path("phewas_volcano_ranged.pdf"),),
        dependencies=("phewas_results.tsv",),
        required=False,
        note="Requires phewas_results.tsv, which is produced by the BigQuery-backed PheWAS pipeline.",
        group="Associations",
    ),
    FigureTask(
        name="PheWAS Manhattan panels",
        script=Path("stats/manhattan_phe.py"),
        outputs=("phewas_plots/*.pdf",),
        dependencies=("phewas_results.tsv",),
        optional_dependencies=("inv_properties.tsv",),
        required=False,
        note="Requires phewas_results.tsv exported from the production pipeline.",
        group="Associations",
    ),
    FigureTask(
        name="PheWAS odds ratio matrix",
        script=Path("stats/OR_matrix.py"),
        outputs=(Path("phewas_heatmap.pdf"), Path("phewas_heatmap.svg")),
        dependencies=("phewas_results.tsv",),
        required=False,
        note="Requires phewas_results.tsv exported from the production pipeline.",
        group="Associations",
    ),
    FigureTask(
        name="PGS control volcano plot",
        script=Path("stats/PGS_control_plot.py"),
        outputs=(Path("PGS_control_volcano.pdf"), Path("PGS_control_volcano.png")),
        dependencies=("PGS_controls.tsv",),
        required=False,
        note="Requires PGS_controls.tsv showing effect of polygenic score adjustment.",
        group="Associations",
    ),
    FigureTask(
        name="Family History vs Main PheWAS Forest",
        script=Path("stats/family_forest.py"),
        outputs=(Path("family_vs_main_forest.pdf"), Path("family_vs_main_forest.png")),
        dependencies=("family_phewas.tsv", "phewas_results.tsv"),
        required=True,
        group="Associations",
    ),
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def log_boxed(message: str) -> None:
    """Pretty-print a message inside a simple text banner."""

    border = "=" * len(message)
    print(f"\n{border}\n{message}\n{border}")


def open_file(file_path: Path) -> None:
    """Open a file using the platform default application."""

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(file_path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", str(file_path)], check=True)
        else:
            subprocess.run(["xdg-open", str(file_path)], check=True)
    except FileNotFoundError:
        print(
            f"ERROR: Could not find application to open '{file_path.name}'. "
            f"Is a default viewer installed for '{file_path.suffix}' files?"
        )
    except Exception as exc:  # pragma: no cover - defensive programming
        print(f"ERROR: Failed to open '{file_path.name}': {exc}")


def build_download_plan(paths: Sequence[RemoteResource]) -> Dict[str, str]:
    """Create a mapping from relative path (under PUBLIC_PREFIX) to a local source."""

    plan: Dict[str, str] = {}
    for entry in paths:
        rel_path = entry[0] if isinstance(entry, tuple) else entry
        key = str(Path(rel_path).as_posix()).lstrip("/")
        plan[key] = str(DOWNLOAD_ROOT / key)
    return plan


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


def _iter_dependency_aliases(name: str) -> Sequence[str]:
    """Yield ``name`` and any alternative identifiers without duplicates."""

    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in (name, *DEPENDENCY_ALIASES.get(name, ())):
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return tuple(ordered)


def _iter_compressed_aliases(name: str) -> Sequence[str]:
    """Yield dependency aliases plus possible ``.gz`` variants."""

    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in _iter_dependency_aliases(name):
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
        if not candidate.endswith(".gz"):
            gz_candidate = f"{candidate}.gz"
            if gz_candidate not in seen:
                ordered.append(gz_candidate)
                seen.add(gz_candidate)
    return tuple(ordered)


def find_local_data_file(name: str) -> Optional[Path]:
    """Return a repository-local data file matching ``name`` if available."""

    for candidate_name in _iter_compressed_aliases(name):
        candidate_path = Path(candidate_name)

        # If the alias includes a directory component, check that exact path
        # relative to the repository root before scanning the standard data
        # directories.
        if candidate_path.parent != Path("."):
            resolved = (REPO_ROOT / candidate_path).resolve()
            try:
                if resolved.exists() and is_valid_data_file(resolved):
                    return resolved
            except FileNotFoundError:
                pass

        for directory in LOCAL_DATA_DIRECTORIES:
            if not directory.exists():
                continue
            # Prefer a direct lookup before performing an expensive recursive search.
            direct_candidate = directory / candidate_path.name
            if direct_candidate.exists() and is_valid_data_file(direct_candidate):
                return direct_candidate
            for candidate in directory.rglob(candidate_path.name):
                if candidate.is_file() and is_valid_data_file(candidate):
                    return candidate

        candidate = REPO_ROOT / candidate_path.name
        if candidate.exists() and is_valid_data_file(candidate):
            return candidate

    return None


def has_expected_output(target: Union[Path, str]) -> bool:
    """Check whether the requested output artefact exists."""

    text = str(target)
    if any(ch in text for ch in "*?[]"):
        matches = [p for p in REPO_ROOT.glob(text) if p.is_file()]
        return any(is_valid_data_file(match) for match in matches)

    path = REPO_ROOT / Path(text)
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file() and is_valid_data_file(candidate):
                return True
        return False

    return is_valid_data_file(path)


def ensure_local_copy(name: str, index: Dict[str, List[Path]]) -> Optional[Path]:
    """Ensure that ``name`` exists in the repository root.

    Returns the resolved Path to the local copy, creating a symlink in the
    repository root if necessary.  ``index`` maps file basenames to candidate
    paths.
    """

    if glob.has_magic(name):
        pattern = Path(name)
        first_linked: Optional[Path] = None

        for paths in index.values():
            for candidate in paths:
                try:
                    resolved = candidate.resolve()
                except FileNotFoundError:
                    continue
                if not fnmatch.fnmatch(candidate.name, pattern.name):
                    continue

                target = REPO_ROOT / candidate.name
                if target.exists() and is_valid_data_file(target):
                    first_linked = first_linked or target
                    continue
                if target.is_symlink() and not target.exists():
                    target.unlink()

                try:
                    target.symlink_to(resolved)
                except OSError:
                    shutil.copy2(resolved, target)

                if is_valid_data_file(target):
                    first_linked = first_linked or target

        return first_linked

    target = REPO_ROOT / name
    if target.exists() and is_valid_data_file(target):
        return target
    if target.is_symlink() and not target.exists():
        target.unlink()

    def _link_candidate(candidate: Path) -> Optional[Path]:
        if not candidate.exists() or not is_valid_data_file(candidate):
            return None
        try:
            if candidate.resolve() == target.resolve():
                return target
        except FileNotFoundError:
            return None
        try:
            target.symlink_to(candidate.resolve())
            return target
        except OSError:
            shutil.copy2(candidate, target)
            return target

    # Try to satisfy the dependency using direct copies/symlinks first.
    for candidate_name in _iter_dependency_aliases(name):
        basename = Path(candidate_name).name
        for candidate in index.get(basename, []):
            linked = _link_candidate(candidate)
            if linked is not None:
                return linked

    # Attempt to resolve any repo-relative alias paths explicitly.
    for alias in _iter_dependency_aliases(name):
        alias_path = Path(alias)
        if alias_path.parent == Path("."):
            continue
        resolved_alias = REPO_ROOT / alias_path
        if not resolved_alias.exists() or not is_valid_data_file(resolved_alias):
            continue
        linked = _link_candidate(resolved_alias)
        if linked is not None:
            return linked

    # Fall back to checking for gzip-compressed artefacts.
    for candidate_name in _iter_compressed_aliases(name):
        if not candidate_name.endswith(".gz"):
            continue
        basename = Path(candidate_name).name
        for candidate in index.get(basename, []):
            decompressed = _decompress_gzip_to_target(candidate, target)
            if decompressed is not None:
                return decompressed
        alias_path = Path(candidate_name)
        if alias_path.parent != Path("."):
            resolved_alias = REPO_ROOT / alias_path
            if resolved_alias.exists():
                decompressed = _decompress_gzip_to_target(resolved_alias, target)
                if decompressed is not None:
                    return decompressed
    return None


def _decompress_gzip_to_target(source: Path, target: Path) -> Optional[Path]:
    """Decompress ``source`` into ``target`` if possible."""

    import gzip

    if not source.exists() or not is_valid_data_file(source):
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = target.with_name(target.name + ".tmp")
    try:
        with gzip.open(source, "rb") as src, temp_name.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        temp_name.replace(target)
    except OSError:
        if temp_name.exists():
            temp_name.unlink()
        return None

    if not is_valid_data_file(target):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        return None

    return target


def build_file_index(plan: Dict[str, str]) -> Dict[str, List[Path]]:
    """Index downloaded and repository-local data files by basename."""

    index: Dict[str, List[Path]] = {}

    # Index downloaded artefacts.
    if DOWNLOAD_ROOT.exists():
        for path in DOWNLOAD_ROOT.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, []).append(path)

    # Index known local data directories that ship with the repository.
    for local_dir in LOCAL_DATA_DIRECTORIES:
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

    # Register alias mappings so downstream dependency checks can locate files
    # that ship under alternative names.
    for dest, aliases in DEPENDENCY_ALIASES.items():
        dest_list = index.setdefault(dest, [])
        seen: set[Path] = set()
        for existing in dest_list:
            try:
                seen.add(existing.resolve())
            except FileNotFoundError:
                continue
        for alias in aliases:
            alias_path = Path(alias)
            for candidate in index.get(alias_path.name, []):
                try:
                    resolved = candidate.resolve()
                except FileNotFoundError:
                    continue
                if resolved in seen:
                    continue
                dest_list.append(candidate)
                seen.add(resolved)
            resolved_alias = REPO_ROOT / alias_path
            if resolved_alias.exists() and is_valid_data_file(resolved_alias):
                try:
                    resolved = resolved_alias.resolve()
                except FileNotFoundError:
                    continue
                if resolved not in seen:
                    dest_list.append(resolved_alias)
                    seen.add(resolved)

    return index


def run_task(task: FigureTask, env: Dict[str, str]) -> tuple[str, str]:
    """Execute a figure task and return (status, message)."""

    missing: List[str] = []
    invalid: List[str] = []
    for dep in task.dependencies:
        if glob.has_magic(dep):
            matches = [p for p in REPO_ROOT.glob(dep)]
            if not matches:
                pattern = Path(dep).name
                for directory in LOCAL_DATA_DIRECTORIES:
                    if not directory.exists():
                        continue
                    matches.extend(directory.rglob(pattern))

            valid_matches = [p for p in matches if p.is_file() and is_valid_data_file(p)]
            if not valid_matches:
                missing.append(dep)
            elif len(valid_matches) < len(matches):
                invalid.append(dep)
            continue

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

    missing_python: List[str] = []
    for module in task.python_dependencies:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing_python.append(module)
    if missing_python:
        return "missing_inputs", "missing python packages: " + ", ".join(sorted(missing_python))

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
        if not has_expected_output(path)
    ]
    if missing_outputs:
        return "missing_outputs", "expected outputs not found: " + ", ".join(missing_outputs)

    return "success", "ok"


def _resolve_output_paths(target: Union[Path, str]) -> List[Path]:
    """Return existing files matching the declared output target."""

    path = target if isinstance(target, Path) else Path(target)
    pattern = path if path.is_absolute() else REPO_ROOT / path
    text = str(pattern)

    matches: List[Path] = []
    if any(ch in text for ch in "*?[]"):
        for match in glob.glob(text, recursive=True):
            candidate = Path(match)
            if candidate.is_file() and is_valid_data_file(candidate):
                matches.append(candidate)
        return matches

    if pattern.is_dir():
        for candidate in pattern.rglob("*"):
            if candidate.is_file() and is_valid_data_file(candidate):
                matches.append(candidate)
        return matches

    if pattern.is_file() and is_valid_data_file(pattern):
        matches.append(pattern)
    return matches


def collect_outputs(tasks: Sequence[FigureTask], destination: Path, summary: Sequence[tuple[FigureTask, str, str]]) -> List[Path]:
    """Copy resolved outputs for successful tasks into ``destination``.

    Returns the list of copied file paths (within ``destination``).
    """

    destination.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []

    success_lookup = {task.name: status for task, status, _ in summary}
    for task in tasks:
        if success_lookup.get(task.name) != "success":
            continue
        for output in task.outputs:
            for resolved in _resolve_output_paths(output):
                try:
                    relative = resolved.relative_to(REPO_ROOT)
                except ValueError:
                    relative = Path(resolved.name)
                target_path = destination / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(resolved, target_path)
                    copied.append(target_path)
                except Exception as exc:  # pragma: no cover - defensive programming
                    print(f"WARNING: Failed to copy {resolved} -> {target_path}: {exc}")

    if copied:
        print(f"\nCollected {len(copied)} artefacts into {destination}")
    else:
        print("\nNo outputs were copied; check earlier logs for missing artefacts.")

    return copied


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
        "--group",
        choices=TASK_GROUPS,
        help="Restrict execution to a single task group for parallel workflows.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    collection_dir = REPO_ROOT / "final_plots"

    plan = build_download_plan(REMOTE_PATHS)
    download_results: List[DownloadResult] = []

    if not args.skip_downloads:
        log_boxed("Staging analysis artefacts from local cache")
        for rel_path, source in plan.items():
            target = DOWNLOAD_ROOT / Path(rel_path)
            if target.exists() and is_valid_data_file(target):
                download_results.append(DownloadResult(url=source, destination=target, ok=True, message="found"))
                print(f"[CACHED] {rel_path} already present under analysis_downloads")
                continue

            local_copy = find_local_data_file(Path(rel_path).name)
            if local_copy is not None:
                download_results.append(
                    DownloadResult(url=source, destination=local_copy, ok=True, message="using local copy")
                )
                try:
                    local_display = local_copy.relative_to(REPO_ROOT)
                except ValueError:
                    local_display = local_copy
                print(f"[LOCAL] {rel_path} satisfied by {local_display}")
                continue

            download_results.append(
                DownloadResult(
                    url=source,
                    destination=target,
                    ok=False,
                    message="not found; place under data/ or analysis_downloads/",
                )
            )
            print(
                f"[MISSING] {rel_path} not located. Populate data/ or analysis_downloads/ from the manual workflow artefacts."
            )
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
    if args.group:
        selected_tasks = tuple(task for task in FIGURE_TASKS if task.group == args.group)
        print(f"Selected task group: {args.group} ({len(selected_tasks)} tasks)")
        if not selected_tasks:
            print("No tasks matched the requested group; exiting.")
            return 0

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

    copied_paths = collect_outputs(selected_tasks, collection_dir, summary)
    if copied_paths:
        if os.environ.get("CI"):
            print("\nCI environment detected; skipping file open prompts.")
        else:
            print("\nOpening copied outputs ...")
            for file_path in copied_paths:
                open_file(file_path)

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
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("fork")
    sys.exit(main())
